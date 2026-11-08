// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The general convolution's store on CUDA (GpuOps.Conv.cs): y = act(sums + bias[f]) a thread per stored word (a warp, or a
// CTA, per block of a block format), the sums lane values, the bias read where it is stored, the activation the
// element-wise formula's (Ptx.Map.cs) in the lane, each value rounded to the type once.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Params: sums (lane storage), bias (f), y, n, cols (an output channel's positions), oc, the activation's
    /// parameter (a lane value).</summary>
    public static string ConvStore(Gpu gpu, Fmt f, Lane lane, string? act, bool bias, out string entry)
    {
        entry = $"conv_store_{f.Name}_{lane.Name}_{act ?? "none"}{(bias ? "_b" : "")}";
        var b = new PtxBuilder();
        string ps = Param64(b, "p_s"), pb = Param64(b, "p_b"), py = Param64(b, "p_y");
        string n = Param32(b, "p_n"), cols = Param32(b, "p_cols"), oc = Param32(b, "p_oc");
        string pa = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {pa}, [p_a];");
        string Value(string at)
        {
            string v = LoadLane(b, lane, ps, at);
            if (bias) v = LaneOp(b, lane, "add", v, Codec.UnpackAt(b, f, pb, Within(b, at, cols, oc), lane));
            return act is null ? v : Formula(b, lane, act, v, v, v, pa, pa);
        }

        if (f.IsBlock)
        {
            if (BlockByCta(f)) EncodeBlockByCta(b, f, py, n, (at, _) => Value(at));
            else EncodeBlock(b, f, py, BlockWarp(b, f, n), n, (at, _) => Value(at));
        }
        else
        {
            string word = GlobalIndex(b), words = WordsOf(b, f, n), pw = b.P(), last = b.R();
            b.L($"setp.ge.u32 {pw}, {word}, {words};");
            b.L($"@{pw} bra DONE;");
            string code = f.WordBits == 64 ? b.RD() : b.R();
            if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
            var natural = f.Natural;
            b.L($"sub.u32 {last}, {n}, 1;");
            for (int i = 0; i < f.Per; i++)
            {
                // A word's values past the count take the last value's addresses (dropped).
                string at = b.R(), inside = b.P();
                b.L($"mad.lo.u32 {at}, {word}, {f.Per}, {i};");
                b.L($"setp.lt.u32 {inside}, {at}, {n};");
                b.L($"min.u32 {at}, {at}, {last};");
                string value = Value(at);
                string stored = lane == natural ? value : Codec.Convert(b, lane, value, natural, f);
                string kept = natural.NewReg(b);
                b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {inside};");
                Codec.Pack(b, f, kept, natural, i, code);
            }

            string addr = b.RD();
            b.L($"mul.wide.u32 {addr}, {word}, {f.WordBits / 8};");
            b.L($"add.u64 {addr}, {py}, {addr};");
            b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_s, .param .u64 p_b, .param .u64 p_y, .param .u32 p_n, .param .u32 p_cols, .param .u32 p_oc, .param .{lane.Mem} p_a",
            f.IsBlock && BlockByCta(f) ? f.BlockValues * 4 : 0);
    }
}
