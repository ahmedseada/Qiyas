// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A product's store on CUDA (Products.cs ProductEpilogue; GpuOps.Products.cs, and the general convolution's, GpuOps.Conv.cs):
// y = the store terms' value of the lane sums, a thread per stored word (a warp, or a CTA, per block of a block format):
// the sums and a second product's (lane values), the bias, β·y (y's value read by the thread, warp or CTA that writes its
// word or block, before it does), the activation or the gradient through one (the element-wise formula's, Ptx.Map.cs) and
// the residual, every operand read where it is stored and every term in the lane, each value rounded to the type once.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Params: sums (lane storage), the second product's sums (lane storage), bias (an output channel's, the channel
    /// (at / cols) mod oc), residual, the activation's inputs (a gradient's x), y, y's values before (β's term: y, or the
    /// product's y where the pass stores the sums kept), n, cols, oc, the formula's parameters a and b, and β (lane values).
    /// A term the store does not take has its pointer unread.</summary>
    public static string ProductStore(Gpu gpu, Fmt f, Lane lane, StoreTerms t, out string entry)
    {
        entry = $"product_store_{f.Name}_{lane.Name}_{t.Key}";
        var b = new PtxBuilder(gpu);
        string ps = Param64(b, "p_s"), pl = Param64(b, "p_l"), pb = Param64(b, "p_b"), pr = Param64(b, "p_r"), px = Param64(b, "p_x"), py = Param64(b, "p_y"), pc = Param64(b, "p_c");
        string n = Param32(b, "p_n"), cols = Param32(b, "p_cols"), oc = Param32(b, "p_oc");
        string pa = lane.NewReg(b), pab = lane.NewReg(b), beta = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {pa}, [p_a];");
        b.L($"ld.param.{lane.Mem} {pab}, [p_ab];");
        b.L($"ld.param.{lane.Mem} {beta}, [p_beta];");
        var e = new Emit(b, lane);
        string Value(string at)
        {
            string v = LoadLane(b, lane, ps, at);
            if (t.Second) v = e.Add(v, LoadLane(b, lane, pl, at));
            if (t.Bias) v = e.Add(v, Codec.UnpackAt(b, f, pb, Within(b, at, cols, oc), lane));
            if (t.Beta) v = e.Fma(beta, Codec.UnpackAt(b, f, pc, at, lane), v);
            if (t.Grad is not null) v = Formula(b, lane, t.Grad, Codec.UnpackAt(b, f, px, at, lane), v, v, pa, pab);
            else if (t.Act is not null) v = Formula(b, lane, t.Act, v, v, v, pa, pab);
            if (t.Residual) v = e.Add(v, Codec.UnpackAt(b, f, pr, at, lane));
            return v;
        }

        StoreEach(b, f, lane, py, n, Value);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_s, .param .u64 p_l, .param .u64 p_b, .param .u64 p_r, .param .u64 p_x, .param .u64 p_y, .param .u64 p_c, .param .u32 p_n, .param .u32 p_cols, .param .u32 p_oc, .param .{lane.Mem} p_a, .param .{lane.Mem} p_ab, .param .{lane.Mem} p_beta", EachShared(f));
    }

    /// <summary>The n values of f at py from value(at) (a lane register; at a .u32 register below n), each rounded to the type
    /// once: a thread per stored word, or a warp (a CTA where BlockByCta) per block of a block format, whose source must be
    /// straight-line code. Threads past the values go to DONE, which the caller places; the launch is GpuOps.LaunchEach's.
    /// A float type's value goes through its natural lane; an integer type's is packed from the lane itself (Codec.Pack
    /// holds a wide or float lane's value to the type's range, where a narrowing first would cut it).</summary>
    internal static void StoreEach(PtxBuilder b, Fmt f, Lane lane, string py, string n, Func<string, string> value)
    {
        if (f.IsBlock)
        {
            if (BlockByCta(f)) EncodeBlockByCta(b, f, py, n, (at, _) => value(at));
            else EncodeBlock(b, f, py, BlockWarp(b, f, n), n, (at, _) => value(at));
            return;
        }

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
            string v = value(at);
            var into = f.IsFloat ? natural : lane;
            string stored = lane == into ? v : Codec.Convert(b, lane, v, into, f);
            string kept = into.NewReg(b);
            b.L($"selp.{into.Reg} {kept}, {stored}, {into.Zero}, {inside};");
            Codec.Pack(b, f, kept, into, i, code);
        }

        string addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {word}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {py}, {addr};");
        b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
    }

    /// <summary>The shared bytes a StoreEach kernel takes.</summary>
    internal static int EachShared(Fmt f) => f.IsBlock && BlockByCta(f) ? f.BlockValues * 4 : 0;
}
