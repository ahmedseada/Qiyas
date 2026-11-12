// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's optimizer steps (Optimizers.cs): a thread per word of the parameters (its Per values), the gradients'
// word alongside, the moments as lane values at the value's index; the parameters packed back into their word. Scaled:
// every gradient times the lane value at p_gs as it is read (a clipping factor made on the device: GpuOps.Optimizers.cs).
// Table: the step of several tensors in one launch — a thread's word falls in a row of a table on the device (TableRow:
// the tensor's first word, its count, where its p, g, m and v are), and the thread steps that tensor's word.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>The row of a table on the device an index falls in. The table: `count` rows of `width` 64-bit entries, the
    /// first entry of each the first index of its row (ascending, row 0's 0). Returns the address of the last row whose
    /// first index is at most `index` (the last row for an index past them all: the caller's own bound drops it) and the
    /// index past that first, by halving: a few loads a thread, the same for every thread of a block.</summary>
    internal static (string Row, string Local) TableRow(PtxBuilder b, string tab, string count, string index, int width)
    {
        string lo = b.R(), hi = b.R(), mid = b.R(), gap = b.R(), done = b.P(), le = b.P(), addr = b.RD(), first = b.RD(), wide = b.RD();
        b.L($"cvt.u64.u32 {wide}, {index};");
        b.L($"mov.u32 {lo}, 0;");
        b.L($"mov.u32 {hi}, {count};");
        b.Label("TAB_LOOP");
        b.L($"sub.u32 {gap}, {hi}, {lo};");
        b.L($"setp.le.u32 {done}, {gap}, 1;");
        b.L($"@{done} bra TAB_END;");
        b.L($"add.u32 {mid}, {lo}, {hi};");
        b.L($"shr.u32 {mid}, {mid}, 1;");
        b.L($"mul.wide.u32 {addr}, {mid}, {width * 8};");
        b.L($"add.u64 {addr}, {tab}, {addr};");
        b.L($"ld.global.u64 {first}, [{addr}];");
        b.L($"setp.le.u64 {le}, {first}, {wide};");
        b.L($"selp.u32 {lo}, {mid}, {lo}, {le};");
        b.L($"selp.u32 {hi}, {hi}, {mid}, {le};");
        b.L("bra TAB_LOOP;");
        b.Label("TAB_END");
        string row = b.RD(), local = b.R();
        b.L($"mul.wide.u32 {row}, {lo}, {width * 8};");
        b.L($"add.u64 {row}, {tab}, {row};");
        b.L($"ld.global.u64 {first}, [{row}];");
        b.L($"sub.u64 {wide}, {wide}, {first};");
        b.L($"cvt.u32.u64 {local}, {wide};");
        return (row, local);
    }

    /// <summary>A step table's row: the tensor's first word, its values, where its p, g, m and v are.</summary>
    public const int StepRow = 6;

    public static string Optimizer(Gpu gpu, Fmt f, Lane lane, string name, out string entry, bool scaled = false, bool table = false)
    {
        entry = $"optimizer_{name}_{f.Name}_{lane.Name}{(scaled ? "_gs" : "")}{(table ? "_tab" : "")}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), word = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {word}, {bid}, 256, {tid};");
        string pp = Param64(b, "p_p"), pg = Param64(b, "p_g"), pm = Param64(b, "p_m"), pv = Param64(b, "p_v"), n = Param32(b, "p_n");
        var k = StepScalars.Load(b, lane);
        string? gs = scaled ? GradientScale(b, lane) : null;
        if (table)
        {
            // p_p is the table and p_n its rows: the thread's tensor, and its word of it.
            var (row, local) = TableRow(b, pp, n, word, StepRow);
            string count = b.RD();
            b.L($"ld.global.u64 {count}, [{row}+8];");
            b.L($"cvt.u32.u64 {n}, {count};");
            b.L($"ld.global.u64 {pg}, [{row}+24];");
            b.L($"ld.global.u64 {pm}, [{row}+32];");
            b.L($"ld.global.u64 {pv}, [{row}+40];");
            b.L($"ld.global.u64 {pp}, [{row}+16];");
            b.L($"mov.u32 {word}, {local};");
        }

        string words = b.R(), pw = b.P();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else { b.L($"add.u32 {words}, {n}, {f.Per - 1};"); b.L($"div.u32 {words}, {words}, {f.Per};"); }
        b.L($"setp.ge.u32 {pw}, {word}, {words};");
        b.L($"@{pw} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32", addrP = b.RD(), addrG = b.RD(), wordP = f.WordBits == 64 ? b.RD() : b.R(), wordG = f.WordBits == 64 ? b.RD() : b.R();
        b.L($"mul.wide.u32 {addrP}, {word}, {f.WordBits / 8};");
        b.L($"add.u64 {addrG}, {pg}, {addrP};");
        b.L($"add.u64 {addrP}, {pp}, {addrP};");
        b.L($"ld.global.{wt} {wordP}, [{addrP}];");
        b.L($"ld.global.{wt} {wordG}, [{addrG}];");
        string outWord = f.WordBits == 64 ? b.RD() : b.R();
        if (f.WordBits == 32) b.L($"mov.b32 {outWord}, 0;");
        bool plain = name == "sgd", sgd = plain || name == "sgd_momentum";
        var natural = f.Natural;
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R(), pIn = b.P(), mAddr = b.RD(), mv = lane.NewReg(b);
            b.L($"mad.lo.u32 {idx}, {word}, {f.Per}, {i};");
            b.L($"setp.lt.u32 {pIn}, {idx}, {n};");
            b.L($"@!{pIn} bra SKIP{i};");
            string p = Codec.Convert(b, natural, Codec.Unpack(b, f, wordP, i, natural), lane, f);
            string g = Codec.Convert(b, natural, Codec.Unpack(b, f, wordG, i, natural), lane, f);
            if (gs is not null) g = LaneOp(b, lane, "mul", g, gs);
            b.L($"mul.wide.u32 {mAddr}, {idx}, {lane.Size};");
            string mPtr = b.RD();
            if (!plain)
            {
                b.L($"add.u64 {mPtr}, {pm}, {mAddr};");
                b.L($"ld.global.{lane.Mem} {mv}, [{mPtr}];");
            }

            string vPtr = sgd ? "" : b.RD(), vv = sgd ? "" : lane.NewReg(b);
            if (!sgd)
            {
                b.L($"add.u64 {vPtr}, {pv}, {mAddr};");
                b.L($"ld.global.{lane.Mem} {vv}, [{vPtr}];");
            }

            string pNew = Step(b, lane, name, p, g, mv, vv, k);
            if (!sgd) b.L($"st.global.{lane.Mem} [{vPtr}], {vv};");
            if (!plain) b.L($"st.global.{lane.Mem} [{mPtr}], {mv};");
            string stored = lane == natural ? pNew : Codec.Convert(b, lane, pNew, natural, f);
            Codec.Pack(b, f, stored, natural, i, outWord);
            b.Label($"SKIP{i}");
        }

        b.L($"st.global.{wt} [{addrP}], {outWord};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, OptimizerParams(lane));
    }

    /// <summary>The step over n parameters of block format f (m, v as f32 lane storage): a warp per block, each value's step
    /// in f32 registers from p and g decoded where they are stored, m and v stored (only for values below n: a lane past
    /// the end evaluates the last value again), the block re-encoded in place (its values all read before the warp's
    /// first shuffle, so before any of its bytes is written).</summary>
    public static string OptimizerBlocks(Gpu gpu, Fmt f, string name, out string entry, bool scaled = false)
    {
        entry = $"optimizer_{name}_{f.Name}_blocks{(scaled ? "_gs" : "")}";
        var lane = Lane.F32;
        var b = new PtxBuilder(gpu);
        string pp = Param64(b, "p_p"), pg = Param64(b, "p_g"), pm = Param64(b, "p_m"), pv = Param64(b, "p_v"), n = Param32(b, "p_n");
        var k = StepScalars.Load(b, lane);
        string? gs = scaled ? GradientScale(b, lane) : null;
        bool plain = name == "sgd", sgd = plain || name == "sgd_momentum";
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, pp, w, n, (at, pIn) =>
        {
            string p = Codec.UnpackAt(b, f, pp, at, lane), g = Codec.UnpackAt(b, f, pg, at, lane);
            if (gs is not null) g = LaneOp(b, lane, "mul", g, gs);
            string off = b.RD(), mPtr = b.RD(), mv = lane.NewReg(b);
            b.L($"mul.wide.u32 {off}, {at}, {lane.Size};");
            if (!plain)
            {
                b.L($"add.u64 {mPtr}, {pm}, {off};");
                b.L($"ld.global.{lane.Mem} {mv}, [{mPtr}];");
            }

            string vPtr = sgd ? "" : b.RD(), vv = sgd ? "" : lane.NewReg(b);
            if (!sgd)
            {
                b.L($"add.u64 {vPtr}, {pv}, {off};");
                b.L($"ld.global.{lane.Mem} {vv}, [{vPtr}];");
            }

            string pNew = Step(b, lane, name, p, g, mv, vv, k);
            if (!sgd) b.L($"@{pIn} st.global.{lane.Mem} [{vPtr}], {vv};");
            if (!plain) b.L($"@{pIn} st.global.{lane.Mem} [{mPtr}], {mv};");
            return pNew;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, OptimizerParams(lane));
    }

    private static string OptimizerParams(Lane lane)
    {
        string pa = lane.Wide ? "f64" : "f32";
        return $".param .u64 p_p, .param .u64 p_g, .param .u64 p_m, .param .u64 p_v, .param .u32 p_n, .param .{pa} p_lr, .param .{pa} p_b1, .param .{pa} p_b2, .param .{pa} p_ob1, .param .{pa} p_ob2, .param .{pa} p_ic1, .param .{pa} p_ic2, .param .{pa} p_eps, .param .{pa} p_wd, .param .{pa} p_lrwd, .param .u64 p_gs";
    }

    // The gradients' scale: the lane value at p_gs (a scaled step's last argument).
    private static string GradientScale(PtxBuilder b, Lane lane)
    {
        string pgs = Param64(b, "p_gs"), gs = lane.NewReg(b);
        b.L($"ld.global.{lane.Mem} {gs}, [{pgs}];");
        return gs;
    }

    private sealed record StepScalars(string Lr, string B1, string B2, string Ob1, string Ob2, string Ic1, string Ic2, string Eps, string Wd, string LrWd)
    {
        public static StepScalars Load(PtxBuilder b, Lane lane)
        {
            string Scalar(string nm) { string r = lane.NewReg(b); b.L($"ld.param.{lane.Op} {r}, [{nm}];"); return r; }
            return new(Scalar("p_lr"), Scalar("p_b1"), Scalar("p_b2"), Scalar("p_ob1"), Scalar("p_ob2"), Scalar("p_ic1"), Scalar("p_ic2"), Scalar("p_eps"), Scalar("p_wd"), Scalar("p_lrwd"));
        }
    }

    // One value's step: m (and v) updated in their registers, the new parameter returned.
    //   sgd:          p = p − lr·(g + wd·p)
    //   sgd_momentum: m = b1·m + g + wd·p;  p = p − lr·m
    //   adam / adamw: m = b1·m + (1 − b1)·g; v = b2·v + (1 − b2)·g²; p = p − lr·m̂ / (√v̂ + eps) (adamw: p − lr·wd·p first,
    //                 the decay of the parameter the step starts from)
    private static string Step(PtxBuilder b, Lane lane, string name, string p, string g, string mv, string vv, StepScalars k)
    {
        string pNew = lane.NewReg(b), t = lane.NewReg(b);
        if (name == "sgd")
        {
            b.L(lane.Fma(t, k.Wd, p, g));
            b.L(lane.Mul(t, k.Lr, t));
            b.L($"sub.rn.{lane.Op} {pNew}, {p}, {t};");
            return pNew;
        }

        if (name == "sgd_momentum")
        {
            b.L(lane.Fma(t, k.Wd, p, g));
            b.L(lane.Fma(mv, k.B1, mv, t));
            b.L(lane.Mul(t, k.Lr, mv));
            b.L($"sub.rn.{lane.Op} {pNew}, {p}, {t};");
            return pNew;
        }

        string g2 = lane.NewReg(b), mh = lane.NewReg(b), vh = lane.NewReg(b), root = lane.NewReg(b), den = lane.NewReg(b), upd = lane.NewReg(b);
        b.L(lane.Mul(t, k.Ob1, g));
        b.L(lane.Fma(mv, k.B1, mv, t));
        b.L(lane.Mul(g2, g, g));
        b.L(lane.Mul(t, k.Ob2, g2));
        b.L(lane.Fma(vv, k.B2, vv, t));
        b.L(lane.Mul(mh, mv, k.Ic1));
        b.L(lane.Mul(vh, vv, k.Ic2));
        b.L($"sqrt.rn.{lane.Op} {root}, {vh};");
        b.L(lane.Add(den, root, k.Eps));
        b.L(lane.Mul(upd, k.Lr, mh));
        b.L($"div.rn.{lane.Op} {upd}, {upd}, {den};");
        string p0 = p;
        if (name == "adamw")
        {
            string decay = lane.NewReg(b);
            p0 = lane.NewReg(b);
            b.L(lane.Mul(decay, k.LrWd, p));
            b.L($"sub.rn.{lane.Op} {p0}, {p}, {decay};");
        }

        b.L($"sub.rn.{lane.Op} {pNew}, {p0}, {upd};");
        return pNew;
    }
}
