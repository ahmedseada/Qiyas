// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A scaled weight's values on CUDA (Products.cs ScaledWeight): the code decoded into the lane where it is stored, its zero
// subtracted and its scale multiplied there; the stored GEMM's B loads take them so (Ptx.Gemm, GemmStored.Scales), and
// ExpandScaled's kernel stores them as the type. Quantize's kernels: each block's range and scale (a CTA a scale), the zero
// points from the stored scales (a thread each), the codes (StoreEach into TW, each rounded once).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>(code − zero)·scale of the weight's value at (row, col) from its code v (a lane register): the scale and zero
    /// point at (row / br)·sc + col / bc of fs at ps and pz (pz null: no zero points); inside false reads index 0.</summary>
    internal static string ScaleCode(PtxBuilder b, Fmt fs, Lane lane, string v, string ps, string? pz, string row, string col, string br, string bc, string sc, string? inside = null)
    {
        string si = b.R(), t = b.R();
        b.L($"div.u32 {si}, {row}, {br};");
        b.L($"div.u32 {t}, {col}, {bc};");
        b.L($"mad.lo.u32 {si}, {si}, {sc}, {t};");
        if (inside is not null) b.L($"selp.u32 {si}, {si}, 0, {inside};");
        if (pz is not null) v = LaneOp(b, lane, "sub", v, Codec.UnpackAt(b, fs, pz, si, lane));
        return LaneOp(b, lane, "mul", v, Codec.UnpackAt(b, fs, ps, si, lane));
    }

    /// <summary>Each block's scale (lane values at ps) and its range's low end lo = min(x, 0) (lane values at pl, with zero
    /// points), a CTA a scale: max(hi, −lo) / max, or (hi − lo) / (max − min). Params: x (f), ps, pl, n (N), k, br, bc, sc,
    /// min, max (lane).</summary>
    public static string QuantizeRange(Gpu gpu, Fmt f, Lane lane, bool zeros, out string entry)
    {
        entry = $"quantize_range_{f.Name}_{lane.Name}{(zeros ? "_z" : "")}";
        const int Threads = 256;
        var b = new PtxBuilder();
        string tid = b.R(), si = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {si}, %ctaid.x;");
        string px = Param64(b, "p_x"), ps = Param64(b, "p_s"), pl = Param64(b, "p_l");
        string n = Param32(b, "p_n"), k = Param32(b, "p_k"), br = Param32(b, "p_br"), bc = Param32(b, "p_bc"), sc = Param32(b, "p_sc");
        string min = lane.NewReg(b), max = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {min}, [p_min];");
        b.L($"ld.param.{lane.Op} {max}, [p_max];");
        // The block: rows [r0, r0 + rows), columns [c0, c0 + cols).
        string r0 = b.R(), c0 = b.R(), t = b.R(), rows = b.R(), cols = b.R(), count = b.R(), inside = b.P();
        b.L($"div.u32 {r0}, {si}, {sc};");
        b.L($"mul.lo.u32 {t}, {r0}, {sc};");
        b.L($"sub.u32 {c0}, {si}, {t};");
        b.L($"mul.lo.u32 {r0}, {r0}, {br};");
        b.L($"mul.lo.u32 {c0}, {c0}, {bc};");
        b.L($"sub.u32 {rows}, {n}, {r0};");
        b.L($"min.u32 {rows}, {rows}, {br};");
        b.L($"setp.lt.u32 {inside}, {c0}, {k};");
        b.L($"sub.u32 {cols}, {k}, {c0};");
        b.L($"min.u32 {cols}, {cols}, {bc};");
        b.L($"selp.u32 {cols}, {cols}, 0, {inside};");
        b.L($"mul.lo.u32 {count}, {rows}, {cols};");
        string lo = lane.NewReg(b), hi = lane.NewReg(b), e = b.R(), pEnd = b.P();
        b.L($"mov.{lane.Reg} {lo}, {lane.Zero};");
        b.L($"mov.{lane.Reg} {hi}, {lane.Zero};");
        b.L($"mov.u32 {e}, {tid};");
        b.Label("QR_LOOP");
        b.L($"setp.ge.u32 {pEnd}, {e}, {count};");
        b.L($"@{pEnd} bra QR_END;");
        {
            string r = b.R(), c = b.R(), idx = b.R();
            b.L($"div.u32 {r}, {e}, {cols};");
            b.L($"mul.lo.u32 {t}, {r}, {cols};");
            b.L($"sub.u32 {c}, {e}, {t};");
            b.L($"add.u32 {r}, {r}, {r0};");
            b.L($"add.u32 {c}, {c}, {c0};");
            b.L($"mad.lo.u32 {idx}, {r}, {k}, {c};");
            string x = Codec.UnpackAt(b, f, px, idx, lane);
            b.L($"min.{lane.Op} {lo}, {lo}, {x};");
            b.L($"max.{lane.Op} {hi}, {hi}, {x};");
        }

        b.L($"add.u32 {e}, {e}, {Threads};");
        b.L("bra QR_LOOP;");
        b.Label("QR_END");
        string negLo = lane.NewReg(b);
        b.L($"neg.{lane.Op} {negLo}, {lo};");
        string top = BlockReduce(b, lane, tid, hi, "max", Threads), bottom = BlockReduce(b, lane, tid, negLo, "max", Threads);     // hi, −lo
        string scale = zeros ? LaneOp(b, lane, "div", LaneOp(b, lane, "add", top, bottom), LaneOp(b, lane, "sub", max, min))
                             : LaneOp(b, lane, "div", MaxOf(b, lane, top, bottom), max);
        string first = b.P(), a = b.RD();
        b.L($"setp.ne.u32 {first}, {tid}, 0;");
        b.L($"@{first} bra DONE;");
        b.L($"mul.wide.u32 {a}, {si}, {lane.Size};");
        string sa = b.RD();
        b.L($"add.u64 {sa}, {ps}, {a};");
        b.L($"st.global.{lane.Mem} [{sa}], {scale};");
        if (zeros)
        {
            string la = b.RD(), low = lane.NewReg(b);
            b.L($"neg.{lane.Op} {low}, {bottom};");
            b.L($"add.u64 {la}, {pl}, {a};");
            b.L($"st.global.{lane.Mem} [{la}], {low};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_s, .param .u64 p_l, .param .u32 p_n, .param .u32 p_k, .param .u32 p_br, .param .u32 p_bc, .param .u32 p_sc, .param .{lane.Op} p_min, .param .{lane.Op} p_max", ReduceBytes);
    }

    private static string MaxOf(PtxBuilder b, Lane lane, string x, string y)
    {
        string d = lane.NewReg(b);
        b.L($"max.{lane.Op} {d}, {x}, {y};");
        return d;
    }

    // v rounded to an integer (ties to even) in the float lane.
    private static string RoundLane(PtxBuilder b, Lane lane, string v)
    {
        string d = lane.NewReg(b);
        b.L($"cvt.rni.{lane.Op}.{lane.Op} {d}, {v};");
        return d;
    }

    // v held to [min, max].
    private static string HeldLane(PtxBuilder b, Lane lane, string v, string min, string max)
    {
        string d = lane.NewReg(b);
        b.L($"max.{lane.Op} {d}, {v}, {min};");
        b.L($"min.{lane.Op} {d}, {d}, {max};");
        return d;
    }

    /// <summary>Each zero point (lane values at pz) from the stored scale (f at ps) and the block's low end (lane values at pl):
    /// round(min − lo / scale) held to [min, max], 0 where the scale is 0; a thread each. Params: ps, pl, pz, n (the scales),
    /// min, max (lane).</summary>
    public static string QuantizeZeros(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"quantize_zeros_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string ps = Param64(b, "p_s"), pl = Param64(b, "p_l"), pz = Param64(b, "p_z"), n = Param32(b, "p_n");
        string min = lane.NewReg(b), max = lane.NewReg(b), pEnd = b.P();
        b.L($"ld.param.{lane.Op} {min}, [p_min];");
        b.L($"ld.param.{lane.Op} {max}, [p_max];");
        b.L($"setp.ge.u32 {pEnd}, {gid}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string s = Codec.UnpackAt(b, f, ps, gid, lane), lo = LoadLane(b, lane, pl, gid);
        string zero = HeldLane(b, lane, RoundLane(b, lane, LaneOp(b, lane, "sub", min, LaneOp(b, lane, "div", lo, s))), min, max);
        string positive = b.P(), kept = lane.NewReg(b), a = b.RD();
        b.L($"setp.gt.{lane.Op} {positive}, {s}, {lane.Zero};");
        b.L($"selp.{lane.Op} {kept}, {zero}, {lane.Zero}, {positive};");
        b.L($"mul.wide.u32 {a}, {gid}, {lane.Size};");
        b.L($"add.u64 {a}, {pz}, {a};");
        b.L($"st.global.{lane.Mem} [{a}], {kept};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_s, .param .u64 p_l, .param .u64 p_z, .param .u32 p_n, .param .{lane.Op} p_min, .param .{lane.Op} p_max");
    }

    /// <summary>The codes (fw) of the values (f): x / scale (+ the zero point) by the stored scale and zero point (f), held to
    /// [min, max], an integer code rounded to the nearest (ties to even) in the lane, each rounded to fw once (an f64 lane's
    /// value to a narrow float rounded to odd at float32 first). Params: x, scales, zeros, codes, n (N·K), k, br, bc, sc,
    /// min, max (lane).</summary>
    public static string QuantizeCodes(Gpu gpu, Fmt f, Fmt fw, Lane lane, bool zeros, out string entry)
    {
        entry = $"quantize_codes_{f.Name}_{fw.Name}_{lane.Name}{(zeros ? "_z" : "")}";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), ps = Param64(b, "p_s"), pz = Param64(b, "p_z"), pc = Param64(b, "p_c");
        string n = Param32(b, "p_n"), k = Param32(b, "p_k"), br = Param32(b, "p_br"), bc = Param32(b, "p_bc"), sc = Param32(b, "p_sc");
        string min = lane.NewReg(b), max = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {min}, [p_min];");
        b.L($"ld.param.{lane.Op} {max}, [p_max];");
        bool toOdd = Fmt.ConvertPlan(lane.Wide ? new Fmt("float64", 64, true, true) : new Fmt("float32", 32, true, true), fw).ToOdd;
        StoreEach(b, fw, toOdd ? Lane.F32 : lane, pc, n, at =>
        {
            string r = b.R(), c = b.R(), si = b.R(), t = b.R();
            b.L($"div.u32 {r}, {at}, {k};");
            b.L($"rem.u32 {c}, {at}, {k};");
            b.L($"div.u32 {si}, {r}, {br};");
            b.L($"div.u32 {t}, {c}, {bc};");
            b.L($"mad.lo.u32 {si}, {si}, {sc}, {t};");
            string s = Codec.UnpackAt(b, f, ps, si, lane), x = Codec.UnpackAt(b, f, px, at, lane);
            string q = LaneOp(b, lane, "div", x, s), positive = b.P(), kept = lane.NewReg(b);
            b.L($"setp.gt.{lane.Op} {positive}, {s}, {lane.Zero};");
            b.L($"selp.{lane.Op} {kept}, {q}, {lane.Zero}, {positive};");
            q = kept;
            if (!fw.IsFloat) q = RoundLane(b, lane, q);
            if (zeros) q = LaneOp(b, lane, "add", q, Codec.UnpackAt(b, f, pz, si, lane));
            q = HeldLane(b, lane, q, min, max);
            return toOdd ? ToOddF32(b, lane, q) : q;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_s, .param .u64 p_z, .param .u64 p_c, .param .u32 p_n, .param .u32 p_k, .param .u32 p_br, .param .u32 p_bc, .param .u32 p_sc, .param .{lane.Op} p_min, .param .{lane.Op} p_max", EachShared(fw));
    }

    /// <summary>The weight's values as f. Params: codes (fw), scales, zeros (f), y, n (N·K), k, br, bc, sc.</summary>
    public static string ExpandScaled(Gpu gpu, Fmt f, Fmt fw, Lane lane, bool zeros, out string entry)
    {
        entry = $"expand_scaled_{fw.Name}_{f.Name}_{lane.Name}{(zeros ? "_z" : "")}";
        var b = new PtxBuilder();
        string pc = Param64(b, "p_c"), ps = Param64(b, "p_s"), pz = Param64(b, "p_z"), py = Param64(b, "p_y");
        string n = Param32(b, "p_n"), k = Param32(b, "p_k"), br = Param32(b, "p_br"), bc = Param32(b, "p_bc"), sc = Param32(b, "p_sc");
        StoreEach(b, f, lane, py, n, at =>
        {
            string r = b.R(), c = b.R();
            b.L($"div.u32 {r}, {at}, {k};");
            b.L($"rem.u32 {c}, {at}, {k};");
            return ScaleCode(b, f, lane, Codec.UnpackAt(b, fw, pc, at, lane), ps, zeros ? pz : null, r, c, br, bc, sc);
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_c, .param .u64 p_s, .param .u64 p_z, .param .u64 p_y, .param .u32 p_n, .param .u32 p_k, .param .u32 p_br, .param .u32 p_bc, .param .u32 p_sc", EachShared(f));
    }
}
