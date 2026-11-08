// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A scaled weight's values on CUDA (Products.cs ScaledWeight): the code decoded into the lane where it is stored, its zero
// subtracted and its scale multiplied there; the stored GEMM's B loads take them so (Ptx.Gemm, GemmStored.Scales), and
// ExpandScaled's kernel stores them as the type.
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
