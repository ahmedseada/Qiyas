// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Kernels that write a block format themselves: a thread per output block, each of the block's values the op's own f32
// formula over its inputs, each input decoded a value at a time from its block (Codec.UnpackAt), and the block encoded
// from those values (EncodeBlock). No input or output exists in lane storage.
namespace Adad;

internal static partial class Ptx
{
    /// <summary>c = a + b over n values of block format f, in f32.</summary>
    public static string AddBlocks(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"add_{f.Name}_blocks";
        var b = new PtxBuilder();
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), n = Param32(b, "p_n");
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, pc, w, n, at =>
        {
            string x = Codec.UnpackAt(b, f, pa, at, Lane.F32), y = Codec.UnpackAt(b, f, pb, at, Lane.F32), z = b.F();
            b.L(Lane.F32.Add(z, x, y));
            return z;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_n");
    }

    /// <summary>output = op(x, y, z; a, b) over n values of block format f, in f32 (Map's formula); absent inputs never read.</summary>
    public static string MapBlocks(Gpu gpu, Fmt f, MapOp op, out string entry)
    {
        entry = $"map_{op.Name}_{f.Name}_blocks";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), py = Param64(b, "p_y"), pz = Param64(b, "p_z"), po = Param64(b, "p_o"), n = Param32(b, "p_n");
        string a = b.F(), pb = b.F();
        b.L($"ld.param.f32 {a}, [p_a];");
        b.L($"ld.param.f32 {pb}, [p_b];");
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, po, w, n, at =>
        {
            string x;
            if (op.Inputs >= 1) x = Codec.UnpackAt(b, f, px, at, Lane.F32);
            else { x = b.F(); b.L($"mov.f32 {x}, 0f00000000;"); }
            string y = op.Inputs >= 2 ? Codec.UnpackAt(b, f, py, at, Lane.F32) : x;
            string z = op.Inputs >= 3 ? Codec.UnpackAt(b, f, pz, at, Lane.F32) : x;
            return Formula(b, Lane.F32, op.Name, x, y, z, a, pb);
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u64 p_z, .param .u64 p_o, .param .u32 p_n, .param .f32 p_a, .param .f32 p_b");
    }

    /// <summary>output[r][c] = op(x[r][c], v[c]) over n = rows·cols values of block format f, in f32 (MapRow's formula).</summary>
    public static string MapRowBlocks(Gpu gpu, Fmt f, MapOp op, out string entry)
    {
        entry = $"maprow_{op.Name}_{f.Name}_blocks";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), pv = Param64(b, "p_v"), po = Param64(b, "p_o"), n = Param32(b, "p_n"), cols = Param32(b, "p_cols");
        string zero = b.F();
        b.L($"mov.f32 {zero}, 0f00000000;");
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, po, w, n, at =>
        {
            string col = b.R();
            b.L($"rem.u32 {col}, {at}, {cols};");
            string x = Codec.UnpackAt(b, f, px, at, Lane.F32), y = Codec.UnpackAt(b, f, pv, col, Lane.F32);
            return Formula(b, Lane.F32, op.Name, x, y, zero, zero, zero);
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_v, .param .u64 p_o, .param .u32 p_n, .param .u32 p_cols");
    }

    /// <summary>output (cols × rows) = xᵀ (x rows × cols) of block format f: each value moved, its block re-encoded.</summary>
    public static string TransposeBlocks(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"transpose_{f.Name}_blocks";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), po = Param64(b, "p_o"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols"), n = b.R();
        b.L($"mul.lo.u32 {n}, {rows}, {cols};");
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, po, w, n, at =>
        {
            // Output value at = c·rows + r is x[r][c].
            string c = b.R(), r = b.R(), src = b.R();
            b.L($"div.u32 {c}, {at}, {rows};");
            b.L($"mul.lo.u32 {r}, {c}, {rows};");
            b.L($"sub.u32 {r}, {at}, {r};");
            b.L($"mad.lo.u32 {src}, {r}, {cols}, {c};");
            return Codec.UnpackAt(b, f, px, src, Lane.F32);
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_o, .param .u32 p_rows, .param .u32 p_cols");
    }
}
