// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Kernels that write a block format themselves: a thread per output block, each of the block's values the op's own f32
// formula over its inputs, each input decoded a value at a time from its block (Codec.UnpackAt), and the block encoded
// from those values (EncodeBlock). No input or output exists in lane storage.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>c = a + b over n values of block format f, in f32.</summary>
    public static string AddBlocks(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"add_{f.Name}_blocks";
        var b = new PtxBuilder(gpu);
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), n = Param32(b, "p_n");
        if (BlockByThread(f)) AddBlocksThread(b, f, pa, pb, pc, n);
        else
        {
            var w = BlockWarp(b, f, n);
            EncodeBlock(b, f, pc, w, n, at =>
            {
                string x = Codec.UnpackAt(b, f, pa, at, Lane.F32), y = Codec.UnpackAt(b, f, pb, at, Lane.F32), z = b.F();
                b.L(Lane.F32.Add(z, x, y));
                return z;
            });
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_n");
    }

    /// <summary>output = op(x, y, z; a, b) over n values of block format f, in f32 (Map's formula); absent inputs never read.</summary>
    public static string MapBlocks(Gpu gpu, Fmt f, MapOp op, out string entry)
    {
        entry = $"map_{op.Name}_{f.Name}_blocks";
        var b = new PtxBuilder(gpu);
        string px = Param64(b, "p_x"), py = Param64(b, "p_y"), pz = Param64(b, "p_z"), po = Param64(b, "p_o"), n = Param32(b, "p_n");
        string a = b.F(), pb = b.F();
        b.L($"ld.param.f32 {a}, [p_a];");
        b.L($"ld.param.f32 {pb}, [p_b];");
        if (MapByThread(gpu, f, op)) MapBlocksThread(b, f, op, px, py, pz, po, n, a, pb);
        else
        {
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
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u64 p_z, .param .u64 p_o, .param .u32 p_n, .param .f32 p_a, .param .f32 p_b");
    }

    /// <summary>output[r][c] = op(x[r][c], v[c]) over n = rows·cols values of block format f, in f32 (MapRow's formula).</summary>
    public static string MapRowBlocks(Gpu gpu, Fmt f, MapOp op, out string entry)
    {
        entry = $"maprow_{op.Name}_{f.Name}_blocks";
        var b = new PtxBuilder(gpu);
        string px = Param64(b, "p_x"), pv = Param64(b, "p_v"), po = Param64(b, "p_o"), n = Param32(b, "p_n"), cols = Param32(b, "p_cols");
        if (MapByThread(gpu, f, op)) MapRowBlocksThread(b, f, op, px, pv, po, n, cols);
        else
        {
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
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_v, .param .u64 p_o, .param .u32 p_n, .param .u32 p_cols");
    }

    /// <summary>output (cols × rows) = xᵀ (x rows × cols) of block format f: each value moved, its block re-encoded.</summary>
    public static string TransposeBlocks(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"transpose_{f.Name}_blocks";
        var b = new PtxBuilder(gpu);
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

    /// <summary>The input columns a CTA of TransposeBlocksTiled takes.</summary>
    public const int TransposeBlockCols = 32;

    /// <summary>Whether a transpose of block format f takes the tiled kernel: each output row whole blocks (rows a multiple
    /// of the values a block holds), so a tile's output values are whole blocks.</summary>
    public static bool TransposeBlocksTile(Fmt f, int rows) => rows % f.BlockValues == 0;

    /// <summary>The transpose of a block format by tiles (TransposeBlocksTile): a CTA of 256 threads takes input rows
    /// [r0, r0 + BlockValues) and columns [c0, c0 + 32) (grid: column tiles × row tiles). The tile's values decoded a warp
    /// a row of it (UnpackAt: a row's 32 values read from where they are stored together) into shared f32, then each
    /// column of the tile, one output block, encoded from them: a thread a block for the formats of 32 values a block
    /// (ThreadBlockEncode), a warp a block for the others (EncodeBlock). TransposeBlocks gathered each output value from its
    /// own input block. Every value decoded and every block encoded as there: the same blocks.</summary>
    public static string TransposeBlocksTiled(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"transpose_{f.Name}_blocks_tiled";
        int bv = f.BlockValues, stride = TransposeBlockCols + 1;
        var b = new PtxBuilder(gpu);
        string px = Param64(b, "p_x"), po = Param64(b, "p_o"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols");
        string tid = b.R(), c0 = b.R(), r0 = b.R(), n = b.R(), smem = b.R(), i = b.R(), rr = b.R(), cc = b.R(), r = b.R(), c = b.R(), src = b.R(), sa = b.R();
        string pIn = b.P(), pLoop = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {c0}, %ctaid.x;");
        b.L($"shl.b32 {c0}, {c0}, 5;");
        b.L($"mov.u32 {r0}, %ctaid.y;");
        b.L($"mul.lo.u32 {r0}, {r0}, {bv};");
        b.L($"mul.lo.u32 {n}, {rows}, {cols};");
        b.L($"mov.u32 {smem}, smem;");
        // The tile: value (rr, cc) at smem[rr·33 + cc]; past the last column 0.
        b.L($"mov.u32 {i}, {tid};");
        b.Label("LD");
        b.L($"setp.ge.u32 {pLoop}, {i}, {bv * TransposeBlockCols};");
        b.L($"@{pLoop} bra LD_END;");
        b.L($"shr.u32 {rr}, {i}, 5;");
        b.L($"and.b32 {cc}, {i}, 31;");
        b.L($"add.u32 {r}, {r0}, {rr};");
        b.L($"add.u32 {c}, {c0}, {cc};");
        b.L($"setp.lt.u32 {pIn}, {c}, {cols};");
        b.L($"min.u32 {c}, {c}, {cols};");
        b.L($"sub.u32 {c}, {c}, 1;");
        string held = b.R();
        b.L($"add.u32 {held}, {c0}, {cc};");
        b.L($"selp.u32 {c}, {held}, {c}, {pIn};");
        b.L($"mad.lo.u32 {src}, {r}, {cols}, {c};");
        string v = Codec.UnpackAt(b, f, px, src, Lane.F32), kept = b.F();
        b.L($"selp.f32 {kept}, {v}, 0f00000000, {pIn};");
        b.L($"mad.lo.u32 {sa}, {rr}, {stride}, {cc};");
        b.L($"mad.lo.u32 {sa}, {sa}, 4, {smem};");
        b.L($"st.shared.f32 [{sa}], {kept};");
        b.L($"add.u32 {i}, {i}, 256;");
        b.L("bra LD;");
        b.Label("LD_END");
        b.L("bar.sync 0;");
        // Output block of input column c = c0 + cc: output values c·rows + r0 …, block (c·rows + r0) / BlockValues.
        string Block(string col) { string k = b.R(), q = b.R(); b.L($"mul.lo.u32 {k}, {col}, {rows};"); b.L($"add.u32 {k}, {k}, {r0};"); b.L($"div.u32 {q}, {k}, {bv};"); return q; }
        string Shared(string row, string col) { string a = b.R(), x = b.F(); b.L($"mad.lo.u32 {a}, {row}, {stride}, {col};"); b.L($"mad.lo.u32 {a}, {a}, 4, {smem};"); b.L($"ld.shared.f32 {x}, [{a}];"); return x; }
        if (BlockByThread(f))
        {
            // Threads 0 … 31: column tid's block.
            string pOut = b.P(), col = b.R();
            b.L($"add.u32 {col}, {c0}, {tid};");
            b.L($"setp.ge.u32 {pOut}, {tid}, 32;");
            b.L($"@{pOut} bra DONE;");
            b.L($"setp.ge.u32 {pOut}, {col}, {cols};");
            b.L($"@{pOut} bra DONE;");
            string blk = Block(col);
            var vals = new string[bv];
            for (int k = 0; k < bv; k++) vals[k] = Shared(k.ToString(), tid);
            ThreadBlockEncode(b, f, po, blk, vals);
        }
        else
        {
            // Warp w: columns w, w + 8, w + 16, w + 24 of the tile, a block each.
            string warp = b.R(), lane = b.R(), j = b.R(), col = b.R(), tcol = b.R(), pDone = b.P();
            b.L($"shr.u32 {warp}, {tid}, 5;");
            b.L($"and.b32 {lane}, {tid}, 31;");
            b.L($"mov.u32 {j}, 0;");
            b.Label("ENC");
            b.L($"setp.ge.u32 {pDone}, {j}, 4;");
            b.L($"@{pDone} bra DONE;");
            b.L($"mad.lo.u32 {tcol}, {j}, 8, {warp};");
            b.L($"add.u32 {col}, {c0}, {tcol};");
            b.L($"setp.ge.u32 {pDone}, {col}, {cols};");
            b.L($"@{pDone} bra DONE;");
            string blk = Block(col), blockFirst = b.R();
            b.L($"mul.lo.u32 {blockFirst}, {blk}, {bv};");
            EncodeBlock(b, f, po, (blk, lane), n, at =>
            {
                string row = b.R();
                b.L($"sub.u32 {row}, {at}, {blockFirst};");
                return Shared(row, tcol);
            });
            b.L($"add.u32 {j}, {j}, 1;");
            b.L("bra ENC;");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_o, .param .u32 p_rows, .param .u32 p_cols", bv * stride * 4);
    }
}
