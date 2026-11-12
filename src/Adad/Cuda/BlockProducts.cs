// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Block-scaled integer products: the block formats' own fast path. A row of Q8_0 (or Q4_0) against a vector of the same
// format is Σ over blocks of d_a·d_x·(Σ_i q_a,i · q_x,i): the inner sum is an integer dot of 32 products (exact in s32,
// the s8 dot lane where the CPU has it), the outer sum the float lane. Nothing is decoded to a float per value.
//
//   gemv (y = A·x) for Q8_0 and Q4_0 operands: a thread (CPU: a row at a time on the team; device: a thread per row)

using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>y[row] = Σ_blocks d_a·d_x·(Σ q_a·q_x) for Q8_0 / Q4_0 rows against a vector of the same format; a thread per row, the
    /// integer dot by dp4a where the device has it (else multiplies), the block sums in f32.</summary>
    public static string GemvBlocks(Gpu gpu, Fmt f, bool dp4a, out string entry)
    {
        entry = $"gemv_blocks_{f.Name}{(dp4a ? "_dp" : "")}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), row = b.R(), a = Param64(b, "p_a"), x = Param64(b, "p_x"), y = Param64(b, "p_y"), rows = Param32(b, "p_rows"), blocks = Param32(b, "p_blocks"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {row}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {row}, {rows};");
        b.L($"@{pEnd} bra DONE;");
        int bb = f.BlockBytes;
        string acc = b.F(), blk = b.R(), pB = b.P(), aAddr = b.RD(), xAddr = b.RD(), rowBase = b.RD();
        b.L($"mov.f32 {acc}, 0f00000000;");
        b.L($"mul.wide.u32 {rowBase}, {row}, {blocks};");
        b.L($"mul.lo.u64 {rowBase}, {rowBase}, {bb};");
        b.L($"add.u64 {rowBase}, {a}, {rowBase};");
        b.L($"mov.u32 {blk}, 0;");
        b.Label("BLK");
        b.L($"setp.ge.u32 {pB}, {blk}, {blocks};");
        b.L($"@{pB} bra BLK_END;");
        b.L($"mul.wide.u32 {aAddr}, {blk}, {bb};");
        b.L($"add.u64 {aAddr}, {rowBase}, {aAddr};");
        b.L($"mul.wide.u32 {xAddr}, {blk}, {bb};");
        b.L($"add.u64 {xAddr}, {x}, {xAddr};");
        string ha = b.H(), hx = b.H(), da = b.F(), dx = b.F(), dot = b.R();
        b.L($"ld.global.u16 {ha}, [{aAddr}];");
        b.L($"ld.global.u16 {hx}, [{xAddr}];");
        b.L($"cvt.f32.f16 {da}, {ha};");
        b.L($"cvt.f32.f16 {dx}, {hx};");
        b.L($"mov.u32 {dot}, 0;");
        // A block is 2-byte aligned only (34 or 18 bytes): a 4-byte group of its values is read as two halves.
        void Word(string w, string addr, int offset)
        {
            string lo = b.H(), hi = b.H();
            b.L($"ld.global.u16 {lo}, [{addr}+{offset}];");
            b.L($"ld.global.u16 {hi}, [{addr}+{offset + 2}];");
            b.L($"mov.b32 {w}, {{{lo}, {hi}}};");
        }

        if (f.Is("q8_0"))
        {
            // 32 int8 at +2: eight 4-byte groups.
            for (int g = 0; g < 8; g++)
            {
                string wa = b.R(), wx = b.R();
                Word(wa, aAddr, 2 + g * 4);
                Word(wx, xAddr, 2 + g * 4);
                if (dp4a) b.L($"dp4a.s32.s32 {dot}, {wa}, {wx}, {dot};");
                else
                {
                    for (int i = 0; i < 4; i++)
                    {
                        string qa = b.R(), qx = b.R();
                        b.L($"bfe.s32 {qa}, {wa}, {i * 8}, 8;");
                        b.L($"bfe.s32 {qx}, {wx}, {i * 8}, 8;");
                        b.L($"mad.lo.s32 {dot}, {qa}, {qx}, {dot};");
                    }
                }
            }
        }
        else
        {
            // 16 bytes of nibbles at +2: value i (low) and i + 16 (high), each minus 8.
            for (int g = 0; g < 4; g++)
            {
                string wa = b.R(), wx = b.R();
                Word(wa, aAddr, 2 + g * 4);
                Word(wx, xAddr, 2 + g * 4);
                for (int i = 0; i < 8; i++)
                {
                    string qa = b.R(), qx = b.R();
                    b.L($"bfe.u32 {qa}, {wa}, {i * 4}, 4;");
                    b.L($"bfe.u32 {qx}, {wx}, {i * 4}, 4;");
                    b.L($"sub.s32 {qa}, {qa}, 8;");
                    b.L($"sub.s32 {qx}, {qx}, 8;");
                    b.L($"mad.lo.s32 {dot}, {qa}, {qx}, {dot};");
                }
            }
        }

        string dotF = b.F(), scale = b.F();
        b.L($"cvt.rn.f32.s32 {dotF}, {dot};");
        b.L($"mul.rn.f32 {scale}, {da}, {dx};");
        b.L($"fma.rn.f32 {acc}, {scale}, {dotF}, {acc};");
        b.L($"add.u32 {blk}, {blk}, 1;");
        b.L("bra BLK;");
        b.Label("BLK_END");
        string yAddr = b.RD();
        b.L($"mul.wide.u32 {yAddr}, {row}, 4;");
        b.L($"add.u64 {yAddr}, {y}, {yAddr};");
        b.L($"st.global.f32 [{yAddr}], {acc};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_blocks");
    }
}

public sealed partial class GpuOps
{
    /// <summary>y (float32 lanes) = A·x for Q8_0 / Q4_0 operands with cols in whole blocks; returns the lane's name.</summary>
    public string GemvBlocks<TN>(ulong a, ulong x, ulong y, int rows, int cols) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!f.IsBlock || f.Name is not ("q8_0" or "q4_0") || cols % 32 != 0) throw new NotSupportedException($"{f.Name}: the block product takes Q8_0 or Q4_0 with columns in whole blocks");
        bool dp = f.Is("q8_0") && Dp4aSupported(true);
        var fn = gpu.Cached($"gemv_blocks:{f.Name}:{dp}", (out string e) => Ptx.GemvBlocks(gpu, f, dp, out e));
        Gpu.Launch(fn, Grid(rows, 256), 1, 256, 1, 0, a, x, y, Gpu.U(rows), Gpu.U(cols / 32));
        return dp ? "s8dp blocks" : "s32 blocks";
    }
}
