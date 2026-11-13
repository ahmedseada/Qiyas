// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The block-scaled product on the matrix units (CUDA's Cuda/MatMulBlocks.cs): C = A·Bᵀ for A (m × k) and B (n × k) both
// Q8_0 or both Q4_0, k in whole blocks of 32 along each row. C_ij = Σ_blocks dA·dB·(Σ_32 qA·qB): every 32-term dot an
// exact integer (one s8 unit product of k = 32 from zero), the block scales applied in f32, nothing decoded to a float per
// value. A workgroup of four subgroups holds 64 × 64 of C, a subgroup 32 × 32 (2 × 2 of the units' 16 × 16); per k-block
// A's and B's 64 rows' codes are staged into workgroup memory as int8 (Q4_0's nibbles less 8), rows 48 bytes apart, and
// their scales' f16 bits into scale tiles, read from memory 16 bits at a time (a 34- or 18-byte block is 2-byte aligned
// only). The scales' outer product dA_i·dB_j is itself a unit product (f16 → f32: column 0 of A's scale tile, A's rows'
// scales, against row 0 of B's, B's rows'; every other value 0: exact), and each tile's s32 result converted, times it,
// added to the f32 sums. C comes out as f32 lanes; the caller encodes it by blocks.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int BlocksTile = 64, BlocksWarps = 4;

    /// <summary>Whether the block product runs here: the s8 unit of 16 × 16 × 32 and the f16 → f32 unit of 16 × 16 × 16, 16-bit
    /// loads of memory, four subgroups a workgroup.</summary>
    public static bool BlocksRun(Vulkan vk) =>
        vk.Storage16 && vk.Float16 && vk.CooperativeMatrices.Contains(new CooperativeMatrix(16, 16, 32, "s8", "s8", "s32", "s32", false))
        && vk.CooperativeMatrices.Contains(new CooperativeMatrix(16, 16, 16, "f16", "f16", "f32", "f32", false)) && BlocksWarps * vk.SubgroupSize <= vk.MaxWorkgroup;

    /// <summary>C (f32 lanes, m × n) = A·Bᵀ for Q8_0 / Q4_0 rows of `blocks` blocks. Arguments: a, b, c, m, n, blocks;
    /// workgroups (⌈n/64⌉, ⌈m/64⌉).</summary>
    public static SpvKernel MatMulBlocks(Vulkan vk, Fmt f)
    {
        bool q8 = f.Is("q8_0");
        int sg = vk.SubgroupSize, threads = BlocksWarps * sg, bb = f.BlockBytes, tile = BlocksTile;
        const int rowWords = 12;                                                 // 32 codes, then 16 bytes of padding
        var b = new Spv($"matmul_blocks_{f.Name}", threads, vk.Controls);
        V pa = b.Arg(0), pb = b.Arg(1), pc = b.Arg(2), M = b.Arg(3, K.U32), N = b.Arg(4, K.U32), blocks = b.Arg(5, K.U32);
        var tid = b.LocalIdX;
        var warp = b.SubgroupId;
        var lane = tid - warp * (uint)sg;
        var m0 = b.GroupY * tile;
        var n0 = b.GroupX * tile;
        var codesA = b.Shared(K.U32, tile * rowWords);
        var codesB = b.Shared(K.U32, tile * rowWords);
        var scalesA = b.Shared(K.U16, 4 * 256);                                   // four 16 × 16 f16 tiles: column 0 the rows' scales
        var scalesB = b.Shared(K.U16, 4 * 256);                                   // four 16 × 16 f16 tiles (k × n): row 0 the columns' scales
        var scratch = b.Shared(K.U32, BlocksWarps * 256);                         // a subgroup's 16 × 16 of C past m or n
        var zero16 = b.Convert(b.U(0), K.U16);
        b.For(tid, b.U(2 * 4 * 256), b.U(threads), w => b.If(w < 4 * 256, () => scalesA[w] = zero16, () => scalesB[w - 4 * 256] = zero16));
        var rowBytes = blocks * (uint)bb;

        // An invocation's jobs a block: rows' halves of A, then of B (2·64 a side), the workgroup's invocations at a time.
        int perSide = (2 * tile + threads - 1) / threads;
        // A job's half of its row's codes as four int8 words (Q8_0: codes 16·half .. + 15; Q4_0: codes 8·half .. + 7 and
        // 16 + 8·half .. + 7, the nibbles less 8) and, on half 0, its scale's f16 bits: past the rows (or a job past the
        // side's) all 0. Read from block kb (kb past the last: the last's, never used).
        V[] Fetch(bool sideA, int it, V kb)
        {
            var job = tid + (uint)(it * threads);
            var r = job >> 1;
            var half = job & 1UL;
            var g = (sideA ? m0 : n0) + r;
            var inside = (job < 2 * tile) & (g < (sideA ? M : N));
            var at = (sideA ? pa : pb) + b.Convert(b.Select(inside, g, b.U(0)), K.U64) * b.Convert(rowBytes, K.U64) + b.Convert(b.Min(kb, blocks - 1) * (uint)bb, K.U64);
            V Half(V offset) => b.Convert(b.Load(K.U16, at + b.Convert(offset, K.U64)), K.U32);
            var got = new V[5];
            if (q8)
            {
                for (int w = 0; w < 4; w++)
                {
                    var byteAt = half * 16 + (uint)(2 + 4 * w);
                    got[w] = Half(byteAt) | Half(byteAt + 2) << 16;
                }
            }
            else
            {
                var bytes = new V[8];
                for (int h = 0; h < 4; h++)
                {
                    var pair = Half(half * 8 + (uint)(2 + 2 * h));
                    bytes[2 * h] = pair & 0xFFUL;
                    bytes[2 * h + 1] = pair >> 8;
                }

                V Code(V nibble) => (nibble - 8) & 0xFFUL;
                for (int w = 0; w < 2; w++)
                {
                    var low = b.U(0);
                    var high = b.U(0);
                    for (int q = 0; q < 4; q++)
                    {
                        low = low | Code(bytes[4 * w + q] & 0xFUL) << (8 * q);
                        high = high | Code(bytes[4 * w + q] >> 4) << (8 * q);
                    }

                    got[w] = low;
                    got[2 + w] = high;
                }
            }

            for (int w = 0; w < 4; w++) got[w] = b.Select(inside, got[w], b.U(0));
            got[4] = b.Select(inside, Half(b.U(0)), b.U(0));
            return got;
        }

        // The fetched job into the stage: its words at its row (Q8_0: words 4·half ..; Q4_0: 2·half .. and 4 + 2·half ..)
        // and, on half 0, its scale (A's: tile r / 16, row r % 16, column 0; B's: tile r / 16, row 0, column r % 16).
        void Put(bool sideA, int it, V[] got)
        {
            var job = tid + (uint)(it * threads);
            var r = job >> 1;
            var half = job & 1UL;
            var codes = sideA ? codesA : codesB;
            b.If(job < 2 * tile, () =>
            {
                var row = r * (uint)rowWords;
                for (int w = 0; w < 4; w++)
                {
                    var at = q8 ? row + half * 4 + (uint)w : row + (uint)(w < 2 ? 0 : 4) + half * 2 + (uint)(w & 1);
                    codes[at] = got[w];
                }

                b.If(half == 0, () =>
                {
                    if (sideA) scalesA[(r >> 4) * 256 + (r & 15UL) * 16] = b.Convert(got[4], K.U16);
                    else scalesB[(r >> 4) * 256 + (r & 15UL)] = b.Convert(got[4], K.U16);
                });
            });
        }

        var ta = b.Coop(K.S8, 16, 32, Use.A);
        var tb = b.Coop(K.S8, 32, 16, Use.B);
        var ti = b.Coop(K.S32, 16, 16, Use.Accumulator);
        var tf = b.Coop(K.F32, 16, 16, Use.Accumulator);
        var sa16 = b.Coop(K.F16, 16, 16, Use.A);
        var sb16 = b.Coop(K.F16, 16, 16, Use.B);
        var wr = warp >> 1;
        var wc = warp & 1UL;
        var acc = new MVar[2, 2];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) acc[i, j] = b.Local(b.Fill(tf, 0));
        var zeroI = b.Fill(ti, 0);
        var zeroF = b.Fill(tf, 0);

        // The jobs' fetched values carried from one block to the next: block kb + 1's read while kb's products run.
        var held = new Var[2, perSide, 5];
        for (int side = 0; side < 2; side++)
        {
            for (int it = 0; it < perSide; it++)
            {
                var got = Fetch(side == 0, it, b.U(0));
                for (int w = 0; w < 5; w++) held[side, it, w] = b.Local(got[w]);
            }
        }

        b.For(b.U(0), blocks, kb =>
        {
            b.Barrier();
            for (int side = 0; side < 2; side++)
                for (int it = 0; it < perSide; it++)
                    Put(side == 0, it, [.. Enumerable.Range(0, 5).Select(w => held[side, it, w].V)]);
            b.Barrier();
            var next = new V[2, perSide][];
            for (int side = 0; side < 2; side++)
                for (int it = 0; it < perSide; it++)
                    next[side, it] = Fetch(side == 0, it, kb + 1);
            for (int i = 0; i < 2; i++)
            {
                var rowTile = wr * 2 + (uint)i;                                  // A's 16-row block among the four
                var af = b.Load(ta, codesA, rowTile * (16 * rowWords), b.U(rowWords));
                var sf = b.Load(sa16, scalesA, rowTile * 256, b.U(16));
                for (int j = 0; j < 2; j++)
                {
                    var colTile = wc * 2 + (uint)j;
                    var bf = b.Load(tb, codesB, colTile * (16 * rowWords), b.U(rowWords), columnMajor: true);
                    var dots = b.MulAdd(af, bf, zeroI);
                    var scales = b.MulAdd(sf, b.Load(sb16, scalesB, colTile * 256, b.U(16)), zeroF);
                    acc[i, j].M = b.Add(acc[i, j].M, b.Mul(b.Convert(dots, tf), scales));
                }
            }

            for (int side = 0; side < 2; side++)
                for (int it = 0; it < perSide; it++)
                    for (int w = 0; w < 5; w++) held[side, it, w].V = next[side, it][w];
        });

        // C: a whole 16 × 16 by the units; one past m or n through the subgroup's scratch, each value inside alone.
        var region = warp * 256;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                var r0 = m0 + (wr * 2 + (uint)i) * 16;
                var c0 = n0 + (wc * 2 + (uint)j) * 16;
                var m = acc[i, j].M;
                var at = pc + (b.Convert(r0, K.U64) * b.Convert(N, K.U64) + b.Convert(c0, K.U64)) * 4;
                b.If((r0 + 16 <= M) & (c0 + 16 <= N), () => b.Store(m, K.F32, at, N, false, 4), () =>
                {
                    b.Store(m, scratch, region, b.U(16));
                    b.SubgroupBarrier();
                    for (int e = 0; e * sg < 256; e++)
                    {
                        var idx = lane + (uint)(e * sg);
                        var rr = idx >> 4;
                        var cc = idx & 15UL;
                        b.If((r0 + rr < M) & (c0 + cc < N), () => b.Store(at + (b.Convert(rr, K.U64) * b.Convert(N, K.U64) + b.Convert(cc, K.U64)) * 4, scratch[region + idx]));
                    }

                    b.SubgroupBarrier();
                });
            }
        }

        return b.Build();
    }
}
