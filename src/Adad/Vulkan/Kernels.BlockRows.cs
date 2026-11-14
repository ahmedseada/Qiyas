// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The block formats' plain product on the matrix units: C = A·B for A (m × k) and B (k × n) both Q8_0 or both Q4_0, k
// and n in whole blocks of 32 along the rows. A's blocks run along k, so a k-block of A's row is its codes times one
// scale; B's run along n, so B's scale changes from one k to the next and no integer block dot exists (the block
// product, Kernels.BlockProducts.cs, takes A·Bᵀ). Here every value of B, d·q (an f16 times a code of at most 7 bits: at
// most 18 significant bits), is two f16 slices exactly, hi = d·q rounded and lo = d·q − hi (its last bit at least
// 2^-24, as d's: an f16 value; |d·q| ≤ 65504, the caller's bound, so hi is finite), and A's codes are f16 exactly. Per
// k-block P = codes·(hi + lo) by four f16 → f32 unit products (two halves of k, two slices), each term exact, and C +=
// P ⊙ (dA_i) with the row scales' matrix by a unit product of A's scales against ones (exact). A workgroup of four
// subgroups holds 64 × 64 of C, a subgroup 32 × 32; the next block's values are read while the current one's products
// run. C comes out as f32 lanes; the caller encodes it by blocks.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Whether the block rows' product runs here: the f16 → f32 unit of 16 × 16 × 16, 16-bit loads, f16 arithmetic,
    /// four subgroups a workgroup.</summary>
    public static bool BlockRowsRun(Vulkan vk) =>
        vk.Storage16 && vk.Float16 && vk.CooperativeMatrices.Contains(new CooperativeMatrix(16, 16, 16, "f16", "f16", "f32", "f32", false))
        && BlocksWarps * vk.SubgroupSize <= vk.MaxWorkgroup;

    /// <summary>C (f32 lanes, m × n) = A·B for Q8_0 / Q4_0 with k and n whole blocks (|B| ≤ 65504). Arguments: a, b, c, m, n,
    /// blocks (k / 32); workgroups (⌈n/64⌉, ⌈m/64⌉).</summary>
    public static SpvKernel MatMulBlockRows(Vulkan vk, Fmt f)
    {
        bool q8 = f.Is("q8_0");
        int sg = vk.SubgroupSize, threads = BlocksWarps * sg, bb = f.BlockBytes, tile = BlocksTile;
        const int aVecs = 5;                                                     // A's row: 32 codes as f16 (4 vectors), 1 of padding
        const int bVecs = 9;                                                     // B's k row: 64 values as f16 (8 vectors), 1 of padding
        var b = new Spv($"matmul_block_rows_{f.Name}", threads, vk.Controls);
        V pa = b.Arg(0), pb = b.Arg(1), pc = b.Arg(2), M = b.Arg(3, K.U32), N = b.Arg(4, K.U32), blocks = b.Arg(5, K.U32);
        var tid = b.LocalIdX;
        var warp = b.SubgroupId;
        var lane = tid - warp * (uint)sg;
        var m0 = b.GroupY * tile;
        var n0 = b.GroupX * tile;
        // Vectors of 4 words: A's codes (64 rows), B's hi and lo slices (32 k rows each); after the products the first 4·64
        // vectors are the subgroups' scratch for C's tiles past m or n. The scale tiles: A's rows (2 vectors a row: the
        // scale, then zeros) and the ones' tile (column-major: row 0 all ones).
        int hiOffset = tile * aVecs, loOffset = hiOffset + 32 * bVecs;
        var stage = b.SharedVectors(K.U32, 4, loOffset + 32 * bVecs);
        var scales = b.SharedVectors(K.U32, 4, tile * 2);
        var ones = b.SharedVectors(K.U32, 4, 32);
        var z = b.U(0);
        b.If(tid < 32, () => ones.Store(tid, b.Select((tid & 1UL) == 0, b.U(0x3C00), z), z, z, z));      // column c: (1, 0, ..., 0)
        var rowBytesA = blocks * (uint)bb;
        var rowBytesB = (N >> 5) * (uint)bb;

        // A job of A's: a row's half (codes 16·half .. + 15 as f16 pairs: 8 words) and on half 0 its scale's bits; a job of
        // B's: a k row's quarter of the tile's 64 columns (16 values of one block: hi and lo slices, 8 words each).
        int jobsA = 2 * tile, jobsB = 32 * 4;
        int perA = (jobsA + threads - 1) / threads, perB = (jobsB + threads - 1) / threads;
        V Half(V at, V offset) => b.Convert(b.Load(K.U16, at + b.Convert(offset, K.U64)), K.U32);
        V H(V f32) => b.Convert(b.Bitcast(b.Convert(f32, K.F16), K.U16), K.U32);

        // The 16 codes of a half (Q8_0: bytes 2 + 16·half ..; Q4_0: the 16 bytes' low nibbles (half 0) or high, less 8) as
        // signed integers.
        V[] Codes(V at, V half)
        {
            var q = new V[16];
            if (q8)
            {
                for (int p = 0; p < 8; p++)
                {
                    var pair = Half(at, half * 16 + (uint)(2 + 2 * p));
                    q[2 * p] = b.Bitcast(pair << 24, K.S32) >> 24;
                    q[2 * p + 1] = b.Bitcast(pair << 16, K.S32) >> 24;
                }
            }
            else
            {
                var shift = half * 4;
                for (int p = 0; p < 8; p++)
                {
                    var pair = Half(at, b.U((uint)(2 + 2 * p)));
                    q[2 * p] = b.Bitcast((pair & 0xFFUL).Shr(shift) & 0xFUL, K.S32) - 8;
                    q[2 * p + 1] = b.Bitcast((pair >> 8).Shr(shift) & 0xFUL, K.S32) - 8;
                }
            }

            return q;
        }

        // A's job it: 8 words of f16 codes and the scale (0 past the rows); from block kb (past the last: the last's).
        V[] FetchA(int it, V kb)
        {
            var job = tid + (uint)(it * threads);
            var r = job >> 1;
            var half = job & 1UL;
            var g = m0 + r;
            var inside = (job < jobsA) & (g < M);
            var at = pa + b.Convert(b.Select(inside, g, z), K.U64) * b.Convert(rowBytesA, K.U64) + b.Convert(b.Min(kb, blocks - 1) * (uint)bb, K.U64);
            var q = Codes(at, half);
            var got = new V[9];
            for (int w = 0; w < 8; w++)
                got[w] = b.Select(inside, H(b.Convert(q[2 * w], K.F32)) | H(b.Convert(q[2 * w + 1], K.F32)) << 16, z);
            got[8] = b.Select(inside, Half(at, z), z);
            return got;
        }

        // B's job it: k row k0 + job / 4, columns n0 + 16·(job % 4) .. + 15: the values d·q as hi (8 words) and lo slices.
        V[] FetchB(int it, V kb)
        {
            var job = tid + (uint)(it * threads);
            var kr = job >> 2;
            var quarter = job & 3UL;
            var col = n0 + quarter * 16;
            var gk = b.Min(kb, blocks - 1) * 32 + kr;
            var inside = (job < jobsB) & (col < N);
            var block = b.Select(inside, col >> 5, z);
            var at = pb + b.Convert(gk, K.U64) * b.Convert(rowBytesB, K.U64) + b.Convert(block * (uint)bb, K.U64);
            var d = b.Convert(b.Bitcast(b.Convert(Half(at, z), K.U16), K.F16), K.F32);
            var q = Codes(at, (quarter & 1UL));
            var got = new V[16];
            for (int w = 0; w < 8; w++)
            {
                var v0 = d * b.Convert(q[2 * w], K.F32);
                var v1 = d * b.Convert(q[2 * w + 1], K.F32);
                var h0 = H(v0);
                var h1 = H(v1);
                var l0 = H(v0 - b.Convert(b.Bitcast(b.Convert(h0, K.U16), K.F16), K.F32));
                var l1 = H(v1 - b.Convert(b.Bitcast(b.Convert(h1, K.U16), K.F16), K.F32));
                got[w] = b.Select(inside, h0 | h1 << 16, z);
                got[8 + w] = b.Select(inside, l0 | l1 << 16, z);
            }

            return got;
        }

        void PutA(int it, V[] got)
        {
            var job = tid + (uint)(it * threads);
            var r = job >> 1;
            var half = job & 1UL;
            b.If(job < jobsA, () =>
            {
                var at = r * (uint)aVecs + half * 2;
                stage.Store(at, got[0], got[1], got[2], got[3]);
                stage.Store(at + 1, got[4], got[5], got[6], got[7]);
                scales.Store(r * 2 + half, b.Select(half == 0, got[8], z), z, z, z);
            });
        }

        void PutB(int it, V[] got)
        {
            var job = tid + (uint)(it * threads);
            var kr = job >> 2;
            var quarter = job & 3UL;
            b.If(job < jobsB, () =>
            {
                var at = kr * (uint)bVecs + quarter * 2;
                stage.Store(at + (uint)hiOffset, got[0], got[1], got[2], got[3]);
                stage.Store(at + (uint)(hiOffset + 1), got[4], got[5], got[6], got[7]);
                stage.Store(at + (uint)loOffset, got[8], got[9], got[10], got[11]);
                stage.Store(at + (uint)(loOffset + 1), got[12], got[13], got[14], got[15]);
            });
        }

        var ta = b.Coop(K.F16, 16, 16, Use.A);
        var tb = b.Coop(K.F16, 16, 16, Use.B);
        var tf = b.Coop(K.F32, 16, 16, Use.Accumulator);
        var wr = warp >> 1;
        var wc = warp & 1UL;
        var acc = new MVar[2, 2];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) acc[i, j] = b.Local(b.Fill(tf, 0));
        var zeroF = b.Fill(tf, 0);

        var heldA = new Var[perA, 9];
        var heldB = new Var[perB, 16];
        for (int it = 0; it < perA; it++)
        {
            var got = FetchA(it, z);
            for (int w = 0; w < 9; w++) heldA[it, w] = b.Local(got[w]);
        }

        for (int it = 0; it < perB; it++)
        {
            var got = FetchB(it, z);
            for (int w = 0; w < 16; w++) heldB[it, w] = b.Local(got[w]);
        }

        b.Barrier();                                                             // the ones' tile written
        var onesB = b.Load(tb, ones, b.U(0), b.U(2), columnMajor: true);
        b.For(z, blocks, kb =>
        {
            b.Barrier();
            for (int it = 0; it < perA; it++) PutA(it, [.. Enumerable.Range(0, 9).Select(w => heldA[it, w].V)]);
            for (int it = 0; it < perB; it++) PutB(it, [.. Enumerable.Range(0, 16).Select(w => heldB[it, w].V)]);
            b.Barrier();
            var nextA = new V[perA][];
            var nextB = new V[perB][];
            for (int it = 0; it < perA; it++) nextA[it] = FetchA(it, kb + 1);
            for (int it = 0; it < perB; it++) nextB[it] = FetchB(it, kb + 1);
            for (int i = 0; i < 2; i++)
            {
                var rowTile = wr * 2 + (uint)i;
                var a0 = b.Load(ta, stage, rowTile * (16 * aVecs), b.U(aVecs));             // k 0 .. 15
                var a1 = b.Load(ta, stage, rowTile * (16 * aVecs) + 2.0, b.U(aVecs));       // k 16 .. 31
                var s = b.MulAdd(b.Load(ta, scales, rowTile * 32, b.U(2)), onesB, zeroF);  // s_ij = dA_i
                for (int j = 0; j < 2; j++)
                {
                    var colTile = wc * 2 + (uint)j;
                    V B0(int offset) => colTile * 2 + (double)offset;
                    var p = b.MulAdd(a0, b.Load(tb, stage, B0(hiOffset), b.U(bVecs)), zeroF);
                    p = b.MulAdd(a1, b.Load(tb, stage, B0(hiOffset + 16 * bVecs), b.U(bVecs)), p);
                    p = b.MulAdd(a0, b.Load(tb, stage, B0(loOffset), b.U(bVecs)), p);
                    p = b.MulAdd(a1, b.Load(tb, stage, B0(loOffset + 16 * bVecs), b.U(bVecs)), p);
                    acc[i, j].M = b.Add(acc[i, j].M, b.Mul(p, s));
                }
            }

            for (int it = 0; it < perA; it++) for (int w = 0; w < 9; w++) heldA[it, w].V = nextA[it][w];
            for (int it = 0; it < perB; it++) for (int w = 0; w < 16; w++) heldB[it, w].V = nextB[it][w];
        });

        // C: a whole 16 × 16 by the units; one past m or n through the subgroup's scratch (the stage's vectors, every
        // subgroup's products done), each value inside alone.
        b.Barrier();
        var region = warp * 64;
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
                    b.Store(m, stage, region, b.U(4));
                    b.SubgroupBarrier();
                    for (int e = 0; e * sg < 256; e++)
                    {
                        var idx = lane + (uint)(e * sg);
                        var rr = idx >> 4;
                        var cc = idx & 15UL;
                        b.If((r0 + rr < M) & (c0 + cc < N), () =>
                        {
                            var vec = stage.Load(region + (idx >> 2));
                            var w = vec[0];
                            for (int q = 1; q < 4; q++) w = b.Select((idx & 3UL) == (ulong)q, vec[q], w);
                            b.Store(at + (b.Convert(rr, K.U64) * b.Convert(N, K.U64) + b.Convert(cc, K.U64)) * 4, w);
                        });
                    }

                    b.SubgroupBarrier();
                });
            }
        }

        return b.Build();
    }
}
