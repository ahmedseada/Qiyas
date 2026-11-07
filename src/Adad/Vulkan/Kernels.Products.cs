// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The products on Vulkan: C = op(A)·op(B) over a view's batches by workgroup tiles. A workgroup of 16 × 16 invocations
// holds a 64 × 64 tile of C, each invocation 4 × 4 of it; A's and B's 64 × 16 slices are read where they are stored
// (Codec.UnpackAt: transposed or at a batch's offset by the index alone) into workgroup memory in the product's lane,
// and each value of C accumulates its k terms in order by fused multiply-adds (the integer lanes exactly). C is the
// lane's values: written as C where the lane is the type's storage, otherwise into lane storage and stored as the type
// once (FromLanesKeep, which keeps the values between the batches' regions).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int TileM = 64, TileN = 64, TileK = 16, TileSide = 16;

    /// <summary>The product kernel, A and B of one format. Arguments: a, b, c (lane values at C's positions), m, n, k,
    /// strideA, strideB, strideC (values), tilesM, offsetA, offsetB (values: A's and B's first); workgroups (⌈n/64⌉,
    /// ⌈m/64⌉ · batch).</summary>
    public static SpvKernel MatMul(Vulkan vk, Fmt f, Lane lane, bool transA, bool transB) => MatMul(vk, f, f, lane, transA, transB);

    /// <summary>The product kernel with A stored as fa and B as fb (attention's P in lane storage against V as stored).</summary>
    public static SpvKernel MatMul(Vulkan vk, Fmt fa, Fmt fb, Lane lane, bool transA, bool transB)
    {
        var k = Kinds.Of(lane);
        string formats = fa.Name == fb.Name ? fa.Name : $"{fa.Name}_{fb.Name}";
        var b = new Spv($"matmul_{formats}_{lane.Name}{(transA ? "_ta" : "")}{(transB ? "_tb" : "")}", TileSide, vk.Controls, TileSide);
        var pa = b.Arg(0);
        var pb = b.Arg(1);
        var pc = b.Arg(2);
        var M = b.Arg(3, K.U32);
        var N = b.Arg(4, K.U32);
        var Kd = b.Arg(5, K.U32);
        var strideA = b.Arg(6, K.U32);
        var strideB = b.Arg(7, K.U32);
        var strideC = b.Arg(8, K.U32);
        var tilesM = b.Arg(9, K.U32);
        var offsetA = b.Arg(10, K.U32);
        var offsetB = b.Arg(11, K.U32);
        var tx = b.LocalIdX;
        var ty = b.LocalIdY;
        var tid = ty * TileSide + tx;
        var gy = b.GroupY;
        var batch = gy / tilesM;
        var row0 = (gy - batch * tilesM) * TileM;
        var col0 = b.GroupX * TileN;
        var baseA = batch * strideA + offsetA;
        var baseB = batch * strideB + offsetB;
        var baseC = batch * strideC;
        var sa = b.Shared(k, TileK * TileM);                                         // sa[kk·64 + i] = A(row0 + i, k0 + kk)
        var sb = b.Shared(k, TileK * TileN);                                         // sb[kk·64 + j] = B(k0 + kk, col0 + j)
        var acc = new Var[4, 4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i, j] = b.Local(b.C(k, 0));
        var zero = b.C(k, 0);
        int threads = TileSide * TileSide;
        b.For(b.U(0), Kd, b.U(TileK), k0 =>
        {
            for (int l = 0; l < TileK * TileM / threads; l++)
            {
                // Consecutive invocations along the operand's stored rows (k for A, i for Aᵀ).
                var e = tid + l * threads;
                var ii = transA ? e & (TileM - 1) : e >> 4;
                var kk = transA ? e >> 6 : e & (TileK - 1);
                var gr = row0 + ii;
                var gk = k0 + kk;
                var inside = (gr < M) & (gk < Kd);
                var at = transA ? gk * M + gr : gr * Kd + gk;
                sa[kk * TileM + ii] = b.Select(inside, Codec.UnpackAt(b, fa, pa, baseA + b.Select(inside, at, b.U(0)), lane), zero);
            }

            for (int l = 0; l < TileK * TileN / threads; l++)
            {
                var e = tid + l * threads;
                var kk = transB ? e & (TileK - 1) : e >> 6;
                var jj = transB ? e >> 4 : e & (TileN - 1);
                var gk = k0 + kk;
                var gc = col0 + jj;
                var inside = (gk < Kd) & (gc < N);
                var at = transB ? gc * Kd + gk : gk * N + gc;
                sb[kk * TileN + jj] = b.Select(inside, Codec.UnpackAt(b, fb, pb, baseB + b.Select(inside, at, b.U(0)), lane), zero);
            }

            b.Barrier();
            for (int q = 0; q < TileK; q++)
            {
                var av = new V[4];
                var bv = new V[4];
                for (int i = 0; i < 4; i++) av[i] = sa[ty + (q * TileM + i * TileSide)];
                for (int j = 0; j < 4; j++) bv[j] = sb[tx + (q * TileN + j * TileSide)];
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i, j].V = b.Fma(av[i], bv[j], acc[i, j].V);
            }

            b.Barrier();
        });

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                var r = row0 + ty + i * TileSide;
                var c = col0 + tx + j * TileSide;
                var value = acc[i, j].V;
                b.If((r < M) & (c < N), () => b.Store(b.At(pc, baseC + r * N + c, lane.Size), value));
            }
        }

        return b.Build();
    }

    /// <summary>A lane array's n values into a stored array a thread per word, the values at positions p with p mod period
    /// ≥ span (between a batch's region and the next) kept as they are stored. Arguments: lanes, stored, n, period, span.</summary>
    public static SpvKernel FromLanesKeep(Vulkan vk, Lane lane, Fmt f)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"back_keep_{lane.Name}_{f.Name}", Threads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var period = b.Arg(3, K.U32);
        var span = b.Arg(4, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var address = b.At(dst, gid, f.WordBits / 8);
            var old = b.Load(wk, address);
            var word = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var idx = gid * (double)f.Per + i;
                var fresh = idx < n & idx % period < span;
                var v = b.Local(b.C(k, 0));
                b.If(fresh, () => v.V = b.Load(k, b.At(src, idx, lane.Size)));
                var packed = Codec.Pack(b, f, v.V, lane, i, b.CBits(wk, 0));
                ulong mask = f.WordBits == 64 ? ulong.MaxValue : ((1UL << f.Bits) - 1) << (i * f.Bits);
                word = word | b.Select(fresh, packed, old & mask);
            }

            b.Store(address, word);
        });
        return b.Build();
    }
}
