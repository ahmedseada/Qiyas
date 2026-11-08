// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts' kernels on Vulkan (VulkanOps.Experts.cs; Experts.cs), CUDA's (Cuda/Ptx.Experts.cs): the routing's
// ranks, picks and rows; the experts' tiles and each row's expert; the grouped product (every expert's tiles in one
// dispatch, its rows and weights from the offsets read where they are; the product's tiles, Kernels.Products.cs); and the
// store with each row's expert's bias and the gate (Map's formula, Kernels.Map.cs), rounded to the type once.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>flags[t·E + e] = 1 where expert e's rank among token t's probabilities is below k. Slots: p, flags, count
    /// (tokens·E), experts, k.</summary>
    public static SpvKernel ExpertRanks(Vulkan vk, Fmt f, Lane lane)
    {
        var b = new Spv($"expert_ranks_{f.Name}_{lane.Name}", Threads, vk.Controls);
        V pp = b.Arg(0), pf = b.Arg(1);
        V count = b.Arg(2, K.U32), E = b.Arg(3, K.U32), k = b.Arg(4, K.U32);
        var i = b.GlobalX;
        b.If(i < count, () =>
        {
            var t = i / E;
            var row = t * E;
            var e = i - row;
            var pe = Codec.UnpackAt(b, f, pp, row + e, lane);
            var rank = b.Local(b.U(0));
            b.For(b.U(0), E, q =>
            {
                var pq = Codec.UnpackAt(b, f, pp, row + q, lane);
                var above = (pq > pe) | ((pq == pe) & (q < e));
                rank.V = rank.V + b.Select(above, b.U(1), b.U(0));
            });
            b.Store(b.At(pf, i, 4), b.Select(rank.V < k, b.U(1), b.U(0)));
        });
        return b.Build();
    }

    /// <summary>An invocation per token: the flagged experts in increasing order at its slots (chosen) and their
    /// probabilities (lane storage), normalized (norm 1) by their sum in that order. Slots: p, flags, chosen, lanes, tokens,
    /// experts, k, norm.</summary>
    public static SpvKernel ExpertPicks(Vulkan vk, Fmt f, Lane lane)
    {
        var kk = Kinds.Of(lane);
        var b = new Spv($"expert_picks_{f.Name}_{lane.Name}", Threads, vk.Controls);
        V pp = b.Arg(0), pf = b.Arg(1), pc = b.Arg(2), pl = b.Arg(3);
        V tokens = b.Arg(4, K.U32), E = b.Arg(5, K.U32), k = b.Arg(6, K.U32), norm = b.Arg(7, K.U32);
        var t = b.GlobalX;
        b.If(t < tokens, () =>
        {
            var row = t * E;
            var slot0 = t * k;
            var s = b.Local(b.U(0));
            var sum = b.Local(b.C(kk, 0));
            b.For(b.U(0), E, e =>
            {
                var at = row + e;
                b.If(b.Load(K.U32, b.At(pf, at, 4)) != 0.0, () =>
                {
                    var slot = slot0 + s.V;
                    b.Store(b.At(pc, slot, 4), e);
                    var v = Codec.UnpackAt(b, f, pp, at, lane);
                    b.Store(b.At(pl, slot, lane.Size), v);
                    sum.V = sum.V + v;
                    s.V = s.V + 1.0;
                });
            });
            b.If(norm != 0.0, () => b.For(b.U(0), k, i =>
            {
                var a = b.At(pl, slot0 + i, lane.Size);
                b.Store(a, b.Load(kk, a) / sum.V);
            }));
        });
        return b.Build();
    }

    /// <summary>An invocation per row: slot = order[row], tokenOfRow[row] = slot / k, slotRow[slot] = row. Slots: order,
    /// tokenOfRow, slotRow, slots, k.</summary>
    public static SpvKernel ExpertRows(Vulkan vk)
    {
        var b = new Spv("expert_rows", Threads, vk.Controls);
        V po = b.Arg(0), pt = b.Arg(1), ps = b.Arg(2);
        V slots = b.Arg(3, K.U32), k = b.Arg(4, K.U32);
        var r = b.GlobalX;
        b.If(r < slots, () =>
        {
            var slot = b.Load(K.U32, b.At(po, r, 4));
            b.Store(b.At(pt, r, 4), slot / k);
            b.Store(b.At(ps, slot, 4), r);
        });
        return b.Build();
    }

    /// <summary>One invocation: tiles[0] = 0, tiles[e + 1] = tiles[e] + ⌈max(offsets[e + 1] − offsets[e], 0) / tile⌉. Slots:
    /// offsets, tiles, experts, tile.</summary>
    public static SpvKernel ExpertTiles(Vulkan vk)
    {
        var b = new Spv("expert_tiles", 1, vk.Controls);
        V po = b.Arg(0), pt = b.Arg(1);
        V E = b.Arg(2, K.U32), tile = b.Arg(3, K.U32);
        var acc = b.Local(b.U(0));
        b.Store(b.At(pt, b.U(0), 4), b.U(0));
        b.For(b.U(0), E, e =>
        {
            var lo = b.Load(K.U32, b.At(po, e, 4));
            var hi = b.Load(K.U32, b.At(po, e + 1.0, 4));
            var c = b.Select(hi > lo, hi - lo, b.U(0));
            acc.V = acc.V + (c + tile - 1.0) / tile;
            b.Store(b.At(pt, e + 1.0, 4), acc.V);
        });
        return b.Build();
    }

    /// <summary>An invocation per row: rowExpert[r] = the last expert e with offsets[e] ≤ r, or experts where r is outside
    /// [offsets[0], offsets[experts]). Slots: offsets, rowExpert, rows, experts.</summary>
    public static SpvKernel ExpertOfRows(Vulkan vk)
    {
        var b = new Spv("expert_of_rows", Threads, vk.Controls);
        V po = b.Arg(0), pr = b.Arg(1);
        V rows = b.Arg(2, K.U32), E = b.Arg(3, K.U32);
        var r = b.GlobalX;
        b.If(r < rows, () =>
        {
            var e = b.Local(b.U(0));
            b.For(b.U(1), E, j => e.V = b.Select(b.Load(K.U32, b.At(po, j, 4)) <= r, j, e.V));
            var inside = (r >= b.Load(K.U32, b.At(po, b.U(0), 4))) & (r < b.Load(K.U32, b.At(po, E, 4)));
            b.Store(b.At(pr, r, 4), b.Select(inside, e.V, E));
        });
        return b.Build();
    }

    /// <summary>The grouped product: every expert's rows of x (stored as f) times its weights (f; K × N, or N × K with
    /// transB, expert e's from e·K·N) into lane values, the workgroup's y over the experts' tiles (tiles: their prefix) and
    /// past them. Slots: x, w, c, n, k, offsets, tiles, experts; workgroups (⌈n/64⌉, ⌈rows/64⌉ + experts).</summary>
    public static SpvKernel MatMulGrouped(Vulkan vk, Fmt f, Lane lane, bool transB)
    {
        var b = new Spv($"matmul_grouped_{f.Name}_{lane.Name}{(transB ? "_tb" : "")}", TileSide, vk.Controls, TileSide);
        V pa = b.Arg(0), pw = b.Arg(1), pc = b.Arg(2);
        V N = b.Arg(3, K.U32), Kd = b.Arg(4, K.U32);
        V po = b.Arg(5), pt = b.Arg(6);
        V E = b.Arg(7, K.U32);
        var gy = b.GroupY;
        var ex = b.Local(b.U(0));
        b.For(b.U(1), E, j => ex.V = b.Select(b.Load(K.U32, b.At(pt, j, 4)) <= gy, j, ex.V));
        b.If(gy < b.Load(K.U32, b.At(pt, E, 4)), () =>
        {
            var e = ex.V;
            var row0 = b.Load(K.U32, b.At(po, e, 4)) + (gy - b.Load(K.U32, b.At(pt, e, 4))) * (double)TileM;
            var end = b.Load(K.U32, b.At(po, e + 1.0, 4));
            var baseB = e * (Kd * N);
            ProductTiles(b, lane, false, transB, end, N, Kd, row0, pc, b.U(0),
                (gr, gk, inside) => Codec.UnpackAt(b, f, pa, b.Select(inside, gr * Kd + gk, b.U(0)), lane),
                (gk, gc, inside) => Codec.UnpackAt(b, f, pw, baseB + b.Select(inside, transB ? gc * Kd + gk : gk * N + gc, b.U(0)), lane));
        });
        return b.Build();
    }

    private readonly record struct ExpertPointers(V G, V GB, V U, V UB, V RowExpert, V Cols, V Experts);

    /// <summary>y = gate(g + gb[e], u + ub[e]) (no gate: g + gb[e]) of the lane sums, e the row's expert (0 outside every
    /// expert's rows). Slots: g, gb, u, ub, rowExpert, y, n, cols, experts.</summary>
    public static SpvKernel ExpertStore(Vulkan vk, Fmt f, Lane lane, string? gate, bool gateBias, bool upBias)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"expert_store_{f.Name}_{lane.Name}_{gate ?? "none"}{(gateBias ? "_gb" : "")}{(upBias ? "_ub" : "")}", Threads, vk.Controls);
        V pg = b.Arg(0), pgb = b.Arg(1), pu = b.Arg(2), pub = b.Arg(3), pre = b.Arg(4), py = b.Arg(5);
        V n = b.Arg(6, K.U32), cols = b.Arg(7, K.U32), E = b.Arg(8, K.U32);
        StoreEach(b, f, lane, py, n, at => ExpertValue(b, f, lane, k, gate, gateBias, upBias, new ExpertPointers(pg, pgb, pu, pub, pre, cols, E), at));
        return b.Build();
    }

    /// <summary>The store into blocks (f32). Slots: g, gb, u, ub, rowExpert, cols, experts.</summary>
    public static SpvKernel BlockExpertStore(Vulkan vk, Fmt f, string? gate, bool gateBias, bool upBias) =>
        BlockEncoder(vk, f, $"expert_store_{gate ?? "none"}{(gateBias ? "_gb" : "")}{(upBias ? "_ub" : "")}", b =>
        {
            V pg = b.Arg(0), pgb = b.Arg(1), pu = b.Arg(2), pub = b.Arg(3), pre = b.Arg(4);
            V cols = b.Arg(5, K.U32), E = b.Arg(6, K.U32);
            return (at, _) => ExpertValue(b, f, Lane.F32, K.F32, gate, gateBias, upBias, new ExpertPointers(pg, pgb, pu, pub, pre, cols, E), at);
        });

    private static V ExpertValue(Spv b, Fmt f, Lane lane, K k, string? gate, bool gateBias, bool upBias, ExpertPointers p, V at)
    {
        var r = at / p.Cols;
        var c = at - r * p.Cols;
        var e = b.Load(K.U32, b.At(p.RowExpert, r, 4));
        var inside = e < p.Experts;
        var bi = b.Select(inside, e, b.U(0)) * p.Cols + c;
        var g = b.Load(k, b.At(p.G, at, lane.Size));
        if (gateBias) g = g + Codec.UnpackAt(b, f, p.GB, bi, lane);
        var v = g;
        if (gate is not null)
        {
            var u = b.Load(k, b.At(p.U, at, lane.Size));
            if (upBias) u = u + Codec.UnpackAt(b, f, p.UB, bi, lane);
            v = Formula(b, k, gate, g, u, u, b.C(k, 0), b.C(k, 0));
        }

        return b.Select(inside, v, b.C(k, 0));
    }
}
