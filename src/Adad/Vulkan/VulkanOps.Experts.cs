// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts on Vulkan (Experts.cs; its kernels Kernels.Experts.cs), CUDA's (Cuda/GpuOps.Experts.cs): the
// routing by the ranks and picks kernels, the route of the slots (the device's ordering) and the rows; the experts'
// products by the grouped product (every expert's tiles in one dispatch, the offsets read where they are), into the float
// lane's sums, then the store with each row's expert's bias and the gate.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    public string RouteTopK<TN>(ulong p, int tokens, int experts, int k, bool normalize, ulong weights, ulong offsets, ulong tokenOfRow, ulong slotRow) where TN : struct, IFloatAdad<TN>
    {
        Experts.ValidateRoute(tokens, experts, k);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        int slots = tokens * k;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong q = vk.Rent(Math.Max(16, bytes)); rented.Add(q); return q; }
        try
        {
            ulong ranked = Rent((long)tokens * experts * 4), chosen = Rent(slots * 4L), lanes = Rent(slots * (long)lane.Size), counts = Rent(experts * 4L), order = Rent(slots * 4L);
            if (tokens > 0)
            {
                var ranks = vk.Cached($"expert_ranks:{f.Name}:{lane.Name}", () => VkKernels.ExpertRanks(vk, f, lane));
                vk.Launch(ranks, Groups((long)tokens * experts, VkKernels.Threads), 1, p, ranked, Vulkan.U(tokens * experts), Vulkan.U(experts), Vulkan.U(k));
                var picks = vk.Cached($"expert_picks:{f.Name}:{lane.Name}", () => VkKernels.ExpertPicks(vk, f, lane));
                vk.Launch(picks, Groups(tokens, VkKernels.Threads), 1, p, ranked, chosen, lanes, Vulkan.U(tokens), Vulkan.U(experts), Vulkan.U(k), Vulkan.U(normalize ? 1 : 0));
                Back(f, lane, lanes, weights, slots);
            }

            Order(chosen, slots, experts, counts, offsets, order);
            if (slots > 0)
            {
                var rows = vk.Cached("expert_rows", () => VkKernels.ExpertRows(vk));
                vk.Launch(rows, Groups(slots, VkKernels.Threads), 1, order, tokenOfRow, slotRow, Vulkan.U(slots), Vulkan.U(k));
            }
        }
        finally { foreach (var q in rented) vk.Return(q); }

        return lane.Name;
    }

    public string ExpertMatMul<TN>(ExpertView v, ulong x, ulong offsets, ulong weights, ulong bias, ulong y) where TN : struct, IFloatAdad<TN> =>
        ExpertRun<TN>(v, null, x, offsets, weights, bias, 0, 0, y);

    public string ExpertGatedPair<TN>(ExpertView v, string gate, ulong x, ulong offsets, ulong gateWeights, ulong gateBias, ulong upWeights, ulong upBias, ulong hidden)
        where TN : struct, IFloatAdad<TN>
    {
        Experts.Gate(gate);
        return ExpertRun<TN>(v, gate, x, offsets, gateWeights, gateBias, upWeights, upBias, hidden);
    }

    private string ExpertRun<TN>(ExpertView v, string? gate, ulong x, ulong offsets, ulong gw, ulong gb, ulong uw, ulong ub, ulong y) where TN : struct, IFloatAdad<TN>
    {
        v.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (v.YValues == 0) return lane.Name;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong q = vk.Rent(Math.Max(16, bytes)); rented.Add(q); return q; }
        try
        {
            ulong tiles = Rent((v.Experts + 1) * 4L), rowExpert = Rent(v.Rows * 4L);
            var prefix = vk.Cached("expert_tiles", () => VkKernels.ExpertTiles(vk));
            vk.Launch(prefix, 1, 1, offsets, tiles, Vulkan.U(v.Experts), Vulkan.U(VkKernels.TileM));
            var ofRows = vk.Cached("expert_of_rows", () => VkKernels.ExpertOfRows(vk));
            vk.Launch(ofRows, Groups(v.Rows, VkKernels.Threads), 1, offsets, rowExpert, Vulkan.U(v.Rows), Vulkan.U(v.Experts));
            ulong gs = Rent(v.YValues * lane.Size), us = gate is null ? 0 : Rent(v.YValues * lane.Size);
            ExpertProducts(v, f, lane, x, offsets, tiles, gw, gs);
            if (gate is not null) ExpertProducts(v, f, lane, x, offsets, tiles, uw, us);
            bool bg = gb != 0, bu = ub != 0;
            if (f.IsBlock)
                Encode(f, v.YValues, vk.Cached($"block_expert_store:{f.Name}:{gate}:{bg}:{bu}", () => VkKernels.BlockExpertStore(vk, f, gate, bg, bu)), y,
                    gs, gb, us, ub, rowExpert, Vulkan.U(v.N), Vulkan.U(v.Experts));
            else
            {
                var store = vk.Cached($"expert_store:{f.Name}:{lane.Name}:{gate}:{bg}:{bu}", () => VkKernels.ExpertStore(vk, f, lane, gate, bg, bu));
                vk.Launch(store, Groups(f.Words(v.YValues), VkKernels.Threads), 1, gs, gb, us, ub, rowExpert, y, Vulkan.U((int)v.YValues), Vulkan.U(v.N), Vulkan.U(v.Experts));
            }
        }
        finally { foreach (var q in rented) vk.Return(q); }

        return lane.Name;
    }

    // Every expert's rows of x times its weights into the lane's sums (Kernels.Experts.cs MatMulGrouped): the workgroups' y
    // every expert's tiles at most (⌈rows / 64⌉ + experts).
    private void ExpertProducts(ExpertView v, Fmt f, Lane lane, ulong x, ulong offsets, ulong tiles, ulong w, ulong sums)
    {
        if (v.Rows == 0 || v.N == 0) return;
        var fn = vk.Cached($"matmul_grouped:{f.Name}:{lane.Name}:{v.TransW}", () => VkKernels.MatMulGrouped(vk, f, lane, v.TransW));
        uint ys = Groups(v.Rows, VkKernels.TileM) + (uint)v.Experts;
        vk.Launch(fn, Groups(v.N, VkKernels.TileN), ys, x, w, sums, Vulkan.U(v.N), Vulkan.U(v.K), offsets, tiles, Vulkan.U(v.Experts));
    }
}
