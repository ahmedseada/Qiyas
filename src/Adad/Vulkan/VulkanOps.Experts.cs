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

    public string ExpertMatMul<TN>(ExpertView v, ulong x, ulong offsets, ulong weights, ulong bias, ulong y) where TN : struct, IAdad<TN> =>
        ExpertRun<TN>(v, null, x, offsets, weights, bias, 0, 0, y);

    public string ExpertGatedPair<TN>(ExpertView v, string gate, ulong x, ulong offsets, ulong gateWeights, ulong gateBias, ulong upWeights, ulong upBias, ulong hidden)
        where TN : struct, IAdad<TN>
    {
        Experts.Gate(gate);
        return ExpertRun<TN>(v, gate, x, offsets, gateWeights, gateBias, upWeights, upBias, hidden);
    }

    private string ExpertRun<TN>(ExpertView v, string? gate, ulong x, ulong offsets, ulong gw, ulong gb, ulong uw, ulong ub, ulong y) where TN : struct, IAdad<TN>
    {
        v.Validate();
        var f = Fmt.Of<TN>();
        if (v.YValues == 0) return f.IsFloat ? RowLane(f).Name : f.Natural.Name;
        var (lane, pl) = f.IsFloat ? (RowLane(f), RowLane(f)) : ExpertLanes<TN>(v, gate, x, gw, gb, uw, ub);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong q = vk.Rent(Math.Max(16, bytes)); rented.Add(q); return q; }
        try
        {
            ulong tiles = Rent((v.Experts + 1) * 4L), rowExpert = Rent(v.Rows * 4L);
            ulong gs = Rent(v.YValues * pl.Size), us = gate is null ? 0 : Rent(v.YValues * pl.Size);
            var prefix = vk.Cached("expert_tiles", () => VkKernels.ExpertTiles(vk));
            int micro = ExpertMicro(v, f, pl, m =>
            {
                vk.Launch(prefix, 1, 1, offsets, tiles, Vulkan.U(v.Experts), Vulkan.U(VkKernels.Tile(m)));
                GroupedProducts(v, f, pl, m, x, offsets, tiles, gw, gs);
            });
            vk.Launch(prefix, 1, 1, offsets, tiles, Vulkan.U(v.Experts), Vulkan.U(VkKernels.Tile(micro)));
            var ofRows = vk.Cached("expert_of_rows", () => VkKernels.ExpertOfRows(vk));
            vk.Launch(ofRows, Groups(v.Rows, VkKernels.Threads), 1, offsets, rowExpert, Vulkan.U(v.Rows), Vulkan.U(v.Experts));
            ExpertProducts(v, f, pl, micro, x, offsets, tiles, gw, gs);
            if (gate is not null) ExpertProducts(v, f, pl, micro, x, offsets, tiles, uw, us);
            bool bg = gb != 0, bu = ub != 0;
            if (f.IsBlock)
                Encode(f, v.YValues, vk.Cached($"block_expert_store:{f.Name}:{gate}:{bg}:{bu}", () => VkKernels.BlockExpertStore(vk, f, gate, bg, bu)), y,
                    gs, gb, us, ub, rowExpert, Vulkan.U(v.N), Vulkan.U(v.Experts));
            else
            {
                var store = vk.Cached($"expert_store:{f.Name}:{lane.Name}:{pl.Name}:{gate}:{bg}:{bu}", () => VkKernels.ExpertStore(vk, f, lane, gate, bg, bu, pl));
                vk.Launch(store, Groups(f.Words(v.YValues), VkKernels.Threads), 1, gs, gb, us, ub, rowExpert, y, Vulkan.U((int)v.YValues), Vulkan.U(v.N), Vulkan.U(v.Experts));
            }
        }
        finally { foreach (var q in rented) vk.Return(q); }

        return lane.Name;
    }

    // A type that is not float: the store's lane by the bounds of every value the store makes (ExpertView.IntegerSums, the
    // ranges read on the device; an unsigned type whose values may fall below 0 takes the float64 lane: ExactSums), and the
    // products' own lane by their bounds alone (x's range, the weights', k terms) where it is narrower: the gated pair's
    // product needs 64 bits where each sum needs 32; the store widens each sum exactly (an integer in either lane).
    private (Lane Store, Lane Products) ExpertLanes<TN>(ExpertView v, string? gate, ulong x, ulong gw, ulong gb, ulong uw, ulong ub) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        (double, double) R(ulong b, long n) => Range<TN>(b, (int)n);
        var rx = R(x, v.XValues);
        var rg = R(gw, v.WeightValues);
        (double, double)? ru = gate is null ? null : R(uw, v.WeightValues);
        var (bounds, negative) = v.IntegerSums(f.Name, f.Signed, gate, rx, rg, gb == 0 ? null : R(gb, v.BiasValues), ru, ub == 0 ? null : R(ub, v.BiasValues));
        var store = negative && !f.Signed ? Lane.F64 : SumLane(f, bounds);
        var (wLo, wHi) = ru is { } u ? (Math.Min(rg.Item1, u.Item1), Math.Max(rg.Item2, u.Item2)) : rg;
        var products = SumLane(f, new Bounds(rx.Item1, rx.Item2, wLo, wHi, v.K));
        return (store, products.Size < store.Size ? products : store);
    }

    private readonly Dictionary<string, int> _expertMicro = [];

    // Values a side an invocation of the grouped product holds: 4 (64 × 64 tiles), or 8 (128 × 128) in a 32-bit lane where
    // the experts' rows average a tile's, n spans one and it measures faster at the shape and type (once; run(micro): the
    // grouped product at it, its tiles' prefix first). The gate's forced tiling forces its values a side here too.
    private int ExpertMicro(ExpertView v, Fmt f, Lane lane, Action<int> run)
    {
        if (lane.Wide) return 4;
        if (TilingForced is PlainTiling t) return t.Micro;
        if (v.N < VkKernels.Tile(8) || v.Rows < (long)VkKernels.Tile(8) * Math.Max(1, v.Experts)) return 4;
        string key = $"{f.Name}:{lane.Name}:{v.Rows}x{v.N}x{v.K}:{v.Experts}:{v.TransW}";
        if (_expertMicro.TryGetValue(key, out int known)) return known;
        var (t8, t4) = Measure.Faster(() => run(8), () => run(4), vk.Synchronize);
        _unitsTimes.Add($"experts {key} grouped: 8x8 {t8:0.000} ms, 4x4 {t4:0.000} ms");
        return _expertMicro[key] = t8 <= t4 ? 8 : 4;
    }

    // Every expert's rows of x times its weights into the lane's sums: float64 by the sliced product where it is taken
    // (ExpertsSliced), else the grouped kernel (Kernels.Experts.cs MatMulGrouped): the workgroups' y every expert's tiles at
    // most (⌈rows / tile⌉ + experts).
    private void ExpertProducts(ExpertView v, Fmt f, Lane lane, int micro, ulong x, ulong offsets, ulong tiles, ulong w, ulong sums)
    {
        if (v.Rows == 0 || v.N == 0) return;
        if (ExpertsSliced(v, f, lane, x, offsets, w, sums, () => GroupedProducts(v, f, lane, micro, x, offsets, tiles, w, sums))) return;
        if (ExpertsUnits(v, f, lane, x, offsets, w, sums, () => GroupedProducts(v, f, lane, micro, x, offsets, tiles, w, sums))) return;
        GroupedProducts(v, f, lane, micro, x, offsets, tiles, w, sums);
    }

    /// <summary>The gate's override of the experts' products on the matrix units: true takes them wherever the type and lane
    /// allow, false never, null as measured.</summary>
    internal bool? ExpertsOnUnits;

    private readonly Dictionary<string, bool> _expertsUnits = [];

    // A type the matrix units take whole (float16, bfloat16, the 8-bit floats; an integer type of at most 8 bits) whose
    // experts' sums are the units' lane (f32; s32, u32): each expert's rows of x times its weights a units product (its rows
    // where they lie in x, its weights at e·k·n values, its sums at its rows of the sums: UnitsProduct), the offsets read
    // back first (a wait), where it measures faster than the grouped kernel over every expert, the read included (once per
    // shape). False where not taken.
    private bool ExpertsUnits(ExpertView v, Fmt f, Lane lane, ulong x, ulong offsets, ulong w, ulong sums, Action grouped)
    {
        if (v.K == 0 || ExpertsOnUnits == false || f.IsBlock) return false;
        string? kind = f.IsFloat ? UnitsKindOf(f, f) : Units != false && f.Bits <= 8 ? (f.Signed ? "s8" : "u8") : null;
        if (kind is null || Unit(kind) is null || UnitsLane(kind) != lane) return false;
        var off = new int[v.Experts + 1];
        void Run()
        {
            vk.Download<int>(offsets, off);
            for (int e = 0; e < v.Experts; e++)
            {
                int rows = off[e + 1] - off[e];
                if (rows <= 0) continue;
                UnitsProduct(new ProductView(rows, v.N, v.K, false, v.TransW), kind, x, f, (long)off[e] * v.K, w, f, (long)e * v.K * v.N,
                    sums + (ulong)((long)off[e] * v.N * lane.Size));
            }
        }

        if (ExpertsOnUnits != true)
        {
            string key = $"{f.Name}:{v.Rows}x{v.N}x{v.K}:{v.Experts}:{v.TransW}";
            if (!_expertsUnits.TryGetValue(key, out bool faster))
            {
                var (tg, tu) = Measure.Faster(grouped, Run, vk.Synchronize);
                _unitsTimes.Add($"experts {key}: grouped {tg:0.000} ms, units {tu:0.000} ms");
                faster = _expertsUnits[key] = tu < tg;
            }

            if (!faster) return false;
        }

        Run();
        return true;
    }

    private readonly Dictionary<string, bool> _expertsSliced = [];

    // float64: each expert's rows of x times its weights by the sliced product (an expert a product: its rows where they lie
    // in x, its weights at e·k·n values, its sums at its rows of the sums), where the device proved the route and it
    // measures faster than the grouped kernel over every expert (once per shape). The offsets read back first (a wait).
    // False where not taken (an expert refused: the grouped kernel then takes every expert).
    private bool ExpertsSliced(ExpertView v, Fmt f, Lane lane, ulong x, ulong offsets, ulong w, ulong sums, Action grouped)
    {
        if (!f.Is("float64") || lane != Lane.F64 || !SlicedProven || v.K == 0 || v.K > SlicedMaxK || Unit("s8") is null) return false;
        var off = new int[v.Experts + 1];
        vk.Download<int>(offsets, off);
        bool Run()
        {
            for (int e = 0; e < v.Experts; e++)
            {
                int rows = off[e + 1] - off[e];
                if (rows <= 0) continue;
                var pv = new ProductView(rows, v.N, v.K, false, v.TransW);
                if (!MatMulSliced(pv, x, (long)off[e] * v.K, w, (long)e * v.K * v.N, sums + (ulong)((long)off[e] * v.N * 8))) return false;
            }

            return true;
        }

        if (SlicedChoice is bool chosen) return chosen && Run();
        string key = $"{v.Rows}x{v.N}x{v.K}:{v.Experts}:{v.TransW}";
        if (!_expertsSliced.TryGetValue(key, out bool faster))
        {
            bool refused = false;
            var (tg, ts) = Measure.Faster(grouped, () => { if (!Run()) refused = true; }, vk.Synchronize);
            faster = _expertsSliced[key] = !refused && ts < tg;
        }

        return faster && Run();
    }

    private void GroupedProducts(ExpertView v, Fmt f, Lane lane, int micro, ulong x, ulong offsets, ulong tiles, ulong w, ulong sums)
    {
        int tile = VkKernels.Tile(micro);
        var fn = vk.Cached($"matmul_grouped:{f.Name}:{lane.Name}:{v.TransW}:{micro}", () => VkKernels.MatMulGrouped(vk, f, lane, v.TransW, micro));
        uint ys = Groups(v.Rows, tile) + (uint)v.Experts;
        vk.Launch(fn, Groups(v.N, tile), ys, x, w, sums, Vulkan.U(v.N), Vulkan.U(v.K), offsets, tiles, Vulkan.U(v.Experts));
    }
}
