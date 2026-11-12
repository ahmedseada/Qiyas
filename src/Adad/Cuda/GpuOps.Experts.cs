// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts on CUDA (Experts.cs; its kernels Ptx.Experts.cs): the routing by the ranks and picks kernels, then
// the route of the slots (the device's ordering, GpuOps.Radix.cs) and the rows; the experts' products by the stored GEMM's
// grouped mode (every expert's tiles in one launch, the offsets read where they are: no copy to the host), into the float
// lane's sums, then the store with each row's expert's bias and the gate.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    public string RouteTopK<TN>(ulong p, int tokens, int experts, int k, bool normalize, ulong weights, ulong offsets, ulong tokenOfRow, ulong slotRow) where TN : struct, IFloatAdad<TN>
    {
        Experts.ValidateRoute(tokens, experts, k);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        int slots = tokens * k;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong q = gpu.Rent(Math.Max(16, bytes)); rented.Add(q); return q; }
        try
        {
            ulong ranked = Rent((long)tokens * experts * 4), chosen = Rent(slots * 4L), lanes = Rent(slots * (long)lane.Size), counts = Rent(experts * 4L), order = Rent(slots * 4L);
            if (tokens > 0)
            {
                var ranks = gpu.Cached($"expert_ranks:{f.Name}:{lane.Name}", (out string e) => Ptx.ExpertRanks(gpu, f, lane, out e));
                Gpu.Launch(ranks, Grid((long)tokens * experts, 256), 1, 256, 1, 0, p, ranked, Gpu.U(tokens * experts), Gpu.U(experts), Gpu.U(k));
                var picks = gpu.Cached($"expert_picks:{f.Name}:{lane.Name}", (out string e) => Ptx.ExpertPicks(gpu, f, lane, out e));
                Gpu.Launch(picks, Grid(tokens, 256), 1, 256, 1, 0, p, ranked, chosen, lanes, Gpu.U(tokens), Gpu.U(experts), Gpu.U(k), Gpu.U(normalize ? 1 : 0));
                Back(f, lane, lanes, weights, slots);
            }

            OrderByTarget(chosen, slots, experts, counts, offsets, order);
            if (slots > 0)
            {
                var rows = gpu.Cached("expert_rows", (out string e) => Ptx.ExpertRows(gpu, out e));
                Gpu.Launch(rows, Grid(slots, 256), 1, 256, 1, 0, order, tokenOfRow, slotRow, Gpu.U(slots), Gpu.U(k));
            }
        }
        finally { foreach (var q in rented) gpu.Return(q); }

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
        var lane = f.IsFloat ? RowLane(f) : ExpertLane<TN>(v, gate, x, gw, gb, uw, ub);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong q = gpu.Rent(Math.Max(16, bytes)); rented.Add(q); return q; }
        try
        {
            const int micro = 4, tile = micro * 16;
            ulong tiles = Rent((v.Experts + 1) * 4L), rowExpert = Rent(v.Rows * 4L);
            var prefix = gpu.Cached("expert_tiles", (out string e) => Ptx.ExpertTiles(gpu, out e));
            Gpu.Launch(prefix, 1, 1, 32, 1, 0, offsets, tiles, Gpu.U(v.Experts), Gpu.U(tile));
            var ofRows = gpu.Cached("expert_of_rows", (out string e) => Ptx.ExpertOfRows(gpu, out e));
            Gpu.Launch(ofRows, Grid(v.Rows, 256), 1, 256, 1, 0, offsets, rowExpert, Gpu.U(v.Rows), Gpu.U(v.Experts));
            ulong gs = Rent(v.YValues * lane.Size), us = gate is null ? 0 : Rent(v.YValues * lane.Size);
            ExpertProducts(v, f, lane, micro, x, offsets, tiles, gw, gs);
            if (gate is not null) ExpertProducts(v, f, lane, micro, x, offsets, tiles, uw, us);
            var store = gpu.Cached($"expert_store:{f.Name}:{lane.Name}:{gate}:{gb != 0}:{ub != 0}", (out string e) => Ptx.ExpertStore(gpu, f, lane, gate, gb != 0, ub != 0, out e));
            LaunchEach(store, f, v.YValues, gs, gb, us, ub, rowExpert, y, Gpu.U((int)v.YValues), Gpu.U(v.N), Gpu.U(v.Experts));
        }
        finally { foreach (var q in rented) gpu.Return(q); }

        return lane.Name;
    }

    // A type that is not float: its products' lane by the bounds of every value the store makes (ExpertView.IntegerSums, the
    // ranges read on the device); an unsigned type whose values may fall below 0 takes the float64 lane (ExactSums).
    private Lane ExpertLane<TN>(ExpertView v, string? gate, ulong x, ulong gw, ulong gb, ulong uw, ulong ub) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        (double, double) R(ulong b, long n) => Range<TN>(b, (int)n);
        var (bounds, negative) = v.IntegerSums(f.Name, f.Signed, gate, R(x, v.XValues), R(gw, v.WeightValues), gb == 0 ? null : R(gb, v.BiasValues),
            gate is null ? null : R(uw, v.WeightValues), ub == 0 ? null : R(ub, v.BiasValues));
        return negative && !f.Signed ? Lane.F64 : SumLane(f, bounds);
    }

    // Every expert's rows of x times its weights into the lane's sums (Ptx.Gemm's grouped mode): the grid's y every
    // expert's tiles at most (⌈rows / tile⌉ + experts), each CTA its expert's from the tiles' prefix.
    private void ExpertProducts(ExpertView v, Fmt f, Lane lane, int micro, ulong x, ulong offsets, ulong tiles, ulong w, ulong sums)
    {
        if (v.Rows == 0 || v.N == 0) return;
        int tile = micro * 16;
        bool runsA = f.IsBlock && v.K % micro == 0, runsB = f.IsBlock && (v.TransW ? v.K : v.N) % micro == 0 && (long)v.K * v.N % micro == 0;
        var stored = new Ptx.GemmStored(f, f, null);
        var fn = gpu.Cached($"gemm_grouped:{f.Name}:{lane.Name}:{micro}:{v.TransW}:{runsA}:{runsB}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e, false, stored, false, v.TransW, runsA, runsB, grouped: true));
        uint ys = Grid(v.Rows, tile) + (uint)v.Experts;
        Gpu.Launch(fn, Grid(v.N, tile), ys, Ptx.GemmThreads, 1, 0, x, w, sums, Gpu.U(v.Rows), Gpu.U(v.N), Gpu.U(v.K), Gpu.U(0), Gpu.U(0), offsets, tiles, Gpu.U(v.Experts));
    }
}
