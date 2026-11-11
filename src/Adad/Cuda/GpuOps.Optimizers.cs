// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's optimizer steps (Optimizers.cs).
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (m not read by sgd, v = 0 for one
    /// moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IAdad<TN>
    {
        a.Validate();
        return Step(Fmt.Of<TN>(), a, p, g, m, v, n, 0);
    }

    // The step; factor: where the gradients' scale is (a lane value), 0 for none.
    private string Step(Fmt f, OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n, ulong factor)
    {
        var lane = RowLane(f);
        if (n == 0) return lane.Name;
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => lane.Wide ? Gpu.D(x) : Gpu.F((float)x);
        bool scaled = factor != 0;
        ulong[] args = [p, g, m, v, Gpu.U(n), S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd), factor];
        if (f.IsBlock)
        {
            // A block format's parameters: a warp per block, the step in f32 registers, the block re-encoded in place.
            var fb = gpu.Cached($"optimizer_blocks:{a.Name}:{f.Name}:{scaled}", (out string e) => Ptx.OptimizerBlocks(gpu, f, a.Name, out e, scaled));
            Gpu.Launch(fb, BlockGrid(f, n), 1, 256, 1, 0, args);
            return "f32 (blocks in place)";
        }

        var fn = gpu.Cached($"optimizer:{a.Name}:{f.Name}:{scaled}", (out string e) => Ptx.Optimizer(gpu, f, lane, a.Name, out e, scaled));
        Gpu.Launch(fn, Grid(f.Words(n), 256), 1, 256, 1, 0, args);
        return lane.Name;
    }

    /// <summary>A step across several tensors under one clipping (IOptimizers.OptimizerMany): every tensor's whole-sum pass
    /// of its gradients' squares into one buffer of float64 partials (the narrow floats' compensated pass, float64's own),
    /// the partials of all the tensors joined once in float64 and rounded once to the lane — one sum across the tensors —
    /// the factor from it by the map's clip_factor in the lane, then each tensor's step reading the factor where it is.
    /// Returns the lane's name.</summary>
    public string OptimizerMany<TN>(OptimizerArgs a, ReadOnlySpan<OptimizerTensor<ulong>> tensors, ulong sumSquares) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong d = gpu.Rent(Math.Max(16, bytes)); rented.Add(d); return d; }
        try
        {
            ulong factor = 0;
            if (a.MaxNorm > 0 || sumSquares != 0)
            {
                long blocks = 0;
                foreach (var t in tensors) if (t.N > 0) blocks += WholeBlocks(t.N);
                ulong sum = sumSquares != 0 ? sumSquares : Rent(lane.Size);
                if (blocks == 0) Gpu.Upload<byte>(new byte[lane.Size], sum);
                else
                {
                    var pass = lane.Wide
                        ? gpu.Cached($"reduce_rows:sum:True:{f.Name}:f64", (out string e) => Ptx.ReduceRows(gpu, f, Lane.F64, "sum", true, out e))
                        : gpu.Cached($"reduce_rows:sum:True:{f.Name}:f32:kahan", (out string e) => Ptx.ReduceRows(gpu, f, Lane.F32, "sum", true, out e, compensated: true));
                    var join = gpu.Cached($"reduce_rows:sum:False:{Ptx.LaneFormat(Lane.F64).Name}:f64", (out string e) => Ptx.ReduceRows(gpu, Ptx.LaneFormat(Lane.F64), Lane.F64, "sum", false, out e));
                    ulong partials = Rent(blocks * 8), wide = lane.Wide ? sum : Rent(8);
                    long at = 0;
                    foreach (var t in tensors)
                    {
                        if (t.N == 0) continue;
                        int tb = WholeBlocks(t.N);
                        Gpu.Launch(pass, (uint)tb, 1, Ptx.ReduceThreads, 1, 0, t.G, partials + (ulong)(at * 8), Gpu.U(t.N), Gpu.U(WholeChunk(t.N)));
                        at += tb;
                    }

                    Gpu.Launch(join, 1, 1, Ptx.ReduceThreads, 1, 0, partials, wide, Gpu.U((int)blocks), Gpu.U(0));
                    if (!lane.Wide) Back(Ptx.LaneFormat(Lane.F32), Lane.F64, wide, sum, 1);
                }

                if (a.MaxNorm > 0)
                {
                    factor = Rent(lane.Size);
                    var clip = Ops.Find("clip_factor");
                    if (lane.Wide) Map<Float64>(clip, sum, 0, 0, factor, 1, a.MaxNorm, 0); else Map<Float32>(clip, sum, 0, 0, factor, 1, a.MaxNorm, 0);
                }
            }

            foreach (var t in tensors) Step(f, a, t.P, t.G, t.M, t.V, t.N, factor);
        }
        finally { foreach (var d in rented) gpu.Return(d); }

        return f.IsBlock ? "f32 (blocks in place)" : lane.Name;
    }
}
