// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Group normalization and the rotary embedding by tables on CUDA (Ptx.Norms.cs): the groups' coefficients by a block a
// group (lane values), then the outputs a thread per stored word; the parameters' gradients by a block a channel; the
// rotation's rows normalized by the row kernels' statistics (NormStats) when a gain is given.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    public string GroupNorm<TN>(ulong x, ulong gamma, ulong beta, ulong mean, ulong variance, ulong y, GroupShape shape, double eps, bool given) where TN : struct, IFloatAdad<TN> =>
        Group(Fmt.Of<TN>(), x, 0, gamma, beta, mean, variance, y, 0, 0, shape, eps, given, false);

    public string GroupNormBwd<TN>(ulong x, ulong dy, ulong gamma, ulong mean, ulong variance, ulong dx, ulong dgamma, ulong dbeta, GroupShape shape, double eps, bool given)
        where TN : struct, IFloatAdad<TN> =>
        Group(Fmt.Of<TN>(), x, dy, gamma, 0, mean, variance, dx, dgamma, dbeta, shape, eps, given, true);

    private string Group(Fmt f, ulong x, ulong dy, ulong gamma, ulong beta, ulong mean, ulong variance, ulong output, ulong dgamma, ulong dbeta, GroupShape s, double eps, bool given, bool backward)
    {
        var lane = RowLane(f);
        if (s.Count == 0) return lane.Name;
        bool g = gamma != 0, bt = beta != 0;
        var coefs = gpu.Cached($"group_coefs:{f.Name}:{lane.Name}:{backward}:{given}:{g}", (out string e) => Ptx.GroupCoefs(gpu, f, lane, backward, given, g, out e));
        ulong coef = gpu.Rent((long)s.Groups * 4 * lane.Size);
        try
        {
            Gpu.Launch(coefs, (uint)s.Groups, 1, 256, 1, 0, x, dy, gamma, mean, variance, coef, Gpu.U(s.Outer), Gpu.U(s.Groups), Gpu.U(s.Inner), Gpu.U(s.Channels), Gpu.U(s.Spatial),
                lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps));
            if (output != 0)
            {
                var outs = gpu.Cached($"group_out:{f.Name}:{lane.Name}:{backward}:{g}:{bt}", (out string e) => Ptx.GroupOut(gpu, f, lane, backward, g, bt, out e));
                uint grid = f.IsBlock ? BlockGrid(f, s.Count) : Grid(f.Words(s.Count), 256);
                Gpu.Launch(outs, grid, 1, 256, 1, 0, x, dy, gamma, beta, coef, output, Gpu.U(s.Count), Gpu.U(s.Groups), Gpu.U(s.Inner), Gpu.U(s.Channels), Gpu.U(s.Spatial));
            }

            if (backward && (dgamma != 0 || dbeta != 0))
            {
                var ps = gpu.Cached($"group_params:{f.Name}:{lane.Name}", (out string e) => Ptx.GroupParams(gpu, f, lane, out e));
                int per = s.Count / s.Channels;
                Gpu.Launch(ps, (uint)s.Channels, 1, 256, 1, 0, x, dy, coef, dgamma, dbeta, Gpu.U(per), Gpu.U(s.Groups), Gpu.U(s.Inner), Gpu.U(s.Channels), Gpu.U(s.Spatial));
            }
        }
        finally
        {
            gpu.Return(coef);
        }

        return lane.Name;
    }

    public string Rope<TN>(RopeArgs a, ulong x, ulong cos, ulong sin, ulong positions, ulong y) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        long n = (long)a.Rows * a.Cols;
        if (n == 0) return lane.Name;
        var fn = gpu.Cached($"rope_out:{f.Name}:{lane.Name}:{a.Interleaved}:{a.Inverse}:{a.Half > 0}",
            (out string e) => Ptx.RopeOut(gpu, f, lane, a.Interleaved, a.Inverse, a.Half > 0, out e));
        uint grid = f.IsBlock ? BlockGrid(f, n) : Grid(f.Words(n), 256);
        Gpu.Launch(fn, grid, 1, 256, 1, 0, x, cos, sin, positions, y, Gpu.U((int)n), Gpu.U(a.Cols), Gpu.U(a.Heads), Gpu.U(a.Steps), Gpu.U(a.Half));
        return lane.Name;
    }

    // stats[2·row] = mean (0 when not centered), stats[2·row + 1] = 1/√(mean (x − mean)² + eps): the rows' statistics kernel (Ptx.NormStats).
    private void RowStatistics(Fmt f, Lane lane, bool centered, ulong x, ulong stats, int rows, int cols, double eps)
    {
        var (threads, perThread, vector, valueWise) = RowKernelShape(f, cols, "norm_stats");
        bool segmented = Segmented(f, cols, perThread, threads, valueWise);
        var fn = gpu.Cached($"norm_stats:{centered}:{f.Name}:{perThread}:{threads}:{vector}:{valueWise}:{segmented}", (out string e) => Ptx.NormStats(gpu, f, lane, centered, perThread, threads, vector, valueWise, out e, segmented));
        if (threads > Gpu.MaxThreads(fn))
        {
            (threads, perThread, vector) = (256, 8, false);
            segmented = Segmented(f, cols, perThread, threads, valueWise);
            fn = gpu.Cached($"norm_stats:{centered}:{f.Name}:{perThread}:{threads}:{vector}:{valueWise}:{segmented}", (out string e) => Ptx.NormStats(gpu, f, lane, centered, perThread, threads, vector, valueWise, out e, segmented));
        }

        Gpu.Launch(fn, (uint)rows, 1, (uint)threads, 1, 0, x, 0, 0, 0, 0, 0, stats, 0, Gpu.U(rows), Gpu.U(cols), lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps), lane.Wide ? Gpu.D(0) : Gpu.F(0));
    }
}
