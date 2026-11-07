// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's row operations and the normalization parameters' backward (Rows.cs), dispatched like softmax: a block
// per row, the row in registers; rows not ending on a word value by value.
namespace Qiyas;

public sealed partial class GpuOps
{
    private (int Threads, int PerThread, bool Vector, bool ValueWise) RowKernelShape(Fmt f, int cols, string what)
    {
        bool valueWise = (long)cols * f.Bits % f.WordBits != 0;
        int threads, perThread;
        if (valueWise)
        {
            threads = 128;
            while (threads < 1024 && (cols + threads - 1) / threads > 4) threads *= 2;
            perThread = (cols + threads - 1) / threads;
        }
        else
        {
            (threads, perThread) = RowShape(f, cols);
        }

        if (perThread > 16) { threads = 256; perThread = 8; }             // longer rows: the row in segments of 256 × 8 words (or values), reloaded per phase (the registers of 16 words × 1024 threads exceed a block's)
        bool vector = !valueWise && f.WordBits == 32 && perThread == 4 && f.Words(cols) % 4 == 0;
        return (threads, perThread, vector, valueWise);
    }

    /// <summary>A row operation (RowOps.All) over x (rows × cols) into y (and y2); null inputs are 0 pointers. Returns the lane's name.</summary>
    public string RowOp<TN>(RowOp op, ulong x, ulong x2, ulong g, ulong bt, ulong s, ulong pos, ulong y, ulong y2, int rows, int cols, double a, double b) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                long n = (long)rows * cols;
                ulong lx = Lanes(f, Lane.F32, x, n, r), lx2 = x2 == 0 ? 0 : Lanes(f, Lane.F32, x2, n, r), lg = g == 0 ? 0 : Lanes(f, Lane.F32, g, cols, r), lb = bt == 0 ? 0 : Lanes(f, Lane.F32, bt, cols, r), ls = s == 0 ? 0 : Lanes(f, Lane.F32, s, rows, r);
                ulong ly = gpu.Rent(n * 4), ly2 = y2 == 0 ? 0 : gpu.Rent(n * 4);
                r.Add(ly); if (ly2 != 0) r.Add(ly2);
                RowOp<Float32>(op, lx, lx2, lg, lb, ls, pos, ly, ly2, rows, cols, a, b);
                Back(f, Lane.F32, ly, y, n);
                if (y2 != 0) Back(f, Lane.F32, ly2, y2, n);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        var lane = RowLane(f);
        var (threads, perThread, vector, valueWise) = RowKernelShape(f, cols, op.Name);
        bool second = x2 != 0, segmented = Segmented(f, cols, perThread, threads, valueWise);
        var fn = gpu.Cached($"row:{op.Name}:{f.Name}:{perThread}:{threads}:{vector}:{valueWise}:{second}:{segmented}", (out string e) => Ptx.RowOp(gpu, f, lane, op, perThread, threads, vector, valueWise, second, out e, segmented));
        if (threads > Gpu.MaxThreads(fn))
        {
            // The row in registers needs more of them than a block of this many threads has on this device: the segmented shape.
            (threads, perThread, vector) = (256, 8, false);
            segmented = Segmented(f, cols, perThread, threads, valueWise);
            fn = gpu.Cached($"row:{op.Name}:{f.Name}:{perThread}:{threads}:{vector}:{valueWise}:{second}:{segmented}", (out string e) => Ptx.RowOp(gpu, f, lane, op, perThread, threads, vector, valueWise, second, out e, segmented));
        }

        try { Gpu.Launch(fn, (uint)rows, 1, (uint)threads, 1, 0, x, x2, g, bt, s, pos, y, y2, Gpu.U(rows), Gpu.U(cols), lane.Wide ? Gpu.D(a) : Gpu.F((float)a), lane.Wide ? Gpu.D(b) : Gpu.F((float)b)); }
        catch (InvalidOperationException e) { throw new InvalidOperationException($"{op.Name} {f.Name} {rows} x {cols}: {threads} threads x {perThread} values, vector {vector}, value-wise {valueWise}, segmented {segmented}: {e.Message}", e); }
        return lane.Name;
    }

    private static bool Segmented(Fmt f, int cols, int perThread, int threads, bool valueWise) => (valueWise ? cols : f.Words(cols)) > (long)perThread * threads;

    /// <summary>dgamma[c] = Σ_r dy·x̂, dbeta[c] = Σ_r dy (lane values, cols each); x̂ by rms or, centered, layer statistics. Returns the lane's name.</summary>
    public string NormParamsBwd<TN>(ulong x, ulong dy, ulong dgamma, ulong dbeta, int rows, int cols, double eps, bool centered) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var (threads, perThread, vector, valueWise) = RowKernelShape(f, cols, "norm_params_bwd");
        bool segmented = Segmented(f, cols, perThread, threads, valueWise);
        var stats = gpu.Cached($"norm_stats:{centered}:{f.Name}:{perThread}:{threads}:{vector}:{valueWise}:{segmented}", (out string e) => Ptx.NormStats(gpu, f, lane, centered, perThread, threads, vector, valueWise, out e, segmented));
        if (threads > Gpu.MaxThreads(stats))
        {
            (threads, perThread, vector) = (256, 8, false);
            segmented = Segmented(f, cols, perThread, threads, valueWise);
            stats = gpu.Cached($"norm_stats:{centered}:{f.Name}:{perThread}:{threads}:{vector}:{valueWise}:{segmented}", (out string e) => Ptx.NormStats(gpu, f, lane, centered, perThread, threads, vector, valueWise, out e, segmented));
        }

        var cols2 = gpu.Cached($"norm_params_cols:{f.Name}:{lane.Name}", (out string e) => Ptx.NormParamsCols(gpu, f, lane, out e));
        ulong st = gpu.Rent((long)rows * 2 * lane.Size);
        int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
        ulong pg = splits == 1 ? dgamma : gpu.Rent((long)splits * cols * lane.Size), pb = splits == 1 ? dbeta : gpu.Rent((long)splits * cols * lane.Size);
        try
        {
            Gpu.Launch(stats, (uint)rows, 1, (uint)threads, 1, 0, x, 0, 0, 0, 0, 0, st, 0, Gpu.U(rows), Gpu.U(cols), lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps), lane.Wide ? Gpu.D(0) : Gpu.F(0));
            Gpu.Launch(cols2, Grid(cols, 256), (uint)splits, 256, 1, 0, x, dy, st, pg, pb, Gpu.U(rows), Gpu.U(cols), Gpu.U(per));
            if (splits > 1)
            {
                var lf = Ptx.LaneFormat(lane);
                var join = gpu.Cached($"reduce_cols:sum:False:{lf.Name}:{lane.Name}", (out string e) => Ptx.ReduceCols(gpu, lf, lane, "sum", false, out e));
                Gpu.Launch(join, Grid(cols, 256), 1, 256, 1, 0, pg, dgamma, Gpu.U(splits), Gpu.U(cols), Gpu.U(splits));
                Gpu.Launch(join, Grid(cols, 256), 1, 256, 1, 0, pb, dbeta, Gpu.U(splits), Gpu.U(cols), Gpu.U(splits));
            }
        }
        finally
        {
            gpu.Return(st);
            if (splits > 1) { gpu.Return(pg); gpu.Return(pb); }
        }

        return lane.Name;
    }
}
