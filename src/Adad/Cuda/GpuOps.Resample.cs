// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on CUDA (Ptx.Resample.cs): the horizontal pass into a rented plane of lane
// values where it runs, then the outputs a thread per stored word (a warp or a CTA per block), the vertical
// pass reading the horizontal one's values (or the input, the width kept) where they are.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    public string Resample<TIn, TN>(ResampleArgs a, ulong x, ulong bounds, ulong weights, ulong values, ulong y) where TIn : struct, IAdad<TIn> where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        Fmt fi = Fmt.Of<TIn>(), f = Fmt.Of<TN>();
        var lane = RowLane(f);
        string name = lane.Name;
        if (a.Count == 0) return name;
        // The horizontal pass's values: the lane's.
        var midFmt = lane.Wide ? new Fmt("float64", 64, true, true) : new Fmt("float32", 32, true, true);
        ulong mid = 0;
        try
        {
            if (a.XTaps > 0)
            {
                long n = a.MiddleCount;
                mid = gpu.Rent(n * lane.Size);
                var rows = gpu.Cached($"resample_rows:{fi.Name}:{lane.Name}", (out string e) => Ptx.ResampleRows(gpu, fi, lane, out e));
                Gpu.Launch(rows, Grid(n, 256), 1, 256, 1, 0, x, bounds, weights, mid, Gpu.U((int)n), Gpu.U(a.Width), Gpu.U(a.OutWidth), Gpu.U(a.XTaps));
            }

            var src = a.XTaps > 0 ? midFmt : fi;
            bool vertical = a.YTaps > 0, norm = values != 0;
            var outs = gpu.Cached($"resample_out:{f.Name}:{src.Name}:{lane.Name}:{vertical}:{norm}", (out string e) => Ptx.ResampleOut(gpu, f, src, lane, vertical, norm, out e));
            ulong[] args = [a.XTaps > 0 ? mid : x, bounds, weights, values, y, Gpu.U(a.Count), Gpu.U(a.OutHeight), Gpu.U(a.OutWidth), Gpu.U(a.Height), Gpu.U(a.Across),
                Gpu.U(a.YTaps), Gpu.U(a.RowBounds), Gpu.U(a.RowWeights), Gpu.U(a.Channels)];
            if (f.IsBlock && Ptx.BlockByCta(f)) Gpu.Launch(outs, (uint)((a.Count + f.BlockValues - 1) / f.BlockValues), 1, (uint)f.BlockValues, 1, 0, args);
            else if (f.IsBlock) Gpu.Launch(outs, BlockGrid(f, a.Count), 1, 256, 1, 0, args);
            else Gpu.Launch(outs, Grid(f.Words(a.Count), 256), 1, 256, 1, 0, args);
        }
        finally
        {
            if (mid != 0) gpu.Return(mid);
        }

        return name;
    }
}
