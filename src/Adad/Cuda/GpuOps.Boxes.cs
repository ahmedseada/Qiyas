// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on CUDA (Abstraction/Boxes.cs; the kernels Ptx.Boxes.cs): one launch in the float lane, the corners
// read where they are stored — the pairs kernel (each pair once, its outputs packed into whole words) where the format is
// words of a power of two values and the arrays are 16-byte aligned, else the value kernel (each output word computed where
// it is stored, a block format's by its encoder).
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>loss[i] of count pairs of corner boxes (count × 4 as the type each) as the type, eps the type's; returns the
    /// lane's name.</summary>
    public string BoxLoss<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong loss, int count) where TN : struct, IFloatAdad<TN> =>
        BoxRun<TN>(overlap, predicted, target, 0, loss, count);

    /// <summary>dpredicted = dloss[i] · ∂loss[i] / ∂predicted[i] (count × 4 as the type); returns the lane's name.</summary>
    public string BoxLossBackward<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong dpredicted, int count) where TN : struct, IFloatAdad<TN> =>
        BoxRun<TN>(overlap, predicted, target, dloss, dpredicted, count);

    private string BoxRun<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong output, int count) where TN : struct, IFloatAdad<TN>
    {
        double eps = TN.Epsilon;
        _ = Boxes.Name(overlap);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        bool backward = dloss != 0;
        long n = (long)count * (backward ? 4 : 1);
        if (4L * count > uint.MaxValue) throw new NotSupportedException($"box losses: {4L * count} corners past 2^32");
        ulong e = lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps);
        if (count == 0) return lane.Name;
        if (Ptx.BoxPairWise(f) && (predicted | target | output) % 16 == 0)
        {
            var fp = gpu.Cached($"box_pairs:{overlap}:{backward}:{f.Name}:{lane.Name}", (out string err) => Ptx.BoxLossPairs(gpu, f, lane, overlap, backward, out err));
            int pairs = Ptx.BoxThreadPairs(f, backward);
            Gpu.Launch(fp, Grid((count + pairs - 1) / pairs, 256), 1, 256, 1, 0, predicted, target, dloss, output, Gpu.U(count), e);
            return lane.Name;
        }

        var fn = gpu.Cached($"box:{overlap}:{backward}:{f.Name}:{lane.Name}", (out string err) => Ptx.BoxLoss(gpu, f, lane, overlap, backward, out err));
        LaunchEach(fn, f, n, predicted, target, dloss, output, Gpu.U((int)n), e);
        return lane.Name;
    }
}
