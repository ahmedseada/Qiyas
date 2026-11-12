// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on CUDA (Abstraction/Boxes.cs; the kernel Ptx.Boxes.cs): one launch, each output word computed
// where it is stored (a block format's by its encoder) in the float lane; the corners read where they are stored.
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
        if (n > uint.MaxValue) throw new NotSupportedException($"box losses: {n} values past 2^32");
        var fn = gpu.Cached($"box:{overlap}:{backward}:{f.Name}:{lane.Name}", (out string e) => Ptx.BoxLoss(gpu, f, lane, overlap, backward, out e));
        LaunchEach(fn, f, n, predicted, target, dloss, output, Gpu.U((int)n), lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps));
        return lane.Name;
    }
}
