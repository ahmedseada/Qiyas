// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on the device (Sampling.cs; the kernels in Ptx.Sampling.cs): the weights are an intermediate in
// lane storage, the scores and the step are read where the device holds them.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered): the weights into lane storage
    /// (Ptx.SampleWeights), then each rule's prefix of the row's order and the draw (Ptx.SampleDraw); the probabilities in
    /// the float lane's storage. Every type but bool. Returns the lane's name.</summary>
    public string SampleFiltered<TN>(SamplingArgs a, ulong x, ulong step, ulong picks, ulong probability) where TN : struct, IAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        if (f.Bits == 1) throw new NotSupportedException($"sampling: {f.Name} holds truth values, not scores");
        var lane = f.FloatLane;
        if (a.Rows == 0) return lane.Name;
        ulong w = gpu.Rent(Math.Max(16, (long)a.Rows * a.Cols * lane.Size));
        try
        {
            ulong Slot(double v) => lane.Wide ? Gpu.D(v) : Gpu.F((float)v);
            var weigh = gpu.Cached($"sample_weights:{f.Name}:{lane.Name}", (out string e) => Ptx.SampleWeights(gpu, f, lane, out e));
            Gpu.Launch(weigh, (uint)a.Rows, 1, 256, 1, 0, x, w, Gpu.U(a.Cols), Gpu.U(a.Stride), Gpu.U(a.RowOffset), Slot(a.Temperature));
            var key = Draws.Key(a.Seed);
            var draw = gpu.Cached($"sample_draw:{f.Name}:{lane.Name}", (out string e) => Ptx.SampleDraw(gpu, f, lane, out e));
            Gpu.Launch(draw, (uint)a.Rows, 1, 256, 1, 0, x, w, picks, probability, step, Gpu.U(a.Cols), Gpu.U(a.Stride), Gpu.U(a.RowOffset),
                Gpu.U(a.TopK), key.A, Slot(a.TopP), Slot(a.MinP), key.B);
        }
        finally
        {
            gpu.Return(w);
        }

        return lane.Name;
    }

    /// <summary>keys[i] (u64) = the order key (Ptx.OrderKey) of x's value i of TN in its natural lane: the key proven on the
    /// device against Routing.Order.</summary>
    internal void OrderKeys<TN>(ulong x, ulong keys, int n) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (n == 0) return;
        var k = gpu.Cached($"order_keys:{f.Name}", (out string e) => Ptx.OrderKeys(gpu, f, out e));
        Gpu.Launch(k, Grid(n, 256), 1, 256, 1, 0, x, keys, Gpu.U(n));
    }
}
