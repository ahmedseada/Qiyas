// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on the device (Sampling.cs; the kernels in Ptx.Sampling.cs): the weights are an intermediate in
// lane storage, the step is read where the device holds it.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered): the weights into lane storage
    /// (Ptx.SampleWeights), then the thresholds and the draw (Ptx.SampleDraw); the probabilities lane values, stored as the
    /// type once. Returns the lane's name.</summary>
    public string SampleFiltered<TN>(SamplingArgs a, ulong x, ulong step, ulong picks, ulong probability) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.Rows == 0) return lane.Name;
        bool own = f.Is("float32") || f.Is("float64");
        ulong w = gpu.Rent(Math.Max(16, (long)a.Rows * a.Cols * lane.Size));
        ulong p = probability == 0 ? 0 : own ? probability : gpu.Rent(Math.Max(16, (long)a.Rows * lane.Size));
        try
        {
            ulong Slot(double v) => lane.Wide ? Gpu.D(v) : Gpu.F((float)v);
            var weigh = gpu.Cached($"sample_weights:{f.Name}:{lane.Name}", (out string e) => Ptx.SampleWeights(gpu, f, lane, out e));
            Gpu.Launch(weigh, (uint)a.Rows, 1, 256, 1, 0, x, w, Gpu.U(a.Cols), Gpu.U(a.Stride), Gpu.U(a.RowOffset), Slot(a.Temperature));
            var draw = gpu.Cached($"sample_draw:{lane.Name}", (out string e) => Ptx.SampleDraw(gpu, lane, out e));
            Gpu.Launch(draw, (uint)a.Rows, 1, 256, 1, 0, w, picks, p, step, Gpu.U(a.Cols), Gpu.U(a.HasTopK ? a.TopK : 0), Draws.Key(a.Seed), Slot(a.HasTopP ? a.TopP : 0), Slot(a.MinP));
            if (probability != 0) FromLane(f, p, probability, a.Rows);
        }
        finally
        {
            gpu.Return(w);
            if (p != 0 && !own) gpu.Return(p);
        }

        return lane.Name;
    }
}
