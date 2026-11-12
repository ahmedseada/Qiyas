// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on Vulkan (Sampling.cs; the kernels in Kernels.Sampling.cs): the weights are an intermediate in
// lane storage, the step is read where the device holds it.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered): the weights into lane storage
    /// (VkKernels.SampleWeights), then the thresholds and the draw (VkKernels.SampleDraw); the probabilities lane values,
    /// stored as the type once. Returns the lane's name.</summary>
    public string SampleFiltered<TN>(SamplingArgs a, ulong x, ulong step, ulong picks, ulong probability) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.Rows == 0) return lane.Name;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong r = vk.Rent(Math.Max(16, bytes)); rented.Add(r); return r; }
        try
        {
            ulong w = Rent((long)a.Rows * a.Cols * lane.Size);
            ulong p = probability == 0 ? 0 : Own(f) ? probability : Rent((long)a.Rows * lane.Size);
            vk.Launch(vk.Cached($"sample_weights:{f.Name}:{lane.Name}", () => VkKernels.SampleWeights(vk, f, lane)), (uint)a.Rows, 1,
                x, w, Vulkan.U(a.Cols), Vulkan.U(a.Stride), Vulkan.U(a.RowOffset), Slot(lane, a.Temperature));
            var key = Draws.Key(a.Seed);
            vk.Launch(vk.Cached($"sample_draw:{lane.Name}", () => VkKernels.SampleDraw(vk, lane)), (uint)a.Rows, 1,
                w, picks, p, step, Vulkan.U(a.Cols), Vulkan.U(a.HasTopK ? a.TopK : 0), key.A, Slot(lane, a.HasTopP ? a.TopP : 0), Slot(lane, a.MinP), key.B);
            if (probability != 0 && !Own(f)) Back(f, lane, p, probability, a.Rows);
        }
        finally
        {
            foreach (var r in rented) vk.Return(r);
        }

        return lane.Name;
    }
}
