// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on Vulkan (Kernels.Resample.cs), CUDA's plan (Cuda/GpuOps.Resample.cs): the
// horizontal pass into a rented plane of lane values where it runs, then the outputs a word an invocation
// (a block format's by its encoder), the vertical pass reading the horizontal one's values (or the input) where they are.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    public string Resample<TIn, TN>(ResampleArgs a, ulong x, ulong bounds, ulong weights, ulong values, ulong y) where TIn : struct, IAdad<TIn> where TN : struct, IAdad<TN>
    {
        a.Validate();
        Fmt fi = Fmt.Of<TIn>(), f = Fmt.Of<TN>();
        var lane = f.FloatLane;
        string name = lane.Name;
        if (a.Count == 0) return name;
        var midFmt = lane.Wide ? new Fmt("float64", 64, true, true) : new Fmt("float32", 32, true, true);
        ulong mid = 0;
        try
        {
            if (a.XTaps > 0)
            {
                long n = a.MiddleCount;
                mid = vk.Rent(n * lane.Size);
                var rows = vk.Cached($"resample_rows:{fi.Name}:{lane.Name}", () => VkKernels.ResampleRows(vk, fi, lane));
                vk.Launch(rows, Groups(n, VkKernels.Threads), 1, x, bounds, weights, mid, Vulkan.U((int)n), Vulkan.U(a.Width), Vulkan.U(a.OutWidth), Vulkan.U(a.XTaps));
            }

            var src = a.XTaps > 0 ? midFmt : fi;
            ulong s = a.XTaps > 0 ? mid : x;
            bool vertical = a.YTaps > 0, norm = values != 0;
            if (f.IsBlock)
                Encode(f, a.Count, vk.Cached($"block_resample_out:{f.Name}:{src.Name}:{vertical}:{norm}", () => VkKernels.BlockResampleOut(vk, f, src, vertical, norm)), y,
                    s, bounds, weights, values, Vulkan.U(a.OutHeight), Vulkan.U(a.OutWidth), Vulkan.U(a.Height), Vulkan.U(a.Across), Vulkan.U(a.YTaps), Vulkan.U(a.RowBounds),
                    Vulkan.U(a.RowWeights), Vulkan.U(a.Channels));
            else
            {
                var k = vk.Cached($"resample_out:{f.Name}:{src.Name}:{lane.Name}:{vertical}:{norm}", () => VkKernels.ResampleOut(vk, f, src, lane, vertical, norm));
                vk.Launch(k, Groups(f.Words(a.Count), VkKernels.Threads), 1, s, bounds, weights, values, y, Vulkan.U(a.Count), Vulkan.U(a.OutHeight), Vulkan.U(a.OutWidth),
                    Vulkan.U(a.Height), Vulkan.U(a.Across), Vulkan.U(a.YTaps), Vulkan.U(a.RowBounds), Vulkan.U(a.RowWeights), Vulkan.U(a.Channels));
            }
        }
        finally
        {
            if (mid != 0) vk.Return(mid);
        }

        return name;
    }
}
