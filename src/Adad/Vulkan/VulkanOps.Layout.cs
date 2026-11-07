// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The layout operations on Vulkan (Kernels.Layout.cs), CUDA's lane rule (Cuda/GpuOps.Layout.cs): the type's own lane for
// moves, the float lane for the weighted ones (bilinear interpolation, the average pools), the sum lane by the bounds of
// the terms for the sums (an integer sum stays an integer); scatter-add's ordering and the max-pool backward's window
// choices built on the device first; then one kernel, a thread per output word (a block format's by its encoder, each
// block from its values).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>A layout operation (LayoutArgs.Names) from x (and the index array, and x2) into output, all on the device. Returns the lane's name.</summary>
    public string Layout<TN>(LayoutArgs k, ulong x, ulong idx, ulong output, ulong x2) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var (rows, cols) = LayoutArgs.OutputRows(k);
        Lane lane;
        if (LayoutArgs.Sums(k.Name))
        {
            var (lo, hi) = Range<TN>(x, k.InputCount);
            var (lo2, hi2) = x2 != 0 && k.Name == "depthwise_conv" ? Range<TN>(x2, k.P[0] * k.P[3] * k.P[4]) : (1, 1);
            lane = SumLane(f, new Bounds(lo, hi, lo2, hi2, LayoutArgs.Terms(k.Name, k.P)));
            if (!f.IsFloat && lane.IsFloat) lane = f.Signed ? Lane.S32 : Lane.U32;   // an integer sum stays an integer (its store converts to the type)
        }
        else lane = LayoutArgs.FloatLane(k.Name) ? RowLane(f) : f.Natural;

        ulong aux = 0, counts = 0;
        try
        {
            if (k.Name == "scatter_add_rows")
            {
                // The rows' offsets, then the sources in order of their row (stable: the radix ordering, VulkanOps.Routing.cs).
                int n = k.P[0], targets = k.P[2];
                counts = vk.Rent(Math.Max(1, targets) * 4L);
                aux = vk.Rent((targets + 1 + Math.Max(1, n)) * 4L);
                Order(idx, n, targets, counts, aux, aux + (ulong)((targets + 1) * 4L));
            }

            if (k.Name is "maxpool_bwd" or "adaptive_maxpool_bwd")
            {
                // Each window's first largest of the forward input, once, into aux.
                var (OH, OW) = k.Name == "maxpool_bwd" ? LayoutArgs.PoolOut(k.P) : (k.P[3], k.P[4]);
                int windows = k.P[0] * OH * OW;
                aux = vk.Rent(Math.Max(1, windows) * 4L);
                var argmax = vk.Cached($"layout_argmax:{k.Name}:{f.Name}:{lane.Name}", () => VkKernels.LayoutWindowArgmax(vk, f, lane, k.Name));
                if (windows > 0) vk.Launch(argmax, Groups(windows, VkKernels.Threads), 1, VkKernels.LayoutSlots(0, x2, 0, 0, aux, windows, 0, 0, k.P));
            }

            if (k.Count > 0 && f.IsBlock)
                Encode(f, k.Count, vk.Cached($"block_layout:{k.Name}:{f.Name}:{lane.Name}", () => VkKernels.BlockLayout(vk, f, lane, k.Name)), output, VkKernels.LayoutSlots(x, x2, idx, aux, output, k.Count, rows, cols, k.P));
            else if (k.Count > 0)
            {
                var fn = vk.Cached($"layout:{k.Name}:{f.Name}:{lane.Name}", () => VkKernels.Layout(vk, f, lane, k.Name));
                vk.Launch(fn, Groups(f.Words(k.Count), VkKernels.Threads), 1, VkKernels.LayoutSlots(x, x2, idx, aux, output, k.Count, rows, cols, k.P));
            }
        }
        finally
        {
            if (aux != 0) vk.Return(aux);
            if (counts != 0) vk.Return(counts);
        }

        return lane.Name;
    }
}
