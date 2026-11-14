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
    public string Layout<TN>(LayoutArgs k, ulong x, ulong idx, ulong output, ulong x2) where TN : struct, IAdad<TN> => Layout<TN>(k, x, idx, output, x2, null);

    /// <summary>inFmt: the inputs' format where it is not the output's (a composition's lane storage, or the type's read into
    /// lane storage), the lane its natural one.</summary>
    internal string Layout<TN>(LayoutArgs k, ulong x, ulong idx, ulong output, ulong x2, Fmt? inFmt) where TN : struct, IAdad<TN>
    {
        k = k.ForKernels();
        var f = Fmt.Of<TN>();
        var (_, cols) = LayoutArgs.OutputRows(k);
        // A strided view takes the kernel of its canonical rank (a division a dimension a value).
        string kn = LayoutArgs.KernelName(k.Name, k.P);
        Lane lane;
        if (inFmt is not null) lane = inFmt.Natural;
        else if (LayoutArgs.Sums(k.Name))
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

            if (k.Name is "interpolate_bilinear" or "interpolate_bilinear_bwd" && k.Count > 0)
            {
                // Each output row's and column's source rows and weight, once (VkKernels.BilinearTables), into aux.
                int oh = k.P[3], ow = k.P[4];
                aux = vk.Rent(VkKernels.BilinearTableBytes(oh, ow, lane));
                var tables = vk.Cached($"bilinear_tables:{lane.Name}", () => VkKernels.BilinearTables(vk, lane));
                vk.Launch(tables, Groups(oh + ow, VkKernels.Threads), 1, VkKernels.LayoutSlots(0, 0, 0, aux, 0, oh + ow, 0, k.P));
            }

            if (k.Name is "maxpool_bwd" or "adaptive_maxpool_bwd")
            {
                // Each window's first-ranked of the forward input in Adad's order, once, into aux.
                var (OH, OW) = k.Name == "maxpool_bwd" ? LayoutArgs.PoolOut(k.P) : (k.P[3], k.P[4]);
                int windows = k.P[0] * OH * OW;
                aux = vk.Rent(Math.Max(1, windows) * 4L);
                var argmax = vk.Cached($"layout_argmax:{k.Name}:{f.Name}:{lane.Name}", () => VkKernels.LayoutWindowArgmax(vk, f, lane, k.Name));
                if (windows > 0) vk.Launch(argmax, Groups(windows, VkKernels.Threads), 1, VkKernels.LayoutSlots(0, x2, 0, 0, aux, windows, 0, k.P));
            }

            // A block format's adaptive pools: the windows by AdaptivePool into lane storage (a workgroup an output value),
            // then the encoder from those values (each block's values once, not each value's window decoded in the encoder).
            if (k.Count > 0 && f.IsBlock && k.Name is "adaptive_maxpool" or "adaptive_avgpool" && inFmt is null && k.Count <= (long)vk.MaxGroupsX * vk.MaxGroupsY)
            {
                var lf = VkKernels.LaneFormat(lane);
                ulong lanes = vk.Rent(k.Count * (long)lane.Size);
                try
                {
                    var pool = vk.Cached($"adaptive_pool:{k.Name}:{f.Name}:{lane.Name}:{lf.Name}", () => VkKernels.AdaptivePool(vk, f, lane, k.Name == "adaptive_maxpool", lf));
                    uint px = (uint)Math.Min(k.Count, vk.MaxGroupsX), py = (uint)((k.Count + px - 1) / px);
                    vk.Launch(pool, px, py, VkKernels.LayoutSlots(x, x2, idx, aux, lanes, k.Count, cols, k.P));
                    Encode(f, k.Count, vk.Cached($"block_lanes:{lane.Name}:{f.Name}", () => VkKernels.BlockFromLanes(vk, lane, f)), output, lanes);
                }
                finally
                {
                    vk.Return(lanes);
                }
            }
            // A block format's encoder reads its sources alone: the output's slot is left empty, so only an output that is one
            // of the inputs makes the encoding go through a buffer of its own (Encode).
            else if (k.Count > 0 && f.IsBlock)
                Encode(f, k.Count, vk.Cached($"block_layout:{kn}:{f.Name}:{lane.Name}:{inFmt?.Name}", () => VkKernels.BlockLayout(vk, f, lane, kn, inFmt)), output, VkKernels.LayoutSlots(x, x2, idx, aux, 0, k.Count, cols, k.P));
            else if (k.Count > 0 && k.Name is "adaptive_maxpool" or "adaptive_avgpool" && inFmt is null && f.Words(k.Count) <= (long)vk.MaxGroupsX * vk.MaxGroupsY)
            {
                // A workgroup an output word, its windows read by every invocation (VkKernels.AdaptivePool).
                var pool = vk.Cached($"adaptive_pool:{k.Name}:{f.Name}:{lane.Name}", () => VkKernels.AdaptivePool(vk, f, lane, k.Name == "adaptive_maxpool"));
                long words = f.Words(k.Count);
                uint px = (uint)Math.Min(words, vk.MaxGroupsX), py = (uint)((words + px - 1) / px);
                vk.Launch(pool, px, py, VkKernels.LayoutSlots(x, x2, idx, aux, output, k.Count, cols, k.P));
            }
            else if (k.Count > 0 && k.Name == "strided" && inFmt is null && StridedTiles(f, k.P) is var (n, j, gx, gy) && j >= 0)
            {
                // A view that transposes a stride-1 dimension with its last: by tiles (VkKernels.StridedTiled).
                var tiled = vk.Cached($"strided_tiled:{n}:{j}:{f.Name}:{lane.Name}", () => VkKernels.StridedTiled(vk, f, lane, n, j));
                vk.Launch(tiled, gx, gy, VkKernels.LayoutSlots(x, x2, idx, aux, output, k.Count, cols, k.P));
            }
            else if (k.Count > 0)
            {
                var fn = vk.Cached($"layout:{kn}:{f.Name}:{lane.Name}:{inFmt?.Name}", () => VkKernels.Layout(vk, f, lane, kn, inFmt));
                vk.Launch(fn, Groups(f.Words(k.Count), VkKernels.Threads), 1, VkKernels.LayoutSlots(x, x2, idx, aux, output, k.Count, cols, k.P));
            }
        }
        finally
        {
            if (aux != 0) vk.Return(aux);
            if (counts != 0) vk.Return(counts);
        }

        return lane.Name;
    }

    // A canonical strided view's tiling (VkKernels.StridedTileDim): its rank, the stride-1 dimension (−1: none), the
    // workgroups along x (the tiles of the last and that dimension) and y (the other dimensions' product), where the device
    // takes them.
    private (int N, int J, uint X, uint Y) StridedTiles(Fmt f, int[] p)
    {
        int n = p.Length / 2, j = VkKernels.StridedTileDim(f, p);
        if (j < 0) return (n, -1, 0, 0);
        long x = (long)Groups(p[n - 1], VkKernels.TransposeTile) * Groups(p[j], VkKernels.TransposeTile), y = 1;
        for (int d = 0; d < n - 1; d++) if (d != j) y *= p[d];
        return x <= vk.MaxGroupsX && y <= vk.MaxGroupsY && y > 0 ? (n, j, (uint)x, (uint)y) : (n, -1, 0, 0);
    }
}
