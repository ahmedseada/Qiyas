// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and the cross-entropy loss on Vulkan (Kernels.Routing.cs, Kernels.Radix.cs), CUDA's lanes: top-k and
// pick in the type's own, the loss and sampling in the float lane. The block formats are refused (no Vulkan block
// encoder yet).
using System.Numerics;
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>top-k of each row: values as the type (rows × k), indices s32; returns the lane's name.</summary>
    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"top-k of {f.Name} (a block format)");
        var lane = f.Natural;
        if (k > VkKernels.MaxTopK) throw new NotSupportedException($"vulkan: top-k with k = {k} exceeds the kernel's passes ({VkKernels.MaxTopK}); a partial sort is the step");
        if (rows == 0 || k == 0) return lane.Name;
        var fn = vk.Cached($"topk:{f.Name}", () => VkKernels.TopKRows(vk, f, lane));
        vk.Launch(fn, (uint)rows, 1, x, indices, Vulkan.U(cols), Vulkan.U(k));
        var gather = vk.Cached($"gather_by_index:{f.Name}", () => VkKernels.GatherByIndex(vk, f));
        vk.Launch(gather, Groups(f.Words((long)rows * k), VkKernels.Threads), 1, x, indices, values, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(k));
        return lane.Name;
    }

    /// <summary>out[r] = x[r][index[r]] as the type.</summary>
    public string Pick<TN>(ulong x, int rows, int cols, ulong index, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"pick of {f.Name} (a block format)");
        if (rows == 0) return f.Natural.Name;
        var gather = vk.Cached($"gather_by_index:{f.Name}", () => VkKernels.GatherByIndex(vk, f));
        vk.Launch(gather, Groups(f.Words(rows), VkKernels.Threads), 1, x, index, output, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(1));
        return f.Natural.Name;
    }

    /// <summary>loss[r] = log Σ exp(x[r]) − x[r][t[r]] as the type (t: s32 targets), in the type's float lane: a workgroup per row.</summary>
    public string CrossEntropy<TN>(ulong x, int rows, int cols, ulong targets, ulong output) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"cross entropy of {f.Name} (a block format)");
        var lane = RowLane(f);
        if (rows == 0) return lane.Name;
        ulong losses = vk.Rent(rows * (long)lane.Size);
        try
        {
            var fn = vk.Cached($"cross_entropy:{f.Name}", () => VkKernels.CrossEntropyRows(vk, f, lane));
            vk.Launch(fn, (uint)rows, 1, x, targets, losses, Vulkan.U(cols));
            Back(f, lane, losses, output, rows);
        }
        finally
        {
            vk.Return(losses);
        }

        return lane.Name;
    }

    /// <summary>The sampled column of each row from weights x and a uniform draw u per row (as the type); in the float lane.</summary>
    public string Sample<TN>(ulong x, int rows, int cols, ulong u, ulong picks) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"sample of {f.Name} (a block format)");
        var lane = RowLane(f);
        if (rows == 0) return lane.Name;
        var fn = vk.Cached($"sample:{f.Name}", () => VkKernels.SampleRows(vk, f, lane));
        vk.Launch(fn, (uint)rows, 1, x, u, picks, Vulkan.U(cols));
        return lane.Name;
    }

    /// <summary>counts[b], offsets[b + 1] and the sources' order by bucket (stable), from n indices into `buckets`.</summary>
    public void Route(ulong index, int count, int buckets, ulong counts, ulong offsets, ulong order) => Order(index, count, buckets, counts, offsets, order);

    // The sources' counts by bucket (integer atomics), their offsets (a scan) and their stable order by bucket (the radix
    // sort of Kernels.Radix.cs): route's, and scatter-add's ordering. Every index within [0, buckets).
    private void Order(ulong index, int n, int buckets, ulong counts, ulong offsets, ulong order)
    {
        vk.Clear(counts, Math.Max(1, buckets) * 4L);
        if (n > 0)
        {
            var count = vk.Cached("scatter_count", () => VkKernels.ScatterCount(vk));
            vk.Launch(count, Groups(n, VkKernels.Threads), 1, index, counts, Vulkan.U(n), Vulkan.U(buckets));
        }

        var scan = vk.Cached("scatter_scan", () => VkKernels.ScatterScan(vk));
        vk.Launch(scan, 1, 1, counts, offsets, Vulkan.U(buckets));
        if (n == 0) return;
        int bits = buckets <= 1 ? 0 : 32 - BitOperations.LeadingZeroCount((uint)(buckets - 1));
        int passes = Math.Max(1, (bits + 7) / 8), tiles = (n + VkKernels.RadixTile - 1) / VkKernels.RadixTile;
        var histogram = vk.Cached("radix_histogram", () => VkKernels.RadixHistogram(vk));
        var prefix = vk.Cached("radix_scan", () => VkKernels.RadixScan(vk));
        var scatter = vk.Cached("radix_scatter", () => VkKernels.RadixScatter(vk));
        ulong hist = vk.Rent((long)VkKernels.RadixDigits * tiles * 4), bases = vk.Rent(VkKernels.RadixDigits * 4L);
        ulong keysA = vk.Rent(n * 4L), keysB = vk.Rent(n * 4L), valuesA = vk.Rent(n * 4L), valuesB = vk.Rent(n * 4L);
        try
        {
            ulong keys = index, values = 0;
            for (int p = 0; p < passes; p++)
            {
                ulong keysOut = p % 2 == 0 ? keysA : keysB, valuesOut = p == passes - 1 ? order : p % 2 == 0 ? valuesA : valuesB;
                vk.Launch(histogram, (uint)tiles, 1, keys, Vulkan.U(n), Vulkan.U(8 * p), hist, Vulkan.U(tiles));
                vk.Launch(prefix, 1, 1, hist, Vulkan.U(tiles), bases);
                vk.Launch(scatter, (uint)tiles, 1, keys, values, Vulkan.U(n), Vulkan.U(8 * p), hist, Vulkan.U(tiles), bases, keysOut, valuesOut);
                keys = keysOut;
                values = valuesOut;
            }
        }
        finally
        {
            vk.Return(hist); vk.Return(bases);
            vk.Return(keysA); vk.Return(keysB); vk.Return(valuesA); vk.Return(valuesB);
        }
    }
}
