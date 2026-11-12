// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and the cross-entropy loss on Vulkan (Kernels.Routing.cs, Kernels.Radix.cs), CUDA's lanes: top-k and
// pick in the type's own, the loss and sampling in the float lane; a block format's values by its encoder.
using System.Numerics;
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>top-k of each row: values as the type (rows × k), indices s32; returns the lane's name.</summary>
    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var f = Fmt.Of<TN>();
        var lane = f.Natural;
        if (k > VkKernels.MaxTopK) throw new NotSupportedException($"vulkan: top-k with k = {k} exceeds the kernel's passes ({VkKernels.MaxTopK}); a partial sort is the step");
        if (rows == 0 || k == 0) return lane.Name;
        var fn = vk.Cached($"topk:{f.Name}", () => VkKernels.TopKRows(vk, f, lane));
        vk.Launch(fn, (uint)rows, 1, x, indices, Vulkan.U(cols), Vulkan.U(k));
        Gather(f, x, indices, values, rows, cols, k);
        return lane.Name;
    }

    /// <summary>The top-k of each row as log-probabilities (IRouting.LogSoftmaxTopK): TopK's indices, then
    /// VkKernels.LogProbRows (the row's largest the first chosen, Σ exp(x − largest) by the workgroup's tree, an invocation a
    /// chosen column storing (x − m) − log Σ as a lane value); stored as the type once after. Returns the lane's name.</summary>
    public string LogSoftmaxTopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IFloatAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (k > VkKernels.MaxTopK) throw new NotSupportedException($"vulkan: top-k with k = {k} exceeds the kernel's passes ({VkKernels.MaxTopK}); a partial sort is the step");
        if (rows == 0 || k == 0) return lane.Name;
        vk.Launch(vk.Cached($"topk:{f.Name}", () => VkKernels.TopKRows(vk, f, f.Natural)), (uint)rows, 1, x, indices, Vulkan.U(cols), Vulkan.U(k));
        bool own = f.Is("float32") || f.Is("float64");
        ulong lanes = own ? values : vk.Rent(Math.Max(16, (long)rows * k * lane.Size));
        try
        {
            var fn = vk.Cached($"log_probs:{f.Name}", () => VkKernels.LogProbRows(vk, f, lane));
            vk.Launch(fn, (uint)rows, 1, x, indices, lanes, Vulkan.U(cols), Vulkan.U(k), Slot(lane, f.IsBlock ? TN.Min : double.NegativeInfinity));
            if (!own) Back(f, lane, lanes, values, (long)rows * k);
        }
        finally
        {
            if (!own) vk.Return(lanes);
        }

        return lane.Name;
    }

    /// <summary>out[r] = x[r][index[r]] as the type.</summary>
    public string Pick<TN>(ulong x, int rows, int cols, ulong index, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (rows == 0) return f.Natural.Name;
        Gather(f, x, index, output, rows, cols, 1);
        return f.Natural.Name;
    }

    // out[r·k + j] = x[r][index[r·k + j]] as the type (a block format's by its encoder).
    private void Gather(Fmt f, ulong x, ulong index, ulong output, int rows, int cols, int k)
    {
        long n = (long)rows * k;
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_gather:{f.Name}", () => VkKernels.BlockGather(vk, f)), output, x, index, Vulkan.U(cols), Vulkan.U(k));
            return;
        }

        var gather = vk.Cached($"gather_by_index:{f.Name}", () => VkKernels.GatherByIndex(vk, f));
        vk.Launch(gather, Groups(f.Words(n), VkKernels.Threads), 1, x, index, output, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(k));
    }

    /// <summary>loss[r] = w[r]·(log Σ exp(x[r]) − x[r][t[r]]) as the type (t: s32 targets; weights as the type, 0 for none), in
    /// the type's float lane: a workgroup per row.</summary>
    public string CrossEntropy<TN>(ulong x, int rows, int cols, ulong targets, ulong weights, ulong output) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (rows == 0) return lane.Name;
        ulong losses = vk.Rent(rows * (long)lane.Size);
        try
        {
            bool weighted = weights != 0;
            var fn = vk.Cached($"cross_entropy:{f.Name}{(weighted ? ":w" : "")}", () => VkKernels.CrossEntropyRows(vk, f, lane, weighted));
            if (weighted) vk.Launch(fn, (uint)rows, 1, x, targets, losses, Vulkan.U(cols), weights);
            else vk.Launch(fn, (uint)rows, 1, x, targets, losses, Vulkan.U(cols));
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

    /// <summary>output[r] = 1 where row r's prediction is right, else 0 (VkKernels.ClassMatchRows): a thread per row into the
    /// lane's storage (each bound the lane's smallest value at least it), stored as the type after.</summary>
    public string ClassMatch<TN>(ulong predictions, ulong targets, ulong output, int rows, int cols, double threshold) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = f.IsBlock ? Lane.F32 : f.Natural;
        ulong Bound(double v) => lane.IsFloat ? Slot(lane, v) : lane == Lane.U32 ? (uint)v : lane == Lane.U64 ? (ulong)v : Slot(lane, v);
        ulong scratch = vk.Rent(rows * (long)lane.Size);
        try
        {
            var k = vk.Cached($"class_match:{f.Name}", () => VkKernels.ClassMatchRows(vk, f, lane));
            vk.Launch(k, Groups(rows, VkKernels.Threads), 1, predictions, targets, scratch, Vulkan.U(rows), Vulkan.U(cols), Bound(lane.AtLeast(threshold)), Bound(lane.AtLeast(0.5)));
            Back(f, lane, scratch, output, rows);
            return lane.Name;
        }
        finally
        {
            vk.Return(scratch);
        }
    }
}
