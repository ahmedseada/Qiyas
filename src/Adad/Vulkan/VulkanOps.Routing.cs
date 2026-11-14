// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and the cross-entropy loss on Vulkan (Kernels.Routing.cs, Kernels.Radix.cs), CUDA's lanes: top-k and
// pick in the type's own, the loss and sampling in the float lane; a block format's values by its encoder.
using System.Numerics;
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>top-k of each row in Adad's order: values as the type (rows × k), indices s32; returns the lane's name.</summary>
    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var f = Fmt.Of<TN>();
        var lane = f.Natural;
        if (k > VkKernels.MaxTopK) throw new NotSupportedException($"vulkan: top-k with k = {k} exceeds the kernel's passes ({VkKernels.MaxTopK}); a partial sort is the step");
        if (rows == 0 || k == 0) return lane.Name;
        TopKIndices(f, x, rows, cols, k, indices);
        Gather(f, x, indices, values, rows, cols, k);
        return lane.Name;
    }

    // The top-k columns of each row (s32), in the type's own lane: the row read once for k up to TopKLocalMax, else k passes.
    private void TopKIndices(Fmt f, ulong x, int rows, int cols, int k, ulong indices)
    {
        var lane = f.Natural;
        if (k <= VkKernels.TopKLocalMax)
        {
            int cap = VkKernels.TopKCap(k);
            var local = vk.Cached($"topk_local:{f.Name}:{cap}", () => VkKernels.TopKRowsLocal(vk, f, lane, cap));
            vk.Launch(local, (uint)rows, 1, x, indices, Vulkan.U(cols), Vulkan.U(k));
            return;
        }

        var fn = vk.Cached($"topk:{f.Name}", () => VkKernels.TopKRows(vk, f, lane));
        vk.Launch(fn, (uint)rows, 1, x, indices, Vulkan.U(cols), Vulkan.U(k));
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
        TopKIndices(f, x, rows, cols, k, indices);
        bool own = f.Is("float32") || f.Is("float64");
        ulong lanes = own ? values : vk.Rent(Math.Max(16, (long)rows * k * lane.Size));
        try
        {
            var fn = vk.Cached($"log_probs:{f.Name}", () => VkKernels.LogProbRows(vk, f, lane));
            vk.Launch(fn, (uint)rows, 1, x, indices, lanes, Vulkan.U(cols), Vulkan.U(k));
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
    // sort of Kernels.Radix.cs): route's, and scatter-add's ordering. An index outside [0, buckets) is in no bucket: not
    // counted, offsets[buckets] the sources with one, the order −1 past them (IRouting.Route).
    private void Order(ulong index, int n, int buckets, ulong counts, ulong offsets, ulong order)
    {
        int bits = buckets <= 1 ? 0 : 32 - BitOperations.LeadingZeroCount((uint)(buckets - 1));
        int passes = Math.Max(1, (bits + 7) / 8), tiles = (n + VkKernels.RadixTile - 1) / VkKernels.RadixTile;
        // One pass (buckets ≤ 256: a key its own digit): the counts and offsets are the pass's digit totals and bases,
        // written by its bases' scan; more passes count the buckets first.
        bool onePass = passes == 1 && n > 0;
        if (!onePass)
        {
            vk.Clear(counts, Math.Max(1, buckets) * 4L);
            if (n > 0)
            {
                var count = vk.Cached("scatter_count", () => VkKernels.ScatterCount(vk));
                vk.Launch(count, Groups(n, VkKernels.Threads), 1, index, counts, Vulkan.U(n), Vulkan.U(buckets));
            }

            var scan = vk.Cached("offsets_scan", () => VkKernels.OffsetsScan(vk));
            vk.Launch(scan, 1, 1, counts, offsets, Vulkan.U(buckets));
        }

        if (n == 0) return;
        ulong kept = offsets + (ulong)(buckets * 4L);
        var histogram = vk.Cached("radix_histogram", () => VkKernels.RadixHistogram(vk));
        var prefix = vk.Cached("radix_scan_tiles", () => VkKernels.RadixScanTiles(vk));
        var digitBases = vk.Cached("radix_scan_bases", () => VkKernels.RadixScanBases(vk));
        var bucketBases = vk.Cached("radix_scan_bases_buckets", () => VkKernels.RadixScanBases(vk, buckets: true));
        var scatter = vk.Cached("radix_scatter", () => VkKernels.RadixScatter(vk));
        ulong hist = vk.Rent((long)VkKernels.RadixDigits * tiles * 4), bases = vk.Rent(VkKernels.RadixDigits * 4L), totals = vk.Rent(VkKernels.RadixDigits * 4L);
        ulong keysA = vk.Rent(n * 4L), keysB = vk.Rent(n * 4L), valuesA = vk.Rent(n * 4L), valuesB = vk.Rent(n * 4L);
        try
        {
            ulong keys = index, values = 0;
            for (int p = 0; p < passes; p++)
            {
                ulong keysOut = p % 2 == 0 ? keysA : keysB, valuesOut = p == passes - 1 ? order : p % 2 == 0 ? valuesA : valuesB;
                // The first pass drops the indices outside [0, buckets); the later ones take the kept (offsets[buckets]).
                ulong valid = p == 0 ? 0 : kept, tail = p == passes - 1 ? kept : 0;
                vk.Launch(histogram, (uint)tiles, 1, keys, Vulkan.U(n), Vulkan.U(8 * p), hist, Vulkan.U(tiles), Vulkan.U(buckets), valid);
                vk.Launch(prefix, VkKernels.RadixDigits, 1, hist, Vulkan.U(tiles), totals);
                if (onePass) vk.Launch(bucketBases, 1, 1, totals, bases, counts, offsets, Vulkan.U(buckets));
                else vk.Launch(digitBases, 1, 1, totals, bases);
                vk.Launch(scatter, (uint)tiles, 1, keys, values, Vulkan.U(n), Vulkan.U(8 * p), hist, Vulkan.U(tiles), bases, keysOut, valuesOut, Vulkan.U(buckets), valid, tail);
                keys = keysOut;
                values = valuesOut;
            }
        }
        finally
        {
            vk.Return(hist); vk.Return(bases); vk.Return(totals);
            vk.Return(keysA); vk.Return(keysB); vk.Return(valuesA); vk.Return(valuesB);
        }
    }
}
