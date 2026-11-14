// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on Vulkan (Abstraction/Cache.cs, ICache; their kernels Kernels.Cache.cs and the decode kernel's
// cache options, Kernels.Attention.cs): the position stays on the device, read by each kernel. A write: an invocation a
// stored word of the cache; a block format's cache by its encoder into staging (the staged encoders, not the direct
// ones) and the compaction of the touched words.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The source's rows into the cache at the position (ICache.CacheWrite); returns the lane's name.</summary>
    public string CacheWrite<TN, TC>(CacheView c, int steps, ulong position, ulong source, ulong cache) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        Fmt f = Fmt.Of<TN>(), fc = Fmt.Of<TC>();
        c.Validate(fc);
        if (steps < 0 || (long)c.Heads * steps * c.D > int.MaxValue) throw new ArgumentException($"cache: a write of {steps} rows a head");
        var lane = RowLane(f);
        if (c.Heads == 0 || steps == 0 || c.D == 0 || c.Capacity == 0) return lane.Name;
        int rows = c.Heads * steps, wpr = fc.WordBits == 64 ? c.D : c.D * fc.Bits / 32;
        if (fc.IsBlock)
        {
            BlockRows(fc, f, c, steps, rows, position, source, cache);
            return lane.Name;
        }

        var fn = vk.Cached($"cache_write:{f.Name}:{fc.Name}:{lane.Name}", () => VkKernels.CacheWrite(vk, f, fc, lane));
        vk.Launch(fn, Groups((long)rows * wpr, VkKernels.Threads), 1, source, cache, position, Vulkan.U(rows), Vulkan.U(steps), Vulkan.U(c.Capacity), Vulkan.U(c.D), Vulkan.U(wpr));
        return lane.Name;
    }

    // A block format cache's write. A row of whole blocks: the written rows' blocks encoded into staging in the source's
    // order, then written into the cache's words the rows cover (invocations for the rows' words alone). Rows that share
    // blocks: the cache's blocks encoded from the written rows' values and the others as stored, and the touched blocks
    // alone compacted (every block of the cache launched, each deciding from the position).
    private void BlockRows(Fmt fc, Fmt fx, CacheView c, int steps, int rows, ulong position, ulong source, ulong cache)
    {
        ulong staging = 0;
        try
        {
            if (c.WholeBlocks(fc))
            {
                int bpr = c.D / fc.BlockValues, rowBytes = bpr * fc.BlockBytes;
                long blocks = (long)rows * bpr, bytes = c.Rows * rowBytes;
                if (bytes > uint.MaxValue) throw new NotSupportedException($"vulkan: a cache of {bytes} bytes of {fc.Name} past 2^32");
                staging = vk.Rent(blocks * VkKernels.BlockSlotWords(fc) * 4L);
                var slots = new ulong[Spv.MaxSlots];
                slots[0] = source;
                slots[VkKernels.StagingSlot] = staging;
                slots[VkKernels.CountSlot] = (ulong)((long)rows * c.D);
                vk.Launch(vk.Cached($"block_convert_staged:{fx.Name}:{fc.Name}", () => VkKernels.BlockConvert(vk, fx, fc, staged: true)), Groups(blocks, VkKernels.Threads), 1, slots);
                long wph = ((long)steps * rowBytes + 3) / 4 + 1;
                vk.Launch(vk.Cached($"cache_rows_compact:{fc.Name}", () => VkKernels.CacheRowsCompact(vk, fc)), Groups(c.Heads * wph, VkKernels.Threads), 1,
                    staging, cache, position, Vulkan.U(c.Heads), Vulkan.U(steps), Vulkan.U(c.Capacity), Vulkan.U(rowBytes), Vulkan.U(wph));
            }
            else
            {
                long n = c.Values, blocks = (n + fc.BlockValues - 1) / fc.BlockValues;
                staging = vk.Rent(blocks * VkKernels.BlockSlotWords(fc) * 4L);
                var slots = new ulong[Spv.MaxSlots];
                slots[0] = source; slots[1] = cache; slots[2] = position; slots[3] = Vulkan.U(c.Capacity); slots[4] = Vulkan.U(steps); slots[5] = Vulkan.U(c.D);
                slots[VkKernels.StagingSlot] = staging;
                slots[VkKernels.CountSlot] = (ulong)n;
                vk.Launch(vk.Cached($"cache_blocks_encode:{fc.Name}:{fx.Name}", () => VkKernels.CacheBlocksEncoder(vk, fc, fx)), Groups(blocks, VkKernels.Threads), 1, slots);
                vk.Launch(vk.Cached($"cache_blocks_compact:{fc.Name}", () => VkKernels.CacheBlocksCompact(vk, fc)), Groups(fc.Words(n), VkKernels.Threads), 1,
                    staging, cache, Vulkan.U(blocks), position, Vulkan.U(n), Vulkan.U(c.Capacity), Vulkan.U(steps), Vulkan.U(c.D));
            }
        }
        finally
        {
            if (staging != 0) vk.Return(staging);
        }
    }

    /// <summary>O for the queries at the position against the caches (ICache.AttentionCached): the decode kernel, a workgroup
    /// a row of KvHeadCount heads of Group·Lq rows, its limits from the position on the device; returns the lane's name.</summary>
    public string AttentionCached<TN, TC>(AttentionArgs a, ulong position, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        if (a.Laid)
            return AttentionLayout.Queries<ulong, ulong, TN>(this, a, q, o, (d, h, qh, oh) => d.AttentionCached<TN, TC>(h, position, qh, k, v, oh));
        CacheRule.Validate(a);
        Fmt f = Fmt.Of<TN>(), fc = Fmt.Of<TC>();
        new CacheView(a.KvHeadCount, a.Lk, a.D).Validate(fc);
        var lane = RowLane(f);
        if (a.D > 4 * VkKernels.DecodeThreads) throw new NotSupportedException($"vulkan: attention over a cache with a head dimension of {a.D} exceeds {4 * VkKernels.DecodeThreads}");
        if (a.QValues == 0) return lane.Name;
        int dims = (a.D + VkKernels.DecodeThreads - 1) / VkKernels.DecodeThreads * VkKernels.DecodeThreads, rows = a.Group * a.Lq;
        var over = new VkKernels.DecodeCache(fc, a.Softcap > 0);
        ulong lo = Own(f) ? o : vk.Rent(a.QValues * lane.Size);
        try
        {
            bool runs = fc.IsBlock && a.D % 8 == 0;             // a block format cache's keys in runs of 8 values
            var fn = vk.Cached($"attention_cached:{f.Name}:{dims}:{over.Key}{(runs ? ":runs" : "")}", () => VkKernels.AttentionDecode(vk, f, lane, dims, over, runs));
            vk.Launch(fn, (uint)(a.KvHeadCount * rows), 1, q, k, v, lo, position, 0, Vulkan.U(rows), Vulkan.U(a.Lk), Vulkan.U(a.D), Slot(lane, a.Scale),
                Vulkan.U(a.Lq), Vulkan.U(a.Window), Slot(lane, a.Softcap));
            if (!Own(f)) Back(f, lane, lo, o, a.QValues);
            return lane.Name;
        }
        finally
        {
            if (!Own(f)) vk.Return(lo);
        }
    }

    /// <summary>The mask of a decoder's steps at the position (ICache.DecoderMask); returns the lane's name.</summary>
    public string DecoderMask<TN>(int rows, int capacity, int window, ulong position, ulong mask) where TN : struct, IFloatAdad<TN>
    {
        long n = CacheView.MaskValues(rows, capacity, window);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (n == 0) return lane.Name;
        double excluded = CacheView.Excluded<TN>();
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_decoder_mask:{f.Name}", () => VkKernels.BlockDecoderMask(vk, f)), mask, position, Vulkan.U(capacity), Vulkan.U(window), Vulkan.F((float)excluded));
            return lane.Name;
        }

        var fn = vk.Cached($"decoder_mask:{f.Name}:{lane.Name}", () => VkKernels.DecoderMask(vk, f, lane));
        vk.Launch(fn, Groups(f.Words(n), VkKernels.Threads), 1, mask, position, Vulkan.U((int)n), Vulkan.U(capacity), Vulkan.U(window), Slot(lane, excluded));
        return lane.Name;
    }

    /// <summary>position[0] += steps on the device.</summary>
    public void PositionAdvance(ulong position, int steps)
    {
        var fn = vk.Cached("position_advance", () => VkKernels.PositionAdvance(vk));
        vk.Launch(fn, 1, 1, position, Vulkan.U(steps));
    }
}
