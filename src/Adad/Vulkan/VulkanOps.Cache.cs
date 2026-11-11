// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on Vulkan (Abstraction/Cache.cs, ICache; their kernels Kernels.Cache.cs and the decode kernel's
// cache options, Kernels.Attention.cs): the position stays on the device, read by each kernel. A write: the rows' scales
// first (a scaled cache), stored as the type before the codes' kernel reads them (float32 and float64 by the scales'
// kernel; a narrower float by the words' kernel; a block format's by its encoder and the compaction of the touched blocks,
// each launched over the whole scales and deciding from the position), then the codes, an invocation a stored word.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The source's rows into the cache at the position (ICache.CacheWrite); returns the lane's name.</summary>
    public string CacheWrite<TN, TC>(CacheView c, int steps, ulong position, ulong source, ulong cache, ulong scales) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC> =>
        CacheWriteFrom<TC>(Fmt.Of<TN>(), null, c, steps, position, source, cache, scales);

    // The write with its source stored as `from` (null: f; a composition's rows in the lane's own storage), the scales f's.
    private string CacheWriteFrom<TC>(Fmt f, Fmt? from, CacheView c, int steps, ulong position, ulong source, ulong cache, ulong scales) where TC : struct, IAdad<TC>
    {
        var fc = Fmt.Of<TC>();
        string via = from is null ? "" : ":" + from.Name;
        c.Validate(fc);
        if (steps < 0 || (long)c.Heads * steps * c.D > int.MaxValue) throw new ArgumentException($"cache: a write of {steps} rows a head");
        var lane = RowLane(f);
        if (c.Heads == 0 || steps == 0 || c.D == 0 || c.Capacity == 0) return lane.Name;
        bool scaled = scales != 0;
        int rows = c.Heads * steps, wpr = fc.WordBits == 64 ? c.D : c.D * fc.Bits / 32;
        ulong min = Slot(lane, TC.Min), max = Slot(lane, TC.Max);
        if (scaled)
        {
            bool direct = Own(f);
            long n = c.Rows;
            ulong sl = direct ? scales : vk.Rent(Math.Max(16, (long)rows * lane.Size)), staging = 0;
            try
            {
                var sfn = vk.Cached($"cache_scales:{f.Name}:{lane.Name}:{direct}{via}", () => VkKernels.CacheScales(vk, f, lane, direct, from));
                vk.Launch(sfn, (uint)rows, 1, source, sl, position, Vulkan.U(steps), Vulkan.U(c.Capacity), Vulkan.U(c.D), max);
                if (f.IsBlock)
                {
                    long blocks = (n + f.BlockValues - 1) / f.BlockValues;
                    staging = vk.Rent(blocks * VkKernels.BlockSlotWords(f) * 4L);
                    var slots = new ulong[Spv.MaxSlots];
                    slots[0] = sl; slots[1] = scales; slots[2] = position; slots[3] = Vulkan.U(c.Capacity); slots[4] = Vulkan.U(steps);
                    slots[VkKernels.StagingSlot] = staging;
                    slots[VkKernels.CountSlot] = (ulong)n;
                    vk.Launch(vk.Cached($"cache_scales_encode:{f.Name}:{lane.Name}", () => VkKernels.CacheScalesEncoder(vk, f, lane)), Groups(blocks, VkKernels.Threads), 1, slots);
                    vk.Launch(vk.Cached($"cache_scales_compact:{f.Name}", () => VkKernels.CacheScalesCompact(vk, f)), Groups(f.Words(n), VkKernels.Threads), 1,
                        staging, scales, Vulkan.U(blocks), position, Vulkan.U(n), Vulkan.U(c.Capacity), Vulkan.U(steps));
                }
                else if (!direct)
                {
                    vk.Launch(vk.Cached($"cache_scales_words:{f.Name}:{lane.Name}", () => VkKernels.CacheScalesWords(vk, f, lane)), Groups(f.Words(n), VkKernels.Threads), 1,
                        sl, scales, position, Vulkan.U(n), Vulkan.U(c.Capacity), Vulkan.U(steps));
                }
            }
            finally
            {
                if (!direct) vk.Return(sl);
                if (staging != 0) vk.Return(staging);
            }
        }

        var fn = vk.Cached($"cache_write:{f.Name}:{fc.Name}:{lane.Name}:{scaled}{via}", () => VkKernels.CacheWrite(vk, f, fc, lane, scaled, from));
        vk.Launch(fn, Groups((long)rows * wpr, VkKernels.Threads), 1, source, cache, scales, position, Vulkan.U(rows), Vulkan.U(steps), Vulkan.U(c.Capacity), Vulkan.U(c.D), Vulkan.U(wpr), min, max);
        return lane.Name;
    }

    /// <summary>A layer's projection rows, normalized and rotated in the lane, into the cache at the position
    /// (ICache.CacheWriteRope): the rope kernel's rows a head at a time in the lane's own storage (not rounded), then
    /// CacheWrite from them. Returns the lane's name.</summary>
    public string CacheWriteRope<TN, TC>(CacheView c, RopeArgs a, ulong position, ulong x, ulong gain, ulong cos, ulong sin, ulong positions, ulong cache, ulong scales)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        var f = Fmt.Of<TN>();
        c.Validate(Fmt.Of<TC>());
        c.ValidateRope(a);
        var lane = RowLane(f);
        long n = (long)a.Rows * a.Cols;
        if (n == 0) return lane.Name;
        ulong rows = vk.Rent(Math.Max(16, n * lane.Size));
        try
        {
            RopeTo(a with { HeadsFirst = true }, f, LaneFormat(lane), x, gain, cos, sin, positions, rows);
            return CacheWriteFrom<TC>(f, LaneFormat(lane), c, a.Steps, position, rows, cache, scales);
        }
        finally { vk.Return(rows); }
    }

    /// <summary>O for the queries at the position against the caches (ICache.AttentionCached): the decode kernel, a workgroup
    /// a row of KvHeadCount heads of Group·Lq rows, its limits from the position on the device; returns the lane's name.</summary>
    public string AttentionCached<TN, TC>(AttentionArgs a, ulong position, ulong q, ulong k, ulong kScales, ulong v, ulong vScales, ulong o) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        if (a.StepsFirst)
            return AttentionLayout.Queries<ulong, ulong, TN>(this, a, q, o, (d, h, qh, oh) => d.AttentionCached<TN, TC>(h, position, qh, k, kScales, v, vScales, oh));
        CacheRule.Validate(a);
        Fmt f = Fmt.Of<TN>(), fc = Fmt.Of<TC>();
        new CacheView(a.KvHeadCount, a.Lk, a.D).Validate(fc);
        var lane = RowLane(f);
        if (a.D > 4 * VkKernels.DecodeThreads) throw new NotSupportedException($"vulkan: attention over a cache with a head dimension of {a.D} exceeds {4 * VkKernels.DecodeThreads}");
        if (a.QValues == 0) return lane.Name;
        int dims = (a.D + VkKernels.DecodeThreads - 1) / VkKernels.DecodeThreads * VkKernels.DecodeThreads, rows = a.Group * a.Lq;
        var over = new VkKernels.DecodeCache(fc, kScales != 0, vScales != 0, a.Softcap > 0);
        ulong lo = Own(f) ? o : vk.Rent(a.QValues * lane.Size);
        try
        {
            var fn = vk.Cached($"attention_cached:{f.Name}:{dims}:{over.Key}", () => VkKernels.AttentionDecode(vk, f, lane, dims, over));
            vk.Launch(fn, (uint)(a.KvHeadCount * rows), 1, q, k, v, lo, position, 0, Vulkan.U(rows), Vulkan.U(a.Lk), Vulkan.U(a.D), Slot(lane, a.Scale),
                kScales, vScales, Vulkan.U(a.Lq), Vulkan.U(a.Window), Slot(lane, a.Softcap));
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
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_decoder_mask:{f.Name}", () => VkKernels.BlockDecoderMask(vk, f)), mask, position, Vulkan.U(capacity), Vulkan.U(window), Vulkan.F((float)TN.Min));
            return lane.Name;
        }

        var fn = vk.Cached($"decoder_mask:{f.Name}:{lane.Name}", () => VkKernels.DecoderMask(vk, f, lane));
        vk.Launch(fn, Groups(f.Words(n), VkKernels.Threads), 1, mask, position, Vulkan.U((int)n), Vulkan.U(capacity), Vulkan.U(window), Slot(lane, TN.Min));
        return lane.Name;
    }

    /// <summary>position[0] += steps on the device.</summary>
    public void PositionAdvance(ulong position, int steps)
    {
        var fn = vk.Cached("position_advance", () => VkKernels.PositionAdvance(vk));
        vk.Launch(fn, 1, 1, position, Vulkan.U(steps));
    }
}
