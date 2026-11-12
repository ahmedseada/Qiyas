// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on CUDA (Abstraction/Cache.cs, ICache; their kernels Ptx.Cache.cs and the decode kernel's cache
// options, Ptx.Attention.cs): the position stays on the device, read by each kernel. A write: a thread a stored word of the
// cache; a block format's cache a warp a block the write touches.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    private static ulong LaneValue(Lane lane, double v) => lane.Wide ? Gpu.D(v) : Gpu.F((float)v);

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
            // A block format cache: a row of whole blocks written a block at a time (the written rows' blocks alone); rows
            // that share blocks by re-encoding the blocks the write touches, their other values kept (every block of the
            // cache launched, each deciding from the position).
            if (c.WholeBlocks(fc))
            {
                var rfn = gpu.Cached($"cache_rows_blocks:{fc.Name}:{f.Name}", (out string e) => Ptx.CacheRowsBlocks(gpu, fc, f, out e));
                Gpu.Launch(rfn, BlockGrid(fc, (long)rows * c.D), 1, 256, 1, 0, source, cache, position, Gpu.U(rows * c.D), Gpu.U(steps), Gpu.U(c.Capacity), Gpu.U(c.D / fc.BlockValues));
            }
            else
            {
                var bfn = gpu.Cached($"cache_blocks:{fc.Name}:{f.Name}", (out string e) => Ptx.CacheBlocks(gpu, fc, f, out e));
                Gpu.Launch(bfn, BlockGrid(fc, c.Values), 1, 256, 1, 0, source, cache, position, Gpu.U((int)c.Values), Gpu.U(c.Capacity), Gpu.U(steps), Gpu.U(c.D));
            }

            return lane.Name;
        }

        var fn = gpu.Cached($"cache_write:{f.Name}:{fc.Name}:{lane.Name}", (out string e) => Ptx.CacheWrite(gpu, f, fc, lane, out e));
        Gpu.Launch(fn, Grid((long)rows * wpr, 256), 1, 256, 1, 0, source, cache, position, Gpu.U(rows), Gpu.U(steps), Gpu.U(c.Capacity), Gpu.U(c.D), Gpu.U(wpr));
        return lane.Name;
    }

    /// <summary>O for the queries at the position against the caches (ICache.AttentionCached): the decode kernel, a block a
    /// row of KvHeadCount heads of Group·Lq rows, its limits from the position on the device; returns the lane's name.</summary>
    public string AttentionCached<TN, TC>(AttentionArgs a, ulong position, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        if (a.Laid)
            return AttentionLayout.Queries<ulong, ulong, TN>(this, a, q, o, (d, h, qh, oh) => d.AttentionCached<TN, TC>(h, position, qh, k, v, oh));
        CacheRule.Validate(a);
        Fmt f = Fmt.Of<TN>(), fc = Fmt.Of<TC>();
        new CacheView(a.KvHeadCount, a.Lk, a.D).Validate(fc);
        var lane = RowLane(f);
        if (a.D > 4 * Ptx.DecodeThreads) throw new NotSupportedException($"attention_cached: a head dimension of {a.D} exceeds {4 * Ptx.DecodeThreads}");
        if (a.QValues == 0) return lane.Name;
        int dCap = (a.D + 127) / 128 * 128, rows = a.Group * a.Lq;
        var over = new Ptx.DecodeCache(fc, a.Softcap > 0);
        // A block format: Q decoded where it is stored, O the queries' f32 values, encoded by blocks after.
        ulong lo = f.IsBlock ? gpu.Rent(a.QValues * 4) : o;
        try
        {
            if ((!fc.IsBlock || a.D % 8 == 0) && DecodeSplit(f, lane, dCap, over, a.KvHeadCount * rows, a.Lk, lo, f.IsBlock ? Fmt.Of<Float32>() : null, [q, k, v],
                    [position, 0, Gpu.U(rows), Gpu.U(a.Lk), Gpu.U(a.D), LaneValue(lane, a.Scale)], [Gpu.U(a.Lq), Gpu.U(a.Window), LaneValue(lane, a.Softcap)], runs: fc.IsBlock))
            {
                if (!f.IsBlock) return lane.Name;
                Back(f, lane, lo, o, a.QValues);
                return "f32 (blocks in place)";
            }

            bool runs = fc.IsBlock && a.D % 8 == 0;                             // a block format cache's keys in runs of 8 values
            var fn = gpu.Cached($"attention_cached:{f.Name}:{dCap}:{over.Key}{(runs ? ":runs" : "")}", (out string e) => Ptx.AttentionDecode(gpu, f, lane, dCap, out e, f.IsBlock ? Fmt.Of<Float32>() : null, over, runs));
            Gpu.Launch(fn, (uint)(a.KvHeadCount * rows), 1, Ptx.DecodeThreads, 1, 0, q, k, v, lo, position, 0, Gpu.U(rows), Gpu.U(a.Lk), Gpu.U(a.D), LaneValue(lane, a.Scale),
                Gpu.U(a.Lq), Gpu.U(a.Window), LaneValue(lane, a.Softcap));
            if (!f.IsBlock) return lane.Name;
            Back(f, lane, lo, o, a.QValues);
            return "f32 (blocks in place)";
        }
        finally
        {
            if (f.IsBlock) gpu.Return(lo);
        }
    }

    /// <summary>The mask of a decoder's steps at the position (ICache.DecoderMask); returns the lane's name.</summary>
    public string DecoderMask<TN>(int rows, int capacity, int window, ulong position, ulong mask) where TN : struct, IFloatAdad<TN>
    {
        long n = CacheView.MaskValues(rows, capacity, window);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (n == 0) return lane.Name;
        var fn = gpu.Cached($"decoder_mask:{f.Name}:{lane.Name}", (out string e) => Ptx.DecoderMask(gpu, f, lane, out e));
        LaunchEach(fn, f, n, mask, position, Gpu.U((int)n), Gpu.U(capacity), Gpu.U(window), LaneValue(lane, CacheView.Excluded<TN>()));
        return lane.Name;
    }

    /// <summary>position[0] += steps on the device.</summary>
    public void PositionAdvance(ulong position, int steps)
    {
        var fn = gpu.Cached("position_advance", (out string e) => Ptx.PositionAdvance(gpu, out e));
        Gpu.Launch(fn, 1, 1, 1, 1, 0, position, Gpu.U(steps));
    }
}
