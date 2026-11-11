// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on CUDA (Abstraction/Cache.cs, ICache; their kernels Ptx.Cache.cs and the decode kernel's cache
// options, Ptx.Attention.cs): the position stays on the device, read by each kernel. A write: the rows' scales first (a
// scaled cache), stored as the type before the codes' kernel reads them; then the codes, a thread a stored word. A block
// format's row scales: lane values first, then the scales' blocks the write touches re-encoded (the whole buffer's blocks
// launched, each deciding from the position whether it holds a written row).
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    private static ulong LaneValue(Lane lane, double v) => lane.Wide ? Gpu.D(v) : Gpu.F((float)v);

    /// <summary>The source's rows into the cache at the position (ICache.CacheWrite); returns the lane's name.</summary>
    public string CacheWrite<TN, TC>(CacheView c, int steps, ulong position, ulong source, ulong cache, ulong scales) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC> =>
        CacheWriteFrom<TC>(Fmt.Of<TN>(), null, c, steps, position, source, cache, scales);

    // The write with its source stored as `from` (null: f; a composition's rows in the lane's own storage), the scales f's.
    private string CacheWriteFrom<TC>(Fmt f, Fmt? from, CacheView c, int steps, ulong position, ulong source, ulong cache, ulong scales) where TC : struct, IAdad<TC>
    {
        var fc = Fmt.Of<TC>();
        string via = from is null ? "" : ":" + from.Name;
        c.Validate(fc, scales != 0);
        if (steps < 0 || (long)c.Heads * steps * c.D > int.MaxValue) throw new ArgumentException($"cache: a write of {steps} rows a head");
        var lane = RowLane(f);
        if (c.Heads == 0 || steps == 0 || c.D == 0 || c.Capacity == 0) return lane.Name;
        bool scaled = scales != 0;
        int rows = c.Heads * steps, wpr = fc.WordBits == 64 ? c.D : c.D * fc.Bits / 32;
        ulong min = LaneValue(lane, TC.Min), max = LaneValue(lane, TC.Max);
        if (scaled)
        {
            bool direct = !f.IsBlock;
            ulong sl = direct ? scales : gpu.Rent(Math.Max(16, (long)rows * lane.Size));
            try
            {
                var sfn = gpu.Cached($"cache_scales:{f.Name}:{lane.Name}:{direct}{via}", (out string e) => Ptx.CacheScales(gpu, f, lane, direct, out e, from));
                Gpu.Launch(sfn, (uint)rows, 1, 256, 1, 0, source, sl, position, Gpu.U(steps), Gpu.U(c.Capacity), Gpu.U(c.D), max);
                if (!direct)
                {
                    var bfn = gpu.Cached($"cache_blocks:{f.Name}:{lane.Name}", (out string e) => Ptx.CacheBlocks(gpu, f, LaneFormat(lane), out e));
                    Gpu.Launch(bfn, BlockGrid(f, c.Rows), 1, 256, 1, 0, sl, scales, position, Gpu.U((int)c.Rows), Gpu.U(c.Capacity), Gpu.U(steps), Gpu.U(1));
                }
            }
            finally
            {
                if (!direct) gpu.Return(sl);
            }
        }

        if (fc.IsBlock)
        {
            // A block format cache: a row of whole blocks written a block at a time (the written rows' blocks alone); rows
            // that share blocks by re-encoding the blocks the write touches, their other values kept (every block of the
            // cache launched, each deciding from the position).
            var fx = from ?? f;
            if (c.WholeBlocks(fc))
            {
                var rfn = gpu.Cached($"cache_rows_blocks:{fc.Name}:{fx.Name}", (out string e) => Ptx.CacheRowsBlocks(gpu, fc, fx, out e));
                Gpu.Launch(rfn, BlockGrid(fc, (long)rows * c.D), 1, 256, 1, 0, source, cache, position, Gpu.U(rows * c.D), Gpu.U(steps), Gpu.U(c.Capacity), Gpu.U(c.D / fc.BlockValues));
            }
            else
            {
                var bfn = gpu.Cached($"cache_blocks:{fc.Name}:{fx.Name}", (out string e) => Ptx.CacheBlocks(gpu, fc, fx, out e));
                Gpu.Launch(bfn, BlockGrid(fc, c.Values), 1, 256, 1, 0, source, cache, position, Gpu.U((int)c.Values), Gpu.U(c.Capacity), Gpu.U(steps), Gpu.U(c.D));
            }

            return lane.Name;
        }

        var fn = gpu.Cached($"cache_write:{f.Name}:{fc.Name}:{lane.Name}:{scaled}{via}", (out string e) => Ptx.CacheWrite(gpu, f, fc, lane, scaled, out e, from));
        Gpu.Launch(fn, Grid((long)rows * wpr, 256), 1, 256, 1, 0, source, cache, scales, position, Gpu.U(rows), Gpu.U(steps), Gpu.U(c.Capacity), Gpu.U(c.D), Gpu.U(wpr), min, max);
        return lane.Name;
    }

    /// <summary>A layer's projection rows, normalized and rotated in the lane, into the cache at the position
    /// (ICache.CacheWriteRope): the rope kernel's rows a head at a time in the lane's own storage (not rounded), then
    /// CacheWrite from them. Returns the lane's name.</summary>
    public string CacheWriteRope<TN, TC>(CacheView c, RopeArgs a, ulong position, ulong x, ulong gain, ulong cos, ulong sin, ulong positions, ulong cache, ulong scales)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        var f = Fmt.Of<TN>();
        c.Validate(Fmt.Of<TC>(), scales != 0);
        c.ValidateRope(a);
        var lane = RowLane(f);
        long n = (long)a.Rows * a.Cols;
        if (n == 0) return lane.Name;
        ulong rows = gpu.Rent(Math.Max(16, n * lane.Size));
        try
        {
            RopeTo(a with { HeadsFirst = true }, f, LaneFormat(lane), x, gain, cos, sin, positions, rows);
            return CacheWriteFrom<TC>(f, LaneFormat(lane), c, a.Steps, position, rows, cache, scales);
        }
        finally { gpu.Return(rows); }
    }

    /// <summary>O for the queries at the position against the caches (ICache.AttentionCached): the decode kernel, a block a
    /// row of KvHeadCount heads of Group·Lq rows, its limits from the position on the device; returns the lane's name.</summary>
    public string AttentionCached<TN, TC>(AttentionArgs a, ulong position, ulong q, ulong k, ulong kScales, ulong v, ulong vScales, ulong o) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        if (a.StepsFirst)
            return AttentionLayout.Queries<ulong, ulong, TN>(this, a, q, o, (d, h, qh, oh) => d.AttentionCached<TN, TC>(h, position, qh, k, kScales, v, vScales, oh));
        CacheRule.Validate(a);
        Fmt f = Fmt.Of<TN>(), fc = Fmt.Of<TC>();
        new CacheView(a.KvHeadCount, a.Lk, a.D).Validate(fc, kScales != 0 || vScales != 0);
        var lane = RowLane(f);
        if (a.D > 4 * Ptx.DecodeThreads) throw new NotSupportedException($"attention_cached: a head dimension of {a.D} exceeds {4 * Ptx.DecodeThreads}");
        if (a.QValues == 0) return lane.Name;
        int dCap = (a.D + 127) / 128 * 128, rows = a.Group * a.Lq;
        var over = new Ptx.DecodeCache(fc, kScales != 0, vScales != 0, a.Softcap > 0);
        // A block format: Q decoded where it is stored, O the queries' f32 values, encoded by blocks after.
        ulong lo = f.IsBlock ? gpu.Rent(a.QValues * 4) : o;
        try
        {
            var fn = gpu.Cached($"attention_cached:{f.Name}:{dCap}:{over.Key}", (out string e) => Ptx.AttentionDecode(gpu, f, lane, dCap, out e, f.IsBlock ? Fmt.Of<Float32>() : null, over));
            Gpu.Launch(fn, (uint)(a.KvHeadCount * rows), 1, Ptx.DecodeThreads, 1, 0, q, k, v, lo, position, 0, Gpu.U(rows), Gpu.U(a.Lk), Gpu.U(a.D), LaneValue(lane, a.Scale),
                kScales, vScales, Gpu.U(a.Lq), Gpu.U(a.Window), LaneValue(lane, a.Softcap));
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
        LaunchEach(fn, f, n, mask, position, Gpu.U((int)n), Gpu.U(capacity), Gpu.U(window), LaneValue(lane, TN.Min));
        return lane.Name;
    }

    /// <summary>position[0] += steps on the device.</summary>
    public void PositionAdvance(ulong position, int steps)
    {
        var fn = gpu.Cached("position_advance", (out string e) => Ptx.PositionAdvance(gpu, out e));
        Gpu.Launch(fn, 1, 1, 1, 1, 0, position, Gpu.U(steps));
    }
}
