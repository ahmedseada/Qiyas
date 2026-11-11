// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of the key and value caches (ICache): their view, the rule of the rows a query at the device's
// position sees, and the references. The devices implement it in their folders: Cpu/Cache.cs; Cuda/GpuOps.Cache.cs with
// its kernels in Cuda/Ptx.Cache.cs; Vulkan/VulkanOps.Cache.cs with its kernels in Vulkan/Kernels.Cache.cs.

namespace Adad.Abstraction;

/// <summary>A key or value cache: Heads × Capacity rows of D values stored as the cache's type, filled from row 0 up to the
/// position the device holds. A scaled cache's rows are codes, each with a scale (stored as the values' type, at
/// head·Capacity + row): value = code·scale. A row of a plain type is whole 32-bit words (D·bits a multiple of 32). A
/// block format's rows are its blocks' values (no row scales: its values are scaled by their blocks): whole blocks where D
/// is a multiple of the format's block, written a block at a time; else rows share blocks, and a write re-encodes each
/// block it touches with the block's other values kept (the type's partial store: lossy, as the type is).</summary>
public sealed record CacheView(int Heads, int Capacity, int D)
{
    public long Values => (long)Heads * Capacity * D;

    /// <summary>The rows (and a scaled cache's scales).</summary>
    public long Rows => (long)Heads * Capacity;

    public void Validate(Fmt fc, bool scaled = false)
    {
        if (Heads < 0 || Capacity < 0 || D < 0) throw new ArgumentException($"cache: {Heads} heads of {Capacity} rows of {D}");
        if (fc.IsBlock)
        {
            if (scaled) throw new ArgumentException($"cache: codes of the block format {fc.Name} with row scales (a block format's values are scaled by their blocks)");
        }
        else if ((long)D * fc.Bits % 32 != 0) throw new ArgumentException($"cache: a row of {D} values of {fc.Name} is not whole 32-bit words");

        if (Values > int.MaxValue) throw new ArgumentException("cache: past 2^31 values");
    }

    /// <summary>Whether a row of the cache is whole blocks of the block format fc (a write then touches no other row's block).</summary>
    public bool WholeBlocks(Fmt fc) => fc.IsBlock && D % fc.BlockValues == 0;

    /// <summary>A write of a layer's projection rows (ICache.CacheWriteRope): a.Rows = Heads·a.Steps rows of D values, a step
    /// at a time, the cache's heads the batches' heads (Heads a whole multiple of a.Heads).</summary>
    public void ValidateRope(RopeArgs a)
    {
        if (a.Cols != D || a.Heads <= 0 || a.Steps < 0 || Heads % a.Heads != 0 || a.Rows != (long)Heads * a.Steps || 2 * a.Half > a.Cols || a.Inverse)
            throw new ArgumentException($"cache: {a.Rows} projection rows of {a.Cols} ({a.Steps} steps of {a.Heads} heads, half {a.Half}{(a.Inverse ? ", inverse" : "")}) into {Heads} heads of rows of {D}");
    }

    /// <summary>The rows of each head a write of steps rows at the position fills: [First, End) = [max(position, 0),
    /// min(position + steps, Capacity)) (empty: End ≤ First).</summary>
    public (int First, int End) Written(int position, int steps) => (Math.Max(position, 0), (int)Math.Min((long)position + steps, Capacity));

    /// <summary>The mask of rows decoder steps over capacity positions (ICache.DecoderMask): a count of values.</summary>
    public static long MaskValues(int rows, int capacity, int window)
    {
        if (rows < 0 || capacity < 0 || window < 0) throw new ArgumentException($"decoder mask: {rows} rows of {capacity}, a window of {window}");
        long n = (long)rows * capacity;
        if (n > int.MaxValue) throw new ArgumentException("decoder mask: past 2^31 values");
        return n;
    }

    /// <summary>Whether step i at position p sees position j (ICache.DecoderMask's rule).</summary>
    public static bool Sees(long position, int i, int j, int window) => j <= position + i && (window == 0 || j > position + i - window);

    /// <summary>The float64 reference of a write's rows: each source row's (Heads × steps × D) scale max |x| / max, and the
    /// cache row it goes to (head·Capacity + position + t; −1 past Capacity, not written).</summary>
    public (double[] Scales, int[] Rows) WriteRows(double[] source, int steps, int position, double max)
    {
        var scales = new double[Heads * steps];
        var rows = new int[Heads * steps];
        for (int h = 0; h < Heads; h++)
            for (int t = 0; t < steps; t++)
            {
                double amax = 0;
                for (int c = 0; c < D; c++) amax = Math.Max(amax, Math.Abs(source[((long)h * steps + t) * D + c]));
                scales[h * steps + t] = amax / max;
                rows[h * steps + t] = position + t < Capacity && position + t >= 0 ? h * Capacity + position + t : -1;
            }

        return (scales, rows);
    }
}

/// <summary>What attention over a cache at the device's position sees (ICache.AttentionCached).</summary>
public static class CacheRule
{
    /// <summary>The attention of Lq queries a head at positions position + i against caches of a.Lk = Capacity rows, as the
    /// grouped heads' rows (AttentionArgs.Grouped) with each row's keys as its span and start: row r (query i = r % Lq)
    /// sees rows [max(0, limit − Window) when a window, limit) with limit = min(position + i + 1, Capacity); a: causal,
    /// no spans or starts of its own.</summary>
    public static AttentionArgs AtPosition(AttentionArgs a, int position)
    {
        Validate(a);
        int rows = a.Group * a.Lq;
        var spans = new int[rows];
        int[]? starts = a.Window > 0 ? new int[rows] : null;
        for (int r = 0; r < rows; r++)
        {
            spans[r] = Limit(a, position, r % Math.Max(1, a.Lq));
            if (starts is not null) starts[r] = Math.Max(0, spans[r] - a.Window);
        }

        return new AttentionArgs(a.KvHeadCount, rows, a.Lk, a.D, a.Scale, false, spans, starts) { Softcap = a.Softcap };
    }

    public static int Limit(AttentionArgs a, int position, int i) => (int)Math.Clamp((long)position + i + 1, 0, a.Lk);

    public static void Validate(AttentionArgs a)
    {
        a.Validate();
        if (!a.Causal || a.Spans is not null || a.SpanStarts is not null) throw new ArgumentException("attention over a cache: causal, its rows from the position (no spans or starts of its own)");
    }
}
