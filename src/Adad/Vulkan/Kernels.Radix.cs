// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The stable order of n sources by their bucket (route's order, scatter-add's ordering) on Vulkan: a least-significant-
// digit radix sort of (bucket, source) pairs, 8 bits a pass, each pass over tiles of 256 sources:
//   RadixHistogram  a workgroup per tile: each digit's count in the tile (counted from the tile's digits in workgroup
//                   memory, no atomics), stored digit-major (hist[d · tiles + t]);
//   RadixScanTiles  a workgroup per digit: the digit's counts prefixed over the tiles (each invocation a run of tiles
//                   summed, the runs prefixed by the workgroup's scan), its total kept;
//   RadixScanBases  one workgroup: the digits' totals prefixed (the digits' bases);
//   RadixScatter    a workgroup per tile: each source to base[d] + hist[d][t] + its rank among the tile's earlier sources
//                   of its digit — stable, so the passes compose to the order by bucket, sources in order within one.
// The counts by integer atomics and the offsets by a workgroup's scan (ScatterCount, OffsetsScan). A source whose bucket is outside
// [0, buckets) (the limit; unsigned, so a negative one too) is in no place: the first pass drops it, the later passes take
// the sources with a bucket (valid: their count on the device) as their n, and the last pass writes −1 ("no source") past
// them (tail), so every value of the order is written.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int RadixTile = 256, RadixDigits = 256;

    /// <summary>Arguments: keys (s32; or the bucket index array on the first pass), n, shift, hist, tiles, limit (the
    /// buckets), valid (0, or the kept count's address: a later pass's n).</summary>
    public static SpvKernel RadixHistogram(Vulkan vk)
    {
        var b = new Spv("radix_histogram", RadixTile, vk.Controls);
        var keys = b.Arg(0);
        var n = Kept(b, b.Arg(1, K.U32), b.Arg(6));
        var shift = b.Arg(2, K.U32);
        var hist = b.Arg(3);
        var tiles = b.Arg(4, K.U32);
        var limit = b.Arg(5, K.U32);
        var tid = b.LocalIdX;
        var tile = b.GroupX;
        var i = tile * RadixTile + tid;
        var digits = b.Shared(K.U32, RadixTile);
        var key = b.Local(b.U(0xFFFFFFFF));                                          // past n or the limit: no digit
        b.If(i < n, () =>
        {
            var k = b.Bitcast(b.Load(K.S32, b.At(keys, i, 4)), K.U32);
            key.V = b.Select(k < limit, k.Shr(shift) & 255UL, b.U(0xFFFFFFFF));
        });
        digits[tid] = key.V;
        b.Barrier();
        // Invocation d counts digit d over the tile.
        var count = b.Local(b.U(0));
        b.For(b.U(0), b.U(RadixTile), j => count.V = count.V + b.Convert(digits[j] == tid, K.U32));
        b.Store(b.At(hist, tid * tiles + tile, 4), count.V);
        return b.Build();
    }

    // The workgroup's exclusive prefix of each invocation's value v (u32) and the total, by a scan in workgroup memory
    // (Hillis–Steele over `threads` values).
    private static (V Before, V Total) GroupPrefix(Spv b, Arr sh, V tid, V v, int threads)
    {
        sh[tid] = v;
        b.Barrier();
        for (int s = 1; s < threads; s *= 2)
        {
            int step = s;
            var add = b.Select(tid >= step, sh[b.Select(tid >= step, tid - step, tid)], b.U(0));
            b.Barrier();
            sh[tid] = sh[tid] + add;
            b.Barrier();
        }

        var inclusive = sh[tid];
        var total = sh[b.U(threads - 1)];
        b.Barrier();
        return (inclusive - v, total);
    }

    /// <summary>hist[d][t] ← Σ_{t' &lt; t} hist[d][t'] in place and totals[d] = Σ_t hist[d][t]: a workgroup per digit d, each
    /// invocation a run of ⌈tiles / 256⌉ tiles. Arguments: hist, tiles, totals.</summary>
    public static SpvKernel RadixScanTiles(Vulkan vk)
    {
        var b = new Spv("radix_scan_tiles", RadixTile, vk.Controls);
        var hist = b.Arg(0);
        var tiles = b.Arg(1, K.U32);
        var totals = b.Arg(2);
        var tid = b.LocalIdX;
        var d = b.GroupX;
        var run = (tiles + (RadixTile - 1)) / (double)RadixTile;
        var first = b.Min(tid * run, tiles);
        var last = b.Min(first + run, tiles);
        var sum = b.Local(b.U(0));
        b.For(first, last, t => sum.V = sum.V + b.Load(K.U32, b.At(hist, d * tiles + t, 4)));
        var (before, total) = GroupPrefix(b, b.Shared(K.U32, RadixTile), tid, sum.V, RadixTile);
        var acc = b.Local(before);
        b.For(first, last, t =>
        {
            var at = b.At(hist, d * tiles + t, 4);
            var c = b.Load(K.U32, at);
            b.Store(at, acc.V);
            acc.V = acc.V + c;
        });
        b.If(tid == 0, () => b.Store(b.At(totals, d, 4), total));
        return b.Build();
    }

    /// <summary>bases[d] = Σ_{d' &lt; d} totals[d']: one workgroup, an invocation a digit. Arguments: totals, bases; with
    /// buckets (route's one pass: every key a digit, below buckets ≤ 256): counts, offsets, buckets too, counts[d] =
    /// totals[d] and offsets[d] = bases[d] for d &lt; buckets, offsets[buckets] the keys kept.</summary>
    public static SpvKernel RadixScanBases(Vulkan vk, bool buckets = false)
    {
        var b = new Spv($"radix_scan_bases{(buckets ? "_buckets" : "")}", RadixDigits, vk.Controls);
        var totals = b.Arg(0);
        var bases = b.Arg(1);
        var d = b.LocalIdX;
        var count = b.Load(K.U32, b.At(totals, d, 4));
        var (before, total) = GroupPrefix(b, b.Shared(K.U32, RadixDigits), d, count, RadixDigits);
        b.Store(b.At(bases, d, 4), before);
        if (buckets)
        {
            V counts = b.Arg(2), offsets = b.Arg(3), targets = b.Arg(4, K.U32);
            b.If(d < targets, () =>
            {
                b.Store(b.At(counts, d, 4), count);
                b.Store(b.At(offsets, d, 4), before);
            });
            b.If(d == 0, () => b.Store(b.At(offsets, targets, 4), total));
        }

        return b.Build();
    }

    /// <summary>offsets[0] = 0, offsets[r + 1] = offsets[r] + counts[r] for r &lt; targets, offsets past targets + 1 untouched:
    /// one workgroup, each invocation a run of ⌈targets / 256⌉ counts summed, the runs prefixed by the workgroup's scan, each
    /// run's offsets written. Arguments: counts, offsets, targets.</summary>
    public static SpvKernel OffsetsScan(Vulkan vk)
    {
        var b = new Spv("offsets_scan", RadixTile, vk.Controls);
        var counts = b.Arg(0);
        var offsets = b.Arg(1);
        var targets = b.Arg(2, K.U32);
        var tid = b.LocalIdX;
        var run = (targets + (RadixTile - 1)) / (double)RadixTile;
        var first = b.Min(tid * run, targets);
        var last = b.Min(first + run, targets);
        var sum = b.Local(b.U(0));
        b.For(first, last, r => sum.V = sum.V + b.Load(K.U32, b.At(counts, r, 4)));
        var (before, _) = GroupPrefix(b, b.Shared(K.U32, RadixTile), tid, sum.V, RadixTile);
        b.If(tid == 0, () => b.Store(b.At(offsets, b.U(0), 4), b.U(0)));
        var acc = b.Local(before);
        b.For(first, last, r =>
        {
            acc.V = acc.V + b.Load(K.U32, b.At(counts, r, 4));
            b.Store(b.At(offsets, r + 1, 4), acc.V);
        });
        return b.Build();
    }

    /// <summary>Each source of the tile to base[d] + hist[d][t] + its rank among the tile's earlier sources of digit d.
    /// Arguments: keys, values (0: the sources' own indices, the first pass), n, shift, hist, tiles, bases, keysOut, valuesOut,
    /// limit, valid (as RadixHistogram's), tail (the last pass: the kept count's address; valuesOut[i] = −1 from it to n).</summary>
    public static SpvKernel RadixScatter(Vulkan vk)
    {
        var b = new Spv("radix_scatter", RadixTile, vk.Controls);
        var keys = b.Arg(0);
        var values = b.Arg(1);
        var all = b.Arg(2, K.U32);
        var n = Kept(b, all, b.Arg(10));
        var shift = b.Arg(3, K.U32);
        var hist = b.Arg(4);
        var tiles = b.Arg(5, K.U32);
        var bases = b.Arg(6);
        var keysOut = b.Arg(7);
        var valuesOut = b.Arg(8);
        var limit = b.Arg(9, K.U32);
        var tail = b.Arg(11);
        var tid = b.LocalIdX;
        var tile = b.GroupX;
        var i = tile * RadixTile + tid;
        var before = i < n;
        var safe = b.Select(before, i, b.U(0));
        var key = b.Load(K.U32, b.At(keys, safe, 4));
        var inside = before & (key < limit);
        var digit = b.Select(inside, key.Shr(shift) & 255UL, b.U(0xFFFFFFFF));
        var digits = b.Shared(K.U32, RadixTile);
        digits[tid] = digit;
        b.Barrier();
        b.If(inside, () =>
        {
            var rank = b.Local(b.U(0));
            b.For(b.U(0), tid, j => rank.V = rank.V + b.Convert(digits[j] == digit, K.U32));
            var dest = b.Load(K.U32, b.At(bases, digit, 4)) + b.Load(K.U32, b.At(hist, digit * tiles + tile, 4)) + rank.V;
            var value = b.Local(i);
            b.If(values != 0UL, () => value.V = b.Load(K.U32, b.At(values, i, 4)));
            b.Store(b.At(keysOut, dest, 4), key);
            b.Store(b.At(valuesOut, dest, 4), value.V);
        });
        // Past the sources placed: −1 at position i itself (no placed source's rank reaches it).
        b.If(tail != 0UL, () =>
        {
            var from = b.Load(K.U32, b.At(tail, b.U(0), 4));
            b.If((i >= from) & (i < all), () => b.Store(b.At(valuesOut, i, 4), b.U(0xFFFFFFFF)));
        });
        return b.Build();
    }

    // A pass's n: n, or the kept count at `valid` where it is given (a later pass: the sources the first kept).
    private static V Kept(Spv b, V n, V valid)
    {
        var kept = b.Local(n);
        b.If(valid != 0UL, () => kept.V = b.Min(n, b.Load(K.U32, b.At(valid, b.U(0), 4))));
        return kept.V;
    }
}
