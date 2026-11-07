// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The stable order of n sources by their bucket (route's order, scatter-add's ordering) on Vulkan: a least-significant-
// digit radix sort of (bucket, source) pairs, 8 bits a pass, each pass over tiles of 256 sources:
//   RadixHistogram  a workgroup per tile: each digit's count in the tile (counted from the tile's digits in workgroup
//                   memory, no atomics), stored digit-major (hist[d · tiles + t]);
//   RadixScan       one workgroup: each digit's counts prefixed over the tiles (an invocation a digit), the digits' totals
//                   prefixed (the digits' bases);
//   RadixScatter    a workgroup per tile: each source to base[d] + hist[d][t] + its rank among the tile's earlier sources
//                   of its digit — stable, so the passes compose to the order by bucket, sources in order within one.
// The counts by integer atomics and the offsets by a scan (ScatterCount, ScatterScan).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int RadixTile = 256, RadixDigits = 256;

    /// <summary>Arguments: keys (s32; or the bucket index array on the first pass), n, shift, hist, tiles.</summary>
    public static SpvKernel RadixHistogram(Vulkan vk)
    {
        var b = new Spv("radix_histogram", RadixTile, vk.Controls);
        var keys = b.Arg(0);
        var n = b.Arg(1, K.U32);
        var shift = b.Arg(2, K.U32);
        var hist = b.Arg(3);
        var tiles = b.Arg(4, K.U32);
        var tid = b.LocalIdX;
        var tile = b.GroupX;
        var i = tile * RadixTile + tid;
        var digits = b.Shared(K.U32, RadixTile);
        var key = b.Local(b.U(0xFFFFFFFF));                                          // past n: no digit
        b.If(i < n, () => key.V = (b.Bitcast(b.Load(K.S32, b.At(keys, i, 4)), K.U32).Shr(shift)) & 255UL);
        digits[tid] = key.V;
        b.Barrier();
        // Invocation d counts digit d over the tile.
        var count = b.Local(b.U(0));
        b.For(b.U(0), b.U(RadixTile), j => count.V = count.V + b.Convert(digits[j] == tid, K.U32));
        b.Store(b.At(hist, tid * tiles + tile, 4), count.V);
        return b.Build();
    }

    /// <summary>hist[d][t] ← Σ_{t' &lt; t} hist[d][t'] in place, bases[d] = Σ_{d' &lt; d} the digits' totals. Arguments: hist, tiles, bases.</summary>
    public static SpvKernel RadixScan(Vulkan vk)
    {
        var b = new Spv("radix_scan", RadixDigits, vk.Controls);
        var hist = b.Arg(0);
        var tiles = b.Arg(1, K.U32);
        var bases = b.Arg(2);
        var d = b.LocalIdX;
        var acc = b.Local(b.U(0));
        b.For(b.U(0), tiles, t =>
        {
            var at = b.At(hist, d * tiles + t, 4);
            var c = b.Load(K.U32, at);
            b.Store(at, acc.V);
            acc.V = acc.V + c;
        });
        var totals = b.Shared(K.U32, RadixDigits);
        totals[d] = acc.V;
        b.Barrier();
        b.If(d == 0, () =>
        {
            var run = b.Local(b.U(0));
            b.For(b.U(0), b.U(RadixDigits), q =>
            {
                var c = totals[q];
                b.Store(b.At(bases, q, 4), run.V);
                run.V = run.V + c;
            });
        });
        return b.Build();
    }

    /// <summary>Each source of the tile to base[d] + hist[d][t] + its rank among the tile's earlier sources of digit d.
    /// Arguments: keys, values (0: the sources' own indices, the first pass), n, shift, hist, tiles, bases, keysOut, valuesOut.</summary>
    public static SpvKernel RadixScatter(Vulkan vk)
    {
        var b = new Spv("radix_scatter", RadixTile, vk.Controls);
        var keys = b.Arg(0);
        var values = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var shift = b.Arg(3, K.U32);
        var hist = b.Arg(4);
        var tiles = b.Arg(5, K.U32);
        var bases = b.Arg(6);
        var keysOut = b.Arg(7);
        var valuesOut = b.Arg(8);
        var tid = b.LocalIdX;
        var tile = b.GroupX;
        var i = tile * RadixTile + tid;
        var inside = i < n;
        var safe = b.Select(inside, i, b.U(0));
        var key = b.Load(K.U32, b.At(keys, safe, 4));
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
        return b.Build();
    }
}
