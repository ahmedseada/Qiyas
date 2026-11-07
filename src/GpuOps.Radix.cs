// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Scatter-add's ordering on the device (item 4f): counts by integer atomics (exact), offsets by the multi-block scan,
// the sources' order by target by the stable LSD radix sort of (target, source) pairs (Ptx.Radix.cs) — linear in the
// sources, for any number of targets; before 4f the order took a quadratic rank loop and the offsets one thread.
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>The gate's override: false takes the counts by atomics and their scan for few targets too.</summary>
    internal bool? TableCounts;

    /// <summary>out[i] = Σ in[0..i) over n values (in and out may be the same buffer): blocks of 1024, the blocks' totals
    /// scanned the same way, then added back.</summary>
    internal void ExclusiveScan(ulong input, ulong output, int n)
    {
        if (n <= 0) return;
        var tile = gpu.Cached("scan_tile", (out string e) => Ptx.ScanTile(gpu, out e));
        int blocks = (n + Ptx.ScanBlock - 1) / Ptx.ScanBlock;
        if (blocks == 1)
        {
            Gpu.Launch(tile, 1, 1, 256, 1, 0, input, output, 0UL, Gpu.U(n));
            return;
        }

        ulong totals = gpu.Rent(blocks * 4L);
        try
        {
            Gpu.Launch(tile, (uint)blocks, 1, 256, 1, 0, input, output, totals, Gpu.U(n));
            ExclusiveScan(totals, totals, blocks);
            var add = gpu.Cached("scan_add", (out string e) => Ptx.ScanAdd(gpu, out e));
            Gpu.Launch(add, Grid(n, 256), 1, 256, 1, 0, output, totals, Gpu.U(n));
        }
        finally { gpu.Return(totals); }
    }

    /// <summary>order[k] = the sources sorted by index[source] (0 ≤ index &lt; targets), stable: the source order within a
    /// target. An LSD radix sort, 8 bits a pass over the bits the targets need.</summary>
    internal void RadixOrder(ulong index, int n, int targets, ulong order) => RadixOrder(index, n, targets, order, 0, 0);

    // countsOut / offsetsOut (item 6j, one pass only: targets ≤ 256): the route's counts and offsets from the pass's scanned
    // table (RadixTableOffsets), the place kernel reading the table only.
    private void RadixOrder(ulong index, int n, int targets, ulong order, ulong countsOut, ulong offsetsOut)
    {
        if (n <= 0) return;
        int bits = targets <= 1 ? 1 : 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)(targets - 1));
        int passes = (bits + 7) / 8, tiles = (n + Ptx.RadixTile - 1) / Ptx.RadixTile;
        var count = gpu.Cached("radix_count", (out string e) => Ptx.RadixCount(gpu, out e));
        var place = gpu.Cached("radix_place", (out string e) => Ptx.RadixPlace(gpu, out e));
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = gpu.Rent(bytes); rented.Add(p); return p; }
        try
        {
            ulong table = Rent(256L * tiles * 4);
            ulong[] keys = [Rent(n * 4L), Rent(n * 4L)], vals = [Rent(n * 4L), Rent(n * 4L)];
            ulong keysIn = index, valsIn = 0;
            for (int pass = 0; pass < passes; pass++)
            {
                uint shift = (uint)(8 * pass);
                ulong keysOut = keys[pass % 2], valsOut = pass == passes - 1 ? order : vals[pass % 2];
                Gpu.Launch(count, (uint)tiles, 1, 256, 1, 0, keysIn, table, Gpu.U(n), Gpu.U(tiles), shift);
                ExclusiveScan(table, table, 256 * tiles);
                if (countsOut != 0 && passes == 1)
                {
                    var fromTable = gpu.Cached("radix_table_offsets", (out string e) => Ptx.RadixTableOffsets(gpu, out e));
                    Gpu.Launch(fromTable, Grid(targets + 1, 256), 1, 256, 1, 0, table, countsOut, offsetsOut, Gpu.U(targets), Gpu.U(tiles), Gpu.U(n));
                }

                Gpu.Launch(place, (uint)tiles, 1, 256, 1, 0, keysIn, valsIn, keysOut, valsOut, table, Gpu.U(n), Gpu.U(tiles), shift);
                (keysIn, valsIn) = (keysOut, valsOut);
            }
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    /// <summary>counts[t] (targets), offsets[t] (targets + 1: offsets[targets] = n) and the sources' order by target
    /// (stable) from n indices: scatter-add's ordering and the route.</summary>
    internal void OrderByTarget(ulong index, int n, int targets, ulong counts, ulong offsets, ulong order)
    {
        if (n > 0 && targets <= 256 && TableCounts != false)
        {
            // One radix pass: the counts and offsets from its scanned table (item 6j), no counting atomics.
            RadixOrder(index, n, targets, order, counts, offsets);
            return;
        }

        Gpu.Clear(counts, targets * 4L);
        if (n > 0)
        {
            var count = gpu.Cached("scatter_count", (out string e) => Ptx.ScatterCount(gpu, out e));
            Gpu.Launch(count, Grid(n, 256), 1, 256, 1, 0, index, counts, Gpu.U(n));
        }

        ExclusiveScan(counts, offsets, targets);
        Gpu.Upload<int>([n], offsets + (ulong)(targets * 4L));
        RadixOrder(index, n, targets, order);
    }
}
