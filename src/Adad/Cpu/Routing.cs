// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and losses: operations whose results are indices, or whose inputs are.
//
//   topk_rows       the k first of each row in Adad's order (values as the type, indices s32): larger first, a NaN above
//                   every number, ties by the lower index (Routing.Precedes; the lane's Widths.Compare)
//   pick_rows       out[r] = x[r][index[r]] (a value per row: a loss from log-probabilities and targets)
//   sample_rows     the index of the first column whose running sum of the row reaches u[r] · (the row's sum): sampling
//                   from unnormalized weights with a uniform draw per row (ties and an empty row give the last column)
//   route           for n indices into `buckets`: counts[b], offsets[b + 1] (exclusive prefix), order (the sources
//                   grouped by bucket in source order): the expert placement of a router
//   cross_entropy   the loss per row from logits and a target per row: log Σ exp(x[r]) − x[r][t[r]], one pass over the
//                   row (also the composition log_softmax → pick_rows, negated); its backward is the row operation
//                   cross_entropy_bwd (softmax(x) − one_hot(target)) in Rows.cs

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>top-k of each row: values stored as the type (rows × k), indices s32; returns the lane's name.</summary>
    public static string TopK<TN>(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) where TN : struct, IAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var v = new TopKVisitor(x, rows, cols, k, values, indices);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    private struct TopKVisitor(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            // Each row read where it is stored; a block format's values as the lane's, encoded once after (a row's k values
            // would re-encode their block row by row).
            bool lanes = TN.Granularity > 8;
            var output = lanes ? new byte[(long)rows * k * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>()] : values;
            Team.For(rows, Kernels.RowChunk<TN>(rows, k, rows / (Threads * 4)), (long)rows * cols * 4, new TopKBody<TN, TLane, TL>(x, cols, k, output, indices, lanes));
            if (lanes) StoreLanes<TN, TLane>(output, values, 0, (long)rows * k);
        }
    }

    private struct TopKBody<TN, TLane, TL>(byte[] x, int cols, int k, byte[] values, int[] indices, bool lanes) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(cols);
            var best = pool.Rent(k);
            var bestAt = ArrayPool<int>.Shared.Rent(k);
            try
            {
                for (int r = first; r < last; r++)
                {
                    TN.Load(x, r * cols, row.AsSpan(0, cols));
                    TopKRow<TLane, TL>(row.AsSpan(0, cols), k, best, bestAt);

                    if (lanes) best.AsSpan(0, k).CopyTo(MemoryMarshal.Cast<byte, TLane>(values.AsSpan()).Slice(r * k, k));
                    else TN.Store(best.AsSpan(0, k), values, r * k);
                    Array.Copy(bestAt, 0, indices, r * k, k);
                }
            }
            finally
            {
                pool.Return(row); pool.Return(best); ArrayPool<int>.Shared.Return(bestAt);
            }
        }
    }

    // One pass keeping a row's k first in Adad's order (Widths.Compare: larger first, a NaN above every number; an equal
    // value after the earlier ones): a value enters only when it ranks strictly above the k-th, then moves past every kept
    // value it ranks strictly above. The order is total, so this is the k passes of first-ranked exactly. Once k are kept,
    // a vector of the row none of whose values ranks strictly above the k-th is passed over whole (its values would each
    // have been passed over), so the values that enter, and the order they enter in, are the same.
    private static void TopKRow<TLane, TL>(ReadOnlySpan<TLane> xs, int k, TLane[] best, int[] bestAt) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        if (k == 0) return;                                                   // nothing kept (the k-th would be best[−1])
        int cols = xs.Length, w = Vector<TLane>.Count, taken = 0, c = 0;
        for (; c < cols && taken < k; c++) Enter(xs[c], c);
        for (; c + w <= cols; c += w)
        {
            if (!AnyAbove(new Vector<TLane>(xs.Slice(c, w)), best[k - 1])) continue;
            for (int l = 0; l < w; l++) Enter(xs[c + l], c + l);
        }

        for (; c < cols; c++) Enter(xs[c], c);

        void Enter(TLane v, int at)
        {
            if (taken == k && Widths.Compare(v, best[k - 1]) <= 0) return;
            int j = taken < k ? taken++ : k - 1;
            while (j > 0 && Widths.Compare(v, best[j - 1]) > 0) { best[j] = best[j - 1]; bestAt[j] = bestAt[j - 1]; j--; }
            best[j] = v;
            bestAt[j] = at;
        }
    }

    // Whether a value of v ranks strictly above t in Adad's order: larger, or a NaN where t is a number (−0 and +0 equal).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool AnyAbove<TLane>(Vector<TLane> v, TLane t) where TLane : unmanaged
    {
        var tv = new Vector<TLane>(t);
        return Vector.GreaterThanAny(v, tv) || (Vector.EqualsAll(tv, tv) && !Vector.EqualsAll(v, v));
    }

    /// <summary>The top-k of each row as log-probabilities (IRouting.LogSoftmaxTopK): the row in the float lane, TopK's
    /// selection, then each chosen value less the row's first-ranked and less the log of the row's Σ exp(x − it); stored as
    /// the type once. Returns the lane's name.</summary>
    public static string LogSoftmaxTopK<TN>(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) where TN : struct, IFloatAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var v = new LogTopKVisitor(x, rows, cols, k, values, indices);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct LogTopKVisitor(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            if (rows == 0 || k == 0) return;
            // A block format's values as the lane's, encoded once after (a row's k values would re-encode their block row by row).
            bool lanes = TN.Granularity > 8;
            var output = lanes ? ArrayPool<byte>.Shared.Rent(rows * k * Unsafe.SizeOf<TLane>()) : values;
            try
            {
                Team.For(rows, Kernels.RowChunk<TN>(rows, k, rows / (Threads * 4)), (long)rows * cols * 6, new LogTopKBody<TN, TLane, TL>(x, cols, k, output, indices, lanes));
                if (lanes) StoreLanes<TN, TLane>(output, values, 0, (long)rows * k);
            }
            finally
            {
                if (lanes) ArrayPool<byte>.Shared.Return(output);
            }
        }
    }

    private readonly struct LogTopKBody<TN, TLane, TL>(byte[] x, int cols, int k, byte[] values, int[] indices, bool lanes) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(Math.Max(1, cols));
            var best = pool.Rent(k);
            var bestAt = ArrayPool<int>.Shared.Rent(k);
            try
            {
                for (int r = first; r < last; r++)
                {
                    var xs = row.AsSpan(0, cols);
                    TN.Load(x, r * cols, xs);
                    // TopK's pass: the k first in Adad's order, an equal value after the earlier ones.
                    TopKRow<TLane, TL>(xs, k, best, bestAt);

                    // The row's first-ranked value (the first chosen) and the log of Σ exp(x − it) (the row is the sum's
                    // scratch after the selection); k is at least 1 here and at most the row's columns.
                    TLane m = best[0];
                    TLane logSum = TL.Log(new System.Numerics.Vector<TLane>(RowMath<TLane, TL>.ExpSum(xs, m, xs)))[0];
                    for (int i = 0; i < k; i++) best[i] = TL.Sub(TL.Sub(best[i], m), logSum);

                    if (lanes) best.AsSpan(0, k).CopyTo(MemoryMarshal.Cast<byte, TLane>(values.AsSpan()).Slice(r * k, k));
                    else TN.Store(best.AsSpan(0, k), values, r * k);
                    Array.Copy(bestAt, 0, indices, r * k, k);
                }
            }
            finally
            {
                pool.Return(row); pool.Return(best); ArrayPool<int>.Shared.Return(bestAt);
            }
        }
    }

    /// <summary>out[r] = x[r][index[r]], stored as the type (a column outside the row: 0).</summary>
    public static string Pick<TN>(byte[] x, int rows, int cols, int[] index, byte[] output) where TN : struct, IAdad<TN>
    {
        var v = new PickVisitor(x, rows, cols, index, output);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    private struct PickVisitor(byte[] x, int rows, int cols, int[] index, byte[] output) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            if (TN.Bits % 8 == 0 && TN.Granularity == 1)
            {
                // Values of whole bytes: the picked value is its stored bytes, copied as they are (a column outside the row
                // reads nothing: the type's 0, stored).
                int size = TN.Bits / 8;
                var zero = new TLane[1];
                zero[0] = TL.Zero;
                for (int r = 0; r < rows; r++)
                {
                    if ((uint)index[r] >= (uint)cols) { TN.Store(zero, output, r); continue; }
                    long from = ((long)r * cols + index[r]) * size;
                    switch (size)
                    {
                        case 1: output[r] = x[from]; break;
                        case 2: Unsafe.WriteUnaligned(ref output[r * 2], Unsafe.ReadUnaligned<ushort>(ref x[from])); break;
                        case 4: Unsafe.WriteUnaligned(ref output[r * 4], Unsafe.ReadUnaligned<uint>(ref x[from])); break;
                        case 8: Unsafe.WriteUnaligned(ref output[r * 8], Unsafe.ReadUnaligned<ulong>(ref x[from])); break;
                        default: x.AsSpan((int)from, size).CopyTo(output.AsSpan(r * size, size)); break;
                    }
                }
                return;
            }

            // Values that share a storage unit (packed bits, a block): each read through the type, its whole unit decoded, so
            // the rows go to the team by the units' bytes (a byte of packed bits stays on the caller; blocks spread).
            var picked = new TLane[rows];
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), rows * TN.Bytes(TN.Granularity),
                new PickBody<TN, TLane, TL>(x, cols, index, picked));
            TN.Store(picked, output, 0);
        }
    }

    private readonly struct PickBody<TN, TLane, TL>(byte[] x, int cols, int[] index, TLane[] picked) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            for (int r = first; r < last; r++)
            {
                // A column outside the row reads nothing: 0.
                if ((uint)index[r] >= (uint)cols) { picked[r] = TL.Zero; continue; }
                TN.Load(x, r * cols + index[r], picked.AsSpan(r, 1));
            }
        }
    }

    /// <summary>loss[r] = w[r]·(log Σ exp(x[r]) − x[r][t[r]]) as the type, in the type's float lane: a row's max, its Σ exp(x − max),
    /// then ((max − x[t]) + log Σ)·w[r] (weights as the type, absent 1).</summary>
    public static string CrossEntropy<TN>(byte[] x, int rows, int cols, int[] targets, byte[]? weights, byte[] output) where TN : struct, IFloatAdad<TN>
    {
        var v = new CrossEntropyVisitor(x, rows, cols, targets, weights, output);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct CrossEntropyVisitor(byte[] x, int rows, int cols, int[] targets, byte[]? weights, byte[] output) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            var losses = new TLane[rows];
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols * 4, new CrossEntropyBody<TN, TLane, TL>(x, cols, targets, losses));
            if (weights is not null)
            {
                var w = new TLane[rows];
                TN.Load(weights, 0, w);
                for (int r = 0; r < rows; r++) losses[r] = TL.Mul(losses[r], w[r]);
            }

            TN.Store(losses, output, 0);
        }
    }

    private struct CrossEntropyBody<TN, TLane, TL>(byte[] x, int cols, int[] targets, TLane[] losses) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = TN.Direct ? null : pool.Rent(cols);
            var e = pool.Rent(cols);
            try
            {
                for (int r = first; r < last; r++)
                {
                    ReadOnlySpan<TLane> xs;
                    if (TN.Direct) xs = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols, cols);
                    else { TN.Load(x, r * cols, row.AsSpan(0, cols)); xs = row.AsSpan(0, cols); }
                    TLane m = RowMath<TLane, TL>.Max(xs);
                    TLane sum = RowMath<TLane, TL>.ExpSum(xs, m, e.AsSpan(0, cols));
                    losses[r] = TL.Add(TL.Sub(m, xs[targets[r]]), RowMath<TLane, TL>.Log(sum));
                }
            }
            finally
            {
                if (row is not null) pool.Return(row);
                pool.Return(e);
            }
        }
    }

    /// <summary>The sampled column of each row from weights x and a uniform draw u per row (as the type); in the type's float lane.</summary>
    public static string Sample<TN>(byte[] x, int rows, int cols, byte[] u, int[] picks) where TN : struct, IFloatAdad<TN>
    {
        var v = new SampleVisitor(x, rows, cols, u, picks);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct SampleVisitor(byte[] x, int rows, int cols, byte[] u, int[] picks) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols * 2, new SampleBody<TN, TLane, TL>(x, cols, u, picks));
        }
    }

    private struct SampleBody<TN, TLane, TL>(byte[] x, int cols, byte[] u, int[] picks) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(cols);
            var one = pool.Rent(1);
            try
            {
                for (int r = first; r < last; r++)
                {
                    TN.Load(x, r * cols, row.AsSpan(0, cols));
                    TN.Load(u, r, one.AsSpan(0, 1));
                    // The positive weights' sum (a weight below 0 adds nothing, a NaN carries into it), whole vectors first.
                    int width = System.Numerics.Vector<TLane>.Count, i = 0;
                    var acc = System.Numerics.Vector<TLane>.Zero;
                    for (; i + width <= cols; i += width)
                    {
                        var e = new System.Numerics.Vector<TLane>(row, i);
                        acc = TL.Add(acc, TL.Select(TL.Greater(System.Numerics.Vector<TLane>.Zero, e), System.Numerics.Vector<TLane>.Zero, e));
                    }

                    TLane total = TL.Zero;
                    for (int l = 0; l < width; l++) total = TL.Add(total, acc[l]);
                    for (; i < cols; i++) if (!(LaneValue(row[i]) < 0)) total = TL.Add(total, row[i]);
                    // The first column of positive weight at which the running sum passes u times the total (the last such
                    // column when rounding leaves none); −1 for a total that is not a positive finite number, or a NaN u.
                    TLane target = TL.Mul(one[0], total), run = TL.Zero;
                    double t = LaneValue(total);
                    int pick = -1;
                    if (t > 0 && !double.IsPositiveInfinity(t) && !double.IsNaN(LaneValue(target)))
                        for (int c = 0; c < cols; c++)
                        {
                            if (!(LaneValue(row[c]) > 0)) continue;
                            pick = c;
                            run = TL.Add(run, row[c]);
                            if (Widths.Compare(run, target) > 0) break;
                        }

                    picks[r] = pick;
                }
            }
            finally
            {
                pool.Return(row); pool.Return(one);
            }
        }
    }

    /// <summary>counts, offsets (buckets + 1) and the sources' order by bucket (source order within a bucket); an index
    /// outside [0, buckets) in no bucket, the order −1 past the sources placed (IRouting.Route).</summary>
    public static void Route(int[] index, int buckets, int[] counts, int[] offsets, int[] order)
    {
        // On the team: each task counts a slice of the sources into its own run of counts, then a pass over the buckets
        // turns them into each task's cursor in each bucket (a bucket's tasks in slice order; at most half as many counts as
        // sources, so that pass stays short), then each task places its slice. A bucket holds its sources in source order,
        // as one pass placing them in turn would: the same counts, offsets and order. On one thread the placement's
        // scattered writes ran at that one core's caches: 65 536 buckets took 2.2 ms with the thread throttled (efficiency
        // cores) and 4.0 with it at full speed (measured).
        int n = index.Length;
        int tasks = Math.Min(Math.Min(Threads, n / RouteSlice), Math.Min(64, Math.Max(1, n / Math.Max(1, 2 * buckets))));
        if (tasks <= 1)
        {
            Array.Clear(counts);
            foreach (int i in index) if ((uint)i < (uint)buckets) counts[i]++;
            offsets[0] = 0;
            for (int b = 0; b < buckets; b++) offsets[b + 1] = offsets[b] + counts[b];
            var cursor = new int[buckets];
            Array.Copy(offsets, cursor, buckets);
            for (int i = 0; i < n; i++) if ((uint)index[i] < (uint)buckets) order[cursor[index[i]]++] = i;
            order.AsSpan(offsets[buckets], n - offsets[buckets]).Fill(-1);
            return;
        }

        int slice = (n + tasks - 1) / tasks;
        var cells = ArrayPool<int>.Shared.Rent(buckets * tasks);
        try
        {
            Team.For(tasks, 1, (long)n * 4, new RouteCount(index, buckets, tasks, slice, cells));
            // cells[t·buckets + b]: task t's count in bucket b, then its first place there.
            offsets[0] = 0;
            int at = 0;
            for (int b = 0; b < buckets; b++)
            {
                int start = at;
                for (int t = 0; t < tasks; t++)
                {
                    int c = cells[t * buckets + b];
                    cells[t * buckets + b] = at;
                    at += c;
                }

                counts[b] = at - start;
                offsets[b + 1] = at;
            }

            Team.For(tasks, 1, (long)n * 4, new RoutePlace(index, buckets, tasks, slice, cells, order));
            order.AsSpan(offsets[buckets], n - offsets[buckets]).Fill(-1);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(cells);
        }
    }

    // Route's team: a task at least this many sources.
    private const int RouteSlice = 1 << 15;

    private readonly struct RouteCount(int[] index, int buckets, int tasks, int slice, int[] cells) : Team.IBody
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            for (int t = first; t < last; t++)
            {
                cells.AsSpan(t * buckets, buckets).Clear();
                int end = Math.Min(index.Length, (t + 1) * slice);
                for (int i = t * slice; i < end; i++)
                {
                    int k = index[i];
                    if ((uint)k < (uint)buckets) cells[t * buckets + k]++;
                }
            }
        }
    }

    private readonly struct RoutePlace(int[] index, int buckets, int tasks, int slice, int[] cells, int[] order) : Team.IBody
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            for (int t = first; t < last; t++)
            {
                int end = Math.Min(index.Length, (t + 1) * slice);
                for (int i = t * slice; i < end; i++)
                {
                    int k = index[i];
                    if ((uint)k < (uint)buckets) order[cells[t * buckets + k]++] = i;
                }
            }
        }
    }
}
