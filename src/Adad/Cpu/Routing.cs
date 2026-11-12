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
            Team.For(rows, Kernels.RowChunk<TN>(rows, k, rows / (Threads * 4)), (long)rows * cols * 4, new TopKBody<TN, TLane, TL>(x, rows, cols, k, output, indices, lanes));
            if (lanes) StoreLanes<TN, TLane>(output, values, 0, (long)rows * k);
        }
    }

    private struct TopKBody<TN, TLane, TL>(byte[] x, int rows, int cols, int k, byte[] values, int[] indices, bool lanes) : Team.IBody
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
                    // One pass keeping the k first in Adad's order (Widths.Compare: larger first, a NaN above every number;
                    // an equal value after the earlier ones): a value enters only when it ranks strictly above the k-th, then
                    // moves past every kept value it ranks strictly above. The order is total, so this is the k passes of
                    // first-ranked exactly.
                    int taken = 0;
                    for (int c = 0; c < cols; c++)
                    {
                        TLane v = row[c];
                        if (taken == k && Widths.Compare(v, best[k - 1]) <= 0) continue;
                        int j = taken < k ? taken++ : k - 1;
                        while (j > 0 && Widths.Compare(v, best[j - 1]) > 0) { best[j] = best[j - 1]; bestAt[j] = bestAt[j - 1]; j--; }
                        best[j] = v; bestAt[j] = c;
                    }

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
                    int taken = 0;
                    for (int c = 0; c < cols; c++)
                    {
                        TLane v = xs[c];
                        if (taken == k && Widths.Compare(v, best[k - 1]) <= 0) continue;
                        int j = taken < k ? taken++ : k - 1;
                        while (j > 0 && Widths.Compare(v, best[j - 1]) > 0) { best[j] = best[j - 1]; bestAt[j] = bestAt[j - 1]; j--; }
                        best[j] = v; bestAt[j] = c;
                    }

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
            var picked = new TLane[rows];
            var one = new TLane[1];
            for (int r = 0; r < rows; r++)
            {
                // A column outside the row reads nothing: 0.
                if ((uint)index[r] >= (uint)cols) { picked[r] = TL.Zero; continue; }
                TN.Load(x, r * cols + index[r], one.AsSpan(0, 1));
                picked[r] = one[0];
            }
            TN.Store(picked, output, 0);
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
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols * 2, new SampleBody<TN, TLane, TL>(x, rows, cols, u, picks));
        }
    }

    private struct SampleBody<TN, TLane, TL>(byte[] x, int rows, int cols, byte[] u, int[] picks) : Team.IBody
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
                    TLane total = RowMath<TLane, TL>.Sum(row.AsSpan(0, cols));
                    TLane threshold = TL.Mul(one[0], total), run = TL.Zero;
                    int pick = cols - 1;
                    for (int c = 0; c < cols; c++)
                    {
                        run = TL.Add(run, row[c]);
                        if (Widths.Compare(run, threshold) >= 0) { pick = c; break; }
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

    /// <summary>counts, offsets (buckets + 1) and the sources' order by bucket (source order within a bucket).</summary>
    public static void Route(int[] index, int buckets, int[] counts, int[] offsets, int[] order)
    {
        Array.Clear(counts);
        foreach (int i in index) counts[i]++;
        offsets[0] = 0;
        for (int b = 0; b < buckets; b++) offsets[b + 1] = offsets[b] + counts[b];
        var cursor = new int[buckets];
        Array.Copy(offsets, cursor, buckets);
        for (int i = 0; i < index.Length; i++) order[cursor[index[i]]++] = i;
    }
}
