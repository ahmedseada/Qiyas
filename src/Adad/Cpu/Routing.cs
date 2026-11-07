// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and losses: operations whose results are indices, or whose inputs are.
//
//   topk_rows       the k largest of each row (values as the type, indices s32), largest first; ties by the lower index
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
                    // One pass keeping the k largest in order (larger first, an equal value after the earlier ones): a value
                    // enters only when strictly larger than the k-th, then moves past every kept value it is strictly larger
                    // than. The comparison is a total order, so this is the k passes of first-largest exactly.
                    int taken = 0;
                    for (int c = 0; c < cols; c++)
                    {
                        TLane v = row[c];
                        if (taken == k && Widths.Compare(v, best[k - 1]) <= 0) continue;
                        int j = taken < k ? taken++ : k - 1;
                        while (j > 0 && Widths.Compare(v, best[j - 1]) > 0) { best[j] = best[j - 1]; bestAt[j] = bestAt[j - 1]; j--; }
                        best[j] = v; bestAt[j] = c;
                    }

                    for (int i = taken; i < k; i++) { best[i] = TL.Zero; bestAt[i] = -1; }          // k past the row: as the passes left them

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

    /// <summary>out[r] = x[r][index[r]], stored as the type.</summary>
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
            for (int r = 0; r < rows; r++) { TN.Load(x, r * cols + index[r], one.AsSpan(0, 1)); picked[r] = one[0]; }
            TN.Store(picked, output, 0);
        }
    }

    /// <summary>output[r] = 1 where row r of the predictions is right, else 0: one column, (p ≥ threshold) = (t ≥ ½) in the lane
    /// (each bound the lane's smallest value at least it); more, the first largest column of p (by >, from column 0) is t's.</summary>
    public static string ClassMatch<TN>(byte[] p, byte[] t, byte[] output, int rows, int cols, double threshold) where TN : struct, IAdad<TN>
    {
        var v = new ClassMatchVisitor(p, t, output, rows, cols, threshold);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    private struct ClassMatchVisitor(byte[] p, byte[] t, byte[] output, int rows, int cols, double threshold) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var lane = typeof(TLane) == typeof(float) ? Abstraction.Lane.F32 : typeof(TLane) == typeof(double) ? Abstraction.Lane.F64
                : typeof(TLane) == typeof(ulong) ? Abstraction.Lane.U64 : typeof(TLane) == typeof(long) ? Abstraction.Lane.S64
                : typeof(TLane) == typeof(uint) || typeof(TLane) == typeof(ushort) || typeof(TLane) == typeof(byte) ? Abstraction.Lane.U32 : Abstraction.Lane.S32;
            TLane th = TL.From(lane.AtLeast(threshold)), half = TL.From(lane.AtLeast(0.5));
            var match = new TLane[rows];
            Team.For(rows, 64, (long)rows * cols * 2, new ClassMatchBody<TN, TLane, TL>(p, t, match, cols, th, half));
            TN.Store(match, output, 0);
        }
    }

    private readonly struct ClassMatchBody<TN, TLane, TL>(byte[] p, byte[] t, TLane[] match, int cols, TLane th, TLane half) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pr = new TLane[cols];
            var tr = new TLane[cols];
            TLane one = TL.From(1), zero = TL.From(0);
            for (int r = first; r < last; r++)
            {
                TN.Load(p, r * cols, pr);
                TN.Load(t, r * cols, tr);
                bool right = cols == 1 ? AtLeast(pr[0], th) == AtLeast(tr[0], half) : First(pr) == First(tr);
                match[r] = right ? one : zero;
            }
        }

        // The first column whose value exceeds every one before it (> as IEEE compares: a NaN never exceeds, nor is exceeded).
        private static int First(TLane[] row)
        {
            int best = 0;
            for (int i = 1; i < row.Length; i++) if (Greater(row[i], row[best])) best = i;
            return best;
        }
    }

    private static bool AtLeast<T>(T a, T b) where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return Unsafe.As<T, float>(ref a) >= Unsafe.As<T, float>(ref b);
        if (typeof(T) == typeof(double)) return Unsafe.As<T, double>(ref a) >= Unsafe.As<T, double>(ref b);
        return Widths.Compare(a, b) >= 0;
    }

    private static bool Greater<T>(T a, T b) where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return Unsafe.As<T, float>(ref a) > Unsafe.As<T, float>(ref b);
        if (typeof(T) == typeof(double)) return Unsafe.As<T, double>(ref a) > Unsafe.As<T, double>(ref b);
        return Widths.Compare(a, b) > 0;
    }

    /// <summary>loss[r] = log Σ exp(x[r]) − x[r][t[r]] as the type, in the type's float lane: a row's max, its Σ exp(x − max),
    /// then (max − x[t]) + log Σ.</summary>
    public static string CrossEntropy<TN>(byte[] x, int rows, int cols, int[] targets, byte[] output) where TN : struct, IFloatAdad<TN>
    {
        var v = new CrossEntropyVisitor(x, rows, cols, targets, output);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct CrossEntropyVisitor(byte[] x, int rows, int cols, int[] targets, byte[] output) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            var losses = new TLane[rows];
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols * 4, new CrossEntropyBody<TN, TLane, TL>(x, cols, targets, losses));
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
