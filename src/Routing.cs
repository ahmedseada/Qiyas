// Routing, sampling and losses: operations whose results are indices, or whose inputs are.
//
//   topk_rows       the k largest of each row (values as the type, indices s32), largest first; ties by the lower index
//   pick_rows       out[r] = x[r][index[r]] (a value per row: a loss from log-probabilities and targets)
//   sample_rows     the index of the first column whose running sum of the row reaches u[r] · (the row's sum): sampling
//                   from unnormalized weights with a uniform draw per row (ties and an empty row give the last column)
//   route           for n indices into `buckets`: counts[b], offsets[b + 1] (exclusive prefix), order (the sources
//                   grouped by bucket in source order): the expert placement of a router
//   cross_entropy   the composition log_softmax → pick_rows (the loss per row, negated by the caller); its backward is
//                   the row operation cross_entropy_bwd (softmax(x) − one_hot(target)) in Rows.cs
using System.Buffers;
using System.Runtime.InteropServices;

namespace Qiyas;

public static class Routing
{
    /// <summary>top-k of each row in float64: values and indices, largest first, ties by the lower index.</summary>
    public static (double[] Values, int[] Indices) TopKReference(double[] x, int rows, int cols, int k)
    {
        var values = new double[rows * k];
        var indices = new int[rows * k];
        for (int r = 0; r < rows; r++)
        {
            var order = Enumerable.Range(0, cols).OrderByDescending(c => x[r * cols + c]).ThenBy(c => c).Take(k).ToArray();
            for (int i = 0; i < k; i++) { indices[r * k + i] = order[i]; values[r * k + i] = x[r * cols + order[i]]; }
        }

        return (values, indices);
    }

    public static double[] PickReference(double[] x, int rows, int cols, int[] index)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++) y[r] = x[r * cols + index[r]];
        return y;
    }

    public static int[] SampleReference(double[] x, int rows, int cols, double[] u)
    {
        var y = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            double total = 0;
            for (int c = 0; c < cols; c++) total += x[r * cols + c];
            double threshold = u[r] * total, run = 0;
            int pick = cols - 1;
            for (int c = 0; c < cols; c++) { run += x[r * cols + c]; if (run >= threshold) { pick = c; break; } }
            y[r] = pick;
        }

        return y;
    }

    public static (int[] Counts, int[] Offsets, int[] Order) RouteReference(int[] index, int buckets)
    {
        var counts = new int[buckets];
        foreach (int i in index) counts[i]++;
        var offsets = new int[buckets + 1];
        for (int b = 0; b < buckets; b++) offsets[b + 1] = offsets[b] + counts[b];
        var order = new int[index.Length];
        var cursor = (int[])offsets.Clone();
        for (int i = 0; i < index.Length; i++) order[cursor[index[i]]++] = i;
        return (counts, offsets, order);
    }
}

internal static partial class Kernels
{
    /// <summary>top-k of each row: values stored as the type (rows × k), indices s32; returns the lane's name.</summary>
    public static string TopK<TN>(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) where TN : struct, IQiyasNumber<TN>
    {
        var v = new TopKVisitor(x, rows, cols, k, values, indices);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    private struct TopKVisitor(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols * 4, new TopKBody<TN, TLane, TL>(x, rows, cols, k, values, indices));
        }
    }

    private struct TopKBody<TN, TLane, TL>(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
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
                    // k passes of the row's argmax over the values not yet taken (k is small against cols; a partial sort is the step for large k).
                    int taken = 0;
                    for (int i = 0; i < k; i++)
                    {
                        int at = -1;
                        for (int c = 0; c < cols; c++)
                        {
                            bool skip = false;
                            for (int t = 0; t < taken; t++) if (bestAt[t] == c) { skip = true; break; }
                            if (skip) continue;
                            if (at < 0 || Widths.Compare(row[c], row[at]) > 0) at = c;
                        }

                        bestAt[taken] = at; best[taken] = at < 0 ? TL.Zero : row[at]; taken++;
                    }

                    TN.Store(best.AsSpan(0, k), values, r * k);
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
    public static string Pick<TN>(byte[] x, int rows, int cols, int[] index, byte[] output) where TN : struct, IQiyasNumber<TN>
    {
        var v = new PickVisitor(x, rows, cols, index, output);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    private struct PickVisitor(byte[] x, int rows, int cols, int[] index, byte[] output) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var picked = new TLane[rows];
            var one = new TLane[1];
            for (int r = 0; r < rows; r++) { TN.Load(x, r * cols + index[r], one.AsSpan(0, 1)); picked[r] = one[0]; }
            TN.Store(picked, output, 0);
        }
    }

    /// <summary>The sampled column of each row from weights x and a uniform draw u per row (as the type); in the type's float lane.</summary>
    public static string Sample<TN>(byte[] x, int rows, int cols, byte[] u, int[] picks) where TN : struct, IQiyasFloat<TN>
    {
        var v = new SampleVisitor(x, rows, cols, u, picks);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct SampleVisitor(byte[] x, int rows, int cols, byte[] u, int[] picks) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols * 2, new SampleBody<TN, TLane, TL>(x, rows, cols, u, picks));
        }
    }

    private struct SampleBody<TN, TLane, TL>(byte[] x, int rows, int cols, byte[] u, int[] picks) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
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
