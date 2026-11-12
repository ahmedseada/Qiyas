// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on the CPU (Sampling.cs): a task a row. The row's scores loaded into the float lane once and
// turned into its weights there (x / temperature, the largest, exp of the difference: whole vectors, the row padded to
// one with values that are no score); each rule's threshold found by halving the lane's bit patterns between 0 and 1 —
// a pass of vector compares a step (top-k: the count of the weights at least the value; the nucleus: their sum) — so the
// row is never ordered. With top-k the halving stops once few weights are left at or above its value: those are
// gathered and ordered (a few hundred values), the k-th largest read from them and the nucleus summed over them. Then
// the draw: the kept weights' sum, and the first kept column at which the running sum reaches the seed's uniform of it.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered); returns the lane's name.</summary>
    public static string SampleFiltered<TN>(SamplingArgs a, byte[] x, int[] step, int[] picks, byte[]? probability) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        return Mixed<TN, FilteredSampleCall>(new FilteredSampleCall(a, x, step[0], picks, probability));
    }

    private readonly struct FilteredSampleCall(SamplingArgs a, byte[] x, int step, int[] picks, byte[]? probability) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.Rows == 0) return TL.Name;
            var p = probability is null ? null : new TLane[a.Rows];
            Team.For(a.Rows, 1, (long)a.Rows * Math.Max(1, a.Cols) * 32, new FilteredSampleBody<TN, TLane, TL>(a, x, step, picks, p));
            if (p is not null) TN.Store(p, probability, 0);
            return TL.Name;
        }
    }

    private readonly struct FilteredSampleBody<TN, TLane, TL>(SamplingArgs a, byte[] x, int step, int[] picks, TLane[]? probability) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int cols = a.Cols, width = Vector<TLane>.Count, padded = (cols + width - 1) / width * width;
            var row = ArrayPool<TLane>.Shared.Rent(Math.Max(width, padded));
            var top = ArrayPool<TLane>.Shared.Rent(TopGathered);
            var key = Draws.Key(a.Seed);
            TLane temperature = TL.From(a.Temperature), none = TL.From(double.NaN);
            try
            {
                for (int r = first; r < last; r++)
                {
                    var w = row.AsSpan(0, padded);
                    TN.Load(x, (int)((long)r * a.Stride + a.RowOffset), w[..cols]);
                    w[cols..].Fill(none);                                         // past the row: no score (a weight of 0)
                    int pick = -1;
                    TLane p = TL.Zero;
                    if (SampleWeights<TLane, TL>(w, temperature))
                    {
                        TLane t = TL.Zero;
                        int gathered = a.HasTopK ? TopWeights<TLane, TL>(w, cols, a.TopK, top, out t) : 0;
                        if (a.HasTopP) t = gathered > 0 ? NucleusOfTop<TLane, TL>(top.AsSpan(0, gathered), t, TL.From(a.TopP)) : NucleusWeight<TLane, TL>(w, t, TL.From(a.TopP));
                        if (a.HasMinP) t = TL.Max(t, TL.From(a.MinP));
                        (pick, p) = DrawKept<TLane, TL>(w, cols, t, TL.From(Draws.Uniform(key, (uint)step, (uint)r)));
                    }

                    picks[r] = pick;
                    if (probability is not null) probability[r] = p;
                }
            }
            finally
            {
                ArrayPool<TLane>.Shared.Return(row);
                ArrayPool<TLane>.Shared.Return(top);
            }
        }
    }

    // A lane value's bits as an integer (a weight's order is its bits': weights are 0 or above), and back.
    private static ulong LaneBits<T>(T v) where T : unmanaged =>
        typeof(T) == typeof(float) ? BitConverter.SingleToUInt32Bits(Unsafe.As<T, float>(ref v)) : BitConverter.DoubleToUInt64Bits(Unsafe.As<T, double>(ref v));

    private static T LaneOfBits<T>(ulong bits) where T : unmanaged
    {
        if (typeof(T) == typeof(float)) { float f = BitConverter.UInt32BitsToSingle((uint)bits); return Unsafe.As<float, T>(ref f); }
        double d = BitConverter.UInt64BitsToDouble(bits);
        return Unsafe.As<double, T>(ref d);
    }

    private static double LaneValue<T>(T v) where T : unmanaged => typeof(T) == typeof(float) ? Unsafe.As<T, float>(ref v) : Unsafe.As<T, double>(ref v);

    // The row's scores (whole vectors; a NaN is no score) into its weights, in place: s = x / temperature, m the largest,
    // exp(s − m) (0 for no score); a largest of +∞: 1 at each +∞, 0 elsewhere. False when the row has no score above −∞.
    private static bool SampleWeights<TLane, TL>(Span<TLane> w, TLane temperature) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int width = Vector<TLane>.Count;
        var tv = new Vector<TLane>(temperature);
        var lowest = new Vector<TLane>(TL.NegativeInfinity);
        var mv = lowest;
        for (int i = 0; i < w.Length; i += width)
        {
            var s = TL.Div(new Vector<TLane>(w[i..]), tv);
            s.CopyTo(w[i..]);
            mv = TL.Max(mv, TL.Select(TL.IsNaN(s), lowest, s));
        }

        TLane m = TL.NegativeInfinity;
        for (int l = 0; l < width; l++) m = TL.Max(m, mv[l]);
        if (Widths.Compare(m, TL.NegativeInfinity) <= 0) return false;
        TLane top = TL.From(double.PositiveInfinity);
        var zero = Vector<TLane>.Zero;
        if (Widths.Compare(m, top) >= 0)
        {
            var one = new Vector<TLane>(TL.From(1));
            var tops = new Vector<TLane>(top);
            for (int i = 0; i < w.Length; i += width) TL.Select(TL.Equal(new Vector<TLane>(w[i..]), tops), one, zero).CopyTo(w[i..]);
            return true;
        }

        var largest = new Vector<TLane>(m);
        for (int i = 0; i < w.Length; i += width)
        {
            var s = new Vector<TLane>(w[i..]);
            TL.Select(TL.IsNaN(s), zero, TL.Exp(TL.Sub(s, largest))).CopyTo(w[i..]);
        }

        return true;
    }

    // The weights at least t: how many, and their sum (the vector's lanes joined in order).
    private static long CountAtLeast<TLane, TL>(ReadOnlySpan<TLane> w, TLane t) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int width = Vector<TLane>.Count;
        var tv = new Vector<TLane>(t);
        var one = new Vector<TLane>(TL.From(1));
        var acc = Vector<TLane>.Zero;
        for (int i = 0; i < w.Length; i += width) acc = TL.Add(acc, TL.Select(TL.Greater(tv, new Vector<TLane>(w[i..])), Vector<TLane>.Zero, one));
        double count = 0;
        for (int l = 0; l < width; l++) count += LaneValue(acc[l]);
        return (long)count;
    }

    private static TLane SumAtLeast<TLane, TL>(ReadOnlySpan<TLane> w, TLane t) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int width = Vector<TLane>.Count;
        var tv = new Vector<TLane>(t);
        var acc = Vector<TLane>.Zero;
        for (int i = 0; i < w.Length; i += width)
        {
            var v = new Vector<TLane>(w[i..]);
            acc = TL.Add(acc, TL.Select(TL.Greater(tv, v), Vector<TLane>.Zero, v));
        }

        TLane sum = TL.Zero;
        for (int l = 0; l < width; l++) sum = TL.Add(sum, acc[l]);
        return sum;
    }

    // Top-k's threshold: the largest value at which k weights or more are at least it (the k-th largest weight, or a value
    // between it and the next: the same kept set), by halving the bit patterns between 0 and 1; a count of exactly k ends it.
    private static TLane KthLargestWeight<TLane, TL>(ReadOnlySpan<TLane> w, int k) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        ulong lo = 0, hi = LaneBits(TL.From(1));
        while (lo < hi)
        {
            ulong mid = lo + (hi - lo + 1) / 2;
            long count = CountAtLeast<TLane, TL>(w, LaneOfBits<TLane>(mid));
            if (count == k) return LaneOfBits<TLane>(mid);
            if (count > k) lo = mid;
            else hi = mid - 1;
        }

        return LaneOfBits<TLane>(lo);
    }

    /// <summary>The most weights top-k gathers and orders (past it, more equal weights than that: the halving runs to its end).</summary>
    private const int TopGathered = 1024;

    // Top-k's threshold, the k-th largest weight: the halving until k weights or more, and no more than four times k (64
    // or more, `top`'s room at most), are at least its value; those gathered into `top` and ordered (ascending), the k-th
    // largest read from them. Returns how many were gathered: 0 when k is past half of `top`'s room, or more equal weights
    // than the room are at the k-th largest (the threshold then the halving's own).
    private static int TopWeights<TLane, TL>(ReadOnlySpan<TLane> w, int cols, int k, TLane[] top, out TLane threshold) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        if (k > TopGathered / 2) { threshold = KthLargestWeight<TLane, TL>(w, k); return 0; }
        int room = Math.Min(TopGathered, Math.Max(64, 4 * k));                 // few to order: the halving's passes are cheaper
        ulong lo = 0, hi = LaneBits(TL.From(1));
        long count = cols;
        while (count > room && lo < hi)
        {
            ulong mid = lo + (hi - lo + 1) / 2;
            long at = CountAtLeast<TLane, TL>(w, LaneOfBits<TLane>(mid));
            if (at >= k) { lo = mid; count = at; }
            else hi = mid - 1;
        }

        threshold = LaneOfBits<TLane>(lo);
        if (count > room) return 0;
        // The weights at least that value: a vector of the row at a time, its lanes looked at only where one of them is.
        int n = 0, width = Vector<TLane>.Count;
        var least = new Vector<TLane>(threshold);
        for (int i = 0; i < cols; i += width)
        {
            if (!Vector.GreaterThanOrEqualAny(new Vector<TLane>(w[i..]), least)) continue;
            for (int c = i, end = Math.Min(cols, i + width); c < end; c++) if (Widths.Compare(w[c], threshold) >= 0) top[n++] = w[c];
        }

        top.AsSpan(0, n).Sort();
        threshold = top[n - k];
        return n;
    }

    // The nucleus's threshold among the gathered weights (ascending) at least t: the largest weight at which the weights at
    // least it, summed from the largest, reach p of the sum of those at least t (equal weights kept together).
    private static TLane NucleusOfTop<TLane, TL>(ReadOnlySpan<TLane> top, TLane t, TLane p) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int first = top.Length;
        TLane mass = TL.Zero;
        for (int i = top.Length - 1; i >= 0 && Widths.Compare(top[i], t) >= 0; i--) { mass = TL.Add(mass, top[i]); first = i; }
        TLane target = TL.Mul(p, mass), run = TL.Zero;
        for (int i = top.Length - 1; i >= first; i--)
        {
            run = TL.Add(run, top[i]);
            if (i > first && Widths.Compare(top[i - 1], top[i]) == 0) continue;
            if (Widths.Compare(run, target) >= 0) return top[i];
        }

        return t;
    }

    // The nucleus's threshold among the weights at least t: the largest value at which the weights at least it sum to p of
    // the weights at least t, by the same halving.
    private static TLane NucleusWeight<TLane, TL>(ReadOnlySpan<TLane> w, TLane t, TLane p) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        TLane target = TL.Mul(p, SumAtLeast<TLane, TL>(w, t));
        ulong lo = LaneBits(t), hi = LaneBits(TL.From(1));
        while (lo < hi)
        {
            ulong mid = lo + (hi - lo + 1) / 2;
            if (Widths.Compare(SumAtLeast<TLane, TL>(w, LaneOfBits<TLane>(mid)), target) >= 0) lo = mid;
            else hi = mid - 1;
        }

        return LaneOfBits<TLane>(lo);
    }

    // The draw among the kept weights (above 0 and at least t) of the row's cols columns: the first kept column at which
    // the running sum reaches u times the kept sum (the last kept one when rounding leaves none; −1 when none is kept),
    // and its weight over the sum.
    private static (int Pick, TLane Probability) DrawKept<TLane, TL>(ReadOnlySpan<TLane> w, int cols, TLane t, TLane u) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        TLane total = SumAtLeast<TLane, TL>(w, t), target = TL.Mul(u, total), run = TL.Zero;
        int pick = -1, last = -1;
        for (int c = 0; c < cols; c++)
        {
            TLane v = w[c];
            if (Widths.Compare(v, TL.Zero) <= 0 || Widths.Compare(v, t) < 0) continue;
            last = c;
            run = TL.Add(run, v);
            if (Widths.Compare(run, target) >= 0) { pick = c; break; }
        }

        if (pick < 0) pick = last;
        return (pick, pick < 0 ? TL.Zero : TL.Div(w[pick], total));
    }
}
