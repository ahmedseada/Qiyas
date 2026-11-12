// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on the CPU (Abstraction/Sampling.cs has the rule): a task a row. The row's scores read once in the
// type's natural lane (exact) and taken into its float lane; the first-ranked score by Adad's order, the weights exp((x −
// m) / temperature) over whole vectors (where top-k gathered its columns, theirs alone). Each rule keeps a prefix of the
// order: where one is on, each column's order key (Routing.OrderKey: its stored score as an unsigned integer that orders
// as Adad's order does) and the prefix's last column found — top-k's in one pass gathering the keys at least a rising
// threshold (StreamTop), else by halving the keys, a pass of vector compares a step counting the keys at least a value
// until few are left; those gathered and ordered (the nucleus summed over them in order), or summing their weights (the
// nucleus over a row top-k did not narrow) — and equal keys at the end of the prefix cut by column. Then
// the draw: the kept weights' sum, and the column at which their running sum, in column order, passes the seed's uniform
// of it.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered); returns the lane's name.</summary>
    public static string SampleFiltered<TN>(SamplingArgs a, byte[] x, int[] step, int[] picks, byte[]? probability) where TN : struct, IAdad<TN>
    {
        a.Validate();
        if (TN.Bits == 1) throw new NotSupportedException($"sampling: {TN.Name} holds truth values, not scores");
        return InFloatLane<TN, SampleCall>(new SampleCall(a, x, step[0], picks, probability));
    }

    private readonly struct SampleCall(SamplingArgs a, byte[] x, int step, int[] picks, byte[]? probability) : IFloatOf
    {
        public string Run<TN, TE, TLane, TL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.Rows == 0) return TL.Name;
            var p = probability is null ? null : new TLane[a.Rows];
            Team.For(a.Rows, 1, (long)a.Rows * Math.Max(1, a.Cols) * 32, new SampleBody<TN, TE, TLane, TL>(a, x, step, picks, p));
            if (p is not null) MemoryMarshal.AsBytes(p.AsSpan()).CopyTo(probability);
            return TL.Name;
        }
    }

    /// <summary>The most columns top-k gathers and orders (past it, the halving runs to its end and equal keys are cut by column).</summary>
    private const int TopGathered = 1024;

    private readonly struct SampleBody<TN, TE, TLane, TL>(SamplingArgs a, byte[] x, int step, int[] picks, TLane[]? probability) : Team.IBody
        where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int cols = a.Cols, width = Vector<TLane>.Count, padded = (cols + width - 1) / width * width;
            var values = ArrayPool<TE>.Shared.Rent(cols);
            var row = ArrayPool<TLane>.Shared.Rent(Math.Max(width, padded));
            bool ranked = a.HasTopK || a.HasTopP;
            var keys = ranked ? ArrayPool<ulong>.Shared.Rent(cols) : null;
            var top = ranked ? ArrayPool<(ulong, int)>.Shared.Rent(TopGathered) : null;
            var candidates = ranked ? ArrayPool<int>.Shared.Rent(TopGathered) : null;
            var sorted = a.HasTopK ? ArrayPool<ulong>.Shared.Rent(TopGathered) : null;
            var few = a.HasTopK ? ArrayPool<TLane>.Shared.Rent(TopGathered + width) : null;
            var key = Draws.Key(a.Seed);
            TLane temperature = TL.From(a.Temperature), q = TL.From(a.MinP);
            try
            {
                for (int r = first; r < last; r++)
                {
                    var w = row.AsSpan(0, padded);
                    var v = values.AsSpan(0, cols);
                    TN.Load(x, (int)((long)r * a.Stride + a.RowOffset), v);
                    if (typeof(TE) == typeof(TLane)) MemoryMarshal.Cast<TE, TLane>(v).CopyTo(w);
                    else Widths.Load<TE, TLane>(v, w[..cols]);
                    w[cols..].Fill(TL.NegativeInfinity);                         // past the row: a weight of 0, never kept
                    TLane m = FirstRanked<TLane, TL>(w);
                    double md = LaneValue(m);
                    if (!double.IsFinite(md))
                    {
                        // The weights are NaN (x − m is NaN at m's own column): no column.
                        picks[r] = -1;
                        if (probability is not null) probability[r] = TL.From(double.NaN);
                        continue;
                    }

                    var b = new Boundary(false, 0, 0);
                    int gathered = 0;
                    if (ranked)
                    {
                        OrderKeys<TE>(v, keys.AsSpan(0, cols));
                        if (a.HasTopK) (b, gathered) = TopBoundary(keys.AsSpan(0, cols), a.TopK, top!, sorted!);
                    }

                    // The weights: where top-k gathered its columns, nothing past them is read, so theirs alone (the same
                    // vector operation on each value, so the same weights).
                    if (gathered > 0) WeighAt<TLane, TL>(w, top.AsSpan(0, gathered), m, temperature, few!);
                    else Weigh<TLane, TL>(w, m, temperature);
                    if (a.HasTopP)
                        b = gathered > 0 ? NucleusOfTop<TLane, TL>(top.AsSpan(0, gathered), w, a.TopK, TL.From(a.TopP)) : NucleusBoundary<TLane, TL>(keys.AsSpan(0, cols), w, b, TL.From(a.TopP));

                    var u = TL.From(Draws.Uniform(key, (uint)step, (uint)r));
                    var (pick, p) = gathered > 0 ? DrawOfTop<TLane, TL>(w[..cols], top.AsSpan(0, gathered), b, a.HasMinP, q, u, candidates!)
                        : Draw<TLane, TL>(w[..cols], keys, b, a.HasMinP, q, u);
                    picks[r] = pick;
                    if (probability is not null) probability[r] = p;
                }
            }
            finally
            {
                ArrayPool<TE>.Shared.Return(values);
                ArrayPool<TLane>.Shared.Return(row);
                if (keys is not null) ArrayPool<ulong>.Shared.Return(keys);
                if (top is not null) ArrayPool<(ulong, int)>.Shared.Return(top);
                if (candidates is not null) ArrayPool<int>.Shared.Return(candidates);
                if (sorted is not null) ArrayPool<ulong>.Shared.Return(sorted);
                if (few is not null) ArrayPool<TLane>.Shared.Return(few);
            }
        }
    }

    /// <summary>The last column of a kept prefix of the order: its key and column (On false: no prefix, every column).</summary>
    private readonly record struct Boundary(bool On, ulong Key, int Col)
    {
        /// <summary>Column c (key k) is within the prefix: a larger key, or the same key at or before the column.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Holds(ulong k, int c) => !On || k > Key || (k == Key && c <= Col);
    }

    private static double LaneValue<T>(T v) where T : unmanaged => typeof(T) == typeof(float) ? Unsafe.As<T, float>(ref v) : Unsafe.As<T, double>(ref v);

    // The first-ranked value of a row in its float lane: a NaN where it holds one, else the largest.
    private static TLane FirstRanked<TLane, TL>(ReadOnlySpan<TLane> w) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int width = Vector<TLane>.Count;
        var mv = new Vector<TLane>(TL.NegativeInfinity);
        var nan = Vector<TLane>.Zero;
        for (int i = 0; i < w.Length; i += width)
        {
            var s = new Vector<TLane>(w[i..]);
            nan |= TL.IsNaN(s);
            mv = TL.Max(mv, s);
        }

        if (nan != Vector<TLane>.Zero) return TL.From(double.NaN);
        TLane m = TL.NegativeInfinity;
        for (int l = 0; l < width; l++) m = TL.Max(m, mv[l]);
        return m;
    }

    // The row's values into its weights, in place: exp((x − m) / temperature) over whole vectors.
    private static void Weigh<TLane, TL>(Span<TLane> w, TLane m, TLane temperature) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int width = Vector<TLane>.Count;
        var mv = new Vector<TLane>(m);
        var tv = new Vector<TLane>(temperature);
        for (int i = 0; i < w.Length; i += width) TL.Exp(TL.Div(TL.Sub(new Vector<TLane>(w[i..]), mv), tv)).CopyTo(w[i..]);
    }

    // Weigh's weights at the gathered columns alone: their values side by side (the last vector filled with m), Weigh over
    // them, each put back at its column.
    private static void WeighAt<TLane, TL>(Span<TLane> w, ReadOnlySpan<(ulong Key, int Col)> at, TLane m, TLane temperature, TLane[] few)
        where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int width = Vector<TLane>.Count, n = at.Length, padded = (n + width - 1) / width * width;
        var side = few.AsSpan(0, padded);
        for (int i = 0; i < n; i++) side[i] = w[at[i].Col];
        side[n..].Fill(m);
        Weigh<TLane, TL>(side, m, temperature);
        for (int i = 0; i < n; i++) w[at[i].Col] = side[i];
    }

    // The keys at least v.
    private static long CountAtLeast(ReadOnlySpan<ulong> k, ulong v)
    {
        int width = Vector<ulong>.Count, i = 0;
        var vv = new Vector<ulong>(v);
        var acc = Vector<ulong>.Zero;
        for (; i + width <= k.Length; i += width) acc -= Vector.GreaterThanOrEqual(new Vector<ulong>(k[i..]), vv);
        long count = (long)Vector.Sum(acc);
        for (; i < k.Length; i++) if (k[i] >= v) count++;
        return count;
    }

    // Top-k's prefix: the k-th column of the order. The largest key at least k columns reach, by halving the keys' range
    // until few columns are left at or above it (64 or more, four times k, `top`'s room at most); those gathered and
    // ordered (`top`: key descending, then column), the k-th read from them. Where equal keys keep more than the room the
    // halving runs to its end and the key's columns are cut by column order. Returns how many were gathered (0: none).
    private static (Boundary, int) TopBoundary(ReadOnlySpan<ulong> k, int topK, (ulong Key, int Col)[] top, ulong[] sorted)
    {
        var streamed = StreamTop(k, topK, top, sorted);
        if (streamed.Item2 > 0) return streamed;
        var (lo, hi) = KeyRange(k);
        int room = Math.Min(TopGathered, Math.Max(64, 4 * topK));
        long count = k.Length;
        if (count > room && lo < hi) (lo, hi, count) = Narrowed(k, topK, lo, hi);
        while (count > room && lo < hi)
        {
            ulong mid = lo + (hi - lo) / 2 + 1;
            long at = CountAtLeast(k, mid);
            if (at >= topK) { lo = mid; count = at; }
            else hi = mid - 1;
        }

        if (count <= room)
        {
            int n = 0;
            for (int c = 0; c < k.Length; c++) if (k[c] >= lo) top[n++] = (k[c], c);
            Array.Sort(top, 0, n, ByOrder.Instance);
            return (new Boundary(true, top[topK - 1].Key, top[topK - 1].Col), n);
        }

        // Equal keys past the room: lo is the k-th column's key; its columns cut by column order. Where the k fit `top`, they
        // are gathered as the room's are (the keys above lo, then lo's first columns in column order: exactly the first k of
        // the order), so the nucleus and the draw read them alone.
        long above = lo == ulong.MaxValue ? 0 : CountAtLeast(k, lo + 1);
        long need = topK - above;
        if (topK <= top.Length)
        {
            int n = 0;
            for (int c = 0; c < k.Length; c++)
            {
                if (k[c] > lo) top[n++] = (k[c], c);
                else if (k[c] == lo && need > 0) { top[n++] = (k[c], c); need--; }
            }

            Array.Sort(top, 0, n, ByOrder.Instance);
            return (new Boundary(true, top[topK - 1].Key, top[topK - 1].Col), n);
        }

        for (int c = 0; c < k.Length; c++)
            if (k[c] == lo && --need == 0) return (new Boundary(true, lo, c), 0);
        return (new Boundary(true, lo, k.Length - 1), 0);
    }

    // Top-k's prefix in one pass, where its room (TopBoundary's) holds twice k: the keys at least a threshold gathered with
    // their columns as the pass meets them (a vector of keys all below it skipped whole), the threshold raised to the k-th
    // largest gathered key each time the room fills and the gathered below it dropped. The threshold is the k-th of a part
    // of the row, so at most the row's k-th: the k first of the order stay gathered. At the end the gathered at least
    // their k-th key, ordered (key descending, then column), the k-th read from them: TopBoundary's prefix and its first k
    // the same. Equal keys that keep the room full: 0, the halving's way instead.
    private static (Boundary, int) StreamTop(ReadOnlySpan<ulong> k, int topK, (ulong Key, int Col)[] top, ulong[] sorted)
    {
        int room = Math.Min(TopGathered, Math.Max(64, 4 * topK));
        if (2 * topK > room || k.Length <= room) return (default, 0);
        int width = Vector<ulong>.Count, n = 0, c = 0;
        ulong t = 0;
        var tv = Vector<ulong>.Zero;
        for (; c + width <= k.Length; c += width)
        {
            var v = new Vector<ulong>(k[c..]);
            if (!Vector.GreaterThanOrEqualAny(v, tv)) continue;
            for (int l = 0; l < width; l++)
            {
                if (v[l] < t) continue;
                top[n++] = (v[l], c + l);
                if (n < room) continue;
                (t, n) = Raise(top, n, topK, sorted);
                if (n >= room - 1) return (default, 0);
                tv = new Vector<ulong>(t);
            }
        }

        for (; c < k.Length; c++)
        {
            if (k[c] < t) continue;
            top[n++] = (k[c], c);
            if (n < room) continue;
            (t, n) = Raise(top, n, topK, sorted);
            if (n >= room - 1) return (default, 0);
        }

        (_, n) = Raise(top, n, topK, sorted);
        Array.Sort(top, 0, n, ByOrder.Instance);
        return (new Boundary(true, top[topK - 1].Key, top[topK - 1].Col), n);

        // The k-th largest of the n gathered keys, and the gathered kept at least it (in the order they came).
        static (ulong, int) Raise((ulong Key, int Col)[] top, int n, int topK, ulong[] sorted)
        {
            var keys = sorted.AsSpan(0, n);
            for (int i = 0; i < n; i++) keys[i] = top[i].Key;
            keys.Sort();
            ulong t = keys[n - topK];
            int kept = 0;
            for (int i = 0; i < n; i++) if (top[i].Key >= t) top[kept++] = top[i];
            return (t, kept);
        }
    }

    // The keys' least and largest (vectors of keys, then the tail).
    private static (ulong Lo, ulong Hi) KeyRange(ReadOnlySpan<ulong> k)
    {
        int width = Vector<ulong>.Count, i = 0;
        var vlo = new Vector<ulong>(ulong.MaxValue);
        var vhi = Vector<ulong>.Zero;
        for (; i + width <= k.Length; i += width)
        {
            var v = new Vector<ulong>(k[i..]);
            vlo = Vector.Min(vlo, v);
            vhi = Vector.Max(vhi, v);
        }

        ulong lo = ulong.MaxValue, hi = 0;
        for (int l = 0; l < width; l++) { lo = Math.Min(lo, vlo[l]); hi = Math.Max(hi, vhi[l]); }
        for (; i < k.Length; i++) { lo = Math.Min(lo, k[i]); hi = Math.Max(hi, k[i]); }
        return (lo, hi);
    }

    // One pass over the keys narrows the halving's range: a histogram of [lo, hi] in Bins equal bins, walked from the top to
    // the bin where the keys at least its lower edge first reach topK. That edge has at least topK keys at or above it and
    // the bin's upper edge + 1 fewer: the halving's own invariants, so it ends where it would have, in fewer passes (none
    // where that bin's columns fit the room). Returns the narrowed range and the count at its lower edge.
    private const int Bins = 2048;

    private static (ulong Lo, ulong Hi, long Count) Narrowed(ReadOnlySpan<ulong> k, int topK, ulong lo, ulong hi)
    {
        int shift = Math.Max(0, 64 - BitOperations.LeadingZeroCount(hi - lo) - 11);        // (hi − lo) >> shift < Bins
        Span<int> bins = stackalloc int[Bins];
        bins.Clear();
        foreach (ulong v in k) bins[(int)((v - lo) >> shift)]++;
        int last = (int)((hi - lo) >> shift);
        long above = 0;
        for (int b = last; b >= 0; b--)
        {
            long at = above + bins[b];
            if (at >= topK)
            {
                ulong edge = lo + ((ulong)b << shift);
                return (edge, b == last ? hi : edge + ((1UL << shift) - 1), at);
            }

            above = at;
        }

        return (lo, hi, k.Length);
    }

    // keys[c] = Routing.OrderKey(v[c]): a float lane's whole vectors by the same rule (±0 the key of 0, a NaN the largest
    // key, a negative value its bits inverted, any other its sign bit set), widened to 64 bits; any other lane value by
    // value.
    private static void OrderKeys<TE>(ReadOnlySpan<TE> v, Span<ulong> keys) where TE : unmanaged
    {
        int c = 0;
        if (typeof(TE) == typeof(float))
        {
            var f = MemoryMarshal.Cast<TE, float>(v);
            int width = Vector<float>.Count, half = width / 2;
            var sign = new Vector<uint>(0x80000000u);
            for (; c + width <= f.Length; c += width)
            {
                var x = new Vector<float>(f[c..]);
                var u = Vector.AndNot(Vector.AsVectorUInt32(x), Vector.AsVectorUInt32(Vector.Equals(x, Vector<float>.Zero)));
                var key = Vector.ConditionalSelect(Vector.Equals(u & sign, sign), ~u, u | sign);
                key = Vector.ConditionalSelect(Vector.AsVectorUInt32(Vector.Equals(x, x)), key, new Vector<uint>(uint.MaxValue));
                Vector.Widen(key, out var k0, out var k1);
                k0.CopyTo(keys[c..]);
                k1.CopyTo(keys[(c + half)..]);
            }
        }
        else if (typeof(TE) == typeof(int))
        {
            // A signed integer's key: its value with the sign bit flipped (Routing.OrderKey), widened to 64 bits first.
            var e = MemoryMarshal.Cast<TE, int>(v);
            int width = Vector<int>.Count, half = width / 2;
            var sign = new Vector<long>(long.MinValue);
            for (; c + width <= e.Length; c += width)
            {
                Vector.Widen(new Vector<int>(e[c..]), out var k0, out var k1);
                Vector.AsVectorUInt64(k0 ^ sign).CopyTo(keys[c..]);
                Vector.AsVectorUInt64(k1 ^ sign).CopyTo(keys[(c + half)..]);
            }
        }
        else if (typeof(TE) == typeof(long))
        {
            var e = MemoryMarshal.Cast<TE, long>(v);
            int width = Vector<long>.Count;
            var sign = new Vector<long>(long.MinValue);
            for (; c + width <= e.Length; c += width) Vector.AsVectorUInt64(new Vector<long>(e[c..]) ^ sign).CopyTo(keys[c..]);
        }
        else if (typeof(TE) == typeof(double))
        {
            var d = MemoryMarshal.Cast<TE, double>(v);
            int width = Vector<double>.Count;
            var sign = new Vector<ulong>(0x8000000000000000ul);
            for (; c + width <= d.Length; c += width)
            {
                var x = new Vector<double>(d[c..]);
                var u = Vector.AndNot(Vector.AsVectorUInt64(x), Vector.AsVectorUInt64(Vector.Equals(x, Vector<double>.Zero)));
                var key = Vector.ConditionalSelect(Vector.Equals(u & sign, sign), ~u, u | sign);
                Vector.ConditionalSelect(Vector.AsVectorUInt64(Vector.Equals(x, x)), key, new Vector<ulong>(ulong.MaxValue)).CopyTo(keys[c..]);
            }
        }

        for (; c < v.Length; c++) keys[c] = Routing.OrderKey(v[c]);
    }

    private sealed class ByOrder : IComparer<(ulong Key, int Col)>
    {
        public static readonly ByOrder Instance = new();

        public int Compare((ulong Key, int Col) a, (ulong Key, int Col) b) => a.Key != b.Key ? b.Key.CompareTo(a.Key) : a.Col.CompareTo(b.Col);
    }

    // The nucleus within top-k's gathered columns (ordered): the shortest prefix of the first k whose weights, summed in
    // order, reach p times the sum of the k.
    private static Boundary NucleusOfTop<TLane, TL>(ReadOnlySpan<(ulong Key, int Col)> top, ReadOnlySpan<TLane> w, int k, TLane p) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        TLane mass = TL.Zero, run = TL.Zero;
        for (int i = 0; i < k; i++) mass = TL.Add(mass, w[top[i].Col]);
        TLane target = TL.Mul(p, mass);
        int end = k - 1;
        for (int i = 0; i < k; i++)
        {
            run = TL.Add(run, w[top[i].Col]);
            if (Widths.Compare(run, target) >= 0) { end = i; break; }
        }

        return new Boundary(true, top[end].Key, top[end].Col);
    }

    // The nucleus over the columns within a prefix (or the row): the largest key at which the weights of the columns at least
    // it reach p of them all, by halving the keys' range (a sum a pass), then its columns cut by column order where the
    // running sum reaches the target.
    private static Boundary NucleusBoundary<TLane, TL>(ReadOnlySpan<ulong> k, ReadOnlySpan<TLane> w, Boundary within, TLane p) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        ulong lo = ulong.MaxValue, hi = 0;
        for (int c = 0; c < k.Length; c++) if (within.Holds(k[c], c)) { lo = Math.Min(lo, k[c]); hi = Math.Max(hi, k[c]); }
        TLane Mass(ulong v, ReadOnlySpan<ulong> keys, ReadOnlySpan<TLane> weights)
        {
            TLane s = TL.Zero;
            for (int c = 0; c < keys.Length; c++) if (keys[c] >= v && within.Holds(keys[c], c)) s = TL.Add(s, weights[c]);
            return s;
        }

        TLane target = TL.Mul(p, Mass(lo, k, w));
        while (lo < hi)
        {
            ulong mid = lo + (hi - lo) / 2 + 1;
            if (Widths.Compare(Mass(mid, k, w), target) >= 0) lo = mid;
            else hi = mid - 1;
        }

        TLane run = lo == ulong.MaxValue ? TL.Zero : Mass(lo + 1, k, w);
        int at = -1;
        for (int c = 0; c < k.Length; c++)
        {
            if (k[c] != lo || !within.Holds(k[c], c)) continue;
            at = c;
            run = TL.Add(run, w[c]);
            if (Widths.Compare(run, target) >= 0) break;
        }

        return new Boundary(true, lo, at);
    }

    // Column c is kept: within the prefix and, with min-p, of weight at least q.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Kept<TLane>(ReadOnlySpan<TLane> w, ulong[]? keys, Boundary b, bool minP, TLane q, int c) where TLane : unmanaged =>
        b.Holds(b.On ? keys![c] : 0, c) && (!minP || Widths.Compare(w[c], q) >= 0);

    // The draw among the kept columns (within the prefix; with min-p, of weight at least q): the column at which the running
    // sum of the kept weights, in column order, passes u times their sum (the last kept column of positive weight when
    // rounding leaves none), and its weight over the sum. A run of whole vectors at a time where every column is a candidate.
    private static (int Pick, TLane Probability) Draw<TLane, TL>(ReadOnlySpan<TLane> w, ulong[]? keys, Boundary b, bool minP, TLane q, TLane u)
        where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int cols = w.Length, width = Vector<TLane>.Count;
        TLane total = TL.Zero;
        if (!b.On)
        {
            var qv = new Vector<TLane>(q);
            var acc = Vector<TLane>.Zero;
            int i = 0;
            for (; i + width <= cols; i += width)
            {
                var e = new Vector<TLane>(w[i..]);
                acc = TL.Add(acc, minP ? TL.Select(TL.Greater(qv, e), Vector<TLane>.Zero, e) : e);
            }

            for (int l = 0; l < width; l++) total = TL.Add(total, acc[l]);
            for (; i < cols; i++) if (Kept(w, keys, b, minP, q, i)) total = TL.Add(total, w[i]);
        }
        else
            for (int c = 0; c < cols; c++) if (Kept(w, keys, b, minP, q, c)) total = TL.Add(total, w[c]);

        if (!(LaneValue(total) > 0)) return (-1, TL.From(double.NaN));
        TLane target = TL.Mul(u, total), run = TL.Zero;
        int pick = -1, last = -1;
        for (int c = 0; c < cols; c++)
        {
            if (!Kept(w, keys, b, minP, q, c) || !(LaneValue(w[c]) > 0)) continue;
            last = c;
            run = TL.Add(run, w[c]);
            if (Widths.Compare(run, target) > 0) { pick = c; break; }
        }

        if (pick < 0) pick = last;
        return (pick, TL.Div(w[pick], total));
    }

    // Draw's rule where top-k gathered its columns (every column within the prefix is among them): the kept ones (within
    // the prefix; with min-p, of weight at least q) put in column order, then Draw's sums in Draw's order over those
    // columns alone.
    private static (int Pick, TLane Probability) DrawOfTop<TLane, TL>(ReadOnlySpan<TLane> w, ReadOnlySpan<(ulong Key, int Col)> top, Boundary b, bool minP, TLane q, TLane u, int[] candidates)
        where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int n = 0;
        foreach (var (k, c) in top) if (b.Holds(k, c) && (!minP || Widths.Compare(w[c], q) >= 0)) candidates[n++] = c;
        var kept = candidates.AsSpan(0, n);
        kept.Sort();
        TLane total = TL.Zero;
        foreach (int c in kept) total = TL.Add(total, w[c]);
        if (!(LaneValue(total) > 0)) return (-1, TL.From(double.NaN));
        TLane target = TL.Mul(u, total), run = TL.Zero;
        int pick = -1, last = -1;
        foreach (int c in kept)
        {
            if (!(LaneValue(w[c]) > 0)) continue;
            last = c;
            run = TL.Add(run, w[c]);
            if (Widths.Compare(run, target) > 0) { pick = c; break; }
        }

        if (pick < 0) pick = last;
        return (pick, TL.Div(w[pick], total));
    }

    /// <summary>keys[i] = the order key (Routing.OrderKey) of x's value i of TN, read as sampling reads it (its float lane
    /// for a float type, else its natural lane): the key proven against Routing.Order.</summary>
    internal static void OrderKeys<TN>(byte[] x, ulong[] keys, int n) where TN : struct, IAdad<TN> => InFloatLane<TN, KeysCall>(new KeysCall(x, keys, n));

    private readonly struct KeysCall(byte[] x, ulong[] keys, int n) : IFloatOf
    {
        public string Run<TN, TE, TLane, TL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            var v = new TE[n];
            TN.Load(x, 0, v);
            for (int i = 0; i < n; i++) keys[i] = Routing.OrderKey(v[i]);
            return TL.Name;
        }
    }
}
