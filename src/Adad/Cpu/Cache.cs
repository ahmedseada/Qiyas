// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on the CPU (Abstraction/Cache.cs, ICache), the position read from its index buffer where it is.
//
//   cache_write       a task a source row: the row into TN's float lane, taken into the cache type's element-wise lane with
//                     one rounding (Convert's rule: an f64 lane's value to a narrow float rounded to odd at float32 first)
//                     and stored at the position's row (whole 32-bit words of the cache's type: no word is two tasks').
//                     A scaled cache: every row's scale max |x| / TC.Max in the lane first (a task a row), the scales
//                     stored as TN at their rows (one store a storage unit of TN: a value, or a block format's block with
//                     its other values kept, as the type stores a part of a block) and read back as stored; then each
//                     code x / scale by the stored scale (an integer type's nearest, ties to even), held to TC's range,
//                     rounded to TC once (Quantize's codes, a row its block).
//   attention_cached  the decode's query at a time (Attention.cs) over the rows the position fills: each row of K and V read
//                     where it is stored as TC into its element-wise lane and taken into the float lane, a scaled cache's
//                     row multiplied by its scale there (code·scale: the row's values, never stored); the scores capped by
//                     the map's softcap where there is a cap.
//   decoder_mask      aligned tiles of the mask (a stored unit of TN written by one task).

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>The source's rows into the cache at the position (ICache.CacheWrite); returns the lane's name.</summary>
    public static string CacheWrite<TN, TC>(CacheView c, int steps, int[] position, byte[] source, byte[] cache, byte[]? scales)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        c.Validate(Fmt.Of<TC>());
        if (steps < 0 || (long)c.Heads * steps * c.D > int.MaxValue) throw new ArgumentException($"cache: a write of {steps} rows a head");
        return Mixed<TN, CacheWriteCall<TC>>(new CacheWriteCall<TC>(c, steps, position[0], source, cache, scales, false));
    }

    // lanes: the source's rows are the lane's values in its own storage (a composition's rows: CacheWriteRope), the scales
    // still TN's.
    private readonly struct CacheWriteCall<TC>(CacheView c, int steps, int position, byte[] source, byte[] cache, byte[]? scales, bool lanes) : IMixed where TC : struct, IAdad<TC>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (c.Heads == 0 || steps == 0 || c.D == 0) return TL.Name;
            int rows = c.Heads * steps;
            TLane[]? rowScales = null;
            if (scales is not null)
            {
                rowScales = new TLane[rows];
                Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * c.D * 2, new CacheScaleBody<TN, TLane, TL>(c.D, source, lanes, TL.From(TC.Max), rowScales));
                StoreRowScales<TN, TLane>(c, steps, position, rowScales, scales);
            }

            var write = new CacheWriteVisitor<TN, TS, TLane, TL>(c, steps, position, source, lanes, cache, rowScales);
            TC.Elementwise(ref write);
            return TL.Name;
        }
    }

    // Each source row's scale max |x| / max (max(hi, −lo) of its range with 0) in the lane.
    // Row r of a write's source into the lane: TN's values, or (lanes) the lane's own.
    private static void SourceRow<TN, TLane>(byte[] source, bool lanes, int r, Span<TLane> x) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        if (lanes) MemoryMarshal.Cast<byte, TLane>(source.AsSpan()).Slice(r * x.Length, x.Length).CopyTo(x);
        else TN.Load(source, r * x.Length, x);
    }

    private readonly struct CacheScaleBody<TN, TLane, TL>(int d, byte[] source, bool laneRows, TLane max, TLane[] rowScales) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int lanes = Vector<TLane>.Count;
            var v = ArrayPool<TLane>.Shared.Rent(d);
            try
            {
                var x = v.AsSpan(0, d);
                for (int r = first; r < last; r++)
                {
                    SourceRow<TN, TLane>(source, laneRows, r, x);
                    Vector<TLane> vlo = Vector<TLane>.Zero, vhi = Vector<TLane>.Zero;
                    TLane lo = TL.Zero, hi = TL.Zero;
                    int i = 0;
                    for (; i + lanes <= d; i += lanes) { var xv = new Vector<TLane>(x[i..]); vlo = Vector.Min(vlo, xv); vhi = Vector.Max(vhi, xv); }
                    for (; i < d; i++) { if (Widths.Compare(x[i], lo) < 0) lo = x[i]; if (Widths.Compare(x[i], hi) > 0) hi = x[i]; }
                    for (int l = 0; l < lanes; l++) { if (Widths.Compare(vlo[l], lo) < 0) lo = vlo[l]; if (Widths.Compare(vhi[l], hi) > 0) hi = vhi[l]; }
                    rowScales[r] = TL.Div(TL.Max(hi, TL.Sub(TL.Zero, lo)), max);
                }
            }
            finally { ArrayPool<TLane>.Shared.Return(v); }
        }
    }

    // The written rows' scales (rowScales[head·steps + step], lane values) stored as TN at head·Capacity + row, then read
    // back as stored: a store a head's run for a type of whole values; for a type whose values share a storage unit (a block
    // format's block), a store a touched unit, its other values kept (one rounding of the unit, as the devices' encoders).
    private static void StoreRowScales<TN, TLane>(CacheView c, int steps, int position, TLane[] rowScales, byte[] scales) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        var (w0, w1) = c.Written(position, steps);
        if (w1 <= w0) return;
        int cap = c.Capacity, run = w1 - w0, t0 = w0 - position, g = TN.Granularity, total = (int)c.Rows;
        if (g <= 1)
        {
            for (int h = 0; h < c.Heads; h++) TN.Store(rowScales.AsSpan(h * steps + t0, run), scales, h * cap + w0);
        }
        else
        {
            Span<TLane> unit = stackalloc TLane[g];
            int done = -1;
            for (int h = 0; h < c.Heads; h++)
            {
                int first = h * cap + w0, end = first + run;
                for (int u = Math.Max(first / g, done + 1); u <= (end - 1) / g; u++)
                {
                    int a = u * g, e = Math.Min(a + g, total);
                    var values = unit[..(e - a)];
                    TN.Load(scales, a, values);
                    for (int x = a; x < e; x++)
                    {
                        int r = x % cap;
                        if (r >= w0 && r < w1) values[x - a] = rowScales[x / cap * steps + (r - position)];
                    }

                    TN.Store(values, scales, a);
                    done = u;
                }
            }
        }

        for (int h = 0; h < c.Heads; h++) TN.Load(scales, h * cap + w0, rowScales.AsSpan(h * steps + t0, run));
    }

    private readonly struct CacheWriteVisitor<TN, TS, TLane, TL>(CacheView c, int steps, int position, byte[] source, bool lanes, byte[] cache, TLane[]? rowScales) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW>
        {
            int rows = c.Heads * steps;
            bool toOdd = Fmt.ConvertPlan(Fmt.Of<TS>(), Fmt.Of<TW>()).ToOdd;
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * c.D * 4, new CacheWriteBody<TN, TW, TLW, TLane, TL>(c, steps, position, source, lanes, cache, rowScales, toOdd));
        }
    }

    // Source rows [first, last) (row r: head r / steps, step r % steps), each into its cache row (whole 32-bit words of TW:
    // no stored word is two tasks'); a scaled cache's codes by the row's scale as stored.
    private readonly struct CacheWriteBody<TN, TW, TLW, TLane, TL>(CacheView c, int steps, int position, byte[] source, bool lanes, byte[] cache, TLane[]? rowScales, bool toOdd) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int d = c.D;
            var v = ArrayPool<TLane>.Shared.Rent(d);
            var raw = ArrayPool<TLW>.Shared.Rent(d);
            try
            {
                bool integer = !TW.IsFloat;
                TLane min = TL.From(TW.Min), max = TL.From(TW.Max);
                var x = v.AsSpan(0, d);
                var codes = raw.AsSpan(0, d);
                for (int r = first; r < last; r++)
                {
                    long at = (long)position + r % steps;
                    if (at < 0 || at >= c.Capacity) continue;
                    int row = (int)((long)(r / steps) * c.Capacity + at);
                    SourceRow<TN, TLane>(source, lanes, r, x);
                    if (rowScales is not null)
                    {
                        TLane s = rowScales[r];
                        bool positive = Widths.Compare(s, TL.Zero) > 0;
                        for (int i = 0; i < d; i++)
                        {
                            TLane q = positive ? TL.Div(x[i], s) : TL.Zero;
                            if (integer) q = TL.Round(new Vector<TLane>(q))[0];
                            x[i] = Held<TLane, TL>(q, min, max);
                        }
                    }

                    if (toOdd)
                    {
                        // TW a float narrower than float32 from the f64 lane: its lane float32, the value rounded to odd there first.
                        var f = MemoryMarshal.Cast<TLW, float>(codes);
                        var wide = MemoryMarshal.Cast<TLane, double>(x);
                        for (int i = 0; i < d; i++) f[i] = ToOdd(wide[i]);
                    }
                    else if (typeof(TLW) == typeof(float) || typeof(TLW) == typeof(double)) Widths.Load<TLane, TLW>(x, codes);
                    else Widths.Store<TLane, TLW>(x, codes);
                    TW.Store(codes, cache, row * d);
                }
            }
            finally
            {
                ArrayPool<TLane>.Shared.Return(v);
                ArrayPool<TLW>.Shared.Return(raw);
            }
        }
    }

    /// <summary>A layer's projection rows, normalized and rotated in the lane, into the cache at the position
    /// (ICache.CacheWriteRope): the rows a head at a time in the lane's own storage (IRows.Rope's values, not rounded), then
    /// CacheWrite from them. Returns the lane's name.</summary>
    public static string CacheWriteRope<TN, TC>(CacheView c, RopeArgs a, int[] position, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] cache, byte[]? scales)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        c.Validate(Fmt.Of<TC>());
        c.ValidateRope(a);
        return Mixed<TN, CacheRopeCall<TC>>(new CacheRopeCall<TC>(c, a with { HeadsFirst = true }, position, x, gain, cos, sin, positions, cache, scales));
    }

    private readonly struct CacheRopeCall<TC>(CacheView c, RopeArgs a, int[] position, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] cache, byte[]? scales) : IMixed
        where TC : struct, IAdad<TC>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.Rows == 0 || a.Cols == 0) return TL.Name;
            // The rows in the lane's own storage, a pooled buffer (every value written before the write reads it).
            var rows = ArrayPool<byte>.Shared.Rent(a.Rows * a.Cols * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>());
            try
            {
                RopeRows<TN, TLane, TL>(a, x, gain, cos, sin, positions, rows, true);
                return new CacheWriteCall<TC>(c, a.Steps, position[0], rows, cache, scales, true).Run<TN, TS, TLane, TL>();
            }
            finally { ArrayPool<byte>.Shared.Return(rows); }
        }
    }

    /// <summary>O for the queries at the position against the caches (ICache.AttentionCached); returns the lane's name.</summary>
    public static string AttentionCached<TN, TC>(AttentionArgs a, int[] position, byte[] q, byte[] k, byte[]? kScales, byte[] v, byte[]? vScales, byte[] o)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        if (a.StepsFirst)
            return StepsFirstQueries<TN>(a, q, o, (h, qh, oh) => AttentionCached<TN, TC>(h, position, qh, k, kScales, v, vScales, oh));
        CacheRule.Validate(a);
        new CacheView(a.KvHeadCount, a.Lk, a.D).Validate(Fmt.Of<TC>());
        return Mixed<TN, CachedCall<TC>>(new CachedCall<TC>(CacheRule.AtPosition(a, position[0]), q, k, kScales, v, vScales, o));
    }

    // a: the rows at the position (CacheRule.AtPosition): KvHeadCount heads of Group·Lq rows, each row's keys its span and start.
    private readonly struct CachedCall<TC>(AttentionArgs a, byte[] q, byte[] k, byte[]? kScales, byte[] v, byte[]? vScales, byte[] o) : IMixed where TC : struct, IAdad<TC>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.QValues == 0) return TL.Name;
            var cached = new CachedVisitor<TN, TLane, TL>(a, q, k, kScales, v, vScales, o);
            TC.Elementwise(ref cached);
            return TL.Name;
        }
    }

    private readonly struct CachedVisitor<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[]? kScales, byte[] v, byte[]? vScales, byte[] o) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW> =>
            Team.For(a.Heads * a.Lq, RowChunk<TN>(a.Heads * a.Lq, a.D, 1), a.KValues * 2, new CachedBody<TN, TW, TLW, TLane, TL>(a, q, k, kScales, v, vScales, o));
    }

    private readonly struct CachedBody<TN, TW, TLW, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[]? kScales, byte[] v, byte[]? vScales, byte[] o) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            int d = a.D, w = Vector<TLane>.Count;
            var qr = pool.Rent(d);
            var kr = pool.Rent(d);
            var acc = pool.Rent(d);
            var s = pool.Rent(Math.Max(1, a.Lk));
            var rs = pool.Rent(Math.Max(1, a.Lk));
            var raw = ArrayPool<TLW>.Shared.Rent(d);
            try
            {
                TLane scale = TL.From(a.Scale);
                bool capped = a.Softcap > 0, own = typeof(TLW) == typeof(TLane);          // the cache's lane is the float lane: its rows loaded into it
                Vector<TLane> capV = new(TL.From(a.Softcap)), scaleV = new(scale);
                Span<TLane> tail = stackalloc TLane[w];
                var row = kr.AsSpan(0, d);
                var codes = raw.AsSpan(0, d);
                for (int hi = first; hi < last; hi++)
                {
                    int h = hi / a.Lq, i = hi % a.Lq, limit = a.Limit(i), start = Math.Min(a.Start(i), limit), n = limit - start;
                    long qi = (long)hi * d, kBase = (long)h * a.Lk * d;
                    int rowBase = h * a.Lk + start;
                    var scores = s.AsSpan(start, Math.Max(0, n));
                    TN.Load(q, (int)qi, qr.AsSpan(0, d));
                    // The scores: q · the row (a scaled cache's: its codes by its scale, in the lane), scaled or capped.
                    if (kScales is not null && n > 0) TN.Load(kScales, rowBase, rs.AsSpan(0, n));
                    for (int j = start; j < limit; j++)
                    {
                        if (own) TW.Load(k, (int)(kBase + (long)j * d), MemoryMarshal.Cast<TLane, TLW>(row));
                        else
                        {
                            TW.Load(k, (int)(kBase + (long)j * d), codes);
                            Widths.Load<TLW, TLane>(codes, row);
                        }

                        if (kScales is not null) RowMath<TLane, TL>.Scale(row, TL.Zero, rs[j - start], row);
                        s[j] = RowMath<TLane, TL>.Dot(qr.AsSpan(0, d), row);
                    }

                    if (capped)
                    {
                        int j = 0;
                        for (; j + w <= n; j += w) MapOps.Softcap<TLane, TL>.Apply(new Vector<TLane>(scores[j..]), default, default, capV, scaleV).CopyTo(scores[j..]);
                        if (j < n)
                        {
                            tail.Clear();
                            scores[j..].CopyTo(tail);
                            var capTail = MapOps.Softcap<TLane, TL>.Apply(new Vector<TLane>(tail), default, default, capV, scaleV);
                            for (int t = 0; j + t < n; t++) scores[j + t] = capTail[t];
                        }
                    }
                    else
                    {
                        for (int j = 0; j < n; j++) scores[j] = TL.Mul(scores[j], scale);
                    }

                    TLane m = n <= 0 ? TL.NegativeInfinity : RowMath<TLane, TL>.Max(scores);
                    TLane sum = n <= 0 ? TL.Zero : RowMath<TLane, TL>.ExpSum(scores, m, scores);
                    if (vScales is not null && n > 0) TN.Load(vScales, rowBase, rs.AsSpan(0, n));
                    acc.AsSpan(0, d).Clear();
                    for (int j = start; j < limit; j++)
                    {
                        if (own) TW.Load(v, (int)(kBase + (long)j * d), MemoryMarshal.Cast<TLane, TLW>(row));
                        else
                        {
                            TW.Load(v, (int)(kBase + (long)j * d), codes);
                            Widths.Load<TLW, TLane>(codes, row);
                        }

                        if (vScales is not null) RowMath<TLane, TL>.Scale(row, TL.Zero, rs[j - start], row);
                        var pj = new Vector<TLane>(s[j]);
                        int t = 0;
                        for (; t + w <= d; t += w) TL.Fma(new Vector<TLane>(kr.AsSpan(t)), pj, new Vector<TLane>(acc.AsSpan(t))).CopyTo(acc.AsSpan(t));
                        for (; t < d; t++) acc[t] = TL.Fma(kr[t], s[j], acc[t]);
                    }

                    TLane inv = n <= 0 ? TL.Zero : TL.Div(TL.From(1), sum);
                    RowMath<TLane, TL>.Scale(acc.AsSpan(0, d), TL.Zero, inv, acc.AsSpan(0, d));
                    TN.Store(acc.AsSpan(0, d), o, (int)qi);
                }
            }
            finally
            {
                pool.Return(qr); pool.Return(kr); pool.Return(acc); pool.Return(s); pool.Return(rs);
                ArrayPool<TLW>.Shared.Return(raw);
            }
        }
    }

    /// <summary>The mask of a decoder's steps at the position (ICache.DecoderMask); returns the lane's name.</summary>
    public static string DecoderMask<TN>(int rows, int capacity, int window, int[] position, byte[] mask) where TN : struct, IFloatAdad<TN>
    {
        CacheView.MaskValues(rows, capacity, window);
        return Mixed<TN, MaskCall>(new MaskCall(rows, capacity, window, position[0], mask));
    }

    private readonly struct MaskCall(int rows, int capacity, int window, int position, byte[] mask) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            int n = rows * capacity;
            if (n == 0) return TL.Name;
            int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, n, new MaskBody<TN, TLane, TL>(capacity, window, position, mask));
            return TL.Name;
        }
    }

    private readonly struct MaskBody<TN, TLane, TL>(int capacity, int window, int position, byte[] mask) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            Span<TLane> v = stackalloc TLane[Tile];
            TLane lowest = TL.From(TN.Min);
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                for (int e = 0; e < len; e++) v[e] = CacheView.Sees(position, (at + e) / capacity, (at + e) % capacity, window) ? TL.Zero : lowest;
                TN.Store(v[..len], mask, at);
            }
        }
    }

    /// <summary>position[0] += steps.</summary>
    public static void PositionAdvance(int[] position, int steps) => position[0] += steps;
}
