// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on the CPU (Abstraction/Cache.cs, ICache), the position read from its index buffer where it is.
//
//   cache_write       a task a source row: the row into TN's float lane, taken into the cache type's element-wise lane with
//                     one rounding (Convert's rule: an f64 lane's value to a narrow float rounded to odd at float32 first)
//                     and stored at the position's row (whole 32-bit words of the cache's type: no word is two tasks'); a
//                     block format's cache a task a block the rows touch, each block encoded once.
//   attention_cached  the decode's query at a time (Attention.cs) over the rows the position fills: each row of K and V read
//                     where it is stored as TC into its element-wise lane and taken into the float lane; the scores capped
//                     by the map's softcap where there is a cap.
//   decoder_mask      aligned tiles of the mask (a stored unit of TN written by one task).

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>The source's rows into the cache at the position (ICache.CacheWrite); returns the lane's name.</summary>
    public static string CacheWrite<TN, TC>(CacheView c, int steps, int[] position, byte[] source, byte[] cache)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        c.Validate(Fmt.Of<TC>());
        if (steps < 0 || (long)c.Heads * steps * c.D > int.MaxValue) throw new ArgumentException($"cache: a write of {steps} rows a head");
        return Mixed<TN, CacheWriteCall<TC>>(new CacheWriteCall<TC>(c, steps, position[0], source, cache));
    }

    private readonly struct CacheWriteCall<TC>(CacheView c, int steps, int position, byte[] source, byte[] cache) : IMixed where TC : struct, IAdad<TC>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (c.Heads == 0 || steps == 0 || c.D == 0) return TL.Name;
            var write = new CacheWriteVisitor<TN, TS, TLane, TL>(c, steps, position, source, cache);
            TC.Elementwise(ref write);
            return TL.Name;
        }
    }

    private readonly struct CacheWriteVisitor<TN, TS, TLane, TL>(CacheView c, int steps, int position, byte[] source, byte[] cache) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW>
        {
            int rows = c.Heads * steps;
            if (Fmt.Of<TW>().IsBlock)
            {
                // A block format: a task a block the rows touch (BlockRows), each block encoded once.
                var (w0, w1) = c.Written(position, steps);
                if (w1 <= w0) return;
                int g = TW.Granularity, run = (w1 - w0) * c.D, done = -1, total = 0;
                int[] first = new int[c.Heads + 1], block = new int[c.Heads];
                for (int h = 0; h < c.Heads; h++)
                {
                    // Head h's blocks: from its run's first, past the last one a head before took, to its run's last.
                    int at = (h * c.Capacity + w0) * c.D, last = (at + run - 1) / g;
                    block[h] = Math.Max(at / g, done + 1);
                    first[h] = total;
                    total += Math.Max(0, last - block[h] + 1);
                    done = Math.Max(done, last);
                }

                first[c.Heads] = total;
                Team.For(total, Math.Max(1, total / (Threads * 4)), (long)total * g * 4, new CacheBlocksBody<TN, TW, TLW, TLane, TL>(c, steps, position, w0, w1, source, cache, first, block));
                return;
            }

            bool toOdd = Fmt.ConvertPlan(Fmt.Of<TS>(), Fmt.Of<TW>()).ToOdd;
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * c.D * 4, new CacheWriteBody<TN, TW, TLW, TLane, TL>(c, steps, position, source, cache, toOdd));
        }
    }

    // Source rows [first, last) (row r: head r / steps, step r % steps), each into its cache row (whole 32-bit words of TW:
    // no stored word is two tasks').
    private readonly struct CacheWriteBody<TN, TW, TLW, TLane, TL>(CacheView c, int steps, int position, byte[] source, byte[] cache, bool toOdd) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int d = c.D;
            var v = ArrayPool<TLane>.Shared.Rent(d);
            var raw = ArrayPool<TLW>.Shared.Rent(d);
            try
            {
                var x = v.AsSpan(0, d);
                var codes = raw.AsSpan(0, d);
                for (int r = first; r < last; r++)
                {
                    long at = (long)position + r % steps;
                    if (at < 0 || at >= c.Capacity) continue;
                    int row = (int)((long)(r / steps) * c.Capacity + at);
                    TN.Load(source, r * d, x);
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

    // A cache of a block format: blocks [first, last) of the flat list of the blocks the write touches (head h's from
    // block[h], first[h] before them), each one task's. A block's values: the written rows' from the source (the lane's
    // values rounded to float32, as the format encodes from), the others as they are stored (decoded: the type's partial
    // store), the block encoded once. A row of whole blocks keeps nothing: its blocks are written whole.
    private readonly struct CacheBlocksBody<TN, TW, TLW, TLane, TL>(CacheView c, int steps, int position, int w0, int w1, byte[] source, byte[] cache, int[] first, int[] block) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int from, int to)
        {
            int g = TW.Granularity, d = c.D, cap = c.Capacity, total = (int)c.Values;
            Span<TLW> values = stackalloc TLW[g];
            Span<TLane> row = stackalloc TLane[g];
            int h = 0;
            for (int i = from; i < to; i++)
            {
                while (i >= first[h + 1]) h++;
                int a = (block[h] + i - first[h]) * g, e = Math.Min(a + g, total);
                var block0 = values[..(e - a)];
                bool kept = false;
                for (int x = a; x < e; x += Math.Min(e - x, d - x % d))
                {
                    int r = x / d % cap;
                    kept |= r < w0 || r >= w1;
                }

                if (kept) TW.Load(cache, a, block0);
                for (int x = a; x < e;)
                {
                    int rowAt = x / d, col = x % d, n = Math.Min(e - x, d - col), r = rowAt % cap;
                    if (r >= w0 && r < w1)
                    {
                        int at = (rowAt / cap * steps + (r - position)) * d + col;
                        var run = row[..n];
                        TN.Load(source, at, run);
                        Widths.Load<TLane, TLW>(run, block0.Slice(x - a, n));
                    }

                    x += n;
                }

                TW.Store(block0, cache, a);
            }
        }
    }

    /// <summary>O for the queries at the position against the caches (ICache.AttentionCached); returns the lane's name.</summary>
    public static string AttentionCached<TN, TC>(AttentionArgs a, int[] position, byte[] q, byte[] k, byte[] v, byte[] o)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>
    {
        if (a.Laid)
            return LaidQueries<TN>(a, q, o, (h, qh, oh) => AttentionCached<TN, TC>(h, position, qh, k, v, oh));
        CacheRule.Validate(a);
        new CacheView(a.KvHeadCount, a.Lk, a.D).Validate(Fmt.Of<TC>());
        return Mixed<TN, CachedCall<TC>>(new CachedCall<TC>(CacheRule.AtPosition(a, position[0]), q, k, v, o));
    }

    // a: the rows at the position (CacheRule.AtPosition): KvHeadCount heads of Group·Lq rows, each row's keys its span and start.
    private readonly struct CachedCall<TC>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : IMixed where TC : struct, IAdad<TC>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.QValues == 0) return TL.Name;
            var cached = new CachedVisitor<TN, TLane, TL>(a, q, k, v, o);
            TC.Elementwise(ref cached);
            return TL.Name;
        }
    }

    private readonly struct CachedVisitor<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW> =>
            Team.For(a.Heads * a.Lq, RowChunk<TN>(a.Heads * a.Lq, a.D, 1), a.KValues * 2, new CachedBody<TN, TW, TLW, TLane, TL>(a, q, k, v, o));
    }

    private readonly struct CachedBody<TN, TW, TLW, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : Team.IBody
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
            var raw = ArrayPool<TLW>.Shared.Rent(d);
            var store = new UnitStore<TN, TLane>(o, d);
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
                    var scores = s.AsSpan(start, Math.Max(0, n));
                    TN.Load(q, (int)qi, qr.AsSpan(0, d));
                    // The scores: q · the row, scaled or capped.
                    for (int j = start; j < limit; j++)
                    {
                        if (own) TW.Load(k, (int)(kBase + (long)j * d), MemoryMarshal.Cast<TLane, TLW>(row));
                        else
                        {
                            TW.Load(k, (int)(kBase + (long)j * d), codes);
                            Widths.Load<TLW, TLane>(codes, row);
                        }

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
                    acc.AsSpan(0, d).Clear();
                    for (int j = start; j < limit; j++)
                    {
                        if (own) TW.Load(v, (int)(kBase + (long)j * d), MemoryMarshal.Cast<TLane, TLW>(row));
                        else
                        {
                            TW.Load(v, (int)(kBase + (long)j * d), codes);
                            Widths.Load<TLW, TLane>(codes, row);
                        }

                        var pj = new Vector<TLane>(s[j]);
                        int t = 0;
                        for (; t + w <= d; t += w) TL.Fma(new Vector<TLane>(kr.AsSpan(t)), pj, new Vector<TLane>(acc.AsSpan(t))).CopyTo(acc.AsSpan(t));
                        for (; t < d; t++) acc[t] = TL.Fma(kr[t], s[j], acc[t]);
                    }

                    TLane inv = n <= 0 ? TL.Zero : TL.Div(TL.From(1), sum);
                    RowMath<TLane, TL>.Scale(acc.AsSpan(0, d), TL.Zero, inv, acc.AsSpan(0, d));
                    store.Put(acc.AsSpan(0, d), qi);
                }

                store.Done();
            }
            finally
            {
                pool.Return(qr); pool.Return(kr); pool.Return(acc); pool.Return(s);
                ArrayPool<TLW>.Shared.Return(raw);
            }
        }
    }

    /// <summary>The mask of a decoder's steps at the position (ICache.DecoderMask); returns the lane's name.</summary>
    public static string DecoderMask<TN>(int rows, int capacity, int window, int[] position, byte[] mask) where TN : struct, IFloatAdad<TN>
    {
        CacheView.MaskValues(rows, capacity, window);
        return Mixed<TN, MaskCall>(new MaskCall(rows, capacity, window, position[0], mask, CacheView.Excluded<TN>()));
    }

    private readonly struct MaskCall(int rows, int capacity, int window, int position, byte[] mask, double excluded) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            int n = rows * capacity;
            if (n == 0) return TL.Name;
            int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, n, new MaskBody<TN, TLane, TL>(capacity, window, position, mask, excluded));
            return TL.Name;
        }
    }

    private readonly struct MaskBody<TN, TLane, TL>(int capacity, int window, int position, byte[] mask, double excluded) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            Span<TLane> v = stackalloc TLane[Tile];
            TLane none = TL.From(excluded);
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                for (int e = 0; e < len; e++) v[e] = CacheView.Sees(position, (at + e) / capacity, (at + e) % capacity, window) ? TL.Zero : none;
                TN.Store(v[..len], mask, at);
            }
        }
    }

    /// <summary>position[0] += steps.</summary>
    public static void PositionAdvance(int[] position, int steps) => position[0] += steps;
}
