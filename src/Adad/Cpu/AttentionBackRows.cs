// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention backward fused (CPU, item 6j): dQ, dK, dV without the composition's five full products. Two passes over a
// group of heads, no reductions across tasks (deterministic):
//
//   rows  a task per head and block of query rows (16; a block format's more, as the forward's): the rows' scores over
//         the keys they see (Kᵀ's tiles, as the fused rows), P = softmax, dP = dO·Vᵀ (Vᵀ's tiles), D = Σ P·dP per row,
//         dS = P·(dP − D)·scale, dQ = dS·K; P and dS written transposed (Pᵀ, dSᵀ: a key's row over the queries) for
//         the columns the block covers, in whole runs of 16 keys, zero where a row does not see the key.
//   keys  a task per head and run of 16 keys: dV = Pᵀ·dO and dK = dSᵀ·Q over the row blocks whose columns cover the
//         run (the others' Pᵀ / dSᵀ are not written: no key of the run is seen there).
//
// Causal and span masks cost nothing past the keys a row sees (the composition computes every score and product in
// full). Every operand read where it is stored: Q and dO a task's rows (the keys pass: a chunk of query rows at a
// time), K and V a tile at a time, loaded into lanes where the type's bytes are not the lane's; P and dS (Pᵀ, dSᵀ) are
// the lane's intermediates; dQ, dK, dV stored as the type by the task that computes them (a block format's, where a
// task's rows are not whole blocks: the lane's values, encoded once).
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    private const int BackKeys = 16;                                        // a keys task's keys; the transposed runs' grain
    private const int BackChunk = 64;                                       // the keys pass's query rows loaded at a time

    // The bytes of Pᵀ and dSᵀ a group of heads may take (heads are grouped to fit; one head at least).
    private const long BackGroupBytes = 32L << 20;

    /// <summary>dQ, dK, dV by the fused rows and keys passes; returns the lane's name.</summary>
    public static string AttentionBackwardRows<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IFloatAdad<TN>
    {
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionBackwardComposed<TN>(a, q, k, v, dO, dq, dk, dv);
        var vis = new BackRowsVisitor(a, q, k, v, dO, dq, dk, dv);
        TN.Float(ref vis);
        return vis.Lane;
    }

    private struct BackRowsVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            int size = Unsafe.SizeOf<TLane>(), rowsPer = RowsPerTask<TN, TLane>(a);
            int blocks = (a.Lq + rowsPer - 1) / rowsPer, tiles = (a.Lk + FusedKeys - 1) / FusedKeys, runs = (a.Lk + BackKeys - 1) / BackKeys;
            var kt = KeyTiles<TN, TLane>(a, k, Scratch.AttnKt);
            var vt = KeyTiles<TN, TLane>(a, v, Scratch.AttnVt);
            // The gradients a task cannot store as whole storage units of the type: in lanes, encoded once.
            var lq = WholeRows<TN>(rowsPer, a.Lq, a.D) ? null : new byte[a.QValues * size];
            var lk = WholeRows<TN>(BackKeys, a.Lk, a.D) ? null : new byte[a.KValues * size];
            var lv = lk is null ? null : new byte[a.KValues * size];

            // Each row block's columns: the keys its rows see, [lo, hi), widened to whole runs of BackKeys (empty: none).
            var wlo = new int[blocks];
            var whi = new int[blocks];
            for (int b = 0; b < blocks; b++)
            {
                int i0 = b * rowsPer, rows = Math.Min(rowsPer, a.Lq - i0), lo = int.MaxValue, hi = 0;
                for (int r = 0; r < rows; r++) { int s0 = a.Start(i0 + r), l0 = a.Limit(i0 + r); if (l0 > s0) { lo = Math.Min(lo, s0); hi = Math.Max(hi, l0); } }
                if (hi <= lo) continue;
                wlo[b] = lo / BackKeys * BackKeys;
                whi[b] = Math.Min(a.Lk, (hi + BackKeys - 1) / BackKeys * BackKeys);
            }

            long perHead = 2L * a.Lk * a.Lq * size;
            int group = (int)Math.Clamp(BackGroupBytes / Math.Max(1, perHead), 1, a.Heads);
            var pt = Scratch.Get(Scratch.AttnPt, group * (long)a.Lk * a.Lq * size);
            var dst = Scratch.Get(Scratch.AttnDst, group * (long)a.Lk * a.Lq * size);
            for (int h0 = 0; h0 < a.Heads; h0 += group)
            {
                int gh = Math.Min(group, a.Heads - h0);
                Team.For(gh * blocks, 1, (long)gh * a.Lq * a.Lk * a.D, new BackRowsBody<TN, TLane, TL>(a, q, kt, vt, k, v, dO, dq, lq, pt, dst, h0, rowsPer, blocks, tiles, wlo, whi));
                Team.For(gh * runs, 1, (long)gh * a.Lq * a.Lk * a.D, new BackKeysBody<TN, TLane, TL>(a, q, dO, dk, dv, lk, lv, pt, dst, h0, rowsPer, runs, blocks, wlo, whi));
            }

            if (lq is not null) StoreLanes<TN, TLane>(lq, dq, 0, a.QValues);
            if (lk is not null) { StoreLanes<TN, TLane>(lk, dk, 0, a.KValues); StoreLanes<TN, TLane>(lv!, dv, 0, a.KValues); }
        }
    }

    // count lanes of a task into the output at value `at`: in place (lanes), into the lanes kept whole, or stored as the type.
    private static void Put<TN, TLane>(ReadOnlySpan<TLane> values, byte[] output, byte[]? lanes, long at) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        if (lanes is not null) values.CopyTo(MemoryMarshal.Cast<byte, TLane>(lanes.AsSpan()).Slice((int)at, values.Length));
        else if (TN.Direct) values.CopyTo(MemoryMarshal.Cast<byte, TLane>(output.AsSpan()).Slice((int)at, values.Length));
        else TN.Store(values, output, (int)at);
    }

    // A tile of Kᵀ (or Vᵀ) for keys [t0, t0 + bc): the laid tile (in place, or loaded), or laid now from x's rows (a block format).
    private static ReadOnlySpan<TLane> TileOf<TN, TLane>(AttentionArgs a, byte[]? tiles, byte[] x, int h, int t0, int count, TLane[]? buffer, TLane[]? rows)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        int d = a.D, bc = FusedKeys;
        if (tiles is not null) return Operand<TN, TLane>.Of(tiles, (long)(h * count + t0 / bc) * d * bc, d * bc, buffer);
        int keys = Math.Min(bc, a.Lk - t0);
        TN.Load(x, (int)((long)h * a.Lk * d + (long)t0 * d), rows.AsSpan(0, keys * d));
        Lay(rows!, keys, d, buffer!);
        return buffer.AsSpan(0, d * bc);
    }

    // The rows pass: dQ of a block of query rows, and its Pᵀ, dSᵀ columns.
    private readonly struct BackRowsBody<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[]? kts, byte[]? vts, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[]? lq,
        byte[] pts, byte[] dsts, int h0, int rowsPer, int blocks, int tiles, int[] wlo, int[] whi) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int d = a.D, w = Vector<TLane>.Count, bc = FusedKeys, group = 4 * w, width = tiles * bc;
            bool direct = TN.Direct;
            var PT = MemoryMarshal.Cast<byte, TLane>(pts.AsSpan());
            var DST = MemoryMarshal.Cast<byte, TLane>(dsts.AsSpan());
            var pool = ArrayPool<TLane>.Shared;
            var s = pool.Rent(rowsPer * width);
            var g = pool.Rent(rowsPer * width);
            var acc = pool.Rent(rowsPer * d);
            TLane[]? qb = direct ? null : pool.Rent(rowsPer * d);
            TLane[]? db = direct ? null : pool.Rent(rowsPer * d);
            TLane[]? ktb = direct && kts is not null ? null : pool.Rent(d * bc);
            TLane[]? vtb = direct && vts is not null ? null : pool.Rent(d * bc);
            TLane[]? kb = direct ? null : pool.Rent(bc * d);
            var from = new int[rowsPer];
            var to = new int[rowsPer];
            TLane scale = TL.From(a.Scale), one = TL.From(1);
            try
            {
                for (int task = first; task < last; task++)
                {
                    int hg = task / blocks, b = task % blocks, h = h0 + hg, i0 = b * rowsPer, rows = Math.Min(rowsPer, a.Lq - i0);
                    long kHead = (long)h * a.Lk * d, qHead = (long)h * a.Lq * d;
                    int lo = int.MaxValue, hi = 0;
                    for (int r = 0; r < rows; r++) { int s0 = a.Start(i0 + r), l0 = a.Limit(i0 + r); if (l0 > s0) { lo = Math.Min(lo, s0); hi = Math.Max(hi, l0); } }
                    if (hi <= lo)
                    {
                        // No row of the block sees a key: dQ is 0, no column written.
                        acc.AsSpan(0, rows * d).Clear();
                        Put<TN, TLane>(acc.AsSpan(0, rows * d), dq, lq, qHead + (long)i0 * d);
                        continue;
                    }

                    var Q = Operand<TN, TLane>.Of(q, qHead + (long)i0 * d, rows * d, qb);
                    var DO = Operand<TN, TLane>.Of(dO, qHead + (long)i0 * d, rows * d, db);
                    int bas = lo / bc * bc, span = (hi - bas + bc - 1) / bc * bc;
                    for (int t0 = bas; t0 < hi; t0 += bc)
                    {
                        int n = Math.Min(bc, hi - t0), ng = (n + group - 1) / group * group, at = t0 - bas;
                        var kt = TileOf<TN, TLane>(a, kts, k, h, t0, tiles, ktb, kb);
                        var vt = TileOf<TN, TLane>(a, vts, v, h, t0, tiles, vtb, kb);
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            FusedBody<TN, TLane, TL>.ScoresOf(Q.Slice(r * d, d), Q.Slice(r1 * d, d), kt, bc, ng, d, s.AsSpan(r * width + at, bc), s.AsSpan(r1 * width + at, bc));
                            FusedBody<TN, TLane, TL>.ScoresOf(DO.Slice(r * d, d), DO.Slice(r1 * d, d), vt, bc, ng, d, g.AsSpan(r * width + at, bc), g.AsSpan(r1 * width + at, bc));
                        }
                    }

                    // Per row: P over its keys (0 elsewhere), then dS = P·(dP − Σ P·dP)·scale in g.
                    int cLo = int.MaxValue, cHi = 0;
                    for (int r = 0; r < rows; r++)
                    {
                        int i = i0 + r;
                        from[r] = Math.Max(a.Start(i), bas) - bas; to[r] = Math.Min(a.Limit(i), hi) - bas;
                        var row = s.AsSpan(r * width, span);
                        var grow = g.AsSpan(r * width, span);
                        if (to[r] <= from[r]) { row.Clear(); grow.Clear(); continue; }
                        var p = row[from[r]..to[r]];
                        for (int c = 0; c < p.Length; c++) p[c] = TL.Mul(p[c], scale);
                        TLane m = RowMath<TLane, TL>.Max(p);
                        TLane sum = RowMath<TLane, TL>.ExpSum(p, m, p);
                        RowMath<TLane, TL>.Scale(p, TL.Zero, TL.Div(one, sum), p);
                        row[..from[r]].Clear(); row[to[r]..].Clear();
                        var dp = grow[from[r]..to[r]];
                        TLane dot = RowMath<TLane, TL>.Dot(p, dp);
                        RowMath<TLane, TL>.Scale(dp, dot, scale, dp);
                        Multiply(p, dp);
                        grow[..from[r]].Clear(); grow[to[r]..].Clear();
                        cLo = Math.Min(cLo, from[r]); cHi = Math.Max(cHi, to[r]);
                    }

                    // dQ = dS·K over the block's keys (weights outside a row's keys are 0), two rows at a time: K in place
                    // over the whole span, or a tile of its rows loaded at a time.
                    acc.AsSpan(0, rows * d).Clear();
                    if (cHi > cLo)
                    {
                        if (direct)
                        {
                            var K = MemoryMarshal.Cast<byte, TLane>(k.AsSpan());
                            for (int r = 0; r < rows; r += 2)
                            {
                                int r1 = Math.Min(r + 1, rows - 1);
                                FusedBody<TN, TLane, TL>.ContextOf(g.AsSpan(r * width, span), g.AsSpan(r1 * width, span), one, one, K, kHead + (long)bas * d, cLo, cHi, d,
                                    acc.AsSpan(r * d, d), acc.AsSpan(r1 * d, d), r1 == r);
                            }
                        }
                        else
                        {
                            for (int t0 = bas; t0 < hi; t0 += bc)
                            {
                                int at = t0 - bas, cl = Math.Max(cLo, at) - at, ch = Math.Min(cHi, at + bc) - at;
                                if (ch <= cl) continue;
                                TN.Load(k, (int)(kHead + (long)(t0 + cl) * d), kb.AsSpan(cl * d, (ch - cl) * d));
                                for (int r = 0; r < rows; r += 2)
                                {
                                    int r1 = Math.Min(r + 1, rows - 1);
                                    FusedBody<TN, TLane, TL>.ContextOf(g.AsSpan(r * width + at, bc), g.AsSpan(r1 * width + at, bc), one, one, kb, 0, cl, ch, d,
                                        acc.AsSpan(r * d, d), acc.AsSpan(r1 * d, d), r1 == r);
                                }
                            }
                        }
                    }

                    Put<TN, TLane>(acc.AsSpan(0, rows * d), dq, lq, qHead + (long)i0 * d);

                    // Pᵀ and dSᵀ of the block's columns: key c's row over the queries, the block's rows at i0.
                    long tHead = (long)hg * a.Lk * a.Lq;
                    for (int c = wlo[b]; c < whi[b]; c++)
                    {
                        int at = c - bas;
                        var prow = PT.Slice((int)(tHead + (long)c * a.Lq + i0), rows);
                        var drow = DST.Slice((int)(tHead + (long)c * a.Lq + i0), rows);
                        for (int r = 0; r < rows; r++) { prow[r] = s[r * width + at]; drow[r] = g[r * width + at]; }
                    }
                }
            }
            finally
            {
                pool.Return(s); pool.Return(g); pool.Return(acc);
                if (qb is not null) pool.Return(qb);
                if (db is not null) pool.Return(db);
                if (ktb is not null) pool.Return(ktb);
                if (vtb is not null) pool.Return(vtb);
                if (kb is not null) pool.Return(kb);
            }
        }

        // y = x·y elementwise.
        private static void Multiply(ReadOnlySpan<TLane> x, Span<TLane> y)
        {
            int w = Vector<TLane>.Count, i = 0;
            ref TLane rx = ref MemoryMarshal.GetReference(x);
            ref TLane ry = ref MemoryMarshal.GetReference(y);
            for (; i + w <= x.Length; i += w) (Vector.LoadUnsafe(ref rx, (nuint)i) * Vector.LoadUnsafe(ref ry, (nuint)i)).StoreUnsafe(ref ry, (nuint)i);
            for (; i < x.Length; i++) y[i] = TL.Mul(x[i], y[i]);
        }
    }

    // The keys pass: dV = Pᵀ·dO and dK = dSᵀ·Q of a run of keys, over the row blocks whose columns cover it.
    private readonly struct BackKeysBody<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] dO, byte[] dk, byte[] dv, byte[]? lk, byte[]? lv, byte[] pts, byte[] dsts,
        int h0, int rowsPer, int runs, int blocks, int[] wlo, int[] whi) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int d = a.D;
            bool direct = TN.Direct;
            var PT = MemoryMarshal.Cast<byte, TLane>(pts.AsSpan());
            var DST = MemoryMarshal.Cast<byte, TLane>(dsts.AsSpan());
            var pool = ArrayPool<TLane>.Shared;
            var gvb = pool.Rent(BackKeys * d);
            var gkb = pool.Rent(BackKeys * d);
            TLane[]? qb = direct ? null : pool.Rent(BackChunk * d);
            TLane[]? db = direct ? null : pool.Rent(BackChunk * d);
            TLane one = TL.From(1);
            try
            {
                for (int task = first; task < last; task++)
                {
                    int hg = task / runs, j0 = task % runs * BackKeys, h = h0 + hg, keys = Math.Min(BackKeys, a.Lk - j0);
                    long kHead = (long)h * a.Lk * d, qHead = (long)h * a.Lq * d, tHead = (long)hg * a.Lk * a.Lq;
                    var gv = gvb.AsSpan(0, keys * d);
                    var gk = gkb.AsSpan(0, keys * d);
                    gv.Clear(); gk.Clear();
                    // Runs of consecutive covering blocks: their query rows in place, or a chunk of them loaded at a time.
                    for (int b = 0; b < blocks;)
                    {
                        if (!(wlo[b] <= j0 && j0 < whi[b])) { b++; continue; }
                        int e = b + 1;
                        while (e < blocks && wlo[e] <= j0 && j0 < whi[e]) e++;
                        int iLo = b * rowsPer, iHi = Math.Min(a.Lq, e * rowsPer);
                        for (int c0 = iLo; c0 < iHi; c0 += direct ? iHi - iLo : BackChunk)
                        {
                            int c1 = direct ? iHi : Math.Min(iHi, c0 + BackChunk);
                            ReadOnlySpan<TLane> DO, Q;
                            long vBase;
                            if (direct) { DO = MemoryMarshal.Cast<byte, TLane>(dO.AsSpan()); Q = MemoryMarshal.Cast<byte, TLane>(q.AsSpan()); vBase = qHead; }
                            else
                            {
                                DO = Operand<TN, TLane>.Of(dO, qHead + (long)c0 * d, (c1 - c0) * d, db);
                                Q = Operand<TN, TLane>.Of(q, qHead + (long)c0 * d, (c1 - c0) * d, qb);
                                vBase = -(long)c0 * d;
                            }

                            for (int c = 0; c < keys; c += 2)
                            {
                                int k1 = Math.Min(c + 1, keys - 1);
                                var p0 = PT.Slice((int)(tHead + (long)(j0 + c) * a.Lq), a.Lq);
                                var p1 = PT.Slice((int)(tHead + (long)(j0 + k1) * a.Lq), a.Lq);
                                FusedBody<TN, TLane, TL>.ContextOf(p0, p1, one, one, DO, vBase, c0, c1, d, gv.Slice(c * d, d), gv.Slice(k1 * d, d), k1 == c);
                                var s0 = DST.Slice((int)(tHead + (long)(j0 + c) * a.Lq), a.Lq);
                                var s1 = DST.Slice((int)(tHead + (long)(j0 + k1) * a.Lq), a.Lq);
                                FusedBody<TN, TLane, TL>.ContextOf(s0, s1, one, one, Q, vBase, c0, c1, d, gk.Slice(c * d, d), gk.Slice(k1 * d, d), k1 == c);
                            }
                        }

                        b = e;
                    }

                    Put<TN, TLane>(gv, dv, lv, kHead + (long)j0 * d);
                    Put<TN, TLane>(gk, dk, lk, kHead + (long)j0 * d);
                }
            }
            finally
            {
                pool.Return(gvb); pool.Return(gkb);
                if (qb is not null) pool.Return(qb);
                if (db is not null) pool.Return(db);
            }
        }
    }
}
