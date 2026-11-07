// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention backward fused (CPU, item 6j): dQ, dK, dV without the composition's five full products. Two passes over a
// group of heads, no reductions across tasks (deterministic):
//
//   rows  a task per head and block of 16 query rows: the rows' scores over the keys they see (Kᵀ's tiles, as the fused
//         rows), P = softmax, dP = dO·Vᵀ (Vᵀ's tiles), D = Σ P·dP per row, dS = P·(dP − D)·scale, dQ = dS·K; P and dS
//         written transposed (Pᵀ, dSᵀ: a key's row over the queries) for the columns the block covers, in whole runs of
//         16 keys, zero where a row does not see the key.
//   keys  a task per head and run of 16 keys: dV = Pᵀ·dO and dK = dSᵀ·Q over the row blocks whose columns cover the
//         run (the others' Pᵀ / dSᵀ are not written: no key of the run is seen there).
//
// Causal and span masks cost nothing past the keys a row sees (the composition computes every score and product in
// full). A narrow type is widened to its float lane once and the gradients narrowed once, as the composition does.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Qiyas;

internal sealed partial class Kernels
{
    private const int BackKeys = 16;                                        // a keys task's keys; the transposed runs' grain

    // The bytes of Pᵀ and dSᵀ a group of heads may take (heads are grouped to fit; one head at least).
    private const long BackGroupBytes = 32L << 20;

    /// <summary>dQ, dK, dV by the fused rows and keys passes; returns the lane's name.</summary>
    public static string AttentionBackwardRows<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IQiyasFloat<TN>
    {
        bool own = TN.Name is "float32" or "float64";
        var lq = ToLane<TN>(q, a.QValues);
        var lk = ToLane<TN>(k, a.KValues);
        var lv = ToLane<TN>(v, a.KValues);
        var ldo = ToLane<TN>(dO, a.QValues);
        int size = Wide<TN>() ? 8 : 4;
        var gq = own ? dq : new byte[a.QValues * size];
        var gk = own ? dk : new byte[a.KValues * size];
        var gv = own ? dv : new byte[a.KValues * size];
        var vis = new BackRowsVisitor(a, lq, lk, lv, ldo, gq, gk, gv);
        if (Wide<TN>()) Float64.Float(ref vis); else Float32.Float(ref vis);
        if (!own) { FromLane<TN>(gq, dq, a.QValues); FromLane<TN>(gk, dk, a.KValues); FromLane<TN>(gv, dv, a.KValues); }
        return size == 8 ? "f64" : "f32";
    }

    private readonly struct BackRowsVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) : IFloatLaneVisitor
    {
        public void Run<TS, TLane, TL>() where TS : struct, IQiyasNumber<TS, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            int size = Unsafe.SizeOf<TLane>();
            int blocks = (a.Lq + FusedRows - 1) / FusedRows, tiles = (a.Lk + FusedKeys - 1) / FusedKeys, runs = (a.Lk + BackKeys - 1) / BackKeys;
            long tileBytes = (long)a.Heads * tiles * a.D * FusedKeys * size;
            var kt = Scratch.Get(Scratch.AttnKt, tileBytes);
            var vt = Scratch.Get(Scratch.AttnVt, tileBytes);
            Team.For(a.Heads * tiles, 1, (long)a.Heads * a.Lk * a.D, new KeyTilesBody<TLane>(a, k, kt, tiles));
            Team.For(a.Heads * tiles, 1, (long)a.Heads * a.Lk * a.D, new KeyTilesBody<TLane>(a, v, vt, tiles));

            // Each row block's columns: the keys its rows see, [lo, hi), widened to whole runs of BackKeys (empty: none).
            var wlo = new int[blocks];
            var whi = new int[blocks];
            for (int b = 0; b < blocks; b++)
            {
                int i0 = b * FusedRows, rows = Math.Min(FusedRows, a.Lq - i0), lo = int.MaxValue, hi = 0;
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
                Team.For(gh * blocks, 1, (long)gh * a.Lq * a.Lk * a.D, new BackRowsBody<TLane, TL>(a, q, kt, vt, k, dO, dq, pt, dst, h0, blocks, tiles, wlo, whi));
                Team.For(gh * runs, 1, (long)gh * a.Lq * a.Lk * a.D, new BackKeysBody<TLane, TL>(a, q, dO, dk, dv, pt, dst, h0, runs, blocks, wlo, whi));
            }
        }
    }

    // The rows pass: dQ of a block of query rows, and its Pᵀ, dSᵀ columns.
    private readonly struct BackRowsBody<TLane, TL>(AttentionArgs a, byte[] q, byte[] kts, byte[] vts, byte[] k, byte[] dO, byte[] dq, byte[] pts, byte[] dsts,
        int h0, int blocks, int tiles, int[] wlo, int[] whi) : Team.IBody
        where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int d = a.D, w = Vector<TLane>.Count, bc = FusedKeys, group = 4 * w, width = tiles * bc;
            var Q = MemoryMarshal.Cast<byte, TLane>(q.AsSpan());
            var KT = MemoryMarshal.Cast<byte, TLane>(kts.AsSpan());
            var VT = MemoryMarshal.Cast<byte, TLane>(vts.AsSpan());
            var K = MemoryMarshal.Cast<byte, TLane>(k.AsSpan());
            var DO = MemoryMarshal.Cast<byte, TLane>(dO.AsSpan());
            var DQ = MemoryMarshal.Cast<byte, TLane>(dq.AsSpan());
            var PT = MemoryMarshal.Cast<byte, TLane>(pts.AsSpan());
            var DST = MemoryMarshal.Cast<byte, TLane>(dsts.AsSpan());
            var pool = ArrayPool<TLane>.Shared;
            var s = pool.Rent(FusedRows * width);
            var g = pool.Rent(FusedRows * width);
            var acc = pool.Rent(FusedRows * d);
            var from = new int[FusedRows];
            var to = new int[FusedRows];
            TLane scale = TL.From(a.Scale), one = TL.From(1);
            try
            {
                for (int task = first; task < last; task++)
                {
                    int hg = task / blocks, b = task % blocks, h = h0 + hg, i0 = b * FusedRows, rows = Math.Min(FusedRows, a.Lq - i0);
                    long kHead = (long)h * a.Lk * d, qHead = (long)h * a.Lq * d;
                    int lo = int.MaxValue, hi = 0;
                    for (int r = 0; r < rows; r++) { int s0 = a.Start(i0 + r), l0 = a.Limit(i0 + r); if (l0 > s0) { lo = Math.Min(lo, s0); hi = Math.Max(hi, l0); } }
                    if (hi <= lo)
                    {
                        // No row of the block sees a key: dQ is 0, no column written.
                        DQ.Slice((int)(qHead + (long)i0 * d), rows * d).Clear();
                        continue;
                    }

                    int bas = lo / bc * bc, span = (hi - bas + bc - 1) / bc * bc;
                    for (int t0 = bas; t0 < hi; t0 += bc)
                    {
                        int n = Math.Min(bc, hi - t0), ng = (n + group - 1) / group * group, at = t0 - bas;
                        var kt = KT.Slice((h * tiles + t0 / bc) * d * bc, d * bc);
                        var vt = VT.Slice((h * tiles + t0 / bc) * d * bc, d * bc);
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            FusedBody<TLane, TL>.ScoresOf(Q.Slice((int)(qHead + (long)(i0 + r) * d), d), Q.Slice((int)(qHead + (long)(i0 + r1) * d), d), kt, bc, ng, d,
                                s.AsSpan(r * width + at, bc), s.AsSpan(r1 * width + at, bc));
                            FusedBody<TLane, TL>.ScoresOf(DO.Slice((int)(qHead + (long)(i0 + r) * d), d), DO.Slice((int)(qHead + (long)(i0 + r1) * d), d), vt, bc, ng, d,
                                g.AsSpan(r * width + at, bc), g.AsSpan(r1 * width + at, bc));
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

                    // dQ = dS·K over the block's keys (weights outside a row's keys are 0), two rows at a time.
                    acc.AsSpan(0, rows * d).Clear();
                    if (cHi > cLo)
                    {
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            FusedBody<TLane, TL>.ContextOf(g.AsSpan(r * width, span), g.AsSpan(r1 * width, span), one, one, K, kHead + (long)bas * d, cLo, cHi, d,
                                acc.AsSpan(r * d, d), acc.AsSpan(r1 * d, d), r1 == r);
                        }
                    }

                    acc.AsSpan(0, rows * d).CopyTo(DQ.Slice((int)(qHead + (long)i0 * d), rows * d));

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
    private readonly struct BackKeysBody<TLane, TL>(AttentionArgs a, byte[] q, byte[] dO, byte[] dk, byte[] dv, byte[] pts, byte[] dsts,
        int h0, int runs, int blocks, int[] wlo, int[] whi) : Team.IBody
        where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int d = a.D;
            var Q = MemoryMarshal.Cast<byte, TLane>(q.AsSpan());
            var DO = MemoryMarshal.Cast<byte, TLane>(dO.AsSpan());
            var DK = MemoryMarshal.Cast<byte, TLane>(dk.AsSpan());
            var DV = MemoryMarshal.Cast<byte, TLane>(dv.AsSpan());
            var PT = MemoryMarshal.Cast<byte, TLane>(pts.AsSpan());
            var DST = MemoryMarshal.Cast<byte, TLane>(dsts.AsSpan());
            TLane one = TL.From(1);
            for (int task = first; task < last; task++)
            {
                int hg = task / runs, j0 = task % runs * BackKeys, h = h0 + hg, keys = Math.Min(BackKeys, a.Lk - j0);
                long kHead = (long)h * a.Lk * d, qHead = (long)h * a.Lq * d, tHead = (long)hg * a.Lk * a.Lq;
                var gv = DV.Slice((int)(kHead + (long)j0 * d), keys * d);
                var gk = DK.Slice((int)(kHead + (long)j0 * d), keys * d);
                gv.Clear(); gk.Clear();
                // Runs of consecutive covering blocks: one context call each.
                for (int b = 0; b < blocks;)
                {
                    if (!(wlo[b] <= j0 && j0 < whi[b])) { b++; continue; }
                    int e = b + 1;
                    while (e < blocks && wlo[e] <= j0 && j0 < whi[e]) e++;
                    int iLo = b * FusedRows, iHi = Math.Min(a.Lq, e * FusedRows);
                    for (int c = 0; c < keys; c += 2)
                    {
                        int c1 = Math.Min(c + 1, keys - 1);
                        var p0 = PT.Slice((int)(tHead + (long)(j0 + c) * a.Lq), a.Lq);
                        var p1 = PT.Slice((int)(tHead + (long)(j0 + c1) * a.Lq), a.Lq);
                        FusedBody<TLane, TL>.ContextOf(p0, p1, one, one, DO, qHead, iLo, iHi, d, gv.Slice(c * d, d), gv.Slice(c1 * d, d), c1 == c);
                        var s0 = DST.Slice((int)(tHead + (long)(j0 + c) * a.Lq), a.Lq);
                        var s1 = DST.Slice((int)(tHead + (long)(j0 + c1) * a.Lq), a.Lq);
                        FusedBody<TLane, TL>.ContextOf(s0, s1, one, one, Q, qHead, iLo, iHi, d, gk.Slice(c * d, d), gk.Slice(c1 * d, d), c1 == c);
                    }

                    b = e;
                }
            }
        }
    }
}
