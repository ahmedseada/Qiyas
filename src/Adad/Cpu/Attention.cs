// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention: O = softmax(scale · Q·Kᵀ + mask) · V per head, in the type's float lane. Three forms:
//
//   attention         rows of queries (prefill): the composition of Adad's own operations — the product Q·Kᵀ reading Q
//                     and K where they are stored, scale_mask_softmax (causal with the key offset, or a key span per
//                     row), the product P·V reading V where it is stored and writing O as the type. The scores and P are
//                     intermediates in the float lane's own type (float32 for the narrow floats, float64 for float64);
//                     no input is widened whole (a block format's O: the lane's values, encoded once).
//   attention_decode  one or a few queries against a long key cache: a fused kernel on the device (a block per query,
//                     keys in chunks, the softmax online, K and V read once); on the CPU a head's rows over a part of
//                     its keys at a time, K and V rows read once for them all, the parts of a row combined after.
//   attention_bwd     dQ, dK, dV from dO: the composition (P recomputed; dV = Pᵀ·dO, dP = dO·Vᵀ, dS = softmax_bwd(P, dP),
//                     dQ = scale·dS·K, dK = scale·dSᵀ·Q).
//
// Causal: query i (of Lq) sees keys j ≤ i + (Lk − Lq): the queries are the last Lq positions of the keys. A span per
// query (spans[i] = the keys it sees, from 0) replaces the causal rule when given; a window keeps its last keys.
//
// Grouped heads (KvHeads below Heads): a group's query heads are adjacent in Q and O, so every form runs the attention of
// KvHeads heads of Group·Lq rows (AttentionArgs.Grouped), each reading its K and V's head; the backward's sums over a
// group's rows into dK and dV are the products' own sums (one rounding). A cap on the scores (Softcap) takes the
// composition: S = cap·tanh(scale·Q·Kᵀ / cap) by the map's softcap, its gradient by softcap_bwd from the products kept.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    // ---------------------------------------------------------------- the composition

    private static readonly Dictionary<string, bool> s_fusedFaster = [];

    /// <summary>O for rows of queries: the fused rows or the composition, whichever this processor measures faster at this
    /// shape and type (once); returns the lane's name.</summary>
    public static string Attention<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IFloatAdad<TN>
    {
        // Tensors through views (AttentionArgs.QRows, KRows, VRows, ORows): every form over their rows moved a head at a time.
        if (a.Laid) return Laid<TN>(a, q, k, v, o, static (h, q, k, v, o) => Attention<TN>(h, q, k, v, o));
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionComposed<TN>(a, q, k, v, o);
        string key = $"{TN.Name}:{a.Heads}x{a.Lq}x{a.Lk}x{a.D}:{a.Causal}:{a.Spans is not null}:{a.HasStarts}";
        bool fused;
        lock (s_fusedFaster)
        {
            if (!s_fusedFaster.TryGetValue(key, out fused))
            {
                // As Measure.Faster settles it: rounds of several calls (a short run's timer grain and team wake-up
                // averaged out), alternating, on until both minimums settle.
                var (tf, tc) = Measure.Faster(() => AttentionRows<TN>(a, q, k, v, o), () => AttentionComposed<TN>(a, q, k, v, o));
                fused = s_fusedFaster[key] = tf <= tc;
                Note($"attention {key}: fused rows {tf:0.000} ms, composition {tc:0.000} ms: {(fused ? "fused rows" : "composition")}");
            }
        }

        return fused ? AttentionRows<TN>(a, q, k, v, o) : AttentionComposed<TN>(a, q, k, v, o);
    }

    /// <summary>O for rows of queries: the composition; returns the lane's name.</summary>
    public static string AttentionComposed<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return Laid<TN>(a, q, k, v, o, static (h, q, k, v, o) => AttentionComposed<TN>(h, q, k, v, o));
        var vis = new ComposeVisitor(a.Grouped(), q, k, v, o, null, null, null, null, null);
        TN.Float(ref vis);
        return vis.Lane;
    }

    /// <summary>O and each query's log-sum-exp of its scores (heads × Lq, as TN): the composition; returns the lane's name.</summary>
    public static string AttentionLse<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, byte[] lse) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return Laid<TN>(a, q, k, v, o, (h, q, k, v, o) => AttentionLse<TN>(h, q, k, v, o, lse));
        var vis = new ComposeVisitor(a.Grouped(), q, k, v, o, lse, null, null, null, null);
        TN.Float(ref vis);
        return vis.Lane;
    }

    // The composition in TN's float lane, the lane's own type TS holding the intermediates (S, P, dP, dS, the rows' lse).
    private struct ComposeVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[]? o, byte[]? lse, byte[]? dO, byte[]? dq, byte[]? dk, byte[]? dv) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            var compose = new Compose<TN, TLane, TL>(a, q, k, v, o, lse, dO, dq, dk, dv);
            TL.Own(ref compose);
        }
    }

    private readonly struct Compose<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[]? o, byte[]? lse, byte[]? dO, byte[]? dq, byte[]? dk, byte[]? dv) : IOwnVisitor<TLane>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TS>() where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane>
        {
            int size = Unsafe.SizeOf<TLane>();
            // A block format's outputs: the lane's values (TS), encoded once (a block may straddle heads); otherwise each
            // product writes its head's rows as the type.
            bool lanes = TN.Granularity > 1;
            if (dO is null)
            {
                // The rows' log-sum-exp: lane values, stored as the type once.
                var ll = lse is null ? null : new byte[(long)a.Heads * a.Lq * size];
                if (!lanes) Forward<TS, TN>(o!, ll);
                else
                {
                    var lo = new byte[a.QValues * size];
                    Forward<TS, TS>(lo, ll);
                    StoreLanes<TN, TLane>(lo, o!, 0, a.QValues);
                }

                if (ll is not null) StoreLanes<TN, TLane>(ll, lse!, 0, (long)a.Heads * a.Lq);
                return;
            }

            if (!lanes) { Backward<TS, TN>(dq!, dk!, dv!); return; }
            var lq = new byte[a.QValues * size];
            var lk = new byte[a.KValues * size];
            var lv = new byte[a.KValues * size];
            Backward<TS, TS>(lq, lk, lv);
            StoreLanes<TN, TLane>(lq, dq!, 0, a.QValues);
            StoreLanes<TN, TLane>(lk, dk!, 0, a.KValues);
            StoreLanes<TN, TLane>(lv, dv!, 0, a.KValues);
        }

        // O (as TO) = softmax(scale·Q·Kᵀ + mask)·V per head (the scores capped when there is a cap); the rows' log-sum-exp
        // into ll (lane storage, heads × Lq) when asked.
        private void Forward<TS, TO>(byte[] lo, byte[]? ll) where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TO : struct, IAdad<TO, TLane>
        {
            int size = Unsafe.SizeOf<TLane>(), pairs = (int)((long)a.Lq * a.Lk);
            var s = Scratch.Get(Scratch.AttnS, (long)pairs * size);
            var row = ll is null ? null : new byte[(long)a.Lq * size];
            var limits = Limits(a);
            var softmax = RowOps.Find(ll is null ? "scale_mask_softmax" : "scale_mask_softmax_lse");
            var mask = SpanMask<TS>(a);
            bool capped = a.Softcap > 0;
            for (int h = 0; h < a.Heads; h++)
            {
                int qOff = (int)((long)h * a.Lq * a.D), kOff = (int)((long)h * a.Lk * a.D);
                MatMul<TN, TN, TS, TLane, TLane, TL>(q, k, s, a.Lq, a.Lk, a.D, new Operands(qOff, false, kOff, true, 0));      // S = Q_h·K_hᵀ
                if (capped) MapFloat<TS>("softcap", s, null, null, s, pairs, a.Softcap, a.Scale);                             // cap·tanh(scale·S / cap)
                RowOp<TS>(softmax, s, mask, null, null, null, limits, s, row, a.Lq, a.Lk, capped ? 1 : a.Scale, 1);
                if (row is not null) Array.Copy(row, 0, ll!, (long)h * a.Lq * size, row.Length);
                MatMul<TS, TN, TO, TLane, TLane, TL>(s, v, lo, a.Lq, a.D, a.Lk, new Operands(0, false, kOff, false, qOff));     // O_h = P·V_h
            }
        }

        // dQ, dK, dV (as TO): P recomputed; dV = Pᵀ·dO, dP = dO·Vᵀ, dS = softmax_bwd(P, dP)·scale (with a cap: times
        // softcap_bwd's slope at the products kept), dQ = dS·K, dK = dSᵀ·Q.
        private void Backward<TS, TO>(byte[] gq, byte[] gk, byte[] gv) where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TO : struct, IAdad<TO, TLane>
        {
            int size = Unsafe.SizeOf<TLane>(), pairs = (int)((long)a.Lq * a.Lk);
            var p = Scratch.Get(Scratch.AttnS, (long)pairs * size);
            var dp = Scratch.Get(Scratch.AttnDp, p.Length);
            bool capped = a.Softcap > 0;
            var raw = capped ? Scratch.Get(Scratch.AttnRaw, p.Length) : null;
            var limits = Limits(a);
            var softmax = RowOps.Find("scale_mask_softmax");
            var softmaxBwd = RowOps.Find("softmax_bwd");
            var mask = SpanMask<TS>(a);
            for (int h = 0; h < a.Heads; h++)
            {
                int qOff = (int)((long)h * a.Lq * a.D), kOff = (int)((long)h * a.Lk * a.D);
                MatMul<TN, TN, TS, TLane, TLane, TL>(q, k, raw ?? p, a.Lq, a.Lk, a.D, new Operands(qOff, false, kOff, true, 0));
                if (capped) MapFloat<TS>("softcap", raw!, null, null, p, pairs, a.Softcap, a.Scale);
                RowOp<TS>(softmax, p, mask, null, null, null, limits, p, null, a.Lq, a.Lk, capped ? 1 : a.Scale, 1);
                MatMul<TS, TN, TO, TLane, TLane, TL>(p, dO!, gv, a.Lk, a.D, a.Lq, new Operands(0, true, qOff, false, kOff));       // dV = Pᵀ·dO
                MatMul<TN, TN, TS, TLane, TLane, TL>(dO!, v, dp, a.Lq, a.Lk, a.D, new Operands(qOff, false, kOff, true, 0));     // dP = dO·Vᵀ
                RowOp<TS>(softmaxBwd, p, dp, null, null, null, null, dp, null, a.Lq, a.Lk, 0, 0);                              // dS = P·(dP − Σ P·dP)
                if (capped) MapFloat<TS>("softcap_bwd", raw!, dp, null, dp, pairs, a.Softcap, a.Scale);                       // · the cap's slope
                else MapFloat<TS>("scale", dp, null, null, dp, pairs, a.Scale, 0);                                             // · scale
                MatMul<TS, TN, TO, TLane, TLane, TL>(dp, k, gq, a.Lq, a.D, a.Lk, new Operands(0, false, kOff, false, qOff));      // dQ = dS·K
                MatMul<TS, TN, TO, TLane, TLane, TL>(dp, q, gk, a.Lk, a.D, a.Lq, new Operands(0, true, qOff, false, kOff));       // dK = dSᵀ·Q
            }
        }
    }

    // Each query's last visible key (limit − 1): scale_mask_softmax's causal rule excludes the keys past it.
    private static int[] Limits(AttentionArgs a)
    {
        var limits = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) limits[i] = a.Limit(i) - 1;
        return limits;
    }

    /// <summary>The additive mask (Lq × Lk, lane storage) excluding the keys below each query's span start by the excluded
    /// score (CacheView.Excluded of the lane's storage: −∞ where it stores −∞, so scale_mask_softmax gives such a key no weight
    /// and a query left with no key 0); null when no span starts.</summary>
    private static byte[]? SpanMask<TS>(AttentionArgs a) where TS : struct, IFloatAdad<TS>
    {
        if (!a.HasStarts) return null;
        double excluded = CacheView.Excluded<TS>();
        var m = new double[(long)a.Lq * a.Lk];
        for (int i = 0; i < a.Lq; i++) for (int j = 0; j < a.Start(i); j++) m[(long)i * a.Lk + j] = excluded;
        return Stored.StoreValues<TS>(m);
    }

    private static readonly Dictionary<string, bool> s_backFusedFaster = [];

    /// <summary>The gate's override of the backward's measured choice (null: measured; true: the fused passes; false: the composition).</summary>
    internal static bool? FusedBackwardCpu;

    /// <summary>dQ, dK, dV: the fused rows and keys passes or the composition, whichever this processor measures faster at
    /// this shape and type (once, as the forward's rows); returns the lane's name.</summary>
    public static string AttentionBackward<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return LaidBackward<TN>(a, q, k, v, dO, dq, dk, dv);
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionBackwardComposed<TN>(a, q, k, v, dO, dq, dk, dv);
        if (FusedBackwardCpu is bool forced) return forced ? AttentionBackwardRows<TN>(a, q, k, v, dO, dq, dk, dv) : AttentionBackwardComposed<TN>(a, q, k, v, dO, dq, dk, dv);
        string key = $"{TN.Name}:{a.Heads}x{a.Lq}x{a.Lk}x{a.D}:{a.Causal}:{a.Spans is not null}:{a.HasStarts}";
        bool fused;
        lock (s_backFusedFaster)
        {
            if (!s_backFusedFaster.TryGetValue(key, out fused))
            {
                var (tf, tc) = Measure.Faster(() => AttentionBackwardRows<TN>(a, q, k, v, dO, dq, dk, dv), () => AttentionBackwardComposed<TN>(a, q, k, v, dO, dq, dk, dv));
                fused = s_backFusedFaster[key] = tf <= tc;
                Note($"attention_bwd {key}: fused passes {tf:0.000} ms, composition {tc:0.000} ms: {(fused ? "fused passes" : "composition")}");
            }
        }

        return fused ? AttentionBackwardRows<TN>(a, q, k, v, dO, dq, dk, dv) : AttentionBackwardComposed<TN>(a, q, k, v, dO, dq, dk, dv);
    }

    /// <summary>dQ, dK, dV: the composition; returns the lane's name.</summary>
    public static string AttentionBackwardComposed<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IFloatAdad<TN>
    {
        var vis = new ComposeVisitor(a.Grouped(), q, k, v, null, null, dO, dq, dk, dv);
        TN.Float(ref vis);
        return vis.Lane;
    }

    // ---------------------------------------------------------------- tensors through views (AttentionArgs.QRows, KRows, VRows, ORows)

    // A view whose rows, strides and offset are whole storage units (a type of 8 bits or more; a block format whose rows
    // are whole blocks) is moved a head at a time as its rows' bytes — exact, a task a run of rows, into pooled buffers —
    // the form run on them, and its outputs' rows moved back into their views. A block format whose rows share blocks
    // goes by AttentionLayout's strided copies, which re-encode by blocks as every layout's output does.

    // Heads × rows of d through the view (null: the tensor as it is) into pooled storage a head at a time.
    private static (byte[] Rows, bool Rented) RowsIn<TN>(HeadRows? view, byte[] source, int heads, int rows, int d) where TN : struct, IAdad<TN>
    {
        if (view is null) return (source, false);
        int rowBytes = (int)TN.Bytes(d), n = heads * rows;
        var rented = ArrayPool<byte>.Shared.Rent(Math.Max(1, n * rowBytes));
        if (n > 0 && rowBytes > 0) Team.For(n, Math.Max(1, n / (Threads * 4)), (long)n * rowBytes, new MoveRowsBody<TN>(view, source, rented, rows, rowBytes, true));
        return (rented, true);
    }

    // A pooled buffer for an output a head at a time (its own when it has no view).
    private static (byte[] Rows, bool Rented) RowsOut(HeadRows? view, byte[] output, long bytes) =>
        view is null ? (output, false) : (ArrayPool<byte>.Shared.Rent((int)Math.Max(1, bytes)), true);

    // An output's rows a head at a time into its view (a dense order: every row of the view written once).
    private static void RowsBack<TN>(HeadRows? view, byte[] heads, byte[] output, int count, int rows, int d) where TN : struct, IAdad<TN>
    {
        if (view is null) return;
        int rowBytes = (int)TN.Bytes(d), n = count * rows;
        if (n > 0 && rowBytes > 0) Team.For(n, Math.Max(1, n / (Threads * 4)), (long)n * rowBytes, new MoveRowsBody<TN>(view, heads, output, rows, rowBytes, false));
    }

    private static void Return(params (byte[] Rows, bool Rented)[] buffers)
    {
        foreach (var (rows, rented) in buffers) if (rented) ArrayPool<byte>.Shared.Return(rows);
    }

    private readonly struct MoveRowsBody<TN>(HeadRows view, byte[] source, byte[] destination, int rows, int rowBytes, bool toHeads) : Team.IBody where TN : struct, IAdad<TN>
    {
        public void Run(int first, int last)
        {
            for (int r = first; r < last; r++)
            {
                // Row r a head at a time (head r / rows, row r % rows) and its place in the view.
                int at = (int)TN.Bytes(view.At(r / rows, r % rows));
                if (toHeads) source.AsSpan(at, rowBytes).CopyTo(destination.AsSpan(r * rowBytes, rowBytes));
                else source.AsSpan(r * rowBytes, rowBytes).CopyTo(destination.AsSpan(at, rowBytes));
            }
        }
    }

    private static string Laid<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, Func<AttentionArgs, byte[], byte[], byte[], byte[], string> run) where TN : struct, IFloatAdad<TN>
    {
        if (!AttentionLayout.WholeUnits<TN>(a)) return AttentionLayout.Forward<byte[], int[], TN>(CpuDevice.Device, a, q, k, v, o, (_, h, qh, kh, vh, oh) => run(h, qh, kh, vh, oh));
        a.Validate();
        int hk = a.KvHeadCount;
        var qh = RowsIn<TN>(a.QRows, q, a.Heads, a.Lq, a.D);
        var kh = RowsIn<TN>(a.KRows, k, hk, a.Lk, a.D);
        var vh = RowsIn<TN>(a.VRows, v, hk, a.Lk, a.D);
        var oh = RowsOut(a.ORows, o, TN.Bytes(a.QValues));
        try
        {
            string lane = run(a.HeadMajor, qh.Rows, kh.Rows, vh.Rows, oh.Rows);
            RowsBack<TN>(a.ORows, oh.Rows, o, a.Heads, a.Lq, a.D);
            return lane;
        }
        finally { Return(qh, kh, vh, oh); }
    }

    /// <summary>O for Q and O through their views over tensors the form reads as they are (a cache's K and V).</summary>
    internal static string LaidQueries<TN>(AttentionArgs a, byte[] q, byte[] o, Func<AttentionArgs, byte[], byte[], string> run) where TN : struct, IFloatAdad<TN>
    {
        if (!AttentionLayout.WholeUnits<TN>(a)) return AttentionLayout.Queries<byte[], int[], TN>(CpuDevice.Device, a, q, o, (_, h, qh, oh) => run(h, qh, oh));
        a.Validate();
        if (a.KRows is not null || a.VRows is not null) throw new ArgumentException("attention over a cache: K and V are the cache's rows (no views)");
        var qh = RowsIn<TN>(a.QRows, q, a.Heads, a.Lq, a.D);
        var oh = RowsOut(a.ORows, o, TN.Bytes(a.QValues));
        try
        {
            string lane = run(a.HeadMajor, qh.Rows, oh.Rows);
            RowsBack<TN>(a.ORows, oh.Rows, o, a.Heads, a.Lq, a.D);
            return lane;
        }
        finally { Return(qh, oh); }
    }

    private static string LaidBackward<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IFloatAdad<TN>
    {
        if (!AttentionLayout.WholeUnits<TN>(a)) return AttentionLayout.Backward<byte[], int[], TN>(CpuDevice.Device, a, q, k, v, dO, dq, dk, dv);
        a.ValidateBackward();
        int hk = a.KvHeadCount;
        var qh = RowsIn<TN>(a.QRows, q, a.Heads, a.Lq, a.D);
        var kh = RowsIn<TN>(a.KRows, k, hk, a.Lk, a.D);
        var vh = RowsIn<TN>(a.VRows, v, hk, a.Lk, a.D);
        var gh = RowsIn<TN>(a.ORows, dO, a.Heads, a.Lq, a.D);
        var dqh = RowsOut(a.QRows, dq, TN.Bytes(a.QValues));
        var dkh = RowsOut(a.KRows, dk, TN.Bytes(a.KValues));
        var dvh = RowsOut(a.VRows, dv, TN.Bytes(a.KValues));
        try
        {
            string lane = AttentionBackward<TN>(a.HeadMajor, qh.Rows, kh.Rows, vh.Rows, gh.Rows, dqh.Rows, dkh.Rows, dvh.Rows);
            RowsBack<TN>(a.QRows, dqh.Rows, dq, a.Heads, a.Lq, a.D);
            RowsBack<TN>(a.KRows, dkh.Rows, dk, hk, a.Lk, a.D);
            RowsBack<TN>(a.VRows, dvh.Rows, dv, hk, a.Lk, a.D);
            return lane;
        }
        finally { Return(qh, kh, vh, gh, dqh, dkh, dvh); }
    }

    // ---------------------------------------------------------------- decode: one query at a time, K and V read once

    /// <summary>O for a few queries against a long cache, each query on its own: scores, softmax and the context in the lane.</summary>
    public static string AttentionDecode<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return Laid<TN>(a, q, k, v, o, static (h, q, k, v, o) => AttentionDecode<TN>(h, q, k, v, o));
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionComposed<TN>(a, q, k, v, o);
        var vis = new DecodeVisitor(a, q, k, v, o);
        TN.Float(ref vis);
        return vis.Lane;
    }

    // The keys go in parts of a fixed length, so a few long rows still give the team many pieces: a part's (max, Σ exp,
    // context) per row, the parts of a row combined after (exp(m_p − M) rescales each). The length is a choice measured
    // once: a part's K and V rows of a 128-wide head are 128 kB each in float32, well past a piece's overhead and inside
    // the caches; it does not follow the thread count, so neither does the result.
    private const int DecodePart = 256;

    private struct DecodeVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            int rows = a.Heads * a.Lq, parts = Math.Max(1, (a.Lk + DecodePart - 1) / DecodePart);
            var pool = ArrayPool<TLane>.Shared;
            var ms = pool.Rent(rows * parts);
            var sums = pool.Rent(rows * parts);
            var accs = pool.Rent(rows * parts * a.D);
            var keys = ArrayPool<int>.Shared.Rent(rows * parts);
            try
            {
                Team.For(a.Heads * parts, 1, a.KValues * 2, new DecodeBody<TN, TLane, TL>(a, q, k, v, parts, ms, sums, accs, keys));
                Team.For(rows, RowChunk<TN>(rows, a.D, rows / (Threads * 4)), (long)rows * parts * a.D * Unsafe.SizeOf<TLane>(),
                    new DecodeJoin<TN, TLane, TL>(a, parts, ms, sums, accs, keys, o));
            }
            finally
            {
                pool.Return(ms); pool.Return(sums); pool.Return(accs); ArrayPool<int>.Shared.Return(keys);
            }
        }
    }

    // One head's rows over one part of its keys: each K row read once for all the head's rows (a group's query rows share
    // it), the scores, each row's max and Σ exp over its keys in the part, then each V row read once into the rows' contexts.
    private readonly struct DecodeBody<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, int parts, TLane[] ms, TLane[] sums, TLane[] accs, int[] keys) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            int d = a.D, lq = a.Lq;
            var qr = pool.Rent(lq * d);
            var kr = pool.Rent(d);
            var s = pool.Rent(lq * DecodePart);
            var from = ArrayPool<int>.Shared.Rent(lq);
            var to = ArrayPool<int>.Shared.Rent(lq);
            try
            {
                TLane scale = TL.From(a.Scale);
                bool direct = TN.Direct;
                int w = Vector<TLane>.Count;
                for (int item = first; item < last; item++)
                {
                    int h = item / parts, p = item % parts, lo = p * DecodePart, hi = Math.Min(a.Lk, lo + DecodePart);
                    long kBase = (long)h * a.Lk * d;
                    int keysLo = hi, keysHi = lo;
                    for (int i = 0; i < lq; i++)
                    {
                        int limit = a.Limit(i), start = Math.Min(a.Start(i), limit);
                        from[i] = Math.Max(lo, start); to[i] = Math.Min(hi, limit);
                        if (from[i] < to[i]) { keysLo = Math.Min(keysLo, from[i]); keysHi = Math.Max(keysHi, to[i]); }
                    }

                    if (keysLo < keysHi) TN.Load(q, (int)((long)h * lq * d), qr.AsSpan(0, lq * d));
                    for (int j = keysLo; j < keysHi; j++)
                    {
                        var kj = Row(k, kBase + (long)j * d, d, direct, kr);
                        for (int i = 0; i < lq; i++)
                        {
                            if (j >= from[i] && j < to[i]) s[i * DecodePart + j - lo] = TL.Mul(RowMath<TLane, TL>.Dot(qr.AsSpan(i * d, d), kj), scale);
                        }
                    }

                    for (int i = 0; i < lq; i++)
                    {
                        int at = (h * lq + i) * parts + p;
                        accs.AsSpan(at * d, d).Clear();
                        keys[at] = Math.Max(0, to[i] - from[i]);
                        if (keys[at] == 0) { ms[at] = TL.NegativeInfinity; sums[at] = TL.Zero; continue; }
                        var si = s.AsSpan(i * DecodePart + from[i] - lo, keys[at]);
                        TLane m = TL.NegativeInfinity;
                        foreach (var x in si) m = TL.Max(m, x);
                        ms[at] = m;
                        sums[at] = RowMath<TLane, TL>.ExpSum(si, m, si);
                    }

                    for (int j = keysLo; j < keysHi; j++)
                    {
                        var vj = Row(v, kBase + (long)j * d, d, direct, kr);
                        for (int i = 0; i < lq; i++)
                        {
                            if (j < from[i] || j >= to[i]) continue;
                            var acc = accs.AsSpan(((h * lq + i) * parts + p) * d, d);
                            TLane pj = s[i * DecodePart + j - lo];
                            var pv = new Vector<TLane>(pj);
                            int t = 0;
                            for (; t + w <= d; t += w) TL.Fma(new Vector<TLane>(vj[t..]), pv, new Vector<TLane>(acc[t..])).CopyTo(acc[t..]);
                            for (; t < d; t++) acc[t] = TL.Fma(vj[t], pj, acc[t]);
                        }
                    }
                }
            }
            finally
            {
                pool.Return(qr); pool.Return(kr); pool.Return(s);
                ArrayPool<int>.Shared.Return(from); ArrayPool<int>.Shared.Return(to);
            }
        }

        // A stored row of d values in the lane: in place when the type is its lane, else loaded into the scratch row.
        private static ReadOnlySpan<TLane> Row(byte[] x, long at, int d, bool direct, TLane[] scratch)
        {
            if (direct) return MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice((int)at, d);
            TN.Load(x, (int)at, scratch.AsSpan(0, d));
            return scratch.AsSpan(0, d);
        }
    }

    // A row's parts combined: M the largest max of the parts that hold keys, each part's sum and context scaled by
    // exp(m_p − M) into the row's, the context divided by the sum (a row that sees no key: 0) and stored as the type.
    // One part: exp(0) = 1, the single pass's result exactly.
    private readonly struct DecodeJoin<TN, TLane, TL>(AttentionArgs a, int parts, TLane[] ms, TLane[] sums, TLane[] accs, int[] keys, byte[] o) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int d = a.D, w = Vector<TLane>.Count;
            var acc = ArrayPool<TLane>.Shared.Rent(d);
            var store = new UnitStore<TN, TLane>(o, d);
            try
            {
                for (int r = first; r < last; r++)
                {
                    var row = acc.AsSpan(0, d);
                    row.Clear();
                    TLane big = TL.NegativeInfinity, sum = TL.Zero;
                    bool any = false;
                    for (int p = 0; p < parts; p++)
                    {
                        int at = r * parts + p;
                        if (keys[at] == 0) continue;
                        big = any ? TL.Max(big, ms[at]) : ms[at];
                        any = true;
                    }

                    for (int p = 0; any && p < parts; p++)
                    {
                        int at = r * parts + p;
                        if (keys[at] == 0) continue;
                        TLane c = TL.Exp(TL.Sub(ms[at], big));
                        sum = TL.Fma(sums[at], c, sum);
                        var part = accs.AsSpan(at * d, d);
                        var cv = new Vector<TLane>(c);
                        int t = 0;
                        for (; t + w <= d; t += w) TL.Fma(new Vector<TLane>(part[t..]), cv, new Vector<TLane>(row[t..])).CopyTo(row[t..]);
                        for (; t < d; t++) row[t] = TL.Fma(part[t], c, row[t]);
                    }

                    TLane inv = any ? TL.Div(TL.From(1), sum) : TL.Zero;
                    RowMath<TLane, TL>.Scale(row, TL.Zero, inv, row);
                    store.Put(row, (long)r * d);
                }

                store.Done();
            }
            finally { ArrayPool<TLane>.Shared.Return(acc); }
        }
    }
}
