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
//                     keys in chunks, the softmax online, K and V read once); the same per query on the CPU.
//   attention_bwd     dQ, dK, dV from dO: the composition (P recomputed; dV = Pᵀ·dO, dP = dO·Vᵀ, dS = softmax_bwd(P, dP),
//                     dQ = scale·dS·K, dK = scale·dSᵀ·Q).
//
// Causal: query i (of Lq) sees keys j ≤ i + (Lk − Lq): the queries are the last Lq positions of the keys. A span per
// query (spans[i] = the keys it sees, from 0) replaces the causal rule when given.

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
        string key = $"{TN.Name}:{a.Heads}x{a.Lq}x{a.Lk}x{a.D}:{a.Causal}:{a.Spans is not null}:{a.SpanStarts is not null}";
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
        var vis = new ComposeVisitor(a, q, k, v, o, null, null, null, null);
        TN.Float(ref vis);
        return vis.Lane;
    }

    // The composition in TN's float lane, the lane's own type TS holding the intermediates (S, P, dP, dS).
    private struct ComposeVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[]? o, byte[]? dO, byte[]? dq, byte[]? dk, byte[]? dv) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            var compose = new Compose<TN, TLane, TL>(a, q, k, v, o, dO, dq, dk, dv);
            TL.Own(ref compose);
        }
    }

    private readonly struct Compose<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[]? o, byte[]? dO, byte[]? dq, byte[]? dk, byte[]? dv) : IOwnVisitor<TLane>
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
                if (!lanes) { Forward<TS, TN>(o!); return; }
                var lo = new byte[a.QValues * size];
                Forward<TS, TS>(lo);
                StoreLanes<TN, TLane>(lo, o!, 0, a.QValues);
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

        // O (as TO) = softmax(scale·Q·Kᵀ + mask)·V per head.
        private void Forward<TS, TO>(byte[] lo) where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TO : struct, IAdad<TO, TLane>
        {
            var s = new byte[(long)a.Lq * a.Lk * Unsafe.SizeOf<TLane>()];
            var limits = Limits(a);
            var softmax = RowOps.Find("scale_mask_softmax");
            var mask = SpanMask<TS>(a);
            for (int h = 0; h < a.Heads; h++)
            {
                int qOff = (int)((long)h * a.Lq * a.D), kOff = (int)((long)h * a.Lk * a.D);
                MatMul<TN, TN, TS, TLane, TLane, TL>(q, k, s, a.Lq, a.Lk, a.D, new Operands(qOff, false, kOff, true, 0));      // S = Q_h·K_hᵀ
                RowOp<TS>(softmax, s, mask, null, null, null, limits, s, null, a.Lq, a.Lk, a.Scale, 1);
                MatMul<TS, TN, TO, TLane, TLane, TL>(s, v, lo, a.Lq, a.D, a.Lk, new Operands(0, false, kOff, false, qOff));     // O_h = P·V_h
            }
        }

        // dQ, dK, dV (as TO): P recomputed; dV = Pᵀ·dO, dP = dO·Vᵀ, dS = softmax_bwd(P, dP)·scale, dQ = dS·K, dK = dSᵀ·Q.
        private void Backward<TS, TO>(byte[] gq, byte[] gk, byte[] gv) where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TO : struct, IAdad<TO, TLane>
        {
            int size = Unsafe.SizeOf<TLane>();
            var p = new byte[(long)a.Lq * a.Lk * size];
            var dp = new byte[p.Length];
            var limits = Limits(a);
            var softmax = RowOps.Find("scale_mask_softmax");
            var softmaxBwd = RowOps.Find("softmax_bwd");
            var mask = SpanMask<TS>(a);
            for (int h = 0; h < a.Heads; h++)
            {
                int qOff = (int)((long)h * a.Lq * a.D), kOff = (int)((long)h * a.Lk * a.D);
                MatMul<TN, TN, TS, TLane, TLane, TL>(q, k, p, a.Lq, a.Lk, a.D, new Operands(qOff, false, kOff, true, 0));
                RowOp<TS>(softmax, p, mask, null, null, null, limits, p, null, a.Lq, a.Lk, a.Scale, 1);
                MatMul<TS, TN, TO, TLane, TLane, TL>(p, dO!, gv, a.Lk, a.D, a.Lq, new Operands(0, true, qOff, false, kOff));       // dV = Pᵀ·dO
                MatMul<TN, TN, TS, TLane, TLane, TL>(dO!, v, dp, a.Lq, a.Lk, a.D, new Operands(qOff, false, kOff, true, 0));     // dP = dO·Vᵀ
                RowOp<TS>(softmaxBwd, p, dp, null, null, null, null, dp, null, a.Lq, a.Lk, 0, 0);                              // dS = P·(dP − Σ P·dP)
                MapFloat<TS>("scale", dp, null, null, dp, (int)((long)a.Lq * a.Lk), a.Scale, 0);                               // · scale
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

    /// <summary>The additive mask (Lq × Lk, lane storage) excluding the keys below each query's span start; null when no span starts.</summary>
    private static byte[]? SpanMask<TS>(AttentionArgs a) where TS : struct, IFloatAdad<TS>
    {
        if (a.SpanStarts is null) return null;
        var m = new double[(long)a.Lq * a.Lk];
        for (int i = 0; i < a.Lq; i++) for (int j = 0; j < a.Start(i); j++) m[(long)i * a.Lk + j] = -400;
        return Stored.StoreValues<TS>(m);
    }

    private static readonly Dictionary<string, bool> s_backFusedFaster = [];

    /// <summary>The gate's override of the backward's measured choice (null: measured; true: the fused passes; false: the composition).</summary>
    internal static bool? FusedBackwardCpu;

    /// <summary>dQ, dK, dV: the fused rows and keys passes or the composition, whichever this processor measures faster at
    /// this shape and type (once, as the forward's rows); returns the lane's name.</summary>
    public static string AttentionBackward<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IFloatAdad<TN>
    {
        if (FusedBackwardCpu is bool forced) return forced ? AttentionBackwardRows<TN>(a, q, k, v, dO, dq, dk, dv) : AttentionBackwardComposed<TN>(a, q, k, v, dO, dq, dk, dv);
        string key = $"{TN.Name}:{a.Heads}x{a.Lq}x{a.Lk}x{a.D}:{a.Causal}:{a.Spans is not null}:{a.SpanStarts is not null}";
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
        var vis = new ComposeVisitor(a, q, k, v, null, dO, dq, dk, dv);
        TN.Float(ref vis);
        return vis.Lane;
    }

    // ---------------------------------------------------------------- decode: one query at a time, K and V read once

    /// <summary>O for a few queries against a long cache, each query on its own: scores, softmax and the context in the lane.</summary>
    public static string AttentionDecode<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IFloatAdad<TN>
    {
        var vis = new DecodeVisitor(a, q, k, v, o);
        TN.Float(ref vis);
        return vis.Lane;
    }

    private struct DecodeVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(a.Heads * a.Lq, RowChunk<TN>(a.Heads * a.Lq, a.D, 1), a.KValues * 2, new DecodeBody<TN, TLane, TL>(a, q, k, v, o));
        }
    }

    private struct DecodeBody<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var qr = pool.Rent(a.D);
            var kr = pool.Rent(a.D);
            var acc = pool.Rent(a.D);
            var s = pool.Rent(a.Lk);
            try
            {
                TLane scale = TL.From(a.Scale);
                for (int hi = first; hi < last; hi++)
                {
                    int h = hi / a.Lq, i = hi % a.Lq, limit = a.Limit(i), start = Math.Min(a.Start(i), limit);
                    long qi = ((long)h * a.Lq + i) * a.D, kBase = (long)h * a.Lk * a.D;
                    TN.Load(q, (int)qi, qr.AsSpan(0, a.D));
                    TLane m = TL.NegativeInfinity;
                    for (int j = start; j < limit; j++)
                    {
                        TN.Load(k, (int)(kBase + (long)j * a.D), kr.AsSpan(0, a.D));
                        s[j] = TL.Mul(RowMath<TLane, TL>.Dot(qr.AsSpan(0, a.D), kr.AsSpan(0, a.D)), scale);
                        m = TL.Max(m, s[j]);
                    }

                    TLane sum = limit <= start ? TL.Zero : RowMath<TLane, TL>.ExpSum(s.AsSpan(start, limit - start), m, s.AsSpan(start, limit - start));
                    acc.AsSpan(0, a.D).Clear();
                    for (int j = start; j < limit; j++)
                    {
                        TN.Load(v, (int)(kBase + (long)j * a.D), kr.AsSpan(0, a.D));
                        var pj = new Vector<TLane>(s[j]);
                        int w = Vector<TLane>.Count, t = 0;
                        for (; t + w <= a.D; t += w) TL.Fma(new Vector<TLane>(kr.AsSpan(t)), pj, new Vector<TLane>(acc.AsSpan(t))).CopyTo(acc.AsSpan(t));
                        for (; t < a.D; t++) acc[t] = TL.Fma(kr[t], s[j], acc[t]);
                    }

                    TLane inv = limit <= start ? TL.Zero : TL.Div(TL.From(1), sum);
                    RowMath<TLane, TL>.Scale(acc.AsSpan(0, a.D), TL.Zero, inv, acc.AsSpan(0, a.D));
                    TN.Store(acc.AsSpan(0, a.D), o, (int)qi);
                }
            }
            finally
            {
                pool.Return(qr); pool.Return(kr); pool.Return(acc); pool.Return(s);
            }
        }
    }
}
