// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention: O = softmax(scale · Q·Kᵀ + mask) · V per head, in the type's float lane. Three forms:
//
//   attention         rows of queries (prefill): the composition of Qiyas's own operations in lane storage — the product
//                     Q·Kᵀ on views, scale_mask_softmax (causal with the key offset, or a key span per row), the product
//                     P·V — every intermediate kept in the lane (float32 for the narrow floats, float64 for float64):
//                     a narrow type's Q, K, V are widened once and O narrowed once.
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

namespace Qiyas;

/// <summary>Q (heads × Lq × d), K and V (heads × Lk × d), O (heads × Lq × d); the scale; causal or spans per query.</summary>
public sealed record AttentionArgs(int Heads, int Lq, int Lk, int D, double Scale, bool Causal, int[]? Spans = null, int[]? SpanStarts = null)
{
    public long QValues => (long)Heads * Lq * D;
    public long KValues => (long)Heads * Lk * D;

    /// <summary>The keys query i sees: [Start(i), Limit(i)).</summary>
    public int Limit(int i) => Spans is not null ? Math.Min(Lk, Spans[i]) : Causal ? Math.Min(Lk, i + (Lk - Lq) + 1) : Lk;

    /// <summary>The first key query i sees (0 unless span starts are given).</summary>
    public int Start(int i) => SpanStarts is null ? 0 : Math.Clamp(SpanStarts[i], 0, Lk);

    /// <summary>The float64 reference of O, and of P when asked (heads × Lq × Lk).</summary>
    public double[] Reference(double[] q, double[] k, double[] v, double[]? p = null)
    {
        var o = new double[QValues];
        Parallel.For(0, Heads * Lq, hi =>
        {
            int h = hi / Lq, i = hi % Lq, limit = Limit(i), start = Start(i);
            var s = new double[Lk];
            double m = double.NegativeInfinity;
            for (int j = start; j < limit; j++)
            {
                double dot = 0;
                for (int t = 0; t < D; t++) dot += q[((long)h * Lq + i) * D + t] * k[((long)h * Lk + j) * D + t];
                s[j] = dot * Scale;
                m = Math.Max(m, s[j]);
            }

            double sum = 0;
            for (int j = start; j < limit; j++) sum += s[j] = Math.Exp(s[j] - m);
            for (int j = start; j < limit; j++) s[j] = sum > 0 ? s[j] / sum : 0;
            for (int j = 0; j < Math.Min(start, Lk); j++) s[j] = 0;
            for (int j = limit; j < Lk; j++) s[j] = 0;
            if (p is not null) Array.Copy(s, 0, p, ((long)h * Lq + i) * Lk, Lk);
            for (int t = 0; t < D; t++)
            {
                double acc = 0;
                for (int j = start; j < limit; j++) acc += s[j] * v[((long)h * Lk + j) * D + t];
                o[((long)h * Lq + i) * D + t] = acc;
            }
        });
        return o;
    }

    /// <summary>The float64 reference of the backward pass.</summary>
    public (double[] DQ, double[] DK, double[] DV) ReferenceBackward(double[] q, double[] k, double[] v, double[] dO)
    {
        var p = new double[(long)Heads * Lq * Lk];
        Reference(q, k, v, p);
        var dq = new double[QValues];
        var dk = new double[KValues];
        var dv = new double[KValues];
        for (int h = 0; h < Heads; h++)
        {
            for (int i = 0; i < Lq; i++)
            {
                long qi = ((long)h * Lq + i) * D;
                // dP_ij = dO_i · V_j; dS_ij = P_ij (dP_ij − Σ_j P_ij dP_ij)
                var dp = new double[Lk];
                double dot = 0;
                for (int j = 0; j < Lk; j++)
                {
                    long kj = ((long)h * Lk + j) * D;
                    double d = 0;
                    for (int t = 0; t < D; t++) d += dO[qi + t] * v[kj + t];
                    dp[j] = d;
                    dot += p[((long)h * Lq + i) * Lk + j] * d;
                }

                for (int j = 0; j < Lk; j++)
                {
                    long kj = ((long)h * Lk + j) * D;
                    double pij = p[((long)h * Lq + i) * Lk + j], ds = pij * (dp[j] - dot) * Scale;
                    for (int t = 0; t < D; t++)
                    {
                        dv[kj + t] += pij * dO[qi + t];
                        dq[qi + t] += ds * k[kj + t];
                        dk[kj + t] += ds * q[qi + t];
                    }
                }
            }
        }

        return (dq, dk, dv);
    }
}

internal sealed partial class Kernels
{
    // ---------------------------------------------------------------- lane storage: a narrow type widened to its lane's storage type

    private static bool Wide<TN>() where TN : struct, IQiyasNumber<TN> => TN.Bits == 64;

    /// <summary>The stored values of x (count of them) as float32 or float64 storage (TN's float lane): TN's own bytes when it is that type.</summary>
    public static byte[] ToLane<TN>(byte[] x, long count) where TN : struct, IQiyasFloat<TN>
    {
        if (TN.Name is "float32" or "float64") return x;
        var v = new WidenVisitor(x, count, null);
        TN.Float(ref v);
        return v.Result!;
    }

    /// <summary>ToLane into an existing lane array of at least count lane values (a narrow type only).</summary>
    public static void ToLaneInto<TN>(byte[] x, long count, byte[] into) where TN : struct, IQiyasFloat<TN>
    {
        var v = new WidenVisitor(x, count, into);
        TN.Float(ref v);
    }

    private struct WidenVisitor(byte[] x, long count, byte[]? into) : IFloatLaneVisitor
    {
        public byte[]? Result;

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            var result = into ?? new byte[count * Unsafe.SizeOf<TLane>()];
            var lanes = MemoryMarshal.Cast<byte, TLane>(result.AsSpan());
            Team.For((int)((count + 4095) / 4096), 1, count, new WidenBody<TN, TLane>(x, result, count));
            Result = result;
        }
    }

    private struct WidenBody<TN, TLane>(byte[] x, byte[] result, long count) : Team.IBody where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        public readonly void Run(int first, int last)
        {
            var lanes = MemoryMarshal.Cast<byte, TLane>(result.AsSpan());
            for (int chunk = first; chunk < last; chunk++)
            {
                long at = (long)chunk * 4096;
                int len = (int)Math.Min(4096, count - at);
                TN.Load(x, (int)at, lanes.Slice((int)at, len));
            }
        }
    }

    /// <summary>Lane storage (float32 / float64 bytes) back into the type at output.</summary>
    public static void FromLane<TN>(byte[] lanes, byte[] output, long count) where TN : struct, IQiyasFloat<TN>
    {
        if (TN.Name is "float32" or "float64") { Array.Copy(lanes, output, output.Length); return; }
        var v = new NarrowVisitor(lanes, output, count);
        TN.Float(ref v);
    }

    private struct NarrowVisitor(byte[] lanes, byte[] output, long count) : IFloatLaneVisitor
    {
        public readonly void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            int per = TN.Granularity;
            int chunk = 4096 / per * per;
            Team.For((int)((count + chunk - 1) / chunk), 1, count, new NarrowBody<TN, TLane>(lanes, output, count, chunk));
        }
    }

    private struct NarrowBody<TN, TLane>(byte[] lanes, byte[] output, long count, int chunk) : Team.IBody where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        public readonly void Run(int first, int last)
        {
            var values = MemoryMarshal.Cast<byte, TLane>(lanes.AsSpan());
            for (int c = first; c < last; c++)
            {
                long at = (long)c * chunk;
                int len = (int)Math.Min(chunk, count - at);
                TN.Store(values.Slice((int)at, len), output, (int)at);
            }
        }
    }

    /// <summary>Values stored as TS, each through TS's float lane (exact for the values the lane holds: a span mask's 0 and −400).</summary>
    internal static byte[] StoreValues<TS>(double[] values) where TS : struct, IQiyasFloat<TS>
    {
        var v = new StoreValuesVisitor(values);
        TS.Float(ref v);
        return v.Result!;
    }

    private struct StoreValuesVisitor(double[] values) : IFloatLaneVisitor
    {
        public byte[]? Result;

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            var lanes = new TLane[values.Length];
            for (int i = 0; i < lanes.Length; i++) lanes[i] = TL.From(values[i]);
            Result = new byte[TN.Bytes(values.Length)];
            TN.Store(lanes, Result, 0);
        }
    }

    // ---------------------------------------------------------------- the composition, in lane storage

    private static readonly Dictionary<string, bool> s_fusedFaster = [];

    /// <summary>O for rows of queries: the fused rows or the composition, whichever this processor measures faster at this
    /// shape and type (once); returns the lane's name.</summary>
    public static string Attention<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IQiyasFloat<TN>
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

    /// <summary>O for rows of queries: the composition in lane storage; returns the lane's name.</summary>
    public static string AttentionComposed<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IQiyasFloat<TN>
    {
        if (Wide<TN>()) { AttentionIn<Float64>(a, ToLane<TN>(q, a.QValues), ToLane<TN>(k, a.KValues), ToLane<TN>(v, a.KValues), o, TN.Name == "float64" ? null : (bytes => FromLane<TN>(bytes, o, a.QValues))); return "f64"; }
        AttentionIn<Float32>(a, ToLane<TN>(q, a.QValues), ToLane<TN>(k, a.KValues), ToLane<TN>(v, a.KValues), o, TN.Name == "float32" ? null : (bytes => FromLane<TN>(bytes, o, a.QValues)));
        return "f32";
    }

    private static void AttentionIn<TS>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, Action<byte[]>? narrow) where TS : struct, IQiyasFloat<TS>
    {
        int size = TS.Bits / 8;
        var lo = narrow is null ? o : new byte[a.QValues * size];
        var s = new byte[(long)a.Lq * a.Lk * size];
        // positions = the last visible key of each query (limit − 1): scale_mask_softmax's causal rule excludes the keys past it.
        var limits = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) limits[i] = a.Limit(i) - 1;
        var softmax = RowOps.Find("scale_mask_softmax");
        var mask = SpanMask<TS>(a);
        for (int h = 0; h < a.Heads; h++)
        {
            long qOff = (long)h * a.Lq * a.D, kOff = (long)h * a.Lk * a.D;
            // S = Q_h · K_hᵀ (Lq × Lk) from the views at the head's offsets.
            var sq = Slice(q, qOff * size, a.Lq * a.D * size);
            var sk = Slice(k, kOff * size, (long)a.Lk * a.D * size);
            var sv = Slice(v, kOff * size, (long)a.Lk * a.D * size);
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), sq, sk, s);
            RowOp<TS>(softmax, s, mask, null, null, null, limits, s, null, a.Lq, a.Lk, a.Scale, 1);
            var so = new byte[(long)a.Lq * a.D * size];
            MatMulView<TS>(new ProductView(a.Lq, a.D, a.Lk, false, false), s, sv, so);
            Array.Copy(so, 0, lo, qOff * size, so.Length);
        }

        narrow?.Invoke(lo);
    }

    /// <summary>The additive mask (Lq × Lk, lane storage) excluding the keys below each query's span start; null when no span starts.</summary>
    private static byte[]? SpanMask<TS>(AttentionArgs a) where TS : struct, IQiyasFloat<TS>
    {
        if (a.SpanStarts is null) return null;
        var m = new double[(long)a.Lq * a.Lk];
        for (int i = 0; i < a.Lq; i++) for (int j = 0; j < a.Start(i); j++) m[(long)i * a.Lk + j] = -400;
        return StoreValues<TS>(m);
    }

    private static byte[] Slice(byte[] x, long offset, long length)
    {
        if (offset == 0 && length == x.Length) return x;
        var r = new byte[length];
        Array.Copy(x, offset, r, 0, length);
        return r;
    }

    private static readonly Dictionary<string, bool> s_backFusedFaster = [];

    /// <summary>The gate's override of the backward's measured choice (null: measured; true: the fused passes; false: the composition).</summary>
    internal static bool? FusedBackwardCpu;

    /// <summary>dQ, dK, dV: the fused rows and keys passes or the composition, whichever this processor measures faster at
    /// this shape and type (once, as the forward's rows); returns the lane's name.</summary>
    public static string AttentionBackward<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IQiyasFloat<TN>
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

    /// <summary>dQ, dK, dV: the composition in lane storage; returns the lane's name.</summary>
    public static string AttentionBackwardComposed<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) where TN : struct, IQiyasFloat<TN>
    {
        if (Wide<TN>())
        {
            BackwardIn<Float64>(a, ToLane<TN>(q, a.QValues), ToLane<TN>(k, a.KValues), ToLane<TN>(v, a.KValues), ToLane<TN>(dO, a.QValues), dq, dk, dv, TN.Name != "float64" ? (b, o, n) => FromLane<TN>(b, o, n) : null);
            return "f64";
        }

        BackwardIn<Float32>(a, ToLane<TN>(q, a.QValues), ToLane<TN>(k, a.KValues), ToLane<TN>(v, a.KValues), ToLane<TN>(dO, a.QValues), dq, dk, dv, TN.Name != "float32" ? (b, o, n) => FromLane<TN>(b, o, n) : null);
        return "f32";
    }

    private static void BackwardIn<TS>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv, Action<byte[], byte[], long>? narrow) where TS : struct, IQiyasFloat<TS>
    {
        int size = TS.Bits / 8;
        var ldq = narrow is null ? dq : new byte[a.QValues * size];
        var ldk = narrow is null ? dk : new byte[a.KValues * size];
        var ldv = narrow is null ? dv : new byte[a.KValues * size];
        var p = new byte[(long)a.Lq * a.Lk * size];
        var dp = new byte[p.Length];
        var limits = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) limits[i] = a.Limit(i) - 1;
        var softmax = RowOps.Find("scale_mask_softmax");
        var softmaxBwd = RowOps.Find("softmax_bwd");
        var scale = Qiyas.Ops.Find("scale");
        for (int h = 0; h < a.Heads; h++)
        {
            long qOff = (long)h * a.Lq * a.D, kOff = (long)h * a.Lk * a.D;
            var sq = Slice(q, qOff * size, a.Lq * a.D * size);
            var sk = Slice(k, kOff * size, (long)a.Lk * a.D * size);
            var sv = Slice(v, kOff * size, (long)a.Lk * a.D * size);
            var sdo = Slice(dO, qOff * size, a.Lq * a.D * size);
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), sq, sk, p);
            RowOp<TS>(softmax, p, SpanMask<TS>(a), null, null, null, limits, p, null, a.Lq, a.Lk, a.Scale, 1);
            var hdv = new byte[(long)a.Lk * a.D * size];
            MatMulView<TS>(new ProductView(a.Lk, a.D, a.Lq, true, false), p, sdo, hdv);            // dV = Pᵀ·dO
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), sdo, sv, dp);             // dP = dO·Vᵀ
            RowOp<TS>(softmaxBwd, p, dp, null, null, null, null, dp, null, a.Lq, a.Lk, 0, 0);       // dS = P·(dP − Σ P·dP)
            MapFloat<TS>("scale", dp, null, null, dp, (int)((long)a.Lq * a.Lk), a.Scale, 0);         // · scale
            var hdq = new byte[a.Lq * a.D * size];
            var hdk = new byte[(long)a.Lk * a.D * size];
            MatMulView<TS>(new ProductView(a.Lq, a.D, a.Lk, false, false), dp, sk, hdq);            // dQ = dS·K
            MatMulView<TS>(new ProductView(a.Lk, a.D, a.Lq, true, false), dp, sq, hdk);             // dK = dSᵀ·Q
            Array.Copy(hdq, 0, ldq, qOff * size, hdq.Length);
            Array.Copy(hdk, 0, ldk, kOff * size, hdk.Length);
            Array.Copy(hdv, 0, ldv, kOff * size, hdv.Length);
        }

        if (narrow is not null) { narrow(ldq, dq, a.QValues); narrow(ldk, dk, a.KValues); narrow(ldv, dv, a.KValues); }
    }

    // ---------------------------------------------------------------- decode: one query at a time, K and V read once

    /// <summary>O for a few queries against a long cache, each query on its own: scores, softmax and the context in the lane.</summary>
    public static string AttentionDecode<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IQiyasFloat<TN>
    {
        var vis = new DecodeVisitor(a, q, k, v, o);
        TN.Float(ref vis);
        return vis.Lane;
    }

    private struct DecodeVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(a.Heads * a.Lq, RowChunk<TN>(a.Heads * a.Lq, a.D, 1), a.KValues * 2, new DecodeBody<TN, TLane, TL>(a, q, k, v, o));
        }
    }

    private struct DecodeBody<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
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
