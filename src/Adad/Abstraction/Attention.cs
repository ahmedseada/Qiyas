// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Attention: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Attention.cs; Cuda/GpuOps.Attention.cs with its kernels in Cuda/Ptx.Attention*.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>Q (heads × Lq × d), K and V (KvHeads × Lk × d), O (heads × Lq × d); the scale; causal or spans per query.</summary>
public sealed record AttentionArgs(int Heads, int Lq, int Lk, int D, double Scale, bool Causal, int[]? Spans = null, int[]? SpanStarts = null)
{
    /// <summary>The heads K and V hold (0: Heads): query head h reads K and V's head h / Group (grouped queries; 1 head: multi-query).</summary>
    public int KvHeads { get; init; }

    /// <summary>A sliding window (0: none): query i sees at most the Window keys below its limit, [Limit(i) − Window, Limit(i)).</summary>
    public int Window { get; init; }

    /// <summary>A cap on the scores (0: none): s = Softcap·tanh(Scale·q·k / Softcap) in place of Scale·q·k.</summary>
    public double Softcap { get; init; }

    public int KvHeadCount => KvHeads == 0 ? Heads : KvHeads;

    /// <summary>The query heads that share a head of K and V.</summary>
    public int Group => KvHeadCount == 0 ? 1 : Heads / KvHeadCount;

    public int KvHead(int h) => h / Group;

    public long QValues => (long)Heads * Lq * D;
    public long KValues => (long)KvHeadCount * Lk * D;

    /// <summary>Whether some query's first key may be past 0 (span starts or a window).</summary>
    public bool HasStarts => SpanStarts is not null || Window > 0;

    /// <summary>The keys query i sees: [Start(i), Limit(i)).</summary>
    public int Limit(int i) => Spans is not null ? Math.Min(Lk, Spans[i]) : Causal ? Math.Min(Lk, i + (Lk - Lq) + 1) : Lk;

    /// <summary>The first key query i sees: its span start (0 unless given), past the window's when there is one.</summary>
    public int Start(int i)
    {
        int s = SpanStarts is null ? 0 : Math.Clamp(SpanStarts[i], 0, Lk);
        return Window > 0 ? Math.Clamp(Math.Max(s, Limit(i) - Window), 0, Lk) : s;
    }

    public void Validate()
    {
        if (Heads < 0 || Lq < 0 || Lk < 0 || D < 0) throw new ArgumentException($"attention: {Heads} heads of {Lq} × {Lk} at d {D}");
        if (KvHeads < 0 || (KvHeads > 0 && Heads % KvHeads != 0)) throw new ArgumentException($"attention: {Heads} query heads do not group over {KvHeads} key and value heads");
        if (Window < 0) throw new ArgumentException($"attention: a window of {Window} keys");
        if (!(Softcap >= 0) || double.IsInfinity(Softcap)) throw new ArgumentException($"attention: a score cap of {Softcap}");
    }

    /// <summary>The same attention over KvHeadCount heads of Group·Lq query rows each: a group's query heads are adjacent in
    /// Q and O, so row r of head g is query r % Lq of head g·Group + r / Lq, every row of head g reading K and V's head g
    /// (no value moved); each row's keys as its span and start. Itself where every query head has its own K and V.</summary>
    public AttentionArgs Grouped()
    {
        Validate();
        if (Group == 1) return this;
        int rows = Group * Lq;
        var spans = new int[rows];
        int[]? starts = HasStarts ? new int[rows] : null;
        for (int r = 0; r < rows; r++)
        {
            spans[r] = Limit(r % Lq);
            if (starts is not null) starts[r] = Start(r % Lq);
        }

        return new AttentionArgs(KvHeadCount, rows, Lk, D, Scale, false, spans, starts) { Softcap = Softcap };
    }

    // A score from its product q·k: scaled, capped when there is a cap.
    private double Score(double dot) => Softcap > 0 ? Softcap * Math.Tanh(Scale * dot / Softcap) : Scale * dot;

    // The score's slope in q·k: Scale, times sech² of the cap's argument when there is a cap.
    private double Slope(double dot) => Softcap > 0 ? Scale * Funcs.SechSquaredReference(Scale * dot / Softcap) : Scale;

    /// <summary>The float64 reference of O, of P when asked (heads × Lq × Lk), and of each query's log-sum-exp of its scores
    /// when asked (heads × Lq; −∞ for a query that sees no key).</summary>
    public double[] Reference(double[] q, double[] k, double[] v, double[]? p = null, double[]? lse = null)
    {
        var o = new double[QValues];
        Parallel.For(0, Heads * Lq, hi =>
        {
            int h = hi / Lq, i = hi % Lq, limit = Limit(i), start = Start(i);
            long kv = (long)KvHead(h) * Lk;
            var s = new double[Lk];
            double m = double.NegativeInfinity;
            for (int j = start; j < limit; j++)
            {
                double dot = 0;
                for (int t = 0; t < D; t++) dot += q[((long)h * Lq + i) * D + t] * k[(kv + j) * D + t];
                s[j] = Score(dot);
                m = Math.Max(m, s[j]);
            }

            double sum = 0;
            for (int j = start; j < limit; j++) sum += s[j] = Math.Exp(s[j] - m);
            if (lse is not null) lse[(long)h * Lq + i] = sum > 0 ? m + Math.Log(sum) : double.NegativeInfinity;
            for (int j = start; j < limit; j++) s[j] = sum > 0 ? s[j] / sum : 0;
            for (int j = 0; j < Math.Min(start, Lk); j++) s[j] = 0;
            for (int j = limit; j < Lk; j++) s[j] = 0;
            if (p is not null) Array.Copy(s, 0, p, ((long)h * Lq + i) * Lk, Lk);
            for (int t = 0; t < D; t++)
            {
                double acc = 0;
                for (int j = start; j < limit; j++) acc += s[j] * v[(kv + j) * D + t];
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
            long kv = (long)KvHead(h) * Lk;
            for (int i = 0; i < Lq; i++)
            {
                long qi = ((long)h * Lq + i) * D;
                // dP_ij = dO_i · V_j; dS_ij = P_ij (dP_ij − Σ_j P_ij dP_ij) times the score's slope (a group's heads summed
                // into their K and V's gradients)
                var dp = new double[Lk];
                double dot = 0;
                for (int j = 0; j < Lk; j++)
                {
                    long kj = (kv + j) * D;
                    double d = 0;
                    for (int t = 0; t < D; t++) d += dO[qi + t] * v[kj + t];
                    dp[j] = d;
                    dot += p[((long)h * Lq + i) * Lk + j] * d;
                }

                for (int j = 0; j < Lk; j++)
                {
                    long kj = (kv + j) * D;
                    double pij = p[((long)h * Lq + i) * Lk + j];
                    if (pij == 0) continue;
                    double qk = 0;
                    if (Softcap > 0) for (int t = 0; t < D; t++) qk += q[qi + t] * k[kj + t];
                    double ds = pij * (dp[j] - dot) * Slope(qk);
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
