// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Attention: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Attention.cs; Cuda/GpuOps.Attention.cs with its kernels in Cuda/Ptx.Attention*.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

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
