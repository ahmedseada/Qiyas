// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of the mixture of experts (IExperts): the routing's and the expert products' arguments and
// float64 references. The devices implement them in their folders: Cpu/Experts.cs; Cuda/GpuOps.Experts.cs with its
// kernels in Cuda/Ptx.Experts.cs (the products by the stored GEMM's grouped mode, Ptx.Gemm); Vulkan/VulkanOps.Experts.cs
// with Vulkan/Kernels.Experts.cs.
namespace Adad.Abstraction;

/// <summary>A mixture's expert products over rows grouped by expert: for expert e, rows offsets[e] .. offsets[e + 1] of x
/// (Rows × K) times W_e (K × N, or N × K stored with TransW; expert e's weights from value e·K·N), plus bias_e (row e of an
/// Experts × N bias). The offsets are RouteTopK's (Experts + 1 of them, read where they are: on a device, no copy to the
/// host); a row outside every expert's range is 0.</summary>
public sealed record ExpertView(int Experts, int Rows, int K, int N, bool TransW = false)
{
    public long XValues => (long)Rows * K;
    public long YValues => (long)Rows * N;
    public long WeightValues => (long)Experts * K * N;
    public long BiasValues => (long)Experts * N;

    public void Validate()
    {
        if (Experts < 1 || Rows < 0 || K < 0 || N < 0) throw new ArgumentException("experts: an extent below zero, or no experts");
        if (WeightValues > int.MaxValue || XValues > int.MaxValue || YValues > int.MaxValue) throw new ArgumentException("experts: past 2^31 values");
    }

    /// <summary>The float64 reference of y (bias: Experts × N, or none).</summary>
    public double[] Reference(double[] x, int[] offsets, double[] w, double[]? bias)
    {
        var y = new double[YValues];
        for (int e = 0; e < Experts; e++)
        {
            int r0 = offsets[e], r1 = offsets[e + 1];
            if (r1 <= r0) continue;
            var xe = x.AsSpan(r0 * K, (r1 - r0) * K).ToArray();
            var we = new double[(long)K * N];
            for (int q = 0; q < K; q++)
                for (int j = 0; j < N; j++)
                    we[q * N + j] = w[(long)e * K * N + (TransW ? (long)j * K + q : (long)q * N + j)];
            var ye = ProductView.Reference(xe, we, r1 - r0, N, K);
            for (int i = 0; i < ye.Length; i++) y[r0 * N + i] = ye[i] + (bias is null ? 0 : bias[e * N + i % N]);
        }

        return y;
    }
}

public static class Experts
{
    /// <summary>The routing's float64 reference (RouteTopK): each token's k first experts in TopK's order (Routing.Precedes), the
    /// expert of rank s at the token's slot s; the slots' experts and weights (normalize: over the sum of the token's k, added
    /// in slot order), and the offsets, tokenOfRow and slotRow of the slots listed expert by expert.</summary>
    public static (double[] Weights, int[] Chosen, int[] Offsets, int[] TokenOfRow, int[] SlotRow) RouteTopKReference(double[] p, int tokens, int experts, int k, bool normalize)
    {
        var weights = new double[tokens * k];
        var chosen = new int[tokens * k];
        for (int t = 0; t < tokens; t++)
        {
            var row = p.AsSpan(t * experts, experts);
            for (int e = 0; e < experts; e++)
            {
                int rank = 0;
                for (int q = 0; q < experts; q++) if (Routing.Precedes(row[q], q, row[e], e)) rank++;
                if (rank >= k) continue;
                chosen[t * k + rank] = e;
                weights[t * k + rank] = row[e];
            }

            double sum = 0;
            for (int i = 0; i < k; i++) sum += weights[t * k + i];
            if (normalize) for (int i = 0; i < k; i++) weights[t * k + i] /= sum;
        }

        var (_, offsets, order) = Routing.RouteReference(chosen, experts);
        var tokenOfRow = new int[order.Length];
        var slotRow = new int[order.Length];
        for (int r = 0; r < order.Length; r++)
        {
            tokenOfRow[r] = order[r] / k;
            slotRow[order[r]] = r;
        }

        return (weights, chosen, offsets, tokenOfRow, slotRow);
    }

    public static void ValidateRoute(int tokens, int experts, int k)
    {
        if (tokens < 0 || experts < 1 || k < 1 || k > experts) throw new ArgumentException($"route top-k: {k} of {experts} experts for {tokens} tokens");
        if ((long)tokens * Math.Max(k, experts) > int.MaxValue) throw new ArgumentException("route top-k: past 2^31 slots");
    }

    /// <summary>The gated pair's formula: a two-input formula of Map (gated_silu, gated_gelu, gated_relu, …) taken as gate(g, u).</summary>
    public static MapOp Gate(string gate)
    {
        var op = Ops.Find(gate);
        if (op.Inputs != 2) throw new ArgumentException($"experts: the gate {gate} is not a two-input formula");
        return op;
    }
}
