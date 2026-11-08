// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Products: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Products.cs; Cuda/GpuOps.Products.cs with its kernels in Cuda/Ptx.cs.

using System.Buffers;

namespace Adad.Abstraction;

/// <summary>A product on views: A stored m × k (k × m when TransA), B stored k × n (n × k when TransB), C m × n;
/// `Batch` products at strides StrideA, StrideB, StrideC (in values; 0 broadcasts the operand across the batch).</summary>
public sealed record ProductView(int M, int N, int K, bool TransA, bool TransB, int Batch = 1, long StrideA = 0, long StrideB = 0, long StrideC = 0)
{
    public long StrideAOrDefault => Batch > 1 ? StrideA : 0;
    public long StrideBOrDefault => Batch > 1 ? StrideB : 0;
    public long StrideCOrDefault => Batch > 1 ? StrideC : (long)M * N;

    /// <summary>The strided view that makes operand A of batch i contiguous (m × k).</summary>
    public int[] ViewA(int i) => TransA ? [1, 1, M, K, 0, 0, 1, M, (int)(i * StrideAOrDefault)] : [1, 1, M, K, 0, 0, K, 1, (int)(i * StrideAOrDefault)];

    /// <summary>The strided view that makes operand B of batch i contiguous (k × n).</summary>
    public int[] ViewB(int i) => TransB ? [1, 1, K, N, 0, 0, 1, K, (int)(i * StrideBOrDefault)] : [1, 1, K, N, 0, 0, N, 1, (int)(i * StrideBOrDefault)];

    public long ValuesA => (TransA ? (long)K * M : (long)M * K) + (Batch - 1) * StrideAOrDefault;
    public long ValuesB => (TransB ? (long)N * K : (long)K * N) + (Batch - 1) * StrideBOrDefault;
    public long ValuesC => (long)M * N + (Batch - 1) * StrideCOrDefault;

    /// <summary>The float64 reference: every batch's product of the operands as viewed.</summary>
    public double[] Reference(double[] a, double[] b)
    {
        var c = new double[ValuesC];
        for (int i = 0; i < Batch; i++)
        {
            var ai = new double[M * K];
            var bi = new double[K * N];
            long baseA = i * StrideAOrDefault, baseB = i * StrideBOrDefault;
            for (int r = 0; r < M; r++) for (int q = 0; q < K; q++) ai[r * K + q] = TransA ? a[baseA + q * M + r] : a[baseA + r * K + q];
            for (int q = 0; q < K; q++) for (int s = 0; s < N; s++) bi[q * N + s] = TransB ? b[baseB + s * K + q] : b[baseB + q * N + s];
            var ci = Reference(ai, bi, M, N, K);
            Array.Copy(ci, 0, c, i * StrideCOrDefault, M * N);
        }

        return c;
    }

    /// <summary>The float64 reference product (A m × k, B k × n), accumulated in double-double (each product exact by a fused
    /// multiply-add, each sum with its error term), so a lane that is exact reads 0 and a float64 lane shows its own rounding.</summary>
    public static double[] Reference(double[] a, double[] b, int m, int n, int k)
    {
        var c = new double[m * n];
        Parallel.For(0, m, i =>
        {
            var hi = new double[n];
            var lo = new double[n];
            for (int p = 0; p < k; p++)
            {
                double x = a[i * k + p];
                var bp = b.AsSpan(p * n, n);
                for (int j = 0; j < n; j++)
                {
                    double y = bp[j], ph = x * y, pl = Math.FusedMultiplyAdd(x, y, -ph);
                    double h = hi[j], sum = h + ph, bv = sum - h, err = (h - (sum - bv)) + (ph - bv);
                    hi[j] = sum;
                    lo[j] += err + pl;
                }
            }

            var row = c.AsSpan(i * n, n);
            for (int j = 0; j < n; j++) row[j] = hi[j] + lo[j];
        });
        return c;
    }
}

/// <summary>What a product's store does with each of its sums s = op(A)·op(B) (the float lane's, every term below added in
/// it) before y is rounded to the type once: s + U·op(V) (the low-rank term, a LoRA adapter's: U m × Rank, V Rank × n, or
/// n × Rank with TransV; one batch), + bias[col] (n values), + β·y (y's value before, read where it is stored; β 0: y is
/// not read); then y = act(s), and aux = s (the activation's input stored as the type) where an aux buffer is given; or,
/// with Grad, y = grad(aux, s) (the gradient through the activation whose inputs aux holds, s the incoming gradient:
/// gelu_bwd: a gelu's backward through the product before it); then y + residual (y's layout). The batches' C contiguous
/// (StrideC m·n).</summary>
public sealed record ProductEpilogue
{
    /// <summary>y's value before, times β, added to the sums (0: y not read).</summary>
    public double Beta { get; init; }

    /// <summary>A one-input formula (Map.cs) applied to the sums, with ActA its parameter; null none.</summary>
    public string? Act { get; init; }

    public double ActA { get; init; }

    /// <summary>A two-input backward formula grad(x, dy) (Map.cs: gelu_bwd, silu_bwd, …) taken as grad(aux, s); null none.</summary>
    public string? Grad { get; init; }

    /// <summary>The low-rank term's rank (0: none).</summary>
    public int Rank { get; init; }

    /// <summary>V stored n × Rank (Vᵀ) rather than Rank × n.</summary>
    public bool TransV { get; init; }

    /// <summary>The terms the store takes for the buffers given (low: the low-rank term's sums).</summary>
    public StoreTerms Terms(bool bias, bool residual, bool aux) => new(Rank > 0, bias, Beta != 0, Act, Grad, residual) { Aux = aux && Act is not null };

    public void Validate(ProductView v, bool bias, bool residual, bool aux, bool u, bool lowV)
    {
        if (v.Batch > 1 && v.StrideCOrDefault != (long)v.M * v.N) throw new ArgumentException("matmul epilogue: the batches' C must be contiguous (StrideC m·n)");
        if (v.ValuesC > int.MaxValue) throw new ArgumentException("matmul epilogue: C past 2^31 values");
        if (Act is not null && Grad is not null) throw new ArgumentException("matmul epilogue: an activation or a gradient, not both");
        if (Act is not null && Ops.Find(Act).Inputs != 1) throw new ArgumentException($"matmul epilogue: the activation {Act} is not a one-input formula");
        if (Grad is not null && Ops.Find(Grad).Inputs != 2) throw new ArgumentException($"matmul epilogue: the gradient {Grad} is not a two-input formula");
        if (Grad is not null && !aux) throw new ArgumentException("matmul epilogue: a gradient reads the activation's inputs (aux)");
        if (Rank < 0) throw new ArgumentException("matmul epilogue: a negative rank");
        if (Rank > 0 && (!u || !lowV)) throw new ArgumentException("matmul epilogue: the low-rank term takes U and V");
        if (Rank > 0 && v.Batch != 1) throw new ArgumentException("matmul epilogue: the low-rank term over one batch");
        if (Rank == 0 && (u || lowV)) throw new ArgumentException("matmul epilogue: U and V given at rank 0");
    }

    /// <summary>The float64 reference: y and the sums s it was made from (y0 y's values before; bias n values; residual and
    /// aux y's layout; u m × Rank; lowV Rank × n, or n × Rank with TransV).</summary>
    public (double[] Y, double[] S) Reference(ProductView v, double[] a, double[] b, double[] y0, double[]? bias, double[]? residual, double[]? aux, double[]? u, double[]? lowV)
    {
        var s = v.Reference(a, b);
        double[]? low = null;
        if (Rank > 0)
        {
            var vt = new double[Rank * v.N];
            for (int r = 0; r < Rank; r++) for (int j = 0; j < v.N; j++) vt[r * v.N + j] = TransV ? lowV![j * Rank + r] : lowV![r * v.N + j];
            low = ProductView.Reference(u!, vt, v.M, v.N, Rank);
        }

        var act = Act is null ? null : Ops.Find(Act);
        var grad = Grad is null ? null : Ops.Find(Grad);
        var y = new double[s.Length];
        for (int i = 0; i < s.Length; i++)
        {
            double t = s[i] + (low is null ? 0 : low[i]) + (bias is null ? 0 : bias[i % v.N]) + (Beta == 0 ? 0 : Beta * y0[i]);
            s[i] = t;
            double o = act is not null ? act.Reference(t, 0, 0, ActA, 0) : grad is not null ? grad.Reference(aux![i], t, 0, ActA, 0) : t;
            y[i] = o + (residual is null ? 0 : residual[i]);
        }

        return (y, s);
    }
}

/// <summary>The terms a product's store adds to its lane sums and what it applies (ProductEpilogue.Terms; the convolution's
/// store takes the bias and the activation): a store kernel's variant. Aux: the pass that writes aux = s before y.</summary>
public readonly record struct StoreTerms(bool Low, bool Bias, bool Beta, string? Act, string? Grad, bool Residual)
{
    public bool Aux { get; init; }

    /// <summary>The pass that stores the activation's input (s, no activation and no residual) into aux.</summary>
    public StoreTerms Input => this with { Act = null, Grad = null, Residual = false, Aux = false };

    public string Key => $"{(Low ? "l" : "")}{(Bias ? "b" : "")}{(Beta ? "c" : "")}_{(Grad is null ? Act ?? "none" : "d" + Grad)}{(Residual ? "_r" : "")}";
}

/// <summary>A product epilogue's operands (ProductEpilogue), each absent (null; 0 on a device) where its term is not taken.</summary>
public readonly record struct ProductTerms<TBuffer>(TBuffer? Bias = default, TBuffer? Residual = default, TBuffer? Aux = default, TBuffer? U = default, TBuffer? V = default)
{
    public static bool Given(TBuffer? x) => !EqualityComparer<TBuffer?>.Default.Equals(x, default);

    public StoreTerms Validate(ProductEpilogue e, ProductView v)
    {
        e.Validate(v, Given(Bias), Given(Residual), Given(Aux), Given(U), Given(V));
        return e.Terms(Given(Bias), Given(Residual), Given(Aux));
    }
}
