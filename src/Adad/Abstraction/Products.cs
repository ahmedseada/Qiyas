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
