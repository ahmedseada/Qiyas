// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The float64 operations that take the sliced product (GpuOps.MatMulSliced) beside MatMul, each forced onto it (the checks'
// SlicedChoice: their small shapes would measure the plain lane faster) and held, as Checks.Sliced holds the product, to
// float64 math accumulated in double-double on the CPU within an ulp of every value: a batched view, a product split in k
// (few tiles, a long k: conv_bwd's dw) and a view of Aᵀ
// (MatMulView: a batch's product, conv_bwd's dcols and dw), the epilogue's sums and its second product's (MatMulEpilogue
// with no activation, no bias, no residual: y the sums; the second alone over an A of zeros), and every expert's product (ExpertMatMul without a bias, the
// experts' rows uneven, one count not a multiple of 8). Only where the device proved the sliced product (SlicedProven).
using Adad.Abstraction;
using Adad.Cuda;

namespace Qiyas;

internal static partial class Checks
{
    private static bool SlicedOps(GpuOps ops)
    {
        if (!ops.SlicedProven) return true;
        var random = new Random(29);
        double[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextDouble() * 2 - 1)];
        var gpu = ops.Gpu;
        bool ok = true;
        ops.SlicedChoice = true;
        try
        {
            // C[i][j] of A (element (i, p) at a[ai(i, p)]) times B (element (p, j) at b[bi(p, j)]) in double-double.
            static double Exact(double[] a, double[] b, int k, Func<int, int> ai, Func<int, int> bi)
            {
                double hi = 0, lo = 0;
                for (int p = 0; p < k; p++)
                {
                    double x = a[ai(p)], y = b[bi(p)], ph = x * y, pl = Math.FusedMultiplyAdd(x, y, -ph);
                    double sum = hi + ph, bv = sum - hi, err = (hi - (sum - bv)) + (ph - bv);
                    hi = sum;
                    lo += err + pl;
                }

                return hi + lo;
            }

            double Worst(double[] got, Func<int, double> want)
            {
                double worst = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    double reference = want(i), ulp = reference == 0 ? double.Epsilon : Math.BitIncrement(Math.Abs(reference)) - Math.Abs(reference);
                    worst = Math.Max(worst, Steps(Gap(got[i], reference), ulp));
                }

                return worst;
            }

            double[] Run(double[] a, double[] b, int outs, Action<ulong, ulong, ulong> op)
            {
                ulong da = gpu.Rent(a.Length * 8L), db = gpu.Rent(b.Length * 8L), dc = gpu.Rent(outs * 8L);
                Adad.Cuda.Gpu.Upload<double>(a, da);
                Adad.Cuda.Gpu.Upload<double>(b, db);
                op(da, db, dc);
                Adad.Cuda.Gpu.Synchronize();
                var got = new double[outs];
                Adad.Cuda.Gpu.Download<double>(dc, got);
                gpu.Return(da); gpu.Return(db); gpu.Return(dc);
                return got;
            }

            void Say(string name, int values, double worst)
            {
                Console.WriteLine($"check {name,-36} {values,9} values: worst {worst:0.000} ulp{(worst <= 1 ? "" : "  DIFFERS: too far")}");
                ok &= worst <= 1;
            }

            {
                // A batched view: 3 batches of 40 × 48 × 96, each its own operands.
                const int B = 3, M = 40, N = 48, K = 96;
                double[] a = Values(B * M * K), b = Values(B * K * N);
                var v = new ProductView(M, N, K, false, false, B, M * K, K * N, M * N);
                var got = Run(a, b, B * M * N, (da, db, dc) => ops.MatMulView<Float64>(v, da, db, dc));
                Say("sliced float64 batched view", B * M * N, Worst(got, at =>
                {
                    int z = at / (M * N), i = at % (M * N) / N, j = at % N;
                    return Exact(a, b, K, p => z * M * K + i * K + p, p => z * K * N + p * N + j);
                }));
            }

            {
                // A product of few tiles and a long k (conv_bwd's dw's shape in small: 16 × 24 × 4096), its diagonals split in k.
                const int M = 16, N = 24, K = 4096;
                double[] a = Values(M * K), b = Values(K * N);
                var v = new ProductView(M, N, K, false, false);
                var got = Run(a, b, M * N, (da, db, dc) => ops.MatMulView<Float64>(v, da, db, dc));
                Say("sliced float64 split in k", M * N, Worst(got, at => Exact(a, b, K, p => at / N * K + p, p => p * N + at % N)));
            }

            {
                // A view of Aᵀ (A stored k × m: conv_bwd's dcols = wᵀ·dy), 56 × 40 × 72.
                const int M = 56, N = 40, K = 72;
                double[] a = Values(K * M), b = Values(K * N);
                var v = new ProductView(M, N, K, true, false);
                var got = Run(a, b, M * N, (da, db, dc) => ops.MatMulView<Float64>(v, da, db, dc));
                Say("sliced float64 view of Aᵀ", M * N, Worst(got, at => Exact(a, b, K, p => p * M + at / N, p => p * N + at % N)));
            }

            {
                // The epilogue's sums (no activation, bias or residual: y the sums), then its second product's alone (A all
                // zeros: y = 0 + the second's sums).
                const int M = 48, N = 40, K = 80, K2 = 16;
                double[] a = Values(M * K), b = Values(K * N), a2 = Values(M * K2), b2 = Values(K2 * N);
                var v = new ProductView(M, N, K, false, false);
                var got = Run(a, b, M * N, (da, db, dc) => ops.MatMulEpilogue<Float64>(v, new ProductEpilogue(), da, db, dc, default));
                Say("sliced float64 epilogue sums", M * N, Worst(got, at => Exact(a, b, K, p => at / N * K + p, p => p * N + at % N)));
                var e = new ProductEpilogue { Second = new ProductView(M, N, K2, false, false) };
                ulong d2a = gpu.Rent(a2.Length * 8L), d2b = gpu.Rent(b2.Length * 8L);
                Adad.Cuda.Gpu.Upload<double>(a2, d2a);
                Adad.Cuda.Gpu.Upload<double>(b2, d2b);
                got = Run(new double[M * K], b, M * N, (da, db, dc) => ops.MatMulEpilogue<Float64>(v, e, da, db, dc, new ProductTerms<ulong>(SecondA: d2a, SecondB: d2b)));
                gpu.Return(d2a); gpu.Return(d2b);
                Say("sliced float64 epilogue second sums", M * N, Worst(got, at => Exact(a2, b2, K2, p => at / N * K2 + p, p => p * N + at % N)));
            }

            {
                // Every expert's product: 3 experts over 61 rows (21, 0, 40: one count not a multiple of 8, one expert empty).
                const int E = 3, Rows = 61, K = 64, N = 48;
                int[] off = [0, 21, 21, 61];
                double[] x = Values(Rows * K), w = Values(E * K * N);
                var v = new ExpertView(E, Rows, K, N);
                ulong doff = gpu.Rent(off.Length * 4L);
                Adad.Cuda.Gpu.Upload<int>(off, doff);
                var got = Run(x, w, Rows * N, (dx, dw, dy) => ops.ExpertMatMul<Float64>(v, dx, doff, dw, 0, dy));
                gpu.Return(doff);
                Say("sliced float64 experts", Rows * N, Worst(got, at =>
                {
                    int r = at / N, j = at % N, g = r < 21 ? 0 : 2;
                    return Exact(x, w, K, p => r * K + p, p => g * K * N + p * N + j);
                }));
            }
        }
        finally
        {
            ops.SlicedChoice = null;
        }

        return ok;
    }
}
