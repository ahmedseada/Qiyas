// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The Vulkan sliced float64 product (VulkanOps.MatMulSliced: the s8 matrix units' digits, CUDA's definition) against float64
// math accumulated in double-double on the CPU: the worst error in ulps of the result, over values of one scale and over
// rows and columns of scales 2^-20 .. 2^20, read through views (A by columns, B by rows of k, two batches at strides) at
// sizes off every tile and k off every slice; a row of zeros; a value that is not finite refused (nothing computed). Sets
// SlicedProven; then the products of every type under the sliced route wherever it is eligible (SlicedChoice), so
// MatMulView's float64 views take it at the gate's shapes.
using Adad.Vulkan;

namespace Qiyas;

internal static partial class Checks
{
    private static bool VulkanSliced(VulkanOps ops)
    {
        var vk = ops.Vk;
        if (vk.Cooperative("s8", "s32") is null)
        {
            Console.WriteLine($"check {"sliced float64 product",-36} not on this device (no s8 matrix units): never chosen");
            return true;
        }

        const int M = 96, N = 80, K = 1000;
        var random = new Random(23);
        bool ok = true;
        foreach (var (transA, transB, batches, scaled) in new[] { (false, false, 1, false), (false, false, 1, true), (true, false, 1, true), (false, true, 2, true), (true, true, 2, false) })
        {
            var v = new ProductView(M, N, K, transA, transB, batches, (long)M * K, (long)K * N, (long)M * N);
            var a = new double[batches * M * K];
            var b = new double[batches * K * N];
            int Ai(int z, int i, int p) => z * M * K + (transA ? p * M + i : i * K + p);
            int Bi(int z, int p, int j) => z * K * N + (transB ? j * K + p : p * N + j);
            for (int z = 0; z < batches; z++)
            {
                for (int i = 0; i < M; i++)
                {
                    double s = scaled ? Math.Pow(2, i % 41 - 20) : 1;
                    for (int p = 0; p < K; p++) a[Ai(z, i, p)] = i == 7 ? 0 : (random.NextDouble() * 2 - 1) * s;            // row 7 zeros
                }

                for (int p = 0; p < K; p++)
                    for (int j = 0; j < N; j++) b[Bi(z, p, j)] = (random.NextDouble() * 2 - 1) * (scaled ? Math.Pow(2, j % 37 - 18) : 1);
            }

            ulong da = vk.Rent(a.Length * 8L), db = vk.Rent(b.Length * 8L), dc = vk.Rent((long)batches * M * N * 8);
            vk.Upload<double>(a, da);
            vk.Upload<double>(b, db);
            var got = new double[batches * M * N];
            Array.Fill(got, double.NaN);
            vk.Upload<double>(got, dc);
            bool ran = ops.MatMulSliced(v, da, db, dc);
            if (ran) vk.Download<double>(dc, got);
            vk.Return(da); vk.Return(db); vk.Return(dc);
            string name = $"sliced float64{(transA ? " Aᵀ" : "")}{(transB ? " Bᵀ" : "")}{(batches > 1 ? " batched" : "")}{(scaled ? " scaled" : "")}";
            if (!ran)
            {
                Console.WriteLine($"check {name,-36} vulkan FAILS: refused finite values within the scale");
                ok = false;
                continue;
            }

            double worst = 0;
            int worstAt = 0;
            for (int z = 0; z < batches; z++)
            {
                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        double hi = 0, lo = 0;
                        for (int p = 0; p < K; p++)
                        {
                            double x = a[Ai(z, i, p)], y = b[Bi(z, p, j)];
                            double ph = x * y, pl = Math.FusedMultiplyAdd(x, y, -ph);
                            double sum = hi + ph, bv = sum - hi, err = (hi - (sum - bv)) + (ph - bv);
                            hi = sum;
                            lo += err + pl;
                        }

                        double reference = hi + lo, value = got[(z * M + i) * N + j];
                        double ulp = reference == 0 ? double.Epsilon : Math.BitIncrement(Math.Abs(reference)) - Math.Abs(reference);
                        double e = Steps(Gap(value, reference), ulp);
                        if (e > worst || double.IsNaN(e)) { worst = double.IsNaN(e) ? double.PositiveInfinity : e; worstAt = (z * M + i) * N + j; }
                    }
                }
            }

            Console.WriteLine($"check {name,-36} vulkan {batches * M * N,9} values: worst {worst:0.000} ulp at {worstAt}{(worst <= 1 ? "" : "  DIFFERS: too far")}");
            ok &= worst <= 1;
        }

        // A value that is not finite (an infinity in A, a NaN in B): refused, nothing computed.
        foreach (bool nanInB in new[] { false, true })
        {
            var v = new ProductView(8, 8, 64, false, false);
            var a = new double[8 * 64];
            var b = new double[64 * 8];
            Array.Fill(a, 1.0);
            Array.Fill(b, 1.0);
            if (nanInB) b[100] = double.NaN; else a[100] = double.PositiveInfinity;
            ulong da = vk.Rent(a.Length * 8L), db = vk.Rent(b.Length * 8L), dc = vk.Rent(64 * 8);
            vk.Upload<double>(a, da);
            vk.Upload<double>(b, db);
            bool ran = ops.MatMulSliced(v, da, db, dc);
            vk.Return(da); vk.Return(db); vk.Return(dc);
            if (ran) { Console.WriteLine($"check {"sliced float64 not finite",-36} vulkan FAILS: {(nanInB ? "a NaN" : "an infinity")} was not refused"); ok = false; }
        }

        ops.SlicedProven = ok;
        if (!ok) return false;
        // Every type's products with the sliced route wherever eligible (float64's views and batches through MatMulView).
        ops.SlicedChoice = true;
        try
        {
            ok &= ProductsWith(ops, "vulkan", 37, 29, 53, " [sliced float64]");
            ok &= ExpertsCheck(ops, "vulkan");                                       // the experts' products an expert a sliced product
        }
        finally { ops.SlicedChoice = null; }
        if (ok) Console.WriteLine($"check {"sliced float64 product",-36} vulkan proven: within an ulp of float64 math, through views and batches; an infinity and a NaN refused; every type's products and the experts' with it forced");
        return ok;
    }
}
