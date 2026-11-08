// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The product's epilogue (Products.cs ProductEpilogue) on every device against its float64 reference: a bias and an
// activation with its input kept (aux), β·C over transposed batches with a residual, the low-rank term (V and Vᵀ), the
// gradient through an activation from its kept inputs, batches sharing B; every float type (the block formats among them).
// Within the lane's rounding of each value's terms through the formula (its slope at most 1.25) and the formula's own,
// plus the type's step.
namespace Qiyas;

internal static partial class Checks
{
    private static readonly (string Name, ProductView View, ProductEpilogue Epilogue, bool Bias, bool Residual, bool Aux)[] EpilogueCases =
    [
        ("37x45x29 bias gelu, its input kept", new ProductView(37, 45, 29, false, false), new ProductEpilogue { Act = "gelu" }, true, false, true),
        ("3 batches A, B transposed 19x23x31, beta 0.5, residual", new ProductView(19, 23, 31, true, true, 3, 19 * 31, 31 * 23, 19 * 23), new ProductEpilogue { Beta = 0.5 }, false, true, false),
        ("33x40x27 B transposed, rank 8 V transposed, bias, beta -1", new ProductView(33, 40, 27, false, true), new ProductEpilogue { Rank = 8, TransV = true, Beta = -1 }, true, false, false),
        ("29x36x21 gelu's gradient", new ProductView(29, 36, 21, false, false), new ProductEpilogue { Grad = "gelu_bwd" }, false, false, true),
        ("2 batches sharing B 21x30x25, bias silu, residual", new ProductView(21, 30, 25, false, false, 2, 21 * 25, 0, 21 * 30), new ProductEpilogue { Act = "silu" }, true, true, false),
        ("40x33x35 A transposed, rank 4, relu", new ProductView(40, 33, 35, true, false), new ProductEpilogue { Rank = 4, Act = "relu" }, false, false, false),
        ("17x32x19 B transposed, silu's gradient, bias, residual", new ProductView(17, 32, 19, false, true), new ProductEpilogue { Grad = "silu_bwd" }, true, true, true),
    ];

    public static bool ProductEpilogues(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            int seed = 80;
            foreach (var (name, v, e, hasBias, hasResidual, hasAux) in EpilogueCases)
            {
                int n = (int)v.ValuesC, r = e.Rank;
                byte[] Draw(long count, double lo = -1, double hi = 1) => Bench.Registry.Make(type, (int)count, seed++, lo, hi);
                double[] Read(byte[]? x, long count) => x is null ? [] : Bench.Registry.Values(type, x, (int)count);
                byte[] a = Draw(v.ValuesA), b = Draw(v.ValuesB), y0 = Draw(n);
                byte[]? bias = hasBias ? Draw(v.N) : null, residual = hasResidual ? Draw(n) : null;
                byte[]? aux = !hasAux ? null : e.Grad is not null ? Draw(n, -3, 3) : new byte[Bench.Registry.Bytes(type, n)];
                byte[]? u = r > 0 ? Draw((long)v.M * r) : null, lv = r > 0 ? Draw((long)r * v.N) : null;
                double[] av = Read(a, v.ValuesA), bv = Read(b, v.ValuesB), yv = Read(y0, n), uv = Read(u, (long)v.M * r), lvv = Read(lv, (long)r * v.N);
                double[]? biasV = bias is null ? null : Read(bias, v.N), resV = residual is null ? null : Read(residual, n);
                double[]? auxV = e.Grad is null ? null : Read(aux, n);
                var (want, sums) = e.Reference(v, av, bv, yv, biasV, resV, auxV, r > 0 ? uv : null, r > 0 ? lvv : null);
                double[] Abs(double[]? x) => x is null ? [] : x.Select(Math.Abs).ToArray();
                var (terms, _) = (e with { Act = null, Grad = null, Beta = Math.Abs(e.Beta) }).Reference(v, Abs(av), Abs(bv), Abs(yv), biasV is null ? null : Abs(biasV), null, null, r > 0 ? Abs(uv) : null, r > 0 ? Abs(lvv) : null);
                int formulaUlps = e.Act is not null ? Ops.Find(e.Act).Ulps : e.Grad is not null ? Ops.Find(e.Grad).Ulps : 0;
                double depth = v.K + r + 4;
                var c = y0.ToArray();
                Bench.Registry.WithFloat(type, new EpilogueRun(ops, v, e, a, b, c, bias, residual, aux, u, lv, e.Act is not null && hasAux));
                // The sums' rounding through the formula (its slope at most 1.25), the formula's own, and the residual's sum.
                double worst = Excess(type, $"matmul epilogue {name}", c, want,
                    i => 1.25 * depth * terms[i] + formulaUlps * (Math.Abs(want[i]) + Math.Abs(sums[i])) + 2 * (resV is null ? 0 : Math.Abs(resV[i])), 1);
                if (worst > 0) { Console.WriteLine($"check matmul epilogue {name,-48} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
                if (e.Act is not null && hasAux)
                {
                    double bad = Excess(type, $"matmul epilogue aux {name}", aux!, sums, i => depth * terms[i], 1);
                    if (bad > 0) { Console.WriteLine($"check matmul epilogue aux {name,-44} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
                }
            }
        }

        if (ok) Console.WriteLine($"check matmul epilogue        {where,-6} every float type{(blocks ? "" : " but the block formats")}: a bias, activations with their inputs kept and the gradients through them, beta C, a residual, the low-rank term (V and its transpose), batches transposed and sharing B, within the products' rounding");
        return ok;
    }

    private sealed class EpilogueRun(IDevice<ulong, ulong>? ops, ProductView v, ProductEpilogue e, byte[] a, byte[] b, byte[] c, byte[]? bias, byte[]? residual, byte[]? aux, byte[]? u, byte[]? lowV, bool auxOut)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.MatMulEpilogue<TN>(v, e, a, b, c, new ProductTerms<byte[]>(bias, residual, aux, u, lowV));
            using var st = new Staged(ops);
            var lane = ops.MatMulEpilogue<TN>(v, e, st.In(a), st.In(b), st.Out(c), new ProductTerms<ulong>(st.In(bias), st.In(residual), auxOut ? st.Out(aux) : st.In(aux), st.In(u), st.In(lowV)));
            st.Collect();
            return lane;
        }
    }
}
