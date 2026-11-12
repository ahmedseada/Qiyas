// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The scaled weight's products (Products.cs ScaledWeight) on every device against the float64 reference: codes of several
// types (int8, uint8 and uint2 with zero points, int4, float8, float16) under each layout (a scale for the tensor, one a
// row, groups of a row's columns, blocks of rows and columns), y = x · Wᵀ with and without an epilogue, every float type
// of x and y (the block formats among them); and the weight's expansion. Within the lane's rounding of each value's
// terms (each weight's value two roundings: the zero's subtraction and the scale's product) plus the type's step.
namespace Qiyas;

internal static partial class Checks
{
    private sealed record ScaledCase(string Name, string Codes, (double Lo, double Hi) CodeRange, (double Lo, double Hi) ScaleRange, (double Lo, double Hi)? ZeroRange,
        Func<int, int, ScaledWeight> Layout, ProductEpilogue Epilogue, bool Bias = false, bool Residual = false);

    private static readonly ScaledCase[] ScaledCases =
    [
        new("int8 a scale a row, bias gelu, residual", "int8", (-127, 127), (0.002, 0.01), null, ScaledWeight.PerRow, new ProductEpilogue { Act = "gelu" }, true, true),
        new("uint8 groups of 32 with zeros, rank 4", "uint8", (0, 255), (0.002, 0.01), (120, 136), (n, k) => ScaledWeight.Grouped(n, k, 32), new ProductEpilogue { Rank = 4 }),
        new("int4 groups of 16 with zeros", "int4", (-8, 7), (0.05, 0.15), (-1, 1), (n, k) => ScaledWeight.Grouped(n, k, 16), new ProductEpilogue()),
        new("float8_e4m3 a scale for the tensor", "float8_e4m3", (-4, 4), (0.1, 0.3), null, ScaledWeight.PerTensor, new ProductEpilogue()),
        new("float16 blocks of 2 rows x 24 with zeros", "float16", (-1, 1), (0.5, 1.5), (-0.25, 0.25), (n, k) => new ScaledWeight(n, k, 2, 24), new ProductEpilogue()),
        new("uint2 a scale a row with zeros, beta 0.5", "uint2", (0, 3), (0.2, 0.5), (1, 2), ScaledWeight.PerRow, new ProductEpilogue { Beta = 0.5 }),
    ];

    public static bool Scaled(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        const int m = 13, n = 37, k = 72;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            int seed = 90;
            foreach (var c in ScaledCases)
            {
                var w = c.Layout(n, k);
                var e = c.Epilogue;
                var v = w.View(m);
                int r = e.Rank, count = (int)w.Scales;
                byte[] codes = Bench.Registry.Make(c.Codes, (int)w.Values, seed++, c.CodeRange.Lo, c.CodeRange.Hi);
                byte[] scales = Bench.Registry.Make(type, count, seed++, c.ScaleRange.Lo, c.ScaleRange.Hi);
                byte[]? zeros = c.ZeroRange is { } zr ? Bench.Registry.Make(type, count, seed++, zr.Lo, zr.Hi) : null;
                byte[] Draw(long values) => Bench.Registry.Make(type, (int)values, seed++, -1, 1);
                double[] Read(byte[]? x, long values) => x is null ? [] : Bench.Registry.Values(type, x, (int)values);
                byte[] x = Draw((long)m * k), y0 = Draw(v.ValuesC);
                byte[]? bias = c.Bias ? Draw(n) : null, residual = c.Residual ? Draw(v.ValuesC) : null;
                byte[]? u = r > 0 ? Draw((long)m * r) : null, lv = r > 0 ? Draw((long)r * n) : null;
                double[] cv = Bench.Registry.Values(c.Codes, codes, (int)w.Values), sv = Read(scales, count), xv = Read(x, (long)m * k), yv = Read(y0, v.ValuesC);
                double[]? zv = zeros is null ? null : Read(zeros, count), biasV = bias is null ? null : Read(bias, n), resV = residual is null ? null : Read(residual, v.ValuesC);
                double[]? uv = u is null ? null : Read(u, (long)m * r), lvv = lv is null ? null : Read(lv, (long)r * n);
                var wv = w.Reference(cv, sv, zv);

                // The expansion: each value within its two roundings (the zero's subtraction, the scale's product).
                var expanded = new byte[Bench.Registry.Bytes(type, w.Values)];
                Bench.Registry.WithFloat(type, new ScaledRun(ops, c.Codes, w, m, e, null, codes, scales, zeros, expanded, default));
                var termsW = new double[w.Values];
                for (int i = 0; i < termsW.Length; i++)
                {
                    int at = w.ScaleAt(i / k, i % k);
                    termsW[i] = (Math.Abs(cv[i]) + (zv is null ? 0 : Math.Abs(zv[at]))) * Math.Abs(sv[at]);
                }

                double bad = Excess(type, $"expand scaled {c.Name}", expanded, wv, i => 2 * termsW[i], 1);
                if (bad > 0) { Console.WriteLine($"check expand scaled {c.Name,-44} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }

                // The product and its epilogue.
                var (want, sums) = e.Reference(v, xv, wv, yv, biasV, resV, null, uv, lvv);
                double[] Abs(double[]? a) => a is null ? [] : a.Select(Math.Abs).ToArray();
                var (terms, _) = (e with { Act = null, Grad = null, Beta = Math.Abs(e.Beta) }).Reference(v, Abs(xv), termsW, Abs(yv), biasV is null ? null : Abs(biasV), null, null,
                    uv is null ? null : Abs(uv), lvv is null ? null : Abs(lvv));
                int formulaUlps = e.Act is not null ? Ops.Find(e.Act).Ulps : 0;
                double depth = k + r + 6;
                var y = y0.ToArray();
                var t = new ProductTerms<byte[]>(bias, residual, null, u, lv);
                Bench.Registry.WithFloat(type, new ScaledRun(ops, c.Codes, w, m, e, x, codes, scales, zeros, y, t));
                bad = Excess(type, $"matmul scaled {c.Name}", y, want,
                    i => 1.25 * depth * terms[i] + formulaUlps * (Math.Abs(want[i]) + Math.Abs(sums[i])) + 2 * (resV is null ? 0 : Math.Abs(resV[i])), 1);
                if (bad > 0) { Console.WriteLine($"check matmul scaled {c.Name,-44} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
            }
        }

        if (ok) Console.WriteLine($"check matmul scaled          {where,-6} every float type{(blocks ? "" : " but the block formats")}: x times a weight of int8, uint8, int4, uint2, float8 or float16 codes with scales and zero points (for the tensor, a row, groups of a row, blocks of rows and columns), with an epilogue, and the weight's expansion, within the lane's rounding");
        return ok;
    }

    // X null: the weight's expansion into Y; else Y = the epilogue's y of X · Wᵀ.
    private sealed record ScaledRun(IDevice<ulong, ulong>? Ops, string CodesType, ScaledWeight W, int M, ProductEpilogue E, byte[]? X, byte[] CodesBytes, byte[] Scales, byte[]? Zeros,
        byte[] Y, ProductTerms<byte[]> T) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CodesType, new Codes<TN>(this));

        private sealed class Codes<TN>(ScaledRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TW>() where TW : struct, IAdad<TW, float>
            {
                if (r.Ops is null)
                    return r.X is null
                        ? CpuDevice.Device.ExpandScaled<TN, TW>(r.W, r.CodesBytes, r.Scales, r.Zeros, r.Y)
                        : CpuDevice.Device.MatMulScaled<TN, TW>(r.W, r.M, r.E, r.X, r.CodesBytes, r.Scales, r.Zeros, r.Y, r.T);
                using var st = new Staged(r.Ops);
                string lane = r.X is null
                    ? r.Ops.ExpandScaled<TN, TW>(r.W, st.In(r.CodesBytes), st.In(r.Scales), st.In(r.Zeros), st.Out(r.Y))
                    : r.Ops.MatMulScaled<TN, TW>(r.W, r.M, r.E, st.In(r.X), st.In(r.CodesBytes), st.In(r.Scales), st.In(r.Zeros), st.Out(r.Y),
                        new ProductTerms<ulong>(st.In(r.T.Bias), st.In(r.T.Residual), st.In(r.T.Aux), st.In(r.T.U), st.In(r.T.V)));
                st.Collect();
                return lane;
            }
        }
    }
}
