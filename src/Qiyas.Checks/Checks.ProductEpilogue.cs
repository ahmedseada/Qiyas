// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The product's epilogue (Products.cs ProductEpilogue) on every device against its float64 reference: a bias and an
// activation with its sums kept, β·C over transposed batches with a residual, a second product (B2 and B2ᵀ; over batches
// with A2 transposed and B2 shared, its sums kept with no activation), the gradient through an activation from its inputs
// (its sums kept too, β·C among them: the kept sums' pass reads y before), batches sharing B; every float type (the
// block formats among them), within the lane's rounding of each value's terms through the formula (its slope at most
// 1.25) and the formula's own, plus the type's step. Every other type through the any-type formulas (relu, clamp, square,
// affine and gradients), integer β (-1 among them: an unsigned type then sums in the float64 lane) and every term,
// exactly: y the reference held to the type's range once; and a case of values far past the small ones (twice the square
// root of the type's largest, at most 2^22): the 64-bit and float64 lanes, y saturating.
namespace Qiyas;

internal static partial class Checks
{
    private static readonly (string Name, ProductView View, ProductEpilogue Epilogue, bool Bias, bool Residual, bool Keep)[] EpilogueCases =
    [
        ("37x45x29 bias gelu, its sums kept", new ProductView(37, 45, 29, false, false), new ProductEpilogue { Act = "gelu" }, true, false, true),
        ("3 batches A, B transposed 19x23x31, beta 0.5, residual", new ProductView(19, 23, 31, true, true, 3, 19 * 31, 31 * 23, 19 * 23), new ProductEpilogue { Beta = 0.5 }, false, true, false),
        ("33x40x27 B transposed, a second 33x40x8 B2 transposed, bias, beta -1", new ProductView(33, 40, 27, false, true),
            new ProductEpilogue { Second = new ProductView(33, 40, 8, false, true), Beta = -1 }, true, false, false),
        ("29x36x21 gelu's gradient", new ProductView(29, 36, 21, false, false), new ProductEpilogue { Grad = "gelu_bwd" }, false, false, false),
        ("2 batches sharing B 21x30x25, bias silu, residual", new ProductView(21, 30, 25, false, false, 2, 21 * 25, 0, 21 * 30), new ProductEpilogue { Act = "silu" }, true, true, false),
        ("40x33x35 A transposed, a second 40x33x4, relu", new ProductView(40, 33, 35, true, false), new ProductEpilogue { Second = new ProductView(40, 33, 4, false, false), Act = "relu" }, false, false, false),
        ("17x32x19 B transposed, silu's gradient, bias, beta 0.5, residual, its sums kept", new ProductView(17, 32, 19, false, true), new ProductEpilogue { Grad = "silu_bwd", Beta = 0.5 }, true, true, true),
        ("3 batches 19x23x31, a second over the batches (A2 transposed, B2 shared), bias, its sums kept", new ProductView(19, 23, 31, false, false, 3, 19 * 31, 31 * 23, 19 * 23),
            new ProductEpilogue { Second = new ProductView(19, 23, 6, true, false, 3, 6 * 19, 0, 19 * 23) }, true, false, true),
    ];

    private static readonly (string Name, ProductView View, ProductEpilogue Epilogue, bool Bias, bool Residual, bool Keep, bool Wide)[] IntegerEpilogueCases =
    [
        ("37x45x29 bias relu, its sums kept", new ProductView(37, 45, 29, false, false), new ProductEpilogue { Act = "relu" }, true, false, true, false),
        ("3 batches A, B transposed 19x23x31, beta 2, residual", new ProductView(19, 23, 31, true, true, 3, 19 * 31, 31 * 23, 19 * 23), new ProductEpilogue { Beta = 2 }, false, true, false, false),
        ("33x40x27 B transposed, a second 33x40x8 B2 transposed, bias, beta -1", new ProductView(33, 40, 27, false, true),
            new ProductEpilogue { Second = new ProductView(33, 40, 8, false, true), Beta = -1 }, true, false, false, false),
        ("29x36x21 relu's gradient", new ProductView(29, 36, 21, false, false), new ProductEpilogue { Grad = "relu_bwd" }, false, false, false, false),
        ("2 batches sharing B 21x30x25, bias, clamp to -9..40, residual", new ProductView(21, 30, 25, false, false, 2, 21 * 25, 0, 21 * 30),
            new ProductEpilogue { Act = "clamp", ActA = -9, ActB = 40 }, true, true, false, false),
        ("40x33x35 A transposed, a second 40x33x4, square", new ProductView(40, 33, 35, true, false), new ProductEpilogue { Second = new ProductView(40, 33, 4, false, false), Act = "square" }, false, false, false, false),
        ("17x32x19 B transposed, square's gradient, bias, residual, its sums kept", new ProductView(17, 32, 19, false, true), new ProductEpilogue { Grad = "square_bwd" }, true, true, true, false),
        ("3 batches 19x23x31, a second over the batches, affine 3 -7, its sums kept", new ProductView(19, 23, 31, false, false, 3, 19 * 31, 31 * 23, 19 * 23),
            new ProductEpilogue { Second = new ProductView(19, 23, 6, true, false, 3, 6 * 19, 0, 19 * 23), Act = "affine", ActA = 3, ActB = -7 }, true, false, true, false),
        ("31x37x29 wide values, bias, beta 1, residual, its sums kept", new ProductView(31, 37, 29, false, false), new ProductEpilogue { Beta = 1 }, true, true, true, true),
    ];

    public static bool ProductEpilogues(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            int seed = 80;
            foreach (var (name, v, e, hasBias, hasResidual, keep) in EpilogueCases)
            {
                int n = (int)v.ValuesC;
                var s2 = e.Second;
                byte[] Draw(long count, double lo = -1, double hi = 1) => Bench.Registry.Make(type, (int)count, seed++, lo, hi);
                double[]? Read(byte[]? x, long count) => x is null ? null : Bench.Registry.Values(type, x, (int)count);
                byte[] a = Draw(v.ValuesA), b = Draw(v.ValuesB), y0 = Draw(n);
                byte[]? bias = hasBias ? Draw(v.N) : null, residual = hasResidual ? Draw(n) : null;
                byte[]? inputs = e.Grad is not null ? Draw(n, -3, 3) : null, kept = keep ? new byte[Bench.Registry.Bytes(type, n)] : null;
                byte[]? a2 = s2 is null ? null : Draw(s2.ValuesA), b2 = s2 is null ? null : Draw(s2.ValuesB);
                double[] av = Read(a, v.ValuesA)!, bv = Read(b, v.ValuesB)!, yv = Read(y0, n)!;
                double[]? biasV = Read(bias, v.N), resV = Read(residual, n), inputsV = Read(inputs, n), a2v = Read(a2, s2?.ValuesA ?? 0), b2v = Read(b2, s2?.ValuesB ?? 0);
                var (want, sums) = e.Reference(v, av, bv, yv, biasV, resV, inputsV, a2v, b2v);
                double[]? Abs(double[]? x) => x?.Select(Math.Abs).ToArray();
                var (terms, _) = (e with { Act = null, Grad = null, Beta = Math.Abs(e.Beta) }).Reference(v, Abs(av)!, Abs(bv)!, Abs(yv)!, Abs(biasV), null, null, Abs(a2v), Abs(b2v));
                int formulaUlps = e.Act is not null ? Ops.Find(e.Act).Ulps : e.Grad is not null ? Ops.Find(e.Grad).Ulps : 0;
                double depth = v.K + (s2?.K ?? 0) + 4;
                var c = y0.ToArray();
                Bench.Registry.WithFloat(type, new EpilogueRun(ops, v, e, a, b, c, new ProductTerms<byte[]>(bias, residual, kept, inputs, a2, b2)));
                // The sums' rounding through the formula (its slope at most 1.25), the formula's own, and the residual's sum.
                double worst = Excess(type, $"matmul epilogue {name}", c, want,
                    i => 1.25 * depth * terms[i] + formulaUlps * (Math.Abs(want[i]) + Math.Abs(sums[i])) + 2 * (resV is null ? 0 : Math.Abs(resV[i])), 1);
                if (worst > 0) { Console.WriteLine($"check matmul epilogue {name,-48} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
                if (kept is not null)
                {
                    double bad = Excess(type, $"matmul epilogue sums {name}", kept, sums, i => depth * terms[i], 1);
                    if (bad > 0) { Console.WriteLine($"check matmul epilogue sums {name,-43} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
                }
            }
        }

        ok &= IntegerEpilogues(ops, where);
        if (ok) Console.WriteLine($"check matmul epilogue        {where,-6} every type{(blocks ? "" : " but the block formats")}: a bias, activations with their sums kept and the gradients through them, beta C, a residual, a second product (transposed, over batches, sharing its B), batches transposed and sharing B; the float types within the products' rounding, the others exactly (the any-type formulas, wide values saturating)");
        return ok;
    }

    private static bool IntegerEpilogues(IDevice<ulong, ulong>? ops, string where)
    {
        bool ok = true;
        foreach (var type in Bench.Registry.Names)
        {
            if (Bench.Registry.IsFloat(type)) continue;
            int seed = 180;
            double max = Bench.Registry.Values(type, Bench.Registry.Encode(type, [1e300]), 1)[0], min = Bench.Registry.Values(type, Bench.Registry.Encode(type, [-1e300]), 1)[0];
            double wide = Math.Min(Math.Floor(2 * Math.Sqrt(max)), 1 << 22);
            foreach (var (name, v, e, hasBias, hasResidual, keep, isWide) in IntegerEpilogueCases)
            {
                int n = (int)v.ValuesC;
                var s2 = e.Second;
                byte[] Draw(long count)
                {
                    if (!isWide) return Bench.Registry.Make(type, (int)count, seed++, -1, 1);
                    var random = new Random(seed++);
                    var values = new double[count];
                    for (long i = 0; i < count; i++) values[i] = Math.Floor((min < 0 ? -wide : 0) + (min < 0 ? 2 * wide + 1 : wide + 1) * random.NextDouble());
                    return Bench.Registry.Encode(type, values);
                }

                double[]? Read(byte[]? x, long count) => x is null ? null : Bench.Registry.Values(type, x, (int)count);
                byte[] a = Draw(v.ValuesA), b = Draw(v.ValuesB), y0 = Draw(n);
                byte[]? bias = hasBias ? Draw(v.N) : null, residual = hasResidual ? Draw(n) : null;
                byte[]? inputs = e.Grad is not null ? Draw(n) : null, kept = keep ? new byte[Bench.Registry.Bytes(type, n)] : null;
                byte[]? a2 = s2 is null ? null : Draw(s2.ValuesA), b2 = s2 is null ? null : Draw(s2.ValuesB);
                var (want, sums) = e.Reference(v, Read(a, v.ValuesA)!, Read(b, v.ValuesB)!, Read(y0, n)!, Read(bias, v.N), Read(residual, n), Read(inputs, n),
                    Read(a2, s2?.ValuesA ?? 0), Read(b2, s2?.ValuesB ?? 0));
                var c = y0.ToArray();
                string lane = Bench.Registry.With(type, new IntegerEpilogueRun(ops, v, e, a, b, c, new ProductTerms<byte[]>(bias, residual, kept, inputs, a2, b2)));
                foreach (var (what, got, reference) in kept is null ? new[] { ("y", c, want) } : new[] { ("y", c, want), ("sums", kept, sums) })
                {
                    var g = Bench.Registry.Values(type, got, n);
                    var r = Bench.Registry.Values(type, Bench.Registry.Encode(type, reference), n);
                    for (int i = 0; i < n; i++)
                    {
                        if (g[i] == r[i]) continue;
                        Console.WriteLine($"check matmul epilogue {what} {name,-48} {where,-6} {type,-12} DIFFERS: [{i}] {g[i]:R}, not {r[i]:R} (of {reference[i]:R}; lane {lane})");
                        ok = false;
                        break;
                    }
                }
            }
        }

        return ok;
    }

    private sealed class IntegerEpilogueRun(IDevice<ulong, ulong>? ops, ProductView v, ProductEpilogue e, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t)
        : Bench.ITypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.MatMulEpilogue<TN>(v, e, a, b, c, t);
            using var st = new Staged(ops);
            var lane = ops.MatMulEpilogue<TN>(v, e, st.In(a), st.In(b), st.Out(c), StagedTerms(st, t));
            st.Collect();
            return lane;
        }
    }

    private sealed class EpilogueRun(IDevice<ulong, ulong>? ops, ProductView v, ProductEpilogue e, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.MatMulEpilogue<TN>(v, e, a, b, c, t);
            using var st = new Staged(ops);
            var lane = ops.MatMulEpilogue<TN>(v, e, st.In(a), st.In(b), st.Out(c), StagedTerms(st, t));
            st.Collect();
            return lane;
        }
    }

    // An epilogue's operands on the device: the sums kept an output, every other term an input.
    private static ProductTerms<ulong> StagedTerms(Staged st, ProductTerms<byte[]> t) =>
        new(st.In(t.Bias), st.In(t.Residual), st.Out(t.Sums), st.In(t.ActInputs), st.In(t.SecondA), st.In(t.SecondB));
}
