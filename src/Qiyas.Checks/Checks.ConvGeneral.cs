// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The convolution's general path (Conv.cs) on every device against its float64 reference: images, groups (a group a
// channel, two filters each, among them), uneven pads, and the product's epilogue over the output (a bias a value an
// output channel, an activation, β·y, a residual, the sums kept, a gradient through the activation's inputs, a second
// product), every type (the block formats among them); the backward's dx, dw and db over the same shapes, every type. A
// float type within the lane's rounding of each value's sums (and the formula's own rounding) plus the type's step; any
// other type exactly (its lanes exact, each output its one rounding: the type's own conversion, saturating), and refused
// where its epilogue is not an integer one (ProductEpilogue.ValidateType: a float formula, a β or parameter not an
// integer).
namespace Qiyas;

internal static partial class Checks
{
    // A general case: the shape, the epilogue, and which of the epilogue's operands are given (the bias, the residual, the
    // sums kept, the activation's inputs; the second product's operands with its view).
    private sealed record ConvCase(string Name, ConvArgs Args, ProductEpilogue E, bool Bias = false, bool Residual = false, bool Keep = false, bool ActInputs = false);

    private static readonly ConvCase[] GeneralConvs =
    [
        new("2 images 3x7x9>4 3x3 s2 p1 bias relu", Padded(new ConvArgs(3, 7, 9, 4, 3, 3) { Sh = 2, Sw = 2, N = 2 }, 1, 1, 1, 1), new() { Act = "relu" }, Bias: true),
        new("groups 3 6x6x5>6 3x2 pads 1,0,2,1 bias silu", Padded(new ConvArgs(6, 6, 5, 6, 3, 2) { Groups = 3 }, 1, 0, 2, 1), new() { Act = "silu" }, Bias: true),
        new("3 images a channel a group x2 4x5x5>8 3x3 d2 p2 gelu", Padded(new ConvArgs(4, 5, 5, 8, 3, 3) { Dh = 2, Dw = 2, N = 3, Groups = 4 }, 2, 2, 2, 2), new() { Act = "gelu" }),
        new("2 images groups 2 4x8x6>6 1x1 bias", new ConvArgs(4, 8, 6, 6, 1, 1) { N = 2, Groups = 2 }, new(), Bias: true),
        new("3x5x5>2 3x3 p1 bias sigmoid", Padded(new ConvArgs(3, 5, 5, 2, 3, 3), 1, 1, 1, 1), new() { Act = "sigmoid" }, Bias: true),
        new("2 images 2x4x4>3 2x2 pad right 2 tanh", new ConvArgs(2, 4, 4, 3, 2, 2) { N = 2, PadRight = 2 }, new() { Act = "tanh" }),
        new("3x6x7>4 3x3 s2 pads 0,2,1,0 bias", Padded(new ConvArgs(3, 6, 7, 4, 3, 3) { Sh = 2, Sw = 2 }, 0, 2, 1, 0), new(), Bias: true),
        new("2 images groups 2 4x6x5>4 3x3 p1 bias b0.5 leaky_relu residual", Padded(new ConvArgs(4, 6, 5, 4, 3, 3) { N = 2, Groups = 2 }, 1, 1, 1, 1), new() { Beta = 0.5, Act = "leaky_relu", ActA = 0.1 }, Bias: true, Residual: true),
        new("3 images 3x5x6>4 2x3 bias gelu sums kept", new ConvArgs(3, 5, 6, 4, 2, 3) { N = 3 }, new() { Act = "gelu" }, Bias: true, Keep: true),
        new("2 images 2x6x6>3 3x3 p1 gelu_bwd through its inputs", Padded(new ConvArgs(2, 6, 6, 3, 3, 3) { N = 2 }, 1, 1, 1, 1), new() { Grad = "gelu_bwd" }, ActInputs: true),
        new("3x5x5>4 3x3 p1 bias a second product of depth 5 silu", Padded(new ConvArgs(3, 5, 5, 4, 3, 3), 1, 1, 1, 1), new() { Act = "silu", Second = new ProductView(4, 25, 5, false, false) }, Bias: true),
        new("2 images groups 2 4x5x6>4 2x2 s2 p1 bias b2 relu residual sums kept", Padded(new ConvArgs(4, 5, 6, 4, 2, 2) { N = 2, Groups = 2, Sh = 2, Sw = 2 }, 1, 1, 1, 1), new() { Beta = 2, Act = "relu" }, Bias: true, Residual: true, Keep: true),
        new("2 images 3x4x5>2 3x3 p1 a second product of depth 3 relu", Padded(new ConvArgs(3, 4, 5, 2, 3, 3) { N = 2 }, 1, 1, 1, 1), new() { Act = "relu", Second = new ProductView(2, 20, 3, false, false, 2, 6, 60, 40) }),
    ];

    private static bool ConvGeneral(IDevice<ulong, ulong>? ops, string where, bool blocks)
    {
        bool ok = true;
        foreach (var type in Bench.Registry.Names)
        {
            if (!blocks && Bench.Registry.IsBlock(type)) continue;
            bool floats = Bench.Registry.IsFloat(type);
            foreach (var c in GeneralConvs)
            {
                var a = c.Args;
                string name = c.Name;
                var x = Bench.Registry.Make(type, (int)a.XValues, 67, -1, 1);
                var w = Bench.Registry.Make(type, (int)a.WValues, 68, -1, 1);
                var xv = Bench.Registry.Values(type, x, (int)a.XValues);
                var wv = Bench.Registry.Values(type, w, (int)a.WValues);
                ok &= ConvGeneralForward(ops, where, type, c, x, w, xv, wv);

                // The backward: dy the gradient of the sums plus the bias.
                var dy = Bench.Registry.Make(type, (int)a.YValues, 70, -1, 1);
                var dyv = Bench.Registry.Values(type, dy, (int)a.YValues);
                var (wantX, wantW, wantB) = a.ReferenceBackward(xv, wv, dyv);
                var (sx, sw, sb) = a.ReferenceBackward(xv.Select(Math.Abs).ToArray(), wv.Select(Math.Abs).ToArray(), dyv.Select(Math.Abs).ToArray());
                byte[] dx = new byte[x.Length], dw = new byte[w.Length], db = new byte[Bench.Registry.Bytes(type, a.OC)];
                Bench.Registry.With(type, new ConvGeneralBackwardRun(ops, a, x, w, dy, dx, dw, db));
                int kx = a.OCg * a.Kh * a.Kw + 2, kw = a.N * a.Cols + 2;
                foreach (var (what, got, reference, s, k) in new[] { ("dx", dx, wantX, sx, kx), ("dw", dw, wantW, sw, kw), ("db", db, wantB, sb, kw) })
                {
                    if (!floats)
                    {
                        ok &= Close(type, $"conv_bwd {what} {name}", where, got, reference, reference.Length);
                        continue;
                    }

                    double bad = Excess(type, $"conv_bwd {what} {name}", got, reference, i => k * s[i], 1);
                    if (bad > 0) { Console.WriteLine($"check conv_bwd {what} {name,-44} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
                }
            }
        }

        return ok;
    }

    // The forward of one case: y (and the sums kept) against the epilogue's float64 reference over the convolution's sums; a
    // type that is not float exactly, or refused where the epilogue is not an integer one.
    private static bool ConvGeneralForward(IDevice<ulong, ulong>? ops, string where, string type, ConvCase c, byte[] x, byte[] w, double[] xv, double[] wv)
    {
        var a = c.Args;
        var e = c.E;
        bool floats = Bench.Registry.IsFloat(type);
        int n = (int)a.YValues;
        byte[]? Made(bool given, long count, int seed) => given ? Bench.Registry.Make(type, (int)count, seed, -1, 1) : null;
        double[]? Of(byte[]? b, long count) => b is null ? null : Bench.Registry.Values(type, b, (int)count);
        var bias = Made(c.Bias, a.OC, 69);
        var residual = Made(c.Residual, n, 71);
        var inputs = Made(c.ActInputs, n, 72);
        var a2 = Made(e.Second is not null, e.Second?.ValuesA ?? 0, 73);
        var b2 = Made(e.Second is not null, e.Second?.ValuesB ?? 0, 74);
        // y's values before: read by β, else zeros.
        var y = e.Beta != 0 ? Bench.Registry.Make(type, n, 75, -1, 1) : new byte[Bench.Registry.Bytes(type, a.YValues)];
        var y0 = Bench.Registry.Values(type, y, n);
        var sums = c.Keep ? new byte[Bench.Registry.Bytes(type, a.YValues)] : null;
        double[]? bv = Of(bias, a.OC), a2v = Of(a2, e.Second?.ValuesA ?? 0), b2v = Of(b2, e.Second?.ValuesB ?? 0);
        var (want, s) = a.Reference(e, xv, wv, y0, bv, Of(residual, n), Of(inputs, n), a2v, b2v);

        // The sums' error scale: Σ|w·x| and the magnitude of every term added to them before the formula.
        var scale = a.Scale(xv, wv);
        var second = e.Second?.Reference(a2v!.Select(Math.Abs).ToArray(), b2v!.Select(Math.Abs).ToArray());
        for (int i = 0; i < n; i++)
            scale[i] += (bv is null ? 0 : Math.Abs(bv[i / a.Cols % a.OC])) + Math.Abs(e.Beta * y0[i]) + (second is null ? 0 : second[i]);

        var run = new ConvGeneralRun(ops, a, e, x, w, y, new ProductTerms<byte[]>(bias, residual, sums, inputs, a2, b2));
        if (!floats && !IntegerEpilogue(e))
        {
            try { Bench.Registry.With(type, run); }
            catch (Exception ex) when (ex is NotSupportedException or ArgumentException) { return true; }
            Console.WriteLine($"check conv {c.Name,-48} {where,-6} {type,-12} FAILS: not refused (its epilogue is not an integer one)");
            return false;
        }

        string lane = Bench.Registry.With(type, run);
        if (!floats)
        {
            bool exact = Close(type, $"conv {c.Name} (lane {lane})", where, y, want, n);
            if (sums is not null) exact &= Close(type, $"conv sums kept {c.Name} (lane {lane})", where, sums, s, n);
            return exact;
        }

        bool ok = true;
        // The sums' rounding through the formula (its slope at most 1.2), and the formula's and the residual's own.
        double worst = Excess(type, $"conv {c.Name}", y, want, i => 1.25 * (a.Rows + (e.Second?.K ?? 0) + 4) * scale[i] + 64 * (Math.Abs(want[i]) + Math.Abs(s[i])), 1);
        if (worst > 0) { Console.WriteLine($"check conv {c.Name,-48} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
        if (sums is not null)
        {
            double kept = Excess(type, $"conv kept {c.Name}", sums, s, i => (a.Rows + 4) * scale[i] + 4 * Math.Abs(s[i]), 1);
            if (kept > 0) { Console.WriteLine($"check conv sums kept {c.Name,-37} {where,-6} {type,-12} DIFFERS by {kept:0.##}"); ok = false; }
        }

        return ok;
    }

    // Whether a type that is not float takes the epilogue (ProductEpilogue.ValidateType).
    private static bool IntegerEpilogue(ProductEpilogue e)
    {
        try { e.ValidateType("int32", false); return true; }
        catch (Exception ex) when (ex is NotSupportedException or ArgumentException) { return false; }
    }

    private sealed class ConvGeneralRun(IDevice<ulong, ulong>? ops, ConvArgs a, ProductEpilogue e, byte[] x, byte[] w, byte[] y, ProductTerms<byte[]> t) : Bench.ITypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.Conv<TN>(a, e, x, w, y, t);
            using var st = new Staged(ops);
            var lane = ops.Conv<TN>(a, e, st.In(x), st.In(w), st.Out(y), StagedTerms(st, t));
            st.Collect();
            return lane;
        }
    }

    private sealed class ConvGeneralBackwardRun(IDevice<ulong, ulong>? ops, ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw, byte[] db) : Bench.ITypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.ConvBackward<TN>(a, x, w, dy, dx, dw, db);
            using var st = new Staged(ops);
            var lane = ops.ConvBackward<TN>(a, st.In(x), st.In(w), st.In(dy), st.Out(dx), st.Out(dw), st.Out(db));
            st.Collect();
            return lane;
        }
    }
}
