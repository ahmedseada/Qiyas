// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The convolution's general path (Conv.cs) on every device against its float64 reference: images, groups (a group a
// channel, two filters each, among them), uneven pads, a bias and an activation, every float type (the block formats
// among them); the backward's dx, dw and db over the same shapes, every type. Within the lane's rounding of each value's
// sums (and the activation's own rounding) plus the type's step; an integer type's backward exactly (its lanes exact, each
// gradient its one rounding: the type's own conversion).
namespace Qiyas;

internal static partial class Checks
{
    private static readonly (string Name, ConvArgs Args, bool Bias)[] GeneralConvs =
    [
        ("2 images 3x7x9>4 3x3 s2 p1 relu", Padded(new ConvArgs(3, 7, 9, 4, 3, 3) { Sh = 2, Sw = 2, N = 2, Act = "relu" }, 1, 1, 1, 1), true),
        ("groups 3 6x6x5>6 3x2 pads 1,0,2,1 silu", Padded(new ConvArgs(6, 6, 5, 6, 3, 2) { Groups = 3, Act = "silu" }, 1, 0, 2, 1), true),
        ("3 images a channel a group x2 4x5x5>8 3x3 d2 p2 gelu", Padded(new ConvArgs(4, 5, 5, 8, 3, 3) { Dh = 2, Dw = 2, N = 3, Groups = 4, Act = "gelu" }, 2, 2, 2, 2), false),
        ("2 images groups 2 4x8x6>6 1x1 bias", new ConvArgs(4, 8, 6, 6, 1, 1) { N = 2, Groups = 2 }, true),
        ("3x5x5>2 3x3 p1 bias sigmoid", Padded(new ConvArgs(3, 5, 5, 2, 3, 3) { Act = "sigmoid" }, 1, 1, 1, 1), true),
        ("2 images 2x4x4>3 2x2 pad right 2 tanh", new ConvArgs(2, 4, 4, 3, 2, 2) { N = 2, PadRight = 2, Act = "tanh" }, false),
        ("3x6x7>4 3x3 s2 pads 0,2,1,0 bias", Padded(new ConvArgs(3, 6, 7, 4, 3, 3) { Sh = 2, Sw = 2 }, 0, 2, 1, 0), true),
    ];

    private static bool ConvGeneral(IDevice<ulong, ulong>? ops, string where, bool blocks)
    {
        bool ok = true;
        foreach (var type in Bench.Registry.Names)
        {
            if (!blocks && Bench.Registry.IsBlock(type)) continue;
            bool floats = Bench.Registry.IsFloat(type);
            foreach (var (name, a, hasBias) in GeneralConvs)
            {
                var x = Bench.Registry.Make(type, (int)a.XValues, 67, -1, 1);
                var w = Bench.Registry.Make(type, (int)a.WValues, 68, -1, 1);
                var xv = Bench.Registry.Values(type, x, (int)a.XValues);
                var wv = Bench.Registry.Values(type, w, (int)a.WValues);
                if (floats)
                {
                    var bias = hasBias ? Bench.Registry.Make(type, a.OC, 69, -1, 1) : null;
                    var bv = bias is null ? null : Bench.Registry.Values(type, bias, a.OC);
                    var want = a.Reference(xv, wv, bv);
                    var pre = (a with { Act = null }).Reference(xv, wv, bv);
                    var scale = a.Scale(xv, wv, bv);
                    var y = new byte[Bench.Registry.Bytes(type, a.YValues)];
                    Bench.Registry.WithFloat(type, new ConvGeneralRun(ops, a, x, w, bias, y));
                    // The sums' rounding through the activation (its slope at most 1.2), and the activation's own.
                    double worst = Excess(type, $"conv {name}", y, want, i => 1.25 * (a.Rows + 2) * scale[i] + 64 * (Math.Abs(want[i]) + Math.Abs(pre[i])), 1);
                    if (worst > 0) { Console.WriteLine($"check conv {name,-48} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
                }

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

    private sealed class ConvGeneralRun(IDevice<ulong, ulong>? ops, ConvArgs a, byte[] x, byte[] w, byte[]? bias, byte[] y) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.Conv<TN>(a, x, w, bias, y);
            using var st = new Staged(ops);
            var lane = ops.Conv<TN>(a, st.In(x), st.In(w), st.In(bias), st.Out(y));
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
