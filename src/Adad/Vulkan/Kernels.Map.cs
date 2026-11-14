// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The element-wise family on Vulkan: the transcendental functions over a float lane (the same tables, reductions and
// steps as Funcs and CUDA's Ptx.Map.cs, evaluated by fused multiply-adds), the catalogue's formulas over any lane, and
// the kernels: add and map a thread per stored word (each value unpacked into the lane, the formula applied, packed),
// the row map a thread per value.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int Threads = 256;

    // ---------------------------------------------------------------- functions on a float lane value

    internal static class Fn
    {
        private static V Poly(Spv b, V t, double[] coefficients, int terms, int first)
        {
            var p = b.C(t.Kind, coefficients[terms - 1]);
            for (int k = terms - 2; k >= first; k--) p = b.Fma(p, t, b.C(t.Kind, coefficients[k]));
            return p;
        }

        /// <summary>e^x by the CPU's polynomial: the exact reduction n·ln 2, a degree-7 (13) Taylor polynomial, the scale built
        /// in the exponent bits in two halves (a result into the subnormals scaled once each way).</summary>
        public static V Exp(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            double lo = w ? -745.1332 : -103.9721, hi = w ? 709.78 : 88.72283;
            var c = b.Min(b.Max(x, b.C(x.Kind, lo)), b.C(x.Kind, hi));
            var n = b.RoundEven(c * (w ? 1.4426950408889634 : 1.44269504));
            var r = b.Fma(n, b.C(x.Kind, w ? -6.93147180369123816490e-01 : -0.693359375), c);
            r = b.Fma(n, b.C(x.Kind, w ? -1.90821492927058770002e-10 : 2.12194440e-4), r);
            V p;
            if (w)
            {
                p = b.D(1.0 / 6227020800);
                foreach (double k in new[] { 1.0 / 479001600, 1.0 / 39916800, 1.0 / 3628800, 1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0, 1.0 })
                    p = b.Fma(p, r, b.D(k));
            }
            else
            {
                p = b.F(1f / 5040);
                foreach (float k in new[] { 1f / 720, 1f / 120, 1f / 24, 1f / 6, 0.5f, 1f, 1f }) p = b.Fma(p, r, b.F(k));
            }

            var ni = b.Convert(n, w ? K.S64 : K.S32);
            var half = ni >> 1;
            var s1 = b.Bitcast((half + (w ? 1023 : 127)) << (w ? 52 : 23), x.Kind);
            var s2 = b.Bitcast(((ni - half) + (w ? 1023 : 127)) << (w ? 52 : 23), x.Kind);
            var y = p * s1 * s2;
            y = b.Select(x < lo, b.C(x.Kind, 0), y);
            y = b.Select(x > hi, b.C(x.Kind, double.PositiveInfinity), y);
            return b.Select(b.IsNan(x), x, y);
        }

        public static V Log(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            var tiny = x < (w ? 2.2250738585072014e-308 : 1.17549435e-38);
            var xs = b.Select(tiny, x * (w ? 4503599627370496.0 : 8388608.0), x);
            V e, m;
            if (w)
            {
                var bits = b.Bitcast(xs, K.S64);
                e = b.Convert(bits >> 52, K.S32) - 1023;
                m = b.Bitcast((b.Bits(xs) & 0x000FFFFFFFFFFFFFUL) | 0x3FF0000000000000UL, K.F64);
            }
            else
            {
                var bits = b.Bitcast(xs, K.S32);
                e = (bits >> 23) - 127;
                m = b.Bitcast((b.Bits(xs) & 0x007FFFFFUL) | 0x3F800000UL, K.F32);
            }

            var big = m > 1.4142135623730951;
            m = b.Select(big, m * 0.5, m);
            e = b.Select(big, e + 1, e);
            e = b.Select(tiny, e - (w ? 52 : 23), e);
            var t = (m - 1.0) / (m + 1.0);
            var t2 = t * t;
            var p = Poly(b, t2, Funcs.Atanh, w ? Funcs.AtanhTerms64 : Funcs.AtanhTerms32, 0);
            var ef = b.Convert(e, x.Kind);
            var y = b.Fma(ef, b.C(x.Kind, w ? Funcs.Ln2Lo : Funcs.Ln2LoF), p * t);
            y = b.Fma(ef, b.C(x.Kind, w ? Funcs.Ln2Hi : Funcs.Ln2HiF), y);
            y = b.Select(x == 0, b.C(x.Kind, double.NegativeInfinity), y);
            y = b.Select(x < 0, b.C(x.Kind, double.NaN), y);
            y = b.Select(x == double.PositiveInfinity, x, y);
            return b.Select(b.IsNan(x), x, y);
        }

        // x − k·π/2 as an unevaluated pair r + lo (CUDA's ReduceHalfPiPair).
        private static (V R, V Lo, V K) ReduceHalfPiPair(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            var k = b.RoundEven(x * 0.63661977236758134);
            var r1 = b.Fma(k, b.C(x.Kind, w ? -Funcs.PiHalf1 : -Funcs.PiHalf1F), x);
            var p2 = k * (w ? -Funcs.PiHalf2 : -Funcs.PiHalf2F);
            var r = r1 + p2;
            var bv = r - r1;
            var t = r1 - (r - bv);
            var lo = (p2 - bv) + t;
            lo = b.Fma(k, b.C(x.Kind, w ? -Funcs.PiHalf3 : -Funcs.PiHalf3F), lo);
            return (r, lo, b.Convert(k, K.S32));
        }

        private static V Trig(Spv b, V x, bool sine)
        {
            bool w = x.Kind == K.F64;
            var (r, lo, k) = ReduceHalfPiPair(b, x);
            var r2 = r * r;
            int terms = w ? Funcs.SinTerms64 : Funcs.SinTerms32;
            var s = Poly(b, r2, Funcs.SinSeries, terms, 1);
            var c = Poly(b, r2, Funcs.CosSeries, terms, 1);
            var r3 = r2 * r;
            var m = b.Fma(r2 * -0.5, lo, lo);
            s = b.Fma(s, r3, m) + r;
            c = b.Fma(c, r2, -(r * lo)) + 1.0;
            var q = k & 3UL;
            var odd = (q & 1UL) == 1;
            var neg = (q & 2UL) == 2;
            var y = sine ? b.Select(odd, c, s) : b.Select(odd, -s, c);
            y = b.Select(neg, -y, y);
            return b.Select(b.Abs(x) >= double.PositiveInfinity, b.C(x.Kind, double.NaN), y);
        }

        public static V Sin(Spv b, V x) => Trig(b, x, true);

        public static V Cos(Spv b, V x) => Trig(b, x, false);

        public static V Expm1(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            double lo = w ? -708.39 : -87.33654, hi = w ? 709.78 : 88.72283;
            var c = b.Min(b.Max(x, b.C(x.Kind, lo)), b.C(x.Kind, hi));
            var n = b.RoundEven(c * (w ? 1.4426950408889634 : 1.44269504));
            var r = b.Fma(n, b.C(x.Kind, w ? -Funcs.Ln2Hi : -Funcs.Ln2HiF), c);
            r = b.Fma(n, b.C(x.Kind, w ? -Funcs.Ln2Lo : -Funcs.Ln2LoF), r);
            var q = Poly(b, r, Funcs.InvFact, w ? Funcs.ExpTerms64 : Funcs.ExpTerms32, 1) * r;
            var ni = b.Convert(n, w ? K.S64 : K.S32);
            var scale = b.Bitcast((ni + (w ? 1023 : 127)) << (w ? 52 : 23), x.Kind);
            var y = b.Fma(q, scale, scale - 1.0);
            y = b.Select(x < lo, b.C(x.Kind, -1), y);
            y = b.Select(x > hi, b.C(x.Kind, double.PositiveInfinity), y);
            return b.Select(b.IsNan(x), x, y);
        }

        public static V Tanh(Spv b, V x)
        {
            var a = b.Min(b.Max(x, b.C(x.Kind, -20)), b.C(x.Kind, 20)) * 2.0;
            var e = Expm1(b, a);
            var y = e / (e + 2.0);
            return b.Select(b.IsNan(x), x, y);
        }

        public static V Sigmoid(Spv b, V x) => b.C(x.Kind, 1) / (Exp(b, -x) + 1.0);

        /// <summary>atan by Funcs.Atan's reduction and series.</summary>
        public static V Atan(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            var a = b.Abs(x);
            var inv = a > 1.0;
            var u = b.Select(inv, b.C(x.Kind, 1) / a, a);
            var mid = u > Funcs.TanPiTwelfth;
            var s3 = b.C(x.Kind, w ? Funcs.Sqrt3 : Funcs.Sqrt3F);
            var num = b.Fma(u, b.C(x.Kind, w ? Funcs.Sqrt3Lo : Funcs.Sqrt3LoF), b.Fma(u, s3, b.C(x.Kind, -1)));
            var t = b.Select(mid, num / (u + s3), u);
            var r = Poly(b, t * t, Funcs.AtanSeries, w ? Funcs.AtanTerms64 : Funcs.AtanTerms32, 0) * t;
            r = b.Select(mid, (r + (w ? Funcs.PiSixthLo : Funcs.PiSixthLoF)) + (w ? Funcs.PiSixth : Funcs.PiSixthF), r);
            r = b.Select(inv, (b.C(x.Kind, w ? Funcs.PiHalf : Funcs.PiHalfF) - r) + (w ? Funcs.PiHalfLo : Funcs.PiHalfLoF), r);
            return b.Select(x < 0.0, -r, r);
        }

        // erfc |x| past its bound: e^(−x²) (x² carried with its rounding) times, on the f64 lane, S(u)/x (Funcs.ErfcTail64:
        // u = 1.5/x − 1), on the f32 lane t·S(u) (the CPU's Funcs.ErfcTail32, step for step: t = 1/x, u = t·(2/span) − 1),
        // a polynomial in place of CUDA's continued fraction of 48 or 80 quotients (the same error, the bench's: erf
        // float32 1.45e-7 by either).
        private static V ErfcTail(Spv b, V ax, V x2)
        {
            bool w = ax.Kind == K.F64;
            var corr = b.C(ax.Kind, 1) - b.Fma(ax, ax, -x2);
            corr = b.Select(b.IsNan(corr), b.C(ax.Kind, 1), corr);             // x² infinite: its low part ∞ − ∞, e^(−x²) 0: no correction
            if (w)
            {
                var u = b.Fma(b.C(ax.Kind, 1) / ax, b.C(ax.Kind, 1.5), b.C(ax.Kind, -1));
                var sp = Poly(b, u, Funcs.ErfcTail64, Funcs.ErfcTail64.Length, 0);
                var e = Exp(b, -x2) * corr;
                return (e * sp) / ax;
            }

            var t = b.C(ax.Kind, 1) / ax;
            var v = b.Fma(t, b.C(ax.Kind, (float)(2 / Funcs.ErfcTailSpan)), b.C(ax.Kind, -1));
            var s = Poly(b, v, Funcs.ErfcTail32, Funcs.ErfcTail32.Length, 0);
            var e32 = Exp(b, -x2) * corr;
            return e32 * (t * s);
        }

        public static V Erf(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            var ax = b.Abs(x);
            var x2 = ax * ax;
            var series = Poly(b, x2, Funcs.ErfSeries, w ? Funcs.ErfTerms64 : Funcs.ErfTerms32, 0) * ax;
            var tail = b.C(x.Kind, 1) - ErfcTail(b, ax, x2);
            var y = b.Select(ax < (w ? Funcs.ErfBound64 : Funcs.ErfBound32), series, tail);
            y = b.Min(y, b.C(x.Kind, 1));
            y = b.Select(x < 0, -y, y);
            return b.Select(b.IsNan(x), x, y);
        }

        public static V Erfc(Spv b, V x)
        {
            bool w = x.Kind == K.F64;
            var ax = b.Abs(x);
            var x2 = ax * ax;
            var series = Poly(b, x2, Funcs.ErfSeries, w ? Funcs.ErfTerms64 : Funcs.ErfTerms32, 0) * x;
            var tail = ErfcTail(b, ax, x2);
            var mid = b.C(x.Kind, 1) - series;
            var two = b.C(x.Kind, 2) - tail;
            var y = b.Select(x < 0, two, tail);
            y = b.Select(ax < (w ? Funcs.ErfcBound64 : Funcs.ErfcBound32), mid, y);
            return b.Select(b.IsNan(x), x, y);
        }
    }

    // ---------------------------------------------------------------- the formulas over a lane

    private sealed class Emit(Spv b, K k)
    {
        public V C(double v) => b.C(k, v);

        public V Neg(V x) => k.IsFloat() || k.IsSigned() ? -x : C(0) - x;

        // Map's div: a float lane's quotient; an integer lane's by Ops.IntegerDivision (toward zero; a signed lane's x / −1
        // is −x, wrapping; x / 0 = 0). The lane divides by 1 where the divisor is 0 or a signed −1 (SDiv and UDiv are
        // undefined there).
        public V Quotient(V x, V y)
        {
            if (k.IsFloat()) return x / y;
            var zero = y == C(0);
            var minusOne = k.IsSigned() ? y == C(-1) : b.Bool(false);
            var q = x / b.Select(zero | minusOne, C(1), y);
            q = b.Select(minusOne, Neg(x), q);
            return b.Select(zero, C(0), q);
        }

        public V Sign(V x) => b.Select(x > 0, C(1), b.Select(x < 0, Neg(C(1)), C(0)));

        public V Exp(V x) => Fn.Exp(b, x);

        public V Sigmoid(V x) => Fn.Sigmoid(b, x);

        public V Phi(V x) => Fn.Erfc(b, x * -0.7071067811865476) * 0.5;

        public V Gelu(V x)
        {
            var x2 = x * x;
            if (k == K.F64) return x * Sigmoid((b.Fma(x2 * 0.044715, x, x)) * (2 * 0.7978845608028654));
            const double c = 2 * 0.7978845608028654;
            double ch = (float)c;
            var p = b.Fma(x2 * (c * 0.044715), x, x * (c - ch));
            var t = b.Fma(C(ch), x, p);
            t = b.Select(b.IsNan(t), x, t);                                     // x infinite: t is ±∞ with x (its parts ∞ − ∞)
            var lo = b.Fma(C(ch), x, -t) + p;
            return XSigmoid(x, t, lo);
        }

        /// <summary>x·σ(t + lo) as x / (1 + e^(−t)·(1 − lo)), the denominator's sum split exactly and its low part folded
        /// into the quotient (CUDA's XSigmoid; the small correction's quotient the device's own, as CUDA's div.approx). Where
        /// e^(−t) is infinite (t below the lane's exp range: gelu past x ≈ −10, silu past −88.7) the quotient x / ∞ = ±0
        /// stands: the low part (∞·lo + ∞) and the split sum (∞ − ∞) would make a NaN of it.</summary>
        public V XSigmoid(V x, V t, V? lo = null)
        {
            if (k == K.F64) return x * Sigmoid(t);
            var e = Exp(-t);
            var huge = b.IsInf(e);
            if (lo is { } l) e = b.Select(huge, e, b.Fma(e, -b.Select(b.IsNan(l), C(0), l), e));  // lo a NaN only where t is infinite (∞ − ∞): no low part
            var one = C(1);
            var d = e + one;
            var bv = d - e;
            var dlo = (e - (d - bv)) + (one - bv);
            var q = x / d;
            var c = b.Op2(dlo, d, SpirvOp.FDiv);
            return b.Select((c == 0) | huge, q, b.Fma(-q, c, q));               // no correction: q as it is (q infinite, c 0: not ∞·0)
        }

        public V GeluGrad(V x)
        {
            var x2 = x * x;
            var u2 = b.Fma(x2 * 0.044715, x, x) * (2 * 0.7978845608028654);
            var s = Sigmoid(u2);
            var sn = Sigmoid(-u2);
            var du = b.Fma(C(3 * 0.044715), x2, C(1)) * 0.7978845608028654;
            return b.Fma((x * 2.0) * (s * sn), du, s);
        }

        public V SiluGrad(V x) => Sigmoid(x) * b.Fma(x, Sigmoid(-x), C(1));

        public V Pow(V x, V a)
        {
            var mag = Exp(a * Fn.Log(b, b.Abs(x)));
            var half = a * 0.5;
            var isInt = b.RoundEven(a) == a;
            var odd = isInt & (b.RoundEven(half) != half);
            return b.Select(x < 0, b.Select(odd, -mag, b.Select(isInt, mag, C(double.NaN))), mag);
        }
    }

    /// <summary>Formula <paramref name="op"/> over lane values x, y, z and parameters a, pb (CUDA's Formula, op for op).</summary>
    internal static V Formula(Spv b, K k, string op, V x, V y, V z, V a, V pb)
    {
        var e = new Emit(b, k);
        switch (op)
        {
            case "copy": return x;
            case "fill": return a;
            case "neg": return e.Neg(x);
            case "abs": return b.Abs(x);
            case "sign": return e.Sign(x);
            case "square": return x * x;
            case "relu": return First(b, x, e.C(0));             // max(x, 0) by the order: a NaN stays a NaN
            case "add_scalar": return x + a;
            case "scale": return x * a;
            case "affine": return b.Fma(a, x, pb);
            case "clamp": return x.Kind.IsFloat() ? b.Select(b.IsNan(x), x, b.Min(b.Max(x, a), pb)) : b.Min(b.Max(x, a), pb);   // a NaN stays a NaN
            case "add": return x + y;
            case "sub": return x - y;
            case "mul": return x * y;
            case "div": return e.Quotient(x, y);
            case "max": return First(b, x, y);                   // by the order: a NaN where either is one
            case "min": return b.Min(x, y);                      // NMin: the number where one is a NaN, as the order's last
            case "axpy": return b.Fma(a, x, y);
            case "muladd": return b.Fma(x, y, z);
            case "where": return b.Select(x != 0, y, z);
            case "square_bwd": return (e.C(2) * x) * y;
            case "abs_bwd": return e.Sign(x) * y;
            case "relu_bwd": return b.Select(x > 0, y, e.C(0));
            case "clamp_bwd": return b.Select((x > a) & (x < pb), y, e.C(0));
            case "where_bwd": return b.Select(x != 0, y, e.C(0));
            case "max_bwd_x": return b.Select(Above(b, y, x), e.C(0), z);
            case "max_bwd_y": return b.Select(Above(b, y, x), z, e.C(0));
            case "min_bwd_x": return b.Select(Above(b, x, y), e.C(0), z);
            case "min_bwd_y": return b.Select(Above(b, x, y), z, e.C(0));
            case "gated_relu": return First(b, x, e.C(0)) * y;
            case "gated_relu_bwd_x": return b.Select(x > 0, y * z, e.C(0));
            case "gated_relu_bwd_y": return First(b, x, e.C(0)) * z;
            case "sqrt": return b.Sqrt(x);
            case "inv_sqrt": return e.C(1) / b.Sqrt(x + a);
            case "exp": return e.Exp(x);
            case "expm1": return Fn.Expm1(b, x);
            case "log": return Fn.Log(b, x);
            case "sin": return Fn.Sin(b, x);
            case "cos": return Fn.Cos(b, x);
            case "tanh": return Fn.Tanh(b, x);
            case "atan": return Fn.Atan(b, x);
            case "sigmoid": return e.Sigmoid(x);
            case "erf": return Fn.Erf(b, x);
            case "gelu": return e.Gelu(x);
            case "gelu_erf": return x * e.Phi(x);
            case "silu": return e.XSigmoid(x, x);
            case "leaky_relu": return b.Select(x > 0, x, a * x);
            case "pow": return e.Pow(x, a);
            case "mask_scale": return b.Select(y != 0, x * a, e.C(0));
            case "gated_silu": return (x * e.Sigmoid(x)) * y;
            case "gated_gelu": return e.Gelu(x) * y;
            case "sqrt_bwd": return y / (e.C(2) * b.Sqrt(x));
            case "inv_sqrt_bwd": { var s = x + a; var r = e.C(1) / b.Sqrt(s); return (e.C(-0.5) * (r / s)) * y; }
            case "exp_bwd": return e.Exp(x) * y;
            case "log_bwd": return y / x;
            case "sin_bwd": return Fn.Cos(b, x) * y;
            case "cos_bwd": return -(Fn.Sin(b, x) * y);
            case "tanh_bwd": { var s = e.Sigmoid(e.C(2) * x); return (e.C(4) * (s * e.Sigmoid(e.C(-2) * x))) * y; }
            case "softcap": return a * Fn.Tanh(b, (pb * x) / a);
            case "softcap_bwd": { var u = (pb * x) / a; var s = e.Sigmoid(e.C(2) * u); return ((e.C(4) * (s * e.Sigmoid(e.C(-2) * u))) * pb) * y; }
            case "sigmoid_bwd": return (e.Sigmoid(x) * e.Sigmoid(-x)) * y;
            case "silu_bwd": return e.SiluGrad(x) * y;
            case "gelu_bwd": return e.GeluGrad(x) * y;
            case "gelu_erf_bwd":
            {
                var phi = e.Phi(x);
                var dens = e.C(0.3989422804014327) * e.Exp(e.C(-0.5) * (x * x));
                return b.Fma(x, dens, phi) * y;
            }
            case "leaky_relu_bwd": return b.Select(x > 0, y, a * y);
            case "pow_bwd": return (a * e.Pow(x, a - e.C(1))) * y;
            case "gated_silu_bwd_x": return (e.SiluGrad(x) * y) * z;
            case "gated_silu_bwd_y": return (x * e.Sigmoid(x)) * z;
            case "gated_gelu_bwd_x": return (e.GeluGrad(x) * y) * z;
            case "gated_gelu_bwd_y": return e.Gelu(x) * z;
            case "gated_tanh_sigmoid": return Fn.Tanh(b, x) * e.Sigmoid(y);
            case "gated_tanh_sigmoid_bwd_x": { var s = e.Sigmoid(e.C(2) * x); var sech2 = e.C(4) * (s * e.Sigmoid(e.C(-2) * x)); return (sech2 * e.Sigmoid(y)) * z; }
            case "gated_tanh_sigmoid_bwd_y": return (Fn.Tanh(b, x) * (e.Sigmoid(y) * e.Sigmoid(-y))) * z;
            case "add_gelu": return e.Gelu(x + y);
            case "clip_factor": return b.Select(x > 0, b.Min(e.C(1), a / b.Sqrt(x)), e.C(1));
            case "sigmoid_focal":
            case "sigmoid_focal_bwd":
            {
                // Abstraction/Map.cs's parts: BCE with log1p by Goldberg's form, 1 − p_t, (1 − p_t)^γ (1 where γ is 0), α_t.
                V one = e.C(1), zero = e.C(0);
                var u0 = e.Exp(-b.Abs(x));
                var w = one + u0;
                var dw = w - one;
                var lp = b.Select(dw > 0, Fn.Log(b, w) * u0 / dw, u0);
                var ce = b.Fma(-x, y, b.Max(x, zero)) + lp;
                V s = e.Sigmoid(x), sn = e.Sigmoid(-x), u = one - y;
                var q = y * sn + u * s;
                var qg = b.Select(pb == 0, one, e.Pow(q, pb));
                var at = a * y + (one - a) * u;
                if (op == "sigmoid_focal") return at * qg * ce;
                var d = u * s - y * sn;
                var second = pb * ce * (qg / q) * (y + y - one) * (s * sn);
                second = b.Select(q == 0, zero, b.Select(pb == 0, zero, second));
                return at * (d * qg - second) * z;
            }

            default: throw new ArgumentException(op);
        }
    }

    // ---------------------------------------------------------------- the kernels

    // Words that hold n values of f (a u32 value).
    private static V WordCount(Spv b, Fmt f, V n) =>
        f.WordBits == 64 ? n : b.Convert((b.Convert(n, K.U64) * (double)f.Bits + 31.0) >> 5, K.U32);

    /// <summary>c = a + b over n values: a thread per word, each value in the type's own lane.</summary>
    public static SpvKernel Add(Vulkan vk, Fmt f)
    {
        var lane = f.Natural;
        var b = new Spv("add_" + f.Name, Threads, vk.Controls);
        var pa = b.Arg(0);
        var pb = b.Arg(1);
        var pc = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            int bytes = f.WordBits / 8;
            var wa = b.Load(wk, b.At(pa, gid, bytes));
            var wb = b.Load(wk, b.At(pb, gid, bytes));
            var wc = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var z = Codec.Unpack(b, f, wa, i, lane) + Codec.Unpack(b, f, wb, i, lane);
                wc = Codec.Pack(b, f, z, lane, i, wc);
            }

            b.Store(b.At(pc, gid, bytes), wc);
        });
        return b.Build();
    }

    /// <summary>output = op(x, y, z; a, b) over n values of f in lane `lane`: a thread per word (absent inputs never read).</summary>
    public static SpvKernel Map(Vulkan vk, Fmt f, Lane lane, MapOp op)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"map_{op.Name}_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var px = b.Arg(0);
        var py = b.Arg(1);
        var pz = b.Arg(2);
        var po = b.Arg(3);
        var n = b.Arg(4, K.U32);
        var a = b.Arg(5, k);
        var pb = b.Arg(6, k);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            int bytes = f.WordBits / 8;
            var wx = op.Inputs >= 1 ? b.Load(wk, b.At(px, gid, bytes)) : b.CBits(wk, 0);
            var wy = op.Inputs >= 2 ? b.Load(wk, b.At(py, gid, bytes)) : wx;
            var wz = op.Inputs >= 3 ? b.Load(wk, b.At(pz, gid, bytes)) : wx;
            var wo = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var x = Codec.Unpack(b, f, wx, i, lane);
                var y = op.Inputs >= 2 ? Codec.Unpack(b, f, wy, i, lane) : x;
                var z = op.Inputs >= 3 ? Codec.Unpack(b, f, wz, i, lane) : x;
                wo = Codec.Pack(b, f, Formula(b, k, op.Name, x, y, z, a, pb), lane, i, wo, wrap: !f.IsFloat);
            }

            b.Store(b.At(po, gid, bytes), wo);
        });
        return b.Build();
    }

    /// <summary>output[r, c] = op(x[r, c], v[c]) over rows × cols values of f (not a block format): a thread per output word,
    /// each of its values' row vector entry read where it is stored.</summary>
    public static SpvKernel MapRow(Vulkan vk, Fmt f, Lane lane, MapOp op)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"maprow_{op.Name}_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var px = b.Arg(0);
        var pv = b.Arg(1);
        var po = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            int bytes = f.WordBits / 8;
            var wx = b.Load(wk, b.At(px, gid, bytes));
            var wo = b.CBits(wk, 0);
            var zero = b.C(k, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var at = gid * (double)f.Per + i;
                var x = Codec.Unpack(b, f, wx, i, lane);
                var v = Codec.UnpackAt(b, f, pv, at % cols, lane);
                wo = Codec.Pack(b, f, Formula(b, k, op.Name, x, v, x, zero, zero), lane, i, wo, wrap: !f.IsFloat);
            }

            b.Store(b.At(po, gid, bytes), wo);
        });
        return b.Build();
    }
}
