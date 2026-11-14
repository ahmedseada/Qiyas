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
            if (w) return Exp64(b, x, lo, hi);
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

        // e^x in f64, Exps.Exp's steps (the definition): n by 32/ln 2, r in two parts, u = r·P(r), 2^(j/32) as hi + lo by a
        // tree of selects on j's bits, (hi + fma(hi, u, lo))·2^q, the scale in two halves.
        private static V Exp64(Spv b, V x, double lo, double hi)
        {
            var c = b.Min(b.Max(x, b.D(lo)), b.D(hi));
            var n = b.RoundEven(c * Exps.ExpInv32);
            var r = b.Fma(n, b.D(-Exps.Ln2Hi32), c);
            r = b.Fma(n, b.D(-Exps.Ln2Lo32), r);
            var p = b.D(Exps.ExpP64[0]);
            for (int k = 1; k < Exps.ExpP64.Length; k++) p = b.Fma(p, r, b.D(Exps.ExpP64[k]));
            var u = p * r;
            var nn = b.Convert(n, K.S64);
            var j = b.Convert(nn, K.U32) & 31UL;
            V Table(double[] t)
            {
                var level = t.Select(v => b.D(v)).ToArray();
                for (int bit = 0; level.Length > 1; bit++)
                {
                    var set = (j & (ulong)(1 << bit)) != 0;
                    var next = new V[level.Length / 2];
                    for (int m = 0; m < next.Length; m++) next[m] = b.Select(set, level[2 * m + 1], level[2 * m]);
                    level = next;
                }

                return level[0];
            }

            var th = Table(Exps.Exp2Hi);
            var tl = Table(Exps.Exp2Lo);
            var wv = th + b.Fma(th, u, tl);
            var ni = nn >> 5;
            var half = ni >> 1;
            var s1 = b.Bitcast((half + 1023) << 52, K.F64);
            var s2 = b.Bitcast(((ni - half) + 1023) << 52, K.F64);
            var y = wv * s1 * s2;
            y = b.Select(x < lo, b.D(0), y);
            y = b.Select(x > hi, b.D(double.PositiveInfinity), y);
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

        /// <summary>σ by t = e^(−|x|): (x ≥ 0 ? 1 : t) / (t + 1), no e^(+large) (Funcs.Sigmoid's steps); a product with σ
        /// takes SigmoidS.</summary>
        public static V Sigmoid(Spv b, V x)
        {
            var t = Exp(b, -b.Abs(x));
            // x ≥ 0 by the sign bit in f64 (an integer test; −0 and a NaN take the other numerator to the same value).
            var pos = x.Kind == K.F64 ? (b.Bitcast(x, K.U64) >> 63) == 0UL : x >= 0.0;
            return b.Select(pos, b.C(x.Kind, 1), t) / (t + 1.0);
        }

        /// <summary>σ(x) and σ(−x) from one t = e^(−|x|) (Sigmoid's steps for each: the same values), and t.</summary>
        public static (V S, V Sn, V T) SigmoidPair(Spv b, V x)
        {
            var t = Exp(b, -b.Abs(x));
            var d = t + 1.0;
            V Pos(V v) => v.Kind == K.F64 ? (b.Bitcast(v, K.U64) >> 63) == 0UL : v >= 0.0;
            var one = b.C(x.Kind, 1);
            return (b.Select(Pos(x), one, t) / d, b.Select(Pos(-x), one, t) / d, t);
        }

        /// <summary>The f32 lane's σ(x) as σ_s·K·K: where x is below −87, σ_s = σ(x)·2^128 (from e^(−|x|)·2^128, a normal
        /// number) and K = 2^−64; else σ_s = σ(x), K = 1 (Funcs.SigmoidS).</summary>
        public static (V S, V K) SigmoidS(Spv b, V x)
        {
            var neg = x < -87.0;
            var big = b.Abs(x) > 87.0;
            var ts = ExpOff32(b, -b.Abs(x), b.Select(big, b.S(128), b.S(0)));
            var s = b.Select(x >= 0.0, b.F(1), ts) / (b.Select(big, b.F(0), ts) + 1.0);
            return (s, b.Select(neg, b.F(5.421010862427522E-20), b.F(1)));
        }

        /// <summary>σ(v)·σ(−v) = t/(1 + t)², t = e^(−|v|), as it is (BellS's form where |v| is at most 87).</summary>
        public static V Bell(Spv b, V v)
        {
            var t = Exp(b, -b.Abs(v));
            var d = t + 1.0;
            return t / (d * d);
        }

        /// <summary>The f32 lane's σ(v)·σ(−v) = t/(1 + t)², t = e^(−|v|), as F·K·K: t kept 2^128 above where |v| is past 87
        /// (Funcs.BellS).</summary>
        public static (V F, V K) BellS(Spv b, V v)
        {
            var big = b.Abs(v) > 87.0;
            var ts = ExpOff32(b, -b.Abs(v), b.Select(big, b.S(128), b.S(0)));
            var d = b.Select(big, b.F(0), ts) + 1.0;
            return (ts / (d * d), b.Select(big, b.F(5.421010862427522E-20), b.F(1)));
        }

        /// <summary>e^v·2^off in the f32 lane (off 128, 0 or −128, an s32), Exp's steps with the scale's exponent offset
        /// split between its halves (±64 each) and v held to [−192.6, 177.3] (Exps.ExpOff). Where off is 128 v is below
        /// −87, where −128 above 87.</summary>
        public static V ExpOff32(Spv b, V v, V off)
        {
            var c = b.Min(b.Max(v, b.F(-192.6)), b.F(177.3));
            var n = b.RoundEven(c * 1.44269504);
            var r = b.Fma(n, b.F(-0.693359375), c);
            r = b.Fma(n, b.F(2.12194440e-4), r);
            var p = b.F(1f / 5040);
            foreach (float kc in new[] { 1f / 720, 1f / 120, 1f / 24, 1f / 6, 0.5f, 1f, 1f }) p = b.Fma(p, r, b.F(kc));
            var ni = b.Convert(n, K.S32);
            var half = ni >> 1;
            var o2 = off >> 1;
            var s1 = b.Bitcast((half + o2 + 127) << 23, K.F32);
            var s2 = b.Bitcast(((ni - half) + o2 + 127) << 23, K.F32);
            var y = p * s1 * s2;
            y = b.Select(v < -192.6, b.F(0), y);
            y = b.Select(v > 177.3, b.F(double.PositiveInfinity), y);
            return b.Select(b.IsNan(v), v, y);
        }

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
            var (s, sn, _) = Fn.SigmoidPair(b, u2);
            var du = b.Fma(C(3 * 0.044715), x2, C(1)) * 0.7978845608028654;
            return b.Fma((x * 2.0) * (s * sn), du, s);
        }

        public V SiluGrad(V x) { var (s, sn, _) = Fn.SigmoidPair(b, x); return s * b.Fma(x, sn, C(1)); }

        // |x| finite (false for a NaN); x nonzero and not a NaN. In f64 on the bits (integer tests at the full rate where the
        // device's f64 compares run at its f64 rate: the same results).
        private const ulong Abs64 = 0x7FFFFFFFFFFFFFFFUL, Inf64 = 0x7FF0000000000000UL;

        public V Finite(V x) => k == K.F64 ? (b.Bitcast(x, K.U64) & Inf64) != Inf64 : b.Abs(x) < double.PositiveInfinity;

        public V NonZero(V x) => k == K.F64
            ? ((b.Bitcast(x, K.U64) << 1) != 0UL) & ((b.Bitcast(x, K.U64) & Abs64) <= Inf64)
            : (x > 0.0) | (x < 0.0);

        // f·y for a factor f whose true value is nonzero where `nonzero`, of the sign of `sign`: where y is 0 or ±∞ the product
        // is sign·y (±0, ±∞), not f·y (no 0·∞; Map.cs's Times).
        public V Times(V f, V y, V nonzero, V sign)
        {
            var edge = k == K.F64
                ? ((b.Bitcast(y, K.U64) << 1) == 0UL) | ((b.Bitcast(y, K.U64) & Abs64) == Inf64)
                : (y == 0.0) | (b.Abs(y) == double.PositiveInfinity);
            return b.Select(nonzero & edge, sign * y, f * y);
        }

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
        if (k == K.F32 && Scaled32(b, e, op, x, y, z, a, pb, false) is V near)
        {
            // The f32 lane's scaled forms where an input is past ±87 (their factor below the normal numbers) or the plain
            // product is a NaN (Times'), the same forms at scale 1 and the plain product elsewhere (the same values), so a
            // workgroup none of whose values is far skips them.
            var result = b.Local(near);
            b.If(Far32(b, e, op, x, y, a, pb, near), () => result.V = Scaled32(b, e, op, x, y, z, a, pb, true)!.Value);
            return result.V;
        }
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
            case "silu": return x * e.Sigmoid(x);
            case "leaky_relu": return b.Select(x > 0, x, a * x);
            case "pow": return e.Pow(x, a);
            case "mask_scale": return b.Select(y != 0, x * a, e.C(0));
            case "gated_silu": return e.Times(x * e.Sigmoid(x), y, e.Finite(x) & e.NonZero(x), x);
            case "gated_gelu": return e.Gelu(x) * y;
            case "sqrt_bwd": return y / (e.C(2) * b.Sqrt(x));
            case "inv_sqrt_bwd": { var s = x + a; var r = e.C(1) / b.Sqrt(s); return (e.C(-0.5) * (r / s)) * y; }
            case "exp_bwd": return e.Times(e.Exp(x), y, e.Finite(x), e.C(1));
            case "log_bwd": return y / x;
            case "sin_bwd": return Fn.Cos(b, x) * y;
            case "cos_bwd": return -(Fn.Sin(b, x) * y);
            case "tanh_bwd": return e.Times(e.C(4) * Fn.Bell(b, e.C(2) * x), y, e.Finite(x), e.C(1));
            case "softcap": return a * Fn.Tanh(b, (pb * x) / a);
            case "softcap_bwd": { var u = (pb * x) / a; return e.Times((e.C(4) * Fn.Bell(b, e.C(2) * u)) * pb, y, e.Finite(u), pb); }
            case "sigmoid_bwd": return e.Times(Fn.Bell(b, x), y, e.Finite(x), e.C(1));
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
            case "gated_silu_bwd_y": return e.Times(x * e.Sigmoid(x), z, e.Finite(x) & e.NonZero(x), x);
            case "gated_gelu_bwd_x": return (e.GeluGrad(x) * y) * z;
            case "gated_gelu_bwd_y": return e.Gelu(x) * z;
            case "gated_tanh_sigmoid": return Fn.Tanh(b, x) * e.Sigmoid(y);
            case "gated_tanh_sigmoid_bwd_x": { var sech2 = e.C(4) * Fn.Bell(b, e.C(2) * x); return e.Times(sech2 * e.Sigmoid(y), z, e.Finite(x) & (y > double.NegativeInfinity), e.C(1)); }
            case "gated_tanh_sigmoid_bwd_y": { var t = Fn.Tanh(b, x); return e.Times(t * Fn.Bell(b, y), z, e.NonZero(x) & e.Finite(y), t); }
            case "add_gelu": return e.Gelu(x + y);
            case "clip_factor": return b.Select(x > 0, b.Min(e.C(1), a / b.Sqrt(x)), e.C(1));
            case "sigmoid_focal":
            case "sigmoid_focal_bwd":
            {
                // Abstraction/Map.cs's parts: BCE with log1p by Goldberg's form, 1 − p_t, (1 − p_t)^γ (1 where γ is 0), α_t.
                V one = e.C(1), zero = e.C(0);
                var (s, sn, u0) = Fn.SigmoidPair(b, x);                  // σ(x), σ(−x) and the BCE's e^(−|x|): one exp
                var w = one + u0;
                var dw = w - one;
                var lp = b.Select(dw > 0, Fn.Log(b, w) * u0 / dw, u0);
                var ce = b.Fma(-x, y, b.Max(x, zero)) + lp;
                V u = one - y;
                var q = y * sn + u * s;
                var qg = b.Select(pb == 0, one, e.Pow(q, pb));
                var at = a * y + (one - a) * u;
                // A hard target the score is far past (Cpu/Map.cs's FocalFar): |x| past 20 in f32, 708 in f64.
                var ax = b.Abs(x);
                var beyond = ax > (x.Kind == K.F32 ? 20.0 : 708.0);
                var low = beyond & (x < 0.0) & (y == 0.0);
                var far = low | (beyond & (x > 0.0) & (y == 1.0));
                // e^(−(γ+1)|x|), in the f32 lane kept 2^128 above where below e^−87 and brought down last (K twice); made only
                // where a value is far (a branch a workgroup none of whose values is far skips).
                bool f32 = x.Kind == K.F32;
                V FarP(out V kk)
                {
                    var v = -(pb + one) * b.Abs(x);
                    if (!f32) { kk = one; return e.Exp(v); }
                    var big = v < -87.0;
                    kk = b.Select(big, b.F(5.421010862427522E-20), b.F(1));
                    return Fn.ExpOff32(b, v, b.Select(big, b.S(128), b.S(0)));
                }

                if (op == "sigmoid_focal")
                {
                    var loss = b.Local(at * qg * ce);
                    b.If(far, () => { var p = FarP(out var kk); loss.V = ((b.Select(low, one - a, a) * p) * kk) * kk; });
                    return loss.V;
                }
                var d = u * s - y * sn;
                var second = pb * ce * (qg / q) * (y + y - one) * (s * sn);
                second = b.Select(q == 0, zero, b.Select(pb == 0, zero, second));
                var grad = b.Local(at * (d * qg - second) * z);
                b.If(far, () =>
                {
                    var p = FarP(out var kk);
                    var g1 = one + pb;
                    var factor = b.Select(low, (one - a) * g1, -(a * g1));
                    grad.V = (e.Times(factor * p, z, e.NonZero(factor), factor) * kk) * kk;
                });
                return grad.V;
            }

            default: throw new ArgumentException(op);
        }
    }

    // The f32 lane's forms whose factor may fall below the lane's normal numbers (the definition: Funcs' F32 forms): σ, its
    // bell σ(v)·σ(−v) and e^x kept 2^128 from it (SigmoidS, BellS, ExpOff32), the product made, then brought back by K·K;
    // a product with 0 or ±∞ by Times (no 0·∞). Null for the other formulas.
    // Where an input is past ±87, or the plain product with Times' operand is a NaN (only there can Times' value differ
    // from it), the forms as defined; elsewhere the same forms at scale 1 and the plain product (the same values).
    // (A product with Times differs from it only where it is a NaN: f·0 and f·±∞ are Times' ±0 and ±∞ wherever they are
    // not, f's sign being Times' sign's.)
    private static V Far32(Spv b, Emit e, string op, V x, V y, V a, V pb, V near)
    {
        V Big(V v) => b.Abs(v) > 87.0;
        var nan = b.IsNan(near);
        return op switch
        {
            "silu" or "silu_bwd" or "gated_silu_bwd_x" => Big(x),
            "gated_silu" or "exp_bwd" or "sigmoid_bwd" or "gated_silu_bwd_y" => Big(x) | nan,
            "tanh_bwd" => Big(e.C(2) * x) | nan,
            "softcap_bwd" => Big(e.C(2) * ((pb * x) / a)) | nan,
            "gated_tanh_sigmoid_bwd_x" => Big(e.C(2) * x) | Big(y) | nan,
            "gated_tanh_sigmoid_bwd_y" => Big(y) | nan,
            _ => throw new ArgumentException(op),
        };
    }

    // far: the forms as defined (SigmoidS, BellS, ExpOff32, brought down by K·K; Times); else the same forms at scale 1
    // (Sigmoid, Bell, Exp) and the plain product: the same values where no input is past ±87 and the product no NaN.
    private static V? Scaled32(Spv b, Emit e, string op, V x, V y, V z, V a, V pb, bool far)
    {
        V Down(V r, V? kk) => kk is V k1 ? (r * k1) * k1 : r;
        (V, V?) Sig(V v) { if (far) { var (s, k1) = Fn.SigmoidS(b, v); return (s, k1); } return (Fn.Sigmoid(b, v), null); }
        (V, V?) BellOf(V v) { if (far) { var (f, k1) = Fn.BellS(b, v); return (f, k1); } return (Fn.Bell(b, v), null); }
        V Tm(V f, V yv, Func<V> nonzero, V sign) => far ? e.Times(f, yv, nonzero(), sign) : f * yv;
        switch (op)
        {
            case "silu": { var (s, kk) = Sig(x); return Down(x * s, kk); }
            case "gated_silu": { var (s, kk) = Sig(x); return Down(Tm(x * s, y, () => e.Finite(x) & e.NonZero(x), x), kk); }
            case "gated_silu_bwd_y": { var (s, kk) = Sig(x); return Down(Tm(x * s, z, () => e.Finite(x) & e.NonZero(x), x), kk); }
            case "silu_bwd":
            {
                if (!far) { var (s0, sn0, _) = Fn.SigmoidPair(b, x); return (s0 * b.Fma(x, sn0, e.C(1))) * y; }
                var (s, kk) = Sig(x);
                return Down((s * b.Fma(x, Fn.Sigmoid(b, -x), e.C(1))) * y, kk);
            }
            case "gated_silu_bwd_x":
            {
                if (!far) { var (s0, sn0, _) = Fn.SigmoidPair(b, x); return ((s0 * b.Fma(x, sn0, e.C(1))) * y) * z; }
                var (s, kk) = Sig(x);
                return Down(((s * b.Fma(x, Fn.Sigmoid(b, -x), e.C(1))) * y) * z, kk);
            }
            case "exp_bwd":
            {
                if (!far) return Tm(e.Exp(x), y, () => e.Finite(x), e.C(1));
                var low = x < -87.0;
                var high = x > 87.0;
                var t = Fn.ExpOff32(b, x, b.Select(low, b.S(128), b.Select(high, b.S(-128), b.S(0))));
                var kk = b.Select(low, b.F(5.421010862427522E-20), b.Select(high, b.F(1.8446744073709552E+19), b.F(1)));
                return Down(Tm(t, y, () => e.Finite(x), e.C(1)), kk);
            }
            case "sigmoid_bwd": { var (f, kk) = BellOf(x); return Down(Tm(f, y, () => e.Finite(x), e.C(1)), kk); }
            case "tanh_bwd": { var (f, kk) = BellOf(e.C(2) * x); return Down(Tm(e.C(4) * f, y, () => e.Finite(x), e.C(1)), kk); }
            case "softcap_bwd": { var u = (pb * x) / a; var (f, kk) = BellOf(e.C(2) * u); return Down(Tm((e.C(4) * f) * pb, y, () => e.Finite(u), pb), kk); }
            case "gated_tanh_sigmoid_bwd_x":
            {
                var (f, ka) = BellOf(e.C(2) * x);
                var (s, kb) = Sig(y);
                return Down(Down(Tm((e.C(4) * f) * s, z, () => e.Finite(x) & (y > double.NegativeInfinity), e.C(1)), ka), kb);
            }
            case "gated_tanh_sigmoid_bwd_y": { var t = Fn.Tanh(b, x); var (f, kk) = BellOf(y); return Down(Tm(t * f, z, () => e.NonZero(x) & e.Finite(y), t), kk); }
            default: return null;
        }
    }

    // ---------------------------------------------------------------- the kernels

    // Words that hold n values of f (a u32 value).
    private static V WordCount(Spv b, Fmt f, V n) =>
        f.WordBits == 64 ? n : b.Convert((b.Convert(n, K.U64) * (double)f.Bits + 31.0) >> 5, K.U32);

    /// <summary>The stored words a vectorized element-wise kernel's invocation takes (Words: 16 bytes of them).</summary>
    public static int VectorWords(Fmt f) => 16 / (f.WordBits / 8);

    /// <summary>Whether an element-wise kernel takes 16 bytes of words an invocation (Words' vector) where the buffers allow:
    /// for a type of several values a word, whose word an invocation leaves little in flight (float16 map:copy 0.116 ms
    /// against 0.123, int8 add 0.059 against 0.075); not where a value is a word (float32 map:copy 0.228 against 0.215:
    /// a quarter of the invocations).</summary>
    public static bool Vectorizes(Fmt f) => f.Per > 1;

    // A thread's words of an element-wise kernel: one word (gid); or, vectorized, VectorWords consecutive words read and
    // written 16 bytes at a time where all of them are below the count (the last thread's words a word at a time). Each word
    // made by `word` from its inputs' words: the same words either way.
    private static void Words(Spv b, Fmt f, V n, bool vector, V[] inputs, V output, Func<V[], V> word) =>
        Words(b, f, n, vector, inputs, output, (w, _) => word(w));

    // As above, `word` also given the word's index.
    private static void Words(Spv b, Fmt f, V n, bool vector, V[] inputs, V output, Func<V[], V, V> word)
    {
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        int bytes = f.WordBits / 8;
        var count = WordCount(b, f, n);
        void One(V w) => b.Store(b.At(output, w, bytes), word([.. inputs.Select(p => b.Load(wk, b.At(p, w, bytes)))], w));
        if (!vector)
        {
            b.If(gid < count, () => One(gid));
            return;
        }

        int per = VectorWords(f);
        var first = gid * (double)per;
        b.If(first + (per - 1.0) < count, () =>
        {
            var loaded = inputs.Select(p => b.LoadN(wk, per, b.At(p, gid, 16))).ToArray();
            var made = new V[per];
            for (int q = 0; q < per; q++) made[q] = word([.. loaded.Select(l => l[q])], first + q);
            b.StoreN(b.At(output, gid, 16), made);
        }, () => b.For(first, b.Min(first + (double)per, count), One));
    }

    /// <summary>c = a + b over n values: a thread per word (or, vector, per 16 bytes of words: VectorWords), each value in the
    /// type's own lane.</summary>
    public static SpvKernel Add(Vulkan vk, Fmt f, bool vector = false)
    {
        var lane = f.Natural;
        var b = new Spv("add_" + f.Name + (vector ? "_v" : ""), Threads, vk.Controls);
        var pa = b.Arg(0);
        var pb = b.Arg(1);
        var pc = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var wk = Codec.WordKind(f);
        Words(b, f, n, vector, [pa, pb], pc, w =>
        {
            var wc = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var z = Codec.Unpack(b, f, w[0], i, lane) + Codec.Unpack(b, f, w[1], i, lane);
                wc = Codec.Pack(b, f, z, lane, i, wc);
            }

            return wc;
        });
        return b.Build();
    }

    /// <summary>output = op(x, y, z; a, b) over n values of f in lane `lane`: a thread per word (or, vector, per 16 bytes of
    /// words: VectorWords; absent inputs never read).</summary>
    public static SpvKernel Map(Vulkan vk, Fmt f, Lane lane, MapOp op, bool vector = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"map_{op.Name}_{f.Name}_{lane.Name}{(vector ? "_v" : "")}", Threads, vk.Controls);
        var px = b.Arg(0);
        var py = b.Arg(1);
        var pz = b.Arg(2);
        var po = b.Arg(3);
        var n = b.Arg(4, K.U32);
        var a = b.Arg(5, k);
        var pb = b.Arg(6, k);
        var wk = Codec.WordKind(f);
        V[] inputs = [.. new[] { px, py, pz }.Take(op.Inputs)];
        Words(b, f, n, vector, inputs, po, w =>
        {
            var wx = op.Inputs >= 1 ? w[0] : b.CBits(wk, 0);
            var wy = op.Inputs >= 2 ? w[1] : wx;
            var wz = op.Inputs >= 3 ? w[2] : wx;
            var wo = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var x = Codec.Unpack(b, f, wx, i, lane);
                var y = op.Inputs >= 2 ? Codec.Unpack(b, f, wy, i, lane) : x;
                var z = op.Inputs >= 3 ? Codec.Unpack(b, f, wz, i, lane) : x;
                wo = Codec.Pack(b, f, Formula(b, k, op.Name, x, y, z, a, pb), lane, i, wo, wrap: !f.IsFloat);
            }

            return wo;
        });
        return b.Build();
    }

    /// <summary>output[r, c] = op(x[r, c], v[c]) over rows × cols values of f (not a block format): a thread per output word
    /// (or, vector, per 16 bytes of words: VectorWords), each of its values' row vector entry read where it is stored.</summary>
    public static SpvKernel MapRow(Vulkan vk, Fmt f, Lane lane, MapOp op, bool vector = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"maprow_{op.Name}_{f.Name}_{lane.Name}{(vector ? "_v" : "")}", Threads, vk.Controls);
        var px = b.Arg(0);
        var pv = b.Arg(1);
        var po = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var wk = Codec.WordKind(f);
        var zero = b.C(k, 0);
        Words(b, f, n, vector, [px], po, (w, word) =>
        {
            var wo = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var at = word * (double)f.Per + i;
                var x = Codec.Unpack(b, f, w[0], i, lane);
                var v = Codec.UnpackAt(b, f, pv, at % cols, lane);
                wo = Codec.Pack(b, f, Formula(b, k, op.Name, x, v, x, zero, zero), lane, i, wo, wrap: !f.IsFloat);
            }

            return wo;
        });
        return b.Build();
    }
}
