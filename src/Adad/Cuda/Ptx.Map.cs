// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device side of the element-wise family: the transcendental functions as PTX (the same tables and reductions as
// Funcs, evaluated by fused multiply-adds on the device), the formulas of the catalogue emitted over a lane, and the
// map kernel: a thread per stored word, each value unpacked into the lane, the formula applied, the result packed.
using System.Globalization;
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    // ---------------------------------------------------------------- functions on a float lane register

    internal static class Fn
    {
        private static string L(Lane lane, double v) => lane.Literal(v);

        private static string Poly(PtxBuilder b, Lane lane, string t, double[] coefficients, int terms, int first)
        {
            // Horner from the highest kept term down to `first`: p = c[terms−1]; p = p·t + c[k] …
            string p = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {p}, {L(lane, coefficients[terms - 1])};");
            for (int k = terms - 2; k >= first; k--) b.L($"fma.rn.{lane.Op} {p}, {p}, {t}, {L(lane, coefficients[k])};");
            return p;
        }

        public static string Log(PtxBuilder b, Lane lane, string x)
        {
            bool w = lane.Wide;
            string op = lane.Op;
            string tiny = b.P(), xs = lane.NewReg(b), scaled = lane.NewReg(b);
            b.L($"setp.lt.{op} {tiny}, {x}, {L(lane, w ? 2.2250738585072014e-308 : 1.17549435e-38)};");
            b.L($"mul.rn.{op} {scaled}, {x}, {L(lane, w ? 4503599627370496.0 : 8388608.0)};");
            b.L($"selp.{op} {xs}, {scaled}, {x}, {tiny};");
            string bits = w ? b.RD() : b.R(), e = b.R(), mbits = w ? b.RD() : b.R(), m = lane.NewReg(b);
            if (w)
            {
                string e64 = b.RD();
                b.L($"mov.b64 {bits}, {xs};");
                b.L($"shr.s64 {e64}, {bits}, 52;");
                b.L($"cvt.s32.s64 {e}, {e64};");
                b.L($"sub.s32 {e}, {e}, 1023;");
                b.L($"and.b64 {mbits}, {bits}, 0x000FFFFFFFFFFFFF;");
                b.L($"or.b64 {mbits}, {mbits}, 0x3FF0000000000000;");
                b.L($"mov.b64 {m}, {mbits};");
            }
            else
            {
                b.L($"mov.b32 {bits}, {xs};");
                b.L($"shr.s32 {e}, {bits}, 23;");
                b.L($"sub.s32 {e}, {e}, 127;");
                b.L($"and.b32 {mbits}, {bits}, 0x007FFFFF;");
                b.L($"or.b32 {mbits}, {mbits}, 0x3F800000;");
                b.L($"mov.b32 {m}, {mbits};");
            }

            string big = b.P(), half = lane.NewReg(b), e1 = b.R();
            b.L($"setp.gt.{op} {big}, {m}, {L(lane, 1.4142135623730951)};");
            b.L($"mul.rn.{op} {half}, {m}, {L(lane, 0.5)};");
            b.L($"selp.{op} {m}, {half}, {m}, {big};");
            b.L($"add.s32 {e1}, {e}, 1;");
            b.L($"selp.s32 {e}, {e1}, {e}, {big};");
            b.L($"sub.s32 {e1}, {e}, {(w ? 52 : 23)};");
            b.L($"selp.s32 {e}, {e1}, {e}, {tiny};");
            string num = lane.NewReg(b), den = lane.NewReg(b), t = lane.NewReg(b), t2 = lane.NewReg(b);
            b.L($"sub.rn.{op} {num}, {m}, {L(lane, 1)};");
            b.L($"add.rn.{op} {den}, {m}, {L(lane, 1)};");
            b.L($"div.rn.{op} {t}, {num}, {den};");
            b.L($"mul.rn.{op} {t2}, {t}, {t};");
            string p = Poly(b, lane, t2, Funcs.Atanh, w ? Funcs.AtanhTerms64 : Funcs.AtanhTerms32, 0);
            string ef = lane.NewReg(b), y = lane.NewReg(b), pt = lane.NewReg(b);
            b.L($"cvt.rn.{op}.s32 {ef}, {e};");
            b.L($"mul.rn.{op} {pt}, {p}, {t};");
            b.L($"fma.rn.{op} {y}, {ef}, {L(lane, w ? Funcs.Ln2Lo : Funcs.Ln2LoF)}, {pt};");
            b.L($"fma.rn.{op} {y}, {ef}, {L(lane, w ? Funcs.Ln2Hi : Funcs.Ln2HiF)}, {y};");
            string pz = b.P(), pn = b.P(), pi = b.P(), pnan = b.P();
            b.L($"setp.eq.{op} {pz}, {x}, {L(lane, 0)};");
            b.L($"selp.{op} {y}, {L(lane, double.NegativeInfinity)}, {y}, {pz};");
            b.L($"setp.lt.{op} {pn}, {x}, {L(lane, 0)};");
            b.L($"selp.{op} {y}, {L(lane, double.NaN)}, {y}, {pn};");
            b.L($"setp.eq.{op} {pi}, {x}, {L(lane, double.PositiveInfinity)};");
            b.L($"selp.{op} {y}, {x}, {y}, {pi};");
            b.L($"testp.notanumber.{op} {pnan}, {x};");
            b.L($"selp.{op} {y}, {x}, {y}, {pnan};");
            return y;
        }

        private static (string R, string K) ReduceHalfPi(PtxBuilder b, Lane lane, string x)
        {
            bool w = lane.Wide;
            string op = lane.Op, k = lane.NewReg(b), r = lane.NewReg(b), ki = b.R();
            b.L($"mul.rn.{op} {k}, {x}, {L(lane, 0.63661977236758134)};");
            b.L($"cvt.rni.{op}.{op} {k}, {k};");
            b.L($"fma.rn.{op} {r}, {k}, {L(lane, w ? -Funcs.PiHalf1 : -Funcs.PiHalf1F)}, {x};");
            b.L($"fma.rn.{op} {r}, {k}, {L(lane, w ? -Funcs.PiHalf2 : -Funcs.PiHalf2F)}, {r};");
            b.L($"fma.rn.{op} {r}, {k}, {L(lane, w ? -Funcs.PiHalf3 : -Funcs.PiHalf3F)}, {r};");
            b.L($"cvt.rni.s32.{op} {ki}, {k};");
            return (r, ki);
        }

        // x − k·π/2 as an unevaluated pair r + lo: the first part's product exact (π/2's leading part has trailing zeros),
        // the second's sum with it split exactly (two-sum), the third part's product added to the low part.
        private static (string R, string Lo, string K) ReduceHalfPiPair(PtxBuilder b, Lane lane, string x)
        {
            bool w = lane.Wide;
            string op = lane.Op, k = lane.NewReg(b), r1 = lane.NewReg(b), p2 = lane.NewReg(b), r = lane.NewReg(b), bv = lane.NewReg(b), t = lane.NewReg(b), lo = lane.NewReg(b), ki = b.R();
            b.L($"mul.rn.{op} {k}, {x}, {L(lane, 0.63661977236758134)};");
            b.L($"cvt.rni.{op}.{op} {k}, {k};");
            b.L($"fma.rn.{op} {r1}, {k}, {L(lane, w ? -Funcs.PiHalf1 : -Funcs.PiHalf1F)}, {x};");
            b.L($"mul.rn.{op} {p2}, {k}, {L(lane, w ? -Funcs.PiHalf2 : -Funcs.PiHalf2F)};");
            b.L($"add.rn.{op} {r}, {r1}, {p2};");
            // lo = (r1 − (r − bv)) + (p2 − bv), bv = r − r1: the sum's rounding error exactly.
            b.L($"sub.rn.{op} {bv}, {r}, {r1};");
            b.L($"sub.rn.{op} {t}, {r}, {bv};");
            b.L($"sub.rn.{op} {t}, {r1}, {t};");
            b.L($"sub.rn.{op} {lo}, {p2}, {bv};");
            b.L($"add.rn.{op} {lo}, {lo}, {t};");
            b.L($"fma.rn.{op} {lo}, {k}, {L(lane, w ? -Funcs.PiHalf3 : -Funcs.PiHalf3F)}, {lo};");
            b.L($"cvt.rni.s32.{op} {ki}, {k};");
            return (r, lo, ki);
        }

        private static string Trig(PtxBuilder b, Lane lane, string x, bool sine)
        {
            bool w = lane.Wide;
            string op = lane.Op;
            var (r, lo, k) = ReduceHalfPiPair(b, lane, x);
            string r2 = lane.NewReg(b);
            b.L($"mul.rn.{op} {r2}, {r}, {r};");
            int terms = w ? Funcs.SinTerms64 : Funcs.SinTerms32;
            // sin(r + lo) = r + (lo·(1 − r²/2) + r³·(−1/6 + r²/120 − …)) and cos(r + lo) = 1 + (r²·(−1/2 + r²/24 − …) − r·lo):
            // the leading term exact, one rounding of a small correction that carries the reduction's low part.
            string s = Poly(b, lane, r2, Funcs.SinSeries, terms, 1), c = Poly(b, lane, r2, Funcs.CosSeries, terms, 1), r3 = lane.NewReg(b), m = lane.NewReg(b), rl = lane.NewReg(b);
            b.L($"mul.rn.{op} {r3}, {r2}, {r};");
            b.L($"mul.rn.{op} {m}, {r2}, {L(lane, -0.5)};");
            b.L($"fma.rn.{op} {m}, {m}, {lo}, {lo};");
            b.L($"fma.rn.{op} {s}, {s}, {r3}, {m};");
            b.L($"add.rn.{op} {s}, {s}, {r};");
            b.L($"mul.rn.{op} {rl}, {r}, {lo};");
            b.L($"neg.{op} {rl}, {rl};");
            b.L($"fma.rn.{op} {c}, {c}, {r2}, {rl};");
            b.L($"add.rn.{op} {c}, {c}, {L(lane, 1)};");
            string q = b.R(), odd = b.P(), neg = b.P(), y = lane.NewReg(b), ns = lane.NewReg(b), ny = lane.NewReg(b), bad = b.P(), ax = lane.NewReg(b);
            b.L($"and.b32 {q}, {k}, 3;");
            b.L($"and.b32 {q}, {q}, 3;");
            string t = b.R();
            b.L($"and.b32 {t}, {q}, 1;");
            b.L($"setp.eq.u32 {odd}, {t}, 1;");
            b.L($"and.b32 {t}, {q}, 2;");
            b.L($"setp.eq.u32 {neg}, {t}, 2;");
            b.L($"neg.{op} {ns}, {s};");
            if (sine) b.L($"selp.{op} {y}, {c}, {s}, {odd};");
            else b.L($"selp.{op} {y}, {ns}, {c}, {odd};");
            b.L($"neg.{op} {ny}, {y};");
            b.L($"selp.{op} {y}, {ny}, {y}, {neg};");
            b.L($"abs.{op} {ax}, {x};");
            b.L($"setp.ge.{op} {bad}, {ax}, {L(lane, double.PositiveInfinity)};");
            b.L($"selp.{op} {y}, {L(lane, double.NaN)}, {y}, {bad};");
            return y;
        }

        public static string Sin(PtxBuilder b, Lane lane, string x) => Trig(b, lane, x, true);

        public static string Cos(PtxBuilder b, Lane lane, string x) => Trig(b, lane, x, false);

        public static string Expm1(PtxBuilder b, Lane lane, string x)
        {
            bool w = lane.Wide;
            string op = lane.Op;
            double lo = w ? -708.39 : -87.33654, hi = w ? 709.78 : 88.72283;
            string c = lane.NewReg(b), n = lane.NewReg(b), r = lane.NewReg(b);
            b.L($"max.{op} {c}, {x}, {L(lane, lo)};");
            b.L($"min.{op} {c}, {c}, {L(lane, hi)};");
            b.L($"mul.rn.{op} {n}, {c}, {L(lane, w ? 1.4426950408889634 : 1.44269504)};");
            b.L($"cvt.rni.{op}.{op} {n}, {n};");
            b.L($"fma.rn.{op} {r}, {n}, {L(lane, w ? -Funcs.Ln2Hi : -Funcs.Ln2HiF)}, {c};");
            b.L($"fma.rn.{op} {r}, {n}, {L(lane, w ? -Funcs.Ln2Lo : -Funcs.Ln2LoF)}, {r};");
            string q = Poly(b, lane, r, Funcs.InvFact, w ? Funcs.ExpTerms64 : Funcs.ExpTerms32, 1);
            b.L($"mul.rn.{op} {q}, {q}, {r};");
            string scale = lane.NewReg(b), sm1 = lane.NewReg(b), y = lane.NewReg(b);
            if (w)
            {
                string ni = b.RD(), sb = b.RD();
                b.L($"cvt.rni.s64.f64 {ni}, {n};");
                b.L($"add.s64 {sb}, {ni}, 1023;");
                b.L($"shl.b64 {sb}, {sb}, 52;");
                b.L($"mov.b64 {scale}, {sb};");
            }
            else
            {
                string ni = b.R(), sb = b.R();
                b.L($"cvt.rni.s32.f32 {ni}, {n};");
                b.L($"add.s32 {sb}, {ni}, 127;");
                b.L($"shl.b32 {sb}, {sb}, 23;");
                b.L($"mov.b32 {scale}, {sb};");
            }

            b.L($"sub.rn.{op} {sm1}, {scale}, {L(lane, 1)};");
            b.L($"fma.rn.{op} {y}, {q}, {scale}, {sm1};");
            string pl = b.P(), ph = b.P(), pn = b.P();
            b.L($"setp.lt.{op} {pl}, {x}, {L(lane, lo)};");
            b.L($"selp.{op} {y}, {L(lane, -1)}, {y}, {pl};");
            b.L($"setp.gt.{op} {ph}, {x}, {L(lane, hi)};");
            b.L($"selp.{op} {y}, {L(lane, double.PositiveInfinity)}, {y}, {ph};");
            b.L($"testp.notanumber.{op} {pn}, {x};");
            b.L($"selp.{op} {y}, {x}, {y}, {pn};");
            return y;
        }

        public static string Tanh(PtxBuilder b, Lane lane, string x)
        {
            string op = lane.Op, a = lane.NewReg(b), two = lane.NewReg(b), e2 = lane.NewReg(b), y = lane.NewReg(b), pn = b.P();
            b.L($"max.{op} {a}, {x}, {L(lane, -20)};");
            b.L($"min.{op} {a}, {a}, {L(lane, 20)};");
            b.L($"mul.rn.{op} {a}, {a}, {L(lane, 2)};");
            string e = Expm1(b, lane, a);
            b.L($"add.rn.{op} {e2}, {e}, {L(lane, 2)};");
            b.L($"div.rn.{op} {y}, {e}, {e2};");
            b.L($"testp.notanumber.{op} {pn}, {x};");
            b.L($"selp.{op} {y}, {x}, {y}, {pn};");
            _ = two;
            return y;
        }

        public static string Sigmoid(PtxBuilder b, Lane lane, string x)
        {
            string op = lane.Op, nx = lane.NewReg(b), d = lane.NewReg(b), y = lane.NewReg(b);
            b.L($"neg.{op} {nx}, {x};");
            string e = Codec.Exp(b, lane, nx);
            b.L($"add.rn.{op} {d}, {e}, {L(lane, 1)};");
            b.L($"div.rn.{op} {y}, {L(lane, 1)}, {d};");
            return y;
        }

        public static string Erf(PtxBuilder b, Lane lane, string x)
        {
            bool w = lane.Wide;
            string op = lane.Op, ax = lane.NewReg(b), x2 = lane.NewReg(b);
            b.L($"abs.{op} {ax}, {x};");
            b.L($"mul.rn.{op} {x2}, {ax}, {ax};");
            string series = Poly(b, lane, x2, Funcs.ErfSeries, w ? Funcs.ErfTerms64 : Funcs.ErfTerms32, 0);
            b.L($"mul.rn.{op} {series}, {series}, {ax};");
            string f = lane.NewReg(b), q = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {f}, {ax};");
            for (int d = (w ? Funcs.ErfDepth64 : Funcs.ErfDepth32); d >= 1; d--)
            {
                b.L($"div.rn.{op} {q}, {L(lane, d * 0.5)}, {f};");
                b.L($"add.rn.{op} {f}, {ax}, {q};");
            }

            string nx2 = lane.NewReg(b), tail = lane.NewReg(b), y = lane.NewReg(b), ny = lane.NewReg(b), pb = b.P(), pn = b.P(), pneg = b.P();
            string x2lo = lane.NewReg(b), corr = lane.NewReg(b), nx2b = lane.NewReg(b);
            b.L($"neg.{op} {nx2b}, {x2};");
            b.L($"fma.rn.{op} {x2lo}, {ax}, {ax}, {nx2b};");
            b.L($"sub.rn.{op} {corr}, {L(lane, 1)}, {x2lo};");
            b.L($"neg.{op} {nx2}, {x2};");
            string e = Codec.Exp(b, lane, nx2);
            b.L($"mul.rn.{op} {e}, {e}, {corr};");
            b.L($"mul.rn.{op} {tail}, {e}, {L(lane, Funcs.InvSqrtPi)};");
            b.L($"div.rn.{op} {tail}, {tail}, {f};");
            b.L($"sub.rn.{op} {tail}, {L(lane, 1)}, {tail};");
            b.L($"setp.lt.{op} {pb}, {ax}, {L(lane, w ? Funcs.ErfBound64 : Funcs.ErfBound32)};");
            b.L($"selp.{op} {y}, {series}, {tail}, {pb};");
            b.L($"min.{op} {y}, {y}, {L(lane, 1)};");
            b.L($"neg.{op} {ny}, {y};");
            b.L($"setp.lt.{op} {pneg}, {x}, {L(lane, 0)};");
            b.L($"selp.{op} {y}, {ny}, {y}, {pneg};");
            b.L($"testp.notanumber.{op} {pn}, {x};");
            b.L($"selp.{op} {y}, {x}, {y}, {pn};");
            return y;
        }

        public static string Sqrt(PtxBuilder b, Lane lane, string x)
        {
            string y = lane.NewReg(b);
            b.L($"sqrt.rn.{lane.Op} {y}, {x};");
            return y;
        }

        public static string Erfc(PtxBuilder b, Lane lane, string x)
        {
            bool w = lane.Wide;
            string op = lane.Op, ax = lane.NewReg(b), x2 = lane.NewReg(b);
            b.L($"abs.{op} {ax}, {x};");
            b.L($"mul.rn.{op} {x2}, {ax}, {ax};");
            string series = Poly(b, lane, x2, Funcs.ErfSeries, w ? Funcs.ErfTerms64 : Funcs.ErfTerms32, 0);
            b.L($"mul.rn.{op} {series}, {series}, {x};");
            string f = lane.NewReg(b), q = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {f}, {ax};");
            for (int d = (w ? Funcs.ErfcDepth64 : Funcs.ErfcDepth32); d >= 1; d--)
            {
                b.L($"div.rn.{op} {q}, {L(lane, d * 0.5)}, {f};");
                b.L($"add.rn.{op} {f}, {ax}, {q};");
            }

            string nx2 = lane.NewReg(b), tail = lane.NewReg(b), mid = lane.NewReg(b), two = lane.NewReg(b), y = lane.NewReg(b), pb = b.P(), pn = b.P(), pneg = b.P();
            string x2lo = lane.NewReg(b), corr = lane.NewReg(b), nx2b = lane.NewReg(b);
            b.L($"neg.{op} {nx2b}, {x2};");
            b.L($"fma.rn.{op} {x2lo}, {ax}, {ax}, {nx2b};");
            b.L($"sub.rn.{op} {corr}, {L(lane, 1)}, {x2lo};");
            b.L($"neg.{op} {nx2}, {x2};");
            string e = Codec.Exp(b, lane, nx2);
            b.L($"mul.rn.{op} {e}, {e}, {corr};");
            b.L($"mul.rn.{op} {tail}, {e}, {L(lane, Funcs.InvSqrtPi)};");
            b.L($"div.rn.{op} {tail}, {tail}, {f};");
            b.L($"sub.rn.{op} {mid}, {L(lane, 1)}, {series};");
            b.L($"sub.rn.{op} {two}, {L(lane, 2)}, {tail};");
            b.L($"setp.lt.{op} {pneg}, {x}, {L(lane, 0)};");
            b.L($"selp.{op} {y}, {two}, {tail}, {pneg};");
            b.L($"setp.lt.{op} {pb}, {ax}, {L(lane, w ? Funcs.ErfcBound64 : Funcs.ErfcBound32)};");
            b.L($"selp.{op} {y}, {mid}, {y}, {pb};");
            b.L($"testp.notanumber.{op} {pn}, {x};");
            b.L($"selp.{op} {y}, {x}, {y}, {pn};");
            return y;
        }
    }

    // ---------------------------------------------------------------- the formulas over a lane

    private sealed class Emit(PtxBuilder b, Lane lane)
    {
        private readonly string _op = lane.Op;
        private readonly bool _f = lane.IsFloat;

        public string C(double v)
        {
            string r = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {r}, {lane.Literal(v)};");
            return r;
        }

        private string Bin(string ins, string x, string y)
        {
            string r = lane.NewReg(b);
            b.L($"{ins} {r}, {x}, {y};");
            return r;
        }

        public string Add(string x, string y) => Bin(_f ? $"add.rn.{_op}" : $"add.{_op}", x, y);
        public string Sub(string x, string y) => Bin(_f ? $"sub.rn.{_op}" : $"sub.{_op}", x, y);
        public string Mul(string x, string y) => Bin(_f ? $"mul.rn.{_op}" : $"mul.lo.{_op}", x, y);
        public string Div(string x, string y) => Bin(_f ? $"div.rn.{_op}" : $"div.{_op}", x, y);
        public string Max(string x, string y) => Bin($"max.{_op}", x, y);
        public string Min(string x, string y) => Bin($"min.{_op}", x, y);

        public string Fma(string x, string y, string z)
        {
            string r = lane.NewReg(b);
            b.L(_f ? $"fma.rn.{_op} {r}, {x}, {y}, {z};" : $"mad.lo.{_op} {r}, {x}, {y}, {z};");
            return r;
        }

        public string Neg(string x)
        {
            string r = lane.NewReg(b);
            if (_f || _op[0] == 's') b.L($"neg.{_op} {r}, {x};");
            else b.L($"sub.{_op} {r}, 0, {x};");
            return r;
        }

        public string Abs(string x)
        {
            if (_op[0] == 'u') return x;
            string r = lane.NewReg(b);
            b.L($"abs.{_op} {r}, {x};");
            return r;
        }

        public string Greater(string x, string y)
        {
            string p = b.P();
            b.L($"setp.gt.{_op} {p}, {x}, {y};");
            return p;
        }

        public string Less(string x, string y)
        {
            string p = b.P();
            b.L($"setp.lt.{_op} {p}, {x}, {y};");
            return p;
        }

        public string NotZero(string x)
        {
            string p = b.P();
            b.L($"setp.ne.{_op} {p}, {x}, {lane.Literal(0)};");
            return p;
        }

        public string Select(string p, string x, string y)
        {
            string r = lane.NewReg(b);
            b.L($"selp.{(_f ? _op : lane.Reg)} {r}, {x}, {y}, {p};");
            return r;
        }

        public string And(string p, string q)
        {
            string r = b.P();
            b.L($"and.pred {r}, {p}, {q};");
            return r;
        }

        public string Sign(string x)
        {
            string one = C(1), zero = C(0), neg = Neg(one);
            return Select(Greater(x, zero), one, Select(Less(x, zero), neg, zero));
        }

        // float lanes
        public string Exp(string x) => Codec.Exp(b, lane, x);
        public string Log(string x) => Fn.Log(b, lane, x);
        public string Sqrt(string x) => Fn.Sqrt(b, lane, x);
        public string Tanh(string x) => Fn.Tanh(b, lane, x);
        public string Sigmoid(string x) => Fn.Sigmoid(b, lane, x);
        public string Erf(string x) => Fn.Erf(b, lane, x);
        public string Erfc(string x) => Fn.Erfc(b, lane, x);
        public string Phi(string x) => Mul(C(0.5), Erfc(Mul(x, C(-0.7071067811865476))));

        public string Gelu(string x)
        {
            string x2 = Mul(x, x);
            if (lane.Wide) return Mul(x, Sigmoid(Mul(C(2 * 0.7978845608028654), Fma(Mul(C(0.044715), x2), x, x))));
            // t = c·x + c·0.044715·x³ as a pair t + lo: c split into its lane value and remainder, the leading product
            // taken by fma, its rounding error recovered exactly by a second fma.
            const double c = 2 * 0.7978845608028654;
            double ch = (float)c;
            string p = Fma(Mul(C(c * 0.044715), x2), x, Mul(C(c - ch), x));
            string t = Fma(C(ch), x, p);
            string lo = Add(Fma(C(ch), x, Neg(t)), p);
            return XSigmoid(x, t, lo);
        }

        /// <summary>x·σ(t + lo) as x / (1 + e^(−t)·(1 − lo)), the denominator's sum split exactly and its low part
        /// folded into the quotient: one rounding fewer than x times the rounded σ(t).</summary>
        public string XSigmoid(string x, string t, string lo = null)
        {
            if (lane.Wide) return Mul(x, Sigmoid(t));
            string e = Exp(Neg(t));
            if (lo != null) e = Fma(e, Neg(lo), e);
            string one = C(1), d = Add(e, one), bv = Sub(d, e);
            string dlo = Add(Sub(e, Sub(d, bv)), Sub(one, bv));
            string q = Div(x, d);
            return Fma(Neg(q), Bin($"div.approx.{_op}", dlo, d), q);
        }

        public string GeluGrad(string x)
        {
            string x2 = Mul(x, x);
            string u2 = Mul(C(2 * 0.7978845608028654), Fma(Mul(C(0.044715), x2), x, x));
            string s = Sigmoid(u2), sn = Sigmoid(Neg(u2));
            string du = Mul(C(0.7978845608028654), Fma(C(3 * 0.044715), x2, C(1)));
            return Fma(Mul(Mul(C(2), x), Mul(s, sn)), du, s);
        }

        public string SiluGrad(string x)
        {
            string s = Sigmoid(x);
            return Mul(s, Fma(x, Sigmoid(Neg(x)), C(1)));
        }

        public string Pow(string x, string a)
        {
            string ax = Abs(x), mag = Exp(Mul(a, Log(ax)));
            string half = Mul(a, C(0.5)), ra = lane.NewReg(b), rh = lane.NewReg(b), pInt = b.P(), pHalfInt = b.P(), pOdd = b.P();
            b.L($"cvt.rni.{_op}.{_op} {ra}, {a};");
            b.L($"cvt.rni.{_op}.{_op} {rh}, {half};");
            b.L($"setp.eq.{_op} {pInt}, {ra}, {a};");
            b.L($"setp.ne.{_op} {pHalfInt}, {rh}, {half};");
            b.L($"and.pred {pOdd}, {pInt}, {pHalfInt};");
            string neg = Less(x, C(0));
            return Select(neg, Select(pOdd, Neg(mag), Select(pInt, mag, C(double.NaN))), mag);
        }
    }

    /// <summary>The result register of formula <paramref name="op"/> over lane registers x, y, z and parameters a, b.</summary>
    private static string Formula(PtxBuilder b, Lane lane, string op, string x, string y, string z, string a, string pb)
    {
        var e = new Emit(b, lane);
        string zero = lane.Literal(0);
        switch (op)
        {
            case "copy": return x;
            case "fill": return a;
            case "neg": return e.Neg(x);
            case "abs": return e.Abs(x);
            case "sign": return e.Sign(x);
            case "square": return e.Mul(x, x);
            case "relu": return e.Max(x, e.C(0));
            case "add_scalar": return e.Add(x, a);
            case "scale": return e.Mul(x, a);
            case "affine": return e.Fma(a, x, pb);
            case "clamp": return e.Min(e.Max(x, a), pb);
            case "add": return e.Add(x, y);
            case "sub": return e.Sub(x, y);
            case "mul": return e.Mul(x, y);
            case "div": return e.Div(x, y);
            case "max": return e.Max(x, y);
            case "min": return e.Min(x, y);
            case "axpy": return e.Fma(a, x, y);
            case "muladd": return e.Fma(x, y, z);
            case "where": return e.Select(e.NotZero(x), y, z);
            case "square_bwd": return e.Mul(e.Mul(e.C(2), x), y);
            case "abs_bwd": return e.Mul(e.Sign(x), y);
            case "relu_bwd": return e.Select(e.Greater(x, e.C(0)), y, e.C(0));
            case "clamp_bwd": return e.Select(e.And(e.Greater(x, a), e.Less(x, pb)), y, e.C(0));
            case "where_bwd": return e.Select(e.NotZero(x), y, e.C(0));
            case "max_bwd_x": return e.Select(e.Greater(y, x), e.C(0), z);
            case "max_bwd_y": return e.Select(e.Greater(y, x), z, e.C(0));
            case "min_bwd_x": return e.Select(e.Greater(x, y), e.C(0), z);
            case "min_bwd_y": return e.Select(e.Greater(x, y), z, e.C(0));
            case "gated_relu": return e.Mul(e.Max(x, e.C(0)), y);
            case "gated_relu_bwd_x": return e.Select(e.Greater(x, e.C(0)), e.Mul(y, z), e.C(0));
            case "gated_relu_bwd_y": return e.Mul(e.Max(x, e.C(0)), z);
            case "sqrt": return e.Sqrt(x);
            case "inv_sqrt": return e.Div(e.C(1), e.Sqrt(e.Add(x, a)));
            case "exp": return e.Exp(x);
            case "expm1": return Fn.Expm1(b, lane, x);
            case "log": return e.Log(x);
            case "sin": return Fn.Sin(b, lane, x);
            case "cos": return Fn.Cos(b, lane, x);
            case "tanh": return e.Tanh(x);
            case "sigmoid": return e.Sigmoid(x);
            case "erf": return e.Erf(x);
            case "gelu": return e.Gelu(x);
            case "gelu_erf": return e.Mul(x, e.Phi(x));
            case "silu": return e.XSigmoid(x, x);
            case "leaky_relu": return e.Select(e.Greater(x, e.C(0)), x, e.Mul(a, x));
            case "pow": return e.Pow(x, a);
            case "dropout": return e.Select(e.NotZero(y), e.Mul(x, a), e.C(0));
            case "gated_silu": return e.Mul(e.Mul(x, e.Sigmoid(x)), y);
            case "gated_gelu": return e.Mul(e.Gelu(x), y);
            case "sqrt_bwd": return e.Div(y, e.Mul(e.C(2), e.Sqrt(x)));
            case "inv_sqrt_bwd": { string s = e.Add(x, a); string r = e.Div(e.C(1), e.Sqrt(s)); return e.Mul(e.Mul(e.C(-0.5), e.Div(r, s)), y); }
            case "exp_bwd": return e.Mul(e.Exp(x), y);
            case "log_bwd": return e.Div(y, x);
            case "sin_bwd": return e.Mul(Fn.Cos(b, lane, x), y);
            case "cos_bwd": return e.Neg(e.Mul(Fn.Sin(b, lane, x), y));
            case "tanh_bwd": { string s = e.Sigmoid(e.Mul(e.C(2), x)); return e.Mul(e.Mul(e.C(4), e.Mul(s, e.Sigmoid(e.Mul(e.C(-2), x)))), y); }
            case "sigmoid_bwd": return e.Mul(e.Mul(e.Sigmoid(x), e.Sigmoid(e.Neg(x))), y);
            case "silu_bwd": return e.Mul(e.SiluGrad(x), y);
            case "gelu_bwd": return e.Mul(e.GeluGrad(x), y);
            case "gelu_erf_bwd":
            {
                string phi = e.Phi(x);
                string dens = e.Mul(e.C(0.3989422804014327), e.Exp(e.Mul(e.C(-0.5), e.Mul(x, x))));
                return e.Mul(e.Fma(x, dens, phi), y);
            }
            case "leaky_relu_bwd": return e.Select(e.Greater(x, e.C(0)), y, e.Mul(a, y));
            case "pow_bwd": return e.Mul(e.Mul(a, e.Pow(x, e.Sub(a, e.C(1)))), y);
            case "gated_silu_bwd_x": return e.Mul(e.Mul(e.SiluGrad(x), y), z);
            case "gated_silu_bwd_y": return e.Mul(e.Mul(x, e.Sigmoid(x)), z);
            case "gated_gelu_bwd_x": return e.Mul(e.Mul(e.GeluGrad(x), y), z);
            case "gated_gelu_bwd_y": return e.Mul(e.Gelu(x), z);
            case "gated_tanh_sigmoid": return e.Mul(e.Tanh(x), e.Sigmoid(y));
            case "gated_tanh_sigmoid_bwd_x": { string s = e.Sigmoid(e.Mul(e.C(2), x)); string sech2 = e.Mul(e.C(4), e.Mul(s, e.Sigmoid(e.Mul(e.C(-2), x)))); return e.Mul(e.Mul(sech2, e.Sigmoid(y)), z); }
            case "gated_tanh_sigmoid_bwd_y": return e.Mul(e.Mul(e.Tanh(x), e.Mul(e.Sigmoid(y), e.Sigmoid(e.Neg(y)))), z);
            case "add_gelu": return e.Gelu(e.Add(x, y));
            case "clip_factor": return e.Select(e.Greater(x, e.C(0)), e.Min(e.C(1), e.Div(a, e.Sqrt(x))), e.C(1));
            default: throw new ArgumentException(op);
        }
    }

    // ---------------------------------------------------------------- the mixed map: operands of their own formats

    /// <summary>output = op(x, y; a, b) over n values in the lane, a thread a value: x of fx from value p_xoff, y of fy from
    /// value p_yoff (row: y[i mod p_cols], a row vector), the output of fo (whole bytes, not blocks). A composition's adds of
    /// the type's tensors and its f32 intermediates, each read where it is stored.</summary>
    public static string MapMixed(Gpu gpu, Fmt fx, Fmt fy, Fmt fo, Lane lane, MapOp op, bool row, out string entry)
    {
        if (fo.IsBlock || fo.Bits < 8) throw new ArgumentException($"{fo.Name}: the mixed map stores whole bytes");
        entry = $"mapmix_{op.Name}_{fx.Name}_{fy.Name}_{fo.Name}_{lane.Name}{(row ? "_row" : "")}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string px = Param64(b, "p_x"), py = Param64(b, "p_y"), po = Param64(b, "p_o"), n = Param32(b, "p_n"), cols = Param32(b, "p_cols"), xoff = Param32(b, "p_xoff"), yoff = Param32(b, "p_yoff");
        string a = lane.NewReg(b), pb = lane.NewReg(b), p = b.P(), ix = b.R(), iy = b.R();
        b.L($"ld.param.{lane.Mem} {a}, [p_a];");
        b.L($"ld.param.{lane.Mem} {pb}, [p_b];");
        b.L($"setp.ge.u32 {p}, {gid}, {n};");
        b.L($"@{p} bra DONE;");
        b.L($"add.u32 {ix}, {gid}, {xoff};");
        if (row) b.L($"rem.u32 {iy}, {gid}, {cols};"); else b.L($"mov.u32 {iy}, {gid};");
        b.L($"add.u32 {iy}, {iy}, {yoff};");
        string x = Codec.UnpackAt(b, fx, px, ix, lane), y = Codec.UnpackAt(b, fy, py, iy, lane);
        StoreAs(b, fo, lane, Formula(b, lane, op.Name, x, y, x, a, pb), po, gid);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_y, .param .u64 p_o, .param .u32 p_n, .param .u32 p_cols, .param .u32 p_xoff, .param .u32 p_yoff, .param .{lane.Mem} p_a, .param .{lane.Mem} p_b");
    }

    // ---------------------------------------------------------------- the map kernel: a thread per word

    /// <summary>output = op(x, y, z; a, b) over n values of format f in lane `lane`; absent inputs are passed as any pointer and never read.</summary>
    public static string Map(Gpu gpu, Fmt f, Lane lane, MapOp op, out string entry)
    {
        entry = $"map_{op.Name}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string px = Param64(b, "p_x"), py = Param64(b, "p_y"), pz = Param64(b, "p_z"), po = Param64(b, "p_o"), n = Param32(b, "p_n");
        string a = lane.NewReg(b), pb = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {a}, [p_a];");
        b.L($"ld.param.{lane.Mem} {pb}, [p_b];");
        string words = b.R(), p = b.P();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else
        {
            string bits = b.RD(), wq = b.RD();
            b.L($"mul.wide.u32 {bits}, {n}, {f.Bits};");
            b.L($"add.u64 {bits}, {bits}, 31;");
            b.L($"shr.u64 {wq}, {bits}, 5;");
            b.L($"cvt.u32.u64 {words}, {wq};");
        }

        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        string off = b.RD(), ax = b.RD(), ay = b.RD(), az = b.RD(), ao = b.RD();
        b.L($"mul.wide.u32 {off}, {gid}, {f.WordBits / 8};");
        b.L($"add.u64 {ax}, {px}, {off};");
        b.L($"add.u64 {ay}, {py}, {off};");
        b.L($"add.u64 {az}, {pz}, {off};");
        b.L($"add.u64 {ao}, {po}, {off};");
        string wx = f.WordBits == 64 ? b.RD() : b.R(), wy = f.WordBits == 64 ? b.RD() : b.R(), wz = f.WordBits == 64 ? b.RD() : b.R(), wo = f.WordBits == 64 ? b.RD() : b.R();
        if (op.Inputs >= 1) b.L($"ld.global.{wt} {wx}, [{ax}];"); else b.L($"mov.{wt} {wx}, 0;");
        if (op.Inputs >= 2) b.L($"ld.global.{wt} {wy}, [{ay}];"); else b.L($"mov.{wt} {wy}, 0;");
        if (op.Inputs >= 3) b.L($"ld.global.{wt} {wz}, [{az}];"); else b.L($"mov.{wt} {wz}, 0;");
        if (f.WordBits == 32) b.L($"mov.b32 {wo}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string x = Codec.Unpack(b, f, wx, i, lane);
            string y = op.Inputs >= 2 ? Codec.Unpack(b, f, wy, i, lane) : x;
            string z = op.Inputs >= 3 ? Codec.Unpack(b, f, wz, i, lane) : x;
            string r = Formula(b, lane, op.Name, x, y, z, a, pb);
            Codec.Pack(b, f, r, lane, i, wo, wrap: !f.IsFloat);
        }

        b.L($"st.global.{wt} [{ao}], {wo};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_y, .param .u64 p_z, .param .u64 p_o, .param .u32 p_n, .param .{lane.Mem} p_a, .param .{lane.Mem} p_b");
    }
}
