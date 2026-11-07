// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The transcendental functions on vectors, for the float lanes (f32 and f64), written once as coefficient tables
// and reductions that the CPU (here) and the device (Ptx.Funcs) evaluate the same way: fused multiply-adds over an
// exactly reduced argument. Every function is proven against a scalar reference in Checks before anything is timed.
//
//   Log      x = 2^e · m, m in [√½, √2): log x = e·ln2 + 2·atanh(t), t = (m − 1)/(m + 1) (|t| ≤ 0.1716), by its series
//   Sin, Cos x = k·π/2 + r (π/2 in three parts: the reduction is exact to |x| < 2^20 (f32) / 2^30 (f64)), the series on r
//   Expm1    e^x − 1 with the constant term left out of the exp polynomial, so small x keep their bits
//   Tanh     expm1(2x) / (expm1(2x) + 2); Sigmoid 1 / (1 + e^−x)
//   Erf      the series Σ (−1)^k x^(2k+1) / (k! (2k+1)) · 2/√π for |x| below a bound, else 1 − erfc by the continued
//            fraction erfc x = e^(−x²)/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + …)))) evaluated bottom-up a fixed depth (the
//            device; the CPU's f64 lane as a ratio of two recurrences); the CPU's f32 lane takes erfc's tail as
//            e^(−x²)·t·S(t), t = 1/x, S a polynomial fitted to 3e-9 (ErfcTail32)
using System.Numerics;

namespace Adad;

public static class Funcs
{
    // ---------------------------------------------------------------- the tables (shared with the device emitter)

    public const double Ln2Hi = 6.93147180369123816490e-01, Ln2Lo = 1.90821492927058770002e-10;
    public const float Ln2HiF = 0.693359375f, Ln2LoF = -2.12194440e-4f;
    // π/2 in three 33-bit parts (k · part is exact for k below 2^20), the rest below 2^-103.
    public const double PiHalf1 = 1.57079632673412561417e+00, PiHalf2 = 6.07710050630396597660e-11, PiHalf3 = 2.02226624871116645580e-21;
    public const float PiHalf1F = 1.5707963705062866f, PiHalf2F = -4.3711388286737929e-08f, PiHalf3F = -1.7151245100059e-15f;
    public const double TwoOverSqrtPi = 1.1283791670955126, InvSqrtPi = 0.5641895835477563;

    /// <summary>1/n! for n = 0 .. Count − 1 (the exp series).</summary>
    public static readonly double[] InvFact = Build(26, n => 1.0 / Factorial(n));

    /// <summary>2/(2k+1) for k = 0 .. : the atanh series 2·Σ t^(2k+1)/(2k+1).</summary>
    public static readonly double[] Atanh = Build(16, k => 2.0 / (2 * k + 1));

    /// <summary>(−1)^k / (2k+1)! for sin, (−1)^k / (2k)! for cos.</summary>
    public static readonly double[] SinSeries = Build(12, k => (k % 2 == 0 ? 1.0 : -1.0) / Factorial(2 * k + 1));
    public static readonly double[] CosSeries = Build(12, k => (k % 2 == 0 ? 1.0 : -1.0) / Factorial(2 * k));

    /// <summary>(−1)^k / (k! (2k+1)) · 2/√π for erf's series.</summary>
    public static readonly double[] ErfSeries = Build(64, k => (k % 2 == 0 ? 1.0 : -1.0) / (Factorial(k) * (2 * k + 1)) * TwoOverSqrtPi);

    /// <summary>Series terms and continued-fraction depth each lane uses (past its last bit at the series' bound).</summary>
    public const int ExpTerms32 = 8, ExpTerms64 = 14, AtanhTerms32 = 6, AtanhTerms64 = 12, SinTerms32 = 6, SinTerms64 = 11;
    public const int ErfTerms32 = 16, ErfTerms64 = 30, ErfDepth32 = 48, ErfDepth64 = 100;
    public const float ErfBound32 = 1.0f;
    public const double ErfBound64 = 1.5;

    /// <summary>erfc's tail on the CPU's f32 lane: S(t) = x·e^(x²)·erfc x, t = 1/x in [0, 4/3] (x ≥ 0.75), as a polynomial in
    /// u = 1.5·t − 1 (coefficient k of u^k): S interpolated at the 17 Chebyshev nodes of the interval (S computed at 40 digits),
    /// within 3.2e-9 of S everywhere in it; Horner in u keeps its f32 evaluation within about one rounding (the monomials
    /// in t would cancel to 1e-4). erfc x = e^(−x²) · t · S: one division and 16 multiply-adds for the continued
    /// fraction's 48 to 80 serial levels.</summary>
    public static readonly double[] ErfcTail32 = [0.4823781246814764, -0.11422653907699147, 0.003084741801664283, 0.017759010812733424, -0.013435622651384071, 0.006117608127435746, -0.0013524073877195708, -0.0007986113371849265, 0.0012025644360486776, -0.0006279183665454736, 0.00031967429535662565, -0.0006445173610058078, 0.0003125770636888739, 0.000514276050142984, -0.0004131862318729763, -8.64806258308527e-05, 9.994421419212651e-05];
    public const double ErfcTailSpan = 4.0 / 3.0;

    /// <summary>erfc keeps 1 − series only where the cancellation is mild (erf below 0.71), and takes the fraction deeper.</summary>
    public const int ErfcDepth32 = 80, ErfcDepth64 = 400;
    public const float ErfcBound32 = 0.75f;
    public const double ErfcBound64 = 0.75;

    private static double[] Build(int count, Func<int, double> f)
    {
        var a = new double[count];
        for (int i = 0; i < count; i++) a[i] = f(i);
        return a;
    }

    private static double Factorial(int n)
    {
        double f = 1;
        for (int i = 2; i <= n; i++) f *= i;
        return f;
    }

    // ---------------------------------------------------------------- f32

    private static Vector<float> F(Vector<uint> bits) => Vector.AsVectorSingle(bits);
    private static Vector<uint> U(Vector<float> v) => Vector.AsVectorUInt32(v);
    private static Vector<float> F(Vector<int> bits) => Vector.AsVectorSingle(bits);
    private static Vector<int> I(Vector<float> v) => Vector.AsVectorInt32(v);

    public static Vector<float> Log(Vector<float> x)
    {
        // Subnormals scaled up by 2^23 first; the exponent from the bits, the mantissa put in [1, 2) then [√½, √2).
        var tiny = Vector.LessThan(x, new Vector<float>(1.17549435e-38f));
        var xs = Vector.ConditionalSelect(tiny, x * new Vector<float>(8388608f), x);
        var bits = I(xs);
        var e = Vector.ShiftRightArithmetic(bits, 23) - new Vector<int>(127);
        var m = F((bits & new Vector<int>(0x007FFFFF)) | new Vector<int>(0x3F800000));
        var big = Vector.GreaterThan(m, new Vector<float>(1.41421356f));
        m = Vector.ConditionalSelect(big, m * new Vector<float>(0.5f), m);
        e = Vector.ConditionalSelect(big, e + Vector<int>.One, e);
        e = Vector.ConditionalSelect(Vector.AsVectorInt32(tiny), e - new Vector<int>(23), e);
        var t = (m - Vector<float>.One) / (m + Vector<float>.One);
        var t2 = t * t;
        var p = new Vector<float>((float)Atanh[AtanhTerms32 - 1]);
        for (int k = AtanhTerms32 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, t2, new Vector<float>((float)Atanh[k]));
        var ef = Vector.ConvertToSingle(e);
        var y = Vector.FusedMultiplyAdd(ef, new Vector<float>(Ln2HiF), Vector.FusedMultiplyAdd(ef, new Vector<float>(Ln2LoF), p * t));
        y = Vector.ConditionalSelect(Vector.Equals(x, Vector<float>.Zero), new Vector<float>(float.NegativeInfinity), y);
        y = Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), new Vector<float>(float.NaN), y);
        y = Vector.ConditionalSelect(Vector.Equals(x, new Vector<float>(float.PositiveInfinity)), x, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    private static (Vector<float> R, Vector<int> K) ReduceHalfPi(Vector<float> x)
    {
        var k = Vector.Round(x * new Vector<float>(0.63661977236758134f));
        var r = Vector.FusedMultiplyAdd(k, new Vector<float>(-PiHalf1F), x);
        r = Vector.FusedMultiplyAdd(k, new Vector<float>(-PiHalf2F), r);
        r = Vector.FusedMultiplyAdd(k, new Vector<float>(-PiHalf3F), r);
        return (r, Vector.ConvertToInt32(k));
    }

    private static Vector<float> SinPoly(Vector<float> r)
    {
        var r2 = r * r;
        var p = new Vector<float>((float)SinSeries[SinTerms32 - 1]);
        for (int k = SinTerms32 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, r2, new Vector<float>((float)SinSeries[k]));
        return p * r;
    }

    private static Vector<float> CosPoly(Vector<float> r)
    {
        var r2 = r * r;
        var p = new Vector<float>((float)CosSeries[SinTerms32 - 1]);
        for (int k = SinTerms32 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, r2, new Vector<float>((float)CosSeries[k]));
        return p;
    }

    public static Vector<float> Sin(Vector<float> x)
    {
        var (r, k) = ReduceHalfPi(x);
        var s = SinPoly(r);
        var c = CosPoly(r);
        var q = k & new Vector<int>(3);
        var odd = Vector.Equals(q & Vector<int>.One, Vector<int>.One);
        var y = Vector.ConditionalSelect(Vector.AsVectorSingle(odd), c, s);
        var neg = Vector.Equals(q & new Vector<int>(2), new Vector<int>(2));
        y = Vector.ConditionalSelect(Vector.AsVectorSingle(neg), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x) & Vector.LessThan(Vector.Abs(x), new Vector<float>(float.PositiveInfinity)), y, new Vector<float>(float.NaN));
    }

    /// <summary>Sin and Cos of the same values from one reduction and one pair of polynomials: each equal bit for bit to
    /// Sin and Cos (the same steps; rope takes both of every angle).</summary>
    public static (Vector<float> Sin, Vector<float> Cos) SinCos(Vector<float> x)
    {
        var (r, k) = ReduceHalfPi(x);
        var s = SinPoly(r);
        var c = CosPoly(r);
        var q = k & new Vector<int>(3);
        var odd = Vector.AsVectorSingle(Vector.Equals(q & Vector<int>.One, Vector<int>.One));
        var neg = Vector.AsVectorSingle(Vector.Equals(q & new Vector<int>(2), new Vector<int>(2)));
        var ys = Vector.ConditionalSelect(odd, c, s);
        ys = Vector.ConditionalSelect(neg, -ys, ys);
        var yc = Vector.ConditionalSelect(odd, -s, c);
        yc = Vector.ConditionalSelect(neg, -yc, yc);
        var finite = Vector.Equals(x, x) & Vector.LessThan(Vector.Abs(x), new Vector<float>(float.PositiveInfinity));
        return (Vector.ConditionalSelect(finite, ys, new Vector<float>(float.NaN)), Vector.ConditionalSelect(finite, yc, new Vector<float>(float.NaN)));
    }

    public static Vector<float> Cos(Vector<float> x)
    {
        var (r, k) = ReduceHalfPi(x);
        var s = SinPoly(r);
        var c = CosPoly(r);
        var q = k & new Vector<int>(3);
        var odd = Vector.Equals(q & Vector<int>.One, Vector<int>.One);
        var y = Vector.ConditionalSelect(Vector.AsVectorSingle(odd), -s, c);
        var neg = Vector.Equals(q & new Vector<int>(2), new Vector<int>(2));
        y = Vector.ConditionalSelect(Vector.AsVectorSingle(neg), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x) & Vector.LessThan(Vector.Abs(x), new Vector<float>(float.PositiveInfinity)), y, new Vector<float>(float.NaN));
    }

    /// <summary>e^x − 1, with small x kept to their bits.</summary>
    public static Vector<float> Expm1(Vector<float> x)
    {
        var c = Vector.Min(Vector.Max(x, new Vector<float>(-87.33654f)), new Vector<float>(88.72283f));
        var n = Vector.Round(c * new Vector<float>(1.44269504f));
        var r = Vector.FusedMultiplyAdd(n, new Vector<float>(-Ln2HiF), c);
        r = Vector.FusedMultiplyAdd(n, new Vector<float>(-Ln2LoF), r);
        // q = p(r) − 1 = r·(1 + r/2 + r²/6 + …): the constant term left out.
        var q = new Vector<float>((float)InvFact[ExpTerms32 - 1]);
        for (int k = ExpTerms32 - 2; k >= 1; k--) q = Vector.FusedMultiplyAdd(q, r, new Vector<float>((float)InvFact[k]));
        q *= r;
        var ni = Vector.ConvertToInt32(n);
        var scale = F((ni + new Vector<int>(127)) << 23);
        // (q + 1)·2^n − 1 = q·2^n + (2^n − 1): exact when n = 0.
        var y = Vector.FusedMultiplyAdd(q, scale, scale - Vector<float>.One);
        y = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<float>(-87.33654f)), new Vector<float>(-1f), y);
        y = Vector.ConditionalSelect(Vector.GreaterThan(x, new Vector<float>(88.72283f)), new Vector<float>(float.PositiveInfinity), y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    public static Vector<float> Tanh(Vector<float> x)
    {
        var a = Vector.Min(Vector.Max(x, new Vector<float>(-20f)), new Vector<float>(20f));
        var e = Expm1(a * new Vector<float>(2f));
        var y = e / (e + new Vector<float>(2f));
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    public static Vector<float> Sigmoid(Vector<float> x) => Vector<float>.One / (Vector<float>.One + Exps.Exp(-x));

    // erfc |x| for |x| ≥ 0.75 on the f32 lane: e^(−x²) (x² carried with its rounding, x2lo) · t · S(u) (ErfcTail32).
    private static Vector<float> ErfcTail(Vector<float> ax, Vector<float> x2)
    {
        var t = Vector<float>.One / ax;
        var u = Vector.FusedMultiplyAdd(t, new Vector<float>((float)(2 / ErfcTailSpan)), -Vector<float>.One);
        var sp = new Vector<float>((float)ErfcTail32[^1]);
        for (int k = ErfcTail32.Length - 2; k >= 0; k--) sp = Vector.FusedMultiplyAdd(sp, u, new Vector<float>((float)ErfcTail32[k]));
        var x2lo = Vector.FusedMultiplyAdd(ax, ax, -x2);
        return Exps.Exp(-x2) * (Vector<float>.One - x2lo) * (t * sp);
    }

    public static Vector<float> Erf(Vector<float> x)
    {
        var ax = Vector.Abs(x);
        var x2 = ax * ax;
        var series = new Vector<float>((float)ErfSeries[ErfTerms32 - 1]);
        for (int k = ErfTerms32 - 2; k >= 0; k--) series = Vector.FusedMultiplyAdd(series, x2, new Vector<float>((float)ErfSeries[k]));
        series *= ax;
        var tail = Vector<float>.One - ErfcTail(ax, x2);
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<float>(ErfBound32)), series, tail);
        y = Vector.Min(y, Vector<float>.One);
        y = Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    /// <summary>erfc without cancellation: the continued fraction for x above the bound, 2 − erfc(−x) below −bound, 1 − erf between.</summary>
    public static Vector<float> Erfc(Vector<float> x)
    {
        var ax = Vector.Abs(x);
        var x2 = ax * ax;
        var series = new Vector<float>((float)ErfSeries[ErfTerms32 - 1]);
        for (int k = ErfTerms32 - 2; k >= 0; k--) series = Vector.FusedMultiplyAdd(series, x2, new Vector<float>((float)ErfSeries[k]));
        series *= x;                                                       // erf(x), signed, for |x| < bound
        var tail = ErfcTail(ax, x2);                                       // erfc(|x|)
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<float>(ErfcBound32)), Vector<float>.One - series,
            Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), new Vector<float>(2f) - tail, tail));
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    // ---------------------------------------------------------------- f64

    private static Vector<double> D(Vector<long> bits) => Vector.AsVectorDouble(bits);
    private static Vector<long> L(Vector<double> v) => Vector.AsVectorInt64(v);

    public static Vector<double> Log(Vector<double> x)
    {
        var tiny = Vector.LessThan(x, new Vector<double>(2.2250738585072014e-308));
        var xs = Vector.ConditionalSelect(tiny, x * new Vector<double>(4503599627370496.0), x);
        var bits = L(xs);
        var e = Vector.ShiftRightArithmetic(bits, 52) - new Vector<long>(1023);
        var m = D((bits & new Vector<long>(0x000FFFFFFFFFFFFFL)) | new Vector<long>(0x3FF0000000000000L));
        var big = Vector.GreaterThan(m, new Vector<double>(1.4142135623730951));
        m = Vector.ConditionalSelect(big, m * new Vector<double>(0.5), m);
        e = Vector.ConditionalSelect(big, e + Vector<long>.One, e);
        e = Vector.ConditionalSelect(Vector.AsVectorInt64(tiny), e - new Vector<long>(52), e);
        var t = (m - Vector<double>.One) / (m + Vector<double>.One);
        var t2 = t * t;
        var p = new Vector<double>(Atanh[AtanhTerms64 - 1]);
        for (int k = AtanhTerms64 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, t2, new Vector<double>(Atanh[k]));
        var ef = Vector.ConvertToDouble(e);
        var y = Vector.FusedMultiplyAdd(ef, new Vector<double>(Ln2Hi), Vector.FusedMultiplyAdd(ef, new Vector<double>(Ln2Lo), p * t));
        y = Vector.ConditionalSelect(Vector.Equals(x, Vector<double>.Zero), new Vector<double>(double.NegativeInfinity), y);
        y = Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), new Vector<double>(double.NaN), y);
        y = Vector.ConditionalSelect(Vector.Equals(x, new Vector<double>(double.PositiveInfinity)), x, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    private static (Vector<double> R, Vector<long> K) ReduceHalfPi(Vector<double> x)
    {
        var k = Vector.Round(x * new Vector<double>(0.63661977236758134));
        var r = Vector.FusedMultiplyAdd(k, new Vector<double>(-PiHalf1), x);
        r = Vector.FusedMultiplyAdd(k, new Vector<double>(-PiHalf2), r);
        r = Vector.FusedMultiplyAdd(k, new Vector<double>(-PiHalf3), r);
        var magic = new Vector<double>(6755399441055744.0);
        var ki = Vector.AsVectorInt64(k + magic) - Vector.AsVectorInt64(magic);
        return (r, ki);
    }

    private static Vector<double> SinPoly(Vector<double> r)
    {
        var r2 = r * r;
        var p = new Vector<double>(SinSeries[SinTerms64 - 1]);
        for (int k = SinTerms64 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, r2, new Vector<double>(SinSeries[k]));
        return p * r;
    }

    private static Vector<double> CosPoly(Vector<double> r)
    {
        var r2 = r * r;
        var p = new Vector<double>(CosSeries[SinTerms64 - 1]);
        for (int k = SinTerms64 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, r2, new Vector<double>(CosSeries[k]));
        return p;
    }

    public static Vector<double> Sin(Vector<double> x)
    {
        var (r, k) = ReduceHalfPi(x);
        var s = SinPoly(r);
        var c = CosPoly(r);
        var q = k & new Vector<long>(3);
        var odd = Vector.Equals(q & Vector<long>.One, Vector<long>.One);
        var y = Vector.ConditionalSelect(Vector.AsVectorDouble(odd), c, s);
        var neg = Vector.Equals(q & new Vector<long>(2), new Vector<long>(2));
        y = Vector.ConditionalSelect(Vector.AsVectorDouble(neg), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x) & Vector.LessThan(Vector.Abs(x), new Vector<double>(double.PositiveInfinity)), y, new Vector<double>(double.NaN));
    }

    /// <summary>Sin and Cos of the same values from one reduction (each equal bit for bit to Sin and Cos).</summary>
    public static (Vector<double> Sin, Vector<double> Cos) SinCos(Vector<double> x)
    {
        var (r, k) = ReduceHalfPi(x);
        var s = SinPoly(r);
        var c = CosPoly(r);
        var q = k & new Vector<long>(3);
        var odd = Vector.AsVectorDouble(Vector.Equals(q & Vector<long>.One, Vector<long>.One));
        var neg = Vector.AsVectorDouble(Vector.Equals(q & new Vector<long>(2), new Vector<long>(2)));
        var ys = Vector.ConditionalSelect(odd, c, s);
        ys = Vector.ConditionalSelect(neg, -ys, ys);
        var yc = Vector.ConditionalSelect(odd, -s, c);
        yc = Vector.ConditionalSelect(neg, -yc, yc);
        var finite = Vector.Equals(x, x) & Vector.LessThan(Vector.Abs(x), new Vector<double>(double.PositiveInfinity));
        return (Vector.ConditionalSelect(finite, ys, new Vector<double>(double.NaN)), Vector.ConditionalSelect(finite, yc, new Vector<double>(double.NaN)));
    }

    public static Vector<double> Cos(Vector<double> x)
    {
        var (r, k) = ReduceHalfPi(x);
        var s = SinPoly(r);
        var c = CosPoly(r);
        var q = k & new Vector<long>(3);
        var odd = Vector.Equals(q & Vector<long>.One, Vector<long>.One);
        var y = Vector.ConditionalSelect(Vector.AsVectorDouble(odd), -s, c);
        var neg = Vector.Equals(q & new Vector<long>(2), new Vector<long>(2));
        y = Vector.ConditionalSelect(Vector.AsVectorDouble(neg), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x) & Vector.LessThan(Vector.Abs(x), new Vector<double>(double.PositiveInfinity)), y, new Vector<double>(double.NaN));
    }

    public static Vector<double> Expm1(Vector<double> x)
    {
        var c = Vector.Min(Vector.Max(x, new Vector<double>(-708.39)), new Vector<double>(709.78));
        var n = Vector.Round(c * new Vector<double>(1.4426950408889634));
        var r = Vector.FusedMultiplyAdd(n, new Vector<double>(-Ln2Hi), c);
        r = Vector.FusedMultiplyAdd(n, new Vector<double>(-Ln2Lo), r);
        var q = new Vector<double>(InvFact[ExpTerms64 - 1]);
        for (int k = ExpTerms64 - 2; k >= 1; k--) q = Vector.FusedMultiplyAdd(q, r, new Vector<double>(InvFact[k]));
        q *= r;
        var magic = new Vector<double>(6755399441055744.0);
        var ni = Vector.AsVectorInt64(n + magic) - Vector.AsVectorInt64(magic);
        var scale = D((ni + new Vector<long>(1023)) << 52);
        var y = Vector.FusedMultiplyAdd(q, scale, scale - Vector<double>.One);
        y = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<double>(-708.39)), new Vector<double>(-1.0), y);
        y = Vector.ConditionalSelect(Vector.GreaterThan(x, new Vector<double>(709.78)), new Vector<double>(double.PositiveInfinity), y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    public static Vector<double> Tanh(Vector<double> x)
    {
        var a = Vector.Min(Vector.Max(x, new Vector<double>(-20.0)), new Vector<double>(20.0));
        var e = Expm1(a * new Vector<double>(2.0));
        var y = e / (e + new Vector<double>(2.0));
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    public static Vector<double> Sigmoid(Vector<double> x) => Vector<double>.One / (Vector<double>.One + Exps.Exp(-x));

    // The continued fraction f = x + (depth/2) / f, …, down to f = x + ½ / f (f starting at x), as a ratio P / Q: each level
    // P' = x·P + (d/2)·Q, Q' = P — two multiply-adds, no division — the pair rescaled to (1, Q/P) every 16th level (one
    // division, the magnitudes held far from overflow) and divided once at the end; x taken at most 28 (e^(−x²) is 0 in
    // f64 past it: the fraction's value no longer counts).
    private static Vector<double> Fraction(Vector<double> ax, int depth)
    {
        var a = Vector.Min(ax, new Vector<double>(28.0));
        var p = a;
        var q = Vector<double>.One;
        for (int d = depth; d >= 1; d--)
        {
            var np = Vector.FusedMultiplyAdd(a, p, new Vector<double>(d * 0.5) * q);
            q = p;
            p = np;
            if (d % 16 == 1 && d > 1) { q /= p; p = Vector<double>.One; }
        }

        return p / q;
    }

    public static Vector<double> Erf(Vector<double> x)
    {
        var ax = Vector.Abs(x);
        var x2 = ax * ax;
        var series = new Vector<double>(ErfSeries[ErfTerms64 - 1]);
        for (int k = ErfTerms64 - 2; k >= 0; k--) series = Vector.FusedMultiplyAdd(series, x2, new Vector<double>(ErfSeries[k]));
        series *= ax;
        var f = Fraction(ax, ErfDepth64);
        var x2lo = Vector.FusedMultiplyAdd(ax, ax, -x2);
        var tail = Vector<double>.One - Exps.Exp(-x2) * (Vector<double>.One - x2lo) * new Vector<double>(InvSqrtPi) / f;
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<double>(ErfBound64)), series, tail);
        y = Vector.Min(y, Vector<double>.One);
        y = Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    public static Vector<double> Erfc(Vector<double> x)
    {
        var ax = Vector.Abs(x);
        var x2 = ax * ax;
        var series = new Vector<double>(ErfSeries[ErfTerms64 - 1]);
        for (int k = ErfTerms64 - 2; k >= 0; k--) series = Vector.FusedMultiplyAdd(series, x2, new Vector<double>(ErfSeries[k]));
        series *= x;
        var f = Fraction(ax, ErfcDepth64);
        var x2lo = Vector.FusedMultiplyAdd(ax, ax, -x2);
        var tail = Exps.Exp(-x2) * (Vector<double>.One - x2lo) * new Vector<double>(InvSqrtPi) / f;
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<double>(ErfcBound64)), Vector<double>.One - series,
            Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), new Vector<double>(2.0) - tail, tail));
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    // ---------------------------------------------------------------- scalar references (double): what the proofs compare against

    public static double ErfcReference(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 0.75) return 1 - ErfReference(x);
        double f = ax;
        for (int d = 1200; d >= 1; d--) f = ax + d * 0.5 / f;
        double hi = ax * ax, lo = Math.FusedMultiplyAdd(ax, ax, -hi);
        double tail = Math.Exp(-hi) * (1 - lo) * InvSqrtPi / f;
        return x < 0 ? 2 - tail : tail;
    }

    /// <summary>1 − tanh²(x) = sech²(x) without cancellation.</summary>
    public static double SechSquaredReference(double x)
    {
        double c = 2 / (Math.Exp(x) + Math.Exp(-x));
        return c * c;
    }

    /// <summary>erf in double by a method independent of the vector one's parameters: the series summed from its smallest
    /// term up for |x| < 1.5 (40 terms; no cancellation there), the continued fraction 400 deep beyond.</summary>
    public static double ErfReference(double x)
    {
        double ax = Math.Abs(x), y;
        if (ax < 1.5)
        {
            double sum = 0;
            for (int k = 39; k >= 0; k--)
            {
                double term = Math.Pow(ax, 2 * k + 1) / (Factorial(k) * (2 * k + 1));
                sum += k % 2 == 0 ? term : -term;
            }

            y = sum * TwoOverSqrtPi;
        }
        else
        {
            double f = ax;
            for (int d = 400; d >= 1; d--) f = ax + d * 0.5 / f;
            double hi = ax * ax, lo = Math.FusedMultiplyAdd(ax, ax, -hi);
            y = 1 - Math.Exp(-hi) * (1 - lo) * InvSqrtPi / f;
        }

        return x < 0 ? -y : y;
    }

    /// <summary>e^x − 1 in double without cancellation: the series for |x| < 0.5 (summed from its smallest term), else exp − 1.</summary>
    public static double Expm1Reference(double x)
    {
        if (Math.Abs(x) >= 0.5) return Math.Exp(x) - 1;
        double sum = 0;
        for (int k = 30; k >= 1; k--) sum += Math.Pow(x, k) / Factorial(k);
        return sum;
    }
}
