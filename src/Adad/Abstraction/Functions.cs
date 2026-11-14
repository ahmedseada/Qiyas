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
//   Erf      the series Σ (−1)^k x^(2k+1) / (k! (2k+1)) · 2/√π for |x| below a bound, else 1 − erfc; erfc's tail (|x| past
//            its bound) on the f64 lane e^(−x²)·S(t)/x, t = 1/x, S a polynomial fitted to 6.3e-17 (ErfcTail64), on every
//            device; on the f32 lane e^(−x²)·t·S(t) (ErfcTail32) on the CPU and Vulkan, CUDA's the continued fraction
//            erfc x = e^(−x²)/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + …)))) evaluated bottom-up a fixed depth
using System.Numerics;

namespace Adad.Abstraction;

public static class Funcs
{
    // ---------------------------------------------------------------- the tables (shared with the device emitter)

    public const double Ln2Hi = 6.93147180369123816490e-01, Ln2Lo = 1.90821492927058770002e-10;
    public const float Ln2HiF = 0.693359375f, Ln2LoF = -2.12194440e-4f;
    // π/2 in three 33-bit parts (k · part is exact for k below 2^20), the rest below 2^-103.
    public const double PiHalf1 = 1.57079632673412561417e+00, PiHalf2 = 6.07710050630396597660e-11, PiHalf3 = 2.02226624871116645580e-21;
    public const float PiHalf1F = 1.5707963705062866f, PiHalf2F = -4.3711388286737929e-08f, PiHalf3F = -1.7151245100059e-15f;
    public const double TwoOverSqrtPi = 1.1283791670955126, InvSqrtPi = 0.5641895835477563;
    // atan's reduction: π/2 − atan(1/u) past 1, π/6 + atan((u√3 − 1)/(u + √3)) past tan(π/12) = 2 − √3; π/2, π/6 and √3 each
    // as the lane's nearest value and the rest (hi + lo), so the reduction and the sums add no rounding of the constants.
    public const double PiHalf = 1.5707963267948966, PiHalfLo = 6.123233995736766e-17, PiSixth = 0.5235987755982989, PiSixthLo = -5.360408832255455e-17;
    public const double Sqrt3 = 1.7320508075688772, Sqrt3Lo = 1.0035084221806903e-16, TanPiTwelfth = 0.26794919243112270;
    public const float PiHalfF = 1.5707963705062866f, PiHalfLoF = -4.3711390001862426e-08f, PiSixthF = 0.5235987901687622f, PiSixthLoF = -1.4570463333954142e-08f;
    public const float Sqrt3F = 1.7320507764816284f, Sqrt3LoF = 3.10872488755587e-08f;

    /// <summary>1/n! for n = 0 .. Count − 1 (the exp series).</summary>
    public static readonly double[] InvFact = Build(26, n => 1.0 / Factorial(n));

    /// <summary>2/(2k+1) for k = 0 .. : the atanh series 2·Σ t^(2k+1)/(2k+1).</summary>
    public static readonly double[] Atanh = Build(16, k => 2.0 / (2 * k + 1));

    /// <summary>(−1)^k/(2k+1): the atan series Σ (−1)^k t^(2k+1)/(2k+1), kept to |t| ≤ tan(π/12) (t² ≤ 0.072).</summary>
    public static readonly double[] AtanSeries = Build(16, k => (k % 2 == 0 ? 1.0 : -1.0) / (2 * k + 1));

    /// <summary>(−1)^k / (2k+1)! for sin, (−1)^k / (2k)! for cos.</summary>
    public static readonly double[] SinSeries = Build(12, k => (k % 2 == 0 ? 1.0 : -1.0) / Factorial(2 * k + 1));
    public static readonly double[] CosSeries = Build(12, k => (k % 2 == 0 ? 1.0 : -1.0) / Factorial(2 * k));

    /// <summary>(−1)^k / (k! (2k+1)) · 2/√π for erf's series.</summary>
    public static readonly double[] ErfSeries = Build(64, k => (k % 2 == 0 ? 1.0 : -1.0) / (Factorial(k) * (2 * k + 1)) * TwoOverSqrtPi);

    /// <summary>Series terms and continued-fraction depth each lane uses (past its last bit at the series' bound).</summary>
    public const int ExpTerms32 = 8, ExpTerms64 = 14, AtanhTerms32 = 6, AtanhTerms64 = 12, SinTerms32 = 6, SinTerms64 = 11;
    /// <summary>atan's terms: the first left out is below 2^−26 (f32) and 2^−58 (f64) of the result at tan(π/12).</summary>
    public const int AtanTerms32 = 6, AtanTerms64 = 14;
    public const int ErfTerms32 = 16, ErfTerms64 = 30, ErfDepth32 = 48;
    public const float ErfBound32 = 1.0f;
    public const double ErfBound64 = 1.5;

    /// <summary>erfc's tail on the CPU's f32 lane: S(t) = x·e^(x²)·erfc x, t = 1/x in [0, 4/3] (x ≥ 0.75), as a polynomial in
    /// u = 1.5·t − 1 (coefficient k of u^k): S interpolated at the 17 Chebyshev nodes of the interval (S computed at 40 digits),
    /// within 3.2e-9 of S everywhere in it; Horner in u keeps its f32 evaluation within about one rounding (the monomials
    /// in t would cancel to 1e-4). erfc x = e^(−x²) · t · S: one division and 16 multiply-adds for the continued
    /// fraction's 48 to 80 serial levels.</summary>
    public static readonly double[] ErfcTail32 = [0.4823781246814764, -0.11422653907699147, 0.003084741801664283, 0.017759010812733424, -0.013435622651384071, 0.006117608127435746, -0.0013524073877195708, -0.0007986113371849265, 0.0012025644360486776, -0.0006279183665454736, 0.00031967429535662565, -0.0006445173610058078, 0.0003125770636888739, 0.000514276050142984, -0.0004131862318729763, -8.64806258308527e-05, 9.994421419212651e-05];
    public const double ErfcTailSpan = 4.0 / 3.0;

    /// <summary>erfc's tail on the f64 lane, every device: S(t) = x·e^(x²)·erfc x, t = 1/x in [0, 4/3] (x ≥ 0.75), as a polynomial
    /// in u = 1.5·t − 1 (coefficient k of u^k): S interpolated at the 41 Chebyshev nodes of the interval (S at 40 digits), within
    /// 6.3e-17 of S everywhere in it (local/erf/fit.py). erfc x = e^(−x²) · S / x: two divisions and 41 multiply-adds where the
    /// continued fraction took 100 (erf) and 400 (erfc) levels of a division each; against erfc at 30 digits over 72 000
    /// values of [0.75, 27.3] its worst 3.77 ulp, the fraction's 3.97.</summary>
    public static readonly double[] ErfcTail64 = [0.4823781246814763, -0.11422655978321611, 0.0030847543971922026, 0.017759999555634187, -0.013436227291945987, 0.0061039438608561725, -0.0013439442141137603, -0.0007147457249016516, 0.0011494978643724041, -0.0008959364314937305, 0.0004948953014102612, -0.00017564271525386842, -9.583498742993935e-06, 8.46848474905137e-05, -9.264468470325167e-05, 7.047224486285331e-05, -4.1537611169995586e-05, 1.6160720426845348e-05, 8.849842502353517e-07, -4.411641125058762e-06, 1.2469865117226472e-06, -1.309613661185913e-05, 2.7484060664809906e-05, 4.612504558563784e-06, -4.704535973536538e-05, -4.140003697812916e-06, 7.9949470856811e-05, -9.712610662098297e-06, -0.00010206542523302866, 3.314851258905504e-05, 9.260281692627065e-05, -4.640796860989371e-05, -5.929970231014169e-05, 4.1383979746813566e-05, 2.336705303566087e-05, -2.3297481387232848e-05, -3.6101198146452892e-06, 7.359089203460391e-06, -6.892050995755141e-07, -9.877323579876705e-07, 2.501299795586983e-07];

    /// <summary>erfc keeps 1 − series only where the cancellation is mild (erf below 0.71), and takes the fraction deeper.</summary>
    public const int ErfcDepth32 = 80;
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

    /// <summary>atan: u = |x| (1/|x| past 1), t = u or, past tan(π/12), (u√3 − 1)/(u + √3); the series in t²; π/6 and π/2 −
    /// added back; the sign of x (NaN kept, ±∞ to ±π/2).</summary>
    public static Vector<float> Atan(Vector<float> x)
    {
        var a = Vector.Abs(x);
        var inv = Vector.GreaterThan(a, Vector<float>.One);
        var u = Vector.ConditionalSelect(inv, Vector<float>.One / a, a);
        var mid = Vector.GreaterThan(u, new Vector<float>((float)TanPiTwelfth));
        var s3 = new Vector<float>(Sqrt3F);
        var num = Vector.FusedMultiplyAdd(u, new Vector<float>(Sqrt3LoF), Vector.FusedMultiplyAdd(u, s3, -Vector<float>.One));
        var t = Vector.ConditionalSelect(mid, num / (u + s3), u);
        var t2 = t * t;
        var p = new Vector<float>((float)AtanSeries[AtanTerms32 - 1]);
        for (int k = AtanTerms32 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, t2, new Vector<float>((float)AtanSeries[k]));
        var r = p * t;
        r = Vector.ConditionalSelect(mid, (r + new Vector<float>(PiSixthLoF)) + new Vector<float>(PiSixthF), r);
        r = Vector.ConditionalSelect(inv, (new Vector<float>(PiHalfF) - r) + new Vector<float>(PiHalfLoF), r);
        return Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), -r, r);
    }

    // erfc |x| for |x| ≥ 0.75 on the f32 lane: e^(−x²) (x² carried with its rounding, x2lo) · t · S(u) (ErfcTail32).
    private static Vector<float> ErfcTail(Vector<float> ax, Vector<float> x2)
    {
        var t = Vector<float>.One / ax;
        var u = Vector.FusedMultiplyAdd(t, new Vector<float>((float)(2 / ErfcTailSpan)), -Vector<float>.One);
        var sp = new Vector<float>((float)ErfcTail32[^1]);
        for (int k = ErfcTail32.Length - 2; k >= 0; k--) sp = Vector.FusedMultiplyAdd(sp, u, new Vector<float>((float)ErfcTail32[k]));
        var x2lo = Vector.FusedMultiplyAdd(ax, ax, -x2);
        x2lo = Vector.ConditionalSelect(Vector.Equals(x2lo, x2lo), x2lo, Vector<float>.Zero);   // x² infinite: ∞ − ∞, and e^(−x²) 0
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

    public static Vector<double> Atan(Vector<double> x)
    {
        var a = Vector.Abs(x);
        var inv = Vector.GreaterThan(a, Vector<double>.One);
        var u = Vector.ConditionalSelect(inv, Vector<double>.One / a, a);
        var mid = Vector.GreaterThan(u, new Vector<double>(TanPiTwelfth));
        var s3 = new Vector<double>(Sqrt3);
        var num = Vector.FusedMultiplyAdd(u, new Vector<double>(Sqrt3Lo), Vector.FusedMultiplyAdd(u, s3, -Vector<double>.One));
        var t = Vector.ConditionalSelect(mid, num / (u + s3), u);
        var t2 = t * t;
        var p = new Vector<double>(AtanSeries[AtanTerms64 - 1]);
        for (int k = AtanTerms64 - 2; k >= 0; k--) p = Vector.FusedMultiplyAdd(p, t2, new Vector<double>(AtanSeries[k]));
        var r = p * t;
        r = Vector.ConditionalSelect(mid, (r + new Vector<double>(PiSixthLo)) + new Vector<double>(PiSixth), r);
        r = Vector.ConditionalSelect(inv, (new Vector<double>(PiHalf) - r) + new Vector<double>(PiHalfLo), r);
        return Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), -r, r);
    }

    // erfc |x| for |x| ≥ 0.75 on the f64 lane: e^(−x²) (x² carried with its rounding, x2lo) · S(u) / x (ErfcTail64); x² infinite:
    // its low part ∞ − ∞ taken 0, e^(−x²) 0.
    private static Vector<double> ErfcTail(Vector<double> ax)
    {
        var x2 = ax * ax;
        var x2lo = Vector.FusedMultiplyAdd(ax, ax, -x2);
        x2lo = Vector.ConditionalSelect(Vector.Equals(x2lo, x2lo), x2lo, Vector<double>.Zero);
        var t = Vector<double>.One / ax;
        var u = Vector.FusedMultiplyAdd(t, new Vector<double>(1.5), -Vector<double>.One);
        var sp = new Vector<double>(ErfcTail64[^1]);
        for (int k = ErfcTail64.Length - 2; k >= 0; k--) sp = Vector.FusedMultiplyAdd(sp, u, new Vector<double>(ErfcTail64[k]));
        return Exps.Exp(-x2) * (Vector<double>.One - x2lo) * sp / ax;
    }

    public static Vector<double> Erf(Vector<double> x)
    {
        var ax = Vector.Abs(x);
        var x2 = ax * ax;
        var series = new Vector<double>(ErfSeries[ErfTerms64 - 1]);
        for (int k = ErfTerms64 - 2; k >= 0; k--) series = Vector.FusedMultiplyAdd(series, x2, new Vector<double>(ErfSeries[k]));
        series *= ax;
        var tail = Vector<double>.One - ErfcTail(ax);
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
        var tail = ErfcTail(ax);
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<double>(ErfcBound64)), Vector<double>.One - series,
            Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), new Vector<double>(2.0) - tail, tail));
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    // ---------------------------------------------------------------- f64 erf and erfc four vectors at a time
    // The same arithmetic as Erf and Erfc above, lane for lane and step for step, over four vectors side by side: their
    // series and tails are chains of dependent multiply-adds, so one vector at a time runs at the multiply-add's latency;
    // four chains in step run at its throughput. Every value is the one Erf or Erfc gives (Checks.ErfFour proves it bit
    // for bit).

    // ErfcTail for four vectors, their polynomials' steps in turn.
    private static void ErfcTail4(Vector<double> a0, Vector<double> a1, Vector<double> a2, Vector<double> a3,
        out Vector<double> t0, out Vector<double> t1, out Vector<double> t2, out Vector<double> t3)
    {
        var one = Vector<double>.One;
        var c15 = new Vector<double>(1.5);
        Vector<double> u0 = Vector.FusedMultiplyAdd(one / a0, c15, -one), u1 = Vector.FusedMultiplyAdd(one / a1, c15, -one);
        Vector<double> u2 = Vector.FusedMultiplyAdd(one / a2, c15, -one), u3 = Vector.FusedMultiplyAdd(one / a3, c15, -one);
        Vector<double> p0 = new(ErfcTail64[^1]), p1 = p0, p2 = p0, p3 = p0;
        for (int k = ErfcTail64.Length - 2; k >= 0; k--)
        {
            var c = new Vector<double>(ErfcTail64[k]);
            p0 = Vector.FusedMultiplyAdd(p0, u0, c);
            p1 = Vector.FusedMultiplyAdd(p1, u1, c);
            p2 = Vector.FusedMultiplyAdd(p2, u2, c);
            p3 = Vector.FusedMultiplyAdd(p3, u3, c);
        }

        t0 = Scaled(a0, p0); t1 = Scaled(a1, p1); t2 = Scaled(a2, p2); t3 = Scaled(a3, p3);
    }

    // e^(−x²) · (1 − x2lo) · sp / x: ErfcTail's last steps from its polynomial's value.
    private static Vector<double> Scaled(Vector<double> ax, Vector<double> sp)
    {
        var x2 = ax * ax;
        var x2lo = Vector.FusedMultiplyAdd(ax, ax, -x2);
        x2lo = Vector.ConditionalSelect(Vector.Equals(x2lo, x2lo), x2lo, Vector<double>.Zero);
        return Exps.Exp(-x2) * (Vector<double>.One - x2lo) * sp / ax;
    }

    // The f64 series Σ ErfSeries[k]·x2^k (ErfTerms64 terms) for four vectors.
    private static void Series4(Vector<double> s0, Vector<double> s1, Vector<double> s2, Vector<double> s3,
        out Vector<double> r0, out Vector<double> r1, out Vector<double> r2, out Vector<double> r3)
    {
        r0 = r1 = r2 = r3 = new Vector<double>(ErfSeries[ErfTerms64 - 1]);
        for (int k = ErfTerms64 - 2; k >= 0; k--)
        {
            var c = new Vector<double>(ErfSeries[k]);
            r0 = Vector.FusedMultiplyAdd(r0, s0, c);
            r1 = Vector.FusedMultiplyAdd(r1, s1, c);
            r2 = Vector.FusedMultiplyAdd(r2, s2, c);
            r3 = Vector.FusedMultiplyAdd(r3, s3, c);
        }
    }

    // Erf's or Erfc's last steps for one vector, from its series sum (not yet times x) and its tail (ErfcTail's value).
    private static Vector<double> ErfFrom(Vector<double> x, Vector<double> series, Vector<double> erfcTail)
    {
        var ax = Vector.Abs(x);
        series *= ax;
        var tail = Vector<double>.One - erfcTail;
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<double>(ErfBound64)), series, tail);
        y = Vector.Min(y, Vector<double>.One);
        y = Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), -y, y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    private static Vector<double> ErfcFrom(Vector<double> x, Vector<double> series, Vector<double> tail)
    {
        var ax = Vector.Abs(x);
        series *= x;
        var y = Vector.ConditionalSelect(Vector.LessThan(ax, new Vector<double>(ErfcBound64)), Vector<double>.One - series,
            Vector.ConditionalSelect(Vector.LessThan(x, Vector<double>.Zero), new Vector<double>(2.0) - tail, tail));
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    /// <summary>Erf of four vectors at once: each the value Erf gives it.</summary>
    public static void Erf4(ref Vector<double> x0, ref Vector<double> x1, ref Vector<double> x2, ref Vector<double> x3)
    {
        Vector<double> a0 = Vector.Abs(x0), a1 = Vector.Abs(x1), a2 = Vector.Abs(x2), a3 = Vector.Abs(x3);
        Series4(a0 * a0, a1 * a1, a2 * a2, a3 * a3, out var s0, out var s1, out var s2, out var s3);
        ErfcTail4(a0, a1, a2, a3, out var f0, out var f1, out var f2, out var f3);
        x0 = ErfFrom(x0, s0, f0); x1 = ErfFrom(x1, s1, f1); x2 = ErfFrom(x2, s2, f2); x3 = ErfFrom(x3, s3, f3);
    }

    /// <summary>Erfc of four vectors at once: each the value Erfc gives it.</summary>
    public static void Erfc4(ref Vector<double> x0, ref Vector<double> x1, ref Vector<double> x2, ref Vector<double> x3)
    {
        Vector<double> a0 = Vector.Abs(x0), a1 = Vector.Abs(x1), a2 = Vector.Abs(x2), a3 = Vector.Abs(x3);
        Series4(a0 * a0, a1 * a1, a2 * a2, a3 * a3, out var s0, out var s1, out var s2, out var s3);
        ErfcTail4(a0, a1, a2, a3, out var f0, out var f1, out var f2, out var f3);
        x0 = ErfcFrom(x0, s0, f0); x1 = ErfcFrom(x1, s1, f1); x2 = ErfcFrom(x2, s2, f2); x3 = ErfcFrom(x3, s3, f3);
    }

    // ---------------------------------------------------------------- scalar references (double): what the proofs compare against

    public static double ErfcReference(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 0.75) return 1 - ErfReference(x);
        double f = ax;
        for (int d = 1200; d >= 1; d--) f = ax + d * 0.5 / f;
        double hi = ax * ax, lo = Math.FusedMultiplyAdd(ax, ax, -hi);
        double tail = double.IsPositiveInfinity(hi) ? 0 : Math.Exp(-hi) * (1 - lo) * InvSqrtPi / f;   // x² past the range: e^−x² is 0 (lo ∞ − ∞)
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
            y = double.IsPositiveInfinity(hi) ? 1 : 1 - Math.Exp(-hi) * (1 - lo) * InvSqrtPi / f;   // x² past the range: e^−x² is 0 (lo ∞ − ∞)
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
