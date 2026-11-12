// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Element-wise operations ("map"): one formula applied to every value, written once over <TN, TLane, TL>. A formula
// takes up to three inputs (x, y, z) and two scalars (a, b) and gives one value; the catalogue (Ops) names each,
// gives its float64 reference and its device emitter, and says which lanes it has (every lane, or float lanes only).
//
// On the CPU a formula is a struct over the lane (IMapOp); the kernel runs it over tiles on the stack where the stored
// bytes are not lanes, in place where they are, on the team. The lane is the number's element-wise lane (integer types
// their own; the small floats f32; float64 f64): what the type stores is what the formula computes in, with the
// result rounded once on the way back. Backward formulas are formulas too: inputs (x, dy) or (x, y, dz).

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

// ---------------------------------------------------------------------------------------------- the formulas as lane structs

public interface IMapOp<TLane, TL> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
{
    static abstract Vector<TLane> Apply(Vector<TLane> x, Vector<TLane> y, Vector<TLane> z, Vector<TLane> a, Vector<TLane> b);

    /// <summary>Whether Apply4 runs four vectors side by side: a formula whose dependent chains are long enough that one
    /// vector at a time waits on each step's latency (the float64 erf and erfc's continued fractions).</summary>
    static virtual bool Four => false;

    /// <summary>Four vectors (x and y 4·Count values each; o may be x) at once, each value the one Apply gives.</summary>
    static virtual void Apply4(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> y, Span<TLane> o, Vector<TLane> a, Vector<TLane> b) => throw new NotSupportedException();
}

// The float64 erf-based formulas four vectors at a time (Funcs.Erf4, Erfc4: the same values as Erf and Erfc, their chains
// in step).
internal static class Four
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Load<T>(ReadOnlySpan<T> s, out Vector<double> v0, out Vector<double> v1, out Vector<double> v2, out Vector<double> v3) where T : unmanaged
    {
        var d = MemoryMarshal.Cast<T, double>(s);
        int w = Vector<double>.Count;
        v0 = new(d); v1 = new(d[w..]); v2 = new(d[(2 * w)..]); v3 = new(d[(3 * w)..]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store<T>(Span<T> s, Vector<double> v0, Vector<double> v1, Vector<double> v2, Vector<double> v3) where T : unmanaged
    {
        var d = MemoryMarshal.Cast<T, double>(s);
        int w = Vector<double>.Count;
        v0.CopyTo(d); v1.CopyTo(d[w..]); v2.CopyTo(d[(2 * w)..]); v3.CopyTo(d[(3 * w)..]);
    }

    // Φ(x) = ½·erfc(−x/√2) of four vectors, as PhiF computes it in the f64 lane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Phi(ref Vector<double> x0, ref Vector<double> x1, ref Vector<double> x2, ref Vector<double> x3)
    {
        var k = new Vector<double>(-MapOps.InvSqrt2);
        Vector<double> t0 = x0 * k, t1 = x1 * k, t2 = x2 * k, t3 = x3 * k;
        Funcs.Erfc4(ref t0, ref t1, ref t2, ref t3);
        var h = new Vector<double>(0.5);
        x0 = h * t0; x1 = h * t1; x2 = h * t2; x3 = h * t3;
    }
}

/// <summary>What runs a formula once its struct is known: a map over values, over rows.</summary>
public interface IOpUse<TLane, TL> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
{
    void Use<TOp>() where TOp : struct, IMapOp<TLane, TL>;
}

internal static class MapOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> C<T, TL>(double v) where T : unmanaged where TL : struct, ILane<TL, T> => new(TL.From(v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    // Whether p ranks above q in Adad's order (Routing.Order), lane by lane: larger, or a NaN against a number (an integer
    // lane's p > q: no lane equals itself falsely there).
    private static Vector<T> RanksAbove<T, TL>(Vector<T> p, Vector<T> q) where T : unmanaged where TL : struct, ILane<TL, T> =>
        TL.Greater(p, q) | Vector.AndNot(TL.Equal(q, q), TL.Equal(p, p));

    // Integer division by Ops.IntegerDivision: toward zero; a signed lane's x / −1 is −x, wrapping (its least value over −1
    // is itself); x / 0 = 0. The lane divides by 1 where the divisor is 0 or a signed −1: no division traps.
    private static Vector<T> Quotient<T, TL>(Vector<T> x, Vector<T> y) where T : unmanaged where TL : struct, ILane<TL, T>
    {
        bool signed = typeof(T) == typeof(int) || typeof(T) == typeof(long) || typeof(T) == typeof(short) || typeof(T) == typeof(sbyte);
        var zero = Vector.Equals(y, Vector<T>.Zero);
        var minusOne = signed ? Vector.Equals(y, Vector<T>.AllBitsSet) : Vector<T>.Zero;
        var q = TL.Div(x, Vector.ConditionalSelect(zero | minusOne, Vector<T>.One, y));
        q = Vector.ConditionalSelect(minusOne, Vector<T>.Zero - x, q);
        return Vector.ConditionalSelect(zero, Vector<T>.Zero, q);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> Sign<T, TL>(Vector<T> x) where T : unmanaged where TL : struct, ILane<TL, T>
    {
        var one = C<T, TL>(1);
        var pos = TL.Greater(x, Vector<T>.Zero);
        var neg = TL.Greater(Vector<T>.Zero, x);
        return TL.Select(pos, one, TL.Select(neg, TL.Neg(one), Vector<T>.Zero));
    }

    // any lane
    public struct Copy<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => x; }
    public struct Fill<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => a; }
    public struct Neg<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Neg(x); }
    public struct Abs<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Abs(x); }
    public struct SignOp<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => Sign<T, TL>(x); }
    public struct Square<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, x); }
    public struct Relu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Max(x, Vector<T>.Zero); }
    public struct AddScalar<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Add(x, a); }
    public struct Scale<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, a); }
    public struct Affine<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Fma(a, x, b); }
    public struct Clamp<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Min(TL.Max(x, a), b); }
    public struct Add<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Add(x, y); }
    public struct Sub<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Sub(x, y); }
    public struct Mul<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, y); }
    public struct Div<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => typeof(T) == typeof(float) || typeof(T) == typeof(double) ? TL.Div(x, y) : Quotient<T, TL>(x, y); }
    public struct Max<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Max(x, y); }
    // max and min by Adad's order: the first-ranked (the lane's max: IEEE maximum, a NaN where either is one) and the
    // last-ranked (IEEE minimumNumber: the number where one is a NaN).
    public struct Min<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => Vector.MinNumber(x, y); }
    public struct Axpy<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Fma(a, x, y); }
    public struct MulAdd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Fma(x, y, z); }
    public struct Where<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Equal(x, Vector<T>.Zero), z, y); }
    public struct SquareBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(C<T, TL>(2), x), y); }
    public struct AbsBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(Sign<T, TL>(x), y); }
    public struct ReluBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), y, Vector<T>.Zero); }
    public struct ClampBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, a) & TL.Greater(b, x), y, Vector<T>.Zero); }
    public struct WhereBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Equal(x, Vector<T>.Zero), Vector<T>.Zero, y); }
    // The gradient of max and min to the operand they give (a tie to x).
    public struct MaxBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(RanksAbove<T, TL>(y, x), Vector<T>.Zero, z); }
    public struct MaxBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(RanksAbove<T, TL>(y, x), z, Vector<T>.Zero); }
    public struct MinBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(RanksAbove<T, TL>(x, y), Vector<T>.Zero, z); }
    public struct MinBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(RanksAbove<T, TL>(x, y), z, Vector<T>.Zero); }
    public struct GatedRelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Max(x, Vector<T>.Zero), y); }
    public struct GatedReluBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), TL.Mul(y, z), Vector<T>.Zero); }
    public struct GatedReluBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Max(x, Vector<T>.Zero), z); }

    // float lanes
    internal const double GeluK = 0.7978845608028654, GeluC = 0.044715, InvSqrt2 = 0.7071067811865476, InvSqrt2Pi = 0.3989422804014327;

    // gelu = x·σ(2u), u = √(2/π)(x + 0.044715x³); its gradient σ(2u) + 2x·σ(2u)σ(−2u)·u′ (1 − tanh²u = 4σ(2u)σ(−2u)): no cancellation.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> GeluF<T, TL>(Vector<T> x) where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        var u2 = TL.Mul(C<T, TL>(2 * GeluK), TL.Fma(TL.Mul(C<T, TL>(GeluC), TL.Mul(x, x)), x, x));
        return TL.Mul(x, TL.Sigmoid(u2));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> GeluGradF<T, TL>(Vector<T> x) where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        var x2 = TL.Mul(x, x);
        var u2 = TL.Mul(C<T, TL>(2 * GeluK), TL.Fma(TL.Mul(C<T, TL>(GeluC), x2), x, x));
        var s = TL.Sigmoid(u2);
        var sn = TL.Sigmoid(TL.Neg(u2));
        var du = TL.Mul(C<T, TL>(GeluK), TL.Fma(C<T, TL>(3 * GeluC), x2, C<T, TL>(1)));
        return TL.Fma(TL.Mul(TL.Mul(C<T, TL>(2), x), TL.Mul(s, sn)), du, s);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> SiluGradF<T, TL>(Vector<T> x) where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        var s = TL.Sigmoid(x);
        return TL.Mul(s, TL.Fma(x, TL.Sigmoid(TL.Neg(x)), C<T, TL>(1)));
    }

    // Φ(x) = ½·erfc(−x/√2), the normal distribution, without cancellation for x < 0.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> PhiF<T, TL>(Vector<T> x) where T : unmanaged where TL : struct, IFloatLane<TL, T> =>
        TL.Mul(C<T, TL>(0.5), TL.Erfc(TL.Mul(x, C<T, TL>(-InvSqrt2))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> PowF<T, TL>(Vector<T> x, Vector<T> a) where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        // |x|^a; x < 0 keeps its sign when a is an odd integer (otherwise the result is NaN as the real power is).
        var ax = TL.Abs(x);
        var mag = TL.Exp(TL.Mul(a, TL.Log(ax)));
        var half = TL.Mul(a, C<T, TL>(0.5));
        var integer = TL.Equal(TL.Round(a), a);
        var odd = integer & ~TL.Equal(TL.Round(half), half);
        var neg = TL.Greater(Vector<T>.Zero, x);
        var nan = C<T, TL>(double.NaN);
        return TL.Select(neg, TL.Select(odd, TL.Neg(mag), TL.Select(integer, mag, nan)), mag);
    }

    public struct Sqrt<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Sqrt(x); }
    public struct InvSqrt<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Div(C<T, TL>(1), TL.Sqrt(TL.Add(x, a))); }
    public struct Exp<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Exp(x); }
    public struct Expm1<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Expm1(x); }
    public struct Log<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Log(x); }
    public struct Sin<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Sin(x); }
    public struct Cos<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Cos(x); }
    public struct Tanh<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Tanh(x); }
    public struct Atan<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Atan(x); }
    public struct Sigmoid<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Sigmoid(x); }
    public struct Erf<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Erf(x);

        public static bool Four => typeof(T) == typeof(double);

        public static void Apply4(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> o, Vector<T> a, Vector<T> b)
        {
            Cpu.Four.Load(x, out var v0, out var v1, out var v2, out var v3);
            Funcs.Erf4(ref v0, ref v1, ref v2, ref v3);
            Cpu.Four.Store(o, v0, v1, v2, v3);
        }
    }
    public struct Gelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => GeluF<T, TL>(x); }
    public struct GeluErf<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, PhiF<T, TL>(x));

        public static bool Four => typeof(T) == typeof(double);

        public static void Apply4(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> o, Vector<T> a, Vector<T> b)
        {
            Cpu.Four.Load(x, out var x0, out var x1, out var x2, out var x3);
            Vector<double> p0 = x0, p1 = x1, p2 = x2, p3 = x3;
            Cpu.Four.Phi(ref p0, ref p1, ref p2, ref p3);
            Cpu.Four.Store(o, x0 * p0, x1 * p1, x2 * p2, x3 * p3);
        }
    }
    public struct Silu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, TL.Sigmoid(x)); }
    public struct LeakyRelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), x, TL.Mul(a, x)); }
    public struct Pow<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => PowF<T, TL>(x, a); }
    public struct MaskScale<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Equal(y, Vector<T>.Zero), Vector<T>.Zero, TL.Mul(x, a)); }
    public struct GatedSilu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(x, TL.Sigmoid(x)), y); }
    public struct GatedGelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(GeluF<T, TL>(x), y); }
    public struct SqrtBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Div(y, TL.Mul(C<T, TL>(2), TL.Sqrt(x))); }
    public struct InvSqrtBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var s = TL.Add(x, a); var r = TL.Div(C<T, TL>(1), TL.Sqrt(s)); return TL.Mul(TL.Mul(C<T, TL>(-0.5), TL.Div(r, s)), y); } }
    public struct ExpBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Exp(x), y); }
    public struct LogBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Div(y, x); }
    public struct SinBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Cos(x), y); }
    public struct CosBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Neg(TL.Mul(TL.Sin(x), y)); }
    public struct TanhBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var s = TL.Sigmoid(TL.Mul(C<T, TL>(2), x)); return TL.Mul(TL.Mul(C<T, TL>(4), TL.Mul(s, TL.Sigmoid(TL.Mul(C<T, TL>(-2), x)))), y); } }
    public struct Softcap<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(a, TL.Tanh(TL.Div(TL.Mul(b, x), a))); }
    public struct SoftcapBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var u = TL.Div(TL.Mul(b, x), a); var s = TL.Sigmoid(TL.Mul(C<T, TL>(2), u)); return TL.Mul(TL.Mul(TL.Mul(C<T, TL>(4), TL.Mul(s, TL.Sigmoid(TL.Mul(C<T, TL>(-2), u)))), b), y); } }
    public struct SigmoidBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(TL.Sigmoid(x), TL.Sigmoid(TL.Neg(x))), y); }
    public struct SiluBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(SiluGradF<T, TL>(x), y); }
    public struct GeluBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(GeluGradF<T, TL>(x), y); }
    public struct GeluErfBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b)
        {
            var phi = PhiF<T, TL>(x);
            var dens = TL.Mul(C<T, TL>(InvSqrt2Pi), TL.Exp(TL.Mul(C<T, TL>(-0.5), TL.Mul(x, x))));
            return TL.Mul(TL.Fma(x, dens, phi), y);
        }

        public static bool Four => typeof(T) == typeof(double);

        public static void Apply4(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> o, Vector<T> a, Vector<T> b)
        {
            Cpu.Four.Load(x, out var x0, out var x1, out var x2, out var x3);
            Cpu.Four.Load(y, out var y0, out var y1, out var y2, out var y3);
            Vector<double> p0 = x0, p1 = x1, p2 = x2, p3 = x3;
            Cpu.Four.Phi(ref p0, ref p1, ref p2, ref p3);
            Cpu.Four.Store(o, Back(x0, p0, y0), Back(x1, p1, y1), Back(x2, p2, y2), Back(x3, p3, y3));
        }

        // Apply's arithmetic in the f64 lane from Φ.
        private static Vector<double> Back(Vector<double> x, Vector<double> phi, Vector<double> y)
        {
            var dens = new Vector<double>(InvSqrt2Pi) * Exps.Exp(new Vector<double>(-0.5) * (x * x));
            return Vector.FusedMultiplyAdd(x, dens, phi) * y;
        }
    }
    public struct LeakyReluBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), y, TL.Mul(a, y)); }
    public struct PowBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(a, PowF<T, TL>(x, TL.Sub(a, C<T, TL>(1)))), y); }
    public struct GatedSiluBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(SiluGradF<T, TL>(x), y), z); }
    public struct GatedSiluBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(x, TL.Sigmoid(x)), z); }
    public struct GatedGeluBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(GeluGradF<T, TL>(x), y), z); }
    public struct GatedGeluBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(GeluF<T, TL>(x), z); }
    public struct GatedTanhSigmoid<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Tanh(x), TL.Sigmoid(y)); }
    public struct GatedTanhSigmoidBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var s = TL.Sigmoid(TL.Mul(C<T, TL>(2), x)); var sech2 = TL.Mul(C<T, TL>(4), TL.Mul(s, TL.Sigmoid(TL.Mul(C<T, TL>(-2), x)))); return TL.Mul(TL.Mul(sech2, TL.Sigmoid(y)), z); } }
    public struct GatedTanhSigmoidBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(TL.Tanh(x), TL.Mul(TL.Sigmoid(y), TL.Sigmoid(TL.Neg(y)))), z); }
    public struct AddGelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => GeluF<T, TL>(TL.Add(x, y)); }
    // log1p(u) = log(w)·u / (w − 1), w = 1 + u (Goldberg's), u where w is 1.
    private static Vector<T> Log1pF<T, TL>(Vector<T> u) where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        var one = C<T, TL>(1);
        var w = TL.Add(one, u);
        var d = TL.Sub(w, one);
        return TL.Select(TL.Greater(d, Vector<T>.Zero), TL.Div(TL.Mul(TL.Log(w), u), d), u);
    }

    // The focal loss's parts (Abstraction/Map.cs): BCE, σ(x), σ(−x), 1 − p_t, (1 − p_t)^γ (1 where γ is 0, 0^0 among them),
    // α_t = α·t + (1 − α)·(1 − t).
    private static (Vector<T> Ce, Vector<T> S, Vector<T> Sn, Vector<T> Q, Vector<T> Qg, Vector<T> At) FocalParts<T, TL>(Vector<T> x, Vector<T> t, Vector<T> a, Vector<T> g)
        where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        var one = C<T, TL>(1);
        var ce = TL.Add(TL.Fma(TL.Neg(x), t, TL.Max(x, Vector<T>.Zero)), Log1pF<T, TL>(TL.Exp(TL.Neg(TL.Abs(x)))));
        var s = TL.Sigmoid(x);
        var sn = TL.Sigmoid(TL.Neg(x));
        var u = TL.Sub(one, t);
        var q = TL.Add(TL.Mul(t, sn), TL.Mul(u, s));
        var qg = TL.Select(TL.Equal(g, Vector<T>.Zero), one, PowF<T, TL>(q, g));
        var at = TL.Add(TL.Mul(a, t), TL.Mul(TL.Sub(one, a), u));
        return (ce, s, sn, q, qg, at);
    }

    public struct SigmoidFocal<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b)
        {
            var f = FocalParts<T, TL>(x, y, a, b);
            return TL.Mul(TL.Mul(f.At, f.Qg), f.Ce);
        }
    }

    public struct SigmoidFocalBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b)
        {
            var f = FocalParts<T, TL>(x, y, a, b);
            var one = C<T, TL>(1);
            var d = TL.Sub(TL.Mul(TL.Sub(one, y), f.S), TL.Mul(y, f.Sn));
            // γ·BCE·q^(γ−1)·(2t − 1)·σ(x)σ(−x), 0 where γ or q is 0 (its limit at q = 0 for γ > −1).
            var dead = TL.Equal(f.Q, Vector<T>.Zero) | TL.Equal(b, Vector<T>.Zero);
            var second = TL.Mul(TL.Mul(TL.Mul(TL.Mul(b, f.Ce), TL.Div(f.Qg, f.Q)), TL.Sub(TL.Add(y, y), one)), TL.Mul(f.S, f.Sn));
            second = TL.Select(dead, Vector<T>.Zero, second);
            return TL.Mul(TL.Mul(f.At, TL.Sub(TL.Mul(d, f.Qg), second)), z);
        }
    }

    public struct ClipFactor<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var one = C<T, TL>(1); return TL.Select(TL.Greater(x, Vector<T>.Zero), TL.Min(one, TL.Div(a, TL.Sqrt(x))), one); } }

    /// <summary>The formulas every lane has: runs `use` with op's struct; false for any other name.</summary>
    public static bool AnyOp<T, TL, TU>(string op, ref TU use) where T : unmanaged where TL : struct, ILane<TL, T> where TU : struct, IOpUse<T, TL>
    {
        switch (op)
        {
            case "copy": use.Use<Copy<T, TL>>(); return true;
            case "fill": use.Use<Fill<T, TL>>(); return true;
            case "neg": use.Use<Neg<T, TL>>(); return true;
            case "abs": use.Use<Abs<T, TL>>(); return true;
            case "sign": use.Use<SignOp<T, TL>>(); return true;
            case "square": use.Use<Square<T, TL>>(); return true;
            case "relu": use.Use<Relu<T, TL>>(); return true;
            case "add_scalar": use.Use<AddScalar<T, TL>>(); return true;
            case "scale": use.Use<Scale<T, TL>>(); return true;
            case "affine": use.Use<Affine<T, TL>>(); return true;
            case "clamp": use.Use<Clamp<T, TL>>(); return true;
            case "add": use.Use<Add<T, TL>>(); return true;
            case "sub": use.Use<Sub<T, TL>>(); return true;
            case "mul": use.Use<Mul<T, TL>>(); return true;
            case "div": use.Use<Div<T, TL>>(); return true;
            case "max": use.Use<Max<T, TL>>(); return true;
            case "min": use.Use<Min<T, TL>>(); return true;
            case "axpy": use.Use<Axpy<T, TL>>(); return true;
            case "muladd": use.Use<MulAdd<T, TL>>(); return true;
            case "where": use.Use<Where<T, TL>>(); return true;
            case "square_bwd": use.Use<SquareBwd<T, TL>>(); return true;
            case "abs_bwd": use.Use<AbsBwd<T, TL>>(); return true;
            case "relu_bwd": use.Use<ReluBwd<T, TL>>(); return true;
            case "clamp_bwd": use.Use<ClampBwd<T, TL>>(); return true;
            case "where_bwd": use.Use<WhereBwd<T, TL>>(); return true;
            case "max_bwd_x": use.Use<MaxBwdX<T, TL>>(); return true;
            case "max_bwd_y": use.Use<MaxBwdY<T, TL>>(); return true;
            case "min_bwd_x": use.Use<MinBwdX<T, TL>>(); return true;
            case "min_bwd_y": use.Use<MinBwdY<T, TL>>(); return true;
            case "gated_relu": use.Use<GatedRelu<T, TL>>(); return true;
            case "gated_relu_bwd_x": use.Use<GatedReluBwdX<T, TL>>(); return true;
            case "gated_relu_bwd_y": use.Use<GatedReluBwdY<T, TL>>(); return true;
            default: return false;
        }
    }

    /// <summary>The float formulas: runs `use` with op's struct; false for any other name.</summary>
    public static bool FloatOp<T, TL, TU>(string op, ref TU use) where T : unmanaged where TL : struct, IFloatLane<TL, T> where TU : struct, IOpUse<T, TL>
    {
        switch (op)
        {
            case "sqrt": use.Use<Sqrt<T, TL>>(); return true;
            case "inv_sqrt": use.Use<InvSqrt<T, TL>>(); return true;
            case "exp": use.Use<Exp<T, TL>>(); return true;
            case "expm1": use.Use<Expm1<T, TL>>(); return true;
            case "log": use.Use<Log<T, TL>>(); return true;
            case "sin": use.Use<Sin<T, TL>>(); return true;
            case "cos": use.Use<Cos<T, TL>>(); return true;
            case "tanh": use.Use<Tanh<T, TL>>(); return true;
            case "atan": use.Use<Atan<T, TL>>(); return true;
            case "sigmoid": use.Use<Sigmoid<T, TL>>(); return true;
            case "erf": use.Use<Erf<T, TL>>(); return true;
            case "gelu": use.Use<Gelu<T, TL>>(); return true;
            case "gelu_erf": use.Use<GeluErf<T, TL>>(); return true;
            case "silu": use.Use<Silu<T, TL>>(); return true;
            case "leaky_relu": use.Use<LeakyRelu<T, TL>>(); return true;
            case "pow": use.Use<Pow<T, TL>>(); return true;
            case "mask_scale": use.Use<MaskScale<T, TL>>(); return true;
            case "gated_silu": use.Use<GatedSilu<T, TL>>(); return true;
            case "gated_gelu": use.Use<GatedGelu<T, TL>>(); return true;
            case "sqrt_bwd": use.Use<SqrtBwd<T, TL>>(); return true;
            case "inv_sqrt_bwd": use.Use<InvSqrtBwd<T, TL>>(); return true;
            case "exp_bwd": use.Use<ExpBwd<T, TL>>(); return true;
            case "log_bwd": use.Use<LogBwd<T, TL>>(); return true;
            case "sin_bwd": use.Use<SinBwd<T, TL>>(); return true;
            case "cos_bwd": use.Use<CosBwd<T, TL>>(); return true;
            case "tanh_bwd": use.Use<TanhBwd<T, TL>>(); return true;
            case "softcap": use.Use<Softcap<T, TL>>(); return true;
            case "softcap_bwd": use.Use<SoftcapBwd<T, TL>>(); return true;
            case "sigmoid_bwd": use.Use<SigmoidBwd<T, TL>>(); return true;
            case "silu_bwd": use.Use<SiluBwd<T, TL>>(); return true;
            case "gelu_bwd": use.Use<GeluBwd<T, TL>>(); return true;
            case "gelu_erf_bwd": use.Use<GeluErfBwd<T, TL>>(); return true;
            case "leaky_relu_bwd": use.Use<LeakyReluBwd<T, TL>>(); return true;
            case "pow_bwd": use.Use<PowBwd<T, TL>>(); return true;
            case "gated_silu_bwd_x": use.Use<GatedSiluBwdX<T, TL>>(); return true;
            case "gated_silu_bwd_y": use.Use<GatedSiluBwdY<T, TL>>(); return true;
            case "gated_gelu_bwd_x": use.Use<GatedGeluBwdX<T, TL>>(); return true;
            case "gated_gelu_bwd_y": use.Use<GatedGeluBwdY<T, TL>>(); return true;
            case "gated_tanh_sigmoid": use.Use<GatedTanhSigmoid<T, TL>>(); return true;
            case "gated_tanh_sigmoid_bwd_x": use.Use<GatedTanhSigmoidBwdX<T, TL>>(); return true;
            case "gated_tanh_sigmoid_bwd_y": use.Use<GatedTanhSigmoidBwdY<T, TL>>(); return true;
            case "add_gelu": use.Use<AddGelu<T, TL>>(); return true;
            case "clip_factor": use.Use<ClipFactor<T, TL>>(); return true;
            case "sigmoid_focal": use.Use<SigmoidFocal<T, TL>>(); return true;
            case "sigmoid_focal_bwd": use.Use<SigmoidFocalBwd<T, TL>>(); return true;
            default: return false;
        }
    }
}

// ---------------------------------------------------------------------------------------------- the CPU kernel

internal sealed partial class Kernels
{
    /// <summary>Runs a formula over n stored values in the number's element-wise lane (the formulas every lane has).</summary>
    public static string Map<TN>(string op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) where TN : struct, IAdad<TN>
    {
        (a, b) = Stored.Held<TN>(a, b);
        var any = new MapVisitor(op, x, y, z, output, n, a, b);
        if (!any.Known) throw new NotSupportedException($"{op}: a float formula; {TN.Name} runs it through MapFloat");
        TN.Elementwise(ref any);
        return any.Lane;
    }

    /// <summary>Runs any formula over a float type: the float formulas in its float lane, the others in its element-wise lane.</summary>
    public static string MapFloat<TN>(string op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) where TN : struct, IFloatAdad<TN>
    {
        (a, b) = Stored.Held<TN>(a, b);
        var any = new MapVisitor(op, x, y, z, output, n, a, b);
        if (any.Known)
        {
            TN.Elementwise(ref any);
            return any.Lane;
        }

        var fl = new FloatMapVisitor(op, x, y, z, output, n, a, b);
        TN.Float(ref fl);
        return fl.Lane;
    }

    public struct MapVisitor(string op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : ILaneVisitor
    {
        public string Lane = "";

        public readonly bool Known => !Ops.Find(op).FloatOnly;

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var use = new MapUse<TN, TLane, TL>(x, y, z, output, n, a, b);
            if (!MapOps.AnyOp<TLane, TL, MapUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException(op);
        }
    }

    public struct FloatMapVisitor(string op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            var use = new MapUse<TN, TLane, TL>(x, y, z, output, n, a, b);
            if (!MapOps.FloatOp<TLane, TL, MapUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException(op);
        }
    }

    private readonly struct MapUse<TN, TLane, TL>(byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : IOpUse<TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Use<TOp>() where TOp : struct, IMapOp<TLane, TL> => MapRun<TN, TLane, TL, TOp>(x, y, z, output, n, a, b);
    }

    private static void MapRun<TN, TLane, TL, TOp>(byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
        int inputs = (x is null ? 0 : 1) + (y is null ? 0 : 1) + (z is null ? 0 : 1);
        bool stream = Streams.Past((inputs + 1) * TN.Bytes(n));
        Team.For(n, size, (long)n * 4, new MapBody<TN, TLane, TL, TOp>(x, y, z, output, TL.From(a), TL.From(b), stream));
    }

    // stream: the result copied from the stack into the output past the caches (Streams), a tile at a time.
    private struct MapBody<TN, TLane, TL, TOp>(byte[]? x, byte[]? y, byte[]? z, byte[] output, TLane a, TLane b, bool stream) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var va = new Vector<TLane>(a);
            var vb = new Vector<TLane>(b);
            if (TN.Direct && !stream)
            {
                var ox = x is null ? default : MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(first, last - first);
                var oy = y is null ? default : MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(first, last - first);
                var oz = z is null ? default : MemoryMarshal.Cast<byte, TLane>(z.AsSpan()).Slice(first, last - first);
                Apply(ox, oy, oz, MemoryMarshal.Cast<byte, TLane>(output.AsSpan()).Slice(first, last - first), va, vb);
                return;
            }

            if (TN.Direct)
            {
                // Each tile computed on the stack from the inputs in place, then streamed into the output.
                Span<TLane> to = stackalloc TLane[Tile];
                var lanes = MemoryMarshal.Cast<byte, TLane>(output.AsSpan());
                for (int at = first; at < last; at += Tile)
                {
                    int len = Math.Min(Tile, last - at);
                    var ox = x is null ? default : MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(at, len);
                    var oy = y is null ? default : MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(at, len);
                    var oz = z is null ? default : MemoryMarshal.Cast<byte, TLane>(z.AsSpan()).Slice(at, len);
                    Apply(ox, oy, oz, to[..len], va, vb);
                    Streams.Copy(MemoryMarshal.AsBytes(to[..len]), MemoryMarshal.AsBytes(lanes.Slice(at, len)));
                }

                Streams.Fence();
                return;
            }

            Span<TLane> tx = stackalloc TLane[Tile];
            Span<TLane> ty = stackalloc TLane[Tile];
            Span<TLane> tz = stackalloc TLane[Tile];
            Span<byte> stored = stream ? stackalloc byte[(int)TN.Bytes(Tile)] : default;   // once: a stackalloc in the loop grows the stack each tile
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                if (x is not null) TN.Load(x, at, tx[..len]);
                if (y is not null) TN.Load(y, at, ty[..len]);
                if (z is not null) TN.Load(z, at, tz[..len]);
                Apply(x is null ? default : tx[..len], y is null ? default : ty[..len], z is null ? default : tz[..len], tx[..len], va, vb);
                if (!TN.IsFloat && TN.Bits < 8 * Unsafe.SizeOf<TLane>()) Wrap(tx[..len], TN.Bits, TN.Min < 0);
                if (!stream)
                {
                    TN.Store(tx[..len], output, at);
                    continue;
                }

                // The tile's stored bytes (a tile is whole storage units) made on the stack, then streamed.
                int bytes = (int)TN.Bytes(len);
                TN.Store(tx[..len], stored[..bytes], 0);
                Streams.Copy(stored[..bytes], output.AsSpan((int)TN.Bytes(at), bytes));
            }

            if (stream) Streams.Fence();
        }

        // Integer results of a type narrower than its lane, wrapped at the type's width (the arithmetic is modular there, as
        // it is in a lane of the type's own width): the low bits, sign-extended for a signed type.
        private static void Wrap(Span<TLane> v, int bits, bool signed)
        {
            int w = Vector<TLane>.Count, i = 0;
            var mask = new Vector<TLane>(Widths.FromLong<TLane>((1L << bits) - 1));
            var half = new Vector<TLane>(Widths.FromLong<TLane>(1L << (bits - 1)));
            ref TLane r = ref MemoryMarshal.GetReference(v);
            for (; i + w <= v.Length; i += w)
            {
                var t = Vector.LoadUnsafe(ref r, (nuint)i);
                t = signed ? ((t + half) & mask) - half : t & mask;             // ((v + 2^(b−1)) mod 2^b) − 2^(b−1), or v mod 2^b
                t.StoreUnsafe(ref r, (nuint)i);
            }

            for (; i < v.Length; i++)
            {
                long l = Widths.AsLong(v[i]), m = (1L << bits) - 1, low = l & m;
                if (signed && (low & (1L << (bits - 1))) != 0) low -= 1L << bits;
                v[i] = Widths.FromLong<TLane>(low);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Apply(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> y, ReadOnlySpan<TLane> z, Span<TLane> o, Vector<TLane> a, Vector<TLane> b)
        {
            int w = Vector<TLane>.Count, i = 0, count = o.Length;
            ref TLane rx = ref x.IsEmpty ? ref Unsafe.NullRef<TLane>() : ref MemoryMarshal.GetReference(x);
            ref TLane ry = ref y.IsEmpty ? ref Unsafe.NullRef<TLane>() : ref MemoryMarshal.GetReference(y);
            ref TLane rz = ref z.IsEmpty ? ref Unsafe.NullRef<TLane>() : ref MemoryMarshal.GetReference(z);
            ref TLane ro = ref MemoryMarshal.GetReference(o);
            if (TOp.Four)
                for (; i + 4 * w <= count; i += 4 * w) TOp.Apply4(x.Slice(i, 4 * w), y.IsEmpty ? default : y.Slice(i, 4 * w), o.Slice(i, 4 * w), a, b);
            for (; i + w <= count; i += w)
            {
                var vx = x.IsEmpty ? default : Vector.LoadUnsafe(ref rx, (nuint)i);
                var vy = y.IsEmpty ? default : Vector.LoadUnsafe(ref ry, (nuint)i);
                var vz = z.IsEmpty ? default : Vector.LoadUnsafe(ref rz, (nuint)i);
                TOp.Apply(vx, vy, vz, a, b).StoreUnsafe(ref ro, (nuint)i);
            }

            if (i < count)
            {
                // The tail through one padded vector.
                // The padding lanes hold 1 (a divisor must not be 0); their results are dropped.
                Span<TLane> px = stackalloc TLane[w];
                Span<TLane> py = stackalloc TLane[w];
                Span<TLane> pz = stackalloc TLane[w];
                px.Fill(TL.From(1)); py.Fill(TL.From(1)); pz.Fill(TL.From(1));
                int rest = count - i;
                if (!x.IsEmpty) x.Slice(i, rest).CopyTo(px);
                if (!y.IsEmpty) y.Slice(i, rest).CopyTo(py);
                if (!z.IsEmpty) z.Slice(i, rest).CopyTo(pz);
                var r = TOp.Apply(new Vector<TLane>(px), new Vector<TLane>(py), new Vector<TLane>(pz), a, b);
                for (int l = 0; l < rest; l++) o[i + l] = r[l];
            }
        }
    }

    // ---------------------------------------------------------------- dropout from a seed

    /// <summary>output = residual + x·(1/(1 − p)) where the seed's draw i keeps value i, the residual where it does not (no
    /// residual: 0), every type but bool: a kept value in the type's float lane (a kept value with a residual one fused
    /// multiply-add) stored by the type's own conversion, a dropped one the residual as it is stored.</summary>
    public static string Dropout<TN>(byte[] x, byte[]? residual, byte[] output, int n, double p, ulong seed) where TN : struct, IAdad<TN>
    {
        if (TN.Bits == 1) throw new NotSupportedException($"dropout: {TN.Name} holds truth values, which 1/(1 − p) cannot scale");
        return InFloatLane<TN, DropoutCall>(new DropoutCall(x, residual, output, n, Draws.Key(seed), Draws.Threshold(p), 1 / (1 - p)));
    }

    private readonly struct DropoutCall(byte[] x, byte[]? residual, byte[] output, int n, DrawKey key, uint threshold, double scale) : IFloatOf
    {
        public string Run<TN, TE, TLane, TL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, (long)n * 4, new DropoutBody<TN, TE, TLane, TL>(x, residual, output, key, threshold, TL.From(scale)));
            return TL.Name;
        }
    }

    // keep[i] all ones where draw first + i keeps its value (Draws.Keep: its bits reach the threshold), all zeros where not:
    // the draws a vector of counters at a time (Draws.Bits's steps over Vector<uint>, exact in every lane), each mask then
    // widened to the lane's width.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Keeps<TLane>(DrawKey key, uint first, uint threshold, Span<TLane> keep) where TLane : unmanaged
    {
        int n = keep.Length, u = Vector<uint>.Count, i = 0, size = Unsafe.SizeOf<TLane>();
        if (size is 4 or 8)
        {
            Vector<uint> a = new(key.A), b = new(key.B), t = new(threshold), step = new((uint)u), counter = Vector<uint>.Indices + new Vector<uint>(first);
            ref TLane r = ref MemoryMarshal.GetReference(keep);
            for (; i + u <= n; i += u, counter += step)
            {
                var mask = Vector.GreaterThanOrEqual(Mix(Mix(counter ^ a) + b), t);
                if (size == 4)
                {
                    Vector.AsVectorUInt32(mask).StoreUnsafe(ref Unsafe.As<TLane, uint>(ref r), (nuint)i);
                }
                else
                {
                    Vector.Widen(Vector.AsVectorInt32(mask), out var lo, out var hi);    // −1 stays −1: all ones at 64 bits
                    ref long rl = ref Unsafe.As<TLane, long>(ref r);
                    lo.StoreUnsafe(ref rl, (nuint)i);
                    hi.StoreUnsafe(ref rl, (nuint)(i + u / 2));
                }
            }
        }

        var bytes = MemoryMarshal.AsBytes(keep);
        for (; i < n; i++) bytes.Slice(i * size, size).Fill(Draws.Keep(key, first + (uint)i, threshold) ? (byte)0xFF : (byte)0);
    }

    // Draws.Mix over a vector of 32-bit words: the same xor-shifts and odd products, modulo 2^32 in every lane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> Mix(Vector<uint> h)
    {
        h ^= Vector.ShiftRightLogical(h, 16);
        h *= new Vector<uint>(Draws.M1);
        h ^= Vector.ShiftRightLogical(h, Draws.S);
        h *= new Vector<uint>(Draws.M2);
        return h ^ Vector.ShiftRightLogical(h, 16);
    }

    private readonly struct DropoutBody<TN, TE, TLane, TL>(byte[] x, byte[]? residual, byte[] output, DrawKey key, uint threshold, TLane scale) : Team.IBody
        where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            Span<TLane> tx = stackalloc TLane[Tile];
            Span<TLane> tr = stackalloc TLane[Tile];
            Span<TLane> keep = stackalloc TLane[Tile];
            bool own = typeof(TE) == typeof(TLane);
            Span<TE> nx = own ? default : stackalloc TE[Tile];                   // an integer's values in its natural lane
            Span<TE> nr = own ? default : stackalloc TE[Tile];
            int w = Vector<TLane>.Count;
            Span<TLane> px = stackalloc TLane[w], pr = stackalloc TLane[w], pk = stackalloc TLane[w];
            var s = new Vector<TLane>(scale);
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                if (own)
                {
                    TN.Load(x, at, MemoryMarshal.Cast<TLane, TE>(tx[..len]));
                    if (residual is not null) TN.Load(residual, at, MemoryMarshal.Cast<TLane, TE>(tr[..len]));
                    else tr[..len].Clear();
                }
                else
                {
                    TN.Load(x, at, nx[..len]);
                    Widths.Load<TE, TLane>(nx[..len], tx[..len]);
                    if (residual is not null)
                    {
                        TN.Load(residual, at, nr[..len]);
                        Widths.Load<TE, TLane>(nr[..len], tr[..len]);
                    }
                    else
                    {
                        nr[..len].Clear();
                        tr[..len].Clear();
                    }
                }

                Keeps(key, (uint)at, threshold, keep[..len]);
                int l = 0;
                for (; l + w <= len; l += w)
                {
                    var vx = new Vector<TLane>(tx[l..]);
                    var vr = new Vector<TLane>(tr[l..]);
                    var kept = residual is null ? TL.Mul(vx, s) : TL.Fma(vx, s, vr);
                    TL.Select(new Vector<TLane>(keep[l..]), kept, vr).CopyTo(tx[l..]);
                }

                if (l < len)
                {
                    px.Clear(); pr.Clear(); pk.Clear();
                    tx[l..len].CopyTo(px); tr[l..len].CopyTo(pr); keep[l..len].CopyTo(pk);
                    var vx = new Vector<TLane>(px);
                    var vr = new Vector<TLane>(pr);
                    var kept = residual is null ? TL.Mul(vx, s) : TL.Fma(vx, s, vr);
                    var res = TL.Select(new Vector<TLane>(pk), kept, vr);
                    for (int i = l; i < len; i++) tx[i] = res[i - l];
                }

                if (own) TN.Store(MemoryMarshal.Cast<TLane, TE>(tx[..len]), output, at);
                else
                {
                    // A kept value by the type's conversion, a dropped one the residual as it is stored.
                    Widths.Store<TLane, TE>(tx[..len], nx[..len]);
                    var masks = MemoryMarshal.AsBytes(keep[..len]);
                    int size = Unsafe.SizeOf<TLane>();
                    for (int i = 0; i < len; i++) if (masks[i * size] == 0) nx[i] = nr[i];
                    TN.Store(nx[..len], output, at);
                }
            }
        }
    }
}
