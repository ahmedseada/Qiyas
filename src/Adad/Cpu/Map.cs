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
}

internal static class MapOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> C<T, TL>(double v) where T : unmanaged where TL : struct, ILane<TL, T> => new(TL.From(v));

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
    public struct Div<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Div(x, y); }
    public struct Max<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Max(x, y); }
    public struct Min<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Min(x, y); }
    public struct Axpy<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Fma(a, x, y); }
    public struct MulAdd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Fma(x, y, z); }
    public struct Where<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Equal(x, Vector<T>.Zero), z, y); }
    public struct SquareBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(C<T, TL>(2), x), y); }
    public struct AbsBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(Sign<T, TL>(x), y); }
    public struct ReluBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), y, Vector<T>.Zero); }
    public struct ClampBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, a) & TL.Greater(b, x), y, Vector<T>.Zero); }
    public struct WhereBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, ILane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Equal(x, Vector<T>.Zero), Vector<T>.Zero, y); }

    // float lanes
    private const double GeluK = 0.7978845608028654, GeluC = 0.044715, InvSqrt2 = 0.7071067811865476, InvSqrt2Pi = 0.3989422804014327;

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
    public struct Sigmoid<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Sigmoid(x); }
    public struct Erf<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Erf(x); }
    public struct Gelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => GeluF<T, TL>(x); }
    public struct GeluErf<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, PhiF<T, TL>(x)); }
    public struct Silu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(x, TL.Sigmoid(x)); }
    public struct LeakyRelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), x, TL.Mul(a, x)); }
    public struct Pow<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => PowF<T, TL>(x, a); }
    public struct Dropout<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Equal(y, Vector<T>.Zero), Vector<T>.Zero, TL.Mul(x, a)); }
    public struct GatedSilu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(x, TL.Sigmoid(x)), y); }
    public struct GatedGelu<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(GeluF<T, TL>(x), y); }
    public struct SqrtBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Div(y, TL.Mul(C<T, TL>(2), TL.Sqrt(x))); }
    public struct InvSqrtBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var s = TL.Add(x, a); var r = TL.Div(C<T, TL>(1), TL.Sqrt(s)); return TL.Mul(TL.Mul(C<T, TL>(-0.5), TL.Div(r, s)), y); } }
    public struct ExpBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Exp(x), y); }
    public struct LogBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Div(y, x); }
    public struct SinBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Cos(x), y); }
    public struct CosBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Neg(TL.Mul(TL.Sin(x), y)); }
    public struct TanhBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) { var s = TL.Sigmoid(TL.Mul(C<T, TL>(2), x)); return TL.Mul(TL.Mul(C<T, TL>(4), TL.Mul(s, TL.Sigmoid(TL.Mul(C<T, TL>(-2), x)))), y); } }
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
    }
    public struct LeakyReluBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Select(TL.Greater(x, Vector<T>.Zero), y, TL.Mul(a, y)); }
    public struct PowBwd<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(a, PowF<T, TL>(x, TL.Sub(a, C<T, TL>(1)))), y); }
    public struct GatedSiluBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(SiluGradF<T, TL>(x), y), z); }
    public struct GatedSiluBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(x, TL.Sigmoid(x)), z); }
    public struct GatedGeluBwdX<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(TL.Mul(GeluGradF<T, TL>(x), y), z); }
    public struct GatedGeluBwdY<T, TL> : IMapOp<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T> { public static Vector<T> Apply(Vector<T> x, Vector<T> y, Vector<T> z, Vector<T> a, Vector<T> b) => TL.Mul(GeluF<T, TL>(x), z); }
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

        public readonly bool Known => op is "copy" or "fill" or "neg" or "abs" or "sign" or "square" or "relu" or "add_scalar" or "scale" or "affine" or "clamp"
            or "add" or "sub" or "mul" or "div" or "max" or "min" or "axpy" or "muladd" or "where" or "square_bwd" or "abs_bwd" or "relu_bwd" or "clamp_bwd" or "where_bwd";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            switch (op)
            {
                case "copy": MapRun<TN, TLane, TL, MapOps.Copy<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "fill": MapRun<TN, TLane, TL, MapOps.Fill<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "neg": MapRun<TN, TLane, TL, MapOps.Neg<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "abs": MapRun<TN, TLane, TL, MapOps.Abs<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sign": MapRun<TN, TLane, TL, MapOps.SignOp<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "square": MapRun<TN, TLane, TL, MapOps.Square<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "relu": MapRun<TN, TLane, TL, MapOps.Relu<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "add_scalar": MapRun<TN, TLane, TL, MapOps.AddScalar<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "scale": MapRun<TN, TLane, TL, MapOps.Scale<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "affine": MapRun<TN, TLane, TL, MapOps.Affine<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "clamp": MapRun<TN, TLane, TL, MapOps.Clamp<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "add": MapRun<TN, TLane, TL, MapOps.Add<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sub": MapRun<TN, TLane, TL, MapOps.Sub<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "mul": MapRun<TN, TLane, TL, MapOps.Mul<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "div": MapRun<TN, TLane, TL, MapOps.Div<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "max": MapRun<TN, TLane, TL, MapOps.Max<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "min": MapRun<TN, TLane, TL, MapOps.Min<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "axpy": MapRun<TN, TLane, TL, MapOps.Axpy<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "muladd": MapRun<TN, TLane, TL, MapOps.MulAdd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "where": MapRun<TN, TLane, TL, MapOps.Where<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "square_bwd": MapRun<TN, TLane, TL, MapOps.SquareBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "abs_bwd": MapRun<TN, TLane, TL, MapOps.AbsBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "relu_bwd": MapRun<TN, TLane, TL, MapOps.ReluBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "clamp_bwd": MapRun<TN, TLane, TL, MapOps.ClampBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "where_bwd": MapRun<TN, TLane, TL, MapOps.WhereBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                default: throw new ArgumentException(op);
            }
        }
    }

    public struct FloatMapVisitor(string op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            switch (op)
            {
                case "sqrt": MapRun<TN, TLane, TL, MapOps.Sqrt<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "inv_sqrt": MapRun<TN, TLane, TL, MapOps.InvSqrt<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "exp": MapRun<TN, TLane, TL, MapOps.Exp<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "expm1": MapRun<TN, TLane, TL, MapOps.Expm1<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "log": MapRun<TN, TLane, TL, MapOps.Log<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sin": MapRun<TN, TLane, TL, MapOps.Sin<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "cos": MapRun<TN, TLane, TL, MapOps.Cos<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "tanh": MapRun<TN, TLane, TL, MapOps.Tanh<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sigmoid": MapRun<TN, TLane, TL, MapOps.Sigmoid<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "erf": MapRun<TN, TLane, TL, MapOps.Erf<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gelu": MapRun<TN, TLane, TL, MapOps.Gelu<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gelu_erf": MapRun<TN, TLane, TL, MapOps.GeluErf<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "silu": MapRun<TN, TLane, TL, MapOps.Silu<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "leaky_relu": MapRun<TN, TLane, TL, MapOps.LeakyRelu<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "pow": MapRun<TN, TLane, TL, MapOps.Pow<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "dropout": MapRun<TN, TLane, TL, MapOps.Dropout<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gated_silu": MapRun<TN, TLane, TL, MapOps.GatedSilu<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gated_gelu": MapRun<TN, TLane, TL, MapOps.GatedGelu<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sqrt_bwd": MapRun<TN, TLane, TL, MapOps.SqrtBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "inv_sqrt_bwd": MapRun<TN, TLane, TL, MapOps.InvSqrtBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "exp_bwd": MapRun<TN, TLane, TL, MapOps.ExpBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "log_bwd": MapRun<TN, TLane, TL, MapOps.LogBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sin_bwd": MapRun<TN, TLane, TL, MapOps.SinBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "cos_bwd": MapRun<TN, TLane, TL, MapOps.CosBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "tanh_bwd": MapRun<TN, TLane, TL, MapOps.TanhBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "sigmoid_bwd": MapRun<TN, TLane, TL, MapOps.SigmoidBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "silu_bwd": MapRun<TN, TLane, TL, MapOps.SiluBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gelu_bwd": MapRun<TN, TLane, TL, MapOps.GeluBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gelu_erf_bwd": MapRun<TN, TLane, TL, MapOps.GeluErfBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "leaky_relu_bwd": MapRun<TN, TLane, TL, MapOps.LeakyReluBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "pow_bwd": MapRun<TN, TLane, TL, MapOps.PowBwd<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gated_silu_bwd_x": MapRun<TN, TLane, TL, MapOps.GatedSiluBwdX<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gated_silu_bwd_y": MapRun<TN, TLane, TL, MapOps.GatedSiluBwdY<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gated_gelu_bwd_x": MapRun<TN, TLane, TL, MapOps.GatedGeluBwdX<TLane, TL>>(x, y, z, output, n, a, b); break;
                case "gated_gelu_bwd_y": MapRun<TN, TLane, TL, MapOps.GatedGeluBwdY<TLane, TL>>(x, y, z, output, n, a, b); break;
                default: throw new ArgumentException(op);
            }
        }
    }

    private static void MapRun<TN, TLane, TL, TOp>(byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
        Team.For(n, size, (long)n * 4, new MapBody<TN, TLane, TL, TOp>(x, y, z, output, TL.From(a), TL.From(b)));
    }

    private struct MapBody<TN, TLane, TL, TOp>(byte[]? x, byte[]? y, byte[]? z, byte[] output, TLane a, TLane b) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var va = new Vector<TLane>(a);
            var vb = new Vector<TLane>(b);
            if (TN.Direct)
            {
                var ox = x is null ? default : MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(first, last - first);
                var oy = y is null ? default : MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(first, last - first);
                var oz = z is null ? default : MemoryMarshal.Cast<byte, TLane>(z.AsSpan()).Slice(first, last - first);
                Apply(ox, oy, oz, MemoryMarshal.Cast<byte, TLane>(output.AsSpan()).Slice(first, last - first), va, vb);
                return;
            }

            Span<TLane> tx = stackalloc TLane[Tile];
            Span<TLane> ty = stackalloc TLane[Tile];
            Span<TLane> tz = stackalloc TLane[Tile];
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                if (x is not null) TN.Load(x, at, tx[..len]);
                if (y is not null) TN.Load(y, at, ty[..len]);
                if (z is not null) TN.Load(z, at, tz[..len]);
                Apply(x is null ? default : tx[..len], y is null ? default : ty[..len], z is null ? default : tz[..len], tx[..len], va, vb);
                if (!TN.IsFloat && TN.Bits < 8 * Unsafe.SizeOf<TLane>()) Wrap(tx[..len], TN.Bits, TN.Min < 0);
                TN.Store(tx[..len], output, at);
            }
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
}
