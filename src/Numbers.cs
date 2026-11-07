// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Qiyas: numbers computed in the lane the hardware does best for them. Two parts:
//
//   - a number (IQiyasNumber<TSelf>): a stored type of any width (64 bits down to 2; floats of 64 down to 8; values
//     smaller than a byte packed, the first in the low bits), described as data (name, bits, range), plus, for each
//     compute lane it offers (IQiyasNumber<TSelf, TLane>), how values load into lanes and store back (rounded to
//     nearest, ties to even, held to the type's range), Direct when the stored bytes are the lanes themselves.
//   - a lane (IQiyasLane<TSelf, T>): a hardware type kernels compute in, its vector and scalar operations;
//     IQiyasFloatLane adds exp, max, sqrt and division; IQiyasDotLane is an accumulator lane fed by groups of narrower
//     inputs through the hardware's widening multiply-add (a dot product per lane per instruction).
//
// Kernels are written once over <TN, TLane, TL>. Which lane an operation takes is the number's choice, per operation:
// element-wise in the type's own lane, sums in the cheapest lane that is exact for the bounds measured on the run's
// data (Sum), row operations in a float lane (Float). Nothing here is fixed to a type: every choice is a rule on the
// data and on what the hardware reports. Every load and store runs on vectors; nothing on a hot path allocates,
// boxes or calls through a delegate.
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Qiyas;

// ---------------------------------------------------------------------------------------------- the contract

public interface IQiyasNumber<TSelf> where TSelf : struct, IQiyasNumber<TSelf>
{
    static abstract string Name { get; }

    /// <summary>Bits a value takes (2, 4, 8, 16, 32, 64).</summary>
    static abstract int Bits { get; }

    static abstract bool IsFloat { get; }

    /// <summary>An integer type's smallest and largest value (a float type: its largest finite, negated and not).</summary>
    static abstract double Min { get; }

    static abstract double Max { get; }

    /// <summary>The smallest and largest of <paramref name="count"/> stored values (the type's range when it is not read).</summary>
    static virtual (double Min, double Max) Range(ReadOnlySpan<byte> stored, int count) => (TSelf.Min, TSelf.Max);

    /// <summary>The bytes `values` of the type take as stored (whole bytes; a block format's whole blocks).</summary>
    static virtual long Bytes(long values) => (values * TSelf.Bits + 7) / 8;

    /// <summary>Values that share a storage unit (a byte of packed values, a block): a store of fewer than this many,
    /// or not starting on one, rewrites the unit; two threads must not write one unit.</summary>
    static virtual int Granularity => TSelf.Bits >= 8 ? 1 : 8 / TSelf.Bits;

    /// <summary>Runs <paramref name="visitor"/> with the lane element-wise operations take.</summary>
    static abstract void Elementwise<TV>(ref TV visitor) where TV : struct, ILaneVisitor;

    /// <summary>Runs <paramref name="visitor"/> with the cheapest lane in which sums of products within <paramref name="bounds"/> are exact.</summary>
    static abstract void Sum<TV>(ref TV visitor, in Bounds bounds) where TV : struct, ISumVisitor;
}

/// <summary>A float number: the lane its row operations (softmax, norms) take.</summary>
public interface IQiyasFloat<TSelf> : IQiyasNumber<TSelf> where TSelf : struct, IQiyasFloat<TSelf>
{
    static abstract void Float<TV>(ref TV visitor) where TV : struct, IFloatLaneVisitor;
}

public interface IQiyasNumber<TSelf, TLane> : IQiyasNumber<TSelf>
    where TSelf : struct, IQiyasNumber<TSelf, TLane> where TLane : unmanaged
{
    /// <summary>The stored bytes are TLane values: kernels read and write them in place.</summary>
    static virtual bool Direct => false;

    /// <summary>Values first .. first + lanes.Length into lanes (any first).</summary>
    static abstract void Load(ReadOnlySpan<byte> stored, int first, Span<TLane> lanes);

    /// <summary>Lanes into values first ..: rounded to nearest (ties to even), held to the type's range.</summary>
    static abstract void Store(ReadOnlySpan<TLane> lanes, Span<byte> stored, int first);
}

/// <summary>What a sum of products sees: the measured range of each operand and the number of terms.</summary>
public readonly struct Bounds(double aMin, double aMax, double bMin, double bMax, long terms)
{
    public double AMin => aMin;
    public double AMax => aMax;
    public double BMin => bMin;
    public double BMax => bMax;
    public long Terms => terms;

    /// <summary>The largest magnitude any sum can reach: terms · max|a| · max|b|.</summary>
    public double Bound => terms * Math.Max(Math.Abs(aMin), Math.Abs(aMax)) * Math.Max(Math.Abs(bMin), Math.Abs(bMax));

    /// <summary>Both operands lie within [low, high].</summary>
    public bool Within(double low, double high) => aMin >= low && aMax <= high && bMin >= low && bMax <= high;
}

public interface IQiyasLane<TSelf, T> where TSelf : struct, IQiyasLane<TSelf, T> where T : unmanaged
{
    static abstract string Name { get; }

    static abstract T Zero { get; }

    static abstract T From(double value);

    static abstract Vector<T> Add(Vector<T> a, Vector<T> b);

    /// <summary>a·b + c (a dot lane: c + the dot product of the groups a and b hold).</summary>
    static abstract Vector<T> Fma(Vector<T> a, Vector<T> b, Vector<T> c);

    static abstract T Add(T a, T b);

    static abstract T Fma(T a, T b, T c);

    static abstract T Mul(T a, T b);

    // The element-wise operations every lane has, on vectors (the defaults are Vector<T>'s own, exact for every lane).
    static virtual Vector<T> Sub(Vector<T> a, Vector<T> b) => a - b;

    static virtual Vector<T> Mul(Vector<T> a, Vector<T> b) => a * b;

    static virtual Vector<T> Div(Vector<T> a, Vector<T> b) => a / b;

    static virtual Vector<T> Max(Vector<T> a, Vector<T> b) => Vector.Max(a, b);

    static virtual Vector<T> Min(Vector<T> a, Vector<T> b) => Vector.Min(a, b);

    static virtual Vector<T> Neg(Vector<T> a) => Vector<T>.Zero - a;

    static virtual Vector<T> Abs(Vector<T> a) => Vector.Abs(a);

    /// <summary>All ones where a &gt; b.</summary>
    static virtual Vector<T> Greater(Vector<T> a, Vector<T> b) => Vector.GreaterThan(a, b);

    static virtual Vector<T> Equal(Vector<T> a, Vector<T> b) => Vector.Equals(a, b);

    static virtual Vector<T> Select(Vector<T> mask, Vector<T> a, Vector<T> b) => Vector.ConditionalSelect(mask, a, b);
}

public interface IQiyasFloatLane<TSelf, T> : IQiyasLane<TSelf, T> where TSelf : struct, IQiyasFloatLane<TSelf, T> where T : unmanaged
{
    static abstract T NegativeInfinity { get; }

    static abstract Vector<T> Exp(Vector<T> x);

    static abstract Vector<T> Expm1(Vector<T> x);

    static abstract Vector<T> Log(Vector<T> x);

    static abstract Vector<T> Sin(Vector<T> x);

    static abstract Vector<T> Cos(Vector<T> x);

    /// <summary>Sin and Cos of the same values; a lane may share their reduction (each must equal Sin and Cos).</summary>
    static virtual (Vector<T> Sin, Vector<T> Cos) SinCos(Vector<T> x) => (TSelf.Sin(x), TSelf.Cos(x));

    static abstract Vector<T> Tanh(Vector<T> x);

    static abstract Vector<T> Sigmoid(Vector<T> x);

    static abstract Vector<T> Erf(Vector<T> x);

    static abstract Vector<T> Erfc(Vector<T> x);

    static abstract Vector<T> Sqrt(Vector<T> x);

    static abstract Vector<T> Round(Vector<T> x);

    static abstract Vector<T> Floor(Vector<T> x);

    static abstract Vector<T> IsNaN(Vector<T> x);

    static abstract T Exp(T x);

    static abstract T Max(T a, T b);

    static abstract T Sub(T a, T b);

    static abstract T Sqrt(T x);

    static abstract T Div(T a, T b);
}

/// <summary>An accumulator lane TAcc fed by sizeof(TAcc)/sizeof(TIn) inputs packed in each lane: Fma(a, b, c) is c plus
/// the dot product of the groups, one instruction per vector where the hardware has it.</summary>
public interface IQiyasDotLane<TSelf, TIn, TAcc> : IQiyasLane<TSelf, TAcc>
    where TSelf : struct, IQiyasDotLane<TSelf, TIn, TAcc> where TIn : unmanaged where TAcc : unmanaged
{
    /// <summary>The processor this runs on has the instruction at the vector width in use.</summary>
    static abstract bool IsSupported { get; }
}

public interface ILaneVisitor
{
    void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>;
}

public interface ISumVisitor : ILaneVisitor
{
    void RunDot<TN, TIn, TAcc, TD>()
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IQiyasDotLane<TD, TIn, TAcc>;
}

public interface IFloatLaneVisitor
{
    void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>;
}

/// <summary>Members of a number that take two lane instantiations apart.</summary>
internal static class Lanes
{
    public static bool Direct<TN, T>() where TN : struct, IQiyasNumber<TN, T> where T : unmanaged => TN.Direct;
}

// ---------------------------------------------------------------------------------------------- lanes

public readonly struct F32 : IQiyasFloatLane<F32, float>
{
    public static string Name => "f32";
    public static float Zero => 0f;
    public static float NegativeInfinity => float.NegativeInfinity;
    public static float From(double value) => (float)value;
    public static Vector<float> Add(Vector<float> a, Vector<float> b) => a + b;
    public static Vector<float> Fma(Vector<float> a, Vector<float> b, Vector<float> c) => Vector.FusedMultiplyAdd(a, b, c);
    public static float Add(float a, float b) => a + b;
    public static float Mul(float a, float b) => a * b;
    public static float Fma(float a, float b, float c) => MathF.FusedMultiplyAdd(a, b, c);
    public static Vector<float> Exp(Vector<float> x) => Exps.Exp(x);
    public static Vector<float> Expm1(Vector<float> x) => Funcs.Expm1(x);
    public static Vector<float> Log(Vector<float> x) => Funcs.Log(x);
    public static Vector<float> Sin(Vector<float> x) => Funcs.Sin(x);
    public static Vector<float> Cos(Vector<float> x) => Funcs.Cos(x);
    public static (Vector<float> Sin, Vector<float> Cos) SinCos(Vector<float> x) => Funcs.SinCos(x);
    public static Vector<float> Tanh(Vector<float> x) => Funcs.Tanh(x);
    public static Vector<float> Sigmoid(Vector<float> x) => Funcs.Sigmoid(x);
    public static Vector<float> Erf(Vector<float> x) => Funcs.Erf(x);
    public static Vector<float> Erfc(Vector<float> x) => Funcs.Erfc(x);
    public static Vector<float> Sqrt(Vector<float> x) => Vector.SquareRoot(x);
    public static Vector<float> Round(Vector<float> x) => Vector.Round(x);
    public static Vector<float> Floor(Vector<float> x) => Vector.Floor(x);
    public static Vector<float> IsNaN(Vector<float> x) => Vector.AsVectorSingle(~Vector.AsVectorInt32(Vector.Equals(x, x)));
    public static float Exp(float x) => MathF.Exp(x);
    public static float Max(float a, float b) => MathF.Max(a, b);
    public static float Sub(float a, float b) => a - b;
    public static float Sqrt(float x) => MathF.Sqrt(x);
    public static float Div(float a, float b) => a / b;
}

public readonly struct F64 : IQiyasFloatLane<F64, double>
{
    public static string Name => "f64";
    public static double Zero => 0d;
    public static double NegativeInfinity => double.NegativeInfinity;
    public static double From(double value) => value;
    public static Vector<double> Add(Vector<double> a, Vector<double> b) => a + b;
    public static Vector<double> Fma(Vector<double> a, Vector<double> b, Vector<double> c) => Vector.FusedMultiplyAdd(a, b, c);
    public static double Add(double a, double b) => a + b;
    public static double Mul(double a, double b) => a * b;
    public static double Fma(double a, double b, double c) => Math.FusedMultiplyAdd(a, b, c);
    public static Vector<double> Exp(Vector<double> x) => Exps.Exp(x);
    public static Vector<double> Expm1(Vector<double> x) => Funcs.Expm1(x);
    public static Vector<double> Log(Vector<double> x) => Funcs.Log(x);
    public static Vector<double> Sin(Vector<double> x) => Funcs.Sin(x);
    public static Vector<double> Cos(Vector<double> x) => Funcs.Cos(x);
    public static (Vector<double> Sin, Vector<double> Cos) SinCos(Vector<double> x) => Funcs.SinCos(x);
    public static Vector<double> Tanh(Vector<double> x) => Funcs.Tanh(x);
    public static Vector<double> Sigmoid(Vector<double> x) => Funcs.Sigmoid(x);
    public static Vector<double> Erf(Vector<double> x) => Funcs.Erf(x);
    public static Vector<double> Erfc(Vector<double> x) => Funcs.Erfc(x);
    public static Vector<double> Sqrt(Vector<double> x) => Vector.SquareRoot(x);
    public static Vector<double> Round(Vector<double> x) => Vector.Round(x);
    public static Vector<double> Floor(Vector<double> x) => Vector.Floor(x);
    public static Vector<double> IsNaN(Vector<double> x) => Vector.AsVectorDouble(~Vector.AsVectorInt64(Vector.Equals(x, x)));
    public static double Exp(double x) => Math.Exp(x);
    public static double Max(double a, double b) => Math.Max(a, b);
    public static double Sub(double a, double b) => a - b;
    public static double Sqrt(double x) => Math.Sqrt(x);
    public static double Div(double a, double b) => a / b;
}

public readonly struct I64 : IQiyasLane<I64, long>
{
    public static string Name => "s64";
    public static long Zero => 0;
    public static long From(double value) => (long)value;
    public static Vector<long> Add(Vector<long> a, Vector<long> b) => a + b;
    public static Vector<long> Fma(Vector<long> a, Vector<long> b, Vector<long> c) => a * b + c;
    public static long Add(long a, long b) => a + b;
    public static long Mul(long a, long b) => a * b;
    public static long Fma(long a, long b, long c) => a * b + c;
}

public readonly struct U64 : IQiyasLane<U64, ulong>
{
    public static string Name => "u64";
    public static ulong Zero => 0;
    public static ulong From(double value) => (ulong)value;
    public static Vector<ulong> Add(Vector<ulong> a, Vector<ulong> b) => a + b;
    public static Vector<ulong> Fma(Vector<ulong> a, Vector<ulong> b, Vector<ulong> c) => a * b + c;
    public static ulong Add(ulong a, ulong b) => a + b;
    public static ulong Mul(ulong a, ulong b) => a * b;
    public static ulong Fma(ulong a, ulong b, ulong c) => a * b + c;
}

public readonly struct I32 : IQiyasLane<I32, int>
{
    public static string Name => "s32";
    public static int Zero => 0;
    public static int From(double value) => (int)value;
    public static Vector<int> Add(Vector<int> a, Vector<int> b) => a + b;
    public static Vector<int> Fma(Vector<int> a, Vector<int> b, Vector<int> c) => a * b + c;
    public static int Add(int a, int b) => a + b;
    public static int Mul(int a, int b) => a * b;
    public static int Fma(int a, int b, int c) => a * b + c;
}

public readonly struct U32 : IQiyasLane<U32, uint>
{
    public static string Name => "u32";
    public static uint Zero => 0;
    public static uint From(double value) => (uint)value;
    public static Vector<uint> Add(Vector<uint> a, Vector<uint> b) => a + b;
    public static Vector<uint> Fma(Vector<uint> a, Vector<uint> b, Vector<uint> c) => a * b + c;
    public static uint Add(uint a, uint b) => a + b;
    public static uint Mul(uint a, uint b) => a * b;
    public static uint Fma(uint a, uint b, uint c) => a * b + c;
}

public readonly struct I16 : IQiyasLane<I16, short>
{
    public static string Name => "s16";
    public static short Zero => 0;
    public static short From(double value) => (short)value;
    public static Vector<short> Add(Vector<short> a, Vector<short> b) => a + b;
    public static Vector<short> Fma(Vector<short> a, Vector<short> b, Vector<short> c) => a * b + c;
    public static short Add(short a, short b) => (short)(a + b);
    public static short Mul(short a, short b) => (short)(a * b);
    public static short Fma(short a, short b, short c) => (short)(a * b + c);
}

public readonly struct U16 : IQiyasLane<U16, ushort>
{
    public static string Name => "u16";
    public static ushort Zero => 0;
    public static ushort From(double value) => (ushort)value;
    public static Vector<ushort> Add(Vector<ushort> a, Vector<ushort> b) => a + b;
    public static Vector<ushort> Fma(Vector<ushort> a, Vector<ushort> b, Vector<ushort> c) => a * b + c;
    public static ushort Add(ushort a, ushort b) => (ushort)(a + b);
    public static ushort Mul(ushort a, ushort b) => (ushort)(a * b);
    public static ushort Fma(ushort a, ushort b, ushort c) => (ushort)(a * b + c);
}

public readonly struct I8 : IQiyasLane<I8, sbyte>
{
    public static string Name => "s8";
    public static sbyte Zero => 0;
    public static sbyte From(double value) => (sbyte)value;
    public static Vector<sbyte> Add(Vector<sbyte> a, Vector<sbyte> b) => a + b;
    public static Vector<sbyte> Fma(Vector<sbyte> a, Vector<sbyte> b, Vector<sbyte> c) => a * b + c;
    public static sbyte Add(sbyte a, sbyte b) => (sbyte)(a + b);
    public static sbyte Mul(sbyte a, sbyte b) => (sbyte)(a * b);
    public static sbyte Fma(sbyte a, sbyte b, sbyte c) => (sbyte)(a * b + c);
}

public readonly struct U8 : IQiyasLane<U8, byte>
{
    public static string Name => "u8";
    public static byte Zero => 0;
    public static byte From(double value) => (byte)value;
    public static Vector<byte> Add(Vector<byte> a, Vector<byte> b) => a + b;
    public static Vector<byte> Fma(Vector<byte> a, Vector<byte> b, Vector<byte> c) => a * b + c;
    public static byte Add(byte a, byte b) => (byte)(a + b);
    public static byte Mul(byte a, byte b) => (byte)(a * b);
    public static byte Fma(byte a, byte b, byte c) => (byte)(a * b + c);
}

// ---------------------------------------------------------------------------------------------- dot lanes
// Four 8-bit (two 16-bit) inputs packed in each 32-bit lane; one instruction multiplies the groups pairwise and adds
// the products into the lane. Which instruction is a run-time fact of the processor (x86 VNNI at the vector width in
// use, Arm's dot product); a lane that is not supported is never chosen.

internal static class Dots
{
    private static bool Width => Vector<byte>.Count == 32 || Vector<byte>.Count == 16;
    public static bool X86Vnni8 => Width && AvxVnniInt8.IsSupported;
    public static bool X86Vnni16 => Width && AvxVnni.IsSupported;
    public static bool X86Vnni16U => Width && AvxVnniInt16.IsSupported;
    public static bool ArmDot => Vector<byte>.Count == 16 && Dp.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Dot8(int a, int b, int c) =>
        c + (sbyte)a * (sbyte)b + (sbyte)(a >> 8) * (sbyte)(b >> 8) + (sbyte)(a >> 16) * (sbyte)(b >> 16) + (sbyte)(a >> 24) * (sbyte)(b >> 24);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Dot8(uint a, uint b, uint c) =>
        c + (a & 0xFF) * (b & 0xFF) + ((a >> 8) & 0xFF) * ((b >> 8) & 0xFF) + ((a >> 16) & 0xFF) * ((b >> 16) & 0xFF) + (a >> 24) * (b >> 24);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Dot16(int a, int b, int c) => c + (short)a * (short)b + (short)(a >> 16) * (short)(b >> 16);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Dot16(uint a, uint b, uint c) => c + (a & 0xFFFF) * (b & 0xFFFF) + (a >> 16) * (b >> 16);
}

/// <summary>Signed bytes, four a lane, into s32.</summary>
public readonly struct S8Dot : IQiyasDotLane<S8Dot, sbyte, int>
{
    public static string Name => "s8dp";
    public static bool IsSupported => Dots.X86Vnni8 || Dots.ArmDot;
    public static int Zero => 0;
    public static int From(double value) => (int)value;
    public static Vector<int> Add(Vector<int> a, Vector<int> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<int> Fma(Vector<int> a, Vector<int> b, Vector<int> c)
    {
        if (Vector<byte>.Count == 32) return AvxVnniInt8.MultiplyWideningAndAdd(c.AsVector256(), a.AsVector256().AsSByte(), b.AsVector256().AsSByte()).AsVector();
        if (AvxVnniInt8.IsSupported) return AvxVnniInt8.MultiplyWideningAndAdd(c.AsVector128(), a.AsVector128().AsSByte(), b.AsVector128().AsSByte()).AsVector();
        return Dp.DotProduct(c.AsVector128(), a.AsVector128().AsSByte(), b.AsVector128().AsSByte()).AsVector();
    }

    public static int Add(int a, int b) => a + b;
    public static int Mul(int a, int b) => Dots.Dot8(a, b, 0);
    public static int Fma(int a, int b, int c) => Dots.Dot8(a, b, c);
}

/// <summary>Unsigned bytes, four a lane, into u32.</summary>
public readonly struct U8Dot : IQiyasDotLane<U8Dot, byte, uint>
{
    public static string Name => "u8dp";
    public static bool IsSupported => Dots.X86Vnni8 || Dots.ArmDot;
    public static uint Zero => 0;
    public static uint From(double value) => (uint)value;
    public static Vector<uint> Add(Vector<uint> a, Vector<uint> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<uint> Fma(Vector<uint> a, Vector<uint> b, Vector<uint> c)
    {
        if (Vector<byte>.Count == 32) return AvxVnniInt8.MultiplyWideningAndAdd(c.AsVector256(), a.AsVector256().AsByte(), b.AsVector256().AsByte()).AsVector();
        if (AvxVnniInt8.IsSupported) return AvxVnniInt8.MultiplyWideningAndAdd(c.AsVector128(), a.AsVector128().AsByte(), b.AsVector128().AsByte()).AsVector();
        return Dp.DotProduct(c.AsVector128(), a.AsVector128().AsByte(), b.AsVector128().AsByte()).AsVector();
    }

    public static uint Add(uint a, uint b) => a + b;
    public static uint Mul(uint a, uint b) => Dots.Dot8(a, b, 0u);
    public static uint Fma(uint a, uint b, uint c) => Dots.Dot8(a, b, c);
}

/// <summary>Signed 16-bit values, two a lane, into s32.</summary>
public readonly struct S16Dot : IQiyasDotLane<S16Dot, short, int>
{
    public static string Name => "s16dp";
    public static bool IsSupported => Dots.X86Vnni16;
    public static int Zero => 0;
    public static int From(double value) => (int)value;
    public static Vector<int> Add(Vector<int> a, Vector<int> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<int> Fma(Vector<int> a, Vector<int> b, Vector<int> c)
    {
        if (Vector<byte>.Count == 32) return AvxVnni.MultiplyWideningAndAdd(c.AsVector256(), a.AsVector256().AsInt16(), b.AsVector256().AsInt16()).AsVector();
        return AvxVnni.MultiplyWideningAndAdd(c.AsVector128(), a.AsVector128().AsInt16(), b.AsVector128().AsInt16()).AsVector();
    }

    public static int Add(int a, int b) => a + b;
    public static int Mul(int a, int b) => Dots.Dot16(a, b, 0);
    public static int Fma(int a, int b, int c) => Dots.Dot16(a, b, c);
}

/// <summary>Unsigned 16-bit values, two a lane, into u32.</summary>
public readonly struct U16Dot : IQiyasDotLane<U16Dot, ushort, uint>
{
    public static string Name => "u16dp";
    public static bool IsSupported => Dots.X86Vnni16U;
    public static uint Zero => 0;
    public static uint From(double value) => (uint)value;
    public static Vector<uint> Add(Vector<uint> a, Vector<uint> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<uint> Fma(Vector<uint> a, Vector<uint> b, Vector<uint> c)
    {
        if (Vector<byte>.Count == 32) return AvxVnniInt16.MultiplyWideningAndAdd(c.AsVector256(), a.AsVector256().AsUInt16(), b.AsVector256().AsUInt16()).AsVector();
        return AvxVnniInt16.MultiplyWideningAndAdd(c.AsVector128(), a.AsVector128().AsUInt16(), b.AsVector128().AsUInt16()).AsVector();
    }

    public static uint Add(uint a, uint b) => a + b;
    public static uint Mul(uint a, uint b) => Dots.Dot16(a, b, 0u);
    public static uint Fma(uint a, uint b, uint c) => Dots.Dot16(a, b, c);
}

// ---------------------------------------------------------------------------------------------- exp on vectors
// x = n·ln2 + r (ln2 split in two for an exact reduction, |r| <= ln2/2), e^r by its Taylor series through fused
// multiply-adds (degree 7 for f32, 13 for f64: past each type's last bit), times 2^n built in the exponent bits in
// two halves (so the largest finite results stay finite and the smallest subnormal ones exact: the first scaling is
// exact, the second rounds once). Below the smallest subnormal 0, above the largest finite +inf, NaN kept.

internal static class Exps
{
    public static Vector<float> Exp(Vector<float> x)
    {
        var c = Vector.Min(Vector.Max(x, new Vector<float>(-103.9721f)), new Vector<float>(88.72283f));
        var n = Vector.Round(c * new Vector<float>(1.44269504f));
        var r = Vector.FusedMultiplyAdd(n, new Vector<float>(-0.693359375f), c);
        r = Vector.FusedMultiplyAdd(n, new Vector<float>(2.12194440e-4f), r);
        var p = new Vector<float>(1f / 5040);
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 720));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 120));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 24));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 6));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(0.5f));
        p = Vector.FusedMultiplyAdd(p, r, Vector<float>.One);
        p = Vector.FusedMultiplyAdd(p, r, Vector<float>.One);
        var ni = Vector.ConvertToInt32(n);
        var half = ni >> 1;
        var s1 = Vector.AsVectorSingle((half + new Vector<int>(127)) << 23);
        var s2 = Vector.AsVectorSingle((ni - half + new Vector<int>(127)) << 23);
        var y = p * s1 * s2;
        y = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<float>(-103.9721f)), Vector<float>.Zero, y);
        y = Vector.ConditionalSelect(Vector.GreaterThan(x, new Vector<float>(88.72283f)), new Vector<float>(float.PositiveInfinity), y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }

    public static Vector<double> Exp(Vector<double> x)
    {
        var c = Vector.Min(Vector.Max(x, new Vector<double>(-745.1332)), new Vector<double>(709.78));
        var n = Vector.Round(c * new Vector<double>(1.4426950408889634));
        var r = Vector.FusedMultiplyAdd(n, new Vector<double>(-6.93147180369123816490e-01), c);
        r = Vector.FusedMultiplyAdd(n, new Vector<double>(-1.90821492927058770002e-10), r);
        var p = new Vector<double>(1.0 / 6227020800);
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 479001600));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 39916800));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 3628800));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 362880));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 40320));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 5040));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 720));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 120));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 24));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(1.0 / 6));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<double>(0.5));
        p = Vector.FusedMultiplyAdd(p, r, Vector<double>.One);
        p = Vector.FusedMultiplyAdd(p, r, Vector<double>.One);
        // n as an integer through the 1.5·2^52 shift (no 64-bit float-to-integer instruction needed).
        var magic = new Vector<double>(6755399441055744.0);
        var ni = Vector.AsVectorInt64(n + magic) - Vector.AsVectorInt64(magic);
        var half = ni >> 1;
        var s1 = Vector.AsVectorDouble((half + new Vector<long>(1023)) << 52);
        var s2 = Vector.AsVectorDouble((ni - half + new Vector<long>(1023)) << 52);
        var y = p * s1 * s2;
        y = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<double>(-745.1332)), Vector<double>.Zero, y);
        y = Vector.ConditionalSelect(Vector.GreaterThan(x, new Vector<double>(709.78)), new Vector<double>(double.PositiveInfinity), y);
        return Vector.ConditionalSelect(Vector.Equals(x, x), y, x);
    }
}
