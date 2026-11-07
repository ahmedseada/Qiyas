// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The numbers: every per-value type (bool aside: not arithmetic). Each is one struct: its description and the lanes
// it offers. Sums take the cheapest lane that is exact for the bounds measured on the run's data, in this order:
//
//   a dot lane (the processor's widening multiply-add on 8- or 16-bit inputs) when the type's values are that narrow,
//   or the measured range says they fit, and the bound fits the 32-bit accumulator;
//   then f32 (exact to 2^24), s32/u32 (to 2^31 / 2^32), f64 (to 2^53), s64/u64.
//
// Element-wise operations take the type's own lane (in place where stored bytes are lanes); floats' row operations
// their f32 or f64 lane: f32 is the narrowest float lane the hardware computes in, f64 the widest.
using System.Runtime.InteropServices;

namespace Adad;

internal static class Exact
{
    public const double Float32 = 1L << 24;
    public const double Float64 = 1L << 53;
}

// The loads and stores every type shares: its stored values as T, into lanes TL and back.
internal static class Stored
{
    public static void Load<T, TL>(ReadOnlySpan<byte> s, int first, Span<TL> l) where T : unmanaged where TL : unmanaged =>
        Widths.Load(Cvt.As<T>(s).Slice(first, l.Length), l);

    public static void Store<TL, T>(ReadOnlySpan<TL> l, Span<byte> s, int first) where T : unmanaged where TL : unmanaged =>
        Widths.Store(l, Cvt.As<T>(s).Slice(first, l.Length));

    public static (double Min, double Max) Range<T>(ReadOnlySpan<byte> s, int count) where T : unmanaged
    {
        var (lo, hi) = Cvt.RangeSigned(Cvt.As<T>(s)[..count]);
        return (lo, hi);
    }
}

// ---------------------------------------------------------------------------------------------- floats

public readonly struct Float64 : IFloatAdad<Float64>, IAdad<Float64, double>, IAdad<Float64, float>
{
    public static string Name => "float64";
    public static int Bits => 64;
    public static bool IsFloat => true;
    public static double Min => double.MinValue;
    public static double Max => double.MaxValue;
    static bool IAdad<Float64, double>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<double, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, double>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<double, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, double>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Float64, double, F64>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => v.Run<Float64, double, F64>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Float64, double, F64>();
}

public readonly struct Float32 : IFloatAdad<Float32>, IAdad<Float32, float>, IAdad<Float32, double>
{
    public static string Name => "float32";
    public static int Bits => 32;
    public static bool IsFloat => true;
    public static double Min => float.MinValue;
    public static double Max => float.MaxValue;
    static bool IAdad<Float32, float>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<float, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, float>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<float, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, float>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Float32, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => v.Run<Float32, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Float32, float, F32>();
}

public readonly struct Float16 : IFloatAdad<Float16>, IAdad<Float16, float>
{
    public static string Name => "float16";
    public static int Bits => 16;
    public static bool IsFloat => true;
    public static double Min => -65504;
    public static double Max => 65504;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Cvt.FromHalf(Cvt.As<ushort>(s).Slice(first, l.Length), l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Cvt.ToHalf(l, Cvt.As<ushort>(s).Slice(first, l.Length));
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Float16, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => v.Run<Float16, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Float16, float, F32>();
}

public readonly struct BFloat16 : IFloatAdad<BFloat16>, IAdad<BFloat16, float>
{
    public static string Name => "bfloat16";
    public static int Bits => 16;
    public static bool IsFloat => true;
    public static double Min => -3.3895313892515355e38;
    public static double Max => 3.3895313892515355e38;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Cvt.FromBFloat16(Cvt.As<ushort>(s).Slice(first, l.Length), l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Cvt.ToBFloat16(l, Cvt.As<ushort>(s).Slice(first, l.Length));
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<BFloat16, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => v.Run<BFloat16, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<BFloat16, float, F32>();
}

public readonly struct Float8E4M3 : IFloatAdad<Float8E4M3>, IAdad<Float8E4M3, float>
{
    public static string Name => "float8_e4m3";
    public static int Bits => 8;
    public static bool IsFloat => true;
    public static double Min => -448;
    public static double Max => 448;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Cvt.E4M3ToFloat(s.Slice(first, l.Length), l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Cvt.E4M3FromFloat(l, s.Slice(first, l.Length));
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Float8E4M3, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => v.Run<Float8E4M3, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Float8E4M3, float, F32>();
}

public readonly struct Float8E5M2 : IFloatAdad<Float8E5M2>, IAdad<Float8E5M2, float>
{
    public static string Name => "float8_e5m2";
    public static int Bits => 8;
    public static bool IsFloat => true;
    public static double Min => -57344;
    public static double Max => 57344;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Cvt.E5M2ToFloat(s.Slice(first, l.Length), l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Cvt.E5M2FromFloat(l, s.Slice(first, l.Length));
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Float8E5M2, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => v.Run<Float8E5M2, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Float8E5M2, float, F32>();
}

// ---------------------------------------------------------------------------------------------- signed integers

public readonly struct Int64 : IAdad<Int64, long>, IAdad<Int64, float>, IAdad<Int64, double>, IAdad<Int64, int>,
    IAdad<Int64, sbyte>, IAdad<Int64, short>
{
    public static string Name => "int64";
    public static int Bits => 64;
    public static bool IsFloat => false;
    public static double Min => long.MinValue;
    public static double Max => long.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<long>(s, count);
    static bool IAdad<Int64, long>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<long> l) => Stored.Load<long, long>(s, first, l);
    public static void Store(ReadOnlySpan<long> l, Span<byte> s, int first) => Stored.Store<long, long>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<long, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, long>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<long, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, long>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<int> l) => Stored.Load<long, int>(s, first, l);
    public static void Store(ReadOnlySpan<int> l, Span<byte> s, int first) => Stored.Store<int, long>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<sbyte> l) => Stored.Load<long, sbyte>(s, first, l);
    public static void Store(ReadOnlySpan<sbyte> l, Span<byte> s, int first) => Stored.Store<sbyte, long>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<short> l) => Stored.Load<long, short>(s, first, l);
    public static void Store(ReadOnlySpan<short> l, Span<byte> s, int first) => Stored.Store<short, long>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Int64, long, I64>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (S8Dot.IsSupported && b.Within(sbyte.MinValue, sbyte.MaxValue) && b.Bound <= int.MaxValue) v.RunDot<Int64, sbyte, int, S8Dot>();
        else if (S16Dot.IsSupported && b.Within(short.MinValue, short.MaxValue) && b.Bound <= int.MaxValue) v.RunDot<Int64, short, int, S16Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<Int64, float, F32>();
        else if (b.Bound <= int.MaxValue) v.Run<Int64, int, I32>();
        else if (b.Bound <= Exact.Float64) v.Run<Int64, double, F64>();
        else v.Run<Int64, long, I64>();
    }
}

public readonly struct Int32 : IAdad<Int32, int>, IAdad<Int32, float>, IAdad<Int32, double>, IAdad<Int32, long>,
    IAdad<Int32, sbyte>, IAdad<Int32, short>
{
    public static string Name => "int32";
    public static int Bits => 32;
    public static bool IsFloat => false;
    public static double Min => int.MinValue;
    public static double Max => int.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<int>(s, count);
    static bool IAdad<Int32, int>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<int> l) => Stored.Load<int, int>(s, first, l);
    public static void Store(ReadOnlySpan<int> l, Span<byte> s, int first) => Stored.Store<int, int>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<int, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, int>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<int, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, int>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<long> l) => Stored.Load<int, long>(s, first, l);
    public static void Store(ReadOnlySpan<long> l, Span<byte> s, int first) => Stored.Store<long, int>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<sbyte> l) => Stored.Load<int, sbyte>(s, first, l);
    public static void Store(ReadOnlySpan<sbyte> l, Span<byte> s, int first) => Stored.Store<sbyte, int>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<short> l) => Stored.Load<int, short>(s, first, l);
    public static void Store(ReadOnlySpan<short> l, Span<byte> s, int first) => Stored.Store<short, int>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Int32, int, I32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (S8Dot.IsSupported && b.Within(sbyte.MinValue, sbyte.MaxValue) && b.Bound <= int.MaxValue) v.RunDot<Int32, sbyte, int, S8Dot>();
        else if (S16Dot.IsSupported && b.Within(short.MinValue, short.MaxValue) && b.Bound <= int.MaxValue) v.RunDot<Int32, short, int, S16Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<Int32, float, F32>();
        else if (b.Bound <= int.MaxValue) v.Run<Int32, int, I32>();
        else if (b.Bound <= Exact.Float64) v.Run<Int32, double, F64>();
        else v.Run<Int32, long, I64>();
    }
}

public readonly struct Int16 : IAdad<Int16, short>, IAdad<Int16, float>, IAdad<Int16, double>, IAdad<Int16, int>,
    IAdad<Int16, long>, IAdad<Int16, sbyte>
{
    public static string Name => "int16";
    public static int Bits => 16;
    public static bool IsFloat => false;
    public static double Min => short.MinValue;
    public static double Max => short.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<short>(s, count);
    static bool IAdad<Int16, short>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<short> l) => Stored.Load<short, short>(s, first, l);
    public static void Store(ReadOnlySpan<short> l, Span<byte> s, int first) => Stored.Store<short, short>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<short, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, short>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<short, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, short>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<int> l) => Stored.Load<short, int>(s, first, l);
    public static void Store(ReadOnlySpan<int> l, Span<byte> s, int first) => Stored.Store<int, short>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<long> l) => Stored.Load<short, long>(s, first, l);
    public static void Store(ReadOnlySpan<long> l, Span<byte> s, int first) => Stored.Store<long, short>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<sbyte> l) => Stored.Load<short, sbyte>(s, first, l);
    public static void Store(ReadOnlySpan<sbyte> l, Span<byte> s, int first) => Stored.Store<sbyte, short>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Int16, short, I16>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (S8Dot.IsSupported && b.Within(sbyte.MinValue, sbyte.MaxValue) && b.Bound <= int.MaxValue) v.RunDot<Int16, sbyte, int, S8Dot>();
        else if (S16Dot.IsSupported && b.Bound <= int.MaxValue) v.RunDot<Int16, short, int, S16Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<Int16, float, F32>();
        else if (b.Bound <= int.MaxValue) v.Run<Int16, int, I32>();
        else if (b.Bound <= Exact.Float64) v.Run<Int16, double, F64>();
        else v.Run<Int16, long, I64>();
    }
}

public readonly struct Int8 : IAdad<Int8, sbyte>, IAdad<Int8, float>, IAdad<Int8, double>, IAdad<Int8, int>
{
    public static string Name => "int8";
    public static int Bits => 8;
    public static bool IsFloat => false;
    public static double Min => sbyte.MinValue;
    public static double Max => sbyte.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<sbyte>(s, count);
    static bool IAdad<Int8, sbyte>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<sbyte> l) => Stored.Load<sbyte, sbyte>(s, first, l);
    public static void Store(ReadOnlySpan<sbyte> l, Span<byte> s, int first) => Stored.Store<sbyte, sbyte>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<sbyte, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, sbyte>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<sbyte, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, sbyte>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<int> l) => Stored.Load<sbyte, int>(s, first, l);
    public static void Store(ReadOnlySpan<int> l, Span<byte> s, int first) => Stored.Store<int, sbyte>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Int8, sbyte, I8>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (S8Dot.IsSupported && b.Bound <= int.MaxValue) v.RunDot<Int8, sbyte, int, S8Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<Int8, float, F32>();
        else if (b.Bound <= int.MaxValue) v.Run<Int8, int, I32>();
        else v.Run<Int8, double, F64>();
    }
}

// The packed integers: values smaller than a byte, unpacked to 8-bit lanes in chunks on the stack (a chunk a call,
// nothing allocated), and to wider lanes through those.
internal static class Packed
{
    public const int Chunk = 2048;

    /// <summary>Values first .. first + l.Length as 8-bit lanes (any first: the values before the first whole byte one by one).</summary>
    public static void Load8(ReadOnlySpan<byte> s, int first, Span<byte> l, int bits, bool signed)
    {
        int per = 8 / bits, skip = first % per, mask = (1 << bits) - 1;
        int head = skip == 0 ? 0 : Math.Min(per - skip, l.Length);
        for (int i = 0; i < head; i++)
        {
            int v = (s[(first + i) / per] >> ((first + i) % per * bits)) & mask;
            l[i] = (byte)(signed && v >= 1 << (bits - 1) ? v - (1 << bits) : v);
        }

        if (head == l.Length) return;
        var rest = l[head..];
        var src = s[((first + head) / per)..];
        if (bits == 1)
        {
            for (int i = 0; i < rest.Length; i++) rest[i] = (byte)((src[i / 8] >> (i % 8)) & 1);
            return;
        }

        if (bits == 4) Cvt.Int4ToSByte(src, MemoryMarshal.Cast<byte, sbyte>(rest));
        else Cvt.Int2ToByte(src, rest, signed);
    }

    /// <summary>8-bit lanes into values first .. (any first), held to the range of the type.</summary>
    public static void Store8(ReadOnlySpan<byte> l, Span<byte> s, int first, int bits, bool signed)
    {
        int per = 8 / bits, skip = first % per, mask = (1 << bits) - 1;
        int head = skip == 0 ? 0 : Math.Min(per - skip, l.Length);
        long low = signed ? -(1 << (bits - 1)) : 0, high = signed ? (1 << (bits - 1)) - 1 : mask;
        for (int i = 0; i < head; i++)
        {
            int at = first + i, shift = at % per * bits;
            int v = (int)Math.Clamp(signed ? (sbyte)l[i] : l[i], low, high) & mask;
            s[at / per] = (byte)((s[at / per] & ~(mask << shift)) | v << shift);
        }

        if (head == l.Length) return;
        var rest = l[head..];
        var dst = s[((first + head) / per)..];
        if (bits == 4) Cvt.SByteToInt4(MemoryMarshal.Cast<byte, sbyte>(rest), dst);
        else if (bits == 2) Cvt.ByteToInt2(rest, dst, signed);
        else
        {
            // 1-bit values: 8 a byte, the first in the low bit (a whole byte at a time, the last partial byte kept beyond the values).
            // Values held to 0 or 1 (as the head values are), not their low bit.
            int whole = rest.Length / 8;
            for (int i = 0; i < whole; i++)
            {
                int v = 0;
                for (int t = 0; t < 8; t++) v |= (int)Math.Clamp(signed ? (sbyte)rest[i * 8 + t] : rest[i * 8 + t], low, high) << t;
                dst[i] = (byte)v;
            }

            for (int i = whole * 8; i < rest.Length; i++)
            {
                int t = i % 8;
                dst[whole] = (byte)((dst[whole] & ~(1 << t)) | (int)Math.Clamp(signed ? (sbyte)rest[i] : rest[i], low, high) << t);
            }
        }
    }

    public static void Load<TL>(ReadOnlySpan<byte> s, int first, Span<TL> l, int bits, bool signed) where TL : unmanaged
    {
        Span<byte> t = stackalloc byte[Chunk];
        for (int at = 0; at < l.Length; at += Chunk)
        {
            int n = Math.Min(Chunk, l.Length - at);
            Load8(s, first + at, t[..n], bits, signed);
            if (signed) Widths.Load(MemoryMarshal.Cast<byte, sbyte>(t[..n]), l.Slice(at, n));
            else Widths.Load(t[..n], l.Slice(at, n));
        }
    }

    public static void Store<TL>(ReadOnlySpan<TL> l, Span<byte> s, int first, int bits, bool signed) where TL : unmanaged
    {
        Span<byte> t = stackalloc byte[Chunk];
        for (int at = 0; at < l.Length; at += Chunk)
        {
            int n = Math.Min(Chunk, l.Length - at);
            if (signed) Widths.Store(l.Slice(at, n), MemoryMarshal.Cast<byte, sbyte>(t[..n]));
            else Widths.Store(l.Slice(at, n), t[..n]);
            Store8(t[..n], s, first + at, bits, signed);
        }
    }
}

public readonly struct Int4 : IAdad<Int4, sbyte>, IAdad<Int4, float>, IAdad<Int4, double>, IAdad<Int4, int>
{
    public static string Name => "int4";
    public static int Bits => 4;
    public static bool IsFloat => false;
    public static double Min => -8;
    public static double Max => 7;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<sbyte> l) => Packed.Load8(s, first, MemoryMarshal.AsBytes(l), 4, signed: true);
    public static void Store(ReadOnlySpan<sbyte> l, Span<byte> s, int first) => Packed.Store8(MemoryMarshal.AsBytes(l), s, first, 4, signed: true);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Packed.Load(s, first, l, 4, signed: true);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Packed.Store(l, s, first, 4, signed: true);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Packed.Load(s, first, l, 4, signed: true);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Packed.Store(l, s, first, 4, signed: true);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<int> l) => Packed.Load(s, first, l, 4, signed: true);
    public static void Store(ReadOnlySpan<int> l, Span<byte> s, int first) => Packed.Store(l, s, first, 4, signed: true);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Int4, sbyte, I8>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (S8Dot.IsSupported && b.Bound <= int.MaxValue) v.RunDot<Int4, sbyte, int, S8Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<Int4, float, F32>();
        else if (b.Bound <= int.MaxValue) v.Run<Int4, int, I32>();
        else v.Run<Int4, double, F64>();
    }
}

public readonly struct Int2 : IAdad<Int2, sbyte>, IAdad<Int2, float>, IAdad<Int2, double>, IAdad<Int2, int>
{
    public static string Name => "int2";
    public static int Bits => 2;
    public static bool IsFloat => false;
    public static double Min => -2;
    public static double Max => 1;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<sbyte> l) => Packed.Load8(s, first, MemoryMarshal.AsBytes(l), 2, signed: true);
    public static void Store(ReadOnlySpan<sbyte> l, Span<byte> s, int first) => Packed.Store8(MemoryMarshal.AsBytes(l), s, first, 2, signed: true);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Packed.Load(s, first, l, 2, signed: true);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Packed.Store(l, s, first, 2, signed: true);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Packed.Load(s, first, l, 2, signed: true);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Packed.Store(l, s, first, 2, signed: true);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<int> l) => Packed.Load(s, first, l, 2, signed: true);
    public static void Store(ReadOnlySpan<int> l, Span<byte> s, int first) => Packed.Store(l, s, first, 2, signed: true);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Int2, sbyte, I8>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (S8Dot.IsSupported && b.Bound <= int.MaxValue) v.RunDot<Int2, sbyte, int, S8Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<Int2, float, F32>();
        else if (b.Bound <= int.MaxValue) v.Run<Int2, int, I32>();
        else v.Run<Int2, double, F64>();
    }
}

// ---------------------------------------------------------------------------------------------- unsigned integers

public readonly struct UInt64 : IAdad<UInt64, ulong>, IAdad<UInt64, float>, IAdad<UInt64, double>, IAdad<UInt64, uint>,
    IAdad<UInt64, byte>, IAdad<UInt64, ushort>
{
    public static string Name => "uint64";
    public static int Bits => 64;
    public static bool IsFloat => false;
    public static double Min => 0;
    public static double Max => ulong.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<ulong>(s, count);
    static bool IAdad<UInt64, ulong>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<ulong> l) => Stored.Load<ulong, ulong>(s, first, l);
    public static void Store(ReadOnlySpan<ulong> l, Span<byte> s, int first) => Stored.Store<ulong, ulong>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<ulong, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, ulong>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<ulong, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, ulong>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<uint> l) => Stored.Load<ulong, uint>(s, first, l);
    public static void Store(ReadOnlySpan<uint> l, Span<byte> s, int first) => Stored.Store<uint, ulong>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<byte> l) => Stored.Load<ulong, byte>(s, first, l);
    public static void Store(ReadOnlySpan<byte> l, Span<byte> s, int first) => Stored.Store<byte, ulong>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<ushort> l) => Stored.Load<ulong, ushort>(s, first, l);
    public static void Store(ReadOnlySpan<ushort> l, Span<byte> s, int first) => Stored.Store<ushort, ulong>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<UInt64, ulong, U64>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (U8Dot.IsSupported && b.Within(0, byte.MaxValue) && b.Bound <= uint.MaxValue) v.RunDot<UInt64, byte, uint, U8Dot>();
        else if (U16Dot.IsSupported && b.Within(0, ushort.MaxValue) && b.Bound <= uint.MaxValue) v.RunDot<UInt64, ushort, uint, U16Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<UInt64, float, F32>();
        else if (b.Bound <= uint.MaxValue) v.Run<UInt64, uint, U32>();
        else if (b.Bound <= Exact.Float64) v.Run<UInt64, double, F64>();
        else v.Run<UInt64, ulong, U64>();
    }
}

public readonly struct UInt32 : IAdad<UInt32, uint>, IAdad<UInt32, float>, IAdad<UInt32, double>, IAdad<UInt32, ulong>,
    IAdad<UInt32, byte>, IAdad<UInt32, ushort>
{
    public static string Name => "uint32";
    public static int Bits => 32;
    public static bool IsFloat => false;
    public static double Min => 0;
    public static double Max => uint.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<uint>(s, count);
    static bool IAdad<UInt32, uint>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<uint> l) => Stored.Load<uint, uint>(s, first, l);
    public static void Store(ReadOnlySpan<uint> l, Span<byte> s, int first) => Stored.Store<uint, uint>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<uint, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, uint>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<uint, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, uint>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<ulong> l) => Stored.Load<uint, ulong>(s, first, l);
    public static void Store(ReadOnlySpan<ulong> l, Span<byte> s, int first) => Stored.Store<ulong, uint>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<byte> l) => Stored.Load<uint, byte>(s, first, l);
    public static void Store(ReadOnlySpan<byte> l, Span<byte> s, int first) => Stored.Store<byte, uint>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<ushort> l) => Stored.Load<uint, ushort>(s, first, l);
    public static void Store(ReadOnlySpan<ushort> l, Span<byte> s, int first) => Stored.Store<ushort, uint>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<UInt32, uint, U32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (U8Dot.IsSupported && b.Within(0, byte.MaxValue) && b.Bound <= uint.MaxValue) v.RunDot<UInt32, byte, uint, U8Dot>();
        else if (U16Dot.IsSupported && b.Within(0, ushort.MaxValue) && b.Bound <= uint.MaxValue) v.RunDot<UInt32, ushort, uint, U16Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<UInt32, float, F32>();
        else if (b.Bound <= uint.MaxValue) v.Run<UInt32, uint, U32>();
        else if (b.Bound <= Exact.Float64) v.Run<UInt32, double, F64>();
        else v.Run<UInt32, ulong, U64>();
    }
}

public readonly struct UInt16 : IAdad<UInt16, ushort>, IAdad<UInt16, float>, IAdad<UInt16, double>, IAdad<UInt16, uint>,
    IAdad<UInt16, ulong>, IAdad<UInt16, byte>
{
    public static string Name => "uint16";
    public static int Bits => 16;
    public static bool IsFloat => false;
    public static double Min => 0;
    public static double Max => ushort.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<ushort>(s, count);
    static bool IAdad<UInt16, ushort>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<ushort> l) => Stored.Load<ushort, ushort>(s, first, l);
    public static void Store(ReadOnlySpan<ushort> l, Span<byte> s, int first) => Stored.Store<ushort, ushort>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<ushort, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, ushort>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<ushort, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, ushort>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<uint> l) => Stored.Load<ushort, uint>(s, first, l);
    public static void Store(ReadOnlySpan<uint> l, Span<byte> s, int first) => Stored.Store<uint, ushort>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<ulong> l) => Stored.Load<ushort, ulong>(s, first, l);
    public static void Store(ReadOnlySpan<ulong> l, Span<byte> s, int first) => Stored.Store<ulong, ushort>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<byte> l) => Stored.Load<ushort, byte>(s, first, l);
    public static void Store(ReadOnlySpan<byte> l, Span<byte> s, int first) => Stored.Store<byte, ushort>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<UInt16, ushort, U16>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (U8Dot.IsSupported && b.Within(0, byte.MaxValue) && b.Bound <= uint.MaxValue) v.RunDot<UInt16, byte, uint, U8Dot>();
        else if (U16Dot.IsSupported && b.Bound <= uint.MaxValue) v.RunDot<UInt16, ushort, uint, U16Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<UInt16, float, F32>();
        else if (b.Bound <= uint.MaxValue) v.Run<UInt16, uint, U32>();
        else if (b.Bound <= Exact.Float64) v.Run<UInt16, double, F64>();
        else v.Run<UInt16, ulong, U64>();
    }
}

public readonly struct UInt8 : IAdad<UInt8, byte>, IAdad<UInt8, float>, IAdad<UInt8, double>, IAdad<UInt8, uint>
{
    public static string Name => "uint8";
    public static int Bits => 8;
    public static bool IsFloat => false;
    public static double Min => 0;
    public static double Max => byte.MaxValue;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => Stored.Range<byte>(s, count);
    static bool IAdad<UInt8, byte>.Direct => true;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<byte> l) => Stored.Load<byte, byte>(s, first, l);
    public static void Store(ReadOnlySpan<byte> l, Span<byte> s, int first) => Stored.Store<byte, byte>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Stored.Load<byte, float>(s, first, l);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Stored.Store<float, byte>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Stored.Load<byte, double>(s, first, l);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Stored.Store<double, byte>(l, s, first);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<uint> l) => Stored.Load<byte, uint>(s, first, l);
    public static void Store(ReadOnlySpan<uint> l, Span<byte> s, int first) => Stored.Store<uint, byte>(l, s, first);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<UInt8, byte, U8>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (U8Dot.IsSupported && b.Bound <= uint.MaxValue) v.RunDot<UInt8, byte, uint, U8Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<UInt8, float, F32>();
        else if (b.Bound <= uint.MaxValue) v.Run<UInt8, uint, U32>();
        else v.Run<UInt8, double, F64>();
    }
}

public readonly struct UInt2 : IAdad<UInt2, byte>, IAdad<UInt2, float>, IAdad<UInt2, double>, IAdad<UInt2, uint>
{
    public static string Name => "uint2";
    public static int Bits => 2;
    public static bool IsFloat => false;
    public static double Min => 0;
    public static double Max => 3;
    public static void Load(ReadOnlySpan<byte> s, int first, Span<byte> l) => Packed.Load8(s, first, l, 2, signed: false);
    public static void Store(ReadOnlySpan<byte> l, Span<byte> s, int first) => Packed.Store8(l, s, first, 2, signed: false);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Packed.Load(s, first, l, 2, signed: false);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Packed.Store(l, s, first, 2, signed: false);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Packed.Load(s, first, l, 2, signed: false);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Packed.Store(l, s, first, 2, signed: false);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<uint> l) => Packed.Load(s, first, l, 2, signed: false);
    public static void Store(ReadOnlySpan<uint> l, Span<byte> s, int first) => Packed.Store(l, s, first, 2, signed: false);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<UInt2, byte, U8>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (U8Dot.IsSupported && b.Bound <= uint.MaxValue) v.RunDot<UInt2, byte, uint, U8Dot>();
        else if (b.Bound <= Exact.Float32) v.Run<UInt2, float, F32>();
        else if (b.Bound <= uint.MaxValue) v.Run<UInt2, uint, U32>();
        else v.Run<UInt2, double, F64>();
    }
}
