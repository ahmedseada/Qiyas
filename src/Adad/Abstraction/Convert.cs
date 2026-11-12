// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Every load and store of Adad's types on vectors (Vector<T>: the width the processor reports). Rounding is to
// nearest, ties to even, and stores are held to the target's range; the float encoders follow IEEE 754 (binary16),
// bfloat16 and the OCP fp8 formats (E4M3: no infinity, 0x7F NaN; E5M2: IEEE-shaped). Checks proves each against an
// independent scalar reference written from the format's definition.
//
//   Cvt     the conversions, type by type
//   Widths  the generic entry points Load<TS, TL> (exact) and Store<TL, TS> (rounded, held) over any pair of lanes
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

internal static class Cvt
{
    public static ReadOnlySpan<T> As<T>(ReadOnlySpan<byte> bytes) where T : unmanaged => MemoryMarshal.Cast<byte, T>(bytes);

    public static Span<T> As<T>(Span<byte> bytes) where T : unmanaged => MemoryMarshal.Cast<byte, T>(bytes);

    private static Vector<float> F(Vector<uint> bits) => Vector.AsVectorSingle(bits);

    private static Vector<uint> U(Vector<float> v) => Vector.AsVectorUInt32(v);

    // ---------------------------------------------------------------- integers to float32 (exact: 24 bits of value)

    public static void ToFloat(ReadOnlySpan<sbyte> s, Span<float> d)
    {
        int i = 0, w = Vector<sbyte>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<sbyte>(s[i..]), out Vector<short> lo, out Vector<short> hi);
            Vector.Widen(lo, out Vector<int> a, out Vector<int> b);
            Vector.Widen(hi, out Vector<int> c, out Vector<int> e);
            Vector.ConvertToSingle(a).CopyTo(d[i..]);
            Vector.ConvertToSingle(b).CopyTo(d[(i + q)..]);
            Vector.ConvertToSingle(c).CopyTo(d[(i + 2 * q)..]);
            Vector.ConvertToSingle(e).CopyTo(d[(i + 3 * q)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToFloat(ReadOnlySpan<byte> s, Span<float> d)
    {
        int i = 0, w = Vector<byte>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<byte>(s[i..]), out Vector<ushort> lo, out Vector<ushort> hi);
            Vector.Widen(lo, out Vector<uint> a, out Vector<uint> b);
            Vector.Widen(hi, out Vector<uint> c, out Vector<uint> e);
            Vector.ConvertToSingle(Vector.AsVectorInt32(a)).CopyTo(d[i..]);
            Vector.ConvertToSingle(Vector.AsVectorInt32(b)).CopyTo(d[(i + q)..]);
            Vector.ConvertToSingle(Vector.AsVectorInt32(c)).CopyTo(d[(i + 2 * q)..]);
            Vector.ConvertToSingle(Vector.AsVectorInt32(e)).CopyTo(d[(i + 3 * q)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToFloat(ReadOnlySpan<short> s, Span<float> d)
    {
        int i = 0, w = Vector<short>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<short>(s[i..]), out Vector<int> a, out Vector<int> b);
            Vector.ConvertToSingle(a).CopyTo(d[i..]);
            Vector.ConvertToSingle(b).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToFloat(ReadOnlySpan<ushort> s, Span<float> d)
    {
        int i = 0, w = Vector<ushort>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<ushort>(s[i..]), out Vector<uint> a, out Vector<uint> b);
            Vector.ConvertToSingle(Vector.AsVectorInt32(a)).CopyTo(d[i..]);
            Vector.ConvertToSingle(Vector.AsVectorInt32(b)).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToFloat(ReadOnlySpan<int> s, Span<float> d)
    {
        int i = 0, w = Vector<int>.Count;
        for (; i + w <= s.Length; i += w) Vector.ConvertToSingle(new Vector<int>(s[i..])).CopyTo(d[i..]);
        for (; i < s.Length; i++) d[i] = s[i];
    }

    // Unsigned and 64-bit integers: a vector whose values all fit int32 converts through it (one rounding, the
    // hardware's); any other vector value by value (one rounding to nearest, ties to even).
    public static void ToFloat(ReadOnlySpan<uint> s, Span<float> d)
    {
        int i = 0, w = Vector<uint>.Count;
        var top = new Vector<uint>(int.MaxValue);
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<uint>(s[i..]);
            if (Vector.LessThanOrEqualAll(v, top)) Vector.ConvertToSingle(Vector.AsVectorInt32(v)).CopyTo(d[i..]);
            else for (int l = i; l < i + w; l++) d[l] = s[l];
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToFloat(ReadOnlySpan<long> s, Span<float> d)
    {
        int i = 0, w = Vector<long>.Count;
        Vector<long> low = new(int.MinValue), high = new(int.MaxValue);
        for (; i + 2 * w <= s.Length; i += 2 * w)
        {
            var a = new Vector<long>(s[i..]);
            var b = new Vector<long>(s[(i + w)..]);
            if (Vector.GreaterThanOrEqualAll(Vector.Min(a, b), low) && Vector.LessThanOrEqualAll(Vector.Max(a, b), high))
                Vector.ConvertToSingle(Vector.Narrow(a, b)).CopyTo(d[i..]);
            else for (int l = i; l < i + 2 * w; l++) d[l] = s[l];
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToFloat(ReadOnlySpan<ulong> s, Span<float> d)
    {
        int i = 0, w = Vector<ulong>.Count;
        var high = new Vector<ulong>(int.MaxValue);
        for (; i + 2 * w <= s.Length; i += 2 * w)
        {
            var a = new Vector<ulong>(s[i..]);
            var b = new Vector<ulong>(s[(i + w)..]);
            if (Vector.LessThanOrEqualAll(Vector.Max(a, b), high))
                Vector.ConvertToSingle(Vector.AsVectorInt32(Vector.Narrow(a, b))).CopyTo(d[i..]);
            else for (int l = i; l < i + 2 * w; l++) d[l] = s[l];
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    // ---------------------------------------------------------------- integers to float64 (exact to 2^53; 64-bit values
    // past it rounded by the hardware, to nearest even)

    public static void ToDouble(ReadOnlySpan<long> s, Span<double> d)
    {
        int i = 0, w = Vector<long>.Count;
        for (; i + w <= s.Length; i += w) Vector.ConvertToDouble(new Vector<long>(s[i..])).CopyTo(d[i..]);
        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToDouble(ReadOnlySpan<ulong> s, Span<double> d)
    {
        int i = 0, w = Vector<ulong>.Count;
        for (; i + w <= s.Length; i += w) Vector.ConvertToDouble(new Vector<ulong>(s[i..])).CopyTo(d[i..]);
        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToDouble(ReadOnlySpan<int> s, Span<double> d)
    {
        int i = 0, w = Vector<int>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<int>(s[i..]), out Vector<long> a, out Vector<long> b);
            Vector.ConvertToDouble(a).CopyTo(d[i..]);
            Vector.ConvertToDouble(b).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToDouble(ReadOnlySpan<uint> s, Span<double> d)
    {
        int i = 0, w = Vector<uint>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<uint>(s[i..]), out Vector<ulong> a, out Vector<ulong> b);
            Vector.ConvertToDouble(a).CopyTo(d[i..]);
            Vector.ConvertToDouble(b).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    public static void ToDouble(ReadOnlySpan<float> s, Span<double> d)
    {
        int i = 0, w = Vector<float>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<float>(s[i..]), out Vector<double> a, out Vector<double> b);
            a.CopyTo(d[i..]);
            b.CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i];
    }

    // ---------------------------------------------------------------- float32 to integers (rounded, held to the range)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> RoundClamp(Vector<float> v, float low, float high) =>
        Vector.Min(Vector.Max(Vector.Round(v), new Vector<float>(low)), new Vector<float>(high));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float RoundClamp(float v, float low, float high) => float.IsNaN(v) ? 0 : Math.Clamp(MathF.Round(v), low, high);

    public static void FromFloat(ReadOnlySpan<float> s, Span<sbyte> d, float low = sbyte.MinValue, float high = sbyte.MaxValue)
    {
        int i = 0, w = Vector<sbyte>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            var a = Vector.ConvertToInt32(RoundClamp(new Vector<float>(s[i..]), low, high));
            var b = Vector.ConvertToInt32(RoundClamp(new Vector<float>(s[(i + q)..]), low, high));
            var c = Vector.ConvertToInt32(RoundClamp(new Vector<float>(s[(i + 2 * q)..]), low, high));
            var e = Vector.ConvertToInt32(RoundClamp(new Vector<float>(s[(i + 3 * q)..]), low, high));
            Vector.Narrow(Vector.Narrow(a, b), Vector.Narrow(c, e)).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = (sbyte)RoundClamp(s[i], low, high);
    }

    public static void FromFloat(ReadOnlySpan<float> s, Span<byte> d, float high = byte.MaxValue)
    {
        int i = 0, w = Vector<byte>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            var a = Vector.ConvertToUInt32(RoundClamp(new Vector<float>(s[i..]), 0, high));
            var b = Vector.ConvertToUInt32(RoundClamp(new Vector<float>(s[(i + q)..]), 0, high));
            var c = Vector.ConvertToUInt32(RoundClamp(new Vector<float>(s[(i + 2 * q)..]), 0, high));
            var e = Vector.ConvertToUInt32(RoundClamp(new Vector<float>(s[(i + 3 * q)..]), 0, high));
            Vector.Narrow(Vector.Narrow(a, b), Vector.Narrow(c, e)).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = (byte)RoundClamp(s[i], 0, high);
    }

    public static void FromFloat(ReadOnlySpan<float> s, Span<short> d)
    {
        int i = 0, w = Vector<short>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            var a = Vector.ConvertToInt32(RoundClamp(new Vector<float>(s[i..]), short.MinValue, short.MaxValue));
            var b = Vector.ConvertToInt32(RoundClamp(new Vector<float>(s[(i + h)..]), short.MinValue, short.MaxValue));
            Vector.Narrow(a, b).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = (short)RoundClamp(s[i], short.MinValue, short.MaxValue);
    }

    public static void FromFloat(ReadOnlySpan<float> s, Span<ushort> d)
    {
        int i = 0, w = Vector<ushort>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            var a = Vector.ConvertToUInt32(RoundClamp(new Vector<float>(s[i..]), 0, ushort.MaxValue));
            var b = Vector.ConvertToUInt32(RoundClamp(new Vector<float>(s[(i + h)..]), 0, ushort.MaxValue));
            Vector.Narrow(a, b).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = (ushort)RoundClamp(s[i], 0, ushort.MaxValue);
    }

    // The integer types' largest values are not floats: a value from the next power of two on holds to the largest itself.
    public static void FromFloat(ReadOnlySpan<float> s, Span<int> d)
    {
        int i = 0, w = Vector<float>.Count;
        Vector<float> top = new(2147483648f);
        Vector<int> max = new(int.MaxValue);
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<float>(s[i..]);
            Vector.ConditionalSelect(Vector.GreaterThanOrEqual(v, top), max, Vector.ConvertToInt32(RoundClamp(v, int.MinValue, 2147483520f))).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = s[i] >= 2147483648f ? int.MaxValue : (int)RoundClamp(s[i], int.MinValue, 2147483520f);
    }

    public static void FromFloat(ReadOnlySpan<float> s, Span<uint> d)
    {
        int i = 0, w = Vector<float>.Count;
        Vector<float> top = new(4294967296f);
        Vector<uint> max = new(uint.MaxValue);
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<float>(s[i..]);
            Vector.ConditionalSelect(Vector.AsVectorUInt32(Vector.GreaterThanOrEqual(v, top)), max, Vector.ConvertToUInt32(RoundClamp(v, 0, 4294967040f))).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = s[i] >= 4294967296f ? uint.MaxValue : (uint)RoundClamp(s[i], 0, 4294967040f);
    }

    public static void FromFloat(ReadOnlySpan<float> s, Span<long> d)
    {
        int i = 0, w = Vector<float>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<float>(s[i..]);
            Vector.Widen(RoundClamp(v, -9.223372e18f, 9.223371e18f), out Vector<double> a, out Vector<double> b);
            Vector.Widen(v, out Vector<double> va, out Vector<double> vb);
            Vector<double> top = new(9223372036854775808.0);
            Vector<long> max = new(long.MaxValue);
            Vector.ConditionalSelect(Vector.GreaterThanOrEqual(va, top), max, Vector.ConvertToInt64(a)).CopyTo(d[i..]);
            Vector.ConditionalSelect(Vector.GreaterThanOrEqual(vb, top), max, Vector.ConvertToInt64(b)).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i] >= 9.223372036854775808e18f ? long.MaxValue : (long)RoundClamp(s[i], -9.223372e18f, 9.223371e18f);
    }

    public static void FromFloat(ReadOnlySpan<float> s, Span<ulong> d)
    {
        int i = 0, w = Vector<float>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<float>(s[i..]);
            Vector.Widen(RoundClamp(v, 0, 1.8446743e19f), out Vector<double> a, out Vector<double> b);
            Vector.Widen(v, out Vector<double> va, out Vector<double> vb);
            Vector<double> top = new(18446744073709551616.0);
            Vector<ulong> max = new(ulong.MaxValue);
            Vector.ConditionalSelect(Vector.AsVectorUInt64(Vector.GreaterThanOrEqual(va, top)), max, Vector.ConvertToUInt64(a)).CopyTo(d[i..]);
            Vector.ConditionalSelect(Vector.AsVectorUInt64(Vector.GreaterThanOrEqual(vb, top)), max, Vector.ConvertToUInt64(b)).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = s[i] >= 1.8446744073709551616e19f ? ulong.MaxValue : (ulong)RoundClamp(s[i], 0, 1.8446743e19f);
    }

    // ---------------------------------------------------------------- float64 to integers (rounded, held to the range)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> RoundClamp(Vector<double> v, double low, double high) =>
        Vector.Min(Vector.Max(Vector.Round(v), new Vector<double>(low)), new Vector<double>(high));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double RoundClamp(double v, double low, double high) => double.IsNaN(v) ? 0 : Math.Clamp(Math.Round(v), low, high);

    public static void FromDouble(ReadOnlySpan<double> s, Span<long> d, double low = long.MinValue, double high = 9223372036854774784.0)
    {
        int i = 0, w = Vector<double>.Count;
        bool whole = high >= 9223372036854774784.0;                           // the whole range: from 2^63 on, the largest
        Vector<double> top = new(9223372036854775808.0);
        Vector<long> max = new(long.MaxValue);
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<double>(s[i..]);
            var r = Vector.ConvertToInt64(RoundClamp(v, low, high));
            (whole ? Vector.ConditionalSelect(Vector.GreaterThanOrEqual(v, top), max, r) : r).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = whole && s[i] >= 9223372036854775808.0 ? long.MaxValue : (long)RoundClamp(s[i], low, high);
    }

    public static void FromDouble(ReadOnlySpan<double> s, Span<ulong> d, double high = 18446744073709549568.0)
    {
        int i = 0, w = Vector<double>.Count;
        bool whole = high >= 18446744073709549568.0;
        Vector<double> top = new(18446744073709551616.0);
        Vector<ulong> max = new(ulong.MaxValue);
        for (; i + w <= s.Length; i += w)
        {
            var v = new Vector<double>(s[i..]);
            var r = Vector.ConvertToUInt64(RoundClamp(v, 0, high));
            (whole ? Vector.ConditionalSelect(Vector.AsVectorUInt64(Vector.GreaterThanOrEqual(v, top)), max, r) : r).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = whole && s[i] >= 18446744073709551616.0 ? ulong.MaxValue : (ulong)RoundClamp(s[i], 0, high);
    }

    public static void FromDouble(ReadOnlySpan<double> s, Span<float> d)
    {
        int i = 0, w = Vector<double>.Count;
        for (; i + 2 * w <= s.Length; i += 2 * w) Vector.Narrow(new Vector<double>(s[i..]), new Vector<double>(s[(i + w)..])).CopyTo(d[i..]);
        for (; i < s.Length; i++) d[i] = (float)s[i];
    }

    // ---------------------------------------------------------------- float16 (binary16: round to nearest even in integer
    // arithmetic, subnormal halves through one float addition, from 65520 an infinity, a NaN quiet with its top ten
    // payload bits)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> HalfBits(Vector<float> v)
    {
        var bits = U(v);
        var sign = (bits >> 16) & new Vector<uint>(0x8000u);
        var abs = bits & new Vector<uint>(0x7FFFFFFFu);
        var normal = (abs + new Vector<uint>(0xC8000FFFu) + ((abs >> 13) & Vector<uint>.One)) >> 13;
        var sub = U(F(abs) + new Vector<float>(0.5f)) - new Vector<uint>(0x3F000000u);
        var r = Vector.ConditionalSelect(Vector.LessThan(abs, new Vector<uint>(0x38800000u)), sub, normal);
        r = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(abs, new Vector<uint>(0x477FF000u)), new Vector<uint>(0x7C00u), r);
        r = Vector.ConditionalSelect(Vector.GreaterThan(abs, new Vector<uint>(0x7F800000u)), new Vector<uint>(0x7E00u) | ((abs >> 13) & new Vector<uint>(0x3FFu)), r);
        return r | sign;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> HalfValue(Vector<uint> h)
    {
        var sign = (h & new Vector<uint>(0x8000u)) << 16;
        var e = (h >> 10) & new Vector<uint>(0x1Fu);
        var m = h & new Vector<uint>(0x3FFu);
        var normal = ((e + new Vector<uint>(112u)) << 23) | (m << 13);
        var sub = U(Vector.ConvertToSingle(Vector.AsVectorInt32(m)) * new Vector<float>(1f / 16777216f));
        // Exponent 31: infinity, or a NaN made quiet (its payload kept), as a conversion must.
        var quiet = Vector.ConditionalSelect(Vector.Equals(m, Vector<uint>.Zero), Vector<uint>.Zero, new Vector<uint>(0x400000u));
        var top = new Vector<uint>(0x7F800000u) | (m << 13) | quiet;
        var r = Vector.ConditionalSelect(Vector.Equals(e, Vector<uint>.Zero), sub, normal);
        r = Vector.ConditionalSelect(Vector.Equals(e, new Vector<uint>(31u)), top, r);
        return r | sign;
    }

    public static void ToHalf(ReadOnlySpan<float> s, Span<ushort> d)
    {
        int i = 0, w = Vector<ushort>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w) Vector.Narrow(HalfBits(new Vector<float>(s[i..])), HalfBits(new Vector<float>(s[(i + h)..]))).CopyTo(d[i..]);
        if (i < s.Length)
        {
            Span<float> ps = stackalloc float[w];
            Span<ushort> pd = stackalloc ushort[w];
            s[i..].CopyTo(ps);
            ToHalf(ps, pd);
            pd[..(s.Length - i)].CopyTo(d[i..]);
        }
    }

    public static void FromHalf(ReadOnlySpan<ushort> s, Span<float> d)
    {
        int i = 0, w = Vector<ushort>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<ushort>(s[i..]), out Vector<uint> a, out Vector<uint> b);
            F(HalfValue(a)).CopyTo(d[i..]);
            F(HalfValue(b)).CopyTo(d[(i + h)..]);
        }

        if (i < s.Length)
        {
            Span<ushort> ps = stackalloc ushort[w];
            Span<float> pd = stackalloc float[w];
            s[i..].CopyTo(ps);
            FromHalf(ps, pd);
            pd[..(s.Length - i)].CopyTo(d[i..]);
        }
    }

    // ---------------------------------------------------------------- bfloat16 (the top 16 bits of binary32, rounded to
    // nearest even; a NaN kept quiet)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> BF16Bits(Vector<float> v)
    {
        var bits = U(v);
        var rounded = (bits + new Vector<uint>(0x7FFFu) + ((bits >> 16) & Vector<uint>.One)) >> 16;
        var nan = ~Vector.AsVectorUInt32(Vector.Equals(v, v));
        return Vector.ConditionalSelect(nan, (bits >> 16) | new Vector<uint>(0x40u), rounded);
    }

    public static void ToBFloat16(ReadOnlySpan<float> s, Span<ushort> d)
    {
        int i = 0, w = Vector<ushort>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w) Vector.Narrow(BF16Bits(new Vector<float>(s[i..])), BF16Bits(new Vector<float>(s[(i + h)..]))).CopyTo(d[i..]);
        if (i < s.Length)
        {
            Span<float> ps = stackalloc float[w];
            Span<ushort> pd = stackalloc ushort[w];
            s[i..].CopyTo(ps);
            ToBFloat16(ps, pd);
            pd[..(s.Length - i)].CopyTo(d[i..]);
        }
    }

    public static void FromBFloat16(ReadOnlySpan<ushort> s, Span<float> d)
    {
        int i = 0, w = Vector<ushort>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<ushort>(s[i..]), out Vector<uint> a, out Vector<uint> b);
            F(a << 16).CopyTo(d[i..]);
            F(b << 16).CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = BitConverter.UInt32BitsToSingle((uint)s[i] << 16);
    }

    // ---------------------------------------------------------------- fp8, both ways on vectors (a NaN code to the quiet NaN
    // of its sign; to a code: nearest, ties to the even code; past the largest finite's rounding point E4M3's NaN code
    // and E5M2's infinity; a NaN to 0x7F)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> E4M3Value(Vector<uint> c)
    {
        var sign = (c & new Vector<uint>(0x80u)) << 24;
        var e = (c >> 3) & new Vector<uint>(15u);
        var m = c & new Vector<uint>(7u);
        var normal = sign | ((e + new Vector<uint>(120u)) << 23) | (m << 20);
        var sub = U(Vector.ConvertToSingle(Vector.AsVectorInt32(m)) * new Vector<float>(1f / 512f)) | sign;
        var r = Vector.ConditionalSelect(Vector.Equals(e, Vector<uint>.Zero), sub, normal);
        var nan = Vector.Equals(c & new Vector<uint>(0x7Fu), new Vector<uint>(0x7Fu));
        return F(Vector.ConditionalSelect(nan, sign | new Vector<uint>(0x7FC00000u), r));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> E5M2Value(Vector<uint> c)
    {
        var sign = (c & new Vector<uint>(0x80u)) << 24;
        var e = (c >> 2) & new Vector<uint>(31u);
        var m = c & new Vector<uint>(3u);
        var normal = sign | ((e + new Vector<uint>(112u)) << 23) | (m << 21);
        var sub = U(Vector.ConvertToSingle(Vector.AsVectorInt32(m)) * new Vector<float>(1f / 65536f)) | sign;
        var r = Vector.ConditionalSelect(Vector.Equals(e, Vector<uint>.Zero), sub, normal);
        var top = Vector.Equals(e, new Vector<uint>(31u));
        r = Vector.ConditionalSelect(top, sign | new Vector<uint>(0x7F800000u), r);
        var nan = top & ~Vector.Equals(m, Vector<uint>.Zero);
        return F(Vector.ConditionalSelect(nan, sign | new Vector<uint>(0x7FC00000u), r));
    }

    private static void Float8To(ReadOnlySpan<byte> s, Span<float> d, bool e4m3)
    {
        int i = 0, w = Vector<byte>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Widen(new Vector<byte>(s[i..]), out Vector<ushort> lo, out Vector<ushort> hi);
            Vector.Widen(lo, out Vector<uint> a, out Vector<uint> b);
            Vector.Widen(hi, out Vector<uint> c, out Vector<uint> e);
            (e4m3 ? E4M3Value(a) : E5M2Value(a)).CopyTo(d[i..]);
            (e4m3 ? E4M3Value(b) : E5M2Value(b)).CopyTo(d[(i + q)..]);
            (e4m3 ? E4M3Value(c) : E5M2Value(c)).CopyTo(d[(i + 2 * q)..]);
            (e4m3 ? E4M3Value(e) : E5M2Value(e)).CopyTo(d[(i + 3 * q)..]);
        }

        if (i < s.Length)
        {
            Span<byte> ps = stackalloc byte[w];
            Span<float> pd = stackalloc float[w];
            s[i..].CopyTo(ps);
            Float8To(ps, pd, e4m3);
            pd[..(s.Length - i)].CopyTo(d[i..]);
        }
    }

    public static void E4M3ToFloat(ReadOnlySpan<byte> s, Span<float> d) => Float8To(s, d, e4m3: true);

    public static void E5M2ToFloat(ReadOnlySpan<byte> s, Span<float> d) => Float8To(s, d, e4m3: false);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> E4M3Code(Vector<float> v)
    {
        var bits = U(v);
        var sign = (bits >> 24) & new Vector<uint>(0x80u);
        var abs = bits & new Vector<uint>(0x7FFFFFFFu);
        var normal = ((abs + new Vector<uint>(0x7FFFFu) + ((abs >> 20) & Vector<uint>.One)) >> 20) - new Vector<uint>(960u);
        var sub = Vector.ConvertToUInt32(Vector.Round(F(abs) * new Vector<float>(512f)));
        var code = Vector.ConditionalSelect(Vector.LessThan(abs, new Vector<uint>(0x3C800000u)), sub, normal);
        // From 464 (the largest finite plus half its ulp) the value overflows: E4M3 has no infinity, so its NaN code.
        code = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(abs, new Vector<uint>(0x43E80000u)), new Vector<uint>(0x7Fu), code);
        return code | sign;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> E5M2Code(Vector<float> v)
    {
        var bits = U(v);
        var sign = (bits >> 24) & new Vector<uint>(0x80u);
        var abs = bits & new Vector<uint>(0x7FFFFFFFu);
        var normal = ((abs + new Vector<uint>(0xFFFFFu) + ((abs >> 21) & Vector<uint>.One)) >> 21) - new Vector<uint>(448u);
        var sub = Vector.ConvertToUInt32(Vector.Round(F(abs) * new Vector<float>(65536f)));
        var code = Vector.ConditionalSelect(Vector.LessThan(abs, new Vector<uint>(0x38800000u)), sub, normal);
        // From 61440 (the largest finite plus half its ulp) the value overflows to infinity; a NaN stays a NaN.
        code = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(abs, new Vector<uint>(0x47700000u)), new Vector<uint>(0x7Cu), code);
        code = Vector.ConditionalSelect(Vector.GreaterThan(abs, new Vector<uint>(0x7F800000u)), new Vector<uint>(0x7Fu), code);
        return code | sign;
    }

    private static void Float8From(ReadOnlySpan<float> s, Span<byte> d, bool e4m3)
    {
        int i = 0, w = Vector<byte>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            Vector.Narrow(Vector.Narrow(Code(s, i, e4m3), Code(s, i + q, e4m3)), Vector.Narrow(Code(s, i + 2 * q, e4m3), Code(s, i + 3 * q, e4m3))).CopyTo(d[i..]);
        }

        if (i < s.Length)
        {
            Span<float> ps = stackalloc float[w];
            Span<byte> pd = stackalloc byte[w];
            s[i..].CopyTo(ps);
            Float8From(ps, pd, e4m3);
            pd[..(s.Length - i)].CopyTo(d[i..]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<uint> Code(ReadOnlySpan<float> s, int at, bool e4m3) => e4m3 ? E4M3Code(new Vector<float>(s[at..])) : E5M2Code(new Vector<float>(s[at..]));

    public static void E4M3FromFloat(ReadOnlySpan<float> s, Span<byte> d) => Float8From(s, d, e4m3: true);

    public static void E5M2FromFloat(ReadOnlySpan<float> s, Span<byte> d) => Float8From(s, d, e4m3: false);

    // ---------------------------------------------------------------- packed integers on vectors, in order

    // int4 (two a byte, the first in the low nibble) to sbyte lanes: each byte widened to a 16-bit lane that becomes the
    // pair (low nibble, high nibble), sign-extended by shifts; no shuffles.
    public static void Int4ToSByte(ReadOnlySpan<byte> s, Span<sbyte> d)
    {
        int bytes = d.Length / 2, i = 0, w = Vector<byte>.Count, h = Vector<ushort>.Count;
        var low = new Vector<short>(0x00FF);
        for (; i + w <= bytes; i += w)
        {
            Vector.Widen(new Vector<byte>(s[i..]), out Vector<ushort> a, out Vector<ushort> b);
            Vector.AsVectorSByte(Pair(Vector.AsVectorInt16(a), low)).CopyTo(d[(2 * i)..]);
            Vector.AsVectorSByte(Pair(Vector.AsVectorInt16(b), low)).CopyTo(d[(2 * i + 2 * h)..]);
        }

        for (; i < bytes; i++)
        {
            int v = s[i];
            d[2 * i] = (sbyte)((sbyte)(v << 4) >> 4);
            d[2 * i + 1] = (sbyte)((sbyte)v >> 4);
        }

        if ((d.Length & 1) != 0) d[^1] = (sbyte)((sbyte)(s[bytes] << 4) >> 4);

        static Vector<short> Pair(Vector<short> x, Vector<short> low) =>
            (Vector.ShiftRightArithmetic(Vector.ShiftLeft(x, 12), 12) & low) | Vector.ShiftLeft(Vector.ShiftRightArithmetic(Vector.ShiftLeft(x, 8), 12), 8);
    }

    // sbyte lanes held to [-8, 7] packed two a byte: each pair read as a 16-bit lane, its two nibbles joined, narrowed.
    public static void SByteToInt4(ReadOnlySpan<sbyte> s, Span<byte> d)
    {
        int bytes = s.Length / 2, i = 0, w = Vector<byte>.Count, h = Vector<ushort>.Count;
        Vector<sbyte> lo = new(-8), hi = new(7);
        var nib = new Vector<ushort>(0x000F);
        for (; i + w <= bytes; i += w)
        {
            var a = Vector.AsVectorUInt16(Vector.Min(Vector.Max(new Vector<sbyte>(s[(2 * i)..]), lo), hi));
            var b = Vector.AsVectorUInt16(Vector.Min(Vector.Max(new Vector<sbyte>(s[(2 * i + 2 * h)..]), lo), hi));
            Vector.Narrow((a & nib) | (((a >> 8) & nib) << 4), (b & nib) | (((b >> 8) & nib) << 4)).CopyTo(d[i..]);
        }

        for (; i < bytes; i++)
        {
            int l = Math.Clamp((int)s[2 * i], -8, 7) & 15, u = Math.Clamp((int)s[2 * i + 1], -8, 7) & 15;
            d[i] = (byte)(l | u << 4);
        }

        if ((s.Length & 1) != 0) d[bytes] = (byte)((d[bytes] & 0xF0) | (Math.Clamp((int)s[^1], -8, 7) & 15));
    }

    // 2-bit values (four a byte, the first in the low bits) to 8-bit lanes: each byte widened to a 32-bit lane that becomes
    // its four values, sign-extended by shifts (signed) or masked (unsigned).
    public static void Int2ToByte(ReadOnlySpan<byte> s, Span<byte> d, bool signed)
    {
        int bytes = d.Length / 4, i = 0, w = Vector<byte>.Count, u = Vector<uint>.Count;
        var low = new Vector<int>(0xFF);
        for (; i + w <= bytes; i += w)
        {
            Vector.Widen(new Vector<byte>(s[i..]), out Vector<ushort> lo, out Vector<ushort> hi);
            Vector.Widen(lo, out Vector<uint> a, out Vector<uint> b);
            Vector.Widen(hi, out Vector<uint> c, out Vector<uint> e);
            Vector.AsVectorByte(Four(Vector.AsVectorInt32(a), low, signed)).CopyTo(d[(4 * i)..]);
            Vector.AsVectorByte(Four(Vector.AsVectorInt32(b), low, signed)).CopyTo(d[(4 * i + 4 * u)..]);
            Vector.AsVectorByte(Four(Vector.AsVectorInt32(c), low, signed)).CopyTo(d[(4 * i + 8 * u)..]);
            Vector.AsVectorByte(Four(Vector.AsVectorInt32(e), low, signed)).CopyTo(d[(4 * i + 12 * u)..]);
        }

        for (int at = 4 * i; at < d.Length; at++)
        {
            int v = (s[at / 4] >> (at % 4 * 2)) & 3;
            d[at] = (byte)(signed ? (sbyte)((sbyte)(v << 6) >> 6) : v);
        }

        static Vector<int> Four(Vector<int> x, Vector<int> low, bool signed)
        {
            Vector<int> V(int k) => signed ? Vector.ShiftRightArithmetic(Vector.ShiftLeft(x, 30 - 2 * k), 30) & low : Vector.ShiftRightLogical(x, 2 * k) & new Vector<int>(3);
            return V(0) | Vector.ShiftLeft(V(1), 8) | Vector.ShiftLeft(V(2), 16) | Vector.ShiftLeft(V(3), 24);
        }
    }

    // 8-bit lanes held to the 2-bit range packed four a byte.
    public static void ByteToInt2(ReadOnlySpan<byte> s, Span<byte> d, bool signed)
    {
        int bytes = s.Length / 4, i = 0, w = Vector<byte>.Count, u = Vector<uint>.Count;
        for (; i + w <= bytes; i += w)
        {
            Vector.Narrow(Vector.Narrow(Join(s, 4 * i, signed), Join(s, 4 * i + 4 * u, signed)), Vector.Narrow(Join(s, 4 * i + 8 * u, signed), Join(s, 4 * i + 12 * u, signed))).CopyTo(d[i..]);
        }

        for (int at = 4 * i; at < s.Length; at++)
        {
            int v = signed ? Math.Clamp((int)(sbyte)s[at], -2, 1) & 3 : Math.Min((int)s[at], 3);
            int shift = at % 4 * 2;
            d[at / 4] = (byte)((d[at / 4] & ~(3 << shift)) | v << shift);
        }

        static Vector<uint> Join(ReadOnlySpan<byte> s, int at, bool signed)
        {
            var v = new Vector<byte>(s[at..]);
            v = signed ? Vector.AsVectorByte(Vector.Min(Vector.Max(Vector.AsVectorSByte(v), new Vector<sbyte>(-2)), new Vector<sbyte>(1)))
                : Vector.Min(v, new Vector<byte>(3));
            var x = Vector.AsVectorUInt32(v);
            var three = new Vector<uint>(3);
            return (x & three) | (((x >> 8) & three) << 2) | (((x >> 16) & three) << 4) | (((x >> 24) & three) << 6);
        }
    }

    // ---------------------------------------------------------------- the range of stored integers

    // Across the team (chunks of 64K values, each scanned on vectors), then the chunks' results joined.
    public static unsafe (long Min, long Max) RangeSigned<T>(ReadOnlySpan<T> s) where T : unmanaged
    {
        const int Size = 1 << 16;
        if (s.Length <= Size) return RangeOf(s);
        int chunks = (s.Length + Size - 1) / Size;
        Span<long> lows = chunks <= 512 ? stackalloc long[chunks] : new long[chunks];
        Span<long> highs = chunks <= 512 ? stackalloc long[chunks] : new long[chunks];
        fixed (T* p = s)
        fixed (long* lo = lows)
        fixed (long* hi = highs)
        {
            Team.For(s.Length, Size, (long)s.Length * 4, new RangeBody<T>(p, lo, hi, Size));
        }

        long min = long.MaxValue, max = long.MinValue;
        for (int c = 0; c < chunks; c++)
        {
            min = Math.Min(min, lows[c]);
            max = Math.Max(max, highs[c]);
        }

        return (min, max);
    }

    private readonly unsafe struct RangeBody<T>(T* values, long* lows, long* highs, int size) : Team.IBody where T : unmanaged
    {
        public void Run(int first, int last)
        {
            for (int at = first; at < last; at += size)
            {
                var (lo, hi) = RangeOf(new ReadOnlySpan<T>(values + at, Math.Min(size, last - at)));
                lows[at / size] = lo;
                highs[at / size] = hi;
            }
        }
    }

    private static (long Min, long Max) RangeOf<T>(ReadOnlySpan<T> s) where T : unmanaged
    {
        if (s.IsEmpty) return (0, 0);
        int i = 0, w = Vector<T>.Count;
        long lo = long.MaxValue, hi = long.MinValue;
        if (s.Length >= w)
        {
            // Two chains each of minimum and maximum by unchecked loads (item 6j), joined lane by lane (integers: exact).
            ref T r = ref MemoryMarshal.GetReference(s);
            Vector<T> mn = Vector.LoadUnsafe(ref r), mx = mn;
            i = w;
            if (s.Length >= 2 * w)
            {
                Vector<T> mn1 = Vector.LoadUnsafe(ref r, (nuint)w), mx1 = mn1;
                for (i = 2 * w; i + 2 * w <= s.Length; i += 2 * w)
                {
                    var v0 = Vector.LoadUnsafe(ref r, (nuint)i);
                    var v1 = Vector.LoadUnsafe(ref r, (nuint)(i + w));
                    mn = Vector.Min(mn, v0); mx = Vector.Max(mx, v0);
                    mn1 = Vector.Min(mn1, v1); mx1 = Vector.Max(mx1, v1);
                }

                mn = Vector.Min(mn, mn1);
                mx = Vector.Max(mx, mx1);
            }

            for (; i + w <= s.Length; i += w)
            {
                var v = Vector.LoadUnsafe(ref r, (nuint)i);
                mn = Vector.Min(mn, v);
                mx = Vector.Max(mx, v);
            }

            for (int l = 0; l < w; l++)
            {
                lo = Math.Min(lo, Widths.AsLong(mn[l]));
                hi = Math.Max(hi, Widths.AsLong(mx[l]));
            }
        }

        for (; i < s.Length; i++)
        {
            long v = Widths.AsLong(s[i]);
            lo = Math.Min(lo, v);
            hi = Math.Max(hi, v);
        }

        return (lo, hi);
    }
}

// ---------------------------------------------------------------------------------------------- any lane to any lane
// Load<TS, TL>: stored values into lanes, exact (a narrowing load is asked for only when the measured range fits).
// Store<TL, TS>: lanes into stored values, rounded to nearest even and held to the target's range. Every pair of
// integer widths goes through the hardware's widen and narrow in registers; the type tests are constants to the JIT.

internal static class Widths
{
    private static int Size<T>() where T : unmanaged => Unsafe.SizeOf<T>();

    private static bool IsSigned<T>() => typeof(T) == typeof(sbyte) || typeof(T) == typeof(short) || typeof(T) == typeof(int) || typeof(T) == typeof(long);

    private static bool IsInteger<T>() => typeof(T) != typeof(float) && typeof(T) != typeof(double);

    public static long AsLong<T>(T v) where T : unmanaged => Size<T>() switch
    {
        1 => typeof(T) == typeof(sbyte) ? Unsafe.As<T, sbyte>(ref v) : Unsafe.As<T, byte>(ref v),
        2 => typeof(T) == typeof(short) ? Unsafe.As<T, short>(ref v) : Unsafe.As<T, ushort>(ref v),
        4 => typeof(T) == typeof(int) ? Unsafe.As<T, int>(ref v) : Unsafe.As<T, uint>(ref v),
        _ => typeof(T) == typeof(long) ? Unsafe.As<T, long>(ref v) : (long)Math.Min(Unsafe.As<T, ulong>(ref v), long.MaxValue),
    };

    /// <summary>Orders two lane values by Adad's order (Routing.Precedes without the columns): floats by value, a NaN above
    /// every number and two NaNs equal, −0 equal to +0; unsigned 64-bit values as unsigned. Negative, zero or positive.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Compare<T>(T a, T b) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            // In the lane's own width (no widening a value): a scan's comparison inlined where it runs.
            float x = Unsafe.As<T, float>(ref a), y = Unsafe.As<T, float>(ref b);
            return x > y ? 1 : x < y ? -1 : x == y ? 0 : float.IsNaN(x) ? (float.IsNaN(y) ? 0 : 1) : -1;
        }

        if (typeof(T) == typeof(double))
        {
            double x = Unsafe.As<T, double>(ref a), y = Unsafe.As<T, double>(ref b);
            return x > y ? 1 : x < y ? -1 : x == y ? 0 : double.IsNaN(x) ? (double.IsNaN(y) ? 0 : 1) : -1;
        }

        if (typeof(T) == typeof(ulong)) return Unsafe.As<T, ulong>(ref a).CompareTo(Unsafe.As<T, ulong>(ref b));
        return AsLong(a).CompareTo(AsLong(b));
    }

    public static T FromLong<T>(long v) where T : unmanaged
    {
        if (typeof(T) == typeof(sbyte)) { var x = (sbyte)v; return Unsafe.As<sbyte, T>(ref x); }
        if (typeof(T) == typeof(byte)) { var x = (byte)v; return Unsafe.As<byte, T>(ref x); }
        if (typeof(T) == typeof(short)) { var x = (short)v; return Unsafe.As<short, T>(ref x); }
        if (typeof(T) == typeof(ushort)) { var x = (ushort)v; return Unsafe.As<ushort, T>(ref x); }
        if (typeof(T) == typeof(int)) { var x = (int)v; return Unsafe.As<int, T>(ref x); }
        if (typeof(T) == typeof(uint)) { var x = (uint)v; return Unsafe.As<uint, T>(ref x); }
        if (typeof(T) == typeof(long)) return Unsafe.As<long, T>(ref v);
        if (typeof(T) == typeof(ulong)) { var x = (ulong)v; return Unsafe.As<ulong, T>(ref x); }
        throw new NotSupportedException(typeof(T).Name);
    }

    private static Span<TTo> Re<TFrom, TTo>(Span<TFrom> s) where TFrom : unmanaged where TTo : unmanaged => MemoryMarshal.Cast<TFrom, TTo>(s);

    private static ReadOnlySpan<TTo> Re<TFrom, TTo>(ReadOnlySpan<TFrom> s) where TFrom : unmanaged where TTo : unmanaged => MemoryMarshal.Cast<TFrom, TTo>(s);

    public static void Load<TS, TL>(ReadOnlySpan<TS> s, Span<TL> d) where TS : unmanaged where TL : unmanaged
    {
        if (typeof(TS) == typeof(TL)) { Re<TS, TL>(s).CopyTo(d); return; }
        if (typeof(TL) == typeof(float))
        {
            var f = Re<TL, float>(d);
            if (typeof(TS) == typeof(sbyte)) Cvt.ToFloat(Re<TS, sbyte>(s), f);
            else if (typeof(TS) == typeof(byte)) Cvt.ToFloat(Re<TS, byte>(s), f);
            else if (typeof(TS) == typeof(short)) Cvt.ToFloat(Re<TS, short>(s), f);
            else if (typeof(TS) == typeof(ushort)) Cvt.ToFloat(Re<TS, ushort>(s), f);
            else if (typeof(TS) == typeof(int)) Cvt.ToFloat(Re<TS, int>(s), f);
            else if (typeof(TS) == typeof(uint)) Cvt.ToFloat(Re<TS, uint>(s), f);
            else if (typeof(TS) == typeof(long)) Cvt.ToFloat(Re<TS, long>(s), f);
            else if (typeof(TS) == typeof(ulong)) Cvt.ToFloat(Re<TS, ulong>(s), f);
            else if (typeof(TS) == typeof(double)) Cvt.FromDouble(Re<TS, double>(s), f);
            else throw new NotSupportedException($"{typeof(TS).Name} → float");
            return;
        }

        if (typeof(TL) == typeof(double))
        {
            var f = Re<TL, double>(d);
            if (typeof(TS) == typeof(float)) Cvt.ToDouble(Re<TS, float>(s), f);
            else if (typeof(TS) == typeof(int)) Cvt.ToDouble(Re<TS, int>(s), f);
            else if (typeof(TS) == typeof(uint)) Cvt.ToDouble(Re<TS, uint>(s), f);
            else if (typeof(TS) == typeof(long)) Cvt.ToDouble(Re<TS, long>(s), f);
            else if (typeof(TS) == typeof(ulong)) Cvt.ToDouble(Re<TS, ulong>(s), f);
            else if (IsSigned<TS>()) Through<TS, long, double>(s, f);
            else Through<TS, ulong, double>(s, f);
            return;
        }

        if (IsInteger<TS>() && IsInteger<TL>())
        {
            if (Size<TL>() > Size<TS>()) Widen(s, d);
            else Narrow(s, d);
            return;
        }

        throw new NotSupportedException($"{typeof(TS).Name} → {typeof(TL).Name}");
    }

    public static void Store<TL, TS>(ReadOnlySpan<TL> s, Span<TS> d) where TL : unmanaged where TS : unmanaged
    {
        if (typeof(TL) == typeof(TS)) { Re<TL, TS>(s).CopyTo(d); return; }
        if (typeof(TL) == typeof(float))
        {
            var f = Re<TL, float>(s);
            if (typeof(TS) == typeof(sbyte)) Cvt.FromFloat(f, Re<TS, sbyte>(d));
            else if (typeof(TS) == typeof(byte)) Cvt.FromFloat(f, Re<TS, byte>(d));
            else if (typeof(TS) == typeof(short)) Cvt.FromFloat(f, Re<TS, short>(d));
            else if (typeof(TS) == typeof(ushort)) Cvt.FromFloat(f, Re<TS, ushort>(d));
            else if (typeof(TS) == typeof(int)) Cvt.FromFloat(f, Re<TS, int>(d));
            else if (typeof(TS) == typeof(uint)) Cvt.FromFloat(f, Re<TS, uint>(d));
            else if (typeof(TS) == typeof(long)) Cvt.FromFloat(f, Re<TS, long>(d));
            else if (typeof(TS) == typeof(ulong)) Cvt.FromFloat(f, Re<TS, ulong>(d));
            else if (typeof(TS) == typeof(double)) Cvt.ToDouble(f, Re<TS, double>(d));
            else throw new NotSupportedException($"float → {typeof(TS).Name}");
            return;
        }

        if (typeof(TL) == typeof(double))
        {
            var f = Re<TL, double>(s);
            if (typeof(TS) == typeof(float)) Cvt.FromDouble(f, Re<TS, float>(d));
            else if (typeof(TS) == typeof(long)) Cvt.FromDouble(f, Re<TS, long>(d));
            else if (typeof(TS) == typeof(ulong)) Cvt.FromDouble(f, Re<TS, ulong>(d));
            else if (IsSigned<TS>()) Through<double, long, TS>(f, d, IntMin<TS>(), IntMax<TS>());
            else Through<double, ulong, TS>(f, d, 0, IntMax<TS>());
            return;
        }

        if (IsInteger<TL>() && IsInteger<TS>())
        {
            if (Size<TS>() > Size<TL>()) Widen(s, d);
            else Narrow(s, d);
            return;
        }

        throw new NotSupportedException($"{typeof(TL).Name} → {typeof(TS).Name}");
    }

    private static double IntMin<T>() => typeof(T) == typeof(sbyte) ? sbyte.MinValue : typeof(T) == typeof(short) ? short.MinValue : typeof(T) == typeof(int) ? int.MinValue : 0;

    private static double IntMax<T>() => typeof(T) == typeof(sbyte) ? sbyte.MaxValue : typeof(T) == typeof(byte) ? byte.MaxValue
        : typeof(T) == typeof(short) ? short.MaxValue : typeof(T) == typeof(ushort) ? ushort.MaxValue : typeof(T) == typeof(int) ? int.MaxValue : uint.MaxValue;

    // A pair with no direct path: through a 64-bit integer, a chunk on the stack at a time.
    private static void Through<TS, TM, TL>(ReadOnlySpan<TS> s, Span<TL> d, double low = 0, double high = 0) where TS : unmanaged where TM : unmanaged where TL : unmanaged
    {
        const int Chunk = 1024;
        Span<TM> t = stackalloc TM[Chunk];
        for (int at = 0; at < s.Length; at += Chunk)
        {
            int n = Math.Min(Chunk, s.Length - at);
            if (typeof(TS) == typeof(double))
            {
                if (typeof(TM) == typeof(long)) Cvt.FromDouble(Re<TS, double>(s.Slice(at, n)), Re<TM, long>(t[..n]), low, high);
                else Cvt.FromDouble(Re<TS, double>(s.Slice(at, n)), Re<TM, ulong>(t[..n]), high);
                Narrow(t[..n], d.Slice(at, n));
            }
            else
            {
                Widen(s.Slice(at, n), t[..n]);
                if (typeof(TM) == typeof(long)) Cvt.ToDouble(Re<TM, long>(t[..n]), Re<TL, double>(d.Slice(at, n)));
                else Cvt.ToDouble(Re<TM, ulong>(t[..n]), Re<TL, double>(d.Slice(at, n)));
            }
        }
    }

    // ---------------------------------------------------------------- integer widths, in registers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void W<TA, TB>(Vector<TA> v, out Vector<TB> lo, out Vector<TB> hi) where TA : unmanaged where TB : unmanaged
    {
        if (typeof(TA) == typeof(sbyte)) { Vector.Widen(Unsafe.BitCast<Vector<TA>, Vector<sbyte>>(v), out var l, out var h); lo = Unsafe.BitCast<Vector<short>, Vector<TB>>(l); hi = Unsafe.BitCast<Vector<short>, Vector<TB>>(h); }
        else if (typeof(TA) == typeof(byte)) { Vector.Widen(Unsafe.BitCast<Vector<TA>, Vector<byte>>(v), out var l, out var h); lo = Unsafe.BitCast<Vector<ushort>, Vector<TB>>(l); hi = Unsafe.BitCast<Vector<ushort>, Vector<TB>>(h); }
        else if (typeof(TA) == typeof(short)) { Vector.Widen(Unsafe.BitCast<Vector<TA>, Vector<short>>(v), out var l, out var h); lo = Unsafe.BitCast<Vector<int>, Vector<TB>>(l); hi = Unsafe.BitCast<Vector<int>, Vector<TB>>(h); }
        else if (typeof(TA) == typeof(ushort)) { Vector.Widen(Unsafe.BitCast<Vector<TA>, Vector<ushort>>(v), out var l, out var h); lo = Unsafe.BitCast<Vector<uint>, Vector<TB>>(l); hi = Unsafe.BitCast<Vector<uint>, Vector<TB>>(h); }
        else if (typeof(TA) == typeof(int)) { Vector.Widen(Unsafe.BitCast<Vector<TA>, Vector<int>>(v), out var l, out var h); lo = Unsafe.BitCast<Vector<long>, Vector<TB>>(l); hi = Unsafe.BitCast<Vector<long>, Vector<TB>>(h); }
        else if (typeof(TA) == typeof(uint)) { Vector.Widen(Unsafe.BitCast<Vector<TA>, Vector<uint>>(v), out var l, out var h); lo = Unsafe.BitCast<Vector<ulong>, Vector<TB>>(l); hi = Unsafe.BitCast<Vector<ulong>, Vector<TB>>(h); }
        else throw new NotSupportedException(typeof(TA).Name);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<TB> N<TA, TB>(Vector<TA> a, Vector<TA> b) where TA : unmanaged where TB : unmanaged
    {
        if (typeof(TA) == typeof(short)) return Unsafe.BitCast<Vector<sbyte>, Vector<TB>>(Vector.Narrow(Unsafe.BitCast<Vector<TA>, Vector<short>>(a), Unsafe.BitCast<Vector<TA>, Vector<short>>(b)));
        if (typeof(TA) == typeof(ushort)) return Unsafe.BitCast<Vector<byte>, Vector<TB>>(Vector.Narrow(Unsafe.BitCast<Vector<TA>, Vector<ushort>>(a), Unsafe.BitCast<Vector<TA>, Vector<ushort>>(b)));
        if (typeof(TA) == typeof(int)) return Unsafe.BitCast<Vector<short>, Vector<TB>>(Vector.Narrow(Unsafe.BitCast<Vector<TA>, Vector<int>>(a), Unsafe.BitCast<Vector<TA>, Vector<int>>(b)));
        if (typeof(TA) == typeof(uint)) return Unsafe.BitCast<Vector<ushort>, Vector<TB>>(Vector.Narrow(Unsafe.BitCast<Vector<TA>, Vector<uint>>(a), Unsafe.BitCast<Vector<TA>, Vector<uint>>(b)));
        if (typeof(TA) == typeof(long)) return Unsafe.BitCast<Vector<int>, Vector<TB>>(Vector.Narrow(Unsafe.BitCast<Vector<TA>, Vector<long>>(a), Unsafe.BitCast<Vector<TA>, Vector<long>>(b)));
        if (typeof(TA) == typeof(ulong)) return Unsafe.BitCast<Vector<uint>, Vector<TB>>(Vector.Narrow(Unsafe.BitCast<Vector<TA>, Vector<ulong>>(a), Unsafe.BitCast<Vector<TA>, Vector<ulong>>(b)));
        throw new NotSupportedException(typeof(TA).Name);
    }

    // Sign-extending for signed sources, zero-extending for unsigned; the midway containers follow the source's sign.
    public static void Widen<TS, TD>(ReadOnlySpan<TS> s, Span<TD> d) where TS : unmanaged where TD : unmanaged
    {
        int factor = Size<TD>() / Size<TS>();
        bool signed = IsSigned<TS>();
        if (factor == 2) Widen2(s, d);
        else if (factor == 4)
        {
            if (signed) { if (Size<TS>() == 1) Widen4<TS, short, TD>(s, d); else Widen4<TS, int, TD>(s, d); }
            else { if (Size<TS>() == 1) Widen4<TS, ushort, TD>(s, d); else Widen4<TS, uint, TD>(s, d); }
        }
        else if (signed) Widen8<TS, short, int, TD>(s, d);
        else Widen8<TS, ushort, uint, TD>(s, d);
    }

    private static void Widen2<TS, TD>(ReadOnlySpan<TS> s, Span<TD> d) where TS : unmanaged where TD : unmanaged
    {
        int i = 0, w = Vector<TS>.Count, h = w / 2;
        for (; i + w <= s.Length; i += w)
        {
            W(new Vector<TS>(s[i..]), out Vector<TD> lo, out Vector<TD> hi);
            lo.CopyTo(d[i..]);
            hi.CopyTo(d[(i + h)..]);
        }

        for (; i < s.Length; i++) d[i] = FromLong<TD>(AsLong(s[i]));
    }

    private static void Widen4<TS, TM, TD>(ReadOnlySpan<TS> s, Span<TD> d) where TS : unmanaged where TM : unmanaged where TD : unmanaged
    {
        int i = 0, w = Vector<TS>.Count, q = w / 4;
        for (; i + w <= s.Length; i += w)
        {
            W(new Vector<TS>(s[i..]), out Vector<TM> lo, out Vector<TM> hi);
            W(lo, out Vector<TD> a, out Vector<TD> b);
            W(hi, out Vector<TD> c, out Vector<TD> e);
            a.CopyTo(d[i..]);
            b.CopyTo(d[(i + q)..]);
            c.CopyTo(d[(i + 2 * q)..]);
            e.CopyTo(d[(i + 3 * q)..]);
        }

        for (; i < s.Length; i++) d[i] = FromLong<TD>(AsLong(s[i]));
    }

    private static void Widen8<TS, TM1, TM2, TD>(ReadOnlySpan<TS> s, Span<TD> d) where TS : unmanaged where TM1 : unmanaged where TM2 : unmanaged where TD : unmanaged
    {
        int i = 0, w = Vector<TS>.Count, e = w / 8;
        for (; i + w <= s.Length; i += w)
        {
            W(new Vector<TS>(s[i..]), out Vector<TM1> lo, out Vector<TM1> hi);
            W(lo, out Vector<TM2> a, out Vector<TM2> b);
            W(hi, out Vector<TM2> c, out Vector<TM2> f);
            W(a, out Vector<TD> a0, out Vector<TD> a1);
            W(b, out Vector<TD> b0, out Vector<TD> b1);
            W(c, out Vector<TD> c0, out Vector<TD> c1);
            W(f, out Vector<TD> f0, out Vector<TD> f1);
            a0.CopyTo(d[i..]); a1.CopyTo(d[(i + e)..]); b0.CopyTo(d[(i + 2 * e)..]); b1.CopyTo(d[(i + 3 * e)..]);
            c0.CopyTo(d[(i + 4 * e)..]); c1.CopyTo(d[(i + 5 * e)..]); f0.CopyTo(d[(i + 6 * e)..]); f1.CopyTo(d[(i + 7 * e)..]);
        }

        for (; i < s.Length; i++) d[i] = FromLong<TD>(AsLong(s[i]));
    }

    // Held to the target's range at the source's width, then narrowed (truncation of values that fit).
    public static void Narrow<TS, TD>(ReadOnlySpan<TS> s, Span<TD> d) where TS : unmanaged where TD : unmanaged
    {
        int factor = Size<TS>() / Size<TD>();
        bool signed = IsSigned<TS>();
        long low = signed ? -(1L << (8 * Size<TD>() - 1)) : 0;
        long high = signed ? (1L << (8 * Size<TD>() - 1)) - 1 : (1L << (8 * Size<TD>())) - 1;
        Vector<TS> lo = new(FromLong<TS>(low)), hi = new(FromLong<TS>(high));
        if (factor == 2) Narrow2(s, d, lo, hi, low, high);
        else if (factor == 4)
        {
            if (signed) { if (Size<TS>() == 8) Narrow4<TS, int, TD>(s, d, lo, hi, low, high); else Narrow4<TS, short, TD>(s, d, lo, hi, low, high); }
            else { if (Size<TS>() == 8) Narrow4<TS, uint, TD>(s, d, lo, hi, low, high); else Narrow4<TS, ushort, TD>(s, d, lo, hi, low, high); }
        }
        else if (signed) Narrow8<TS, int, short, TD>(s, d, lo, hi, low, high);
        else Narrow8<TS, uint, ushort, TD>(s, d, lo, hi, low, high);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> Hold<T>(ReadOnlySpan<T> s, int at, Vector<T> lo, Vector<T> hi) where T : unmanaged => Vector.Min(Vector.Max(new Vector<T>(s[at..]), lo), hi);

    private static void Narrow2<TS, TD>(ReadOnlySpan<TS> s, Span<TD> d, Vector<TS> lo, Vector<TS> hi, long low, long high) where TS : unmanaged where TD : unmanaged
    {
        int i = 0, w = Vector<TS>.Count;
        for (; i + 2 * w <= s.Length; i += 2 * w) N<TS, TD>(Hold(s, i, lo, hi), Hold(s, i + w, lo, hi)).CopyTo(d[i..]);
        for (; i < s.Length; i++) d[i] = FromLong<TD>(Math.Clamp(AsLong(s[i]), low, high));
    }

    private static void Narrow4<TS, TM, TD>(ReadOnlySpan<TS> s, Span<TD> d, Vector<TS> lo, Vector<TS> hi, long low, long high) where TS : unmanaged where TM : unmanaged where TD : unmanaged
    {
        int i = 0, w = Vector<TS>.Count;
        for (; i + 4 * w <= s.Length; i += 4 * w)
        {
            N<TM, TD>(N<TS, TM>(Hold(s, i, lo, hi), Hold(s, i + w, lo, hi)), N<TS, TM>(Hold(s, i + 2 * w, lo, hi), Hold(s, i + 3 * w, lo, hi))).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = FromLong<TD>(Math.Clamp(AsLong(s[i]), low, high));
    }

    private static void Narrow8<TS, TM1, TM2, TD>(ReadOnlySpan<TS> s, Span<TD> d, Vector<TS> lo, Vector<TS> hi, long low, long high)
        where TS : unmanaged where TM1 : unmanaged where TM2 : unmanaged where TD : unmanaged
    {
        int i = 0, w = Vector<TS>.Count;
        for (; i + 8 * w <= s.Length; i += 8 * w)
        {
            var a = N<TS, TM1>(Hold(s, i, lo, hi), Hold(s, i + w, lo, hi));
            var b = N<TS, TM1>(Hold(s, i + 2 * w, lo, hi), Hold(s, i + 3 * w, lo, hi));
            var c = N<TS, TM1>(Hold(s, i + 4 * w, lo, hi), Hold(s, i + 5 * w, lo, hi));
            var e = N<TS, TM1>(Hold(s, i + 6 * w, lo, hi), Hold(s, i + 7 * w, lo, hi));
            N<TM2, TD>(N<TM1, TM2>(a, b), N<TM1, TM2>(c, e)).CopyTo(d[i..]);
        }

        for (; i < s.Length; i++) d[i] = FromLong<TD>(Math.Clamp(AsLong(s[i]), low, high));
    }
}
