// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// One value out of a stored word into a lane and back, as SPIR-V: the same steps as CUDA's codec (Cuda/Ptx.cs Codec),
// over the same device-neutral description (Abstraction/Formats.cs) — a float format's fields to a float32's, an integer
// format's bits sign- or zero-extended, a block format's value from its block's scales and codes; back, rounded to nearest
// even and held to the type's range. Memory is read in aligned 32-bit words (a byte or a half taken out of its word), so
// no 8- or 16-bit storage access is asked of the device.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static class Codec
{
    /// <summary>The kind of a format's words: u64 for the 64-bit formats, u32 for the others.</summary>
    public static K WordKind(Fmt f) => f.WordBits == 64 ? K.U64 : K.U32;

    // ---------------------------------------------------------------- a value out of a word

    /// <summary>Value <paramref name="i"/> of <paramref name="word"/> as a lane value.</summary>
    public static V Unpack(Spv b, Fmt f, V word, int i, Lane lane) => Unpack(b, f, word, b.U(i * f.Bits), lane);

    /// <summary>The value whose bits start at <paramref name="shift"/> (a u32) of <paramref name="word"/>, as a lane value.</summary>
    public static V Unpack(Spv b, Fmt f, V word, V shift, Lane lane)
    {
        var natural = f.IsFloat ? FloatDecode(b, f, word, shift) : IntegerDecode(b, f, word, shift);
        return Convert(b, f.Natural, natural, lane);
    }

    /// <summary>The value at linear index <paramref name="index"/> (a u32) of the array at <paramref name="basePtr"/>, as a lane value.</summary>
    public static V UnpackAt(Spv b, Fmt f, V basePtr, V index, Lane lane)
    {
        if (f.IsBlock) return Convert(b, Lane.F32, BlockRun(b, f, basePtr, index, 1)[0], lane);
        if (f.WordBits == 64) return Convert(b, f.Natural, Natural64(b, f, b.Load(K.U64, b.At(basePtr, index, 8))), lane);
        var bits = index * (double)f.Bits;
        var word = b.Word(basePtr, bits >> 5);
        return Unpack(b, f, word, bits & 31UL, lane);
    }

    /// <summary>`count` consecutive values of a block format from <paramref name="index"/> (a multiple of count: one block,
    /// one sub-block), as lane values.</summary>
    public static V[] UnpackRun(Spv b, Fmt f, V basePtr, V index, int count, Lane lane) =>
        [.. BlockRun(b, f, basePtr, index, count).Select(v => Convert(b, Lane.F32, v, lane))];

    // A 64-bit word as its format's natural lane (the word is the value).
    private static V Natural64(Spv b, Fmt f, V word) => f.IsFloat ? b.Bitcast(word, K.F64) : f.Signed ? b.Bitcast(word, K.S64) : word;

    private static V IntegerDecode(Spv b, Fmt f, V word, V shift)
    {
        if (f.WordBits == 64) return Natural64(b, f, word);
        if (f.Bits == 32) return f.Signed ? b.Bitcast(word, K.S32) : word;
        return f.Signed ? b.Extract(b.Bitcast(word, K.S32), shift, b.U(f.Bits)) : b.Extract(word, shift, b.U(f.Bits));
    }

    // The float value whose bits start at `shift` of the word: float16 by the GLSL unpack (exact), the others by their fields.
    private static V FloatDecode(Spv b, Fmt f, V word, V shift)
    {
        if (f.Is("float64")) return b.Bitcast(word, K.F64);
        if (f.Is("float32")) return b.Bitcast(word, K.F32);
        if (f.Is("float16")) return HalfToFloat(b, b.Extract(word, shift, b.U(16)));
        if (f.Is("bfloat16")) return b.Bitcast(b.Extract(word, shift, b.U(16)) << 16, K.F32);

        // fp8: the code's fields to a float32's.
        bool e4m3 = f.Is("float8_e4m3");
        var c = b.Extract(word, shift, b.U(8));
        var sg = (c & 0x80UL) << 24;
        var e = (c >> (e4m3 ? 3 : 2)) & (e4m3 ? 15UL : 31UL);
        var m = c & (e4m3 ? 7UL : 3UL);
        var norm = (((e + (e4m3 ? 120 : 112)) << 23) | (m << (e4m3 ? 20 : 21))) | sg;
        var fm = b.Convert(m, K.F32) * (e4m3 ? 1.0 / 512 : 1.0 / 65536);
        var sub = b.Bits(fm) | sg;
        var res = b.Select(e == 0, sub, norm);
        var nan = sg | 0x7FC00000UL;
        if (e4m3) res = b.Select((c & 0x7FUL) == 0x7F, nan, res);
        else
        {
            var top = e == 31;
            res = b.Select(top, sg | 0x7F800000UL, res);
            res = b.Select(top & (m != 0), nan, res);
        }

        return b.Bitcast(res, K.F32);
    }

    /// <summary>A float16's bits (the low 16 of a u32) as the float32 of the same value (exact: every float16 is a float32).</summary>
    public static V HalfToFloat(Spv b, V h)
    {
        // sign, exponent, mantissa; a normal's exponent rebased (15 → 127), a subnormal's value m·2^−24 (exact in float32),
        // ±∞ and NaN kept (the mantissa's top bits with them).
        var sign = (h & 0x8000UL) << 16;
        var e = (h >> 10) & 0x1FUL;
        var m = h & 0x3FFUL;
        var normal = ((e + 112) << 23) | (m << 13);
        var special = (m << 13) | 0x7F800000UL | b.Select(m != 0, b.U(0x00400000), b.U(0));
        var sub = b.Bits(b.Convert(m, K.F32) * (1.0 / 16777216));
        var bits = b.Select(e == 0, sub, b.Select(e == 31, special, normal));
        return b.Bitcast(bits | sign, K.F32);
    }

    /// <summary>A float32 as float16 bits (the low 16 of a u32): rounded to nearest even, past the largest to ±∞, NaN quiet.</summary>
    public static V FloatToHalf(Spv b, V x)
    {
        var bits = b.Bits(x);
        var sign = (bits >> 16) & 0x8000UL;
        var abs = bits & 0x7FFFFFFFUL;
        var lsb = (abs >> 13) & 1UL;
        var normal = ((abs + 0xFFFUL + lsb) >> 13) - 0x1C000UL;
        var sub = b.Convert(b.RoundEven(b.Bitcast(abs, K.F32) * 16777216.0), K.U32);
        var code = b.Select(abs < 0x38800000UL, sub, normal);
        code = b.Select(abs >= 0x477FF000UL, b.U(0x7C00), code);
        code = b.Select(abs > 0x7F800000UL, b.U(0x7E00) | ((abs >> 13) & 0x3FFUL), code);
        return code | sign;
    }

    /// <summary>A value of lane <paramref name="from"/> as lane <paramref name="to"/> (exact by the rules that chose the lanes).</summary>
    public static V Convert(Spv b, Lane from, V v, Lane to) => from == to ? v : b.Convert(v, Kinds.Of(to));

    // ---------------------------------------------------------------- block formats

    // The block's bytes and halves, from the aligned words that hold them.
    private static V ByteAt(Spv b, V address)
    {
        var word = b.Load(K.U32, address & ~3UL);
        var shift = b.Convert(address & 3UL, K.U32) * 8.0;
        return b.Extract(word, shift, b.U(8));
    }

    private static V HalfAt(Spv b, V address)
    {
        var word = b.Load(K.U32, address & ~3UL);
        var shift = b.Convert(address & 2UL, K.U32) * 8.0;
        return HalfToFloat(b, b.Extract(word, shift, b.U(16)));
    }

    private static V SignedByte(Spv b, V u) => b.Extract(b.Bitcast(u, K.S32), 0, 8);

    // `count` consecutive values from the linear index of a block-format array, as f32 (CUDA's BlockRun, step for step).
    private static V[] BlockRun(Spv b, Fmt f, V basePtr, V index, int count)
    {
        var blk = index / (double)f.BlockValues;
        var off = index - blk * (double)f.BlockValues;
        var bAddr = b.At(basePtr, blk, f.BlockBytes);
        V Byte(V offset, int at) => ByteAt(b, b.At(bAddr, offset, 1) + (ulong)at);
        V I2F(V s) => b.Convert(s, K.F32);
        var v = new V[count];
        switch (f.Name)
        {
            case "q8_0":
            {
                var d = HalfAt(b, bAddr);
                for (int k = 0; k < count; k++) v[k] = d * I2F(SignedByte(b, Byte(off, 2 + k)));
                return v;
            }

            case "q4_0":
            {
                var d = HalfAt(b, bAddr);
                var byteOff = off & 15UL;
                var hi = off >= 16;
                for (int k = 0; k < count; k++)
                {
                    var by = Byte(byteOff, 2 + k);
                    var q = b.Bitcast(b.Select(hi, by >> 4, by & 15UL), K.S32) - 8;
                    v[k] = d * I2F(q);
                }

                return v;
            }

            case "q4_k":
            {
                var d = HalfAt(b, bAddr);
                var dmin = HalfAt(b, bAddr + 2UL);
                var j = off >> 5;
                var i = off & 31UL;
                var pair = j >> 1;
                var odd = (j & 1UL) != 0;
                var byteOff = (pair << 5) + i;
                var low = j < 4;
                var jp4 = j + 4;
                var jm4 = b.Select(low, j, j - 4);
                var scJ = Byte(j, 4);
                var scJp4 = Byte(jp4, 4);
                var scJm4 = Byte(jm4, 4);
                var lowS = scJ & 63UL;
                var lowM = scJp4 & 63UL;
                var highS = (scJp4 & 15UL) | ((scJm4 >> 6) << 4);
                var highM = (scJp4 >> 4) | ((scJ >> 6) << 4);
                var scale = b.Select(low, lowS, highS);
                var min = b.Select(low, lowM, highM);
                var ds = d * I2F(b.Bitcast(scale, K.S32));
                var dm = dmin * I2F(b.Bitcast(min, K.S32));
                for (int k = 0; k < count; k++)
                {
                    var by = Byte(byteOff, 16 + k);
                    var q = b.Select(odd, by >> 4, by & 15UL);
                    v[k] = ds * I2F(b.Bitcast(q, K.S32)) - dm;
                }

                return v;
            }

            case "q6_k":
            {
                var d = HalfAt(b, bAddr + 208UL);
                var half = off >> 7;
                var rest = off & 127UL;
                var quarter = rest >> 5;
                var l = rest & 31UL;
                var lBase = half << 6;
                var hBase = half << 5;
                var sBase = half << 3;
                var pOdd = (quarter & 1UL) != 0;
                var pHigh = quarter >= 2;
                var qlOff = lBase + l + b.Select(pOdd, b.U(32), b.U(0));
                var qhOff = hBase + l;
                var shiftH = quarter << 1;
                var sIdx = (l >> 4) + sBase + quarter * 2.0;
                var scS = SignedByte(b, Byte(sIdx, 192));
                var ds = d * I2F(scS);
                for (int k = 0; k < count; k++)
                {
                    var qlByte = Byte(qlOff, k);
                    var qn = b.Select(pHigh, qlByte >> 4, qlByte & 15UL);
                    var hbits = (Byte(qhOff, 128 + k).Shr(shiftH) & 3UL) << 4;
                    var q = b.Bitcast(qn | hbits, K.S32) - 32;
                    v[k] = ds * I2F(q);
                }

                return v;
            }

            default: throw new NotSupportedException($"{f.Name}: no block decoder");
        }
    }

    // ---------------------------------------------------------------- a lane value into a word

    /// <summary>Value <paramref name="v"/> (a <paramref name="lane"/> value) into position <paramref name="i"/> of
    /// <paramref name="word"/> (a 64-bit format: the value is the word); returns the new word. wrap: an integer lane's low
    /// bits as they are (modular at the type's width), not held to the range.</summary>
    public static V Pack(Spv b, Fmt f, V v, Lane lane, int i, V word, bool wrap = false) => Pack(b, f, v, lane, b.U(i * f.Bits), word, wrap);

    public static V Pack(Spv b, Fmt f, V v, Lane lane, V shift, V word, bool wrap = false)
    {
        if (f.IsBlock) throw new NotSupportedException($"{f.Name}: a block format is encoded a block at a time, not a value at a time");
        if (f.IsFloat)
        {
            if (f.Is("float64")) return b.Bits(lane.Wide ? v : b.Convert(v, K.F64));
            var x = lane.Wide ? b.Convert(v, K.F32) : v;
            if (f.Is("float32")) return b.Bits(x);
            V code;
            if (f.Is("float16")) code = FloatToHalf(b, x);
            else if (f.Is("bfloat16"))
            {
                var bits = b.Bits(x);
                var rounded = (bits + 0x7FFFUL + ((bits >> 16) & 1UL)) >> 16;
                code = b.Select(b.IsNan(x), (bits >> 16) | 0x40UL, rounded);
            }
            else code = Fp8(b, f, x);

            return b.Insert(word, code, shift, b.U(f.Bits));
        }

        // Integers: the lane value rounded (floats) and held to the type's range, then its low bits into the word.
        if (f.WordBits == 64)
        {
            var nat = Kinds.Of(f.Natural);
            if (lane.IsFloat) return b.Bits(Saturate(b, v, nat));
            return b.Bits(b.Convert(v, nat));
        }

        var nat32 = f.Signed ? K.S32 : K.U32;
        if (wrap && !lane.IsFloat && !lane.Wide) return Insert(b, f, b.Bits(v), shift, word);
        V n;
        if (lane.IsFloat) n = Saturate(b, v, nat32);
        else if (lane.Wide)
        {
            long low = f.Signed ? -(1L << (f.Bits - 1)) : 0, high = f.Signed ? (1L << (f.Bits - 1)) - 1 : (1L << f.Bits) - 1;
            n = b.Convert(b.Min(b.Max(v, b.C(v.Kind, low)), b.C(v.Kind, high)), nat32);
        }
        else n = b.Bitcast(v, nat32);

        if (f.Bits < 32)
        {
            long low = f.Signed ? -(1L << (f.Bits - 1)) : 0, high = f.Signed ? (1L << (f.Bits - 1)) - 1 : (1L << f.Bits) - 1;
            if (f.Signed) n = b.Max(n, b.S(low));
            n = b.Min(n, b.C(n.Kind, high));
        }

        return Insert(b, f, b.Bits(n), shift, word);
    }

    private static V Insert(Spv b, Fmt f, V code, V shift, V word) => f.Bits == 32 ? code : b.Insert(word, code, shift, b.U(f.Bits));

    /// <summary>A float lane value as an integer kind: rounded to nearest even and held to the kind's range, NaN to 0 (PTX's cvt.rni.sat).</summary>
    public static V Saturate(Spv b, V v, K to)
    {
        var r = b.RoundEven(v);
        double lo = to switch { K.S32 => -2147483648.0, K.S64 => -9223372036854775808.0, _ => 0 };
        double hi = to switch { K.S32 => 2147483648.0, K.U32 => 4294967296.0, K.S64 => 9223372036854775808.0, _ => 18446744073709551616.0 };
        ulong max = to switch { K.S32 => int.MaxValue, K.U32 => uint.MaxValue, K.S64 => long.MaxValue, _ => ulong.MaxValue };
        ulong min = to switch { K.S32 => unchecked((ulong)int.MinValue), K.S64 => unchecked((ulong)long.MinValue), _ => 0 };
        var inRange = b.Select(r < lo, b.C(r.Kind, lo), r);
        var t = b.Convert(b.Select(r >= hi, b.C(r.Kind, 0), inRange), to);
        t = b.Select(r >= hi, b.CBits(to, max), t);
        t = b.Select(r < lo, b.CBits(to, min), t);
        return b.Select(b.IsNan(v), b.C(to, 0), t);
    }

    // A float32 as an fp8 code (CUDA's steps): rounded to nearest even, past the largest to the largest (e4m3) or ∞ (e5m2).
    private static V Fp8(Spv b, Fmt f, V x)
    {
        bool e4m3 = f.Is("float8_e4m3");
        var bits = b.Bits(x);
        var sg = (bits >> 24) & 0x80UL;
        var abs = bits & 0x7FFFFFFFUL;
        var lsb = (abs >> (e4m3 ? 20 : 21)) & 1UL;
        var norm = ((abs + (e4m3 ? 0x7FFFFUL : 0xFFFFFUL) + lsb) >> (e4m3 ? 20 : 21)) - (e4m3 ? 960 : 448);
        var sub = b.Convert(b.RoundEven(b.Bitcast(abs, K.F32) * (e4m3 ? 512.0 : 65536.0)), K.U32);
        var code = b.Select(abs < (e4m3 ? 0x3C800000UL : 0x38800000UL), sub, norm);
        code = b.Select(abs >= (e4m3 ? 0x43E80000UL : 0x47700000UL), b.U(e4m3 ? 0x7F : 0x7C), code);
        if (!e4m3) code = b.Select(abs > 0x7F800000UL, b.U(0x7F), code);
        return code | sg;
    }
}
