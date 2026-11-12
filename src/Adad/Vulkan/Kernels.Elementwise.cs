// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Element-wise kernels past the formulas on Vulkan: dropout from a seed (Draws: the same 32-bit arithmetic as the CPU),
// the conversion between any two types (Fmt.ConvertPlan: one rounding; the rounding to odd done on the bits, so a device
// that flushes float32 subnormals keeps them).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    // ---------------------------------------------------------------- draws

    // Draws.Mix: h ^= h >> 16; h *= 0x7FEB352D; h ^= h >> 15; h *= 0x846CA68B; h ^= h >> 16.
    private static V Mix(V h)
    {
        h ^= h >> 16;
        h *= (double)0x7FEB352Du;
        h ^= h >> 15;
        h *= (double)0x846CA68Bu;
        return h ^ (h >> 16);
    }

    /// <summary>Draws.Bits(key, idx) (u32 values).</summary>
    internal static V Draw(V key, V idx) => Mix(Mix(idx ^ key) + key);

    // The kept value or the dropped one: x·s (+ r, one fma) where bits ≥ thr, else r (+0 without a residual).
    private static V DropValue(Spv b, K k, V x, V? r, V s, V bits, V thr) =>
        b.Select(bits >= thr, r is null ? x * s : b.Fma(x, s, r.Value), r ?? b.C(k, 0));

    /// <summary>output = residual + x·s where the seed's draw keeps a value, the residual where it does not, over n values of
    /// f in its float lane: a thread per word. Slots: x, residual, output, n, key, threshold, s.</summary>
    public static SpvKernel Dropout(Vulkan vk, Fmt f, Lane lane, bool residual)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"dropout_{f.Name}_{lane.Name}{(residual ? "_res" : "")}", Threads, vk.Controls);
        var px = b.Arg(0);
        var pr = b.Arg(1);
        var po = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var key = b.Arg(4, K.U32);
        var thr = b.Arg(5, K.U32);
        var s = b.Arg(6, k);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            int bytes = f.WordBits / 8;
            var wx = b.Load(wk, b.At(px, gid, bytes));
            var wr = residual ? b.Load(wk, b.At(pr, gid, bytes)) : wx;
            var wo = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var x = Codec.Unpack(b, f, wx, i, lane);
                V? r = residual ? Codec.Unpack(b, f, wr, i, lane) : null;
                var idx = gid * (double)f.Per + i;
                wo = Codec.Pack(b, f, DropValue(b, k, x, r, s, Draw(key, idx), thr), lane, i, wo);
            }

            b.Store(b.At(po, gid, bytes), wo);
        });
        return b.Build();
    }

    /// <summary>Dropout into blocks, in f32. Slots: x, residual, key, threshold, s (f32).</summary>
    public static SpvKernel BlockDropout(Vulkan vk, Fmt f, bool residual) =>
        BlockEncoder(vk, f, $"dropout{(residual ? "_res" : "")}", b =>
        {
            V px = b.Arg(0), pr = b.Arg(1), key = b.Arg(2, K.U32), thr = b.Arg(3, K.U32), s = b.Arg(4, K.F32);
            return (at, _) => DropValue(b, K.F32, Codec.UnpackAt(b, f, px, at, Lane.F32), residual ? Codec.UnpackAt(b, f, pr, at, Lane.F32) : null, s, Draw(key, at), thr);
        });

    // ---------------------------------------------------------------- conversion

    /// <summary>A lane value rounded to odd at float32 — toward zero, the last bit set when anything was dropped — by integer
    /// operations on its bits (a float64's fields; an integer's top 24 bits), so no float32 subnormal is flushed.</summary>
    private static V ToOddF32(Spv b, Lane lane, V v)
    {
        if (lane == Lane.F64)
        {
            var bits = b.Bits(v);
            var sign = b.Convert((bits >> 32) & 0x80000000UL, K.U32);
            var e = b.Convert((bits >> 52) & 0x7FFUL, K.S32);
            var m = bits & 0xFFFFFFFFFFFFFUL;
            var unbiased = e - 1023.0;
            // Normal at float32: the exponent rebased, 23 bits kept, the rest's sticky bit.
            var normal = b.Convert((b.Convert(unbiased + 127.0, K.U64) << 23) | (m >> 29), K.U32) | b.Select((m & 0x1FFFFFFFUL) != 0UL, b.U(1), b.U(0));
            // Subnormal at float32 (and below): the full significand shifted past the exponent's floor, the rest sticky.
            var full = m | b.UL(1UL << 52);
            var shift = b.Convert(b.Min(b.Max(b.S(-126) - unbiased, b.S(0)), b.S(40)) + 29.0, K.U64);
            var t = full.Shr(shift);
            var sticky = (full & (b.UL(1).Shl(shift) - 1.0)) != 0UL;
            var sub = b.Convert(t, K.U32) | b.Select(sticky, b.U(1), b.U(0));
            var tiny = b.Select(m != 0UL, b.U(1), b.U(0));                    // a float64 subnormal: below every float32
            var r = b.Select(unbiased > 127.0, b.U(0x7F7FFFFF), b.Select(unbiased >= -126.0, normal, b.Select(e == 0.0, tiny, sub)));
            var special = b.Select(m != 0UL, b.U(0x7FC00000) | b.Convert(m >> 29, K.U32), b.U(0x7F800000));
            r = b.Select(e == 2047.0, special, r);
            return b.Bitcast(r | sign, K.F32);
        }

        // An integer: its magnitude's top 24 bits, the rest sticky; below 2^24 exact.
        bool signed = lane == Lane.S32 || lane == Lane.S64;
        var w = b.Convert(v, signed ? K.S64 : K.U64);
        V? neg = signed ? w < b.CBits(K.S64, 0) : null;
        var mag = neg is { } ng ? b.Bitcast(b.Select(ng, -w, w), K.U64) : w;
        var hi = b.Convert(mag >> 32, K.U32);
        var lo = b.Convert(mag, K.U32);
        var msb = b.Select(hi != 0UL, b.HighBit(hi) + 32.0, b.HighBit(lo));
        var sh = b.Convert(b.Max(msb - 23.0, b.S(0)), K.U64);
        var top = mag.Shr(sh);
        var rest = (mag & (b.UL(1).Shl(sh) - 1.0)) != 0UL;
        var code = (b.Convert(msb + 127.0, K.U32) << 23) | (b.Convert(top, K.U32) & 0x7FFFFFUL) | b.Select(rest, b.U(1), b.U(0));
        var exact = b.Bits(b.Convert(mag, K.F32));                                 // below 2^24: exact
        var bitsOut = b.Select(mag < b.UL(1UL << 24), exact, code);
        return b.Bitcast(neg is { } n2 ? bitsOut | b.Select(n2, b.U(0x80000000), b.U(0)) : bitsOut, K.F32);
    }

    /// <summary>output (ft) = x (ff) over n values, a thread per output word, each value rounded once (Fmt.ConvertPlan); the
    /// values past n in the last word 0. Slots: x, output, n.</summary>
    public static SpvKernel Convert(Vulkan vk, Fmt ff, Fmt ft)
    {
        var (lane, toOdd) = Fmt.ConvertPlan(ff, ft);
        var nat = ff.IsBlock ? Lane.F32 : ff.Natural;
        var b = new Spv($"convert_{ff.Name}_{ft.Name}", Threads, vk.Controls);
        var px = b.Arg(0);
        var po = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(ft);
        b.If(gid < WordCount(b, ft, n), () =>
        {
            var wo = b.CBits(wk, 0);
            var last = n - 1.0;
            for (int i = 0; i < ft.Per; i++)
            {
                var idx = gid * (double)ft.Per + i;
                var inside = idx < n;
                var v = Codec.UnpackAt(b, ff, px, b.Min(idx, last), nat);
                V val;
                Lane vl;
                if (!ff.IsFloat && !ft.IsFloat)
                {
                    // Integers: widened to the 64-bit lane, held to ft's range there, then stored from s64.
                    var w = b.Convert(v, Kinds.Of(lane));
                    if (lane == Lane.U64)
                    {
                        if (ft.Signed || ft.Bits < 64) w = b.Min(w, b.UL(ft.Signed ? (1UL << (ft.Bits - 1)) - 1 : (1UL << ft.Bits) - 1));
                        w = b.Bitcast(w, K.S64);
                    }
                    else
                    {
                        long low = ft.Signed ? (ft.Bits == 64 ? long.MinValue : -(1L << (ft.Bits - 1))) : 0;
                        long high = ft.Signed ? (ft.Bits == 64 ? long.MaxValue : (1L << (ft.Bits - 1)) - 1) : ft.Bits == 64 ? long.MaxValue : (1L << ft.Bits) - 1;
                        w = b.Min(b.Max(w, b.CBits(K.S64, (ulong)low)), b.CBits(K.S64, (ulong)high));
                    }

                    val = b.Select(inside, w, b.CBits(K.S64, 0));
                    vl = Lane.S64;
                }
                else
                {
                    if (toOdd) { val = ToOddF32(b, nat, v); vl = Lane.F32; }
                    else if (ft.IsFloat && !nat.IsFloat) { vl = ft.Is("float64") ? Lane.F64 : Lane.F32; val = b.Convert(v, Kinds.Of(vl)); }
                    else { val = v; vl = nat; }

                    if (vl.IsFloat) val = b.Select(inside, val, b.C(Kinds.Of(vl), 0));
                }

                wo = Codec.Pack(b, ft, val, vl, i, wo);
            }

            b.Store(b.At(po, gid, ft.WordBits / 8), wo);
        });
        return b.Build();
    }

    /// <summary>x (ff) into the block format ft, each value read as float32 (rounded to nearest where wider: the format
    /// encodes from float32). Slots: x.</summary>
    public static SpvKernel BlockConvert(Vulkan vk, Fmt ff, Fmt ft) =>
        BlockEncoder(vk, ft, $"convert_{ff.Name}", b =>
        {
            var px = b.Arg(0);
            return (at, _) => Codec.UnpackAt(b, ff, px, at, Lane.F32);
        });
}
