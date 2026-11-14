// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Block formats: values stored in blocks with shared scales (the GGUF layouts Q8_0, Q4_0, Q4_K, Q6_K) and the 1-bit
// bool, as stored types of the contract. A block type's Load decodes whole blocks into the float lane; its Store
// encodes whole blocks (a partial block re-encodes the block with the new values: lossy, as the type is). The lane is
// f32 (float64 for nothing here); sums take the float lane (the block-scaled integer dot products are the step past
// this). Bits is the nominal bits per value; Bytes(count) gives the storage, blocks whole.
//
//   Q8_0   blocks of 32: an f16 scale d, then 32 int8 q; value = d · q                               (34 bytes a block)
//   Q4_0   blocks of 32: an f16 scale d, then 16 bytes of nibbles (value i in the low nibble of byte i for i < 16, the
//          high nibble of byte i − 16 for i ≥ 16); value = d · (q − 8)                                (18 bytes a block)
//   Q4_K   super-blocks of 256 (8 sub-blocks of 32): f16 d, f16 dmin, 12 bytes of 6-bit sub-scales and sub-mins,
//          128 bytes of nibbles (sub-block pairs share a byte range: the low nibbles are sub-block 2j, the high 2j + 1);
//          value = d·scale_j · q − dmin·min_j                                                         (144 bytes a block)
//   Q6_K   super-blocks of 256 (16 sub-blocks of 16): 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, 16 int8
//          sub-scales, f16 d; value = d · scale_j · (q − 32)                                           (210 bytes a block)
//   bool   1 bit a value (8 a byte, the first value in the low bit): 0 or 1; and / or / not are the formulas mul / max /
//          (1 − x) in its integer lane (u8 lanes, u32 sums).
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Adad.Abstraction;

/// <summary>The block codecs: decode a block's values to float32; encode float32 values into a block.</summary>
internal static class Blocks
{
    private static float Half(ReadOnlySpan<byte> s, int at) => (float)BitConverter.ToHalf(s.Slice(at, 2));
    private static void Half(float v, Span<byte> s, int at) => BitConverter.TryWriteBytes(s.Slice(at, 2), (Half)v);

    // The codecs on 128-bit vectors where the processor has them: the same operations on each value in the same order as
    // the scalar lines they stand for (a code widened to an integer, less its bias, converted, times the scale, less the
    // offset; a value times the inverse scale, rounded or offset and cut, held to the codes' range), so the same floats
    // and the same codes. A block with a value that is not finite keeps the scalar lines (their conversions of NaN and
    // the infinities are theirs).
    private static readonly bool s_vectors = Vector128.IsHardwareAccelerated;

    // Sixteen codes (bytes below 128) as values: (code − bias) · scale − offset into v[at .. at + 16).
    private static void Values16(Vector128<byte> q, int bias, float scale, float offset, Span<float> v, int at)
    {
        var (lo, hi) = Vector128.Widen(q);
        var (a0, a1) = Vector128.Widen(lo);
        var (a2, a3) = Vector128.Widen(hi);
        Vector128<int> b = Vector128.Create(bias);
        Vector128<float> k = Vector128.Create(scale), m = Vector128.Create(offset);
        (Vector128.ConvertToSingle(a0.AsInt32() - b) * k - m).CopyTo(v.Slice(at));
        (Vector128.ConvertToSingle(a1.AsInt32() - b) * k - m).CopyTo(v.Slice(at + 4));
        (Vector128.ConvertToSingle(a2.AsInt32() - b) * k - m).CopyTo(v.Slice(at + 8));
        (Vector128.ConvertToSingle(a3.AsInt32() - b) * k - m).CopyTo(v.Slice(at + 12));
    }

    // Four values' codes: round((x + add) / div) to even, held to [lo, hi] (as floats, then converted: the same integers
    // as converting first and holding after, the values finite).
    private static Vector128<int> Codes4(ReadOnlySpan<float> x, int at, float add, float div, float lo, float hi) =>
        Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(Vector128.Round((Vector128.Create(x.Slice(at, 4)) + Vector128.Create(add)) / Vector128.Create(div)),
            Vector128.Create(lo)), Vector128.Create(hi)));

    private static bool Finite(ReadOnlySpan<float> v)
    {
        foreach (float x in v) if (!float.IsFinite(x)) return false;
        return true;
    }

    // ---- Q8_0
    public static void DecodeQ8_0(ReadOnlySpan<byte> block, Span<float> v)
    {
        float d = Half(block, 0);
        if (s_vectors)
        {
            var dv = Vector128.Create(d);
            for (int i = 0; i < 32; i += 16)
            {
                var (lo, hi) = Vector128.Widen(Vector128.Create(block.Slice(2 + i, 16)).AsSByte());
                var (a0, a1) = Vector128.Widen(lo);
                var (a2, a3) = Vector128.Widen(hi);
                (dv * Vector128.ConvertToSingle(a0)).CopyTo(v.Slice(i));
                (dv * Vector128.ConvertToSingle(a1)).CopyTo(v.Slice(i + 4));
                (dv * Vector128.ConvertToSingle(a2)).CopyTo(v.Slice(i + 8));
                (dv * Vector128.ConvertToSingle(a3)).CopyTo(v.Slice(i + 12));
            }

            return;
        }

        for (int i = 0; i < 32; i++) v[i] = d * (sbyte)block[2 + i];
    }

    public static void EncodeQ8_0(ReadOnlySpan<float> v, Span<byte> block)
    {
        float amax = 0;
        for (int i = 0; i < 32; i++) amax = Math.Max(amax, Math.Abs(v[i]));
        float d = amax / 127, id = d == 0 ? 0 : 1 / d;
        Half(d, block, 0);
        d = Half(block, 0); id = d == 0 ? 0 : 1 / d;
        if (s_vectors && float.IsFinite(amax))
        {
            Vector128<float> idv = Vector128.Create(id), lo = Vector128.Create(-127f), hi = Vector128.Create(127f);
            for (int i = 0; i < 32; i += 16)
            {
                var q0 = Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(Vector128.Round(Vector128.Create(v.Slice(i, 4)) * idv), lo), hi));
                var q1 = Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(Vector128.Round(Vector128.Create(v.Slice(i + 4, 4)) * idv), lo), hi));
                var q2 = Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(Vector128.Round(Vector128.Create(v.Slice(i + 8, 4)) * idv), lo), hi));
                var q3 = Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(Vector128.Round(Vector128.Create(v.Slice(i + 12, 4)) * idv), lo), hi));
                Vector128.Narrow(Vector128.Narrow(q0, q1), Vector128.Narrow(q2, q3)).AsByte().CopyTo(block.Slice(2 + i));
            }

            return;
        }

        for (int i = 0; i < 32; i++) block[2 + i] = (byte)(sbyte)Math.Clamp(MathF.Round(v[i] * id), -127, 127);
    }

    // ---- Q4_0
    public static void DecodeQ4_0(ReadOnlySpan<byte> block, Span<float> v)
    {
        float d = Half(block, 0);
        if (s_vectors)
        {
            var b = Vector128.Create(block.Slice(2, 16));
            Values16(b & Vector128.Create((byte)0xF), 8, d, 0, v, 0);
            Values16(b >> 4, 8, d, 0, v, 16);
            return;
        }

        for (int i = 0; i < 16; i++)
        {
            v[i] = d * ((block[2 + i] & 0xF) - 8);
            v[i + 16] = d * ((block[2 + i] >> 4) - 8);
        }
    }

    public static void EncodeQ4_0(ReadOnlySpan<float> v, Span<byte> block)
    {
        float amax = 0, max = 0;
        for (int i = 0; i < 32; i++) if (Math.Abs(v[i]) > amax) { amax = Math.Abs(v[i]); max = v[i]; }
        float d = max / -8, id = d == 0 ? 0 : 1 / d;
        Half(d, block, 0);
        d = Half(block, 0); id = d == 0 ? 0 : 1 / d;
        if (s_vectors && Finite(v[..32]) && float.IsFinite(id))
        {
            // Each value times id plus 8.5, cut toward zero, held to [0, 15]: the low nibbles from the first 16, the high
            // from the last.
            Vector128<float> idv = Vector128.Create(id), half = Vector128.Create(8.5f);
            Vector128<int> zero = Vector128<int>.Zero, top = Vector128.Create(15);
            Vector128<int> Code(ReadOnlySpan<float> x, int at) =>
                Vector128.Min(Vector128.Max(Vector128.ConvertToInt32(Vector128.Create(x.Slice(at, 4)) * idv + half), zero), top);
            var low = Vector128.Narrow(Vector128.Narrow(Code(v, 0), Code(v, 4)), Vector128.Narrow(Code(v, 8), Code(v, 12))).AsByte();
            var high = Vector128.Narrow(Vector128.Narrow(Code(v, 16), Code(v, 20)), Vector128.Narrow(Code(v, 24), Code(v, 28))).AsByte();
            (low | (high << 4)).CopyTo(block.Slice(2));
            return;
        }

        for (int i = 0; i < 16; i++)
        {
            int lo = Math.Clamp((int)(v[i] * id + 8.5f), 0, 15), hi = Math.Clamp((int)(v[i + 16] * id + 8.5f), 0, 15);
            block[2 + i] = (byte)(lo | hi << 4);
        }
    }

    // ---- Q4_K (256 values: d, dmin, 12 bytes of scales/mins, 128 bytes of nibbles)
    private static void ScaleMin(ReadOnlySpan<byte> sc, int j, out int scale, out int min)
    {
        if (j < 4) { scale = sc[j] & 63; min = sc[j + 4] & 63; }
        else { scale = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); min = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    }

    public static void DecodeQ4_K(ReadOnlySpan<byte> block, Span<float> v)
    {
        float d = Half(block, 0), dmin = Half(block, 2);
        var sc = block.Slice(4, 12);
        var qs = block.Slice(16, 128);
        for (int pair = 0; pair < 4; pair++)
        {
            ScaleMin(sc, 2 * pair, out int s0, out int m0);
            ScaleMin(sc, 2 * pair + 1, out int s1, out int m1);
            float d0 = d * s0, dm0 = dmin * m0, d1 = d * s1, dm1 = dmin * m1;
            if (s_vectors)
            {
                for (int i = 0; i < 32; i += 16)
                {
                    var q = Vector128.Create(qs.Slice(pair * 32 + i, 16));
                    Values16(q & Vector128.Create((byte)0xF), 0, d0, dm0, v, pair * 64 + i);
                    Values16(q >> 4, 0, d1, dm1, v, pair * 64 + 32 + i);
                }

                continue;
            }

            for (int i = 0; i < 32; i++)
            {
                byte q = qs[pair * 32 + i];
                v[pair * 64 + i] = d0 * (q & 0xF) - dm0;
                v[pair * 64 + 32 + i] = d1 * (q >> 4) - dm1;
            }
        }
    }

    public static void EncodeQ4_K(ReadOnlySpan<float> v, Span<byte> block)
    {
        // Each sub-block of 32: its min and the scale (max − min)/15; the 8 scales and mins quantized to 6 bits against the block's largest.
        Span<float> scales = stackalloc float[8];                         // on the stack: an encode a block, no allocation
        Span<float> mins = stackalloc float[8];
        float maxScale = 0, maxMin = 0;
        for (int j = 0; j < 8; j++)
        {
            float lo = float.MaxValue, hi = float.MinValue;
            for (int i = 0; i < 32; i++) { float x = v[j * 32 + i]; lo = Math.Min(lo, x); hi = Math.Max(hi, x); }
            lo = Math.Min(lo, 0);
            scales[j] = (hi - lo) / 15; mins[j] = -lo;
            maxScale = Math.Max(maxScale, scales[j]); maxMin = Math.Max(maxMin, mins[j]);
        }

        float d = maxScale / 63, dmin = maxMin / 63;
        Half(d, block, 0); Half(dmin, block, 2);
        d = Half(block, 0); dmin = Half(block, 2);
        var sc = block.Slice(4, 12);
        sc.Clear();
        Span<int> ls = stackalloc int[8];
        Span<int> lm = stackalloc int[8];
        for (int j = 0; j < 8; j++)
        {
            ls[j] = d == 0 ? 0 : Math.Clamp((int)MathF.Round(scales[j] / d), 0, 63);
            lm[j] = dmin == 0 ? 0 : Math.Clamp((int)MathF.Round(mins[j] / dmin), 0, 63);
        }

        for (int j = 0; j < 8; j++)
        {
            if (j < 4) { sc[j] = (byte)ls[j]; sc[j + 4] = (byte)lm[j]; }
            else
            {
                sc[j + 4] = (byte)((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
                sc[j - 4] |= (byte)((ls[j] >> 4) << 6);
                sc[j] |= (byte)((lm[j] >> 4) << 6);
            }
        }

        var qs = block.Slice(16, 128);
        if (s_vectors && Finite(v[..256]) && float.IsFinite(d) && float.IsFinite(dmin))
        {
            // A pair of sub-blocks a byte range: sub-block 2·pair's codes the low nibbles, 2·pair + 1's the high.
            for (int pair = 0; pair < 4; pair++)
            {
                float d0 = d * ls[2 * pair], m0 = dmin * lm[2 * pair], d1 = d * ls[2 * pair + 1], m1 = dmin * lm[2 * pair + 1];
                for (int i = 0; i < 32; i += 16)
                {
                    int a = 2 * pair * 32 + i, b = a + 32;
                    var low = d0 == 0 ? Vector128<byte>.Zero : Vector128.Narrow(Vector128.Narrow(Codes4(v, a, m0, d0, 0, 15), Codes4(v, a + 4, m0, d0, 0, 15)),
                        Vector128.Narrow(Codes4(v, a + 8, m0, d0, 0, 15), Codes4(v, a + 12, m0, d0, 0, 15))).AsByte();
                    var high = d1 == 0 ? Vector128<byte>.Zero : Vector128.Narrow(Vector128.Narrow(Codes4(v, b, m1, d1, 0, 15), Codes4(v, b + 4, m1, d1, 0, 15)),
                        Vector128.Narrow(Codes4(v, b + 8, m1, d1, 0, 15), Codes4(v, b + 12, m1, d1, 0, 15))).AsByte();
                    (low | (high << 4)).CopyTo(qs.Slice(pair * 32 + i));
                }
            }

            return;
        }

        for (int j = 0; j < 8; j++)
        {
            float dj = d * ls[j], mj = dmin * lm[j];
            for (int i = 0; i < 32; i++)
            {
                int q = dj == 0 ? 0 : Math.Clamp((int)MathF.Round((v[j * 32 + i] + mj) / dj), 0, 15);
                int pair = j / 2, at = pair * 32 + i;
                if (j % 2 == 0) qs[at] = (byte)((qs[at] & 0xF0) | q); else qs[at] = (byte)((qs[at] & 0x0F) | q << 4);
            }
        }
    }

    // ---- Q6_K (256 values: 128 bytes ql, 64 bytes qh, 16 int8 scales, f16 d)
    public static void DecodeQ6_K(ReadOnlySpan<byte> block, Span<float> v)
    {
        var ql = block.Slice(0, 128);
        var qh = block.Slice(128, 64);
        var sc = block.Slice(192, 16);
        float d = Half(block, 208);
        for (int half = 0; half < 2; half++)
        {
            int lBase = half * 64, hBase = half * 32, sBase = half * 8, vBase = half * 128;
            if (s_vectors)
            {
                var three = Vector128.Create((byte)3);
                var low = Vector128.Create((byte)0xF);
                for (int l = 0; l < 32; l += 16)
                {
                    int is0 = l / 16;
                    var a = Vector128.Create(ql.Slice(lBase + l, 16));
                    var b = Vector128.Create(ql.Slice(lBase + l + 32, 16));
                    var h = Vector128.Create(qh.Slice(hBase + l, 16));
                    Values16((a & low) | ((h & three) << 4), 32, d * (sbyte)sc[sBase + is0], 0, v, vBase + l);
                    Values16((b & low) | (((h >> 2) & three) << 4), 32, d * (sbyte)sc[sBase + is0 + 2], 0, v, vBase + l + 32);
                    Values16((a >> 4) | (((h >> 4) & three) << 4), 32, d * (sbyte)sc[sBase + is0 + 4], 0, v, vBase + l + 64);
                    Values16((b >> 4) | (((h >> 6) & three) << 4), 32, d * (sbyte)sc[sBase + is0 + 6], 0, v, vBase + l + 96);
                }

                continue;
            }

            for (int l = 0; l < 32; l++)
            {
                int is0 = l / 16;
                int q1 = ((ql[lBase + l] & 0xF) | (((qh[hBase + l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[lBase + l + 32] & 0xF) | (((qh[hBase + l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[lBase + l] >> 4) | (((qh[hBase + l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[lBase + l + 32] >> 4) | (((qh[hBase + l] >> 6) & 3) << 4)) - 32;
                v[vBase + l] = d * (sbyte)sc[sBase + is0] * q1;
                v[vBase + l + 32] = d * (sbyte)sc[sBase + is0 + 2] * q2;
                v[vBase + l + 64] = d * (sbyte)sc[sBase + is0 + 4] * q3;
                v[vBase + l + 96] = d * (sbyte)sc[sBase + is0 + 6] * q4;
            }
        }
    }

    public static void EncodeQ6_K(ReadOnlySpan<float> v, Span<byte> block)
    {
        // 16 sub-blocks of 16: each scale = max|x| / 31 (6-bit values −32..31), the scales as int8 against the block's largest.
        Span<float> scales = stackalloc float[16];                        // on the stack: an encode a block, no allocation
        float maxScale = 0;
        for (int j = 0; j < 16; j++)
        {
            float amax = 0;
            for (int i = 0; i < 16; i++) amax = Math.Max(amax, Math.Abs(v[j * 16 + i]));
            scales[j] = amax / 31;
            maxScale = Math.Max(maxScale, scales[j]);
        }

        float d = maxScale / 127;
        Half(d, block, 208);
        d = Half(block, 208);
        var sc = block.Slice(192, 16);
        Span<int> ls = stackalloc int[16];
        for (int j = 0; j < 16; j++) { ls[j] = d == 0 ? 0 : Math.Clamp((int)MathF.Round(scales[j] / d), -127, 127); sc[j] = (byte)(sbyte)ls[j]; }
        Span<int> q = stackalloc int[256];
        bool vectors = s_vectors && Finite(v[..256]) && float.IsFinite(d);
        for (int j = 0; j < 16; j++)
        {
            float dj = d * ls[j];
            if (vectors)
            {
                var bias = Vector128.Create(32);
                for (int i = 0; i < 16; i += 4)
                    (dj == 0 ? bias : Codes4(v, j * 16 + i, 0, dj, -32, 31) + bias).CopyTo(q.Slice(j * 16 + i));
                continue;
            }

            for (int i = 0; i < 16; i++) q[j * 16 + i] = (dj == 0 ? 0 : Math.Clamp((int)MathF.Round(v[j * 16 + i] / dj), -32, 31)) + 32;
        }

        var ql = block.Slice(0, 128);
        var qh = block.Slice(128, 64);
        ql.Clear(); qh.Clear();
        for (int half = 0; half < 2; half++)
        {
            int lBase = half * 64, hBase = half * 32, vBase = half * 128;
            for (int l = 0; l < 32; l++)
            {
                int q1 = q[vBase + l], q2 = q[vBase + l + 32], q3 = q[vBase + l + 64], q4 = q[vBase + l + 96];
                ql[lBase + l] = (byte)((q1 & 0xF) | ((q3 & 0xF) << 4));
                ql[lBase + l + 32] = (byte)((q2 & 0xF) | ((q4 & 0xF) << 4));
                qh[hBase + l] = (byte)(((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6));
            }
        }
    }

    /// <summary>values first .. first + count of a block-formatted array as float32 (the containing blocks decoded).</summary>
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l, int blockValues, int blockBytes, Action<ReadOnlySpan<byte>, Span<float>> decode)
    {
        Span<float> block = stackalloc float[256];
        int at = 0;
        while (at < l.Length)
        {
            int index = first + at, b = index / blockValues, offset = index % blockValues;
            decode(s.Slice(b * blockBytes, blockBytes), block[..blockValues]);
            int n = Math.Min(blockValues - offset, l.Length - at);
            block.Slice(offset, n).CopyTo(l.Slice(at, n));
            at += n;
        }
    }

    /// <summary>float32 values into the blocks at first .. first + count (a partial block re-encoded with its other values kept:
    /// the block's scale then covers the kept values too, so an output of a block format goes into a buffer whose blocks it
    /// covers whole or whose other values belong with it — not one holding stale values of another scale).</summary>
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first, int blockValues, int blockBytes, Action<ReadOnlySpan<byte>, Span<float>> decode, Action<ReadOnlySpan<float>, Span<byte>> encode)
    {
        Span<float> block = stackalloc float[256];
        int at = 0;
        while (at < l.Length)
        {
            int index = first + at, b = index / blockValues, offset = index % blockValues;
            int n = Math.Min(blockValues - offset, l.Length - at);
            var bytes = s.Slice(b * blockBytes, blockBytes);
            if (n < blockValues) decode(bytes, block[..blockValues]); else block[..blockValues].Clear();
            l.Slice(at, n).CopyTo(block.Slice(offset, n));
            encode(block[..blockValues], bytes);
            at += n;
        }
    }
}

/// <summary>What the block types share: the lane is float32; sums in f32 or f64 by the bound; one block in the type's own float visitor.</summary>
internal static class BlockType<TSelf> where TSelf : struct, IFloatAdad<TSelf>, IAdad<TSelf, float>
{
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (b.Bound <= Exact.Float32 || true) v.Run<TSelf, float, F32>();   // the float lane: values are scaled integers, exact in f32 as decoded
    }
}

public readonly struct Q8_0 : IFloatAdad<Q8_0>, IAdad<Q8_0, float>
{
    public const int BlockValues = 32, BlockBytes = 34;
    public static string Name => "q8_0";
    public static ScaledBlock Scaled => new(BlockValues, BlockBytes, 8, 0);
    public static int Bits => 8;
    public static bool IsFloat => true;
    public static double Min => -65504;
    public static double Max => 65504;
    public static double Epsilon => Math.Pow(2, -7);                 // codes −127 … 127
    public static long Bytes(long values) => (values + BlockValues - 1) / BlockValues * BlockBytes;
    public static int Granularity => BlockValues;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => BlockRange.Of(s, count, BlockValues, BlockBytes, Blocks.DecodeQ8_0);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Blocks.Load(s, first, l, BlockValues, BlockBytes, Blocks.DecodeQ8_0);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Blocks.Store(l, s, first, BlockValues, BlockBytes, Blocks.DecodeQ8_0, Blocks.EncodeQ8_0);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Q8_0, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Q8_0, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => BlockType<Q8_0>.Sum(ref v, in b);
}

public readonly struct Q4_0 : IFloatAdad<Q4_0>, IAdad<Q4_0, float>
{
    public const int BlockValues = 32, BlockBytes = 18;
    public static string Name => "q4_0";
    public static ScaledBlock Scaled => new(BlockValues, BlockBytes, 4, -8);
    public static int Bits => 4;
    public static bool IsFloat => true;
    public static double Min => -65504;
    public static double Max => 65504;
    public static double Epsilon => Math.Pow(2, -3);                 // codes −8 … 7
    public static long Bytes(long values) => (values + BlockValues - 1) / BlockValues * BlockBytes;
    public static int Granularity => BlockValues;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => BlockRange.Of(s, count, BlockValues, BlockBytes, Blocks.DecodeQ4_0);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Blocks.Load(s, first, l, BlockValues, BlockBytes, Blocks.DecodeQ4_0);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Blocks.Store(l, s, first, BlockValues, BlockBytes, Blocks.DecodeQ4_0, Blocks.EncodeQ4_0);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Q4_0, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Q4_0, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => BlockType<Q4_0>.Sum(ref v, in b);
}

public readonly struct Q4_K : IFloatAdad<Q4_K>, IAdad<Q4_K, float>
{
    public const int BlockValues = 256, BlockBytes = 144;
    public static string Name => "q4_k";
    public static int Bits => 4;
    public static bool IsFloat => true;
    public static double Min => -65504;
    public static double Max => 65504;
    public static double Epsilon => Math.Pow(2, -4);                 // codes 0 … 15 above the sub-block's minimum
    public static long Bytes(long values) => (values + BlockValues - 1) / BlockValues * BlockBytes;
    public static int Granularity => BlockValues;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => BlockRange.Of(s, count, BlockValues, BlockBytes, Blocks.DecodeQ4_K);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Blocks.Load(s, first, l, BlockValues, BlockBytes, Blocks.DecodeQ4_K);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Blocks.Store(l, s, first, BlockValues, BlockBytes, Blocks.DecodeQ4_K, Blocks.EncodeQ4_K);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Q4_K, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Q4_K, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => BlockType<Q4_K>.Sum(ref v, in b);
}

public readonly struct Q6_K : IFloatAdad<Q6_K>, IAdad<Q6_K, float>
{
    public const int BlockValues = 256, BlockBytes = 210;
    public static string Name => "q6_k";
    public static int Bits => 6;
    public static bool IsFloat => true;
    public static double Min => -65504;
    public static double Max => 65504;
    public static double Epsilon => Math.Pow(2, -5);                 // codes −32 … 31
    public static long Bytes(long values) => (values + BlockValues - 1) / BlockValues * BlockBytes;
    public static int Granularity => BlockValues;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => BlockRange.Of(s, count, BlockValues, BlockBytes, Blocks.DecodeQ6_K);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Blocks.Load(s, first, l, BlockValues, BlockBytes, Blocks.DecodeQ6_K);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Blocks.Store(l, s, first, BlockValues, BlockBytes, Blocks.DecodeQ6_K, Blocks.EncodeQ6_K);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Q6_K, float, F32>();
    public static void Float<TV>(ref TV v) where TV : struct, IFloatLaneVisitor => v.Run<Q6_K, float, F32>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor => BlockType<Q6_K>.Sum(ref v, in b);
}

internal static class BlockRange
{
    public static (double Min, double Max) Of(ReadOnlySpan<byte> s, int count, int blockValues, int blockBytes, Action<ReadOnlySpan<byte>, Span<float>> decode)
    {
        Span<float> block = stackalloc float[256];
        double lo = double.MaxValue, hi = double.MinValue;
        for (int b = 0; b * blockValues < count; b++)
        {
            decode(s.Slice(b * blockBytes, blockBytes), block[..blockValues]);
            int n = Math.Min(blockValues, count - b * blockValues);
            for (int i = 0; i < n; i++) { lo = Math.Min(lo, block[i]); hi = Math.Max(hi, block[i]); }
        }

        return count == 0 ? (0, 0) : (lo, hi);
    }
}

/// <summary>bool: 1 bit a value, 0 or 1; the integer lanes u8 (own) and u32 (sums).</summary>
public readonly struct Bool : IAdad<Bool, byte>, IAdad<Bool, float>, IAdad<Bool, double>, IAdad<Bool, uint>
{
    public static string Name => "bool";
    public static int Bits => 1;
    public static bool IsFloat => false;
    public static double Min => 0;
    public static double Max => 1;
    public static (double Min, double Max) Range(ReadOnlySpan<byte> s, int count) => (0, 1);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<byte> l) => Packed.Load8(s, first, l, 1, false);
    public static void Store(ReadOnlySpan<byte> l, Span<byte> s, int first) => Packed.Store8(l, s, first, 1, false);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<float> l) => Packed.Load(s, first, l, 1, false);
    public static void Store(ReadOnlySpan<float> l, Span<byte> s, int first) => Packed.Store(l, s, first, 1, false);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<double> l) => Packed.Load(s, first, l, 1, false);
    public static void Store(ReadOnlySpan<double> l, Span<byte> s, int first) => Packed.Store(l, s, first, 1, false);
    public static void Load(ReadOnlySpan<byte> s, int first, Span<uint> l) => Packed.Load(s, first, l, 1, false);
    public static void Store(ReadOnlySpan<uint> l, Span<byte> s, int first) => Packed.Store(l, s, first, 1, false);
    public static void Elementwise<TV>(ref TV v) where TV : struct, ILaneVisitor => v.Run<Bool, byte, U8>();
    public static void Sum<TV>(ref TV v, in Bounds b) where TV : struct, ISumVisitor
    {
        if (b.Bound <= Exact.Float32) v.Run<Bool, float, F32>();
        else if (b.Bound <= uint.MaxValue) v.Run<Bool, uint, U32>();
        else v.Run<Bool, double, F64>();
    }
}
