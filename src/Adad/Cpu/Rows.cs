// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Row operations ("rows"): one formula over each row of a rows × cols array whose result needs the row's statistics
// (a max, a sum, a sum of squares, a dot product): the normalizations and their backward passes, log-softmax and the
// softmax backward passes, weight normalization, rotary embedding, scaled and masked softmax, the fused residual-add +
// normalization. Written once over <TN, TLane, TL> in the type's float lane (f32 for every float narrower than 32 bits,
// f64 for float64): the stored type is what is read and written, the row is computed in the lane.
//
// An operation takes a row of x, optionally a second row (x2: dy, the residual, an additive mask), per-column vectors
// (g: gamma / weight, bt: beta / bias), a per-row scalar (s: weight_norm's g), per-row positions (rope), two scalars
// (a, b) and writes a row of y and optionally a second row (y2: the fused add's sum). The catalogue (RowOps.All) names
// each, gives its float64 reference and its error scale for the gate.
//
// The backward passes of the parameters (dgamma, dbeta) are column sums with the rows' statistics: NormParamsBwd.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

/// <summary>A row operation over lane spans: x, the second row, gamma, beta (empty when the operation has none), the
/// row's scalar, position and index, the two scalars, and the outputs (y2 empty when the operation has none).</summary>
public interface IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
{
    static abstract void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2);
}

/// <summary>Row statistics and element-wise passes on lane vectors.</summary>
internal static class RowMath<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
{
    private static int W => Vector<TLane>.Count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<TLane> At(ref TLane r, int i) => Vector.LoadUnsafe(ref r, (nuint)i);

    public static TLane Max(ReadOnlySpan<TLane> x)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        TLane max = TL.NegativeInfinity;
        if (vec > 0)
        {
            Vector<TLane> m0 = new(max), m1 = m0;
            int i = 0;
            for (; i + 2 * w <= vec; i += 2 * w) { m0 = Vector.Max(m0, At(ref rx, i)); m1 = Vector.Max(m1, At(ref rx, i + w)); }
            for (; i < vec; i += w) m0 = Vector.Max(m0, At(ref rx, i));
            m0 = Vector.Max(m0, m1);
            for (int l = 0; l < w; l++) max = TL.Max(max, m0[l]);
        }

        for (int i = vec; i < x.Length; i++) max = TL.Max(max, x[i]);
        return max;
    }

    public static TLane Sum(ReadOnlySpan<TLane> x)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        Vector<TLane> s0 = Vector<TLane>.Zero, s1 = s0;
        int i = 0;
        for (; i + 2 * w <= vec; i += 2 * w) { s0 += At(ref rx, i); s1 += At(ref rx, i + w); }
        for (; i < vec; i += w) s0 += At(ref rx, i);
        TLane sum = Vector.Sum(s0 + s1);
        for (i = vec; i < x.Length; i++) sum = TL.Add(sum, x[i]);
        return sum;
    }

    /// <summary>Σ (x − c)² (c = 0 for a plain sum of squares).</summary>
    public static TLane SumSq(ReadOnlySpan<TLane> x, TLane c)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        var cv = new Vector<TLane>(c);
        Vector<TLane> s0 = Vector<TLane>.Zero, s1 = s0;
        int i = 0;
        for (; i + 2 * w <= vec; i += 2 * w)
        {
            var d0 = At(ref rx, i) - cv; var d1 = At(ref rx, i + w) - cv;
            s0 = TL.Fma(d0, d0, s0); s1 = TL.Fma(d1, d1, s1);
        }

        for (; i < vec; i += w) { var d0 = At(ref rx, i) - cv; s0 = TL.Fma(d0, d0, s0); }
        TLane sum = Vector.Sum(s0 + s1);
        for (i = vec; i < x.Length; i++) { var d = TL.Sub(x[i], c); sum = TL.Fma(d, d, sum); }
        return sum;
    }

    public static TLane Dot(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> y)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        Vector<TLane> s0 = Vector<TLane>.Zero, s1 = s0;
        int i = 0;
        for (; i + 2 * w <= vec; i += 2 * w) { s0 = TL.Fma(At(ref rx, i), At(ref ry, i), s0); s1 = TL.Fma(At(ref rx, i + w), At(ref ry, i + w), s1); }
        for (; i < vec; i += w) s0 = TL.Fma(At(ref rx, i), At(ref ry, i), s0);
        TLane sum = Vector.Sum(s0 + s1);
        for (i = vec; i < x.Length; i++) sum = TL.Fma(x[i], y[i], sum);
        return sum;
    }

    /// <summary>Σ x·y·(z − c).</summary>
    public static TLane Dot3(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> y, ReadOnlySpan<TLane> z, TLane c)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        ref TLane rz = ref MemoryMarshal.GetReference(z);
        var cv = new Vector<TLane>(c);
        Vector<TLane> s0 = Vector<TLane>.Zero, s1 = s0;
        int i = 0;
        for (; i + 2 * w <= vec; i += 2 * w)
        {
            s0 = TL.Fma(At(ref rx, i) * At(ref ry, i), At(ref rz, i) - cv, s0);
            s1 = TL.Fma(At(ref rx, i + w) * At(ref ry, i + w), At(ref rz, i + w) - cv, s1);
        }

        for (; i < vec; i += w) s0 = TL.Fma(At(ref rx, i) * At(ref ry, i), At(ref rz, i) - cv, s0);
        TLane sum = Vector.Sum(s0 + s1);
        for (i = vec; i < x.Length; i++) sum = TL.Fma(TL.Mul(x[i], y[i]), TL.Sub(z[i], c), sum);
        return sum;
    }

    /// <summary>e = exp(x − m), returning Σ e (e may be x itself).</summary>
    public static TLane ExpSum(ReadOnlySpan<TLane> x, TLane m, Span<TLane> e)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane re = ref MemoryMarshal.GetReference(e);
        var mv = new Vector<TLane>(m);
        Vector<TLane> s0 = Vector<TLane>.Zero, s1 = s0;
        int i = 0;
        for (; i + 2 * w <= vec; i += 2 * w)
        {
            var e0 = TL.Exp(At(ref rx, i) - mv); var e1 = TL.Exp(At(ref rx, i + w) - mv);
            e0.StoreUnsafe(ref re, (nuint)i); e1.StoreUnsafe(ref re, (nuint)(i + w));
            s0 += e0; s1 += e1;
        }

        for (; i < vec; i += w) { var e0 = TL.Exp(At(ref rx, i) - mv); e0.StoreUnsafe(ref re, (nuint)i); s0 += e0; }
        TLane sum = Vector.Sum(s0 + s1);
        for (i = vec; i < x.Length; i++) { var t = TL.Exp(TL.Sub(x[i], m)); e[i] = t; sum = TL.Add(sum, t); }
        return sum;
    }

    /// <summary>y = (x − c)·k.</summary>
    public static void Scale(ReadOnlySpan<TLane> x, TLane c, TLane k, Span<TLane> y)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        Vector<TLane> cv = new(c), kv = new(k);
        int i = 0;
        for (; i < vec; i += w) ((At(ref rx, i) - cv) * kv).StoreUnsafe(ref ry, (nuint)i);
        for (; i < x.Length; i++) y[i] = TL.Mul(TL.Sub(x[i], c), k);
    }

    /// <summary>y = (x − c)·k·g + bt (bt empty: none).</summary>
    public static void Normalize(ReadOnlySpan<TLane> x, TLane c, TLane k, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, Span<TLane> y)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane rg = ref MemoryMarshal.GetReference(g);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        Vector<TLane> cv = new(c), kv = new(k);
        int i = 0;
        if (bt.IsEmpty)
        {
            for (; i < vec; i += w) ((At(ref rx, i) - cv) * kv * At(ref rg, i)).StoreUnsafe(ref ry, (nuint)i);
            for (; i < x.Length; i++) y[i] = TL.Mul(TL.Mul(TL.Sub(x[i], c), k), g[i]);
            return;
        }

        ref TLane rb = ref MemoryMarshal.GetReference(bt);
        for (; i < vec; i += w) TL.Fma((At(ref rx, i) - cv) * kv, At(ref rg, i), At(ref rb, i)).StoreUnsafe(ref ry, (nuint)i);
        for (; i < x.Length; i++) y[i] = TL.Fma(TL.Mul(TL.Sub(x[i], c), k), g[i], bt[i]);
    }

    /// <summary>y = x + x2.</summary>
    public static void Add(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, Span<TLane> y)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane r2 = ref MemoryMarshal.GetReference(x2);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        int i = 0;
        for (; i < vec; i += w) (At(ref rx, i) + At(ref r2, i)).StoreUnsafe(ref ry, (nuint)i);
        for (; i < x.Length; i++) y[i] = TL.Add(x[i], x2[i]);
    }

    /// <summary>y = k·(g·dy − m1 − (x − c)·k2) (the normalization backward passes; m1 = 0, c = 0 for rms).</summary>
    public static void NormBackward(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> dy, ReadOnlySpan<TLane> g, TLane c, TLane k, TLane m1, TLane k2, Span<TLane> y)
    {
        int w = W, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane rd = ref MemoryMarshal.GetReference(dy);
        ref TLane rg = ref MemoryMarshal.GetReference(g);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        Vector<TLane> cv = new(c), kv = new(k), m1v = new(m1), k2v = new(k2);
        int i = 0;
        for (; i < vec; i += w)
        {
            var t = At(ref rg, i) * At(ref rd, i) - m1v - (At(ref rx, i) - cv) * k2v;
            (t * kv).StoreUnsafe(ref ry, (nuint)i);
        }

        for (; i < x.Length; i++) y[i] = TL.Mul(TL.Sub(TL.Sub(TL.Mul(g[i], dy[i]), m1), TL.Mul(TL.Sub(x[i], c), k2)), k);
    }

    public static TLane Log(TLane v) => TL.Log(new Vector<TLane>(v))[0];
}

/// <summary>The row kernels, one struct per operation (the catalogue's references in lane arithmetic).</summary>
internal static class RowKernels
{
    public struct LogSoftmax<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane m = RowMath<TLane, TL>.Max(x);
            TLane sum = RowMath<TLane, TL>.ExpSum(x, m, y);
            TLane ls = TL.Add(m, RowMath<TLane, TL>.Log(sum));
            RowMath<TLane, TL>.Scale(x, ls, TL.From(1), y);
        }
    }

    public struct SoftmaxBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane d = RowMath<TLane, TL>.Dot(x, x2);
            int w = Vector<TLane>.Count, vec = x.Length / w * w, i = 0;
            var dv = new Vector<TLane>(d);
            for (; i < vec; i += w) (new Vector<TLane>(x[i..]) * (new Vector<TLane>(x2[i..]) - dv)).CopyTo(y[i..]);
            for (; i < x.Length; i++) y[i] = TL.Mul(x[i], TL.Sub(x2[i], d));
        }
    }

    public struct LogSoftmaxBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane sd = RowMath<TLane, TL>.Sum(x2);
            int w = Vector<TLane>.Count, vec = x.Length / w * w, i = 0;
            var sv = new Vector<TLane>(sd);
            for (; i < vec; i += w) (new Vector<TLane>(x2[i..]) - TL.Exp(new Vector<TLane>(x[i..])) * sv).CopyTo(y[i..]);
            for (; i < x.Length; i++) y[i] = TL.Sub(x2[i], TL.Mul(TL.Exp(x[i]), sd));
        }
    }

    private static TLane Rstd<TLane, TL>(TLane sumSq, int n, TLane eps) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
        TL.Div(TL.From(1), TL.Sqrt(TL.Add(TL.Div(sumSq, TL.From(n)), eps)));

    public struct RmsNormAffine<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane rstd = Rstd<TLane, TL>(RowMath<TLane, TL>.SumSq(x, TL.Zero), x.Length, a);
            RowMath<TLane, TL>.Normalize(x, TL.Zero, rstd, g, default, y);
        }
    }

    public struct RmsNormBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane rstd = Rstd<TLane, TL>(RowMath<TLane, TL>.SumSq(x, TL.Zero), x.Length, a);
            TLane t = RowMath<TLane, TL>.Dot3(g, x2, x, TL.Zero);                 // Σ g·dy·x
            TLane m = TL.Div(TL.Mul(rstd, t), TL.From(x.Length));                 // mean(g·dy·x̂)
            RowMath<TLane, TL>.NormBackward(x, x2, g, TL.Zero, rstd, TL.Zero, TL.Mul(rstd, m), y);
        }
    }

    public struct LayerNorm<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane mu = TL.Div(RowMath<TLane, TL>.Sum(x), TL.From(x.Length));
            TLane rstd = Rstd<TLane, TL>(RowMath<TLane, TL>.SumSq(x, mu), x.Length, a);
            RowMath<TLane, TL>.Normalize(x, mu, rstd, g, bt, y);
        }
    }

    public struct LayerNormBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane n = TL.From(x.Length);
            TLane mu = TL.Div(RowMath<TLane, TL>.Sum(x), n);
            TLane rstd = Rstd<TLane, TL>(RowMath<TLane, TL>.SumSq(x, mu), x.Length, a);
            TLane m1 = TL.Div(RowMath<TLane, TL>.Dot(g, x2), n);                              // mean(g·dy)
            TLane m2 = TL.Div(TL.Mul(rstd, RowMath<TLane, TL>.Dot3(g, x2, x, mu)), n);       // mean(g·dy·x̂)
            RowMath<TLane, TL>.NormBackward(x, x2, g, mu, rstd, m1, TL.Mul(rstd, m2), y);
        }
    }

    public struct WeightNorm<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane k = TL.Div(s, TL.Sqrt(TL.Add(RowMath<TLane, TL>.SumSq(x, TL.Zero), a)));
            RowMath<TLane, TL>.Scale(x, TL.Zero, k, y);
        }
    }

    // dx = (s/n)·(dy − x·k), k = (Σ dy·x)/(Σ x² + a), n = √(Σ x² + a); y2 (one lane value) = (Σ dy·x)/n.
    public struct WeightNormBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane nn = TL.Add(RowMath<TLane, TL>.SumSq(x, TL.Zero), a);
            TLane dot = RowMath<TLane, TL>.Dot(x2, x);
            TLane n = TL.Sqrt(nn), k = TL.Div(dot, nn), f = TL.Div(s, n);
            int w = Vector<TLane>.Count, vec = x.Length / w * w, i = 0;
            Vector<TLane> kv = new(k), fv = new(f);
            for (; i < vec; i += w) ((new Vector<TLane>(x2[i..]) - new Vector<TLane>(x[i..]) * kv) * fv).CopyTo(y[i..]);
            for (; i < x.Length; i++) y[i] = TL.Mul(TL.Sub(x2[i], TL.Mul(x[i], k)), f);
            if (!y2.IsEmpty) y2[0] = TL.Div(dot, n);
        }
    }

    public struct Rope<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            int n = x.Length, half = n / 2, w = Vector<TLane>.Count, vec = half / w * w;
            TLane p = TL.From(pos);
            var freq = Frequencies(n, a);
            Vector<TLane> pv = new(p);
            int i = 0;
            for (; i < vec; i += w)
            {
                var theta = pv * new Vector<TLane>(freq.AsSpan(i));
                var (sn, c) = TL.SinCos(theta);
                var lo = new Vector<TLane>(x[i..]); var hi = new Vector<TLane>(x[(i + half)..]);
                (lo * c - hi * sn).CopyTo(y[i..]);
                (lo * sn + hi * c).CopyTo(y[(i + half)..]);
            }

            for (; i < half; i++)
            {
                TLane theta = TL.Mul(p, freq[i]);
                var tv = new Vector<TLane>(theta);
                TLane c = TL.Cos(tv)[0], sn = TL.Sin(tv)[0];
                y[i] = TL.Sub(TL.Mul(x[i], c), TL.Mul(x[i + half], sn));
                y[i + half] = TL.Add(TL.Mul(x[i], sn), TL.Mul(x[i + half], c));
            }

            if (n % 2 == 1) y[n - 1] = x[n - 1];
        }

        // The frequencies a^(−2i/n) = exp(i · k), k = −2 ln a / n, of the pairs (i < n/2): the same for every row, kept per
        // thread for the last (n, a) (computed as the row did: the vector exp over whole vectors, the scalar over the rest).
        [ThreadStatic] private static TLane[]? s_freq;
        [ThreadStatic] private static int s_freqN;
        [ThreadStatic] private static TLane s_freqA;

        private static TLane[] Frequencies(int n, TLane a)
        {
            if (s_freq is not null && s_freqN == n && EqualityComparer<TLane>.Default.Equals(s_freqA, a)) return s_freq;
            int half = n / 2, w = Vector<TLane>.Count, vec = half / w * w;
            var freq = new TLane[Math.Max(1, half)];
            TLane k = TL.Mul(TL.From(-2.0 / n), RowMath<TLane, TL>.Log(a));
            Span<TLane> ramp = stackalloc TLane[w];
            for (int l = 0; l < w; l++) ramp[l] = TL.From(l);
            var rampV = new Vector<TLane>(ramp);
            Vector<TLane> kv = new(k);
            int i = 0;
            for (; i < vec; i += w) TL.Exp((new Vector<TLane>(TL.From(i)) + rampV) * kv).CopyTo(freq.AsSpan(i));
            for (; i < half; i++) freq[i] = TL.Exp(TL.Mul(TL.From(i), k));
            (s_freq, s_freqN, s_freqA) = (freq, n, a);
            return freq;
        }
    }

    public struct ScaleMaskSoftmax<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2) =>
            Softmax(x, x2, pos, a, b, y);

        // The softmax into y; returns the row's log-sum-exp m + log Σ e^(z − m) over the kept columns (−∞ when none).
        internal static TLane Softmax(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, int pos, TLane a, TLane b, Span<TLane> y)
        {
            // z = a·x + mask into y over the columns kept (causal: columns past the row's position excluded, never read:
            // their outputs are 0 whatever x and the mask hold there).
            int n = x.Length, w = Vector<TLane>.Count, i = 0;
            int limit = TL.From(0).Equals(b) ? n : Math.Max(0, Math.Min(n, pos + 1)), vec = limit / w * w;
            if (limit == 0) { y.Fill(TL.Zero); return TL.NegativeInfinity; }
            var av = new Vector<TLane>(a);
            if (x2.IsEmpty) for (; i < vec; i += w) (new Vector<TLane>(x[i..]) * av).CopyTo(y[i..]);
            else for (; i < vec; i += w) TL.Fma(new Vector<TLane>(x[i..]), av, new Vector<TLane>(x2[i..])).CopyTo(y[i..]);
            for (; i < limit; i++) y[i] = x2.IsEmpty ? TL.Mul(x[i], a) : TL.Fma(x[i], a, x2[i]);
            TLane m = RowMath<TLane, TL>.Max(y[..limit]);
            // Every kept column at −∞ (the mask's): no column kept.
            if (EqualityComparer<TLane>.Default.Equals(m, TL.NegativeInfinity)) { y.Fill(TL.Zero); return TL.NegativeInfinity; }
            TLane sum = RowMath<TLane, TL>.ExpSum(y[..limit], m, y[..limit]);
            RowMath<TLane, TL>.Scale(y[..limit], TL.Zero, TL.Div(TL.From(1), sum), y[..limit]);
            if (limit < n) y[limit..].Fill(TL.Zero);
            return TL.Add(m, RowMath<TLane, TL>.Log(sum));
        }
    }

    // The softmax, and y2 (one lane value) = the row's log-sum-exp.
    public struct ScaleMaskSoftmaxLse<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane lse = ScaleMaskSoftmax<TLane, TL>.Softmax(x, x2, pos, a, b, y);
            if (!y2.IsEmpty) y2[0] = lse;
        }
    }

    public struct CrossEntropyBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            TLane m = RowMath<TLane, TL>.Max(x);
            TLane sum = RowMath<TLane, TL>.ExpSum(x, m, y);
            RowMath<TLane, TL>.Scale(y, TL.Zero, TL.Div(TL.From(1), sum), y);
            if ((uint)pos < (uint)x.Length) y[pos] = TL.Sub(y[pos], TL.From(1));
        }
    }

    public struct WeightedCrossEntropyBwd<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            // (softmax − one_hot) first, then the factor a·s: the target's p − 1 keeps its digits before the scaling.
            CrossEntropyBwd<TLane, TL>.Row(x, x2, g, bt, s, pos, row, a, b, y, y2);
            RowMath<TLane, TL>.Scale(y, TL.Zero, TL.Mul(a, s), y);
        }
    }

    public struct AddRmsNorm<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            RowMath<TLane, TL>.Add(x, x2, y2);
            RmsNormAffine<TLane, TL>.Row(y2, default, g, default, s, pos, row, a, b, y, default);
        }
    }

    public struct AddLayerNorm<TLane, TL> : IRowKernel<TLane, TL> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public static void Row(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> x2, ReadOnlySpan<TLane> g, ReadOnlySpan<TLane> bt, TLane s, int pos, int row, TLane a, TLane b, Span<TLane> y, Span<TLane> y2)
        {
            RowMath<TLane, TL>.Add(x, x2, y2);
            LayerNorm<TLane, TL>.Row(y2, default, g, bt, s, pos, row, a, b, y, default);
        }
    }
}

internal sealed partial class Kernels
{
    /// <summary>A row operation over x (rows × cols, stored as TN) into y (and y2) in the type's float lane; returns the lane's name.</summary>
    public static string RowOp<TN>(RowOp op, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) where TN : struct, IFloatAdad<TN>
    {
        var v = new RowOpVisitor(op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct RowOpVisitor(RowOp op, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            switch (op.Name)
            {
                case "log_softmax": Go<TN, TLane, TL, RowKernels.LogSoftmax<TLane, TL>>(); break;
                case "softmax_bwd": Go<TN, TLane, TL, RowKernels.SoftmaxBwd<TLane, TL>>(); break;
                case "log_softmax_bwd": Go<TN, TLane, TL, RowKernels.LogSoftmaxBwd<TLane, TL>>(); break;
                case "rms_norm_affine": Go<TN, TLane, TL, RowKernels.RmsNormAffine<TLane, TL>>(); break;
                case "rms_norm_bwd": Go<TN, TLane, TL, RowKernels.RmsNormBwd<TLane, TL>>(); break;
                case "layer_norm": Go<TN, TLane, TL, RowKernels.LayerNorm<TLane, TL>>(); break;
                case "layer_norm_bwd": Go<TN, TLane, TL, RowKernels.LayerNormBwd<TLane, TL>>(); break;
                case "weight_norm": Go<TN, TLane, TL, RowKernels.WeightNorm<TLane, TL>>(); break;
                case "weight_norm_bwd": Go<TN, TLane, TL, RowKernels.WeightNormBwd<TLane, TL>>(); break;
                case "rope": Go<TN, TLane, TL, RowKernels.Rope<TLane, TL>>(); break;
                case "scale_mask_softmax": Go<TN, TLane, TL, RowKernels.ScaleMaskSoftmax<TLane, TL>>(); break;
                case "scale_mask_softmax_lse": Go<TN, TLane, TL, RowKernels.ScaleMaskSoftmaxLse<TLane, TL>>(); break;
                case "add_rms_norm": Go<TN, TLane, TL, RowKernels.AddRmsNorm<TLane, TL>>(); break;
                case "add_layer_norm": Go<TN, TLane, TL, RowKernels.AddLayerNorm<TLane, TL>>(); break;
                case "cross_entropy_bwd": Go<TN, TLane, TL, RowKernels.CrossEntropyBwd<TLane, TL>>(); break;
                case "weighted_cross_entropy_bwd": Go<TN, TLane, TL, RowKernels.WeightedCrossEntropyBwd<TLane, TL>>(); break;
                default: throw new NotSupportedException($"row operation {op.Name}");
            }
        }

        private readonly void Go<TN, TLane, TL, TK>()
            where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> where TK : struct, IRowKernel<TLane, TL>
        {
            var body = new RowOpBody<TN, TLane, TL, TK>(op, x, x2, g, bt, s, pos, y, y2, cols, TL.From(a), TL.From(b));
            Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), (long)rows * cols * 4, body);
        }
    }

    private struct RowOpBody<TN, TLane, TL, TK>(RowOp op, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int cols, TLane a, TLane b) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> where TK : struct, IRowKernel<TLane, TL>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            // Lane rows for what the stored bytes are not (direct types read and write their own storage).
            var xl = TN.Direct ? null : pool.Rent(cols);
            var x2l = x2 is null || TN.Direct ? null : pool.Rent(cols);
            var yl = TN.Direct ? null : pool.Rent(cols);
            var y2l = y2 is null || TN.Direct || op.RowOut ? null : pool.Rent(cols);
            var gl = g is null ? null : pool.Rent(cols);
            var btl = bt is null ? null : pool.Rent(cols);
            var sl = s is null ? null : pool.Rent(1);
            try
            {
                if (g is not null) TN.Load(g, 0, gl.AsSpan(0, cols));
                if (bt is not null) TN.Load(bt, 0, btl.AsSpan(0, cols));
                for (int r = first; r < last; r++)
                {
                    ReadOnlySpan<TLane> xs, x2s = default;
                    Span<TLane> ys, y2s = default;
                    if (TN.Direct)
                    {
                        xs = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols, cols);
                        if (x2 is not null) x2s = MemoryMarshal.Cast<byte, TLane>(x2.AsSpan()).Slice(r * cols, cols);
                        ys = MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(r * cols, cols);
                        if (y2 is not null && !op.RowOut) y2s = MemoryMarshal.Cast<byte, TLane>(y2.AsSpan()).Slice(r * cols, cols);
                    }
                    else
                    {
                        TN.Load(x, r * cols, xl.AsSpan(0, cols));
                        xs = xl.AsSpan(0, cols);
                        if (x2 is not null) { TN.Load(x2, r * cols, x2l.AsSpan(0, cols)); x2s = x2l.AsSpan(0, cols); }
                        ys = yl.AsSpan(0, cols);
                        if (y2 is not null && !op.RowOut) y2s = y2l.AsSpan(0, cols);
                    }

                    if (y2 is not null && op.RowOut) y2s = MemoryMarshal.Cast<byte, TLane>(y2.AsSpan()).Slice(r, 1);   // the row's value, lane storage

                    TLane sv = TL.From(1);
                    if (s is not null) { TN.Load(s, r, sl.AsSpan(0, 1)); sv = sl![0]; }
                    int p = pos is null ? r : pos[r];
                    TK.Row(xs, x2s, g is null ? default : gl.AsSpan(0, cols), bt is null ? default : btl.AsSpan(0, cols), sv, p, r, a, b, ys, y2s);
                    if (!TN.Direct)
                    {
                        TN.Store(ys, y, r * cols);
                        if (y2 is not null && !op.RowOut) TN.Store(y2s, y2, r * cols);
                    }
                }
            }
            finally
            {
                if (xl is not null) pool.Return(xl);
                if (x2l is not null) pool.Return(x2l);
                if (yl is not null) pool.Return(yl);
                if (y2l is not null) pool.Return(y2l);
                if (gl is not null) pool.Return(gl);
                if (btl is not null) pool.Return(btl);
                if (sl is not null) pool.Return(sl);
            }
        }
    }

    /// <summary>dgamma[c] = Σ_r dy·x̂, dbeta[c] = Σ_r dy (x̂ by the row's rms or, centered, layer statistics): results as lane values (cols each); returns the lane's name.</summary>
    public static string NormParamsBwd<TN>(byte[] x, byte[] dy, byte[] dgamma, byte[] dbeta, int rows, int cols, double eps, bool centered) where TN : struct, IFloatAdad<TN>
    {
        var v = new NormParamsVisitor(x, dy, dgamma, dbeta, rows, cols, eps, centered);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct NormParamsVisitor(byte[] x, byte[] dy, byte[] dgamma, byte[] dbeta, int rows, int cols, double eps, bool centered) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            // The rows' statistics (mean, rstd), then the columns across the team.
            var stats = new TLane[rows * 2];
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols, new RowStatsBody<TN, TLane, TL>(x, stats, cols, TL.From(eps), centered));
            // The columns as tiles (Reduce.cs ColumnPlan), the blocks' sums joined in block order.
            var plan = ColumnPlan.Of<TLane>(rows, cols, 1);
            var dg = plan.Blocks == 1 ? dgamma : plan.Partials<TLane>(cols);
            var db = plan.Blocks == 1 ? dbeta : plan.Partials<TLane>(cols);
            Team.For(plan.Tasks, 1, (long)rows * cols * 2, new ColumnTiles<NormParamsTile<TN, TLane, TL>>(new(x, dy, stats, dg, db, cols), plan, rows, cols));
            if (plan.Blocks > 1)
            {
                Join<TLane, TL>(dg, plan.Blocks, cols, dgamma, JoinOp.Add);
                Join<TLane, TL>(db, plan.Blocks, cols, dbeta, JoinOp.Add);
            }
        }
    }

    private struct RowStatsBody<TN, TLane, TL>(byte[] x, TLane[] stats, int cols, TLane eps, bool centered) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var xl = TN.Direct ? null : pool.Rent(cols);
            try
            {
                for (int r = first; r < last; r++)
                {
                    ReadOnlySpan<TLane> xs;
                    if (TN.Direct) xs = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols, cols);
                    else { TN.Load(x, r * cols, xl.AsSpan(0, cols)); xs = xl.AsSpan(0, cols); }
                    TLane mu = centered ? TL.Div(RowMath<TLane, TL>.Sum(xs), TL.From(cols)) : TL.Zero;
                    TLane rstd = TL.Div(TL.From(1), TL.Sqrt(TL.Add(TL.Div(RowMath<TLane, TL>.SumSq(xs, mu), TL.From(cols)), eps)));
                    stats[2 * r] = mu;
                    stats[2 * r + 1] = rstd;
                }
            }
            finally
            {
                if (xl is not null) pool.Return(xl);
            }
        }
    }

    // dgamma and dbeta of a tile: Σ dy·(x − mean)·rstd and Σ dy over its rows, in place where the stored values are the lane's.
    private readonly struct NormParamsTile<TN, TLane, TL>(byte[] x, byte[] dy, TLane[] stats, byte[] dgamma, byte[] dbeta, int cols) : IColumnTile
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Tile(int r0, int r1, int c0, int c1, int block)
        {
            int n = c1 - c0, w = Vector<TLane>.Count;
            var ag = MemoryMarshal.Cast<byte, TLane>(dgamma.AsSpan()).Slice(block * cols + c0, n);
            var ab = MemoryMarshal.Cast<byte, TLane>(dbeta.AsSpan()).Slice(block * cols + c0, n);
            ag.Clear();
            ab.Clear();
            var pool = ArrayPool<TLane>.Shared;
            var xl = TN.Direct ? null : pool.Rent(n);
            var dl = TN.Direct ? null : pool.Rent(n);
            ref TLane rg = ref MemoryMarshal.GetReference(ag);
            ref TLane rb = ref MemoryMarshal.GetReference(ab);
            for (int r = r0; r < r1; r++)
            {
                ReadOnlySpan<TLane> xs, ds;
                if (TN.Direct)
                {
                    xs = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols + c0, n);
                    ds = MemoryMarshal.Cast<byte, TLane>(dy.AsSpan()).Slice(r * cols + c0, n);
                }
                else
                {
                    TN.Load(x, r * cols + c0, xl.AsSpan(0, n));
                    TN.Load(dy, r * cols + c0, dl.AsSpan(0, n));
                    xs = xl.AsSpan(0, n);
                    ds = dl.AsSpan(0, n);
                }

                ref TLane rx = ref MemoryMarshal.GetReference(xs);
                ref TLane rd = ref MemoryMarshal.GetReference(ds);
                Vector<TLane> mu = new(stats[2 * r]), rstd = new(stats[2 * r + 1]);
                int i = 0;
                for (; i + w <= n; i += w)
                {
                    var d = Vector.LoadUnsafe(ref rd, (nuint)i);
                    var h = (Vector.LoadUnsafe(ref rx, (nuint)i) - mu) * rstd;
                    TL.Fma(d, h, Vector.LoadUnsafe(ref rg, (nuint)i)).StoreUnsafe(ref rg, (nuint)i);
                    (Vector.LoadUnsafe(ref rb, (nuint)i) + d).StoreUnsafe(ref rb, (nuint)i);
                }

                for (; i < n; i++)
                {
                    TLane h = TL.Mul(TL.Sub(xs[i], stats[2 * r]), stats[2 * r + 1]);
                    ag[i] = TL.Fma(ds[i], h, ag[i]);
                    ab[i] = TL.Add(ab[i], ds[i]);
                }
            }

            if (xl is not null) pool.Return(xl);
            if (dl is not null) pool.Return(dl);
        }
    }
}
