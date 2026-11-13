// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Group normalization (batch, group, instance: GroupShape) and the rotary embedding by tables (RopeArgs) on the CPU, in the
// type's float lane (Rows.cs has their references). A group's statistics by a task a group, its Outer runs of Inner values
// gathered; the outputs by aligned tiles of the index (a stored word or block written by one task), each run of values with
// one group and one channel taken with that group's statistics and that channel's gain and bias; the parameters' gradients by
// a task a channel. The rotary embedding by a task a run of rows (RowChunk: whole words and blocks), each row rotated in the
// lane.
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    public static string GroupNorm<TN>(byte[] x, byte[]? gamma, byte[]? beta, byte[]? mean, byte[]? variance, byte[] y, GroupShape shape, double eps, bool given) where TN : struct, IFloatAdad<TN>
    {
        var v = new GroupNormVisitor(x, null, gamma, beta, mean, variance, y, null, null, shape, eps, given, false);
        TN.Float(ref v);
        return v.Lane;
    }

    public static string GroupNormBwd<TN>(byte[] x, byte[] dy, byte[]? gamma, byte[]? mean, byte[]? variance, byte[]? dx, byte[]? dgamma, byte[]? dbeta, GroupShape shape, double eps, bool given)
        where TN : struct, IFloatAdad<TN>
    {
        var v = new GroupNormVisitor(x, dy, gamma, null, mean, variance, dx, dgamma, dbeta, shape, eps, given, true);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct GroupNormVisitor(byte[] x, byte[]? dy, byte[]? gamma, byte[]? beta, byte[]? mean, byte[]? variance, byte[]? output, byte[]? dgamma, byte[]? dbeta,
        GroupShape s, double eps, bool given, bool backward) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            if (s.Count == 0) return;
            TLane[]? gl = null, bl = null;
            if (gamma is not null) { gl = new TLane[s.Channels]; TN.Load(gamma, 0, gl); }
            if (beta is not null) { bl = new TLane[s.Channels]; TN.Load(beta, 0, bl); }

            // Each group's coefficients: μ, rstd and, backward, m1 = mean(γ·dy) and k2 = rstd·mean(γ·dy·x̂).
            var coef = new TLane[s.Groups * 4];
            Team.For(s.Groups, 1, (long)s.Count * (backward ? 3 : 2), new GroupCoefBody<TN, TLane, TL>(x, dy, gl, mean, variance, coef, s, TL.From(eps), given, backward));
            if (output is not null)
            {
                int size = Math.Max(Tile, (s.Count / (Threads * 4) + Tile - 1) / Tile * Tile);
                Team.For(s.Count, size, (long)s.Count * 2, new GroupOutBody<TN, TLane, TL>(x, dy, gl, bl, coef, output, s, backward));
            }

            if (backward && (dgamma is not null || dbeta is not null))
                Team.For(s.Channels, 1, (long)s.Count * 2, new GroupParamsBody<TN, TLane, TL>(x, dy!, coef, dgamma, dbeta, s));
        }
    }

    private readonly struct GroupCoefBody<TN, TLane, TL>(byte[] x, byte[]? dy, TLane[]? gl, byte[]? mean, byte[]? variance, TLane[] coef, GroupShape s, TLane eps, bool given, bool backward)
        : Team.IBody where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int m = s.PerGroup;
            bool sums = backward && !given;
            var pool = ArrayPool<TLane>.Shared;
            var xs = given ? null : pool.Rent(m);
            var ds = sums ? pool.Rent(m) : null;
            var gs = sums ? pool.Rent(m) : null;
            try
            {
                TLane count = TL.From(m), one = TL.From(1);
                for (int g = first; g < last; g++)
                {
                    TLane mu, rstd, m1 = TL.Zero, k2 = TL.Zero;
                    if (given)
                    {
                        mu = MemoryMarshal.Cast<byte, TLane>(mean.AsSpan())[g];
                        rstd = TL.Div(one, TL.Sqrt(TL.Add(MemoryMarshal.Cast<byte, TLane>(variance.AsSpan())[g], eps)));
                    }
                    else
                    {
                        for (int o = 0; o < s.Outer; o++) TN.Load(x, s.Index(o, g, 0), xs.AsSpan(o * s.Inner, s.Inner));
                        var xm = xs.AsSpan(0, m);
                        mu = TL.Div(RowMath<TLane, TL>.Sum(xm), count);
                        TLane var = TL.Div(RowMath<TLane, TL>.SumSq(xm, mu), count);
                        rstd = TL.Div(one, TL.Sqrt(TL.Add(var, eps)));
                        if (mean is not null) MemoryMarshal.Cast<byte, TLane>(mean.AsSpan())[g] = mu;
                        if (variance is not null) MemoryMarshal.Cast<byte, TLane>(variance.AsSpan())[g] = var;
                    }

                    if (sums)
                    {
                        for (int o = 0; o < s.Outer; o++)
                        {
                            int at = s.Index(o, g, 0);
                            TN.Load(dy!, at, ds.AsSpan(o * s.Inner, s.Inner));
                            for (int i = 0; i < s.Inner; i++) gs![o * s.Inner + i] = gl is null ? one : gl[s.ChannelOf(at + i)];
                        }

                        TLane s1 = RowMath<TLane, TL>.Dot(gs.AsSpan(0, m), ds.AsSpan(0, m));                       // Σ γ·dy
                        TLane s2 = RowMath<TLane, TL>.Dot3(gs.AsSpan(0, m), ds.AsSpan(0, m), xs.AsSpan(0, m), mu);   // Σ γ·dy·(x − μ)
                        m1 = TL.Div(s1, count);
                        k2 = TL.Mul(rstd, TL.Div(TL.Mul(rstd, s2), count));
                    }

                    coef[4 * g] = mu;
                    coef[4 * g + 1] = rstd;
                    coef[4 * g + 2] = m1;
                    coef[4 * g + 3] = k2;
                }
            }
            finally
            {
                if (xs is not null) pool.Return(xs);
                if (ds is not null) pool.Return(ds);
                if (gs is not null) pool.Return(gs);
            }
        }
    }

    // Forward: y = x̂·γ + β (x̂ = (x − μ)·rstd; fma with a bias, a product without, γ = 1 absent). Backward: dx = (γ·dy − m1 − (x − μ)·k2)·rstd.
    private readonly struct GroupOutBody<TN, TLane, TL>(byte[] x, byte[]? dy, TLane[]? gl, TLane[]? bl, TLane[] coef, byte[] output, GroupShape s, bool backward)
        : Team.IBody where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var xs = pool.Rent(Tile);
            var ds = backward ? pool.Rent(Tile) : null;
            var ys = pool.Rent(Tile);
            int w = Vector<TLane>.Count;
            TLane one = TL.From(1);
            try
            {
                for (int at = first; at < last; at += Tile)
                {
                    int len = Math.Min(Tile, last - at);
                    TN.Load(x, at, xs.AsSpan(0, len));
                    if (backward) TN.Load(dy!, at, ds.AsSpan(0, len));
                    for (int i = 0; i < len;)
                    {
                        int idx = at + i, g = s.GroupOf(idx), c = s.ChannelOf(idx);
                        int end = Math.Min(len, i + Math.Min(s.Inner - idx % s.Inner, s.Spatial - idx % s.Spatial));
                        TLane mu = coef[4 * g], rstd = coef[4 * g + 1], gv = gl is null ? one : gl[c];
                        Vector<TLane> muv = new(mu), rv = new(rstd), gvv = new(gv);
                        int j = i;
                        if (backward)
                        {
                            TLane m1 = coef[4 * g + 2], k2 = coef[4 * g + 3];
                            Vector<TLane> m1v = new(m1), k2v = new(k2);
                            for (; j + w <= end; j += w)
                                ((gvv * new Vector<TLane>(ds.AsSpan(j)) - m1v - (new Vector<TLane>(xs.AsSpan(j)) - muv) * k2v) * rv).CopyTo(ys.AsSpan(j));
                            for (; j < end; j++) ys[j] = TL.Mul(TL.Sub(TL.Sub(TL.Mul(gv, ds![j]), m1), TL.Mul(TL.Sub(xs[j], mu), k2)), rstd);
                        }
                        else if (bl is not null)
                        {
                            TLane bv = bl[c];
                            Vector<TLane> bvv = new(bv);
                            for (; j + w <= end; j += w) TL.Fma((new Vector<TLane>(xs.AsSpan(j)) - muv) * rv, gvv, bvv).CopyTo(ys.AsSpan(j));
                            for (; j < end; j++) ys[j] = TL.Fma(TL.Mul(TL.Sub(xs[j], mu), rstd), gv, bv);
                        }
                        else
                        {
                            for (; j + w <= end; j += w) ((new Vector<TLane>(xs.AsSpan(j)) - muv) * rv * gvv).CopyTo(ys.AsSpan(j));
                            for (; j < end; j++) ys[j] = TL.Mul(TL.Mul(TL.Sub(xs[j], mu), rstd), gv);
                        }

                        i = end;
                    }

                    TN.Store(ys.AsSpan(0, len), output, at);
                }
            }
            finally
            {
                pool.Return(xs);
                pool.Return(ys);
                if (ds is not null) pool.Return(ds);
            }
        }
    }

    // dgamma[c] = Σ dy·(x − μ)·rstd and dbeta[c] = Σ dy over channel c's runs of Spatial values (each with its group's statistics).
    private readonly struct GroupParamsBody<TN, TLane, TL>(byte[] x, byte[] dy, TLane[] coef, byte[]? dgamma, byte[]? dbeta, GroupShape s)
        : Team.IBody where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var xs = pool.Rent(s.Spatial);
            var ds = pool.Rent(s.Spatial);
            int w = Vector<TLane>.Count, blocks = s.Count / (s.Channels * s.Spatial);
            try
            {
                for (int c = first; c < last; c++)
                {
                    Vector<TLane> ag = Vector<TLane>.Zero, ab = Vector<TLane>.Zero;
                    TLane sg = TL.Zero, sb = TL.Zero;
                    for (int k = 0; k < blocks; k++)
                    {
                        int start = (k * s.Channels + c) * s.Spatial;
                        TN.Load(x, start, xs.AsSpan(0, s.Spatial));
                        TN.Load(dy, start, ds.AsSpan(0, s.Spatial));
                        for (int i = 0; i < s.Spatial;)
                        {
                            int idx = start + i, g = s.GroupOf(idx), end = Math.Min(s.Spatial, i + (s.Inner - idx % s.Inner));
                            TLane mu = coef[4 * g], rstd = coef[4 * g + 1];
                            Vector<TLane> muv = new(mu), rv = new(rstd);
                            int j = i;
                            for (; j + w <= end; j += w)
                            {
                                var d = new Vector<TLane>(ds.AsSpan(j));
                                ag = TL.Fma(d, (new Vector<TLane>(xs.AsSpan(j)) - muv) * rv, ag);
                                ab += d;
                            }

                            for (; j < end; j++)
                            {
                                sg = TL.Fma(ds[j], TL.Mul(TL.Sub(xs[j], mu), rstd), sg);
                                sb = TL.Add(sb, ds[j]);
                            }

                            i = end;
                        }
                    }

                    if (dgamma is not null) MemoryMarshal.Cast<byte, TLane>(dgamma.AsSpan())[c] = TL.Add(Vector.Sum(ag), sg);
                    if (dbeta is not null) MemoryMarshal.Cast<byte, TLane>(dbeta.AsSpan())[c] = TL.Add(Vector.Sum(ab), sb);
                }
            }
            finally
            {
                pool.Return(xs);
                pool.Return(ds);
            }
        }
    }

    // ---------------------------------------------------------------- the rotary embedding by tables

    public static string Rope<TN>(RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y) where TN : struct, IFloatAdad<TN>
    {
        var v = new RopeVisitor(a, x, cos, sin, positions, y);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct RopeVisitor(RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            if (a.Rows == 0 || a.Cols == 0) return;
            Team.For(a.Rows, RowChunk<TN>(a.Rows, a.Cols, a.Rows / (Threads * 4)), (long)a.Rows * a.Cols * 3, new RopeBody<TN, TLane, TL>(a, x, cos, sin, positions, y));
        }
    }

    // y = x with pair (i, j) rotated: y_i = fma(−x_j, s, x_i·c), y_j = fma(x_i, s, x_j·c) (s = −sin inverse), each as
    // fma(x_j, −s, x_i·c) (the same product, its sign on s). A direct type is read and written in place (x may be y: each pair
    // read before it is written); another type through lane rows. The split pairs (p, p + Half) a vector of p at a time.
    private readonly struct RopeBody<TN, TLane, TL>(RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int n = a.Cols;
            bool direct = TN.Direct;
            var pool = ArrayPool<TLane>.Shared;
            var xl = direct ? null : pool.Rent(n);
            var yl = direct ? null : pool.Rent(n);
            var ct = MemoryMarshal.Cast<byte, TLane>(cos.AsSpan());
            var st = MemoryMarshal.Cast<byte, TLane>(sin.AsSpan());
            var store = new UnitStore<TN, TLane>(y, n);
            try
            {
                for (int r = first; r < last; r++)
                {
                    Span<TLane> xs, ys;
                    if (direct)
                    {
                        xs = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * n, n);
                        ys = MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(r * n, n);
                    }
                    else
                    {
                        xs = xl.AsSpan(0, n);
                        ys = yl.AsSpan(0, n);
                        TN.Load(x, r * n, xs);
                    }

                    int pos = a.Position(positions, r);
                    Row(xs, ys, ct.Slice(pos * a.Half, a.Half), st.Slice(pos * a.Half, a.Half));
                    if (!direct) store.Put(ys, (long)r * n);
                }

                store.Done();
            }
            finally
            {
                if (!direct)
                {
                    pool.Return(xl!);
                    pool.Return(yl!);
                }
            }
        }

        private void Row(ReadOnlySpan<TLane> xs, Span<TLane> ys, ReadOnlySpan<TLane> c, ReadOnlySpan<TLane> s)
        {
            int half = a.Half, p = 0;
            xs[(2 * half)..].CopyTo(ys[(2 * half)..]);
            if (!a.Interleaved && Vector.IsHardwareAccelerated && Vector<TLane>.Count > 1)
            {
                var sign = new Vector<TLane>(Negate(default(TLane)));
                for (int w = Vector<TLane>.Count; p + w <= half; p += w)
                {
                    Vector<TLane> xi = new(xs[p..]), xj = new(xs[(half + p)..]), cv = new(c[p..]), sv = new(s[p..]);
                    var nv = Vector.Xor(sv, sign);
                    if (a.Inverse) (sv, nv) = (nv, sv);
                    TL.Fma(xj, nv, TL.Mul(xi, cv)).CopyTo(ys[p..]);
                    TL.Fma(xi, sv, TL.Mul(xj, cv)).CopyTo(ys[(half + p)..]);
                }
            }

            for (; p < half; p++)
            {
                var (i, j) = a.Pair(p);
                TLane xi = xs[i], xj = xs[j], cp = c[p], sn = s[p];
                if (a.Inverse) sn = Negate(sn);
                ys[i] = TL.Fma(xj, Negate(sn), TL.Mul(xi, cp));
                ys[j] = TL.Fma(xi, sn, TL.Mul(xj, cp));
            }
        }
    }

    // −v exactly (the sign flipped, as the devices' negation: −(+0) = −0).
    private static T Negate<T>(T v) where T : unmanaged
    {
        if (typeof(T) == typeof(float)) { float f = -System.Runtime.CompilerServices.Unsafe.As<T, float>(ref v); return System.Runtime.CompilerServices.Unsafe.As<float, T>(ref f); }
        double d = -System.Runtime.CompilerServices.Unsafe.As<T, double>(ref v);
        return System.Runtime.CompilerServices.Unsafe.As<double, T>(ref d);
    }
}
