// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The generic CPU side: one implementation per operation over T : INumberBase<T> (the float formulas need exp, sqrt, atan:
// T : IFloatingPointIeee754<T>). Each runs Vector<T> where the runtime supports T (every primitive number; not Half) and
// T's own operators otherwise, split across threads by Parallel.For with simple cut-overs, so only the kernel differs.
// Sums are in T (no wider type), the generic code's natural choice: what that does to small types is part of the result.
// Where an operation's state comes in another type (an optimizer's moments, rope's tables: float64 for double, float32
// otherwise), its arithmetic is in that type, TM, with T converted on load and store.
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Qiyas;

internal static class CpuGeneric
{
    private const long SplitWork = 1 << 15;                                   // work below which an operation runs on one thread
    private const int ParallelElements = 1 << 16;                             // values a chunk of an element-wise split holds at least
    private static readonly int Threads = Environment.ProcessorCount;
    private static readonly ParallelOptions Options = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    public static bool Vectorized<T>() where T : struct, INumberBase<T> => Vector.IsHardwareAccelerated && Vector<T>.IsSupported;

    // The split of an element-wise kernel: chunks of whole vectors.
    private static void Split(int n, long work, Action<int, int> body)
    {
        if (work < SplitWork)
        {
            body(0, n);
            return;
        }

        int chunks = Math.Max(1, Math.Min(Threads * 2, n / Math.Max(1, ParallelElements / 4)));
        int size = (n + chunks - 1) / chunks;
        Parallel.For(0, chunks, Options, c =>
        {
            int start = c * size;
            if (start < n)
            {
                body(start, Math.Min(start + size, n));
            }
        });
    }

    // The split of a row kernel.
    private static void Rows(int rows, long work, Action<int, int> body)
    {
        if (rows <= 1 || work < SplitWork)
        {
            body(0, rows);
            return;
        }

        int chunks = Math.Min(rows, Threads * 4);
        int size = (rows + chunks - 1) / chunks;
        Parallel.For(0, chunks, Options, c =>
        {
            int start = c * size;
            if (start < rows)
            {
                body(start, Math.Min(start + size, rows));
            }
        });
    }

    // ------------------------------------------------------------------ c = a + b

    public static void Add<T>(T[] a, T[] b, T[] c, int n) where T : struct, INumberBase<T> => Split(n, n, (start, end) => AddRange(a, b, c, start, end));

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AddRange<T>(T[] a, T[] b, T[] c, int start, int end) where T : struct, INumberBase<T>
    {
        var x = a.AsSpan(start, end - start);
        var y = b.AsSpan(start, end - start);
        var z = c.AsSpan(start, end - start);
        int i = 0;
        if (Vectorized<T>())
        {
            var xv = MemoryMarshal.Cast<T, Vector<T>>(x);
            var yv = MemoryMarshal.Cast<T, Vector<T>>(y);
            var zv = MemoryMarshal.Cast<T, Vector<T>>(z);
            for (int v = 0; v < xv.Length; v++)
            {
                zv[v] = xv[v] + yv[v];
            }

            i = xv.Length * Vector<T>.Count;
        }

        for (; i < x.Length; i++)
        {
            z[i] = x[i] + y[i];
        }
    }

    // ------------------------------------------------------------------ C = A·B ([m,k]·[k,n], row-major), sums in T

    private const int BlockRows = 4;                                         // a portable register block of four rows

    public static void MatMul<T>(T[] a, T[] b, T[] c, int m, int n, int k) where T : struct, INumberBase<T>
    {
        int blocks = (m + BlockRows - 1) / BlockRows;
        Rows(blocks, (long)m * n * k, (first, last) =>
        {
            for (int block = first; block < last; block++)
            {
                int i0 = block * BlockRows, rows = Math.Min(BlockRows, m - i0);
                if (Vectorized<T>() && rows == BlockRows)
                {
                    Block4(a, b, c, i0, n, k);
                }
                else
                {
                    for (int i = i0; i < i0 + rows; i++)
                    {
                        RowScalar(a, b, c, i, 0, n, k);
                    }
                }
            }
        });
    }

    // Four rows of C, two vectors of columns at a time held in registers across all of k; the columns past the last
    // pair of vectors by T's operators.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Block4<T>(T[] a, T[] b, T[] c, int i0, int n, int k) where T : struct, INumberBase<T>
    {
        int w = Vector<T>.Count, j = 0;
        ref T ra = ref MemoryMarshal.GetArrayDataReference(a);
        ref T rb = ref MemoryMarshal.GetArrayDataReference(b);
        ref T rc = ref MemoryMarshal.GetArrayDataReference(c);
        for (; j + 2 * w <= n; j += 2 * w)
        {
            Vector<T> c00 = Vector<T>.Zero, c01 = Vector<T>.Zero, c10 = Vector<T>.Zero, c11 = Vector<T>.Zero;
            Vector<T> c20 = Vector<T>.Zero, c21 = Vector<T>.Zero, c30 = Vector<T>.Zero, c31 = Vector<T>.Zero;
            for (int p = 0; p < k; p++)
            {
                var b0 = Vector.LoadUnsafe(ref rb, (nuint)(p * n + j));
                var b1 = Vector.LoadUnsafe(ref rb, (nuint)(p * n + j + w));
                var a0 = new Vector<T>(Unsafe.Add(ref ra, i0 * k + p));
                var a1 = new Vector<T>(Unsafe.Add(ref ra, (i0 + 1) * k + p));
                var a2 = new Vector<T>(Unsafe.Add(ref ra, (i0 + 2) * k + p));
                var a3 = new Vector<T>(Unsafe.Add(ref ra, (i0 + 3) * k + p));
                c00 += a0 * b0; c01 += a0 * b1;
                c10 += a1 * b0; c11 += a1 * b1;
                c20 += a2 * b0; c21 += a2 * b1;
                c30 += a3 * b0; c31 += a3 * b1;
            }

            c00.StoreUnsafe(ref rc, (nuint)(i0 * n + j)); c01.StoreUnsafe(ref rc, (nuint)(i0 * n + j + w));
            c10.StoreUnsafe(ref rc, (nuint)((i0 + 1) * n + j)); c11.StoreUnsafe(ref rc, (nuint)((i0 + 1) * n + j + w));
            c20.StoreUnsafe(ref rc, (nuint)((i0 + 2) * n + j)); c21.StoreUnsafe(ref rc, (nuint)((i0 + 2) * n + j + w));
            c30.StoreUnsafe(ref rc, (nuint)((i0 + 3) * n + j)); c31.StoreUnsafe(ref rc, (nuint)((i0 + 3) * n + j + w));
        }

        for (int i = i0; i < i0 + BlockRows; i++)
        {
            RowScalar(a, b, c, i, j, n, k);
        }
    }

    // Row i of C from column j0 on, by T's operators: C[i, j] = Σ_p A[i, p]·B[p, j], walked p-major so B is read in order.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RowScalar<T>(T[] a, T[] b, T[] c, int i, int j0, int n, int k) where T : struct, INumberBase<T>
    {
        var row = c.AsSpan(i * n + j0, n - j0);
        row.Clear();
        for (int p = 0; p < k; p++)
        {
            T x = a[i * k + p];
            var bp = b.AsSpan(p * n + j0, n - j0);
            for (int j = 0; j < row.Length; j++)
            {
                row[j] += x * bp[j];
            }
        }
    }

    // ------------------------------------------------------------------ convolution: im2col, col2im, transpose (in T)

    // The columns of one image (C × H × W): row (c, a, b), column (i, j) = x[c, i·Sh − PadTop + a·Dh, j·Sw − PadLeft + b·Dw],
    // 0 off the image.
    public static void Im2col<T>(T[] x, T[] cols, ConvArgs p) where T : struct, INumberBase<T>
    {
        var (oh, ow) = p.Out;
        int rows = p.C * p.Kh * p.Kw, n = oh * ow;
        Rows(rows, (long)rows * n, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                int c = r / (p.Kh * p.Kw), a = r / p.Kw % p.Kh, b = r % p.Kw;
                var dst = cols.AsSpan(r * n, n);
                for (int i = 0; i < oh; i++)
                {
                    int ih = i * p.Sh - p.PadTop + a * p.Dh;
                    for (int j = 0; j < ow; j++)
                    {
                        int iw = j * p.Sw - p.PadLeft + b * p.Dw;
                        dst[i * ow + j] = (uint)ih < (uint)p.H && (uint)iw < (uint)p.W ? x[(c * p.H + ih) * p.W + iw] : T.Zero;
                    }
                }
            }
        });
    }

    // The image's gradient from the columns' (col2im), gathered: each value of dx the sum in T of the column values its
    // position fed, taps in order (no two threads write one value).
    public static void Col2im<T>(T[] dcols, T[] dx, ConvArgs p) where T : struct, INumberBase<T>
    {
        var (oh, ow) = p.Out;
        int n = oh * ow;
        Rows(p.C * p.H, (long)p.C * p.H * p.W * p.Kh * p.Kw, (first, last) =>
        {
            for (int ch = first; ch < last; ch++)
            {
                int c = ch / p.H, ih = ch % p.H;
                for (int iw = 0; iw < p.W; iw++)
                {
                    T s = T.Zero;
                    for (int a = 0; a < p.Kh; a++)
                    {
                        int ti = ih + p.PadTop - a * p.Dh;
                        if (ti < 0 || ti % p.Sh != 0 || ti / p.Sh >= oh) continue;
                        for (int b = 0; b < p.Kw; b++)
                        {
                            int tj = iw + p.PadLeft - b * p.Dw;
                            if (tj < 0 || tj % p.Sw != 0 || tj / p.Sw >= ow) continue;
                            s += dcols[((c * p.Kh + a) * p.Kw + b) * n + ti / p.Sh * ow + tj / p.Sw];
                        }
                    }

                    dx[(c * p.H + ih) * p.W + iw] = s;
                }
            }
        });
    }

    // t = xᵀ (x rows × cols).
    public static void Transpose<T>(T[] x, T[] t, int rows, int cols) where T : struct, INumberBase<T> =>
        Rows(cols, (long)rows * cols, (first, last) =>
        {
            for (int j = first; j < last; j++)
                for (int i = 0; i < rows; i++)
                    t[j * rows + i] = x[i * cols + j];
        });

    // ------------------------------------------------------------------ softmax over rows

    public static void Softmax<T>(T[] x, T[] y, int rows, int cols) where T : struct, IFloatingPointIeee754<T>
    {
        Rows(rows, (long)rows * cols * 8, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                SoftmaxRow(x.AsSpan(r * cols, cols), y.AsSpan(r * cols, cols));
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SoftmaxRow<T>(ReadOnlySpan<T> x, Span<T> y) where T : struct, IFloatingPointIeee754<T>
    {
        T max = T.NegativeInfinity;
        int i = 0;
        if (Vectorized<T>() && x.Length >= Vector<T>.Count)
        {
            var mv = new Vector<T>(T.NegativeInfinity);
            var xv = MemoryMarshal.Cast<T, Vector<T>>(x);
            for (int v = 0; v < xv.Length; v++)
            {
                mv = Vector.Max(mv, xv[v]);
            }

            for (int l = 0; l < Vector<T>.Count; l++)
            {
                max = T.Max(max, mv[l]);
            }

            i = xv.Length * Vector<T>.Count;
        }

        for (; i < x.Length; i++)
        {
            max = T.Max(max, x[i]);
        }

        T sum = T.Zero;
        for (int j = 0; j < x.Length; j++)
        {
            T e = T.Exp(x[j] - max);                                         // no generic vector exp: T's own
            y[j] = e;
            sum += e;
        }

        T scale = T.One / sum;
        Scale(y, scale);
    }

    private static void Scale<T>(Span<T> y, T scale) where T : struct, INumberBase<T>
    {
        int i = 0;
        if (Vectorized<T>())
        {
            var yv = MemoryMarshal.Cast<T, Vector<T>>(y);
            var s = new Vector<T>(scale);
            for (int v = 0; v < yv.Length; v++)
            {
                yv[v] *= s;
            }

            i = yv.Length * Vector<T>.Count;
        }

        for (; i < y.Length; i++)
        {
            y[i] *= scale;
        }
    }

    // ------------------------------------------------------------------ rms_norm over rows

    public static void RmsNorm<T>(T[] x, T[] y, T[] inv, int rows, int cols, T eps) where T : struct, IFloatingPointIeee754<T>
    {
        T count = T.CreateTruncating(cols);
        Rows(rows, (long)rows * cols * 2, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                var row = x.AsSpan(r * cols, cols);
                T sum = SumSquares(row);
                T scale = T.One / T.Sqrt(sum / count + eps);
                inv[r] = scale;
                var output = y.AsSpan(r * cols, cols);
                row.CopyTo(output);
                Scale(output, scale);
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static T SumSquares<T>(ReadOnlySpan<T> x) where T : struct, INumberBase<T>
    {
        T sum = T.Zero;
        int i = 0;
        if (Vectorized<T>())
        {
            var acc = Vector<T>.Zero;
            var xv = MemoryMarshal.Cast<T, Vector<T>>(x);
            for (int v = 0; v < xv.Length; v++)
            {
                acc += xv[v] * xv[v];
            }

            sum = Vector.Sum(acc);
            i = xv.Length * Vector<T>.Count;
        }

        for (; i < x.Length; i++)
        {
            sum += x[i] * x[i];
        }

        return sum;
    }

    // ------------------------------------------------------------------ optimizer steps (OptimizerArgs), in the moments' type TM

    /// <summary>A step's scalars in TM: its kind (0 sgd, 1 sgd_momentum, 2 adam, 3 adamw), lr, b1, b2, eps, wd, the bias
    /// corrections c1 = 1 − b1ᵗ and c2 = 1 − b2ᵗ, and 1 − b1, 1 − b2.</summary>
    public readonly record struct Step<TM>(int Kind, TM Lr, TM B1, TM B2, TM Eps, TM Wd, TM C1, TM C2, TM B1c, TM B2c)
        where TM : struct, IFloatingPointIeee754<TM>
    {
        public static Step<TM> Of(OptimizerArgs a)
        {
            int kind = a.Name switch { "sgd" => 0, "sgd_momentum" => 1, "adam" => 2, "adamw" => 3, _ => throw new ArgumentException(a.Name) };
            TM b1 = TM.CreateTruncating(a.B1), b2 = TM.CreateTruncating(a.B2), t = TM.CreateTruncating(a.Step);
            return new(kind, TM.CreateTruncating(a.Lr), b1, b2, TM.CreateTruncating(a.Eps), TM.CreateTruncating(a.Wd),
                TM.One - TM.Pow(b1, t), TM.One - TM.Pow(b2, t), TM.One - b1, TM.One - b2);
        }
    }

    /// <summary>One step over p[offset, offset + n) and g, m, v there (m, v as the step keeps them), each gradient times
    /// scale as it is read: sgd p −= lr·(g + wd·p); sgd_momentum m = b1·m + g + wd·p, p −= lr·m; adam m = b1·m + (1 − b1)·g,
    /// v = b2·v + (1 − b2)·g², p −= lr·(m/c1)/(√(v/c2) + eps); adamw adam with p −= lr·wd·p first.</summary>
    public static void Optimizer<T, TM>(Step<TM> s, T[] p, T[] g, TM[]? m, TM[]? v, int offset, int n, TM scale)
        where T : struct, INumberBase<T> where TM : struct, IFloatingPointIeee754<TM> =>
        Split(n, n, (start, end) => StepRange(s, p, g, m, v, offset + start, offset + end, scale));

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void StepRange<T, TM>(Step<TM> s, T[] p, T[] g, TM[]? m, TM[]? v, int start, int end, TM scale)
        where T : struct, INumberBase<T> where TM : struct, IFloatingPointIeee754<TM>
    {
        int i = start;
        if (typeof(T) == typeof(TM) && Vectorized<TM>())
        {
            int count = end - start;
            var pv = MemoryMarshal.Cast<T, Vector<TM>>(p.AsSpan(start, count));
            var gv = MemoryMarshal.Cast<T, Vector<TM>>(g.AsSpan(start, count));
            var mv = s.Kind > 0 ? MemoryMarshal.Cast<TM, Vector<TM>>(m.AsSpan(start, count)) : default;
            var vv = s.Kind > 1 ? MemoryMarshal.Cast<TM, Vector<TM>>(v.AsSpan(start, count)) : default;
            Vector<TM> lr = new(s.Lr), b1 = new(s.B1), b2 = new(s.B2), eps = new(s.Eps), wd = new(s.Wd), c1 = new(s.C1), c2 = new(s.C2);
            Vector<TM> b1c = new(s.B1c), b2c = new(s.B2c), k = new(scale);
            switch (s.Kind)
            {
                case 0:
                    for (int j = 0; j < pv.Length; j++)
                    {
                        Vector<TM> pj = pv[j], gj = gv[j] * k;
                        pv[j] = pj - lr * (gj + wd * pj);
                    }

                    break;
                case 1:
                    for (int j = 0; j < pv.Length; j++)
                    {
                        Vector<TM> pj = pv[j], gj = gv[j] * k, mj = b1 * mv[j] + gj + wd * pj;
                        mv[j] = mj;
                        pv[j] = pj - lr * mj;
                    }

                    break;
                default:
                    for (int j = 0; j < pv.Length; j++)
                    {
                        Vector<TM> pj = pv[j], gj = gv[j] * k;
                        Vector<TM> mj = b1 * mv[j] + b1c * gj, vj = b2 * vv[j] + b2c * gj * gj;
                        mv[j] = mj;
                        vv[j] = vj;
                        var update = lr * (mj / c1) / (Vector.SquareRoot(vj / c2) + eps);
                        if (s.Kind == 3) pj -= lr * wd * pj;
                        pv[j] = pj - update;
                    }

                    break;
            }

            i = start + pv.Length * Vector<TM>.Count;
        }

        for (; i < end; i++)
        {
            TM pi = TM.CreateTruncating(p[i]), gi = TM.CreateTruncating(g[i]) * scale;
            switch (s.Kind)
            {
                case 0:
                    pi -= s.Lr * (gi + s.Wd * pi);
                    break;
                case 1:
                    TM mi = s.B1 * m![i] + gi + s.Wd * pi;
                    m[i] = mi;
                    pi -= s.Lr * mi;
                    break;
                default:
                    TM ma = s.B1 * m![i] + s.B1c * gi, va = s.B2 * v![i] + s.B2c * gi * gi;
                    m[i] = ma;
                    v[i] = va;
                    TM update = s.Lr * (ma / s.C1) / (TM.Sqrt(va / s.C2) + s.Eps);
                    if (s.Kind == 3) pi -= s.Lr * s.Wd * pi;
                    pi -= update;
                    break;
            }

            p[i] = T.CreateTruncating(pi);
        }
    }

    /// <summary>Σ g² over g[0, n) in TM: chunks as Split's (each chunk's sum in TM, by Vector&lt;TM&gt; where T is TM), added in order.</summary>
    public static TM SumSquares<T, TM>(T[] g, int n) where T : struct, INumberBase<T> where TM : struct, IFloatingPointIeee754<TM>
    {
        int chunks = n < SplitWork ? 1 : Math.Max(1, Math.Min(Threads * 2, n / Math.Max(1, ParallelElements / 4)));
        int size = (n + chunks - 1) / chunks;
        var partial = new TM[chunks];
        Parallel.For(0, chunks, Options, c =>
        {
            int start = c * size;
            if (start < n)
            {
                var range = g.AsSpan(start, Math.Min(start + size, n) - start);
                if (typeof(T) == typeof(TM))
                {
                    partial[c] = SumSquares<TM>(MemoryMarshal.Cast<T, TM>(range));
                }
                else
                {
                    TM sum = TM.Zero;
                    foreach (T x in range)
                    {
                        TM y = TM.CreateTruncating(x);
                        sum += y * y;
                    }

                    partial[c] = sum;
                }
            }
        });
        TM total = TM.Zero;
        foreach (TM x in partial) total += x;
        return total;
    }

    // ------------------------------------------------------------------ rope by tables (RopeArgs), in the tables' type TL

    /// <summary>Row r at position positions[(r / Heads) % Steps] rotates pair p &lt; Half — (2p, 2p + 1) interleaved, else
    /// (p, p + Half) — by cos[pos·Half + p], sin[pos·Half + p] (−sin for the inverse); the columns past 2·Half copied.</summary>
    public static void Rope<T, TL>(RopeArgs a, T[] x, TL[] cos, TL[] sin, int[] positions, T[] y)
        where T : struct, INumberBase<T> where TL : struct, IFloatingPointIeee754<TL>
    {
        int cols = a.Cols;
        Rows(a.Rows, (long)a.Rows * cols * 2, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                RopeRow(a, x.AsSpan(r * cols, cols), cos, sin, positions[r / a.Heads % a.Steps], y.AsSpan(r * cols, cols));
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RopeRow<T, TL>(RopeArgs a, ReadOnlySpan<T> x, TL[] cos, TL[] sin, int pos, Span<T> y)
        where T : struct, INumberBase<T> where TL : struct, IFloatingPointIeee754<TL>
    {
        int half = a.Half, p = 0;
        var c = cos.AsSpan(pos * half, half);
        var s = sin.AsSpan(pos * half, half);
        if (!a.Interleaved && typeof(T) == typeof(TL) && Vectorized<TL>())
        {
            ReadOnlySpan<TL> lo = MemoryMarshal.Cast<T, TL>(x[..half]), hi = MemoryMarshal.Cast<T, TL>(x.Slice(half, half));
            Span<TL> ylo = MemoryMarshal.Cast<T, TL>(y[..half]), yhi = MemoryMarshal.Cast<T, TL>(y.Slice(half, half));
            for (int w = Vector<TL>.Count; p + w <= half; p += w)
            {
                Vector<TL> xl = new(lo[p..]), xh = new(hi[p..]), cv = new(c[p..]), sv = new(s[p..]);
                if (a.Inverse) sv = -sv;
                (xl * cv - xh * sv).CopyTo(ylo[p..]);
                (xl * sv + xh * cv).CopyTo(yhi[p..]);
            }
        }

        for (; p < half; p++)
        {
            int i = a.Interleaved ? 2 * p : p, j = a.Interleaved ? 2 * p + 1 : p + half;
            TL xi = TL.CreateTruncating(x[i]), xj = TL.CreateTruncating(x[j]), cp = c[p], sp = a.Inverse ? -s[p] : s[p];
            y[i] = T.CreateTruncating(xi * cp - xj * sp);
            y[j] = T.CreateTruncating(xi * sp + xj * cp);
        }

        x[(2 * half)..].CopyTo(y[(2 * half)..]);
    }

    // ------------------------------------------------------------------ box overlap losses (BoxOverlap) over pairs of corner boxes, in T

    /// <summary>The loss of each pair (dloss null) or its gradient with respect to the predicted corners times dloss[i],
    /// the targets constants. eps is T's precision (the step from 1 to the next T); at a tie of a min or a max the
    /// predicted corner takes the whole derivative.</summary>
    public static void BoxLoss<T>(BoxOverlap o, T[] p, T[] t, T[]? dloss, T[] y, int count) where T : struct, IFloatingPointIeee754<T>
    {
        T eps = T.BitIncrement(T.One) - T.One, aspect = T.CreateTruncating(4) / (T.Pi * T.Pi);
        Split(count, count * 32L, (start, end) =>
        {
            for (int i = start; i < end; i++)
            {
                if (dloss is null) y[i] = BoxPairLoss(o, p.AsSpan(4 * i, 4), t.AsSpan(4 * i, 4), eps, aspect);
                else BoxPairGrad(o, p.AsSpan(4 * i, 4), t.AsSpan(4 * i, 4), eps, aspect, dloss[i], y.AsSpan(4 * i, 4));
            }
        });
    }

    // IoU = inter / (union + eps); loss 1 − IoU, GIoU + (|C| − union) / (|C| + eps), DIoU + ρ² / (diag² + eps), CIoU + α·v.
    private static T BoxPairLoss<T>(BoxOverlap o, ReadOnlySpan<T> p, ReadOnlySpan<T> t, T eps, T aspect) where T : struct, IFloatingPointIeee754<T>
    {
        T x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3], gx1 = t[0], gy1 = t[1], gx2 = t[2], gy2 = t[3];
        T iw = T.Min(x2, gx2) - T.Max(x1, gx1), ih = T.Min(y2, gy2) - T.Max(y1, gy1);
        T inter = iw > T.Zero && ih > T.Zero ? iw * ih : T.Zero;
        T wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
        T union = wp * hp + wg * hg - inter, iou = inter / (union + eps), loss = T.One - iou;
        if (o == BoxOverlap.IoU) return loss;
        T cw = T.Max(x2, gx2) - T.Min(x1, gx1), ch = T.Max(y2, gy2) - T.Min(y1, gy1);
        if (o == BoxOverlap.GIoU)
        {
            T area = cw * ch;
            return loss + (area - union) / (area + eps);
        }

        T half = T.CreateTruncating(0.5), diag = cw * cw + ch * ch + eps;
        T dx = ((x1 - gx1) + (x2 - gx2)) * half, dy = ((y1 - gy1) + (y2 - gy2)) * half;
        loss += (dx * dx + dy * dy) / diag;
        if (o == BoxOverlap.DIoU) return loss;
        T diff = T.Atan(wg / hg) - T.Atan(wp / hp), v = aspect * diff * diff, alpha = v / (T.One - iou + v + eps);
        return loss + alpha * v;
    }

    // The loss's derivative with respect to the predicted corners, its parts chained: the intersection, the union, the
    // enclosing box's sides (cw, ch), the centres' offset (dPx, dPy), the width and height (dW, dH).
    private static void BoxPairGrad<T>(BoxOverlap o, ReadOnlySpan<T> p, ReadOnlySpan<T> t, T eps, T aspect, T dl, Span<T> grad) where T : struct, IFloatingPointIeee754<T>
    {
        T zero = T.Zero, one = T.One, two = one + one, half = T.CreateTruncating(0.5);
        T x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3], gx1 = t[0], gy1 = t[1], gx2 = t[2], gy2 = t[3];
        T iw = T.Min(x2, gx2) - T.Max(x1, gx1), ih = T.Min(y2, gy2) - T.Max(y1, gy1);
        bool overlaps = iw > zero && ih > zero;
        T inter = overlaps ? iw * ih : zero;
        T wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
        T union = wp * hp + wg * hg - inter, ue = union + eps, iou = inter / ue;
        T cw = T.Max(x2, gx2) - T.Min(x1, gx1), ch = T.Max(y2, gy2) - T.Min(y1, gy1);
        T dInter = -one / ue, dUnion = iou / ue, dCw = zero, dCh = zero, dPx = zero, dPy = zero, dW = zero, dH = zero;
        if (o == BoxOverlap.GIoU)
        {
            T area = cw * ch, ae = area + eps;
            dUnion -= one / ae;
            T dArea = ue / (ae * ae);
            dCw = dArea * ch;
            dCh = dArea * cw;
        }
        else if (o is BoxOverlap.DIoU or BoxOverlap.CIoU)
        {
            T diag = cw * cw + ch * ch + eps;
            T dx = ((x1 - gx1) + (x2 - gx2)) * half, dy = ((y1 - gy1) + (y2 - gy2)) * half, rho = dx * dx + dy * dy;
            dPx = two * dx / diag;
            dPy = two * dy / diag;
            T dDiag = -(rho / (diag * diag));
            dCw = two * dDiag * cw;
            dCh = two * dDiag * ch;
            if (o == BoxOverlap.CIoU)
            {
                T diff = T.Atan(wg / hg) - T.Atan(wp / hp), v = aspect * diff * diff, alpha = v / (one - iou + v + eps);
                T held = one - alpha * alpha;                                 // ∂(α·v)/∂IoU = α², ∂(α·v)/∂v = α·(2 − α)
                dInter *= held;
                dUnion *= held;
                T dv = alpha * (two - alpha), r2 = wp * wp + hp * hp, k = two * aspect * dv * diff / r2;
                dW = -(k * hp);
                dH = k * wp;
            }
        }

        dInter -= dUnion;                                                     // the union holds minus the intersection
        dW += dUnion * hp;
        dH += dUnion * wp;
        T ix = overlaps ? ih * dInter : zero, iy = overlaps ? iw * dInter : zero;
        grad[0] = (-dW + dPx * half - ix * Above(x1, gx1) - dCw * Below(x1, gx1)) * dl;
        grad[1] = (-dH + dPy * half - iy * Above(y1, gy1) - dCh * Below(y1, gy1)) * dl;
        grad[2] = (dW + dPx * half + ix * Below(x2, gx2) + dCw * Above(x2, gx2)) * dl;
        grad[3] = (dH + dPy * half + iy * Below(y2, gy2) + dCh * Above(y2, gy2)) * dl;
    }

    // The share of max(a, b)'s derivative a takes (min's: Below): all where max (min) gives a, a tie included (the
    // operation's definition: the predicted corner, the first operand, takes the whole gradient at a tie), else none.
    private static T Above<T>(T a, T b) where T : struct, IFloatingPointIeee754<T> => a >= b ? T.One : T.Zero;

    private static T Below<T>(T a, T b) where T : struct, IFloatingPointIeee754<T> => a <= b ? T.One : T.Zero;

    // ------------------------------------------------------------------ ctc (CtcArgs): a sequence per Parallel.For iteration, in T

    /// <summary>The loss of each sequence b (and with grad the gradient with respect to the logits): log_softmax per frame
    /// (each frame's log-sum-exp kept in lse[b]), α (and β) in log space over its 2·L + 1 extended labels (blanks at the
    /// even states) in alpha[b] (beta[b]), frames × states; loss +∞ and gradient 0 where no alignment fits, the gradient 0
    /// past the sequence's frames. offsets[b]: its first label.</summary>
    public static void Ctc<T>(CtcArgs a, int[] offsets, T[] x, T[] loss, T[]? grad, T[][] alpha, T[][]? beta, T[][] lse) where T : struct, IFloatingPointIeee754<T> =>
        Parallel.For(0, a.B, Options, b => CtcSequence(a, b, offsets[b], x, loss, grad, alpha[b], beta?[b], lse[b]));

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void CtcSequence<T>(CtcArgs a, int b, int offset, T[] x, T[] loss, T[]? grad, T[] alpha, T[]? beta, T[] lse) where T : struct, IFloatingPointIeee754<T>
    {
        int frames = a.InputLengths[b], S = 2 * a.LabelLengths[b] + 1, B = a.B, C = a.C;
        T none = T.NegativeInfinity;
        for (int t = 0; t < frames; t++)
        {
            var row = x.AsSpan((t * B + b) * C, C);
            T m = Max(row), sum = T.Zero;
            foreach (T v in row) sum += T.Exp(v - m);
            lse[t] = m + T.Log(sum);
        }

        int Class(int s) => (s & 1) == 0 ? a.Blank : a.Labels[offset + (s >> 1)];
        bool Skip(int s) => s >= 2 && Class(s) != a.Blank && Class(s) != Class(s - 2);
        T LogY(int t, int s) => x[(t * B + b) * C + Class(s)] - lse[t];
        for (int s = 0; s < S; s++) alpha[s] = s < 2 ? LogY(0, s) : none;
        for (int t = 1; t < frames; t++)
        {
            int at = t * S, before = at - S;
            for (int s = 0; s < S; s++)
            {
                alpha[at + s] = Lse(alpha[before + s], s >= 1 ? alpha[before + s - 1] : none, Skip(s) ? alpha[before + s - 2] : none) + LogY(t, s);
            }
        }

        int last = (frames - 1) * S;
        T logP = Lse(alpha[last + S - 1], S >= 2 ? alpha[last + S - 2] : none, none);
        loss[b] = -logP;
        if (grad is null) return;
        for (int t = 0; t < a.T; t++) grad.AsSpan((t * B + b) * C, C).Clear();
        if (T.IsNegativeInfinity(logP)) return;                              // no alignment: the gradient stays 0
        for (int s = 0; s < S; s++) beta![last + s] = s >= S - 2 ? LogY(frames - 1, s) : none;
        for (int t = frames - 2; t >= 0; t--)
        {
            int at = t * S, after = at + S;
            for (int s = 0; s < S; s++)
            {
                beta![at + s] = Lse(beta[after + s], s + 1 < S ? beta[after + s + 1] : none, s + 2 < S && Skip(s + 2) ? beta[after + s + 2] : none) + LogY(t, s);
            }
        }

        for (int t = 0; t < frames; t++)
        {
            int row = (t * B + b) * C, at = t * S;
            for (int c = 0; c < C; c++) grad[row + c] = T.Exp(x[row + c] - lse[t]);
            for (int s = 0; s < S; s++) grad[row + Class(s)] -= T.Exp(alpha[at + s] + beta![at + s] - LogY(t, s) - logP);
        }
    }

    // log(eᵃ + eᵇ + eᶜ) by the largest: −∞ when all are.
    private static T Lse<T>(T a, T b, T c) where T : struct, IFloatingPointIeee754<T>
    {
        T m = T.Max(a, T.Max(b, c));
        return T.IsNegativeInfinity(m) ? m : m + T.Log(T.Exp(a - m) + T.Exp(b - m) + T.Exp(c - m));
    }

    private static T Max<T>(ReadOnlySpan<T> x) where T : struct, IFloatingPointIeee754<T>
    {
        T max = T.NegativeInfinity;
        int i = 0;
        if (Vectorized<T>() && x.Length >= Vector<T>.Count)
        {
            var mv = new Vector<T>(T.NegativeInfinity);
            var xv = MemoryMarshal.Cast<T, Vector<T>>(x);
            for (int v = 0; v < xv.Length; v++) mv = Vector.Max(mv, xv[v]);
            for (int l = 0; l < Vector<T>.Count; l++) max = T.Max(max, mv[l]);
            i = xv.Length * Vector<T>.Count;
        }

        for (; i < x.Length; i++) max = T.Max(max, x[i]);
        return max;
    }
}
