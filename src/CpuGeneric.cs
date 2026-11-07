// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The generic CPU side: one implementation per operation over T : INumberBase<T> (softmax and rms_norm need exp and sqrt:
// T : IFloatingPointIeee754<T>). Each runs Vector<T> where the runtime supports T (every primitive number; not Half) and
// T's own operators otherwise, split across threads by Parallel.For with simple cut-overs, so only the kernel differs.
// Sums are in T (no wider type), the generic code's natural choice: what that does to small types is part of the result.
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
}
