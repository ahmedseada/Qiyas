// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Block-scaled integer products: the block formats' own fast path. A row of Q8_0 (or Q4_0) against a vector of the same
// format is Σ over blocks of d_a·d_x·(Σ_i q_a,i · q_x,i): the inner sum is an integer dot of 32 products (exact in s32,
// the s8 dot lane where the CPU has it), the outer sum the float lane. Nothing is decoded to a float per value.
//
//   gemv (y = A·x) for Q8_0 and Q4_0 operands: a thread (CPU: a row at a time on the team; device: a thread per row)

using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>y = A·x for A (rows × cols) and x (cols) both Q8_0 or both Q4_0 (cols a multiple of 32); y as float32 lanes. Returns the lane's name.</summary>
    public static string GemvBlocks(string format, byte[] a, byte[] x, float[] y, int rows, int cols) =>
        GemvBlocks(Routes.Layout(format), a, x, y, rows, cols);

    /// <summary>y = A·x for scaled-int blocks of the layout <paramref name="format"/> (cols a multiple of 32); y as float32 lanes.</summary>
    public static string GemvBlocks(ScaledBlock format, byte[] a, byte[] x, float[] y, int rows, int cols)
    {
        if (cols % 32 != 0) throw new ArgumentException("the block product takes columns in whole blocks of 32");
        int blocks = cols / 32, blockBytes = format.Bytes;
        // x's blocks decoded once into scales and integer values (int8 for both formats: Q4_0's q − 8).
        var xs = new float[blocks];
        var xq = new sbyte[cols];
        for (int b = 0; b < blocks; b++) UnpackBlock(format, x.AsSpan(b * blockBytes, blockBytes), out xs[b], xq.AsSpan(b * 32, 32));
        Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols, new GemvBlockBody(format, a, xs, xq, y, cols, blockBytes));
        return S8Dot.IsSupported ? "s8dp blocks" : "s32 blocks";
    }

    /// <summary>A block's scale and its 32 integer values (Q4_0: q − 8) as int8.</summary>
    public static void UnpackBlock(string format, ReadOnlySpan<byte> block, out float scale, Span<sbyte> q) =>
        UnpackBlock(Routes.Layout(format), block, out scale, q);

    /// <summary>A block's scale and its 32 integer values (code + the layout's bias) as int8, by the layout.</summary>
    public static void UnpackBlock(ScaledBlock format, ReadOnlySpan<byte> block, out float scale, Span<sbyte> q)
    {
        scale = (float)BitConverter.ToHalf(block[..2]);
        if (format.CodeBits == 8) { for (int i = 0; i < 32; i++) q[i] = (sbyte)((sbyte)block[2 + i] + format.CodeBias); return; }
        for (int i = 0; i < 16; i++) { q[i] = (sbyte)((block[2 + i] & 0xF) + format.CodeBias); q[i + 16] = (sbyte)((block[2 + i] >> 4) + format.CodeBias); }
    }

    private struct GemvBlockBody(ScaledBlock format, byte[] a, float[] xs, sbyte[] xq, float[] y, int cols, int blockBytes) : Team.IBody
    {
        public readonly void Run(int first, int last)
        {
            int blocks = cols / 32;
            Span<sbyte> q = stackalloc sbyte[32];
            for (int r = first; r < last; r++)
            {
                float acc = 0;
                for (int b = 0; b < blocks; b++)
                {
                    UnpackBlock(format, a.AsSpan((int)(((long)r * blocks + b) * blockBytes), blockBytes), out float d, q);
                    int dot = Dot32(q, xq.AsSpan(b * 32, 32));
                    acc += d * xs[b] * dot;
                }

                y[r] = acc;
            }
        }

        // The 32-term integer dot: the s8 dot lane over 4-byte groups where the CPU has it, else plain integers.
        private static int Dot32(ReadOnlySpan<sbyte> p, ReadOnlySpan<sbyte> q)
        {
            if (S8Dot.IsSupported && Vector<int>.Count == 8)
            {
                var pv = new Vector<int>(MemoryMarshal.Cast<sbyte, int>(p));
                var qv = new Vector<int>(MemoryMarshal.Cast<sbyte, int>(q));
                return Vector.Sum(S8Dot.Fma(pv, qv, Vector<int>.Zero));
            }

            int sum = 0;
            for (int i = 0; i < 32; i++) sum += p[i] * q[i];
            return sum;
        }
    }
}
