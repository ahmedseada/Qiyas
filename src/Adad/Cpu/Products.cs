// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Products on views: C = op(A)·op(B) over batches, where op transposes a stored operand, and each batch is at a
// stride in values; the matmul backward passes dA = dC·Bᵀ and dB = Aᵀ·dC. The product kernel reads a view's operands
// where they are stored — Aᵀ's k-rows straight into its A blocks, Bᵀ's rows into its packed strips, a batch's offset as
// a base index (Kernels.Operands) — for every type whose rows start on whole bytes; otherwise a view is made contiguous by
// the strided copy (Layout.cs: exact in any type, any offset) first. The same lanes, the same bounds rule, the same proofs.

using System.Buffers;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    private static int BytesOf<TN>(long values) where TN : struct, IAdad<TN> => (int)TN.Bytes(values);

    /// <summary>C = op(A)·op(B) over the view's batches on the CPU; returns the product's lane. A·Bᵀ of two Q8_0 / Q4_0
    /// operands with k in whole blocks takes the block-scaled product where it measures faster (MatMulBlocks.cs).</summary>
    public static string MatMulView<TN>(ProductView v, byte[] a, byte[] b, byte[] c) where TN : struct, IAdad<TN>
    {
        if (BlockView<TN>(v) && UseBlocks<TN>(v, a, b, c)) return MatMulViewBlocks<TN>(v, a, b, c);
        return MatMulViewRoute<TN>(v, a, b, c);
    }

    /// <summary>Whether the product kernel can read the view's operands where they are stored: every row on a whole byte
    /// (a type of a byte or more, or a block format, read a value at a time, at even shapes), every batch offset on
    /// whole storage units.</summary>
    private static bool InPlace<TN>(ProductView v) where TN : struct, IAdad<TN>
    {
        int per = TN.Bits >= 8 ? 1 : 8 / TN.Bits, unit = TN.Granularity;
        bool rows = per == 1 || (unit % 32 == 0 && v.M % per == 0 && v.N % per == 0 && v.K % per == 0);
        return rows && v.StrideAOrDefault % unit == 0 && v.StrideBOrDefault % unit == 0 && (v.Batch == 1 || v.StrideCOrDefault % unit == 0)
            && v.ValuesA <= int.MaxValue && v.ValuesB <= int.MaxValue && v.ValuesC <= int.MaxValue;
    }

    /// <summary>C = op(A)·op(B): the product kernel reading each batch's operands where they are stored (transposed or
    /// at an offset; Kernels.Operands), or, where it cannot, by the strided copies.</summary>
    private static string MatMulViewRoute<TN>(ProductView v, byte[] a, byte[] b, byte[] c) where TN : struct, IAdad<TN>
    {
        if (v.Batch == 1 && !v.TransA && !v.TransB || !InPlace<TN>(v)) return MatMulViewCopies<TN>(v, a, b, c);
        string lane = "";
        for (int i = 0; i < v.Batch; i++)
        {
            int ao = (int)(i * v.StrideAOrDefault), bo = (int)(i * v.StrideBOrDefault), co = (int)(i * v.StrideCOrDefault);
            var (aLo, aHi) = TN.IsFloat ? (0, 0) : TN.Range(a.AsSpan((int)TN.Bytes(ao)), v.M * v.K);     // a float type's lane does not depend on them
            var (bLo, bHi) = TN.IsFloat ? (0, 0) : TN.Range(b.AsSpan((int)TN.Bytes(bo)), v.K * v.N);
            var product = new MatMulVisitor(a, b, c, v.M, v.N, v.K) { Where = new Operands(ao, v.TransA, bo, v.TransB, co) };
            TN.Sum(ref product, new Bounds(aLo, aHi, bLo, bHi, v.K));
            lane = product.Lane;
        }

        return lane;
    }

    /// <summary>C = op(A)·op(B) by the strided copies and the product kernel.</summary>
    private static string MatMulViewCopies<TN>(ProductView v, byte[] a, byte[] b, byte[] c) where TN : struct, IAdad<TN>
    {
        string lane = "";
        bool direct = v.Batch == 1 && !v.TransA && !v.TransB;
        var ca = direct ? a : new byte[BytesOf<TN>((long)v.M * v.K)];
        var cb = direct ? b : new byte[BytesOf<TN>((long)v.K * v.N)];
        var cc = direct ? c : new byte[BytesOf<TN>((long)v.M * v.N)];
        for (int i = 0; i < v.Batch; i++)
        {
            if (!direct)
            {
                Layout<TN>(new LayoutArgs { Name = "strided", X = a, P = v.ViewA(i), Output = ca, Count = v.M * v.K, InputCount = (int)v.ValuesA });
                Layout<TN>(new LayoutArgs { Name = "strided", X = b, P = v.ViewB(i), Output = cb, Count = v.K * v.N, InputCount = (int)v.ValuesB });
            }

            var (aLo, aHi) = TN.IsFloat ? (0, 0) : TN.Range(ca, v.M * v.K);
            var (bLo, bHi) = TN.IsFloat ? (0, 0) : TN.Range(cb, v.K * v.N);
            var product = new MatMulVisitor(ca, cb, cc, v.M, v.N, v.K);
            TN.Sum(ref product, new Bounds(aLo, aHi, bLo, bHi, v.K));
            lane = product.Lane;
            if (!direct) CopyValues<TN>(cc, 0, c, i * v.StrideCOrDefault, (long)v.M * v.N);
        }

        return lane;
    }

    /// <summary>dA = dC·Bᵀ (m × k) and dB = Aᵀ·dC (k × n) for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public static string MatMulBackward<TN>(byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }

    /// <summary>count values of src from srcOffset into dst at dstOffset (any offsets, through the lane).</summary>
    public static void CopyValues<TN>(byte[] src, long srcOffset, byte[] dst, long dstOffset, long count) where TN : struct, IAdad<TN>
    {
        var v = new CopyVisitor(src, srcOffset, dst, dstOffset, count);
        TN.Elementwise(ref v);
    }

    private struct CopyVisitor(byte[] src, long srcOffset, byte[] dst, long dstOffset, long count) : ILaneVisitor
    {
        public readonly void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            var buffer = ArrayPool<TLane>.Shared.Rent(4096);
            try
            {
                for (long at = 0; at < count; at += 4096)
                {
                    int len = (int)Math.Min(4096, count - at);
                    TN.Load(src, (int)(srcOffset + at), buffer.AsSpan(0, len));
                    TN.Store(buffer.AsSpan(0, len), dst, (int)(dstOffset + at));
                }
            }
            finally
            {
                ArrayPool<TLane>.Shared.Return(buffer);
            }
        }
    }
}
