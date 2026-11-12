// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The convolution (Abstraction/Conv.cs) as the composition of Adad's own operations: the image to columns (im2col, exact
// in any type), the product with the weights (the product kernel: tensor and dot lanes by the bounds), and back (col2im
// for the input's gradient). One image of one group (C × H × W; the weights OC × C·Kh·Kw; the output OC × OH × OW):
//
//   conv           y = w · im2col(x)
//   conv_bwd_x     dx = col2im(wᵀ · dy)
//   conv_bwd_w     dw = dy · im2col(x)ᵀ
//
// The forward is fused (item 4g, implicit GEMM): the product's packing builds each row of the columns from the image
// itself, so no columns are in memory — the same packed values as the composition, so the same result bit for bit
// (CPU: every type whose rows start on whole bytes; the device: GpuOps.Conv.cs). The backward keeps the composition for
// dx; on the CPU dw's columns are read from the image the same way (the transposed packing, a k-range of a row a task).
// The depthwise convolution is a layout operation (Layout.cs: a gather with weights).
//
// Images, groups, a bias and an activation (ConvArgs; the float types) take the general path:
// the columns of every image's channels into the float lane's storage (exact), each image's and group's product with the
// group's weights into the lane's sums, then y = act(sums + bias[f]) stored as the type once (Epilogue.cs ProductStore). Its backward:
// dcols = w_gᵀ·dy_g in the lane, dx = col2im(dcols); dy and the columns laid side by side over the images (exact moves into
// the lane), dw_g = dy_g · cols_gᵀ in one product a group, stored once; db the lane's sums of dy's rows.

using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>The gate's override: false takes the composition (im2col, then the product), null the fused path where it applies.</summary>
    internal static bool? FusedConv;

    // The fused path reads A's rows, the image's rows and C at value indices: each on whole bytes.
    private static bool ConvInPlace<TN>(ConvArgs a) where TN : struct, IAdad<TN>
    {
        int per = TN.Bits >= 8 ? 1 : 8 / TN.Bits;
        return a.XValues <= int.MaxValue && a.YValues <= int.MaxValue && a.WValues <= int.MaxValue
            && (per == 1 || (a.OC % per == 0 && a.Rows % per == 0 && a.Cols % per == 0 && (TN.Granularity % 32 == 0 || a.W % per == 0)));
    }

    /// <summary>y = act(conv(x, w) + bias) on the CPU; returns the product's lane.</summary>
    public static string Conv<TN>(ConvArgs a, byte[] x, byte[] w, byte[]? bias, byte[] y) where TN : struct, IAdad<TN>
    {
        a.Validate();
        if (!a.Single || bias is not null)
        {
            var general = new MixedVisitor<ConvGeneralCall>(new ConvGeneralCall(a, x, w, bias, y));
            if (!TN.TryFloat(ref general)) throw new NotSupportedException($"conv: images, groups, a bias or an activation take a float type, not {TN.Name}");
            return general.Lane;
        }

        return Conv<TN>(a, x, w, y);
    }

    /// <summary>y = conv(x, w) of one image of one group; returns the product's lane.</summary>
    public static string Conv<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) where TN : struct, IAdad<TN>
    {
        if (FusedConv != false && ConvInPlace<TN>(a))
        {
            // The columns' values are the image's and 0 (padding): the image's range, with 0, bounds them.
            var (wLo, wHi) = TN.IsFloat ? (0, 0) : TN.Range(w, (int)a.WValues);              // a float type's lane does not depend on them
            var (xLo, xHi) = TN.IsFloat ? (0, 0) : TN.Range(x, (int)a.XValues);
            var product = new MatMulVisitor(w, x, y, a.OC, a.Cols, a.Rows) { Where = new Operands(ImageB: a) };
            TN.Sum(ref product, new Bounds(wLo, wHi, Math.Min(xLo, 0), Math.Max(xHi, 0), a.Rows));
            return product.Lane;
        }

        return ConvComposed<TN>(a, x, w, y);
    }

    /// <summary>y = conv(x, w) by the composition: im2col, then the product.</summary>
    private static string ConvComposed<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) where TN : struct, IAdad<TN>
    {
        var cols = new byte[BytesOf<TN>((long)a.Rows * a.Cols)];
        Layout<TN>(new LayoutArgs { Name = "im2col", X = x, P = a.P, Output = cols, Count = a.Rows * a.Cols, InputCount = (int)a.XValues });
        return MatMulView<TN>(new ProductView(a.OC, a.Cols, a.Rows, false, false), w, cols, y);
    }

    /// <summary>dx and dw from dy: x, w, dy read where they are stored, dx and dw stored as the type; the columns'
    /// gradient (a sum of products the type cannot hold between the two passes) an intermediate in the float lane's own
    /// type; returns the lane's name.</summary>
    public static string ConvBackward<TN>(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw, byte[]? db = null) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        if (!a.Single || db is not null) return Mixed<TN, ConvBackwardGeneralCall>(new(a, x, w, dy, dx, dw, db));
        return Mixed<TN, ConvBackwardCall>(new(a, x, w, dy, dx, dw));
    }

    // The general forward: the columns of every image's channels in the lane (image n group g's at rows (n·G + g)·Rows), a
    // product an image and group into the lane's sums, then the store with the bias and the activation.
    private readonly struct ConvGeneralCall(ConvArgs a, byte[] x, byte[] w, byte[]? bias, byte[] y) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.YValues == 0) return TL.Name;
            a.ColumnsFit();
            int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>(), G = a.Groups;
            var cols = new byte[a.ColValues * size];
            var sums = new byte[a.YValues * size];
            LayoutToLanes<TN, TLane, TL>(new LayoutArgs { Name = "im2col", X = x, P = a.ImagesP, Output = cols, Count = (int)a.ColValues, InputCount = (int)a.XValues });
            for (int n = 0; n < a.N; n++)
                for (int g = 0; g < G; g++)
                    MatMul<TN, TS, TS, TLane, TLane, TL>(w, cols, sums, a.OCg, a.Cols, a.Rows,
                        new Operands(AOff: g * a.OCg * a.Rows, BOff: (n * G + g) * a.Rows * a.Cols, COff: (n * a.OC + g * a.OCg) * a.Cols));
            ProductStore<TN, TLane, TL>(new StoreTerms(false, bias is not null, false, a.Act, null, false), new StoreSources(sums, null, bias, null, null, a.Cols, a.OC, a.ActA, 0), y, (int)a.YValues);
            return TL.Name;
        }
    }

    // The general backward (see the file's note).
    private readonly struct ConvBackwardGeneralCall(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw, byte[]? db) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            a.ColumnsFit();
            int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>(), G = a.Groups, R = a.Rows, P = a.Cols, NP = a.N * a.Cols;
            var cols = new byte[a.ColValues * size];
            var dcols = new byte[a.ColValues * size];
            LayoutToLanes<TN, TLane, TL>(new LayoutArgs { Name = "im2col", X = x, P = a.ImagesP, Output = cols, Count = (int)a.ColValues, InputCount = (int)a.XValues });
            for (int n = 0; n < a.N; n++)
                for (int g = 0; g < G; g++)
                    MatMul<TN, TN, TS, TLane, TLane, TL>(w, dy, dcols, R, P, a.OCg,
                        new Operands(AOff: g * a.OCg * R, TransA: true, BOff: (n * a.OC + g * a.OCg) * P, COff: (n * G + g) * R * P));      // w_gᵀ · dy_g
            LayoutFromLanes<TN, TLane, TL>(new LayoutArgs { Name = "col2im", X = dcols, P = a.ImagesP, Output = dx, Count = (int)a.XValues, InputCount = (int)a.ColValues });
            // dy as (OC, N·P) and the columns as (G·R, N·P): every image's positions side by side, exact moves into the lane.
            var dyp = new byte[(long)a.OC * NP * size];
            var colsp = new byte[a.ColValues * size];
            LayoutToLanes<TN, TLane, TL>(new LayoutArgs { Name = "strided", X = dy, P = [1, a.OC, a.N, P, 0, P, a.OC * P, 1, 0], Output = dyp, Count = a.OC * NP, InputCount = (int)a.YValues });
            LayoutToLanes<TS, TLane, TL>(new LayoutArgs { Name = "strided", X = cols, P = [1, G * R, a.N, P, 0, P, G * R * P, 1, 0], Output = colsp, Count = (int)a.ColValues, InputCount = (int)a.ColValues });
            for (int g = 0; g < G; g++)
                MatMul<TS, TS, TN, TLane, TLane, TL>(dyp, colsp, dw, a.OCg, R, NP, new Operands(AOff: g * a.OCg * NP, BOff: g * R * NP, TransB: true, COff: g * a.OCg * R));   // dy_g · cols_gᵀ
            if (db is not null)
            {
                var rows = MemoryMarshal.Cast<byte, TLane>(dyp.AsSpan());
                var sums = new TLane[a.OC];
                for (int f = 0; f < a.OC; f++) sums[f] = RowMath<TLane, TL>.Sum(rows.Slice(f * NP, NP));
                TN.Store(sums, db, 0);
            }

            return TL.Name;
        }
    }

    // dx = col2im(wᵀ · dy), the columns' gradient (TS) in a buffer the thread keeps (exact length, every value written by
    // the product), col2im gathering it in the lane and storing dx as the type; dw = dy · colsᵀ with colsᵀ read from the
    // image by the transposed packing (the implicit im2col: the same packed values as the columns', so the same result bit
    // for bit), or the columns themselves (im2col into the lane's own type) where that does not apply.
    private readonly struct ConvBackwardCall(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            long values = (long)a.Rows * a.Cols;
            var dcols = Scratch.Get(Scratch.ConvDcols, values * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>());
            MatMul<TN, TN, TS, TLane, TLane, TL>(w, dy, dcols, a.Rows, a.Cols, a.OC, new Operands(TransA: true));    // wᵀ · dy
            LayoutFromLanes<TN, TLane, TL>(new LayoutArgs { Name = "col2im", X = dcols, P = a.P, Output = dx, Count = (int)a.XValues, InputCount = a.Rows * a.Cols });
            if (FusedConv != false && ConvInPlace<TN>(a))
            {
                MatMul<TN, TN, TN, TLane, TLane, TL>(dy, x, dw, a.OC, a.Rows, a.Cols, new Operands(TransB: true, ImageB: a));
                return TL.Name;
            }

            var cols = Scratch.Get(Scratch.ConvCols, values * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>());
            LayoutToLanes<TN, TLane, TL>(new LayoutArgs { Name = "im2col", X = x, P = a.P, Output = cols, Count = a.Rows * a.Cols, InputCount = (int)a.XValues });
            MatMul<TN, TS, TN, TLane, TLane, TL>(dy, cols, dw, a.OC, a.Rows, a.Cols, new Operands(TransB: true));     // dy · colsᵀ
            return TL.Name;
        }
    }
}
