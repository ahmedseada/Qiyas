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
// Images, groups or any term of the product's epilogue (ConvArgs.Output; the float types) take the general path: the
// columns of every image's channels into the float lane's storage (exact), each image's and group's product with the
// group's weights into the lane's sums, then the epilogue's store over the output, the bias a value an output channel, y
// stored as the type once (Epilogue.cs EpilogueStore). Its backward:
// dcols = w_gᵀ·dy_g in the lane, dx = col2im(dcols); dy and the columns laid side by side over the images (exact moves into
// the lane), dw_g = dy_g · cols_gᵀ in one product a group, stored once; db the lane's sums of dy's rows.
//
// The backward takes every type: dx's sums (w·dy, OCg·Kh·Kw terms a value at most) in the lane TN.Sum gives for their
// bounds, dw's and db's (dy·x and dy·1 over the images' positions) in the lane for theirs: a float type's float lane, an
// integer type's exact one (a dot lane for the products of TN's own operands where the number picks one). The
// intermediates (dcols, the columns, dy side by side) are the lane's own stored type (ILane.Kept's TS), products and sums
// over them in its plain lane (TP); each gradient stored as the type once.

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

    /// <summary>y = the epilogue's y of conv(x, w) on the CPU (IConvolution.Conv); returns the product's lane.</summary>
    public static string Conv<TN>(ConvArgs a, ProductEpilogue e, byte[] x, byte[] w, byte[] y, ProductTerms<byte[]> t) where TN : struct, IAdad<TN>
    {
        a.Validate();
        var terms = t.Validate(e, a.Output);
        if (a.Single && terms == default) return Conv<TN>(a, x, w, y);
        var general = new MixedVisitor<ConvGeneralCall>(new ConvGeneralCall(a, e, terms, x, w, y, t));
        if (!TN.TryFloat(ref general)) throw new NotSupportedException($"conv: images, groups or an epilogue term take a float type, not {TN.Name}");
        return general.Lane;
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

    /// <summary>dx and dw (and db) from dy, every type (see the file's note): x, w, dy read where they are stored, each gradient
    /// stored as the type once; returns the lanes' names (dx's; dw's after it where it differs).</summary>
    public static string ConvBackward<TN>(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw, byte[]? db = null) where TN : struct, IAdad<TN>
    {
        a.Validate();
        bool general = !a.Single || db is not null;
        if (general) a.ColumnsFit();
        // The bounds (a float type's lane does not depend on them).
        var (wLo, wHi) = TN.IsFloat ? (0, 0) : TN.Range(w, (int)a.WValues);
        var (xLo, xHi) = TN.IsFloat ? (0, 0) : TN.Range(x, (int)a.XValues);
        var (dLo, dHi) = TN.IsFloat ? (0, 0) : TN.Range(dy, (int)a.YValues);
        var forX = new BackwardVisitor<ConvDx>(new ConvDx(a, w, dy, dx, general));
        TN.Sum(ref forX, new Bounds(wLo, wHi, dLo, dHi, (long)a.OCg * a.Kh * a.Kw));
        var forW = new BackwardVisitor<ConvDw>(new ConvDw(a, x, dy, dw, db, general));
        TN.Sum(ref forW, new Bounds(dLo, dHi, Math.Min(xLo, 0), Math.Max(xHi, 1), (long)a.N * a.Cols));
        return forX.Lane == forW.Lane ? forX.Lane : $"{forX.Lane}; dw in {forW.Lane}";
    }

    // A part of the backward, run with its product lane TL (a dot lane where the number picks one: products of TN's own
    // operands), the lane's own stored type TS (the intermediates) and its plain lane TP (products and sums over them).
    private interface IBackwardPart
    {
        void Run<TN, TS, TIn, TAcc, TL, TP>() where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TS : struct, IAdad<TS, TAcc>
            where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc> where TP : struct, ILane<TP, TAcc>;
    }

    private struct BackwardVisitor<TPart>(TPart part) : ISumVisitor where TPart : struct, IBackwardPart
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var kept = new BackwardKept<TN, TLane, TLane, TL, TPart>(part);
            TL.Kept(ref kept);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            Lane = TD.Name;
            var kept = new BackwardKept<TN, TIn, TAcc, TD, TPart>(part);
            TD.Kept(ref kept);
        }
    }

    private readonly struct BackwardKept<TN, TIn, TAcc, TL, TPart>(TPart part) : IKeptVisitor<TAcc>
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc> where TPart : struct, IBackwardPart
    {
        public void Run<TS, TP>() where TS : struct, IAdad<TS, TAcc> where TP : struct, ILane<TP, TAcc> => part.Run<TN, TS, TIn, TAcc, TL, TP>();
    }

    // dx = col2im(wᵀ·dy): one image of one group (dcols in a buffer the thread keeps: exact length, every value written by the
    // product), or every image and group (image n group g's dcols = w_gᵀ·dy_{n,g} at rows (n·G + g)·Rows); col2im sums them
    // in the plain lane and stores dx as the type once.
    private readonly struct ConvDx(ConvArgs a, byte[] w, byte[] dy, byte[] dx, bool general) : IBackwardPart
    {
        public void Run<TN, TS, TIn, TAcc, TL, TP>() where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TS : struct, IAdad<TS, TAcc>
            where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc> where TP : struct, ILane<TP, TAcc>
        {
            int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TAcc>();
            if (!general)
            {
                var one = Scratch.Get(Scratch.ConvDcols, (long)a.Rows * a.Cols * size);
                MatMul<TN, TN, TS, TIn, TAcc, TL>(w, dy, one, a.Rows, a.Cols, a.OC, new Operands(TransA: true));                       // wᵀ · dy
                LayoutFromLanes<TN, TAcc, TP>(new LayoutArgs { Name = "col2im", X = one, P = a.P, Output = dx, Count = (int)a.XValues, InputCount = a.Rows * a.Cols });
                return;
            }

            int G = a.Groups, R = a.Rows, P = a.Cols;
            var dcols = new byte[a.ColValues * size];
            for (int n = 0; n < a.N; n++)
                for (int g = 0; g < G; g++)
                    MatMul<TN, TN, TS, TIn, TAcc, TL>(w, dy, dcols, R, P, a.OCg,
                        new Operands(AOff: g * a.OCg * R, TransA: true, BOff: (n * a.OC + g * a.OCg) * P, COff: (n * G + g) * R * P));      // w_gᵀ · dy_g
            LayoutFromLanes<TN, TAcc, TP>(new LayoutArgs { Name = "col2im", X = dcols, P = a.ImagesP, Output = dx, Count = (int)a.XValues, InputCount = (int)a.ColValues });
        }
    }

    // dw = dy · colsᵀ (and db = Σ dy): one image of one group with colsᵀ read from the image by the transposed packing (the
    // implicit im2col: the same packed values as the columns', so the same result bit for bit) or the columns themselves
    // (im2col into the lane's own type) where that does not apply; or every image and group: dy as (OC, N·P) and the columns
    // as (G·R, N·P), every image's positions side by side (exact moves into the lane), dw_g = dy_g · cols_gᵀ in one product a
    // group, db the plain lane's sums of dy's rows.
    private readonly struct ConvDw(ConvArgs a, byte[] x, byte[] dy, byte[] dw, byte[]? db, bool general) : IBackwardPart
    {
        public void Run<TN, TS, TIn, TAcc, TL, TP>() where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TS : struct, IAdad<TS, TAcc>
            where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc> where TP : struct, ILane<TP, TAcc>
        {
            int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TAcc>();
            if (!general)
            {
                if (FusedConv != false && ConvInPlace<TN>(a))
                {
                    MatMul<TN, TN, TN, TIn, TAcc, TL>(dy, x, dw, a.OC, a.Rows, a.Cols, new Operands(TransB: true, ImageB: a));
                    return;
                }

                var one = Scratch.Get(Scratch.ConvCols, (long)a.Rows * a.Cols * size);
                LayoutToLanes<TN, TAcc, TP>(new LayoutArgs { Name = "im2col", X = x, P = a.P, Output = one, Count = a.Rows * a.Cols, InputCount = (int)a.XValues });
                MatMul<TN, TS, TN, TAcc, TAcc, TP>(dy, one, dw, a.OC, a.Rows, a.Cols, new Operands(TransB: true));                  // dy · colsᵀ
                return;
            }

            int G = a.Groups, R = a.Rows, P = a.Cols, NP = a.N * a.Cols;
            var cols = new byte[a.ColValues * size];
            LayoutToLanes<TN, TAcc, TP>(new LayoutArgs { Name = "im2col", X = x, P = a.ImagesP, Output = cols, Count = (int)a.ColValues, InputCount = (int)a.XValues });
            var dyp = new byte[(long)a.OC * NP * size];
            var colsp = new byte[a.ColValues * size];
            LayoutToLanes<TN, TAcc, TP>(new LayoutArgs { Name = "strided", X = dy, P = [1, a.OC, a.N, P, 0, P, a.OC * P, 1, 0], Output = dyp, Count = a.OC * NP, InputCount = (int)a.YValues });
            LayoutToLanes<TS, TAcc, TP>(new LayoutArgs { Name = "strided", X = cols, P = [1, G * R, a.N, P, 0, P, G * R * P, 1, 0], Output = colsp, Count = (int)a.ColValues, InputCount = (int)a.ColValues });
            for (int g = 0; g < G; g++)
                MatMul<TS, TS, TN, TAcc, TAcc, TP>(dyp, colsp, dw, a.OCg, R, NP, new Operands(AOff: g * a.OCg * NP, BOff: g * R * NP, TransB: true, COff: g * a.OCg * R));   // dy_g · cols_gᵀ
            if (db is not null)
            {
                var rows = MemoryMarshal.Cast<byte, TAcc>(dyp.AsSpan());
                var sums = new TAcc[a.OC];
                for (int f = 0; f < a.OC; f++) sums[f] = LaneSum<TAcc, TP>(rows.Slice(f * NP, NP));
                TN.Store(sums, db, 0);
            }
        }
    }

    // Σ x in the lane: two vectors of sums, then their lanes, then the tail (RowMath.Sum's order, in any lane).
    private static TLane LaneSum<TLane, TL>(ReadOnlySpan<TLane> x) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int w = Vector<TLane>.Count, vec = x.Length / w * w;
        Vector<TLane> s0 = Vector<TLane>.Zero, s1 = s0;
        int i = 0;
        for (; i + 2 * w <= vec; i += 2 * w) { s0 = TL.Add(s0, new Vector<TLane>(x.Slice(i, w))); s1 = TL.Add(s1, new Vector<TLane>(x.Slice(i + w, w))); }
        for (; i < vec; i += w) s0 = TL.Add(s0, new Vector<TLane>(x.Slice(i, w)));
        TLane sum = Vector.Sum(TL.Add(s0, s1));
        for (i = vec; i < x.Length; i++) sum = TL.Add(sum, x[i]);
        return sum;
    }

    // The general forward: the columns of every image's channels in the lane (image n group g's at rows (n·G + g)·Rows), a
    // product an image and group into the lane's sums, then the epilogue's store over the output.
    private readonly struct ConvGeneralCall(ConvArgs a, ProductEpilogue e, StoreTerms terms, byte[] x, byte[] w, byte[] y, ProductTerms<byte[]> t) : IMixed
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
            EpilogueStore<TN, TS, TLane, TL>(a.Output, e, terms, t, sums, y, a.Cols, a.OC);
            return TL.Name;
        }
    }

}
