// Convolution as the composition of Qiyas's own operations: the image to columns (im2col, exact in any type), the
// product with the weights (the product kernel: tensor and dot lanes by the bounds), and back (col2im for the input's
// gradient). One image at a time (C × H × W; the weights OC × C·kh·kw; the output OC × OH × OW):
//
//   conv           y = w · im2col(x)
//   conv_bwd_x     dx = col2im(wᵀ · dy)
//   conv_bwd_w     dw = dy · im2col(x)ᵀ
//
// The forward is fused (item 4g, implicit GEMM): the product's packing builds each row of the columns from the image
// itself, so no columns are in memory — the same packed values as the composition, so the same result bit for bit
// (CPU: every type whose rows start on whole bytes; the device: GpuOps.Conv.cs). The backward keeps the composition.
// The depthwise convolution is a layout operation (Layout.cs: a gather with weights).
namespace Qiyas;

/// <summary>C H W kh kw sh sw ph pw dh dw and the output channels.</summary>
public sealed record ConvArgs(int C, int H, int W, int OC, int Kh, int Kw, int Sh = 1, int Sw = 1, int Ph = 0, int Pw = 0, int Dh = 1, int Dw = 1)
{
    public int[] P => [C, H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw, 0];
    public (int OH, int OW) Out => LayoutArgs.ConvOut(P);
    public int Rows => C * Kh * Kw;                      // the columns' rows
    public int Cols => Out.OH * Out.OW;                  // the columns' columns (= output pixels)
    public long XValues => (long)C * H * W;
    public long WValues => (long)OC * Rows;
    public long YValues => (long)OC * Cols;

    /// <summary>The float64 reference of y.</summary>
    public double[] Reference(double[] x, double[] w)
    {
        var cols = LayoutArgs.Reference("im2col", x, null, P, Rows * Cols);
        return global::References.MatMul(w, cols, OC, Cols, Rows);
    }

    /// <summary>The float64 references of dx and dw.</summary>
    public (double[] DX, double[] DW) ReferenceBackward(double[] x, double[] w, double[] dy)
    {
        var cols = LayoutArgs.Reference("im2col", x, null, P, Rows * Cols);
        var wt = new double[Rows * OC];
        for (int r = 0; r < OC; r++) for (int c = 0; c < Rows; c++) wt[c * OC + r] = w[r * Rows + c];
        var dcols = global::References.MatMul(wt, dy, Rows, Cols, OC);
        var dx = LayoutArgs.Reference("col2im", dcols, null, P, (int)XValues);
        var colsT = new double[Cols * Rows];
        for (int r = 0; r < Rows; r++) for (int c = 0; c < Cols; c++) colsT[c * Rows + r] = cols[r * Cols + c];
        var dw = global::References.MatMul(dy, colsT, OC, Rows, Cols);
        return (dx, dw);
    }
}

internal static partial class Kernels
{
    /// <summary>The gate's override: false takes the composition (im2col, then the product), null the fused path where it applies.</summary>
    internal static bool? FusedConv;

    // The fused path reads A's rows, the image's rows and C at value indices: each on whole bytes.
    private static bool ConvInPlace<TN>(ConvArgs a) where TN : struct, IQiyasNumber<TN>
    {
        int per = TN.Bits >= 8 ? 1 : 8 / TN.Bits;
        return a.XValues <= int.MaxValue && a.YValues <= int.MaxValue && a.WValues <= int.MaxValue
            && (per == 1 || (a.OC % per == 0 && a.Rows % per == 0 && a.Cols % per == 0 && (TN.Granularity % 32 == 0 || a.W % per == 0)));
    }

    /// <summary>y = conv(x, w) on the CPU; returns the product's lane.</summary>
    public static string Conv<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) where TN : struct, IQiyasNumber<TN>
    {
        if (FusedConv != false && ConvInPlace<TN>(a))
        {
            // The columns' values are the image's and 0 (padding): the image's range, with 0, bounds them.
            var (wLo, wHi) = TN.Range(w, (int)a.WValues);
            var (xLo, xHi) = TN.Range(x, (int)a.XValues);
            var product = new MatMulVisitor(w, x, y, a.OC, a.Cols, a.Rows) { Where = new Operands(ImageB: a) };
            TN.Sum(ref product, new Bounds(wLo, wHi, Math.Min(xLo, 0), Math.Max(xHi, 0), a.Rows));
            return product.Lane;
        }

        return ConvComposed<TN>(a, x, w, y);
    }

    /// <summary>y = conv(x, w) by the composition: im2col, then the product.</summary>
    public static string ConvComposed<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) where TN : struct, IQiyasNumber<TN>
    {
        var cols = new byte[BytesOf<TN>((long)a.Rows * a.Cols)];
        Layout<TN>(new LayoutArgs { Name = "im2col", X = x, P = a.P, Output = cols, Count = a.Rows * a.Cols, InputCount = (int)a.XValues });
        return MatMulView<TN>(new ProductView(a.OC, a.Cols, a.Rows, false, false), w, cols, y);
    }

    /// <summary>dx and dw from dy, in the type's float lane storage (the columns' gradient is a sum of products: a narrow
    /// type cannot hold it between the two passes); returns the lane's name.</summary>
    public static string ConvBackward<TN>(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) where TN : struct, IQiyasFloat<TN>
    {
        bool own = TN.Name is "float32" or "float64";
        var lx = ToLane<TN>(x, a.XValues); var lw = ToLane<TN>(w, a.WValues); var ldy = ToLane<TN>(dy, a.YValues);
        int size = TN.Bits == 64 ? 8 : 4;
        var ldx = own ? dx : new byte[a.XValues * size];
        var ldw = own ? dw : new byte[a.WValues * size];
        if (TN.Bits == 64) BackwardIn<Float64>(a, lx, lw, ldy, ldx, ldw); else BackwardIn<Float32>(a, lx, lw, ldy, ldx, ldw);
        if (!own) { FromLane<TN>(ldx, dx, a.XValues); FromLane<TN>(ldw, dw, a.WValues); }
        return TN.Bits == 64 ? "f64" : "f32";
    }

    private static void BackwardIn<TS>(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) where TS : struct, IQiyasFloat<TS>
    {
        var cols = new byte[BytesOf<TS>((long)a.Rows * a.Cols)];
        Layout<TS>(new LayoutArgs { Name = "im2col", X = x, P = a.P, Output = cols, Count = a.Rows * a.Cols, InputCount = (int)a.XValues });
        var dcols = new byte[cols.Length];
        MatMulView<TS>(new ProductView(a.Rows, a.Cols, a.OC, true, false), w, dy, dcols);                   // wᵀ · dy
        Layout<TS>(new LayoutArgs { Name = "col2im", X = dcols, P = a.P, Output = dx, Count = (int)a.XValues, InputCount = a.Rows * a.Cols });
        MatMulView<TS>(new ProductView(a.OC, a.Rows, a.Cols, false, true), dy, cols, dw);                   // dy · colsᵀ
    }
}
