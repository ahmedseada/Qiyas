// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of the convolution: its arguments, its definition and its float64 references. The devices
// implement it in their folders: Cpu/Conv.cs; Cuda/GpuOps.Conv.cs with its kernels in Cuda/Ptx.cs; Vulkan/VulkanOps.Conv.cs.
//
// The convolution of N images x (N × C × H × W) by OC filters w (OC × C/Groups × Kh × Kw) is y (N × OC × OH × OW):
//
//   y[n, f, i, j] = Σ over c < C/Groups, a < Kh, b < Kw of  w[f, c, a, b] · x[n, g·C/Groups + c, i·Sh − PadTop + a·Dh, j·Sw − PadLeft + b·Dw]
//
// g = f / (OC/Groups) the filter's group, whose C/Groups channels it reads; x is 0 off the image (a pad: the zeros its side
// adds). OH and OW count the windows that fit in the padded image (LayoutArgs.ConvOut). Through the columns (im2col: row
// (c, a, b), column (i, j)) it is a product a group: y_g = w_g · cols_g.
//
// y is the product's epilogue (ProductEpilogue, Products.cs) over these sums, a product of OC × OH·OW an image (Output):
// + a bias, a value a filter (an output channel), + β·y, + a second product, then an activation or a gradient through one,
// then + a residual; the sums kept where asked; y rounded to the type once.
//
// One image of one group with no epilogue term (Single) takes the fused forward; anything else the general path: the
// columns of all the images' channels (im2col of N·C channels: image n group g's columns at rows (n·Groups + g)·Rows) in
// the lane's storage, each group's product of its weights with its columns in the lane, then the epilogue's store. A
// float type's lane is its float lane; any other type's the lane its sums take by the bounds of every value the store
// makes (IntegerSums: exact), so y is its one rounding (an integer type's saturating).
// The backward is of the sums: dy is the gradient of the sums plus the bias (an activation's backward is the caller's, or
// the epilogue's gradient term): dcols = w_gᵀ·dy_g, dx = col2im(dcols), dw_g = dy_g · cols_gᵀ over every image (the
// images' columns and dy side by side), db = Σ dy over the images and positions.

namespace Adad.Abstraction;

/// <summary>A convolution's shape (Conv.cs): N images of C channels of H × W; OC filters of Kh × Kw taps, each over its
/// group's C / Groups channels; the windows Sh and Sw apart, their taps Dh and Dw apart; PadTop, PadLeft, PadBottom and
/// PadRight zeros on the image's sides.</summary>
public sealed record ConvArgs(int C, int H, int W, int OC, int Kh, int Kw)
{
    public int Sh { get; init; } = 1;
    public int Sw { get; init; } = 1;
    public int Dh { get; init; } = 1;
    public int Dw { get; init; } = 1;
    public int PadTop { get; init; }
    public int PadLeft { get; init; }
    public int PadBottom { get; init; }
    public int PadRight { get; init; }
    public int N { get; init; } = 1;
    public int Groups { get; init; } = 1;

    /// <summary>One image, one group: the fused path's (with no epilogue term).</summary>
    public bool Single => N == 1 && Groups == 1;

    /// <summary>The output as the epilogue's view: N products of OC × OH·OW (the weights times an image's columns), contiguous.</summary>
    public ProductView Output => new(OC, Cols, Rows, false, false, N, 0, 0, (long)OC * Cols);

    /// <summary>A group's input and output channels.</summary>
    public int Cg => C / Groups;
    public int OCg => OC / Groups;

    /// <summary>im2col's parameters over one image (C channels).</summary>
    public int[] P => [C, H, W, Kh, Kw, Sh, Sw, PadTop, PadLeft, PadBottom, PadRight, Dh, Dw];

    /// <summary>im2col's parameters over every image's channels (N·C of them).</summary>
    public int[] ImagesP => [N * C, H, W, Kh, Kw, Sh, Sw, PadTop, PadLeft, PadBottom, PadRight, Dh, Dw];

    public (int OH, int OW) Out => LayoutArgs.ConvOut(P);
    public int Rows => Cg * Kh * Kw;                     // a group's columns' rows
    public int Cols => Out.OH * Out.OW;                  // the columns' columns (= an image's output pixels)
    public long XValues => (long)N * C * H * W;
    public long WValues => (long)OC * Rows;
    public long YValues => (long)N * OC * Cols;

    /// <summary>The columns of every image and group: N·Groups blocks of Rows × Cols.</summary>
    public long ColValues => (long)N * Groups * Rows * Cols;

    /// <summary>The shape in words, for a refusal or a measured choice's key.</summary>
    public string Shape => $"{N} × {C} × {H} × {W} by {OC} × {Cg} × {Kh} × {Kw}, {Groups} groups, strides {Sh} × {Sw}, dilations {Dh} × {Dw}, pads {PadTop} {PadLeft} {PadBottom} {PadRight}";

    /// <summary>Throws on a shape the operation does not take (ArgumentException), or one past the devices' 2^31 values a
    /// tensor (NotSupportedException).</summary>
    public void Validate()
    {
        if (C < 0 || H < 0 || W < 0 || OC < 0 || N < 0) throw new ArgumentException($"conv: a negative extent ({Shape})");
        if (Kh < 1 || Kw < 1) throw new ArgumentException($"conv: a filter of no taps ({Shape})");
        if (Sh < 1 || Sw < 1 || Dh < 1 || Dw < 1) throw new ArgumentException($"conv: a stride or a dilation below 1 ({Shape})");
        if (PadTop < 0 || PadLeft < 0 || PadBottom < 0 || PadRight < 0) throw new ArgumentException($"conv: a negative pad ({Shape})");
        if (Groups < 1 || C % Groups != 0 || OC % Groups != 0) throw new ArgumentException($"conv: {C} channels and {OC} filters in {Groups} groups");
        if (XValues > int.MaxValue || WValues > int.MaxValue || YValues > int.MaxValue) throw new NotSupportedException($"conv: {Shape} past 2^31 values a tensor");
    }

    /// <summary>Throws where the columns of every image and group (the general path's and the composition's) pass 2^31 values.</summary>
    public void ColumnsFit()
    {
        if (ColValues > int.MaxValue) throw new NotSupportedException($"conv: the columns of {Shape} past 2^31 values");
    }

    /// <summary>The float64 reference of the convolution's sums (y's layout), by the definition.</summary>
    public double[] Reference(double[] x, double[] w)
    {
        var y = new double[YValues];
        int plane = H * W;
        for (int n = 0; n < N; n++)
            for (int g = 0; g < Groups; g++)
            {
                var xs = x.AsSpan((n * C + g * Cg) * plane, Cg * plane).ToArray();
                var cols = LayoutArgs.Reference("im2col", xs, null, GroupP, Rows * Cols);
                var ws = w.AsSpan(g * OCg * Rows, OCg * Rows).ToArray();
                ProductView.Reference(ws, cols, OCg, Cols, Rows).CopyTo(y, ((long)n * OC + g * OCg) * Cols);
            }

        return y;
    }

    /// <summary>The float64 reference of y and the sums s it was made from, with the epilogue over Output (ProductEpilogue.Apply:
    /// the bias a value an output channel; y0 y's values before; residual and actInputs y's layout; a2 and b2 the second
    /// product's operands as its view stores them).</summary>
    public (double[] Y, double[] S) Reference(ProductEpilogue e, double[] x, double[] w, double[] y0, double[]? bias, double[]? residual, double[]? actInputs, double[]? a2, double[]? b2) =>
        e.Apply(Reference(x, w), Cols, OC, y0, bias, residual, actInputs, e.Second?.Reference(a2!, b2!));

    /// <summary>The bounds a type that is not float takes its lane by (ProductEpilogue.IntegerSums over Output): k = Rows, the
    /// weights' range and the image's with 0 (a pad's value), each term's operands' ranges; range(buffer, count) a buffer's
    /// (low, high).</summary>
    public (Bounds Bounds, bool Negative) IntegerSums<TBuffer>(ProductEpilogue e, StoreTerms terms, string type, bool signed,
        Func<TBuffer, long, (double Lo, double Hi)> range, TBuffer x, TBuffer w, TBuffer y, ProductTerms<TBuffer> t)
    {
        var (xLo, xHi) = range(x, XValues);
        var s = e.Second;
        return e.IntegerSums(type, signed, Rows, range(w, WValues), (Math.Min(xLo, 0), Math.Max(xHi, 0)),
            s is null ? null : range(t.SecondA!, s.ValuesA), s is null ? null : range(t.SecondB!, s.ValuesB),
            terms.Bias ? range(t.Bias!, OC) : null, terms.Beta ? range(y, YValues) : null,
            terms.Residual ? range(t.Residual!, YValues) : null, terms.Grad is not null ? range(t.ActInputs!, YValues) : null);
    }

    /// <summary>Σ|w·x| of each output: the sums' error scale.</summary>
    public double[] Scale(double[] x, double[] w) => Reference(x.Select(Math.Abs).ToArray(), w.Select(Math.Abs).ToArray());

    /// <summary>The float64 references of dx, dw and db (dy the gradient of the sums plus the bias).</summary>
    public (double[] DX, double[] DW, double[] DB) ReferenceBackward(double[] x, double[] w, double[] dy)
    {
        var dx = new double[XValues];
        var dw = new double[WValues];
        var db = new double[OC];
        int plane = H * W;
        for (int n = 0; n < N; n++)
            for (int g = 0; g < Groups; g++)
            {
                var xs = x.AsSpan((n * C + g * Cg) * plane, Cg * plane).ToArray();
                var ws = w.AsSpan(g * OCg * Rows, OCg * Rows).ToArray();
                var dys = dy.AsSpan((n * OC + g * OCg) * Cols, OCg * Cols).ToArray();
                var cols = LayoutArgs.Reference("im2col", xs, null, GroupP, Rows * Cols);
                var wt = new double[Rows * OCg];
                for (int r = 0; r < OCg; r++) for (int c = 0; c < Rows; c++) wt[c * OCg + r] = ws[r * Rows + c];
                var dcols = ProductView.Reference(wt, dys, Rows, Cols, OCg);
                var dxs = LayoutArgs.Reference("col2im", dcols, null, GroupP, Cg * plane);
                dxs.CopyTo(dx, (n * C + g * Cg) * plane);
                var colsT = new double[Cols * Rows];
                for (int r = 0; r < Rows; r++) for (int c = 0; c < Cols; c++) colsT[c * Rows + r] = cols[r * Cols + c];
                var dws = ProductView.Reference(dys, colsT, OCg, Rows, Cols);
                for (int i = 0; i < dws.Length; i++) dw[g * OCg * Rows + i] += dws[i];
                for (int f = 0; f < OCg; f++) for (int i = 0; i < Cols; i++) db[g * OCg + f] += dys[f * Cols + i];
            }

        return (dx, dw, db);
    }

    // im2col's parameters over one group's channels.
    private int[] GroupP => [Cg, H, W, Kh, Kw, Sh, Sw, PadTop, PadLeft, PadBottom, PadRight, Dh, Dw];
}
