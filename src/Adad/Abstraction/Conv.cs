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
// One image of one group without a bias or an activation (Single) takes the fused forward; anything else the general path:
// the columns of all the images' channels (im2col of N·C channels: image n group g's columns at rows (n·Groups + g)·Rows)
// in the float lane's storage, each group's product of its weights with its columns in the lane, then y = act(sums +
// bias[f]) stored as the type once. The backward is the activation's caller's (dy is the gradient of the sums plus the
// bias): dcols = w_gᵀ·dy_g, dx = col2im(dcols), dw_g = dy_g · cols_gᵀ over every image (the images' columns and dy side by
// side), db = Σ dy over the images and positions.

namespace Adad.Abstraction;

/// <summary>A convolution's shape (Conv.cs): N images of C channels of H × W; OC filters of Kh × Kw taps, each over its
/// group's C / Groups channels; the windows Sh and Sw apart, their taps Dh and Dw apart; PadTop, PadLeft, PadBottom and
/// PadRight zeros on the image's sides. Act: an activation (a one-input element-wise formula by its name, Ops: relu,
/// sigmoid, tanh, gelu, gelu_erf, silu, leaky_relu with ActA, ...; null none).</summary>
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
    public string? Act { get; init; }
    public double ActA { get; init; }

    /// <summary>One image, one group, no activation: the fused path's (a bias takes the general one).</summary>
    public bool Single => N == 1 && Groups == 1 && Act is null;

    /// <summary>A group's input and output channels.</summary>
    public int Cg => C / Groups;
    public int OCg => OC / Groups;

    /// <summary>im2col's parameters over one image (C channels).</summary>
    public int[] P => [C, H, W, Kh, Kw, Sh, Sw, PadTop, PadLeft, Dh, Dw, 0, PadBottom, PadRight];

    /// <summary>im2col's parameters over every image's channels (N·C of them).</summary>
    public int[] ImagesP => [N * C, H, W, Kh, Kw, Sh, Sw, PadTop, PadLeft, Dh, Dw, 0, PadBottom, PadRight];

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
        if (Act is not null && Ops.Find(Act).Inputs != 1) throw new ArgumentException($"conv: the activation {Act} is not a one-input formula");
        if (XValues > int.MaxValue || WValues > int.MaxValue || YValues > int.MaxValue) throw new NotSupportedException($"conv: {Shape} past 2^31 values a tensor");
    }

    /// <summary>Throws where the columns of every image and group (the general path's and the composition's) pass 2^31 values.</summary>
    public void ColumnsFit()
    {
        if (ColValues > int.MaxValue) throw new NotSupportedException($"conv: the columns of {Shape} past 2^31 values");
    }

    /// <summary>The float64 reference of y (bias: OC values or none).</summary>
    public double[] Reference(double[] x, double[] w, double[]? bias = null)
    {
        var y = new double[YValues];
        var act = Act is null ? null : Ops.Find(Act);
        int plane = H * W;
        for (int n = 0; n < N; n++)
            for (int g = 0; g < Groups; g++)
            {
                var xs = x.AsSpan((n * C + g * Cg) * plane, Cg * plane).ToArray();
                var cols = LayoutArgs.Reference("im2col", xs, null, GroupP, Rows * Cols);
                var ws = w.AsSpan(g * OCg * Rows, OCg * Rows).ToArray();
                var prod = ProductView.Reference(ws, cols, OCg, Cols, Rows);
                for (int f = 0; f < OCg; f++)
                    for (int i = 0; i < Cols; i++)
                    {
                        double v = prod[f * Cols + i] + (bias is null ? 0 : bias[g * OCg + f]);
                        y[((long)n * OC + g * OCg + f) * Cols + i] = act is null ? v : act.Reference(v, 0, 0, ActA, 0);
                    }
            }

        return y;
    }

    /// <summary>Σ|w·x| (and |bias|) of each output: the products' error scale.</summary>
    public double[] Scale(double[] x, double[] w, double[]? bias = null)
    {
        var ax = x.Select(Math.Abs).ToArray();
        var aw = w.Select(Math.Abs).ToArray();
        var s = (this with { Act = null }).Reference(ax, aw, bias?.Select(Math.Abs).ToArray());
        return s;
    }

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
    private int[] GroupP => [Cg, H, W, Kh, Kw, Sh, Sw, PadTop, PadLeft, Dh, Dw, 0, PadBottom, PadRight];
}
