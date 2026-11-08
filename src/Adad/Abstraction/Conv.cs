// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Conv: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Conv.cs; Cuda/GpuOps.Conv.cs with its kernels in Cuda/Ptx.cs; Vulkan/VulkanOps.Conv.cs.
//
// One image of one group (Single) takes the fused forward; images (N, NCHW), groups (filter f of group f / OCg reading
// that group's Cg channels: PyTorch's weights [OC, C / groups, kh, kw]), the bottom and right pads (ONNX's four pads), a
// bias and an activation take the general path: the columns of all the images' channels (im2col of N·C channels: image n
// group g's columns at rows (n·Groups + g)·Rows) in the float lane's storage, each group's product of its weights with its
// columns in the lane, then y = act(sums + bias[f]) stored as the type once. The backward is the activation's caller's
// (dy is the gradient of the sums plus the bias): dcols = w_gᵀ·dy_g, dx = col2im(dcols), dw_g = dy_g · cols_gᵀ over every
// image (the images' columns and dy side by side), db = Σ dy over the images and positions.

namespace Adad.Abstraction;

/// <summary>C H W kh kw sh sw ph pw dh dw and the output channels; the images, groups, bottom and right pads, and the
/// activation (a one-input element-wise formula by its name, Ops: relu, sigmoid, tanh, gelu, gelu_erf, silu, leaky_relu
/// with ActA, ...; null none) as init properties.</summary>
public sealed record ConvArgs(int C, int H, int W, int OC, int Kh, int Kw, int Sh = 1, int Sw = 1, int Ph = 0, int Pw = 0, int Dh = 1, int Dw = 1)
{
    public int N { get; init; } = 1;
    public int Groups { get; init; } = 1;
    public int? PadBottom { get; init; }
    public int? PadRight { get; init; }
    public string? Act { get; init; }
    public double ActA { get; init; }

    public int Pb => PadBottom ?? Ph;
    public int Pr => PadRight ?? Pw;

    /// <summary>One image, one group, symmetric pads, no activation: the fused path's (a bias takes the general one).</summary>
    public bool Single => N == 1 && Groups == 1 && Pb == Ph && Pr == Pw && Act is null;

    /// <summary>A group's input and output channels.</summary>
    public int Cg => C / Groups;
    public int OCg => OC / Groups;

    /// <summary>im2col's parameters over one image (C channels; the bottom and right pads last).</summary>
    public int[] P => [C, H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw, 0, Pb, Pr];

    /// <summary>im2col's parameters over every image's channels (N·C of them).</summary>
    public int[] ImagesP => [N * C, H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw, 0, Pb, Pr];

    public (int OH, int OW) Out => LayoutArgs.ConvOut(P);
    public int Rows => Cg * Kh * Kw;                     // a group's columns' rows
    public int Cols => Out.OH * Out.OW;                  // the columns' columns (= an image's output pixels)
    public long XValues => (long)N * C * H * W;
    public long WValues => (long)OC * Rows;
    public long YValues => (long)N * OC * Cols;

    /// <summary>The columns of every image and group: N·Groups blocks of Rows × Cols.</summary>
    public long ColValues => (long)N * Groups * Rows * Cols;

    /// <summary>Throws on a shape the operation does not take.</summary>
    public void Validate()
    {
        if (N < 0 || Groups <= 0 || C % Groups != 0 || OC % Groups != 0) throw new ArgumentException($"conv: {C} channels and {OC} filters in {Groups} groups");
        if (Pb < 0 || Pr < 0) throw new ArgumentException("conv: a negative pad");
        if (Act is not null && Ops.Find(Act).Inputs != 1) throw new ArgumentException($"conv: the activation {Act} is not a one-input formula");
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
    private int[] GroupP => [Cg, H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw, 0, Pb, Pr];
}
