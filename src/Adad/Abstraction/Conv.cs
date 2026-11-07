// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Conv: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Conv.cs; Cuda/GpuOps.Conv.cs with its kernels in Cuda/Ptx.cs.

namespace Adad.Abstraction;

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
        return ProductView.Reference(w, cols, OC, Cols, Rows);
    }

    /// <summary>The float64 references of dx and dw.</summary>
    public (double[] DX, double[] DW) ReferenceBackward(double[] x, double[] w, double[] dy)
    {
        var cols = LayoutArgs.Reference("im2col", x, null, P, Rows * Cols);
        var wt = new double[Rows * OC];
        for (int r = 0; r < OC; r++) for (int c = 0; c < Rows; c++) wt[c * OC + r] = w[r * Rows + c];
        var dcols = ProductView.Reference(wt, dy, Rows, Cols, OC);
        var dx = LayoutArgs.Reference("col2im", dcols, null, P, (int)XValues);
        var colsT = new double[Cols * Rows];
        for (int r = 0; r < Rows; r++) for (int c = 0; c < Cols; c++) colsT[c * Rows + r] = cols[r * Cols + c];
        var dw = ProductView.Reference(dy, colsT, OC, Rows, Cols);
        return (dx, dw);
    }
}
