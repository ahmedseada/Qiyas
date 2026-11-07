// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Rows: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Rows.cs; Cuda/GpuOps.Rows.cs with its kernels in Cuda/Ptx.Rows.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>The float64 reference over one row; returns the row's error scale (the magnitude the lane's rounding acts on).</summary>
public delegate double RowReference(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2);

/// <summary>What the bench and the checks need to know about a row operation.</summary>
public sealed record RowOp(string Name, bool Second, bool Gamma, bool Beta, bool RowScalar, bool Positions, bool SecondOut, double A, double B, RowReference Reference, string Note = "", int Ulps = 8)
{
    public (double Low, double High) XRange { get; init; } = (-4, 4);

    /// <summary>The second row's values: dy in (−1, 1), a residual in (−4, 4); the mask is a pattern (see Mask).</summary>
    public (double Low, double High) SecondRange { get; init; } = (-1, 1);

    /// <summary>The second row is an additive mask: 0 keeps a value, −400 excludes it (exp underflows to 0 in f32; within
    /// every float type's range, 448 being the smallest of the maxima); a pattern over (row, column).</summary>
    public bool Mask => Name == "scale_mask_softmax";

    public static double MaskValue(int row, int col) => (row * 7 + col) % 3 == 0 ? -400 : 0;
}

public static class RowOps
{
    private const double Eps = 1e-5;

    public static readonly RowOp[] All =
    [
        new("log_softmax", false, false, false, false, false, false, 0, 0, LogSoftmax),
        new("softmax_bwd", true, false, false, false, false, false, 0, 0, SoftmaxBwd, "(y, dy) → dx = y·(dy − Σ y·dy)") { XRange = (0, 1) },
        new("log_softmax_bwd", true, false, false, false, false, false, 0, 0, LogSoftmaxBwd, "(log y, dy) → dx = dy − exp(log y)·Σ dy") { XRange = (-8, 0) },
        new("rms_norm_affine", false, true, false, false, false, false, Eps, 0, RmsNormAffine, "y = x·g / √(mean x² + a)"),
        new("rms_norm_bwd", true, true, false, false, false, false, Eps, 0, RmsNormBwd, "(x, dy, g) → dx; x̂ = x·rstd: dx = rstd·(g·dy − x̂·mean(g·dy·x̂))"),
        new("layer_norm", false, true, true, false, false, false, Eps, 0, LayerNorm, "y = (x − μ)·g / √(var + a) + bt"),
        new("layer_norm_bwd", true, true, false, false, false, false, Eps, 0, LayerNormBwd, "(x, dy, g) → dx = rstd·(g·dy − mean(g·dy) − x̂·mean(g·dy·x̂))"),
        new("weight_norm", false, false, false, true, false, false, 1e-12, 0, WeightNorm, "y = x·s / √(Σ x² + a), s the row's scalar"),
        new("rope", false, false, false, false, true, false, 10000, 0, Rope, "pairs (i, i + cols/2) rotated by pos·a^(−2i/cols); an odd row's last value kept", Ulps: 64),
        new("scale_mask_softmax", true, false, false, false, false, false, 0.125, 1, ScaleMaskSoftmax, "softmax(a·x + mask); b ≠ 0: columns past the row's position excluded (causal; the position is the row's index, or the one given: a key span)"),
        new("add_rms_norm", true, true, false, false, false, true, Eps, 0, AddRmsNorm, "h = x + x2 (written to y2); y = rms_norm_affine(h)") { SecondRange = (-4, 4) },
        new("add_layer_norm", true, true, true, false, false, true, Eps, 0, AddLayerNorm, "h = x + x2 (written to y2); y = layer_norm(h)") { SecondRange = (-4, 4) },
        new("cross_entropy_bwd", false, false, false, false, true, false, 0, 0, CrossEntropyBwd, "softmax(x) − one_hot(the row's position: the target class); the loss is −log_softmax(x) picked at the target (pick_rows)"),
    ];

    public static RowOp Find(string name) => Array.Find(All, o => o.Name == name) ?? throw new ArgumentException($"unknown row operation {name}");

    private static double MaxAbs(ReadOnlySpan<double> v) { double m = 0; foreach (var t in v) m = Math.Max(m, Math.Abs(t)); return m; }

    private static double LogSoftmax(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double m = double.NegativeInfinity, sum = 0;
        foreach (var v in x) m = Math.Max(m, v);
        foreach (var v in x) sum += Math.Exp(v - m);
        double ls = m + Math.Log(sum);
        for (int i = 0; i < x.Length; i++) y[i] = x[i] - ls;
        return MaxAbs(x) + Math.Abs(ls);
    }

    private static double SoftmaxBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double d = 0;
        for (int i = 0; i < x.Length; i++) d += x[i] * x2[i];
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * (x2[i] - d);
        return MaxAbs(x2) * (1 + MaxAbs(x) * x.Length);
    }

    private static double LogSoftmaxBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double sd = 0, abs = 0;
        foreach (var v in x2) { sd += v; abs += Math.Abs(v); }
        for (int i = 0; i < x.Length; i++) y[i] = x2[i] - Math.Exp(x[i]) * sd;
        return abs + MaxAbs(x2);
    }

    private static double Rstd(ReadOnlySpan<double> x, double mean, double eps)
    {
        double v = 0;
        foreach (var t in x) v += (t - mean) * (t - mean);
        return 1 / Math.Sqrt(v / x.Length + eps);
    }

    private static double RmsNormAffine(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double rstd = Rstd(x, 0, a);
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * rstd * g[i];
        return MaxAbs(y) * Math.Sqrt(x.Length);
    }

    private static double RmsNormBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double rstd = Rstd(x, 0, a), t = 0;
        for (int i = 0; i < x.Length; i++) t += g[i] * x2[i] * x[i] * rstd;
        double m = t / x.Length;
        for (int i = 0; i < x.Length; i++) y[i] = rstd * (g[i] * x2[i] - x[i] * rstd * m);
        return rstd * MaxAbs(g) * MaxAbs(x2) * (1 + MaxAbs(x) * rstd * MaxAbs(x) * rstd) * Math.Sqrt(x.Length);
    }

    private static double LayerNorm(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double mu = 0;
        foreach (var t in x) mu += t;
        mu /= x.Length;
        double rstd = Rstd(x, mu, a);
        for (int i = 0; i < x.Length; i++) y[i] = (x[i] - mu) * rstd * g[i] + bt[i];
        return (MaxAbs(y) + MaxAbs(bt)) * Math.Sqrt(x.Length);
    }

    private static double LayerNormBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double mu = 0;
        foreach (var t in x) mu += t;
        mu /= x.Length;
        double rstd = Rstd(x, mu, a), m1 = 0, m2 = 0, xh = 0;
        for (int i = 0; i < x.Length; i++) { double h = (x[i] - mu) * rstd; m1 += g[i] * x2[i]; m2 += g[i] * x2[i] * h; xh = Math.Max(xh, Math.Abs(h)); }
        m1 /= x.Length; m2 /= x.Length;
        for (int i = 0; i < x.Length; i++) y[i] = rstd * (g[i] * x2[i] - m1 - (x[i] - mu) * rstd * m2);
        return rstd * MaxAbs(g) * MaxAbs(x2) * (2 + xh * xh) * Math.Sqrt(x.Length);
    }

    private static double WeightNorm(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double ss = 0;
        foreach (var t in x) ss += t * t;
        double k = s / Math.Sqrt(ss + a);
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * k;
        return MaxAbs(y) * Math.Sqrt(x.Length);
    }

    private static double Rope(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        int n = x.Length, half = n / 2;
        for (int i = 0; i < half; i++)
        {
            double theta = pos * Math.Exp(-2.0 * i / n * Math.Log(a)), c = Math.Cos(theta), sn = Math.Sin(theta);
            y[i] = x[i] * c - x[i + half] * sn;
            y[i + half] = x[i] * sn + x[i + half] * c;
        }

        if (n % 2 == 1) y[n - 1] = x[n - 1];
        return MaxAbs(x) * (1 + 2 * Math.Abs(pos));
    }

    private static double ScaleMaskSoftmax(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        int n = x.Length;
        double m = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            double z = b != 0 && i > pos ? double.NegativeInfinity : a * x[i] + (x2.IsEmpty ? 0 : x2[i]);
            y[i] = z;
            m = Math.Max(m, z);
        }

        double sum = 0;
        for (int i = 0; i < n; i++) sum += y[i] = double.IsNegativeInfinity(y[i]) ? 0 : Math.Exp(y[i] - m);
        for (int i = 0; i < n; i++) y[i] = sum > 0 ? y[i] / sum : 0;
        return 1 + Math.Abs(a) * MaxAbs(x);
    }

    private static double CrossEntropyBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double m = double.NegativeInfinity, sum = 0;
        foreach (var v in x) m = Math.Max(m, v);
        for (int i = 0; i < x.Length; i++) sum += y[i] = Math.Exp(x[i] - m);
        for (int i = 0; i < x.Length; i++) y[i] = y[i] / sum - (i == pos ? 1 : 0);
        return 1 + MaxAbs(x);
    }

    private static double AddRmsNorm(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        for (int i = 0; i < x.Length; i++) y2[i] = x[i] + x2[i];
        return RmsNormAffine(y2, default, g, bt, s, pos, row, a, b, y, default) + MaxAbs(y2);
    }

    private static double AddLayerNorm(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        for (int i = 0; i < x.Length; i++) y2[i] = x[i] + x2[i];
        return LayerNorm(y2, default, g, bt, s, pos, row, a, b, y, default) + MaxAbs(y2);
    }

    /// <summary>dgamma[c] = Σ_r dy·x̂, dbeta[c] = Σ_r dy with x̂ = (x − μ_r)·rstd_r (centered) or x·rstd_r; returns the error scale.</summary>
    public static double NormParamsReference(double[] x, double[] dy, int rows, int cols, double eps, bool centered, double[] dgamma, double[] dbeta)
    {
        double scale = 0;
        for (int r = 0; r < rows; r++)
        {
            var xr = x.AsSpan(r * cols, cols);
            double mu = 0;
            if (centered) { foreach (var t in xr) mu += t; mu /= cols; }
            double rstd = Rstd(xr, mu, eps);
            for (int c = 0; c < cols; c++)
            {
                double h = (xr[c] - mu) * rstd, d = dy[r * cols + c];
                dgamma[c] += d * h;
                dbeta[c] += d;
                scale = Math.Max(scale, Math.Abs(d) * (1 + Math.Abs(h)));
            }
        }

        return scale * rows;
    }
}

// ---------------------------------------------------------------------------------------------- the CPU kernels
