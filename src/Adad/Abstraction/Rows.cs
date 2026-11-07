// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Rows: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Rows.cs; Cuda/GpuOps.Rows.cs with its kernels in Cuda/Ptx.Rows.cs. Group normalization (GroupShape) and the
// rotary embedding by tables (RopeArgs) are here too, their kernels in each device's Norms files.

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

    /// <summary>y2 holds one value a row, in the float lane's storage (a per-row parameter's gradient), not a second row.</summary>
    public bool RowOut { get; init; }

    /// <summary>b is an offset of the gain: the operation scales by g + b (0: by g).</summary>
    public bool Offset { get; init; }
}

/// <summary>The layout a group normalization reads (IRows.GroupNorm): x as [Outer, Groups, Inner], each group's statistics over
/// its Outer·Inner values; the gain and bias per channel c = (index / Spatial) % Channels of a value's index.</summary>
public readonly record struct GroupShape(int Outer, int Groups, int Inner, int Channels, int Spatial)
{
    /// <summary>Batch normalization of [n, channels, spatial]: a channel's statistics over the batch and the positions.</summary>
    public static GroupShape Batch(int n, int channels, int spatial) => new(n, channels, spatial, channels, spatial);

    /// <summary>Group normalization of [n, channels, spatial] in groups of channels / groups channels: instance normalization
    /// with groups = channels, a layer's over channels and positions with 1.</summary>
    public static GroupShape Group(int n, int channels, int spatial, int groups) => new(1, n * groups, channels / groups * spatial, channels, spatial);

    public int Count => Outer * Groups * Inner;

    /// <summary>The values each group's statistics are over.</summary>
    public int PerGroup => Outer * Inner;

    public int GroupOf(int index) => index / Inner % Groups;

    public int ChannelOf(int index) => index / Spatial % Channels;

    /// <summary>The index of value i of row o of group g.</summary>
    public int Index(int o, int g, int i) => (o * Groups + g) * Inner + i;
}

/// <summary>A rotary embedding by tables (IRows.Rope) over Rows rows of Cols values (rows = batch · Steps · Heads, the heads of a
/// step adjacent): row r at position positions[(r / Heads) % Steps] rotates its pair p &lt; Half — (2p, 2p + 1) Interleaved, else
/// (p, p + Half) — by the angle whose cosine and sine are cos[pos, p] and sin[pos, p] (tables of Half columns in the float lane's
/// storage); Inverse rotates back (the gradient); the columns past 2·Half are kept. With a gain the row is first
/// RMS-normalized: x·(gain + Offset) / √(mean x² + Eps).</summary>
public sealed record RopeArgs(int Rows, int Cols, int Heads, int Steps, int Half, bool Interleaved, bool Inverse = false, double Eps = 1e-6, double Offset = 0)
{
    public int Position(ReadOnlySpan<int> positions, int row) => positions[row / Heads % Steps];

    /// <summary>The columns of pair p.</summary>
    public (int Low, int High) Pair(int p) => Interleaved ? (2 * p, 2 * p + 1) : (p, p + Half);
}

public static class RowOps
{
    private const double Eps = 1e-5;

    public static readonly RowOp[] All =
    [
        new("log_softmax", false, false, false, false, false, false, 0, 0, LogSoftmax),
        new("softmax_bwd", true, false, false, false, false, false, 0, 0, SoftmaxBwd, "(y, dy) → dx = y·(dy − Σ y·dy)") { XRange = (0, 1) },
        new("log_softmax_bwd", true, false, false, false, false, false, 0, 0, LogSoftmaxBwd, "(log y, dy) → dx = dy − exp(log y)·Σ dy") { XRange = (-8, 0) },
        new("rms_norm_affine", false, true, false, false, false, false, Eps, 0, RmsNormAffine, "y = x·(g + b) / √(mean x² + a)") { Offset = true },
        new("rms_norm_bwd", true, true, false, false, false, false, Eps, 0, RmsNormBwd, "(x, dy, g) → dx; x̂ = x·rstd, γ = g + b: dx = rstd·(γ·dy − x̂·mean(γ·dy·x̂))") { Offset = true },
        new("layer_norm", false, true, true, false, false, false, Eps, 0, LayerNorm, "y = (x − μ)·g / √(var + a) + bt"),
        new("layer_norm_bwd", true, true, false, false, false, false, Eps, 0, LayerNormBwd, "(x, dy, g) → dx = rstd·(g·dy − mean(g·dy) − x̂·mean(g·dy·x̂))"),
        new("weight_norm", false, false, false, true, false, false, 1e-12, 0, WeightNorm, "y = x·s / √(Σ x² + a), s the row's scalar"),
        new("weight_norm_bwd", true, false, false, true, false, false, 1e-12, 0, WeightNormBwd, "(x, dy, s) → dx = s/n·(dy − x·(Σ dy·x)/n²), n = √(Σ x² + a); y2[row] = (Σ dy·x)/n, the scalar's gradient") { RowOut = true },
        new("rope", false, false, false, false, true, false, 10000, 0, Rope, "pairs (i, i + cols/2) rotated by pos·a^(−2i/cols); an odd row's last value kept", Ulps: 64),
        new("scale_mask_softmax", true, false, false, false, false, false, 0.125, 1, ScaleMaskSoftmax, "softmax(a·x + mask); b ≠ 0: columns past the row's position excluded (causal; the position is the row's index, or the one given: a key span)"),
        new("add_rms_norm", true, true, false, false, false, true, Eps, 0, AddRmsNorm, "h = x + x2 (written to y2); y = rms_norm_affine(h)") { SecondRange = (-4, 4), Offset = true },
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
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * rstd * (g[i] + b);
        return MaxAbs(y) * Math.Sqrt(x.Length);
    }

    private static double RmsNormBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double rstd = Rstd(x, 0, a), t = 0, gm = 0;
        for (int i = 0; i < x.Length; i++) { t += (g[i] + b) * x2[i] * x[i] * rstd; gm = Math.Max(gm, Math.Abs(g[i] + b)); }
        double m = t / x.Length;
        for (int i = 0; i < x.Length; i++) y[i] = rstd * ((g[i] + b) * x2[i] - x[i] * rstd * m);
        return rstd * gm * MaxAbs(x2) * (1 + MaxAbs(x) * rstd * MaxAbs(x) * rstd) * Math.Sqrt(x.Length);
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

    private static double WeightNormBwd(ReadOnlySpan<double> x, ReadOnlySpan<double> x2, ReadOnlySpan<double> g, ReadOnlySpan<double> bt, double s, int pos, int row, double a, double b, Span<double> y, Span<double> y2)
    {
        double ss = 0, dot = 0, abs = 0;
        for (int i = 0; i < x.Length; i++) { ss += x[i] * x[i]; dot += x2[i] * x[i]; abs += Math.Abs(x[i]); }
        double n = Math.Sqrt(ss + a), k = dot / (n * n);
        for (int i = 0; i < x.Length; i++) y[i] = s / n * (x2[i] - x[i] * k);
        y2[0] = dot / n;
        // dot sums cols terms of |dy|·|x|: its rounding reaches every dx through x/n² and the scalar's gradient through 1/n.
        return (1 + Math.Abs(s) / n) * MaxAbs(x2) * (1 + MaxAbs(x) * abs / (n * n)) * Math.Sqrt(x.Length) * (1 + abs / n);
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

    // ---------------------------------------------------------------- group normalization (GroupShape)

    // Each group's mean and rstd, from x or (given) from mean and variance; the computed statistics written to them. Cond:
    // the standardization's conditioning, rstd·max |x| of the group (0 given): the mean's rounding, and the variance's through
    // it, reach x̂ at that times the lane's unit, which a group of nearly equal values makes large.
    private static (double[] Mu, double[] Rstd, double[] Cond) GroupStats(double[] x, GroupShape s, double eps, bool given, double[] mean, double[] variance)
    {
        var mu = new double[s.Groups];
        var rstd = new double[s.Groups];
        var cond = new double[s.Groups];
        int m = s.PerGroup;
        for (int g = 0; g < s.Groups; g++)
        {
            if (!given)
            {
                double sum = 0, ss = 0, big = 0;
                for (int o = 0; o < s.Outer; o++) for (int i = 0; i < s.Inner; i++) big = Math.Max(big, Math.Abs(x[s.Index(o, g, i)]));
                for (int o = 0; o < s.Outer; o++) for (int i = 0; i < s.Inner; i++) sum += x[s.Index(o, g, i)];
                mean[g] = sum / m;
                for (int o = 0; o < s.Outer; o++) for (int i = 0; i < s.Inner; i++) { double d = x[s.Index(o, g, i)] - mean[g]; ss += d * d; }
                variance[g] = ss / m;
                cond[g] = big / Math.Sqrt(variance[g] + eps);
            }

            mu[g] = mean[g];
            rstd[g] = 1 / Math.Sqrt(variance[g] + eps);
        }

        return (mu, rstd, cond);
    }

    /// <summary>Group normalization in float64 (IRows.GroupNorm): y, and each group's mean and biased variance (read from them
    /// when given); returns each group's error scale.</summary>
    public static double[] GroupNormReference(double[] x, double[]? gamma, double[]? beta, GroupShape s, double eps, bool given, double[] mean, double[] variance, double[] y)
    {
        var (mu, rstd, cond) = GroupStats(x, s, eps, given, mean, variance);
        var scale = new double[s.Groups];
        for (int i = 0; i < s.Count; i++)
        {
            int g = s.GroupOf(i), c = s.ChannelOf(i);
            double h = (x[i] - mu[g]) * rstd[g], gm = Math.Abs(gamma?[c] ?? 1);
            y[i] = h * (gamma?[c] ?? 1) + (beta?[c] ?? 0);
            scale[g] = Math.Max(scale[g], (Math.Abs(h) * gm + Math.Abs(beta?[c] ?? 0) + cond[g] * (1 + Math.Abs(h)) * gm) * (given ? 1 : Math.Sqrt(s.PerGroup)));
        }

        return scale;
    }

    /// <summary>GroupNorm's gradients in float64 (IRows.GroupNormBwd): dx, dgamma and dbeta; returns each group's error scale
    /// for dx and each channel's (Σ |dy|·(1 + |x̂|)·(1 + the group's conditioning)) for dgamma and dbeta.</summary>
    public static (double[] Dx, double[] Params) GroupNormBwdReference(double[] x, double[] dy, double[]? gamma, GroupShape s, double eps, bool given, double[] mean, double[] variance, double[] dx, double[] dgamma, double[] dbeta)
    {
        var (mu, rstd, cond) = GroupStats(x, s, eps, given, mean, variance);
        int m = s.PerGroup;
        var s1 = new double[s.Groups];
        var s2 = new double[s.Groups];
        var hMax = new double[s.Groups];
        var gdMax = new double[s.Groups];
        var ps = new double[s.Channels];
        for (int i = 0; i < s.Count; i++)
        {
            int g = s.GroupOf(i), c = s.ChannelOf(i);
            double h = (x[i] - mu[g]) * rstd[g], gd = (gamma?[c] ?? 1) * dy[i];
            s1[g] += gd;
            s2[g] += gd * h;
            hMax[g] = Math.Max(hMax[g], Math.Abs(h));
            gdMax[g] = Math.Max(gdMax[g], Math.Abs(gd));
            dgamma[c] += dy[i] * h;
            dbeta[c] += dy[i];
            ps[c] += Math.Abs(dy[i]) * (1 + Math.Abs(h)) * (1 + cond[g]);
        }

        var scale = new double[s.Groups];
        for (int i = 0; i < s.Count; i++)
        {
            int g = s.GroupOf(i), c = s.ChannelOf(i);
            double h = (x[i] - mu[g]) * rstd[g], gd = (gamma?[c] ?? 1) * dy[i];
            dx[i] = given ? rstd[g] * gd : rstd[g] * (gd - s1[g] / m - h * s2[g] / m);
        }

        for (int g = 0; g < s.Groups; g++)
            scale[g] = rstd[g] * gdMax[g] * (given ? 1 : ((2 + hMax[g] * hMax[g]) + 2 * (1 + hMax[g]) * (1 + hMax[g]) * cond[g]) * Math.Sqrt(m));
        return (scale, ps);
    }

    // ---------------------------------------------------------------- rotary embedding by tables (RopeArgs)

    /// <summary>The table rotary embedding in float64 (IRows.Rope): y; returns each row's error scale.</summary>
    public static double[] RopeReference(double[] x, double[]? gain, double[] cos, double[] sin, int[] positions, RopeArgs a, double[] y)
    {
        int n = a.Cols;
        var scale = new double[a.Rows];
        var v = new double[n];
        for (int r = 0; r < a.Rows; r++)
        {
            var row = x.AsSpan(r * n, n);
            double rstd = 1;
            if (gain is not null) { double ss = 0; foreach (var t in row) ss += t * t; rstd = 1 / Math.Sqrt(ss / n + a.Eps); }
            for (int i = 0; i < n; i++) v[i] = gain is null ? row[i] : row[i] * rstd * (gain[i] + a.Offset);
            int pos = a.Position(positions, r);
            for (int i = 0; i < n; i++) y[r * n + i] = v[i];
            for (int p = 0; p < a.Half; p++)
            {
                var (i, j) = a.Pair(p);
                double c = cos[pos * a.Half + p], sn = a.Inverse ? -sin[pos * a.Half + p] : sin[pos * a.Half + p];
                y[r * n + i] = v[i] * c - v[j] * sn;
                y[r * n + j] = v[i] * sn + v[j] * c;
            }

            scale[r] = 2 * MaxAbs(v) * (gain is null ? 1 : Math.Sqrt(n));
        }

        return scale;
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
