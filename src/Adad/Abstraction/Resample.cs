// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Image resampling by separable filters with a per-channel normalization, in one operation (ILayouts.Resample): Pillow's
// ImagingResample. Each output column's (then row's) first input position, tap count and weights come from the filter's
// support stretched by the shrink factor (so shrinking antialiases); the horizontal pass runs, then the vertical one, a pass
// skipped where its size is kept. The plan (bounds and weights) is made here once, on the host, so every device sums the
// same weights: 8-bit (Pillow's 8bpc) as integers with 22 fractional bits, each pass 2^21 + Σ byte·weight shifted right by
// 22 and clipped to a byte; otherwise as values of the output's float lane, each pass Σ x·w in order by fma. The values then
// map per channel: 8-bit by a table of 256 lane values a channel, the others by a scale and a shift (one fma).
// The devices implement it in their folders: Cpu/Resample.cs; Cuda/GpuOps.Resample.cs (Cuda/Ptx.Resample.cs);
// Vulkan/VulkanOps.Resample.cs (Vulkan/Kernels.Resample.cs).
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>Pillow's resampling filters, by its numbering.</summary>
public enum ResampleFilter { Lanczos = 1, Bilinear = 2, Bicubic = 3, Box = 4, Hamming = 5 }

/// <summary>A resampling's shape: Planes planes of Height × Width (plane p is channel p % Channels) into OutHeight × OutWidth;
/// XTaps and YTaps the weights a column and a row of the plan hold (0: that pass skipped, its size kept); Bytes: Pillow's
/// 8-bit passes (the input uint8).</summary>
public sealed record ResampleArgs(int Planes, int Channels, int Height, int Width, int OutHeight, int OutWidth, int XTaps, int YTaps, bool Bytes)
{
    /// <summary>The horizontal pass's width: the output's, or the input's where the pass is skipped.</summary>
    public int Across => XTaps > 0 ? OutWidth : Width;

    public int Count => Planes * OutHeight * OutWidth;

    public int InputCount => Planes * Height * Width;

    /// <summary>The horizontal pass's values (planes × height × across), held between the passes.</summary>
    public int MiddleCount => Planes * Height * Across;

    /// <summary>Where the rows' bounds and weights start in the plan (after the columns').</summary>
    public int RowBounds => XTaps > 0 ? 2 * OutWidth : 0;

    public int RowWeights => XTaps > 0 ? OutWidth * XTaps : 0;

    public int BoundCount => RowBounds + (YTaps > 0 ? 2 * OutHeight : 0);

    public int WeightCount => RowWeights + (YTaps > 0 ? OutHeight * YTaps : 0);

    /// <summary>The normalization's lane values: 256 a channel (8-bit: a table), else a scale and a shift a channel.</summary>
    public int ValueCount => Bytes ? 256 * Channels : 2 * Channels;

    /// <summary>Throws on a shape the operation does not take.</summary>
    public void Validate(string inputType)
    {
        if (Planes < 0 || Channels <= 0 || Height <= 0 || Width <= 0 || OutHeight <= 0 || OutWidth <= 0 || XTaps < 0 || YTaps < 0) throw new ArgumentException($"resample: {this}");
        if ((XTaps == 0 && OutWidth != Width) || (YTaps == 0 && OutHeight != Height)) throw new ArgumentException("resample: a pass without taps keeps its size");
        if (Bytes && inputType != "uint8") throw new ArgumentException($"resample: Pillow's 8-bit passes read uint8, not {inputType}");
    }
}

/// <summary>The resampling's plan and its float64 reference.</summary>
public static class Resampling
{
    /// <summary>The 8-bit weights' fractional bits (Pillow's PRECISION_BITS: 32 − 8 − 2).</summary>
    public const int PrecisionBits = 22;

    /// <summary>The plan of resizing planes of h × w to oh × ow by filter: the shape, the bounds (each output column's first
    /// input column and count, then each row's) and the weights (XTaps a column, then YTaps a row, past a count 0): 8-bit
    /// Pillow's integers (exact as doubles), else the normalized float64 weights (InLane, LaneBytes round them to the lane).
    /// Both arrays hold one value at least.</summary>
    public static (ResampleArgs Args, int[] Bounds, double[] Weights) Plan(int planes, int channels, int h, int w, int oh, int ow, ResampleFilter filter, bool bytes)
    {
        var cols = w != ow ? Coefficients(w, ow, filter) : default;
        var rows = h != oh ? Coefficients(h, oh, filter) : default;
        var a = new ResampleArgs(planes, channels, h, w, oh, ow, cols.Weights is null ? 0 : cols.Taps, rows.Weights is null ? 0 : rows.Taps, bytes);
        var bounds = new int[Math.Max(1, a.BoundCount)];
        var weights = new double[Math.Max(1, a.WeightCount)];
        if (a.XTaps > 0) { cols.Bounds.CopyTo(bounds, 0); cols.Weights.CopyTo(weights, 0); }
        if (a.YTaps > 0) { rows.Bounds.CopyTo(bounds, a.RowBounds); rows.Weights.CopyTo(weights, a.RowWeights); }
        if (bytes) for (int i = 0; i < weights.Length; i++) weights[i] = Fixed(weights[i]);
        return (a, bounds, weights);
    }

    /// <summary>Pillow's precompute_coeffs for one axis: each output position's first input position and count (bounds, in
    /// pairs) and its weights (taps a position, normalized to sum 1).</summary>
    public static (int[] Bounds, double[] Weights, int Taps) Coefficients(int inSize, int outSize, ResampleFilter filter)
    {
        double support = filter switch
        {
            ResampleFilter.Box => 0.5,
            ResampleFilter.Bilinear or ResampleFilter.Hamming => 1.0,
            ResampleFilter.Bicubic => 2.0,
            ResampleFilter.Lanczos => 3.0,
            _ => throw new ArgumentException($"resample: no filter {filter}"),
        };
        double scale = (double)inSize / outSize, filterScale = Math.Max(scale, 1.0);
        support *= filterScale;
        int taps = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize * 2];
        var weights = new double[outSize * taps];
        var k = new double[taps];
        double ss = 1.0 / filterScale;
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale, sum = 0;
            int xmin = Math.Max(0, (int)(center - support + 0.5));
            int xmax = Math.Min(inSize, (int)(center + support + 0.5)) - xmin;
            for (int x = 0; x < xmax; x++)
            {
                double w = Filter(filter, (x + xmin - center + 0.5) * ss);
                k[x] = w;
                sum += w;
            }

            for (int x = 0; x < xmax; x++) weights[xx * taps + x] = sum != 0 ? k[x] / sum : k[x];
            bounds[2 * xx] = xmin;
            bounds[2 * xx + 1] = Math.Max(0, xmax);
        }

        return (bounds, weights, taps);
    }

    /// <summary>Pillow's normalize_coeffs_8bpc: w · 2^22 rounded half away from zero.</summary>
    public static int Fixed(double w) => w < 0 ? (int)(-0.5 + w * (1 << PrecisionBits)) : (int)(0.5 + w * (1 << PrecisionBits));

    /// <summary>Pillow's clip8: a pass's sum shifted right by 22 and held to a byte.</summary>
    public static int Clip8(int sum) => sum >= 1 << (PrecisionBits + 8) ? 255 : sum <= 0 ? 0 : sum >> PrecisionBits;

    /// <summary>The weights as the operation reads them: 8-bit int32, else the lane's (f64: float64, else float32).</summary>
    public static byte[] WeightBytes(ResampleArgs a, double[] weights, bool f64) =>
        a.Bytes ? MemoryMarshal.AsBytes(weights.Select(v => (int)v).ToArray().AsSpan()).ToArray() : LaneBytes(weights, f64);

    /// <summary>Lane values stored (f64: float64, else float32).</summary>
    public static byte[] LaneBytes(double[] v, bool f64) =>
        f64 ? MemoryMarshal.AsBytes(v.AsSpan()).ToArray() : MemoryMarshal.AsBytes(v.Select(t => (float)t).ToArray().AsSpan()).ToArray();

    /// <summary>Values as the lane holds them.</summary>
    public static double[] InLane(double[] v, bool f64) => f64 ? v : [.. v.Select(t => (double)(float)t)];

    // Pillow's filters (Resample.c), its float constants in Hamming's window as it has them.
    private static double Filter(ResampleFilter filter, double x)
    {
        switch (filter)
        {
            case ResampleFilter.Box:
                return x > -0.5 && x <= 0.5 ? 1.0 : 0.0;
            case ResampleFilter.Bilinear:
                x = Math.Abs(x);
                return x < 1.0 ? 1.0 - x : 0.0;
            case ResampleFilter.Hamming:
                x = Math.Abs(x);
                if (x == 0.0) return 1.0;
                if (x >= 1.0) return 0.0;
                x *= Math.PI;
                return Math.Sin(x) / x * (0.54f + 0.46f * Math.Cos(x));
            case ResampleFilter.Bicubic:
            {
                const double a = -0.5;
                x = Math.Abs(x);
                return x < 1.0 ? ((a + 2.0) * x - (a + 3.0)) * x * x + 1 : x < 2.0 ? (((x - 5) * x + 8) * x - 4) * a : 0.0;
            }

            default:
                return x >= -3.0 && x < 3.0 ? Sinc(x) * Sinc(x / 3) : 0.0;
        }

        static double Sinc(double x)
        {
            if (x == 0.0) return 1.0;
            x *= Math.PI;
            return Math.Sin(x) / x;
        }
    }

    /// <summary>The float64 reference on the input's values (InputCount), the plan's weights as read (8-bit the integers, else
    /// InLane's) and the normalization's values as read (or none): y and each value's error scale in lane ulps — 8-bit 0 (the
    /// integers exact, the table's value read), else (XTaps + YTaps + 2)·|scale|·Σ|w_y|·Σ|w_x·x| + 2|y|: the passes' sums,
    /// the input read into the lane, and the normalization's fma.</summary>
    public static (double[] Y, double[] Scale) Reference(ResampleArgs a, double[] x, int[] bounds, double[] weights, double[]? values)
    {
        int H = a.Height, W = a.Width, OH = a.OutHeight, OW = a.OutWidth, across = a.Across;
        var y = new double[a.Count];
        var scale = new double[a.Count];
        var mid = new double[H * across];
        var mabs = new double[H * across];
        double terms = a.XTaps + a.YTaps + 2;
        for (int p = 0; p < a.Planes; p++)
        {
            int c = p % a.Channels;
            for (int r = 0; r < H; r++)
            {
                for (int ox = 0; ox < across; ox++)
                {
                    int row = (p * H + r) * W;
                    double s = 0, sa = 0;
                    if (a.XTaps == 0) { s = x[row + ox]; sa = Math.Abs(s); }
                    else
                    {
                        int first = bounds[2 * ox], n = bounds[2 * ox + 1], wb = ox * a.XTaps;
                        if (a.Bytes)
                        {
                            int sum = 1 << (PrecisionBits - 1);
                            for (int t = 0; t < n; t++) sum += (int)x[row + first + t] * (int)weights[wb + t];
                            s = Clip8(sum);
                        }
                        else
                        {
                            for (int t = 0; t < n; t++) { s += x[row + first + t] * weights[wb + t]; sa += Math.Abs(x[row + first + t] * weights[wb + t]); }
                        }
                    }

                    mid[r * across + ox] = s;
                    mabs[r * across + ox] = sa;
                }
            }

            for (int oy = 0; oy < OH; oy++)
            {
                for (int ox = 0; ox < OW; ox++)
                {
                    int o = (p * OH + oy) * OW + ox;
                    double v = 0, va = 0;
                    if (a.YTaps == 0) { v = mid[oy * across + ox]; va = mabs[oy * across + ox]; }
                    else
                    {
                        int first = bounds[a.RowBounds + 2 * oy], n = bounds[a.RowBounds + 2 * oy + 1], wb = a.RowWeights + oy * a.YTaps;
                        if (a.Bytes)
                        {
                            int sum = 1 << (PrecisionBits - 1);
                            for (int t = 0; t < n; t++) sum += (int)mid[(first + t) * across + ox] * (int)weights[wb + t];
                            v = Clip8(sum);
                        }
                        else
                        {
                            for (int t = 0; t < n; t++) { v += mid[(first + t) * across + ox] * weights[wb + t]; va += Math.Abs(weights[wb + t]) * mabs[(first + t) * across + ox]; }
                        }
                    }

                    if (a.Bytes)
                    {
                        y[o] = values is null ? v : values[256 * c + (int)v];
                        continue;
                    }

                    double sc = values is null ? 1 : Math.Abs(values[2 * c]);
                    y[o] = values is null ? v : v * values[2 * c] + values[2 * c + 1];
                    scale[o] = terms * sc * va + 2 * Math.Abs(y[o]);
                }
            }
        }

        return (y, scale);
    }
}
