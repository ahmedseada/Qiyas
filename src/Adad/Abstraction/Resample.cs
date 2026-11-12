// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling planes of values by a separable filter, with a scale and a shift a channel, in one operation
// (ILayouts.Resample). Output position o of an axis of In values resampled to Out has its centre at c = (o + ½)·In/Out in
// the input's coordinates (input position i at i + ½); the filter's kernel k is stretched by s = max(1, In/Out), so a
// shrinking pass averages over the input each output covers; its taps are the input positions whose centres lie within
// the stretched support around c, each weighted k((i + ½ − c)/s), the weights normalized to sum 1. The horizontal pass
// runs, then the vertical one, a pass skipped where its size is kept. The plan (bounds and weights) is made here once, on
// the host, in float64 and rounded once to the output's float lane (Fmt.FloatLane), so every device sums the same
// weights; each pass is Σ x·w in the lane in order by fma, the input read where it is stored. Then y = v·scale[c] +
// shift[c] (one fma), and y rounded to the output's type once by its own conversion (every type: an integer's or bool's
// to nearest, ties to even, held to its range).
// The devices implement it in their folders: Cpu/Resample.cs; Cuda/GpuOps.Resample.cs (Cuda/Ptx.Resample.cs);
// Vulkan/VulkanOps.Resample.cs (Vulkan/Kernels.Resample.cs).
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>A resampling filter's kernel k(x) (Resampling.Kernel): the box, 1 on (−½, ½]; the triangle, 1 − |x| on |x| &lt; 1
/// (linear interpolation); Hamming's, sinc(x)·(0.54 + 0.46·cos πx) on |x| &lt; 1; the cubic convolution with a = −½ (the a
/// whose interpolation is exact for quadratics) on |x| &lt; 2; Lanczos's, sinc(x)·sinc(x/3) on |x| &lt; 3.</summary>
public enum ResampleFilter { Box, Triangle, Hamming, Cubic, Lanczos }

/// <summary>A resampling's shape: Planes planes of Height × Width (plane p is channel p % Channels) into OutHeight × OutWidth;
/// XTaps and YTaps the weights a column and a row of the plan hold (0: that pass skipped, its size kept).</summary>
public sealed record ResampleArgs(int Planes, int Channels, int Height, int Width, int OutHeight, int OutWidth, int XTaps, int YTaps)
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

    /// <summary>The channels' lane values: a scale and a shift a channel.</summary>
    public int ValueCount => 2 * Channels;

    /// <summary>Throws on a shape the operation does not take.</summary>
    public void Validate()
    {
        if (Planes < 0 || Channels <= 0 || Height <= 0 || Width <= 0 || OutHeight <= 0 || OutWidth <= 0 || XTaps < 0 || YTaps < 0) throw new ArgumentException($"resample: {this}");
        if ((XTaps == 0 && OutWidth != Width) || (YTaps == 0 && OutHeight != Height)) throw new ArgumentException("resample: a pass without taps keeps its size");
    }
}

/// <summary>The resampling's plan and its float64 reference.</summary>
public static class Resampling
{
    /// <summary>The plan of resizing planes of h × w to oh × ow by filter: the shape, the bounds (each output column's first
    /// input column and count, then each row's) and the normalized float64 weights (XTaps a column, then YTaps a row, past a
    /// count 0; InLane and LaneBytes round them to the lane). Both arrays hold one value at least.</summary>
    public static (ResampleArgs Args, int[] Bounds, double[] Weights) Plan(int planes, int channels, int h, int w, int oh, int ow, ResampleFilter filter)
    {
        // An axis whose size stays has no taps: no coefficients.
        (int[] Bounds, double[] Weights, int Taps) none = ([], [], 0);
        var cols = w != ow ? Coefficients(w, ow, filter) : none;
        var rows = h != oh ? Coefficients(h, oh, filter) : none;
        var a = new ResampleArgs(planes, channels, h, w, oh, ow, cols.Taps, rows.Taps);
        var bounds = new int[Math.Max(1, a.BoundCount)];
        var weights = new double[Math.Max(1, a.WeightCount)];
        if (a.XTaps > 0) { cols.Bounds.CopyTo(bounds, 0); cols.Weights.CopyTo(weights, 0); }
        if (a.YTaps > 0) { rows.Bounds.CopyTo(bounds, a.RowBounds); rows.Weights.CopyTo(weights, a.RowWeights); }
        return (a, bounds, weights);
    }

    /// <summary>The kernel's support: k is 0 at |x| at least it.</summary>
    public static double Support(ResampleFilter filter) => filter switch
    {
        ResampleFilter.Box => 0.5,
        ResampleFilter.Triangle or ResampleFilter.Hamming => 1.0,
        ResampleFilter.Cubic => 2.0,
        ResampleFilter.Lanczos => 3.0,
        _ => throw new ArgumentException($"resample: no filter {filter}"),
    };

    /// <summary>One axis of In values resampled to Out: each output position's first input position and count (bounds, in
    /// pairs) and its weights (taps a position, normalized to sum 1), by the rule at the top of this file.</summary>
    public static (int[] Bounds, double[] Weights, int Taps) Coefficients(int inSize, int outSize, ResampleFilter filter)
    {
        double scale = (double)inSize / outSize, stretch = Math.Max(scale, 1.0), support = Support(filter) * stretch;
        int taps = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize * 2];
        var weights = new double[outSize * taps];
        var k = new double[taps];
        for (int o = 0; o < outSize; o++)
        {
            // The positions whose centres i + ½ lie within the support around the output's centre c.
            double c = (o + 0.5) * scale, sum = 0;
            int first = Math.Max(0, (int)Math.Floor(c - support + 0.5));
            int count = Math.Min(inSize, (int)Math.Floor(c + support + 0.5)) - first;
            for (int i = 0; i < count; i++)
            {
                double v = Kernel(filter, (first + i + 0.5 - c) / stretch);
                k[i] = v;
                sum += v;
            }

            for (int i = 0; i < count; i++) weights[o * taps + i] = sum != 0 ? k[i] / sum : k[i];
            bounds[2 * o] = first;
            bounds[2 * o + 1] = Math.Max(0, count);
        }

        return (bounds, weights, taps);
    }

    /// <summary>Lane values stored (f64: float64, else float32).</summary>
    public static byte[] LaneBytes(double[] v, bool f64) =>
        f64 ? MemoryMarshal.AsBytes(v.AsSpan()).ToArray() : MemoryMarshal.AsBytes(v.Select(t => (float)t).ToArray().AsSpan()).ToArray();

    /// <summary>Values as the lane holds them.</summary>
    public static double[] InLane(double[] v, bool f64) => f64 ? v : [.. v.Select(t => (double)(float)t)];

    /// <summary>The filter's kernel k(x) in float64 (ResampleFilter).</summary>
    public static double Kernel(ResampleFilter filter, double x)
    {
        switch (filter)
        {
            case ResampleFilter.Box:
                return x > -0.5 && x <= 0.5 ? 1.0 : 0.0;
            case ResampleFilter.Triangle:
                x = Math.Abs(x);
                return x < 1.0 ? 1.0 - x : 0.0;
            case ResampleFilter.Hamming:
                x = Math.Abs(x);
                return x < 1.0 ? Sinc(x) * (0.54 + 0.46 * Math.Cos(Math.PI * x)) : 0.0;
            case ResampleFilter.Cubic:
            {
                const double a = -0.5;
                x = Math.Abs(x);
                return x < 1.0 ? ((a + 2.0) * x - (a + 3.0)) * x * x + 1 : x < 2.0 ? (((x - 5) * x + 8) * x - 4) * a : 0.0;
            }

            case ResampleFilter.Lanczos:
                return Math.Abs(x) < 3.0 ? Sinc(x) * Sinc(x / 3) : 0.0;
            default:
                throw new ArgumentException($"resample: no filter {filter}");
        }

        static double Sinc(double x)
        {
            if (x == 0.0) return 1.0;
            x *= Math.PI;
            return Math.Sin(x) / x;
        }
    }

    /// <summary>The float64 reference on the input's values (InputCount), the plan's weights as read (InLane's) and the
    /// channels' scale and shift as read (or none): y and each value's error scale in lane ulps, (XTaps + YTaps + 2)·|scale|·
    /// Σ|w_y|·Σ|w_x·x| + 2|y|: the passes' sums, the input read into the lane, and the scale and shift's fma.</summary>
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
                        for (int t = 0; t < n; t++) { s += x[row + first + t] * weights[wb + t]; sa += Math.Abs(x[row + first + t] * weights[wb + t]); }
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
                        for (int t = 0; t < n; t++) { v += mid[(first + t) * across + ox] * weights[wb + t]; va += Math.Abs(weights[wb + t]) * mabs[(first + t) * across + ox]; }
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
