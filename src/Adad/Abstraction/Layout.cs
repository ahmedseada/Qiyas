// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Layout: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Layout.cs; Cuda/GpuOps.Layout.cs with its kernels in Cuda/Ptx.Layout.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>A layout operation's arguments: the stored input, an index array where the operation has one, integer
/// parameters in the operation's order, and the output (stored as the type; count values).</summary>
public sealed class LayoutArgs
{
    public required string Name;
    public required byte[] X;
    public byte[]? X2;                       // a second input of the type (depthwise_conv's weights; maxpool_bwd's x alongside dy)
    public int[]? Index;
    public required int[] P;
    public required byte[] Output;
    public required int Count;
    public int InputCount;

    /// <summary>The second input's values: the forward input of a maxpool backward, the weights of a depthwise convolution.</summary>
    public int SecondCount => Name switch
    {
        "maxpool_bwd" or "avgpool_bwd" or "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => P[0] * P[1] * P[2],
        "depthwise_conv" => P[0] * P[3] * P[4],
        _ => 0,
    };

    public static readonly string[] Names = ["strided", "one_hot", "gather_rows", "gather_cols", "scatter_add_rows", "im2col", "col2im", "interpolate_nearest", "interpolate_bilinear", "shift", "pad2d",
        "maxpool", "avgpool", "maxpool_bwd", "avgpool_bwd", "adaptive_maxpool", "adaptive_avgpool", "depthwise_conv", "adaptive_maxpool_bwd", "adaptive_avgpool_bwd"];

    /// <summary>Whether the operation applies to the type: bilinear interpolation and the average pools compute in a float lane.</summary>
    public static bool Applies(string name, bool floatType) => floatType || name is not ("interpolate_bilinear" or "avgpool" or "avgpool_bwd" or "adaptive_avgpool" or "adaptive_avgpool_bwd");

    /// <summary>Whether the operation sums values (its lane follows the sum rule).</summary>
    public static bool Sums(string name) => name is "scatter_add_rows" or "col2im" or "maxpool_bwd" or "depthwise_conv" or "adaptive_maxpool_bwd";

    /// <summary>Whether the operation computes in the float lane (weighted sums of float values).</summary>
    public static bool FloatLane(string name) => name is "interpolate_bilinear" or "avgpool" or "avgpool_bwd" or "adaptive_avgpool" or "adaptive_avgpool_bwd";

    /// <summary>About how many input values an output value reads: a window's for the pools and the depthwise
    /// convolution, 1 for the moves (the team's measure of a job, which a few outputs over large windows otherwise
    /// understate: an adaptive pool's 3136 outputs of 37 × 37 windows ran on the caller alone).</summary>
    public static long ValueReads(string name, int[] p) => name switch
    {
        "maxpool" or "avgpool" or "depthwise_conv" or "maxpool_bwd" or "avgpool_bwd" or "col2im" => (long)p[3] * p[4],
        "adaptive_maxpool" or "adaptive_avgpool" => (long)(p[1] / p[3] + 2) * (p[2] / p[4] + 2),
        "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => (long)(p[1] / p[3] + 2) * (p[2] / p[4] + 2),
        _ => 1,
    };

    /// <summary>The number of terms a summed value has (the sum rule's bound).</summary>
    public static int Terms(string name, int[] p) => name switch
    {
        "scatter_add_rows" => p[0],
        "col2im" or "maxpool_bwd" or "depthwise_conv" => p[3] * p[4],
        "adaptive_maxpool_bwd" => (p[1] / p[3] + 2) * (p[2] / p[4] + 2),
        _ => 1,
    };

    // ---- the shapes behind P, in each operation's order
    // strided: d0 d1 d2 d3 | s0 s1 s2 s3 | base            output d0·d1·d2·d3
    // one_hot: n classes                                   output n·classes
    // gather_rows: rows cols n                             output n·cols
    // gather_cols: rows cols m                             output rows·m
    // scatter_add_rows: n cols rows                        output rows·cols
    // im2col / col2im: C H W kh kw sh sw ph pw dh dw       cols (C·kh·kw) × (OH·OW)
    // interpolate: C H W OH OW                             output C·OH·OW
    // shift: rows cols k                                   output rows·cols
    // pad2d: rows cols newCols                             output rows·newCols

    /// <summary>A copy with another output buffer (the bench's per-subject outputs).</summary>
    public LayoutArgs With(byte[] output) => new() { Name = Name, X = X, X2 = X2, Index = Index, P = P, Output = output, Count = Count, InputCount = InputCount };

    /// <summary>The output as rows of values (the natural row of each operation): a thread owns rows on the CPU, a word on the device.</summary>
    public static (int Rows, int Cols) OutputRows(LayoutArgs k)
    {
        var p = k.P;
        return k.Name switch
        {
            "strided" => (p[0] * p[1] * p[2], p[3]),
            "one_hot" => (p[0], p[1]),
            "gather_rows" => (p[2], p[1]),
            "gather_cols" => (p[0], p[2]),
            "scatter_add_rows" => (p[2], p[1]),
            "im2col" => (p[0] * p[3] * p[4], k.Count / (p[0] * p[3] * p[4])),
            "col2im" => (p[0] * p[1], p[2]),
            "interpolate_nearest" or "interpolate_bilinear" => (p[0] * p[3], p[4]),
            "shift" => (p[0], p[1]),
            "pad2d" => (p[0], p[2]),
            "maxpool" or "avgpool" or "depthwise_conv" => (p[0] * PoolOut(p).OH, PoolOut(p).OW),
            "maxpool_bwd" or "avgpool_bwd" => (p[0] * p[1], p[2]),
            "adaptive_maxpool" or "adaptive_avgpool" => (p[0] * p[3], p[4]),
            "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => (p[0] * p[1], p[2]),
            _ => throw new ArgumentException(k.Name),
        };
    }

    /// <summary>A pool's output size: C H W kh kw sh sw ph pw (dilation 1).</summary>
    public static (int OH, int OW) PoolOut(int[] p)
    {
        int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
        int dh = p.Length > 9 ? p[9] : 1, dw = p.Length > 10 ? p[10] : 1;
        return ((H + 2 * ph - dh * (kh - 1) - 1) / sh + 1, (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
    }

    /// <summary>An adaptive window along one axis: [⌊i·n/o⌋, ⌈(i + 1)·n/o⌉).</summary>
    public static (int From, int To) Adaptive(int i, int n, int o) => (i * n / o, ((i + 1) * n + o - 1) / o);

    public static (int OH, int OW) ConvOut(int[] p)
    {
        int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
        return ((H + 2 * ph - dh * (kh - 1) - 1) / sh + 1, (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
    }

    /// <summary>The float64 reference of the operation on the input's values (x2: the second input's).</summary>
    public static double[] Reference(string name, double[] x, int[]? index, int[] p, int count, double[]? x2 = null)
    {
        var y = new double[count];
        switch (name)
        {
            case "maxpool":
            case "avgpool":
            {
                var (OH, OW) = PoolOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double best = double.NegativeInfinity, sum = 0;
                    int n = 0;
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - ph + a, iw = ow * sw - pw + b;
                        if ((uint)ih >= (uint)H || (uint)iw >= (uint)W) continue;
                        double v = x[(c * H + ih) * W + iw];
                        best = Math.Max(best, v); sum += v; n++;
                    }

                    y[(c * OH + oh) * OW + ow] = name == "maxpool" ? (n == 0 ? 0 : best) : (n == 0 ? 0 : sum / n);
                }

                return y;
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // x = dy (C × OH × OW), x2 = the forward input (maxpool: the window's first largest takes the gradient).
                var (OH, OW) = PoolOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double dy = x[(c * OH + oh) * OW + ow], best = double.NegativeInfinity;
                    int n = 0, bestAt = -1;
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - ph + a, iw = ow * sw - pw + b;
                        if ((uint)ih >= (uint)H || (uint)iw >= (uint)W) continue;
                        n++;
                        if (name == "maxpool_bwd") { double v = x2![(c * H + ih) * W + iw]; if (v > best) { best = v; bestAt = (c * H + ih) * W + iw; } }
                        else y[(c * H + ih) * W + iw] += 0;    // counted below
                    }

                    if (name == "maxpool_bwd") { if (bestAt >= 0) y[bestAt] += dy; continue; }
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - ph + a, iw = ow * sw - pw + b;
                        if ((uint)ih >= (uint)H || (uint)iw >= (uint)W) continue;
                        y[(c * H + ih) * W + iw] += dy / n;
                    }
                }

                return y;
            }

            case "adaptive_maxpool":
            case "adaptive_avgpool":
            {
                int C = p[0], H = p[1], W = p[2], OH = p[3], OW = p[4];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    var (h0, h1) = Adaptive(oh, H, OH);
                    var (w0, w1) = Adaptive(ow, W, OW);
                    double best = double.NegativeInfinity, sum = 0;
                    for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++) { double v = x[(c * H + ih) * W + iw]; best = Math.Max(best, v); sum += v; }
                    y[(c * OH + oh) * OW + ow] = name == "adaptive_maxpool" ? best : sum / ((h1 - h0) * (w1 - w0));
                }

                return y;
            }

            case "adaptive_maxpool_bwd":
            case "adaptive_avgpool_bwd":
            {
                // x = dy (C × OH × OW), x2 = the forward input (max): every window's gradient to its first largest, or spread over the window.
                int C = p[0], H = p[1], W = p[2], OH = p[3], OW = p[4];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    var (h0, h1) = Adaptive(oh, H, OH);
                    var (w0, w1) = Adaptive(ow, W, OW);
                    double dy = x[(c * OH + oh) * OW + ow];
                    if (name == "adaptive_avgpool_bwd")
                    {
                        double share = dy / ((h1 - h0) * (w1 - w0));
                        for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++) y[(c * H + ih) * W + iw] += share;
                        continue;
                    }

                    double best = double.NegativeInfinity;
                    int bestAt = -1;
                    for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++) { double v = x2![(c * H + ih) * W + iw]; if (bestAt < 0 || v > best) { best = v; bestAt = (c * H + ih) * W + iw; } }
                    y[bestAt] += dy;
                }

                return y;
            }

            case "depthwise_conv":
            {
                var (OH, OW) = PoolOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double acc = 0;
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - ph + a * dh, iw = ow * sw - pw + b * dw;
                        if ((uint)ih >= (uint)H || (uint)iw >= (uint)W) continue;
                        acc += x2![(c * kh + a) * kw + b] * x[(c * H + ih) * W + iw];
                    }

                    y[(c * OH + oh) * OW + ow] = acc;
                }

                return y;
            }

            case "strided":
            {
                int d0 = p[0], d1 = p[1], d2 = p[2], d3 = p[3];
                for (int i0 = 0; i0 < d0; i0++) for (int i1 = 0; i1 < d1; i1++) for (int i2 = 0; i2 < d2; i2++) for (int i3 = 0; i3 < d3; i3++)
                    y[((i0 * d1 + i1) * d2 + i2) * d3 + i3] = x[p[8] + i0 * p[4] + i1 * p[5] + i2 * p[6] + i3 * p[7]];
                return y;
            }

            case "one_hot":
                for (int i = 0; i < p[0]; i++) if ((uint)index![i] < (uint)p[1]) y[i * p[1] + index[i]] = 1;
                return y;
            case "gather_rows":
                for (int i = 0; i < p[2]; i++) Array.Copy(x, index![i] * p[1], y, i * p[1], p[1]);
                return y;
            case "gather_cols":
                for (int r = 0; r < p[0]; r++) for (int j = 0; j < p[2]; j++) y[r * p[2] + j] = x[r * p[1] + index![j]];
                return y;
            case "scatter_add_rows":
                for (int i = 0; i < p[0]; i++) for (int c = 0; c < p[1]; c++) y[index![i] * p[1] + c] += x[i * p[1] + c];
                return y;
            case "im2col":
            {
                var (OH, OW) = ConvOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4];
                for (int c = 0; c < C; c++) for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                    {
                        int ih = oh * p[5] - p[7] + a * p[9], iw = ow * p[6] - p[8] + b * p[10];
                        y[((c * kh + a) * kw + b) * (OH * OW) + oh * OW + ow] = (uint)ih < (uint)H && (uint)iw < (uint)W ? x[(c * H + ih) * W + iw] : 0;
                    }
                return y;
            }

            case "col2im":
            {
                var (OH, OW) = ConvOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4];
                for (int c = 0; c < C; c++) for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                    {
                        int ih = oh * p[5] - p[7] + a * p[9], iw = ow * p[6] - p[8] + b * p[10];
                        if ((uint)ih < (uint)H && (uint)iw < (uint)W) y[(c * H + ih) * W + iw] += x[((c * kh + a) * kw + b) * (OH * OW) + oh * OW + ow];
                    }
                return y;
            }

            case "interpolate_nearest":
            case "interpolate_bilinear":
            {
                int C = p[0], H = p[1], W = p[2], OH = p[3], OW = p[4];
                bool bilinear = name == "interpolate_bilinear";
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double sy = (oh + 0.5) * H / OH - 0.5, sx = (ow + 0.5) * W / OW - 0.5;
                    if (!bilinear)
                    {
                        // floor((oh + ½)·H / OH) exactly: (2·oh + 1)·H / (2·OH) in integers.
                        int ih = Math.Min(H - 1, (2 * oh + 1) * H / (2 * OH)), iw = Math.Min(W - 1, (2 * ow + 1) * W / (2 * OW));
                        y[(c * OH + oh) * OW + ow] = x[(c * H + ih) * W + iw];
                        continue;
                    }

                    sy = Math.Max(sy, 0); sx = Math.Max(sx, 0);
                    int y0 = Math.Min((int)Math.Floor(sy), H - 1), x0 = Math.Min((int)Math.Floor(sx), W - 1), y1 = Math.Min(y0 + 1, H - 1), x1 = Math.Min(x0 + 1, W - 1);
                    double fy = sy - y0, fx = sx - x0;
                    double top = x[(c * H + y0) * W + x0] * (1 - fx) + x[(c * H + y0) * W + x1] * fx;
                    double bottom = x[(c * H + y1) * W + x0] * (1 - fx) + x[(c * H + y1) * W + x1] * fx;
                    y[(c * OH + oh) * OW + ow] = top * (1 - fy) + bottom * fy;
                }
                return y;
            }

            case "shift":
                for (int r = 0; r < p[0]; r++) for (int c = 0; c < p[1]; c++) { int s = c - p[2]; y[r * p[1] + c] = (uint)s < (uint)p[1] ? x[r * p[1] + s] : 0; }
                return y;
            case "pad2d":
                for (int r = 0; r < p[0]; r++) for (int c = 0; c < p[2]; c++) y[r * p[2] + c] = c < p[1] ? x[r * p[1] + c] : 0;
                return y;
            default: throw new ArgumentException($"unknown layout operation {name}");
        }
    }
}

// ---------------------------------------------------------------------------------------------- the CPU kernels
