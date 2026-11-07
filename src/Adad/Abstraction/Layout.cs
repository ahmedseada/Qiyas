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
        "maxpool", "avgpool", "maxpool_bwd", "avgpool_bwd", "adaptive_maxpool", "adaptive_avgpool", "depthwise_conv", "adaptive_maxpool_bwd", "adaptive_avgpool_bwd",
        "interpolate_nearest_bwd", "interpolate_bilinear_bwd"];

    /// <summary>Whether the operation applies to the type: bilinear interpolation and the average pools compute in a float lane.</summary>
    public static bool Applies(string name, bool floatType) => floatType || !FloatLane(name);

    /// <summary>Whether the operation sums values (its lane follows the sum rule).</summary>
    public static bool Sums(string name) => name is "scatter_add_rows" or "col2im" or "maxpool_bwd" or "depthwise_conv" or "adaptive_maxpool_bwd";

    /// <summary>Whether the operation computes in the float lane (weighted sums of float values).</summary>
    public static bool FloatLane(string name) => name is "interpolate_bilinear" or "avgpool" or "avgpool_bwd" or "adaptive_avgpool" or "adaptive_avgpool_bwd"
        or "interpolate_nearest_bwd" or "interpolate_bilinear_bwd";

    /// <summary>About how many input values an output value reads: a window's for the pools and the depthwise
    /// convolution, 1 for the moves (the team's measure of a job, which a few outputs over large windows otherwise
    /// understate: an adaptive pool's 3136 outputs of 37 × 37 windows ran on the caller alone).</summary>
    public static long ValueReads(string name, int[] p) => name switch
    {
        "maxpool" or "avgpool" or "depthwise_conv" or "maxpool_bwd" or "avgpool_bwd" or "col2im" => (long)p[3] * p[4],
        "adaptive_maxpool" or "adaptive_avgpool" => (long)(p[1] / p[3] + 2) * (p[2] / p[4] + 2),
        "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => (long)(p[1] / p[3] + 2) * (p[2] / p[4] + 2),
        "interpolate_nearest_bwd" or "interpolate_bilinear_bwd" => (long)(p[3] / p[1] + 2) * (p[4] / p[2] + 2),
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
    // strided: d0 d1 d2 d3 | s0 s1 s2 s3 | base            output d0·d1·d2·d3 (or six of each: d0..d5 | s0..s5 | base)
    // one_hot: n classes                                   output n·classes
    // gather_rows: rows cols n                             output n·cols
    // gather_cols: rows cols m                             output rows·m
    // scatter_add_rows: n cols rows                        output rows·cols
    // im2col / col2im: C H W kh kw sh sw ph pw dh dw       cols (C·kh·kw) × (OH·OW)
    // interpolate: C H W OH OW [mode [numH denH numW denW]]   output C·OH·OW; mode 0 half-pixel centres, 1 align corners
    //              (bilinear), 2 asymmetric (o·scale: PyTorch's nearest); the scale in/out, or num/den when given (1/scale_factor)
    // interpolate_*_bwd: the forward's parameters            dx (C·H·W) from dy (C·OH·OW)
    // pools: C H W kh kw sh sw ph pw [dh dw [ceil [countIncludePad]]]   (dilation 1 for the pools; ceil: PyTorch's ceil_mode)
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
            "strided" => p.Length == 13 ? (p[0] * p[1] * p[2] * p[3] * p[4], p[5]) : (p[0] * p[1] * p[2], p[3]),
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
            "interpolate_nearest_bwd" or "interpolate_bilinear_bwd" => (p[0] * p[1], p[2]),
            _ => throw new ArgumentException(k.Name),
        };
    }

    /// <summary>A pool's output size: C H W kh kw sh sw ph pw [dh dw [ceil]] (ceil: PyTorch's ceil_mode).</summary>
    public static (int OH, int OW) PoolOut(int[] p)
    {
        int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
        int dh = p.Length > 9 ? p[9] : 1, dw = p.Length > 10 ? p[10] : 1;
        bool ceil = p.Length > 11 && p[11] != 0;
        return (PoolSize(H, kh, sh, ph, dh, ceil), PoolSize(W, kw, sw, pw, dw, ceil));
    }

    /// <summary>The windows along an axis of n values: floor, or with ceil the windows that start inside the input or its
    /// leading pad (PyTorch's rule: the last one may run past the trailing pad).</summary>
    public static int PoolSize(int n, int k, int s, int pad, int d, bool ceil)
    {
        int span = n + 2 * pad - d * (k - 1) - 1;
        if (!ceil) return span / s + 1;
        int o = (span + s - 1) / s + 1;
        return (o - 1) * s >= n + pad ? o - 1 : o;
    }

    /// <summary>An average pool's divisor for window (oh, ow) of `inside` input values: those, or counting the pads (p[12]) the
    /// window's positions up to the padded end H + ph, W + pw (PyTorch's count_include_pad).</summary>
    public static int PoolDivisor(int[] p, int oh, int ow, int inside)
    {
        if (p.Length <= 12 || p[12] == 0) return inside;
        int hs = oh * p[5] - p[7], ws = ow * p[6] - p[8];
        return (Math.Min(hs + p[3], p[1] + p[7]) - hs) * (Math.Min(ws + p[4], p[2] + p[8]) - ws);
    }

    // ---- the parameters as the kernels take them

    /// <summary>A copy with the parameters the kernels read (KernelP).</summary>
    public LayoutArgs ForKernels() => new() { Name = Name, X = X, X2 = X2, Index = Index, P = KernelP(Name, P), Output = Output, Count = Count, InputCount = InputCount };

    /// <summary>The parameters the device kernels read: a strided view canonical (StridedKernel), an interpolation's axes as
    /// C H W OH OW αh βh δh αw βw δw (InterpolateAxis); the others as given.</summary>
    public static int[] KernelP(string name, int[] p) => name switch
    {
        "strided" => StridedKernel(p),
        "interpolate_nearest" or "interpolate_bilinear" or "interpolate_nearest_bwd" or "interpolate_bilinear_bwd" => InterpolateKernel(name, p),
        // The pools take positions 9 and 10 (depthwise_conv's dilations) only so ceil and the pad count keep one order: 1.
        "maxpool" or "avgpool" or "maxpool_bwd" or "avgpool_bwd" when p.Length > 9 && (p[9] != 1 || p[10] != 1)
            => throw new ArgumentException($"{name}: the pools' dilations (P[9], P[10]) are 1"),
        _ => p,
    };

    /// <summary>A strided view (four or six dimensions) canonical: its dimensions of size 1 dropped and each that continues the
    /// one inside it (stride = the inner's stride × its size) merged into it; four or fewer then as d0..d3 s0..s3 base, more
    /// as d0..d5 s0..s5 base (leading dimensions of size 1). A view with no values is kept as given.</summary>
    public static int[] StridedKernel(int[] p)
    {
        int rank = (p.Length - 1) / 2;
        for (int i = 0; i < rank; i++) if (p[i] == 0) return p;
        var d = new List<int>();
        var st = new List<int>();
        for (int i = 0; i < rank; i++)
        {
            if (p[i] == 1) continue;
            if (d.Count > 0 && st[^1] == (long)p[rank + i] * p[i]) { d[^1] *= p[i]; st[^1] = p[rank + i]; continue; }
            d.Add(p[i]);
            st.Add(p[rank + i]);
        }

        int r = d.Count <= 4 ? 4 : 6, lead = r - d.Count;
        var q = new int[2 * r + 1];
        for (int i = 0; i < r; i++)
        {
            q[i] = i < lead ? 1 : d[i - lead];
            q[r + i] = i < lead ? 0 : st[i - lead];
        }

        q[2 * r] = p[2 * rank];
        return q;
    }

    /// <summary>An interpolation's mode (p[5]: 0 half-pixel, 1 align corners, 2 asymmetric) and scales (num/den per axis, p[6..9];
    /// 0: in/out).</summary>
    public static (int Mode, int NumH, int DenH, int NumW, int DenW) Interpolation(int[] p) =>
        (p.Length > 5 ? p[5] : 0, p.Length > 9 ? p[6] : 0, p.Length > 9 ? p[7] : 0, p.Length > 9 ? p[8] : 0, p.Length > 9 ? p[9] : 0);

    /// <summary>An axis of an interpolation as integers: output o's source coordinate is (o·α + β) / δ exactly — bilinear's
    /// clamped at 0 below (its rows the floor and the next, held to the input, the weight the remainder / δ), nearest's floor
    /// (held to the input). Half-pixel: ((2o + 1)·num − den) / 2den (bilinear), (2o + 1)·num / 2den (nearest); align corners:
    /// o·(in − 1) / (out − 1); asymmetric: o·num / den.</summary>
    public static (int Alpha, int Beta, int Delta) InterpolateAxis(bool bilinear, int mode, int inp, int outp, int num, int den)
    {
        if (num <= 0 || den <= 0) { num = inp; den = outp; }
        return mode switch
        {
            0 => bilinear ? (2 * num, num - den, 2 * den) : (2 * num, num, 2 * den),
            1 when bilinear => outp > 1 ? (inp - 1, 0, outp - 1) : (0, 0, 1),
            1 => throw new ArgumentException("nearest interpolation has no align-corners mode"),
            2 => (num, 0, den),
            _ => throw new ArgumentException($"interpolation mode {mode}"),
        };
    }

    private static int[] InterpolateKernel(string name, int[] p)
    {
        bool bilinear = name.StartsWith("interpolate_bilinear", StringComparison.Ordinal);
        var (mode, nh, dh, nw, dw) = Interpolation(p);
        var (ah, bh, ch) = InterpolateAxis(bilinear, mode, p[1], p[3], nh, dh);
        var (aw, bw, cw) = InterpolateAxis(bilinear, mode, p[2], p[4], nw, dw);
        return [p[0], p[1], p[2], p[3], p[4], ah, bh, ch, aw, bw, cw];
    }

    /// <summary>Bilinear output o's two source positions along an axis (InterpolateAxis) and the second's weight's numerator
    /// over δ: past the input's last position both are the last, the weight 0.</summary>
    public static (int I0, int I1, int Rem) BilinearAt(int o, int inp, int alpha, int beta, int delta)
    {
        long t = Math.Max((long)o * alpha + beta, 0);
        long i0 = t / delta;
        if (i0 >= inp - 1) return (inp - 1, inp - 1, 0);
        return ((int)i0, (int)i0 + 1, (int)(t - i0 * delta));
    }

    /// <summary>Nearest output o's source position along an axis (InterpolateAxis).</summary>
    public static int NearestAt(int o, int inp, int alpha, int beta, int delta) => (int)Math.Min(((long)o * alpha + beta) / delta, inp - 1);

    /// <summary>The outputs [lo, hi] along an axis whose source rows include input row i (a backward's gather): bilinear's
    /// first row i − 1 or i, nearest's i; the last input row also takes every output past it.</summary>
    public static (int Lo, int Hi) InterpolateSources(bool bilinear, int i, int inp, int outp, int alpha, int beta, int delta)
    {
        if (alpha == 0) return (0, outp - 1);
        static long CeilDiv(long n, long d) => n >= 0 ? (n + d - 1) / d : -(-n / d);
        long lo, hi;
        if (bilinear)
        {
            lo = i <= 1 ? 0 : CeilDiv((long)(i - 1) * delta - beta, alpha);
            hi = CeilDiv((long)(i + 1) * delta - beta, alpha) - 1;
        }
        else
        {
            lo = CeilDiv((long)i * delta - beta, alpha);
            hi = CeilDiv((long)(i + 1) * delta - beta, alpha) - 1;
        }

        if (i == inp - 1) hi = outp - 1;
        return ((int)Math.Max(lo, 0), (int)Math.Min(hi, outp - 1));
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

                    y[(c * OH + oh) * OW + ow] = name == "maxpool" ? (n == 0 ? 0 : best) : (n == 0 ? 0 : sum / PoolDivisor(p, oh, ow, n));
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
                        y[(c * H + ih) * W + iw] += dy / PoolDivisor(p, oh, ow, n);
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
                // output index v in row-major order over the dimensions; source = base + Σ i_d · s_d.
                int rank = (p.Length - 1) / 2;
                for (int v = 0; v < count; v++)
                {
                    long src = p[2 * rank];
                    for (int d = rank - 1, t = v; d >= 0; d--) { src += (long)(t % p[d]) * p[rank + d]; t /= p[d]; }
                    y[v] = x[src];
                }

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
            case "interpolate_nearest_bwd":
            case "interpolate_bilinear_bwd":
            {
                // The source positions exactly, in integers (the coordinate rational: InterpolateAxis); the backward spreads each
                // dy over the positions its output read, by their weights.
                int C = p[0], H = p[1], W = p[2], OH = p[3], OW = p[4];
                bool bilinear = name.StartsWith("interpolate_bilinear", StringComparison.Ordinal), backward = name.EndsWith("_bwd", StringComparison.Ordinal);
                var q = KernelP(name, p);
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    int o = (c * OH + oh) * OW + ow;
                    if (!bilinear)
                    {
                        int ih = NearestAt(oh, H, q[5], q[6], q[7]), iw = NearestAt(ow, W, q[8], q[9], q[10]);
                        if (backward) y[(c * H + ih) * W + iw] += x[o];
                        else y[o] = x[(c * H + ih) * W + iw];
                        continue;
                    }

                    var (y0, y1, ry) = BilinearAt(oh, H, q[5], q[6], q[7]);
                    var (x0, x1, rx) = BilinearAt(ow, W, q[8], q[9], q[10]);
                    double fy = (double)ry / q[7], fx = (double)rx / q[10];
                    int a00 = (c * H + y0) * W + x0, a01 = (c * H + y0) * W + x1, a10 = (c * H + y1) * W + x0, a11 = (c * H + y1) * W + x1;
                    if (backward)
                    {
                        y[a00] += x[o] * (1 - fy) * (1 - fx);
                        y[a01] += x[o] * (1 - fy) * fx;
                        y[a10] += x[o] * fy * (1 - fx);
                        y[a11] += x[o] * fy * fx;
                        continue;
                    }

                    double top = x[a00] * (1 - fx) + x[a01] * fx, bottom = x[a10] * (1 - fx) + x[a11] * fx;
                    y[o] = top * (1 - fy) + bottom * fy;
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
