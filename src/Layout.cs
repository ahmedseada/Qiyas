// Layout and indexing: data movement by index arithmetic, exact in every type (the value moves as stored; the lane is
// the type's own; where values are summed — scatter-add, col2im — the sum lane by the bounds, stored as the type).
//
//   strided    a contiguous copy of a strided view: output[i0, i1, i2, i3] = x[base + Σ i_d · stride_d] (permutes,
//              transposed views, slices, broadcasts with a zero stride)
//   one_hot    output[i][c] = (c == index[i]) ? 1 : 0 over `classes` columns
//   gather_rows   output[i] = x[index[i]] (a row each); gather_cols output[r][j] = x[r][index[j]]
//   scatter_add_rows   output[index[i]] += x[i] (the embedding gradient): every output row sums its sources in source
//              order (deterministic on both devices)
//   im2col     x (C × H × W) to columns ((C·kh·kw) × (OH·OW)) for a kernel kh × kw, strides, pads, dilations;
//   col2im     the reverse sum (each input pixel gathers its column entries in kernel order: exact order)
//   interpolate   x (C × H × W) resized to (C × OH × OW), nearest or bilinear (half-pixel centers)
//   shift      output[r][c] = x[r][c − k] inside the row, 0 outside (k may be negative)
//   pad2d      rows × cols into rows × newCols: the row's values then zeros (the operands of a product whose shape the
//              product kernels' vector loads cannot take as stored)
//   maxpool / avgpool   x (C × H × W) over kh × kw windows at strides and pads (pads excluded from a window's count)
//   maxpool_bwd / avgpool_bwd   dx from dy: each input pixel gathers the windows it lies in (the first largest wins a
//              maxpool tie, recomputed: deterministic)
//   adaptive_maxpool / adaptive_avgpool   x (C × H × W) to C × OH × OW with windows [⌊i·H/OH⌋, ⌈(i + 1)·H/OH⌉)
//   depthwise_conv   y[c][oh][ow] = Σ w[c][a][b] · x[c][oh·sh − ph + a·dh][ow·sw − pw + b·dw] (the weights as a second input)
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Qiyas;

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

internal static partial class Kernels
{
    /// <summary>A layout operation on the CPU; returns the lane's name (the type's own, or the sum lane where values are summed).</summary>
    public static string Layout<TN>(LayoutArgs k) where TN : struct, IQiyasNumber<TN>
    {
        if (LayoutArgs.Sums(k.Name))
        {
            // The bounds choose an integer type's sum lane; a float type's lane is its float lane whatever the values, so its
            // range is not read (a block format's range decodes every block, serially).
            var (lo, hi) = TN.IsFloat ? (0, 0) : TN.Range(k.X, k.InputCount);
            var (lo2, hi2) = TN.IsFloat || k.X2 is null || k.Name != "depthwise_conv" ? (1, 1) : TN.Range(k.X2, k.P[0] * k.P[3] * k.P[4]);
            var sum = new LayoutSumVisitor(k);
            TN.Sum(ref sum, new Bounds(lo, hi, lo2, hi2, LayoutArgs.Terms(k.Name, k.P)));
            return sum.Lane;
        }

        if (LayoutArgs.FloatLane(k.Name))
        {
            var fl = new LayoutFloatVisitor(k);
            TN.Elementwise(ref fl);
            return fl.Lane;
        }

        var v = new LayoutVisitor(k);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    // The team's measure of a layout job: its outputs by the reads each makes.
    private static long Work(LayoutArgs k) => (long)k.Count * LayoutArgs.ValueReads(k.Name, k.P);

    private struct LayoutVisitor(LayoutArgs k) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            var xl = Decoded<TN, TLane>(k);
            var at = WindowArgmax<TN, TLane>(k);
            var x2l = DecodedWeights<TN, TLane>(k);
            if (StridedTiled<TN, TLane>(k, cols, xl, out int dim))
            {
                int tile = Math.Max(16, 64 / System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>());
                var p = k.P;
                long groups = (long)p[0] * p[1] * p[2] / p[dim] * ((p[dim] + tile - 1) / tile);
                Team.For((int)groups, Math.Max(1, (int)(groups / (Threads * 4))), (long)k.Count, new StridedTileBody<TN, TLane>(k, cols, dim, tile, xl));
            }
            else Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, xl, at, x2l));
            if (xl is not null) ArrayPool<TLane>.Shared.Return(xl);
            if (x2l is not null) ArrayPool<TLane>.Shared.Return(x2l);
        }
    }

    private struct LayoutFloatVisitor(LayoutArgs k) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            var xl = Decoded<TN, TLane>(k);
            var at = WindowArgmax<TN, TLane>(k);
            var x2l = DecodedWeights<TN, TLane>(k);
            Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, xl, at, x2l));
            if (xl is not null) ArrayPool<TLane>.Shared.Return(xl);
            if (x2l is not null) ArrayPool<TLane>.Shared.Return(x2l);
        }
    }

    private struct LayoutSumVisitor(LayoutArgs k) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            var xl = Decoded<TN, TLane>(k);
            var at = WindowArgmax<TN, TLane>(k);
            var x2l = DecodedWeights<TN, TLane>(k);
            Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, xl, at, x2l));
            if (xl is not null) ArrayPool<TLane>.Shared.Return(xl);
            if (x2l is not null) ArrayPool<TLane>.Shared.Return(x2l);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IQiyasDotLane<TD, TIn, TAcc>
        {
            // Sums of single values need no dot product: the accumulator lane's plain arithmetic (s32/u32 for the 8- and
            // 16-bit types). The plain lane struct is chosen by the accumulator type at run time (the number implements
            // IQiyasNumber<TN, TAcc>, which the instantiation checks).
            var plain = typeof(TAcc) == typeof(int) ? typeof(I32) : typeof(U32);
            Lane = typeof(TAcc) == typeof(int) ? "s32" : "u32";
            var method = typeof(LayoutSumVisitor).GetMethod(nameof(Plain), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
            method.MakeGenericMethod(typeof(TN), typeof(TAcc), plain).Invoke(null, [k]);
        }

        private static void Plain<TN, TLane, TL>(LayoutArgs k) where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            var (rows, cols) = OutputRows(k);
            var xl = Decoded<TN, TLane>(k);
            var at = WindowArgmax<TN, TLane>(k);
            var x2l = DecodedWeights<TN, TLane>(k);
            Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, xl, at, x2l));
            if (xl is not null) ArrayPool<TLane>.Shared.Return(xl);
            if (x2l is not null) ArrayPool<TLane>.Shared.Return(x2l);
        }
    }

    private static (int Rows, int Cols) OutputRows(LayoutArgs k) => LayoutArgs.OutputRows(k);

    /// <summary>A strided copy whose output rows run across the input (the last dimension's stride not 1) while one of the
    /// three outer dimensions has stride 1 (a transposition): taken by tiles of output rows along that dimension (see
    /// StridedTileBody). Rows ending on whole bytes only (each row stored by one thread), values read in place or decoded.</summary>
    private static bool StridedTiled<TN, TLane>(LayoutArgs k, int cols, TLane[]? xl, out int dim) where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        dim = -1;
        if (k.Name != "strided" || k.P[7] == 1 || (long)cols * TN.Bits % 8 != 0 || (!TN.Direct && xl is null)) return false;
        for (int j = 2; j >= 0; j--) if (k.P[4 + j] == 1 && k.P[j] > 1) { dim = j; return true; }
        return false;
    }

    /// <summary>The tiled strided copy: a group is `tile` output rows that differ only in dimension `dim` (whose input
    /// stride is 1) at one index of the other two; for each output column the group's values lie side by side in the input
    /// (a cache line read once, not a line a value), gathered into the group's rows (a tile × cols block in lane storage),
    /// each row then stored whole. Groups are disjoint sets of whole rows.</summary>
    private struct StridedTileBody<TN, TLane>(LayoutArgs k, int cols, int dim, int tile, TLane[]? xl) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var p = k.P;
            int d0 = p[0], d1 = p[1], d2 = p[2], dj = p[dim], blocks = (dj + tile - 1) / tile;
            long s7 = p[7];
            ReadOnlySpan<TLane> src = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(k.X.AsSpan()) : xl.AsSpan();
            var buffer = ArrayPool<TLane>.Shared.Rent(tile * cols);
            try
            {
                for (int g = first; g < last; g++)
                {
                    // g = outer · blocks + block; outer enumerates the two other dimensions (in their order).
                    int block = g % blocks, outer = g / blocks;
                    int lo = block * tile, count = Math.Min(tile, dj - lo);
                    int i0, i1, i2;
                    if (dim == 0) { i2 = outer % d2; i1 = outer / d2; i0 = lo; }
                    else if (dim == 1) { i2 = outer % d2; i0 = outer / d2; i1 = lo; }
                    else { i1 = outer % d1; i0 = outer / d1; i2 = lo; }
                    long b0 = p[8] + (long)i0 * p[4] + (long)i1 * p[5] + (long)i2 * p[6];  // value lo of the group (stride 1 in t)
                    var t2 = buffer.AsSpan(0, tile * cols);
                    for (int c = 0; c < cols; c++)
                    {
                        var line = src.Slice((int)(b0 + c * s7), count);
                        for (int t = 0; t < count; t++) t2[t * cols + c] = line[t];
                    }

                    for (int t = 0; t < count; t++)
                    {
                        int r0 = dim == 0 ? i0 + t : i0, r1 = dim == 1 ? i1 + t : i1, r2 = dim == 2 ? i2 + t : i2;
                        long r = ((long)r0 * d1 + r1) * d2 + r2;
                        TN.Store(t2.Slice(t * cols, cols), k.Output, (int)(r * cols));
                    }
                }
            }
            finally
            {
                ArrayPool<TLane>.Shared.Return(buffer);
            }
        }
    }

    /// <summary>The max-pool backward's first largest of every window of the forward input (x2), found once by the team
    /// (index into the input, -1 for an empty window): each input row then reads its windows' choice instead of scanning
    /// every window it lies in. Null for the other operations.</summary>
    private static long[]? WindowArgmax<TN, TLane>(LayoutArgs k) where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        if (k.Name is not ("maxpool_bwd" or "adaptive_maxpool_bwd")) return null;
        var p = k.P;
        int C = p[0], H = p[1], W = p[2];
        var (OH, OW) = k.Name == "maxpool_bwd" ? LayoutArgs.PoolOut(p) : (p[3], p[4]);
        int windows = C * OH * OW, n = C * H * W;
        TLane[]? lanes = null;
        if (!TN.Direct)
        {
            lanes = ArrayPool<TLane>.Shared.Rent(n);
            int unit = Math.Max(256, TN.Granularity);
            Team.For(n, Math.Max(4096, n / (Threads * 2)) / unit * unit + unit, n, new LoadBody<TN, TLane>(k.X2!, lanes));
        }

        var at = new long[windows];
        Team.For(windows, Math.Max(1, windows / (Threads * 4)), (long)n, new ArgmaxBody<TN, TLane>(k, lanes, at, OH, OW));
        if (lanes is not null) ArrayPool<TLane>.Shared.Return(lanes);
        return at;
    }

    private readonly struct ArgmaxBody<TN, TLane>(LayoutArgs k, TLane[]? lanes, long[] at, int OH, int OW) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        public void Run(int first, int last)
        {
            var p = k.P;
            int H = p[1], W = p[2];
            ReadOnlySpan<TLane> x = lanes ?? MemoryMarshal.Cast<byte, TLane>(k.X2!.AsSpan());
            bool adaptive = k.Name == "adaptive_maxpool_bwd";
            for (int i = first; i < last; i++)
            {
                int ow = i % OW, oh = i / OW % OH, c = i / (OW * OH);
                int h0, h1, w0, w1;
                if (adaptive) { (h0, h1) = LayoutArgs.Adaptive(oh, H, OH); (w0, w1) = LayoutArgs.Adaptive(ow, W, OW); }
                else { h0 = oh * p[5] - p[7]; h1 = h0 + p[3]; w0 = ow * p[6] - p[8]; w1 = w0 + p[4]; }
                h0 = Math.Max(h0, 0); h1 = Math.Min(h1, H); w0 = Math.Max(w0, 0); w1 = Math.Min(w1, W);
                // The first largest in row-major order: a later value takes it only when strictly larger.
                long bestAt = -1;
                TLane best = default;
                for (int hh = h0; hh < h1; hh++) for (int ww = w0; ww < w1; ww++)
                {
                    long j = ((long)c * H + hh) * W + ww;
                    TLane v = x[(int)j];
                    if (bestAt < 0 || Widths.Compare(v, best) > 0) { best = v; bestAt = j; }
                }

                at[i] = bestAt;
            }
        }
    }

    /// <summary>The depthwise convolution's weights (x2) decoded to lane values once for the types not read in place (each
    /// weight is read for every output value); null otherwise.</summary>
    private static TLane[]? DecodedWeights<TN, TLane>(LayoutArgs k) where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        if (TN.Direct || k.Name != "depthwise_conv") return null;
        int n = k.P[0] * k.P[3] * k.P[4];
        var lanes = ArrayPool<TLane>.Shared.Rent(n);
        TN.Load(k.X2!, 0, lanes.AsSpan(0, n));
        return lanes;
    }

    /// <summary>The input of a type not read in place (narrow floats, packed and block types) decoded to lane values once, by
    /// the team (a value read one at a time would convert, or decode its unit or block, each time); null for the types read
    /// in place and the operations that read few values.</summary>
    private static TLane[]? Decoded<TN, TLane>(LayoutArgs k) where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        if (TN.Direct || k.Name is "one_hot" or "gather_rows" or "pad2d") return null;      // those read rows whole already
        int n = (int)k.InputCount;
        var lanes = ArrayPool<TLane>.Shared.Rent(n);
        int unit = Math.Max(256, TN.Granularity);
        Team.For(n, Math.Max(4096, n / (Threads * 2)) / unit * unit + unit, n, new LoadBody<TN, TLane>(k.X, lanes));
        return lanes;
    }

    /// <summary>Every output row computed value by value from the input's lane values (loaded one at a time by index:
    /// the operations are gathers), the row stored once; rows chunked on whole bytes for the packed types.</summary>
    private struct MoveBody<TN, TLane, TL>(LayoutArgs k, int rows, int cols, TLane[]? xl = null, long[]? at = null, TLane[]? x2l = null) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            // Chunks of rows that share bytes are aligned by the caller's chunking only when cols·bits ends on a byte; here
            // every row store is through a whole-row buffer and the packed types store with Packed.Store8 (any offset),
            // so two threads may touch one byte at a row boundary: take whole bytes by widening the chunk to byte ends.
            int per = TN.Granularity;
            long startBit = (long)first * cols * TN.Bits, endBit = (long)last * cols * TN.Bits;
            if (per > 1 && (startBit % 8 != 0 || endBit % 8 != 0))
            {
                // Compute into a lane buffer for these rows and store value-aligned pieces: the first and last partial bytes
                // are stored one value at a time by the same thread only when it owns the byte's first value.
                RunRows(first, last, k, rows, cols, exclusiveBytes: true, xl, at, x2l);
                return;
            }

            RunRows(first, last, k, rows, cols, exclusiveBytes: false, xl, at, x2l);
        }

        // xl: the input decoded to lane values once (packed and block types), else null and values are loaded one at a time.
        // at: the max-pool backward's first largest of every window (input index, -1 for an empty window), found once.
        private static void RunRows(int first, int last, LayoutArgs k, int rows, int cols, bool exclusiveBytes, TLane[]? xl, long[]? at, TLane[]? x2l)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(cols);
            var one = pool.Rent(1);
            TLane[]? source = null;
            try
            {
                var p = k.P;
                var idx = k.Index;
                for (int r = first; r < last; r++)
                {
                    var o = row.AsSpan(0, cols);
                    switch (k.Name)
                    {
                        case "strided":
                        {
                            int i2 = r % p[2], i1 = r / p[2] % p[1], i0 = r / (p[2] * p[1]);
                            long b = p[8] + (long)i0 * p[4] + (long)i1 * p[5] + (long)i2 * p[6];
                            for (int c = 0; c < cols; c++) o[c] = Value(xl, k.X, b + (long)c * p[7], one);
                            break;
                        }

                        case "one_hot":
                            o.Clear();
                            if ((uint)idx![r] < (uint)cols) o[idx[r]] = TL.From(1);
                            break;
                        case "gather_rows":
                            TN.Load(k.X, idx![r] * cols, o);
                            break;
                        case "gather_cols":
                            for (int c = 0; c < cols; c++) o[c] = Value(xl, k.X, (long)r * p[1] + idx![c], one);
                            break;
                        case "scatter_add_rows":
                        {
                            // Output row r: the sum of every source row whose index is r, in source order; a source row read
                            // whole (one load of its values: a block format decodes each block once, not once per value).
                            o.Clear();
                            for (int i = 0; i < p[0]; i++)
                            {
                                if (idx![i] != r) continue;
                                ReadOnlySpan<TLane> s;
                                if (TN.Direct) s = MemoryMarshal.Cast<byte, TLane>(k.X.AsSpan()).Slice(i * cols, cols);
                                else if (xl is not null) s = xl.AsSpan(i * cols, cols);
                                else
                                {
                                    source ??= pool.Rent(cols);
                                    TN.Load(k.X, i * cols, source.AsSpan(0, cols));
                                    s = source.AsSpan(0, cols);
                                }

                                for (int c = 0; c < cols; c++) o[c] = TL.Add(o[c], s[c]);
                            }

                            break;
                        }

                        case "im2col":
                        {
                            var (OH, OW) = LayoutArgs.ConvOut(p);
                            int kw = p[4], kh = p[3], H = p[1], W = p[2];
                            int b = r % kw, a = r / kw % kh, c = r / (kw * kh);
                            for (int oh = 0; oh < OH; oh++)
                            {
                                int ih = oh * p[5] - p[7] + a * p[9];
                                for (int ow = 0; ow < OW; ow++)
                                {
                                    int iw = ow * p[6] - p[8] + b * p[10];
                                    o[oh * OW + ow] = (uint)ih < (uint)H && (uint)iw < (uint)W ? Value(xl, k.X, ((long)c * H + ih) * W + iw, one) : TL.Zero;
                                }
                            }

                            break;
                        }

                        case "col2im":
                        {
                            // Input pixel (c, ih, iw = col): every (a, b, oh, ow) that maps onto it, in kernel order.
                            var (OH, OW) = LayoutArgs.ConvOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                            int ih = r % H, c = r / H;
                            for (int iw = 0; iw < W; iw++)
                            {
                                TLane acc = TL.Zero;
                                for (int a = 0; a < kh; a++)
                                {
                                    int th = ih + ph - a * dh;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int tw = iw + pw - b * dw;
                                        if (tw < 0 || tw % sw != 0 || tw / sw >= OW) continue;
                                        acc = TL.Add(acc, Value(xl, k.X, ((long)((c * kh + a) * kw + b) * (OH * OW)) + (th / sh) * OW + tw / sw, one));
                                    }
                                }

                                o[iw] = acc;
                            }

                            break;
                        }

                        case "interpolate_nearest":
                        {
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            int ih = Math.Min(H - 1, (2 * oh + 1) * H / (2 * OH));
                            for (int ow = 0; ow < OW; ow++)
                            {
                                int iw = Math.Min(W - 1, (2 * ow + 1) * W / (2 * OW));
                                o[ow] = Value(xl, k.X, ((long)c * H + ih) * W + iw, one);
                            }

                            break;
                        }

                        case "interpolate_bilinear":
                        {
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            double sy = Math.Max((oh + 0.5) * H / OH - 0.5, 0);
                            int y0 = Math.Min((int)Math.Floor(sy), H - 1), y1 = Math.Min(y0 + 1, H - 1);
                            TLane fy = TL.From(sy - y0), gy = TL.From(1 - (sy - y0));
                            for (int ow = 0; ow < OW; ow++)
                            {
                                double sx = Math.Max((ow + 0.5) * W / OW - 0.5, 0);
                                int x0 = Math.Min((int)Math.Floor(sx), W - 1), x1 = Math.Min(x0 + 1, W - 1);
                                TLane fx = TL.From(sx - x0), gx = TL.From(1 - (sx - x0));
                                TLane top = TL.Fma(Value(xl, k.X, ((long)c * H + y0) * W + x1, one), fx, TL.Mul(Value(xl, k.X, ((long)c * H + y0) * W + x0, one), gx));
                                TLane bottom = TL.Fma(Value(xl, k.X, ((long)c * H + y1) * W + x1, one), fx, TL.Mul(Value(xl, k.X, ((long)c * H + y1) * W + x0, one), gx));
                                o[ow] = TL.Fma(bottom, fy, TL.Mul(top, gy));
                            }

                            break;
                        }

                        case "shift":
                            for (int c = 0; c < cols; c++) { int s = c - p[2]; o[c] = (uint)s < (uint)cols ? Value(xl, k.X, (long)r * cols + s, one) : TL.Zero; }
                            break;
                        case "pad2d":
                            o.Clear();
                            TN.Load(k.X, r * p[1], o[..p[1]]);
                            break;
                        case "maxpool":
                        case "avgpool":
                        {
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                            int oh = r % OH, c = r / OH;
                            bool max = k.Name == "maxpool";
                            for (int ow = 0; ow < OW; ow++)
                            {
                                TLane acc = max ? TL.From(double.NegativeInfinity) : TL.Zero;
                                int n = 0;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - ph + a;
                                    if ((uint)ih >= (uint)H) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int iw = ow * sw - pw + b;
                                        if ((uint)iw >= (uint)W) continue;
                                        TLane v = Value(xl, k.X, ((long)c * H + ih) * W + iw, one);
                                        acc = max ? (n == 0 || Widths.Compare(v, acc) > 0 ? v : acc) : TL.Add(acc, v);
                                        n++;
                                    }
                                }

                                o[ow] = n == 0 ? TL.Zero : max ? acc : TL.Mul(acc, TL.From(1.0 / n));
                            }

                            break;
                        }

                        case "maxpool_bwd":
                        case "avgpool_bwd":
                        {
                            // Input pixel (c, ih, iw = col): the windows it lies in; maxpool: those whose first largest it is.
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                            int ih = r % H, c = r / H;
                            bool max = k.Name == "maxpool_bwd";
                            if (max && at is not null)
                            {
                                // The windows over row ih in the order a pixel met them (kernel row a ascending: oh descending;
                                // then ow descending), each adding dy where its first largest lies on this row.
                                o.Clear();
                                long rowAt = (long)c * H + ih;
                                for (int a = 0; a < kh; a++)
                                {
                                    int th = ih + ph - a;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    int oh = th / sh;
                                    for (int ow = OW - 1; ow >= 0; ow--)
                                    {
                                        long w = at[((long)c * OH + oh) * OW + ow];
                                        if (w >= 0 && w / W == rowAt) o[(int)(w % W)] = TL.Add(o[(int)(w % W)], Value(xl, k.X, ((long)c * OH + oh) * OW + ow, one));
                                    }
                                }

                                break;
                            }

                            for (int iw = 0; iw < W; iw++)
                            {
                                TLane acc = TL.Zero;
                                for (int a = 0; a < kh; a++)
                                {
                                    int th = ih + ph - a;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    int oh = th / sh;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int tw = iw + pw - b;
                                        if (tw < 0 || tw % sw != 0 || tw / sw >= OW) continue;
                                        int ow = tw / sw;
                                        TLane dy = Value(xl, k.X, ((long)c * OH + oh) * OW + ow, one);
                                        if (!max)
                                        {
                                            int n = 0;
                                            for (int aa = 0; aa < kh; aa++) { int hh = oh * sh - ph + aa; if ((uint)hh >= (uint)H) continue; for (int bb = 0; bb < kw; bb++) { int ww = ow * sw - pw + bb; if ((uint)ww < (uint)W) n++; } }
                                            acc = TL.Add(acc, TL.Mul(dy, TL.From(1.0 / n)));
                                            continue;
                                        }

                                        // The window's first largest: this pixel takes dy when it is that one.
                                        long bestAt = -1;
                                        TLane best = TL.From(double.NegativeInfinity);
                                        for (int aa = 0; aa < kh; aa++)
                                        {
                                            int hh = oh * sh - ph + aa;
                                            if ((uint)hh >= (uint)H) continue;
                                            for (int bb = 0; bb < kw; bb++)
                                            {
                                                int ww = ow * sw - pw + bb;
                                                if ((uint)ww >= (uint)W) continue;
                                                TLane v = Value(x2l, k.X2!, ((long)c * H + hh) * W + ww, one);
                                                if (bestAt < 0 || Widths.Compare(v, best) > 0) { best = v; bestAt = ((long)c * H + hh) * W + ww; }
                                            }
                                        }

                                        if (bestAt == ((long)c * H + ih) * W + iw) acc = TL.Add(acc, dy);
                                    }
                                }

                                o[iw] = acc;
                            }

                            break;
                        }

                        case "adaptive_maxpool":
                        case "adaptive_avgpool":
                        {
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            bool max = k.Name == "adaptive_maxpool";
                            var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                            for (int ow = 0; ow < OW; ow++)
                            {
                                var (w0, w1) = LayoutArgs.Adaptive(ow, W, OW);
                                TLane acc = TL.Zero;
                                bool unset = true;
                                for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++)
                                {
                                    TLane v = Value(xl, k.X, ((long)c * H + ih) * W + iw, one);
                                    acc = max ? (unset || Widths.Compare(v, acc) > 0 ? v : acc) : TL.Add(acc, v);
                                    unset = false;
                                }

                                o[ow] = max ? acc : TL.Mul(acc, TL.From(1.0 / ((h1 - h0) * (w1 - w0))));
                            }

                            break;
                        }

                        case "adaptive_maxpool_bwd":
                        case "adaptive_avgpool_bwd":
                        {
                            // Input pixel (c, ih, iw = col): the windows holding it (an adaptive window's rows are those oh with h0(oh) ≤ ih < h1(oh)).
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int ih = r % H, c = r / H;
                            bool max = k.Name == "adaptive_maxpool_bwd";
                            if (!max)
                            {
                                // Each window holding row ih adds its share to its columns, the windows in the order a pixel
                                // met them (oh, then ow ascending): each pixel's sum the same terms in the same order.
                                o.Clear();
                                for (int oh = 0; oh < OH; oh++)
                                {
                                    var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                                    if (ih < h0 || ih >= h1) continue;
                                    for (int ow = 0; ow < OW; ow++)
                                    {
                                        var (w0, w1) = LayoutArgs.Adaptive(ow, W, OW);
                                        TLane share = TL.Mul(Value(xl, k.X, ((long)c * OH + oh) * OW + ow, one), TL.From(1.0 / ((h1 - h0) * (w1 - w0))));
                                        for (int iw = w0; iw < w1; iw++) o[iw] = TL.Add(o[iw], share);
                                    }
                                }

                                break;
                            }

                            if (max && at is not null)
                            {
                                // The windows over row ih in the order a pixel met them (oh, then ow ascending).
                                o.Clear();
                                long rowAt = (long)c * H + ih;
                                for (int oh = 0; oh < OH; oh++)
                                {
                                    var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                                    if (ih < h0 || ih >= h1) continue;
                                    for (int ow = 0; ow < OW; ow++)
                                    {
                                        long w = at[((long)c * OH + oh) * OW + ow];
                                        if (w >= 0 && w / W == rowAt) o[(int)(w % W)] = TL.Add(o[(int)(w % W)], Value(xl, k.X, ((long)c * OH + oh) * OW + ow, one));
                                    }
                                }

                                break;
                            }

                            for (int iw = 0; iw < W; iw++)
                            {
                                TLane acc = TL.Zero;
                                for (int oh = 0; oh < OH; oh++)
                                {
                                    var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                                    if (ih < h0 || ih >= h1) continue;
                                    for (int ow = 0; ow < OW; ow++)
                                    {
                                        var (w0, w1) = LayoutArgs.Adaptive(ow, W, OW);
                                        if (iw < w0 || iw >= w1) continue;
                                        TLane dy = Value(xl, k.X, ((long)c * OH + oh) * OW + ow, one);
                                        if (!max) { acc = TL.Add(acc, TL.Mul(dy, TL.From(1.0 / ((h1 - h0) * (w1 - w0))))); continue; }
                                        long bestAt = -1;
                                        TLane best = TL.Zero;
                                        for (int hh = h0; hh < h1; hh++) for (int ww = w0; ww < w1; ww++)
                                        {
                                            TLane v = Value(x2l, k.X2!, ((long)c * H + hh) * W + ww, one);
                                            if (bestAt < 0 || Widths.Compare(v, best) > 0) { best = v; bestAt = ((long)c * H + hh) * W + ww; }
                                        }

                                        if (bestAt == ((long)c * H + ih) * W + iw) acc = TL.Add(acc, dy);
                                    }
                                }

                                o[iw] = acc;
                            }

                            break;
                        }

                        case "depthwise_conv":
                        {
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                            int oh = r % OH, c = r / OH;
                            if (sw == 1 && (xl is not null || TN.Direct))
                            {
                                // Stride 1 along the row: each tap (a, b) a vector pass over the outputs its column range reaches,
                                // the taps in the scalar loop's order (each output's sum the same terms in the same order, bit for
                                // bit: the lane's vector and scalar Fma are the same operation).
                                ReadOnlySpan<TLane> xs = xl is not null ? xl.AsSpan() : MemoryMarshal.Cast<byte, TLane>(k.X.AsSpan());
                                o.Clear();
                                int vl = Vector<TLane>.Count;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - ph + a * dh;
                                    if ((uint)ih >= (uint)H) continue;
                                    long rowAt = ((long)c * H + ih) * W;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int off = b * dw - pw, lo = Math.Max(0, -off), hi = Math.Min(OW, W - off);
                                        if (lo >= hi) continue;
                                        TLane wv = Value(x2l, k.X2!, (long)(c * kh + a) * kw + b, one);
                                        var wvec = new Vector<TLane>(wv);
                                        var src = xs.Slice((int)(rowAt + lo + off), hi - lo);
                                        var dst = o.Slice(lo, hi - lo);
                                        int i = 0;
                                        for (; i + vl <= dst.Length; i += vl)
                                            TL.Fma(wvec, new Vector<TLane>(src.Slice(i)), new Vector<TLane>(dst.Slice(i))).CopyTo(dst.Slice(i));
                                        for (; i < dst.Length; i++) dst[i] = TL.Fma(wv, src[i], dst[i]);
                                    }
                                }

                                break;
                            }

                            for (int ow = 0; ow < OW; ow++)
                            {
                                TLane acc = TL.Zero;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - ph + a * dh;
                                    if ((uint)ih >= (uint)H) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int iw = ow * sw - pw + b * dw;
                                        if ((uint)iw >= (uint)W) continue;
                                        acc = TL.Fma(Value(x2l, k.X2!, (long)(c * kh + a) * kw + b, one), Value(xl, k.X, ((long)c * H + ih) * W + iw, one), acc);
                                    }
                                }

                                o[ow] = acc;
                            }

                            break;
                        }
                    }

                    TN.Store(o, k.Output, r * cols);
                }
            }
            finally
            {
                pool.Return(row);
                pool.Return(one);
                if (source is not null) pool.Return(source);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TLane Value(TLane[]? lanes, byte[] x, long index, TLane[] one) => lanes is not null ? lanes[index] : Value(x, index, one);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TLane Value(byte[] x, long index, TLane[] one)
        {
            if (TN.Direct) return MemoryMarshal.Cast<byte, TLane>(x.AsSpan())[(int)index];
            TN.Load(x, (int)index, one.AsSpan(0, 1));
            return one[0];
        }
    }
}
