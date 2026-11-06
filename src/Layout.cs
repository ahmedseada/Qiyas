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
        "maxpool", "avgpool", "maxpool_bwd", "avgpool_bwd", "adaptive_maxpool", "adaptive_avgpool", "depthwise_conv"];

    /// <summary>Whether the operation applies to the type: bilinear interpolation and the average pools compute in a float lane.</summary>
    public static bool Applies(string name, bool floatType) => floatType || name is not ("interpolate_bilinear" or "avgpool" or "avgpool_bwd" or "adaptive_avgpool");

    /// <summary>Whether the operation sums values (its lane follows the sum rule).</summary>
    public static bool Sums(string name) => name is "scatter_add_rows" or "col2im" or "maxpool_bwd" or "depthwise_conv";

    /// <summary>Whether the operation computes in the float lane (weighted sums of float values).</summary>
    public static bool FloatLane(string name) => name is "interpolate_bilinear" or "avgpool" or "avgpool_bwd" or "adaptive_avgpool";

    /// <summary>The number of terms a summed value has (the sum rule's bound).</summary>
    public static int Terms(string name, int[] p) => name switch
    {
        "scatter_add_rows" => p[0],
        "col2im" or "maxpool_bwd" or "depthwise_conv" => p[3] * p[4],
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
            var (lo, hi) = TN.Range(k.X, k.InputCount);
            var (lo2, hi2) = k.X2 is null || k.Name != "depthwise_conv" ? (1, 1) : TN.Range(k.X2, k.P[0] * k.P[3] * k.P[4]);
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

    private struct LayoutVisitor(LayoutArgs k) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)k.Count, new MoveBody<TN, TLane, TL>(k, rows, cols));
        }
    }

    private struct LayoutFloatVisitor(LayoutArgs k) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)k.Count, new MoveBody<TN, TLane, TL>(k, rows, cols));
        }
    }

    private struct LayoutSumVisitor(LayoutArgs k) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)k.Count, new MoveBody<TN, TLane, TL>(k, rows, cols));
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
            Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)k.Count, new MoveBody<TN, TLane, TL>(k, rows, cols));
        }
    }

    private static (int Rows, int Cols) OutputRows(LayoutArgs k) => LayoutArgs.OutputRows(k);

    /// <summary>Every output row computed value by value from the input's lane values (loaded one at a time by index:
    /// the operations are gathers), the row stored once; rows chunked on whole bytes for the packed types.</summary>
    private struct MoveBody<TN, TLane, TL>(LayoutArgs k, int rows, int cols) : Team.IBody
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
                RunRows(first, last, k, rows, cols, exclusiveBytes: true);
                return;
            }

            RunRows(first, last, k, rows, cols, exclusiveBytes: false);
        }

        private static void RunRows(int first, int last, LayoutArgs k, int rows, int cols, bool exclusiveBytes)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(cols);
            var one = pool.Rent(1);
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
                            for (int c = 0; c < cols; c++) o[c] = Value(k.X, b + (long)c * p[7], one);
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
                            for (int c = 0; c < cols; c++) o[c] = Value(k.X, (long)r * p[1] + idx![c], one);
                            break;
                        case "scatter_add_rows":
                        {
                            // Output row r: the sum of every source row whose index is r, in source order.
                            o.Clear();
                            for (int i = 0; i < p[0]; i++)
                            {
                                if (idx![i] != r) continue;
                                for (int c = 0; c < cols; c++) o[c] = TL.Add(o[c], Value(k.X, (long)i * cols + c, one));
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
                                    o[oh * OW + ow] = (uint)ih < (uint)H && (uint)iw < (uint)W ? Value(k.X, ((long)c * H + ih) * W + iw, one) : TL.Zero;
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
                                        acc = TL.Add(acc, Value(k.X, ((long)((c * kh + a) * kw + b) * (OH * OW)) + (th / sh) * OW + tw / sw, one));
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
                                o[ow] = Value(k.X, ((long)c * H + ih) * W + iw, one);
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
                                TLane top = TL.Fma(Value(k.X, ((long)c * H + y0) * W + x1, one), fx, TL.Mul(Value(k.X, ((long)c * H + y0) * W + x0, one), gx));
                                TLane bottom = TL.Fma(Value(k.X, ((long)c * H + y1) * W + x1, one), fx, TL.Mul(Value(k.X, ((long)c * H + y1) * W + x0, one), gx));
                                o[ow] = TL.Fma(bottom, fy, TL.Mul(top, gy));
                            }

                            break;
                        }

                        case "shift":
                            for (int c = 0; c < cols; c++) { int s = c - p[2]; o[c] = (uint)s < (uint)cols ? Value(k.X, (long)r * cols + s, one) : TL.Zero; }
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
                                        TLane v = Value(k.X, ((long)c * H + ih) * W + iw, one);
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
                                        TLane dy = Value(k.X, ((long)c * OH + oh) * OW + ow, one);
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
                                                TLane v = Value(k.X2!, ((long)c * H + hh) * W + ww, one);
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
                                    TLane v = Value(k.X, ((long)c * H + ih) * W + iw, one);
                                    acc = max ? (unset || Widths.Compare(v, acc) > 0 ? v : acc) : TL.Add(acc, v);
                                    unset = false;
                                }

                                o[ow] = max ? acc : TL.Mul(acc, TL.From(1.0 / ((h1 - h0) * (w1 - w0))));
                            }

                            break;
                        }

                        case "depthwise_conv":
                        {
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                            int oh = r % OH, c = r / OH;
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
                                        acc = TL.Fma(Value(k.X2!, (long)(c * kh + a) * kw + b, one), Value(k.X, ((long)c * H + ih) * W + iw, one), acc);
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
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TLane Value(byte[] x, long index, TLane[] one)
        {
            if (TN.Direct) return MemoryMarshal.Cast<byte, TLane>(x.AsSpan())[(int)index];
            TN.Load(x, (int)index, one.AsSpan(0, 1));
            return one[0];
        }
    }
}
