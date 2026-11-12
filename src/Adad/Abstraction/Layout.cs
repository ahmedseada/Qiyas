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
        "interpolate_nearest_bwd" or "interpolate_bilinear_bwd" => InterpolateReads(name, p),
        _ => 1,
    };

    // The most outputs an input position of an interpolation gives its gradient to: the most along each axis, multiplied.
    private static long InterpolateReads(string name, int[] p)
    {
        bool bilinear = name.StartsWith("interpolate_bilinear", StringComparison.Ordinal);
        static int Most(bool bilinear, int n, int outs, int alpha, int beta, int delta)
        {
            int most = 0;
            for (int i = 0; i < n; i++) { var (lo, hi) = InterpolateSources(bilinear, i, n, outs, alpha, beta, delta); most = Math.Max(most, hi - lo + 1); }
            return most;
        }

        return (long)Most(bilinear, p[1], p[3], p[5], p[6], p[7]) * Most(bilinear, p[2], p[4], p[8], p[9], p[10]);
    }

    /// <summary>The number of terms a summed value has (the sum rule's bound).</summary>
    public static int Terms(string name, int[] p) => name switch
    {
        "scatter_add_rows" => p[0],
        "col2im" or "maxpool_bwd" or "depthwise_conv" => p[3] * p[4],
        "adaptive_maxpool_bwd" => (p[1] / p[3] + 2) * (p[2] / p[4] + 2),
        _ => 1,
    };

    // ---- the shapes behind P, in each operation's order
    // strided: d0 … d(r−1) | s0 … s(r−1) | base             output d0·…·d(r−1), any rank r (P has 2r + 1 integers)
    // An index selects within its range [0, size) or nothing: one_hot's row has no 1, a gather's value is 0, scatter-add's
    // source adds nowhere (the gather's gradient: what it read nothing from takes nothing back).
    // one_hot: n classes                                   output n·classes: row i a 1 at index[i]
    // gather_rows: rows cols n                             output n·cols: row i x's row index[i]
    // gather_cols: rows cols m                             output rows·m: column j x's column index[j]
    // scatter_add_rows: n cols rows                        output rows·cols: row r the sum of the source rows i with index[i] = r
    // im2col / col2im: C H W kh kw sh sw top left bottom right dh dw    cols (C·kh·kw) × (OH·OW): window (oh, ow) tap (a, b)
    //              reads x[c][oh·sh − top + a·dh][ow·sw − left + b·dw], 0 off the image; OH, OW the windows that fit
    //              (WindowCount: the bottom and right pads only add windows)
    // interpolate: C H W OH OW αh βh δh αw βw δw           output C·OH·OW; along each axis one affine map: output o's source
    //              is the input coordinate s = (o·α + β) / δ (α ≥ 0, δ ≥ 1; input position i at coordinate i), held to the
    //              input; nearest takes the position nearest s, bilinear the two around it (CentreMap: the usual map)
    // interpolate_*_bwd: the forward's parameters            dx (C·H·W) from dy (C·OH·OW)
    // pools: C H W kh kw sh sw top left bottom right      output C·OH·OW: window (oh, ow) the input positions of rows
    //              oh·sh − top … + kh − 1 and columns ow·sw − left … + kw − 1; max its first-ranked value, avg the mean of
    //              its values (a pad is no value: each pad shorter than the window, so every window holds one)
    // depthwise_conv: C H W kh kw sh sw top left bottom right dh dw   y[c][oh][ow] = Σ w[c][a][b]·x[c][oh·sh − top + a·dh]
    //              [ow·sw − left + b·dw] (the weights C × kh × kw the second input; x 0 off the image)
    // adaptive pools: C H W OH OW                          window (oh, ow) rows [⌊oh·H/OH⌋, ⌈(oh + 1)·H/OH⌉), columns likewise
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
            "strided" => StridedRows(p),
            "one_hot" => (p[0], p[1]),
            "gather_rows" => (p[2], p[1]),
            "gather_cols" => (p[0], p[2]),
            "scatter_add_rows" => (p[2], p[1]),
            "im2col" => (p[0] * p[3] * p[4], k.Count / (p[0] * p[3] * p[4])),
            "col2im" => (p[0] * p[1], p[2]),
            "interpolate_nearest" or "interpolate_bilinear" => (p[0] * p[3], p[4]),
            "shift" => (p[0], p[1]),
            "pad2d" => (p[0], p[2]),
            "maxpool" or "avgpool" => (p[0] * PoolOut(p).OH, PoolOut(p).OW),
            "depthwise_conv" => (p[0] * ConvOut(p).OH, ConvOut(p).OW),
            "maxpool_bwd" or "avgpool_bwd" => (p[0] * p[1], p[2]),
            "adaptive_maxpool" or "adaptive_avgpool" => (p[0] * p[3], p[4]),
            "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => (p[0] * p[1], p[2]),
            "interpolate_nearest_bwd" or "interpolate_bilinear_bwd" => (p[0] * p[1], p[2]),
            _ => throw new ArgumentException(k.Name),
        };
    }

    /// <summary>A pool's output size from C H W kh kw sh sw top left bottom right: the windows that fit an axis
    /// (WindowCount).</summary>
    public static (int OH, int OW) PoolOut(int[] p) => (WindowCount(p[1], p[3], p[5], p[7], p[9], 1), WindowCount(p[2], p[4], p[6], p[8], p[10], 1));

    /// <summary>A convolution window's (im2col's, col2im's, depthwise_conv's) output size from C H W kh kw sh sw top left
    /// bottom right dh dw: the windows that fit an axis (WindowCount).</summary>
    public static (int OH, int OW) ConvOut(int[] p) => (WindowCount(p[1], p[3], p[5], p[7], p[9], p[11]), WindowCount(p[2], p[4], p[6], p[8], p[10], p[12]));

    /// <summary>The windows of k taps d apart at stride s that fit an axis of n values with `before` and `after` pads:
    /// ⌊(before + n + after − d·(k − 1) − 1) / s⌋ + 1, or none where one window is longer than the padded axis.</summary>
    public static int WindowCount(int n, int k, int s, int before, int after, int d)
    {
        long span = (long)before + n + after - (long)d * (k - 1) - 1;
        return span < 0 ? 0 : (int)(span / s + 1);
    }

    /// <summary>The input positions [from, to) window o covers along an axis of n (k taps at stride s from o·s − before),
    /// held to the input.</summary>
    public static (int From, int To) Window(int o, int n, int k, int s, int before) =>
        (Math.Max(o * s - before, 0), Math.Min(o * s - before + k, n));

    // ---- the parameters as the kernels take them

    /// <summary>A copy with the parameters the kernels read (KernelP), the shape validated first.</summary>
    public LayoutArgs ForKernels()
    {
        Validate();
        return new() { Name = Name, X = X, X2 = X2, Index = Index, P = KernelP(Name, P), Output = Output, Count = Count, InputCount = InputCount };
    }

    /// <summary>Throws ArgumentException on a shape the operation does not take.</summary>
    public void Validate()
    {
        if (Name == "strided")
        {
            // 2r + 1 integers; every size ≥ 0, the output their product; a view with values reads inside the input.
            if (P.Length % 2 == 0) throw new ArgumentException($"strided: {P.Length} parameters (a view of rank r takes 2r + 1)");
            int rank = P.Length / 2;
            long count = 1, low = P[2 * rank], high = P[2 * rank];
            for (int i = 0; i < rank; i++)
            {
                if (P[i] < 0) throw new ArgumentException($"strided: dimension {i} of size {P[i]}");
                count *= P[i];
                long reach = (long)Math.Max(P[i] - 1, 0) * P[rank + i];
                if (reach < 0) low += reach; else high += reach;
            }

            if (count != Count) throw new ArgumentException($"strided: a view of {count} values into an output of {Count}");
            if (count > 0 && (low < 0 || high >= InputCount)) throw new ArgumentException($"strided: the view reads positions {low} to {high} of an input of {InputCount}");
        }
        else if (Name.StartsWith("interpolate_", StringComparison.Ordinal))
        {
            // C H W OH OW and a map an axis: α ≥ 0 (the map runs forward: the backward's gather), δ ≥ 1; an input of positions
            // where there are outputs; bilinear's δ at most 2^24 (its weight's numerator and δ exact in a float32 lane, so the
            // weight rounds once).
            if (P.Length != 11) throw new ArgumentException($"{Name}: {P.Length} parameters (C H W OH OW αh βh δh αw βw δw)");
            if (P[0] < 0 || P[1] < 0 || P[2] < 0 || P[3] < 0 || P[4] < 0) throw new ArgumentException($"{Name}: a negative size");
            if ((long)P[0] * P[3] * P[4] > 0 && (P[1] == 0 || P[2] == 0)) throw new ArgumentException($"{Name}: outputs from an input of no positions");
            if (P[5] < 0 || P[8] < 0 || P[7] < 1 || P[10] < 1) throw new ArgumentException($"{Name}: the maps ({P[5]}, {P[6]}, {P[7]}) and ({P[8]}, {P[9]}, {P[10]}) (α ≥ 0, δ ≥ 1)");
            if (Name.StartsWith("interpolate_bilinear", StringComparison.Ordinal) && (P[7] > 1 << 24 || P[10] > 1 << 24)) throw new ArgumentException($"{Name}: δ past 2^24");
            long forward = (long)P[0] * P[3] * P[4], image = (long)P[0] * P[1] * P[2];
            if (Count != (Name.EndsWith("_bwd", StringComparison.Ordinal) ? image : forward)) throw new ArgumentException($"{Name}: an output of {Count} values");
        }
        else if (Name is "im2col" or "col2im" or "depthwise_conv" or "maxpool" or "avgpool" or "maxpool_bwd" or "avgpool_bwd")
        {
            // C H W kh kw sh sw top left bottom right (dh dw for the convolution windows): sizes ≥ 0, windows, strides and
            // dilations ≥ 1, pads ≥ 0; a pool's pad shorter than its window (a window of pads alone holds no value) over an
            // image of positions; the output's count.
            bool pool = Name.Contains("pool", StringComparison.Ordinal);
            if (P.Length != (pool ? 11 : 13)) throw new ArgumentException($"{Name}: {P.Length} parameters (C H W kh kw sh sw top left bottom right{(pool ? "" : " dh dw")})");
            if (P[0] < 0 || P[1] < 0 || P[2] < 0) throw new ArgumentException($"{Name}: a negative size");
            if (P[3] < 1 || P[4] < 1 || P[5] < 1 || P[6] < 1 || (!pool && (P[11] < 1 || P[12] < 1))) throw new ArgumentException($"{Name}: a window, a stride or a dilation below 1");
            if (P[7] < 0 || P[8] < 0 || P[9] < 0 || P[10] < 0) throw new ArgumentException($"{Name}: a negative pad");
            if (pool && (P[7] >= P[3] || P[9] >= P[3] || P[8] >= P[4] || P[10] >= P[4])) throw new ArgumentException($"{Name}: pads ({P[7]}, {P[8]}, {P[9]}, {P[10]}) as long as a {P[3]} × {P[4]} window: a window of pads alone has no value");
            if (pool && P[0] > 0 && (P[1] == 0 || P[2] == 0)) throw new ArgumentException($"{Name}: a pool over an image of no positions");
            var (oh, ow) = pool ? PoolOut(P) : ConvOut(P);
            long image = (long)P[0] * P[1] * P[2], windows = (long)P[0] * oh * ow;
            long count = Name switch { "im2col" => windows * P[3] * P[4], "col2im" or "maxpool_bwd" or "avgpool_bwd" => image, _ => windows };
            if (Count != count) throw new ArgumentException($"{Name}: an output of {Count} values ({count} by its shape)");
        }
        else if (Name.StartsWith("adaptive_", StringComparison.Ordinal))
        {
            // C H W OH OW: an image of positions where there are windows.
            if (P.Length != 5) throw new ArgumentException($"{Name}: {P.Length} parameters (C H W OH OW)");
            if (P[0] < 0 || P[1] < 0 || P[2] < 0 || P[3] < 0 || P[4] < 0) throw new ArgumentException($"{Name}: a negative size");
            if ((long)P[0] * P[3] * P[4] > 0 && (P[1] == 0 || P[2] == 0)) throw new ArgumentException($"{Name}: windows over an image of no positions");
            long count = Name.EndsWith("_bwd", StringComparison.Ordinal) ? (long)P[0] * P[1] * P[2] : (long)P[0] * P[3] * P[4];
            if (Count != count) throw new ArgumentException($"{Name}: an output of {Count} values ({count} by its shape)");
        }
        else if (Name is "one_hot" or "gather_rows" or "gather_cols" or "scatter_add_rows" or "shift" or "pad2d")
        {
            // Sizes ≥ 0 (shift's k any); a gather's values from an input with some (an index of no row reads none, and the
            // kernels' safe address is the input's first value); the indices the CPU reads, one a selection; the output's count.
            int length = Name == "one_hot" ? 2 : 3;
            if (P.Length != length) throw new ArgumentException($"{Name}: {P.Length} parameters ({length} by its shape)");
            if (P[0] < 0 || P[1] < 0 || (length == 3 && Name != "shift" && P[2] < 0)) throw new ArgumentException($"{Name}: a negative size");
            if (Name == "gather_rows" && (long)P[2] * P[1] > 0 && P[0] == 0) throw new ArgumentException("gather_rows: rows gathered from an input of none");
            if (Name == "gather_cols" && (long)P[0] * P[2] > 0 && P[1] == 0) throw new ArgumentException("gather_cols: columns gathered from rows of none");
            int indices = Name switch { "one_hot" or "scatter_add_rows" => P[0], "gather_rows" or "gather_cols" => P[2], _ => 0 };
            if (Index is not null && Index.Length < indices) throw new ArgumentException($"{Name}: {Index.Length} indices ({indices} by its shape)");
            long count = Name switch
            {
                "one_hot" => (long)P[0] * P[1],
                "gather_rows" => (long)P[2] * P[1],
                "gather_cols" => (long)P[0] * P[2],
                "scatter_add_rows" => (long)P[2] * P[1],
                "shift" => (long)P[0] * P[1],
                _ => (long)P[0] * P[2],
            };
            if (Count != count) throw new ArgumentException($"{Name}: an output of {Count} values ({count} by its shape)");
        }
    }

    /// <summary>The integers a device's layout kernel takes (its parameters past the buffers and counts).</summary>
    public const int KernelParams = 16;

    /// <summary>The largest rank of a canonical strided view (StridedKernel) the device kernels take: its 2r + 1 integers
    /// within KernelParams.</summary>
    public const int KernelRank = (KernelParams - 1) / 2;

    /// <summary>The device kernel of the operation as the kernels take it (KernelP): a strided view's by its rank (strided1 …
    /// strided7), the others their own; NotSupportedException where its integers pass the kernels' KernelParams.</summary>
    public static string KernelName(string name, int[] kernelP)
    {
        if (kernelP.Length > KernelParams)
            throw new NotSupportedException(name == "strided"
                ? $"strided: a view of {kernelP.Length / 2} dimensions that do not merge (the device kernels take {KernelRank})"
                : $"{name}: {kernelP.Length} parameters (the device kernels take {KernelParams})");
        if (name.StartsWith("interpolate_", StringComparison.Ordinal))
        {
            // The kernels' 32-bit signed arithmetic on a map: the nearest form (2α, 2β + δ, 2δ) at every output, and the
            // backward's bounds up to (n + 1)·2δ.
            static bool Fits(int n, int outs, int alpha, int beta, int delta) =>
                2L * outs * alpha + Math.Abs(2L * beta + delta) + 2L * (n + 1) * delta <= int.MaxValue;
            if (!Fits(kernelP[1], kernelP[3], kernelP[5], kernelP[6], kernelP[7]) || !Fits(kernelP[2], kernelP[4], kernelP[8], kernelP[9], kernelP[10]))
                throw new NotSupportedException($"{name}: a map whose numerators pass 2^31 (the device kernels' integers)");
        }

        return name == "strided" ? $"strided{kernelP.Length / 2}" : name;
    }

    // A strided view's output rows: the outer dimensions' product by the last's size (one row of one value at rank 0).
    private static (int Rows, int Cols) StridedRows(int[] p)
    {
        int rank = p.Length / 2;
        if (rank == 0) return (1, 1);
        long rows = 1;
        for (int i = 0; i < rank - 1; i++) rows *= p[i];
        return ((int)rows, p[rank - 1]);
    }

    /// <summary>The parameters the device kernels read: a strided view canonical (StridedKernel), a window's with its output's
    /// size; the others as given.</summary>
    public static int[] KernelP(string name, int[] p) => name switch
    {
        "strided" => StridedKernel(p),
        // A window's parameters and the output's size (WindowCount, once): OH OW at 13 and 14, a pool's at 11 and 12.
        "im2col" or "col2im" or "depthwise_conv" => [.. p, ConvOut(p).OH, ConvOut(p).OW],
        "maxpool" or "avgpool" or "maxpool_bwd" or "avgpool_bwd" => [.. p, PoolOut(p).OH, PoolOut(p).OW],
        _ => p,
    };

    /// <summary>A strided view of any rank canonical: its dimensions of size 1 dropped and each that continues the one inside
    /// it (stride = the inner's stride × its size) merged into it, the rest kept in order: rank 1 or more (a single value
    /// is [1 | 0 | base]); a view with no values is [0, 1 | 0, 0 | base] (no rows).</summary>
    public static int[] StridedKernel(int[] p)
    {
        int rank = p.Length / 2;
        for (int i = 0; i < rank; i++) if (p[i] == 0) return [0, 1, 0, 0, p[2 * rank]];
        var d = new List<int>();
        var st = new List<int>();
        for (int i = 0; i < rank; i++)
        {
            if (p[i] == 1) continue;
            if (d.Count > 0 && st[^1] == (long)p[rank + i] * p[i]) { d[^1] *= p[i]; st[^1] = p[rank + i]; continue; }
            d.Add(p[i]);
            st.Add(p[rank + i]);
        }

        if (d.Count == 0) { d.Add(1); st.Add(0); }
        int r = d.Count;
        var q = new int[2 * r + 1];
        for (int i = 0; i < r; i++) { q[i] = d[i]; q[r + i] = st[i]; }
        q[2 * r] = p[2 * rank];
        return q;
    }

    /// <summary>The map placing output centres at the scale num/den (input positions an output step: in/out for a resize to
    /// out): s = (o + ½)·num/den − ½, as (α, β, δ) = (2num, num − den, 2den) over their common divisor.</summary>
    public static (int Alpha, int Beta, int Delta) CentreMap(int num, int den)
    {
        if (num <= 0 || den <= 0) throw new ArgumentException($"interpolation: a scale of {num}/{den}");
        long a = 2L * num, b = (long)num - den, d = 2L * den, g = Gcd(Gcd(a, Math.Abs(b)), d);
        return ((int)(a / g), (int)(b / g), (int)(d / g));
    }

    private static long Gcd(long a, long b) { while (b != 0) (a, b) = (b, a % b); return a; }

    /// <summary>Bilinear output o's two source positions along an axis of n under the map (α, β, δ) and the second's weight's
    /// numerator over δ: s = (o·α + β) / δ held to [0, n − 1], its positions ⌊s⌋ and the next, the weight s − ⌊s⌋ (at the
    /// input's last position both the last, the weight 0).</summary>
    public static (int I0, int I1, int Rem) BilinearAt(int o, int n, int alpha, int beta, int delta)
    {
        long t = Math.Max((long)o * alpha + beta, 0);
        long i0 = t / delta;
        if (i0 >= n - 1) return (n - 1, n - 1, 0);
        return ((int)i0, (int)i0 + 1, (int)(t - i0 * delta));
    }

    /// <summary>Nearest output o's source position along an axis of n under the map (α, β, δ): the input position nearest
    /// s = (o·α + β) / δ — position i takes the coordinates [i − ½, i + ½), so a tie goes to the higher — held to [0, n − 1]:
    /// ⌊(2(o·α + β) + δ) / 2δ⌋.</summary>
    public static int NearestAt(int o, int n, int alpha, int beta, int delta) =>
        (int)Math.Min(Math.Max(2 * ((long)o * alpha + beta) + delta, 0) / (2L * delta), n - 1);

    /// <summary>The outputs [lo, hi] along an axis whose source positions include input position i (a backward's gather):
    /// bilinear's first position i − 1 or i, nearest's i (the floor of its map (2α, 2β + δ, 2δ)); the first position also
    /// takes every output before the input, the last every output past it.</summary>
    public static (int Lo, int Hi) InterpolateSources(bool bilinear, int i, int n, int outs, int alpha, int beta, int delta)
    {
        if (alpha == 0) return (0, outs - 1);
        static long CeilDiv(long a, long d) => a >= 0 ? (a + d - 1) / d : -(-a / d);
        long al = alpha, be = beta, de = delta;
        if (!bilinear) { al = 2L * alpha; be = 2L * beta + delta; de = 2L * delta; }
        long first = bilinear ? i - 1 : i;
        long lo = first <= 0 ? 0 : CeilDiv(first * de - be, al);
        long hi = i == n - 1 ? outs - 1 : CeilDiv((i + 1) * de - be, al) - 1;
        return ((int)Math.Max(lo, 0), (int)Math.Min(hi, outs - 1));
    }

    /// <summary>An adaptive window along one axis: [⌊i·n/o⌋, ⌈(i + 1)·n/o⌉).</summary>
    public static (int From, int To) Adaptive(int i, int n, int o) => (i * n / o, ((i + 1) * n + o - 1) / o);


    /// <summary>The float64 reference of the operation on the input's values (x2: the second input's).</summary>
    public static double[] Reference(string name, double[] x, int[]? index, int[] p, int count, double[]? x2 = null)
    {
        var y = new double[count];
        switch (name)
        {
            case "maxpool":
            case "avgpool":
            {
                // Every window holds an input value (Validate): max its first-ranked, avg the mean of its values.
                var (OH, OW) = PoolOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double best = double.NegativeInfinity, sum = 0;
                    int n = 0;
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - top + a, iw = ow * sw - left + b;
                        if ((uint)ih >= (uint)H || (uint)iw >= (uint)W) continue;
                        double v = x[(c * H + ih) * W + iw];
                        if (n == 0 || Routing.Order(v, best) > 0) best = v;
                        sum += v; n++;
                    }

                    y[(c * OH + oh) * OW + ow] = name == "maxpool" ? best : sum / n;
                }

                return y;
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // x = dy (C × OH × OW), x2 = the forward input: maxpool's gradient to the window's first-ranked value, avgpool's
                // spread over the window's values, dy / n each.
                var (OH, OW) = PoolOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double dy = x[(c * OH + oh) * OW + ow], best = double.NegativeInfinity;
                    int n = 0, bestAt = -1;
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - top + a, iw = ow * sw - left + b;
                        if ((uint)ih >= (uint)H || (uint)iw >= (uint)W) continue;
                        n++;
                        if (name == "maxpool_bwd") { double v = x2![(c * H + ih) * W + iw]; if (bestAt < 0 || Routing.Order(v, best) > 0) { best = v; bestAt = (c * H + ih) * W + iw; } }
                    }

                    if (name == "maxpool_bwd") { y[bestAt] += dy; continue; }
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - top + a, iw = ow * sw - left + b;
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
                    for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++) { double v = x[(c * H + ih) * W + iw]; if ((ih == h0 && iw == w0) || Routing.Order(v, best) > 0) best = v; sum += v; }
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
                    for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++) { double v = x2![(c * H + ih) * W + iw]; if (bestAt < 0 || Routing.Order(v, best) > 0) { best = v; bestAt = (c * H + ih) * W + iw; } }
                    y[bestAt] += dy;
                }

                return y;
            }

            case "depthwise_conv":
            {
                var (OH, OW) = ConvOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8], dh = p[11], dw = p[12];
                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    double acc = 0;
                    for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    {
                        int ih = oh * sh - top + a * dh, iw = ow * sw - left + b * dw;
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
                for (int i = 0; i < p[2]; i++) if ((uint)index![i] < (uint)p[0]) Array.Copy(x, (long)index[i] * p[1], y, (long)i * p[1], p[1]);
                return y;
            case "gather_cols":
                for (int r = 0; r < p[0]; r++) for (int j = 0; j < p[2]; j++) if ((uint)index![j] < (uint)p[1]) y[r * p[2] + j] = x[r * p[1] + index[j]];
                return y;
            case "scatter_add_rows":
                for (int i = 0; i < p[0]; i++) if ((uint)index![i] < (uint)p[2]) for (int c = 0; c < p[1]; c++) y[index[i] * p[1] + c] += x[i * p[1] + c];
                return y;
            case "im2col":
            {
                var (OH, OW) = ConvOut(p);
                int C = p[0], H = p[1], W = p[2], kh = p[3], kw = p[4];
                for (int c = 0; c < C; c++) for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++)
                    for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                    {
                        int ih = oh * p[5] - p[7] + a * p[11], iw = ow * p[6] - p[8] + b * p[12];
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
                        int ih = oh * p[5] - p[7] + a * p[11], iw = ow * p[6] - p[8] + b * p[12];
                        if ((uint)ih < (uint)H && (uint)iw < (uint)W) y[(c * H + ih) * W + iw] += x[((c * kh + a) * kw + b) * (OH * OW) + oh * OW + ow];
                    }
                return y;
            }

            case "interpolate_nearest":
            case "interpolate_bilinear":
            case "interpolate_nearest_bwd":
            case "interpolate_bilinear_bwd":
            {
                // Along each axis output o's source is the rational s = (o·α + β) / δ, held to [0, n − 1]: nearest reads the
                // position nearest s (a tie to the higher), bilinear the positions ⌊s⌋ and ⌊s⌋ + 1 weighted 1 − f and f, f = s − ⌊s⌋.
                // The backward spreads each dy over the positions its output read, by their weights.
                int C = p[0], H = p[1], W = p[2], OH = p[3], OW = p[4];
                bool bilinear = name.StartsWith("interpolate_bilinear", StringComparison.Ordinal), backward = name.EndsWith("_bwd", StringComparison.Ordinal);
                static long FloorDiv(long a, long d) => a >= 0 ? a / d : -((-a + d - 1) / d);
                static int Nearest(int o, int n, int alpha, int beta, int delta) =>
                    (int)Math.Clamp(FloorDiv(2 * ((long)o * alpha + beta) + delta, 2L * delta), 0, n - 1);
                static (int I0, int I1, double F) Linear(int o, int n, int alpha, int beta, int delta)
                {
                    long t = (long)o * alpha + beta;
                    if (t <= 0) return (0, 0, 0);
                    if (t >= (long)(n - 1) * delta) return (n - 1, n - 1, 0);
                    long i0 = t / delta;
                    return ((int)i0, (int)i0 + 1, (double)(t - i0 * delta) / delta);
                }

                for (int c = 0; c < C; c++) for (int oh = 0; oh < OH; oh++) for (int ow = 0; ow < OW; ow++)
                {
                    int o = (c * OH + oh) * OW + ow;
                    if (!bilinear)
                    {
                        int ih = Nearest(oh, H, p[5], p[6], p[7]), iw = Nearest(ow, W, p[8], p[9], p[10]);
                        if (backward) y[(c * H + ih) * W + iw] += x[o];
                        else y[o] = x[(c * H + ih) * W + iw];
                        continue;
                    }

                    var (y0, y1, fy) = Linear(oh, H, p[5], p[6], p[7]);
                    var (x0, x1, fx) = Linear(ow, W, p[8], p[9], p[10]);
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
