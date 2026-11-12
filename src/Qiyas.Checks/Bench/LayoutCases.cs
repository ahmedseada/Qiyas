// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The layout operations' inputs: small odd shapes for the gate, realistic shapes for the bench.
namespace Qiyas.Bench;

public static class LayoutCases
{
    private static int[] Indices(int count, int range, int seed)
    {
        var random = new Random(seed);
        var idx = new int[count];
        for (int i = 0; i < count; i++) idx[i] = random.Next(range);
        return idx;
    }

    private static LayoutArgs Make(string name, string type, int inputCount, int[]? index, int[] p, int count, int seed, int secondCount = 0) => new()
    {
        Name = name, X = Registry.Make(type, inputCount, seed, -1, 1), X2 = secondCount == 0 ? null : Registry.Make(type, secondCount, seed + 100, -1, 1),
        Index = index, P = p, Count = count, InputCount = inputCount, Output = new byte[Registry.Bytes(type, count)],
    };

    private static int PoolCount(int[] p) { var (oh, ow) = LayoutArgs.PoolOut(p); return p[0] * oh * ow; }

    /// <summary>A contiguous array of the shape read with its dimensions in the given order (a strided view: the order's
    /// sizes, each with its dimension's stride in the array).</summary>
    private static LayoutArgs Permuted(string type, int[] shape, int[] order, int seed)
    {
        int rank = shape.Length, n = 1;
        var stride = new int[rank];
        for (int i = rank - 1; i >= 0; i--) { stride[i] = n; n *= shape[i]; }
        var p = new int[2 * rank + 1];
        for (int i = 0; i < rank; i++) { p[i] = shape[order[i]]; p[rank + i] = stride[order[i]]; }
        return Make("strided", type, n, null, p, n, seed);
    }

    /// <summary>A view of eight dimensions that do not merge (a reversal of 2⁸ values): past the device kernels' rank
    /// (LayoutArgs.KernelRank), taken by the CPU and refused by a device.</summary>
    public static LayoutArgs PastKernelRank(string type) => Permuted(type, [2, 2, 2, 2, 2, 2, 2, 2], [7, 6, 5, 4, 3, 2, 1, 0], 58);

    public static string Shape(LayoutArgs k) => k.Name switch
    {
        "strided" => k.P.Length == 1 ? "one value" : string.Join("x", k.P.Take(k.P.Length / 2)),
        "one_hot" => $"{k.P[0]}x{k.P[1]}",
        "gather_rows" => $"{k.P[2]} of {k.P[0]}x{k.P[1]}",
        "gather_cols" => $"{k.P[2]} of {k.P[0]}x{k.P[1]}",
        "scatter_add_rows" => $"{k.P[0]} into {k.P[2]}x{k.P[1]}",
        "im2col" or "col2im" => $"{k.P[0]}x{k.P[1]}x{k.P[2]} k{k.P[3]}x{k.P[4]}{(Even(k.P) ? "" : $" p{Pads(k.P)}")}",
        "interpolate_nearest" or "interpolate_bilinear" or "interpolate_nearest_bwd" or "interpolate_bilinear_bwd" =>
            $"{k.P[0]}x{k.P[1]}x{k.P[2]} to {k.P[3]}x{k.P[4]}{(IsCentres(k.P) ? "" : $" maps ({k.P[5]}, {k.P[6]}, {k.P[7]}), ({k.P[8]}, {k.P[9]}, {k.P[10]})")}",
        "shift" => $"{k.P[0]}x{k.P[1]} by {k.P[2]}",
        "pad2d" => $"{k.P[0]}x{k.P[1]} to {k.P[2]}",
        "maxpool" or "avgpool" or "maxpool_bwd" or "avgpool_bwd" or "depthwise_conv" =>
            $"{k.P[0]}x{k.P[1]}x{k.P[2]} k{k.P[3]}x{k.P[4]} s{k.P[5]} p{(Even(k.P) ? k.P[7].ToString() : Pads(k.P))}",
        "adaptive_maxpool" or "adaptive_avgpool" or "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => $"{k.P[0]}x{k.P[1]}x{k.P[2]} to {k.P[3]}x{k.P[4]}",
        _ => "",
    };

    private static int ConvCols(int[] p) { var (oh, ow) = LayoutArgs.ConvOut(p); return p[0] * p[3] * p[4] * oh * ow; }

    private static int ConvCount(int[] p) { var (oh, ow) = LayoutArgs.ConvOut(p); return p[0] * oh * ow; }

    // A window's four pads (top, left, bottom, right): all one, or each.
    private static bool Even(int[] p) => p[7] == p[8] && p[8] == p[9] && p[9] == p[10];

    private static string Pads(int[] p) => $"{p[7]},{p[8]},{p[9]},{p[10]}";

    // An interpolation C H W OH OW with a map an axis.
    private static int[] Resize(int c, int h, int w, int oh, int ow, (int A, int B, int D) mh, (int A, int B, int D) mw) => [c, h, w, oh, ow, mh.A, mh.B, mh.D, mw.A, mw.B, mw.D];

    // The resize by the centre maps at in/out (LayoutArgs.CentreMap).
    private static int[] Centres(int c, int h, int w, int oh, int ow) => Resize(c, h, w, oh, ow, LayoutArgs.CentreMap(h, oh), LayoutArgs.CentreMap(w, ow));

    private static bool IsCentres(int[] p) => p.AsSpan().SequenceEqual(Centres(p[0], p[1], p[2], p[3], p[4]));

    // The pools' gate shapes: C H W kh kw sh sw top left bottom right — even pads; a bottom and right pad that add a window
    // (a last window past the input); uneven pads each side; a window longer than its image (no windows: the backward's dx 0).
    private static readonly int[][] PoolShapes =
    [
        [3, 7, 9, 3, 3, 2, 2, 1, 1, 1, 1], [2, 11, 5, 2, 3, 1, 2, 0, 1, 0, 1], [1, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2],
        [3, 7, 9, 3, 3, 2, 2, 1, 1, 2, 2], [2, 11, 5, 2, 3, 1, 2, 0, 1, 1, 2], [1, 6, 7, 3, 3, 2, 2, 0, 0, 1, 1], [2, 8, 8, 3, 2, 3, 3, 0, 1, 2, 1],
        [1, 2, 5, 3, 3, 1, 1, 0, 1, 0, 1],
    ];

    /// <summary>The gate's cases for an operation: odd dimensions, transposed and broadcast views, pads and strides, duplicates.</summary>
    public static IEnumerable<LayoutArgs> Gate(string name, string type)
    {
        switch (name)
        {
            case "strided":
                // x of shape (7, 5, 3, 2) read as (2, 3, 5, 7): the full permutation; a broadcast (stride 0) of a row; a slice with an offset.
                yield return Make(name, type, 7 * 5 * 3 * 2, null, [2, 3, 5, 7, 1, 2, 6, 30, 0], 210, 61);
                yield return Make(name, type, 37 * 11, null, [1, 5, 3, 11, 0, 0, 11, 1, 11 * 7], 165, 62);
                yield return Make(name, type, 37 * 11, null, [1, 1, 9, 5, 0, 0, 11, 1, 11 * 3 + 2], 45, 63);
                // Transpositions along the middle and the third dimension (stride 1 there, not last): the tiled copy, partial tiles.
                yield return Make(name, type, 360, null, [3, 4, 5, 6, 120, 1, 24, 4, 0], 360, 66);
                yield return Make(name, type, 360, null, [3, 4, 5, 6, 120, 30, 1, 5, 0], 360, 67);
                // Six dimensions: a pixel shuffle's (n, c, r, r, h, w) read as (n, c, h, r, w, r) (n and c merge: five left), a full
                // permutation of six, and one that merges to a single contiguous run.
                yield return Make(name, type, 480, null, [2, 3, 5, 2, 4, 2, 240, 80, 4, 40, 1, 20, 0], 480, 68);
                yield return Make(name, type, 144, null, [2, 3, 3, 2, 2, 2, 1, 4, 24, 2, 12, 72, 0], 144, 69);
                yield return Make(name, type, 130, null, [1, 2, 1, 3, 4, 5, 0, 60, 0, 20, 5, 1, 7], 120, 60);
                // Other ranks: a single value (rank 0), a strided slice (1), a transposition (2), a broadcast over a third (3),
                // a permutation of five, seven dimensions that do not merge (the device kernels' largest rank), and eight that
                // merge to two.
                yield return Make(name, type, 7, null, [5], 1, 51);
                yield return Make(name, type, 30, null, [9, 3, 2], 9, 52);
                yield return Make(name, type, 35, null, [5, 7, 1, 5, 0], 35, 53);
                yield return Make(name, type, 20, null, [3, 4, 5, 0, 5, 1, 0], 60, 54);
                yield return Permuted(type, [2, 3, 2, 3, 2], [3, 0, 4, 2, 1], 55);
                yield return Permuted(type, [2, 2, 2, 2, 2, 2, 2], [6, 5, 4, 3, 2, 1, 0], 56);
                yield return Permuted(type, [2, 3, 1, 4, 5, 1, 2, 3], [4, 5, 6, 7, 0, 1, 2, 3], 57);
                // Negative strides: a row reversed, and a transposition with its columns reversed (the tiled copy).
                yield return Make(name, type, 9, null, [9, -1, 8], 9, 58);
                yield return Make(name, type, 35, null, [5, 7, 1, -5, 30], 35, 59);
                break;
            case "one_hot":
                yield return Make(name, type, 1, Indices(37, 17, 64), [37, 17], 37 * 17, 64);
                break;
            case "gather_rows":
                yield return Make(name, type, 37 * 1001, Indices(23, 37, 65), [37, 1001, 23], 23 * 1001, 65);
                yield return Make(name, type, 11 * 5, Indices(13, 11, 66), [11, 5, 13], 13 * 5, 66);
                break;
            case "gather_cols":
                yield return Make(name, type, 37 * 1001, Indices(19, 1001, 67), [37, 1001, 19], 37 * 19, 67);
                break;
            case "scatter_add_rows":
                yield return Make(name, type, 37 * 1001, Indices(37, 11, 68), [37, 1001, 11], 11 * 1001, 68);
                yield return Make(name, type, 9 * 5, Indices(9, 3, 69), [9, 5, 3], 3 * 5, 69);
                break;
            case "im2col":
            case "col2im":
            {
                // C H W kh kw sh sw top left bottom right dh dw: even pads, dilations, a 1 × 1 kernel, uneven pads with a column
                // dilation, and a window longer than its image (no columns: col2im's dx 0).
                int[][] shapes = [[3, 7, 9, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1], [2, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2], [1, 11, 7, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                    [2, 6, 7, 3, 3, 2, 1, 0, 2, 1, 0, 1, 2], [1, 2, 5, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1]];
                foreach (var p in shapes)
                {
                    int image = p[0] * p[1] * p[2], cols = ConvCols(p);
                    yield return name == "im2col" ? Make(name, type, image, null, p, cols, 70) : Make(name, type, cols, null, p, image, 71);
                }

                break;
            }

            case "interpolate_nearest":
            case "interpolate_bilinear":
            case "interpolate_nearest_bwd":
            case "interpolate_bilinear_bwd":
            {
                // Maps: the centres at in/out, down and up, and at given scales (1/2 exactly; 2/3 where in/out is 5/7); the origins
                // (s = o·scale); the corners (s = o·(in − 1)/(out − 1), and a constant α = 0 for an axis of one output); ties for
                // nearest (s = 2o + ½); a map before the input's start (held at 0) and one past its end (held at the last).
                bool backward = name.EndsWith("_bwd", StringComparison.Ordinal);
                var shapes = new List<int[]>
                {
                    Centres(3, 7, 9, 5, 11), Centres(2, 7, 9, 13, 4), Centres(1, 4, 6, 8, 12),
                    Resize(1, 4, 6, 8, 12, LayoutArgs.CentreMap(1, 2), LayoutArgs.CentreMap(1, 2)), Resize(1, 5, 5, 7, 7, LayoutArgs.CentreMap(2, 3), LayoutArgs.CentreMap(2, 3)),
                    Resize(2, 7, 9, 13, 4, (7, 0, 13), (9, 0, 4)), Resize(2, 5, 5, 7, 7, (2, 0, 3), (2, 0, 3)),
                    Resize(3, 7, 9, 5, 11, (6, 0, 4), (8, 0, 10)), Resize(2, 4, 6, 9, 1, (3, 0, 8), (0, 0, 1)),
                    Centres(1, 8, 8, 4, 4), Resize(2, 6, 7, 5, 6, (3, -5, 2), (5, 3, 2)),
                };
                int seed = 72;
                foreach (var p in shapes)
                {
                    int image = p[0] * p[1] * p[2], resized = p[0] * p[3] * p[4];
                    yield return backward ? Make(name, type, resized, null, p, image, seed++) : Make(name, type, image, null, p, resized, seed++);
                }

                break;
            }
            case "shift":
                yield return Make(name, type, 37 * 1001, null, [37, 1001, 5], 37 * 1001, 75);
                yield return Make(name, type, 37 * 1001, null, [37, 1001, -3], 37 * 1001, 76);
                break;
            case "pad2d":
                yield return Make(name, type, 37 * 53, null, [37, 53, 56], 37 * 56, 77);
                yield return Make(name, type, 29 * 7, null, [29, 7, 8], 29 * 8, 78);
                break;
            case "maxpool":
            case "avgpool":
            {
                foreach (var p in PoolShapes) yield return Make(name, type, p[0] * p[1] * p[2], null, p, PoolCount(p), 92);
                break;
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                foreach (var p in PoolShapes) yield return Make(name, type, PoolCount(p), null, p, p[0] * p[1] * p[2], 93, p[0] * p[1] * p[2]);
                break;
            }

            case "adaptive_maxpool":
            case "adaptive_avgpool":
                yield return Make(name, type, 3 * 7 * 9, null, [3, 7, 9, 5, 4], 3 * 5 * 4, 94);
                yield return Make(name, type, 2 * 5 * 5, null, [2, 5, 5, 7, 3], 2 * 7 * 3, 95);
                break;
            case "adaptive_maxpool_bwd":
            case "adaptive_avgpool_bwd":
                yield return Make(name, type, 3 * 5 * 4, null, [3, 7, 9, 5, 4], 3 * 7 * 9, 97, 3 * 7 * 9);
                yield return Make(name, type, 2 * 7 * 3, null, [2, 5, 5, 7, 3], 2 * 5 * 5, 98, 2 * 5 * 5);
                break;
            case "depthwise_conv":
            {
                // C H W kh kw sh sw top left bottom right dh dw. A column stride of 2 (the value-at-a-time loop); rows of 37 with a
                // column dilation of 3 (the stride-1 vector passes over several vectors and a tail, taps reaching past both
                // edges); uneven pads; pads longer than the window (windows of zeros alone).
                int[][] shapes = [[3, 7, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1], [2, 11, 5, 3, 2, 2, 1, 1, 0, 1, 0, 2, 1], [2, 9, 11, 3, 3, 1, 2, 1, 1, 1, 1, 1, 2],
                    [1, 5, 37, 2, 3, 1, 1, 0, 2, 0, 2, 1, 3], [2, 6, 8, 3, 3, 1, 1, 2, 0, 0, 1, 1, 1], [1, 4, 4, 2, 2, 1, 1, 3, 0, 0, 3, 1, 1]];
                foreach (var p in shapes) yield return Make(name, type, p[0] * p[1] * p[2], null, p, ConvCount(p), 96, p[0] * p[3] * p[4]);
                break;
            }
        }
    }

    /// <summary>The bench's case for an operation.</summary>
    public static LayoutArgs Bench(string name, string type) => name switch
    {
        // a (64, 256, 1024) array read as (1024, 64, 256): a 3-d permutation of 16M values
        "strided" => Make(name, type, 64 * 256 * 1024, null, [1, 1024, 64, 256, 0, 1, 256 * 1024, 1024, 0], 64 * 256 * 1024, 81),
        "one_hot" => Make(name, type, 1, Indices(16384, 1024, 82), [16384, 1024], 16384 * 1024, 82),
        "gather_rows" => Make(name, type, 4096 * 4096, Indices(4096, 4096, 83), [4096, 4096, 4096], 4096 * 4096, 83),
        "gather_cols" => Make(name, type, 4096 * 4096, Indices(4096, 4096, 84), [4096, 4096, 4096], 4096 * 4096, 84),
        "scatter_add_rows" => Make(name, type, 4096 * 4096, Indices(4096, 4096, 85), [4096, 4096, 4096], 4096 * 4096, 85),
        "im2col" => Make(name, type, 64 * 128 * 128, null, [64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1], ConvCols([64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1]), 86),
        "col2im" => Make(name, type, ConvCols([64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1]), null, [64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1], 64 * 128 * 128, 87),
        "interpolate_nearest" => Make(name, type, 16 * 512 * 512, null, Centres(16, 512, 512, 1024, 1024), 16 * 1024 * 1024, 88),
        "interpolate_bilinear" => Make(name, type, 16 * 512 * 512, null, Centres(16, 512, 512, 1024, 1024), 16 * 1024 * 1024, 89),
        "interpolate_nearest_bwd" => Make(name, type, 16 * 1024 * 1024, null, Centres(16, 512, 512, 1024, 1024), 16 * 512 * 512, 88),
        "interpolate_bilinear_bwd" => Make(name, type, 16 * 1024 * 1024, null, Centres(16, 512, 512, 1024, 1024), 16 * 512 * 512, 89),
        "shift" => Make(name, type, 4096 * 4096, null, [4096, 4096, 3], 4096 * 4096, 90),
        "pad2d" => Make(name, type, 4096 * 4093, null, [4096, 4093, 4096], 4096 * 4096, 91),
        // 64 channels of 256 × 256: 2 × 2 windows at stride 2 (the common pool), 3 × 3 depthwise at stride 1
        "maxpool" or "avgpool" => Make(name, type, 64 * 256 * 256, null, [64, 256, 256, 2, 2, 2, 2, 0, 0, 0, 0], PoolCount([64, 256, 256, 2, 2, 2, 2, 0, 0, 0, 0]), 92),
        "maxpool_bwd" or "avgpool_bwd" => Make(name, type, PoolCount([64, 256, 256, 2, 2, 2, 2, 0, 0, 0, 0]), null, [64, 256, 256, 2, 2, 2, 2, 0, 0, 0, 0], 64 * 256 * 256, 93, 64 * 256 * 256),
        "adaptive_maxpool" or "adaptive_avgpool" => Make(name, type, 64 * 256 * 256, null, [64, 256, 256, 7, 7], 64 * 7 * 7, 94),
        "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => Make(name, type, 64 * 7 * 7, null, [64, 256, 256, 7, 7], 64 * 256 * 256, 97, 64 * 256 * 256),
        "depthwise_conv" => Make(name, type, 64 * 256 * 256, null, [64, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1], ConvCount([64, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1]), 96, 64 * 9),
        _ => throw new ArgumentException(name),
    };
}
