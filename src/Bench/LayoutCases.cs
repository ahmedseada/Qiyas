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

    /// <summary>The second input's values: the forward input of a maxpool backward, the weights of a depthwise convolution.</summary>
    public static int SecondCount(LayoutArgs k) => k.Name switch
    {
        "maxpool_bwd" or "avgpool_bwd" or "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => k.P[0] * k.P[1] * k.P[2],
        "depthwise_conv" => k.P[0] * k.P[3] * k.P[4],
        _ => 0,
    };

    private static int PoolCount(int[] p) { var (oh, ow) = LayoutArgs.PoolOut(p); return p[0] * oh * ow; }

    public static string Shape(LayoutArgs k) => k.Name switch
    {
        "strided" => $"{k.P[0]}x{k.P[1]}x{k.P[2]}x{k.P[3]}",
        "one_hot" => $"{k.P[0]}x{k.P[1]}",
        "gather_rows" => $"{k.P[2]} of {k.P[0]}x{k.P[1]}",
        "gather_cols" => $"{k.P[2]} of {k.P[0]}x{k.P[1]}",
        "scatter_add_rows" => $"{k.P[0]} into {k.P[2]}x{k.P[1]}",
        "im2col" or "col2im" => $"{k.P[0]}x{k.P[1]}x{k.P[2]} k{k.P[3]}x{k.P[4]}",
        "interpolate_nearest" or "interpolate_bilinear" => $"{k.P[0]}x{k.P[1]}x{k.P[2]} to {k.P[3]}x{k.P[4]}",
        "shift" => $"{k.P[0]}x{k.P[1]} by {k.P[2]}",
        "pad2d" => $"{k.P[0]}x{k.P[1]} to {k.P[2]}",
        "maxpool" or "avgpool" or "maxpool_bwd" or "avgpool_bwd" or "depthwise_conv" => $"{k.P[0]}x{k.P[1]}x{k.P[2]} k{k.P[3]}x{k.P[4]} s{k.P[5]} p{k.P[7]}",
        "adaptive_maxpool" or "adaptive_avgpool" or "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => $"{k.P[0]}x{k.P[1]}x{k.P[2]} to {k.P[3]}x{k.P[4]}",
        _ => "",
    };

    private static int ConvCols(int[] p) { var (oh, ow) = LayoutArgs.ConvOut(p); return p[0] * p[3] * p[4] * oh * ow; }

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
                int[][] shapes = [[3, 7, 9, 3, 2, 2, 1, 1, 1, 1, 1, 0], [2, 5, 5, 3, 3, 1, 1, 1, 1, 2, 2, 0], [1, 11, 7, 1, 1, 1, 1, 0, 0, 1, 1, 0]];
                foreach (var p in shapes)
                {
                    int image = p[0] * p[1] * p[2], cols = ConvCols(p);
                    yield return name == "im2col" ? Make(name, type, image, null, p, cols, 70) : Make(name, type, cols, null, p, image, 71);
                }

                break;
            }

            case "interpolate_nearest":
            case "interpolate_bilinear":
                yield return Make(name, type, 3 * 7 * 9, null, [3, 7, 9, 5, 11], 3 * 5 * 11, 72);
                yield return Make(name, type, 2 * 7 * 9, null, [2, 7, 9, 13, 4], 2 * 13 * 4, 73);
                yield return Make(name, type, 1 * 4 * 6, null, [1, 4, 6, 8, 12], 8 * 12, 74);
                break;
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
                int[][] shapes = [[3, 7, 9, 3, 3, 2, 2, 1, 1], [2, 11, 5, 2, 3, 1, 2, 0, 1], [1, 6, 6, 4, 4, 4, 4, 2, 2]];
                foreach (var p in shapes) yield return Make(name, type, p[0] * p[1] * p[2], null, p, PoolCount(p), 92);
                break;
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                int[][] shapes = [[3, 7, 9, 3, 3, 2, 2, 1, 1], [2, 11, 5, 2, 3, 1, 2, 0, 1], [1, 6, 6, 4, 4, 4, 4, 2, 2]];
                foreach (var p in shapes) yield return Make(name, type, PoolCount(p), null, p, p[0] * p[1] * p[2], 93, p[0] * p[1] * p[2]);
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
                // The last two (6j): a column stride of 2 (the value-at-a-time loop), and rows of 37 with a column dilation of 3
                // (the stride-1 vector passes over several vectors and a tail, taps reaching past both edges).
                int[][] shapes = [[3, 7, 9, 3, 3, 1, 1, 1, 1, 1, 1], [2, 11, 5, 3, 2, 2, 1, 1, 0, 2, 1], [2, 9, 11, 3, 3, 1, 2, 1, 1, 1, 2], [1, 5, 37, 2, 3, 1, 1, 0, 2, 1, 3]];
                foreach (var p in shapes) yield return Make(name, type, p[0] * p[1] * p[2], null, p, PoolCount(p), 96, p[0] * p[3] * p[4]);
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
        "im2col" => Make(name, type, 64 * 128 * 128, null, [64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0], ConvCols([64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0]), 86),
        "col2im" => Make(name, type, ConvCols([64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0]), null, [64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 0], 64 * 128 * 128, 87),
        "interpolate_nearest" => Make(name, type, 16 * 512 * 512, null, [16, 512, 512, 1024, 1024], 16 * 1024 * 1024, 88),
        "interpolate_bilinear" => Make(name, type, 16 * 512 * 512, null, [16, 512, 512, 1024, 1024], 16 * 1024 * 1024, 89),
        "shift" => Make(name, type, 4096 * 4096, null, [4096, 4096, 3], 4096 * 4096, 90),
        "pad2d" => Make(name, type, 4096 * 4093, null, [4096, 4093, 4096], 4096 * 4096, 91),
        // 64 channels of 256 × 256: 2 × 2 windows at stride 2 (the common pool), 3 × 3 depthwise at stride 1
        "maxpool" or "avgpool" => Make(name, type, 64 * 256 * 256, null, [64, 256, 256, 2, 2, 2, 2, 0, 0], PoolCount([64, 256, 256, 2, 2, 2, 2, 0, 0]), 92),
        "maxpool_bwd" or "avgpool_bwd" => Make(name, type, PoolCount([64, 256, 256, 2, 2, 2, 2, 0, 0]), null, [64, 256, 256, 2, 2, 2, 2, 0, 0], 64 * 256 * 256, 93, 64 * 256 * 256),
        "adaptive_maxpool" or "adaptive_avgpool" => Make(name, type, 64 * 256 * 256, null, [64, 256, 256, 7, 7], 64 * 7 * 7, 94),
        "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => Make(name, type, 64 * 7 * 7, null, [64, 256, 256, 7, 7], 64 * 256 * 256, 97, 64 * 256 * 256),
        "depthwise_conv" => Make(name, type, 64 * 256 * 256, null, [64, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1], PoolCount([64, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1]), 96, 64 * 9),
        _ => throw new ArgumentException(name),
    };
}
