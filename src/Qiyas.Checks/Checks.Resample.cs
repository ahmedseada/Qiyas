// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The resampling with a per-channel normalization (Resampling), on every device against its float64 reference: every float
// output type from uint8 by Pillow's 8-bit passes (by a table or none: the integers exact, so the stored values equal the
// reference's), and from uint8, int16, float16, float32, float64 and the output's own type by the float passes (by a scale
// and a shift or none, within the lane's rounding), over each filter, shrinking, enlarging, one axis or neither.
namespace Qiyas;

internal static partial class Checks
{
    private static readonly (string Name, int Planes, int Channels, int H, int W, int OH, int OW, ResampleFilter Filter)[] ResampleCases =
    [
        ("17x23 to 7x11 bicubic", 3, 3, 17, 23, 7, 11, ResampleFilter.Bicubic),
        ("5x6 to 13x9 bilinear", 2, 1, 5, 6, 13, 9, ResampleFilter.Bilinear),
        ("9x9 to 9x4 lanczos", 3, 3, 9, 9, 9, 4, ResampleFilter.Lanczos),
        ("8x5 to 3x5 box", 6, 3, 8, 5, 3, 5, ResampleFilter.Box),
        ("12x10 to 5x7 hamming", 2, 2, 12, 10, 5, 7, ResampleFilter.Hamming),
        ("37x3 to 11x1 lanczos", 1, 1, 37, 3, 11, 1, ResampleFilter.Lanczos),
        ("4x4 kept", 3, 3, 4, 4, 4, 4, ResampleFilter.Bicubic),
    ];

    private static readonly double[] ImageMean = [0.485, 0.456, 0.406], ImageStd = [0.229, 0.224, 0.225];

    public static bool Resamples(IDevice<ulong, ulong>? ops, string device = "device")
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        string[] inputs = ["uint8", "int16", "float16", "float32", "float64"];
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            bool f64 = type == "float64";
            foreach (var k in ResampleCases)
                foreach (bool bytes in new[] { true, false })
                    foreach (string input in bytes ? ["uint8"] : inputs.Append(type).Distinct())
                        foreach (bool norm in new[] { true, false })
                        {
                            var (a, bounds, raw) = Resampling.Plan(k.Planes, k.Channels, k.H, k.W, k.OH, k.OW, k.Filter, bytes);
                            var weights = bytes ? raw : Resampling.InLane(raw, f64);
                            var (low, high) = input switch { "uint8" => (0.0, 255.0), "int16" => (-300.0, 300.0), _ => (-4.0, 4.0) };
                            var x = Bench.Registry.Make(input, a.InputCount, 91, low, high);
                            var xv = Bench.Registry.Values(input, x, a.InputCount);
                            double[]? values = null;
                            if (norm)
                            {
                                // ImageNet's normalization: by a table of each byte's ((v / 255) − mean) / std, else by v / (255·std) − mean / std.
                                values = new double[a.ValueCount];
                                for (int c = 0; c < k.Channels; c++)
                                {
                                    double m = ImageMean[c % 3], s = ImageStd[c % 3];
                                    if (bytes) for (int v = 0; v < 256; v++) values[256 * c + v] = (v / 255.0 - m) / s;
                                    else { values[2 * c] = 1 / (255 * s); values[2 * c + 1] = -m / s; }
                                }

                                values = Resampling.InLane(values, f64);
                            }

                            var (ry, scale) = Resampling.Reference(a, xv, bounds, weights, values);
                            var y = new byte[Bench.Registry.Bytes(type, a.Count)];
                            var vb = values is null ? null : Resampling.LaneBytes(values, f64);
                            Bench.Registry.With(input, new ResampleRun(ops, type, a, x, bounds, Resampling.WeightBytes(a, weights, f64), vb, y));
                            string what = $"resample {k.Name} {(bytes ? "8-bit" : input)}{(norm ? " normalized" : "")}";
                            double worst;
                            if (bytes)
                            {
                                // The integers and the table's values exact: the stored values are the reference's stored.
                                var want = Bench.Registry.Values(type, Bench.Registry.Encode(type, ry), a.Count);
                                var got = Bench.Registry.Values(type, y, a.Count);
                                int bad = Enumerable.Range(0, a.Count).FirstOrDefault(i => !got[i].Equals(want[i]), -1);
                                worst = bad < 0 ? 0 : double.PositiveInfinity;
                                if (bad >= 0 && Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1") Console.WriteLine($"  {what} {type} [{bad}] got {got[bad]:R} want {want[bad]:R}");
                            }
                            else worst = Excess(type, what, y, ry, i => scale[i], 1);
                            if (worst > 0) { Console.WriteLine($"check {what,-56} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
                        }
        }

        if (ok) Console.WriteLine($"check resamples              {where,-6} every float type: Pillow's filters shrinking, enlarging, one axis and neither, 8-bit exact (by a table or none) and from 6 input types in the lane (by a scale and a shift or none): as the references");
        return ok;
    }

    private sealed class ResampleRun(IDevice<ulong, ulong>? ops, string type, ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : Bench.ITypeVisitor<string>
    {
        public string Visit<TI>() where TI : struct, IAdad<TI, float> => Bench.Registry.WithFloat(type, new Out<TI>(ops, a, x, bounds, weights, values, y));

        private sealed class Out<TI>(IDevice<ulong, ulong>? ops, ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : Bench.IFloatTypeVisitor<string>
            where TI : struct, IAdad<TI, float>
        {
            public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
            {
                if (ops is null) return CpuDevice.Device.Resample<TI, TN>(a, x, bounds, weights, values, y);
                using var st = new Staged(ops);
                var lane = ops.Resample<TI, TN>(a, st.In(x), st.Indices(bounds), st.In(weights), st.In(values), st.Out(y));
                st.Collect();
                return lane;
            }
        }
    }
}
