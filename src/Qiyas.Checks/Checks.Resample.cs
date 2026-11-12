// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The resampling with a scale and a shift a channel (Resampling), on every device against its float64 reference: every
// float output type from uint8, int16, float16, float32, float64 and the output's own type (by a scale and a shift a
// channel or none, within the lane's rounding), over each filter, shrinking, enlarging, one axis or neither.
namespace Qiyas;

internal static partial class Checks
{
    private static readonly (string Name, int Planes, int Channels, int H, int W, int OH, int OW, ResampleFilter Filter)[] ResampleCases =
    [
        ("17x23 to 7x11 cubic", 3, 3, 17, 23, 7, 11, ResampleFilter.Cubic),
        ("5x6 to 13x9 triangle", 2, 1, 5, 6, 13, 9, ResampleFilter.Triangle),
        ("9x9 to 9x4 lanczos", 3, 3, 9, 9, 9, 4, ResampleFilter.Lanczos),
        ("8x5 to 3x5 box", 6, 3, 8, 5, 3, 5, ResampleFilter.Box),
        ("12x10 to 5x7 hamming", 2, 2, 12, 10, 5, 7, ResampleFilter.Hamming),
        ("37x3 to 11x1 lanczos", 1, 1, 37, 3, 11, 1, ResampleFilter.Lanczos),
        ("4x4 kept", 3, 3, 4, 4, 4, 4, ResampleFilter.Cubic),
    ];

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
                foreach (string input in inputs.Append(type).Distinct())
                    foreach (bool norm in new[] { true, false })
                    {
                        var (a, bounds, raw) = Resampling.Plan(k.Planes, k.Channels, k.H, k.W, k.OH, k.OW, k.Filter);
                        var weights = Resampling.InLane(raw, f64);
                        var (low, high) = input switch { "uint8" => (0.0, 255.0), "int16" => (-300.0, 300.0), _ => (-4.0, 4.0) };
                        var x = Bench.Registry.Make(input, a.InputCount, 91, low, high);
                        var xv = Bench.Registry.Values(input, x, a.InputCount);
                        double[]? values = null;
                        if (norm)
                        {
                            // A scale and a shift a channel that bring the input's range near [−2, 2], a different pair each.
                            values = new double[a.ValueCount];
                            for (int c = 0; c < k.Channels; c++) { values[2 * c] = (3.7 + c) / (high - low) / 1.9; values[2 * c + 1] = -0.43 - 0.17 * c; }
                            values = Resampling.InLane(values, f64);
                        }

                        var (ry, scale) = Resampling.Reference(a, xv, bounds, weights, values);
                        var y = new byte[Bench.Registry.Bytes(type, a.Count)];
                        var vb = values is null ? null : Resampling.LaneBytes(values, f64);
                        Bench.Registry.With(input, new ResampleRun(ops, type, a, x, bounds, Resampling.LaneBytes(weights, f64), vb, y));
                        string what = $"resample {k.Name} {input}{(norm ? " scaled" : "")}";
                        double worst = Excess(type, what, y, ry, i => scale[i], 1);
                        if (worst > 0) { Console.WriteLine($"check {what,-56} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
                    }
        }

        if (ok) Console.WriteLine($"check resamples              {where,-6} every float type: the box, triangle, Hamming, cubic and Lanczos filters shrinking, enlarging, one axis and neither, from 6 input types in the lane (by a scale and a shift a channel or none): as the references");
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
