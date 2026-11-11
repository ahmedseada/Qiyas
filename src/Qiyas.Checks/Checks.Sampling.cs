// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling (IDecoding.SampleFiltered) on every device, every float type (the block formats among them): rows
// of 601 scores (past a block of threads) and rows a stride apart from an offset, at a temperature, held to the top-k, to
// the nucleus, to min-p and to all three, top-k of 1 and top-k past the row, rows of 20011 scores, and a top-k that leaves
// more columns than a device gathers (the nucleus then by the passes of the row); on the types with an infinity a row whose
// largest score is +∞ (drawn among its +∞ columns), a row with a NaN among its scores (never drawn) and a row of −∞ (no
// column: −1), and rows whose scores at the threshold of top-k and of the nucleus are one step of the type apart (two
// weights the lane may hold as one, kept together then). Each row's column is held to the float64 rule within the lane's rounding: it is a column the rule keeps
// (the thresholds' weights taken a few lane steps either way, the weights the lane cannot hold not counted as kept), no
// kept column before it has reached the seed's draw at (step, row), and it has (or it is the last kept one); its
// probability within the type's step of its weight over the kept sum.
namespace Qiyas;

internal static partial class Checks
{
    private static bool Samplings(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        bool infinities = type is "float32" or "float64" or "float16" or "bfloat16";
        double laneEps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23), floor = type == "float64" ? 1e-290 : 1e-30;
        int mant = Bench.Registry.Mantissa(type);
        var cases = new (string Name, SamplingArgs A, int Step, bool Probability)[]
        {
            ("every column", new SamplingArgs(37, 601, 1, 0, 0, 0, 7), 0, true),
            ("a temperature and top-k", new SamplingArgs(37, 601, 0.7, 40, 0, 0, 7), 5, true),
            ("the nucleus", new SamplingArgs(37, 601, 1.3, 0, 0.9, 0, 11), 5, false),
            ("min-p", new SamplingArgs(37, 601, 1, 0, 0, 0.05, 11), 123456, true),
            ("top-k, the nucleus and min-p, rows apart", new SamplingArgs(5, 53, 0.8, 20, 0.95, 0.02, 13) { RowStride = 61, RowOffset = 3 }, 9, true),
            ("the largest alone", new SamplingArgs(37, 601, 1, 1, 0, 0, 7), 2, true),
            ("top-k past the row", new SamplingArgs(7, 37, 1, 64, 0.5, 0, 7), 2, false),
            ("rows of 20011, all three", new SamplingArgs(5, 20011, 0.8, 40, 0.95, 0.02, 19), 4, true),
            ("rows of 20011, every column", new SamplingArgs(5, 20011, 1, 0, 0, 0, 19), 4, true),
            ("the top 600 of 20011 and the nucleus", new SamplingArgs(5, 20011, 1, 600, 0.9, 0, 23), 1, true),
            ("the top 300 of 601 and the nucleus", new SamplingArgs(5, 601, 1, 300, 0.9, 0, 23), 1, false),
            ("scores a step apart at top-k", new SamplingArgs(37, 601, 0.8, 4, 0, 0, 29), 6, true),
            ("scores a step apart at the nucleus", new SamplingArgs(37, 601, 0.8, 6, 0.66, 0, 29), 6, true),
        };
        foreach (var (name, a, step, withProbability) in cases)
        {
            string what = $"sample {name}";
            int xn = (a.Rows - 1) * a.Stride + a.RowOffset + a.Cols;
            var x = Bench.Registry.Make(type, xn, 97, -4, 4);
            var xv = Bench.Registry.Values(type, x, xn);
            if (name.StartsWith("scores a step apart", StringComparison.Ordinal))
            {
                // Each row's six largest scores planted, the fourth and fifth one step of the type apart (the same value where
                // the type has no step that small): the threshold of top-k (4), and of the nucleus (0.66 of the six: reached at
                // the fourth), lies between two weights the lane may hold as one. The float64 rule keeps four; a lane in which
                // the two weights are equal keeps them together, five: either is the rule within the lane's rounding.
                double TypeRound(double v) => Bench.Registry.Values(type, Bench.Registry.Encode(type, [v]), 1)[0];
                double fifth = TypeRound(3.96), fourth = TypeRound(fifth + Math.Pow(2, Math.Floor(Math.Log2(fifth)) - mant));
                // float32 and float64: the first such pair from 3.96 whose scores over the temperature are one value of the
                // lane, so the lane's weights are equal (float64's are the reference's too: kept together by both).
                if (type == "float32")
                {
                    float f = 3.96f, t = (float)a.Temperature;
                    while (f / t != MathF.BitIncrement(f) / t) f = MathF.BitIncrement(f);
                    fifth = f; fourth = MathF.BitIncrement(f);
                }
                else if (type == "float64")
                {
                    double f = 3.96;
                    while (f / a.Temperature != Math.BitIncrement(f) / a.Temperature) f = Math.BitIncrement(f);
                    fifth = f; fourth = Math.BitIncrement(f);
                }

                for (int r = 0; r < a.Rows; r++)
                {
                    int at = r * a.Stride + a.RowOffset;
                    for (int c = 0; c < a.Cols; c++) if (xv[at + c] > 3.5) xv[at + c] -= 1;
                    xv[at + 7] = 3.99; xv[at + 100] = 3.98; xv[at + 205] = 3.97; xv[at + 333] = fifth; xv[at + 420] = fourth; xv[at + 590] = 3.9;
                }

                x = Bench.Registry.Encode(type, xv);
                xv = Bench.Registry.Values(type, x, xn);
            }

            if (infinities && a.Rows > 3)
            {
                // A row whose largest is +∞ (twice), a row with a NaN among its scores, a row of −∞.
                xv[1 * a.Stride + a.RowOffset + 5] = xv[1 * a.Stride + a.RowOffset + 17] = double.PositiveInfinity;
                xv[2 * a.Stride + a.RowOffset + 4] = double.NaN;
                for (int c = 0; c < a.Cols; c++) xv[3 * a.Stride + a.RowOffset + c] = double.NegativeInfinity;
                x = Bench.Registry.Encode(type, xv);
                xv = Bench.Registry.Values(type, x, xn);
            }

            var picks = Enumerable.Repeat(-9, a.Rows).ToArray();
            var probability = withProbability ? new byte[Bench.Registry.Bytes(type, a.Rows)] : null;
            Bench.Registry.WithFloat(type, new SampleFilteredRun(ops, a, x, step, picks, probability));
            var got = probability is null ? null : Bench.Registry.Values(type, probability, a.Rows);
            uint key = Draws.Key(a.Seed);
            string? wrong = null;
            if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1")
            {
                var (reference, _) = a.Reference(xv, step);
                var off = a.PickErrors(xv, step, picks);
                Console.WriteLine($"  {what} {where} {type}: {Enumerable.Range(0, a.Rows).Count(r => picks[r] == reference[r])} of {a.Rows} rows the reference's column");
                for (int r = 0; r < a.Rows; r++)
                    if (picks[r] != reference[r]) Console.WriteLine($"    row {r}: column {picks[r]}, the reference's {reference[r]}, off the draw by {off[r]:E3} of the probability (draw {Draws.Uniform(key, (uint)step, (uint)r):R})");
            }

            for (int r = 0; r < a.Rows && wrong is null; r++)
            {
                var e = a.Weights(xv.AsSpan(r * a.Stride + a.RowOffset, a.Cols));
                wrong = SamplePlausible(a, e, Draws.Uniform(key, (uint)step, (uint)r), picks[r], laneEps, floor, out double low, out double high);
                if (wrong is null && got is not null)
                {
                    // The probability: the pick's weight over the kept sum, within the kept sum's bounds, the lane's rounding
                    // and the type's step (a block format's: its block's, the largest probability 1).
                    double middle = (low + high) / 2, typeStep = Bench.Registry.IsBlock(type) ? Math.Pow(2, 1 - mant)
                        : Math.Max(Bench.Registry.MinStep(type), middle > 0 ? Math.Pow(2, Math.Floor(Math.Log2(middle)) - mant) : 0);
                    if (!(Math.Abs(got[r] - middle) <= (high - low) / 2 + typeStep + 256 * laneEps)) wrong = $"probability {got[r]:R}, not within {low:R} .. {high:R}";
                }

                if (wrong is not null) wrong = $"row {r}: {wrong}";
            }

            if (wrong is not null) { Console.WriteLine($"check {what,-50} {where,-6} {type,-12} DIFFERS: {wrong}"); ok = false; }
        }

        return ok;
    }

    // Null when `pick` is a column the rule allows for a row of float64 weights e and the draw u within the lane's rounding;
    // else what is wrong. low .. high: the bounds of the pick's probability among the kept.
    private static string? SamplePlausible(SamplingArgs a, double[] e, double u, int pick, double laneEps, double floor, out double low, out double high)
    {
        int n = e.Length;
        double rho = 256 * laneEps, eps = 4 * n * laneEps;
        low = high = 0;
        var sorted = (double[])e.Clone();
        Array.Sort(sorted);
        Array.Reverse(sorted);
        var prefix = new double[n + 1];
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + sorted[i];
        // The sum of the weights at least t.
        double Mass(double t)
        {
            int lo = 0, hi = n;
            while (lo < hi) { int mid = (lo + hi) / 2; if (sorted[mid] >= t) lo = mid + 1; else hi = mid; }
            return prefix[lo];
        }

        // The threshold's bounds: a device's is between them.
        double tLow = 0, tHigh = 0;
        if (a.HasTopK) { tLow = sorted[a.TopK - 1] * (1 - rho); tHigh = sorted[a.TopK - 1] * (1 + rho); }
        if (a.HasTopP)
        {
            double massLow = Mass(tHigh), massHigh = Mass(tLow);
            double? upper = null, lower = null;
            for (int i = 0; i < n && sorted[i] >= tLow && (upper is null || lower is null); i++)
            {
                double v = sorted[i];
                if (upper is null && Mass(Math.Max(v * (1 - rho), tLow)) * (1 + eps) >= a.TopP * massLow * (1 - eps)) upper = v;
                if (lower is null && Mass(Math.Max(v * (1 + rho), tHigh)) * (1 - eps) >= a.TopP * massHigh * (1 + eps)) lower = v;
            }

            tHigh = Math.Max(tHigh, (upper ?? 0) * (1 + rho));
            tLow = Math.Max(tLow, (lower ?? 0) * (1 - rho));
        }

        if (a.HasMinP) { tLow = Math.Max(tLow, a.MinP * (1 - rho)); tHigh = Math.Max(tHigh, a.MinP * (1 + rho)); }
        bool Sure(double v) => v >= tHigh && v > floor;                          // kept on every device
        bool May(double v) => v >= tLow && v > 0;                                // kept on some
        double sure = 0, may = 0;
        for (int c = 0; c < n; c++) { if (Sure(e[c])) sure += e[c]; if (May(e[c])) may += e[c]; }
        if (pick < 0) return sure > 0 ? "no column, where the row keeps some" : null;
        if (pick >= n) return $"column {pick} is past the row";
        if (!May(e[pick])) return $"column {pick} (weight {e[pick]:R}) is not kept (the threshold {tLow:R} .. {tHigh:R})";
        double before = 0, through = 0;
        bool last = true;
        for (int c = 0; c < n; c++)
        {
            if (c < pick && Sure(e[c])) before += e[c];
            if (c <= pick && May(e[c])) through += e[c];
            if (c > pick && Sure(e[c])) last = false;
        }

        if (before > 0 && before * (1 - eps) > u * may * (1 + eps)) return $"column {pick}: the kept columns before it ({before:R}) had reached the draw ({u:R} of {may:R})";
        if (!last && through * (1 + eps) < u * sure * (1 - eps)) return $"column {pick}: the running sum ({through:R}) has not reached the draw ({u:R} of {sure:R})";
        low = e[pick] / may;
        high = sure > 0 ? Math.Min(1, e[pick] / sure) : 1;
        return null;
    }

    private sealed class SampleFilteredRun(IDevice<ulong, ulong>? ops, SamplingArgs a, byte[] x, int step, int[] picks, byte[]? probability) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.SampleFiltered<TN>(a, x, [step], picks, probability);
            using var st = new Staged(ops);
            string lane = ops.SampleFiltered<TN>(a, st.In(x), st.Indices([step]), st.IndicesOut(picks), st.Out(probability));
            st.Collect();
            return lane;
        }
    }
}
