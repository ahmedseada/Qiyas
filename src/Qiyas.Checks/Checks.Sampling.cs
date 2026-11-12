// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling (IDecoding.SampleFiltered) on every device, every type but bool (the integers and the block formats
// among them): rows of 601 scores (past a block of threads) and rows a stride apart from an offset, at a temperature,
// held to the top-k, to the nucleus, to min-p and to all three; top-k of 1, the nucleus of 0 and min-p of 1; rows of
// 20011 scores; top-k at, under and over the most columns a device gathers for the nucleus, and the nucleus over a whole
// row (by the passes of the row). On the types with an infinity a row whose first-ranked score is +∞, a row with a NaN
// among its scores and a row of −∞ (no column: −1 and a NaN probability) and a row with some −∞ (never drawn); on the
// float types rows whose scores at the boundary of top-k and of the nucleus are one step of the type apart (two weights
// the lane may hold as one: the order of the scores as stored decides) and −0 and +0 at top-k's boundary (equal: cut by
// column); on the 64-bit integers scores past 2^53 (several one float64 value: the order as stored decides, top-k and
// the nucleus among them). The integers' rows hold few distinct scores, so equal scores at a boundary are cut by column.
// Each row is held to the float64 rule within the lane's rounding: the kept prefix of the row's order is top-k's exactly
// and the nucleus's and min-p's to within a lane's rounding of their sums and weights; for some such prefix the pick is
// a kept column, the kept columns before it had not passed the seed's draw at (step, row) and it does (or it is the last
// kept one), and its probability (the float lane's, as stored: the lane named) is its weight over the kept sum within the
// lane's rounding. bool refused.
namespace Qiyas;

internal static partial class Checks
{
    private static bool Samplings(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        bool isFloat = Bench.Registry.IsFloat(type), infinities = type is "float32" or "float64" or "float16" or "bfloat16";
        int bits = Bench.Registry.Bits(type), mant = Bench.Registry.Mantissa(type);
        bool wideLane = isFloat ? bits == 64 : bits > 16, wide64 = !isFloat && bits == 64;                  // Fmt.FloatLane
        string laneName = wideLane ? "f64" : "f32";
        double laneEps = wideLane ? Math.Pow(2, -52) : Math.Pow(2, -23), tiny = wideLane ? 1e-290 : 1e-30;
        var cases = new List<(string Name, SamplingArgs A, int Step, bool Probability)>
        {
            ("every column", new SamplingArgs(37, 601, 1, 601, 1, 0, 7), 0, true),
            ("a temperature and top-k", new SamplingArgs(37, 601, 0.7, 40, 1, 0, 7), 5, true),
            ("the nucleus", new SamplingArgs(37, 601, 1.3, 601, 0.9, 0, 11), 5, false),
            ("min-p", new SamplingArgs(37, 601, 1, 601, 1, 0.05, 11), 123456, true),
            ("top-k, the nucleus and min-p, rows apart", new SamplingArgs(5, 53, 0.8, 20, 0.95, 0.02, 13) { RowStride = 61, RowOffset = 3 }, 9, true),
            ("the first alone", new SamplingArgs(37, 601, 1, 1, 1, 0, 7), 2, true),
            ("the nucleus of 0", new SamplingArgs(7, 601, 1, 601, 0, 0, 5), 3, true),
            ("min-p of 1", new SamplingArgs(7, 601, 1, 601, 1, 1, 5), 3, true),
            ("top-k the whole row and the nucleus", new SamplingArgs(7, 37, 1, 37, 0.5, 0, 7), 2, false),
            ("rows of 20011, all three", new SamplingArgs(5, 20011, 0.8, 40, 0.95, 0.02, 19), 4, true),
            ("rows of 20011, every column", new SamplingArgs(5, 20011, 1, 20011, 1, 0, 19), 4, true),
            ("rows of 20011, the nucleus", new SamplingArgs(5, 20011, 1, 20011, 0.5, 0, 31), 2, true),
            ("the top 600 of 20011 and the nucleus", new SamplingArgs(5, 20011, 1, 600, 0.9, 0, 23), 1, true),
            ("the top 256 of 601 and the nucleus", new SamplingArgs(5, 601, 1, 256, 0.9, 0, 23), 1, true),
            ("the top 257 of 601 and the nucleus", new SamplingArgs(5, 601, 1, 257, 0.9, 0, 23), 1, false),
        };
        if (isFloat)
        {
            cases.Add(("scores a step apart at top-k", new SamplingArgs(37, 601, 0.8, 4, 1, 0, 29), 6, true));
            cases.Add(("scores a step apart at the nucleus", new SamplingArgs(37, 601, 0.8, 6, 0.66, 0, 29), 6, true));
            cases.Add(("-0 and +0 at top-k", new SamplingArgs(5, 601, 1, 2, 1, 0, 43), 1, true));
        }

        if (wide64)
        {
            cases.Add(("scores past 2^53 at top-k", new SamplingArgs(5, 601, 1, 2, 1, 0, 37), 1, true));
            cases.Add(("scores past 2^53 at the nucleus", new SamplingArgs(5, 601, 1, 3, 0.5, 0, 37), 1, true));
        }

        foreach (var (name, a, step, withProbability) in cases)
        {
            string what = $"sample {name}";
            int xn = (a.Rows - 1) * a.Stride + a.RowOffset + a.Cols;
            byte[] x;
            if (name.StartsWith("scores past 2^53", StringComparison.Ordinal))
            {
                // Each row's first four scores the type's largest and the three below it (one float64 value), apart from the
                // others by far more than the temperature, and its smallest: the first-ranked are told apart as stored.
                x = Wide64(type, xn, 41);
                for (int r = 0; r < a.Rows; r++)
                {
                    int at = r * a.Stride + a.RowOffset;
                    void Put(int c, long below)
                    {
                        if (type == "uint64") BitConverter.TryWriteBytes(x.AsSpan((at + c) * 8), below < 0 ? 0UL : ulong.MaxValue - (ulong)below);
                        else BitConverter.TryWriteBytes(x.AsSpan((at + c) * 8), below < 0 ? long.MinValue : long.MaxValue - below);
                    }

                    Put(400, 0); Put(11, 1); Put(300, 2); Put(5, 3); Put(r, -1);
                }
            }
            else x = Bench.Registry.Make(type, xn, 97, -4, 4);

            var xv = Bench.Registry.Values(type, x, xn);
            if (name.StartsWith("scores a step apart", StringComparison.Ordinal))
            {
                // Each row's six largest scores planted, the fourth and fifth one step of the type apart (the same value where
                // the type has no step that small): the boundary of top-k (4), and of the nucleus (0.66 of the six: reached
                // at the fourth), lies between two weights the lane may hold as one; the scores' order as stored decides.
                double TypeRound(double v) => Bench.Registry.Values(type, Bench.Registry.Encode(type, [v]), 1)[0];
                double fifth = TypeRound(3.96), fourth = TypeRound(fifth + Math.Pow(2, Math.Floor(Math.Log2(fifth)) - mant));
                if (type == "float32")
                {
                    float f = 3.96f, t = (float)a.Temperature;
                    while (f / t != MathF.BitIncrement(f) / t) f = MathF.BitIncrement(f);
                    fifth = f; fourth = MathF.BitIncrement(f);
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

            if (name.StartsWith("-0 and +0", StringComparison.Ordinal))
            {
                // Each row's scores below −1 but four zeros, −0 at columns 3 and 50, +0 at 9 and 200: they rank the same, so
                // top-k of 2 keeps columns 3 and 9.
                for (int r = 0; r < a.Rows; r++)
                {
                    int at = r * a.Stride + a.RowOffset;
                    for (int c = 0; c < a.Cols; c++) xv[at + c] = -1 - Math.Abs(xv[at + c]);
                    xv[at + 3] = -0.0; xv[at + 50] = -0.0; xv[at + 9] = 0.0; xv[at + 200] = 0.0;
                }

                x = Bench.Registry.Encode(type, xv);
                xv = Bench.Registry.Values(type, x, xn);
            }

            if (infinities && a.Rows > 4)
            {
                // A row whose first-ranked is +∞ (twice), a row with a NaN among its scores, a row of −∞, a row with some −∞.
                xv[1 * a.Stride + a.RowOffset + 5] = xv[1 * a.Stride + a.RowOffset + 17] = double.PositiveInfinity;
                xv[2 * a.Stride + a.RowOffset + 4] = double.NaN;
                for (int c = 0; c < a.Cols; c++) xv[3 * a.Stride + a.RowOffset + c] = double.NegativeInfinity;
                for (int c = 0; c < a.Cols; c += 3) xv[4 * a.Stride + a.RowOffset + c] = double.NegativeInfinity;
                x = Bench.Registry.Encode(type, xv);
                xv = Bench.Registry.Values(type, x, xn);
            }

            var picks = Enumerable.Repeat(-9, a.Rows).ToArray();
            var probability = withProbability ? new byte[a.Rows * 8] : null;
            string lane = Bench.Registry.With(type, new SampleFilteredRun(ops, a, x, step, picks, probability));
            string? wrong = lane == laneName ? null : $"the lane {lane}, not {laneName}";
            var key = Draws.Key(a.Seed);
            for (int r = 0; r < a.Rows && wrong is null; r++)
            {
                int at = r * a.Stride + a.RowOffset;
                var row = xv.AsSpan(at, a.Cols);
                var order = wide64 ? StoredOrder(type, x, at, a.Cols) : SamplingArgs.Order(row);
                double? got = probability is null ? null : wideLane ? BitConverter.ToDouble(probability, r * 8) : BitConverter.ToSingle(probability, r * 4);
                wrong = SamplePlausible(a, row, order, Draws.Uniform(key, (uint)step, (uint)r), picks[r], got, laneEps, tiny);
                if (wrong is not null) wrong = $"row {r}: {wrong}";
            }

            if (wrong is not null) { Console.WriteLine($"check {what,-50} {where,-6} {type,-12} DIFFERS: {wrong}"); ok = false; }
        }

        return ok;
    }

    // A row's columns first to last by its 64-bit integers as stored (float64 holds several as one value).
    private static int[] StoredOrder(string type, byte[] x, int at, int cols)
    {
        var order = Enumerable.Range(0, cols).ToArray();
        if (type == "uint64")
        {
            var v = Enumerable.Range(0, cols).Select(c => BitConverter.ToUInt64(x, (at + c) * 8)).ToArray();
            Array.Sort(order, (i, j) => v[i] != v[j] ? v[j].CompareTo(v[i]) : i.CompareTo(j));
        }
        else
        {
            var v = Enumerable.Range(0, cols).Select(c => BitConverter.ToInt64(x, (at + c) * 8)).ToArray();
            Array.Sort(order, (i, j) => v[i] != v[j] ? v[j].CompareTo(v[i]) : i.CompareTo(j));
        }

        return order;
    }

    // Null when `pick` (and its probability `got`, where stored) is what the rule allows for a row x in its order and the
    // draw u within the lane's rounding; else what is wrong. The kept prefix: top-k's length exactly; the nucleus's where the
    // running sum of the order's weights reaches p of the top-k's sum within the lane's rounding of the sums (eps of the
    // sum); min-p's where a weight is below q within the lane's rounding of the weights (rho). For some length between, the
    // pick is a kept column whose interval of the kept weights' running sum in column order holds u times their sum within
    // eps of it (or it is the last kept column of a weight the lane holds), and its probability is its weight over the sum.
    private static string? SamplePlausible(SamplingArgs a, ReadOnlySpan<double> x, int[] order, double u, int pick, double? got, double laneEps, double tiny)
    {
        int n = x.Length;
        var (e, _) = a.Rule(x, order);
        if (!double.IsFinite(x[order[0]]))
        {
            if (pick != -1) return $"column {pick}, where the first-ranked score is {x[order[0]]:R}: none (-1)";
            return got is { } p0 && !double.IsNaN(p0) ? $"probability {p0:R} with no column (NaN)" : null;
        }

        double rho = 64 * laneEps, eps = (n + 16) * laneEps;
        int k = a.TopK, nucleusLow = k, nucleusHigh = k, minLow = k, minHigh = k;
        if (a.HasTopP)
        {
            double mass = 0, run = 0;
            for (int i = 0; i < k; i++) mass += e[order[i]];
            double target = a.TopP * mass, slack = 2 * eps * mass;
            for (int i = 0; i < k; i++)
            {
                run += e[order[i]];
                if (nucleusLow == k && run >= target - slack) nucleusLow = i + 1;
                if (nucleusHigh == k && run >= target + slack) nucleusHigh = i + 1;
            }
        }

        if (a.HasMinP)
            for (int i = 1; i < k; i++)
            {
                double w = e[order[i]];
                if (minLow == k && w < a.MinP * (1 + rho)) minLow = i;
                if (minHigh == k && w < a.MinP * (1 - rho)) { minHigh = i; break; }
            }

        int shortest = Math.Min(nucleusLow, minLow), longest = Math.Min(nucleusHigh, minHigh);
        if (pick < 0 || pick >= n) return $"column {pick}, where the row keeps {shortest} to {longest} columns";
        var rank = new int[n];
        for (int i = 0; i < n; i++) rank[order[i]] = i;
        if (rank[pick] >= longest) return $"column {pick} (rank {rank[pick]}) is not kept (the first {shortest} to {longest} are)";
        // The kept sums at the shortest prefix, then a column of the order more each length.
        double total = 0, before = 0;
        int after = 0;                                                         // kept columns past the pick of a weight the lane holds
        for (int i = 0; i < shortest; i++)
        {
            int c = order[i];
            total += e[c];
            if (c < pick) before += e[c];
            if (c > pick && e[c] > tiny) after++;
        }

        string? why = null;
        for (int length = shortest; length <= longest; length++)
        {
            if (length > shortest)
            {
                int c = order[length - 1];
                total += e[c];
                if (c < pick) before += e[c];
                if (c > pick && e[c] > tiny) after++;
            }

            if (rank[pick] >= length) { why ??= $"column {pick} (rank {rank[pick]}) is not among the first {length}"; continue; }
            double target = u * total, slack = eps * total;
            if (before - slack > target) { why = $"column {pick}: the kept columns before it ({before:R}) had passed the draw ({u:R} of {total:R})"; continue; }
            if (after > 0 && before + e[pick] + slack < target) { why = $"column {pick}: the running sum ({before + e[pick]:R}) has not passed the draw ({u:R} of {total:R})"; continue; }
            if (got is { } p)
            {
                double want = e[pick] / total;
                if (!(Math.Abs(p - want) <= (eps + 2 * rho) * want + 2 * laneEps)) { why = $"probability {p:R}, not {want:R}"; continue; }
            }

            return null;
        }

        return why;
    }

    private sealed class SampleFilteredRun(IDevice<ulong, ulong>? ops, SamplingArgs a, byte[] x, int step, int[] picks, byte[]? probability) : Bench.ITypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.SampleFiltered<TN>(a, x, [step], picks, probability);
            using var st = new Staged(ops);
            string lane = ops.SampleFiltered<TN>(a, st.In(x), st.Indices([step]), st.IndicesOut(picks), st.Out(probability));
            st.Collect();
            return lane;
        }
    }

    // bool holds truth values, not scores: the penalties and sampling refuse it.
    private static bool BoolRefused(IDevice<ulong, ulong>? ops, string where)
    {
        bool ok = true;
        var x = Bench.Registry.Make("bool", 64, 3, 0, 1);
        var calls = new (string Name, Action Call)[]
        {
            ("penalize", () => Bench.Registry.With("bool", new PenalizeRun(ops, new PenaltyArgs(2, 32, 4, 4, 1.1, 0, 0), x, new int[8], 1, new byte[x.Length]))),
            ("sample", () => Bench.Registry.With("bool", new SampleFilteredRun(ops, new SamplingArgs(2, 32, 1, 32, 1, 0, 1), x, 0, new int[2], null))),
        };
        foreach (var (name, call) in calls)
        {
            bool refused = false;
            try { call(); }
            catch (NotSupportedException) { refused = true; }
            if (!refused) { Console.WriteLine($"check {name} refuses bool {where,-6} FAILS: not refused"); ok = false; }
        }

        return ok;
    }

    // Routing.OrderKey, the devices' OrderKey, proven against Routing.Order on every type: each stored value's key as the
    // device makes it (Kernels / GpuOps / VulkanOps.OrderKeys), the values ordered by Order (a 64-bit integer's as stored),
    // and each two neighbours' keys compared with the sign of their Order. The values: every bit pattern of the types of 16
    // bits or fewer (NaNs of every payload and both signs, ±∞, ±0, subnormals), the wider ones' specials and 4096 random bit
    // patterns, a block format's 4096 values.
    private static bool OrderKeysCheck(IDevice<ulong, ulong>? ops, string where)
    {
        bool ok = true;
        var random = new Random(53);
        foreach (var type in Bench.Registry.Names)
        {
            int bits = Bench.Registry.Bits(type);
            byte[] x;
            int n;
            if (Bench.Registry.IsBlock(type)) { n = 4096; x = Bench.Registry.Make(type, n, 53, -4, 4); }
            else if (bits < 8) { n = 1 << bits; x = Bench.Registry.Encode(type, [.. Enumerable.Range(-(1 << (bits - 1)) - 1, n + 2).Select(v => (double)v)]); n += 2; }
            else if (bits <= 16) { n = 1 << bits; x = new byte[n * bits / 8]; for (int i = 0; i < n; i++) { x[i * bits / 8] = (byte)i; if (bits == 16) x[2 * i + 1] = (byte)(i >> 8); } }
            else
            {
                int size = bits / 8;
                var specials = bits == 32
                    ? new ulong[] { 0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF }
                    : new ulong[] { 0, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001, 0xFFF0000000000001, 0x7FFFFFFFFFFFFFFF, 1, 0x8000000000000001, 0x000FFFFFFFFFFFFF, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF, ulong.MaxValue };
                n = specials.Length + 4096;
                x = new byte[n * size];
                for (int i = 0; i < n; i++)
                {
                    ulong v = i < specials.Length ? specials[i] : (ulong)random.NextInt64() ^ ((ulong)random.Next(4) << 62);
                    for (int j = 0; j < size; j++) x[i * size + j] = (byte)(v >> (8 * j));
                }
            }

            var keys = new ulong[n];
            Bench.Registry.With(type, new OrderKeysRun(ops, x, keys, n));
            var values = Bench.Registry.Values(type, x, n);
            Comparison<int> order = type switch
            {
                "int64" => (i, j) => BitConverter.ToInt64(x, 8 * i).CompareTo(BitConverter.ToInt64(x, 8 * j)),
                "uint64" => (i, j) => BitConverter.ToUInt64(x, 8 * i).CompareTo(BitConverter.ToUInt64(x, 8 * j)),
                _ => (i, j) => Routing.Order(values[i], values[j]),
            };
            var sorted = Enumerable.Range(0, n).ToArray();
            Array.Sort(sorted, order);
            for (int t = 1; t < n; t++)
            {
                int i = sorted[t - 1], j = sorted[t];
                if (Math.Sign(keys[i].CompareTo(keys[j])) == Math.Sign(order(i, j))) continue;
                Console.WriteLine($"check order keys {where,-6} {type,-12} DIFFERS: {values[i]:R} (key {keys[i]:X}) and {values[j]:R} (key {keys[j]:X}) against their order {order(i, j)}");
                ok = false;
                break;
            }
        }

        if (ok) Console.WriteLine($"check order keys             {where,-6} every type: each stored value's order key (Routing.OrderKey, the device's OrderKey) orders as Routing.Order (every bit pattern to 16 bits: NaNs of every payload and sign, the infinities, both zeros, subnormals; the wider types' specials and random patterns; 64-bit integers as stored)");
        return ok;
    }

    private sealed class OrderKeysRun(IDevice<ulong, ulong>? ops, byte[] x, ulong[] keys, int n) : Bench.ITypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) { Adad.Cpu.Kernels.OrderKeys<TN>(x, keys, n); return ""; }
            var bytes = new byte[n * 8];
            using (var st = new Staged(ops))
            {
                ulong dx = st.In(x), dk = st.Out(bytes);
                if (ops is GpuOps cuda) cuda.OrderKeys<TN>(dx, dk, n);
                else ((Adad.Vulkan.VulkanOps)ops).OrderKeys<TN>(dx, dk, n);
                st.Collect();
            }

            Buffer.BlockCopy(bytes, 0, keys, 0, bytes.Length);
            return "";
        }
    }
}
