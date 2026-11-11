// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Decoding's operations on every device, every float type (the block formats among them):
//   log_softmax_topk   the top-k of each row as log-probabilities (IRouting.LogSoftmaxTopK): the indices TopK's exactly
//                      (largest first, an equal value by the lower index), each value within the type's step plus the
//                      lane's rounding of the row's log-sum-exp of the float64 reference; rows wider than a block of
//                      threads; and k past the row (the values there −∞ as the type stores it, a block format's lowest,
//                      the indices −1).
//   history, penalize  the ring and the penalties of its window (Checks.Penalties.cs).
namespace Qiyas;

internal static partial class Checks
{
    public static bool DecodingCheck(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        ok &= HistoryRings(ops, where);
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            ok &= LogSoftmaxTopKs(ops, where, type);
            ok &= Penalties(ops, where, type);
        }

        if (ok) Console.WriteLine($"check decoding               {where,-6} every float type{(blocks ? "" : " but the block formats")}: the top-k of each row as log-probabilities (TopK's indices exactly, the values within the lane's rounding of the row's log-sum-exp, k past the row); each row's ring of tokens pushed and advanced on the device, and the penalties of its window (repeat, presence, frequency: the last tokens, a wrapped ring, tokens outside the row, rows a stride apart) within the lane's rounding, every other score x's own");
        return ok;
    }

    private static bool LogSoftmaxTopKs(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        double laneEps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23);
        foreach (var (rows, cols, k) in new[] { (7, 53, 5), (3, 601, 16), (37, 29, 1), (2, 4, 6) })
        {
            string what = $"log_softmax_topk {rows}x{cols} k {k}";
            var x = Bench.Registry.Make(type, rows * cols, 91, -4, 4);
            var (want, wantAt) = Routing.LogSoftmaxTopKReference(Bench.Registry.Values(type, x, rows * cols), rows, cols, k);
            var values = new byte[Bench.Registry.Bytes(type, rows * k)];
            var indices = new int[rows * k];
            Bench.Registry.WithFloat(type, new LogTopKRun(ops, x, rows, cols, k, values, indices));
            string? wrong = null;
            for (int i = 0; i < rows * k && wrong is null; i++) if (indices[i] != wantAt[i]) wrong = $"index [{i}] {indices[i]}, not {wantAt[i]}";
            if (wrong is null && k <= cols)
            {
                // The values' error: the row's largest (up to 4) and the log of its sum, each a few lane roundings.
                double bad = Excess(type, what, values, want, _ => 8, 16);
                if (bad > 0) wrong = $"values DIFFER by {bad:0.##}";
            }
            else if (wrong is null)
            {
                // k past the row: the row's own values as above; past it −∞ as the type stores it (a block format's lowest).
                var got = Bench.Registry.Values(type, values, rows * k);
                double none = Bench.Registry.IsBlock(type) ? Bench.Registry.Lowest(type) : Bench.Registry.Values(type, Bench.Registry.Encode(type, [double.NegativeInfinity]), 1)[0];
                int mant = Bench.Registry.Mantissa(type);
                for (int i = 0; i < rows * k && wrong is null; i++)
                {
                    if (wantAt[i] < 0)
                    {
                        if (Bench.Registry.IsBlock(type) ? Math.Abs(got[i] - none) > Math.Abs(none) * Math.Pow(2, 1 - mant) : !got[i].Equals(none)) wrong = $"value [{i}] past the row {got[i]:R}, not {none:R}";
                        continue;
                    }

                    if (Bench.Registry.IsBlock(type)) continue;                              // beside the lowest value its block's step swallows a log-probability
                    double step = Math.Max(Bench.Registry.MinStep(type), Math.Pow(2, Math.Floor(Math.Log2(Math.Abs(want[i]))) - mant));
                    if (!(Math.Abs(got[i] - want[i]) <= step + 16 * laneEps * 8)) wrong = $"value [{i}] {got[i]:R}, not {want[i]:R}";
                }
            }

            if (wrong is not null) { Console.WriteLine($"check {what,-40} {where,-6} {type,-12} {wrong}"); ok = false; }
        }

        return ok;
    }

    private sealed class LogTopKRun(IDevice<ulong, ulong>? ops, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.LogSoftmaxTopK<TN>(x, rows, cols, k, values, indices);
            using var st = new Staged(ops);
            string lane = ops.LogSoftmaxTopK<TN>(st.In(x), rows, cols, k, st.Out(values), st.IndicesOut(indices));
            st.Collect();
            return lane;
        }
    }
}
