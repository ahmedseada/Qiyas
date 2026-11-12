// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Decoding's operations on every device, every float type (the block formats among them):
//   log_softmax_topk   the top-k of each row as log-probabilities (IRouting.LogSoftmaxTopK): the indices TopK's exactly
//                      (Adad's order: larger first, an equal value by the lower index), each value within the type's step plus the
//                      lane's rounding of the row's log-sum-exp of the float64 reference; rows wider than a block of
//                      threads; k the whole row; and k past the row refused.
//   history, penalize  the ring and the penalties of its window, every type but bool (Checks.Penalties.cs).
//   sample             the draw from the kept columns, every type but bool (Checks.Sampling.cs).
//   bool               refused by the penalties and sampling (truth values, not scores), and by dropout.
//   order keys         Routing.OrderKey and the devices' OrderKey against Routing.Order, every type (Checks.Sampling.cs).
namespace Qiyas;

internal static partial class Checks
{
    public static bool DecodingCheck(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        ok &= HistoryRings(ops, where);
        ok &= OrderKeysCheck(ops, where);
        foreach (var type in Bench.Registry.Names)
        {
            if (!blocks && Bench.Registry.IsBlock(type)) continue;
            if (Bench.Registry.Bits(type) == 1) { ok &= BoolRefused(ops, where); continue; }
            ok &= Penalties(ops, where, type);
            ok &= Samplings(ops, where, type);
            if (Bench.Registry.IsFloat(type)) ok &= LogSoftmaxTopKs(ops, where, type);
        }

        if (ok) Console.WriteLine($"check decoding               {where,-6} every type{(blocks ? "" : " but the block formats")}: the top-k of each row as log-probabilities over the float types (TopK's indices exactly, the values within the lane's rounding of the row's log-sum-exp, k the whole row, k past it refused); each row's ring of tokens pushed and advanced on the device, and the penalties of its window over every type but bool (repeat, presence, frequency: the last tokens, a wrapped ring, tokens outside the row, rows a stride apart; a window past the ring refused) within the float lane's rounding, every other score x's own; a column of each row drawn from its scores over every type but bool at a temperature, held to top-k, the nucleus and min-p (the order of the scores as stored; rows of +inf, with a NaN and of -inf drawing none, scores a step of the type apart at a boundary, 64-bit integers past 2^53; the seed's draw at the device's step), each within the lane's rounding of the float64 rule; bool refused");
        return ok;
    }

    private static bool LogSoftmaxTopKs(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        foreach (var (rows, cols, k) in new[] { (7, 53, 5), (3, 601, 16), (37, 29, 1), (2, 4, 4) })
        {
            string what = $"log_softmax_topk {rows}x{cols} k {k}";
            var x = Bench.Registry.Make(type, rows * cols, 91, -4, 4);
            var (want, wantAt) = Routing.LogSoftmaxTopKReference(Bench.Registry.Values(type, x, rows * cols), rows, cols, k);
            var values = new byte[Bench.Registry.Bytes(type, rows * k)];
            var indices = new int[rows * k];
            Bench.Registry.WithFloat(type, new LogTopKRun(ops, x, rows, cols, k, values, indices));
            string? wrong = null;
            for (int i = 0; i < rows * k && wrong is null; i++) if (indices[i] != wantAt[i]) wrong = $"index [{i}] {indices[i]}, not {wantAt[i]}";
            if (wrong is null)
            {
                // The values' error: the row's largest (up to 4) and the log of its sum, each a few lane roundings.
                double bad = Excess(type, what, values, want, _ => 8, 16);
                if (bad > 0) wrong = $"values by {bad:0.##}";
            }

            if (wrong is not null) { Console.WriteLine($"check {what,-40} {where,-6} {type,-12} DIFFERS: {wrong}"); ok = false; }
        }

        // k past the row: refused (the k largest of fewer values are not defined).
        {
            var x = Bench.Registry.Make(type, 2 * 4, 91, -4, 4);
            bool refused = false;
            try { Bench.Registry.WithFloat(type, new LogTopKRun(ops, x, 2, 4, 6, new byte[Bench.Registry.Bytes(type, 12)], new int[12])); }
            catch (ArgumentException) { refused = true; }
            if (!refused) { Console.WriteLine($"check log_softmax_topk past the row {where,-6} {type,-12} FAILS: k = 6 of 4 columns was not refused"); ok = false; }
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
