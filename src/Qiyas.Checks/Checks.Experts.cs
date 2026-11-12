// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts (Experts.cs) on every device against its float64 reference, every float type (the block formats
// among them): the routing (each token's experts in TopK's order with ties among the probabilities, and with NaN and
// signed zeros, normalized and not: offsets, rows and slots exactly, the weights within the sum's and the quotient's
// rounding, NaN where the reference's is), the experts' products and the gated pair
// over uneven groups (an expert without rows, rows outside every expert's), the weights stored and transposed, with and
// without biases; within the lane's rounding of each value's terms (through the gate) plus the type's step.
namespace Qiyas;

internal static partial class Checks
{
    public static bool ExpertsCheck(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            ok &= RouteTopKCheck(ops, where, type);
            ok &= ExpertProductsCheck(ops, where, type);
        }

        if (ok) Console.WriteLine($"check experts                {where,-6} every float type{(blocks ? "" : " but the block formats")}: the top-k routing in TopK's order with ties, NaN and ±0 (normalized and not) exactly, and the experts' products and gated pairs over uneven groups (an empty expert, rows outside every expert), weights stored and transposed, with and without biases, within the products' rounding");
        return ok;
    }

    private static bool RouteTopKCheck(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        const int tokens = 37, experts = 9, k = 3;
        var drawn = Bench.Registry.Values(type, Bench.Registry.Make(type, tokens * experts, 101, 0, 1), tokens * experts);
        for (int t = 0; t < tokens; t += 4) { drawn[t * experts + 5] = drawn[t * experts + 2]; drawn[t * experts + 7] = drawn[t * experts + 2]; }   // ties
        // TopK's order past the numbers: a NaN above every number (one in a token, two, more than k), NaN beside NaN and −0
        // beside +0 by the lower index.
        var odd = (double[])drawn.Clone();
        odd[1 * experts + 4] = double.NaN;
        odd[2 * experts + 6] = odd[2 * experts + 1] = double.NaN;
        for (int e = 0; e < 4; e++) odd[3 * experts + e] = double.NaN;
        for (int e = 0; e < experts; e++) odd[5 * experts + e] = e % 2 == 0 ? -0.0 : 0.0;
        foreach (var (input, special) in new[] { (drawn, false), (odd, true) })
        {
            var stored = Bench.Registry.Encode(type, input);
            var p = Bench.Registry.Values(type, stored, tokens * experts);
            foreach (bool normalize in new[] { false, true })
            {
                var (want, _, wantOffsets, wantTokens, wantSlots) = Experts.RouteTopKReference(p, tokens, experts, k, normalize);
                var weights = new byte[Bench.Registry.Bytes(type, tokens * k)];
                int[] offsets = new int[experts + 1], tokenOfRow = new int[tokens * k], slotRow = new int[tokens * k];
                Bench.Registry.WithFloat(type, new RouteTopKRun(ops, stored, tokens, experts, k, normalize, weights, offsets, tokenOfRow, slotRow));
                string what = $"route top-{k} of {experts}{(special ? " NaN ±0" : "")}{(normalize ? " normalized" : "")}";
                if (!offsets.SequenceEqual(wantOffsets) || !tokenOfRow.SequenceEqual(wantTokens) || !slotRow.SequenceEqual(wantSlots))
                {
                    Console.WriteLine($"check {what,-50} {where,-6} {type,-12} DIFFERS in its rows");
                    ok = false;
                }

                // The weights: NaN exactly where the reference's are, the others within the rounding. A block format's
                // weights share a block's scale with its neighbours, a NaN among them the codec's own matter (its checks), so
                // with NaN in the input a block format is held to its rows alone.
                if (special && Bench.Registry.IsBlock(type)) continue;
                var got = Bench.Registry.Values(type, weights, tokens * k);
                if (Enumerable.Range(0, tokens * k).Any(i => double.IsNaN(got[i]) != double.IsNaN(want[i])))
                {
                    Console.WriteLine($"check {what,-50} {where,-6} {type,-12} DIFFERS in its NaN weights");
                    ok = false;
                }

                var wantNumbers = want.Select(w => double.IsNaN(w) ? 0 : w).ToArray();
                var gotNumbers = special ? Bench.Registry.Encode(type, got.Select(w => double.IsNaN(w) ? 0 : w).ToArray()) : weights;
                double bad = Excess(type, what, gotNumbers, wantNumbers, i => normalize ? (k + 2) * Math.Abs(wantNumbers[i]) : 0, 1);
                if (bad > 0) { Console.WriteLine($"check {what,-50} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
            }
        }

        return ok;
    }

    private static readonly (string Name, bool TransW, string? Gate, bool Biases, int[] Offsets)[] ExpertCases =
    [
        ("products, bias", false, null, true, [0, 9, 9, 23, 30, 41]),
        ("products W transposed, rows outside", true, null, false, [2, 9, 9, 23, 30, 38]),
        ("gated silu, biases", false, "gated_silu", true, [0, 9, 9, 23, 30, 41]),
        ("gated gelu W transposed, rows outside", true, "gated_gelu", false, [2, 9, 9, 23, 30, 38]),
    ];

    private static bool ExpertProductsCheck(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        int seed = 110;
        foreach (var (name, transW, gate, biases, offsets) in ExpertCases)
        {
            var v = new ExpertView(5, 41, 33, 27, transW);
            byte[] Draw(long n) => Bench.Registry.Make(type, (int)n, seed++, -1, 1);
            double[] Read(byte[]? b, long n) => b is null ? [] : Bench.Registry.Values(type, b, (int)n);
            byte[] x = Draw(v.XValues), gw = Draw(v.WeightValues), uw = Draw(v.WeightValues);
            byte[]? gb = biases ? Draw(v.BiasValues) : null, ub = biases ? Draw(v.BiasValues) : null;
            double[] xv = Read(x, v.XValues), gwv = Read(gw, v.WeightValues), uwv = Read(uw, v.WeightValues);
            double[]? gbv = gb is null ? null : Read(gb, v.BiasValues), ubv = ub is null ? null : Read(ub, v.BiasValues);
            double[] Abs(double[]? a) => a is null ? [] : a.Select(Math.Abs).ToArray();
            var g = v.Reference(xv, offsets, gwv, gbv);
            var sg = v.Reference(Abs(xv), offsets, Abs(gwv), gbv is null ? null : Abs(gbv));
            double[] want, scale;
            int depth = v.K + 4;
            if (gate is null)
            {
                want = g;
                scale = sg.Select(s => depth * s).ToArray();
            }
            else
            {
                var op = Ops.Find(gate);
                var u = v.Reference(xv, offsets, uwv, ubv);
                var su = v.Reference(Abs(xv), offsets, Abs(uwv), ubv is null ? null : Abs(ubv));
                want = new double[g.Length];
                scale = new double[g.Length];
                for (int i = 0; i < g.Length; i++)
                {
                    want[i] = op.Reference(g[i], u[i], 0, 0, 0);
                    double act = op.Reference(g[i], 1, 0, 0, 0);
                    scale[i] = depth * (1.25 * sg[i] * Math.Abs(u[i]) + Math.Abs(act) * su[i]) + (op.Ulps + 2) * Math.Abs(want[i]);
                }
            }

            var y = new byte[Bench.Registry.Bytes(type, v.YValues)];
            Bench.Registry.WithFloat(type, new ExpertRunCase(ops, v, gate, x, offsets, gw, gb, uw, ub, y));
            double bad = Excess(type, $"experts {name}", y, want, i => scale[i], 1);
            if (bad > 0) { Console.WriteLine($"check experts {name,-44} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
        }

        return ok;
    }

    private sealed class RouteTopKRun(IDevice<ulong, ulong>? ops, byte[] p, int tokens, int experts, int k, bool normalize, byte[] weights, int[] offsets, int[] tokenOfRow, int[] slotRow)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.RouteTopK<TN>(p, tokens, experts, k, normalize, weights, offsets, tokenOfRow, slotRow);
            using var st = new Staged(ops);
            var lane = ops.RouteTopK<TN>(st.In(p), tokens, experts, k, normalize, st.Out(weights), st.IndicesOut(offsets), st.IndicesOut(tokenOfRow), st.IndicesOut(slotRow));
            st.Collect();
            return lane;
        }
    }

    private sealed class ExpertRunCase(IDevice<ulong, ulong>? ops, ExpertView v, string? gate, byte[] x, int[] offsets, byte[] gw, byte[]? gb, byte[] uw, byte[]? ub, byte[] y)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null)
                return gate is null
                    ? CpuDevice.Device.ExpertMatMul<TN>(v, x, offsets, gw, gb, y)
                    : CpuDevice.Device.ExpertGatedPair<TN>(v, gate, x, offsets, gw, gb, uw, ub, y);
            using var st = new Staged(ops);
            string lane = gate is null
                ? ops.ExpertMatMul<TN>(v, st.In(x), st.Indices(offsets), st.In(gw), st.In(gb), st.Out(y))
                : ops.ExpertGatedPair<TN>(v, gate, st.In(x), st.Indices(offsets), st.In(gw), st.In(gb), st.In(uw), st.In(ub), st.Out(y));
            st.Collect();
            return lane;
        }
    }
}
