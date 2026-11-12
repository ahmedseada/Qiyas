// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention over the layouts a layer's projections come in, on every device, every float type (the block formats among
// them): every form over Q, K, V and O a step at a time (AttentionArgs.StepsFirst, in batches, grouped heads), the
// backward, and attention over a cache, against the reference of the tensors as the strided copies store them (a block
// format's: re-encoded by its blocks).
namespace Qiyas;

internal static partial class Checks
{
    public static bool HeadsCheck(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            ok &= StepsFirstAttentions(ops, where, type);
        }

        if (ok) Console.WriteLine($"check heads                  {where,-6} every float type{(blocks ? "" : " but the block formats")}: attention over tensors a step at a time by every form, the backward and a cache");
        return ok;
    }

    private static bool StepsFirstAttentions(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        var cases = new (string Name, AttentionArgs A)[]
        {
            ("causal, 2 batches", new AttentionArgs(4, 5, 9, 8, 0.3, true) { Batch = 2, StepsFirst = true }),
            ("grouped, 2 batches", new AttentionArgs(8, 6, 6, 16, 0.3, true) { Batch = 2, KvHeads = 4, StepsFirst = true }),
            ("window", new AttentionArgs(3, 7, 11, 8, 0.3, false) { Window = 4, StepsFirst = true }),
            ("rows of whole blocks", new AttentionArgs(4, 3, 5, 32, 0.3, true) { Batch = 2, StepsFirst = true }),
        };
        int unit = type switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 1 };
        int d0 = 0;
        // A tensor as the strided copy stores it a head at a time: the type's values as they are where a row is whole storage
        // units; a block format whose rows share blocks re-encoded by its blocks.
        double[] Stored(double[] headsFirst) => d0 % unit == 0 ? headsFirst : Bench.Registry.Values(type, Bench.Registry.Encode(type, headsFirst), headsFirst.Length);
        foreach (var (name, a) in cases)
        {
            d0 = a.D;
            int b = a.BatchCount, hq = a.Heads / b, hk = a.KvHeadCount / b, nq = (int)a.QValues, nk = (int)a.KValues;
            var h = a with { StepsFirst = false };
            byte[] q = Bench.Registry.Make(type, nq, 41, -1, 1), k = Bench.Registry.Make(type, nk, 42, -1, 1), v = Bench.Registry.Make(type, nk, 43, -1, 1), dO = Bench.Registry.Make(type, nq, 44, -1, 1);
            double[] qh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, q, nq), b, a.Lq, hq, a.D)), kh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, k, nk), b, a.Lk, hk, a.D)),
                vh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, v, nk), b, a.Lk, hk, a.D)), gh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, dO, nq), b, a.Lq, hq, a.D));
            var want = h.Reference(qh, kh, vh);
            int terms = a.Lk + a.D;
            double[] Queries(double[] t) => AttentionLayout.FromHeads(t, b, a.Lq, hq, a.D);
            double[] Keys(double[] t) => AttentionLayout.FromHeads(t, b, a.Lk, hk, a.D);
            foreach (string form in new[] { "measured", "fused", "composed", "decode" })
            {
                var o = new byte[q.Length];
                Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, o, form == "decode", null, null, null, fused: form == "fused", composed: form == "composed"));
                ok &= CloseMoved(type, $"attention steps first {name} {form}", where, o, want, Queries, terms);
            }

            var (wq, wk, wv) = h.ReferenceBackward(qh, kh, vh, gh);
            byte[] dq = new byte[q.Length], dk = new byte[k.Length], dv = new byte[v.Length];
            Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, dO, false, dq, dk, dv));
            int rows = a.Group * a.Lq;
            ok &= CloseMoved(type, $"attention_bwd steps first dQ {name}", where, dq, wq, Queries, 2 * (a.Lk + a.D));
            ok &= CloseMoved(type, $"attention_bwd steps first dK {name}", where, dk, wk, Keys, 2 * (rows + a.D));
            ok &= CloseMoved(type, $"attention_bwd steps first dV {name}", where, dv, wv, Keys, rows + a.D);
        }

        // Over a cache: Q and O a step at a time, K and V the cache's.
        {
            var a = new AttentionArgs(4, 3, 24, 16, 0.3, true) { Batch = 2, KvHeads = 2, StepsFirst = true };
            d0 = a.D;
            string cacheType = Bench.Registry.IsBlock(type) ? "float32" : type;
            int b = a.BatchCount, hq = a.Heads / b, nq = (int)a.QValues, nk = (int)a.KValues, p = 15;
            byte[] q = Bench.Registry.Make(type, nq, 45, -1, 1), k = Bench.Registry.Make(cacheType, nk, 46, -1, 1), v = Bench.Registry.Make(cacheType, nk, 47, -1, 1);
            double[] qh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, q, nq), b, a.Lq, hq, a.D));
            var wantH = CacheRule.AtPosition(a with { StepsFirst = false }, p).Reference(qh, Bench.Registry.Values(cacheType, k, nk), Bench.Registry.Values(cacheType, v, nk));
            var o = new byte[q.Length];
            Bench.Registry.WithFloat(type, new CacheAttentionRun(ops, cacheType, a, p, q, k, v, o));
            ok &= CloseMoved(type, "attention_cached steps first", where, o, wantH, t => AttentionLayout.FromHeads(t, b, a.Lq, hq, a.D), a.Lk + a.D);
        }

        return ok;
    }

    // An output a form stores a head at a time and a strided copy then moves a step at a time, against the reference a head
    // at a time. A type of whole values: the reference as the type stores it, moved (Close). A block format is encoded twice,
    // by the form in the blocks of the one order and by the copy in the blocks of the other: each value within two steps of
    // its block in each (a block's step set by its largest value), and the lane's rounding over the terms.
    private static bool CloseMoved(string type, string what, string where, byte[] output, double[] wantHeads, Func<double[], double[]> move, int terms)
    {
        int n = wantHeads.Length;
        if (!Bench.Registry.IsBlock(type))
            return Close(type, what, where, output, move(Bench.Registry.Values(type, Bench.Registry.Encode(type, wantHeads), n)), n, terms);
        int blockValues = type is "q8_0" or "q4_0" ? 32 : 256;
        double unit = Math.Pow(2, -Bench.Registry.Mantissa(type)), laneEps = Math.Pow(2, -23);
        double[] Steps(double[] values)
        {
            var steps = new double[n];
            for (int at = 0; at < n; at += blockValues)
            {
                int end = Math.Min(n, at + blockValues);
                double max = 0;
                for (int i = at; i < end; i++) max = Math.Max(max, Math.Abs(values[i]));
                for (int i = at; i < end; i++) steps[i] = (max == 0 ? 1 : max) * unit;
            }

            return steps;
        }

        double[] want = move(wantHeads), first = move(Steps(wantHeads)), second = Steps(want), got = Bench.Registry.Values(type, output, n);
        double worst = 0;
        for (int i = 0; i < n; i++)
        {
            double tol = 2 * first[i] + 2 * second[i] + (4 * Math.Abs(want[i]) + 4 * terms) * laneEps, err = Math.Abs(got[i] - want[i]);
            if (double.IsNaN(err) || err > tol) worst = Math.Max(worst, double.IsNaN(err) ? double.PositiveInfinity : err / second[i]);
        }

        if (worst > 0) Console.WriteLine($"check {what,-12} {where,-6} {type,-12} DIFFERS by {worst:0.##} steps");
        return worst == 0;
    }
}
