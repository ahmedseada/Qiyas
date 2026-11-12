// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention over tensors laid through views (AttentionArgs.QRows, KRows, VRows, ORows: HeadRows, a strided view of a head's
// rows), on every device, every float type (the block formats among them): every form over Q, K, V and O a step at a time
// (in batches, grouped heads, a window), and over Q, K and V read at their offsets in one projection's rows; the backward
// over dense views; attention over a cache with Q and O a step at a time. Each against the reference of the tensors as the
// strided copies store them (a block format's rows that share blocks re-encoded by its blocks).
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
            ok &= LaidAttentions(ops, where, type);
        }

        if (ok) Console.WriteLine($"check heads                  {where,-6} every float type{(blocks ? "" : " but the block formats")}: attention over tensors through views (a step at a time in batches, Q, K and V at their offsets in one projection's rows) by every form, the backward and a cache");
        return ok;
    }

    // Views of a step-first layer ([batches, rows, heads, d]) for every tensor.
    private static AttentionArgs StepsFirst(AttentionArgs a, int batches) => a with
    {
        QRows = HeadRows.StepsFirst(a.Heads, a.Lq, a.D, batches), ORows = HeadRows.StepsFirst(a.Heads, a.Lq, a.D, batches),
        KRows = HeadRows.StepsFirst(a.KvHeadCount, a.Lk, a.D, batches), VRows = HeadRows.StepsFirst(a.KvHeadCount, a.Lk, a.D, batches),
    };

    private static bool LaidAttentions(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        // Q, K and V read where one projection's rows hold them: row s of [s, (hq + 2·hk)·d], Q's heads, then K's, then V's.
        var fusedA = new AttentionArgs(4, 6, 6, 32, 0.3, true) { KvHeads = 2 };
        long fusedRow = (long)(fusedA.Heads + 2 * fusedA.KvHeadCount) * fusedA.D;
        fusedA = fusedA with
        {
            QRows = new HeadRows(fusedA.Heads, fusedA.D, fusedRow),
            KRows = new HeadRows(fusedA.KvHeadCount, fusedA.D, fusedRow, 0, (long)fusedA.Heads * fusedA.D),
            VRows = new HeadRows(fusedA.KvHeadCount, fusedA.D, fusedRow, 0, (long)(fusedA.Heads + fusedA.KvHeadCount) * fusedA.D),
            ORows = HeadRows.StepsFirst(fusedA.Heads, fusedA.Lq, fusedA.D),
        };
        var cases = new (string Name, AttentionArgs A, bool Fused)[]
        {
            ("causal, 2 batches", StepsFirst(new AttentionArgs(4, 5, 9, 8, 0.3, true), 2), false),
            ("grouped, 2 batches", StepsFirst(new AttentionArgs(8, 6, 6, 16, 0.3, true) { KvHeads = 4 }, 2), false),
            ("window", StepsFirst(new AttentionArgs(3, 7, 11, 8, 0.3, false) { Window = 4 }, 1), false),
            ("rows of whole blocks", StepsFirst(new AttentionArgs(4, 3, 5, 32, 0.3, true), 2), false),
            ("one projection's rows", fusedA, true),
        };
        int unit = type switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 1 };
        // A tensor as the strided copy stores it a head at a time: the type's values as they are where its rows, strides and
        // offset are whole storage units; a block format whose rows share blocks re-encoded by its blocks.
        bool Whole(HeadRows? v, int d) => d % unit == 0 && (v is null || (v.HeadStride % unit == 0 && v.RowStride % unit == 0 && v.BatchStride % unit == 0 && v.Offset % unit == 0));
        double[] Stored(double[] headsFirst, HeadRows? v, int d) => Whole(v, d) ? headsFirst : Bench.Registry.Values(type, Bench.Registry.Encode(type, headsFirst), headsFirst.Length);
        foreach (var (name, a, fused) in cases)
        {
            int hk = a.KvHeadCount, nq = (int)a.QValues, nk = (int)a.KValues;
            var h = a.HeadMajor;
            byte[] q, k, v;
            double[] qh, kh, vh;
            if (fused)
            {
                int n = (int)(a.Lq * fusedRow);
                q = k = v = Bench.Registry.Make(type, n, 41, -1, 1);
                var all = Bench.Registry.Values(type, q, n);
                qh = Stored(AttentionLayout.ToHeads(all, a.QRows, a.Heads, a.Lq, a.D), a.QRows, a.D);
                kh = Stored(AttentionLayout.ToHeads(all, a.KRows, hk, a.Lk, a.D), a.KRows, a.D);
                vh = Stored(AttentionLayout.ToHeads(all, a.VRows, hk, a.Lk, a.D), a.VRows, a.D);
            }
            else
            {
                q = Bench.Registry.Make(type, nq, 41, -1, 1); k = Bench.Registry.Make(type, nk, 42, -1, 1); v = Bench.Registry.Make(type, nk, 43, -1, 1);
                qh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, q, nq), a.QRows, a.Heads, a.Lq, a.D), a.QRows, a.D);
                kh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, k, nk), a.KRows, hk, a.Lk, a.D), a.KRows, a.D);
                vh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, v, nk), a.VRows, hk, a.Lk, a.D), a.VRows, a.D);
            }

            var want = h.Reference(qh, kh, vh);
            int terms = a.Lk + a.D;
            double[] Queries(double[] t) => AttentionLayout.FromHeads(t, a.ORows, a.Heads, a.Lq, a.D);
            foreach (string form in new[] { "measured", "fused", "composed", "decode" })
            {
                var o = new byte[Bench.Registry.Bytes(type, nq)];
                Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, o, form == "decode", null, null, null, fused: form == "fused", composed: form == "composed"));
                ok &= CloseMoved(type, $"attention laid {name} {form}", where, o, want, Queries, terms);
            }

            if (fused) continue;                                            // the backward writes dQ, dK and dV as Q, K and V: dense views
            var dO = Bench.Registry.Make(type, nq, 44, -1, 1);
            var gh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, dO, nq), a.ORows, a.Heads, a.Lq, a.D), a.ORows, a.D);
            var (wq, wk, wv) = h.ReferenceBackward(qh, kh, vh, gh);
            byte[] dq = new byte[q.Length], dk = new byte[k.Length], dv = new byte[v.Length];
            Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, dO, false, dq, dk, dv));
            int rows = a.Group * a.Lq;
            ok &= CloseMoved(type, $"attention_bwd laid dQ {name}", where, dq, wq, Queries, 2 * (a.Lk + a.D));
            ok &= CloseMoved(type, $"attention_bwd laid dK {name}", where, dk, wk, t => AttentionLayout.FromHeads(t, a.KRows, hk, a.Lk, a.D), 2 * (rows + a.D));
            ok &= CloseMoved(type, $"attention_bwd laid dV {name}", where, dv, wv, t => AttentionLayout.FromHeads(t, a.VRows, hk, a.Lk, a.D), rows + a.D);
        }

        // Over a cache: Q and O a step at a time in two batches, K and V the cache's.
        {
            var a0 = new AttentionArgs(4, 3, 24, 16, 0.3, true) { KvHeads = 2 };
            var a = a0 with { QRows = HeadRows.StepsFirst(a0.Heads, a0.Lq, a0.D, 2), ORows = HeadRows.StepsFirst(a0.Heads, a0.Lq, a0.D, 2) };
            string cacheType = Bench.Registry.IsBlock(type) ? "float32" : type;
            int nq = (int)a.QValues, nk = (int)a.KValues, p = 15;
            byte[] q = Bench.Registry.Make(type, nq, 45, -1, 1), k = Bench.Registry.Make(cacheType, nk, 46, -1, 1), v = Bench.Registry.Make(cacheType, nk, 47, -1, 1);
            double[] qh = Stored(AttentionLayout.ToHeads(Bench.Registry.Values(type, q, nq), a.QRows, a.Heads, a.Lq, a.D), a.QRows, a.D);
            var wantH = CacheRule.AtPosition(a0, p).Reference(qh, Bench.Registry.Values(cacheType, k, nk), Bench.Registry.Values(cacheType, v, nk));
            var o = new byte[q.Length];
            Bench.Registry.WithFloat(type, new CacheAttentionRun(ops, cacheType, a, p, q, k, v, o));
            ok &= CloseMoved(type, "attention_cached laid", where, o, wantH, t => AttentionLayout.FromHeads(t, a.ORows, a.Heads, a.Lq, a.D), a.Lk + a.D);
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
