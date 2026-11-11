// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A layer's projections into the layouts attention reads, on every device, every float type (the block formats among them):
//   rope, heads first     IRows.Rope with its output's rows a head at a time (RopeArgs.HeadsFirst): half and interleaved
//                         pairs, after an RMS normalization, and the layout alone, against the float64 reference.
//   cache_write_rope      ICache.CacheWriteRope: the rows normalized and rotated in the lane and written to the cache at the
//                         position with one rounding — a cache of the type itself (a block format's: float32) against the
//                         reference's values, and int8 codes with row scales: the scales against the reference's, each
//                         code within half a scale of its value (a code at the range's end past it), the rows outside the
//                         write as they were.
//   attention, steps first  every form over Q, K, V and O a step at a time (AttentionArgs.StepsFirst, in batches, grouped
//                         heads), the backward, and attention over a cache, against the reference of the tensors as the
//                         strided copies store them (a block format's: re-encoded by its blocks).
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
            ok &= RopeHeads(ops, where, type);
            ok &= CacheRopes(ops, where, type);
            ok &= StepsFirstAttentions(ops, where, type);
        }

        if (ok) Console.WriteLine($"check heads                  {where,-6} every float type{(blocks ? "" : " but the block formats")}: the rotary embedding with its rows a head at a time, a layer's rows normalized, rotated and written to a cache with one rounding (the type itself, int8 with row scales), and attention over tensors a step at a time by every form, the backward and a cache");
        return ok;
    }

    // The rotation's tables (lane values) and the steps' positions of a case.
    private static (double[] Cos, double[] Sin, int[] Positions) RopeTables(string lane, int steps, int half)
    {
        const int MaxPos = 64;
        var cos = new double[MaxPos * Math.Max(1, half)];
        var sin = new double[MaxPos * Math.Max(1, half)];
        for (int p = 0; p < MaxPos; p++)
            for (int i = 0; i < half; i++)
            {
                double theta = p * Math.Pow(10000, -(double)i / half);
                cos[p * half + i] = Math.Cos(theta);
                sin[p * half + i] = Math.Sin(theta);
            }

        return (LaneRound(lane, cos), LaneRound(lane, sin), [.. Enumerable.Range(0, steps).Select(t => t * 7 % MaxPos)]);
    }

    private static bool RopeHeads(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        string lane = type == "float64" ? "f64" : "f32";
        var cases = new (int Batch, int Steps, int Heads, int Cols, int Half, bool Interleaved, bool Gain)[]
        {
            (2, 5, 3, 64, 32, false, false),
            (1, 4, 3, 40, 20, true, true),
            (3, 7, 2, 17, 8, false, true),
            (2, 3, 4, 24, 0, false, false),
        };
        foreach (var k in cases)
        {
            int rows = k.Batch * k.Steps * k.Heads;
            var a = new RopeArgs(rows, k.Cols, k.Heads, k.Steps, k.Half, k.Interleaved, false, 1e-6, k.Gain ? 1 : 0) { HeadsFirst = true };
            var (cos, sin, positions) = RopeTables(lane, k.Steps, k.Half);
            var x = Bench.Registry.Make(type, rows * k.Cols, 81, -4, 4);
            var gain = k.Gain ? Bench.Registry.Make(type, k.Cols, 82, 0.5, 2) : null;
            var ry = new double[rows * k.Cols];
            var scale = RowOps.RopeReference(Bench.Registry.Values(type, x, rows * k.Cols), gain is null ? null : Bench.Registry.Values(type, gain, k.Cols), cos, sin, positions, a, ry);
            var y = new byte[x.Length];
            Bench.Registry.WithFloat(type, new RopeRun(ops, a, x, gain, LaneBytes(lane, cos), LaneBytes(lane, sin), positions, y));
            string what = $"rope heads first {rows}x{k.Cols} half {k.Half}{(k.Interleaved ? " interleaved" : "")}{(k.Gain ? " rms" : "")}";
            double worst = Excess(type, what, y, ry, i => scale[i / k.Cols], 8);
            if (worst > 0) { Console.WriteLine($"check {what,-44} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
        }

        return ok;
    }

    private static bool CacheRopes(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        string lane = type == "float64" ? "f64" : "f32", own = Bench.Registry.IsBlock(type) ? "float32" : type;
        double laneEps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23);
        var cases = new (int Batch, int Steps, int Heads, int Cols, int Half, bool Interleaved, bool Gain, int Position)[]
        {
            (2, 4, 3, 40, 20, true, true, 3),
            (1, 5, 2, 24, 8, false, false, 18),
            (2, 3, 2, 24, 0, false, false, 0),
        };
        const int cap = 21;
        foreach (var k in cases)
        foreach (var (cacheType, scaled) in new[] { (own, false), ("int8", true) })
        {
            int rows = k.Batch * k.Steps * k.Heads, heads = k.Batch * k.Heads, d = k.Cols;
            var view = new CacheView(heads, cap, d);
            var a = new RopeArgs(rows, d, k.Heads, k.Steps, k.Half, k.Interleaved, false, 1e-6, k.Gain ? 1 : 0);
            var (cos, sin, positions) = RopeTables(lane, k.Steps, k.Half);
            var x = Bench.Registry.Make(type, rows * d, 83, -2, 2);
            var gain = k.Gain ? Bench.Registry.Make(type, d, 84, 0.5, 2) : null;
            // The reference's rows a head at a time: what the cache's rows hold.
            var want = new double[rows * d];
            var scale = RowOps.RopeReference(Bench.Registry.Values(type, x, rows * d), gain is null ? null : Bench.Registry.Values(type, gain, d), cos, sin, positions, a with { HeadsFirst = true }, want);
            int rowBytes = (int)Bench.Registry.Bytes(cacheType, d);
            var cache = new byte[rowBytes * heads * cap];
            var scales = scaled ? new byte[Bench.Registry.Bytes(type, heads * cap)] : null;
            Bench.Registry.WithFloat(type, new CacheRopeRun(ops, cacheType, view, a, k.Position, x, gain, LaneBytes(lane, cos), LaneBytes(lane, sin), positions, cache, scales));
            string what = $"cache_write_rope {cacheType}{(scaled ? " scaled" : "")} {rows}x{d} half {k.Half}{(k.Gain ? " rms" : "")} at {k.Position}";
            string? wrong = null;
            double[] stored = scales is null ? [] : Bench.Registry.Values(type, scales, heads * cap);
            if (scales is not null)
            {
                var (wantScales, at) = view.WriteRows(want, k.Steps, k.Position, 127);
                var all = new double[heads * cap];
                var err = new double[heads * cap];
                for (int i = 0; i < at.Length; i++) if (at[i] >= 0) { all[at[i]] = wantScales[i]; err[at[i]] = scale[i] / 127; }
                double bad = Excess(type, what + " scales", scales, all, i => err[i], 8);
                if (bad > 0) { Console.WriteLine($"check {what,-56} {where,-6} {type,-12} scales DIFFER by {bad:0.##}"); ok = false; continue; }
            }

            for (int h = 0; h < heads && wrong is null; h++)
            for (int row = 0; row < cap && wrong is null; row++)
            {
                int offset = (h * cap + row) * rowBytes, step = row - k.Position;
                var bytes = cache.AsSpan(offset, rowBytes);
                if (step < 0 || step >= k.Steps)
                {
                    if (bytes.IndexOfAnyExcept((byte)0) >= 0) wrong = $"row {row} of head {h}, outside the write, changed";
                    continue;
                }

                var got = Bench.Registry.Values(cacheType, bytes.ToArray(), d);
                int r = h * k.Steps + step;
                var xs = want.AsSpan(r * d, d).ToArray();
                if (!scaled)
                {
                    double bad = Excess(cacheType, what, bytes.ToArray(), xs, _ => scale[r], 8);
                    if (bad > 0) wrong = $"row {row} of head {h} DIFFERS by {bad:0.##}";
                    continue;
                }

                // Each code within half a scale of its value by the scale as stored (and the lane's rounding of the value); a
                // code at the range's end may stand for a value past it (a stored scale below the row's).
                double s = stored[h * cap + row];
                for (int c = 0; c < d && wrong is null; c++)
                {
                    double miss = Math.Abs(got[c] * s - xs[c]), tol = 0.5 * s * (1 + 1e-6) + 16 * laneEps * scale[r];
                    bool held = Math.Abs(got[c]) >= 127 && Math.Abs(xs[c]) >= Math.Abs(got[c] * s);
                    if (double.IsNaN(got[c]) || got[c] < -128 || got[c] > 127 || (miss > tol && !held)) wrong = $"row {row} of head {h} [{c}]: the code {got[c]:R} by the scale {s:R} for {xs[c]:R}";
                }
            }

            if (wrong is not null) { Console.WriteLine($"check {what,-56} {where,-6} {type,-12} {wrong}"); ok = false; }
        }

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
            Bench.Registry.WithFloat(type, new CacheAttentionRun(ops, cacheType, a, p, q, k, null, v, null, o));
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

    private sealed record CacheRopeRun(IDevice<ulong, ulong>? Ops, string CacheType, CacheView C, RopeArgs A, int Position, byte[] X, byte[]? Gain, byte[] Cos, byte[] Sin, int[] Positions, byte[] Cache, byte[]? Scales)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CacheType, new Typed<TN>(this));

        private sealed class Typed<TN>(CacheRopeRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TC>() where TC : struct, IAdad<TC, float>
            {
                if (r.Ops is null) return CpuDevice.Device.CacheWriteRope<TN, TC>(r.C, r.A, [r.Position], r.X, r.Gain, r.Cos, r.Sin, r.Positions, r.Cache, r.Scales);
                using var st = new Staged(r.Ops);
                string lane = r.Ops.CacheWriteRope<TN, TC>(r.C, r.A, st.Indices([r.Position]), st.In(r.X), st.In(r.Gain), st.In(r.Cos), st.In(r.Sin), st.Indices(r.Positions), st.Out(r.Cache), st.Out(r.Scales));
                st.Collect();
                return lane;
            }
        }
    }
}
