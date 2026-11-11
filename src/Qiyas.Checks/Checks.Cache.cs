// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches (Cache.cs, ICache) on every device, every float type of the values (the block formats among them):
//   the write       caches of the type itself, of float16 and int8 by Convert's rule, and scaled caches of int8, int4 and
//                   float8 e4m3: several rows at a position, a write that runs past the capacity and one that starts before
//                   row 0 (the rows outside the capacity not written), a row of zeros among the rows. Each unscaled value
//                   the type's own rounding of the source's; each row scale within the type's rounding of the float64
//                   reference's (a block format's: its block's, the block's other scales re-rounded with it); each code the
//                   nearest code of x / scale by the scale as stored (either way at a tie within the lane's rounding); every
//                   row and every scale outside the write as it was (a block format's scales: every untouched block).
//   attention       against the float64 reference of the rows the position fills (CacheRule.AtPosition) over the values as
//                   stored (code · stored scale): one query, grouped heads of several queries, a window, a cap on the
//                   scores, positions at the capacity's end and before the first row, and a cache past a chunk of keys.
//   the mask        every value: 0 where the step sees the position, the type's lowest elsewhere; and the position advanced.
namespace Qiyas;

internal static partial class Checks
{
    public static bool CacheCheck(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            foreach (var (cacheType, scaled) in CacheCases(type))
            {
                ok &= CacheWrites(ops, where, type, cacheType, scaled);
                ok &= CacheAttentions(ops, where, type, cacheType, scaled);
            }

            ok &= DecoderMasks(ops, where, type);
        }

        ok &= PositionAdvances(ops, where);
        if (ok) Console.WriteLine($"check caches                 {where,-6} every float type{(blocks ? "" : " but the block formats")}: rows written at the device's position (the type itself, float16 and int8 by one rounding; int8, int4 and float8 by row scales), attention over the rows filled (grouped heads, a window, a cap), the decoder's mask and the position advanced");
        return ok;
    }

    // The caches of a values type: (the cache's type, with row scales or not).
    private static IEnumerable<(string Type, bool Scaled)> CacheCases(string type)
    {
        if (!Bench.Registry.IsBlock(type)) yield return (type, false);
        if (type != "float16") yield return ("float16", false);
        if (type == "float32") yield return ("float64", false);
        yield return ("int8", false);
        yield return ("int8", true);
        yield return ("int4", true);
        yield return ("float8_e4m3", true);
    }

    private static bool CacheWrites(IDevice<ulong, ulong>? ops, string where, string type, string cacheType, bool scaled)
    {
        const int heads = 2, cap = 21, d = 24;
        var view = new CacheView(heads, cap, d);
        string what = $"cache_write {cacheType}{(scaled ? " scaled" : "")}";
        var (min, max) = Bench.Registry.With(cacheType, new CodeRange());
        bool floatCodes = Bench.Registry.IsFloat(cacheType), blockScales = Bench.Registry.IsBlock(type);
        double laneEps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23);
        int rowBytes = (int)Bench.Registry.Bytes(cacheType, d), scaleCount = heads * cap;
        int blockValues = type switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 1 };
        int scaleUnitBytes = (int)Bench.Registry.Bytes(type, blockValues);
        // The buffers before the first write: codes in range, scales of the magnitude a write stores.
        var cache = Bench.Registry.Make(cacheType, heads * cap * d, 61, floatCodes ? -1 : Math.Max(min, -3), floatCodes ? 1 : Math.Min(max, 3));
        byte[]? scales = scaled ? Bench.Registry.Encode(type, [.. Enumerable.Range(0, scaleCount).Select(i => (1 + i % 7 / 8.0) / max)]) : null;
        foreach (var (p, steps, seed) in new[] { (3, 4, 62), (19, 5, 63), (-2, 3, 64) })
        {
            string at = $"{what} at {p}";
            int n = heads * steps * d;
            var x = Bench.Registry.Values(type, Bench.Registry.Make(type, n, seed, -2, 1.25), n);
            for (int c = 0; c < d; c++) x[(steps + 1) * d + c] = 0;                              // a row of zeros (head 1, step 1)
            var source = Bench.Registry.Encode(type, x);
            x = Bench.Registry.Values(type, source, n);
            var cacheBefore = (byte[])cache.Clone();
            var scalesBefore = (byte[]?)scales?.Clone();
            Bench.Registry.WithFloat(type, new CacheWriteRun(ops, cacheType, view, steps, p, source, cache, scales));

            bool Written(int row) => row >= p && row < p + steps;
            string? wrong = null;
            double[] stored = scales is null ? [] : Bench.Registry.Values(type, scales, scaleCount);
            if (scales is not null)
            {
                // The scales: the written rows' against the reference, the others as they were (a block's re-rounded with it).
                var want = Bench.Registry.Values(type, scalesBefore!, scaleCount);
                var (wantScales, rows) = view.WriteRows(x, steps, p, max);
                for (int i = 0; i < rows.Length; i++) if (rows[i] >= 0) want[rows[i]] = wantScales[i];
                double bad = Excess(type, at + " scales", scales, want, i => Math.Abs(want[i]), 2);
                if (bad > 0) { Console.WriteLine($"check {at,-40} {where,-6} {type,-12} scales DIFFER by {bad:0.##}"); return false; }
                for (int u = 0; u * blockValues < scaleCount && wrong is null; u++)
                {
                    bool touched = false;
                    for (int i = u * blockValues; i < Math.Min(scaleCount, (u + 1) * blockValues); i++) touched |= Written(i % cap);
                    if (!touched && !scales.AsSpan(u * scaleUnitBytes, scaleUnitBytes).SequenceEqual(scalesBefore.AsSpan(u * scaleUnitBytes, scaleUnitBytes)))
                        wrong = $"scale{(blockScales ? "s of block" : "")} {u}, outside the write, changed";
                }
            }

            bool Tie(double q) => Math.Abs(Math.Abs(q - Math.Floor(q)) - 0.5) < 64 * laneEps * Math.Max(1, Math.Abs(q));
            for (int h = 0; h < heads && wrong is null; h++)
            for (int row = 0; row < cap && wrong is null; row++)
            {
                int offset = (h * cap + row) * rowBytes;
                if (!Written(row))
                {
                    if (!cache.AsSpan(offset, rowBytes).SequenceEqual(cacheBefore.AsSpan(offset, rowBytes))) wrong = $"row {row} of head {h}, outside the write, changed";
                    continue;
                }

                var got = Bench.Registry.Values(cacheType, cache.AsSpan(offset, rowBytes).ToArray(), d);
                var xs = x.AsSpan((h * steps + row - p) * d, d).ToArray();
                if (!scaled)
                {
                    // One rounding of the source's value into the cache's type.
                    var want = Bench.Registry.Values(cacheType, Bench.Registry.Encode(cacheType, xs), d);
                    for (int c = 0; c < d && wrong is null; c++) if (!got[c].Equals(want[c])) wrong = $"row {row} of head {h} [{c}]: {got[c]:R}, not {want[c]:R} (the source's {xs[c]:R})";
                    continue;
                }

                double s = stored[h * cap + row];
                for (int c = 0; c < d && wrong is null; c++)
                {
                    double q = s > 0 ? xs[c] / s : 0, code = got[c];
                    if (code < min || code > max || double.IsNaN(code)) { wrong = $"row {row} of head {h} [{c}]: the code {code:R} outside [{min}, {max}]"; break; }
                    if (floatCodes)
                    {
                        double held = Math.Clamp(q, min, max), nearest = Bench.Registry.Values(cacheType, Bench.Registry.Encode(cacheType, [held]), 1)[0];
                        if (Math.Abs(code - held) > Math.Abs(nearest - held) + 4 * laneEps * Math.Abs(held)) wrong = $"row {row} of head {h} [{c}]: the code {code:R}, not {nearest:R} (x / scale = {q:R})";
                    }
                    else
                    {
                        double want = Math.Clamp(Math.Round(q, MidpointRounding.ToEven), min, max);
                        bool near = code == want || (Tie(q) && Math.Abs(code - Math.Clamp(q, min, max)) <= 0.5 + 64 * laneEps * Math.Max(1, Math.Abs(q)));
                        if (!near) wrong = $"row {row} of head {h} [{c}]: the code {code:R}, not {want:R} (x / scale = {q:R})";
                    }
                }
            }

            if (wrong is not null) { Console.WriteLine($"check {at,-40} {where,-6} {type,-12} {wrong}"); return false; }
        }

        return true;
    }

    private static bool CacheAttentions(IDevice<ulong, ulong>? ops, string where, string type, string cacheType, bool scaled)
    {
        const int cap = 21, d = 24;
        var cases = new (string Name, AttentionArgs A, int Position)[]
        {
            ("one query", new AttentionArgs(2, 1, cap, d, 0.2, true), 9),
            ("grouped 3 queries", new AttentionArgs(4, 3, cap, d, 0.2, true) { KvHeads = 2 }, 9),
            ("window", new AttentionArgs(2, 2, cap, d, 0.2, true) { Window = 5 }, 12),
            ("softcap", new AttentionArgs(2, 2, cap, d, 0.6, true) { Softcap = 1 }, 9),
            ("capacity's end", new AttentionArgs(2, 3, cap, d, 0.2, true), 19),
            ("before row 0", new AttentionArgs(2, 2, cap, d, 0.2, true), -1),
            ("long shared window", new AttentionArgs(3, 2, 300, d, 0.2, true) { KvHeads = 1, Window = 270 }, 290),
        };
        var (min, max) = Bench.Registry.With(cacheType, new CodeRange());
        bool floatCodes = Bench.Registry.IsFloat(cacheType);
        // Codes within [-1, 1] (a float type) or the registry's small integers (within [-2, 2]); row scales that keep the values near 1.
        double low = floatCodes ? -1 : Math.Max(min, -2), high = floatCodes ? 1 : Math.Min(max, 2), unit = floatCodes ? 1 : 0.5;
        bool ok = true;
        foreach (var (name, a, p) in cases)
        {
            int rows = a.KvHeadCount * a.Lk, n = rows * d;
            byte[] kc = Bench.Registry.Make(cacheType, n, 71, low, high), vc = Bench.Registry.Make(cacheType, n, 72, low, high);
            byte[]? ks = scaled ? Bench.Registry.Encode(type, [.. Enumerable.Range(0, rows).Select(i => unit * (0.5 + i * 37 % 11 / 20.0))]) : null;
            byte[]? vs = scaled ? Bench.Registry.Encode(type, [.. Enumerable.Range(0, rows).Select(i => unit * (0.5 + i * 29 % 13 / 24.0))]) : null;
            // The rows' values as stored: the codes, by their row's scale as the type stores it.
            double[] k = Bench.Registry.Values(cacheType, kc, n), v = Bench.Registry.Values(cacheType, vc, n);
            if (scaled)
            {
                double[] sk = Bench.Registry.Values(type, ks!, rows), sv = Bench.Registry.Values(type, vs!, rows);
                for (int i = 0; i < n; i++) { k[i] *= sk[i / d]; v[i] *= sv[i / d]; }
            }

            var q = Bench.Registry.Make(type, (int)a.QValues, 73, -1, 1);
            var want = CacheRule.AtPosition(a, p).Reference(Bench.Registry.Values(type, q, (int)a.QValues), k, v);
            var o = new byte[q.Length];
            Bench.Registry.WithFloat(type, new CacheAttentionRun(ops, cacheType, a, p, q, kc, ks, vc, vs, o));
            ok &= Close(type, $"attention_cached {name}, {cacheType}{(scaled ? " scaled" : "")}", where, o, want, (int)a.QValues, a.Lk + a.D);
        }

        return ok;
    }

    private static bool DecoderMasks(IDevice<ulong, ulong>? ops, string where, string type)
    {
        const int rows = 5, cap = 21;
        var (lowest, _) = Bench.Registry.With(type, new CodeRange());
        bool ok = true;
        foreach (var (p, window) in new[] { (7, 0), (7, 4), (18, 3), (-2, 0) })
        {
            int n = rows * cap;
            var mask = Bench.Registry.Make(type, n, 81, -1, 1);
            Bench.Registry.WithFloat(type, new DecoderMaskRun(ops, rows, cap, window, p, mask));
            var want = new double[n];
            for (int i = 0; i < n; i++) want[i] = CacheView.Sees(p, i / cap, i % cap, window) ? 0 : lowest;
            string what = $"decoder_mask at {p} window {window}";
            bool close = Close(type, what, where, mask, want, n);
            ok &= close;
            if (!close || Bench.Registry.IsBlock(type)) continue;
            var got = Bench.Registry.Values(type, mask, n);
            for (int i = 0; i < n; i++)
            {
                if (got[i] == want[i]) continue;
                Console.WriteLine($"check {what,-40} {where,-6} {type,-12} [{i}]: {got[i]:R}, not {want[i]:R}");
                ok = false;
                break;
            }
        }

        return ok;
    }

    private static bool PositionAdvances(IDevice<ulong, ulong>? ops, string where)
    {
        var position = new[] { 7, 1234 };
        if (ops is null)
        {
            CpuDevice.Device.PositionAdvance(position, 3);
            CpuDevice.Device.PositionAdvance(position, -4);
        }
        else
        {
            using var st = new Staged(ops);
            ulong at = st.IndicesOut(position);
            ops.PositionAdvance(at, 3);
            ops.PositionAdvance(at, -4);
            st.Collect();
        }

        if (position[0] == 6 && position[1] == 1234) return true;
        Console.WriteLine($"check position_advance       {where,-6} the position {position[0]} (its neighbour {position[1]}) after +3 and -4 from 7, not 6 (1234)");
        return false;
    }

    private sealed record CacheWriteRun(IDevice<ulong, ulong>? Ops, string CacheType, CacheView C, int Steps, int Position, byte[] Source, byte[] Cache, byte[]? Scales) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CacheType, new Typed<TN>(this));

        private sealed class Typed<TN>(CacheWriteRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TC>() where TC : struct, IAdad<TC, float>
            {
                if (r.Ops is null) return CpuDevice.Device.CacheWrite<TN, TC>(r.C, r.Steps, [r.Position], r.Source, r.Cache, r.Scales);
                using var st = new Staged(r.Ops);
                string lane = r.Ops.CacheWrite<TN, TC>(r.C, r.Steps, st.Indices([r.Position]), st.In(r.Source), st.Out(r.Cache), st.Out(r.Scales));
                st.Collect();
                return lane;
            }
        }
    }

    private sealed record CacheAttentionRun(IDevice<ulong, ulong>? Ops, string CacheType, AttentionArgs A, int Position, byte[] Q, byte[] K, byte[]? KScales, byte[] V, byte[]? VScales, byte[] O)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CacheType, new Typed<TN>(this));

        private sealed class Typed<TN>(CacheAttentionRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TC>() where TC : struct, IAdad<TC, float>
            {
                if (r.Ops is null) return CpuDevice.Device.AttentionCached<TN, TC>(r.A, [r.Position], r.Q, r.K, r.KScales, r.V, r.VScales, r.O);
                using var st = new Staged(r.Ops);
                string lane = r.Ops.AttentionCached<TN, TC>(r.A, st.Indices([r.Position]), st.In(r.Q), st.In(r.K), st.In(r.KScales), st.In(r.V), st.In(r.VScales), st.Out(r.O));
                st.Collect();
                return lane;
            }
        }
    }

    private sealed class DecoderMaskRun(IDevice<ulong, ulong>? ops, int rows, int capacity, int window, int position, byte[] mask) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.DecoderMask<TN>(rows, capacity, window, [position], mask);
            using var st = new Staged(ops);
            string lane = ops.DecoderMask<TN>(rows, capacity, window, st.Indices([position]), st.Out(mask));
            st.Collect();
            return lane;
        }
    }
}
