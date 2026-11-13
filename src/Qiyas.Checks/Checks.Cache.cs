// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches (Cache.cs, ICache) on every device, every float type of the values (the block formats among them):
//   the write       caches of the type itself, of float16, float64 and int8 by Convert's rule, and caches of each block
//                   format (Q8_0, Q4_0, Q4_K, Q6_K) with rows of whole blocks and with rows that share blocks (each value
//                   within its block's rounding of the source's or, kept, of what was stored; every block the write does not
//                   touch as it was): several rows at a position, a write that runs past the capacity and one that starts
//                   before row 0 (the rows outside the capacity not written), a row of zeros among the rows. Each value of a
//                   plain type the type's own rounding of the source's; every row outside the write as it was.
//   attention       against the float64 reference of the rows the position fills (CacheRule.AtPosition) over the values as
//                   stored: one query, grouped heads of several queries, a window, a cap on the scores, positions at the
//                   capacity's end and before the first row, and a cache past a chunk of keys.
//   the mask        every value: 0 where the step sees the position, elsewhere the excluded score (−∞ as the type stores it;
//                   the type's lowest where it holds no −∞); and the position advanced.
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
            foreach (var cacheType in CacheCases(type))
            {
                ok &= CacheWrites(ops, where, type, cacheType, 24);
                // A block format's rows of whole blocks too (24 values a row share them).
                if (Bench.Registry.IsBlock(cacheType)) ok &= CacheWrites(ops, where, type, cacheType, cacheType is "q8_0" or "q4_0" ? 64 : 256);
                ok &= CacheAttentions(ops, where, type, cacheType);
            }

            ok &= DecoderMasks(ops, where, type);
        }

        ok &= PositionAdvances(ops, where);
        if (ok) Console.WriteLine($"check caches                 {where,-6} every float type{(blocks ? "" : " but the block formats")}: rows written at the device's position (the type itself, float16, float64 and int8 by one rounding; Q8_0, Q4_0, Q4_K and Q6_K by whole blocks and by shared ones), attention over the rows filled (grouped heads, a window, a cap), the decoder's mask and the position advanced");
        return ok;
    }

    // The caches of a values type: the cache's types.
    private static IEnumerable<string> CacheCases(string type)
    {
        if (!Bench.Registry.IsBlock(type)) yield return type;
        if (type != "float16") yield return "float16";
        if (type == "float32") yield return "float64";
        yield return "int8";
        foreach (string block in new[] { "q8_0", "q4_0", "q4_k", "q6_k" }) yield return block;
    }

    private static bool CacheWrites(IDevice<ulong, ulong>? ops, string where, string type, string cacheType, int d)
    {
        const int heads = 2, cap = 21;
        var view = new CacheView(heads, cap, d);
        string what = $"cache_write {cacheType} d {d}";
        int cacheBlock = cacheType switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 0 };
        var (min, max) = Bench.Registry.With(cacheType, new CodeRange());
        bool floatCodes = Bench.Registry.IsFloat(cacheType);
        int rowBytes = cacheBlock > 0 ? 0 : (int)Bench.Registry.Bytes(cacheType, d);
        // The cache before the first write: values in range.
        var cache = Bench.Registry.Make(cacheType, heads * cap * d, 61, floatCodes ? -1 : Math.Max(min, -3), floatCodes ? 1 : Math.Min(max, 3));
        foreach (var (p, steps, seed) in new[] { (3, 4, 62), (19, 5, 63), (-2, 3, 64) })
        {
            string at = $"{what} at {p}";
            int n = heads * steps * d;
            var x = Bench.Registry.Values(type, Bench.Registry.Make(type, n, seed, -2, 1.25), n);
            for (int c = 0; c < d; c++) x[(steps + 1) * d + c] = 0;                              // a row of zeros (head 1, step 1)
            var source = Bench.Registry.Encode(type, x);
            x = Bench.Registry.Values(type, source, n);
            var cacheBefore = (byte[])cache.Clone();
            Bench.Registry.WithFloat(type, new CacheWriteRun(ops, cacheType, view, steps, p, source, cache));

            bool Written(int row) => row >= p && row < p + steps;
            string? wrong = null;
            if (cacheBlock > 0)
            {
                // A block format cache: every value within its block's rounding of the source's (a written row) or of what was
                // stored (a row whose block a written row shares); every block with no written value as it was.
                int all = heads * cap * d, blockBytes = (int)Bench.Registry.Bytes(cacheType, cacheBlock);
                var want = Bench.Registry.Values(cacheType, cacheBefore, all);
                for (int h = 0; h < heads; h++)
                    for (int row = Math.Max(p, 0); row < Math.Min(p + steps, cap); row++) Array.Copy(x, (h * steps + row - p) * d, want, (h * cap + row) * d, d);
                double bad = Excess(cacheType, at, cache, want, i => Math.Abs(want[i]), 2);
                if (bad > 0) { Console.WriteLine($"check {at,-40} {where,-6} {type,-12} DIFFERS by {bad:0.##}"); return false; }
                for (int u = 0; u * cacheBlock < all; u++)
                {
                    bool touched = false;
                    for (int i = u * cacheBlock; i < Math.Min(all, (u + 1) * cacheBlock); i += 1) touched |= Written(i / d % cap);
                    int bytes = Math.Min(blockBytes, cache.Length - u * blockBytes);
                    if (touched || cache.AsSpan(u * blockBytes, bytes).SequenceEqual(cacheBefore.AsSpan(u * blockBytes, bytes))) continue;
                    Console.WriteLine($"check {at,-40} {where,-6} {type,-12} DIFFERS: block {u}, outside the write, changed");
                    return false;
                }

                continue;
            }

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
                // One rounding of the source's value into the cache's type.
                var want = Bench.Registry.Values(cacheType, Bench.Registry.Encode(cacheType, xs), d);
                for (int c = 0; c < d && wrong is null; c++) if (!got[c].Equals(want[c])) wrong = $"row {row} of head {h} [{c}]: {got[c]:R}, not {want[c]:R} (the source's {xs[c]:R})";
            }

            if (wrong is not null) { Console.WriteLine($"check {at,-40} {where,-6} {type,-12} DIFFERS: {wrong}"); return false; }
        }

        return true;
    }

    private static bool CacheAttentions(IDevice<ulong, ulong>? ops, string where, string type, string cacheType)
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
        // Values within [-1, 1] (a float type) or the registry's small integers (within [-2, 2]).
        double low = floatCodes ? -1 : Math.Max(min, -2), high = floatCodes ? 1 : Math.Min(max, 2);
        bool ok = true;
        foreach (var (name, a, p) in cases)
        {
            int rows = a.KvHeadCount * a.Lk, n = rows * d;
            byte[] kc = Bench.Registry.Make(cacheType, n, 71, low, high), vc = Bench.Registry.Make(cacheType, n, 72, low, high);
            // The rows' values as stored.
            double[] k = Bench.Registry.Values(cacheType, kc, n), v = Bench.Registry.Values(cacheType, vc, n);

            var q = Bench.Registry.Make(type, (int)a.QValues, 73, -1, 1);
            var want = CacheRule.AtPosition(a, p).Reference(Bench.Registry.Values(type, q, (int)a.QValues), k, v);
            var o = Bench.Registry.Poisoned(type, a.QValues);
            Bench.Registry.WithFloat(type, new CacheAttentionRun(ops, cacheType, a, p, q, kc, vc, o));
            ok &= Close(type, $"attention_cached {name}, {cacheType}", where, o, want, (int)a.QValues, a.Lk + a.D);
        }

        return ok;
    }

    private static bool DecoderMasks(IDevice<ulong, ulong>? ops, string where, string type)
    {
        const int rows = 5, cap = 21;
        // The excluded score: −∞ as the type stores it, the type's lowest where it holds no −∞.
        var (lowest, _) = Bench.Registry.With(type, new CodeRange());
        double back = Bench.Registry.IsBlock(type) ? double.NaN : Bench.Registry.Values(type, Bench.Registry.Encode(type, [double.NegativeInfinity]), 1)[0];
        double excluded = double.IsNegativeInfinity(back) ? back : lowest;
        bool ok = true;
        foreach (var (p, window) in new[] { (7, 0), (7, 4), (18, 3), (-2, 0) })
        {
            int n = rows * cap;
            var mask = Bench.Registry.Make(type, n, 81, -1, 1);
            Bench.Registry.WithFloat(type, new DecoderMaskRun(ops, rows, cap, window, p, mask));
            var want = new double[n];
            for (int i = 0; i < n; i++) want[i] = CacheView.Sees(p, i / cap, i % cap, window) ? 0 : excluded;
            string what = $"decoder_mask at {p} window {window}";
            if (Bench.Registry.IsBlock(type))
            {
                ok &= Close(type, what, where, mask, want, n);
                continue;
            }

            var got = Bench.Registry.Values(type, mask, n);
            for (int i = 0; i < n; i++)
            {
                if (got[i] == want[i]) continue;
                Console.WriteLine($"check {what,-40} {where,-6} {type,-12} DIFFERS: [{i}] {got[i]:R}, not {want[i]:R}");
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
        Console.WriteLine($"check position_advance       {where,-6} DIFFERS: the position {position[0]} (its neighbour {position[1]}) after +3 and -4 from 7, not 6 (1234)");
        return false;
    }

    private sealed record CacheWriteRun(IDevice<ulong, ulong>? Ops, string CacheType, CacheView C, int Steps, int Position, byte[] Source, byte[] Cache) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CacheType, new Typed<TN>(this));

        private sealed class Typed<TN>(CacheWriteRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TC>() where TC : struct, IAdad<TC, float>
            {
                if (r.Ops is null) return CpuDevice.Device.CacheWrite<TN, TC>(r.C, r.Steps, [r.Position], r.Source, r.Cache);
                using var st = new Staged(r.Ops);
                string lane = r.Ops.CacheWrite<TN, TC>(r.C, r.Steps, st.Indices([r.Position]), st.In(r.Source), st.Out(r.Cache));
                st.Collect();
                return lane;
            }
        }
    }

    private sealed record CacheAttentionRun(IDevice<ulong, ulong>? Ops, string CacheType, AttentionArgs A, int Position, byte[] Q, byte[] K, byte[] V, byte[] O)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CacheType, new Typed<TN>(this));

        private sealed class Typed<TN>(CacheAttentionRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TC>() where TC : struct, IAdad<TC, float>
            {
                if (r.Ops is null) return CpuDevice.Device.AttentionCached<TN, TC>(r.A, [r.Position], r.Q, r.K, r.V, r.O);
                using var st = new Staged(r.Ops);
                string lane = r.Ops.AttentionCached<TN, TC>(r.A, st.Indices([r.Position]), st.In(r.Q), st.In(r.K), st.In(r.V), st.Out(r.O));
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
