// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention's options (Attention.cs) on every device against the float64 reference, every float type: grouped heads
// (KvHeads: grouped and multi-query) by every form (the measured choice, the fused rows, the composition, decode, the
// backward by each route), a sliding window (with the causal rule, with spans and starts, with grouped heads), a cap on the
// scores (forward and backward: the composition), each query's log-sum-exp, and queries that see no key (a span of none, a
// start at or past the span's end) by every form and backward route: O and the gradients 0, the log-sum-exp −∞ (the types
// that store −∞); within the lane's rounding over the keys, as the attention check's.
namespace Qiyas;

internal static partial class Checks
{
    public static bool AttentionOptions(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        var cuda = ops as GpuOps;
        var vulkan = ops as Adad.Vulkan.VulkanOps;
        static int[] Range(int n, Func<int, int> f) => Enumerable.Range(0, n).Select(f).ToArray();
        var cases = new (string Name, AttentionArgs A)[]
        {
            ("grouped causal", new AttentionArgs(4, 37, 53, 24, 0.2, true) { KvHeads = 2 }),
            ("multi-query", new AttentionArgs(3, 37, 53, 17, 0.2, false) { KvHeads = 1 }),
            ("window causal", new AttentionArgs(2, 37, 53, 24, 0.2, true) { Window = 9 }),
            ("grouped window spans starts", new AttentionArgs(4, 37, 53, 24, 0.2, false, Range(37, i => 20 + i % 30), Range(37, i => i % 25)) { KvHeads = 2, Window = 11 }),
            ("softcap causal", new AttentionArgs(2, 37, 53, 24, 0.6, true) { Softcap = 1 }),
            ("grouped softcap window", new AttentionArgs(4, 37, 53, 24, 0.6, false) { KvHeads = 2, Softcap = 1, Window = 20 }),
            ("keyless spans", new AttentionArgs(2, 5, 9, 8, 0.3, false, [0, 3, 9, 1, 0]) { KvHeads = 1 }),
            ("keyless starts", new AttentionArgs(2, 5, 9, 8, 0.3, false, [4, 3, 9, 6, 9], [4, 0, 9, 2, 3]) { KvHeads = 1 }),
        };
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            bool infinite = Bench.Registry.WithFloat(type, new StoresInfinity());
            foreach (var (name, a) in cases)
            {
                var q = Bench.Registry.Make(type, (int)a.QValues, 41, -1, 1);
                var k = Bench.Registry.Make(type, (int)a.KValues, 42, -1, 1);
                var v = Bench.Registry.Make(type, (int)a.KValues, 43, -1, 1);
                double[] qv = Bench.Registry.Values(type, q, (int)a.QValues), kv = Bench.Registry.Values(type, k, (int)a.KValues), vv = Bench.Registry.Values(type, v, (int)a.KValues);
                var lseWant = new double[(long)a.Heads * a.Lq];
                var want = a.Reference(qv, kv, vv, null, lseWant);
                int terms = a.Lk + a.D;

                // O by every form: the measured choice, the fused rows, the composition, decode.
                foreach (string form in new[] { "measured", "fused", "composed", "decode" })
                {
                    var o = new byte[q.Length];
                    Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, o, form == "decode", null, null, null, fused: form == "fused", composed: form == "composed"));
                    ok &= CloseNumbers(type, $"attention {name} {form}", where, o, want, (int)a.QValues, terms);
                }

                // O and each query's log-sum-exp (a query that sees no key: −∞, on the types that store it).
                if (infinite || !name.StartsWith("keyless")) ok &= LseCase(ops, where, type, $"attention_lse {name}", a, q, k, v, want, lseWant, terms);

                // The backward by each route (a cap takes the composition by every one).
                var dO = Bench.Registry.Make(type, (int)a.QValues, 44, -1, 1);
                var (wq, wk, wv) = a.ReferenceBackward(qv, kv, vv, Bench.Registry.Values(type, dO, (int)a.QValues));
                foreach (int mode in cuda is null ? new[] { 0, 1 } : [0, 1, 2])
                {
                    // CPU: 0 the fused passes, 1 the composition. CUDA: 0 the default, 1 the composition, 2 the FMA fused
                    // kernels. Vulkan: 0 the default, 1 the composition.
                    if (cuda is not null) { cuda.FusedBackward = mode == 1 ? false : null; cuda.MmaBackward = mode == 2 ? false : null; cuda.BackwardRoute = null; }
                    else if (vulkan is not null) { vulkan.FusedBackward = mode == 1 ? false : null; vulkan.BackwardRoute = null; }
                    else Kernels.FusedBackwardCpu = mode == 0;
                    byte[] dq = new byte[q.Length], dk = new byte[k.Length], dv = new byte[v.Length];
                    Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, dO, false, dq, dk, dv));
                    string tag = $" {name}{(mode == 1 ? " composed" : mode == 2 ? " fma" : "")}";
                    int rows = a.Group * a.Lq;
                    ok &= CloseNumbers(type, "attention_bwd dQ" + tag, where, dq, wq, (int)a.QValues, 2 * (a.Lk + a.D));
                    ok &= CloseNumbers(type, "attention_bwd dK" + tag, where, dk, wk, (int)a.KValues, 2 * (rows + a.D));
                    ok &= CloseNumbers(type, "attention_bwd dV" + tag, where, dv, wv, (int)a.KValues, rows + a.D);
                    string? route = cuda?.BackwardRoute ?? vulkan?.BackwardRoute;
                    if (a.Softcap > 0 && ops is not null && route != "composed")
                    {
                        Console.WriteLine($"check attention_bwd{tag} {where} {type,-12} FAILS: took the {route} route, not the composition");
                        ok = false;
                    }

                    if (cuda is not null) { cuda.FusedBackward = null; cuda.MmaBackward = null; }
                    else if (vulkan is not null) vulkan.FusedBackward = null;
                    else Kernels.FusedBackwardCpu = null;
                }
            }
        }

        if (ok) Console.WriteLine($"check attention options      {where,-6} every float type{(blocks ? "" : " but the block formats")}: grouped heads and multi-query by every form, a sliding window, a cap on the scores, each query's log-sum-exp, and queries that see no key (by a span or a start: O and the gradients 0, the log-sum-exp -inf where the type stores it), forward and backward within the lane's rounding");
        return ok;
    }

    // O and the log-sum-exp by AttentionLse against the reference: −∞ exactly where the reference is, the rest within the
    // rounding over the keys.
    private static bool LseCase(IDevice<ulong, ulong>? ops, string where, string type, string what, AttentionArgs a, byte[] q, byte[] k, byte[] v, double[] want, double[] lseWant, int terms)
    {
        var o = new byte[q.Length];
        int n = a.Heads * a.Lq;
        var lse = new byte[Bench.Registry.Bytes(type, n)];
        Bench.Registry.WithFloat(type, new AttentionLseRun(ops, a, q, k, v, o, lse));
        bool ok = CloseNumbers(type, what + " O", where, o, want, (int)a.QValues, terms);
        ok &= Close(type, what, where, lse, lseWant, n, terms);
        var got = Bench.Registry.Values(type, lse, n);
        for (int i = 0; i < n; i++)
        {
            if (double.IsNegativeInfinity(lseWant[i]) == double.IsNegativeInfinity(got[i]) && !double.IsNaN(got[i])) continue;
            Console.WriteLine($"check {what} {where,-6} {type,-12} DIFFERS: row {i} {got[i]:R}, not {lseWant[i]:R}");
            return false;
        }

        return ok;
    }

    // Close, and a NaN only where the reference is NaN (Close's comparison lets a NaN pass).
    private static bool CloseNumbers(string type, string what, string where, byte[] output, double[] reference, int n, int terms)
    {
        bool ok = Close(type, what, where, output, reference, n, terms);
        var got = Bench.Registry.Values(type, output, n);
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(got[i]) == double.IsNaN(reference[i])) continue;
            Console.WriteLine($"check {what} {where,-6} {type,-12} DIFFERS: [{i}] {got[i]:R}, not {reference[i]:R}");
            return false;
        }

        return ok;
    }

    // Whether the type stores −∞ (CacheView.Excluded: −∞ where it does).
    private readonly struct StoresInfinity : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => double.IsNegativeInfinity(CacheView.Excluded<TN>());
    }

    private sealed class AttentionLseRun(IDevice<ulong, ulong>? ops, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, byte[] lse) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.AttentionLse<TN>(a, q, k, v, o, lse);
            using var st = new Staged(ops);
            var lane = ops.AttentionLse<TN>(a, st.In(q), st.In(k), st.In(v), st.Out(o), st.Out(lse));
            st.Collect();
            return lane;
        }
    }
}
