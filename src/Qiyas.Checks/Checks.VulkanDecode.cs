// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The Vulkan decode with its keys in parts (VulkanOps.DecodePartsForced: each part's running softmax and context, joined
// after), which a run's measurement may take: every float type, the decode cases of Attentions (a span start among them),
// within the lane's rounding.
using Adad.Vulkan;

namespace Qiyas;

internal static partial class Checks
{
    private static bool VulkanDecodeParts(VulkanOps ops)
    {
        bool ok = true;
        try
        {
            ops.DecodePartsForced = 3;
            foreach (var type in Bench.Registry.Names)
            {
                if (!Bench.Registry.IsFloat(type)) continue;
                foreach (var a in new AttentionArgs[] { new(2, 3, 1001, 24, 0.2, false, null, [5, 300, 900]), new(2, 3, 1001, 24, 0.2, false), new(2, 5, 1001, 17, 0.2, true) })
                {
                    var q = Bench.Registry.Make(type, (int)a.QValues, 41, -1, 1);
                    var k = Bench.Registry.Make(type, (int)a.KValues, 42, -1, 1);
                    var v = Bench.Registry.Make(type, (int)a.KValues, 43, -1, 1);
                    var o = Bench.Registry.Poisoned(type, a.QValues);
                    var want = a.Reference(Bench.Registry.Values(type, q, (int)a.QValues), Bench.Registry.Values(type, k, (int)a.KValues), Bench.Registry.Values(type, v, (int)a.KValues));
                    Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, o, true, null, null, null));
                    ok &= Close(type, $"attention decode [3 parts]{(a.Causal ? " causal" : "")}{(a.Spans is not null ? " spans" : "")} d{a.D}", "vulkan", o, want, (int)a.QValues, a.Lk + a.D);
                }
            }
        }
        finally { ops.DecodePartsForced = null; }
        if (ok) Console.WriteLine($"check {"attention decode in parts",-36} vulkan every float type: the keys in 3 parts joined after, plain, causal and with span starts, within the lane's rounding");

        return ok;
    }
}
