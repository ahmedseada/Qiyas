// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The Vulkan device's proofs before its timing: every family, every type (the block formats among them), against the same
// references (and the CPU's own results, bit for bit, where the operation is exact) as the CPU and CUDA are held to.
namespace Qiyas;

internal static partial class Checks
{
    public static bool Vulkan(Adad.Vulkan.VulkanOps ops)
    {
        Console.WriteLine($"vulkan: {ops.Vk.Facts}");
        bool ok = true;
        ok &= Adds(ops, "vulkan");
        ok &= Maps(ops, "vulkan");
        ok &= Elementwise(ops, "vulkan");
        ok &= Group2(ops, "vulkan");
        ok &= Rows(ops, "vulkan");
        ok &= Norms(ops, "vulkan");
        ok &= Resamples(ops, "vulkan");
        ok &= BoxLosses(ops, "vulkan");
        ok &= Layouts(ops, "vulkan");
        ok &= Products(ops, "vulkan");
        ok &= Optimizers(ops, "vulkan");
        ok &= Routings(ops, "vulkan");
        ok &= ExpertsCheck(ops, "vulkan");
        ok &= Ctcs(ops, "vulkan");
        ok &= Attentions(ops, "vulkan");
        ok &= AttentionOptions(ops, "vulkan");
        ok &= CacheCheck(ops, "vulkan");
        ok &= HeadsCheck(ops, "vulkan");
        ok &= DecodingCheck(ops, "vulkan");
        ok &= Convs(ops, "vulkan");
        ok &= Recurrents(ops, "vulkan");
        ok &= BlockFormats(ops);
        ok &= MemoryCheck<ulong, ulong>(ops, "vulkan");
        return ok;
    }

    /// <summary>c = a + b on the device against the CPU's, value for value (both in the type's own lane: the same sums), at
    /// 1001 and 4099 values (a last word, or block, part-filled), every type (a block format's sums encoded as the CPU's).</summary>
    private static bool Adds(IDevice<ulong, ulong> ops, string where)
    {
        bool ok = true;
        foreach (var type in Bench.Registry.Names)
        {
            foreach (int n in new[] { 1001, 4099 })
            {
                var a = Bench.Registry.Make(type, n, 51, -100, 100);
                var b = Bench.Registry.Make(type, n, 52, -100, 100);
                var cpu = new byte[Bench.Registry.Bytes(type, n)];
                var dev = new byte[cpu.Length];
                Bench.Registry.With(type, new AddRun(null, a, b, cpu, n));
                Bench.Registry.With(type, new AddRun(ops, a, b, dev, n));
                var want = Bench.Registry.Values(type, cpu, n);
                var got = Bench.Registry.Values(type, dev, n);
                for (int i = 0; i < n; i++)
                {
                    if (want[i].Equals(got[i])) continue;
                    Console.WriteLine($"check add                  {where,-6} {type,-12} DIFFERS from the CPU at [{i}] of {n}: {got[i]:R} against {want[i]:R}");
                    ok = false;
                    break;
                }
            }
        }

        if (ok) Console.WriteLine($"check add                  {where,-6} every type: the CPU's sums, value for value");
        return ok;
    }

    private sealed class AddRun(IDevice<ulong, ulong>? ops, byte[] a, byte[] b, byte[] c, int n) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) { CpuDevice.Device.Add<TN>(a, b, c, n); return true; }
            ulong da = ops.Rent(a.Length + 16), db = ops.Rent(b.Length + 16), dc = ops.Rent(c.Length + 16);
            ops.Upload(a, da); ops.Upload(b, db);
            ops.Add<TN>(da, db, dc, n);
            ops.Synchronize();
            ops.Download(dc, c);
            ops.Return(da); ops.Return(db); ops.Return(dc);
            return true;
        }
    }
}
