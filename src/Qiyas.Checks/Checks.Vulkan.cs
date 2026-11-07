// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The Vulkan device's proofs before its timing: each family it has kernels for, against the same references (and the
// CPU's own results, bit for bit, where the operation is exact) as the CPU and CUDA are held to. A family it refuses is
// said so, not proved.
namespace Qiyas;

internal static partial class Checks
{
    public static bool Vulkan(Adad.Vulkan.VulkanOps ops)
    {
        Console.WriteLine($"vulkan: {ops.Vk.Facts}");
        bool ok = true;
        ok &= Adds(ops, "vulkan");
        ok &= Maps(ops, "vulkan");
        ok &= Group2(ops, "vulkan", blocks: false);
        ok &= Rows(ops, "vulkan");
        ok &= Layouts(ops, "vulkan", blocks: false);
        return ok;
    }

    /// <summary>c = a + b on the device against the CPU's, value for value (both in the type's own lane: the same sums), at
    /// 1001 and 4099 values (a last word part-filled), every type but the block formats (refused: said so).</summary>
    private static bool Adds(IDevice<ulong, ulong> ops, string where)
    {
        bool ok = true;
        foreach (var type in Bench.Registry.Names)
        {
            if (Bench.Registry.IsBlock(type)) continue;
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

        if (ok) Console.WriteLine($"check add                  {where,-6} every type but the block formats: the CPU's sums, value for value");
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
