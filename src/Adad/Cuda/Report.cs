// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// What a run of Adad is made of on this machine: the facts the kernels were chosen by (measured, or taken from the
// runtime's reports — nothing assumed about a vendor, a width or a capability), for --report.

using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    private readonly List<string> _decisions = [];

    /// <summary>A decision taken from a measurement or a refusal, kept for the report.</summary>
    internal void Note(string decision) { if (!_decisions.Contains(decision)) _decisions.Add(decision); }

    /// <summary>Whether the device's JIT accepts the kernel (Gpu.Accepts), a refusal noted for the report.</summary>
    private bool Accepts(string ptx, string entry)
    {
        if (gpu.Accepts(ptx, entry, out string? reason)) return true;
        if (reason is not null) Note($"refuses {entry}: {reason}");
        return false;
    }

    /// <summary>The device decisions taken from measurements so far (the bench takes most of them after the report is printed).</summary>
    public string Measured()
    {
        var sb = new StringBuilder();
        foreach (var d in _decisions) sb.AppendLine($"  measured device: {d}");
        return sb.ToString();
    }

    public string Report()
    {
        var sb = new StringBuilder();
        sb.AppendLine("report device");
        sb.AppendLine($"  {gpu.Name}: {gpu.Multiprocessors} multiprocessors, capability {gpu.Capability} (as the driver reports); PTX target {gpu.Target.Trim().Replace('\n', ' ')}");
        sb.AppendLine($"  tensor kinds accepted by the JIT: s8 {(MmaSupported(Ptx.Kind.S8) ? "yes" : "no")}, s8k32 {(MmaSupported(Ptx.Kind.S8K32) ? "yes" : "no")}, u8 {(MmaSupported(Ptx.Kind.U8) ? "yes" : "no")}, s4 {(MmaSupported(Ptx.Kind.S4) ? "yes" : "no")}, u4 {(MmaSupported(Ptx.Kind.U4) ? "yes" : "no")}, f16 {(MmaSupported(Ptx.Kind.F16) ? "yes" : "no")}, bf16 {(MmaSupported(Ptx.Kind.BF16) ? "yes" : "no")}; dot (dp4a) s8 {(Dp4aSupported(true) ? "yes" : "no")}, u8 {(Dp4aSupported(false) ? "yes" : "no")} — a kind the JIT refuses is not used (the plain lanes take its place)");
        sb.AppendLine($"  sliced float64 product proven on this device: {(SlicedProven ? "yes" : "no")}; taken where measured faster than the f64 lane, at shapes in multiples of 8");
        sb.AppendLine("  lane rules: own (element-wise), float (f32 / f64), sum (tensor or dot kinds for 8-bit-ranged inputs → f32 ≤ 2^24 → s32/u32 → s64/u64; 16/32-bit integer sums by the type's range)");
        sb.AppendLine("  row kernels: a block per row, rows in registers up to 16 words a thread of 1024, longer rows in segments; rows not ending on a word value by value");
        foreach (var d in _decisions) sb.AppendLine($"  measured: {d}");
        // Buffers: outside a scope the size-keyed pool, inside one the arena (src/Adad/Arena.cs); a rent/return pair of each, measured,
        // and a transposed float16 product's second call in a scope.
        ulong warm = gpu.Rent(1 << 20); gpu.Return(warm);
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int i = 0; i < 10000; i++) { ulong p = gpu.Rent(1 << 20); gpu.Return(p); }
        double ns = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1e6 / 10000;
        double arenaNs;
        using (Scope())
        {
            ulong w = gpu.Rent(1 << 20); gpu.Return(w);
            start = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < 10000; i++) { ulong p = gpu.Rent(1 << 20); gpu.Return(p); }
            arenaNs = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1e6 / 10000;
        }

        var view = new ProductView(256, 256, 256, true, false);
        ulong da = gpu.Rent(256 * 256 * 2), db = gpu.Rent(256 * 256 * 2), dc = gpu.Rent(256 * 256 * 2);
        Gpu.Clear(da, 256 * 256 * 2); Gpu.Clear(db, 256 * 256 * 2);
        using (Scope()) MatMulView<Float16>(view, da, db, dc);
        var before = gpu.Counts;
        using (Scope()) MatMulView<Float16>(view, da, db, dc);
        var after = gpu.Counts;
        gpu.Return(da); gpu.Return(db); gpu.Return(dc);
        sb.AppendLine($"  buffers: outside a scope a size-keyed pool (a rent/return pair {ns:0} ns), inside a scope the arena, one region bumped and released from the top (a pair {arenaNs:0} ns); a transposed float16 256³ product's second call in a scope: {after.ArenaRents - before.ArenaRents} arena rents, {after.PoolRents - before.PoolRents} pool rents, {after.DriverAllocations - before.DriverAllocations} driver allocations (region {gpu.ArenaCapacity >> 20} MB)");
        return sb.ToString();
    }
}
