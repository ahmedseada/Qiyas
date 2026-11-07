// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// What a run of Qiyas is made of on this machine: the facts the kernels were chosen by (measured, or taken from the
// runtime's reports — nothing assumed about a vendor, a width or a capability), for --report.
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;

namespace Qiyas;

internal static partial class Kernels
{
    private static readonly List<string> s_decisions = [];

    /// <summary>A decision taken from a measurement on the CPU, kept for the report.</summary>
    internal static void Note(string decision) { lock (s_decisions) if (!s_decisions.Contains(decision)) s_decisions.Add(decision); }

    /// <summary>The CPU decisions taken from measurements so far (the bench takes most of them after the report is printed).</summary>
    public static string Measured()
    {
        var sb = new StringBuilder();
        lock (s_decisions) foreach (var d in s_decisions) sb.AppendLine($"  measured cpu: {d}");
        return sb.ToString();
    }

    public static string Report()
    {
        var sb = new StringBuilder();
        sb.AppendLine("report cpu");
        sb.AppendLine($"  team: {Team.Size} threads");
        sb.AppendLine($"  vector: {Vector<byte>.Count * 8} bits ({Vector<float>.Count} f32, {Vector<byte>.Count} bytes a vector); 128-bit hardware {(Vector128.IsHardwareAccelerated ? "yes" : "no")}, 256-bit {(Vector256.IsHardwareAccelerated ? "yes" : "no")}, 512-bit {(Vector512.IsHardwareAccelerated ? "yes" : "no")}");
        sb.AppendLine($"  dot lanes: s8 {(S8Dot.IsSupported ? "yes" : "no")}, u8 {(U8Dot.IsSupported ? "yes" : "no")}, s16 {(S16Dot.IsSupported ? "yes" : "no")}, u16 {(U16Dot.IsSupported ? "yes" : "no")} (the runtime's report; a lane not supported is not used)");
        sb.AppendLine("  lane rules: own (element-wise), float (f32 for narrow floats, f64 for float64), sum (dot lanes → f32 ≤ 2^24 → s32/u32 → f64 ≤ 2^53 → s64/u64, by the bounds measured on the data)");
        sb.AppendLine("  block formats: Q8_0, Q4_0, Q4_K, Q6_K read a value at a time, written a block at a time; bool at 1 bit");
        // The product's workspace (src/Workspace.cs): a float16 product with Aᵀ stored, twice; the second call's counts.
        {
            var v = new ProductView(96, 80, 72, true, false);
            byte[] a = new byte[96 * 72 * 2], b = new byte[72 * 80 * 2], c = new byte[96 * 80 * 2];
            MatMulView<Float16>(v, a, b, c);
            var (r0, g0, p0) = Workspace.Counts;
            MatMulView<Float16>(v, a, b, c);
            var (r1, g1, p1) = Workspace.Counts;
            sb.AppendLine($"  buffers: the product's packed B, loaded A / B, C lanes and A panels from a per-thread workspace grown to the largest shape seen; a float16 product's second call: {r1 - r0} workspace rents, {g1 - g0} allocations, {p1 - p0} pool fallbacks");
        }

        lock (s_decisions) foreach (var d in s_decisions) sb.AppendLine($"  measured: {d}");
        return sb.ToString();
    }
}

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
        // Buffers: outside a scope the size-keyed pool, inside one the arena (src/Arena.cs); a rent/return pair of each, measured,
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
