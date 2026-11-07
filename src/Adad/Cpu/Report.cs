// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// What a run of Adad is made of on this machine: the facts the kernels were chosen by (measured, or taken from the
// runtime's reports — nothing assumed about a vendor, a width or a capability), for --report.

using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
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
        // The product's workspace (src/Adad/Workspace.cs): a float16 product with Aᵀ stored, twice; the second call's counts.
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
