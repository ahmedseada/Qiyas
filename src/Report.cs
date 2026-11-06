// What a run of Qiyas is made of on this machine: the facts the kernels were chosen by (measured, or taken from the
// runtime's reports — nothing assumed about a vendor, a width or a capability), for --report.
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Text;

namespace Qiyas;

internal static partial class Kernels
{
    public static string Report()
    {
        var sb = new StringBuilder();
        sb.AppendLine("report cpu");
        sb.AppendLine($"  team: {Team.Size} threads");
        sb.AppendLine($"  vector: {Vector<byte>.Count * 8} bits ({Vector<float>.Count} f32, {Vector<byte>.Count} bytes a vector); 128-bit hardware {(Vector128.IsHardwareAccelerated ? "yes" : "no")}, 256-bit {(Vector256.IsHardwareAccelerated ? "yes" : "no")}, 512-bit {(Vector512.IsHardwareAccelerated ? "yes" : "no")}");
        sb.AppendLine($"  dot lanes: s8 {(S8Dot.IsSupported ? "yes" : "no")}, u8 {(U8Dot.IsSupported ? "yes" : "no")}, s16 {(S16Dot.IsSupported ? "yes" : "no")}, u16 {(U16Dot.IsSupported ? "yes" : "no")} (the runtime's report; a lane not supported is not used)");
        sb.AppendLine("  lane rules: own (element-wise), float (f32 for narrow floats, f64 for float64), sum (dot lanes → f32 ≤ 2^24 → s32/u32 → f64 ≤ 2^53 → s64/u64, by the bounds measured on the data)");
        sb.AppendLine("  block formats: Q8_0, Q4_0, Q4_K, Q6_K read a value at a time, written a block at a time; bool at 1 bit");
        return sb.ToString();
    }
}

public sealed partial class GpuOps
{
    private readonly List<string> _decisions = [];

    /// <summary>A decision taken from a measurement or a refusal, kept for the report.</summary>
    internal void Note(string decision) { if (!_decisions.Contains(decision)) _decisions.Add(decision); }

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
        // The buffer pool is the arena: a size-keyed free list, so after a shape's first call no driver allocation happens; its cost per rent/return pair, measured.
        ulong warm = gpu.Rent(1 << 20); gpu.Return(warm);
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int i = 0; i < 10000; i++) { ulong p = gpu.Rent(1 << 20); gpu.Return(p); }
        double ns = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1e6 / 10000;
        sb.AppendLine($"  buffers: a size-keyed pool; a rent/return pair after a shape's first call costs {ns:0} ns and no driver call (the arena)");
        return sb.ToString();
    }
}
