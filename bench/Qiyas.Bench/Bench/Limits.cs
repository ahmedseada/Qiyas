// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's limits, measured at the start of a run on the device the rows then run on: never a table of processors or
// cards, never a name. Two kinds of limit bound a row:
//
//   - the copy bandwidth: bytes read plus bytes written a second by the fastest copy tried (on the CPU the runtime's
//     memmove on one processor and on every processor, each a contiguous part, and a copy by non-temporal vector stores;
//     on CUDA the driver's copy and a copy kernel of 16-byte words), at sizes from 64 KB up by fours to 64 MB, then
//     doubled until the rate settles within 3 % (past the caches, whatever their size) or the device has no more memory
//     to give: a curve, so a row is held to the copy of its own size (a working set the caches hold runs at their rate).
//     On CUDA each copy is a launch and a wait, as every row's run is (a launch is the least a GPU row costs); on the CPU
//     a pass copies its buffers over and over, some 64 MB in all, so no dispatch is in the rate (a CPU row may run on the
//     caller with none: a small row's dispatch shows as its gap to the bound);
//   - each lane's peak multiply-add rate: independent chains of the lane's own multiply-add (Adad's lanes on the CPU,
//     ILane.Fma at the vector width the runtime uses, on every processor at once; on CUDA the lane's instruction — fma,
//     mad, dp4a, or the tensor instruction of each kind the JIT accepts — on every multiprocessor's resident warps). An
//     integer chain multiplies its own value, c = c·c + b: with both factors fixed (c = a·b + c) a compiler folds a·b
//     into a constant and the chain into adds, exact for integers (one such run read 72 T multiply-adds a second for
//     s32 on a GPU whose f32 fma ran at 22 T).
//
// A row's bound is the larger of its bytes at the copy bandwidth of their size and its operations at its lane's peak (a
// multiply-add two operations). Each limit is the median of five timed passes after one untimed; the copy curve is
// measured twice over and each size's faster kept (a pass that met the device busy elsewhere sets no limit). Before any
// of it the device is held busy copying until its rate holds: a device idle before the run has its clocks down (a GPU's
// memory clock by a factor of tens), and a limit measured before they are back up is not the device's.
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;

namespace Qiyas.Bench;

/// <summary>A device's measured limits: its copy bandwidth (bytes read and written a second) by the size of a copy, and each
/// lane's multiply-adds a second.</summary>
internal sealed class Limits(Device device, IReadOnlyList<(long Size, double Rate)> copies, string copyHow, IReadOnlyList<(string Lane, double Rate, string How)> lanes)
{
    public Device Device => device;

    /// <summary>Bytes read plus bytes written a second by the fastest copy measured past the caches (the largest size's).</summary>
    public double Copy => copies.Count == 0 ? double.NaN : copies[^1].Rate;

    /// <summary>The seconds a copy moving <paramref name="bytes"/> (read plus written) takes: the copies measured at the sizes
    /// around half of them (a copy of size s moves 2s), their times interpolated by size; below the smallest, its rate.</summary>
    public double CopySeconds(double bytes)
    {
        double size = bytes / 2;
        for (int i = 0; i < copies.Count; i++)
        {
            var (s, r) = copies[i];
            if (s < size) continue;
            if (i == 0) return bytes / r;
            var (s0, r0) = copies[i - 1];
            double t0 = 2.0 * s0 / r0, t1 = 2.0 * s / r;
            return t0 + (t1 - t0) * (size - s0) / (s - s0);
        }

        return bytes / Copy;
    }

    /// <summary>Operations a second in the lane (two a multiply-add), or null where the lane was not measured.</summary>
    public double? Ops(string lane)
    {
        foreach (var (name, rate, _) in lanes) if (name == lane) return 2 * rate;
        return null;
    }

    public string Describe()
    {
        var sb = new StringBuilder();
        sb.Append(string.Create(CultureInfo.InvariantCulture, $"limits {device.ToString().ToLowerInvariant()} (measured now): copy {Copy / 1e9:0.0} GB/s ({copyHow}; by size:"));
        foreach (var (size, rate) in copies) sb.Append(string.Create(CultureInfo.InvariantCulture, $" {Work.Size(size, "B")} {rate / 1e9:0.#}"));
        sb.Append(" GB/s); multiply-adds a second by lane:");
        foreach (var (name, rate, how) in lanes) sb.Append(string.Create(CultureInfo.InvariantCulture, $" {name} {Rate(rate)} ({how});"));
        return sb.ToString().TrimEnd(';');
    }

    /// <summary>The line the results file states the limits by.</summary>
    public string Short()
    {
        var sb = new StringBuilder(string.Create(CultureInfo.InvariantCulture, $"{device.ToString().ToLowerInvariant()}: copy {Copy / 1e9:0.0} GB/s past the caches ("));
        sb.Append(string.Join(", ", copies.Select(c => string.Create(CultureInfo.InvariantCulture, $"{Work.Size(c.Size, "B")} {c.Rate / 1e9:0.#}"))));
        sb.Append(" GB/s by size)");
        foreach (var (name, rate, _) in lanes) sb.Append(string.Create(CultureInfo.InvariantCulture, $", {name} {Rate(rate)} MA/s"));
        return sb.ToString();
    }

    private static string Rate(double r) => r >= 1e12 ? string.Create(CultureInfo.InvariantCulture, $"{r / 1e12:0.00}T") : string.Create(CultureInfo.InvariantCulture, $"{r / 1e9:0.0}G");

    public static Limits Measure(Device device) => device switch
    {
        Device.Cpu => CpuLimits.Measure(),
        Device.Cuda => CudaLimits.Measure(),
        _ => new Limits(device, [], "not measured on this device", []),
    };

    /// <summary>Runs `batch` (some 100 ms of work a call, returning its rate) until two batches in a row agree within 3 % after
    /// half a second, or three seconds pass: the device's clocks up to what sustained work gets.</summary>
    internal static void Warm(Func<double> batch)
    {
        var clock = Stopwatch.StartNew();
        double last = 0;
        while (clock.Elapsed.TotalSeconds < 3)
        {
            double r = batch();
            if (clock.Elapsed.TotalSeconds > 0.5 && Math.Abs(r - last) <= 0.03 * last) return;
            last = r;
        }
    }

    /// <summary>The median of five timed passes after an untimed one (each pass's rate from the work it did and its seconds).</summary>
    internal static double Median(Func<double> rate)
    {
        rate();
        var r = new double[5];
        for (int i = 0; i < r.Length; i++) r[i] = rate();
        Array.Sort(r);
        return r[r.Length / 2];
    }

    /// <summary>The copy rates by size: 64 KB up by fours to 64 MB, then doubled until the rate settles within 3 % of the size
    /// before (or the next size cannot be had); then every size measured again, the faster of its two kept (one pass at a
    /// size can meet the device busy with another process, the display's or the system's).</summary>
    internal static List<(long Size, double Rate)> Curve(Func<long, double?> rateAt, long most)
    {
        var curve = new List<(long, double)>();
        double last = 0;
        for (long size = 64L << 10; size <= most; size = size < (64L << 20) ? size * 4 : size * 2)
        {
            if (rateAt(size) is not { } r) break;
            curve.Add((size, r));
            if (size > (64L << 20) && Math.Abs(r - last) <= 0.03 * last) break;
            last = r;
        }

        for (int i = 0; i < curve.Count; i++)
        {
            if (rateAt(curve[i].Item1) is { } again && again > curve[i].Item2) curve[i] = (curve[i].Item1, again);
        }

        return curve;
    }
}

// ---------------------------------------------------------------------------------------------- the CPU

internal static unsafe class CpuLimits
{
    private static int Threads => Environment.ProcessorCount;

    public static Limits Measure()
    {
        // The copy: what the GC says this process may have, an eighth of it at most for the two buffers.
        long most = Math.Max(64L << 20, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 16);
        WarmUp(Math.Min(most, 256L << 20));
        string how = "";
        var copies = Limits.Curve(size =>
        {
            var (one, memmove, nt) = CopyRates(size);
            how = one >= memmove && one >= nt ? "memmove on one thread" : memmove >= nt ? $"memmove on {Threads} threads" : $"non-temporal vector stores on {Threads} threads";
            return one + memmove + nt == 0 ? null : Math.Max(one, Math.Max(memmove, nt));
        }, most);
        var lanes = new List<(string, double, string)>();
        string across = $"{Threads} threads, Vector<T> {Vector<byte>.Count * 8} bits";
        lanes.Add((F64.Name, Peak<double, F64>(1), across));
        lanes.Add((F32.Name, Peak<float, F32>(1), across));
        lanes.Add((I64.Name, Peak<long, I64>(1), across));
        lanes.Add((U64.Name, Peak<ulong, U64>(1), across));
        lanes.Add((I32.Name, Peak<int, I32>(1), across));
        lanes.Add((U32.Name, Peak<uint, U32>(1), across));
        if (S8Dot.IsSupported) lanes.Add((S8Dot.Name, Peak<int, S8Dot>(4), across));
        if (U8Dot.IsSupported) lanes.Add((U8Dot.Name, Peak<uint, U8Dot>(4), across));
        if (S16Dot.IsSupported) lanes.Add((S16Dot.Name, Peak<int, S16Dot>(2), across));
        if (U16Dot.IsSupported) lanes.Add((U16Dot.Name, Peak<uint, U16Dot>(2), across));
        return new Limits(Device.Cpu, copies, $"{how} past the caches", lanes);
    }

    // The copies' rates over two buffers of size bytes (bytes read plus written a second): memmove on one thread, memmove
    // and non-temporal stores on every processor, each pass the copy repeated to some 64 MB (one dispatch a pass, its cost
    // spread thin; each thread's part copied in a loop of its own); zeros when the memory cannot be had.
    private static (double One, double Memmove, double NonTemporal) CopyRates(long size)
    {
        byte* a = null, b = null;
        try
        {
            a = (byte*)NativeMemory.AlignedAlloc((nuint)size, 64);
            b = (byte*)NativeMemory.AlignedAlloc((nuint)size, 64);
        }
        catch (OutOfMemoryException)
        {
            if (a is not null) NativeMemory.AlignedFree(a);
            return (0, 0, 0);
        }

        try
        {
            NativeMemory.Fill(a, (nuint)size, 1);
            NativeMemory.Fill(b, (nuint)size, 2);
            byte* src = a, dst = b;
            int threads = Threads;
            // Each thread a contiguous part, a multiple of 4 KB.
            long part = (size / threads) & ~4095L;
            int reps = (int)Math.Max(1, (64L << 20) / size);
            double Pass(bool nonTemporal)
            {
                long start = Stopwatch.GetTimestamp();
                Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
                {
                    long from = t * part, count = t == threads - 1 ? size - from : part;
                    for (int r = 0; r < reps; r++)
                    {
                        if (nonTemporal) StreamCopy(src + from, dst + from, count);
                        else Buffer.MemoryCopy(src + from, dst + from, count, count);
                    }
                });
                return 2.0 * size * reps / Stopwatch.GetElapsedTime(start).TotalSeconds;
            }

            double One()
            {
                long start = Stopwatch.GetTimestamp();
                for (int r = 0; r < reps; r++) Buffer.MemoryCopy(src, dst, size, size);
                return 2.0 * size * reps / Stopwatch.GetElapsedTime(start).TotalSeconds;
            }

            return (Limits.Median(One), Limits.Median(() => Pass(false)), Vector.IsHardwareAccelerated ? Limits.Median(() => Pass(true)) : 0);
        }
        finally
        {
            NativeMemory.AlignedFree(a);
            NativeMemory.AlignedFree(b);
        }
    }

    // Every processor copying its part of two buffers of size bytes, batches of some 100 ms, until the rate holds.
    private static void WarmUp(long size)
    {
        byte* a = null, b = null;
        try
        {
            a = (byte*)NativeMemory.AlignedAlloc((nuint)size, 64);
            b = (byte*)NativeMemory.AlignedAlloc((nuint)size, 64);
            NativeMemory.Fill(a, (nuint)size, 1);
            NativeMemory.Fill(b, (nuint)size, 2);
            byte* src = a, dst = b;
            int threads = Threads;
            long part = (size / threads) & ~4095L;
            Limits.Warm(() =>
            {
                long start = Stopwatch.GetTimestamp(), copies = 0;
                do
                {
                    Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
                    {
                        long from = t * part, count = t == threads - 1 ? size - from : part;
                        Buffer.MemoryCopy(src + from, dst + from, count, count);
                    });
                    copies++;
                }
                while (Stopwatch.GetElapsedTime(start).TotalSeconds < 0.1);
                return 2.0 * size * copies / Stopwatch.GetElapsedTime(start).TotalSeconds;
            });
        }
        catch (OutOfMemoryException)
        {
        }
        finally
        {
            if (a is not null) NativeMemory.AlignedFree(a);
            if (b is not null) NativeMemory.AlignedFree(b);
        }
    }

    // A copy whose stores bypass the caches (the destination 64-byte aligned, the length a multiple of the vector's).
    private static void StreamCopy(byte* src, byte* dst, long count)
    {
        int w = Vector<byte>.Count;
        long i = 0, end = count - 4 * w;
        for (; i <= end; i += 4 * w)
        {
            var v0 = Vector.Load(src + i);
            var v1 = Vector.Load(src + i + w);
            var v2 = Vector.Load(src + i + 2 * w);
            var v3 = Vector.Load(src + i + 3 * w);
            Vector.StoreAlignedNonTemporal(v0, dst + i);
            Vector.StoreAlignedNonTemporal(v1, dst + i + w);
            Vector.StoreAlignedNonTemporal(v2, dst + i + 2 * w);
            Vector.StoreAlignedNonTemporal(v3, dst + i + 3 * w);
        }

        Buffer.MemoryCopy(src + i, dst + i, count - i, count - i);
    }

    private static long s_sink;

    // The lane's multiply-adds a second on every processor at once: ten independent chains of TL.Fma a thread (enough to
    // cover the instruction's latency at its issue rate on a processor of two units of four cycles; more chains than
    // registers would spill), each Fma doing Vector<T>.Count × perLane multiply-adds.
    private static double Peak<T, TL>(int perLane) where T : unmanaged where TL : struct, ILane<TL, T>
    {
        // Iterations for a pass of about 50 ms: from one thread's rate over a short run.
        long probe = 1 << 16;
        long t0 = Stopwatch.GetTimestamp();
        s_sink += Chains<T, TL>(probe);
        double secs = Math.Max(Stopwatch.GetElapsedTime(t0).TotalSeconds, 1e-6);
        long iterations = Math.Max(probe, (long)(probe * 0.05 / secs));
        int threads = Threads;
        double perPass = (double)threads * iterations * 10 * Vector<T>.Count * perLane;
        return Limits.Median(() =>
        {
            long start = Stopwatch.GetTimestamp();
            Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, _ => Interlocked.Add(ref s_sink, Chains<T, TL>(iterations)));
            return perPass / Stopwatch.GetElapsedTime(start).TotalSeconds;
        });
    }

    private static long Chains<T, TL>(long iterations) where T : unmanaged where TL : struct, ILane<TL, T>
    {
        Vector<T> c0 = new(TL.From(1)), c1 = new(TL.From(2)), c2 = new(TL.From(3)), c3 = new(TL.From(4)), c4 = new(TL.From(5));
        Vector<T> c5 = new(TL.From(6)), c6 = new(TL.From(7)), c7 = new(TL.From(8)), c8 = new(TL.From(9)), c9 = new(TL.From(10));
        if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
        {
            // c = a·b + c in each chain: small factors, so a float chain stays far from overflow (and from subnormals,
            // slow on some processors) over any pass. One fused instruction: nothing of it can move out of the loop.
            var a = new Vector<T>(TL.From(1.0 / 1024));
            var b = new Vector<T>(TL.From(1.0 / 512));
            for (long i = 0; i < iterations; i++)
            {
                c0 = TL.Fma(a, b, c0); c1 = TL.Fma(a, b, c1); c2 = TL.Fma(a, b, c2); c3 = TL.Fma(a, b, c3); c4 = TL.Fma(a, b, c4);
                c5 = TL.Fma(a, b, c5); c6 = TL.Fma(a, b, c6); c7 = TL.Fma(a, b, c7); c8 = TL.Fma(a, b, c8); c9 = TL.Fma(a, b, c9);
            }
        }
        else
        {
            // An integer lane's Fma is a multiply and an add (a dot lane's: a group's dot product): c = c·c + b, each step's
            // product of the chain's own value, so no product is fixed and moved out of the loop (values wrap: no matter).
            var b = new Vector<T>(TL.From(3));
            for (long i = 0; i < iterations; i++)
            {
                c0 = TL.Fma(c0, c0, b); c1 = TL.Fma(c1, c1, b); c2 = TL.Fma(c2, c2, b); c3 = TL.Fma(c3, c3, b); c4 = TL.Fma(c4, c4, b);
                c5 = TL.Fma(c5, c5, b); c6 = TL.Fma(c6, c6, b); c7 = TL.Fma(c7, c7, b); c8 = TL.Fma(c8, c8, b); c9 = TL.Fma(c9, c9, b);
            }
        }

        var s = TL.Add(TL.Add(TL.Add(c0, c1), TL.Add(c2, c3)), TL.Add(TL.Add(TL.Add(c4, c5), TL.Add(c6, c7)), TL.Add(c8, c9)));
        return s[0].GetHashCode();
    }
}

// ---------------------------------------------------------------------------------------------- CUDA

internal static unsafe class CudaLimits
{
    public static Limits Measure()
    {
        var gpu = Devices.Gpu;
        WarmUp();
        string how = "";
        var copies = Limits.Curve(size =>
        {
            if (Rates(gpu, size) is not var (driver, kernel)) return null;
            how = driver >= kernel ? "the driver's copy" : "a copy kernel of 16-byte words";
            return Math.Max(driver, kernel);
        }, 4L << 30);
        var lanes = new List<(string, double, string)>();
        string across = $"{gpu.Multiprocessors} multiprocessors × {gpu.ThreadsPerMultiprocessor} threads";
        foreach (var (lane, op, type, perLane) in Scalar)
        {
            if (Peak(gpu, ScalarKernel(gpu, lane, op, type), 8, perLane) is { } r) lanes.Add((lane, r, $"{op}, {across}"));
        }

        // Each tensor kind the JIT accepts, by Adad's kinds; an 8-bit kind with two shapes keeps the faster.
        foreach (var kind in new[] { Ptx.Kind.F16, Ptx.Kind.BF16, Ptx.Kind.S8, Ptx.Kind.S8K32, Ptx.Kind.U8, Ptx.Kind.U8K32, Ptx.Kind.S4, Ptx.Kind.U4 })
        {
            string lane = kind.Name + "mma";
            if (Peak(gpu, MmaKernel(gpu, kind), 1, 16 * 8 * kind.KStep / 32.0) is not { } r) continue;
            int at = lanes.FindIndex(l => l.Item1 == lane);
            string say = $"m16n8k{kind.KStep}, {across}";
            if (at < 0) lanes.Add((lane, r, say));
            else if (r > lanes[at].Item2) lanes[at] = (lane, r, say);
        }

        return new Limits(Device.Cuda, copies, $"{how} past the caches", lanes);
    }

    // The driver's copy and the copy kernel's rates over two buffers of size bytes; null when the device cannot give them.
    // The driver's copy between the largest two buffers the device gives (256 MB down to 16 MB), batches of some 100 ms,
    // until the rate holds.
    private static void WarmUp()
    {
        for (long size = 256L << 20; size >= 16L << 20; size /= 2)
        {
            if (Driver.cuMemAlloc(out ulong a, (nuint)size) != 0) continue;
            if (Driver.cuMemAlloc(out ulong b, (nuint)size) != 0) { Driver.cuMemFree(a); continue; }
            try
            {
                Gpu.Clear(a, size);
                Gpu.Clear(b, size);
                Gpu.Synchronize();
                Limits.Warm(() =>
                {
                    long start = Stopwatch.GetTimestamp(), copies = 0;
                    do
                    {
                        Gpu.Copy(a, b, size);
                        Gpu.Synchronize();
                        copies++;
                    }
                    while (Stopwatch.GetElapsedTime(start).TotalSeconds < 0.1);
                    return 2.0 * size * copies / Stopwatch.GetElapsedTime(start).TotalSeconds;
                });
                return;
            }
            finally
            {
                Driver.cuMemFree(a);
                Driver.cuMemFree(b);
            }
        }
    }

    private static (double Driver, double Kernel)? Rates(Gpu gpu, long size)
    {
        if (Driver.cuMemAlloc(out ulong a, (nuint)size) != 0) return null;
        if (Driver.cuMemAlloc(out ulong b, (nuint)size) != 0) { Driver.cuMemFree(a); return null; }
        try
        {
            Gpu.Clear(a, size);
            Gpu.Clear(b, size);
            Gpu.Synchronize();
            double driver = Limits.Median(() =>
            {
                long start = Stopwatch.GetTimestamp();
                Gpu.Copy(a, b, size);
                Gpu.Synchronize();
                return 2.0 * size / Stopwatch.GetElapsedTime(start).TotalSeconds;
            });
            var f = gpu.Function(CopyKernel(gpu), "limits_copy");
            ulong words = (ulong)(size / 16);
            uint blocks = (uint)(gpu.Multiprocessors * (gpu.ThreadsPerMultiprocessor / 256) * 4);
            double kernel = Limits.Median(() =>
            {
                long start = Stopwatch.GetTimestamp();
                Gpu.Launch(f, blocks, 1, 256, 1, 0, a, b, words);
                Gpu.Synchronize();
                return 2.0 * size / Stopwatch.GetElapsedTime(start).TotalSeconds;
            });
            return (driver, kernel);
        }
        finally
        {
            Driver.cuMemFree(a);
            Driver.cuMemFree(b);
        }
    }

    // A grid-stride copy of 16-byte words.
    private static string CopyKernel(Gpu gpu) => gpu.Target + """
        .visible .entry limits_copy(.param .u64 p_a, .param .u64 p_b, .param .u64 p_n)
        {
            .reg .pred %p<2>;
            .reg .b32 %r<8>;
            .reg .b64 %rd<10>;
            ld.param.u64 %rd1, [p_a];
            ld.param.u64 %rd2, [p_b];
            ld.param.u64 %rd3, [p_n];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mov.u32 %r3, %tid.x;
            mov.u32 %r4, %nctaid.x;
            mad.lo.u32 %r5, %r1, %r2, %r3;
            cvt.u64.u32 %rd4, %r5;
            mul.wide.u32 %rd5, %r4, %r2;
        LOOP:
            setp.ge.u64 %p1, %rd4, %rd3;
            @%p1 bra DONE;
            shl.b64 %rd6, %rd4, 4;
            add.u64 %rd7, %rd1, %rd6;
            add.u64 %rd8, %rd2, %rd6;
            ld.global.cs.v4.u32 {%r1, %r2, %r3, %r6}, [%rd7];
            st.global.cs.v4.u32 [%rd8], {%r1, %r2, %r3, %r6};
            mov.u32 %r2, %ntid.x;
            add.u64 %rd4, %rd4, %rd5;
            bra LOOP;
        DONE:
            ret;
        }
        """;

    // Adad's scalar lanes on the device: the lane, its multiply-add, the register type, multiply-adds an instruction.
    private static readonly (string Lane, string Op, string Type, int PerLane)[] Scalar =
    [
        ("f64", "fma.rn.f64", "f64", 1), ("f32", "fma.rn.f32", "f32", 1),
        ("s64", "mad.lo.s64", "s64", 1), ("u64", "mad.lo.u64", "u64", 1),
        ("s32", "mad.lo.s32", "s32", 1), ("u32", "mad.lo.u32", "u32", 1),
        ("s8dp", "dp4a.s32.s32", "s32", 4), ("u8dp", "dp4a.u32.u32", "u32", 4),
    ];

    // Eight independent chains a thread, unrolled four times a trip: c = a·b + c for a float lane; c = c·c + b for an
    // integer one (mad.lo; dp4a: b + c's four bytes · c's), so ptxas can fold no product into a constant.
    private static (string Ptx, string Entry) ScalarKernel(Gpu gpu, string lane, string op, string type)
    {
        string entry = "limits_" + lane;
        bool dot = op.StartsWith("dp4a", StringComparison.Ordinal), f = type[0] == 'f';
        string one = type switch { "f64" => "0d3F50000000000000", "f32" => "0f3A800000", _ => "3" };
        var body = new StringBuilder();
        for (int u = 0; u < 4; u++)
            for (int c = 0; c < 8; c++) body.Append(f ? $"    {op} %c{c}, %a, %b, %c{c};\n" : $"    {op} %c{c}, %c{c}, %c{c}, %b;\n");
        var init = new StringBuilder();
        for (int c = 0; c < 8; c++) init.Append(f ? $"    cvt.rn.{type}.u32 %c{c}, %r1;\n" : $"    cvt.{type}.u32 %c{c}, %r1;\n");
        var sum = new StringBuilder();
        for (int c = 1; c < 8; c++) sum.Append($"    add{(f ? ".rn" : "")}.{type} %c0, %c0, %c{c};\n");
        string ptx = gpu.Target + $$"""
            .visible .entry {{entry}}(.param .u64 p_out, .param .u32 p_n)
            {
                .reg .pred %p<2>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<4>;
                .reg .{{type}} %c<8>;
                .reg .{{(dot ? "b32" : type)}} %a, %b;
                mov.u32 %r1, %tid.x;
                mov.{{(dot ? "b32" : type)}} %a, {{(dot ? "0x01020304" : one)}};
                mov.{{(dot ? "b32" : type)}} %b, {{(dot ? "0x04030201" : one)}};
            {{init}}
                ld.param.u32 %r2, [p_n];
            LOOP:
            {{body}}
                sub.u32 %r2, %r2, 1;
                setp.ne.u32 %p1, %r2, 0;
                @%p1 bra LOOP;
            {{sum}}
                mov.u32 %r3, %ctaid.x;
                mov.u32 %r4, %ntid.x;
                mad.lo.u32 %r5, %r3, %r4, %r1;
                ld.param.u64 %rd1, [p_out];
                cvta.to.global.u64 %rd1, %rd1;
                mul.wide.u32 %rd2, %r5, 8;
                add.u64 %rd1, %rd1, %rd2;
                st.global.{{type}} [%rd1], %c0;
                ret;
            }
            """;
        return (ptx, entry);
    }

    // Four independent accumulators a warp of the kind's tensor instruction, unrolled four times a trip.
    private static (string Ptx, string Entry) MmaKernel(Gpu gpu, Ptx.Kind kind)
    {
        string entry = $"limits_{kind.Tag}";
        int aRegs = kind.KStep * kind.Bits / 64, bRegs = kind.KStep * kind.Bits / 128;
        string acc = kind.IsFloat ? "f32" : "s32";
        string Regs(string p, int n, int at = 0) => string.Join(", ", Enumerable.Range(at, n).Select(i => $"%{p}{i}"));
        var body = new StringBuilder();
        for (int u = 0; u < 4; u++)
            for (int c = 0; c < 4; c++)
                body.Append($"    {kind.Mma} {{{Regs("c", 4, 4 * c)}}}, {{{Regs("a", aRegs)}}}, {{{Regs("b", bRegs)}}}, {{{Regs("c", 4, 4 * c)}}};\n");
        var init = new StringBuilder();
        for (int i = 0; i < aRegs; i++) init.Append($"    mov.b32 %a{i}, {(kind.IsFloat ? "0x3C003C00" : "0x01010101")};\n");
        for (int i = 0; i < bRegs; i++) init.Append($"    mov.b32 %b{i}, {(kind.IsFloat ? "0x14001400" : "0x01010101")};\n");
        for (int i = 0; i < 16; i++) init.Append($"    mov.b32 %c{i}, 0;\n");
        var sum = new StringBuilder();
        for (int i = 1; i < 16; i++) sum.Append($"    add.{acc} %c0, %c0, %c{i};\n");
        string ptx = gpu.Target + $$"""
            .visible .entry {{entry}}(.param .u64 p_out, .param .u32 p_n)
            {
                .reg .pred %p<2>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<4>;
                .reg .b32 %a<{{aRegs}}>;
                .reg .b32 %b<{{bRegs}}>;
                .reg .{{acc}} %c<16>;
            {{init}}
                ld.param.u32 %r2, [p_n];
            LOOP:
            {{body}}
                sub.u32 %r2, %r2, 1;
                setp.ne.u32 %p1, %r2, 0;
                @%p1 bra LOOP;
            {{sum}}
                mov.u32 %r1, %tid.x;
                mov.u32 %r3, %ctaid.x;
                mov.u32 %r4, %ntid.x;
                mad.lo.u32 %r5, %r3, %r4, %r1;
                ld.param.u64 %rd1, [p_out];
                cvta.to.global.u64 %rd1, %rd1;
                mul.wide.u32 %rd2, %r5, 4;
                add.u64 %rd1, %rd1, %rd2;
                st.global.{{acc}} [%rd1], %c0;
                ret;
            }
            """;
        return (ptx, entry);
    }

    // The kernel's multiply-adds a second over every multiprocessor's resident threads (blocks of 256): perThread is the
    // instructions a thread issues a trip (32 for the scalar chains, 16 for a warp's tensor instructions, counted a
    // thread), perLane the multiply-adds an instruction does a thread; null when the JIT refuses the kernel.
    private static double? Peak(Gpu gpu, (string Ptx, string Entry) k, int chains, double perLane)
    {
        if (!gpu.Accepts(k.Ptx, k.Entry, out _)) return null;
        var f = gpu.Function(k.Ptx, k.Entry);
        int perBlock = 256, blocks = gpu.Multiprocessors * Math.Max(1, gpu.ThreadsPerMultiprocessor / perBlock);
        int perTrip = chains == 8 ? 32 : 16;
        Driver.Check(Driver.cuMemAlloc(out ulong output, (nuint)((long)blocks * perBlock * 8)), "cuMemAlloc");
        try
        {
            double Pass(uint trips)
            {
                long start = Stopwatch.GetTimestamp();
                Gpu.Launch(f, (uint)blocks, 1, (uint)perBlock, 1, 0, output, trips);
                Gpu.Synchronize();
                return Stopwatch.GetElapsedTime(start).TotalSeconds;
            }

            // Trips for a launch of about 20 ms, from a short one's time.
            uint probe = 256;
            Pass(probe);
            double secs = Math.Max(Pass(probe), 1e-6);
            uint trips = (uint)Math.Clamp(probe * 0.02 / secs, probe, 1 << 24);
            double work = (double)blocks * perBlock * trips * perTrip * perLane;
            return Limits.Median(() => work / Pass(trips));
        }
        finally
        {
            Driver.cuMemFree(output);
        }
    }
}
