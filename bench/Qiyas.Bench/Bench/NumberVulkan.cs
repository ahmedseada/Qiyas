// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The INumberBase<T> version on Vulkan: NumberSubject's cells there, each CUDA's version (NumberSubject's *Cuda, its
// GenericPtx kernels) with GenericSpv's kernels on the Vulkan device Adad runs on, the same launches, the same buffers
// held. Its gate (Check): every kernel of every cell built into a pipeline on every type its dispatcher takes, and every
// cell run once on a small case against the CPU's version of the same cell (CpuGeneric: the same T, so an integer cell
// exactly and a float cell within the float steps' rounding).
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Vulkan;

namespace Qiyas.Bench;

internal static partial class NumberVulkan
{
    private static Vulkan Vk => Devices.Vk.Vk;

    // A cell's kernel: made (SPIR-V words, cheap) and its pipeline taken by the kernel's name, built the first time.
    private static (ulong Pipeline, SpvKernel Kernel) Kernel(Vulkan.Maker make)
    {
        var k = make();
        return Vk.Cached("number:" + k.Name, () => k);
    }

    /// <summary>The device buffers of a run, each with its bytes uploaded or a size, all returned on Dispose.</summary>
    private sealed class Buffers
    {
        private readonly List<ulong> _rented = [];

        public ulong In(byte[] data)
        {
            ulong d = Rent(data.Length);
            Vk.Upload<byte>(data, d);
            return d;
        }

        public ulong Rent(long bytes)
        {
            ulong d = Vk.Rent(Math.Max(16, bytes));
            _rented.Add(d);
            return d;
        }

        public void Free()
        {
            foreach (ulong d in _rented) Vk.Return(d);
        }
    }

    public static IBenchRun Add<T>(byte[] x, byte[] y, byte[] z, int n) where T : unmanaged, INumberBase<T>
    {
        var bufs = new Buffers();
        ulong da = bufs.In(x), db = bufs.In(y), dc = bufs.Rent(z.Length);
        var f = Kernel(() => GenericSpv.Add<T>(Vk));
        var (gx, gy) = GenericSpv.Grid(Vk, n);
        return new BenchRun(() => { Vk.Launch(f, gx, gy, da, db, dc, Vulkan.U(n)); Vk.Synchronize(); },
            () => Vk.Download<byte>(dc, z), bufs.Free, () => f.Kernel.Name, () => new(0, 0));
    }

    /// <summary>Softmax (eps null) or rms_norm over rows × cols, a workgroup a row; rms_norm's inverse norms (8 bytes a row)
    /// held, as CUDA's.</summary>
    public static IBenchRun Rows<T>(byte[] x, byte[] y, int rows, int cols, float? eps) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var bufs = new Buffers();
        ulong dx = bufs.In(x), dy = bufs.Rent(y.Length), dinv = bufs.Rent(rows * 8L);
        var f = eps is null ? Kernel(() => GenericSpv.Softmax<T>(Vk)) : Kernel(() => GenericSpv.RmsNorm<T>(Vk));
        return new BenchRun(() =>
        {
            if (eps is null) Vk.Launch(f, (uint)rows, 1, dx, dy, Vulkan.U(cols));
            else Vk.Launch(f, (uint)rows, 1, dx, dy, dinv, Vulkan.U(cols), Vulkan.F(eps.Value));
            Vk.Synchronize();
        }, () => Vk.Download<byte>(dy, y), bufs.Free, () => f.Kernel.Name, () => new(0, eps is null ? 0 : rows * 8L));
    }

    /// <summary>The step on the device: a thread a value, many's sum of squares by 1024 workgroups read back and added on the
    /// host in TM (a wait inside the step), a launch a tensor (as CUDA's). Held: many's partial sums (device and host).</summary>
    public static IBenchRun Optimizer<T, TM>(OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int[] counts, bool many)
        where T : unmanaged, INumberBase<T> where TM : unmanaged, IFloatingPointIeee754<TM>
    {
        var s = CpuGeneric.Step<TM>.Of(a);
        int tensors = counts.Length;
        var at = new int[tensors + 1];
        for (int i = 0; i < tensors; i++) at[i + 1] = at[i] + counts[i];
        int n = at[tensors];
        bool hasM = a.Moments > 0 && m is not null, hasV = a.Moments > 1 && v is not null;
        TM Factor(TM sum) => a.MaxNorm > 0 && sum > TM.Zero ? TM.Min(TM.One, TM.CreateTruncating(a.MaxNorm) / TM.Sqrt(sum)) : TM.One;
        const int Blocks = 1024;
        int tSize = Marshal.SizeOf<T>(), mSize = Marshal.SizeOf<TM>();
        var bufs = new Buffers();
        ulong dp = bufs.Rent(p.Length), sp = bufs.In(p), dg = bufs.In(g);
        ulong dm = hasM ? bufs.Rent(m!.Length) : 0, sm = hasM ? bufs.In(m!) : 0, dv = hasV ? bufs.Rent(v!.Length) : 0, sv = hasV ? bufs.In(v!) : 0;
        ulong dpart = many ? bufs.Rent(Blocks * mSize) : 0;
        void Restore()
        {
            Vk.Copy(sp, dp, p.Length);
            if (hasM) Vk.Copy(sm, dm, m!.Length);
            if (hasV) Vk.Copy(sv, dv, v!.Length);
        }

        Restore();
        var f = Kernel(() => GenericSpv.Optimizer<T, TM>(Vk, a.Name));
        var sq = many ? Kernel(() => GenericSpv.SumSquares<T, TM>(Vk)) : default;
        var part = new TM[many ? Blocks : 0];
        static ulong Param(TM x) => typeof(TM) == typeof(double) ? Vulkan.D(double.CreateTruncating(x)) : Vulkan.F(float.CreateTruncating(x));
        void Step()
        {
            TM scale = TM.One;
            if (many)
            {
                Vk.Launch(sq, Blocks, 1, dg, dpart, Vulkan.U(n));
                Vk.Download<TM>(dpart, part);                                 // waits for the sums
                TM sum = TM.Zero;
                foreach (TM x in part) sum += x;
                scale = Factor(sum);
            }

            for (int t = 0; t < tensors; t++)
            {
                ulong first = (ulong)at[t];
                var (gx, gy) = GenericSpv.Grid(Vk, counts[t]);
                Vk.Launch(f, gx, gy, dp + first * (ulong)tSize, dg + first * (ulong)tSize, hasM ? dm + first * (ulong)mSize : 0, hasV ? dv + first * (ulong)mSize : 0,
                    Vulkan.U(counts[t]), Param(s.Lr), Param(s.B1), Param(s.B2), Param(s.Eps), Param(s.Wd), Param(s.C1), Param(s.C2), Param(s.B1c), Param(s.B2c), Param(scale));
            }

            Vk.Synchronize();
        }

        return new BenchRun(Step, () =>
        {
            Restore();
            Step();
            Vk.Download<byte>(dp, p);
            if (hasM) Vk.Download<byte>(dm, m!);
            if (hasV) Vk.Download<byte>(dv, v!);
        }, bufs.Free, () => f.Kernel.Name, () => new(many ? Blocks * mSize : 0, many ? Blocks * mSize : 0));
    }

    public static IBenchRun Rope<T, TL>(RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y)
        where T : unmanaged, INumberBase<T> where TL : unmanaged, IFloatingPointIeee754<TL>
    {
        var bufs = new Buffers();
        ulong dx = bufs.In(x), dc = bufs.In(cos), ds = bufs.In(sin), dpos = bufs.Rent(positions.Length * 4L), dy = bufs.Rent(y.Length);
        Vk.Upload<int>(positions, dpos);
        var f = Kernel(() => GenericSpv.Rope<T, TL>(Vk));
        int n = a.Rows * a.Cols;
        var (gx, gy) = GenericSpv.Grid(Vk, n);
        return new BenchRun(() =>
        {
            Vk.Launch(f, gx, gy, dx, dc, ds, dpos, dy, Vulkan.U(n), Vulkan.U(a.Cols), Vulkan.U(a.Heads), Vulkan.U(a.Steps), Vulkan.U(a.Half),
                Vulkan.U(a.Interleaved ? 1 : 0), Vulkan.U(a.Inverse ? 1 : 0));
            Vk.Synchronize();
        }, () => Vk.Download<byte>(dy, y), bufs.Free, () => f.Kernel.Name, () => new(0, 0));
    }

    /// <summary>A range of bytes on device buffers, addresses into them: the device's copy and fill commands; rent_zeroed a
    /// buffer made by the driver, filled and given back (Vulkan.Rent of a size no buffer is kept for, Clear, Release).</summary>
    public static IBenchRun? Memory(string op, byte[] source, byte[] output, int offset)
    {
        if (op is not ("copy" or "clear" or "rent_zeroed")) return null;
        int n = source.Length, bytes = n - 2 * offset;
        var bufs = new Buffers();
        ulong src = bufs.In(source), dst = bufs.In(output), at = (ulong)offset;
        // The device's fill takes whole words from an aligned address: clear's range is the bytes from the first whole
        // word, the ends by the copy command from a buffer of zeros (as the device's memory family does it).
        ulong zeros = bufs.Rent(256);
        Vk.Clear(zeros, 256);
        Action run = op switch
        {
            "copy" => () => { Vk.Copy(src + at, dst + at, bytes); Vk.Synchronize(); },
            "clear" => () => { Fill(dst + at, bytes, zeros); Vk.Synchronize(); },
            _ => () =>
            {
                ulong b = Vk.Rent(n);
                Vk.Clear(b, n);
                Vk.Synchronize();
                Vk.Release(b);
            },
        };
        string lane = op switch { "copy" => "vkCmdCopyBuffer", "clear" => "vkCmdFillBuffer", _ => "vkAllocateMemory, vkCmdFillBuffer, vkFreeMemory" };
        return new BenchRun(run, () => Vk.Download<byte>(dst, output), bufs.Free, () => lane, () => new(0, op == "rent_zeroed" ? n : 0));
    }

    // Zeros over [at, at + bytes): the fill command over the whole words, the copy command for the bytes before and after.
    private static void Fill(ulong at, long bytes, ulong zeros)
    {
        ulong end = at + (ulong)bytes, first = (at + 3) & ~3UL, last = end & ~3UL;
        if (first >= last) { Vk.Copy(zeros, at, bytes); return; }
        if (first > at) Vk.Copy(zeros, at, (long)(first - at));
        Vk.Clear(first, (long)(last - first));
        if (end > last) Vk.Copy(zeros, last, (long)(end - last));
    }

    // ---------------------------------------------------------------- the gate

    /// <summary>Every Vulkan kernel the cells build, each cell on each type its dispatcher takes.</summary>
    public static IEnumerable<(string Cell, string Type, Vulkan.Maker Make)> Kernels() => Own().Concat(Products());

    private static IEnumerable<(string, string, Vulkan.Maker)> Own() =>
        Every<double>("float64").Concat(Every<float>("float32")).Concat(Every<Half>("float16"))
            .Concat(Every<long>("int64")).Concat(Every<int>("int32")).Concat(Every<short>("int16")).Concat(Every<sbyte>("int8"))
            .Concat(Every<ulong>("uint64")).Concat(Every<uint>("uint32")).Concat(Every<ushort>("uint16")).Concat(Every<byte>("uint8"))
            .Concat(Floats<double, double>("float64")).Concat(Floats<float, float>("float32")).Concat(Floats<Half, float>("float16"));

    private static IEnumerable<(string, string, Vulkan.Maker)> Every<T>(string type) where T : unmanaged, INumberBase<T>
    {
        yield return ("add", type, () => GenericSpv.Add<T>(Vk));
    }

    private static IEnumerable<(string, string, Vulkan.Maker)> Floats<T, TM>(string type)
        where T : unmanaged, IFloatingPointIeee754<T> where TM : unmanaged, IFloatingPointIeee754<TM>
    {
        yield return ("softmax", type, () => GenericSpv.Softmax<T>(Vk));
        yield return ("rms_norm", type, () => GenericSpv.RmsNorm<T>(Vk));
        foreach (var name in OptimizerArgs.Names) yield return ("optimizer:" + name, type, () => GenericSpv.Optimizer<T, TM>(Vk, name));
        yield return ("optimizer_many", type, () => GenericSpv.SumSquares<T, TM>(Vk));
        yield return ("rope_table", type, () => GenericSpv.Rope<T, TM>(Vk));
    }

    /// <summary>The gate of the cells on Vulkan: every kernel of Kernels built into a pipeline (the driver's compiler takes
    /// it), then every cell of this file run once on a small case on each type its dispatcher takes, against the CPU's
    /// version of the cell on the same inputs (the outputs poisoned first, so a value the kernel did not write differs). A
    /// line per kernel that does not build and per cell that differs; true when all hold or the run has no Vulkan device.</summary>
    public static bool Check(IReadOnlyCollection<Device> devices)
    {
        if (!devices.Contains(Device.Vulkan) || !Devices.HasVulkan) return true;
        bool ok = true;
        int built = 0, cells = 0;
        foreach (var (cell, type, make) in Kernels())
        {
            try
            {
                Kernel(make);
                built++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"check number cell {cell} {type} vulkan: FAILS to build: {ex.Message.Split('\n')[0]}");
                ok = false;
            }
        }

        foreach (var (cell, type, run) in Cases().Concat(ProductCases()))
        {
            string? fault;
            try { fault = run(); }
            catch (Exception ex) { fault = ex.Message.Split('\n')[0]; }
            if (fault is null) { cells++; continue; }
            Console.WriteLine($"check number cell {cell} {type} vulkan: DIFFERS from the CPU's: {fault}");
            ok = false;
        }

        if (ok) Console.WriteLine($"check {"number cells",-28} {"vulkan",-6} every cell on every type it takes: {built} kernels build, {cells} cells agree with the CPU's");
        return ok;
    }

    private static readonly string[] Plain = ["float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"];

    // The cells' small cases: each runs the cell on the CPU and on Vulkan, each on copies of its own of the arrays the
    // cell writes (fresh: poisoned outputs, the optimizer's state as given), and says how they differ (null: they agree).
    private static IEnumerable<(string Cell, string Type, Func<string?> Run)> Cases()
    {
        var s = new NumberSubject();
        foreach (string type in Plain)
        {
            const int n = 1001;
            byte[] x = Registry.Make(type, n, 1, -50, 50), y = Registry.Make(type, n, 2, -50, 50);
            yield return ("add", type, () => Agree(d => { var z = Registry.Poisoned(type, n); return (s.Add(d, type, x, y, z, n), [(type, n, z)]); }));
        }

        foreach (string type in new[] { "float64", "float32", "float16" })
        {
            const int rows = 5, cols = 300, n = rows * cols;
            byte[] x = Registry.Make(type, n, 3, -4, 4);
            yield return ("softmax", type, () => Agree(d => { var z = Registry.Poisoned(type, n); return (s.Softmax(d, type, x, z, rows, cols), [(type, n, z)]); }));
            yield return ("rms_norm", type, () => Agree(d => { var z = Registry.Poisoned(type, n); return (s.RmsNorm(d, type, x, z, rows, cols, 1e-5), [(type, n, z)]); }));

            string lane = type == "float64" ? "float64" : "float32";
            foreach (string name in OptimizerArgs.Names)
            {
                foreach (int[] counts in new[] { new[] { 777 }, new[] { 300, 457 } })
                {
                    int total = counts.Sum();
                    bool many = counts.Length > 1;
                    var a = new OptimizerArgs(name, 1e-2, 0.9, 0.999, 1e-8, 0.01, 3) { MaxNorm = many ? 1 : 0 };
                    byte[] p = Registry.Make(type, total, 4, -1, 1), g = Registry.Make(type, total, 5, -1, 1);
                    byte[] m = Registry.Make(lane, total, 6, -0.1, 0.1), v = Registry.Make(lane, total, 7, 0, 0.01);
                    yield return ((many ? "optimizer_many:" : "optimizer:") + name, type, () => Agree(d =>
                    {
                        byte[] pc = (byte[])p.Clone(), mc = (byte[])m.Clone(), vc = (byte[])v.Clone();
                        byte[]? mm = a.Moments > 0 ? mc : null, vv = a.Moments > 1 ? vc : null;
                        var run = many ? s.OptimizerMany(d, type, a, pc, g, mm, vv, counts) : s.Optimizer(d, type, a, pc, g, mc, vv, total);
                        return (run, [(type, total, pc), (lane, total, mc), (lane, total, vc)]);
                    }));
                }
            }

            foreach (bool interleaved in new[] { false, true })
            {
                var ra = new RopeArgs(24, 70, 3, 4, 32, interleaved, Inverse: interleaved);
                int rn = ra.Rows * ra.Cols;
                byte[] xr = Registry.Make(type, rn, 8, -2, 2);
                double[] angles = [.. Enumerable.Range(0, 8 * ra.Half).Select(i => 0.37 * i)];
                byte[] cos = Registry.Encode(lane, [.. angles.Select(Math.Cos)]), sin = Registry.Encode(lane, [.. angles.Select(Math.Sin)]);
                int[] positions = [5, 0, 7, 2];
                yield return ("rope_table", type, () => Agree(d => { var z = Registry.Poisoned(type, rn); return (s.Rope(d, type, ra, xr, cos, sin, positions, z), [(type, rn, z)]); }));
            }
        }

        byte[] source = new byte[1001];
        for (int i = 0; i < source.Length; i++) source[i] = (byte)(i * 37 + 11);
        foreach (string op in new[] { "copy", "clear", "rent_zeroed" })
        {
            yield return ("memory:" + op, "uint8", () => Agree(d =>
            {
                var z = new byte[source.Length];
                Array.Fill(z, (byte)0xA5);
                return (s.Memory(d, op, source, z, 3), [("uint8", z.Length, z)]);
            }));
        }
    }

    // The cell made on the CPU and on Vulkan by make (each with arrays of its own, returned with their types and counts),
    // run once and collected; the arrays held to each other.
    private static string? Agree(Func<Device, (IBenchRun? Run, (string Type, int Count, byte[] Data)[] Outputs)> make)
    {
        var got = new (string Type, int Count, byte[] Data)[2][];
        Device[] devices = [Device.Cpu, Device.Vulkan];
        for (int k = 0; k < 2; k++)
        {
            var (run, outputs) = make(devices[k]);
            if (run is null) return $"no cell on {devices[k]}";
            using (run)
            {
                run.Run();
                run.Collect();
            }

            got[k] = outputs;
        }

        for (int i = 0; i < got[0].Length; i++)
        {
            if (Differs(got[0][i].Type, got[0][i].Count, got[0][i].Data, got[1][i].Data) is { } fault) return $"output {i}, {fault}";
        }

        return null;
    }

    // Two outputs of a type held to each other: exactly for an integer type; for a float type within the float steps'
    // rounding (the CPU's exp and order of sums are not the kernel's: f16 1e-2, f32 1e-4, f64 1e-6 of the larger of 1 and
    // the CPU's value; a kernel that misplaces or skips a value is off by far more), a NaN or an infinity only where the
    // other has the same.
    private static string? Differs(string type, int count, byte[] cpu, byte[] vulkan)
    {
        double[] a = Registry.Values(type, cpu, count), b = Registry.Values(type, vulkan, count);
        double tol = !Registry.IsFloat(type) ? 0 : type == "float16" ? 1e-2 : type == "float32" ? 1e-4 : 1e-6;
        for (int i = 0; i < count; i++)
        {
            if (a[i].Equals(b[i]) || Math.Abs(a[i] - b[i]) <= tol * Math.Max(1, Math.Abs(a[i]))) continue;
            return $"value {i}: cpu {a[i]:R}, vulkan {b[i]:R}";
        }

        return null;
    }
}
