// The INumberBase<T> version as a bench subject: one implementation over T : INumberBase<T> (CpuGeneric.cs on the
// CPU, GenericPtx.cs on the device), for the types .NET has such a T for. Stored bytes of those types are the .NET
// layout, so they are used in place.
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Qiyas.Bench;

public sealed class NumberSubject : IBenchSubject
{
    public string Name => "number";

    public string? Describe(Device device) => device == Device.Cpu ? $"number: INumberBase<T>, Vector<T> where .NET has it, {Environment.ProcessorCount} threads" : null;

    public IBenchRun? Add(Device device, string type, byte[] a, byte[] b, byte[] c, int n) => type switch
    {
        "float64" => Add<double>(device, a, b, c, n),
        "float32" => Add<float>(device, a, b, c, n),
        "float16" => Add<Half>(device, a, b, c, n),
        "int64" => Add<long>(device, a, b, c, n),
        "int32" => Add<int>(device, a, b, c, n),
        "int16" => Add<short>(device, a, b, c, n),
        "int8" => Add<sbyte>(device, a, b, c, n),
        "uint64" => Add<ulong>(device, a, b, c, n),
        "uint32" => Add<uint>(device, a, b, c, n),
        "uint16" => Add<ushort>(device, a, b, c, n),
        "uint8" => Add<byte>(device, a, b, c, n),
        _ => null,
    };

    public IBenchRun? MatMul(Device device, string type, byte[] a, byte[] b, byte[] c, int m, int n, int k) => type switch
    {
        "float64" => MatMul<double>(device, a, b, c, m, n, k),
        "float32" => MatMul<float>(device, a, b, c, m, n, k),
        "float16" => MatMul<Half>(device, a, b, c, m, n, k),
        "int64" => MatMul<long>(device, a, b, c, m, n, k),
        "int32" => MatMul<int>(device, a, b, c, m, n, k),
        "int16" => MatMul<short>(device, a, b, c, m, n, k),
        "int8" => MatMul<sbyte>(device, a, b, c, m, n, k),
        "uint64" => MatMul<ulong>(device, a, b, c, m, n, k),
        "uint32" => MatMul<uint>(device, a, b, c, m, n, k),
        "uint16" => MatMul<ushort>(device, a, b, c, m, n, k),
        "uint8" => MatMul<byte>(device, a, b, c, m, n, k),
        _ => null,
    };

    public IBenchRun? Softmax(Device device, string type, byte[] x, byte[] y, int rows, int cols) => type switch
    {
        "float64" => Rows<double>(device, x, y, rows, cols, null),
        "float32" => Rows<float>(device, x, y, rows, cols, null),
        "float16" => Rows<Half>(device, x, y, rows, cols, null),
        _ => null,
    };

    public IBenchRun? RmsNorm(Device device, string type, byte[] x, byte[] y, int rows, int cols, double eps) => type switch
    {
        "float64" => Rows<double>(device, x, y, rows, cols, (float)eps),
        "float32" => Rows<float>(device, x, y, rows, cols, (float)eps),
        "float16" => Rows<Half>(device, x, y, rows, cols, (float)eps),
        _ => null,
    };

    private static T[] Of<T>(byte[] bytes) where T : unmanaged => MemoryMarshal.Cast<byte, T>(bytes.AsSpan()).ToArray();

    private static void Back<T>(T[] values, byte[] bytes) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(bytes);

    private static IBenchRun Add<T>(Device device, byte[] x, byte[] y, byte[] z, int n) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), b = Of<T>(y), c = new T[n];
            return new BenchRun(() => CpuGeneric.Add(a, b, c, n), () => Back(c, z), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar");
        }

        var gpu = Devices.Gpu;
        ulong da = gpu.Rent(x.Length), db = gpu.Rent(y.Length), dc = gpu.Rent(z.Length);
        Gpu.Upload<byte>(x, da);
        Gpu.Upload<byte>(y, db);
        var f = gpu.Function(GenericPtx.Add<T>(out string entry), entry);
        return new BenchRun(() => { Gpu.Launch(f, (uint)((n + 255) / 256), 1, 256, 1, 0, da, db, dc, Gpu.U(n)); Gpu.Synchronize(); },
            () => Gpu.Download<byte>(dc, z), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => entry);
    }

    private static IBenchRun MatMul<T>(Device device, byte[] x, byte[] y, byte[] z, int m, int n, int k) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), b = Of<T>(y), c = new T[m * n];
            return new BenchRun(() => CpuGeneric.MatMul(a, b, c, m, n, k), () => Back(c, z), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar");
        }

        var gpu = Devices.Gpu;
        ulong da = gpu.Rent(x.Length), db = gpu.Rent(y.Length), dc = gpu.Rent(z.Length);
        Gpu.Upload<byte>(x, da);
        Gpu.Upload<byte>(y, db);
        // The product's 64 and 128 tiles both measured once, the faster kept.
        double best = double.MaxValue;
        string chosen = "";
        IntPtr function = IntPtr.Zero;
        int tile = 64;
        foreach (int micro in new[] { 8, 4 })
        {
            var f = gpu.Function(GenericPtx.MatMul<T>(micro, out string entry), entry);
            int t = 16 * micro;
            void Launch() => Gpu.Launch(f, (uint)((n + t - 1) / t), (uint)((m + t - 1) / t), 16, 16, 0, da, db, dc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Launch();
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) Launch();
            Gpu.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            if (ms < best) (best, chosen, function, tile) = (ms, entry, f, t);
        }

        return new BenchRun(() => { Gpu.Launch(function, (uint)((n + tile - 1) / tile), (uint)((m + tile - 1) / tile), 16, 16, 0, da, db, dc, Gpu.U(m), Gpu.U(n), Gpu.U(k)); Gpu.Synchronize(); },
            () => Gpu.Download<byte>(dc, z), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => chosen);
    }

    private static IBenchRun Rows<T>(Device device, byte[] x, byte[] y, int rows, int cols, float? eps) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), d = new T[rows * cols], inv = new T[rows];
            T e = T.CreateTruncating(eps ?? 0);
            return new BenchRun(() => { if (eps is null) CpuGeneric.Softmax(a, d, rows, cols); else CpuGeneric.RmsNorm(a, d, inv, rows, cols, e); },
                () => Back(d, y), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar");
        }

        var gpu = Devices.Gpu;
        ulong dx = gpu.Rent(x.Length), dy = gpu.Rent(y.Length), dinv = gpu.Rent(rows * 8L);
        Gpu.Upload<byte>(x, dx);
        string ptx = eps is null ? GenericPtx.Softmax<T>(out string entry) : GenericPtx.RmsNorm<T>(out entry);
        var g = gpu.Function(ptx, entry);
        return new BenchRun(() =>
        {
            if (eps is null) Gpu.Launch(g, (uint)rows, 1, 256, 1, 0, dx, dy, Gpu.U(cols), Gpu.F(0f));
            else Gpu.Launch(g, (uint)rows, 1, 256, 1, 0, dx, dy, dinv, Gpu.U(cols), Gpu.F(eps.Value));
            Gpu.Synchronize();
        }, () => Gpu.Download<byte>(dy, y), () => { gpu.Return(dx); gpu.Return(dy); gpu.Return(dinv); }, () => entry);
    }
}
