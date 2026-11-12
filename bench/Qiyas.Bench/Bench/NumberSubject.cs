// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The INumberBase<T> version as a bench subject: one implementation over T : INumberBase<T> (CpuGeneric.cs on the
// CPU, GenericPtx.cs on CUDA; there is no Vulkan version, so a Vulkan row has no number cell), for the types .NET has
// such a T for. Stored bytes of those types are the .NET layout, so they are used in place. Its runs hold nothing
// beyond their operands but rms_norm's row of inverse norms (a value a row on the CPU, 8 bytes a row on CUDA) and ctc's
// α, β and frames' log-sum-exp (per sequence), prepared with the run and said in Held; rent_zeroed's new byte[] the host
// meter counts.
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Qiyas.Bench;

public sealed class NumberSubject : IBenchSubject
{
    public string Name => "number";

    public string? Describe(Device device) => device == Device.Cpu ? $"number: INumberBase<T>, Vector<T> where .NET has it, {Environment.ProcessorCount} threads" : null;

    // The devices it has a version for: the CPU and CUDA (any other device would time CUDA's kernels under its name).
    private static bool Runs(Device device) => device is Device.Cpu or Device.Cuda;

    public IBenchRun? Add(Device device, string type, byte[] a, byte[] b, byte[] c, int n) => !Runs(device) ? null : type switch
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

    public IBenchRun? MatMul(Device device, string type, byte[] a, byte[] b, byte[] c, int m, int n, int k) => !Runs(device) ? null : type switch
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

    public IBenchRun? Softmax(Device device, string type, byte[] x, byte[] y, int rows, int cols) => !Runs(device) ? null : type switch
    {
        "float64" => Rows<double>(device, x, y, rows, cols, null),
        "float32" => Rows<float>(device, x, y, rows, cols, null),
        "float16" => Rows<Half>(device, x, y, rows, cols, null),
        _ => null,
    };

    public IBenchRun? RmsNorm(Device device, string type, byte[] x, byte[] y, int rows, int cols, double eps) => !Runs(device) ? null : type switch
    {
        "float64" => Rows<double>(device, x, y, rows, cols, (float)eps),
        "float32" => Rows<float>(device, x, y, rows, cols, (float)eps),
        "float16" => Rows<Half>(device, x, y, rows, cols, (float)eps),
        _ => null,
    };

    // The rows below have a CPU version only so far (the device's comes with GenericPtx's kernels for them).

    // One image of one group (the bench's row); images and groups are not this version's.
    public IBenchRun? Conv(Device device, string type, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) => device != Device.Cpu || a.N != 1 || a.Groups != 1 ? null : type switch
    {
        "float64" => Conv<double>(a, x, w, y, dx, dw),
        "float32" => Conv<float>(a, x, w, y, dx, dw),
        "float16" => Conv<Half>(a, x, w, y, dx, dw),
        "int64" => Conv<long>(a, x, w, y, dx, dw),
        "int32" => Conv<int>(a, x, w, y, dx, dw),
        "int16" => Conv<short>(a, x, w, y, dx, dw),
        "int8" => Conv<sbyte>(a, x, w, y, dx, dw),
        "uint64" => Conv<ulong>(a, x, w, y, dx, dw),
        "uint32" => Conv<uint>(a, x, w, y, dx, dw),
        "uint16" => Conv<ushort>(a, x, w, y, dx, dw),
        "uint8" => Conv<byte>(a, x, w, y, dx, dw),
        _ => null,
    };

    // The convolution as the generic code writes it: im2col, then the product (the backward: wᵀ·dy then col2im for dx;
    // im2col, its transpose, then dy·colsᵀ for dw), every step in T. y is dy for the backward. The columns and the
    // backward's transposes are kept with the run (Held).
    private static IBenchRun Conv<T>(ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) where T : unmanaged, INumberBase<T>
    {
        int rows = a.Rows, cols = a.Cols, oc = a.OC, size = Marshal.SizeOf<T>();
        T[] xs = Of<T>(x), ws = Of<T>(w), c = new T[rows * cols];
        string Lane() => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar";
        if (dx is null)
        {
            var ys = new T[oc * cols];
            return new BenchRun(() => { CpuGeneric.Im2col(xs, c, a); CpuGeneric.MatMul(ws, c, ys, oc, cols, rows); }, () => Back(ys, y), lane: Lane,
                held: () => new((long)rows * cols * size, null));
        }

        T[] dys = Of<T>(y), wt = new T[rows * oc], ct = new T[cols * rows], dc = new T[rows * cols], gx = new T[a.XValues], gw = new T[oc * rows];
        return new BenchRun(() =>
        {
            CpuGeneric.Transpose(ws, wt, oc, rows);
            CpuGeneric.MatMul(wt, dys, dc, rows, cols, oc);                              // dcols = wᵀ·dy
            CpuGeneric.Col2im(dc, gx, a);
            CpuGeneric.Im2col(xs, c, a);
            CpuGeneric.Transpose(c, ct, rows, cols);
            CpuGeneric.MatMul(dys, ct, gw, oc, rows, cols);                              // dw = dy·colsᵀ
        }, () => { Back(gx, dx); Back(gw, dw!); }, lane: Lane, held: () => new(((long)rows * oc + 3L * rows * cols) * size, null));
    }

    // The moments come as float64 for float64, float32 otherwise: the step is in that type (TM).
    public IBenchRun? Optimizer(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) => device != Device.Cpu ? null : type switch
    {
        "float64" => Optimizer<double, double>(a, p, g, m, v, [n], false),
        "float32" => Optimizer<float, float>(a, p, g, m, v, [n], false),
        "float16" => Optimizer<Half, float>(a, p, g, m, v, [n], false),
        _ => null,
    };

    public IBenchRun? OptimizerMany(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int[] counts) => device != Device.Cpu ? null : type switch
    {
        "float64" => Optimizer<double, double>(a, p, g, m, v, counts, true),
        "float32" => Optimizer<float, float>(a, p, g, m, v, counts, true),
        "float16" => Optimizer<Half, float>(a, p, g, m, v, counts, true),
        _ => null,
    };

    // The tables come as float64 for float64, float32 otherwise: the rotation is in that type (TL).
    public IBenchRun? Rope(Device device, string type, RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y) => device != Device.Cpu ? null : type switch
    {
        "float64" => Rope<double, double>(a, x, cos, sin, positions, y),
        "float32" => Rope<float, float>(a, x, cos, sin, positions, y),
        "float16" => Rope<Half, float>(a, x, cos, sin, positions, y),
        _ => null,
    };

    public IBenchRun? BoxLoss(Device device, string type, BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count) => device != Device.Cpu ? null : type switch
    {
        "float64" => BoxLoss<double>(overlap, predicted, target, dloss, output, count),
        "float32" => BoxLoss<float>(overlap, predicted, target, dloss, output, count),
        "float16" => BoxLoss<Half>(overlap, predicted, target, dloss, output, count),
        _ => null,
    };

    public IBenchRun? Ctc(Device device, string type, CtcArgs a, byte[] logits, byte[] loss, byte[]? grad) => device != Device.Cpu ? null : type switch
    {
        "float64" => Ctc<double>(a, logits, loss, grad),
        "float32" => Ctc<float>(a, logits, loss, grad),
        "float16" => Ctc<Half>(a, logits, loss, grad),
        _ => null,
    };

    // A range of bytes: copy (the source's range into the output's), clear (the output's), rent_zeroed (a zeroed buffer of
    // the source's length made and released; the output not written), on the arrays given (Array.Copy, Array.Clear, a new
    // byte[], which the host meter counts).
    public IBenchRun? Memory(Device device, string op, byte[] source, byte[] output, int offset)
    {
        int n = source.Length, bytes = n - 2 * offset;
        if (device != Device.Cpu) return null;
        return op switch
        {
            "copy" => new BenchRun(() => Array.Copy(source, offset, output, offset, bytes), lane: () => "Array.Copy", held: () => new(0, null)),
            "clear" => new BenchRun(() => Array.Clear(output, offset, bytes), lane: () => "Array.Clear", held: () => new(0, null)),
            "rent_zeroed" => new BenchRun(() => GC.KeepAlive(new byte[n]), lane: () => "new byte[]", held: () => new(0, null)),
            _ => null,
        };
    }

    // An optimizer step over the tensors of counts (one: the single step; many: every gradient clipped together, by
    // min(1, MaxNorm / √Σ g²), the sum in TM). A run is the step alone, from the state the runs before it left; Collect
    // restores p, m, v to the row's state, takes one step and reads them back, so the scored p is one step from it.
    private static IBenchRun Optimizer<T, TM>(OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int[] counts, bool many)
        where T : unmanaged, INumberBase<T> where TM : unmanaged, IFloatingPointIeee754<TM>
    {
        var s = CpuGeneric.Step<TM>.Of(a);
        int tensors = counts.Length;
        var at = new int[tensors + 1];
        for (int i = 0; i < tensors; i++) at[i + 1] = at[i] + counts[i];
        int n = at[tensors];
        bool hasM = a.Moments > 0 && m is not null, hasV = a.Moments > 1 && v is not null;
        TM Factor(TM sum) => a.MaxNorm > 0 && sum > TM.Zero ? TM.Min(TM.One, TM.CreateTruncating(a.MaxNorm) / TM.Sqrt(sum)) : TM.One;
        T[] pa = Of<T>(p), ga = Of<T>(g), p0 = (T[])pa.Clone();
        TM[]? ma = hasM ? Of<TM>(m!) : null, va = hasV ? Of<TM>(v!) : null, m0 = (TM[]?)ma?.Clone(), v0 = (TM[]?)va?.Clone();
        void StepArrays()
        {
            TM scale = many ? Factor(CpuGeneric.SumSquares<T, TM>(ga, n)) : TM.One;
            for (int t = 0; t < tensors; t++) CpuGeneric.Optimizer(s, pa, ga, ma, va, at[t], counts[t], scale);
        }

        string lane = typeof(T) == typeof(TM) && CpuGeneric.Vectorized<TM>() ? $"Vector<{typeof(TM).Name}>" : $"scalar, in {typeof(TM).Name}";
        return new BenchRun(StepArrays, () =>
        {
            p0.CopyTo(pa, 0);
            m0?.CopyTo(ma!, 0);
            v0?.CopyTo(va!, 0);
            StepArrays();
            Back(pa, p);
            if (ma is not null) Back(ma, m!);
            if (va is not null) Back(va, v!);
        }, lane: () => lane, held: () => new(0, null));
    }

    private static IBenchRun Rope<T, TL>(RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y)
        where T : unmanaged, INumberBase<T> where TL : unmanaged, IFloatingPointIeee754<TL>
    {
        T[] xa = Of<T>(x), ya = new T[xa.Length];
        TL[] ca = Of<TL>(cos), sa = Of<TL>(sin);
        string lane = !a.Interleaved && typeof(T) == typeof(TL) && CpuGeneric.Vectorized<TL>() ? $"Vector<{typeof(TL).Name}>" : $"scalar, in {typeof(TL).Name}";
        return new BenchRun(() => CpuGeneric.Rope(a, xa, ca, sa, positions, ya), () => Back(ya, y), lane: () => lane, held: () => new(0, null));
    }

    private static IBenchRun BoxLoss<T>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T[] pa = Of<T>(predicted), ta = Of<T>(target), ya = new T[output.Length / Marshal.SizeOf<T>()];
        T[]? da = dloss is null ? null : Of<T>(dloss);
        return new BenchRun(() => CpuGeneric.BoxLoss(overlap, pa, ta, da, ya, count), () => Back(ya, output), lane: () => "scalar", held: () => new(0, null));
    }

    // A sequence per Parallel.For iteration, its α (and β: frames × states) and its frames' log-sum-exp kept with the run.
    private static IBenchRun Ctc<T>(CtcArgs a, byte[] logits, byte[] loss, byte[]? grad) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var offsets = new int[a.B];
        for (int b = 1; b < a.B; b++) offsets[b] = offsets[b - 1] + a.LabelLengths[b - 1];
        T[] x = Of<T>(logits), la = new T[a.B];
        T[]? ga = grad is null ? null : new T[a.Values];
        T[][] alpha = new T[a.B][], lse = new T[a.B][];
        T[][]? beta = grad is null ? null : new T[a.B][];
        long kept = 0;
        for (int b = 0; b < a.B; b++)
        {
            int frames = a.InputLengths[b], states = 2 * a.LabelLengths[b] + 1;
            alpha[b] = new T[frames * states];
            if (beta is not null) beta[b] = new T[frames * states];
            lse[b] = new T[frames];
            kept += (long)frames * states * (beta is null ? 1 : 2) + frames;
        }

        kept *= Marshal.SizeOf<T>();
        return new BenchRun(() => CpuGeneric.Ctc(a, offsets, x, la, ga, alpha, beta, lse), () =>
        {
            Back(la, loss);
            if (ga is not null) Back(ga, grad!);
        }, lane: () => "scalar", held: () => new(kept, null));
    }

    private static T[] Of<T>(byte[] bytes) where T : unmanaged => MemoryMarshal.Cast<byte, T>(bytes.AsSpan()).ToArray();

    private static void Back<T>(T[] values, byte[] bytes) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(bytes);

    private static IBenchRun Add<T>(Device device, byte[] x, byte[] y, byte[] z, int n) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), b = Of<T>(y), c = new T[n];
            return new BenchRun(() => CpuGeneric.Add(a, b, c, n), () => Back(c, z), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar", held: () => new(0, null));
        }

        var gpu = Devices.Gpu;
        ulong da = gpu.Rent(x.Length), db = gpu.Rent(y.Length), dc = gpu.Rent(z.Length);
        Gpu.Upload<byte>(x, da);
        Gpu.Upload<byte>(y, db);
        var f = gpu.Function(GenericPtx.Add<T>(out string entry), entry);
        return new BenchRun(() => { Gpu.Launch(f, (uint)((n + 255) / 256), 1, 256, 1, 0, da, db, dc, Gpu.U(n)); Gpu.Synchronize(); },
            () => Gpu.Download<byte>(dc, z), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => entry, () => new(0, 0));
    }

    private static IBenchRun MatMul<T>(Device device, byte[] x, byte[] y, byte[] z, int m, int n, int k) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), b = Of<T>(y), c = new T[m * n];
            return new BenchRun(() => CpuGeneric.MatMul(a, b, c, m, n, k), () => Back(c, z), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar", held: () => new(0, null));
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
            () => Gpu.Download<byte>(dc, z), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => chosen, () => new(0, 0));
    }

    private static IBenchRun Rows<T>(Device device, byte[] x, byte[] y, int rows, int cols, float? eps) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), d = new T[rows * cols], inv = new T[rows];
            T e = T.CreateTruncating(eps ?? 0);
            return new BenchRun(() => { if (eps is null) CpuGeneric.Softmax(a, d, rows, cols); else CpuGeneric.RmsNorm(a, d, inv, rows, cols, e); },
                () => Back(d, y), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar", held: () => new(eps is null ? 0 : (long)rows * Marshal.SizeOf<T>(), null));
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
        }, () => Gpu.Download<byte>(dy, y), () => { gpu.Return(dx); gpu.Return(dy); gpu.Return(dinv); }, () => entry, () => new(0, eps is null ? 0 : rows * 8L));
    }
}
