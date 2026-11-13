// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The INumberBase<T> version as a bench subject: one implementation over T : INumberBase<T> (CpuGeneric.cs on the
// CPU, GenericPtx.cs on CUDA, GenericSpv.cs on Vulkan: NumberVulkan.cs, the cells CUDA has), for the types .NET has
// such a T for. Stored bytes of those types are the .NET layout, so they are used in place. Its runs hold nothing
// beyond their operands but rms_norm's row of inverse norms (a value a row on the CPU, 8 bytes a row on CUDA), ctc's
// α, β and frames' log-sum-exp (per sequence), the convolution's columns and its backward's transposes,
// optimizer_many's partial sums of squares on CUDA (1024 of the moments' type, and their copy on the host) and
// rent_zeroed's buffer on CUDA, prepared with the run and said in Held; the CPU's rent_zeroed new byte[] the host meter
// counts.
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Qiyas.Bench;

public sealed class NumberSubject : IBenchSubject
{
    public string Name => "number";

    public string? Describe(Device device) => device == Device.Cpu ? $"number: INumberBase<T>, Vector<T> where .NET has it, {Environment.ProcessorCount} threads" : null;

    // The devices it has a version for: the CPU and CUDA, and Vulkan where the cell has its Vulkan version (any other
    // device would time CUDA's kernels under its name).
    private static bool Runs(Device device, bool vulkan = false) => device is Device.Cpu or Device.Cuda || (vulkan && device == Device.Vulkan);

    public IBenchRun? Add(Device device, string type, byte[] a, byte[] b, byte[] c, int n) => !Runs(device, vulkan: true) ? null : type switch
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

    public IBenchRun? MatMul(Device device, string type, byte[] a, byte[] b, byte[] c, int m, int n, int k) => !Runs(device, vulkan: true) ? null : type switch
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

    public IBenchRun? Softmax(Device device, string type, byte[] x, byte[] y, int rows, int cols) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => Rows<double>(device, x, y, rows, cols, null),
        "float32" => Rows<float>(device, x, y, rows, cols, null),
        "float16" => Rows<Half>(device, x, y, rows, cols, null),
        _ => null,
    };

    public IBenchRun? RmsNorm(Device device, string type, byte[] x, byte[] y, int rows, int cols, double eps) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => Rows<double>(device, x, y, rows, cols, (float)eps),
        "float32" => Rows<float>(device, x, y, rows, cols, (float)eps),
        "float16" => Rows<Half>(device, x, y, rows, cols, (float)eps),
        _ => null,
    };

    // One image of one group (the bench's row); images and groups are not this version's.
    public IBenchRun? Conv(Device device, string type, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) => !Runs(device, vulkan: true) || a.N != 1 || a.Groups != 1 ? null : type switch
    {
        "float64" => Conv<double>(device, a, x, w, y, dx, dw),
        "float32" => Conv<float>(device, a, x, w, y, dx, dw),
        "float16" => Conv<Half>(device, a, x, w, y, dx, dw),
        "int64" => Conv<long>(device, a, x, w, y, dx, dw),
        "int32" => Conv<int>(device, a, x, w, y, dx, dw),
        "int16" => Conv<short>(device, a, x, w, y, dx, dw),
        "int8" => Conv<sbyte>(device, a, x, w, y, dx, dw),
        "uint64" => Conv<ulong>(device, a, x, w, y, dx, dw),
        "uint32" => Conv<uint>(device, a, x, w, y, dx, dw),
        "uint16" => Conv<ushort>(device, a, x, w, y, dx, dw),
        "uint8" => Conv<byte>(device, a, x, w, y, dx, dw),
        _ => null,
    };

    // The convolution as the generic code writes it: im2col, then the product (the backward: wᵀ·dy then col2im for dx;
    // im2col, its transpose, then dy·colsᵀ for dw), every step in T. y is dy for the backward. The columns and the
    // backward's transposes are kept with the run (Held).
    private static IBenchRun? Conv<T>(Device device, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Cuda) return ConvCuda<T>(a, x, w, y, dx, dw);
        if (device == Device.Vulkan) return NumberVulkan.Conv<T>(a, x, w, y, dx, dw);
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
    public IBenchRun? Optimizer(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => Optimizer<double, double>(device, a, p, g, m, v, [n], false),
        "float32" => Optimizer<float, float>(device, a, p, g, m, v, [n], false),
        "float16" => Optimizer<Half, float>(device, a, p, g, m, v, [n], false),
        _ => null,
    };

    public IBenchRun? OptimizerMany(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int[] counts) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => Optimizer<double, double>(device, a, p, g, m, v, counts, true),
        "float32" => Optimizer<float, float>(device, a, p, g, m, v, counts, true),
        "float16" => Optimizer<Half, float>(device, a, p, g, m, v, counts, true),
        _ => null,
    };

    // The tables come as float64 for float64, float32 otherwise: the rotation is in that type (TL).
    public IBenchRun? Rope(Device device, string type, RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => Rope<double, double>(device, a, x, cos, sin, positions, y),
        "float32" => Rope<float, float>(device, a, x, cos, sin, positions, y),
        "float16" => Rope<Half, float>(device, a, x, cos, sin, positions, y),
        _ => null,
    };

    public IBenchRun? BoxLoss(Device device, string type, BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => BoxLoss<double>(device, overlap, predicted, target, dloss, output, count),
        "float32" => BoxLoss<float>(device, overlap, predicted, target, dloss, output, count),
        "float16" => BoxLoss<Half>(device, overlap, predicted, target, dloss, output, count),
        _ => null,
    };

    public IBenchRun? Ctc(Device device, string type, CtcArgs a, byte[] logits, byte[] loss, byte[]? grad) => !Runs(device, vulkan: true) ? null : type switch
    {
        "float64" => Ctc<double>(device, a, logits, loss, grad),
        "float32" => Ctc<float>(device, a, logits, loss, grad),
        "float16" => Ctc<Half>(device, a, logits, loss, grad),
        _ => null,
    };

    // A range of bytes: copy (the source's range into the output's), clear (the output's), rent_zeroed (a zeroed buffer of
    // the source's length made and released; the output not written), on the arrays given (Array.Copy, Array.Clear, a new
    // byte[], which the host meter counts).
    public IBenchRun? Memory(Device device, string op, byte[] source, byte[] output, int offset)
    {
        int n = source.Length, bytes = n - 2 * offset;
        if (device == Device.Cuda) return MemoryCuda(op, source, output, offset);
        if (device == Device.Vulkan) return NumberVulkan.Memory(op, source, output, offset);
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
    private static IBenchRun? Optimizer<T, TM>(Device device, OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int[] counts, bool many)
        where T : unmanaged, INumberBase<T> where TM : unmanaged, IFloatingPointIeee754<TM>
    {
        if (device == Device.Cuda) return OptimizerCuda<T, TM>(a, p, g, m, v, counts, many);
        if (device == Device.Vulkan) return NumberVulkan.Optimizer<T, TM>(a, p, g, m, v, counts, many);
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

    private static IBenchRun? Rope<T, TL>(Device device, RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y)
        where T : unmanaged, INumberBase<T> where TL : unmanaged, IFloatingPointIeee754<TL>
    {
        if (device == Device.Cuda) return RopeCuda<T, TL>(a, x, cos, sin, positions, y);
        if (device == Device.Vulkan) return NumberVulkan.Rope<T, TL>(a, x, cos, sin, positions, y);
        T[] xa = Of<T>(x), ya = new T[xa.Length];
        TL[] ca = Of<TL>(cos), sa = Of<TL>(sin);
        string lane = !a.Interleaved && typeof(T) == typeof(TL) && CpuGeneric.Vectorized<TL>() ? $"Vector<{typeof(TL).Name}>" : $"scalar, in {typeof(TL).Name}";
        return new BenchRun(() => CpuGeneric.Rope(a, xa, ca, sa, positions, ya), () => Back(ya, y), lane: () => lane, held: () => new(0, null));
    }

    private static IBenchRun? BoxLoss<T>(Device device, BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (device == Device.Cuda) return BoxLossCuda<T>(overlap, predicted, target, dloss, output, count);
        if (device == Device.Vulkan) return NumberVulkan.BoxLoss<T>(overlap, predicted, target, dloss, output, count);
        T[] pa = Of<T>(predicted), ta = Of<T>(target), ya = new T[output.Length / Marshal.SizeOf<T>()];
        T[]? da = dloss is null ? null : Of<T>(dloss);
        return new BenchRun(() => CpuGeneric.BoxLoss(overlap, pa, ta, da, ya, count), () => Back(ya, output), lane: () => "scalar", held: () => new(0, null));
    }

    // A sequence per Parallel.For iteration, its α (and β: frames × states) and its frames' log-sum-exp kept with the run.
    private static IBenchRun? Ctc<T>(Device device, CtcArgs a, byte[] logits, byte[] loss, byte[]? grad) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var offsets = new int[a.B];
        for (int b = 1; b < a.B; b++) offsets[b] = offsets[b - 1] + a.LabelLengths[b - 1];
        if (device == Device.Cuda) return CtcCuda<T>(a, offsets, logits, loss, grad);
        if (device == Device.Vulkan) return NumberVulkan.Ctc<T>(a, offsets, logits, loss, grad);
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

    // ---------------------------------------------------------------- CUDA's versions (GenericPtx's kernels)

    // The range of bytes on device buffers, addresses into them: cuMemcpyDtoD, cuMemsetD8; the rent by cuMemAlloc,
    // cuMemsetD8 and cuMemFree (its buffer the run's own).
    private static IBenchRun? MemoryCuda(string op, byte[] source, byte[] output, int offset)
    {
        if (op is not ("copy" or "clear" or "rent_zeroed")) return null;
        int n = source.Length, bytes = n - 2 * offset;
        var gpu = Devices.Gpu;
        ulong src = gpu.Rent(n), dst = gpu.Rent(n), at = (ulong)offset;
        Gpu.Upload<byte>(source, src);
        Gpu.Upload<byte>(output, dst);
        Action run = op switch
        {
            "copy" => () => { Gpu.Copy(src + at, dst + at, bytes); Gpu.Synchronize(); },
            "clear" => () => { Gpu.Clear(dst + at, bytes); Gpu.Synchronize(); },
            _ => () =>
            {
                Driver.Check(Driver.cuMemAlloc(out ulong b, (nuint)n), "cuMemAlloc");
                Gpu.Clear(b, n);
                Gpu.Synchronize();
                Driver.Check(Driver.cuMemFree(b), "cuMemFree");
            },
        };
        string lane = op switch { "copy" => "cuMemcpyDtoD", "clear" => "cuMemsetD8", _ => "cuMemAlloc, cuMemsetD8, cuMemFree" };
        return new BenchRun(run, () => Gpu.Download<byte>(dst, output), () => { gpu.Return(src); gpu.Return(dst); }, () => lane, () => new(0, op == "rent_zeroed" ? n : 0));
    }

    // The step on the device: a thread a value (GenericPtx.Optimizer), many's sum of squares by 1024 blocks read back and
    // added on the host in TM (a sync inside the step), a launch a tensor. Held: many's partial sums (device and host).
    private static IBenchRun OptimizerCuda<T, TM>(OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int[] counts, bool many)
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
        var gpu = Devices.Gpu;
        ulong dp = gpu.Rent(p.Length), sp = gpu.Rent(p.Length), dg = gpu.Rent(g.Length);
        ulong dm = hasM ? gpu.Rent(m!.Length) : 0, sm = hasM ? gpu.Rent(m!.Length) : 0, dv = hasV ? gpu.Rent(v!.Length) : 0, sv = hasV ? gpu.Rent(v!.Length) : 0;
        ulong dpart = many ? gpu.Rent(Blocks * mSize) : 0;
        Gpu.Upload<byte>(p, sp);
        Gpu.Upload<byte>(g, dg);
        if (hasM) Gpu.Upload<byte>(m!, sm);
        if (hasV) Gpu.Upload<byte>(v!, sv);
        void Restore()
        {
            Gpu.Copy(sp, dp, p.Length);
            if (hasM) Gpu.Copy(sm, dm, m!.Length);
            if (hasV) Gpu.Copy(sv, dv, v!.Length);
        }

        Restore();
        var f = gpu.Function(GenericPtx.Optimizer<T, TM>(a.Name, out string entry), entry);
        var sq = many ? gpu.Function(GenericPtx.SumSquares<T, TM>(out string sqEntry), sqEntry) : IntPtr.Zero;
        var part = new TM[many ? Blocks : 0];
        static ulong Param(TM x) => typeof(TM) == typeof(double) ? Gpu.D(double.CreateTruncating(x)) : Gpu.F(float.CreateTruncating(x));
        void Step()
        {
            TM scale = TM.One;
            if (many)
            {
                Gpu.Launch(sq, Blocks, 1, 256, 1, 0, dg, dpart, Gpu.U(n));
                Gpu.Download<TM>(dpart, part);                                // waits for the sums
                TM sum = TM.Zero;
                foreach (TM x in part) sum += x;
                scale = Factor(sum);
            }

            for (int t = 0; t < tensors; t++)
            {
                ulong first = (ulong)at[t];
                Gpu.Launch(f, (uint)((counts[t] + 255) / 256), 1, 256, 1, 0, dp + first * (ulong)tSize, dg + first * (ulong)tSize,
                    hasM ? dm + first * (ulong)mSize : 0, hasV ? dv + first * (ulong)mSize : 0, Gpu.U(counts[t]),
                    Param(s.Lr), Param(s.B1), Param(s.B2), Param(s.Eps), Param(s.Wd), Param(s.C1), Param(s.C2), Param(s.B1c), Param(s.B2c), Param(scale));
            }

            Gpu.Synchronize();
        }

        return new BenchRun(Step, () =>
        {
            Restore();
            Step();
            Gpu.Download<byte>(dp, p);
            if (hasM) Gpu.Download<byte>(dm, m!);
            if (hasV) Gpu.Download<byte>(dv, v!);
        }, () =>
        {
            foreach (ulong b in new[] { dp, sp, dg, dm, sm, dv, sv, dpart }) if (b != 0) gpu.Return(b);
        }, () => entry, () => new(many ? Blocks * mSize : 0, many ? Blocks * mSize : 0));
    }

    private static IBenchRun RopeCuda<T, TL>(RopeArgs a, byte[] x, byte[] cos, byte[] sin, int[] positions, byte[] y)
        where T : unmanaged, INumberBase<T> where TL : unmanaged, IFloatingPointIeee754<TL>
    {
        var gpu = Devices.Gpu;
        ulong dx = gpu.Rent(x.Length), dc = gpu.Rent(cos.Length), ds = gpu.Rent(sin.Length), dpos = gpu.Rent(positions.Length * 4L), dy = gpu.Rent(y.Length);
        Gpu.Upload<byte>(x, dx);
        Gpu.Upload<byte>(cos, dc);
        Gpu.Upload<byte>(sin, ds);
        Gpu.Upload<int>(positions, dpos);
        var f = gpu.Function(GenericPtx.Rope<T, TL>(out string entry), entry);
        int n = a.Rows * a.Cols;
        return new BenchRun(() =>
        {
            Gpu.Launch(f, (uint)((n + 255) / 256), 1, 256, 1, 0, dx, dc, ds, dpos, dy, Gpu.U(n), Gpu.U(a.Cols), Gpu.U(a.Heads), Gpu.U(a.Steps), Gpu.U(a.Half),
                Gpu.U(a.Interleaved ? 1 : 0), Gpu.U(a.Inverse ? 1 : 0));
            Gpu.Synchronize();
        }, () => Gpu.Download<byte>(dy, y), () => { gpu.Return(dx); gpu.Return(dc); gpu.Return(ds); gpu.Return(dpos); gpu.Return(dy); }, () => entry, () => new(0, 0));
    }

    private static IBenchRun BoxLossCuda<T>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var gpu = Devices.Gpu;
        ulong dp = gpu.Rent(predicted.Length), dt = gpu.Rent(target.Length), dl = dloss is null ? 0 : gpu.Rent(dloss.Length), dy = gpu.Rent(output.Length);
        Gpu.Upload<byte>(predicted, dp);
        Gpu.Upload<byte>(target, dt);
        if (dloss is not null) Gpu.Upload<byte>(dloss, dl);
        var f = gpu.Function(GenericPtx.BoxLoss<T>(overlap, dloss is not null, out string entry), entry);
        return new BenchRun(() =>
        {
            Gpu.Launch(f, (uint)((count + 255) / 256), 1, 256, 1, 0, dp, dt, dl, dy, Gpu.U(count));
            Gpu.Synchronize();
        }, () => Gpu.Download<byte>(dy, output), () => { gpu.Return(dp); gpu.Return(dt); if (dl != 0) gpu.Return(dl); gpu.Return(dy); }, () => entry, () => new(0, 0));
    }

    // A block a sequence (GenericPtx.Ctc), the scratch α (β) of T × the most states and the log-sum-exp of T frames a sequence.
    private static IBenchRun CtcCuda<T>(CtcArgs a, int[] offsets, byte[] logits, byte[] loss, byte[]? grad) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var gpu = Devices.Gpu;
        int states = 2 * a.LabelLengths.Max() + 1;
        long lseBytes = (long)a.B * a.T * Marshal.SizeOf<T>(), alphaBytes = lseBytes * states;
        ulong dx = gpu.Rent(logits.Length), dloss = gpu.Rent(loss.Length), dgrad = grad is null ? 0 : gpu.Rent(grad.Length);
        ulong dlse = gpu.Rent(lseBytes), dalpha = gpu.Rent(alphaBytes), dbeta = grad is null ? 0 : gpu.Rent(alphaBytes);
        ulong dframes = gpu.Rent(a.B * 4L), dlengths = gpu.Rent(a.B * 4L), doffsets = gpu.Rent(a.B * 4L), dlabels = gpu.Rent(Math.Max(1, a.Labels.Length) * 4L);
        Gpu.Upload<byte>(logits, dx);
        Gpu.Upload<int>(a.InputLengths, dframes);
        Gpu.Upload<int>(a.LabelLengths, dlengths);
        Gpu.Upload<int>(offsets, doffsets);
        Gpu.Upload<int>(a.Labels, dlabels);
        var f = gpu.Function(GenericPtx.Ctc<T>(out string entry), entry);
        return new BenchRun(() =>
        {
            Gpu.Launch(f, (uint)a.B, 1, 256, 1, 0, dx, dloss, dgrad, dlse, dalpha, dbeta, dframes, dlengths, doffsets, dlabels,
                Gpu.U(a.T), Gpu.U(a.B), Gpu.U(a.C), Gpu.U(a.Blank), Gpu.U(states), Gpu.U(grad is null ? 0 : 1));
            Gpu.Synchronize();
        }, () =>
        {
            Gpu.Download<byte>(dloss, loss);
            if (grad is not null) Gpu.Download<byte>(dgrad, grad);
        }, () =>
        {
            foreach (ulong b in new[] { dx, dloss, dgrad, dlse, dalpha, dbeta, dframes, dlengths, doffsets, dlabels }) if (b != 0) gpu.Return(b);
        }, () => entry, () => new(0, lseBytes + alphaBytes * (grad is null ? 1 : 2)));
    }

    // The convolution on the device as the generic code writes it: im2col (GenericPtx.Im2col), then the product (Product);
    // the backward: wᵀ (GenericPtx.Transpose) · dy, col2im (GenericPtx.Col2im) for dx; im2col, its transpose, then dy·colsᵀ
    // for dw. The columns and the backward's transposes are the run's (Held).
    private static IBenchRun ConvCuda<T>(ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) where T : unmanaged, INumberBase<T>
    {
        int rows = a.Rows, cols = a.Cols, oc = a.OC;
        var gpu = Devices.Gpu;
        int size = Marshal.SizeOf<T>();
        var (oh, ow) = a.Out;
        ulong[] p = [Gpu.U(a.H), Gpu.U(a.W), Gpu.U(a.Kh), Gpu.U(a.Kw), Gpu.U(a.Sh), Gpu.U(a.Sw), Gpu.U(a.PadTop), Gpu.U(a.PadLeft), Gpu.U(a.Dh), Gpu.U(a.Dw), Gpu.U(oh), Gpu.U(ow)];
        var im2col = gpu.Function(GenericPtx.Im2col<T>(out string ie), ie);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong d = gpu.Rent(Math.Max(16, bytes)); rented.Add(d); return d; }
        ulong dX = Rent(x.Length), dW = Rent(w.Length), dCols = Rent((long)rows * cols * size);
        Gpu.Upload<byte>(x, dX);
        Gpu.Upload<byte>(w, dW);
        void Columns() => Gpu.Launch(im2col, (uint)(((long)rows * cols + 255) / 256), 1, 256, 1, 0, [dX, dCols, Gpu.U(rows * cols), .. p]);
        void Free() { foreach (var d in rented) gpu.Return(d); }
        if (dx is null)
        {
            ulong dY = Rent(y.Length);
            Columns();
            var product = Product<T>(gpu, dW, dCols, dY, oc, cols, rows);
            return new BenchRun(() => { Columns(); product.Launch(dW, dCols, dY, oc, cols, rows); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dY, y), Free, () => $"{ie} + {product.Entry}", () => new(0, (long)rows * cols * size));
        }

        var col2im = gpu.Function(GenericPtx.Col2im<T>(out string ce), ce);
        var transpose = gpu.Function(GenericPtx.Transpose<T>(out string te), te);
        ulong dDy = Rent(y.Length), dWt = Rent((long)rows * oc * size), dDc = Rent((long)rows * cols * size), dCt = Rent((long)cols * rows * size);
        ulong dDx = Rent(dx.Length), dDw = Rent(dw!.Length);
        Gpu.Upload<byte>(y, dDy);
        void Transpose(ulong from, ulong to, int r, int c) => Gpu.Launch(transpose, (uint)(((long)r * c + 255) / 256), 1, 256, 1, 0, from, to, Gpu.U(r), Gpu.U(c));
        Transpose(dW, dWt, oc, rows);
        Columns();
        Transpose(dCols, dCt, rows, cols);
        var dcols = Product<T>(gpu, dWt, dDy, dDc, rows, cols, oc);
        var dweights = Product<T>(gpu, dDy, dCt, dDw, oc, rows, cols);
        return new BenchRun(() =>
        {
            Transpose(dW, dWt, oc, rows);
            dcols.Launch(dWt, dDy, dDc, rows, cols, oc);                                  // dcols = wᵀ·dy
            Gpu.Launch(col2im, (uint)((a.XValues + 255) / 256), 1, 256, 1, 0, [dDc, dDx, Gpu.U((int)a.XValues), .. p]);
            Columns();
            Transpose(dCols, dCt, rows, cols);
            dweights.Launch(dDy, dCt, dDw, oc, rows, cols);                               // dw = dy·colsᵀ
            Gpu.Synchronize();
        }, () => { Gpu.Download<byte>(dDx, dx); Gpu.Download<byte>(dDw, dw); }, Free, () => $"{ie}, {te}, {ce} + {dcols.Entry}",
            () => new(0, ((long)rows * oc + 3L * rows * cols) * size));
    }

    // ---------------------------------------------------------------- every CUDA kernel of the cells, for the gate

    /// <summary>Every CUDA kernel the cells above build, each cell on each type its dispatcher takes, as (cell, type, PTX,
    /// entry). The gate loads them all (CheckCudaKernels): a row loads only the types it benches, and the rows run after
    /// the gate.</summary>
    public static IEnumerable<(string Cell, string Type, string Ptx, string Entry)> CudaKernels() =>
        Every<double>("float64").Concat(Every<float>("float32")).Concat(Every<Half>("float16"))
            .Concat(Every<long>("int64")).Concat(Every<int>("int32")).Concat(Every<short>("int16")).Concat(Every<sbyte>("int8"))
            .Concat(Every<ulong>("uint64")).Concat(Every<uint>("uint32")).Concat(Every<ushort>("uint16")).Concat(Every<byte>("uint8"))
            .Concat(Floats<double, double>("float64")).Concat(Floats<float, float>("float32")).Concat(Floats<Half, float>("float16"));

    // The cells every type takes: add, matmul (both tiles), conv (im2col, col2im, the transpose; its products matmul's).
    private static IEnumerable<(string, string, string, string)> Every<T>(string type) where T : unmanaged, INumberBase<T>
    {
        string e;
        yield return ("add", type, GenericPtx.Add<T>(out e), e);
        foreach (int micro in new[] { 8, 4 }) yield return ("matmul", type, GenericPtx.MatMul<T>(micro, out e), e);
        yield return ("conv", type, GenericPtx.Im2col<T>(out e), e);
        yield return ("conv", type, GenericPtx.Col2im<T>(out e), e);
        yield return ("conv", type, GenericPtx.Transpose<T>(out e), e);
    }

    // The float types' cells, TM their moments' and tables' type (the dispatchers' pairs).
    private static IEnumerable<(string, string, string, string)> Floats<T, TM>(string type)
        where T : unmanaged, IFloatingPointIeee754<T> where TM : unmanaged, IFloatingPointIeee754<TM>
    {
        string e;
        yield return ("softmax", type, GenericPtx.Softmax<T>(out e), e);
        yield return ("rms_norm", type, GenericPtx.RmsNorm<T>(out e), e);
        foreach (var name in OptimizerArgs.Names) yield return ("optimizer:" + name, type, GenericPtx.Optimizer<T, TM>(name, out e), e);
        yield return ("optimizer_many", type, GenericPtx.SumSquares<T, TM>(out e), e);
        yield return ("rope_table", type, GenericPtx.Rope<T, TM>(out e), e);
        foreach (var overlap in Enum.GetValues<BoxOverlap>())
            foreach (bool backward in new[] { false, true }) yield return ("box_loss", type, GenericPtx.BoxLoss<T>(overlap, backward, out e), e);
        yield return ("ctc", type, GenericPtx.Ctc<T>(out e), e);
    }

    /// <summary>The gate's check of the cells on CUDA: every kernel of CudaKernels loaded by the driver (the JIT's ptxas,
    /// and the module load's refusal of a non-ASCII text). A line per kernel that does not load; true when all do or
    /// the run has no CUDA device.</summary>
    public static bool CheckCudaKernels(IReadOnlyCollection<Device> devices)
    {
        if (!devices.Contains(Device.Cuda) || !Devices.HasCuda) return true;
        bool ok = true;
        int loaded = 0;
        foreach (var (cell, type, ptx, entry) in CudaKernels())
        {
            try
            {
                Devices.Gpu.Function(ptx, entry);
                loaded++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"check number cell {cell} {type} cuda: FAILS: {ex.Message.Split('\n')[0]}");
                ok = false;
            }
        }

        if (ok) Console.WriteLine($"check {"number cells",-28} {"cuda",-6} every cell on every type it takes: {loaded} kernels load");
        return ok;
    }

    private static T[] Of<T>(byte[] bytes) where T : unmanaged => MemoryMarshal.Cast<byte, T>(bytes.AsSpan()).ToArray();

    private static void Back<T>(T[] values, byte[] bytes) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(bytes);

    private static IBenchRun Add<T>(Device device, byte[] x, byte[] y, byte[] z, int n) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Vulkan) return NumberVulkan.Add<T>(x, y, z, n);
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

    private static IBenchRun? MatMul<T>(Device device, byte[] x, byte[] y, byte[] z, int m, int n, int k) where T : unmanaged, INumberBase<T>
    {
        if (device == Device.Vulkan) return NumberVulkan.MatMul<T>(x, y, z, m, n, k);
        if (device == Device.Cpu)
        {
            T[] a = Of<T>(x), b = Of<T>(y), c = new T[m * n];
            return new BenchRun(() => CpuGeneric.MatMul(a, b, c, m, n, k), () => Back(c, z), lane: () => CpuGeneric.Vectorized<T>() ? $"Vector<{typeof(T).Name}>" : "scalar", held: () => new(0, null));
        }

        var gpu = Devices.Gpu;
        ulong da = gpu.Rent(x.Length), db = gpu.Rent(y.Length), dc = gpu.Rent(z.Length);
        Gpu.Upload<byte>(x, da);
        Gpu.Upload<byte>(y, db);
        var product = Product<T>(gpu, da, db, dc, m, n, k);
        return new BenchRun(() => { product.Launch(da, db, dc, m, n, k); Gpu.Synchronize(); },
            () => Gpu.Download<byte>(dc, z), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => product.Entry, () => new(0, 0));
    }

    // The generic product's kernel for an m × n × k shape: its 64 and 128 tiles both measured once on these operands, the
    // faster kept.
    private sealed record GenericProduct(IntPtr Function, int Tile, string Entry)
    {
        public void Launch(ulong a, ulong b, ulong c, int m, int n, int k) =>
            Gpu.Launch(Function, (uint)((n + Tile - 1) / Tile), (uint)((m + Tile - 1) / Tile), 16, 16, 0, a, b, c, Gpu.U(m), Gpu.U(n), Gpu.U(k));
    }

    private static GenericProduct Product<T>(Gpu gpu, ulong da, ulong db, ulong dc, int m, int n, int k) where T : unmanaged, INumberBase<T>
    {
        GenericProduct? best = null;
        double bestMs = double.MaxValue;
        foreach (int micro in new[] { 8, 4 })
        {
            var p = new GenericProduct(gpu.Function(GenericPtx.MatMul<T>(micro, out string entry), entry), 16 * micro, entry);
            p.Launch(da, db, dc, m, n, k);
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) p.Launch(da, db, dc, m, n, k);
            Gpu.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            if (ms < bestMs) (bestMs, best) = (ms, p);
        }

        return best!;
    }

    private static IBenchRun Rows<T>(Device device, byte[] x, byte[] y, int rows, int cols, float? eps) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (device == Device.Vulkan) return NumberVulkan.Rows<T>(x, y, rows, cols, eps);
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
