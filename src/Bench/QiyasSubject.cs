// Qiyas as a bench subject: the CPU kernels (Kernels.cs) and the device operations (GpuOps.cs) over the contract.
using System.Numerics;

namespace Qiyas.Bench;

public sealed class QiyasSubject : IBenchSubject
{
    public string Name => "qiyas";

    public string? Describe(Device device) => device == Device.Cpu
        ? $"qiyas: team of {Team.Size}, Vector<byte> {Vector<byte>.Count} lanes, dot lanes s8 {S8Dot.IsSupported} u8 {U8Dot.IsSupported} s16 {S16Dot.IsSupported} u16 {U16Dot.IsSupported}"
        : $"qiyas: {Devices.Gpu.Name}, {Devices.Gpu.Multiprocessors} multiprocessors, capability {Devices.Gpu.Capability} (as the device reports)";

    public IBenchRun? Add(Device device, string type, byte[] a, byte[] b, byte[] c, int n) => Registry.With(type, new AddVisitor(device, a, b, c, n));

    public IBenchRun? MatMul(Device device, string type, byte[] a, byte[] b, byte[] c, int m, int n, int k) => Registry.With(type, new MatMulVisitor(device, a, b, c, m, n, k));

    public IBenchRun? Softmax(Device device, string type, byte[] x, byte[] y, int rows, int cols) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RowVisitor(device, x, y, rows, cols, null)) : null;

    public IBenchRun? RmsNorm(Device device, string type, byte[] x, byte[] y, int rows, int cols, double eps) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RowVisitor(device, x, y, rows, cols, eps)) : null;

    public IBenchRun? Map(Device device, MapOp op, string type, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b)
    {
        if (op.FloatOnly && !Registry.IsFloat(type)) return null;
        if (type == "bool" && op.Name == "div") return null;                     // a divisor of 0 is half of bool's values
        return Registry.IsFloat(type)
            ? Registry.WithFloat(type, new FloatMapVisitor(device, op, x, y, z, output, n, a, b))
            : Registry.With(type, new MapVisitor(device, op, x, y, z, output, n, a, b));
    }

    private sealed class MapVisitor(Device device, MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Map<TN>(op.Name, x, y, z, output, n, a, b), lane: () => lane);
            return DeviceRun<TN>(op, x, y, z, output, n, a, b);
        }
    }

    private sealed class FloatMapVisitor(Device device, MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.MapFloat<TN>(op.Name, x, y, z, output, n, a, b), lane: () => lane);
            return DeviceRun<TN>(op, x, y, z, output, n, a, b);
        }
    }

    private static IBenchRun DeviceRun<TN>(MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) where TN : struct, IQiyasNumber<TN>
    {
        var ops = Devices.Ops;
        var gpu = ops.Gpu;
        var rented = new List<ulong>();
        ulong Up(byte[]? array)
        {
            if (array is null) return 0;
            ulong d = gpu.Rent(array.Length);
            rented.Add(d);
            Gpu.Upload<byte>(array, d);
            return d;
        }

        ulong dout = gpu.Rent(output.Length);
        rented.Add(dout);
        ulong dx = Up(x), dy = Up(y), dz = Up(z);
        if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1") Console.WriteLine($"  {op.Name}: inputs {op.Inputs} x {dx:X} y {dy:X} z {dz:X} out {dout:X} n {n} bytes {output.Length}");
        string lane = "";
        return new BenchRun(() => { lane = ops.Map<TN>(op, dx, dy, dz, dout, n, a, b); Gpu.Synchronize(); },
            () => Gpu.Download<byte>(dout, output), () => { foreach (var p in rented) gpu.Return(p); }, () => lane);
    }

    public IBenchRun? Reduce(Device device, string op, Axis axis, string type, byte[] x, int rows, int cols, byte[] output) =>
        Registry.With(type, new ReduceVisitor(device, op, axis, x, rows, cols, output));

    public IBenchRun? MapRow(Device device, MapOp op, string type, byte[] x, byte[] v, byte[] output, int rows, int cols) =>
        Registry.With(type, new MapRowVisitor(device, op, x, v, output, rows, cols));

    public IBenchRun? Transpose(Device device, string type, byte[] x, byte[] output, int rows, int cols) =>
        Registry.With(type, new TransposeVisitor(device, x, output, rows, cols));

    public IBenchRun? Gemv(Device device, string type, byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) =>
        Registry.With(type, new GemvVisitor(device, a, x, y, rows, cols, transposed));

    public IBenchRun? RowOp(Device device, RowOp op, string type, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RowOpVisitor(device, op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b)) : null;

    public IBenchRun? NormParamsBwd(Device device, string type, byte[] x, byte[] dy, byte[] output, int rows, int cols, double eps, bool centered) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new NormParamsVisitor(device, x, dy, output, rows, cols, eps, centered)) : null;

    public IBenchRun? Layout(Device device, string type, LayoutArgs k) =>
        LayoutArgs.Applies(k.Name, Registry.IsFloat(type)) ? Registry.With(type, new LayoutVisitor(device, k)) : null;

    public IBenchRun? MatMulView(Device device, string type, ProductView v, byte[] a, byte[] b, byte[] c) =>
        Registry.With(type, new ProductVisitor(device, v, a, b, c, null, null));

    public IBenchRun? MatMulBackward(Device device, string type, byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) =>
        Registry.With(type, new ProductVisitor(device, new ProductView(m, n, k, false, false), a, b, dc, da, db));

    /// <summary>A product on a view (da null) or the backward pair (da, db given: a is A, b is B, c is dC).</summary>
    private sealed class ProductVisitor(Device device, ProductView v, byte[] a, byte[] b, byte[] c, byte[]? da, byte[]? db) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            bool backward = da is not null;
            if (device == Device.Cpu)
            {
                return new BenchRun(() => lane = backward ? Kernels.MatMulBackward<TN>(a, b, c, da!, db!, v.M, v.N, v.K) : Kernels.MatMulView<TN>(v, a, b, c), lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dA = gpu.Rent(a.Length), dB = gpu.Rent(b.Length), dC = gpu.Rent(c.Length);
            Gpu.Upload<byte>(a, dA);
            Gpu.Upload<byte>(b, dB);
            ulong dDa = 0, dDb = 0;
            if (backward) { Gpu.Upload<byte>(c, dC); dDa = gpu.Rent(da!.Length); dDb = gpu.Rent(db!.Length); }
            return new BenchRun(() =>
            {
                lane = backward ? ops.MatMulBackward<TN>(dA, dB, dC, dDa, dDb, v.M, v.N, v.K) : ops.MatMulView<TN>(v, dA, dB, dC);
                Gpu.Synchronize();
            }, () => { if (backward) { Gpu.Download<byte>(dDa, da!); Gpu.Download<byte>(dDb, db!); } else Gpu.Download<byte>(dC, c); },
                () => { gpu.Return(dA); gpu.Return(dB); gpu.Return(dC); if (backward) { gpu.Return(dDa); gpu.Return(dDb); } }, () => lane);
        }
    }

    public IBenchRun? Attention(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, bool decode) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new AttentionVisitor(device, a, q, k, v, o, decode, null, null, null)) : null;

    public IBenchRun? AttentionBackward(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new AttentionVisitor(device, a, q, k, v, dO, false, dq, dk, dv)) : null;

    public IBenchRun? Optimizer(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new OptimizerVisitor(device, a, p, g, m, v, n)) : null;

    public IBenchRun? Conv(Device device, string type, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) =>
        dx is null ? Registry.With(type, new ConvVisitor(device, a, x, w, y, null, null)) : Registry.IsFloat(type) ? Registry.WithFloat(type, new ConvBackwardVisitor(device, a, x, w, y, dx, dw!)) : null;

    public IBenchRun? TopK(Device device, string type, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) =>
        Registry.With(type, new TopKVisitor(device, x, rows, cols, k, values, indices));

    public IBenchRun? Pick(Device device, string type, byte[] x, int rows, int cols, int[] index, byte[] output) =>
        Registry.With(type, new PickVisitor(device, x, rows, cols, index, output));

    public IBenchRun? Sample(Device device, string type, byte[] x, int rows, int cols, byte[] u, int[] picks) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new SampleVisitor(device, x, rows, cols, u, picks)) : null;

    public IBenchRun? Route(Device device, int[] index, int buckets, int[] counts, int[] offsets, int[] order)
    {
        if (device == Device.Cpu) return new BenchRun(() => Kernels.Route(index, buckets, counts, offsets, order), lane: () => "s32");
        var ops = Devices.Ops;
        var gpu = ops.Gpu;
        ulong di = gpu.Rent(index.Length * 4L), dc = gpu.Rent(buckets * 4L), dof = gpu.Rent((buckets + 1) * 4L), dor = gpu.Rent(index.Length * 4L);
        Gpu.Upload<int>(index, di);
        return new BenchRun(() => { ops.Route(di, index.Length, buckets, dc, dof, dor); Gpu.Synchronize(); },
            () => { Gpu.Download<int>(dc, counts); Gpu.Download<int>(dof, offsets); Gpu.Download<int>(dor, order); },
            () => { gpu.Return(di); gpu.Return(dc); gpu.Return(dof); gpu.Return(dor); }, () => "s32");
    }

    private sealed class TopKVisitor(Device device, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.TopK<TN>(x, rows, cols, k, values, indices), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dv = gpu.Rent(values.Length), di = gpu.Rent(indices.Length * 4L);
            Gpu.Upload<byte>(x, dx);
            return new BenchRun(() => { lane = ops.TopK<TN>(dx, rows, cols, k, dv, di); Gpu.Synchronize(); },
                () => { Gpu.Download<byte>(dv, values); Gpu.Download<int>(di, indices); },
                () => { gpu.Return(dx); gpu.Return(dv); gpu.Return(di); }, () => lane);
        }
    }

    private sealed class PickVisitor(Device device, byte[] x, int rows, int cols, int[] index, byte[] output) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Pick<TN>(x, rows, cols, index, output), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), di = gpu.Rent(index.Length * 4L), dout = gpu.Rent(output.Length);
            Gpu.Upload<byte>(x, dx); Gpu.Upload<int>(index, di);
            return new BenchRun(() => { lane = ops.Pick<TN>(dx, rows, cols, di, dout); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dout, output), () => { gpu.Return(dx); gpu.Return(di); gpu.Return(dout); }, () => lane);
        }
    }

    private sealed class SampleVisitor(Device device, byte[] x, int rows, int cols, byte[] u, int[] picks) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Sample<TN>(x, rows, cols, u, picks), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), du = gpu.Rent(u.Length), dp = gpu.Rent(picks.Length * 4L);
            Gpu.Upload<byte>(x, dx); Gpu.Upload<byte>(u, du);
            return new BenchRun(() => { lane = ops.Sample<TN>(dx, rows, cols, du, dp); Gpu.Synchronize(); },
                () => Gpu.Download<int>(dp, picks), () => { gpu.Return(dx); gpu.Return(du); gpu.Return(dp); }, () => lane);
        }
    }

    public IBenchRun? Recurrent(Device device, string type, CellArgs a, byte[] x, byte[] h, byte[]? c, byte[] w, byte[] u, byte[] b, byte[]? bh, byte[] hOut, byte[]? cOut,
        byte[]? dx, byte[]? dh, byte[]? dc, byte[]? dw, byte[]? du, byte[]? db, byte[]? dbh) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RecurrentVisitor(device, a, x, h, c, w, u, b, bh, hOut, cOut, dx, dh, dc, dw, du, db, dbh)) : null;

    public IBenchRun? LstmSequence(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] b, byte[] hOut, byte[] cOut,
        byte[]? dx, byte[]? dh0, byte[]? dc0, byte[]? dw, byte[]? du, byte[]? db) =>
        Registry.IsFloat(type) && !Registry.IsBlock(type) ? Registry.WithFloat(type, new SequenceVisitor(device, a, T, x, h0, c0, w, u, b, hOut, cOut, dx, dh0, dc0, dw, du, db)) : null;

    private sealed class SequenceVisitor(Device device, CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] b, byte[] hOut, byte[] cOut,
        byte[]? dx, byte[]? dh0, byte[]? dc0, byte[]? dw, byte[]? du, byte[]? db) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            bool backward = dx is not null;
            if (device == Device.Cpu)
            {
                return new BenchRun(() => lane = backward
                    ? Kernels.LstmSequenceBackward<TN>(a, T, x, h0, c0, w, u, b, hOut, cOut, dx!, dh0!, dc0!, dw!, du!, db!)
                    : Kernels.LstmSequence<TN>(a, T, x, h0, c0, w, u, b, hOut, cOut), lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            var rented = new List<ulong>();
            ulong Up(byte[]? array) { if (array is null) return 0; ulong d = gpu.Rent(array.Length + 4); rented.Add(d); Gpu.Upload<byte>(array, d); return d; }
            ulong Out(byte[]? array) { if (array is null) return 0; ulong d = gpu.Rent(array.Length + 4); rented.Add(d); return d; }
            ulong dX = Up(x), dH0 = Up(h0), dC0 = Up(c0), dW = Up(w), dU = Up(u), dB = Up(b);
            ulong dHo = backward ? Up(hOut) : Out(hOut), dCo = backward ? Up(cOut) : Out(cOut);
            ulong gx = Out(dx), gh0 = Out(dh0), gc0 = Out(dc0), gw = Out(dw), gu = Out(du), gb = Out(db);
            return new BenchRun(() =>
            {
                lane = backward ? ops.LstmSequenceBackward<TN>(a, T, dX, dH0, dC0, dW, dU, dB, dHo, dCo, gx, gh0, gc0, gw, gu, gb) : ops.LstmSequence<TN>(a, T, dX, dH0, dC0, dW, dU, dB, dHo, dCo);
                Gpu.Synchronize();
            }, () =>
            {
                if (backward) { Gpu.Download<byte>(gx, dx!); Gpu.Download<byte>(gh0, dh0!); Gpu.Download<byte>(gc0, dc0!); Gpu.Download<byte>(gw, dw!); Gpu.Download<byte>(gu, du!); Gpu.Download<byte>(gb, db!); }
                else { Gpu.Download<byte>(dHo, hOut); Gpu.Download<byte>(dCo, cOut); }
            }, () => { foreach (var p in rented) gpu.Return(p); }, () => lane);
        }
    }

    private sealed class RecurrentVisitor(Device device, CellArgs a, byte[] x, byte[] h, byte[]? c, byte[] w, byte[] u, byte[] b, byte[]? bh, byte[] hOut, byte[]? cOut,
        byte[]? dx, byte[]? dh, byte[]? dc, byte[]? dw, byte[]? du, byte[]? db, byte[]? dbh) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            bool lstm = a.Kind == "lstm", backward = dx is not null;
            if (device == Device.Cpu)
            {
                return new BenchRun(() => lane = backward
                    ? (lstm ? Kernels.LstmStepBackward<TN>(a, x, h, c!, w, u, b, hOut, cOut!, dx!, dh!, dc!, dw!, du!, db!) : Kernels.GruStepBackward<TN>(a, x, h, w, u, b, bh!, hOut, dx!, dh!, dw!, du!, db!, dbh!))
                    : (lstm ? Kernels.LstmStep<TN>(a, x, h, c!, w, u, b, hOut, cOut!) : Kernels.GruStep<TN>(a, x, h, w, u, b, bh!, hOut)), lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            var rented = new List<ulong>();
            ulong Up(byte[]? array) { if (array is null) return 0; ulong d = gpu.Rent(array.Length); rented.Add(d); Gpu.Upload<byte>(array, d); return d; }
            ulong Out(byte[]? array) { if (array is null) return 0; ulong d = gpu.Rent(array.Length); rented.Add(d); return d; }
            ulong dX = Up(x), dH = Up(h), dC = Up(c), dW = Up(w), dU = Up(u), dB = Up(b), dBh = Up(bh);
            ulong dHo = backward ? Up(hOut) : Out(hOut), dCo = backward ? Up(cOut) : Out(cOut);
            ulong gx = Out(dx), gh = Out(dh), gc = Out(dc), gw = Out(dw), gu = Out(du), gb = Out(db), gbh = Out(dbh);
            return new BenchRun(() =>
            {
                lane = backward
                    ? (lstm ? ops.LstmStepBackward<TN>(a, dX, dH, dC, dW, dU, dB, dHo, dCo, gx, gh, gc, gw, gu, gb) : ops.GruStepBackward<TN>(a, dX, dH, dW, dU, dB, dBh, dHo, gx, gh, gw, gu, gb, gbh))
                    : (lstm ? ops.LstmStep<TN>(a, dX, dH, dC, dW, dU, dB, dHo, dCo) : ops.GruStep<TN>(a, dX, dH, dW, dU, dB, dBh, dHo));
                Gpu.Synchronize();
            }, () =>
            {
                if (backward)
                {
                    Gpu.Download<byte>(gx, dx!); Gpu.Download<byte>(gh, dh!); Gpu.Download<byte>(gw, dw!); Gpu.Download<byte>(gu, du!); Gpu.Download<byte>(gb, db!);
                    if (dc is not null) Gpu.Download<byte>(gc, dc); if (dbh is not null) Gpu.Download<byte>(gbh, dbh);
                }
                else { Gpu.Download<byte>(dHo, hOut); if (cOut is not null) Gpu.Download<byte>(dCo, cOut); }
            }, () => { foreach (var p in rented) gpu.Return(p); }, () => lane);
        }
    }

    private sealed class ConvBackwardVisitor(Device device, ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.ConvBackward<TN>(a, x, w, dy, dx, dw), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dX = gpu.Rent(x.Length), dW = gpu.Rent(w.Length), dY = gpu.Rent(dy.Length), gx = gpu.Rent(dx.Length), gw = gpu.Rent(dw.Length);
            Gpu.Upload<byte>(x, dX); Gpu.Upload<byte>(w, dW); Gpu.Upload<byte>(dy, dY);
            return new BenchRun(() => { lane = ops.ConvBackward<TN>(a, dX, dW, dY, gx, gw); Gpu.Synchronize(); },
                () => { Gpu.Download<byte>(gx, dx); Gpu.Download<byte>(gw, dw); },
                () => { gpu.Return(dX); gpu.Return(dW); gpu.Return(dY); gpu.Return(gx); gpu.Return(gw); }, () => lane);
        }
    }

    private sealed class ConvVisitor(Device device, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            bool backward = dx is not null;
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Conv<TN>(a, x, w, y), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dX = gpu.Rent(x.Length), dW = gpu.Rent(w.Length), dY = gpu.Rent(y.Length);
            Gpu.Upload<byte>(x, dX); Gpu.Upload<byte>(w, dW);
            ulong gx = 0, gw = 0;
            if (backward) { Gpu.Upload<byte>(y, dY); gx = gpu.Rent(dx!.Length); gw = gpu.Rent(dw!.Length); }
            return new BenchRun(() => { lane = ops.Conv<TN>(a, dX, dW, dY); Gpu.Synchronize(); },
                () => { if (backward) { Gpu.Download<byte>(gx, dx!); Gpu.Download<byte>(gw, dw!); } else Gpu.Download<byte>(dY, y); },
                () => { gpu.Return(dX); gpu.Return(dW); gpu.Return(dY); if (backward) { gpu.Return(gx); gpu.Return(gw); } }, () => lane);
        }
    }

    private sealed class OptimizerVisitor(Device device, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            // The step changes p, m, v: each run starts from the same state (the restore is timed, as a caller's would not be: it is named).
            var p0 = (byte[])p.Clone();
            var m0 = (byte[])m.Clone();
            var v0 = v is null ? null : (byte[])v.Clone();
            if (device == Device.Cpu)
            {
                return new BenchRun(() => { p0.CopyTo(p, 0); m0.CopyTo(m, 0); v0?.CopyTo(v!, 0); lane = Kernels.Optimizer<TN>(a, p, g, m, v, n); }, lane: () => lane + "; + restore of p, m, v");
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dp = gpu.Rent(p.Length), dg = gpu.Rent(g.Length), dm = gpu.Rent(m.Length), dv = v is null ? 0 : gpu.Rent(v.Length);
            ulong sp = gpu.Rent(p.Length), sm = gpu.Rent(m.Length), sv = v is null ? 0 : gpu.Rent(v.Length);
            Gpu.Upload<byte>(p, sp); Gpu.Upload<byte>(g, dg); Gpu.Upload<byte>(m, sm);
            if (v is not null) Gpu.Upload<byte>(v, sv);
            return new BenchRun(() =>
            {
                Gpu.Copy(sp, dp, p.Length); Gpu.Copy(sm, dm, m.Length);
                if (v is not null) Gpu.Copy(sv, dv, v.Length);
                lane = ops.Optimizer<TN>(a, dp, dg, dm, dv, n);
                Gpu.Synchronize();
            }, () => { Gpu.Download<byte>(dp, p); Gpu.Download<byte>(dm, m); if (v is not null) Gpu.Download<byte>(dv, v); },
                () => { gpu.Return(dp); gpu.Return(dg); gpu.Return(dm); gpu.Return(sp); gpu.Return(sm); if (v is not null) { gpu.Return(dv); gpu.Return(sv); } },
                () => lane + "; + restore of p, m, v");
        }
    }

    private sealed class AttentionVisitor(Device device, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, bool decode, byte[]? dq, byte[]? dk, byte[]? dv) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            bool backward = dq is not null;
            if (device == Device.Cpu)
            {
                return new BenchRun(() => lane = backward ? Kernels.AttentionBackward<TN>(a, q, k, v, o, dq!, dk!, dv!) : decode ? Kernels.AttentionDecode<TN>(a, q, k, v, o) : Kernels.Attention<TN>(a, q, k, v, o), lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dQ = gpu.Rent(q.Length), dK = gpu.Rent(k.Length), dV = gpu.Rent(v.Length), dO = gpu.Rent(o.Length);
            Gpu.Upload<byte>(q, dQ); Gpu.Upload<byte>(k, dK); Gpu.Upload<byte>(v, dV);
            ulong gq = 0, gk = 0, gv = 0;
            if (backward) { Gpu.Upload<byte>(o, dO); gq = gpu.Rent(dq!.Length); gk = gpu.Rent(dk!.Length); gv = gpu.Rent(dv!.Length); }
            return new BenchRun(() =>
            {
                lane = backward ? ops.AttentionBackward<TN>(a, dQ, dK, dV, dO, gq, gk, gv) : decode ? ops.AttentionDecode<TN>(a, dQ, dK, dV, dO) : ops.Attention<TN>(a, dQ, dK, dV, dO);
                Gpu.Synchronize();
            }, () => { if (backward) { Gpu.Download<byte>(gq, dq!); Gpu.Download<byte>(gk, dk!); Gpu.Download<byte>(gv, dv!); } else Gpu.Download<byte>(dO, o); },
                () => { gpu.Return(dQ); gpu.Return(dK); gpu.Return(dV); gpu.Return(dO); if (backward) { gpu.Return(gq); gpu.Return(gk); gpu.Return(gv); } }, () => lane);
        }
    }

    private sealed class LayoutVisitor(Device device, LayoutArgs k) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Layout<TN>(k), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(k.X.Length), dout = gpu.Rent(k.Output.Length), didx = k.Index is null ? 0 : gpu.Rent(k.Index.Length * 4L), dx2 = k.X2 is null ? 0 : gpu.Rent(k.X2.Length);
            Gpu.Upload<byte>(k.X, dx);
            if (k.Index is not null) Gpu.Upload<int>(k.Index, didx);
            if (k.X2 is not null) Gpu.Upload<byte>(k.X2, dx2);
            return new BenchRun(() => { lane = ops.Layout<TN>(k, dx, didx, dout, dx2); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dout, k.Output),
                () => { gpu.Return(dx); gpu.Return(dout); if (didx != 0) gpu.Return(didx); if (dx2 != 0) gpu.Return(dx2); }, () => lane);
        }
    }

    private sealed class RowOpVisitor(Device device, RowOp op, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.RowOp<TN>(op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            var rented = new List<ulong>();
            ulong Up(byte[]? array)
            {
                if (array is null) return 0;
                ulong d = gpu.Rent(array.Length);
                rented.Add(d);
                Gpu.Upload<byte>(array, d);
                return d;
            }

            ulong dx = Up(x), dx2 = Up(x2), dg = Up(g), dbt = Up(bt), ds = Up(s), dpos = 0;
            if (pos is not null)
            {
                dpos = gpu.Rent(pos.Length * 4L);
                rented.Add(dpos);
                Gpu.Upload<int>(pos, dpos);
            }

            ulong dy = gpu.Rent(y.Length), dy2 = y2 is null ? 0 : gpu.Rent(y2.Length);
            rented.Add(dy);
            if (dy2 != 0) rented.Add(dy2);
            return new BenchRun(() => { lane = ops.RowOp<TN>(op, dx, dx2, dg, dbt, ds, dpos, dy, dy2, rows, cols, a, b); Gpu.Synchronize(); },
                () => { Gpu.Download<byte>(dy, y); if (y2 is not null) Gpu.Download<byte>(dy2, y2); },
                () => { foreach (var p in rented) gpu.Return(p); }, () => lane);
        }
    }

    private sealed class NormParamsVisitor(Device device, byte[] x, byte[] dy, byte[] output, int rows, int cols, double eps, bool centered) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = FloatLane<TN>();
            int size = TN.Bits == 64 ? 8 : 4;
            var dgamma = new byte[cols * size];
            var dbeta = new byte[cols * size];
            void Collect() { dgamma.CopyTo(output, 0); dbeta.CopyTo(output, cols * size); }
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.NormParamsBwd<TN>(x, dy, dgamma, dbeta, rows, cols, eps, centered), Collect, lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), ddy = gpu.Rent(dy.Length), dg = gpu.Rent(dgamma.Length), db = gpu.Rent(dbeta.Length);
            Gpu.Upload<byte>(x, dx);
            Gpu.Upload<byte>(dy, ddy);
            return new BenchRun(() => { lane = ops.NormParamsBwd<TN>(dx, ddy, dg, db, rows, cols, eps, centered); Gpu.Synchronize(); },
                () => { Gpu.Download<byte>(dg, dgamma); Gpu.Download<byte>(db, dbeta); Collect(); },
                () => { gpu.Return(dx); gpu.Return(ddy); gpu.Return(dg); gpu.Return(db); }, () => lane);
        }
    }

    private sealed class ReduceVisitor(Device device, string op, Axis axis, byte[] x, int rows, int cols, byte[] output) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu)
            {
                byte[]? result = null;
                return new BenchRun(() => (result, lane) = Kernels.Reduce<TN>(op, axis, x, rows, cols), () => result!.CopyTo(output.AsSpan()), lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dout = gpu.Rent(output.Length);
            Gpu.Upload<byte>(x, dx);
            return new BenchRun(() => { lane = ops.Reduce<TN>(op, axis, dx, rows, cols, dout); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dout, output), () => { gpu.Return(dx); gpu.Return(dout); }, () => lane);
        }
    }

    private sealed class MapRowVisitor(Device device, MapOp op, byte[] x, byte[] v, byte[] output, int rows, int cols) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.MapRow<TN>(op.Name, x, v, output, rows, cols), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dv = gpu.Rent(v.Length), dout = gpu.Rent(output.Length);
            Gpu.Upload<byte>(x, dx);
            Gpu.Upload<byte>(v, dv);
            return new BenchRun(() => { lane = ops.MapRow<TN>(op, dx, dv, dout, rows, cols); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dout, output), () => { gpu.Return(dx); gpu.Return(dv); gpu.Return(dout); }, () => lane);
        }
    }

    private sealed class TransposeVisitor(Device device, byte[] x, byte[] output, int rows, int cols) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Transpose<TN>(x, output, rows, cols), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dout = gpu.Rent(output.Length);
            Gpu.Upload<byte>(x, dx);
            return new BenchRun(() => { lane = ops.Transpose<TN>(dx, dout, rows, cols); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dout, output), () => { gpu.Return(dx); gpu.Return(dout); }, () => lane);
        }
    }

    private sealed class GemvVisitor(Device device, byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu) return new BenchRun(() => lane = Kernels.Gemv<TN>(a, x, y, rows, cols, transposed), lane: () => lane);
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong da = gpu.Rent(a.Length), dx = gpu.Rent(x.Length), dy = gpu.Rent(y.Length);
            Gpu.Upload<byte>(a, da);
            Gpu.Upload<byte>(x, dx);
            return new BenchRun(() => { lane = ops.Gemv<TN>(da, dx, dy, rows, cols, transposed); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dy, y), () => { gpu.Return(da); gpu.Return(dx); gpu.Return(dy); }, () => lane);
        }
    }

    // ---------------------------------------------------------------- the lanes operations take (for the report)

    private struct LaneName : ISumVisitor
    {
        public string Name;

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane> => Name = TL.Name;

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IQiyasDotLane<TD, TIn, TAcc> => Name = TD.Name;
    }

    private struct FloatLaneName : IFloatLaneVisitor
    {
        public string Name;

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane> => Name = TL.Name;
    }

    private static string ElementwiseLane<TN>() where TN : struct, IQiyasNumber<TN>
    {
        var probe = new LaneName();
        TN.Elementwise(ref probe);
        return probe.Name;
    }

    private static string FloatLane<TN>() where TN : struct, IQiyasFloat<TN>
    {
        var probe = new FloatLaneName();
        TN.Float(ref probe);
        return probe.Name;
    }

    // ---------------------------------------------------------------- the operations

    private sealed class AddVisitor(Device device, byte[] a, byte[] b, byte[] c, int n) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = ElementwiseLane<TN>();
            if (device == Device.Cpu)
            {
                return new BenchRun(() =>
                {
                    var add = new Kernels.AddVisitor(a, b, c, n);
                    TN.Elementwise(ref add);
                }, lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong da = gpu.Rent(a.Length), db = gpu.Rent(b.Length), dc = gpu.Rent(c.Length);
            Gpu.Upload<byte>(a, da);
            Gpu.Upload<byte>(b, db);
            return new BenchRun(() => { ops.Add<TN>(da, db, dc, n); Gpu.Synchronize(); },
                () => Gpu.Download<byte>(dc, c), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => lane);
        }
    }

    private sealed class MatMulVisitor(Device device, byte[] a, byte[] b, byte[] c, int m, int n, int k) : ITypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            string lane = "";
            if (device == Device.Cpu)
            {
                // The sum lane comes from the bounds measured on the data: part of the timed work.
                return new BenchRun(() =>
                {
                    var (aLow, aHigh) = TN.Range(a, m * k);
                    var (bLow, bHigh) = TN.Range(b, k * n);
                    var bounds = new Bounds(aLow, aHigh, bLow, bHigh, k);
                    var product = new Kernels.MatMulVisitor(a, b, c, m, n, k);
                    TN.Sum(ref product, in bounds);
                    lane = product.Lane;
                }, lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong da = gpu.Rent(a.Length), db = gpu.Rent(b.Length), dc = gpu.Rent(c.Length);
            Gpu.Upload<byte>(a, da);
            Gpu.Upload<byte>(b, db);
            return new BenchRun(() =>
            {
                var ((aLow, aHigh), (bLow, bHigh)) = ops.RangePair<TN>(da, m * k, db, k * n);
                var bounds = new Bounds(aLow, aHigh, bLow, bHigh, k);
                lane = ops.MatMul<TN>(da, db, dc, m, n, k, in bounds);
                Gpu.Synchronize();
            }, () => Gpu.Download<byte>(dc, c), () => { gpu.Return(da); gpu.Return(db); gpu.Return(dc); }, () => lane);
        }
    }

    private sealed class RowVisitor(Device device, byte[] x, byte[] y, int rows, int cols, double? eps) : IFloatTypeVisitor<IBenchRun?>
    {
        public IBenchRun? Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            string lane = FloatLane<TN>();
            if (device == Device.Cpu)
            {
                return new BenchRun(() =>
                {
                    if (eps is null) { var softmax = new Kernels.SoftmaxVisitor(x, y, rows, cols); TN.Float(ref softmax); }
                    else { var rms = new Kernels.RmsNormVisitor(x, y, rows, cols, eps.Value); TN.Float(ref rms); }
                }, lane: () => lane);
            }

            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dy = gpu.Rent(y.Length);
            Gpu.Upload<byte>(x, dx);
            return new BenchRun(() =>
            {
                lane = eps is null ? ops.Softmax<TN>(dx, dy, rows, cols) : ops.RmsNorm<TN>(dx, dy, rows, cols, eps.Value);
                Gpu.Synchronize();
            }, () => Gpu.Download<byte>(dy, y), () => { gpu.Return(dx); gpu.Return(dy); }, () => lane);
        }
    }
}
