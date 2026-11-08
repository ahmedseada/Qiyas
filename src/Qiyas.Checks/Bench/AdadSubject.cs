// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Adad as a bench subject: every operation once, over Adad's device contract (src/Adad/Abstractions/IDevice.cs), on the
// device asked for — the CPU (Kernels.Device.cs) or CUDA (GpuOps.Device.cs). A run stages the host arrays in the device's
// memory, runs the operation and waits for it (the timed part), and reads the outputs back on Collect.
using System.Numerics;

namespace Qiyas.Bench;

public sealed class AdadSubject : IBenchSubject
{
    public string Name => "adad";

    public string? Describe(Device device) => device switch
    {
        Device.Cpu => $"adad: team of {Team.Size}, Vector<byte> {Vector<byte>.Count} lanes, dot lanes s8 {S8Dot.IsSupported} u8 {U8Dot.IsSupported} s16 {S16Dot.IsSupported} u16 {U16Dot.IsSupported}",
        Device.Vulkan => $"adad: {Devices.Vk.Vk.Facts} (as the device reports)",
        _ => $"adad: {Devices.Gpu.Name}, {Devices.Gpu.Multiprocessors} multiprocessors, capability {Devices.Gpu.Capability} (as the device reports)",
    };

    // ---------------------------------------------------------------- a run on a device

    /// <summary>A run's host arrays in a device's memory: inputs uploaded as they are staged, outputs read back on Collect,
    /// every buffer returned on Dispose. Slack: the bytes past its values a sequence's buffer is given (the device's kernels
    /// address whole words; the CPU's take arrays of exactly their values).</summary>
    private sealed class Staged<TB, TI>(IDevice<TB, TI> device, int slack)
    {
        public int Slack => slack;

        private readonly List<TB> _buffers = [];
        private readonly List<TI> _indices = [];
        private readonly List<Action> _collect = [];

        public IDevice<TB, TI> D => device;

        public TB Rent(long bytes)
        {
            var b = device.Rent(bytes);
            _buffers.Add(b);
            return b;
        }

        public TB In(byte[] data, int pad = 0)
        {
            var b = Rent(data.Length + pad);
            device.Upload(data, b);
            return b;
        }

        public TB? InOrNone(byte[]? data, int pad = 0) => data is null ? default : In(data, pad);

        public TB Out(byte[] data, int pad = 0)
        {
            var b = Rent(data.Length + pad);
            Collect(b, data);
            return b;
        }

        public TB? OutOrNone(byte[]? data, int pad = 0) => data is null ? default : Out(data, pad);

        /// <summary>The buffer read back into data on Collect.</summary>
        public void Collect(TB buffer, byte[] data) => _collect.Add(() => device.Download(buffer, data));

        /// <summary>A step of Collect after the buffers before it are read back.</summary>
        public void Then(Action after) => _collect.Add(after);

        public TI Index(int[] data)
        {
            var i = device.RentIndices(data.Length);
            _indices.Add(i);
            device.UploadIndices(data, i);
            return i;
        }

        public TI? IndexOrNone(int[]? data) => data is null ? default : Index(data);

        public TI IndexOut(int[] data)
        {
            var i = device.RentIndices(data.Length);
            _indices.Add(i);
            _collect.Add(() => device.DownloadIndices(i, data));
            return i;
        }

        public IBenchRun Run(Func<string> op, string note = "")
        {
            string lane = "";
            return new BenchRun(() => { lane = op(); device.Synchronize(); }, () => { foreach (var c in _collect) c(); },
                () => { foreach (var b in _buffers) device.Return(b); foreach (var i in _indices) device.ReturnIndices(i); }, () => lane + note);
        }
    }

    // An operation over any type, or over a float type, made on the device's buffers once the type is known.
    private interface INumberRun
    {
        IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>;
    }

    private interface IFloatRun
    {
        IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>;
    }

    private static IBenchRun On<TN, TR>(Device device, TR run) where TN : struct, IAdad<TN, float> where TR : INumberRun =>
        device == Device.Cpu ? run.Make<TN, byte[], int[]>(new(CpuDevice.Device, 0)) : run.Make<TN, ulong, ulong>(new(Devices.Of(device), 4));

    private static IBenchRun OnFloat<TN, TR>(Device device, TR run) where TN : struct, IFloatAdad<TN>, IAdad<TN, float> where TR : IFloatRun =>
        device == Device.Cpu ? run.Make<TN, byte[], int[]>(new(CpuDevice.Device, 0)) : run.Make<TN, ulong, ulong>(new(Devices.Of(device), 4));

    // ---------------------------------------------------------------- the operations

    public IBenchRun? Add(Device device, string type, byte[] a, byte[] b, byte[] c, int n) => Registry.With(type, new AddVisitor(device, a, b, c, n));

    private sealed class AddVisitor(Device device, byte[] a, byte[] b, byte[] c, int n) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, AddVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            string lane = ElementwiseLane<TN>();
            TB da = s.In(a), db = s.In(b), dc = s.Out(c);
            return s.Run(() => { s.D.Add<TN>(da, db, dc, n); return lane; });
        }
    }

    public IBenchRun? MatMul(Device device, string type, byte[] a, byte[] b, byte[] c, int m, int n, int k) => Registry.With(type, new MatMulVisitor(device, a, b, c, m, n, k));

    // The sum lane comes from the bounds measured on the data: part of the timed work.
    private sealed class MatMulVisitor(Device device, byte[] a, byte[] b, byte[] c, int m, int n, int k) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, MatMulVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB da = s.In(a), db = s.In(b), dc = s.Out(c);
            return s.Run(() => s.D.MatMul<TN>(da, db, dc, m, n, k));
        }
    }

    public IBenchRun? Softmax(Device device, string type, byte[] x, byte[] y, int rows, int cols) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RowVisitor(device, x, y, rows, cols, null)) : null;

    public IBenchRun? RmsNorm(Device device, string type, byte[] x, byte[] y, int rows, int cols, double eps) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RowVisitor(device, x, y, rows, cols, eps)) : null;

    private sealed class RowVisitor(Device device, byte[] x, byte[] y, int rows, int cols, double? eps) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, RowVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dx = s.In(x), dy = s.Out(y);
            return s.Run(() => eps is null ? s.D.Softmax<TN>(dx, dy, rows, cols) : s.D.RmsNorm<TN>(dx, dy, rows, cols, eps.Value));
        }
    }

    public IBenchRun? Map(Device device, MapOp op, string type, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b)
    {
        if (op.FloatOnly && !Registry.IsFloat(type)) return null;
        if (type == "bool" && op.Name == "div") return null;                     // a divisor of 0 is half of bool's values
        return Registry.IsFloat(type)
            ? Registry.WithFloat(type, new FloatMapVisitor(device, op, x, y, z, output, n, a, b))
            : Registry.With(type, new MapVisitor(device, op, x, y, z, output, n, a, b));
    }

    private sealed class MapVisitor(Device device, MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, MapVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dout = s.Out(output);
            TB? dx = s.InOrNone(x), dy = s.InOrNone(y), dz = s.InOrNone(z);
            return s.Run(() => s.D.Map<TN>(op, dx, dy, dz, dout, n, a, b));
        }
    }

    private sealed class FloatMapVisitor(Device device, MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, FloatMapVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dout = s.Out(output);
            TB? dx = s.InOrNone(x), dy = s.InOrNone(y), dz = s.InOrNone(z);
            return s.Run(() => s.D.MapFloat<TN>(op, dx, dy, dz, dout, n, a, b));
        }
    }

    public IBenchRun? Reduce(Device device, string op, Axis axis, string type, byte[] x, int rows, int cols, byte[] output) =>
        Registry.With(type, new ReduceVisitor(device, op, axis, x, rows, cols, output));

    private sealed class ReduceVisitor(Device device, string op, Axis axis, byte[] x, int rows, int cols, byte[] output) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, ReduceVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dx = s.In(x), dout = s.Out(output);
            return s.Run(() => s.D.Reduce<TN>(op, axis, dx, rows, cols, dout));
        }
    }

    public IBenchRun? MapRow(Device device, MapOp op, string type, byte[] x, byte[] v, byte[] output, int rows, int cols) =>
        Registry.With(type, new MapRowVisitor(device, op, x, v, output, rows, cols));

    private sealed class MapRowVisitor(Device device, MapOp op, byte[] x, byte[] v, byte[] output, int rows, int cols) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, MapRowVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dx = s.In(x), dv = s.In(v), dout = s.Out(output);
            return s.Run(() => s.D.MapRow<TN>(op, dx, dv, dout, rows, cols));
        }
    }

    public IBenchRun? Transpose(Device device, string type, byte[] x, byte[] output, int rows, int cols) =>
        Registry.With(type, new TransposeVisitor(device, x, output, rows, cols));

    private sealed class TransposeVisitor(Device device, byte[] x, byte[] output, int rows, int cols) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, TransposeVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dx = s.In(x), dout = s.Out(output);
            return s.Run(() => s.D.Transpose<TN>(dx, dout, rows, cols));
        }
    }

    public IBenchRun? Gemv(Device device, string type, byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) =>
        Registry.With(type, new GemvVisitor(device, a, x, y, rows, cols, transposed));

    private sealed class GemvVisitor(Device device, byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, GemvVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB da = s.In(a), dx = s.In(x), dy = s.Out(y);
            return s.Run(() => s.D.Gemv<TN>(da, dx, dy, rows, cols, transposed));
        }
    }

    public IBenchRun? RowOp(Device device, RowOp op, string type, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RowOpVisitor(device, op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b)) : null;

    private sealed class RowOpVisitor(Device device, RowOp op, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b)
        : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, RowOpVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> st) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dx = st.In(x), dy = st.Out(y);
            TB? dx2 = st.InOrNone(x2), dg = st.InOrNone(g), dbt = st.InOrNone(bt), ds = st.InOrNone(s), dy2 = st.OutOrNone(y2);
            TI? dpos = st.IndexOrNone(pos);
            return st.Run(() => st.D.RowOp<TN>(op, dx, dx2, dg, dbt, ds, dpos, dy, dy2, rows, cols, a, b));
        }
    }

    public IBenchRun? NormParamsBwd(Device device, string type, byte[] x, byte[] dy, byte[] output, int rows, int cols, double eps, bool centered) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new NormParamsVisitor(device, x, dy, output, rows, cols, eps, centered)) : null;

    // dgamma and dbeta (cols each, in the float lane's storage) one after the other in output.
    private sealed class NormParamsVisitor(Device device, byte[] x, byte[] dy, byte[] output, int rows, int cols, double eps, bool centered) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, NormParamsVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            int size = TN.Bits == 64 ? 8 : 4;
            var dgamma = new byte[cols * size];
            var dbeta = new byte[cols * size];
            TB dx = s.In(x), ddy = s.In(dy), dg = s.Out(dgamma), db = s.Out(dbeta);
            s.Then(() => { dgamma.CopyTo(output, 0); dbeta.CopyTo(output, cols * size); });
            return s.Run(() => s.D.NormParamsBwd<TN>(dx, ddy, dg, db, rows, cols, eps, centered));
        }
    }

    public IBenchRun? Layout(Device device, string type, LayoutArgs k) =>
        LayoutArgs.Applies(k.Name, Registry.IsFloat(type)) ? Registry.With(type, new LayoutVisitor(device, k)) : null;

    private sealed class LayoutVisitor(Device device, LayoutArgs k) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, LayoutVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dx = s.In(k.X), dout = s.Out(k.Output);
            TB? dx2 = s.InOrNone(k.X2);
            TI? didx = s.IndexOrNone(k.Index);
            return s.Run(() => s.D.Layout<TN>(k, dx, didx, dout, dx2));
        }
    }

    public IBenchRun? MatMulView(Device device, string type, ProductView v, byte[] a, byte[] b, byte[] c) =>
        Registry.With(type, new ProductVisitor(device, v, a, b, c, null, null));

    public IBenchRun? MatMulBackward(Device device, string type, byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) =>
        Registry.With(type, new ProductVisitor(device, new ProductView(m, n, k, false, false), a, b, dc, da, db));

    /// <summary>A product on a view (da null) or the backward pair (da, db given: a is A, b is B, c is dC).</summary>
    private sealed class ProductVisitor(Device device, ProductView v, byte[] a, byte[] b, byte[] c, byte[]? da, byte[]? db) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, ProductVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dA = s.In(a), dB = s.In(b);
            if (da is not null)
            {
                TB dC = s.In(c), dDa = s.Out(da), dDb = s.Out(db!);
                return s.Run(() => s.D.MatMulBackward<TN>(dA, dB, dC, dDa, dDb, v.M, v.N, v.K));
            }

            TB dOut = s.Out(c);
            return s.Run(() => s.D.MatMulView<TN>(v, dA, dB, dOut));
        }
    }

    public IBenchRun? MatMulBlocks(Device device, string type, int m, int n, int k, byte[] a, byte[] b, byte[] c) =>
        type is "q8_0" or "q4_0" && k % 32 == 0 && (device == Device.Cpu ? CpuDevice.Device.BlockProductAvailable : Devices.Of(device).BlockProductAvailable)
            ? Registry.With(type, new MatMulBlocksVisitor(device, new ProductView(m, n, k, false, true), a, b, c)) : null;

    private sealed class MatMulBlocksVisitor(Device device, ProductView v, byte[] a, byte[] b, byte[] c) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, MatMulBlocksVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dA = s.In(a), dB = s.In(b), dC = s.Out(c);
            return s.Run(() => s.D.MatMulViewBlocks<TN>(v, dA, dB, dC));
        }
    }

    public IBenchRun? Attention(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, bool decode) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new AttentionVisitor(device, a, q, k, v, o, decode, null, null, null)) : null;

    public IBenchRun? AttentionFused(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) =>
        Registry.IsFloat(type) && a.D <= GpuOps.FusedHeadMax ? Registry.WithFloat(type, new AttentionVisitor(device, a, q, k, v, o, false, null, null, null, fused: true)) : null;

    public IBenchRun? AttentionBackward(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new AttentionVisitor(device, a, q, k, v, dO, false, dq, dk, dv)) : null;

    private sealed class AttentionVisitor(Device device, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, bool decode, byte[]? dq, byte[]? dk, byte[]? dv, bool fused = false)
        : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, AttentionVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dQ = s.In(q), dK = s.In(k), dV = s.In(v);
            if (dq is not null)
            {
                TB dO = s.In(o), gq = s.Out(dq), gk = s.Out(dk!), gv = s.Out(dv!);
                return s.Run(() => s.D.AttentionBackward<TN>(a, dQ, dK, dV, dO, gq, gk, gv));
            }

            TB dOut = s.Out(o);
            return s.Run(() => decode ? s.D.AttentionDecode<TN>(a, dQ, dK, dV, dOut) : fused ? s.D.AttentionRows<TN>(a, dQ, dK, dV, dOut) : s.D.Attention<TN>(a, dQ, dK, dV, dOut));
        }
    }

    public IBenchRun? Optimizer(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new OptimizerVisitor(device, a, p, g, m, v, n)) : null;

    // The step changes p, m, v: each run starts from the same state, copied on the device from the state staged once (the
    // restore is timed, as a caller's would not be: it is named).
    private sealed class OptimizerVisitor(Device device, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, OptimizerVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB sp = s.In(p), sm = s.In(m), dg = s.In(g), dp = s.Out(p), dm = s.Out(m);
            TB? sv = s.InOrNone(v), dv = s.OutOrNone(v);
            return s.Run(() =>
            {
                s.D.Copy(sp, dp, p.Length);
                s.D.Copy(sm, dm, m.Length);
                if (v is not null) s.D.Copy(sv!, dv!, v.Length);
                return s.D.Optimizer<TN>(a, dp, dg, dm, dv, n);
            }, "; + restore of p, m, v");
        }
    }

    public IBenchRun? Conv(Device device, string type, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) =>
        dx is null ? Registry.With(type, new ConvVisitor(device, a, x, w, y)) : Registry.IsFloat(type) ? Registry.WithFloat(type, new ConvBackwardVisitor(device, a, x, w, y, dx, dw!)) : null;

    private sealed class ConvVisitor(Device device, ConvArgs a, byte[] x, byte[] w, byte[] y) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, ConvVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dX = s.In(x), dW = s.In(w), dY = s.Out(y);
            return s.Run(() => s.D.Conv<TN>(a, dX, dW, default, dY));
        }
    }

    private sealed class ConvBackwardVisitor(Device device, ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, ConvBackwardVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dX = s.In(x), dW = s.In(w), dY = s.In(dy), gx = s.Out(dx), gw = s.Out(dw);
            return s.Run(() => s.D.ConvBackward<TN>(a, dX, dW, dY, gx, gw, default));
        }
    }

    public IBenchRun? TopK(Device device, string type, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) =>
        Registry.With(type, new TopKVisitor(device, x, rows, cols, k, values, indices));

    private sealed class TopKVisitor(Device device, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, TopKVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dx = s.In(x), dv = s.Out(values);
            TI di = s.IndexOut(indices);
            return s.Run(() => s.D.TopK<TN>(dx, rows, cols, k, dv, di));
        }
    }

    public IBenchRun? Pick(Device device, string type, byte[] x, int rows, int cols, int[] index, byte[] output) =>
        Registry.With(type, new PickVisitor(device, x, rows, cols, index, output));

    private sealed class PickVisitor(Device device, byte[] x, int rows, int cols, int[] index, byte[] output) : ITypeVisitor<IBenchRun?>, INumberRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IAdad<TN, float> => On<TN, PickVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IAdad<TN, float>
        {
            TB dx = s.In(x), dout = s.Out(output);
            TI di = s.Index(index);
            return s.Run(() => s.D.Pick<TN>(dx, rows, cols, di, dout));
        }
    }

    public IBenchRun? CrossEntropy(Device device, string type, byte[] x, int rows, int cols, int[] targets, byte[] loss) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new CrossEntropyVisitor(device, x, rows, cols, targets, loss)) : null;

    private sealed class CrossEntropyVisitor(Device device, byte[] x, int rows, int cols, int[] targets, byte[] loss) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, CrossEntropyVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dx = s.In(x), dl = s.Out(loss);
            TI dt = s.Index(targets);
            return s.Run(() => s.D.CrossEntropy<TN>(dx, rows, cols, dt, dl));
        }
    }

    public IBenchRun? Ctc(Device device, string type, CtcArgs a, byte[] logits, byte[] loss, byte[]? grad) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new CtcVisitor(device, a, logits, loss, grad)) : null;

    private sealed class CtcVisitor(Device device, CtcArgs a, byte[] x, byte[] loss, byte[]? grad) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, CtcVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dx = s.In(x), dl = s.Out(loss);
            TB? dg = s.OutOrNone(grad);
            return s.Run(() => s.D.Ctc<TN>(a, dx, dl, dg));
        }
    }

    public IBenchRun? Sample(Device device, string type, byte[] x, int rows, int cols, byte[] u, int[] picks) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new SampleVisitor(device, x, rows, cols, u, picks)) : null;

    private sealed class SampleVisitor(Device device, byte[] x, int rows, int cols, byte[] u, int[] picks) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, SampleVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            TB dx = s.In(x), du = s.In(u);
            TI dp = s.IndexOut(picks);
            return s.Run(() => s.D.Sample<TN>(dx, rows, cols, du, dp));
        }
    }

    public IBenchRun? Route(Device device, int[] index, int buckets, int[] counts, int[] offsets, int[] order)
    {
        IBenchRun Make<TB, TI>(Staged<TB, TI> s)
        {
            TI di = s.Index(index), dc = s.IndexOut(counts), dof = s.IndexOut(offsets), dor = s.IndexOut(order);
            return s.Run(() => { s.D.Route(di, index.Length, buckets, dc, dof, dor); return "s32"; });
        }

        return device == Device.Cpu ? Make<byte[], int[]>(new(CpuDevice.Device, 0)) : Make<ulong, ulong>(new(Devices.Of(device), 4));
    }

    public IBenchRun? Recurrent(Device device, string type, CellArgs a, byte[] x, byte[] h, byte[]? c, byte[] w, byte[] u, byte[] b, byte[]? bh, byte[] hOut, byte[]? cOut,
        byte[]? dx, byte[]? dh, byte[]? dc, byte[]? dw, byte[]? du, byte[]? db, byte[]? dbh) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new RecurrentVisitor(device, a, x, h, c, w, u, b, bh, hOut, cOut, dx, dh, dc, dw, du, db, dbh)) : null;

    // A step (LSTM: c and cOut given; GRU: bh given), or with dx given its backward (hOut, cOut are then dh', dc').
    private sealed class RecurrentVisitor(Device device, CellArgs a, byte[] x, byte[] h, byte[]? c, byte[] w, byte[] u, byte[] b, byte[]? bh, byte[] hOut, byte[]? cOut,
        byte[]? dx, byte[]? dh, byte[]? dc, byte[]? dw, byte[]? du, byte[]? db, byte[]? dbh) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, RecurrentVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            bool lstm = a.Kind == "lstm";
            TB dX = s.In(x), dH = s.In(h), dW = s.In(w), dU = s.In(u), dB = s.In(b);
            if (dx is not null)
            {
                TB dHo = s.In(hOut), gx = s.Out(dx), gh = s.Out(dh!), gw = s.Out(dw!), gu = s.Out(du!), gb = s.Out(db!);
                if (lstm)
                {
                    TB dC = s.In(c!), dCo = s.In(cOut!), gc = s.Out(dc!);
                    return s.Run(() => s.D.LstmStepBackward<TN>(a, dX, dH, dC, dW, dU, dB, dHo, dCo, gx, gh, gc, gw, gu, gb));
                }

                TB dBh = s.In(bh!), gbh = s.Out(dbh!);
                return s.Run(() => s.D.GruStepBackward<TN>(a, dX, dH, dW, dU, dB, dBh, dHo, gx, gh, gw, gu, gb, gbh));
            }

            TB dOut = s.Out(hOut);
            if (lstm)
            {
                TB dC = s.In(c!), dCo = s.Out(cOut!);
                return s.Run(() => s.D.LstmStep<TN>(a, dX, dH, dC, dW, dU, dB, dOut, dCo));
            }

            TB dBias = s.In(bh!);
            return s.Run(() => s.D.GruStep<TN>(a, dX, dH, dW, dU, dB, dBias, dOut));
        }
    }

    public IBenchRun? LstmSequence(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] b, byte[] hOut, byte[] cOut,
        byte[]? dx, byte[]? dh0, byte[]? dc0, byte[]? dw, byte[]? du, byte[]? db) =>
        Registry.IsFloat(type) && !Registry.IsBlock(type) ? Registry.WithFloat(type, new SequenceVisitor(device, a, T, x, h0, c0, w, u, b, hOut, cOut, dx, dh0, dc0, dw, du, db)) : null;

    // An LSTM sequence, or with dx given its backward (hOut is then dh per step, cOut the last step's dc).
    private sealed class SequenceVisitor(Device device, CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] b, byte[] hOut, byte[] cOut,
        byte[]? dx, byte[]? dh0, byte[]? dc0, byte[]? dw, byte[]? du, byte[]? db) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, SequenceVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            int pad = s.Slack;
            TB dX = s.In(x, pad), dH0 = s.In(h0, pad), dC0 = s.In(c0, pad), dW = s.In(w, pad), dU = s.In(u, pad), dB = s.In(b, pad);
            if (dx is not null)
            {
                TB dHo = s.In(hOut, pad), dCo = s.In(cOut, pad);
                TB gx = s.Out(dx, pad), gh0 = s.Out(dh0!, pad), gc0 = s.Out(dc0!, pad), gw = s.Out(dw!, pad), gu = s.Out(du!, pad), gb = s.Out(db!, pad);
                return s.Run(() => s.D.LstmSequenceBackward<TN>(a, T, dX, dH0, dC0, dW, dU, dB, dHo, dCo, gx, gh0, gc0, gw, gu, gb));
            }

            TB oH = s.Out(hOut, pad), oC = s.Out(cOut, pad);
            return s.Run(() => s.D.LstmSequence<TN>(a, T, dX, dH0, dC0, dW, dU, dB, oH, oC));
        }
    }

    public IBenchRun? GruSequence(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] hOut) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new GruSequenceVisitor(device, a, T, x, h0, w, u, b, bh, hOut)) : null;

    private sealed class GruSequenceVisitor(Device device, CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] hOut) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, GruSequenceVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            int pad = s.Slack;
            TB dX = s.In(x, pad), dH0 = s.In(h0, pad), dW = s.In(w, pad), dU = s.In(u, pad), dB = s.In(b, pad), dBh = s.In(bh, pad), dHo = s.Out(hOut, pad);
            return s.Run(() => s.D.GruSequence<TN>(a, T, dX, dH0, dW, dU, dB, dBh, dHo));
        }
    }

    public IBenchRun? GruSequenceBackward(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] dhOut,
        byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh) =>
        Registry.IsFloat(type) ? Registry.WithFloat(type, new GruSequenceBackwardVisitor(device, a, T, x, h0, w, u, b, bh, dhOut, dx, dh0, dw, du, db, dbh)) : null;

    private sealed class GruSequenceBackwardVisitor(Device device, CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] dhOut,
        byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh) : IFloatTypeVisitor<IBenchRun?>, IFloatRun
    {
        public IBenchRun? Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => OnFloat<TN, GruSequenceBackwardVisitor>(device, this);

        public IBenchRun Make<TN, TB, TI>(Staged<TB, TI> s) where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            int pad = s.Slack;
            TB dX = s.In(x, pad), dH0 = s.In(h0, pad), dW = s.In(w, pad), dU = s.In(u, pad), dB = s.In(b, pad), dBh = s.In(bh, pad), dHo = s.In(dhOut, pad);
            TB gx = s.Out(dx, pad), gh0 = s.Out(dh0, pad), gw = s.Out(dw, pad), gu = s.Out(du, pad), gb = s.Out(db, pad), gbh = s.Out(dbh, pad);
            return s.Run(() => s.D.GruSequenceBackward<TN>(a, T, dX, dH0, dW, dU, dB, dBh, dHo, gx, gh0, gw, gu, gb, gbh));
        }
    }

    // ---------------------------------------------------------------- the element-wise lane (add's report)

    private struct LaneName : ILaneVisitor
    {
        public string Name;

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> => Name = TL.Name;
    }

    private static string ElementwiseLane<TN>() where TN : struct, IAdad<TN>
    {
        var probe = new LaneName();
        TN.Elementwise(ref probe);
        return probe.Name;
    }
}
