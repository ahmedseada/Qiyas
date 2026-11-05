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
