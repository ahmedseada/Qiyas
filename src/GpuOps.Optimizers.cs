// The device's optimizer steps (Optimizers.cs).
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (v = 0 for one moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var fn = gpu.Cached($"optimizer:{a.Name}:{f.Name}", (out string e) => Ptx.Optimizer(gpu, f, lane, a.Name, out e));
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => lane.Wide ? Gpu.D(x) : Gpu.F((float)x);
        Gpu.Launch(fn, Grid(f.Words(n), 256), 1, 256, 1, 0, p, g, m, v, Gpu.U(n),
            S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd));
        return lane.Name;
    }
}
