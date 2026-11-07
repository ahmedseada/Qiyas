// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's optimizer steps (Optimizers.cs).
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (v = 0 for one moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (f.IsBlock)
        {
            // A block format's parameters: the step in float32 storage, the parameters encoded back by blocks.
            var r = new List<ulong>();
            try
            {
                ulong lp = Lanes(f, Lane.F32, p, n, r), lg = Lanes(f, Lane.F32, g, n, r);
                Optimizer<Float32>(a, lp, lg, m, v, n);
                Back(f, Lane.F32, lp, p, n);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var q in r) gpu.Return(q); }
        }

        var fn = gpu.Cached($"optimizer:{a.Name}:{f.Name}", (out string e) => Ptx.Optimizer(gpu, f, lane, a.Name, out e));
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => lane.Wide ? Gpu.D(x) : Gpu.F((float)x);
        Gpu.Launch(fn, Grid(f.Words(n), 256), 1, 256, 1, 0, p, g, m, v, Gpu.U(n),
            S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd));
        return lane.Name;
    }
}
