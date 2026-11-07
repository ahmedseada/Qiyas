// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's optimizer steps (Optimizers.cs).
namespace Adad;

public sealed partial class GpuOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (v = 0 for one moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => lane.Wide ? Gpu.D(x) : Gpu.F((float)x);
        ulong[] args = [p, g, m, v, Gpu.U(n), S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd)];
        if (f.IsBlock)
        {
            // A block format's parameters: a warp per block, the step in f32 registers, the block re-encoded in place.
            var fb = gpu.Cached($"optimizer_blocks:{a.Name}:{f.Name}", (out string e) => Ptx.OptimizerBlocks(gpu, f, a.Name, out e));
            Gpu.Launch(fb, BlockGrid(f, n), 1, 256, 1, 0, args);
            return "f32 (blocks in place)";
        }

        var fn = gpu.Cached($"optimizer:{a.Name}:{f.Name}", (out string e) => Ptx.Optimizer(gpu, f, lane, a.Name, out e));
        Gpu.Launch(fn, Grid(f.Words(n), 256), 1, 256, 1, 0, args);
        return lane.Name;
    }
}
