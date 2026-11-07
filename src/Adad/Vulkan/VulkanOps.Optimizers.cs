// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The optimizer steps on Vulkan (Kernels.Optimizers.cs): float types in the f32 lane (float64 in f64), as on CUDA; the
// block formats have no Vulkan encoder yet (refused).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (v = 0 for one moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"optimizer {a.Name} of {f.Name} (a block format)");
        var lane = RowLane(f);
        if (n == 0) return lane.Name;
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => Slot(lane, x);
        var fn = vk.Cached($"optimizer:{a.Name}:{f.Name}", () => VkKernels.Optimizer(vk, f, lane, a.Name));
        vk.Launch(fn, Groups(f.Words(n), VkKernels.Threads), 1, p, g, m, v, Vulkan.U(n), S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd));
        return lane.Name;
    }
}
