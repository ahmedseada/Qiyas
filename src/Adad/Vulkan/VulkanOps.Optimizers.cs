// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The optimizer steps on Vulkan (Kernels.Optimizers.cs): float types in the f32 lane (float64 in f64), as on CUDA; a
// block format's parameters by its encoder (each block from its values' steps, the moments updated as they are taken).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (v = 0 for one moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (n == 0) return lane.Name;
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => Slot(lane, x);
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_optimizer:{a.Name}:{f.Name}", () => VkKernels.BlockOptimizer(vk, f, lane, a.Name)), p,
                p, g, m, v, S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd));
            return lane.Name;
        }

        var fn = vk.Cached($"optimizer:{a.Name}:{f.Name}", () => VkKernels.Optimizer(vk, f, lane, a.Name));
        vk.Launch(fn, Groups(f.Words(n), VkKernels.Threads), 1, p, g, m, v, Vulkan.U(n), S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd));
        return lane.Name;
    }
}
