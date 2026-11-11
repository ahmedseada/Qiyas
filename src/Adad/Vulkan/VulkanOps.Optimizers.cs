// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The optimizer steps on Vulkan (Kernels.Optimizers.cs): float types in the f32 lane (float64 in f64), as on CUDA; a
// block format's parameters by its encoder (each block from its values' steps, the moments updated as they are taken).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (m not read by sgd, v = 0 for one
    /// moment). Returns the lane's name.</summary>
    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        return Step(Fmt.Of<TN>(), a, p, g, m, v, n, 0);
    }

    // The step; factor: where the gradients' scale is (a lane value), 0 for none.
    private string Step(Fmt f, OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n, ulong factor)
    {
        var lane = RowLane(f);
        if (n == 0) return lane.Name;
        double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
        ulong S(double x) => Slot(lane, x);
        bool scaled = factor != 0;
        if (f.IsBlock)
        {
            ulong[] slots = [p, g, m, v, S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd)];
            if (scaled) slots[VkKernels.BlockOptimizerScaleSlot(a.Name)] = factor;      // the slot of a scalar this step does not read
            Encode(f, n, vk.Cached($"block_optimizer:{a.Name}:{f.Name}:{scaled}", () => VkKernels.BlockOptimizer(vk, f, lane, a.Name, scaled)), p, slots);
            return lane.Name;
        }

        var fn = vk.Cached($"optimizer:{a.Name}:{f.Name}:{scaled}", () => VkKernels.Optimizer(vk, f, lane, a.Name, scaled));
        vk.Launch(fn, Groups(f.Words(n), VkKernels.Threads), 1, p, g, m, v, Vulkan.U(n), S(a.Lr), S(a.B1), S(a.B2), S(1 - a.B1), S(1 - a.B2), S(1 / c1), S(1 / c2), S(a.Eps), S(a.Wd), S(a.Lr * a.Wd), factor);
        return lane.Name;
    }

    /// <summary>A step across several tensors under one clipping (IOptimizers.OptimizerMany; CUDA's): every tensor's
    /// whole-sum pass of its gradients' squares into one buffer of float64 partials, the partials of all the tensors joined
    /// once in float64 and rounded once to the lane — one sum across the tensors — the factor from it by the map's
    /// clip_factor in the lane, then each tensor's step reading the factor where it is. Returns the lane's name.</summary>
    public string OptimizerMany<TN>(OptimizerArgs a, ReadOnlySpan<OptimizerTensor<ulong>> tensors, ulong sumSquares) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong d = vk.Rent(Math.Max(16, bytes)); rented.Add(d); return d; }
        try
        {
            ulong factor = 0;
            if (a.MaxNorm > 0 || sumSquares != 0)
            {
                long blocks = 0;
                foreach (var t in tensors) if (t.N > 0) blocks += WholeBlocks(t.N);
                ulong sum = sumSquares != 0 ? sumSquares : Rent(lane.Size);
                if (blocks == 0) vk.Upload<byte>(new byte[lane.Size], sum);
                else
                {
                    var lf = VkKernels.LaneFormat(Lane.F64);
                    var pass = lane.Wide
                        ? vk.Cached($"reduce_rows:sum:True:{f.Name}:f64", () => VkKernels.ReduceRows(vk, f, Lane.F64, "sum", true))
                        : vk.Cached($"reduce_rows:sum:True:{f.Name}:f32:kahan", () => VkKernels.ReduceRows(vk, f, Lane.F32, "sum", true, compensated: true));
                    var join = vk.Cached($"reduce_rows:sum:False:{lf.Name}:f64", () => VkKernels.ReduceRows(vk, lf, Lane.F64, "sum", false));
                    ulong partials = Rent(blocks * 8), wide = lane.Wide ? sum : Rent(8);
                    long at = 0;
                    foreach (var t in tensors)
                    {
                        if (t.N == 0) continue;
                        int tb = WholeBlocks(t.N);
                        vk.Launch(pass, (uint)tb, 1, t.G, partials + (ulong)(at * 8), Vulkan.U(t.N), Vulkan.U(WholeChunk(t.N)));
                        at += tb;
                    }

                    vk.Launch(join, 1, 1, partials, wide, Vulkan.U((int)blocks), Vulkan.U(0));
                    if (!lane.Wide) Back(VkKernels.LaneFormat(Lane.F32), Lane.F64, wide, sum, 1);
                }

                if (a.MaxNorm > 0)
                {
                    factor = Rent(lane.Size);
                    var clip = Ops.Find("clip_factor");
                    if (lane.Wide) Map<Float64>(clip, sum, 0, 0, factor, 1, a.MaxNorm, 0); else Map<Float32>(clip, sum, 0, 0, factor, 1, a.MaxNorm, 0);
                }
            }

            foreach (var t in tensors) Step(f, a, t.P, t.G, t.M, t.V, t.N, factor);
        }
        finally { foreach (var d in rented) vk.Return(d); }

        return lane.Name;
    }
}
