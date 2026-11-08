// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on Vulkan (Abstraction/Boxes.cs; the kernels Kernels.Boxes.cs): one launch, each output word
// computed where it is stored (a block format's by its encoder) in the float lane; the corners read where they are stored.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>loss[i] of count pairs of corner boxes (count × 4 as the type each) as the type; returns the lane's name.</summary>
    public string BoxLoss<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong loss, int count, double eps) where TN : struct, IFloatAdad<TN> =>
        BoxRun<TN>(overlap, predicted, target, 0, loss, count, eps);

    /// <summary>dpredicted = dloss[i] · ∂loss[i] / ∂predicted[i] (count × 4 as the type); returns the lane's name.</summary>
    public string BoxLossBackward<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong dpredicted, int count, double eps) where TN : struct, IFloatAdad<TN> =>
        BoxRun<TN>(overlap, predicted, target, dloss, dpredicted, count, eps);

    private string BoxRun<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong output, int count, double eps) where TN : struct, IFloatAdad<TN>
    {
        _ = Boxes.Name(overlap);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        bool backward = dloss != 0;
        long n = (long)count * (backward ? 4 : 1);
        if (n == 0) return lane.Name;
        if (n > uint.MaxValue) throw new NotSupportedException($"box losses: {n} values past 2^32");
        if (f.IsBlock)
            Encode(f, n, vk.Cached($"block_box:{overlap}:{backward}:{f.Name}", () => VkKernels.BlockBoxLoss(vk, f, overlap, backward)), output,
                predicted, target, dloss, Vulkan.F((float)eps));
        else
        {
            var k = vk.Cached($"box:{overlap}:{backward}:{f.Name}:{lane.Name}", () => VkKernels.BoxLoss(vk, f, lane, overlap, backward));
            vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, predicted, target, dloss, output, Vulkan.U((int)n), Slot(lane, eps));
        }

        return lane.Name;
    }
}
