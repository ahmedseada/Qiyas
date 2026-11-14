// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on Vulkan (Abstraction/Boxes.cs; the kernels Kernels.Boxes.cs): one launch, each output word
// computed where it is stored (a block format's by its encoder) in the float lane; the corners read where they are stored.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>loss[i] of count pairs of corner boxes (count × 4 as the type each) as the type, eps the type's; returns the
    /// lane's name.</summary>
    public string BoxLoss<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong loss, int count) where TN : struct, IFloatAdad<TN> =>
        BoxRun<TN>(overlap, predicted, target, 0, loss, count);

    /// <summary>dpredicted = dloss[i] · ∂loss[i] / ∂predicted[i] (count × 4 as the type); returns the lane's name.</summary>
    public string BoxLossBackward<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong dpredicted, int count) where TN : struct, IFloatAdad<TN> =>
        BoxRun<TN>(overlap, predicted, target, dloss, dpredicted, count);

    private string BoxRun<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong output, int count) where TN : struct, IFloatAdad<TN>
    {
        double eps = TN.Epsilon;
        _ = Boxes.Name(overlap);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        bool backward = dloss != 0;
        long n = (long)count * (backward ? 4 : 1);
        if (n == 0) return lane.Name;
        if (n > uint.MaxValue) throw new NotSupportedException($"box losses: {n} values past 2^32");
        if (f.IsBlock)
        {
            // A block format: its encoder computing each block's values (a value at a time), or the values into f32 lanes in
            // parallel (the backward a pair an invocation) and the encoder over them: the same values, so the same blocks;
            // whichever this device measures faster at the format, overlap and count (once).
            void Fused() => Encode(f, n, vk.Cached($"block_box:{overlap}:{backward}:{f.Name}", () => VkKernels.BlockBoxLoss(vk, f, overlap, backward)), output,
                predicted, target, dloss, Vulkan.F((float)eps));
            void Lanes()
            {
                var into = VkKernels.LaneFormat(Lane.F32);
                var k = vk.Cached($"box:{overlap}:{backward}:{f.Name}:f32_lanes", () => VkKernels.BoxLoss(vk, f, Lane.F32, overlap, backward, into));
                ulong lanes = vk.Rent(n * 4);
                try
                {
                    vk.Launch(k, Groups(backward ? count : n, VkKernels.Threads), 1, predicted, target, dloss, lanes, Vulkan.U((int)n), Vulkan.F((float)eps));
                    Back(f, Lane.F32, lanes, output, n);
                }
                finally { vk.Return(lanes); }
            }

            if (BoxLanes ?? BoxLanesFaster($"{f.Name}:{overlap}:{backward}:{count}", Fused, Lanes)) Lanes(); else Fused();
        }
        else
        {
            var k = vk.Cached($"box:{overlap}:{backward}:{f.Name}:{lane.Name}", () => VkKernels.BoxLoss(vk, f, lane, overlap, backward));
            long items = backward && VkKernels.BoxPairWords(f) > 0 ? count : f.Words(n);
            vk.Launch(k, Groups(items, VkKernels.Threads), 1, predicted, target, dloss, output, Vulkan.U((int)n), Slot(lane, eps));
        }

        return lane.Name;
    }

    /// <summary>The gate's override of a block format's box route: true the f32 lanes then the encoder, false the encoder
    /// computing its values, null as measured.</summary>
    internal bool? BoxLanes;

    private readonly Dictionary<string, bool> _boxLanes = [];

    private bool BoxLanesFaster(string key, Action fused, Action lanes)
    {
        if (_boxLanes.TryGetValue(key, out bool known)) return known;
        var (tf, tl) = Measure.Faster(fused, lanes, vk.Synchronize);
        _unitsTimes.Add($"box {key}: encoder {tf:0.000} ms, lanes {tl:0.000} ms");
        return _boxLanes[key] = tl < tf;
    }
}
