// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on Vulkan (Sampling.cs; the kernels in Kernels.Sampling.cs): the weights made where each pass needs
// them in the f32 lane, an intermediate in lane storage in the f64 lane; the scores and the step read where the device
// holds them.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered): the weights into lane storage
    /// (VkKernels.SampleWeights), then each rule's prefix of the row's order and the draw (VkKernels.SampleDraw); the
    /// probabilities in the float lane's storage. Every type but bool. Returns the lane's name.</summary>
    public string SampleFiltered<TN>(SamplingArgs a, ulong x, ulong step, ulong picks, ulong probability) where TN : struct, IAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        if (f.Bits == 1) throw new NotSupportedException($"sampling: {f.Name} holds truth values, not scores");
        var lane = f.FloatLane;
        if (a.Rows == 0) return lane.Name;
        // The f32 lane's weights made where each pass needs them (VkKernels.SampleWeighsInPlace); the f64 lane's into lane
        // storage first.
        bool inPlace = VkKernels.SampleWeighsInPlace(lane);
        ulong w = inPlace ? 0 : vk.Rent(Math.Max(16, (long)a.Rows * a.Cols * lane.Size));
        try
        {
            if (!inPlace)
                vk.Launch(vk.Cached($"sample_weights:{f.Name}:{lane.Name}", () => VkKernels.SampleWeights(vk, f, lane)), (uint)a.Rows, 1,
                    x, w, Vulkan.U(a.Cols), Vulkan.U(a.Stride), Vulkan.U(a.RowOffset), Slot(lane, a.Temperature));
            var key = Draws.Key(a.Seed);
            // The kernel of the call's rules only (VkKernels.SampleRules): a rule's code costs its registers and workgroup
            // memory in every call.
            var rules = VkKernels.SampleRules.Of(a, lane);
            bool gathered = rules.Gathered(a.TopK);
            vk.Launch(vk.Cached($"sample_draw:{f.Name}:{lane.Name}:{rules.TopK}:{rules.Nucleus}:{gathered}", () => VkKernels.SampleDraw(vk, f, lane, rules, gathered)), (uint)a.Rows, 1,
                x, inPlace ? Slot(lane, a.Temperature) : w, picks, probability, step, Vulkan.U(a.Cols), Vulkan.U(a.Stride), Vulkan.U(a.RowOffset),
                Vulkan.U(a.TopK), key.A, Slot(lane, a.TopP), Slot(lane, a.MinP), key.B);
        }
        finally
        {
            if (w != 0) vk.Return(w);
        }

        return lane.Name;
    }

    /// <summary>keys[i] (u64) = the order key (VkKernels.OrderKey) of x's value i of TN in its natural lane: the key proven on
    /// the device against Routing.Order.</summary>
    internal void OrderKeys<TN>(ulong x, ulong keys, int n) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (n == 0) return;
        vk.Launch(vk.Cached($"order_keys:{f.Name}", () => VkKernels.OrderKeys(vk, f)), Groups(n, VkKernels.Threads), 1, x, keys, Vulkan.U(n));
    }
}
