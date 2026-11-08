// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The general convolution's store on Vulkan (VulkanOps.Conv.cs), CUDA's (Cuda/Ptx.ConvStore.cs): y = act(sums + bias[f]) an
// invocation a stored word (a block format's by its encoder), the activation the element-wise formula's (Kernels.Map.cs).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Slots: sums (lane storage), bias (f), y, n, cols (an output channel's positions), oc, the activation's parameter (lane).</summary>
    public static SpvKernel ConvStore(Vulkan vk, Fmt f, Lane lane, string? act, bool bias)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"conv_store_{f.Name}_{lane.Name}_{act ?? "none"}{(bias ? "_b" : "")}", Threads, vk.Controls);
        V ps = b.Arg(0), pb = b.Arg(1), py = b.Arg(2);
        V n = b.Arg(3, K.U32), cols = b.Arg(4, K.U32), oc = b.Arg(5, K.U32), pa = b.Arg(6, k);
        var gid = b.GlobalX;
        var natural = f.Natural;
        var nk = Kinds.Of(natural);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var word = b.CBits(Codec.WordKind(f), 0);
            var last = n - 1;
            for (int i = 0; i < f.Per; i++)
            {
                // A word's values past the count take the last value's addresses (dropped).
                var at = gid * (double)f.Per + i;
                var value = ConvStoreValue(b, f, lane, k, act, bias, ps, pb, cols, oc, pa, b.Min(at, last));
                var stored = lane.Name == natural.Name ? value : Codec.Convert(b, lane, value, natural);
                word = Codec.Pack(b, f, b.Select(at < n, stored, b.C(nk, 0)), natural, i, word);
            }

            b.Store(b.At(py, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>The store into blocks (f32). Slots: sums, bias, cols, oc, the activation's parameter (f32).</summary>
    public static SpvKernel BlockConvStore(Vulkan vk, Fmt f, string? act, bool bias) =>
        BlockEncoder(vk, f, $"conv_store_{act ?? "none"}{(bias ? "_b" : "")}", b =>
        {
            V ps = b.Arg(0), pb = b.Arg(1);
            V cols = b.Arg(2, K.U32), oc = b.Arg(3, K.U32), pa = b.Arg(4, K.F32);
            return (at, _) => ConvStoreValue(b, f, Lane.F32, K.F32, act, bias, ps, pb, cols, oc, pa, at);
        });

    private static V ConvStoreValue(Spv b, Fmt f, Lane lane, K k, string? act, bool bias, V ps, V pb, V cols, V oc, V pa, V at)
    {
        var v = b.Load(k, b.At(ps, at, lane.Size));
        if (bias) v = v + Codec.UnpackAt(b, f, pb, at / cols % oc, lane);
        return act is null ? v : Formula(b, k, act, v, v, v, pa, pa);
    }
}
