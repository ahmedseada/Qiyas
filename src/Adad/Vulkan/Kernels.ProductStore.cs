// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A product's store on Vulkan (VulkanOps.Products.cs, and the general convolution's, VulkanOps.Conv.cs), CUDA's
// (Cuda/Ptx.ProductStore.cs): y = the store terms' value of the lane sums, an invocation a stored word (a block format's by
// its encoder, y's values before read by the encoder and written by the compaction after), the formula the element-wise
// one's (Kernels.Map.cs), every term in the lane and each value rounded to the type once.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Slots: sums (lane storage), the low-rank term's sums (lane storage), bias (the channel (at / cols) mod oc),
    /// residual, the activation's inputs (aux), y, n, cols, oc, the activation's parameter and β (lane values).</summary>
    public static SpvKernel ProductStore(Vulkan vk, Fmt f, Lane lane, StoreTerms t)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"product_store_{f.Name}_{lane.Name}_{t.Key}", Threads, vk.Controls);
        V ps = b.Arg(0), pl = b.Arg(1), pb = b.Arg(2), pr = b.Arg(3), px = b.Arg(4), py = b.Arg(5);
        V n = b.Arg(6, K.U32), cols = b.Arg(7, K.U32), oc = b.Arg(8, K.U32), pa = b.Arg(9, k), beta = b.Arg(10, k);
        StoreEach(b, f, lane, py, n, at => StoreValue(b, f, lane, k, t, new StorePointers(ps, pl, pb, pr, px, py, cols, oc, pa, beta), at));
        return b.Build();
    }

    /// <summary>The n values of f at py from value(at) (a lane value; at below n), an invocation a stored word, each rounded to
    /// the type once (a block format's go through BlockEncoder instead).</summary>
    internal static void StoreEach(Spv b, Fmt f, Lane lane, V py, V n, Func<V, V> value)
    {
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
                var v = value(b.Min(at, last));
                var stored = lane.Name == natural.Name ? v : Codec.Convert(b, lane, v, natural);
                word = Codec.Pack(b, f, b.Select(at < n, stored, b.C(nk, 0)), natural, i, word);
            }

            b.Store(b.At(py, gid, f.WordBits / 8), word);
        });
    }

    /// <summary>The store into blocks (f32). Slots: sums, the low-rank term's sums, bias, residual, aux, y (its values
    /// before), cols, oc, the activation's parameter and β (f32).</summary>
    public static SpvKernel BlockProductStore(Vulkan vk, Fmt f, StoreTerms t) =>
        BlockEncoder(vk, f, $"product_store_{t.Key}", b =>
        {
            V ps = b.Arg(0), pl = b.Arg(1), pb = b.Arg(2), pr = b.Arg(3), px = b.Arg(4), py = b.Arg(5);
            V cols = b.Arg(6, K.U32), oc = b.Arg(7, K.U32), pa = b.Arg(8, K.F32), beta = b.Arg(9, K.F32);
            return (at, _) => StoreValue(b, f, Lane.F32, K.F32, t, new StorePointers(ps, pl, pb, pr, px, py, cols, oc, pa, beta), at);
        });

    private readonly record struct StorePointers(V Sums, V Low, V Bias, V Residual, V Aux, V Y, V Cols, V OC, V A, V Beta);

    private static V StoreValue(Spv b, Fmt f, Lane lane, K k, StoreTerms t, StorePointers p, V at)
    {
        var v = b.Load(k, b.At(p.Sums, at, lane.Size));
        if (t.Low) v = v + b.Load(k, b.At(p.Low, at, lane.Size));
        if (t.Bias) v = v + Codec.UnpackAt(b, f, p.Bias, at / p.Cols % p.OC, lane);
        if (t.Beta) v = b.Fma(p.Beta, Codec.UnpackAt(b, f, p.Y, at, lane), v);
        if (t.Grad is not null) v = Formula(b, k, t.Grad, Codec.UnpackAt(b, f, p.Aux, at, lane), v, v, p.A, p.A);
        else if (t.Act is not null) v = Formula(b, k, t.Act, v, v, v, p.A, p.A);
        if (t.Residual) v = v + Codec.UnpackAt(b, f, p.Residual, at, lane);
        return v;
    }
}
