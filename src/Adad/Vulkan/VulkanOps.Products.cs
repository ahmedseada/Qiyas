// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The products on Vulkan (Kernels.Products.cs): the lane by the bounds of the operands' values and k (CUDA's sum lane:
// f32 while exact, the integers' own, f64), every batch's operands read where they are stored (transposed, at their
// offsets, broadcast at stride 0), C written in the lane where the lane is the type's storage, otherwise kept as lane
// values and stored as the type once (a block format's by its encoder). The block-scaled product (the blocks' integer
// dot products) is not on Vulkan: the decoded route stands.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    public string MatMul<TN>(ulong a, ulong b, ulong c, int m, int n, int k) where TN : struct, IAdad<TN> =>
        MatMulView<TN>(new ProductView(m, n, k, false, false), a, b, c);

    /// <summary>C = op(A)·op(B) over the view's batches; returns the product's lane.</summary>
    public string MatMulView<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var (aLo, aHi) = Range<TN>(a, (int)v.ValuesA);
        var (bLo, bHi) = Range<TN>(b, (int)v.ValuesB);
        var lane = SumLane(f, new Bounds(aLo, aHi, bLo, bHi, v.K));
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return lane.Name;
        bool direct = f.Natural.Name == lane.Name && f.Bits == lane.Size * 8;
        long valuesC = v.ValuesC;
        ulong lanes = direct ? c : vk.Rent(valuesC * lane.Size);
        try
        {
            var fn = vk.Cached($"matmul:{f.Name}:{lane.Name}:{v.TransA}:{v.TransB}", () => VkKernels.MatMul(vk, f, lane, v.TransA, v.TransB));
            int tilesM = (v.M + VkKernels.TileM - 1) / VkKernels.TileM;
            vk.Launch(fn, Groups(v.N, VkKernels.TileN), (uint)((long)tilesM * v.Batch), a, b, lanes, Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K),
                Vulkan.U(v.StrideAOrDefault), Vulkan.U(v.StrideBOrDefault), Vulkan.U(v.StrideCOrDefault), Vulkan.U(tilesM), 0, 0);
            if (!direct)
            {
                // C's values between the batches' regions (a stride past m·n) kept as they are stored.
                long span = (long)v.M * v.N, period = v.Batch > 1 ? Math.Max(span, v.StrideCOrDefault) : span;
                BackKeep(f, lane, lanes, c, valuesC, period, span);
            }
        }
        finally
        {
            if (!direct) vk.Return(lanes);
        }

        return lane.Name;
    }

    // A product's lane values into C as the type: the values at positions p with p mod period below span (and below
    // count), the rest of C's words (between the batches' regions, past its last value) kept as they are stored.
    private void BackKeep(Fmt f, Lane lane, ulong lanes, ulong c, long count, long period, long span)
    {
        if (f.IsBlock)
        {
            // A block re-encoded whole: its kept values decoded from C as it is stored.
            Encode(f, count, vk.Cached($"block_lanes_keep:{lane.Name}:{f.Name}", () => VkKernels.BlockFromLanesKeep(vk, lane, f)), c, lanes, c, Vulkan.U(period), Vulkan.U(span));
            return;
        }

        var back = vk.Cached($"back_keep:{lane.Name}:{f.Name}", () => VkKernels.FromLanesKeep(vk, lane, f));
        vk.Launch(back, Groups(f.Words(count), VkKernels.Threads), 1, lanes, c, Vulkan.U(count), Vulkan.U(period), Vulkan.U(span));
    }

    /// <summary>C = the epilogue's y of op(A)·op(B) (ProductEpilogue): each batch's product into the float lane's sums (the
    /// operands read where they are stored), the low-rank term's into sums of its own, then the store
    /// (Kernels.ProductStore.cs; aux first where it is written); returns the lane's name.</summary>
    public string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IFloatAdad<TN>
    {
        var terms = t.Validate(e, v);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return lane.Name;
        long span = (long)v.M * v.N;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong sums = Rent(v.ValuesC * lane.Size), low = 0;
            var one = new ProductView(v.M, v.N, v.K, v.TransA, v.TransB);
            for (int i = 0; i < v.Batch; i++)
                Product(one, lane, a, f, i * v.StrideAOrDefault, b, f, i * v.StrideBOrDefault, sums + (ulong)(i * span * lane.Size));
            if (terms.Low)
            {
                low = Rent(span * lane.Size);
                Product(new ProductView(v.M, v.N, e.Rank, false, e.TransV), lane, t.U, f, 0, t.V, f, 0, low);
            }

            if (terms.Aux) ProductStore(f, lane, terms.Input, sums, low, t.Bias, 0, 0, t.Aux, v.ValuesC, 1, v.N, e.ActA, e.Beta);
            ProductStore(f, lane, terms, sums, low, t.Bias, t.Residual, t.Aux, c, v.ValuesC, 1, v.N, e.ActA, e.Beta);
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    /// <summary>y (count values of f) = the store terms' value of the lane sums (Kernels.ProductStore.cs): the bias the
    /// channel (at / cols) mod oc's, an absent term's pointer 0.</summary>
    private void ProductStore(Fmt f, Lane lane, StoreTerms t, ulong sums, ulong low, ulong bias, ulong residual, ulong aux, ulong y, long count, int cols, int oc, double actA, double beta)
    {
        if (f.IsBlock)
        {
            Encode(f, count, vk.Cached($"block_product_store:{f.Name}:{t.Key}", () => VkKernels.BlockProductStore(vk, f, t)), y,
                sums, low, bias, residual, aux, y, Vulkan.U(cols), Vulkan.U(oc), Vulkan.F((float)actA), Vulkan.F((float)beta));
            return;
        }

        var k = vk.Cached($"product_store:{f.Name}:{lane.Name}:{t.Key}", () => VkKernels.ProductStore(vk, f, lane, t));
        vk.Launch(k, Groups(f.Words(count), VkKernels.Threads), 1, sums, low, bias, residual, aux, y, Vulkan.U((int)count), Vulkan.U(cols), Vulkan.U(oc), Slot(lane, actA), Slot(lane, beta));
    }

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }

    public string MatMulViewBlocks<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN> =>
        throw new NotSupportedException("vulkan: the block-scaled product has no Vulkan kernels (the decoded route, MatMulView, stands)");

    public bool BlockProductAvailable => false;
}
