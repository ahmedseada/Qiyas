// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The products on Vulkan (Kernels.Products.cs): the lane by the bounds of the operands' values and k (CUDA's sum lane:
// f32 while exact, the integers' own, f64), every batch's operands read where they are stored (transposed, at their
// offsets, broadcast at stride 0), C written in the lane where the lane is the type's storage, otherwise kept as lane
// values and stored as the type once. Block formats are refused (no Vulkan block encoder yet), and so is the
// block-scaled product.
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
        if (f.IsBlock) throw Missing($"matmul of {f.Name} (a block format)");
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
                var back = vk.Cached($"back_keep:{lane.Name}:{f.Name}", () => VkKernels.FromLanesKeep(vk, lane, f));
                vk.Launch(back, Groups(f.Words(valuesC), VkKernels.Threads), 1, lanes, c, Vulkan.U(valuesC), Vulkan.U(period), Vulkan.U(span));
            }
        }
        finally
        {
            if (!direct) vk.Return(lanes);
        }

        return lane.Name;
    }

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }

    public string MatMulViewBlocks<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN> => throw Missing("block-scaled matmul");

    public bool BlockProductAvailable => false;
}
