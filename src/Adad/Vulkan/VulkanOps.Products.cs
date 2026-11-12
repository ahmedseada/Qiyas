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
    /// operands read where they are stored), the second product's into sums of its own, then the store
    /// (Kernels.ProductStore.cs; the sums kept first where they are); returns the lane's name.</summary>
    public string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IFloatAdad<TN>
    {
        var terms = t.Validate(e, v);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return lane.Name;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong sums = Rent(v.ValuesC * lane.Size);
            ViewSums(v, lane, f, a, b, sums);
            EpilogueStore(f, lane, v, e, terms, t, sums, c, Rent);
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    /// <summary>y = the epilogue's y of x · Wᵀ, W the scaled weight's values made in the lane as the product's tiles read each
    /// code (Kernels.Scaled.cs); returns the lane's name.</summary>
    public string MatMulScaled<TN, TW>(ScaledWeight w, int m, ProductEpilogue e, ulong x, ulong codes, ulong scales, ulong zeros, ulong y, ProductTerms<ulong> t)
        where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        var v = w.View(m);
        var terms = t.Validate(e, v);
        Fmt f = Fmt.Of<TN>(), fw = Fmt.Of<TW>();
        var lane = RowLane(f);
        if (v.M == 0 || v.N == 0) return lane.Name;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong sums = Rent(v.ValuesC * lane.Size);
            bool z = zeros != 0;
            var fn = vk.Cached($"matmul_scaled:{f.Name}:{fw.Name}:{lane.Name}:{z}", () => VkKernels.MatMulScaled(vk, f, fw, lane, z));
            vk.Launch(fn, Groups(v.N, VkKernels.TileN), Groups(v.M, VkKernels.TileM), x, codes, sums, Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K),
                scales, zeros, Vulkan.U(w.BlockRows), Vulkan.U(w.BlockColumns), Vulkan.U(w.ScaleColumns));
            EpilogueStore(f, lane, v, e, terms, t, sums, y, Rent);
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    /// <summary>output = the scaled weight's values as TN (Kernels.Scaled.cs); returns the lane's name.</summary>
    public string ExpandScaled<TN, TW>(ScaledWeight w, ulong codes, ulong scales, ulong zeros, ulong output) where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        Fmt f = Fmt.Of<TN>(), fw = Fmt.Of<TW>();
        var lane = RowLane(f);
        bool z = zeros != 0;
        if (w.Values == 0) return lane.Name;
        if (f.IsBlock)
        {
            Encode(f, w.Values, vk.Cached($"block_expand_scaled:{f.Name}:{fw.Name}:{z}", () => VkKernels.BlockExpandScaled(vk, f, fw, z)), output,
                codes, scales, zeros, Vulkan.U(w.K), Vulkan.U(w.BlockRows), Vulkan.U(w.BlockColumns), Vulkan.U(w.ScaleColumns));
            return lane.Name;
        }

        var k = vk.Cached($"expand_scaled:{f.Name}:{fw.Name}:{lane.Name}:{z}", () => VkKernels.ExpandScaled(vk, f, fw, lane, z));
        vk.Launch(k, Groups(f.Words(w.Values), VkKernels.Threads), 1, codes, scales, zeros, output, Vulkan.U((int)w.Values), Vulkan.U(w.K),
            Vulkan.U(w.BlockRows), Vulkan.U(w.BlockColumns), Vulkan.U(w.ScaleColumns));
        return lane.Name;
    }

    /// <summary>The scaled weight's codes, scales and zero points from its values (IProducts.Quantize; Kernels.Scaled.cs):
    /// the scales and zero points lane values, stored as TN before the next pass reads them. Returns the lane's name.</summary>
    public string Quantize<TN, TW>(ScaledWeight w, ulong values, ulong codes, ulong scales, ulong zeros) where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        Fmt f = Fmt.Of<TN>(), fw = Fmt.Of<TW>();
        ScaledWeight.ValidateCodes(fw, zeros != 0);
        var lane = RowLane(f);
        if (w.Values == 0) return lane.Name;
        bool z = zeros != 0, own = Own(f);
        ulong min = Slot(lane, TW.Min), max = Slot(lane, TW.Max);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong sl = own ? scales : Rent(w.Scales * lane.Size), lows = z ? Rent(w.Scales * lane.Size) : 0;
            var range = vk.Cached($"quantize_range:{f.Name}:{lane.Name}:{z}", () => VkKernels.QuantizeRange(vk, f, lane, z));
            vk.Launch(range, (uint)w.Scales, 1, values, sl, lows, Vulkan.U(w.N), Vulkan.U(w.K), Vulkan.U(w.BlockRows), Vulkan.U(w.BlockColumns), Vulkan.U(w.ScaleColumns), min, max);
            if (!own) Back(f, lane, sl, scales, w.Scales);
            if (z)
            {
                ulong zl = own ? zeros : Rent(w.Scales * lane.Size);
                var zfn = vk.Cached($"quantize_zeros:{f.Name}:{lane.Name}", () => VkKernels.QuantizeZeros(vk, f, lane));
                vk.Launch(zfn, Groups(w.Scales, VkKernels.Threads), 1, scales, lows, zl, Vulkan.U((int)w.Scales), min, max);
                if (!own) Back(f, lane, zl, zeros, w.Scales);
            }

            var cfn = vk.Cached($"quantize_codes:{f.Name}:{fw.Name}:{lane.Name}:{z}", () => VkKernels.QuantizeCodes(vk, f, fw, lane, z));
            vk.Launch(cfn, Groups(fw.Words(w.Values), VkKernels.Threads), 1, values, scales, zeros, codes, Vulkan.U((int)w.Values), Vulkan.U(w.K),
                Vulkan.U(w.BlockRows), Vulkan.U(w.BlockColumns), Vulkan.U(w.ScaleColumns), min, max);
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    // Every batch's op(A)·op(B) of the view into lane sums (the batches' C contiguous, m·n apart), the operands read at
    // their offsets.
    private void ViewSums(ProductView v, Lane lane, Fmt f, ulong a, ulong b, ulong sums)
    {
        long span = (long)v.M * v.N;
        var one = new ProductView(v.M, v.N, v.K, v.TransA, v.TransB);
        for (int i = 0; i < v.Batch; i++)
            Product(one, lane, a, f, i * v.StrideAOrDefault, b, f, i * v.StrideBOrDefault, sums + (ulong)(i * span * lane.Size));
    }

    // After a product's sums (lane storage, the view's C contiguous): the second product's into sums of its own, then the
    // store (the sums kept first where they are).
    private void EpilogueStore(Fmt f, Lane lane, ProductView v, ProductEpilogue e, StoreTerms terms, ProductTerms<ulong> t, ulong sums, ulong c, Func<long, ulong> rent)
    {
        ulong second = 0;
        if (terms.Second)
        {
            second = rent(v.ValuesC * lane.Size);
            ViewSums(e.Second!, lane, f, t.SecondA, t.SecondB, second);
        }

        if (terms.Keep) ProductStore(f, lane, terms.KeepPass, sums, second, t.Bias, 0, 0, t.Sums, v.ValuesC, 1, v.N, e.ActA, e.Beta);
        ProductStore(f, lane, terms, sums, second, t.Bias, t.Residual, t.ActInputs, c, v.ValuesC, 1, v.N, e.ActA, e.Beta);
    }

    /// <summary>y (count values of f) = the store terms' value of the lane sums (Kernels.ProductStore.cs): the bias the
    /// channel (at / cols) mod oc's, an absent term's pointer 0.</summary>
    private void ProductStore(Fmt f, Lane lane, StoreTerms t, ulong sums, ulong second, ulong bias, ulong residual, ulong actInputs, ulong y, long count, int cols, int oc, double actA, double beta)
    {
        if (f.IsBlock)
        {
            Encode(f, count, vk.Cached($"block_product_store:{f.Name}:{t.Key}", () => VkKernels.BlockProductStore(vk, f, t)), y,
                sums, second, bias, residual, actInputs, y, Vulkan.U(cols), Vulkan.U(oc), Vulkan.F((float)actA), Vulkan.F((float)beta));
            return;
        }

        var k = vk.Cached($"product_store:{f.Name}:{lane.Name}:{t.Key}", () => VkKernels.ProductStore(vk, f, lane, t));
        vk.Launch(k, Groups(f.Words(count), VkKernels.Threads), 1, sums, second, bias, residual, actInputs, y, Vulkan.U((int)count), Vulkan.U(cols), Vulkan.U(oc), Slot(lane, actA), Slot(lane, beta));
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
