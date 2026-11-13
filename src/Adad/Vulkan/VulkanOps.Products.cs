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
        var bounds = new Bounds(aLo, aHi, bLo, bHi, v.K);
        var lane = SumLane(f, bounds);
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return lane.Name;
        if (UnitsKind(f, bounds) is string kind) return MatMulUnits<TN>(v, kind, a, b, c);
        if (UseBlocks(f, v, () => MatMulViewBlocks<TN>(v, a, b, c), () => MatMulPlain(f, lane, v, a, b, c))) return MatMulViewBlocks<TN>(v, a, b, c);
        if (f.Is("float64") && lane == Lane.F64 && Sliced(v, a, 0, b, 0, c, () => MatMulPlain(f, lane, v, a, b, c))) return "f64x";
        return MatMulPlain(f, lane, v, a, b, c);
    }

    // C = op(A)·op(B) over the view's batches in the lane by the plain product kernel (FMAs), the operands read where they
    // are stored; returns the lane's name.
    private string MatMulPlain(Fmt f, Lane lane, ProductView v, ulong a, ulong b, ulong c)
    {
        bool direct = f.Natural.Name == lane.Name && f.Bits == lane.Size * 8;
        long valuesC = v.ValuesC;
        ulong lanes = direct ? c : vk.Rent(valuesC * lane.Size);
        try
        {
            var t = Tiling(lane, v.M, v.N, v.K, v.Batch, tt => PlainLaunch(tt, v, lane, a, f, 0, b, f, 0, lanes));
            PlainLaunch(t, v, lane, a, f, 0, b, f, 0, lanes);
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

    /// <summary>A plain product's tiling: values a side each invocation holds (8: a 128 × 128 tile a workgroup; 4: 64 × 64)
    /// and the slices k is split in (1: none).</summary>
    internal readonly record struct PlainTiling(int Micro, int Slices);

    /// <summary>The gate's override of the measured tiling (null: measured); a split applies to a single product's k.</summary>
    internal PlainTiling? TilingForced;

    private readonly Dictionary<string, PlainTiling> _tilings = [];

    // The plain product's tiling for a shape and lane: the one this device runs fastest, measured once by run (the product
    // at a tiling into its real output), among 8 or 4 values a side (8 in a 32-bit lane: 64 accumulators of 64 bits would be
    // 128 registers) and, for a single product whose C has fewer than 256 tiles, k split in 2 to 16 slices of at least 256.
    // Every route over the same shape and lane takes the same tiling (the fused convolution and its composition: the same
    // slices, so the same sums).
    private PlainTiling Tiling(Lane lane, int m, int n, int k, int batch, Action<PlainTiling> run)
    {
        if (TilingForced is PlainTiling forced) return forced with { Micro = lane.Wide ? 4 : forced.Micro, Slices = batch == 1 ? forced.Slices : 1 };
        string key = $"{lane.Name}:{m}x{n}x{k}:{batch}";
        if (_tilings.TryGetValue(key, out var known)) return known;
        var candidates = new List<PlainTiling>();
        foreach (int micro in lane.Wide ? new[] { 4 } : new[] { 8, 4 })
        {
            candidates.Add(new(micro, 1));
            long tiles = (long)Groups(n, VkKernels.Tile(micro)) * Groups(m, VkKernels.Tile(micro));
            if (batch == 1 && tiles < 256)
                for (int s = 2; s <= 16 && k / s >= 256; s *= 2) candidates.Add(new(micro, s));
        }

        if (candidates.Count == 1) return _tilings[key] = candidates[0];
        var best = candidates[0];
        double bestMs = double.MaxValue;
        var times = new List<string>();
        foreach (var t in candidates)
        {
            run(t);
            vk.Synchronize();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) run(t);
            vk.Synchronize();
            double ms = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            times.Add($"{t.Micro}/{t.Slices} {ms:0.000}");
            if (ms < bestMs) (bestMs, best) = (ms, t);
        }

        _unitsTimes.Add($"plain product {key} (micro/slices ms): {string.Join(", ", times)}: {best.Micro}/{best.Slices}");
        return _tilings[key] = best;
    }

    // C (lane values at c, the view's batches C's stride apart) = op(A)·op(B) by the plain kernel at a tiling, A stored as
    // fa from value aOff, B as fb from bOff; split, each slice's partial C into lane storage and the slices added in order.
    private void PlainLaunch(PlainTiling t, ProductView v, Lane lane, ulong a, Fmt fa, long aOff, ulong b, Fmt fb, long bOff, ulong c)
    {
        if (aOff > uint.MaxValue || bOff > uint.MaxValue) throw new NotSupportedException("vulkan: a product's operand past 2^32 values");
        int tile = VkKernels.Tile(t.Micro), tilesM = (v.M + tile - 1) / tile;
        bool split = t.Slices > 1 && v.K > 0;
        // A block format's operand read as runs of values where its stored rows, offset and batch stride are whole runs of 8.
        bool runsA = fa.IsBlock && (v.TransA ? v.M : v.K) % 8 == 0 && aOff % 8 == 0 && v.StrideAOrDefault % 8 == 0;
        bool runsB = fb.IsBlock && (v.TransB ? v.K : v.N) % 8 == 0 && bOff % 8 == 0 && v.StrideBOrDefault % 8 == 0;
        var fn = vk.Cached($"matmul:{fa.Name}:{fb.Name}:{lane.Name}:{v.TransA}:{v.TransB}:{t.Micro}:{split}:{runsA}:{runsB}",
            () => VkKernels.MatMul(vk, fa, fb, lane, v.TransA, v.TransB, t.Micro, split, runsA, runsB));
        if (!split)
        {
            vk.Launch(fn, Groups(v.N, tile), (uint)((long)tilesM * v.Batch), a, b, c, Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K),
                Vulkan.U(v.StrideAOrDefault), Vulkan.U(v.StrideBOrDefault), Vulkan.U(v.StrideCOrDefault), Vulkan.U(tilesM), Vulkan.U(aOff), Vulkan.U(bOff));
            return;
        }

        long mn = (long)v.M * v.N;
        int kslice = ((v.K + t.Slices - 1) / t.Slices + VkKernels.TileK - 1) / VkKernels.TileK * VkKernels.TileK, slices = (v.K + kslice - 1) / kslice;
        ulong parts = vk.Rent(slices * mn * lane.Size);
        try
        {
            vk.Launch(fn, Groups(v.N, tile), (uint)(tilesM * slices), a, b, parts, Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K), 0, 0, Vulkan.U(mn), Vulkan.U(tilesM),
                Vulkan.U(aOff), Vulkan.U(bOff), Vulkan.U(kslice));
            JoinSlices(lane, parts, c, mn, slices);
        }
        finally { vk.Return(parts); }
    }

    // c = the slices' partial sums (count values each, consecutive) added in order.
    private void JoinSlices(Lane lane, ulong parts, ulong c, long count, int slices)
    {
        var join = vk.Cached($"join_slices:{lane.Name}", () => VkKernels.JoinSlices(vk, lane));
        vk.Launch(join, Groups(count, VkKernels.Threads), 1, parts, c, Vulkan.U(count), Vulkan.U(slices));
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

    /// <summary>C = the epilogue's y of op(A)·op(B) (ProductEpilogue): each batch's product into the lane's sums (the
    /// operands read where they are stored), the second product's into sums of its own, then the store
    /// (Kernels.ProductStore.cs; the sums kept first where they are). A float type's lane is its float lane; another
    /// type's its products' by the bounds of every value the store makes (EpilogueLane); returns the lane's name.</summary>
    public string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IAdad<TN>
    {
        var terms = t.Validate(e, v);
        var f = Fmt.Of<TN>();
        e.ValidateType(f.Name, f.IsFloat);
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return f.IsFloat ? RowLane(f).Name : f.Natural.Name;
        var lane = f.IsFloat ? RowLane(f) : EpilogueLane<TN>(v, e, terms, a, b, c, t);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong sums = Rent(v.ValuesC * lane.Size);
            ViewSums(v, lane, f, a, b, sums);
            EpilogueStore(f, lane, v, e, terms, t, sums, c, 1, v.N, Rent);
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
            EpilogueStore(f, lane, v, e, terms, t, sums, y, 1, v.N, Rent);
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

    // A type that is not float: its products' lane by the bounds of every value the store makes (ProductEpilogue.IntegerSums,
    // the ranges read on the device); an unsigned type whose values may fall below 0 takes the float64 lane (ExactSums).
    private Lane EpilogueLane<TN>(ProductView v, ProductEpilogue e, StoreTerms terms, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        (double, double) R(ulong x, long n) => Range<TN>(x, (int)n);
        var w = e.Second;
        var (bounds, negative) = e.IntegerSums(f.Name, f.Signed, v.K, R(a, v.ValuesA), R(b, v.ValuesB), w is null ? null : R(t.SecondA, w.ValuesA),
            w is null ? null : R(t.SecondB, w.ValuesB), terms.Bias ? R(t.Bias, v.N) : null, terms.Beta ? R(c, v.ValuesC) : null,
            terms.Residual ? R(t.Residual, v.ValuesC) : null, terms.Grad is not null ? R(t.ActInputs, v.ValuesC) : null);
        return negative && !f.Signed ? Lane.F64 : SumLane(f, bounds);
    }

    // After a product's sums (lane storage, the view's C contiguous): the second product's into sums of its own, then the
    // store (the sums kept first where they are).
    private void EpilogueStore(Fmt f, Lane lane, ProductView v, ProductEpilogue e, StoreTerms terms, ProductTerms<ulong> t, ulong sums, ulong c, int cols, int oc, Func<long, ulong> rent)
    {
        ulong second = 0;
        if (terms.Second)
        {
            second = rent(v.ValuesC * lane.Size);
            ViewSums(e.Second!, lane, f, t.SecondA, t.SecondB, second);
        }

        if (terms.Keep) ProductStore(f, lane, terms.KeepPass, sums, second, t.Bias, 0, 0, t.Sums, v.ValuesC, cols, oc, e.ActA, e.Beta, e.ActB, before: c);
        ProductStore(f, lane, terms, sums, second, t.Bias, t.Residual, t.ActInputs, c, v.ValuesC, cols, oc, e.ActA, e.Beta, e.ActB);
    }

    /// <summary>y (count values of f) = the store terms' value of the lane sums (Kernels.ProductStore.cs): the bias the
    /// channel (at / cols) mod oc's, an absent term's pointer 0.</summary>
    private void ProductStore(Fmt f, Lane lane, StoreTerms t, ulong sums, ulong second, ulong bias, ulong residual, ulong actInputs, ulong y, long count, int cols, int oc,
        double actA, double beta, double actB = 0, ulong before = 0)
    {
        if (before == 0) before = y;
        if (f.IsBlock)
        {
            Encode(f, count, vk.Cached($"block_product_store:{f.Name}:{t.Key}", () => VkKernels.BlockProductStore(vk, f, t)), y,
                sums, second, bias, residual, actInputs, before, Vulkan.U(cols), Vulkan.U(oc), Vulkan.F((float)actA), Vulkan.F((float)actB), Vulkan.F((float)beta));
            return;
        }

        var k = vk.Cached($"product_store:{f.Name}:{lane.Name}:{t.Key}", () => VkKernels.ProductStore(vk, f, lane, t));
        vk.Launch(k, Groups(f.Words(count), VkKernels.Threads), 1, sums, second, bias, residual, actInputs, y, before, Vulkan.U((int)count), Vulkan.U(cols), Vulkan.U(oc), Slot(lane, actA),
            Slot(lane, actB), Slot(lane, beta));
    }

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }

    /// <summary>C = A·Bᵀ by the block-scaled product (Kernels.BlockProducts.cs) of two Q8_0 / Q4_0 operands, k in whole
    /// blocks, C encoded to the type by blocks; returns the lane's name.</summary>
    public string MatMulViewBlocks<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.Name is not ("q8_0" or "q4_0") || v.K % 32 != 0 || v.TransA || !v.TransB)
            throw new NotSupportedException($"{f.Name}: the block product takes Q8_0 or Q4_0, A·Bᵀ, with k in whole blocks");
        if (!BlockProductAvailable) throw new NotSupportedException("vulkan: the block product needs the s8 matrix unit at k = 32 and 16-bit loads");
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return "s8mma blocks";
        if (v.Batch > 1 && (v.StrideAOrDefault % 32 != 0 || v.StrideBOrDefault % 32 != 0))
            throw new NotSupportedException("vulkan: the block product's batches at whole blocks");
        var fn = vk.Cached($"matmul_blocks:{f.Name}", () => VkKernels.MatMulBlocks(vk, f));
        long mn = (long)v.M * v.N;
        ulong lanes = vk.Rent(v.ValuesC * 4);
        try
        {
            if (v.K == 0) vk.Clear(lanes, v.ValuesC * 4);                     // no terms: the empty sums (the kernel reads a block)
            for (int z = 0; z < v.Batch && v.K > 0; z++)
            {
                ulong az = a + (ulong)(z * v.StrideAOrDefault / 32 * f.BlockBytes), bz = b + (ulong)(z * v.StrideBOrDefault / 32 * f.BlockBytes);
                vk.Launch(fn, Groups(v.N, VkKernels.BlocksTile), Groups(v.M, VkKernels.BlocksTile), az, bz, lanes + (ulong)(z * v.StrideCOrDefault * 4),
                    Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K / 32));
            }

            long period = v.Batch > 1 ? Math.Max(mn, v.StrideCOrDefault) : mn;
            BackKeep(f, Lane.F32, lanes, c, v.ValuesC, period, mn);
        }
        finally { vk.Return(lanes); }

        return "s8mma blocks";
    }

    public bool BlockProductAvailable => VkKernels.BlocksRun(vk);

    /// <summary>The gate's override: true takes the block product wherever the view allows it, false never, null as measured.</summary>
    internal bool? BlockProducts;

    private readonly Dictionary<string, bool> _blocksFaster = [];

    // The block product or the decoded route for a view A·Bᵀ of Q8_0 / Q4_0 (k whole blocks), whichever this device
    // measures faster at the shape and format (once).
    private bool UseBlocks(Fmt f, ProductView v, Func<string> blocks, Func<string> decoded)
    {
        if (f.Name is not ("q8_0" or "q4_0") || v.TransA || !v.TransB || v.K % 32 != 0 || v.K == 0 || !BlockProductAvailable
            || (v.Batch > 1 && (v.StrideAOrDefault % 32 != 0 || v.StrideBOrDefault % 32 != 0))) return false;
        if (BlockProducts is bool forced) return forced;
        string key = $"{f.Name}:{v.M}x{v.N}x{v.K}:{v.Batch}";
        if (_blocksFaster.TryGetValue(key, out bool known)) return known;
        var (tb, td) = Measure.Faster(() => blocks(), () => decoded(), vk.Synchronize);
        _unitsTimes.Add($"product {key} A·Bᵀ: block-scaled {tb:0.000} ms, decoded {td:0.000} ms");
        return _blocksFaster[key] = tb <= td;
    }
}
