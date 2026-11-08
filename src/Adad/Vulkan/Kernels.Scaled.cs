// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A scaled weight's values on Vulkan (Products.cs ScaledWeight), CUDA's (Cuda/Ptx.Scaled.cs): the code decoded into the lane
// where it is stored, its zero subtracted and its scale multiplied there; the product's B tile loads take them so
// (MatMulScaled, the product's tiles: Kernels.Products.cs), and ExpandScaled's kernels store them as the type. Quantize's
// kernels (CUDA's QuantizeRange, QuantizeZeros, QuantizeCodes): each block's range and scale (a workgroup a scale), the zero
// points from the stored scales, the codes (StoreEach into TW, each rounded once).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>(code − zero)·scale of the weight's value at (row, col) from its code v (a lane value): the scale and zero
    /// point at (row / br)·sc + col / bc of fs at ps and pz (none: no zero points); inside false reads index 0.</summary>
    internal static V ScaleCode(Spv b, Fmt fs, Lane lane, V v, V ps, V? pz, V row, V col, V br, V bc, V sc, V? inside = null)
    {
        var si = row / br * sc + col / bc;
        if (inside is { } p) si = b.Select(p, si, b.U(0));
        if (pz is { } z) v = v - Codec.UnpackAt(b, fs, z, si, lane);
        return v * Codec.UnpackAt(b, fs, ps, si, lane);
    }

    /// <summary>y (lane values, m × n) = x (m × k, stored as fa) · Wᵀ, W the scaled weight's values from its codes (n × k,
    /// stored as fw) and its scales and zero points (fa). Arguments: x, codes, y, m, n, k, scales, zeros, br, bc, sc;
    /// workgroups (⌈n/64⌉, ⌈m/64⌉).</summary>
    public static SpvKernel MatMulScaled(Vulkan vk, Fmt fa, Fmt fw, Lane lane, bool zeros)
    {
        var b = new Spv($"matmul_scaled_{fa.Name}_{fw.Name}_{lane.Name}{(zeros ? "_z" : "")}", TileSide, vk.Controls, TileSide);
        V pa = b.Arg(0), pw = b.Arg(1), pc = b.Arg(2);
        V M = b.Arg(3, K.U32), N = b.Arg(4, K.U32), Kd = b.Arg(5, K.U32);
        V ps = b.Arg(6), pz = b.Arg(7);
        V br = b.Arg(8, K.U32), bc = b.Arg(9, K.U32), sc = b.Arg(10, K.U32);
        ProductTiles(b, lane, false, true, M, N, Kd, b.GroupY * TileM, pc, b.U(0),
            (gr, gk, inside) => Codec.UnpackAt(b, fa, pa, b.Select(inside, gr * Kd + gk, b.U(0)), lane),
            (gk, gc, inside) => ScaleCode(b, fa, lane, Codec.UnpackAt(b, fw, pw, b.Select(inside, gc * Kd + gk, b.U(0)), lane), ps, zeros ? pz : null, gc, gk, br, bc, sc, inside));
        return b.Build();
    }

    /// <summary>Each block's scale (lane values at ps) and its range's low end lo = min(x, 0) (lane values at pl, with zero
    /// points), a workgroup a scale: max(hi, −lo) / max, or (hi − lo) / (max − min). Slots: x (f), ps, pl, n (N), k, br, bc,
    /// sc, min, max (lane).</summary>
    public static SpvKernel QuantizeRange(Vulkan vk, Fmt f, Lane lane, bool zeros)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"quantize_range_{f.Name}_{lane.Name}{(zeros ? "_z" : "")}", ReduceThreads, vk.Controls);
        V px = b.Arg(0), ps = b.Arg(1), pl = b.Arg(2);
        V n = b.Arg(3, K.U32), kd = b.Arg(4, K.U32), br = b.Arg(5, K.U32), bc = b.Arg(6, K.U32), sc = b.Arg(7, K.U32), min = b.Arg(8, k), max = b.Arg(9, k);
        var tid = b.LocalIdX;
        var si = b.GroupX;
        var r0 = si / sc * br;
        var c0 = si % sc * bc;
        var rows = b.Min(n - r0, br);
        var cols = b.Select(c0 < kd, b.Min(kd - c0, bc), b.U(0));
        var lo = b.Local(b.C(k, 0));
        var hi = b.Local(b.C(k, 0));
        b.For(tid, rows * cols, b.U(ReduceThreads), e =>
        {
            var x = Codec.UnpackAt(b, f, px, (e / cols + r0) * kd + e % cols + c0, lane);
            lo.V = b.Min(lo.V, x);
            hi.V = b.Max(hi.V, x);
        });
        var top = GroupReduce(b, b.Shared(k, ReduceThreads), tid, hi.V, (x, y) => b.Max(x, y), ReduceThreads);
        var bottom = GroupReduce(b, b.Shared(k, ReduceThreads), tid, -lo.V, (x, y) => b.Max(x, y), ReduceThreads);      // −lo
        var scale = zeros ? (top + bottom) / (max - min) : b.Max(top, bottom) / max;
        b.If(tid == 0, () =>
        {
            b.Store(b.At(ps, si, lane.Size), scale);
            if (zeros) b.Store(b.At(pl, si, lane.Size), -bottom);
        });
        return b.Build();
    }

    /// <summary>Each zero point (lane values at pz) from the stored scale (f at ps) and the block's low end (lane values at
    /// pl): round(min − lo / scale) held to [min, max], 0 where the scale is 0. Slots: ps, pl, pz, n (the scales), min, max.</summary>
    public static SpvKernel QuantizeZeros(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"quantize_zeros_{f.Name}_{lane.Name}", Threads, vk.Controls);
        V ps = b.Arg(0), pl = b.Arg(1), pz = b.Arg(2);
        V n = b.Arg(3, K.U32), min = b.Arg(4, k), max = b.Arg(5, k);
        var i = b.GlobalX;
        b.If(i < n, () =>
        {
            var s = Codec.UnpackAt(b, f, ps, i, lane);
            var lo = b.Load(k, b.At(pl, i, lane.Size));
            var zero = b.Min(b.Max(b.RoundEven(min - lo / s), min), max);
            b.Store(b.At(pz, i, lane.Size), b.Select(s > 0.0, zero, b.C(k, 0)));
        });
        return b.Build();
    }

    /// <summary>The codes (fw) of the values (f): x / scale (+ the zero point) by the stored scale and zero point (f), held to
    /// [min, max], an integer code rounded to the nearest (ties to even) in the lane, each rounded to fw once (an f64 lane's
    /// value to a narrow float rounded to odd at float32 first). Slots: x, scales, zeros, codes, n (N·K), k, br, bc, sc, min,
    /// max (lane).</summary>
    public static SpvKernel QuantizeCodes(Vulkan vk, Fmt f, Fmt fw, Lane lane, bool zeros)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"quantize_codes_{f.Name}_{fw.Name}_{lane.Name}{(zeros ? "_z" : "")}", Threads, vk.Controls);
        V px = b.Arg(0), ps = b.Arg(1), pz = b.Arg(2), pc = b.Arg(3);
        V n = b.Arg(4, K.U32), kd = b.Arg(5, K.U32), br = b.Arg(6, K.U32), bc = b.Arg(7, K.U32), sc = b.Arg(8, K.U32), min = b.Arg(9, k), max = b.Arg(10, k);
        bool toOdd = Fmt.ConvertPlan(LaneFormat(lane), fw).ToOdd;
        StoreEach(b, fw, toOdd ? Lane.F32 : lane, pc, n, at =>
        {
            var si = at / kd / br * sc + at % kd / bc;
            var s = Codec.UnpackAt(b, f, ps, si, lane);
            var q = b.Select(s > 0.0, Codec.UnpackAt(b, f, px, at, lane) / s, b.C(k, 0));
            if (!fw.IsFloat) q = b.RoundEven(q);
            if (zeros) q = q + Codec.UnpackAt(b, f, pz, si, lane);
            q = b.Min(b.Max(q, min), max);
            return toOdd ? ToOddF32(b, lane, q) : q;
        });
        return b.Build();
    }

    /// <summary>The weight's values as f. Slots: codes (fw), scales, zeros (f), y, n (N·K), k, br, bc, sc.</summary>
    public static SpvKernel ExpandScaled(Vulkan vk, Fmt f, Fmt fw, Lane lane, bool zeros)
    {
        var b = new Spv($"expand_scaled_{fw.Name}_{f.Name}_{lane.Name}{(zeros ? "_z" : "")}", Threads, vk.Controls);
        V pc = b.Arg(0), ps = b.Arg(1), pz = b.Arg(2), py = b.Arg(3);
        V n = b.Arg(4, K.U32), k = b.Arg(5, K.U32), br = b.Arg(6, K.U32), bc = b.Arg(7, K.U32), sc = b.Arg(8, K.U32);
        StoreEach(b, f, lane, py, n, at => ScaleCode(b, f, lane, Codec.UnpackAt(b, fw, pc, at, lane), ps, zeros ? pz : null, at / k, at % k, br, bc, sc));
        return b.Build();
    }

    /// <summary>The weight's values into blocks (f32). Slots: codes, scales, zeros, k, br, bc, sc.</summary>
    public static SpvKernel BlockExpandScaled(Vulkan vk, Fmt f, Fmt fw, bool zeros) =>
        BlockEncoder(vk, f, $"expand_scaled_{fw.Name}{(zeros ? "_z" : "")}", b =>
        {
            V pc = b.Arg(0), ps = b.Arg(1), pz = b.Arg(2);
            V k = b.Arg(3, K.U32), br = b.Arg(4, K.U32), bc = b.Arg(5, K.U32), sc = b.Arg(6, K.U32);
            return (at, _) => ScaleCode(b, f, Lane.F32, Codec.UnpackAt(b, fw, pc, at, Lane.F32), ps, zeros ? pz : null, at / k, at % k, br, bc, sc);
        });
}
