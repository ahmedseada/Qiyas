// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A scaled weight's values on Vulkan (Products.cs ScaledWeight), CUDA's (Cuda/Ptx.Scaled.cs): the code decoded into the lane
// where it is stored, its zero subtracted and its scale multiplied there; the product's B tile loads take them so
// (MatMulScaled, the product's tiles: Kernels.Products.cs), and ExpandScaled's kernels store them as the type.
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
