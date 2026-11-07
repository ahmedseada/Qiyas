// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Group normalization and the rotary embedding by tables on Vulkan (Kernels.Norms.cs): CUDA's passes — the groups'
// coefficients by a workgroup a group, the outputs a thread per stored word (a block format's blocks by their encoder),
// the parameters' gradients by a workgroup a channel; the rotation's rows normalized by NormStats with a gain.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    public string GroupNorm<TN>(ulong x, ulong gamma, ulong beta, ulong mean, ulong variance, ulong y, GroupShape shape, double eps, bool given) where TN : struct, IFloatAdad<TN> =>
        Group(Fmt.Of<TN>(), x, 0, gamma, beta, mean, variance, y, 0, 0, shape, eps, given, false);

    public string GroupNormBwd<TN>(ulong x, ulong dy, ulong gamma, ulong mean, ulong variance, ulong dx, ulong dgamma, ulong dbeta, GroupShape shape, double eps, bool given)
        where TN : struct, IFloatAdad<TN> =>
        Group(Fmt.Of<TN>(), x, dy, gamma, 0, mean, variance, dx, dgamma, dbeta, shape, eps, given, true);

    private string Group(Fmt f, ulong x, ulong dy, ulong gamma, ulong beta, ulong mean, ulong variance, ulong output, ulong dgamma, ulong dbeta, GroupShape s, double eps, bool given, bool backward)
    {
        var lane = RowLane(f);
        if (s.Count == 0) return lane.Name;
        bool g = gamma != 0, bt = beta != 0;
        var coefs = vk.Cached($"group_coefs:{f.Name}:{lane.Name}:{backward}:{given}:{g}", () => VkKernels.GroupCoefs(vk, f, lane, backward, given, g));
        ulong coef = vk.Rent((long)s.Groups * 4 * lane.Size);
        try
        {
            vk.Launch(coefs, (uint)s.Groups, 1, x, dy, gamma, mean, variance, coef, Vulkan.U(s.Outer), Vulkan.U(s.Groups), Vulkan.U(s.Inner), Vulkan.U(s.Channels), Vulkan.U(s.Spatial), Slot(lane, eps));
            if (output != 0)
            {
                if (f.IsBlock)
                    Encode(f, s.Count, vk.Cached($"block_group_out:{f.Name}:{backward}:{g}:{bt}", () => VkKernels.BlockGroupOut(vk, f, backward, g, bt)), output,
                        x, dy, gamma, beta, coef, Vulkan.U(s.Groups), Vulkan.U(s.Inner), Vulkan.U(s.Channels), Vulkan.U(s.Spatial));
                else
                {
                    var k = vk.Cached($"group_out:{f.Name}:{lane.Name}:{backward}:{g}:{bt}", () => VkKernels.GroupOut(vk, f, lane, backward, g, bt));
                    vk.Launch(k, Groups(f.Words(s.Count), VkKernels.Threads), 1, x, dy, gamma, beta, coef, output, Vulkan.U(s.Count), Vulkan.U(s.Groups), Vulkan.U(s.Inner), Vulkan.U(s.Channels), Vulkan.U(s.Spatial));
                }
            }

            if (backward && (dgamma != 0 || dbeta != 0))
            {
                var ps = vk.Cached($"group_params:{f.Name}:{lane.Name}", () => VkKernels.GroupParams(vk, f, lane));
                vk.Launch(ps, (uint)s.Channels, 1, x, dy, coef, dgamma, dbeta, Vulkan.U(s.Count / s.Channels), Vulkan.U(s.Groups), Vulkan.U(s.Inner), Vulkan.U(s.Channels), Vulkan.U(s.Spatial));
            }
        }
        finally
        {
            vk.Return(coef);
        }

        return lane.Name;
    }

    public string Rope<TN>(RopeArgs a, ulong x, ulong gain, ulong cos, ulong sin, ulong positions, ulong y) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        long n = (long)a.Rows * a.Cols;
        if (n == 0) return lane.Name;
        bool norm = gain != 0, rotate = a.Half > 0;
        ulong st = norm ? vk.Rent((long)a.Rows * 2 * lane.Size) : 0;
        try
        {
            if (norm)
            {
                var stats = vk.Cached($"norm_stats:False:{f.Name}", () => VkKernels.NormStats(vk, f, lane, false));
                vk.Launch(stats, (uint)a.Rows, 1, x, 0, 0, 0, 0, 0, st, 0, Vulkan.U(a.Rows), Vulkan.U(a.Cols), Slot(lane, a.Eps), Slot(lane, 0), 0);
            }

            if (f.IsBlock)
                Encode(f, n, vk.Cached($"block_rope_out:{f.Name}:{norm}:{a.Interleaved}:{a.Inverse}:{rotate}", () => VkKernels.BlockRopeOut(vk, f, norm, a.Interleaved, a.Inverse, rotate)), y,
                    x, gain, cos, sin, positions, st, Vulkan.U(a.Cols), Vulkan.U(a.Heads), Vulkan.U(a.Steps), Vulkan.U(a.Half), Vulkan.F((float)a.Offset));
            else
            {
                var k = vk.Cached($"rope_out:{f.Name}:{lane.Name}:{norm}:{a.Interleaved}:{a.Inverse}:{rotate}", () => VkKernels.RopeOut(vk, f, lane, norm, a.Interleaved, a.Inverse, rotate));
                vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, x, gain, cos, sin, positions, st, y, Vulkan.U((int)n), Vulkan.U(a.Cols), Vulkan.U(a.Heads), Vulkan.U(a.Steps), Vulkan.U(a.Half), Slot(lane, a.Offset));
            }
        }
        finally
        {
            if (st != 0) vk.Return(st);
        }

        return lane.Name;
    }
}
