// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on Vulkan (VulkanOps.Boxes.cs; Abstraction/Boxes.cs), CUDA's (Cuda/Ptx.Boxes.cs): an invocation per
// stored word of the output (a block format's by its encoder), each value its pair's loss, or a component of its pair's
// gradient times the pair's dloss, from the corners read where they are stored, in the float lane by Boxes.Reference's
// formulas and order.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Slots: predicted, target, dloss (the backward's), y, n (the outputs: count, or 4·count), eps (lane).</summary>
    public static SpvKernel BoxLoss(Vulkan vk, Fmt f, Lane lane, BoxOverlap overlap, bool backward)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"box_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}_{f.Name}_{lane.Name}", Threads, vk.Controls);
        V pp = b.Arg(0), pt = b.Arg(1), pd = b.Arg(2), py = b.Arg(3), n = b.Arg(4, K.U32), eps = b.Arg(5, k);
        StoreEach(b, f, lane, py, n, at => BoxValue(b, f, lane, k, overlap, backward, pp, pt, pd, eps, at));
        return b.Build();
    }

    /// <summary>Into blocks (f32). Slots: predicted, target, dloss, eps (f32).</summary>
    public static SpvKernel BlockBoxLoss(Vulkan vk, Fmt f, BoxOverlap overlap, bool backward) =>
        BlockEncoder(vk, f, $"box_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}", b =>
        {
            V pp = b.Arg(0), pt = b.Arg(1), pd = b.Arg(2), eps = b.Arg(3, K.F32);
            return (at, _) => BoxValue(b, f, Lane.F32, K.F32, overlap, backward, pp, pt, pd, eps, at);
        });

    // Output at's value: the loss of pair at, or component at % 4 of pair at / 4's gradient times the pair's dloss.
    private static V BoxValue(Spv b, Fmt f, Lane lane, K k, BoxOverlap overlap, bool backward, V pp, V pt, V pd, V eps, V at)
    {
        var pair = backward ? at >> 2 : at;
        V Corner(V ptr, int c) => Codec.UnpackAt(b, f, ptr, pair * 4.0 + c, lane);
        V x1 = Corner(pp, 0), y1 = Corner(pp, 1), x2 = Corner(pp, 2), y2 = Corner(pp, 3);
        V gx1 = Corner(pt, 0), gy1 = Corner(pt, 1), gx2 = Corner(pt, 2), gy2 = Corner(pt, 3);
        V zero = b.C(k, 0), one = b.C(k, 1), half = b.C(k, 0.5), two = b.C(k, 2);
        var iw = b.Min(x2, gx2) - b.Max(x1, gx1);
        var ih = b.Min(y2, gy2) - b.Max(y1, gy1);
        var overlaps = (iw > 0.0) & (ih > 0.0);
        var inter = b.Select(overlaps, iw * ih, zero);
        V wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
        var union = wp * hp + wg * hg - inter;
        var ue = union + eps;
        var iou = inter / ue;
        var loss = one - iou;
        var cw = b.Max(x2, gx2) - b.Min(x1, gx1);
        var ch = b.Max(y2, gy2) - b.Min(y1, gy1);
        var dInter = -(one / ue);
        var dUnion = iou / ue;
        V dCw = zero, dCh = zero, dPx = zero, dPy = zero, dW = zero, dH = zero;
        if (overlap == BoxOverlap.GIoU)
        {
            var area = cw * ch;
            var ae = area + eps;
            loss = loss + (area - union) / ae;
            dUnion = dUnion - one / ae;
            var dArea = ue / (ae * ae);
            (dCw, dCh) = (dArea * ch, dArea * cw);
        }
        else if (overlap is BoxOverlap.DIoU or BoxOverlap.CIoU)
        {
            var diag = cw * cw + ch * ch + eps;
            var dx = ((x1 - gx1) + (x2 - gx2)) * half;
            var dy = ((y1 - gy1) + (y2 - gy2)) * half;
            var rho = dx * dx + dy * dy;
            loss = loss + rho / diag;
            (dPx, dPy) = (two * dx / diag, two * dy / diag);
            var dDiag = -(rho / (diag * diag));
            (dCw, dCh) = (two * dDiag * cw, two * dDiag * ch);
            if (overlap == BoxOverlap.CIoU)
            {
                var diff = Fn.Atan(b, wg / hg) - Fn.Atan(b, wp / hp);
                var af = b.C(k, Boxes.AspectFactor);
                var v = af * diff * diff;
                var alpha = v / (one - iou + v + eps);
                loss = loss + alpha * v;
                var held = one - alpha * alpha;                             // ∂(α·v)/∂IoU = α², ∂(α·v)/∂v = α·(2 − α)
                (dInter, dUnion) = (dInter * held, dUnion * held);
                var kk = two * af * (alpha * (two - alpha)) * diff / (wp * wp + hp * hp);
                (dW, dH) = (-(kk * hp), kk * wp);
            }
        }

        if (!backward) return loss;
        dInter = dInter - dUnion;
        dW = dW + dUnion * hp;
        dH = dH + dUnion * wp;
        var ix = b.Select(overlaps, ih * dInter, zero);
        var iy = b.Select(overlaps, iw * dInter, zero);
        V Above(V p, V q) => b.Select(p > q, one, b.Select(p == q, half, zero));
        V Below(V p, V q) => Above(q, p);
        var g0 = -dW + dPx * half - ix * Above(x1, gx1) - dCw * Below(x1, gx1);
        var g1 = -dH + dPy * half - iy * Above(y1, gy1) - dCh * Below(y1, gy1);
        var g2 = dW + dPx * half + ix * Below(x2, gx2) + dCw * Above(x2, gx2);
        var g3 = dH + dPy * half + iy * Below(y2, gy2) + dCh * Above(y2, gy2);
        var comp = at & 3UL;
        var g = b.Select(comp == 0.0, g0, b.Select(comp == 1.0, g1, b.Select(comp == 2.0, g2, g3)));
        return g * Codec.UnpackAt(b, f, pd, pair, lane);
    }
}
