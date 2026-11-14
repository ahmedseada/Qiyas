// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on Vulkan (VulkanOps.Boxes.cs; Abstraction/Boxes.cs), CUDA's (Cuda/Ptx.Boxes.cs), from the corners
// read where they are stored, in the float lane by Boxes.Reference's formulas and order: the loss an invocation per stored
// word of the output, each value its pair's loss; the gradient an invocation per pair, its four components times the
// pair's dloss computed once and stored as the words the pair's four values fill (a block format's, by its encoder, a
// value at a time).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Slots: predicted, target, dloss (the backward's), y, n (the outputs: count, or 4·count), eps (lane). The
    /// corners and dloss read as f, y stored as `into` (f when null; a block format's values go to f32 lanes, its encoder
    /// after). The backward into a format of whole bytes: an invocation per pair (BoxPairWords).</summary>
    public static SpvKernel BoxLoss(Vulkan vk, Fmt f, Lane lane, BoxOverlap overlap, bool backward, Fmt? into = null)
    {
        var k = Kinds.Of(lane);
        var o = into ?? f;
        var b = new Spv($"box_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}_{f.Name}_{lane.Name}{(o.Name == f.Name ? "" : $"_to_{o.Name}")}", Threads, vk.Controls);
        V pp = b.Arg(0), pt = b.Arg(1), pd = b.Arg(2), py = b.Arg(3), n = b.Arg(4, K.U32), eps = b.Arg(5, k);
        if (backward && BoxPairWords(o) > 0)
        {
            int words = BoxPairWords(o);
            var pair = b.GlobalX;
            b.If(pair < n >> 2, () =>
            {
                var g = BoxPair(b, f, lane, k, overlap, true, pp, pt, pd, eps, pair);
                var natural = o.Natural;
                var packed = new V[words];
                for (int w = 0; w < words; w++)
                {
                    var word = b.CBits(Codec.WordKind(o), 0);
                    for (int i = 0; i < o.Per; i++)
                    {
                        var v = g[w * o.Per + i];
                        word = Codec.Pack(b, o, lane.Name == natural.Name ? v : Codec.Convert(b, lane, v, natural), natural, i, word);
                    }

                    packed[w] = word;
                }

                // The pair's words in one access where they are 2 or 4 (8 to 32 bytes, aligned to their size).
                if (words is 2 or 4) b.StoreN(b.At(py, pair * words, o.WordBits / 8), packed);
                else for (int w = 0; w < words; w++) b.Store(b.At(py, pair * words + w, o.WordBits / 8), packed[w]);
            });
        }
        else StoreEach(b, o, lane, py, n, at => BoxValue(b, f, lane, k, overlap, backward, pp, pt, pd, eps, at));
        return b.Build();
    }

    /// <summary>The stored words a pair's four gradient values fill (a float format of whole bytes: 4·bits / word bits), or 0
    /// where its values share words with another pair's or are blocks (then an invocation a word).</summary>
    public static int BoxPairWords(Fmt f) => f.IsBlock || 4 * f.Bits % f.WordBits != 0 ? 0 : 4 * f.Bits / f.WordBits;


    /// <summary>Into blocks (f32), each block's values computed where it is encoded (a value at a time). Slots: predicted,
    /// target, dloss, eps (f32).</summary>
    public static SpvKernel BlockBoxLoss(Vulkan vk, Fmt f, BoxOverlap overlap, bool backward) =>
        BlockEncoder(vk, f, $"box_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}", b =>
        {
            V pp = b.Arg(0), pt = b.Arg(1), pd = b.Arg(2), eps = b.Arg(3, K.F32);
            return (at, _) => BoxValue(b, f, Lane.F32, K.F32, overlap, backward, pp, pt, pd, eps, at);
        });

    // Output at's value: the loss of pair at, or component at % 4 of pair at / 4's gradient times the pair's dloss.
    private static V BoxValue(Spv b, Fmt f, Lane lane, K k, BoxOverlap overlap, bool backward, V pp, V pt, V pd, V eps, V at)
    {
        if (!backward) return BoxPair(b, f, lane, k, overlap, false, pp, pt, pd, eps, at)[0];
        var g = BoxPair(b, f, lane, k, overlap, true, pp, pt, pd, eps, at >> 2);
        var comp = at & 3UL;
        return b.Select(comp == 0.0, g[0], b.Select(comp == 1.0, g[1], b.Select(comp == 2.0, g[2], g[3])));
    }

    // A box's four corners (values 4·pair .. 4·pair + 3) as lane values: a plain format whose four values fill whole words
    // read as those words in one access (4 to 32 bytes, aligned to their size), else a value at a time.
    private static V[] Corners(Spv b, Fmt f, V ptr, V pair, Lane lane)
    {
        if (!f.IsBlock && f.WordBits == 64)
        {
            var w64 = b.LoadN(K.U64, 4, b.At(ptr, pair * 4.0, 8));
            return [.. w64.Select(w => Codec.Convert(b, f.Natural, f.IsFloat ? b.Bitcast(w, K.F64) : f.Signed ? b.Bitcast(w, K.S64) : w, lane))];
        }

        if (!f.IsBlock && f.WordBits == 32 && 4 % f.Per == 0)
        {
            int words = 4 / f.Per;
            var at = b.At(ptr, pair * (double)words, 4);
            V[] loaded = words == 1 ? [b.Load(K.U32, at)] : b.LoadN(K.U32, words, at);
            var values = new V[4];
            for (int c = 0; c < 4; c++) values[c] = Codec.Unpack(b, f, loaded[c / f.Per], c % f.Per, lane);
            return values;
        }

        return [.. Enumerable.Range(0, 4).Select(c => Codec.UnpackAt(b, f, ptr, pair * 4.0 + c, lane))];
    }

    // Pair pair's loss ([loss]), or its gradient's four components times its dloss.
    private static V[] BoxPair(Spv b, Fmt f, Lane lane, K k, BoxOverlap overlap, bool backward, V pp, V pt, V pd, V eps, V pair)
    {
        var p = Corners(b, f, pp, pair, lane);
        var t = Corners(b, f, pt, pair, lane);
        V x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
        V gx1 = t[0], gy1 = t[1], gx2 = t[2], gy2 = t[3];
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

        if (!backward) return [loss];
        dInter = dInter - dUnion;
        dW = dW + dUnion * hp;
        dH = dH + dUnion * wp;
        var ix = b.Select(overlaps, ih * dInter, zero);
        var iy = b.Select(overlaps, iw * dInter, zero);
        V Above(V p, V q) => b.Select(VkKernels.Above(b, q, p), zero, one);                   // all unless q ranks above p (Adad's order)
        V Below(V p, V q) => Above(q, p);
        var g0 = -dW + dPx * half - ix * Above(x1, gx1) - dCw * Below(x1, gx1);
        var g1 = -dH + dPy * half - iy * Above(y1, gy1) - dCh * Below(y1, gy1);
        var g2 = dW + dPx * half + ix * Below(x2, gx2) + dCw * Above(x2, gx2);
        var g3 = dH + dPy * half + iy * Below(y2, gy2) + dCh * Above(y2, gy2);
        var dl = Codec.UnpackAt(b, f, pd, pair, lane);
        return [g0 * dl, g1 * dl, g2 * dl, g3 * dl];
    }
}
