// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of the box overlap losses over pairs of corner boxes (x1, y1, x2, y2): the kinds and the float64
// reference with its gradient. The devices implement it in their folders: Cpu/Boxes.cs; Cuda/Ptx.Boxes.cs with
// Cuda/GpuOps.Boxes.cs; Vulkan/Kernels.Boxes.cs with Vulkan/VulkanOps.Boxes.cs. Each computes a pair in the float lane in
// the reference's order: every width, and the centres' offset as (x1 − x1') + (x2 − x2'), a difference of two stored
// corners (within a rounding of its own value, not of the corners'); the partials chained as here.
namespace Adad.Abstraction;

/// <summary>How a pair's overlap becomes its loss, IoU = |p ∩ t| / (|p ∪ t| + eps): IoU's is 1 − IoU; GIoU adds the share of
/// the enclosing box C outside the union, (|C| − |p ∪ t|) / (|C| + eps); DIoU the centres' squared distance over C's squared
/// diagonal + eps; CIoU DIoU's and α·v, v = (4/π²)·(atan(w_t/h_t) − atan(w_p/h_p))² the aspects' disagreement and
/// α = v / (1 − IoU + v + eps). eps, added to each divisor that can be 0, is the stored type's precision (IFloatAdad.Epsilon:
/// float32's 2^−23, float16's 2^−10, …), so it moves no quotient whose divisor is 1 or more by more than the type's step;
/// the gradient is the loss's derivative, α's included.</summary>
public enum BoxOverlap { IoU, GIoU, DIoU, CIoU }

public static class Boxes
{
    /// <summary>4/π², CIoU's aspect-ratio factor.</summary>
    public const double AspectFactor = 0.40528473456935108578;

    public static string Name(BoxOverlap overlap) => overlap switch
    {
        BoxOverlap.IoU => "iou", BoxOverlap.GIoU => "giou", BoxOverlap.DIoU => "diou", BoxOverlap.CIoU => "ciou",
        _ => throw new ArgumentOutOfRangeException(nameof(overlap), overlap, "not a box overlap"),
    };

    /// <summary>A pair's loss in float64 from its predicted corners p and target t; with grad (4 values) the loss's derivative
    /// with respect to p there (where the two sides of a minimum or a maximum tie, each takes half: of the derivatives the
    /// sides and their mixtures give, the least), with gradScale the magnitude each component's terms add to (the scale the
    /// lane's rounding acts on). Returns the loss and its terms' magnitude.</summary>
    public static (double Loss, double Scale) Reference(BoxOverlap overlap, ReadOnlySpan<double> p, ReadOnlySpan<double> t, double eps, Span<double> grad = default, Span<double> gradScale = default)
    {
        _ = Name(overlap);
        double x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3], gx1 = t[0], gy1 = t[1], gx2 = t[2], gy2 = t[3];
        double iw = Math.Min(x2, gx2) - Math.Max(x1, gx1), ih = Math.Min(y2, gy2) - Math.Max(y1, gy1);
        bool overlaps = iw > 0 && ih > 0;
        double inter = overlaps ? iw * ih : 0;
        double wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
        double union = wp * hp + wg * hg - inter, ue = union + eps, iou = inter / ue;
        double loss = 1 - iou, scale = 1 + iou;
        double cw = Math.Max(x2, gx2) - Math.Min(x1, gx1), ch = Math.Max(y2, gy2) - Math.Min(y1, gy1);

        // ∂loss / ∂(each part), and their terms' magnitudes (…S).
        double dInter = -1 / ue, dUnion = iou / ue, dUnionS = Math.Abs(dUnion);
        double dCw = 0, dCh = 0, dPx = 0, dPy = 0, dW = 0, dH = 0, dCwS = 0, dChS = 0, dPxS = 0, dPyS = 0, dWS = 0, dHS = 0, heldS = 1;
        if (overlap == BoxOverlap.GIoU)
        {
            double area = cw * ch, ae = area + eps;
            loss += (area - union) / ae;
            scale += (area + union) / ae;
            dUnion -= 1 / ae;
            dUnionS += 1 / ae;
            double dArea = ue / (ae * ae);
            (dCw, dCh) = (dArea * ch, dArea * cw);
            (dCwS, dChS) = (Math.Abs(dCw), Math.Abs(dCh));
        }
        else if (overlap is BoxOverlap.DIoU or BoxOverlap.CIoU)
        {
            double diag = cw * cw + ch * ch + eps;
            double dx = ((x1 - gx1) + (x2 - gx2)) * 0.5, dy = ((y1 - gy1) + (y2 - gy2)) * 0.5;
            double sx = (Math.Abs(x1 - gx1) + Math.Abs(x2 - gx2)) * 0.5, sy = (Math.Abs(y1 - gy1) + Math.Abs(y2 - gy2)) * 0.5;
            double rho = dx * dx + dy * dy, rhoS = sx * sx + sy * sy;
            loss += rho / diag;
            scale += rhoS / diag;
            (dPx, dPy) = (2 * dx / diag, 2 * dy / diag);
            (dPxS, dPyS) = (2 * sx / diag, 2 * sy / diag);
            double dDiag = -(rho / (diag * diag)), dDiagS = rhoS / (diag * diag);
            (dCw, dCh) = (2 * dDiag * cw, 2 * dDiag * ch);
            (dCwS, dChS) = (2 * dDiagS * Math.Abs(cw), 2 * dDiagS * Math.Abs(ch));
            if (overlap == BoxOverlap.CIoU)
            {
                // α·v = v² / D, D = 1 − IoU + v + eps: ∂(α·v)/∂IoU = v²/D² = α², ∂(α·v)/∂v = 2v/D − v²/D² = α·(2 − α).
                double ag = Math.Atan(wg / hg), ap = Math.Atan(wp / hp), diff = ag - ap, spread = Math.Abs(ag) + Math.Abs(ap);
                double v = AspectFactor * diff * diff, alpha = v / (1 - iou + v + eps);
                loss += alpha * v;
                scale += alpha * (v + 2 * AspectFactor * Math.Abs(diff) * spread + 1);
                double held = 1 - alpha * alpha;
                heldS = 1 + alpha * alpha;
                (dInter, dUnion, dUnionS) = (dInter * held, dUnion * held, dUnionS * heldS);
                double dv = alpha * (2 - alpha), dvS = alpha * (2 + alpha);
                double r2 = wp * wp + hp * hp, k = 2 * AspectFactor * dv * diff / r2, kS = 2 * AspectFactor * dvS * (Math.Abs(diff) + spread) / r2;
                (dW, dH) = (-(k * hp), k * wp);
                (dWS, dHS) = (kS * Math.Abs(hp), kS * Math.Abs(wp));
            }
        }

        if (grad.IsEmpty) return (loss, scale);

        // The union holds the predicted area and minus the intersection.
        dInter -= dUnion;
        double dInterS = heldS / ue + dUnionS;
        dW += dUnion * hp;
        dH += dUnion * wp;
        dWS += dUnionS * Math.Abs(hp);
        dHS += dUnionS * Math.Abs(wp);
        double ix = overlaps ? ih * dInter : 0, iy = overlaps ? iw * dInter : 0;
        double ixS = overlaps ? Math.Abs(ih) * dInterS : 0, iyS = overlaps ? Math.Abs(iw) * dInterS : 0;
        // x1: the width, the centre, the intersection's left (a maximum), the enclosing box's left (a minimum); y1 the same; x2, y2
        // the other way.
        grad[0] = -dW + dPx * 0.5 - ix * Above(x1, gx1) - dCw * Below(x1, gx1);
        grad[1] = -dH + dPy * 0.5 - iy * Above(y1, gy1) - dCh * Below(y1, gy1);
        grad[2] = dW + dPx * 0.5 + ix * Below(x2, gx2) + dCw * Above(x2, gx2);
        grad[3] = dH + dPy * 0.5 + iy * Below(y2, gy2) + dCh * Above(y2, gy2);
        if (!gradScale.IsEmpty)
        {
            gradScale[0] = dWS + dPxS * 0.5 + ixS * Above(x1, gx1) + dCwS * Below(x1, gx1);
            gradScale[1] = dHS + dPyS * 0.5 + iyS * Above(y1, gy1) + dChS * Below(y1, gy1);
            gradScale[2] = dWS + dPxS * 0.5 + ixS * Below(x2, gx2) + dCwS * Above(x2, gx2);
            gradScale[3] = dHS + dPyS * 0.5 + iyS * Below(y2, gy2) + dChS * Above(y2, gy2);
        }

        return (loss, scale);
    }

    /// <summary>The share of max(a, b)'s gradient a takes (min's: Below): all of it, none, or at a tie half.</summary>
    public static double Above(double a, double b) => a > b ? 1 : a == b ? 0.5 : 0;

    public static double Below(double a, double b) => a < b ? 1 : a == b ? 0.5 : 0;
}
