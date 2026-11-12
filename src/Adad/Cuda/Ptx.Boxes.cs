// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on CUDA (GpuOps.Boxes.cs; Abstraction/Boxes.cs): a thread per stored word of the output (a warp, or
// a CTA, per block of a block format), each value its pair's loss, or a component of its pair's gradient times the pair's
// dloss, computed from the corners read where they are stored, in the float lane by Boxes.Reference's formulas and order
// (straight-line code: selects, no branches).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Params: predicted, target, dloss (the backward's), y, n (the outputs: count, or 4·count backward), eps (a lane value).</summary>
    public static string BoxLoss(Gpu gpu, Fmt f, Lane lane, BoxOverlap overlap, bool backward, out string entry)
    {
        entry = $"box_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string pp = Param64(b, "p_p"), pt = Param64(b, "p_t"), pd = Param64(b, "p_d"), py = Param64(b, "p_y"), n = Param32(b, "p_n");
        string eps = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {eps}, [p_eps];");
        StoreEach(b, f, lane, py, n, at => BoxValue(b, f, lane, overlap, backward, pp, pt, pd, eps, at));
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_p, .param .u64 p_t, .param .u64 p_d, .param .u64 p_y, .param .u32 p_n, .param .{lane.Mem} p_eps", EachShared(f));
    }

    // Output at's value: the loss of pair at, or component at % 4 of pair at / 4's gradient times the pair's dloss.
    private static string BoxValue(PtxBuilder b, Fmt f, Lane lane, BoxOverlap overlap, bool backward, string pp, string pt, string pd, string eps, string at)
    {
        var e = new Emit(b, lane);
        string pair = at, comp = "";
        if (backward)
        {
            pair = b.R();
            comp = b.R();
            b.L($"shr.u32 {pair}, {at}, 2;");
            b.L($"and.b32 {comp}, {at}, 3;");
        }

        string Corner(string ptr, int k)
        {
            string i = b.R();
            b.L($"mad.lo.u32 {i}, {pair}, 4, {k};");
            return Codec.UnpackAt(b, f, ptr, i, lane);
        }

        string x1 = Corner(pp, 0), y1 = Corner(pp, 1), x2 = Corner(pp, 2), y2 = Corner(pp, 3);
        string gx1 = Corner(pt, 0), gy1 = Corner(pt, 1), gx2 = Corner(pt, 2), gy2 = Corner(pt, 3);
        string zero = e.C(0), one = e.C(1), half = e.C(0.5), two = e.C(2);
        string iw = e.Sub(e.Min(x2, gx2), e.Max(x1, gx1)), ih = e.Sub(e.Min(y2, gy2), e.Max(y1, gy1));
        string overlaps = e.And(e.Greater(iw, zero), e.Greater(ih, zero));
        string inter = e.Select(overlaps, e.Mul(iw, ih), zero);
        string wp = e.Sub(x2, x1), hp = e.Sub(y2, y1), wg = e.Sub(gx2, gx1), hg = e.Sub(gy2, gy1);
        string union = e.Sub(e.Add(e.Mul(wp, hp), e.Mul(wg, hg)), inter), ue = e.Add(union, eps), iou = e.Div(inter, ue);
        string loss = e.Sub(one, iou);
        string cw = e.Sub(e.Max(x2, gx2), e.Min(x1, gx1)), ch = e.Sub(e.Max(y2, gy2), e.Min(y1, gy1));
        string dInter = e.Neg(e.Div(one, ue)), dUnion = e.Div(iou, ue);
        string dCw = zero, dCh = zero, dPx = zero, dPy = zero, dW = zero, dH = zero;
        if (overlap == BoxOverlap.GIoU)
        {
            string area = e.Mul(cw, ch), ae = e.Add(area, eps);
            loss = e.Add(loss, e.Div(e.Sub(area, union), ae));
            dUnion = e.Sub(dUnion, e.Div(one, ae));
            string dArea = e.Div(ue, e.Mul(ae, ae));
            (dCw, dCh) = (e.Mul(dArea, ch), e.Mul(dArea, cw));
        }
        else if (overlap is BoxOverlap.DIoU or BoxOverlap.CIoU)
        {
            string diag = e.Add(e.Add(e.Mul(cw, cw), e.Mul(ch, ch)), eps);
            string dx = e.Mul(e.Add(e.Sub(x1, gx1), e.Sub(x2, gx2)), half), dy = e.Mul(e.Add(e.Sub(y1, gy1), e.Sub(y2, gy2)), half);
            string rho = e.Add(e.Mul(dx, dx), e.Mul(dy, dy));
            loss = e.Add(loss, e.Div(rho, diag));
            (dPx, dPy) = (e.Div(e.Mul(two, dx), diag), e.Div(e.Mul(two, dy), diag));
            string dDiag = e.Neg(e.Div(rho, e.Mul(diag, diag)));
            (dCw, dCh) = (e.Mul(e.Mul(two, dDiag), cw), e.Mul(e.Mul(two, dDiag), ch));
            if (overlap == BoxOverlap.CIoU)
            {
                string diff = e.Sub(e.Atan(e.Div(wg, hg)), e.Atan(e.Div(wp, hp))), af = e.C(Boxes.AspectFactor);
                string v = e.Mul(e.Mul(af, diff), diff);
                string alpha = e.Div(v, e.Add(e.Add(e.Sub(one, iou), v), eps));
                loss = e.Add(loss, e.Mul(alpha, v));
                string held = e.Sub(one, e.Mul(alpha, alpha));             // ∂(α·v)/∂IoU = α², ∂(α·v)/∂v = α·(2 − α)
                (dInter, dUnion) = (e.Mul(dInter, held), e.Mul(dUnion, held));
                string k = e.Div(e.Mul(e.Mul(e.Mul(two, af), e.Mul(alpha, e.Sub(two, alpha))), diff), e.Add(e.Mul(wp, wp), e.Mul(hp, hp)));
                (dW, dH) = (e.Neg(e.Mul(k, hp)), e.Mul(k, wp));
            }
        }

        if (!backward) return loss;
        dInter = e.Sub(dInter, dUnion);
        dW = e.Add(dW, e.Mul(dUnion, hp));
        dH = e.Add(dH, e.Mul(dUnion, wp));
        string ix = e.Select(overlaps, e.Mul(ih, dInter), zero), iy = e.Select(overlaps, e.Mul(iw, dInter), zero);
        string Above(string p, string q) => e.Select(e.Above(q, p), zero, one);              // all unless q ranks above p (Adad's order)
        string Below(string p, string q) => Above(q, p);
        string g0 = e.Sub(e.Sub(e.Add(e.Neg(dW), e.Mul(dPx, half)), e.Mul(ix, Above(x1, gx1))), e.Mul(dCw, Below(x1, gx1)));
        string g1 = e.Sub(e.Sub(e.Add(e.Neg(dH), e.Mul(dPy, half)), e.Mul(iy, Above(y1, gy1))), e.Mul(dCh, Below(y1, gy1)));
        string g2 = e.Add(e.Add(e.Add(dW, e.Mul(dPx, half)), e.Mul(ix, Below(x2, gx2))), e.Mul(dCw, Above(x2, gx2)));
        string g3 = e.Add(e.Add(e.Add(dH, e.Mul(dPy, half)), e.Mul(iy, Below(y2, gy2))), e.Mul(dCh, Above(y2, gy2)));
        string Is(int k)
        {
            string p = b.P();
            b.L($"setp.eq.u32 {p}, {comp}, {k};");
            return p;
        }

        string g = e.Select(Is(0), g0, e.Select(Is(1), g1, e.Select(Is(2), g2, g3)));
        return e.Mul(g, Codec.UnpackAt(b, f, pd, pair, lane));
    }
}
