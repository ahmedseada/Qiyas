// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on CUDA (GpuOps.Boxes.cs; Abstraction/Boxes.cs), in the float lane by Boxes.Reference's formulas
// and order (straight-line code: selects, no branches), the corners read where they are stored. Two kernels:
//   - the pairs kernel (formats of whole words of a power of two values, the arrays 16-byte aligned): a thread per pair
//     (per two of a 4-bit format's gradients, per a word's pairs of losses), its corners read as its whole words (one
//     vector load where they fill one), each pair computed once, its loss or the four components of its gradient times its
//     dloss packed into the thread's own output words, stored as one vector where they fill one;
//   - the value kernel (any format: a block format's, a warp or a CTA per block): a thread per stored word of the output,
//     each value its pair's loss or one component of its pair's gradient (the pair computed for each component).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Params: predicted, target, dloss (the backward's), y, n (the outputs: count, or 4·count backward), eps (a lane value).</summary>
    public static string BoxLoss(Gpu gpu, Fmt f, Lane lane, BoxOverlap overlap, bool backward, out string entry)
    {
        entry = $"box_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string pp = Param64(b, "p_p"), pt = Param64(b, "p_t"), pd = Param64(b, "p_d"), py = Param64(b, "p_y"), n = Param32(b, "p_n");
        string eps = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {eps}, [p_eps];");
        StoreEach(b, f, lane, py, n, at => BoxValue(b, f, lane, overlap, backward, pp, pt, pd, eps, at));
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_p, .param .u64 p_t, .param .u64 p_d, .param .u64 p_y, .param .u32 p_n, .param .{lane.Mem} p_eps", EachShared(f));
    }

    /// <summary>Formats the pairs kernel takes: whole words of a power of two values (not a block format).</summary>
    public static bool BoxPairWise(Fmt f) => !f.IsBlock && f.Per is 1 or 2 or 4 or 8;

    /// <summary>The pairs a thread of the pairs kernel takes: as many as fill whole words of its outputs and of its corners.</summary>
    public static int BoxThreadPairs(Fmt f, bool backward) => backward ? Math.Max(1, f.Per / 4) : f.Per;

    /// <summary>The pairs kernel (BoxPairWise formats; predicted, target and y 16-byte aligned). Params as BoxLoss's, n the
    /// pairs (4·n below 2^32).</summary>
    public static string BoxLossPairs(Gpu gpu, Fmt f, Lane lane, BoxOverlap overlap, bool backward, out string entry)
    {
        entry = $"box_pairs_{Boxes.Name(overlap)}{(backward ? "_bwd" : "")}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string pp = Param64(b, "p_p"), pt = Param64(b, "p_t"), pd = Param64(b, "p_d"), py = Param64(b, "p_y"), count = Param32(b, "p_n");
        string eps = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {eps}, [p_eps];");
        int pairs = BoxThreadPairs(f, backward), per = backward ? 4 : 1, inWords = pairs * 4 / f.Per, outWords = pairs * per / f.Per;
        string gid = GlobalIndex(b), first = b.R(), past = b.P(), corners = b.R(), last = b.R();
        b.L($"mul.lo.u32 {first}, {gid}, {pairs};");
        b.L($"setp.ge.u32 {past}, {first}, {count};");
        b.L($"@{past} bra DONE;");
        b.L($"shl.b32 {corners}, {count}, 2;");
        b.L($"sub.u32 {last}, {count}, 1;");
        string cornerWords = WordsOf(b, f, corners);
        var pw = ThreadWords(b, f, pp, gid, inWords, cornerWords);
        var tw = ThreadWords(b, f, pt, gid, inWords, cornerWords);
        var natural = f.Natural;
        var code = new string[outWords];
        for (int w = 0; w < outWords; w++)
        {
            code[w] = f.WordBits == 64 ? b.RD() : b.R();
            if (f.WordBits == 32) b.L($"mov.b32 {code[w]}, 0;");
        }

        for (int j = 0; j < pairs; j++)
        {
            string Corner(string[] words, int k) => Codec.Unpack(b, f, words[(j * 4 + k) / f.Per], (j * 4 + k) % f.Per, lane);
            var r = BoxPair(b, lane, overlap, backward, Corner(pw, 0), Corner(pw, 1), Corner(pw, 2), Corner(pw, 3), Corner(tw, 0), Corner(tw, 1), Corner(tw, 2), Corner(tw, 3), eps);
            // A pair past the count (the last thread of a format of two pairs a thread): its values 0, dropped.
            string at = first, inside = "";
            if (pairs > 1)
            {
                at = b.R();
                inside = b.P();
                b.L($"add.u32 {at}, {first}, {j};");
                b.L($"setp.lt.u32 {inside}, {at}, {count};");
                b.L($"min.u32 {at}, {at}, {last};");
            }

            string[] outs = [r.Loss];
            if (backward)
            {
                var e = new Emit(b, lane);
                string dl = Codec.UnpackAt(b, f, pd, at, lane);
                outs = [e.Mul(r.G0, dl), e.Mul(r.G1, dl), e.Mul(r.G2, dl), e.Mul(r.G3, dl)];
            }

            for (int k = 0; k < per; k++)
            {
                int v = j * per + k;
                string stored = lane == natural ? outs[k] : Codec.Convert(b, lane, outs[k], natural, f);
                if (pairs > 1)
                {
                    string kept = natural.NewReg(b);
                    b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {inside};");
                    stored = kept;
                }

                Codec.Pack(b, f, stored, natural, v % f.Per, code[v / f.Per]);
            }
        }

        // The thread's output words: one vector where they fill one (its words are whole: inside the output).
        string oa = b.RD();
        b.L($"mul.wide.u32 {oa}, {gid}, {outWords * f.WordBits / 8};");
        b.L($"add.u64 {oa}, {py}, {oa};");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        if (f.WordBits == 32 && outWords == 4) b.L($"st.global.v4.b32 [{oa}], {{{code[0]}, {code[1]}, {code[2]}, {code[3]}}};");
        else if (f.WordBits == 32 && outWords == 2) b.L($"st.global.v2.b32 [{oa}], {{{code[0]}, {code[1]}}};");
        else if (f.WordBits == 64 && outWords == 4)
        {
            b.L($"st.global.v2.b64 [{oa}], {{{code[0]}, {code[1]}}};");
            b.L($"st.global.v2.b64 [{oa}+16], {{{code[2]}, {code[3]}}};");
        }
        else for (int w = 0; w < outWords; w++) b.L($"st.global.{wt} [{oa}+{w * f.WordBits / 8}], {code[w]};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_p, .param .u64 p_t, .param .u64 p_d, .param .u64 p_y, .param .u32 p_n, .param .{lane.Mem} p_eps");
    }

    // The thread's k words (from k·gid) of an array of `total` words: one vector load where all k are inside (16 bytes; 8
    // for two 32-bit words), else each word inside alone; 0 past the array.
    private static string[] ThreadWords(PtxBuilder b, Fmt f, string ptr, string gid, int k, string total)
    {
        bool wide = f.WordBits == 64;
        string wt = wide ? "b64" : "b32";
        var w = new string[k];
        for (int i = 0; i < k; i++)
        {
            w[i] = wide ? b.RD() : b.R();
            b.L($"mov.{wt} {w[i]}, 0;");
        }

        string w0 = b.R(), end = b.R(), all = b.P(), addr = b.RD();
        b.L($"mul.lo.u32 {w0}, {gid}, {k};");
        b.L($"add.u32 {end}, {w0}, {k};");
        b.L($"setp.le.u32 {all}, {end}, {total};");
        b.L($"mul.wide.u32 {addr}, {w0}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {ptr}, {addr};");
        if (!wide && k == 4) b.L($"@{all} ld.global.v4.b32 {{{w[0]}, {w[1]}, {w[2]}, {w[3]}}}, [{addr}];");
        else if (!wide && k == 2) b.L($"@{all} ld.global.v2.b32 {{{w[0]}, {w[1]}}}, [{addr}];");
        else if (wide && k == 4)
        {
            b.L($"@{all} ld.global.v2.b64 {{{w[0]}, {w[1]}}}, [{addr}];");
            b.L($"@{all} ld.global.v2.b64 {{{w[2]}, {w[3]}}}, [{addr}+16];");
        }
        else if (wide && k == 2) b.L($"@{all} ld.global.v2.b64 {{{w[0]}, {w[1]}}}, [{addr}];");
        else if (k == 1) b.L($"@{all} ld.global.{wt} {w[0]}, [{addr}];");
        else throw new NotSupportedException($"box losses: {k} words a thread");
        if (k > 1)
        {
            // The last thread's words, each one inside loaded alone.
            string some = b.P();
            b.L($"not.pred {some}, {all};");
            for (int i = 0; i < k; i++)
            {
                string at = b.R(), inside = b.P();
                b.L($"add.u32 {at}, {w0}, {i};");
                b.L($"setp.lt.and.u32 {inside}, {at}, {total}, {some};");
                b.L($"@{inside} ld.global.{wt} {w[i]}, [{addr}+{i * f.WordBits / 8}];");
            }
        }

        return w;
    }

    // Output at's value (the value kernel): the loss of pair at, or component at % 4 of pair at / 4's gradient times the
    // pair's dloss.
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
        var r = BoxPair(b, lane, overlap, backward, x1, y1, x2, y2, gx1, gy1, gx2, gy2, eps);
        if (!backward) return r.Loss;
        string Is(int k)
        {
            string p = b.P();
            b.L($"setp.eq.u32 {p}, {comp}, {k};");
            return p;
        }

        string g = e.Select(Is(0), r.G0, e.Select(Is(1), r.G1, e.Select(Is(2), r.G2, r.G3)));
        return e.Mul(g, Codec.UnpackAt(b, f, pd, pair, lane));
    }

    // A pair's loss and, backward, its gradient's four components (before dloss).
    private static (string Loss, string G0, string G1, string G2, string G3) BoxPair(PtxBuilder b, Lane lane, BoxOverlap overlap, bool backward,
        string x1, string y1, string x2, string y2, string gx1, string gy1, string gx2, string gy2, string eps)
    {
        var e = new Emit(b, lane);
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

        if (!backward) return (loss, "", "", "", "");
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
        return (loss, g0, g1, g2, g3);
    }
}
