// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// number's product cells on Vulkan: GenericPtx's matmul, im2col, col2im, transpose, box loss and ctc, the same kernels
// written over Spv, one generator per operation specialized by T at run time (SpvType.Of<T>), every step in T's own
// arithmetic. Where CUDA's cell computes a step in f32 (GenericPtx.Gaps: f16's division, min and max; every type's atan
// by the polynomial; ctc's exp and log by the device's base-2 instructions), this one does the same, so number's error on
// Vulkan is its error on CUDA. ctc's gradient adds by the device's float atomic add where it has one for T, else by a
// compare-and-swap of the 32-bit word holding the value (f16: Vulkan has no f16 atomic add here).
using System.Numerics;
using Adad.Vulkan;

namespace Qiyas;

internal static partial class GenericSpv
{
    private static V Load(Spv b, SpvType t, V p, V index) => b.Convert(b.Load(t.Mem, b.At(p, index, t.Size)), t.Reg);

    private static void Store(Spv b, SpvType t, V p, V index, V value) => b.Store(b.At(p, index, t.Size), b.Convert(value, t.Mem));

    /// <summary>C = A·B (row-major, [m,k]·[k,n]), sums in T: workgroups of 16 × 16 invocations, each a micro × micro tile of C,
    /// the workgroup a (16·micro)² tile; A and B through workgroup memory 16 of k at a time (GenericPtx.MatMul). Arguments: a,
    /// b, c, m, n, k; workgroups (⌈n/tile⌉, ⌈m/tile⌉).</summary>
    public static SpvKernel MatMul<T>(Vulkan vk, int micro) where T : INumberBase<T>
    {
        var t = SpvType.Of<T>();
        int tile = 16 * micro, loads = tile * 16 / 256;
        var b = new Spv($"gemm{tile}_{t.Name}", 16, vk.Controls, 16);
        var pa = b.Arg(0);
        var pb = b.Arg(1);
        var pc = b.Arg(2);
        var m = b.Arg(3, K.U32);
        var n = b.Arg(4, K.U32);
        var k = b.Arg(5, K.U32);
        var tx = b.LocalIdX;
        var ty = b.LocalIdY;
        var lin = ty * 16 + tx;
        var col0 = b.GroupX * tile;
        var row0 = b.GroupY * tile;
        var sa = b.Shared(t.Reg, 16 * tile);                                   // A's tile k-major: sa[kc·tile + row]
        var sb = b.Shared(t.Reg, 16 * tile);                                   // B's tile: sb[kr·tile + col]
        var zero = b.C(t.Reg, 0);
        var acc = new Var[micro * micro];
        for (int i = 0; i < acc.Length; i++) acc[i] = b.Local(zero);
        b.For(b.U(0), k, b.U(16), t0 =>
        {
            for (int e = 0; e < loads; e++)
            {
                var idx = lin + e * 256;
                var r2 = idx >> 4;
                var r3 = idx & 15UL;
                var ga = row0 + r2;
                var gk = t0 + r3;
                var ok = (ga < m) & (gk < k);
                sa[r3 * tile + r2] = b.Select(ok, Load(b, t, pa, b.Select(ok, ga * k + gk, b.U(0))), zero);
                var kr = idx >> BitOperations.Log2((uint)tile);
                var cc = idx & (ulong)(tile - 1);
                var gkb = t0 + kr;
                var gc = col0 + cc;
                var okb = (gkb < k) & (gc < n);
                sb[kr * tile + cc] = b.Select(okb, Load(b, t, pb, b.Select(okb, gkb * n + gc, b.U(0))), zero);
            }

            b.Barrier();
            var rowBase = ty * micro;
            var colBase = tx * micro;
            for (int kk = 0; kk < 16; kk++)
            {
                var ra = new V[micro];
                var rb = new V[micro];
                for (int i = 0; i < micro; i++) ra[i] = sa[rowBase + (kk * tile + i)];
                for (int j = 0; j < micro; j++) rb[j] = sb[colBase + (kk * tile + j)];
                for (int i = 0; i < micro; i++) for (int j = 0; j < micro; j++) acc[i * micro + j].V = b.Fma(ra[i], rb[j], acc[i * micro + j].V);
            }

            b.Barrier();
        });

        for (int i = 0; i < micro; i++)
        {
            for (int j = 0; j < micro; j++)
            {
                var r = ty * micro + i + row0;
                var c = tx * micro + j + col0;
                var v = acc[i * micro + j].V;
                b.If((r < m) & (c < n), () => Store(b, t, pc, r * n + c, v));
            }
        }

        return b.Build();
    }

    // The convolution's parameters as the kernels take them, a slot each from slot 3: H W kh kw sh sw pt pl dh dw OH OW.
    private static V[] ConvParams(Spv b)
    {
        var p = new V[12];
        for (int i = 0; i < 12; i++) p[i] = b.Arg(3 + i, K.U32);
        return p;
    }

    /// <summary>The columns of one image: a thread per column value (row r = (c, a, b), column q = (i, j)), x's value at
    /// (c, i·sh − pt + a·dh, j·sw − pl + b·dw) or 0 off the image. Arguments: x, cols, count (rows·OH·OW), then the
    /// parameters (ConvParams).</summary>
    public static SpvKernel Im2col<T>(Vulkan vk) where T : INumberBase<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("im2col_" + t.Name, Threads, vk.Controls);
        var px = b.Arg(0);
        var pc = b.Arg(1);
        var count = b.Arg(2, K.U32);
        var p = ConvParams(b);
        V H = p[0], W = p[1], kh = p[2], kw = p[3], sh = p[4], sw = p[5], pt = p[6], pl = p[7], dh = p[8], dw = p[9], oh = p[10], ow = p[11];
        var gid = b.GlobalX;
        b.If(gid < count, () =>
        {
            var hw = oh * ow;
            var r = gid / hw;
            var q = gid - r * hw;
            var i = q / ow;
            var j = q - i * ow;
            var bb = r % kw;
            var r24 = r / kw;
            var a = r24 % kh;
            var c = r24 / kh;
            var ih = i * sh - pt + a * dh;                                        // above the image: past H unsigned
            var iw = j * sw - pl + bb * dw;
            var inside = (ih < H) & (iw < W);
            var v = b.Select(inside, Load(b, t, px, b.Select(inside, (c * H + ih) * W + iw, b.U(0))), b.C(t.Reg, 0));
            Store(b, t, pc, gid, v);
        });
        return b.Build();
    }

    /// <summary>The image's gradient from the columns' (col2im) as a gather: a thread per value of dx (c, ih, iw), the sum in
    /// T of every column value its position fed, taps in order. Arguments: dcols, dx, count (C·H·W), then the parameters.</summary>
    public static SpvKernel Col2im<T>(Vulkan vk) where T : INumberBase<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("col2im_" + t.Name, Threads, vk.Controls);
        var pc = b.Arg(0);
        var px = b.Arg(1);
        var count = b.Arg(2, K.U32);
        var p = ConvParams(b);
        V H = p[0], W = p[1], kh = p[2], kw = p[3], sh = p[4], sw = p[5], pt = p[6], pl = p[7], dh = p[8], dw = p[9], oh = p[10], ow = p[11];
        var gid = b.GlobalX;
        b.If(gid < count, () =>
        {
            var hw = H * W;
            var c = gid / hw;
            var rem = gid - c * hw;
            var ih = rem / W;
            var iw = rem - ih * W;
            var ohw = oh * ow;
            var sum = b.Local(b.C(t.Reg, 0));
            b.For(b.U(0), kh, a =>
            {
                var top = ih + pt;
                var off = a * dh;
                b.If(top >= off, () =>
                {
                    var y = top - off;
                    b.If(y % sh == 0, () =>
                    {
                        var i = y / sh;
                        b.If(i < oh, () => b.For(b.U(0), kw, bb =>
                        {
                            var left = iw + pl;
                            var offw = bb * dw;
                            b.If(left >= offw, () =>
                            {
                                var x = left - offw;
                                b.If(x % sw == 0, () =>
                                {
                                    var j = x / sw;
                                    b.If(j < ow, () =>
                                    {
                                        var row = (c * kh + a) * kw + bb;                 // the columns' row (c, a, b)
                                        var col = i * ow + j;                             // the column (i, j)
                                        sum.V = sum.V + Load(b, t, pc, row * ohw + col);
                                    });
                                });
                            });
                        }));
                    });
                });
            });
            Store(b, t, px, gid, sum.V);
        });
        return b.Build();
    }

    /// <summary>t = xᵀ (x rows × cols): a thread per value. Arguments: x, t, rows, cols.</summary>
    public static SpvKernel Transpose<T>(Vulkan vk) where T : INumberBase<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("transpose_" + t.Name, Threads, vk.Controls);
        var px = b.Arg(0);
        var pt = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        var gid = b.GlobalX;
        b.If(gid < rows * cols, () =>
        {
            var i = gid / cols;
            var j = gid - i * cols;
            Store(b, t, pt, j * rows + i, Load(b, t, px, gid));
        });
        return b.Build();
    }

    /// <summary>The box overlap loss of a pair (backward: its gradient times dloss), a thread per pair, in T (CpuGeneric.BoxLoss's
    /// formulas in their order, GenericPtx.BoxLoss): f16's division, min and max in f32, atan by GenericPtx's polynomial in
    /// f32. Arguments: predicted, target, dloss, y, n.</summary>
    public static SpvKernel BoxLoss<T>(Vulkan vk, BoxOverlap overlap, bool backward) where T : IFloatingPointIeee754<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv($"box_{overlap.ToString().ToLowerInvariant()}{(backward ? "_bwd" : "")}_{t.Name}", Threads, vk.Controls);
        var pp = b.Arg(0);
        var ptg = b.Arg(1);
        var pdl = b.Arg(2);
        var py = b.Arg(3);
        var count = b.Arg(4, K.U32);
        var e = new Scalar(b, t);
        double D(T value) => double.CreateTruncating(value);
        var gid = b.GlobalX;
        b.If(gid < count, () =>
        {
            V zero = e.Const(0), one = e.Const(1), two = e.Const(2), half = e.Const(0.5);
            V eps = e.Const(D(T.BitIncrement(T.One) - T.One)), aspect = e.Const(D(T.CreateTruncating(4) / (T.Pi * T.Pi)));
            var at = gid * 4;
            V x1 = Load(b, t, pp, at), y1 = Load(b, t, pp, at + 1), x2 = Load(b, t, pp, at + 2), y2 = Load(b, t, pp, at + 3);
            V gx1 = Load(b, t, ptg, at), gy1 = Load(b, t, ptg, at + 1), gx2 = Load(b, t, ptg, at + 2), gy2 = Load(b, t, ptg, at + 3);
            V iw = e.Sub(e.Min(x2, gx2), e.Max(x1, gx1)), ih = e.Sub(e.Min(y2, gy2), e.Max(y1, gy1));
            V overlaps = (iw > zero) & (ih > zero);
            V inter = b.Select(overlaps, e.Mul(iw, ih), zero);
            V wp = e.Sub(x2, x1), hp = e.Sub(y2, y1), wg = e.Sub(gx2, gx1), hg = e.Sub(gy2, gy1);
            V union = e.Sub(e.Add(e.Mul(wp, hp), e.Mul(wg, hg)), inter), ue = e.Add(union, eps), iou = e.Div(inter, ue);
            V cw = zero, ch = zero;
            if (overlap != BoxOverlap.IoU)
            {
                cw = e.Sub(e.Max(x2, gx2), e.Min(x1, gx1));
                ch = e.Sub(e.Max(y2, gy2), e.Min(y1, gy1));
            }

            if (!backward)
            {
                V loss = e.Sub(one, iou);
                if (overlap == BoxOverlap.GIoU)
                {
                    V area = e.Mul(cw, ch);
                    loss = e.Add(loss, e.Div(e.Sub(area, union), e.Add(area, eps)));
                }
                else if (overlap != BoxOverlap.IoU)
                {
                    V diag = e.Add(e.Add(e.Mul(cw, cw), e.Mul(ch, ch)), eps);
                    V dx = e.Mul(e.Add(e.Sub(x1, gx1), e.Sub(x2, gx2)), half), dy = e.Mul(e.Add(e.Sub(y1, gy1), e.Sub(y2, gy2)), half);
                    loss = e.Add(loss, e.Div(e.Add(e.Mul(dx, dx), e.Mul(dy, dy)), diag));
                    if (overlap == BoxOverlap.CIoU)
                    {
                        V diff = e.Sub(e.Atan(e.Div(wg, hg)), e.Atan(e.Div(wp, hp)));
                        V v = e.Mul(e.Mul(aspect, diff), diff), alpha = e.Div(v, e.Add(e.Add(e.Sub(one, iou), v), eps));
                        loss = e.Add(loss, e.Mul(alpha, v));
                    }
                }

                Store(b, t, py, gid, loss);
                return;
            }

            V dInter = e.Div(e.Const(-1), ue), dUnion = e.Div(iou, ue), dCw = zero, dCh = zero, dPx = zero, dPy = zero, dW = zero, dH = zero;
            if (overlap == BoxOverlap.GIoU)
            {
                V area = e.Mul(cw, ch), ae = e.Add(area, eps);
                dUnion = e.Sub(dUnion, e.Div(one, ae));
                V dArea = e.Div(ue, e.Mul(ae, ae));
                dCw = e.Mul(dArea, ch);
                dCh = e.Mul(dArea, cw);
            }
            else if (overlap != BoxOverlap.IoU)
            {
                V diag = e.Add(e.Add(e.Mul(cw, cw), e.Mul(ch, ch)), eps);
                V dx = e.Mul(e.Add(e.Sub(x1, gx1), e.Sub(x2, gx2)), half), dy = e.Mul(e.Add(e.Sub(y1, gy1), e.Sub(y2, gy2)), half);
                V rho = e.Add(e.Mul(dx, dx), e.Mul(dy, dy));
                dPx = e.Div(e.Mul(two, dx), diag);
                dPy = e.Div(e.Mul(two, dy), diag);
                V dDiag = -e.Div(rho, e.Mul(diag, diag));
                dCw = e.Mul(e.Mul(two, dDiag), cw);
                dCh = e.Mul(e.Mul(two, dDiag), ch);
                if (overlap == BoxOverlap.CIoU)
                {
                    V diff = e.Sub(e.Atan(e.Div(wg, hg)), e.Atan(e.Div(wp, hp)));
                    V v = e.Mul(e.Mul(aspect, diff), diff), alpha = e.Div(v, e.Add(e.Add(e.Sub(one, iou), v), eps));
                    V held = e.Sub(one, e.Mul(alpha, alpha));
                    dInter = e.Mul(dInter, held);
                    dUnion = e.Mul(dUnion, held);
                    V dv = e.Mul(alpha, e.Sub(two, alpha)), r2 = e.Add(e.Mul(wp, wp), e.Mul(hp, hp));
                    V kc = e.Div(e.Mul(e.Mul(e.Mul(two, aspect), dv), diff), r2);
                    dW = -e.Mul(kc, hp);
                    dH = e.Mul(kc, wp);
                }
            }

            dInter = e.Sub(dInter, dUnion);
            dW = e.Add(dW, e.Mul(dUnion, hp));
            dH = e.Add(dH, e.Mul(dUnion, wp));
            V ix = b.Select(overlaps, e.Mul(ih, dInter), zero), iy = b.Select(overlaps, e.Mul(iw, dInter), zero);
            V Above(V a, V c) => b.Select(a > c, one, b.Select(a == c, one, zero));      // a tie: the predicted corner takes it all (Boxes.Above)
            V Below(V a, V c) => b.Select(a < c, one, b.Select(a == c, one, zero));      // (Boxes.Below)
            V dl = Load(b, t, pdl, gid);
            Store(b, t, py, at, e.Mul(e.Sub(e.Sub(e.Add(-dW, e.Mul(dPx, half)), e.Mul(ix, Above(x1, gx1))), e.Mul(dCw, Below(x1, gx1))), dl));
            Store(b, t, py, at + 1, e.Mul(e.Sub(e.Sub(e.Add(-dH, e.Mul(dPy, half)), e.Mul(iy, Above(y1, gy1))), e.Mul(dCh, Below(y1, gy1))), dl));
            Store(b, t, py, at + 2, e.Mul(e.Add(e.Add(e.Add(dW, e.Mul(dPx, half)), e.Mul(ix, Below(x2, gx2))), e.Mul(dCw, Above(x2, gx2))), dl));
            Store(b, t, py, at + 3, e.Mul(e.Add(e.Add(e.Add(dH, e.Mul(dPy, half)), e.Mul(iy, Below(y2, gy2))), e.Mul(dCh, Above(y2, gy2))), dl));
        });
        return b.Build();
    }

    /// <summary>Straight-line arithmetic in T: what Vulkan's GenericPtx counterpart takes in T (add, sub, mul, negation,
    /// comparisons, selection), and in f32 what CUDA's cell takes there: f16's division, min and max, every type's atan by
    /// GenericPtx's polynomial (atan(x) = base + x + x·z·P(z), z = x², after the reduction |x| > tan(3π/8): base π/2,
    /// x → −1/x; |x| > tan(π/8): base π/4, x → (x − 1)/(x + 1)).</summary>
    private sealed class Scalar(Spv b, SpvType t)
    {
        private bool Half => t.Reg == K.F16;

        public V Const(double value) => b.C(t.Reg, value);

        public V Add(V x, V y) => x + y;

        public V Sub(V x, V y) => x - y;

        public V Mul(V x, V y) => x * y;

        public V Div(V x, V y) => Half ? b.Convert(b.Convert(x, K.F32) / b.Convert(y, K.F32), K.F16) : x / y;

        public V Min(V x, V y) => Half ? b.Convert(b.Min(b.Convert(x, K.F32), b.Convert(y, K.F32)), K.F16) : b.Min(x, y);

        public V Max(V x, V y) => Half ? b.Convert(b.Max(b.Convert(x, K.F32), b.Convert(y, K.F32)), K.F16) : b.Max(x, y);

        private V F(uint bits) => b.CBits(K.F32, bits);

        public V Atan(V a)
        {
            var x = b.Convert(a, K.F32);
            var ax = b.Abs(x);
            var big = ax > F(0x401A827A);                                      // tan(3π/8)
            var middle = ax > F(0x3ED413CD);                                   // tan(π/8)
            var one = F(0x3F800000);
            var inv = F(0xBF800000) / ax;
            var mid = (ax - one) / (ax + one);
            var xr = b.Select(middle, mid, ax);
            xr = b.Select(big, inv, xr);
            var bs = b.Select(middle, F(0x3F490FDB), F(0));                    // π/4
            bs = b.Select(big, F(0x3FC90FDB), bs);                             // π/2
            var zz = xr * xr;
            var p = F(0x3DA4F0D1);                                             // 8.05374449538e-2
            p = b.Fma(p, zz, F(0xBE0E1B85));                                   // − 1.38776856032e-1
            p = b.Fma(p, zz, F(0x3E4C925F));                                   // + 1.99777106478e-1
            p = b.Fma(p, zz, F(0xBEAAAA2A));                                   // − 3.33329491539e-1
            p = p * zz;
            p = b.Fma(p, xr, xr);
            var y = bs + p;
            y = b.Select(x < F(0), -y, y);
            return b.Convert(y, t.Reg);
        }
    }

    /// <summary>ctc (CpuGeneric.Ctc's definition and order; GenericPtx.Ctc), a workgroup of 256 per sequence b, the frame
    /// loops inside: each frame's log-sum-exp over its C logits (workgroup reductions in T) into lse; α frame by frame, an
    /// invocation per state, into alpha; the loss by invocation 0; with withgrad, β back from the last frame into beta, then
    /// per frame the row's exp(x − lse) (0 past the sequence's frames) less each state's exp(α + β − log y − log P), added
    /// atomically (states share a class). The scratch is the sequence's at b·T (lse) and b·T·states (α, β). exp and log in
    /// f32 as CUDA's (2^(x·log2 e), log2(x)·ln 2). Arguments: x, loss, grad, lse, alpha, beta, frames, lengths, offsets,
    /// labels, T, B, C, blank, states, withgrad.</summary>
    public static SpvKernel Ctc<T>(Vulkan vk) where T : IFloatingPointIeee754<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("ctc_" + t.Name, Threads, vk.Controls);
        var px = b.Arg(0);
        var ploss = b.Arg(1);
        var pgrad = b.Arg(2);
        var plse0 = b.Arg(3);
        var palpha0 = b.Arg(4);
        var pbeta0 = b.Arg(5);
        var pframes = b.Arg(6);
        var plengths = b.Arg(7);
        var poffsets = b.Arg(8);
        var plabels = b.Arg(9);
        var nT = b.Arg(10, K.U32);
        var nB = b.Arg(11, K.U32);
        var nC = b.Arg(12, K.U32);
        var blank = b.Arg(13, K.U32);
        var states = b.Arg(14, K.U32);
        var withgrad = b.Arg(15, K.U32);
        var red = b.Shared(t.Reg, Threads);
        var tx = b.LocalIdX;
        var seq = b.GroupX;
        var ninf = b.C(t.Reg, double.NegativeInfinity);
        var zero = b.C(t.Reg, 0);
        var frames = b.Load(K.U32, b.At(pframes, seq, 4));
        var S = b.Load(K.U32, b.At(plengths, seq, 4)) * 2 + 1;
        var off = b.Load(K.U32, b.At(poffsets, seq, 4));
        var plse = b.At(plse0, seq * nT, t.Size);
        var palpha = b.At(palpha0, seq * nT * states, t.Size);
        var pbeta = b.At(pbeta0, seq * nT * states, t.Size);

        V Exp(V v) => b.Convert(b.Exp2(b.Convert(v, K.F32) * b.CBits(K.F32, 0x3FB8AA3B)), t.Reg);
        V Log(V v) => b.Convert(b.Log2(b.Convert(v, K.F32)) * b.CBits(K.F32, 0x3F317218), t.Reg);

        // log(e^a0 + e^a1 + e^a2) by the largest (−∞ when all are).
        V Lse3(V a0, V a1, V a2)
        {
            var mx = b.Select(a2 > a1, a2, a1);
            mx = b.Select(mx > a0, mx, a0);
            var sum = Exp(a0 - mx);
            sum = sum + Exp(a1 - mx);
            sum = sum + Exp(a2 - mx);
            var res = mx + Log(sum);
            return b.Select(mx == ninf, mx, res);
        }

        // The class of state st (the blank at even states).
        V Class(V st)
        {
            var even = (st & 1UL) == 0;
            var label = b.Load(K.U32, b.At(plabels, b.Select(even, b.U(0), (st >> 1) + off), 4));
            return b.Select(even, blank, label);
        }

        // log y of frame f at class cls: x[(f·B + seq)·C + cls] − lse[f].
        V LogY(V f, V cls) => Load(b, t, px, (f * nB + seq) * nC + cls) - Load(b, t, plse, f);

        // State s may come from s − 2 (odd s ≥ 3 whose label differs from the label before it).
        V Skip(V s)
        {
            var q = ((s & 1UL) != 0) & (s >= 3);
            var at = b.Select(q, (s >> 1) + off, b.U(1));
            var l1 = b.Load(K.U32, b.At(plabels, at, 4));
            var l0 = b.Load(K.U32, b.At(plabels, at - 1, 4));
            return q & (l1 != l0);
        }

        V At(V p, V f, V s) => b.At(p, f * states + s, t.Size);

        // A workgroup reduction of acc in T (add, or max by a greater-than), the result for every invocation.
        V Reduce(V acc, bool max)
        {
            red[tx] = acc;
            b.MemoryBarrierAll();
            for (int s = Threads / 2; s > 0; s /= 2)
            {
                b.If(tx < s, () =>
                {
                    var w = red[tx + s];
                    var a = red[tx];
                    red[tx] = max ? b.Select(w > a, w, a) : a + w;
                });
                b.MemoryBarrierAll();
            }

            var all = red[0];
            b.MemoryBarrierAll();
            return all;
        }

        // Each frame's log-sum-exp.
        b.For(b.U(0), frames, ft =>
        {
            var row = (ft * nB + seq) * nC;
            var acc = b.Local(ninf);
            b.For(tx, nC, b.U(Threads), c =>
            {
                var x = Load(b, t, px, row + c);
                acc.V = b.Select(x > acc.V, x, acc.V);
            });
            var m = Reduce(acc.V, max: true);
            var sum = b.Local(zero);
            b.For(tx, nC, b.U(Threads), c => sum.V = sum.V + Exp(Load(b, t, px, row + c) - m));
            var total = Reduce(sum.V, max: false);
            var lse = m + Log(total);
            b.If(tx == 0, () => Store(b, t, plse, ft, lse));
        });

        // α at frame 0, then frame by frame.
        b.MemoryBarrierAll();
        b.For(tx, S, b.U(Threads), st =>
        {
            var ly = LogY(b.U(0), Class(st));
            b.Store(At(palpha, b.U(0), st), b.Convert(b.Select(st < 2, ly, ninf), t.Mem));
        });
        b.MemoryBarrierAll();
        b.For(b.U(1), frames, ft =>
        {
            var fp = ft - 1;
            b.For(tx, S, b.U(Threads), st =>
            {
                var a0 = Load(b, t, palpha, fp * states + st);
                var a1 = b.Select(st >= 1, Load(b, t, palpha, fp * states + b.Select(st >= 1, st - 1, st)), ninf);
                var skip = Skip(st);
                var a2 = b.Select(skip, Load(b, t, palpha, fp * states + b.Select(skip, st - 2, st)), ninf);
                var res = Lse3(a0, a1, a2) + LogY(ft, Class(st));
                b.Store(At(palpha, ft, st), b.Convert(res, t.Mem));
            });
            b.MemoryBarrierAll();
        });

        // The loss.
        var fl = frames - 1;
        var l0 = Load(b, t, palpha, fl * states + S - 1);
        var l1 = b.Select(S >= 2, Load(b, t, palpha, fl * states + b.Select(S >= 2, S - 2, S - 1)), ninf);
        var logp = Lse3(l0, l1, ninf);
        b.If(tx == 0, () => Store(b, t, ploss, seq, -logp));
        b.If(withgrad != 0, () => b.If(logp == ninf, () =>
        {
            // No alignment: the gradient 0 on every frame.
            b.For(b.U(0), nT, ft =>
            {
                var row = (ft * nB + seq) * nC;
                b.For(tx, nC, b.U(Threads), c => Store(b, t, pgrad, row + c, zero));
            });
        }, () =>
        {
            // β at the last frame, then back.
            b.For(tx, S, b.U(Threads), st =>
            {
                var ly = LogY(fl, Class(st));
                b.Store(At(pbeta, fl, st), b.Convert(b.Select(st + 2 >= S, ly, ninf), t.Mem));
            });
            b.MemoryBarrierAll();
            b.For(b.U(0), fl, u =>
            {
                var ft = fl - u;
                var fp = ft - 1;
                b.For(tx, S, b.U(Threads), st =>
                {
                    var a0 = Load(b, t, pbeta, ft * states + st);
                    var next = st + 1 < S;
                    var a1 = b.Select(next, Load(b, t, pbeta, ft * states + b.Select(next, st + 1, st)), ninf);
                    var st2 = st + 2;
                    var skip = Skip(st2) & (st2 < S);
                    var a2 = b.Select(skip, Load(b, t, pbeta, ft * states + b.Select(skip, st2, st)), ninf);
                    var res = Lse3(a0, a1, a2) + LogY(fp, Class(st));
                    b.Store(At(pbeta, fp, st), b.Convert(res, t.Mem));
                });
                b.MemoryBarrierAll();
            });

            // The gradient: per frame exp(x − lse) (0 past the frames), less each state's share.
            b.For(b.U(0), nT, ft =>
            {
                var inFrames = ft < frames;
                var row = (ft * nB + seq) * nC;
                var lse = b.Select(inFrames, Load(b, t, plse, b.Select(inFrames, ft, b.U(0))), zero);
                b.For(tx, nC, b.U(Threads), c =>
                {
                    var e = Exp(Load(b, t, px, row + c) - lse);
                    Store(b, t, pgrad, row + c, b.Select(inFrames, e, zero));
                });
                b.MemoryBarrierAll();
                b.If(inFrames, () => b.For(tx, S, b.U(Threads), st =>
                {
                    var a0 = Load(b, t, palpha, ft * states + st);
                    var a1 = Load(b, t, pbeta, ft * states + st);
                    var cls = Class(st);
                    var e = Exp(a0 + a1 - LogY(ft, cls) - logp);
                    AtomicAdd(b, vk, t, b.At(pgrad, row + cls, t.Size), -e);
                }));
            });
        }));
        return b.Build();
    }

    // value added to the T at address atomically: the device's float atomic add where it has one for T, else a
    // compare-and-swap of the 32-bit word holding it (an f16 in its half of the word), until no other add came between.
    private static void AtomicAdd(Spv b, Vulkan vk, SpvType t, V address, V value)
    {
        if ((t.Reg == K.F32 && vk.AtomicAdd32) || (t.Reg == K.F64 && vk.AtomicAdd64))
        {
            b.AtomicFAdd(address, value);
            return;
        }

        if (t.Size > 4) throw new NotSupportedException($"vulkan: number's ctc in {t.Name} needs the device's {t.Name} atomic add");
        var word = address & ~3UL;
        var shift = b.Convert((address & 3UL) * 8, K.U32);
        int bits = t.Size * 8;
        var seen = b.Local(b.Load(K.U32, word));
        var done = b.Local(b.Bool(false));
        b.While(() => !done.V, () =>
        {
            var old = seen.V;
            var mine = b.Extract(old, shift, b.U(bits));
            var sum = b.Bitcast(b.Convert(mine, t.Reg.BitsKind()), t.Reg) + value;
            var fresh = b.Insert(old, b.Convert(b.Bits(sum), K.U32), shift, b.U(bits));
            var got = b.AtomicCompareExchange(word, fresh, old);
            done.V = got == old;
            seen.V = got;
        });
    }
}
