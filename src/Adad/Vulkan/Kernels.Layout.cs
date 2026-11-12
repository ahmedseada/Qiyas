// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The layout operations on Vulkan (Layout.cs; CUDA's Ptx.Layout.cs, whose rules LayoutValue repeats): a thread per
// output word computes each of its values from the output index (a gather; the sums — scatter-add, col2im, the pools'
// backwards, the depthwise convolution — in the sum lane, converted on the store), so every word is written by one
// invocation and no format needs atomics. Inputs are read where they are stored (Codec.UnpackAt).
// Arguments: x, x2, idx, aux, out (slots 0–4), count and cols (5, cols in the high half), then the operation's sixteen
// integers (LayoutArgs.KernelP) two to a slot, low half first (6–13): the slots a block encoder's source takes.
// Scatter-add's ordering is built first: the sources counted per target (integer atomics: exact), the offsets by a
// scan, the sources' stable order by target by the radix sort (Kernels.Radix.cs).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>The layout arguments in their slots (see the file's note).</summary>
    public static ulong[] LayoutSlots(ulong x, ulong x2, ulong idx, ulong aux, ulong output, int count, int cols, int[] p)
    {
        if (p.Length > LayoutParamCount) throw new ArgumentException($"layout: {p.Length} parameters (the kernels take {LayoutParamCount})");
        var args = new ulong[6 + LayoutParamCount / 2];
        args[0] = x; args[1] = x2; args[2] = idx; args[3] = aux; args[4] = output;
        args[5] = (uint)count | (ulong)(uint)cols << 32;
        for (int j = 0; j < LayoutParamCount / 2; j++)
        {
            uint lo = 2 * j < p.Length ? (uint)p[2 * j] : 0, hi = 2 * j + 1 < p.Length ? (uint)p[2 * j + 1] : 0;
            args[6 + j] = lo | (ulong)hi << 32;
        }

        return args;
    }

    /// <summary>The integer parameters a layout kernel takes (LayoutArgs.KernelParams).</summary>
    public const int LayoutParamCount = LayoutArgs.KernelParams;

    private readonly struct LayoutIn
    {
        public readonly V X, X2, Idx, Aux, Out, Count, Cols;
        public readonly V[] P;

        public LayoutIn(Spv b)
        {
            X = b.Arg(0); X2 = b.Arg(1); Idx = b.Arg(2); Aux = b.Arg(3); Out = b.Arg(4);
            var sizes = b.Arg(5);
            Count = b.Convert(sizes, K.U32); Cols = b.Convert(sizes >> 32, K.U32);
            P = new V[LayoutParamCount];
            for (int i = 0; i < LayoutParamCount; i++)
            {
                var raw = b.Arg(6 + i / 2);
                P[i] = b.Convert(i % 2 == 0 ? raw : raw >> 32, K.U32);
            }
        }
    }

    /// <summary>A layout operation's output as f, a thread per stored word (values past the count stored as zero); inFmt: the
    /// inputs' format where it is not the output's (lane storage, an intermediate's).</summary>
    public static SpvKernel Layout(Vulkan vk, Fmt f, Lane lane, string name, Fmt? inFmt = null)
    {
        var b = new Spv($"layout_{name}_{f.Name}_{lane.Name}{(inFmt is null ? "" : "_i" + inFmt.Name)}", Threads, vk.Controls);
        var a = new LayoutIn(b);
        var gid = b.GlobalX;
        var natural = f.Natural;
        var nk = Kinds.Of(natural);
        b.If(gid < WordCount(b, f, a.Count), () =>
        {
            var word = b.CBits(Codec.WordKind(f), 0);
            var last = a.Count - 1;
            for (int i = 0; i < f.Per; i++)
            {
                // A word's values past the count take the last value's addresses (dropped).
                var at = gid * (double)f.Per + i;
                var v = b.Min(at, last);
                var r = v / a.Cols;
                var c = v - r * a.Cols;
                var value = LayoutValue(b, inFmt ?? f, lane, name, a, r, c, v);
                var stored = lane.Name == natural.Name ? value : Codec.Convert(b, lane, value, natural);
                word = Codec.Pack(b, f, b.Select(at < a.Count, stored, b.C(nk, 0)), natural, i, word);
            }

            b.Store(b.At(a.Out, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>The max-pool backward's first-ranked of every window of the forward input (x2) in Adad's order, a thread per window (count
    /// windows, in the output's order): its index into the input, 0xFFFFFFFF for a window with no pixel, into out (u32);
    /// compared in the backward's lane, as a scan of the window would.</summary>
    public static SpvKernel LayoutWindowArgmax(Vulkan vk, Fmt f, Lane lane, string name)
    {
        if (name is not ("maxpool_bwd" or "adaptive_maxpool_bwd")) throw new NotSupportedException($"window first-ranked of {name}");
        bool adaptive = name == "adaptive_maxpool_bwd";
        var k = Kinds.Of(lane);
        var b = new Spv($"layout_argmax_{name}_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var a = new LayoutIn(b);
        var p = a.P;
        var i = b.GlobalX;
        var none = b.U(0xFFFFFFFF);
        b.If(i < a.Count, () =>
        {
            V H = p[1], W = p[2];
            var OH = adaptive ? p[3] : PoolSize(p[1], p[3], p[5], p[7], null, p[11]);
            var OW = adaptive ? p[4] : PoolSize(p[2], p[4], p[6], p[8], null, p[11]);
            var ow = i % OW;
            var t = i / OW;
            var oh = t % OH;
            var ch = t / OH;
            V h0, kh, w0, kw;
            if (adaptive)
            {
                (h0, kh) = AdaptiveSpan(oh, H, OH);
                (w0, kw) = AdaptiveSpan(ow, W, OW);
            }
            else
            {
                // From o·s − pad over k pixels; a pixel outside the input (above or left: unsigned past H or W) is skipped.
                h0 = oh * p[5] - p[7];
                kh = p[3];
                w0 = ow * p[6] - p[8];
                kw = p[4];
            }

            var best = b.Local(Identity(b, k, "max"));
            var bestAt = b.Local(none);
            b.For(b.U(0), kh, aa =>
            {
                var hh = h0 + aa;
                b.If(hh < H, () => b.For(b.U(0), kw, bb =>
                {
                    var ww = w0 + bb;
                    b.If(ww < W, () =>
                    {
                        // The first-ranked in row-major order (Adad's order, Rank): a later value takes it only when it
                        // ranks strictly above.
                        var cand = (ch * H + hh) * W + ww;
                        var xv = Codec.UnpackAt(b, f, a.X2, cand, lane);
                        var take = Rank(b, lane, xv, best.V).Above | (bestAt.V == none);
                        best.V = b.Select(take, xv, best.V);
                        bestAt.V = b.Select(take, cand, bestAt.V);
                    });
                }));
            });
            b.Store(b.At(a.Out, i, 4), bestAt.V);
        });
        return b.Build();
    }

    /// <summary>im2col's value at column row r = (ch·kh + a)·kw + bb, column c = oh·OW + ow (p: its parameters,
    /// ConvArgs.P): the image's index (ch, oh·sh − ph + a·dh, ow·sw − pw + bb·dw), and whether it is inside the image (else
    /// the value is 0, the index 0: a safe address). The layout's im2col and the convolution's product share it.</summary>
    internal static (V Src, V Inside) Im2colAt(Spv b, V[] p, V r, V c)
    {
        V kw = p[4], kh = p[3], H = p[1], W = p[2];
        // The layout's parameters give the right pad (LayoutArgs.KernelP); the product's (ConvArgs.P's twelve) are symmetric.
        var OW = p.Length > 13 ? ConvSize(p[2], p[4], p[6], p[8], p[13], p[10]) : PoolSize(p[2], p[4], p[6], p[8], p[10]);
        var ow = c % OW;
        var oh = c / OW;
        var bb = r % kw;
        var t = r / kw;
        var ka = t % kh;
        var ch = t / kh;
        var ih = oh * p[5] - p[7] + ka * p[9];                                          // above or left: unsigned past H or W
        var iw = ow * p[6] - p[8] + bb * p[10];
        var inside = (ih < H) & (iw < W);
        return (b.Select(inside, (ch * H + ih) * W + iw, b.U(0)), inside);
    }

    // [⌊o·n/outs⌋, ⌈(o + 1)·n/outs⌉): the start and the length.
    private static (V Start, V Length) AdaptiveSpan(V o, V n, V outs)
    {
        var start = o * n / outs;
        var end = ((o + 1) * n + outs - 1) / outs;
        return (start, end - start);
    }

    // A pool's (dilation 1) or a convolution's output size (LayoutArgs.PoolSize): (size + 2·pad − (k − 1)·d − 1) / s + 1, or
    // with ceil (≠ 0) rounded up and the last window dropped when it starts past the input and its leading pad.
    private static V PoolSize(V size, V k, V s, V pad, V? d, V? ceil = null)
    {
        var u = k - 1;
        if (d is V dd) u = u * dd;
        var span = (pad << 1) + size - u - 1;
        var floor = span / s + 1;
        if (ceil is not V cm) return floor;
        var b = size.B;
        var up = (span + s - 1) / s;
        up = b.Select(up * s >= size + pad, up, up + 1);
        return b.Select(cm != 0, up, floor);
    }

    // A convolution's output size with its leading and trailing pads (LayoutArgs.ConvOut): (pad + end + size − (k − 1)·d − 1) / s + 1.
    private static V ConvSize(V size, V k, V s, V pad, V end, V d) => (pad + end + size - (k - 1) * d - 1) / s + 1;

    // LayoutArgs.PoolDivisor: window (oh, ow)'s divisor, its `inside` count or, counting the pads (p[12] ≠ 0), its positions up
    // to H + ph, W + pw (signed: a window may start in the leading pad).
    private static V PoolDivisor(Spv b, V[] p, V oh, V ow, V inside)
    {
        V Extent(V o, V size, V k, V s, V pad)
        {
            var hs = b.Bitcast(o * s - pad, K.S32);
            var he = b.Min(hs + b.Bitcast(k, K.S32), b.Bitcast(size + pad, K.S32));
            return b.Bitcast(he - hs, K.U32);
        }

        var d = Extent(oh, p[1], p[3], p[5], p[7]) * Extent(ow, p[2], p[4], p[6], p[8]);
        return b.Select(p[12] != 0, d, inside);
    }

    // An interpolation axis as integers (LayoutArgs.InterpolateAxis): t = max(o·α + β, 0) (β may be negative), i0 = t / δ;
    // past the input's last position both rows the last and the weight 0; f = (t − i0·δ) / δ in the lane, g = 1 − f.
    private static (V I0, V I1, V F, V G) BilinearAxis(Spv b, K k, V o, V inp, V alpha, V beta, V delta)
    {
        var t = b.Bitcast(b.Max(b.Bitcast(o * alpha + beta, K.S32), b.S(0)), K.U32);
        var i0 = t / delta;
        var rem = t - i0 * delta;
        var last = inp - 1;
        var past = i0 >= last;
        i0 = b.Min(i0, last);
        var i1 = b.Min(i0 + 1, last);
        var f = b.Convert(b.Select(past, b.U(0), rem), k) / b.Convert(delta, k);
        return (i0, i1, f, b.C(k, 1) - f);
    }

    // A nearest axis: min((o·α + β) / δ, in − 1).
    private static V NearestAxis(V o, V inp, V alpha, V beta, V delta) => o.B.Min((o * alpha + beta) / delta, inp - 1);

    // The outputs [lo, end) whose source rows include input row i (LayoutArgs.InterpolateSources, its hi + 1).
    private static (V Lo, V End) SourceRange(Spv b, bool bilinear, V i, V inp, V outp, V alpha, V beta, V delta)
    {
        var none = alpha == 0;                                                          // every output reads every row
        var a1 = b.Select(none, b.U(1), alpha);
        // ⌈n / α⌉ of a signed n, 0 where n ≤ 0 (the range starts at 0, or is empty).
        V CeilPos(V n) => b.Select(b.Bitcast(n, K.S32) > b.S(0), (n + a1 - 1) / a1, b.U(0));
        var lo = bilinear ? b.Select(i <= 1, b.U(0), CeilPos((i - 1) * delta - beta)) : CeilPos(i * delta - beta);
        var end = b.Min(b.Select(i == inp - 1, outp, CeilPos((i + 1) * delta - beta)), outp);
        return (b.Select(none, b.U(0), lo), b.Select(none, outp, end));
    }

    // The weight output o of an axis gives input position i: nearest 1 where it reads i; bilinear g where its first row is
    // i, else f where its second is. Returns whether it reads i at all.
    private static (V Has, V W) AxisWeight(Spv b, K k, bool bilinear, V o, V i, V inp, V alpha, V beta, V delta)
    {
        if (!bilinear) return (NearestAxis(o, inp, alpha, beta, delta) == i, b.C(k, 1));
        var (i0, i1, f, g) = BilinearAxis(b, k, o, inp, alpha, beta, delta);
        var first = i0 == i;
        return (first | (i1 == i), b.Select(first, g, f));
    }

    // The value of output v (row r, column c) as a lane value.
    private static V LayoutValue(Spv b, Fmt f, Lane lane, string name, LayoutIn a, V r, V c, V v)
    {
        var k = Kinds.Of(lane);
        var p = a.P;
        var cols = a.Cols;
        V Load(V index) => Codec.UnpackAt(b, f, a.X, index, lane);
        V Load2(V index) => Codec.UnpackAt(b, f, a.X2, index, lane);
        V Index(V ptr, V i) => b.Bitcast(b.Load(K.S32, b.At(ptr, i, 4)), K.U32);
        V Zero() => b.C(k, 0);
        if (name.StartsWith("strided", StringComparison.Ordinal))
        {
            // A canonical view of rank n (strided1 … strided7: LayoutArgs.StridedKernel): row r over the outer dimensions, column
            // c along the last; source = base + Σ i_d · s_d (strides wrap as u32: the source itself is inside the input).
            int n = int.Parse(name.AsSpan(7));
            var src = c * p[2 * n - 1] + p[2 * n];
            var t = r;
            for (int d = n - 2; d >= 1; d--)
            {
                var i = t % p[d];
                t = t / p[d];
                src = i * p[n + d] + src;
            }

            if (n >= 2) src = t * p[n] + src;
            return Load(src);
        }

        switch (name)
        {
            case "one_hot":
                return b.Select(Index(a.Idx, r) == c, b.C(k, 1), Zero());

            case "gather_rows":
                return Load(Index(a.Idx, r) * cols + c);

            case "gather_cols":
                return Load(r * p[1] + Index(a.Idx, c));

            case "scatter_add_rows":
            {
                // aux: offsets[targets + 1] then the sources in order; output (r, c) = Σ x[source·cols + c] over the row's sources.
                var acc = b.Local(Zero());
                var from = Index(a.Aux, r);
                var to = Index(a.Aux, r + 1);
                var first = p[2] + 1;
                b.For(from, to, q =>
                {
                    var s = Index(a.Aux, first + q);
                    acc.V = acc.V + Load(s * cols + c);
                });
                return acc.V;
            }

            case "im2col":
            {
                var (src, inside) = Im2colAt(b, p, r, c);
                return b.Select(inside, Load(src), Zero());
            }

            case "col2im":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih: Σ over (a, bb) with (ih + ph − a·dh) = oh·sh, (iw + pw − bb·dw) = ow·sw in range.
                V H = p[1], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                var OW = ConvSize(p[2], p[4], p[6], p[8], p[13], p[10]);
                var OH = ConvSize(p[1], p[3], p[5], p[7], p[12], p[9]);
                var ih = r % H;
                var ch = r / H;
                var plane = OH * OW;
                var acc = b.Local(Zero());
                b.For(b.U(0), kh, ka =>
                {
                    var th = ih + ph - ka * dh;                                         // may go negative: a signed test
                    var okH = (b.Bitcast(th, K.S32) >= b.S(0)) & (th % sh == 0) & (th / sh < OH);
                    b.If(okH, () => b.For(b.U(0), kw, bb =>
                    {
                        var tw = c + pw - bb * dw;
                        var okW = (b.Bitcast(tw, K.S32) >= b.S(0)) & (tw % sw == 0) & (tw / sw < OW);
                        b.If(okW, () =>
                        {
                            // column row index = (ch·kh + a)·kw + bb; column = oh·OW + ow
                            var rowIdx = (ch * kh + ka) * kw + bb;
                            acc.V = acc.V + Load(rowIdx * plane + th / sh * OW + tw / sw);
                        });
                    }));
                });
                return acc.V;
            }

            case "interpolate_nearest":
            {
                // r = ch·OH + oh, c = ow; p: C H W OH OW αh βh δh αw βw δw: ih = min((oh·αh + βh) / δh, H − 1), iw likewise.
                V H = p[1], W = p[2], OH = p[3];
                var oh = r % OH;
                var ch = r / OH;
                var ih = NearestAxis(oh, H, p[5], p[6], p[7]);
                var iw = NearestAxis(c, W, p[8], p[9], p[10]);
                return Load((ch * H + ih) * W + iw);
            }

            case "interpolate_nearest_bwd":
            case "interpolate_bilinear_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW): the outputs whose source rows and columns include
                // the position (oh, then ow ascending; LayoutArgs.InterpolateSources), each dy by its weight: acc = fma(dy, wy·wx, acc).
                bool bilinear = name == "interpolate_bilinear_bwd";
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var ih = r % H;
                var ch = r / H;
                var (h0, h1) = SourceRange(b, bilinear, ih, H, OH, p[5], p[6], p[7]);
                var (w0, w1) = SourceRange(b, bilinear, c, W, OW, p[8], p[9], p[10]);
                var acc = b.Local(Zero());
                b.For(h0, h1, oh =>
                {
                    var (hasY, wy) = AxisWeight(b, k, bilinear, oh, ih, H, p[5], p[6], p[7]);
                    b.If(hasY, () => b.For(w0, w1, ow =>
                    {
                        var (hasX, wx) = AxisWeight(b, k, bilinear, ow, c, W, p[8], p[9], p[10]);
                        b.If(hasX, () => acc.V = b.Fma(Load((ch * OH + oh) * OW + ow), wy * wx, acc.V));
                    }));
                });
                return acc.V;
            }

            case "interpolate_bilinear":
            {
                // p: C H W OH OW αh βh δh αw βw δw: each axis's two rows and weight exactly (BilinearAxis).
                V H = p[1], W = p[2], OH = p[3];
                var oh = r % OH;
                var ch = r / OH;
                var (y0, y1, fy, gy) = BilinearAxis(b, k, oh, H, p[5], p[6], p[7]);
                var (x0, x1, fx, gx) = BilinearAxis(b, k, c, W, p[8], p[9], p[10]);
                V Pixel(V yy, V xx) => Load((ch * H + yy) * W + xx);
                var top = b.Fma(Pixel(y0, x1), fx, Pixel(y0, x0) * gx);
                var bottom = b.Fma(Pixel(y1, x1), fx, Pixel(y1, x0) * gx);
                return b.Fma(bottom, fy, top * gy);
            }

            case "shift":
            {
                var s = c - p[2];
                var inside = s < cols;                                                    // unsigned: negative values are out of range
                var src = b.Select(inside, r * cols + s, b.U(0));
                return b.Select(inside, Load(src), Zero());
            }

            case "pad2d":
            {
                var inside = c < p[1];
                var src = b.Select(inside, r * p[1] + c, b.U(0));
                return b.Select(inside, Load(src), Zero());
            }

            case "maxpool":
            case "avgpool":
            case "depthwise_conv":
            {
                // r = ch·OH + oh, c = ow; the window over (a, bb) with the pads and dilations (1 for the pools).
                bool max = name == "maxpool", conv = name == "depthwise_conv";
                V H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                V dh = conv ? p[9] : b.U(1), dw = conv ? p[10] : b.U(1);
                var OH = PoolSize(p[1], p[3], p[5], p[7], conv ? p[9] : null, p[11]);
                var oh = r % OH;
                var ch = r / OH;
                var acc = b.Local(max ? Identity(b, k, "max") : Zero());
                var n = b.Local(b.U(0));
                b.For(b.U(0), kh, ka =>
                {
                    var ih = oh * sh - ph + ka * dh;
                    b.If(ih < H, () => b.For(b.U(0), kw, bb =>                             // unsigned: negative rows are out of range
                    {
                        var iw = c * sw - pw + bb * dw;
                        b.If(iw < W, () =>
                        {
                            var xv = Load((ch * H + ih) * W + iw);
                            if (max) acc.V = First(b, acc.V, xv);
                            else if (conv) acc.V = b.Fma(Load2((ch * kh + ka) * kw + bb), xv, acc.V);
                            else acc.V = acc.V + xv;
                            n.V = n.V + 1;
                        });
                    }));
                });
                if (conv) return acc.V;
                var empty = n.V == 0;
                if (max) return b.Select(empty, Zero(), acc.V);
                return b.Select(empty, Zero(), acc.V / b.Convert(PoolDivisor(b, p, oh, c, n.V), k));
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW), aux = each window's first-ranked (maxpool).
                bool max = name == "maxpool_bwd";
                V H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                var OH = PoolSize(p[1], p[3], p[5], p[7], null, p[11]);
                var OW = PoolSize(p[2], p[4], p[6], p[8], null, p[11]);
                var ih = r % H;
                var ch = r / H;
                var me = (ch * H + ih) * W + c;
                var acc = b.Local(Zero());
                b.For(b.U(0), kh, ka =>
                {
                    var th = ih + ph - ka;
                    var okH = (b.Bitcast(th, K.S32) >= b.S(0)) & (th % sh == 0) & (th / sh < OH);
                    b.If(okH, () => b.For(b.U(0), kw, bb =>
                    {
                        var tw = c + pw - bb;
                        var okW = (b.Bitcast(tw, K.S32) >= b.S(0)) & (tw % sw == 0) & (tw / sw < OW);
                        b.If(okW, () =>
                        {
                            var oh = th / sh;
                            var ow = tw / sw;
                            var dyIdx = (ch * OH + oh) * OW + ow;
                            var dy = Load(dyIdx);
                            if (max)
                            {
                                // The window's first-ranked, found once for all its pixels (LayoutWindowArgmax, in aux).
                                acc.V = b.Select(Index(a.Aux, dyIdx) == me, acc.V + dy, acc.V);
                                return;
                            }

                            // The window (oh, ow): its valid count.
                            var cnt = b.Local(b.U(0));
                            b.For(b.U(0), kh, aa =>
                            {
                                var hh = oh * sh - ph + aa;
                                b.If(hh < H, () => b.For(b.U(0), kw, bbb =>
                                {
                                    var ww = ow * sw - pw + bbb;
                                    b.If(ww < W, () => cnt.V = cnt.V + 1);
                                }));
                            });
                            acc.V = acc.V + dy / b.Convert(PoolDivisor(b, p, oh, ow, cnt.V), k);
                        });
                    }));
                });
                return acc.V;
            }

            case "adaptive_maxpool_bwd":
            case "adaptive_avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih: over the output rows and columns whose window holds the pixel.
                bool max = name == "adaptive_maxpool_bwd";
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var ih = r % H;
                var ch = r / H;
                var me = (ch * H + ih) * W + c;
                var acc = b.Local(Zero());
                b.For(b.U(0), OH, oh =>
                {
                    var (h0, hl) = AdaptiveSpan(oh, H, OH);
                    b.If((ih >= h0) & (ih < h0 + hl), () => b.For(b.U(0), OW, ow =>
                    {
                        var (w0, wl) = AdaptiveSpan(ow, W, OW);
                        b.If((c >= w0) & (c < w0 + wl), () =>
                        {
                            var dyIdx = (ch * OH + oh) * OW + ow;
                            var dy = Load(dyIdx);
                            if (max) acc.V = b.Select(Index(a.Aux, dyIdx) == me, acc.V + dy, acc.V);
                            else acc.V = acc.V + dy / b.Convert(hl * wl, k);
                        });
                    }));
                });
                return acc.V;
            }

            case "adaptive_maxpool":
            case "adaptive_avgpool":
            {
                bool max = name == "adaptive_maxpool";
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var oh = r % OH;
                var ch = r / OH;
                var (h0, hl) = AdaptiveSpan(oh, H, OH);
                var (w0, wl) = AdaptiveSpan(c, W, OW);
                var acc = b.Local(max ? Identity(b, k, "max") : Zero());
                b.For(h0, h0 + hl, ih => b.For(w0, w0 + wl, iw =>
                {
                    var xv = Load((ch * H + ih) * W + iw);
                    acc.V = max ? First(b, acc.V, xv) : acc.V + xv;
                }));
                if (max) return acc.V;
                return acc.V / b.Convert(hl * wl, k);
            }

            default: throw new NotSupportedException($"vulkan: layout {name} has no Vulkan kernels yet");
        }
    }

    // ---------------------------------------------------------------- scatter-add's ordering

    /// <summary>counts[index[i]] += 1 over n sources (integer atomics: exact; an index outside [0, targets) counts nowhere).
    /// Arguments: idx, counts, n, targets.</summary>
    public static SpvKernel ScatterCount(Vulkan vk)
    {
        var b = new Spv("scatter_count", Threads, vk.Controls);
        var idx = b.Arg(0);
        var counts = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var targets = b.Arg(3, K.U32);
        var i = b.GlobalX;
        b.If(i < n, () =>
        {
            var target = b.Bitcast(b.Load(K.S32, b.At(idx, i, 4)), K.U32);
            b.If(target < targets, () => b.AtomicAdd(b.At(counts, target, 4), b.U(1)));
        });
        return b.Build();
    }

    /// <summary>offsets[0] = 0, offsets[r + 1] = offsets[r] + counts[r], and offsets[targets + 1 ..] untouched: one invocation,
    /// a serial scan. Arguments: counts, offsets, targets.</summary>
    public static SpvKernel ScatterScan(Vulkan vk)
    {
        var b = new Spv("scatter_scan", 1, vk.Controls);
        var counts = b.Arg(0);
        var offsets = b.Arg(1);
        var targets = b.Arg(2, K.U32);
        var acc = b.Local(b.U(0));
        b.Store(b.At(offsets, b.U(0), 4), b.U(0));
        b.For(b.U(0), targets, r =>
        {
            acc.V = acc.V + b.Load(K.U32, b.At(counts, r, 4));
            b.Store(b.At(offsets, r + 1, 4), acc.V);
        });
        return b.Build();
    }

}
