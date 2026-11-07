// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The layout operations on Vulkan (Layout.cs; CUDA's Ptx.Layout.cs, whose rules LayoutValue repeats): a thread per
// output word computes each of its values from the output index (a gather; the sums — scatter-add, col2im, the pools'
// backwards, the depthwise convolution — in the sum lane, converted on the store), so every word is written by one
// invocation and no format needs atomics. Inputs are read where they are stored (Codec.UnpackAt).
// Arguments: x, x2, idx, aux, out (slots 0–4), count (5), rows (6), cols (7), then the operation's twelve integers
// (LayoutArgs order) two to a slot, low half first (8–13).
// Scatter-add's ordering is built first: the sources counted per target (integer atomics: exact), the offsets by a
// scan, each source placed by its rank among the earlier sources of its target (deterministic: the sources in order).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>The layout arguments in their slots (see the file's note).</summary>
    public static ulong[] LayoutSlots(ulong x, ulong x2, ulong idx, ulong aux, ulong output, int count, int rows, int cols, int[] p)
    {
        var args = new ulong[14];
        args[0] = x; args[1] = x2; args[2] = idx; args[3] = aux; args[4] = output;
        args[5] = (uint)count; args[6] = (uint)rows; args[7] = (uint)cols;
        for (int j = 0; j < 6; j++)
        {
            uint lo = 2 * j < p.Length ? (uint)p[2 * j] : 0, hi = 2 * j + 1 < p.Length ? (uint)p[2 * j + 1] : 0;
            args[8 + j] = lo | (ulong)hi << 32;
        }

        return args;
    }

    private readonly struct LayoutIn
    {
        public readonly V X, X2, Idx, Aux, Out, Count, Cols;
        public readonly V[] P;

        public LayoutIn(Spv b)
        {
            X = b.Arg(0); X2 = b.Arg(1); Idx = b.Arg(2); Aux = b.Arg(3); Out = b.Arg(4);
            Count = b.Arg(5, K.U32); Cols = b.Arg(7, K.U32);
            P = new V[12];
            for (int i = 0; i < 12; i++)
            {
                var raw = b.Arg(8 + i / 2);
                P[i] = b.Convert(i % 2 == 0 ? raw : raw >> 32, K.U32);
            }
        }
    }

    /// <summary>A layout operation's output as f, a thread per stored word (values past the count stored as zero).</summary>
    public static SpvKernel Layout(Vulkan vk, Fmt f, Lane lane, string name)
    {
        var b = new Spv($"layout_{name}_{f.Name}_{lane.Name}", Threads, vk.Controls);
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
                var value = LayoutValue(b, f, lane, name, a, r, c, v);
                var stored = lane.Name == natural.Name ? value : Codec.Convert(b, lane, value, natural);
                word = Codec.Pack(b, f, b.Select(at < a.Count, stored, b.C(nk, 0)), natural, i, word);
            }

            b.Store(b.At(a.Out, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>The max-pool backward's first largest of every window of the forward input (x2), a thread per window (count
    /// windows, in the output's order): its index into the input, 0xFFFFFFFF for a window with no pixel, into out (u32);
    /// compared in the backward's lane, as a scan of the window would.</summary>
    public static SpvKernel LayoutWindowArgmax(Vulkan vk, Fmt f, Lane lane, string name)
    {
        if (name is not ("maxpool_bwd" or "adaptive_maxpool_bwd")) throw new NotSupportedException($"window first largest of {name}");
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
            var OH = adaptive ? p[3] : PoolSize(p[1], p[3], p[5], p[7], null);
            var OW = adaptive ? p[4] : PoolSize(p[2], p[4], p[6], p[8], null);
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
                        // The first largest in row-major order: a later value takes it only when strictly larger.
                        var cand = (ch * H + hh) * W + ww;
                        var xv = Codec.UnpackAt(b, f, a.X2, cand, lane);
                        var take = (xv > best.V) | (bestAt.V == none);
                        best.V = b.Select(take, xv, best.V);
                        bestAt.V = b.Select(take, cand, bestAt.V);
                    });
                }));
            });
            b.Store(b.At(a.Out, i, 4), bestAt.V);
        });
        return b.Build();
    }

    // [⌊o·n/outs⌋, ⌈(o + 1)·n/outs⌉): the start and the length.
    private static (V Start, V Length) AdaptiveSpan(V o, V n, V outs)
    {
        var start = o * n / outs;
        var end = ((o + 1) * n + outs - 1) / outs;
        return (start, end - start);
    }

    // A pool's (dilation 1) or a depthwise convolution's output size: (size + 2·pad − (k − 1)·d − 1) / s + 1.
    private static V PoolSize(V size, V k, V s, V pad, V? d)
    {
        var u = k - 1;
        if (d is V dd) u = u * dd;
        return ((pad << 1) + size - u - 1) / s + 1;
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
        switch (name)
        {
            case "strided":
            {
                // v = ((i0·d1 + i1)·d2 + i2)·d3 + i3; source = base + Σ i_d · s_d.
                var i3 = v % p[3];
                var t = v / p[3];
                var i2 = t % p[2];
                t = t / p[2];
                var i1 = t % p[1];
                var i0 = t / p[1];
                var src = i0 * p[4] + p[8];
                src = i1 * p[5] + src;
                src = i2 * p[6] + src;
                src = i3 * p[7] + src;
                return Load(src);
            }

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
                // r = (ch·kh + a)·kw + bb; c = oh·OW + ow; source (ch, oh·sh − ph + a·dh, ow·sw − pw + bb·dw) or 0.
                V kw = p[4], kh = p[3], H = p[1], W = p[2];
                var OW = PoolSize(p[2], p[4], p[6], p[8], p[10]);
                var ow = c % OW;
                var oh = c / OW;
                var bb = r % kw;
                var t = r / kw;
                var ka = t % kh;
                var ch = t / kh;
                var ih = oh * p[5] - p[7] + ka * p[9];
                var iw = ow * p[6] - p[8] + bb * p[10];
                var inside = (ih < H) & (iw < W);
                var src = b.Select(inside, (ch * H + ih) * W + iw, b.U(0));             // a safe address when out of range (dropped)
                return b.Select(inside, Load(src), Zero());
            }

            case "col2im":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih: Σ over (a, bb) with (ih + ph − a·dh) = oh·sh, (iw + pw − bb·dw) = ow·sw in range.
                V H = p[1], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                var OW = PoolSize(p[2], p[4], p[6], p[8], p[10]);
                var OH = PoolSize(p[1], p[3], p[5], p[7], p[9]);
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
                // r = ch·OH + oh, c = ow: ih = min(H − 1, ((2·oh + 1)·H) / (2·OH)), iw likewise.
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var oh = r % OH;
                var ch = r / OH;
                var ih = b.Min(((oh << 1) + 1) * H / (OH << 1), H - 1);
                var iw = b.Min(((c << 1) + 1) * W / (OW << 1), W - 1);
                return Load((ch * H + ih) * W + iw);
            }

            case "interpolate_bilinear":
            {
                // sy = max((oh + ½)·H/OH − ½, 0) in the lane; y0 = min(floor sy, H − 1), y1 = min(y0 + 1, H − 1); x likewise.
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var oh = r % OH;
                var ch = r / OH;
                var (y0, y1, fy) = Axis(b, k, oh, H, OH);
                var (x0, x1, fx) = Axis(b, k, c, W, OW);
                var gy = b.C(k, 1) - fy;
                var gx = b.C(k, 1) - fx;
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
                var OH = PoolSize(p[1], p[3], p[5], p[7], conv ? p[9] : null);
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
                            if (max) acc.V = b.Max(acc.V, xv);
                            else if (conv) acc.V = b.Fma(Load2((ch * kh + ka) * kw + bb), xv, acc.V);
                            else acc.V = acc.V + xv;
                            n.V = n.V + 1;
                        });
                    }));
                });
                if (conv) return acc.V;
                var empty = n.V == 0;
                if (max) return b.Select(empty, Zero(), acc.V);
                return b.Select(empty, Zero(), acc.V / b.Convert(n.V, k));
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW), aux = each window's first largest (maxpool).
                bool max = name == "maxpool_bwd";
                V H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                var OH = PoolSize(p[1], p[3], p[5], p[7], null);
                var OW = PoolSize(p[2], p[4], p[6], p[8], null);
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
                                // The window's first largest, found once for all its pixels (LayoutWindowArgmax, in aux).
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
                            acc.V = acc.V + dy / b.Convert(cnt.V, k);
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
                    acc.V = max ? b.Max(acc.V, xv) : acc.V + xv;
                }));
                if (max) return acc.V;
                return acc.V / b.Convert(hl * wl, k);
            }

            default: throw new NotSupportedException($"vulkan: layout {name} has no Vulkan kernels yet");
        }
    }

    // One axis of a bilinear resize: (i0, i1, fraction) for output index o over `inp` input and `outp` output pixels.
    private static (V I0, V I1, V F) Axis(Spv b, K k, V o, V inp, V outp)
    {
        var s = (b.Convert(o, k) + 0.5) * b.Convert(inp, k);
        s = s / b.Convert(outp, k);
        s = b.Max(s - 0.5, b.C(k, 0));
        var last = inp - 1;
        var i0 = b.Min(b.Convert(b.Floor(s), K.U32), last);
        var i1 = b.Min(i0 + 1, last);
        return (i0, i1, s - b.Convert(i0, k));
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

    /// <summary>sources[offsets[index[i]] + rank] = i, rank = the number of earlier sources with the same index (a scan over
    /// the earlier indices: deterministic, quadratic in the source count). Arguments: idx, offsets, sources, n, targets.</summary>
    public static SpvKernel ScatterPlace(Vulkan vk)
    {
        var b = new Spv("scatter_place", Threads, vk.Controls);
        var idx = b.Arg(0);
        var offsets = b.Arg(1);
        var sources = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var targets = b.Arg(4, K.U32);
        var i = b.GlobalX;
        b.If(i < n, () =>
        {
            var mine = b.Load(K.S32, b.At(idx, i, 4));
            b.If(b.Bitcast(mine, K.U32) < targets, () =>
            {
                var rank = b.Local(b.U(0));
                b.For(b.U(0), i, j => rank.V = b.Select(b.Load(K.S32, b.At(idx, j, 4)) == mine, rank.V + 1, rank.V));
                var slot = b.Load(K.U32, b.At(offsets, mine, 4)) + rank.V;
                b.Store(b.At(sources, slot, 4), i);
            });
        });
        return b.Build();
    }
}
