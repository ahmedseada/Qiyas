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
            var OH = adaptive ? p[3] : p[11];                                                // a pool's OH OW: LayoutArgs.KernelP
            var OW = adaptive ? p[4] : p[12];
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

    /// <summary>im2col's value at column row r = (ch·kh + a)·kw + bb, column c = oh·OW + ow (p: im2col's parameters as the
    /// kernels take them, LayoutArgs.KernelP: C H W kh kw sh sw top left bottom right dh dw OH OW): the image's index
    /// (ch, oh·sh − top + a·dh, ow·sw − left + bb·dw), and whether it is inside the image (else the value is 0, the index 0:
    /// a safe address). The layout's im2col and the convolution's product share it.</summary>
    internal static (V Src, V Inside) Im2colAt(Spv b, V[] p, V r, V c)
    {
        V kw = p[4], kh = p[3], H = p[1], W = p[2];
        var OW = p[14];
        var ow = c % OW;
        var oh = c / OW;
        var bb = r % kw;
        var t = r / kw;
        var ka = t % kh;
        var ch = t / kh;
        var ih = oh * p[5] - p[7] + ka * p[11];                                         // above or left: unsigned past H or W
        var iw = ow * p[6] - p[8] + bb * p[12];
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

    // The outputs [first, end) along an axis whose adaptive windows (AdaptiveSpan) can hold input position i: window o
    // starts at or before i when o·n < (i + 1)·outs, and ends after it when (o + 1)·n > i·outs, so o runs from ⌊i·outs/n⌋
    // to ⌊((i + 1)·outs − 1)/n⌋ (held to the outputs).
    private static (V First, V End) AdaptiveRange(V i, V n, V outs)
    {
        var b = i.B;
        var first = b.Min(i * outs / n, outs);
        var end = b.Min(((i + 1) * outs - 1) / n + 1, outs);
        return (first, end);
    }

    // The input positions window o covers along an axis of n (LayoutArgs.Window): min(o·s − before + k, n) − max(o·s − before,
    // 0), signed (a window may start in the leading pad).
    private static V WindowSpan(Spv b, V o, V s, V before, V k, V n)
    {
        var from = b.Bitcast(o * s - before, K.S32);
        var to = b.Min(from + b.Bitcast(k, K.S32), b.Bitcast(n, K.S32));
        return b.Bitcast(to - b.Max(from, b.S(0)), K.U32);
    }

    // A bilinear axis as integers (LayoutArgs.BilinearAt): t = max(o·α + β, 0) (β may be negative), i0 = t / δ;
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

    /// <summary>The bilinear interpolations' axes (BilinearAxis of every output row oh &lt; OH, then of every output column),
    /// once a call, so the kernels read each position's two rows and weight instead of making them for every value with a
    /// division of integers and one of the lane: at aux, i0 (u32) for the OH + OW positions, then i1 (u32), then f (lane). The
    /// same operations on the same integers: the same weights. The layout's slots (p: C H W OH OW αh βh δh αw βw δw).</summary>
    public static SpvKernel BilinearTables(Vulkan vk, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"bilinear_tables_{lane.Name}", Threads, vk.Controls);
        var a = new LayoutIn(b);
        var p = a.P;
        var o = b.GlobalX;
        V H = p[1], W = p[2], OH = p[3], OW = p[4];
        var n = OH + OW;
        b.If(o < n, () =>
        {
            var rows = o < OH;
            var at = b.Select(rows, o, o - OH);
            var (i0, i1, f, _) = BilinearAxis(b, k, at, b.Select(rows, H, W), b.Select(rows, p[5], p[8]), b.Select(rows, p[6], p[9]), b.Select(rows, p[7], p[10]));
            b.Store(b.At(a.Aux, o, 4), i0);
            b.Store(b.At(a.Aux, n + o, 4), i1);
            b.Store(b.At(a.Aux + b.Convert(n, K.U64) * 8.0, o, lane.Size), f);
        });
        return b.Build();
    }

    /// <summary>The dimension a canonical strided view (P: d0 … d(n−1) | s0 … s(n−1) | base) reads with stride 1, other than its
    /// last, where its last's stride is not 1: the view is then a transposition of that dimension with the last (batched
    /// over the others), which StridedTiled copies by tiles; −1 where none (the last reads in order, or no dimension does).
    /// A packed type's tiles store whole words: its last dimension's values whole words.</summary>
    public static int StridedTileDim(Fmt f, int[] p)
    {
        int n = p.Length / 2;
        if (n < 2 || p[2 * n - 1] == 1) return -1;
        if (f.Per > 1 && (long)p[n - 1] * f.Bits % 32 != 0) return -1;
        for (int j = n - 2; j >= 0; j--) if (p[n + j] == 1 && p[j] > 1) return j;
        return -1;
    }

    /// <summary>A canonical strided view of rank n whose dimension j reads with stride 1 (StridedTileDim), copied by tiles of 32
    /// of dimension j × 32 of the last through workgroup memory (in the lane): a workgroup of 32 × 8 invocations reads its
    /// tile along j (consecutive invocations consecutive input values) and writes it along the last (consecutive output
    /// values; a packed type a word an invocation). Workgroups (⌈d_last / 32⌉ · ⌈d_j / 32⌉, the product of the other
    /// dimensions). The layout's slots (P as Layout's strided).</summary>
    public static SpvKernel StridedTiled(Vulkan vk, Fmt f, Lane lane, int n, int j)
    {
        const int T = TransposeTile, Y = 8, P = T + 1;
        var k = Kinds.Of(lane);
        var b = new Spv($"strided_tiled{n}_{j}_{f.Name}", T, vk.Controls, Y);
        var a = new LayoutIn(b);
        var p = a.P;
        V tx = b.LocalIdX, ty = b.LocalIdY;
        V dl = p[n - 1], dj = p[j], sl = p[2 * n - 1];
        var tilesL = (dl + (T - 1.0)) / (double)T;
        var tl = b.GroupX % tilesL;
        var tj = b.GroupX / tilesL;
        // The other dimensions' indices from the workgroup's y (innermost first), their input offset and output row.
        var t = b.GroupY;
        var inBase = p[2 * n];
        var rowBase = b.U(0);
        var rowStride = b.U(1);                                                    // the output rows a step of dimension d is
        var strideJ = b.U(1);
        for (int d = n - 2; d >= 0; d--)
        {
            if (d == j)
            {
                strideJ = rowStride;
            }
            else
            {
                var i = t % p[d];
                t = t / p[d];
                inBase = inBase + i * p[n + d];
                rowBase = rowBase + i * rowStride;
            }

            rowStride = rowStride * p[d];
        }

        var tile = b.Shared(k, T * P);
        var j0 = tj * (double)T;
        var l0 = tl * (double)T;
        for (int y = 0; y < T; y += Y)
        {
            var aj = j0 + tx;
            var al = l0 + ty + y;
            var at = (ty + y) * (double)P + tx;
            b.If((aj < dj) & (al < dl), () => tile[at] = Codec.UnpackAt(b, f, a.X, inBase + aj + al * sl, lane));
        }

        b.Barrier();
        var wk = Codec.WordKind(f);
        int bytes = f.WordBits / 8;
        if (f.Per == 1)
        {
            for (int y = 0; y < T; y += Y)
            {
                var aj = j0 + ty + y;
                var al = l0 + tx;
                var value = tile[tx * (double)P + ty + y];
                b.If((aj < dj) & (al < dl), () => b.Store(b.At(a.Out, (rowBase + aj * strideJ) * dl + al, bytes), Codec.Pack(b, f, value, lane, 0, b.CBits(wk, 0))));
            }

            return b.Build();
        }

        int words = T / f.Per;
        var tid = ty * (double)T + tx;
        b.For(tid, b.U(T * words), b.U(T * Y), w =>
        {
            var row = w / (double)words;
            var word = w - row * (double)words;
            var aj = j0 + row;
            var first = l0 + word * (double)f.Per;
            b.If((aj < dj) & (first < dl), () =>
            {
                var packed = b.CBits(wk, 0);
                for (int i = 0; i < f.Per; i++) packed = Codec.Pack(b, f, tile[(word * (double)f.Per + i) * P + row], lane, i, packed);
                b.Store(b.At(a.Out, ((rowBase + aj * strideJ) * dl + first) / (double)f.Per, bytes), packed);
            });
        });
        return b.Build();
    }

    /// <summary>The adaptive pools (forward) a workgroup an output word: each of the word's outputs' window read by the
    /// workgroup's invocations in strides (the window's pixels in row-major order j = tid, tid + 256, …), joined by the tree
    /// in workgroup memory: max pooling by First (Adad's order: the largest, a NaN where the window holds one), average
    /// pooling's sum (its terms' order the tree's), then divided by the window's size. The word packed and stored by the
    /// first invocation. Workgroups: the output's words (along x, then rows of them). The layout's slots.</summary>
    public static SpvKernel AdaptivePool(Vulkan vk, Fmt f, Lane lane, bool max)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"adaptive_pool_{(max ? "max" : "avg")}_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        var a = new LayoutIn(b);
        var p = a.P;
        V H = p[1], W = p[2], OH = p[3], OW = p[4];
        var tid = b.LocalIdX;
        var word = b.GroupY * b.GroupsX + b.GroupX;
        var natural = f.Natural;
        var nk = Kinds.Of(natural);
        var wk = Codec.WordKind(f);
        var sv = b.Shared(k, ReduceThreads);
        var packed = b.CBits(wk, 0);
        var last = a.Count - 1;
        for (int i = 0; i < f.Per; i++)
        {
            var at = word * (double)f.Per + i;
            var o = b.Min(at, last);
            var ow = o % OW;
            var r = o / OW;
            var oh = r % OH;
            var ch = r / OH;
            var (h0, hl) = AdaptiveSpan(oh, H, OH);
            var (w0, wl) = AdaptiveSpan(ow, W, OW);
            var acc = b.Local(max ? Identity(b, k, "max") : b.C(k, 0));
            b.For(tid, hl * wl, b.U(ReduceThreads), j =>
            {
                var dy = j / wl;
                var xv = Codec.UnpackAt(b, f, a.X, (ch * H + h0 + dy) * W + w0 + (j - dy * wl), lane);
                acc.V = max ? First(b, acc.V, xv) : acc.V + xv;
            });
            V value = max
                ? GroupReduce(b, sv, tid, acc.V, (x, y) => First(b, x, y), ReduceThreads)
                : GroupReduce(b, sv, tid, acc.V, (x, y) => x + y, ReduceThreads) / b.Convert(hl * wl, k);
            var stored = lane.Name == natural.Name ? value : Codec.Convert(b, lane, value, natural);
            packed = Codec.Pack(b, f, b.Select(at < a.Count, stored, b.C(nk, 0)), natural, i, packed);
        }

        var finalWord = packed;
        b.If((tid == 0) & (word < WordCount(b, f, a.Count)), () => b.Store(b.At(a.Out, word, f.WordBits / 8), finalWord));
        return b.Build();
    }

    /// <summary>The bytes BilinearTables takes for an output of OH × OW positions.</summary>
    public static long BilinearTableBytes(int oh, int ow, Lane lane) => (long)(oh + ow) * (8 + lane.Size);

    // An axis's position o (rows: o an output row, else an output column) from BilinearTables at aux: i0, i1, f, g = 1 − f.
    private static (V I0, V I1, V F, V G) TableAxis(Spv b, K k, int size, V aux, V oh, V ow, bool rows, V o)
    {
        var n = oh + ow;
        var at = rows ? o : oh + o;
        var f = b.Load(k, b.At(aux + b.Convert(n, K.U64) * 8.0, at, size));
        return (b.Load(K.U32, b.At(aux, at, 4)), b.Load(K.U32, b.At(aux, n + at, 4)), f, b.C(k, 1) - f);
    }

    // A nearest axis (LayoutArgs.NearestAt): min(max(2(o·α + β) + δ, 0) / 2δ, in − 1), signed (β may be negative).
    private static V NearestAxis(V o, V inp, V alpha, V beta, V delta)
    {
        var b = o.B;
        var t = b.Bitcast(b.Max(b.Bitcast(((o * alpha + beta) << 1) + delta, K.S32), b.S(0)), K.U32);
        return b.Min(t / (delta << 1), inp - 1);
    }

    // The outputs [lo, end) whose source rows include input row i (LayoutArgs.InterpolateSources, its hi + 1).
    private static (V Lo, V End) SourceRange(Spv b, bool bilinear, V i, V inp, V outp, V alpha, V beta, V delta)
    {
        var none = alpha == 0;                                                          // every output reads every row
        var a1 = b.Select(none, b.U(1), alpha);
        // ⌈n / α⌉ of a signed n, 0 where n ≤ 0 (the range starts at 0, or is empty).
        V CeilPos(V n) => b.Select(b.Bitcast(n, K.S32) > b.S(0), (n + a1 - 1) / a1, b.U(0));
        if (!bilinear) { a1 = a1 << 1; beta = (beta << 1) + delta; delta = delta << 1; }    // nearest: the floor of the map (2α, 2β + δ, 2δ)
        var lo = bilinear ? b.Select(i <= 1, b.U(0), CeilPos((i - 1) * delta - beta)) : b.Select(i == 0, b.U(0), CeilPos(i * delta - beta));
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
            {
                // An index outside the input's rows (p[0], unsigned: a negative one too) reads nothing: 0 (the address 0 a
                // safe one).
                var row = Index(a.Idx, r);
                var inside = row < p[0];
                return b.Select(inside, Load(b.Select(inside, row * cols + c, b.U(0))), Zero());
            }

            case "gather_cols":
            {
                var col = Index(a.Idx, c);
                var inside = col < p[1];
                return b.Select(inside, Load(b.Select(inside, r * p[1] + col, b.U(0))), Zero());
            }

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
                // Output (ch, ih, iw = c), r = ch·H + ih: Σ over (a, bb) with (ih + top − a·dh) = oh·sh, (iw + left − bb·dw) = ow·sw in range.
                V H = p[1], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8], dh = p[11], dw = p[12];
                var OH = p[13];
                var OW = p[14];
                var ih = r % H;
                var ch = r / H;
                var plane = OH * OW;
                var acc = b.Local(Zero());
                b.For(b.U(0), kh, ka =>
                {
                    var th = ih + top - ka * dh;                                        // may go negative: a signed test
                    var okH = (b.Bitcast(th, K.S32) >= b.S(0)) & (th % sh == 0) & (th / sh < OH);
                    b.If(okH, () => b.For(b.U(0), kw, bb =>
                    {
                        var tw = c + left - bb * dw;
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
                // r = ch·OH + oh, c = ow; p: C H W OH OW αh βh δh αw βw δw: ih the position nearest (oh·αh + βh) / δh (NearestAxis), iw likewise.
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
                // Bilinear: each position's rows and weight from the tables at aux (BilinearTables), AxisWeight's rule.
                (V, V) Weight(bool rows, V o, V i)
                {
                    if (!bilinear) return rows ? AxisWeight(b, k, false, o, i, H, p[5], p[6], p[7]) : AxisWeight(b, k, false, o, i, W, p[8], p[9], p[10]);
                    var (i0, i1, f, g) = TableAxis(b, k, lane.Size, a.Aux, OH, OW, rows, o);
                    var first = i0 == i;
                    return (first | (i1 == i), b.Select(first, g, f));
                }

                b.For(h0, h1, oh =>
                {
                    var (hasY, wy) = Weight(true, oh, ih);
                    b.If(hasY, () => b.For(w0, w1, ow =>
                    {
                        var (hasX, wx) = Weight(false, ow, c);
                        b.If(hasX, () => acc.V = b.Fma(Load((ch * OH + oh) * OW + ow), wy * wx, acc.V));
                    }));
                });
                return acc.V;
            }

            case "interpolate_bilinear":
            {
                // p: C H W OH OW αh βh δh αw βw δw: each axis's two rows and weight exactly (BilinearAxis), from the tables at
                // aux (BilinearTables).
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var oh = r % OH;
                var ch = r / OH;
                var (y0, y1, fy, gy) = TableAxis(b, k, lane.Size, a.Aux, OH, OW, true, oh);
                var (x0, x1, fx, gx) = TableAxis(b, k, lane.Size, a.Aux, OH, OW, false, c);
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
                // r = ch·OH + oh, c = ow; the window over (a, bb) from (oh·sh − top, ow·sw − left) with the dilations (1 for the
                // pools); OH from LayoutArgs.KernelP. A pool's window holds an input value (LayoutArgs.Validate): no empty case.
                bool max = name == "maxpool", conv = name == "depthwise_conv";
                V H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                V dh = conv ? p[11] : b.U(1), dw = conv ? p[12] : b.U(1);
                var OH = conv ? p[13] : p[11];
                var oh = r % OH;
                var ch = r / OH;
                var acc = b.Local(max ? Identity(b, k, "max") : Zero());
                var n = b.Local(b.U(0));
                b.For(b.U(0), kh, ka =>
                {
                    var ih = oh * sh - top + ka * dh;
                    b.If(ih < H, () => b.For(b.U(0), kw, bb =>                             // unsigned: negative rows are out of range
                    {
                        var iw = c * sw - left + bb * dw;
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
                if (conv || max) return acc.V;
                return acc.V / b.Convert(n.V, k);
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW), aux = each window's first-ranked (maxpool).
                bool max = name == "maxpool_bwd";
                V H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                var OH = p[11];
                var OW = p[12];
                var ih = r % H;
                var ch = r / H;
                var me = (ch * H + ih) * W + c;
                var acc = b.Local(Zero());
                b.For(b.U(0), kh, ka =>
                {
                    var th = ih + top - ka;
                    var okH = (b.Bitcast(th, K.S32) >= b.S(0)) & (th % sh == 0) & (th / sh < OH);
                    b.If(okH, () => b.For(b.U(0), kw, bb =>
                    {
                        var tw = c + left - bb;
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

                            // The window (oh, ow): its n values, dy / n each.
                            var cnt = WindowSpan(b, oh, sh, top, kh, H) * WindowSpan(b, ow, sw, left, kw, W);
                            acc.V = acc.V + dy / b.Convert(cnt, k);
                        });
                    }));
                });
                return acc.V;
            }

            case "adaptive_maxpool_bwd":
            case "adaptive_avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih: over the output rows and columns whose window holds the pixel, in
                // ascending order. Only the outputs from AdaptiveRange's first to its last can hold it (the windows' starts and
                // ends rise with the output), so the scan starts and ends there: the same windows in the same order.
                bool max = name == "adaptive_maxpool_bwd";
                V H = p[1], W = p[2], OH = p[3], OW = p[4];
                var ih = r % H;
                var ch = r / H;
                var me = (ch * H + ih) * W + c;
                var acc = b.Local(Zero());
                var (oh0, oh1) = AdaptiveRange(ih, H, OH);
                var (ow0, ow1) = AdaptiveRange(c, W, OW);
                b.For(oh0, oh1, oh =>
                {
                    var (h0, hl) = AdaptiveSpan(oh, H, OH);
                    b.If((ih >= h0) & (ih < h0 + hl), () => b.For(ow0, ow1, ow =>
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

}
