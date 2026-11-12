// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Layout and indexing: data movement by index arithmetic, exact in every type (the value moves as stored; the lane is
// the type's own; where values are summed — scatter-add, col2im — the sum lane by the bounds, stored as the type).
//
//   strided    a contiguous copy of a strided view of any rank: output[i0, …] = x[base + Σ i_d · stride_d] (permutes,
//              transposed views, slices, broadcasts with a zero stride); canonical first (LayoutArgs.StridedKernel)
//   one_hot    output[i][c] = (c == index[i]) ? 1 : 0 over `classes` columns
//   gather_rows   output[i] = x[index[i]] (a row each); gather_cols output[r][j] = x[r][index[j]]
//   scatter_add_rows   output[index[i]] += x[i] (the embedding gradient): every output row sums its sources in source
//              order (deterministic on every device)
//   An index outside its range selects nothing: no 1, a gathered 0, a source that adds nowhere.
//   im2col     x (C × H × W) to columns ((C·kh·kw) × (OH·OW)) for a kernel kh × kw, strides, a pad a side, dilations;
//   col2im     the reverse sum (each input pixel gathers its column entries in kernel order: exact order)
//   interpolate   x (C × H × W) resized to (C × OH × OW), nearest or bilinear, along each axis the source coordinate
//              s = (o·α + β) / δ exact in integers (LayoutArgs.NearestAt, BilinearAt); its backward gathers each input
//              position's outputs (LayoutArgs.InterpolateSources) with their weights
//   shift      output[r][c] = x[r][c − k] inside the row, 0 outside (k may be negative)
//   pad2d      rows × cols into rows × newCols: the row's values then zeros (the operands of a product whose shape the
//              product kernels' vector loads cannot take as stored)
//   maxpool / avgpool   x (C × H × W) over kh × kw windows at strides with a pad a side (LayoutArgs.PoolOut): max the
//              window's first-ranked value, avg the mean of its values (a pad is no value)
//   maxpool_bwd / avgpool_bwd   dx from dy: each input pixel gathers the windows it lies in (the first largest wins a
//              maxpool tie, recomputed: deterministic)
//   adaptive_maxpool / adaptive_avgpool   x (C × H × W) to C × OH × OW with windows [⌊i·H/OH⌋, ⌈(i + 1)·H/OH⌉)
//   depthwise_conv   y[c][oh][ow] = Σ w[c][a][b] · x[c][oh·sh − top + a·dh][ow·sw − left + b·dw] (the weights as a second
//              input; x 0 off the image)

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>A layout operation on the CPU; returns the lane's name (the type's own, or the sum lane where values are summed).</summary>
    public static string Layout<TN>(LayoutArgs k) where TN : struct, IAdad<TN>
    {
        k = k.ForKernels();
        if (LayoutArgs.Sums(k.Name))
        {
            // The bounds choose an integer type's sum lane; a float type's lane is its float lane whatever the values, so its
            // range is not read (a block format's range decodes every block, serially).
            var (lo, hi) = TN.IsFloat ? (0, 0) : TN.Range(k.X, k.InputCount);
            var (lo2, hi2) = TN.IsFloat || k.X2 is null || k.Name != "depthwise_conv" ? (1, 1) : TN.Range(k.X2, k.P[0] * k.P[3] * k.P[4]);
            var sum = new LayoutSumVisitor(k);
            TN.Sum(ref sum, new Bounds(lo, hi, lo2, hi2, LayoutArgs.Terms(k.Name, k.P)));
            return sum.Lane;
        }

        if (LayoutArgs.FloatLane(k.Name))
        {
            var fl = new LayoutFloatVisitor(k);
            TN.Elementwise(ref fl);
            return fl.Lane;
        }

        var v = new LayoutVisitor(k);
        TN.Elementwise(ref v);
        return v.Lane;
    }

    /// <summary>A task's reads of a layout input: in place where the type's bytes are the lane's (or the input is lane
    /// storage); otherwise chunks of 256 values (whole storage units of every type) decoded into lanes the first time a
    /// read reaches them, kept in a small cache of 64 sets of 4 (a chunk's set by a multiplicative hash of its index, so
    /// a gather's rows a power-of-two apart — a convolution's taps — do not share one) — an input is never decoded whole,
    /// and reads near one another share a chunk's decode.</summary>
    private sealed class Reader<TN, TLane> where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        private const int Size = 256, Shift = 8, Sets = 64, Ways = 4;
        private readonly byte[] x;
        private readonly long count;
        private readonly TLane[]? lanes;
        private readonly long[]? tags;
        private readonly byte[]? next;
        private readonly TLane[]? data;
        private long last = -1;
        private int lastAt;

        public Reader(byte[] x, long count, bool lanes = false)
        {
            if (TN.Granularity > Size || Size % TN.Granularity != 0) throw new NotSupportedException($"{TN.Name}: storage units of {TN.Granularity} values");
            this.x = x;
            this.count = count;
            if (lanes || TN.Direct)
            {
                // The bytes are lanes: read as the lane's array (indices within its values only).
                this.lanes = Unsafe.As<TLane[]>(x);
                return;
            }

            tags = new long[Sets * Ways];
            Array.Fill(tags, -1L);
            next = new byte[Sets];
            data = ArrayPool<TLane>.Shared.Rent(Sets * Ways * Size);
        }

        /// <summary>The input's bytes are lanes: read in place.</summary>
        public bool InPlace => lanes is not null;

        public TLane this[long index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (lanes is not null) return lanes[index];
                long c = index >> Shift;
                if (c != last) Find(c);
                return data![lastAt + (int)(index & (Size - 1))];
            }
        }

        /// <summary>count values from `at`: in place, or loaded into `buffer`.</summary>
        public ReadOnlySpan<TLane> Span(long at, int length, TLane[] buffer)
        {
            if (lanes is not null) return lanes.AsSpan((int)at, length);
            TN.Load(x, (int)at, buffer.AsSpan(0, length));
            return buffer.AsSpan(0, length);
        }

        private void Find(long c)
        {
            int set = (int)(((ulong)c * 0x9E3779B97F4A7C15UL) >> 58) * Ways;
            for (int w = 0; w < Ways; w++)
            {
                if (tags![set + w] != c) continue;
                last = c;
                lastAt = (set + w) * Size;
                return;
            }

            // A miss: the set's next way in turn.
            int slot = set + next![set / Ways];
            next[set / Ways] = (byte)((next[set / Ways] + 1) & (Ways - 1));
            long at = c << Shift;
            TN.Load(x, (int)at, data.AsSpan(slot * Size, (int)Math.Min(Size, count - at)));
            tags![slot] = c;
            last = c;
            lastAt = slot * Size;
        }

        public void Return()
        {
            if (data is not null) ArrayPool<TLane>.Shared.Return(data);
        }
    }

    // The values of a layout's second input (depthwise_conv's weights; the max-pool backward's forward input).
    private static long SecondCount(LayoutArgs k) => k.Name == "depthwise_conv" ? (long)k.P[0] * k.P[3] * k.P[4] : (long)k.P[0] * k.P[1] * k.P[2];

    /// <summary>A layout whose input (k.X) is lane storage — the float lane's own values, an intermediate (the convolution's
    /// columns' gradient) — and whose output is TN: each output row computed in the lane and stored as the type.</summary>
    internal static void LayoutFromLanes<TN, TLane, TL>(LayoutArgs k) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        var (rows, cols) = OutputRows(k);
        Move<TN, TLane, TL>(k, rows, cols, null, inLanes: true);
    }

    /// <summary>A layout whose output (k.Output) is lane storage — an intermediate (the convolution's columns) — from TN:
    /// each output row computed in the lane and kept there.</summary>
    internal static void LayoutToLanes<TN, TLane, TL>(LayoutArgs k) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        var (rows, cols) = OutputRows(k);
        Team.For(rows, Math.Max(1, rows / (Threads * 4)), Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, null, outLanes: true));
    }

    /// <summary>The output rows computed by the team: tasks of whole rows, or — where the rows that share an output storage
    /// unit are too many to give the team its tasks (a block format's rows far shorter than its block) — tasks of whole
    /// storage units, a row crossing a task's edge computed by both and each storing its own values.</summary>
    private static void Move<TN, TLane, TL>(LayoutArgs k, int rows, int cols, long[]? at, bool inLanes = false)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        if ((long)rows * cols == 0) return;                                        // no outputs (a window longer than its image)
        int wanted = Math.Max(1, rows / (Threads * 4)), chunk = RowChunk<TN>(rows, cols, wanted), g = TN.Granularity;
        if (g <= 1 || chunk <= 2 * wanted)
        {
            Team.For(rows, chunk, Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, at, inLanes));
            return;
        }

        // Each task at least four rows' values (a crossing row computed twice at most once in four).
        long n = (long)rows * cols;
        int units = (int)((n + g - 1) / g);
        int per = Math.Max(Math.Max(1, units / (Threads * 4)), (int)((4L * cols + g - 1) / g));
        Team.For(units, per, Work(k), new MoveBody<TN, TLane, TL>(k, rows, cols, at, inLanes, byUnits: true));
    }

    // The team's measure of a layout job: its outputs by the reads each makes.
    private static long Work(LayoutArgs k) => (long)k.Count * LayoutArgs.ValueReads(k.Name, k.P);

    private struct LayoutVisitor(LayoutArgs k) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            var at = Aux<TN, TLane>(k);
            if (StridedTiled<TN, TLane>(k, cols, out int dim))
            {
                // A tile's line is whole storage units where the type's bytes are not lanes (a block decoded once, not once
                // a tile).
                int tile = Math.Max(16, 64 / System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>());
                if (!TN.Direct) tile = Math.Max(tile, TN.Granularity);
                // The output's columns a group covers: 256, whole storage units (a group's buffer of tile × span lanes stays
                // in the nearest caches, and its reads keep to span lines of the input until the next group takes the lines
                // after them).
                int g = TN.Granularity, span = Math.Min((256 + g - 1) / g * g, cols);
                var p = k.P;
                long groups = (long)rows / p[dim] * ((p[dim] + tile - 1) / tile) * ((cols + span - 1) / span);
                Team.For((int)groups, Math.Max(1, (int)(groups / (Threads * 4))), (long)k.Count, new StridedTileBody<TN, TLane>(k, cols, dim, tile, span));
            }
            else Move<TN, TLane, TL>(k, rows, cols, at);
        }
    }

    private struct LayoutFloatVisitor(LayoutArgs k) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            var at = Aux<TN, TLane>(k);
            Move<TN, TLane, TL>(k, rows, cols, at);
        }
    }

    private struct LayoutSumVisitor(LayoutArgs k) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var (rows, cols) = OutputRows(k);
            var at = Aux<TN, TLane>(k);
            Move<TN, TLane, TL>(k, rows, cols, at);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            // Sums of single values need no dot product: the accumulator lane's plain arithmetic (s32/u32 for the 8- and
            // 16-bit types). The plain lane struct is chosen by the accumulator type at run time (the number implements
            // IAdad<TN, TAcc>, which the instantiation checks).
            var plain = typeof(TAcc) == typeof(int) ? typeof(I32) : typeof(U32);
            Lane = typeof(TAcc) == typeof(int) ? "s32" : "u32";
            var method = typeof(LayoutSumVisitor).GetMethod(nameof(Plain), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
            method.MakeGenericMethod(typeof(TN), typeof(TAcc), plain).Invoke(null, [k]);
        }

        private static void Plain<TN, TLane, TL>(LayoutArgs k) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            var (rows, cols) = OutputRows(k);
            var at = Aux<TN, TLane>(k);
            Move<TN, TLane, TL>(k, rows, cols, at);
        }
    }

    private static (int Rows, int Cols) OutputRows(LayoutArgs k) => LayoutArgs.OutputRows(k);

    // A mean in the lane: sum / n, one rounding (the devices' division); the float lanes' scalar division where the lane is
    // one (the JIT keeps only the branch of the lane).
    private static TLane Mean<TLane, TL>(TLane sum, int n) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        if (typeof(TLane) == typeof(float)) { float m = Unsafe.As<TLane, float>(ref sum) / n; return Unsafe.As<float, TLane>(ref m); }
        if (typeof(TLane) == typeof(double)) { double m = Unsafe.As<TLane, double>(ref sum) / n; return Unsafe.As<double, TLane>(ref m); }
        return TL.Div(new Vector<TLane>(sum), new Vector<TLane>(TL.From(n)))[0];
    }

    // A bilinear axis's weights in the lane: f = rem / δ (one rounding), g = 1 − f.
    private static (TLane F, TLane G) Weights<TLane, TL>(int rem, int delta) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        var f = TL.Div(new Vector<TLane>(TL.From(rem)), new Vector<TLane>(TL.From(delta)));
        return (f[0], TL.Sub(new Vector<TLane>(TL.From(1)), f)[0]);
    }

    // An interpolation's column axis, the same for every row (found once a task): nearest, each output column's source;
    // bilinear, each output column's two sources (the result) and weights f g (w); a backward, for each input column its
    // outputs (the result: W + 1 starts, then the outputs in order) and their weights (w).
    private static int[] InterpolateAxis<TLane, TL>(string name, int[] p, out TLane[]? w) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int W = p[2], OW = p[4];
        w = null;
        if (name == "interpolate_nearest")
        {
            var at = new int[OW];
            for (int ow = 0; ow < OW; ow++) at[ow] = LayoutArgs.NearestAt(ow, W, p[8], p[9], p[10]);
            return at;
        }

        if (name == "interpolate_bilinear")
        {
            var at = new int[2 * OW];
            w = new TLane[2 * OW];
            for (int ow = 0; ow < OW; ow++)
            {
                var (x0, x1, rx) = LayoutArgs.BilinearAt(ow, W, p[8], p[9], p[10]);
                (w[2 * ow], w[2 * ow + 1]) = Weights<TLane, TL>(rx, p[10]);
                (at[2 * ow], at[2 * ow + 1]) = (x0, x1);
            }

            return at;
        }

        bool bilinear = name == "interpolate_bilinear_bwd";
        var starts = new int[W + 1];
        var outs = new List<int>();
        var weights = new List<TLane>();
        for (int iw = 0; iw < W; iw++)
        {
            starts[iw] = outs.Count;
            var (w0, w1) = LayoutArgs.InterpolateSources(bilinear, iw, W, OW, p[8], p[9], p[10]);
            for (int ow = w0; ow <= w1; ow++)
                if (InterpolateWeight<TLane, TL>(bilinear, ow, iw, W, p[8], p[9], p[10], out TLane wx)) { outs.Add(ow); weights.Add(wx); }
        }

        starts[W] = outs.Count;
        w = [.. weights];
        return [.. starts, .. outs];
    }

    // The weight output o of an interpolation axis gives input position i: nearest 1 where it reads i; bilinear the first
    // row's weight g where its first row is i, else f where its second is (both are i only past the input's end, f then 0).
    private static bool InterpolateWeight<TLane, TL>(bool bilinear, int o, int i, int inp, int alpha, int beta, int delta, out TLane w)
        where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        w = TL.From(1);
        if (!bilinear) return LayoutArgs.NearestAt(o, inp, alpha, beta, delta) == i;
        var (i0, i1, rem) = LayoutArgs.BilinearAt(o, inp, alpha, beta, delta);
        if (i0 != i && i1 != i) return false;
        var (f, g) = Weights<TLane, TL>(rem, delta);
        w = i0 == i ? g : f;
        return true;
    }

    /// <summary>A strided copy whose output rows run across the input (the last dimension's stride not 1) while an outer
    /// dimension has stride 1 (a transposition): taken by tiles of output rows along the innermost such dimension (see
    /// StridedTileBody). Rows ending on whole bytes only (each row stored by one thread), values read in place or through
    /// the task's reader.</summary>
    private static bool StridedTiled<TN, TLane>(LayoutArgs k, int cols, out int dim) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        dim = -1;
        int rank = k.P.Length / 2;
        if (k.Name != "strided" || rank < 2 || k.P[2 * rank - 1] == 1 || (long)cols * TN.Bits % 8 != 0) return false;
        for (int j = rank - 2; j >= 0; j--) if (k.P[rank + j] == 1 && k.P[j] > 1) { dim = j; return true; }
        return false;
    }

    /// <summary>The tiled strided copy: a group is `tile` output rows that differ only in dimension `dim` (whose input
    /// stride is 1) at one index of the other outer dimensions, over `span` of their columns; for each output column the
    /// group's values lie side by side in the input (a cache line read once, not a line a value), gathered into the
    /// group's rows (a tile × span block in lane storage), each row's span then stored. Groups are disjoint blocks of whole
    /// storage units; consecutive groups take the next tile of rows over the same columns (the input lines after the last
    /// group's).</summary>
    private struct StridedTileBody<TN, TLane>(LayoutArgs k, int cols, int dim, int tile, int span) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var p = k.P;
            int rank = p.Length / 2, outer = rank - 1, dj = p[dim], blocks = (dj + tile - 1) / tile;
            long sLast = p[2 * rank - 1], rowStep = 1;
            for (int i = dim + 1; i < outer; i++) rowStep *= p[i];      // the output rows between two indices of dim
            ReadOnlySpan<TLane> src = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(k.X.AsSpan()) : default;
            var input = TN.Direct ? null : new Reader<TN, TLane>(k.X, k.InputCount);
            var lineBuffer = TN.Direct ? null : ArrayPool<TLane>.Shared.Rent(tile);
            var buffer = ArrayPool<TLane>.Shared.Rent(tile * span);
            int spans = (cols + span - 1) / span;
            try
            {
                for (int g = first; g < last; g++)
                {
                    // g = (rest · spans + s) · blocks + block; rest enumerates the other outer dimensions (in their order).
                    int block = g % blocks, lo = block * tile, count = Math.Min(tile, dj - lo);
                    int s = g / blocks % spans, c0 = s * span, cc = Math.Min(span, cols - c0);
                    long rest = g / blocks / spans, b0 = p[2 * rank] + lo, r0 = 0, step = 1;     // value lo of the group (stride 1 in t)
                    for (int i = outer - 1; i >= 0; i--)
                    {
                        long index = i == dim ? lo : rest % p[i];
                        if (i != dim) { rest /= p[i]; b0 += index * p[rank + i]; }
                        r0 += index * step;
                        step *= p[i];
                    }

                    var t2 = buffer.AsSpan(0, tile * cc);
                    for (int c = 0; c < cc; c++)
                    {
                        long at = b0 + (c0 + c) * sLast;
                        var line = input is null ? src.Slice((int)at, count) : input.Span(at, count, lineBuffer!);
                        for (int t = 0; t < count; t++) t2[t * cc + c] = line[t];
                    }

                    for (int t = 0; t < count; t++) TN.Store(t2.Slice(t * cc, cc), k.Output, (int)((r0 + t * rowStep) * cols + c0));
                }
            }
            finally
            {
                ArrayPool<TLane>.Shared.Return(buffer);
                if (lineBuffer is not null) ArrayPool<TLane>.Shared.Return(lineBuffer);
                input?.Return();
            }
        }
    }

    /// <summary>What the rows read besides the inputs, found once. The max-pool backwards: the first largest of every
    /// window of the forward input (x2), found by the team (index into the input; every window holds a value), so each
    /// input row reads its windows' choice instead of scanning every window it lies in. scatter_add_rows: its offsets
    /// (rows + 1), then its sources in order of their row (Route: source order within a row, an index outside the rows in
    /// none), as the devices' aux. Null for the other operations.</summary>
    private static long[]? Aux<TN, TLane>(LayoutArgs k) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        if (k.Name == "scatter_add_rows")
        {
            int sources = k.P[0], targets = k.P[2];
            var index = k.Index!.Length == sources ? k.Index : k.Index[..sources];
            var offsets = new int[targets + 1];
            var order = new int[sources];
            Route(index, targets, new int[targets], offsets, order);
            var aux = new long[targets + 1 + sources];
            for (int t = 0; t <= targets; t++) aux[t] = offsets[t];
            for (int q = 0; q < sources; q++) aux[targets + 1 + q] = order[q];
            return aux;
        }

        if (k.Name is not ("maxpool_bwd" or "adaptive_maxpool_bwd")) return null;
        var p = k.P;
        int C = p[0], H = p[1], W = p[2];
        var (OH, OW) = k.Name == "maxpool_bwd" ? LayoutArgs.PoolOut(p) : (p[3], p[4]);
        int windows = C * OH * OW, n = C * H * W;
        var at = new long[windows];
        Team.For(windows, Math.Max(1, windows / (Threads * 4)), (long)n, new ArgmaxBody<TN, TLane>(k, at, OH, OW));
        return at;
    }

    private readonly struct ArgmaxBody<TN, TLane>(LayoutArgs k, long[] at, int OH, int OW) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        public void Run(int first, int last)
        {
            var p = k.P;
            int H = p[1], W = p[2];
            var x = new Reader<TN, TLane>(k.X2!, (long)p[0] * H * W);
            bool adaptive = k.Name == "adaptive_maxpool_bwd";
            for (int i = first; i < last; i++)
            {
                int ow = i % OW, oh = i / OW % OH, c = i / (OW * OH);
                int h0, h1, w0, w1;
                if (adaptive) { (h0, h1) = LayoutArgs.Adaptive(oh, H, OH); (w0, w1) = LayoutArgs.Adaptive(ow, W, OW); }
                else { h0 = oh * p[5] - p[7]; h1 = h0 + p[3]; w0 = ow * p[6] - p[8]; w1 = w0 + p[4]; }
                h0 = Math.Max(h0, 0); h1 = Math.Min(h1, H); w0 = Math.Max(w0, 0); w1 = Math.Min(w1, W);
                // The first-ranked in row-major order (Adad's order, Widths.Compare): a later value takes it only when it
                // ranks strictly above.
                long bestAt = -1;
                TLane best = default;
                for (int hh = h0; hh < h1; hh++) for (int ww = w0; ww < w1; ww++)
                {
                    long j = ((long)c * H + hh) * W + ww;
                    TLane v = x[j];
                    if (bestAt < 0 || Widths.Compare(v, best) > 0) { best = v; bestAt = j; }
                }

                at[i] = bestAt;
            }

            x.Return();
        }
    }

    /// <summary>Every output row computed value by value from the input's lane values (loaded one at a time by index:
    /// the operations are gathers), the row stored once; rows chunked on whole bytes for the packed types.</summary>
    // inLanes: the input is lane storage (an intermediate); outLanes: the output is kept as lane storage.
    // byUnits: first and last count the output's storage units (TN.Granularity values), not its rows.
    private struct MoveBody<TN, TLane, TL>(LayoutArgs k, int rows, int cols, long[]? at = null, bool inLanes = false, bool outLanes = false, bool byUnits = false) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            if (byUnits)
            {
                long v0 = (long)first * TN.Granularity, v1 = Math.Min((long)rows * cols, (long)last * TN.Granularity);
                RunRows((int)(v0 / cols), (int)((v1 - 1) / cols) + 1, k, rows, cols, exclusiveBytes: false, at, inLanes, outLanes: false, v0, v1);
                return;
            }

            // Chunks of rows that share bytes are aligned by the caller's chunking only when cols·bits ends on a byte; here
            // every row store is through a whole-row buffer and the packed types store with Packed.Store8 (any offset),
            // so two threads may touch one byte at a row boundary: take whole bytes by widening the chunk to byte ends.
            int per = TN.Granularity;
            long startBit = (long)first * cols * TN.Bits, endBit = (long)last * cols * TN.Bits;
            if (per > 1 && (startBit % 8 != 0 || endBit % 8 != 0))
            {
                // Compute into a lane buffer for these rows and store value-aligned pieces: the first and last partial bytes
                // are stored one value at a time by the same thread only when it owns the byte's first value.
                RunRows(first, last, k, rows, cols, exclusiveBytes: true, at, inLanes, outLanes);
                return;
            }

            RunRows(first, last, k, rows, cols, exclusiveBytes: false, at, inLanes, outLanes);
        }

        // The input and the second input read through the task's readers (in place, or chunks decoded as reached).
        // at: what the rows read besides the inputs (Aux): the max-pool backward's first largest of every window, or
        // scatter-add's offsets and ordered sources.
        // v0 ≥ 0: only the output values [v0, v1) are this task's (whole storage units), gathered and stored at once.
        private static void RunRows(int first, int last, LayoutArgs k, int rows, int cols, bool exclusiveBytes, long[]? at, bool inLanes, bool outLanes, long v0 = -1, long v1 = -1)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(cols);
            var tile = v0 >= 0 ? pool.Rent((int)(v1 - v0)) : null;
            TLane[]? source = null, source2 = null;
            TLane[]? weights = null;                                                    // an interpolation backward's row weights and their places
            int[]? places = null;
            // An interpolation's column axis, found once a task (InterpolateAxis): the sources and weights of every column.
            int[]? axis = null;
            TLane[]? axisW = null;
            var input = new Reader<TN, TLane>(k.X, k.InputCount, inLanes);
            var second = k.X2 is null ? null : new Reader<TN, TLane>(k.X2, SecondCount(k));
            try
            {
                var p = k.P;
                var idx = k.Index;
                for (int r = first; r < last; r++)
                {
                    var o = row.AsSpan(0, cols);
                    switch (k.Name)
                    {
                        case "strided":
                        {
                            // Row r over the outer dimensions, the row along the last.
                            int rank = p.Length / 2;
                            long b = p[2 * rank];
                            for (int d = rank - 2, t = r; d >= 0; d--) { b += (long)(t % p[d]) * p[rank + d]; t /= p[d]; }
                            long step = p[2 * rank - 1];
                            for (int c = 0; c < cols; c++) o[c] = input[b + c * step];
                            break;
                        }

                        case "one_hot":
                            o.Clear();
                            if ((uint)idx![r] < (uint)cols) o[idx[r]] = TL.From(1);
                            break;
                        case "gather_rows":
                            if ((uint)idx![r] < (uint)p[0]) TN.Load(k.X, idx[r] * cols, o);
                            else o.Clear();
                            break;
                        case "gather_cols":
                        {
                            source ??= pool.Rent(p[1]);
                            GatherColsRow(input, idx!, p[1], r, o, source);
                            break;
                        }
                        case "scatter_add_rows":
                        {
                            // Output row r: the sum of its source rows in source order (at: the offsets, then the sources in
                            // order of their row); a source row read whole (one load of its values: a block format decodes
                            // each block once, not once per value).
                            o.Clear();
                            for (long q = at![r], end = at[r + 1], sources = p[2] + 1; q < end; q++)
                            {
                                source ??= pool.Rent(cols);
                                var s = input.Span(at[sources + q] * cols, cols, source);
                                for (int c = 0; c < cols; c++) o[c] = TL.Add(o[c], s[c]);
                            }

                            break;
                        }

                        case "im2col":
                        {
                            var (OH, OW) = LayoutArgs.ConvOut(p);
                            int kw = p[4], kh = p[3], H = p[1], W = p[2];
                            int b = r % kw, a = r / kw % kh, c = r / (kw * kh);
                            for (int oh = 0; oh < OH; oh++)
                            {
                                int ih = oh * p[5] - p[7] + a * p[11];
                                for (int ow = 0; ow < OW; ow++)
                                {
                                    int iw = ow * p[6] - p[8] + b * p[12];
                                    o[oh * OW + ow] = (uint)ih < (uint)H && (uint)iw < (uint)W ? input[((long)c * H + ih) * W + iw] : TL.Zero;
                                }
                            }

                            break;
                        }

                        case "col2im":
                            Col2ImRow(input, p, r, o);
                            break;

                        case "interpolate_nearest":
                        {
                            // p: C H W OH OW αh βh δh αw βw δw (LayoutArgs.NearestAt).
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            int ih = LayoutArgs.NearestAt(oh, H, p[5], p[6], p[7]);
                            source ??= pool.Rent(W);
                            var line = input.Span(((long)c * H + ih) * W, W, source);
                            axis ??= InterpolateAxis<TLane, TL>(k.Name, p, out axisW);
                            for (int ow = 0; ow < OW; ow++) o[ow] = line[axis[ow]];
                            break;
                        }

                        case "interpolate_bilinear":
                        {
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            var (y0, y1, ry) = LayoutArgs.BilinearAt(oh, H, p[5], p[6], p[7]);
                            var (fy, gy) = Weights<TLane, TL>(ry, p[7]);
                            // The two input rows read in place or loaded once each.
                            source ??= pool.Rent(W);
                            source2 ??= pool.Rent(W);
                            var row0 = input.Span(((long)c * H + y0) * W, W, source);
                            var row1 = input.Span(((long)c * H + y1) * W, W, source2);
                            axis ??= InterpolateAxis<TLane, TL>(k.Name, p, out axisW);
                            for (int ow = 0; ow < OW; ow++)
                            {
                                int x0 = axis[2 * ow], x1 = axis[2 * ow + 1];
                                TLane fx = axisW![2 * ow], gx = axisW[2 * ow + 1];
                                TLane top = TL.Fma(row0[x1], fx, TL.Mul(row0[x0], gx));
                                TLane bottom = TL.Fma(row1[x1], fx, TL.Mul(row1[x0], gx));
                                o[ow] = TL.Fma(bottom, fy, TL.Mul(top, gy));
                            }

                            break;
                        }

                        case "interpolate_nearest_bwd":
                        case "interpolate_bilinear_bwd":
                        {
                            weights ??= pool.Rent(p[3]);
                            places ??= ArrayPool<int>.Shared.Rent(p[3]);
                            axis ??= InterpolateAxis<TLane, TL>(k.Name, p, out axisW);
                            InterpolateBackRow(k.Name == "interpolate_bilinear_bwd", input, p, r, o, weights, places, axis, axisW!);
                            break;
                        }

                        case "shift":
                            for (int c = 0; c < cols; c++) { int s = c - p[2]; o[c] = (uint)s < (uint)cols ? input[(long)r * cols + s] : TL.Zero; }
                            break;
                        case "pad2d":
                            o.Clear();
                            TN.Load(k.X, r * p[1], o[..p[1]]);
                            break;
                        case "maxpool":
                        case "avgpool":
                        {
                            // Every window holds an input value (LayoutArgs.Validate): the mean is the sum over its n values / n.
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                            int oh = r % OH, c = r / OH;
                            bool max = k.Name == "maxpool";
                            for (int ow = 0; ow < OW; ow++)
                            {
                                TLane acc = max ? TL.From(double.NegativeInfinity) : TL.Zero;
                                int n = 0;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - top + a;
                                    if ((uint)ih >= (uint)H) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int iw = ow * sw - left + b;
                                        if ((uint)iw >= (uint)W) continue;
                                        TLane v = input[((long)c * H + ih) * W + iw];
                                        acc = max ? (n == 0 || Widths.Compare(v, acc) > 0 ? v : acc) : TL.Add(acc, v);
                                        n++;
                                    }
                                }

                                o[ow] = max ? acc : Mean<TLane, TL>(acc, n);
                            }

                            break;
                        }

                        case "maxpool_bwd":
                        case "avgpool_bwd":
                        {
                            // Input pixel (c, ih, iw = col): the windows it lies in; maxpool: those whose first largest it is.
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                            int ih = r % H, c = r / H;
                            bool max = k.Name == "maxpool_bwd";
                            if (max && at is not null)
                            {
                                // The windows over row ih in the order a pixel met them (kernel row a ascending: oh descending;
                                // then ow descending), each adding dy where its first largest lies on this row.
                                o.Clear();
                                long rowAt = (long)c * H + ih;
                                for (int a = 0; a < kh; a++)
                                {
                                    int th = ih + top - a;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    int oh = th / sh;
                                    for (int ow = OW - 1; ow >= 0; ow--)
                                    {
                                        long w = at[((long)c * OH + oh) * OW + ow];
                                        if (w >= 0 && w / W == rowAt) o[(int)(w % W)] = TL.Add(o[(int)(w % W)], input[((long)c * OH + oh) * OW + ow]);
                                    }
                                }

                                break;
                            }

                            for (int iw = 0; iw < W; iw++)
                            {
                                TLane acc = TL.Zero;
                                for (int a = 0; a < kh; a++)
                                {
                                    int th = ih + top - a;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    int oh = th / sh;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int tw = iw + left - b;
                                        if (tw < 0 || tw % sw != 0 || tw / sw >= OW) continue;
                                        int ow = tw / sw;
                                        TLane dy = input[((long)c * OH + oh) * OW + ow];
                                        if (!max)
                                        {
                                            // The window's share: dy over its n values.
                                            var (h0, h1) = LayoutArgs.Window(oh, H, kh, sh, top);
                                            var (w0, w1) = LayoutArgs.Window(ow, W, kw, sw, left);
                                            acc = TL.Add(acc, Mean<TLane, TL>(dy, (h1 - h0) * (w1 - w0)));
                                            continue;
                                        }

                                        // The window's first largest: this pixel takes dy when it is that one.
                                        long bestAt = -1;
                                        TLane best = TL.From(double.NegativeInfinity);
                                        for (int aa = 0; aa < kh; aa++)
                                        {
                                            int hh = oh * sh - top + aa;
                                            if ((uint)hh >= (uint)H) continue;
                                            for (int bb = 0; bb < kw; bb++)
                                            {
                                                int ww = ow * sw - left + bb;
                                                if ((uint)ww >= (uint)W) continue;
                                                TLane v = second![((long)c * H + hh) * W + ww];
                                                if (bestAt < 0 || Widths.Compare(v, best) > 0) { best = v; bestAt = ((long)c * H + hh) * W + ww; }
                                            }
                                        }

                                        if (bestAt == ((long)c * H + ih) * W + iw) acc = TL.Add(acc, dy);
                                    }
                                }

                                o[iw] = acc;
                            }

                            break;
                        }

                        case "adaptive_maxpool":
                        case "adaptive_avgpool":
                        {
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            bool max = k.Name == "adaptive_maxpool";
                            var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                            for (int ow = 0; ow < OW; ow++)
                            {
                                var (w0, w1) = LayoutArgs.Adaptive(ow, W, OW);
                                TLane acc = TL.Zero;
                                bool unset = true;
                                for (int ih = h0; ih < h1; ih++) for (int iw = w0; iw < w1; iw++)
                                {
                                    TLane v = input[((long)c * H + ih) * W + iw];
                                    acc = max ? (unset || Widths.Compare(v, acc) > 0 ? v : acc) : TL.Add(acc, v);
                                    unset = false;
                                }

                                o[ow] = max ? acc : Mean<TLane, TL>(acc, (h1 - h0) * (w1 - w0));
                            }

                            break;
                        }

                        case "adaptive_maxpool_bwd":
                        case "adaptive_avgpool_bwd":
                        {
                            // Input pixel (c, ih, iw = col): the windows holding it (an adaptive window's rows are those oh with h0(oh) ≤ ih < h1(oh)).
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int ih = r % H, c = r / H;
                            bool max = k.Name == "adaptive_maxpool_bwd";
                            if (!max)
                            {
                                // Each window holding row ih adds its share to its columns, the windows in the order a pixel
                                // met them (oh, then ow ascending): each pixel's sum the same terms in the same order.
                                o.Clear();
                                for (int oh = 0; oh < OH; oh++)
                                {
                                    var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                                    if (ih < h0 || ih >= h1) continue;
                                    for (int ow = 0; ow < OW; ow++)
                                    {
                                        var (w0, w1) = LayoutArgs.Adaptive(ow, W, OW);
                                        TLane share = Mean<TLane, TL>(input[((long)c * OH + oh) * OW + ow], (h1 - h0) * (w1 - w0));
                                        for (int iw = w0; iw < w1; iw++) o[iw] = TL.Add(o[iw], share);
                                    }
                                }

                                break;
                            }

                            if (max && at is not null)
                            {
                                // The windows over row ih in the order a pixel met them (oh, then ow ascending).
                                o.Clear();
                                long rowAt = (long)c * H + ih;
                                for (int oh = 0; oh < OH; oh++)
                                {
                                    var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                                    if (ih < h0 || ih >= h1) continue;
                                    for (int ow = 0; ow < OW; ow++)
                                    {
                                        long w = at[((long)c * OH + oh) * OW + ow];
                                        if (w >= 0 && w / W == rowAt) o[(int)(w % W)] = TL.Add(o[(int)(w % W)], input[((long)c * OH + oh) * OW + ow]);
                                    }
                                }

                                break;
                            }

                            for (int iw = 0; iw < W; iw++)
                            {
                                TLane acc = TL.Zero;
                                for (int oh = 0; oh < OH; oh++)
                                {
                                    var (h0, h1) = LayoutArgs.Adaptive(oh, H, OH);
                                    if (ih < h0 || ih >= h1) continue;
                                    for (int ow = 0; ow < OW; ow++)
                                    {
                                        var (w0, w1) = LayoutArgs.Adaptive(ow, W, OW);
                                        if (iw < w0 || iw >= w1) continue;
                                        TLane dy = input[((long)c * OH + oh) * OW + ow];
                                        if (!max) { acc = TL.Add(acc, Mean<TLane, TL>(dy, (h1 - h0) * (w1 - w0))); continue; }
                                        long bestAt = -1;
                                        TLane best = TL.Zero;
                                        for (int hh = h0; hh < h1; hh++) for (int ww = w0; ww < w1; ww++)
                                        {
                                            TLane v = second![((long)c * H + hh) * W + ww];
                                            if (bestAt < 0 || Widths.Compare(v, best) > 0) { best = v; bestAt = ((long)c * H + hh) * W + ww; }
                                        }

                                        if (bestAt == ((long)c * H + ih) * W + iw) acc = TL.Add(acc, dy);
                                    }
                                }

                                o[iw] = acc;
                            }

                            break;
                        }

                        case "depthwise_conv":
                        {
                            var (OH, OW) = LayoutArgs.ConvOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8], dh = p[11], dw = p[12];
                            int oh = r % OH, c = r / OH;
                            if (sw == 1)
                            {
                                // Stride 1 along the row: each tap (a, b) a vector pass over the outputs its column range reaches,
                                // the taps in the scalar loop's order (each output's sum the same terms in the same order, bit for
                                // bit: the lane's vector and scalar Fma are the same operation); an input row read in place or
                                // loaded once for its taps.
                                source ??= pool.Rent(W);
                                o.Clear();
                                int vl = Vector<TLane>.Count;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - top + a * dh;
                                    if ((uint)ih >= (uint)H) continue;
                                    long rowAt = ((long)c * H + ih) * W;
                                    var line = input.Span(rowAt, W, source);
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int off = b * dw - left, lo = Math.Max(0, -off), hi = Math.Min(OW, W - off);
                                        if (lo >= hi) continue;
                                        TLane wv = second![(long)(c * kh + a) * kw + b];
                                        var wvec = new Vector<TLane>(wv);
                                        var src = line.Slice(lo + off, hi - lo);
                                        var dst = o.Slice(lo, hi - lo);
                                        int i = 0;
                                        for (; i + vl <= dst.Length; i += vl)
                                            TL.Fma(wvec, new Vector<TLane>(src.Slice(i)), new Vector<TLane>(dst.Slice(i))).CopyTo(dst.Slice(i));
                                        for (; i < dst.Length; i++) dst[i] = TL.Fma(wv, src[i], dst[i]);
                                    }
                                }

                                break;
                            }

                            for (int ow = 0; ow < OW; ow++)
                            {
                                TLane acc = TL.Zero;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - top + a * dh;
                                    if ((uint)ih >= (uint)H) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int iw = ow * sw - left + b * dw;
                                        if ((uint)iw >= (uint)W) continue;
                                        acc = TL.Fma(second![(long)(c * kh + a) * kw + b], input[((long)c * H + ih) * W + iw], acc);
                                    }
                                }

                                o[ow] = acc;
                            }

                            break;
                        }
                    }

                    if (outLanes) o.CopyTo(MemoryMarshal.Cast<byte, TLane>(k.Output.AsSpan()).Slice(r * cols, cols));
                    else if (tile is not null)
                    {
                        long rs = (long)r * cols, s0 = Math.Max(v0, rs), s1 = Math.Min(v1, rs + cols);
                        o.Slice((int)(s0 - rs), (int)(s1 - s0)).CopyTo(tile.AsSpan((int)(s0 - v0)));
                    }
                    else TN.Store(o, k.Output, r * cols);
                }

                if (tile is not null) TN.Store(tile.AsSpan(0, (int)(v1 - v0)), k.Output, (int)v0);
            }
            finally
            {
                pool.Return(row);
                if (tile is not null) pool.Return(tile);
                if (source is not null) pool.Return(source);
                if (source2 is not null) pool.Return(source2);
                if (weights is not null) pool.Return(weights);
                if (places is not null) ArrayPool<int>.Shared.Return(places);
                input.Return();
                second?.Return();
            }
        }

        // The row methods below are methods of their own. Within the row method, as it grew, the same code ran slower:
        // col2im's loops 1.1 to 1.5 times its time before, even with the runtime's tiers off (everything compiled at the
        // final tier from the start), so the row method's size, not its tier. gather_cols, the first operation of a run, was
        // timed in its own method's first tier (two to three times slower): compiled at the final tier from the first call,
        // as the hot kernels are. The others stay in the runtime's tiers, whose last (with the profile) ran the
        // interpolation's backward 1.2 to 1.5 times faster than the final tier from the first call.

        // gather_cols, output row r: input row r (of size values) read in place or loaded once, then gathered; an index
        // outside the row selects nothing (the test against the row's own length: the read needs no bounds check of its own).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private static void GatherColsRow(Reader<TN, TLane> input, int[] idx, int size, int r, Span<TLane> o, TLane[] source)
        {
            var line = input.Span((long)r * size, size, source);
            for (int c = 0; c < o.Length; c++) { int j = idx[c]; o[c] = (uint)j < (uint)line.Length ? line[j] : default; }
        }

        // col2im, input row r (c, ih): each pixel iw the sum of every (a, b, oh, ow) that maps onto it, in kernel order.
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void Col2ImRow(Reader<TN, TLane> input, int[] p, int r, Span<TLane> o)
        {
            var (OH, OW) = LayoutArgs.ConvOut(p);
            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8], dh = p[11], dw = p[12];
            int ih = r % H, c = r / H;
            for (int iw = 0; iw < W; iw++)
            {
                TLane acc = TL.Zero;
                for (int a = 0; a < kh; a++)
                {
                    int th = ih + top - a * dh;
                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                    for (int b = 0; b < kw; b++)
                    {
                        int tw = iw + left - b * dw;
                        if (tw < 0 || tw % sw != 0 || tw / sw >= OW) continue;
                        acc = TL.Add(acc, input[((long)((c * kh + a) * kw + b) * (OH * OW)) + (th / sh) * OW + tw / sw]);
                    }
                }

                o[iw] = acc;
            }
        }

        // An interpolation's backward, input row r (c, ih): each position iw sums the outputs whose source rows and columns
        // include it (oh, then ow ascending), each dy by its weight there: acc = fma(dy, wy·wx, acc). The row's weights wy
        // are found once a row (weights and places: OH each) and the columns' wx once a task (axis and axisW, InterpolateAxis:
        // W + 1 starts, then each input column's outputs, and their weights), not again for every term: the same terms in
        // the same order.
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void InterpolateBackRow(bool bilinear, Reader<TN, TLane> input, int[] p, int r, Span<TLane> o, TLane[] weights, int[] places, int[] axis, TLane[] axisW)
        {
            int H = p[1], W = p[2], OH = p[3], OW = p[4];
            int ih = r % H, c = r / H;
            var (h0, h1) = LayoutArgs.InterpolateSources(bilinear, ih, H, OH, p[5], p[6], p[7]);
            int rows = 0;
            for (int oh = h0; oh <= h1; oh++)
                if (InterpolateWeight<TLane, TL>(bilinear, oh, ih, H, p[5], p[6], p[7], out TLane wy)) { places[rows] = oh; weights[rows++] = wy; }
            for (int iw = 0; iw < W; iw++)
            {
                int e0 = axis[iw], e1 = axis[iw + 1];
                TLane acc = TL.Zero;
                for (int j = 0; j < rows; j++)
                {
                    long dyRow = ((long)c * OH + places[j]) * OW;
                    TLane wy = weights[j];
                    for (int e = e0; e < e1; e++) acc = TL.Fma(input[dyRow + axis[W + 1 + e]], TL.Mul(wy, axisW[e]), acc);
                }

                o[iw] = acc;
            }
        }
    }
}
