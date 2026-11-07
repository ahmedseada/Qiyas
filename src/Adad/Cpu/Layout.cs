// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Layout and indexing: data movement by index arithmetic, exact in every type (the value moves as stored; the lane is
// the type's own; where values are summed — scatter-add, col2im — the sum lane by the bounds, stored as the type).
//
//   strided    a contiguous copy of a strided view of up to six dimensions: output[i0, …] = x[base + Σ i_d · stride_d]
//              (permutes, transposed views, slices, broadcasts with a zero stride); canonical first (LayoutArgs.StridedKernel)
//   one_hot    output[i][c] = (c == index[i]) ? 1 : 0 over `classes` columns
//   gather_rows   output[i] = x[index[i]] (a row each); gather_cols output[r][j] = x[r][index[j]]
//   scatter_add_rows   output[index[i]] += x[i] (the embedding gradient): every output row sums its sources in source
//              order (deterministic on both devices)
//   im2col     x (C × H × W) to columns ((C·kh·kw) × (OH·OW)) for a kernel kh × kw, strides, pads, dilations;
//   col2im     the reverse sum (each input pixel gathers its column entries in kernel order: exact order)
//   interpolate   x (C × H × W) resized to (C × OH × OW), nearest or bilinear, its source positions exact in integers
//              (half-pixel, align-corners or asymmetric coordinates, the scales given or in/out); its backward gathers
//              each input position's outputs (LayoutArgs.InterpolateSources) with their weights
//   shift      output[r][c] = x[r][c − k] inside the row, 0 outside (k may be negative)
//   pad2d      rows × cols into rows × newCols: the row's values then zeros (the operands of a product whose shape the
//              product kernels' vector loads cannot take as stored)
//   maxpool / avgpool   x (C × H × W) over kh × kw windows at strides and pads (pads excluded from a window's count unless
//              counted: LayoutArgs.PoolDivisor; ceil mode by LayoutArgs.PoolOut)
//   maxpool_bwd / avgpool_bwd   dx from dy: each input pixel gathers the windows it lies in (the first largest wins a
//              maxpool tie, recomputed: deterministic)
//   adaptive_maxpool / adaptive_avgpool   x (C × H × W) to C × OH × OW with windows [⌊i·H/OH⌋, ⌈(i + 1)·H/OH⌉)
//   depthwise_conv   y[c][oh][ow] = Σ w[c][a][b] · x[c][oh·sh − ph + a·dh][ow·sw − pw + b·dw] (the weights as a second input)

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
            var at = WindowArgmax<TN, TLane>(k);
            if (StridedTiled<TN, TLane>(k, cols, out int dim))
            {
                // A tile's line is whole storage units where the type's bytes are not lanes (a block decoded once, not once
                // a tile).
                int tile = Math.Max(16, 64 / System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>());
                if (!TN.Direct) tile = Math.Max(tile, TN.Granularity);
                var p = k.P;
                long groups = (long)p[0] * p[1] * p[2] / p[dim] * ((p[dim] + tile - 1) / tile);
                Team.For((int)groups, Math.Max(1, (int)(groups / (Threads * 4))), (long)k.Count, new StridedTileBody<TN, TLane>(k, cols, dim, tile));
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
            var at = WindowArgmax<TN, TLane>(k);
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
            var at = WindowArgmax<TN, TLane>(k);
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
            var at = WindowArgmax<TN, TLane>(k);
            Move<TN, TLane, TL>(k, rows, cols, at);
        }
    }

    private static (int Rows, int Cols) OutputRows(LayoutArgs k) => LayoutArgs.OutputRows(k);

    // A bilinear axis's weights in the lane: f = rem / δ (one rounding), g = 1 − f.
    private static (TLane F, TLane G) Weights<TLane, TL>(int rem, int delta) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        var f = TL.Div(new Vector<TLane>(TL.From(rem)), new Vector<TLane>(TL.From(delta)));
        return (f[0], TL.Sub(new Vector<TLane>(TL.From(1)), f)[0]);
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

    /// <summary>A strided copy whose output rows run across the input (the last dimension's stride not 1) while one of the
    /// three outer dimensions has stride 1 (a transposition): taken by tiles of output rows along that dimension (see
    /// StridedTileBody). Rows ending on whole bytes only (each row stored by one thread), values read in place or through
    /// the task's reader.</summary>
    private static bool StridedTiled<TN, TLane>(LayoutArgs k, int cols, out int dim) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        dim = -1;
        if (k.Name != "strided" || k.P.Length != 9 || k.P[7] == 1 || (long)cols * TN.Bits % 8 != 0) return false;   // the four-dimensional form
        for (int j = 2; j >= 0; j--) if (k.P[4 + j] == 1 && k.P[j] > 1) { dim = j; return true; }
        return false;
    }

    /// <summary>The tiled strided copy: a group is `tile` output rows that differ only in dimension `dim` (whose input
    /// stride is 1) at one index of the other two; for each output column the group's values lie side by side in the input
    /// (a cache line read once, not a line a value), gathered into the group's rows (a tile × cols block in lane storage),
    /// each row then stored whole. Groups are disjoint sets of whole rows.</summary>
    private struct StridedTileBody<TN, TLane>(LayoutArgs k, int cols, int dim, int tile) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var p = k.P;
            int d0 = p[0], d1 = p[1], d2 = p[2], dj = p[dim], blocks = (dj + tile - 1) / tile;
            long s7 = p[7];
            ReadOnlySpan<TLane> src = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(k.X.AsSpan()) : default;
            var input = TN.Direct ? null : new Reader<TN, TLane>(k.X, k.InputCount);
            var lineBuffer = TN.Direct ? null : ArrayPool<TLane>.Shared.Rent(tile);
            var buffer = ArrayPool<TLane>.Shared.Rent(tile * cols);
            try
            {
                for (int g = first; g < last; g++)
                {
                    // g = outer · blocks + block; outer enumerates the two other dimensions (in their order).
                    int block = g % blocks, outer = g / blocks;
                    int lo = block * tile, count = Math.Min(tile, dj - lo);
                    int i0, i1, i2;
                    if (dim == 0) { i2 = outer % d2; i1 = outer / d2; i0 = lo; }
                    else if (dim == 1) { i2 = outer % d2; i0 = outer / d2; i1 = lo; }
                    else { i1 = outer % d1; i0 = outer / d1; i2 = lo; }
                    long b0 = p[8] + (long)i0 * p[4] + (long)i1 * p[5] + (long)i2 * p[6];  // value lo of the group (stride 1 in t)
                    var t2 = buffer.AsSpan(0, tile * cols);
                    for (int c = 0; c < cols; c++)
                    {
                        var line = input is null ? src.Slice((int)(b0 + c * s7), count) : input.Span(b0 + c * s7, count, lineBuffer!);
                        for (int t = 0; t < count; t++) t2[t * cols + c] = line[t];
                    }

                    for (int t = 0; t < count; t++)
                    {
                        int r0 = dim == 0 ? i0 + t : i0, r1 = dim == 1 ? i1 + t : i1, r2 = dim == 2 ? i2 + t : i2;
                        long r = ((long)r0 * d1 + r1) * d2 + r2;
                        TN.Store(t2.Slice(t * cols, cols), k.Output, (int)(r * cols));
                    }
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

    /// <summary>The max-pool backward's first largest of every window of the forward input (x2), found once by the team
    /// (index into the input, -1 for an empty window): each input row then reads its windows' choice instead of scanning
    /// every window it lies in. Null for the other operations.</summary>
    private static long[]? WindowArgmax<TN, TLane>(LayoutArgs k) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
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
                // The first largest in row-major order: a later value takes it only when strictly larger.
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
        // at: the max-pool backward's first largest of every window (input index, -1 for an empty window), found once.
        // v0 ≥ 0: only the output values [v0, v1) are this task's (whole storage units), gathered and stored at once.
        private static void RunRows(int first, int last, LayoutArgs k, int rows, int cols, bool exclusiveBytes, long[]? at, bool inLanes, bool outLanes, long v0 = -1, long v1 = -1)
        {
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(cols);
            var tile = v0 >= 0 ? pool.Rent((int)(v1 - v0)) : null;
            TLane[]? source = null, source2 = null;
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
                            // Row r over the outer dimensions (three, or five of a six-dimensional view), the row along the last.
                            int rank = (p.Length - 1) / 2;
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
                            TN.Load(k.X, idx![r] * cols, o);
                            break;
                        case "gather_cols":
                        {
                            // The input row read in place or loaded once, then gathered.
                            source ??= pool.Rent(p[1]);
                            var line = input.Span((long)r * p[1], p[1], source);
                            for (int c = 0; c < cols; c++) o[c] = line[idx![c]];
                            break;
                        }
                        case "scatter_add_rows":
                        {
                            // Output row r: the sum of every source row whose index is r, in source order; a source row read
                            // whole (one load of its values: a block format decodes each block once, not once per value).
                            o.Clear();
                            for (int i = 0; i < p[0]; i++)
                            {
                                if (idx![i] != r) continue;
                                source ??= pool.Rent(cols);
                                var s = input.Span((long)i * cols, cols, source);

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
                                int ih = oh * p[5] - p[7] + a * p[9];
                                for (int ow = 0; ow < OW; ow++)
                                {
                                    int iw = ow * p[6] - p[8] + b * p[10];
                                    o[oh * OW + ow] = (uint)ih < (uint)H && (uint)iw < (uint)W ? input[((long)c * H + ih) * W + iw] : TL.Zero;
                                }
                            }

                            break;
                        }

                        case "col2im":
                        {
                            // Input pixel (c, ih, iw = col): every (a, b, oh, ow) that maps onto it, in kernel order.
                            var (OH, OW) = LayoutArgs.ConvOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                            int ih = r % H, c = r / H;
                            for (int iw = 0; iw < W; iw++)
                            {
                                TLane acc = TL.Zero;
                                for (int a = 0; a < kh; a++)
                                {
                                    int th = ih + ph - a * dh;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int tw = iw + pw - b * dw;
                                        if (tw < 0 || tw % sw != 0 || tw / sw >= OW) continue;
                                        acc = TL.Add(acc, input[((long)((c * kh + a) * kw + b) * (OH * OW)) + (th / sh) * OW + tw / sw]);
                                    }
                                }

                                o[iw] = acc;
                            }

                            break;
                        }

                        case "interpolate_nearest":
                        {
                            // p: C H W OH OW αh βh δh αw βw δw (LayoutArgs.InterpolateAxis).
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int oh = r % OH, c = r / OH;
                            int ih = LayoutArgs.NearestAt(oh, H, p[5], p[6], p[7]);
                            source ??= pool.Rent(W);
                            var line = input.Span(((long)c * H + ih) * W, W, source);
                            for (int ow = 0; ow < OW; ow++) o[ow] = line[LayoutArgs.NearestAt(ow, W, p[8], p[9], p[10])];
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
                            for (int ow = 0; ow < OW; ow++)
                            {
                                var (x0, x1, rx) = LayoutArgs.BilinearAt(ow, W, p[8], p[9], p[10]);
                                var (fx, gx) = Weights<TLane, TL>(rx, p[10]);
                                TLane top = TL.Fma(row0[x1], fx, TL.Mul(row0[x0], gx));
                                TLane bottom = TL.Fma(row1[x1], fx, TL.Mul(row1[x0], gx));
                                o[ow] = TL.Fma(bottom, fy, TL.Mul(top, gy));
                            }

                            break;
                        }

                        case "interpolate_nearest_bwd":
                        case "interpolate_bilinear_bwd":
                        {
                            // Input position (c, ih, iw = col): the outputs whose source rows and columns include it (oh, then ow
                            // ascending), each dy by its weight there: acc = fma(dy, wy·wx, acc).
                            int H = p[1], W = p[2], OH = p[3], OW = p[4];
                            int ih = r % H, c = r / H;
                            bool bilinear = k.Name == "interpolate_bilinear_bwd";
                            var (h0, h1) = LayoutArgs.InterpolateSources(bilinear, ih, H, OH, p[5], p[6], p[7]);
                            for (int iw = 0; iw < W; iw++)
                            {
                                var (w0, w1) = LayoutArgs.InterpolateSources(bilinear, iw, W, OW, p[8], p[9], p[10]);
                                TLane acc = TL.Zero;
                                for (int oh = h0; oh <= h1; oh++)
                                {
                                    if (!InterpolateWeight<TLane, TL>(bilinear, oh, ih, H, p[5], p[6], p[7], out TLane wy)) continue;
                                    for (int ow = w0; ow <= w1; ow++)
                                    {
                                        if (!InterpolateWeight<TLane, TL>(bilinear, ow, iw, W, p[8], p[9], p[10], out TLane wx)) continue;
                                        acc = TL.Fma(input[((long)c * OH + oh) * OW + ow], TL.Mul(wy, wx), acc);
                                    }
                                }

                                o[iw] = acc;
                            }

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
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                            int oh = r % OH, c = r / OH;
                            bool max = k.Name == "maxpool";
                            for (int ow = 0; ow < OW; ow++)
                            {
                                TLane acc = max ? TL.From(double.NegativeInfinity) : TL.Zero;
                                int n = 0;
                                for (int a = 0; a < kh; a++)
                                {
                                    int ih = oh * sh - ph + a;
                                    if ((uint)ih >= (uint)H) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int iw = ow * sw - pw + b;
                                        if ((uint)iw >= (uint)W) continue;
                                        TLane v = input[((long)c * H + ih) * W + iw];
                                        acc = max ? (n == 0 || Widths.Compare(v, acc) > 0 ? v : acc) : TL.Add(acc, v);
                                        n++;
                                    }
                                }

                                o[ow] = n == 0 ? TL.Zero : max ? acc : TL.Mul(acc, TL.From(1.0 / LayoutArgs.PoolDivisor(p, oh, ow, n)));
                            }

                            break;
                        }

                        case "maxpool_bwd":
                        case "avgpool_bwd":
                        {
                            // Input pixel (c, ih, iw = col): the windows it lies in; maxpool: those whose first largest it is.
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
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
                                    int th = ih + ph - a;
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
                                    int th = ih + ph - a;
                                    if (th < 0 || th % sh != 0 || th / sh >= OH) continue;
                                    int oh = th / sh;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int tw = iw + pw - b;
                                        if (tw < 0 || tw % sw != 0 || tw / sw >= OW) continue;
                                        int ow = tw / sw;
                                        TLane dy = input[((long)c * OH + oh) * OW + ow];
                                        if (!max)
                                        {
                                            int n = 0;
                                            for (int aa = 0; aa < kh; aa++) { int hh = oh * sh - ph + aa; if ((uint)hh >= (uint)H) continue; for (int bb = 0; bb < kw; bb++) { int ww = ow * sw - pw + bb; if ((uint)ww < (uint)W) n++; } }
                                            acc = TL.Add(acc, TL.Mul(dy, TL.From(1.0 / LayoutArgs.PoolDivisor(p, oh, ow, n))));
                                            continue;
                                        }

                                        // The window's first largest: this pixel takes dy when it is that one.
                                        long bestAt = -1;
                                        TLane best = TL.From(double.NegativeInfinity);
                                        for (int aa = 0; aa < kh; aa++)
                                        {
                                            int hh = oh * sh - ph + aa;
                                            if ((uint)hh >= (uint)H) continue;
                                            for (int bb = 0; bb < kw; bb++)
                                            {
                                                int ww = ow * sw - pw + bb;
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

                                o[ow] = max ? acc : TL.Mul(acc, TL.From(1.0 / ((h1 - h0) * (w1 - w0))));
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
                                        TLane share = TL.Mul(input[((long)c * OH + oh) * OW + ow], TL.From(1.0 / ((h1 - h0) * (w1 - w0))));
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
                                        if (!max) { acc = TL.Add(acc, TL.Mul(dy, TL.From(1.0 / ((h1 - h0) * (w1 - w0))))); continue; }
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
                            var (OH, OW) = LayoutArgs.PoolOut(p);
                            int H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
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
                                    int ih = oh * sh - ph + a * dh;
                                    if ((uint)ih >= (uint)H) continue;
                                    long rowAt = ((long)c * H + ih) * W;
                                    var line = input.Span(rowAt, W, source);
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int off = b * dw - pw, lo = Math.Max(0, -off), hi = Math.Min(OW, W - off);
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
                                    int ih = oh * sh - ph + a * dh;
                                    if ((uint)ih >= (uint)H) continue;
                                    for (int b = 0; b < kw; b++)
                                    {
                                        int iw = ow * sw - pw + b * dw;
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
                input.Return();
                second?.Return();
            }
        }
    }
}
