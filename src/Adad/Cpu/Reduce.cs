// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Reductions, row-vector broadcasts, transposes and matrix-vector products on the CPU, written once over the contract.
//
//   Reduce   sum, sumsq, max, min, argmax over a whole array, each row, or each column. Sums take the number's sum
//            lane for the bounds measured on the data (a dot lane sums packed groups against ones: the processor's
//            widening multiply-add is its fastest adder for 8- and 16-bit values); max/min/argmax take the type's own
//            lane and Adad's order (Routing.Order: a NaN above every number, −0 equal to +0): max the first-ranked value,
//            min the last-ranked, argmax the first position of the first-ranked. The result is lane values (the lane is
//            reported): exact by the rule, the caller converts as it likes.
//   MapRow   output[r][c] = op(x[r][c], v[c]) for the two-input formulas (a row vector broadcast down the rows).
//   Transpose
//   Gemv     y = A·x and y = Aᵀ·x in the sum lane, the result stored as the type.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    // ---------------------------------------------------------------- reductions

    /// <summary>Reduces x (rows × cols) along the axis: the result as lane values (count 1, rows, or cols; argmax as int32 indices) and the lane's name.</summary>
    public static (byte[] Result, string Lane) Reduce<TN>(string op, Axis axis, byte[] x, int rows, int cols) where TN : struct, IAdad<TN>
    {
        if (op is "max" or "min" or "argmax")
        {
            Reductions.ValidateExtreme(op, axis, rows, cols);
            var cmp = new CompareVisitor(op, axis, x, rows, cols);
            TN.Elementwise(ref cmp);
            return (cmp.Result!, op == "argmax" ? "s32 index" : cmp.Lane);
        }

        // A float type's sum lane does not depend on the values: its range is not read (a block format's decodes every block).
        // An integer type's range is read only where it can change the lane (item 6j): where the type's own bounds take
        // the lane the narrowest values would, every range takes it, and the sum is exact in it either way.
        long terms = axis switch { Axis.Whole => (long)rows * cols, Axis.Rows => cols, _ => rows };
        var (lo, hi) = TN.IsFloat ? (0, 0) : !RangeMatters<TN>(op == "sumsq", terms) ? (TN.Min, TN.Max) : TN.Range(x, rows * cols);
        var bounds = op == "sumsq" ? new Bounds(lo, hi, lo, hi, terms) : new Bounds(lo, hi, 1, 1, terms);
        var sum = new SumVisitor(op == "sumsq", axis, x, rows, cols);
        TN.Sum(ref sum, in bounds);
        return (sum.Result!, sum.Lane);
    }

    // Whether an integer sum's lane can depend on the values' range: the lane the type's own bounds take against the lane
    // the narrowest values (all 0) take.
    private static bool RangeMatters<TN>(bool squares, long terms) where TN : struct, IAdad<TN>
    {
        var wide = squares ? new Bounds(TN.Min, TN.Max, TN.Min, TN.Max, terms) : new Bounds(TN.Min, TN.Max, 1, 1, terms);
        var narrow = squares ? new Bounds(0, 0, 0, 0, terms) : new Bounds(0, 0, 1, 1, terms);
        var a = new LaneProbe();
        TN.Sum(ref a, in wide);
        var b = new LaneProbe();
        TN.Sum(ref b, in narrow);
        return a.Lane != b.Lane;
    }

    // The lane a sum's bounds take, nothing run.
    private struct LaneProbe : ISumVisitor
    {
        public string Lane;

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> => Lane = TL.Name;

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc> => Lane = TD.Name;
    }

    private struct SumVisitor(bool squares, Axis axis, byte[] x, int rows, int cols) : ISumVisitor
    {
        public byte[]? Result;
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            Result = SumIn<TN, TLane, TLane, TL>(squares, axis, x, rows, cols);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            Lane = TD.Name;
            Result = SumIn<TN, TIn, TAcc, TD>(squares, axis, x, rows, cols);
        }
    }

    // The sum in lane TL over inputs TIn packed into TAcc words (a plain lane: TIn = TAcc, one value a word).
    private static byte[] SumIn<TN, TIn, TAcc, TL>(bool squares, Axis axis, byte[] x, int rows, int cols)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>();
        if (axis == Axis.Whole) { cols = rows * cols; rows = 1; }
        int count = axis == Axis.Cols ? cols : rows;
        var result = new byte[count * Unsafe.SizeOf<TAcc>()];
        var outLanes = MemoryMarshal.Cast<byte, TAcc>(result.AsSpan());
        if (axis == Axis.Cols)
        {
            // Tiles of a strip of columns by a block of rows (ColumnPlan), g rows at a time (one word of g values per
            // column); the blocks' sums joined in block order.
            var plan = ColumnPlan.Of<TAcc>(rows, cols, g);
            var dst = plan.Blocks == 1 ? result : plan.Partials<TAcc>(cols);
            Team.For(plan.Tasks, 1, (long)rows * cols, new ColumnTiles<ColumnSumTile<TN, TIn, TAcc, TL>>(new(x, dst, cols, g, squares), plan, rows, cols));
            if (plan.Blocks > 1) Join<TAcc, TL>(dst, plan.Blocks, cols, result, JoinOp.Add);
            return result;
        }

        // Rows across the team (the whole array is one row): each row's words against ones (or themselves).
        var rowBody = new RowSumBody<TN, TIn, TAcc, TL>(x, result, cols, g, squares);
        int rowChunk = Math.Max(1, rows / (Threads * 4));
        if (axis == Axis.Whole)
        {
            // One long row: split across the team into partial sums, then joined in the lane.
            // Parts of whole words (a multiple of g values), as many as cover every value: rounding a part down to words
            // and keeping the part count would leave the last values out.
            var partials = WholeParts<TN, TIn, TAcc, TL>(squares, x, cols);
            var p = MemoryMarshal.Cast<byte, TAcc>(partials.AsSpan());
            TAcc total = TL.Zero;
            foreach (var v in p) total = TL.Add(total, v);
            outLanes[0] = total;
            return result;
        }

        Team.For(rows, rowChunk, (long)rows * cols, rowBody);
        return result;
    }

    // A whole array's sum (or sum of squares) as its parts' sums, lane values in order: n values split across the team into
    // parts of whole words (a multiple of g values), as many as cover every value. Joined in order they are the whole sum;
    // several arrays' parts joined in one chain are one sum across them (Optimizers.cs).
    internal static byte[] WholeParts<TN, TIn, TAcc, TL>(bool squares, byte[] x, int n)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>();
        var (parts, per) = WholePlan(n, g);
        var partials = new byte[parts * Unsafe.SizeOf<TAcc>()];
        Team.For(parts, 1, (long)n, new PartBody<TN, TIn, TAcc, TL>(x, partials, n, per, g, squares));
        return partials;
    }

    // The parts of a whole sum over n values in words of g: how many, and the values of each.
    internal static (int Parts, int Per) WholePlan(int n, int g)
    {
        int parts = Math.Min(Threads * 4, Math.Max(1, n / 4096));
        int per = ((n + parts - 1) / parts + g - 1) / g * g;
        return (Math.Max(1, (n + per - 1) / per), per);
    }

    // Several arrays' whole sums of squares as one run of the team: job i is part i − first[t] of array t, each array's
    // parts as WholeParts makes them (into its own partials), so the arrays' parts joined in order are one sum across them.
    internal readonly struct ManyPartsBody<TN, TIn, TAcc, TL>(byte[][] x, byte[][] partials, int[] n, int[] per, int[] first, int g) : Team.IBody
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        public void Run(int from, int to)
        {
            int t = 0;
            while (from < to)
            {
                while (first[t + 1] <= from) t++;
                int end = Math.Min(to, first[t + 1]);
                new PartBody<TN, TIn, TAcc, TL>(x[t], partials[t], n[t], per[t], g, true).Run(from - first[t], end - first[t]);
                from = end;
            }
        }
    }

    // Σ over one row's values [first, first + n): words of g inputs each, Fma against ones (sum) or themselves (squares).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static TAcc DotRow<TN, TIn, TAcc, TL>(byte[] x, int first, int n, int g, bool squares, Span<TIn> scratch)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int w = Vector<TAcc>.Count;
        var ones = new Vector<TAcc>(Ones<TIn, TAcc>());
        Vector<TAcc> acc0 = Vector<TAcc>.Zero, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        TAcc tail = TL.Zero;
        int per = TN.Granularity;
        bool direct = Lanes.Direct<TN, TIn>() && (first % per == 0);
        for (int at = 0; at < n;)
        {
            int len = Math.Min(scratch.Length, n - at);
            ReadOnlySpan<TIn> values;
            if (direct) values = MemoryMarshal.Cast<byte, TIn>(x.AsSpan()).Slice(first + at, len);
            else
            {
                TN.Load(x, first + at, scratch[..len]);
                values = scratch[..len];
            }

            int whole = len / g * g;
            var words = MemoryMarshal.Cast<TIn, TAcc>(values[..whole]);
            ref TAcc r = ref MemoryMarshal.GetReference(words);
            int i = 0;
            for (; i + 4 * w <= words.Length; i += 4 * w)
            {
                var v0 = Vector.LoadUnsafe(ref r, (nuint)i); var v1 = Vector.LoadUnsafe(ref r, (nuint)(i + w));
                var v2 = Vector.LoadUnsafe(ref r, (nuint)(i + 2 * w)); var v3 = Vector.LoadUnsafe(ref r, (nuint)(i + 3 * w));
                acc0 = TL.Fma(v0, squares ? v0 : ones, acc0); acc1 = TL.Fma(v1, squares ? v1 : ones, acc1);
                acc2 = TL.Fma(v2, squares ? v2 : ones, acc2); acc3 = TL.Fma(v3, squares ? v3 : ones, acc3);
            }

            for (; i + w <= words.Length; i += w) { var v = Vector.LoadUnsafe(ref r, (nuint)i); acc0 = TL.Fma(v, squares ? v : ones, acc0); }
            for (; i < words.Length; i++) tail = TL.Fma(words[i], squares ? words[i] : ones[0], tail);
            // Values past the last whole word (g > 1 only): widened one by one.
            for (int t = whole; t < len; t++)
            {
                TAcc v = Widths.FromLong<TAcc>(Widths.AsLong(values[t]));
                tail = TL.Add(tail, squares ? Plain<TAcc>.Mul(v, v) : v);
            }

            at += len;
        }

        return TL.Add(tail, Vector.Sum(acc0 + acc1 + acc2 + acc3));
    }

    // A word of g ones (1 in each TIn slot): for a plain lane the lane's 1.
    private static TAcc Ones<TIn, TAcc>() where TIn : unmanaged where TAcc : unmanaged
    {
        if (typeof(TAcc) == typeof(float)) return (TAcc)(object)1f;
        if (typeof(TAcc) == typeof(double)) return (TAcc)(object)1.0;
        if (typeof(TIn) == typeof(TAcc)) return Widths.FromLong<TAcc>(1);
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>(), bits = 8 * Unsafe.SizeOf<TIn>();
        long word = 0;
        for (int t = 0; t < g; t++) word |= 1L << (bits * t);
        return Widths.FromLong<TAcc>(word);
    }

    // Scalar arithmetic in a lane type without a lane struct at hand (the tail of a dot lane's row): integers only reach here.
    private static class Plain<T> where T : unmanaged
    {
        public static T Mul(T a, T b)
        {
            if (typeof(T) == typeof(float)) return (T)(object)((float)(object)a * (float)(object)b);
            if (typeof(T) == typeof(double)) return (T)(object)((double)(object)a * (double)(object)b);
            return Widths.FromLong<T>(Widths.AsLong(a) * Widths.AsLong(b));
        }
    }

    private struct RowSumBody<TN, TIn, TAcc, TL>(byte[] x, byte[] result, int cols, int g, bool squares) : Team.IBody
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        public readonly void Run(int first, int last)
        {
            var outLanes = MemoryMarshal.Cast<byte, TAcc>(result.AsSpan());
            Span<TIn> scratch = stackalloc TIn[4096];
            for (int r = first; r < last; r++) outLanes[r] = DotRow<TN, TIn, TAcc, TL>(x, r * cols, cols, g, squares, scratch);
        }
    }

    private struct PartBody<TN, TIn, TAcc, TL>(byte[] x, byte[] partials, int n, int per, int g, bool squares) : Team.IBody
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        public readonly void Run(int first, int last)
        {
            var outLanes = MemoryMarshal.Cast<byte, TAcc>(partials.AsSpan());
            Span<TIn> scratch = stackalloc TIn[4096];
            for (int p = first; p < last; p++)
            {
                int start = p * per, len = Math.Max(0, Math.Min(per, n - start));
                outLanes[p] = len == 0 ? TL.Zero : DotRow<TN, TIn, TAcc, TL>(x, start, len, g, squares, scratch);
            }
        }
    }

    // ---------------------------------------------------------------- column passes (6h)

    /// <summary>A pass down the columns as tiles: a strip of columns (about 8 KB of accumulators, in L1, each row's part
    /// read as one contiguous run) by a block of rows (whole words of g rows), at least four tiles a thread when the
    /// shape allows (cores of unequal speed balance through the team's counter); each block's results in its own slot,
    /// joined in block order (the same result whatever the schedule).</summary>
    private readonly record struct ColumnPlan(int Strip, int Strips, int RowBlock, int Blocks)
    {
        public int Tasks => Strips * Blocks;

        public static ColumnPlan Of<TAcc>(int rows, int cols, int g) where TAcc : unmanaged
        {
            int w = Vector<TAcc>.Count;
            int strip = Math.Max(1, Math.Min(cols, Math.Max(4 * w, 8192 / Unsafe.SizeOf<TAcc>() / w * w)));
            int strips = (cols + strip - 1) / strip;
            int minRows = Math.Max(16, g) / g * g;                              // a block at least 16 rows (partials ≤ 1/16 of the input)
            int blocks = Math.Clamp((Threads * 4 + strips - 1) / Math.Max(1, strips), 1, Math.Max(1, (rows + minRows - 1) / minRows));
            int rowBlock = Math.Max(g, ((rows + blocks - 1) / blocks + g - 1) / g * g);
            blocks = Math.Max(1, (rows + rowBlock - 1) / rowBlock);
            return new ColumnPlan(strip, strips, rowBlock, blocks);
        }

        /// <summary>Blocks × cols lanes for the blocks' results (every value written by its tile).</summary>
        public byte[] Partials<TAcc>(int cols) where TAcc : unmanaged => new byte[checked((long)Blocks * cols * Unsafe.SizeOf<TAcc>())];
    }

    private interface IColumnTile
    {
        /// <summary>Rows [r0, r1) of columns [c0, c1) into the block's slot (values block·cols + c0 ..).</summary>
        void Tile(int r0, int r1, int c0, int c1, int block);
    }

    private struct ColumnTiles<TB>(TB tile, ColumnPlan plan, int rows, int cols) : Team.IBody where TB : struct, IColumnTile
    {
        public readonly void Run(int first, int last)
        {
            for (int t = first; t < last; t++)
            {
                int s = t % plan.Strips, b = t / plan.Strips, c0 = s * plan.Strip, r0 = b * plan.RowBlock;
                tile.Tile(r0, Math.Min(rows, r0 + plan.RowBlock), c0, Math.Min(cols, c0 + plan.Strip), b);
            }
        }
    }

    private enum JoinOp { Add, Max, Min }

    // The blocks' results joined in block order into dst (cols lanes): the sums added, the extremes compared as the rows were.
    private static void Join<TAcc, TL>(byte[] partials, int blocks, int cols, byte[] dst, JoinOp op) where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int w = Vector<TAcc>.Count, chunk = Math.Max(16 * w, cols / (Threads * 2) / w * w);
        Team.For(cols, chunk, (long)blocks * cols, new JoinBody<TAcc, TL>(partials, blocks, cols, dst, op));
    }

    private struct JoinBody<TAcc, TL>(byte[] partials, int blocks, int cols, byte[] dst, JoinOp op) : Team.IBody
        where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            int w = Vector<TAcc>.Count;
            var p = MemoryMarshal.Cast<byte, TAcc>(partials.AsSpan());
            var o = MemoryMarshal.Cast<byte, TAcc>(dst.AsSpan());
            ref TAcc rp = ref MemoryMarshal.GetReference(p);
            ref TAcc ro = ref MemoryMarshal.GetReference(o);
            int i = first;
            for (; i + w <= last; i += w)
            {
                var v = Vector.LoadUnsafe(ref rp, (nuint)i);
                for (int b = 1; b < blocks; b++)
                {
                    var u = Vector.LoadUnsafe(ref rp, (nuint)((long)b * cols + i));
                    v = op == JoinOp.Add ? TL.Add(v, u) : op == JoinOp.Max ? First(v, u) : Last(v, u);
                }

                v.StoreUnsafe(ref ro, (nuint)i);
            }

            for (; i < last; i++)
            {
                TAcc v = p[i];
                for (int b = 1; b < blocks; b++)
                {
                    TAcc u = p[b * cols + i];
                    v = op == JoinOp.Add ? TL.Add(v, u) : op == JoinOp.Max ? (Widths.Compare(u, v) > 0 ? u : v) : (Widths.Compare(u, v) < 0 ? u : v);
                }

                o[i] = v;
            }
        }
    }

    // Column sums (or sums of squares) of a tile: g rows a word (loaded and interleaved), or in place where the stored
    // values are the lane's (g = 1).
    private readonly struct ColumnSumTile<TN, TIn, TAcc, TL>(byte[] x, byte[] dst, int cols, int g, bool squares) : IColumnTile
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Tile(int r0, int r1, int c0, int c1, int block)
        {
            int n = c1 - c0, w = Vector<TAcc>.Count;
            var acc = MemoryMarshal.Cast<byte, TAcc>(dst.AsSpan()).Slice(block * cols + c0, n);
            acc.Clear();
            var ones = new Vector<TAcc>(Ones<TIn, TAcc>());
            bool direct = g == 1 && Lanes.Direct<TN, TIn>();
            var rowsIn = direct ? null : ArrayPool<TIn>.Shared.Rent(g * n);
            var units = direct ? null : ArrayPool<TAcc>.Shared.Rent(n);
            ref TAcc ra = ref MemoryMarshal.GetReference(acc);
            for (int r = r0; r < r1; r += g)
            {
                ReadOnlySpan<TAcc> u;
                if (direct) u = MemoryMarshal.Cast<byte, TAcc>(x.AsSpan()).Slice(r * cols + c0, n);
                else
                {
                    int have = Math.Min(g, r1 - r);
                    for (int t = 0; t < g; t++)
                    {
                        var d = rowsIn.AsSpan(t * n, n);
                        if (t < have) TN.Load(x, (r + t) * cols + c0, d); else d.Clear();
                    }

                    Interleave<TIn, TAcc>(rowsIn.AsSpan(0, g * n), units.AsSpan(0, n), n);
                    u = units.AsSpan(0, n);
                }

                ref TAcc ru = ref MemoryMarshal.GetReference(u);
                int i = 0;
                for (; i + w <= n; i += w)
                {
                    var v = Vector.LoadUnsafe(ref ru, (nuint)i);
                    TL.Fma(v, squares ? v : ones, Vector.LoadUnsafe(ref ra, (nuint)i)).StoreUnsafe(ref ra, (nuint)i);
                }

                for (; i < n; i++) acc[i] = TL.Fma(u[i], squares ? u[i] : ones[0], acc[i]);
            }

            if (rowsIn is not null) ArrayPool<TIn>.Shared.Return(rowsIn);
            if (units is not null) ArrayPool<TAcc>.Shared.Return(units);
        }
    }

    // ---------------------------------------------------------------- max, min, argmax in the type's own lane

    // Adad's order on lane vectors, lane by lane (Widths.Compare's on values): the first-ranked of each pair, a NaN when
    // either is one (IEEE maximum, which propagates NaN), and the last-ranked, the number when one is a NaN (IEEE
    // minimumNumber); an integer lane's max and min. Of −0 and +0, which rank the same, either may be the result.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> First<T>(Vector<T> a, Vector<T> b) => Vector.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> Last<T>(Vector<T> a, Vector<T> b) => Vector.MinNumber(a, b);

    private struct CompareVisitor(string op, Axis axis, byte[] x, int rows, int cols) : ILaneVisitor
    {
        public byte[]? Result;
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            bool max = op != "min";
            int r = rows, c = cols;
            if (axis == Axis.Whole) { r = 1; c = rows * cols; }
            if (axis == Axis.Cols && op == "argmax")
            {
                // Tiles (ColumnPlan): each block's first largest and its row, then the blocks in order (a later block's
                // only when strictly larger: the first row holding the column's largest, as the rows read in order).
                Result = new byte[c * 4];
                Lane = "s32 index";
                var plan = ColumnPlan.Of<TLane>(r, c, 1);
                var values = plan.Partials<TLane>(c);
                var rowsAt = plan.Partials<int>(c);
                Team.For(plan.Tasks, 1, (long)r * c, new ColumnTiles<ColumnArgmaxTile<TN, TLane, TL>>(new(x, values, rowsAt, c), plan, r, c));
                Team.For(c, Math.Max(256, c / (Threads * 2)), (long)plan.Blocks * c, new ArgmaxJoinBody<TLane>(values, rowsAt, plan.Blocks, c, Result));
                return;
            }

            if (axis == Axis.Cols)
            {
                // Tiles (ColumnPlan), the blocks' extremes joined in block order.
                Result = new byte[c * Unsafe.SizeOf<TLane>()];
                var plan = ColumnPlan.Of<TLane>(r, c, 1);
                var dst = plan.Blocks == 1 ? Result : plan.Partials<TLane>(c);
                Team.For(plan.Tasks, 1, (long)r * c, new ColumnTiles<ColumnCompareTile<TN, TLane, TL>>(new(x, dst, c, max), plan, r, c));
                if (plan.Blocks > 1) Join<TLane, TL>(dst, plan.Blocks, c, Result, max ? JoinOp.Max : JoinOp.Min);
                return;
            }

            if (axis == Axis.Whole)
            {
                // Parts across the team (each a run of whole bytes), then the parts' extremes joined (the first index wins a tie).
                int per = TN.Granularity;
                int parts = Math.Min(Threads * 4, Math.Max(1, c / 4096)), span = ((c + parts - 1) / parts + per - 1) / per * per;
                parts = (c + span - 1) / span;
                var values = new byte[parts * Unsafe.SizeOf<TLane>()];
                var indices = new byte[parts * 4];
                Team.For(parts, 1, (long)c, new PartCompareBody<TN, TLane, TL>(x, values, indices, c, span, max, op == "argmax"));
                var pv = MemoryMarshal.Cast<byte, TLane>(values.AsSpan());
                var pi = MemoryMarshal.Cast<byte, int>(indices.AsSpan());
                int bestPart = 0;
                for (int p = 1; p < parts; p++)
                {
                    if (pi[p] < 0) continue;
                    int cmp = Widths.Compare(pv[p], pv[bestPart]);
                    if (pi[bestPart] < 0 || (max ? cmp > 0 : cmp < 0)) bestPart = p;
                }

                if (op == "argmax") { Result = new byte[4]; BitConverter.TryWriteBytes(Result, pi[bestPart]); }
                else { Result = new byte[Unsafe.SizeOf<TLane>()]; MemoryMarshal.Write(Result, in pv[bestPart]); }
                return;
            }

            if (op == "argmax")
            {
                Result = new byte[r * 4];
                Team.For(r, Math.Max(1, r / (Threads * 4)), (long)r * c, new ArgmaxBody<TN, TLane, TL>(x, Result, c));
                return;
            }

            Result = new byte[r * Unsafe.SizeOf<TLane>()];
            Team.For(r, Math.Max(1, r / (Threads * 4)), (long)r * c, new RowCompareBody<TN, TLane, TL>(x, Result, c, max));
        }
    }

    // The extreme of x[first, first + n) and the first position holding it; wantIndex false (item 6j: max / min, whose
    // result is the value): no second pass for the position (index 0 when there is a value, −1 when n is 0).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLane Extreme<TN, TLane, TL>(byte[] x, int first, int n, bool max, Span<TLane> scratch, out int index, bool wantIndex = true)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int w = Vector<TLane>.Count;
        TLane best = default;
        bool have = false;
        index = -1;
        ReadOnlySpan<TLane> direct = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(first, n) : default;
        for (int at = 0; at < n;)
        {
            int len = TN.Direct ? n : Math.Min(scratch.Length, n - at);
            ReadOnlySpan<TLane> s;
            if (TN.Direct) s = direct;
            else
            {
                TN.Load(x, first + at, scratch[..len]);
                s = scratch[..len];
            }
            int i = 0;
            if (len >= w)
            {
                // Four chains of vectors (item 6j), joined lane by lane: each lane still the extreme of the same positions
                // (First / Last associative and commutative in the order; integers exact).
                ref TLane rs = ref MemoryMarshal.GetReference(s);
                var acc = Vector.LoadUnsafe(ref rs);
                i = w;
                if (len >= 4 * w)
                {
                    Vector<TLane> a1 = Vector.LoadUnsafe(ref rs, (nuint)w), a2 = Vector.LoadUnsafe(ref rs, (nuint)(2 * w)), a3 = Vector.LoadUnsafe(ref rs, (nuint)(3 * w));
                    for (i = 4 * w; i + 4 * w <= len; i += 4 * w)
                    {
                        nuint at2 = (nuint)i;
                        var v0 = Vector.LoadUnsafe(ref rs, at2); var v1 = Vector.LoadUnsafe(ref rs, at2 + (nuint)w);
                        var v2 = Vector.LoadUnsafe(ref rs, at2 + (nuint)(2 * w)); var v3 = Vector.LoadUnsafe(ref rs, at2 + (nuint)(3 * w));
                        if (max) { acc = First(acc, v0); a1 = First(a1, v1); a2 = First(a2, v2); a3 = First(a3, v3); }
                        else { acc = Last(acc, v0); a1 = Last(a1, v1); a2 = Last(a2, v2); a3 = Last(a3, v3); }
                    }

                    acc = max ? First(First(acc, a1), First(a2, a3)) : Last(Last(acc, a1), Last(a2, a3));
                }

                for (; i + w <= len; i += w) { var v = Vector.LoadUnsafe(ref rs, (nuint)i); acc = max ? First(acc, v) : Last(acc, v); }
                TLane e = acc[0];
                for (int l = 1; l < w; l++) e = max ? (Widths.Compare(acc[l], e) > 0 ? acc[l] : e) : (Widths.Compare(acc[l], e) < 0 ? acc[l] : e);
                if (!have || (max ? Widths.Compare(e, best) > 0 : Widths.Compare(e, best) < 0)) { best = e; have = true; }
            }

            for (; i < len; i++)
            {
                if (!have || (max ? Widths.Compare(s[i], best) > 0 : Widths.Compare(s[i], best) < 0)) { best = s[i]; have = true; }
            }

            at += len;
        }

        // The first position holding the extreme.
        if (have && !wantIndex) { index = 0; return best; }
        if (have)
        {
            if (TN.Direct)
            {
                for (int i = 0; i < n; i++) if (Widths.Compare(direct[i], best) == 0) { index = i; return best; }
                return best;
            }

            for (int at = 0; at < n;)
            {
                int len = Math.Min(scratch.Length, n - at);
                TN.Load(x, first + at, scratch[..len]);
                for (int i = 0; i < len; i++) if (Widths.Compare(scratch[i], best) == 0) { index = at + i; return best; }
                at += len;
            }
        }

        return best;
    }

    private struct PartCompareBody<TN, TLane, TL>(byte[] x, byte[] values, byte[] indices, int n, int span, bool max, bool wantIndex = true) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var pv = MemoryMarshal.Cast<byte, TLane>(values.AsSpan());
            var pi = MemoryMarshal.Cast<byte, int>(indices.AsSpan());
            Span<TLane> scratch = stackalloc TLane[4096];
            for (int p = first; p < last; p++)
            {
                int start = p * span, len = Math.Min(span, n - start);
                if (len <= 0) { pi[p] = -1; continue; }
                pv[p] = Extreme<TN, TLane, TL>(x, start, len, max, scratch, out int index, wantIndex);
                pi[p] = index < 0 ? -1 : start + index;
            }
        }
    }

    private struct RowCompareBody<TN, TLane, TL>(byte[] x, byte[] result, int cols, bool max) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var o = MemoryMarshal.Cast<byte, TLane>(result.AsSpan());
            Span<TLane> scratch = stackalloc TLane[4096];
            for (int r = first; r < last; r++) o[r] = Extreme<TN, TLane, TL>(x, r * cols, cols, max, scratch, out _, wantIndex: false);
        }
    }

    private struct ArgmaxBody<TN, TLane, TL>(byte[] x, byte[] result, int cols) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var o = MemoryMarshal.Cast<byte, int>(result.AsSpan());
            Span<TLane> scratch = stackalloc TLane[4096];
            for (int r = first; r < last; r++) { Extreme<TN, TLane, TL>(x, r * cols, cols, true, scratch, out int index); o[r] = index; }
        }
    }

    /// <summary>A tile's first-ranked value of each column and its row, the rows read in order: a row's value taken only
    /// when Adad's order (Widths.Compare: a NaN above every number, ±0 equal) puts it strictly above the column's best so
    /// far. A vector compare a row gives exactly those lanes (above by the lane's compare, or the value a NaN and the best
    /// not): 4-byte lanes take value and row by it; other lanes take the vectors it marks by the order, value by value. In
    /// place where the stored values are the lane's.</summary>
    private readonly struct ColumnArgmaxTile<TN, TLane, TL>(byte[] x, byte[] values, byte[] rowsAt, int cols) : IColumnTile
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Tile(int r0, int r1, int c0, int c1, int block)
        {
            int n = c1 - c0, w = Vector<TLane>.Count;
            var best = MemoryMarshal.Cast<byte, TLane>(values.AsSpan()).Slice(block * cols + c0, n);
            var at = MemoryMarshal.Cast<byte, int>(rowsAt.AsSpan()).Slice(block * cols + c0, n);
            var row = TN.Direct ? null : ArrayPool<TLane>.Shared.Rent(n);
            ReadOnlySpan<TLane> stored = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(x.AsSpan()) : default;
            if (TN.Direct) stored.Slice(r0 * cols + c0, n).CopyTo(best); else TN.Load(x, r0 * cols + c0, best);
            at.Fill(r0);
            ref TLane rb = ref MemoryMarshal.GetReference(best);
            for (int r = r0 + 1; r < r1; r++)
            {
                ReadOnlySpan<TLane> s;
                if (TN.Direct) s = stored.Slice(r * cols + c0, n);
                else { TN.Load(x, r * cols + c0, row.AsSpan(0, n)); s = row.AsSpan(0, n); }
                ref TLane rs = ref MemoryMarshal.GetReference(s);
                int i = 0;
                if (Unsafe.SizeOf<TLane>() == 4)
                {
                    // 4-byte lanes: the rows beside the values (a lane of int each), both taken by the mask, no branch.
                    ref int ra = ref MemoryMarshal.GetReference(at);
                    var rv = new Vector<int>(r);
                    for (; i + w <= n; i += w)
                    {
                        var v = Vector.LoadUnsafe(ref rs, (nuint)i);
                        var b = Vector.LoadUnsafe(ref rb, (nuint)i);
                        var above = TL.Greater(v, b) | Vector.AndNot(TL.Equal(b, b), TL.Equal(v, v));
                        Vector.ConditionalSelect(above, v, b).StoreUnsafe(ref rb, (nuint)i);
                        Vector.ConditionalSelect(Vector.As<TLane, int>(above), rv, Vector.LoadUnsafe(ref ra, (nuint)i)).StoreUnsafe(ref ra, (nuint)i);
                    }
                }

                for (; i + w <= n; i += w)
                {
                    var v = Vector.LoadUnsafe(ref rs, (nuint)i);
                    var b = Vector.LoadUnsafe(ref rb, (nuint)i);
                    // Above, or the value not equal to itself (a NaN) where the best is: exactly the lanes the order puts
                    // above (an integer lane equals itself: only the first term).
                    var above = TL.Greater(v, b) | Vector.AndNot(TL.Equal(b, b), TL.Equal(v, v));
                    if (above == Vector<TLane>.Zero) continue;
                    for (int l = i; l < i + w; l++) if (Widths.Compare(s[l], best[l]) > 0) { best[l] = s[l]; at[l] = r; }
                }

                for (; i < n; i++) if (Widths.Compare(s[i], best[i]) > 0) { best[i] = s[i]; at[i] = r; }
            }

            if (row is not null) ArrayPool<TLane>.Shared.Return(row);
        }
    }

    // The blocks' first-ranked joined in block order: a later block's only when it ranks strictly above.
    private struct ArgmaxJoinBody<TLane>(byte[] values, byte[] rowsAt, int blocks, int cols, byte[] result) : Team.IBody where TLane : unmanaged
    {
        public readonly void Run(int first, int last)
        {
            var v = MemoryMarshal.Cast<byte, TLane>(values.AsSpan());
            var a = MemoryMarshal.Cast<byte, int>(rowsAt.AsSpan());
            var o = MemoryMarshal.Cast<byte, int>(result.AsSpan());
            for (int c = first; c < last; c++)
            {
                TLane best = v[c];
                int at = a[c];
                for (int b = 1; b < blocks; b++)
                {
                    if (Widths.Compare(v[b * cols + c], best) > 0) { best = v[b * cols + c]; at = a[b * cols + c]; }
                }

                o[c] = at;
            }
        }
    }

    // A tile's column extremes in Adad's order: First / Last on whole vectors, Widths.Compare on the rest (as the rows'
    // pass), in place where the stored values are the lane's.
    private readonly struct ColumnCompareTile<TN, TLane, TL>(byte[] x, byte[] dst, int cols, bool max) : IColumnTile
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Tile(int r0, int r1, int c0, int c1, int block)
        {
            int n = c1 - c0, w = Vector<TLane>.Count;
            var o = MemoryMarshal.Cast<byte, TLane>(dst.AsSpan()).Slice(block * cols + c0, n);
            var row = TN.Direct ? null : ArrayPool<TLane>.Shared.Rent(n);
            ReadOnlySpan<TLane> stored = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(x.AsSpan()) : default;
            if (TN.Direct) stored.Slice(r0 * cols + c0, n).CopyTo(o); else TN.Load(x, r0 * cols + c0, o);
            ref TLane ro = ref MemoryMarshal.GetReference(o);
            for (int r = r0 + 1; r < r1; r++)
            {
                ReadOnlySpan<TLane> s;
                if (TN.Direct) s = stored.Slice(r * cols + c0, n);
                else { TN.Load(x, r * cols + c0, row.AsSpan(0, n)); s = row.AsSpan(0, n); }
                ref TLane rs = ref MemoryMarshal.GetReference(s);
                int i = 0;
                if (max) for (; i + w <= n; i += w) First(Vector.LoadUnsafe(ref ro, (nuint)i), Vector.LoadUnsafe(ref rs, (nuint)i)).StoreUnsafe(ref ro, (nuint)i);
                else for (; i + w <= n; i += w) Last(Vector.LoadUnsafe(ref ro, (nuint)i), Vector.LoadUnsafe(ref rs, (nuint)i)).StoreUnsafe(ref ro, (nuint)i);
                for (; i < n; i++) if (max ? Widths.Compare(s[i], o[i]) > 0 : Widths.Compare(s[i], o[i]) < 0) o[i] = s[i];
            }

            if (row is not null) ArrayPool<TLane>.Shared.Return(row);
        }
    }

    // ---------------------------------------------------------------- row-vector broadcast: output[r][c] = op(x[r][c], v[c])

    public static string MapRow<TN>(string op, byte[] x, byte[] v, byte[] output, int rows, int cols) where TN : struct, IAdad<TN>
    {
        if (Ops.Find(op).FloatOnly) throw new NotSupportedException($"{op}: a float formula; {TN.Name} runs it through MapRowFloat");
        var visitor = new RowMapVisitor(op, x, v, output, rows, cols);
        TN.Elementwise(ref visitor);
        return visitor.Lane;
    }

    /// <summary>MapRow for a float type: the formulas every lane has in its element-wise lane, the float formulas in its float lane.</summary>
    public static string MapRowFloat<TN>(string op, byte[] x, byte[] v, byte[] output, int rows, int cols) where TN : struct, IFloatAdad<TN>
    {
        if (!Ops.Find(op).FloatOnly) return MapRow<TN>(op, x, v, output, rows, cols);
        var visitor = new FloatRowMapVisitor(op, x, v, output, rows, cols);
        TN.Float(ref visitor);
        return visitor.Lane;
    }

    public struct RowMapVisitor(string op, byte[] x, byte[] v, byte[] output, int rows, int cols) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            var use = new RowMapUse<TN, TLane, TL>(x, v, output, rows, cols);
            if (!MapOps.AnyOp<TLane, TL, RowMapUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException(op);
        }
    }

    public struct FloatRowMapVisitor(string op, byte[] x, byte[] v, byte[] output, int rows, int cols) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            var use = new RowMapUse<TN, TLane, TL>(x, v, output, rows, cols);
            if (!MapOps.FloatOp<TLane, TL, RowMapUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException(op);
        }
    }

    private readonly struct RowMapUse<TN, TLane, TL>(byte[] x, byte[] v, byte[] output, int rows, int cols) : IOpUse<TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Use<TOp>() where TOp : struct, IMapOp<TLane, TL> => RowMapRun<TN, TLane, TL, TOp>(x, v, output, rows, cols);
    }

    private static void RowMapRun<TN, TLane, TL, TOp>(byte[] x, byte[] v, byte[] output, int rows, int cols)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        // Packed types whose rows do not end on a byte: chunks of whole bytes (a multiple of the values a byte holds), so two
        // threads never write the same byte.
        int per = TN.Granularity, align = cols % per == 0 ? 1 : per;
        int chunk = Math.Max(align, (rows / (Threads * 4) + align - 1) / align * align);
        Team.For(rows, chunk, (long)rows * cols, new RowMapBody<TN, TLane, TL, TOp>(x, v, output, cols));
    }

    private struct RowMapBody<TN, TLane, TL, TOp>(byte[] x, byte[] v, byte[] output, int cols) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            int w = Vector<TLane>.Count;
            var pool = ArrayPool<TLane>.Shared;
            var vl = pool.Rent(cols);
            var xl = TN.Direct ? null : pool.Rent(cols);
            Span<TLane> px = stackalloc TLane[w];
            Span<TLane> py = stackalloc TLane[w];
            try
            {
                TN.Load(v, 0, vl.AsSpan(0, cols));
                for (int r = first; r < last; r++)
                {
                    Span<TLane> row = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols, cols) : xl.AsSpan(0, cols);
                    if (!TN.Direct) TN.Load(x, r * cols, row);
                    Span<TLane> o = TN.Direct ? MemoryMarshal.Cast<byte, TLane>(output.AsSpan()).Slice(r * cols, cols) : row;
                    int i = 0;
                    for (; i + w <= cols; i += w) TOp.Apply(new Vector<TLane>(row[i..]), new Vector<TLane>(vl.AsSpan(i)), default, default, default).CopyTo(o[i..]);
                    if (i < cols)
                    {
                        px.Fill(TL.From(1)); py.Fill(TL.From(1));
                        row[i..].CopyTo(px);
                        vl.AsSpan(i, cols - i).CopyTo(py);
                        var res = TOp.Apply(new Vector<TLane>(px), new Vector<TLane>(py), default, default, default);
                        for (int l = i; l < cols; l++) o[l] = res[l - i];
                    }

                    if (!TN.Direct)
                    {
                        if (!TN.IsFloat && TN.Bits < 8 * Unsafe.SizeOf<TLane>()) MapWrap(o, TN.Bits, TN.Min < 0);
                        TN.Store(o, output, r * cols);
                    }
                }
            }
            finally
            {
                pool.Return(vl);
                if (xl is not null) pool.Return(xl);
            }
        }
    }

    // Integer results narrower than their lane, wrapped at the type's width (shared with the map kernel).
    internal static void MapWrap<TLane>(Span<TLane> v, int bits, bool signed) where TLane : unmanaged
    {
        int w = Vector<TLane>.Count, i = 0;
        var mask = new Vector<TLane>(Widths.FromLong<TLane>((1L << bits) - 1));
        var half = new Vector<TLane>(Widths.FromLong<TLane>(1L << (bits - 1)));
        for (; i + w <= v.Length; i += w)
        {
            var t = new Vector<TLane>(v[i..]);
            (signed ? ((t + half) & mask) - half : t & mask).CopyTo(v[i..]);
        }

        for (; i < v.Length; i++)
        {
            long l = Widths.AsLong(v[i]), m = (1L << bits) - 1, low = l & m;
            if (signed && (low & (1L << (bits - 1))) != 0) low -= 1L << bits;
            v[i] = Widths.FromLong<TLane>(low);
        }
    }

    // ---------------------------------------------------------------- transpose: output (cols × rows) = xᵀ

    public static string Transpose<TN>(byte[] x, byte[] output, int rows, int cols) where TN : struct, IAdad<TN>
    {
        var visitor = new TransposeVisitor(x, output, rows, cols);
        TN.Elementwise(ref visitor);
        return visitor.Lane;
    }

    public struct TransposeVisitor(byte[] x, byte[] output, int rows, int cols) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            const int B = 64;
            int rowBlocks = (rows + B - 1) / B, colBlocks = (cols + B - 1) / B;
            int per = TN.Granularity;
            if (rows % per != 0)
            {
                // Output rows that do not end on a byte: one thread owns whole output rows (all the row blocks of a column block).
                Team.For(colBlocks, RowChunk<TN>(colBlocks, B * rows, 1), (long)rows * cols, new TransposeColumnsBody<TN, TLane>(x, output, rows, cols, B, rowBlocks));
                return;
            }

            Team.For(rowBlocks * colBlocks, Math.Max(1, rowBlocks * colBlocks / (Threads * 4)), (long)rows * cols, new TransposeBody<TN, TLane>(x, output, rows, cols, B, colBlocks));
        }
    }

    private struct TransposeColumnsBody<TN, TLane>(byte[] x, byte[] output, int rows, int cols, int block, int rowBlocks) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        public readonly void Run(int first, int last)
        {
            Span<TLane> tile = stackalloc TLane[block * block];
            Span<TLane> line = stackalloc TLane[block];
            for (int cb = first; cb < last; cb++)
            {
                int c0 = cb * block, wd = Math.Min(block, cols - c0);
                for (int rb = 0; rb < rowBlocks; rb++)
                {
                    int r0 = rb * block, h = Math.Min(block, rows - r0);
                    for (int r = 0; r < h; r++) TN.Load(x, (r0 + r) * cols + c0, tile.Slice(r * block, wd));
                    for (int c = 0; c < wd; c++)
                    {
                        for (int r = 0; r < h; r++) line[r] = tile[r * block + c];
                        TN.Store(line[..h], output, (c0 + c) * rows + r0);
                    }
                }
            }
        }
    }

    private struct TransposeBody<TN, TLane>(byte[] x, byte[] output, int rows, int cols, int block, int colBlocks) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            Span<TLane> tile = stackalloc TLane[block * block];
            Span<TLane> line = stackalloc TLane[block];
            for (int t = first; t < last; t++)
            {
                int r0 = t / colBlocks * block, c0 = t % colBlocks * block;
                int h = Math.Min(block, rows - r0), wd = Math.Min(block, cols - c0);
                for (int r = 0; r < h; r++) TN.Load(x, (r0 + r) * cols + c0, tile.Slice(r * block, wd));
                // Output row c (= input column) holds input rows r0 .. r0 + h.
                for (int c = 0; c < wd; c++)
                {
                    for (int r = 0; r < h; r++) line[r] = tile[r * block + c];
                    TN.Store(line[..h], output, (c0 + c) * rows + r0);
                }
            }
        }
    }

    // ---------------------------------------------------------------- matrix · vector

    /// <summary>y = A·x (A rows × cols, x cols, y rows) or, transposed, y = Aᵀ·x (x rows, y cols), in the sum lane; y stored as the type.</summary>
    public static string Gemv<TN>(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) where TN : struct, IAdad<TN>
    {
        if (TN.Name is "q8_0" or "q4_0" && !transposed && cols % 32 == 0)
        {
            // The block-scaled integer product: the format's own fast path (y encoded by blocks from float32 lanes).
            var yl = new float[rows];
            string blockLane = GemvBlocks(TN.Name, a, x, yl, rows, cols);
            CopyLanes<TN>(MemoryMarshal.AsBytes(yl.AsSpan()).ToArray(), y, rows);
            return blockLane;
        }

        var (aLo, aHi) = TN.IsFloat ? (0, 0) : TN.Range(a, rows * cols);                 // a float type's lane does not depend on them
        var (xLo, xHi) = TN.IsFloat ? (0, 0) : TN.Range(x, transposed ? rows : cols);
        var bounds = new Bounds(aLo, aHi, xLo, xHi, transposed ? rows : cols);
        var visitor = new GemvVisitor(a, x, y, rows, cols, transposed);
        TN.Sum(ref visitor, in bounds);
        return visitor.Lane;
    }

    private struct GemvVisitor(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            GemvIn<TN, TLane, TLane, TL>(a, x, y, rows, cols, transposed);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            Lane = TD.Name;
            GemvIn<TN, TIn, TAcc, TD>(a, x, y, rows, cols, transposed);
        }
    }

    private static void GemvIn<TN, TIn, TAcc, TL>(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>();
        int xn = transposed ? rows : cols, yn = transposed ? cols : rows;
        int xWords = (xn + g - 1) / g;
        var pool = ArrayPool<TAcc>.Shared;
        var xw = pool.Rent(xWords);
        var yl = pool.Rent(yn);
        try
        {
            // x in words of g values (zero past the end).
            var xin = ArrayPool<TIn>.Shared.Rent(xWords * g);
            xin.AsSpan(0, xWords * g).Clear();
            TN.Load(x, 0, xin.AsSpan(0, xn));
            MemoryMarshal.Cast<TIn, TAcc>(xin.AsSpan(0, xWords * g)).CopyTo(xw);
            ArrayPool<TIn>.Shared.Return(xin);
            if (!transposed)
            {
                Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * cols, new GemvRowBody<TN, TIn, TAcc, TL>(a, xw, yl, cols, g));
            }
            else
            {
                // Tiles (ColumnPlan), the blocks' sums joined in block order.
                var plan = ColumnPlan.Of<TAcc>(rows, cols, g);
                var sums = new byte[cols * Unsafe.SizeOf<TAcc>()];
                var dst = plan.Blocks == 1 ? sums : plan.Partials<TAcc>(cols);
                Team.For(plan.Tasks, 1, (long)rows * cols, new ColumnTiles<GemvColumnTile<TN, TIn, TAcc, TL>>(new(a, xw, dst, cols, g), plan, rows, cols));
                if (plan.Blocks > 1) Join<TAcc, TL>(dst, plan.Blocks, cols, sums, JoinOp.Add);
                MemoryMarshal.Cast<byte, TAcc>(sums.AsSpan()).CopyTo(yl);
            }

            TN.Store(yl.AsSpan(0, yn), y, 0);
        }
        finally
        {
            pool.Return(xw);
            pool.Return(yl);
        }
    }

    private struct GemvRowBody<TN, TIn, TAcc, TL>(byte[] a, TAcc[] xw, TAcc[] yl, int cols, int g) : Team.IBody
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            int w = Vector<TAcc>.Count, words = cols / g, per = TN.Granularity;
            bool direct = Lanes.Direct<TN, TIn>() && cols % g == 0;
            // One buffer for the chunk (a stackalloc inside the row loop would grow the stack by a row).
            Span<TIn> buf = direct ? default : stackalloc TIn[4096];
            ref TAcc rx = ref MemoryMarshal.GetArrayDataReference(xw);
            for (int r = first; r < last; r++)
            {
                Vector<TAcc> acc0 = Vector<TAcc>.Zero, acc1 = acc0;
                TAcc tail = TL.Zero;
                if (direct)
                {
                    var row = MemoryMarshal.Cast<TIn, TAcc>(MemoryMarshal.Cast<byte, TIn>(a.AsSpan()).Slice(r * cols, cols));
                    ref TAcc ra = ref MemoryMarshal.GetReference(row);
                    int i = 0;
                    for (; i + 2 * w <= words; i += 2 * w)
                    {
                        acc0 = TL.Fma(Vector.LoadUnsafe(ref ra, (nuint)i), Vector.LoadUnsafe(ref rx, (nuint)i), acc0);
                        acc1 = TL.Fma(Vector.LoadUnsafe(ref ra, (nuint)(i + w)), Vector.LoadUnsafe(ref rx, (nuint)(i + w)), acc1);
                    }

                    for (; i + w <= words; i += w) acc0 = TL.Fma(Vector.LoadUnsafe(ref ra, (nuint)i), Vector.LoadUnsafe(ref rx, (nuint)i), acc0);
                    for (; i < words; i++) tail = TL.Fma(row[i], xw[i], tail);
                }
                else
                {
                    // Loaded in pieces of whole words (the last word padded with zeros).
                    int wordsAll = (cols + g - 1) / g;
                    for (int at = 0, wi = 0; at < cols; at += buf.Length)
                    {
                        int len = Math.Min(buf.Length, cols - at), padded = (len + g - 1) / g * g;
                        buf[..padded].Clear();
                        TN.Load(a, r * cols + at, buf[..len]);
                        var ws = MemoryMarshal.Cast<TIn, TAcc>(buf[..padded]);
                        int i = 0;
                        for (; i + w <= ws.Length && wi + i + w <= wordsAll; i += w)
                            acc0 = TL.Fma(new Vector<TAcc>(ws[i..]), Vector.LoadUnsafe(ref rx, (nuint)(wi + i)), acc0);
                        for (; i < ws.Length; i++) tail = TL.Fma(ws[i], xw[wi + i], tail);
                        wi += ws.Length;
                    }
                }

                yl[r] = TL.Add(tail, Vector.Sum(acc0 + acc1));
            }
        }
    }

    // y's columns of a tile: Σ over the tile's rows of A's row (g rows a word) times x's word, in place where the stored
    // values are the lane's (g = 1).
    private readonly struct GemvColumnTile<TN, TIn, TAcc, TL>(byte[] a, TAcc[] xw, byte[] dst, int cols, int g) : IColumnTile
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Tile(int r0, int r1, int c0, int c1, int block)
        {
            int n = c1 - c0, w = Vector<TAcc>.Count;
            var acc = MemoryMarshal.Cast<byte, TAcc>(dst.AsSpan()).Slice(block * cols + c0, n);
            acc.Clear();
            bool direct = g == 1 && Lanes.Direct<TN, TIn>();
            var rowsIn = direct ? null : ArrayPool<TIn>.Shared.Rent(g * n);
            var units = direct ? null : ArrayPool<TAcc>.Shared.Rent(n);
            ref TAcc ra = ref MemoryMarshal.GetReference(acc);
            for (int r = r0; r < r1; r += g)
            {
                ReadOnlySpan<TAcc> u;
                if (direct) u = MemoryMarshal.Cast<byte, TAcc>(a.AsSpan()).Slice(r * cols + c0, n);
                else
                {
                    int have = Math.Min(g, r1 - r);
                    for (int t = 0; t < g; t++)
                    {
                        var d = rowsIn.AsSpan(t * n, n);
                        if (t < have) TN.Load(a, (r + t) * cols + c0, d); else d.Clear();
                    }

                    Interleave<TIn, TAcc>(rowsIn.AsSpan(0, g * n), units.AsSpan(0, n), n);
                    u = units.AsSpan(0, n);
                }

                TAcc xs = xw[r / g];
                var xv = new Vector<TAcc>(xs);
                ref TAcc ru = ref MemoryMarshal.GetReference(u);
                int i = 0;
                for (; i + w <= n; i += w) TL.Fma(Vector.LoadUnsafe(ref ru, (nuint)i), xv, Vector.LoadUnsafe(ref ra, (nuint)i)).StoreUnsafe(ref ra, (nuint)i);
                for (; i < n; i++) acc[i] = TL.Fma(u[i], xs, acc[i]);
            }

            if (rowsIn is not null) ArrayPool<TIn>.Shared.Return(rowsIn);
            if (units is not null) ArrayPool<TAcc>.Shared.Return(units);
        }
    }
}
