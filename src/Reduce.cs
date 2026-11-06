// Reductions, row-vector broadcasts, transposes and matrix-vector products on the CPU, written once over the contract.
//
//   Reduce   sum, sumsq, max, min, argmax over a whole array, each row, or each column. Sums take the number's sum
//            lane for the bounds measured on the data (a dot lane sums packed groups against ones: the processor's
//            widening multiply-add is its fastest adder for 8- and 16-bit values); max/min/argmax take the type's own
//            lane. The result is lane values (the lane is reported): exact by the rule, the caller converts as it likes.
//   MapRow   output[r][c] = op(x[r][c], v[c]) for the two-input formulas (a row vector broadcast down the rows).
//   Transpose
//   Gemv     y = A·x and y = Aᵀ·x in the sum lane, the result stored as the type.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Qiyas;

public enum Axis
{
    Whole,
    Rows,
    Cols,
}

internal static partial class Kernels
{
    // ---------------------------------------------------------------- reductions

    /// <summary>Reduces x (rows × cols) along the axis: the result as lane values (count 1, rows, or cols; argmax as int32 indices) and the lane's name.</summary>
    public static (byte[] Result, string Lane) Reduce<TN>(string op, Axis axis, byte[] x, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        if (op is "max" or "min" or "argmax")
        {
            var cmp = new CompareVisitor(op, axis, x, rows, cols);
            TN.Elementwise(ref cmp);
            return (cmp.Result!, op == "argmax" ? "s32 index" : cmp.Lane);
        }

        var (lo, hi) = TN.Range(x, rows * cols);
        long terms = axis switch { Axis.Whole => (long)rows * cols, Axis.Rows => cols, _ => rows };
        var bounds = op == "sumsq" ? new Bounds(lo, hi, lo, hi, terms) : new Bounds(lo, hi, 1, 1, terms);
        var sum = new SumVisitor(op == "sumsq", axis, x, rows, cols);
        TN.Sum(ref sum, in bounds);
        return (sum.Result!, sum.Lane);
    }

    private struct SumVisitor(bool squares, Axis axis, byte[] x, int rows, int cols) : ISumVisitor
    {
        public byte[]? Result;
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            Result = SumIn<TN, TLane, TLane, TL>(squares, axis, x, rows, cols);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IQiyasDotLane<TD, TIn, TAcc>
        {
            Lane = TD.Name;
            Result = SumIn<TN, TIn, TAcc, TD>(squares, axis, x, rows, cols);
        }
    }

    // The sum in lane TL over inputs TIn packed into TAcc words (a plain lane: TIn = TAcc, one value a word).
    private static byte[] SumIn<TN, TIn, TAcc, TL>(bool squares, Axis axis, byte[] x, int rows, int cols)
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
    {
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>();
        if (axis == Axis.Whole) { cols = rows * cols; rows = 1; }
        int count = axis == Axis.Cols ? cols : rows;
        var result = new byte[count * Unsafe.SizeOf<TAcc>()];
        var outLanes = MemoryMarshal.Cast<byte, TAcc>(result.AsSpan());
        if (axis == Axis.Cols)
        {
            // Columns across the team; each takes all rows, g at a time (one word of g values per column).
            int chunk = Math.Max(Vector<TAcc>.Count * 4, (cols + Threads - 1) / Threads / Vector<TAcc>.Count * Vector<TAcc>.Count);
            var body = new ColumnSumBody<TN, TIn, TAcc, TL>(x, result, rows, cols, g, squares);
            Team.For(cols, chunk, (long)rows * cols, body);
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
            int parts = Math.Min(Threads * 4, Math.Max(1, cols / 4096));
            int per = ((cols + parts - 1) / parts + g - 1) / g * g;
            parts = Math.Max(1, (cols + per - 1) / per);
            var partials = new byte[parts * Unsafe.SizeOf<TAcc>()];
            Team.For(parts, 1, (long)cols, new PartBody<TN, TIn, TAcc, TL>(x, partials, cols, per, g, squares));
            var p = MemoryMarshal.Cast<byte, TAcc>(partials.AsSpan());
            TAcc total = TL.Zero;
            foreach (var v in p) total = TL.Add(total, v);
            outLanes[0] = total;
            return result;
        }

        Team.For(rows, rowChunk, (long)rows * cols, rowBody);
        return result;
    }

    // Σ over one row's values [first, first + n): words of g inputs each, Fma against ones (sum) or themselves (squares).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static TAcc DotRow<TN, TIn, TAcc, TL>(byte[] x, int first, int n, int g, bool squares, Span<TIn> scratch)
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
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
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
    {
        public readonly void Run(int first, int last)
        {
            var outLanes = MemoryMarshal.Cast<byte, TAcc>(result.AsSpan());
            Span<TIn> scratch = stackalloc TIn[4096];
            for (int r = first; r < last; r++) outLanes[r] = DotRow<TN, TIn, TAcc, TL>(x, r * cols, cols, g, squares, scratch);
        }
    }

    private struct PartBody<TN, TIn, TAcc, TL>(byte[] x, byte[] partials, int n, int per, int g, bool squares) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
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

    private struct ColumnSumBody<TN, TIn, TAcc, TL>(byte[] x, byte[] result, int rows, int cols, int g, bool squares) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            int n = last - first, w = Vector<TAcc>.Count;
            var outLanes = MemoryMarshal.Cast<byte, TAcc>(result.AsSpan()).Slice(first, n);
            var pool = ArrayPool<TIn>.Shared;
            var rowsIn = pool.Rent(g * n);
            var accPool = ArrayPool<TAcc>.Shared;
            var units = accPool.Rent(n);
            var acc = accPool.Rent(n);
            try
            {
                acc.AsSpan(0, n).Clear();
                var ones = new Vector<TAcc>(Ones<TIn, TAcc>());
                for (int r0 = 0; r0 < rows; r0 += g)
                {
                    int have = Math.Min(g, rows - r0);
                    for (int t = 0; t < g; t++)
                    {
                        var dst = rowsIn.AsSpan(t * n, n);
                        if (t < have) TN.Load(x, (r0 + t) * cols + first, dst); else dst.Clear();
                    }

                    Interleave<TIn, TAcc>(rowsIn.AsSpan(0, g * n), units.AsSpan(0, n), n);
                    ref TAcc ru = ref MemoryMarshal.GetArrayDataReference(units);
                    ref TAcc ra = ref MemoryMarshal.GetArrayDataReference(acc);
                    int i = 0;
                    for (; i + w <= n; i += w)
                    {
                        var u = Vector.LoadUnsafe(ref ru, (nuint)i);
                        TL.Fma(u, squares ? u : ones, Vector.LoadUnsafe(ref ra, (nuint)i)).StoreUnsafe(ref ra, (nuint)i);
                    }

                    for (; i < n; i++) acc[i] = TL.Fma(units[i], squares ? units[i] : ones[0], acc[i]);
                }

                acc.AsSpan(0, n).CopyTo(outLanes);
            }
            finally
            {
                pool.Return(rowsIn);
                accPool.Return(units);
                accPool.Return(acc);
            }
        }
    }

    // ---------------------------------------------------------------- max, min, argmax in the type's own lane

    private struct CompareVisitor(string op, Axis axis, byte[] x, int rows, int cols) : ILaneVisitor
    {
        public byte[]? Result;
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            bool max = op != "min";
            int r = rows, c = cols;
            if (axis == Axis.Whole) { r = 1; c = rows * cols; }
            if (axis == Axis.Cols && op == "argmax")
            {
                Result = new byte[c * 4];
                Lane = "s32 index";
                var body = new ColumnArgmaxBody<TN, TLane>(x, Result, r, c);
                Team.For(c, Math.Max(64, (c + Threads - 1) / Threads), (long)r * c, body);
                return;
            }

            if (axis == Axis.Cols)
            {
                Result = new byte[c * Unsafe.SizeOf<TLane>()];
                var body = new ColumnCompareBody<TN, TLane, TL>(x, Result, r, c, max);
                Team.For(c, Math.Max(Vector<TLane>.Count * 4, (c + Threads - 1) / Threads), (long)r * c, body);
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
                Team.For(parts, 1, (long)c, new PartCompareBody<TN, TLane, TL>(x, values, indices, c, span, max));
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLane Extreme<TN, TLane, TL>(byte[] x, int first, int n, bool max, Span<TLane> scratch, out int index)
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
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
                var acc = new Vector<TLane>(s);
                for (i = w; i + w <= len; i += w) { var v = new Vector<TLane>(s[i..]); acc = max ? TL.Max(acc, v) : TL.Min(acc, v); }
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

    private struct PartCompareBody<TN, TLane, TL>(byte[] x, byte[] values, byte[] indices, int n, int span, bool max) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
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
                pv[p] = Extreme<TN, TLane, TL>(x, start, len, max, scratch, out int index);
                pi[p] = index < 0 ? -1 : start + index;
            }
        }
    }

    private struct RowCompareBody<TN, TLane, TL>(byte[] x, byte[] result, int cols, bool max) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var o = MemoryMarshal.Cast<byte, TLane>(result.AsSpan());
            Span<TLane> scratch = stackalloc TLane[4096];
            for (int r = first; r < last; r++) o[r] = Extreme<TN, TLane, TL>(x, r * cols, cols, max, scratch, out _);
        }
    }

    private struct ArgmaxBody<TN, TLane, TL>(byte[] x, byte[] result, int cols) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            var o = MemoryMarshal.Cast<byte, int>(result.AsSpan());
            Span<TLane> scratch = stackalloc TLane[4096];
            for (int r = first; r < last; r++) { Extreme<TN, TLane, TL>(x, r * cols, cols, true, scratch, out int index); o[r] = index; }
        }
    }

    /// <summary>The first largest value's row index of each column (s32), the rows read in order.</summary>
    private struct ColumnArgmaxBody<TN, TLane>(byte[] x, byte[] result, int rows, int cols) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
    {
        public readonly void Run(int first, int last)
        {
            int n = last - first;
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(n);
            var best = pool.Rent(n);
            var at = ArrayPool<int>.Shared.Rent(n);
            try
            {
                for (int r = 0; r < rows; r++)
                {
                    TN.Load(x, r * cols + first, row.AsSpan(0, n));
                    for (int i = 0; i < n; i++)
                    {
                        if (r == 0 || Widths.Compare(row[i], best[i]) > 0) { best[i] = row[i]; at[i] = r; }
                    }
                }

                MemoryMarshal.Cast<byte, int>(result.AsSpan()).Slice(first, n).Clear();
                at.AsSpan(0, n).CopyTo(MemoryMarshal.Cast<byte, int>(result.AsSpan()).Slice(first, n));
            }
            finally
            {
                pool.Return(row); pool.Return(best); ArrayPool<int>.Shared.Return(at);
            }
        }
    }

    private struct ColumnCompareBody<TN, TLane, TL>(byte[] x, byte[] result, int rows, int cols, bool max) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            int n = last - first, w = Vector<TLane>.Count;
            var o = MemoryMarshal.Cast<byte, TLane>(result.AsSpan()).Slice(first, n);
            var pool = ArrayPool<TLane>.Shared;
            var row = pool.Rent(n);
            try
            {
                TN.Load(x, first, row.AsSpan(0, n));
                row.AsSpan(0, n).CopyTo(o);
                for (int r = 1; r < rows; r++)
                {
                    TN.Load(x, r * cols + first, row.AsSpan(0, n));
                    int i = 0;
                    for (; i + w <= n; i += w)
                    {
                        var a = new Vector<TLane>(o[i..]);
                        var v = new Vector<TLane>(row.AsSpan(i));
                        (max ? TL.Max(a, v) : TL.Min(a, v)).CopyTo(o[i..]);
                    }

                    for (; i < n; i++) if (max ? Widths.Compare(row[i], o[i]) > 0 : Widths.Compare(row[i], o[i]) < 0) o[i] = row[i];
                }
            }
            finally
            {
                pool.Return(row);
            }
        }
    }

    // ---------------------------------------------------------------- row-vector broadcast: output[r][c] = op(x[r][c], v[c])

    public static string MapRow<TN>(string op, byte[] x, byte[] v, byte[] output, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        var visitor = new RowMapVisitor(op, x, v, output, rows, cols);
        TN.Elementwise(ref visitor);
        return visitor.Lane;
    }

    public struct RowMapVisitor(string op, byte[] x, byte[] v, byte[] output, int rows, int cols) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            switch (op)
            {
                case "add": RowMapRun<TN, TLane, TL, MapOps.Add<TLane, TL>>(x, v, output, rows, cols); break;
                case "sub": RowMapRun<TN, TLane, TL, MapOps.Sub<TLane, TL>>(x, v, output, rows, cols); break;
                case "mul": RowMapRun<TN, TLane, TL, MapOps.Mul<TLane, TL>>(x, v, output, rows, cols); break;
                case "div": RowMapRun<TN, TLane, TL, MapOps.Div<TLane, TL>>(x, v, output, rows, cols); break;
                case "max": RowMapRun<TN, TLane, TL, MapOps.Max<TLane, TL>>(x, v, output, rows, cols); break;
                case "min": RowMapRun<TN, TLane, TL, MapOps.Min<TLane, TL>>(x, v, output, rows, cols); break;
                default: throw new ArgumentException($"{op}: not a two-input formula every lane has");
            }
        }
    }

    private static void RowMapRun<TN, TLane, TL, TOp>(byte[] x, byte[] v, byte[] output, int rows, int cols)
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        // Packed types whose rows do not end on a byte: chunks of whole bytes (a multiple of the values a byte holds), so two
        // threads never write the same byte.
        int per = TN.Granularity, align = cols % per == 0 ? 1 : per;
        int chunk = Math.Max(align, (rows / (Threads * 4) + align - 1) / align * align);
        Team.For(rows, chunk, (long)rows * cols, new RowMapBody<TN, TLane, TL, TOp>(x, v, output, cols));
    }

    private struct RowMapBody<TN, TLane, TL, TOp>(byte[] x, byte[] v, byte[] output, int cols) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
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

    public static string Transpose<TN>(byte[] x, byte[] output, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        var visitor = new TransposeVisitor(x, output, rows, cols);
        TN.Elementwise(ref visitor);
        return visitor.Lane;
    }

    public struct TransposeVisitor(byte[] x, byte[] output, int rows, int cols) : ILaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
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
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
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
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged
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
    public static string Gemv<TN>(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) where TN : struct, IQiyasNumber<TN>
    {
        if (TN.Name is "q8_0" or "q4_0" && !transposed && cols % 32 == 0)
        {
            // The block-scaled integer product: the format's own fast path (y encoded by blocks from float32 lanes).
            var yl = new float[rows];
            string blockLane = GemvBlocks(TN.Name, a, x, yl, rows, cols);
            CopyLanes<TN>(MemoryMarshal.AsBytes(yl.AsSpan()).ToArray(), y, rows);
            return blockLane;
        }

        var (aLo, aHi) = TN.Range(a, rows * cols);
        var (xLo, xHi) = TN.Range(x, transposed ? rows : cols);
        var bounds = new Bounds(aLo, aHi, xLo, xHi, transposed ? rows : cols);
        var visitor = new GemvVisitor(a, x, y, rows, cols, transposed);
        TN.Sum(ref visitor, in bounds);
        return visitor.Lane;
    }

    private struct GemvVisitor(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            Lane = TL.Name;
            GemvIn<TN, TLane, TLane, TL>(a, x, y, rows, cols, transposed);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IQiyasDotLane<TD, TIn, TAcc>
        {
            Lane = TD.Name;
            GemvIn<TN, TIn, TAcc, TD>(a, x, y, rows, cols, transposed);
        }
    }

    private static void GemvIn<TN, TIn, TAcc, TL>(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed)
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
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
                int chunk = Math.Max(Vector<TAcc>.Count * 4, (cols + Threads - 1) / Threads / Vector<TAcc>.Count * Vector<TAcc>.Count);
                Team.For(cols, chunk, (long)rows * cols, new GemvColumnBody<TN, TIn, TAcc, TL>(a, xw, yl, rows, cols, g));
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
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
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

    private struct GemvColumnBody<TN, TIn, TAcc, TL>(byte[] a, TAcc[] xw, TAcc[] yl, int rows, int cols, int g) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TIn>, IQiyasNumber<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, IQiyasLane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            int n = last - first, w = Vector<TAcc>.Count;
            var inPool = ArrayPool<TIn>.Shared;
            var accPool = ArrayPool<TAcc>.Shared;
            var rowsIn = inPool.Rent(g * n);
            var units = accPool.Rent(n);
            var acc = accPool.Rent(n);
            try
            {
                acc.AsSpan(0, n).Clear();
                for (int r0 = 0; r0 < rows; r0 += g)
                {
                    int have = Math.Min(g, rows - r0);
                    for (int t = 0; t < g; t++)
                    {
                        var dst = rowsIn.AsSpan(t * n, n);
                        if (t < have) TN.Load(a, (r0 + t) * cols + first, dst); else dst.Clear();
                    }

                    Interleave<TIn, TAcc>(rowsIn.AsSpan(0, g * n), units.AsSpan(0, n), n);
                    var xv = new Vector<TAcc>(xw[r0 / g]);
                    int i = 0;
                    for (; i + w <= n; i += w) TL.Fma(new Vector<TAcc>(units.AsSpan(i)), xv, new Vector<TAcc>(acc.AsSpan(i))).CopyTo(acc.AsSpan(i));
                    for (; i < n; i++) acc[i] = TL.Fma(units[i], xw[r0 / g], acc[i]);
                }

                acc.AsSpan(0, n).CopyTo(yl.AsSpan(first, n));
            }
            finally
            {
                inPool.Return(rowsIn);
                accPool.Return(units);
                accPool.Return(acc);
            }
        }
    }
}
