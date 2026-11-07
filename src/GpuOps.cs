// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The operations on the device over the contract: the same lane rules as the CPU (Types.cs), with the device's lanes
// and the device's measured facts (whether its JIT takes dp4a; which product tile is faster for a shape).
using System.Diagnostics;

namespace Qiyas;

public sealed partial class GpuOps(Gpu gpu)
{
    private readonly Dictionary<string, int> _tiles = [];

    public Gpu Gpu => gpu;

    private static uint Grid(long items, int block) => (uint)((items + block - 1) / block);

    /// <summary>When set (profile runs only), MatMul adds each phase's milliseconds after a synchronization: pack A, pack B, product, back.</summary>
    public static double[]? Phases;

    private static long Mark(long since, int phase)
    {
        if (Phases is null) return since;
        Gpu.Synchronize();
        long now = Stopwatch.GetTimestamp();
        Phases[phase] += Stopwatch.GetElapsedTime(since, now).TotalMilliseconds;
        return now;
    }

    // ---------------------------------------------------------------- add

    public void Add<TN>(ulong a, ulong b, ulong c, int n) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            // A block format: the sum in float32 storage, the result encoded by blocks.
            var r = new List<ulong>();
            try
            {
                ulong la = Lanes(f, Lane.F32, a, n, r), lb = Lanes(f, Lane.F32, b, n, r), lc = gpu.Rent(n * 4L);
                r.Add(lc);
                Add<Float32>(la, lb, lc, n);
                Back(f, Lane.F32, lc, c, n);
                return;
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        var fn = gpu.Cached($"add:{f.Name}", (out string e) => Ptx.Add(gpu, f, out e));
        long words = f.Words(n);
        int perThread = f.WordBits == 64 ? 2 : 4;
        Gpu.Launch(fn, Grid((words + perThread - 1) / perThread, 256), 1, 256, 1, 0, a, b, c, Gpu.U(n));
    }

    // ---------------------------------------------------------------- reductions, broadcasts, transposes, matrix · vector

    /// <summary>The lane a sum takes on this device for the bounds: f32 while exact, s32/u32, f64, s64/u64 (a dot lane's
    /// accumulator on the CPU is one of these: the device sums narrow values in 32-bit integers directly).</summary>
    public static Lane SumLane(Fmt f, in Bounds bounds)
    {
        if (f.IsFloat) return f.Bits == 64 ? Lane.F64 : Lane.F32;
        double bound = bounds.Bound;
        if (bound <= Exact.Float32) return Lane.F32;
        if (bound <= (f.Signed ? int.MaxValue : uint.MaxValue)) return f.Signed ? Lane.S32 : Lane.U32;
        // Past 32 bits an integer sum takes the 64-bit integer lane: exact further than f64 (2^63 against 2^53) and, on a
        // device whose f64 rate is a fraction of its f32 rate, several times faster (measured: int32 rows 0.133 ms in f64,
        // 0.089 in s64 at 4096 × 4096).
        return f.Signed ? Lane.S64 : Lane.U64;
    }

    /// <summary>Reduces x (rows × cols) along the axis into lane values at `output` (1, rows or cols of them; argmax: s32 indices). Returns the lane's name.</summary>
    public string Reduce<TN>(string op, Axis axis, ulong x, int rows, int cols, ulong output) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        int total = rows * cols;
        if (op is "max" or "min" or "argmax")
        {
            var own = f.Natural;
            if (op == "argmax")
            {
                if (axis == Axis.Cols)
                {
                    var fc = gpu.Cached($"argmax_cols:{f.Name}", (out string e) => Ptx.ArgmaxCols(gpu, f, own, out e));
                    Gpu.Launch(fc, Grid(cols, 256), 1, 256, 1, 0, x, output, Gpu.U(rows), Gpu.U(cols));
                    return "s32 index";
                }

                var fn = gpu.Cached($"argmax:{f.Name}", (out string e) => Ptx.ArgmaxRows(gpu, f, own, out e));
                int r = axis == Axis.Whole ? 1 : rows, c = axis == Axis.Whole ? total : cols;
                Gpu.Launch(fn, (uint)r, 1, Ptx.ReduceThreads, 1, 0, x, output, Gpu.U(c));
                return "s32 index";
            }

            Launch(f, own, op, false, axis, x, rows, cols, output);
            return own.Name;
        }

        // The bounds: the type's range for floats and 8-bit values (it decides the lane already) and for 16- and 32-bit
        // integers (measuring costs a pass as long as the sum, and the lane the type's range gives is exact and runs at the
        // same bandwidth); measured for 64-bit integers, where the type's range fits no lane.
        var (lo, hi) = !f.IsFloat && f.Bits is 16 or 32 ? (TN.Min, TN.Max) : Range<TN>(x, total);
        long terms = axis switch { Axis.Whole => total, Axis.Rows => cols, _ => rows };
        var bounds = op == "sumsq" ? new Bounds(lo, hi, lo, hi, terms) : new Bounds(lo, hi, 1, 1, terms);
        var lane = SumLane(f, in bounds);
        if (axis == Axis.Whole && f.IsFloat && !lane.Wide)
        {
            // A narrow float type's whole sum: each thread's running sum in the f32 lane compensated (Kahan), the threads'
            // and the chunks' sums joined in float64, the result rounded once to the f32 lane — the f32 lane's speed, and the
            // rounding of millions of terms no longer growing with them.
            int n = rows * cols, blocks = WholeBlocks(n), chunk = WholeChunk(n);
            var kahan = gpu.Cached($"reduce_rows:sum:{op == "sumsq"}:{f.Name}:f32:kahan", (out string e) => Ptx.ReduceRows(gpu, f, Lane.F32, "sum", op == "sumsq", out e, compensated: true));
            var join = gpu.Cached($"reduce_rows:sum:False:{Ptx.LaneFormat(Lane.F64).Name}:f64", (out string e) => Ptx.ReduceRows(gpu, Ptx.LaneFormat(Lane.F64), Lane.F64, "sum", false, out e));
            ulong partials = gpu.Rent((long)blocks * 8), wide = gpu.Rent(8);
            try
            {
                Gpu.Launch(kahan, (uint)blocks, 1, Ptx.ReduceThreads, 1, 0, x, partials, Gpu.U(n), Gpu.U(chunk));
                Gpu.Launch(join, 1, 1, Ptx.ReduceThreads, 1, 0, partials, wide, Gpu.U(blocks), Gpu.U(0));
                Back(Ptx.LaneFormat(Lane.F32), Lane.F64, wide, output, 1);
            }
            finally
            {
                gpu.Return(partials); gpu.Return(wide);
            }

            return lane.Name;
        }

        if (axis == Axis.Whole)
        {
            // The chunk pass needs only its own terms' bound: a narrower lane there (s32 for 8-bit values over a few thousand
            // terms) and the wide lane over the partials, instead of the wide lane over every value.
            var chunkBounds = op == "sumsq" ? new Bounds(lo, hi, lo, hi, WholeChunk(total)) : new Bounds(lo, hi, 1, 1, WholeChunk(total));
            var chunkLane = SumLane(f, in chunkBounds);
            // Partials of an integer sum stay integers (a float partial would need a float-to-integer step).
            if (!lane.IsFloat && chunkLane.IsFloat) chunkLane = f.Signed ? Lane.S32 : Lane.U32;
            Launch(f, lane, "sum", op == "sumsq", axis, x, rows, cols, output, chunkLane);
            return lane.Name;
        }

        if (axis == Axis.Rows && f.IsFloat && !lane.Wide)
        {
            // A narrow float type's row sums: the compensated pass, a block a row, the threads' sums joined as exact f32 pairs
            // and rounded once (a float64 join per row ran at the device's f64 rate: float16 rows 0.040 ms against 0.023).
            // Whole words a row (cols a multiple of the values a word holds): one load for every value of the word.
            var rowsKahan = !f.IsBlock && f.WordBits == 32 && cols % f.Per == 0
                ? gpu.Cached($"sum_rows_words:{op == "sumsq"}:{f.Name}", (out string e) => Ptx.SumRowsWords(gpu, f, op == "sumsq", out e))
                : gpu.Cached($"reduce_rows:sum:{op == "sumsq"}:{f.Name}:f32:kahan:f32", (out string e) => Ptx.ReduceRows(gpu, f, Lane.F32, "sum", op == "sumsq", out e, compensated: true, storeF32: true));
            Gpu.Launch(rowsKahan, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, output, Gpu.U(cols), Gpu.U(0));
            return lane.Name;
        }

        Launch(f, lane, "sum", op == "sumsq", axis, x, rows, cols, output);
        return lane.Name;
    }

    // A whole reduction's chunking: at most 1024 blocks of at least 4096 values.
    private static int WholeBlocks(int n) => (int)Math.Min(1024, Math.Max(1, (n + 4095) / 4096));
    private static int WholeChunk(int n) => (n + WholeBlocks(n) - 1) / WholeBlocks(n);

    private void Launch(Fmt f, Lane lane, string op, bool squares, Axis axis, ulong x, int rows, int cols, ulong output, Lane? chunkLane = null)
    {
        if (axis == Axis.Cols)
        {
            // Rows in splits so the device has enough threads (a thread per column per split); the splits joined by one more pass.
            var fc = gpu.Cached($"reduce_cols:{op}:{squares}:{f.Name}:{lane.Name}", (out string e) => Ptx.ReduceCols(gpu, f, lane, op, squares, out e));
            int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
            if (splits == 1)
            {
                Gpu.Launch(fc, Grid(cols, 256), 1, 256, 1, 0, x, output, Gpu.U(rows), Gpu.U(cols), Gpu.U(per));
                return;
            }

            ulong colParts = gpu.Rent((long)splits * cols * lane.Size);
            try
            {
                Gpu.Launch(fc, Grid(cols, 256), (uint)splits, 256, 1, 0, x, colParts, Gpu.U(rows), Gpu.U(cols), Gpu.U(per));
                var lf = Ptx.LaneFormat(lane);
                var f2 = gpu.Cached($"reduce_cols:{op}:False:{lf.Name}:{lane.Name}", (out string e) => Ptx.ReduceCols(gpu, lf, lane, op, false, out e));
                Gpu.Launch(f2, Grid(cols, 256), 1, 256, 1, 0, colParts, output, Gpu.U(splits), Gpu.U(cols), Gpu.U(splits));
            }
            finally
            {
                gpu.Return(colParts);
            }

            return;
        }

        var fr = gpu.Cached($"reduce_rows:{op}:{squares}:{f.Name}:{lane.Name}", (out string e) => Ptx.ReduceRows(gpu, f, lane, op, squares, out e));
        if (axis == Axis.Rows)
        {
            Gpu.Launch(fr, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, output, Gpu.U(cols), Gpu.U(0));
            return;
        }

        // Whole: blocks over chunks into partials (lane values; the chunk lane, narrower when a chunk's bound allows), then
        // the partials as one row in the lane (the identity on squares: plain sum).
        int n = rows * cols, blocks = WholeBlocks(n), chunk = WholeChunk(n);
        var first = chunkLane ?? lane;
        if (first.Name != lane.Name) fr = gpu.Cached($"reduce_rows:{op}:{squares}:{f.Name}:{first.Name}", (out string e) => Ptx.ReduceRows(gpu, f, first, op, squares, out e));
        ulong partials = gpu.Rent((long)blocks * first.Size);
        try
        {
            Gpu.Launch(fr, (uint)blocks, 1, Ptx.ReduceThreads, 1, 0, x, partials, Gpu.U(n), Gpu.U(chunk));
            var lf = Ptx.LaneFormat(first);
            var f2 = gpu.Cached($"reduce_rows:{op}:False:{lf.Name}:{lane.Name}", (out string e) => Ptx.ReduceRows(gpu, lf, lane, op, false, out e));
            Gpu.Launch(f2, 1, 1, Ptx.ReduceThreads, 1, 0, partials, output, Gpu.U(blocks), Gpu.U(0));
        }
        finally
        {
            gpu.Return(partials);
        }
    }

    // Row splits for a per-column pass: enough column·split threads to fill the device, at most 64.
    private int Splits(int rows, int cols)
    {
        int want = gpu.Multiprocessors * 2048;
        int splits = (int)Math.Min(64, Math.Max(1, (want + cols - 1) / cols));
        return Math.Min(splits, Math.Max(1, rows / 64));
    }

    /// <summary>output[r][c] = op(x[r][c], v[c]) in the format's natural lane.</summary>
    public string MapRow<TN>(MapOp op, ulong x, ulong v, ulong output, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                ulong lx = Lanes(f, Lane.F32, x, (long)rows * cols, r), lv = Lanes(f, Lane.F32, v, cols, r), lo = gpu.Rent((long)rows * cols * 4);
                r.Add(lo);
                MapRow<Float32>(op, lx, lv, lo, rows, cols);
                Back(f, Lane.F32, lo, output, (long)rows * cols);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        var lane = f.Natural;
        var fn = gpu.Cached($"maprow:{op.Name}:{f.Name}", (out string e) => Ptx.MapRow(gpu, f, lane, op, out e));
        int n = rows * cols;
        Gpu.Launch(fn, Grid(f.Words(n), 256), 1, 256, 1, 0, x, v, output, Gpu.U(n), Gpu.U(cols));
        return lane.Name;
    }

    public string Transpose<TN>(ulong x, ulong output, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                ulong lx = Lanes(f, Lane.F32, x, (long)rows * cols, r), lo = gpu.Rent((long)rows * cols * 4);
                r.Add(lo);
                Transpose<Float32>(lx, lo, rows, cols);
                Back(f, Lane.F32, lo, output, (long)rows * cols);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        var lane = f.Natural;
        // Output rows end on words when the input row count is a multiple of the values a word holds; otherwise values are
        // written one by one (sub-byte values ORed into a zeroed output).
        bool valueWise = rows % f.Per != 0;
        if (valueWise && f.Bits < 8) Gpu.Clear(output, f.Words((long)rows * cols) * (f.WordBits / 8));
        var fn = gpu.Cached($"transpose:{f.Name}:{valueWise}", (out string e) => Ptx.Transpose(gpu, f, lane, valueWise, out e));
        Gpu.Launch(fn, Grid(cols, Ptx.TransposeTile), Grid(rows, Ptx.TransposeTile), 256, 1, 0, x, output, Gpu.U(rows), Gpu.U(cols));
        return lane.Name;
    }

    /// <summary>y = A·x (or Aᵀ·x) in the sum lane for the measured bounds, y stored as the type.</summary>
    public string Gemv<TN>(ulong a, ulong x, ulong y, int rows, int cols, bool transposed) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.Name is "q8_0" or "q4_0" && !transposed && cols % 32 == 0)
        {
            ulong yBlocks = gpu.Rent(rows * 4L);
            try { string blockLane = GemvBlocks<TN>(a, x, yBlocks, rows, cols); Back(f, Lane.F32, yBlocks, y, rows); return blockLane; }
            finally { gpu.Return(yBlocks); }
        }

        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                int xCount = transposed ? rows : cols, yCount = transposed ? cols : rows;
                ulong la = Lanes(f, Lane.F32, a, (long)rows * cols, r), lx = Lanes(f, Lane.F32, x, xCount, r), ly = gpu.Rent(yCount * 4L);
                r.Add(ly);
                Gemv<Float32>(la, lx, ly, rows, cols, transposed);
                Back(f, Lane.F32, ly, y, yCount);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        int xn = transposed ? rows : cols, yn = transposed ? cols : rows;
        var ((aLo, aHi), (xLo, xHi)) = RangePair<TN>(a, rows * cols, x, xn);
        var bounds = new Bounds(aLo, aHi, xLo, xHi, xn);
        var lane = SumLane(f, in bounds);
        ulong yl = gpu.Rent((long)yn * lane.Size);
        try
        {
            if (transposed)
            {
                var fc = gpu.Cached($"gemv_cols:{f.Name}:{lane.Name}", (out string e) => Ptx.GemvCols(gpu, f, lane, out e));
                int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
                if (splits == 1) Gpu.Launch(fc, Grid(cols, 256), 1, 256, 1, 0, a, x, yl, Gpu.U(rows), Gpu.U(cols), Gpu.U(per));
                else
                {
                    ulong partials = gpu.Rent((long)splits * cols * lane.Size);
                    try
                    {
                        Gpu.Launch(fc, Grid(cols, 256), (uint)splits, 256, 1, 0, a, x, partials, Gpu.U(rows), Gpu.U(cols), Gpu.U(per));
                        var lf = Ptx.LaneFormat(lane);
                        var f2 = gpu.Cached($"reduce_cols:sum:False:{lf.Name}:{lane.Name}", (out string e) => Ptx.ReduceCols(gpu, lf, lane, "sum", false, out e));
                        Gpu.Launch(f2, Grid(cols, 256), 1, 256, 1, 0, partials, yl, Gpu.U(splits), Gpu.U(cols), Gpu.U(splits));
                    }
                    finally
                    {
                        gpu.Return(partials);
                    }
                }
            }
            else
            {
                var fr = gpu.Cached($"gemv_rows:{f.Name}:{lane.Name}", (out string e) => Ptx.GemvRows(gpu, f, lane, out e));
                Gpu.Launch(fr, Grid(rows, 8), 1, 256, 1, 0, a, x, yl, Gpu.U(rows), Gpu.U(cols));
            }

            Back(f, lane, yl, y, yn);
        }
        finally
        {
            gpu.Return(yl);
        }

        return lane.Name;
    }

    // ---------------------------------------------------------------- element-wise formulas

    /// <summary>output = op(x, y, z; a, b) over n values: the formulas every lane has in the format's natural lane, the float
    /// ones in its float lane (f32 for the small floats, f64 for float64). Absent inputs may be 0.</summary>
    public string Map<TN>(MapOp op, ulong x, ulong y, ulong z, ulong output, int n, double a, double b) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                ulong lx = x == 0 ? 0 : Lanes(f, Lane.F32, x, n, r), ly = y == 0 ? 0 : Lanes(f, Lane.F32, y, n, r), lz = z == 0 ? 0 : Lanes(f, Lane.F32, z, n, r), lo = gpu.Rent(n * 4L);
                r.Add(lo);
                Map<Float32>(op, lx, ly, lz, lo, n, a, b);
                Back(f, Lane.F32, lo, output, n);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        if (op.FloatOnly && !f.IsFloat) throw new NotSupportedException($"{op.Name}: a float formula; {f.Name} is not a float type");
        var lane = f.Natural;
        (a, b) = Kernels.Held<TN>(a, b);
        var fn = gpu.Cached($"map:{op.Name}:{f.Name}:{lane.Name}", (out string e) => Ptx.Map(gpu, f, lane, op, out e));
        ulong pa = lane.IsFloat ? (lane.Wide ? Gpu.D(a) : Gpu.F((float)a)) : lane.Wide ? (ulong)(long)a : (ulong)(uint)(int)a;
        ulong pb = lane.IsFloat ? (lane.Wide ? Gpu.D(b) : Gpu.F((float)b)) : lane.Wide ? (ulong)(long)b : (ulong)(uint)(int)b;
        Gpu.Launch(fn, Grid(f.Words(n), 256), 1, 256, 1, 0, x, y, z, output, Gpu.U(n), pa, pb);
        return lane.Name;
    }

    // ---------------------------------------------------------------- the range of stored values

    public (double Min, double Max) Range<TN>(ulong x, int n) where TN : struct, IQiyasNumber<TN> => RangePair<TN>(x, n, 0, 0).A;

    /// <summary>The ranges of two stored arrays (the operands of a product) with one download.</summary>
    public ((double Min, double Max) A, (double Min, double Max) B) RangePair<TN>(ulong x, int nx, ulong y, int ny) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsFloat || f.Bits <= 8) return ((TN.Min, TN.Max), (TN.Min, TN.Max));   // the type's range decides the lane already
        var fn = gpu.Cached($"range:{f.Name}", (out string e) => Ptx.Range(gpu, f, out e));
        int Blocks(int n) => (int)Math.Min(512, Math.Max(1, (f.Words(n) + Ptx.RangeThreads - 1) / Ptx.RangeThreads));
        int bx = Blocks(nx), by = ny > 0 ? Blocks(ny) : 0;
        ulong outBuffer = gpu.Rent((bx + by) * 16L);
        Gpu.Launch(fn, (uint)bx, 1, Ptx.RangeThreads, 1, 0, x, Gpu.U(nx), outBuffer);
        if (by > 0) Gpu.Launch(fn, (uint)by, 1, Ptx.RangeThreads, 1, 0, y, Gpu.U(ny), outBuffer + (ulong)(bx * 16));
        var pairs = new long[(bx + by) * 2];
        Gpu.Download<long>(outBuffer, pairs);
        gpu.Return(outBuffer);
        (long, long) Join(int from, int count)
        {
            long lo = long.MaxValue, hi = long.MinValue;
            for (int i = from; i < from + count; i++) { lo = Math.Min(lo, pairs[2 * i]); hi = Math.Max(hi, pairs[2 * i + 1]); }
            return (lo, hi);
        }

        var ra = Join(0, bx);
        var rb = by > 0 ? Join(bx, by) : (0L, 0L);
        return ((ra.Item1, ra.Item2), (rb.Item1, rb.Item2));
    }

    // ---------------------------------------------------------------- C = A·B

    /// <summary>The lane the product takes for these bounds, by the same rule as the CPU, on this device's lanes: the tensor
    /// instructions first (exact for every input they are chosen for), then the dot instruction, then the plain lanes.</summary>
    public string ProductLane<TN>(in Bounds bounds) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsFloat)
        {
            if (f.Bits == 64) return "f64";
            if (f.Bits == 32 || f.IsBlock) return "f32";                     // a block's values (an f16 scale times an integer) are exact in f32, not in f16
            var kind = f.Is("bfloat16") ? Ptx.Kind.BF16 : Ptx.Kind.F16;
            return MmaSupported(kind) ? kind.Name + "mma" : "f32";
        }

        double bound = bounds.Bound;
        bool narrow = f.Bits <= 8 || bounds.Within(f.Signed ? sbyte.MinValue : 0, f.Signed ? sbyte.MaxValue : byte.MaxValue);
        if (narrow && bound <= (f.Signed ? int.MaxValue : uint.MaxValue))
        {
            // Values that fit 4 bits may take the 4-bit tensor instruction: whichever of the 4- and 8-bit ones the device
            // runs faster (measured once per shape), where it has them.
            bool tiny = f.Bits <= 4 || bounds.Within(f.Signed ? -8 : 0, f.Signed ? 7 : 15);
            var kind = EightBitKind(f.Signed, bounds.Terms);
            var kind4 = f.Signed ? Ptx.Kind.S4 : Ptx.Kind.U4;
            bool has8 = MmaSupported(kind), has4 = tiny && MmaSupported(kind4);
            if (has4 && (!has8 || (FourBit ?? FasterKind(kind4, kind, bounds.Terms)))) return kind4.Name + "mma";
            if (has8) return kind.Name + "mma";
            if (Dp4aSupported(f.Signed)) return kind.Name + "dp";
        }

        if (bound <= Exact.Float32) return "f32";
        if (bound <= (f.Signed ? int.MaxValue : uint.MaxValue)) return f.Signed ? "s32" : "u32";
        if (bound <= Exact.Float64) return "f64";
        return f.Signed ? "s64" : "u64";
    }

    public readonly record struct MmaConfig(int RowBytes, int Stages, int Bn, bool Swizzle);

    private readonly Dictionary<string, MmaConfig?> _shapes = [];

    /// <summary>The tensor kernel's shape for a kind: of the row widths, pipeline depths and block widths the device's
    /// shared memory takes and its JIT accepts, the one it runs fastest (measured once, on a 2048³ product); null when it has none.</summary>
    /// <summary>The tensor kernel's configurations the device measures (rows of k bytes a stage, stages, columns a block, swizzle).</summary>
    public static readonly MmaConfig[] MmaConfigs =
    [
        new(64, 3, 128, false), new(64, 4, 128, false), new(128, 2, 128, false), new(64, 3, 256, false),
        new(64, 3, 256, true), new(64, 4, 256, true), new(128, 2, 256, true), new(128, 3, 128, true), new(64, 6, 128, true),
    ];

    /// <summary>The gate's override of the measured configuration (null: measured): every configuration is checked, not only
    /// the one a timing picked (a configuration chosen by one run's timing once read past k: see KPadded).</summary>
    internal MmaConfig? ForcedShape;

    private readonly Dictionary<string, bool> _accepted = [];

    private bool Usable(Ptx.Kind kind, MmaConfig cfg)
    {
        string key = $"{kind.Tag}:{cfg}";
        if (_accepted.TryGetValue(key, out bool known)) return known;
        int shared = Ptx.MmaShared(cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle);
        bool ok = shared <= gpu.SharedOptIn && Accepts(Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out string entry), entry);
        if (ok) gpu.AllowShared(gpu.Cached($"mma:{kind.Tag}:{cfg}", (out string e) => Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out e)), shared);
        return _accepted[key] = ok;
    }

    private MmaConfig? MmaShape(Ptx.Kind kind)
    {
        if (ForcedShape is { } forced && _shapes.TryGetValue(kind.Tag, out var measured) && measured is not null && Usable(kind, forced)) return forced;
        if (_shapes.TryGetValue(kind.Tag, out var known)) return known;
        MmaConfig? best = null;
        double bestMs = double.MaxValue;
        var report = new List<string>();
        const int Size = 2048;
        int kp = Ptx.Padded(Size);
        ulong x = gpu.Rent((long)Size * kp * kind.Bits / 8), y = gpu.Rent((long)Size * kp * kind.Bits / 8), z = gpu.Rent((long)Size * Size * 4);
        foreach (var cfg in MmaConfigs)
        {
            int shared = Ptx.MmaShared(cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle);
            if (shared > gpu.SharedOptIn) continue;
            string ptx = Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out string entry);
            if (!Accepts(ptx, entry)) continue;
            var fn = gpu.Cached($"mma:{kind.Tag}:{cfg}", (out string e) => Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out e));
            gpu.AllowShared(fn, shared);
            double ms = TimeMma(fn, Dynamic(shared), cfg.Bn, x, y, z, Size, kp);
            report.Add($"rows {cfg.RowBytes} B × {cfg.Stages} stages × {cfg.Bn} wide{(cfg.Swizzle ? " swizzled" : "")} {ms:0.000} ms");
            if (ms < bestMs) (bestMs, best) = (ms, cfg);
        }

        gpu.Return(x); gpu.Return(y); gpu.Return(z);
        if (best is { } shape) Note($"{kind.Tag} tensor kernel at {Size}³: {string.Join(", ", report)}: rows {shape.RowBytes} B × {shape.Stages} stages × {shape.Bn} wide{(shape.Swizzle ? " swizzled" : "")} kept");
        return _shapes[kind.Tag] = best;
    }

    /// <summary>k padded for a tensor kind's kernel: whole 128-value tiles and whole k-chunks of the configuration the device
    /// chose (a chunk is RowBytes of a row: 256 values for a 4-bit kind at 128 bytes — a k padded to 128 only would read a
    /// row's chunk into the next row's values).</summary>
    private int KPadded(Ptx.Kind kind, int k)
    {
        int rowBytes = _shapes.ContainsKey(kind.Tag) && MmaShape(kind) is { } c ? c.RowBytes : 64;       // a kind not measured has no kernel chosen yet
        int chunk = Math.Max(128, rowBytes * 8 / kind.Bits);
        return (k + chunk - 1) / chunk * chunk;
    }

    private IntPtr MmaFunction(Ptx.Kind kind, out uint shared, out int bn, Fmt? store = null, bool kMajorA = false, bool kMajorB = false)
    {
        var cfg = MmaShape(kind) ?? throw new InvalidOperationException($"{kind.Name}: no tensor kernel on this device");
        shared = Dynamic(Ptx.MmaShared(cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle));
        bn = cfg.Bn;
        var fn = gpu.Cached($"mma:{kind.Tag}:{cfg}:{store?.Name}:{kMajorA}:{kMajorB}", (out string e) => Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out e, store, kMajorA, kMajorB));
        gpu.AllowShared(fn, Ptx.MmaShared(cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle));
        return fn;
    }

    /// <summary>The gate's override: false stores a 16-bit float product's C in f32 and converts it after (the pass the
    /// epilogue replaces).</summary>
    public bool? Epilogue16;

    /// <summary>The gate's override: false packs a 16-bit float product's operands even where the tensor kernel reads them
    /// as stored (the passes the in-place reads replace).</summary>
    public bool? InPlace16;

    // Shared bytes to ask for at launch: only what a kernel declares as extern (beyond the static limit).
    private static uint Dynamic(int bytes) => bytes > Ptx.StaticShared ? (uint)bytes : 0;

    private readonly Dictionary<string, bool> _supported = [];

    private bool MmaSupported(Ptx.Kind kind) => MmaShape(kind) is not null;

    private readonly Dictionary<string, Ptx.Kind> _eightBit = [];

    /// <summary>The gates' overrides of the measured tensor-kind choices (null: measured): the 4-bit kinds over the 8-bit ones
    /// where values fit, and the 32-deep 8-bit instruction over the 16-deep one — so every kind the device accepts is
    /// checked, not only the one a timing happened to pick.</summary>
    internal bool? FourBit, K32;

    // The 8-bit tensor instruction the device runs faster: one taking 16 or 32 values of k (measured once).
    private Ptx.Kind EightBitKind(bool signed, long terms)
    {
        string key = $"{signed}:{K32}";
        if (_eightBit.TryGetValue(key, out var known)) return known;
        var k16 = signed ? Ptx.Kind.S8 : Ptx.Kind.U8;
        var k32 = signed ? Ptx.Kind.S8K32 : Ptx.Kind.U8K32;
        bool has16 = MmaSupported(k16), has32 = MmaSupported(k32);
        var kind = has32 && (!has16 || (K32 ?? FasterKind(k32, k16, terms))) ? k32 : k16;
        return _eightBit[key] = kind;
    }

    private readonly Dictionary<string, bool> _faster = [];

    // Whether the first kind's kernel beats the second's on a square product of this many terms, measured once.
    private bool FasterKind(Ptx.Kind a, Ptx.Kind b, long terms)
    {
        string key = $"{a.Tag}<{b.Tag}:{terms}";
        if (_faster.TryGetValue(key, out bool faster)) return faster;
        int size = (int)Math.Clamp(terms, 256, 2048), kp = Math.Max(KPadded(a, size), KPadded(b, size));     // at least the kernels' widest tile
        int bits = Math.Max(a.Bits, b.Bits);
        ulong x = gpu.Rent((long)size * kp * bits / 8), y = gpu.Rent((long)size * kp * bits / 8), z = gpu.Rent((long)size * size * 4);
        var fa = MmaFunction(a, out uint sa, out int bna);
        var fb = MmaFunction(b, out uint sb, out int bnb);
        double ta = TimeMma(fa, sa, bna, x, y, z, size, kp), tb = TimeMma(fb, sb, bnb, x, y, z, size, kp);
        gpu.Return(x); gpu.Return(y); gpu.Return(z);
        Note($"{a.Tag} {ta:0.000} ms against {b.Tag} {tb:0.000} ms at {size}³ (k = {kp}): {(ta < tb ? a.Tag : b.Tag)}");
        return _faster[key] = ta < tb;
    }

    private double TimeMma(IntPtr fn, uint shared, int bn, ulong x, ulong y, ulong z, int size, int kp)
    {
        Gpu.Launch(fn, Grid(size, bn), Grid(size, 128), Ptx.GemmThreads, 1, shared, x, y, z, Gpu.U(size), Gpu.U(size), Gpu.U(kp), Gpu.U(kp));
        Gpu.Synchronize();
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < 3; i++) Gpu.Launch(fn, Grid(size, bn), Grid(size, 128), Ptx.GemmThreads, 1, shared, x, y, z, Gpu.U(size), Gpu.U(size), Gpu.U(kp), Gpu.U(kp));
        Gpu.Synchronize();
        return Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
    }

    private bool Dp4aSupported(bool signed)
    {
        if (_supported.TryGetValue("dp4a:" + signed, out bool known)) return known;
        return _supported["dp4a:" + signed] = Accepts(Ptx.GemmDot(gpu, signed, out string entry), entry);
    }

    private static Lane LaneOf(string name) => name switch
    {
        "f32" or "f16mma" or "bf16mma" => Lane.F32,
        "f64" => Lane.F64,
        "s32" or "s8dp" or "s8mma" or "s4mma" => Lane.S32,
        "u32" or "u8dp" or "u8mma" or "u4mma" => Lane.U32,
        "s64" => Lane.S64,
        _ => Lane.U64,
    };

    /// <summary>Whether MatMul takes operands stored transposed at this shape and these bounds: the tensor or dot route
    /// (its packing reads either orientation) at whole tiles.</summary>
    public bool PackedRoute<TN>(in Bounds bounds, int m, int n, int k) where TN : struct, IQiyasNumber<TN>
    {
        if (m % 128 != 0 || n % 256 != 0 || k % 128 != 0) return false;
        string laneName = ProductLane<TN>(bounds);
        return laneName.EndsWith("mma", StringComparison.Ordinal) || laneName.EndsWith("dp", StringComparison.Ordinal);
    }

    /// <summary>C = A·B (A m × k, B k × n); with transA A is stored k × m (Aᵀ), with transB B is stored n × k (Bᵀ), read
    /// that way by the packing (only where PackedRoute says so).</summary>
    public string MatMul<TN>(ulong a, ulong b, ulong c, int m, int n, int k, in Bounds bounds, bool transA = false, bool transB = false) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if ((transA || transB) && !PackedRoute<TN>(bounds, m, n, k)) throw new InvalidOperationException("transposed operands are read in place by the tensor and dot routes at whole tiles only");
        if (m % 128 != 0 || n % 256 != 0 || k % 128 != 0) return MatMulPadded<TN>(a, b, c, m, n, k, in bounds);
        string laneName = ProductLane<TN>(bounds);
        var lane = LaneOf(laneName);
        var rented = new List<ulong>();
        try
        {
            if (laneName.EndsWith("mma", StringComparison.Ordinal) || laneName.EndsWith("dp", StringComparison.Ordinal))
            {
                var kind = KindOf(laneName, k);
                int kp = KPadded(kind, k);
                // A's rows as stored already have the kind's layout when the type is the kind and k is whole blocks.
                bool aAsIs = f.Bits == kind.Bits && kp == k && (f.IsFloat == kind.IsFloat) && (!f.IsFloat || f.Is(kind == Ptx.Kind.F16 ? "float16" : "bfloat16"));
                long t0 = Phases is null ? 0 : Stopwatch.GetTimestamp();
                // A 16-bit float type in its own kind at whole k-blocks: the tensor kernel reads both operands as stored (Aᵀ and
                // B k-major by ldmatrix.trans, Bᵀ as its packed rows), no packing pass.
                bool inPlace = InPlace16 != false && aAsIs && kind.Bits == 16 && laneName.EndsWith("mma", StringComparison.Ordinal);
                bool kMajorA = inPlace && transA, kMajorB = inPlace && !transB;
                ulong pa = (aAsIs && !transA) || kMajorA ? a : Pack(f, kind, a, m, k, kp, transposed: transA, rented);
                t0 = Mark(t0, 0);
                ulong pb = inPlace ? b : Pack(f, kind, b, n, k, kp, transposed: !transB, rented);
                t0 = Mark(t0, 1);
                bool cAsIs = f.Bits == 32 && f.IsFloat == lane.IsFloat;
                if (laneName.EndsWith("mma", StringComparison.Ordinal) && Epilogue16 != false && (f.Is("float16") || f.Is("bfloat16")))
                {
                    // The 16-bit epilogue: C stored as the type by the kernel (no f32 C, no conversion pass).
                    var fe = MmaFunction(kind, out uint sharedE, out int bnE, f, kMajorA, kMajorB);
                    Gpu.Launch(fe, Grid(n, bnE), Grid(m, 128), Ptx.GemmThreads, 1, sharedE, pa, pb, c, Gpu.U(m), Gpu.U(n), Gpu.U(kp), Gpu.U(kp));
                    Gpu.Flush();                                       // the product submitted now, not batched behind later launches
                    Mark(Mark(t0, 2), 3);
                    return laneName;
                }

                ulong lc = cAsIs ? c : gpu.Rent((long)m * n * 4);
                if (!cAsIs) rented.Add(lc);
                if (laneName.EndsWith("mma", StringComparison.Ordinal))
                {
                    var fn = MmaFunction(kind, out uint shared, out int bn, null, kMajorA, kMajorB);
                    Gpu.Launch(fn, Grid(n, bn), Grid(m, 128), Ptx.GemmThreads, 1, shared, pa, pb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(kp), Gpu.U(kp));
                    Gpu.Flush();
                }
                else
                {
                    var fn = gpu.Cached($"dp4a:{f.Signed}", (out string e) => Ptx.GemmDot(gpu, f.Signed, out e));
                    Gpu.Launch(fn, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, 0, pa, pb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(kp / 4));
                    Gpu.Flush();
                }

                t0 = Mark(t0, 2);
                if (!cAsIs) Back(f, lane, lc, c, (long)m * n);
                Mark(t0, 3);
                return laneName;
            }

            // The sliced route's kernels take shapes in multiples of 8 (their slices and tiles); other shapes take the float64 lane.
            if (f.Is("float64") && SlicedProven && m % 8 == 0 && n % 8 == 0 && k % 8 == 0 && SlicedFaster(m, n, k, a, b, c) && MatMulSliced(a, b, c, m, n, k)) return "f64x";
            bool direct = f.Natural == lane && f.Bits == lane.Size * 8 && (f.IsFloat == lane.IsFloat);
            ulong la = direct ? a : Lanes(f, lane, a, (long)m * k, rented);
            ulong lb = direct ? b : Lanes(f, lane, b, (long)k * n, rented);
            ulong lcl = direct ? c : gpu.Rent((long)m * n * lane.Size);
            if (!direct) rented.Add(lcl);
            // The staged kernel where the device builds it (cp.async), its pipeline depth measured once per lane; 0: none.
            int stages = Staged != false && !lane.Wide && k % 4 == 0 && n % 4 == 0 ? Gemm2Stages(lane, la, lb, lcl, m, n, k) : 0;
            if (stages > 0)
            {
                int slices = SplitK(m, n, k, out int sliceBlocks);
                if (slices > 1)
                {
                    // Split-k: too few tiles of C to fill the device; the slices' partial C joined in order.
                    var gk = gpu.Cached($"gemm2:{lane.Name}:{stages}:k", (out string e) => Ptx.Gemm2(gpu, lane, stages, out e, split: true));
                    gpu.AllowShared(gk, Ptx.Gemm2Shared(stages));
                    ulong parts = gpu.Rent((long)slices * m * n * lane.Size);
                    rented.Add(parts);
                    Gpu.Launch(gk, Grid(n, 128), Grid(m, 128) * (uint)slices, Ptx.GemmThreads, 1, Dynamic(Ptx.Gemm2Shared(stages)), la, lb, parts, Gpu.U(m), Gpu.U(n), Gpu.U(k), Gpu.U((int)Grid(m, 128)), Gpu.U(sliceBlocks));
                    Gpu.Flush();
                    var join = gpu.Cached($"sum_slices:{lane.Name}", (out string e) => Ptx.SumSlices(gpu, lane, out e));
                    Gpu.Launch(join, Grid((long)m * n / 4, 256), 1, 256, 1, 0, parts, lcl, Gpu.U(m * n), Gpu.U(slices));
                }
                else
                {
                    var g2 = gpu.Cached($"gemm2:{lane.Name}:{stages}", (out string e) => Ptx.Gemm2(gpu, lane, stages, out e));
                    gpu.AllowShared(g2, Ptx.Gemm2Shared(stages));
                    Gpu.Launch(g2, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, Dynamic(Ptx.Gemm2Shared(stages)), la, lb, lcl, Gpu.U(m), Gpu.U(n), Gpu.U(k));
                    Gpu.Flush();
                }
            }
            else
            {
                int micro = Tile(lane, m, n, k, la, lb, lcl);
                var g = gpu.Cached($"gemm:{lane.Name}:{micro}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e));
                int tile = micro * 16;
                Gpu.Launch(g, Grid(n, tile), Grid(m, tile), Ptx.GemmThreads, 1, 0, la, lb, lcl, Gpu.U(m), Gpu.U(n), Gpu.U(k));
                Gpu.Flush();
            }
            if (!direct) Back(f, lane, lcl, c, (long)m * n);
            return laneName;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>The gate's override: false takes the register-tiled kernel as on a device that refuses the staged one.</summary>
    internal bool? Staged;

    /// <summary>The gate's override of split-k's slice count (null: SplitK's rule).</summary>
    public int? SplitSlices;

    /// <summary>Split-k's slices for a staged product (item 6j): 1 unless C's 128 × 128 tiles are fewer than the device's
    /// multiprocessors and k is at least 1024; then enough slices for two tiles a multiprocessor, each at least 256 of k
    /// (whole 16-value blocks: sliceBlocks a slice).</summary>
    private int SplitK(int m, int n, int k, out int sliceBlocks)
    {
        int tiles = (int)(Grid(n, 128) * Grid(m, 128)), kBlocks = (k + 15) / 16, sms = gpu.Multiprocessors;
        int slices = SplitSlices ?? (tiles >= sms || k < 1024 ? 1 : Math.Min(k / 256, (2 * sms + tiles - 1) / tiles));
        slices = Math.Clamp(slices, 1, kBlocks);
        sliceBlocks = (kBlocks + slices - 1) / slices;
        slices = (kBlocks + sliceBlocks - 1) / sliceBlocks;
        return slices;
    }

    // The tensor or dot kind of a packed lane ("s8mma", "u8dp", …) at k terms.
    private Ptx.Kind KindOf(string laneName, int k) => laneName[..^(laneName.EndsWith("mma", StringComparison.Ordinal) ? 3 : 2)] switch
    {
        "s8" => laneName.EndsWith("mma", StringComparison.Ordinal) ? EightBitKind(true, k) : Ptx.Kind.S8,
        "u8" => laneName.EndsWith("mma", StringComparison.Ordinal) ? EightBitKind(false, k) : Ptx.Kind.U8,
        "s4" => Ptx.Kind.S4, "u4" => Ptx.Kind.U4, "f16" => Ptx.Kind.F16, _ => Ptx.Kind.BF16,
    };

    /// <summary>A product whose shape the kernels' tiles and vector loads cannot take as stored: A padded to mp × kp, B to
    /// kp × np (multiples of 128; zeros beyond m, k and n: the sums are unchanged), the product at the padded shape, C's
    /// m × n taken from mp × np.</summary>
    private string MatMulPadded<TN>(ulong a, ulong b, ulong c, int m, int n, int k, in Bounds bounds) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        int mp = (m + 127) / 128 * 128, kp = (k + 127) / 128 * 128, np = (n + 255) / 256 * 256;   // the tensor kernels' widest tile is 256 columns
        long bytesA = f.Words((long)mp * kp) * (f.WordBits / 8), bytesB = f.Words((long)kp * np) * (f.WordBits / 8), bytesC = f.Words((long)mp * np) * (f.WordBits / 8);
        ulong pa = gpu.Rent(bytesA), pb = gpu.Rent(bytesB), pc = gpu.Rent(bytesC);
        try
        {
            Gpu.Clear(pa, bytesA);
            Layout<TN>(new LayoutArgs { Name = "pad2d", X = [], P = [m, k, kp], Output = [], Count = m * kp, InputCount = m * k }, a, 0, pa);
            Gpu.Clear(pb, bytesB);
            Layout<TN>(new LayoutArgs { Name = "pad2d", X = [], P = [k, n, np], Output = [], Count = k * np, InputCount = k * n }, b, 0, pb);
            string lane = MatMul<TN>(pa, pb, pc, mp, np, kp, in bounds);
            Layout<TN>(new LayoutArgs { Name = "strided", X = [], P = [1, 1, m, n, 0, 0, np, 1, 0], Output = [], Count = m * n, InputCount = mp * np }, pc, 0, c);
            return lane;
        }
        finally
        {
            gpu.Return(pa); gpu.Return(pb); gpu.Return(pc);
        }
    }

    // ---------------------------------------------------------------- float64 products through the 8-bit tensor lane

    /// <summary>Set by the device proof (Checks.Gpu): the sliced product reproduces float64 math to within an ulp on this device.</summary>
    public bool SlicedProven { get; set; }

    private readonly Dictionary<string, bool> _slicedFaster = [];

    /// <summary>Whether the sliced product beats the plain float64 kernel at this shape (measured once per shape).</summary>
    private bool SlicedFaster(int m, int n, int k, ulong a, ulong b, ulong c)
    {
        string key = $"{m}x{n}x{k}";
        if (_slicedFaster.TryGetValue(key, out bool known)) return known;
        double Time(Action run)
        {
            run();
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) run();
            Gpu.Synchronize();
            return Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
        }

        double plain = Time(() => PlainF64(a, b, c, m, n, k));
        bool refused = false;
        double sliced = Time(() => { if (!MatMulSliced(a, b, c, m, n, k)) refused = true; });
        if (refused) { Note($"sliced float64 product refused at {key} (exponents beyond the slicing scale): the f64 lane"); return _slicedFaster[key] = false; }
        Note($"float64 product at {key}: f64 lane {plain:0.000} ms, sliced {sliced:0.000} ms: {(sliced < plain ? "sliced" : "f64 lane")}");
        return _slicedFaster[key] = sliced < plain;
    }

    private void PlainF64(ulong a, ulong b, ulong c, int m, int n, int k)
    {
        var lane = Lane.F64;
        var g = gpu.Cached($"gemm:{lane.Name}:4", (out string e) => Ptx.Gemm(gpu, lane, 4, out e));
        Gpu.Launch(g, Grid(n, 64), Grid(m, 64), Ptx.GemmThreads, 1, 0, a, b, c, Gpu.U(m), Gpu.U(n), Gpu.U(k));
    }

    /// <summary>C = A·B for float64 operands through the 8-bit tensor instructions (see Ptx: sliced float64). False when the
    /// device has no 8-bit tensor kind or the data's exponents are beyond what the slicing scale takes: nothing was computed.</summary>
    public bool MatMulSliced(ulong a, ulong b, ulong c, int m, int n, int k)
    {
        if (!MmaSupported(Ptx.Kind.S8) && !MmaSupported(Ptx.Kind.S8K32)) return false;
        var kind = EightBitKind(true, k);
        int kpad = Ptx.Padded(k), rowLen = Ptx.Slices * kpad;
        long mn = (long)m * n;
        ulong e = gpu.Rent(m * 4L), f = gpu.Rent(n * 4L), sa = gpu.Rent((long)m * rowLen), stack = gpu.Rent((long)rowLen * n), bt = gpu.Rent((long)n * rowLen), d = gpu.Rent(Ptx.Diagonals * mn * 4);
        try
        {
            var rowExp = gpu.Cached("rowexp_f64", (out string x) => Ptx.RowExponents(gpu, out x));
            var colExp = gpu.Cached("colexp_f64", (out string x) => Ptx.ColumnExponents(gpu, out x));
            var sliceRows = gpu.Cached("slice_rows_f64", (out string x) => Ptx.SliceRows(gpu, out x));
            var sliceCols = gpu.Cached("slice_cols_f64", (out string x) => Ptx.SliceColumns(gpu, out x));
            var combine = gpu.Cached("combine_f64x", (out string x) => Ptx.CombineSlices(gpu, out x));
            Gpu.Launch(rowExp, (uint)m, 1, 256, 1, 0, a, e, Gpu.U(k));
            Gpu.Launch(colExp, Grid(n, 256), 1, 256, 1, 0, b, f, Gpu.U(k), Gpu.U(n));
            // The exponents must keep every scale a normal float64 (the slicing scale 2^(64 − e), the result's 2^(e + f + shift − 72)).
            var ea = new int[m];
            var fa = new int[n];
            Gpu.Download<int>(e, ea);
            Gpu.Download<int>(f, fa);
            // A row or column of zeros has no exponent (far below any scale): it takes 0, any finite scale slicing it to zeros.
            bool zeroRows = false, zeroCols = false;
            for (int i = 0; i < m; i++) if (ea[i] < -900) { ea[i] = 0; zeroRows = true; }
            for (int i = 0; i < n; i++) if (fa[i] < -900) { fa[i] = 0; zeroCols = true; }
            if (zeroRows) Gpu.Upload<int>(ea, e);
            if (zeroCols) Gpu.Upload<int>(fa, f);
            int eMin = ea.Min(), eMax = ea.Max(), fMin = fa.Min(), fMax = fa.Max();
            if (eMax > 1086 || fMax > 1086 || eMin + fMin < -918 || eMax + fMax > 1057) return false;
            if (kpad != k) Gpu.Clear(sa, (long)m * rowLen);
            Gpu.Clear(stack, (long)rowLen * n);
            Gpu.Launch(sliceRows, Grid((long)m * k, 256), 1, 256, 1, 0, a, e, sa, Gpu.U(m), Gpu.U(k), Gpu.U(kpad));
            Gpu.Launch(sliceCols, Grid((long)k * n, 256), 1, 256, 1, 0, b, f, stack, Gpu.U(k), Gpu.U(n), Gpu.U(kpad));
            // The stacked digits of B (rowLen × n, signed bytes) transposed into packed rows Bt[j][0..rowLen).
            var int8 = new Fmt("int8", 8, false, true);
            var ft = gpu.Cached($"packt:{int8.Name}:{kind.Name}", (out string x) => Ptx.PackTransposed(gpu, int8, kind, out x));
            Gpu.Launch(ft, Grid(n, 64), (uint)(rowLen / 32), 256, 1, 0, stack, bt, Gpu.U(n), Gpu.U(rowLen), Gpu.U(rowLen));
            // One product per diagonal d: the digit pairs (t, u) with t + u = d. B's digits are stacked most significant last,
            // so for d below the slice count A's first d + 1 slices meet B's last d + 1; above it, A's last 15 − d meet B's first.
            var fn = MmaFunction(kind, out uint shared, out int bn);
            for (int diag = 0; diag < Ptx.Diagonals; diag++)
            {
                int pairs = diag < Ptx.Slices ? diag + 1 : 2 * Ptx.Slices - 1 - diag, extent = pairs * kpad;
                ulong pa = diag < Ptx.Slices ? sa : sa + (ulong)((diag - Ptx.Slices + 1) * kpad);
                ulong pb = diag < Ptx.Slices ? bt + (ulong)((Ptx.Slices - 1 - diag) * kpad) : bt;
                ulong pd = d + (ulong)(diag * mn * 4);
                Gpu.Launch(fn, Grid(n, bn), Grid(m, 128), Ptx.GemmThreads, 1, shared, pa, pb, pd, Gpu.U(m), Gpu.U(n), Gpu.U(extent), Gpu.U(rowLen));
            }

            Gpu.Launch(combine, Grid(mn, 256), 1, 256, 1, 0, d, e, f, c, Gpu.U(m), Gpu.U(n));
            return true;
        }
        finally
        {
            foreach (var p in new[] { e, f, sa, stack, bt, d }) gpu.Return(p);
        }
    }

    private ulong Pack(Fmt f, Ptx.Kind kind, ulong stored, int rows, int k, int kp, bool transposed, List<ulong> rented)
    {
        ulong packed = gpu.Rent((long)rows * kp * kind.Bits / 8);
        rented.Add(packed);
        if (transposed && f.Bits == 16 && kind.Bits == 16 && f.Is(kind == Ptx.Kind.F16 ? "float16" : "bfloat16") && rows % 2 == 0)
        {
            // A 16-bit type in its own kind: the word-wide transposing tile.
            var f16 = gpu.Cached($"packt16:{f.Name}:{kind.Name}", (out string e) => Ptx.PackTransposed16(gpu, f, kind, out e));
            Gpu.Launch(f16, Grid(rows, 64), (uint)((kp + 63) / 64), 256, 1, 0, stored, packed, Gpu.U(rows), Gpu.U(k), Gpu.U(kp));
            return packed;
        }

        if (transposed)
        {
            var ft = gpu.Cached($"packt:{f.Name}:{kind.Name}", (out string e) => Ptx.PackTransposed(gpu, f, kind, out e));
            Gpu.Launch(ft, Grid(rows, 64), (uint)(kp / 32), 256, 1, 0, stored, packed, Gpu.U(rows), Gpu.U(k), Gpu.U(kp));
            return packed;
        }

        var fn = gpu.Cached($"pack:{f.Name}:{kind.Name}:n", (out string e) => Ptx.PackRows(gpu, f, kind, false, out e));
        long words = (long)rows * kp / kind.Per;
        Gpu.Launch(fn, Grid(words, 256), 1, 256, 1, 0, stored, packed, Gpu.U(rows), Gpu.U(k), Gpu.U(kp), Gpu.U(k));
        return packed;
    }

    private readonly Dictionary<string, int> _gemm2Stages = [];

    // The pipeline depth (2 or 3 stages) the device runs faster for this lane, measured once on the first product's shape.
    private int Gemm2Stages(Lane lane, ulong la, ulong lb, ulong lc, int m, int n, int k)
    {
        if (_gemm2Stages.TryGetValue(lane.Name, out int known)) return known;
        int best = 0;                                                    // no depth the device takes: the register-tiled kernel
        double bestMs = double.MaxValue;
        var report = new List<string>();
        foreach (int stages in new[] { 2, 3 })
        {
            int shared = Ptx.Gemm2Shared(stages);
            if (shared > gpu.SharedOptIn) continue;
            string ptx = Ptx.Gemm2(gpu, lane, stages, out string entry);
            if (!Accepts(ptx, entry)) continue;
            var fn = gpu.Cached($"gemm2:{lane.Name}:{stages}", (out string e) => Ptx.Gemm2(gpu, lane, stages, out e));
            gpu.AllowShared(fn, shared);
            Gpu.Launch(fn, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, Dynamic(shared), la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) Gpu.Launch(fn, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, Dynamic(shared), la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            report.Add($"{stages} stages {ms:0.000} ms");
            if (ms < bestMs) (bestMs, best) = (ms, stages);
        }

        Note(best > 0 ? $"{lane.Name} staged product at {m}x{n}x{k}: {string.Join(", ", report)}: {best} stages kept"
            : $"no staged {lane.Name} product (its shared memory or asynchronous copies): the register-tiled kernel");
        return _gemm2Stages[lane.Name] = best;
    }

    // The faster of the 128 and 64 tiles for this lane and shape, measured on the device once (a 64-bit lane's
    // accumulators do not fit the 128 tile's registers: 64 only).
    private int Tile(Lane lane, int m, int n, int k, ulong la, ulong lb, ulong lc)
    {
        if (lane.Wide) return 4;
        string key = $"{lane.Name}:{m}x{n}x{k}";
        if (_tiles.TryGetValue(key, out int micro)) return micro;
        double best = double.MaxValue;
        foreach (int candidate in new[] { 8, 4 })
        {
            var fn = gpu.Cached($"gemm:{lane.Name}:{candidate}", (out string e) => Ptx.Gemm(gpu, lane, candidate, out e));
            int tile = candidate * 16;
            Gpu.Launch(fn, Grid(n, tile), Grid(m, tile), Ptx.GemmThreads, 1, 0, la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) Gpu.Launch(fn, Grid(n, tile), Grid(m, tile), Ptx.GemmThreads, 1, 0, la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            if (ms < best) (best, micro) = (ms, candidate);
        }

        _tiles[key] = micro;
        return micro;
    }

    private ulong Lanes(Fmt f, Lane lane, ulong stored, long count, List<ulong> rented)
    {
        ulong buffer = gpu.Rent(count * lane.Size);
        rented.Add(buffer);
        if (f.IsBlock)
        {
            var widen = gpu.Cached($"widen:{f.Name}:{lane.Name}", (out string e) => Ptx.Widen(gpu, f, lane, out e));
            Gpu.Launch(widen, Grid(count, 256), 1, 256, 1, 0, stored, buffer, Gpu.U((int)count));
            return buffer;
        }

        var fn = gpu.Cached($"to:{f.Name}:{lane.Name}", (out string e) => Ptx.ToLanes(gpu, f, lane, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, stored, buffer, Gpu.U((int)count));
        return buffer;
    }

    private void Back(Fmt f, Lane lane, ulong lanes, ulong stored, long count)
    {
        if (f.IsBlock)
        {
            var encode = gpu.Cached($"lanes_to:{f.Name}:{lane.Name}", (out string e) => Ptx.LanesToBlock(gpu, f, lane, out e));
            Gpu.Launch(encode, Grid((count + f.BlockValues - 1) / f.BlockValues, 256), 1, 256, 1, 0, lanes, stored, Gpu.U((int)count));
            return;
        }

        var fn = gpu.Cached($"back:{lane.Name}:{f.Name}", (out string e) => Ptx.FromLanes(gpu, lane, f, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, lanes, stored, Gpu.U((int)count));
    }

    /// <summary>A stored array into a lane array (for checks).</summary>
    public void ToLanes<TN>(Lane lane, ulong stored, ulong lanes, int count) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var widen = gpu.Cached($"widen:{f.Name}:{lane.Name}", (out string e) => Ptx.Widen(gpu, f, lane, out e));
            Gpu.Launch(widen, Grid(count, 256), 1, 256, 1, 0, stored, lanes, Gpu.U(count));
            return;
        }

        var fn = gpu.Cached($"to:{f.Name}:{lane.Name}", (out string e) => Ptx.ToLanes(gpu, f, lane, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, stored, lanes, Gpu.U(count));
    }

    public void FromLanes<TN>(Lane lane, ulong lanes, ulong stored, int count) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) { Back(f, lane, lanes, stored, count); return; }
        var fn = gpu.Cached($"back:{lane.Name}:{f.Name}", (out string e) => Ptx.FromLanes(gpu, lane, f, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, lanes, stored, Gpu.U(count));
    }

    // ---------------------------------------------------------------- rows

    private static Lane RowLane(Fmt f) => f.Bits == 64 ? Lane.F64 : Lane.F32;

    /// <summary>An arena scope on this device (src/Arena.cs): the operations called until it is disposed take their buffers
    /// from one region, bumped and released from the top.</summary>
    public Gpu.ArenaScope Scope() => gpu.Scope();

    // A row's block: enough threads (128 to 1024) that each holds about four words; the words per thread (a row in
    // registers up to 16; longer rows take the three-pass kernels).
    private static (int Threads, int WordsPerThread) RowShape(Fmt f, int cols)
    {
        long words = f.Words(cols);
        int threads = 128;
        while (threads < 1024 && (words + threads - 1) / threads > 4) threads *= 2;
        return (threads, (int)((words + threads - 1) / threads));
    }

    private static void CheckRows(Fmt f, int cols)
    {
        if ((long)cols * f.Bits % f.WordBits != 0) throw new NotSupportedException($"{f.Name}: a row of {cols} values does not end on a word; the device's row kernels take whole words.");
    }

    /// <summary>A block format's row operation: the rows in float32 storage, the result encoded by blocks.</summary>
    private string RowsThroughLanes(Fmt f, ulong x, ulong y, int rows, int cols, Func<ulong, ulong, string> run)
    {
        var r = new List<ulong>();
        try
        {
            long n = (long)rows * cols;
            ulong lx = Lanes(f, Lane.F32, x, n, r), ly = gpu.Rent(n * 4);
            r.Add(ly);
            run(lx, ly);
            Back(f, Lane.F32, ly, y, n);
            return "f32 (blocks through lane storage)";
        }
        finally { foreach (var p in r) gpu.Return(p); }
    }

    public string Softmax<TN>(ulong x, ulong y, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) return RowsThroughLanes(f, x, y, rows, cols, (lx, ly) => Softmax<Float32>(lx, ly, rows, cols));
        CheckRows(f, cols);
        var lane = RowLane(f);
        var (threads, perThread) = RowShape(f, cols);
        bool vector = f.WordBits == 32 && perThread == 4 && f.Words(cols) % 4 == 0;
        var fn = perThread <= 16
            ? gpu.Cached($"softmax:{f.Name}:{perThread}:{threads}:{vector}", (out string e) => Ptx.SoftmaxRegs(gpu, f, lane, perThread, threads, vector, out e))
            : gpu.Cached($"softmaxloop:{f.Name}:{threads}", (out string e) => Ptx.Softmax(gpu, f, lane, threads, out e));
        if (threads > Gpu.MaxThreads(fn)) fn = gpu.Cached($"softmaxloop:{f.Name}:{threads}", (out string e) => Ptx.Softmax(gpu, f, lane, threads, out e));
        if (threads > Gpu.MaxThreads(fn)) { threads = 256; fn = gpu.Cached($"softmaxloop:{f.Name}:{threads}", (out string e) => Ptx.Softmax(gpu, f, lane, threads, out e)); }
        Gpu.Launch(fn, (uint)rows, 1, (uint)threads, 1, 0, x, y, Gpu.U(cols));
        return lane.Name;
    }

    public string RmsNorm<TN>(ulong x, ulong y, int rows, int cols, double eps) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) return RowsThroughLanes(f, x, y, rows, cols, (lx, ly) => RmsNorm<Float32>(lx, ly, rows, cols, eps));
        CheckRows(f, cols);
        var lane = RowLane(f);
        var (threads, perThread) = RowShape(f, cols);
        bool vector = f.WordBits == 32 && perThread == 4 && f.Words(cols) % 4 == 0;
        var fn = perThread <= 16
            ? gpu.Cached($"rms:{f.Name}:{perThread}:{threads}:{vector}", (out string e) => Ptx.RmsNormRegs(gpu, f, lane, perThread, threads, vector, out e))
            : gpu.Cached($"rmsloop:{f.Name}:{threads}", (out string e) => Ptx.RmsNorm(gpu, f, lane, threads, out e));
        if (threads > Gpu.MaxThreads(fn)) fn = gpu.Cached($"rmsloop:{f.Name}:{threads}", (out string e) => Ptx.RmsNorm(gpu, f, lane, threads, out e));
        if (threads > Gpu.MaxThreads(fn)) { threads = 256; fn = gpu.Cached($"rmsloop:{f.Name}:{threads}", (out string e) => Ptx.RmsNorm(gpu, f, lane, threads, out e)); }
        Gpu.Launch(fn, (uint)rows, 1, (uint)threads, 1, 0, x, y, Gpu.U(cols), lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps));
        return lane.Name;
    }
}
