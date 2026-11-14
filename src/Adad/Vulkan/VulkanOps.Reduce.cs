// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Reductions, transposes and matrix·vector products on Vulkan, by CUDA's lane rules (Cuda/GpuOps.cs): a sum's lane from
// the bounds of its terms (f32 while exact, the 32-bit integers, f64, the 64-bit integers), max/min/argmax in the type's
// own lane; a narrow float's whole sum compensated in f32 and its chunks joined in f64, its row sums joined as exact f32
// pairs, each rounded once to the f32 lane.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The lane a sum takes on this device for the bounds (CUDA's rule, on the same lanes).</summary>
    public static Lane SumLane(Fmt f, in Bounds bounds)
    {
        if (f.IsFloat) return f.Bits == 64 ? Lane.F64 : Lane.F32;
        double bound = bounds.Bound;
        if (bound <= Exact.Float32) return Lane.F32;
        if (bound <= (f.Signed ? int.MaxValue : uint.MaxValue)) return f.Signed ? Lane.S32 : Lane.U32;
        return f.Signed ? Lane.S64 : Lane.U64;
    }

    // A whole reduction's chunking: at most 1024 workgroups of at least 4096 values.
    private static int WholeBlocks(int n) => (int)Math.Min(1024, Math.Max(1, (n + 4095) / 4096));
    private static int WholeChunk(int n) => (n + WholeBlocks(n) - 1) / WholeBlocks(n);

    // Row splits for a per-column pass: about 128k column·split invocations, at most 64 splits of at least 64 rows.
    private static int Splits(int rows, int cols)
    {
        int splits = (int)Math.Min(64, Math.Max(1, (128 * 1024 + cols - 1) / cols));
        return Math.Min(splits, Math.Max(1, rows / 64));
    }

    public string Reduce<TN>(string op, Axis axis, ulong x, int rows, int cols, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int total = rows * cols;
        if (op is "max" or "min" or "argmax")
        {
            Reductions.ValidateExtreme(op, axis, rows, cols);
            var own = f.Natural;
            if (op == "argmax")
            {
                if (axis == Axis.Cols)
                {
                    // A thread per column over a split of the rows (as the column sums), the splits joined in order.
                    bool byWords = VkKernels.ColsByWords(f, cols);
                    var kc = vk.Cached($"argmax_cols:{f.Name}:{byWords}", () => VkKernels.ArgmaxCols(vk, f, own, byWords));
                    uint across = Groups(byWords ? cols / f.Per : cols, VkKernels.ReduceThreads);
                    int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
                    if (splits == 1)
                    {
                        vk.Launch(kc, across, 1, x, output, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per), 0);
                        return "s32 index";
                    }

                    ulong at = vk.Rent((long)splits * cols * 4), vals = vk.Rent((long)splits * cols * own.Size);
                    try
                    {
                        vk.Launch(kc, across, (uint)splits, x, at, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per), vals);
                        var join = vk.Cached($"argmax_cols_join:{own.Name}", () => VkKernels.ArgmaxColsJoin(vk, own));
                        vk.Launch(join, Groups(cols, VkKernels.ReduceThreads), 1, at, vals, output, Vulkan.U(splits), Vulkan.U(cols));
                    }
                    finally
                    {
                        vk.Return(at);
                        vk.Return(vals);
                    }

                    return "s32 index";
                }

                int r = axis == Axis.Whole ? 1 : rows, c = axis == Axis.Whole ? total : cols;
                bool byGroups = VkKernels.ByGroups(f, c, x);
                var kr = vk.Cached($"argmax:{f.Name}:{byGroups}", () => VkKernels.ArgmaxRows(vk, f, own, byGroups));
                vk.Launch(kr, (uint)r, 1, x, output, Vulkan.U(c));
                return "s32 index";
            }

            Launch(f, own, op, false, axis, x, rows, cols, output);
            return own.Name;
        }

        var (lo, hi) = !f.IsFloat && f.Bits is 16 or 32 ? (TN.Min, TN.Max) : Range<TN>(x, total);
        long terms = axis switch { Axis.Whole => total, Axis.Rows => cols, _ => rows };
        var bounds = op == "sumsq" ? new Bounds(lo, hi, lo, hi, terms) : new Bounds(lo, hi, 1, 1, terms);
        var lane = SumLane(f, in bounds);
        if (axis == Axis.Whole && f.IsFloat && !lane.Wide)
        {
            // A narrow float type's whole sum: each invocation's running sum in f32 compensated, the invocations' and the
            // chunks' sums joined in float64, the result rounded once to the f32 lane.
            int n = total, blocks = WholeBlocks(n), chunk = WholeChunk(n);
            bool wholeGroups = VkKernels.ByGroups(f, chunk, x) && n % VkKernels.GroupValues(f) == 0;
            var kahan = vk.Cached($"reduce_rows:sum:{op == "sumsq"}:{f.Name}:f32:kahan:{wholeGroups}", () => VkKernels.ReduceRows(vk, f, Lane.F32, "sum", op == "sumsq", compensated: true, groups: wholeGroups));
            var lf = VkKernels.LaneFormat(Lane.F64);
            var join = vk.Cached($"reduce_rows:sum:False:{lf.Name}:f64", () => VkKernels.ReduceRows(vk, lf, Lane.F64, "sum", false));
            ulong partials = vk.Rent((long)blocks * 8), wide = vk.Rent(8);
            try
            {
                vk.Launch(kahan, (uint)blocks, 1, x, partials, Vulkan.U(n), Vulkan.U(chunk));
                vk.Launch(join, 1, 1, partials, wide, Vulkan.U(blocks), Vulkan.U(0));
                Back(VkKernels.LaneFormat(Lane.F32), Lane.F64, wide, output, 1);
            }
            finally
            {
                vk.Return(partials);
                vk.Return(wide);
            }

            return lane.Name;
        }

        if (axis == Axis.Whole)
        {
            var chunkBounds = op == "sumsq" ? new Bounds(lo, hi, lo, hi, WholeChunk(total)) : new Bounds(lo, hi, 1, 1, WholeChunk(total));
            var chunkLane = SumLane(f, in chunkBounds);
            if (!lane.IsFloat && chunkLane.IsFloat) chunkLane = f.Signed ? Lane.S32 : Lane.U32;
            Launch(f, lane, "sum", op == "sumsq", axis, x, rows, cols, output, chunkLane);
            return lane.Name;
        }

        if (axis == Axis.Rows && f.IsFloat && !lane.Wide)
        {
            // A narrow float type's row sums: the compensated pass, a workgroup a row, joined as exact f32 pairs, rounded once.
            bool rowGroups = VkKernels.ByGroups(f, cols, x);
            var kr = vk.Cached($"reduce_rows:sum:{op == "sumsq"}:{f.Name}:f32:kahan:f32:{rowGroups}", () => VkKernels.ReduceRows(vk, f, Lane.F32, "sum", op == "sumsq", compensated: true, storeF32: true, groups: rowGroups));
            vk.Launch(kr, (uint)rows, 1, x, output, Vulkan.U(cols), Vulkan.U(0));
            return lane.Name;
        }

        Launch(f, lane, "sum", op == "sumsq", axis, x, rows, cols, output);
        return lane.Name;
    }

    private void Launch(Fmt f, Lane lane, string op, bool squares, Axis axis, ulong x, int rows, int cols, ulong output, Lane? chunkLane = null)
    {
        if (axis == Axis.Cols)
        {
            // A thread a stored word of the columns where a word holds several and the rows are whole words.
            bool byWords = VkKernels.ColsByWords(f, cols);
            var kc = vk.Cached($"reduce_cols:{op}:{squares}:{f.Name}:{lane.Name}:{byWords}", () => VkKernels.ReduceCols(vk, f, lane, op, squares, byWords));
            uint across = Groups(byWords ? cols / f.Per : cols, VkKernels.ReduceThreads);
            int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
            if (splits == 1)
            {
                vk.Launch(kc, across, 1, x, output, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per));
                return;
            }

            ulong parts = vk.Rent((long)splits * cols * lane.Size);
            try
            {
                vk.Launch(kc, across, (uint)splits, x, parts, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per));
                var lf = VkKernels.LaneFormat(lane);
                var k2 = vk.Cached($"reduce_cols:{op}:False:{lf.Name}:{lane.Name}", () => VkKernels.ReduceCols(vk, lf, lane, op, false));
                vk.Launch(k2, Groups(cols, VkKernels.ReduceThreads), 1, parts, output, Vulkan.U(splits), Vulkan.U(cols), Vulkan.U(splits));
            }
            finally
            {
                vk.Return(parts);
            }

            return;
        }

        if (axis == Axis.Rows)
        {
            // A 16-byte group of words a step where a word holds several values and the rows are whole groups.
            bool rowGroups = VkKernels.ByGroups(f, cols, x);
            var kr = vk.Cached($"reduce_rows:{op}:{squares}:{f.Name}:{lane.Name}:{rowGroups}", () => VkKernels.ReduceRows(vk, f, lane, op, squares, groups: rowGroups));
            vk.Launch(kr, (uint)rows, 1, x, output, Vulkan.U(cols), Vulkan.U(0));
            return;
        }

        int n = rows * cols, blocks = WholeBlocks(n), chunk = WholeChunk(n);
        var first = chunkLane ?? lane;
        bool chunkGroups = VkKernels.ByGroups(f, chunk, x) && n % VkKernels.GroupValues(f) == 0;
        var kc2 = vk.Cached($"reduce_rows:{op}:{squares}:{f.Name}:{first.Name}:{chunkGroups}", () => VkKernels.ReduceRows(vk, f, first, op, squares, groups: chunkGroups));
        ulong partials = vk.Rent((long)blocks * first.Size);
        try
        {
            vk.Launch(kc2, (uint)blocks, 1, x, partials, Vulkan.U(n), Vulkan.U(chunk));
            var lf = VkKernels.LaneFormat(first);
            var k2 = vk.Cached($"reduce_rows:{op}:False:{lf.Name}:{lane.Name}", () => VkKernels.ReduceRows(vk, lf, lane, op, false));
            vk.Launch(k2, 1, 1, partials, output, Vulkan.U(blocks), Vulkan.U(0));
        }
        finally
        {
            vk.Return(partials);
        }
    }

    /// <summary>The smallest and largest stored value (the type's range for floats and 8-bit values: it decides the lane already).</summary>
    public (double Min, double Max) Range<TN>(ulong x, int n) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsFloat || f.Bits <= 8) return (TN.Min, TN.Max);
        var k = vk.Cached($"range:{f.Name}", () => VkKernels.Range(vk, f));
        int blocks = (int)Math.Min(512, Math.Max(1, (f.Words(n) + VkKernels.RangeThreads - 1) / VkKernels.RangeThreads));
        ulong outBuffer = vk.Rent(blocks * 16L);
        try
        {
            vk.Launch(k, (uint)blocks, 1, x, Vulkan.U(n), outBuffer);
            var pairs = new long[blocks * 2];
            vk.Download<long>(outBuffer, pairs);
            long lo = long.MaxValue, hi = long.MinValue;
            for (int i = 0; i < blocks; i++) { lo = Math.Min(lo, pairs[2 * i]); hi = Math.Max(hi, pairs[2 * i + 1]); }
            return (lo, hi);
        }
        finally
        {
            vk.Return(outBuffer);
        }
    }

    // A lane array into stored values (a block format's by its encoder).
    private void Back(Fmt f, Lane lane, ulong lanes, ulong stored, long count)
    {
        if (f.IsBlock)
        {
            Encode(f, count, vk.Cached($"block_lanes:{lane.Name}:{f.Name}", () => VkKernels.BlockFromLanes(vk, lane, f)), stored, lanes);
            return;
        }

        var k = vk.Cached($"back:{lane.Name}:{f.Name}", () => VkKernels.FromLanes(vk, lane, f));
        vk.Launch(k, Groups(f.Words(count), VkKernels.Threads), 1, lanes, stored, Vulkan.U((int)count));
    }

    /// <summary>A stored array into a lane array (for checks).</summary>
    public void ToLanes<TN>(Lane lane, ulong stored, ulong lanes, int count) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var k = vk.Cached($"to:{f.Name}:{lane.Name}", () => VkKernels.ToLanes(vk, f, lane));
        vk.Launch(k, Groups(count, VkKernels.Threads), 1, stored, lanes, Vulkan.U(count));
    }

    /// <summary>A lane array into a stored array (for checks; a block format's by its encoder).</summary>
    public void FromLanes<TN>(Lane lane, ulong lanes, ulong stored, int count) where TN : struct, IAdad<TN> =>
        Back(Fmt.Of<TN>(), lane, lanes, stored, count);

    public string Transpose<TN>(ulong x, ulong output, int rows, int cols) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = f.Natural;
        if (f.IsBlock)
        {
            Encode(f, (long)rows * cols, vk.Cached($"block_transpose:{f.Name}", () => VkKernels.BlockTranspose(vk, f)), output, x, Vulkan.U(rows), Vulkan.U(cols));
            return lane.Name;
        }

        // By tiles through workgroup memory where a tile's output rows are whole words (VkKernels.TransposeTiles); else a
        // thread an output word, its values gathered where they are.
        uint tilesX = Groups(cols, VkKernels.TransposeTile), tilesY = Groups(rows, VkKernels.TransposeTile);
        if (VkKernels.TransposeTiles(f, rows) && tilesX <= vk.MaxGroupsX && tilesY <= vk.MaxGroupsY)
        {
            var tiled = vk.Cached($"transpose_tiled:{f.Name}", () => VkKernels.TransposeTiled(vk, f, lane));
            vk.Launch(tiled, tilesX, tilesY, x, output, Vulkan.U(rows), Vulkan.U(cols));
            return lane.Name;
        }

        var k = vk.Cached($"transpose:{f.Name}", () => VkKernels.Transpose(vk, f, lane));
        vk.Launch(k, Groups(f.Words((long)rows * cols), VkKernels.Threads), 1, x, output, Vulkan.U(rows), Vulkan.U(cols));
        return lane.Name;
    }

    /// <summary>y = A·x (or Aᵀ·x) in the sum lane for the measured bounds, y stored as the type.</summary>
    public string Gemv<TN>(ulong a, ulong x, ulong y, int rows, int cols, bool transposed) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int xn = transposed ? rows : cols, yn = transposed ? cols : rows;
        var (aLo, aHi) = Range<TN>(a, rows * cols);
        var (xLo, xHi) = Range<TN>(x, xn);
        var bounds = new Bounds(aLo, aHi, xLo, xHi, xn);
        var lane = SumLane(f, in bounds);
        ulong yl = vk.Rent((long)yn * lane.Size);
        try
        {
            if (transposed)
            {
                bool byWords = VkKernels.ColsByWords(f, cols);
                var kc = vk.Cached($"gemv_cols:{f.Name}:{lane.Name}:{byWords}", () => VkKernels.GemvCols(vk, f, lane, byWords));
                uint across = Groups(byWords ? cols / f.Per : cols, VkKernels.Threads);
                int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
                if (splits == 1) vk.Launch(kc, across, 1, a, x, yl, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per));
                else
                {
                    ulong partials = vk.Rent((long)splits * cols * lane.Size);
                    try
                    {
                        vk.Launch(kc, across, (uint)splits, a, x, partials, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per));
                        var lf = VkKernels.LaneFormat(lane);
                        var k2 = vk.Cached($"reduce_cols:sum:False:{lf.Name}:{lane.Name}", () => VkKernels.ReduceCols(vk, lf, lane, "sum", false));
                        vk.Launch(k2, Groups(cols, VkKernels.ReduceThreads), 1, partials, yl, Vulkan.U(splits), Vulkan.U(cols), Vulkan.U(splits));
                    }
                    finally
                    {
                        vk.Return(partials);
                    }
                }
            }
            else
            {
                bool byGroups = VkKernels.ByGroups(f, cols, a) && x % 16 == 0;
                var kr = vk.Cached($"gemv_rows:{f.Name}:{lane.Name}:{byGroups}", () => VkKernels.GemvRows(vk, f, lane, byGroups));
                vk.Launch(kr, (uint)rows, 1, a, x, yl, Vulkan.U(rows), Vulkan.U(cols));
            }

            Back(f, lane, yl, y, yn);
        }
        finally
        {
            vk.Return(yl);
        }

        return lane.Name;
    }
}
