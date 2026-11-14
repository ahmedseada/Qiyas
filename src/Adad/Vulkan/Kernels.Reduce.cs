// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Reductions, transposes and matrix·vector products on Vulkan: CUDA's kernels (Cuda/Ptx.Reduce.cs) over the same lanes,
// a workgroup's partial results joined by a tree in workgroup memory (a fixed order at any subgroup size, where CUDA's
// joins by warp shuffles). Values are addressed by index (Codec.UnpackAt), so every shape works for every format,
// packed ones included; rows need not end on a word.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int ReduceThreads = 256;

    /// <summary>A lane's values as a stored format (partial results live in lane arrays).</summary>
    public static Fmt LaneFormat(Lane lane) => lane.Name switch
    {
        "f32" => new Fmt("float32", 32, true, true),
        "f64" => new Fmt("float64", 64, true, true),
        "s32" => new Fmt("int32", 32, false, true),
        "u32" => new Fmt("uint32", 32, false, false),
        "s64" => new Fmt("int64", 64, false, true),
        _ => new Fmt("uint64", 64, false, false),
    };

    /// <summary>The value a reduction starts from: −∞ / the smallest value for max; for min a NaN (the top of Adad's order:
    /// NMin of it and x is x, of NaNs alone a NaN) / the largest value; zero for sums.</summary>
    private static V Identity(Spv b, K k, string op) => op switch
    {
        "max" => k.IsFloat() ? b.C(k, double.NegativeInfinity) : k is K.U32 or K.U64 ? b.CBits(k, 0) : b.CBits(k, k.Wide() ? 0x8000000000000000UL : 0x80000000UL),
        "min" => k.IsFloat() ? b.C(k, double.NaN) : k is K.U32 or K.U64 ? b.CBits(k, ulong.MaxValue) : b.CBits(k, k.Wide() ? 0x7FFFFFFFFFFFFFFFUL : 0x7FFFFFFFUL),
        _ => b.C(k, 0),
    };

    // a op v in Adad's order (Routing.Order): max the first-ranked (First: a NaN when either is one), min the last-ranked
    // (NMin: the number when one is a NaN, a NaN only when both are), else the sum.
    private static V Combine(Spv b, string op, V a, V v) => op switch
    {
        "max" => First(b, a, v),
        "min" => b.Min(a, v),
        _ => a + v,
    };

    /// <summary>The workgroup's value of `combine` over every invocation's <paramref name="v"/> (a power-of-two width; a
    /// tree in workgroup memory, the same order at any subgroup size), returned to all of them. Uniform control flow only.</summary>
    internal static V GroupReduce(Spv b, Arr sh, V tid, V v, Func<V, V, V> combine, int threads)
    {
        sh[tid] = v;
        b.Barrier();
        for (int s = threads / 2; s >= 1; s /= 2)
        {
            int step = s;
            b.If(tid < step, () => sh[tid] = combine(sh[tid], sh[tid + step]));
            b.Barrier();
        }

        var result = sh[0];
        b.Barrier();
        return result;
    }

    // The sum of f32 pairs hi + lo over the workgroup (CUDA's BlockReducePair: each step's sum split exactly, its error
    // added to the low parts, the pair renormalized), returned to all.
    private static (V Hi, V Lo) GroupReducePair(Spv b, Arr shHi, Arr shLo, V tid, V hi, V lo, int threads)
    {
        shHi[tid] = hi;
        shLo[tid] = lo;
        b.Barrier();
        for (int s = threads / 2; s >= 1; s /= 2)
        {
            int step = s;
            b.If(tid < step, () =>
            {
                var h = shHi[tid];
                var l = shLo[tid];
                var oh = shHi[tid + step];
                var ol = shLo[tid + step];
                var sum = h + oh;
                var bv = sum - h;
                var t = h - (sum - bv);
                var e = (oh - bv) + t;
                l = (l + ol) + e;
                var nh = sum + l;
                l = l - (nh - sum);
                shHi[tid] = nh;
                shLo[tid] = l;
            });
            b.Barrier();
        }

        var rh = shHi[0];
        var rl = shLo[0];
        b.Barrier();
        return (rh, rl);
    }

    // ---------------------------------------------------------------- per row (a workgroup a row) and whole (workgroups over chunks)

    /// <summary>out[row] = op over x[row·cols ..] for cols values (a workgroup per row; rows = the workgroup's x), or over
    /// one long row's chunk when p_chunk > 0: workgroup r covers values [r·chunk, min(n, (r+1)·chunk)).</summary>
    /// <param name="compensated">A float sum's chunk pass in f32: each invocation's running sum compensated (Kahan), the
    /// invocations' sums joined in float64 and stored as f64 partials (or, <paramref name="storeF32"/>, joined as exact f32
    /// pairs and rounded once to f32: a row's sum).</param>
    /// <param name="table">The chunk pass of several arrays in one dispatch: slot 0 is a table on the device (TableRow) of
    /// `cols` rows — an array's first block, its values, its chunk, where it is — and the workgroup is the array's its
    /// index falls in; the partials at dst in block order.</param>
    /// <summary>The values of f a 16-byte group of words holds (ReduceRows's and ArgmaxRows's group pass).</summary>
    public static int GroupValues(Fmt f) => 4 * f.Per;

    /// <summary>Whether a pass over runs of `values` values of f starting at multiples of them from `address` may read them
    /// a 16-byte group at a time: a type of several values a word, the runs whole groups, the address a group's.</summary>
    public static bool ByGroups(Fmt f, long values, ulong address) => !f.IsBlock && f.Per > 1 && values % GroupValues(f) == 0 && address % 16 == 0;

    /// <param name="groups">The values read a 16-byte group of words at a time, an invocation the groups tid, tid + 256, … of
    /// the row (or chunk), each group's values in order (ByGroups: the row's or chunk's values whole groups): a load a group
    /// in place of a load a value. A sum's order of terms is the groups' (a float sum's rounding may differ in its last
    /// bits); the extremes are the same.</param>
    public static SpvKernel ReduceRows(Vulkan vk, Fmt f, Lane lane, string op, bool squares, bool compensated = false, bool storeF32 = false, bool table = false, bool groups = false)
    {
        if (storeF32 && !compensated) throw new ArgumentException("the f32 store is the compensated pass's");
        if (compensated && (lane.Name != "f32" || op != "sum")) throw new ArgumentException("the compensated pass is a float sum in the f32 lane");
        if (table && storeF32) throw new ArgumentException("the table is the chunk pass's");
        if (groups && table) throw new ArgumentException("the table's pass reads a value at a time");
        var k = Kinds.Of(lane);
        var b = new Spv($"reduce_rows_{op}{(squares ? "sq" : "")}_{f.Name}_{lane.Name}{(compensated ? "_kahan" : "")}{(storeF32 ? "_f32" : "")}{(table ? "_tab" : "")}{(groups ? "_g" : "")}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var cols = b.Arg(2, K.U32);
        var chunk = b.Arg(3, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        if (table)
        {
            var (tr, local) = TableRow(b, src, cols, row, SumRow);
            dst = dst + b.Load(K.U64, tr) * (double)(compensated ? 8 : lane.Size);
            cols = b.Convert(b.Load(K.U64, b.At(tr, b.U(0), 8, 8)), K.U32);
            chunk = b.Convert(b.Load(K.U64, b.At(tr, b.U(0), 8, 16)), K.U32);
            src = b.Load(K.U64, b.At(tr, b.U(0), 8, 24));
            row = local;
        }

        var byChunk = chunk != 0;
        var cs = row * chunk;
        var start = b.Select(byChunk, cs, row * cols);
        var count = b.Select(byChunk, b.Min(cs + chunk, cols) - cs, cols);
        var acc = b.Local(Identity(b, k, op));
        var comp = b.Local(b.F(0));
        void Take(V v)
        {
            if (compensated)
            {
                // y = term − c; t = acc + y; c = (t − acc) − y; acc = t (each step rounded as written).
                var y = (squares ? v * v : v) - comp.V;
                var t = acc.V + y;
                comp.V = (t - acc.V) - y;
                acc.V = t;
            }
            else if (squares) acc.V = b.Fma(v, v, acc.V);
            else acc.V = Combine(b, op, acc.V, v);
        }

        if (groups)
        {
            // A group of four words (GroupValues values) a step: the row's or chunk's first value a group's.
            int per = GroupValues(f);
            var first = b.At(src, start / (double)per, 16);
            b.For(tid, count / (double)per, b.U(ReduceThreads), g =>
            {
                var w = b.LoadN(Codec.WordKind(f), 4, b.At(first, g, 16));
                for (int q = 0; q < 4; q++)
                    for (int i = 0; i < f.Per; i++) Take(Codec.Unpack(b, f, w[q], i, lane));
            });
        }
        else b.For(tid, count, b.U(ReduceThreads), i => Take(Codec.UnpackAt(b, f, src, start + i, lane)));

        if (storeF32)
        {
            var (ph, pl) = GroupReducePair(b, b.Shared(K.F32, ReduceThreads), b.Shared(K.F32, ReduceThreads), tid, acc.V, -comp.V, ReduceThreads);
            var rowSum = ph + pl;
            b.If(tid == 0, () => b.Store(b.At(dst, row, 4), rowSum));
            return b.Build();
        }

        var total = acc.V;
        var kk = k;
        if (compensated)
        {
            // The invocation's sum in float64: acc − c, then the workgroup's sum in float64, stored as an f64 partial.
            total = b.Convert(acc.V, K.F64) - b.Convert(comp.V, K.F64);
            kk = K.F64;
        }

        string join = op == "max" ? "max" : op == "min" ? "min" : "sum";
        var sum = GroupReduce(b, b.Shared(kk, ReduceThreads), tid, total, (x, y) => Combine(b, join, x, y), ReduceThreads);
        b.If(tid == 0, () => b.Store(b.At(dst, row, kk.Bytes()), sum));
        return b.Build();
    }

    /// <summary>Whether ReduceCols takes the columns a stored word at a time (ReduceCols's words): a type of two to eight
    /// values a word whose rows are whole words. Not 16 or 32 a word: an invocation's accumulators are its word's values,
    /// and the invocations across a row as many times fewer (bool sum:cols 0.084 ms against 0.061 a column an invocation,
    /// gemv_t 0.227 against 0.145).</summary>
    public static bool ColsByWords(Fmt f, int cols) => !f.IsBlock && f.Per is > 1 and <= 8 && cols % f.Per == 0;

    /// <summary>out[split][col] = op over the column's rows in split (the workgroup's y): rows [split·per, min(rows, (split+1)·per)); a thread per column.
    /// With <paramref name="words"/> (ColsByWords), a thread per word of a row's columns: each row's word read once and
    /// its values joined into their columns' sums, each column's rows in the same order: the same results.</summary>
    public static SpvKernel ReduceCols(Vulkan vk, Fmt f, Lane lane, string op, bool squares, bool words = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"reduce_cols_{op}{(squares ? "sq" : "")}_{f.Name}_{lane.Name}{(words ? "_w" : "")}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        var per = b.Arg(4, K.U32);
        var col = b.GlobalX;
        var split = b.GroupY;
        if (words)
        {
            var wk = Codec.WordKind(f);
            var rowWords = cols / (double)f.Per;
            b.If(col < rowWords, () =>
            {
                var acc = new Var[f.Per];
                for (int i = 0; i < f.Per; i++) acc[i] = b.Local(Identity(b, k, op));
                var r0 = split * per;
                b.For(r0, b.Min(r0 + per, rows), r =>
                {
                    var word = b.Load(wk, b.At(src, r * rowWords + col, f.WordBits / 8));
                    for (int i = 0; i < f.Per; i++)
                    {
                        var v = Codec.Unpack(b, f, word, i, lane);
                        acc[i].V = squares ? b.Fma(v, v, acc[i].V) : Combine(b, op, acc[i].V, v);
                    }
                });
                for (int i = 0; i < f.Per; i++) b.Store(b.At(dst, split * cols + col * (double)f.Per + i, k.Bytes()), acc[i].V);
            });
            return b.Build();
        }

        b.If(col < cols, () =>
        {
            var acc = b.Local(Identity(b, k, op));
            var r0 = split * per;
            b.For(r0, b.Min(r0 + per, rows), r =>
            {
                var v = Codec.UnpackAt(b, f, src, r * cols + col, lane);
                acc.V = squares ? b.Fma(v, v, acc.V) : Combine(b, op, acc.V, v);
            });
            b.Store(b.At(dst, split * cols + col, k.Bytes()), acc.V);
        });
        return b.Build();
    }

    /// <summary>out[row] = the column of the row's first-ranked value in Adad's order (Rank; the lowest column among those
    /// that rank the same), s32, a workgroup per row.</summary>
    public static SpvKernel ArgmaxRows(Vulkan vk, Fmt f, Lane lane, bool groups = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"argmax_rows_{f.Name}_{lane.Name}{(groups ? "_g" : "")}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var cols = b.Arg(2, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var none = b.U(0xFFFFFFFF);
        var best = b.Local(Identity(b, k, "max"));
        var bestAt = b.Local(none);
        // An invocation's values in ascending columns (a later one taken only when it ranks strictly above): a value a
        // step, or (ReduceRows's groups) a 16-byte group of words a step.
        void Take(V v, V i)
        {
            var take = Rank(b, lane, v, best.V).Above | (bestAt.V == none);
            best.V = b.Select(take, v, best.V);
            bestAt.V = b.Select(take, i, bestAt.V);
        }

        if (groups)
        {
            int per = GroupValues(f);
            var first = b.At(src, row * cols / (double)per, 16);
            b.For(tid, cols / (double)per, b.U(ReduceThreads), g =>
            {
                var w = b.LoadN(Codec.WordKind(f), 4, b.At(first, g, 16));
                for (int q = 0; q < 4; q++)
                    for (int i = 0; i < f.Per; i++) Take(Codec.Unpack(b, f, w[q], i, lane), g * (double)per + (q * f.Per + i));
            });
        }
        else b.For(tid, cols, b.U(ReduceThreads), i => Take(Codec.UnpackAt(b, f, src, row * cols + i, lane), i));

        // The workgroup's first-ranked: a tree over (value, index) pairs in Adad's order; ties keep the lower index.
        var sv = b.Shared(k, ReduceThreads);
        var si = b.Shared(K.U32, ReduceThreads);
        sv[tid] = best.V;
        si[tid] = bestAt.V;
        b.Barrier();
        for (int s = ReduceThreads / 2; s >= 1; s /= 2)
        {
            int step = s;
            b.If(tid < step, () =>
            {
                var mv = sv[tid];
                var mi = si[tid];
                var ov = sv[tid + step];
                var oi = si[tid + step];
                var (above, same) = Rank(b, lane, ov, mv);
                var take = (above | (same & (oi < mi)) | (mi == none)) & (oi != none);
                sv[tid] = b.Select(take, ov, mv);
                si[tid] = b.Select(take, oi, mi);
            });
            b.Barrier();
        }

        b.If(tid == 0, () => b.Store(b.At(dst, row, 4), si[0]));
        return b.Build();
    }

    /// <summary>out[col] = the row of the column's first-ranked value in Adad's order (Rank; the first row among those that
    /// rank the same), s32, a thread per column over rows [split·per, min(rows, (split + 1)·per)) (split the workgroup's
    /// y). With one split the row goes to out; with several each split's row (u32, absolute) and its value (lane storage)
    /// go to out and vals at split·cols + col, joined by ArgmaxColsJoin. Slots: x, out, rows, cols, per, vals.</summary>
    public static SpvKernel ArgmaxCols(Vulkan vk, Fmt f, Lane lane, bool words = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"argmax_cols_{f.Name}_{lane.Name}{(words ? "_w" : "")}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        var per = b.Arg(4, K.U32);
        var vals = b.Arg(5);
        var col = b.GlobalX;
        var split = b.GroupY;
        var none = b.U(0xFFFFFFFF);
        if (words)
        {
            // A thread per word of a row's columns (ColsByWords): each column's rows in the same order, the same picks.
            var wk = Codec.WordKind(f);
            var rowWords = cols / (double)f.Per;
            b.If(col < rowWords, () =>
            {
                var best = new Var[f.Per];
                var bestAt = new Var[f.Per];
                for (int i = 0; i < f.Per; i++)
                {
                    best[i] = b.Local(Identity(b, k, "max"));
                    bestAt[i] = b.Local(none);
                }

                var r0 = split * per;
                b.For(r0, b.Min(r0 + per, rows), r =>
                {
                    var word = b.Load(wk, b.At(src, r * rowWords + col, f.WordBits / 8));
                    for (int i = 0; i < f.Per; i++)
                    {
                        var v = Codec.Unpack(b, f, word, i, lane);
                        var take = Rank(b, lane, v, best[i].V).Above | (bestAt[i].V == none);
                        best[i].V = b.Select(take, v, best[i].V);
                        bestAt[i].V = b.Select(take, r, bestAt[i].V);
                    }
                });
                for (int i = 0; i < f.Per; i++)
                {
                    var at = split * cols + col * (double)f.Per + i;
                    var value = best[i].V;
                    b.Store(b.At(dst, at, 4), bestAt[i].V);
                    b.If(vals != 0UL, () => b.Store(b.At(vals, at, lane.Size), value));
                }
            });
            return b.Build();
        }

        b.If(col < cols, () =>
        {
            var best = b.Local(Identity(b, k, "max"));
            var bestAt = b.Local(none);
            var r0 = split * per;
            b.For(r0, b.Min(r0 + per, rows), r =>
            {
                var v = Codec.UnpackAt(b, f, src, r * cols + col, lane);
                var take = Rank(b, lane, v, best.V).Above | (bestAt.V == none);
                best.V = b.Select(take, v, best.V);
                bestAt.V = b.Select(take, r, bestAt.V);
            });
            var at = split * cols + col;
            b.Store(b.At(dst, at, 4), bestAt.V);
            b.If(vals != 0UL, () => b.Store(b.At(vals, at, lane.Size), best.V));
        });
        return b.Build();
    }

    /// <summary>The splits' first-ranked rows of each column (ArgmaxCols's) joined in split order: a later split's taken only
    /// when its value ranks strictly above (a split with no row never), so the first row among those that rank the same, as
    /// one scan of the column. Slots: rows (u32), vals (lane storage), out (s32), splits, cols.</summary>
    public static SpvKernel ArgmaxColsJoin(Vulkan vk, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"argmax_cols_join_{lane.Name}", ReduceThreads, vk.Controls);
        var pr = b.Arg(0);
        var pv = b.Arg(1);
        var dst = b.Arg(2);
        var splits = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var col = b.GlobalX;
        var none = b.U(0xFFFFFFFF);
        b.If(col < cols, () =>
        {
            var best = b.Local(Identity(b, k, "max"));
            var bestAt = b.Local(none);
            b.For(b.U(0), splits, s =>
            {
                var at = s * cols + col;
                var r = b.Load(K.U32, b.At(pr, at, 4));
                var v = b.Load(k, b.At(pv, at, lane.Size));
                var take = (Rank(b, lane, v, best.V).Above | (bestAt.V == none)) & (r != none);
                best.V = b.Select(take, v, best.V);
                bestAt.V = b.Select(take, r, bestAt.V);
            });
            b.Store(b.At(dst, col, 4), bestAt.V);
        });
        return b.Build();
    }

    public const int RangeThreads = 256;

    /// <summary>The smallest and largest stored value of an integer format (as s64, u64 for uint64 — its largest held to
    /// s64's): a pair per workgroup, joined on the host.</summary>
    public static SpvKernel Range(Vulkan vk, Fmt f)
    {
        bool u64 = f.Is("uint64");
        var wideLane = u64 ? Lane.U64 : Lane.S64;
        var wk = u64 ? K.U64 : K.S64;
        var b = new Spv($"range_{f.Name}", RangeThreads, vk.Controls);
        var src = b.Arg(0);
        var n = b.Arg(1, K.U32);
        var dst = b.Arg(2);
        var tid = b.LocalIdX;
        var lo = b.Local(u64 ? b.CBits(wk, ulong.MaxValue) : b.CBits(wk, 0x7FFFFFFFFFFFFFFFUL));
        var hi = b.Local(u64 ? b.CBits(wk, 0) : b.CBits(wk, 0x8000000000000000UL));
        var words = WordCount(b, f, n);
        var wordKind = Codec.WordKind(f);
        b.For(b.GlobalX, words, b.StrideX, w =>
        {
            var word = b.Load(wordKind, b.At(src, w, f.WordBits / 8));
            for (int i = 0; i < f.Per; i++)
            {
                int at = i;
                b.If(w * (double)f.Per + at < n, () =>
                {
                    var v = Codec.Unpack(b, f, word, at, wideLane);
                    lo.V = b.Min(lo.V, v);
                    hi.V = b.Max(hi.V, v);
                });
            }
        });
        var minimum = GroupReduce(b, b.Shared(wk, RangeThreads), tid, lo.V, b.Min, RangeThreads);
        var maximum = GroupReduce(b, b.Shared(wk, RangeThreads), tid, hi.V, b.Max, RangeThreads);
        b.If(tid == 0, () =>
        {
            var top = u64 ? b.Min(maximum, b.CBits(wk, 0x7FFFFFFFFFFFFFFFUL)) : maximum;
            var at = b.At(dst, b.GroupX, 16);
            b.Store(at, b.Bits(minimum));
            b.Store(at + 8UL, b.Bits(top));
        });
        return b.Build();
    }

    /// <summary>A lane array's n values into a stored array: a thread per word (values past n in its last word zero).</summary>
    public static SpvKernel FromLanes(Vulkan vk, Lane lane, Fmt f)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"back_{lane.Name}_{f.Name}", Threads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var word = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var idx = gid * (double)f.Per + i;
                var v = b.Local(b.C(k, 0));
                b.If(idx < n, () => v.V = b.Load(k, b.At(src, idx, lane.Size)));
                word = Codec.Pack(b, f, v.V, lane, i, word);
            }

            b.Store(b.At(dst, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>A stored array's n values into a lane array: a thread per value, each read where it is stored.</summary>
    public static SpvKernel ToLanes(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"to_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var gid = b.GlobalX;
        b.If(gid < n, () => b.Store(b.At(dst, gid, lane.Size), Codec.UnpackAt(b, f, src, gid, lane)));
        return b.Build();
    }

    /// <summary>output (cols × rows) = xᵀ: a thread per output word, each of its values read where it is stored (an output
    /// row need not end on a word: a word's values are the flattened output's).</summary>
    public static SpvKernel Transpose(Vulkan vk, Fmt f, Lane lane)
    {
        var b = new Spv($"transpose_{f.Name}", Threads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        var n = rows * cols;
        b.If(gid < WordCount(b, f, n), () =>
        {
            var word = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var o = gid * (double)f.Per + i;
                var v = b.Local(b.C(Kinds.Of(lane), 0));
                b.If(o < n, () => v.V = Codec.UnpackAt(b, f, src, (o % rows) * cols + o / rows, lane));
                word = Codec.Pack(b, f, v.V, lane, i, word);
            }

            b.Store(b.At(dst, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>The side of TransposeTiled's tiles.</summary>
    public const int TransposeTile = 32;

    /// <summary>Whether TransposeTiled takes a rows × cols of f: values of whole words, or rows of the output (rows values
    /// each) that end on a word, so a tile's 32 values of an output row are whole words.</summary>
    public static bool TransposeTiles(Fmt f, int rows) => f.Per == 1 || (long)rows * f.Bits % 32 == 0;

    /// <summary>out (cols × rows) = xᵀ by tiles of 32 × 32 values through workgroup memory (in the lane): a workgroup of 32 × 8
    /// invocations reads its tile's rows of x (consecutive invocations consecutive values) and writes the tile's rows of the
    /// output the same way, a value an invocation where a value is a word, else a word an invocation packed from the
    /// tile's column. Workgroups (⌈cols / 32⌉, ⌈rows / 32⌉). Slots: x, out, rows, cols.</summary>
    public static SpvKernel TransposeTiled(Vulkan vk, Fmt f, Lane lane)
    {
        const int T = TransposeTile, Y = 8, P = T + 1;
        var k = Kinds.Of(lane);
        var b = new Spv($"transpose_tiled_{f.Name}", T, vk.Controls, Y);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        V tx = b.LocalIdX, ty = b.LocalIdY;
        var c0 = b.GroupX * (double)T;
        var r0 = b.GroupY * (double)T;
        var tile = b.Shared(k, T * P);
        for (int j = 0; j < T; j += Y)
        {
            var r = r0 + ty + j;
            var c = c0 + tx;
            var ok = (r < rows) & (c < cols);
            var at = (ty + j) * (double)P + tx;
            b.If(ok, () => tile[at] = Codec.UnpackAt(b, f, src, r * cols + c, lane));
        }

        b.Barrier();
        var wk = Codec.WordKind(f);
        int bytes = f.WordBits / 8;
        if (f.Per == 1)
        {
            for (int j = 0; j < T; j += Y)
            {
                var oc = c0 + ty + j;                                                // the output's row: x's column
                var orr = r0 + tx;
                var value = tile[tx * (double)P + ty + j];
                b.If((oc < cols) & (orr < rows), () => b.Store(b.At(dst, oc * rows + orr, bytes), Codec.Pack(b, f, value, lane, 0, b.CBits(wk, 0))));
            }

            return b.Build();
        }

        // A tile's output row is `words` whole words (rows a multiple of Per: a word wholly inside the output or past it).
        int words = T / f.Per;
        var tid = ty * (double)T + tx;
        b.For(tid, b.U(T * words), b.U(T * Y), t =>
        {
            var orow = t / (double)words;
            var w = t - orow * (double)words;
            var oc = c0 + orow;
            var first = r0 + w * (double)f.Per;
            b.If((oc < cols) & (first < rows), () =>
            {
                var word = b.CBits(wk, 0);
                for (int i = 0; i < f.Per; i++) word = Codec.Pack(b, f, tile[(w * (double)f.Per + i) * P + orow], lane, i, word);
                b.Store(b.At(dst, (oc * rows + first) / (double)f.Per, bytes), word);
            });
        });
        return b.Build();
    }

    public const int GemvThreads = 64;

    /// <summary>y[row] = Σ_c A[row][c]·x[c] in the lane (a workgroup per row, its partial sums joined by the tree), y as lane values.</summary>
    /// <param name="groups">A's row and x read a 16-byte group of words at a time (ByGroups: the row whole groups, both
    /// addresses a group's), an invocation the groups tid, tid + 64, …, each group's terms in order: a load a group of each
    /// in place of a load a value (a float sum's order of terms the groups').</param>
    public static SpvKernel GemvRows(Vulkan vk, Fmt f, Lane lane, bool groups = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"gemv_rows_{f.Name}_{lane.Name}{(groups ? "_g" : "")}", GemvThreads, vk.Controls);
        var pa = b.Arg(0);
        var px = b.Arg(1);
        var py = b.Arg(2);
        var cols = b.Arg(4, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var acc = b.Local(b.C(k, 0));
        if (groups)
        {
            int per = GroupValues(f);
            var wk = Codec.WordKind(f);
            var first = b.At(pa, row * cols / (double)per, 16);
            b.For(tid, cols / (double)per, b.U(GemvThreads), g =>
            {
                var wa = b.LoadN(wk, 4, b.At(first, g, 16));
                var wx = b.LoadN(wk, 4, b.At(px, g, 16));
                for (int q = 0; q < 4; q++)
                    for (int i = 0; i < f.Per; i++) acc.V = b.Fma(Codec.Unpack(b, f, wa[q], i, lane), Codec.Unpack(b, f, wx[q], i, lane), acc.V);
            });
        }
        else b.For(tid, cols, b.U(GemvThreads), i =>
        {
            var av = Codec.UnpackAt(b, f, pa, row * cols + i, lane);
            var xv = Codec.UnpackAt(b, f, px, i, lane);
            acc.V = b.Fma(av, xv, acc.V);
        });
        var sum = GroupReduce(b, b.Shared(k, GemvThreads), tid, acc.V, (x, y) => x + y, GemvThreads);
        b.If(tid == 0, () => b.Store(b.At(py, row, lane.Size), sum));
        return b.Build();
    }

    /// <summary>y[split][col] = Σ over the rows of split (the workgroup's y) of A[r][col]·x[r]; a thread per column. With
    /// <paramref name="words"/> (ColsByWords), a thread per word of a row's columns, x[r] read once for them: each column's
    /// rows in the same order, the same sums.</summary>
    public static SpvKernel GemvCols(Vulkan vk, Fmt f, Lane lane, bool words = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"gemv_cols_{f.Name}_{lane.Name}{(words ? "_w" : "")}", Threads, vk.Controls);
        var pa = b.Arg(0);
        var px = b.Arg(1);
        var py = b.Arg(2);
        var rows = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var per = b.Arg(5, K.U32);
        var col = b.GlobalX;
        var split = b.GroupY;
        if (words)
        {
            var wk = Codec.WordKind(f);
            var rowWords = cols / (double)f.Per;
            b.If(col < rowWords, () =>
            {
                var acc = new Var[f.Per];
                for (int i = 0; i < f.Per; i++) acc[i] = b.Local(b.C(k, 0));
                var r0 = split * per;
                b.For(r0, b.Min(r0 + per, rows), r =>
                {
                    var word = b.Load(wk, b.At(pa, r * rowWords + col, f.WordBits / 8));
                    var xv = Codec.UnpackAt(b, f, px, r, lane);
                    for (int i = 0; i < f.Per; i++) acc[i].V = b.Fma(Codec.Unpack(b, f, word, i, lane), xv, acc[i].V);
                });
                for (int i = 0; i < f.Per; i++) b.Store(b.At(py, split * cols + col * (double)f.Per + i, lane.Size), acc[i].V);
            });
            return b.Build();
        }

        b.If(col < cols, () =>
        {
            var acc = b.Local(b.C(k, 0));
            var r0 = split * per;
            b.For(r0, b.Min(r0 + per, rows), r =>
            {
                var av = Codec.UnpackAt(b, f, pa, r * cols + col, lane);
                var xv = Codec.UnpackAt(b, f, px, r, lane);
                acc.V = b.Fma(av, xv, acc.V);
            });
            b.Store(b.At(py, split * cols + col, lane.Size), acc.V);
        });
        return b.Build();
    }
}
