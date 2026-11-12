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
    public static SpvKernel ReduceRows(Vulkan vk, Fmt f, Lane lane, string op, bool squares, bool compensated = false, bool storeF32 = false, bool table = false)
    {
        if (storeF32 && !compensated) throw new ArgumentException("the f32 store is the compensated pass's");
        if (compensated && (lane.Name != "f32" || op != "sum")) throw new ArgumentException("the compensated pass is a float sum in the f32 lane");
        if (table && storeF32) throw new ArgumentException("the table is the chunk pass's");
        var k = Kinds.Of(lane);
        var b = new Spv($"reduce_rows_{op}{(squares ? "sq" : "")}_{f.Name}_{lane.Name}{(compensated ? "_kahan" : "")}{(storeF32 ? "_f32" : "")}{(table ? "_tab" : "")}", ReduceThreads, vk.Controls);
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
        b.For(tid, count, b.U(ReduceThreads), i =>
        {
            var v = Codec.UnpackAt(b, f, src, start + i, lane);
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
        });

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

    /// <summary>out[split][col] = op over the column's rows in split (the workgroup's y): rows [split·per, min(rows, (split+1)·per)); a thread per column.</summary>
    public static SpvKernel ReduceCols(Vulkan vk, Fmt f, Lane lane, string op, bool squares)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"reduce_cols_{op}{(squares ? "sq" : "")}_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        var per = b.Arg(4, K.U32);
        var col = b.GlobalX;
        var split = b.GroupY;
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
    public static SpvKernel ArgmaxRows(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"argmax_rows_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var cols = b.Arg(2, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var none = b.U(0xFFFFFFFF);
        var best = b.Local(Identity(b, k, "max"));
        var bestAt = b.Local(none);
        b.For(tid, cols, b.U(ReduceThreads), i =>
        {
            var v = Codec.UnpackAt(b, f, src, row * cols + i, lane);
            var take = Rank(b, lane, v, best.V).Above | (bestAt.V == none);
            best.V = b.Select(take, v, best.V);
            bestAt.V = b.Select(take, i, bestAt.V);
        });

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
    /// rank the same), s32, a thread per column.</summary>
    public static SpvKernel ArgmaxCols(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"argmax_cols_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var cols = b.Arg(3, K.U32);
        var col = b.GlobalX;
        var none = b.U(0xFFFFFFFF);
        b.If(col < cols, () =>
        {
            var best = b.Local(Identity(b, k, "max"));
            var bestAt = b.Local(none);
            b.For(b.U(0), rows, r =>
            {
                var v = Codec.UnpackAt(b, f, src, r * cols + col, lane);
                var take = Rank(b, lane, v, best.V).Above | (bestAt.V == none);
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

    public const int GemvThreads = 64;

    /// <summary>y[row] = Σ_c A[row][c]·x[c] in the lane (a workgroup per row, its partial sums joined by the tree), y as lane values.</summary>
    public static SpvKernel GemvRows(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"gemv_rows_{f.Name}_{lane.Name}", GemvThreads, vk.Controls);
        var pa = b.Arg(0);
        var px = b.Arg(1);
        var py = b.Arg(2);
        var cols = b.Arg(4, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var acc = b.Local(b.C(k, 0));
        b.For(tid, cols, b.U(GemvThreads), i =>
        {
            var av = Codec.UnpackAt(b, f, pa, row * cols + i, lane);
            var xv = Codec.UnpackAt(b, f, px, i, lane);
            acc.V = b.Fma(av, xv, acc.V);
        });
        var sum = GroupReduce(b, b.Shared(k, GemvThreads), tid, acc.V, (x, y) => x + y, GemvThreads);
        b.If(tid == 0, () => b.Store(b.At(py, row, lane.Size), sum));
        return b.Build();
    }

    /// <summary>y[split][col] = Σ over the rows of split (the workgroup's y) of A[r][col]·x[r]; a thread per column.</summary>
    public static SpvKernel GemvCols(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"gemv_cols_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var pa = b.Arg(0);
        var px = b.Arg(1);
        var py = b.Arg(2);
        var rows = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var per = b.Arg(5, K.U32);
        var col = b.GlobalX;
        var split = b.GroupY;
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
