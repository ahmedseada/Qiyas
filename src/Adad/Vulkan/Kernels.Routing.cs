// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and the cross-entropy loss on Vulkan (Routing.cs; CUDA's Ptx.Routing.cs): a workgroup per row for
// top-k (k up to TopKLocalMax: one read of the row, each invocation keeping its own first k in order, then k rounds of
// the workgroup's first among the invocations' heads; a larger k: k passes of the workgroup's first largest over the
// columns not yet taken) and for sampling (each invocation's
// chunk summed, the chunks' prefix by the first, the first crossing found per chunk and the least taken) and for the
// loss (a running max and sum of exponentials per invocation, joined by the tree). Values are stored a thread per word
// by gathers (top-k's values, pick) and the losses kept as lane values and stored once, so no two invocations store into
// one word.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int MaxTopK = 256;

    /// <summary>The largest k the one-read top-k takes (each invocation's first k held in registers, in caps of 8, 16, 32).</summary>
    public const int TopKLocalMax = 32;

    /// <summary>The one-read top-k's list length an invocation holds for k (8, 16 or 32).</summary>
    public static int TopKCap(int k) => k <= 8 ? 8 : k <= 16 ? 16 : 32;

    /// <summary>indices[r][i] for i &lt; k ≤ cap (s32): the row's k first in Adad's order (Routing.Precedes: larger first, a
    /// NaN above every number, ties by the lower index), the row read once. Each invocation keeps its columns' first cap in
    /// order (a column ranks below an equal value of a lower column), then k rounds take the workgroup's first of the
    /// invocations' heads by the tree (the lowest index among equals) and the invocation holding it moves its list up. The
    /// same columns as TopKRows. A 32- or 64-bit float lane's values are ranked by integer keys (OrderKey: the same order,
    /// the comparisons at the integer rate). Arguments: x, indices, cols, k.</summary>
    public static SpvKernel TopKRowsLocal(Vulkan vk, Fmt f, Lane lane, int cap)
    {
        var lk = Kinds.Of(lane);
        bool keyed = lk is K.F32 or K.F64;
        var held = keyed ? (lk == K.F64 ? K.U64 : K.U32) : lk;
        var b = new Spv($"topk_local_{f.Name}_{lane.Name}_{cap}", ReduceThreads, vk.Controls);
        var x = b.Arg(0);
        var indices = b.Arg(1);
        var cols = b.Arg(2, K.U32);
        var k = b.Arg(3, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        var none = b.U(0xFFFFFFFF);
        var empty = keyed ? b.CBits(held, 0) : Identity(b, lk, "max");
        var vals = new Var[cap];
        var idx = new Var[cap];
        for (int j = 0; j < cap; j++) { vals[j] = b.Local(empty); idx[j] = b.Local(none); }

        // Whether (v, i) comes before (w, j) in the list's order (an empty slot last).
        V Before(V v, V i, V w, V j)
        {
            var (above, same) = keyed ? (v > w, v == w) : Rank(b, lane, v, w);
            return (i != none) & ((j == none) | above | (same & (i < j)));
        }

        // (value, column) into the list: down it by compare-and-swap (column `none`: nothing enters).
        void Insert(V value, V ci)
        {
            var cv = keyed ? OrderKey(b, value) : value;
            for (int j = 0; j < cap; j++)
            {
                var sv0 = vals[j].V;
                var si0 = idx[j].V;
                var take = Before(cv, ci, sv0, si0);
                vals[j].V = b.Select(take, cv, sv0);
                idx[j].V = b.Select(take, ci, si0);
                cv = b.Select(take, sv0, cv);
                ci = b.Select(take, si0, ci);
            }
        }

        if (f.IsBlock)
        {
            // A block format: runs of 8 values (one read of their block's scales; a run never crosses a block), the row's
            // first and last runs taking only the row's values.
            const int Run = 8;
            var g0 = start / (double)Run;
            var runs = (start + cols + (Run - 1)) / (double)Run - g0;
            b.For(tid, runs, b.U(ReduceThreads), gi =>
            {
                var at0 = (g0 + gi) * Run;
                var values = Codec.UnpackRun(b, f, x, at0, Run, lane);
                for (int q = 0; q < Run; q++)
                {
                    var at = at0 + q;
                    var inside = (at >= start) & (at - start < cols);
                    Insert(values[q], b.Select(inside, at - start, none));
                }
            });
        }
        else b.For(tid, cols, b.U(ReduceThreads), c => Insert(Codec.UnpackAt(b, f, x, start + c, lane), c));

        var sv = b.Shared(held, ReduceThreads);
        var si = b.Shared(K.U32, ReduceThreads);
        b.For(b.U(0), k, i =>
        {
            sv[tid] = vals[0].V;
            si[tid] = idx[0].V;
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
                    var take = Before(ov, oi, mv, mi);
                    sv[tid] = b.Select(take, ov, mv);
                    si[tid] = b.Select(take, oi, mi);
                });
                b.Barrier();
            }

            var chosen = si[0];
            b.If(tid == 0, () => b.Store(b.At(indices, row * k + i, 4), chosen));
            // The invocation holding it moves its list up (columns are each one invocation's: one holds it).
            var won = idx[0].V == chosen;
            for (int j = 0; j < cap - 1; j++)
            {
                vals[j].V = b.Select(won, vals[j + 1].V, vals[j].V);
                idx[j].V = b.Select(won, idx[j + 1].V, idx[j].V);
            }

            vals[cap - 1].V = b.Select(won, empty, vals[cap - 1].V);
            idx[cap - 1].V = b.Select(won, none, idx[cap - 1].V);
            b.Barrier();
        });
        return b.Build();
    }

    /// <summary>indices[r][i] for i &lt; k (s32): the row's k first in Adad's order (Routing.Precedes: larger first, a NaN
    /// above every number, ties by the lower index). Arguments: x, indices, cols, k.</summary>
    public static SpvKernel TopKRows(Vulkan vk, Fmt f, Lane lane)
    {
        var lk = Kinds.Of(lane);
        var b = new Spv($"topk_rows_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        var x = b.Arg(0);
        var indices = b.Arg(1);
        var cols = b.Arg(2, K.U32);
        var k = b.Arg(3, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        var none = b.U(0xFFFFFFFF);
        var taken = b.Shared(K.U32, MaxTopK);
        var sv = b.Shared(lk, ReduceThreads);
        var si = b.Shared(K.U32, ReduceThreads);
        b.For(b.U(0), k, i =>
        {
            // This invocation's first-ranked over its columns not already taken (its columns rise: an equal value keeps the
            // earlier).
            var best = b.Local(Identity(b, lk, "max"));
            var bestAt = b.Local(none);
            b.For(tid, cols, b.U(ReduceThreads), c =>
            {
                var isTaken = b.Local(b.Bool(false));
                b.For(b.U(0), i, t => isTaken.V = isTaken.V | (taken[t] == c));
                b.If(!isTaken.V, () =>
                {
                    var v = Codec.UnpackAt(b, f, x, start + c, lane);
                    var take = Rank(b, lane, v, best.V).Above | (bestAt.V == none);
                    best.V = b.Select(take, v, best.V);
                    bestAt.V = b.Select(take, c, bestAt.V);
                });
            });

            // The workgroup's first-ranked: Adad's order, the lowest index among equals.
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

            var chosen = si[0];
            b.If(tid == 0, () =>
            {
                taken[i] = chosen;
                b.Store(b.At(indices, row * k + i, 4), chosen);
            });
            b.Barrier();
        });
        return b.Build();
    }

    /// <summary>Adad's order on two lane values (Routing.Order): whether a ranks above c (larger, or a
    /// NaN against a number) and whether they rank the same (equal, −0 and +0 among them, or two NaNs).</summary>
    internal static (V Above, V Same) Rank(Spv b, Lane lane, V a, V c)
    {
        if (!lane.IsFloat) return (a > c, a == c);
        V na = b.IsNan(a), nc = b.IsNan(c);
        return ((a > c) | (na & !nc), (a == c) | (na & nc));
    }

    /// <summary>Whether a ranks above c in Adad's order (Rank's Above, from the values' kind).</summary>
    internal static V Above(Spv b, V a, V c) => a.Kind.IsFloat() ? (a > c) | (b.IsNan(a) & !b.IsNan(c)) : a > c;

    /// <summary>The first-ranked of a and c in Adad's order (Rank): the larger, a NaN when either is one (NMax gives the
    /// number); of −0 and +0, which rank the same, the one NMax gives. An integer kind's max.</summary>
    internal static V First(Spv b, V a, V c)
    {
        var m = b.Max(a, c);
        if (!a.Kind.IsFloat()) return m;
        return b.Select(b.IsNan(a), a, b.Select(b.IsNan(c), c, m));
    }

    /// <summary>v's order key (Routing.OrderKey): an unsigned value of v's width that orders as Routing.Order does: a NaN
    /// above every number and every NaN one key, −0 and +0 one key. A float's bits with the sign brought to the order (a
    /// positive one's top bit set, a negative one's inverted), a signed integer's with its top bit flipped, an unsigned
    /// one's as they are.</summary>
    internal static V OrderKey(Spv b, V v)
    {
        var kk = v.Kind.BitsKind();
        ulong sign = kk == K.U64 ? 0x8000000000000000UL : 0x80000000UL, ones = kk == K.U64 ? ulong.MaxValue : uint.MaxValue;
        if (v.Kind.IsFloat())
        {
            var raw = b.Bitcast(v, kk);
            var bits = b.Select((raw & (sign - 1)) == b.CBits(kk, 0), b.CBits(kk, 0), raw);   // −0 as +0, by the bits (a compare may flush a subnormal to 0)
            var key = b.Select(bits >= b.CBits(kk, sign), ~bits, bits | sign);
            return b.Select(b.IsNan(v), b.CBits(kk, ones), key);
        }

        return v.Kind.IsSigned() ? b.Bitcast(v, kk) ^ sign : v;
    }

    /// <summary>out[r·k + j] = x[r][index[r·k + j]] (s32 columns) over rows · k values, a thread per stored word: top-k's
    /// values, pick's with k = 1. A column outside the row (below 0 or past it; below 0 is past it unsigned) reads the
    /// row's first value, dropped: 0. Arguments: x, index, out, rows, cols, k.</summary>
    public static SpvKernel GatherByIndex(Vulkan vk, Fmt f)
    {
        var lane = f.Natural;
        var b = new Spv($"gather_by_index_{f.Name}", Threads, vk.Controls);
        var x = b.Arg(0);
        var index = b.Arg(1);
        var output = b.Arg(2);
        var rows = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var k = b.Arg(5, K.U32);
        var gid = b.GlobalX;
        var n = rows * k;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var word = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var at = gid * (double)f.Per + i;
                var inside = at < n;
                var safe = b.Min(at, n - 1);
                var col = b.Bitcast(b.Load(K.S32, b.At(index, safe, 4)), K.U32);
                var inRow = col < cols;
                var v = Codec.UnpackAt(b, f, x, safe / k * cols + b.Select(inRow, col, b.U(0)), lane);
                word = Codec.Pack(b, f, b.Select(inside & inRow, v, b.C(Kinds.Of(lane), 0)), lane, i, word);
            }

            b.Store(b.At(output, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    // (m, s) joined with a value v (sv = null: s gains exp(v − m)) or with another pair (v, sv): the larger max kept, the
    // other side's sum scaled by e = exp(−|v − m|) — one exp (CUDA's MaxSumStep).
    private static (V M, V S) MaxSumStep(Spv b, V m, V s, V v, V? sv)
    {
        var d = v - m;
        var ex = Fn.Exp(b, -b.Abs(d));
        var other = sv ?? b.C(v.Kind, 1);
        var up = b.Fma(s, ex, other);
        var keep = b.Fma(other, ex, s);
        return (b.Max(m, v), b.Select(d > 0, up, keep));
    }

    /// <summary>The top-k's values as log-probabilities (IRouting.LogSoftmaxTopK), a workgroup a row: targets holds `count`
    /// columns a row (TopK's: the first the row's first-ranked, every one in the row). m = x at the first column; Σ exp(x − m)
    /// over the row, each invocation four sums over its columns (a fixed order; a block format's read in runs of 8) joined by
    /// the workgroup's tree; then an invocation a chosen column: (x[t] − m) − log Σ as a lane value at losses[row·count + j].
    /// Arguments: x, targets, losses, cols, count.</summary>
    public static SpvKernel LogProbRows(Vulkan vk, Fmt f, Lane lane)
    {
        const int Sums = 4;
        var k = Kinds.Of(lane);
        var b = new Spv($"log_prob_rows_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        V x = b.Arg(0), targets = b.Arg(1), losses = b.Arg(2);
        V cols = b.Arg(3, K.U32), count = b.Arg(4, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        var last = cols - 1;
        var first = b.Load(K.U32, b.At(targets, row * count, 4));
        var m = Codec.UnpackAt(b, f, x, start + first, lane);
        var sums = new Var[Sums];
        for (int u = 0; u < Sums; u++) sums[u] = b.Local(b.C(k, 0));
        if (f.IsBlock)
        {
            // A block format: runs of 8 values (one read of their block's scales), value q of a run into sum q % 4, the row's
            // first and last runs taking only the row's values.
            const int Run = 8;
            var g0 = start / (double)Run;
            var runs = (start + cols + (Run - 1)) / (double)Run - g0;
            b.For(tid, runs, b.U(ReduceThreads), gi =>
            {
                var at0 = (g0 + gi) * Run;
                var values = Codec.UnpackRun(b, f, x, at0, Run, lane);
                for (int q = 0; q < Run; q++)
                {
                    var at = at0 + q;
                    var inside = (at >= start) & (at - start < cols);
                    sums[q % Sums].V = sums[q % Sums].V + b.Select(inside, Fn.Exp(b, values[q] - m), b.C(k, 0));
                }
            });
        }
        else b.For(tid, cols, b.U(Sums * ReduceThreads), i =>
        {
            for (int u = 0; u < Sums; u++)
            {
                // Column i + u·threads (one past the row reads the last column, its term dropped).
                var c = i + (double)(u * ReduceThreads);
                var term = Fn.Exp(b, Codec.UnpackAt(b, f, x, start + b.Min(c, last), lane) - m);
                sums[u].V = sums[u].V + b.Select(c < cols, term, b.C(k, 0));
            }
        });
        var total = GroupReduce(b, b.Shared(k, ReduceThreads), tid, (sums[0].V + sums[1].V) + (sums[2].V + sums[3].V), static (p, q) => p + q, ReduceThreads);
        var logSum = Fn.Log(b, total);
        b.For(tid, count, b.U(ReduceThreads), j =>
        {
            var at = row * count + j;
            var col = b.Load(K.U32, b.At(targets, at, 4));
            var chosen = Codec.UnpackAt(b, f, x, start + col, lane);
            b.Store(b.At(losses, at, lane.Size), (chosen - m) - logSum);
        });
        return b.Build();
    }

    /// <summary>loss[r] = log Σ_c exp(x[r][c]) − x[r][t[r]] as lane values (stored as the type after), a workgroup per row;
    /// weighted: times the row's weight (as the type). Arguments: x, targets, losses, cols (, weights).</summary>
    public static SpvKernel CrossEntropyRows(Vulkan vk, Fmt f, Lane lane, bool weighted)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"cross_entropy_rows_{f.Name}_{lane.Name}{(weighted ? "_w" : "")}", ReduceThreads, vk.Controls);
        var x = b.Arg(0);
        var targets = b.Arg(1);
        var losses = b.Arg(2);
        var cols = b.Arg(3, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        // The running pair starts at the lane's lowest finite value (not −∞: m − M stays finite when both are empty).
        var lowest = b.C(k, k == K.F64 ? -double.MaxValue : -float.MaxValue);
        var m = b.Local(lowest);
        var s = b.Local(b.C(k, 0));
        b.For(tid, cols, b.U(ReduceThreads), c =>
        {
            var (nm, ns) = MaxSumStep(b, m.V, s.V, Codec.UnpackAt(b, f, x, start + c, lane), null);
            m.V = nm;
            s.V = ns;
        });

        var shM = b.Shared(k, ReduceThreads);
        var shS = b.Shared(k, ReduceThreads);
        shM[tid] = m.V;
        shS[tid] = s.V;
        b.Barrier();
        for (int step = ReduceThreads / 2; step >= 1; step /= 2)
        {
            int at = step;
            b.If(tid < at, () =>
            {
                var (jm, js) = MaxSumStep(b, shM[tid], shS[tid], shM[tid + at], shS[tid + at]);
                shM[tid] = jm;
                shS[tid] = js;
            });
            b.Barrier();
        }

        b.If(tid == 0, () =>
        {
            var t = b.Bitcast(b.Load(K.S32, b.At(targets, row, 4)), K.U32);
            var xt = Codec.UnpackAt(b, f, x, start + t, lane);
            var loss = (shM[0] - xt) + Fn.Log(b, shS[0]);
            if (weighted) loss = loss * Codec.UnpackAt(b, f, b.Arg(4), row, lane);
            b.Store(b.At(losses, row, lane.Size), loss);
        });
        return b.Build();
    }

    /// <summary>picks[r] = the first column of positive weight at which the running sum of the positive weights passes
    /// u[r] times their sum, strictly (the last column of positive weight when none; −1 where the sum is not a positive
    /// finite number or u[r] is a NaN): Routing.SampleReference's rule, a workgroup per row, each invocation a contiguous
    /// chunk. Arguments: x, u, picks, cols.</summary>
    public static SpvKernel SampleRows(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"sample_rows_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        var x = b.Arg(0);
        var u = b.Arg(1);
        var picks = b.Arg(2);
        var cols = b.Arg(3, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        var chunk = (cols + (ReduceThreads - 1)) >> 8;
        var c0 = tid * chunk;
        var c1 = b.Min(c0 + chunk, cols);
        // This invocation's chunk sum of the positive weights (a weight below 0 adds nothing, a NaN carries into it).
        var zero = b.C(k, 0);
        var sum = b.Local(zero);
        b.For(c0, c1, c =>
        {
            var v = Codec.UnpackAt(b, f, x, start + c, lane);
            sum.V = sum.V + b.Select(v < zero, zero, v);
        });
        var sums = b.Shared(k, ReduceThreads);
        var total = b.Shared(k, 1);
        sums[tid] = sum.V;
        b.Barrier();
        // The exclusive prefix of the chunk sums and the total, by the first invocation.
        b.If(tid == 0, () =>
        {
            var acc = b.Local(b.C(k, 0));
            b.For(b.U(0), b.U(ReduceThreads), t =>
            {
                var sv = sums[t];
                sums[t] = acc.V;
                acc.V = acc.V + sv;
            });
            total[0] = acc.V;
        });
        b.Barrier();
        var threshold = Codec.UnpackAt(b, f, u, row, lane) * total[0];
        // The first crossing in this invocation's chunk at a column of positive weight (cols when none), the least over the
        // workgroup; the chunk's last column of positive weight (+ 1; 0: none), the greatest over it.
        var run = b.Local(sums[tid]);
        var found = b.Local(cols);
        var lastMine = b.Local(b.U(0));
        b.For(c0, c1, c =>
        {
            var w = Codec.UnpackAt(b, f, x, start + c, lane);
            var positive = w > zero;
            run.V = run.V + b.Select(positive, w, zero);
            lastMine.V = b.Select(positive, c + 1, lastMine.V);
            found.V = b.Select((run.V > threshold) & positive & (found.V == cols), c, found.V);
        });
        var least = GroupReduce(b, b.Shared(K.U32, ReduceThreads), tid, found.V, b.Min, ReduceThreads);
        var lastPos = GroupReduce(b, b.Shared(K.U32, ReduceThreads), tid, lastMine.V, b.Max, ReduceThreads);
        // −1 for a total that is not a positive finite number, or a NaN u.
        var t = total[0];
        var draws = (t > zero) & !b.IsInf(t) & !b.IsNan(threshold);
        b.If(tid == 0, () => b.Store(b.At(picks, row, 4), b.Bitcast(b.Select(draws, b.Select(least >= cols, lastPos - 1, least), b.U(0xFFFFFFFF)), K.S32)));
        return b.Build();
    }
}
