// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Routing, sampling and the cross-entropy loss on Vulkan (Routing.cs; CUDA's Ptx.Routing.cs): a workgroup per row for
// top-k (k passes of the workgroup's first largest over the columns not yet taken) and for sampling (each invocation's
// chunk summed, the chunks' prefix by the first, the first crossing found per chunk and the least taken) and for the
// loss (a running max and sum of exponentials per invocation, joined by the tree). Values are stored a thread per word
// by gathers (top-k's values, pick) and the losses kept as lane values and stored once, so no two invocations store into
// one word.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int MaxTopK = 256;

    /// <summary>indices[r][i] for i &lt; k (s32): the row's largest first, ties by the lower index. Arguments: x, indices, cols, k.</summary>
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
            // This invocation's best over its columns not already taken.
            var best = b.Local(Identity(b, lk, "max"));
            var bestAt = b.Local(none);
            b.For(tid, cols, b.U(ReduceThreads), c =>
            {
                var isTaken = b.Local(b.Bool(false));
                b.For(b.U(0), i, t => isTaken.V = isTaken.V | (taken[t] == c));
                b.If(!isTaken.V, () =>
                {
                    var v = Codec.UnpackAt(b, f, x, start + c, lane);
                    var take = (v > best.V) | (bestAt.V == none);
                    best.V = b.Select(take, v, best.V);
                    bestAt.V = b.Select(take, c, bestAt.V);
                });
            });

            // The workgroup's best: the largest value, the lowest index among equals.
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
                    var take = ((ov > mv) | ((ov == mv) & (oi < mi)) | (mi == none)) & (oi != none);
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

    /// <summary>out[r·k + j] = x[r][index[r·k + j]] (s32 columns) over rows · k values, a thread per stored word: top-k's
    /// values, pick's with k = 1. Arguments: x, index, out, rows, cols, k.</summary>
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
                var v = Codec.UnpackAt(b, f, x, safe / k * cols + col, lane);
                word = Codec.Pack(b, f, b.Select(inside, v, b.C(Kinds.Of(lane), 0)), lane, i, word);
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
    /// columns a row (TopK's: the first the row's largest; one below 0 or past the row: none). m = x at the first column;
    /// Σ exp(x − m) over the row, each invocation four sums over its columns (a fixed order) joined by the workgroup's tree;
    /// then an invocation a chosen column: (x[t] − m) − log Σ as a lane value at losses[row·count + j] (`none` for none).
    /// Arguments: x, targets, losses, cols, count, none (lane).</summary>
    public static SpvKernel LogProbRows(Vulkan vk, Fmt f, Lane lane)
    {
        const int Sums = 4;
        var k = Kinds.Of(lane);
        var b = new Spv($"log_prob_rows_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        V x = b.Arg(0), targets = b.Arg(1), losses = b.Arg(2);
        V cols = b.Arg(3, K.U32), count = b.Arg(4, K.U32), none = b.Arg(5, k);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        var last = cols - 1;
        var first = b.Load(K.U32, b.At(targets, row * count, 4));
        var m = Codec.UnpackAt(b, f, x, start + b.Select(first < cols, first, b.U(0)), lane);
        var sums = new Var[Sums];
        for (int u = 0; u < Sums; u++) sums[u] = b.Local(b.C(k, 0));
        b.For(tid, cols, b.U(Sums * ReduceThreads), i =>
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
            var valid = col < cols;                                                   // −1 (no column) is past every row
            var chosen = Codec.UnpackAt(b, f, x, start + b.Select(valid, col, b.U(0)), lane);
            b.Store(b.At(losses, at, lane.Size), b.Select(valid, (chosen - m) - logSum, none));
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

    /// <summary>picks[r] = the first column whose running sum reaches u[r] · Σ row (the last column when none): a workgroup per
    /// row, each invocation a contiguous chunk. Arguments: x, u, picks, cols.</summary>
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
        // This invocation's chunk sum.
        var sum = b.Local(b.C(k, 0));
        b.For(c0, c1, c => sum.V = sum.V + Codec.UnpackAt(b, f, x, start + c, lane));
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
        // The first crossing in this invocation's chunk (cols when none), then the least over the workgroup.
        var run = b.Local(sums[tid]);
        var found = b.Local(cols);
        b.For(c0, c1, c =>
        {
            run.V = run.V + Codec.UnpackAt(b, f, x, start + c, lane);
            found.V = b.Select((run.V >= threshold) & (found.V == cols), c, found.V);
        });
        var least = GroupReduce(b, b.Shared(K.U32, ReduceThreads), tid, found.V, b.Min, ReduceThreads);
        b.If(tid == 0, () => b.Store(b.At(picks, row, 4), b.Bitcast(b.Select(least >= cols, cols - 1, least), K.S32)));
        return b.Build();
    }
}
