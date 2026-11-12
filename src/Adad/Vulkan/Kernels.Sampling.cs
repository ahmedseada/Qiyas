// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on Vulkan (Sampling.cs), CUDA's (Cuda/Ptx.Sampling.cs), a workgroup a row in two kernels:
//   SampleWeights  the row's scores read where they are stored into its weights in lane storage: s = x / temperature in
//                  the lane, the largest by the workgroup's tree (a NaN is no score), then exp(s − largest) (1 at each +∞
//                  of a row whose largest is +∞; 0 for a row with no score).
//   SampleDraw     over the weights (lane storage: one kernel a lane, whatever the type): each rule's threshold by
//                  narrowing the lane's bit patterns between 0 and 1 — a pass of the row a step, an invocation its
//                  columns against the step's values, the workgroup's trees of the counts (top-k: sixteen ways a pass,
//                  the weights' bits compared as integers) or of the sums (the nucleus: sixteen ways in the f32 lane, two
//                  in the f64 lane, a sum a value being an add a weight; where 256 weights or fewer are at least top-k's
//                  threshold they are gathered into workgroup memory instead, an invocation a gathered weight summing
//                  those at least its own) — then, the weights below the threshold taken as 0 where they are read, the kept
//                  weights' chunk sums and their prefix, the seed's uniform at (step, row) with the step read where it
//                  is, and the first kept column at which the running sum reaches it (the last kept one when rounding
//                  leaves none; −1 for a row that keeps none).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>w (lane storage, rows × cols) = each row's weights. Slots: x (f; row r at r·stride + off), w, cols, stride,
    /// off, temperature (lane); a workgroup a row.</summary>
    public static SpvKernel SampleWeights(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"sample_weights_{f.Name}_{lane.Name}", ReduceThreads, vk.Controls);
        V px = b.Arg(0), pw = b.Arg(1), cols = b.Arg(2, K.U32), stride = b.Arg(3, K.U32), off = b.Arg(4, K.U32), temp = b.Arg(5, k);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var start = row * stride + off;
        var wbase = row * cols;
        V lowest = b.C(k, double.NegativeInfinity), highest = b.C(k, double.PositiveInfinity), zero = b.C(k, 0);
        V Score(V c) => Codec.UnpackAt(b, f, px, start + c, lane) / temp;
        var best = b.Local(lowest);
        b.For(tid, cols, b.U(ReduceThreads), c =>
        {
            var s = Score(c);
            best.V = b.Max(best.V, b.Select(b.IsNan(s), lowest, s));
        });
        var m = GroupReduce(b, b.Shared(k, ReduceThreads), tid, best.V, b.Max, ReduceThreads);
        V none = m <= lowest, top = m >= highest;
        b.For(tid, cols, b.U(ReduceThreads), c =>
        {
            var s = Score(c);
            var e = b.Select(b.IsNan(s), zero, Fn.Exp(b, s - m));
            e = b.Select(top, b.Select(s >= highest, b.C(k, 1), zero), e);
            b.Store(b.At(pw, wbase + c, lane.Size), b.Select(none, zero, e));
        });
        return b.Build();
    }

    /// <summary>picks[row] and (prob not 0) prob[row] (lane storage) from the row's weights w (lane storage, rows × cols,
    /// read only). Slots: w, picks, prob, step (the device's, read where it is), cols, k (0:
    /// no top-k), the seed's key A, p (0: no nucleus), q (0: no min-p) (lane), the key's B; a workgroup a row.</summary>
    public static SpvKernel SampleDraw(Vulkan vk, Lane lane)
    {
        var k = Kinds.Of(lane);
        var bk = k.BitsKind();
        ulong oneBits = lane.Wide ? 0x3FF0000000000000UL : 0x3F800000UL;
        var b = new Spv($"sample_draw_{lane.Name}", ReduceThreads, vk.Controls);
        V pw = b.Arg(0), picks = b.Arg(1), prob = b.Arg(2), pstep = b.Arg(3);
        V cols = b.Arg(4, K.U32), kk = b.Arg(5, K.U32), key = b.Arg(6, K.U32), p = b.Arg(7, k), q = b.Arg(8, k), kb = b.Arg(9, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var wbase = row * cols;
        var zero = b.C(k, 0);
        var stride = b.U(ReduceThreads);
        var counts = b.Shared(K.U32, ReduceThreads);
        var sums = b.Shared(k, ReduceThreads);
        var whole = b.Shared(k, 1);
        var gathered = b.Shared(k, ReduceThreads);
        V Stored(V c) => b.Load(k, b.At(pw, wbase + c, lane.Size));
        var lo = b.Local(b.CBits(bk, 0));
        var hi = b.Local(b.CBits(bk, oneBits));
        var t = b.Local(zero);                                                  // the threshold: 0 keeps every weight
        // A step's values between lo and hi: lo + j · max(1, (hi − lo + 1) / ways) held to hi, j = 1 .. ways − 1 (the rule
        // holds up to some value and fails past it: the step keeps the last value at which it holds as lo and the bits below
        // the first at which it fails as hi).
        const int CountWays = 16;
        int sumWays = lane.Wide ? 2 : 16;
        V[] Step(int ways)
        {
            var step = b.Max((hi.V - lo.V + 1.0) >> int.Log2(ways), b.CBits(bk, 1));
            var at = new V[ways];
            for (int j = 1; j < ways; j++) at[j] = b.Min(lo.V + step * (double)j, hi.V);
            return at;
        }

        void Narrow(V[] at, V[] holds)
        {
            V low = lo.V, high = hi.V;
            for (int j = 1; j < at.Length; j++) low = b.Select(holds[j], at[j], low);
            for (int j = at.Length - 1; j >= 1; j--) high = b.Select(holds[j], high, at[j] - 1.0);
            lo.V = low;
            hi.V = high;
        }

        // Top-k: the largest value at which k weights or more are at least it (a count of exactly k ends the search).
        b.If(kk != b.U(0), () =>
        {
            b.While(() => lo.V < hi.V, () =>
            {
                var at = Step(CountWays);
                var count = new Var[CountWays];
                for (int j = 1; j < CountWays; j++) count[j] = b.Local(b.U(0));
                b.For(tid, cols, stride, c =>
                {
                    // A weight is 0 or above: its bits order as it does (integer compares, whatever the lane).
                    var bits = b.Load(bk, b.At(pw, wbase + c, lane.Size));
                    for (int j = 1; j < CountWays; j++) count[j].V = count[j].V + b.Select(bits >= at[j], b.U(1), b.U(0));
                });
                var holds = new V[CountWays];
                var exact = new V[CountWays];
                for (int j = 1; j < CountWays; j++)
                {
                    var total = GroupReduce(b, counts, tid, count[j].V, static (x, y) => x + y, ReduceThreads);
                    holds[j] = total >= kk;
                    exact[j] = total == kk;
                }

                Narrow(at, holds);
                for (int j = 1; j < CountWays; j++)
                {
                    lo.V = b.Select(exact[j], at[j], lo.V);
                    hi.V = b.Select(exact[j], at[j], hi.V);
                }
            });
            t.V = b.Bitcast(lo.V, k);
        });

        // The nucleus: the largest value at which the weights at least it sum to p of the weights at least t.
        // The weights at least t: where they fit a workgroup's invocations (top-k has left few), gathered into workgroup
        // memory — an invocation's count, the counts' exclusive prefix by the first, each invocation's weights at its place —
        // and the nucleus read from them: invocation i's weight v, the gathered weights at least v summed in their order, v
        // kept where that reaches p of them all; the largest such v is the threshold (no pass of the row a step).
        var few = b.Local(b.Bool(false));
        b.If(p > 0.0, () =>
        {
            var from = t.V;
            var mineCount = b.Local(b.U(0));
            b.For(tid, cols, stride, c => mineCount.V = mineCount.V + b.Select(Stored(c) >= from, b.U(1), b.U(0)));
            var n = GroupReduce(b, counts, tid, mineCount.V, static (x, y) => x + y, ReduceThreads);
            few.V = n <= (double)ReduceThreads;
            b.If(few.V, () =>
            {
                counts[tid] = mineCount.V;
                b.Barrier();
                b.If(tid == 0, () =>
                {
                    var acc = b.Local(b.U(0));
                    b.For(b.U(0), b.U(ReduceThreads), i =>
                    {
                        var v = counts[i];
                        counts[i] = acc.V;
                        acc.V = acc.V + v;
                    });
                });
                b.Barrier();
                var place = b.Local(counts[tid]);
                b.For(tid, cols, stride, c =>
                {
                    var e = Stored(c);
                    b.If(e >= from, () =>
                    {
                        gathered[place.V] = e;
                        place.V = place.V + 1.0;
                    });
                });
                b.Barrier();
                var own = gathered[b.Min(tid, b.U(ReduceThreads - 1))];
                var all = b.Local(zero);
                var above = b.Local(zero);
                b.For(b.U(0), n, i =>
                {
                    var w = gathered[i];
                    all.V = all.V + w;
                    above.V = above.V + b.Select(w >= own, w, zero);
                });
                var candidate = b.Select((tid < n) & (above.V >= p * all.V), own, zero);
                t.V = b.Max(from, GroupReduce(b, sums, tid, candidate, b.Max, ReduceThreads));
            });
        });
        b.If((p > 0.0) & !few.V, () =>
        {
            var from = t.V;
            var part = b.Local(zero);
            b.For(tid, cols, stride, c =>
            {
                var e = Stored(c);
                part.V = part.V + b.Select(e >= from, e, zero);
            });
            var target = p * GroupReduce(b, sums, tid, part.V, static (x, y) => x + y, ReduceThreads);
            lo.V = b.Bitcast(from, bk);
            hi.V = b.CBits(bk, oneBits);
            b.While(() => lo.V < hi.V, () =>
            {
                var at = Step(sumWays);
                var sum = new Var[sumWays];
                for (int j = 1; j < sumWays; j++) sum[j] = b.Local(zero);
                b.For(tid, cols, stride, c =>
                {
                    var e = Stored(c);
                    for (int j = 1; j < sumWays; j++) sum[j].V = sum[j].V + b.Select(e >= b.Bitcast(at[j], k), e, zero);
                });
                var holds = new V[sumWays];
                for (int j = 1; j < sumWays; j++) holds[j] = GroupReduce(b, sums, tid, sum[j].V, static (x, y) => x + y, ReduceThreads) >= target;
                Narrow(at, holds);
            });
            t.V = b.Bitcast(lo.V, k);
        });

        var threshold = b.Max(t.V, q);                                          // min-p
        // The kept weight of column c: its weight where that is at least the threshold, else 0.
        V Weight(V c)
        {
            var e = Stored(c);
            return b.Select(e >= threshold, e, zero);
        }

        // The kept weights' sums over this invocation's chunk of consecutive columns, their exclusive prefix and the total by
        // the first invocation.
        var chunk = (cols + (double)(ReduceThreads - 1)) / (double)ReduceThreads;
        var c0 = b.Min(tid * chunk, cols);
        var c1 = b.Min(c0 + chunk, cols);
        var mine = b.Local(zero);
        b.For(c0, c1, c => mine.V = mine.V + Weight(c));
        sums[tid] = mine.V;
        b.Barrier();
        b.If(tid == 0, () =>
        {
            var acc = b.Local(zero);
            b.For(b.U(0), b.U(ReduceThreads), i =>
            {
                var v = sums[i];
                sums[i] = acc.V;
                acc.V = acc.V + v;
            });
            whole[b.U(0)] = acc.V;
        });
        b.Barrier();
        var total2 = whole[b.U(0)];
        // The draw: Draws.Uniform(key, step, row) = Bits(Stream(key, step), row) · 2^−32 in the lane, times the total.
        var u = b.Convert(Draw(Draw(key, kb, b.Load(K.U32, pstep)), kb, row), k) * (1.0 / 4294967296);
        var target2 = u * total2;
        var run = b.Local(sums[tid]);
        var found = b.Local(cols);
        var lastMine = b.Local(b.U(0));
        b.For(c0, c1, c =>
        {
            var w = Weight(c);
            run.V = run.V + w;
            found.V = b.Select((w > 0.0) & (run.V >= target2) & (found.V == cols), c, found.V);
            lastMine.V = b.Select(w > 0.0, c + 1.0, lastMine.V);                   // the chunk's columns ascend
        });
        var least = GroupReduce(b, counts, tid, found.V, b.Min, ReduceThreads);
        var lastKept = GroupReduce(b, counts, tid, lastMine.V, b.Max, ReduceThreads);      // the last kept column (+ 1; 0: none)
        b.If(tid == 0, () =>
        {
            var pick = b.Select(least >= cols, lastKept - 1.0, least);              // the last kept column; −1 when none
            b.Store(b.At(picks, row, 4), pick);
            b.If(prob != b.UL(0), () =>
            {
                var valid = b.Bitcast(pick, K.S32) >= b.S(0);
                var w = Weight(b.Select(valid, pick, b.U(0)));
                b.Store(b.At(prob, row, lane.Size), b.Select(valid, w / total2, zero));
            });
        });
        return b.Build();
    }
}
