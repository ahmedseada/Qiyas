// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on Vulkan (Abstraction/Sampling.cs has the rule; CUDA's in Cuda/Ptx.Sampling.cs), a workgroup a
// row in two kernels:
//   SampleWeights  the row's scores read where they are stored, in the type's float lane: m the first-ranked in Adad's
//                  order (First: a NaN where the row holds one, else the largest) by the workgroup's tree, then the
//                  weights exp((x − m) / t) in lane storage (NaN throughout a row whose m is not finite).
//   SampleDraw     each rule's prefix of the row's order, by the scores' order keys (Routing.Order's integer form: a
//                  stored value as an unsigned integer that orders as Adad's order does, made where it is read): top-k's
//                  k-th key by the digits of its offset from the row's least, eight bits a pass, for a float or block
//                  format (an invocation its columns tid, tid + 256, …; a 256-bin count by workgroup atomics, scanned
//                  from the top), by narrowing the keys' range sixteen ways a pass for an integer format (the
//                  workgroup's trees of the counts), its key's columns cut by column order (an invocation a
//                  chunk of consecutive columns, the chunks' counts' prefix); the nucleus over top-k's columns gathered
//                  into workgroup memory where k fits a workgroup's invocations (each ranks itself among them, the first
//                  sums them in order), else by narrowing the keys' range sixteen ways with the weights' sums and its
//                  key's columns cut where the running sum reaches the target; min-p a weight at least q. Then the kept
//                  weights' chunk sums and their prefix, the seed's uniform at (step, row) with the step read where it
//                  is, and the column at which the running sum passes it (the last kept column of positive weight when
//                  rounding leaves none; −1 and a NaN probability for a row whose sum is not positive).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>The most columns top-k may keep for the nucleus to take them gathered (an invocation a column).</summary>
    public const int SampleGathered = ReduceThreads;

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
        V lowest = b.C(k, double.NegativeInfinity), highest = b.C(k, double.PositiveInfinity);
        V Value(V c) => Codec.UnpackAt(b, f, px, start + c, lane);
        // The first-ranked value in Adad's order (First: a NaN where the row holds one, else the largest).
        var best = b.Local(lowest);
        b.For(tid, cols, b.U(ReduceThreads), c => best.V = First(b, best.V, Value(c)));
        var m = GroupReduce(b, b.Shared(k, ReduceThreads), tid, best.V, (x, y) => First(b, x, y), ReduceThreads);
        var finite = (m > lowest) & (m < highest);
        b.For(tid, cols, b.U(ReduceThreads), c =>
        {
            var e = b.Select(finite, Fn.Exp(b, (Value(c) - m) / temp), b.C(k, double.NaN));   // x − m is NaN at m's own column
            b.Store(b.At(pw, wbase + c, lane.Size), e);
        });
        return b.Build();
    }

    /// <summary>Whether SampleDraw makes each weight where a pass needs it (SampleWeights's operations on the value read where
    /// it is stored: the same weight) instead of reading it from lane storage: in the f32 lane, where a weight is some
    /// fifty instructions at the device's full rate and the stored weights a write and two or more reads of a row of lane
    /// values (four bytes a column for a type of one); not in the f64 lane, whose exp and quotient run at its f64 rate.</summary>
    public static bool SampleWeighsInPlace(Lane lane) => !lane.Wide;

    /// <summary>Whether SampleDraw finds top-k's k-th key by 8-bit digits (workgroup atomics) rather than by narrowing its range
    /// sixteen ways a pass: for a float or block format, whose scores spread over many keys (measured: sample_filtered f32
    /// 7.74 ms against 5.65, q4_0 9.56 against 5.78); not for an integer format, whose few values' adds wait on one
    /// another's bins (int4 2.68 against 3.59, uint8 4.53 against 4.74).</summary>
    public static bool RadixTopK(Fmt f) => f.IsFloat || f.IsBlock;

    /// <summary>The rules a call's SampleDraw holds code for (SampleRules): top-k (k below cols), the nucleus (p below 1 in
    /// the lane), the nucleus over top-k's columns gathered (both, k at most SampleGathered). A kernel without a rule's
    /// code takes the rule's absence; each kept rule still tests its argument as before.</summary>
    public readonly record struct SampleRules(bool TopK, bool Nucleus)
    {
        public bool Gathered(int k) => TopK && Nucleus && k <= SampleGathered;

        public static SampleRules Of(SamplingArgs a, Lane lane) => new(a.TopK < a.Cols, lane.Wide ? a.TopP < 1 : (float)a.TopP < 1f);
    }

    /// <summary>picks[row] and (prob not 0) prob[row] (lane storage) from the row's scores x (f, where they are stored) and
    /// weights w (lane storage, rows × cols). Slots: x, w, picks, prob, step (the device's, read where it is), cols, stride,
    /// off, k (cols: no top-k), the seed's key A, p (1: no nucleus), q (0: no min-p) (lane), the key's B; a workgroup a
    /// row. The code of the rules in `rules` only (gathered: the nucleus's gathered form, else its narrowing).</summary>
    public static SpvKernel SampleDraw(Vulkan vk, Fmt f, Lane lane, SampleRules rules, bool gathered)
    {
        var k = Kinds.Of(lane);
        var natural = f.Natural;
        var nk = Kinds.Of(natural);
        var kk = nk.BitsKind();
        ulong ones = kk == K.U64 ? ulong.MaxValue : uint.MaxValue;
        bool inPlace = SampleWeighsInPlace(lane);
        bool anyRule = rules.TopK || rules.Nucleus;
        var b = new Spv($"sample_draw_{f.Name}_{lane.Name}{(inPlace ? "_w" : "")}{(rules.TopK ? "_k" : "")}{(rules.Nucleus ? (gathered ? "_pg" : "_p") : "")}",
            ReduceThreads, vk.Controls);
        V px = b.Arg(0), pw = b.Arg(1), picks = b.Arg(2), prob = b.Arg(3), pstep = b.Arg(4);
        V cols = b.Arg(5, K.U32), stride = b.Arg(6, K.U32), off = b.Arg(7, K.U32), topK = b.Arg(8, K.U32), ka = b.Arg(9, K.U32);
        V p = b.Arg(10, k), q = b.Arg(11, k), kb = b.Arg(12, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var wbase = row * cols;
        var start = row * stride + off;
        var zero = b.C(k, 0);
        var none = b.U(uint.MaxValue);
        var strided = b.U(ReduceThreads);
        var counts = b.Shared(K.U32, ReduceThreads);
        var sums = b.Shared(k, ReduceThreads);
        var keyTree = anyRule ? b.Shared(kk, ReduceThreads) : default;
        var total = b.Shared(k, 1);
        Arr bound = default, boundCol = default, gKey = default, gWeight = default, gCol = default, rKey = default, rWeight = default, rCol = default;
        if (gathered)
        {
            bound = b.Shared(kk, 1);
            boundCol = b.Shared(K.U32, 1);
            gKey = b.Shared(kk, SampleGathered);
            gWeight = b.Shared(k, SampleGathered);
            gCol = b.Shared(K.U32, SampleGathered);
            rKey = b.Shared(kk, SampleGathered);
            rWeight = b.Shared(k, SampleGathered);
            rCol = b.Shared(K.U32, SampleGathered);
        }

        V hasTopK = topK < cols, hasNucleus = p < 1.0, ranked = hasTopK | hasNucleus;
        V gather = hasTopK & hasNucleus & (topK <= (double)SampleGathered);

        // Column c's order key (OrderKey), read where it is stored in the natural lane.
        V KeyOf(V c) => OrderKey(b, Codec.UnpackAt(b, f, px, start + c, natural));

        // Column c's weight: read from lane storage (w), or made in place (slot 1 then the temperature in the lane):
        // SampleWeights's m (the first-ranked, by the same tree) once, then its exp((x − m) / t) where a pass needs a weight.
        Func<V, V> Stored;
        if (inPlace)
        {
            var temp = b.Bitcast(b.Convert(pw, K.U32), K.F32);
            V lowest = b.C(k, double.NegativeInfinity), highest = b.C(k, double.PositiveInfinity);
            V Value(V c) => Codec.UnpackAt(b, f, px, start + c, lane);
            var best = b.Local(lowest);
            b.For(tid, cols, b.U(ReduceThreads), c => best.V = First(b, best.V, Value(c)));
            var m = GroupReduce(b, sums, tid, best.V, (x, y) => First(b, x, y), ReduceThreads);
            var finite = (m > lowest) & (m < highest);
            Stored = c => b.Select(finite, Fn.Exp(b, (Value(c) - m) / temp), b.C(k, double.NaN));
        }
        else Stored = c => b.Load(k, b.At(pw, wbase + c, lane.Size));

        // The prefix's last column (key, column): a column is within it when its key is larger, or equal at or before it.
        var bKey = b.Local(b.CBits(kk, 0));
        var bCol = b.Local(none);                                             // every column until a rule cuts
        V Holds(V key, V c) => (key > bKey.V) | ((key == bKey.V) & (c <= bCol.V));

        // This invocation's chunk of consecutive columns.
        var chunk = (cols + (double)(ReduceThreads - 1)) / (double)ReduceThreads;
        var c0 = b.Min(tid * chunk, cols);
        var c1 = b.Min(c0 + chunk, cols);

        // The exclusive prefix of the invocations' values in `arr` by the first invocation (each then holds what comes before
        // its chunk), the total into `whole` where given.
        void Prefix(Arr arr, Arr? whole)
        {
            b.Barrier();
            b.If(tid == 0, () =>
            {
                var acc = b.Local(arr.Kind.IsFloat() ? b.C(arr.Kind, 0) : b.CBits(arr.Kind, 0));
                b.For(b.U(0), b.U(ReduceThreads), i =>
                {
                    var v = arr[i];
                    arr[i] = acc.V;
                    acc.V = acc.V + v;
                });
                if (whole is { } w) w[b.U(0)] = acc.V;
            });
            b.Barrier();
        }

        // A step's values between lo and hi (keys): lo + j·step held to hi, step = (hi − lo) / ways + 1, j = 1 .. ways − 1 (the
        // rule holds up to some key and fails past it: lo becomes the last at which it holds, hi the key below the first at
        // which it fails).
        const int Ways = 16;
        V[] Step(V lo, V hi)
        {
            var span = hi - lo;
            var step = (span >> int.Log2(Ways)) + 1.0;
            var at = new V[Ways];
            for (int j = 1; j < Ways; j++) at[j] = lo + b.Min(step * (double)j, span);
            return at;
        }

        void Narrow(Var lo, Var hi, V[] at, V[] holds)
        {
            V low = lo.V, high = hi.V;
            for (int j = 1; j < Ways; j++) low = b.Select(holds[j], at[j], low);
            for (int j = Ways - 1; j >= 1; j--) high = b.Select(holds[j], high, at[j] - 1.0);
            lo.V = low;
            hi.V = high;
        }

        // The cut of key `key`'s columns (within the prefix so far) by column order: the first at which the count (n) or the
        // weights (target) of the columns of a larger key and of its columns before reaches the goal. The new prefix ends
        // there (none reaches it: at the key's last).
        void CutByColumn(V key, V? target, V? n)
        {
            bool weighed = target is not null;
            var of = weighed ? k : K.U32;
            var arr = weighed ? sums : counts;
            V nothing = weighed ? zero : b.U(0);
            V Value(V c) => weighed ? Stored(c) : b.U(1);
            var aboveMine = b.Local(nothing);
            var tieMine = b.Local(nothing);
            b.For(c0, c1, c =>
            {
                var kc = KeyOf(c);
                var within = Holds(kc, c);
                var v = Value(c);
                aboveMine.V = aboveMine.V + b.Select((kc > key) & within, v, nothing);
                tieMine.V = tieMine.V + b.Select((kc == key) & within, v, nothing);
            });
            var above = GroupReduce(b, arr, tid, aboveMine.V, static (x, y) => x + y, ReduceThreads);
            arr[tid] = tieMine.V;
            Prefix(arr, null);
            var before = arr[tid];
            var goal = weighed ? target!.Value - above : n!.Value - above;
            var found = b.Local(none);
            b.If(before + tieMine.V >= goal, () =>
            {
                var run = b.Local(before);
                b.For(c0, c1, c =>
                {
                    var kc = KeyOf(c);
                    var tie = (kc == key) & Holds(kc, c);
                    run.V = run.V + b.Select(tie, Value(c), nothing);
                    found.V = b.Select(tie & (run.V >= goal) & (found.V == none), c, found.V);
                });
            });
            var least = GroupReduce(b, counts, tid, found.V, b.Min, ReduceThreads);
            bKey.V = key;
            bCol.V = least;
        }

        // The row's keys' range (where a rule ranks).
        var kmin = b.Local(b.CBits(kk, ones));
        var kmax = b.Local(b.CBits(kk, 0));
        if (anyRule) b.If(ranked, () =>
        {
            var lo = b.Local(b.CBits(kk, ones));
            var hi = b.Local(b.CBits(kk, 0));
            b.For(tid, cols, strided, c =>
            {
                var key = KeyOf(c);
                lo.V = b.Min(lo.V, key);
                hi.V = b.Max(hi.V, key);
            });
            kmin.V = GroupReduce(b, keyTree, tid, lo.V, b.Min, ReduceThreads);
            kmax.V = GroupReduce(b, keyTree, tid, hi.V, b.Max, ReduceThreads);
        });

        // Top-k: the largest key at least k columns reach (the k-th largest), then its columns cut by column order. A float or
        // block format's keys (RadixTopK): by the digits of key − kmin from the top, eight bits a pass, the keys that hold
        // the digits so far counted a bin a digit (workgroup atomics), from the top by a scan, the digit whose count reaches
        // what is left of k taken; the digits above the span's highest bit are 0 without a pass. An integer format's: by
        // narrowing the keys' range sixteen ways a pass (its values few: their atomics would wait on one another's bins).
        if (rules.TopK && RadixTopK(f)) b.If(hasTopK, () =>
        {
            int bits = kk == K.U64 ? 64 : 32;
            V Wide(V u32) => kk == K.U32 ? u32 : b.Convert(u32, kk);
            V Narrow32(V key) => kk == K.U32 ? key : b.Convert(key, K.U32);
            var picked = b.Shared(K.U32, 2);
            var prefix = b.Local(b.CBits(kk, 0));
            var need = b.Local(topK);
            var span = kmax.V - kmin.V;
            var last = b.U(ReduceThreads - 1);
            for (int s = bits - 8; s >= 0; s -= 8)
            {
                int shift = s;
                ulong above = shift + 8 >= bits ? 0UL : (ones << (shift + 8)) & ones;    // the digits taken before this one
                b.If((span >> shift) != 0.0, () =>
                {
                    counts[tid] = b.U(0);
                    b.Barrier();
                    b.For(tid, cols, strided, c =>
                    {
                        var off = KeyOf(c) - kmin.V;
                        b.If((off & above) == prefix.V, () => counts.AtomicAdd(Narrow32((off >> shift) & 255UL), b.U(1)));
                    });
                    b.Barrier();
                    // counts[d] = the keys of digit d or above (a scan from the top, eight steps).
                    for (int o = 1; o < ReduceThreads; o *= 2)
                    {
                        var v = counts[tid] + b.Select(tid + (double)o <= (double)(ReduceThreads - 1), counts[b.Min(tid + (double)o, last)], b.U(0));
                        b.Barrier();
                        counts[tid] = v;
                        b.Barrier();
                    }

                    var mine = counts[tid];
                    var next = b.Select(tid < (double)(ReduceThreads - 1), counts[b.Min(tid + 1.0, last)], b.U(0));
                    b.If((mine >= need.V) & (next < need.V), () => { picked[0] = tid; picked[1] = next; });
                    b.Barrier();
                    need.V = need.V - picked[1];
                    prefix.V = prefix.V | (Wide(picked[0]) << shift);
                    b.Barrier();
                });
            }

            CutByColumn(kmin.V + prefix.V, null, topK);
        });
        else if (rules.TopK) b.If(hasTopK, () =>
        {
            var lo = b.Local(kmin.V);
            var hi = b.Local(kmax.V);
            b.While(() => lo.V < hi.V, () =>
            {
                var at = Step(lo.V, hi.V);
                var count = new Var[Ways];
                for (int j = 1; j < Ways; j++) count[j] = b.Local(b.U(0));
                b.For(tid, cols, strided, c =>
                {
                    var key = KeyOf(c);
                    for (int j = 1; j < Ways; j++) count[j].V = count[j].V + b.Select(key >= at[j], b.U(1), b.U(0));
                });
                var holds = new V[Ways];
                for (int j = 1; j < Ways; j++) holds[j] = GroupReduce(b, counts, tid, count[j].V, static (x, y) => x + y, ReduceThreads) >= topK;
                Narrow(lo, hi, at, holds);
            });
            CutByColumn(lo.V, null, topK);
        });

        // The nucleus within the prefix: the shortest prefix of the order whose weights reach p of the prefix's.
        if (gathered) b.If(gather, () =>
        {
            // Top-k's k columns gathered (an invocation's chunk's columns at the place its count's prefix gives), each ranked
            // by the gathered columns before it in the order, then summed in rank order by the first invocation.
            var mineCount = b.Local(b.U(0));
            b.For(c0, c1, c => mineCount.V = mineCount.V + b.Select(Holds(KeyOf(c), c), b.U(1), b.U(0)));
            counts[tid] = mineCount.V;
            Prefix(counts, null);
            var place = b.Local(counts[tid]);
            b.For(c0, c1, c =>
            {
                var key = KeyOf(c);
                b.If(Holds(key, c), () =>
                {
                    gKey[place.V] = key;
                    gWeight[place.V] = Stored(c);
                    gCol[place.V] = c;
                    place.V = place.V + 1.0;
                });
            });
            b.Barrier();
            b.If(tid < topK, () =>
            {
                var own = gKey[tid];
                var ownCol = gCol[tid];
                var rank = b.Local(b.U(0));
                b.For(b.U(0), topK, i =>
                {
                    var ki = gKey[i];
                    rank.V = rank.V + b.Select((ki > own) | ((ki == own) & (gCol[i] < ownCol)), b.U(1), b.U(0));
                });
                rKey[rank.V] = own;
                rWeight[rank.V] = gWeight[tid];
                rCol[rank.V] = ownCol;
            });
            b.Barrier();
            b.If(tid == 0, () =>
            {
                var mass = b.Local(zero);
                b.For(b.U(0), topK, j => mass.V = mass.V + rWeight[j]);
                var target = p * mass.V;
                var run = b.Local(zero);
                var end = b.Local(none);
                b.For(b.U(0), topK, j =>
                {
                    run.V = run.V + rWeight[j];
                    end.V = b.Select((run.V >= target) & (end.V == none), j, end.V);
                });
                var at = b.Select(end.V == none, topK - 1.0, end.V);
                bound[b.U(0)] = rKey[at];
                boundCol[b.U(0)] = rCol[at];
            });
            b.Barrier();
            bKey.V = bound[b.U(0)];
            bCol.V = boundCol[b.U(0)];
            b.Barrier();
        });

        if (rules.Nucleus && !gathered) b.If(hasNucleus & !gather, () =>
        {
            // The weights within the prefix, their sum's p, then the largest key at which the weights at or above it within
            // the prefix reach it, and its columns cut by column order where the running sum reaches it.
            var part = b.Local(zero);
            b.For(tid, cols, strided, c => part.V = part.V + b.Select(Holds(KeyOf(c), c), Stored(c), zero));
            var target = p * GroupReduce(b, sums, tid, part.V, static (x, y) => x + y, ReduceThreads);
            var lo = b.Local(kmin.V);
            var hi = b.Local(kmax.V);
            b.While(() => lo.V < hi.V, () =>
            {
                var at = Step(lo.V, hi.V);
                var sum = new Var[Ways];
                for (int j = 1; j < Ways; j++) sum[j] = b.Local(zero);
                b.For(tid, cols, strided, c =>
                {
                    var key = KeyOf(c);
                    var within = Holds(key, c);
                    var w = Stored(c);
                    for (int j = 1; j < Ways; j++) sum[j].V = sum[j].V + b.Select((key >= at[j]) & within, w, zero);
                });
                var holds = new V[Ways];
                for (int j = 1; j < Ways; j++) holds[j] = GroupReduce(b, sums, tid, sum[j].V, static (x, y) => x + y, ReduceThreads) >= target;
                Narrow(lo, hi, at, holds);
            });
            CutByColumn(lo.V, target, null);
        });

        // A column's kept weight: its weight where it is within the prefix and at least q (0 without min-p: every weight).
        V KeptWeight(V c)
        {
            var e = Stored(c);
            if (!anyRule) return b.Select(e >= q, e, zero);
            var within = b.Local(b.Bool(true));
            b.If(ranked, () => within.V = Holds(KeyOf(c), c));
            return b.Select(within.V & (e >= q), e, zero);
        }

        // The kept weights' sums over this invocation's chunk, their exclusive prefix and the total; the chunk's first and
        // last columns of positive weight (+ 1; 0: none) on the way.
        var mine = b.Local(zero);
        var firstMine = b.Local(cols);
        var lastMine = b.Local(b.U(0));
        b.For(c0, c1, c =>
        {
            var w = KeptWeight(c);
            mine.V = mine.V + w;
            firstMine.V = b.Select((w > 0.0) & (firstMine.V == cols), c, firstMine.V);
            lastMine.V = b.Select(w > 0.0, c + 1.0, lastMine.V);                   // the chunk's columns ascend
        });
        sums[tid] = mine.V;
        Prefix(sums, total);
        var sum2 = total[b.U(0)];
        // The draw: Draws.Uniform(key, step, row) = Bits(Stream(key, step), row) · 2^−32 in the lane, times the total.
        var u = b.Convert(Draw(Draw(ka, kb, b.Load(K.U32, pstep)), kb, row), k) * (1.0 / 4294967296);
        var target2 = u * sum2;
        var before = sums[tid];
        var found2 = b.Local(cols);
        // A chunk whose running sum starts past the target passes it at its first positive weight (before + w rounds to at
        // least before): that column, without the scan. Every other chunk scans as the running sum goes.
        b.If(before > target2, () => found2.V = firstMine.V, () =>
        {
            var run2 = b.Local(before);
            b.For(c0, c1, c =>
            {
                var w = KeptWeight(c);
                run2.V = run2.V + w;
                found2.V = b.Select((w > 0.0) & (run2.V > target2) & (found2.V == cols), c, found2.V);
            });
        });
        var least2 = GroupReduce(b, counts, tid, found2.V, b.Min, ReduceThreads);
        var lastKept = GroupReduce(b, counts, tid, lastMine.V, b.Max, ReduceThreads);  // the last kept column (+ 1; 0: none)
        b.If(tid == 0, () =>
        {
            var pick = b.Select(least2 >= cols, lastKept - 1.0, least2);              // the last kept column; −1 when none
            pick = b.Select(sum2 > 0.0, pick, none);                                  // a sum not positive (NaN): no column
            b.Store(b.At(picks, row, 4), pick);
            b.If(prob != b.UL(0), () =>
            {
                var valid = pick != none;
                var w = Stored(b.Select(valid, pick, b.U(0)));
                b.Store(b.At(prob, row, lane.Size), b.Select(valid, w / sum2, b.C(k, double.NaN)));
            });
        });
        return b.Build();
    }

    /// <summary>keys[i] (u64) = the order key (OrderKey) of x's value i in f's natural lane, i below n: the key proven on
    /// the device against Routing.Order (Checks.Sampling.cs). Slots: x, keys, n; an invocation a value.</summary>
    public static SpvKernel OrderKeys(Vulkan vk, Fmt f)
    {
        var b = new Spv($"order_keys_{f.Name}", Threads, vk.Controls);
        V px = b.Arg(0), pk = b.Arg(1), n = b.Arg(2, K.U32);
        var gid = b.GlobalX;
        b.If(gid < n, () =>
        {
            var key = OrderKey(b, Codec.UnpackAt(b, f, px, gid, f.Natural));
            b.Store(b.At(pk, gid, 8), key.Kind == K.U64 ? key : b.Convert(key, K.U64));
        });
        return b.Build();
    }
}
