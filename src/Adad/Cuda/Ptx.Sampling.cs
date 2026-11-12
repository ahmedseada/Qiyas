// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on CUDA (Abstraction/Sampling.cs has the rule), a CTA a row in two kernels:
//   SampleWeights  the row's scores read where they are stored, in the type's float lane: m the first-ranked in Adad's
//                  order (First: a NaN where the row holds one, else the largest) by the block's reduction, then the
//                  weights exp((x − m) / t) in lane storage (NaN throughout a row whose m is not finite).
//   SampleDraw     each rule's prefix of the row's order, by the scores' order keys (Routing.Order's integer form: a
//                  stored value as an unsigned integer that orders as Adad's order does, made where it is read): top-k's
//                  k-th column by narrowing the keys' range sixteen ways a pass (a thread its columns tid, tid + 256, …;
//                  the block's reductions of the counts), its key's columns cut by column order (a thread a chunk of
//                  consecutive columns, the chunks' counts' prefix); the nucleus over top-k's columns gathered into
//                  shared memory where k fits a block's threads (each ranks itself among them, the first thread sums
//                  them in order), else by narrowing the keys' range sixteen ways with the weights' sums and its key's
//                  columns cut where the running sum reaches the target; min-p a weight at least q. Then the kept
//                  weights' chunk sums and their prefix, the seed's uniform at (step, row) with the step read where it
//                  is, and the column at which the running sum passes it (the last kept column of positive weight when
//                  rounding leaves none; −1 and a NaN probability for a row whose sum is not positive).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    // SampleDraw's shared memory: the reductions' slots, a slot a thread (chunk counts and sums, their prefix), the total,
    // the boundary (key, column), top-k's gathered columns (key, weight, column) and the same in the order's ranks.
    private const int DrawSlots = ReduceBytes, DrawTotal = DrawSlots + 256 * 8, DrawBound = DrawTotal + 16, DrawGatherKey = DrawBound + 16,
        DrawGatherWeight = DrawGatherKey + 256 * 8, DrawGatherCol = DrawGatherWeight + 256 * 8, DrawRankKey = DrawGatherCol + 256 * 4,
        DrawRankWeight = DrawRankKey + 256 * 8, DrawRankCol = DrawRankWeight + 256 * 8;

    /// <summary>The shared bytes of SampleDraw.</summary>
    public const int SampleDrawShared = DrawRankCol + 256 * 4;

    /// <summary>The most columns top-k may keep for the nucleus to take them gathered (a thread a column).</summary>
    public const int SampleGathered = 256;

    /// <summary>w (lane storage, rows × cols) = each row's weights. Params: x (f; row r at r·stride + off), w, cols, stride,
    /// off, temperature (lane); a CTA a row.</summary>
    public static string SampleWeights(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"sample_weights_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string px = Param64(b, "p_x"), pw = Param64(b, "p_w"), cols = Param32(b, "p_cols"), stride = Param32(b, "p_stride"), off = Param32(b, "p_off");
        string temp = lane.NewReg(b), start = b.R(), wbase = b.R();
        b.L($"ld.param.{lane.Op} {temp}, [p_temp];");
        b.L($"mad.lo.u32 {start}, {row}, {stride}, {off};");
        b.L($"mul.lo.u32 {wbase}, {row}, {cols};");
        string Value(string c)
        {
            string idx = b.R();
            b.L($"add.u32 {idx}, {start}, {c};");
            return Codec.UnpackAt(b, f, px, idx, lane);
        }

        // The first-ranked value in Adad's order (First: a NaN where the row holds one, else the largest).
        string best = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {best}, {lane.Literal(double.NegativeInfinity)};");
        {
            string c = b.R(), pEnd = b.P();
            b.L($"mov.u32 {c}, {tid};");
            b.Label("TOP");
            b.L($"setp.ge.u32 {pEnd}, {c}, {cols};");
            b.L($"@{pEnd} bra TOP_END;");
            First(b, lane, best, best, Value(c));
            b.L($"add.u32 {c}, {c}, 256;");
            b.L("bra TOP;");
            b.Label("TOP_END");
        }

        string m = BlockReduce(b, lane, tid, best, "first", 256), pFinite = b.P();
        b.L($"testp.finite.{lane.Op} {pFinite}, {m};");
        {
            string c = b.R(), pEnd = b.P(), d = lane.NewReg(b), e = lane.NewReg(b), a = b.RD(), idx = b.R();
            b.L($"mov.u32 {c}, {tid};");
            b.Label("WEIGH");
            b.L($"setp.ge.u32 {pEnd}, {c}, {cols};");
            b.L($"@{pEnd} bra DONE;");
            b.L($"sub.rn.{lane.Op} {d}, {Value(c)}, {m};");
            b.L($"div.rn.{lane.Op} {d}, {d}, {temp};");
            string ex = Codec.Exp(b, lane, d);
            b.L($"selp.{lane.Op} {e}, {ex}, {lane.Literal(double.NaN)}, {pFinite};");     // x − m is NaN at m's own column
            b.L($"add.u32 {idx}, {wbase}, {c};");
            b.L($"mul.wide.u32 {a}, {idx}, {lane.Size};");
            b.L($"add.u64 {a}, {pw}, {a};");
            b.L($"st.global.{lane.Mem} [{a}], {e};");
            b.L($"add.u32 {c}, {c}, 256;");
            b.L("bra WEIGH;");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_w, .param .u32 p_cols, .param .u32 p_stride, .param .u32 p_off, .param .{lane.Op} p_temp", ReduceBytes);
    }

    /// <summary>picks[row] and (prob not 0) prob[row] (lane storage) from the row's scores x (f, where they are stored) and
    /// weights w (lane storage, rows × cols). Params: x, w, picks, prob, step (the device's, read where it is), cols, stride,
    /// off, k (cols: no top-k), the seed's key A, p (1: no nucleus), q (0: no min-p) (lane), the key's B; a CTA a row.</summary>
    public static string SampleDraw(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"sample_draw_{f.Name}_{lane.Name}";
        var natural = f.Natural;
        bool wide = natural.Wide;
        string kt = wide ? "u64" : "u32", kbits = wide ? "b64" : "b32";
        var keyLane = wide ? Lane.U64 : Lane.U32;
        var b = new PtxBuilder(gpu);
        string Key() => wide ? b.RD() : b.R();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string px = Param64(b, "p_x"), pw = Param64(b, "p_w"), picks = Param64(b, "p_picks"), prob = Param64(b, "p_prob"), pstep = Param64(b, "p_step");
        string cols = Param32(b, "p_cols"), stride = Param32(b, "p_stride"), off = Param32(b, "p_off"), k = Param32(b, "p_k"), ka = Param32(b, "p_ka"), kb = Param32(b, "p_kb");
        string p = lane.NewReg(b), q = lane.NewReg(b), wbase = b.R(), start = b.R(), smem = b.R(), mine = b.R();
        b.L($"ld.param.{lane.Op} {p}, [p_p];");
        b.L($"ld.param.{lane.Op} {q}, [p_q];");
        b.L($"mul.lo.u32 {wbase}, {row}, {cols};");
        b.L($"mad.lo.u32 {start}, {row}, {stride}, {off};");
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mad.lo.u32 {mine}, {tid}, 8, {smem};");
        string pTopK = b.P(), pNucleus = b.P(), pGather = b.P(), pRanked = b.P();
        b.L($"setp.lt.u32 {pTopK}, {k}, {cols};");
        b.L($"setp.lt.{lane.Op} {pNucleus}, {p}, {lane.Literal(1)};");
        b.L($"setp.le.u32 {pGather}, {k}, {SampleGathered};");
        b.L($"and.pred {pGather}, {pGather}, {pTopK};");
        b.L($"or.pred {pRanked}, {pTopK}, {pNucleus};");

        // Column c's order key (OrderKey), read where it is stored in the natural lane.
        string KeyOf(string c)
        {
            string idx = b.R();
            b.L($"add.u32 {idx}, {start}, {c};");
            return OrderKey(b, natural, Codec.UnpackAt(b, f, px, idx, natural));
        }

        string Weight(string c)
        {
            string idx = b.R(), a = b.RD(), e = lane.NewReg(b);
            b.L($"add.u32 {idx}, {wbase}, {c};");
            b.L($"mul.wide.u32 {a}, {idx}, {lane.Size};");
            b.L($"add.u64 {a}, {pw}, {a};");
            b.L($"ld.global.{lane.Mem} {e}, [{a}];");
            return e;
        }

        // A pass of the row: this thread's columns tid, tid + 256, … (strided) or its chunk of consecutive ones (chunked).
        string chunk = b.R(), c0 = b.R(), c1 = b.R();
        b.L($"add.u32 {chunk}, {cols}, 255;");
        b.L($"shr.u32 {chunk}, {chunk}, 8;");
        b.L($"mul.lo.u32 {c0}, {tid}, {chunk};");
        b.L($"min.u32 {c0}, {c0}, {cols};");
        b.L($"add.u32 {c1}, {c0}, {chunk};");
        b.L($"min.u32 {c1}, {c1}, {cols};");
        void Pass(string label, bool chunked, Action<string> body)
        {
            string c = b.R(), pEnd = b.P();
            b.L($"mov.u32 {c}, {(chunked ? c0 : tid)};");
            b.Label(label);
            b.L($"setp.ge.u32 {pEnd}, {c}, {(chunked ? c1 : cols)};");
            b.L($"@{pEnd} bra {label}_END;");
            body(c);
            b.L($"add.u32 {c}, {c}, {(chunked ? "1" : "256")};");
            b.L($"bra {label};");
            b.Label($"{label}_END");
        }

        // The slots' exclusive prefix by the first thread (each thread's own slot then holds what comes before its chunk),
        // the total after them.
        void Prefix(string tag, Lane of)
        {
            b.L("bar.sync 0;");
            string p0 = b.P(), s = b.R(), acc = of.NewReg(b), addr = b.R(), sv = of.NewReg(b), pS = b.P();
            b.L($"setp.ne.u32 {p0}, {tid}, 0;");
            b.L($"@{p0} bra {tag}_SKIP;");
            b.L($"mov.{of.Reg} {acc}, {of.Zero};");
            b.L($"mov.u32 {s}, 0;");
            b.Label(tag);
            b.L($"setp.ge.u32 {pS}, {s}, 256;");
            b.L($"@{pS} bra {tag}_END;");
            b.L($"mad.lo.u32 {addr}, {s}, 8, {smem};");
            b.L($"ld.shared.{of.Mem} {sv}, [{addr}+{DrawSlots}];");
            b.L($"st.shared.{of.Mem} [{addr}+{DrawSlots}], {acc};");
            b.L(of.Add(acc, acc, sv));
            b.L($"add.u32 {s}, {s}, 1;");
            b.L($"bra {tag};");
            b.Label($"{tag}_END");
            b.L($"st.shared.{of.Mem} [{smem}+{DrawTotal}], {acc};");
            b.Label($"{tag}_SKIP");
            b.L("bar.sync 0;");
        }

        // The prefix's last column (key, column): a column is within it when its key is larger, or equal at or before it.
        string bKey = Key(), bCol = b.R();
        b.L($"mov.{kbits} {bKey}, 0;");
        b.L($"mov.u32 {bCol}, 0xFFFFFFFF;");                                      // every column until a rule cuts
        string Holds(string key, string c)
        {
            string pa = b.P(), pe = b.P(), pc = b.P();
            b.L($"setp.gt.{kt} {pa}, {key}, {bKey};");
            b.L($"setp.eq.{kt} {pe}, {key}, {bKey};");
            b.L($"setp.le.u32 {pc}, {c}, {bCol};");
            b.L($"and.pred {pe}, {pe}, {pc};");
            b.L($"or.pred {pa}, {pa}, {pe};");
            return pa;
        }

        // A step's values between lo and hi (each a key): lo + j·step held to hi, step = (hi − lo) / ways + 1, j = 1 .. ways − 1,
        // and the key below each (the rule holds up to some key and fails past it: lo becomes the last at which it holds, hi
        // the key below the first at which it fails).
        (string[] At, string[] Below) Step(string lo, string hi, int ways)
        {
            string span = Key(), step = Key();
            b.L($"sub.{kt} {span}, {hi}, {lo};");
            b.L($"shr.{kt} {step}, {span}, {int.Log2(ways)};");
            b.L($"add.{kt} {step}, {step}, 1;");
            var at = new string[ways];
            var below = new string[ways];
            for (int j = 1; j < ways; j++)
            {
                at[j] = Key(); below[j] = Key();
                b.L($"mul.lo.{kt} {at[j]}, {step}, {j};");
                b.L($"min.{kt} {at[j]}, {at[j]}, {span};");
                b.L($"add.{kt} {at[j]}, {at[j]}, {lo};");
                b.L($"sub.{kt} {below[j]}, {at[j]}, 1;");
            }

            return (at, below);
        }

        void Narrow(string lo, string hi, string[] at, string[] below, string[] holds)
        {
            for (int j = 1; j < at.Length; j++) b.L($"selp.{kt} {lo}, {at[j]}, {lo}, {holds[j]};");
            for (int j = at.Length - 1; j >= 1; j--) b.L($"selp.{kt} {hi}, {hi}, {below[j]}, {holds[j]};");
        }

        // The row's keys' range (where a rule ranks).
        string kmin = Key(), kmax = Key();
        b.L($"@!{pRanked} bra KEYS_DONE;");
        {
            string lo = Key(), hi = Key();
            b.L($"mov.{kbits} {lo}, {(wide ? "0xFFFFFFFFFFFFFFFF" : "0xFFFFFFFF")};");
            b.L($"mov.{kbits} {hi}, 0;");
            Pass("RANGE", false, c =>
            {
                string key = KeyOf(c);
                b.L($"min.{kt} {lo}, {lo}, {key};");
                b.L($"max.{kt} {hi}, {hi}, {key};");
            });
            b.L($"mov.{kbits} {kmin}, {BlockReduce(b, keyLane, tid, lo, "min", 256)};");
            b.L($"mov.{kbits} {kmax}, {BlockReduce(b, keyLane, tid, hi, "max", 256)};");
        }

        b.Label("KEYS_DONE");

        // Top-k: the largest key at least k columns reach (sixteen ways a pass), then its columns cut by column order: the
        // (k − above)-th of them, above the columns of a larger key.
        b.L($"@!{pTopK} bra K_END;");
        {
            const int Ways = 16;
            string lo = Key(), hi = Key(), pDone = b.P();
            b.L($"mov.{kbits} {lo}, {kmin};");
            b.L($"mov.{kbits} {hi}, {kmax};");
            b.Label("K_LOOP");
            b.L($"setp.ge.{kt} {pDone}, {lo}, {hi};");
            b.L($"@{pDone} bra K_DONE;");
            var (at, below) = Step(lo, hi, Ways);
            var count = new string[Ways];
            for (int j = 1; j < Ways; j++) { count[j] = b.R(); b.L($"mov.u32 {count[j]}, 0;"); }
            Pass("K_PASS", false, c =>
            {
                string key = KeyOf(c), pIn = b.P(), one = b.R();
                for (int j = 1; j < Ways; j++)
                {
                    b.L($"setp.ge.{kt} {pIn}, {key}, {at[j]};");
                    b.L($"selp.u32 {one}, 1, 0, {pIn};");
                    b.L($"add.u32 {count[j]}, {count[j]}, {one};");
                }
            });
            var holds = new string[Ways];
            for (int j = 1; j < Ways; j++)
            {
                string reached = BlockReduce(b, Lane.U32, tid, count[j], "add", 256);
                holds[j] = b.P();
                b.L($"setp.ge.u32 {holds[j]}, {reached}, {k};");
            }

            Narrow(lo, hi, at, below, holds);
            b.L("bra K_LOOP;");
            b.Label("K_DONE");
            CutByColumn("KC", lo, null, k);
        }

        b.Label("K_END");

        // The cut of key `key`'s columns (within the prefix so far) by column order: with no weights, the n-th of them, n the
        // count less those of a larger key; with weights, the first at which the running sum of the weights of a larger key
        // and of its columns before reaches the target. The new prefix ends there (no column reaches it: at its last).
        void CutByColumn(string tag, string key, string? target, string? n)
        {
            var of = target is null ? Lane.U32 : lane;
            string aboveMine = of.NewReg(b), tieMine = of.NewReg(b);
            b.L($"mov.{of.Reg} {aboveMine}, {of.Zero};");
            b.L($"mov.{of.Reg} {tieMine}, {of.Zero};");
            Pass($"{tag}_PASS", true, c =>
            {
                string kk = KeyOf(c), within = Holds(kk, c), pGreater = b.P(), pEqual = b.P(), v = of.NewReg(b), add = of.NewReg(b);
                if (target is null) b.L($"mov.u32 {v}, 1;");
                else b.L($"mov.{lane.Reg} {v}, {Weight(c)};");
                b.L($"setp.gt.{kt} {pGreater}, {kk}, {key};");
                b.L($"setp.eq.{kt} {pEqual}, {kk}, {key};");
                b.L($"and.pred {pGreater}, {pGreater}, {within};");
                b.L($"and.pred {pEqual}, {pEqual}, {within};");
                b.L($"selp.{of.Op} {add}, {v}, {of.Zero}, {pGreater};");
                b.L(of.Add(aboveMine, aboveMine, add));
                b.L($"selp.{of.Op} {add}, {v}, {of.Zero}, {pEqual};");
                b.L(of.Add(tieMine, tieMine, add));
            });
            string above = BlockReduce(b, of, tid, aboveMine, "add", 256);
            b.L($"st.shared.{of.Mem} [{mine}+{DrawSlots}], {tieMine};");
            Prefix($"{tag}_SCAN", of);
            // The goal within the key's columns (n − above columns, or target − above of weight); a chunk whose running sum
            // reaches it walks to its first column that does, the least over the block the cut (none: the key's last).
            string before = of.NewReg(b), goal = of.NewReg(b), pMineHas = b.P(), reach = of.NewReg(b), found = b.R();
            b.L($"ld.shared.{of.Mem} {before}, [{mine}+{DrawSlots}];");
            if (target is null) b.L($"sub.u32 {goal}, {n}, {above};");
            else b.L($"sub.rn.{lane.Op} {goal}, {target}, {above};");
            b.L(of.Add(reach, before, tieMine));
            b.L($"setp.ge.{of.Op} {pMineHas}, {reach}, {goal};");
            b.L($"mov.u32 {found}, 0xFFFFFFFF;");
            b.L($"@!{pMineHas} bra {tag}_FOUND;");
            {
                string run = of.NewReg(b);
                b.L($"mov.{of.Reg} {run}, {before};");
                Pass($"{tag}_WALK", true, c =>
                {
                    string kk = KeyOf(c), within = Holds(kk, c), pEqual = b.P(), v = of.NewReg(b), pReached = b.P();
                    b.L($"setp.eq.{kt} {pEqual}, {kk}, {key};");
                    b.L($"and.pred {pEqual}, {pEqual}, {within};");
                    b.L($"@!{pEqual} bra {tag}_NEXT;");
                    if (target is null) b.L($"mov.u32 {v}, 1;");
                    else b.L($"mov.{lane.Reg} {v}, {Weight(c)};");
                    b.L(of.Add(run, run, v));
                    b.L($"setp.ge.{of.Op} {pReached}, {run}, {goal};");
                    b.L($"@{pReached} mov.u32 {found}, {c};");
                    b.L($"@{pReached} bra {tag}_WALK_END;");
                    b.Label($"{tag}_NEXT");
                });
            }

            b.Label($"{tag}_FOUND");
            b.L($"mov.u32 {bCol}, {BlockReduce(b, Lane.U32, tid, found, "min", 256)};");
            b.L($"mov.{kbits} {bKey}, {key};");
        }

        // The nucleus within the prefix: the shortest prefix of the order whose weights reach p of the prefix's.
        b.L($"@!{pNucleus} bra P_END;");
        b.L($"@!{pGather} bra P_SEARCH;");
        {
            // Top-k's k columns gathered (a thread's chunk's columns at the place its count's prefix gives), each ranked by
            // the gathered columns before it in the order, then summed in rank order by the first thread.
            string mineCount = b.R();
            b.L($"mov.u32 {mineCount}, 0;");
            Pass("G_COUNT", true, c =>
            {
                string one = b.R();
                b.L($"selp.u32 {one}, 1, 0, {Holds(KeyOf(c), c)};");
                b.L($"add.u32 {mineCount}, {mineCount}, {one};");
            });
            b.L($"st.shared.u32 [{mine}+{DrawSlots}], {mineCount};");
            Prefix("G_SCAN", Lane.U32);
            string place = b.R();
            b.L($"ld.shared.u32 {place}, [{mine}+{DrawSlots}];");
            Pass("G_PUT", true, c =>
            {
                string kk = KeyOf(c), pIn = Holds(kk, c), one = b.R(), ak = b.R(), aw = b.R(), ac = b.R();
                b.L($"@!{pIn} bra G_PUT_SKIP;");
                b.L($"mad.lo.u32 {ak}, {place}, 8, {smem};");
                b.L($"mad.lo.u32 {ac}, {place}, 4, {smem};");
                b.L($"st.shared.{kbits} [{ak}+{DrawGatherKey}], {kk};");
                b.L($"st.shared.{lane.Mem} [{ak}+{DrawGatherWeight}], {Weight(c)};");
                b.L($"st.shared.u32 [{ac}+{DrawGatherCol}], {c};");
                b.L($"add.u32 {place}, {place}, 1;");
                b.Label("G_PUT_SKIP");
                _ = one; _ = aw;
            });
            b.L("bar.sync 0;");
            // Thread i < k: its rank among the gathered.
            string pMine = b.P(), ownK = Key(), ownC = b.R(), rank = b.R(), i = b.R(), pEnd = b.P(), ai = b.R(), ki = Key(), ci = b.R(), pB = b.P(), pE = b.P(), pC = b.P(), one2 = b.R();
            b.L($"setp.lt.u32 {pMine}, {tid}, {k};");
            b.L($"@!{pMine} bra G_RANKED;");
            b.L($"mad.lo.u32 {ai}, {tid}, 8, {smem};");
            b.L($"ld.shared.{kbits} {ownK}, [{ai}+{DrawGatherKey}];");
            b.L($"mad.lo.u32 {ai}, {tid}, 4, {smem};");
            b.L($"ld.shared.u32 {ownC}, [{ai}+{DrawGatherCol}];");
            b.L($"mov.u32 {rank}, 0;");
            b.L($"mov.u32 {i}, 0;");
            b.Label("G_RANK");
            b.L($"setp.ge.u32 {pEnd}, {i}, {k};");
            b.L($"@{pEnd} bra G_RANK_END;");
            b.L($"mad.lo.u32 {ai}, {i}, 8, {smem};");
            b.L($"ld.shared.{kbits} {ki}, [{ai}+{DrawGatherKey}];");
            b.L($"mad.lo.u32 {ai}, {i}, 4, {smem};");
            b.L($"ld.shared.u32 {ci}, [{ai}+{DrawGatherCol}];");
            b.L($"setp.gt.{kt} {pB}, {ki}, {ownK};");
            b.L($"setp.eq.{kt} {pE}, {ki}, {ownK};");
            b.L($"setp.lt.u32 {pC}, {ci}, {ownC};");
            b.L($"and.pred {pE}, {pE}, {pC};");
            b.L($"or.pred {pB}, {pB}, {pE};");
            b.L($"selp.u32 {one2}, 1, 0, {pB};");
            b.L($"add.u32 {rank}, {rank}, {one2};");
            b.L($"add.u32 {i}, {i}, 1;");
            b.L("bra G_RANK;");
            b.Label("G_RANK_END");
            {
                string ar = b.R(), ac = b.R(), aw = b.R(), w = lane.NewReg(b);
                b.L($"mad.lo.u32 {aw}, {tid}, 8, {smem};");
                b.L($"ld.shared.{lane.Mem} {w}, [{aw}+{DrawGatherWeight}];");
                b.L($"mad.lo.u32 {ar}, {rank}, 8, {smem};");
                b.L($"mad.lo.u32 {ac}, {rank}, 4, {smem};");
                b.L($"st.shared.{kbits} [{ar}+{DrawRankKey}], {ownK};");
                b.L($"st.shared.{lane.Mem} [{ar}+{DrawRankWeight}], {w};");
                b.L($"st.shared.u32 [{ac}+{DrawRankCol}], {ownC};");
            }

            b.Label("G_RANKED");
            b.L("bar.sync 0;");
            // The first thread: the k weights' sum in rank order, then the first rank at which the running sum reaches p of it.
            string pFirst = b.P();
            b.L($"setp.ne.u32 {pFirst}, {tid}, 0;");
            b.L($"@{pFirst} bra G_SUMMED;");
            {
                string mass = lane.NewReg(b), run = lane.NewReg(b), target = lane.NewReg(b), j = b.R(), aj = b.R(), w = lane.NewReg(b), pJ = b.P(), pR = b.P(), end = b.R(), last = b.R();
                b.L($"mov.{lane.Reg} {mass}, {lane.Zero};");
                b.L($"mov.u32 {j}, 0;");
                b.Label("G_MASS");
                b.L($"setp.ge.u32 {pJ}, {j}, {k};");
                b.L($"@{pJ} bra G_MASS_END;");
                b.L($"mad.lo.u32 {aj}, {j}, 8, {smem};");
                b.L($"ld.shared.{lane.Mem} {w}, [{aj}+{DrawRankWeight}];");
                b.L(lane.Add(mass, mass, w));
                b.L($"add.u32 {j}, {j}, 1;");
                b.L("bra G_MASS;");
                b.Label("G_MASS_END");
                b.L(lane.Mul(target, p, mass));
                b.L($"sub.u32 {end}, {k}, 1;");
                b.L($"mov.{lane.Reg} {run}, {lane.Zero};");
                b.L($"mov.u32 {j}, 0;");
                b.Label("G_RUN");
                b.L($"setp.ge.u32 {pJ}, {j}, {k};");
                b.L($"@{pJ} bra G_RUN_END;");
                b.L($"mad.lo.u32 {aj}, {j}, 8, {smem};");
                b.L($"ld.shared.{lane.Mem} {w}, [{aj}+{DrawRankWeight}];");
                b.L(lane.Add(run, run, w));
                b.L($"setp.ge.{lane.Op} {pR}, {run}, {target};");
                b.L($"@{pR} mov.u32 {end}, {j};");
                b.L($"@{pR} bra G_RUN_END;");
                b.L($"add.u32 {j}, {j}, 1;");
                b.L("bra G_RUN;");
                b.Label("G_RUN_END");
                string ak = b.R(), ac = b.R(), ek = Key(), ec = b.R();
                b.L($"mad.lo.u32 {ak}, {end}, 8, {smem};");
                b.L($"mad.lo.u32 {ac}, {end}, 4, {smem};");
                b.L($"ld.shared.{kbits} {ek}, [{ak}+{DrawRankKey}];");
                b.L($"ld.shared.u32 {ec}, [{ac}+{DrawRankCol}];");
                b.L($"st.shared.{kbits} [{smem}+{DrawBound}], {ek};");
                b.L($"st.shared.u32 [{smem}+{DrawBound + 8}], {ec};");
                _ = last;
            }

            b.Label("G_SUMMED");
            b.L("bar.sync 0;");
            b.L($"ld.shared.{kbits} {bKey}, [{smem}+{DrawBound}];");
            b.L($"ld.shared.u32 {bCol}, [{smem}+{DrawBound + 8}];");
            b.L("bar.sync 0;");
            b.L("bra P_END;");
        }

        b.Label("P_SEARCH");
        {
            // The weights within the prefix, their sum's p, then the largest key at which the weights at or above it within the
            // prefix reach it, and its columns cut by column order where the running sum reaches it.
            string part = lane.NewReg(b), target = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {part}, {lane.Zero};");
            Pass("M_PASS", false, c =>
            {
                string kept = lane.NewReg(b);
                b.L($"selp.{lane.Op} {kept}, {Weight(c)}, {lane.Zero}, {Holds(KeyOf(c), c)};");
                b.L(lane.Add(part, part, kept));
            });
            string mass = BlockReduce(b, lane, tid, part, "add", 256);
            b.L(lane.Mul(target, p, mass));
            const int ways = 16;
            string lo = Key(), hi = Key(), pDone = b.P();
            b.L($"mov.{kbits} {lo}, {kmin};");
            b.L($"mov.{kbits} {hi}, {kmax};");
            b.Label("P_LOOP");
            b.L($"setp.ge.{kt} {pDone}, {lo}, {hi};");
            b.L($"@{pDone} bra P_DONE;");
            var (at, below) = Step(lo, hi, ways);
            var sum = new string[ways];
            for (int j = 1; j < ways; j++) { sum[j] = lane.NewReg(b); b.L($"mov.{lane.Reg} {sum[j]}, {lane.Zero};"); }
            Pass("P_PASS", false, c =>
            {
                string kk = KeyOf(c), within = Holds(kk, c), w = Weight(c), pIn = b.P(), kept = lane.NewReg(b);
                for (int j = 1; j < ways; j++)
                {
                    b.L($"setp.ge.{kt} {pIn}, {kk}, {at[j]};");
                    b.L($"and.pred {pIn}, {pIn}, {within};");
                    b.L($"selp.{lane.Op} {kept}, {w}, {lane.Zero}, {pIn};");
                    b.L(lane.Add(sum[j], sum[j], kept));
                }
            });
            var holds = new string[ways];
            for (int j = 1; j < ways; j++)
            {
                string held = BlockReduce(b, lane, tid, sum[j], "add", 256);
                holds[j] = b.P();
                b.L($"setp.ge.{lane.Op} {holds[j]}, {held}, {target};");
            }

            Narrow(lo, hi, at, below, holds);
            b.L("bra P_LOOP;");
            b.Label("P_DONE");
            CutByColumn("PC", lo, target, null);
        }

        b.Label("P_END");

        // A column's kept weight: its weight where it is within the prefix (and, with min-p, at least q), else 0.
        string KeptWeight(string c)
        {
            string e = Weight(c), w = lane.NewReg(b), pIn = b.P(), pq = b.P(), tag = $"KW{pIn.TrimStart('%')}";
            b.L($"mov.pred {pIn}, 1;");
            b.L($"@!{pRanked} bra {tag};");
            b.L($"mov.pred {pIn}, {Holds(KeyOf(c), c)};");
            b.Label(tag);
            b.L($"setp.ge.{lane.Op} {pq}, {e}, {q};");                              // q 0 without min-p: every weight
            b.L($"and.pred {pIn}, {pIn}, {pq};");
            b.L($"selp.{lane.Op} {w}, {e}, {lane.Zero}, {pIn};");
            return w;
        }

        // The kept weights' sums over this thread's chunk, their exclusive prefix and the total.
        {
            string sum = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
            Pass("SUM", true, c => b.L(lane.Add(sum, sum, KeptWeight(c))));
            b.L($"st.shared.{lane.Mem} [{mine}+{DrawSlots}], {sum};");
            Prefix("SCAN", lane);
        }

        string prefix = lane.NewReg(b), total = lane.NewReg(b), target2 = lane.NewReg(b);
        b.L($"ld.shared.{lane.Mem} {prefix}, [{mine}+{DrawSlots}];");
        b.L($"ld.shared.{lane.Mem} {total}, [{smem}+{DrawTotal}];");
        // The draw: Draws.Uniform(key, step, row) = Bits(Stream(key, step), row) · 2^−32 in the lane, times the total.
        {
            string step = b.R(), u = lane.NewReg(b);
            b.L($"ld.global.u32 {step}, [{pstep}];");
            string bits = Draw(b, Draw(b, ka, kb, step), kb, row);
            b.L($"cvt.rn.{lane.Op}.u32 {u}, {bits};");
            b.L(lane.Mul(u, u, lane.Literal(1.0 / 4294967296)));
            b.L(lane.Mul(target2, u, total));
        }

        // The first column of this thread's chunk at which the running sum passes the target (cols: none), the least over the
        // block; the chunk's last column of positive kept weight.
        string found = b.R(), lastMine = b.R();
        {
            string run = lane.NewReg(b), pKept = b.P(), pPassed = b.P(), pOpen = b.P(), next = b.R();
            b.L($"mov.{lane.Reg} {run}, {prefix};");
            b.L($"mov.u32 {found}, {cols};");
            b.L($"mov.u32 {lastMine}, 0;");
            Pass("FIND", true, c =>
            {
                string w = KeptWeight(c);
                b.L(lane.Add(run, run, w));
                b.L($"setp.gt.{lane.Op} {pKept}, {w}, {lane.Zero};");
                b.L($"add.u32 {next}, {c}, 1;");
                b.L($"selp.u32 {lastMine}, {next}, {lastMine}, {pKept};");            // the chunk's columns ascend
                b.L($"setp.gt.{lane.Op} {pPassed}, {run}, {target2};");
                b.L($"setp.eq.u32 {pOpen}, {found}, {cols};");
                b.L($"and.pred {pPassed}, {pPassed}, {pKept};");
                b.L($"and.pred {pPassed}, {pPassed}, {pOpen};");
                b.L($"selp.u32 {found}, {c}, {found}, {pPassed};");
            });
        }

        string least = BlockReduce(b, Lane.U32, tid, found, "min", 256);
        string lastKept = BlockReduce(b, Lane.U32, tid, lastMine, "max", 256);       // the last kept column (+ 1; 0: none)
        {
            string pFirst = b.P(), pNone = b.P(), pick = b.R(), fallback = b.R(), pa = b.RD(), pNoProb = b.P(), pValid = b.P(), at = b.R(), value = lane.NewReg(b), qa = b.RD();
            b.L($"setp.ne.u32 {pFirst}, {tid}, 0;");
            b.L($"@{pFirst} bra DONE;");
            b.L($"sub.u32 {fallback}, {lastKept}, 1;");                             // the last kept column; −1 when none
            b.L($"setp.ge.u32 {pNone}, {least}, {cols};");
            b.L($"selp.u32 {pick}, {fallback}, {least}, {pNone};");
            b.L($"setp.gt.{lane.Op} {pValid}, {total}, {lane.Zero};");             // a sum not positive (NaN): no column
            b.L($"selp.u32 {pick}, {pick}, 0xFFFFFFFF, {pValid};");
            b.L($"mul.wide.u32 {pa}, {row}, 4;");
            b.L($"add.u64 {pa}, {picks}, {pa};");
            b.L($"st.global.u32 [{pa}], {pick};");
            b.L($"setp.eq.u64 {pNoProb}, {prob}, 0;");
            b.L($"@{pNoProb} bra DONE;");
            b.L($"setp.ge.s32 {pValid}, {pick}, 0;");
            b.L($"selp.u32 {at}, {pick}, 0, {pValid};");
            string w = Weight(at);
            b.L($"div.rn.{lane.Op} {value}, {w}, {total};");
            b.L($"selp.{lane.Op} {value}, {value}, {lane.Literal(double.NaN)}, {pValid};");
            b.L($"mul.wide.u32 {qa}, {row}, {lane.Size};");
            b.L($"add.u64 {qa}, {prob}, {qa};");
            b.L($"st.global.{lane.Mem} [{qa}], {value};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_w, .param .u64 p_picks, .param .u64 p_prob, .param .u64 p_step, .param .u32 p_cols, .param .u32 p_stride, .param .u32 p_off, .param .u32 p_k, .param .u32 p_ka, .param .{lane.Op} p_p, .param .{lane.Op} p_q, .param .u32 p_kb",
            SampleDrawShared);
    }

    /// <summary>keys[i] (u64) = the order key (OrderKey) of x's value i in f's natural lane, i below n: the key proven on
    /// the device against Routing.Order (Checks.Sampling.cs). Params: x, keys, n; a thread a value.</summary>
    public static string OrderKeys(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"order_keys_{f.Name}";
        var b = new PtxBuilder(gpu);
        string gid = GlobalIndex(b);
        string px = Param64(b, "p_x"), pk = Param64(b, "p_keys"), n = Param32(b, "p_n"), pEnd = b.P();
        b.L($"setp.ge.u32 {pEnd}, {gid}, {n};");
        b.L($"@{pEnd} bra DONE;");
        var natural = f.Natural;
        string key = OrderKey(b, natural, Codec.UnpackAt(b, f, px, gid, natural)), wide = b.RD(), a = b.RD();
        b.L(natural.Wide ? $"mov.b64 {wide}, {key};" : $"cvt.u64.u32 {wide}, {key};");
        b.L($"mul.wide.u32 {a}, {gid}, 8;");
        b.L($"add.u64 {a}, {pk}, {a};");
        b.L($"st.global.u64 [{a}], {wide};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_keys, .param .u32 p_n");
    }
}
