// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's sampling on CUDA (Sampling.cs), a CTA a row in two kernels:
//   SampleWeights  the row's scores read where they are stored into its weights in lane storage: s = x / temperature in
//                  the lane, the largest by the block's reduction (a NaN is no score), then exp(s − largest) (1 at each
//                  +∞ of a row whose largest is +∞; 0 for a row with no score).
//   SampleDraw     over the weights (lane storage: one kernel a lane, whatever the type): each rule's threshold by
//                  narrowing the lane's bit patterns between 0 and 1 — a pass of the row a step, a thread its columns
//                  (tid, tid + 256, …) against the step's values, the block's reductions of the counts (top-k: sixteen
//                  ways a pass, the weights' bits compared as integers) or of the sums (the nucleus: sixteen ways in the
//                  f32 lane, two in the f64 lane, a sum a value being an add a weight; where 256 weights or fewer are
//                  at least top-k's threshold they are gathered into shared memory instead, a thread a gathered weight
//                  summing those at least its own) — then, the weights below the threshold taken as 0 where they are
//                  read, the kept weights' chunk sums and their prefix (SampleRows's), the seed's uniform at (step, row)
//                  with the step read where it is, and the first kept column at which the running sum reaches it (the
//                  last kept one when rounding leaves none; −1 for a row that keeps none).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>The shared bytes of SampleDraw: the reduction's slots, a chunk sum a thread, the total, the gathered weights.</summary>
    public const int SampleDrawShared = ReduceBytes + 256 * 8 + 16 + 256 * 8;

    /// <summary>w (lane storage, rows × cols) = each row's weights. Params: x (f; row r at r·stride + off), w, cols, stride,
    /// off, temperature (lane); a CTA a row.</summary>
    public static string SampleWeights(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"sample_weights_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string px = Param64(b, "p_x"), pw = Param64(b, "p_w"), cols = Param32(b, "p_cols"), stride = Param32(b, "p_stride"), off = Param32(b, "p_off");
        string temp = lane.NewReg(b), start = b.R(), wbase = b.R();
        b.L($"ld.param.{lane.Op} {temp}, [p_temp];");
        b.L($"mad.lo.u32 {start}, {row}, {stride}, {off};");
        b.L($"mul.lo.u32 {wbase}, {row}, {cols};");
        string lowest = lane.Literal(double.NegativeInfinity), highest = lane.Literal(double.PositiveInfinity);
        // The score of column c (s = x / temperature) and whether it is a NaN.
        (string S, string IsNan) Score(string c)
        {
            string idx = b.R(), pn = b.P();
            b.L($"add.u32 {idx}, {start}, {c};");
            string s = LaneOp(b, lane, "div", Codec.UnpackAt(b, f, px, idx, lane), temp);
            b.L($"testp.notanumber.{lane.Op} {pn}, {s};");
            return (s, pn);
        }

        // The largest score.
        string best = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {best}, {lowest};");
        {
            string c = b.R(), pEnd = b.P(), held = lane.NewReg(b);
            b.L($"mov.u32 {c}, {tid};");
            b.Label("TOP");
            b.L($"setp.ge.u32 {pEnd}, {c}, {cols};");
            b.L($"@{pEnd} bra TOP_END;");
            var (s, pn) = Score(c);
            b.L($"selp.{lane.Op} {held}, {lowest}, {s}, {pn};");
            b.L($"max.{lane.Op} {best}, {best}, {held};");
            b.L($"add.u32 {c}, {c}, 256;");
            b.L("bra TOP;");
            b.Label("TOP_END");
        }

        string m = BlockReduce(b, lane, tid, best, "max", 256);
        string pNone = b.P(), pTop = b.P();
        b.L($"setp.le.{lane.Op} {pNone}, {m}, {lowest};");                        // no score above −∞
        b.L($"setp.ge.{lane.Op} {pTop}, {m}, {highest};");                         // a row whose largest is +∞
        {
            string c = b.R(), pEnd = b.P(), d = lane.NewReg(b), e = lane.NewReg(b), pIsTop = b.P(), tops = lane.NewReg(b), a = b.RD(), idx = b.R();
            b.L($"mov.u32 {c}, {tid};");
            b.Label("WEIGH");
            b.L($"setp.ge.u32 {pEnd}, {c}, {cols};");
            b.L($"@{pEnd} bra DONE;");
            var (s, pn) = Score(c);
            b.L($"sub.rn.{lane.Op} {d}, {s}, {m};");
            string ex = Codec.Exp(b, lane, d);
            b.L($"selp.{lane.Op} {e}, {lane.Zero}, {ex}, {pn};");
            b.L($"setp.ge.{lane.Op} {pIsTop}, {s}, {highest};");
            b.L($"selp.{lane.Op} {tops}, {lane.Literal(1)}, {lane.Zero}, {pIsTop};");
            b.L($"selp.{lane.Op} {e}, {tops}, {e}, {pTop};");
            b.L($"selp.{lane.Op} {e}, {lane.Zero}, {e}, {pNone};");
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

    /// <summary>picks[row] and (prob not 0) prob[row] (lane storage) from the row's weights w (lane storage, rows × cols,
    /// read only). Params: w, picks, prob, step (the device's, read where it is), cols,
    /// k (0: no top-k), key (the seed's), p (0: no nucleus), q (0: no min-p) (lane); a CTA a row.</summary>
    public static string SampleDraw(Gpu gpu, Lane lane, out string entry)
    {
        entry = $"sample_draw_{lane.Name}";
        string bk = lane.Wide ? "64" : "32", oneBits = lane.Wide ? "0x3FF0000000000000" : "0x3F800000";
        var b = new PtxBuilder();
        string Bits() => lane.Wide ? b.RD() : b.R();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string pw = Param64(b, "p_w"), picks = Param64(b, "p_picks"), prob = Param64(b, "p_prob"), pstep = Param64(b, "p_step");
        string cols = Param32(b, "p_cols"), k = Param32(b, "p_k"), key = Param32(b, "p_key");
        string p = lane.NewReg(b), q = lane.NewReg(b), wbase = b.R(), smem = b.R();
        b.L($"ld.param.{lane.Op} {p}, [p_p];");
        b.L($"ld.param.{lane.Op} {q}, [p_q];");
        b.L($"mul.lo.u32 {wbase}, {row}, {cols};");
        b.L($"mov.u32 {smem}, smem;");
        // A pass of the row: this thread's columns (tid, tid + 256, …), each weight loaded (its address for a store).
        void Pass(string label, Action<string, string, string> body)
        {
            string c = b.R(), pEnd = b.P(), a = b.RD(), idx = b.R(), e = lane.NewReg(b);
            b.L($"mov.u32 {c}, {tid};");
            b.Label(label);
            b.L($"setp.ge.u32 {pEnd}, {c}, {cols};");
            b.L($"@{pEnd} bra {label}_END;");
            b.L($"add.u32 {idx}, {wbase}, {c};");
            b.L($"mul.wide.u32 {a}, {idx}, {lane.Size};");
            b.L($"add.u64 {a}, {pw}, {a};");
            b.L($"ld.global.{lane.Mem} {e}, [{a}];");
            body(e, c, a);
            b.L($"add.u32 {c}, {c}, 256;");
            b.L($"bra {label};");
            b.Label($"{label}_END");
        }

        // A step's values between lo and hi: lo + j · max(1, (hi − lo + 1) / ways) held to hi, j = 1 .. ways − 1, as bits, as
        // lane values, and the bits below each (the rule holds up to some value and fails past it: the step keeps the last
        // value at which it holds as lo and the bits below the first at which it fails as hi).
        const int CountWays = 16;
        int sumWays = lane.Wide ? 2 : 16;
        (string[] At, string[] Value, string[] Below) Step(string lo, string hi, int ways)
        {
            string span = Bits(), step = Bits();
            b.L($"sub.u{bk} {span}, {hi}, {lo};");
            b.L($"add.u{bk} {span}, {span}, 1;");
            b.L($"shr.u{bk} {step}, {span}, {int.Log2(ways)};");
            b.L($"max.u{bk} {step}, {step}, 1;");
            var at = new string[ways];
            var value = new string[ways];
            var below = new string[ways];
            for (int j = 1; j < ways; j++)
            {
                at[j] = Bits(); value[j] = lane.NewReg(b); below[j] = Bits();
                b.L($"mul.lo.u{bk} {at[j]}, {step}, {j};");
                b.L($"add.u{bk} {at[j]}, {at[j]}, {lo};");
                b.L($"min.u{bk} {at[j]}, {at[j]}, {hi};");
                b.L($"mov.b{bk} {value[j]}, {at[j]};");
                b.L($"sub.u{bk} {below[j]}, {at[j]}, 1;");
            }

            return (at, value, below);
        }

        // lo and hi after a step whose rule holds at value j where holds[j].
        void Narrow(string lo, string hi, string[] at, string[] below, string[] holds)
        {
            for (int j = 1; j < at.Length; j++) b.L($"selp.u{bk} {lo}, {at[j]}, {lo}, {holds[j]};");
            for (int j = at.Length - 1; j >= 1; j--) b.L($"selp.u{bk} {hi}, {hi}, {below[j]}, {holds[j]};");
        }

        string t = lane.NewReg(b);                                               // the threshold: 0 keeps every weight
        b.L($"mov.{lane.Reg} {t}, {lane.Zero};");

        // Top-k: the largest value at which k weights or more are at least it (a count of exactly k ends the search).
        {
            string pNoK = b.P(), lo = Bits(), hi = Bits(), pDone = b.P();
            b.L($"setp.eq.u32 {pNoK}, {k}, 0;");
            b.L($"@{pNoK} bra K_END;");
            b.L($"mov.u{bk} {lo}, 0;");
            b.L($"mov.u{bk} {hi}, {oneBits};");
            b.Label("K_LOOP");
            b.L($"setp.ge.u{bk} {pDone}, {lo}, {hi};");
            b.L($"@{pDone} bra K_DONE;");
            var (at, _, below) = Step(lo, hi, CountWays);
            var count = new string[CountWays];
            for (int j = 1; j < CountWays; j++) { count[j] = b.R(); b.L($"mov.u32 {count[j]}, 0;"); }
            Pass("K_PASS", (e, _, _) =>
            {
                // A weight is 0 or above: its bits order as it does (integer compares, whatever the lane).
                string pIn = b.P(), one = b.R(), bits = Bits();
                b.L($"mov.b{bk} {bits}, {e};");
                for (int j = 1; j < CountWays; j++)
                {
                    b.L($"setp.ge.u{bk} {pIn}, {bits}, {at[j]};");
                    b.L($"selp.u32 {one}, 1, 0, {pIn};");
                    b.L($"add.u32 {count[j]}, {count[j]}, {one};");
                }
            });
            var holds = new string[CountWays];
            var exact = new string[CountWays];
            for (int j = 1; j < CountWays; j++)
            {
                string total = BlockReduce(b, Lane.U32, tid, count[j], "add", 256);
                holds[j] = b.P(); exact[j] = b.P();
                b.L($"setp.ge.u32 {holds[j]}, {total}, {k};");
                b.L($"setp.eq.u32 {exact[j]}, {total}, {k};");
            }

            Narrow(lo, hi, at, below, holds);
            for (int j = 1; j < CountWays; j++)
            {
                b.L($"@{exact[j]} mov.u{bk} {lo}, {at[j]};");
                b.L($"@{exact[j]} mov.u{bk} {hi}, {at[j]};");
            }

            b.L("bra K_LOOP;");
            b.Label("K_DONE");
            b.L($"mov.b{bk} {t}, {lo};");
            b.Label("K_END");
        }

        // The nucleus: the largest value at which the weights at least it sum to p of the weights at least t.
        {
            string pNoP = b.P(), lo = Bits(), hi = Bits(), pDone = b.P(), part = lane.NewReg(b), target = lane.NewReg(b);
            b.L($"setp.gt.{lane.Op} {pNoP}, {p}, {lane.Zero};");
            b.L($"@!{pNoP} bra P_END;");
            // The weights at least t: where they fit a block's threads (top-k has left few), gathered into shared memory — a
            // thread's count, the counts' exclusive prefix by the first thread, each thread's weights at its place — and the
            // nucleus read from them: thread i's weight v, the gathered weights at least v summed in their order, v kept
            // where that reaches p of them all; the largest such v is the threshold (no pass of the row a step).
            {
                int counts = ReduceBytes, slots = ReduceBytes + 256 * 8 + 16;
                string mineCount = b.R(), pMany = b.P(), slot = b.R();
                b.L($"mov.u32 {mineCount}, 0;");
                Pass("N_PASS", (e, _, _) =>
                {
                    string pIn = b.P(), one = b.R();
                    b.L($"setp.ge.{lane.Op} {pIn}, {e}, {t};");
                    b.L($"selp.u32 {one}, 1, 0, {pIn};");
                    b.L($"add.u32 {mineCount}, {mineCount}, {one};");
                });
                string n = BlockReduce(b, Lane.U32, tid, mineCount, "add", 256);
                b.L($"setp.gt.u32 {pMany}, {n}, 256;");
                b.L($"@{pMany} bra P_SEARCH;");
                b.L($"mad.lo.u32 {slot}, {tid}, 8, {smem};");
                b.L($"st.shared.u32 [{slot}+{counts}], {mineCount};");
                b.L("bar.sync 0;");
                string p0 = b.P(), s = b.R(), acc = b.R(), addr = b.R(), sv = b.R(), pS = b.P();
                b.L($"setp.ne.u32 {p0}, {tid}, 0;");
                b.L($"@{p0} bra G_SCAN_SKIP;");
                b.L($"mov.u32 {acc}, 0;");
                b.L($"mov.u32 {s}, 0;");
                b.Label("G_SCAN");
                b.L($"setp.ge.u32 {pS}, {s}, 256;");
                b.L($"@{pS} bra G_SCAN_SKIP;");
                b.L($"mad.lo.u32 {addr}, {s}, 8, {smem};");
                b.L($"ld.shared.u32 {sv}, [{addr}+{counts}];");
                b.L($"st.shared.u32 [{addr}+{counts}], {acc};");
                b.L($"add.u32 {acc}, {acc}, {sv};");
                b.L($"add.u32 {s}, {s}, 1;");
                b.L("bra G_SCAN;");
                b.Label("G_SCAN_SKIP");
                b.L("bar.sync 0;");
                string place = b.R();
                b.L($"ld.shared.u32 {place}, [{slot}+{counts}];");
                Pass("G_PASS", (e, _, _) =>
                {
                    string pIn = b.P(), one = b.R(), at = b.R();
                    b.L($"setp.ge.{lane.Op} {pIn}, {e}, {t};");
                    b.L($"mad.lo.u32 {at}, {place}, 8, {smem};");
                    b.L($"@{pIn} st.shared.{lane.Mem} [{at}+{slots}], {e};");
                    b.L($"selp.u32 {one}, 1, 0, {pIn};");
                    b.L($"add.u32 {place}, {place}, {one};");
                });
                b.L("bar.sync 0;");
                string pMine = b.P(), own = b.R(), v = lane.NewReg(b), all = lane.NewReg(b), above = lane.NewReg(b), i = b.R(), pEnd = b.P(), ia = b.R(), w = lane.NewReg(b), pIn2 = b.P(), kept = lane.NewReg(b);
                b.L($"setp.lt.u32 {pMine}, {tid}, {n};");
                b.L($"mad.lo.u32 {own}, {tid}, 8, {smem};");
                b.L($"ld.shared.{lane.Mem} {v}, [{own}+{slots}];");
                b.L($"mov.{lane.Reg} {all}, {lane.Zero};");
                b.L($"mov.{lane.Reg} {above}, {lane.Zero};");
                b.L($"mov.u32 {i}, 0;");
                b.Label("G_SUM");
                b.L($"setp.ge.u32 {pEnd}, {i}, {n};");
                b.L($"@{pEnd} bra G_SUM_END;");
                b.L($"mad.lo.u32 {ia}, {i}, 8, {smem};");
                b.L($"ld.shared.{lane.Mem} {w}, [{ia}+{slots}];");
                b.L(lane.Add(all, all, w));
                b.L($"setp.ge.{lane.Op} {pIn2}, {w}, {v};");
                b.L($"selp.{lane.Op} {kept}, {w}, {lane.Zero}, {pIn2};");
                b.L(lane.Add(above, above, kept));
                b.L($"add.u32 {i}, {i}, 1;");
                b.L("bra G_SUM;");
                b.Label("G_SUM_END");
                string goal = lane.NewReg(b), pHolds = b.P(), candidate = lane.NewReg(b);
                b.L(lane.Mul(goal, p, all));
                b.L($"setp.ge.{lane.Op} {pHolds}, {above}, {goal};");
                b.L($"and.pred {pHolds}, {pHolds}, {pMine};");
                b.L($"selp.{lane.Op} {candidate}, {v}, {lane.Zero}, {pHolds};");
                string nucleus = BlockReduce(b, lane, tid, candidate, "max", 256);
                b.L($"max.{lane.Op} {t}, {t}, {nucleus};");
                b.L("bra P_END;");
                b.Label("P_SEARCH");
            }

            b.L($"mov.{lane.Reg} {part}, {lane.Zero};");
            Pass("M_PASS", (e, _, _) =>
            {
                string pIn = b.P(), kept = lane.NewReg(b);
                b.L($"setp.ge.{lane.Op} {pIn}, {e}, {t};");
                b.L($"selp.{lane.Op} {kept}, {e}, {lane.Zero}, {pIn};");
                b.L(lane.Add(part, part, kept));
            });
            string mass = BlockReduce(b, lane, tid, part, "add", 256);
            b.L(lane.Mul(target, p, mass));
            b.L($"mov.b{bk} {lo}, {t};");
            b.L($"mov.u{bk} {hi}, {oneBits};");
            b.Label("P_LOOP");
            b.L($"setp.ge.u{bk} {pDone}, {lo}, {hi};");
            b.L($"@{pDone} bra P_DONE;");
            var (at, value, below) = Step(lo, hi, sumWays);
            var sum = new string[sumWays];
            for (int j = 1; j < sumWays; j++) { sum[j] = lane.NewReg(b); b.L($"mov.{lane.Reg} {sum[j]}, {lane.Zero};"); }
            Pass("P_PASS", (e, _, _) =>
            {
                string pIn = b.P(), kept = lane.NewReg(b);
                for (int j = 1; j < sumWays; j++)
                {
                    b.L($"setp.ge.{lane.Op} {pIn}, {e}, {value[j]};");
                    b.L($"selp.{lane.Op} {kept}, {e}, {lane.Zero}, {pIn};");
                    b.L(lane.Add(sum[j], sum[j], kept));
                }
            });
            var holds = new string[sumWays];
            for (int j = 1; j < sumWays; j++)
            {
                string held = BlockReduce(b, lane, tid, sum[j], "add", 256);
                holds[j] = b.P();
                b.L($"setp.ge.{lane.Op} {holds[j]}, {held}, {target};");
            }

            Narrow(lo, hi, at, below, holds);
            b.L("bra P_LOOP;");
            b.Label("P_DONE");
            b.L($"mov.b{bk} {t}, {lo};");
            b.Label("P_END");
        }

        b.L($"max.{lane.Op} {t}, {t}, {q};");                                      // min-p

        // The kept weights' sums over this thread's chunk of consecutive columns, their exclusive prefix and the total by the
        // first thread (SampleRows's).
        string chunk = b.R(), c0 = b.R(), c1 = b.R(), mine = b.R();
        b.L($"add.u32 {chunk}, {cols}, 255;");
        b.L($"shr.u32 {chunk}, {chunk}, 8;");
        b.L($"mul.lo.u32 {c0}, {tid}, {chunk};");
        b.L($"min.u32 {c0}, {c0}, {cols};");
        b.L($"add.u32 {c1}, {c0}, {chunk};");
        b.L($"min.u32 {c1}, {c1}, {cols};");
        b.L($"mad.lo.u32 {mine}, {tid}, 8, {smem};");
        // The kept weight of column c: its weight where that is at least the threshold, else 0.
        string Weight(string c)
        {
            string idx = b.R(), a = b.RD(), e = lane.NewReg(b), w = lane.NewReg(b), pIn = b.P();
            b.L($"add.u32 {idx}, {wbase}, {c};");
            b.L($"mul.wide.u32 {a}, {idx}, {lane.Size};");
            b.L($"add.u64 {a}, {pw}, {a};");
            b.L($"ld.global.{lane.Mem} {e}, [{a}];");
            b.L($"setp.ge.{lane.Op} {pIn}, {e}, {t};");
            b.L($"selp.{lane.Op} {w}, {e}, {lane.Zero}, {pIn};");
            return w;
        }

        {
            string sum = lane.NewReg(b), c = b.R(), pEnd = b.P();
            b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
            b.L($"mov.u32 {c}, {c0};");
            b.Label("SUM");
            b.L($"setp.ge.u32 {pEnd}, {c}, {c1};");
            b.L($"@{pEnd} bra SUM_END;");
            b.L(lane.Add(sum, sum, Weight(c)));
            b.L($"add.u32 {c}, {c}, 1;");
            b.L("bra SUM;");
            b.Label("SUM_END");
            b.L($"st.shared.{lane.Mem} [{mine}+{ReduceBytes}], {sum};");
            b.L("bar.sync 0;");
            string p0 = b.P(), s = b.R(), acc = lane.NewReg(b), addr = b.R(), sv = lane.NewReg(b), pS = b.P();
            b.L($"setp.ne.u32 {p0}, {tid}, 0;");
            b.L($"@{p0} bra SCAN_SKIP;");
            b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
            b.L($"mov.u32 {s}, 0;");
            b.Label("SCAN");
            b.L($"setp.ge.u32 {pS}, {s}, 256;");
            b.L($"@{pS} bra SCAN_END;");
            b.L($"mad.lo.u32 {addr}, {s}, 8, {smem};");
            b.L($"ld.shared.{lane.Mem} {sv}, [{addr}+{ReduceBytes}];");
            b.L($"st.shared.{lane.Mem} [{addr}+{ReduceBytes}], {acc};");
            b.L(lane.Add(acc, acc, sv));
            b.L($"add.u32 {s}, {s}, 1;");
            b.L("bra SCAN;");
            b.Label("SCAN_END");
            b.L($"st.shared.{lane.Mem} [{smem}+{ReduceBytes + 256 * 8}], {acc};");
            b.Label("SCAN_SKIP");
            b.L("bar.sync 0;");
        }

        string prefix = lane.NewReg(b), total2 = lane.NewReg(b), target2 = lane.NewReg(b);
        b.L($"ld.shared.{lane.Mem} {prefix}, [{mine}+{ReduceBytes}];");
        b.L($"ld.shared.{lane.Mem} {total2}, [{smem}+{ReduceBytes + 256 * 8}];");
        // The draw: Draws.Uniform(key, step, row) = the top 24 bits of Bits(Bits(key, step), row) over 2^24, times the total.
        {
            string step = b.R(), top = b.R(), u = lane.NewReg(b);
            b.L($"ld.global.u32 {step}, [{pstep}];");
            string bits = Draw(b, Draw(b, key, step), row);
            b.L($"shr.u32 {top}, {bits}, 8;");
            b.L($"cvt.rn.{lane.Op}.u32 {u}, {top};");
            b.L(lane.Mul(u, u, lane.Literal(1.0 / 16777216)));
            b.L(lane.Mul(target2, u, total2));
        }

        // The first kept column of this thread's chunk at which the running sum reaches the target (cols: none), the least
        // over the block.
        string found = b.R(), lastMine = b.R();
        {
            string run = lane.NewReg(b), c = b.R(), pEnd = b.P(), pKept = b.P(), pReached = b.P(), pOpen = b.P(), next = b.R();
            b.L($"mov.{lane.Reg} {run}, {prefix};");
            b.L($"mov.u32 {found}, {cols};");
            b.L($"mov.u32 {lastMine}, 0;");
            b.L($"mov.u32 {c}, {c0};");
            b.Label("FIND");
            b.L($"setp.ge.u32 {pEnd}, {c}, {c1};");
            b.L($"@{pEnd} bra FIND_END;");
            string w = Weight(c);
            b.L(lane.Add(run, run, w));
            b.L($"setp.gt.{lane.Op} {pKept}, {w}, {lane.Zero};");
            b.L($"add.u32 {next}, {c}, 1;");
            b.L($"selp.u32 {lastMine}, {next}, {lastMine}, {pKept};");                // the chunk's columns ascend
            b.L($"setp.ge.{lane.Op} {pReached}, {run}, {target2};");
            b.L($"setp.eq.u32 {pOpen}, {found}, {cols};");
            b.L($"and.pred {pReached}, {pReached}, {pKept};");
            b.L($"and.pred {pReached}, {pReached}, {pOpen};");
            b.L($"selp.u32 {found}, {c}, {found}, {pReached};");
            b.L($"add.u32 {c}, {c}, 1;");
            b.L("bra FIND;");
            b.Label("FIND_END");
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
            b.L($"mul.wide.u32 {pa}, {row}, 4;");
            b.L($"add.u64 {pa}, {picks}, {pa};");
            b.L($"st.global.u32 [{pa}], {pick};");
            b.L($"setp.eq.u64 {pNoProb}, {prob}, 0;");
            b.L($"@{pNoProb} bra DONE;");
            b.L($"setp.ge.s32 {pValid}, {pick}, 0;");
            b.L($"selp.u32 {at}, {pick}, 0, {pValid};");
            string w = Weight(at);
            b.L($"div.rn.{lane.Op} {value}, {w}, {total2};");
            b.L($"selp.{lane.Op} {value}, {value}, {lane.Zero}, {pValid};");
            b.L($"mul.wide.u32 {qa}, {row}, {lane.Size};");
            b.L($"add.u64 {qa}, {prob}, {qa};");
            b.L($"st.global.{lane.Mem} [{qa}], {value};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_w, .param .u64 p_picks, .param .u64 p_prob, .param .u64 p_step, .param .u32 p_cols, .param .u32 p_k, .param .u32 p_key, .param .{lane.Op} p_p, .param .{lane.Op} p_q",
            SampleDrawShared);
    }
}
