// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's ctc (Ctc.cs): a block of 256 threads per sequence, on lane storage (f32; f64 for float64).
//   1. (CtcRows, a kernel of its own over every row) each (frame, sequence) row's log-sum-exp over the classes: a warp
//      per row (the classes strided over its lanes, max and sum by shuffles); for the gradient the row's softmax y_t(c),
//      and 0 for the rows past the sequence's frames;
//   2. α forward: the threads over the extended states, the previous frame's α in shared memory (two buffers, two −∞
//      pads before state 0), a barrier per frame; α of every frame kept in global scratch for the backward pass;
//      thread 0 takes log P and stores the loss;
//   3. (the backward kernel) β backward in the same shared buffers (pads after the last state); per frame (its row
//      holding the softmax from 1) each state's exp(α + β − log y − log P) into shared memory, and the
//      first state of each class (the host marks it) sums its class's states in state order and writes y − sum: no
//      atomics, the same order on every run. A sequence without an alignment: loss +∞, gradient 0.
// The host gives each sequence's state classes and flags (bit 0: α may skip from s − 2; bit 1: the class's first
// state), with a row stride of max S + 2 (zeros past S).
namespace Qiyas;

internal static partial class Ptx
{
    public const int CtcThreads = 256;

    /// <summary>Shared bytes of the ctc kernel for states up to cap: two state buffers with pads, the per-state terms, log P.</summary>
    public static int CtcShared(Lane lane, int cap) => (3 * cap + 5) * lane.Size;

    /// <summary>Shared bytes of the one-state kernel (cap ≤ CtcThreads): CtcShared, the warps' blank partials and each
    /// state's next state of its class.</summary>
    public static int CtcOneShared(Lane lane, int cap) => CtcShared(lane, cap) + CtcThreads / 32 * lane.Size + cap * 4;

    /// <summary>Each (frame, sequence) row's log-sum-exp over the classes, a warp per row; for the gradient the row's softmax
    /// (and 0 for the rows past the sequence's frames).</summary>
    public static string CtcRows(Gpu gpu, Lane lane, bool backward, out string entry)
    {
        entry = $"ctc_rows_{lane.Name}_{(backward ? "bwd" : "fwd")}";
        int size = lane.Size;
        string op = lane.Op, mem = lane.Mem, negInf = lane.Literal(double.NegativeInfinity), zero = lane.Literal(0);
        var b = new PtxBuilder();
        string tid = b.R(), blk = b.R(), warp = b.R(), ln = b.R(), r = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {blk}, %ctaid.x;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {ln}, {tid}, 31;");
        b.L($"mad.lo.u32 {r}, {blk}, {CtcThreads / 32}, {warp};");
        string x = Param64(b, "p_x"), lseG = Param64(b, "p_lse"), ilen = Param64(b, "p_ilen"), grad = Param64(b, "p_grad");
        string T = Param32(b, "p_t"), B = Param32(b, "p_b"), C = Param32(b, "p_c");
        string rows = b.R(), pOut = b.P();
        b.L($"mul.lo.u32 {rows}, {T}, {B};");
        b.L($"setp.ge.u32 {pOut}, {r}, {rows};");
        b.L($"@{pOut} bra DONE;");

        string Addr(string basePtr, string index, int elem)
        {
            string a = b.RD();
            b.L($"mul.wide.u32 {a}, {index}, {elem};");
            b.L($"add.u64 {a}, {basePtr}, {a};");
            return a;
        }

        string Sub(string p, string q) { string d = lane.NewReg(b); b.L($"sub.rn.{op} {d}, {p}, {q};"); return d; }

        void Shuffle(string acc, bool max)
        {
            for (int off = 16; off >= 1; off >>= 1)
            {
                string other = lane.NewReg(b);
                if (lane.Wide)
                {
                    string l0 = b.R(), h0 = b.R(), l1 = b.R(), h1 = b.R();
                    b.L($"mov.b64 {{{l0}, {h0}}}, {acc};");
                    b.L($"shfl.sync.bfly.b32 {l1}, {l0}, {off}, 0x1f, 0xffffffff;");
                    b.L($"shfl.sync.bfly.b32 {h1}, {h0}, {off}, 0x1f, 0xffffffff;");
                    b.L($"mov.b64 {other}, {{{l1}, {h1}}};");
                }
                else
                {
                    string bits = b.R(), bits2 = b.R();
                    b.L($"mov.b32 {bits}, {acc};");
                    b.L($"shfl.sync.bfly.b32 {bits2}, {bits}, {off}, 0x1f, 0xffffffff;");
                    b.L($"mov.b32 {other}, {bits2};");
                }

                b.L(max ? $"max.{op} {acc}, {acc}, {other};" : lane.Add(acc, acc, other));
            }
        }

        // The row's frame and sequence; its first value.
        string t = b.R(), seq = b.R(), tb = b.R(), row = b.R(), pPast = b.P();
        b.L($"div.u32 {t}, {r}, {B};");
        b.L($"rem.u32 {seq}, {r}, {B};");
        b.L($"ld.global.u32 {tb}, [{Addr(ilen, seq, 4)}];");
        b.L($"mul.lo.u32 {row}, {r}, {C};");
        b.L($"setp.ge.u32 {pPast}, {t}, {tb};");
        string c = b.R(), pC = b.P(), i = b.R();
        if (backward)
        {
            // Past the sequence's frames: 0.
            b.L($"@!{pPast} bra ROW;");
            b.L($"mov.u32 {c}, {ln};");
            b.Label("ZERO_C");
            b.L($"setp.ge.u32 {pC}, {c}, {C};");
            b.L($"@{pC} bra DONE;");
            b.L($"add.u32 {i}, {row}, {c};");
            b.L($"st.global.{mem} [{Addr(grad, i, size)}], {zero};");
            b.L($"add.u32 {c}, {c}, 32;");
            b.L("bra ZERO_C;");
        }
        else b.L($"@{pPast} bra DONE;");

        b.Label("ROW");
        string m = lane.NewReg(b), sum = lane.NewReg(b);
        b.L($"mov.{op} {m}, {negInf};");
        b.L($"mov.u32 {c}, {ln};");
        b.Label("MAX");
        b.L($"setp.ge.u32 {pC}, {c}, {C};");
        b.L($"@{pC} bra MAX_END;");
        b.L($"add.u32 {i}, {row}, {c};");
        string v = lane.NewReg(b);
        b.L($"ld.global.{mem} {v}, [{Addr(x, i, size)}];");
        b.L($"max.{op} {m}, {m}, {v};");
        b.L($"add.u32 {c}, {c}, 32;");
        b.L("bra MAX;");
        b.Label("MAX_END");
        Shuffle(m, true);
        b.L($"mov.{op} {sum}, {zero};");
        b.L($"mov.u32 {c}, {ln};");
        b.Label("SUM");
        b.L($"setp.ge.u32 {pC}, {c}, {C};");
        b.L($"@{pC} bra SUM_END;");
        b.L($"add.u32 {i}, {row}, {c};");
        string v2 = lane.NewReg(b);
        b.L($"ld.global.{mem} {v2}, [{Addr(x, i, size)}];");
        string ev = Codec.Exp(b, lane, Sub(v2, m));
        b.L(lane.Add(sum, sum, ev));
        b.L($"add.u32 {c}, {c}, 32;");
        b.L("bra SUM;");
        b.Label("SUM_END");
        Shuffle(sum, false);
        string lg = Fn.Log(b, lane, sum), res = lane.NewReg(b), p0 = b.P();
        b.L(lane.Add(res, m, lg));
        b.L($"setp.ne.u32 {p0}, {ln}, 0;");
        b.L($"@!{p0} st.global.{mem} [{Addr(lseG, r, size)}], {res};");
        if (backward)
        {
            b.L($"mov.u32 {c}, {ln};");
            b.Label("Y");
            b.L($"setp.ge.u32 {pC}, {c}, {C};");
            b.L($"@{pC} bra DONE;");
            b.L($"add.u32 {i}, {row}, {c};");
            string a = Addr(x, i, size), v3 = lane.NewReg(b);
            b.L($"ld.global.{mem} {v3}, [{a}];");
            string y = Codec.Exp(b, lane, Sub(v3, res));
            b.L($"st.global.{mem} [{Addr(grad, i, size)}], {y};");
            b.L($"add.u32 {c}, {c}, 32;");
            b.L("bra Y;");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_lse, .param .u64 p_ilen, .param .u64 p_grad, .param .u32 p_t, .param .u32 p_b, .param .u32 p_c");
    }

    public static string Ctc(Gpu gpu, Lane lane, int cap, bool backward, out string entry) => Ctc(gpu, lane, cap, backward, false, out entry);

    /// <summary>oneState (states up to cap ≤ CtcThreads: a thread per state): each state's class and flags read once into
    /// registers, the next frame's logit, log-sum-exp (and α, backward) loaded before the frame's barrier, and in the
    /// backward the blank's sum over the even states by the warps' shuffles (fixed order), a label class's by its states'
    /// links (flags bits 2 up, CtcArgs.States) instead of a scan over every later state.</summary>
    public static string Ctc(Gpu gpu, Lane lane, int cap, bool backward, bool oneState, out string entry)
    {
        if (oneState && cap > CtcThreads) throw new NotSupportedException($"ctc: the one-state kernel takes up to {CtcThreads} states, not {cap}");
        entry = $"ctc{(oneState ? "1" : "")}_{lane.Name}_{(backward ? "bwd" : "fwd")}_{cap}";
        int size = lane.Size, bufBytes = (cap + 2) * size, gOff = 2 * bufBytes, pOff = gOff + cap * size;
        string op = lane.Op, mem = lane.Mem, negInf = lane.Literal(double.NegativeInfinity), zero = lane.Literal(0);
        var b = new PtxBuilder();
        string tid = b.R(), seq = b.R(), sm = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {seq}, %ctaid.x;");
        b.L($"mov.u32 {sm}, smem;");
        string x = Param64(b, "p_x"), lseG = Param64(b, "p_lse"), alphaG = Param64(b, "p_alpha"), clsG = Param64(b, "p_cls"), flagsG = Param64(b, "p_flags");
        string ilen = Param64(b, "p_ilen"), slen = Param64(b, "p_slen"), loss = Param64(b, "p_loss"), grad = Param64(b, "p_grad");
        string T = Param32(b, "p_t"), B = Param32(b, "p_b"), C = Param32(b, "p_c"), smax = Param32(b, "p_smax");

        string Addr(string basePtr, string index, int elem)
        {
            string a = b.RD();
            b.L($"mul.wide.u32 {a}, {index}, {elem};");
            b.L($"add.u64 {a}, {basePtr}, {a};");
            return a;
        }

        string LdG(string basePtr, string index)
        {
            string v = lane.NewReg(b);
            b.L($"ld.global.{mem} {v}, [{Addr(basePtr, index, size)}];");
            return v;
        }

        string LdU(string basePtr, string index)
        {
            string v = b.R();
            b.L($"ld.global.u32 {v}, [{Addr(basePtr, index, 4)}];");
            return v;
        }

        string LdS(string bufBase, string index)
        {
            string a = b.R(), v = lane.NewReg(b);
            b.L($"mad.lo.u32 {a}, {index}, {size}, {bufBase};");
            b.L($"ld.shared.{mem} {v}, [{a}];");
            return v;
        }

        void StS(string bufBase, string index, string v)
        {
            string a = b.R();
            b.L($"mad.lo.u32 {a}, {index}, {size}, {bufBase};");
            b.L($"st.shared.{mem} [{a}], {v};");
        }

        string Sub(string p, string q) { string r = lane.NewReg(b); b.L($"sub.rn.{op} {r}, {p}, {q};"); return r; }

        string Lse3(string p, string q, string r)
        {
            string m = lane.NewReg(b), ms = lane.NewReg(b), none = b.P();
            b.L($"max.{op} {m}, {p}, {q};");
            b.L($"max.{op} {m}, {m}, {r};");
            b.L($"setp.eq.{op} {none}, {m}, {negInf};");
            b.L($"selp.{op} {ms}, {zero}, {m}, {none};");
            string e1 = Codec.Exp(b, lane, Sub(p, ms)), e2 = Codec.Exp(b, lane, Sub(q, ms)), e3 = Codec.Exp(b, lane, Sub(r, ms)), sum = lane.NewReg(b);
            b.L(lane.Add(sum, e1, e2));
            b.L(lane.Add(sum, sum, e3));
            string lg = Fn.Log(b, lane, sum), res = lane.NewReg(b);
            b.L(lane.Add(res, ms, lg));
            b.L($"selp.{op} {res}, {negInf}, {res}, {none};");
            return res;
        }

        // The sequence's frames, states and rows.
        string tb = LdU(ilen, seq), S = LdU(slen, seq), flagRow = b.R(), alphaRow = b.R();
        b.L($"add.u32 {flagRow}, {smax}, 2;");
        b.L($"mul.lo.u32 {flagRow}, {flagRow}, {seq};");                       // the sequence's first entry in cls / flags
        b.L($"mul.lo.u32 {alphaRow}, {seq}, {T};");                            // (seq · T + t) · smax: α_t's first state

        string RowBase(string t)
        {
            string r = b.R();
            b.L($"mad.lo.u32 {r}, {t}, {B}, {seq};");
            b.L($"mul.lo.u32 {r}, {r}, {C};");
            return r;
        }

        string LseAt(string t)
        {
            string i = b.R();
            b.L($"mad.lo.u32 {i}, {t}, {B}, {seq};");
            return LdG(lseG, i);
        }

        // log y_t(l'_s) = x[t][seq][cls(s)] − lse_t.
        string Emit(string rowBase, string lseT, string s)
        {
            string fi = b.R(), ci = b.R();
            b.L($"add.u32 {fi}, {flagRow}, {s};");
            string cls = LdU(clsG, fi);
            b.L($"add.u32 {ci}, {rowBase}, {cls};");
            return Sub(LdG(x, ci), lseT);
        }

        // Both state buffers −∞ (the pads included).
        void ClearBuffers(string tag)
        {
            string s = b.R(), p = b.P();
            b.L($"mov.u32 {s}, {tid};");
            b.Label($"CLR_{tag}");
            b.L($"setp.ge.u32 {p}, {s}, {cap + 2};");
            b.L($"@{p} bra CLR_{tag}_END;");
            StS(sm, s, negInf);
            string s1 = b.R();
            b.L($"add.u32 {s1}, {s}, {cap + 2};");
            StS(sm, s1, negInf);
            b.L($"add.u32 {s}, {s}, {CtcThreads};");
            b.L($"bra CLR_{tag};");
            b.Label($"CLR_{tag}_END");
            b.L("bar.sync 0;");
        }

        string BufBase(string t)
        {
            string r = b.R();
            b.L($"and.b32 {r}, {t}, 1;");
            b.L($"mad.lo.u32 {r}, {r}, {bufBytes}, {sm};");
            return r;
        }

        if (oneState)
        {
            int wOff = pOff + size, lOff = wOff + CtcThreads / 32 * size;
            // This thread's state (s = tid; past S it computes on state 0's addresses and stores nothing): its class, flags,
            // and for the backward the flags of s + 2 (the row's stride is smax + 2: zeros past S).
            string act = b.P(), sc = b.R(), fiO = b.R(), sc1 = b.R(), sc2 = b.R();
            b.L($"setp.lt.u32 {act}, {tid}, {S};");
            b.L($"selp.u32 {sc}, {tid}, 0, {act};");
            b.L($"add.u32 {fiO}, {flagRow}, {sc};");
            b.L($"add.u32 {sc1}, {sc}, 1;");
            b.L($"add.u32 {sc2}, {sc}, 2;");
            string myCls = LdU(clsG, fiO), myFl = LdU(flagsG, fiO), bit = b.R(), skipA = b.P();
            b.L($"and.b32 {bit}, {myFl}, 1;");
            b.L($"setp.eq.u32 {skipA}, {bit}, 0;");

            string XAt(string t)
            {
                string i = b.R();
                b.L($"add.u32 {i}, {RowBase(t)}, {myCls};");
                return LdG(x, i);
            }

            string AlphaAt(string t)
            {
                string ai = b.R();
                b.L($"add.u32 {ai}, {alphaRow}, {t};");
                b.L($"mad.lo.u32 {ai}, {ai}, {smax}, {sc};");
                return ai;
            }

            // Both state buffers −∞, each state's link to the next state of its class.
            {
                string link = b.R(), la = b.R();
                b.L($"shr.u32 {link}, {myFl}, 2;");
                b.L($"mad.lo.u32 {la}, {sc}, 4, {sm};");
                b.L($"@{act} st.shared.u32 [{la}+{lOff}], {link};");
            }

            ClearBuffers("A1");
            {
                string t = b.R(), pT = b.P(), xv = lane.NewReg(b), lv = lane.NewReg(b), zeroT = b.R();
                b.L($"mov.u32 {zeroT}, 0;");
                b.L($"mov.{op} {xv}, {XAt(zeroT)};");
                b.L($"mov.{op} {lv}, {LseAt(zeroT)};");
                b.L($"mov.u32 {t}, 0;");
                b.Label("A1_T");
                b.L($"setp.ge.u32 {pT}, {t}, {tb};");
                b.L($"@{pT} bra A1_END;");
                string e = Sub(xv, lv), tm1 = b.R();
                b.L($"sub.u32 {tm1}, {t}, 1;");
                string cur = BufBase(t), prev = BufBase(tm1), first = b.P();
                b.L($"setp.eq.u32 {first}, {t}, 0;");
                string p = LdS(prev, sc2), q = LdS(prev, sc1), r = LdS(prev, sc);
                b.L($"selp.{op} {r}, {negInf}, {r}, {skipA};");
                string v = Lse3(p, q, r);
                b.L(lane.Add(v, v, e));
                string v0 = lane.NewReg(b), pLow = b.P();
                b.L($"setp.lt.u32 {pLow}, {tid}, 2;");
                b.L($"selp.{op} {v0}, {e}, {negInf}, {pLow};");
                b.L($"selp.{op} {v}, {v0}, {v}, {first};");
                // The next frame's logit and log-sum-exp (the last frame's again past the end), loaded across the barrier.
                string tn = b.R(), tl = b.R();
                b.L($"add.u32 {tn}, {t}, 1;");
                b.L($"sub.u32 {tl}, {tb}, 1;");
                b.L($"min.u32 {tn}, {tn}, {tl};");
                b.L($"mov.{op} {xv}, {XAt(tn)};");
                b.L($"mov.{op} {lv}, {LseAt(tn)};");
                string ca = b.R();
                b.L($"mad.lo.u32 {ca}, {sc2}, {size}, {cur};");
                b.L($"@{act} st.shared.{mem} [{ca}], {v};");
                b.L($"@{act} st.global.{mem} [{Addr(alphaG, AlphaAt(t), size)}], {v};");
                b.L("bar.sync 0;");
                b.L($"add.u32 {t}, {t}, 1;");
                b.L("bra A1_T;");
                b.Label("A1_END");

                // log P from the last frame's last two states (index S + 1 and S: the pad when S = 1); the loss.
                string p0 = b.P();
                b.L($"setp.ne.u32 {p0}, {tid}, 0;");
                b.L($"@{p0} bra A1_LOSS_END;");
                string tl2 = b.R();
                b.L($"sub.u32 {tl2}, {tb}, 1;");
                string last = BufBase(tl2), sp1 = b.R();
                b.L($"add.u32 {sp1}, {S}, 1;");
                string a1 = LdS(last, sp1), a2 = LdS(last, S);
                string logP = Lse3(a1, a2, negInf), nl = lane.NewReg(b);
                b.L($"st.shared.{mem} [{sm}+{pOff}], {logP};");
                b.L($"neg.{op} {nl}, {logP};");
                b.L($"st.global.{mem} [{Addr(loss, seq, size)}], {nl};");
                b.Label("A1_LOSS_END");
                b.L("bar.sync 0;");
            }

            if (backward)
            {
                string logP = lane.NewReg(b), none = b.P();
                b.L($"ld.shared.{mem} {logP}, [{sm}+{pOff}];");
                b.L($"setp.eq.{op} {none}, {logP}, {negInf};");
                // No alignment: the sequence's frames' rows 0 too (the rows past its frames are the rows kernel's).
                {
                    string t = b.R(), pT = b.P(), end = b.R();
                    b.L($"mov.u32 {t}, 0;");
                    b.L($"selp.u32 {end}, {tb}, 0, {none};");
                    b.Label("Z1_T");
                    b.L($"setp.ge.u32 {pT}, {t}, {end};");
                    b.L($"@{pT} bra Z1_END;");
                    string row = RowBase(t), c = b.R(), pC = b.P(), i = b.R();
                    b.L($"mov.u32 {c}, {tid};");
                    b.Label("Z1_C");
                    b.L($"setp.ge.u32 {pC}, {c}, {C};");
                    b.L($"@{pC} bra Z1_C_END;");
                    b.L($"add.u32 {i}, {row}, {c};");
                    b.L($"st.global.{mem} [{Addr(grad, i, size)}], {zero};");
                    b.L($"add.u32 {c}, {c}, {CtcThreads};");
                    b.L("bra Z1_C;");
                    b.Label("Z1_C_END");
                    b.L($"add.u32 {t}, {t}, 1;");
                    b.L("bra Z1_T;");
                    b.Label("Z1_END");
                }

                b.L($"@{none} bra DONE;");
                string fi2 = b.R(), skipB = b.P(), rep = b.P(), even = b.P(), bit2 = b.R(), bit3 = b.R(), lane32 = b.R(), warp = b.R();
                b.L($"add.u32 {fi2}, {fiO}, 2;");
                string fl2 = LdU(flagsG, fi2);
                b.L($"and.b32 {bit2}, {fl2}, 1;");
                b.L($"setp.eq.u32 {skipB}, {bit2}, 0;");
                b.L($"and.b32 {bit3}, {myFl}, 2;");
                b.L($"setp.ne.and.u32 {rep}, {bit3}, 0, {act};");
                b.L($"and.b32 {bit3}, {tid}, 1;");
                b.L($"setp.eq.and.u32 {even}, {bit3}, 0, {act};");
                b.L($"and.b32 {lane32}, {tid}, 31;");
                b.L($"shr.u32 {warp}, {tid}, 5;");
                ClearBuffers("B1");
                string t2 = b.R(), pDone = b.P(), tLast = b.R(), xv = lane.NewReg(b), lv = lane.NewReg(b), av = lane.NewReg(b);
                b.L($"sub.u32 {tLast}, {tb}, 1;");
                b.L($"mov.u32 {t2}, {tLast};");
                b.L($"mov.{op} {xv}, {XAt(t2)};");
                b.L($"mov.{op} {lv}, {LseAt(t2)};");
                b.L($"mov.{op} {av}, {LdG(alphaG, AlphaAt(t2))};");
                b.Label("B1_T");
                b.L($"setp.lt.s32 {pDone}, {t2}, 0;");
                b.L($"@{pDone} bra DONE;");
                {
                    string t = t2, row = RowBase(t), e = Sub(xv, lv), avT = lane.NewReg(b), tp1 = b.R();
                    b.L($"mov.{op} {avT}, {av};");
                    b.L($"add.u32 {tp1}, {t}, 1;");
                    string cur = BufBase(t), prev = BufBase(tp1), isLast = b.P();
                    b.L($"setp.eq.u32 {isLast}, {t}, {tLast};");
                    string p = LdS(prev, sc), q = LdS(prev, sc1), r = LdS(prev, sc2);
                    b.L($"selp.{op} {r}, {negInf}, {r}, {skipB};");
                    string v = Lse3(p, q, r);
                    b.L(lane.Add(v, v, e));
                    string v0 = lane.NewReg(b), pHigh = b.P(), s2 = b.R();
                    b.L($"add.u32 {s2}, {tid}, 2;");
                    b.L($"setp.ge.u32 {pHigh}, {s2}, {S};");
                    b.L($"selp.{op} {v0}, {e}, {negInf}, {pHigh};");
                    b.L($"selp.{op} {v}, {v0}, {v}, {isLast};");
                    string ca = b.R();
                    b.L($"mad.lo.u32 {ca}, {sc}, {size}, {cur};");
                    b.L($"@{act} st.shared.{mem} [{ca}], {v};");
                    string g = lane.NewReg(b);
                    b.L(lane.Add(g, avT, v));
                    b.L($"sub.rn.{op} {g}, {g}, {e};");
                    b.L($"sub.rn.{op} {g}, {g}, {logP};");
                    string eg = Codec.Exp(b, lane, g), gAddr = b.R();
                    b.L($"mad.lo.u32 {gAddr}, {sc}, {size}, {sm};");
                    b.L($"@{act} st.shared.{mem} [{gAddr}+{gOff}], {eg};");
                    // The blank's terms (the even states): each warp's sum by shuffles, its lane 0 into shared memory.
                    string part = lane.NewReg(b);
                    b.L($"selp.{op} {part}, {eg}, {zero}, {even};");
                    for (int off = 16; off >= 1; off >>= 1)
                    {
                        string other = lane.NewReg(b);
                        if (lane.Wide)
                        {
                            string l0 = b.R(), h0 = b.R(), l1 = b.R(), h1 = b.R();
                            b.L($"mov.b64 {{{l0}, {h0}}}, {part};");
                            b.L($"shfl.sync.bfly.b32 {l1}, {l0}, {off}, 0x1f, 0xffffffff;");
                            b.L($"shfl.sync.bfly.b32 {h1}, {h0}, {off}, 0x1f, 0xffffffff;");
                            b.L($"mov.b64 {other}, {{{l1}, {h1}}};");
                        }
                        else
                        {
                            string bits = b.R(), bits2 = b.R();
                            b.L($"mov.b32 {bits}, {part};");
                            b.L($"shfl.sync.bfly.b32 {bits2}, {bits}, {off}, 0x1f, 0xffffffff;");
                            b.L($"mov.b32 {other}, {bits2};");
                        }

                        b.L(lane.Add(part, part, other));
                    }

                    string wa = b.R(), pL0 = b.P();
                    b.L($"mad.lo.u32 {wa}, {warp}, {size}, {sm};");
                    b.L($"setp.eq.u32 {pL0}, {lane32}, 0;");
                    b.L($"@{pL0} st.shared.{mem} [{wa}+{wOff}], {part};");
                    // The previous frame's logit, log-sum-exp and α (frame 0's again past the start), loaded across the barrier.
                    string tn = b.R();
                    b.L($"sub.u32 {tn}, {t}, 1;");
                    b.L($"max.s32 {tn}, {tn}, 0;");
                    b.L($"mov.{op} {xv}, {XAt(tn)};");
                    b.L($"mov.{op} {lv}, {LseAt(tn)};");
                    b.L($"mov.{op} {av}, {LdG(alphaG, AlphaAt(tn))};");
                    b.L("bar.sync 0;");
                    // Each class's first state: y − Σ of its class's terms. The blank (state 0): the warps' sums in order;
                    // a label: its own term and its links' in state order.
                    string sum = lane.NewReg(b), pBlank = b.P();
                    b.L($"@!{rep} bra B1_CLS_END;");
                    b.L($"setp.eq.u32 {pBlank}, {tid}, 0;");
                    b.L($"@!{pBlank} bra B1_LABEL;");
                    b.L($"ld.shared.{mem} {sum}, [{sm}+{wOff}];");
                    for (int w = 1; w < CtcThreads / 32; w++)
                    {
                        string pw = lane.NewReg(b);
                        b.L($"ld.shared.{mem} {pw}, [{sm}+{wOff + w * size}];");
                        b.L(lane.Add(sum, sum, pw));
                    }

                    b.L("bra B1_Y;");
                    b.Label("B1_LABEL");
                    {
                        string k = b.R(), pK = b.P(), ka = b.R(), gv = lane.NewReg(b);
                        b.L($"mov.{op} {sum}, {eg};");
                        b.L($"mad.lo.u32 {ka}, {sc}, 4, {sm};");
                        b.L($"ld.shared.u32 {k}, [{ka}+{lOff}];");
                        b.Label("B1_K");
                        b.L($"setp.eq.u32 {pK}, {k}, 0;");
                        b.L($"@{pK} bra B1_Y;");
                        b.L($"mad.lo.u32 {ka}, {k}, {size}, {sm};");
                        b.L($"ld.shared.{mem} {gv}, [{ka}+{gOff}];");
                        b.L(lane.Add(sum, sum, gv));
                        b.L($"mad.lo.u32 {ka}, {k}, 4, {sm};");
                        b.L($"ld.shared.u32 {k}, [{ka}+{lOff}];");
                        b.L("bra B1_K;");
                    }

                    b.Label("B1_Y");
                    {
                        string y = Codec.Exp(b, lane, e), ci = b.R();
                        b.L($"sub.rn.{op} {y}, {y}, {sum};");
                        b.L($"add.u32 {ci}, {row}, {myCls};");
                        b.L($"st.global.{mem} [{Addr(grad, ci, size)}], {y};");
                    }

                    b.Label("B1_CLS_END");
                    b.L("bar.sync 0;");
                    b.L($"sub.u32 {t2}, {t2}, 1;");
                    b.L("bra B1_T;");
                }
            }

            b.Label("DONE");
            b.L("ret;");
            string pa1 = ".param .u64 p_x, .param .u64 p_lse, .param .u64 p_alpha, .param .u64 p_cls, .param .u64 p_flags, .param .u64 p_ilen, .param .u64 p_slen, .param .u64 p_loss, .param .u64 p_grad, .param .u32 p_t, .param .u32 p_b, .param .u32 p_c, .param .u32 p_smax";
            return b.Finish(Header(gpu), entry, pa1, CtcOneShared(lane, cap));
        }

        // 2. α forward.
        ClearBuffers("A");
        {
            string t = b.R(), pT = b.P();
            b.L($"mov.u32 {t}, 0;");
            b.Label("ALPHA_T");
            b.L($"setp.ge.u32 {pT}, {t}, {tb};");
            b.L($"@{pT} bra ALPHA_END;");
            string row = RowBase(t), lseT = LseAt(t), tm1 = b.R();
            b.L($"sub.u32 {tm1}, {t}, 1;");
            string cur = BufBase(t), prev = BufBase(tm1), first = b.P();
            b.L($"setp.eq.u32 {first}, {t}, 0;");
            string s = b.R(), pS = b.P();
            b.L($"mov.u32 {s}, {tid};");
            b.Label("ALPHA_S");
            b.L($"setp.ge.u32 {pS}, {s}, {S};");
            b.L($"@{pS} bra ALPHA_S_END;");
            string e = Emit(row, lseT, s), s1 = b.R(), s2 = b.R(), fi = b.R(), fl = b.R(), pSkip = b.P();
            b.L($"add.u32 {s1}, {s}, 1;");
            b.L($"add.u32 {s2}, {s}, 2;");
            string p = LdS(prev, s2), q = LdS(prev, s1), r = LdS(prev, s);
            b.L($"add.u32 {fi}, {flagRow}, {s};");
            b.L($"ld.global.u32 {fl}, [{Addr(flagsG, fi, 4)}];");
            b.L($"and.b32 {fl}, {fl}, 1;");
            b.L($"setp.eq.u32 {pSkip}, {fl}, 0;");
            b.L($"selp.{op} {r}, {negInf}, {r}, {pSkip};");
            string v = Lse3(p, q, r);
            b.L(lane.Add(v, v, e));
            // The first frame: state 0 and 1 take their emissions, the rest −∞.
            string v0 = lane.NewReg(b), pLow = b.P();
            b.L($"setp.lt.u32 {pLow}, {s}, 2;");
            b.L($"selp.{op} {v0}, {e}, {negInf}, {pLow};");
            b.L($"selp.{op} {v}, {v0}, {v}, {first};");
            StS(cur, s2, v);
            string ai = b.R();
            b.L($"add.u32 {ai}, {alphaRow}, {t};");
            b.L($"mad.lo.u32 {ai}, {ai}, {smax}, {s};");
            b.L($"st.global.{mem} [{Addr(alphaG, ai, size)}], {v};");
            b.L($"add.u32 {s}, {s}, {CtcThreads};");
            b.L("bra ALPHA_S;");
            b.Label("ALPHA_S_END");
            b.L("bar.sync 0;");
            b.L($"add.u32 {t}, {t}, 1;");
            b.L("bra ALPHA_T;");
            b.Label("ALPHA_END");

            // log P from the last frame's last two states (index S + 1 and S: the pad when S = 1); the loss.
            string p0 = b.P();
            b.L($"setp.ne.u32 {p0}, {tid}, 0;");
            b.L($"@{p0} bra LOSS_END;");
            string tl = b.R();
            b.L($"sub.u32 {tl}, {tb}, 1;");
            string last = BufBase(tl), sp1 = b.R();
            b.L($"add.u32 {sp1}, {S}, 1;");
            string a1 = LdS(last, sp1), a2 = LdS(last, S);
            string logP = Lse3(a1, a2, negInf), nl = lane.NewReg(b);
            b.L($"st.shared.{mem} [{sm}+{pOff}], {logP};");
            b.L($"neg.{op} {nl}, {logP};");
            b.L($"st.global.{mem} [{Addr(loss, seq, size)}], {nl};");
            b.Label("LOSS_END");
            b.L("bar.sync 0;");
        }

        if (backward)
        {
            string logP = lane.NewReg(b), none = b.P();
            b.L($"ld.shared.{mem} {logP}, [{sm}+{pOff}];");
            b.L($"setp.eq.{op} {none}, {logP}, {negInf};");
            // No alignment: the sequence's frames' rows 0 too (the rows past its frames are the rows kernel's).
            {
                string t = b.R(), pT = b.P(), end = b.R();
                b.L($"mov.u32 {t}, 0;");
                b.L($"selp.u32 {end}, {tb}, 0, {none};");
                b.Label("ZERO_T");
                b.L($"setp.ge.u32 {pT}, {t}, {end};");
                b.L($"@{pT} bra ZERO_END;");
                string row = RowBase(t), c = b.R(), pC = b.P(), i = b.R();
                b.L($"mov.u32 {c}, {tid};");
                b.Label("ZERO_C");
                b.L($"setp.ge.u32 {pC}, {c}, {C};");
                b.L($"@{pC} bra ZERO_C_END;");
                b.L($"add.u32 {i}, {row}, {c};");
                b.L($"st.global.{mem} [{Addr(grad, i, size)}], {zero};");
                b.L($"add.u32 {c}, {c}, {CtcThreads};");
                b.L("bra ZERO_C;");
                b.Label("ZERO_C_END");
                b.L($"add.u32 {t}, {t}, 1;");
                b.L("bra ZERO_T;");
                b.Label("ZERO_END");
            }

            b.L($"@{none} bra DONE;");
            ClearBuffers("B");
            string t2 = b.R(), pDone = b.P(), tLast = b.R();
            b.L($"sub.u32 {tLast}, {tb}, 1;");
            b.L($"mov.u32 {t2}, {tLast};");
            b.Label("BETA_T");
            b.L($"setp.lt.s32 {pDone}, {t2}, 0;");
            b.L($"@{pDone} bra DONE;");
            {
                string t = t2, row = RowBase(t), lseT = LseAt(t), tp1 = b.R();
                b.L($"add.u32 {tp1}, {t}, 1;");
                string cur = BufBase(t), prev = BufBase(tp1), isLast = b.P();
                b.L($"setp.eq.u32 {isLast}, {t}, {tLast};");
                // β_t over the states; each state's term into shared memory.
                string s = b.R(), pS = b.P();
                b.L($"mov.u32 {s}, {tid};");
                b.Label("BETA_S");
                b.L($"setp.ge.u32 {pS}, {s}, {S};");
                b.L($"@{pS} bra BETA_S_END;");
                string e = Emit(row, lseT, s), s1 = b.R(), s2 = b.R(), fi = b.R(), fl = b.R(), pSkip = b.P();
                b.L($"add.u32 {s1}, {s}, 1;");
                b.L($"add.u32 {s2}, {s}, 2;");
                string p = LdS(prev, s), q = LdS(prev, s1), r = LdS(prev, s2);
                b.L($"add.u32 {fi}, {flagRow}, {s2};");
                b.L($"ld.global.u32 {fl}, [{Addr(flagsG, fi, 4)}];");
                b.L($"and.b32 {fl}, {fl}, 1;");
                b.L($"setp.eq.u32 {pSkip}, {fl}, 0;");
                b.L($"selp.{op} {r}, {negInf}, {r}, {pSkip};");
                string v = Lse3(p, q, r);
                b.L(lane.Add(v, v, e));
                string v0 = lane.NewReg(b), pHigh = b.P();
                b.L($"setp.ge.u32 {pHigh}, {s2}, {S};");
                b.L($"selp.{op} {v0}, {e}, {negInf}, {pHigh};");
                b.L($"selp.{op} {v}, {v0}, {v}, {isLast};");
                StS(cur, s, v);
                string ai = b.R();
                b.L($"add.u32 {ai}, {alphaRow}, {t};");
                b.L($"mad.lo.u32 {ai}, {ai}, {smax}, {s};");
                string av = LdG(alphaG, ai), g = lane.NewReg(b);
                b.L(lane.Add(g, av, v));
                b.L($"sub.rn.{op} {g}, {g}, {e};");
                b.L($"sub.rn.{op} {g}, {g}, {logP};");
                string eg = Codec.Exp(b, lane, g), gAddr = b.R();
                b.L($"mad.lo.u32 {gAddr}, {s}, {size}, {sm};");
                b.L($"st.shared.{mem} [{gAddr}+{gOff}], {eg};");
                b.L($"add.u32 {s}, {s}, {CtcThreads};");
                b.L("bra BETA_S;");
                b.Label("BETA_S_END");
                b.L("bar.sync 0;");
                // each class's first state: y − Σ of its class's terms, in state order.
                string s3 = b.R(), pS3 = b.P();
                b.L($"mov.u32 {s3}, {tid};");
                b.Label("CLS_S");
                b.L($"setp.ge.u32 {pS3}, {s3}, {S};");
                b.L($"@{pS3} bra CLS_S_END;");
                string fi3 = b.R(), fl3 = b.R(), pRep = b.P();
                b.L($"add.u32 {fi3}, {flagRow}, {s3};");
                b.L($"ld.global.u32 {fl3}, [{Addr(flagsG, fi3, 4)}];");
                b.L($"and.b32 {fl3}, {fl3}, 2;");
                b.L($"setp.eq.u32 {pRep}, {fl3}, 0;");
                b.L($"@{pRep} bra CLS_NEXT;");
                {
                    string myc = LdU(clsG, fi3), sum = lane.NewReg(b), k = b.R(), pK = b.P(), fk = b.R(), pSame = b.P(), ga = b.R(), gv = lane.NewReg(b);
                    b.L($"mov.{op} {sum}, {zero};");
                    b.L($"mov.u32 {k}, {s3};");
                    b.Label("SUM_K");
                    b.L($"setp.ge.u32 {pK}, {k}, {S};");
                    b.L($"@{pK} bra SUM_K_END;");
                    b.L($"add.u32 {fk}, {flagRow}, {k};");
                    string ck = LdU(clsG, fk);
                    b.L($"setp.ne.u32 {pSame}, {ck}, {myc};");
                    b.L($"@{pSame} bra SUM_K_NEXT;");
                    b.L($"mad.lo.u32 {ga}, {k}, {size}, {sm};");
                    b.L($"ld.shared.{mem} {gv}, [{ga}+{gOff}];");
                    b.L(lane.Add(sum, sum, gv));
                    b.Label("SUM_K_NEXT");
                    b.L($"add.u32 {k}, {k}, 2;");
                    b.L("bra SUM_K;");
                    b.Label("SUM_K_END");
                    string ci = b.R();
                    b.L($"add.u32 {ci}, {row}, {myc};");
                    string y = Codec.Exp(b, lane, Sub(LdG(x, ci), lseT));
                    b.L($"sub.rn.{op} {y}, {y}, {sum};");
                    b.L($"st.global.{mem} [{Addr(grad, ci, size)}], {y};");
                }

                b.Label("CLS_NEXT");
                b.L($"add.u32 {s3}, {s3}, {CtcThreads};");
                b.L("bra CLS_S;");
                b.Label("CLS_S_END");
                b.L("bar.sync 0;");
                b.L($"sub.u32 {t2}, {t2}, 1;");
                b.L("bra BETA_T;");
            }
        }

        b.Label("DONE");
        b.L("ret;");
        string pa = ".param .u64 p_x, .param .u64 p_lse, .param .u64 p_alpha, .param .u64 p_cls, .param .u64 p_flags, .param .u64 p_ilen, .param .u64 p_slen, .param .u64 p_loss, .param .u64 p_grad, .param .u32 p_t, .param .u32 p_b, .param .u32 p_c, .param .u32 p_smax";
        return b.Finish(Header(gpu), entry, pa, CtcShared(lane, cap));
    }
}
