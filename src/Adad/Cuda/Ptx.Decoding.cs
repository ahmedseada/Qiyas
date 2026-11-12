// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's steps on CUDA (Decoding.cs): the ring's push (a thread a row); the penalties in two kernels — each row's
// window ordered into a table of P slots (a thread a slot: its token's rank among the window's by (token, slot), its
// occurrences; the slots past the window the largest integer, so the table is ascending) with a mask of 16·P bits after it
// (bit token mod 16·P set for each token of the window; a thread a word of it), then the store (StoreEach: a thread a
// stored word, a warp or CTA a block): a score whose column's bit is clear is x's (one load: most of a row); the others'
// columns sought in the table by a fixed number of halvings (log2 P loads), the score penalized where it is found; every
// value rounded to the type once.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>The most tokens a window holds on the device (the window's kernel reads the window once a slot).</summary>
    public const int MaxPenaltyWindow = 4095;

    /// <summary>The slots of a row's table: the power of two above the window's most tokens (a slot past the window always).</summary>
    public static int PenaltySlots(int maxWindow)
    {
        if (maxWindow > MaxPenaltyWindow) throw new NotSupportedException($"penalties: a window of {maxWindow} tokens exceeds the kernel's {MaxPenaltyWindow}");
        int p = 2;
        while (p <= maxWindow) p *= 2;
        return p;
    }

    /// <summary>history[r·cap + length[0] mod cap] = ids[r], a thread a row. Params: ids, history, length, rows, cap.</summary>
    public static string HistoryPush(Gpu gpu, out string entry)
    {
        entry = "history_push";
        var b = new PtxBuilder();
        string r = GlobalIndex(b);
        string ids = Param64(b, "p_ids"), hist = Param64(b, "p_hist"), len = Param64(b, "p_len"), rows = Param32(b, "p_rows"), cap = Param32(b, "p_cap");
        string pEnd = b.P(), l = b.R(), slot = b.R(), at = b.R(), ia = b.RD(), ha = b.RD(), id = b.R();
        b.L($"setp.ge.u32 {pEnd}, {r}, {rows};");
        b.L($"@{pEnd} bra DONE;");
        b.L($"ld.global.u32 {l}, [{len}];");
        b.L($"rem.u32 {slot}, {l}, {cap};");
        b.L($"mad.lo.u32 {at}, {r}, {cap}, {slot};");
        b.L($"mul.wide.u32 {ia}, {r}, 4;");
        b.L($"add.u64 {ia}, {ids}, {ia};");
        b.L($"ld.global.u32 {id}, [{ia}];");
        b.L($"mul.wide.u32 {ha}, {at}, 4;");
        b.L($"add.u64 {ha}, {hist}, {ha};");
        b.L($"st.global.u32 [{ha}], {id};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_ids, .param .u64 p_hist, .param .u64 p_len, .param .u32 p_rows, .param .u32 p_cap");
    }

    /// <summary>The words a row's table takes: `slots` tokens, `slots` occurrences, slots / 2 words of the mask (4·slots a row).</summary>
    public static int PenaltyRowWords(int slots) => 4 * slots;

    /// <summary>Each row's window as a table of `slots` tokens ascending and their occurrences after them (win[r·4·slots + i],
    /// win[r·4·slots + slots + i], s32): a thread a slot j — its token the ring's at length − 1 − j (j below the window's n =
    /// min(max(length[0], 0), nmax)), its place the tokens before it by (token, slot); a slot past the window the largest
    /// integer at its own place. Then the mask (win[r·4·slots + 2·slots + w], w below slots / 2): thread w's word, bit
    /// (token mod 16·slots) of each token of the window. Params: history, length, win, rows, cap, nmax.</summary>
    public static string PenaltyWindow(Gpu gpu, int slots, out string entry)
    {
        entry = $"penalty_window_{slots}";
        int shift = int.Log2(slots);
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string hist = Param64(b, "p_hist"), len = Param64(b, "p_len"), win = Param64(b, "p_win"), rows = Param32(b, "p_rows"), cap = Param32(b, "p_cap"), nmax = Param32(b, "p_nmax");
        string r = b.R(), j = b.R(), pEnd = b.P(), l = b.R(), n = b.R(), rowBase = b.R(), last = b.R();
        b.L($"shr.u32 {r}, {gid}, {shift};");
        b.L($"and.b32 {j}, {gid}, {slots - 1};");
        b.L($"setp.ge.u32 {pEnd}, {r}, {rows};");
        b.L($"@{pEnd} bra DONE;");
        b.L($"ld.global.s32 {l}, [{len}];");
        b.L($"max.s32 {l}, {l}, 0;");
        b.L($"min.u32 {n}, {l}, {nmax};");
        b.L($"mul.lo.u32 {rowBase}, {r}, {cap};");
        b.L($"sub.u32 {last}, {l}, 1;");                                          // the newest token's count: its slot is last mod cap
        // The ring's token of window slot q (below n) into `into`.
        void Token(string q, string into)
        {
            string back = b.R(), slot = b.R(), at = b.R(), a = b.RD();
            b.L($"sub.u32 {back}, {last}, {q};");
            b.L($"rem.u32 {slot}, {back}, {cap};");
            b.L($"add.u32 {at}, {rowBase}, {slot};");
            b.L($"mul.wide.u32 {a}, {at}, 4;");
            b.L($"add.u64 {a}, {hist}, {a};");
            b.L($"ld.global.s32 {into}, [{a}];");
        }

        string mine = b.R(), rank = b.R(), count = b.R(), pIn = b.P();
        b.L($"setp.lt.u32 {pIn}, {j}, {n};");
        b.L($"mov.u32 {mine}, 0x7FFFFFFF;");
        b.L($"mov.u32 {rank}, {j};");                                             // a slot past the window: its own place
        b.L($"mov.u32 {count}, 0;");
        b.L($"@!{pIn} bra STORE;");
        Token(j, mine);
        b.L($"mov.u32 {rank}, 0;");
        {
            string q = b.R(), pDone = b.P(), other = b.R(), pLess = b.P(), pSame = b.P(), pBefore = b.P(), one = b.R();
            b.L($"mov.u32 {q}, 0;");
            b.Label("SCAN");
            b.L($"setp.ge.u32 {pDone}, {q}, {n};");
            b.L($"@{pDone} bra STORE;");
            Token(q, other);
            b.L($"setp.lt.s32 {pLess}, {other}, {mine};");
            b.L($"setp.eq.s32 {pSame}, {other}, {mine};");
            b.L($"setp.lt.u32 {pBefore}, {q}, {j};");
            b.L($"and.pred {pBefore}, {pBefore}, {pSame};");
            b.L($"or.pred {pLess}, {pLess}, {pBefore};");
            b.L($"selp.u32 {one}, 1, 0, {pLess};");
            b.L($"add.u32 {rank}, {rank}, {one};");
            b.L($"selp.u32 {one}, 1, 0, {pSame};");
            b.L($"add.u32 {count}, {count}, {one};");
            b.L($"add.u32 {q}, {q}, 1;");
            b.L("bra SCAN;");
        }

        b.Label("STORE");
        string place = b.R(), ta = b.RD();
        b.L($"shl.b32 {place}, {r}, {shift + 2};");
        b.L($"add.u32 {place}, {place}, {rank};");
        b.L($"mul.wide.u32 {ta}, {place}, 4;");
        b.L($"add.u64 {ta}, {win}, {ta};");
        b.L($"st.global.s32 [{ta}], {mine};");
        b.L($"st.global.s32 [{ta}+{slots * 4}], {count};");
        // The mask's word j (j below slots / 2): the bits of the window's tokens that fall in it.
        {
            string pMask = b.P(), word = b.R(), q = b.R(), pDone = b.P(), other = b.R(), hb = b.R(), w = b.R(), pHere = b.P(), bit = b.R(), ma = b.RD(), at = b.R();
            b.L($"setp.ge.u32 {pMask}, {j}, {slots / 2};");
            b.L($"@{pMask} bra DONE;");
            b.L($"mov.u32 {word}, 0;");
            b.L($"mov.u32 {q}, 0;");
            b.Label("MASK");
            b.L($"setp.ge.u32 {pDone}, {q}, {n};");
            b.L($"@{pDone} bra MASK_END;");
            Token(q, other);
            b.L($"and.b32 {hb}, {other}, {16 * slots - 1};");
            b.L($"shr.u32 {w}, {hb}, 5;");
            b.L($"setp.eq.u32 {pHere}, {w}, {j};");
            b.L($"and.b32 {hb}, {hb}, 31;");
            b.L($"shl.b32 {bit}, 1, {hb};");
            b.L($"selp.u32 {bit}, {bit}, 0, {pHere};");
            b.L($"or.b32 {word}, {word}, {bit};");
            b.L($"add.u32 {q}, {q}, 1;");
            b.L("bra MASK;");
            b.Label("MASK_END");
            b.L($"shl.b32 {at}, {r}, {shift + 2};");
            b.L($"add.u32 {at}, {at}, {j};");
            b.L($"mul.wide.u32 {ma}, {at}, 4;");
            b.L($"add.u64 {ma}, {win}, {ma};");
            b.L($"st.global.u32 [{ma}+{2 * slots * 4}], {word};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_hist, .param .u64 p_len, .param .u64 p_win, .param .u32 p_rows, .param .u32 p_cap, .param .u32 p_nmax");
    }

    /// <summary>y (n = rows · cols values of f) = the rows' scores (row r of x at r·stride + off) with the tokens of each row's
    /// table (PenaltyWindow's, `slots` a row) penalized: (x above 0 ? x / rep : x · rep) − (pres + freq · occurrences) in the
    /// float lane, taken back to f's natural lane by its conversion; the others x's value, moved in the natural lane exactly. Params: x, win, y, n, cols, stride, off, rep, pres, freq (lane). sparse (contiguous rows
    /// of whole stored words, not a block format, y already x's copy): a thread a slot of a row's table (n the rows) stores
    /// the word of y that holds its token, the first slot of the ascending table with that word alone (one writer a word);
    /// every other word is left as the copy made it.</summary>
    public static string PenalizeRows(Gpu gpu, Fmt f, Lane lane, int slots, bool sparse, out string entry)
    {
        entry = $"penalize_rows_{f.Name}_{lane.Name}_{slots}{(sparse ? "_sparse" : "")}";
        int shift = int.Log2(slots);
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), win = Param64(b, "p_win"), py = Param64(b, "p_y");
        string n = Param32(b, "p_n"), cols = Param32(b, "p_cols"), stride = Param32(b, "p_stride"), off = Param32(b, "p_off");
        string rep = lane.NewReg(b), pres = lane.NewReg(b), freq = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {rep}, [p_rep];");
        b.L($"ld.param.{lane.Op} {pres}, [p_pres];");
        b.L($"ld.param.{lane.Op} {freq}, [p_freq];");
        var natural = f.Natural;
        // The value of column c of row r, in f's natural lane.
        string Value(string r, string c)
        {
            string src = b.R(), table = b.R(), pos = b.R(), y = natural.NewReg(b);
            string skip = $"PEN{b.P().TrimStart('%')}";
            b.L($"mad.lo.u32 {src}, {r}, {stride}, {off};");
            b.L($"add.u32 {src}, {src}, {c};");
            string xn = Codec.UnpackAt(b, f, px, src, natural);
            b.L($"mov.{natural.Reg} {y}, {xn};");
            b.L($"shl.b32 {table}, {r}, {shift + 2};");
            // The column's bit of the row's mask: clear, the column is not in the window (the score is x's).
            {
                string hb = b.R(), mw = b.R(), ma = b.RD(), word = b.R(), pClear = b.P();
                b.L($"and.b32 {hb}, {c}, {16 * slots - 1};");
                b.L($"shr.u32 {mw}, {hb}, 5;");
                b.L($"add.u32 {mw}, {mw}, {table};");
                b.L($"mul.wide.u32 {ma}, {mw}, 4;");
                b.L($"add.u64 {ma}, {win}, {ma};");
                b.L($"ld.global.u32 {word}, [{ma}+{2 * slots * 4}];");
                b.L($"and.b32 {hb}, {hb}, 31;");
                b.L($"shr.u32 {word}, {word}, {hb};");
                b.L($"and.b32 {word}, {word}, 1;");
                b.L($"setp.eq.u32 {pClear}, {word}, 0;");
                b.L($"@{pClear} bra {skip};");
            }

            // The entries of the row's table at or below the column: halvings over an ascending table whose last slot is past
            // every column.
            b.L($"mov.u32 {pos}, 0;");
            for (int step = slots / 2; step >= 1; step /= 2)
            {
                string idx = b.R(), a = b.RD(), t = b.R(), le = b.P(), next = b.R();
                b.L($"add.u32 {idx}, {table}, {pos};");
                b.L($"mul.wide.u32 {a}, {idx}, 4;");
                b.L($"add.u64 {a}, {win}, {a};");
                b.L($"ld.global.s32 {t}, [{a}+{(step - 1) * 4}];");
                b.L($"setp.le.s32 {le}, {t}, {c};");
                b.L($"add.u32 {next}, {pos}, {step};");
                b.L($"selp.u32 {pos}, {next}, {pos}, {le};");
            }

            // The entry before that place is the column's when the column is in the window.
            string any = b.P(), at1 = b.R(), ea = b.RD(), token = b.R(), count = b.R(), hit = b.P();
            b.L($"setp.ne.u32 {any}, {pos}, 0;");
            b.L($"max.u32 {at1}, {pos}, 1;");
            b.L($"sub.u32 {at1}, {at1}, 1;");
            b.L($"add.u32 {at1}, {at1}, {table};");
            b.L($"mul.wide.u32 {ea}, {at1}, 4;");
            b.L($"add.u64 {ea}, {win}, {ea};");
            b.L($"ld.global.s32 {token}, [{ea}];");
            b.L($"ld.global.s32 {count}, [{ea}+{slots * 4}];");
            b.L($"setp.eq.s32 {hit}, {token}, {c};");
            b.L($"and.pred {hit}, {hit}, {any};");
            string times = lane.NewReg(b), positive = b.P(), scaled = lane.NewReg(b), x = Codec.Convert(b, natural, xn, lane, f);
            b.L($"cvt.rn.{lane.Op}.s32 {times}, {count};");
            b.L($"setp.gt.{lane.Op} {positive}, {x}, {lane.Zero};");
            string down = LaneOp(b, lane, "div", x, rep), up = LaneOp(b, lane, "mul", x, rep);
            b.L($"selp.{lane.Op} {scaled}, {down}, {up}, {positive};");
            string less = lane.NewReg(b);
            b.L(lane.Fma(less, freq, times, pres));
            string penalized = Codec.Convert(b, lane, LaneOp(b, lane, "sub", scaled, less), natural, f);
            b.L($"selp.{natural.Op} {y}, {penalized}, {xn}, {hit};");
            b.Label(skip);
            return y;
        }

        if (sparse)
        {
            // This thread's slot of its row's table: its token's word of the row, stored when no slot before it holds a token
            // of the same word (the table ascends: such a slot is the one before).
            int perShift = int.Log2(f.Per);
            string gid = GlobalIndex(b), row = b.R(), slot = b.R(), pEnd = b.P(), table = b.R(), token = b.R(), before = b.R(), ta = b.RD(), ba = b.RD(), at0 = b.R(), bi = b.R();
            string wx = b.R(), wb = b.R(), pOut = b.P(), pFirst = b.P(), pBeforeIn = b.P(), pSame = b.P(), words = b.R();
            string code = f.WordBits == 64 ? b.RD() : b.R(), at = b.R(), addr = b.RD();
            b.L($"shr.u32 {row}, {gid}, {shift};");
            b.L($"and.b32 {slot}, {gid}, {slots - 1};");
            b.L($"setp.ge.u32 {pEnd}, {row}, {n};");
            b.L($"@{pEnd} bra DONE;");
            b.L($"shl.b32 {table}, {row}, {shift + 2};");
            b.L($"add.u32 {at0}, {table}, {slot};");
            b.L($"mul.wide.u32 {ta}, {at0}, 4;");
            b.L($"add.u64 {ta}, {win}, {ta};");
            b.L($"ld.global.u32 {token}, [{ta}];");
            b.L($"setp.ge.u32 {pOut}, {token}, {cols};");                      // below 0 (past it unsigned), past the row, or no token
            b.L($"@{pOut} bra DONE;");
            b.L($"shr.u32 {wx}, {token}, {perShift};");
            b.L($"setp.eq.u32 {pFirst}, {slot}, 0;");
            b.L($"max.u32 {bi}, {slot}, 1;");
            b.L($"sub.u32 {bi}, {bi}, 1;");
            b.L($"add.u32 {bi}, {bi}, {table};");
            b.L($"mul.wide.u32 {ba}, {bi}, 4;");
            b.L($"add.u64 {ba}, {win}, {ba};");
            b.L($"ld.global.u32 {before}, [{ba}];");
            b.L($"setp.lt.u32 {pBeforeIn}, {before}, {cols};");
            b.L($"shr.u32 {wb}, {before}, {perShift};");
            b.L($"setp.eq.u32 {pSame}, {wb}, {wx};");
            b.L($"and.pred {pSame}, {pSame}, {pBeforeIn};");
            b.L($"not.pred {pFirst}, {pFirst};");
            b.L($"and.pred {pSame}, {pSame}, {pFirst};");
            b.L($"@{pSame} bra DONE;");
            if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
            for (int i = 0; i < f.Per; i++)
            {
                string c = b.R();
                b.L($"mad.lo.u32 {c}, {wx}, {f.Per}, {i};");
                Codec.Pack(b, f, Value(row, c), natural, i, code);
            }

            b.L($"shr.u32 {words}, {cols}, {perShift};");                      // the words a row (cols a multiple of Per)
            b.L($"mad.lo.u32 {at}, {row}, {words}, {wx};");
            b.L($"mul.wide.u32 {addr}, {at}, {f.WordBits / 8};");
            b.L($"add.u64 {addr}, {py}, {addr};");
            b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        }
        else
        {
            StoreEach(b, f, natural, py, n, at =>
            {
                string r = b.R(), c = b.R();
                b.L($"div.u32 {r}, {at}, {cols};");
                b.L($"mul.lo.u32 {c}, {r}, {cols};");
                b.L($"sub.u32 {c}, {at}, {c};");
                return Value(r, c);
            });
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_win, .param .u64 p_y, .param .u32 p_n, .param .u32 p_cols, .param .u32 p_stride, .param .u32 p_off, .param .{lane.Op} p_rep, .param .{lane.Op} p_pres, .param .{lane.Op} p_freq",
            EachShared(f));
    }
}
