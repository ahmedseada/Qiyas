// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's block encoders (Blocks.cs), a warp per block: each lane holds BlockValues / 32 consecutive values of it (one
// for Q8_0 / Q4_0, eight for Q4_K / Q6_K) in registers, the block's statistics by butterflies over the warp (maxima and
// minima: exact in any order, so the bits of one thread scanning the block), the scales as the formats store them, then
// each lane quantizes its own values and the codes are gathered by shuffles into the formats' byte order.
//
// An encoder takes its values from a source (EncodeBlock): lane storage (LanesToBlock), or the op's own formula over its
// inputs, each decoded a value at a time (the kernels that write block formats themselves, so no lane array of the
// output exists). The source is evaluated once a value.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Lane storage into blocks of f: a warp per block.</summary>
    public static string LanesToBlock(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"lanes_to_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string x = Param64(b, "p_x"), outp = Param64(b, "p_out"), n = Param32(b, "p_n");
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, outp, w, n, at =>
        {
            string addr = b.RD(), raw = lane.Wide ? b.FD() : b.F();
            b.L($"mul.wide.u32 {addr}, {at}, {lane.Size};");
            b.L($"add.u64 {addr}, {x}, {addr};");
            b.L($"ld.global.{lane.Mem} {raw}, [{addr}];");
            if (!lane.Wide) return raw;
            string v = b.F();
            b.L($"cvt.rn.f32.f64 {v}, {raw};");
            return v;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }

    /// <summary>Threads of a warp-per-block kernel over count values of f.</summary>
    public static long BlockThreads(Fmt f, long count) => (count + f.BlockValues - 1) / f.BlockValues * 32;

    /// <summary>This warp's block of f over n values and the thread's lane in it (a warp per block); a warp past the last
    /// block goes to DONE whole (its lanes share the block), so every lane of a working warp takes part in the shuffles.</summary>
    internal static (string Blk, string Lane) BlockWarp(PtxBuilder b, Fmt f, string n)
    {
        string gid = GlobalIndex(b), blk = b.R(), lane = b.R(), blocks = b.R(), p = b.P();
        b.L($"shr.u32 {blk}, {gid}, 5;");
        b.L($"and.b32 {lane}, {gid}, 31;");
        b.L($"add.u32 {blocks}, {n}, {f.BlockValues - 1};");
        b.L($"div.u32 {blocks}, {blocks}, {f.BlockValues};");
        b.L($"setp.ge.u32 {p}, {blk}, {blocks};");
        b.L($"@{p} bra DONE;");
        return (blk, lane);
    }

    /// <summary>Block w.Blk of the array of f at outp (n values) encoded from value(at): at is a .u32 register holding the
    /// value's linear index (held below n), the result an f32 register; values past n are 0. Each lane evaluates the
    /// source for its own values, once each (straight-line code: no labels).</summary>
    internal static void EncodeBlock(PtxBuilder b, Fmt f, string outp, (string Blk, string Lane) w, string n, Func<string, string> value) =>
        EncodeBlock(b, f, outp, w, n, (at, _) => value(at));

    /// <summary>As above, the source also given the predicate of its value being below n (false: `at` is the last value
    /// again, so a source with side effects predicates them on it).</summary>
    internal static void EncodeBlock(PtxBuilder b, Fmt f, string outp, (string Blk, string Lane) w, string n, Func<string, string, string> value)
    {
        int per = f.BlockValues / 32;
        string first = b.R(), last = b.R(), bAddr = b.RD();
        b.L($"mul.lo.u32 {first}, {w.Blk}, {f.BlockValues};");
        b.L($"mad.lo.u32 {first}, {w.Lane}, {per}, {first};");
        b.L($"sub.u32 {last}, {n}, 1;");
        b.L($"mul.wide.u32 {bAddr}, {w.Blk}, {f.BlockBytes};");
        b.L($"add.u64 {bAddr}, {outp}, {bAddr};");
        var vals = new string[per];
        for (int k = 0; k < per; k++)
        {
            string idx = b.R(), pIn = b.P(), at = b.R();
            b.L($"add.u32 {idx}, {first}, {k};");
            b.L($"setp.le.u32 {pIn}, {idx}, {last};");
            b.L($"min.u32 {at}, {idx}, {last};");
            string raw = value(at, pIn);
            vals[k] = b.F();
            b.L($"selp.f32 {vals[k]}, {raw}, 0f00000000, {pIn};");
        }

        var e = new BlockEmit(b, bAddr, w.Lane);
        if (f.Is("q4_k")) EncodeQ4K(e, vals);
        else if (f.Is("q6_k")) EncodeQ6K(e, vals);
        else EncodeQ0(e, f, vals[0]);
    }

    /// <summary>Whether a kernel whose values are costly (a window or a gather each) takes a CTA per block of f: the formats of
    /// more values than a warp's lanes, whose warp per block would give each lane BlockValues / 32 of them in turn.</summary>
    public static bool BlockByCta(Fmt f) => f.BlockValues > 32;

    /// <summary>A CTA per block of f (BlockValues threads, the kernel's shared bytes BlockValues · 4): each thread evaluates
    /// one value of the block (value(at, pIn) as EncodeBlock's source) into shared memory, then the first warp encodes the
    /// block from them. The other warps go to DONE.</summary>
    internal static void EncodeBlockByCta(PtxBuilder b, Fmt f, string outp, string n, Func<string, string, string> value)
    {
        string blk = b.R(), t = b.R(), first = b.R(), idx = b.R(), last = b.R(), at = b.R(), pIn = b.P(), smem = b.R(), sAddr = b.R();
        b.L($"mov.u32 {blk}, %ctaid.x;");
        b.L($"mov.u32 {t}, %tid.x;");
        b.L($"mul.lo.u32 {first}, {blk}, {f.BlockValues};");
        b.L($"add.u32 {idx}, {first}, {t};");
        b.L($"sub.u32 {last}, {n}, 1;");
        b.L($"setp.le.u32 {pIn}, {idx}, {last};");
        b.L($"min.u32 {at}, {idx}, {last};");
        string raw = value(at, pIn), v = b.F();
        b.L($"selp.f32 {v}, {raw}, 0f00000000, {pIn};");
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mad.lo.u32 {sAddr}, {t}, 4, {smem};");
        b.L($"st.shared.f32 [{sAddr}], {v};");
        b.L("bar.sync 0;");
        string pRest = b.P();
        b.L($"setp.ge.u32 {pRest}, {t}, 32;");
        b.L($"@{pRest} bra DONE;");
        EncodeBlock(b, f, outp, (blk, t), n, a =>
        {
            string o = b.R(), ad = b.R(), r = b.F();
            b.L($"sub.u32 {o}, {a}, {first};");
            b.L($"mad.lo.u32 {ad}, {o}, 4, {smem};");
            b.L($"ld.shared.f32 {r}, [{ad}];");
            return r;
        });
    }

    // The encoders' instructions: butterflies over the warp, shuffles, rounding, stores at the block's bytes.
    private sealed class BlockEmit(PtxBuilder b, string bAddr, string lane)
    {
        public PtxBuilder B => b;
        public string Lane => lane;
        public string BAddr => bAddr;

        public string Const(string bits) { string d = b.F(); b.L($"mov.f32 {d}, {bits};"); return d; }
        public string F(string op, string x, string y) { string d = b.F(); b.L($"{op}.f32 {d}, {x}, {y};"); return d; }
        public string Abs(string x) { string d = b.F(); b.L($"abs.f32 {d}, {x};"); return d; }
        public string Neg(string x) { string d = b.F(); b.L($"neg.f32 {d}, {x};"); return d; }

        /// <summary>op ("max", "min") of x over the lanes whose index differs only in the bits of `mask` (a butterfly a bit).</summary>
        public string Butterfly(string op, string x, int mask)
        {
            for (int off = 16; off >= 1; off >>= 1)
            {
                if ((mask & off) == 0) continue;
                string o = b.F(), d = b.F();
                b.L($"shfl.sync.bfly.b32 {o}, {x}, {off}, 0x1f, 0xffffffff;");
                b.L($"{op}.f32 {d}, {x}, {o};");
                x = d;
            }

            return x;
        }

        public string Down(string x, int by) { string d = b.R(); b.L($"shfl.sync.down.b32 {d}, {x}, {by}, 0x1f, 0xffffffff;"); return d; }
        public string From(string x, int src) { string d = b.R(); b.L($"shfl.sync.idx.b32 {d}, {x}, {src}, 0x1f, 0xffffffff;"); return d; }

        public string Rni(string fv) { string r = b.R(); b.L($"cvt.rni.s32.f32 {r}, {fv};"); return r; }
        public string Clamp(string r, int lo, int hi) { b.L($"max.s32 {r}, {r}, {lo};"); b.L($"min.s32 {r}, {r}, {hi};"); return r; }
        public string I2F(string r) { string fv = b.F(); b.L($"cvt.rn.f32.s32 {fv}, {r};"); return fv; }
        public string HalfRound(string fv) { string h = b.H(), back = b.F(); b.L($"cvt.rn.f16.f32 {h}, {fv};"); b.L($"cvt.f32.f16 {back}, {h};"); return back; }
        public string IsZero(string fv) { string p = b.P(); b.L($"setp.eq.f32 {p}, {fv}, 0f00000000;"); return p; }
        public string Bits(string word, int at, int width) { string d = b.R(); b.L($"bfe.u32 {d}, {word}, {at}, {width};"); return d; }
        public void Put(string word, string bits, int at, int width) => b.L($"bfi.b32 {word}, {bits}, {word}, {at}, {width};");
        public string U(string op, string x, int y) { string d = b.R(); b.L($"{op} {d}, {x}, {y};"); return d; }

        /// <summary>A predicate: (lane &amp; mask) == value.</summary>
        public string LaneIs(int mask, int value)
        {
            string t = b.R(), p = b.P();
            b.L($"and.b32 {t}, {lane}, {mask};");
            b.L($"setp.eq.u32 {p}, {t}, {value};");
            return p;
        }

        public void StoreHalf(string fv, int at, string p) { string h = b.H(); b.L($"cvt.rn.f16.f32 {h}, {fv};"); b.L($"@{p} st.global.b16 [{bAddr}+{at}], {h};"); }

        /// <summary>The low byte of r at the block's byte offset + at (offset a .u32 register), where p (null: always).</summary>
        public void StoreByte(string r, string offset, int at, string? p)
        {
            string a = b.RD(), h = b.H();
            b.L($"cvt.u64.u32 {a}, {offset};");
            b.L($"add.u64 {a}, {bAddr}, {a};");
            b.L($"cvt.u16.u32 {h}, {r};");
            b.L($"{(p is null ? "" : $"@{p} ")}st.global.b8 [{a}+{at}], {h};");
        }
    }

    // Q8_0 / Q4_0: lane l holds value l. The block's max |x| (Q4_0: the signed value of the first largest |x|), the scale as
    // stored (f16) and read back, its inverse, each lane's integer; Q4_0's nibbles pair value i with value i + 16.
    private static void EncodeQ0(BlockEmit e, Fmt f, string v)
    {
        var b = e.B;
        string a = e.Abs(v), amax = e.Butterfly("max", e.F("max", a, "0f00000000"), 31);   // NaN as 0, as max drops it
        string maxv = v;
        if (f.Is("q4_0"))
        {
            // The first lane with |x| = the max (a max above 0); none: +0 (as a scan from 0 that takes only a larger |x|).
            string pick = b.R(), pM = b.P(), pPos = b.P(), pNone = b.P();
            b.L($"setp.eq.f32 {pM}, {a}, {amax};");
            b.L($"setp.gt.f32 {pPos}, {amax}, 0f00000000;");
            b.L($"and.pred {pM}, {pM}, {pPos};");
            b.L($"selp.u32 {pick}, {e.Lane}, 32, {pM};");
            for (int off = 16; off >= 1; off >>= 1)
            {
                string o = b.R(), m = b.R();
                b.L($"shfl.sync.bfly.b32 {o}, {pick}, {off}, 0x1f, 0xffffffff;");
                b.L($"min.u32 {m}, {pick}, {o};");
                pick = m;
            }

            string src = b.R(), got = b.F();
            b.L($"min.u32 {src}, {pick}, 31;");
            b.L($"shfl.sync.idx.b32 {got}, {v}, {src}, 0x1f, 0xffffffff;");
            b.L($"setp.eq.u32 {pNone}, {pick}, 32;");
            maxv = b.F();
            b.L($"selp.f32 {maxv}, 0f00000000, {got}, {pNone};");
        }

        string d = b.F(), h = b.H(), dBack = b.F(), id = b.F(), pZero = b.P();
        if (f.Is("q8_0")) b.L($"div.rn.f32 {d}, {amax}, 0f42FE0000;");               // / 127
        else b.L($"div.rn.f32 {d}, {maxv}, 0fC1000000;");                             // / −8
        b.L($"cvt.rn.f16.f32 {h}, {d};");
        b.L($"cvt.f32.f16 {dBack}, {h};");
        b.L($"setp.eq.f32 {pZero}, {dBack}, 0f00000000;");
        b.L($"div.rn.f32 {id}, 0f3F800000, {dBack};");
        b.L($"selp.f32 {id}, 0f00000000, {id}, {pZero};");
        b.L($"@{e.LaneIs(31, 0)} st.global.b16 [{e.BAddr}], {h};");
        if (f.Is("q8_0"))
        {
            string t = b.F();
            b.L($"mul.rn.f32 {t}, {v}, {id};");
            e.StoreByte(e.Clamp(e.Rni(t), -127, 127), e.Lane, 2, null);
            return;
        }

        string tl = b.F(), ql = b.R();
        b.L($"fma.rn.f32 {tl}, {v}, {id}, 0f41080000;");                              // + 8.5
        b.L($"cvt.rmi.s32.f32 {ql}, {tl};");                                            // floor
        e.Clamp(ql, 0, 15);
        string qh = e.Down(ql, 16), code = b.R(), low = b.P();
        b.L($"shl.b32 {code}, {qh}, 4;");
        b.L($"or.b32 {code}, {code}, {ql};");
        b.L($"setp.lt.u32 {low}, {e.Lane}, 16;");
        e.StoreByte(code, e.Lane, 2, low);
    }

    // Q4_K: lane l holds values 8l .. 8l + 7, so sub-block j (32 values) is lanes 4j .. 4j + 3. Sub-block statistics lo (≤ 0),
    // hi → scale_j = (hi − lo)/15, min_j = −lo; the block's largest scale and min → d, dmin (f16); ls_j, lm_j (6 bits);
    // q = clamp(rni((v + m_j)/d_j), 0, 15), d_j = d·ls_j, m_j = dmin·lm_j; pairs of sub-blocks share 32 bytes (the even
    // sub-block's value the low nibble, the odd's the high).
    private static void EncodeQ4K(BlockEmit e, string[] v)
    {
        var b = e.B;
        string lo = e.Const("0f7F7FFFFF"), hi = e.Const("0fFF7FFFFF");
        foreach (var x in v) { lo = e.F("min", lo, x); hi = e.F("max", hi, x); }
        lo = e.Butterfly("min", lo, 3); hi = e.Butterfly("max", hi, 3);
        lo = e.F("min", lo, "0f00000000");
        string scale = e.F("div.rn", e.F("sub.rn", hi, lo), "0f41700000");             // / 15
        string mn = e.Neg(lo);
        string maxScale = e.Butterfly("max", e.F("max", scale, "0f00000000"), 28);       // over the sub-blocks
        string maxMin = e.Butterfly("max", e.F("max", mn, "0f00000000"), 28);
        string d = e.F("div.rn", maxScale, "0f427C0000"), dmin = e.F("div.rn", maxMin, "0f427C0000");   // / 63
        string first = e.LaneIs(31, 0);
        e.StoreHalf(d, 0, first); e.StoreHalf(dmin, 2, first);
        d = e.HalfRound(d); dmin = e.HalfRound(dmin);
        string pD0 = e.IsZero(d), pM0 = e.IsZero(dmin);
        string ls = e.Clamp(e.Rni(e.F("div.rn", scale, d)), 0, 63), lm = e.Clamp(e.Rni(e.F("div.rn", mn, dmin)), 0, 63);
        b.L($"selp.b32 {ls}, 0, {ls}, {pD0};");
        b.L($"selp.b32 {lm}, 0, {lm}, {pM0};");

        // The 12 scale bytes from every sub-block's ls, lm (lane 4j's): j < 4: sc[j] = ls, sc[j+4] = lm; else
        // sc[j+4] = (ls & 15) | (lm & 15) << 4, sc[j−4] |= (ls >> 4) << 6, sc[j] |= (lm >> 4) << 6. Stored by lane 0.
        var lsj = new string[8]; var lmj = new string[8];
        for (int j = 0; j < 8; j++) { lsj[j] = e.From(ls, 4 * j); lmj[j] = e.From(lm, 4 * j); }
        var sc = new string[12];
        for (int k = 0; k < 12; k++) { sc[k] = b.R(); b.L($"mov.u32 {sc[k]}, 0;"); }
        for (int j = 0; j < 4; j++) { b.L($"mov.u32 {sc[j]}, {lsj[j]};"); b.L($"mov.u32 {sc[j + 4]}, {lmj[j]};"); }
        for (int j = 4; j < 8; j++)
        {
            string t = b.R(), u = b.R();
            b.L($"and.b32 {t}, {lsj[j]}, 15;"); b.L($"and.b32 {u}, {lmj[j]}, 15;"); b.L($"shl.b32 {u}, {u}, 4;"); b.L($"or.b32 {sc[j + 4]}, {t}, {u};");
            b.L($"shr.u32 {t}, {lsj[j]}, 4;"); b.L($"shl.b32 {t}, {t}, 6;"); b.L($"or.b32 {sc[j - 4]}, {sc[j - 4]}, {t};");
            b.L($"shr.u32 {t}, {lmj[j]}, 4;"); b.L($"shl.b32 {t}, {t}, 6;"); b.L($"or.b32 {sc[j]}, {sc[j]}, {t};");
        }

        string zero = b.R();
        b.L($"mov.u32 {zero}, 0;");
        for (int k = 0; k < 12; k++) e.StoreByte(sc[k], zero, 4 + k, first);

        // This lane's 8 codes as nibbles; the odd sub-block's lane (lane + 4) hands its codes to the even one's, which stores
        // the 8 bytes at 16 + pair·32 + (lane mod 4)·8.
        string dj = e.F("mul.rn", d, e.I2F(ls)), mj = e.F("mul.rn", dmin, e.I2F(lm)), pz = e.IsZero(dj);
        string nib = b.R();
        b.L($"mov.u32 {nib}, 0;");
        for (int k = 0; k < 8; k++)
        {
            string q = e.Clamp(e.Rni(e.F("div.rn", e.F("add.rn", v[k], mj), dj)), 0, 15);
            b.L($"selp.b32 {q}, 0, {q}, {pz};");
            e.Put(nib, q, 4 * k, 4);
        }

        string partner = e.Down(nib, 4), lead = e.LaneIs(4, 0);
        string pair = e.U("shr.u32", e.Lane, 3), inner = e.U("and.b32", e.Lane, 3), at = b.R();
        b.L($"mul.lo.u32 {at}, {pair}, 32;");
        b.L($"mad.lo.u32 {at}, {inner}, 8, {at};");
        for (int k = 0; k < 8; k++)
        {
            string code = e.Bits(nib, 4 * k, 4);
            e.Put(code, e.Bits(partner, 4 * k, 4), 4, 4);
            e.StoreByte(code, at, 16 + k, lead);
        }
    }

    // Q6_K: lane l holds values 8l .. 8l + 7, so sub-block j (16 values) is lanes 2j, 2j + 1. scale_j = max|v|/31; d = max
    // scale / 127 (f16); ls_j = clamp(rni(scale_j/d), −127, 127); q = clamp(rni(v/(d·ls_j)), −32, 31) + 32 into ql (low
    // nibbles) and qh (2 bits) by the layout: per half (128 values), for l in 0..31: q1 = v[l], q2 = v[l+32], q3 = v[l+64],
    // q4 = v[l+96], held by lanes r, r + 4, r + 8, r + 12 of the half's 16 (r = l / 8), which lane r gathers and stores.
    private static void EncodeQ6K(BlockEmit e, string[] v)
    {
        var b = e.B;
        string amax = e.Const("0f00000000");
        foreach (var x in v) amax = e.F("max", amax, e.Abs(x));
        amax = e.Butterfly("max", amax, 1);
        string scale = e.F("div.rn", amax, "0f41F80000");                               // / 31
        string maxScale = e.Butterfly("max", e.F("max", scale, "0f00000000"), 30);
        string d = e.F("div.rn", maxScale, "0f42FE0000");                               // / 127
        string first = e.LaneIs(31, 0);
        e.StoreHalf(d, 208, first);
        d = e.HalfRound(d);
        string pD0 = e.IsZero(d);
        string ls = e.Clamp(e.Rni(e.F("div.rn", scale, d)), -127, 127);
        b.L($"selp.b32 {ls}, 0, {ls}, {pD0};");
        string sb = b.R(), sub = e.U("shr.u32", e.Lane, 1);
        b.L($"and.b32 {sb}, {ls}, 255;");
        e.StoreByte(sb, sub, 192, e.LaneIs(1, 0));
        string dj = e.F("mul.rn", d, e.I2F(ls)), pZ = e.IsZero(dj);

        string lo4 = b.R(), hi2 = b.R();
        b.L($"mov.u32 {lo4}, 0;");
        b.L($"mov.u32 {hi2}, 0;");
        for (int k = 0; k < 8; k++)
        {
            string q = e.Clamp(e.Rni(e.F("div.rn", v[k], dj)), -32, 31);
            b.L($"selp.b32 {q}, 0, {q}, {pZ};");
            b.L($"add.s32 {q}, {q}, 32;");
            e.Put(lo4, q, 4 * k, 4);
            e.Put(hi2, e.U("shr.u32", q, 4), 2 * k, 2);
        }

        string lo4b = e.Down(lo4, 4), hi2b = e.Down(hi2, 4), lo4c = e.Down(lo4, 8), hi2c = e.Down(hi2, 8), lo4d = e.Down(lo4, 12), hi2d = e.Down(hi2, 12);
        string lead = e.LaneIs(12, 0), half = e.U("shr.u32", e.Lane, 4), r = e.U("and.b32", e.Lane, 15);
        string l0 = b.R(), qlAt = b.R(), qhAt = b.R();
        b.L($"mul.lo.u32 {l0}, {r}, 8;");
        b.L($"mad.lo.u32 {qlAt}, {half}, 64, {l0};");
        b.L($"mad.lo.u32 {qhAt}, {half}, 32, {l0};");
        for (int k = 0; k < 8; k++)
        {
            string lo1 = e.Bits(lo4, 4 * k, 4), lo2 = e.Bits(lo4b, 4 * k, 4), hq = e.Bits(hi2, 2 * k, 2);
            e.Put(lo1, e.Bits(lo4c, 4 * k, 4), 4, 4);
            e.Put(lo2, e.Bits(lo4d, 4 * k, 4), 4, 4);
            e.Put(hq, e.Bits(hi2b, 2 * k, 2), 2, 2);
            e.Put(hq, e.Bits(hi2c, 2 * k, 2), 4, 2);
            e.Put(hq, e.Bits(hi2d, 2 * k, 2), 6, 2);
            e.StoreByte(lo1, qlAt, k, lead);
            e.StoreByte(lo2, qlAt, 32 + k, lead);
            e.StoreByte(hq, qhAt, 128 + k, lead);
        }
    }
}
