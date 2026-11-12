// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's layout operations (Layout.cs): a thread per output word computes each of its values from the output
// index by the operation's rule (a gather; the sums — scatter-add, col2im — in the sum lane, converted on the store),
// so every word is written by one thread and no format needs atomics. Parameters: the input, the index array, an
// auxiliary buffer (scatter-add's row offsets and ordered sources), the output, the counts and sixteen integers in the
// operation's order as the kernels take them (LayoutArgs.KernelP).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>The integer parameters a layout kernel takes (LayoutArgs.KernelParams).</summary>
    public const int LayoutParamCount = LayoutArgs.KernelParams;

    private static readonly string LayoutParams = ".param .u64 p_x, .param .u64 p_x2, .param .u64 p_idx, .param .u64 p_aux, .param .u64 p_out, .param .u32 p_count, .param .u32 p_rows, .param .u32 p_cols, "
        + string.Join(", ", Enumerable.Range(0, LayoutParamCount).Select(k => $".param .u32 p_p{k}"))
        + ", " + string.Join(", ", Enumerable.Range(0, LayoutParamCount).Select(k => $".param .u32 p_m{k}, .param .u32 p_n{k}")) + ", .param .u32 p_mc, .param .u32 p_nc";

    /// <summary>The arguments a layout kernel takes after its 8 + LayoutParamCount: each integer's invariant-division
    /// constants (magic, shifts), then cols' (DivisionConstants; the kernels divide by them, RegisterDivisor).</summary>
    public const int LayoutDivisorArgs = 2 * LayoutParamCount + 2;

    /// <summary>Fills a layout launch's division constants from its integers (args[8 …]) and cols (args[7]).</summary>
    public static void LayoutDivisors(ulong[] args)
    {
        for (int k = 0; k < LayoutParamCount; k++)
        {
            var (m, s) = DivisionConstants((uint)args[8 + k]);
            args[8 + LayoutParamCount + 2 * k] = m;
            args[8 + LayoutParamCount + 2 * k + 1] = s;
        }

        var (mc, sc) = DivisionConstants((uint)args[7]);
        args[8 + 3 * LayoutParamCount] = mc;
        args[8 + 3 * LayoutParamCount + 1] = sc;
    }

    public static string Layout(Gpu gpu, Fmt f, Lane lane, string name, out string entry) => Layout(gpu, f, lane, name, false, out entry);

    /// <summary>perValue (a packed 32-bit word of 2, 4, 8, 16 or 32 values; GpuOps.LayoutPerValue): a thread per output
    /// value, not per word, the word's values joined by the word's lanes (shfl.sync, a group of Per lanes a word) and
    /// stored by its first: Per times the threads for an output too small to fill the device (an adaptive pool's few
    /// windows).
    /// inFmt: the inputs' format where it is not the output's (a composition's lane storage, as col2im of a product's sums).</summary>
    public static string Layout(Gpu gpu, Fmt f, Lane lane, string name, bool perValue, out string entry, Fmt? inFmt = null)
    {
        if (perValue && !PerValueFits(f)) throw new NotSupportedException($"{f.Name}: a thread per value takes a word of 2 to 32 whole values");
        if (perValue && inFmt is not null) throw new NotSupportedException("a thread per value: the inputs as the output's format");
        if (perValue) return LayoutPerValue(gpu, f, lane, name, out entry);
        entry = $"layout_{name}_{f.Name}_{lane.Name}{(inFmt is null ? "" : "_i" + inFmt.Name)}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), word = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {word}, {bid}, 256, {tid};");
        string x = Param64(b, "p_x"), x2 = Param64(b, "p_x2"), idx = Param64(b, "p_idx"), aux = Param64(b, "p_aux"), outp = Param64(b, "p_out");
        string count = Param32(b, "p_count"), cols = Param32(b, "p_cols");
        RegisterDivisor(b, cols, "p_mc", "p_nc");
        var p = new string[LayoutParamCount];
        for (int k = 0; k < LayoutParamCount; k++) p[k] = Param32(b, $"p_p{k}");
        for (int k = 0; k < LayoutParamCount; k++) RegisterDivisor(b, p[k], $"p_m{k}", $"p_n{k}");
        string words = b.R(), pw = b.P();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {count};");
        else { b.L($"add.u32 {words}, {count}, {f.Per - 1};"); b.L($"div.u32 {words}, {words}, {f.Per};"); }
        b.L($"setp.ge.u32 {pw}, {word}, {words};");
        b.L($"@{pw} bra DONE;");
        string code = f.WordBits == 64 ? b.RD() : b.R();
        if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
        var natural = f.Natural;
        string last = b.R();
        b.L($"sub.u32 {last}, {count}, 1;");
        for (int i = 0; i < f.Per; i++)
        {
            string v = b.R(), pv = b.P(), r = b.R(), c = b.R(), t = b.R();
            b.L($"mad.lo.u32 {v}, {word}, {f.Per}, {i};");
            b.L($"setp.lt.u32 {pv}, {v}, {count};");
            b.L($"min.u32 {v}, {v}, {last};");                        // a word's values past the count take the last value's addresses (dropped)
            DivU(b, r, v, cols);
            b.L($"mul.lo.u32 {t}, {r}, {cols};");
            b.L($"sub.u32 {c}, {v}, {t};");
            string value = LayoutValue(b, inFmt ?? f, lane, name, x, x2, idx, aux, p, cols, v, r, c, i);
            string stored = lane == natural ? value : Codec.Convert(b, lane, value, natural, f);
            string kept = natural.NewReg(b);
            b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {pv};");
            Codec.Pack(b, f, kept, natural, i, code);
        }

        string addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {word}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, LayoutParams);
    }

    /// <summary>A layout operation's output as blocks of f: each value the operation's own (LayoutValue, its inputs decoded
    /// where they are stored, in f32), the block encoded from them: a warp per block, or a CTA per block (a thread a value)
    /// for the formats of more values than a warp's lanes (BlockByCta), a value's window or gather being costly.</summary>
    public static string LayoutBlocks(Gpu gpu, Fmt f, string name, out string entry, Fmt? inFmt = null)
    {
        entry = $"layout_{name}_{f.Name}_blocks{(inFmt is null ? "" : "_i" + inFmt.Name)}";
        var lane = Lane.F32;
        var b = new PtxBuilder(gpu);
        string x = Param64(b, "p_x"), x2 = Param64(b, "p_x2"), idx = Param64(b, "p_idx"), aux = Param64(b, "p_aux"), outp = Param64(b, "p_out");
        string count = Param32(b, "p_count"), cols = Param32(b, "p_cols");
        RegisterDivisor(b, cols, "p_mc", "p_nc");
        var p = new string[LayoutParamCount];
        for (int k = 0; k < LayoutParamCount; k++) p[k] = Param32(b, $"p_p{k}");
        for (int k = 0; k < LayoutParamCount; k++) RegisterDivisor(b, p[k], $"p_m{k}", $"p_n{k}");
        int tag = 0;
        string Value(string at, string _)
        {
            string r = b.R(), c = b.R(), t = b.R();
            DivU(b, r, at, cols);
            b.L($"mul.lo.u32 {t}, {r}, {cols};");
            b.L($"sub.u32 {c}, {at}, {t};");
            return LayoutValue(b, inFmt ?? f, lane, name, x, x2, idx, aux, p, cols, at, r, c, tag++);     // its own labels a value
        }

        bool cta = BlockByCta(f);
        if (cta) EncodeBlockByCta(b, f, outp, count, Value);
        else EncodeBlock(b, f, outp, BlockWarp(b, f, count), count, Value);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, LayoutParams, cta ? f.BlockValues * 4 : 0);
    }

    /// <summary>A format the per-value layout kernel packs: whole values filling a 32-bit word, 2 to 32 of them.</summary>
    public static bool PerValueFits(Fmt f) => !f.IsBlock && f.WordBits == 32 && f.Per >= 2 && f.Per * f.Bits == 32;

    private static string LayoutPerValue(Gpu gpu, Fmt f, Lane lane, string name, out string entry)
    {
        entry = $"layout_{name}_{f.Name}_{lane.Name}_v";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), at = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {at}, {bid}, 256, {tid};");
        string x = Param64(b, "p_x"), x2 = Param64(b, "p_x2"), idx = Param64(b, "p_idx"), aux = Param64(b, "p_aux"), outp = Param64(b, "p_out");
        string count = Param32(b, "p_count"), cols = Param32(b, "p_cols");
        RegisterDivisor(b, cols, "p_mc", "p_nc");
        var p = new string[LayoutParamCount];
        for (int k = 0; k < LayoutParamCount; k++) p[k] = Param32(b, $"p_p{k}");
        for (int k = 0; k < LayoutParamCount; k++) RegisterDivisor(b, p[k], $"p_m{k}", $"p_n{k}");
        // Every launched thread runs to the shuffles (a warp's lanes all take part); past the count a thread computes the
        // last value (dropped) and past the last word stores nothing.
        string words = b.R(), word = b.R(), slot = b.R(), last = b.R(), v = b.R(), pv = b.P(), pw = b.P(), r = b.R(), c = b.R(), t = b.R();
        b.L($"add.u32 {words}, {count}, {f.Per - 1};");
        b.L($"div.u32 {words}, {words}, {f.Per};");
        b.L($"shr.u32 {word}, {at}, {System.Numerics.BitOperations.Log2((uint)f.Per)};");
        b.L($"and.b32 {slot}, {at}, {f.Per - 1};");
        b.L($"sub.u32 {last}, {count}, 1;");
        b.L($"setp.lt.u32 {pv}, {at}, {count};");
        b.L($"min.u32 {v}, {at}, {last};");
        DivU(b, r, v, cols);
        b.L($"mul.lo.u32 {t}, {r}, {cols};");
        b.L($"sub.u32 {c}, {v}, {t};");
        var natural = f.Natural;
        string value = LayoutValue(b, f, lane, name, x, x2, idx, aux, p, cols, v, r, c, 0);
        string stored = lane == natural ? value : Codec.Convert(b, lane, value, natural, f);
        string kept = natural.NewReg(b), code = b.R(), shift = b.R();
        b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {pv};");
        b.L($"mov.b32 {code}, 0;");
        Codec.Pack(b, f, kept, natural, 0, code);                    // the value's bits low in the word, the rest 0
        b.L($"mul.lo.u32 {shift}, {slot}, {f.Bits};");
        b.L($"shl.b32 {code}, {code}, {shift};");
        for (int off = f.Per / 2; off >= 1; off /= 2)
        {
            string other = b.R();
            b.L($"shfl.sync.bfly.b32 {other}, {code}, {off}, 0x1f, 0xffffffff;");
            b.L($"or.b32 {code}, {code}, {other};");
        }

        string pFirst = b.P(), addr = b.RD();
        b.L($"setp.ne.u32 {pFirst}, {slot}, 0;");
        b.L($"setp.ge.or.u32 {pw}, {word}, {words}, {pFirst};");
        b.L($"@{pw} bra DONE;");
        b.L($"mul.wide.u32 {addr}, {word}, 4;");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"st.global.b32 [{addr}], {code};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, LayoutParams);
    }

    /// <summary>The max-pool backward's first-ranked of every window of the forward input (p_x2) in Adad's order, a thread per window
    /// (p_count windows, in the output's order): its index into the input, 0xFFFFFFFF for a window with no pixel, into
    /// p_out (u32). The backward then reads each window's choice (LayoutValue) instead of scanning every window a pixel
    /// lies in; compared in the backward's lane, as the scan was.</summary>
    public static string LayoutWindowArgmax(Gpu gpu, Fmt f, Lane lane, string name, out string entry)
    {
        if (name is not ("maxpool_bwd" or "adaptive_maxpool_bwd")) throw new NotSupportedException($"window first-ranked of {name}");
        bool adaptive = name == "adaptive_maxpool_bwd";
        entry = $"layout_argmax_{name}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), i = b.R(), pd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        string x2 = Param64(b, "p_x2"), outp = Param64(b, "p_out"), count = Param32(b, "p_count");
        var p = new string[LayoutParamCount];
        for (int k = 0; k < LayoutParamCount; k++) p[k] = Param32(b, $"p_p{k}");
        for (int k = 0; k < LayoutParamCount; k++) RegisterDivisor(b, p[k], $"p_m{k}", $"p_n{k}");
        b.L($"setp.ge.u32 {pd}, {i}, {count};");
        b.L($"@{pd} bra DONE;");
        string H = p[1], W = p[2], OH = adaptive ? p[3] : p[11], OW = adaptive ? p[4] : p[12];      // a pool's OH OW: LayoutArgs.KernelP
        string ow = b.R(), oh = b.R(), ch = b.R(), t = b.R(), h0 = b.R(), kh = b.R(), w0 = b.R(), kw = b.R();
        RemU(b, ow, i, OW);
        DivU(b, t, i, OW);
        RemU(b, oh, t, OH);
        DivU(b, ch, t, OH);
        if (adaptive)
        {
            // [floor(i*n/o), ceil((i + 1)*n/o)) along each axis.
            void Span(string o, string n, string outs, string start, string len)
            {
                string e = b.R();
                b.L($"mul.lo.u32 {start}, {o}, {n};");
                DivU(b, start, start, outs);
                b.L($"add.u32 {e}, {o}, 1;");
                b.L($"mul.lo.u32 {e}, {e}, {n};");
                b.L($"add.u32 {e}, {e}, {outs};");
                b.L($"sub.u32 {e}, {e}, 1;");
                DivU(b, e, e, outs);
                b.L($"sub.u32 {len}, {e}, {start};");
            }

            Span(oh, H, OH, h0, kh);
            Span(ow, W, OW, w0, kw);
        }
        else
        {
            // From o*s - pad over k pixels; a pixel outside the input (above or left: unsigned past H or W) is skipped.
            b.L($"mul.lo.u32 {h0}, {oh}, {p[5]};");
            b.L($"sub.u32 {h0}, {h0}, {p[7]};");
            b.L($"mov.u32 {kh}, {p[3]};");
            b.L($"mul.lo.u32 {w0}, {ow}, {p[6]};");
            b.L($"sub.u32 {w0}, {w0}, {p[8]};");
            b.L($"mov.u32 {kw}, {p[4]};");
        }

        string aa = b.R(), bb = b.R(), hh = b.R(), ww = b.R(), cand = b.R(), bestAt = b.R(), best = lane.NewReg(b), pa = b.P(), pb = b.P();
        b.L($"mov.u32 {bestAt}, 0xFFFFFFFF;");
        b.L($"mov.{lane.Reg} {best}, {ReduceIdentity(lane, "max")};");
        b.L($"mov.u32 {aa}, 0;");
        b.Label("WA");
        b.L($"setp.ge.u32 {pa}, {aa}, {kh};");
        b.L($"@{pa} bra WA_END;");
        b.L($"add.u32 {hh}, {h0}, {aa};");
        b.L($"setp.ge.u32 {pa}, {hh}, {H};");
        b.L($"@{pa} bra WA_NEXT;");
        b.L($"mov.u32 {bb}, 0;");
        b.Label("WB");
        b.L($"setp.ge.u32 {pb}, {bb}, {kw};");
        b.L($"@{pb} bra WA_NEXT;");
        b.L($"add.u32 {ww}, {w0}, {bb};");
        b.L($"setp.ge.u32 {pb}, {ww}, {W};");
        b.L($"@{pb} bra WB_NEXT;");
        b.L($"mad.lo.u32 {cand}, {ch}, {H}, {hh};");
        b.L($"mad.lo.u32 {cand}, {cand}, {W}, {ww};");
        // The first-ranked in row-major order (Adad's order, Rank): a later value takes it only when it ranks strictly above.
        string xv = Codec.UnpackAt(b, f, x2, cand, lane), pFirst = b.P();
        b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
        string pGt = RankAbove(b, lane, xv, best);
        b.L($"or.pred {pGt}, {pGt}, {pFirst};");
        b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {best}, {xv}, {best}, {pGt};");
        b.L($"selp.u32 {bestAt}, {cand}, {bestAt}, {pGt};");
        b.Label("WB_NEXT");
        b.L($"add.u32 {bb}, {bb}, 1;");
        b.L("bra WB;");
        b.Label("WA_NEXT");
        b.L($"add.u32 {aa}, {aa}, 1;");
        b.L("bra WA;");
        b.Label("WA_END");
        string addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {i}, 4;");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"st.global.u32 [{addr}], {bestAt};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, LayoutParams);
    }

    /// <summary>Whether an adaptive pool's windows take a warp each (AdaptiveMaxWarp, LayoutWindowArgmaxWarp): windows of 32
    /// values or more (a lane at least one of each), where a thread a window leaves the device with a few thousand threads
    /// each walking a thousand values (64 × 256² to 7 × 7: 3136 threads). A stated choice: a warp's lanes are its width.</summary>
    public static bool AdaptiveByWarp(int[] p) => p.Length >= 5 && p[3] > 0 && p[4] > 0 && (long)(p[1] / p[3]) * (p[2] / p[4]) >= 32;

    // One adaptive window's first-ranked by a warp: (ch, oh, ow) from the window index w; each lane walks the window's
    // values e = lane, lane + 32, … (row-major in the window: ih = h0 + e / kw, iw = w0 + e mod kw, advanced by 32 a step
    // without a division); index: the first-ranked's input position too (the lowest on a tie), joined by shuffles.
    // Returns (value, position) as every lane holds them after the butterflies (position 0xFFFFFFFF: none).
    private static (string Value, string At) AdaptiveWindowWarp(PtxBuilder b, Fmt f, Lane lane, string x, string[] p, string w, string wl, bool index, string tag)
    {
        string H = p[1], W = p[2], OH = p[3], OW = p[4];
        string ow = b.R(), oh = b.R(), ch = b.R(), t = b.R(), h0 = b.R(), h1 = b.R(), w0 = b.R(), w1 = b.R(), kh = b.R(), kw = b.R(), area = b.R();
        RemU(b, ow, w, OW);
        DivU(b, t, w, OW);
        RemU(b, oh, t, OH);
        DivU(b, ch, t, OH);
        void Span(string o, string n, string outs, string start, string end)
        {
            b.L($"mul.lo.u32 {start}, {o}, {n};");
            DivU(b, start, start, outs);
            b.L($"add.u32 {end}, {o}, 1;");
            b.L($"mul.lo.u32 {end}, {end}, {n};");
            b.L($"add.u32 {end}, {end}, {outs};");
            b.L($"sub.u32 {end}, {end}, 1;");
            DivU(b, end, end, outs);
        }

        Span(oh, H, OH, h0, h1);
        Span(ow, W, OW, w0, w1);
        b.L($"sub.u32 {kh}, {h1}, {h0};");
        b.L($"sub.u32 {kw}, {w1}, {w0};");
        b.L($"mul.lo.u32 {area}, {kh}, {kw};");
        string e = b.R(), ih = b.R(), iw = b.R(), stepH = b.R(), stepW = b.R(), pE = b.P(), src = b.R(), pWrap = b.P();
        DivU(b, ih, wl, kw);
        RemU(b, iw, wl, kw);
        b.L($"add.u32 {ih}, {ih}, {h0};");
        DivU(b, stepH, "32", kw);
        RemU(b, stepW, "32", kw);
        b.L($"mov.u32 {e}, {wl};");
        string best = lane.NewReg(b), bestAt = b.R(), sel = lane.IsFloat ? lane.Op : lane.Reg;
        b.L($"mov.{lane.Reg} {best}, {ReduceIdentity(lane, "max")};");
        b.L($"mov.u32 {bestAt}, 0xFFFFFFFF;");
        b.Label($"{tag}_WALK");
        b.L($"setp.ge.u32 {pE}, {e}, {area};");
        b.L($"@{pE} bra {tag}_WALKED;");
        b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
        b.L($"mad.lo.u32 {src}, {src}, {W}, {w0};");
        b.L($"add.u32 {src}, {src}, {iw};");
        string xv = Codec.UnpackAt(b, f, x, src, lane);
        if (index)
        {
            // A lane's own values rise in position: a later one takes only when it ranks strictly above.
            string pFirst = b.P();
            b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
            string pGt = RankAbove(b, lane, xv, best);
            b.L($"or.pred {pGt}, {pGt}, {pFirst};");
            b.L($"selp.{sel} {best}, {xv}, {best}, {pGt};");
            b.L($"selp.u32 {bestAt}, {src}, {bestAt}, {pGt};");
        }
        else
        {
            First(b, lane, best, best, xv);
            b.L($"mov.u32 {bestAt}, 0;");
        }

        b.L($"add.u32 {e}, {e}, 32;");
        b.L($"add.u32 {ih}, {ih}, {stepH};");
        b.L($"add.u32 {iw}, {iw}, {stepW};");
        b.L($"setp.ge.u32 {pWrap}, {iw}, {kw};");
        b.L($"@{pWrap} sub.u32 {iw}, {iw}, {kw};");
        b.L($"@{pWrap} add.u32 {ih}, {ih}, 1;");
        b.L($"bra {tag}_WALK;");
        b.Label($"{tag}_WALKED");
        for (int off = 16; off >= 1; off /= 2)
        {
            string ov = lane.NewReg(b), oa = b.R();
            if (lane.Wide)
            {
                string lo = b.R(), hi = b.R(), lo2 = b.R(), hi2 = b.R();
                b.L($"mov.b64 {{{lo}, {hi}}}, {best};");
                b.L($"shfl.sync.bfly.b32 {lo2}, {lo}, {off}, 0x1f, 0xffffffff;");
                b.L($"shfl.sync.bfly.b32 {hi2}, {hi}, {off}, 0x1f, 0xffffffff;");
                b.L($"mov.b64 {ov}, {{{lo2}, {hi2}}};");
            }
            else
            {
                string bits = b.R(), bits2 = b.R();
                b.L($"mov.b32 {bits}, {best};");
                b.L($"shfl.sync.bfly.b32 {bits2}, {bits}, {off}, 0x1f, 0xffffffff;");
                b.L($"mov.b32 {ov}, {bits2};");
            }

            if (!index)
            {
                First(b, lane, best, best, ov);
                continue;
            }

            // The other's (ov, oa) where it ranks above, or the same with a lower position; an empty one never.
            b.L($"shfl.sync.bfly.b32 {oa}, {bestAt}, {off}, 0x1f, 0xffffffff;");
            var (pGt2, pEq2) = Rank(b, lane, ov, best);
            string pLt = b.P(), pTake = b.P(), pMine = b.P(), pOther = b.P();
            b.L($"setp.lt.u32 {pLt}, {oa}, {bestAt};");
            b.L($"and.pred {pEq2}, {pEq2}, {pLt};");
            b.L($"or.pred {pTake}, {pGt2}, {pEq2};");
            b.L($"setp.eq.u32 {pMine}, {bestAt}, 0xFFFFFFFF;");
            b.L($"or.pred {pTake}, {pTake}, {pMine};");
            b.L($"setp.ne.u32 {pOther}, {oa}, 0xFFFFFFFF;");
            b.L($"and.pred {pTake}, {pTake}, {pOther};");
            b.L($"selp.{sel} {best}, {ov}, {best}, {pTake};");
            b.L($"selp.u32 {bestAt}, {oa}, {bestAt}, {pTake};");
        }

        return (best, bestAt);
    }

    /// <summary>adaptive_maxpool by a warp a window (AdaptiveByWarp): a warp per stored word of the output, its values' windows
    /// one after another, each window's first-ranked in Adad's order (First: the order's max, a NaN where there is one) by
    /// the warp's lanes and shuffles, the word stored by lane 0. The order's max is the same whichever way the window is
    /// walked. Params as Layout's.</summary>
    public static string AdaptiveMaxWarp(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        if (f.IsBlock) throw new NotSupportedException("a block format's output takes the block kernels");
        entry = $"layout_adaptive_maxpool_warp_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), word = b.R(), wl = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mul.lo.u32 {word}, {bid}, 8;");
        string wib = b.R();
        b.L($"shr.u32 {wib}, {tid}, 5;");
        b.L($"add.u32 {word}, {word}, {wib};");
        b.L($"and.b32 {wl}, {tid}, 31;");
        string x = Param64(b, "p_x"), outp = Param64(b, "p_out"), count = Param32(b, "p_count");
        var p = new string[LayoutParamCount];
        for (int k = 0; k < LayoutParamCount; k++) p[k] = Param32(b, $"p_p{k}");
        for (int k = 0; k < LayoutParamCount; k++) RegisterDivisor(b, p[k], $"p_m{k}", $"p_n{k}");
        string words = b.R(), pw = b.P(), last = b.R();
        if (f.WordBits == 64 || f.Per == 1) b.L($"mov.u32 {words}, {count};");
        else { b.L($"add.u32 {words}, {count}, {f.Per - 1};"); b.L($"div.u32 {words}, {words}, {f.Per};"); }
        b.L($"setp.ge.u32 {pw}, {word}, {words};");
        b.L($"@{pw} bra DONE;");
        b.L($"sub.u32 {last}, {count}, 1;");
        var natural = f.Natural;
        string code = f.WordBits == 64 ? b.RD() : b.R();
        if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string v = b.R(), pv = b.P();
            b.L($"mad.lo.u32 {v}, {word}, {f.Per}, {i};");
            b.L($"setp.lt.u32 {pv}, {v}, {count};");
            b.L($"min.u32 {v}, {v}, {last};");
            var (best, _) = AdaptiveWindowWarp(b, f, lane, x, p, v, wl, false, $"AW{i}");
            string stored = lane == natural ? best : Codec.Convert(b, lane, best, natural, f);
            string kept = natural.NewReg(b);
            b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {pv};");
            Codec.Pack(b, f, kept, natural, i, code);
        }

        string p0 = b.P(), addr = b.RD();
        b.L($"setp.ne.u32 {p0}, {wl}, 0;");
        b.L($"@{p0} bra DONE;");
        b.L($"mul.wide.u32 {addr}, {word}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, LayoutParams);
    }

    /// <summary>LayoutWindowArgmax of adaptive_maxpool_bwd by a warp a window (AdaptiveByWarp): the window's first-ranked
    /// position in Adad's order, the lowest on a tie (the scan's own answer, whichever way the window is walked), into
    /// p_out (u32; 0xFFFFFFFF for none). Params as LayoutWindowArgmax's.</summary>
    public static string LayoutWindowArgmaxWarp(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"layout_argmax_adaptive_warp_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), w = b.R(), wl = b.R(), wib = b.R(), pd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"shr.u32 {wib}, {tid}, 5;");
        b.L($"mad.lo.u32 {w}, {bid}, 8, {wib};");
        b.L($"and.b32 {wl}, {tid}, 31;");
        string x2 = Param64(b, "p_x2"), outp = Param64(b, "p_out"), count = Param32(b, "p_count");
        var p = new string[LayoutParamCount];
        for (int k = 0; k < LayoutParamCount; k++) p[k] = Param32(b, $"p_p{k}");
        for (int k = 0; k < LayoutParamCount; k++) RegisterDivisor(b, p[k], $"p_m{k}", $"p_n{k}");
        b.L($"setp.ge.u32 {pd}, {w}, {count};");
        b.L($"@{pd} bra DONE;");
        var (_, at) = AdaptiveWindowWarp(b, f, lane, x2, p, w, wl, true, "AA");
        string p0 = b.P(), addr = b.RD();
        b.L($"setp.ne.u32 {p0}, {wl}, 0;");
        b.L($"@{p0} bra DONE;");
        b.L($"mul.wide.u32 {addr}, {w}, 4;");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"st.global.u32 [{addr}], {at};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, LayoutParams);
    }

    // The value of output (v; row r, column c) as a lane register.
    private static string LayoutValue(PtxBuilder b, Fmt f, Lane lane, string name, string x, string x2, string idx, string aux, string[] p, string cols, string v, string r, string c, int tag)
    {
        string Load(string index) => Codec.UnpackAt(b, f, x, index, lane);
        string Load2(string index) => Codec.UnpackAt(b, f, x2, index, lane);
        string Index(string ptr, string i) { string a = b.RD(), o = b.R(); b.L($"mul.wide.u32 {a}, {i}, 4;"); b.L($"add.u64 {a}, {ptr}, {a};"); b.L($"ld.global.s32 {o}, [{a}];"); return o; }
        string Zero() { string z = lane.NewReg(b); b.L($"mov.{lane.Reg} {z}, {lane.Zero};"); return z; }
        string Sel(string pred, string a, string z) { string d = lane.NewReg(b); b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {d}, {a}, {z}, {pred};"); return d; }
        if (name.StartsWith("strided", StringComparison.Ordinal))
        {
            // A canonical view of rank n (strided1 … strided7: LayoutArgs.StridedKernel): row r over the outer dimensions, column
            // c along the last; source = base + Σ i_d · s_d (strides wrap as u32: the source itself is inside the input).
            int n = int.Parse(name.AsSpan(7));
            string src = b.R(), t = b.R();
            b.L($"mad.lo.u32 {src}, {c}, {p[2 * n - 1]}, {p[2 * n]};");
            b.L($"mov.u32 {t}, {r};");
            for (int d = n - 2; d >= 1; d--)
            {
                string i = b.R();
                RemU(b, i, t, p[d]);
                DivU(b, t, t, p[d]);
                b.L($"mad.lo.u32 {src}, {i}, {p[n + d]}, {src};");
            }

            if (n >= 2) b.L($"mad.lo.u32 {src}, {t}, {p[n]}, {src};");
            return Load(src);
        }

        switch (name)
        {

            case "one_hot":
            {
                string target = Index(idx, r), pe = b.P(), one = lane.NewReg(b);
                b.L($"setp.eq.u32 {pe}, {target}, {c};");
                b.L($"mov.{lane.Reg} {one}, {lane.Literal(1)};");
                return Sel(pe, one, Zero());
            }

            case "gather_rows":
            {
                // An index outside the input's rows (p[0], unsigned: a negative one too) reads nothing: 0.
                string row = Index(idx, r), src = b.R(), pin = b.P();
                b.L($"setp.lt.u32 {pin}, {row}, {p[0]};");
                b.L($"mad.lo.u32 {src}, {row}, {cols}, {c};");
                b.L($"selp.u32 {src}, {src}, 0, {pin};");             // a safe address when out of range (dropped)
                return Sel(pin, Load(src), Zero());
            }

            case "gather_cols":
            {
                string col = Index(idx, c), src = b.R(), pin = b.P();
                b.L($"setp.lt.u32 {pin}, {col}, {p[1]};");
                b.L($"mad.lo.u32 {src}, {r}, {p[1]}, {col};");
                b.L($"selp.u32 {src}, {src}, 0, {pin};");
                return Sel(pin, Load(src), Zero());
            }

            case "scatter_add_rows":
            {
                // aux: offsets[rows + 1] then the sources in order; output (r, c) = Σ x[source·cols + c] over the row's sources.
                string acc = Zero(), from = Index(aux, r), rp1 = b.R(), to = b.R(), q = b.R(), pEnd = b.P(), src = b.R(), sIdx = b.R();
                b.L($"add.u32 {rp1}, {r}, 1;");
                string toV = Index(aux, rp1);
                b.L($"mov.u32 {to}, {toV};");
                b.L($"mov.u32 {q}, {from};");
                b.Label($"SA{tag}");
                b.L($"setp.ge.u32 {pEnd}, {q}, {to};");
                b.L($"@{pEnd} bra SA{tag}_END;");
                string rowsP1 = b.R();
                b.L($"add.u32 {rowsP1}, {p[2]}, 1;");
                b.L($"add.u32 {sIdx}, {rowsP1}, {q};");
                string s = Index(aux, sIdx);
                b.L($"mad.lo.u32 {src}, {s}, {cols}, {c};");
                string xv = Load(src);
                b.L(lane.Add(acc, acc, xv));
                b.L($"add.u32 {q}, {q}, 1;");
                b.L($"bra SA{tag};");
                b.Label($"SA{tag}_END");
                return acc;
            }

            case "im2col":
            {
                // r = (ch·kh + a)·kw + bb; c = oh·OW + ow; source (ch, oh·sh − top + a·dh, ow·sw − left + bb·dw) or 0 (p: C H W kh kw
                // sh sw top left bottom right dh dw OH OW, LayoutArgs.KernelP).
                string kw = p[4], kh = p[3], H = p[1], W = p[2], ow = b.R(), oh = b.R(), bb = b.R(), a = b.R(), ch = b.R(), t = b.R(), ih = b.R(), iw = b.R(), pin = b.P(), p2 = b.P(), src = b.R(), ow2 = b.R();
                string OW = p[14];
                RemU(b, ow, c, OW);
                DivU(b, oh, c, OW);
                RemU(b, bb, r, kw);
                DivU(b, t, r, kw);
                RemU(b, a, t, kh);
                DivU(b, ch, t, kh);
                b.L($"mul.lo.u32 {ih}, {oh}, {p[5]};");
                b.L($"sub.u32 {ih}, {ih}, {p[7]};");
                b.L($"mad.lo.u32 {ih}, {a}, {p[11]}, {ih};");
                b.L($"mul.lo.u32 {iw}, {ow}, {p[6]};");
                b.L($"sub.u32 {iw}, {iw}, {p[8]};");
                b.L($"mad.lo.u32 {iw}, {bb}, {p[12]}, {iw};");
                b.L($"setp.lt.u32 {pin}, {ih}, {H};");
                b.L($"setp.lt.u32 {p2}, {iw}, {W};");
                b.L($"and.pred {pin}, {pin}, {p2};");
                b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {src}, {src}, {W}, {iw};");
                b.L($"selp.u32 {src}, {src}, 0, {pin};");             // a safe address when out of range (dropped)
                _ = ow2;
                return Sel(pin, Load(src), Zero());
            }

            case "col2im":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih: Σ over (a, bb) with (ih + top − a·dh) = oh·sh, (iw + left − bb·dw) = ow·sw in range.
                string H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8], dh = p[11], dw = p[12];
                string OW = p[14], OH = p[13];
                string ih = b.R(), ch = b.R(), acc = Zero(), a = b.R(), bb = b.R(), th = b.R(), tw = b.R(), oh = b.R(), ow = b.R(), rem = b.R(), pa = b.P(), pb = b.P(), pc = b.P(), src = b.R(), plane = b.R(), rowIdx = b.R();
                RemU(b, ih, r, H);
                DivU(b, ch, r, H);
                b.L($"mul.lo.u32 {plane}, {OH}, {OW};");
                b.L($"mov.u32 {a}, 0;");
                b.Label($"CA{tag}");
                b.L($"setp.ge.u32 {pa}, {a}, {kh};");
                b.L($"@{pa} bra CA{tag}_END;");
                b.L($"add.u32 {th}, {ih}, {top};");
                b.L($"mul.lo.u32 {rem}, {a}, {dh};");
                b.L($"sub.s32 {th}, {th}, {rem};");                      // may go negative: a signed test
                b.L($"setp.lt.s32 {pb}, {th}, 0;");
                b.L($"@{pb} bra CA{tag}_NEXT;");
                RemU(b, rem, th, sh);
                b.L($"setp.ne.u32 {pb}, {rem}, 0;");
                b.L($"@{pb} bra CA{tag}_NEXT;");
                DivU(b, oh, th, sh);
                b.L($"setp.ge.u32 {pb}, {oh}, {OH};");
                b.L($"@{pb} bra CA{tag}_NEXT;");
                b.L($"mov.u32 {bb}, 0;");
                b.Label($"CB{tag}");
                b.L($"setp.ge.u32 {pc}, {bb}, {kw};");
                b.L($"@{pc} bra CB{tag}_END;");
                b.L($"add.u32 {tw}, {c}, {left};");
                b.L($"mul.lo.u32 {rem}, {bb}, {dw};");
                b.L($"sub.s32 {tw}, {tw}, {rem};");
                b.L($"setp.lt.s32 {pc}, {tw}, 0;");
                b.L($"@{pc} bra CB{tag}_NEXT;");
                RemU(b, rem, tw, sw);
                b.L($"setp.ne.u32 {pc}, {rem}, 0;");
                b.L($"@{pc} bra CB{tag}_NEXT;");
                DivU(b, ow, tw, sw);
                b.L($"setp.ge.u32 {pc}, {ow}, {OW};");
                b.L($"@{pc} bra CB{tag}_NEXT;");
                // column row index = (ch·kh + a)·kw + bb; column = oh·OW + ow
                b.L($"mad.lo.u32 {rowIdx}, {ch}, {kh}, {a};");
                b.L($"mad.lo.u32 {rowIdx}, {rowIdx}, {kw}, {bb};");
                b.L($"mul.lo.u32 {src}, {rowIdx}, {plane};");
                b.L($"mad.lo.u32 {src}, {oh}, {OW}, {src};");
                b.L($"add.u32 {src}, {src}, {ow};");
                string xv = Load(src);
                b.L(lane.Add(acc, acc, xv));
                b.Label($"CB{tag}_NEXT");
                b.L($"add.u32 {bb}, {bb}, 1;");
                b.L($"bra CB{tag};");
                b.Label($"CB{tag}_END");
                b.Label($"CA{tag}_NEXT");
                b.L($"add.u32 {a}, {a}, 1;");
                b.L($"bra CA{tag};");
                b.Label($"CA{tag}_END");
                return acc;
            }

            case "interpolate_nearest":
            {
                // r = ch·OH + oh, c = ow; p: C H W OH OW αh βh δh αw βw δw: ih the position nearest (oh·αh + βh) / δh (NearestAxis), iw likewise.
                string H = p[1], W = p[2], OH = p[3], oh = b.R(), ch = b.R(), src = b.R();
                RemU(b, oh, r, OH);
                DivU(b, ch, r, OH);
                string ih = NearestAxis(b, oh, H, p[5], p[6], p[7]), iw = NearestAxis(b, c, W, p[8], p[9], p[10]);
                b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {src}, {src}, {W}, {iw};");
                return Load(src);
            }

            case "interpolate_nearest_bwd":
            case "interpolate_bilinear_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW): the outputs whose source rows and columns include
                // the position (oh, then ow ascending; LayoutArgs.InterpolateSources), each dy by its weight: acc = fma(dy, wy·wx, acc).
                bool bilinear = name == "interpolate_bilinear_bwd";
                string H = p[1], W = p[2], OH = p[3], OW = p[4], ih = b.R(), ch = b.R(), acc = lane.NewReg(b);
                RemU(b, ih, r, H);
                DivU(b, ch, r, H);
                var (h0, h1) = SourceRange(b, bilinear, ih, H, OH, p[5], p[6], p[7]);
                var (w0, w1) = SourceRange(b, bilinear, c, W, OW, p[8], p[9], p[10]);
                b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
                string oh = b.R(), ow = b.R(), pa = b.P(), pb = b.P(), dyIdx = b.R();
                b.L($"mov.u32 {oh}, {h0};");
                b.Label($"IA{tag}");
                b.L($"setp.gt.s32 {pa}, {oh}, {h1};");
                b.L($"@{pa} bra IA{tag}_END;");
                var (hasY, wy) = AxisWeight(b, lane, bilinear, oh, ih, H, p[5], p[6], p[7]);
                b.L($"@!{hasY} bra IA{tag}_NEXT;");
                b.L($"mov.u32 {ow}, {w0};");
                b.Label($"IB{tag}");
                b.L($"setp.gt.s32 {pb}, {ow}, {w1};");
                b.L($"@{pb} bra IA{tag}_NEXT;");
                var (hasX, wx) = AxisWeight(b, lane, bilinear, ow, c, W, p[8], p[9], p[10]);
                b.L($"@!{hasX} bra IB{tag}_NEXT;");
                b.L($"mad.lo.u32 {dyIdx}, {ch}, {OH}, {oh};");
                b.L($"mad.lo.u32 {dyIdx}, {dyIdx}, {OW}, {ow};");
                string w = lane.NewReg(b);
                b.L(lane.Mul(w, wy, wx));
                b.L(lane.Fma(acc, Load(dyIdx), w, acc));
                b.Label($"IB{tag}_NEXT");
                b.L($"add.u32 {ow}, {ow}, 1;");
                b.L($"bra IB{tag};");
                b.Label($"IA{tag}_NEXT");
                b.L($"add.u32 {oh}, {oh}, 1;");
                b.L($"bra IA{tag};");
                b.Label($"IA{tag}_END");
                return acc;
            }

            case "interpolate_bilinear":
            {
                // p: C H W OH OW αh βh δh αw βw δw: each axis's two rows and weight exactly (BilinearAxis).
                string H = p[1], W = p[2], OH = p[3], oh = b.R(), ch = b.R();
                RemU(b, oh, r, OH);
                DivU(b, ch, r, OH);
                var (y0, y1, fy, gy) = BilinearAxis(b, lane, oh, H, p[5], p[6], p[7]);
                var (x0, x1, fx, gx) = BilinearAxis(b, lane, c, W, p[8], p[9], p[10]);
                string Pixel(string yy, string xx) { string s = b.R(); b.L($"mad.lo.u32 {s}, {ch}, {H}, {yy};"); b.L($"mad.lo.u32 {s}, {s}, {W}, {xx};"); return Load(s); }
                string top = lane.NewReg(b), bottom = lane.NewReg(b), o = lane.NewReg(b), t1 = lane.NewReg(b), t2 = lane.NewReg(b);
                b.L(lane.Mul(t1, Pixel(y0, x0), gx));
                b.L(lane.Fma(top, Pixel(y0, x1), fx, t1));
                b.L(lane.Mul(t2, Pixel(y1, x0), gx));
                b.L(lane.Fma(bottom, Pixel(y1, x1), fx, t2));
                b.L(lane.Mul(t1, top, gy));
                b.L(lane.Fma(o, bottom, fy, t1));
                return o;
            }

            case "shift":
            {
                string s = b.R(), pin = b.P(), src = b.R();
                b.L($"sub.s32 {s}, {c}, {p[2]};");
                b.L($"setp.lt.u32 {pin}, {s}, {cols};");                  // unsigned: negative values are out of range
                b.L($"mad.lo.u32 {src}, {r}, {cols}, {s};");
                b.L($"selp.u32 {src}, {src}, 0, {pin};");
                return Sel(pin, Load(src), Zero());
            }

            case "pad2d":
            {
                string pin = b.P(), src = b.R();
                b.L($"setp.lt.u32 {pin}, {c}, {p[1]};");
                b.L($"mad.lo.u32 {src}, {r}, {p[1]}, {c};");
                b.L($"selp.u32 {src}, {src}, 0, {pin};");
                return Sel(pin, Load(src), Zero());
            }

            case "maxpool":
            case "avgpool":
            case "depthwise_conv":
            {
                // r = ch·OH + oh, c = ow; the window over (a, bb) from (oh·sh − top, ow·sw − left) with the dilations (1 for the
                // pools); OH from LayoutArgs.KernelP. A pool's window holds an input value (LayoutArgs.Validate): no empty case.
                bool max = name == "maxpool", conv = name == "depthwise_conv";
                string H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                string dh = conv ? p[11] : "1", dw = conv ? p[12] : "1";
                string OH = conv ? p[13] : p[11], oh = b.R(), ch = b.R(), a = b.R(), bb = b.R(), ih = b.R(), iw = b.R(), pa = b.P(), pb = b.P(), src = b.R(), n = b.R();
                string acc = lane.NewReg(b);
                RemU(b, oh, r, OH);
                DivU(b, ch, r, OH);
                b.L($"mov.{lane.Reg} {acc}, {(max ? ReduceIdentity(lane, "max") : lane.Zero)};");
                b.L($"mov.u32 {n}, 0;");
                b.L($"mov.u32 {a}, 0;");
                b.Label($"PA{tag}");
                b.L($"setp.ge.u32 {pa}, {a}, {kh};");
                b.L($"@{pa} bra PA{tag}_END;");
                b.L($"mul.lo.u32 {ih}, {oh}, {sh};");
                b.L($"sub.u32 {ih}, {ih}, {top};");
                b.L($"mad.lo.u32 {ih}, {a}, {dh}, {ih};");
                b.L($"setp.ge.u32 {pa}, {ih}, {H};");                      // unsigned: negative rows are out of range
                b.L($"@{pa} bra PA{tag}_NEXT;");
                b.L($"mov.u32 {bb}, 0;");
                b.Label($"PB{tag}");
                b.L($"setp.ge.u32 {pb}, {bb}, {kw};");
                b.L($"@{pb} bra PB{tag}_END;");
                b.L($"mul.lo.u32 {iw}, {c}, {sw};");
                b.L($"sub.u32 {iw}, {iw}, {left};");
                b.L($"mad.lo.u32 {iw}, {bb}, {dw}, {iw};");
                b.L($"setp.ge.u32 {pb}, {iw}, {W};");
                b.L($"@{pb} bra PB{tag}_NEXT;");
                b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {src}, {src}, {W}, {iw};");
                string xv = Load(src);
                if (max) First(b, lane, acc, acc, xv);
                else if (conv)
                {
                    string wIdx = b.R();
                    b.L($"mad.lo.u32 {wIdx}, {ch}, {kh}, {a};");
                    b.L($"mad.lo.u32 {wIdx}, {wIdx}, {kw}, {bb};");
                    string wv = Load2(wIdx);
                    b.L(lane.Fma(acc, wv, xv, acc));
                }
                else b.L(lane.Add(acc, acc, xv));
                b.L($"add.u32 {n}, {n}, 1;");
                b.Label($"PB{tag}_NEXT");
                b.L($"add.u32 {bb}, {bb}, 1;");
                b.L($"bra PB{tag};");
                b.Label($"PB{tag}_END");
                b.Label($"PA{tag}_NEXT");
                b.L($"add.u32 {a}, {a}, 1;");
                b.L($"bra PA{tag};");
                b.Label($"PA{tag}_END");
                if (conv || max) return acc;
                string nf = lane.NewReg(b), mean = lane.NewReg(b);
                b.L($"cvt.rn.{lane.Op}.u32 {nf}, {n};");
                b.L($"div.rn.{lane.Op} {mean}, {acc}, {nf};");
                return mean;
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW), x2 = the forward input (maxpool).
                bool max = name == "maxpool_bwd";
                string H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], top = p[7], left = p[8];
                string OH = p[11], OW = p[12];
                string ih = b.R(), ch = b.R(), acc = lane.NewReg(b), a = b.R(), bb = b.R(), th = b.R(), tw = b.R(), oh = b.R(), ow = b.R(), rem = b.R(), pa = b.P(), pb = b.P(), dyIdx = b.R(), me = b.R();
                RemU(b, ih, r, H);
                DivU(b, ch, r, H);
                b.L($"mad.lo.u32 {me}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {me}, {me}, {W}, {c};");
                b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
                b.L($"mov.u32 {a}, 0;");
                b.Label($"BA{tag}");
                b.L($"setp.ge.u32 {pa}, {a}, {kh};");
                b.L($"@{pa} bra BA{tag}_END;");
                b.L($"add.u32 {th}, {ih}, {top};");
                b.L($"sub.s32 {th}, {th}, {a};");
                b.L($"setp.lt.s32 {pa}, {th}, 0;");
                b.L($"@{pa} bra BA{tag}_NEXT;");
                RemU(b, rem, th, sh);
                b.L($"setp.ne.u32 {pa}, {rem}, 0;");
                b.L($"@{pa} bra BA{tag}_NEXT;");
                DivU(b, oh, th, sh);
                b.L($"setp.ge.u32 {pa}, {oh}, {OH};");
                b.L($"@{pa} bra BA{tag}_NEXT;");
                b.L($"mov.u32 {bb}, 0;");
                b.Label($"BB{tag}");
                b.L($"setp.ge.u32 {pb}, {bb}, {kw};");
                b.L($"@{pb} bra BB{tag}_END;");
                b.L($"add.u32 {tw}, {c}, {left};");
                b.L($"sub.s32 {tw}, {tw}, {bb};");
                b.L($"setp.lt.s32 {pb}, {tw}, 0;");
                b.L($"@{pb} bra BB{tag}_NEXT;");
                RemU(b, rem, tw, sw);
                b.L($"setp.ne.u32 {pb}, {rem}, 0;");
                b.L($"@{pb} bra BB{tag}_NEXT;");
                DivU(b, ow, tw, sw);
                b.L($"setp.ge.u32 {pb}, {ow}, {OW};");
                b.L($"@{pb} bra BB{tag}_NEXT;");
                b.L($"mad.lo.u32 {dyIdx}, {ch}, {OH}, {oh};");
                b.L($"mad.lo.u32 {dyIdx}, {dyIdx}, {OW}, {ow};");
                string dy = Load(dyIdx);
                if (max)
                {
                    // The window's first-ranked, found once for all its pixels (LayoutWindowArgmax, in aux).
                    string chosen = Index(aux, dyIdx), pChosen = b.P();
                    b.L($"setp.eq.u32 {pChosen}, {chosen}, {me};");
                    b.L($"@{pChosen} {lane.Add(acc, acc, dy)}");
                }
                else
                {
                    // The window (oh, ow): its n values (avg), dy / n each.
                    string cnt = b.R(), nf = lane.NewReg(b), share = lane.NewReg(b);
                    b.L($"mul.lo.u32 {cnt}, {WindowSpan(b, oh, sh, top, kh, H)}, {WindowSpan(b, ow, sw, left, kw, W)};");
                    b.L($"cvt.rn.{lane.Op}.u32 {nf}, {cnt};");
                    b.L($"div.rn.{lane.Op} {share}, {dy}, {nf};");
                    b.L(lane.Add(acc, acc, share));
                }

                b.Label($"BB{tag}_NEXT");
                b.L($"add.u32 {bb}, {bb}, 1;");
                b.L($"bra BB{tag};");
                b.Label($"BB{tag}_END");
                b.Label($"BA{tag}_NEXT");
                b.L($"add.u32 {a}, {a}, 1;");
                b.L($"bra BA{tag};");
                b.Label($"BA{tag}_END");
                return acc;
            }

            case "adaptive_maxpool_bwd":
            case "adaptive_avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih: over the output rows and columns whose window holds the pixel.
                bool max = name == "adaptive_maxpool_bwd";
                string H = p[1], W = p[2], OH = p[3], OW = p[4], ih = b.R(), ch = b.R(), acc = lane.NewReg(b), oh = b.R(), ow = b.R(), h0 = b.R(), h1 = b.R(), w0 = b.R(), w1 = b.R(), t = b.R(), pa = b.P(), pb = b.P(), me = b.R();
                RemU(b, ih, r, H);
                DivU(b, ch, r, H);
                b.L($"mad.lo.u32 {me}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {me}, {me}, {W}, {c};");
                b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
                b.L($"mov.u32 {oh}, 0;");
                b.Label($"DA{tag}");
                b.L($"setp.ge.u32 {pa}, {oh}, {OH};");
                b.L($"@{pa} bra DA{tag}_END;");
                b.L($"mul.lo.u32 {h0}, {oh}, {H};"); DivU(b, h0, h0, OH);
                b.L($"add.u32 {t}, {oh}, 1;"); b.L($"mul.lo.u32 {t}, {t}, {H};"); b.L($"add.u32 {t}, {t}, {OH};"); b.L($"sub.u32 {t}, {t}, 1;"); DivU(b, h1, t, OH);
                b.L($"setp.lt.u32 {pa}, {ih}, {h0};");
                b.L($"@{pa} bra DA{tag}_NEXT;");
                b.L($"setp.ge.u32 {pa}, {ih}, {h1};");
                b.L($"@{pa} bra DA{tag}_NEXT;");
                b.L($"mov.u32 {ow}, 0;");
                b.Label($"DB{tag}");
                b.L($"setp.ge.u32 {pb}, {ow}, {OW};");
                b.L($"@{pb} bra DB{tag}_END;");
                b.L($"mul.lo.u32 {w0}, {ow}, {W};"); DivU(b, w0, w0, OW);
                b.L($"add.u32 {t}, {ow}, 1;"); b.L($"mul.lo.u32 {t}, {t}, {W};"); b.L($"add.u32 {t}, {t}, {OW};"); b.L($"sub.u32 {t}, {t}, 1;"); DivU(b, w1, t, OW);
                b.L($"setp.lt.u32 {pb}, {c}, {w0};");
                b.L($"@{pb} bra DB{tag}_NEXT;");
                b.L($"setp.ge.u32 {pb}, {c}, {w1};");
                b.L($"@{pb} bra DB{tag}_NEXT;");
                string dyIdx = b.R();
                b.L($"mad.lo.u32 {dyIdx}, {ch}, {OH}, {oh};");
                b.L($"mad.lo.u32 {dyIdx}, {dyIdx}, {OW}, {ow};");
                string dy = Load(dyIdx);
                if (!max)
                {
                    string area = b.R(), af = lane.NewReg(b), share = lane.NewReg(b);
                    b.L($"sub.u32 {area}, {h1}, {h0};"); b.L($"sub.u32 {t}, {w1}, {w0};"); b.L($"mul.lo.u32 {area}, {area}, {t};");
                    b.L($"cvt.rn.{lane.Op}.u32 {af}, {area};");
                    b.L($"div.rn.{lane.Op} {share}, {dy}, {af};");
                    b.L(lane.Add(acc, acc, share));
                }
                else
                {
                    // The window's first-ranked of the forward input, found once for all its pixels (LayoutWindowArgmax, in aux).
                    string chosen = Index(aux, dyIdx), pChosen = b.P();
                    b.L($"setp.eq.u32 {pChosen}, {chosen}, {me};");
                    b.L($"@{pChosen} {lane.Add(acc, acc, dy)}");
                }

                b.Label($"DB{tag}_NEXT");
                b.L($"add.u32 {ow}, {ow}, 1;");
                b.L($"bra DB{tag};");
                b.Label($"DB{tag}_END");
                b.Label($"DA{tag}_NEXT");
                b.L($"add.u32 {oh}, {oh}, 1;");
                b.L($"bra DA{tag};");
                b.Label($"DA{tag}_END");
                return acc;
            }

            case "adaptive_maxpool":
            case "adaptive_avgpool":
            {
                bool max = name == "adaptive_maxpool";
                string H = p[1], W = p[2], OH = p[3], OW = p[4], oh = b.R(), ch = b.R(), h0 = b.R(), h1 = b.R(), w0 = b.R(), w1 = b.R(), ih = b.R(), iw = b.R(), pa = b.P(), pb = b.P(), src = b.R(), t = b.R();
                string acc = lane.NewReg(b);
                RemU(b, oh, r, OH);
                DivU(b, ch, r, OH);
                // [⌊i·n/o⌋, ⌈(i + 1)·n/o⌉)
                b.L($"mul.lo.u32 {h0}, {oh}, {H};");
                DivU(b, h0, h0, OH);
                b.L($"add.u32 {t}, {oh}, 1;");
                b.L($"mul.lo.u32 {t}, {t}, {H};");
                b.L($"add.u32 {t}, {t}, {OH};");
                b.L($"sub.u32 {t}, {t}, 1;");
                DivU(b, h1, t, OH);
                b.L($"mul.lo.u32 {w0}, {c}, {W};");
                DivU(b, w0, w0, OW);
                b.L($"add.u32 {t}, {c}, 1;");
                b.L($"mul.lo.u32 {t}, {t}, {W};");
                b.L($"add.u32 {t}, {t}, {OW};");
                b.L($"sub.u32 {t}, {t}, 1;");
                DivU(b, w1, t, OW);
                b.L($"mov.{lane.Reg} {acc}, {(max ? ReduceIdentity(lane, "max") : lane.Zero)};");
                b.L($"mov.u32 {ih}, {h0};");
                b.Label($"AH{tag}");
                b.L($"setp.ge.u32 {pa}, {ih}, {h1};");
                b.L($"@{pa} bra AH{tag}_END;");
                b.L($"mov.u32 {iw}, {w0};");
                b.Label($"AW{tag}");
                b.L($"setp.ge.u32 {pb}, {iw}, {w1};");
                b.L($"@{pb} bra AW{tag}_END;");
                b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {src}, {src}, {W}, {iw};");
                string xv = Load(src);
                if (max) First(b, lane, acc, acc, xv); else b.L(lane.Add(acc, acc, xv));
                b.L($"add.u32 {iw}, {iw}, 1;");
                b.L($"bra AW{tag};");
                b.Label($"AW{tag}_END");
                b.L($"add.u32 {ih}, {ih}, 1;");
                b.L($"bra AH{tag};");
                b.Label($"AH{tag}_END");
                if (max) return acc;
                string area = b.R(), af = lane.NewReg(b), mean = lane.NewReg(b);
                b.L($"sub.u32 {area}, {h1}, {h0};");
                b.L($"sub.u32 {t}, {w1}, {w0};");
                b.L($"mul.lo.u32 {area}, {area}, {t};");
                b.L($"cvt.rn.{lane.Op}.u32 {af}, {area};");
                b.L($"div.rn.{lane.Op} {mean}, {acc}, {af};");
                return mean;
            }

            default: throw new NotSupportedException($"layout operation {name} on the device");
        }
    }

    // One axis of a bilinear resize: (i0, i1, fraction) for output index o over `inp` input and `outp` output pixels.
    // A bilinear axis as integers (LayoutArgs.BilinearAt): t = max(o·α + β, 0), i0 = t / δ; past the input's
    // last position both rows the last and the weight 0; f = (t − i0·δ) / δ in the lane (one rounding), g = 1 − f.
    private static (string I0, string I1, string F, string G) BilinearAxis(PtxBuilder b, Lane lane, string o, string inp, string alpha, string beta, string delta)
    {
        string t = b.R(), i0 = b.R(), i1 = b.R(), last = b.R(), rem = b.R(), past = b.P();
        b.L($"mad.lo.s32 {t}, {o}, {alpha}, {beta};");
        b.L($"max.s32 {t}, {t}, 0;");
        DivU(b, i0, t, delta);
        b.L($"mul.lo.u32 {rem}, {i0}, {delta};");
        b.L($"sub.u32 {rem}, {t}, {rem};");
        b.L($"sub.u32 {last}, {inp}, 1;");
        b.L($"setp.ge.u32 {past}, {i0}, {last};");
        b.L($"min.u32 {i0}, {i0}, {last};");
        b.L($"add.u32 {i1}, {i0}, 1;");
        b.L($"min.u32 {i1}, {i1}, {last};");
        b.L($"selp.u32 {rem}, 0, {rem}, {past};");
        string rf = lane.NewReg(b), df = lane.NewReg(b), f = lane.NewReg(b), g = lane.NewReg(b);
        b.L($"cvt.rn.{lane.Op}.u32 {rf}, {rem};");
        b.L($"cvt.rn.{lane.Op}.u32 {df}, {delta};");
        b.L($"div.rn.{lane.Op} {f}, {rf}, {df};");
        b.L($"sub.rn.{lane.Op} {g}, {lane.Literal(1)}, {f};");
        return (i0, i1, f, g);
    }

    // A nearest axis (LayoutArgs.NearestAt): min(max(2(o·α + β) + δ, 0) / 2δ, in − 1), signed (β may be negative).
    private static string NearestAxis(PtxBuilder b, string o, string inp, string alpha, string beta, string delta)
    {
        string i = b.R(), d2 = b.R(), last = b.R();
        b.L($"mad.lo.s32 {i}, {o}, {alpha}, {beta};");
        b.L($"shl.b32 {i}, {i}, 1;");
        b.L($"add.s32 {i}, {i}, {delta};");
        b.L($"max.s32 {i}, {i}, 0;");
        b.L($"shl.b32 {d2}, {delta}, 1;");
        DivU(b, i, i, d2);
        b.L($"sub.u32 {last}, {inp}, 1;");
        b.L($"min.u32 {i}, {i}, {last};");
        return i;
    }

    // The outputs [lo, hi] (s32) whose source rows include input row i (LayoutArgs.InterpolateSources).
    private static (string Lo, string Hi) SourceRange(PtxBuilder b, bool bilinear, string i, string inp, string outp, string alpha, string beta, string delta)
    {
        // ceil(n / α) of a signed n (α > 0): n ≥ 0 ? (n + α − 1) / α : −(−n / α).
        string CeilDiv(string n)
        {
            string pos = b.R(), neg = b.R(), q = b.R(), pn = b.P(), am1 = b.R();
            b.L($"sub.s32 {am1}, {alpha}, 1;");
            b.L($"add.s32 {pos}, {n}, {am1};");
            b.L($"div.s32 {pos}, {pos}, {alpha};");
            b.L($"neg.s32 {neg}, {n};");
            b.L($"div.s32 {neg}, {neg}, {alpha};");
            b.L($"neg.s32 {neg}, {neg};");
            b.L($"setp.ge.s32 {pn}, {n}, 0;");
            b.L($"selp.b32 {q}, {pos}, {neg}, {pn};");
            return q;
        }

        string lo = b.R(), hi = b.R(), n = b.R(), m = b.R(), last = b.R(), om1 = b.R(), pz = b.P(), pl = b.P(), pa = b.P();
        b.L($"setp.eq.u32 {pa}, {alpha}, 0;");
        // a safe divisor where α is 0 (every output then; selected below)
        string a1 = b.R();
        b.L($"selp.u32 {a1}, 1, {alpha}, {pa};");
        alpha = a1;
        if (!bilinear)
        {
            // nearest: the floor of the map (2α, 2β + δ, 2δ)
            string a2 = b.R(), b2 = b.R(), d2 = b.R();
            b.L($"shl.b32 {a2}, {alpha}, 1;");
            b.L($"shl.b32 {b2}, {beta}, 1;");
            b.L($"add.s32 {b2}, {b2}, {delta};");
            b.L($"shl.b32 {d2}, {delta}, 1;");
            alpha = a2; beta = b2; delta = d2;
        }

        // The first output reading position i (its first position i − 1 or i for bilinear, i for nearest), 0 for the first.
        b.L($"sub.s32 {n}, {i}, {(bilinear ? 1 : 0)};");
        b.L($"mul.lo.s32 {n}, {n}, {delta};");
        b.L($"sub.s32 {n}, {n}, {beta};");
        string c0 = CeilDiv(n);
        b.L($"setp.le.u32 {pz}, {i}, {(bilinear ? 1 : 0)};");
        b.L($"selp.b32 {lo}, 0, {c0}, {pz};");

        b.L($"add.s32 {m}, {i}, 1;");
        b.L($"mul.lo.s32 {m}, {m}, {delta};");
        b.L($"sub.s32 {m}, {m}, {beta};");
        b.L($"sub.s32 {hi}, {CeilDiv(m)}, 1;");
        b.L($"sub.u32 {last}, {inp}, 1;");
        b.L($"sub.u32 {om1}, {outp}, 1;");
        b.L($"setp.eq.u32 {pl}, {i}, {last};");
        b.L($"selp.b32 {hi}, {om1}, {hi}, {pl};");
        b.L($"max.s32 {lo}, {lo}, 0;");
        b.L($"min.s32 {hi}, {hi}, {om1};");
        b.L($"selp.b32 {lo}, 0, {lo}, {pa};");
        b.L($"selp.b32 {hi}, {om1}, {hi}, {pa};");
        return (lo, hi);
    }

    // The weight output o of an axis gives input position i (LayoutArgs.InterpolateSources' outputs): nearest 1 where it reads
    // i; bilinear g where its first row is i, else f where its second is. Returns whether it reads i at all.
    private static (string Has, string W) AxisWeight(PtxBuilder b, Lane lane, bool bilinear, string o, string i, string inp, string alpha, string beta, string delta)
    {
        string has = b.P(), w = lane.NewReg(b);
        if (!bilinear)
        {
            b.L($"setp.eq.u32 {has}, {NearestAxis(b, o, inp, alpha, beta, delta)}, {i};");
            b.L($"mov.{lane.Reg} {w}, {lane.Literal(1)};");
            return (has, w);
        }

        var (i0, i1, f, g) = BilinearAxis(b, lane, o, inp, alpha, beta, delta);
        string first = b.P(), second = b.P();
        b.L($"setp.eq.u32 {first}, {i0}, {i};");
        b.L($"setp.eq.u32 {second}, {i1}, {i};");
        b.L($"or.pred {has}, {first}, {second};");
        b.L($"selp.{lane.Op} {w}, {g}, {f}, {first};");
        return (has, w);
    }

    // The input positions window o covers along an axis of n (LayoutArgs.Window): min(o·s − before + k, n) − max(o·s − before, 0).
    private static string WindowSpan(PtxBuilder b, string o, string s, string before, string k, string n)
    {
        string from = b.R(), to = b.R();
        b.L($"mul.lo.s32 {from}, {o}, {s};");
        b.L($"sub.s32 {from}, {from}, {before};");
        b.L($"add.s32 {to}, {from}, {k};");
        b.L($"min.s32 {to}, {to}, {n};");
        b.L($"max.s32 {from}, {from}, 0;");
        b.L($"sub.s32 {to}, {to}, {from};");
        return to;
    }

    /// <summary>The bits of count values of src (format f, from value 0) placed at value offset `offset` of dst, which starts
    /// zeroed over the region: a thread per value, OR-ed into the word (sub-word formats at offsets that do not end on a byte).</summary>
    public static string PlaceBits(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"place_bits_{f.Name}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), i = b.R(), src = Param64(b, "p_src"), dst = Param64(b, "p_dst"), count = Param32(b, "p_count"), offset = Param32(b, "p_offset"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {i}, {count};");
        b.L($"@{pEnd} bra DONE;");
        // The value's bits out of its source word.
        string sBits = b.R(), sWord = b.R(), sShift = b.R(), sAddr = b.RD(), w = b.R(), bits = b.R();
        b.L($"mul.lo.u32 {sBits}, {i}, {f.Bits};");
        b.L($"shr.u32 {sWord}, {sBits}, 5;");
        b.L($"and.b32 {sShift}, {sBits}, 31;");
        b.L($"mul.wide.u32 {sAddr}, {sWord}, 4;");
        b.L($"add.u64 {sAddr}, {src}, {sAddr};");
        b.L($"ld.global.b32 {w}, [{sAddr}];");
        b.L($"bfe.u32 {bits}, {w}, {sShift}, {f.Bits};");
        // Into the destination word at the value's bit position (the region was zeroed).
        string dBits = b.R(), dWord = b.R(), dShift = b.R(), dAddr = b.RD(), shifted = b.R(), old = b.R(), dIdx = b.R();
        b.L($"add.u32 {dIdx}, {offset}, {i};");
        b.L($"mul.lo.u32 {dBits}, {dIdx}, {f.Bits};");
        b.L($"shr.u32 {dWord}, {dBits}, 5;");
        b.L($"and.b32 {dShift}, {dBits}, 31;");
        b.L($"shl.b32 {shifted}, {bits}, {dShift};");
        b.L($"mul.wide.u32 {dAddr}, {dWord}, 4;");
        b.L($"add.u64 {dAddr}, {dst}, {dAddr};");
        b.L($"atom.global.or.b32 {old}, [{dAddr}], {shifted};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_src, .param .u64 p_dst, .param .u32 p_count, .param .u32 p_offset");
    }

    // ---------------------------------------------------------------- the ordering's counts (GpuOps.OrderByTarget)

    /// <summary>counts[index[i]] += 1 over n sources (integer atomics: exact); an index outside [0, targets) counts nowhere.</summary>
    public static string ScatterCount(Gpu gpu, out string entry)
    {
        entry = "scatter_count";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), i = b.R(), n = Param32(b, "p_n"), targets = Param32(b, "p_targets"), idx = Param64(b, "p_idx"), counts = Param64(b, "p_counts"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {i}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string a = b.RD(), target = b.R(), ca = b.RD(), old = b.R();
        b.L($"mul.wide.u32 {a}, {i}, 4;");
        b.L($"add.u64 {a}, {idx}, {a};");
        b.L($"ld.global.s32 {target}, [{a}];");
        b.L($"setp.ge.u32 {pEnd}, {target}, {targets};");            // unsigned: a negative index too
        b.L($"@{pEnd} bra DONE;");
        b.L($"mul.wide.u32 {ca}, {target}, 4;");
        b.L($"add.u64 {ca}, {counts}, {ca};");
        b.L($"atom.global.add.u32 {old}, [{ca}], 1;");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_idx, .param .u64 p_counts, .param .u32 p_n, .param .u32 p_targets");
    }
}
