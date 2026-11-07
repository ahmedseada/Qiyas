// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's layout operations (Layout.cs): a thread per output word computes each of its values from the output
// index by the operation's rule (a gather; the sums — scatter-add, col2im — in the sum lane, converted on the store),
// so every word is written by one thread and no format needs atomics. Parameters: the input, the index array, an
// auxiliary buffer (scatter-add's row offsets and ordered sources), the output, the counts and twelve integers in the
// operation's order (LayoutArgs).
namespace Adad;

internal static partial class Ptx
{
    private const string LayoutParams = ".param .u64 p_x, .param .u64 p_x2, .param .u64 p_idx, .param .u64 p_aux, .param .u64 p_out, .param .u32 p_count, .param .u32 p_rows, .param .u32 p_cols, "
        + ".param .u32 p_p0, .param .u32 p_p1, .param .u32 p_p2, .param .u32 p_p3, .param .u32 p_p4, .param .u32 p_p5, .param .u32 p_p6, .param .u32 p_p7, .param .u32 p_p8, .param .u32 p_p9, .param .u32 p_p10, .param .u32 p_p11";

    public static string Layout(Gpu gpu, Fmt f, Lane lane, string name, out string entry) => Layout(gpu, f, lane, name, false, out entry);

    /// <summary>perValue (a packed 32-bit word of 2, 4, 8, 16 or 32 values; GpuOps.LayoutPerValue): a thread per output
    /// value, not per word, the word's values joined by the word's lanes (shfl.sync, a group of Per lanes a word) and
    /// stored by its first: Per times the threads for an output too small to fill the device (an adaptive pool's few
    /// windows).</summary>
    public static string Layout(Gpu gpu, Fmt f, Lane lane, string name, bool perValue, out string entry)
    {
        if (perValue && !PerValueFits(f)) throw new NotSupportedException($"{f.Name}: a thread per value takes a word of 2 to 32 whole values");
        if (perValue) return LayoutPerValue(gpu, f, lane, name, out entry);
        entry = $"layout_{name}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), word = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {word}, {bid}, 256, {tid};");
        string x = Param64(b, "p_x"), x2 = Param64(b, "p_x2"), idx = Param64(b, "p_idx"), aux = Param64(b, "p_aux"), outp = Param64(b, "p_out");
        string count = Param32(b, "p_count"), cols = Param32(b, "p_cols");
        var p = new string[12];
        for (int k = 0; k < 12; k++) p[k] = Param32(b, $"p_p{k}");
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
            b.L($"div.u32 {r}, {v}, {cols};");
            b.L($"mul.lo.u32 {t}, {r}, {cols};");
            b.L($"sub.u32 {c}, {v}, {t};");
            string value = LayoutValue(b, f, lane, name, x, x2, idx, aux, p, cols, v, r, c, i);
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

    /// <summary>A format the per-value layout kernel packs: whole values filling a 32-bit word, 2 to 32 of them.</summary>
    public static bool PerValueFits(Fmt f) => !f.IsBlock && f.WordBits == 32 && f.Per >= 2 && f.Per * f.Bits == 32;

    private static string LayoutPerValue(Gpu gpu, Fmt f, Lane lane, string name, out string entry)
    {
        entry = $"layout_{name}_{f.Name}_{lane.Name}_v";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), at = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {at}, {bid}, 256, {tid};");
        string x = Param64(b, "p_x"), x2 = Param64(b, "p_x2"), idx = Param64(b, "p_idx"), aux = Param64(b, "p_aux"), outp = Param64(b, "p_out");
        string count = Param32(b, "p_count"), cols = Param32(b, "p_cols");
        var p = new string[12];
        for (int k = 0; k < 12; k++) p[k] = Param32(b, $"p_p{k}");
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
        b.L($"div.u32 {r}, {v}, {cols};");
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

    /// <summary>The max-pool backward's first largest of every window of the forward input (p_x2), a thread per window
    /// (p_count windows, in the output's order): its index into the input, 0xFFFFFFFF for a window with no pixel, into
    /// p_out (u32). The backward then reads each window's choice (LayoutValue) instead of scanning every window a pixel
    /// lies in; compared in the backward's lane, as the scan was.</summary>
    public static string LayoutWindowArgmax(Gpu gpu, Fmt f, Lane lane, string name, out string entry)
    {
        if (name is not ("maxpool_bwd" or "adaptive_maxpool_bwd")) throw new NotSupportedException($"window first largest of {name}");
        bool adaptive = name == "adaptive_maxpool_bwd";
        entry = $"layout_argmax_{name}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), i = b.R(), pd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        string x2 = Param64(b, "p_x2"), outp = Param64(b, "p_out"), count = Param32(b, "p_count");
        var p = new string[12];
        for (int k = 0; k < 12; k++) p[k] = Param32(b, $"p_p{k}");
        b.L($"setp.ge.u32 {pd}, {i}, {count};");
        b.L($"@{pd} bra DONE;");
        string H = p[1], W = p[2], OH = adaptive ? p[3] : PoolHeight(b, p, false), OW = adaptive ? p[4] : PoolWidth(b, p, false);
        string ow = b.R(), oh = b.R(), ch = b.R(), t = b.R(), h0 = b.R(), kh = b.R(), w0 = b.R(), kw = b.R();
        b.L($"rem.u32 {ow}, {i}, {OW};");
        b.L($"div.u32 {t}, {i}, {OW};");
        b.L($"rem.u32 {oh}, {t}, {OH};");
        b.L($"div.u32 {ch}, {t}, {OH};");
        if (adaptive)
        {
            // [floor(i*n/o), ceil((i + 1)*n/o)) along each axis.
            void Span(string o, string n, string outs, string start, string len)
            {
                string e = b.R();
                b.L($"mul.lo.u32 {start}, {o}, {n};");
                b.L($"div.u32 {start}, {start}, {outs};");
                b.L($"add.u32 {e}, {o}, 1;");
                b.L($"mul.lo.u32 {e}, {e}, {n};");
                b.L($"add.u32 {e}, {e}, {outs};");
                b.L($"sub.u32 {e}, {e}, 1;");
                b.L($"div.u32 {e}, {e}, {outs};");
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
        // The first largest in row-major order: a later value takes it only when strictly larger.
        string xv = Codec.UnpackAt(b, f, x2, cand, lane), pFirst = b.P(), pGt = b.P();
        b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
        b.L($"setp.gt.{lane.Op} {pGt}, {xv}, {best};");
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

    // The value of output (v; row r, column c) as a lane register.
    private static string LayoutValue(PtxBuilder b, Fmt f, Lane lane, string name, string x, string x2, string idx, string aux, string[] p, string cols, string v, string r, string c, int tag)
    {
        string Load(string index) => Codec.UnpackAt(b, f, x, index, lane);
        string Load2(string index) => Codec.UnpackAt(b, f, x2, index, lane);
        string Gt(string a, string z) { string pr = b.P(); b.L($"setp.gt.{(lane.IsFloat ? lane.Op : lane.Op)} {pr}, {a}, {z};"); return pr; }
        string Index(string ptr, string i) { string a = b.RD(), o = b.R(); b.L($"mul.wide.u32 {a}, {i}, 4;"); b.L($"add.u64 {a}, {ptr}, {a};"); b.L($"ld.global.s32 {o}, [{a}];"); return o; }
        string Zero() { string z = lane.NewReg(b); b.L($"mov.{lane.Reg} {z}, {lane.Zero};"); return z; }
        string Sel(string pred, string a, string z) { string d = lane.NewReg(b); b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {d}, {a}, {z}, {pred};"); return d; }
        switch (name)
        {
            case "strided":
            {
                // v = ((i0·d1 + i1)·d2 + i2)·d3 + i3; source = base + Σ i_d · s_d.
                string i3 = b.R(), t = b.R(), i2 = b.R(), i1 = b.R(), i0 = b.R(), src = b.R(), q = b.R();
                b.L($"rem.u32 {i3}, {v}, {p[3]};");
                b.L($"div.u32 {t}, {v}, {p[3]};");
                b.L($"rem.u32 {i2}, {t}, {p[2]};");
                b.L($"div.u32 {t}, {t}, {p[2]};");
                b.L($"rem.u32 {i1}, {t}, {p[1]};");
                b.L($"div.u32 {i0}, {t}, {p[1]};");
                b.L($"mad.lo.u32 {src}, {i0}, {p[4]}, {p[8]};");
                b.L($"mad.lo.u32 {src}, {i1}, {p[5]}, {src};");
                b.L($"mad.lo.u32 {src}, {i2}, {p[6]}, {src};");
                b.L($"mad.lo.u32 {src}, {i3}, {p[7]}, {src};");
                _ = q;
                return Load(src);
            }

            case "one_hot":
            {
                string target = Index(idx, r), pe = b.P(), one = lane.NewReg(b);
                b.L($"setp.eq.u32 {pe}, {target}, {c};");
                b.L($"mov.{lane.Reg} {one}, {lane.Literal(1)};");
                return Sel(pe, one, Zero());
            }

            case "gather_rows":
            {
                string row = Index(idx, r), src = b.R();
                b.L($"mad.lo.u32 {src}, {row}, {cols}, {c};");
                return Load(src);
            }

            case "gather_cols":
            {
                string col = Index(idx, c), src = b.R();
                b.L($"mad.lo.u32 {src}, {r}, {p[1]}, {col};");
                return Load(src);
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
                // r = (ch·kh + a)·kw + bb; c = oh·OW + ow; source (ch, oh·sh − ph + a·dh, ow·sw − pw + bb·dw) or 0.
                string kw = p[4], kh = p[3], H = p[1], W = p[2], ow = b.R(), oh = b.R(), bb = b.R(), a = b.R(), ch = b.R(), t = b.R(), ih = b.R(), iw = b.R(), pin = b.P(), p2 = b.P(), src = b.R(), ow2 = b.R();
                string OW = OutWidth(b, p, false);
                b.L($"rem.u32 {ow}, {c}, {OW};");
                b.L($"div.u32 {oh}, {c}, {OW};");
                b.L($"rem.u32 {bb}, {r}, {kw};");
                b.L($"div.u32 {t}, {r}, {kw};");
                b.L($"rem.u32 {a}, {t}, {kh};");
                b.L($"div.u32 {ch}, {t}, {kh};");
                b.L($"mul.lo.u32 {ih}, {oh}, {p[5]};");
                b.L($"sub.u32 {ih}, {ih}, {p[7]};");
                b.L($"mad.lo.u32 {ih}, {a}, {p[9]}, {ih};");
                b.L($"mul.lo.u32 {iw}, {ow}, {p[6]};");
                b.L($"sub.u32 {iw}, {iw}, {p[8]};");
                b.L($"mad.lo.u32 {iw}, {bb}, {p[10]}, {iw};");
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
                // Output (ch, ih, iw = c), r = ch·H + ih: Σ over (a, bb) with (ih + ph − a·dh) = oh·sh, (iw + pw − bb·dw) = ow·sw in range.
                string H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8], dh = p[9], dw = p[10];
                string OW = OutWidth(b, p, false), OH = OutWidth(b, p, true);
                string ih = b.R(), ch = b.R(), acc = Zero(), a = b.R(), bb = b.R(), th = b.R(), tw = b.R(), oh = b.R(), ow = b.R(), rem = b.R(), pa = b.P(), pb = b.P(), pc = b.P(), src = b.R(), plane = b.R(), rowIdx = b.R();
                b.L($"rem.u32 {ih}, {r}, {H};");
                b.L($"div.u32 {ch}, {r}, {H};");
                b.L($"mul.lo.u32 {plane}, {OH}, {OW};");
                b.L($"mov.u32 {a}, 0;");
                b.Label($"CA{tag}");
                b.L($"setp.ge.u32 {pa}, {a}, {kh};");
                b.L($"@{pa} bra CA{tag}_END;");
                b.L($"add.u32 {th}, {ih}, {ph};");
                b.L($"mul.lo.u32 {rem}, {a}, {dh};");
                b.L($"sub.s32 {th}, {th}, {rem};");                      // may go negative: a signed test
                b.L($"setp.lt.s32 {pb}, {th}, 0;");
                b.L($"@{pb} bra CA{tag}_NEXT;");
                b.L($"rem.u32 {rem}, {th}, {sh};");
                b.L($"setp.ne.u32 {pb}, {rem}, 0;");
                b.L($"@{pb} bra CA{tag}_NEXT;");
                b.L($"div.u32 {oh}, {th}, {sh};");
                b.L($"setp.ge.u32 {pb}, {oh}, {OH};");
                b.L($"@{pb} bra CA{tag}_NEXT;");
                b.L($"mov.u32 {bb}, 0;");
                b.Label($"CB{tag}");
                b.L($"setp.ge.u32 {pc}, {bb}, {kw};");
                b.L($"@{pc} bra CB{tag}_END;");
                b.L($"add.u32 {tw}, {c}, {pw};");
                b.L($"mul.lo.u32 {rem}, {bb}, {dw};");
                b.L($"sub.s32 {tw}, {tw}, {rem};");
                b.L($"setp.lt.s32 {pc}, {tw}, 0;");
                b.L($"@{pc} bra CB{tag}_NEXT;");
                b.L($"rem.u32 {rem}, {tw}, {sw};");
                b.L($"setp.ne.u32 {pc}, {rem}, 0;");
                b.L($"@{pc} bra CB{tag}_NEXT;");
                b.L($"div.u32 {ow}, {tw}, {sw};");
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
                // r = ch·OH + oh, c = ow: ih = min(H − 1, ((2·oh + 1)·H) / (2·OH)), iw likewise.
                string H = p[1], W = p[2], OH = p[3], OW = p[4], oh = b.R(), ch = b.R(), ih = b.R(), iw = b.R(), t = b.R(), hm1 = b.R(), wm1 = b.R(), src = b.R();
                b.L($"rem.u32 {oh}, {r}, {OH};");
                b.L($"div.u32 {ch}, {r}, {OH};");
                b.L($"shl.b32 {t}, {oh}, 1;");
                b.L($"add.u32 {t}, {t}, 1;");
                b.L($"mul.lo.u32 {t}, {t}, {H};");
                b.L($"shl.b32 {ih}, {OH}, 1;");
                b.L($"div.u32 {ih}, {t}, {ih};");
                b.L($"sub.u32 {hm1}, {H}, 1;");
                b.L($"min.u32 {ih}, {ih}, {hm1};");
                b.L($"shl.b32 {t}, {c}, 1;");
                b.L($"add.u32 {t}, {t}, 1;");
                b.L($"mul.lo.u32 {t}, {t}, {W};");
                b.L($"shl.b32 {iw}, {OW}, 1;");
                b.L($"div.u32 {iw}, {t}, {iw};");
                b.L($"sub.u32 {wm1}, {W}, 1;");
                b.L($"min.u32 {iw}, {iw}, {wm1};");
                b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {src}, {src}, {W}, {iw};");
                return Load(src);
            }

            case "interpolate_bilinear":
            {
                // sy = max((oh + ½)·H/OH − ½, 0) in the lane; y0 = min(floor sy, H − 1), y1 = min(y0 + 1, H − 1); x likewise.
                string H = p[1], W = p[2], OH = p[3], OW = p[4], oh = b.R(), ch = b.R();
                b.L($"rem.u32 {oh}, {r}, {OH};");
                b.L($"div.u32 {ch}, {r}, {OH};");
                var (y0, y1, fy) = Axis(b, lane, oh, H, OH);
                var (x0, x1, fx) = Axis(b, lane, c, W, OW);
                string gy = lane.NewReg(b), gx = lane.NewReg(b), one = lane.Literal(1);
                b.L($"sub.rn.{lane.Op} {gy}, {one}, {fy};");
                b.L($"sub.rn.{lane.Op} {gx}, {one}, {fx};");
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
                // r = ch·OH + oh, c = ow; the window over (a, bb) with the pads and dilations (1 for the pools).
                bool max = name == "maxpool", conv = name == "depthwise_conv";
                string H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                string dh = conv ? p[9] : "1", dw = conv ? p[10] : "1";
                string OH = PoolHeight(b, p, conv), oh = b.R(), ch = b.R(), a = b.R(), bb = b.R(), ih = b.R(), iw = b.R(), pa = b.P(), pb = b.P(), src = b.R(), n = b.R();
                string acc = lane.NewReg(b);
                b.L($"rem.u32 {oh}, {r}, {OH};");
                b.L($"div.u32 {ch}, {r}, {OH};");
                b.L($"mov.{lane.Reg} {acc}, {(max ? ReduceIdentity(lane, "max") : lane.Zero)};");
                b.L($"mov.u32 {n}, 0;");
                b.L($"mov.u32 {a}, 0;");
                b.Label($"PA{tag}");
                b.L($"setp.ge.u32 {pa}, {a}, {kh};");
                b.L($"@{pa} bra PA{tag}_END;");
                b.L($"mul.lo.u32 {ih}, {oh}, {sh};");
                b.L($"sub.u32 {ih}, {ih}, {ph};");
                b.L($"mad.lo.u32 {ih}, {a}, {dh}, {ih};");
                b.L($"setp.ge.u32 {pa}, {ih}, {H};");                      // unsigned: negative rows are out of range
                b.L($"@{pa} bra PA{tag}_NEXT;");
                b.L($"mov.u32 {bb}, 0;");
                b.Label($"PB{tag}");
                b.L($"setp.ge.u32 {pb}, {bb}, {kw};");
                b.L($"@{pb} bra PB{tag}_END;");
                b.L($"mul.lo.u32 {iw}, {c}, {sw};");
                b.L($"sub.u32 {iw}, {iw}, {pw};");
                b.L($"mad.lo.u32 {iw}, {bb}, {dw}, {iw};");
                b.L($"setp.ge.u32 {pb}, {iw}, {W};");
                b.L($"@{pb} bra PB{tag}_NEXT;");
                b.L($"mad.lo.u32 {src}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {src}, {src}, {W}, {iw};");
                string xv = Load(src);
                if (max) b.L($"max.{lane.Op} {acc}, {acc}, {xv};");
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
                if (conv) return acc;
                string pNone = b.P();
                b.L($"setp.eq.u32 {pNone}, {n}, 0;");
                if (max) return Sel(pNone, Zero(), acc);
                string nf = lane.NewReg(b), mean = lane.NewReg(b);
                b.L($"cvt.rn.{lane.Op}.u32 {nf}, {n};");
                b.L($"div.rn.{lane.Op} {mean}, {acc}, {nf};");
                return Sel(pNone, Zero(), mean);
            }

            case "maxpool_bwd":
            case "avgpool_bwd":
            {
                // Output (ch, ih, iw = c), r = ch·H + ih; x = dy (C × OH × OW), x2 = the forward input (maxpool).
                bool max = name == "maxpool_bwd";
                string H = p[1], W = p[2], kh = p[3], kw = p[4], sh = p[5], sw = p[6], ph = p[7], pw = p[8];
                string OH = PoolHeight(b, p, false), OW = PoolWidth(b, p, false);
                string ih = b.R(), ch = b.R(), acc = lane.NewReg(b), a = b.R(), bb = b.R(), th = b.R(), tw = b.R(), oh = b.R(), ow = b.R(), rem = b.R(), pa = b.P(), pb = b.P(), dyIdx = b.R(), me = b.R();
                b.L($"rem.u32 {ih}, {r}, {H};");
                b.L($"div.u32 {ch}, {r}, {H};");
                b.L($"mad.lo.u32 {me}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {me}, {me}, {W}, {c};");
                b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
                b.L($"mov.u32 {a}, 0;");
                b.Label($"BA{tag}");
                b.L($"setp.ge.u32 {pa}, {a}, {kh};");
                b.L($"@{pa} bra BA{tag}_END;");
                b.L($"add.u32 {th}, {ih}, {ph};");
                b.L($"sub.s32 {th}, {th}, {a};");
                b.L($"setp.lt.s32 {pa}, {th}, 0;");
                b.L($"@{pa} bra BA{tag}_NEXT;");
                b.L($"rem.u32 {rem}, {th}, {sh};");
                b.L($"setp.ne.u32 {pa}, {rem}, 0;");
                b.L($"@{pa} bra BA{tag}_NEXT;");
                b.L($"div.u32 {oh}, {th}, {sh};");
                b.L($"setp.ge.u32 {pa}, {oh}, {OH};");
                b.L($"@{pa} bra BA{tag}_NEXT;");
                b.L($"mov.u32 {bb}, 0;");
                b.Label($"BB{tag}");
                b.L($"setp.ge.u32 {pb}, {bb}, {kw};");
                b.L($"@{pb} bra BB{tag}_END;");
                b.L($"add.u32 {tw}, {c}, {pw};");
                b.L($"sub.s32 {tw}, {tw}, {bb};");
                b.L($"setp.lt.s32 {pb}, {tw}, 0;");
                b.L($"@{pb} bra BB{tag}_NEXT;");
                b.L($"rem.u32 {rem}, {tw}, {sw};");
                b.L($"setp.ne.u32 {pb}, {rem}, 0;");
                b.L($"@{pb} bra BB{tag}_NEXT;");
                b.L($"div.u32 {ow}, {tw}, {sw};");
                b.L($"setp.ge.u32 {pb}, {ow}, {OW};");
                b.L($"@{pb} bra BB{tag}_NEXT;");
                b.L($"mad.lo.u32 {dyIdx}, {ch}, {OH}, {oh};");
                b.L($"mad.lo.u32 {dyIdx}, {dyIdx}, {OW}, {ow};");
                string dy = Load(dyIdx);
                if (max)
                {
                    // The window's first largest, found once for all its pixels (LayoutWindowArgmax, in aux).
                    string chosen = Index(aux, dyIdx), pChosen = b.P();
                    b.L($"setp.eq.u32 {pChosen}, {chosen}, {me};");
                    b.L($"@{pChosen} {lane.Add(acc, acc, dy)}");
                }
                else
                {
                    // The window (oh, ow): its valid count (avg).
                    string aa = b.R(), bbb = b.R(), hh = b.R(), ww = b.R(), pc = b.P(), pd = b.P(), cnt = b.R();
                    b.L($"mov.u32 {cnt}, 0;");
                    b.L($"mov.u32 {aa}, 0;");
                    b.Label($"WA{tag}");
                    b.L($"setp.ge.u32 {pc}, {aa}, {kh};");
                    b.L($"@{pc} bra WA{tag}_END;");
                    b.L($"mul.lo.u32 {hh}, {oh}, {sh};");
                    b.L($"sub.u32 {hh}, {hh}, {ph};");
                    b.L($"add.u32 {hh}, {hh}, {aa};");
                    b.L($"setp.ge.u32 {pc}, {hh}, {H};");
                    b.L($"@{pc} bra WA{tag}_NEXT;");
                    b.L($"mov.u32 {bbb}, 0;");
                    b.Label($"WB{tag}");
                    b.L($"setp.ge.u32 {pd}, {bbb}, {kw};");
                    b.L($"@{pd} bra WB{tag}_END;");
                    b.L($"mul.lo.u32 {ww}, {ow}, {sw};");
                    b.L($"sub.u32 {ww}, {ww}, {pw};");
                    b.L($"add.u32 {ww}, {ww}, {bbb};");
                    b.L($"setp.ge.u32 {pd}, {ww}, {W};");
                    b.L($"@{pd} bra WB{tag}_NEXT;");
                    b.L($"add.u32 {cnt}, {cnt}, 1;");
                    b.Label($"WB{tag}_NEXT");
                    b.L($"add.u32 {bbb}, {bbb}, 1;");
                    b.L($"bra WB{tag};");
                    b.Label($"WB{tag}_END");
                    b.Label($"WA{tag}_NEXT");
                    b.L($"add.u32 {aa}, {aa}, 1;");
                    b.L($"bra WA{tag};");
                    b.Label($"WA{tag}_END");
                    string nf = lane.NewReg(b), share = lane.NewReg(b);
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
                b.L($"rem.u32 {ih}, {r}, {H};");
                b.L($"div.u32 {ch}, {r}, {H};");
                b.L($"mad.lo.u32 {me}, {ch}, {H}, {ih};");
                b.L($"mad.lo.u32 {me}, {me}, {W}, {c};");
                b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
                b.L($"mov.u32 {oh}, 0;");
                b.Label($"DA{tag}");
                b.L($"setp.ge.u32 {pa}, {oh}, {OH};");
                b.L($"@{pa} bra DA{tag}_END;");
                b.L($"mul.lo.u32 {h0}, {oh}, {H};"); b.L($"div.u32 {h0}, {h0}, {OH};");
                b.L($"add.u32 {t}, {oh}, 1;"); b.L($"mul.lo.u32 {t}, {t}, {H};"); b.L($"add.u32 {t}, {t}, {OH};"); b.L($"sub.u32 {t}, {t}, 1;"); b.L($"div.u32 {h1}, {t}, {OH};");
                b.L($"setp.lt.u32 {pa}, {ih}, {h0};");
                b.L($"@{pa} bra DA{tag}_NEXT;");
                b.L($"setp.ge.u32 {pa}, {ih}, {h1};");
                b.L($"@{pa} bra DA{tag}_NEXT;");
                b.L($"mov.u32 {ow}, 0;");
                b.Label($"DB{tag}");
                b.L($"setp.ge.u32 {pb}, {ow}, {OW};");
                b.L($"@{pb} bra DB{tag}_END;");
                b.L($"mul.lo.u32 {w0}, {ow}, {W};"); b.L($"div.u32 {w0}, {w0}, {OW};");
                b.L($"add.u32 {t}, {ow}, 1;"); b.L($"mul.lo.u32 {t}, {t}, {W};"); b.L($"add.u32 {t}, {t}, {OW};"); b.L($"sub.u32 {t}, {t}, 1;"); b.L($"div.u32 {w1}, {t}, {OW};");
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
                    // The window's first largest of the forward input, found once for all its pixels (LayoutWindowArgmax, in aux).
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
                b.L($"rem.u32 {oh}, {r}, {OH};");
                b.L($"div.u32 {ch}, {r}, {OH};");
                // [⌊i·n/o⌋, ⌈(i + 1)·n/o⌉)
                b.L($"mul.lo.u32 {h0}, {oh}, {H};");
                b.L($"div.u32 {h0}, {h0}, {OH};");
                b.L($"add.u32 {t}, {oh}, 1;");
                b.L($"mul.lo.u32 {t}, {t}, {H};");
                b.L($"add.u32 {t}, {t}, {OH};");
                b.L($"sub.u32 {t}, {t}, 1;");
                b.L($"div.u32 {h1}, {t}, {OH};");
                b.L($"mul.lo.u32 {w0}, {c}, {W};");
                b.L($"div.u32 {w0}, {w0}, {OW};");
                b.L($"add.u32 {t}, {c}, 1;");
                b.L($"mul.lo.u32 {t}, {t}, {W};");
                b.L($"add.u32 {t}, {t}, {OW};");
                b.L($"sub.u32 {t}, {t}, 1;");
                b.L($"div.u32 {w1}, {t}, {OW};");
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
                if (max) b.L($"max.{lane.Op} {acc}, {acc}, {xv};"); else b.L(lane.Add(acc, acc, xv));
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
    private static (string I0, string I1, string F) Axis(PtxBuilder b, Lane lane, string o, string inp, string outp)
    {
        string of = lane.NewReg(b), inF = lane.NewReg(b), outF = lane.NewReg(b), s = lane.NewReg(b), half = lane.Literal(0.5), zero = lane.Zero;
        b.L($"cvt.rn.{lane.Op}.u32 {of}, {o};");
        b.L($"cvt.rn.{lane.Op}.u32 {inF}, {inp};");
        b.L($"cvt.rn.{lane.Op}.u32 {outF}, {outp};");
        b.L($"add.rn.{lane.Op} {s}, {of}, {half};");
        b.L(lane.Mul(s, s, inF));
        b.L($"div.rn.{lane.Op} {s}, {s}, {outF};");
        b.L($"sub.rn.{lane.Op} {s}, {s}, {half};");
        b.L($"max.{lane.Op} {s}, {s}, {zero};");
        string fl = lane.NewReg(b), i0 = b.R(), i1 = b.R(), m1 = b.R(), fr = lane.NewReg(b), i0f = lane.NewReg(b);
        b.L($"cvt.rmi.{lane.Op}.{lane.Op} {fl}, {s};");
        b.L($"cvt.rzi.u32.{lane.Op} {i0}, {fl};");
        b.L($"sub.u32 {m1}, {inp}, 1;");
        b.L($"min.u32 {i0}, {i0}, {m1};");
        b.L($"add.u32 {i1}, {i0}, 1;");
        b.L($"min.u32 {i1}, {i1}, {m1};");
        b.L($"cvt.rn.{lane.Op}.u32 {i0f}, {i0};");
        b.L($"sub.rn.{lane.Op} {fr}, {s}, {i0f};");
        return (i0, i1, fr);
    }

    // A pool's (dilation 1) or a depthwise convolution's output height / width: C H W kh kw sh sw ph pw [dh dw].
    private static string PoolHeight(PtxBuilder b, string[] p, bool dilated) => PoolSize(b, p[1], p[3], p[5], p[7], dilated ? p[9] : null);
    private static string PoolWidth(PtxBuilder b, string[] p, bool dilated) => PoolSize(b, p[2], p[4], p[6], p[8], dilated ? p[10] : null);

    private static string PoolSize(PtxBuilder b, string size, string k, string s, string pad, string? d)
    {
        string t = b.R(), u = b.R();
        b.L($"shl.b32 {t}, {pad}, 1;");
        b.L($"add.u32 {t}, {t}, {size};");
        b.L($"sub.u32 {u}, {k}, 1;");
        if (d is not null) b.L($"mul.lo.u32 {u}, {u}, {d};");
        b.L($"sub.u32 {t}, {t}, {u};");
        b.L($"sub.u32 {t}, {t}, 1;");
        b.L($"div.u32 {t}, {t}, {s};");
        b.L($"add.u32 {t}, {t}, 1;");
        return t;
    }

    // OH or OW of a convolution's output from the twelve parameters (C H W kh kw sh sw ph pw dh dw).
    private static string OutWidth(PtxBuilder b, string[] p, bool height)
    {
        string size = height ? p[1] : p[2], k = height ? p[3] : p[4], s = height ? p[5] : p[6], pad = height ? p[7] : p[8], d = height ? p[9] : p[10];
        string t = b.R(), u = b.R();
        b.L($"shl.b32 {t}, {pad}, 1;");
        b.L($"add.u32 {t}, {t}, {size};");
        b.L($"sub.u32 {u}, {k}, 1;");
        b.L($"mul.lo.u32 {u}, {u}, {d};");
        b.L($"sub.u32 {t}, {t}, {u};");
        b.L($"sub.u32 {t}, {t}, 1;");
        b.L($"div.u32 {t}, {t}, {s};");
        b.L($"add.u32 {t}, {t}, 1;");
        return t;
    }

    /// <summary>The bits of count values of src (format f, from value 0) placed at value offset `offset` of dst, which starts
    /// zeroed over the region: a thread per value, OR-ed into the word (sub-word formats at offsets that do not end on a byte).</summary>
    public static string PlaceBits(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"place_bits_{f.Name}";
        var b = new PtxBuilder();
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

    // ---------------------------------------------------------------- scatter-add's ordering: counts, offsets, ordered sources

    /// <summary>counts[index[i]] += 1 over n sources (integer atomics: exact).</summary>
    public static string ScatterCount(Gpu gpu, out string entry)
    {
        entry = "scatter_count";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), i = b.R(), n = Param32(b, "p_n"), idx = Param64(b, "p_idx"), counts = Param64(b, "p_counts"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {i}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string a = b.RD(), target = b.R(), ca = b.RD(), old = b.R();
        b.L($"mul.wide.u32 {a}, {i}, 4;");
        b.L($"add.u64 {a}, {idx}, {a};");
        b.L($"ld.global.s32 {target}, [{a}];");
        b.L($"mul.wide.u32 {ca}, {target}, 4;");
        b.L($"add.u64 {ca}, {counts}, {ca};");
        b.L($"atom.global.add.u32 {old}, [{ca}], 1;");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_idx, .param .u64 p_counts, .param .u32 p_n");
    }

    /// <summary>offsets[0] = 0, offsets[r + 1] = offsets[r] + counts[r]: one block, a serial scan by its first thread
    /// (rows are thousands here; a parallel scan is the step for millions).</summary>
    public static string ScatterScan(Gpu gpu, out string entry)
    {
        entry = "scatter_scan";
        var b = new PtxBuilder();
        string tid = b.R(), rows = Param32(b, "p_rows"), counts = Param64(b, "p_counts"), offsets = Param64(b, "p_offsets"), p0 = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra DONE;");
        string r = b.R(), acc = b.R(), cnt = b.R(), ca = b.RD(), oa = b.RD(), pEnd = b.P();
        b.L($"mov.u32 {acc}, 0;");
        b.L($"st.global.u32 [{offsets}], {acc};");
        b.L($"mov.u32 {r}, 0;");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {r}, {rows};");
        b.L($"@{pEnd} bra DONE;");
        b.L($"mul.wide.u32 {ca}, {r}, 4;");
        b.L($"add.u64 {oa}, {offsets}, {ca};");
        b.L($"add.u64 {ca}, {counts}, {ca};");
        b.L($"ld.global.u32 {cnt}, [{ca}];");
        b.L($"add.u32 {acc}, {acc}, {cnt};");
        b.L($"st.global.u32 [{oa}+4], {acc};");
        b.L($"add.u32 {r}, {r}, 1;");
        b.L("bra LOOP;");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_counts, .param .u64 p_offsets, .param .u32 p_rows");
    }

    /// <summary>sources[offsets[index[i]] + rank] = i, rank = the number of earlier sources with the same index (a scan
    /// over the earlier indices: deterministic, quadratic in the source count — the radix step is for millions).</summary>
    public static string ScatterPlace(Gpu gpu, out string entry)
    {
        entry = "scatter_place";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), i = b.R(), n = Param32(b, "p_n"), idx = Param64(b, "p_idx"), offsets = Param64(b, "p_offsets"), sources = Param64(b, "p_sources"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {i}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string a = b.RD(), mine = b.R(), j = b.R(), rank = b.R(), ja = b.RD(), other = b.R(), pj = b.P(), peq = b.P(), oa = b.RD(), off = b.R(), sa = b.RD();
        b.L($"mul.wide.u32 {a}, {i}, 4;");
        b.L($"add.u64 {a}, {idx}, {a};");
        b.L($"ld.global.s32 {mine}, [{a}];");
        b.L($"mov.u32 {rank}, 0;");
        b.L($"mov.u32 {j}, 0;");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pj}, {j}, {i};");
        b.L($"@{pj} bra END;");
        b.L($"mul.wide.u32 {ja}, {j}, 4;");
        b.L($"add.u64 {ja}, {idx}, {ja};");
        b.L($"ld.global.s32 {other}, [{ja}];");
        b.L($"setp.eq.u32 {peq}, {other}, {mine};");
        b.L($"@{peq} add.u32 {rank}, {rank}, 1;");
        b.L($"add.u32 {j}, {j}, 1;");
        b.L("bra LOOP;");
        b.Label("END");
        b.L($"mul.wide.u32 {oa}, {mine}, 4;");
        b.L($"add.u64 {oa}, {offsets}, {oa};");
        b.L($"ld.global.u32 {off}, [{oa}];");
        b.L($"add.u32 {off}, {off}, {rank};");
        b.L($"mul.wide.u32 {sa}, {off}, 4;");
        b.L($"add.u64 {sa}, {sources}, {sa};");
        b.L($"st.global.u32 [{sa}], {i};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_idx, .param .u64 p_offsets, .param .u64 p_sources, .param .u32 p_n");
    }
}
