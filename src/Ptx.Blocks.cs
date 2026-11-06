// The device's block encoders (Blocks.cs): float32 lane storage into Q8_0 / Q4_0 blocks, a thread per block (the
// block's max, its scale as f16, the values rounded to the integers). Q4_K and Q6_K are decoded on the device (Codec) and
// encoded on the CPU; their device encoders are the step past this.
namespace Qiyas;

internal static partial class Ptx
{
    public static string LanesToBlock(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"lanes_to_{f.Name}_{lane.Name}";
        if (f.Name is "q4_k" or "q6_k") return LanesToKBlock(gpu, f, lane, entry);
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), blk = b.R(), x = Param64(b, "p_x"), outp = Param64(b, "p_out"), n = Param32(b, "p_n"), blocks = b.R(), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {blk}, {bid}, 256, {tid};");
        b.L($"add.u32 {blocks}, {n}, {f.BlockValues - 1};");
        b.L($"div.u32 {blocks}, {blocks}, {f.BlockValues};");
        b.L($"setp.ge.u32 {pEnd}, {blk}, {blocks};");
        b.L($"@{pEnd} bra DONE;");
        string first = b.R(), last = b.R(), xAddr = b.RD(), bAddr = b.RD();
        b.L($"mul.lo.u32 {first}, {blk}, {f.BlockValues};");
        b.L($"sub.u32 {last}, {n}, 1;");
        b.L($"mul.wide.u32 {xAddr}, {first}, {lane.Size};");
        b.L($"add.u64 {xAddr}, {x}, {xAddr};");
        b.L($"mul.wide.u32 {bAddr}, {blk}, {f.BlockBytes};");
        b.L($"add.u64 {bAddr}, {outp}, {bAddr};");
        // The 32 values (0 past the count) and the block's max |x| (Q4_0: the signed max).
        var vals = new string[32];
        string amax = b.F(), maxv = b.F(), pIn = b.P(), idx = b.R();
        b.L($"mov.f32 {amax}, 0f00000000;");
        b.L($"mov.f32 {maxv}, 0f00000000;");
        for (int i = 0; i < 32; i++)
        {
            vals[i] = b.F();
            string raw = lane.Wide ? b.FD() : b.F();
            b.L($"add.u32 {idx}, {first}, {i};");
            b.L($"setp.le.u32 {pIn}, {idx}, {last};");
            b.L(lane.Wide ? $"mov.f64 {raw}, 0d0000000000000000;" : $"mov.f32 {raw}, 0f00000000;");
            b.L($"@{pIn} ld.global.{lane.Mem} {raw}, [{xAddr}+{i * lane.Size}];");
            if (lane.Wide) b.L($"cvt.rn.f32.f64 {vals[i]}, {raw};"); else b.L($"mov.f32 {vals[i]}, {raw};");
            string a = b.F(), pGt = b.P();
            b.L($"abs.f32 {a}, {vals[i]};");
            b.L($"setp.gt.f32 {pGt}, {a}, {amax};");
            b.L($"selp.f32 {maxv}, {vals[i]}, {maxv}, {pGt};");
            b.L($"max.f32 {amax}, {amax}, {a};");
        }

        // The scale as stored (f16) and read back, its inverse.
        string d = b.F(), h = b.H(), dBack = b.F(), id = b.F(), pZero = b.P();
        if (f.Is("q8_0")) b.L($"div.rn.f32 {d}, {amax}, 0f42FE0000;");               // / 127
        else b.L($"div.rn.f32 {d}, {maxv}, 0fC1000000;");                             // / −8
        b.L($"cvt.rn.f16.f32 {h}, {d};");
        b.L($"cvt.f32.f16 {dBack}, {h};");
        b.L($"setp.eq.f32 {pZero}, {dBack}, 0f00000000;");
        b.L($"div.rn.f32 {id}, 0f3F800000, {dBack};");
        b.L($"selp.f32 {id}, 0f00000000, {id}, {pZero};");
        b.L($"st.global.b16 [{bAddr}], {h};");
        if (f.Is("q8_0"))
        {
            for (int i = 0; i < 32; i++)
            {
                string t = b.F(), q = b.R(), qb = b.H();
                b.L($"mul.rn.f32 {t}, {vals[i]}, {id};");
                b.L($"cvt.rni.s32.f32 {q}, {t};");
                b.L($"max.s32 {q}, {q}, -127;");
                b.L($"min.s32 {q}, {q}, 127;");
                b.L($"cvt.u16.u32 {qb}, {q};");
                b.L($"st.global.b8 [{bAddr}+{2 + i}], {qb};");
            }
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                string tl = b.F(), th = b.F(), ql = b.R(), qh = b.R(), code = b.R(), cb = b.H();
                b.L($"fma.rn.f32 {tl}, {vals[i]}, {id}, 0f41080000;");               // + 8.5
                b.L($"fma.rn.f32 {th}, {vals[i + 16]}, {id}, 0f41080000;");
                b.L($"cvt.rmi.s32.f32 {ql}, {tl};");                                   // floor
                b.L($"cvt.rmi.s32.f32 {qh}, {th};");
                b.L($"max.s32 {ql}, {ql}, 0;"); b.L($"min.s32 {ql}, {ql}, 15;");
                b.L($"max.s32 {qh}, {qh}, 0;"); b.L($"min.s32 {qh}, {qh}, 15;");
                b.L($"shl.b32 {code}, {qh}, 4;");
                b.L($"or.b32 {code}, {code}, {ql};");
                b.L($"cvt.u16.u32 {cb}, {code};");
                b.L($"st.global.b8 [{bAddr}+{2 + i}], {cb};");
            }
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }

    // The K blocks (256 values): the values read from lane storage as f32 (0 past the count), the sub-block statistics,
    // the scales as the formats pack them, then every value quantized and packed. One thread a block.
    private static string LanesToKBlock(Gpu gpu, Fmt f, Lane lane, string entry)
    {
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), blk = b.R(), x = Param64(b, "p_x"), outp = Param64(b, "p_out"), n = Param32(b, "p_n"), blocks = b.R(), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {blk}, {bid}, 256, {tid};");
        b.L($"add.u32 {blocks}, {n}, 255;");
        b.L($"shr.u32 {blocks}, {blocks}, 8;");
        b.L($"setp.ge.u32 {pEnd}, {blk}, {blocks};");
        b.L($"@{pEnd} bra DONE;");
        string first = b.R(), xAddr = b.RD(), bAddr = b.RD();
        b.L($"shl.b32 {first}, {blk}, 8;");
        b.L($"mul.wide.u32 {xAddr}, {first}, {lane.Size};");
        b.L($"add.u64 {xAddr}, {x}, {xAddr};");
        b.L($"mul.wide.u32 {bAddr}, {blk}, {f.BlockBytes};");
        b.L($"add.u64 {bAddr}, {outp}, {bAddr};");
        // value i of the block as f32 (0 past the count), by a loop body: V(i register) -> f32 register
        string V(string i)
        {
            string idx = b.R(), pIn = b.P(), a = b.RD(), raw = lane.Wide ? b.FD() : b.F(), v = b.F();
            b.L($"add.u32 {idx}, {first}, {i};");
            b.L($"setp.lt.u32 {pIn}, {idx}, {n};");
            b.L($"mul.wide.u32 {a}, {i}, {lane.Size};");
            b.L($"add.u64 {a}, {xAddr}, {a};");
            b.L(lane.Wide ? $"mov.f64 {raw}, 0d0000000000000000;" : $"mov.f32 {raw}, 0f00000000;");
            b.L($"@{pIn} ld.global.{lane.Mem} {raw}, [{a}];");
            if (lane.Wide) b.L($"cvt.rn.f32.f64 {v}, {raw};"); else b.L($"mov.f32 {v}, {raw};");
            return v;
        }

        string Rni(string fv) { string r = b.R(); b.L($"cvt.rni.s32.f32 {r}, {fv};"); return r; }
        string ClampI(string r, int lo, int hi) { b.L($"max.s32 {r}, {r}, {lo};"); b.L($"min.s32 {r}, {r}, {hi};"); return r; }
        string I2F(string r) { string fv = b.F(); b.L($"cvt.rn.f32.s32 {fv}, {r};"); return fv; }
        string HalfRound(string fv) { string h = b.H(), back = b.F(); b.L($"cvt.rn.f16.f32 {h}, {fv};"); b.L($"cvt.f32.f16 {back}, {h};"); _ = h; return back; }
        void StoreHalf(string fv, int at) { string h = b.H(); b.L($"cvt.rn.f16.f32 {h}, {fv};"); b.L($"st.global.b16 [{bAddr}+{at}], {h};"); }
        void StoreByte(string r, string offset, int at) { string a = b.RD(), hh = b.H(); b.L($"cvt.u64.u32 {a}, {offset};"); b.L($"add.u64 {a}, {bAddr}, {a};"); b.L($"cvt.u16.u32 {hh}, {r};"); b.L($"st.global.b8 [{a}+{at}], {hh};"); }
        string LoadByte(string offset, int at) { string a = b.RD(), r = b.R(); b.L($"cvt.u64.u32 {a}, {offset};"); b.L($"add.u64 {a}, {bAddr}, {a};"); b.L($"ld.global.u8 {r}, [{a}+{at}];"); return r; }
        if (f.Is("q4_k"))
        {
            // Sub-block statistics: lo (≤ 0), hi → scale_j = (hi − lo)/15, min_j = −lo; the block's largest scale and min.
            var scales = new string[8]; var mins = new string[8];
            string maxScale = b.F(), maxMin = b.F();
            b.L($"mov.f32 {maxScale}, 0f00000000;"); b.L($"mov.f32 {maxMin}, 0f00000000;");
            for (int j = 0; j < 8; j++)
            {
                string lo = b.F(), hi = b.F(), i = b.R(), pI = b.P();
                b.L($"mov.f32 {lo}, 0f7F7FFFFF;"); b.L($"mov.f32 {hi}, 0fFF7FFFFF;");
                b.L($"mov.u32 {i}, {j * 32};");
                b.Label($"ST{j}");
                b.L($"setp.ge.u32 {pI}, {i}, {j * 32 + 32};");
                b.L($"@{pI} bra ST{j}_END;");
                string v = V(i);
                b.L($"min.f32 {lo}, {lo}, {v};"); b.L($"max.f32 {hi}, {hi}, {v};");
                b.L($"add.u32 {i}, {i}, 1;");
                b.L($"bra ST{j};");
                b.Label($"ST{j}_END");
                b.L($"min.f32 {lo}, {lo}, 0f00000000;");
                scales[j] = b.F(); mins[j] = b.F();
                b.L($"sub.rn.f32 {scales[j]}, {hi}, {lo};");
                b.L($"div.rn.f32 {scales[j]}, {scales[j]}, 0f41700000;");          // / 15
                b.L($"neg.f32 {mins[j]}, {lo};");
                b.L($"max.f32 {maxScale}, {maxScale}, {scales[j]};");
                b.L($"max.f32 {maxMin}, {maxMin}, {mins[j]};");
            }

            string d = b.F(), dmin = b.F();
            b.L($"div.rn.f32 {d}, {maxScale}, 0f427C0000;");                        // / 63
            b.L($"div.rn.f32 {dmin}, {maxMin}, 0f427C0000;");
            StoreHalf(d, 0); StoreHalf(dmin, 2);
            d = HalfRound(d); dmin = HalfRound(dmin);
            string pD0 = b.P(), pM0 = b.P();
            b.L($"setp.eq.f32 {pD0}, {d}, 0f00000000;");
            b.L($"setp.eq.f32 {pM0}, {dmin}, 0f00000000;");
            var ls = new string[8]; var lm = new string[8];
            for (int j = 0; j < 8; j++)
            {
                string qs = b.F(), qm = b.F();
                b.L($"div.rn.f32 {qs}, {scales[j]}, {d};");
                b.L($"div.rn.f32 {qm}, {mins[j]}, {dmin};");
                ls[j] = ClampI(Rni(qs), 0, 63); lm[j] = ClampI(Rni(qm), 0, 63);
                b.L($"selp.b32 {ls[j]}, 0, {ls[j]}, {pD0};");
                b.L($"selp.b32 {lm[j]}, 0, {lm[j]}, {pM0};");
            }

            // The 12 scale bytes: j < 4: sc[j] = ls, sc[j+4] = lm; else sc[j+4] = (ls & 15) | (lm & 15) << 4, sc[j−4] |= (ls >> 4) << 6, sc[j] |= (lm >> 4) << 6.
            var sc = new string[12];
            for (int k = 0; k < 12; k++) { sc[k] = b.R(); b.L($"mov.u32 {sc[k]}, 0;"); }
            for (int j = 0; j < 4; j++) { b.L($"mov.u32 {sc[j]}, {ls[j]};"); b.L($"mov.u32 {sc[j + 4]}, {lm[j]};"); }
            for (int j = 4; j < 8; j++)
            {
                string t = b.R(), u = b.R();
                b.L($"and.b32 {t}, {ls[j]}, 15;"); b.L($"and.b32 {u}, {lm[j]}, 15;"); b.L($"shl.b32 {u}, {u}, 4;"); b.L($"or.b32 {sc[j + 4]}, {t}, {u};");
                b.L($"shr.u32 {t}, {ls[j]}, 4;"); b.L($"shl.b32 {t}, {t}, 6;"); b.L($"or.b32 {sc[j - 4]}, {sc[j - 4]}, {t};");
                b.L($"shr.u32 {t}, {lm[j]}, 4;"); b.L($"shl.b32 {t}, {t}, 6;"); b.L($"or.b32 {sc[j]}, {sc[j]}, {t};");
            }

            string zero = b.R();
            b.L($"mov.u32 {zero}, 0;");
            for (int k = 0; k < 12; k++) StoreByte(sc[k], zero, 4 + k);
            // The nibbles: pairs of sub-blocks share 32 bytes; q = clamp(rni((v + m_j)/d_j), 0, 15), d_j = d·ls_j, m_j = dmin·lm_j.
            for (int pair = 0; pair < 4; pair++)
            {
                int j0 = 2 * pair, j1 = 2 * pair + 1;
                string dj0 = b.F(), dj1 = b.F(), mj0 = b.F(), mj1 = b.F(), p0 = b.P(), p1 = b.P();
                b.L($"mul.rn.f32 {dj0}, {d}, {I2F(ls[j0])};"); b.L($"mul.rn.f32 {dj1}, {d}, {I2F(ls[j1])};");
                b.L($"mul.rn.f32 {mj0}, {dmin}, {I2F(lm[j0])};"); b.L($"mul.rn.f32 {mj1}, {dmin}, {I2F(lm[j1])};");
                b.L($"setp.eq.f32 {p0}, {dj0}, 0f00000000;"); b.L($"setp.eq.f32 {p1}, {dj1}, 0f00000000;");
                string i = b.R(), pI = b.P();
                b.L($"mov.u32 {i}, 0;");
                b.Label($"NB{pair}");
                b.L($"setp.ge.u32 {pI}, {i}, 32;");
                b.L($"@{pI} bra NB{pair}_END;");
                string i0 = b.R(), i1 = b.R();
                b.L($"add.u32 {i0}, {i}, {j0 * 32};"); b.L($"add.u32 {i1}, {i}, {j1 * 32};");
                string v0 = V(i0), v1 = V(i1), t0 = b.F(), t1 = b.F();
                b.L($"add.rn.f32 {t0}, {v0}, {mj0};"); b.L($"div.rn.f32 {t0}, {t0}, {dj0};");
                b.L($"add.rn.f32 {t1}, {v1}, {mj1};"); b.L($"div.rn.f32 {t1}, {t1}, {dj1};");
                string q0 = ClampI(Rni(t0), 0, 15), q1 = ClampI(Rni(t1), 0, 15), code = b.R(), off = b.R();
                b.L($"selp.b32 {q0}, 0, {q0}, {p0};"); b.L($"selp.b32 {q1}, 0, {q1}, {p1};");
                b.L($"shl.b32 {code}, {q1}, 4;"); b.L($"or.b32 {code}, {code}, {q0};");
                b.L($"add.u32 {off}, {i}, {pair * 32};");
                StoreByte(code, off, 16);
                b.L($"add.u32 {i}, {i}, 1;");
                b.L($"bra NB{pair};");
                b.Label($"NB{pair}_END");
            }
        }
        else
        {
            // Q6_K: 16 sub-blocks of 16: scale_j = max|v|/31; d = max scale / 127 (f16); ls_j = clamp(rni(scale_j/d), −127, 127);
            // q = clamp(rni(v/(d·ls_j)), −32, 31) + 32 into ql (low nibbles) and qh (2 bits) by the layout.
            var scales = new string[16];
            string maxScale = b.F();
            b.L($"mov.f32 {maxScale}, 0f00000000;");
            for (int j = 0; j < 16; j++)
            {
                string amax = b.F(), i = b.R(), pI = b.P();
                b.L($"mov.f32 {amax}, 0f00000000;");
                b.L($"mov.u32 {i}, {j * 16};");
                b.Label($"SA{j}");
                b.L($"setp.ge.u32 {pI}, {i}, {j * 16 + 16};");
                b.L($"@{pI} bra SA{j}_END;");
                string v = V(i), a = b.F();
                b.L($"abs.f32 {a}, {v};"); b.L($"max.f32 {amax}, {amax}, {a};");
                b.L($"add.u32 {i}, {i}, 1;");
                b.L($"bra SA{j};");
                b.Label($"SA{j}_END");
                scales[j] = b.F();
                b.L($"div.rn.f32 {scales[j]}, {amax}, 0f41F80000;");                   // / 31
                b.L($"max.f32 {maxScale}, {maxScale}, {scales[j]};");
            }

            string d = b.F();
            b.L($"div.rn.f32 {d}, {maxScale}, 0f42FE0000;");                             // / 127
            StoreHalf(d, 208);
            d = HalfRound(d);
            string pD0 = b.P(), zero = b.R();
            b.L($"setp.eq.f32 {pD0}, {d}, 0f00000000;");
            b.L($"mov.u32 {zero}, 0;");
            var ls = new string[16]; var dj = new string[16]; var pZ = new string[16];
            for (int j = 0; j < 16; j++)
            {
                string qs = b.F();
                b.L($"div.rn.f32 {qs}, {scales[j]}, {d};");
                ls[j] = ClampI(Rni(qs), -127, 127);
                b.L($"selp.b32 {ls[j]}, 0, {ls[j]}, {pD0};");
                string sb = b.R();
                b.L($"and.b32 {sb}, {ls[j]}, 255;");
                StoreByte(sb, zero, 192 + j);
                dj[j] = b.F(); pZ[j] = b.P();
                b.L($"mul.rn.f32 {dj[j]}, {d}, {I2F(ls[j])};");
                b.L($"setp.eq.f32 {pZ[j]}, {dj[j]}, 0f00000000;");
            }

            // The quantized values, per half (128 values): for l in 0..31: q1 = v[l], q2 = v[l+32], q3 = v[l+64], q4 = v[l+96] (sub-block j = index / 16).
            for (int half = 0; half < 2; half++)
            {
                string l = b.R(), pL = b.P();
                b.L($"mov.u32 {l}, 0;");
                b.Label($"QH{half}");
                b.L($"setp.ge.u32 {pL}, {l}, 32;");
                b.L($"@{pL} bra QH{half}_END;");
                var q = new string[4];
                for (int part = 0; part < 4; part++)
                {
                    string idx = b.R();
                    b.L($"add.u32 {idx}, {l}, {half * 128 + part * 32};");
                    string v = V(idx), t = b.F(), sub = b.R(), djSel = b.F(), pZero = b.P();
                    // the sub-block of this value: idx / 16 (a runtime select over the 16 scales)
                    b.L($"shr.u32 {sub}, {idx}, 4;");
                    b.L($"mov.f32 {djSel}, 0f00000000;");
                    b.L($"setp.ne.u32 {pZero}, {sub}, {sub};");                       // false
                    for (int j = 0; j < 16; j++)
                    {
                        string pj = b.P();
                        b.L($"setp.eq.u32 {pj}, {sub}, {j};");
                        b.L($"selp.f32 {djSel}, {dj[j]}, {djSel}, {pj};");
                        string pzj = b.P();
                        b.L($"and.pred {pzj}, {pj}, {pZ[j]};");
                        b.L($"or.pred {pZero}, {pZero}, {pzj};");
                    }

                    b.L($"div.rn.f32 {t}, {v}, {djSel};");
                    q[part] = ClampI(Rni(t), -32, 31);
                    b.L($"selp.b32 {q[part]}, 0, {q[part]}, {pZero};");
                    b.L($"add.s32 {q[part]}, {q[part]}, 32;");
                }

                string lo1 = b.R(), lo2 = b.R(), hi = b.R(), t1 = b.R(), t2 = b.R(), off = b.R();
                b.L($"and.b32 {lo1}, {q[0]}, 15;"); b.L($"and.b32 {t1}, {q[2]}, 15;"); b.L($"shl.b32 {t1}, {t1}, 4;"); b.L($"or.b32 {lo1}, {lo1}, {t1};");
                b.L($"and.b32 {lo2}, {q[1]}, 15;"); b.L($"and.b32 {t2}, {q[3]}, 15;"); b.L($"shl.b32 {t2}, {t2}, 4;"); b.L($"or.b32 {lo2}, {lo2}, {t2};");
                b.L($"shr.u32 {hi}, {q[0]}, 4;"); b.L($"and.b32 {hi}, {hi}, 3;");
                b.L($"shr.u32 {t1}, {q[1]}, 4;"); b.L($"and.b32 {t1}, {t1}, 3;"); b.L($"shl.b32 {t1}, {t1}, 2;"); b.L($"or.b32 {hi}, {hi}, {t1};");
                b.L($"shr.u32 {t1}, {q[2]}, 4;"); b.L($"and.b32 {t1}, {t1}, 3;"); b.L($"shl.b32 {t1}, {t1}, 4;"); b.L($"or.b32 {hi}, {hi}, {t1};");
                b.L($"shr.u32 {t1}, {q[3]}, 4;"); b.L($"and.b32 {t1}, {t1}, 3;"); b.L($"shl.b32 {t1}, {t1}, 6;"); b.L($"or.b32 {hi}, {hi}, {t1};");
                b.L($"add.u32 {off}, {l}, {half * 64};");
                StoreByte(lo1, off, 0);
                b.L($"add.u32 {off}, {l}, {half * 64 + 32};");
                StoreByte(lo2, off, 0);
                b.L($"add.u32 {off}, {l}, {half * 32};");
                StoreByte(hi, off, 128);
                b.L($"add.u32 {l}, {l}, 1;");
                b.L($"bra QH{half};");
                b.Label($"QH{half}_END");
            }
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }
}
