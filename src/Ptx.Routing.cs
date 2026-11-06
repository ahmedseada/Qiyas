// The device's routing and sampling kernels (Routing.cs): a block per row for top-k (k passes of the block's argmax
// over the values not yet taken) and for sampling (each thread's chunk summed, the chunks' prefix by the first thread,
// the first crossing found per chunk and the least taken); pick as a thread per row.
namespace Qiyas;

internal static partial class Ptx
{
    /// <summary>values[r][i], indices[r][i] for i &lt; k: the row's largest first, ties by the lower index.</summary>
    public static string TopKRows(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"topk_rows_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string x = Param64(b, "p_x"), values = Param64(b, "p_values"), indices = Param64(b, "p_indices"), cols = Param32(b, "p_cols"), k = Param32(b, "p_k");
        string start = b.R(), i = b.R(), pK = b.P(), smem = b.R();
        b.L($"mul.lo.u32 {start}, {row}, {cols};");
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mov.u32 {i}, 0;");
        b.Label("PASS");
        b.L($"setp.ge.u32 {pK}, {i}, {k};");
        b.L($"@{pK} bra PASS_END;");
        // This thread's best over its columns not already taken (the taken indices in shared memory past the reduction's slots).
        string c = b.R(), pEnd = b.P(), best = lane.NewReg(b), bestAt = b.R();
        b.L($"mov.{lane.Reg} {best}, {ReduceIdentity(lane, "max")};");
        b.L($"mov.u32 {bestAt}, 0xFFFFFFFF;");
        b.L($"mov.u32 {c}, {tid};");
        b.Label("COLS");
        b.L($"setp.ge.u32 {pEnd}, {c}, {cols};");
        b.L($"@{pEnd} bra COLS_END;");
        string t = b.R(), pTaken = b.P(), tAddr = b.R(), tv = b.R(), pEq = b.P();
        b.L($"mov.u32 {t}, 0;");
        b.L($"setp.ne.u32 {pTaken}, {t}, {t};");                                   // false
        b.Label("TAKEN");
        string pT = b.P();
        b.L($"setp.ge.u32 {pT}, {t}, {i};");
        b.L($"@{pT} bra TAKEN_END;");
        b.L($"mad.lo.u32 {tAddr}, {t}, 4, {smem};");
        b.L($"ld.shared.u32 {tv}, [{tAddr}+{ReduceBytes}];");
        b.L($"setp.eq.u32 {pEq}, {tv}, {c};");
        b.L($"or.pred {pTaken}, {pTaken}, {pEq};");
        b.L($"add.u32 {t}, {t}, 1;");
        b.L("bra TAKEN;");
        b.Label("TAKEN_END");
        b.L($"@{pTaken} bra COLS_NEXT;");
        string idx = b.R();
        b.L($"add.u32 {idx}, {start}, {c};");
        string v = Codec.UnpackAt(b, f, x, idx, lane);
        string pGt = b.P(), pFirst = b.P();
        b.L($"setp.gt.{lane.Op} {pGt}, {v}, {best};");
        b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
        b.L($"or.pred {pGt}, {pGt}, {pFirst};");
        b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {best}, {v}, {best}, {pGt};");
        b.L($"selp.u32 {bestAt}, {c}, {bestAt}, {pGt};");
        b.Label("COLS_NEXT");
        b.L($"add.u32 {c}, {c}, 256;");
        b.L("bra COLS;");
        b.Label("COLS_END");
        // The block's best (value, index): the largest value, the lowest index among equals, through shared memory pairs.
        var (bv, bi) = BlockArgmax(b, lane, tid, best, bestAt, smem);
        string p0 = b.P();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra RECORD_END;");
        {
            string tAddr2 = b.R(), oIdx = b.R(), vAddr = b.RD(), iAddr = b.RD();
            b.L($"mad.lo.u32 {tAddr2}, {i}, 4, {smem};");
            b.L($"st.shared.u32 [{tAddr2}+{ReduceBytes}], {bi};");
            b.L($"mad.lo.u32 {oIdx}, {row}, {k}, {i};");
            b.L($"mul.wide.u32 {iAddr}, {oIdx}, 4;");
            b.L($"add.u64 {iAddr}, {indices}, {iAddr};");
            b.L($"st.global.s32 [{iAddr}], {bi};");
            StoreAnyValue(b, f, lane, bv, values, oIdx);
            _ = vAddr;
        }

        b.Label("RECORD_END");
        b.L("bar.sync 0;");
        b.L($"add.u32 {i}, {i}, 1;");
        b.L("bra PASS;");
        b.Label("PASS_END");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_values, .param .u64 p_indices, .param .u32 p_cols, .param .u32 p_k", ReduceBytes + 256 * 4 + 256 * 16);
    }

    // A lane value stored at value index idx of an array of format f: whole words for 32/64-bit formats, 16- and 8-bit stores
    // as such, and sub-byte values OR-ed into their word (the output zeroed beforehand by the caller).
    private static void StoreAnyValue(PtxBuilder b, Fmt f, Lane lane, string v, string ptr, string idx)
    {
        var natural = f.Natural;
        string stored = lane == natural ? v : Codec.Convert(b, lane, v, natural, f);
        if (f.Bits >= 8)
        {
            string addr = b.RD();
            b.L($"mul.wide.u32 {addr}, {idx}, {f.Bits / 8};");
            b.L($"add.u64 {addr}, {ptr}, {addr};");
            if (f.Bits >= 32)
            {
                string code = f.WordBits == 64 ? b.RD() : b.R();
                if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, f, stored, natural, 0, code);
                b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
            }
            else
            {
                string code = b.R(), h = b.H();
                b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, f, stored, natural, 0, code);
                b.L($"cvt.u16.u32 {h}, {code};");
                b.L($"st.global.b{f.Bits} [{addr}], {h};");
            }

            return;
        }

        string code2 = b.R(), bits = b.R(), word = b.R(), shift = b.R(), shifted = b.R(), old = b.R(), wAddr = b.RD();
        b.L($"mov.b32 {code2}, 0;");
        Codec.Pack(b, f, stored, natural, 0, code2);
        b.L($"mul.lo.u32 {bits}, {idx}, {f.Bits};");
        b.L($"shr.u32 {word}, {bits}, 5;");
        b.L($"and.b32 {shift}, {bits}, 31;");
        b.L($"shl.b32 {shifted}, {code2}, {shift};");
        b.L($"mul.wide.u32 {wAddr}, {word}, 4;");
        b.L($"add.u64 {wAddr}, {ptr}, {wAddr};");
        b.L($"atom.global.or.b32 {old}, [{wAddr}], {shifted};");
    }

    // The block's (largest value, lowest index among equals) of each thread's pair: pairs in shared memory past the reduction's
    // and the taken slots, a tree by the first thread's warp... kept simple: a serial pass by thread 0 over 256 pairs, then broadcast.
    private static (string Value, string Index) BlockArgmax(PtxBuilder b, Lane lane, string tid, string v, string at, string smem)
    {
        int pairs = ReduceBytes + 256 * 4;
        string mine = b.R();
        b.L($"mad.lo.u32 {mine}, {tid}, 16, {smem};");
        b.L($"st.shared.{lane.Mem} [{mine}+{pairs}], {v};");
        b.L($"st.shared.u32 [{mine}+{pairs + 8}], {at};");
        b.L("bar.sync 0;");
        string p0 = b.P(), best = lane.NewReg(b), bestAt = b.R(), t = b.R(), pEnd = b.P(), addr = b.R(), ov = lane.NewReg(b), oa = b.R(), pGt = b.P(), pEq = b.P(), pLt = b.P(), pNone = b.P(), pTake = b.P();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra ARG_SKIP;");
        b.L($"mov.{lane.Reg} {best}, {ReduceIdentity(lane, "max")};");
        b.L($"mov.u32 {bestAt}, 0xFFFFFFFF;");
        b.L($"mov.u32 {t}, 0;");
        b.Label("ARG");
        b.L($"setp.ge.u32 {pEnd}, {t}, 256;");
        b.L($"@{pEnd} bra ARG_END;");
        b.L($"mad.lo.u32 {addr}, {t}, 16, {smem};");
        b.L($"ld.shared.{lane.Mem} {ov}, [{addr}+{pairs}];");
        b.L($"ld.shared.u32 {oa}, [{addr}+{pairs + 8}];");
        b.L($"setp.ne.u32 {pNone}, {oa}, 0xFFFFFFFF;");                          // a thread without a candidate
        b.L($"setp.gt.{lane.Op} {pGt}, {ov}, {best};");
        b.L($"setp.eq.{lane.Op} {pEq}, {ov}, {best};");
        b.L($"setp.lt.u32 {pLt}, {oa}, {bestAt};");
        b.L($"and.pred {pEq}, {pEq}, {pLt};");
        b.L($"or.pred {pTake}, {pGt}, {pEq};");
        string pFirst = b.P();
        b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
        b.L($"or.pred {pTake}, {pTake}, {pFirst};");
        b.L($"and.pred {pTake}, {pTake}, {pNone};");
        b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {best}, {ov}, {best}, {pTake};");
        b.L($"selp.u32 {bestAt}, {oa}, {bestAt}, {pTake};");
        b.L($"add.u32 {t}, {t}, 1;");
        b.L("bra ARG;");
        b.Label("ARG_END");
        b.L($"st.shared.{lane.Mem} [{smem}+{pairs}], {best};");
        b.L($"st.shared.u32 [{smem}+{pairs + 8}], {bestAt};");
        b.Label("ARG_SKIP");
        b.L("bar.sync 0;");
        string rv = lane.NewReg(b), ra = b.R();
        b.L($"ld.shared.{lane.Mem} {rv}, [{smem}+{pairs}];");
        b.L($"ld.shared.u32 {ra}, [{smem}+{pairs + 8}];");
        return (rv, ra);
    }

    /// <summary>out[r] = x[r][index[r]], a thread per row, stored as the type (value-wise).</summary>
    public static string PickRows(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"pick_rows_{f.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), r = b.R(), x = Param64(b, "p_x"), idx = Param64(b, "p_idx"), outp = Param64(b, "p_out"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {r}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {r}, {rows};");
        b.L($"@{pEnd} bra DONE;");
        string ia = b.RD(), col = b.R(), src = b.R();
        b.L($"mul.wide.u32 {ia}, {r}, 4;");
        b.L($"add.u64 {ia}, {idx}, {ia};");
        b.L($"ld.global.s32 {col}, [{ia}];");
        b.L($"mad.lo.u32 {src}, {r}, {cols}, {col};");
        string v = Codec.UnpackAt(b, f, x, src, lane);
        StoreAnyValue(b, f, lane, v, outp, r);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_idx, .param .u64 p_out, .param .u32 p_rows, .param .u32 p_cols");
    }

    /// <summary>picks[r] = the first column whose running sum reaches u[r] · Σ row (the last column when none): a block per row.</summary>
    public static string SampleRows(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"sample_rows_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string x = Param64(b, "p_x"), u = Param64(b, "p_u"), picks = Param64(b, "p_picks"), cols = Param32(b, "p_cols");
        string start = b.R(), smem = b.R(), chunk = b.R(), c0 = b.R(), c1 = b.R();
        b.L($"mul.lo.u32 {start}, {row}, {cols};");
        b.L($"mov.u32 {smem}, smem;");
        b.L($"add.u32 {chunk}, {cols}, 255;");
        b.L($"shr.u32 {chunk}, {chunk}, 8;");
        b.L($"mul.lo.u32 {c0}, {tid}, {chunk};");
        b.L($"add.u32 {c1}, {c0}, {chunk};");
        b.L($"min.u32 {c1}, {c1}, {cols};");
        // This thread's chunk sum.
        string sum = lane.NewReg(b), c = b.R(), pEnd = b.P(), idx = b.R();
        b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
        b.L($"mov.u32 {c}, {c0};");
        b.Label("SUM");
        b.L($"setp.ge.u32 {pEnd}, {c}, {c1};");
        b.L($"@{pEnd} bra SUM_END;");
        b.L($"add.u32 {idx}, {start}, {c};");
        string v = Codec.UnpackAt(b, f, x, idx, lane);
        b.L(lane.Add(sum, sum, v));
        b.L($"add.u32 {c}, {c}, 1;");
        b.L("bra SUM;");
        b.Label("SUM_END");
        string mine = b.R();
        b.L($"mad.lo.u32 {mine}, {tid}, {lane.Size}, {smem};");
        b.L($"st.shared.{lane.Mem} [{mine}+{ReduceBytes}], {sum};");
        b.L("bar.sync 0;");
        // The exclusive prefix of the chunk sums and the total, by the first thread.
        string p0 = b.P(), t = b.R(), acc = lane.NewReg(b), addr = b.R(), sv = lane.NewReg(b), pT = b.P();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra SCAN_SKIP;");
        b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
        b.L($"mov.u32 {t}, 0;");
        b.Label("SCAN");
        b.L($"setp.ge.u32 {pT}, {t}, 256;");
        b.L($"@{pT} bra SCAN_END;");
        b.L($"mad.lo.u32 {addr}, {t}, {lane.Size}, {smem};");
        b.L($"ld.shared.{lane.Mem} {sv}, [{addr}+{ReduceBytes}];");
        b.L($"st.shared.{lane.Mem} [{addr}+{ReduceBytes}], {acc};");
        b.L(lane.Add(acc, acc, sv));
        b.L($"add.u32 {t}, {t}, 1;");
        b.L("bra SCAN;");
        b.Label("SCAN_END");
        b.L($"st.shared.{lane.Mem} [{smem}+{ReduceBytes + 256 * 8}], {acc};");      // the total
        b.Label("SCAN_SKIP");
        b.L("bar.sync 0;");
        string prefix = lane.NewReg(b), total = lane.NewReg(b), uv = lane.NewReg(b), threshold = lane.NewReg(b);
        b.L($"ld.shared.{lane.Mem} {prefix}, [{mine}+{ReduceBytes}];");
        b.L($"ld.shared.{lane.Mem} {total}, [{smem}+{ReduceBytes + 256 * 8}];");
        string uval = Codec.UnpackAt(b, f, u, row, lane);
        b.L($"mov.{lane.Reg} {uv}, {uval};");
        b.L(lane.Mul(threshold, uv, total));
        // The first crossing in this thread's chunk (cols when none), then the least over the block.
        string run = lane.NewReg(b), found = b.R(), pGe = b.P(), pDone = b.P();
        b.L($"mov.{lane.Reg} {run}, {prefix};");
        b.L($"mov.u32 {found}, {cols};");
        b.L($"mov.u32 {c}, {c0};");
        b.Label("FIND");
        b.L($"setp.ge.u32 {pEnd}, {c}, {c1};");
        b.L($"@{pEnd} bra FIND_END;");
        b.L($"add.u32 {idx}, {start}, {c};");
        string v2 = Codec.UnpackAt(b, f, x, idx, lane);
        b.L(lane.Add(run, run, v2));
        b.L($"setp.ge.{lane.Op} {pGe}, {run}, {threshold};");
        b.L($"setp.ne.u32 {pDone}, {found}, {cols};");
        b.L($"not.pred {pDone}, {pDone};");
        b.L($"and.pred {pGe}, {pGe}, {pDone};");
        b.L($"selp.u32 {found}, {c}, {found}, {pGe};");
        b.L($"add.u32 {c}, {c}, 1;");
        b.L("bra FIND;");
        b.Label("FIND_END");
        string foundF = b.R();
        b.L($"mov.u32 {foundF}, {found};");
        string least = BlockReduce(b, Lane.U32, tid, foundF, "min", 256);
        string pFirst = b.P(), pa = b.RD(), lastCol = b.R(), pick = b.R(), pNone = b.P();
        b.L($"setp.ne.u32 {pFirst}, {tid}, 0;");
        b.L($"@{pFirst} bra DONE;");
        b.L($"sub.u32 {lastCol}, {cols}, 1;");
        b.L($"setp.ge.u32 {pNone}, {least}, {cols};");
        b.L($"selp.u32 {pick}, {lastCol}, {least}, {pNone};");
        b.L($"mul.wide.u32 {pa}, {row}, 4;");
        b.L($"add.u64 {pa}, {picks}, {pa};");
        b.L($"st.global.s32 [{pa}], {pick};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_u, .param .u64 p_picks, .param .u32 p_cols", ReduceBytes + 256 * 8 + 16);
    }
}
