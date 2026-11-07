// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's attention decode kernel (Attention.cs) and the lane-storage conversions.
//
// Decode: a block per (head, query); the query in shared memory; the keys in chunks of 256, a thread a key: its
// score (the dot over d, scaled), the chunk's max and sum by block reductions with the running softmax rescaled
// (online), then a thread per output dimension accumulating p_j · v_j over the chunk; K and V read once per query.
namespace Adad;

internal static partial class Ptx
{
    public const int DecodeThreads = 256;

    /// <summary>Shared bytes the decode kernel takes for a head dimension capacity: the reduction's slots, the query, the chunk's scores and weights.</summary>
    public static int DecodeShared(Lane lane, int dCap) => ReduceBytes + (dCap + 2 * DecodeThreads) * lane.Size;

    public static string AttentionDecode(Gpu gpu, Fmt f, Lane lane, int dCap, out string entry, Fmt? outFmt = null)
    {
        entry = $"attention_decode_{f.Name}_{lane.Name}_d{dCap}{(outFmt is null ? "" : "_o" + outFmt.Name)}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), o = Param64(b, "p_o"), lim = Param64(b, "p_lim"), starts = Param64(b, "p_start");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {scale}, [p_scale];");
        string h = b.R(), i = b.R(), t = b.R(), limit = b.R(), qBase = b.R(), kBase = b.R(), la = b.RD();
        b.L($"div.u32 {h}, {bid}, {lq};");
        b.L($"mul.lo.u32 {t}, {h}, {lq};");
        b.L($"sub.u32 {i}, {bid}, {t};");
        b.L($"mul.wide.u32 {la}, {i}, 4;");
        b.L($"add.u64 {la}, {lim}, {la};");
        b.L($"ld.global.s32 {limit}, [{la}];");
        b.L($"min.u32 {limit}, {limit}, {lk};");
        string start = b.R(), sa = b.RD();
        b.L($"mul.wide.u32 {sa}, {i}, 4;");
        b.L($"add.u64 {sa}, {starts}, {sa};");
        b.L($"ld.global.s32 {start}, [{sa}];");
        b.L($"min.u32 {start}, {start}, {limit};");
        b.L($"mad.lo.u32 {qBase}, {bid}, {d}, 0;");                       // (h·lq + i)·d = bid·d
        b.L($"mul.lo.u32 {kBase}, {h}, {lk};");
        b.L($"mul.lo.u32 {kBase}, {kBase}, {d};");
        // Shared: [smem] the reduction's slots; the query at ReduceBytes; the scores and weights after it.
        string smem = b.R(), sq = b.R(), ss = b.R(), sp = b.R();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"add.u32 {sq}, {smem}, {ReduceBytes};");
        b.L($"add.u32 {ss}, {sq}, {dCap * lane.Size};");
        b.L($"add.u32 {sp}, {ss}, {DecodeThreads * lane.Size};");
        // The query into shared memory.
        {
            string tt = b.R(), pEnd = b.P(), idx = b.R(), addr = b.R();
            b.L($"mov.u32 {tt}, {tid};");
            b.Label("QLOAD");
            b.L($"setp.ge.u32 {pEnd}, {tt}, {d};");
            b.L($"@{pEnd} bra QLOAD_END;");
            b.L($"add.u32 {idx}, {qBase}, {tt};");
            string qv = Codec.UnpackAt(b, f, q, idx, lane);
            b.L($"mad.lo.u32 {addr}, {tt}, {lane.Size}, {sq};");
            b.L($"st.shared.{lane.Mem} [{addr}], {qv};");
            b.L($"add.u32 {tt}, {tt}, {DecodeThreads};");
            b.L("bra QLOAD;");
            b.Label("QLOAD_END");
        }

        b.L("bar.sync 0;");
        const int PerThread = 4;                                         // dimensions a thread owns: tid, tid + 256, … (d ≤ 1024)
        string m = lane.NewReg(b), l = lane.NewReg(b), negInf = lane.Literal(double.NegativeInfinity);
        b.L($"mov.{lane.Reg} {m}, {negInf};");
        b.L($"mov.{lane.Reg} {l}, {lane.Zero};");
        var acc = new string[PerThread];
        for (int a = 0; a < PerThread; a++) { acc[a] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[a]}, {lane.Zero};"); }
        string chunk = b.R(), pDone = b.P();
        b.L($"mov.u32 {chunk}, {start};");
        b.Label("CHUNK");
        b.L($"setp.ge.u32 {pDone}, {chunk}, {limit};");
        b.L($"@{pDone} bra CHUNK_END;");
        // This thread's key: its score.
        string j = b.R(), pValid = b.P(), s = lane.NewReg(b);
        b.L($"add.u32 {j}, {chunk}, {tid};");
        b.L($"setp.lt.u32 {pValid}, {j}, {limit};");
        b.L($"mov.{lane.Reg} {s}, {lane.Zero};");
        {
            string tt = b.R(), pEnd = b.P(), idx = b.R(), addr = b.R(), qv = lane.NewReg(b), rowBase = b.R(), jc = b.R();
            b.L($"min.u32 {jc}, {j}, {limit} ;");
            b.L($"sub.u32 {jc}, {jc}, 0;");
            string lastKey = b.R();
            b.L($"sub.u32 {lastKey}, {limit}, 1;");
            b.L($"min.u32 {jc}, {j}, {lastKey};");                           // a thread past the limit reads a valid row (dropped)
            b.L($"mad.lo.u32 {rowBase}, {jc}, {d}, {kBase};");
            b.L($"mov.u32 {tt}, 0;");
            b.Label("DOT");
            b.L($"setp.ge.u32 {pEnd}, {tt}, {d};");
            b.L($"@{pEnd} bra DOT_END;");
            b.L($"add.u32 {idx}, {rowBase}, {tt};");
            string kv = Codec.UnpackAt(b, f, k, idx, lane);
            b.L($"mad.lo.u32 {addr}, {tt}, {lane.Size}, {sq};");
            b.L($"ld.shared.{lane.Mem} {qv}, [{addr}];");
            b.L(lane.Fma(s, qv, kv, s));
            b.L($"add.u32 {tt}, {tt}, 1;");
            b.L("bra DOT;");
            b.Label("DOT_END");
        }

        b.L(lane.Mul(s, s, scale));
        b.L($"selp.{lane.Op} {s}, {s}, {negInf}, {pValid};");
        string cmax = BlockReduce(b, lane, tid, s, "max", DecodeThreads);
        string mNew = lane.NewReg(b), alpha = lane.NewReg(b), dm = lane.NewReg(b), ds = lane.NewReg(b);
        b.L($"max.{lane.Op} {mNew}, {m}, {cmax};");
        b.L($"sub.rn.{lane.Op} {dm}, {m}, {mNew};");
        string alphaV = Codec.Exp(b, lane, dm);
        b.L($"mov.{lane.Reg} {alpha}, {alphaV};");
        b.L($"sub.rn.{lane.Op} {ds}, {s}, {mNew};");
        string pv = Codec.Exp(b, lane, ds);
        string p = lane.NewReg(b);
        b.L($"selp.{lane.Op} {p}, {pv}, {lane.Zero}, {pValid};");
        string pAddr = b.R();
        b.L($"mad.lo.u32 {pAddr}, {tid}, {lane.Size}, {sp};");
        b.L($"st.shared.{lane.Mem} [{pAddr}], {p};");
        string csum = BlockReduce(b, lane, tid, p, "add", DecodeThreads);
        b.L(lane.Fma(l, l, alpha, csum));
        b.L($"mov.{lane.Reg} {m}, {mNew};");
        b.L("bar.sync 0;");
        // The context: a thread per dimension over the chunk's keys.
        string chunkLen = b.R(), rest = b.R();
        b.L($"sub.u32 {rest}, {limit}, {chunk};");
        b.L($"min.u32 {chunkLen}, {rest}, {DecodeThreads};");
        for (int a = 0; a < PerThread; a++)
        {
            string dim = b.R(), pDim = b.P(), jj = b.R(), pEnd = b.P(), pw = lane.NewReg(b), idx = b.R(), wAddr = b.R(), vBase = b.R();
            b.L($"add.u32 {dim}, {tid}, {a * DecodeThreads};");
            b.L($"setp.ge.u32 {pDim}, {dim}, {d};");
            b.L($"@{pDim} bra CTX{a}_END;");
            b.L(lane.Mul(acc[a], acc[a], alpha));
            b.L($"mad.lo.u32 {vBase}, {chunk}, {d}, {kBase};");
            b.L($"add.u32 {vBase}, {vBase}, {dim};");
            b.L($"mov.u32 {jj}, 0;");
            b.Label($"CTX{a}");
            b.L($"setp.ge.u32 {pEnd}, {jj}, {chunkLen};");
            b.L($"@{pEnd} bra CTX{a}_END;");
            b.L($"mad.lo.u32 {wAddr}, {jj}, {lane.Size}, {sp};");
            b.L($"ld.shared.{lane.Mem} {pw}, [{wAddr}];");
            b.L($"mad.lo.u32 {idx}, {jj}, {d}, {vBase};");
            string vv = Codec.UnpackAt(b, f, v, idx, lane);
            b.L(lane.Fma(acc[a], pw, vv, acc[a]));
            b.L($"add.u32 {jj}, {jj}, 1;");
            b.L($"bra CTX{a};");
            b.Label($"CTX{a}_END");
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {chunk}, {chunk}, {DecodeThreads};");
        b.L("bra CHUNK;");
        b.Label("CHUNK_END");
        // o = acc / l, each dimension stored by its thread (a value-wise store: 16-bit and 8-bit formats as such).
        string inv = lane.NewReg(b), pAny = b.P();
        b.L($"setp.gt.{lane.Op} {pAny}, {l}, {lane.Zero};");
        b.L($"div.rn.{lane.Op} {inv}, {lane.Literal(1)}, {l};");
        b.L($"selp.{lane.Op} {inv}, {inv}, {lane.Zero}, {pAny};");
        var fo = outFmt ?? f;                                                   // O's format: the type, or lane storage for a block format
        var natural = fo.Natural;
        for (int a = 0; a < PerThread; a++)
        {
            string dim = b.R(), pDim = b.P(), val = lane.NewReg(b), idx = b.R(), addr = b.RD();
            b.L($"add.u32 {dim}, {tid}, {a * DecodeThreads};");
            b.L($"setp.ge.u32 {pDim}, {dim}, {d};");
            b.L($"@{pDim} bra OUT{a}_END;");
            b.L(lane.Mul(val, acc[a], inv));
            b.L($"add.u32 {idx}, {qBase}, {dim};");
            string stored = lane == natural ? val : Codec.Convert(b, lane, val, natural, fo);
            if (fo.Bits >= 32)
            {
                string code = fo.WordBits == 64 ? b.RD() : b.R();
                if (fo.WordBits == 32) b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, fo, stored, natural, 0, code);
                b.L($"mul.wide.u32 {addr}, {idx}, {fo.Bits / 8};");
                b.L($"add.u64 {addr}, {o}, {addr};");
                b.L($"st.global.{(fo.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
            }
            else
            {
                string code = b.R(), hh = b.H();
                b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, fo, stored, natural, 0, code);
                b.L($"mul.wide.u32 {addr}, {idx}, {fo.Bits / 8};");
                b.L($"add.u64 {addr}, {o}, {addr};");
                b.L($"cvt.u16.u32 {hh}, {code};");
                b.L($"st.global.b{fo.Bits} [{addr}], {hh};");
            }

            b.Label($"OUT{a}_END");
        }

        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return b.Finish(Header(gpu), entry, $".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_o, .param .u64 p_lim, .param .u64 p_start, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .{pa} p_scale", DecodeShared(lane, dCap));
    }

    // ---------------------------------------------------------------- lane storage: widen a format to its float lane's storage, and back

    /// <summary>out[i] (lane storage) = x[i] (format f): a thread per value.</summary>
    public static string Widen(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"widen_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), i = b.R(), n = Param32(b, "p_n"), x = Param64(b, "p_x"), outp = Param64(b, "p_out"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {i}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string v = Codec.UnpackAt(b, f, x, i, lane), addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {i}, {lane.Size};");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"st.global.{lane.Mem} [{addr}], {v};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }

    /// <summary>out (format f) = x (lane storage): a thread per output word.</summary>
    public static string Narrow(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"narrow_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), word = b.R(), n = Param32(b, "p_n"), x = Param64(b, "p_x"), outp = Param64(b, "p_out"), pEnd = b.P(), words = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {word}, {bid}, 256, {tid};");
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else { b.L($"add.u32 {words}, {n}, {f.Per - 1};"); b.L($"div.u32 {words}, {words}, {f.Per};"); }
        b.L($"setp.ge.u32 {pEnd}, {word}, {words};");
        b.L($"@{pEnd} bra DONE;");
        string code = f.WordBits == 64 ? b.RD() : b.R(), last = b.R();
        if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
        b.L($"sub.u32 {last}, {n}, 1;");
        var natural = f.Natural;
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R(), pv = b.P(), addr = b.RD(), val = lane.NewReg(b);
            b.L($"mad.lo.u32 {idx}, {word}, {f.Per}, {i};");
            b.L($"setp.lt.u32 {pv}, {idx}, {n};");
            b.L($"min.u32 {idx}, {idx}, {last};");
            b.L($"mul.wide.u32 {addr}, {idx}, {lane.Size};");
            b.L($"add.u64 {addr}, {x}, {addr};");
            b.L($"ld.global.{lane.Mem} {val}, [{addr}];");
            string stored = lane == natural ? val : Codec.Convert(b, lane, val, natural, f);
            string kept = natural.NewReg(b);
            b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {pv};");
            Codec.Pack(b, f, kept, natural, i, code);
        }

        string oAddr = b.RD();
        b.L($"mul.wide.u32 {oAddr}, {word}, {f.WordBits / 8};");
        b.L($"add.u64 {oAddr}, {outp}, {oAddr};");
        b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{oAddr}], {code};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }
}
