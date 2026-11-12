// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's attention decode kernel (Attention.cs) and the lane-storage conversions.
//
// Decode: a block per (head, query); the query in shared memory; the keys in chunks of 256, a thread a key: its
// score (the dot over d, scaled), the chunk's max and sum by block reductions with the running softmax rescaled
// (online), then a thread per output dimension accumulating p_j · v_j over the chunk; K and V read once per query.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    public const int DecodeThreads = 256;

    /// <summary>Shared bytes the decode kernel takes for a head dimension capacity: the reduction's slots, the query, the chunk's weights.</summary>
    public static int DecodeShared(Lane lane, int dCap) => ReduceBytes + (dCap + DecodeThreads) * lane.Size;

    /// <summary>The decode kernel over caches at a position the device holds (ICache.AttentionCached): K and V stored as Fc,
    /// the scores capped.</summary>
    internal sealed record DecodeCache(Fmt Fc, bool Capped)
    {
        public string Key => $"{Fc.Name}{(Capped ? "_cap" : "")}";
    }

    /// <summary>cache null: a block a (head, query), each query's limit and start read from p_lim and p_start. Over caches:
    /// lq is the rows a head of K and V serves (a group's query heads × p_lq0 queries), p_lim the position (an s32 at the
    /// device): row r's query i = r mod p_lq0 sees keys [max(0, limit − p_win) when a window, limit), limit = p + i + 1 held
    /// to [0, lk]; p_cap the cap.</summary>
    /// runs (K and V a block format, d a multiple of 8): a key's row read in runs of 8 values (Codec.UnpackRun: the block's
    /// and the sub-block's scales read once a run), each value's arithmetic and the sum's order as without.
    public static string AttentionDecode(Gpu gpu, Fmt f, Lane lane, int dCap, out string entry, Fmt? outFmt = null, DecodeCache? cache = null, bool runs = false)
    {
        entry = $"attention_decode_{f.Name}_{lane.Name}_d{dCap}{(outFmt is null ? "" : "_o" + outFmt.Name)}{(cache is null ? "" : "_at_" + cache.Key)}{(runs ? "_runs" : "")}";
        var fk = cache?.Fc ?? f;                                                 // K and V's stored format
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), o = Param64(b, "p_o"), lim = Param64(b, "p_lim"), starts = Param64(b, "p_start");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {scale}, [p_scale];");
        string cap = "";
        string h = b.R(), i = b.R(), t = b.R(), limit = b.R(), qBase = b.R(), kBase = b.R(), hk = b.R(), la = b.RD();
        b.L($"div.u32 {h}, {bid}, {lq};");
        b.L($"mul.lo.u32 {t}, {h}, {lq};");
        b.L($"sub.u32 {i}, {bid}, {t};");
        string start = b.R(), sa = b.RD();
        if (cache is null)
        {
            b.L($"mul.wide.u32 {la}, {i}, 4;");
            b.L($"add.u64 {la}, {lim}, {la};");
            b.L($"ld.global.s32 {limit}, [{la}];");
            b.L($"min.u32 {limit}, {limit}, {lk};");
            b.L($"mul.wide.u32 {sa}, {i}, 4;");
            b.L($"add.u64 {sa}, {starts}, {sa};");
            b.L($"ld.global.s32 {start}, [{sa}];");
        }
        else
        {
            string lq0 = Param32(b, "p_lq0"), win = Param32(b, "p_win"), pos = b.R(), i0 = b.R(), noWin = b.P();
            cap = lane.NewReg(b);
            b.L($"ld.param.{lane.Op} {cap}, [p_cap];");
            b.L($"ld.global.s32 {pos}, [{lim}];");
            b.L($"rem.u32 {i0}, {i}, {lq0};");
            b.L($"add.s32 {limit}, {pos}, {i0};");
            b.L($"add.s32 {limit}, {limit}, 1;");
            b.L($"max.s32 {limit}, {limit}, 0;");
            b.L($"min.s32 {limit}, {limit}, {lk};");
            b.L($"sub.s32 {start}, {limit}, {win};");
            b.L($"max.s32 {start}, {start}, 0;");
            b.L($"setp.eq.u32 {noWin}, {win}, 0;");
            b.L($"selp.u32 {start}, 0, {start}, {noWin};");
        }

        b.L($"min.u32 {start}, {start}, {limit};");
        b.L($"mad.lo.u32 {qBase}, {bid}, {d}, 0;");                       // (h·lq + i)·d = bid·d
        b.L($"mul.lo.u32 {hk}, {h}, {lk};");
        b.L($"mul.lo.u32 {kBase}, {hk}, {d};");
        // Shared: [smem] the reduction's slots; the query at ReduceBytes; the chunk's weights after it.
        string smem = b.R(), sq = b.R(), sp = b.R();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"add.u32 {sq}, {smem}, {ReduceBytes};");
        b.L($"add.u32 {sp}, {sq}, {dCap * lane.Size};");
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
        string j = b.R(), pValid = b.P(), s = lane.NewReg(b), jc = b.R();
        b.L($"add.u32 {j}, {chunk}, {tid};");
        b.L($"setp.lt.u32 {pValid}, {j}, {limit};");
        b.L($"mov.{lane.Reg} {s}, {lane.Zero};");
        {
            string tt = b.R(), pEnd = b.P(), idx = b.R(), addr = b.R(), qv = lane.NewReg(b), rowBase = b.R();
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
            int run = runs ? 8 : 1;
            string[] keys = runs ? Codec.UnpackRun(b, fk, k, idx, run, lane) : [Codec.UnpackAt(b, fk, k, idx, lane)];
            b.L($"mad.lo.u32 {addr}, {tt}, {lane.Size}, {sq};");
            for (int r = 0; r < run; r++)
            {
                string kv = keys[r];
                b.L($"ld.shared.{lane.Mem} {qv}, [{addr}+{r * lane.Size}];");
                b.L(lane.Fma(s, qv, kv, s));
            }

            b.L($"add.u32 {tt}, {tt}, {run};");
            b.L("bra DOT;");
            b.Label("DOT_END");
        }

        if (cache is { Capped: true })
        {
            // The map's softcap: cap·tanh(scale·s / cap).
            string capped = Fn.Tanh(b, lane, LaneOp(b, lane, "div", LaneOp(b, lane, "mul", scale, s), cap));
            b.L(lane.Mul(s, cap, capped));
        }
        else b.L(lane.Mul(s, s, scale));

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
            string vv = Codec.UnpackAt(b, fk, v, idx, lane);
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
        string over = cache is null ? "" : $", .param .u32 p_lq0, .param .u32 p_win, .param .{pa} p_cap";
        return b.Finish(Header(gpu), entry, $".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_o, .param .u64 p_lim, .param .u64 p_start, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .{pa} p_scale{over}", DecodeShared(lane, dCap));
    }

    /// <summary>The partials each warp of the split decode kernel writes: m, l and the context's d values (lane values).</summary>
    public static int DecodePartialLanes(int dCap) => 2 + dCap;

    /// <summary>The decode kernel in splits of the keys (flash decoding): a grid of (head, query) × p_splits blocks, each
    /// warp of a block its own part of the query's keys [start, limit) (p_splits · 8 contiguous parts), its lanes the
    /// head's dimensions (lane, lane + 32, …: each key's row read by the warp's lanes together), four keys at a time: their
    /// scores by the lanes' products joined by shuffles, the running softmax rescaled (online), the context p · v. Each warp
    /// writes its part's m, l and context (unnormalized) to p_ws; DecodeCombine joins the parts in order. The device's
    /// warps all at work where a block a query left most of them idle (32 heads, one query: 32 blocks on 70 multiprocessors)
    /// and K read along its rows (a thread a key read across them).
    /// runs (K and V a block format, d a multiple of 8, dCap at most 256): each lane takes 8 consecutive dimensions
    /// (8·lane …), a key's run of them decoded at once (Codec.UnpackRun: the block's and the sub-block's scales read once a
    /// run) where a value at a time decoded them per value.</summary>
    public static string AttentionDecodeSplit(Gpu gpu, Fmt f, Lane lane, int dCap, out string entry, DecodeCache? cache = null, bool runs = false)
    {
        entry = $"attention_decode_split_{f.Name}_{lane.Name}_d{dCap}{(cache is null ? "" : "_at_" + cache.Key)}{(runs ? "_runs" : "")}";
        var fk = cache?.Fc ?? f;
        if (fk.IsBlock && !runs) throw new NotSupportedException("the split decode reads a block format's K and V in runs");
        if (runs && dCap > 256) throw new NotSupportedException("the split decode's runs: 8 dimensions a lane, dCap at most 256");
        const int Run = 8;
        int perLane = runs ? Run : dCap / 32;
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R(), split = b.R(), warp = b.R(), wl = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mov.u32 {split}, %ctaid.y;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {wl}, {tid}, 31;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), ws = Param64(b, "p_ws"), lim = Param64(b, "p_lim"), starts = Param64(b, "p_start");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d"), splits = Param32(b, "p_splits");
        string scale = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {scale}, [p_scale];");
        string cap = "";
        string h = b.R(), i = b.R(), t = b.R(), limit = b.R(), start = b.R(), la = b.RD(), sa = b.RD();
        b.L($"div.u32 {h}, {bid}, {lq};");
        b.L($"mul.lo.u32 {t}, {h}, {lq};");
        b.L($"sub.u32 {i}, {bid}, {t};");
        if (cache is null)
        {
            b.L($"mul.wide.u32 {la}, {i}, 4;");
            b.L($"add.u64 {la}, {lim}, {la};");
            b.L($"ld.global.s32 {limit}, [{la}];");
            b.L($"min.u32 {limit}, {limit}, {lk};");
            b.L($"mul.wide.u32 {sa}, {i}, 4;");
            b.L($"add.u64 {sa}, {starts}, {sa};");
            b.L($"ld.global.s32 {start}, [{sa}];");
        }
        else
        {
            string lq0 = Param32(b, "p_lq0"), win = Param32(b, "p_win"), pos = b.R(), i0 = b.R(), noWin = b.P();
            cap = lane.NewReg(b);
            b.L($"ld.param.{lane.Op} {cap}, [p_cap];");
            b.L($"ld.global.s32 {pos}, [{lim}];");
            b.L($"rem.u32 {i0}, {i}, {lq0};");
            b.L($"add.s32 {limit}, {pos}, {i0};");
            b.L($"add.s32 {limit}, {limit}, 1;");
            b.L($"max.s32 {limit}, {limit}, 0;");
            b.L($"min.s32 {limit}, {limit}, {lk};");
            b.L($"sub.s32 {start}, {limit}, {win};");
            b.L($"max.s32 {start}, {start}, 0;");
            b.L($"setp.eq.u32 {noWin}, {win}, 0;");
            b.L($"selp.u32 {start}, 0, {start}, {noWin};");
        }

        b.L($"min.u32 {start}, {start}, {limit};");
        // This warp's part: [start + part·per, …) of the query's keys, per = ⌈(limit − start) / parts⌉.
        string parts = b.R(), part = b.R(), total = b.R(), per = b.R(), lo = b.R(), hi = b.R();
        b.L($"shl.b32 {parts}, {splits}, 3;");
        b.L($"mad.lo.u32 {part}, {split}, 8, {warp};");
        b.L($"sub.u32 {total}, {limit}, {start};");
        b.L($"add.u32 {per}, {total}, {parts};");
        b.L($"sub.u32 {per}, {per}, 1;");
        b.L($"div.u32 {per}, {per}, {parts};");
        b.L($"mad.lo.u32 {lo}, {part}, {per}, {start};");
        b.L($"min.u32 {lo}, {lo}, {limit};");
        b.L($"add.u32 {hi}, {lo}, {per};");
        b.L($"min.u32 {hi}, {hi}, {limit};");
        string qBase = b.R(), kBase = b.R(), hk = b.R();
        b.L($"mul.lo.u32 {qBase}, {bid}, {d};");
        b.L($"mul.lo.u32 {hk}, {h}, {lk};");
        b.L($"mul.lo.u32 {kBase}, {hk}, {d};");
        // The query's dimensions this lane owns, in registers (0 past d); which of them are inside d.
        var qr = new string[perLane];
        var inD = new string[perLane];
        var dim = new string[perLane];
        string dm1 = b.R();
        b.L($"sub.u32 {dm1}, {d}, 1;");
        for (int e = 0; e < perLane; e++)
        {
            dim[e] = b.R(); inD[e] = b.P(); qr[e] = lane.NewReg(b);
            string at = b.R(), idx = b.R();
            if (runs) b.L($"mad.lo.u32 {dim[e]}, {wl}, {Run}, {e};");
            else b.L($"add.u32 {dim[e]}, {wl}, {32 * e};");
            b.L($"setp.lt.u32 {inD[e]}, {dim[e]}, {d};");
            b.L($"min.u32 {at}, {dim[e]}, {dm1};");
            b.L($"add.u32 {idx}, {qBase}, {at};");
            string qv = Codec.UnpackAt(b, f, q, idx, lane);
            b.L($"selp.{lane.Op} {qr[e]}, {qv}, {lane.Zero}, {inD[e]};");
        }

        const int G = 4;
        string negInf = lane.Literal(double.NegativeInfinity);
        string m = lane.NewReg(b), l = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {m}, {negInf};");
        b.L($"mov.{lane.Reg} {l}, {lane.Zero};");
        var acc = new string[perLane];
        for (int e = 0; e < perLane; e++) { acc[e] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[e]}, {lane.Zero};"); }
        string key0 = b.R(), pDone = b.P(), hm1 = b.R();
        b.L($"mov.u32 {key0}, {lo};");
        b.Label("KEYS");
        b.L($"setp.ge.u32 {pDone}, {key0}, {hi};");
        b.L($"@{pDone} bra KEYS_END;");
        b.L($"sub.u32 {hm1}, {hi}, 1;");
        var valid = new string[G];
        var rowAt = new string[G];
        var s = new string[G];
        for (int g = 0; g < G; g++)
        {
            string key = b.R(), kc = b.R();
            valid[g] = b.P(); rowAt[g] = b.R(); s[g] = lane.NewReg(b);
            b.L($"add.u32 {key}, {key0}, {g};");
            b.L($"setp.lt.u32 {valid[g]}, {key}, {hi};");
            b.L($"min.u32 {kc}, {key}, {hm1};");
            b.L($"mad.lo.u32 {rowAt[g]}, {kc}, {d}, {kBase};");
            b.L($"mov.{lane.Reg} {s[g]}, {lane.Zero};");
        }

        // A key's run of this lane's 8 dimensions (runs): decoded at once, 0 where the run is past d.
        string[] RunOf(string src, string row)
        {
            string idx = b.R();
            b.L($"add.u32 {idx}, {row}, {dim[0]};");
            b.L($"@!{inD[0]} mov.u32 {idx}, {row};");
            var vals = Codec.UnpackRun(b, fk, src, idx, Run, lane);
            var kept = new string[Run];
            for (int e = 0; e < Run; e++) { kept[e] = lane.NewReg(b); b.L($"selp.{lane.Op} {kept[e]}, {vals[e]}, {lane.Zero}, {inD[0]};"); }
            return kept;
        }

        // The scores: this lane's products, then the warp's sum by butterflies (every lane holds it).
        if (runs)
        {
            for (int g = 0; g < G; g++)
            {
                var kr = RunOf(k, rowAt[g]);
                for (int e = 0; e < Run; e++) b.L(lane.Fma(s[g], qr[e], kr[e], s[g]));
            }
        }
        else for (int e = 0; e < perLane; e++)
        {
            for (int g = 0; g < G; g++)
            {
                string idx = b.R(), kv = lane.NewReg(b);
                b.L($"add.u32 {idx}, {rowAt[g]}, {dim[e]};");
                b.L($"@!{inD[e]} mov.u32 {idx}, {rowAt[g]};");
                string loaded = Codec.UnpackAt(b, fk, k, idx, lane);
                b.L($"selp.{lane.Op} {kv}, {loaded}, {lane.Zero}, {inD[e]};");
                b.L(lane.Fma(s[g], qr[e], kv, s[g]));
            }
        }

        for (int g = 0; g < G; g++) WarpReduce(b, lane, s[g], "add");
        string tmax = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {tmax}, {negInf};");
        for (int g = 0; g < G; g++)
        {
            if (cache is { Capped: true })
            {
                string capped = Fn.Tanh(b, lane, LaneOp(b, lane, "div", LaneOp(b, lane, "mul", scale, s[g]), cap));
                b.L(lane.Mul(s[g], cap, capped));
            }
            else b.L(lane.Mul(s[g], s[g], scale));

            b.L($"selp.{lane.Op} {s[g]}, {s[g]}, {negInf}, {valid[g]};");
            b.L($"max.{lane.Op} {tmax}, {tmax}, {s[g]};");
        }

        string mNew = lane.NewReg(b), mUse = lane.NewReg(b), pInf = b.P(), dmv = lane.NewReg(b);
        b.L($"max.{lane.Op} {mNew}, {m}, {tmax};");
        b.L($"setp.eq.{lane.Op} {pInf}, {mNew}, {negInf};");
        b.L($"selp.{lane.Op} {mUse}, {lane.Zero}, {mNew}, {pInf};");
        b.L($"sub.rn.{lane.Op} {dmv}, {m}, {mUse};");
        string alpha = Codec.Exp(b, lane, dmv);
        var p = new string[G];
        string psum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {psum}, {lane.Zero};");
        for (int g = 0; g < G; g++)
        {
            string ds = lane.NewReg(b);
            p[g] = lane.NewReg(b);
            b.L($"sub.rn.{lane.Op} {ds}, {s[g]}, {mUse};");
            string e = Codec.Exp(b, lane, ds);
            b.L($"selp.{lane.Op} {p[g]}, {e}, {lane.Zero}, {valid[g]};");
            b.L(lane.Add(psum, psum, p[g]));
        }

        b.L(lane.Fma(l, l, alpha, psum));
        b.L($"mov.{lane.Reg} {m}, {mNew};");
        // The context: acc = alpha·acc + Σ_g p_g · v_g (this lane's dimensions).
        if (runs)
        {
            for (int e = 0; e < Run; e++) b.L(lane.Mul(acc[e], acc[e], alpha));
            for (int g = 0; g < G; g++)
            {
                var vr = RunOf(v, rowAt[g]);
                for (int e = 0; e < Run; e++) b.L(lane.Fma(acc[e], p[g], vr[e], acc[e]));
            }
        }
        else for (int e = 0; e < perLane; e++)
        {
            b.L(lane.Mul(acc[e], acc[e], alpha));
            for (int g = 0; g < G; g++)
            {
                string idx = b.R(), vv = lane.NewReg(b);
                b.L($"add.u32 {idx}, {rowAt[g]}, {dim[e]};");
                b.L($"@!{inD[e]} mov.u32 {idx}, {rowAt[g]};");
                string loaded = Codec.UnpackAt(b, fk, v, idx, lane);
                b.L($"selp.{lane.Op} {vv}, {loaded}, {lane.Zero}, {inD[e]};");
                b.L(lane.Fma(acc[e], p[g], vv, acc[e]));
            }
        }

        b.L($"add.u32 {key0}, {key0}, {G};");
        b.L("bra KEYS;");
        b.Label("KEYS_END");
        // This part's m, l and context into p_ws at (bid · parts + part) · (2 + dCap) lanes.
        string slot = b.R(), wAddr = b.RD(), p0 = b.P();
        b.L($"mad.lo.u32 {slot}, {bid}, {parts}, {part};");
        b.L($"mul.lo.u32 {slot}, {slot}, {DecodePartialLanes(dCap)};");
        b.L($"mul.wide.u32 {wAddr}, {slot}, {lane.Size};");
        b.L($"add.u64 {wAddr}, {ws}, {wAddr};");
        b.L($"setp.ne.u32 {p0}, {wl}, 0;");
        b.L($"@!{p0} st.global.{lane.Mem} [{wAddr}], {m};");
        b.L($"@!{p0} st.global.{lane.Mem} [{wAddr}+{lane.Size}], {l};");
        for (int e = 0; e < perLane; e++)
        {
            string at = b.RD();
            b.L($"mul.wide.u32 {at}, {dim[e]}, {lane.Size};");
            b.L($"add.u64 {at}, {wAddr}, {at};");
            b.L($"@{inD[e]} st.global.{lane.Mem} [{at}+{2 * lane.Size}], {acc[e]};");          // dimensions past d (a run's lanes past it) not stored
        }

        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        string over = cache is null ? "" : $", .param .u32 p_lq0, .param .u32 p_win, .param .{pa} p_cap";
        return b.Finish(Header(gpu), entry, $".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_ws, .param .u64 p_lim, .param .u64 p_start, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .{pa} p_scale, .param .u32 p_splits{over}");
    }

    /// <summary>The split decode's parts joined (AttentionDecodeSplit): a block a (head, query), M the parts' largest m, L =
    /// Σ l·e^(m − M) and O = Σ context·e^(m − M) · (1/L), the parts in order; O 0 where no key was seen. A thread a
    /// dimension (tid, tid + 256, …), O stored value by value as the type (outFmt: lane storage for a block format's).</summary>
    public static string DecodeCombine(Gpu gpu, Fmt f, Lane lane, int dCap, out string entry, Fmt? outFmt = null)
    {
        entry = $"attention_decode_combine_{f.Name}_{lane.Name}_d{dCap}{(outFmt is null ? "" : "_o" + outFmt.Name)}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bid = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        string ws = Param64(b, "p_ws"), o = Param64(b, "p_o"), parts = Param32(b, "p_parts"), d = Param32(b, "p_d");
        string negInf = lane.Literal(double.NegativeInfinity);
        string first = b.R(), at = b.R(), pp = b.R(), pEnd = b.P(), addr = b.RD(), mp = lane.NewReg(b), lp = lane.NewReg(b);
        b.L($"mul.lo.u32 {first}, {bid}, {parts};");
        // M: the parts' largest m.
        string mx = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {mx}, {negInf};");
        b.L($"mov.u32 {pp}, 0;");
        b.Label("MAX");
        b.L($"setp.ge.u32 {pEnd}, {pp}, {parts};");
        b.L($"@{pEnd} bra MAX_END;");
        b.L($"add.u32 {at}, {first}, {pp};");
        b.L($"mul.lo.u32 {at}, {at}, {DecodePartialLanes(dCap)};");
        b.L($"mul.wide.u32 {addr}, {at}, {lane.Size};");
        b.L($"add.u64 {addr}, {ws}, {addr};");
        b.L($"ld.global.{lane.Mem} {mp}, [{addr}];");
        b.L($"max.{lane.Op} {mx}, {mx}, {mp};");
        b.L($"add.u32 {pp}, {pp}, 1;");
        b.L("bra MAX;");
        b.Label("MAX_END");
        string pNone = b.P(), mUse = lane.NewReg(b);
        b.L($"setp.eq.{lane.Op} {pNone}, {mx}, {negInf};");
        b.L($"selp.{lane.Op} {mUse}, {lane.Zero}, {mx}, {pNone};");
        const int PerThread = 4;
        var acc = new string[PerThread];
        for (int a = 0; a < PerThread; a++) { acc[a] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[a]}, {lane.Zero};"); }
        string lsum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {lsum}, {lane.Zero};");
        string p2 = b.R(), pEnd2 = b.P();
        b.L($"mov.u32 {p2}, 0;");
        b.Label("JOIN");
        b.L($"setp.ge.u32 {pEnd2}, {p2}, {parts};");
        b.L($"@{pEnd2} bra JOIN_END;");
        string at2 = b.R(), a2 = b.RD(), dmv = lane.NewReg(b);
        b.L($"add.u32 {at2}, {first}, {p2};");
        b.L($"mul.lo.u32 {at2}, {at2}, {DecodePartialLanes(dCap)};");
        b.L($"mul.wide.u32 {a2}, {at2}, {lane.Size};");
        b.L($"add.u64 {a2}, {ws}, {a2};");
        b.L($"ld.global.{lane.Mem} {mp}, [{a2}];");
        b.L($"ld.global.{lane.Mem} {lp}, [{a2}+{lane.Size}];");
        b.L($"sub.rn.{lane.Op} {dmv}, {mp}, {mUse};");
        string w = Codec.Exp(b, lane, dmv);
        b.L(lane.Fma(lsum, lp, w, lsum));
        for (int a = 0; a < PerThread; a++)
        {
            string dimR = b.R(), pDim = b.P(), cv = lane.NewReg(b), ca = b.RD();
            b.L($"add.u32 {dimR}, {tid}, {a * DecodeThreads};");
            b.L($"setp.lt.u32 {pDim}, {dimR}, {d};");
            b.L($"mul.wide.u32 {ca}, {dimR}, {lane.Size};");
            b.L($"add.u64 {ca}, {a2}, {ca};");
            b.L($"mov.{lane.Reg} {cv}, {lane.Zero};");
            b.L($"@{pDim} ld.global.{lane.Mem} {cv}, [{ca}+{2 * lane.Size}];");
            b.L(lane.Fma(acc[a], cv, w, acc[a]));
        }

        b.L($"add.u32 {p2}, {p2}, 1;");
        b.L("bra JOIN;");
        b.Label("JOIN_END");
        string inv = lane.NewReg(b), pAny = b.P();
        b.L($"setp.gt.{lane.Op} {pAny}, {lsum}, {lane.Zero};");
        b.L($"div.rn.{lane.Op} {inv}, {lane.Literal(1)}, {lsum};");
        b.L($"selp.{lane.Op} {inv}, {inv}, {lane.Zero}, {pAny};");
        var fo = outFmt ?? f;
        var natural = fo.Natural;
        string qBase = b.R();
        b.L($"mul.lo.u32 {qBase}, {bid}, {d};");
        for (int a = 0; a < PerThread; a++)
        {
            string dimR = b.R(), pDim = b.P(), val = lane.NewReg(b), idx = b.R(), oa = b.RD();
            b.L($"add.u32 {dimR}, {tid}, {a * DecodeThreads};");
            b.L($"setp.ge.u32 {pDim}, {dimR}, {d};");
            b.L($"@{pDim} bra OUT{a}_END;");
            b.L(lane.Mul(val, acc[a], inv));
            b.L($"add.u32 {idx}, {qBase}, {dimR};");
            string stored = lane == natural ? val : Codec.Convert(b, lane, val, natural, fo);
            if (fo.Bits >= 32)
            {
                string code = fo.WordBits == 64 ? b.RD() : b.R();
                if (fo.WordBits == 32) b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, fo, stored, natural, 0, code);
                b.L($"mul.wide.u32 {oa}, {idx}, {fo.Bits / 8};");
                b.L($"add.u64 {oa}, {o}, {oa};");
                b.L($"st.global.{(fo.WordBits == 64 ? "b64" : "b32")} [{oa}], {code};");
            }
            else
            {
                string code = b.R(), hh = b.H();
                b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, fo, stored, natural, 0, code);
                b.L($"mul.wide.u32 {oa}, {idx}, {fo.Bits / 8};");
                b.L($"add.u64 {oa}, {o}, {oa};");
                b.L($"cvt.u16.u32 {hh}, {code};");
                b.L($"st.global.b{fo.Bits} [{oa}], {hh};");
            }

            b.Label($"OUT{a}_END");
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_ws, .param .u64 p_o, .param .u32 p_parts, .param .u32 p_d");
    }

    // ---------------------------------------------------------------- lane storage: widen a format to its float lane's storage, and back

    /// <summary>out[i] (lane storage) = x[i] (format f): a thread per value.</summary>
    public static string Widen(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"widen_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
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
        var b = new PtxBuilder(gpu);
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
