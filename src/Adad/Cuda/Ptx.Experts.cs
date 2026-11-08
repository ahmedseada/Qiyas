// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts' kernels on CUDA (GpuOps.Experts.cs; Experts.cs): the routing's ranks (a thread per token and
// expert: how many of the token's experts are more probable, or as probable at a lower index), its picks (a thread per
// token: the experts of rank below k in increasing order, their probabilities in the lane, normalized by their sum in that
// order) and its rows (from the route's order); the experts' tiles (each expert's ⌈rows / tile⌉, prefix-summed: the grouped
// GEMM's grid, Ptx.Gemm) and each row's expert; and the store: y = gate(g + gb[e], u + ub[e]) (or g + gb[e]) of the lane
// sums, the biases read where they are stored, the gate Map's formula in the lane, rounded to the type once.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    // The thread's global index; past `count`, to DONE.
    private static string EachIndex(PtxBuilder b, string count)
    {
        string tid = b.R(), bid = b.R(), i = b.R(), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {i}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {i}, {count};");
        b.L($"@{pEnd} bra DONE;");
        return i;
    }

    private static void StoreU32(PtxBuilder b, string ptr, string index, string value)
    {
        string a = b.RD();
        b.L($"mul.wide.u32 {a}, {index}, 4;");
        b.L($"add.u64 {a}, {ptr}, {a};");
        b.L($"st.global.u32 [{a}], {value};");
    }

    private static string LoadU32(PtxBuilder b, string ptr, string index)
    {
        string a = b.RD(), v = b.R();
        b.L($"mul.wide.u32 {a}, {index}, 4;");
        b.L($"add.u64 {a}, {ptr}, {a};");
        b.L($"ld.global.u32 {v}, [{a}];");
        return v;
    }

    /// <summary>flags[t·E + e] = 1 where expert e's rank among token t's probabilities is below k. Params: p, flags, count
    /// (tokens·E), experts, k.</summary>
    public static string ExpertRanks(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"expert_ranks_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string pp = Param64(b, "p_p"), pf = Param64(b, "p_flags"), count = Param32(b, "p_count"), E = Param32(b, "p_experts"), k = Param32(b, "p_k");
        string i = EachIndex(b, count);
        string t = b.R(), e = b.R(), row = b.R(), at = b.R(), q = b.R(), rank = b.R(), inc = b.R();
        string pDone = b.P(), pGt = b.P(), pEq = b.P(), pLo = b.P();
        b.L($"div.u32 {t}, {i}, {E};");
        b.L($"rem.u32 {e}, {i}, {E};");
        b.L($"mul.lo.u32 {row}, {t}, {E};");
        b.L($"add.u32 {at}, {row}, {e};");
        string pe = Codec.UnpackAt(b, f, pp, at, lane);
        b.L($"mov.u32 {q}, 0;");
        b.L($"mov.u32 {rank}, 0;");
        b.Label("RLOOP");
        b.L($"setp.ge.u32 {pDone}, {q}, {E};");
        b.L($"@{pDone} bra RDONE;");
        string qa = b.R();
        b.L($"add.u32 {qa}, {row}, {q};");
        string pq = Codec.UnpackAt(b, f, pp, qa, lane);
        b.L($"setp.gt.{lane.Op} {pGt}, {pq}, {pe};");
        b.L($"setp.eq.{lane.Op} {pEq}, {pq}, {pe};");
        b.L($"setp.lt.u32 {pLo}, {q}, {e};");
        b.L($"and.pred {pEq}, {pEq}, {pLo};");
        b.L($"or.pred {pGt}, {pGt}, {pEq};");
        b.L($"selp.u32 {inc}, 1, 0, {pGt};");
        b.L($"add.u32 {rank}, {rank}, {inc};");
        b.L($"add.u32 {q}, {q}, 1;");
        b.L("bra RLOOP;");
        b.Label("RDONE");
        string flag = b.R(), pK = b.P();
        b.L($"setp.lt.u32 {pK}, {rank}, {k};");
        b.L($"selp.u32 {flag}, 1, 0, {pK};");
        StoreU32(b, pf, i, flag);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_p, .param .u64 p_flags, .param .u32 p_count, .param .u32 p_experts, .param .u32 p_k");
    }

    /// <summary>A thread per token: the flagged experts in increasing order at its slots (chosen, s32) and their probabilities
    /// (lane storage), normalized (p_norm 1) by their sum in that order. Params: p, flags, chosen, lanes, tokens, experts, k, norm.</summary>
    public static string ExpertPicks(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"expert_picks_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string pp = Param64(b, "p_p"), pf = Param64(b, "p_flags"), pc = Param64(b, "p_chosen"), pl = Param64(b, "p_lanes");
        string tokens = Param32(b, "p_tokens"), E = Param32(b, "p_experts"), k = Param32(b, "p_k"), norm = Param32(b, "p_norm");
        string t = EachIndex(b, tokens);
        string row = b.R(), slot0 = b.R(), e = b.R(), s = b.R(), pDone = b.P(), pSkip = b.P();
        string sum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
        b.L($"mul.lo.u32 {row}, {t}, {E};");
        b.L($"mul.lo.u32 {slot0}, {t}, {k};");
        b.L($"mov.u32 {e}, 0;");
        b.L($"mov.u32 {s}, 0;");
        b.Label("PLOOP");
        b.L($"setp.ge.u32 {pDone}, {e}, {E};");
        b.L($"@{pDone} bra PDONE;");
        string at = b.R();
        b.L($"add.u32 {at}, {row}, {e};");
        string flag = LoadU32(b, pf, at);
        b.L($"setp.eq.u32 {pSkip}, {flag}, 0;");
        b.L($"@{pSkip} bra PNEXT;");
        string slot = b.R();
        b.L($"add.u32 {slot}, {slot0}, {s};");
        StoreU32(b, pc, slot, e);
        string v = Codec.UnpackAt(b, f, pp, at, lane);
        string la = b.RD();
        b.L($"mul.wide.u32 {la}, {slot}, {lane.Size};");
        b.L($"add.u64 {la}, {pl}, {la};");
        b.L($"st.global.{lane.Mem} [{la}], {v};");
        b.L($"add.rn.{lane.Op} {sum}, {sum}, {v};");
        b.L($"add.u32 {s}, {s}, 1;");
        b.Label("PNEXT");
        b.L($"add.u32 {e}, {e}, 1;");
        b.L("bra PLOOP;");
        b.Label("PDONE");
        string pNo = b.P(), i = b.R(), pEnd = b.P();
        b.L($"setp.eq.u32 {pNo}, {norm}, 0;");
        b.L($"@{pNo} bra DONE;");
        b.L($"mov.u32 {i}, 0;");
        b.Label("NLOOP");
        b.L($"setp.ge.u32 {pEnd}, {i}, {k};");
        b.L($"@{pEnd} bra DONE;");
        string ns = b.R(), na = b.RD(), w = lane.NewReg(b);
        b.L($"add.u32 {ns}, {slot0}, {i};");
        b.L($"mul.wide.u32 {na}, {ns}, {lane.Size};");
        b.L($"add.u64 {na}, {pl}, {na};");
        b.L($"ld.global.{lane.Mem} {w}, [{na}];");
        b.L($"div.rn.{lane.Op} {w}, {w}, {sum};");
        b.L($"st.global.{lane.Mem} [{na}], {w};");
        b.L($"add.u32 {i}, {i}, 1;");
        b.L("bra NLOOP;");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_p, .param .u64 p_flags, .param .u64 p_chosen, .param .u64 p_lanes, .param .u32 p_tokens, .param .u32 p_experts, .param .u32 p_k, .param .u32 p_norm");
    }

    /// <summary>A thread per row: slot = order[row], tokenOfRow[row] = slot / k, slotRow[slot] = row. Params: order,
    /// tokenOfRow, slotRow, slots, k.</summary>
    public static string ExpertRows(Gpu gpu, out string entry)
    {
        entry = "expert_rows";
        var b = new PtxBuilder();
        string po = Param64(b, "p_order"), pt = Param64(b, "p_token"), ps = Param64(b, "p_slot"), slots = Param32(b, "p_slots"), k = Param32(b, "p_k");
        string r = EachIndex(b, slots);
        string slot = LoadU32(b, po, r), token = b.R();
        b.L($"div.u32 {token}, {slot}, {k};");
        StoreU32(b, pt, r, token);
        StoreU32(b, ps, slot, r);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_order, .param .u64 p_token, .param .u64 p_slot, .param .u32 p_slots, .param .u32 p_k");
    }

    /// <summary>One thread: tiles[0] = 0, tiles[e + 1] = tiles[e] + ⌈max(offsets[e + 1] − offsets[e], 0) / tile⌉. Params:
    /// offsets, tiles, experts, tile.</summary>
    public static string ExpertTiles(Gpu gpu, out string entry)
    {
        entry = "expert_tiles";
        var b = new PtxBuilder();
        string po = Param64(b, "p_off"), pt = Param64(b, "p_tiles"), E = Param32(b, "p_experts"), tile = Param32(b, "p_tile");
        string tid = b.R(), acc = b.R(), e = b.R(), pDone = b.P(), zero = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"setp.ne.u32 {pDone}, {tid}, 0;");
        b.L($"@{pDone} bra DONE;");
        b.L($"mov.u32 {acc}, 0;");
        b.L($"mov.u32 {zero}, 0;");
        StoreU32(b, pt, zero, acc);
        b.L($"mov.u32 {e}, 0;");
        b.Label("TLOOP");
        b.L($"setp.ge.u32 {pDone}, {e}, {E};");
        b.L($"@{pDone} bra DONE;");
        string e1 = b.R(), lo = b.R(), hi = b.R(), c = b.R();
        b.L($"add.u32 {e1}, {e}, 1;");
        b.L($"mov.u32 {lo}, {LoadU32(b, po, e)};");
        b.L($"mov.u32 {hi}, {LoadU32(b, po, e1)};");
        b.L($"sub.s32 {c}, {hi}, {lo};");
        b.L($"max.s32 {c}, {c}, 0;");
        b.L($"add.u32 {c}, {c}, {tile};");
        b.L($"sub.u32 {c}, {c}, 1;");
        b.L($"div.u32 {c}, {c}, {tile};");
        b.L($"add.u32 {acc}, {acc}, {c};");
        StoreU32(b, pt, e1, acc);
        b.L($"mov.u32 {e}, {e1};");
        b.L("bra TLOOP;");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_off, .param .u64 p_tiles, .param .u32 p_experts, .param .u32 p_tile");
    }

    /// <summary>A thread per row: rowExpert[r] = the last expert e with offsets[e] ≤ r, or experts where r is outside
    /// [offsets[0], offsets[experts]). Params: offsets, rowExpert, rows, experts.</summary>
    public static string ExpertOfRows(Gpu gpu, out string entry)
    {
        entry = "expert_of_rows";
        var b = new PtxBuilder();
        string po = Param64(b, "p_off"), pr = Param64(b, "p_rows"), rows = Param32(b, "p_count"), E = Param32(b, "p_experts");
        string r = EachIndex(b, rows);
        string e = b.R(), j = b.R(), pDone = b.P(), pLe = b.P();
        b.L($"mov.u32 {e}, 0;");
        b.L($"mov.u32 {j}, 1;");
        b.Label("ELOOP");
        b.L($"setp.ge.u32 {pDone}, {j}, {E};");
        b.L($"@{pDone} bra EDONE;");
        string oj = LoadU32(b, po, j);
        b.L($"setp.le.s32 {pLe}, {oj}, {r};");
        b.L($"selp.u32 {e}, {j}, {e}, {pLe};");
        b.L($"add.u32 {j}, {j}, 1;");
        b.L("bra ELOOP;");
        b.Label("EDONE");
        string zero = b.R(), first = b.R(), last = b.R(), pIn = b.P(), pLast = b.P();
        b.L($"mov.u32 {zero}, 0;");
        b.L($"mov.u32 {first}, {LoadU32(b, po, zero)};");
        b.L($"mov.u32 {last}, {LoadU32(b, po, E)};");
        b.L($"setp.ge.s32 {pIn}, {r}, {first};");
        b.L($"setp.lt.s32 {pLast}, {r}, {last};");
        b.L($"and.pred {pIn}, {pIn}, {pLast};");
        b.L($"selp.u32 {e}, {e}, {E}, {pIn};");
        StoreU32(b, pr, r, e);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_off, .param .u64 p_rows, .param .u32 p_count, .param .u32 p_experts");
    }

    /// <summary>y = gate(g + gb[e], u + ub[e]) (pair false: g + gb[e]) of the lane sums g and u, e the row's expert (0 outside
    /// every expert's rows). Params: g, gb, u, ub, rowExpert, y, n, cols, experts.</summary>
    public static string ExpertStore(Gpu gpu, Fmt f, Lane lane, string? gate, bool gateBias, bool upBias, out string entry)
    {
        entry = $"expert_store_{f.Name}_{lane.Name}_{gate ?? "none"}{(gateBias ? "_gb" : "")}{(upBias ? "_ub" : "")}";
        var b = new PtxBuilder();
        string pg = Param64(b, "p_g"), pgb = Param64(b, "p_gb"), pu = Param64(b, "p_u"), pub = Param64(b, "p_ub"), pre = Param64(b, "p_re"), py = Param64(b, "p_y");
        string n = Param32(b, "p_n"), cols = Param32(b, "p_cols"), E = Param32(b, "p_experts");
        var em = new Emit(b, lane);
        string Value(string at)
        {
            string r = b.R(), c = b.R(), ei = b.R(), bi = b.R(), pIn = b.P();
            b.L($"div.u32 {r}, {at}, {cols};");
            b.L($"rem.u32 {c}, {at}, {cols};");
            string e = LoadU32(b, pre, r);
            b.L($"setp.lt.u32 {pIn}, {e}, {E};");
            b.L($"selp.u32 {ei}, {e}, 0, {pIn};");
            b.L($"mad.lo.u32 {bi}, {ei}, {cols}, {c};");
            string g = LoadLane(b, lane, pg, at);
            if (gateBias) g = em.Add(g, Codec.UnpackAt(b, f, pgb, bi, lane));
            string v = g;
            if (gate is not null)
            {
                string u = LoadLane(b, lane, pu, at);
                if (upBias) u = em.Add(u, Codec.UnpackAt(b, f, pub, bi, lane));
                v = Formula(b, lane, gate, g, u, u, lane.Literal(0), lane.Literal(0));
            }

            string kept = lane.NewReg(b);
            b.L($"selp.{lane.Op} {kept}, {v}, {lane.Zero}, {pIn};");
            return kept;
        }

        StoreEach(b, f, lane, py, n, Value);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_g, .param .u64 p_gb, .param .u64 p_u, .param .u64 p_ub, .param .u64 p_re, .param .u64 p_y, .param .u32 p_n, .param .u32 p_cols, .param .u32 p_experts", EachShared(f));
    }
}
