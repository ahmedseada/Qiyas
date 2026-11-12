// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Group normalization (GroupShape) and the rotary embedding by tables (RopeArgs) on CUDA, in the type's float lane, the
// CPU's arithmetic value for value (Cpu/Norms.cs):
//   GroupCoefs  a block per group: its statistics by block reductions over the group's values in strides (decoded where
//               they are), then — backward — the sums of γ·dy and γ·dy·(x − μ); the group's coefficients (μ, rstd, m1, k2)
//               kept as lane values;
//   GroupOut    a thread per stored word (a warp per block of a block format): each value with its group's coefficients
//               and its channel's gain and bias, the word packed and stored once (a word's values may span two groups);
//   GroupParams a block per channel: Σ dy·x̂ and Σ dy over the channel's values;
//   RopeOut     a thread per stored word (a warp per block): each value and its pair's partner, rotated by the tables.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    private const int GroupThreads = 256;

    // for (j = tid; j < count; j += GroupThreads) body(j)
    private static void StrideLoop(PtxBuilder b, string tid, string count, string label, Action<string> body)
    {
        string j = b.R(), p = b.P();
        b.L($"mov.u32 {j}, {tid};");
        b.Label(label);
        b.L($"setp.ge.u32 {p}, {j}, {count};");
        b.L($"@{p} bra {label}_END;");
        body(j);
        b.L($"add.u32 {j}, {j}, {GroupThreads};");
        b.L($"bra {label};");
        b.Label($"{label}_END");
    }

    private static string LaneOp(PtxBuilder b, Lane lane, string op, string x, string y)
    {
        string d = lane.NewReg(b);
        b.L($"{op}.rn.{lane.Op} {d}, {x}, {y};");
        return d;
    }

    private static string LoadLane(PtxBuilder b, Lane lane, string ptr, string index, int offset = 0)
    {
        string a = b.RD(), v = lane.NewReg(b);
        b.L($"mul.wide.u32 {a}, {index}, {lane.Size};");
        b.L($"add.u64 {a}, {ptr}, {a};");
        b.L($"ld.global.{lane.Mem} {v}, [{a}+{offset * lane.Size}];");
        return v;
    }

    private static void StoreLaneIf(PtxBuilder b, Lane lane, string ptr, string index, string v, string label)
    {
        string p = b.P(), a = b.RD();
        b.L($"setp.eq.u64 {p}, {ptr}, 0;");
        b.L($"@{p} bra {label};");
        b.L($"mul.wide.u32 {a}, {index}, {lane.Size};");
        b.L($"add.u64 {a}, {ptr}, {a};");
        b.L($"st.global.{lane.Mem} [{a}], {v};");
        b.Label(label);
    }

    // (index / inner) % groups, (index / spatial) % channels
    private static string Within(PtxBuilder b, string index, string inner, string count)
    {
        string t = b.R(), r = b.R();
        b.L($"div.u32 {t}, {index}, {inner};");
        b.L($"rem.u32 {r}, {t}, {count};");
        return r;
    }

    private static string Rstd(PtxBuilder b, Lane lane, string var, string eps)
    {
        string s = Fn.Sqrt(b, lane, LaneOp(b, lane, "add", var, eps));
        return LaneOp(b, lane, "div", lane.Literal(1), s);
    }

    /// <summary>A block per group: coef[4g..] = μ, rstd and (backward, statistics not given) m1 = Σ γ·dy / M, k2 = rstd·(rstd·Σ γ·dy·(x − μ) / M);
    /// the computed μ and σ² written to mean and variance when given (non-zero).</summary>
    public static string GroupCoefs(Gpu gpu, Fmt f, Lane lane, bool backward, bool given, bool gamma, out string entry)
    {
        entry = $"group_coefs_{f.Name}_{lane.Name}{(backward ? "_bwd" : "")}{(given ? "_given" : "")}{(gamma ? "_g" : "")}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), g = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {g}, %ctaid.x;");
        string px = Param64(b, "p_x"), pdy = Param64(b, "p_dy"), pg = Param64(b, "p_g"), pmean = Param64(b, "p_mean"), pvar = Param64(b, "p_var"), pcoef = Param64(b, "p_coef");
        string outer = Param32(b, "p_outer"), groups = Param32(b, "p_groups"), inner = Param32(b, "p_inner"), channels = Param32(b, "p_channels"), spatial = Param32(b, "p_spatial");
        string eps = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {eps}, [p_eps];");
        string m = b.R(), count = lane.NewReg(b);
        b.L($"mul.lo.u32 {m}, {outer}, {inner};");
        b.L($"cvt.rn.{lane.Op}.u32 {count}, {m};");

        // Value j of the group: row o = j / inner, i = j − o·inner, at (o·groups + g)·inner + i.
        string At(string j)
        {
            string o = b.R(), i = b.R(), idx = b.R();
            b.L($"div.u32 {o}, {j}, {inner};");
            b.L($"mul.lo.u32 {i}, {o}, {inner};");
            b.L($"sub.u32 {i}, {j}, {i};");
            b.L($"mad.lo.u32 {idx}, {o}, {groups}, {g};");
            b.L($"mad.lo.u32 {idx}, {idx}, {inner}, {i};");
            return idx;
        }

        string mu, rstd;
        if (given)
        {
            mu = LoadLane(b, lane, pmean, g);
            rstd = Rstd(b, lane, LoadLane(b, lane, pvar, g), eps);
        }
        else
        {
            string acc = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
            StrideLoop(b, tid, m, "SUM", j => b.L(lane.Add(acc, acc, Codec.UnpackAt(b, f, px, At(j), lane))));
            mu = LaneOp(b, lane, "div", BlockReduce(b, lane, tid, acc, "add", GroupThreads), count);
            string acc2 = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {acc2}, {lane.Zero};");
            StrideLoop(b, tid, m, "SQ", j =>
            {
                string d = LaneOp(b, lane, "sub", Codec.UnpackAt(b, f, px, At(j), lane), mu);
                b.L(lane.Fma(acc2, d, d, acc2));
            });
            string var = LaneOp(b, lane, "div", BlockReduce(b, lane, tid, acc2, "add", GroupThreads), count);
            rstd = Rstd(b, lane, var, eps);
            string p0 = b.P();
            b.L($"setp.ne.u32 {p0}, {tid}, 0;");
            b.L($"@{p0} bra STATS_DONE;");
            StoreLaneIf(b, lane, pmean, g, mu, "NO_MEAN");
            StoreLaneIf(b, lane, pvar, g, var, "NO_VAR");
            b.Label("STATS_DONE");
        }

        string m1 = lane.NewReg(b), k2 = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {m1}, {lane.Zero};");
        b.L($"mov.{lane.Reg} {k2}, {lane.Zero};");
        if (backward && !given)
        {
            string a1 = lane.NewReg(b), a2 = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {a1}, {lane.Zero};");
            b.L($"mov.{lane.Reg} {a2}, {lane.Zero};");
            StrideLoop(b, tid, m, "GD", j =>
            {
                string idx = At(j);
                string xv = Codec.UnpackAt(b, f, px, idx, lane), dv = Codec.UnpackAt(b, f, pdy, idx, lane);
                string d = LaneOp(b, lane, "sub", xv, mu);
                if (gamma)
                {
                    string gv = Codec.UnpackAt(b, f, pg, Within(b, idx, spatial, channels), lane);
                    b.L(lane.Fma(a1, gv, dv, a1));
                    b.L(lane.Fma(a2, LaneOp(b, lane, "mul", gv, dv), d, a2));
                }
                else
                {
                    b.L(lane.Add(a1, a1, dv));
                    b.L(lane.Fma(a2, dv, d, a2));
                }
            });
            string s1 = BlockReduce(b, lane, tid, a1, "add", GroupThreads), s2 = BlockReduce(b, lane, tid, a2, "add", GroupThreads);
            b.L($"div.rn.{lane.Op} {m1}, {s1}, {count};");
            string t = LaneOp(b, lane, "div", LaneOp(b, lane, "mul", rstd, s2), count);
            b.L(lane.Mul(k2, rstd, t));
        }

        string pt = b.P(), addr = b.RD();
        b.L($"setp.ne.u32 {pt}, {tid}, 0;");
        b.L($"@{pt} bra DONE;");
        b.L($"mul.wide.u32 {addr}, {g}, {4 * lane.Size};");
        b.L($"add.u64 {addr}, {pcoef}, {addr};");
        b.L($"st.global.{lane.Mem} [{addr}], {mu};");
        b.L($"st.global.{lane.Mem} [{addr}+{lane.Size}], {rstd};");
        b.L($"st.global.{lane.Mem} [{addr}+{2 * lane.Size}], {m1};");
        b.L($"st.global.{lane.Mem} [{addr}+{3 * lane.Size}], {k2};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_dy, .param .u64 p_g, .param .u64 p_mean, .param .u64 p_var, .param .u64 p_coef, .param .u32 p_outer, .param .u32 p_groups, .param .u32 p_inner, .param .u32 p_channels, .param .u32 p_spatial, .param .{lane.Mem} p_eps",
            ReduceBytes);
    }

    // A thread per stored word of an n-value array of f (a warp per block of a block format): value(index, x, dy) for each of its values.
    private static void WordOut(PtxBuilder b, Fmt f, Lane lane, string px, string pdy, string py, string n, bool second, Func<string, string, string?, string> value)
    {
        if (f.IsBlock)
        {
            var w = BlockWarp(b, f, n);
            EncodeBlock(b, f, py, w, n, at => value(at, Codec.UnpackAt(b, f, px, at, lane), second ? Codec.UnpackAt(b, f, pdy, at, lane) : null));
            return;
        }

        string gid = GlobalIndex(b), words = WordsOf(b, f, n), p = b.P();
        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        string off = b.RD(), ax = b.RD(), ad = b.RD(), ao = b.RD();
        b.L($"mul.wide.u32 {off}, {gid}, {f.WordBits / 8};");
        b.L($"add.u64 {ax}, {px}, {off};");
        b.L($"add.u64 {ad}, {pdy}, {off};");
        b.L($"add.u64 {ao}, {py}, {off};");
        string wx = f.WordBits == 64 ? b.RD() : b.R(), wd = f.WordBits == 64 ? b.RD() : b.R(), wo = f.WordBits == 64 ? b.RD() : b.R();
        b.L($"ld.global.{wt} {wx}, [{ax}];");
        if (second) b.L($"ld.global.{wt} {wd}, [{ad}];");
        if (f.WordBits == 32) b.L($"mov.b32 {wo}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R();
            b.L($"mad.lo.u32 {idx}, {gid}, {f.Per}, {i};");
            string x = Codec.Unpack(b, f, wx, i, lane);
            string? d = second ? Codec.Unpack(b, f, wd, i, lane) : null;
            Codec.Pack(b, f, value(idx, x, d), lane, i, wo);
        }

        b.L($"st.global.{wt} [{ao}], {wo};");
    }

    /// <summary>Forward: y = x̂·γ + β (x̂ = (x − μ)·rstd: fma with a gain and a bias, an add with a bias only, a product with a gain
    /// only). Backward: dx = (γ·dy − m1 − (x − μ)·k2)·rstd. Each value with its group's coefficients and its channel's γ, β.</summary>
    public static string GroupOut(Gpu gpu, Fmt f, Lane lane, bool backward, bool gamma, bool beta, out string entry)
    {
        entry = $"group_out_{f.Name}_{lane.Name}{(backward ? "_bwd" : "")}{(gamma ? "_g" : "")}{(beta ? "_b" : "")}";
        var b = new PtxBuilder(gpu);
        string px = Param64(b, "p_x"), pdy = Param64(b, "p_dy"), pg = Param64(b, "p_g"), pb = Param64(b, "p_b"), pcoef = Param64(b, "p_coef"), py = Param64(b, "p_y");
        string n = Param32(b, "p_n"), groups = Param32(b, "p_groups"), inner = Param32(b, "p_inner"), channels = Param32(b, "p_channels"), spatial = Param32(b, "p_spatial");
        WordOut(b, f, lane, px, pdy, py, n, backward, (idx, x, dy) =>
        {
            string g = Within(b, idx, inner, groups), c = Within(b, idx, spatial, channels);
            string mu = lane.NewReg(b), rstd = lane.NewReg(b);
            string ca = b.RD();
            b.L($"mul.wide.u32 {ca}, {g}, {4 * lane.Size};");
            b.L($"add.u64 {ca}, {pcoef}, {ca};");
            b.L($"ld.global.{lane.Mem} {mu}, [{ca}];");
            b.L($"ld.global.{lane.Mem} {rstd}, [{ca}+{lane.Size}];");
            string? gv = gamma ? Codec.UnpackAt(b, f, pg, c, lane) : null;
            string d = LaneOp(b, lane, "sub", x, mu);
            if (backward)
            {
                string m1 = lane.NewReg(b), k2 = lane.NewReg(b);
                b.L($"ld.global.{lane.Mem} {m1}, [{ca}+{2 * lane.Size}];");
                b.L($"ld.global.{lane.Mem} {k2}, [{ca}+{3 * lane.Size}];");
                string gd = gv is null ? dy! : LaneOp(b, lane, "mul", gv, dy!);
                string t = LaneOp(b, lane, "sub", LaneOp(b, lane, "sub", gd, m1), LaneOp(b, lane, "mul", d, k2));
                return LaneOp(b, lane, "mul", t, rstd);
            }

            string h = LaneOp(b, lane, "mul", d, rstd);
            if (beta)
            {
                string bv = Codec.UnpackAt(b, f, pb, c, lane);
                if (gv is null) return LaneOp(b, lane, "add", h, bv);
                string r = lane.NewReg(b);
                b.L(lane.Fma(r, h, gv, bv));
                return r;
            }

            return gv is null ? h : LaneOp(b, lane, "mul", h, gv);
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_x, .param .u64 p_dy, .param .u64 p_g, .param .u64 p_b, .param .u64 p_coef, .param .u64 p_y, .param .u32 p_n, .param .u32 p_groups, .param .u32 p_inner, .param .u32 p_channels, .param .u32 p_spatial");
    }

    /// <summary>A block per channel c: dgamma[c] = Σ dy·(x − μ)·rstd, dbeta[c] = Σ dy over its per values ((k·channels + c)·spatial + s), each
    /// output when non-zero (lane values).</summary>
    public static string GroupParams(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"group_params_{f.Name}_{lane.Name}";
        var b = new PtxBuilder(gpu);
        string tid = b.R(), c = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {c}, %ctaid.x;");
        string px = Param64(b, "p_x"), pdy = Param64(b, "p_dy"), pcoef = Param64(b, "p_coef"), pdg = Param64(b, "p_dg"), pdb = Param64(b, "p_db");
        string per = Param32(b, "p_per"), groups = Param32(b, "p_groups"), inner = Param32(b, "p_inner"), channels = Param32(b, "p_channels"), spatial = Param32(b, "p_spatial");
        string ag = lane.NewReg(b), ab = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {ag}, {lane.Zero};");
        b.L($"mov.{lane.Reg} {ab}, {lane.Zero};");
        StrideLoop(b, tid, per, "CH", j =>
        {
            string k = b.R(), s = b.R(), idx = b.R();
            b.L($"div.u32 {k}, {j}, {spatial};");
            b.L($"mul.lo.u32 {s}, {k}, {spatial};");
            b.L($"sub.u32 {s}, {j}, {s};");
            b.L($"mad.lo.u32 {idx}, {k}, {channels}, {c};");
            b.L($"mad.lo.u32 {idx}, {idx}, {spatial}, {s};");
            string g = Within(b, idx, inner, groups), ca = b.RD(), mu = lane.NewReg(b), rstd = lane.NewReg(b);
            b.L($"mul.wide.u32 {ca}, {g}, {4 * lane.Size};");
            b.L($"add.u64 {ca}, {pcoef}, {ca};");
            b.L($"ld.global.{lane.Mem} {mu}, [{ca}];");
            b.L($"ld.global.{lane.Mem} {rstd}, [{ca}+{lane.Size}];");
            string xv = Codec.UnpackAt(b, f, px, idx, lane), dv = Codec.UnpackAt(b, f, pdy, idx, lane);
            string h = LaneOp(b, lane, "mul", LaneOp(b, lane, "sub", xv, mu), rstd);
            b.L(lane.Fma(ag, dv, h, ag));
            b.L(lane.Add(ab, ab, dv));
        });
        string sg = BlockReduce(b, lane, tid, ag, "add", GroupThreads), sb = BlockReduce(b, lane, tid, ab, "add", GroupThreads);
        string p0 = b.P();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra DONE;");
        StoreLaneIf(b, lane, pdg, c, sg, "NO_DG");
        StoreLaneIf(b, lane, pdb, c, sb, "NO_DB");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_x, .param .u64 p_dy, .param .u64 p_coef, .param .u64 p_dg, .param .u64 p_db, .param .u32 p_per, .param .u32 p_groups, .param .u32 p_inner, .param .u32 p_channels, .param .u32 p_spatial",
            ReduceBytes);
    }

    /// <summary>The rotary embedding by tables: each value v and its pair's partner w: fma(∓w, s, v·c) (− for a pair's low
    /// value; s = −sin inverse) in a pair, v past 2·half.</summary>
    public static string RopeOut(Gpu gpu, Fmt f, Lane lane, bool interleaved, bool inverse, bool rotate, out string entry)
    {
        entry = $"rope_out_{f.Name}_{lane.Name}{(interleaved ? "_il" : "")}{(inverse ? "_inv" : "")}{(rotate ? "" : "_norot")}";
        var b = new PtxBuilder(gpu);
        string px = Param64(b, "p_x"), pcos = Param64(b, "p_cos"), psin = Param64(b, "p_sin"), ppos = Param64(b, "p_pos"), py = Param64(b, "p_y");
        string n = Param32(b, "p_n"), cols = Param32(b, "p_cols"), heads = Param32(b, "p_heads"), steps = Param32(b, "p_steps"), half = Param32(b, "p_half");
        string nm1 = b.R();
        b.L($"sub.u32 {nm1}, {n}, 1;");
        WordOut(b, f, lane, px, px, py, n, false, (at, x, _) =>
        {
            // A value past the last (in the last word) takes the last one's row and partner: no read past the arrays.
            string idx = b.R(), row = b.R(), col = b.R();
            b.L($"min.u32 {idx}, {at}, {nm1};");
            b.L($"div.u32 {row}, {idx}, {cols};");
            b.L($"mul.lo.u32 {col}, {row}, {cols};");
            b.L($"sub.u32 {col}, {idx}, {col};");
            string v = x;
            if (!rotate) return v;
            string st = Within(b, row, heads, steps), pa = b.RD(), pos = b.R();
            b.L($"mul.wide.u32 {pa}, {st}, 4;");
            b.L($"add.u64 {pa}, {ppos}, {pa};");
            b.L($"ld.global.u32 {pos}, [{pa}];");
            string low = b.P(), pr = b.R(), partner = b.R(), t = b.R();
            if (interleaved)
            {
                b.L($"and.b32 {t}, {col}, 1;");
                b.L($"setp.eq.u32 {low}, {t}, 0;");
                b.L($"shr.u32 {pr}, {col}, 1;");
                b.L($"xor.b32 {partner}, {col}, 1;");
            }
            else
            {
                string up = b.R();
                b.L($"setp.lt.u32 {low}, {col}, {half};");
                b.L($"sub.u32 {t}, {col}, {half};");
                b.L($"add.u32 {up}, {col}, {half};");
                b.L($"selp.u32 {pr}, {col}, {t}, {low};");
                b.L($"selp.u32 {partner}, {up}, {t}, {low};");
            }

            string twice = b.R(), inPair = b.P(), last = b.R(), hm1 = b.R(), pIdx = b.R(), tIdx = b.R();
            b.L($"shl.b32 {twice}, {half}, 1;");
            b.L($"setp.lt.u32 {inPair}, {col}, {twice};");
            b.L($"sub.u32 {last}, {cols}, 1;");
            b.L($"min.u32 {partner}, {partner}, {last};");
            b.L($"sub.u32 {hm1}, {half}, 1;");
            b.L($"min.u32 {pr}, {pr}, {hm1};");
            b.L($"mad.lo.u32 {pIdx}, {row}, {cols}, {partner};");
            string w = Codec.UnpackAt(b, f, px, pIdx, lane);
            b.L($"mad.lo.u32 {tIdx}, {pos}, {half}, {pr};");
            string cs = LoadLane(b, lane, pcos, tIdx), sn = LoadLane(b, lane, psin, tIdx);
            if (inverse) { string ns = lane.NewReg(b); b.L($"neg.{lane.Op} {ns}, {sn};"); sn = ns; }
            string nw = lane.NewReg(b), ws = lane.NewReg(b), r = lane.NewReg(b), o = lane.NewReg(b);
            b.L($"neg.{lane.Op} {nw}, {w};");
            b.L($"selp.{lane.Op} {ws}, {nw}, {w}, {low};");
            b.L(lane.Fma(r, ws, sn, LaneOp(b, lane, "mul", v, cs)));
            b.L($"selp.{lane.Op} {o}, {r}, {v}, {inPair};");
            return o;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_x, .param .u64 p_cos, .param .u64 p_sin, .param .u64 p_pos, .param .u64 p_y, .param .u32 p_n, .param .u32 p_cols, .param .u32 p_heads, .param .u32 p_steps, .param .u32 p_half");
    }
}
