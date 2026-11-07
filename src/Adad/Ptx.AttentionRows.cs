// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's fused attention rows (flash): a block per (head, Br query rows); the rows' queries in shared memory once;
// the keys walked in tiles of Bc through shared memory (K, then V in the same space); the scores, the online softmax
// and the context accumulated without the scores leaving the block. 256 threads: a row's tpr = 256 / Br threads share
// it — each scores Bc / tpr keys of the tile (a vector of the row's query against a vector of each key, 16 bytes a
// shared load) and owns d / tpr of the row's output dimensions in registers (runs of 16 bytes, so each P·V step is one
// vector load per run). The row's max and sum are reduced across its threads by shuffles; the tile's weights pass to the
// output through shared memory. The lane is the row family's: f32, f64 for float64; Q, K, V read as stored (any float
// format, decoded on the load into shared memory), O stored value by value as the type.
namespace Adad;

internal static partial class Ptx
{
    public const int RowsThreads = 256;

    /// <summary>Rows of queries and keys a block's tile holds: (32, 32) in f32, (16, 16) in f64.</summary>
    public static (int Br, int Bc) AttentionTile(Lane lane) => lane.Wide ? (16, 16) : (32, 32);

    /// <summary>Shared bytes of the fused rows: the queries, the keys' (or values') tile, the tile's weights.</summary>
    public static int AttentionRowsShared(Lane lane, int dCap)
    {
        var (br, bc) = AttentionTile(lane);
        int vec = 16 / lane.Size, stride = dCap + vec;
        return ((br + bc) * stride + br * (bc + 1)) * lane.Size;
    }

    public static string AttentionRows(Gpu gpu, Fmt f, Lane lane, int dCap, out string entry)
    {
        var (Br, Bc) = AttentionTile(lane);
        int T = RowsThreads, tpr = T / Br, spt = Bc / tpr, vec = 16 / lane.Size, stride = dCap + vec, nvec = dCap / (vec * tpr), size = lane.Size;
        int logTpr = System.Numerics.BitOperations.Log2((uint)tpr), logD = System.Numerics.BitOperations.Log2((uint)dCap);
        entry = $"attention_rows_{f.Name}_{lane.Name}_d{dCap}";
        var b = new PtxBuilder();
        string tid = b.R(), qb = b.R(), h = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {qb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), o = Param64(b, "p_o"), lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = lane.NewReg(b), negInf = lane.Literal(double.NegativeInfinity);
        b.L($"ld.param.{lane.Op} {scale}, [p_scale];");
        // This thread's row and place in it; the row's query index, keys [start, limit) (none past the queries).
        string r = b.R(), sub = b.R(), i = b.R(), rowValid = b.P(), iClamped = b.R(), lqm1 = b.R(), limit = b.R(), start = b.R(), addr = b.RD();
        b.L($"shr.u32 {r}, {tid}, {logTpr};");
        b.L($"and.b32 {sub}, {tid}, {tpr - 1};");
        b.L($"mad.lo.u32 {i}, {qb}, {Br}, {r};");
        b.L($"setp.lt.u32 {rowValid}, {i}, {lq};");
        b.L($"sub.u32 {lqm1}, {lq}, 1;");
        b.L($"min.u32 {iClamped}, {i}, {lqm1};");
        b.L($"mul.wide.u32 {addr}, {iClamped}, 4;");
        string la = b.RD(), sa = b.RD();
        b.L($"add.u64 {la}, {lim}, {addr};");
        b.L($"add.u64 {sa}, {starts}, {addr};");
        b.L($"ld.global.u32 {limit}, [{la}];");
        b.L($"ld.global.u32 {start}, [{sa}];");
        b.L($"min.u32 {limit}, {limit}, {lk};");
        b.L($"selp.b32 {limit}, {limit}, 0, {rowValid};");
        // The heads' first values; the block's key range.
        string qHead = b.R(), kHead = b.R(), lo = b.R(), hi = b.R(), ra = b.RD();
        b.L($"mul.lo.u32 {qHead}, {h}, {lq};");
        b.L($"mul.lo.u32 {qHead}, {qHead}, {d};");
        b.L($"mul.lo.u32 {kHead}, {h}, {lk};");
        b.L($"mul.lo.u32 {kHead}, {kHead}, {d};");
        b.L($"mul.wide.u32 {ra}, {qb}, 8;");
        b.L($"add.u64 {ra}, {ranges}, {ra};");
        b.L($"ld.global.u32 {lo}, [{ra}];");
        b.L($"ld.global.u32 {hi}, [{ra}+4];");
        // Shared: the queries, the key/value tile, the weights.
        string sQ = b.R(), sKV = b.R(), sP = b.R();
        b.L($"mov.u32 {sQ}, smem;");
        b.L($"add.u32 {sKV}, {sQ}, {Br * stride * size};");
        b.L($"add.u32 {sP}, {sKV}, {Bc * stride * size};");

        // rows [first, first + count) of a tensor (count rows of d values from value `head`; rows past `rowLimit` and
        // columns past d as 0) into shared at dst, stride `stride` lanes.
        // A thread's column of a tile is fixed (tid mod dCap) and its rows step by T / dCap: the index and the shared address
        // advance by constants.
        int rowsPerStep = T / dCap;
        string myCol = b.R(), myRow = b.R(), colValid = b.P(), stepValues = b.R(), mySlot = b.R();
        b.L($"and.b32 {myCol}, {tid}, {dCap - 1};");
        b.L($"shr.u32 {myRow}, {tid}, {logD};");
        b.L($"setp.lt.u32 {colValid}, {myCol}, {d};");
        b.L($"mul.lo.u32 {stepValues}, {d}, {rowsPerStep};");
        b.L($"mad.lo.u32 {mySlot}, {myRow}, {stride}, {myCol};");
        b.L($"mul.lo.u32 {mySlot}, {mySlot}, {size};");
        void LoadTile(string dst, int count, string head, string first, string rowLimit, string src)
        {
            string gr = b.R(), idx = b.R(), sBase = b.R();
            b.L($"add.u32 {gr}, {first}, {myRow};");
            b.L($"mad.lo.u32 {idx}, {gr}, {d}, {myCol};");
            b.L($"add.u32 {idx}, {idx}, {head};");
            b.L($"add.u32 {sBase}, {dst}, {mySlot};");
            for (int it = 0; it < count / rowsPerStep; it++)
            {
                string pv = b.P(), at = b.R(), val = lane.NewReg(b);
                if (it > 0) { b.L($"add.u32 {gr}, {gr}, {rowsPerStep};"); b.L($"add.u32 {idx}, {idx}, {stepValues};"); }
                b.L($"setp.lt.u32 {pv}, {gr}, {rowLimit};");
                b.L($"and.pred {pv}, {pv}, {colValid};");
                b.L($"selp.b32 {at}, {idx}, {head}, {pv};");                  // a dropped value reads the head's first (valid)
                string loaded = Codec.UnpackAt(b, f, src, at, lane);
                b.L($"selp.{lane.Op} {val}, {loaded}, {lane.Zero}, {pv};");
                b.L($"st.shared.{lane.Mem} [{sBase}+{it * rowsPerStep * stride * size}], {val};");
            }
        }

        // A reduction across the row's tpr threads (aligned lanes of one warp).
        void GroupReduce(string acc, string op)
        {
            for (int off = tpr / 2; off >= 1; off /= 2)
            {
                string other = lane.NewReg(b);
                if (lane.Wide)
                {
                    string l0 = b.R(), h0 = b.R(), l1 = b.R(), h1 = b.R();
                    b.L($"mov.b64 {{{l0}, {h0}}}, {acc};");
                    b.L($"shfl.sync.bfly.b32 {l1}, {l0}, {off}, 0x1f, 0xffffffff;");
                    b.L($"shfl.sync.bfly.b32 {h1}, {h0}, {off}, 0x1f, 0xffffffff;");
                    b.L($"mov.b64 {other}, {{{l1}, {h1}}};");
                }
                else
                {
                    string bits = b.R(), bits2 = b.R();
                    b.L($"mov.b32 {bits}, {acc};");
                    b.L($"shfl.sync.bfly.b32 {bits2}, {bits}, {off}, 0x1f, 0xffffffff;");
                    b.L($"mov.b32 {other}, {bits2};");
                }

                b.L(op == "max" ? $"max.{lane.Op} {acc}, {acc}, {other};" : lane.Add(acc, acc, other));
            }
        }

        string Vec(int n) { var regs = new string[n]; for (int e = 0; e < n; e++) regs[e] = lane.NewReg(b); return string.Join(", ", regs); }

        string qFirst = b.R();
        b.L($"mul.lo.u32 {qFirst}, {qb}, {Br};");
        LoadTile(sQ, Br, qHead, qFirst, lq, q);
        b.L("bar.sync 0;");

        // The row's running max, sum and output runs.
        string m = lane.NewReg(b), l = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {m}, {negInf};");
        b.L($"mov.{lane.Reg} {l}, {lane.Zero};");
        var acc = new string[nvec, vec];
        for (int j = 0; j < nvec; j++) for (int e = 0; e < vec; e++) { acc[j, e] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[j, e]}, {lane.Zero};"); }
        string qRow = b.R(), pRow = b.R(), vThread = b.R();
        b.L($"mad.lo.u32 {qRow}, {r}, {stride * size}, {sQ};");
        b.L($"mad.lo.u32 {pRow}, {r}, {(Bc + 1) * size}, {sP};");
        b.L($"mad.lo.u32 {vThread}, {sub}, {vec * size}, {sKV};");
        var c = new string[spt];
        var kRow = new string[spt];
        for (int jj = 0; jj < spt; jj++)
        {
            c[jj] = b.R(); kRow[jj] = b.R();
            b.L($"add.u32 {c[jj]}, {sub}, {tpr * jj};");
            b.L($"mad.lo.u32 {kRow[jj]}, {c[jj]}, {stride * size}, {sKV};");
        }

        string kt = b.R(), pDone = b.P();
        b.L($"mov.u32 {kt}, {lo};");
        b.Label("TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {hi};");
        b.L($"@{pDone} bra TILE_END;");
        LoadTile(sKV, Bc, kHead, kt, lk, k);
        b.L("bar.sync 0;");
        // Scores: s[jj] = q_r · k_c over the padded row (zeros past d).
        var s = new string[spt];
        for (int jj = 0; jj < spt; jj++) { s[jj] = lane.NewReg(b); b.L($"mov.{lane.Reg} {s[jj]}, {lane.Zero};"); }
        for (int t = 0; t < dCap; t += vec)
        {
            string qv = Vec(vec);
            b.L($"ld.shared.v{vec}.{lane.Mem} {{{qv}}}, [{qRow}+{t * size}];");
            var qs = qv.Split(", ");
            for (int jj = 0; jj < spt; jj++)
            {
                string kv = Vec(vec);
                b.L($"ld.shared.v{vec}.{lane.Mem} {{{kv}}}, [{kRow[jj]}+{t * size}];");
                var ks = kv.Split(", ");
                for (int e = 0; e < vec; e++) b.L(lane.Fma(s[jj], qs[e], ks[e], s[jj]));
            }
        }

        // The mask, the tile's max, the new running max (0 in the exponent while no key is seen).
        var valid = new string[spt];
        string tmax = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {tmax}, {negInf};");
        for (int jj = 0; jj < spt; jj++)
        {
            string key = b.R(), p1 = b.P();
            valid[jj] = b.P();
            b.L(lane.Mul(s[jj], s[jj], scale));
            b.L($"add.u32 {key}, {kt}, {c[jj]};");
            b.L($"setp.ge.u32 {valid[jj]}, {key}, {start};");
            b.L($"setp.lt.u32 {p1}, {key}, {limit};");
            b.L($"and.pred {valid[jj]}, {valid[jj]}, {p1};");
            b.L($"selp.{lane.Op} {s[jj]}, {s[jj]}, {negInf}, {valid[jj]};");
            b.L($"max.{lane.Op} {tmax}, {tmax}, {s[jj]};");
        }

        GroupReduce(tmax, "max");
        string mNew = lane.NewReg(b), mUse = lane.NewReg(b), pInf = b.P(), dm = lane.NewReg(b);
        b.L($"max.{lane.Op} {mNew}, {m}, {tmax};");
        b.L($"setp.eq.{lane.Op} {pInf}, {mNew}, {negInf};");
        b.L($"selp.{lane.Op} {mUse}, {lane.Zero}, {mNew}, {pInf};");
        b.L($"sub.rn.{lane.Op} {dm}, {m}, {mUse};");
        string alpha = Codec.Exp(b, lane, dm);
        string psum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {psum}, {lane.Zero};");
        for (int jj = 0; jj < spt; jj++)
        {
            string ds = lane.NewReg(b), p = lane.NewReg(b), pAddr = b.R();
            b.L($"sub.rn.{lane.Op} {ds}, {s[jj]}, {mUse};");
            string e = Codec.Exp(b, lane, ds);
            b.L($"selp.{lane.Op} {p}, {e}, {lane.Zero}, {valid[jj]};");
            b.L($"mad.lo.u32 {pAddr}, {c[jj]}, {size}, {pRow};");
            b.L($"st.shared.{lane.Mem} [{pAddr}], {p};");
            b.L(lane.Add(psum, psum, p));
        }

        GroupReduce(psum, "add");
        b.L(lane.Fma(l, l, alpha, psum));
        b.L($"mov.{lane.Reg} {m}, {mNew};");
        b.L("bar.sync 0;");
        LoadTile(sKV, Bc, kHead, kt, lk, v);
        b.L("bar.sync 0;");
        // The context: acc = alpha·acc + Σ_c p_c · v_c (this thread's runs of the row's dimensions).
        for (int j = 0; j < nvec; j++) for (int e = 0; e < vec; e++) b.L(lane.Mul(acc[j, e], acc[j, e], alpha));
        for (int cc = 0; cc < Bc; cc++)
        {
            string p = lane.NewReg(b);
            b.L($"ld.shared.{lane.Mem} {p}, [{pRow}+{cc * size}];");
            for (int j = 0; j < nvec; j++)
            {
                string vv = Vec(vec);
                b.L($"ld.shared.v{vec}.{lane.Mem} {{{vv}}}, [{vThread}+{(cc * stride + vec * tpr * j) * size}];");
                var vs = vv.Split(", ");
                for (int e = 0; e < vec; e++) b.L(lane.Fma(acc[j, e], p, vs[e], acc[j, e]));
            }
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {kt}, {kt}, {Bc};");
        b.L("bra TILE;");
        b.Label("TILE_END");

        // O = acc / l, value by value as the type.
        string inv = lane.NewReg(b), pAny = b.P(), oRow = b.R();
        b.L($"setp.gt.{lane.Op} {pAny}, {l}, {lane.Zero};");
        b.L($"div.rn.{lane.Op} {inv}, {lane.Literal(1)}, {l};");
        b.L($"selp.{lane.Op} {inv}, {inv}, {lane.Zero}, {pAny};");
        b.L($"mad.lo.u32 {oRow}, {i}, {d}, {qHead};");
        var natural = f.Natural;
        for (int j = 0; j < nvec; j++)
        {
            for (int e = 0; e < vec; e++)
            {
                string dim = b.R(), pDim = b.P(), val = lane.NewReg(b), idx = b.R(), oAddr = b.RD();
                string skip = $"OUT{j}_{e}";
                b.L($"mad.lo.u32 {dim}, {sub}, {vec}, {vec * tpr * j + e};");
                b.L($"setp.ge.u32 {pDim}, {dim}, {d};");
                b.L($"@{pDim} bra {skip};");
                b.L($"@!{rowValid} bra {skip};");
                b.L(lane.Mul(val, acc[j, e], inv));
                b.L($"add.u32 {idx}, {oRow}, {dim};");
                string stored = lane == natural ? val : Codec.Convert(b, lane, val, natural, f);
                if (f.Bits >= 32)
                {
                    string code = f.WordBits == 64 ? b.RD() : b.R();
                    if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
                    Codec.Pack(b, f, stored, natural, 0, code);
                    b.L($"mul.wide.u32 {oAddr}, {idx}, {f.Bits / 8};");
                    b.L($"add.u64 {oAddr}, {o}, {oAddr};");
                    b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{oAddr}], {code};");
                }
                else
                {
                    string code = b.R(), hh = b.H();
                    b.L($"mov.b32 {code}, 0;");
                    Codec.Pack(b, f, stored, natural, 0, code);
                    b.L($"mul.wide.u32 {oAddr}, {idx}, {f.Bits / 8};");
                    b.L($"add.u64 {oAddr}, {o}, {oAddr};");
                    b.L($"cvt.u16.u32 {hh}, {code};");
                    b.L($"st.global.b{f.Bits} [{oAddr}], {hh};");
                }

                b.Label(skip);
            }
        }

        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_o, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .{pa} p_scale",
            AttentionRowsShared(lane, dCap));
    }
}
