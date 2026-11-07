// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's fused attention backward (item 6i): two kernels over the lane storage of Q, K, V and dO (f32, f64 for
// float64), no scores in memory, no atomics (each output value written by one thread: the same result every run).
//
//   rows (dQ)    a block per (head, Br query rows), as the fused rows: pass 1 the forward itself (the scores, the online
//                max and sum, the context) → the row's max m, 1/l, and D = dO · O (= Σ_j P·dP); pass 2 over the same key
//                tiles: P = e^(s − m)/l, dP = dO·Vᵀ, dS = scale·P·(dP − D), dQ += dS·K. m, 1/l, D stored per row.
//   keys (dK dV) a block per (head, Bc keys), the queries walked in tiles (only those whose keys meet the block):
//                P and dS of the tile recomputed from the rows' m, 1/l, D; dV += Pᵀ·dO, dK += dSᵀ·Q, in registers.
//
// 256 threads. Rows kernel: a row's 256 / Br threads share it as the fused rows (each scores Bc / tpr keys, owns d / tpr
// of the row's dimensions). Keys kernel: a key's 256 / Bc threads share it (each takes Br / tpk of the tile's rows for
// the scores, owns d / tpk of the key's dimensions of dK and dV). The mask and the softmax are the fused rows'.
namespace Qiyas;

internal static partial class Ptx
{
    /// <summary>Shared bytes of the backward's rows kernel: Q and dO of the rows, the K and V tiles, the tile's dS.</summary>
    public static int AttentionBackRowsShared(Lane lane, int dCap)
    {
        var (br, bc) = AttentionTile(lane);
        int stride = dCap + 16 / lane.Size;
        return ((2 * br + 2 * bc) * stride + br * (bc + 1)) * lane.Size;
    }

    /// <summary>Shared bytes of the backward's keys kernel: K and V of the keys, the Q and dO tiles, P and dS, the rows'
    /// statistics and key ranges.</summary>
    public static int AttentionBackKeysShared(Lane lane, int dCap)
    {
        var (br, bc) = AttentionTile(lane);
        int stride = dCap + 16 / lane.Size;
        return ((2 * bc + 2 * br) * stride + 2 * bc * (br + 1) + 3 * br) * lane.Size + 2 * br * 4;
    }

    // The pieces both kernels share: tile loads through shared memory, group reductions, vector registers.
    private sealed class BackPieces(PtxBuilder b, Lane lane, int dCap, string tid, string d)
    {
        public int Vec => 16 / lane.Size;
        public int Stride => dCap + Vec;

        // `count` rows [first, first + count) of a lane tensor (rows of d values from value `head`; rows past rowLimit and
        // dimensions past d as 0) into shared at dst. A thread's column is fixed (tid mod dCap), its rows step by 256 / dCap.
        public void LoadTile(string dst, int count, string head, string first, string rowLimit, string src)
        {
            int rowsPerStep = 256 / dCap, size = lane.Size;
            string myCol = b.R(), myRow = b.R(), colValid = b.P(), slot = b.R(), gr = b.R(), idx = b.R(), sBase = b.R();
            b.L($"and.b32 {myCol}, {tid}, {dCap - 1};");
            b.L($"shr.u32 {myRow}, {tid}, {System.Numerics.BitOperations.Log2((uint)dCap)};");
            b.L($"setp.lt.u32 {colValid}, {myCol}, {d};");
            b.L($"mad.lo.u32 {slot}, {myRow}, {Stride}, {myCol};");
            b.L($"mul.lo.u32 {slot}, {slot}, {size};");
            b.L($"add.u32 {gr}, {first}, {myRow};");
            b.L($"mad.lo.u32 {idx}, {gr}, {d}, {myCol};");
            b.L($"add.u32 {idx}, {idx}, {head};");
            b.L($"add.u32 {sBase}, {dst}, {slot};");
            for (int it = 0; it < count / rowsPerStep; it++)
            {
                string pv = b.P(), at = b.R(), val = lane.NewReg(b), g2 = b.R(), i2 = b.R(), addr = b.RD();
                b.L($"add.u32 {g2}, {gr}, {it * rowsPerStep};");
                b.L($"mad.lo.u32 {i2}, {d}, {it * rowsPerStep}, {idx};");
                b.L($"setp.lt.u32 {pv}, {g2}, {rowLimit};");
                b.L($"and.pred {pv}, {pv}, {colValid};");
                b.L($"selp.b32 {at}, {i2}, {head}, {pv};");
                b.L($"mul.wide.u32 {addr}, {at}, {size};");
                b.L($"add.u64 {addr}, {src}, {addr};");
                b.L($"ld.global.{lane.Mem} {val}, [{addr}];");
                b.L($"selp.{lane.Op} {val}, {val}, {lane.Zero}, {pv};");
                b.L($"st.shared.{lane.Mem} [{sBase}+{it * rowsPerStep * Stride * size}], {val};");
            }
        }

        // A reduction across `group` aligned lanes of one warp.
        public void GroupReduce(string acc, string op, int group)
        {
            for (int off = group / 2; off >= 1; off /= 2)
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

        public string[] Vector()
        {
            var regs = new string[Vec];
            for (int e = 0; e < Vec; e++) regs[e] = lane.NewReg(b);
            return regs;
        }

        // dst[jj] = Σ over the padded row of rowA's values · rowB[jj]'s values (shared addresses), 16 bytes a load.
        public void Dots(string[] dst, string rowA, string[] rowB)
        {
            for (int jj = 0; jj < dst.Length; jj++) b.L($"mov.{lane.Reg} {dst[jj]}, {lane.Zero};");
            for (int t = 0; t < dCap; t += Vec)
            {
                var av = Vector();
                b.L($"ld.shared.v{Vec}.{lane.Mem} {{{string.Join(", ", av)}}}, [{rowA}+{t * lane.Size}];");
                for (int jj = 0; jj < dst.Length; jj++)
                {
                    var bv = Vector();
                    b.L($"ld.shared.v{Vec}.{lane.Mem} {{{string.Join(", ", bv)}}}, [{rowB[jj]}+{t * lane.Size}];");
                    for (int e = 0; e < Vec; e++) b.L(lane.Fma(dst[jj], av[e], bv[e], dst[jj]));
                }
            }
        }

        // acc[j, e] += Σ_c w[c] · rows[c][this thread's runs]: weights from shared at wRow + c·wStep (lanes), rows from
        // shared at runs + c·Stride (this thread's first run at `runs`, the next runs runStep values on).
        public void Accumulate(string[,] acc, int count, string wRow, int wStep, string runs, int runStep)
        {
            int size = lane.Size, nvec = acc.GetLength(0);
            for (int c = 0; c < count; c++)
            {
                string w = lane.NewReg(b);
                b.L($"ld.shared.{lane.Mem} {w}, [{wRow}+{c * wStep * size}];");
                for (int j = 0; j < nvec; j++)
                {
                    var vv = Vector();
                    b.L($"ld.shared.v{Vec}.{lane.Mem} {{{string.Join(", ", vv)}}}, [{runs}+{(c * Stride + runStep * j) * size}];");
                    for (int e = 0; e < Vec; e++) b.L(lane.Fma(acc[j, e], w, vv[e], acc[j, e]));
                }
            }
        }

        public string[,] Zeros(int nvec)
        {
            var acc = new string[nvec, Vec];
            for (int j = 0; j < nvec; j++) for (int e = 0; e < Vec; e++) { acc[j, e] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[j, e]}, {lane.Zero};"); }
            return acc;
        }

        // A row of runs (this thread's: dimension sub·Vec + Vec·group·j + e) stored to dst[row·d + dimension] as lane values.
        public void StoreRuns(string[,] acc, string dst, string rowBase, string sub, int group, string valid, string label)
        {
            int nvec = acc.GetLength(0);
            for (int j = 0; j < nvec; j++)
            {
                for (int e = 0; e < Vec; e++)
                {
                    string dim = b.R(), pDim = b.P(), idx = b.R(), addr = b.RD(), skip = $"{label}{j}_{e}";
                    b.L($"mad.lo.u32 {dim}, {sub}, {Vec}, {Vec * group * j + e};");
                    b.L($"setp.ge.u32 {pDim}, {dim}, {d};");
                    b.L($"@{pDim} bra {skip};");
                    b.L($"@!{valid} bra {skip};");
                    b.L($"add.u32 {idx}, {rowBase}, {dim};");
                    b.L($"mul.wide.u32 {addr}, {idx}, {lane.Size};");
                    b.L($"add.u64 {addr}, {dst}, {addr};");
                    b.L($"st.global.{lane.Mem} [{addr}], {acc[j, e]};");
                    b.Label(skip);
                }
            }
        }
    }

    public static string AttentionBackRows(Gpu gpu, Lane lane, int dCap, out string entry)
    {
        var (Br, Bc) = AttentionTile(lane);
        int T = RowsThreads, tpr = T / Br, spt = Bc / tpr, size = lane.Size;
        entry = $"attention_back_rows_{lane.Name}_d{dCap}";
        var b = new PtxBuilder();
        string tid = b.R(), qb = b.R(), h = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {qb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), dO = Param64(b, "p_do"), dq = Param64(b, "p_dq"), stats = Param64(b, "p_stats");
        string lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = lane.NewReg(b), negInf = lane.Literal(double.NegativeInfinity);
        b.L($"ld.param.{lane.Op} {scale}, [p_scale];");
        var P = new BackPieces(b, lane, dCap, tid, d);
        int vec = P.Vec, stride = P.Stride, nvec = dCap / (vec * tpr);

        string r = b.R(), sub = b.R(), i = b.R(), rowValid = b.P(), iClamped = b.R(), lqm1 = b.R(), limit = b.R(), start = b.R(), addr = b.RD();
        b.L($"shr.u32 {r}, {tid}, {System.Numerics.BitOperations.Log2((uint)tpr)};");
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
        string qHead = b.R(), kHead = b.R(), lo = b.R(), hi = b.R(), ra = b.RD();
        b.L($"mul.lo.u32 {qHead}, {h}, {lq};");
        b.L($"mul.lo.u32 {qHead}, {qHead}, {d};");
        b.L($"mul.lo.u32 {kHead}, {h}, {lk};");
        b.L($"mul.lo.u32 {kHead}, {kHead}, {d};");
        b.L($"mul.wide.u32 {ra}, {qb}, 8;");
        b.L($"add.u64 {ra}, {ranges}, {ra};");
        b.L($"ld.global.u32 {lo}, [{ra}];");
        b.L($"ld.global.u32 {hi}, [{ra}+4];");
        string sQ = b.R(), sDO = b.R(), sK = b.R(), sV = b.R(), sP = b.R();
        b.L($"mov.u32 {sQ}, smem;");
        b.L($"add.u32 {sDO}, {sQ}, {Br * stride * size};");
        b.L($"add.u32 {sK}, {sDO}, {Br * stride * size};");
        b.L($"add.u32 {sV}, {sK}, {Bc * stride * size};");
        b.L($"add.u32 {sP}, {sV}, {Bc * stride * size};");
        string qFirst = b.R();
        b.L($"mul.lo.u32 {qFirst}, {qb}, {Br};");
        P.LoadTile(sQ, Br, qHead, qFirst, lq, q);
        P.LoadTile(sDO, Br, qHead, qFirst, lq, dO);
        b.L("bar.sync 0;");

        string qRow = b.R(), doRow = b.R(), pRow = b.R(), runK = b.R(), runV = b.R(), runDO = b.R();
        b.L($"mad.lo.u32 {qRow}, {r}, {stride * size}, {sQ};");
        b.L($"mad.lo.u32 {doRow}, {r}, {stride * size}, {sDO};");
        b.L($"mad.lo.u32 {pRow}, {r}, {(Bc + 1) * size}, {sP};");
        b.L($"mad.lo.u32 {runK}, {sub}, {vec * size}, {sK};");
        b.L($"mad.lo.u32 {runV}, {sub}, {vec * size}, {sV};");
        b.L($"mad.lo.u32 {runDO}, {sub}, {vec * size}, {doRow};");
        var c = new string[spt];
        var kRow = new string[spt];
        var vRow = new string[spt];
        for (int jj = 0; jj < spt; jj++)
        {
            c[jj] = b.R(); kRow[jj] = b.R(); vRow[jj] = b.R();
            b.L($"add.u32 {c[jj]}, {sub}, {tpr * jj};");
            b.L($"mad.lo.u32 {kRow[jj]}, {c[jj]}, {stride * size}, {sK};");
            b.L($"mad.lo.u32 {vRow[jj]}, {c[jj]}, {stride * size}, {sV};");
        }

        // The tile's scores with the scale and the mask (masked: −∞), valid[jj] the keys this row sees.
        (string[] S, string[] Valid) Scores(string kt)
        {
            var s = new string[spt];
            for (int jj = 0; jj < spt; jj++) s[jj] = lane.NewReg(b);
            P.Dots(s, qRow, kRow);
            var valid = new string[spt];
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
            }

            return (s, valid);
        }

        // Pass 1: the forward (online softmax, the context in this thread's runs).
        string m = lane.NewReg(b), l = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {m}, {negInf};");
        b.L($"mov.{lane.Reg} {l}, {lane.Zero};");
        var acc = P.Zeros(nvec);
        string kt = b.R(), pDone = b.P();
        b.L($"mov.u32 {kt}, {lo};");
        b.Label("F_TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {hi};");
        b.L($"@{pDone} bra F_END;");
        P.LoadTile(sK, Bc, kHead, kt, lk, k);
        P.LoadTile(sV, Bc, kHead, kt, lk, v);
        b.L("bar.sync 0;");
        {
            var (s, valid) = Scores(kt);
            string tmax = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {tmax}, {negInf};");
            for (int jj = 0; jj < spt; jj++) b.L($"max.{lane.Op} {tmax}, {tmax}, {s[jj]};");
            P.GroupReduce(tmax, "max", tpr);
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

            P.GroupReduce(psum, "add", tpr);
            b.L(lane.Fma(l, l, alpha, psum));
            b.L($"mov.{lane.Reg} {m}, {mNew};");
            b.L("bar.sync 0;");
            for (int j = 0; j < nvec; j++) for (int e = 0; e < vec; e++) b.L(lane.Mul(acc[j, e], acc[j, e], alpha));
            P.Accumulate(acc, Bc, pRow, 1, runV, vec * tpr);
            b.L("bar.sync 0;");
        }

        b.L($"add.u32 {kt}, {kt}, {Bc};");
        b.L("bra F_TILE;");
        b.Label("F_END");

        // The row's m (0 while no key: P is 0 there), 1/l (0 when none) and D = Σ_dims dO · O.
        string inv = lane.NewReg(b), pAny = b.P(), mUseF = lane.NewReg(b), pInfF = b.P(), dsum = lane.NewReg(b);
        b.L($"setp.gt.{lane.Op} {pAny}, {l}, {lane.Zero};");
        b.L($"div.rn.{lane.Op} {inv}, {lane.Literal(1)}, {l};");
        b.L($"selp.{lane.Op} {inv}, {inv}, {lane.Zero}, {pAny};");
        b.L($"setp.eq.{lane.Op} {pInfF}, {m}, {negInf};");
        b.L($"selp.{lane.Op} {mUseF}, {lane.Zero}, {m}, {pInfF};");
        b.L($"mov.{lane.Reg} {dsum}, {lane.Zero};");
        for (int j = 0; j < nvec; j++)
        {
            var dv = P.Vector();
            b.L($"ld.shared.v{vec}.{lane.Mem} {{{string.Join(", ", dv)}}}, [{runDO}+{vec * tpr * j * size}];");
            for (int e = 0; e < vec; e++)
            {
                string ov = lane.NewReg(b);
                b.L(lane.Mul(ov, acc[j, e], inv));
                b.L(lane.Fma(dsum, dv[e], ov, dsum));
            }
        }

        P.GroupReduce(dsum, "add", tpr);
        {
            string pW = b.P(), sOff = b.RD(), sAddr = b.RD(), skip = "STATS_DONE";
            b.L($"setp.eq.and.u32 {pW}, {sub}, 0, {rowValid};");
            b.L($"@!{pW} bra {skip};");
            b.L($"mul.wide.u32 {sOff}, {i}, {3 * size};");
            b.L($"cvt.u64.u32 {sAddr}, {h};");
            b.L($"mul.lo.u64 {sAddr}, {sAddr}, {3 * size};");
            string lq64 = b.RD();
            b.L($"cvt.u64.u32 {lq64}, {lq};");
            b.L($"mul.lo.u64 {sAddr}, {sAddr}, {lq64};");
            b.L($"add.u64 {sAddr}, {sAddr}, {sOff};");
            b.L($"add.u64 {sAddr}, {stats}, {sAddr};");
            b.L($"st.global.{lane.Mem} [{sAddr}], {mUseF};");
            b.L($"st.global.{lane.Mem} [{sAddr}+{size}], {inv};");
            b.L($"st.global.{lane.Mem} [{sAddr}+{2 * size}], {dsum};");
            b.Label(skip);
        }

        // Pass 2: dS = scale·P·(dP − D) per key, dQ += dS·K.
        var dqAcc = P.Zeros(nvec);
        b.L($"mov.u32 {kt}, {lo};");
        b.Label("B_TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {hi};");
        b.L($"@{pDone} bra B_END;");
        P.LoadTile(sK, Bc, kHead, kt, lk, k);
        P.LoadTile(sV, Bc, kHead, kt, lk, v);
        b.L("bar.sync 0;");
        {
            var (s, valid) = Scores(kt);
            var dp = new string[spt];
            for (int jj = 0; jj < spt; jj++) dp[jj] = lane.NewReg(b);
            P.Dots(dp, doRow, vRow);
            for (int jj = 0; jj < spt; jj++)
            {
                string ds = lane.NewReg(b), p = lane.NewReg(b), w = lane.NewReg(b), pAddr = b.R();
                b.L($"sub.rn.{lane.Op} {ds}, {s[jj]}, {mUseF};");
                string e = Codec.Exp(b, lane, ds);
                b.L(lane.Mul(p, e, inv));
                b.L($"selp.{lane.Op} {p}, {p}, {lane.Zero}, {valid[jj]};");
                b.L($"sub.rn.{lane.Op} {w}, {dp[jj]}, {dsum};");
                b.L(lane.Mul(w, w, p));
                b.L(lane.Mul(w, w, scale));
                b.L($"mad.lo.u32 {pAddr}, {c[jj]}, {size}, {pRow};");
                b.L($"st.shared.{lane.Mem} [{pAddr}], {w};");
            }

            b.L("bar.sync 0;");
            P.Accumulate(dqAcc, Bc, pRow, 1, runK, vec * tpr);
            b.L("bar.sync 0;");
        }

        b.L($"add.u32 {kt}, {kt}, {Bc};");
        b.L("bra B_TILE;");
        b.Label("B_END");
        string oRow = b.R();
        b.L($"mad.lo.u32 {oRow}, {i}, {d}, {qHead};");
        P.StoreRuns(dqAcc, dq, oRow, sub, tpr, rowValid, "DQ");
        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_do, .param .u64 p_dq, .param .u64 p_stats, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .{pa} p_scale",
            AttentionBackRowsShared(lane, dCap));
    }

    public static string AttentionBackKeys(Gpu gpu, Lane lane, int dCap, out string entry)
    {
        var (Br, Bc) = AttentionTile(lane);
        int T = RowsThreads, tpk = T / Bc, spt = Br / tpk, size = lane.Size;
        entry = $"attention_back_keys_{lane.Name}_d{dCap}";
        var b = new PtxBuilder();
        string tid = b.R(), kb = b.R(), h = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {kb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), dO = Param64(b, "p_do"), stats = Param64(b, "p_stats"), dk = Param64(b, "p_dk"), dv = Param64(b, "p_dv");
        string lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {scale}, [p_scale];");
        var P = new BackPieces(b, lane, dCap, tid, d);
        int vec = P.Vec, stride = P.Stride, nvec = dCap / (vec * tpk);

        // This thread's key and place in it.
        string ck = b.R(), sub = b.R(), key = b.R(), keyValid = b.P();
        b.L($"shr.u32 {ck}, {tid}, {System.Numerics.BitOperations.Log2((uint)tpk)};");
        b.L($"and.b32 {sub}, {tid}, {tpk - 1};");
        b.L($"mad.lo.u32 {key}, {kb}, {Bc}, {ck};");
        b.L($"setp.lt.u32 {keyValid}, {key}, {lk};");
        string qHead = b.R(), kHead = b.R(), lo = b.R(), hi = b.R(), ra = b.RD(), statHead = b.R();
        b.L($"mul.lo.u32 {qHead}, {h}, {lq};");
        b.L($"mul.lo.u32 {statHead}, {qHead}, 3;");
        b.L($"mul.lo.u32 {qHead}, {qHead}, {d};");
        b.L($"mul.lo.u32 {kHead}, {h}, {lk};");
        b.L($"mul.lo.u32 {kHead}, {kHead}, {d};");
        b.L($"mul.wide.u32 {ra}, {kb}, 8;");
        b.L($"add.u64 {ra}, {ranges}, {ra};");
        b.L($"ld.global.u32 {lo}, [{ra}];");
        b.L($"ld.global.u32 {hi}, [{ra}+4];");
        // Shared: K, V of the keys; Q, dO of the tile; P and dS [key][row]; the rows' m, 1/l, D; their limits and starts.
        string sK = b.R(), sV = b.R(), sQ = b.R(), sDO = b.R(), sP = b.R(), sS = b.R(), sStat = b.R(), sLim = b.R();
        b.L($"mov.u32 {sK}, smem;");
        b.L($"add.u32 {sV}, {sK}, {Bc * stride * size};");
        b.L($"add.u32 {sQ}, {sV}, {Bc * stride * size};");
        b.L($"add.u32 {sDO}, {sQ}, {Br * stride * size};");
        b.L($"add.u32 {sP}, {sDO}, {Br * stride * size};");
        b.L($"add.u32 {sS}, {sP}, {Bc * (Br + 1) * size};");
        b.L($"add.u32 {sStat}, {sS}, {Bc * (Br + 1) * size};");
        b.L($"add.u32 {sLim}, {sStat}, {3 * Br * size};");
        string kFirst = b.R();
        b.L($"mul.lo.u32 {kFirst}, {kb}, {Bc};");
        P.LoadTile(sK, Bc, kHead, kFirst, lk, k);
        P.LoadTile(sV, Bc, kHead, kFirst, lk, v);

        string kRow = b.R(), vRow = b.R(), pRow = b.R(), sRow = b.R(), runQ = b.R(), runDO = b.R();
        b.L($"mad.lo.u32 {kRow}, {ck}, {stride * size}, {sK};");
        b.L($"mad.lo.u32 {vRow}, {ck}, {stride * size}, {sV};");
        b.L($"mad.lo.u32 {pRow}, {ck}, {(Br + 1) * size}, {sP};");
        b.L($"mad.lo.u32 {sRow}, {ck}, {(Br + 1) * size}, {sS};");
        b.L($"mad.lo.u32 {runQ}, {sub}, {vec * size}, {sQ};");
        b.L($"mad.lo.u32 {runDO}, {sub}, {vec * size}, {sDO};");
        var rr = new string[spt];
        var qRows = new string[spt];
        var doRows = new string[spt];
        for (int jj = 0; jj < spt; jj++)
        {
            rr[jj] = b.R(); qRows[jj] = b.R(); doRows[jj] = b.R();
            b.L($"add.u32 {rr[jj]}, {sub}, {tpk * jj};");
            b.L($"mad.lo.u32 {qRows[jj]}, {rr[jj]}, {stride * size}, {sQ};");
            b.L($"mad.lo.u32 {doRows[jj]}, {rr[jj]}, {stride * size}, {sDO};");
        }

        var dkAcc = P.Zeros(nvec);
        var dvAcc = P.Zeros(nvec);
        string qt = b.R(), pDone = b.P();
        b.L($"mov.u32 {qt}, {lo};");
        b.Label("Q_TILE");
        b.L($"setp.ge.u32 {pDone}, {qt}, {hi};");
        b.L($"@{pDone} bra Q_END;");
        P.LoadTile(sQ, Br, qHead, qt, lq, q);
        P.LoadTile(sDO, Br, qHead, qt, lq, dO);
        {
            // The tile's rows' m, 1/l, D (3 lanes each) and keys [start, limit) (0, 0 past the queries): threads 0 .. Br − 1.
            string pT = b.P(), row = b.R(), pRowV = b.P(), sAddr = b.RD(), idx = b.R(), lAddr = b.RD(), lv = b.R(), sv = b.R(), dst = b.R(), skip = "STAT_LOADED";
            b.L($"setp.ge.u32 {pT}, {tid}, {Br};");
            b.L($"@{pT} bra {skip};");
            b.L($"add.u32 {row}, {qt}, {tid};");
            b.L($"setp.lt.u32 {pRowV}, {row}, {lq};");
            b.L($"selp.b32 {idx}, {row}, 0, {pRowV};");
            var st = new string[3];
            b.L($"mad.lo.u32 {dst}, {idx}, 3, {statHead};");                // the head's row idx (row 0 past the queries)
            b.L($"mul.wide.u32 {sAddr}, {dst}, {size};");
            b.L($"add.u64 {sAddr}, {stats}, {sAddr};");
            string sw = b.R();
            b.L($"mad.lo.u32 {sw}, {tid}, {3 * size}, {sStat};");
            for (int t = 0; t < 3; t++)
            {
                st[t] = lane.NewReg(b);
                b.L($"ld.global.{lane.Mem} {st[t]}, [{sAddr}+{t * size}];");
                b.L($"st.shared.{lane.Mem} [{sw}+{t * size}], {st[t]};");
            }

            b.L($"mul.wide.u32 {lAddr}, {idx}, 4;");
            string la2 = b.RD(), sa2 = b.RD();
            b.L($"add.u64 {la2}, {lim}, {lAddr};");
            b.L($"add.u64 {sa2}, {starts}, {lAddr};");
            b.L($"ld.global.u32 {lv}, [{la2}];");
            b.L($"ld.global.u32 {sv}, [{sa2}];");
            b.L($"min.u32 {lv}, {lv}, {lk};");
            b.L($"selp.b32 {lv}, {lv}, 0, {pRowV};");
            string lw = b.R();
            b.L($"mad.lo.u32 {lw}, {tid}, 8, {sLim};");
            b.L($"st.shared.u32 [{lw}], {lv};");
            b.L($"st.shared.u32 [{lw}+4], {sv};");
            b.Label(skip);
        }

        b.L("bar.sync 0;");
        {
            // P and dS of this key for the rows this thread takes.
            var s = new string[spt];
            var dp = new string[spt];
            for (int jj = 0; jj < spt; jj++) { s[jj] = lane.NewReg(b); dp[jj] = lane.NewReg(b); }
            // Σ over dims of row · key: the key's row is the shared one, the rows step.
            for (int jj = 0; jj < spt; jj++) { b.L($"mov.{lane.Reg} {s[jj]}, {lane.Zero};"); b.L($"mov.{lane.Reg} {dp[jj]}, {lane.Zero};"); }
            for (int t = 0; t < dCap; t += vec)
            {
                var kv = P.Vector();
                var vv = P.Vector();
                b.L($"ld.shared.v{vec}.{lane.Mem} {{{string.Join(", ", kv)}}}, [{kRow}+{t * size}];");
                b.L($"ld.shared.v{vec}.{lane.Mem} {{{string.Join(", ", vv)}}}, [{vRow}+{t * size}];");
                for (int jj = 0; jj < spt; jj++)
                {
                    var qv = P.Vector();
                    var ov = P.Vector();
                    b.L($"ld.shared.v{vec}.{lane.Mem} {{{string.Join(", ", qv)}}}, [{qRows[jj]}+{t * size}];");
                    b.L($"ld.shared.v{vec}.{lane.Mem} {{{string.Join(", ", ov)}}}, [{doRows[jj]}+{t * size}];");
                    for (int e = 0; e < vec; e++)
                    {
                        b.L(lane.Fma(s[jj], qv[e], kv[e], s[jj]));
                        b.L(lane.Fma(dp[jj], ov[e], vv[e], dp[jj]));
                    }
                }
            }

            for (int jj = 0; jj < spt; jj++)
            {
                string lw = b.R(), lv = b.R(), sv = b.R(), pv = b.P(), p1 = b.P(), sw = b.R(), mm = lane.NewReg(b), il = lane.NewReg(b), dd = lane.NewReg(b);
                string x = lane.NewReg(b), p = lane.NewReg(b), w = lane.NewReg(b), pAddr = b.R(), sAddr = b.R();
                b.L($"mad.lo.u32 {lw}, {rr[jj]}, 8, {sLim};");
                b.L($"ld.shared.u32 {lv}, [{lw}];");
                b.L($"ld.shared.u32 {sv}, [{lw}+4];");
                b.L($"setp.ge.u32 {pv}, {key}, {sv};");
                b.L($"setp.lt.u32 {p1}, {key}, {lv};");
                b.L($"and.pred {pv}, {pv}, {p1};");
                b.L($"mad.lo.u32 {sw}, {rr[jj]}, {3 * size}, {sStat};");
                b.L($"ld.shared.{lane.Mem} {mm}, [{sw}];");
                b.L($"ld.shared.{lane.Mem} {il}, [{sw}+{size}];");
                b.L($"ld.shared.{lane.Mem} {dd}, [{sw}+{2 * size}];");
                b.L(lane.Mul(x, s[jj], scale));
                b.L($"sub.rn.{lane.Op} {x}, {x}, {mm};");
                b.L($"selp.{lane.Op} {x}, {x}, {lane.Zero}, {pv};");             // no exponent of a masked −∞ or garbage
                string e = Codec.Exp(b, lane, x);
                b.L(lane.Mul(p, e, il));
                b.L($"selp.{lane.Op} {p}, {p}, {lane.Zero}, {pv};");
                b.L($"sub.rn.{lane.Op} {w}, {dp[jj]}, {dd};");
                b.L(lane.Mul(w, w, p));
                b.L(lane.Mul(w, w, scale));
                b.L($"mad.lo.u32 {pAddr}, {rr[jj]}, {size}, {pRow};");
                b.L($"mad.lo.u32 {sAddr}, {rr[jj]}, {size}, {sRow};");
                b.L($"st.shared.{lane.Mem} [{pAddr}], {p};");
                b.L($"st.shared.{lane.Mem} [{sAddr}], {w};");
            }

            b.L("bar.sync 0;");
            P.Accumulate(dvAcc, Br, pRow, 1, runDO, vec * tpk);
            P.Accumulate(dkAcc, Br, sRow, 1, runQ, vec * tpk);
            b.L("bar.sync 0;");
        }

        b.L($"add.u32 {qt}, {qt}, {Br};");
        b.L("bra Q_TILE;");
        b.Label("Q_END");
        string kOut = b.R();
        b.L($"mad.lo.u32 {kOut}, {key}, {d}, {kHead};");
        P.StoreRuns(dkAcc, dk, kOut, sub, tpk, keyValid, "DK");
        P.StoreRuns(dvAcc, dv, kOut, sub, tpk, keyValid, "DV");
        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_do, .param .u64 p_stats, .param .u64 p_dk, .param .u64 p_dv, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .{pa} p_scale",
            AttentionBackKeysShared(lane, dCap));
    }
}
