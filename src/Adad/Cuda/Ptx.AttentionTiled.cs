// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The fused attention rows of a float32 type by register tiles (flash, in the f32 lane): a block per (head, 64 query rows),
// 256 threads as 16 × 16. The scores of a tile of 64 keys: each thread 4 rows (ty·4 …) by 4 keys (tx, tx + 16, …), the
// rows' queries and the keys' rows read from shared memory 16 bytes at a time along d (8 loads for 64 multiply-adds; the
// row kernel's thread scored a row against keys one dot at a time, 5 loads for 16). The context the same way: each thread
// 4 rows by d/16 dimensions, a key's 4 weights in one load and its values in d/64 more. Q, the key tile and the value
// tile as rows of d + 4 values (the 16 keys a load touches in different banks); the weights by row group ([ty][key][4]).
// The next tile's keys and this tile's values are read into registers while the scores and the context are computed,
// stored to shared memory after: the loads' latency under the arithmetic. The online softmax and its order are the row
// kernel's (the tile's max, alpha = e^(m − m'), p = e^(s − m'), l = l·alpha + Σp), over tiles of 64 keys.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Query rows and keys a tile of the register-tiled rows holds.</summary>
    public const int TiledBr = 64, TiledBc = 64;

    /// <summary>Shared bytes of the register-tiled rows: Q and the key/value tile as rows of dCap + 4 lanes, the weights 64 × 64.</summary>
    public static int AttentionTiledShared(int dCap) => (2 * TiledBr * (dCap + 4) + TiledBr * TiledBc) * 4;

    /// <summary>The register-tiled fused rows (float32 Q, K, V and O; d a multiple of 4, dCap 32, 64 or 128): params as
    /// AttentionRows'. Ranges (p_range) by blocks of TiledBr rows.</summary>
    public static string AttentionRowsTiled(Gpu gpu, int dCap, out string entry)
    {
        if (dCap is not (32 or 64 or 128)) throw new ArgumentException($"tiled rows: a head capacity of {dCap}");
        entry = $"attention_rows_tiled_f32_d{dCap}";
        const int T = 256, Br = TiledBr, Bc = TiledBc, R = 4, C = 4;
        int stride = dCap + 4, rowBytes = stride * 4, cpt = dCap / 16, chunksPerRow = dCap / 4, chunksPerThread = Br * chunksPerRow / T;
        var lane = Lane.F32;
        var b = new PtxBuilder(gpu);
        string tid = b.R(), qb = b.R(), h = b.R(), ty = b.R(), tx = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {qb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        b.L($"shr.u32 {ty}, {tid}, 4;");
        b.L($"and.b32 {tx}, {tid}, 15;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), o = Param64(b, "p_o"), lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = b.F();
        b.L($"ld.param.f32 {scale}, [p_scale];");
        // The thread's 4 rows: their query index, validity, and key range [start, limit).
        string row0 = b.R(), lqm1 = b.R();
        b.L($"mul.lo.u32 {row0}, {qb}, {Br};");
        b.L($"sub.u32 {lqm1}, {lq}, 1;");
        var rowI = new string[R];
        var rowOk = new string[R];
        var lo = new string[R];
        var hi = new string[R];
        for (int i = 0; i < R; i++)
        {
            rowI[i] = b.R(); rowOk[i] = b.P(); lo[i] = b.R(); hi[i] = b.R();
            string clamped = b.R(), a = b.RD();
            b.L($"mad.lo.u32 {rowI[i]}, {ty}, {R}, {i};");
            b.L($"add.u32 {rowI[i]}, {rowI[i]}, {row0};");
            b.L($"setp.lt.u32 {rowOk[i]}, {rowI[i]}, {lq};");
            b.L($"min.u32 {clamped}, {rowI[i]}, {lqm1};");
            b.L($"mul.wide.u32 {a}, {clamped}, 4;");
            string la = b.RD(), sa = b.RD();
            b.L($"add.u64 {la}, {lim}, {a};");
            b.L($"add.u64 {sa}, {starts}, {a};");
            b.L($"ld.global.u32 {hi[i]}, [{la}];");
            b.L($"ld.global.u32 {lo[i]}, [{sa}];");
            b.L($"min.u32 {hi[i]}, {hi[i]}, {lk};");
            b.L($"selp.b32 {hi[i]}, {hi[i]}, 0, {rowOk[i]};");
        }

        string qHead = b.R(), kHead = b.R(), tileLo = b.R(), tileHi = b.R(), ra = b.RD();
        b.L($"mul.lo.u32 {qHead}, {h}, {lq};");
        b.L($"mul.lo.u32 {qHead}, {qHead}, {d};");
        b.L($"mul.lo.u32 {kHead}, {h}, {lk};");
        b.L($"mul.lo.u32 {kHead}, {kHead}, {d};");
        b.L($"mul.wide.u32 {ra}, {qb}, 8;");
        b.L($"add.u64 {ra}, {ranges}, {ra};");
        b.L($"ld.global.u32 {tileLo}, [{ra}];");
        b.L($"ld.global.u32 {tileHi}, [{ra}+4];");
        string sQ = b.R(), sKV = b.R(), sP = b.R();
        b.L($"mov.u32 {sQ}, smem;");
        b.L($"add.u32 {sKV}, {sQ}, {Br * rowBytes};");
        b.L($"add.u32 {sP}, {sKV}, {Bc * rowBytes};");

        // A tile of 64 rows (from `first`, rows past `limit` and dimensions past d as 0) into registers: chunk tid + 256·it
        // → row chunk / (dCap/4), 16 bytes at chunk mod (dCap/4); then into shared rows of dCap + 4 lanes.
        string cRow = b.R(), cPart = b.R(), cDim = b.R(), cOk = b.P(), cSlot = b.R();
        b.L($"shr.u32 {cRow}, {tid}, {System.Numerics.BitOperations.Log2((uint)chunksPerRow)};");
        b.L($"and.b32 {cPart}, {tid}, {chunksPerRow - 1};");
        b.L($"shl.b32 {cDim}, {cPart}, 2;");
        b.L($"setp.lt.u32 {cOk}, {cDim}, {d};");
        b.L($"mad.lo.u32 {cSlot}, {cRow}, {rowBytes}, 0;");
        b.L($"mad.lo.u32 {cSlot}, {cPart}, 16, {cSlot};");
        int rowsPerIt = T / chunksPerRow;
        string[] LoadRegs(string src, string head, string first, string limit)
        {
            var regs = new string[chunksPerThread * 4];
            for (int it = 0; it < chunksPerThread; it++)
            {
                string gr = b.R(), pv = b.P(), idx = b.R(), addr = b.RD();
                for (int e = 0; e < 4; e++) { regs[it * 4 + e] = b.F(); b.L($"mov.f32 {regs[it * 4 + e]}, 0f00000000;"); }
                b.L($"add.u32 {gr}, {first}, {cRow};");
                b.L($"add.u32 {gr}, {gr}, {it * rowsPerIt};");
                b.L($"setp.lt.u32 {pv}, {gr}, {limit};");
                b.L($"and.pred {pv}, {pv}, {cOk};");
                b.L($"mad.lo.u32 {idx}, {gr}, {d}, {cDim};");
                b.L($"add.u32 {idx}, {idx}, {head};");
                b.L($"mul.wide.u32 {addr}, {idx}, 4;");
                b.L($"add.u64 {addr}, {src}, {addr};");
                b.L($"@{pv} ld.global.v4.f32 {{{regs[it * 4]}, {regs[it * 4 + 1]}, {regs[it * 4 + 2]}, {regs[it * 4 + 3]}}}, [{addr}];");
            }

            return regs;
        }

        void StoreRegs(string[] regs, string dst)
        {
            string at = b.R();
            b.L($"add.u32 {at}, {dst}, {cSlot};");
            for (int it = 0; it < chunksPerThread; it++)
                b.L($"st.shared.v4.f32 [{at}+{it * rowsPerIt * rowBytes}], {{{regs[it * 4]}, {regs[it * 4 + 1]}, {regs[it * 4 + 2]}, {regs[it * 4 + 3]}}};");
        }

        // Q once; the first key tile.
        StoreRegs(LoadRegs(q, qHead, row0, lq), sQ);
        string kt = b.R(), pNone = b.P();
        b.L($"mov.u32 {kt}, {tileLo};");
        b.L($"setp.ge.u32 {pNone}, {kt}, {tileHi};");
        b.L($"@{pNone} bra NO_KEYS;");
        StoreRegs(LoadRegs(k, kHead, kt, lk), sKV);
        b.Label("NO_KEYS");
        b.L("bar.sync 0;");

        var m = new string[R];
        var l = new string[R];
        var acc = new string[R, cpt];
        string negInf = lane.Literal(double.NegativeInfinity);
        for (int i = 0; i < R; i++)
        {
            m[i] = b.F(); l[i] = b.F();
            b.L($"mov.f32 {m[i]}, {negInf};");
            b.L($"mov.f32 {l[i]}, 0f00000000;");
            for (int e = 0; e < cpt; e++) { acc[i, e] = b.F(); b.L($"mov.f32 {acc[i, e]}, 0f00000000;"); }
        }

        string qAddr = b.R(), kAddr = b.R(), pW = b.R(), pR = b.R(), vAddr = b.R();
        b.L($"mad.lo.u32 {qAddr}, {ty}, {R * rowBytes}, {sQ};");
        b.L($"mad.lo.u32 {kAddr}, {tx}, {rowBytes}, {sKV};");
        b.L($"mad.lo.u32 {pW}, {ty}, {Bc * R * 4}, {sP};");
        b.L($"mad.lo.u32 {pW}, {tx}, {R * 4}, {pW};");
        b.L($"mad.lo.u32 {pR}, {ty}, {Bc * R * 4}, {sP};");
        b.L($"mad.lo.u32 {vAddr}, {tx}, {cpt * 4}, {sKV};");
        string pDone = b.P();
        b.Label("TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {tileHi};");
        b.L($"@{pDone} bra TILE_END;");
        // This tile's values on their way while the scores are computed.
        var vRegs = LoadRegs(v, kHead, kt, lk);
        // The scores: s[i][j] = q_(ty·4+i) · k_(tx+16j) over d (zeros past d), 16 bytes a load.
        var s = new string[R, C];
        for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) { s[i, j] = b.F(); b.L($"mov.f32 {s[i, j]}, 0f00000000;"); }
        for (int t4 = 0; t4 < chunksPerRow; t4++)
        {
            var qv = new string[R][];
            var kv = new string[C][];
            for (int i = 0; i < R; i++)
            {
                qv[i] = [b.F(), b.F(), b.F(), b.F()];
                b.L($"ld.shared.v4.f32 {{{string.Join(", ", qv[i])}}}, [{qAddr}+{i * rowBytes + t4 * 16}];");
            }

            for (int j = 0; j < C; j++)
            {
                kv[j] = [b.F(), b.F(), b.F(), b.F()];
                b.L($"ld.shared.v4.f32 {{{string.Join(", ", kv[j])}}}, [{kAddr}+{j * 16 * rowBytes + t4 * 16}];");
            }

            for (int e = 0; e < 4; e++)
                for (int i = 0; i < R; i++)
                    for (int j = 0; j < C; j++) b.L($"fma.rn.f32 {s[i, j]}, {qv[i][e]}, {kv[j][e]}, {s[i, j]};");
        }

        // The mask, each row's tile max over its 16 threads, the new running max (0 in the exponent while no key is seen),
        // the weights into shared memory by row group, the row's sum.
        var alpha = new string[R];
        for (int i = 0; i < R; i++)
        {
            var valid = new string[C];
            string tmax = b.F();
            b.L($"mov.f32 {tmax}, {negInf};");
            for (int j = 0; j < C; j++)
            {
                string key = b.R(), p1 = b.P();
                valid[j] = b.P();
                b.L($"mul.rn.f32 {s[i, j]}, {s[i, j]}, {scale};");
                b.L($"add.u32 {key}, {kt}, {tx};");
                b.L($"add.u32 {key}, {key}, {16 * j};");
                b.L($"setp.ge.u32 {valid[j]}, {key}, {lo[i]};");
                b.L($"setp.lt.u32 {p1}, {key}, {hi[i]};");
                b.L($"and.pred {valid[j]}, {valid[j]}, {p1};");
                b.L($"selp.f32 {s[i, j]}, {s[i, j]}, {negInf}, {valid[j]};");
                b.L($"max.f32 {tmax}, {tmax}, {s[i, j]};");
            }

            Group16(b, tmax, "max");
            string mNew = b.F(), mUse = b.F(), pInf = b.P(), dm = b.F();
            b.L($"max.f32 {mNew}, {m[i]}, {tmax};");
            b.L($"setp.eq.f32 {pInf}, {mNew}, {negInf};");
            b.L($"selp.f32 {mUse}, 0f00000000, {mNew}, {pInf};");
            b.L($"sub.rn.f32 {dm}, {m[i]}, {mUse};");
            alpha[i] = Codec.Exp(b, lane, dm);
            string psum = b.F();
            b.L($"mov.f32 {psum}, 0f00000000;");
            for (int j = 0; j < C; j++)
            {
                string ds = b.F(), p = b.F();
                b.L($"sub.rn.f32 {ds}, {s[i, j]}, {mUse};");
                string e = Codec.Exp(b, lane, ds);
                b.L($"selp.f32 {p}, {e}, 0f00000000, {valid[j]};");
                b.L($"st.shared.f32 [{pW}+{(j * 16 * R + i) * 4}], {p};");
                b.L($"add.rn.f32 {psum}, {psum}, {p};");
            }

            Group16(b, psum, "add");
            b.L($"fma.rn.f32 {l[i]}, {l[i]}, {alpha[i]}, {psum};");
            b.L($"mov.f32 {m[i]}, {mNew};");
        }

        b.L("bar.sync 0;");
        StoreRegs(vRegs, sKV);
        // The next tile's keys on their way while the context is computed.
        string ktNext = b.R(), pLast = b.P();
        b.L($"add.u32 {ktNext}, {kt}, {Bc};");
        b.L($"setp.ge.u32 {pLast}, {ktNext}, {tileHi};");
        var kRegs = LoadRegs(k, kHead, ktNext, lk);
        b.L("bar.sync 0;");
        // The context: acc = alpha·acc + Σ_key p_(row, key) · v_key (this thread's 4 rows by cpt dimensions).
        for (int i = 0; i < R; i++) for (int e = 0; e < cpt; e++) b.L($"mul.rn.f32 {acc[i, e]}, {acc[i, e]}, {alpha[i]};");
        for (int key = 0; key < Bc; key++)
        {
            string[] pv = [b.F(), b.F(), b.F(), b.F()];
            b.L($"ld.shared.v4.f32 {{{string.Join(", ", pv)}}}, [{pR}+{key * R * 4}];");
            for (int c4 = 0; c4 < cpt; c4 += 4)
            {
                string[] vv = [b.F(), b.F(), b.F(), b.F()];
                if (cpt >= 4) b.L($"ld.shared.v4.f32 {{{string.Join(", ", vv)}}}, [{vAddr}+{key * rowBytes + c4 * 4}];");
                for (int e = 0; e < Math.Min(4, cpt); e++)
                {
                    if (cpt < 4) b.L($"ld.shared.f32 {vv[e]}, [{vAddr}+{key * rowBytes + (c4 + e) * 4}];");
                    for (int i = 0; i < R; i++) b.L($"fma.rn.f32 {acc[i, c4 + e]}, {pv[i]}, {vv[e]}, {acc[i, c4 + e]};");
                }
            }
        }

        b.L("bar.sync 0;");
        b.L($"@{pLast} bra NEXT;");
        StoreRegs(kRegs, sKV);
        b.Label("NEXT");
        b.L("bar.sync 0;");
        b.L($"mov.u32 {kt}, {ktNext};");
        b.L("bra TILE;");
        b.Label("TILE_END");

        // O = acc / l (0 where no key was seen), 16 bytes a store; rows past lq and dimensions past d not stored.
        for (int i = 0; i < R; i++)
        {
            string inv = b.F(), pAny = b.P(), oRow = b.R();
            b.L($"setp.gt.f32 {pAny}, {l[i]}, 0f00000000;");
            b.L($"div.rn.f32 {inv}, 0f3F800000, {l[i]};");
            b.L($"selp.f32 {inv}, {inv}, 0f00000000, {pAny};");
            b.L($"mad.lo.u32 {oRow}, {rowI[i]}, {d}, {qHead};");
            for (int c4 = 0; c4 < cpt; c4 += Math.Min(4, cpt))
            {
                int w = Math.Min(4, cpt);
                string dim = b.R(), pOut = b.P(), idx = b.R(), oa = b.RD();
                var vals = new string[w];
                b.L($"mad.lo.u32 {dim}, {tx}, {cpt}, {c4};");
                b.L($"setp.lt.u32 {pOut}, {dim}, {d};");
                b.L($"and.pred {pOut}, {pOut}, {rowOk[i]};");
                for (int e = 0; e < w; e++) { vals[e] = b.F(); b.L($"mul.rn.f32 {vals[e]}, {acc[i, c4 + e]}, {inv};"); }
                b.L($"add.u32 {idx}, {oRow}, {dim};");
                b.L($"mul.wide.u32 {oa}, {idx}, 4;");
                b.L($"add.u64 {oa}, {o}, {oa};");
                if (w == 4) b.L($"@{pOut} st.global.v4.f32 [{oa}], {{{string.Join(", ", vals)}}};");
                else b.L($"@{pOut} st.global.v2.f32 [{oa}], {{{string.Join(", ", vals)}}};");
            }
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_o, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .f32 p_scale",
            AttentionTiledShared(dCap));
    }

    // A row's reduction across its 16 threads (the half-warp of one ty): butterflies over offsets 8, 4, 2, 1.
    private static void Group16(PtxBuilder b, string acc, string op)
    {
        for (int off = 8; off >= 1; off /= 2)
        {
            string bits = b.R(), bits2 = b.R(), other = b.F();
            b.L($"mov.b32 {bits}, {acc};");
            b.L($"shfl.sync.bfly.b32 {bits2}, {bits}, {off}, 0x1f, 0xffffffff;");
            b.L($"mov.b32 {other}, {bits2};");
            b.L(op == "max" ? $"max.f32 {acc}, {acc}, {other};" : $"add.rn.f32 {acc}, {acc}, {other};");
        }
    }
}
