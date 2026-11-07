// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's fused attention backward on the tensor instructions (item 6j): Q, K, V and dO in the f16 or bf16 kind
// (exact, as the tensor rows: float16, bfloat16 and the 8-bit floats), every product accumulated in f32, no scores in
// memory, no atomics (each output value written by one thread). Two kernels, as the FMA backward (Ptx.AttentionBackward):
//
//   rows (dQ)    4 warps per (head, 64 query rows), a warp 16 rows; Q's and dO's fragments held in registers. Pass 1 the
//                tensor rows' forward (S = Q·Kᵀ, the online max and sum, O += P·V in two slices) → the row's max m, 1/l
//                and D = dO · O, stored per row (f32). Pass 2 over the same key tiles, 32 keys at a time: S = Q·Kᵀ,
//                dP = dO·Vᵀ, P = e^(s − m)/l, dS = scale·P·(dP − D), dQ += dS·K (dS from the accumulators as the A
//                operand in two slices of the kind, K's fragments by ldmatrix.trans).
//   keys (dK dV) 4 warps per (head, 64 keys), a warp 16 keys; the queries walked in tiles of 64 (only those that see the
//                block's keys), 32 at a time: Sᵀ = K·Qᵀ, dPᵀ = V·dOᵀ, Pᵀ and dSᵀ from the rows' m, 1/l, D, then
//                dV += Pᵀ·dO and dK += dSᵀ·Q (two slices each; dO's and Q's fragments by ldmatrix.trans).
//
// The two slices (hi = x rounded to the kind, lo = x − hi rounded) keep P and dS to 22 bits in f16 and 16 in bf16 as the
// tensor rows keep P. Loads as the tensor rows: 16-byte copies, 32-bit words of four 8-bit values, or value by value.
// dQ, dK, dV stored value by value as the type.
namespace Adad;

internal static partial class Ptx
{
    /// <summary>Shared bytes of the tensor backward's keys kernel: K and V of the keys, the Q and dO tiles, the tile's
    /// rows' m, 1/l, D and key ranges. (The rows kernel takes the tensor rows' AttentionMmaShared.)</summary>
    public static int AttentionBackKeysMmaShared(int dCap) => 4 * MmaRowsBc * (dCap * 2 + 16) + MmaRowsBr * (3 * 4 + 2 * 4);

    // The pieces both tensor backward kernels share: 64-row tile loads in the kind, fragment addresses, slices, stores.
    private sealed class MmaBackPieces(PtxBuilder b, Fmt f, Kind kind, int dCap, char load, string tid, string ln, string warp, string d)
    {
        public const int T = MmaRowsThreads;
        public readonly int Stride = dCap * 2 + 16;
        private readonly bool bf = kind == Kind.BF16;
        private string Pair => bf ? "bf16x2" : "f16x2";

        // 64 rows [first, first + 64) of a tensor (d values a row from value `head`; rows past rowLimit and dimensions
        // past d as 0) into shared at dst in the kind, rows Stride bytes apart (the tensor rows' loads).
        public void LoadTile(string dst, string head, string first, string rowLimit, string src)
        {
            var F = Lane.F32;
            int stride = Stride;
            if (load == 'c')
            {
                int parts = dCap / 8, runs = 64 * parts / T, partShift = System.Numerics.BitOperations.Log2((uint)parts);
                for (int j = 0; j < runs; j++)
                {
                    string c = b.R(), row = b.R(), part = b.R(), gr = b.R(), col = b.R(), pv = b.P(), pc = b.P(), idx = b.R(), size = b.R(), sa = b.R(), ga = b.RD();
                    b.L($"add.u32 {c}, {tid}, {T * j};");
                    b.L($"shr.u32 {row}, {c}, {partShift};");
                    b.L($"and.b32 {part}, {c}, {parts - 1};");
                    b.L($"add.u32 {gr}, {first}, {row};");
                    b.L($"shl.b32 {col}, {part}, 3;");
                    b.L($"setp.lt.u32 {pv}, {gr}, {rowLimit};");
                    b.L($"setp.lt.u32 {pc}, {col}, {d};");
                    b.L($"and.pred {pv}, {pv}, {pc};");
                    b.L($"mad.lo.u32 {idx}, {gr}, {d}, {col};");
                    b.L($"add.u32 {idx}, {idx}, {head};");
                    b.L($"selp.b32 {idx}, {idx}, {head}, {pv};");
                    b.L($"selp.b32 {size}, 16, 0, {pv};");
                    b.L($"mul.wide.u32 {ga}, {idx}, 2;");
                    b.L($"add.u64 {ga}, {src}, {ga};");
                    b.L($"mad.lo.u32 {sa}, {row}, {stride}, {dst};");
                    b.L($"mad.lo.u32 {sa}, {part}, 16, {sa};");
                    b.L($"cp.async.cg.shared.global [{sa}], [{ga}], 16, {size};");
                }

                return;
            }

            if (load == 'w')
            {
                int wordsRow = dCap / 4, words = 64 * wordsRow / T, wShift = System.Numerics.BitOperations.Log2((uint)wordsRow);
                for (int j = 0; j < words; j++)
                {
                    string c = b.R(), row = b.R(), col = b.R(), gr = b.R(), pv = b.P(), pc = b.P(), idx = b.R(), ga = b.RD(), w = b.R(), sa = b.R(), p01 = b.R(), p23 = b.R();
                    b.L($"add.u32 {c}, {tid}, {T * j};");
                    b.L($"shr.u32 {row}, {c}, {wShift};");
                    b.L($"and.b32 {col}, {c}, {wordsRow - 1};");
                    b.L($"shl.b32 {col}, {col}, 2;");
                    b.L($"add.u32 {gr}, {first}, {row};");
                    b.L($"setp.lt.u32 {pv}, {gr}, {rowLimit};");
                    b.L($"setp.lt.u32 {pc}, {col}, {d};");
                    b.L($"and.pred {pv}, {pv}, {pc};");
                    b.L($"mad.lo.u32 {idx}, {gr}, {d}, {col};");
                    b.L($"add.u32 {idx}, {idx}, {head};");
                    b.L($"selp.b32 {idx}, {idx}, {head}, {pv};");
                    b.L($"cvt.u64.u32 {ga}, {idx};");
                    b.L($"add.u64 {ga}, {src}, {ga};");
                    b.L($"ld.global.b32 {w}, [{ga}];");
                    b.L($"selp.b32 {w}, {w}, 0, {pv};");
                    var x = new string[4];
                    for (int t = 0; t < 4; t++) x[t] = Codec.Unpack(b, f, w, t, F);
                    b.L($"cvt.rn.{Pair}.f32 {p01}, {x[1]}, {x[0]};");
                    b.L($"cvt.rn.{Pair}.f32 {p23}, {x[3]}, {x[2]};");
                    b.L($"mad.lo.u32 {sa}, {row}, {stride}, {dst};");
                    b.L($"mad.lo.u32 {sa}, {col}, 2, {sa};");
                    b.L($"st.shared.v2.b32 [{sa}], {{{p01}, {p23}}};");
                }

                return;
            }

            int step = T / dCap;
            string col2 = b.R(), row0 = b.R(), pc2 = b.P(), gr0 = b.R(), idx0 = b.R(), sBase = b.R();
            b.L($"and.b32 {col2}, {tid}, {dCap - 1};");
            b.L($"shr.u32 {row0}, {tid}, {System.Numerics.BitOperations.Log2((uint)dCap)};");
            b.L($"setp.lt.u32 {pc2}, {col2}, {d};");
            b.L($"add.u32 {gr0}, {first}, {row0};");
            b.L($"mad.lo.u32 {idx0}, {gr0}, {d}, {col2};");
            b.L($"add.u32 {idx0}, {idx0}, {head};");
            b.L($"mad.lo.u32 {sBase}, {row0}, {stride}, {dst};");
            b.L($"mad.lo.u32 {sBase}, {col2}, 2, {sBase};");
            for (int it = 0; it < 64 / step; it++)
            {
                string gr = b.R(), idx = b.R(), pv = b.P(), at = b.R(), hv = b.H(), zero = b.H();
                b.L($"add.u32 {gr}, {gr0}, {it * step};");
                b.L($"mad.lo.u32 {idx}, {d}, {it * step}, {idx0};");
                b.L($"setp.lt.u32 {pv}, {gr}, {rowLimit};");
                b.L($"and.pred {pv}, {pv}, {pc2};");
                b.L($"selp.b32 {at}, {idx}, {head}, {pv};");
                string val = Codec.UnpackAt(b, f, src, at, F);
                b.L($"cvt.rn.{(bf ? "bf16" : "f16")}.f32 {hv}, {val};");
                b.L($"mov.b16 {zero}, 0;");
                b.L($"selp.b16 {hv}, {hv}, {zero}, {pv};");
                b.L($"st.shared.b16 [{sBase}+{it * step * stride}], {hv};");
            }
        }

        // Loads done: the copies waited for (others are plain stores), then the block's barrier.
        public void Landed()
        {
            if (load == 'c') { b.L("cp.async.commit_group;"); b.L("cp.async.wait_group 0;"); }
            b.L("bar.sync 0;");
        }

        // The A operand's lane address in a tile (the warp's 16 rows: row warp·16 + lane mod 16, chunk lane / 16);
        // k-step ks at +ks·32.
        public string AAddr(string tile)
        {
            string a = b.R(), t1 = b.R(), t2 = b.R();
            b.L($"and.b32 {t1}, {ln}, 15;");
            b.L($"mad.lo.u32 {t1}, {warp}, 16, {t1};");
            b.L($"mad.lo.u32 {a}, {t1}, {Stride}, {tile};");
            b.L($"shr.u32 {t2}, {ln}, 4;");
            b.L($"mad.lo.u32 {a}, {t2}, 16, {a};");
            return a;
        }

        // The B operand from a tile whose rows are the n index (row (lane mod 8) + 8·(lane / 16), chunk (lane / 8) mod 2):
        // n-tiles 2jp, 2jp + 1 at k-step ks at +jp·16·Stride + ks·32.
        public string BRowsAddr(string tile)
        {
            string a = b.R(), t3 = b.R(), t5 = b.R();
            b.L($"and.b32 {t3}, {ln}, 7;");
            b.L($"shr.u32 {t5}, {ln}, 4;");
            b.L($"mad.lo.u32 {t3}, {t5}, 8, {t3};");
            b.L($"mad.lo.u32 {a}, {t3}, {Stride}, {tile};");
            b.L($"shr.u32 {t5}, {ln}, 3;");
            b.L($"and.b32 {t5}, {t5}, 1;");
            b.L($"mad.lo.u32 {a}, {t5}, 16, {a};");
            return a;
        }

        // The B operand from a tile whose rows are the k index (transposed: row (lane mod 8) + 8·((lane / 8) mod 2),
        // dimensions 8·(lane / 16)): k-step kk (16 rows), n-tiles 2np, 2np + 1 at +kk·16·Stride + np·32.
        public string BColsAddr(string tile)
        {
            string a = b.R(), t6 = b.R(), t7 = b.R();
            b.L($"and.b32 {t6}, {ln}, 7;");
            b.L($"shr.u32 {t7}, {ln}, 3;");
            b.L($"and.b32 {t7}, {t7}, 1;");
            b.L($"mad.lo.u32 {t6}, {t7}, 8, {t6};");
            b.L($"mad.lo.u32 {a}, {t6}, {Stride}, {tile};");
            b.L($"shr.u32 {t7}, {ln}, 4;");
            b.L($"mad.lo.u32 {a}, {t7}, 16, {a};");
            return a;
        }

        public string[,] Zeros(int tiles)
        {
            var acc = new string[tiles, 4];
            for (int j = 0; j < tiles; j++) for (int e = 0; e < 4; e++) { acc[j, e] = b.F(); b.L($"mov.f32 {acc[j, e]}, 0f00000000;"); }
            return acc;
        }

        // acc[jFirst .. jFirst + count) += A·B over dCap/16 k-steps: A's fragments a[ks] (registers) or from aAddr
        // (ldmatrix at +ks·32), B from a rows tile at bAddr (n-tiles from jFirst on).
        public void Scores(string[,] acc, int jFirst, int count, string[][]? a, string? aAddr, string bAddr)
        {
            for (int ks = 0; ks < dCap / 16; ks++)
            {
                string[] af;
                if (a is not null) af = a[ks];
                else
                {
                    af = [b.R(), b.R(), b.R(), b.R()];
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", af)}}}, [{aAddr}+{ks * 32}];");
                }

                for (int jp = jFirst / 2; jp < (jFirst + count) / 2; jp++)
                {
                    string k0 = b.R(), k1 = b.R(), k2 = b.R(), k3 = b.R();
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{k0}, {k1}, {k2}, {k3}}}, [{bAddr}+{jp * 16 * Stride + ks * 32}];");
                    for (int half = 0; half < 2; half++)
                    {
                        int j = 2 * jp + half - jFirst;
                        string dd = $"{{{acc[j, 0]}, {acc[j, 1]}, {acc[j, 2]}, {acc[j, 3]}}}";
                        b.L($"{kind.Mma} {dd}, {{{string.Join(", ", af)}}}, {{{(half == 0 ? k0 : k2)}, {(half == 0 ? k1 : k3)}}}, {dd};");
                    }
                }
            }
        }

        // x as the kind in two slices (lo half = x0): hi = x rounded; lo = x − hi rounded.
        public (string Hi, string Lo) Split(string x0, string x1)
        {
            string hiw = b.R(), low = b.R(), r0 = b.F(), r1 = b.F();
            b.L($"cvt.rn.{Pair}.f32 {hiw}, {x1}, {x0};");
            var (f0, f1) = Unpack(hiw);
            b.L($"sub.rn.f32 {r0}, {x0}, {f0};");
            b.L($"sub.rn.f32 {r1}, {x1}, {f1};");
            b.L($"cvt.rn.{Pair}.f32 {low}, {r1}, {r0};");
            return (hiw, low);
        }

        // The two values of a pair of the kind (lo half first) as f32.
        public (string Lo, string Hi) Unpack(string word)
        {
            string f0 = b.F(), f1 = b.F();
            if (bf)
            {
                string w0 = b.R(), w1 = b.R();
                b.L($"shl.b32 {w0}, {word}, 16;");
                b.L($"and.b32 {w1}, {word}, 0xFFFF0000;");
                b.L($"mov.b32 {f0}, {w0};");
                b.L($"mov.b32 {f1}, {w1};");
            }
            else
            {
                string h0 = b.H(), h1 = b.H();
                b.L($"mov.b32 {{{h0}, {h1}}}, {word};");
                b.L($"cvt.f32.f16 {f0}, {h0};");
                b.L($"cvt.f32.f16 {f1}, {h1};");
            }

            return (f0, f1);
        }

        // acc += W·B: W the f32 accumulators w (n-tiles of 8 along this product's k, 2·kSteps of them from wFirst) as the
        // A operand in two slices; B from a cols tile at bAddr, its k rows from row kRow0 on (16 a step).
        public void Accumulate(string[,] acc, string[,] w, int wFirst, int kSteps, string bAddr, int kRow0)
        {
            int ntO = acc.GetLength(0);
            for (int kk = 0; kk < kSteps; kk++)
            {
                int j0 = wFirst + 2 * kk;
                var a0 = Split(w[j0, 0], w[j0, 1]);
                var a1 = Split(w[j0, 2], w[j0, 3]);
                var a2 = Split(w[j0 + 1, 0], w[j0 + 1, 1]);
                var a3 = Split(w[j0 + 1, 2], w[j0 + 1, 3]);
                string aHi = $"{{{a0.Hi}, {a1.Hi}, {a2.Hi}, {a3.Hi}}}", aLo = $"{{{a0.Lo}, {a1.Lo}, {a2.Lo}, {a3.Lo}}}";
                for (int np = 0; np < ntO / 2; np++)
                {
                    string v0 = b.R(), v1 = b.R(), v2 = b.R(), v3 = b.R();
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{v0}, {v1}, {v2}, {v3}}}, [{bAddr}+{(kRow0 + kk * 16) * Stride + np * 32}];");
                    for (int half = 0; half < 2; half++)
                    {
                        int j = 2 * np + half;
                        string dd = $"{{{acc[j, 0]}, {acc[j, 1]}, {acc[j, 2]}, {acc[j, 3]}}}", bb = half == 0 ? $"{{{v0}, {v1}}}" : $"{{{v2}, {v3}}}";
                        b.L($"{kind.Mma} {dd}, {aHi}, {bb}, {dd};");
                        b.L($"{kind.Mma} {dd}, {aLo}, {bb}, {dd};");
                    }
                }
            }
        }

        // A row's values lie on the quad's four threads.
        public void QuadReduce(string x, bool max)
        {
            foreach (int off in new[] { 1, 2 })
            {
                string bits = b.R(), other = b.R(), fo = b.F();
                b.L($"mov.b32 {bits}, {x};");
                b.L($"shfl.sync.bfly.b32 {other}, {bits}, {off}, 0x1f, 0xffffffff;");
                b.L($"mov.b32 {fo}, {other};");
                b.L(max ? $"max.f32 {x}, {x}, {fo};" : $"add.rn.f32 {x}, {x}, {fo};");
            }
        }

        // The accumulators' rows (row A: values 0, 1; row B: 2, 3) stored as the type at out[rowIndex·d + dimension],
        // dimension 8j + 2·(lane mod 4) + e mod 2.
        public void Store(string[,] acc, string outPtr, string head, (string I, string Valid) rowA, (string I, string Valid) rowB, string dim0, string label)
        {
            var F = Lane.F32;
            var natural = f.Natural;
            foreach (var (row, es, tag) in new[] { (rowA, new[] { 0, 1 }, "A"), (rowB, new[] { 2, 3 }, "B") })
            {
                string oRow = b.R();
                b.L($"mad.lo.u32 {oRow}, {row.I}, {d}, {head};");
                for (int j = 0; j < acc.GetLength(0); j++)
                {
                    foreach (int e in es)
                    {
                        string dim = b.R(), pDim = b.P(), idx = b.R(), oAddr = b.RD(), code = b.R(), hh = b.H(), skip = $"{label}{tag}{j}_{e}";
                        b.L($"add.u32 {dim}, {dim0}, {8 * j + (e & 1)};");
                        b.L($"setp.ge.u32 {pDim}, {dim}, {d};");
                        b.L($"@{pDim} bra {skip};");
                        b.L($"@!{row.Valid} bra {skip};");
                        b.L($"add.u32 {idx}, {oRow}, {dim};");
                        string stored = natural == F ? acc[j, e] : Codec.Convert(b, F, acc[j, e], natural, f);
                        b.L($"mov.b32 {code}, 0;");
                        Codec.Pack(b, f, stored, natural, 0, code);
                        b.L($"mul.wide.u32 {oAddr}, {idx}, {f.Bits / 8};");
                        b.L($"add.u64 {oAddr}, {outPtr}, {oAddr};");
                        b.L($"cvt.u16.u32 {hh}, {code};");
                        b.L($"st.global.b{f.Bits} [{oAddr}], {hh};");
                        b.Label(skip);
                    }
                }
            }
        }
    }

    public static string AttentionBackRowsMma(Gpu gpu, Fmt f, Kind kind, int dCap, char load, out string entry)
    {
        const int Br = MmaRowsBr, Bc = MmaRowsBc;
        int ksteps = dCap / 16, ntO = dCap / 8, ntS = Bc / 8;
        var F = Lane.F32;
        entry = $"attention_back_rows_mma_{f.Name}_{kind.Name}_d{dCap}_{load}";
        var b = new PtxBuilder();
        string tid = b.R(), qb = b.R(), h = b.R(), warp = b.R(), ln = b.R(), g = b.R(), t4 = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {qb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {ln}, {tid}, 31;");
        b.L($"shr.u32 {g}, {ln}, 2;");
        b.L($"and.b32 {t4}, {ln}, 3;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), dO = Param64(b, "p_do"), dq = Param64(b, "p_dq"), stats = Param64(b, "p_stats");
        string lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = b.F(), negInf = F.Literal(double.NegativeInfinity);
        b.L($"ld.param.f32 {scale}, [p_scale];");
        var P = new MmaBackPieces(b, f, kind, dCap, load, tid, ln, warp, d);
        int stride = P.Stride;

        // The thread's two rows (g and g + 8 of the warp's 16): index, validity, keys [start, limit).
        string lqm1 = b.R();
        b.L($"sub.u32 {lqm1}, {lq}, 1;");
        (string I, string Valid, string Limit, string Start) Row(int plus)
        {
            string i = b.R(), valid = b.P(), ic = b.R(), limit = b.R(), start = b.R(), off = b.RD(), la = b.RD(), sa = b.RD();
            b.L($"mad.lo.u32 {i}, {qb}, {Br}, {plus};");
            b.L($"mad.lo.u32 {i}, {warp}, 16, {i};");
            b.L($"add.u32 {i}, {i}, {g};");
            b.L($"setp.lt.u32 {valid}, {i}, {lq};");
            b.L($"min.u32 {ic}, {i}, {lqm1};");
            b.L($"mul.wide.u32 {off}, {ic}, 4;");
            b.L($"add.u64 {la}, {lim}, {off};");
            b.L($"add.u64 {sa}, {starts}, {off};");
            b.L($"ld.global.u32 {limit}, [{la}];");
            b.L($"ld.global.u32 {start}, [{sa}];");
            b.L($"min.u32 {limit}, {limit}, {lk};");
            b.L($"selp.b32 {limit}, {limit}, 0, {valid};");
            return (i, valid, limit, start);
        }

        var rowA = Row(0);
        var rowB = Row(8);
        string qHead = b.R(), kHead = b.R(), lo = b.R(), hi = b.R(), ra = b.RD();
        b.L($"mul.lo.u32 {qHead}, {h}, {lq};");
        b.L($"mul.lo.u32 {qHead}, {qHead}, {d};");
        b.L($"mul.lo.u32 {kHead}, {h}, {lk};");
        b.L($"mul.lo.u32 {kHead}, {kHead}, {d};");
        b.L($"mul.wide.u32 {ra}, {qb}, 8;");
        b.L($"add.u64 {ra}, {ranges}, {ra};");
        b.L($"ld.global.u32 {lo}, [{ra}];");
        b.L($"ld.global.u32 {hi}, [{ra}+4];");
        string sK = b.R(), sV = b.R();
        b.L($"mov.u32 {sK}, smem;");
        b.L($"add.u32 {sV}, {sK}, {Bc * stride};");

        // Q's and dO's fragments into registers, each through the V tile's space.
        string qFirst = b.R();
        b.L($"mul.lo.u32 {qFirst}, {qb}, {Br};");
        string aAddr = P.AAddr(sV);
        string[][] Fragments(string src)
        {
            P.LoadTile(sV, qHead, qFirst, lq, src);
            P.Landed();
            var fr = new string[ksteps][];
            for (int ks = 0; ks < ksteps; ks++)
            {
                fr[ks] = [b.R(), b.R(), b.R(), b.R()];
                b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", fr[ks])}}}, [{aAddr}+{ks * 32}];");
            }

            b.L("bar.sync 0;");
            return fr;
        }

        var qf = Fragments(q);
        var dof = Fragments(dO);
        string kRows = P.BRowsAddr(sK), vRows = P.BRowsAddr(sV), vCols = P.BColsAddr(sV), kCols = P.BColsAddr(sK);
        string key0 = b.R();
        b.L($"shl.b32 {key0}, {t4}, 1;");

        // A tile's K and V (both before the products).
        string kt = b.R(), pDone = b.P();
        void LoadKeys()
        {
            P.LoadTile(sK, kHead, kt, lk, k);
            P.LoadTile(sV, kHead, kt, lk, v);
            P.Landed();
        }

        // valid[j, e]: the key of value e of n-tile j (first n-tile jFirst) seen by its row.
        string[,] Masked(string[,] s, int jFirst, int count)
        {
            var valid = new string[count, 4];
            for (int j = 0; j < count; j++)
            {
                for (int e = 0; e < 4; e++)
                {
                    var row = e < 2 ? rowA : rowB;
                    string key = b.R(), p1 = b.P();
                    valid[j, e] = b.P();
                    b.L($"add.u32 {key}, {kt}, {key0};");
                    b.L($"add.u32 {key}, {key}, {8 * (jFirst + j) + (e & 1)};");
                    b.L($"setp.ge.u32 {valid[j, e]}, {key}, {row.Start};");
                    b.L($"setp.lt.u32 {p1}, {key}, {row.Limit};");
                    b.L($"and.pred {valid[j, e]}, {valid[j, e]}, {p1};");
                }
            }

            return valid;
        }

        // Pass 1: the forward (the online max and sum; the context in f32 accumulators).
        string mA = b.F(), mB = b.F(), lA = b.F(), lB = b.F();
        foreach (var r in new[] { mA, mB }) b.L($"mov.f32 {r}, {negInf};");
        foreach (var r in new[] { lA, lB }) b.L($"mov.f32 {r}, 0f00000000;");
        var acc = P.Zeros(ntO);
        b.L($"mov.u32 {kt}, {lo};");
        b.Label("F_TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {hi};");
        b.L($"@{pDone} bra F_END;");
        LoadKeys();
        {
            var s = P.Zeros(ntS);
            P.Scores(s, 0, ntS, qf, null, kRows);
            var valid = Masked(s, 0, ntS);
            string tmA = b.F(), tmB = b.F();
            b.L($"mov.f32 {tmA}, {negInf};");
            b.L($"mov.f32 {tmB}, {negInf};");
            for (int j = 0; j < ntS; j++)
            {
                for (int e = 0; e < 4; e++)
                {
                    b.L($"mul.rn.f32 {s[j, e]}, {s[j, e]}, {scale};");
                    b.L($"selp.f32 {s[j, e]}, {s[j, e]}, {negInf}, {valid[j, e]};");
                    string tm = e < 2 ? tmA : tmB;
                    b.L($"max.f32 {tm}, {tm}, {s[j, e]};");
                }
            }

            P.QuadReduce(tmA, true);
            P.QuadReduce(tmB, true);
            (string Use, string Alpha, string New) Running(string m, string tm)
            {
                string mNew = b.F(), mUse = b.F(), pInf = b.P(), dm = b.F();
                b.L($"max.f32 {mNew}, {m}, {tm};");
                b.L($"setp.eq.f32 {pInf}, {mNew}, {negInf};");
                b.L($"selp.f32 {mUse}, 0f00000000, {mNew}, {pInf};");
                b.L($"sub.rn.f32 {dm}, {m}, {mUse};");
                return (mUse, Codec.Exp(b, F, dm), mNew);
            }

            var ruA = Running(mA, tmA);
            var ruB = Running(mB, tmB);
            string psA = b.F(), psB = b.F();
            b.L($"mov.f32 {psA}, 0f00000000;");
            b.L($"mov.f32 {psB}, 0f00000000;");
            for (int j = 0; j < ntS; j++)
            {
                for (int e = 0; e < 4; e++)
                {
                    var ru = e < 2 ? ruA : ruB;
                    string ds = b.F();
                    b.L($"sub.rn.f32 {ds}, {s[j, e]}, {ru.Use};");
                    string ex = Codec.Exp(b, F, ds);
                    b.L($"selp.f32 {s[j, e]}, {ex}, 0f00000000, {valid[j, e]};");
                    string ps = e < 2 ? psA : psB;
                    b.L($"add.rn.f32 {ps}, {ps}, {s[j, e]};");
                }
            }

            P.QuadReduce(psA, false);
            P.QuadReduce(psB, false);
            b.L($"fma.rn.f32 {lA}, {lA}, {ruA.Alpha}, {psA};");
            b.L($"fma.rn.f32 {lB}, {lB}, {ruB.Alpha}, {psB};");
            b.L($"mov.f32 {mA}, {ruA.New};");
            b.L($"mov.f32 {mB}, {ruB.New};");
            for (int j = 0; j < ntO; j++)
            {
                b.L($"mul.rn.f32 {acc[j, 0]}, {acc[j, 0]}, {ruA.Alpha};");
                b.L($"mul.rn.f32 {acc[j, 1]}, {acc[j, 1]}, {ruA.Alpha};");
                b.L($"mul.rn.f32 {acc[j, 2]}, {acc[j, 2]}, {ruB.Alpha};");
                b.L($"mul.rn.f32 {acc[j, 3]}, {acc[j, 3]}, {ruB.Alpha};");
            }

            P.Accumulate(acc, s, 0, Bc / 16, vCols, 0);
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {kt}, {kt}, {Bc};");
        b.L("bra F_TILE;");
        b.Label("F_END");

        // Each row's m (0 while no key: P is 0 there), 1/l (0 when none) and D = Σ_dims dO · O (O = acc / l; dO's values of
        // n-tile 2ks + t are fragment registers 2t (row A) and 2t + 1 (row B) of k-step ks).
        (string M, string Inv, string D) Stats(string m, string l, int rowSel)
        {
            string inv = b.F(), pAny = b.P(), mUse = b.F(), pInf = b.P(), dsum = b.F();
            b.L($"setp.gt.f32 {pAny}, {l}, 0f00000000;");
            b.L($"div.rn.f32 {inv}, 0f3F800000, {l};");
            b.L($"selp.f32 {inv}, {inv}, 0f00000000, {pAny};");
            b.L($"setp.eq.f32 {pInf}, {m}, {negInf};");
            b.L($"selp.f32 {mUse}, 0f00000000, {m}, {pInf};");
            b.L($"mov.f32 {dsum}, 0f00000000;");
            for (int ks = 0; ks < ksteps; ks++)
            {
                for (int t = 0; t < 2; t++)
                {
                    int j = 2 * ks + t;
                    var (x0, x1) = P.Unpack(dof[ks][2 * t + rowSel]);
                    string o0 = b.F(), o1 = b.F();
                    b.L($"mul.rn.f32 {o0}, {acc[j, 2 * rowSel]}, {inv};");
                    b.L($"mul.rn.f32 {o1}, {acc[j, 2 * rowSel + 1]}, {inv};");
                    b.L($"fma.rn.f32 {dsum}, {x0}, {o0}, {dsum};");
                    b.L($"fma.rn.f32 {dsum}, {x1}, {o1}, {dsum};");
                }
            }

            P.QuadReduce(dsum, false);
            return (mUse, inv, dsum);
        }

        var stA = Stats(mA, lA, 0);
        var stB = Stats(mB, lB, 1);
        foreach (var (row, st, tag) in new[] { (rowA, stA, "A"), (rowB, stB, "B") })
        {
            string pW = b.P(), sAddr = b.RD(), sIdx = b.R(), skip = $"STATS_{tag}";
            b.L($"setp.eq.and.u32 {pW}, {t4}, 0, {row.Valid};");
            b.L($"@!{pW} bra {skip};");
            b.L($"mad.lo.u32 {sIdx}, {h}, {lq}, {row.I};");
            b.L($"mul.wide.u32 {sAddr}, {sIdx}, 12;");
            b.L($"add.u64 {sAddr}, {stats}, {sAddr};");
            b.L($"st.global.f32 [{sAddr}], {st.M};");
            b.L($"st.global.f32 [{sAddr}+4], {st.Inv};");
            b.L($"st.global.f32 [{sAddr}+8], {st.D};");
            b.Label(skip);
        }

        // Pass 2: 32 keys at a time, dS = scale·P·(dP − D), dQ += dS·K.
        var dqAcc = P.Zeros(ntO);
        b.L($"mov.u32 {kt}, {lo};");
        b.Label("B_TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {hi};");
        b.L($"@{pDone} bra B_END;");
        LoadKeys();
        for (int half = 0; half < 2; half++)
        {
            int jFirst = 4 * half;
            var s = P.Zeros(4);
            var dp = P.Zeros(4);
            P.Scores(s, jFirst, 4, qf, null, kRows);
            P.Scores(dp, jFirst, 4, dof, null, vRows);
            var valid = Masked(s, jFirst, 4);
            for (int j = 0; j < 4; j++)
            {
                for (int e = 0; e < 4; e++)
                {
                    var st = e < 2 ? stA : stB;
                    string x = b.F(), p = b.F(), w = b.F();
                    b.L($"mul.rn.f32 {x}, {s[j, e]}, {scale};");
                    b.L($"sub.rn.f32 {x}, {x}, {st.M};");
                    b.L($"selp.f32 {x}, {x}, 0f00000000, {valid[j, e]};");
                    string ex = Codec.Exp(b, F, x);
                    b.L($"mul.rn.f32 {p}, {ex}, {st.Inv};");
                    b.L($"selp.f32 {p}, {p}, 0f00000000, {valid[j, e]};");
                    b.L($"sub.rn.f32 {w}, {dp[j, e]}, {st.D};");
                    b.L($"mul.rn.f32 {w}, {w}, {p};");
                    b.L($"mul.rn.f32 {s[j, e]}, {w}, {scale};");
                }
            }

            P.Accumulate(dqAcc, s, 0, 2, kCols, 32 * half);
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {kt}, {kt}, {Bc};");
        b.L("bra B_TILE;");
        b.Label("B_END");
        P.Store(dqAcc, dq, qHead, (rowA.I, rowA.Valid), (rowB.I, rowB.Valid), key0, "DQ");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_do, .param .u64 p_dq, .param .u64 p_stats, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .f32 p_scale",
            AttentionMmaShared(dCap));
    }

    public static string AttentionBackKeysMma(Gpu gpu, Fmt f, Kind kind, int dCap, char load, out string entry)
    {
        const int Br = MmaRowsBr, Bc = MmaRowsBc;
        int ntO = dCap / 8;
        var F = Lane.F32;
        entry = $"attention_back_keys_mma_{f.Name}_{kind.Name}_d{dCap}_{load}";
        var b = new PtxBuilder();
        string tid = b.R(), kb = b.R(), h = b.R(), warp = b.R(), ln = b.R(), g = b.R(), t4 = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {kb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {ln}, {tid}, 31;");
        b.L($"shr.u32 {g}, {ln}, 2;");
        b.L($"and.b32 {t4}, {ln}, 3;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), dO = Param64(b, "p_do"), stats = Param64(b, "p_stats"), dk = Param64(b, "p_dk"), dv = Param64(b, "p_dv");
        string lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = b.F();
        b.L($"ld.param.f32 {scale}, [p_scale];");
        var P = new MmaBackPieces(b, f, kind, dCap, load, tid, ln, warp, d);
        int stride = P.Stride;

        // The thread's two keys (rows g and g + 8 of the warp's 16).
        (string I, string Valid) Key(int plus)
        {
            string i = b.R(), valid = b.P();
            b.L($"mad.lo.u32 {i}, {kb}, {Bc}, {plus};");
            b.L($"mad.lo.u32 {i}, {warp}, 16, {i};");
            b.L($"add.u32 {i}, {i}, {g};");
            b.L($"setp.lt.u32 {valid}, {i}, {lk};");
            return (i, valid);
        }

        var keyA = Key(0);
        var keyB = Key(8);
        string qHead = b.R(), kHead = b.R(), lo = b.R(), hi = b.R(), ra = b.RD();
        b.L($"mul.lo.u32 {qHead}, {h}, {lq};");
        b.L($"mul.lo.u32 {qHead}, {qHead}, {d};");
        b.L($"mul.lo.u32 {kHead}, {h}, {lk};");
        b.L($"mul.lo.u32 {kHead}, {kHead}, {d};");
        b.L($"mul.wide.u32 {ra}, {kb}, 8;");
        b.L($"add.u64 {ra}, {ranges}, {ra};");
        b.L($"ld.global.u32 {lo}, [{ra}];");
        b.L($"ld.global.u32 {hi}, [{ra}+4];");
        // Shared: K, V of the keys; Q, dO of the tile; the tile's rows' m, 1/l, D (f32) and [start, limit).
        string sK = b.R(), sV = b.R(), sQ = b.R(), sDO = b.R(), sStat = b.R(), sLim = b.R();
        b.L($"mov.u32 {sK}, smem;");
        b.L($"add.u32 {sV}, {sK}, {Bc * stride};");
        b.L($"add.u32 {sQ}, {sV}, {Bc * stride};");
        b.L($"add.u32 {sDO}, {sQ}, {Br * stride};");
        b.L($"add.u32 {sStat}, {sDO}, {Br * stride};");
        b.L($"add.u32 {sLim}, {sStat}, {Br * 12};");
        string kFirst = b.R();
        b.L($"mul.lo.u32 {kFirst}, {kb}, {Bc};");
        P.LoadTile(sK, kHead, kFirst, lk, k);
        P.LoadTile(sV, kHead, kFirst, lk, v);
        P.Landed();
        string kA = P.AAddr(sK), vA = P.AAddr(sV), qRows = P.BRowsAddr(sQ), doRows = P.BRowsAddr(sDO), qCols = P.BColsAddr(sQ), doCols = P.BColsAddr(sDO);
        string col0 = b.R();
        b.L($"shl.b32 {col0}, {t4}, 1;");                                   // the thread's first query of an n-tile

        var dkAcc = P.Zeros(ntO);
        var dvAcc = P.Zeros(ntO);
        string qt = b.R(), pDone = b.P();
        b.L($"mov.u32 {qt}, {lo};");
        b.Label("Q_TILE");
        b.L($"setp.ge.u32 {pDone}, {qt}, {hi};");
        b.L($"@{pDone} bra Q_END;");
        P.LoadTile(sQ, qHead, qt, lq, q);
        P.LoadTile(sDO, qHead, qt, lq, dO);
        {
            // The tile's rows' m, 1/l, D and keys [start, limit) (limit 0 past the queries): threads 0 .. 63.
            string pT = b.P(), row = b.R(), pRowV = b.P(), idx = b.R(), sAddr = b.RD(), lAddr = b.RD(), lv = b.R(), sv = b.R(), sw = b.R(), lw = b.R(), skip = "STAT_LOADED";
            b.L($"setp.ge.u32 {pT}, {tid}, {Br};");
            b.L($"@{pT} bra {skip};");
            b.L($"add.u32 {row}, {qt}, {tid};");
            b.L($"setp.lt.u32 {pRowV}, {row}, {lq};");
            b.L($"selp.b32 {idx}, {row}, 0, {pRowV};");
            b.L($"mad.lo.u32 {idx}, {h}, {lq}, {idx};");
            b.L($"mul.wide.u32 {sAddr}, {idx}, 12;");
            b.L($"add.u64 {sAddr}, {stats}, {sAddr};");
            b.L($"mad.lo.u32 {sw}, {tid}, 12, {sStat};");
            for (int t = 0; t < 3; t++)
            {
                string x = b.F();
                b.L($"ld.global.f32 {x}, [{sAddr}+{4 * t}];");
                b.L($"st.shared.f32 [{sw}+{4 * t}], {x};");
            }

            string ri = b.R();
            b.L($"selp.b32 {ri}, {row}, 0, {pRowV};");
            b.L($"mul.wide.u32 {lAddr}, {ri}, 4;");
            string la2 = b.RD(), sa2 = b.RD();
            b.L($"add.u64 {la2}, {lim}, {lAddr};");
            b.L($"add.u64 {sa2}, {starts}, {lAddr};");
            b.L($"ld.global.u32 {lv}, [{la2}];");
            b.L($"ld.global.u32 {sv}, [{sa2}];");
            b.L($"min.u32 {lv}, {lv}, {lk};");
            b.L($"selp.b32 {lv}, {lv}, 0, {pRowV};");
            b.L($"mad.lo.u32 {lw}, {tid}, 8, {sLim};");
            b.L($"st.shared.u32 [{lw}], {lv};");
            b.L($"st.shared.u32 [{lw}+4], {sv};");
            b.Label(skip);
        }

        P.Landed();
        for (int half = 0; half < 2; half++)
        {
            int jFirst = 4 * half;
            var s = P.Zeros(4);
            var dp = P.Zeros(4);
            P.Scores(s, jFirst, 4, null, kA, qRows);
            P.Scores(dp, jFirst, 4, null, vA, doRows);
            for (int j = 0; j < 4; j++)
            {
                for (int e = 0; e < 4; e++)
                {
                    // Value e of n-tile j: key A (e < 2) or B, query qt + 8·(jFirst + j) + 2·(lane mod 4) + e mod 2.
                    var key = e < 2 ? keyA : keyB;
                    string qi = b.R(), sw = b.R(), lw = b.R(), mm = b.F(), il = b.F(), dd = b.F(), lv = b.R(), sv = b.R(), pv = b.P(), p1 = b.P();
                    string x = b.F(), p = b.F(), w = b.F();
                    b.L($"add.u32 {qi}, {col0}, {8 * (jFirst + j) + (e & 1)};");
                    b.L($"mad.lo.u32 {sw}, {qi}, 12, {sStat};");
                    b.L($"mad.lo.u32 {lw}, {qi}, 8, {sLim};");
                    b.L($"ld.shared.f32 {mm}, [{sw}];");
                    b.L($"ld.shared.f32 {il}, [{sw}+4];");
                    b.L($"ld.shared.f32 {dd}, [{sw}+8];");
                    b.L($"ld.shared.u32 {lv}, [{lw}];");
                    b.L($"ld.shared.u32 {sv}, [{lw}+4];");
                    b.L($"setp.ge.u32 {pv}, {key.I}, {sv};");
                    b.L($"setp.lt.u32 {p1}, {key.I}, {lv};");
                    b.L($"and.pred {pv}, {pv}, {p1};");
                    b.L($"mul.rn.f32 {x}, {s[j, e]}, {scale};");
                    b.L($"sub.rn.f32 {x}, {x}, {mm};");
                    b.L($"selp.f32 {x}, {x}, 0f00000000, {pv};");
                    string ex = Codec.Exp(b, F, x);
                    b.L($"mul.rn.f32 {p}, {ex}, {il};");
                    b.L($"selp.f32 {p}, {p}, 0f00000000, {pv};");
                    b.L($"sub.rn.f32 {w}, {dp[j, e]}, {dd};");
                    b.L($"mul.rn.f32 {w}, {w}, {p};");
                    b.L($"mul.rn.f32 {dp[j, e]}, {w}, {scale};");
                    b.L($"mov.f32 {s[j, e]}, {p};");
                }
            }

            P.Accumulate(dvAcc, s, 0, 2, doCols, 32 * half);
            P.Accumulate(dkAcc, dp, 0, 2, qCols, 32 * half);
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {qt}, {qt}, {Br};");
        b.L("bra Q_TILE;");
        b.Label("Q_END");
        P.Store(dkAcc, dk, kHead, keyA, keyB, col0, "DK");
        P.Store(dvAcc, dv, kHead, keyA, keyB, col0, "DV");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_do, .param .u64 p_stats, .param .u64 p_dk, .param .u64 p_dv, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .f32 p_scale",
            AttentionBackKeysMmaShared(dCap));
    }
}
