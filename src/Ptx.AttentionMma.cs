// The device's fused attention rows on the tensor instructions (item 6i): Q, K and V in the f16 or bf16 kind (exact: the
// float16, bfloat16 and 8-bit float formats' values are the kind's), the scores and the context accumulated in f32.
// A block of 4 warps per (head, 64 query rows), each warp 16 rows; the keys walked in tiles of 64 through shared memory.
//
//   S = Q·Kᵀ        Q's fragments held in registers (ldmatrix once), K's from the tile (ldmatrix): m16n8k16 → f32
//   softmax online  a row's 64 scores of a tile lie on the four threads of a quad: max and sum by two shuffles; the
//                   running max, sum and the context rescaled as the f32 fused rows do (Codec.Exp, the same masks)
//   O += P·V        P from the score registers (an m16n8 accumulator pair is an m16k16 operand), as two slices of the
//                   kind (hi = P rounded, lo = P − hi rounded: P to 22 bits in f16, 16 in bf16, so the context keeps the
//                   f32 lane's rounding); V's fragments from the tile by ldmatrix.trans (V stored key-major)
//
// Loads: a 16-bit format in its own kind with d a multiple of 8 copies 16-byte runs (cp.async, zero-filled past the
// rows and dimensions; V's copies land while the scores run); an 8-bit format with d a multiple of 4 reads 32-bit words
// (four values decoded into the kind); any other decodes value by value. O stored value by value as the type.
namespace Qiyas;

internal static partial class Ptx
{
    public const int MmaRowsThreads = 128, MmaRowsBr = 64, MmaRowsBc = 64;

    /// <summary>The tensor kind whose values hold a format's exactly (null: none, the FMA rows).</summary>
    public static Kind? AttentionKind(Fmt f) => f.Name switch
    {
        "float16" or "float8_e4m3" or "float8_e5m2" => Kind.F16,
        "bfloat16" => Kind.BF16,
        _ => null,
    };

    /// <summary>Shared bytes: the K tile and the V tile (the Q tile first), rows of dCap values padded by 16 bytes.</summary>
    public static int AttentionMmaShared(int dCap) => 2 * MmaRowsBc * (dCap * 2 + 16);

    /// <summary>load: 'c' 16-byte copies (a 16-bit format in its own kind, d a multiple of 8), 'w' 32-bit words of an
    /// 8-bit format (d a multiple of 4), 'v' value by value.</summary>
    public static string AttentionRowsMma(Gpu gpu, Fmt f, Kind kind, int dCap, char load, out string entry)
    {
        const int T = MmaRowsThreads, Br = MmaRowsBr, Bc = MmaRowsBc;
        int stride = dCap * 2 + 16, ksteps = dCap / 16, ntS = Bc / 8, ntO = dCap / 8;
        var F = Lane.F32;
        bool bf = kind == Kind.BF16;
        bool copies = load == 'c';
        entry = $"attention_rows_mma_{f.Name}_{kind.Name}_d{dCap}_{load}";
        var b = new PtxBuilder();
        string tid = b.R(), qb = b.R(), h = b.R(), warp = b.R(), ln = b.R(), g = b.R(), t4 = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {qb}, %ctaid.x;");
        b.L($"mov.u32 {h}, %ctaid.y;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {ln}, {tid}, 31;");
        b.L($"shr.u32 {g}, {ln}, 2;");
        b.L($"and.b32 {t4}, {ln}, 3;");
        string q = Param64(b, "p_q"), k = Param64(b, "p_k"), v = Param64(b, "p_v"), o = Param64(b, "p_o"), lim = Param64(b, "p_lim"), starts = Param64(b, "p_start"), ranges = Param64(b, "p_range");
        string lq = Param32(b, "p_lq"), lk = Param32(b, "p_lk"), d = Param32(b, "p_d");
        string scale = b.F(), negInf = F.Literal(double.NegativeInfinity);
        b.L($"ld.param.f32 {scale}, [p_scale];");

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

        // 64 rows [first, first + 64) of a tensor (d values a row from value `head`; rows past rowLimit and dimensions
        // past d as 0) into shared at dst in the kind, rows `stride` bytes apart.
        void LoadTile(string dst, string head, string first, string rowLimit, string src)
        {
            if (copies)
            {
                // 16-byte runs: run c = tid + T·j → row c / (dCap/8), part c mod (dCap/8).
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
                    b.L($"selp.b32 {idx}, {idx}, {head}, {pv};");             // a dropped run reads the head's first (valid)
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
                // Words of four 8-bit values: word c = tid + T·j → row c / (dCap/4), values 4·(c mod (dCap/4)) ..; four
                // values decoded, stored as two pairs of the kind (8 bytes).
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
                    b.L($"selp.b32 {w}, {w}, 0, {pv};");                         // code 0: +0 in every 8-bit float
                    var x = new string[4];
                    for (int t = 0; t < 4; t++) x[t] = Codec.Unpack(b, f, w, t, F);
                    b.L($"cvt.rn.{(bf ? "bf16x2" : "f16x2")}.f32 {p01}, {x[1]}, {x[0]};");
                    b.L($"cvt.rn.{(bf ? "bf16x2" : "f16x2")}.f32 {p23}, {x[3]}, {x[2]};");
                    b.L($"mad.lo.u32 {sa}, {row}, {stride}, {dst};");
                    b.L($"mad.lo.u32 {sa}, {col}, 2, {sa};");
                    b.L($"st.shared.v2.b32 [{sa}], {{{p01}, {p23}}};");
                }

                return;
            }

            // Value by value: a thread's column is fixed (tid mod dCap), its rows step by T / dCap.
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

        // Q into the V tile's space; its fragments (rows warp·16 + lane mod 16, chunk lane / 16) into registers.
        string qFirst = b.R();
        b.L($"mul.lo.u32 {qFirst}, {qb}, {Br};");
        LoadTile(sV, qHead, qFirst, lq, q);
        if (copies) { b.L("cp.async.commit_group;"); b.L("cp.async.wait_group 0;"); }
        b.L("bar.sync 0;");
        string qAddr = b.R(), t1 = b.R(), t2 = b.R();
        b.L($"and.b32 {t1}, {ln}, 15;");
        b.L($"mad.lo.u32 {t1}, {warp}, 16, {t1};");
        b.L($"mad.lo.u32 {qAddr}, {t1}, {stride}, {sV};");
        b.L($"shr.u32 {t2}, {ln}, 4;");
        b.L($"mad.lo.u32 {qAddr}, {t2}, 16, {qAddr};");
        var qf = new string[ksteps][];
        for (int ks = 0; ks < ksteps; ks++)
        {
            qf[ks] = [b.R(), b.R(), b.R(), b.R()];
            b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", qf[ks])}}}, [{qAddr}+{ks * 32}];");
        }

        b.L("bar.sync 0;");

        // K fragments' lane address: key (lane mod 8) + 8·(lane / 16), chunk (lane / 8) mod 2.
        string kAddr = b.R(), t3 = b.R(), t5 = b.R();
        b.L($"and.b32 {t3}, {ln}, 7;");
        b.L($"shr.u32 {t5}, {ln}, 4;");
        b.L($"mad.lo.u32 {t3}, {t5}, 8, {t3};");
        b.L($"mad.lo.u32 {kAddr}, {t3}, {stride}, {sK};");
        b.L($"shr.u32 {t5}, {ln}, 3;");
        b.L($"and.b32 {t5}, {t5}, 1;");
        b.L($"mad.lo.u32 {kAddr}, {t5}, 16, {kAddr};");
        // V fragments' lane address (transposed): key (lane mod 8) + 8·((lane / 8) mod 2), dimensions 8·(lane / 16).
        string vAddr = b.R(), t6 = b.R(), t7 = b.R();
        b.L($"and.b32 {t6}, {ln}, 7;");
        b.L($"shr.u32 {t7}, {ln}, 3;");
        b.L($"and.b32 {t7}, {t7}, 1;");
        b.L($"mad.lo.u32 {t6}, {t7}, 8, {t6};");
        b.L($"mad.lo.u32 {vAddr}, {t6}, {stride}, {sV};");
        b.L($"shr.u32 {t7}, {ln}, 4;");
        b.L($"mad.lo.u32 {vAddr}, {t7}, 16, {vAddr};");

        // The rows' running max and sum, the context (dCap/8 m16n8 tiles).
        string mA = b.F(), mB = b.F(), lA = b.F(), lB = b.F();
        foreach (var r in new[] { mA, mB }) b.L($"mov.f32 {r}, {negInf};");
        foreach (var r in new[] { lA, lB }) b.L($"mov.f32 {r}, 0f00000000;");
        var acc = new string[ntO, 4];
        for (int j = 0; j < ntO; j++) for (int e = 0; e < 4; e++) { acc[j, e] = b.F(); b.L($"mov.f32 {acc[j, e]}, 0f00000000;"); }
        string key0 = b.R();
        b.L($"shl.b32 {key0}, {t4}, 1;");                                   // the thread's first key of an n-tile

        string kt = b.R(), pDone = b.P();
        b.L($"mov.u32 {kt}, {lo};");
        b.Label("TILE");
        b.L($"setp.ge.u32 {pDone}, {kt}, {hi};");
        b.L($"@{pDone} bra TILE_END;");
        // K, then V (copies: two groups, the scores waiting for K only: V lands while they run).
        LoadTile(sK, kHead, kt, lk, k);
        if (copies) b.L("cp.async.commit_group;");
        LoadTile(sV, kHead, kt, lk, v);
        if (copies) { b.L("cp.async.commit_group;"); b.L("cp.async.wait_group 1;"); }
        b.L("bar.sync 0;");

        // S = Q·Kᵀ: 8 n-tiles of 8 keys.
        var s = new string[ntS, 4];
        for (int j = 0; j < ntS; j++) for (int e = 0; e < 4; e++) { s[j, e] = b.F(); b.L($"mov.f32 {s[j, e]}, 0f00000000;"); }
        for (int ks = 0; ks < ksteps; ks++)
        {
            for (int jp = 0; jp < ntS / 2; jp++)
            {
                string k0 = b.R(), k1 = b.R(), k2 = b.R(), k3 = b.R();
                b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{k0}, {k1}, {k2}, {k3}}}, [{kAddr}+{jp * 16 * stride + ks * 32}];");
                for (int half = 0; half < 2; half++)
                {
                    int j = 2 * jp + half;
                    string dd = $"{{{s[j, 0]}, {s[j, 1]}, {s[j, 2]}, {s[j, 3]}}}";
                    b.L($"{kind.Mma} {dd}, {{{string.Join(", ", qf[ks])}}}, {{{(half == 0 ? k0 : k2)}, {(half == 0 ? k1 : k3)}}}, {dd};");
                }
            }
        }

        // Scale and mask (a value e of n-tile j: key kt + 8j + 2·(lane mod 4) + e mod 2, row A for e < 2, else B).
        var valid = new string[ntS, 4];
        string tmA = b.F(), tmB = b.F();
        b.L($"mov.f32 {tmA}, {negInf};");
        b.L($"mov.f32 {tmB}, {negInf};");
        for (int j = 0; j < ntS; j++)
        {
            for (int e = 0; e < 4; e++)
            {
                var row = e < 2 ? rowA : rowB;
                string key = b.R(), p1 = b.P();
                valid[j, e] = b.P();
                b.L($"add.u32 {key}, {kt}, {key0};");
                if (8 * j + (e & 1) != 0) b.L($"add.u32 {key}, {key}, {8 * j + (e & 1)};");
                b.L($"mul.rn.f32 {s[j, e]}, {s[j, e]}, {scale};");
                b.L($"setp.ge.u32 {valid[j, e]}, {key}, {row.Start};");
                b.L($"setp.lt.u32 {p1}, {key}, {row.Limit};");
                b.L($"and.pred {valid[j, e]}, {valid[j, e]}, {p1};");
                b.L($"selp.f32 {s[j, e]}, {s[j, e]}, {negInf}, {valid[j, e]};");
                string tm = e < 2 ? tmA : tmB;
                b.L($"max.f32 {tm}, {tm}, {s[j, e]};");
            }
        }

        // A row's values lie on the quad's four threads (lanes 4g .. 4g + 3).
        void QuadReduce(string x, bool max)
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

        QuadReduce(tmA, true);
        QuadReduce(tmB, true);
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

        QuadReduce(psA, false);
        QuadReduce(psB, false);
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

        // P as the kind, two slices: (lo half = the lower key). hi = P rounded; lo = P − hi rounded.
        (string Hi, string Lo) Split(string p0, string p1)
        {
            string hiw = b.R(), low = b.R(), f0 = b.F(), f1 = b.F(), r0 = b.F(), r1 = b.F();
            b.L($"cvt.rn.{(bf ? "bf16x2" : "f16x2")}.f32 {hiw}, {p1}, {p0};");
            if (bf)
            {
                string w0 = b.R(), w1 = b.R();
                b.L($"shl.b32 {w0}, {hiw}, 16;");
                b.L($"and.b32 {w1}, {hiw}, 0xFFFF0000;");
                b.L($"mov.b32 {f0}, {w0};");
                b.L($"mov.b32 {f1}, {w1};");
            }
            else
            {
                string h0 = b.H(), h1 = b.H();
                b.L($"mov.b32 {{{h0}, {h1}}}, {hiw};");
                b.L($"cvt.f32.f16 {f0}, {h0};");
                b.L($"cvt.f32.f16 {f1}, {h1};");
            }

            b.L($"sub.rn.f32 {r0}, {p0}, {f0};");
            b.L($"sub.rn.f32 {r1}, {p1}, {f1};");
            b.L($"cvt.rn.{(bf ? "bf16x2" : "f16x2")}.f32 {low}, {r1}, {r0};");
            return (hiw, low);
        }

        if (copies) { b.L("cp.async.wait_group 0;"); b.L("bar.sync 0;"); }

        // O += P·V: 16 keys a step (n-tiles 2kk and 2kk + 1 of S), 16 dimensions a ldmatrix.x4.trans.
        for (int kk = 0; kk < Bc / 16; kk++)
        {
            var a0 = Split(s[2 * kk, 0], s[2 * kk, 1]);
            var a1 = Split(s[2 * kk, 2], s[2 * kk, 3]);
            var a2 = Split(s[2 * kk + 1, 0], s[2 * kk + 1, 1]);
            var a3 = Split(s[2 * kk + 1, 2], s[2 * kk + 1, 3]);
            string aHi = $"{{{a0.Hi}, {a1.Hi}, {a2.Hi}, {a3.Hi}}}", aLo = $"{{{a0.Lo}, {a1.Lo}, {a2.Lo}, {a3.Lo}}}";
            for (int np = 0; np < ntO / 2; np++)
            {
                string v0 = b.R(), v1 = b.R(), v2 = b.R(), v3 = b.R();
                b.L($"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{{v0}, {v1}, {v2}, {v3}}}, [{vAddr}+{kk * 16 * stride + np * 32}];");
                for (int half = 0; half < 2; half++)
                {
                    int j = 2 * np + half;
                    string dd = $"{{{acc[j, 0]}, {acc[j, 1]}, {acc[j, 2]}, {acc[j, 3]}}}", bb = half == 0 ? $"{{{v0}, {v1}}}" : $"{{{v2}, {v3}}}";
                    b.L($"{kind.Mma} {dd}, {aHi}, {bb}, {dd};");
                    b.L($"{kind.Mma} {dd}, {aLo}, {bb}, {dd};");
                }
            }
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {kt}, {kt}, {Bc};");
        b.L("bra TILE;");
        b.Label("TILE_END");

        // O = acc / l, value by value as the type (value e of n-tile j: dimension 8j + 2·(lane mod 4) + e mod 2).
        var natural = f.Natural;
        foreach (var (row, l, es) in new[] { (rowA, lA, new[] { 0, 1 }), (rowB, lB, new[] { 2, 3 }) })
        {
            string inv = b.F(), pAny = b.P(), oRow = b.R();
            b.L($"setp.gt.f32 {pAny}, {l}, 0f00000000;");
            b.L($"div.rn.f32 {inv}, 0f3F800000, {l};");
            b.L($"selp.f32 {inv}, {inv}, 0f00000000, {pAny};");
            b.L($"mad.lo.u32 {oRow}, {row.I}, {d}, {qHead};");
            for (int j = 0; j < ntO; j++)
            {
                foreach (int e in es)
                {
                    string dim = b.R(), pDim = b.P(), val = b.F(), idx = b.R(), oAddr = b.RD(), code = b.R(), hh = b.H();
                    string skip = $"OUT{(e < 2 ? "A" : "B")}{j}_{e}";
                    b.L($"add.u32 {dim}, {key0}, {8 * j + (e & 1)};");
                    b.L($"setp.ge.u32 {pDim}, {dim}, {d};");
                    b.L($"@{pDim} bra {skip};");
                    b.L($"@!{row.Valid} bra {skip};");
                    b.L($"mul.rn.f32 {val}, {acc[j, e]}, {inv};");
                    b.L($"add.u32 {idx}, {oRow}, {dim};");
                    string stored = natural == F ? val : Codec.Convert(b, F, val, natural, f);
                    b.L($"mov.b32 {code}, 0;");
                    Codec.Pack(b, f, stored, natural, 0, code);
                    b.L($"mul.wide.u32 {oAddr}, {idx}, {f.Bits / 8};");
                    b.L($"add.u64 {oAddr}, {o}, {oAddr};");
                    b.L($"cvt.u16.u32 {hh}, {code};");
                    b.L($"st.global.b{f.Bits} [{oAddr}], {hh};");
                    b.Label(skip);
                }
            }
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_o, .param .u64 p_lim, .param .u64 p_start, .param .u64 p_range, .param .u32 p_lq, .param .u32 p_lk, .param .u32 p_d, .param .f32 p_scale",
            AttentionMmaShared(dCap));
    }
}
