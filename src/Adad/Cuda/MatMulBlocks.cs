// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The block-scaled tiled product: C = A·Bᵀ for A (m × k) and B (n × k) both Q8_0 or both Q4_0, k in whole blocks of 32
// along each row (a linear layer's weights stored n × k). C_ij = Σ_blocks dA·dB·(Σ_32 qA·qB): every 32-term dot an
// exact integer, the block scales applied in float32 — gemv's semantics on a tile, nothing decoded to a float per value.
//   CPU: A's and B's blocks unpacked once into int8 rows (Q4_0: q − 8) and per-4-byte scales (the scale of the block
//        each int lane of an s8 dot covers, so any vector width works); tasks of 4 rows, a 4 × 2 register tile of float
//        vectors, per vector of k one s8 dot, converted, times sA·sB, added. Where the runtime has the s8 dot lane.
//   CUDA: the s8 tensor instruction at k = 32 (one per block per 16 × 8 tile): a block of 128 threads per 64 × 64 of C, a
//        warp 32 × 32; per k-block A's and B's 64 blocks staged into shared memory by 16-bit loads (a 34-byte Q8_0
//        block is 2-byte aligned only; Q4_0's nibbles widened to int8 − 8 on the way), rows 48 bytes apart (the
//        fragments' 32-bit loads free of bank conflicts), the scales in f32; each tile's s32 result converted and added
//        as acc += d·sA_row·sB_col.
// C comes out as float32 lanes; the callers encode it to the type by blocks.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    public const int BlocksThreads = 128, BlocksTile = 64, BlocksRow = 48, BlocksStage = 4;

    /// <summary>C (f32, m × n) = A·Bᵀ for Q8_0 / Q4_0 rows of k (whole blocks) by the s8 tensor instruction at k = 32. Four
    /// k-blocks are staged at a time: a row's four blocks are contiguous in memory, read by consecutive lanes 16 bits each.</summary>
    public static string MatMulBlocks(Gpu gpu, Fmt f, out string entry)
    {
        entry = $"matmul_blocks_{f.Name}";
        bool q8 = Routes.Bytewise(f);
        int bb = f.BlockBytes, halves = bb / 2, tile = BlocksTile, rowB = BlocksRow, ks = BlocksStage, T = BlocksThreads;
        int rowHalves = ks * halves, slab = tile * rowB;                       // a row's halves per stage; one k-block's tile
        int aOff = 0, bOff = ks * slab, saOff = 2 * ks * slab, sbOff = saOff + ks * tile * 4, shared = sbOff + ks * tile * 4;
        var b = new PtxBuilder(gpu);
        string tid = b.R(), bx = b.R(), by = b.R(), sm = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"mov.u32 {sm}, smem;");
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), M = Param32(b, "p_m"), N = Param32(b, "p_n"), blocks = Param32(b, "p_blocks");
        string m0 = b.R(), n0 = b.R(), rowStride = b.RD();
        b.L($"mul.lo.u32 {m0}, {by}, {tile};");
        b.L($"mul.lo.u32 {n0}, {bx}, {tile};");
        b.L($"mul.wide.u32 {rowStride}, {blocks}, {bb};");
        // The warp's 32 × 32 and the lane's place in the fragments.
        string warp = b.R(), lane = b.R(), gid = b.R(), tig = b.R(), wm = b.R(), wn = b.R();
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"shr.u32 {gid}, {lane}, 2;");
        b.L($"and.b32 {tig}, {lane}, 3;");
        b.L($"shr.u32 {wm}, {warp}, 1;");
        b.L($"mul.lo.u32 {wm}, {wm}, 32;");
        b.L($"and.b32 {wn}, {warp}, 1;");
        b.L($"mul.lo.u32 {wn}, {wn}, 32;");
        var acc = new string[2, 4, 4];
        for (int mt = 0; mt < 2; mt++) for (int nt = 0; nt < 4; nt++) for (int e = 0; e < 4; e++) { acc[mt, nt, e] = b.F(); b.L($"mov.f32 {acc[mt, nt, e]}, 0f00000000;"); }
        // Fragment byte offsets in shared memory (row · 48 + tig · 4), fixed across k-blocks.
        string aFrag0 = b.R(), bFrag0 = b.R(), sa0 = b.R(), sb0 = b.R(), t4 = b.R();
        b.L($"shl.b32 {t4}, {tig}, 2;");
        b.L($"add.u32 {aFrag0}, {wm}, {gid};");
        b.L($"mad.lo.u32 {aFrag0}, {aFrag0}, {rowB}, {t4};");
        b.L($"add.u32 {aFrag0}, {aFrag0}, {sm};");
        b.L($"add.u32 {bFrag0}, {wn}, {gid};");
        b.L($"mad.lo.u32 {bFrag0}, {bFrag0}, {rowB}, {t4};");
        b.L($"add.u32 {bFrag0}, {bFrag0}, {sm};");
        b.L($"add.u32 {sa0}, {wm}, {gid};");
        b.L($"mad.lo.u32 {sa0}, {sa0}, 4, {sm};");                             // sA[wm + gid]
        b.L($"shl.b32 {sb0}, {tig}, 1;");
        b.L($"add.u32 {sb0}, {sb0}, {wn};");
        b.L($"mad.lo.u32 {sb0}, {sb0}, 4, {sm};");                             // sB[wn + tig·2]

        string kb = b.R(), pK = b.P();
        b.L($"mov.u32 {kb}, 0;");
        b.Label("KB");
        b.L($"setp.ge.u32 {pK}, {kb}, {blocks};");
        b.L($"@{pK} bra KB_END;");
        {
            // Stage k-blocks kb .. kb + 3 of A's and B's tiles: element i of the 64 × rowHalves halves goes to thread i % T.
            string kbOff = b.RD();
            b.L($"mul.wide.u32 {kbOff}, {kb}, {bb};");
            foreach (var (gPtr, limit, rowBase0, tileOff, scaleOff, tag) in new[] { (pa, M, m0, aOff, saOff, "A"), (pb, N, n0, bOff, sbOff, "B") })
            {
                for (int it = 0; it < tile * rowHalves / T; it++)
                {
                    string idx = b.R(), row = b.R(), hw = b.R(), blk = b.R(), h = b.R(), gRow = b.R(), kAt = b.R(), valid = b.P(), kValid = b.P(), addr = b.RD(), hv = b.R();
                    b.L($"add.u32 {idx}, {tid}, {it * T};");
                    b.L($"div.u32 {row}, {idx}, {rowHalves};");
                    b.L($"rem.u32 {hw}, {idx}, {rowHalves};");
                    b.L($"div.u32 {blk}, {hw}, {halves};");
                    b.L($"rem.u32 {h}, {hw}, {halves};");
                    b.L($"add.u32 {gRow}, {rowBase0}, {row};");
                    b.L($"add.u32 {kAt}, {kb}, {blk};");
                    b.L($"setp.lt.u32 {valid}, {gRow}, {limit};");
                    b.L($"setp.lt.and.u32 {kValid}, {kAt}, {blocks}, {valid};");
                    b.L($"mul.wide.u32 {addr}, {gRow}, 1;");
                    b.L($"mul.lo.u64 {addr}, {addr}, {rowStride};");
                    b.L($"add.u64 {addr}, {addr}, {kbOff};");
                    b.L($"add.u64 {addr}, {addr}, {gPtr};");
                    string hw2 = b.RD();
                    b.L($"mul.wide.u32 {hw2}, {hw}, 2;");
                    b.L($"add.u64 {addr}, {addr}, {hw2};");
                    b.L($"mov.u32 {hv}, 0;");
                    b.L($"@{kValid} ld.global.u16 {hv}, [{addr}];");
                    // Where it goes: the scale (h = 0) into sS[blk][row], values into the k-block's tile row.
                    string isScale = b.P(), sAddr = b.R(), dRow = b.R();
                    b.L($"setp.eq.u32 {isScale}, {h}, 0;");
                    b.L($"@!{isScale} bra VAL_{tag}_{it};");
                    {
                        string hh = b.H(), d = b.F();
                        b.L($"cvt.u16.u32 {hh}, {hv};");
                        b.L($"cvt.f32.f16 {d}, {hh};");
                        b.L($"mad.lo.u32 {sAddr}, {blk}, {tile}, {row};");
                        b.L($"mad.lo.u32 {sAddr}, {sAddr}, 4, {sm};");
                        b.L($"st.shared.f32 [{sAddr}+{scaleOff}], {d};");
                        b.L($"bra NEXT_{tag}_{it};");
                    }

                    b.Label($"VAL_{tag}_{it}");
                    b.L($"mad.lo.u32 {dRow}, {blk}, {tile}, {row};");
                    b.L($"mad.lo.u32 {dRow}, {dRow}, {rowB}, {sm};");                 // the k-block's tile, the row
                    string j = b.R();
                    b.L($"sub.u32 {j}, {h}, 1;");
                    b.L($"shl.b32 {j}, {j}, 1;");                                       // 2 · (h − 1): the byte in the block's values
                    b.L($"add.u32 {dRow}, {dRow}, {j};");
                    if (q8)
                    {
                        string hh = b.H();
                        b.L($"cvt.u16.u32 {hh}, {hv};");
                        b.L($"st.shared.u16 [{dRow}+{tileOff}], {hh};");
                    }
                    else
                    {
                        // Two bytes of nibbles: byte j's low nibble is value j, its high nibble value j + 16; each − 8 as int8.
                        string lo0 = b.R(), lo1 = b.R(), hi0 = b.R(), hi1 = b.R(), lo = b.R(), hi = b.R(), h1 = b.H(), h2 = b.H();
                        b.L($"bfe.u32 {lo0}, {hv}, 0, 4;");
                        b.L($"bfe.u32 {hi0}, {hv}, 4, 4;");
                        b.L($"bfe.u32 {lo1}, {hv}, 8, 4;");
                        b.L($"bfe.u32 {hi1}, {hv}, 12, 4;");
                        foreach (var r in new[] { lo0, hi0, lo1, hi1 }) { b.L($"sub.u32 {r}, {r}, 8;"); b.L($"and.b32 {r}, {r}, 255;"); }
                        b.L($"shl.b32 {lo1}, {lo1}, 8;");
                        b.L($"or.b32 {lo}, {lo0}, {lo1};");
                        b.L($"shl.b32 {hi1}, {hi1}, 8;");
                        b.L($"or.b32 {hi}, {hi0}, {hi1};");
                        b.L($"cvt.u16.u32 {h1}, {lo};");
                        b.L($"cvt.u16.u32 {h2}, {hi};");
                        b.L($"st.shared.u16 [{dRow}+{tileOff}], {h1};");
                        b.L($"st.shared.u16 [{dRow}+{tileOff + 16}], {h2};");
                    }

                    b.Label($"NEXT_{tag}_{it}");
                }
            }

            b.L("bar.sync 0;");
            // Per staged k-block, the warp's 2 × 4 tiles: one m16n8k32 each, its s32 result scaled into the f32 accumulators.
            for (int kk = 0; kk < ks; kk++)
            {
                string kAt = b.R(), pSkip = b.P();
                b.L($"add.u32 {kAt}, {kb}, {kk};");
                b.L($"setp.ge.u32 {pSkip}, {kAt}, {blocks};");
                b.L($"@{pSkip} bra COMPUTED;");
                var sbv = new string[4, 2];
                for (int nt = 0; nt < 4; nt++)
                {
                    for (int e = 0; e < 2; e++)
                    {
                        sbv[nt, e] = b.F();
                        b.L($"ld.shared.f32 {sbv[nt, e]}, [{sb0}+{sbOff + (kk * tile + nt * 8 + e) * 4}];");
                    }
                }

                for (int mt = 0; mt < 2; mt++)
                {
                    string a0 = b.R(), a1 = b.R(), a2 = b.R(), a3 = b.R(), s0 = b.F(), s1 = b.F();
                    int ro = aOff + kk * slab + mt * 16 * rowB;
                    b.L($"ld.shared.b32 {a0}, [{aFrag0}+{ro}];");
                    b.L($"ld.shared.b32 {a1}, [{aFrag0}+{ro + 8 * rowB}];");
                    b.L($"ld.shared.b32 {a2}, [{aFrag0}+{ro + 16}];");
                    b.L($"ld.shared.b32 {a3}, [{aFrag0}+{ro + 8 * rowB + 16}];");
                    b.L($"ld.shared.f32 {s0}, [{sa0}+{saOff + (kk * tile + mt * 16) * 4}];");
                    b.L($"ld.shared.f32 {s1}, [{sa0}+{saOff + (kk * tile + mt * 16 + 8) * 4}];");
                    for (int nt = 0; nt < 4; nt++)
                    {
                        string b0 = b.R(), b1 = b.R(), d0 = b.R(), d1 = b.R(), d2 = b.R(), d3 = b.R();
                        int co = bOff + kk * slab + nt * 8 * rowB;
                        b.L($"ld.shared.b32 {b0}, [{bFrag0}+{co}];");
                        b.L($"ld.shared.b32 {b1}, [{bFrag0}+{co + 16}];");
                        b.L($"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{0, 0, 0, 0}};");
                        var d = new[] { d0, d1, d2, d3 };
                        for (int e = 0; e < 4; e++)
                        {
                            string df = b.F(), sc = b.F();
                            b.L($"cvt.rn.f32.s32 {df}, {d[e]};");
                            b.L($"mul.rn.f32 {sc}, {(e < 2 ? s0 : s1)}, {sbv[nt, e & 1]};");
                            b.L($"fma.rn.f32 {acc[mt, nt, e]}, {df}, {sc}, {acc[mt, nt, e]};");
                        }
                    }
                }
            }

            b.Label("COMPUTED");
            b.L("bar.sync 0;");
            b.L($"add.u32 {kb}, {kb}, {ks};");
            b.L("bra KB;");
        }

        b.Label("KB_END");
        // The accumulators to C (inside m × n only).
        for (int mt = 0; mt < 2; mt++)
        {
            for (int nt = 0; nt < 4; nt++)
            {
                for (int e = 0; e < 4; e++)
                {
                    string row = b.R(), col = b.R(), pr = b.P(), pcl = b.P(), idx = b.R(), addr = b.RD();
                    b.L($"add.u32 {row}, {m0}, {wm};");
                    b.L($"add.u32 {row}, {row}, {gid};");
                    b.L($"add.u32 {row}, {row}, {mt * 16 + (e < 2 ? 0 : 8)};");
                    b.L($"shl.b32 {col}, {tig}, 1;");
                    b.L($"add.u32 {col}, {col}, {n0};");
                    b.L($"add.u32 {col}, {col}, {wn};");
                    b.L($"add.u32 {col}, {col}, {nt * 8 + (e & 1)};");
                    b.L($"setp.lt.u32 {pr}, {row}, {M};");
                    b.L($"setp.lt.and.u32 {pcl}, {col}, {N}, {pr};");
                    b.L($"mad.lo.u32 {idx}, {row}, {N}, {col};");
                    b.L($"mul.wide.u32 {addr}, {idx}, 4;");
                    b.L($"add.u64 {addr}, {pc}, {addr};");
                    b.L($"@{pcl} st.global.f32 [{addr}], {acc[mt, nt, e]};");
                }
            }
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_blocks", shared);
    }
}

public sealed partial class GpuOps
{
    /// <summary>Whether the device has the block product (the s8 tensor instruction at k = 32, as the JIT accepts it).</summary>
    public bool BlockProductAvailable => MmaSupported(Ptx.Kind.S8K32);

    /// <summary>C (float32 lanes, m × n) = A·Bᵀ for A (m × k), B (n × k) both Q8_0 or both Q4_0, k a multiple of 32; returns the lane's name.</summary>
    public string MatMulBlocks<TN>(ulong a, ulong b, ulong cLanes, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!Routes.Takes(f) || k % 32 != 0) throw new NotSupportedException($"{f.Name}: the block product takes Q8_0 or Q4_0 with k in whole blocks");
        if (!BlockProductAvailable) throw new NotSupportedException("the block product needs the s8 tensor instruction at k = 32");
        var fn = gpu.Cached($"matmul_blocks:{f.Name}", (out string e) => Ptx.MatMulBlocks(gpu, f, out e));
        Gpu.Launch(fn, Grid(n, Ptx.BlocksTile), Grid(m, Ptx.BlocksTile), Ptx.BlocksThreads, 1, 0, a, b, cLanes, Gpu.U(m), Gpu.U(n), Gpu.U(k / 32));
        return "s8 mma blocks";
    }

    /// <summary>The gate's override: true takes the block product wherever the view allows it, false never, null as measured.</summary>
    internal bool? BlockProducts;

    private readonly Dictionary<string, bool> _blocksFaster = [];

    private bool BlockView<TN>(ProductView v) where TN : struct, IAdad<TN> =>
        Routes.Route(Fmt.Of<TN>()) && v.Batch == 1 && !v.TransA && v.TransB && v.K % 32 == 0 && BlockProductAvailable;

    /// <summary>The block product for the view (A·Bᵀ), C encoded to the type by blocks.</summary>
    public string MatMulViewBlocks<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        ulong lanes = gpu.Rent((long)v.M * v.N * 4);
        try
        {
            string lane = MatMulBlocks<TN>(a, b, lanes, v.M, v.N, v.K);
            Back(Fmt.Of<TN>(), Lane.F32, lanes, c, (long)v.M * v.N);
            return lane;
        }
        finally { gpu.Return(lanes); }
    }

    // The block product or the strided route, whichever this device measures faster at the shape and format (once, as
    // Measure.Faster settles it).
    private bool UseBlocks<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        if (BlockProducts is bool forced) return forced;
        string key = $"{Fmt.Of<TN>().Name}:{v.M}x{v.N}x{v.K}";
        if (_blocksFaster.TryGetValue(key, out bool blocks)) return blocks;
        ulong scratch = gpu.Rent(Fmt.Of<TN>().Words(v.ValuesC) * 4 + 64);
        try
        {
            var (tb, tr) = Measure.Faster(() => MatMulViewBlocks<TN>(v, a, b, scratch), () => MatMulViewRoute<TN>(v, a, b, scratch), Gpu.Synchronize);
            blocks = _blocksFaster[key] = tb <= tr;
            Note($"product {key} A·Bᵀ: block-scaled {tb:0.000} ms, decoded {tr:0.000} ms: {(blocks ? "block-scaled" : "decoded")}");
            return blocks;
        }
        finally { gpu.Return(scratch); }
    }
}
