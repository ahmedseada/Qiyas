// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on CUDA (Abstraction/Cache.cs, ICache; GpuOps.Cache.cs), the position (an s32 at the device) read by
// each kernel where it is. The write's kernels: each source row's scale (a CTA a row, the block reduction), stored as the
// type at its row (a value of 8 bits or more by its own store; a block format's by re-encoding the blocks the write
// touches, their other values kept), then the codes, a thread a stored word of the cache (a row is whole 32-bit words).
// Attention over the caches is the decode kernel's (Ptx.Attention.cs, DecodeCache). The mask is a StoreEach kernel.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    // The rows of each head a write of `steps` rows at the device's position fills: [w0, w1) = [max(p, 0), min(p + steps,
    // cap)) (s32 registers), and whether there are any.
    private static (string P, string W0, string W1, string Any) WrittenRows(PtxBuilder b, string pos, string steps, string cap)
    {
        string p = b.R(), w0 = b.R(), w1 = b.R(), any = b.P();
        b.L($"ld.global.s32 {p}, [{pos}];");
        b.L($"max.s32 {w0}, {p}, 0;");
        b.L($"add.s32 {w1}, {p}, {steps};");
        b.L($"min.s32 {w1}, {w1}, {cap};");
        b.L($"setp.gt.s32 {any}, {w1}, {w0};");
        return (p, w0, w1, any);
    }

    /// <summary>Each source row's scale max |x| / max in the lane, a CTA a row (row = head·steps + step). direct: stored as f at
    /// head·cap + p + step (a row past the capacity not written; f a type of 8 bits or more); else a lane value at ps[row].
    /// Params: x (f), ps, pos, steps, cap, d, max (lane).</summary>
    public static string CacheScales(Gpu gpu, Fmt f, Lane lane, bool direct, out string entry, Fmt? source = null)
    {
        var fx = source ?? f;                                                   // x's format: the type's, or a composition's lane storage
        entry = $"cache_scales_{f.Name}_{lane.Name}{(direct ? "" : "_lanes")}{(source is null ? "" : "_from_" + fx.Name)}";
        const int Threads = 256;
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string px = Param64(b, "p_x"), ps = Param64(b, "p_s"), pos = Param64(b, "p_pos");
        string steps = Param32(b, "p_steps"), cap = Param32(b, "p_cap"), d = Param32(b, "p_d");
        string max = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {max}, [p_max];");
        string amax = lane.NewReg(b), e = b.R(), pEnd = b.P(), first = b.R();
        b.L($"mov.{lane.Reg} {amax}, {lane.Zero};");
        b.L($"mul.lo.u32 {first}, {row}, {d};");
        b.L($"mov.u32 {e}, {tid};");
        b.Label("CS_LOOP");
        b.L($"setp.ge.u32 {pEnd}, {e}, {d};");
        b.L($"@{pEnd} bra CS_END;");
        {
            string idx = b.R(), ax = lane.NewReg(b);
            b.L($"add.u32 {idx}, {first}, {e};");
            string x = Codec.UnpackAt(b, fx, px, idx, lane);
            b.L($"abs.{lane.Op} {ax}, {x};");
            b.L($"max.{lane.Op} {amax}, {amax}, {ax};");
        }

        b.L($"add.u32 {e}, {e}, {Threads};");
        b.L("bra CS_LOOP;");
        b.Label("CS_END");
        string top = BlockReduce(b, lane, tid, amax, "max", Threads);
        string scale = LaneOp(b, lane, "div", top, max);
        string rest = b.P();
        b.L($"setp.ne.u32 {rest}, {tid}, 0;");
        b.L($"@{rest} bra DONE;");
        if (direct)
        {
            string h = b.R(), t = b.R(), p = b.R(), at = b.R(), si = b.R(), past = b.P();
            b.L($"div.u32 {h}, {row}, {steps};");
            b.L($"mul.lo.u32 {t}, {h}, {steps};");
            b.L($"sub.u32 {t}, {row}, {t};");
            b.L($"ld.global.s32 {p}, [{pos}];");
            b.L($"add.s32 {at}, {p}, {t};");
            b.L($"setp.ge.u32 {past}, {at}, {cap};");                     // below 0 too (unsigned)
            b.L($"@{past} bra DONE;");
            b.L($"mad.lo.u32 {si}, {h}, {cap}, {at};");
            StoreAnyValue(b, f, lane, scale, ps, si);
        }
        else
        {
            string a = b.RD();
            b.L($"mul.wide.u32 {a}, {row}, {lane.Size};");
            b.L($"add.u64 {a}, {ps}, {a};");
            b.L($"st.global.{lane.Mem} [{a}], {scale};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_s, .param .u64 p_pos, .param .u32 p_steps, .param .u32 p_cap, .param .u32 p_d, .param .{lane.Op} p_max", ReduceBytes);
    }

    /// <summary>Rows of a block format written where rows share blocks (a block-format type's row scales, unit 1; a block
    /// format cache's rows of `unit` values not whole blocks): each block of the destination (fb, n values: heads·cap rows
    /// of unit values) that holds a written value re-encoded, a warp a block, from the written rows' values (fx at
    /// px[(head·steps + step)·unit + column], read as float32) and its other values as they are stored; a block the write
    /// does not touch is left as it is. Params: x, destination, pos, n, cap, steps, unit.</summary>
    public static string CacheBlocks(Gpu gpu, Fmt fb, Fmt fx, out string entry)
    {
        entry = $"cache_blocks_{fb.Name}_from_{fx.Name}";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), ps = Param64(b, "p_s"), pos = Param64(b, "p_pos");
        string n = Param32(b, "p_n"), cap = Param32(b, "p_cap"), steps = Param32(b, "p_steps"), unit = Param32(b, "p_unit");
        var w = BlockWarp(b, fb, n);
        var (p, w0, w1, any) = WrittenRows(b, pos, steps, cap);
        b.L($"@!{any} bra DONE;");
        // The written values of every head: x mod (cap·unit) in [w0·unit, w1·unit). The first one at or past the block's
        // first value a: the block is touched when it is below the block's end.
        string a = b.R(), end = b.R(), r = b.R(), x0 = b.R(), t = b.R(), period = b.R(), lo = b.R(), hi = b.R(), below = b.P(), inside = b.P(), untouched = b.P();
        b.L($"mul.lo.u32 {period}, {cap}, {unit};");
        b.L($"mul.lo.u32 {lo}, {w0}, {unit};");
        b.L($"mul.lo.u32 {hi}, {w1}, {unit};");
        b.L($"mul.lo.u32 {a}, {w.Blk}, {fb.BlockValues};");
        b.L($"add.u32 {end}, {a}, {fb.BlockValues};");
        b.L($"min.u32 {end}, {end}, {n};");
        b.L($"rem.u32 {r}, {a}, {period};");
        b.L($"setp.lt.u32 {below}, {r}, {lo};");
        b.L($"setp.lt.u32 {inside}, {r}, {hi};");
        b.L($"sub.u32 {t}, {period}, {r};");
        b.L($"add.u32 {t}, {t}, {lo};");
        b.L($"add.u32 {t}, {t}, {a};");                                    // the next head's first written value
        b.L($"selp.u32 {x0}, {a}, {t}, {inside};");
        b.L($"sub.u32 {t}, {lo}, {r};");
        b.L($"add.u32 {t}, {t}, {a};");
        b.L($"selp.u32 {x0}, {t}, {x0}, {below};");
        b.L($"setp.ge.u32 {untouched}, {x0}, {end};");
        b.L($"@{untouched} bra DONE;");
        EncodeBlock(b, fb, ps, w, n, at =>
        {
            string rowAt = b.R(), col = b.R(), h = b.R(), rr = b.R(), idx = b.R(), ge = b.P(), lt = b.P(), v = b.F();
            b.L($"div.u32 {rowAt}, {at}, {unit};");
            b.L($"mul.lo.u32 {col}, {rowAt}, {unit};");
            b.L($"sub.u32 {col}, {at}, {col};");
            b.L($"div.u32 {h}, {rowAt}, {cap};");
            b.L($"mul.lo.u32 {rr}, {h}, {cap};");
            b.L($"sub.u32 {rr}, {rowAt}, {rr};");
            b.L($"setp.ge.u32 {ge}, {rr}, {w0};");
            b.L($"setp.lt.u32 {lt}, {rr}, {w1};");
            b.L($"and.pred {ge}, {ge}, {lt};");
            b.L($"sub.u32 {idx}, {rr}, {p};");                             // the step (p may be below 0: modular)
            b.L($"mad.lo.u32 {idx}, {h}, {steps}, {idx};");
            b.L($"mad.lo.u32 {idx}, {idx}, {unit}, {col};");
            b.L($"selp.u32 {idx}, {idx}, 0, {ge};");
            string fresh = Codec.UnpackAt(b, fx, px, idx, Lane.F32), old = Codec.UnpackAt(b, fb, ps, at, Lane.F32);
            b.L($"selp.f32 {v}, {fresh}, {old}, {ge};");
            return v;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_s, .param .u64 p_pos, .param .u32 p_n, .param .u32 p_cap, .param .u32 p_steps, .param .u32 p_unit");
    }

    /// <summary>Rows of whole blocks into a block format cache at the device's position, a warp a block of the written rows
    /// (n = heads·steps·d source values of fx, read as float32; bpr blocks a row): source block row·bpr + j is the cache's
    /// block (head·cap + p + step)·bpr + j, encoded whole; a row past the capacity not written. Params: x, cache, pos, n,
    /// steps, cap, bpr.</summary>
    public static string CacheRowsBlocks(Gpu gpu, Fmt fb, Fmt fx, out string entry)
    {
        entry = $"cache_rows_blocks_{fb.Name}_from_{fx.Name}";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), pc = Param64(b, "p_c"), pos = Param64(b, "p_pos");
        string n = Param32(b, "p_n"), steps = Param32(b, "p_steps"), cap = Param32(b, "p_cap"), bpr = Param32(b, "p_bpr");
        var w = BlockWarp(b, fb, n);
        string row = b.R(), j = b.R(), h = b.R(), t = b.R(), p = b.R(), at = b.R(), dest = b.R(), past = b.P(), off = b.RD(), outp = b.RD();
        b.L($"div.u32 {row}, {w.Blk}, {bpr};");
        b.L($"mul.lo.u32 {j}, {row}, {bpr};");
        b.L($"sub.u32 {j}, {w.Blk}, {j};");
        b.L($"div.u32 {h}, {row}, {steps};");
        b.L($"mul.lo.u32 {t}, {h}, {steps};");
        b.L($"sub.u32 {t}, {row}, {t};");
        b.L($"ld.global.s32 {p}, [{pos}];");
        b.L($"add.s32 {at}, {p}, {t};");
        b.L($"setp.ge.u32 {past}, {at}, {cap};");                         // below 0 too (unsigned)
        b.L($"@{past} bra DONE;");
        b.L($"mad.lo.u32 {dest}, {h}, {cap}, {at};");
        b.L($"mad.lo.u32 {dest}, {dest}, {bpr}, {j};");
        // The encoder addresses its block from the base: the base moved by the cache's block less the source's.
        b.L($"sub.s32 {dest}, {dest}, {w.Blk};");
        b.L($"cvt.s64.s32 {off}, {dest};");
        b.L($"mul.lo.s64 {off}, {off}, {fb.BlockBytes};");
        b.L($"add.s64 {outp}, {pc}, {off};");
        EncodeBlock(b, fb, outp, w, n, idx => Codec.UnpackAt(b, fx, px, idx, Lane.F32));
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_c, .param .u64 p_pos, .param .u32 p_n, .param .u32 p_steps, .param .u32 p_cap, .param .u32 p_bpr");
    }

    /// <summary>The source's rows into the cache at the device's position, a thread a stored word of fc (a row: wpr words):
    /// each value of f read into the lane and rounded to fc once (Convert's rule: the lane's value, to odd at float32 first
    /// where Fmt.ConvertPlan says); scaled: x / scale by the row's stored scale (f at head·cap + row; 0 where the scale is
    /// 0), an integer code rounded to the nearest (ties to even), held to [min, max]. Params: x, cache, scales, pos, rows
    /// (heads·steps), steps, cap, d, wpr, min, max (lane).</summary>
    public static string CacheWrite(Gpu gpu, Fmt f, Fmt fc, Lane lane, bool scaled, out string entry, Fmt? source = null)
    {
        var fx = source ?? f;                                                   // x's format: the type's, or a composition's lane storage
        entry = $"cache_write_{f.Name}_{fc.Name}_{lane.Name}{(scaled ? "_scaled" : "")}{(source is null ? "" : "_from_" + fx.Name)}";
        var b = new PtxBuilder();
        string word = GlobalIndex(b);
        string px = Param64(b, "p_x"), pc = Param64(b, "p_c"), ps = Param64(b, "p_s"), pos = Param64(b, "p_pos");
        string rows = Param32(b, "p_rows"), steps = Param32(b, "p_steps"), cap = Param32(b, "p_cap"), d = Param32(b, "p_d"), wpr = Param32(b, "p_wpr");
        string min = lane.NewReg(b), max = lane.NewReg(b);
        b.L($"ld.param.{lane.Op} {min}, [p_min];");
        b.L($"ld.param.{lane.Op} {max}, [p_max];");
        bool toOdd = Fmt.ConvertPlan(lane.Wide ? new Fmt("float64", 64, true, true) : new Fmt("float32", 32, true, true), fc).ToOdd;
        string row = b.R(), wc = b.R(), t = b.R(), h = b.R(), p = b.R(), at = b.R(), crow = b.R(), past = b.P();
        b.L($"div.u32 {row}, {word}, {wpr};");
        b.L($"mul.lo.u32 {t}, {row}, {wpr};");
        b.L($"sub.u32 {wc}, {word}, {t};");
        b.L($"setp.ge.u32 {past}, {row}, {rows};");
        b.L($"@{past} bra DONE;");
        b.L($"div.u32 {h}, {row}, {steps};");
        b.L($"mul.lo.u32 {t}, {h}, {steps};");
        b.L($"sub.u32 {t}, {row}, {t};");
        b.L($"ld.global.s32 {p}, [{pos}];");
        b.L($"add.s32 {at}, {p}, {t};");
        b.L($"setp.ge.u32 {past}, {at}, {cap};");                         // below 0 too (unsigned)
        b.L($"@{past} bra DONE;");
        b.L($"mad.lo.u32 {crow}, {h}, {cap}, {at};");
        string scale = "", positive = b.P();
        if (scaled)
        {
            scale = Codec.UnpackAt(b, f, ps, crow, lane);
            b.L($"setp.gt.{lane.Op} {positive}, {scale}, {lane.Zero};");
        }

        string code = fc.WordBits == 64 ? b.RD() : b.R(), first = b.R();
        if (fc.WordBits == 32) b.L($"mov.b32 {code}, 0;");
        b.L($"mul.lo.u32 {first}, {row}, {d};");
        b.L($"mad.lo.u32 {first}, {wc}, {fc.Per}, {first};");
        for (int i = 0; i < fc.Per; i++)
        {
            string idx = b.R();
            b.L($"add.u32 {idx}, {first}, {i};");
            string v = Codec.UnpackAt(b, fx, px, idx, lane);
            if (scaled)
            {
                string q = LaneOp(b, lane, "div", v, scale), kept = lane.NewReg(b);
                b.L($"selp.{lane.Op} {kept}, {q}, {lane.Zero}, {positive};");
                v = fc.IsFloat ? kept : RoundLane(b, lane, kept);
                v = HeldLane(b, lane, v, min, max);
            }

            if (toOdd) Codec.Pack(b, fc, ToOddF32(b, lane, v), Lane.F32, i, code);
            else Codec.Pack(b, fc, v, lane, i, code);
        }

        string wi = b.R(), addr = b.RD();
        b.L($"mad.lo.u32 {wi}, {crow}, {wpr}, {wc};");
        b.L($"mul.wide.u32 {addr}, {wi}, {fc.WordBits / 8};");
        b.L($"add.u64 {addr}, {pc}, {addr};");
        b.L($"st.global.{(fc.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_c, .param .u64 p_s, .param .u64 p_pos, .param .u32 p_rows, .param .u32 p_steps, .param .u32 p_cap, .param .u32 p_d, .param .u32 p_wpr, .param .{lane.Op} p_min, .param .{lane.Op} p_max");
    }

    /// <summary>The mask of a decoder's steps (n = rows·cap values of f): 0 where p + i − win &lt; j ≤ p + i (win 0: every
    /// j ≤ p + i) for value i·cap + j, else `low`. Params: y, pos, n, cap, win, low (lane).</summary>
    public static string DecoderMask(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"decoder_mask_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string py = Param64(b, "p_y"), pos = Param64(b, "p_pos"), n = Param32(b, "p_n"), cap = Param32(b, "p_cap"), win = Param32(b, "p_win");
        string low = lane.NewReg(b), p = b.R(), noWin = b.P();
        b.L($"ld.param.{lane.Op} {low}, [p_low];");
        b.L($"ld.global.s32 {p}, [{pos}];");
        b.L($"setp.eq.u32 {noWin}, {win}, 0;");
        StoreEach(b, f, lane, py, n, at =>
        {
            string i = b.R(), j = b.R(), hi = b.R(), lo = b.R(), seen = b.P(), past = b.P(), v = lane.NewReg(b);
            b.L($"div.u32 {i}, {at}, {cap};");
            b.L($"rem.u32 {j}, {at}, {cap};");
            b.L($"add.s32 {hi}, {p}, {i};");
            b.L($"sub.s32 {lo}, {hi}, {win};");
            b.L($"setp.le.s32 {seen}, {j}, {hi};");
            b.L($"setp.gt.s32 {past}, {j}, {lo};");
            b.L($"or.pred {past}, {past}, {noWin};");
            b.L($"and.pred {seen}, {seen}, {past};");
            b.L($"selp.{lane.Op} {v}, {lane.Zero}, {low}, {seen};");
            return v;
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_y, .param .u64 p_pos, .param .u32 p_n, .param .u32 p_cap, .param .u32 p_win, .param .{lane.Op} p_low", EachShared(f));
    }

    /// <summary>position[0] += steps (one thread). Params: pos, steps.</summary>
    public static string PositionAdvance(Gpu gpu, out string entry)
    {
        entry = "position_advance";
        var b = new PtxBuilder();
        string pos = Param64(b, "p_pos"), steps = Param32(b, "p_steps"), p = b.R();
        b.L($"ld.global.s32 {p}, [{pos}];");
        b.L($"add.s32 {p}, {p}, {steps};");
        b.L($"st.global.s32 [{pos}], {p};");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_pos, .param .u32 p_steps");
    }
}
