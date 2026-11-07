// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel normalization on CUDA (GpuOps.Resample.cs), the CPU's arithmetic value for value
// (Cpu/Resample.cs):
//   ResampleRows  a thread per value of the horizontal pass (planes × height × out width): its taps over the input row,
//                 read where it is stored, summed in the lane (fma, in order) or, 8-bit, as Pillow's integers clipped to
//                 a byte; stored as a lane value (s32);
//   ResampleOut   a thread per stored output word (a warp, or a CTA, per block of a block format): each value's vertical
//                 taps over the horizontal pass (or the input, the width kept), the channel's normalization, the word
//                 packed and stored once.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>The horizontal pass into mid. Params: x, bounds, weights, mid, n (values), width, across (the output width), taps.</summary>
    public static string ResampleRows(Gpu gpu, Fmt inFmt, Lane lane, bool bytes, out string entry)
    {
        entry = $"resample_rows_{inFmt.Name}_{(bytes ? "s32" : lane.Name)}";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), pb = Param64(b, "p_bounds"), pw = Param64(b, "p_w"), pm = Param64(b, "p_mid");
        string n = Param32(b, "p_n"), width = Param32(b, "p_width"), across = Param32(b, "p_across"), taps = Param32(b, "p_taps");
        string i = GlobalIndex(b), p = b.P();
        b.L($"setp.ge.u32 {p}, {i}, {n};");
        b.L($"@{p} bra DONE;");
        string ox = b.R(), row = b.R(), bi = b.R(), src = b.R(), wi = b.R();
        b.L($"rem.u32 {ox}, {i}, {across};");
        b.L($"div.u32 {row}, {i}, {across};");
        b.L($"shl.b32 {bi}, {ox}, 1;");
        var (first, cnt) = ResampleBounds(b, pb, bi);
        b.L($"mad.lo.u32 {src}, {row}, {width}, {first};");
        b.L($"mul.lo.u32 {wi}, {ox}, {taps};");
        string v = ResampleTaps(b, inFmt, lane, bytes, px, pw, src, "1", wi, cnt, "H");
        string a = b.RD();
        int size = bytes ? 4 : lane.Size;
        b.L($"mul.wide.u32 {a}, {i}, {size};");
        b.L($"add.u64 {a}, {pm}, {a};");
        b.L($"st.global.{(bytes ? "s32" : lane.Mem)} [{a}], {v};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_bounds, .param .u64 p_w, .param .u64 p_mid, .param .u32 p_n, .param .u32 p_width, .param .u32 p_across, .param .u32 p_taps");
    }

    /// <summary>The output of f: each value's vertical pass over src (the horizontal pass's values, or the input, as srcFmt)
    /// or its value there, then the normalization. Params: src, bounds, weights, values, y, count, oh, ow, h, across, taps,
    /// the rows' bounds and weights offsets, channels.</summary>
    public static string ResampleOut(Gpu gpu, Fmt f, Fmt srcFmt, Lane lane, bool bytes, bool vertical, bool norm, out string entry)
    {
        entry = $"resample_out_{f.Name}_{srcFmt.Name}_{lane.Name}{(bytes ? "_8" : "")}{(vertical ? "_v" : "")}{(norm ? "_n" : "")}";
        var b = new PtxBuilder();
        string ps = Param64(b, "p_src"), pb = Param64(b, "p_bounds"), pw = Param64(b, "p_w"), pv = Param64(b, "p_values"), py = Param64(b, "p_y");
        string count = Param32(b, "p_count"), oh = Param32(b, "p_oh"), ow = Param32(b, "p_ow"), h = Param32(b, "p_h"), across = Param32(b, "p_across"), taps = Param32(b, "p_taps");
        string rb = Param32(b, "p_rb"), rw = Param32(b, "p_rw"), channels = Param32(b, "p_channels");
        int tag = 0;
        string Value(string at)
        {
            string ox = b.R(), t = b.R(), oy = b.R(), p = b.R(), c = b.R(), src = b.R(), v;
            b.L($"rem.u32 {ox}, {at}, {ow};");
            b.L($"div.u32 {t}, {at}, {ow};");
            b.L($"rem.u32 {oy}, {t}, {oh};");
            b.L($"div.u32 {p}, {t}, {oh};");
            b.L($"rem.u32 {c}, {p}, {channels};");
            if (vertical)
            {
                string bi = b.R(), wi = b.R();
                b.L($"mad.lo.u32 {bi}, {oy}, 2, {rb};");
                var (first, cnt) = ResampleBounds(b, pb, bi);
                b.L($"mad.lo.u32 {src}, {p}, {h}, {first};");
                b.L($"mad.lo.u32 {src}, {src}, {across}, {ox};");
                b.L($"mad.lo.u32 {wi}, {oy}, {taps}, {rw};");
                v = ResampleTaps(b, srcFmt, lane, bytes, ps, pw, src, across, wi, cnt, $"V{tag++}");
            }
            else
            {
                b.L($"mad.lo.u32 {src}, {p}, {h}, {oy};");
                b.L($"mad.lo.u32 {src}, {src}, {across}, {ox};");
                v = Codec.UnpackAt(b, srcFmt, ps, src, bytes ? srcFmt.Natural : lane);
            }

            if (bytes)
            {
                // A byte's lane value: the channel's table entry, or the byte itself.
                if (!norm)
                {
                    string fv = lane.NewReg(b);
                    b.L($"cvt.rn.{lane.Op}.s32 {fv}, {v};");
                    return fv;
                }

                string ti = b.R();
                b.L($"mad.lo.u32 {ti}, {c}, 256, {v};");
                return LoadLane(b, lane, pv, ti);
            }

            if (!norm) return v;
            string si = b.R(), o = lane.NewReg(b);
            b.L($"shl.b32 {si}, {c}, 1;");
            b.L(lane.Fma(o, v, LoadLane(b, lane, pv, si), LoadLane(b, lane, pv, si, 1)));
            return o;
        }

        if (f.IsBlock)
        {
            if (BlockByCta(f)) EncodeBlockByCta(b, f, py, count, (at, _) => Value(at));
            else EncodeBlock(b, f, py, BlockWarp(b, f, count), count, (at, _) => Value(at));
        }
        else
        {
            string word = GlobalIndex(b), words = WordsOf(b, f, count), pw2 = b.P(), last = b.R();
            b.L($"setp.ge.u32 {pw2}, {word}, {words};");
            b.L($"@{pw2} bra DONE;");
            string code = f.WordBits == 64 ? b.RD() : b.R();
            if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
            var natural = f.Natural;
            b.L($"sub.u32 {last}, {count}, 1;");
            for (int i = 0; i < f.Per; i++)
            {
                // A word's values past the count take the last value's addresses (dropped).
                string at = b.R(), inside = b.P();
                b.L($"mad.lo.u32 {at}, {word}, {f.Per}, {i};");
                b.L($"setp.lt.u32 {inside}, {at}, {count};");
                b.L($"min.u32 {at}, {at}, {last};");
                string value = Value(at);
                string stored = lane == natural ? value : Codec.Convert(b, lane, value, natural, f);
                string kept = natural.NewReg(b);
                b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {inside};");
                Codec.Pack(b, f, kept, natural, i, code);
            }

            string addr = b.RD();
            b.L($"mul.wide.u32 {addr}, {word}, {f.WordBits / 8};");
            b.L($"add.u64 {addr}, {py}, {addr};");
            b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry,
            ".param .u64 p_src, .param .u64 p_bounds, .param .u64 p_w, .param .u64 p_values, .param .u64 p_y, .param .u32 p_count, .param .u32 p_oh, .param .u32 p_ow, .param .u32 p_h, "
            + ".param .u32 p_across, .param .u32 p_taps, .param .u32 p_rb, .param .u32 p_rw, .param .u32 p_channels",
            f.IsBlock && BlockByCta(f) ? f.BlockValues * 4 : 0);
    }

    // A pass's first position and tap count: bounds[bi], bounds[bi + 1] (int32).
    private static (string First, string Count) ResampleBounds(PtxBuilder b, string pb, string bi)
    {
        string a = b.RD(), first = b.R(), cnt = b.R();
        b.L($"mul.wide.u32 {a}, {bi}, 4;");
        b.L($"add.u64 {a}, {pb}, {a};");
        b.L($"ld.global.u32 {first}, [{a}];");
        b.L($"ld.global.u32 {cnt}, [{a}+4];");
        return (first, cnt);
    }

    // Σ over cnt taps of fmt's values from src (step apart) by the weights from wi: the lane's fma in order from 0, or, 8-bit,
    // 2^21 + Σ v·w as s32 clipped to a byte (Pillow's clip8). Its labels by tag.
    private static string ResampleTaps(PtxBuilder b, Fmt fmt, Lane lane, bool bytes, string px, string pw, string src, string step, string wi, string cnt, string tag)
    {
        string acc = bytes ? b.R() : lane.NewReg(b);
        if (bytes) b.L($"mov.u32 {acc}, {1 << (Resampling.PrecisionBits - 1)};");
        else b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
        string t = b.R(), p = b.P(), at = b.R(), wa = b.R();
        b.L($"mov.u32 {t}, 0;");
        b.Label($"RT{tag}");
        b.L($"setp.ge.u32 {p}, {t}, {cnt};");
        b.L($"@{p} bra RT{tag}_END;");
        b.L($"mad.lo.u32 {at}, {t}, {step}, {src};");
        b.L($"add.u32 {wa}, {wi}, {t};");
        if (bytes)
        {
            string v = Codec.UnpackAt(b, fmt, px, at, fmt.Natural), w = b.R(), wad = b.RD();
            b.L($"mul.wide.u32 {wad}, {wa}, 4;");
            b.L($"add.u64 {wad}, {pw}, {wad};");
            b.L($"ld.global.s32 {w}, [{wad}];");
            b.L($"mad.lo.s32 {acc}, {v}, {w}, {acc};");
        }
        else b.L(lane.Fma(acc, Codec.UnpackAt(b, fmt, px, at, lane), LoadLane(b, lane, pw, wa), acc));
        b.L($"add.u32 {t}, {t}, 1;");
        b.L($"bra RT{tag};");
        b.Label($"RT{tag}_END");
        if (!bytes) return acc;
        string r = b.R(), hi = b.P(), lo = b.P();
        b.L($"shr.s32 {r}, {acc}, {Resampling.PrecisionBits};");
        b.L($"setp.ge.s32 {hi}, {acc}, {1 << (Resampling.PrecisionBits + 8)};");
        b.L($"setp.le.s32 {lo}, {acc}, 0;");
        b.L($"selp.b32 {r}, 255, {r}, {hi};");
        b.L($"selp.b32 {r}, 0, {r}, {lo};");
        return r;
    }
}
