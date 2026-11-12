// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Element-wise kernels past the formulas: dropout from a seed (Draws: the same 32-bit arithmetic as the CPU), the
// conversion between any two types (Fmt.ConvertPlan: one rounding).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    // ---------------------------------------------------------------- draws

    // Draws.Mix: h ^= h >> 16; h *= 0x7FEB352D; h ^= h >> 15; h *= 0x846CA68B; h ^= h >> 16.
    private static string Mix(PtxBuilder b, string h0)
    {
        string h = b.R(), t = b.R();
        b.L($"shr.u32 {t}, {h0}, 16;");
        b.L($"xor.b32 {h}, {h0}, {t};");
        b.L($"mul.lo.u32 {h}, {h}, 0x7FEB352D;");
        b.L($"shr.u32 {t}, {h}, 15;");
        b.L($"xor.b32 {h}, {h}, {t};");
        b.L($"mul.lo.u32 {h}, {h}, 0x846CA68B;");
        b.L($"shr.u32 {t}, {h}, 16;");
        b.L($"xor.b32 {h}, {h}, {t};");
        return h;
    }

    /// <summary>Draws.Bits(key, idx): a u32 register.</summary>
    internal static string Draw(PtxBuilder b, string key, string idx)
    {
        string a = b.R(), c = b.R();
        b.L($"xor.b32 {a}, {idx}, {key};");
        string m = Mix(b, a);
        b.L($"add.u32 {c}, {m}, {key};");
        return Mix(b, c);
    }

    // The kept value or the dropped one: x·s (+ r, one fma) where bits ≥ thr, else r (+0 without a residual).
    private static string DropValue(PtxBuilder b, Lane lane, string x, string? r, string s, string bits, string thr)
    {
        string keep = b.P(), kept = lane.NewReg(b), drop = lane.NewReg(b), res = lane.NewReg(b);
        b.L($"setp.hs.u32 {keep}, {bits}, {thr};");
        if (r is null)
        {
            b.L($"mul.rn.{lane.Op} {kept}, {x}, {s};");
            b.L($"mov.{lane.Op} {drop}, {lane.Literal(0)};");
        }
        else
        {
            b.L($"fma.rn.{lane.Op} {kept}, {x}, {s}, {r};");
            b.L($"mov.{lane.Op} {drop}, {r};");
        }

        b.L($"selp.{lane.Op} {res}, {kept}, {drop}, {keep};");
        return res;
    }

    // ---------------------------------------------------------------- dropout: a thread per word

    /// <summary>output = residual + x·s where the seed's draw keeps a value, the residual where it does not, over n values of
    /// f in its float lane: a thread per word (the residual and the output may be x).</summary>
    public static string Dropout(Gpu gpu, Fmt f, Lane lane, bool residual, out string entry)
    {
        entry = $"dropout_{f.Name}_{lane.Name}{(residual ? "_res" : "")}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string px = Param64(b, "p_x"), pr = Param64(b, "p_r"), po = Param64(b, "p_o"), n = Param32(b, "p_n"), key = Param32(b, "p_key"), thr = Param32(b, "p_thr");
        string s = lane.NewReg(b);
        b.L($"ld.param.{lane.Mem} {s}, [p_s];");
        string words = WordsOf(b, f, n), p = b.P();
        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        string off = b.RD(), ax = b.RD(), ar = b.RD(), ao = b.RD();
        b.L($"mul.wide.u32 {off}, {gid}, {f.WordBits / 8};");
        b.L($"add.u64 {ax}, {px}, {off};");
        b.L($"add.u64 {ar}, {pr}, {off};");
        b.L($"add.u64 {ao}, {po}, {off};");
        string wx = f.WordBits == 64 ? b.RD() : b.R(), wr = f.WordBits == 64 ? b.RD() : b.R(), wo = f.WordBits == 64 ? b.RD() : b.R();
        b.L($"ld.global.{wt} {wx}, [{ax}];");
        if (residual) b.L($"ld.global.{wt} {wr}, [{ar}];");
        if (f.WordBits == 32) b.L($"mov.b32 {wo}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string x = Codec.Unpack(b, f, wx, i, lane);
            string? r = residual ? Codec.Unpack(b, f, wr, i, lane) : null;
            string idx = b.R();
            b.L($"mad.lo.u32 {idx}, {gid}, {f.Per}, {i};");
            Codec.Pack(b, f, DropValue(b, lane, x, r, s, Draw(b, key, idx), thr), lane, i, wo);
        }

        b.L($"st.global.{wt} [{ao}], {wo};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_r, .param .u64 p_o, .param .u32 p_n, .param .u32 p_key, .param .u32 p_thr, .param .{lane.Mem} p_s");
    }

    /// <summary>Dropout over a block format: each block's values decoded in place, kept or dropped in f32, the block encoded.</summary>
    public static string DropoutBlocks(Gpu gpu, Fmt f, bool residual, out string entry)
    {
        entry = $"dropout_{f.Name}_blocks{(residual ? "_res" : "")}";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), pr = Param64(b, "p_r"), po = Param64(b, "p_o"), n = Param32(b, "p_n"), key = Param32(b, "p_key"), thr = Param32(b, "p_thr");
        string s = b.F();
        b.L($"ld.param.f32 {s}, [p_s];");
        var w = BlockWarp(b, f, n);
        EncodeBlock(b, f, po, w, n, at =>
        {
            string x = Codec.UnpackAt(b, f, px, at, Lane.F32);
            string? r = residual ? Codec.UnpackAt(b, f, pr, at, Lane.F32) : null;
            return DropValue(b, Lane.F32, x, r, s, Draw(b, key, at), thr);
        });
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_r, .param .u64 p_o, .param .u32 p_n, .param .u32 p_key, .param .u32 p_thr, .param .f32 p_s");
    }

    // Words that hold n values of f (a u32 register).
    private static string WordsOf(PtxBuilder b, Fmt f, string n)
    {
        string words = b.R();
        if (f.WordBits == 64) { b.L($"mov.u32 {words}, {n};"); return words; }
        string bits = b.RD(), wq = b.RD();
        b.L($"mul.wide.u32 {bits}, {n}, {f.Bits};");
        b.L($"add.u64 {bits}, {bits}, 31;");
        b.L($"shr.u64 {wq}, {bits}, 5;");
        b.L($"cvt.u32.u64 {words}, {wq};");
        return words;
    }

    // ---------------------------------------------------------------- conversion

    /// <summary>A lane value rounded to odd at float32 (toward zero, the last bit set when anything was dropped): an f32 register.</summary>
    private static string ToOddF32(PtxBuilder b, Lane lane, string v)
    {
        string r = b.F(), bits = b.R(), odd = b.R(), p = b.P();
        if (lane == Lane.F64)
        {
            string back = b.RD();
            b.L($"cvt.rz.f32.f64 {r}, {v};");
            string bd = Lane.F64.NewReg(b);
            b.L($"cvt.f64.f32 {bd}, {r};");
            b.L($"setp.ne.f64 {p}, {bd}, {v};");                                // ordered: a NaN stays as narrowed
            _ = back;
        }
        else
        {
            string back = lane.Wide ? b.RD() : b.R();
            b.L($"cvt.rz.f32.{lane.Op} {r}, {v};");
            b.L($"cvt.rzi.{lane.Op}.f32 {back}, {r};");
            b.L($"setp.ne.{lane.Op} {p}, {back}, {v};");
        }

        b.L($"mov.b32 {bits}, {r};");
        b.L($"or.b32 {odd}, {bits}, 1;");
        b.L($"selp.b32 {bits}, {odd}, {bits}, {p};");
        string res = b.F();
        b.L($"mov.b32 {res}, {bits};");
        return res;
    }

    /// <summary>output (ft) = x (ff) over n values, a thread per output word, each value rounded once (Fmt.ConvertPlan); the
    /// values past n in the last word 0. Not a block format's output (ConvertBlocks).</summary>
    public static string Convert(Gpu gpu, Fmt ff, Fmt ft, out string entry)
    {
        entry = $"convert_{ff.Name}_{ft.Name}";
        var (lane, toOdd) = Fmt.ConvertPlan(ff, ft);
        var nat = ff.IsBlock ? Lane.F32 : ff.Natural;
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string px = Param64(b, "p_x"), po = Param64(b, "p_o"), n = Param32(b, "p_n");
        string words = WordsOf(b, ft, n), p = b.P(), last = b.R();
        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        b.L($"sub.u32 {last}, {n}, 1;");
        string wo = ft.WordBits == 64 ? b.RD() : b.R();
        if (ft.WordBits == 32) b.L($"mov.b32 {wo}, 0;");
        for (int i = 0; i < ft.Per; i++)
        {
            string idx = b.R(), at = b.R(), inside = b.P();
            b.L($"mad.lo.u32 {idx}, {gid}, {ft.Per}, {i};");
            b.L($"setp.lt.u32 {inside}, {idx}, {n};");
            b.L($"min.u32 {at}, {idx}, {last};");
            string v = Codec.UnpackAt(b, ff, px, at, nat);
            string val;
            Lane vl;
            if (!ff.IsFloat && !ft.IsFloat)
            {
                // Integers: widened to the 64-bit lane, held to ft's range there, then stored from s64.
                string w = b.RD();
                if (nat == lane) b.L($"mov.b64 {w}, {v};");
                else b.L(nat.Wide ? $"mov.b64 {w}, {v};" : $"cvt.{(nat.Name == "s32" ? "s64.s32" : "u64.u32")} {w}, {v};");
                if (lane == Lane.U64)
                {
                    if (ft.Signed || ft.Bits < 64) b.L($"min.u64 {w}, {w}, {(ft.Signed ? (1UL << (ft.Bits - 1)) - 1 : (1UL << ft.Bits) - 1)};");
                }
                else
                {
                    long low = ft.Signed ? (ft.Bits == 64 ? long.MinValue : -(1L << (ft.Bits - 1))) : 0;
                    long high = ft.Signed ? (ft.Bits == 64 ? long.MaxValue : (1L << (ft.Bits - 1)) - 1) : ft.Bits == 64 ? long.MaxValue : (1L << ft.Bits) - 1;
                    b.L($"max.s64 {w}, {w}, {(low == long.MinValue ? "0x8000000000000000" : low.ToString(System.Globalization.CultureInfo.InvariantCulture))};");
                    b.L($"min.s64 {w}, {w}, {high};");
                }

                b.L($"selp.b64 {w}, {w}, 0, {inside};");
                val = w;
                vl = Lane.S64;
            }
            else if (toOdd) { val = ToOddF32(b, nat, v); vl = Lane.F32; }
            else if (ft.IsFloat && !nat.IsFloat)
            {
                vl = ft.Is("float64") ? Lane.F64 : Lane.F32;
                val = vl.NewReg(b);
                b.L($"cvt.rn.{vl.Op}.{nat.Op} {val}, {v};");
            }
            else { val = v; vl = nat; }

            if (vl.IsFloat)
            {
                string z = vl.NewReg(b);
                b.L($"selp.{vl.Op} {z}, {val}, {vl.Literal(0)}, {inside};");
                val = z;
            }

            Codec.Pack(b, ft, val, vl, i, wo);
        }

        string ao = b.RD();
        b.L($"mul.wide.u32 {ao}, {gid}, {ft.WordBits / 8};");
        b.L($"add.u64 {ao}, {po}, {ao};");
        b.L($"st.global.{(ft.WordBits == 64 ? "b64" : "b32")} [{ao}], {wo};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_o, .param .u32 p_n");
    }

    /// <summary>x (ff) into the block format ft: each block's values read as float32 (rounded to nearest where wider, as the
    /// format defines its encoding from float32), the block encoded.</summary>
    public static string ConvertBlocks(Gpu gpu, Fmt ff, Fmt ft, out string entry)
    {
        entry = $"convert_{ff.Name}_{ft.Name}_blocks";
        var b = new PtxBuilder();
        string px = Param64(b, "p_x"), po = Param64(b, "p_o"), n = Param32(b, "p_n");
        var w = BlockWarp(b, ft, n);
        EncodeBlock(b, ft, po, w, n, at => Codec.UnpackAt(b, ff, px, at, Lane.F32));
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_o, .param .u32 p_n");
    }
}
