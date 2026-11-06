// Qiyas on CUDA: PTX generated once per (stored format, lane) from the number's description — the same contract as the
// CPU side, in the device's terms. A stored format (Fmt) says how values sit in 32-bit words (64-bit words for the
// 64-bit types) and emits the instructions that take one value out of a word into a lane register or put a lane value
// back (rounded to nearest even, held to the type's range); a lane (Lane) is a register type with its arithmetic. The
// kernels are written once over those:
//
//   add            c = a + b on 16-byte vectors, each value unpacked, added in the lane, packed
//   cvt / back     a stored array into a lane array and back (the operands of a product in lanes other than the dot lane)
//   packa / packb  integer operands into groups of four 8-bit values per 32-bit word (A by rows, B transposed), for dp4a
//   gemm           C = A·B in a lane: 128×128 (or 64×64) tiles of C per block, 8×8 (4×4) values per thread, through shared memory
//   gemm_dp4a      the same shape over the groups, one dp4a per four products
//   range          the smallest and largest stored value (the bounds a product's lane choice is made on)
//   softmax, rms   one block per row in a float lane, three (two) passes over the row's words
//
// Exp is the same polynomial as the CPU's (fused multiply-adds, exact reduction, the scale built in the exponent
// bits), so the two devices agree to the last bit of the method.
using System.Globalization;
using System.Text;

namespace Qiyas;

/// <summary>A value's stored format as the PTX sees it.</summary>
public sealed record Fmt(string Name, int Bits, bool IsFloat, bool Signed)
{
    public int WordBits => Bits == 64 ? 64 : 32;

    /// <summary>Values a word holds.</summary>
    public int Per => WordBits / Bits;

    /// <summary>A block format: values in blocks of BlockValues taking BlockBytes (0 for the plain formats).</summary>
    public int BlockValues => Name switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 0 };
    public int BlockBytes => Name switch { "q8_0" => 34, "q4_0" => 18, "q4_k" => 144, "q6_k" => 210, _ => 0 };
    public bool IsBlock => BlockValues > 0;

    public static Fmt Of<TN>() where TN : struct, IQiyasNumber<TN> => new(TN.Name, TN.Bits, TN.IsFloat, TN.Min < 0);

    public bool Is(string name) => Name == name;

    /// <summary>Words that hold <paramref name="values"/> values (a block format: its whole blocks, in words).</summary>
    public long Words(long values) => IsBlock ? ((values + BlockValues - 1) / BlockValues * BlockBytes + 3) / 4 : (values * Bits + WordBits - 1) / WordBits;

    /// <summary>The lane this format's values are read into before any other (floats f32/f64, integers s32/u32/s64/u64).</summary>
    public Lane Natural => IsFloat ? (Bits == 64 ? Lane.F64 : Lane.F32) : Bits == 64 ? (Signed ? Lane.S64 : Lane.U64) : Signed ? Lane.S32 : Lane.U32;
}

/// <summary>A register type kernels compute in.</summary>
public sealed record Lane(string Name, string Reg, string Op, int Size, bool IsFloat)
{
    public static readonly Lane F32 = new("f32", "f32", "f32", 4, true);
    public static readonly Lane F64 = new("f64", "f64", "f64", 8, true);
    public static readonly Lane S32 = new("s32", "b32", "s32", 4, false);
    public static readonly Lane U32 = new("u32", "b32", "u32", 4, false);
    public static readonly Lane S64 = new("s64", "b64", "s64", 8, false);
    public static readonly Lane U64 = new("u64", "b64", "u64", 8, false);

    public bool Wide => Size == 8;

    public string Zero => IsFloat ? (Wide ? "0d0000000000000000" : "0f00000000") : "0";

    public string NewReg(PtxBuilder b) => Reg switch { "f32" => b.F(), "f64" => b.FD(), "b32" => b.R(), _ => b.RD() };

    public string Fma(string d, string a, string b, string c) => IsFloat ? $"fma.rn.{Op} {d}, {a}, {b}, {c};" : $"mad.lo.{Op} {d}, {a}, {b}, {c};";

    public string Add(string d, string a, string b) => IsFloat ? $"add.rn.{Op} {d}, {a}, {b};" : $"add.{Op} {d}, {a}, {b};";

    public string Mul(string d, string a, string b) => IsFloat ? $"mul.rn.{Op} {d}, {a}, {b};" : $"mul.lo.{Op} {d}, {a}, {b};";

    /// <summary>The memory type of a load or store of this lane.</summary>
    public string Mem => IsFloat ? Op : Wide ? "b64" : "b32";

    public string Literal(double value) => IsFloat
        ? Wide ? "0d" + BitConverter.DoubleToUInt64Bits(value).ToString("X16") : "0f" + BitConverter.SingleToUInt32Bits((float)value).ToString("X8")
        : ((long)value).ToString(CultureInfo.InvariantCulture);
}

/// <summary>A kernel's text: registers named as they are asked for, declared when the kernel is finished.</summary>
public sealed class PtxBuilder
{
    private readonly StringBuilder _body = new();
    private int _r, _rd, _f, _fd, _h, _p;

    public string R() => $"%r{_r++}";
    public string RD() => $"%rd{_rd++}";
    public string F() => $"%f{_f++}";
    public string FD() => $"%fd{_fd++}";
    public string H() => $"%h{_h++}";
    public string P() => $"%p{_p++}";

    public void L(string line) => _body.Append("    ").Append(line).Append('\n');

    public void Label(string name) => _body.Append(name).Append(":\n");

    public string Finish(string header, string entry, string parameters, int sharedBytes = 0)
    {
        var sb = new StringBuilder(header);
        if (sharedBytes > 48 * 1024) sb.Append(".extern .shared .align 16 .b8 smem[];\n");
        sb.Append(".visible .entry ").Append(entry).Append('(').Append(parameters).Append(")\n{\n");
        sb.Append($"    .reg .pred %p<{Math.Max(1, _p)}>;\n    .reg .b32 %r<{Math.Max(1, _r)}>;\n    .reg .b64 %rd<{Math.Max(1, _rd)}>;\n");
        sb.Append($"    .reg .f32 %f<{Math.Max(1, _f)}>;\n    .reg .f64 %fd<{Math.Max(1, _fd)}>;\n    .reg .b16 %h<{Math.Max(1, _h)}>;\n");
        if (sharedBytes is > 0 and <= 48 * 1024) sb.Append($"    .shared .align 16 .b8 smem[{sharedBytes}];\n");
        sb.Append(_body).Append("}\n");
        return sb.ToString();
    }
}

/// <summary>One value out of a word and back, and the lane conversions, as PTX.</summary>
internal static class Codec
{
    private static string Hex(float v) => "0f" + BitConverter.SingleToUInt32Bits(v).ToString("X8");

    // ---------------------------------------------------------------- a value out of a word

    /// <summary>Value <paramref name="i"/> of <paramref name="word"/> (a .b32 register, .b64 for 64-bit formats) as a <paramref name="lane"/> register.</summary>
    public static string Unpack(PtxBuilder b, Fmt f, string word, int i, Lane lane)
    {
        string natural = f.IsFloat ? Float(b, f, word, i) : Integer(b, f, word, i, lane);
        return Convert(b, f.Natural, natural, lane, f);
    }

    /// <summary>The value at linear index <paramref name="index"/> (a .b32 register) of the array at <paramref name="basePtr"/>, as a lane register (integer formats).</summary>
    public static string UnpackAt(PtxBuilder b, Fmt f, string basePtr, string index, Lane lane)
    {
        if (f.IsBlock) return Convert(b, Lane.F32, BlockDecode(b, f, basePtr, index), lane, f);
        if (f.WordBits == 64)
        {
            string a = b.RD(), w = b.RD();
            b.L($"mul.wide.u32 {a}, {index}, 8;");
            b.L($"add.u64 {a}, {basePtr}, {a};");
            b.L($"ld.global.b64 {w}, [{a}];");
            return Convert(b, f.Natural, w, lane, f);
        }

        string bits = b.R(), wordIndex = b.R(), shift = b.R(), addr = b.RD(), wr = b.R();
        b.L($"mul.lo.u32 {bits}, {index}, {f.Bits};");
        b.L($"shr.u32 {wordIndex}, {bits}, 5;");
        b.L($"and.b32 {shift}, {bits}, 31;");
        b.L($"mul.wide.u32 {addr}, {wordIndex}, 4;");
        b.L($"add.u64 {addr}, {basePtr}, {addr};");
        b.L($"ld.global.b32 {wr}, [{addr}];");
        if (f.IsFloat) return Convert(b, f.Natural, FloatDecode(b, f, wr, shift), lane, f);
        string v = b.R();
        if (f.Bits == 32) b.L($"mov.b32 {v}, {wr};");
        else b.L($"bfe.{(f.Signed ? "s32" : "u32")} {v}, {wr}, {shift}, {f.Bits};");
        return Convert(b, f.Natural, v, lane, f);
    }

    // ---- block formats: the value at a linear index, decoded from its block (byte loads), as f32
    private static string BlockDecode(PtxBuilder b, Fmt f, string basePtr, string index)
    {
        string blk = b.R(), off = b.R(), bAddr = b.RD(), t = b.R(), v = b.F();
        b.L($"div.u32 {blk}, {index}, {f.BlockValues};");
        b.L($"mul.lo.u32 {t}, {blk}, {f.BlockValues};");
        b.L($"sub.u32 {off}, {index}, {t};");
        b.L($"mul.wide.u32 {bAddr}, {blk}, {f.BlockBytes};");
        b.L($"add.u64 {bAddr}, {basePtr}, {bAddr};");
        string HalfAt(int at) { string h = b.H(), r = b.F(); b.L($"ld.global.u16 {h}, [{bAddr}+{at}];"); b.L($"cvt.f32.f16 {r}, {h};"); return r; }
        string ByteAt(string offset, int at) { string a = b.RD(), r = b.R(); b.L($"cvt.u64.u32 {a}, {offset};"); b.L($"add.u64 {a}, {bAddr}, {a};"); b.L($"ld.global.u8 {r}, [{a}+{at}];"); return r; }
        string I2F(string r) { string fr = b.F(); b.L($"cvt.rn.f32.s32 {fr}, {r};"); return fr; }
        switch (f.Name)
        {
            case "q8_0":
            {
                string d = HalfAt(0), q = ByteAt(off, 2), qs = b.R();
                b.L($"cvt.s32.s8 {qs}, {q};");
                b.L($"mul.rn.f32 {v}, {d}, {I2F(qs)};");
                return v;
            }

            case "q4_0":
            {
                // value i: the low nibble of byte i (i < 16), the high nibble of byte i − 16.
                string d = HalfAt(0), lo = b.R(), pHi = b.P(), nib = b.R(), q = b.R(), byteOff = b.R();
                b.L($"and.b32 {byteOff}, {off}, 15;");
                string by = ByteAt(byteOff, 2);
                b.L($"setp.ge.u32 {pHi}, {off}, 16;");
                b.L($"and.b32 {lo}, {by}, 15;");
                b.L($"shr.u32 {nib}, {by}, 4;");
                b.L($"selp.u32 {q}, {nib}, {lo}, {pHi};");
                b.L($"sub.s32 {q}, {q}, 8;");
                b.L($"mul.rn.f32 {v}, {d}, {I2F(q)};");
                return v;
            }

            case "q4_k":
            {
                // sub-block j = off / 32 (pair j/2, the low nibbles for even j, the high for odd); scale_j, min_j from the 12 bytes at 4.
                string d = HalfAt(0), dmin = HalfAt(2), j = b.R(), i = b.R(), pair = b.R(), pOdd = b.R(), byteOff = b.R(), lo = b.R(), hi = b.R(), q = b.R(), pIsOdd = b.P();
                b.L($"shr.u32 {j}, {off}, 5;");
                b.L($"and.b32 {i}, {off}, 31;");
                b.L($"shr.u32 {pair}, {j}, 1;");
                b.L($"and.b32 {pOdd}, {j}, 1;");
                b.L($"setp.ne.u32 {pIsOdd}, {pOdd}, 0;");
                b.L($"shl.b32 {byteOff}, {pair}, 5;");
                b.L($"add.u32 {byteOff}, {byteOff}, {i};");
                string by = ByteAt(byteOff, 16);
                b.L($"and.b32 {lo}, {by}, 15;");
                b.L($"shr.u32 {hi}, {by}, 4;");
                b.L($"selp.u32 {q}, {hi}, {lo}, {pIsOdd};");
                // scale, min: j < 4: sc[j] & 63, sc[j+4] & 63; else (sc[j+4] & 0xF) | (sc[j-4] >> 6) << 4, (sc[j+4] >> 4) | (sc[j] >> 6) << 4
                string pLow = b.P(), sA = b.R(), sB = b.R(), sC = b.R(), scale = b.R(), min = b.R(), jp4 = b.R(), jm4 = b.R(), jc = b.R();
                b.L($"setp.lt.u32 {pLow}, {j}, 4;");
                b.L($"add.u32 {jp4}, {j}, 4;");
                b.L($"sub.u32 {jm4}, {j}, 4;");
                b.L($"selp.u32 {jm4}, {j}, {jm4}, {pLow};");                 // a valid byte either way
                b.L($"mov.u32 {jc}, {j};");
                string scJ = ByteAt(jc, 4), scJp4 = ByteAt(jp4, 4), scJm4 = ByteAt(jm4, 4);
                string lowS = b.R(), lowM = b.R(), highS = b.R(), highM = b.R();
                b.L($"and.b32 {lowS}, {scJ}, 63;");
                b.L($"and.b32 {lowM}, {scJp4}, 63;");
                b.L($"and.b32 {sA}, {scJp4}, 15;");
                b.L($"shr.u32 {sB}, {scJm4}, 6;");
                b.L($"shl.b32 {sB}, {sB}, 4;");
                b.L($"or.b32 {highS}, {sA}, {sB};");
                b.L($"shr.u32 {sC}, {scJp4}, 4;");
                b.L($"shr.u32 {sB}, {scJ}, 6;");
                b.L($"shl.b32 {sB}, {sB}, 4;");
                b.L($"or.b32 {highM}, {sC}, {sB};");
                b.L($"selp.u32 {scale}, {lowS}, {highS}, {pLow};");
                b.L($"selp.u32 {min}, {lowM}, {highM}, {pLow};");
                string ds = b.F(), dm = b.F(), prod = b.F();
                b.L($"mul.rn.f32 {ds}, {d}, {I2F(scale)};");
                b.L($"mul.rn.f32 {dm}, {dmin}, {I2F(min)};");
                b.L($"mul.rn.f32 {prod}, {ds}, {I2F(q)};");
                b.L($"sub.rn.f32 {v}, {prod}, {dm};");
                return v;
            }

            case "q6_k":
            {
                // half = off / 128; l = off % 32; quarter = (off % 128) / 32 (0..3): the ql byte and the qh bits by the layout.
                string d = HalfAt(208), half = b.R(), rest = b.R(), quarter = b.R(), l = b.R(), lBase = b.R(), hBase = b.R(), sBase = b.R();
                b.L($"shr.u32 {half}, {off}, 7;");
                b.L($"and.b32 {rest}, {off}, 127;");
                b.L($"shr.u32 {quarter}, {rest}, 5;");
                b.L($"and.b32 {l}, {rest}, 31;");
                b.L($"shl.b32 {lBase}, {half}, 6;");
                b.L($"shl.b32 {hBase}, {half}, 5;");
                b.L($"shl.b32 {sBase}, {half}, 3;");
                // ql byte: l (quarters 0, 2) or l + 32 (quarters 1, 3); the nibble: low (quarters 0, 1) or high (2, 3); qh shift = 2·quarter.
                string qlOff = b.R(), odd = b.R(), pOdd = b.P(), pHigh = b.P(), qlo = b.R(), qhi = b.R(), qn = b.R(), qhOff = b.R(), shiftH = b.R(), hbits = b.R(), q = b.R();
                b.L($"and.b32 {odd}, {quarter}, 1;");
                b.L($"setp.ne.u32 {pOdd}, {odd}, 0;");
                b.L($"setp.ge.u32 {pHigh}, {quarter}, 2;");
                b.L($"add.u32 {qlOff}, {lBase}, {l};");
                b.L($"@{pOdd} add.u32 {qlOff}, {qlOff}, 32;");
                string qlByte = ByteAt(qlOff, 0);
                b.L($"and.b32 {qlo}, {qlByte}, 15;");
                b.L($"shr.u32 {qhi}, {qlByte}, 4;");
                b.L($"selp.u32 {qn}, {qhi}, {qlo}, {pHigh};");
                b.L($"add.u32 {qhOff}, {hBase}, {l};");
                string qhByte = ByteAt(qhOff, 128);
                b.L($"shl.b32 {shiftH}, {quarter}, 1;");
                b.L($"shr.u32 {hbits}, {qhByte}, {shiftH};");
                b.L($"and.b32 {hbits}, {hbits}, 3;");
                b.L($"shl.b32 {hbits}, {hbits}, 4;");
                b.L($"or.b32 {q}, {qn}, {hbits};");
                b.L($"sub.s32 {q}, {q}, 32;");
                // scale index: sBase + (l / 16) + 2·quarter
                string sIdx = b.R(), sc = b.R(), scS = b.R();
                b.L($"shr.u32 {sIdx}, {l}, 4;");
                b.L($"add.u32 {sIdx}, {sIdx}, {sBase};");
                b.L($"mad.lo.u32 {sIdx}, {quarter}, 2, {sIdx};");
                string scByte = ByteAt(sIdx, 192);
                b.L($"cvt.s32.s8 {scS}, {scByte};");
                string ds = b.F();
                b.L($"mul.rn.f32 {ds}, {d}, {I2F(scS)};");
                b.L($"mul.rn.f32 {v}, {ds}, {I2F(q)};");
                _ = sc;
                return v;
            }

            default: throw new NotSupportedException($"{f.Name}: no block decoder");
        }
    }

    private static string Float(PtxBuilder b, Fmt f, string word, int i) => FloatDecode(b, f, word, (i * f.Bits).ToString());

    // The float value whose bits start at bit `shift` (an immediate or a register) of the word.
    private static string FloatDecode(PtxBuilder b, Fmt f, string word, string shift)
    {
        if (f.Is("float64")) return word;
        if (f.Is("float32")) return Move32(b, word);
        string t = b.R();
        if (f.Is("float16"))
        {
            string h = b.H(), v = b.F();
            b.L($"shr.u32 {t}, {word}, {shift};");
            b.L($"cvt.u16.u32 {h}, {t};");
            b.L($"cvt.f32.f16 {v}, {h};");
            return v;
        }

        if (f.Is("bfloat16"))
        {
            string v = b.F();
            b.L($"shr.u32 {t}, {word}, {shift};");
            b.L($"shl.b32 {t}, {t}, 16;");
            b.L($"mov.b32 {v}, {t};");
            return v;
        }

        // fp8: the code's fields to a float32's.
        bool e4m3 = f.Is("float8_e4m3");
        string c = b.R(), sg = b.R(), e = b.R(), m = b.R(), t1 = b.R(), t2 = b.R(), norm = b.R(), fm = b.F(), sub = b.R(), res = b.R(), p = b.P(), outF = b.F();
        b.L($"bfe.u32 {c}, {word}, {shift}, 8;");
        b.L($"and.b32 {sg}, {c}, 0x80;");
        b.L($"shl.b32 {sg}, {sg}, 24;");
        b.L($"shr.u32 {e}, {c}, {(e4m3 ? 3 : 2)};");
        b.L($"and.b32 {e}, {e}, {(e4m3 ? 15 : 31)};");
        b.L($"and.b32 {m}, {c}, {(e4m3 ? 7 : 3)};");
        b.L($"add.u32 {t1}, {e}, {(e4m3 ? 120 : 112)};");
        b.L($"shl.b32 {t1}, {t1}, 23;");
        b.L($"shl.b32 {t2}, {m}, {(e4m3 ? 20 : 21)};");
        b.L($"or.b32 {norm}, {t1}, {t2};");
        b.L($"or.b32 {norm}, {norm}, {sg};");
        b.L($"cvt.rn.f32.u32 {fm}, {m};");
        b.L($"mul.f32 {fm}, {fm}, {Hex(e4m3 ? 1f / 512 : 1f / 65536)};");
        b.L($"mov.b32 {sub}, {fm};");
        b.L($"or.b32 {sub}, {sub}, {sg};");
        b.L($"setp.eq.u32 {p}, {e}, 0;");
        b.L($"selp.b32 {res}, {sub}, {norm}, {p};");
        string nan = b.R(), pn = b.P();
        b.L($"or.b32 {nan}, {sg}, 0x7FC00000;");
        if (e4m3)
        {
            string t3 = b.R();
            b.L($"and.b32 {t3}, {c}, 0x7F;");
            b.L($"setp.eq.u32 {pn}, {t3}, 0x7F;");
            b.L($"selp.b32 {res}, {nan}, {res}, {pn};");
        }
        else
        {
            string inf = b.R(), pt = b.P(), pm = b.P();
            b.L($"setp.eq.u32 {pt}, {e}, 31;");
            b.L($"or.b32 {inf}, {sg}, 0x7F800000;");
            b.L($"selp.b32 {res}, {inf}, {res}, {pt};");
            b.L($"setp.ne.u32 {pm}, {m}, 0;");
            b.L($"and.pred {pn}, {pt}, {pm};");
            b.L($"selp.b32 {res}, {nan}, {res}, {pn};");
        }

        b.L($"mov.b32 {outF}, {res};");
        return outF;
    }

    private static string Move32(PtxBuilder b, string word)
    {
        string v = b.F();
        b.L($"mov.b32 {v}, {word};");
        return v;
    }

    private static string Integer(PtxBuilder b, Fmt f, string word, int i, Lane lane)
    {
        if (f.WordBits == 64 || f.Bits == 32) return word;                   // the word is the value (s32/u32/s64/u64)
        string v = b.R();
        b.L($"bfe.{(f.Signed ? "s32" : "u32")} {v}, {word}, {f.Bits * i}, {f.Bits};");
        return v;
    }

    /// <summary>A value in lane <paramref name="from"/> as lane <paramref name="to"/> (exact by the rules that chose the lanes).</summary>
    public static string Convert(PtxBuilder b, Lane from, string v, Lane to, Fmt f)
    {
        if (from == to) return v;
        string d = to.NewReg(b);
        if (from.IsFloat && to.IsFloat) b.L(to.Wide ? $"cvt.f64.f32 {d}, {v};" : $"cvt.rn.f32.f64 {d}, {v};");
        else if (!from.IsFloat && to.IsFloat) b.L($"cvt.rn.{to.Op}.{from.Op} {d}, {v};");
        else if (!from.IsFloat && !to.IsFloat)
        {
            if (from.Size == to.Size) b.L($"mov.{to.Reg} {d}, {v};");
            else if (to.Wide) b.L($"cvt.{to.Op}.{from.Op} {d}, {v};");
            else b.L($"cvt.{to.Op}.{from.Op} {d}, {v};");                 // a narrowing load asked for only when the range fits
        }
        else throw new NotSupportedException($"{f.Name}: {from.Name} to {to.Name}");
        return d;
    }

    // ---------------------------------------------------------------- a lane value into a word

    /// <summary>Value <paramref name="v"/> (a <paramref name="lane"/> register) into position <paramref name="i"/> of <paramref name="word"/>, which starts at zero (64-bit formats: becomes the word).</summary>
    public static void Pack(PtxBuilder b, Fmt f, string v, Lane lane, int i, string word, bool wrap = false)
    {
        if (f.IsBlock) throw new NotSupportedException($"{f.Name}: a block format is encoded a block at a time (Ptx.LanesToBlock), not a value at a time");
        if (f.IsFloat)
        {
            if (f.Is("float64"))
            {
                b.L(lane.Wide ? $"mov.b64 {word}, {v};" : $"cvt.f64.f32 {word}, {v};");
                return;
            }

            string x = lane.Wide ? Convert(b, Lane.F64, v, Lane.F32, f) : v;
            if (f.Is("float32")) { b.L($"mov.b32 {word}, {x};"); return; }
            string code = b.R();
            if (f.Is("float16"))
            {
                string h = b.H();
                b.L($"cvt.rn.f16.f32 {h}, {x};");
                b.L($"cvt.u32.u16 {code}, {h};");
            }
            else if (f.Is("bfloat16"))
            {
                string bits = b.R(), t1 = b.R(), t2 = b.R(), t3 = b.R(), p = b.P();
                b.L($"mov.b32 {bits}, {x};");
                b.L($"shr.u32 {t1}, {bits}, 16;");
                b.L($"and.b32 {t1}, {t1}, 1;");
                b.L($"add.u32 {t2}, {bits}, 0x7FFF;");
                b.L($"add.u32 {t2}, {t2}, {t1};");
                b.L($"shr.u32 {t2}, {t2}, 16;");
                b.L($"shr.u32 {t3}, {bits}, 16;");
                b.L($"or.b32 {t3}, {t3}, 0x40;");
                b.L($"testp.notanumber.f32 {p}, {x};");
                b.L($"selp.b32 {code}, {t3}, {t2}, {p};");
            }
            else
            {
                bool e4m3 = f.Is("float8_e4m3");
                string bits = b.R(), sg = b.R(), abs = b.R(), lsb = b.R(), norm = b.R(), fa = b.F(), fs = b.F(), sub = b.R(), p = b.P(), p2 = b.P(), top = b.R();
                b.L($"mov.b32 {bits}, {x};");
                b.L($"shr.u32 {sg}, {bits}, 24;");
                b.L($"and.b32 {sg}, {sg}, 0x80;");
                b.L($"and.b32 {abs}, {bits}, 0x7FFFFFFF;");
                b.L($"shr.u32 {lsb}, {abs}, {(e4m3 ? 20 : 21)};");
                b.L($"and.b32 {lsb}, {lsb}, 1;");
                b.L($"add.u32 {norm}, {abs}, {(e4m3 ? "0x7FFFF" : "0xFFFFF")};");
                b.L($"add.u32 {norm}, {norm}, {lsb};");
                b.L($"shr.u32 {norm}, {norm}, {(e4m3 ? 20 : 21)};");
                b.L($"sub.u32 {norm}, {norm}, {(e4m3 ? 960 : 448)};");
                b.L($"mov.b32 {fa}, {abs};");
                b.L($"mul.f32 {fs}, {fa}, {Hex(e4m3 ? 512f : 65536f)};");
                b.L($"cvt.rni.u32.f32 {sub}, {fs};");
                b.L($"setp.lt.u32 {p}, {abs}, {(e4m3 ? "0x3C800000" : "0x38800000")};");
                b.L($"selp.b32 {code}, {sub}, {norm}, {p};");
                b.L($"setp.ge.u32 {p2}, {abs}, {(e4m3 ? "0x43E80000" : "0x47700000")};");
                b.L($"selp.b32 {code}, {(e4m3 ? "0x7F" : "0x7C")}, {code}, {p2};");
                if (!e4m3)
                {
                    string pn = b.P();
                    b.L($"setp.gt.u32 {pn}, {abs}, 0x7F800000;");
                    b.L($"selp.b32 {code}, 0x7F, {code}, {pn};");
                }

                b.L($"or.b32 {code}, {code}, {sg};");
                _ = top;
            }

            Insert(b, f, code, i, word);
            return;
        }

        // Integers: the lane value rounded (floats) and held to the type's range, then its low bits into the word.
        if (f.WordBits == 64)
        {
            if (lane.IsFloat) b.L($"cvt.rni.sat.{f.Natural.Op}.{lane.Op} {word}, {v};");
            else if (lane.Wide) b.L($"mov.b64 {word}, {v};");
            else b.L($"cvt.{f.Natural.Op}.{lane.Op} {word}, {v};");
            return;
        }

        string n = b.R();
        string nat = f.Signed ? "s32" : "u32";
        if (wrap && !lane.IsFloat && !lane.Wide)
        {
            Insert(b, f, v, i, word);                                          // the low bits: modular at the type's width
            return;
        }

        if (lane.IsFloat) b.L($"cvt.rni.sat.{nat}.{lane.Op} {n}, {v};");
        else if (lane.Wide)
        {
            // Held at 64 bits first (a value past 32 bits), then taken down.
            string held = b.RD();
            long low = f.Signed ? -(1L << (f.Bits - 1)) : 0, high = f.Signed ? (1L << (f.Bits - 1)) - 1 : (1L << f.Bits) - 1;
            b.L($"max.{lane.Op} {held}, {v}, {low};");
            b.L($"min.{lane.Op} {held}, {held}, {high};");
            b.L($"cvt.{nat}.{lane.Op} {n}, {held};");
        }
        else b.L($"mov.b32 {n}, {v};");

        if (f.Bits < 32)
        {
            long low = f.Signed ? -(1L << (f.Bits - 1)) : 0, high = f.Signed ? (1L << (f.Bits - 1)) - 1 : (1L << f.Bits) - 1;
            if (f.Signed) b.L($"max.s32 {n}, {n}, {low};");
            b.L($"min.{nat} {n}, {n}, {high};");
        }

        Insert(b, f, n, i, word);
    }

    private static void Insert(PtxBuilder b, Fmt f, string code, int i, string word)
    {
        if (f.Bits == 32) b.L($"mov.b32 {word}, {code};");
        else b.L($"bfi.b32 {word}, {code}, {word}, {f.Bits * i}, {f.Bits};");
    }

    // ---------------------------------------------------------------- exp (the CPU's polynomial)

    public static string Exp(PtxBuilder b, Lane lane, string x)
    {
        if (!lane.Wide)
        {
            string c = b.F(), n = b.F(), r = b.F(), p = b.F(), ni = b.R(), half = b.R(), s1 = b.R(), s2 = b.R(), fs1 = b.F(), fs2 = b.F(), y = b.F(), pl = b.P(), ph = b.P(), pn = b.P();
            b.L($"max.f32 {c}, {x}, {Hex(-103.9721f)};");
            b.L($"min.f32 {c}, {c}, {Hex(88.72283f)};");
            b.L($"mul.rn.f32 {n}, {c}, {Hex(1.44269504f)};");
            b.L($"cvt.rni.f32.f32 {n}, {n};");
            b.L($"fma.rn.f32 {r}, {n}, {Hex(-0.693359375f)}, {c};");
            b.L($"fma.rn.f32 {r}, {n}, {Hex(2.12194440e-4f)}, {r};");
            b.L($"mov.f32 {p}, {Hex(1f / 5040)};");
            foreach (float k in new[] { 1f / 720, 1f / 120, 1f / 24, 1f / 6, 0.5f, 1f, 1f }) b.L($"fma.rn.f32 {p}, {p}, {r}, {Hex(k)};");
            b.L($"cvt.rni.s32.f32 {ni}, {n};");
            b.L($"shr.s32 {half}, {ni}, 1;");
            b.L($"add.s32 {s1}, {half}, 127;");
            b.L($"shl.b32 {s1}, {s1}, 23;");
            b.L($"sub.s32 {s2}, {ni}, {half};");
            b.L($"add.s32 {s2}, {s2}, 127;");
            b.L($"shl.b32 {s2}, {s2}, 23;");
            b.L($"mov.b32 {fs1}, {s1};");
            b.L($"mov.b32 {fs2}, {s2};");
            b.L($"mul.rn.f32 {y}, {p}, {fs1};");
            b.L($"mul.rn.f32 {y}, {y}, {fs2};");
            b.L($"setp.lt.f32 {pl}, {x}, {Hex(-103.9721f)};");
            b.L($"selp.f32 {y}, 0f00000000, {y}, {pl};");
            b.L($"setp.gt.f32 {ph}, {x}, {Hex(88.72283f)};");
            b.L($"selp.f32 {y}, 0f7F800000, {y}, {ph};");
            b.L($"testp.notanumber.f32 {pn}, {x};");
            b.L($"selp.f32 {y}, {x}, {y}, {pn};");
            return y;
        }
        else
        {
            string c = b.FD(), n = b.FD(), r = b.FD(), p = b.FD(), ni = b.RD(), half = b.RD(), s1 = b.RD(), s2 = b.RD(), fs1 = b.FD(), fs2 = b.FD(), y = b.FD(), pl = b.P(), ph = b.P(), pn = b.P();
            string D(double v) => Lane.F64.Literal(v);
            b.L($"max.f64 {c}, {x}, {D(-745.1332)};");
            b.L($"min.f64 {c}, {c}, {D(709.78)};");
            b.L($"mul.rn.f64 {n}, {c}, {D(1.4426950408889634)};");
            b.L($"cvt.rni.f64.f64 {n}, {n};");
            b.L($"fma.rn.f64 {r}, {n}, {D(-6.93147180369123816490e-01)}, {c};");
            b.L($"fma.rn.f64 {r}, {n}, {D(-1.90821492927058770002e-10)}, {r};");
            b.L($"mov.f64 {p}, {D(1.0 / 6227020800)};");
            foreach (double k in new[] { 1.0 / 479001600, 1.0 / 39916800, 1.0 / 3628800, 1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0, 1.0 })
                b.L($"fma.rn.f64 {p}, {p}, {r}, {D(k)};");
            b.L($"cvt.rni.s64.f64 {ni}, {n};");
            b.L($"shr.s64 {half}, {ni}, 1;");
            b.L($"add.s64 {s1}, {half}, 1023;");
            b.L($"shl.b64 {s1}, {s1}, 52;");
            b.L($"sub.s64 {s2}, {ni}, {half};");
            b.L($"add.s64 {s2}, {s2}, 1023;");
            b.L($"shl.b64 {s2}, {s2}, 52;");
            b.L($"mov.b64 {fs1}, {s1};");
            b.L($"mov.b64 {fs2}, {s2};");
            b.L($"mul.rn.f64 {y}, {p}, {fs1};");
            b.L($"mul.rn.f64 {y}, {y}, {fs2};");
            b.L($"setp.lt.f64 {pl}, {x}, {D(-745.1332)};");
            b.L($"selp.f64 {y}, 0d0000000000000000, {y}, {pl};");
            b.L($"setp.gt.f64 {ph}, {x}, {D(709.78)};");
            b.L($"selp.f64 {y}, 0d7FF0000000000000, {y}, {ph};");
            b.L($"testp.notanumber.f64 {pn}, {x};");
            b.L($"selp.f64 {y}, {x}, {y}, {pn};");
            return y;
        }
    }
}

/// <summary>The kernels.</summary>
internal static partial class Ptx
{
    private static string Header(Gpu gpu) => gpu.Target;

    // The thread's global index in %r (returns the register).
    private static string GlobalIndex(PtxBuilder b)
    {
        string tid = b.R(), ctaid = b.R(), ntid = b.R(), gid = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {ctaid}, %ctaid.x;");
        b.L($"mov.u32 {ntid}, %ntid.x;");
        b.L($"mad.lo.u32 {gid}, {ctaid}, {ntid}, {tid};");
        return gid;
    }

    private static string Param64(PtxBuilder b, string name)
    {
        string r = b.RD();
        b.L($"ld.param.u64 {r}, [{name}];");
        return r;
    }

    private static string Param32(PtxBuilder b, string name)
    {
        string r = b.R();
        b.L($"ld.param.u32 {r}, [{name}];");
        return r;
    }

    // ---------------------------------------------------------------- add

    /// <summary>c = a + b over n values: a thread per 16 bytes (a 64-bit format: two values), the lane the type's own.</summary>
    public static string Add(Gpu gpu, Fmt f, out string entry)
    {
        entry = "add_" + f.Name;
        var lane = f.Natural;
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), n = Param32(b, "p_n");
        int wordsPerThread = f.WordBits == 64 ? 2 : 4, wordBytes = f.WordBits / 8;
        string words = b.R(), w0 = b.R(), pEnd = b.P(), pWhole = b.P(), last = b.R();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else
        {
            string bits = b.RD(), wq = b.RD();
            b.L($"mul.wide.u32 {bits}, {n}, {f.Bits};");
            b.L($"add.u64 {bits}, {bits}, 31;");
            b.L($"shr.u64 {wq}, {bits}, 5;");
            b.L($"cvt.u32.u64 {words}, {wq};");
        }

        b.L($"mul.lo.u32 {w0}, {gid}, {wordsPerThread};");
        b.L($"setp.ge.u32 {pEnd}, {w0}, {words};");
        b.L($"@{pEnd} bra DONE;");
        string off = b.RD(), aa = b.RD(), ab = b.RD(), ac = b.RD();
        b.L($"mul.wide.u32 {off}, {w0}, {wordBytes};");
        b.L($"add.u64 {aa}, {pa}, {off};");
        b.L($"add.u64 {ab}, {pb}, {off};");
        b.L($"add.u64 {ac}, {pc}, {off};");
        b.L($"add.u32 {last}, {w0}, {wordsPerThread};");
        b.L($"setp.le.u32 {pWhole}, {last}, {words};");
        b.L($"@!{pWhole} bra TAIL;");

        // The whole 16 bytes at once.
        string wt = f.WordBits == 64 ? "b64" : "b32";
        var wa = new string[wordsPerThread];
        var wb = new string[wordsPerThread];
        var wc = new string[wordsPerThread];
        for (int j = 0; j < wordsPerThread; j++) { wa[j] = f.WordBits == 64 ? b.RD() : b.R(); wb[j] = f.WordBits == 64 ? b.RD() : b.R(); wc[j] = f.WordBits == 64 ? b.RD() : b.R(); }
        string vec = wordsPerThread == 4 ? "v4" : "v2";
        b.L($"ld.global.{vec}.{wt} {{{string.Join(", ", wa)}}}, [{aa}];");
        b.L($"ld.global.{vec}.{wt} {{{string.Join(", ", wb)}}}, [{ab}];");
        for (int j = 0; j < wordsPerThread; j++) AddWord(b, f, lane, wa[j], wb[j], wc[j]);
        b.L($"st.global.{vec}.{wt} [{ac}], {{{string.Join(", ", wc)}}};");
        b.L("bra DONE;");

        // The last few words, one by one.
        b.Label("TAIL");
        for (int j = 0; j < wordsPerThread; j++)
        {
            string p = b.P(), idx = b.R();
            b.L($"add.u32 {idx}, {w0}, {j};");
            b.L($"setp.ge.u32 {p}, {idx}, {words};");
            b.L($"@{p} bra DONE;");
            string ra = f.WordBits == 64 ? b.RD() : b.R(), rb = f.WordBits == 64 ? b.RD() : b.R(), rc = f.WordBits == 64 ? b.RD() : b.R();
            b.L($"ld.global.{wt} {ra}, [{aa}+{j * wordBytes}];");
            b.L($"ld.global.{wt} {rb}, [{ab}+{j * wordBytes}];");
            AddWord(b, f, lane, ra, rb, rc);
            b.L($"st.global.{wt} [{ac}+{j * wordBytes}], {rc};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_n");
    }

    private static void AddWord(PtxBuilder b, Fmt f, Lane lane, string wa, string wb, string wc)
    {
        if (f.WordBits == 32) b.L($"mov.b32 {wc}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string x = Codec.Unpack(b, f, wa, i, lane), y = Codec.Unpack(b, f, wb, i, lane), z = lane.NewReg(b);
            b.L(lane.Add(z, x, y));
            Codec.Pack(b, f, z, lane, i, wc);
        }
    }

    // ---------------------------------------------------------------- stored array ↔ lane array

    /// <summary>The n values of a stored array into a lane array: a thread per word.</summary>
    public static string ToLanes(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"cvt_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string src = Param64(b, "p_x"), dst = Param64(b, "p_y"), n = Param32(b, "p_n");
        string words = WordCount(b, f, n), p = b.P();
        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        string off = b.RD(), addr = b.RD(), w = f.WordBits == 64 ? b.RD() : b.R();
        b.L($"mul.wide.u32 {off}, {gid}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {src}, {off};");
        b.L($"ld.global.{wt} {w}, [{addr}];");
        string first = b.R(), outOff = b.RD(), outAddr = b.RD();
        b.L($"mul.lo.u32 {first}, {gid}, {f.Per};");
        b.L($"mul.wide.u32 {outOff}, {first}, {lane.Size};");
        b.L($"add.u64 {outAddr}, {dst}, {outOff};");
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R(), pi = b.P();
            b.L($"add.u32 {idx}, {first}, {i};");
            b.L($"setp.ge.u32 {pi}, {idx}, {n};");
            b.L($"@{pi} bra DONE;");
            string v = Codec.Unpack(b, f, w, i, lane);
            b.L($"st.global.{lane.Mem} [{outAddr}+{i * lane.Size}], {v};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_n");
    }

    /// <summary>A lane array's n values into a stored array: a thread per word (values past n in its last word zero).</summary>
    public static string FromLanes(Gpu gpu, Lane lane, Fmt f, out string entry)
    {
        entry = $"back_{lane.Name}_{f.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string src = Param64(b, "p_x"), dst = Param64(b, "p_y"), n = Param32(b, "p_n");
        string words = WordCount(b, f, n), p = b.P();
        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        string first = b.R(), inOff = b.RD(), inAddr = b.RD();
        b.L($"mul.lo.u32 {first}, {gid}, {f.Per};");
        b.L($"mul.wide.u32 {inOff}, {first}, {lane.Size};");
        b.L($"add.u64 {inAddr}, {src}, {inOff};");
        string w = f.WordBits == 64 ? b.RD() : b.R();
        if (f.WordBits == 32) b.L($"mov.b32 {w}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R(), pi = b.P(), v = lane.NewReg(b);
            b.L($"add.u32 {idx}, {first}, {i};");
            b.L($"setp.ge.u32 {pi}, {idx}, {n};");
            b.L($"mov.{lane.Reg} {v}, {lane.Zero};");
            b.L($"@!{pi} ld.global.{lane.Mem} {v}, [{inAddr}+{i * lane.Size}];");
            Codec.Pack(b, f, v, lane, i, w);
        }

        string off = b.RD(), addr = b.RD();
        b.L($"mul.wide.u32 {off}, {gid}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {dst}, {off};");
        b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {w};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_n");
    }

    private static string WordCount(PtxBuilder b, Fmt f, string n)
    {
        string words = b.R();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else
        {
            string bits = b.RD(), wq = b.RD();
            b.L($"mul.wide.u32 {bits}, {n}, {f.Bits};");
            b.L($"add.u64 {bits}, {bits}, 31;");
            b.L($"shr.u64 {wq}, {bits}, 5;");
            b.L($"cvt.u32.u64 {words}, {wq};");
        }

        return words;
    }

    // ---------------------------------------------------------------- operands in k-contiguous packed rows
    // The input kinds a product's operands are packed to: 8-bit integers four a word (dp4a and the s8/u8 tensor
    // instructions), 4-bit integers eight a word (the s4/u4 tensor instructions), f16 or bf16 two a word (the f16/bf16
    // tensor instructions; exact for every narrower float).

    public sealed record Kind(string Name, int Bits, Lane Acc, string Mma, int KStep)
    {
        public static readonly Kind S8 = new("s8", 8, Lane.S32, "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32", 16);
        public static readonly Kind U8 = new("u8", 8, Lane.U32, "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32", 16);
        public static readonly Kind S8K32 = new("s8", 8, Lane.S32, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", 32);
        public static readonly Kind U8K32 = new("u8", 8, Lane.U32, "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32", 32);
        public static readonly Kind S4 = new("s4", 4, Lane.S32, "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32", 32);
        public static readonly Kind U4 = new("u4", 4, Lane.U32, "mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32", 32);
        public static readonly Kind F16 = new("f16", 16, Lane.F32, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 16);
        public static readonly Kind BF16 = new("bf16", 16, Lane.F32, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 16);

        public int Per => 32 / Bits;

        public bool IsFloat => Bits == 16;

        /// <summary>16-byte chunks of a row one tensor instruction consumes (one, or two: a fragment then spans two chunks).</summary>
        public int StepChunks => KStep * Bits / 128;

        public string Tag => $"{Name}k{KStep}";

        /// <summary>Values in a 16-byte chunk of a packed row.</summary>
        public int ChunkValues => 128 / Bits;

        /// <summary>Values of k one block of the pipeline holds: a 64-byte row slice.</summary>
        public int BlockK => 512 / Bits;
    }

    /// <summary>k padded to the packed rows' length: a multiple of the largest block (128 values).</summary>
    public static int Padded(int k) => (k + 127) / 128 * 128;

    /// <summary>Rows of a stored matrix (or of its transpose) into kind-packed rows of kp values: Y[r][0..kp) = X[r][0..k) (zero past k).
    /// A thread per word; p_inner is the stored matrix's row length (k for A, n for the transpose of B).</summary>
    public static string PackRows(Gpu gpu, Fmt f, Kind kind, bool transposed, out string entry)
    {
        entry = $"pack_{f.Name}_{kind.Name}_{(transposed ? "t" : "n")}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string src = Param64(b, "p_x"), dst = Param64(b, "p_y"), rows = Param32(b, "p_rows"), k = Param32(b, "p_k"), kp = Param32(b, "p_kp"), inner = Param32(b, "p_inner");
        string wordsPerRow = b.R(), total = b.R(), p = b.P(), row = b.R(), w = b.R();
        b.L($"shr.u32 {wordsPerRow}, {kp}, {kind.Per switch { 8 => 3, 4 => 2, _ => 1 }};");
        b.L($"mul.lo.u32 {total}, {rows}, {wordsPerRow};");
        b.L($"setp.ge.u32 {p}, {gid}, {total};");
        b.L($"@{p} bra DONE;");
        b.L($"div.u32 {row}, {gid}, {wordsPerRow};");
        b.L($"rem.u32 {w}, {gid}, {wordsPerRow};");
        string word = b.R();
        b.L($"mov.b32 {word}, 0;");
        for (int t = 0; t < kind.Per; t++)
        {
            string kk = b.R(), pk = b.P(), idx = b.R();
            b.L($"mad.lo.u32 {kk}, {w}, {kind.Per}, {t};");
            b.L($"setp.ge.u32 {pk}, {kk}, {k};");
            b.L($"@{pk} bra NEXT{t};");
            if (transposed) b.L($"mad.lo.u32 {idx}, {kk}, {inner}, {row};");
            else b.L($"mad.lo.u32 {idx}, {row}, {inner}, {kk};");
            if (kind.IsFloat)
            {
                string v = Codec.UnpackAt(b, f, src, idx, Lane.F32), code = b.R();
                if (kind == Kind.F16)
                {
                    string h = b.H();
                    b.L($"cvt.rn.f16.f32 {h}, {v};");
                    b.L($"cvt.u32.u16 {code}, {h};");
                }
                else
                {
                    b.L($"mov.b32 {code}, {v};");
                    b.L($"shr.u32 {code}, {code}, 16;");
                }

                b.L($"bfi.b32 {word}, {code}, {word}, {16 * t}, 16;");
            }
            else
            {
                string v = Codec.UnpackAt(b, f, src, idx, f.Signed ? Lane.S32 : Lane.U32);
                b.L($"bfi.b32 {word}, {v}, {word}, {kind.Bits * t}, {kind.Bits};");
            }

            b.Label($"NEXT{t}");
        }

        string off = b.RD(), addr = b.RD();
        b.L($"mul.wide.u32 {off}, {gid}, 4;");
        b.L($"add.u64 {addr}, {dst}, {off};");
        b.L($"st.global.b32 [{addr}], {word};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_k, .param .u32 p_kp, .param .u32 p_inner");
    }

    // ---------------------------------------------------------------- C = A·Bᵀ on the tensor instructions
    // A m×kp and Bt n×kp packed rows; 128×BN of C per block of 8 warps (2 × 4; each 64 × BN/4 as 4 × BN/32 m16n8 tiles).
    // k walks in blocks of rowBytes a row through a ring of shared-memory stages filled by asynchronous copies (the
    // blocks ahead land while this one computes); fragments come from shared memory by ldmatrix, rows padded by 16
    // bytes so the eight rows a load touches sit in different banks, and the next k-step's fragments are loaded while
    // this step's tensor instructions run.

    /// <summary>Shared bytes the tensor kernel takes for a configuration: (128 + BN) rows per stage, padded by 16 bytes
    /// unless the chunks are swizzled instead.</summary>
    public static int MmaShared(int rowBytes, int stages, int bn, bool swizzle) => stages * (128 + bn) * (rowBytes + (swizzle ? 0 : 16));

    /// <summary>Static shared memory a kernel may declare without the device's opt-in.</summary>
    public const int StaticShared = 48 * 1024;

    // The chunk swizzle of a row: 64-byte rows (4 chunks) by (row / 2) mod 4, 128-byte rows (8 chunks) by row mod 8.
    private static void Swizzle(PtxBuilder b, string dst, string row, int rowBytes)
    {
        if (rowBytes == 64) { b.L($"shr.u32 {dst}, {row}, 1;"); b.L($"and.b32 {dst}, {dst}, 3;"); }
        else b.L($"and.b32 {dst}, {row}, 7;");
    }

    public static string GemmMma(Gpu gpu, Kind kind, int rowBytes, int stages, int bn, bool swizzle, out string entry)
    {
        entry = $"gemm_mma_{kind.Tag}_r{rowBytes}_s{stages}_n{bn}{(swizzle ? "_x" : "")}";
        const int BM = 128;
        int wn = bn / 4, nt = wn / 8;                                        // a warp's columns and its n-tiles
        // Rows padded by 16 bytes, or unpadded with each row's 16-byte chunks XOR-swizzled by the row (64-byte rows: by
        // row/2 mod 4; 128-byte rows: by row mod 8), so the eight rows a load touches sit in different banks either way.
        int stride = rowBytes + (swizzle ? 0 : 16), stageBytes = (BM + bn) * stride, totalShared = stages * stageBytes, bBase = BM * stride;
        int chunks = rowBytes / 16, copiesA = BM * chunks / GemmThreads, copiesB = bn * chunks / GemmThreads;
        int chunkShift = chunks switch { 4 => 2, 8 => 3, _ => throw new NotSupportedException("row bytes") };
        int stepBytes = kind.KStep * kind.Bits / 8, ksteps = rowBytes / stepBytes;
        int blockValues = rowBytes * 8 / kind.Bits, blockShift = (int)Math.Log2(blockValues);
        var acc = kind.Acc;
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), warp = b.R(), lane = b.R(), wmR = b.R(), wnR = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"shr.u32 {wmR}, {warp}, 2;");
        b.L($"and.b32 {wnR}, {warp}, 3;");
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), m = Param32(b, "p_m"), n = Param32(b, "p_n"), kp = Param32(b, "p_kp"), ld = Param32(b, "p_ld");
        string row0 = b.R(), col0 = b.R(), smem = b.R();
        b.L($"mul.lo.u32 {row0}, {by}, {BM};");
        b.L($"mul.lo.u32 {col0}, {bx}, {bn};");
        b.L($"mov.u32 {smem}, smem;");

        var c = new string[4, nt, 4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < nt; j++) for (int q = 0; q < 4; q++) { c[i, j, q] = acc.NewReg(b); b.L($"mov.{acc.Reg} {c[i, j, q]}, {acc.Zero};"); }

        // Copies: chunk tid + 256·j → row chunk / chunks, part chunk % chunks, for A (128 rows) and B (BN rows).
        // A row past m (B past n) copies nothing and zero-fills.
        string rowBytesK = b.RD();                                            // bytes a packed row takes (the stride)
        b.L($"mul.wide.u32 {rowBytesK}, {ld}, {kind.Bits};");
        b.L($"shr.u64 {rowBytesK}, {rowBytesK}, 3;");
        (string[] G, string[] S, string[] Sz) Copies(int count, string basePtr, string origin, string limit, int sharedBase)
        {
            var g = new string[count]; var s = new string[count]; var sz = new string[count];
            for (int j = 0; j < count; j++)
            {
                string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), p = b.P(), t64 = b.RD(), partOff = b.RD(), rowSel = b.R();
                b.L($"add.u32 {chunk}, {tid}, {j * GemmThreads};");
                b.L($"shr.u32 {row}, {chunk}, {chunkShift};");
                b.L($"and.b32 {part}, {chunk}, {chunks - 1};");
                b.L($"add.u32 {gRow}, {origin}, {row};");
                b.L($"setp.lt.u32 {p}, {gRow}, {limit};");
                b.L($"selp.u32 {rowSel}, {gRow}, 0, {p};");
                g[j] = b.RD(); s[j] = b.R(); sz[j] = b.R();
                b.L($"cvt.u64.u32 {t64}, {rowSel};");
                b.L($"mul.lo.u64 {t64}, {t64}, {rowBytesK};");
                b.L($"mul.wide.u32 {partOff}, {part}, 16;");
                b.L($"add.u64 {t64}, {t64}, {partOff};");
                b.L($"add.u64 {g[j]}, {basePtr}, {t64};");
                b.L($"mad.lo.u32 {s[j]}, {row}, {stride}, {smem};");
                if (swizzle)
                {
                    string swz = b.R(), partX = b.R();
                    Swizzle(b, swz, row, rowBytes);
                    b.L($"xor.b32 {partX}, {part}, {swz};");
                    b.L($"mad.lo.u32 {s[j]}, {partX}, 16, {s[j]};");
                }
                else b.L($"mad.lo.u32 {s[j]}, {part}, 16, {s[j]};");
                if (sharedBase != 0) b.L($"add.u32 {s[j]}, {s[j]}, {sharedBase};");
                b.L($"selp.u32 {sz[j]}, 16, 0, {p};");
            }

            return (g, s, sz);
        }

        var ca = Copies(copiesA, pa, row0, m, 0);
        var cb = Copies(copiesB, pb, col0, n, bBase);
        string blocksReg = b.R();
        b.L($"shr.u32 {blocksReg}, {kp}, {blockShift};");

        // Issues block `block` into stage `stageOff` (bytes) when the block exists; commits a group either way.
        void Issue(string block, string stageOff, string label)
        {
            string pHas = b.P(), kBytes = b.RD();
            b.L($"setp.ge.u32 {pHas}, {block}, {blocksReg};");
            b.L($"@{pHas} bra {label};");
            b.L($"mul.wide.u32 {kBytes}, {block}, {rowBytes};");
            foreach (var (g, s, sz) in new[] { ca, cb })
            {
                for (int j = 0; j < g.Length; j++)
                {
                    string src = b.RD(), dst = b.R();
                    b.L($"add.u64 {src}, {g[j]}, {kBytes};");
                    b.L($"add.u32 {dst}, {s[j]}, {stageOff};");
                    b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {sz[j]};");
                }
            }

            b.Label(label);
            b.L("cp.async.commit_group;");
        }

        // Fragment addresses for ldmatrix: each lane's row address, plus the chunk its lane supplies within a k-step
        // (f16: lane / 16 for A, (lane / 8) & 1 for B; 8-bit: none) XORed with the row's swizzle. Swizzled, the k-step's
        // chunk is XORed in at use (a k-step's chunk count is even, so adding it is XORing it).
        string aAddr = b.R(), bAddr = b.R(), t1 = b.R(), t2 = b.R(), aChunk = b.R(), bChunk = b.R(), aRow = b.R(), bRow = b.R();
        b.L($"and.b32 {t1}, {lane}, 15;");
        b.L($"mad.lo.u32 {aRow}, {wmR}, 64, {t1};");
        b.L($"mad.lo.u32 {aAddr}, {aRow}, {stride}, {smem};");
        b.L($"shr.u32 {aChunk}, {lane}, 4;");
        if (kind.StepChunks == 2)
        {
            // Four matrices: (n 0-7, chunk 0), (n 0-7, chunk 1), (n 8-15, chunk 0), (n 8-15, chunk 1): b0, b1 of two n-tiles.
            b.L($"and.b32 {t1}, {lane}, 7;");
            b.L($"shr.u32 {t2}, {lane}, 4;");
            b.L($"mad.lo.u32 {t1}, {t2}, 8, {t1};");
            b.L($"mad.lo.u32 {bRow}, {wnR}, {wn}, {t1};");
            b.L($"shr.u32 {bChunk}, {lane}, 3;");
            b.L($"and.b32 {bChunk}, {bChunk}, 1;");
        }
        else
        {
            // Four matrices: n 0-7, 8-15, 16-23, 24-31 at the same chunk: b0 of four n-tiles.
            b.L($"mad.lo.u32 {bRow}, {wnR}, {wn}, {lane};");
            b.L($"mov.u32 {bChunk}, 0;");
        }

        b.L($"mad.lo.u32 {bAddr}, {bRow}, {stride}, {smem};");
        b.L($"add.u32 {bAddr}, {bAddr}, {bBase};");
        if (swizzle)
        {
            string sa = b.R(), sb2 = b.R();
            Swizzle(b, sa, aRow, rowBytes);
            Swizzle(b, sb2, bRow, rowBytes);
            b.L($"xor.b32 {aChunk}, {aChunk}, {sa};");
            b.L($"xor.b32 {bChunk}, {bChunk}, {sb2};");
        }
        else
        {
            b.L($"mad.lo.u32 {aAddr}, {aChunk}, 16, {aAddr};");
            b.L($"mad.lo.u32 {bAddr}, {bChunk}, 16, {bAddr};");
        }

        // The address of a fragment's row at k-step chunk `chunk` (an even immediate for f16, any for 8-bit): swizzled, the
        // per-lane chunk term XOR the step's chunk, times 16, from the row address; padded, an immediate offset.
        string FragAddr(string stageAddr, string chunkTerm, int rowOffset, int chunk)
        {
            if (!swizzle) return $"[{stageAddr}+{rowOffset + chunk * 16}]";
            string t = b.R(), a2 = b.R();
            b.L($"xor.b32 {t}, {chunkTerm}, {chunk};");
            b.L($"mad.lo.u32 {a2}, {t}, 16, {stageAddr};");
            return $"[{a2}+{rowOffset}]";
        }

        // Fragment registers, two sets (this k-step's and the next's).
        var af = new string[2][][];
        var bf = new string[2][][];
        for (int set = 0; set < 2; set++)
        {
            af[set] = new string[4][];
            bf[set] = new string[nt][];
            for (int i = 0; i < 4; i++) af[set][i] = kind.StepChunks == 2 ? [b.R(), b.R(), b.R(), b.R()] : [b.R(), b.R()];
            for (int j = 0; j < nt; j++) bf[set][j] = kind.StepChunks == 2 ? [b.R(), b.R()] : [b.R()];
        }

        void LoadFrags(int set, string aStage, string bStage, int ks)
        {
            // Rows i·16 (A) or j·8 (B) further down keep the row's swizzle class only when the offset is a multiple of 8 rows: it is.
            if (kind.StepChunks == 2)
            {
                for (int i = 0; i < 4; i++)
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", af[set][i])}}}, {FragAddr(aStage, aChunk, i * 16 * stride, ks * 2)};");
                for (int j = 0; j < nt; j += 2)
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bf[set][j][0]}, {bf[set][j][1]}, {bf[set][j + 1][0]}, {bf[set][j + 1][1]}}}, {FragAddr(bStage, bChunk, j * 8 * stride, ks * 2)};");
            }
            else
            {
                for (int i = 0; i < 4; i++)
                    b.L($"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{string.Join(", ", af[set][i])}}}, {FragAddr(aStage, aChunk, i * 16 * stride, ks)};");
                for (int j = 0; j < nt; j += 4)
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bf[set][j][0]}, {bf[set][j + 1][0]}, {bf[set][j + 2][0]}, {bf[set][j + 3][0]}}}, {FragAddr(bStage, bChunk, j * 8 * stride, ks)};");
            }
        }

        void Compute(int set)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < nt; j++)
                {
                    string d = $"{{{c[i, j, 0]}, {c[i, j, 1]}, {c[i, j, 2]}, {c[i, j, 3]}}}";
                    b.L($"{kind.Mma} {d}, {{{string.Join(", ", af[set][i])}}}, {{{string.Join(", ", bf[set][j])}}}, {d};");
                }
            }
        }

        // The pipeline: stages 0 .. S-2 issued ahead; each block waits for its own, computes, then issues block + S - 1.
        string kb = b.R(), stage = b.R(), issueBlock = b.R(), issueStage = b.R(), pLoop = b.P(), pWrap = b.P(), pWrap2 = b.P();
        for (int sIdx = 0; sIdx < stages - 1; sIdx++)
        {
            string blk = b.R(), off = b.R();
            b.L($"mov.u32 {blk}, {sIdx};");
            b.L($"mov.u32 {off}, {sIdx * stageBytes};");
            Issue(blk, off, $"PRO{sIdx}");
        }

        b.L($"mov.u32 {kb}, 0;");
        b.L($"mov.u32 {stage}, 0;");
        b.L($"mov.u32 {issueBlock}, {stages - 1};");
        b.L($"mov.u32 {issueStage}, {(stages - 1) * stageBytes};");
        b.Label("KLOOP");
        b.L($"setp.ge.u32 {pLoop}, {kb}, {blocksReg};");
        b.L($"@{pLoop} bra KDONE;");
        b.L($"cp.async.wait_group {stages - 2};");
        b.L("bar.sync 0;");
        string aStage = b.R(), bStage = b.R();
        b.L($"add.u32 {aStage}, {aAddr}, {stage};");
        b.L($"add.u32 {bStage}, {bAddr}, {stage};");
        LoadFrags(0, aStage, bStage, 0);
        for (int ks = 0; ks < ksteps; ks++)
        {
            if (ks + 1 < ksteps) LoadFrags((ks + 1) % 2, aStage, bStage, ks + 1);
            Compute(ks % 2);
        }

        b.L("bar.sync 0;");
        Issue(issueBlock, issueStage, "ISSUE");
        b.L($"add.u32 {issueBlock}, {issueBlock}, 1;");
        b.L($"add.u32 {issueStage}, {issueStage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap}, {issueStage}, {totalShared};");
        b.L($"@{pWrap} mov.u32 {issueStage}, 0;");
        b.L($"add.u32 {stage}, {stage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap2}, {stage}, {totalShared};");
        b.L($"@{pWrap2} mov.u32 {stage}, 0;");
        b.L($"add.u32 {kb}, {kb}, 1;");
        b.L("bra KLOOP;");
        b.Label("KDONE");
        b.L("cp.async.wait_group 0;");

        // C: rows row0 + wm·64 + i·16 + lane/4 (+8), columns col0 + wn·WN + j·8 + 2·(lane%4) (+1).
        string l4 = b.R(), l3 = b.R(), rBase = b.R(), cBase = b.R();
        b.L($"shr.u32 {l4}, {lane}, 2;");
        b.L($"and.b32 {l3}, {lane}, 3;");
        b.L($"mad.lo.u32 {rBase}, {wmR}, 64, {row0};");
        b.L($"add.u32 {rBase}, {rBase}, {l4};");
        b.L($"mad.lo.u32 {cBase}, {wnR}, {wn}, {col0};");
        b.L($"mad.lo.u32 {cBase}, {l3}, 2, {cBase};");
        for (int i = 0; i < 4; i++)
        {
            for (int half = 0; half < 2; half++)
            {
                string row = b.R(), pr = b.P(), rowOff = b.RD();
                b.L($"add.u32 {row}, {rBase}, {i * 16 + half * 8};");
                b.L($"setp.lt.u32 {pr}, {row}, {m};");
                b.L($"mul.wide.u32 {rowOff}, {row}, {n};");
                for (int j = 0; j < nt; j++)
                {
                    string col = b.R(), pcol = b.P(), p2 = b.P(), idx = b.RD(), addr = b.RD(), col2 = b.R(), pOnly = b.P();
                    b.L($"add.u32 {col}, {cBase}, {j * 8};");
                    b.L($"add.u32 {col2}, {col}, 1;");
                    b.L($"setp.lt.u32 {pcol}, {col}, {n};");
                    b.L($"and.pred {pcol}, {pcol}, {pr};");
                    b.L($"setp.lt.u32 {p2}, {col2}, {n};");
                    b.L($"and.pred {p2}, {p2}, {pcol};");
                    b.L($"not.pred {pOnly}, {p2};");
                    b.L($"and.pred {pOnly}, {pOnly}, {pcol};");
                    b.L($"cvt.u64.u32 {idx}, {col};");
                    b.L($"add.u64 {idx}, {idx}, {rowOff};");
                    b.L($"shl.b64 {idx}, {idx}, 2;");
                    b.L($"add.u64 {addr}, {pc}, {idx};");
                    b.L($"@{p2} st.global.v2.{acc.Mem} [{addr}], {{{c[i, j, half * 2]}, {c[i, j, half * 2 + 1]}}};");
                    b.L($"@{pOnly} st.global.{acc.Mem} [{addr}], {c[i, j, half * 2]};");
                }
            }
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_kp, .param .u32 p_ld", totalShared);
    }

    // ---------------------------------------------------------------- float64 products on the 8-bit tensor instructions, exactly
    // Each value is scaled by the power of two of its row's (A) or column's (B) largest magnitude and rounded to a 62-bit
    // integer, which is written as eight signed 8-bit digits (balanced base 256). A product of two such values is a sum
    // of digit products weighted by powers of 2^-8; all pairs with the same weight (a diagonal d = t + u) form one
    // integer product over a k that is (d + 1) times the original, computed on the tensor instructions in 32-bit
    // integers, which cannot overflow (|digit| ≤ 128, so a product ≤ 2^14, and the terms are counted). Twelve
    // diagonals are kept (the fifteen there are, less the three below 2^-96 of the scale, which no result the proof
    // covers can feel), combined in 128-bit integer arithmetic and rounded once, to nearest even, into the float64 result.

    /// <summary>Digits a value is sliced into (8 bits each: 64 bits, 62 of them significant).</summary>
    public const int Slices = 8;

    /// <summary>Diagonals (digit-product weights) kept: d = t + u from 0 to Diagonals − 1.</summary>
    public const int Diagonals = 12;

    private static string Exponent(PtxBuilder b, string absMax)
    {
        // floor(log2 x) + 3 for a normal x (a subnormal or zero: as if the smallest normal), held to what the slicing scale can take.
        string bits = b.RD(), exp = b.R(), e = b.R(), p = b.P();
        b.L($"mov.b64 {bits}, {absMax};");
        b.L($"shr.u64 {bits}, {bits}, 52;");
        b.L($"cvt.u32.u64 {exp}, {bits};");
        b.L($"and.b32 {exp}, {exp}, 0x7FF;");
        b.L($"setp.eq.u32 {p}, {exp}, 0;");
        b.L($"@{p} mov.u32 {exp}, 1;");
        b.L($"sub.s32 {e}, {exp}, 1020;");
        b.L($"max.s32 {e}, {e}, -959;");
        return e;
    }

    /// <summary>e[i] for each row of A (m × k float64): the slicing exponent of the row's largest magnitude; a block per row.</summary>
    public static string RowExponents(Gpu gpu, out string entry)
    {
        entry = "rowexp_f64";
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string pa = Param64(b, "p_a"), pe = Param64(b, "p_e"), k = Param32(b, "p_k");
        string rowBase = b.RD(), off = b.RD(), max = b.FD(), p2 = b.R(), pEnd = b.P();
        b.L($"mul.wide.u32 {rowBase}, {row}, {k};");
        b.L($"shl.b64 {rowBase}, {rowBase}, 3;");
        b.L($"add.u64 {rowBase}, {pa}, {rowBase};");
        b.L($"mov.f64 {max}, 0d0000000000000000;");
        b.L($"mov.u32 {p2}, {tid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {p2}, {k};");
        b.L($"@{pEnd} bra DONE;");
        string v = b.FD(), addr = b.RD();
        b.L($"mul.wide.u32 {off}, {p2}, 8;");
        b.L($"add.u64 {addr}, {rowBase}, {off};");
        b.L($"ld.global.f64 {v}, [{addr}];");
        b.L($"abs.f64 {v}, {v};");
        b.L($"max.f64 {max}, {max}, {v};");
        b.L($"add.u32 {p2}, {p2}, 256;");
        b.L("bra LOOP;");
        b.Label("DONE");
        string all = BlockReduce(b, Lane.F64, tid, max, "max", 256);
        string p0 = b.P();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra OUT;");
        string e = Exponent(b, all), eAddr = b.RD();
        b.L($"mul.wide.u32 {eAddr}, {row}, 4;");
        b.L($"add.u64 {eAddr}, {pe}, {eAddr};");
        b.L($"st.global.s32 [{eAddr}], {e};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_e, .param .u32 p_k", ReduceBytes);
    }

    /// <summary>f[j] for each column of B (k × n float64): a thread per column.</summary>
    public static string ColumnExponents(Gpu gpu, out string entry)
    {
        entry = "colexp_f64";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pb = Param64(b, "p_b"), pf = Param64(b, "p_f"), k = Param32(b, "p_k"), n = Param32(b, "p_n");
        string pEnd = b.P(), max = b.FD(), p2 = b.R(), stride = b.RD(), addr = b.RD(), pLoop = b.P();
        b.L($"setp.ge.u32 {pEnd}, {gid}, {n};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"mov.f64 {max}, 0d0000000000000000;");
        b.L($"mul.wide.u32 {stride}, {n}, 8;");
        b.L($"mul.wide.u32 {addr}, {gid}, 8;");
        b.L($"add.u64 {addr}, {pb}, {addr};");
        b.L($"mov.u32 {p2}, 0;");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pLoop}, {p2}, {k};");
        b.L($"@{pLoop} bra DONE;");
        string v = b.FD();
        b.L($"ld.global.f64 {v}, [{addr}];");
        b.L($"abs.f64 {v}, {v};");
        b.L($"max.f64 {max}, {max}, {v};");
        b.L($"add.u64 {addr}, {addr}, {stride};");
        b.L($"add.u32 {p2}, {p2}, 1;");
        b.L("bra LOOP;");
        b.Label("DONE");
        string e = Exponent(b, max), eAddr = b.RD();
        b.L($"mul.wide.u32 {eAddr}, {gid}, 4;");
        b.L($"add.u64 {eAddr}, {pf}, {eAddr};");
        b.L($"st.global.s32 [{eAddr}], {e};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_b, .param .u64 p_f, .param .u32 p_k, .param .u32 p_n");
    }

    // The value's 62-bit integer Y = round(v · 2^(64 - e)) and its eight balanced digits, least significant first.
    private static string[] Digits(PtxBuilder b, string v, string e)
    {
        string scaleBits = b.RD(), scale = b.FD(), scaled = b.FD(), y = b.RD(), t32 = b.R();
        b.L($"sub.s32 {t32}, 1087, {e};");
        b.L($"cvt.u64.u32 {scaleBits}, {t32};");
        b.L($"shl.b64 {scaleBits}, {scaleBits}, 52;");
        b.L($"mov.b64 {scale}, {scaleBits};");
        b.L($"mul.f64 {scaled}, {v}, {scale};");
        b.L($"cvt.rni.s64.f64 {y}, {scaled};");
        var digits = new string[Slices];
        for (int t = Slices - 1; t >= 0; t--)
        {
            string d = b.RD(), t1 = b.RD();
            b.L($"add.s64 {t1}, {y}, 128;");
            b.L($"and.b64 {t1}, {t1}, 255;");
            b.L($"sub.s64 {d}, {t1}, 128;");
            b.L($"sub.s64 {y}, {y}, {d};");
            b.L($"shr.s64 {y}, {y}, 8;");
            digits[t] = d;
        }

        return digits;
    }

    /// <summary>A's values into packed rows of digits: S[i][t·kpad + p] = digit t of A[i][p] (t = 0 the most significant); a thread per value.</summary>
    public static string SliceRows(Gpu gpu, out string entry)
    {
        entry = "slice_rows_f64";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pa = Param64(b, "p_a"), pe = Param64(b, "p_e"), ps = Param64(b, "p_s"), m = Param32(b, "p_m"), k = Param32(b, "p_k"), kpad = Param32(b, "p_kpad");
        string total = b.R(), pEnd = b.P(), i = b.R(), p2 = b.R(), eAddr = b.RD(), e = b.R(), aAddr = b.RD(), v = b.FD(), rowLen = b.R(), sBase = b.RD();
        b.L($"mul.lo.u32 {total}, {m}, {k};");
        b.L($"setp.ge.u32 {pEnd}, {gid}, {total};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"div.u32 {i}, {gid}, {k};");
        b.L($"rem.u32 {p2}, {gid}, {k};");
        b.L($"mul.wide.u32 {eAddr}, {i}, 4;");
        b.L($"add.u64 {eAddr}, {pe}, {eAddr};");
        b.L($"ld.global.s32 {e}, [{eAddr}];");
        b.L($"mul.wide.u32 {aAddr}, {gid}, 8;");
        b.L($"add.u64 {aAddr}, {pa}, {aAddr};");
        b.L($"ld.global.f64 {v}, [{aAddr}];");
        var digits = Digits(b, v, e);
        b.L($"mul.lo.u32 {rowLen}, {kpad}, {Slices};");
        b.L($"mul.wide.u32 {sBase}, {i}, {rowLen};");
        b.L($"add.u64 {sBase}, {ps}, {sBase};");
        string pOff = b.RD();
        b.L($"cvt.u64.u32 {pOff}, {p2};");
        b.L($"add.u64 {sBase}, {sBase}, {pOff};");
        string kpad64 = b.RD();
        b.L($"cvt.u64.u32 {kpad64}, {kpad};");
        for (int t = 0; t < Slices; t++)
        {
            string h = b.H();
            b.L($"cvt.u16.u64 {h}, {digits[t]};");
            b.L($"st.global.b8 [{sBase}], {h};");
            if (t + 1 < Slices) b.L($"add.u64 {sBase}, {sBase}, {kpad64};");
        }

        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_e, .param .u64 p_s, .param .u32 p_m, .param .u32 p_k, .param .u32 p_kpad");
    }

    /// <summary>B's values into stacked rows of digits: Bs[(Slices − 1 − t)·kpad + p][j] = digit t of B[p][j]; a thread per value, j fastest.</summary>
    public static string SliceColumns(Gpu gpu, out string entry)
    {
        entry = "slice_cols_f64";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pb = Param64(b, "p_b"), pf = Param64(b, "p_f"), ps = Param64(b, "p_s"), k = Param32(b, "p_k"), n = Param32(b, "p_n"), kpad = Param32(b, "p_kpad");
        string total = b.R(), pEnd = b.P(), p2 = b.R(), j = b.R(), fAddr = b.RD(), e = b.R(), bAddr = b.RD(), v = b.FD();
        b.L($"mul.lo.u32 {total}, {k}, {n};");
        b.L($"setp.ge.u32 {pEnd}, {gid}, {total};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"div.u32 {p2}, {gid}, {n};");
        b.L($"rem.u32 {j}, {gid}, {n};");
        b.L($"mul.wide.u32 {fAddr}, {j}, 4;");
        b.L($"add.u64 {fAddr}, {pf}, {fAddr};");
        b.L($"ld.global.s32 {e}, [{fAddr}];");
        b.L($"mul.wide.u32 {bAddr}, {gid}, 8;");
        b.L($"add.u64 {bAddr}, {pb}, {bAddr};");
        b.L($"ld.global.f64 {v}, [{bAddr}];");
        var digits = Digits(b, v, e);
        // Row (Slices − 1 − t)·kpad + p of the stacked matrix, column j.
        string blockRows = b.RD(), rowIdx = b.RD(), sAddr = b.RD(), n64 = b.RD(), p64 = b.RD(), j64 = b.RD();
        b.L($"cvt.u64.u32 {n64}, {n};");
        b.L($"cvt.u64.u32 {p64}, {p2};");
        b.L($"cvt.u64.u32 {j64}, {j};");
        b.L($"cvt.u64.u32 {blockRows}, {kpad};");
        b.L($"mul.lo.u64 {blockRows}, {blockRows}, {n64};");                  // bytes per slice block
        for (int t = 0; t < Slices; t++)
        {
            string h = b.H();
            b.L($"mul.lo.u64 {rowIdx}, {p64}, {n64};");
            b.L($"add.u64 {rowIdx}, {rowIdx}, {j64};");
            b.L($"mad.lo.u64 {sAddr}, {blockRows}, {Slices - 1 - t}, {rowIdx};");
            b.L($"add.u64 {sAddr}, {ps}, {sAddr};");
            b.L($"cvt.u16.u64 {h}, {digits[t]};");
            b.L($"st.global.b8 [{sAddr}], {h};");
        }

        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_b, .param .u64 p_f, .param .u64 p_s, .param .u32 p_k, .param .u32 p_n, .param .u32 p_kpad");
    }

    /// <summary>C[i][j] = round(Σ_d D_d[i][j] · 2^(8(Diagonals−1−d))) · 2^(e_i + f_j − 16 − 8(Diagonals−1)): the diagonals' integers
    /// summed in 128 bits, rounded once to nearest even into float64, scaled by the exact power of two; a thread per value.</summary>
    public static string CombineSlices(Gpu gpu, out string entry)
    {
        entry = "combine_f64x";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pd = Param64(b, "p_d"), pe = Param64(b, "p_e"), pf = Param64(b, "p_f"), pc = Param64(b, "p_c"), m = Param32(b, "p_m"), n = Param32(b, "p_n");
        string total = b.R(), pEnd = b.P(), i = b.R(), j = b.R(), e = b.R(), f = b.R(), tAddr = b.RD();
        b.L($"mul.lo.u32 {total}, {m}, {n};");
        b.L($"setp.ge.u32 {pEnd}, {gid}, {total};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"div.u32 {i}, {gid}, {n};");
        b.L($"rem.u32 {j}, {gid}, {n};");
        b.L($"mul.wide.u32 {tAddr}, {i}, 4;");
        b.L($"add.u64 {tAddr}, {pe}, {tAddr};");
        b.L($"ld.global.s32 {e}, [{tAddr}];");
        b.L($"mul.wide.u32 {tAddr}, {j}, 4;");
        b.L($"add.u64 {tAddr}, {pf}, {tAddr};");
        b.L($"ld.global.s32 {f}, [{tAddr}];");

        // S = Σ_d D_d << 8(Diagonals − 1 − d), in 128 bits (lo unsigned, hi signed).
        string lo = b.RD(), hi = b.RD(), dAddr = b.RD(), stride = b.RD();
        b.L($"mov.b64 {lo}, 0;");
        b.L($"mov.b64 {hi}, 0;");
        b.L($"mul.wide.u32 {stride}, {total}, 4;");
        b.L($"mul.wide.u32 {dAddr}, {gid}, 4;");
        b.L($"add.u64 {dAddr}, {pd}, {dAddr};");
        for (int d = 0; d < Diagonals; d++)
        {
            string d32 = b.R(), d64 = b.RD(), tlo = b.RD(), thi = b.RD();
            int sh = 8 * (Diagonals - 1 - d);
            b.L($"ld.global.s32 {d32}, [{dAddr}];");
            b.L($"cvt.s64.s32 {d64}, {d32};");
            if (sh == 0)
            {
                b.L($"mov.b64 {tlo}, {d64};");
                b.L($"shr.s64 {thi}, {d64}, 63;");
            }
            else if (sh < 64)
            {
                b.L($"shl.b64 {tlo}, {d64}, {sh};");
                b.L($"shr.s64 {thi}, {d64}, {64 - sh};");
            }
            else
            {
                b.L($"mov.b64 {tlo}, 0;");
                b.L($"shl.b64 {thi}, {d64}, {sh - 64};");
            }

            b.L($"add.cc.u64 {lo}, {lo}, {tlo};");
            b.L($"addc.s64 {hi}, {hi}, {thi};");
            if (d + 1 < Diagonals) b.L($"add.u64 {dAddr}, {dAddr}, {stride};");
        }

        // Sign and magnitude.
        string neg = b.P(), nlo = b.RD(), nhi = b.RD();
        b.L($"setp.lt.s64 {neg}, {hi}, 0;");
        b.L($"not.b64 {nlo}, {lo};");
        b.L($"not.b64 {nhi}, {hi};");
        b.L($"add.cc.u64 {nlo}, {nlo}, 1;");
        b.L($"addc.s64 {nhi}, {nhi}, 0;");
        b.L($"selp.b64 {lo}, {nlo}, {lo}, {neg};");
        b.L($"selp.b64 {hi}, {nhi}, {hi}, {neg};");

        // Bits in the magnitude; the top 53 with the rest rounded to nearest even.
        string clzHi = b.R(), clzLo = b.R(), nbits = b.R(), pHi = b.P(), shift = b.R(), mant = b.RD(), round = b.RD(), sticky = b.RD(), pSmall = b.P();
        b.L($"clz.b64 {clzHi}, {hi};");
        b.L($"clz.b64 {clzLo}, {lo};");
        b.L($"setp.ne.s64 {pHi}, {hi}, 0;");
        b.L($"sub.u32 {nbits}, 128, {clzHi};");
        b.L($"sub.u32 {clzLo}, 64, {clzLo};");
        b.L($"selp.u32 {nbits}, {nbits}, {clzLo}, {pHi};");
        b.L($"setp.le.u32 {pSmall}, {nbits}, 53;");
        b.L($"mov.u32 {shift}, 0;");
        b.L($"mov.b64 {mant}, {lo};");
        b.L($"@{pSmall} bra SCALE;");
        b.L($"sub.u32 {shift}, {nbits}, 53;");
        string pLow = b.P(), s1 = b.R(), a1 = b.RD(), a2 = b.RD(), one = b.RD(), mask = b.RD(), s2 = b.R(), pZero = b.P();
        b.L($"setp.lt.u32 {pLow}, {shift}, 64;");
        b.L($"@!{pLow} bra WIDE;");
        // shift in 1..63: mant = (hi << (64 − shift)) | (lo >> shift); dropped bits are lo's low `shift` bits.
        b.L($"sub.u32 {s1}, 64, {shift};");
        b.L($"shl.b64 {a1}, {hi}, {s1};");
        b.L($"shr.u64 {a2}, {lo}, {shift};");
        b.L($"or.b64 {mant}, {a1}, {a2};");
        b.L($"sub.u32 {s1}, {shift}, 1;");
        b.L($"shr.u64 {round}, {lo}, {s1};");
        b.L($"and.b64 {round}, {round}, 1;");
        b.L($"mov.b64 {one}, 1;");
        b.L($"shl.b64 {mask}, {one}, {s1};");
        b.L($"sub.u64 {mask}, {mask}, 1;");
        b.L($"and.b64 {sticky}, {lo}, {mask};");
        b.L("bra ROUND;");
        b.Label("WIDE");
        // shift in 64..75: mant = hi >> (shift − 64); dropped: hi's low (shift − 64) bits and all of lo.
        b.L($"sub.u32 {s2}, {shift}, 64;");
        b.L($"shr.u64 {mant}, {hi}, {s2};");
        b.L($"setp.eq.u32 {pZero}, {s2}, 0;");
        string rh = b.RD(), sh2 = b.RD(), rl = b.RD(), sl = b.RD(), s3 = b.R();
        b.L($"sub.u32 {s3}, {s2}, 1;");
        b.L($"shr.u64 {rh}, {hi}, {s3};");
        b.L($"and.b64 {rh}, {rh}, 1;");
        b.L($"shl.b64 {mask}, {one}, {s3};");
        b.L($"sub.u64 {mask}, {mask}, 1;");
        b.L($"and.b64 {sh2}, {hi}, {mask};");
        b.L($"or.b64 {sh2}, {sh2}, {lo};");
        b.L($"shr.u64 {rl}, {lo}, 63;");
        b.L($"and.b64 {sl}, {lo}, 0x7FFFFFFFFFFFFFFF;");
        b.L($"selp.b64 {round}, {rl}, {rh}, {pZero};");
        b.L($"selp.b64 {sticky}, {sl}, {sh2}, {pZero};");
        b.Label("ROUND");
        string odd = b.RD(), up = b.RD(), pSticky = b.P();
        b.L($"and.b64 {odd}, {mant}, 1;");
        b.L($"setp.ne.u64 {pSticky}, {sticky}, 0;");
        b.L($"selp.b64 {up}, 1, {odd}, {pSticky};");
        b.L($"and.b64 {up}, {up}, {round};");
        b.L($"add.u64 {mant}, {mant}, {up};");
        b.Label("SCALE");
        string val = b.FD(), expo = b.R(), scaleBits = b.RD(), scale = b.FD(), cAddr = b.RD();
        b.L($"cvt.rn.f64.u64 {val}, {mant};");
        b.L($"add.s32 {expo}, {e}, {f};");
        b.L($"add.s32 {expo}, {expo}, {shift};");
        b.L($"add.s32 {expo}, {expo}, {1023 - 16 - 8 * (Diagonals - 1)};");
        b.L($"cvt.u64.u32 {scaleBits}, {expo};");
        b.L($"shl.b64 {scaleBits}, {scaleBits}, 52;");
        b.L($"mov.b64 {scale}, {scaleBits};");
        b.L($"mul.f64 {val}, {val}, {scale};");
        b.L($"@{neg} neg.f64 {val}, {val};");
        b.L($"mul.wide.u32 {cAddr}, {gid}, 8;");
        b.L($"add.u64 {cAddr}, {pc}, {cAddr};");
        b.L($"st.global.f64 [{cAddr}], {val};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_d, .param .u64 p_e, .param .u64 p_f, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n");
    }

    // ---------------------------------------------------------------- C = A·B in a lane (A m×k, B k×n, row-major lane arrays)

    public const int GemmThreads = 256;

    /// <summary>Tiles of micro·16 × micro·16 values of C per block of 256 threads (micro² per thread), k in steps of 8 through shared memory.</summary>
    /// <summary>With conv (item 4g) B is the convolution's columns read from the image at p_b (C × H × W, in the lane's
    /// storage): the B tile's loads gather x[c][oh·sh − ph + kh·dh][ow·sw − pw + kw·dw] (0 off the image) for row
    /// kk = (c, kh, kw) and column j = (oh, ow) — the implicit im2col; the rest of the kernel unchanged.</summary>
    public static string Gemm(Gpu gpu, Lane lane, int micro, out string entry, bool conv = false)
    {
        int tile = micro * 16, bk = 16, e = lane.Size;
        entry = conv ? $"gemmconv{tile}_{lane.Name}" : $"gemm{tile}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), tx = b.R(), ty = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"and.b32 {tx}, {tid}, 15;");
        b.L($"shr.u32 {ty}, {tid}, 4;");
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), m = Param32(b, "p_m"), n = Param32(b, "p_n"), k = Param32(b, "p_k");
        string[] g = conv ? [.. new[] { "h", "w", "kh", "kw", "sh", "sw", "ph", "pw", "dh", "dw", "ow" }.Select(name => Param32(b, "p_" + name))] : [];
        string row0 = b.R(), col0 = b.R();
        b.L($"mul.lo.u32 {row0}, {by}, {tile};");
        b.L($"mul.lo.u32 {col0}, {bx}, {tile};");
        int asBytes = bk * tile * e, bsBytes = bk * tile * e;
        string smem = b.R(), asBase = b.R(), bsBase = b.R();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mov.u32 {asBase}, {smem};");
        b.L($"add.u32 {bsBase}, {smem}, {asBytes};");

        // Accumulators.
        var acc = new string[micro, micro];
        for (int r = 0; r < micro; r++) for (int c = 0; c < micro; c++) { acc[r, c] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[r, c]}, {lane.Zero};"); }

        // This thread's share of the tiles: A's tile (tile rows × 8) and B's (8 × tile): tile·8/256 values each.
        int perThread = tile * bk / GemmThreads;                           // 2 (tile 64) or 4 (tile 128)
        string aRow = b.R(), aCol = b.R(), bRow = b.R(), bCol = b.R();
        int aRowsPerPass = GemmThreads / (bk / perThread);                   // threads cover (bk/perThread) column groups per row
        b.L($"div.u32 {aRow}, {tid}, {bk / perThread};");
        b.L($"rem.u32 {aCol}, {tid}, {bk / perThread};");
        b.L($"mul.lo.u32 {aCol}, {aCol}, {perThread};");
        b.L($"div.u32 {bRow}, {tid}, {tile / perThread};");
        b.L($"rem.u32 {bCol}, {tid}, {tile / perThread};");
        b.L($"mul.lo.u32 {bCol}, {bCol}, {perThread};");
        _ = aRowsPerPass;
        string gRow = b.R(), pRow = b.P(), gColB = b.R(), pColB = b.P();
        b.L($"add.u32 {gRow}, {row0}, {aRow};");
        b.L($"setp.lt.u32 {pRow}, {gRow}, {m};");
        b.L($"add.u32 {gColB}, {col0}, {bCol};");
        // A row base address and B column base.
        string aBase = b.RD(), t64 = b.RD();
        b.L($"mul.wide.u32 {t64}, {gRow}, {k};");
        b.L($"mul.lo.u64 {t64}, {t64}, {e};");
        b.L($"add.u64 {aBase}, {pa}, {t64};");

        string k0 = b.R(), pLoop = b.P();
        b.L($"mov.u32 {k0}, 0;");
        b.Label("KLOOP");
        b.L($"setp.ge.u32 {pLoop}, {k0}, {k};");
        b.L($"@{pLoop} bra KDONE;");

        // A's tile: this thread's perThread consecutive k values of row aRow at k0 + aCol, stored transposed As[kk][row]:
        // whole vectors when the row's values are in range and aligned, else one by one (guarded).
        {
            var av0 = new string[perThread];
            for (int q = 0; q < perThread; q++) { av0[q] = lane.NewReg(b); b.L($"mov.{lane.Reg} {av0[q]}, {lane.Zero};"); }
            string kk0 = b.R(), kEnd = b.R(), pVec = b.P(), pAlign = b.P(), kMod = b.R(), off0 = b.RD(), addr0 = b.RD();
            b.L($"add.u32 {kk0}, {k0}, {aCol};");
            b.L($"add.u32 {kEnd}, {kk0}, {perThread};");
            b.L($"setp.le.u32 {pVec}, {kEnd}, {k};");
            b.L($"and.b32 {kMod}, {k}, {(e == 4 ? 3 : 1)};");
            b.L($"setp.eq.u32 {pAlign}, {kMod}, 0;");
            b.L($"and.pred {pVec}, {pVec}, {pAlign};");
            b.L($"and.pred {pVec}, {pVec}, {pRow};");
            b.L($"mul.wide.u32 {off0}, {kk0}, {e};");
            b.L($"add.u64 {addr0}, {aBase}, {off0};");
            b.L($"@!{pVec} bra ASCALAR;");
            int perVec = e == 4 ? 4 : 2;
            for (int q = 0; q < perThread; q += perVec)
            {
                string regs = perVec == 4 ? $"{{{av0[q]}, {av0[q + 1]}, {av0[q + 2]}, {av0[q + 3]}}}" : $"{{{av0[q]}, {av0[q + 1]}}}";
                b.L($"ld.global.v{perVec}.{lane.Mem} {regs}, [{addr0}+{q * e}];");
            }

            b.L("bra ASTORE;");
            b.Label("ASCALAR");
            for (int q = 0; q < perThread; q++)
            {
                string kk = b.R(), pk = b.P();
                b.L($"add.u32 {kk}, {kk0}, {q};");
                b.L($"setp.lt.u32 {pk}, {kk}, {k};");
                b.L($"and.pred {pk}, {pk}, {pRow};");
                b.L($"@{pk} ld.global.{lane.Mem} {av0[q]}, [{addr0}+{q * e}];");
            }

            b.Label("ASTORE");
            for (int q = 0; q < perThread; q++)
            {
                string sAddr = b.R(), sOff = b.R();
                b.L($"mad.lo.u32 {sOff}, {aCol}, {tile}, {aRow};");
                b.L($"add.u32 {sOff}, {sOff}, {q * tile};");
                b.L($"mad.lo.u32 {sAddr}, {sOff}, {e}, {asBase};");
                b.L($"st.shared.{lane.Mem} [{sAddr}], {av0[q]};");
            }
        }

        // B's tile: row k0 + bRow, perThread consecutive columns at col0 + bCol, stored Bs[bRow][col].
        if (conv)
        {
            // The implicit im2col: row kk = (c, kh, kw), column = (oh, ow), each value gathered from the image or 0.
            string kk = b.R(), pk = b.P(), khw = b.R(), c = b.R(), r = b.R(), kh = b.R(), kw = b.R(), dih = b.R(), diw = b.R();
            b.L($"add.u32 {kk}, {k0}, {bRow};");
            b.L($"setp.lt.u32 {pk}, {kk}, {k};");
            b.L($"mul.lo.u32 {khw}, {g[2]}, {g[3]};");
            b.L($"div.u32 {c}, {kk}, {khw};");
            b.L($"rem.u32 {r}, {kk}, {khw};");
            b.L($"div.u32 {kh}, {r}, {g[3]};");
            b.L($"rem.u32 {kw}, {r}, {g[3]};");
            b.L($"mul.lo.s32 {dih}, {kh}, {g[8]};");
            b.L($"sub.s32 {dih}, {dih}, {g[6]};");
            b.L($"mul.lo.s32 {diw}, {kw}, {g[9]};");
            b.L($"sub.s32 {diw}, {diw}, {g[7]};");
            string cBase = b.R();
            b.L($"mul.lo.u32 {cBase}, {c}, {g[0]};");
            var bv0 = new string[perThread];
            for (int q = 0; q < perThread; q++)
            {
                bv0[q] = lane.NewReg(b);
                b.L($"mov.{lane.Reg} {bv0[q]}, {lane.Zero};");
                string colq = b.R(), pq = b.P(), oh = b.R(), ow = b.R(), ih = b.R(), iw = b.R(), pin = b.P(), idx = b.R(), addr = b.RD();
                b.L($"add.u32 {colq}, {gColB}, {q};");
                b.L($"setp.lt.u32 {pq}, {colq}, {n};");
                b.L($"and.pred {pq}, {pq}, {pk};");
                b.L($"div.u32 {oh}, {colq}, {g[10]};");
                b.L($"rem.u32 {ow}, {colq}, {g[10]};");
                b.L($"mad.lo.s32 {ih}, {oh}, {g[4]}, {dih};");
                b.L($"mad.lo.s32 {iw}, {ow}, {g[5]}, {diw};");
                b.L($"setp.lt.u32 {pin}, {ih}, {g[0]};");                       // a negative index reads as a large one
                b.L($"and.pred {pq}, {pq}, {pin};");
                b.L($"setp.lt.u32 {pin}, {iw}, {g[1]};");
                b.L($"and.pred {pq}, {pq}, {pin};");
                b.L($"add.u32 {idx}, {cBase}, {ih};");
                b.L($"mad.lo.u32 {idx}, {idx}, {g[1]}, {iw};");
                b.L($"mul.wide.u32 {addr}, {idx}, {e};");
                b.L($"add.u64 {addr}, {pb}, {addr};");
                b.L($"@{pq} ld.global.{lane.Mem} {bv0[q]}, [{addr}];");
            }

            string sOff0 = b.R(), sAddr0 = b.R();
            int perVec = e == 4 ? 4 : 2;
            b.L($"mad.lo.u32 {sOff0}, {bRow}, {tile}, {bCol};");
            b.L($"mad.lo.u32 {sAddr0}, {sOff0}, {e}, {bsBase};");
            for (int q = 0; q < perThread; q += perVec)
            {
                string regs = perVec == 4 ? $"{{{bv0[q]}, {bv0[q + 1]}, {bv0[q + 2]}, {bv0[q + 3]}}}" : $"{{{bv0[q]}, {bv0[q + 1]}}}";
                b.L($"st.shared.v{perVec}.{lane.Mem} [{sAddr0}+{q * e}], {regs};");
            }
        }
        else
        {
            string kk = b.R(), pk = b.P(), off = b.RD(), rowBase = b.RD();
            b.L($"add.u32 {kk}, {k0}, {bRow};");
            b.L($"setp.lt.u32 {pk}, {kk}, {k};");
            b.L($"mul.wide.u32 {off}, {kk}, {n};");
            b.L($"mul.lo.u64 {off}, {off}, {e};");
            b.L($"add.u64 {rowBase}, {pb}, {off};");
            var bv0 = new string[perThread];
            for (int q = 0; q < perThread; q++) { bv0[q] = lane.NewReg(b); b.L($"mov.{lane.Reg} {bv0[q]}, {lane.Zero};"); }
            string colEnd = b.R(), pVec = b.P(), pAlign = b.P(), nMod = b.R(), cOff0 = b.RD(), addr0 = b.RD();
            b.L($"add.u32 {colEnd}, {gColB}, {perThread};");
            b.L($"setp.le.u32 {pVec}, {colEnd}, {n};");
            b.L($"and.b32 {nMod}, {n}, {(e == 4 ? 3 : 1)};");
            b.L($"setp.eq.u32 {pAlign}, {nMod}, 0;");
            b.L($"and.pred {pVec}, {pVec}, {pAlign};");
            b.L($"and.pred {pVec}, {pVec}, {pk};");
            b.L($"mul.wide.u32 {cOff0}, {gColB}, {e};");
            b.L($"add.u64 {addr0}, {rowBase}, {cOff0};");
            b.L($"@!{pVec} bra BSCALAR;");
            int perVec = e == 4 ? 4 : 2;
            for (int q = 0; q < perThread; q += perVec)
            {
                string regs = perVec == 4 ? $"{{{bv0[q]}, {bv0[q + 1]}, {bv0[q + 2]}, {bv0[q + 3]}}}" : $"{{{bv0[q]}, {bv0[q + 1]}}}";
                b.L($"ld.global.v{perVec}.{lane.Mem} {regs}, [{addr0}+{q * e}];");
            }

            b.L("bra BSTORE;");
            b.Label("BSCALAR");
            for (int q = 0; q < perThread; q++)
            {
                string col = b.R(), pq = b.P();
                b.L($"add.u32 {col}, {gColB}, {q};");
                b.L($"setp.lt.u32 {pq}, {col}, {n};");
                b.L($"and.pred {pq}, {pq}, {pk};");
                b.L($"@{pq} ld.global.{lane.Mem} {bv0[q]}, [{addr0}+{q * e}];");
            }

            b.Label("BSTORE");
            string sOff0 = b.R(), sAddr0 = b.R();
            b.L($"mad.lo.u32 {sOff0}, {bRow}, {tile}, {bCol};");
            b.L($"mad.lo.u32 {sAddr0}, {sOff0}, {e}, {bsBase};");
            for (int q = 0; q < perThread; q += perVec)
            {
                string regs = perVec == 4 ? $"{{{bv0[q]}, {bv0[q + 1]}, {bv0[q + 2]}, {bv0[q + 3]}}}" : $"{{{bv0[q]}, {bv0[q + 1]}}}";
                b.L($"st.shared.v{perVec}.{lane.Mem} [{sAddr0}+{q * e}], {regs};");
            }
        }

        b.L("bar.sync 0;");

        // The block of C: rows ty·micro .., columns tx·micro ..; per kk, micro values of each.
        string aStart = b.R(), bStart = b.R();
        b.L($"mul.lo.u32 {aStart}, {ty}, {micro * e};");
        b.L($"add.u32 {aStart}, {aStart}, {asBase};");
        b.L($"mul.lo.u32 {bStart}, {tx}, {micro * e};");
        b.L($"add.u32 {bStart}, {bStart}, {bsBase};");
        var av = new string[micro];
        var bv = new string[micro];
        for (int kk = 0; kk < bk; kk++)
        {
            for (int r = 0; r < micro; r++) av[r] = lane.NewReg(b);
            for (int c = 0; c < micro; c++) bv[c] = lane.NewReg(b);
            int rowBytes = tile * e;
            // Vector loads of the thread's micro consecutive values (4 at a time: 16 or 32 bytes).
            for (int r = 0; r < micro; r += 4)
            {
                string vec = e == 4 ? "v4" : "v2";
                if (e == 4) b.L($"ld.shared.v4.{lane.Mem} {{{av[r]}, {av[r + 1]}, {av[r + 2]}, {av[r + 3]}}}, [{aStart}+{kk * rowBytes + r * e}];");
                else
                {
                    b.L($"ld.shared.v2.{lane.Mem} {{{av[r]}, {av[r + 1]}}}, [{aStart}+{kk * rowBytes + r * e}];");
                    b.L($"ld.shared.v2.{lane.Mem} {{{av[r + 2]}, {av[r + 3]}}}, [{aStart}+{kk * rowBytes + (r + 2) * e}];");
                }

                _ = vec;
            }

            for (int c = 0; c < micro; c += 4)
            {
                if (e == 4) b.L($"ld.shared.v4.{lane.Mem} {{{bv[c]}, {bv[c + 1]}, {bv[c + 2]}, {bv[c + 3]}}}, [{bStart}+{kk * rowBytes + c * e}];");
                else
                {
                    b.L($"ld.shared.v2.{lane.Mem} {{{bv[c]}, {bv[c + 1]}}}, [{bStart}+{kk * rowBytes + c * e}];");
                    b.L($"ld.shared.v2.{lane.Mem} {{{bv[c + 2]}, {bv[c + 3]}}}, [{bStart}+{kk * rowBytes + (c + 2) * e}];");
                }
            }

            for (int r = 0; r < micro; r++) for (int c = 0; c < micro; c++) b.L(lane.Fma(acc[r, c], av[r], bv[c], acc[r, c]));
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {k0}, {k0}, {bk};");
        b.L("bra KLOOP;");
        b.Label("KDONE");

        // Store: C[row0 + ty·micro + r][col0 + tx·micro + c], guarded.
        for (int r = 0; r < micro; r++)
        {
            string row = b.R(), pr = b.P(), rowOff = b.RD();
            b.L($"mad.lo.u32 {row}, {ty}, {micro}, {row0};");
            b.L($"add.u32 {row}, {row}, {r};");
            b.L($"setp.lt.u32 {pr}, {row}, {m};");
            b.L($"mul.wide.u32 {rowOff}, {row}, {n};");
            for (int c = 0; c < micro; c++)
            {
                string col = b.R(), pcol = b.P(), idx = b.RD(), addr = b.RD();
                b.L($"mad.lo.u32 {col}, {tx}, {micro}, {col0};");
                b.L($"add.u32 {col}, {col}, {c};");
                b.L($"setp.lt.u32 {pcol}, {col}, {n};");
                b.L($"and.pred {pcol}, {pcol}, {pr};");
                b.L($"cvt.u64.u32 {idx}, {col};");
                b.L($"add.u64 {idx}, {idx}, {rowOff};");
                b.L($"mul.lo.u64 {idx}, {idx}, {e};");
                b.L($"add.u64 {addr}, {pc}, {idx};");
                b.L($"@{pcol} st.global.{lane.Mem} [{addr}], {acc[r, c]};");
            }
        }

        b.L("ret;");
        string convParams = conv ? ", .param .u32 p_h, .param .u32 p_w, .param .u32 p_kh, .param .u32 p_kw, .param .u32 p_sh, .param .u32 p_sw, .param .u32 p_ph, .param .u32 p_pw, .param .u32 p_dh, .param .u32 p_dw, .param .u32 p_ow" : "";
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_k" + convParams, asBytes + bsBytes);
    }

    /// <summary>Shared bytes the 32-bit-lane product takes: A 128 rows × 80 B and B 16 rows × 512 B per stage.</summary>
    public static int Gemm2Shared(int stages) => stages * (128 * 80 + 16 * 512);

    /// <summary>C = A·B for a 32-bit lane (A m×k, B k×n row-major, k and n multiples of 4): 128×128 of C per block, each thread
    /// rows ty + 16r and columns tx·8 .. +7, k in blocks of 16 through staged asynchronous copies; A rows padded to 80 bytes
    /// (scalar broadcast reads), B rows 512 bytes (vector reads); the thread's 8 consecutive columns stored as two vectors.</summary>
    public static string Gemm2(Gpu gpu, Lane lane, int stages, out string entry)
    {
        entry = $"gemm2_{lane.Name}_s{stages}";
        const int Tile = 128, Bk = 16, AStride = 80, BStride = 512, ABytes = Tile * AStride, BBytes = Bk * BStride;
        int stageBytes = ABytes + BBytes, totalShared = stages * stageBytes;
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), tx = b.R(), ty = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"and.b32 {tx}, {tid}, 15;");
        b.L($"shr.u32 {ty}, {tid}, 4;");
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), m = Param32(b, "p_m"), n = Param32(b, "p_n"), k = Param32(b, "p_k");
        string row0 = b.R(), col0 = b.R(), smem = b.R();
        b.L($"mul.lo.u32 {row0}, {by}, {Tile};");
        b.L($"mul.lo.u32 {col0}, {bx}, {Tile};");
        b.L($"mov.u32 {smem}, smem;");
        var acc = new string[8, 8];
        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { acc[r, c] = lane.NewReg(b); b.L($"mov.{lane.Reg} {acc[r, c]}, {lane.Zero};"); }

        // Copies: A chunk c = tid + 256j → row c/4, part c%4 (16 bytes of the row's 16 k values); B chunk → k-row c/32, part c%32.
        var gA = new string[2]; var gB = new string[2]; var sA = new string[2]; var sB = new string[2]; var pA = new string[2]; var pB = new string[2]; var aPart = new string[2]; var bPart = new string[2];
        string kBytes4 = b.RD(), nBytes4 = b.RD();
        b.L($"mul.wide.u32 {kBytes4}, {k}, 4;");
        b.L($"mul.wide.u32 {nBytes4}, {n}, 4;");
        for (int j = 0; j < 2; j++)
        {
            string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), t64 = b.RD(), rowHeld = b.R();
            b.L($"add.u32 {chunk}, {tid}, {j * 256};");
            b.L($"shr.u32 {row}, {chunk}, 2;");
            b.L($"and.b32 {part}, {chunk}, 3;");
            b.L($"add.u32 {gRow}, {row0}, {row};");
            pA[j] = b.P();
            b.L($"setp.lt.u32 {pA[j]}, {gRow}, {m};");
            b.L($"selp.u32 {rowHeld}, {gRow}, 0, {pA[j]};");
            gA[j] = b.RD();
            b.L($"cvt.u64.u32 {t64}, {rowHeld};");
            b.L($"mul.lo.u64 {t64}, {t64}, {kBytes4};");
            b.L($"add.u64 {gA[j]}, {pa}, {t64};");
            aPart[j] = b.R();
            b.L($"shl.b32 {aPart[j]}, {part}, 2;");                        // k offset of the chunk within the block
            sA[j] = b.R();
            b.L($"mad.lo.u32 {sA[j]}, {row}, {AStride}, {smem};");
            b.L($"mad.lo.u32 {sA[j]}, {part}, 16, {sA[j]};");

            string kRow = b.R(), bpart = b.R(), bRowHeld = b.R();
            b.L($"shr.u32 {kRow}, {chunk}, 5;");
            b.L($"and.b32 {bpart}, {chunk}, 31;");
            bPart[j] = b.R();
            b.L($"shl.b32 {bPart[j]}, {bpart}, 2;");                       // column offset of the chunk within the tile
            string gCol = b.R();
            b.L($"add.u32 {gCol}, {col0}, {bPart[j]};");
            pB[j] = b.P();
            b.L($"setp.lt.u32 {pB[j]}, {gCol}, {n};");
            b.L($"selp.u32 {bRowHeld}, {gCol}, 0, {pB[j]};");
            gB[j] = b.RD();
            b.L($"mul.wide.u32 {gB[j]}, {bRowHeld}, 4;");
            b.L($"add.u64 {gB[j]}, {pb}, {gB[j]};");
            string kRow64 = b.RD();
            b.L($"cvt.u64.u32 {kRow64}, {kRow};");
            b.L($"mul.lo.u64 {kRow64}, {kRow64}, {nBytes4};");
            b.L($"add.u64 {gB[j]}, {gB[j]}, {kRow64};");                     // row k0 + kRow is added per block below
            sB[j] = b.R();
            b.L($"mad.lo.u32 {sB[j]}, {kRow}, {BStride}, {smem};");
            b.L($"add.u32 {sB[j]}, {sB[j]}, {ABytes};");
            b.L($"mad.lo.u32 {sB[j]}, {bpart}, 16, {sB[j]};");
            _ = kRow;
        }

        string blocks = b.R();
        b.L($"add.u32 {blocks}, {k}, {Bk - 1};");
        b.L($"shr.u32 {blocks}, {blocks}, 4;");
        string kRows = new string[2].Length.ToString();
        _ = kRows;
        var kRowOf = new string[2];
        for (int j = 0; j < 2; j++)
        {
            kRowOf[j] = b.R();
            string chunk = b.R();
            b.L($"add.u32 {chunk}, {tid}, {j * 256};");
            b.L($"shr.u32 {kRowOf[j]}, {chunk}, 5;");
        }

        void Issue(string block, string stageOff, string label)
        {
            string pHas = b.P(), k0 = b.R();
            b.L($"setp.ge.u32 {pHas}, {block}, {blocks};");
            b.L($"@{pHas} bra {label};");
            b.L($"shl.b32 {k0}, {block}, 4;");
            for (int j = 0; j < 2; j++)
            {
                string kk = b.R(), pk = b.P(), size = b.R(), src = b.RD(), off = b.RD(), dst = b.R();
                // A: k0 + part·4 < k (k a multiple of 4: the chunk is whole).
                b.L($"add.u32 {kk}, {k0}, {aPart[j]};");
                b.L($"setp.lt.u32 {pk}, {kk}, {k};");
                b.L($"and.pred {pk}, {pk}, {pA[j]};");
                b.L($"selp.u32 {size}, 16, 0, {pk};");
                b.L($"mul.wide.u32 {off}, {kk}, 4;");
                b.L($"add.u64 {src}, {gA[j]}, {off};");
                b.L($"add.u32 {dst}, {sA[j]}, {stageOff};");
                b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {size};");
                // B: row k0 + kRow < k.
                string kr = b.R(), pkr = b.P(), sizeB = b.R(), srcB = b.RD(), offB = b.RD(), dstB = b.R();
                b.L($"add.u32 {kr}, {k0}, {kRowOf[j]};");
                b.L($"setp.lt.u32 {pkr}, {kr}, {k};");
                b.L($"and.pred {pkr}, {pkr}, {pB[j]};");
                b.L($"selp.u32 {sizeB}, 16, 0, {pkr};");
                b.L($"mul.wide.u32 {offB}, {k0}, 4;");
                b.L($"mul.lo.u64 {offB}, {offB}, {nBytes4};");
                b.L($"shr.u64 {offB}, {offB}, 2;");                          // k0 · n · 4 bytes
                b.L($"add.u64 {srcB}, {gB[j]}, {offB};");
                b.L($"add.u32 {dstB}, {sB[j]}, {stageOff};");
                b.L($"cp.async.cg.shared.global [{dstB}], [{srcB}], 16, {sizeB};");
            }

            b.Label(label);
            b.L("cp.async.commit_group;");
        }

        string aRead = b.R(), bRead = b.R();
        b.L($"mad.lo.u32 {aRead}, {ty}, {AStride}, {smem};");
        b.L($"mad.lo.u32 {bRead}, {tx}, 32, {smem};");
        b.L($"add.u32 {bRead}, {bRead}, {ABytes};");

        string kb = b.R(), stage = b.R(), issueBlock = b.R(), issueStage = b.R(), pLoop = b.P(), pWrap = b.P(), pWrap2 = b.P();
        for (int sIdx = 0; sIdx < stages - 1; sIdx++)
        {
            string blk = b.R(), off = b.R();
            b.L($"mov.u32 {blk}, {sIdx};");
            b.L($"mov.u32 {off}, {sIdx * stageBytes};");
            Issue(blk, off, $"PRO{sIdx}");
        }

        b.L($"mov.u32 {kb}, 0;");
        b.L($"mov.u32 {stage}, 0;");
        b.L($"mov.u32 {issueBlock}, {stages - 1};");
        b.L($"mov.u32 {issueStage}, {(stages - 1) * stageBytes};");
        b.Label("KLOOP");
        b.L($"setp.ge.u32 {pLoop}, {kb}, {blocks};");
        b.L($"@{pLoop} bra KDONE;");
        b.L($"cp.async.wait_group {stages - 2};");
        b.L("bar.sync 0;");
        string aStage = b.R(), bStage = b.R();
        b.L($"add.u32 {aStage}, {aRead}, {stage};");
        b.L($"add.u32 {bStage}, {bRead}, {stage};");
        for (int kk = 0; kk < Bk; kk++)
        {
            var av = new string[8];
            var bv = new string[8];
            for (int r = 0; r < 8; r++) { av[r] = lane.NewReg(b); b.L($"ld.shared.{lane.Mem} {av[r]}, [{aStage}+{r * 16 * AStride + kk * 4}];"); }
            for (int c = 0; c < 8; c++) bv[c] = lane.NewReg(b);
            b.L($"ld.shared.v4.{lane.Mem} {{{bv[0]}, {bv[1]}, {bv[2]}, {bv[3]}}}, [{bStage}+{kk * BStride}];");
            b.L($"ld.shared.v4.{lane.Mem} {{{bv[4]}, {bv[5]}, {bv[6]}, {bv[7]}}}, [{bStage}+{kk * BStride + 16}];");
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) b.L(lane.Fma(acc[r, c], av[r], bv[c], acc[r, c]));
        }

        b.L("bar.sync 0;");
        Issue(issueBlock, issueStage, "ISSUE");
        b.L($"add.u32 {issueBlock}, {issueBlock}, 1;");
        b.L($"add.u32 {issueStage}, {issueStage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap}, {issueStage}, {totalShared};");
        b.L($"@{pWrap} mov.u32 {issueStage}, 0;");
        b.L($"add.u32 {stage}, {stage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap2}, {stage}, {totalShared};");
        b.L($"@{pWrap2} mov.u32 {stage}, 0;");
        b.L($"add.u32 {kb}, {kb}, 1;");
        b.L("bra KLOOP;");
        b.Label("KDONE");
        b.L("cp.async.wait_group 0;");

        // C: rows row0 + ty + 16r, columns col0 + tx·8 .. +7: two vectors when the eight columns are in range, else one by one.
        string cCol = b.R(), pVec = b.P(), colEnd = b.R();
        b.L($"mad.lo.u32 {cCol}, {tx}, 8, {col0};");
        b.L($"add.u32 {colEnd}, {cCol}, 8;");
        b.L($"setp.le.u32 {pVec}, {colEnd}, {n};");
        for (int r = 0; r < 8; r++)
        {
            string row = b.R(), pr = b.P(), rowOff = b.RD(), addr = b.RD(), pv = b.P();
            b.L($"add.u32 {row}, {row0}, {ty};");
            b.L($"add.u32 {row}, {row}, {r * 16};");
            b.L($"setp.lt.u32 {pr}, {row}, {m};");
            b.L($"mul.wide.u32 {rowOff}, {row}, {n};");
            b.L($"add.u64 {addr}, {rowOff}, {Cvt64Of(b, cCol)};");
            b.L($"shl.b64 {addr}, {addr}, 2;");
            b.L($"add.u64 {addr}, {pc}, {addr};");
            b.L($"and.pred {pv}, {pVec}, {pr};");
            b.L($"@{pv} st.global.v4.{lane.Mem} [{addr}], {{{acc[r, 0]}, {acc[r, 1]}, {acc[r, 2]}, {acc[r, 3]}}};");
            b.L($"@{pv} st.global.v4.{lane.Mem} [{addr}+16], {{{acc[r, 4]}, {acc[r, 5]}, {acc[r, 6]}, {acc[r, 7]}}};");
            b.L($"@{pv} bra ROW{r}_DONE;");
            for (int c = 0; c < 8; c++)
            {
                string col = b.R(), pcol = b.P();
                b.L($"add.u32 {col}, {cCol}, {c};");
                b.L($"setp.lt.u32 {pcol}, {col}, {n};");
                b.L($"and.pred {pcol}, {pcol}, {pr};");
                b.L($"@{pcol} st.global.{lane.Mem} [{addr}+{c * 4}], {acc[r, c]};");
            }

            b.Label($"ROW{r}_DONE");
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_k", totalShared);
    }

    private static string Cvt64Of(PtxBuilder b, string r)
    {
        string wide = b.RD();
        b.L($"cvt.u64.u32 {wide}, {r};");
        return wide;
    }

    /// <summary>C = G·Btᵀ over groups of four 8-bit values: G m×kg and Bt n×kg words; 128×128 tile, 8×8 per thread, 16 groups a step.</summary>
    public static string GemmDot(Gpu gpu, bool signed, out string entry)
    {
        const int Tile = 128, Micro = 8, Bk = 16, Stride = Bk + 1;
        entry = signed ? "gemm_dp4a_s" : "gemm_dp4a_u";
        string op = signed ? "dp4a.s32.s32" : "dp4a.u32.u32";
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), tx = b.R(), ty = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"and.b32 {tx}, {tid}, 15;");
        b.L($"shr.u32 {ty}, {tid}, 4;");
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), m = Param32(b, "p_m"), n = Param32(b, "p_n"), kg = Param32(b, "p_kg");
        string row0 = b.R(), col0 = b.R();
        b.L($"mul.lo.u32 {row0}, {by}, {Tile};");
        b.L($"mul.lo.u32 {col0}, {bx}, {Tile};");
        int tileBytes = Tile * Stride * 4;
        string smem = b.R(), asBase = b.R(), bsBase = b.R();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mov.u32 {asBase}, {smem};");
        b.L($"add.u32 {bsBase}, {smem}, {tileBytes};");
        var acc = new string[Micro, Micro];
        for (int r = 0; r < Micro; r++) for (int c = 0; c < Micro; c++) { acc[r, c] = b.R(); b.L($"mov.b32 {acc[r, c]}, 0;"); }

        // Loads: each tile is 128 rows × 16 words = 2048 words, 8 per thread: row tid/2, words (tid&1)·8 ..
        string lRow = b.R(), lCol = b.R();
        b.L($"shr.u32 {lRow}, {tid}, 1;");
        b.L($"and.b32 {lCol}, {tid}, 1;");
        b.L($"shl.b32 {lCol}, {lCol}, 3;");
        string gRowA = b.R(), pRowA = b.P(), gRowB = b.R(), pRowB = b.P(), aBase = b.RD(), bBase = b.RD(), t64 = b.RD();
        b.L($"add.u32 {gRowA}, {row0}, {lRow};");
        b.L($"setp.lt.u32 {pRowA}, {gRowA}, {m};");
        b.L($"add.u32 {gRowB}, {col0}, {lRow};");
        b.L($"setp.lt.u32 {pRowB}, {gRowB}, {n};");
        b.L($"mul.wide.u32 {t64}, {gRowA}, {kg};");
        b.L($"shl.b64 {t64}, {t64}, 2;");
        b.L($"add.u64 {aBase}, {pa}, {t64};");
        b.L($"mul.wide.u32 {t64}, {gRowB}, {kg};");
        b.L($"shl.b64 {t64}, {t64}, 2;");
        b.L($"add.u64 {bBase}, {pb}, {t64};");
        string sRowA = b.R(), sRowB = b.R();
        b.L($"mad.lo.u32 {sRowA}, {lRow}, {Stride * 4}, {asBase};");
        b.L($"mad.lo.u32 {sRowB}, {lRow}, {Stride * 4}, {bsBase};");

        string g0 = b.R(), pLoop = b.P();
        b.L($"mov.u32 {g0}, 0;");
        b.Label("KLOOP");
        b.L($"setp.ge.u32 {pLoop}, {g0}, {kg};");
        b.L($"@{pLoop} bra KDONE;");
        for (int q = 0; q < 8; q++)
        {
            string g = b.R(), pg = b.P(), va = b.R(), vb = b.R(), off = b.RD(), addrA = b.RD(), addrB = b.RD(), pa2 = b.P(), pb2 = b.P();
            b.L($"add.u32 {g}, {g0}, {lCol};");
            b.L($"add.u32 {g}, {g}, {q};");
            b.L($"setp.lt.u32 {pg}, {g}, {kg};");
            b.L($"and.pred {pa2}, {pg}, {pRowA};");
            b.L($"and.pred {pb2}, {pg}, {pRowB};");
            b.L($"mul.wide.u32 {off}, {g}, 4;");
            b.L($"add.u64 {addrA}, {aBase}, {off};");
            b.L($"add.u64 {addrB}, {bBase}, {off};");
            b.L($"mov.b32 {va}, 0;");
            b.L($"mov.b32 {vb}, 0;");
            b.L($"@{pa2} ld.global.b32 {va}, [{addrA}];");
            b.L($"@{pb2} ld.global.b32 {vb}, [{addrB}];");
            string sa = b.R(), sb2 = b.R();
            b.L($"mad.lo.u32 {sa}, {lCol}, 4, {sRowA};");
            b.L($"mad.lo.u32 {sb2}, {lCol}, 4, {sRowB};");
            b.L($"st.shared.b32 [{sa}+{q * 4}], {va};");
            b.L($"st.shared.b32 [{sb2}+{q * 4}], {vb};");
        }

        b.L("bar.sync 0;");
        // This thread's rows ty + 16r and columns tx + 16c (interleaved: conflict-free shared reads, coalesced stores).
        string aStart = b.R(), bStart = b.R();
        b.L($"mad.lo.u32 {aStart}, {ty}, {Stride * 4}, {asBase};");
        b.L($"mad.lo.u32 {bStart}, {tx}, {Stride * 4}, {bsBase};");
        var av = new string[Micro];
        var bv = new string[Micro];
        for (int kk = 0; kk < Bk; kk++)
        {
            for (int r = 0; r < Micro; r++) { av[r] = b.R(); b.L($"ld.shared.b32 {av[r]}, [{aStart}+{(r * 16 * Stride + kk) * 4}];"); }
            for (int c = 0; c < Micro; c++) { bv[c] = b.R(); b.L($"ld.shared.b32 {bv[c]}, [{bStart}+{(c * 16 * Stride + kk) * 4}];"); }
            for (int r = 0; r < Micro; r++) for (int c = 0; c < Micro; c++) b.L($"{op} {acc[r, c]}, {av[r]}, {bv[c]}, {acc[r, c]};");
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {g0}, {g0}, {Bk};");
        b.L("bra KLOOP;");
        b.Label("KDONE");
        for (int r = 0; r < Micro; r++)
        {
            string row = b.R(), pr = b.P(), rowOff = b.RD();
            b.L($"add.u32 {row}, {row0}, {ty};");
            b.L($"add.u32 {row}, {row}, {r * 16};");
            b.L($"setp.lt.u32 {pr}, {row}, {m};");
            b.L($"mul.wide.u32 {rowOff}, {row}, {n};");
            for (int c = 0; c < Micro; c++)
            {
                string col = b.R(), pcol = b.P(), idx = b.RD(), addr = b.RD();
                b.L($"add.u32 {col}, {col0}, {tx};");
                b.L($"add.u32 {col}, {col}, {c * 16};");
                b.L($"setp.lt.u32 {pcol}, {col}, {n};");
                b.L($"and.pred {pcol}, {pcol}, {pr};");
                b.L($"cvt.u64.u32 {idx}, {col};");
                b.L($"add.u64 {idx}, {idx}, {rowOff};");
                b.L($"shl.b64 {idx}, {idx}, 2;");
                b.L($"add.u64 {addr}, {pc}, {idx};");
                b.L($"@{pcol} st.global.b32 [{addr}], {acc[r, c]};");
            }
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_kg", 2 * tileBytes);
    }

    // ---------------------------------------------------------------- the range of stored integers

    public const int RangeThreads = 256;

    /// <summary>Each block's smallest and largest value (as s64; unsigned 64-bit values past s64's range held to it) into out[block·2 ..].
    /// A thread per word, every value of the word unpacked (values past n ignored).</summary>
    public static string Range(Gpu gpu, Fmt f, out string entry)
    {
        entry = "range_" + f.Name;
        bool u64 = f.Is("uint64");
        var wide = u64 ? Lane.U64 : Lane.S64;
        var b = new PtxBuilder();
        string tid = b.R(), ctaid = b.R(), nctaid = b.R(), gid = b.R(), stride = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {ctaid}, %ctaid.x;");
        b.L($"mov.u32 {nctaid}, %nctaid.x;");
        b.L($"mad.lo.u32 {gid}, {ctaid}, {RangeThreads}, {tid};");
        b.L($"mul.lo.u32 {stride}, {nctaid}, {RangeThreads};");
        string src = Param64(b, "p_x"), n = Param32(b, "p_n"), dst = Param64(b, "p_out");
        string words = WordCount(b, f, n);
        string lo = b.RD(), hi = b.RD(), w = b.R(), pEnd = b.P();
        string cmp = u64 ? "u64" : "s64";
        b.L(u64 ? $"mov.b64 {lo}, 0xFFFFFFFFFFFFFFFF;" : $"mov.b64 {lo}, 0x7FFFFFFFFFFFFFFF;");
        b.L(u64 ? $"mov.b64 {hi}, 0;" : $"mov.b64 {hi}, 0x8000000000000000;");
        b.L($"mov.u32 {w}, {gid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {w}, {words};");
        b.L($"@{pEnd} bra REDUCE;");
        string off = b.RD(), addr = b.RD(), word = f.WordBits == 64 ? b.RD() : b.R(), first = b.R();
        b.L($"mul.wide.u32 {off}, {w}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {src}, {off};");
        b.L($"ld.global.{(f.WordBits == 64 ? "b64" : "b32")} {word}, [{addr}];");
        b.L($"mul.lo.u32 {first}, {w}, {f.Per};");
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R(), pi = b.P();
            b.L($"add.u32 {idx}, {first}, {i};");
            b.L($"setp.ge.u32 {pi}, {idx}, {n};");
            b.L($"@{pi} bra NEXT{i};");
            string v = Codec.Unpack(b, f, word, i, wide);
            b.L($"min.{cmp} {lo}, {lo}, {v};");
            b.L($"max.{cmp} {hi}, {hi}, {v};");
            b.Label($"NEXT{i}");
        }

        b.L($"add.u32 {w}, {w}, {stride};");
        b.L("bra LOOP;");
        b.Label("REDUCE");
        string smem = b.R(), mine = b.R();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mad.lo.u32 {mine}, {tid}, 16, {smem};");
        b.L($"st.shared.b64 [{mine}], {lo};");
        b.L($"st.shared.b64 [{mine}+8], {hi};");
        b.L("bar.sync 0;");
        for (int s = RangeThreads / 2; s >= 1; s /= 2)
        {
            string p = b.P(), other = b.R(), olo = b.RD(), ohi = b.RD();
            b.L($"setp.ge.u32 {p}, {tid}, {s};");
            b.L($"@{p} bra SKIP{s};");
            b.L($"add.u32 {other}, {mine}, {s * 16};");
            b.L($"ld.shared.b64 {olo}, [{other}];");
            b.L($"ld.shared.b64 {ohi}, [{other}+8];");
            b.L($"min.{cmp} {lo}, {lo}, {olo};");
            b.L($"max.{cmp} {hi}, {hi}, {ohi};");
            b.L($"st.shared.b64 [{mine}], {lo};");
            b.L($"st.shared.b64 [{mine}+8], {hi};");
            b.Label($"SKIP{s}");
            b.L("bar.sync 0;");
        }

        string p0 = b.P(), oOff = b.RD(), oAddr = b.RD();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra DONE;");
        if (u64) b.L($"min.u64 {hi}, {hi}, 0x7FFFFFFFFFFFFFFF;");
        b.L($"mul.wide.u32 {oOff}, {ctaid}, 16;");
        b.L($"add.u64 {oAddr}, {dst}, {oOff};");
        b.L($"st.global.b64 [{oAddr}], {lo};");
        b.L($"st.global.b64 [{oAddr}+8], {hi};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u32 p_n, .param .u64 p_out", RangeThreads * 16);
    }

    // ---------------------------------------------------------------- B transposed into packed rows, through a shared tile

    /// <summary>Bt[j][0..kp) = B[0..k)[j] in the kind's packing, zero past k: blocks of 64 columns × 32 rows of k, read along the
    /// stored rows (coalesced), turned in shared memory, written along the packed rows (coalesced). With conv (item 4g) B is
    /// the convolution's columns read from the image (p_x: C × H × W): B[kk][j] = x[c][oh·sh − ph + kh·dh][ow·sw − pw + kw·dw]
    /// for kk = (c, kh, kw), j = (oh, ow), 0 off the image — the implicit im2col.</summary>
    /// <summary>The transposed packing of a 16-bit type that is its tensor kind's own (float16 → f16, bfloat16 → bf16; n
    /// even): stored k × n, packed n × kp (zeros past k). A 64 × 64 tile a block: each warp row loads 32 words (128
    /// bytes) of a stored row, the tile transposed through shared memory, each warp stores 32 words of a packed column
    /// (pairs of k values); 32-bit accesses both ways, the shared reads without bank conflicts.</summary>
    public static string PackTransposed16(Gpu gpu, Fmt f, Kind kind, out string entry)
    {
        entry = $"packt16_{f.Name}_{kind.Name}";
        if (f.Bits != 16 || kind.Bits != 16) throw new ArgumentException("a 16-bit type in its own 16-bit tensor kind");
        const int Tile = 64, Stride = 66;                                    // stride in halves: 132 bytes, a word multiple
        var b = new PtxBuilder();
        string tid = b.R(), col0 = b.R(), k0 = b.R(), smem = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {col0}, %ctaid.x;");
        b.L($"mul.lo.u32 {col0}, {col0}, {Tile};");
        b.L($"mov.u32 {k0}, %ctaid.y;");
        b.L($"mul.lo.u32 {k0}, {k0}, {Tile};");
        b.L($"mov.u32 {smem}, smem;");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_y"), n = Param32(b, "p_n"), k = Param32(b, "p_k"), kp = Param32(b, "p_kp");
        // Load: word wc = tid & 31 of the tile's row (columns 2wc, 2wc + 1), rows (tid >> 5) + 8i.
        string wc = b.R(), c2 = b.R(), nn = b.R(), pCol = b.P(), r0 = b.R();
        b.L($"and.b32 {wc}, {tid}, 31;");
        b.L($"shl.b32 {c2}, {wc}, 1;");
        b.L($"add.u32 {nn}, {col0}, {c2};");
        b.L($"setp.lt.u32 {pCol}, {nn}, {n};");
        b.L($"shr.u32 {r0}, {tid}, 5;");
        for (int i = 0; i < Tile / 8; i++)
        {
            string row = b.R(), kk = b.R(), pk = b.P(), word = b.R(), off = b.RD(), addr = b.RD(), lo = b.R(), hi = b.R(), s0 = b.R();
            b.L($"add.u32 {row}, {r0}, {8 * i};");
            b.L($"add.u32 {kk}, {k0}, {row};");
            b.L($"setp.lt.u32 {pk}, {kk}, {k};");
            b.L($"and.pred {pk}, {pk}, {pCol};");
            b.L($"mov.b32 {word}, 0;");
            b.L($"mad.wide.u32 {off}, {kk}, {n}, {Cvt64Of(b, nn)};");
            b.L($"shl.b64 {off}, {off}, 1;");
            b.L($"add.u64 {addr}, {src}, {off};");
            b.L($"@{pk} ld.global.b32 {word}, [{addr}];");
            b.L($"and.b32 {lo}, {word}, 0xFFFF;");
            b.L($"shr.u32 {hi}, {word}, 16;");
            // T[c][row] at (c·Stride + row)·2: columns 2wc and 2wc + 1.
            b.L($"mad.lo.u32 {s0}, {c2}, {Stride}, {row};");
            b.L($"shl.b32 {s0}, {s0}, 1;");
            b.L($"add.u32 {s0}, {s0}, {smem};");
            b.L($"st.shared.u16 [{s0}], {lo};");
            b.L($"st.shared.u16 [{s0}+{Stride * 2}], {hi};");
        }

        b.L("bar.sync 0;");
        // Store: word w = tid + 256j → column w >> 5, word w & 31 (k values 2(w & 31), +1) of the tile's 64.
        string kw0 = b.R(), wordsK = b.R();
        b.L($"shr.u32 {kw0}, {k0}, 1;");
        b.L($"shr.u32 {wordsK}, {kp}, 1;");
        for (int j = 0; j < Tile * Tile / 2 / 256; j++)
        {
            string w = b.R(), c = b.R(), wi = b.R(), ncol = b.R(), pn = b.P(), kwi = b.R(), sAddr = b.R(), word = b.R(), wIdx = b.RD(), oAddr = b.RD();
            b.L($"add.u32 {w}, {tid}, {j * 256};");
            b.L($"shr.u32 {c}, {w}, 5;");
            b.L($"and.b32 {wi}, {w}, 31;");
            b.L($"add.u32 {ncol}, {col0}, {c};");
            b.L($"add.u32 {kwi}, {kw0}, {wi};");
            b.L($"setp.ge.u32 {pn}, {ncol}, {n};");
            string pk = b.P();
            b.L($"setp.ge.u32 {pk}, {kwi}, {wordsK};");
            b.L($"or.pred {pn}, {pn}, {pk};");
            b.L($"@{pn} bra STORED{j};");
            b.L($"mad.lo.u32 {sAddr}, {c}, {Stride}, {wi};");
            b.L($"add.u32 {sAddr}, {sAddr}, {wi};");
            b.L($"shl.b32 {sAddr}, {sAddr}, 1;");
            b.L($"add.u32 {sAddr}, {sAddr}, {smem};");
            b.L($"ld.shared.b32 {word}, [{sAddr}];");
            b.L($"mad.wide.u32 {wIdx}, {ncol}, {wordsK}, {Cvt64Of(b, kwi)};");
            b.L($"shl.b64 {wIdx}, {wIdx}, 2;");
            b.L($"add.u64 {oAddr}, {dst}, {wIdx};");
            b.L($"st.global.b32 [{oAddr}], {word};");
            b.Label($"STORED{j}");
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_n, .param .u32 p_k, .param .u32 p_kp", Tile * Stride * 2);
    }

    public static string PackTransposed(Gpu gpu, Fmt f, Kind kind, out string entry, bool conv = false)
    {
        entry = conv ? $"packconv_{f.Name}_{kind.Name}" : $"packt_{f.Name}_{kind.Name}";
        const int Cols = 64, Rows = 32, Stride = 33;                        // the tile: n columns, k rows; stride in elements
        int elem = kind.IsFloat ? 2 : 1;
        int wordsPerRow = Rows / kind.Per, totalWords = Cols * wordsPerRow, perThread = Math.Max(1, totalWords / 256);
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), col0 = b.R(), k0 = b.R(), smem = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"mul.lo.u32 {col0}, {bx}, {Cols};");
        b.L($"mul.lo.u32 {k0}, {by}, {Rows};");
        b.L($"mov.u32 {smem}, smem;");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_y"), n = Param32(b, "p_n"), k = Param32(b, "p_k"), kp = Param32(b, "p_kp");
        string[] g = conv ? [.. new[] { "h", "w", "kh", "kw", "sh", "sw", "ph", "pw", "dh", "dw", "ow" }.Select(name => Param32(b, "p_" + name))] : [];

        // Load: column col0 + (tid & 63), rows k0 + (tid >> 6) + 4i.
        string col = b.R(), nn = b.R(), pCol = b.P(), kBase = b.R();
        b.L($"and.b32 {col}, {tid}, 63;");
        b.L($"add.u32 {nn}, {col0}, {col};");
        b.L($"setp.lt.u32 {pCol}, {nn}, {n};");
        b.L($"shr.u32 {kBase}, {tid}, 6;");
        string ihBase = "", iwBase = "", khw = "";
        if (conv)
        {
            // The column's output pixel: ih = oh·sh − ph + kh·dh, iw = ow·sw − pw + kw·dw (the kh, kw terms per k-row).
            string oh = b.R(), ow = b.R();
            ihBase = b.R(); iwBase = b.R(); khw = b.R();
            b.L($"div.u32 {oh}, {nn}, {g[10]};");
            b.L($"rem.u32 {ow}, {nn}, {g[10]};");
            b.L($"mul.lo.s32 {ihBase}, {oh}, {g[4]};");
            b.L($"sub.s32 {ihBase}, {ihBase}, {g[6]};");
            b.L($"mul.lo.s32 {iwBase}, {ow}, {g[5]};");
            b.L($"sub.s32 {iwBase}, {iwBase}, {g[7]};");
            b.L($"mul.lo.u32 {khw}, {g[2]}, {g[3]};");
        }

        for (int i = 0; i < 8; i++)
        {
            string kr = b.R(), kk = b.R(), pk = b.P(), idx = b.R(), code = b.R(), sAddr = b.R();
            b.L($"add.u32 {kr}, {kBase}, {4 * i};");
            b.L($"add.u32 {kk}, {k0}, {kr};");
            b.L($"setp.lt.u32 {pk}, {kk}, {k};");
            b.L($"and.pred {pk}, {pk}, {pCol};");
            b.L($"mov.b32 {code}, 0;");
            b.L($"@!{pk} bra LOADED{i};");
            if (conv)
            {
                string c = b.R(), r = b.R(), kh = b.R(), kw = b.R(), ih = b.R(), iw = b.R(), pin = b.P(), pin2 = b.P();
                b.L($"div.u32 {c}, {kk}, {khw};");
                b.L($"rem.u32 {r}, {kk}, {khw};");
                b.L($"div.u32 {kh}, {r}, {g[3]};");
                b.L($"rem.u32 {kw}, {r}, {g[3]};");
                b.L($"mad.lo.s32 {ih}, {kh}, {g[8]}, {ihBase};");
                b.L($"mad.lo.s32 {iw}, {kw}, {g[9]}, {iwBase};");
                b.L($"setp.lt.u32 {pin}, {ih}, {g[0]};");                       // a negative index reads as a large one
                b.L($"setp.lt.u32 {pin2}, {iw}, {g[1]};");
                b.L($"and.pred {pin}, {pin}, {pin2};");
                b.L($"@!{pin} bra LOADED{i};");
                b.L($"mad.lo.u32 {idx}, {c}, {g[0]}, {ih};");
                b.L($"mad.lo.u32 {idx}, {idx}, {g[1]}, {iw};");
            }
            else b.L($"mad.lo.u32 {idx}, {kk}, {n}, {nn};");
            if (kind.IsFloat)
            {
                string v = Codec.UnpackAt(b, f, src, idx, Lane.F32);
                if (kind == Kind.F16)
                {
                    string h = b.H();
                    b.L($"cvt.rn.f16.f32 {h}, {v};");
                    b.L($"cvt.u32.u16 {code}, {h};");
                }
                else
                {
                    b.L($"mov.b32 {code}, {v};");
                    b.L($"shr.u32 {code}, {code}, 16;");
                }
            }
            else
            {
                string v = Codec.UnpackAt(b, f, src, idx, f.Signed ? Lane.S32 : Lane.U32);
                b.L($"mov.b32 {code}, {v};");
            }

            b.Label($"LOADED{i}");
            // T[col][kr]
            b.L($"mad.lo.u32 {sAddr}, {col}, {Stride * elem}, {smem};");
            b.L($"mad.lo.u32 {sAddr}, {kr}, {elem}, {sAddr};");
            b.L(elem == 2 ? $"st.shared.u16 [{sAddr}], {code};" : $"st.shared.u8 [{sAddr}], {code};");
        }

        b.L("bar.sync 0;");

        // Store: word w = tid + 256j → column w / wordsPerRow, word w % wordsPerRow of the column's 32 values.
        string wordsK = b.R(), kw0 = b.R();
        b.L($"shr.u32 {wordsK}, {kp}, {kind.Per switch { 8 => 3, 4 => 2, _ => 1 }};");
        b.L($"shr.u32 {kw0}, {k0}, {kind.Per switch { 8 => 3, 4 => 2, _ => 1 }};");
        for (int j = 0; j < perThread; j++)
        {
            string w = b.R(), crow = b.R(), wi = b.R(), word = b.R(), pw = b.P(), ncol = b.R(), pn = b.P(), tBase = b.R();
            b.L($"add.u32 {w}, {tid}, {j * 256};");
            b.L($"setp.ge.u32 {pw}, {w}, {totalWords};");
            b.L($"@{pw} bra STORED{j};");
            b.L($"div.u32 {crow}, {w}, {wordsPerRow};");
            b.L($"rem.u32 {wi}, {w}, {wordsPerRow};");
            b.L($"add.u32 {ncol}, {col0}, {crow};");
            b.L($"setp.ge.u32 {pn}, {ncol}, {n};");
            b.L($"@{pn} bra STORED{j};");
            b.L($"mad.lo.u32 {tBase}, {crow}, {Stride * elem}, {smem};");
            b.L($"mad.lo.u32 {tBase}, {wi}, {kind.Per * elem}, {tBase};");
            b.L($"mov.b32 {word}, 0;");
            for (int v = 0; v < kind.Per; v++)
            {
                string e = b.R();
                b.L(elem == 2 ? $"ld.shared.u16 {e}, [{tBase}+{v * 2}];" : $"ld.shared.u8 {e}, [{tBase}+{v}];");
                b.L($"bfi.b32 {word}, {e}, {word}, {kind.Bits * v}, {kind.Bits};");
            }

            string wIdx = b.RD(), oOff = b.RD(), oAddr = b.RD();
            b.L($"mul.wide.u32 {wIdx}, {ncol}, {wordsK};");
            b.L($"add.u64 {wIdx}, {wIdx}, {Cvt64(b, kw0, wi)};");
            b.L($"shl.b64 {oOff}, {wIdx}, 2;");
            b.L($"add.u64 {oAddr}, {dst}, {oOff};");
            b.L($"st.global.b32 [{oAddr}], {word};");
            b.Label($"STORED{j}");
        }

        b.L("ret;");
        string convParams = conv ? ", .param .u32 p_h, .param .u32 p_w, .param .u32 p_kh, .param .u32 p_kw, .param .u32 p_sh, .param .u32 p_sw, .param .u32 p_ph, .param .u32 p_pw, .param .u32 p_dh, .param .u32 p_dw, .param .u32 p_ow" : "";
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_n, .param .u32 p_k, .param .u32 p_kp" + convParams, Cols * Stride * elem);
    }

    // kw0 + wi as a 64-bit register.
    private static string Cvt64(PtxBuilder b, string a, string c)
    {
        string sum = b.R(), wide = b.RD();
        b.L($"add.u32 {sum}, {a}, {c};");
        b.L($"cvt.u64.u32 {wide}, {sum};");
        return wide;
    }

    // ---------------------------------------------------------------- rows: softmax and rms_norm

    public const int RowThreads = 256;

    /// <summary>One block per row; the row's words (cols·bits a multiple of the word) at tid, tid + 256, …; three passes.</summary>
    public static string Softmax(Gpu gpu, Fmt f, Lane lane, int threads, out string entry)
    {
        entry = $"softmax_{f.Name}_{lane.Name}_t{threads}";
        var b = new PtxBuilder();
        var (tid, src, dst, wordsPerRow, rowSrc, rowDst) = RowPrologue(b, f);
        string max = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {max}, {lane.Literal(double.NegativeInfinity)};");
        RowPass(b, f, lane, tid, wordsPerRow, rowSrc, null, "MAX", (v, _) => b.L($"max.{lane.Op} {max}, {max}, {v};"), threads);
        max = BlockReduce(b, lane, tid, max, "max", threads);
        string sum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
        RowPass(b, f, lane, tid, wordsPerRow, rowSrc, null, "SUM", (v, _) =>
        {
            string d = lane.NewReg(b);
            b.L($"sub.rn.{lane.Op} {d}, {v}, {max};");
            string e = Codec.Exp(b, lane, d);
            b.L(lane.Add(sum, sum, e));
        }, threads);
        sum = BlockReduce(b, lane, tid, sum, "add", threads);
        string scale = lane.NewReg(b);
        b.L($"div.rn.{lane.Op} {scale}, {lane.Literal(1)}, {sum};");
        RowPass(b, f, lane, tid, wordsPerRow, rowSrc, rowDst, "OUT", (v, outReg) =>
        {
            string d = lane.NewReg(b);
            b.L($"sub.rn.{lane.Op} {d}, {v}, {max};");
            string e = Codec.Exp(b, lane, d);
            Quotient(b, lane, outReg, e, sum, scale);
        }, threads);
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_cols", ReduceBytes);
    }

    public static string RmsNorm(Gpu gpu, Fmt f, Lane lane, int threads, out string entry)
    {
        entry = $"rms_{f.Name}_{lane.Name}_t{threads}";
        var b = new PtxBuilder();
        var (tid, src, dst, wordsPerRow, rowSrc, rowDst) = RowPrologue(b, f);
        string eps = lane.NewReg(b), cols = b.R();
        b.L($"ld.param.u32 {cols}, [p_cols];");
        if (lane.Wide) b.L($"ld.param.f64 {eps}, [p_eps];");
        else b.L($"ld.param.f32 {eps}, [p_eps];");
        string sum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
        RowPass(b, f, lane, tid, wordsPerRow, rowSrc, null, "SQ", (v, _) => b.L(lane.Fma(sum, v, v, sum)), threads);
        sum = BlockReduce(b, lane, tid, sum, "add", threads);
        string count = lane.NewReg(b), mean = lane.NewReg(b), root = lane.NewReg(b), scale = lane.NewReg(b);
        b.L($"cvt.rn.{lane.Op}.u32 {count}, {cols};");
        b.L($"div.rn.{lane.Op} {mean}, {sum}, {count};");
        b.L(lane.Add(mean, mean, eps));
        b.L($"sqrt.rn.{lane.Op} {root}, {mean};");
        b.L($"div.rn.{lane.Op} {scale}, {lane.Literal(1)}, {root};");
        RowPass(b, f, lane, tid, wordsPerRow, rowSrc, rowDst, "OUT", (v, outReg) => b.L(lane.Mul(outReg, v, scale)), threads);
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_y, .param .u32 p_cols, .param .{(lane.Wide ? "f64" : "f32")} p_eps", ReduceBytes);
    }

    // d = x / y from y's rounded reciprocal r: x·r, then one residual step (x − q·y exact by fma) — near-correctly rounded,
    // one rounding fewer than x·r alone.
    private static void Quotient(PtxBuilder b, Lane lane, string d, string x, string y, string r)
    {
        string q = lane.NewReg(b), res = lane.NewReg(b);
        b.L(lane.Mul(q, x, r));
        b.L($"neg.{lane.Op} {res}, {q};");
        b.L(lane.Fma(res, res, y, x));
        b.L(lane.Fma(d, res, r, q));
    }

    /// <summary>Softmax with the row in registers: each thread's wordsPerThread words (tid, tid + 256, …) loaded once, exp once.</summary>
    public static string SoftmaxRegs(Gpu gpu, Fmt f, Lane lane, int wordsPerThread, int threads, bool vector, out string entry)
    {
        entry = $"softmax_{f.Name}_{lane.Name}_r{wordsPerThread}_t{threads}{(vector ? "_v" : "")}";
        var b = new PtxBuilder();
        var (tid, _, _, words, rowSrc, rowDst) = RowPrologue(b, f);
        var (present, inWords, values) = RowLoad(b, f, lane, tid, words, rowSrc, wordsPerThread, lane.Literal(double.NegativeInfinity), threads, vector);
        string max = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {max}, {lane.Literal(double.NegativeInfinity)};");
        foreach (var row in values) foreach (var v in row) b.L($"max.{lane.Op} {max}, {max}, {v};");
        max = BlockReduce(b, lane, tid, max, "max", threads);
        string sum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
        var exps = new string[wordsPerThread][];
        for (int q = 0; q < wordsPerThread; q++)
        {
            exps[q] = new string[f.Per];
            for (int i = 0; i < f.Per; i++)
            {
                string d = lane.NewReg(b);
                b.L($"sub.rn.{lane.Op} {d}, {values[q][i]}, {max};");
                exps[q][i] = Codec.Exp(b, lane, d);
                b.L(lane.Add(sum, sum, exps[q][i]));
            }
        }

        sum = BlockReduce(b, lane, tid, sum, "add", threads);
        string scale = lane.NewReg(b);
        b.L($"div.rn.{lane.Op} {scale}, {lane.Literal(1)}, {sum};");
        RowStore(b, f, lane, present, inWords, rowDst, wordsPerThread, (q, i) =>
        {
            string o = lane.NewReg(b);
            Quotient(b, lane, o, exps[q][i], sum, scale);
            return o;
        }, words, vector);
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_y, .param .u32 p_cols", ReduceBytes);
    }

    public static string RmsNormRegs(Gpu gpu, Fmt f, Lane lane, int wordsPerThread, int threads, bool vector, out string entry)
    {
        entry = $"rms_{f.Name}_{lane.Name}_r{wordsPerThread}_t{threads}{(vector ? "_v" : "")}";
        var b = new PtxBuilder();
        var (tid, _, _, words, rowSrc, rowDst) = RowPrologue(b, f);
        string eps = lane.NewReg(b), cols = b.R();
        b.L($"ld.param.u32 {cols}, [p_cols];");
        b.L(lane.Wide ? $"ld.param.f64 {eps}, [p_eps];" : $"ld.param.f32 {eps}, [p_eps];");
        var (present, inWords, values) = RowLoad(b, f, lane, tid, words, rowSrc, wordsPerThread, lane.Zero, threads, vector);
        string sum = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {sum}, {lane.Zero};");
        foreach (var row in values) foreach (var v in row) b.L(lane.Fma(sum, v, v, sum));
        sum = BlockReduce(b, lane, tid, sum, "add", threads);
        string count = lane.NewReg(b), mean = lane.NewReg(b), root = lane.NewReg(b), scale = lane.NewReg(b);
        b.L($"cvt.rn.{lane.Op}.u32 {count}, {cols};");
        b.L($"div.rn.{lane.Op} {mean}, {sum}, {count};");
        b.L(lane.Add(mean, mean, eps));
        b.L($"sqrt.rn.{lane.Op} {root}, {mean};");
        b.L($"div.rn.{lane.Op} {scale}, {lane.Literal(1)}, {root};");
        RowStore(b, f, lane, present, inWords, rowDst, wordsPerThread, (q, i) =>
        {
            string o = lane.NewReg(b);
            b.L(lane.Mul(o, values[q][i], scale));
            return o;
        }, words, vector);
        b.L("ret;");
        return b.Finish(Header(gpu), entry, $".param .u64 p_x, .param .u64 p_y, .param .u32 p_cols, .param .{(lane.Wide ? "f64" : "f32")} p_eps", ReduceBytes);
    }

    // This thread's words of the row into lane registers; a word past the row gives `missing` in every value.
    // The vector mapping: a thread's four words consecutive (tid·4 ..), loaded and stored as one 16-byte vector when all
    // four are in the row (the row's words a multiple of four keep the vectors aligned).
    private static bool Vectored(Fmt f, int wordsPerThread, bool vector) => vector && f.WordBits == 32 && wordsPerThread == 4;

    private static (string[] Present, string[] Words, string[][] Values) RowLoad(PtxBuilder b, Fmt f, Lane lane, string tid, string words, string rowSrc, int wordsPerThread, string missing, int threads, bool vector = false)
    {
        var present = new string[wordsPerThread];
        var inWords = new string[wordsPerThread];
        var values = new string[wordsPerThread][];
        string wt = f.WordBits == 64 ? "b64" : "b32";
        bool vec = Vectored(f, wordsPerThread, vector);
        string? pAll = null, addr0 = null;
        var vWords = new string[4];
        if (vec)
        {
            string w0 = b.R(), wEnd = b.R(), off0 = b.RD();
            pAll = b.P(); addr0 = b.RD();
            b.L($"shl.b32 {w0}, {tid}, 2;");
            b.L($"add.u32 {wEnd}, {w0}, 4;");
            b.L($"setp.le.u32 {pAll}, {wEnd}, {words};");
            b.L($"mul.wide.u32 {off0}, {w0}, 4;");
            b.L($"add.u64 {addr0}, {rowSrc}, {off0};");
            for (int q = 0; q < 4; q++) { vWords[q] = b.R(); b.L($"mov.b32 {vWords[q]}, 0;"); }
            b.L($"@{pAll} ld.global.v4.b32 {{{vWords[0]}, {vWords[1]}, {vWords[2]}, {vWords[3]}}}, [{addr0}];");
        }

        for (int q = 0; q < wordsPerThread; q++)
        {
            string w = b.R(), p = b.P(), off = b.RD(), addr = b.RD(), word = vec ? vWords[q] : f.WordBits == 64 ? b.RD() : b.R();
            if (vec)
            {
                string pOne = b.P();
                b.L($"mad.lo.u32 {w}, {tid}, 4, {q};");
                b.L($"setp.lt.u32 {p}, {w}, {words};");
                b.L($"not.pred {pOne}, {pAll};");
                b.L($"and.pred {pOne}, {pOne}, {p};");
                b.L($"@{pOne} ld.global.b32 {word}, [{addr0}+{q * 4}];");
            }
            else
            {
                b.L($"add.u32 {w}, {tid}, {q * threads};");
                b.L($"setp.lt.u32 {p}, {w}, {words};");
                b.L($"mul.wide.u32 {off}, {w}, {f.WordBits / 8};");
                b.L($"add.u64 {addr}, {rowSrc}, {off};");
                b.L($"mov.{wt} {word}, 0;");
                b.L($"@{p} ld.global.{wt} {word}, [{addr}];");
            }

            present[q] = p;
            inWords[q] = w;
            values[q] = new string[f.Per];
            for (int i = 0; i < f.Per; i++)
            {
                string v = Codec.Unpack(b, f, word, i, lane);
                string kept = lane.NewReg(b);
                b.L($"selp.{lane.Reg} {kept}, {v}, {missing}, {p};");
                values[q][i] = kept;
            }
        }

        return (present, inWords, values);
    }

    private static void RowStore(PtxBuilder b, Fmt f, Lane lane, string[] present, string[] inWords, string rowDst, int wordsPerThread, Func<int, int, string> output, string? words = null, bool vector = false)
    {
        string wt = f.WordBits == 64 ? "b64" : "b32";
        bool vec = Vectored(f, wordsPerThread, vector) && words is not null;
        var outWords = new string[wordsPerThread];
        for (int q = 0; q < wordsPerThread; q++)
        {
            string outWord = f.WordBits == 64 ? b.RD() : b.R();
            if (f.WordBits == 32) b.L($"mov.b32 {outWord}, 0;");
            for (int i = 0; i < f.Per; i++) Codec.Pack(b, f, output(q, i), lane, i, outWord);
            outWords[q] = outWord;
        }

        if (vec)
        {
            string wEnd = b.R(), pAll = b.P(), off0 = b.RD(), addr0 = b.RD();
            b.L($"add.u32 {wEnd}, {inWords[0]}, 4;");
            b.L($"setp.le.u32 {pAll}, {wEnd}, {words};");
            b.L($"mul.wide.u32 {off0}, {inWords[0]}, 4;");
            b.L($"add.u64 {addr0}, {rowDst}, {off0};");
            b.L($"@{pAll} st.global.v4.b32 [{addr0}], {{{outWords[0]}, {outWords[1]}, {outWords[2]}, {outWords[3]}}};");
            for (int q = 0; q < 4; q++)
            {
                string pOne = b.P();
                b.L($"not.pred {pOne}, {pAll};");
                b.L($"and.pred {pOne}, {pOne}, {present[q]};");
                b.L($"@{pOne} st.global.b32 [{addr0}+{q * 4}], {outWords[q]};");
            }

            return;
        }

        for (int q = 0; q < wordsPerThread; q++)
        {
            string off = b.RD(), addr = b.RD();
            b.L($"mul.wide.u32 {off}, {inWords[q]}, {f.WordBits / 8};");
            b.L($"add.u64 {addr}, {rowDst}, {off};");
            b.L($"@{present[q]} st.global.{wt} [{addr}], {outWords[q]};");
        }
    }

    private static (string Tid, string Src, string Dst, string WordsPerRow, string RowSrc, string RowDst) RowPrologue(PtxBuilder b, Fmt f)
    {
        string tid = b.R(), row = b.R(), cols = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        b.L($"ld.param.u32 {cols}, [p_cols];");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_y");
        string words = b.R(), off = b.RD(), rowSrc = b.RD(), rowDst = b.RD();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {cols};");
        else { b.L($"mul.lo.u32 {words}, {cols}, {f.Bits};"); b.L($"shr.u32 {words}, {words}, 5;"); }
        b.L($"mul.wide.u32 {off}, {row}, {words};");
        b.L($"mul.lo.u64 {off}, {off}, {f.WordBits / 8};");
        b.L($"add.u64 {rowSrc}, {src}, {off};");
        b.L($"add.u64 {rowDst}, {dst}, {off};");
        return (tid, src, dst, words, rowSrc, rowDst);
    }

    // A pass over the row's words: for each word of this thread, each value unpacked into the lane and handed to
    // `each` (with an output register when the pass stores), the outputs packed and stored.
    private static void RowPass(PtxBuilder b, Fmt f, Lane lane, string tid, string words, string rowSrc, string? rowDst, string label, Action<string, string> each, int threads)
    {
        string w = b.R(), p = b.P();
        b.L($"mov.u32 {w}, {tid};");
        b.Label(label);
        b.L($"setp.ge.u32 {p}, {w}, {words};");
        b.L($"@{p} bra {label}_END;");
        string off = b.RD(), addr = b.RD(), word = f.WordBits == 64 ? b.RD() : b.R();
        b.L($"mul.wide.u32 {off}, {w}, {f.WordBits / 8};");
        b.L($"add.u64 {addr}, {rowSrc}, {off};");
        b.L($"ld.global.{(f.WordBits == 64 ? "b64" : "b32")} {word}, [{addr}];");
        string? outWord = rowDst is null ? null : f.WordBits == 64 ? b.RD() : b.R();
        if (outWord is not null && f.WordBits == 32) b.L($"mov.b32 {outWord}, 0;");
        for (int i = 0; i < f.Per; i++)
        {
            string v = Codec.Unpack(b, f, word, i, lane);
            if (outWord is null) each(v, "");
            else
            {
                string o = lane.NewReg(b);
                each(v, o);
                Codec.Pack(b, f, o, lane, i, outWord);
            }
        }

        if (outWord is not null)
        {
            string dAddr = b.RD();
            b.L($"add.u64 {dAddr}, {rowDst}, {off};");
            b.L($"st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{dAddr}], {outWord};");
        }

        b.L($"add.u32 {w}, {w}, {threads};");
        b.L($"bra {label};");
        b.Label($"{label}_END");
    }

    /// <summary>Shared bytes a reduction uses: a slot per warp and one for the result.</summary>
    private const int ReduceBytes = 33 * 8;

    // The block's reduction of a lane value: butterflies within each warp, the warps' results through shared memory,
    // the first warp's butterfly over those, the result read by all. Two barriers.
    private static string BlockReduce(PtxBuilder b, Lane lane, string tid, string v, string op, int threads)
    {
        int warps = threads / 32;
        string tag = $"RED{b.P().TrimStart('%')}";
        string acc = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {acc}, {v};");
        WarpReduce(b, lane, acc, op);
        string smem = b.R(), laneId = b.R(), warp = b.R(), slot = b.R(), p0 = b.P();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"and.b32 {laneId}, {tid}, 31;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"mad.lo.u32 {slot}, {warp}, 8, {smem};");
        b.L($"setp.ne.u32 {p0}, {laneId}, 0;");
        b.L($"@!{p0} st.shared.{lane.Mem} [{slot}], {acc};");
        b.L("bar.sync 0;");
        string pw = b.P(), pl = b.P(), mine = b.R(), identity = op == "max" ? ReduceIdentity(lane, "max") : op == "min" ? ReduceIdentity(lane, "min") : lane.Zero;
        b.L($"setp.ne.u32 {pw}, {warp}, 0;");
        b.L($"@{pw} bra {tag}_SKIP;");
        b.L($"setp.lt.u32 {pl}, {laneId}, {warps};");
        b.L($"mad.lo.u32 {mine}, {laneId}, 8, {smem};");
        b.L($"mov.{lane.Reg} {acc}, {identity};");
        b.L($"@{pl} ld.shared.{lane.Mem} {acc}, [{mine}];");
        WarpReduce(b, lane, acc, op);
        b.L($"@!{p0} st.shared.{lane.Mem} [{smem}+256], {acc};");
        b.Label($"{tag}_SKIP");
        b.L("bar.sync 0;");
        string result = lane.NewReg(b);
        b.L($"ld.shared.{lane.Mem} {result}, [{smem}+256];");
        return result;
    }

    // A block's sum of f32 pairs hi + lo (the same butterflies and shared slots as BlockReduce): each step's hi sum split
    // exactly (two-sum), its error added to the low parts, the pair renormalized. All in the f32 rate, where a float64 join
    // runs at the device's f64 rate. The result read by all.
    private static (string Hi, string Lo) BlockReducePair(PtxBuilder b, string tid, string hi, string lo, int threads)
    {
        int warps = threads / 32;
        string tag = $"REDP{b.P().TrimStart('%')}";
        string h = b.F(), l = b.F();
        b.L($"mov.f32 {h}, {hi};");
        b.L($"mov.f32 {l}, {lo};");
        void Add(string oh, string ol)
        {
            string s = b.F(), bv = b.F(), e = b.F(), t = b.F();
            b.L($"add.rn.f32 {s}, {h}, {oh};");
            b.L($"sub.rn.f32 {bv}, {s}, {h};");
            b.L($"sub.rn.f32 {t}, {s}, {bv};");
            b.L($"sub.rn.f32 {t}, {h}, {t};");
            b.L($"sub.rn.f32 {e}, {oh}, {bv};");
            b.L($"add.rn.f32 {e}, {e}, {t};");
            b.L($"add.rn.f32 {l}, {l}, {ol};");
            b.L($"add.rn.f32 {l}, {l}, {e};");
            b.L($"add.rn.f32 {h}, {s}, {l};");
            b.L($"sub.rn.f32 {t}, {h}, {s};");
            b.L($"sub.rn.f32 {l}, {l}, {t};");
        }

        void Warp()
        {
            for (int off = 16; off >= 1; off /= 2)
            {
                string oh = b.F(), ol = b.F();
                b.L($"shfl.sync.bfly.b32 {oh}, {h}, {off}, 0x1f, 0xffffffff;");
                b.L($"shfl.sync.bfly.b32 {ol}, {l}, {off}, 0x1f, 0xffffffff;");
                Add(oh, ol);
            }
        }

        Warp();
        string smem = b.R(), laneId = b.R(), warp = b.R(), slot = b.R(), p0 = b.P();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"and.b32 {laneId}, {tid}, 31;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"mad.lo.u32 {slot}, {warp}, 8, {smem};");
        b.L($"setp.ne.u32 {p0}, {laneId}, 0;");
        b.L($"@!{p0} st.shared.v2.f32 [{slot}], {{{h}, {l}}};");
        b.L("bar.sync 0;");
        string pw = b.P(), pl = b.P(), mine = b.R();
        b.L($"setp.ne.u32 {pw}, {warp}, 0;");
        b.L($"@{pw} bra {tag}_SKIP;");
        b.L($"setp.lt.u32 {pl}, {laneId}, {warps};");
        b.L($"mad.lo.u32 {mine}, {laneId}, 8, {smem};");
        b.L($"mov.f32 {h}, 0f00000000;");
        b.L($"mov.f32 {l}, 0f00000000;");
        b.L($"@{pl} ld.shared.v2.f32 {{{h}, {l}}}, [{mine}];");
        Warp();
        b.L($"@!{p0} st.shared.v2.f32 [{smem}+256], {{{h}, {l}}};");
        b.Label($"{tag}_SKIP");
        b.L("bar.sync 0;");
        string rh = b.F(), rl = b.F();
        b.L($"ld.shared.v2.f32 {{{rh}, {rl}}}, [{smem}+256];");
        return (rh, rl);
    }

    /// <summary>The value a reduction starts from: −∞ / the smallest value for max, +∞ / the largest for min, zero for sums.</summary>
    private static string ReduceIdentity(Lane lane, string op) => op switch
    {
        "max" => lane.IsFloat ? lane.Literal(double.NegativeInfinity) : lane.Name[0] == 'u' ? "0" : lane.Wide ? "0x8000000000000000" : "0x80000000",
        "min" => lane.IsFloat ? lane.Literal(double.PositiveInfinity) : lane.Name[0] == 'u' ? (lane.Wide ? "0xFFFFFFFFFFFFFFFF" : "0xFFFFFFFF") : lane.Wide ? "0x7FFFFFFFFFFFFFFF" : "0x7FFFFFFF",
        _ => lane.Zero,
    };

    private static void WarpReduce(PtxBuilder b, Lane lane, string acc, string op)
    {
        for (int off = 16; off >= 1; off /= 2)
        {
            string other = lane.NewReg(b);
            if (lane.Wide)
            {
                string lo = b.R(), hi = b.R(), lo2 = b.R(), hi2 = b.R();
                b.L($"mov.b64 {{{lo}, {hi}}}, {acc};");
                b.L($"shfl.sync.bfly.b32 {lo2}, {lo}, {off}, 0x1f, 0xffffffff;");
                b.L($"shfl.sync.bfly.b32 {hi2}, {hi}, {off}, 0x1f, 0xffffffff;");
                b.L($"mov.b64 {other}, {{{lo2}, {hi2}}};");
            }
            else
            {
                string bits = b.R(), bits2 = b.R();
                b.L($"mov.b32 {bits}, {acc};");
                b.L($"shfl.sync.bfly.b32 {bits2}, {bits}, {off}, 0x1f, 0xffffffff;");
                b.L($"mov.b32 {other}, {bits2};");
            }

            b.L(op == "max" ? $"max.{lane.Op} {acc}, {acc}, {other};" : op == "min" ? $"min.{lane.Op} {acc}, {acc}, {other};" : lane.Add(acc, acc, other));
        }
    }
}
