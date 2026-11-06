// The device's block encoders (Blocks.cs): float32 lane storage into Q8_0 / Q4_0 blocks, a thread per block (the
// block's max, its scale as f16, the values rounded to the integers). Q4_K and Q6_K are decoded on the device (Codec) and
// encoded on the CPU; their device encoders are the step past this.
namespace Qiyas;

internal static partial class Ptx
{
    public static string LanesToBlock(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"lanes_to_{f.Name}_{lane.Name}";
        if (f.Name is not ("q8_0" or "q4_0")) throw new NotSupportedException($"{f.Name}: the device encodes Q8_0 and Q4_0 blocks; this format's encoder is the step past this");
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), blk = b.R(), x = Param64(b, "p_x"), outp = Param64(b, "p_out"), n = Param32(b, "p_n"), blocks = b.R(), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {blk}, {bid}, 256, {tid};");
        b.L($"add.u32 {blocks}, {n}, {f.BlockValues - 1};");
        b.L($"div.u32 {blocks}, {blocks}, {f.BlockValues};");
        b.L($"setp.ge.u32 {pEnd}, {blk}, {blocks};");
        b.L($"@{pEnd} bra DONE;");
        string first = b.R(), last = b.R(), xAddr = b.RD(), bAddr = b.RD();
        b.L($"mul.lo.u32 {first}, {blk}, {f.BlockValues};");
        b.L($"sub.u32 {last}, {n}, 1;");
        b.L($"mul.wide.u32 {xAddr}, {first}, {lane.Size};");
        b.L($"add.u64 {xAddr}, {x}, {xAddr};");
        b.L($"mul.wide.u32 {bAddr}, {blk}, {f.BlockBytes};");
        b.L($"add.u64 {bAddr}, {outp}, {bAddr};");
        // The 32 values (0 past the count) and the block's max |x| (Q4_0: the signed max).
        var vals = new string[32];
        string amax = b.F(), maxv = b.F(), pIn = b.P(), idx = b.R();
        b.L($"mov.f32 {amax}, 0f00000000;");
        b.L($"mov.f32 {maxv}, 0f00000000;");
        for (int i = 0; i < 32; i++)
        {
            vals[i] = b.F();
            string raw = lane.Wide ? b.FD() : b.F();
            b.L($"add.u32 {idx}, {first}, {i};");
            b.L($"setp.le.u32 {pIn}, {idx}, {last};");
            b.L(lane.Wide ? $"mov.f64 {raw}, 0d0000000000000000;" : $"mov.f32 {raw}, 0f00000000;");
            b.L($"@{pIn} ld.global.{lane.Mem} {raw}, [{xAddr}+{i * lane.Size}];");
            if (lane.Wide) b.L($"cvt.rn.f32.f64 {vals[i]}, {raw};"); else b.L($"mov.f32 {vals[i]}, {raw};");
            string a = b.F(), pGt = b.P();
            b.L($"abs.f32 {a}, {vals[i]};");
            b.L($"setp.gt.f32 {pGt}, {a}, {amax};");
            b.L($"selp.f32 {maxv}, {vals[i]}, {maxv}, {pGt};");
            b.L($"max.f32 {amax}, {amax}, {a};");
        }

        // The scale as stored (f16) and read back, its inverse.
        string d = b.F(), h = b.H(), dBack = b.F(), id = b.F(), pZero = b.P();
        if (f.Is("q8_0")) b.L($"div.rn.f32 {d}, {amax}, 0f42FE0000;");               // / 127
        else b.L($"div.rn.f32 {d}, {maxv}, 0fC1000000;");                             // / −8
        b.L($"cvt.rn.f16.f32 {h}, {d};");
        b.L($"cvt.f32.f16 {dBack}, {h};");
        b.L($"setp.eq.f32 {pZero}, {dBack}, 0f00000000;");
        b.L($"div.rn.f32 {id}, 0f3F800000, {dBack};");
        b.L($"selp.f32 {id}, 0f00000000, {id}, {pZero};");
        b.L($"st.global.b16 [{bAddr}], {h};");
        if (f.Is("q8_0"))
        {
            for (int i = 0; i < 32; i++)
            {
                string t = b.F(), q = b.R(), qb = b.H();
                b.L($"mul.rn.f32 {t}, {vals[i]}, {id};");
                b.L($"cvt.rni.s32.f32 {q}, {t};");
                b.L($"max.s32 {q}, {q}, -127;");
                b.L($"min.s32 {q}, {q}, 127;");
                b.L($"cvt.u16.u32 {qb}, {q};");
                b.L($"st.global.b8 [{bAddr}+{2 + i}], {qb};");
            }
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                string tl = b.F(), th = b.F(), ql = b.R(), qh = b.R(), code = b.R(), cb = b.H();
                b.L($"fma.rn.f32 {tl}, {vals[i]}, {id}, 0f41080000;");               // + 8.5
                b.L($"fma.rn.f32 {th}, {vals[i + 16]}, {id}, 0f41080000;");
                b.L($"cvt.rmi.s32.f32 {ql}, {tl};");                                   // floor
                b.L($"cvt.rmi.s32.f32 {qh}, {th};");
                b.L($"max.s32 {ql}, {ql}, 0;"); b.L($"min.s32 {ql}, {ql}, 15;");
                b.L($"max.s32 {qh}, {qh}, 0;"); b.L($"min.s32 {qh}, {qh}, 15;");
                b.L($"shl.b32 {code}, {qh}, 4;");
                b.L($"or.b32 {code}, {code}, {ql};");
                b.L($"cvt.u16.u32 {cb}, {code};");
                b.L($"st.global.b8 [{bAddr}+{2 + i}], {cb};");
            }
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }
}
