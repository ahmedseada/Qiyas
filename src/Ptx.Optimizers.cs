// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's optimizer steps (Optimizers.cs): a thread per word of the parameters (its Per values), the gradients'
// word alongside, the moments as lane values at the value's index; the parameters packed back into their word.
namespace Qiyas;

internal static partial class Ptx
{
    public static string Optimizer(Gpu gpu, Fmt f, Lane lane, string name, out string entry)
    {
        entry = $"optimizer_{name}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), word = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {word}, {bid}, 256, {tid};");
        string pp = Param64(b, "p_p"), pg = Param64(b, "p_g"), pm = Param64(b, "p_m"), pv = Param64(b, "p_v"), n = Param32(b, "p_n");
        string Scalar(string nm) { string r = lane.NewReg(b); b.L($"ld.param.{lane.Op} {r}, [{nm}];"); return r; }
        string lr = Scalar("p_lr"), b1 = Scalar("p_b1"), b2 = Scalar("p_b2"), ob1 = Scalar("p_ob1"), ob2 = Scalar("p_ob2"), ic1 = Scalar("p_ic1"), ic2 = Scalar("p_ic2"), eps = Scalar("p_eps"), wd = Scalar("p_wd"), lrwd = Scalar("p_lrwd");
        string words = b.R(), pw = b.P();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else { b.L($"add.u32 {words}, {n}, {f.Per - 1};"); b.L($"div.u32 {words}, {words}, {f.Per};"); }
        b.L($"setp.ge.u32 {pw}, {word}, {words};");
        b.L($"@{pw} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32", addrP = b.RD(), addrG = b.RD(), wordP = f.WordBits == 64 ? b.RD() : b.R(), wordG = f.WordBits == 64 ? b.RD() : b.R();
        b.L($"mul.wide.u32 {addrP}, {word}, {f.WordBits / 8};");
        b.L($"add.u64 {addrG}, {pg}, {addrP};");
        b.L($"add.u64 {addrP}, {pp}, {addrP};");
        b.L($"ld.global.{wt} {wordP}, [{addrP}];");
        b.L($"ld.global.{wt} {wordG}, [{addrG}];");
        string outWord = f.WordBits == 64 ? b.RD() : b.R();
        if (f.WordBits == 32) b.L($"mov.b32 {outWord}, 0;");
        bool sgd = name == "sgd_momentum", adamw = name == "adamw";
        var natural = f.Natural;
        for (int i = 0; i < f.Per; i++)
        {
            string idx = b.R(), pIn = b.P(), mAddr = b.RD(), mv = lane.NewReg(b);
            b.L($"mad.lo.u32 {idx}, {word}, {f.Per}, {i};");
            b.L($"setp.lt.u32 {pIn}, {idx}, {n};");
            b.L($"@!{pIn} bra SKIP{i};");
            string p = Codec.Convert(b, natural, Codec.Unpack(b, f, wordP, i, natural), lane, f);
            string g = Codec.Convert(b, natural, Codec.Unpack(b, f, wordG, i, natural), lane, f);
            b.L($"mul.wide.u32 {mAddr}, {idx}, {lane.Size};");
            string mPtr = b.RD();
            b.L($"add.u64 {mPtr}, {pm}, {mAddr};");
            b.L($"ld.global.{lane.Mem} {mv}, [{mPtr}];");
            string pNew = lane.NewReg(b), t = lane.NewReg(b);
            if (sgd)
            {
                // m = b1·m + g + wd·p;  p = p − lr·m
                b.L(lane.Fma(t, wd, p, g));
                b.L(lane.Fma(mv, b1, mv, t));
                b.L(lane.Mul(t, lr, mv));
                b.L($"sub.rn.{lane.Op} {pNew}, {p}, {t};");
            }
            else
            {
                string vPtr = b.RD(), vv = lane.NewReg(b), g2 = lane.NewReg(b), mh = lane.NewReg(b), vh = lane.NewReg(b), root = lane.NewReg(b), den = lane.NewReg(b), upd = lane.NewReg(b);
                b.L($"add.u64 {vPtr}, {pv}, {mAddr};");
                b.L($"ld.global.{lane.Mem} {vv}, [{vPtr}];");
                b.L(lane.Mul(t, ob1, g));
                b.L(lane.Fma(mv, b1, mv, t));
                b.L(lane.Mul(g2, g, g));
                b.L(lane.Mul(t, ob2, g2));
                b.L(lane.Fma(vv, b2, vv, t));
                b.L(lane.Mul(mh, mv, ic1));
                b.L(lane.Mul(vh, vv, ic2));
                b.L($"sqrt.rn.{lane.Op} {root}, {vh};");
                b.L(lane.Add(den, root, eps));
                b.L(lane.Mul(upd, lr, mh));
                b.L($"div.rn.{lane.Op} {upd}, {upd}, {den};");
                b.L($"sub.rn.{lane.Op} {pNew}, {p}, {upd};");
                if (adamw)
                {
                    string decay = lane.NewReg(b);
                    b.L(lane.Mul(decay, lrwd, pNew));
                    b.L($"sub.rn.{lane.Op} {pNew}, {pNew}, {decay};");
                }

                b.L($"st.global.{lane.Mem} [{vPtr}], {vv};");
            }

            b.L($"st.global.{lane.Mem} [{mPtr}], {mv};");
            string stored = lane == natural ? pNew : Codec.Convert(b, lane, pNew, natural, f);
            Codec.Pack(b, f, stored, natural, i, outWord);
            b.Label($"SKIP{i}");
        }

        b.L($"st.global.{wt} [{addrP}], {outWord};");
        b.Label("DONE");
        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return b.Finish(Header(gpu), entry, $".param .u64 p_p, .param .u64 p_g, .param .u64 p_m, .param .u64 p_v, .param .u32 p_n, .param .{pa} p_lr, .param .{pa} p_b1, .param .{pa} p_b2, .param .{pa} p_ob1, .param .{pa} p_ob2, .param .{pa} p_ic1, .param .{pa} p_ic2, .param .{pa} p_eps, .param .{pa} p_wd, .param .{pa} p_lrwd");
    }
}
