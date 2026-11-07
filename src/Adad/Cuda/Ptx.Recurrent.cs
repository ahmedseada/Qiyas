// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's recurrent cells (Recurrent.cs): a thread per output word (its Per hidden units of one row), the gate
// values read by index, the lane's sigmoid and tanh (Functions.cs); the cell backward likewise.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>The formats a mixed cell reads its inputs as: the gates (g, g2), the state, dh' and dc'. O: the format its
    /// state outputs (h', c'; dc; dh) are stored as (whole bytes, not blocks; null: lane storage, as the gates' gradients
    /// always are). Mirror: h' and c' also as lane values at p_o4, p_o5 (a sequence's state for the next step).</summary>
    public sealed record CellFormats(Fmt G, Fmt G2, Fmt S, Fmt D1, Fmt D2, Fmt? O = null, bool Mirror = false)
    {
        public string Name => $"{G.Name}_{G2.Name}_{S.Name}_{D1.Name}_{D2.Name}{(O is null ? "" : "_o" + O.Name)}{(Mirror ? "_mirror" : "")}";
    }

    /// <summary>LSTM: (h', c') from the gates (B × 4H) and c; GRU: h' from gx, gh (B × 3H) and h. Backward variants give the
    /// gates' gradients (and dc or dh) from dh' (and dc'). formats: each input read as its own format where it is stored
    /// (f the outputs': lane storage), as a composition's cell of the type's tensors and its f32 intermediates.</summary>
    public static string Cell(Gpu gpu, Fmt f, Lane lane, string kind, bool backward, out string entry, CellFormats? formats = null)
    {
        entry = $"cell_{kind}{(backward ? "_bwd" : "")}_{f.Name}_{lane.Name}{(formats is null ? "" : "_m" + formats.Name)}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), word = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {word}, {bid}, 256, {tid};");
        // p_g, p_g2: the gates (GRU: gx, gh); p_s: the state (c or h); p_d1, p_d2: dh', dc'; outputs p_o1, p_o2 (h', c' | dg, dc or dgx, dgh) and p_o3 (GRU backward: dh).
        string g = Param64(b, "p_g"), g2 = Param64(b, "p_g2"), st = Param64(b, "p_s"), d1 = Param64(b, "p_d1"), d2 = Param64(b, "p_d2");
        string o1 = Param64(b, "p_o1"), o2 = Param64(b, "p_o2"), o3 = Param64(b, "p_o3");
        string o4 = formats is null ? "" : Param64(b, "p_o4"), o5 = formats is null ? "" : Param64(b, "p_o5");
        string B = Param32(b, "p_b"), H = Param32(b, "p_h");
        if (formats is not null && f.Per != 1) throw new ArgumentException("a mixed cell: a value a thread (f the lane's storage)");
        bool lstm = kind == "lstm";
        int G = lstm ? 4 : 3;
        string count = b.R(), words = b.R(), pw = b.P();
        b.L($"mul.lo.u32 {count}, {B}, {H};");
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {count};");
        else { b.L($"add.u32 {words}, {count}, {f.Per - 1};"); b.L($"div.u32 {words}, {words}, {f.Per};"); }
        b.L($"setp.ge.u32 {pw}, {word}, {words};");
        b.L($"@{pw} bra DONE;");
        var natural = f.Natural;
        Fmt FormatOf(string ptr) => formats is null ? f : ptr == g ? formats.G : ptr == g2 ? formats.G2 : ptr == st ? formats.S : ptr == d1 ? formats.D1 : formats.D2;
        string Load(string ptr, string idx) => Codec.UnpackAt(b, FormatOf(ptr), ptr, idx, lane);
        string Sig(string x) => Fn.Sigmoid(b, lane, x);
        string Tanh(string x) => Fn.Tanh(b, lane, x);
        string Mul(string x, string y) { string d = lane.NewReg(b); b.L(lane.Mul(d, x, y)); return d; }
        string Add(string x, string y) { string d = lane.NewReg(b); b.L(lane.Add(d, x, y)); return d; }
        string Sub(string x, string y) { string d = lane.NewReg(b); b.L($"sub.rn.{lane.Op} {d}, {x}, {y};"); return d; }
        string Fma(string x, string y, string z) { string d = lane.NewReg(b); b.L(lane.Fma(d, x, y, z)); return d; }
        string One() => lane.Literal(1);
        string OneMinus(string x) => Sub(One(), x);
        string last = b.R();
        b.L($"sub.u32 {last}, {count}, 1;");
        // Output words: up to three (o1, o2, o3) assembled per value and stored once.
        string[] outWords = new string[3], outVals = new string[3];
        int outs = lstm ? 2 : backward ? 3 : 1;
        for (int k = 0; k < outs; k++) { outWords[k] = f.WordBits == 64 ? b.RD() : b.R(); if (f.WordBits == 32) b.L($"mov.b32 {outWords[k]}, 0;"); }
        // GRU backward writes dgx, dgh (B × 3H) by a separate word mapping: a thread per hidden unit word writes its three gate values' words via value-wise stores.
        for (int i = 0; i < f.Per; i++)
        {
            string v = b.R(), pv = b.P(), row = b.R(), j = b.R(), t = b.R(), gi = b.R();
            b.L($"mad.lo.u32 {v}, {word}, {f.Per}, {i};");
            b.L($"setp.lt.u32 {pv}, {v}, {count};");
            b.L($"min.u32 {v}, {v}, {last};");
            b.L($"div.u32 {row}, {v}, {H};");
            b.L($"mul.lo.u32 {t}, {row}, {H};");
            b.L($"sub.u32 {j}, {v}, {t};");
            b.L($"mul.lo.u32 {gi}, {row}, {G * 1};");
            b.L($"mul.lo.u32 {gi}, {gi}, {H};");
            b.L($"add.u32 {gi}, {gi}, {j};");                                   // the row's gate base + j: gate k at gi + k·H
            string[] gIdx = new string[G];
            for (int k = 0; k < G; k++) { gIdx[k] = b.R(); b.L($"mad.lo.u32 {gIdx[k]}, {H}, {k}, {gi};"); }
            if (lstm && !backward)
            {
                string ig = Sig(Load(g, gIdx[0])), fg = Sig(Load(g, gIdx[1])), gg = Tanh(Load(g, gIdx[2])), og = Sig(Load(g, gIdx[3]));
                string cn = Fma(fg, Load(st, v), Mul(ig, gg));
                string hn = Mul(og, Tanh(cn));
                PackValue(b, f, lane, natural, hn, pv, i, outWords[0]);
                PackValue(b, f, lane, natural, cn, pv, i, outWords[1]);
                outVals[0] = hn; outVals[1] = cn;
            }
            else if (lstm)
            {
                // dg (B × 4H) value-wise, dc into the word.
                string ig = Sig(Load(g, gIdx[0])), fg = Sig(Load(g, gIdx[1])), gg = Tanh(Load(g, gIdx[2])), og = Sig(Load(g, gIdx[3]));
                string c = Load(st, v), dh = Load(d1, v), dcNext = Load(d2, v);
                string cn = Fma(fg, c, Mul(ig, gg)), tc = Tanh(cn);
                string dcn = Fma(Mul(dh, og), OneMinus(Mul(tc, tc)), dcNext);
                string dgo = Mul(Mul(dh, tc), Mul(og, OneMinus(og)));
                string dgi = Mul(Mul(dcn, gg), Mul(ig, OneMinus(ig)));
                string dgf = Mul(Mul(dcn, c), Mul(fg, OneMinus(fg)));
                string dgg = Mul(Mul(dcn, ig), OneMinus(Mul(gg, gg)));
                StoreValue(b, f, lane, natural, dgi, pv, o1, gIdx[0]);
                StoreValue(b, f, lane, natural, dgf, pv, o1, gIdx[1]);
                StoreValue(b, f, lane, natural, dgg, pv, o1, gIdx[2]);
                StoreValue(b, f, lane, natural, dgo, pv, o1, gIdx[3]);
                outVals[1] = Mul(dcn, fg);
                PackValue(b, f, lane, natural, outVals[1], pv, i, outWords[1]);
            }
            else if (!backward)
            {
                string r = Sig(Add(Load(g, gIdx[0]), Load(g2, gIdx[0]))), z = Sig(Add(Load(g, gIdx[1]), Load(g2, gIdx[1])));
                string n = Tanh(Fma(r, Load(g2, gIdx[2]), Load(g, gIdx[2])));
                string hn = Fma(z, Load(st, v), Mul(OneMinus(z), n));
                PackValue(b, f, lane, natural, hn, pv, i, outWords[0]);
                outVals[0] = hn;
            }
            else
            {
                string ghn = Load(g2, gIdx[2]);
                string r = Sig(Add(Load(g, gIdx[0]), Load(g2, gIdx[0]))), z = Sig(Add(Load(g, gIdx[1]), Load(g2, gIdx[1])));
                string n = Tanh(Fma(r, ghn, Load(g, gIdx[2])));
                string h = Load(st, v), d = Load(d1, v);
                string dn = Mul(d, OneMinus(z)), dz = Mul(d, Sub(h, n));
                string dpre = Mul(dn, OneMinus(Mul(n, n)));
                string dr = Mul(dpre, ghn);
                string dgr = Mul(dr, Mul(r, OneMinus(r))), dgz = Mul(dz, Mul(z, OneMinus(z)));
                StoreValue(b, f, lane, natural, dgr, pv, o1, gIdx[0]);
                StoreValue(b, f, lane, natural, dgz, pv, o1, gIdx[1]);
                StoreValue(b, f, lane, natural, dpre, pv, o1, gIdx[2]);
                StoreValue(b, f, lane, natural, dgr, pv, o2, gIdx[0]);
                StoreValue(b, f, lane, natural, dgz, pv, o2, gIdx[1]);
                StoreValue(b, f, lane, natural, Mul(dpre, r), pv, o2, gIdx[2]);
                outVals[2] = Mul(d, z);
                PackValue(b, f, lane, natural, outVals[2], pv, i, outWords[2]);
            }
        }

        // The packed words: LSTM forward h', c' (o1, o2); LSTM backward dc (o2); GRU forward h' (o1); GRU backward dh (o3).
        string addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {word}, {f.WordBits / 8};");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        void StoreWord(string ptr, string w) { string a = b.RD(); b.L($"add.u64 {a}, {ptr}, {addr};"); b.L($"st.global.{wt} [{a}], {w};"); }
        // A state output: its word in lane storage, or (formats.O) its value stored as that format, mirrored when asked.
        void StoreOut(string ptr, int k, string mirror)
        {
            if (formats?.O is not { } fo) { StoreWord(ptr, outWords[k]); return; }
            StoreAs(b, fo, lane, outVals[k], ptr, word);
            if (formats.Mirror) StoreWord(mirror, outWords[k]);
        }

        if (lstm && !backward) { StoreOut(o1, 0, o4); StoreOut(o2, 1, o5); }
        else if (lstm) StoreOut(o2, 1, o5);
        else if (!backward) StoreOut(o1, 0, o4);
        else StoreOut(o3, 2, o5);
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_g, .param .u64 p_g2, .param .u64 p_s, .param .u64 p_d1, .param .u64 p_d2, .param .u64 p_o1, .param .u64 p_o2, .param .u64 p_o3"
            + (formats is null ? "" : ", .param .u64 p_o4, .param .u64 p_o5") + ", .param .u32 p_b, .param .u32 p_h");
    }

    // A lane value into position i of an output word (zero when the value is past the count).
    private static void PackValue(PtxBuilder b, Fmt f, Lane lane, Lane natural, string v, string pv, int i, string word)
    {
        string stored = lane == natural ? v : Codec.Convert(b, lane, v, natural, f);
        string kept = natural.NewReg(b);
        b.L($"selp.{natural.Reg} {kept}, {stored}, {natural.Zero}, {pv};");
        Codec.Pack(b, f, kept, natural, i, word);
    }

    // A lane value stored at value index idx of an array of the type, value-wise (16- and 8-bit formats as such; a 32/64-bit word as a whole).
    private static void StoreValue(PtxBuilder b, Fmt f, Lane lane, Lane natural, string v, string pv, string ptr, string idx)
    {
        string stored = lane == natural ? v : Codec.Convert(b, lane, v, natural, f);
        string addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {idx}, {f.Bits / 8};");
        b.L($"add.u64 {addr}, {ptr}, {addr};");
        if (f.Bits >= 32)
        {
            string code = f.WordBits == 64 ? b.RD() : b.R();
            if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
            Codec.Pack(b, f, stored, natural, 0, code);
            b.L($"@{pv} st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{addr}], {code};");
        }
        else
        {
            string code = b.R(), h = b.H();
            b.L($"mov.b32 {code}, 0;");
            Codec.Pack(b, f, stored, natural, 0, code);
            b.L($"cvt.u16.u32 {h}, {code};");
            b.L($"@{pv} st.global.b{f.Bits} [{addr}], {h};");
        }
    }
}
