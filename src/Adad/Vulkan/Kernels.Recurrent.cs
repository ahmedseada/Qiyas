// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The recurrent cells on Vulkan (Recurrent.cs; CUDA's Ptx.Recurrent.cs): an invocation per hidden unit of a row, each
// input read where it is stored as its own format (the type's tensors, or the f32/f64 lane storage of a composition's
// intermediates), the lane's sigmoid and tanh, the outputs lane values (no sub-word stores: the host stores them as the
// type once). The operations in CUDA's order. And the mixed add of the compositions: o = x + y (y a row vector of cols
// when broadcast), each read as its own format, o lane values.
using Adad.Abstraction;

namespace Adad.Vulkan;

/// <summary>The formats a cell reads its inputs as: the gates (g, g2), the state, dh' and dc'.</summary>
internal sealed record CellFormats(Fmt G, Fmt G2, Fmt S, Fmt D1, Fmt D2)
{
    public string Name => $"{G.Name}_{G2.Name}_{S.Name}_{D1.Name}_{D2.Name}";
}

internal static partial class VkKernels
{
    /// <summary>LSTM: (h', c') from the gates (B × 4H) and c; GRU: h' from gx, gh (B × 3H) and h. The backward: LSTM dg
    /// (B × 4H) and dc from dh', dc'; GRU dgx, dgh (B × 3H) and dh from dh'. Arguments: g, g2, s, d1, d2, o1, o2, o3 (lane
    /// values: h', c' | dg, dc | h' | dgx, dgh, dh), B, H; an invocation per value of B × H.</summary>
    public static SpvKernel Cell(Vulkan vk, Lane lane, string kind, bool backward, CellFormats fm)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"cell_{kind}{(backward ? "_bwd" : "")}_{lane.Name}_{fm.Name}", Threads, vk.Controls);
        var g = b.Arg(0);
        var g2 = b.Arg(1);
        var st = b.Arg(2);
        var d1 = b.Arg(3);
        var d2 = b.Arg(4);
        var o1 = b.Arg(5);
        var o2 = b.Arg(6);
        var o3 = b.Arg(7);
        var B = b.Arg(8, K.U32);
        var H = b.Arg(9, K.U32);
        bool lstm = kind == "lstm";
        int G = lstm ? 4 : 3;
        var v = b.GlobalX;
        var one = b.C(k, 1);
        V Load(Fmt f, V ptr, V idx) => Codec.UnpackAt(b, f, ptr, idx, lane);
        V Sig(V x) => Fn.Sigmoid(b, x);
        V Tanh(V x) => Fn.Tanh(b, x);
        V OneMinus(V x) => one - x;
        void Store(V ptr, V idx, V value) => b.Store(b.At(ptr, idx, lane.Size), value);
        b.If(v < B * H, () =>
        {
            var row = v / H;
            var j = v - row * H;
            var gi = row * b.U(G) * H + j;                                              // the row's gate base + j: gate q at gi + q·H
            var gIdx = new V[G];
            for (int q = 0; q < G; q++) gIdx[q] = gi + H * b.U(q);
            if (lstm && !backward)
            {
                V ig = Sig(Load(fm.G, g, gIdx[0])), fg = Sig(Load(fm.G, g, gIdx[1])), gg = Tanh(Load(fm.G, g, gIdx[2])), og = Sig(Load(fm.G, g, gIdx[3]));
                var cn = b.Fma(fg, Load(fm.S, st, v), ig * gg);
                Store(o1, v, og * Tanh(cn));
                Store(o2, v, cn);
            }
            else if (lstm)
            {
                V ig = Sig(Load(fm.G, g, gIdx[0])), fg = Sig(Load(fm.G, g, gIdx[1])), gg = Tanh(Load(fm.G, g, gIdx[2])), og = Sig(Load(fm.G, g, gIdx[3]));
                V c = Load(fm.S, st, v), dh = Load(fm.D1, d1, v), dcNext = Load(fm.D2, d2, v);
                var cn = b.Fma(fg, c, ig * gg);
                var tc = Tanh(cn);
                var dcn = b.Fma(dh * og, OneMinus(tc * tc), dcNext);
                Store(o1, gIdx[0], (dcn * gg) * (ig * OneMinus(ig)));
                Store(o1, gIdx[1], (dcn * c) * (fg * OneMinus(fg)));
                Store(o1, gIdx[2], (dcn * ig) * OneMinus(gg * gg));
                Store(o1, gIdx[3], (dh * tc) * (og * OneMinus(og)));
                Store(o2, v, dcn * fg);
            }
            else if (!backward)
            {
                var r = Sig(Load(fm.G, g, gIdx[0]) + Load(fm.G2, g2, gIdx[0]));
                var z = Sig(Load(fm.G, g, gIdx[1]) + Load(fm.G2, g2, gIdx[1]));
                var n = Tanh(b.Fma(r, Load(fm.G2, g2, gIdx[2]), Load(fm.G, g, gIdx[2])));
                Store(o1, v, b.Fma(z, Load(fm.S, st, v), OneMinus(z) * n));
            }
            else
            {
                var ghn = Load(fm.G2, g2, gIdx[2]);
                var r = Sig(Load(fm.G, g, gIdx[0]) + Load(fm.G2, g2, gIdx[0]));
                var z = Sig(Load(fm.G, g, gIdx[1]) + Load(fm.G2, g2, gIdx[1]));
                var n = Tanh(b.Fma(r, ghn, Load(fm.G, g, gIdx[2])));
                V h = Load(fm.S, st, v), d = Load(fm.D1, d1, v);
                var dn = d * OneMinus(z);
                var dz = d * (h - n);
                var dpre = dn * OneMinus(n * n);
                var dr = dpre * ghn;
                var dgr = dr * (r * OneMinus(r));
                var dgz = dz * (z * OneMinus(z));
                Store(o1, gIdx[0], dgr);
                Store(o1, gIdx[1], dgz);
                Store(o1, gIdx[2], dpre);
                Store(o2, gIdx[0], dgr);
                Store(o2, gIdx[1], dgz);
                Store(o2, gIdx[2], dpre * r);
                Store(o3, v, d * z);
            }
        });
        return b.Build();
    }

    /// <summary>o[i] = x[xOff + i] + y[yOff + (row ? i mod cols : i)], x and y read as fx and fy, o lane values.
    /// Arguments: x, y, o, n, cols, xOff, yOff; an invocation per value.</summary>
    public static SpvKernel AddMixed(Vulkan vk, Fmt fx, Fmt fy, Lane lane, bool row)
    {
        var b = new Spv($"add_mixed_{fx.Name}_{fy.Name}_{lane.Name}{(row ? "_row" : "")}", Threads, vk.Controls);
        var x = b.Arg(0);
        var y = b.Arg(1);
        var o = b.Arg(2);
        var n = b.Arg(3, K.U32);
        var cols = b.Arg(4, K.U32);
        var xOff = b.Arg(5, K.U32);
        var yOff = b.Arg(6, K.U32);
        var i = b.GlobalX;
        b.If(i < n, () =>
        {
            var yi = row ? i % cols : i;
            var sum = Codec.UnpackAt(b, fx, x, xOff + i, lane) + Codec.UnpackAt(b, fy, y, yOff + yi, lane);
            b.Store(b.At(o, i, lane.Size), sum);
        });
        return b.Build();
    }
}
