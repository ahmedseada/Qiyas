// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The optimizer steps on Vulkan (Optimizers.cs; CUDA's Ptx.Optimizers.cs, whose step Step repeats): a thread per word of
// the parameters (its values), the gradients' word alongside, the moments as lane values at the value's index, the
// parameters packed back into their word. Arguments: p, g, m, v, n, then the lane scalars lr, b1, b2, 1 − b1, 1 − b2,
// 1/c1, 1/c2, eps, wd, lr·wd, then (scaled) where the gradients' scale is: every gradient times that lane value as it is
// read (a clipping factor made on the device: VulkanOps.Optimizers.cs). Table: the step of several tensors in one
// dispatch — an invocation's word falls in a row of a table on the device (TableRow: the tensor's first word, its count,
// where its p, g, m and v are; slot 0 the table, slot 4 its rows), and the invocation steps that tensor's word.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>The row of a table on the device an index falls in (CUDA's Ptx.TableRow). The table: `count` rows of
    /// `width` 64-bit entries, the first entry of each the first index of its row (ascending, row 0's 0). Returns the
    /// address of the last row whose first index is at most `index` (the last row for an index past them all: the
    /// caller's own bound drops it) and the index past that first, by halving.</summary>
    internal static (V Row, V Local) TableRow(Spv b, V tab, V count, V index, int width)
    {
        var wide = b.Convert(index, K.U64);
        var lo = b.Local(b.U(0));
        var hi = b.Local(count);
        b.While(() => (hi.V - lo.V) > 1, () =>
        {
            var mid = (lo.V + hi.V) >> 1;
            var le = b.Load(K.U64, b.At(tab, mid, width * 8)) <= wide;
            var l = b.Select(le, mid, lo.V);
            var h = b.Select(le, hi.V, mid);
            lo.V = l;
            hi.V = h;
        });
        var row = b.At(tab, lo.V, width * 8);
        return (row, b.Convert(wide - b.Load(K.U64, row), K.U32));
    }

    /// <summary>A step table's row: the tensor's first word, its values, where its p, g, m and v are; a sum table's: the
    /// array's first block, its values, its chunk, where it is.</summary>
    public const int StepRow = 6, SumRow = 4;

    public static SpvKernel Optimizer(Vulkan vk, Fmt f, Lane lane, string name, bool scaled = false, bool table = false)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"optimizer_{name}_{f.Name}_{lane.Name}{(scaled ? "_gs" : "")}{(table ? "_tab" : "")}", Threads, vk.Controls);
        var pp = b.Arg(0);
        var pg = b.Arg(1);
        var pm = b.Arg(2);
        var pv = b.Arg(3);
        var n = b.Arg(4, K.U32);
        var s = new StepScalars(b.Arg(5, k), b.Arg(6, k), b.Arg(7, k), b.Arg(8, k), b.Arg(9, k), b.Arg(10, k), b.Arg(11, k), b.Arg(12, k), b.Arg(13, k), b.Arg(14, k));
        bool plain = name == "sgd", sgd = plain || name == "sgd_momentum";
        var gs = scaled ? b.Arg(15) : default;
        var word = b.GlobalX;
        if (table)
        {
            var (row, local) = TableRow(b, pp, n, word, StepRow);
            n = b.Convert(b.Load(K.U64, b.At(row, b.U(0), 8, 8)), K.U32);
            pg = b.Load(K.U64, b.At(row, b.U(0), 8, 24));
            pm = b.Load(K.U64, b.At(row, b.U(0), 8, 32));
            pv = b.Load(K.U64, b.At(row, b.U(0), 8, 40));
            pp = b.Load(K.U64, b.At(row, b.U(0), 8, 16));
            word = local;
        }

        var wk = Codec.WordKind(f);
        b.If(word < WordCount(b, f, n), () =>
        {
            var scale = scaled ? b.Load(k, gs) : default;
            var addressP = b.At(pp, word, f.WordBits / 8);
            var wordP = b.Load(wk, addressP);
            var wordG = b.Load(wk, b.At(pg, word, f.WordBits / 8));
            var output = b.Local(b.CBits(wk, 0));
            for (int i = 0; i < f.Per; i++)
            {
                int at = i;
                var idx = word * (double)f.Per + at;
                b.If(idx < n, () =>
                {
                    var p = Codec.Unpack(b, f, wordP, at, lane);
                    var g = Codec.Unpack(b, f, wordG, at, lane);
                    if (scaled) g = g * scale;
                    var mAddress = b.At(pm, idx, lane.Size);
                    var vAddress = b.At(pv, idx, lane.Size);
                    var m = plain ? p : b.Load(k, mAddress);
                    var v = sgd ? m : b.Load(k, vAddress);
                    var (pNew, mNew, vNew) = Step(b, name, p, g, m, v, s);
                    if (!sgd) b.Store(vAddress, vNew);
                    if (!plain) b.Store(mAddress, mNew);
                    output.V = Codec.Pack(b, f, pNew, lane, at, output.V);
                });
            }

            b.Store(addressP, output.V);
        });
        return b.Build();
    }

    private readonly record struct StepScalars(V Lr, V B1, V B2, V Ob1, V Ob2, V Ic1, V Ic2, V Eps, V Wd, V LrWd);

    // One value's step: the new parameter and moments.
    //   sgd:          p = p − lr·(g + wd·p)
    //   sgd_momentum: m = b1·m + g + wd·p;  p = p − lr·m
    //   adam / adamw: m = b1·m + (1 − b1)·g; v = b2·v + (1 − b2)·g²; p = p − lr·m̂ / (√v̂ + eps) (adamw: p − lr·wd·p first,
    //                 the decay of the parameter the step starts from)
    private static (V P, V M, V V) Step(Spv b, string name, V p, V g, V m, V v, StepScalars k)
    {
        if (name == "sgd") return (p - k.Lr * b.Fma(k.Wd, p, g), m, v);
        if (name == "sgd_momentum")
        {
            var mm = b.Fma(k.B1, m, b.Fma(k.Wd, p, g));
            return (p - k.Lr * mm, mm, v);
        }

        var m1 = b.Fma(k.B1, m, k.Ob1 * g);
        var v1 = b.Fma(k.B2, v, k.Ob2 * (g * g));
        var mh = m1 * k.Ic1;
        var vh = v1 * k.Ic2;
        var den = b.Sqrt(vh) + k.Eps;
        var p0 = name == "adamw" ? p - k.LrWd * p : p;
        return (p0 - k.Lr * mh / den, m1, v1);
    }
}
