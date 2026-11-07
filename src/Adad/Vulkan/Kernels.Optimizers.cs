// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The optimizer steps on Vulkan (Optimizers.cs; CUDA's Ptx.Optimizers.cs, whose step Step repeats): a thread per word of
// the parameters (its values), the gradients' word alongside, the moments as lane values at the value's index, the
// parameters packed back into their word. Arguments: p, g, m, v, n, then the lane scalars lr, b1, b2, 1 − b1, 1 − b2,
// 1/c1, 1/c2, eps, wd, lr·wd.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public static SpvKernel Optimizer(Vulkan vk, Fmt f, Lane lane, string name)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"optimizer_{name}_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var pp = b.Arg(0);
        var pg = b.Arg(1);
        var pm = b.Arg(2);
        var pv = b.Arg(3);
        var n = b.Arg(4, K.U32);
        var s = new StepScalars(b.Arg(5, k), b.Arg(6, k), b.Arg(7, k), b.Arg(8, k), b.Arg(9, k), b.Arg(10, k), b.Arg(11, k), b.Arg(12, k), b.Arg(13, k), b.Arg(14, k));
        bool sgd = name == "sgd_momentum";
        var word = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(word < WordCount(b, f, n), () =>
        {
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
                    var mAddress = b.At(pm, idx, lane.Size);
                    var vAddress = b.At(pv, idx, lane.Size);
                    var m = b.Load(k, mAddress);
                    var v = sgd ? m : b.Load(k, vAddress);
                    var (pNew, mNew, vNew) = Step(b, name, p, g, m, v, s);
                    if (!sgd) b.Store(vAddress, vNew);
                    b.Store(mAddress, mNew);
                    output.V = Codec.Pack(b, f, pNew, lane, at, output.V);
                });
            }

            b.Store(addressP, output.V);
        });
        return b.Build();
    }

    private readonly record struct StepScalars(V Lr, V B1, V B2, V Ob1, V Ob2, V Ic1, V Ic2, V Eps, V Wd, V LrWd);

    // One value's step: the new parameter and moments.
    //   sgd_momentum: m = b1·m + g + wd·p;  p = p − lr·m
    //   adam / adamw: m = b1·m + (1 − b1)·g; v = b2·v + (1 − b2)·g²; p = p − lr·m̂ / (√v̂ + eps) (adamw: then p − lr·wd·p)
    private static (V P, V M, V V) Step(Spv b, string name, V p, V g, V m, V v, StepScalars k)
    {
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
        var pNew = p - k.Lr * mh / den;
        if (name == "adamw") pNew = pNew - k.LrWd * pNew;
        return (pNew, m1, v1);
    }
}
