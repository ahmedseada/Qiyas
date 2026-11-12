// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on Vulkan (VulkanOps.Resample.cs), CUDA's kernels' rules (Cuda/Ptx.Resample.cs):
// the horizontal pass an invocation a value (a whole lane value stored), the outputs an invocation a stored word (a
// block format's by its encoder, each block from its values).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>The horizontal pass into mid. Slots: x, bounds, weights, mid, n (values), width, across (the output width), taps.</summary>
    public static SpvKernel ResampleRows(Vulkan vk, Fmt inFmt, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"resample_rows_{inFmt.Name}_{lane.Name}", Threads, vk.Controls);
        V px = b.Arg(0), pb = b.Arg(1), pw = b.Arg(2), pm = b.Arg(3);
        V n = b.Arg(4, K.U32), width = b.Arg(5, K.U32), across = b.Arg(6, K.U32), taps = b.Arg(7, K.U32);
        var i = b.GlobalX;
        b.If(i < n, () =>
        {
            var ox = i % across;
            var row = i / across;
            var (first, cnt) = ResampleBounds(b, pb, ox * 2);
            var v = ResampleTaps(b, inFmt, lane, k, px, pw, row * width + first, b.U(1), ox * taps, cnt);
            b.Store(b.At(pm, i, lane.Size), v);
        });
        return b.Build();
    }

    /// <summary>The output of f a word an invocation. Slots: src, bounds, weights, values, y, count, oh, ow, h, across, taps,
    /// the rows' bounds and weights offsets, channels.</summary>
    public static SpvKernel ResampleOut(Vulkan vk, Fmt f, Fmt src, Lane lane, bool vertical, bool norm)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"resample_out_{f.Name}_{src.Name}_{lane.Name}{(vertical ? "_v" : "")}{(norm ? "_n" : "")}", Threads, vk.Controls);
        V ps = b.Arg(0), pb = b.Arg(1), pw = b.Arg(2), pv = b.Arg(3), py = b.Arg(4);
        V count = b.Arg(5, K.U32), oh = b.Arg(6, K.U32), ow = b.Arg(7, K.U32), h = b.Arg(8, K.U32), across = b.Arg(9, K.U32), taps = b.Arg(10, K.U32);
        V rb = b.Arg(11, K.U32), rw = b.Arg(12, K.U32), channels = b.Arg(13, K.U32);
        var gid = b.GlobalX;
        var natural = f.Natural;
        var nk = Kinds.Of(natural);
        b.If(gid < WordCount(b, f, count), () =>
        {
            var word = b.CBits(Codec.WordKind(f), 0);
            var last = count - 1;
            for (int i = 0; i < f.Per; i++)
            {
                // A word's values past the count take the last value's addresses (dropped).
                var at = gid * (double)f.Per + i;
                var value = ResampleValue(b, src, lane, k, vertical, norm, ps, pb, pw, pv, oh, ow, h, across, taps, rb, rw, channels, b.Min(at, last));
                var stored = lane.Name == natural.Name ? value : Codec.Convert(b, lane, value, natural);
                word = Codec.Pack(b, f, b.Select(at < count, stored, b.C(nk, 0)), natural, i, word);
            }

            b.Store(b.At(py, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>The output as blocks of f (f32). Slots: src, bounds, weights, values, oh, ow, h, across, taps, the rows' bounds and
    /// weights offsets, channels.</summary>
    public static SpvKernel BlockResampleOut(Vulkan vk, Fmt f, Fmt src, bool vertical, bool norm) =>
        BlockEncoder(vk, f, $"resample_out_{src.Name}{(vertical ? "_v" : "")}{(norm ? "_n" : "")}", b =>
        {
            V ps = b.Arg(0), pb = b.Arg(1), pw = b.Arg(2), pv = b.Arg(3);
            V oh = b.Arg(4, K.U32), ow = b.Arg(5, K.U32), h = b.Arg(6, K.U32), across = b.Arg(7, K.U32), taps = b.Arg(8, K.U32);
            V rb = b.Arg(9, K.U32), rw = b.Arg(10, K.U32), channels = b.Arg(11, K.U32);
            return (at, _) => ResampleValue(b, src, Lane.F32, K.F32, vertical, norm, ps, pb, pw, pv, oh, ow, h, across, taps, rb, rw, channels, at);
        });

    // Output value `at`: its vertical pass over src (or its value there), then the channel's scale and shift, as a lane value.
    private static V ResampleValue(Spv b, Fmt src, Lane lane, K k, bool vertical, bool norm, V ps, V pb, V pw, V pv, V oh, V ow, V h, V across, V taps, V rb, V rw,
        V channels, V at)
    {
        var ox = at % ow;
        var t = at / ow;
        var oy = t % oh;
        var p = t / oh;
        var c = p % channels;
        V v;
        if (vertical)
        {
            var (first, cnt) = ResampleBounds(b, pb, oy * 2 + rb);
            v = ResampleTaps(b, src, lane, k, ps, pw, (p * h + first) * across + ox, across, oy * taps + rw, cnt);
        }
        else v = Codec.UnpackAt(b, src, ps, (p * h + oy) * across + ox, lane);

        return norm ? b.Fma(v, b.Load(k, b.At(pv, c * 2, lane.Size)), b.Load(k, b.At(pv, c * 2 + 1, lane.Size))) : v;
    }

    // A pass's first position and tap count: bounds[bi], bounds[bi + 1] (int32).
    private static (V First, V Count) ResampleBounds(Spv b, V pb, V bi) =>
        (b.Bitcast(b.Load(K.S32, b.At(pb, bi, 4)), K.U32), b.Bitcast(b.Load(K.S32, b.At(pb, bi + 1, 4)), K.U32));

    // Σ over cnt taps of fmt's values from src (step apart) by the weights from wi: the lane's fma in order from 0.
    private static V ResampleTaps(Spv b, Fmt fmt, Lane lane, K k, V px, V pw, V src, V step, V wi, V cnt)
    {
        var acc = b.Local(b.C(k, 0));
        b.For(b.U(0), cnt, t => acc.V = b.Fma(Codec.UnpackAt(b, fmt, px, t * step + src, lane), b.Load(k, b.At(pw, wi + t, lane.Size)), acc.V));
        return acc.V;
    }
}
