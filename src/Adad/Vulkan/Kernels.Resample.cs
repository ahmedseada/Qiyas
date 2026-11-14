// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on Vulkan (VulkanOps.Resample.cs), CUDA's kernels' rules (Cuda/Ptx.Resample.cs):
// the horizontal pass an invocation a value (a whole lane value stored), the outputs an invocation a stored word (a
// block format's by its encoder, each block from its values); or both passes in one kernel by tiles (ResampleTiled), the
// horizontal pass's values for a tile's rows in workgroup memory.
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
        b.If(gid < WordCount(b, f, count), () =>
        {
            var word = b.CBits(Codec.WordKind(f), 0);
            var last = count - 1;
            for (int i = 0; i < f.Per; i++)
            {
                // A word's values past the count take the last value's addresses (dropped).
                var at = gid * (double)f.Per + i;
                var value = ResampleValue(b, src, lane, k, vertical, norm, ps, pb, pw, pv, oh, ow, h, across, taps, rb, rw, channels, b.Min(at, last));
                word = Codec.Pack(b, f, b.Select(at < count, value, b.C(k, 0)), lane, i, word);   // an integer's: rounded, held to its range
            }

            b.Store(b.At(py, gid, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>ResampleTiled's tile: output columns and rows.</summary>
    public const int ResampleTileX = 64, ResampleTileY = 32;

    /// <summary>The middle rows ResampleTiled holds for a tile (a multiple of 8), from the plan's sizes: an output row's first
    /// input row at most ⌈h/oh⌉ + 1 past the row before's (the centres h/oh apart, each first rounded), and the last row's
    /// taps. A tap past them is made where it is read (the kernel's fallback), so the estimate bounds only the memory.</summary>
    public static int ResampleTileRows(ResampleArgs a)
    {
        int step = (a.Height + a.OutHeight - 1) / Math.Max(1, a.OutHeight) + 1;
        long rows = (long)(ResampleTileY - 1) * step + a.YTaps + 1;
        return (int)Math.Min((rows + 7) / 8 * 8, int.MaxValue / 2);
    }

    /// <summary>The workgroup memory ResampleTiled takes for `rows` middle rows.</summary>
    public static long ResampleTiledBytes(Fmt f, Lane lane, int rows) =>
        ((long)rows * ResampleTileX + (f.Per > 1 ? ResampleTileX * ResampleTileY : 0)) * lane.Size;

    /// <summary>Both passes in one kernel (where the plan has both): a workgroup a tile of ResampleTileY × ResampleTileX
    /// outputs of a plane. The horizontal pass's values of the input rows from the tile's first row's first tap to its last
    /// row's last (at most `rows` of them), the tile's columns, made into workgroup memory by ResampleTaps as ResampleRows
    /// makes them; then each output by its vertical taps over them in order (a tap's row past those held made where it is
    /// read, the same way), the channel's scale and shift, and the store: a value a word as made; else the tile's values
    /// into workgroup memory and a word an invocation (the output's rows whole words). The same operations on the same
    /// values as ResampleRows and ResampleOut with the middle plane between them: the same values. Slots: x, bounds,
    /// weights, values, y, oh, ow, h, w, taps (horizontal), taps (vertical), the rows' bounds and weights offsets,
    /// channels, tiles of rows a plane. Workgroups (⌈ow / ResampleTileX⌉, planes · tiles).</summary>
    public static SpvKernel ResampleTiled(Vulkan vk, Fmt f, Fmt inFmt, Lane lane, bool norm, int rows)
    {
        const int TX = ResampleTileX, TY = ResampleTileY;
        var k = Kinds.Of(lane);
        var b = new Spv($"resample_tiled_{f.Name}_{inFmt.Name}_{lane.Name}_{rows}{(norm ? "_n" : "")}", Threads, vk.Controls);
        V px = b.Arg(0), pb = b.Arg(1), pw = b.Arg(2), pv = b.Arg(3), py = b.Arg(4);
        V oh = b.Arg(5, K.U32), ow = b.Arg(6, K.U32), h = b.Arg(7, K.U32), w = b.Arg(8, K.U32), xTaps = b.Arg(9, K.U32), yTaps = b.Arg(10, K.U32);
        V rb = b.Arg(11, K.U32), rw = b.Arg(12, K.U32), channels = b.Arg(13, K.U32), tilesY = b.Arg(14, K.U32);
        var tid = b.LocalIdX;
        var plane = b.GroupY / tilesY;
        var oy0 = (b.GroupY - plane * tilesY) * (double)TY;
        var ox0 = b.GroupX * (double)TX;
        var mid = b.Shared(k, rows * TX);

        // A middle value: row r's horizontal pass at output column ox (ResampleRows's).
        V Middle(V r, V ox)
        {
            var (first, cnt) = ResampleBounds(b, pb, ox * 2);
            return ResampleTaps(b, inFmt, lane, k, px, pw, (plane * h + r) * w + first, b.U(1), ox * xTaps, cnt);
        }

        // The rows held: [r0, r0 + held), r0 the tile's first row's first tap, to its last row's last (unsigned: a plan whose
        // rows do not rise gives at most `rows`, the rest made where read).
        var (r0, _) = ResampleBounds(b, pb, oy0 * 2 + rb);
        var (rl, nl) = ResampleBounds(b, pb, (b.Min(oy0 + (double)TY, oh) - 1) * 2 + rb);
        var held = b.Min(rl + nl - r0, b.U(rows));
        b.For(tid, held * (double)TX, b.U(Threads), i =>
        {
            var rr = i / (double)TX;
            var ox = ox0 + (i - rr * (double)TX);
            b.If(ox < ow, () => mid[i] = Middle(r0 + rr, ox));
        });
        b.Barrier();
        var outs = f.Per > 1 ? b.Shared(k, TY * TX) : default;
        var channel = plane % channels;
        b.For(tid, b.U(TY * TX), b.U(Threads), i =>
        {
            var yl = i / (double)TX;
            var c = i - yl * (double)TX;
            var oy = oy0 + yl;
            var ox = ox0 + c;
            b.If((oy < oh) & (ox < ow), () =>
            {
                var (first, cnt) = ResampleBounds(b, pb, oy * 2 + rb);
                var acc = b.Local(b.C(k, 0));
                var wi = oy * yTaps + rw;
                b.For(b.U(0), cnt, t =>
                {
                    var rel = first + t - r0;
                    var m = b.Local(b.C(k, 0));
                    b.If(rel < held, () => m.V = mid[rel * (double)TX + c], () => m.V = Middle(first + t, ox));
                    acc.V = b.Fma(m.V, b.Load(k, b.At(pw, wi + t, lane.Size)), acc.V);
                });
                var v = norm ? b.Fma(acc.V, b.Load(k, b.At(pv, channel * 2, lane.Size)), b.Load(k, b.At(pv, channel * 2 + 1, lane.Size))) : acc.V;
                if (f.Per > 1) outs[i] = v;
                else b.Store(b.At(py, (plane * oh + oy) * ow + ox, f.WordBits / 8), Codec.Pack(b, f, v, lane, 0, b.CBits(Codec.WordKind(f), 0)));
            });
        });
        if (f.Per == 1) return b.Build();

        // A packed type's words: the tile's values from workgroup memory, a word an invocation (the rows whole words).
        b.Barrier();
        int words = TX / f.Per;
        var wk = Codec.WordKind(f);
        b.For(tid, b.U(TY * words), b.U(Threads), j =>
        {
            var yl = j / (double)words;
            var wd = j - yl * (double)words;
            var oy = oy0 + yl;
            var c0 = wd * (double)f.Per;
            b.If((oy < oh) & (ox0 + c0 < ow), () =>
            {
                var packed = b.CBits(wk, 0);
                for (int q = 0; q < f.Per; q++) packed = Codec.Pack(b, f, outs[yl * (double)TX + c0 + q], lane, q, packed);
                b.Store(b.At(py, ((plane * oh + oy) * ow + ox0 + c0) / (double)f.Per, f.WordBits / 8), packed);
            });
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
