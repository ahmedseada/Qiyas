// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on Vulkan (Abstraction/Cache.cs, ICache; VulkanOps.Cache.cs), CUDA's (Cuda/Ptx.Cache.cs): the
// position (an s32 at the device) read by each kernel where it is. The write's kernels: each source row's scale (a
// workgroup a row), stored as the type at its row — float32 and float64 by the scale's kernel itself; a narrower float by
// an invocation a 32-bit word of the scales, the word's other values kept as they are; a block format's by its encoder
// over the scales' blocks and a compaction that writes the touched blocks only — then the codes, an invocation a stored
// word of the cache (a row is whole 32-bit words). Attention over the caches is the decode kernel's (Kernels.Attention.cs,
// DecodeCache). No sub-word stores: every stored word has one writer.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    // The rows of each head a write of `steps` rows at the device's position fills, [W0, W1) = [max(p, 0), min(p + steps,
    // cap)) (u32; W1 = W0 when there are none), the position's bits, and whether there are any.
    private static (V P, V W0, V W1, V Any) WrittenRows(Spv b, V pos, V steps, V cap)
    {
        var p = b.Load(K.S32, pos);
        var w0 = b.Max(p, b.S(0));
        var w1 = b.Min(p + b.Convert(steps, K.S32), b.Convert(cap, K.S32));
        return (b.Convert(p, K.U32), b.Convert(w0, K.U32), b.Convert(b.Max(w1, w0), K.U32), w1 > w0);
    }

    // Whether a written row's index (x mod cap in [w0, w1), every head) lies in [a, end): the first one at or past a is.
    private static V Touched(Spv b, V a, V end, V cap, (V P, V W0, V W1, V Any) w)
    {
        var r = a % cap;
        var x0 = b.Select(r < w.W0, a + (w.W0 - r), b.Select(r < w.W1, a, a + (cap - r) + w.W0));
        return w.Any & (x0 < end);
    }

    // Value `at` of the scales: whether the write fills it, and its new scale's index (head·steps + step; 0 when not).
    private static (V Written, V Index) WrittenAt(Spv b, V at, V cap, V steps, (V P, V W0, V W1, V Any) w)
    {
        var r = at % cap;
        var written = w.Any & (r >= w.W0) & (r < w.W1);
        return (written, b.Select(written, at / cap * steps + (r - w.P), b.U(0)));
    }

    /// <summary>Each source row's scale max |x| / max in the lane, a workgroup a row (row = head·steps + step). direct: a lane
    /// value at head·cap + p + step of ps (a row past the capacity not written; the type's storage is the lane's); else at
    /// ps[row]. Slots: x (f), ps, pos, steps, cap, d, max (lane).</summary>
    public static SpvKernel CacheScales(Vulkan vk, Fmt f, Lane lane, bool direct)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"cache_scales_{f.Name}_{lane.Name}{(direct ? "" : "_lanes")}", ReduceThreads, vk.Controls);
        V px = b.Arg(0), ps = b.Arg(1), pos = b.Arg(2);
        V steps = b.Arg(3, K.U32), cap = b.Arg(4, K.U32), d = b.Arg(5, K.U32), max = b.Arg(6, k);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var amax = b.Local(b.C(k, 0));
        b.For(tid, d, b.U(ReduceThreads), e => amax.V = b.Max(amax.V, b.Abs(Codec.UnpackAt(b, f, px, row * d + e, lane))));
        var top = GroupReduce(b, b.Shared(k, ReduceThreads), tid, amax.V, (x, y) => b.Max(x, y), ReduceThreads);
        var scale = top / max;
        b.If(tid == 0, () =>
        {
            if (!direct) { b.Store(b.At(ps, row, lane.Size), scale); return; }
            var at = b.Convert(b.Load(K.S32, pos) + b.Convert(row % steps, K.S32), K.U32);      // below 0: past the capacity (unsigned)
            b.If(at < cap, () => b.Store(b.At(ps, row / steps * cap + at, lane.Size), scale));
        });
        return b.Build();
    }

    /// <summary>The row scales (lane values at px[head·steps + step]) into the scales of f (a float of 8 or 16 bits; n =
    /// heads·cap values), an invocation a 32-bit word: the written rows' values packed, the word's other values kept; a word
    /// with no written row not stored. Slots: x, scales, pos, n, cap, steps.</summary>
    public static SpvKernel CacheScalesWords(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"cache_scales_words_{f.Name}_{lane.Name}", Threads, vk.Controls);
        V px = b.Arg(0), ps = b.Arg(1), pos = b.Arg(2);
        V n = b.Arg(3, K.U32), cap = b.Arg(4, K.U32), steps = b.Arg(5, K.U32);
        var gid = b.GlobalX;
        var w = WrittenRows(b, pos, steps, cap);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var address = b.At(ps, gid, 4);
            var word = b.Load(K.U32, address);
            var touched = b.Bool(false);
            var last = n - 1;
            for (int i = 0; i < f.Per; i++)
            {
                var at = gid * (double)f.Per + i;
                var (written, index) = WrittenAt(b, b.Min(at, last), cap, steps, w);
                written = written & (at < n);
                var fresh = b.Load(k, b.At(px, b.Select(written, index, b.U(0)), lane.Size));
                word = b.Select(written, Codec.Pack(b, f, fresh, lane, i, word), word);
                touched = touched | written;
            }

            b.If(touched, () => b.Store(address, word));
        });
        return b.Build();
    }

    /// <summary>A block format's row scales into blocks (f32): the new scales (lane values at x[head·steps + step]) where the
    /// write fills the value, else the value as it is stored. Slots: x, scales, pos, cap, steps.</summary>
    public static SpvKernel CacheScalesEncoder(Vulkan vk, Fmt f, Lane lane) =>
        BlockEncoder(vk, f, $"cache_scales_{lane.Name}", b =>
        {
            var k = Kinds.Of(lane);
            V px = b.Arg(0), ps = b.Arg(1), pos = b.Arg(2);
            V cap = b.Arg(3, K.U32), steps = b.Arg(4, K.U32);
            var w = WrittenRows(b, pos, steps, cap);
            return (at, _) =>
            {
                var (written, index) = WrittenAt(b, at, cap, steps, w);
                var fresh = b.Load(k, b.At(px, index, lane.Size));
                return b.Select(written, lane.Wide ? b.Convert(fresh, K.F32) : fresh, Codec.UnpackAt(b, f, ps, at, Lane.F32));
            };
        });

    /// <summary>The encoder's staging slots into the scales' words (BlockCompact), the blocks that hold a written row only: a
    /// block the write does not touch keeps its bytes. Slots: staging, scales, blocks, pos, n, cap, steps.</summary>
    public static SpvKernel CacheScalesCompact(Vulkan vk, Fmt f)
    {
        var b = new Spv($"cache_scales_compact_{f.Name}", Threads, vk.Controls);
        V staging = b.Arg(0), output = b.Arg(1), blocks = b.Arg(2, K.U32), pos = b.Arg(3);
        V n = b.Arg(4, K.U32), cap = b.Arg(5, K.U32), steps = b.Arg(6, K.U32);
        var gid = b.GlobalX;
        var bytes = b.Convert(blocks, K.U64) * (double)f.BlockBytes;
        var words = b.Convert((bytes + 3.0) >> 2, K.U32);
        var w = WrittenRows(b, pos, steps, cap);
        b.If(gid < words, () =>
        {
            var address = b.At(output, gid, 4);
            var old = b.Load(K.U32, address);
            var word = b.U(0);
            for (int j = 0; j < 4; j++)
            {
                var p = gid * 4.0 + j;
                var inside = b.Convert(p, K.U64) < bytes;
                var blk = b.Select(inside, p / (double)f.BlockBytes, b.U(0));
                var off = p - blk * (double)f.BlockBytes;
                var first = blk * (double)f.BlockValues;
                var take = inside & Touched(b, first, b.Min(first + (double)f.BlockValues, n), cap, w);
                var src = b.Load(K.U32, b.At(staging, b.Select(inside, blk * (double)BlockSlotWords(f) + (off >> 2), b.U(0)), 4));
                var fresh = src.Shr((off & 3UL) << 3) & 0xFFUL;
                var kept = (old >> (8 * j)) & 0xFFUL;
                word = word | (b.Select(take, fresh, kept) << (8 * j));
            }

            b.Store(address, word);
        });
        return b.Build();
    }

    /// <summary>The source's rows into the cache at the device's position, an invocation a stored word of fc (a row: wpr
    /// words): each value of f read into the lane and rounded to fc once (Convert's rule: the lane's value, to odd at
    /// float32 first where Fmt.ConvertPlan says); scaled: x / scale by the row's stored scale (f at head·cap + row; 0 where
    /// the scale is 0), an integer code rounded to the nearest (ties to even), held to [min, max]. Slots: x, cache, scales,
    /// pos, rows (heads·steps), steps, cap, d, wpr, min, max (lane).</summary>
    public static SpvKernel CacheWrite(Vulkan vk, Fmt f, Fmt fc, Lane lane, bool scaled)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"cache_write_{f.Name}_{fc.Name}_{lane.Name}{(scaled ? "_scaled" : "")}", Threads, vk.Controls);
        V px = b.Arg(0), pc = b.Arg(1), ps = b.Arg(2), pos = b.Arg(3);
        V rows = b.Arg(4, K.U32), steps = b.Arg(5, K.U32), cap = b.Arg(6, K.U32), d = b.Arg(7, K.U32), wpr = b.Arg(8, K.U32), min = b.Arg(9, k), max = b.Arg(10, k);
        bool toOdd = Fmt.ConvertPlan(LaneFormat(lane), fc).ToOdd;
        var gid = b.GlobalX;
        var row = gid / wpr;
        var wc = gid % wpr;
        b.If(row < rows, () =>
        {
            var at = b.Convert(b.Load(K.S32, pos) + b.Convert(row % steps, K.S32), K.U32);      // below 0: past the capacity (unsigned)
            b.If(at < cap, () =>
            {
                var crow = row / steps * cap + at;
                V? scale = scaled ? Codec.UnpackAt(b, f, ps, crow, lane) : null;
                var word = b.CBits(Codec.WordKind(fc), 0);
                var first = row * d + wc * (double)fc.Per;
                for (int i = 0; i < fc.Per; i++)
                {
                    var v = Codec.UnpackAt(b, f, px, first + i, lane);
                    if (scale is { } s)
                    {
                        var q = b.Select(s > 0.0, v / s, b.C(k, 0));
                        if (!fc.IsFloat) q = b.RoundEven(q);
                        v = b.Min(b.Max(q, min), max);
                    }

                    word = toOdd ? Codec.Pack(b, fc, ToOddF32(b, lane, v), Lane.F32, i, word) : Codec.Pack(b, fc, v, lane, i, word);
                }

                b.Store(b.At(pc, crow * wpr + wc, fc.WordBits / 8), word);
            });
        });
        return b.Build();
    }

    // Value `at` of a decoder's mask: 0 where p + i − win < j ≤ p + i (win 0: every j ≤ p + i) for at = i·cap + j, else low.
    private static V MaskValue(Spv b, K k, V at, V cap, V win, V p, V low)
    {
        var hi = p + b.Convert(at / cap, K.S32);
        var j = b.Convert(at % cap, K.S32);
        var seen = (j <= hi) & ((win == 0) | (j > hi - b.Convert(win, K.S32)));
        return b.Select(seen, b.C(k, 0), low);
    }

    /// <summary>The mask of a decoder's steps (n = rows·cap values of f). Slots: y, pos, n, cap, win, low (lane).</summary>
    public static SpvKernel DecoderMask(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"decoder_mask_{f.Name}_{lane.Name}", Threads, vk.Controls);
        V py = b.Arg(0), pos = b.Arg(1);
        V n = b.Arg(2, K.U32), cap = b.Arg(3, K.U32), win = b.Arg(4, K.U32), low = b.Arg(5, k);
        var p = b.Load(K.S32, pos);
        StoreEach(b, f, lane, py, n, at => MaskValue(b, k, at, cap, win, p, low));
        return b.Build();
    }

    /// <summary>The mask into blocks (f32). Slots: pos, cap, win, low (f32).</summary>
    public static SpvKernel BlockDecoderMask(Vulkan vk, Fmt f) =>
        BlockEncoder(vk, f, "decoder_mask", b =>
        {
            V pos = b.Arg(0), cap = b.Arg(1, K.U32), win = b.Arg(2, K.U32), low = b.Arg(3, K.F32);
            var p = b.Load(K.S32, pos);
            return (at, _) => MaskValue(b, K.F32, at, cap, win, p, low);
        });

    /// <summary>position[0] += steps (one invocation). Slots: pos, steps.</summary>
    public static SpvKernel PositionAdvance(Vulkan vk)
    {
        var b = new Spv("position_advance", 1, vk.Controls);
        V pos = b.Arg(0), steps = b.Arg(1, K.S32);
        b.If(b.GlobalX == 0, () => b.Store(pos, b.Load(K.S32, pos) + steps));
        return b.Build();
    }
}
