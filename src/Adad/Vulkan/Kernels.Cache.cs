// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Key and value caches on Vulkan (Abstraction/Cache.cs, ICache; VulkanOps.Cache.cs), CUDA's (Cuda/Ptx.Cache.cs): the
// position (an s32 at the device) read by each kernel where it is. The write: an invocation a stored word of the cache (a
// row is whole 32-bit words); a block format's cache by the block encoder into staging, then the words of the touched
// blocks (rows that share blocks) or of the written rows (rows of whole blocks) compacted into the cache. Attention over
// the caches is the decode kernel's (Kernels.Attention.cs, DecodeCache). No sub-word stores: every stored word has one
// writer.
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

    // Value `at` of rows of `unit` values: whether the write fills it, and its source's index ((head·steps + step)·unit +
    // column; 0 when not).
    private static (V Written, V Index) WrittenAt(Spv b, V at, V cap, V steps, V unit, (V P, V W0, V W1, V Any) w)
    {
        var row = at / unit;
        var head = row / cap;
        var r = row - head * cap;
        var written = w.Any & (r >= w.W0) & (r < w.W1);
        return (written, b.Select(written, (head * steps + (r - w.P)) * unit + (at - row * unit), b.U(0)));
    }

    /// <summary>Rows of a block format written where rows share blocks (a block format cache's rows of `unit` values not
    /// whole blocks) into blocks (f32): the written rows' values (fx at
    /// x[(head·steps + step)·unit + column]) where the write fills the value, else the value as it is stored. Slots: x,
    /// destination, pos, cap, steps, unit.</summary>
    public static SpvKernel CacheBlocksEncoder(Vulkan vk, Fmt fb, Fmt fx) =>
        BlockEncoder(vk, fb, $"cache_blocks_from_{fx.Name}", b =>
        {
            V px = b.Arg(0), ps = b.Arg(1), pos = b.Arg(2);
            V cap = b.Arg(3, K.U32), steps = b.Arg(4, K.U32), unit = b.Arg(5, K.U32);
            var w = WrittenRows(b, pos, steps, cap);
            return (at, _) =>
            {
                var (written, index) = WrittenAt(b, at, cap, steps, unit, w);
                return b.Select(written, Codec.UnpackAt(b, fx, px, index, Lane.F32), Codec.UnpackAt(b, fb, ps, at, Lane.F32));
            };
        });

    /// <summary>The encoder's staging slots into the destination's words (BlockCompact), the blocks that hold a written value
    /// only: a block the write does not touch keeps its bytes. Slots: staging, destination, blocks, pos, n, cap, steps, unit.</summary>
    public static SpvKernel CacheBlocksCompact(Vulkan vk, Fmt f)
    {
        var b = new Spv($"cache_blocks_compact_{f.Name}", Threads, vk.Controls);
        V staging = b.Arg(0), output = b.Arg(1), blocks = b.Arg(2, K.U32), pos = b.Arg(3);
        V n = b.Arg(4, K.U32), cap = b.Arg(5, K.U32), steps = b.Arg(6, K.U32), unit = b.Arg(7, K.U32);
        var gid = b.GlobalX;
        var bytes = b.Convert(blocks, K.U64) * (double)f.BlockBytes;
        var words = b.Convert((bytes + 3.0) >> 2, K.U32);
        var rows = WrittenRows(b, pos, steps, cap);
        var w = (rows.P, rows.W0 * unit, rows.W1 * unit, rows.Any);                // the written values of a head: [W0·unit, W1·unit) of cap·unit
        var period = cap * unit;
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
                var take = inside & Touched(b, first, b.Min(first + (double)f.BlockValues, n), period, w);
                var src = b.Load(K.U32, b.At(staging, b.Select(inside, blk * (double)BlockSlotWords(f) + (off >> 2), b.U(0)), 4));
                var fresh = src.Shr((off & 3UL) << 3) & 0xFFUL;
                var kept = (old >> (8 * j)) & 0xFFUL;
                word = word | (b.Select(take, fresh, kept) << (8 * j));
            }

            b.Store(address, word);
        });
        return b.Build();
    }

    /// <summary>Rows of whole blocks into a block format cache at the device's position: the written rows' blocks, encoded
    /// into the staging slots in the source's order (the block encoder over the source), into the cache's words — head h's
    /// run of rows [W0, W1) is bytes [(h·cap + W0)·rowBytes, (h·cap + W1)·rowBytes); invocation (h, w) owns the w-th word
    /// of the run unless the head before reaches it, and writes each of its bytes that a written row holds (from the row's
    /// block in the staging), the others as they are. Slots: staging, cache, pos, heads, steps, cap, rowBytes, wph (the
    /// invocations a head: the most words a run spans).</summary>
    public static SpvKernel CacheRowsCompact(Vulkan vk, Fmt f)
    {
        var b = new Spv($"cache_rows_compact_{f.Name}", Threads, vk.Controls);
        V staging = b.Arg(0), cache = b.Arg(1), pos = b.Arg(2);
        V heads = b.Arg(3, K.U32), steps = b.Arg(4, K.U32), cap = b.Arg(5, K.U32), rowBytes = b.Arg(6, K.U32), wph = b.Arg(7, K.U32);
        var gid = b.GlobalX;
        var h = gid / wph;
        var w = WrittenRows(b, pos, steps, cap);
        b.If(w.Any & (h < heads), () =>
        {
            var firstWord = (h * cap + w.W0) * rowBytes >> 2;
            var lastWord = ((h * cap + w.W1) * rowBytes - 1) >> 2;
            var word = firstWord + (gid - h * wph);
            // The head before reaches at most its own run's last word.
            var before = b.Select(h > 0, ((b.Max(h, b.U(1)) - 1) * cap + w.W1) * rowBytes - 1 >> 2, b.U(0));
            b.If((word <= lastWord) & ((h == 0) | (word > before)), () =>
            {
                var address = b.At(cache, word, 4);
                var old = b.Load(K.U32, address);
                var value = b.U(0);
                for (int j = 0; j < 4; j++)
                {
                    var at = word * 4.0 + j;
                    var row = at / rowBytes;
                    var off = at - row * rowBytes;
                    var head = row / cap;
                    var r = row - head * cap;
                    var written = (head < heads) & (r >= w.W0) & (r < w.W1);
                    // The row's block in the staging: (head·steps + step) blocks-a-row in, then the block within the row.
                    var blk = b.Select(written, (head * steps + (r - w.P)) * (rowBytes / (double)f.BlockBytes) + off / (double)f.BlockBytes, b.U(0));
                    var within = off % (double)f.BlockBytes;
                    var src = b.Load(K.U32, b.At(staging, blk * (double)BlockSlotWords(f) + (within >> 2), 4));
                    var fresh = src.Shr((within & 3UL) << 3) & 0xFFUL;
                    var kept = (old >> (8 * j)) & 0xFFUL;
                    value = value | (b.Select(written, fresh, kept) << (8 * j));
                }

                b.Store(address, value);
            });
        });
        return b.Build();
    }

    /// <summary>The source's rows into the cache at the device's position, an invocation a stored word of fc (a row: wpr
    /// words): each value of f read into the lane and rounded to fc once (Convert's rule: the lane's value, to odd at float32
    /// first where Fmt.ConvertPlan says). Slots: x, cache, pos, rows (heads·steps), steps, cap, d, wpr.</summary>
    public static SpvKernel CacheWrite(Vulkan vk, Fmt f, Fmt fc, Lane lane)
    {
        var b = new Spv($"cache_write_{f.Name}_{fc.Name}_{lane.Name}", Threads, vk.Controls);
        V px = b.Arg(0), pc = b.Arg(1), pos = b.Arg(2);
        V rows = b.Arg(3, K.U32), steps = b.Arg(4, K.U32), cap = b.Arg(5, K.U32), d = b.Arg(6, K.U32), wpr = b.Arg(7, K.U32);
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
                var word = b.CBits(Codec.WordKind(fc), 0);
                var first = row * d + wc * (double)fc.Per;
                for (int i = 0; i < fc.Per; i++)
                {
                    var v = Codec.UnpackAt(b, f, px, first + i, lane);
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
