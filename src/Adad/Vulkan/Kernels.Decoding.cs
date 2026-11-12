// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's steps on Vulkan (Decoding.cs), CUDA's (Cuda/Ptx.Decoding.cs): the ring's push (an invocation a row); the
// penalties in two kernels — each row's window ordered into a table of P slots (an invocation a slot: its token's rank
// among the window's by (token, slot), its occurrences; the slots past the window the largest integer, so the table is
// ascending) with a mask of 16·P bits after it (bit token mod 16·P set for each token of the window; an invocation a
// word of it), then the store (an invocation a stored word; a block format's by its encoder): a score whose column's bit
// is clear is x's (one load: most of a row); the others' columns sought in the table by a fixed number of halvings, the
// score penalized where it is found; each value rounded once.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>The most tokens a window holds on the device (the window's kernel reads the window once a slot).</summary>
    public const int MaxPenaltyWindow = 4095;

    /// <summary>The slots of a row's table: the power of two above the window's most tokens (a slot past the window always).</summary>
    public static int PenaltySlots(int maxWindow)
    {
        if (maxWindow > MaxPenaltyWindow) throw new NotSupportedException($"vulkan: penalties with a window of {maxWindow} tokens exceed the kernel's {MaxPenaltyWindow}");
        int p = 2;
        while (p <= maxWindow) p *= 2;
        return p;
    }

    /// <summary>history[r·cap + length[0] mod cap] = ids[r]. Slots: ids, history, length, rows, cap.</summary>
    public static SpvKernel HistoryPush(Vulkan vk)
    {
        var b = new Spv("history_push", Threads, vk.Controls);
        V ids = b.Arg(0), hist = b.Arg(1), len = b.Arg(2), rows = b.Arg(3, K.U32), cap = b.Arg(4, K.U32);
        var r = b.GlobalX;
        b.If(r < rows, () => b.Store(b.At(hist, r * cap + b.Load(K.U32, len) % cap, 4), b.Load(K.U32, b.At(ids, r, 4))));
        return b.Build();
    }

    /// <summary>The words a row's table takes: `slots` tokens, `slots` occurrences, slots / 2 words of the mask (4·slots a row).</summary>
    public static int PenaltyRowWords(int slots) => 4 * slots;

    /// <summary>Each row's window as a table of `slots` tokens ascending and their occurrences after them (win[r·4·slots + i],
    /// win[r·4·slots + slots + i], s32): an invocation a slot j — its token the ring's at length − 1 − j (j below the
    /// window's n = min(max(length[0], 0), nmax)), its place the tokens before it by (token, slot); a slot past the window
    /// the largest integer at its own place. Then the mask (win[r·4·slots + 2·slots + w], w below slots / 2): invocation
    /// w's word, bit (token mod 16·slots) of each token of the window. Slots: history, length, win, rows, cap, nmax.</summary>
    public static SpvKernel PenaltyWindow(Vulkan vk, int slots)
    {
        var b = new Spv($"penalty_window_{slots}", Threads, vk.Controls);
        V hist = b.Arg(0), len = b.Arg(1), win = b.Arg(2), rows = b.Arg(3, K.U32), cap = b.Arg(4, K.U32), nmax = b.Arg(5, K.U32);
        var gid = b.GlobalX;
        var r = gid / (double)slots;
        var j = gid % (double)slots;
        b.If(r < rows, () =>
        {
            var l = b.Bitcast(b.Max(b.Load(K.S32, len), b.S(0)), K.U32);
            var n = b.Min(l, nmax);
            var newest = l - 1.0;                                                // its slot is newest mod cap (read below n only)
            V Token(V q) => b.Load(K.S32, b.At(hist, r * cap + (newest - q) % cap, 4));
            var mine = b.Local(b.S(int.MaxValue));
            var rank = b.Local(j);
            var count = b.Local(b.U(0));
            b.If(j < n, () =>
            {
                var t = Token(j);
                var before = b.Local(b.U(0));
                var same = b.Local(b.U(0));
                b.For(b.U(0), n, q =>
                {
                    var o = Token(q);
                    before.V = before.V + b.Select((o < t) | ((o == t) & (q < j)), b.U(1), b.U(0));
                    same.V = same.V + b.Select(o == t, b.U(1), b.U(0));
                });
                mine.V = t;
                rank.V = before.V;
                count.V = same.V;
            });
            var place = r * (double)(4 * slots) + rank.V;
            b.Store(b.At(win, place, 4), mine.V);
            b.Store(b.At(win, place + (double)slots, 4), count.V);
            // The mask's word j (j below slots / 2): the bits of the window's tokens that fall in it.
            b.If(j < (double)(slots / 2), () =>
            {
                var word = b.Local(b.U(0));
                b.For(b.U(0), n, q =>
                {
                    var hb = b.Bitcast(Token(q), K.U32) & (ulong)(16 * slots - 1);
                    word.V = word.V | b.Select((hb >> 5) == j, b.U(1).Shl(hb & 31UL), b.U(0));
                });
                b.Store(b.At(win, r * (double)(4 * slots) + (double)(2 * slots) + j, 4), word.V);
            });
        });
        return b.Build();
    }

    // The value of score `at` of the rows (row at / cols of x at r·stride + off): x's where its column's bit of the row's
    // mask is clear, else penalized where its column is in the row's table — the entries at or below the column counted by
    // halvings over the ascending table (its last slot past every column), the entry before that place the column's when
    // it is in the window.
    private static V PenalizedValue(Spv b, Fmt f, Lane lane, int slots, V at, V px, V win, V cols, V stride, V off, V rep, V pres, V freq)
    {
        var r = at / cols;
        return PenalizedAt(b, f, lane, slots, r, at - r * cols, px, win, stride, off, rep, pres, freq);
    }

    // The value of column col of row r, in f's natural lane: x's as it is stored, or penalized in the float lane and taken
    // back by f's conversion (rounded to nearest even, held to the range).
    private static V PenalizedAt(Spv b, Fmt f, Lane lane, int slots, V r, V col, V px, V win, V stride, V off, V rep, V pres, V freq)
    {
        var natural = f.Natural;
        var c = b.Bitcast(col, K.S32);
        var xn = Codec.UnpackAt(b, f, px, r * stride + off + col, natural);
        var table = r * (double)(4 * slots);
        var hb = col & (ulong)(16 * slots - 1);
        var bits = b.Load(K.U32, b.At(win, table + (double)(2 * slots) + (hb >> 5), 4));
        var y = b.Local(xn);
        b.If((bits.Shr(hb & 31UL) & 1UL) != b.U(0), () =>
        {
            var pos = b.U(0);
            for (int step = slots / 2; step >= 1; step /= 2)
            {
                var t = b.Load(K.S32, b.At(win, table + pos + (double)(step - 1), 4));
                pos = b.Select(t <= c, pos + (double)step, pos);
            }

            var entry = table + b.Max(pos, b.U(1)) - 1.0;
            var hit = (pos > 0.0) & (b.Load(K.S32, b.At(win, entry, 4)) == c);
            var times = b.Convert(b.Load(K.S32, b.At(win, entry + (double)slots, 4)), Kinds.Of(lane));
            var x = Codec.Convert(b, natural, xn, lane);
            var scaled = b.Select(x > 0.0, x / rep, x * rep);
            var penalized = scaled - b.Fma(freq, times, pres);
            y.V = b.Select(hit, lane == natural ? penalized : Codec.Saturate(b, penalized, Kinds.Of(natural)), xn);
        });
        return y.V;
    }

    /// <summary>y (n = rows · cols values of f) = the rows' scores with their tables' tokens penalized. Slots: x, win, y, n,
    /// cols, stride, off, rep, pres, freq (lane).</summary>
    public static SpvKernel PenalizeRows(Vulkan vk, Fmt f, Lane lane, int slots)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"penalize_rows_{f.Name}_{lane.Name}_{slots}", Threads, vk.Controls);
        V px = b.Arg(0), win = b.Arg(1), py = b.Arg(2);
        V n = b.Arg(3, K.U32), cols = b.Arg(4, K.U32), stride = b.Arg(5, K.U32), off = b.Arg(6, K.U32), rep = b.Arg(7, k), pres = b.Arg(8, k), freq = b.Arg(9, k);
        StoreEach(b, f, f.Natural, py, n, at => PenalizedValue(b, f, lane, slots, at, px, win, cols, stride, off, rep, pres, freq));
        return b.Build();
    }

    /// <summary>The sparse store (contiguous rows of whole stored words, not a block format, y already x's copy): an
    /// invocation a slot of a row's table stores the word of y that holds its token, the first slot of the ascending table
    /// with that word alone (one writer a word); every other word is left as the copy made it. Slots: x, win, y, rows,
    /// cols, stride, off, rep, pres, freq (lane).</summary>
    public static SpvKernel PenalizeSparse(Vulkan vk, Fmt f, Lane lane, int slots)
    {
        var k = Kinds.Of(lane);
        var natural = f.Natural;
        int perShift = int.Log2(f.Per);
        var b = new Spv($"penalize_rows_{f.Name}_{lane.Name}_{slots}_sparse", Threads, vk.Controls);
        V px = b.Arg(0), win = b.Arg(1), py = b.Arg(2);
        V rows = b.Arg(3, K.U32), cols = b.Arg(4, K.U32), stride = b.Arg(5, K.U32), off = b.Arg(6, K.U32), rep = b.Arg(7, k), pres = b.Arg(8, k), freq = b.Arg(9, k);
        var gid = b.GlobalX;
        var r = gid / (double)slots;
        var slot = gid % (double)slots;
        var table = r * (double)(4 * slots);
        // The slot's token (below 0 is past the row unsigned; the slots past the window hold the largest integer) and the
        // token of the slot before it: the table ascends, so a token of the same word before this one is there.
        var inRows = r < rows;
        var token = b.Load(K.U32, b.At(win, b.Select(inRows, table + slot, b.U(0)), 4));
        var before = b.Load(K.U32, b.At(win, b.Select(inRows, table + b.Max(slot, b.U(1)) - 1.0, b.U(0)), 4));
        var wx = token >> perShift;
        var taken = (slot > 0.0) & (before < cols) & ((before >> perShift) == wx);
        var words = cols >> perShift;
        b.If(inRows & (token < cols) & !taken, () =>
        {
            var word = b.CBits(Codec.WordKind(f), 0);
            for (int i = 0; i < f.Per; i++)
            {
                var v = PenalizedAt(b, f, lane, slots, r, wx * (double)f.Per + i, px, win, stride, off, rep, pres, freq);
                word = Codec.Pack(b, f, v, natural, i, word);
            }

            b.Store(b.At(py, r * words + wx, f.WordBits / 8), word);
        });
        return b.Build();
    }

    /// <summary>The penalized rows into blocks (f32). Slots: x, win, cols, stride, off, rep, pres, freq (f32).</summary>
    public static SpvKernel BlockPenalizeRows(Vulkan vk, Fmt f, int slots) =>
        BlockEncoder(vk, f, $"penalize_rows_{slots}", b =>
        {
            V px = b.Arg(0), win = b.Arg(1), cols = b.Arg(2, K.U32), stride = b.Arg(3, K.U32), off = b.Arg(4, K.U32);
            V rep = b.Arg(5, K.F32), pres = b.Arg(6, K.F32), freq = b.Arg(7, K.F32);
            return (at, _) => PenalizedValue(b, f, Lane.F32, slots, at, px, win, cols, stride, off, rep, pres, freq);
        });
}
