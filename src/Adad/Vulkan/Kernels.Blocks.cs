// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The block encoders on Vulkan (Blocks.cs, the CPU's encoders step for step; CUDA's Ptx.Blocks.cs): an invocation per
// block evaluates its values from a source (lane storage, or the operation's own formula over its inputs, each decoded
// where it is stored: no lane array of the output exists), the block's statistics and scales as the formats store them
// (the f16 scales rounded once and read back), then its codes; the block's bytes go to a word-aligned staging slot of
// its own (a block of 34, 18 or 210 bytes does not end on a word), and the compaction writes the output a thread per
// stored word from the slots (no sub-word stores). The values of the last block past n are 0.
//
// An encoder kernel: its source's arguments in slots 0 .. 13, the staging at slot 14, n at slot 15; a thread per block.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int StagingSlot = 14, CountSlot = 15;

    /// <summary>Words of a block's staging slot.</summary>
    public static int BlockSlotWords(Fmt f) => (f.BlockBytes + 3) / 4;

    /// <summary>An encoder of f whose values come from source(at, inside) (at below n; inside false for a value past n,
    /// whose result is dropped: a source with side effects predicates them on it), the source built over the kernel's own
    /// arguments (slots 0 .. 13).</summary>
    public static SpvKernel BlockEncoder(Vulkan vk, Fmt f, string name, Func<Spv, Func<V, V, V>> source)
    {
        var b = new Spv($"encode_{name}_{f.Name}", Threads, vk.Controls);
        var value = source(b);
        EncodeBlocks(b, f, b.Arg(StagingSlot), b.Arg(CountSlot, K.U32), value);
        return b.Build();
    }

    /// <summary>The staging slots into the output's words, a thread per stored word (the bytes past the last block kept).
    /// Arguments: staging, output, blocks.</summary>
    public static SpvKernel BlockCompact(Vulkan vk, Fmt f)
    {
        var b = new Spv($"block_compact_{f.Name}", Threads, vk.Controls);
        var staging = b.Arg(0);
        var output = b.Arg(1);
        var blocks = b.Arg(2, K.U32);
        var gid = b.GlobalX;
        var bytes = b.Convert(blocks, K.U64) * (double)f.BlockBytes;
        var words = b.Convert((bytes + 3.0) >> 2, K.U32);
        b.If(gid < words, () =>
        {
            var address = b.At(output, gid, 4);
            var old = b.Load(K.U32, address);
            var word = b.U(0);
            for (int k = 0; k < 4; k++)
            {
                var p = gid * 4.0 + k;
                var inside = b.Convert(p, K.U64) < bytes;
                var blk = p / (double)f.BlockBytes;
                var off = p - blk * (double)f.BlockBytes;
                var src = b.Load(K.U32, b.At(staging, b.Select(inside, blk * (double)BlockSlotWords(f) + (off >> 2), b.U(0)), 4));
                var fresh = src.Shr((off & 3UL) << 3) & 0xFFUL;
                var kept = (old >> (8 * k)) & 0xFFUL;
                word = word | (b.Select(inside, fresh, kept) << (8 * k));
            }

            b.Store(address, word);
        });
        return b.Build();
    }

    // A block slot's bytes, OR-ed into its cleared words (the slot is this invocation's alone).
    private readonly struct BlockOut(Spv b, V staging, V slot)
    {
        public void Byte(V at, V value)
        {
            var address = b.At(staging, slot + (at >> 2), 4);
            b.Store(address, b.Load(K.U32, address) | value.Shl((at & 3UL) << 3));
        }

        public void Byte(int at, V value) => Byte(b.U(at), value);

        public void Half(int at, V code)
        {
            Byte(at, code & 0xFFUL);
            Byte(at + 1, (code >> 8) & 0xFFUL);
        }
    }

    // This invocation's block: its values from the source (0 past n) into a private array, its slot cleared, then encoded.
    private static void EncodeBlocks(Spv b, Fmt f, V staging, V n, Func<V, V, V> source)
    {
        int bv = f.BlockValues, slotWords = BlockSlotWords(f);
        var blk = b.GlobalX;
        var blocks = (n + b.U(bv - 1)) / b.U(bv);
        b.If(blk < blocks, () =>
        {
            var vals = b.Private(K.F32, bv);
            var first = blk * b.U(bv);
            var last = n - b.U(1);
            var zero = b.C(K.F32, 0);
            b.For(b.U(0), b.U(bv), i =>
            {
                var at = first + i;
                var inside = at < n;
                var v = source(b.Min(at, last), inside);
                if (v.Kind != K.F32) v = b.Convert(v, K.F32);                                  // an f64 lane value rounded once
                vals[i] = b.Select(inside, v, zero);
            });
            var slot = blk * b.U(slotWords);
            b.For(b.U(0), b.U(slotWords), w => b.Store(b.At(staging, slot + w, 4), b.U(0)));
            var e = new BlockOut(b, staging, slot);
            if (f.Is("q8_0")) EncodeQ8_0(b, e, vals);
            else if (f.Is("q4_0")) EncodeQ4_0(b, e, vals);
            else if (f.Is("q4_k")) EncodeQ4K(b, e, vals);
            else if (f.Is("q6_k")) EncodeQ6K(b, e, vals);
            else throw new NotSupportedException($"{f.Name}: no block encoder");
        });
    }

    // Math.Clamp((int)x, lo, hi) of an integral f32 (or a truncation toward zero), as s32.
    private static V ClampInt(Spv b, V x, int lo, int hi) => b.Min(b.Max(b.Convert(x, K.S32), b.S(lo)), b.S(hi));

    // An s32 in 0 .. 255 (or its low byte) as u32.
    private static V Low8(Spv b, V s) => b.Bitcast(s, K.U32) & 0xFFUL;

    // The f16 scale's code and its value read back.
    private static (V Code, V Value) Half(Spv b, V x)
    {
        var code = Codec.FloatToHalf(b, x);
        return (code, Codec.HalfToFloat(b, code));
    }

    private static V Inverse(Spv b, V d) => b.Select(d == 0, b.C(K.F32, 0), b.C(K.F32, 1) / d);

    // Q8_0: d = max|x| / 127 (f16), q = clamp(round(x / d), −127, 127).
    private static void EncodeQ8_0(Spv b, BlockOut e, Arr v)
    {
        var amax = b.Local(b.C(K.F32, 0));
        b.For(b.U(0), b.U(32), i => amax.V = b.Max(amax.V, b.Abs(v[i])));
        var (code, d) = Half(b, amax.V / 127.0);
        var id = Inverse(b, d);
        e.Half(0, code);
        b.For(b.U(0), b.U(32), i =>
        {
            var q = b.Min(b.Max(b.RoundEven(v[i] * id), b.C(K.F32, -127)), b.C(K.F32, 127));
            e.Byte(i + 2.0, Low8(b, b.Convert(q, K.S32)));
        });
    }

    // Q4_0: d = (the value of the largest magnitude, the first of them) / −8 (f16), q = clamp((int)(x / d + 8.5), 0, 15).
    private static void EncodeQ4_0(Spv b, BlockOut e, Arr v)
    {
        var amax = b.Local(b.C(K.F32, 0));
        var max = b.Local(b.C(K.F32, 0));
        b.For(b.U(0), b.U(32), i =>
        {
            var x = v[i];
            var a = b.Abs(x);
            var take = a > amax.V;
            amax.V = b.Select(take, a, amax.V);
            max.V = b.Select(take, x, max.V);
        });
        var (code, d) = Half(b, max.V / -8.0);
        var id = Inverse(b, d);
        e.Half(0, code);
        b.For(b.U(0), b.U(16), i =>
        {
            var lo = ClampInt(b, v[i] * id + 8.5, 0, 15);
            var hi = ClampInt(b, v[i + 16.0] * id + 8.5, 0, 15);
            e.Byte(i + 2.0, Low8(b, lo | (hi << 4)));
        });
    }

    // Q4_K: per sub-block of 32 its min (with 0) and scale (max − min) / 15; the 8 scales and mins as 6-bit codes against the
    // block's largest (d, dmin: f16), q = clamp(round((x + dmin·m) / (d·s)), 0, 15).
    private static void EncodeQ4K(Spv b, BlockOut e, Arr v)
    {
        var scales = b.Private(K.F32, 8);
        var mins = b.Private(K.F32, 8);
        var maxScale = b.Local(b.C(K.F32, 0));
        var maxMin = b.Local(b.C(K.F32, 0));
        b.For(b.U(0), b.U(8), j =>
        {
            var lo = b.Local(b.C(K.F32, float.MaxValue));
            var hi = b.Local(b.C(K.F32, float.MinValue));
            b.For(b.U(0), b.U(32), i =>
            {
                var x = v[j * 32.0 + i];
                lo.V = b.Min(lo.V, x);
                hi.V = b.Max(hi.V, x);
            });
            var l0 = b.Min(lo.V, b.C(K.F32, 0));
            var s = (hi.V - l0) / 15.0;
            scales[j] = s;
            mins[j] = -l0;
            maxScale.V = b.Max(maxScale.V, s);
            maxMin.V = b.Max(maxMin.V, -l0);
        });
        var (codeD, d) = Half(b, maxScale.V / 63.0);
        var (codeM, dmin) = Half(b, maxMin.V / 63.0);
        e.Half(0, codeD);
        e.Half(2, codeM);
        var ls = b.Private(K.S32, 8);
        var lm = b.Private(K.S32, 8);
        b.For(b.U(0), b.U(8), j =>
        {
            ls[j] = b.Select(d == 0, b.S(0), ClampInt(b, b.RoundEven(scales[j] / d), 0, 63));
            lm[j] = b.Select(dmin == 0, b.S(0), ClampInt(b, b.RoundEven(mins[j] / dmin), 0, 63));
        });
        // The 12 bytes of 6-bit codes: sub-blocks 0..3 whole in bytes 0..7, 4..7 split (low nibbles in 8..11, the top two
        // bits over the first eight).
        for (int j = 0; j < 4; j++)
        {
            V s0 = ls[j], m0 = lm[j], s1 = ls[j + 4], m1 = lm[j + 4];
            e.Byte(4 + j, Low8(b, s0 | ((s1 >> 4) << 6)));
            e.Byte(8 + j, Low8(b, m0 | ((m1 >> 4) << 6)));
            e.Byte(12 + j, Low8(b, (s1 & 0xFUL) | ((m1 & 0xFUL) << 4)));
        }

        // The codes: sub-block pairs share 32 bytes, the even sub-block in the low nibbles, the odd in the high.
        V Code(V j, V i)
        {
            var dj = d * b.Convert(ls[j], K.F32);
            var mj = dmin * b.Convert(lm[j], K.F32);
            return b.Select(dj == 0, b.S(0), ClampInt(b, b.RoundEven((v[j * 32.0 + i] + mj) / dj), 0, 15));
        }

        b.For(b.U(0), b.U(4), pair => b.For(b.U(0), b.U(32), i =>
        {
            var q0 = Code(pair * 2.0, i);
            var q1 = Code(pair * 2.0 + 1, i);
            e.Byte(pair * 32.0 + i + 16, Low8(b, q0 | (q1 << 4)));
        }));
    }

    // Q6_K: per sub-block of 16 its scale max|x| / 31; the 16 scales as int8 against the block's largest (d: f16),
    // q = clamp(round(x / (d·s)), −32, 31) + 32 as 4 low bits (ql) and 2 high bits (qh).
    private static void EncodeQ6K(Spv b, BlockOut e, Arr v)
    {
        var scales = b.Private(K.F32, 16);
        var maxScale = b.Local(b.C(K.F32, 0));
        b.For(b.U(0), b.U(16), j =>
        {
            var amax = b.Local(b.C(K.F32, 0));
            b.For(b.U(0), b.U(16), i => amax.V = b.Max(amax.V, b.Abs(v[j * 16.0 + i])));
            var s = amax.V / 31.0;
            scales[j] = s;
            maxScale.V = b.Max(maxScale.V, s);
        });
        var (code, d) = Half(b, maxScale.V / 127.0);
        e.Half(208, code);
        var ls = b.Private(K.S32, 16);
        b.For(b.U(0), b.U(16), j =>
        {
            var l = b.Select(d == 0, b.S(0), ClampInt(b, b.RoundEven(scales[j] / d), -127, 127));
            ls[j] = l;
            e.Byte(j + 192.0, Low8(b, l));
        });

        V Q(V idx)
        {
            var dj = d * b.Convert(ls[idx >> 4], K.F32);
            var q = b.Select(dj == 0, b.S(0), ClampInt(b, b.RoundEven(v[idx] / dj), -32, 31));
            return b.Bitcast(q + b.S(32), K.U32);
        }

        for (int half = 0; half < 2; half++)
        {
            int lBase = half * 64, hBase = 128 + half * 32, vBase = half * 128;
            b.For(b.U(0), b.U(32), l =>
            {
                var q1 = Q(l + vBase);
                var q2 = Q(l + (vBase + 32));
                var q3 = Q(l + (vBase + 64));
                var q4 = Q(l + (vBase + 96));
                e.Byte(l + lBase, (q1 & 0xFUL) | ((q3 & 0xFUL) << 4));
                e.Byte(l + (lBase + 32), (q2 & 0xFUL) | ((q4 & 0xFUL) << 4));
                e.Byte(l + hBase, ((q1 >> 4) & 3UL) | (((q2 >> 4) & 3UL) << 2) | (((q3 >> 4) & 3UL) << 4) | (((q4 >> 4) & 3UL) << 6));
            });
        }
    }

    // ---------------------------------------------------------------- the sources

    /// <summary>Lane storage (slot 0) into blocks.</summary>
    public static SpvKernel BlockFromLanes(Vulkan vk, Lane lane, Fmt f) =>
        BlockEncoder(vk, f, $"lanes_{lane.Name}", b =>
        {
            var src = b.Arg(0);
            var k = Kinds.Of(lane);
            return (at, _) => b.Load(k, b.At(src, at, lane.Size));
        });

    /// <summary>Lane storage (slot 0) into blocks at the positions p with p mod period below span, the output's other
    /// values (slot 1, the output itself: read before the compaction writes it) kept. Slots: lanes, output, period, span.</summary>
    public static SpvKernel BlockFromLanesKeep(Vulkan vk, Lane lane, Fmt f) =>
        BlockEncoder(vk, f, $"lanes_keep_{lane.Name}", b =>
        {
            var src = b.Arg(0);
            var dst = b.Arg(1);
            var period = b.Arg(2, K.U32);
            var span = b.Arg(3, K.U32);
            var k = Kinds.Of(lane);
            return (at, _) =>
            {
                var fresh = at % period < span;
                var lv = b.Load(k, b.At(src, b.Select(fresh, at, b.U(0)), lane.Size));
                return b.Select(fresh, lv.Kind == K.F32 ? lv : b.Convert(lv, K.F32), Codec.UnpackAt(b, f, dst, at, Lane.F32));
            };
        });

    /// <summary>c = a + b. Slots: a, b.</summary>
    public static SpvKernel BlockAdd(Vulkan vk, Fmt f) =>
        BlockEncoder(vk, f, "add", b =>
        {
            var pa = b.Arg(0);
            var pb = b.Arg(1);
            return (at, _) => Codec.UnpackAt(b, f, pa, at, Lane.F32) + Codec.UnpackAt(b, f, pb, at, Lane.F32);
        });

    /// <summary>output = op(x, y, z; a, b) in lane `lane`. Slots: x, y, z, a, b (lane scalars).</summary>
    public static SpvKernel BlockMap(Vulkan vk, Fmt f, Lane lane, MapOp op) =>
        BlockEncoder(vk, f, $"map_{op.Name}_{lane.Name}", b =>
        {
            var k = Kinds.Of(lane);
            V px = b.Arg(0), py = b.Arg(1), pz = b.Arg(2), a = b.Arg(3, k), pb = b.Arg(4, k);
            return (at, _) =>
            {
                var x = op.Inputs >= 1 ? Codec.UnpackAt(b, f, px, at, lane) : b.C(k, 0);
                var y = op.Inputs >= 2 ? Codec.UnpackAt(b, f, py, at, lane) : x;
                var z = op.Inputs >= 3 ? Codec.UnpackAt(b, f, pz, at, lane) : x;
                return Formula(b, k, op.Name, x, y, z, a, pb);
            };
        });

    /// <summary>output[r, c] = op(x[r, c], v[c]). Slots: x, v, cols.</summary>
    public static SpvKernel BlockMapRow(Vulkan vk, Fmt f, Lane lane, MapOp op) =>
        BlockEncoder(vk, f, $"maprow_{op.Name}_{lane.Name}", b =>
        {
            var k = Kinds.Of(lane);
            V px = b.Arg(0), pv = b.Arg(1), cols = b.Arg(2, K.U32);
            return (at, _) =>
            {
                var x = Codec.UnpackAt(b, f, px, at, lane);
                var v = Codec.UnpackAt(b, f, pv, at % cols, lane);
                return Formula(b, k, op.Name, x, v, x, b.C(k, 0), b.C(k, 0));
            };
        });

    /// <summary>output (cols × rows) = xᵀ. Slots: x, rows, cols.</summary>
    public static SpvKernel BlockTranspose(Vulkan vk, Fmt f) =>
        BlockEncoder(vk, f, "transpose", b =>
        {
            V src = b.Arg(0), rows = b.Arg(1, K.U32), cols = b.Arg(2, K.U32);
            return (at, _) => Codec.UnpackAt(b, f, src, (at % rows) * cols + at / rows, Lane.F32);
        });

    /// <summary>out[r·k + j] = x[r][index[r·k + j]]. Slots: x, index, cols, k.</summary>
    public static SpvKernel BlockGather(Vulkan vk, Fmt f) =>
        BlockEncoder(vk, f, "gather_by_index", b =>
        {
            V x = b.Arg(0), index = b.Arg(1), cols = b.Arg(2, K.U32), k = b.Arg(3, K.U32);
            return (at, _) =>
            {
                // A column outside the row (top-k's −1 past the row) reads the row's first value, dropped: 0.
                var col = b.Bitcast(b.Load(K.S32, b.At(index, at, 4)), K.U32);
                var inRow = col < cols;
                return b.Select(inRow, Codec.UnpackAt(b, f, x, at / k * cols + b.Select(inRow, col, b.U(0)), Lane.F32), b.F(0));
            };
        });

    /// <summary>A layout operation's output (LayoutValue of the output index). The layout's slots (LayoutSlots: 0 .. 13).</summary>
    public static SpvKernel BlockLayout(Vulkan vk, Fmt f, Lane lane, string name, Fmt? inFmt = null) =>
        BlockEncoder(vk, f, $"layout_{name}_{lane.Name}{(inFmt is null ? "" : "_i" + inFmt.Name)}", b =>
        {
            var a = new LayoutIn(b);
            return (at, _) =>
            {
                var r = at / a.Cols;
                var c = at - r * a.Cols;
                var value = LayoutValue(b, inFmt ?? f, lane, name, a, r, c, at);
                return value.Kind == K.F32 ? value : b.Convert(value, K.F32);
            };
        });

    /// <summary>A row operation's output (y, or y2 when second) replayed with its row's statistics. The rows' slots (0 .. 12).</summary>
    public static SpvKernel BlockRowReplay(Vulkan vk, Fmt f, Lane lane, RowOp op, bool secondInput, bool secondOutput) =>
        BlockEncoder(vk, f, $"rowreplay_{op.Name}_{lane.Name}{(secondInput ? "" : "_nosecond")}{(secondOutput ? "_y2" : "")}", b =>
        {
            var c = new RowCtx(b, f, lane, RowMode.Replay, op.Second && secondInput);
            return (at, _) =>
            {
                c.BeginValue(at);
                EmitRowOp(c, op);
                var v = (secondOutput ? c.Captured2 : c.Captured) ?? throw new InvalidOperationException($"{op.Name}: no output");
                return v.Kind == K.F32 ? v : b.Convert(v, K.F32);
            };
        });

    /// <summary>An optimizer step's new parameters into blocks of p, the moments (lane storage) updated by the source for
    /// each value below n (the old p read before the compaction writes it). Slots: p, g, m, v, then the lane scalars of
    /// Optimizer at 4 .. 13.</summary>
    /// <summary>The slot a scaled block step reads the gradients' scale's address from: lr·wd's (13) for the sgd steps,
    /// wd's (12) for adam and adamw — the scalar that step does not read.</summary>
    public static int BlockOptimizerScaleSlot(string name) => name is "sgd" or "sgd_momentum" ? 13 : 12;

    public static SpvKernel BlockOptimizer(Vulkan vk, Fmt f, Lane lane, string name, bool scaled = false) =>
        BlockEncoder(vk, f, $"optimizer_{name}_{lane.Name}{(scaled ? "_gs" : "")}", b =>
        {
            var k = Kinds.Of(lane);
            V pp = b.Arg(0), pg = b.Arg(1), pm = b.Arg(2), pv = b.Arg(3);
            bool plain = name == "sgd", sgd = plain || name == "sgd_momentum";
            // An encoder's source has slots 0 .. 13, all taken: where the gradients' scale is takes the slot of a scalar
            // the step does not read (BlockOptimizerScaleSlot).
            int scaleSlot = scaled ? BlockOptimizerScaleSlot(name) : -1;
            V Scalar(int slot) => slot == scaleSlot ? b.C(k, 0) : b.Arg(slot, k);
            var s = new StepScalars(Scalar(4), Scalar(5), Scalar(6), Scalar(7), Scalar(8), Scalar(9), Scalar(10), Scalar(11), Scalar(12), Scalar(13));
            var gs = scaled ? b.Arg(scaleSlot) : default;
            return (at, inside) =>
            {
                var p = Codec.UnpackAt(b, f, pp, at, lane);
                var g = Codec.UnpackAt(b, f, pg, at, lane);
                if (scaled) g = g * b.Load(k, gs);
                var mAddress = b.At(pm, at, lane.Size);
                var vAddress = b.At(pv, at, lane.Size);
                var m = plain ? p : b.Load(k, mAddress);
                var v = sgd ? m : b.Load(k, vAddress);
                var (pNew, mNew, vNew) = Step(b, name, p, g, m, v, s);
                b.If(inside, () =>
                {
                    if (!sgd) b.Store(vAddress, vNew);
                    if (!plain) b.Store(mAddress, mNew);
                });
                return pNew;
            };
        });
}
