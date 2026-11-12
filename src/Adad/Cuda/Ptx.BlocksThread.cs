// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The element-wise kernels of q4_0 (and the tiled transposes' encoding of q8_0 and q4_0) take a thread per block: the thread
// reads its blocks whole (the scale, then the codes as halves: a block is 34 or 18 bytes, 2-byte aligned), takes the
// op's f32 formula over its 32 values, and encodes and stores its output block itself. A warp per block (EncodeBlock)
// gives each lane one value and spends, on every 32 values, a warp's address arithmetic and byte loads per value, two
// butterflies of five shuffles and a byte store per value. The arithmetic is the warp's: each value decoded as BlockRun
// does, the block's largest |x| (NaN as 0), q4_0's first value of that |x|, the scale's division and rounding to f16,
// its inverse, each code (EncodeQ0).
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    /// <summary>Whether the element-wise block kernels (AddBlocks, MapBlocks, MapRowBlocks) take a thread per block of f:
    /// q4_0, its 18-byte blocks read and written as 9 halves a thread. q8_0 measured slower so (0.45×: its 34-byte blocks
    /// as 17 halves a thread, 34 bytes apart across the warp) and keeps the warp, as q4_k and q6_k (256 values a block).</summary>
    public static bool BlockByThread(Fmt f) => f.Is("q4_0");

    /// <summary>The most PTX lines a value's formula may take for MapBlocks and MapRowBlocks to take a thread per block (the
    /// thread's 32 formulas unrolled): measured once, exp's 29 lines 2.1× faster than the warp, gelu_erf's 231 0.75× (the
    /// unrolled formulas past the instruction cache).</summary>
    public const int ThreadFormulaLines = 40;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(string, int), int> s_formulaLines = new();

    /// <summary>Whether MapBlocks and MapRowBlocks of op take a thread per block of f: BlockByThread, and op's formula within
    /// ThreadFormulaLines a value.</summary>
    public static bool MapByThread(Gpu gpu, Fmt f, MapOp op) => BlockByThread(f) && FormulaLines(gpu, op.Name) <= ThreadFormulaLines;

    // The PTX lines op's f32 formula takes for one value on this device (emitted once into a probe).
    private static int FormulaLines(Gpu gpu, string op) => s_formulaLines.GetOrAdd((op, gpu.Capability), _ =>
    {
        var probe = new PtxBuilder(gpu);
        string x = probe.F(), y = probe.F(), z = probe.F(), a = probe.F(), c = probe.F();
        Formula(probe, Lane.F32, op, x, y, z, a, c);
        return probe.Finish("", "probe", "").Split('\n').Count(l => l.StartsWith("    ", StringComparison.Ordinal) && !l.TrimStart().StartsWith(".reg", StringComparison.Ordinal));
    });

    /// <summary>The threads an element-wise block kernel of f takes over count values: one a block where it takes a thread
    /// per block, else a warp a block (BlockThreads).</summary>
    public static long BlockKernelThreads(Fmt f, long count, bool byThread) =>
        byThread ? (count + f.BlockValues - 1) / f.BlockValues : BlockThreads(f, count);

    // This thread's block over n values (a thread a block); a thread past the last block goes to DONE.
    private static string ThreadBlock(PtxBuilder b, Fmt f, string n)
    {
        string gid = GlobalIndex(b), blocks = b.R(), p = b.P();
        b.L($"add.u32 {blocks}, {n}, {f.BlockValues - 1};");
        b.L($"div.u32 {blocks}, {blocks}, {f.BlockValues};");
        b.L($"setp.ge.u32 {p}, {gid}, {blocks};");
        b.L($"@{p} bra DONE;");
        return gid;
    }

    // Block blk of the array of f at basePtr: its 32 values as f32 registers, each BlockRun's d · code.
    private static string[] ThreadBlockDecode(PtxBuilder b, Fmt f, string basePtr, string blk)
    {
        bool q8 = f.Is("q8_0");
        string addr = b.RD(), h = b.H(), d = b.F();
        b.L($"mul.wide.u32 {addr}, {blk}, {f.BlockBytes};");
        b.L($"add.u64 {addr}, {basePtr}, {addr};");
        b.L($"ld.global.u16 {h}, [{addr}];");
        b.L($"cvt.f32.f16 {d}, {h};");
        var half = new string[q8 ? 16 : 8];
        for (int k = 0; k < half.Length; k++)
        {
            half[k] = b.R();
            b.L($"ld.global.u16 {half[k]}, [{addr}+{2 + 2 * k}];");
        }

        var v = new string[32];
        for (int i = 0; i < 32; i++)
        {
            string q = b.R(), fq = b.F();
            if (q8) b.L($"bfe.s32 {q}, {half[i / 2]}, {8 * (i % 2)}, 8;");               // byte i, signed
            else
            {
                int at = i % 16;                                                           // value i: byte i's low nibble, i − 16's high
                b.L($"bfe.u32 {q}, {half[at / 2]}, {8 * (at % 2) + (i >= 16 ? 4 : 0)}, 4;");
                b.L($"sub.s32 {q}, {q}, 8;");
            }

            b.L($"cvt.rn.f32.s32 {fq}, {q};");
            v[i] = b.F();
            b.L($"mul.rn.f32 {v[i]}, {d}, {fq};");
        }

        return v;
    }

    // The 32 values v (f32 registers; 0 past the array's end) encoded into block blk of the array at outp (EncodeQ0's
    // arithmetic): the scale stored first, then the codes two bytes at a time.
    private static void ThreadBlockEncode(PtxBuilder b, Fmt f, string outp, string blk, string[] v)
    {
        bool q8 = f.Is("q8_0");
        string addr = b.RD(), amax = b.F();
        b.L($"mul.wide.u32 {addr}, {blk}, {f.BlockBytes};");
        b.L($"add.u64 {addr}, {outp}, {addr};");
        b.L($"mov.f32 {amax}, 0f00000000;");
        var a = new string[32];
        for (int i = 0; i < 32; i++)
        {
            a[i] = b.F();
            string m = b.F();
            b.L($"abs.f32 {a[i]}, {v[i]};");
            b.L($"max.f32 {m}, {amax}, {a[i]};");                                          // NaN as 0, as max drops it
            amax = m;
        }

        string maxv = amax;
        if (!q8)
        {
            // The first value with |x| = the max (a max above 0); none: +0.
            string pick = b.F(), pPos = b.P();
            b.L($"mov.f32 {pick}, 0f00000000;");
            for (int i = 31; i >= 0; i--)
            {
                string p = b.P(), s = b.F();
                b.L($"setp.eq.f32 {p}, {a[i]}, {amax};");
                b.L($"selp.f32 {s}, {v[i]}, {pick}, {p};");
                pick = s;
            }

            maxv = b.F();
            b.L($"setp.gt.f32 {pPos}, {amax}, 0f00000000;");
            b.L($"selp.f32 {maxv}, {pick}, 0f00000000, {pPos};");
        }

        string d = b.F(), h = b.H(), dBack = b.F(), id = b.F(), pZero = b.P();
        if (q8) b.L($"div.rn.f32 {d}, {amax}, 0f42FE0000;");                             // / 127
        else b.L($"div.rn.f32 {d}, {maxv}, 0fC1000000;");                                 // / −8
        b.L($"cvt.rn.f16.f32 {h}, {d};");
        b.L($"cvt.f32.f16 {dBack}, {h};");
        b.L($"setp.eq.f32 {pZero}, {dBack}, 0f00000000;");
        b.L($"div.rn.f32 {id}, 0f3F800000, {dBack};");
        b.L($"selp.f32 {id}, 0f00000000, {id}, {pZero};");
        b.L($"st.global.b16 [{addr}], {h};");
        var code = new string[32];
        for (int i = 0; i < 32; i++)
        {
            string t = b.F();
            code[i] = b.R();
            if (q8)
            {
                b.L($"mul.rn.f32 {t}, {v[i]}, {id};");
                b.L($"cvt.rni.s32.f32 {code[i]}, {t};");
                b.L($"max.s32 {code[i]}, {code[i]}, -127;");
                b.L($"min.s32 {code[i]}, {code[i]}, 127;");
            }
            else
            {
                b.L($"fma.rn.f32 {t}, {v[i]}, {id}, 0f41080000;");                       // + 8.5
                b.L($"cvt.rmi.s32.f32 {code[i]}, {t};");                                   // floor
                b.L($"max.s32 {code[i]}, {code[i]}, 0;");
                b.L($"min.s32 {code[i]}, {code[i]}, 15;");
            }
        }

        // Byte j of the codes: q8_0's code j; q4_0's code j with code j + 16 as its high nibble.
        string Byte(int j)
        {
            if (q8) return code[j];
            string c = b.R();
            b.L($"shl.b32 {c}, {code[j + 16]}, 4;");
            b.L($"or.b32 {c}, {c}, {code[j]};");
            return c;
        }

        for (int k = 0; k < (q8 ? 16 : 8); k++)
        {
            string w = b.R(), hw = b.H();
            b.L($"and.b32 {w}, {Byte(2 * k)}, 255;");
            b.L($"bfi.b32 {w}, {Byte(2 * k + 1)}, {w}, 8, 8;");
            b.L($"cvt.u16.u32 {hw}, {w};");
            b.L($"st.global.b16 [{addr}+{2 + 2 * k}], {hw};");
        }
    }

    // Each of block blk's 32 values: value(i, below n) (an f32 register), 0 past n.
    private static string[] ThreadBlockValues(PtxBuilder b, string blk, string n, Func<int, string> value)
    {
        string first = b.R();
        b.L($"shl.b32 {first}, {blk}, 5;");
        var v = new string[32];
        for (int i = 0; i < 32; i++)
        {
            string at = b.R(), pIn = b.P(), raw = value(i);
            b.L($"add.u32 {at}, {first}, {i};");
            b.L($"setp.lt.u32 {pIn}, {at}, {n};");
            v[i] = b.F();
            b.L($"selp.f32 {v[i]}, {raw}, 0f00000000, {pIn};");
        }

        return v;
    }

    // AddBlocks a thread a block.
    private static void AddBlocksThread(PtxBuilder b, Fmt f, string pa, string pb, string pc, string n)
    {
        string blk = ThreadBlock(b, f, n);
        var x = ThreadBlockDecode(b, f, pa, blk);
        var y = ThreadBlockDecode(b, f, pb, blk);
        ThreadBlockEncode(b, f, pc, blk, ThreadBlockValues(b, blk, n, i =>
        {
            string z = b.F();
            b.L(Lane.F32.Add(z, x[i], y[i]));
            return z;
        }));
    }

    // MapBlocks a thread a block.
    private static void MapBlocksThread(PtxBuilder b, Fmt f, MapOp op, string px, string py, string pz, string po, string n, string a, string pb)
    {
        string blk = ThreadBlock(b, f, n);
        string[] x;
        if (op.Inputs >= 1) x = ThreadBlockDecode(b, f, px, blk);
        else
        {
            string zero = b.F();
            b.L($"mov.f32 {zero}, 0f00000000;");
            x = [.. Enumerable.Repeat(zero, 32)];
        }

        var y = op.Inputs >= 2 ? ThreadBlockDecode(b, f, py, blk) : x;
        var z = op.Inputs >= 3 ? ThreadBlockDecode(b, f, pz, blk) : x;
        ThreadBlockEncode(b, f, po, blk, ThreadBlockValues(b, blk, n, i => Formula(b, Lane.F32, op.Name, x[i], y[i], z[i], a, pb)));
    }

    // MapRowBlocks a thread a block: value i's column the previous one's next (from the block's first value's, one remainder).
    private static void MapRowBlocksThread(PtxBuilder b, Fmt f, MapOp op, string px, string pv, string po, string n, string cols)
    {
        string blk = ThreadBlock(b, f, n), zero = b.F(), first = b.R(), col = b.R();
        b.L($"mov.f32 {zero}, 0f00000000;");
        var x = ThreadBlockDecode(b, f, px, blk);
        b.L($"shl.b32 {first}, {blk}, 5;");
        b.L($"rem.u32 {col}, {first}, {cols};");
        ThreadBlockEncode(b, f, po, blk, ThreadBlockValues(b, blk, n, i =>
        {
            if (i > 0)
            {
                string next = b.R(), wrap = b.P();
                b.L($"add.u32 {next}, {col}, 1;");
                b.L($"setp.eq.u32 {wrap}, {next}, {cols};");
                b.L($"selp.u32 {next}, 0, {next}, {wrap};");
                col = next;
            }

            string y = Codec.UnpackAt(b, f, pv, col, Lane.F32);
            return Formula(b, Lane.F32, op.Name, x[i], y, zero, zero, zero);
        }));
    }
}
