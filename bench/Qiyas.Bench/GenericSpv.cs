// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The generic Vulkan side: one kernel generator per operation, written once and specialized by T at run time (the
// Vulkan counterpart of GenericPtx, as a C# generic the JIT specializes per T), through Adad's kernel writer (Spv) in T's
// own kinds (SpvType.Of<T>), never a per-type copy of the kernel. Each kernel is GenericPtx's, step for step: the same
// threads, the same order of the same operations in T. Where GLSL.std.450 has no operation for T (no f64 exp), or where
// GenericPtx steps out of T (f16's reciprocal, mean and root; a moment's or a table's type), the same steps run in the
// same other type (Gaps). Exp is Exp2 of x·log2(e), as GenericPtx's ex2; a quotient, a reciprocal and a root are
// correctly rounded, as its div.rn, rcp.rn and sqrt.rn.
using System.Numerics;
using Adad.Vulkan;

namespace Qiyas;

/// <summary>T as Spv sees it: its kind in registers and in memory (the same: SPIR-V computes in every width the device
/// grants), its size.</summary>
internal sealed record SpvType(string Name, K Reg, K Mem, int Size, bool IsFloat)
{
    public static SpvType Of<T>() where T : INumberBase<T>
    {
        if (typeof(T) == typeof(double)) return new("f64", K.F64, K.F64, 8, true);
        if (typeof(T) == typeof(float)) return new("f32", K.F32, K.F32, 4, true);
        if (typeof(T) == typeof(Half)) return new("f16", K.F16, K.F16, 2, true);
        if (typeof(T) == typeof(long)) return new("s64", K.S64, K.S64, 8, false);
        if (typeof(T) == typeof(int)) return new("s32", K.S32, K.S32, 4, false);
        if (typeof(T) == typeof(short)) return new("s16", K.S16, K.S16, 2, false);
        if (typeof(T) == typeof(sbyte)) return new("s8", K.S8, K.S8, 1, false);
        if (typeof(T) == typeof(ulong)) return new("u64", K.U64, K.U64, 8, false);
        if (typeof(T) == typeof(uint)) return new("u32", K.U32, K.U32, 4, false);
        if (typeof(T) == typeof(ushort)) return new("u16", K.U16, K.U16, 2, false);
        if (typeof(T) == typeof(byte)) return new("u8", K.U8, K.U8, 1, false);
        throw new NotSupportedException(typeof(T).Name);
    }
}

internal static partial class GenericSpv
{
    private const int Threads = 256;

    // Steps of T run in another type, as GenericPtx.Gaps says of CUDA's (the same steps, so the same errors).
    public static string Gaps<T>(string operation) where T : INumberBase<T>
    {
        var t = SpvType.Of<T>();
        return (operation, t.Name) switch
        {
            ("softmax", "f64") => "exp in f32 (GLSL.std.450 has no f64 exp)",
            ("softmax", "f16") => "1/sum in f32 (as GenericPtx: PTX has no f16 rcp)",
            ("rms_norm", "f16") => "mean, sqrt and 1/x in f32 (as GenericPtx: PTX has no f16 div, sqrt or rcp)",
            ("optimizer", "f16") => "the step in f32, the moments' type",
            ("rope", "f16") => "the rotation in f32, the tables' type",
            _ => "",
        };
    }

    /// <summary>The element a thread takes in a grid of ⌈n / 256⌉ workgroups laid out by Grid (x, then rows of x).</summary>
    public static V Element(Spv b) => b.GroupY * (b.GroupsX * (double)Threads) + b.GlobalX;

    /// <summary>The workgroups for n elements a thread each: as many along x as the device takes, then rows of them.</summary>
    public static (uint X, uint Y) Grid(Vulkan vk, long n)
    {
        long groups = Math.Max(1, (n + Threads - 1) / Threads);
        uint x = (uint)Math.Min(groups, vk.MaxGroupsX);
        return (x, (uint)((groups + x - 1) / x));
    }

    /// <summary>c[i] = a[i] + b[i] in T, a thread per element. Slots: a, b, c, n.</summary>
    public static SpvKernel Add<T>(Vulkan vk) where T : INumberBase<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("number_add_" + t.Name, Threads, vk.Controls);
        V pa = b.Arg(0), pb = b.Arg(1), pc = b.Arg(2), n = b.Arg(3, K.U32);
        var i = Element(b);
        b.If(i < n, () => b.Store(b.At(pc, i, t.Size), b.Load(t.Mem, b.At(pa, i, t.Size)) + b.Load(t.Mem, b.At(pb, i, t.Size))));
        return b.Build();
    }

    // ---------------------------------------------------------------- the rows: a workgroup of 256 a row

    // The workgroup's tree over 256 values of kind k in workgroup memory (GenericPtx.Reduce): each invocation's value
    // stored, then halves joined (the larger, as setp.gt and selp: a NaN never taken; or the sum) until one, returned to all.
    private static V Reduce(Spv b, Arr red, V tx, V acc, bool max)
    {
        var v = b.Local(acc);
        red[tx] = v.V;
        b.Barrier();
        for (int s = Threads / 2; s >= 1; s /= 2)
        {
            int half = s;
            b.If(tx < half, () =>
            {
                var w = red[tx + half];
                var mine = red[tx];
                red[tx] = max ? b.Select(w > mine, w, mine) : mine + w;
            });
            b.Barrier();
        }

        var result = red[0];
        b.Barrier();
        return result;
    }

    // Columns tx, tx + 256, … of the row: body(the column's index in the row).
    private static void Columns(Spv b, V tx, V cols, Action<V> body) => b.For(tx, cols, b.U(Threads), body);

    // e^(x − max) in T where GLSL.std.450 has exp for T (Exp2 of (x − max)·log2(e): f32 and f16), else in f32 (f64).
    private static V Exp(Spv b, SpvType t, V x, V max)
    {
        var d = x - max;
        if (t.Reg == K.F64) return b.Convert(b.Exp2(b.Convert(d, K.F32) * 1.4426950408889634), K.F64);
        return b.Exp2(d * 1.4426950408889634);
    }

    /// <summary>Softmax over each row, a workgroup of 256 a row: the max, Σ e^(x − max) with each e stored in y, then y · 1/Σ.
    /// Slots: x, y, cols.</summary>
    public static SpvKernel Softmax<T>(Vulkan vk) where T : IFloatingPointIeee754<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("number_softmax_" + t.Name, Threads, vk.Controls);
        V px = b.Arg(0), py = b.Arg(1), cols = b.Arg(2, K.U32);
        var tx = b.LocalIdX;
        var start = b.GroupX * cols;
        var red = b.Shared(t.Reg, Threads);
        var acc = b.Local(b.C(t.Reg, double.NegativeInfinity));
        Columns(b, tx, cols, c =>
        {
            var x = b.Load(t.Mem, b.At(px, start + c, t.Size));
            acc.V = b.Select(x > acc.V, x, acc.V);
        });
        var max = Reduce(b, red, tx, acc.V, max: true);
        acc.V = b.C(t.Reg, 0);
        Columns(b, tx, cols, c =>
        {
            var e = Exp(b, t, b.Load(t.Mem, b.At(px, start + c, t.Size)), max);
            b.Store(b.At(py, start + c, t.Size), e);
            acc.V = acc.V + e;
        });
        var sum = Reduce(b, red, tx, acc.V, max: false);
        var scale = t.Reg == K.F16 ? b.Convert(b.C(K.F32, 1) / b.Convert(sum, K.F32), K.F16) : b.C(t.Reg, 1) / sum;
        Columns(b, tx, cols, c =>
        {
            var at = b.At(py, start + c, t.Size);
            b.Store(at, b.Load(t.Mem, at) * scale);
        });
        return b.Build();
    }

    /// <summary>y = x · 1/√(mean(x²) + eps) over each row, the scale stored a row in inv; a workgroup of 256 a row, Σ x² in T
    /// by fused multiply-adds. Slots: x, y, inv, cols, eps (f32).</summary>
    public static SpvKernel RmsNorm<T>(Vulkan vk) where T : IFloatingPointIeee754<T>
    {
        var t = SpvType.Of<T>();
        var b = new Spv("number_rms_norm_" + t.Name, Threads, vk.Controls);
        V px = b.Arg(0), py = b.Arg(1), pinv = b.Arg(2), cols = b.Arg(3, K.U32), eps = b.Arg(4, K.F32);
        var tx = b.LocalIdX;
        var row = b.GroupX;
        var start = row * cols;
        var red = b.Shared(t.Reg, Threads);
        var acc = b.Local(b.C(t.Reg, 0));
        Columns(b, tx, cols, c =>
        {
            var x = b.Load(t.Mem, b.At(px, start + c, t.Size));
            acc.V = b.Fma(x, x, acc.V);
        });
        var sum = Reduce(b, red, tx, acc.V, max: false);
        // f64 and f32 in T; f16 in f32 (Gaps).
        var lane = t.Reg == K.F16 ? K.F32 : t.Reg;
        var mean = b.Convert(sum, lane) / b.Convert(cols, lane);
        var scale = b.C(lane, 1) / b.Sqrt(mean + b.Convert(eps, lane));
        var s = b.Convert(scale, t.Reg);
        b.If(tx == 0, () => b.Store(b.At(pinv, row, t.Size), s));
        Columns(b, tx, cols, c => b.Store(b.At(py, start + c, t.Size), b.Load(t.Mem, b.At(px, start + c, t.Size)) * s));
        return b.Build();
    }

    // ---------------------------------------------------------------- optimizers, rope: a thread a value

    /// <summary>One optimizer step (sgd, sgd_momentum, adam, adamw; CpuGeneric.Step's formulas, GenericPtx.Optimizer's
    /// order), a thread per value: p and g of T, the moments of TM, the step in TM. Slots: p, g, m, v, n, then the scalars
    /// lr, b1, b2, eps, wd, c1, c2, 1 − b1, 1 − b2 and the gradient's scale, as TM.</summary>
    public static SpvKernel Optimizer<T, TM>(Vulkan vk, string name) where T : INumberBase<T> where TM : INumberBase<TM>
    {
        SpvType t = SpvType.Of<T>(), m = SpvType.Of<TM>();
        var b = new Spv($"number_{name}_{t.Name}_{m.Name}", Threads, vk.Controls);
        V pp = b.Arg(0), pg = b.Arg(1), pm = b.Arg(2), pv = b.Arg(3), n = b.Arg(4, K.U32);
        var k = m.Reg;
        V lr = b.Arg(5, k), b1 = b.Arg(6, k), b2 = b.Arg(7, k), eps = b.Arg(8, k), wd = b.Arg(9, k), c1 = b.Arg(10, k), c2 = b.Arg(11, k),
            b1c = b.Arg(12, k), b2c = b.Arg(13, k), scale = b.Arg(14, k);
        var i = Element(b);
        b.If(i < n, () =>
        {
            var atP = b.At(pp, i, t.Size);
            var p = b.Convert(b.Load(t.Mem, atP), k);
            var g = b.Convert(b.Load(t.Mem, b.At(pg, i, t.Size)), k) * scale;
            if (name == "sgd") p = p - lr * (g + wd * p);
            else if (name == "sgd_momentum")
            {
                var atM = b.At(pm, i, m.Size);
                var mv = b.Load(k, atM) * b1 + g;
                mv = mv + wd * p;
                b.Store(atM, mv);
                p = p - lr * mv;
            }
            else
            {
                V atM = b.At(pm, i, m.Size), atV = b.At(pv, i, m.Size);
                var mv = b1 * b.Load(k, atM) + b1c * g;
                var vv = b2 * b.Load(k, atV) + b2c * g * g;
                b.Store(atM, mv);
                b.Store(atV, vv);
                var u = lr * (mv / c1) / (b.Sqrt(vv / c2) + eps);
                if (name == "adamw") p = p - lr * wd * p;
                p = p - u;
            }

            b.Store(atP, b.Convert(p, t.Reg));
        });
        return b.Build();
    }

    /// <summary>Σ g² of T in TM: each workgroup of 256 over the values group·256 + tid, + groups·256, …, its sum (the tree in
    /// workgroup memory) into out[group]. Slots: g, out, n.</summary>
    public static SpvKernel SumSquares<T, TM>(Vulkan vk) where T : INumberBase<T> where TM : INumberBase<TM>
    {
        SpvType t = SpvType.Of<T>(), m = SpvType.Of<TM>();
        var b = new Spv($"number_sumsq_{t.Name}_{m.Name}", Threads, vk.Controls);
        V pg = b.Arg(0), po = b.Arg(1), n = b.Arg(2, K.U32);
        var tx = b.LocalIdX;
        var acc = b.Local(b.C(m.Reg, 0));
        b.For(b.GroupX * (double)Threads + tx, n, b.GroupsX * (double)Threads, i =>
        {
            var y = b.Convert(b.Load(t.Mem, b.At(pg, i, t.Size)), m.Reg);
            acc.V = b.Fma(y, y, acc.V);
        });
        var red = b.Shared(m.Reg, Threads);
        var sum = Reduce(b, red, tx, acc.V, max: false);
        b.If(tx == 0, () => b.Store(b.At(po, b.GroupX, m.Size), sum));
        return b.Build();
    }

    /// <summary>The rotary embedding by tables (CpuGeneric.Rope's definition, GenericPtx.Rope's steps), a thread per output
    /// value: column c of row r below 2·half its pair's rotation by the tables of TL at the row's position, in TL; past
    /// 2·half the value copied. Slots: x, cos, sin, positions, y, n, cols, heads, steps, half, interleaved, inverse.</summary>
    public static SpvKernel Rope<T, TL>(Vulkan vk) where T : INumberBase<T> where TL : INumberBase<TL>
    {
        SpvType t = SpvType.Of<T>(), l = SpvType.Of<TL>();
        var b = new Spv($"number_rope_{t.Name}_{l.Name}", Threads, vk.Controls);
        V px = b.Arg(0), pcos = b.Arg(1), psin = b.Arg(2), ppos = b.Arg(3), py = b.Arg(4), n = b.Arg(5, K.U32), cols = b.Arg(6, K.U32),
            heads = b.Arg(7, K.U32), steps = b.Arg(8, K.U32), half = b.Arg(9, K.U32), interleaved = b.Arg(10, K.U32), inverse = b.Arg(11, K.U32);
        var i = Element(b);
        b.If(i < n, () =>
        {
            var r = i / cols;
            var c = i - r * cols;
            var atY = b.At(py, i, t.Size);
            b.If(c < half * 2.0, () =>
            {
                var position = b.Load(K.U32, b.At(ppos, (r / heads) % steps, 4));
                var inter = interleaved != 0.0;
                var second = c >= half;
                var p = b.Select(inter, c >> 1, b.Select(second, c - half, c));
                second = b.Select(inter, (c & 1UL) != 0UL, second);
                var first = b.Select(inter, p * 2.0, p);
                var other = b.Select(inter, p * 2.0 + 1, p + half);
                var rowStart = i - c;
                var xi = b.Convert(b.Load(t.Mem, b.At(px, rowStart + first, t.Size)), l.Reg);
                var xj = b.Convert(b.Load(t.Mem, b.At(px, rowStart + other, t.Size)), l.Reg);
                var at = position * half + p;
                var cv = b.Load(l.Mem, b.At(pcos, at, l.Size));
                var sv = b.Load(l.Mem, b.At(psin, at, l.Size));
                sv = b.Select(inverse != 0.0, -sv, sv);
                var lo = xi * cv - xj * sv;
                var hi = xi * sv + xj * cv;
                b.Store(atY, b.Convert(b.Select(second, hi, lo), t.Reg));
            }, () => b.Store(atY, b.Load(t.Mem, b.At(px, i, t.Size))));
        });
        return b.Build();
    }
}
