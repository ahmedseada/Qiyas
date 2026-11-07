// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Group normalization (GroupShape) and the rotary embedding by tables (RopeArgs) on Vulkan: CUDA's kernels (Ptx.Norms.cs)
// and the CPU's arithmetic value for value. The groups' coefficients by a workgroup a group (lane values: μ, rstd, and
// backward m1, k2), the outputs a thread per stored word (a block format's blocks by their encoder), the parameters'
// gradients by a workgroup a channel; the rotation's rows normalized by the rows' statistics (NormStats) with a gain.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    private const int GroupThreads = 256;

    private static V Rstd(Spv b, K k, V variance, V eps) => b.C(k, 1) / b.Sqrt(variance + eps);

    /// <summary>coef[4g..] = μ, rstd and (backward, statistics not given) m1 = Σ γ·dy / M, k2 = rstd·(rstd·Σ γ·dy·(x − μ) / M): a
    /// workgroup per group; the computed μ and σ² written to mean and variance when given (non-zero). Arguments: x, dy, g,
    /// mean, variance, coef, outer, groups, inner, channels, spatial, eps.</summary>
    public static SpvKernel GroupCoefs(Vulkan vk, Fmt f, Lane lane, bool backward, bool given, bool gamma)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"group_coefs_{f.Name}_{lane.Name}{(backward ? "_bwd" : "")}{(given ? "_given" : "")}{(gamma ? "_g" : "")}", GroupThreads, vk.Controls);
        V px = b.Arg(0), pdy = b.Arg(1), pg = b.Arg(2), pmean = b.Arg(3), pvar = b.Arg(4), pcoef = b.Arg(5);
        V outer = b.Arg(6, K.U32), groups = b.Arg(7, K.U32), inner = b.Arg(8, K.U32), channels = b.Arg(9, K.U32), spatial = b.Arg(10, K.U32);
        V eps = b.Arg(11, k);
        var tid = b.LocalIdX;
        var g = b.GroupX;
        var m = outer * inner;
        var count = b.Convert(m, k);
        var sh = b.Shared(k, GroupThreads);
        var step = b.U(GroupThreads);
        Func<V, V, V> add = static (p, q) => p + q;

        // Value j of the group: row o = j / inner, at (o·groups + g)·inner + j − o·inner.
        V At(V j)
        {
            var o = j / inner;
            return (o * groups + g) * inner + (j - o * inner);
        }

        V mu, rstd;
        if (given)
        {
            mu = b.Load(k, b.At(pmean, g, lane.Size));
            rstd = Rstd(b, k, b.Load(k, b.At(pvar, g, lane.Size)), eps);
        }
        else
        {
            var acc = b.Local(b.C(k, 0));
            b.For(tid, m, step, j => acc.V = acc.V + Codec.UnpackAt(b, f, px, At(j), lane));
            var mean = GroupReduce(b, sh, tid, acc.V, add, GroupThreads) / count;
            var acc2 = b.Local(b.C(k, 0));
            b.For(tid, m, step, j =>
            {
                var d = Codec.UnpackAt(b, f, px, At(j), lane) - mean;
                acc2.V = b.Fma(d, d, acc2.V);
            });
            var variance = GroupReduce(b, sh, tid, acc2.V, add, GroupThreads) / count;
            mu = mean;
            rstd = Rstd(b, k, variance, eps);
            b.If(tid == 0, () =>
            {
                b.If(pmean != 0UL, () => b.Store(b.At(pmean, g, lane.Size), mean));
                b.If(pvar != 0UL, () => b.Store(b.At(pvar, g, lane.Size), variance));
            });
        }

        var m1 = b.C(k, 0);
        var k2 = b.C(k, 0);
        if (backward && !given)
        {
            var a1 = b.Local(b.C(k, 0));
            var a2 = b.Local(b.C(k, 0));
            var muv = mu;
            b.For(tid, m, step, j =>
            {
                var idx = At(j);
                var xv = Codec.UnpackAt(b, f, px, idx, lane);
                var dv = Codec.UnpackAt(b, f, pdy, idx, lane);
                var d = xv - muv;
                if (gamma)
                {
                    var gv = Codec.UnpackAt(b, f, pg, idx / spatial % channels, lane);
                    a1.V = b.Fma(gv, dv, a1.V);
                    a2.V = b.Fma(gv * dv, d, a2.V);
                }
                else
                {
                    a1.V = a1.V + dv;
                    a2.V = b.Fma(dv, d, a2.V);
                }
            });
            var s1 = GroupReduce(b, sh, tid, a1.V, add, GroupThreads);
            var s2 = GroupReduce(b, sh, tid, a2.V, add, GroupThreads);
            m1 = s1 / count;
            k2 = rstd * (rstd * s2 / count);
        }

        V mo = mu, ro = rstd, m1o = m1, k2o = k2;
        b.If(tid == 0, () =>
        {
            b.Store(b.At(pcoef, g * 4, lane.Size), mo);
            b.Store(b.At(pcoef, g * 4 + 1, lane.Size), ro);
            b.Store(b.At(pcoef, g * 4 + 2, lane.Size), m1o);
            b.Store(b.At(pcoef, g * 4 + 3, lane.Size), k2o);
        });
        return b.Build();
    }

    // A group's value from its coefficients: forward x̂·γ + β (fma with a gain and a bias, an add with a bias only, a product
    // with a gain only), backward (γ·dy − m1 − (x − μ)·k2)·rstd.
    private static V GroupValue(Spv b, Fmt f, Lane lane, K k, bool backward, bool gamma, bool beta, V pg, V pb, V pcoef, V groups, V inner, V channels, V spatial, V idx, V x, V? dy)
    {
        var g = idx / inner % groups;
        var c = idx / spatial % channels;
        var mu = b.Load(k, b.At(pcoef, g * 4, lane.Size));
        var rstd = b.Load(k, b.At(pcoef, g * 4 + 1, lane.Size));
        V? gv = gamma ? Codec.UnpackAt(b, f, pg, c, lane) : null;
        var d = x - mu;
        if (backward)
        {
            var m1 = b.Load(k, b.At(pcoef, g * 4 + 2, lane.Size));
            var k2 = b.Load(k, b.At(pcoef, g * 4 + 3, lane.Size));
            var gd = gv is { } gg ? gg * dy!.Value : dy!.Value;
            return (gd - m1 - d * k2) * rstd;
        }

        var h = d * rstd;
        if (beta)
        {
            var bv = Codec.UnpackAt(b, f, pb, c, lane);
            return gv is { } gg ? b.Fma(h, gg, bv) : h + bv;
        }

        return gv is { } g2 ? h * g2 : h;
    }

    /// <summary>The group outputs a thread per stored word. Arguments: x, dy, g, b, coef, y, n, groups, inner, channels, spatial.</summary>
    public static SpvKernel GroupOut(Vulkan vk, Fmt f, Lane lane, bool backward, bool gamma, bool beta)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"group_out_{f.Name}_{lane.Name}{(backward ? "_bwd" : "")}{(gamma ? "_g" : "")}{(beta ? "_b" : "")}", Threads, vk.Controls);
        V px = b.Arg(0), pdy = b.Arg(1), pg = b.Arg(2), pb = b.Arg(3), pcoef = b.Arg(4), py = b.Arg(5);
        V n = b.Arg(6, K.U32), groups = b.Arg(7, K.U32), inner = b.Arg(8, K.U32), channels = b.Arg(9, K.U32), spatial = b.Arg(10, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            int bytes = f.WordBits / 8;
            var wx = b.Load(wk, b.At(px, gid, bytes));
            var wd = backward ? b.Load(wk, b.At(pdy, gid, bytes)) : wx;
            var wo = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var idx = gid * (double)f.Per + i;
                var x = Codec.Unpack(b, f, wx, i, lane);
                V? dy = backward ? Codec.Unpack(b, f, wd, i, lane) : null;
                wo = Codec.Pack(b, f, GroupValue(b, f, lane, k, backward, gamma, beta, pg, pb, pcoef, groups, inner, channels, spatial, idx, x, dy), lane, i, wo);
            }

            b.Store(b.At(py, gid, bytes), wo);
        });
        return b.Build();
    }

    /// <summary>The group outputs into blocks (f32). Slots: x, dy, g, b, coef, groups, inner, channels, spatial.</summary>
    public static SpvKernel BlockGroupOut(Vulkan vk, Fmt f, bool backward, bool gamma, bool beta) =>
        BlockEncoder(vk, f, $"group_out{(backward ? "_bwd" : "")}{(gamma ? "_g" : "")}{(beta ? "_b" : "")}", b =>
        {
            V px = b.Arg(0), pdy = b.Arg(1), pg = b.Arg(2), pb = b.Arg(3), pcoef = b.Arg(4);
            V groups = b.Arg(5, K.U32), inner = b.Arg(6, K.U32), channels = b.Arg(7, K.U32), spatial = b.Arg(8, K.U32);
            return (at, _) => GroupValue(b, f, Lane.F32, K.F32, backward, gamma, beta, pg, pb, pcoef, groups, inner, channels, spatial, at,
                Codec.UnpackAt(b, f, px, at, Lane.F32), backward ? Codec.UnpackAt(b, f, pdy, at, Lane.F32) : null);
        });

    /// <summary>A workgroup per channel c: dgamma[c] = Σ dy·(x − μ)·rstd, dbeta[c] = Σ dy over its per values, each output when
    /// non-zero. Arguments: x, dy, coef, dgamma, dbeta, per, groups, inner, channels, spatial.</summary>
    public static SpvKernel GroupParams(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"group_params_{f.Name}_{lane.Name}", GroupThreads, vk.Controls);
        V px = b.Arg(0), pdy = b.Arg(1), pcoef = b.Arg(2), pdg = b.Arg(3), pdb = b.Arg(4);
        V per = b.Arg(5, K.U32), groups = b.Arg(6, K.U32), inner = b.Arg(7, K.U32), channels = b.Arg(8, K.U32), spatial = b.Arg(9, K.U32);
        var tid = b.LocalIdX;
        var c = b.GroupX;
        var sh = b.Shared(k, GroupThreads);
        var ag = b.Local(b.C(k, 0));
        var ab = b.Local(b.C(k, 0));
        b.For(tid, per, b.U(GroupThreads), j =>
        {
            var kk = j / spatial;
            var idx = (kk * channels + c) * spatial + (j - kk * spatial);
            var g = idx / inner % groups;
            var mu = b.Load(k, b.At(pcoef, g * 4, lane.Size));
            var rstd = b.Load(k, b.At(pcoef, g * 4 + 1, lane.Size));
            var dv = Codec.UnpackAt(b, f, pdy, idx, lane);
            var h = (Codec.UnpackAt(b, f, px, idx, lane) - mu) * rstd;
            ag.V = b.Fma(dv, h, ag.V);
            ab.V = ab.V + dv;
        });
        Func<V, V, V> add = static (p, q) => p + q;
        var sg = GroupReduce(b, sh, tid, ag.V, add, GroupThreads);
        var sb = GroupReduce(b, sh, tid, ab.V, add, GroupThreads);
        b.If(tid == 0, () =>
        {
            b.If(pdg != 0UL, () => b.Store(b.At(pdg, c, lane.Size), sg));
            b.If(pdb != 0UL, () => b.Store(b.At(pdb, c, lane.Size), sb));
        });
        return b.Build();
    }

    // The rotary embedding of the value at idx: v (normalized with a gain: v·rstd·(g + offset), rstd = stats[2·row + 1]) and
    // its pair's partner w: fma(∓w, s, v·c) in a pair (− for the pair's low value; s = −sin inverse), v past 2·half.
    private static V RopeValue(Spv b, Fmt f, Lane lane, K k, bool gain, bool interleaved, bool inverse, bool rotate,
        V px, V pg, V pcos, V psin, V ppos, V pst, V cols, V heads, V steps, V half, V off, V idx, V x)
    {
        var row = idx / cols;
        var col = idx - row * cols;
        V? rstd = gain ? b.Load(k, b.At(pst, row * 2 + 1, lane.Size)) : null;
        V Norm(V v, V at) => rstd is { } r ? v * r * (Codec.UnpackAt(b, f, pg, at, lane) + off) : v;
        var v = Norm(x, col);
        if (!rotate) return v;
        var pos = b.Load(K.U32, b.At(ppos, row / heads % steps, 4));
        V low, p, partner;
        if (interleaved)
        {
            low = (col & 1UL) == 0;
            p = col >> 1;
            partner = col ^ 1UL;
        }
        else
        {
            low = col < half;
            p = b.Select(low, col, col - half);
            partner = b.Select(low, col + half, col - half);
        }

        var inPair = col < (half << 1);
        partner = b.Min(partner, cols - 1);
        p = b.Min(p, half - 1);
        var w = Norm(Codec.UnpackAt(b, f, px, row * cols + partner, lane), partner);
        var tIdx = pos * half + p;
        var cs = b.Load(k, b.At(pcos, tIdx, lane.Size));
        var sn = b.Load(k, b.At(psin, tIdx, lane.Size));
        if (inverse) sn = -sn;
        var r = b.Fma(b.Select(low, -w, w), sn, v * cs);
        return b.Select(inPair, r, v);
    }

    /// <summary>The rotary embedding a thread per stored word. Arguments: x, g, cos, sin, positions, stats, y, n, cols, heads,
    /// steps, half, offset.</summary>
    public static SpvKernel RopeOut(Vulkan vk, Fmt f, Lane lane, bool gain, bool interleaved, bool inverse, bool rotate)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"rope_out_{f.Name}_{lane.Name}{(gain ? "_norm" : "")}{(interleaved ? "_il" : "")}{(inverse ? "_inv" : "")}{(rotate ? "" : "_norot")}", Threads, vk.Controls);
        V px = b.Arg(0), pg = b.Arg(1), pcos = b.Arg(2), psin = b.Arg(3), ppos = b.Arg(4), pst = b.Arg(5), py = b.Arg(6);
        V n = b.Arg(7, K.U32), cols = b.Arg(8, K.U32), heads = b.Arg(9, K.U32), steps = b.Arg(10, K.U32), half = b.Arg(11, K.U32), off = b.Arg(12, k);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            int bytes = f.WordBits / 8;
            var wx = b.Load(wk, b.At(px, gid, bytes));
            var wo = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                // A value past the last (in the last word) is computed from the last one's index and stored as given.
                var idx = b.Min(gid * (double)f.Per + i, n - 1);
                var x = Codec.Unpack(b, f, wx, i, lane);
                wo = Codec.Pack(b, f, RopeValue(b, f, lane, k, gain, interleaved, inverse, rotate, px, pg, pcos, psin, ppos, pst, cols, heads, steps, half, off, idx, x), lane, i, wo);
            }

            b.Store(b.At(py, gid, bytes), wo);
        });
        return b.Build();
    }

    /// <summary>The rotary embedding into blocks (f32). Slots: x, g, cos, sin, positions, stats, cols, heads, steps, half, offset (f32).</summary>
    public static SpvKernel BlockRopeOut(Vulkan vk, Fmt f, bool gain, bool interleaved, bool inverse, bool rotate) =>
        BlockEncoder(vk, f, $"rope_out{(gain ? "_norm" : "")}{(interleaved ? "_il" : "")}{(inverse ? "_inv" : "")}{(rotate ? "" : "_norot")}", b =>
        {
            V px = b.Arg(0), pg = b.Arg(1), pcos = b.Arg(2), psin = b.Arg(3), ppos = b.Arg(4), pst = b.Arg(5);
            V cols = b.Arg(6, K.U32), heads = b.Arg(7, K.U32), steps = b.Arg(8, K.U32), half = b.Arg(9, K.U32), off = b.Arg(10, K.F32);
            return (at, _) => RopeValue(b, f, Lane.F32, K.F32, gain, interleaved, inverse, rotate, px, pg, pcos, psin, ppos, pst, cols, heads, steps, half, off, at,
                Codec.UnpackAt(b, f, px, at, Lane.F32));
        });
}
