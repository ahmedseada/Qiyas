// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Optimizer steps: the parameters and gradients stored as the type, the moments in the type's float lane storage
// (float32 for the narrow floats, float64 for float64) — the lane rule applied to state that accumulates. Every
// step is one pass over the arrays, written once over <TN, TLane, TL>:
//
//   sgd            p = p − lr·(g + wd·p)
//   sgd_momentum   m = mu·m + g + wd·p;  p = p − lr·m
//   adam           m = b1·m + (1 − b1)·g;  v = b2·v + (1 − b2)·g²;  p = p − lr·(m / (1 − b1ᵗ)) / (√(v / (1 − b2ᵗ)) + eps)
//   adamw          adam, then p = p − lr·wd·p (decoupled)
//
// A step across several tensors under one clipping (OptimizerMany): the squares of every tensor's gradients as the
// whole sum's parts (Reduce.cs WholeParts, in the sum's own lane), all of them joined in one chain — one sum across the
// tensors — the factor min(1, norm / √sum) by the map's clip_factor in the float lane, then each tensor's step with
// every gradient times the factor as it is read (no gradient is rewritten).

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (m not read by sgd, v null for one
    /// moment). Returns the lane's name.</summary>
    public static string Optimizer<TN>(OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int n) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var vis = new OptimizerVisitor(a, p, g, m, v, n, 1);
        TN.Float(ref vis);
        return vis.Lane;
    }

    /// <summary>A step across several tensors under one clipping of all their gradients (IOptimizers.OptimizerMany); returns
    /// the lane's name.</summary>
    public static string OptimizerMany<TN>(OptimizerArgs a, ReadOnlySpan<OptimizerTensor<byte[]>> tensors, byte[]? sumSquares) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var list = tensors.ToArray();
        double sum = 0;
        if (a.MaxNorm > 0 || sumSquares is not null)
        {
            long terms = 0;
            foreach (var t in list) terms += t.N;
            var squares = new GradientSquares(list);
            TN.Sum(ref squares, new Bounds(0, 0, 0, 0, terms));
            sum = squares.Total;
        }

        return Mixed<TN, ManyCall>(new ManyCall(a, list, sum, sumSquares));
    }

    // The squares of every tensor's gradients, one sum: each tensor's parts in the sum's lane, joined in one chain.
    private struct GradientSquares(OptimizerTensor<byte[]>[] tensors) : ISumVisitor
    {
        public double Total;

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> =>
            Total = Chain<TN, TLane, TLane, TL>(tensors);

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc> =>
            Total = Chain<TN, TIn, TAcc, TD>(tensors);

        private static double Chain<TN, TIn, TAcc, TL>(OptimizerTensor<byte[]>[] tensors)
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
        {
            // Every tensor's parts in one run of the team (a tensor at a time would wake the team once a tensor).
            int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>(), count = tensors.Length;
            var x = new byte[count][];
            var partials = new byte[count][];
            int[] n = new int[count], per = new int[count], first = new int[count + 1];
            long values = 0;
            for (int t = 0; t < count; t++)
            {
                x[t] = tensors[t].G;
                n[t] = tensors[t].N;
                int parts = 0;
                if (n[t] > 0) (parts, per[t]) = WholePlan(n[t], g);
                partials[t] = new byte[parts * Unsafe.SizeOf<TAcc>()];
                first[t + 1] = first[t] + parts;
                values += n[t];
            }

            if (first[count] > 0) Team.For(first[count], 1, values, new ManyPartsBody<TN, TIn, TAcc, TL>(x, partials, n, per, first, g));
            TAcc total = TL.Zero;
            foreach (var parts in partials)
                foreach (var part in MemoryMarshal.Cast<byte, TAcc>(parts.AsSpan())) total = TL.Add(total, part);

            Span<TAcc> one = [total];
            Span<double> wide = stackalloc double[1];
            Widths.Load<TAcc, double>(one, wide);
            return wide[0];
        }
    }

    // The steps of the tensors with the float lane known: the sum as a lane value (stored where asked), the factor from it
    // in the lane, each tensor's step.
    private readonly struct ManyCall(OptimizerArgs a, OptimizerTensor<byte[]>[] tensors, double sum, byte[]? sumSquares) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            TLane total = TL.From(sum);
            if (sumSquares is not null) MemoryMarshal.Cast<byte, TLane>(sumSquares.AsSpan())[0] = total;
            double factor = 1;
            if (a.MaxNorm > 0)
            {
                TLane f = MapOps.ClipFactor<TLane, TL>.Apply(new Vector<TLane>(total), default, default, new Vector<TLane>(TL.From(a.MaxNorm)), default)[0];
                Span<TLane> one = [f];
                Span<double> wide = stackalloc double[1];
                Widths.Load<TLane, double>(one, wide);
                factor = wide[0];
            }

            // Every tensor's tiles in one run of the team.
            int tile = OptimizerTile<TN>();
            var first = new int[tensors.Length + 1];
            long values = 0;
            for (int t = 0; t < tensors.Length; t++)
            {
                first[t + 1] = first[t] + (tensors[t].N + tile - 1) / tile;
                values += tensors[t].N;
            }

            int tiles = first[tensors.Length];
            if (tiles > 0) Team.For(tiles, Math.Max(1, tiles / (Threads * 4)), values * 6, new ManyStepBody<TN, TLane, TL>(a, tensors, first, tile, factor));
            return TL.Name;
        }
    }

    // A tile of the step: 2048 values, whole storage units of the type.
    private static int OptimizerTile<TN>() where TN : struct, IAdad<TN> => 2048 / TN.Granularity * TN.Granularity;

    // Job i is tile i − first[t] of tensor t.
    private readonly struct ManyStepBody<TN, TLane, TL>(OptimizerArgs a, OptimizerTensor<byte[]>[] tensors, int[] first, int tile, double gradientScale) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int from, int to)
        {
            int t = 0;
            while (from < to)
            {
                while (first[t + 1] <= from) t++;
                int end = Math.Min(to, first[t + 1]);
                var x = tensors[t];
                new OptimizerBody<TN, TLane, TL>(a, x.P, x.G, x.M, x.V, x.N, tile, gradientScale).Run(from - first[t], end - first[t]);
                from = end;
            }
        }
    }

    // gradientScale: a lane value (1: the gradients as they are).
    private struct OptimizerVisitor(OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int n, double gradientScale) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            if (n == 0) return;
            int tile = OptimizerTile<TN>();
            int tiles = (n + tile - 1) / tile;
            Team.For(tiles, Math.Max(1, tiles / (Threads * 4)), (long)n * 6, new OptimizerBody<TN, TLane, TL>(a, p, g, m, v, n, tile, gradientScale));
        }
    }

    private struct OptimizerBody<TN, TLane, TL>(OptimizerArgs a, byte[] p, byte[] g, byte[]? m, byte[]? v, int n, int tile, double gradientScale) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            var pl = TN.Direct ? null : pool.Rent(tile);
            var gl = TN.Direct ? null : pool.Rent(tile);
            try
            {
                int w = Vector<TLane>.Count;
                bool plain = a.Name == "sgd", sgd = a.Name == "sgd_momentum", adamw = a.Name == "adamw", scaled = gradientScale != 1;
                double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
                Vector<TLane> lr = new(TL.From(a.Lr)), b1 = new(TL.From(a.B1)), b2 = new(TL.From(a.B2)), one = new(TL.From(1));
                Vector<TLane> ob1 = new(TL.From(1 - a.B1)), ob2 = new(TL.From(1 - a.B2)), eps = new(TL.From(a.Eps)), wd = new(TL.From(a.Wd));
                Vector<TLane> ic1 = new(TL.From(1 / c1)), ic2 = new(TL.From(1 / c2)), lrwd = new(TL.From(a.Lr * a.Wd)), gs = new(TL.From(gradientScale));
                var ml = plain ? default : MemoryMarshal.Cast<byte, TLane>(m.AsSpan());
                var vl = v is null || plain || sgd ? default : MemoryMarshal.Cast<byte, TLane>(v.AsSpan());
                for (int t = first; t < last; t++)
                {
                    int at = t * tile, len = Math.Min(tile, n - at);
                    Span<TLane> ps, gsrc;
                    if (TN.Direct)
                    {
                        ps = MemoryMarshal.Cast<byte, TLane>(p.AsSpan()).Slice(at, len);
                        gsrc = MemoryMarshal.Cast<byte, TLane>(g.AsSpan()).Slice(at, len);
                    }
                    else
                    {
                        ps = pl.AsSpan(0, len); gsrc = gl.AsSpan(0, len);
                        TN.Load(p, at, ps);
                        TN.Load(g, at, gsrc);
                    }

                    var ms = plain ? default : ml.Slice(at, len);
                    var vs = plain || sgd ? default : vl.Slice(at, len);
                    int i = 0;
                    for (; i + w <= len; i += w)
                    {
                        var pv = new Vector<TLane>(ps[i..]); var gv = new Vector<TLane>(gsrc[i..]);
                        if (scaled) gv = TL.Mul(gv, gs);
                        if (plain)
                        {
                            pv -= lr * TL.Fma(wd, pv, gv);
                            pv.CopyTo(ps[i..]);
                            continue;
                        }

                        var mv = new Vector<TLane>(ms[i..]);
                        if (sgd)
                        {
                            mv = TL.Fma(b1, mv, TL.Fma(wd, pv, gv));
                            pv -= lr * mv;
                        }
                        else
                        {
                            var vv = new Vector<TLane>(vs[i..]);
                            mv = TL.Fma(b1, mv, ob1 * gv);
                            vv = TL.Fma(b2, vv, ob2 * gv * gv);
                            var update = lr * (mv * ic1) / (TL.Sqrt(vv * ic2) + eps);
                            pv -= update;
                            if (adamw) pv -= lrwd * pv;
                            vv.CopyTo(vs[i..]);
                        }

                        mv.CopyTo(ms[i..]);
                        pv.CopyTo(ps[i..]);
                    }

                    for (; i < len; i++)
                    {
                        TLane gi = scaled ? TL.Mul(gsrc[i], TL.From(gradientScale)) : gsrc[i];
                        if (plain)
                        {
                            ps[i] = TL.Sub(ps[i], TL.Mul(TL.From(a.Lr), TL.Fma(TL.From(a.Wd), ps[i], gi)));
                        }
                        else if (sgd)
                        {
                            ms[i] = TL.Fma(TL.From(a.B1), ms[i], TL.Fma(TL.From(a.Wd), ps[i], gi));
                            ps[i] = TL.Sub(ps[i], TL.Mul(TL.From(a.Lr), ms[i]));
                        }
                        else
                        {
                            ms[i] = TL.Fma(TL.From(a.B1), ms[i], TL.Mul(TL.From(1 - a.B1), gi));
                            vs[i] = TL.Fma(TL.From(a.B2), vs[i], TL.Mul(TL.From(1 - a.B2), TL.Mul(gi, gi)));
                            var root = TL.Sqrt(new Vector<TLane>(TL.Mul(vs[i], TL.From(1 / c2))))[0];
                            TLane update = TL.Div(TL.Mul(TL.From(a.Lr), TL.Mul(ms[i], TL.From(1 / c1))), TL.Add(root, TL.From(a.Eps)));
                            ps[i] = TL.Sub(ps[i], update);
                            if (adamw) ps[i] = TL.Sub(ps[i], TL.Mul(TL.From(a.Lr * a.Wd), ps[i]));
                        }
                    }

                    if (!TN.Direct) TN.Store(ps, p, at);
                }
            }
            finally
            {
                if (pl is not null) pool.Return(pl);
                if (gl is not null) pool.Return(gl);
            }
        }
    }
}
