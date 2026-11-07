// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Optimizer steps: the parameters and gradients stored as the type, the moments in the type's float lane storage
// (float32 for the narrow floats, float64 for float64) — the lane rule applied to state that accumulates. Every
// step is one pass over the arrays, written once over <TN, TLane, TL>:
//
//   sgd_momentum   m = mu·m + g + wd·p;  p = p − lr·m
//   adam           m = b1·m + (1 − b1)·g;  v = b2·v + (1 − b2)·g²;  p = p − lr·(m / (1 − b1ᵗ)) / (√(v / (1 − b2ᵗ)) + eps)
//   adamw          adam, then p = p − lr·wd·p (decoupled)
//
// Gradient clipping by the global norm is the composition sumsq (whole) → factor → scale (Program's clip row).
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad;

/// <summary>An optimizer step's scalars: the learning rate, the momentum / first-moment decay, the second-moment decay,
/// eps, the weight decay, the step count (for Adam's bias correction).</summary>
public sealed record OptimizerArgs(string Name, double Lr, double B1, double B2, double Eps, double Wd, int Step)
{
    public static readonly string[] Names = ["sgd_momentum", "adam", "adamw"];

    public int Moments => Name == "sgd_momentum" ? 1 : 2;

    /// <summary>The float64 reference: p, m, v updated in place.</summary>
    public void Reference(double[] p, double[] g, double[] m, double[]? v)
    {
        double c1 = 1 - Math.Pow(B1, Step), c2 = 1 - Math.Pow(B2, Step);
        for (int i = 0; i < p.Length; i++)
        {
            if (Name == "sgd_momentum")
            {
                m[i] = B1 * m[i] + g[i] + Wd * p[i];
                p[i] -= Lr * m[i];
                continue;
            }

            m[i] = B1 * m[i] + (1 - B1) * g[i];
            v![i] = B2 * v[i] + (1 - B2) * g[i] * g[i];
            double update = Lr * (m[i] / c1) / (Math.Sqrt(v[i] / c2) + Eps);
            p[i] -= update;
            if (Name == "adamw") p[i] -= Lr * Wd * p[i];
        }
    }
}

internal sealed partial class Kernels
{
    /// <summary>An optimizer step over n values: p, g as the type; m, v as lane storage (v null for one moment). Returns the lane's name.</summary>
    public static string Optimizer<TN>(OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) where TN : struct, IFloatAdad<TN>
    {
        var vis = new OptimizerVisitor(a, p, g, m, v, n);
        TN.Float(ref vis);
        return vis.Lane;
    }

    private struct OptimizerVisitor(OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            const int Tile = 2048;
            int per = TN.Granularity, tile = Tile / per * per;
            int tiles = (n + tile - 1) / tile;
            Team.For(tiles, Math.Max(1, tiles / (Threads * 4)), (long)n * 6, new OptimizerBody<TN, TLane, TL>(a, p, g, m, v, n, tile));
        }
    }

    private struct OptimizerBody<TN, TLane, TL>(OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n, int tile) : Team.IBody
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
                bool sgd = a.Name == "sgd_momentum", adamw = a.Name == "adamw";
                double c1 = 1 - Math.Pow(a.B1, a.Step), c2 = 1 - Math.Pow(a.B2, a.Step);
                Vector<TLane> lr = new(TL.From(a.Lr)), b1 = new(TL.From(a.B1)), b2 = new(TL.From(a.B2)), one = new(TL.From(1));
                Vector<TLane> ob1 = new(TL.From(1 - a.B1)), ob2 = new(TL.From(1 - a.B2)), eps = new(TL.From(a.Eps)), wd = new(TL.From(a.Wd));
                Vector<TLane> ic1 = new(TL.From(1 / c1)), ic2 = new(TL.From(1 / c2)), lrwd = new(TL.From(a.Lr * a.Wd));
                var ml = MemoryMarshal.Cast<byte, TLane>(m.AsSpan());
                var vl = v is null ? default : MemoryMarshal.Cast<byte, TLane>(v.AsSpan());
                for (int t = first; t < last; t++)
                {
                    int at = t * tile, len = Math.Min(tile, n - at);
                    Span<TLane> ps, gs;
                    if (TN.Direct)
                    {
                        ps = MemoryMarshal.Cast<byte, TLane>(p.AsSpan()).Slice(at, len);
                        gs = MemoryMarshal.Cast<byte, TLane>(g.AsSpan()).Slice(at, len);
                    }
                    else
                    {
                        ps = pl.AsSpan(0, len); gs = gl.AsSpan(0, len);
                        TN.Load(p, at, ps);
                        TN.Load(g, at, gs);
                    }

                    var ms = ml.Slice(at, len);
                    var vs = v is null ? default : vl.Slice(at, len);
                    int i = 0;
                    for (; i + w <= len; i += w)
                    {
                        var pv = new Vector<TLane>(ps[i..]); var gv = new Vector<TLane>(gs[i..]); var mv = new Vector<TLane>(ms[i..]);
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
                        if (sgd)
                        {
                            ms[i] = TL.Fma(TL.From(a.B1), ms[i], TL.Fma(TL.From(a.Wd), ps[i], gs[i]));
                            ps[i] = TL.Sub(ps[i], TL.Mul(TL.From(a.Lr), ms[i]));
                        }
                        else
                        {
                            ms[i] = TL.Fma(TL.From(a.B1), ms[i], TL.Mul(TL.From(1 - a.B1), gs[i]));
                            vs[i] = TL.Fma(TL.From(a.B2), vs[i], TL.Mul(TL.From(1 - a.B2), TL.Mul(gs[i], gs[i])));
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
