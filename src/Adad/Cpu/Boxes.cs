// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses on the CPU (Abstraction/Boxes.cs): a vector of pairs at a time in the type's float lane, each
// component of the pairs gathered into a lane vector from the stored corners (read where they are stored: in place for
// float32 and float64; a narrower type's tile of pairs converted into lanes at once), the reference's formulas in its
// order; the losses (or the gradients, four a pair) rounded to the type once, each tile of pairs storing its own outputs
// (in place for float32 and float64; a tile whole storage units of the type, so no two tiles write one block).

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>loss[i] of count pairs of corner boxes (count × 4 as the type each) as the type, eps the type's; returns the
    /// lane's name.</summary>
    public static string BoxLoss<TN>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[] loss, int count) where TN : struct, IFloatAdad<TN>
    {
        var v = new BoxVisitor(overlap, predicted, target, null, loss, count, TN.Epsilon);
        TN.Float(ref v);
        return v.Lane;
    }

    /// <summary>dpredicted = dloss[i] · ∂loss[i] / ∂predicted[i] (count × 4 as the type); returns the lane's name.</summary>
    public static string BoxLossBackward<TN>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[] dloss, byte[] dpredicted, int count) where TN : struct, IFloatAdad<TN>
    {
        var v = new BoxVisitor(overlap, predicted, target, dloss, dpredicted, count, TN.Epsilon);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct BoxVisitor(BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count, double eps) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            _ = Boxes.Name(overlap);
            if (count == 0) return;
            // A tile: some 512 pairs, whole vectors of them, whose outputs are whole storage units of the type.
            int per = dloss is null ? 1 : 4, w = Vector<TLane>.Count, g = TN.Granularity;
            int tile = Math.Max(1, 512 / w) * w;
            while (tile * per % g != 0) tile += w;
            int tiles = (count + tile - 1) / tile;
            bool stream = Streams.Past(TN.Bytes(8L * count) + TN.Bytes((long)count * (dloss is null ? 1 : 5)));
            Team.For(tiles, Math.Max(1, tiles / (Threads * 4)), (long)count * per * 64, new BoxBody<TN, TLane, TL>(overlap, predicted, target, dloss, output, count, tile, TL.From(eps), stream));
        }
    }

    // stream: each tile's outputs made in lanes (and stored as the type) apart, then copied into the output past the caches
    // (Streams), the output's lines written whole and not read first.
    private readonly struct BoxBody<TN, TLane, TL>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count, int tile, TLane eps, bool stream) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        // At the final tier from the first call, and the pairs' math too (BoxMath.Pair, not inlined): which tier a run
        // times, and what the runtime's profile made of this loop, must not decide the kernel's speed.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int w = Vector<TLane>.Count, per = dloss is null ? 1 : 4;
            var pool = ArrayPool<TLane>.Shared;
            var pr = TN.Direct ? null : pool.Rent(4 * tile);                // a narrower type's tile in lanes
            var tr = TN.Direct ? null : pool.Rent(4 * tile);
            var dr = dloss is null || TN.Direct ? null : pool.Rent(tile);
            var parts = pool.Rent(9 * w);                                   // the pairs' corners by component, and dloss
            var outs = TN.Direct && !stream ? null : pool.Rent(tile * per); // a tile's outputs in lanes, stored as the type once
            var bytes = stream && !TN.Direct ? ArrayPool<byte>.Shared.Rent((int)TN.Bytes(tile * per)) : null;
            try
            {
                for (int k0 = first; k0 < last; k0++)
                {
                    int p0 = k0 * tile, pairs = Math.Min(tile, count - p0);
                    var lanes = TN.Direct && !stream ? MemoryMarshal.Cast<byte, TLane>(output.AsSpan()).Slice(p0 * per, pairs * per) : outs.AsSpan(0, pairs * per);
                    ReadOnlySpan<TLane> pt, tt, dt = default;                // the tile's corners and dloss in lanes
                    if (TN.Direct)
                    {
                        pt = MemoryMarshal.Cast<byte, TLane>(predicted.AsSpan()).Slice(4 * p0, 4 * pairs);
                        tt = MemoryMarshal.Cast<byte, TLane>(target.AsSpan()).Slice(4 * p0, 4 * pairs);
                        if (dloss is not null) dt = MemoryMarshal.Cast<byte, TLane>(dloss.AsSpan()).Slice(p0, pairs);
                    }
                    else
                    {
                        TN.Load(predicted, 4 * p0, pr.AsSpan(0, 4 * pairs));
                        TN.Load(target, 4 * p0, tr.AsSpan(0, 4 * pairs));
                        pt = pr.AsSpan(0, 4 * pairs);
                        tt = tr.AsSpan(0, 4 * pairs);
                        if (dloss is not null) { TN.Load(dloss, p0, dr.AsSpan(0, pairs)); dt = dr.AsSpan(0, pairs); }
                    }

                    for (int i0 = p0; i0 < p0 + pairs; i0 += w)
                    {
                        int n = Math.Min(w, p0 + pairs - i0), o = (i0 - p0) * per;
                        var ps = pt.Slice(4 * (i0 - p0), 4 * n);
                        var ts = tt.Slice(4 * (i0 - p0), 4 * n);
                        var ds = dloss is null ? default : dt.Slice(i0 - p0, n);

                        // Components by lane; the lanes past the count repeat the last pair (their results dropped).
                        for (int j = 0; j < w; j++)
                        {
                            int s = Math.Min(j, n - 1);
                            for (int k = 0; k < 4; k++)
                            {
                                parts[k * w + j] = ps[4 * s + k];
                                parts[(4 + k) * w + j] = ts[4 * s + k];
                            }

                            parts[8 * w + j] = dloss is null ? TL.From(1) : ds[s];
                        }

                        Vector<TLane> V(int k) => new(parts.AsSpan(k * w, w));
                        var r = BoxMath<TLane, TL>.Pair(overlap, dloss is not null, V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7), new Vector<TLane>(eps));
                        if (dloss is null)
                        {
                            if (n == w) r.Loss.CopyTo(lanes.Slice(o, w));
                            else for (int j = 0; j < n; j++) lanes[o + j] = r.Loss[j];
                            continue;
                        }

                        var dl = V(8);
                        Vector<TLane> g0 = r.G0 * dl, g1 = r.G1 * dl, g2 = r.G2 * dl, g3 = r.G3 * dl;
                        for (int j = 0; j < n; j++)
                        {
                            int at = o + 4 * j;
                            lanes[at] = g0[j];
                            lanes[at + 1] = g1[j];
                            lanes[at + 2] = g2[j];
                            lanes[at + 3] = g3[j];
                        }
                    }

                    // A tile that ends in part of a block (the last) is stored in place: the block's other values are the output's.
                    if (stream && pairs * per % TN.Granularity == 0)
                    {
                        int at = (int)TN.Bytes(p0 * per), len = (int)TN.Bytes(pairs * per);
                        if (TN.Direct) Streams.Copy(MemoryMarshal.AsBytes(lanes), output.AsSpan(at, len));
                        else
                        {
                            TN.Store(lanes, bytes.AsSpan(0, len), 0);
                            Streams.Copy(bytes.AsSpan(0, len), output.AsSpan(at, len));
                        }
                    }
                    else if (stream && TN.Direct) lanes.CopyTo(MemoryMarshal.Cast<byte, TLane>(output.AsSpan()).Slice(p0 * per, pairs * per));
                    else if (!TN.Direct) TN.Store(lanes, output, p0 * per);
                }

                if (stream) Streams.Fence();
            }
            finally
            {
                if (bytes is not null) ArrayPool<byte>.Shared.Return(bytes);
                if (pr is not null) pool.Return(pr);
                if (tr is not null) pool.Return(tr);
                if (dr is not null) pool.Return(dr);
                if (outs is not null) pool.Return(outs);
                pool.Return(parts);
            }
        }
    }

    /// <summary>A vector of pairs by Boxes.Reference's formulas and order, in the lane.</summary>
    private static class BoxMath<T, TL> where T : unmanaged where TL : struct, IFloatLane<TL, T>
    {
        private static Vector<T> C(double v) => new(TL.From(v));

        // The share of max(a, b)'s gradient a takes (min's: Below): all of it unless b ranks above a in Adad's order (larger,
        // or a NaN against a number), none where it does.
        private static Vector<T> Above(Vector<T> a, Vector<T> b) =>
            Vector.ConditionalSelect(Vector.GreaterThan(b, a) | Vector.AndNot(Vector.Equals(a, a), Vector.Equals(b, b)), Vector<T>.Zero, C(1));

        private static Vector<T> Below(Vector<T> a, Vector<T> b) => Above(b, a);

        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        public static (Vector<T> Loss, Vector<T> G0, Vector<T> G1, Vector<T> G2, Vector<T> G3) Pair(BoxOverlap overlap, bool grad,
            Vector<T> x1, Vector<T> y1, Vector<T> x2, Vector<T> y2, Vector<T> gx1, Vector<T> gy1, Vector<T> gx2, Vector<T> gy2, Vector<T> eps)
        {
            var zero = Vector<T>.Zero;
            Vector<T> one = C(1), half = C(0.5), two = C(2);
            var iw = Vector.Min(x2, gx2) - Vector.Max(x1, gx1);
            var ih = Vector.Min(y2, gy2) - Vector.Max(y1, gy1);
            var overlaps = Vector.GreaterThan(iw, zero) & Vector.GreaterThan(ih, zero);
            var inter = Vector.ConditionalSelect(overlaps, iw * ih, zero);
            Vector<T> wp = x2 - x1, hp = y2 - y1, wg = gx2 - gx1, hg = gy2 - gy1;
            var union = wp * hp + wg * hg - inter;
            var ue = union + eps;
            var iou = inter / ue;
            var loss = one - iou;
            var cw = Vector.Max(x2, gx2) - Vector.Min(x1, gx1);
            var ch = Vector.Max(y2, gy2) - Vector.Min(y1, gy1);
            var dInter = -(one / ue);
            var dUnion = iou / ue;
            Vector<T> dCw = zero, dCh = zero, dPx = zero, dPy = zero, dW = zero, dH = zero;
            if (overlap == BoxOverlap.GIoU)
            {
                var area = cw * ch;
                var ae = area + eps;
                loss += (area - union) / ae;
                dUnion -= one / ae;
                var dArea = ue / (ae * ae);
                (dCw, dCh) = (dArea * ch, dArea * cw);
            }
            else if (overlap is BoxOverlap.DIoU or BoxOverlap.CIoU)
            {
                var diag = cw * cw + ch * ch + eps;
                var dx = ((x1 - gx1) + (x2 - gx2)) * half;
                var dy = ((y1 - gy1) + (y2 - gy2)) * half;
                var rho = dx * dx + dy * dy;
                loss += rho / diag;
                (dPx, dPy) = (two * dx / diag, two * dy / diag);
                var dDiag = -(rho / (diag * diag));
                (dCw, dCh) = (two * dDiag * cw, two * dDiag * ch);
                if (overlap == BoxOverlap.CIoU)
                {
                    var diff = TL.Atan(wg / hg) - TL.Atan(wp / hp);
                    var v = C(Boxes.AspectFactor) * diff * diff;
                    var alpha = v / (one - iou + v + eps);
                    loss += alpha * v;
                    var held = one - alpha * alpha;                         // ∂(α·v)/∂IoU = α², ∂(α·v)/∂v = α·(2 − α)
                    (dInter, dUnion) = (dInter * held, dUnion * held);
                    var k = two * C(Boxes.AspectFactor) * (alpha * (two - alpha)) * diff / (wp * wp + hp * hp);
                    (dW, dH) = (-(k * hp), k * wp);
                }
            }

            if (!grad) return (loss, zero, zero, zero, zero);
            dInter -= dUnion;
            dW += dUnion * hp;
            dH += dUnion * wp;
            var ix = Vector.ConditionalSelect(overlaps, ih * dInter, zero);
            var iy = Vector.ConditionalSelect(overlaps, iw * dInter, zero);
            var g0 = -dW + dPx * half - ix * Above(x1, gx1) - dCw * Below(x1, gx1);
            var g1 = -dH + dPy * half - iy * Above(y1, gy1) - dCh * Below(y1, gy1);
            var g2 = dW + dPx * half + ix * Below(x2, gx2) + dCw * Above(x2, gx2);
            var g3 = dH + dPy * half + iy * Below(y2, gy2) + dCh * Above(y2, gy2);
            return (loss, g0, g1, g2, g3);
        }
    }
}
