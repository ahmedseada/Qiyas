// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on the CPU (Abstraction/Resample.cs has the plan and the reference): a
// task a run of planes (RowChunk: whole stored words and blocks of the output), each plane's input rows read where they
// are stored into the output type's float lane (Fmt.FloatLane). The horizontal pass makes each input row's values as the
// output rows reach it, into a ring of the rows an output row's taps read; the vertical pass then sums them a vector of
// output columns at a time (each column's taps in the definition's order), then the channel's scale and shift, and each
// plane's outputs are stored once (an integer's or bool's by the type's own conversion: rounded to nearest, ties to even,
// held to its range).
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    public static string Resample<TIn, TN>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) where TIn : struct, IAdad<TIn> where TN : struct, IAdad<TN>
    {
        a.Validate();
        return InFloatLane<TN, ResampleCall<TIn>>(new ResampleCall<TIn>(a, x, bounds, weights, values, y));
    }

    private readonly struct ResampleCall<TIn>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : IFloatOf where TIn : struct, IAdad<TIn>
    {
        public string Run<TN, TE, TLane, TL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.Count == 0) return TL.Name;
            var v = new ResampleIn<TN, TE, TLane, TL>(a, x, bounds, weights, values, y);
            TIn.Elementwise(ref v);
            return TL.Name;
        }
    }

    private readonly struct ResampleIn<TN, TE, TLane, TL>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : ILaneVisitor
        where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TF, TLF, TLFL>() where TF : struct, IAdad<TF, TLF> where TLF : unmanaged where TLFL : struct, ILane<TLFL, TLF>
        {
            int plane = a.OutHeight * a.OutWidth;
            long work = (long)a.Count * Math.Max(1, a.XTaps + a.YTaps) + (long)a.InputCount;
            Team.For(a.Planes, RowChunk<TN>(a.Planes, plane, a.Planes / (Threads * 4)), work, new ResampleBody<TF, TLF, TN, TE, TLane, TL>(a, x, bounds, weights, values, y));
        }
    }

    private readonly struct ResampleBody<TF, TLF, TN, TE, TLane, TL>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : Team.IBody
        where TF : struct, IAdad<TF, TLF> where TLF : unmanaged where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int H = a.Height, W = a.Width, OH = a.OutHeight, OW = a.OutWidth, across = a.Across, plane = OH * OW;
            // The horizontal pass's rows a ring of `ring` rows holds: the most input rows an output row reads (its taps), or
            // one row where the vertical pass is skipped. Rows enter in order as the output rows reach them and leave once
            // no later output row reads them (an axis's bounds rise with the output position).
            int ring = Math.Max(1, a.YTaps);
            var lanes = ArrayPool<TLane>.Shared;
            var src = ArrayPool<TLF>.Shared.Rent(W);
            var row = lanes.Rent(W);
            var mid = lanes.Rent(ring * across);
            // Each plane stored on its own where planes are whole storage units of the type, else the task's planes at once (a
            // block stored in pieces would be re-encoded with each).
            bool each = plane % TN.Granularity == 0;
            int span = each ? plane : (last - first) * plane;
            var outs = lanes.Rent(span);
            var natural = typeof(TE) == typeof(TLane) ? null : ArrayPool<TE>.Shared.Rent(span);
            try
            {
                ReadOnlySpan<TLane> lw = MemoryMarshal.Cast<byte, TLane>(weights.AsSpan());
                ReadOnlySpan<TLane> vals = values is null ? default : MemoryMarshal.Cast<byte, TLane>(values.AsSpan());
                int w = Vector<TLane>.Count;
                for (int p = first; p < last; p++)
                {
                    int c = p % a.Channels, made = 0;                  // the plane's rows made so far: input rows 0 .. made
                    var o = outs.AsSpan(each ? 0 : (p - first) * plane, plane);
                    var scale = new Vector<TLane>(values is null ? TL.Zero : vals[2 * c]);
                    var shift = new Vector<TLane>(values is null ? TL.Zero : vals[2 * c + 1]);
                    for (int oy = 0; oy < OH; oy++)
                    {
                        int b0 = oy, n = 1, wb = 0;
                        if (a.YTaps > 0) { b0 = bounds[a.RowBounds + 2 * oy]; n = bounds[a.RowBounds + 2 * oy + 1]; wb = a.RowWeights + oy * a.YTaps; }
                        for (; made < b0 + n; made++) Horizontal(p, made, mid.AsSpan(made % ring * across, across), src, row, lw);
                        var dst = o.Slice(oy * OW, OW);
                        if (a.YTaps == 0)
                        {
                            mid.AsSpan(b0 % ring * across, OW).CopyTo(dst);
                        }
                        else
                        {
                            // The vertical pass a vector of output columns at a time: each column's taps in order (fma), as
                            // the definition sums them.
                            int ox = 0;
                            for (; ox + w <= OW; ox += w)
                            {
                                var v = Vector<TLane>.Zero;
                                for (int t = 0; t < n; t++) v = TL.Fma(new Vector<TLane>(mid.AsSpan((b0 + t) % ring * across + ox, w)), new Vector<TLane>(lw[wb + t]), v);
                                v.CopyTo(dst[ox..]);
                            }

                            for (; ox < OW; ox++)
                            {
                                TLane v = TL.Zero;
                                for (int t = 0; t < n; t++) v = TL.Fma(mid[(b0 + t) % ring * across + ox], lw[wb + t], v);
                                dst[ox] = v;
                            }
                        }

                        if (values is null) continue;
                        int q = 0;
                        for (; q + w <= OW; q += w) TL.Fma(new Vector<TLane>(dst[q..]), scale, shift).CopyTo(dst[q..]);
                        for (; q < OW; q++) dst[q] = TL.Fma(dst[q], vals[2 * c], vals[2 * c + 1]);
                    }

                    if (each) Store(o, natural, p * plane);
                }

                if (!each) Store(outs.AsSpan(0, span), natural, first * plane);
            }
            finally
            {
                ArrayPool<TLF>.Shared.Return(src);
                lanes.Return(row);
                lanes.Return(mid);
                lanes.Return(outs);
                if (natural is not null) ArrayPool<TE>.Shared.Return(natural);
            }
        }

        // Outputs stored as the type once (a task's planes start on whole storage units: RowChunk); an integer's (or bool's)
        // values taken to the natural lane by the type's conversion first.
        private void Store(Span<TLane> o, TE[]? natural, int at)
        {
            if (natural is null) { TN.Store(MemoryMarshal.Cast<TLane, TE>(o), y, at); return; }
            Widths.Store<TLane, TE>(o, natural.AsSpan(0, o.Length));
            TN.Store(natural.AsSpan(0, o.Length), y, at);
        }

        // The horizontal pass of input row r of plane p: each output column's taps over the row, in order (the row read where
        // it is stored, into the lane).
        private void Horizontal(int p, int r, Span<TLane> dst, TLF[] src, TLane[] row, ReadOnlySpan<TLane> lw)
        {
            int W = a.Width;
            TF.Load(x, (p * a.Height + r) * W, src.AsSpan(0, W));
            Widths.Load<TLF, TLane>(src.AsSpan(0, W), row.AsSpan(0, W));
            if (a.XTaps == 0)
            {
                row.AsSpan(0, W).CopyTo(dst);
                return;
            }

            for (int ox = 0; ox < dst.Length; ox++)
            {
                int b0 = bounds[2 * ox], n = bounds[2 * ox + 1], wb = ox * a.XTaps;
                TLane sum = TL.Zero;
                for (int t = 0; t < n; t++) sum = TL.Fma(row[b0 + t], lw[wb + t], sum);
                dst[ox] = sum;
            }
        }
    }
}
