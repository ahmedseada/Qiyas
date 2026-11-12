// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on the CPU (Abstraction/Resample.cs has the plan and the reference): a
// task a run of planes (RowChunk: whole stored words and blocks of the output), each plane's input rows read where they
// are stored into the output type's float lane (Fmt.FloatLane), the horizontal pass into a plane of lane values, then the
// vertical pass, the channel's scale and shift, and the task's outputs stored once (an integer's or bool's by the type's
// own conversion: rounded to nearest, ties to even, held to its range).
using System.Buffers;
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
            var lanes = ArrayPool<TLane>.Shared;
            var src = ArrayPool<TLF>.Shared.Rent(W);
            var outs = lanes.Rent((last - first) * plane);
            var row = lanes.Rent(W);
            var mid = lanes.Rent(H * across);
            try
            {
                ReadOnlySpan<TLane> lw = MemoryMarshal.Cast<byte, TLane>(weights.AsSpan());
                ReadOnlySpan<TLane> vals = values is null ? default : MemoryMarshal.Cast<byte, TLane>(values.AsSpan());
                for (int p = first; p < last; p++)
                {
                    int c = p % a.Channels;
                    // The horizontal pass: each output column's taps over the row, in order.
                    for (int r = 0; r < H; r++)
                    {
                        TF.Load(x, (p * H + r) * W, src.AsSpan(0, W));
                        Widths.Load<TLF, TLane>(src.AsSpan(0, W), row.AsSpan(0, W));
                        for (int ox = 0; ox < across; ox++)
                        {
                            int at = r * across + ox;
                            if (a.XTaps == 0)
                            {
                                mid[at] = row[ox];
                                continue;
                            }

                            int b0 = bounds[2 * ox], n = bounds[2 * ox + 1], wb = ox * a.XTaps;
                            TLane sum = TL.Zero;
                            for (int t = 0; t < n; t++) sum = TL.Fma(row[b0 + t], lw[wb + t], sum);
                            mid[at] = sum;
                        }
                    }

                    // The vertical pass over the horizontal one's values, and the channel's scale and shift.
                    var o = outs.AsSpan((p - first) * plane, plane);
                    for (int oy = 0; oy < OH; oy++)
                    {
                        int b0 = oy, n = 0, wb = 0;
                        if (a.YTaps > 0) { b0 = bounds[a.RowBounds + 2 * oy]; n = bounds[a.RowBounds + 2 * oy + 1]; wb = a.RowWeights + oy * a.YTaps; }
                        for (int ox = 0; ox < OW; ox++)
                        {
                            TLane v;
                            if (a.YTaps == 0) v = mid[oy * across + ox];
                            else
                            {
                                v = TL.Zero;
                                for (int t = 0; t < n; t++) v = TL.Fma(mid[(b0 + t) * across + ox], lw[wb + t], v);
                            }

                            o[oy * OW + ox] = values is null ? v : TL.Fma(v, vals[2 * c], vals[2 * c + 1]);
                        }
                    }
                }

                var done = outs.AsSpan(0, (last - first) * plane);
                if (typeof(TE) == typeof(TLane)) TN.Store(MemoryMarshal.Cast<TLane, TE>(done), y, first * plane);
                else
                {
                    // An integer's (or bool's) values: the lane's taken to the natural lane by the type's conversion, then stored.
                    var natural = ArrayPool<TE>.Shared.Rent(done.Length);
                    try
                    {
                        Widths.Store<TLane, TE>(done, natural.AsSpan(0, done.Length));
                        TN.Store(natural.AsSpan(0, done.Length), y, first * plane);
                    }
                    finally
                    {
                        ArrayPool<TE>.Shared.Return(natural);
                    }
                }
            }
            finally
            {
                ArrayPool<TLF>.Shared.Return(src);
                lanes.Return(outs);
                lanes.Return(row);
                lanes.Return(mid);
            }
        }
    }
}
