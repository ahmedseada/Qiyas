// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel normalization on the CPU (Abstraction/Resample.cs has the plan and the reference): a task a
// run of planes (RowChunk: whole stored words and blocks of the output), each plane's input rows read where they are stored
// into TN's float lane (or, 8-bit, as integers), the horizontal pass into a plane of lane values (integers clipped to a
// byte), then the vertical pass, the normalization, and the task's outputs stored once.
using System.Buffers;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    public static string Resample<TIn, TN>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) where TIn : struct, IAdad<TIn> where TN : struct, IFloatAdad<TN>
    {
        a.Validate(TIn.Name);
        var v = new ResampleIn<TN>(a, x, bounds, weights, values, y);
        TIn.Elementwise(ref v);
        return v.Lane;
    }

    private struct ResampleIn<TN>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : ILaneVisitor where TN : struct, IFloatAdad<TN>
    {
        public string Lane = "";

        public void Run<TF, TLF, TLFL>() where TF : struct, IAdad<TF, TLF> where TLF : unmanaged where TLFL : struct, ILane<TLFL, TLF>
        {
            var o = new ResampleOut<TF, TLF>(a, x, bounds, weights, values, y);
            TN.Float(ref o);
            Lane = o.Lane;
        }
    }

    private struct ResampleOut<TF, TLF>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : IFloatLaneVisitor
        where TF : struct, IAdad<TF, TLF> where TLF : unmanaged
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = a.Bytes ? $"s32 → {TL.Name}" : TL.Name;
            if (a.Count == 0) return;
            int plane = a.OutHeight * a.OutWidth;
            long work = (long)a.Count * Math.Max(1, a.XTaps + a.YTaps) + (long)a.InputCount;
            Team.For(a.Planes, RowChunk<TN>(a.Planes, plane, a.Planes / (Threads * 4)), work, new ResampleBody<TF, TLF, TN, TLane, TL>(a, x, bounds, weights, values, y));
        }
    }

    private readonly struct ResampleBody<TF, TLF, TN, TLane, TL>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) : Team.IBody
        where TF : struct, IAdad<TF, TLF> where TLF : unmanaged where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int H = a.Height, W = a.Width, OH = a.OutHeight, OW = a.OutWidth, across = a.Across, plane = OH * OW;
            var lanes = ArrayPool<TLane>.Shared;
            var ints = ArrayPool<int>.Shared;
            var src = ArrayPool<TLF>.Shared.Rent(W);
            var outs = lanes.Rent((last - first) * plane);
            TLane[]? row = null, mid = null;
            int[]? irow = null, imid = null;
            if (a.Bytes) { irow = ints.Rent(W); imid = ints.Rent(H * across); }
            else { row = lanes.Rent(W); mid = lanes.Rent(H * across); }
            try
            {
                ReadOnlySpan<TLane> lw = a.Bytes ? default : MemoryMarshal.Cast<byte, TLane>(weights.AsSpan());
                ReadOnlySpan<int> iw = a.Bytes ? MemoryMarshal.Cast<byte, int>(weights.AsSpan()) : default;
                ReadOnlySpan<TLane> vals = values is null ? default : MemoryMarshal.Cast<byte, TLane>(values.AsSpan());
                for (int p = first; p < last; p++)
                {
                    int c = p % a.Channels;
                    // The horizontal pass: each output column's taps over the row, in order.
                    for (int r = 0; r < H; r++)
                    {
                        TF.Load(x, (p * H + r) * W, src.AsSpan(0, W));
                        if (a.Bytes) Widths.Load<TLF, int>(src.AsSpan(0, W), irow.AsSpan(0, W));
                        else Widths.Load<TLF, TLane>(src.AsSpan(0, W), row.AsSpan(0, W));
                        for (int ox = 0; ox < across; ox++)
                        {
                            int at = r * across + ox;
                            if (a.XTaps == 0)
                            {
                                if (a.Bytes) imid![at] = irow![ox];
                                else mid![at] = row![ox];
                                continue;
                            }

                            int b0 = bounds[2 * ox], n = bounds[2 * ox + 1], wb = ox * a.XTaps;
                            if (a.Bytes)
                            {
                                int sum = 1 << (Resampling.PrecisionBits - 1);
                                for (int t = 0; t < n; t++) sum += irow![b0 + t] * iw[wb + t];
                                imid![at] = Resampling.Clip8(sum);
                            }
                            else
                            {
                                TLane sum = TL.Zero;
                                for (int t = 0; t < n; t++) sum = TL.Fma(row![b0 + t], lw[wb + t], sum);
                                mid![at] = sum;
                            }
                        }
                    }

                    // The vertical pass over the horizontal one's values, and the channel's normalization.
                    var o = outs.AsSpan((p - first) * plane, plane);
                    for (int oy = 0; oy < OH; oy++)
                    {
                        int b0 = oy, n = 0, wb = 0;
                        if (a.YTaps > 0) { b0 = bounds[a.RowBounds + 2 * oy]; n = bounds[a.RowBounds + 2 * oy + 1]; wb = a.RowWeights + oy * a.YTaps; }
                        for (int ox = 0; ox < OW; ox++)
                        {
                            if (a.Bytes)
                            {
                                int v;
                                if (a.YTaps == 0) v = imid![oy * across + ox];
                                else
                                {
                                    int sum = 1 << (Resampling.PrecisionBits - 1);
                                    for (int t = 0; t < n; t++) sum += imid![(b0 + t) * across + ox] * iw[wb + t];
                                    v = Resampling.Clip8(sum);
                                }

                                o[oy * OW + ox] = values is null ? TL.From(v) : vals[256 * c + v];
                            }
                            else
                            {
                                TLane v;
                                if (a.YTaps == 0) v = mid![oy * across + ox];
                                else
                                {
                                    v = TL.Zero;
                                    for (int t = 0; t < n; t++) v = TL.Fma(mid![(b0 + t) * across + ox], lw[wb + t], v);
                                }

                                o[oy * OW + ox] = values is null ? v : TL.Fma(v, vals[2 * c], vals[2 * c + 1]);
                            }
                        }
                    }
                }

                TN.Store(outs.AsSpan(0, (last - first) * plane), y, first * plane);
            }
            finally
            {
                ArrayPool<TLF>.Shared.Return(src);
                lanes.Return(outs);
                if (row is not null) lanes.Return(row);
                if (mid is not null) lanes.Return(mid);
                if (irow is not null) ints.Return(irow);
                if (imid is not null) ints.Return(imid);
            }
        }
    }
}
