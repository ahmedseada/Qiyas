// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A scaled weight's products on the CPU (Products.cs ScaledWeight): y = x · Wᵀ with W's values made in the float lane as
// the product's packing reads the codes (ScaledSource: each output's row of codes read once into TW's element-wise lane,
// taken into the float lane, the zero subtracted and the scale multiplied there, laid into the packed strips): no W in
// memory but its codes and the packed panel the product holds of any operand; the sums in the lane, then the epilogue's
// store (Epilogue.cs). ExpandScaled makes the same values a tile at a time and stores them as TN.

using System.Buffers;
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>y = the epilogue's y of x · Wᵀ, W the scaled weight's values; returns the lane's name.</summary>
    public static string MatMulScaled<TN, TW>(ScaledWeight w, int m, ProductEpilogue e, byte[] x, byte[] codes, byte[] scales, byte[]? zeros, byte[] y, ProductTerms<byte[]> t)
        where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        var v = w.View(m);
        var terms = t.Validate(e, v);
        return Mixed<TN, ScaledCall<TW>>(new ScaledCall<TW>(w, v, e, terms, x, codes, scales, zeros, y, t));
    }

    /// <summary>output = the scaled weight's values as TN; returns the lane's name.</summary>
    public static string ExpandScaled<TN, TW>(ScaledWeight w, byte[] codes, byte[] scales, byte[]? zeros, byte[] output) where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        return Mixed<TN, ExpandCall<TW>>(new ExpandCall<TW>(w, codes, scales, zeros, output));
    }

    // The scales as lane values, and the zero points negated (exact), each read once from TN.
    private static (TLane[] Scales, TLane[]? NegZeros) ScaleLanes<TN, TLane, TL>(ScaledWeight w, byte[] scales, byte[]? zeros)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        var s = new TLane[w.Scales];
        TN.Load(scales, 0, s);
        if (zeros is null) return (s, null);
        var z = new TLane[w.Scales];
        TN.Load(zeros, 0, z);
        var minus = TL.From(-1);
        for (int i = 0; i < z.Length; i++) z[i] = TL.Mul(minus, z[i]);
        return (s, z);
    }

    // A run of one row's values (row r, columns c0 .. c0 + len) from their codes in the lane: (code − zero)·scale.
    private static void ScaleRun<TLane, TL>(ScaledWeight w, TLane[] scales, TLane[]? negZeros, int r, int c0, Span<TLane> run) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int bc = w.BlockColumns, rowScales = r / w.BlockRows * w.ScaleColumns;
        for (int i = 0; i < run.Length;)
        {
            int c = c0 + i, at = rowScales + c / bc, end = Math.Min(run.Length, i + bc - c % bc);
            TLane s = scales[at];
            if (negZeros is null) for (; i < end; i++) run[i] = TL.Mul(run[i], s);
            else
            {
                TLane z = negZeros[at];
                for (; i < end; i++) run[i] = TL.Mul(TL.Add(run[i], z), s);
            }
        }
    }

    private readonly struct ScaledCall<TW>(ScaledWeight w, ProductView v, ProductEpilogue e, StoreTerms terms, byte[] x, byte[] codes, byte[] scales, byte[]? zeros, byte[] y, ProductTerms<byte[]> t) : IMixed
        where TW : struct, IAdad<TW>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (v.M == 0 || v.N == 0) return TL.Name;
            var (sl, zl) = ScaleLanes<TN, TLane, TL>(w, scales, zeros);
            var sums = new byte[v.ValuesC * Unsafe.SizeOf<TLane>()];
            var product = new ScaledProduct<TN, TS, TLane, TL>(w, v.M, x, codes, sl, zl, sums);
            TW.Elementwise(ref product);
            EpilogueStore<TN, TS, TLane, TL>(v, e, terms, t, sums, y);
            return TL.Name;
        }
    }

    // The product with TW's element-wise lane known: the packing's source the codes.
    private readonly struct ScaledProduct<TN, TS, TLane, TL>(ScaledWeight w, int m, byte[] x, byte[] codes, TLane[] scales, TLane[]? negZeros, byte[] sums) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW> =>
            MatMulFrom<TN, TS, TLane, TLane, TL, ScaledSource<TW, TLW, TLane, TL>>(x, new ScaledSource<TW, TLW, TLane, TL>(w, codes, scales, negZeros), sums, m, w.N, w.K,
                new Operands(TransB: true));
    }

    // Bᵀ's rows are the weight's: strips [ps0 + first, ps0 + last), each column's row of codes into the lane, its values
    // made there, laid at the strip's k values (the lane's own: one value a unit).
    private readonly struct ScaledSource<TW, TLW, TLane, TL>(ScaledWeight w, byte[] codes, TLane[] scales, TLane[]? negZeros) : IPanelSource<TLane>
        where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Pack(TLane[] packed, int n, int k, int strip, int kg, int ps0, int ps1, int first, int last)
        {
            var raw = ArrayPool<TLW>.Shared.Rent(Math.Max(1, k));
            var row = ArrayPool<TLane>.Shared.Rent(Math.Max(1, k));
            try
            {
                for (int s = ps0 + first; s < ps0 + last; s++)
                    for (int q = 0; q < strip; q++)
                    {
                        int j = s * strip + q;
                        long at = (long)(s - ps0) * kg * strip + q;
                        if (j >= n)
                        {
                            for (int p = 0; p < kg; p++) packed[at + (long)p * strip] = default;
                            continue;
                        }

                        TW.Load(codes, j * k, raw.AsSpan(0, k));
                        Widths.Load<TLW, TLane>(raw.AsSpan(0, k), row.AsSpan(0, k));
                        ScaleRun<TLane, TL>(w, scales, negZeros, j, 0, row.AsSpan(0, k));
                        for (int p = 0; p < k; p++) packed[at + (long)p * strip] = row[p];
                        for (int p = k; p < kg; p++) packed[at + (long)p * strip] = default;
                    }
            }
            finally
            {
                ArrayPool<TLW>.Shared.Return(raw);
                ArrayPool<TLane>.Shared.Return(row);
            }
        }
    }

    private readonly struct ExpandCall<TW>(ScaledWeight w, byte[] codes, byte[] scales, byte[]? zeros, byte[] output) : IMixed where TW : struct, IAdad<TW>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (w.Values == 0) return TL.Name;
            var (sl, zl) = ScaleLanes<TN, TLane, TL>(w, scales, zeros);
            var expand = new ExpandVisitor<TN, TLane, TL>(w, codes, sl, zl, output);
            TW.Elementwise(ref expand);
            return TL.Name;
        }
    }

    private readonly struct ExpandVisitor<TN, TLane, TL>(ScaledWeight w, byte[] codes, TLane[] scales, TLane[]? negZeros, byte[] output) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW>
        {
            int n = (int)w.Values, size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, (long)n * 4, new ExpandBody<TN, TW, TLW, TLane, TL>(w, codes, scales, negZeros, output));
        }
    }

    // Aligned tiles of the weight's values (a stored word or block of TN written by one task), each row's run scaled.
    private readonly struct ExpandBody<TN, TW, TLW, TLane, TL>(ScaledWeight w, byte[] codes, TLane[] scales, TLane[]? negZeros, byte[] output) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            Span<TLW> raw = stackalloc TLW[Tile];
            Span<TLane> v = stackalloc TLane[Tile];
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                TW.Load(codes, at, raw[..len]);
                Widths.Load<TLW, TLane>(raw[..len], v[..len]);
                for (int i = 0; i < len;)
                {
                    int r = (at + i) / w.K, c = (at + i) % w.K, run = Math.Min(len - i, w.K - c);
                    ScaleRun<TLane, TL>(w, scales, negZeros, r, c, v.Slice(i, run));
                    i += run;
                }

                TN.Store(v[..len], output, at);
            }
        }
    }
}
