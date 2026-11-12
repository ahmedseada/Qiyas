// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A scaled weight's products on the CPU (Products.cs ScaledWeight): y = x · Wᵀ with W's values made in the float lane as
// the product's packing reads the codes (ScaledSource: each output's row of codes read once into TW's element-wise lane,
// taken into the float lane, the zero subtracted and the scale multiplied there, laid into the packed strips): no W in
// memory but its codes and the packed panel the product holds of any operand; the sums in the lane, then the epilogue's
// store (Epilogue.cs). ExpandScaled makes the same values a tile at a time and stores them as TN. Quantize is its inverse:
// each block's range and scale in the lane (stored as TN, read back), the zero points from the stored scales, then each
// code made in the lane and rounded to TW once (Convert's rule: an f64 lane's value to a narrow float rounded to odd at
// float32 first).

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

    /// <summary>The scaled weight's codes, scales and zero points from its values (IProducts.Quantize); returns the lane's name.</summary>
    public static string Quantize<TN, TW>(ScaledWeight w, byte[] values, byte[] codes, byte[] scales, byte[]? zeros) where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        ScaledWeight.ValidateCodes(Fmt.Of<TW>(), zeros is not null);
        return Mixed<TN, QuantizeCall<TW>>(new QuantizeCall<TW>(w, values, codes, scales, zeros));
    }

    private readonly struct QuantizeCall<TW>(ScaledWeight w, byte[] values, byte[] codes, byte[] scales, byte[]? zeros) : IMixed where TW : struct, IAdad<TW>
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (w.Values == 0) return TL.Name;
            TLane min = TL.From(TW.Min), max = TL.From(TW.Max);
            // Each block's range and scale, the scales stored as TN once and read back as stored.
            var s = new TLane[w.Scales];
            var lows = new TLane[w.Scales];
            Team.For((int)w.Scales, 1, w.Values * 2, new QuantizeRangeBody<TN, TLane, TL>(w, values, zeros is not null, min, max, s, lows));
            TN.Store(s, scales, 0);
            TN.Load(scales, 0, s);
            TLane[]? z = null;
            if (zeros is not null)
            {
                // The zero points from the stored scales: round(min − lo / scale) held to TW's range (0 where the scale is 0).
                z = new TLane[w.Scales];
                for (int i = 0; i < z.Length; i++)
                    z[i] = Widths.Compare(s[i], TL.Zero) > 0 ? Held<TLane, TL>(TL.Round(new Vector<TLane>(TL.Sub(min, TL.Div(lows[i], s[i]))))[0], min, max) : TL.Zero;
                TN.Store(z, zeros, 0);
                TN.Load(zeros, 0, z);
            }

            var codesVisitor = new QuantizeVisitor<TN, TS, TLane, TL>(w, values, s, z, min, max, codes);
            TW.Elementwise(ref codesVisitor);
            return TL.Name;
        }
    }

    // v held to [min, max].
    private static TLane Held<TLane, TL>(TLane v, TLane min, TLane max) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
        Widths.Compare(v, min) < 0 ? min : Widths.Compare(v, max) > 0 ? max : v;

    // Each block's range with 0, lo = min(x, 0) and hi = max(x, 0), over its rows' runs read into the lane; its scale
    // max(hi, −lo) / max, or (hi − lo) / (max − min) with zero points.
    private readonly struct QuantizeRangeBody<TN, TLane, TL>(ScaledWeight w, byte[] values, bool zeros, TLane min, TLane max, TLane[] scales, TLane[] lows) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var row = ArrayPool<TLane>.Shared.Rent(Math.Max(1, Math.Min(w.K, w.BlockColumns)));
            try
            {
                for (int si = first; si < last; si++)
                {
                    var (r0, r1, c0, c1) = w.Block(si);
                    int len = c1 - c0, lanes = Vector<TLane>.Count;
                    Vector<TLane> vlo = Vector<TLane>.Zero, vhi = Vector<TLane>.Zero;
                    TLane lo = TL.Zero, hi = TL.Zero;
                    for (int r = r0; r < r1 && len > 0; r++)
                    {
                        TN.Load(values, (int)((long)r * w.K + c0), row.AsSpan(0, len));
                        int c = 0;
                        for (; c + lanes <= len; c += lanes) { var v = new Vector<TLane>(row.AsSpan(c)); vlo = Vector.Min(vlo, v); vhi = Vector.Max(vhi, v); }
                        for (; c < len; c++) { if (Widths.Compare(row[c], lo) < 0) lo = row[c]; if (Widths.Compare(row[c], hi) > 0) hi = row[c]; }
                    }

                    for (int l = 0; l < lanes; l++) { if (Widths.Compare(vlo[l], lo) < 0) lo = vlo[l]; if (Widths.Compare(vhi[l], hi) > 0) hi = vhi[l]; }
                    lows[si] = lo;
                    scales[si] = zeros ? TL.Div(TL.Sub(hi, lo), TL.Sub(max, min)) : TL.Div(TL.Max(hi, TL.Sub(TL.Zero, lo)), max);
                }
            }
            finally { ArrayPool<TLane>.Shared.Return(row); }
        }
    }

    private readonly struct QuantizeVisitor<TN, TS, TLane, TL>(ScaledWeight w, byte[] values, TLane[] scales, TLane[]? zeros, TLane min, TLane max, byte[] codes) : ILaneVisitor
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run<TW, TLW, TLWL>() where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLWL : struct, ILane<TLWL, TLW>
        {
            int n = (int)w.Values, size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            bool toOdd = Fmt.ConvertPlan(Fmt.Of<TS>(), Fmt.Of<TW>()).ToOdd;
            Team.For(n, size, (long)n * 4, new QuantizeBody<TN, TW, TLW, TLane, TL>(w, values, scales, zeros, min, max, codes, toOdd));
        }
    }

    // Aligned tiles of the codes (a stored word of TW written by one task): x / scale (+ the zero point) in the lane, held
    // to TW's range, rounded to TW once.
    private readonly struct QuantizeBody<TN, TW, TLW, TLane, TL>(ScaledWeight w, byte[] values, TLane[] scales, TLane[]? zeros, TLane min, TLane max, byte[] codes, bool toOdd) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TW : struct, IAdad<TW, TLW> where TLW : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            Span<TLane> v = stackalloc TLane[Tile];
            Span<TLW> raw = stackalloc TLW[Tile];
            bool integer = !TW.IsFloat;
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                TN.Load(values, at, v[..len]);
                for (int i = 0; i < len; i++)
                {
                    int r = (at + i) / w.K, c = (at + i) % w.K, si = w.ScaleAt(r, c);
                    TLane s = scales[si], q = Widths.Compare(s, TL.Zero) > 0 ? TL.Div(v[i], s) : TL.Zero;
                    if (integer) q = TL.Round(new Vector<TLane>(q))[0];
                    if (zeros is not null) q = TL.Add(q, zeros[si]);
                    v[i] = Held<TLane, TL>(q, min, max);
                }

                if (toOdd)
                {
                    // TW a float narrower than float32 from the f64 lane: its lane float32, the value rounded to odd there first.
                    var f = MemoryMarshal.Cast<TLW, float>(raw);
                    var d = MemoryMarshal.Cast<TLane, double>(v);
                    for (int i = 0; i < len; i++) f[i] = ToOdd(d[i]);
                }
                else if (typeof(TLW) == typeof(float) || typeof(TLW) == typeof(double)) Widths.Load<TLane, TLW>(v[..len], raw[..len]);
                else Widths.Store<TLane, TLW>(v[..len], raw[..len]);
                TW.Store(raw[..len], codes, at);
            }
        }
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
            EpilogueStore<TN, TS, TLane, TL>(v, e, terms, t, sums, y, 1, v.N);
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
