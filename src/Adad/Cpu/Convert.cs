// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Convert on the CPU: each value of TFrom read into its element-wise lane (exact), taken into TTo's element-wise lane with
// one rounding as Fmt.ConvertPlan describes it — between integers held to TTo's range exactly (through 128 bits), a value
// read wider than float32 going to a narrower float first rounded to odd at float32 — and stored as TTo.
using System.Numerics;
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    public static string Convert<TFrom, TTo>(byte[] x, byte[] output, int n) where TFrom : struct, IAdad<TFrom> where TTo : struct, IAdad<TTo>
    {
        var v = new ConvertFrom<TTo>(x, output, n);
        TFrom.Elementwise(ref v);
        return v.Lane;
    }

    private struct ConvertFrom<TTo>(byte[] x, byte[] output, int n) : ILaneVisitor where TTo : struct, IAdad<TTo>
    {
        public string Lane = "";

        public void Run<TF, TLF, TLFL>() where TF : struct, IAdad<TF, TLF> where TLF : unmanaged where TLFL : struct, ILane<TLFL, TLF>
        {
            var to = new ConvertTo<TF, TLF>(x, output, n);
            TTo.Elementwise(ref to);
            Lane = $"{TLFL.Name} → {to.Lane}";
        }
    }

    private struct ConvertTo<TF, TLF>(byte[] x, byte[] output, int n) : ILaneVisitor where TF : struct, IAdad<TF, TLF> where TLF : unmanaged
    {
        public string Lane = "";

        public void Run<TT, TLT, TLTL>() where TT : struct, IAdad<TT, TLT> where TLT : unmanaged where TLTL : struct, ILane<TLTL, TLT>
        {
            bool toOdd = Fmt.ConvertPlan(Fmt.Of<TF>(), Fmt.Of<TT>()).ToOdd;
            Lane = TLTL.Name + (toOdd ? " (to odd at f32 first)" : "");
            int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Int128 lo = TT.Min < 0 ? -(Int128.One << (TT.Bits - 1)) : Int128.Zero;
            Int128 hi = TT.Min < 0 ? (Int128.One << (TT.Bits - 1)) - 1 : (Int128.One << TT.Bits) - 1;
            Team.For(n, size, (long)n * 2, new ConvertBody<TF, TLF, TT, TLT>(x, output, toOdd, lo, hi));
        }
    }

    private readonly struct ConvertBody<TF, TLF, TT, TLT>(byte[] x, byte[] output, bool toOdd, Int128 lo, Int128 hi) : Team.IBody
        where TF : struct, IAdad<TF, TLF> where TLF : unmanaged where TT : struct, IAdad<TT, TLT> where TLT : unmanaged
    {
        public void Run(int first, int last)
        {
            Span<TLF> s = stackalloc TLF[Tile];
            Span<TLT> d = stackalloc TLT[Tile];
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                TF.Load(x, at, s[..len]);
                Lanes(s[..len], d[..len]);
                TT.Store(d[..len], output, at);
            }
        }

        private void Lanes(ReadOnlySpan<TLF> s, Span<TLT> d)
        {
            if (Integer<TLF>() && Integer<TLT>())
            {
                for (int i = 0; i < s.Length; i++) d[i] = FromWide<TLT>(Int128.Clamp(Wide(s[i]), lo, hi));
                return;
            }

            if (toOdd)
            {
                // TTo is a float narrower than float32: its lane is float32.
                var f = System.Runtime.InteropServices.MemoryMarshal.Cast<TLT, float>(d);
                for (int i = 0; i < s.Length; i++) f[i] = typeof(TLF) == typeof(double) ? ToOdd(Unsafe.As<TLF, double>(ref Unsafe.AsRef(in s[i]))) : ToOdd(Wide(s[i]));
                return;
            }

            if (typeof(TLT) == typeof(float) || typeof(TLT) == typeof(double)) Widths.Load(s, d);
            else Widths.Store(s, d);
        }
    }

    private static bool Integer<T>() => typeof(T) != typeof(float) && typeof(T) != typeof(double);

    private static Int128 Wide<T>(T v) where T : unmanaged
    {
        if (typeof(T) == typeof(sbyte)) return Unsafe.As<T, sbyte>(ref v);
        if (typeof(T) == typeof(byte)) return Unsafe.As<T, byte>(ref v);
        if (typeof(T) == typeof(short)) return Unsafe.As<T, short>(ref v);
        if (typeof(T) == typeof(ushort)) return Unsafe.As<T, ushort>(ref v);
        if (typeof(T) == typeof(int)) return Unsafe.As<T, int>(ref v);
        if (typeof(T) == typeof(uint)) return Unsafe.As<T, uint>(ref v);
        if (typeof(T) == typeof(long)) return Unsafe.As<T, long>(ref v);
        if (typeof(T) == typeof(ulong)) return Unsafe.As<T, ulong>(ref v);
        throw new NotSupportedException(typeof(T).Name);
    }

    private static T FromWide<T>(Int128 v) where T : unmanaged
    {
        if (typeof(T) == typeof(sbyte)) { var r = (sbyte)v; return Unsafe.As<sbyte, T>(ref r); }
        if (typeof(T) == typeof(byte)) { var r = (byte)v; return Unsafe.As<byte, T>(ref r); }
        if (typeof(T) == typeof(short)) { var r = (short)v; return Unsafe.As<short, T>(ref r); }
        if (typeof(T) == typeof(ushort)) { var r = (ushort)v; return Unsafe.As<ushort, T>(ref r); }
        if (typeof(T) == typeof(int)) { var r = (int)v; return Unsafe.As<int, T>(ref r); }
        if (typeof(T) == typeof(uint)) { var r = (uint)v; return Unsafe.As<uint, T>(ref r); }
        if (typeof(T) == typeof(long)) { var r = (long)v; return Unsafe.As<long, T>(ref r); }
        if (typeof(T) == typeof(ulong)) { var r = (ulong)v; return Unsafe.As<ulong, T>(ref r); }
        throw new NotSupportedException(typeof(T).Name);
    }

    /// <summary>A float64 rounded to odd at float32: toward zero, the last bit set when anything was dropped (the largest
    /// finite past the range; NaN as the hardware narrows it).</summary>
    public static float ToOdd(double v)
    {
        float f = (float)v;
        if (double.IsNaN(v) || f == v) return f;
        uint bits = BitConverter.SingleToUInt32Bits(f);
        if (Math.Abs((double)f) > Math.Abs(v)) bits--;                            // the neighbour toward zero (∞ to the largest finite)
        return BitConverter.UInt32BitsToSingle(bits | 1);
    }

    /// <summary>An integer rounded to odd at float32: its top 24 bits, the last set when a lower bit was.</summary>
    public static float ToOdd(Int128 v)
    {
        bool negative = v < 0;
        var m = (UInt128)(negative ? -v : v);
        if (m < (1u << 24)) return negative ? -(float)(uint)m : (float)(uint)m;
        int shift = 127 - (int)UInt128.LeadingZeroCount(m) - 23;
        var t = m >> shift;
        if ((m & ((UInt128.One << shift) - 1)) != 0) t |= 1;
        float r = MathF.ScaleB((float)(uint)t, shift);                            // 24 bits: exact
        return negative ? -r : r;
    }
}
