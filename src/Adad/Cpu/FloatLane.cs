// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The float lane of any type on the CPU (Fmt.FloatLane), for the operations whose formula is a float one over values of
// any type (penalties, sampling, resampling): a float type's values read straight into its float lane; an integer type's
// read in its natural lane, exactly, taken into float32 or float64 by Fmt.FloatLane's rule where the formula needs them,
// and a result taken back by the type's own conversion (Widths: rounded to nearest, ties to even, held to its range).
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>A call over TN's values in its natural lane TE (a float type's: its float lane) and its float lane TLane.</summary>
    internal interface IFloatOf
    {
        string Run<TN, TE, TLane, TL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>;
    }

    /// <summary>Runs <paramref name="call"/> with TN's natural and float lanes; returns what it returns (the lane's name).</summary>
    internal static string InFloatLane<TN, TC>(TC call) where TN : struct, IAdad<TN> where TC : struct, IFloatOf
    {
        var f = new FloatOfFloat<TC>(call);
        if (TN.TryFloat(ref f)) return f.Lane;
        var i = new FloatOfInteger<TC>(call);
        TN.Elementwise(ref i);
        return i.Lane;
    }

    private struct FloatOfFloat<TC>(TC call) : IFloatLaneVisitor where TC : struct, IFloatOf
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            Lane = call.Run<TN, TLane, TLane, TL>();
    }

    private struct FloatOfInteger<TC>(TC call) : ILaneVisitor where TC : struct, IFloatOf
    {
        public string Lane = "";

        public void Run<TN, TE, TEL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TEL : struct, ILane<TEL, TE> =>
            Lane = Fmt.Of<TN>().FloatLane == Adad.Abstraction.Lane.F64 ? call.Run<TN, TE, double, F64>() : call.Run<TN, TE, float, F32>();
    }

    /// <summary>A natural lane value in the float lane (a float type's: itself).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TLane ToFloatLane<TE, TLane>(TE v) where TE : unmanaged where TLane : unmanaged
    {
        if (typeof(TE) == typeof(TLane)) return Unsafe.As<TE, TLane>(ref v);
        TLane r = default;
        Widths.Load(new ReadOnlySpan<TE>(in v), new Span<TLane>(ref r));
        return r;
    }

    /// <summary>A float lane value in the natural lane by the type's conversion (a float type's: itself).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TE FromFloatLane<TLane, TE>(TLane v) where TLane : unmanaged where TE : unmanaged
    {
        if (typeof(TE) == typeof(TLane)) return Unsafe.As<TLane, TE>(ref v);
        TE r = default;
        Widths.Store(new ReadOnlySpan<TLane>(in v), new Span<TE>(ref r));
        return r;
    }
}
