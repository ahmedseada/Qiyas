// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Layout and indexing (Layout.cs): data moved by index arithmetic, exact in every type.</summary>
public interface ILayouts<TBuffer, TIndex>
{
    /// <summary>The layout operation k.Name over k's shape (k.P, k.Count, k.InputCount): x and the optional second input x2
    /// (k.SecondCount values) and index buffer into output.</summary>
    string Layout<TN>(LayoutArgs k, TBuffer x, TIndex? index, TBuffer output, TBuffer? x2) where TN : struct, IAdad<TN>;

    /// <summary>Resampling by a separable filter (Resample.cs; the plan from Resampling.Plan): x (a.InputCount values of TIn,
    /// plane p channel p % Channels, read where they are stored) through the plan's bounds (int32) and weights (in TN's float
    /// lane, Fmt.FloatLane) into y (a.Count values of TN, every type); with values, a scale and a shift a channel in the
    /// lane, y = v·scale + shift (one fma), y rounded to TN once by its own conversion (an integer's or bool's: to nearest,
    /// ties to even, held to its range).</summary>
    string Resample<TIn, TN>(ResampleArgs a, TBuffer x, TIndex bounds, TBuffer weights, TBuffer? values, TBuffer y) where TIn : struct, IAdad<TIn> where TN : struct, IAdad<TN>;
}
