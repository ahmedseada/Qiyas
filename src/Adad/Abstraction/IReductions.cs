// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Reductions, transposes and matrix-vector products (Reduce.cs).</summary>
public interface IReductions<TBuffer>
{
    /// <summary>op (sum, sumsq, max, min, argmax) along the axis of x (rows × cols) into output, as the lane's values (argmax:
    /// s32 positions). max, min and argmax in Adad's order (Routing.Order: a NaN above every number, −0 equal to +0): max
    /// the first-ranked value (a NaN when there is one), min the last-ranked (a NaN only when every value is one), argmax
    /// the first position of the first-ranked.</summary>
    string Reduce<TN>(string op, Axis axis, TBuffer x, int rows, int cols, TBuffer output) where TN : struct, IAdad<TN>;

    /// <summary>output (cols × rows) = xᵀ.</summary>
    string Transpose<TN>(TBuffer x, TBuffer output, int rows, int cols) where TN : struct, IAdad<TN>;

    /// <summary>y = A·x (or Aᵀ·x when transposed) for A rows × cols.</summary>
    string Gemv<TN>(TBuffer a, TBuffer x, TBuffer y, int rows, int cols, bool transposed) where TN : struct, IAdad<TN>;
}
