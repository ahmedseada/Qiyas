// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Element-wise operations (Map.cs): one formula over every value, in the type's element-wise lane, or its float lane
/// for the float formulas.</summary>
public interface IElementwise<TBuffer>
{
    /// <summary>c = a + b over n values.</summary>
    void Add<TN>(TBuffer a, TBuffer b, TBuffer c, int n) where TN : struct, IAdad<TN>;

    /// <summary>output = op(x, y, z; a, b) over n values, for the formulas every type runs (its element-wise lane).</summary>
    string Map<TN>(MapOp op, TBuffer? x, TBuffer? y, TBuffer? z, TBuffer output, int n, double a, double b) where TN : struct, IAdad<TN>;

    /// <summary>Map for a float type: every formula, the float formulas in its float lane.</summary>
    string MapFloat<TN>(MapOp op, TBuffer? x, TBuffer? y, TBuffer? z, TBuffer output, int n, double a, double b) where TN : struct, IFloatAdad<TN>;

    /// <summary>output[r, c] = op(x[r, c], v[c]): a row vector broadcast over rows × cols.</summary>
    string MapRow<TN>(MapOp op, TBuffer x, TBuffer v, TBuffer output, int rows, int cols) where TN : struct, IAdad<TN>;

    /// <summary>MapRow for a float type: every formula, the float formulas in its float lane (a row of biases and its gelu).</summary>
    string MapRowFloat<TN>(MapOp op, TBuffer x, TBuffer v, TBuffer output, int rows, int cols) where TN : struct, IFloatAdad<TN>;

    /// <summary>Dropout: output[i] = r[i] + x[i]·s where the seed's draw i keeps value i (Draws.Keep: its bits at least
    /// Draws.Threshold(p)), r[i] where it does not, s = 1/(1 − p) and r a residual added in the same pass (absent: 0), over n
    /// values in the type's float lane (x·s + r one fused multiply-add, s read into the lane). Its backward is the same
    /// operation on the gradient with the same seed.</summary>
    string Dropout<TN>(TBuffer x, TBuffer? residual, TBuffer output, int n, double p, ulong seed) where TN : struct, IFloatAdad<TN>;

    /// <summary>output (TTo) = x (TFrom) over n values, each value rounded once into TTo (to nearest, ties to even) and held to
    /// its range as TTo's stores hold (Fmt.ConvertPlan: the lane it is read in, exact); a block format encodes from the
    /// values as float32, as its definition does.</summary>
    string Convert<TFrom, TTo>(TBuffer x, TBuffer output, int n) where TFrom : struct, IAdad<TFrom> where TTo : struct, IAdad<TTo>;
}
