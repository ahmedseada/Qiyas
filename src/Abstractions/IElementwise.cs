// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Qiyas;

/// <summary>Element-wise operations (Map.cs): one formula over every value, in the type's element-wise lane, or its float lane
/// for the float formulas.</summary>
public interface IElementwise<TBuffer>
{
    /// <summary>c = a + b over n values.</summary>
    void Add<TN>(TBuffer a, TBuffer b, TBuffer c, int n) where TN : struct, IQiyasNumber<TN>;

    /// <summary>output = op(x, y, z; a, b) over n values, for the formulas every type runs (its element-wise lane).</summary>
    string Map<TN>(MapOp op, TBuffer? x, TBuffer? y, TBuffer? z, TBuffer output, int n, double a, double b) where TN : struct, IQiyasNumber<TN>;

    /// <summary>Map for a float type: every formula, the float formulas in its float lane.</summary>
    string MapFloat<TN>(MapOp op, TBuffer? x, TBuffer? y, TBuffer? z, TBuffer output, int n, double a, double b) where TN : struct, IQiyasFloat<TN>;

    /// <summary>output[r, c] = op(x[r, c], v[c]): a row vector broadcast over rows × cols.</summary>
    string MapRow<TN>(MapOp op, TBuffer x, TBuffer v, TBuffer output, int rows, int cols) where TN : struct, IQiyasNumber<TN>;
}
