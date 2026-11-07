// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Qiyas;

/// <summary>Row operations (Rows.cs): a formula over each row of rows × cols that needs the row's statistics, in the type's
/// float lane.</summary>
public interface IRows<TBuffer, TIndex>
{
    string Softmax<TN>(TBuffer x, TBuffer y, int rows, int cols) where TN : struct, IQiyasFloat<TN>;

    string RmsNorm<TN>(TBuffer x, TBuffer y, int rows, int cols, double eps) where TN : struct, IQiyasFloat<TN>;

    /// <summary>The row operation op (RowOps): x and its optional second input x2, gain g, bias bt, scale s and positions pos;
    /// y and the optional second output y2.</summary>
    string RowOp<TN>(RowOp op, TBuffer x, TBuffer? x2, TBuffer? g, TBuffer? bt, TBuffer? s, TIndex? pos, TBuffer y, TBuffer? y2, int rows, int cols, double a, double b)
        where TN : struct, IQiyasFloat<TN>;

    /// <summary>A normalization's parameter gradients dgamma and dbeta (cols each, in the float lane's storage).</summary>
    string NormParamsBwd<TN>(TBuffer x, TBuffer dy, TBuffer dgamma, TBuffer dbeta, int rows, int cols, double eps, bool centered) where TN : struct, IQiyasFloat<TN>;
}
