// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Row operations (Rows.cs): a formula over each row of rows × cols that needs the row's statistics, in the type's
/// float lane.</summary>
public interface IRows<TBuffer, TIndex>
{
    string Softmax<TN>(TBuffer x, TBuffer y, int rows, int cols) where TN : struct, IFloatAdad<TN>;

    string RmsNorm<TN>(TBuffer x, TBuffer y, int rows, int cols, double eps) where TN : struct, IFloatAdad<TN>;

    /// <summary>The row operation op (RowOps): x and its optional second input x2, gain g, bias bt, scale s and positions pos;
    /// y and the optional second output y2.</summary>
    string RowOp<TN>(RowOp op, TBuffer x, TBuffer? x2, TBuffer? g, TBuffer? bt, TBuffer? s, TIndex? pos, TBuffer y, TBuffer? y2, int rows, int cols, double a, double b)
        where TN : struct, IFloatAdad<TN>;

    /// <summary>A normalization's parameter gradients dgamma and dbeta (cols each, in the float lane's storage).</summary>
    string NormParamsBwd<TN>(TBuffer x, TBuffer dy, TBuffer dgamma, TBuffer dbeta, int rows, int cols, double eps, bool centered) where TN : struct, IFloatAdad<TN>;

    /// <summary>Group normalization over shape (GroupShape: batch, group, instance normalization): y = (x − μ)·rstd·gamma[c] +
    /// beta[c], μ and the biased variance σ² of each group's values, rstd = 1/√(σ² + eps). μ and σ² are written to mean and
    /// variance (each when given; the float lane's storage, a value a group) or, with given, read from them (batch
    /// normalization in inference, by the running statistics). gamma and beta absent: 1 and 0.</summary>
    string GroupNorm<TN>(TBuffer x, TBuffer? gamma, TBuffer? beta, TBuffer? mean, TBuffer? variance, TBuffer y, GroupShape shape, double eps, bool given)
        where TN : struct, IFloatAdad<TN>;

    /// <summary>GroupNorm's gradients from dy, the statistics recomputed from x (given: read from mean and variance), x̂ = (x − μ)·rstd:
    /// dx = rstd·(γ·dy − mean(γ·dy) − x̂·mean(γ·dy·x̂)) over each group (given: rstd·γ·dy); dgamma[c] = Σ dy·x̂ and dbeta[c] = Σ dy
    /// over channel c's values (the float lane's storage). Each output when given.</summary>
    string GroupNormBwd<TN>(TBuffer x, TBuffer dy, TBuffer? gamma, TBuffer? mean, TBuffer? variance, TBuffer? dx, TBuffer? dgamma, TBuffer? dbeta, GroupShape shape, double eps, bool given)
        where TN : struct, IFloatAdad<TN>;

    /// <summary>The rotary embedding by tables (RopeArgs) of x into y (not x itself): positions per step, cos and sin in the float
    /// lane's storage; with a gain, each row RMS-normalized first (the queries' and keys' norm and their rotation in one pass).</summary>
    string Rope<TN>(RopeArgs a, TBuffer x, TBuffer? gain, TBuffer cos, TBuffer sin, TIndex positions, TBuffer y) where TN : struct, IFloatAdad<TN>;
}
