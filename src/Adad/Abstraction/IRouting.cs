// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Routing, sampling and losses (Routing.cs): operations whose results are indices, or whose inputs are.</summary>
public interface IRouting<TBuffer, TIndex>
{
    /// <summary>The k first of each row (rows × cols) in Adad's order (Routing.Precedes: larger first, a NaN above every
    /// number, an equal value by the lower index): values (rows × k) and their columns; k is at most cols
    /// (Routing.ValidateTopK: refused past it).</summary>
    string TopK<TN>(TBuffer x, int rows, int cols, int k, TBuffer values, TIndex indices) where TN : struct, IAdad<TN>;

    /// <summary>The k first of each row of x (rows × cols) as log-probabilities (a beam's candidates, read instead of the
    /// rows): indices as TopK's (Adad's order; k at most cols); values[r·k + j] = (x[r][index] − m) − log Σ_c exp(x[r][c] − m),
    /// m the row's first-ranked value, in the float lane, rounded to TN once.</summary>
    string LogSoftmaxTopK<TN>(TBuffer x, int rows, int cols, int k, TBuffer values, TIndex indices) where TN : struct, IFloatAdad<TN>;

    /// <summary>output[r] = x[r, index[r]]; a column outside the row (below 0, or cols and past) reads nothing: 0.</summary>
    string Pick<TN>(TBuffer x, int rows, int cols, TIndex index, TBuffer output) where TN : struct, IAdad<TN>;

    /// <summary>picks[r] = the first column whose running sum of row r reaches u[r] · Σ row.</summary>
    string Sample<TN>(TBuffer x, int rows, int cols, TBuffer u, TIndex picks) where TN : struct, IFloatAdad<TN>;

    /// <summary>output[r] = log Σ exp(x[r]) − x[r, targets[r]].</summary>
    /// <summary>loss[r] = w[r]·(log Σ exp(x[r]) − x[r][t[r]]): weights (one a row, stored as the type) absent, 1. The gradient
    /// is a row operation: cross_entropy_bwd, weighted_cross_entropy_bwd (the same weights, and the loss's scale).</summary>
    string CrossEntropy<TN>(TBuffer x, int rows, int cols, TIndex targets, TBuffer? weights, TBuffer output) where TN : struct, IFloatAdad<TN>;

    /// <summary>counts (buckets), offsets (buckets + 1) and order (count: the sources grouped by bucket, in source order) of
    /// count indices into buckets.</summary>
    void Route(TIndex index, int count, int buckets, TIndex counts, TIndex offsets, TIndex order);

}
