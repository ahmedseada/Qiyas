// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Routing, sampling and losses (Routing.cs): operations whose results are indices, or whose inputs are.</summary>
public interface IRouting<TBuffer, TIndex>
{
    /// <summary>The k largest of each row (rows × cols): values (rows × k) and their columns, largest first.</summary>
    string TopK<TN>(TBuffer x, int rows, int cols, int k, TBuffer values, TIndex indices) where TN : struct, IAdad<TN>;

    /// <summary>output[r] = x[r, index[r]].</summary>
    string Pick<TN>(TBuffer x, int rows, int cols, TIndex index, TBuffer output) where TN : struct, IAdad<TN>;

    /// <summary>picks[r] = the first column whose running sum of row r reaches u[r] · Σ row.</summary>
    string Sample<TN>(TBuffer x, int rows, int cols, TBuffer u, TIndex picks) where TN : struct, IFloatAdad<TN>;

    /// <summary>output[r] = log Σ exp(x[r]) − x[r, targets[r]].</summary>
    string CrossEntropy<TN>(TBuffer x, int rows, int cols, TIndex targets, TBuffer output) where TN : struct, IFloatAdad<TN>;

    /// <summary>counts (buckets), offsets (buckets + 1) and order (count: the sources grouped by bucket, in source order) of
    /// count indices into buckets.</summary>
    void Route(TIndex index, int count, int buckets, TIndex counts, TIndex offsets, TIndex order);
}
