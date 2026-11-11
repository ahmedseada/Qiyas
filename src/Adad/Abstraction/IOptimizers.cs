// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Optimizer steps (Optimizers.cs): the parameters p and gradients g as the type, the moments m (and v) in the
/// type's float lane storage, updated in place.</summary>
public interface IOptimizers<TBuffer>
{
    /// <summary>A step over n values (sgd keeps no moment: m is not read).</summary>
    string Optimizer<TN>(OptimizerArgs a, TBuffer p, TBuffer g, TBuffer? m, TBuffer? v, int n) where TN : struct, IFloatAdad<TN>;

    /// <summary>A step across several tensors under one clipping (a.MaxNorm; 0: none): the squares of all their gradients
    /// summed as one whole sum by the device's rule for the type (the tensors' parts joined in one chain, the sum a lane
    /// value), the factor min(1, MaxNorm / the sum's root) made from it in the lane where it is (the map's clip_factor: no
    /// value crosses to the host), and each tensor's step with every gradient times the factor as it is read. sumSquares,
    /// when given: the sum (lane storage, one value), for the caller's own use.</summary>
    string OptimizerMany<TN>(OptimizerArgs a, ReadOnlySpan<OptimizerTensor<TBuffer>> tensors, TBuffer? sumSquares) where TN : struct, IFloatAdad<TN>;
}
