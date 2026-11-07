// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad;

/// <summary>Optimizer steps (Optimizers.cs): the parameters p and gradients g as the type, the moments m (and v) in the
/// type's float lane storage, updated in place.</summary>
public interface IOptimizers<TBuffer>
{
    string Optimizer<TN>(OptimizerArgs a, TBuffer p, TBuffer g, TBuffer m, TBuffer? v, int n) where TN : struct, IFloatAdad<TN>;
}
