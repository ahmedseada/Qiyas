// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>The connectionist temporal classification loss (Ctc.cs).</summary>
public interface ICtc<TBuffer>
{
    /// <summary>The loss per sequence (B values) from logits x (T × B × C, frame-major), and the gradient with respect to x when grad is given.</summary>
    string Ctc<TN>(CtcArgs a, TBuffer x, TBuffer loss, TBuffer? grad) where TN : struct, IFloatAdad<TN>;
}
