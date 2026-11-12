// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Convolution (Conv.cs): y (N × OC × OH × OW) from N images x (N × C × H × W) and OC filters w (OC × C/Groups ×
/// Kh × Kw), the shape a ConvArgs.</summary>
public interface IConvolution<TBuffer>
{
    /// <summary>y = act(conv(x, w) + bias) (bias: OC values, a filter's each, or none): one image of one group without a bias
    /// or an activation (ConvArgs.Single) fused, no columns in memory, every type; images, groups, a bias and an activation
    /// the float types, the sums in the float lane and y stored once.</summary>
    string Conv<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer? bias, TBuffer y) where TN : struct, IAdad<TN>;

    /// <summary>dx, dw and db (OC values, or none) from dy, the gradient of the sums plus the bias (the activation's backward
    /// is the caller's: Map's formula backward), every type: dx's sums (w·dy) and dw's and db's (dy·x, dy over the images'
    /// positions) each in the lane the number gives for their bounds (a float type's float lane, an integer type's exact
    /// one), each gradient stored as the type once. Returns the lanes' names (dx's; dw's after it where it differs).</summary>
    string ConvBackward<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer dy, TBuffer dx, TBuffer dw, TBuffer? db) where TN : struct, IAdad<TN>;
}
