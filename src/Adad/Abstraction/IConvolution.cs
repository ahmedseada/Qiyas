// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Convolution (Conv.cs): N images of C × H × W by the weights OC × (C / groups)·kh·kw.</summary>
public interface IConvolution<TBuffer>
{
    /// <summary>y = act(conv(x, w) + bias) (bias: OC values, or none): one image of one group without a bias (ConvArgs.Single)
    /// fused, no columns in memory, every type; images, groups, the bottom and right pads, a bias and an activation the
    /// float types, the sums in the float lane and y stored once.</summary>
    string Conv<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer? bias, TBuffer y) where TN : struct, IAdad<TN>;

    /// <summary>y = w · im2col(x), by the composition (one image of one group).</summary>
    string ConvComposed<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer y) where TN : struct, IAdad<TN>;

    /// <summary>dx, dw and db (OC values, or none) from dy, the gradient of the sums plus the bias (the activation's backward
    /// is the caller's: Map's formula backward).</summary>
    string ConvBackward<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer dy, TBuffer dx, TBuffer dw, TBuffer? db) where TN : struct, IFloatAdad<TN>;
}
