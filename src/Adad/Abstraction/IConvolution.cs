// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Convolution (Conv.cs) of one image: C × H × W by the weights OC × C·kh·kw.</summary>
public interface IConvolution<TBuffer>
{
    /// <summary>y = conv(x, w), fused (no columns in memory).</summary>
    string Conv<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer y) where TN : struct, IAdad<TN>;

    /// <summary>y = w · im2col(x), by the composition.</summary>
    string ConvComposed<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer y) where TN : struct, IAdad<TN>;

    /// <summary>dx and dw from dy.</summary>
    string ConvBackward<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer dy, TBuffer dx, TBuffer dw) where TN : struct, IFloatAdad<TN>;
}
