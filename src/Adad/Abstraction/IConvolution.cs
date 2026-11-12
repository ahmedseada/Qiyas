// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Convolution (Conv.cs): y (N × OC × OH × OW) from N images x (N × C × H × W) and OC filters w (OC × C/Groups ×
/// Kh × Kw), the shape a ConvArgs.</summary>
public interface IConvolution<TBuffer>
{
    /// <summary>y = the epilogue's y of the convolution's sums (ProductEpilogue over ConvArgs.Output: N products of OC × OH·OW;
    /// the bias, t.Bias, a value an output channel; β·y, a second product over the same view, an activation or a gradient
    /// through one, a residual, the sums kept), every term in the lane and y rounded to the type once. One image of one group
    /// with no term (ConvArgs.Single) fused, no columns in memory; images, groups or any term the general path. Every type:
    /// a float type's sums in its float lane, any other type's in the exact lane for the bounds of every value the store
    /// makes (ProductEpilogue.ValidateType: its formulas the integer ones, β and the parameters integers).</summary>
    string Conv<TN>(ConvArgs a, ProductEpilogue e, TBuffer x, TBuffer w, TBuffer y, ProductTerms<TBuffer> t) where TN : struct, IAdad<TN>;

    /// <summary>dx, dw and db (OC values, or none) from dy, the gradient of the sums plus the bias (the activation's backward
    /// is the caller's: Map's formula backward), every type: dx's sums (w·dy) and dw's and db's (dy·x, dy over the images'
    /// positions) each in the lane the number gives for their bounds (a float type's float lane, an integer type's exact
    /// one), each gradient stored as the type once. Returns the lanes' names (dx's; dw's after it where it differs).</summary>
    string ConvBackward<TN>(ConvArgs a, TBuffer x, TBuffer w, TBuffer dy, TBuffer dx, TBuffer dw, TBuffer? db) where TN : struct, IAdad<TN>;
}
