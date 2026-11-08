// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Products (Kernels.cs, Products.cs, MatMulBlocks.cs): the sum lane chosen by the bounds of the operands' values.</summary>
public interface IProducts<TBuffer>
{
    /// <summary>C (m × n) = A (m × k) · B (k × n), the bounds measured on the operands.</summary>
    string MatMul<TN>(TBuffer a, TBuffer b, TBuffer c, int m, int n, int k) where TN : struct, IAdad<TN>;

    /// <summary>C = op(A)·op(B) over the view's batches.</summary>
    string MatMulView<TN>(ProductView v, TBuffer a, TBuffer b, TBuffer c) where TN : struct, IAdad<TN>;

    /// <summary>C = the epilogue's y of op(A)·op(B) (ProductEpilogue: + U·op(V) + bias + β·C, then the activation or a
    /// gradient through one, then + residual), every term in the float lane and C rounded to the type once.</summary>
    string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, TBuffer a, TBuffer b, TBuffer c, ProductTerms<TBuffer> t) where TN : struct, IFloatAdad<TN>;

    /// <summary>dA = dC·Bᵀ (m × k) and dB = Aᵀ·dC (k × n) for C = A·B.</summary>
    string MatMulBackward<TN>(TBuffer a, TBuffer b, TBuffer dc, TBuffer da, TBuffer db, int m, int n, int k) where TN : struct, IAdad<TN>;

    /// <summary>C = A·Bᵀ by the block-scaled product of two Q8_0 / Q4_0 operands (k in whole blocks; BlockProductAvailable).</summary>
    string MatMulViewBlocks<TN>(ProductView v, TBuffer a, TBuffer b, TBuffer c) where TN : struct, IAdad<TN>;

    /// <summary>Whether this device has the block-scaled product.</summary>
    bool BlockProductAvailable { get; }
}
