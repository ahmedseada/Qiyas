// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad;

/// <summary>Attention (Attention.cs): O = softmax(scale · Q·Kᵀ + mask) · V per head, in the type's float lane.</summary>
public interface IAttention<TBuffer>
{
    /// <summary>O for rows of queries: the fused rows or the composition, whichever the device measures faster.</summary>
    string Attention<TN>(AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer o) where TN : struct, IFloatAdad<TN>;

    /// <summary>O for rows of queries by the composition of the device's own operations.</summary>
    string AttentionComposed<TN>(AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer o) where TN : struct, IFloatAdad<TN>;

    /// <summary>O for rows of queries by the fused rows (no Lq × Lk scores in memory).</summary>
    string AttentionRows<TN>(AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer o) where TN : struct, IFloatAdad<TN>;

    /// <summary>O for a few queries against a long key cache.</summary>
    string AttentionDecode<TN>(AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer o) where TN : struct, IFloatAdad<TN>;

    /// <summary>dQ, dK, dV from dO.</summary>
    string AttentionBackward<TN>(AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer dO, TBuffer dq, TBuffer dk, TBuffer dv) where TN : struct, IFloatAdad<TN>;
}
