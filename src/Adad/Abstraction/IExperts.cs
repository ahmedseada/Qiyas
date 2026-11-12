// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>A mixture of experts (Experts.cs): the routing of tokens to their experts, and the experts' products over the
/// rows the routing groups by expert, its offsets read where they are.</summary>
public interface IExperts<TBuffer, TIndex>
{
    /// <summary>The routing of tokens rows of router probabilities (tokens × experts): each token's k first experts in
    /// IRouting.TopK's order (a NaN above every number, then the larger first, −0 equal to +0, an equal value or NaN beside NaN
    /// by the lower index), the expert of rank s at slot t·k + s; weights[slot] its probability (normalize: divided by the sum
    /// of the token's k, added in slot order; in the float lane, stored as TN);
    /// offsets (experts + 1) the first row of each expert when the slots are listed expert by expert (each expert's in slot
    /// order), tokenOfRow[row] that row's token, slotRow[slot] that slot's row.</summary>
    string RouteTopK<TN>(TBuffer probabilities, int tokens, int experts, int k, bool normalize, TBuffer weights, TIndex offsets, TIndex tokenOfRow, TIndex slotRow)
        where TN : struct, IFloatAdad<TN>;

    /// <summary>y's rows of expert e = its rows of x · W_e + bias_e (ExpertView; bias absent: none), the sums in the float lane
    /// (a type that is not float: exactly, in the lane its products take by the bounds of every value, ExpertView.IntegerSums)
    /// and y rounded to TN once.</summary>
    string ExpertMatMul<TN>(ExpertView v, TBuffer x, TIndex offsets, TBuffer weights, TBuffer? bias, TBuffer y) where TN : struct, IAdad<TN>;

    /// <summary>hidden's rows of expert e = gate(x · G_e + gb_e, x · U_e + ub_e): two products a row in the lane, the gate a
    /// two-input formula of Map (gated_silu, gated_gelu, gated_relu, …; a type that is not float takes an any-type one, as
    /// Map) applied there, hidden rounded to TN once.</summary>
    string ExpertGatedPair<TN>(ExpertView v, string gate, TBuffer x, TIndex offsets, TBuffer gateWeights, TBuffer? gateBias, TBuffer upWeights, TBuffer? upBias, TBuffer hidden)
        where TN : struct, IAdad<TN>;
}
