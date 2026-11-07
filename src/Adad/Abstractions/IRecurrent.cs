// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad;

/// <summary>Recurrent cells, steps and sequences (Recurrent.cs): LSTM and GRU over a batch of B rows, hidden size H, input
/// size D, in the type's float lane.</summary>
public interface IRecurrent<TBuffer>
{
    /// <summary>g = x·Wᵀ (+ h·Uᵀ) + bias: a cell's gate pre-activations.</summary>
    void CellGates<TN>(CellArgs a, TBuffer x, TBuffer w, TBuffer? h, TBuffer? u, TBuffer bias, TBuffer g) where TN : struct, IAdad<TN>;

    string LstmCell<TN>(CellArgs a, TBuffer g, TBuffer c, TBuffer hOut, TBuffer cOut) where TN : struct, IFloatAdad<TN>;

    string LstmCellBackward<TN>(CellArgs a, TBuffer g, TBuffer c, TBuffer dh, TBuffer dcNext, TBuffer dg, TBuffer dc) where TN : struct, IFloatAdad<TN>;

    string GruCell<TN>(CellArgs a, TBuffer gx, TBuffer gh, TBuffer h, TBuffer hOut) where TN : struct, IFloatAdad<TN>;

    string GruCellBackward<TN>(CellArgs a, TBuffer gx, TBuffer gh, TBuffer h, TBuffer dhOut, TBuffer dgx, TBuffer dgh, TBuffer dh) where TN : struct, IFloatAdad<TN>;

    string LstmStep<TN>(CellArgs a, TBuffer x, TBuffer h, TBuffer c, TBuffer w, TBuffer u, TBuffer bias, TBuffer hOut, TBuffer cOut) where TN : struct, IFloatAdad<TN>;

    string LstmStepBackward<TN>(CellArgs a, TBuffer x, TBuffer h, TBuffer c, TBuffer w, TBuffer u, TBuffer bias, TBuffer dhOut, TBuffer dcOut,
        TBuffer dx, TBuffer dh, TBuffer dc, TBuffer dw, TBuffer du, TBuffer db) where TN : struct, IFloatAdad<TN>;

    string GruStep<TN>(CellArgs a, TBuffer x, TBuffer h, TBuffer w, TBuffer u, TBuffer bias, TBuffer biasH, TBuffer hOut) where TN : struct, IFloatAdad<TN>;

    string GruStepBackward<TN>(CellArgs a, TBuffer x, TBuffer h, TBuffer w, TBuffer u, TBuffer bias, TBuffer biasH, TBuffer dhOut,
        TBuffer dx, TBuffer dh, TBuffer dw, TBuffer du, TBuffer db, TBuffer dbh) where TN : struct, IFloatAdad<TN>;

    /// <summary>T steps: x (T × B × D), h0, c0 → h and c (T × B × H).</summary>
    string LstmSequence<TN>(CellArgs a, int T, TBuffer x, TBuffer h0, TBuffer c0, TBuffer w, TBuffer u, TBuffer bias, TBuffer hOut, TBuffer cOut)
        where TN : struct, IFloatAdad<TN>;

    /// <summary>From dhOut (every step) and dcLast: dx, dh0, dc0 and dw, du, db summed over the steps.</summary>
    string LstmSequenceBackward<TN>(CellArgs a, int T, TBuffer x, TBuffer h0, TBuffer c0, TBuffer w, TBuffer u, TBuffer bias, TBuffer dhOut, TBuffer dcLast,
        TBuffer dx, TBuffer dh0, TBuffer dc0, TBuffer dw, TBuffer du, TBuffer db) where TN : struct, IFloatAdad<TN>;

    /// <summary>T steps: x (T × B × D), h0 → h (T × B × H).</summary>
    string GruSequence<TN>(CellArgs a, int T, TBuffer x, TBuffer h0, TBuffer w, TBuffer u, TBuffer bias, TBuffer biasH, TBuffer hOut) where TN : struct, IFloatAdad<TN>;

    /// <summary>From dhOut (every step): dx, dh0 and dw, du, db, dbh summed over the steps.</summary>
    string GruSequenceBackward<TN>(CellArgs a, int T, TBuffer x, TBuffer h0, TBuffer w, TBuffer u, TBuffer bias, TBuffer biasH, TBuffer dhOut,
        TBuffer dx, TBuffer dh0, TBuffer dw, TBuffer du, TBuffer db, TBuffer dbh) where TN : struct, IFloatAdad<TN>;
}
