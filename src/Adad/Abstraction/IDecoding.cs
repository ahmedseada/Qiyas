// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>A decoder's steps on the device (Decoding.cs): each row's ring of the tokens it has taken, the penalties of the
/// tokens in its window. The ring's length is a position the device holds (length[0], an index buffer read where it is and
/// advanced by ICache.PositionAdvance: no value crosses to the host between steps).</summary>
public interface IDecoding<TBuffer, TIndex>
{
    /// <summary>history[r·capacity + length[0] mod capacity] = ids[r] for each of rows rows (the ring's next slot; the
    /// length not advanced: PositionAdvance does, once the step's rows are pushed).</summary>
    void HistoryPush(TIndex ids, TIndex history, TIndex length, int rows, int capacity);

    /// <summary>output (a.Rows × a.Cols, TN) = each row of scores x (row r at r·a.Stride + a.RowOffset, read where it is) with
    /// the tokens of the row's window penalized (PenaltyArgs: the last min(length[0], LastN, Capacity) tokens of its ring;
    /// a token outside [0, Cols) penalizes nothing): a token that occurs n times there once, y = (x above 0 ? x / Repeat :
    /// x · Repeat) − (Presence + Frequency · n) in the float lane, rounded to TN once; every other value x's as it is.</summary>
    string Penalize<TN>(PenaltyArgs a, TBuffer x, TIndex history, TIndex length, TBuffer output) where TN : struct, IFloatAdad<TN>;
}
