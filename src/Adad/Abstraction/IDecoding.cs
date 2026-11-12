// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>A decoder's steps on the device (Decoding.cs, Sampling.cs): each row's ring of the tokens it has taken, the
/// penalties of the tokens in its window, the row's next token drawn from its scores. The ring's length and the sampler's
/// step are positions the device holds (an index buffer's first value, read where it is and advanced by
/// ICache.PositionAdvance: no value crosses to the host between steps).</summary>
public interface IDecoding<TBuffer, TIndex>
{
    /// <summary>Each row's token into its ring: history[r·capacity + length[0] mod capacity] = ids[r] for each of rows rows,
    /// the ring's slot after its last token (the length not advanced: PositionAdvance does, once the step's rows are
    /// pushed).</summary>
    void HistoryPush(TIndex ids, TIndex history, TIndex length, int rows, int capacity);

    /// <summary>output (a.Rows × a.Cols, TN) = each row of scores x (row r at r·a.Stride + a.RowOffset, read where it is) with
    /// the tokens of the row's window penalized (PenaltyArgs: the last min(length[0], Window) tokens of its ring; a token
    /// outside [0, Cols) penalizes nothing): a token that occurs n times there once, y = (x above 0 ? x / Repeat : x · Repeat)
    /// − (Presence + Frequency · n) in TN's float lane (Fmt.FloatLane), rounded to TN once by its own conversion; every other
    /// value x's as it is stored. Every type but bool.</summary>
    string Penalize<TN>(PenaltyArgs a, TBuffer x, TIndex history, TIndex length, TBuffer output) where TN : struct, IAdad<TN>;

    /// <summary>picks[r] = a column of row r of the scores x (row r at r·a.Stride + a.RowOffset, read where it is) drawn from
    /// softmax(x / a.Temperature) held in turn to its a.TopK most probable columns, to the nucleus of a.TopP of their
    /// probability and to the columns at least a.MinP times as probable as the most probable (SamplingArgs; Sampling.cs has
    /// the rule: weights exp(s − largest) in the float lane, each rule a threshold on them, equal weights kept together),
    /// by the seed's uniform draw at (step[0], r): the first kept column, in column order, at which the running sum of the
    /// kept weights reaches the draw times their sum; −1 for a row with no score (all NaN or −∞). probability (rows, TN;
    /// absent: not stored): the column's weight over the kept weights' sum (0 with no column), rounded to TN once.</summary>
    string SampleFiltered<TN>(SamplingArgs a, TBuffer x, TIndex step, TIndex picks, TBuffer? probability) where TN : struct, IFloatAdad<TN>;
}
