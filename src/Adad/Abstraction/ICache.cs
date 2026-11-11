// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Key and value caches (Cache.cs) at a position the device holds (position[0], an index buffer read where it is: no
/// value crosses to the host between steps): the rows written there, attention over the rows filled, the mask of a
/// decoder's steps, the position advanced.</summary>
public interface ICache<TBuffer, TIndex>
{
    /// <summary>Rows [p, p + steps) of each head (p = position[0]; rows past Capacity not written) = the source's rows (Heads
    /// × steps × D, TN), each value rounded to TC once; scaled (scales given): each row's scale max |x| / TC.Max in TN's float
    /// lane, stored as TN at head·Capacity + row, each code x / scale by the stored scale (an integer TC's nearest, ties to
    /// even), held to TC's range, rounded to TC once. A block format TC (no scales): each block the rows touch encoded once
    /// from the rows' values as float32 and, where rows share it, its other values as stored.</summary>
    string CacheWrite<TN, TC>(CacheView c, int steps, TIndex position, TBuffer source, TBuffer cache, TBuffer? scales) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>;

    /// <summary>CacheWrite of a layer's projection rows x (a.Rows = Heads·a.Steps rows of D values of TN, a step at a time: row
    /// (b·a.Steps + s)·a.Heads + h is step s of the cache's head b·a.Heads + h), each row first RMS-normalized (a gain given)
    /// and rotated (a.Half above 0) by IRows.Rope's rule in the float lane, its values kept there (no rounding to TN) and
    /// written by CacheWrite's rule: rounded once, to TC. No gain and no rotation: the rows as they are.</summary>
    string CacheWriteRope<TN, TC>(CacheView c, RopeArgs a, TIndex position, TBuffer x, TBuffer? gain, TBuffer cos, TBuffer sin, TIndex positions, TBuffer cache, TBuffer? scales)
        where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>;

    /// <summary>O (Heads × Lq × D, TN) for Lq queries a head (Q as TN) at positions p + i (p = position[0]) against caches of
    /// a.KvHeadCount heads of a.Lk = Capacity rows (CacheRule.AtPosition: query i sees rows up to p + i, the window's last),
    /// a decode kernel's: each row of K and V read where it is stored as TC (scaled: code·scale in the lane), the scores
    /// capped where there is a cap.</summary>
    string AttentionCached<TN, TC>(AttentionArgs a, TIndex position, TBuffer q, TBuffer k, TBuffer? kScales, TBuffer v, TBuffer? vScales, TBuffer o) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>;

    /// <summary>mask (rows × capacity, TN) = 0 where p + i − window &lt; j ≤ p + i (window 0: every j ≤ p + i), else TN's lowest
    /// (excluded by a softmax's additive mask); p = position[0].</summary>
    string DecoderMask<TN>(int rows, int capacity, int window, TIndex position, TBuffer mask) where TN : struct, IFloatAdad<TN>;

    /// <summary>position[0] += steps where it is.</summary>
    void PositionAdvance(TIndex position, int steps);
}
