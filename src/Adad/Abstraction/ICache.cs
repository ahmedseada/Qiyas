// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>Key and value caches (Cache.cs) at a position the device holds (position[0], an index buffer read where it is: no
/// value crosses to the host between steps): the rows written there, attention over the rows filled, the mask of a
/// decoder's steps, the position advanced. A cache's values are stored as its type TC, as every tensor's are: a compact
/// cache is a cache of a compact type (a block format's values scaled by their blocks), TC any type. TN, the rows' and
/// the scores' type, is attention's: a float type (softmax a float formula).</summary>
public interface ICache<TBuffer, TIndex>
{
    /// <summary>Rows [p, p + steps) of each head (p = position[0]; rows past Capacity or before row 0 not written) = the
    /// source's rows (Heads × steps × D, TN), each value rounded to TC once (Convert's rule). A block format TC: each block
    /// the rows touch encoded once from the rows' values as float32 and, where rows share it, its other values as stored
    /// (the type's partial store).</summary>
    string CacheWrite<TN, TC>(CacheView c, int steps, TIndex position, TBuffer source, TBuffer cache) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>;

    /// <summary>O (Heads × Lq × D, TN) for Lq queries a head (Q as TN) at positions p + i (p = position[0]) against caches of
    /// a.KvHeadCount heads of a.Lk = Capacity rows (CacheRule.AtPosition: query i sees rows up to p + i, the window's last),
    /// a decode kernel's: each row of K and V read where it is stored as TC, the scores capped where there is a cap.</summary>
    string AttentionCached<TN, TC>(AttentionArgs a, TIndex position, TBuffer q, TBuffer k, TBuffer v, TBuffer o) where TN : struct, IFloatAdad<TN> where TC : struct, IAdad<TC>;

    /// <summary>mask (rows × capacity, TN) = 0 where p + i − window &lt; j ≤ p + i (window 0: every j ≤ p + i), else the
    /// excluded score (CacheView.Excluded: −∞ where TN stores it, else TN's lowest value); p = position[0].</summary>
    string DecoderMask<TN>(int rows, int capacity, int window, TIndex position, TBuffer mask) where TN : struct, IFloatAdad<TN>;

    /// <summary>position[0] += steps where it is.</summary>
    void PositionAdvance(TIndex position, int steps);
}
