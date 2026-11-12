// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of a decoder's sampling (IDecoding.SampleFiltered): its arguments, the rule and the float64
// reference. The devices implement it in their folders: Cpu/Sampling.cs; Cuda/GpuOps.Sampling.cs with its kernels in
// Cuda/Ptx.Sampling.cs; Vulkan/VulkanOps.Sampling.cs with its kernels in Vulkan/Kernels.Sampling.cs.
//
// A row's order: its columns ranked by their scores as stored, in Adad's one order (Routing.Precedes: a larger score
// first, a NaN above every number, −0 equal to +0, equal scores by the lower column). Every rule below keeps a prefix of
// that order, so a kept set never depends on how a lane rounds two scores.
// Its weights: e = exp((x − m) / Temperature), m the first-ranked score, in the type's float lane (Fmt.FloatLane). A row
// whose first-ranked score is not finite (a NaN, +∞, or a row of −∞) has a NaN weight (x − m is NaN at m's own column):
// no column is drawn from it (−1) and its probability is NaN.
// The columns kept, a prefix of the order, each rule within the one before:
//   top-k    the first TopK columns (1 to Cols; Cols keeps every column);
//   nucleus  of those, the shortest prefix whose weights reach TopP times the sum of theirs (0: the first column alone;
//            1: every one);
//   min-p    of those, the columns whose weight is at least MinP (MinP times the first-ranked weight, which is 1; 0 keeps
//            every column).
// The draw: u = the seed's uniform at (step, row) (Draws.Uniform), each kept column the interval [before, before + e) of
// [0, sum) in column order, the column whose interval holds u times the sum: the first at which the running sum passes
// it (the last kept column of positive weight where the lane's rounding leaves none).

namespace Adad.Abstraction;

/// <summary>The sampling of Rows rows of Cols scores at a temperature (above 0), held to the TopK first columns of the
/// row's order (1 to Cols), to the shortest prefix of those holding TopP of their probability (0 to 1) and to the columns
/// at least MinP times as probable as the first (0 to 1); the draws the Seed's.</summary>
public sealed record SamplingArgs(int Rows, int Cols, double Temperature, int TopK, double TopP, double MinP, ulong Seed)
{
    /// <summary>The values between two rows' first scores in x, Cols or more (absent: Cols, the rows one after another).</summary>
    public int? RowStride { get; init; }

    /// <summary>The first row's first score in x.</summary>
    public int RowOffset { get; init; }

    public int Stride => RowStride ?? Cols;

    /// <summary>Top-k leaves a column out.</summary>
    public bool HasTopK => TopK < Cols;

    /// <summary>The nucleus may leave a column out.</summary>
    public bool HasTopP => TopP < 1;

    /// <summary>Min-p may leave a column out.</summary>
    public bool HasMinP => MinP > 0;

    public void Validate()
    {
        if (Rows < 0 || Cols < 0 || (Rows > 0 && Cols < 1)) throw new ArgumentException($"sampling: {Rows} rows of {Cols} scores");
        if (Stride < Cols || RowOffset < 0) throw new ArgumentException($"sampling: rows of {Cols} scores {Stride} apart from {RowOffset}");
        if (!(Temperature > 0) || double.IsInfinity(Temperature)) throw new ArgumentException($"sampling: a temperature of {Temperature} (above 0; the first-ranked score is the reductions' argmax)");
        if (TopK < 1 || TopK > Math.Max(1, Cols)) throw new ArgumentException($"sampling: top-k {TopK} of {Cols} columns (1 to {Cols})");
        if (!(TopP >= 0 && TopP <= 1) || !(MinP >= 0 && MinP <= 1)) throw new ArgumentException($"sampling: top-p {TopP}, min-p {MinP} (0 to 1)");
        if ((long)Rows * Cols > int.MaxValue || (long)Math.Max(0, Rows - 1) * Stride + RowOffset + Cols > int.MaxValue) throw new ArgumentException("sampling: past 2^31 values");
    }

    /// <summary>A row's order: its columns first to last by Adad's order of its scores.</summary>
    public static int[] Order(ReadOnlySpan<double> x)
    {
        var order = new int[x.Length];
        for (int c = 0; c < order.Length; c++) order[c] = c;
        var v = x.ToArray();
        Array.Sort(order, (i, j) => i == j ? 0 : Routing.Precedes(v[i], i, v[j], j) ? -1 : 1);
        return order;
    }

    /// <summary>A row's float64 weights and the columns the rule keeps (none, the weights NaN, for a row whose first-ranked
    /// score is not finite).</summary>
    public (double[] Weights, bool[] Kept) Rule(ReadOnlySpan<double> x) => Rule(x, Order(x));

    /// <summary>The rule over a row in a given order: the scores' order as stored, where float64 holds two of them as one
    /// value (a 64-bit integer's past 2^53).</summary>
    public (double[] Weights, bool[] Kept) Rule(ReadOnlySpan<double> x, int[] order)
    {
        int n = x.Length;
        var e = new double[n];
        var kept = new bool[n];
        double m = x[order[0]];
        if (!double.IsFinite(m)) { Array.Fill(e, double.NaN); return (e, kept); }
        for (int c = 0; c < n; c++) e[c] = Math.Exp((x[c] - m) / Temperature);
        int length = TopK;
        if (HasTopP)
        {
            double mass = 0, run = 0;
            for (int i = 0; i < TopK; i++) mass += e[order[i]];
            for (int i = 0; i < TopK; i++)
            {
                run += e[order[i]];
                if (run >= TopP * mass) { length = i + 1; break; }
            }
        }

        if (HasMinP)
            for (int i = 1; i < length; i++)
                if (!(e[order[i]] >= MinP)) { length = i; break; }

        for (int i = 0; i < length; i++) kept[order[i]] = true;
        return (e, kept);
    }

    /// <summary>The float64 reference: each row's column (−1: none) and its probability among the kept (NaN: none), x as the
    /// device reads it (rows Stride apart from RowOffset), at a step.</summary>
    public (int[] Picks, double[] Probability) Reference(double[] x, int step)
    {
        var picks = new int[Rows];
        var probability = new double[Rows];
        var key = Draws.Key(Seed);
        Parallel.For(0, Rows, r =>
        {
            var (e, kept) = Rule(x.AsSpan(r * Stride + RowOffset, Cols));
            double total = 0, run = 0;
            for (int c = 0; c < Cols; c++) if (kept[c]) total += e[c];
            int pick = -1, last = -1;
            if (total > 0)
            {
                double target = Draws.Uniform(key, (uint)step, (uint)r) * total;
                for (int c = 0; c < Cols && pick < 0; c++)
                {
                    if (!kept[c] || !(e[c] > 0)) continue;
                    last = c;
                    run += e[c];
                    if (run > target) pick = c;
                }

                if (pick < 0) pick = last;
            }

            picks[r] = pick;
            probability[r] = pick < 0 ? double.NaN : e[pick] / total;
        });

        return (picks, probability);
    }

    /// <summary>How far each row's pick is from the float64 rule, as probability among the kept columns: the kept weights'
    /// running sum over their total splits [0, 1] into a column's interval each, and the draw u lies in the rule's column's.
    /// 0 when u lies in the pick's own interval; else the distance from u to it (a column the rule does not keep has an
    /// empty interval at its place in the running sum, and its weight over the total is added: the probability it should
    /// not have had). 1 for a pick that is no column of the row, or none where the rule has one; for a row the rule draws
    /// nothing from, 0 for −1 and 1 for any column.</summary>
    public double[] PickErrors(double[] x, int step, int[] picks)
    {
        var errors = new double[Rows];
        var key = Draws.Key(Seed);
        Parallel.For(0, Rows, r =>
        {
            var (e, kept) = Rule(x.AsSpan(r * Stride + RowOffset, Cols));
            double total = 0;
            for (int c = 0; c < Cols; c++) if (kept[c]) total += e[c];
            int pick = picks[r];
            if (!(total > 0)) { errors[r] = pick == -1 ? 0 : 1; return; }
            if (pick < 0 || pick >= Cols) { errors[r] = 1; return; }
            double before = 0;
            for (int c = 0; c < pick; c++) if (kept[c]) before += e[c];
            double low = before / total, high = kept[pick] ? (before + e[pick]) / total : low, u = Draws.Uniform(key, (uint)step, (uint)r);
            errors[r] = (u < low ? low - u : u > high ? u - high : 0) + (kept[pick] ? 0 : e[pick] / total);
        });

        return errors;
    }
}
