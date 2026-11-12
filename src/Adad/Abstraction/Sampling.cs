// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of a decoder's sampling (IDecoding.SampleFiltered): its arguments, the rule of a row's weights
// and of the columns kept, and the float64 reference. The devices implement it in their folders: Cpu/Sampling.cs;
// Cuda/GpuOps.Sampling.cs with its kernels in Cuda/Ptx.Sampling.cs; Vulkan/VulkanOps.Sampling.cs with its kernels in
// Vulkan/Kernels.Sampling.cs.
//
// A row's weights: s = x / Temperature and m the largest s (a NaN is no score); e = exp(s − m), 1 for the largest and 0
// for a NaN. A row whose largest score is +∞ has weight 1 at each +∞ and 0 elsewhere; a row with no score but NaN or −∞
// has no weight at all, and no column is drawn from it (−1).
//
// The columns kept, each rule a threshold on the weights (equal weights at a threshold are kept together, so the kept set
// does not depend on the order of the columns):
//   top-k    the TopK largest weights: the weights at least the TopK-th largest;
//   top-p    of those, the nucleus: the weights at least t, t the largest value at which the kept weights at least t sum
//            to TopP of the kept weights' sum;
//   min-p    of those, the weights at least MinP (MinP times the largest, which is 1).
// The draw: u in [0, 1) from the seed's key at (step, row) (Draws.Uniform), the column the first kept one, in column
// order, at which the running sum of the kept weights reaches u times their sum (the last kept one when rounding leaves
// none).

namespace Adad.Abstraction;

/// <summary>The sampling of Rows rows of Cols scores at a temperature (above 0), held to the TopK most probable columns
/// (0, or Cols and more: all), to the nucleus of TopP of their probability (0 or 1: all) and to the columns at least MinP
/// times as probable as the most probable (0: all); the draws the Seed's.</summary>
public sealed record SamplingArgs(int Rows, int Cols, double Temperature, int TopK, double TopP, double MinP, ulong Seed)
{
    /// <summary>The values between two rows' first scores in x (0: Cols, the rows contiguous).</summary>
    public int RowStride { get; init; }

    /// <summary>The first row's first score in x.</summary>
    public int RowOffset { get; init; }

    public int Stride => RowStride == 0 ? Cols : RowStride;

    public bool HasTopK => TopK > 0 && TopK < Cols;

    public bool HasTopP => TopP > 0 && TopP < 1;

    public bool HasMinP => MinP > 0;

    public void Validate()
    {
        if (Rows < 0 || Cols < 0) throw new ArgumentException($"sampling: {Rows} rows of {Cols} scores");
        if (RowStride < 0 || RowOffset < 0 || (RowStride > 0 && RowStride < Cols)) throw new ArgumentException($"sampling: rows of {Cols} scores {RowStride} apart from {RowOffset}");
        if (!(Temperature > 0) || double.IsInfinity(Temperature)) throw new ArgumentException($"sampling: a temperature of {Temperature} (above 0; the largest score is the reductions' argmax)");
        if (TopK < 0 || !(TopP >= 0 && TopP <= 1) || !(MinP >= 0 && MinP <= 1)) throw new ArgumentException($"sampling: top-k {TopK}, top-p {TopP}, min-p {MinP}");
        if ((long)Rows * Cols > int.MaxValue || (long)Math.Max(0, Rows - 1) * Stride + RowOffset + Cols > int.MaxValue) throw new ArgumentException("sampling: past 2^31 values");
    }

    /// <summary>The float64 weights of a row of scores (the rule above).</summary>
    public double[] Weights(ReadOnlySpan<double> x)
    {
        var e = new double[x.Length];
        double m = double.NegativeInfinity;
        bool any = false;
        foreach (double v in x)
        {
            double s = v / Temperature;
            if (double.IsNaN(s)) continue;
            any = true;
            m = Math.Max(m, s);
        }

        if (!any || double.IsNegativeInfinity(m)) return e;
        for (int c = 0; c < x.Length; c++)
        {
            double s = x[c] / Temperature;
            e[c] = double.IsNaN(s) ? 0 : double.IsPositiveInfinity(m) ? (double.IsPositiveInfinity(s) ? 1 : 0) : Math.Exp(s - m);
        }

        return e;
    }

    /// <summary>The threshold of the kept weights in float64: top-k's, raised by the nucleus's, raised by min-p's (0: all).</summary>
    public double Threshold(double[] e)
    {
        double t = 0;
        if (!HasTopK && !HasTopP) return HasMinP ? MinP : 0;
        var sorted = (double[])e.Clone();
        Array.Sort(sorted);
        Array.Reverse(sorted);
        if (HasTopK) t = sorted[TopK - 1];
        if (HasTopP)
        {
            double mass = 0, run = 0, nucleus = t;
            foreach (double v in sorted) if (v >= t) mass += v;
            for (int i = 0; i < sorted.Length && sorted[i] >= t; i++)
            {
                run += sorted[i];
                if (i + 1 < sorted.Length && sorted[i + 1] == sorted[i]) continue;      // an equal weight is kept with it
                if (run >= TopP * mass) { nucleus = sorted[i]; break; }
            }

            t = Math.Max(t, nucleus);
        }

        return HasMinP ? Math.Max(t, MinP) : t;
    }

    /// <summary>The float64 reference: each row's column (−1: none) and its probability among the kept (0: none), x as the
    /// device reads it (rows Stride apart from RowOffset), at a step.</summary>
    public (int[] Picks, double[] Probability) Reference(double[] x, int step)
    {
        var picks = new int[Rows];
        var probability = new double[Rows];
        var key = Draws.Key(Seed);
        Parallel.For(0, Rows, r =>
        {
            var e = Weights(x.AsSpan(r * Stride + RowOffset, Cols));
            double t = Threshold(e), total = 0, run = 0;
            for (int c = 0; c < Cols; c++) if (e[c] > 0 && e[c] >= t) total += e[c];
            double target = Draws.Uniform(key, (uint)step, (uint)r) * total;
            int pick = -1, last = -1;
            for (int c = 0; c < Cols && pick < 0; c++)
            {
                if (!(e[c] > 0 && e[c] >= t)) continue;
                last = c;
                run += e[c];
                if (run >= target) pick = c;
            }

            if (pick < 0) pick = last;
            picks[r] = pick;
            probability[r] = pick < 0 ? 0 : e[pick] / total;
        });

        return (picks, probability);
    }

    /// <summary>How far each row's pick is from the float64 rule, as probability among the kept columns: the kept weights'
    /// running sum over their total splits [0, 1] into a column's interval each, and the draw u lies in the rule's column's.
    /// 0 when u lies in the pick's own interval; else the distance from u to it (a column the rule does not keep has an
    /// empty interval at its place in the running sum, and its weight over the total is added: the probability it should
    /// not have had). 1 for a pick that is no column of the row, or none where the rule has one. A pick a lane's rounding
    /// moved across a boundary reads the rounding; a wrong column reads the probability between it and the draw.</summary>
    public double[] PickErrors(double[] x, int step, int[] picks)
    {
        var errors = new double[Rows];
        var key = Draws.Key(Seed);
        Parallel.For(0, Rows, r =>
        {
            var e = Weights(x.AsSpan(r * Stride + RowOffset, Cols));
            double t = Threshold(e), total = 0;
            for (int c = 0; c < Cols; c++) if (e[c] > 0 && e[c] >= t) total += e[c];
            int pick = picks[r];
            if (!(total > 0)) { errors[r] = pick == -1 ? 0 : 1; return; }
            if (pick < 0 || pick >= Cols) { errors[r] = 1; return; }
            double before = 0;
            for (int c = 0; c < pick; c++) if (e[c] > 0 && e[c] >= t) before += e[c];
            bool kept = e[pick] > 0 && e[pick] >= t;
            double low = before / total, high = kept ? (before + e[pick]) / total : low, u = Draws.Uniform(key, (uint)step, (uint)r);
            errors[r] = (u < low ? low - u : u > high ? u - high : 0) + (kept ? 0 : e[pick] / total);
        });

        return errors;
    }
}
