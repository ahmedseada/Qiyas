// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Ctc: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Ctc.cs; Cuda/GpuOps.Ctc.cs with its kernels in Cuda/Ptx.Ctc.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>A ctc problem: T frames, B sequences, C classes; the blank class; per sequence its frames and its labels.</summary>
public sealed record CtcArgs(int T, int B, int C, int Blank, int[] InputLengths, int[] LabelLengths, int[] Labels)
{
    public long Values => (long)T * B * C;

    /// <summary>x (and the gradient) is B × T × C, a sequence's frames adjacent, rather than T × B × C.</summary>
    public bool BatchFirst { get; init; }

    /// <summary>A sequence no alignment fits counts 0 rather than +∞ (its gradient is 0 either way).</summary>
    public bool ZeroInfinity { get; init; }

    /// <summary>Where sequence b's labels start in Labels (a padded B × maxLabels array: b·maxLabels, the padding never
    /// read); absent, the sequences' labels one after another.</summary>
    public int[]? LabelOffsets { get; init; }

    /// <summary>The row (of C values) of frame t of sequence b.</summary>
    public long Row(int t, int b) => BatchFirst ? (long)b * T + t : (long)t * B + b;

    /// <summary>The rows from a frame to the sequence's next, and from a sequence to the next's same frame.</summary>
    public (int Frame, int Sequence) RowStrides => BatchFirst ? (1, T) : (B, 1);

    /// <summary>The first label of sequence b in Labels.</summary>
    public int Offset(int b)
    {
        if (LabelOffsets is not null) return LabelOffsets[b];
        int at = 0;
        for (int i = 0; i < b; i++) at += LabelLengths[i];
        return at;
    }

    /// <summary>The class of extended state s of sequence b (the blank at even s).</summary>
    public int Class(int b, int s) => (s & 1) == 0 ? Blank : Labels[Offset(b) + (s >> 1)];

    /// <summary>The longest extended label sequence: 2·max L + 1.</summary>
    public int MaxStates => 2 * LabelLengths.Max() + 1;

    /// <summary>Refuses what the definition does not cover: a sequence of no frames or more than T, a label that is the
    /// blank or not a class, label counts that do not add up to Labels (or, with offsets, labels past it).</summary>
    public void Validate()
    {
        if (T < 1 || B < 1 || C < 2 || Blank < 0 || Blank >= C) throw new ArgumentException($"ctc: T {T}, B {B}, C {C}, blank {Blank}");
        if (InputLengths.Length != B || LabelLengths.Length != B) throw new ArgumentException("ctc: a length per sequence");
        if (InputLengths.Any(t => t < 1 || t > T)) throw new ArgumentException("ctc: every sequence has 1 to T frames");
        if (LabelLengths.Any(l => l < 0)) throw new ArgumentException("ctc: label counts are not negative");
        if (LabelOffsets is null && LabelLengths.Sum() != Labels.Length) throw new ArgumentException("ctc: the label counts add up to the labels given");
        if (LabelOffsets is not null && LabelOffsets.Length != B) throw new ArgumentException("ctc: a label offset per sequence");
        for (int b = 0, off = 0; b < B; off += LabelLengths[b], b++)
        {
            int start = LabelOffsets?[b] ?? off, end = start + LabelLengths[b];
            if (start < 0 || end > Labels.Length) throw new ArgumentException($"ctc: sequence {b}'s labels past the labels given");
            for (int i = start; i < end; i++)
                if (Labels[i] < 0 || Labels[i] >= C || Labels[i] == Blank) throw new ArgumentException("ctc: labels are classes other than the blank");
        }
    }

    /// <summary>Per sequence (rows of MaxStates + 2, zeros past its states): the class of each extended state, and its
    /// flags: bit 0 where α may come from s − 2 (a label differing from the label before it), bit 1 on the first state
    /// of each class, bits 2 up the next state of its class (0: none).</summary>
    public (int[] Classes, int[] Flags) States()
    {
        // One walk per sequence (its labels from a running offset): a class's states share a parity (the blank the even
        // states, the labels the odd), so its first state is the first seen, and its next the last seen walking back.
        int stride = MaxStates + 2;
        var cls = new int[B * stride];
        var flags = new int[B * stride];
        var seen = new HashSet<int>();
        var next = new Dictionary<int, int>();
        for (int b = 0, run = 0; b < B; run += LabelLengths[b], b++)
        {
            int S = 2 * LabelLengths[b] + 1, row = b * stride, off = LabelOffsets?[b] ?? run;
            seen.Clear();
            next.Clear();
            for (int s = 0; s < S; s++)
            {
                int c = (s & 1) == 0 ? Blank : Labels[off + (s >> 1)];
                cls[row + s] = c;
                bool skip = s >= 2 && c != Blank && c != cls[row + s - 2];
                bool first = seen.Add(c);
                flags[row + s] = (skip ? 1 : 0) | (first ? 2 : 0);
            }

            for (int s = S - 1; s >= 0; s--)
            {
                int c = cls[row + s];
                if (next.TryGetValue(c, out int k)) flags[row + s] |= k << 2;
                next[c] = s;
            }
        }

        return (cls, flags);
    }

    /// <summary>The float64 reference: the loss per sequence and the gradient with respect to the logits (T × B × C).</summary>
    public (double[] Loss, double[] Grad) Reference(double[] x)
    {
        var loss = new double[B];
        var grad = new double[Values];
        for (int b = 0; b < B; b++)
        {
            int tb = InputLengths[b], S = 2 * LabelLengths[b] + 1;
            var logy = new double[tb, C];
            for (int t = 0; t < tb; t++)
            {
                long row = Row(t, b) * C;
                double m = double.NegativeInfinity, sum = 0;
                for (int c = 0; c < C; c++) m = Math.Max(m, x[row + c]);
                for (int c = 0; c < C; c++) sum += Math.Exp(x[row + c] - m);
                double lse = m + Math.Log(sum);
                for (int c = 0; c < C; c++) logy[t, c] = x[row + c] - lse;
            }

            var cls = new int[S];
            for (int s = 0; s < S; s++) cls[s] = Class(b, s);
            bool Skip(int s) => s >= 2 && cls[s] != Blank && cls[s] != cls[s - 2];
            var alpha = new double[tb, S];
            var beta = new double[tb, S];
            for (int s = 0; s < S; s++) alpha[0, s] = s < 2 ? logy[0, cls[s]] : double.NegativeInfinity;
            for (int t = 1; t < tb; t++)
            {
                for (int s = 0; s < S; s++)
                {
                    double v = Lse(alpha[t - 1, s], s >= 1 ? alpha[t - 1, s - 1] : double.NegativeInfinity, Skip(s) ? alpha[t - 1, s - 2] : double.NegativeInfinity);
                    alpha[t, s] = v + logy[t, cls[s]];
                }
            }

            double logP = Lse(alpha[tb - 1, S - 1], S >= 2 ? alpha[tb - 1, S - 2] : double.NegativeInfinity, double.NegativeInfinity);
            loss[b] = double.IsNegativeInfinity(logP) && ZeroInfinity ? 0 : -logP;
            if (double.IsNegativeInfinity(logP)) continue;                   // no alignment: the gradient stays 0
            for (int s = 0; s < S; s++) beta[tb - 1, s] = s >= S - 2 ? logy[tb - 1, cls[s]] : double.NegativeInfinity;
            for (int t = tb - 2; t >= 0; t--)
            {
                for (int s = 0; s < S; s++)
                {
                    double v = Lse(beta[t + 1, s], s + 1 < S ? beta[t + 1, s + 1] : double.NegativeInfinity, s + 2 < S && Skip(s + 2) ? beta[t + 1, s + 2] : double.NegativeInfinity);
                    beta[t, s] = v + logy[t, cls[s]];
                }
            }

            for (int t = 0; t < tb; t++)
            {
                long row = Row(t, b) * C;
                for (int c = 0; c < C; c++) grad[row + c] = Math.Exp(logy[t, c]);
                for (int s = 0; s < S; s++) grad[row + cls[s]] -= Math.Exp(alpha[t, s] + beta[t, s] - logy[t, cls[s]] - logP);
            }
        }

        return (loss, grad);
    }

    private static double Lse(double a, double b, double c)
    {
        double m = Math.Max(a, Math.Max(b, c));
        return double.IsNegativeInfinity(m) ? m : m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m) + Math.Exp(c - m));
    }
}
