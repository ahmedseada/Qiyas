// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of a decoder's steps (IDecoding): the penalties' arguments, the window's rule and the float64
// references. A row's ring holds its last Capacity tokens in constant memory, its length a position the device holds, so a
// decoder pushes and penalizes step after step with nothing crossing to the host; the window is the ring's last Window
// tokens. Every type but bool (a truth value, not a score): a score is read where it is stored, a penalized one computed
// in the type's float lane (Fmt.FloatLane) and stored through the type's own conversion, every other one stored as it is.
// The devices implement it in their folders: Cpu/Decoding.cs; Cuda/GpuOps.Decoding.cs with its kernels in
// Cuda/Ptx.Decoding.cs; Vulkan/VulkanOps.Decoding.cs with its kernels in Vulkan/Kernels.Decoding.cs.

namespace Adad.Abstraction;

/// <summary>The penalties of Rows rows of Cols scores for the tokens a row has taken: each row's ring of the last Capacity
/// tokens, its window the last Window of them (0 to Capacity), and a token that occurs n times in the window lowered once,
/// its score x to (x above 0 ? x / Repeat : x · Repeat) − (Presence + Frequency · n).</summary>
public sealed record PenaltyArgs(int Rows, int Cols, int Capacity, int Window, double Repeat, double Presence, double Frequency)
{
    /// <summary>The values between two rows' first scores in x, Cols or more (absent: Cols, the rows one after another).</summary>
    public int? RowStride { get; init; }

    /// <summary>The first row's first score in x.</summary>
    public int RowOffset { get; init; }

    public int Stride => RowStride ?? Cols;

    public long Values => (long)Rows * Cols;

    /// <summary>The tokens the window holds at a ring's length: the last of those taken, at most Window.</summary>
    public int Tokens(int length) => (int)Math.Clamp((long)length, 0, Window);

    public void Validate()
    {
        if (Rows < 0 || Cols < 0 || Capacity < 1 || Window < 0 || Window > Capacity) throw new ArgumentException($"penalties: {Rows} rows of {Cols} scores, a window of {Window} in a ring of {Capacity} (a ring holds no more than its capacity)");
        if (Stride < Cols || RowOffset < 0) throw new ArgumentException($"penalties: rows of {Cols} scores {Stride} apart from {RowOffset}");
        if (!(Repeat > 0) || double.IsInfinity(Repeat)) throw new ArgumentException($"penalties: a repeat penalty of {Repeat}");
        if (!double.IsFinite(Presence) || !double.IsFinite(Frequency)) throw new ArgumentException($"penalties: presence {Presence}, frequency {Frequency}");
        if (Values > int.MaxValue || (long)Math.Max(0, Rows - 1) * Stride + RowOffset + Cols > int.MaxValue) throw new ArgumentException("penalties: past 2^31 values");
    }

    /// <summary>The penalized value of a score whose token occurs n times in the window.</summary>
    public double Penalized(double x, int n) => (x > 0 ? x / Repeat : x * Repeat) - (Presence + Frequency * n);

    /// <summary>The float64 reference: x (as the device reads it, rows Stride apart from RowOffset), each row's ring (Rows ×
    /// Capacity) at the length.</summary>
    public double[] Reference(double[] x, int[] history, int length)
    {
        var y = new double[Values];
        int n = Tokens(length);
        var counts = new Dictionary<int, int>();
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++) y[(long)r * Cols + c] = x[(long)r * Stride + RowOffset + c];
            counts.Clear();
            for (int j = 0; j < n; j++)
            {
                int token = history[(long)r * Capacity + (length - 1 - j) % Capacity];
                counts[token] = counts.GetValueOrDefault(token) + 1;
            }

            foreach (var (token, count) in counts)
                if ((uint)token < (uint)Cols) y[(long)r * Cols + token] = Penalized(y[(long)r * Cols + token], count);
        }

        return y;
    }

    /// <summary>The reference of a push: the ring after ids are written at the length's slot.</summary>
    public static int[] PushReference(int[] history, int[] ids, int length, int rows, int capacity)
    {
        var h = (int[])history.Clone();
        for (int r = 0; r < rows; r++) h[(long)r * capacity + length % capacity] = ids[r];
        return h;
    }
}
