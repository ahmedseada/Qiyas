// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of a decoder's steps (IDecoding): the penalties' arguments, the window's rule and the float64
// references. The devices implement it in their folders: Cpu/Decoding.cs; Cuda/GpuOps.Decoding.cs with its kernels in
// Cuda/Ptx.Decoding.cs; Vulkan/VulkanOps.Decoding.cs with its kernels in Vulkan/Kernels.Decoding.cs.

namespace Adad.Abstraction;

/// <summary>The repetition penalties of Rows rows of Cols scores, each row with a ring of Capacity tokens: the window the
/// last LastN tokens (0: the ring's), a token in it scaled toward lower scores by Repeat (1: not) and lowered by Presence
/// and by Frequency a time it occurs there.</summary>
public sealed record PenaltyArgs(int Rows, int Cols, int Capacity, int LastN, double Repeat, double Presence, double Frequency)
{
    /// <summary>The values between two rows' first scores in x (0: Cols, the rows contiguous).</summary>
    public int RowStride { get; init; }

    /// <summary>The first row's first score in x.</summary>
    public int RowOffset { get; init; }

    public int Stride => RowStride == 0 ? Cols : RowStride;

    public long Values => (long)Rows * Cols;

    /// <summary>The most tokens a window holds.</summary>
    public int MaxWindow => LastN > 0 ? Math.Min(LastN, Capacity) : Capacity;

    /// <summary>The tokens the window holds at a ring's length: the last of them, the ring's when it has wrapped.</summary>
    public int Window(int length) => (int)Math.Clamp((long)length, 0, MaxWindow);

    public void Validate()
    {
        if (Rows < 0 || Cols < 0 || Capacity < 1 || LastN < 0) throw new ArgumentException($"penalties: {Rows} rows of {Cols} scores, a ring of {Capacity}, the last {LastN}");
        if (RowStride < 0 || RowOffset < 0 || (RowStride > 0 && RowStride < Cols)) throw new ArgumentException($"penalties: rows of {Cols} scores {RowStride} apart from {RowOffset}");
        if (!(Repeat > 0) || double.IsInfinity(Repeat)) throw new ArgumentException($"penalties: a repeat penalty of {Repeat}");
        if (!double.IsFinite(Presence) || !double.IsFinite(Frequency)) throw new ArgumentException($"penalties: presence {Presence}, frequency {Frequency}");
        if (Values > int.MaxValue || (long)(Rows - 1) * Stride + RowOffset + Cols > int.MaxValue) throw new ArgumentException("penalties: past 2^31 values");
    }

    /// <summary>The penalized value of a score whose token occurs n times in the window.</summary>
    public double Penalized(double x, int n) => (x > 0 ? x / Repeat : x * Repeat) - (Presence + Frequency * n);

    /// <summary>The float64 reference: x (as the device reads it, rows Stride apart from RowOffset), each row's ring (Rows ×
    /// Capacity) at the length.</summary>
    public double[] Reference(double[] x, int[] history, int length)
    {
        var y = new double[Values];
        int n = Window(length);
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
