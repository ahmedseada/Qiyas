// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Routing: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Routing.cs; Cuda/GpuOps.Routing.cs with its kernels in Cuda/Ptx.Routing.cs.

using System.Buffers;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

public static class Routing
{
    /// <summary>top-k of each row in float64: values and indices, largest first, ties by the lower index.</summary>
    public static (double[] Values, int[] Indices) TopKReference(double[] x, int rows, int cols, int k)
    {
        var values = new double[rows * k];
        var indices = new int[rows * k];
        for (int r = 0; r < rows; r++)
        {
            var order = Enumerable.Range(0, cols).OrderByDescending(c => x[r * cols + c]).ThenBy(c => c).Take(k).ToArray();
            for (int i = 0; i < k; i++) { indices[r * k + i] = order[i]; values[r * k + i] = x[r * cols + order[i]]; }
        }

        return (values, indices);
    }

    public static double[] PickReference(double[] x, int rows, int cols, int[] index)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++) y[r] = x[r * cols + index[r]];
        return y;
    }

    /// <summary>The cross-entropy loss per row in float64: w[r]·(max + log Σ exp(x − max) − x[target]), w absent 1.</summary>
    public static double[] CrossEntropyReference(double[] x, int rows, int cols, int[] targets, double[]? weights = null)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double m = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < cols; c++) m = Math.Max(m, x[r * cols + c]);
            for (int c = 0; c < cols; c++) sum += Math.Exp(x[r * cols + c] - m);
            y[r] = (m + Math.Log(sum) - x[r * cols + targets[r]]) * (weights?[r] ?? 1);
        }

        return y;
    }

    public static int[] SampleReference(double[] x, int rows, int cols, double[] u)
    {
        var y = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            double total = 0;
            for (int c = 0; c < cols; c++) total += x[r * cols + c];
            double threshold = u[r] * total, run = 0;
            int pick = cols - 1;
            for (int c = 0; c < cols; c++) { run += x[r * cols + c]; if (run >= threshold) { pick = c; break; } }
            y[r] = pick;
        }

        return y;
    }

    public static (int[] Counts, int[] Offsets, int[] Order) RouteReference(int[] index, int buckets)
    {
        var counts = new int[buckets];
        foreach (int i in index) counts[i]++;
        var offsets = new int[buckets + 1];
        for (int b = 0; b < buckets; b++) offsets[b + 1] = offsets[b] + counts[b];
        var order = new int[index.Length];
        var cursor = (int[])offsets.Clone();
        for (int i = 0; i < index.Length; i++) order[cursor[index[i]]++] = i;
        return (counts, offsets, order);
    }
}
