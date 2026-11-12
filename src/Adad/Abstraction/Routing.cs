// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Routing: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Routing.cs; Cuda/GpuOps.Routing.cs with its kernels in Cuda/Ptx.Routing.cs.

using System.Buffers;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

public static class Routing
{
    /// <summary>Refuses a top-k the definition does not cover: k from 0 to the row's columns (the k largest of fewer than k
    /// values are not defined), rows and columns not negative.</summary>
    public static void ValidateTopK(int rows, int cols, int k)
    {
        if (rows < 0 || cols < 0 || k < 0 || k > cols) throw new ArgumentException($"top-k: the {k} largest of rows of {cols} values (k is at most the row's columns)");
    }

    /// <summary>Adad's order on values, the one every comparison takes on every device (top-k, max, min, argmax, max
    /// pooling): positive when a ranks above b (larger, or a NaN against a number), zero when they rank the same (equal,
    /// −0 and +0 among them, or two NaNs), negative when b ranks above a.</summary>
    public static int Order(double a, double b) => a > b ? 1 : a < b ? -1 : a == b ? 0 : double.IsNaN(a) ? (double.IsNaN(b) ? 0 : 1) : -1;

    /// <summary>Adad's order of a row's values with their columns: a at column ia ranks before b at column ib when Order puts
    /// it above, and of two that rank the same the lower column first. A total order: each column of a row has one rank.
    /// Top-k takes the first k; max is the first's value and argmax its column; min the last's value.</summary>
    public static bool Precedes(double a, int ia, double b, int ib)
    {
        int o = Order(a, b);
        return o != 0 ? o > 0 : ia < ib;
    }

    // Row r's columns in Adad's order, the first k.
    private static int[] Ranked(double[] x, int r, int cols, int k)
    {
        var order = Enumerable.Range(0, cols).ToArray();
        Array.Sort(order, (i, j) => i == j ? 0 : Precedes(x[r * cols + i], i, x[r * cols + j], j) ? -1 : 1);
        return order[..k];
    }

    /// <summary>top-k of each row in float64 by Adad's order (Precedes): values and indices, the first-ranked first.</summary>
    public static (double[] Values, int[] Indices) TopKReference(double[] x, int rows, int cols, int k)
    {
        var values = new double[rows * k];
        var indices = new int[rows * k];
        for (int r = 0; r < rows; r++)
        {
            var order = Ranked(x, r, cols, k);
            for (int i = 0; i < k; i++)
            {
                indices[r * k + i] = order[i];
                values[r * k + i] = x[r * cols + order[i]];
            }
        }

        return (values, indices);
    }

    /// <summary>The top-k of each row as log-probabilities in float64 (IRouting.LogSoftmaxTopK): TopK's columns, each value
    /// less its row's log-sum-exp (m the first-ranked value: a row holding a NaN gives NaNs).</summary>
    public static (double[] Values, int[] Indices) LogSoftmaxTopKReference(double[] x, int rows, int cols, int k)
    {
        var values = new double[rows * k];
        var indices = new int[rows * k];
        for (int r = 0; r < rows; r++)
        {
            double m = double.NegativeInfinity, sum = 0;
            for (int c = 0; c < cols; c++) m = Math.Max(m, x[r * cols + c]);
            for (int c = 0; c < cols; c++) sum += Math.Exp(x[r * cols + c] - m);
            var order = Ranked(x, r, cols, k);
            for (int i = 0; i < k; i++)
            {
                indices[r * k + i] = order[i];
                values[r * k + i] = x[r * cols + order[i]] - m - Math.Log(sum);
            }
        }

        return (values, indices);
    }

    public static double[] PickReference(double[] x, int rows, int cols, int[] index)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++) y[r] = (uint)index[r] < (uint)cols ? x[r * cols + index[r]] : 0;      // a column outside the row: 0
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
