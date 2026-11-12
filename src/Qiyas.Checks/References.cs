// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The float64 references the proofs and the bench score Adad against (each operation's math on the stored inputs' values).
using Qiyas.Bench;

internal static class References
{
    /// <summary>sum and sumsq in float64; max the first value in Adad's order (Routing.Order: a NaN when there is one), min
    /// the last (the least number, a NaN only when every value is one), argmax the first position of the first-ranked.</summary>
    public static double[] Reduce(string op, Axis axis, double[] x, int rows, int cols)
    {
        int r = axis == Axis.Whole ? 1 : rows, c = axis == Axis.Whole ? rows * cols : cols;
        int count = axis == Axis.Cols ? c : r;
        var result = new double[count];
        for (int i = 0; i < count; i++)
        {
            int n = axis == Axis.Cols ? r : c;
            double acc = op == "max" || op == "argmax" ? double.NegativeInfinity : op == "min" ? double.PositiveInfinity : 0;
            int at = -1;
            for (int j = 0; j < n; j++)
            {
                double v = axis == Axis.Cols ? x[j * c + i] : x[i * c + j];
                switch (op)
                {
                    case "sum": acc += v; break;
                    case "sumsq": acc += v * v; break;
                    case "max" or "argmax": if (at < 0 || Routing.Order(v, acc) > 0) { acc = v; at = j; } break;
                    case "min": if (at < 0 || Routing.Order(v, acc) < 0) { acc = v; at = j; } break;
                }
            }

            result[i] = op == "argmax" ? at : acc;
        }

        return result;
    }

    public static double[] MapRow(string type, MapOp op, double[] x, double[] v, int rows, int cols)
    {
        var r = new double[rows * cols];
        bool integer = !Registry.IsFloat(type), signed = !type.StartsWith("uint", StringComparison.Ordinal) && type != "bool";
        double m = integer ? Math.Pow(2, Registry.Bits(type)) : 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                double y = op.Reference(x[i * cols + j], v[j], 0, 0, 0);
                if (integer)
                {
                    // The lanes' arithmetic is modular at the type's width (as Map's reference wraps).
                    y = Math.Truncate(y);
                    if (signed) { if (y >= m / 2) y -= m; else if (y < -m / 2) y += m; }
                    else { if (y < 0) y += m; else if (y >= m) y -= m; }
                }

                r[i * cols + j] = y;
            }

        return r;
    }

    public static double[] Transpose(double[] x, int rows, int cols)
    {
        var r = new double[rows * cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) r[j * rows + i] = x[i * cols + j];
        return r;
    }

    public static double[] Gemv(double[] a, double[] x, int rows, int cols, bool transposed)
    {
        var y = new double[transposed ? cols : rows];
        if (transposed) { for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) y[j] += a[i * cols + j] * x[i]; }
        else { for (int i = 0; i < rows; i++) { double s = 0; for (int j = 0; j < cols; j++) s += a[i * cols + j] * x[j]; y[i] = s; } }
        return y;
    }

    /// <summary>The formula on the stored values; integer types divide by Ops.IntegerDivision (toward zero; x / 0 = 0).</summary>
    public static double[] Map(string type, MapOp op, byte[]? x, byte[]? y, byte[]? z, int n, bool wrap = true)
    {
        double[]? vx = x is null ? null : Registry.Values(type, x, n);
        double[]? vy = y is null ? null : Registry.Values(type, y, n);
        double[]? vz = z is null ? null : Registry.Values(type, z, n);
        bool integer = !Registry.IsFloat(type);
        int bits = Registry.Bits(type);
        bool signed = type.StartsWith("int", StringComparison.Ordinal);
        double a = op.A, b = op.B;
        if (integer)
        {
            double lo = signed ? -Math.Pow(2, bits - 1) : 0, hi = signed ? Math.Pow(2, bits - 1) - 1 : Math.Pow(2, bits) - 1;
            a = Math.Clamp(a, lo, hi);
            b = Math.Clamp(b, lo, hi);
        }

        var r = new double[n];
        Parallel.For(0, n, i =>
        {
            double v = op.Reference(vx?[i] ?? 0, vy?[i] ?? 0, vz?[i] ?? 0, a, b);
            if (integer && op.Name == "div" && (vy?[i] ?? 0) == 0) v = 0;     // Ops.IntegerDivision: x / 0 = 0
            if (integer)
            {
                // The lanes' arithmetic is modular at the type's width.
                v = Math.Truncate(v);
                if (wrap)
                {
                    double m = Math.Pow(2, bits);
                    if (signed) { if (v >= m / 2) v -= m; else if (v < -m / 2) v += m; }
                    else { if (v < 0) v += m; else if (v >= m) v -= m; }
                }
            }

            r[i] = v;
        });

        return r;
    }

    public static double[] Softmax(double[] x, int rows, int cols)
    {
        var y = new double[x.Length];
        Parallel.For(0, rows, r =>
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int j = 0; j < cols; j++) max = Math.Max(max, x[r * cols + j]);
            for (int j = 0; j < cols; j++) sum += y[r * cols + j] = Math.Exp(x[r * cols + j] - max);
            for (int j = 0; j < cols; j++) y[r * cols + j] /= sum;
        });
        return y;
    }

    public static double[] RmsNorm(double[] x, int rows, int cols, double eps)
    {
        var y = new double[x.Length];
        Parallel.For(0, rows, r =>
        {
            double sum = 0;
            for (int j = 0; j < cols; j++) sum += x[r * cols + j] * x[r * cols + j];
            double scale = 1 / Math.Sqrt(sum / cols + eps);
            for (int j = 0; j < cols; j++) y[r * cols + j] = x[r * cols + j] * scale;
        });
        return y;
    }
}
