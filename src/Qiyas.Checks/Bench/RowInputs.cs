// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The inputs, references and tolerance of a row operation (Rows.cs), shared by the checks and the bench.
namespace Qiyas.Bench;

public sealed class RowCase
{
    public required RowOp Op;
    public required string Type;
    public required int Rows, Cols;
    public required byte[] X;
    public byte[]? X2, G, Bt, S;
    public int[]? Pos;
    public required double[] Y;              // the float64 reference (rows × cols)
    public double[]? Y2;
    public required double[] Scale;          // the error scale per row

    /// <summary>The operation's inputs for a shape (seeds fixed by the operation), with positions given when asked.</summary>
    public static RowCase Make(RowOp op, string type, int rows, int cols, bool positions)
    {
        int n = rows * cols;
        var x = Registry.Make(type, n, 51, op.XRange.Low, op.XRange.High);
        byte[]? x2 = null;
        if (op.Second)
        {
            if (op.Mask)
            {
                var m = new double[n];
                for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) m[r * cols + c] = RowOp.MaskValue(r, c);
                x2 = Registry.Encode(type, m);
            }
            else x2 = Registry.Make(type, n, 52, op.SecondRange.Low, op.SecondRange.High);
        }

        var g = op.Gamma ? Registry.Make(type, cols, 53, 0.5, 2) : null;
        var bt = op.Beta ? Registry.Make(type, cols, 54, -1, 1) : null;
        var s = op.RowScalar ? Registry.Make(type, rows, 55, 0.5, 2) : null;
        int[]? pos = op.Positions && positions ? Enumerable.Range(0, rows).Select(r => r * 3 % 50).ToArray() : null;
        var xv = Registry.Values(type, x, n);
        var x2v = x2 is null ? null : Registry.Values(type, x2, n);
        var gv = g is null ? null : Registry.Values(type, g, cols);
        var btv = bt is null ? null : Registry.Values(type, bt, cols);
        var sv = s is null ? null : Registry.Values(type, s, rows);
        var y = new double[n];
        var y2 = op.SecondOut ? new double[n] : op.RowOut ? new double[rows] : null;
        var scale = new double[rows];
        Parallel.For(0, rows, r =>
        {
            ReadOnlySpan<double> gs = gv is null ? default : gv.AsSpan();
            ReadOnlySpan<double> bts = btv is null ? default : btv.AsSpan();
            scale[r] = op.Reference(xv.AsSpan(r * cols, cols), x2v is null ? default : x2v.AsSpan(r * cols, cols), gs, bts,
                sv is null ? 1 : sv[r], pos is null ? r : pos[r], r, op.A, op.B, y.AsSpan(r * cols, cols), y2 is null ? default : op.RowOut ? y2.AsSpan(r, 1) : y2.AsSpan(r * cols, cols));
        });
        return new RowCase { Op = op, Type = type, Rows = rows, Cols = cols, X = x, X2 = x2, G = g, Bt = bt, S = s, Pos = pos, Y = y, Y2 = y2, Scale = scale };
    }

    /// <summary>The worst distance of an output from its reference, in steps of the type past the tolerance (0 when all within):
    /// a value's tolerance is the type's step at it plus Ulps lane ulps of the row's error scale.</summary>
    /// <summary>A per-row output (RowOut: lane values) against its reference: the worst distance in its tolerances (Ulps lane ulps
    /// of the row's error scale) past 1, 0 when all within.</summary>
    public double WorstRow(byte[] got)
    {
        string lane = Type == "float64" ? "f64" : "f32";
        var values = LaneValues.Decode(lane, got, Rows);
        double laneEps = lane == "f64" ? Math.Pow(2, -52) : Math.Pow(2, -23), worst = 0;
        for (int r = 0; r < Rows; r++)
        {
            double tol = Op.Ulps * laneEps * Scale[r], err = values[r] == Y2![r] ? 0 : Math.Abs(values[r] - Y2[r]);
            if (double.IsNaN(err) || err > tol) worst = Math.Max(worst, double.IsNaN(err) ? double.PositiveInfinity : err / tol);
        }

        return worst;
    }

    public double Worst(byte[] got, bool second = false)
    {
        var reference = second ? Y2! : Y;
        var expected = Registry.Values(Type, Registry.Encode(Type, reference), reference.Length);
        var values = Registry.Values(Type, got, reference.Length);
        int mant = Registry.Mantissa(Type);
        double laneEps = Type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23), minStep = Registry.MinStep(Type);
        double worst = 0;
        for (int r = 0; r < Rows; r++)
        {
            double rowTol = Op.Ulps * laneEps * Scale[r];
            for (int c = 0; c < Cols; c++)
            {
                int i = r * Cols + c;
                double want = expected[i], v = values[i];
                double step = Math.Max(minStep, reference[i] == 0 ? Math.Pow(2, -mant) : Math.Pow(2, Math.Floor(Math.Log2(Math.Abs(reference[i]))) - mant));
                double err = Math.Abs(v - want);
                if (double.IsNaN(err) || err > step + rowTol)
                {
                    if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1" && worst == 0) Console.WriteLine($"  {Op.Name} {Type} [{r},{c}] got {v:R} want {want:R} ref {reference[i]:R} tol {step + rowTol:E2}");
                    worst = Math.Max(worst, double.IsNaN(err) ? double.PositiveInfinity : err / step);
                }
            }
        }

        return worst;
    }
}
