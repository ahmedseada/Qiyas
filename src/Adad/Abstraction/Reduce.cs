// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Reduce: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Reduce.cs; Cuda/GpuOps.cs with its kernels in Cuda/Ptx.Reduce.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

public enum Axis
{
    Whole,
    Rows,
    Cols,
}

/// <summary>The reductions' arguments by Adad's definition.</summary>
public static class Reductions
{
    /// <summary>Refuses an extreme (max, min, argmax) the definition does not cover: one over no values (a row of no
    /// columns, a column of no rows, a whole of nothing) has no first- or last-ranked value (Routing.Order). A sum of
    /// nothing is 0, and no output (no rows to reduce along the rows) asks for nothing.</summary>
    public static void ValidateExtreme(string op, Axis axis, int rows, int cols)
    {
        if (op is not ("max" or "min" or "argmax")) return;
        long outputs = axis switch { Axis.Whole => 1, Axis.Rows => rows, _ => cols };
        long each = axis switch { Axis.Whole => (long)rows * cols, Axis.Rows => cols, _ => rows };
        if (rows < 0 || cols < 0 || (outputs > 0 && each == 0))
            throw new ArgumentException($"{op} along {axis.ToString().ToLowerInvariant()} of {rows} × {cols}: an extreme of no values is not defined");
    }
}
