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
