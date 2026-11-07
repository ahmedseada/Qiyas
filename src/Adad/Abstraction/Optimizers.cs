// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Optimizers: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Optimizers.cs; Cuda/GpuOps.Optimizers.cs with its kernels in Cuda/Ptx.Optimizers.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>An optimizer step's scalars: the learning rate, the momentum / first-moment decay, the second-moment decay,
/// eps, the weight decay, the step count (for Adam's bias correction).</summary>
public sealed record OptimizerArgs(string Name, double Lr, double B1, double B2, double Eps, double Wd, int Step)
{
    public static readonly string[] Names = ["sgd_momentum", "adam", "adamw"];

    public int Moments => Name == "sgd_momentum" ? 1 : 2;

    /// <summary>The float64 reference: p, m, v updated in place.</summary>
    public void Reference(double[] p, double[] g, double[] m, double[]? v)
    {
        double c1 = 1 - Math.Pow(B1, Step), c2 = 1 - Math.Pow(B2, Step);
        for (int i = 0; i < p.Length; i++)
        {
            if (Name == "sgd_momentum")
            {
                m[i] = B1 * m[i] + g[i] + Wd * p[i];
                p[i] -= Lr * m[i];
                continue;
            }

            m[i] = B1 * m[i] + (1 - B1) * g[i];
            v![i] = B2 * v[i] + (1 - B2) * g[i] * g[i];
            double update = Lr * (m[i] / c1) / (Math.Sqrt(v[i] / c2) + Eps);
            p[i] -= update;
            if (Name == "adamw") p[i] -= Lr * Wd * p[i];
        }
    }
}
