// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Optimizers: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Optimizers.cs; Cuda/GpuOps.Optimizers.cs with its kernels in Cuda/Ptx.Optimizers.cs;
// Vulkan/VulkanOps.Optimizers.cs with its kernels in Vulkan/Kernels.Optimizers.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>An optimizer step's scalars: the learning rate, the momentum / first-moment decay, the second-moment decay,
/// eps, the weight decay, the step count (for Adam's bias correction). The steps, each value's in the type's float lane:
/// sgd p = p − lr·(g + wd·p); sgd_momentum m = b1·m + g + wd·p, p = p − lr·m; adam m = b1·m + (1 − b1)·g,
/// v = b2·v + (1 − b2)·g², p = p − lr·m̂ / (√v̂ + eps), m̂ = m / (1 − b1ᵗ), v̂ = v / (1 − b2ᵗ); adamw adam with the decay
/// decoupled from the moments and taken on the parameter the step starts from: p = (p − lr·wd·p) − lr·m̂ / (√v̂ + eps).</summary>
public sealed record OptimizerArgs(string Name, double Lr, double B1, double B2, double Eps, double Wd, int Step)
{
    public static readonly string[] Names = ["sgd", "sgd_momentum", "adam", "adamw"];

    /// <summary>The moments a step keeps: none (sgd), m (sgd_momentum), m and v (adam, adamw).</summary>
    public int Moments => Name switch { "sgd" => 0, "sgd_momentum" => 1, _ => 2 };

    /// <summary>The norm the gradients of a step across several tensors are clipped to (IOptimizers.OptimizerMany; 0: no
    /// clipping): each gradient is multiplied, as the step reads it, by min(1, MaxNorm / the norm of all the tensors'
    /// gradients together).</summary>
    public double MaxNorm { get; init; }

    public void Validate()
    {
        if (Array.IndexOf(Names, Name) < 0) throw new ArgumentException($"optimizer: no step named {Name}");
        if (!(MaxNorm >= 0) || double.IsInfinity(MaxNorm)) throw new ArgumentException($"optimizer: a gradient norm of {MaxNorm} to clip to");
    }

    /// <summary>The float64 reference: p, m, v updated in place (m is not read by sgd, v by one moment), each gradient
    /// times <paramref name="scale"/> as it is read.</summary>
    public void Reference(double[] p, double[] g, double[]? m, double[]? v, double scale = 1)
    {
        double c1 = 1 - Math.Pow(B1, Step), c2 = 1 - Math.Pow(B2, Step);
        for (int i = 0; i < p.Length; i++)
        {
            double gi = scale == 1 ? g[i] : g[i] * scale;
            if (Name == "sgd")
            {
                p[i] -= Lr * (gi + Wd * p[i]);
                continue;
            }

            if (Name == "sgd_momentum")
            {
                m![i] = B1 * m[i] + gi + Wd * p[i];
                p[i] -= Lr * m[i];
                continue;
            }

            m![i] = B1 * m[i] + (1 - B1) * gi;
            v![i] = B2 * v[i] + (1 - B2) * gi * gi;
            double update = Lr * (m[i] / c1) / (Math.Sqrt(v[i] / c2) + Eps);
            if (Name == "adamw") p[i] -= Lr * Wd * p[i];                     // the decay of the parameter the step starts from
            p[i] -= update;
        }
    }

    /// <summary>The clipping factor of a sum of squares: min(1, MaxNorm / its root), 1 where it is 0 or there is no clipping
    /// (the map's clip_factor).</summary>
    public double Factor(double sumSquares) => MaxNorm > 0 && sumSquares > 0 ? Math.Min(1, MaxNorm / Math.Sqrt(sumSquares)) : 1;

    /// <summary>The float64 reference of a step across several tensors (IOptimizers.OptimizerMany): the sum of the squares
    /// of all their gradients, then each tensor's step with its gradients times the factor. Returns the sum.</summary>
    public double ReferenceMany(double[][] p, double[][] g, double[]?[] m, double[]?[] v)
    {
        double sum = 0;
        foreach (var t in g) foreach (double x in t) sum += x * x;
        double scale = Factor(sum);
        for (int t = 0; t < p.Length; t++) Reference(p[t], g[t], m[t], v[t], scale);
        return sum;
    }
}

/// <summary>One tensor of a step across several: its N parameters and gradients (as the type) and its moments (lane
/// storage; not read where the step keeps none).</summary>
public readonly record struct OptimizerTensor<TBuffer>(TBuffer P, TBuffer G, TBuffer? M, TBuffer? V, int N);
