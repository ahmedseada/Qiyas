// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Map: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Map.cs; Cuda/GpuOps.cs with its kernels in Cuda/Ptx.Map.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

/// <summary>What the bench and the checks need to know about a formula.</summary>
public sealed record MapOp(string Name, int Inputs, bool FloatOnly, Func<double, double, double, double, double, double> Reference,
    (double Low, double High) XRange, (double Low, double High) YRange, double A, double B, string Note = "", int Ulps = 4)
{
    /// <summary>Inputs (x, y, z) come from this many stored arrays; the rest are unused.</summary>
    public bool IsBackward => Name.EndsWith("_bwd", StringComparison.Ordinal);
}

public static class Ops
{
    private const double GeluK = 0.7978845608028654, GeluC = 0.044715;    // √(2/π), the tanh form's cubic coefficient

    // The references in forms free of cancellation: 0.5·(1 + tanh u) = σ(2u); 1 − tanh² = sech²; 1 − σ(x) = σ(−x); 1 + erf = erfc(−·).
    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    private static double GeluU(double x) => GeluK * (x + GeluC * x * x * x);
    private static double Gelu(double x) => x * Sigmoid(2 * GeluU(x));
    private static double GeluGrad(double x)
    {
        double u = GeluU(x), du = GeluK * (1 + 3 * GeluC * x * x);
        return Sigmoid(2 * u) + 0.5 * x * Funcs.SechSquaredReference(u) * du;
    }
    private static double GeluErf(double x) => 0.5 * x * Funcs.ErfcReference(x * -0.7071067811865476);
    private static double GeluErfGrad(double x) => 0.5 * Funcs.ErfcReference(x * -0.7071067811865476) + x * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    private static double Silu(double x) => x * Sigmoid(x);
    private static double SiluGrad(double x) => Sigmoid(x) * (1 + x * Sigmoid(-x));
    private static double Sign(double x) => x > 0 ? 1 : x < 0 ? -1 : 0;
    private static double Pow(double x, double a) => Math.Exp(a * Math.Log(Math.Abs(x))) * (x < 0 && Math.Abs(a % 2) == 1 ? -1 : 1);

    private static readonly (double, double) Unit = (-1, 1), Positive = (0.5, 4), Any = (-4, 4), Wide = (-20, 20);
    private const string Cond = " (ill-conditioned past |x| ≈ 3: the result follows σ or sech² of a rounded argument that grows as |x|³, so its error in ulps grows with |x|)";

    /// <summary>The catalogue, in Adad's own terms. Integer types run every formula that is not FloatOnly in their own lane
    /// (truncating division, wrapping arithmetic as the lane does); float types run all of them.</summary>
    public static readonly MapOp[] All =
    [
        // ---- any lane
        new("copy", 1, false, (x, y, z, a, b) => x, Any, Any, 0, 0),
        new("fill", 0, false, (x, y, z, a, b) => a, Any, Any, 3, 0),
        new("neg", 1, false, (x, y, z, a, b) => -x, Any, Any, 0, 0),
        new("abs", 1, false, (x, y, z, a, b) => Math.Abs(x), Any, Any, 0, 0),
        new("sign", 1, false, (x, y, z, a, b) => Sign(x), Any, Any, 0, 0),
        new("square", 1, false, (x, y, z, a, b) => x * x, Any, Any, 0, 0),
        new("relu", 1, false, (x, y, z, a, b) => Math.Max(x, 0), Any, Any, 0, 0),
        new("add_scalar", 1, false, (x, y, z, a, b) => x + a, Any, Any, 2, 0),
        new("scale", 1, false, (x, y, z, a, b) => x * a, Any, Any, 3, 0),
        new("affine", 1, false, (x, y, z, a, b) => a * x + b, Any, Any, 2, 1),
        new("clamp", 1, false, (x, y, z, a, b) => Math.Min(Math.Max(x, a), b), Any, Any, -1, 2),
        new("add", 2, false, (x, y, z, a, b) => x + y, Any, Any, 0, 0),
        new("sub", 2, false, (x, y, z, a, b) => x - y, Any, Any, 0, 0),
        new("mul", 2, false, (x, y, z, a, b) => x * y, Any, Any, 0, 0),
        new("div", 2, false, (x, y, z, a, b) => x / y, Any, (1, 4), 0, 0, "integer lanes: truncating"),
        new("max", 2, false, (x, y, z, a, b) => Math.Max(x, y), Any, Any, 0, 0),
        new("min", 2, false, (x, y, z, a, b) => Math.Min(x, y), Any, Any, 0, 0),
        new("axpy", 2, false, (x, y, z, a, b) => a * x + y, Any, Any, 2, 0),
        new("muladd", 3, false, (x, y, z, a, b) => x * y + z, Any, Any, 0, 0),
        new("where", 3, false, (x, y, z, a, b) => x != 0 ? y : z, (0, 1), Any, 0, 0, "x is the mask: nonzero takes y"),
        new("square_bwd", 2, false, (x, y, z, a, b) => 2 * x * y, Any, Any, 0, 0),
        new("abs_bwd", 2, false, (x, y, z, a, b) => Sign(x) * y, Any, Any, 0, 0),
        new("relu_bwd", 2, false, (x, y, z, a, b) => x > 0 ? y : 0, Any, Any, 0, 0),
        new("clamp_bwd", 2, false, (x, y, z, a, b) => x > a && x < b ? y : 0, Any, Any, -1, 2),
        new("where_bwd", 2, false, (x, y, z, a, b) => x != 0 ? y : 0, (0, 1), Any, 0, 0),
        new("max_bwd_x", 3, false, (x, y, z, a, b) => x >= y ? z : 0, Any, Any, 0, 0, "max(x, y)'s gradient to x given dz: x takes it where x ≥ y"),
        new("max_bwd_y", 3, false, (x, y, z, a, b) => x >= y ? 0 : z, Any, Any, 0, 0, "y takes it where x < y"),
        new("min_bwd_x", 3, false, (x, y, z, a, b) => x <= y ? z : 0, Any, Any, 0, 0, "min(x, y)'s gradient to x given dz: x takes it where x ≤ y"),
        new("min_bwd_y", 3, false, (x, y, z, a, b) => x <= y ? 0 : z, Any, Any, 0, 0, "y takes it where x > y"),
        new("gated_relu", 2, false, (x, y, z, a, b) => Math.Max(x, 0) * y, Any, Any, 0, 0),
        new("gated_relu_bwd_x", 3, false, (x, y, z, a, b) => x > 0 ? y * z : 0, Any, Any, 0, 0, "d/dx of relu(x)·y given dz"),
        new("gated_relu_bwd_y", 3, false, (x, y, z, a, b) => Math.Max(x, 0) * z, Any, Any, 0, 0),
        // ---- float lanes
        new("sqrt", 1, true, (x, y, z, a, b) => Math.Sqrt(x), Positive, Any, 0, 0),
        new("inv_sqrt", 1, true, (x, y, z, a, b) => 1 / Math.Sqrt(x + a), Positive, Any, 1e-6, 0),
        new("exp", 1, true, (x, y, z, a, b) => Math.Exp(x), Any, Any, 0, 0),
        new("expm1", 1, true, (x, y, z, a, b) => Funcs.Expm1Reference(x), (-0.5, 0.5), Any, 0, 0),
        new("log", 1, true, (x, y, z, a, b) => Math.Log(x), Positive, Any, 0, 0),
        new("sin", 1, true, (x, y, z, a, b) => Math.Sin(x), Wide, Any, 0, 0),
        new("cos", 1, true, (x, y, z, a, b) => Math.Cos(x), Wide, Any, 0, 0),
        new("tanh", 1, true, (x, y, z, a, b) => Math.Tanh(x), Any, Any, 0, 0),
        new("sigmoid", 1, true, (x, y, z, a, b) => Sigmoid(x), Any, Any, 0, 0),
        new("erf", 1, true, (x, y, z, a, b) => Funcs.ErfReference(x), Any, Any, 0, 0),
        new("gelu", 1, true, (x, y, z, a, b) => Gelu(x), Any, Any, 0, 0, "the tanh form" + Cond, Ulps: 64),
        new("gelu_erf", 1, true, (x, y, z, a, b) => GeluErf(x), Any, Any, 0, 0, "x·Φ(x), Φ by erfc (ill-conditioned past |x| ≈ 3: erfc(t) moves 2t² ulps per ulp of t)", Ulps: 64),
        new("silu", 1, true, (x, y, z, a, b) => Silu(x), Any, Any, 0, 0),
        new("leaky_relu", 1, true, (x, y, z, a, b) => x > 0 ? x : a * x, Any, Any, 0.1, 0),
        new("pow", 1, true, (x, y, z, a, b) => Pow(x, a), Positive, Any, 2.5, 0, "x^a for x > 0; x < 0 with an odd integer a keeps the sign (exp(a·log|x|): a·log|x| ulps)", Ulps: 16),
        new("dropout", 2, true, (x, y, z, a, b) => y != 0 ? x * a : 0, Any, (0, 1), 1.25, 0, "y is the keep mask, a the scale"),
        new("gated_silu", 2, true, (x, y, z, a, b) => Silu(x) * y, Any, Any, 0, 0),
        new("gated_gelu", 2, true, (x, y, z, a, b) => Gelu(x) * y, Any, Any, 0, 0, Cond, Ulps: 64),
        new("sqrt_bwd", 2, true, (x, y, z, a, b) => y / (2 * Math.Sqrt(x)), Positive, Any, 0, 0),
        new("inv_sqrt_bwd", 2, true, (x, y, z, a, b) => -0.5 * y / Math.Pow(x + a, 1.5), Positive, Any, 1e-6, 0),
        new("exp_bwd", 2, true, (x, y, z, a, b) => Math.Exp(x) * y, Any, Any, 0, 0),
        new("log_bwd", 2, true, (x, y, z, a, b) => y / x, Positive, Any, 0, 0),
        new("sin_bwd", 2, true, (x, y, z, a, b) => Math.Cos(x) * y, Wide, Any, 0, 0),
        new("cos_bwd", 2, true, (x, y, z, a, b) => -Math.Sin(x) * y, Wide, Any, 0, 0),
        new("tanh_bwd", 2, true, (x, y, z, a, b) => Funcs.SechSquaredReference(x) * y, Any, Any, 0, 0),
        new("sigmoid_bwd", 2, true, (x, y, z, a, b) => Sigmoid(x) * Sigmoid(-x) * y, Any, Any, 0, 0),
        new("silu_bwd", 2, true, (x, y, z, a, b) => SiluGrad(x) * y, Any, Any, 0, 0, Cond, Ulps: 256),
        new("gelu_bwd", 2, true, (x, y, z, a, b) => GeluGrad(x) * y, Any, Any, 0, 0, Cond, Ulps: 256),
        new("gelu_erf_bwd", 2, true, (x, y, z, a, b) => GeluErfGrad(x) * y, Any, Any, 0, 0, Cond, Ulps: 256),
        new("leaky_relu_bwd", 2, true, (x, y, z, a, b) => (x > 0 ? 1 : a) * y, Any, Any, 0.1, 0),
        new("pow_bwd", 2, true, (x, y, z, a, b) => a * Pow(x, a - 1) * y, Positive, Any, 2.5, 0, "", Ulps: 16),
        new("gated_silu_bwd_x", 3, true, (x, y, z, a, b) => SiluGrad(x) * y * z, Any, Any, 0, 0, "d/dx of silu(x)·y given dz" + Cond, Ulps: 256),
        new("gated_silu_bwd_y", 3, true, (x, y, z, a, b) => Silu(x) * z, Any, Any, 0, 0),
        new("gated_gelu_bwd_x", 3, true, (x, y, z, a, b) => GeluGrad(x) * y * z, Any, Any, 0, 0, Cond, Ulps: 256),
        new("gated_gelu_bwd_y", 3, true, (x, y, z, a, b) => Gelu(x) * z, Any, Any, 0, 0, Cond, Ulps: 64),
        new("gated_tanh_sigmoid", 2, true, (x, y, z, a, b) => Math.Tanh(x) * Sigmoid(y), Any, Any, 0, 0, "tanh(x)·σ(y): a gate whose other side has its own function"),
        new("gated_tanh_sigmoid_bwd_x", 3, true, (x, y, z, a, b) => Funcs.SechSquaredReference(x) * Sigmoid(y) * z, Any, Any, 0, 0, "d/dx given dz (sech² by σ(2x)·σ(−2x), as tanh_bwd)", Ulps: 16),
        new("gated_tanh_sigmoid_bwd_y", 3, true, (x, y, z, a, b) => Math.Tanh(x) * Sigmoid(y) * Sigmoid(-y) * z, Any, Any, 0, 0, "d/dy given dz"),
        new("add_gelu", 2, true, (x, y, z, a, b) => Gelu(x + y), Any, Any, 0, 0, "gelu(x + y): a bias and its gelu in one pass (a row of biases by MapRowFloat); the sum rounded in the lane first, then gelu's conditioning" + Cond, Ulps: 256),
        new("clip_factor", 1, true, (x, y, z, a, b) => x > 0 ? Math.Min(1, a / Math.Sqrt(x)) : 1, (0, 4), Any, 1, 0, "min(1, a/√x), 1 where x is 0: the clipping factor of a sum of squares x to the norm a"),
    ];

    public static MapOp Find(string name) => All.FirstOrDefault(o => o.Name == name) ?? throw new ArgumentException($"no element-wise operation named {name}");
}
