// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The numbers as a device's kernels see them, described once for every device: a stored format (Fmt: how a type's values
// sit in words — 32-bit words, 64-bit for the 64-bit types — or in the block formats' blocks) and a lane (Lane: a register
// type a kernel computes in). Each device maps them to its own instructions from this one description (CUDA: Ptx.cs).
namespace Adad.Abstraction;

/// <summary>A value's stored format as a device's kernels see it.</summary>
public sealed record Fmt(string Name, int Bits, bool IsFloat, bool Signed)
{
    public int WordBits => Bits == 64 ? 64 : 32;

    /// <summary>Values a word holds.</summary>
    public int Per => WordBits / Bits;

    /// <summary>A block format: values in blocks of BlockValues taking BlockBytes (0 for the plain formats).</summary>
    public int BlockValues => Name switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 0 };
    public int BlockBytes => Name switch { "q8_0" => 34, "q4_0" => 18, "q4_k" => 144, "q6_k" => 210, _ => 0 };
    public bool IsBlock => BlockValues > 0;

    public static Fmt Of<TN>() where TN : struct, IAdad<TN> => new(TN.Name, TN.Bits, TN.IsFloat, TN.Min < 0) { Scaled = TN.Scaled, FloatBits = TN.FloatBits };

    /// <summary>The type's scaled-int block layout, as it describes it (default: not one).</summary>
    public ScaledBlock Scaled { get; init; }

    /// <summary>A float type's exponent and mantissa widths, as it describes them.</summary>
    public (int Exponent, int Mantissa) FloatBits { get; init; }

    public bool Is(string name) => Name == name;

    /// <summary>Words that hold <paramref name="values"/> values (a block format: its whole blocks, in words).</summary>
    public long Words(long values) => IsBlock ? ((values + BlockValues - 1) / BlockValues * BlockBytes + 3) / 4 : (values * Bits + WordBits - 1) / WordBits;

    /// <summary>The lane this format's values are read into before any other (floats f32/f64, integers s32/u32/s64/u64).</summary>
    public Lane Natural => IsFloat ? (Bits == 64 ? Lane.F64 : Lane.F32) : Bits == 64 ? (Signed ? Lane.S64 : Lane.U64) : Signed ? Lane.S32 : Lane.U32;

    /// <summary>The float lane a float formula takes for this type: a float type's f64 for 64 bits, else f32 (a block format's
    /// f32); an integer's f32 to 16 bits, where it holds each value exactly, else f64, which holds a 32-bit integer exactly and
    /// a 64-bit one to 53 bits (past them, the nearest f64). Values move in the natural lane, exactly; a formula's result is
    /// stored through the type's own conversion (rounded once, held to the range).</summary>
    public Lane FloatLane => IsFloat ? (Bits == 64 ? Lane.F64 : Lane.F32) : Bits <= 16 ? Lane.F32 : Lane.F64;

    /// <summary>How Convert takes a value of <paramref name="from"/> into <paramref name="to"/> with one rounding: the lane it is
    /// read into (exact: the natural lane; between integers a 64-bit one, so the range is held once, in it), and whether it is
    /// first rounded to odd at float32 — a value read wider than float32 going to a float narrower than it, whose odd last
    /// bit (float32 keeps at least two bits past any narrower float's) leaves the store's rounding the only one. A block
    /// format encodes from float32 values: a value read wider is rounded to nearest float32 first, as the format defines.</summary>
    public static (Lane Lane, bool ToOdd) ConvertPlan(Fmt from, Fmt to)
    {
        if (to.IsBlock) return (Lane.F32, false);
        if (!from.IsFloat && !to.IsFloat) return (from.Bits == 64 && !from.Signed ? Lane.U64 : Lane.S64, false);
        var lane = from.IsBlock ? Lane.F32 : from.Natural;
        return (lane, to.IsFloat && to.Bits < 32 && lane != Lane.F32);
    }
}

/// <summary>
/// A block of integer codes under one float16 scale at the block's start: Values a block taking Bytes; CodeBits 8 (a byte a
/// value) or 4 (byte j's low nibble value j, its high nibble value j + Values/2); CodeBias added to each code. Default: not one.
/// </summary>
public readonly record struct ScaledBlock(int Values, int Bytes, int CodeBits, int CodeBias)
{
    /// <summary>The shape the block product and the block gemv take: 32 values, 8- or 4-bit codes.</summary>
    public bool Is32 => Values == 32 && CodeBits is 8 or 4;
}

/// <summary>
/// How a block format reaches its fast routes (the block product, the block gemv), for the design comparison: "name" (as
/// the code has it: the route tests the type's name), "props" (the route tests the layout the type describes), "generic"
/// (no fast route: the path every type takes). ADAD_ROUTES chooses; default "name".
/// </summary>
public static class Routes
{
    public static readonly string Mode = (Environment.GetEnvironmentVariable("ADAD_ROUTES") ?? "name").Trim().ToLowerInvariant();

    public static bool Generic => Mode == "generic";

    public static bool Props => Mode == "props";

    /// <summary>Whether the format is one the block routes can take (name or layout, as the mode says).</summary>
    public static bool Takes(Fmt f) => Props ? f.Scaled.Is32 : f.Name is "q8_0" or "q4_0";

    public static bool Takes<TN>() where TN : struct, IAdad<TN> => Props ? TN.Scaled.Is32 : TN.Name is "q8_0" or "q4_0";

    /// <summary>Whether the route is taken: the format qualifies and the mode isn't generic.</summary>
    public static bool Route(Fmt f) => !Generic && Takes(f);

    public static bool Route<TN>() where TN : struct, IAdad<TN> => !Generic && Takes<TN>();

    /// <summary>A byte a value (Q8_0's codes): by name or by the layout.</summary>
    public static bool Bytewise(Fmt f) => Props ? f.Scaled.CodeBits == 8 : f.Is("q8_0");

    /// <summary>The layout the routes read: from the name (as the code has it) or as the type describes it.</summary>
    public static ScaledBlock Layout(string name) => name switch { "q8_0" => new(32, 34, 8, 0), "q4_0" => new(32, 18, 4, -8), _ => default };

    public static ScaledBlock Layout<TN>() where TN : struct, IAdad<TN> => Props ? TN.Scaled : Layout(TN.Name);

    /// <summary>Whether the format is the bfloat16 layout (8-bit exponent, 7-bit mantissa): by name or by description.</summary>
    public static bool IsBf16(Fmt f) => Props ? f.FloatBits == (8, 7) : f.Is("bfloat16");

    /// <summary>Whether the format is the matrix unit's own 16-bit float (f16: 5/10, bf16: 8/7): by name or by description.</summary>
    public static bool IsUnitFloat(Fmt f, bool f16) => Props ? f.FloatBits == (f16 ? (5, 10) : (8, 7)) : f.Is(f16 ? "float16" : "bfloat16");
}

/// <summary>A register type kernels compute in: f32, f64, and the 32- and 64-bit integers, signed (s) or not (u).</summary>
public sealed record Lane(string Name, int Size, bool IsFloat)
{
    public static readonly Lane F32 = new("f32", 4, true);
    public static readonly Lane F64 = new("f64", 8, true);
    public static readonly Lane S32 = new("s32", 4, false);
    public static readonly Lane U32 = new("u32", 4, false);
    public static readonly Lane S64 = new("s64", 8, false);
    public static readonly Lane U64 = new("u64", 8, false);

    public bool Wide => Size == 8;
}
