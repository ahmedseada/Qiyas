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

    public static Fmt Of<TN>() where TN : struct, IAdad<TN> => new(TN.Name, TN.Bits, TN.IsFloat, TN.Min < 0);

    public bool Is(string name) => Name == name;

    /// <summary>Words that hold <paramref name="values"/> values (a block format: its whole blocks, in words).</summary>
    public long Words(long values) => IsBlock ? ((values + BlockValues - 1) / BlockValues * BlockBytes + 3) / 4 : (values * Bits + WordBits - 1) / WordBits;

    /// <summary>The lane this format's values are read into before any other (floats f32/f64, integers s32/u32/s64/u64).</summary>
    public Lane Natural => IsFloat ? (Bits == 64 ? Lane.F64 : Lane.F32) : Bits == 64 ? (Signed ? Lane.S64 : Lane.U64) : Signed ? Lane.S32 : Lane.U32;
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
