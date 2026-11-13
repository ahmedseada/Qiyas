// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The stored types by name, and generic dispatch on a name: a visitor's Visit<TN> runs with the type's struct.
using System.Numerics;

namespace Qiyas.Bench;

public interface ITypeVisitor<T>
{
    T Visit<TN>() where TN : struct, IAdad<TN, float>;
}

public interface IFloatTypeVisitor<T>
{
    T Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>;
}

public interface IDoubleTypeVisitor<T>
{
    T Visit<TN>() where TN : struct, IAdad<TN, float>, IAdad<TN, double>;
}

public static class Registry
{
    public static readonly string[] Names =
    [
        "float64", "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2",
        "int64", "int32", "int16", "int8", "int4", "int2",
        "uint64", "uint32", "uint16", "uint8", "uint2",
        "q8_0", "q4_0", "q4_k", "q6_k", "bool",
    ];

    /// <summary>The block formats (float values in blocks with shared scales): their lane is float32.</summary>
    public static bool IsBlock(string name) => name.StartsWith("q", StringComparison.Ordinal);

    public static bool IsFloat(string name) => name.StartsWith("float", StringComparison.Ordinal) || name.StartsWith("bfloat", StringComparison.Ordinal) || IsBlock(name);

    /// <summary>Types whose values read exactly into float64 lanes (all but the 16- and 8-bit floats and the block formats, which read into float32).</summary>
    public static bool HasDoubleLane(string name) => !IsFloat(name) || name is "float64" or "float32";

    public static int Bits(string name) => With(name, new BitsVisitor());

    public static long Bytes(string name, long values) => With(name, new BytesVisitor(values));

    private sealed class BytesVisitor(long values) : ITypeVisitor<long>
    {
        public long Visit<TN>() where TN : struct, IAdad<TN, float> => TN.Bytes(values);
    }

    public static T With<T>(string name, ITypeVisitor<T> v) => name switch
    {
        "float64" => v.Visit<Float64>(),
        "float32" => v.Visit<Float32>(),
        "float16" => v.Visit<Float16>(),
        "bfloat16" => v.Visit<BFloat16>(),
        "float8_e4m3" => v.Visit<Float8E4M3>(),
        "float8_e5m2" => v.Visit<Float8E5M2>(),
        "int64" => v.Visit<Int64>(),
        "int32" => v.Visit<Int32>(),
        "int16" => v.Visit<Int16>(),
        "int8" => v.Visit<Int8>(),
        "int4" => v.Visit<Int4>(),
        "int2" => v.Visit<Int2>(),
        "uint64" => v.Visit<UInt64>(),
        "uint32" => v.Visit<UInt32>(),
        "uint16" => v.Visit<UInt16>(),
        "uint8" => v.Visit<UInt8>(),
        "uint2" => v.Visit<UInt2>(),
        "q8_0" => v.Visit<Q8_0>(),
        "q4_0" => v.Visit<Q4_0>(),
        "q4_k" => v.Visit<Q4_K>(),
        "q6_k" => v.Visit<Q6_K>(),
        "bool" => v.Visit<Bool>(),
        _ => throw new ArgumentException($"no stored type named {name}"),
    };

    public static T WithFloat<T>(string name, IFloatTypeVisitor<T> v) => name switch
    {
        "float64" => v.Visit<Float64>(),
        "float32" => v.Visit<Float32>(),
        "float16" => v.Visit<Float16>(),
        "bfloat16" => v.Visit<BFloat16>(),
        "float8_e4m3" => v.Visit<Float8E4M3>(),
        "float8_e5m2" => v.Visit<Float8E5M2>(),
        "q8_0" => v.Visit<Q8_0>(),
        "q4_0" => v.Visit<Q4_0>(),
        "q4_k" => v.Visit<Q4_K>(),
        "q6_k" => v.Visit<Q6_K>(),
        _ => throw new ArgumentException($"{name} is not a float type"),
    };

    public static T WithDouble<T>(string name, IDoubleTypeVisitor<T> v) => name switch
    {
        "float64" => v.Visit<Float64>(),
        "float32" => v.Visit<Float32>(),
        "int64" => v.Visit<Int64>(),
        "int32" => v.Visit<Int32>(),
        "int16" => v.Visit<Int16>(),
        "int8" => v.Visit<Int8>(),
        "int4" => v.Visit<Int4>(),
        "int2" => v.Visit<Int2>(),
        "uint64" => v.Visit<UInt64>(),
        "uint32" => v.Visit<UInt32>(),
        "uint16" => v.Visit<UInt16>(),
        "uint8" => v.Visit<UInt8>(),
        "uint2" => v.Visit<UInt2>(),
        "bool" => v.Visit<Bool>(),
        _ => throw new ArgumentException($"{name} has no float64 lane"),
    };

    /// <summary>Stored values as float64, exact for every type (through the float64 lane, or the float32 one for the small floats).</summary>
    public static double[] Values(string name, byte[] stored, int count) =>
        HasDoubleLane(name) ? WithDouble(name, new ValuesViaDouble(stored, count)) : With(name, new ValuesViaFloat(stored, count));

    /// <summary>Values stored by the type (rounded as the type rounds, held to its range), through its float64 lane or its float32 one.</summary>
    public static byte[] Encode(string name, double[] values) =>
        HasDoubleLane(name) ? WithDouble(name, new EncodeViaDouble(values)) : With(name, new EncodeViaFloat(values));

    private sealed class EncodeViaDouble(double[] values) : IDoubleTypeVisitor<byte[]>
    {
        public byte[] Visit<TN>() where TN : struct, IAdad<TN, float>, IAdad<TN, double>
        {
            var bytes = new byte[TN.Bytes(values.Length)];
            TN.Store(values, bytes, 0);
            return bytes;
        }
    }

    private sealed class EncodeViaFloat(double[] values) : ITypeVisitor<byte[]>
    {
        public byte[] Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            var f = new float[values.Length];
            for (int i = 0; i < f.Length; i++) f[i] = (float)values[i];
            var bytes = new byte[TN.Bytes(values.Length)];
            TN.Store(f, bytes, 0);
            return bytes;
        }
    }

    /// <summary>The type's lowest value (a float type's: its largest finite, negated).</summary>
    public static double Lowest(string name) => With(name, new LowestVisitor());

    private sealed class LowestVisitor : ITypeVisitor<double>
    {
        public double Visit<TN>() where TN : struct, IAdad<TN, float> => TN.Min;
    }

    /// <summary>The type's smallest step (its subnormals'): float64 2^−1074, float32 2^−149, float16 2^−24, bfloat16 2^−133, float8 e4m3 2^−9, e5m2 2^−16.</summary>
    public static double MinStep(string name) => name switch
    {
        "float64" => Math.Pow(2, -1074), "float32" => Math.Pow(2, -149), "float16" => Math.Pow(2, -24), "bfloat16" => Math.Pow(2, -133),
        "float8_e4m3" => Math.Pow(2, -9), "float8_e5m2" => Math.Pow(2, -16), _ => 0,
    };

    /// <summary>Bits of significand the type keeps (its rounding step), for tolerances.</summary>
    public static int Mantissa(string name) => name switch
    {
        "float64" => 52, "float32" => 23, "float16" => 10, "bfloat16" => 7, "float8_e4m3" => 3, "float8_e5m2" => 2,
        "q8_0" => 6, "q4_0" => 2, "q4_k" => 2, "q6_k" => 4, _ => 0,
    };

    /// <summary>count values stored by the type: floats uniform in [low, high] (as float32 values, so every type stores them
    /// exactly or by its own rounding), integers uniform over the small range the bench uses.</summary>
    public static byte[] Make(string name, int count, int seed, double low, double high) => With(name, new MakeVisitor(count, seed, low, high));

    /// <summary>An output of `count` values of the type as it stands before the operation that writes it: values no operation
    /// writes by chance, so a check reads only values the operation wrote. 0xFF in every byte (NaN for the floats, -1 or the
    /// largest value for the integers, true for bool); a block format's values 1000 each, and the last block's values past
    /// the count 0, as Blocks.Store asks of a buffer whose blocks an output does not cover whole.</summary>
    public static byte[] Poisoned(string name, long count)
    {
        if (IsBlock(name)) return Encode(name, Enumerable.Repeat(1000.0, (int)count).ToArray());
        var a = new byte[Bytes(name, count)];
        Array.Fill(a, (byte)0xFF);
        return a;
    }

    /// <summary>An output in lanes (float32 or float64 values, int32 indices) before the operation that writes it: 0xFF in every
    /// byte (NaN, -1).</summary>
    public static byte[] Poisoned(long bytes)
    {
        var a = new byte[bytes];
        Array.Fill(a, (byte)0xFF);
        return a;
    }

    /// <summary>Indices before the operation that writes them: -1 each.</summary>
    public static int[] PoisonedIndices(int count)
    {
        var a = new int[count];
        Array.Fill(a, -1);
        return a;
    }

    private sealed class BitsVisitor : ITypeVisitor<int>
    {
        public int Visit<TN>() where TN : struct, IAdad<TN, float> => TN.Bits;
    }

    private sealed class ValuesViaDouble(byte[] stored, int count) : IDoubleTypeVisitor<double[]>
    {
        public double[] Visit<TN>() where TN : struct, IAdad<TN, float>, IAdad<TN, double>
        {
            var d = new double[count];
            TN.Load(stored, 0, d);
            return d;
        }
    }

    private sealed class ValuesViaFloat(byte[] stored, int count) : ITypeVisitor<double[]>
    {
        public double[] Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            var f = new float[count];
            TN.Load(stored, 0, f);
            var d = new double[count];
            for (int i = 0; i < count; i++) d[i] = f[i];
            return d;
        }
    }

    private sealed class MakeVisitor(int count, int seed, double low, double high) : ITypeVisitor<byte[]>
    {
        public byte[] Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            var random = new Random(seed);
            var bytes = new byte[TN.Bytes(count)];
            bool signed = TN.Min < 0;
            // Integer types: small values; a positive lower bound asks for values from 1 (divisors, logarithms).
            (double lo, double hi) = TN.IsFloat ? (low, high) : TN.Bits == 2 ? (signed ? (low > 0 ? (1, 1) : (-1, 0)) : (low > 0 ? (1, 1) : (0, 1))) : signed ? (low > 0 ? (1, 2) : (-2, 2)) : (low > 0 ? (1, 3) : (0, 3));
            var f = new float[count];
            for (int i = 0; i < count; i++) f[i] = TN.IsFloat ? (float)(lo + (hi - lo) * random.NextDouble()) : random.Next((int)lo, (int)hi + 1);
            TN.Store(f, bytes, 0);
            return bytes;
        }
    }
}
