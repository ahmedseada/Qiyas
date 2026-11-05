// The stored types by name, and generic dispatch on a name: a visitor's Visit<TN> runs with the type's struct.
using System.Numerics;

namespace Qiyas.Bench;

public interface ITypeVisitor<T>
{
    T Visit<TN>() where TN : struct, IQiyasNumber<TN, float>;
}

public interface IFloatTypeVisitor<T>
{
    T Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>;
}

public interface IDoubleTypeVisitor<T>
{
    T Visit<TN>() where TN : struct, IQiyasNumber<TN, float>, IQiyasNumber<TN, double>;
}

public static class Registry
{
    public static readonly string[] Names =
    [
        "float64", "float32", "float16", "bfloat16", "float8_e4m3", "float8_e5m2",
        "int64", "int32", "int16", "int8", "int4", "int2",
        "uint64", "uint32", "uint16", "uint8", "uint2",
    ];

    public static bool IsFloat(string name) => name.StartsWith("float", StringComparison.Ordinal) || name.StartsWith("bfloat", StringComparison.Ordinal);

    /// <summary>Types whose values read exactly into float64 lanes (all but the 16- and 8-bit floats, which read into float32).</summary>
    public static bool HasDoubleLane(string name) => !IsFloat(name) || name is "float64" or "float32";

    public static int Bits(string name) => With(name, new BitsVisitor());

    public static long Bytes(string name, long values) => (values * Bits(name) + 7) / 8;

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
        _ => throw new ArgumentException($"{name} has no float64 lane"),
    };

    /// <summary>Stored values as float64, exact for every type (through the float64 lane, or the float32 one for the small floats).</summary>
    public static double[] Values(string name, byte[] stored, int count) =>
        HasDoubleLane(name) ? WithDouble(name, new ValuesViaDouble(stored, count)) : With(name, new ValuesViaFloat(stored, count));

    /// <summary>count values stored by the type: floats uniform in [low, high] (as float32 values, so every type stores them
    /// exactly or by its own rounding), integers uniform over the small range the bench uses.</summary>
    public static byte[] Make(string name, int count, int seed, double low, double high) => With(name, new MakeVisitor(count, seed, low, high));

    private sealed class BitsVisitor : ITypeVisitor<int>
    {
        public int Visit<TN>() where TN : struct, IQiyasNumber<TN, float> => TN.Bits;
    }

    private sealed class ValuesViaDouble(byte[] stored, int count) : IDoubleTypeVisitor<double[]>
    {
        public double[] Visit<TN>() where TN : struct, IQiyasNumber<TN, float>, IQiyasNumber<TN, double>
        {
            var d = new double[count];
            TN.Load(stored, 0, d);
            return d;
        }
    }

    private sealed class ValuesViaFloat(byte[] stored, int count) : ITypeVisitor<double[]>
    {
        public double[] Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
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
        public byte[] Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            var random = new Random(seed);
            var bytes = new byte[((long)count * TN.Bits + 7) / 8];
            bool signed = TN.Min < 0;
            (double lo, double hi) = TN.IsFloat ? (low, high) : TN.Bits == 2 ? (signed ? (-1, 0) : (0, 1)) : signed ? (-2, 2) : (0, 3);
            var f = new float[count];
            for (int i = 0; i < count; i++) f[i] = TN.IsFloat ? (float)(lo + (hi - lo) * random.NextDouble()) : random.Next((int)lo, (int)hi + 1);
            TN.Store(f, bytes, 0);
            return bytes;
        }
    }
}
