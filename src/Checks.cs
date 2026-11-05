// Proof before timing: every vector conversion against an independent scalar reference written from the format's
// definition (IEEE 754 binary16 and binary32, bfloat16, the OCP fp8 formats, two's complement; rounding to nearest,
// ties to even), on every fp8 code, every 2- and 4-bit pattern, 4M random float32 bit patterns (every exponent, NaNs
// and subnormals included) and the special values; the dot lanes against scalar sums; the vector exp against the
// runtime's scalar exp. The timing does not start until every line here passes.
using System.Numerics;
using System.Runtime.InteropServices;

namespace Qiyas;

internal static class Checks
{
    public static bool Run()
    {
        const int N = 1 << 22;
        var random = new Random(7);
        var floats = new float[N];
        var bits = MemoryMarshal.Cast<float, uint>(floats.AsSpan());
        for (int i = 0; i < N; i++) bits[i] = (uint)random.NextInt64(0, 1L << 32);
        float[] special = [0f, -0f, float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue, float.PositiveInfinity, float.NegativeInfinity, float.NaN,
            65504f, 65519.99f, 65520f, 448f, 456f, 464f, 464.1f, 480f, 57344f, 61439f, 61440f, 61441f, 1.5f / 512, 2.5f / 512, 1f / 64, 3f / 131072];
        special.CopyTo(floats, 0);
        for (int i = 0; i < 4096; i++) floats[special.Length + i] = (float)(random.NextDouble() * 1000 - 500);   // ordinary magnitudes too

        bool ok = true;
        ok &= Encoded("float32 → float16", floats, 2, (s, d) => Cvt.ToHalf(s, MemoryMarshal.Cast<byte, ushort>(d)), (f, d) => BitConverter.TryWriteBytes(d, BitConverter.HalfToUInt16Bits((Half)f)));
        ok &= Encoded("float32 → bfloat16", floats, 2, (s, d) => Cvt.ToBFloat16(s, MemoryMarshal.Cast<byte, ushort>(d)), (f, d) => BitConverter.TryWriteBytes(d, Reference.BFloat16(f)));
        ok &= Encoded("float32 → fp8 e4m3", floats, 1, Cvt.E4M3FromFloat, (f, d) => d[0] = Reference.Fp8(f, e4m3: true));
        ok &= Encoded("float32 → fp8 e5m2", floats, 1, Cvt.E5M2FromFloat, (f, d) => d[0] = Reference.Fp8(f, e4m3: false));

        // Decoding every 16-bit and 8-bit code (bits compared: NaNs too).
        var codes16 = new ushort[65536 * 4];
        for (int i = 0; i < codes16.Length; i++) codes16[i] = (ushort)i;
        ok &= Decoded("float16 → float32", codes16.Length, d => Cvt.FromHalf(codes16, d), i => (float)BitConverter.UInt16BitsToHalf(codes16[i]));
        ok &= Decoded("bfloat16 → float32", codes16.Length, d => Cvt.FromBFloat16(codes16, d), i => BitConverter.UInt32BitsToSingle((uint)codes16[i] << 16));
        var codes = new byte[256 * 64];
        for (int i = 0; i < codes.Length; i++) codes[i] = (byte)i;
        ok &= Decoded("fp8 e4m3 → float32", codes.Length, d => Cvt.E4M3ToFloat(codes, d), i => Reference.Fp8Value(codes[i], e4m3: true));
        ok &= Decoded("fp8 e5m2 → float32", codes.Length, d => Cvt.E5M2ToFloat(codes, d), i => Reference.Fp8Value(codes[i], e4m3: false));

        // Packed integers: every byte pattern unpacked, and values (including past the range) packed.
        var bytes = new byte[256 * 64];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i * 37 + i / 256);
        var values = new float[1 << 16];
        for (int i = 0; i < values.Length; i++) values[i] = (float)Math.Round(random.NextDouble() * 24 - 12) + (i % 7 == 0 ? 0.5f : 0);
        ok &= Packed<Int4>("int4", bytes, values, 4, signed: true);
        ok &= Packed<Int2>("int2", bytes, values, 2, signed: true);
        ok &= Packed<UInt2>("uint2", bytes, values, 2, signed: false);

        // Rounding and holding on the way into integers (ties to even: 0.5 → 0, 1.5 → 2, 2.5 → 2).
        var mixed = new float[1 << 16];
        for (int i = 0; i < mixed.Length; i++) mixed[i] = i % 5 == 0 ? (i / 5 % 64) - 32 + 0.5f : (float)(random.NextDouble() * 600 - 300);
        ok &= Rounded<sbyte>("float32 → int8", mixed, sbyte.MinValue, sbyte.MaxValue);
        ok &= Rounded<byte>("float32 → uint8", mixed, 0, byte.MaxValue);
        ok &= Rounded<short>("float32 → int16", mixed, short.MinValue, short.MaxValue);
        ok &= Rounded<ushort>("float32 → uint16", mixed, 0, ushort.MaxValue);
        ok &= Rounded<int>("float32 → int32", mixed, int.MinValue, int.MaxValue);
        ok &= Rounded<long>("float32 → int64", mixed, long.MinValue, long.MaxValue);

        // Integer widths both ways (narrowing held to the target's range).
        var longs = new long[1 << 16];
        for (int i = 0; i < longs.Length; i++) longs[i] = i % 3 == 0 ? random.NextInt64(long.MinValue, long.MaxValue) : random.NextInt64(-300, 300);
        ok &= Resized<long, sbyte>("int64 → int8 (held)", longs);
        ok &= Resized<long, short>("int64 → int16 (held)", longs);
        ok &= Resized<long, int>("int64 → int32 (held)", longs);
        var ints = new int[1 << 16];
        for (int i = 0; i < ints.Length; i++) ints[i] = i % 3 == 0 ? random.Next(int.MinValue, int.MaxValue) : random.Next(-300, 300);
        ok &= Resized<int, sbyte>("int32 → int8 (held)", ints);
        ok &= Resized<int, short>("int32 → int16 (held)", ints);
        ok &= Resized<int, long>("int32 → int64", ints);
        var sbytes = new sbyte[1 << 16];
        for (int i = 0; i < sbytes.Length; i++) sbytes[i] = (sbyte)random.Next(-128, 128);
        ok &= Resized<sbyte, short>("int8 → int16", sbytes);
        ok &= Resized<sbyte, int>("int8 → int32", sbytes);
        ok &= Resized<sbyte, long>("int8 → int64", sbytes);
        var ulongs = new ulong[1 << 16];
        for (int i = 0; i < ulongs.Length; i++) ulongs[i] = i % 3 == 0 ? (ulong)random.NextInt64() * 2 : (ulong)random.Next(0, 600);
        ok &= Resized<ulong, byte>("uint64 → uint8 (held)", ulongs);
        ok &= Resized<ulong, ushort>("uint64 → uint16 (held)", ulongs);
        ok &= Resized<ulong, uint>("uint64 → uint32 (held)", ulongs);
        var ubytes = new byte[1 << 16];
        random.NextBytes(ubytes);
        ok &= Resized<byte, uint>("uint8 → uint32", ubytes);
        ok &= Resized<byte, ulong>("uint8 → uint64", ubytes);

        // Integers to floats (exact), and 64-bit values to float32 (one rounding).
        ok &= Exact<long, double>("int64 → float64", longs, v => v);
        ok &= Exact<long, float>("int64 → float32", longs, v => (float)v);
        ok &= Exact<ulong, float>("uint64 → float32", ulongs, v => (float)v);
        ok &= Exact<int, double>("int32 → float64", ints, v => v);
        ok &= Exact<sbyte, float>("int8 → float32", sbytes, v => v);
        ok &= Exact<sbyte, double>("int8 → float64", sbytes, v => v);

        // The dot lanes against scalar sums, every supported one.
        ok &= Dot<S8Dot, sbyte, int>("s8 dot", random);
        ok &= Dot<U8Dot, byte, uint>("u8 dot", random);
        ok &= Dot<S16Dot, short, int>("s16 dot", random);
        ok &= Dot<U16Dot, ushort, uint>("u16 dot", random);

        // exp on vectors against the runtime's scalar exp, in units of the last place.
        ok &= Exp32(random);
        ok &= Exp64(random);
        return ok;
    }

    // ---------------------------------------------------------------- the device's conversions against the CPU's, bit for bit
    // (a NaN against a NaN counts as the same: payloads are the device's own)

    public static bool Gpu(GpuOps ops)
    {
        bool ok = true;
        ok &= Device<Float64, double>(ops, Lane.F64);
        ok &= Device<Float32, float>(ops, Lane.F32);
        ok &= Device<Float16, float>(ops, Lane.F32);
        ok &= Device<BFloat16, float>(ops, Lane.F32);
        ok &= Device<Float8E4M3, float>(ops, Lane.F32);
        ok &= Device<Float8E5M2, float>(ops, Lane.F32);
        ok &= Device<Int64, float>(ops, Lane.F32);
        ok &= Device<Int32, float>(ops, Lane.F32);
        ok &= Device<Int16, float>(ops, Lane.F32);
        ok &= Device<Int8, float>(ops, Lane.F32);
        ok &= Device<Int4, float>(ops, Lane.F32);
        ok &= Device<Int2, float>(ops, Lane.F32);
        ok &= Device<UInt64, float>(ops, Lane.F32);
        ok &= Device<UInt32, float>(ops, Lane.F32);
        ok &= Device<UInt16, float>(ops, Lane.F32);
        ok &= Device<UInt8, float>(ops, Lane.F32);
        ok &= Device<UInt2, float>(ops, Lane.F32);
        ok &= Device<Int32, int>(ops, Lane.S32);
        ok &= Device<Int8, int>(ops, Lane.S32);
        ok &= Device<UInt8, uint>(ops, Lane.U32);
        ok &= Device<Int64, double>(ops, Lane.F64);
        ok &= Device<Int16, double>(ops, Lane.F64);
        return ok;
    }

    private static bool Device<TN, TL>(GpuOps ops, Lane lane) where TN : struct, IQiyasNumber<TN>, IQiyasNumber<TN, TL> where TL : unmanaged
    {
        const int N = 1 << 16;
        var random = new Random(11);
        var stored = new byte[((long)N * TN.Bits + 7) / 8];
        random.NextBytes(stored);
        var gpu = ops.Gpu;
        ulong dStored = gpu.Rent(stored.Length), dLanes = gpu.Rent((long)N * lane.Size);
        Qiyas.Gpu.Upload<byte>(stored, dStored);
        ops.ToLanes<TN>(lane, dStored, dLanes, N);
        var got = new TL[N];
        Qiyas.Gpu.Download<TL>(dLanes, got);
        var want = new TL[N];
        TN.Load(stored, 0, want);
        bool ok = Same($"device {TN.Name} → {lane.Name}", MemoryMarshal.AsBytes(got.AsSpan()), MemoryMarshal.AsBytes(want.AsSpan()), lane.Size, lane.IsFloat);

        // Back: lane values of the type's magnitude (floats: around its range; integers: within and past it).
        var lanes = new TL[N];
        for (int i = 0; i < N; i++)
        {
            double v = TN.IsFloat ? (random.NextDouble() * 2 - 1) * Math.Min(TN.Max, 1e5) * (i % 3 == 0 ? 1e-3 : 1) : Math.Round(random.NextDouble() * 600 - 300) + (i % 7 == 0 ? 0.5 : 0);
            if (TN.IsFloat && i % 97 == 0) v = i % 194 == 0 ? double.NaN : double.PositiveInfinity;
            if (typeof(TL) == typeof(float)) lanes[i] = (TL)(object)(float)v;
            else if (typeof(TL) == typeof(double)) lanes[i] = (TL)(object)v;
            else if (typeof(TL) == typeof(int)) lanes[i] = (TL)(object)(int)v;
            else lanes[i] = (TL)(object)(uint)Math.Max(0, v);
        }

        Qiyas.Gpu.Upload<TL>(lanes, dLanes);
        Qiyas.Gpu.Clear(dStored, stored.Length);
        ops.FromLanes<TN>(lane, dLanes, dStored, N);
        var gotStored = new byte[stored.Length];
        Qiyas.Gpu.Download<byte>(dStored, gotStored);
        var wantStored = new byte[stored.Length];
        TN.Store(lanes, wantStored, 0);
        if (TN.IsFloat)
        {
            // Compared as values (a stored NaN's payload is the device's own).
            var gotValues = new TL[N];
            var wantValues = new TL[N];
            TN.Load(gotStored, 0, gotValues);
            TN.Load(wantStored, 0, wantValues);
            ok &= Same($"device {lane.Name} → {TN.Name}", MemoryMarshal.AsBytes(gotValues.AsSpan()), MemoryMarshal.AsBytes(wantValues.AsSpan()), lane.Size, true);
        }
        else ok &= Same($"device {lane.Name} → {TN.Name}", gotStored, wantStored, 1, false);
        gpu.Return(dStored);
        gpu.Return(dLanes);
        return ok;
    }

    private static bool Same(string name, ReadOnlySpan<byte> got, ReadOnlySpan<byte> want, int size, bool floats)
    {
        int bad = 0, firstBad = -1;
        for (int i = 0; i < got.Length; i += size)
        {
            if (got.Slice(i, size).SequenceEqual(want.Slice(i, size))) continue;
            if (floats && size == 4 && float.IsNaN(BitConverter.ToSingle(got.Slice(i, 4))) && float.IsNaN(BitConverter.ToSingle(want.Slice(i, 4)))) continue;
            if (floats && size == 8 && double.IsNaN(BitConverter.ToDouble(got.Slice(i, 8))) && double.IsNaN(BitConverter.ToDouble(want.Slice(i, 8)))) continue;
            if (bad++ == 0) firstBad = i / size;
        }

        string detail = bad == 0 ? "" : $" (first at {firstBad}: got {Convert.ToHexString(got.Slice(firstBad * size, size))}, the CPU {Convert.ToHexString(want.Slice(firstBad * size, size))})";
        Console.WriteLine($"check {name,-36} {got.Length / size,9} values: {(bad == 0 ? "identical to the CPU, bit for bit" : $"{bad} DIFFER{detail}")}");
        return bad == 0;
    }

    private delegate void Encoder(ReadOnlySpan<float> s, Span<byte> d);

    private delegate void ScalarEncoder(float f, Span<byte> d);

    private static bool Encoded(string name, float[] floats, int size, Encoder vector, ScalarEncoder reference)
    {
        var got = new byte[floats.Length * size];
        var want = new byte[floats.Length * size];
        vector(floats, got);
        for (int i = 0; i < floats.Length; i++) reference(floats[i], want.AsSpan(i * size, size));
        return Report(name, got, want, size, floats);
    }

    private static bool Decoded(string name, int count, Action<float[]> vector, Func<int, float> reference)
    {
        var got = new float[count];
        var want = new float[count];
        vector(got);
        for (int i = 0; i < count; i++) want[i] = reference(i);
        return Report(name, MemoryMarshal.AsBytes(got.AsSpan()).ToArray(), MemoryMarshal.AsBytes(want.AsSpan()).ToArray(), 4, null);
    }

    private static bool Rounded<T>(string name, float[] values, double low, double high) where T : unmanaged
    {
        var got = new T[values.Length];
        var want = new T[values.Length];
        Qiyas.Widths.Store<float, T>(values, got);
        for (int i = 0; i < values.Length; i++) want[i] = Qiyas.Widths.FromLong<T>((long)Math.Clamp(Math.Round(values[i], MidpointRounding.ToEven), low, high));
        return Report(name, MemoryMarshal.AsBytes(got.AsSpan()).ToArray(), MemoryMarshal.AsBytes(want.AsSpan()).ToArray(), System.Runtime.CompilerServices.Unsafe.SizeOf<T>(), values);
    }

    private static bool Resized<TS, TD>(string name, TS[] values) where TS : unmanaged where TD : unmanaged
    {
        var got = new TD[values.Length];
        var want = new TD[values.Length];
        bool narrowing = System.Runtime.CompilerServices.Unsafe.SizeOf<TD>() < System.Runtime.CompilerServices.Unsafe.SizeOf<TS>();
        if (narrowing) Qiyas.Widths.Store(values, got); else Qiyas.Widths.Load(values, got);
        for (int i = 0; i < values.Length; i++) want[i] = Reference.Resize<TS, TD>(values[i]);
        return Report(name, MemoryMarshal.AsBytes(got.AsSpan()).ToArray(), MemoryMarshal.AsBytes(want.AsSpan()).ToArray(), System.Runtime.CompilerServices.Unsafe.SizeOf<TD>(), null);
    }

    private static bool Exact<TS, TF>(string name, TS[] values, Func<TS, TF> reference) where TS : unmanaged where TF : unmanaged
    {
        var got = new TF[values.Length];
        var want = new TF[values.Length];
        Qiyas.Widths.Load(values, got);
        for (int i = 0; i < values.Length; i++) want[i] = reference(values[i]);
        return Report(name, MemoryMarshal.AsBytes(got.AsSpan()).ToArray(), MemoryMarshal.AsBytes(want.AsSpan()).ToArray(), System.Runtime.CompilerServices.Unsafe.SizeOf<TF>(), null);
    }

    private static bool Packed<TN>(string name, byte[] bytes, float[] values, int bitsPer, bool signed) where TN : struct, IQiyasNumber<TN, float>
    {
        int count = bytes.Length * 8 / bitsPer;
        var got = new float[count];
        var want = new float[count];
        TN.Load(bytes, 0, got);
        for (int i = 0; i < count; i++) want[i] = Reference.Unpack(bytes, i, bitsPer, signed);
        bool ok = Report(name + " → float32", MemoryMarshal.AsBytes(got.AsSpan()).ToArray(), MemoryMarshal.AsBytes(want.AsSpan()).ToArray(), 4, null);
        var packedGot = new byte[values.Length * bitsPer / 8];
        var packedWant = new byte[packedGot.Length];
        TN.Store(values, packedGot, 0);
        for (int i = 0; i < values.Length; i++) Reference.Pack(packedWant, i, values[i], bitsPer, signed);
        return ok & Report("float32 → " + name, packedGot, packedWant, 1, null);
    }

    private static bool Dot<TD, TIn, TAcc>(string name, Random random) where TD : struct, IQiyasDotLane<TD, TIn, TAcc> where TIn : unmanaged where TAcc : unmanaged
    {
        if (!TD.IsSupported)
        {
            Console.WriteLine($"check {name,-36} not on this processor: never chosen");
            return true;
        }

        int w = Vector<TAcc>.Count, g = System.Runtime.CompilerServices.Unsafe.SizeOf<TAcc>() / System.Runtime.CompilerServices.Unsafe.SizeOf<TIn>();
        var a = new TAcc[w];
        var b = new TAcc[w];
        var c = new TAcc[w];
        var ab = MemoryMarshal.AsBytes(a.AsSpan());
        var bb = MemoryMarshal.AsBytes(b.AsSpan());
        random.NextBytes(ab);
        random.NextBytes(bb);
        random.NextBytes(MemoryMarshal.AsBytes(c.AsSpan()));
        var got = TD.Fma(new Vector<TAcc>(a), new Vector<TAcc>(b), new Vector<TAcc>(c));
        var want = new TAcc[w];
        var ins = MemoryMarshal.Cast<TAcc, TIn>(a.AsSpan());
        var inb = MemoryMarshal.Cast<TAcc, TIn>(b.AsSpan());
        for (int l = 0; l < w; l++)
        {
            long sum = Qiyas.Widths.AsLong(c[l]);
            for (int t = 0; t < g; t++) sum += Qiyas.Widths.AsLong(ins[l * g + t]) * Qiyas.Widths.AsLong(inb[l * g + t]);
            want[l] = Qiyas.Widths.FromLong<TAcc>(sum);
        }

        var gotLanes = new TAcc[w];
        got.CopyTo(gotLanes);
        return Report(name, MemoryMarshal.AsBytes(gotLanes.AsSpan()).ToArray(), MemoryMarshal.AsBytes(want.AsSpan()).ToArray(), System.Runtime.CompilerServices.Unsafe.SizeOf<TAcc>(), null);
    }

    private static bool Exp32(Random random)
    {
        int w = Vector<float>.Count, worstAt = 0;
        double worst = 0;
        var x = new float[w * 4096];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(random.NextDouble() * 200 - 110);
        for (int i = 0; i < x.Length; i += w)
        {
            var y = Exps.Exp(new Vector<float>(x.AsSpan(i)));
            for (int l = 0; l < w; l++)
            {
                double want = Math.Exp(x[i + l]);
                float wantF = (float)want;
                double ulp = Math.Abs(MathF.BitIncrement(Math.Abs(wantF)) - Math.Abs(wantF));
                double err = wantF == 0 ? (y[l] == 0 ? 0 : 1) : Math.Abs(y[l] - want) / ulp;
                if (err > worst) { worst = err; worstAt = i + l; }
            }
        }

        Console.WriteLine($"check {"exp f32 (ulps)",-36} {x.Length,9} values: worst {worst:0.00} ulp at x = {x[worstAt]:R}{(worst <= 1.5 ? "" : "  TOO FAR")}");
        return worst <= 1.5;
    }

    private static bool Exp64(Random random)
    {
        int w = Vector<double>.Count, worstAt = 0;
        double worst = 0;
        var x = new double[w * 4096];
        for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble() * 1500 - 760;
        for (int i = 0; i < x.Length; i += w)
        {
            var y = Exps.Exp(new Vector<double>(x.AsSpan(i)));
            for (int l = 0; l < w; l++)
            {
                double want = Math.Exp(x[i + l]);
                double ulp = Math.BitIncrement(Math.Abs(want)) - Math.Abs(want);
                double err = want == 0 ? (y[l] == 0 ? 0 : 1) : double.IsInfinity(want) ? (y[l] == want ? 0 : 1) : Math.Abs(y[l] - want) / ulp;
                if (err > worst) { worst = err; worstAt = i + l; }
            }
        }

        Console.WriteLine($"check {"exp f64 (ulps)",-36} {x.Length,9} values: worst {worst:0.00} ulp at x = {x[worstAt]:R}{(worst <= 2 ? "" : "  TOO FAR")}");
        return worst <= 2;
    }

    private static bool Report(string name, byte[] got, byte[] want, int size, float[]? inputs)
    {
        int bad = 0, firstBad = -1;
        for (int i = 0; i < got.Length; i += size)
        {
            if (!got.AsSpan(i, size).SequenceEqual(want.AsSpan(i, size)))
            {
                if (bad++ == 0) firstBad = i / size;
            }
        }

        string detail = bad == 0 ? "" : inputs is null
            ? $" (first at {firstBad}: got {Convert.ToHexString(got, firstBad * size, size)}, reference {Convert.ToHexString(want, firstBad * size, size)})"
            : $" (first at {firstBad}: input {inputs[firstBad]:R} = 0x{BitConverter.SingleToUInt32Bits(inputs[firstBad]):X8}, got {Convert.ToHexString(got, firstBad * size, size)}, reference {Convert.ToHexString(want, firstBad * size, size)})";
        Console.WriteLine($"check {name,-36} {got.Length / size,9} values: {(bad == 0 ? "identical to the reference, bit for bit" : $"{bad} DIFFER{detail}")}");
        return bad == 0;
    }
}

// Scalar references, each written from the format's definition, in double arithmetic where the format allows.
internal static class Reference
{
    // bfloat16: the nearest of the two 16-bit truncations, ties to the even one; NaN kept quiet.
    public static ushort BFloat16(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        if (float.IsNaN(f)) return (ushort)((bits >> 16) | 0x40);
        uint low = bits & 0xFFFF, truncated = bits >> 16;
        if (low > 0x8000 || (low == 0x8000 && (truncated & 1) == 1)) truncated++;
        return (ushort)truncated;
    }

    // fp8: the value of a code from its fields (E4M3: bias 7, 3 mantissa bits, 0x7F NaN, no infinity; E5M2: bias 15,
    // 2 mantissa bits, exponent 31 infinity and NaN).
    public static float Fp8Value(byte code, bool e4m3)
    {
        int sign = code >> 7, e, m, bias, mbits;
        if (e4m3) { e = (code >> 3) & 15; m = code & 7; bias = 7; mbits = 3; }
        else { e = (code >> 2) & 31; m = code & 3; bias = 15; mbits = 2; }
        float nan = BitConverter.UInt32BitsToSingle(0x7FC00000u | (uint)sign << 31);   // the quiet NaN of the code's sign
        if (e4m3 && (code & 0x7F) == 0x7F) return nan;
        if (!e4m3 && e == 31) return m != 0 ? nan : sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
        double value = e == 0 ? m * Math.Pow(2, 1 - bias - mbits) : (1 + m / (double)(1 << mbits)) * Math.Pow(2, e - bias);
        return (float)(sign == 1 ? -value : value);
    }

    // fp8 of a float: the nearest finite code's value (ties to the even code); past the last finite value's rounding
    // point E4M3's NaN code (it has no infinity) and E5M2's infinity; a NaN to 0x7F with its sign.
    public static byte Fp8(float f, bool e4m3)
    {
        byte sign = (byte)(float.IsNegative(f) ? 0x80 : 0);
        if (float.IsNaN(f)) return (byte)(sign | 0x7F);
        double x = Math.Abs(f);
        int last = e4m3 ? 0x7E : 0x7B;                                       // the largest finite code
        double max = Fp8Value((byte)last, e4m3), ulpMax = max - Fp8Value((byte)(last - 1), e4m3);
        if (x >= max + ulpMax / 2) return (byte)(sign | (e4m3 ? 0x7F : 0x7C));
        int best = 0;
        double bestDist = double.PositiveInfinity;
        for (int code = 0; code <= last; code++)
        {
            double v = Fp8Value((byte)code, e4m3), dist = Math.Abs(x - v);
            if (dist < bestDist || (dist == bestDist && (code & 1) == 0)) { bestDist = dist; best = code; }
        }

        return (byte)(sign | best);
    }

    public static float Unpack(byte[] bytes, int index, int bits, bool signed)
    {
        int per = 8 / bits, shift = index % per * bits, mask = (1 << bits) - 1;
        int v = (bytes[index / per] >> shift) & mask;
        if (signed && v >= 1 << (bits - 1)) v -= 1 << bits;
        return v;
    }

    public static void Pack(byte[] bytes, int index, float value, int bits, bool signed)
    {
        int per = 8 / bits, shift = index % per * bits, mask = (1 << bits) - 1;
        long low = signed ? -(1 << (bits - 1)) : 0, high = signed ? (1 << (bits - 1)) - 1 : mask;
        int v = (int)Math.Clamp(Math.Round(value, MidpointRounding.ToEven), low, high) & mask;
        bytes[index / per] = (byte)((bytes[index / per] & ~(mask << shift)) | v << shift);
    }

    public static TD Resize<TS, TD>(TS value) where TS : unmanaged where TD : unmanaged
    {
        bool unsigned = typeof(TS) == typeof(ulong);
        if (unsigned)
        {
            ulong v = System.Runtime.CompilerServices.Unsafe.As<TS, ulong>(ref value);
            ulong high = typeof(TD) == typeof(byte) ? byte.MaxValue : typeof(TD) == typeof(ushort) ? ushort.MaxValue : uint.MaxValue;
            return Qiyas.Widths.FromLong<TD>((long)Math.Min(v, high));
        }

        long x = Qiyas.Widths.AsLong(value);
        int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TD>();
        bool signed = typeof(TD) == typeof(sbyte) || typeof(TD) == typeof(short) || typeof(TD) == typeof(int) || typeof(TD) == typeof(long);
        if (size < System.Runtime.CompilerServices.Unsafe.SizeOf<TS>())
        {
            long low = signed ? -(1L << (8 * size - 1)) : 0, high = signed ? (1L << (8 * size - 1)) - 1 : (1L << (8 * size)) - 1;
            x = Math.Clamp(x, low, high);
        }

        return Qiyas.Widths.FromLong<TD>(x);
    }
}
