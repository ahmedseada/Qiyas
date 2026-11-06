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
        ok &= Function32("log f32", Funcs.Log, Math.Log, 1e-30, 1e30, 2, random, logScale: true);
        ok &= Function64("log f64", Funcs.Log, Math.Log, 1e-300, 1e300, 2, random, logScale: true);
        ok &= Function32("sin f32", Funcs.Sin, Math.Sin, -1000, 1000, 2, random);
        ok &= Function64("sin f64", Funcs.Sin, Math.Sin, -1000, 1000, 2, random);
        ok &= Function32("cos f32", Funcs.Cos, Math.Cos, -1000, 1000, 2, random);
        ok &= Function64("cos f64", Funcs.Cos, Math.Cos, -1000, 1000, 2, random);
        ok &= Function32("expm1 f32", Funcs.Expm1, Funcs.Expm1Reference, -1, 1, 2, random);
        ok &= Function64("expm1 f64", Funcs.Expm1, Funcs.Expm1Reference, -1, 1, 2, random);
        ok &= Function32("tanh f32", Funcs.Tanh, Math.Tanh, -10, 10, 3, random);
        ok &= Function64("tanh f64", Funcs.Tanh, Math.Tanh, -10, 10, 3, random);
        ok &= Function32("sigmoid f32", Funcs.Sigmoid, x => 1 / (1 + Math.Exp(-x)), -20, 20, 3, random);
        ok &= Function64("sigmoid f64", Funcs.Sigmoid, x => 1 / (1 + Math.Exp(-x)), -20, 20, 3, random);
        ok &= Function32("erf f32", Funcs.Erf, Funcs.ErfReference, -5, 5, 4, random);
        ok &= Function64("erf f64", Funcs.Erf, Funcs.ErfReference, -6, 6, 4, random);
        ok &= Function32("erfc f32", Funcs.Erfc, Funcs.ErfcReference, -4, 9, 4, random);
        ok &= Function64("erfc f64", Funcs.Erfc, Funcs.ErfcReference, -4, 26, 4, random);
        ok &= Maps(null);
        ok &= Group2(null);
        ok &= Rows(null);
        ok &= Layouts(null);
        ok &= Products(null);
        ok &= Attentions(null);
        ok &= Optimizers(null);
        ok &= Convs(null);
        ok &= Recurrents(null);
        ok &= Routings(null);
        ok &= Ctcs(null);
        ok &= BlockFormats(null);
        ok &= Arenas(null);
        ok &= WeightsInPlace();
        ok &= TeamJobs();
        return ok;
    }

    // ---------------------------------------------------------------- the thread team

    /// <summary>The team under a run of jobs of alternating chunk counts (a few chunks, then many): every chunk a worker runs
    /// lies within its job and every index runs exactly once. A worker late out of a job must never take a chunk of the
    /// next job's count under the old job's description (the fault once seen as an out-of-range product tile).</summary>
    public static bool TeamJobs()
    {
        int jobs = Stress * 500;
        var hits = new int[1 << 16];
        var bad = new int[1];
        bool ok = true;
        for (int j = 0; j < jobs && ok; j++)
        {
            // A job of a chunk per thread (every worker contending at its end), then one of many chunks.
            int count = (j & 1) == 0 ? 4 * Team.Size : hits.Length, size = (j & 1) == 0 ? 1 : 256;
            Array.Clear(hits, 0, count);
            Team.For(count, size, long.MaxValue, new TeamProbe(hits, bad, count));
            int missed = 0, twice = 0;
            for (int i = 0; i < count; i++) { if (hits[i] == 0) missed++; else if (hits[i] > 1) twice++; }
            if (bad[0] != 0 || missed != 0 || twice != 0)
            {
                Console.WriteLine($"check team                   cpu    job {j} of {count} in chunks of {size}: {bad[0]} chunks outside the job, {missed} indices missed, {twice} run twice");
                ok = false;
            }
        }

        if (ok) Console.WriteLine($"check team                   cpu    {jobs} jobs of alternating chunk counts: every chunk within its job, every index once");
        return ok;
    }

    private readonly struct TeamProbe(int[] hits, int[] bad, int count) : Team.IBody
    {
        public void Run(int first, int last)
        {
            if (first < 0 || last > count || first >= last) { Interlocked.Increment(ref bad[0]); return; }
            for (int i = first; i < last; i++) Interlocked.Increment(ref hits[i]);
        }
    }

    // ---------------------------------------------------------------- the block formats

    /// <summary>Q8_0, Q4_0, Q4_K, Q6_K: 1001 values (not whole blocks) encoded and decoded on the CPU within the block's step;
    /// the device's decode bit for bit against the CPU's; the device's encode (Q8_0, Q4_0) byte for byte against the CPU's.</summary>
    public static bool BlockFormats(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        if (ops is null) ok &= BlockSpecifications();
        var random = new Random(99);
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsBlock(type)) continue;
            const int n = 1001;
            var values = new double[n];
            for (int i = 0; i < n; i++) values[i] = (random.NextDouble() * 8 - 4) * (i % 7 == 0 ? 0.01 : 1);
            var stored = Bench.Registry.Encode(type, values);
            var back = Bench.Registry.Values(type, stored, n);
            int mant = Bench.Registry.Mantissa(type), blockValues = type is "q8_0" or "q4_0" ? 32 : 256;
            double worst = 0;
            for (int i = 0; i < n; i++)
            {
                double bmax = 0;
                for (int j = i / blockValues * blockValues; j < Math.Min(n, i / blockValues * blockValues + blockValues); j++) bmax = Math.Max(bmax, Math.Abs(values[j]));
                double step = bmax * Math.Pow(2, -mant) + 1e-30;
                worst = Math.Max(worst, Math.Abs(back[i] - values[i]) / step);
            }

            if (worst > 1.5) { Console.WriteLine($"check block {type,-6} round trip   cpu    DIFFERS by {worst:0.##} block steps"); ok = false; }
            // The core operations the bench times first (add over 1001 values; softmax and rms_norm over 37 × 1001) against the references.
            {
                const int R = 37, C = 1001;
                var a = Bench.Registry.Make(type, R * C, 61, -2, 2);
                var bb = Bench.Registry.Make(type, R * C, 62, -2, 2);
                var va = Bench.Registry.Values(type, a, R * C);
                var vb = Bench.Registry.Values(type, bb, R * C);
                var sum = new double[R * C];
                for (int i = 0; i < R * C; i++) sum[i] = va[i] + vb[i];
                // A fresh output each (a block format's partial-block store keeps the buffer's other values in the block's scale).
                var output = new byte[a.Length];
                Bench.Registry.WithFloat(type, new CoreRun(ops, "add", a, bb, output, R, C));
                ok &= Close(type, "add", where, output, sum, R * C);
                output = new byte[a.Length];
                Bench.Registry.WithFloat(type, new CoreRun(ops, "softmax", a, null, output, R, C));
                ok &= Close(type, "softmax", where, output, References.Softmax(va, R, C), R * C, terms: C);
                output = new byte[a.Length];
                Bench.Registry.WithFloat(type, new CoreRun(ops, "rms_norm", a, null, output, R, C));
                ok &= Close(type, "rms_norm", where, output, References.RmsNorm(va, R, C, 1e-5), R * C, terms: C);
            }

            if (ops is null) continue;
            var gpu = ops.Gpu;
            // The device decodes every value as the CPU does (its Widen kernel), and encodes Q8_0 / Q4_0 as the CPU does.
            ulong d = gpu.Rent(stored.Length), lanes = gpu.Rent(n * 4L);
            Qiyas.Gpu.Upload<byte>(stored, d);
            var lane = new float[n];
            Bench.Registry.WithFloat(type, new WidenRun(ops, d, lanes, n));
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<float>(lanes, lane);
            int bad = 0;
            for (int i = 0; i < n; i++) if (lane[i] != (float)back[i]) bad++;
            if (bad > 0) { Console.WriteLine($"check block {type,-6} decode       device DIFFERS at {bad} values"); ok = false; }
            {
                var f32 = new float[n];
                for (int i = 0; i < n; i++) f32[i] = (float)values[i];
                ulong src = gpu.Rent(n * 4L), enc = gpu.Rent(stored.Length);
                Qiyas.Gpu.Upload<float>(f32, src);
                Bench.Registry.WithFloat(type, new NarrowRun(ops, src, enc, n));
                Qiyas.Gpu.Synchronize();
                var encoded = new byte[stored.Length];
                Qiyas.Gpu.Download<byte>(enc, encoded);
                int diff = 0;
                for (int i = 0; i < stored.Length; i++) if (encoded[i] != stored[i]) diff++;
                if (diff > 0) { Console.WriteLine($"check block {type,-6} encode       device DIFFERS at {diff} bytes"); ok = false; }
                gpu.Return(src); gpu.Return(enc);
            }

            gpu.Return(d); gpu.Return(lanes);
        }

        if (ok) Console.WriteLine($"check block formats          {where,-6} Q8_0, Q4_0, Q4_K, Q6_K: round trips within the block's step; add, softmax, rms_norm as the references{(ops is null ? "" : "; the device's codecs as the CPU's")}");
        return ok;
    }

    /// <summary>The codecs against the formats' specifications (GGUF): blocks written by hand, their values by the specified formulas.</summary>
    private static bool BlockSpecifications()
    {
        bool ok = true;
        static void Half(float v, Span<byte> s, int at) => BitConverter.TryWriteBytes(s.Slice(at, 2), (Half)v);
        var v = new float[256];
        // Q8_0: d = 0.5, q_i = i − 16 → value_i = 0.5·(i − 16); byte 2 + i holds q_i.
        {
            var block = new byte[34];
            Half(0.5f, block, 0);
            for (int i = 0; i < 32; i++) block[2 + i] = (byte)(sbyte)(i - 16);
            Blocks.DecodeQ8_0(block, v);
            for (int i = 0; i < 32; i++) if (v[i] != 0.5f * (i - 16)) { ok = false; Console.WriteLine($"check block q8_0 spec: value {i} = {v[i]}, expected {0.5f * (i - 16)}"); break; }
            var back = new byte[34];
            Blocks.EncodeQ8_0(v.AsSpan(0, 32), back);
            float dEnc = (float)BitConverter.ToHalf(back.AsSpan(0, 2));
            Blocks.DecodeQ8_0(back, v);
            for (int i = 0; i < 32; i++) if (Math.Abs(v[i] - 0.5f * (i - 16)) > 0.5f * dEnc + 1e-6f) { ok = false; Console.WriteLine($"check block q8_0 spec: round trip value {i} = {v[i]} (scale {dEnc})"); break; }
        }

        // Q4_0: d = 0.25; byte 2 + j holds value j in its low nibble and value j + 16 in its high nibble; value = d·(q − 8).
        {
            var block = new byte[18];
            Half(0.25f, block, 0);
            for (int j = 0; j < 16; j++) block[2 + j] = (byte)(j | ((15 - j) << 4));
            Blocks.DecodeQ4_0(block, v);
            for (int j = 0; j < 16; j++)
            {
                if (v[j] != 0.25f * (j - 8) || v[j + 16] != 0.25f * (15 - j - 8)) { ok = false; Console.WriteLine($"check block q4_0 spec: values {j}, {j + 16} = {v[j]}, {v[j + 16]}"); break; }
            }
        }

        // Q4_K: d = 1, dmin = 0.5; sub-block scales 1..8 and mins 0..7 in the 6-bit layout; nibbles: sub-block j's value i = (i + j) & 15
        // (low nibbles for even j, high for odd, over the pair's 32 bytes); value = d·scale_j·q − dmin·min_j.
        {
            var block = new byte[144];
            Half(1f, block, 0); Half(0.5f, block, 2);
            var sc = block.AsSpan(4, 12);
            int[] ls = [1, 2, 3, 4, 5, 6, 7, 8], lm = [0, 1, 2, 3, 4, 5, 6, 7];
            for (int j = 0; j < 8; j++)
            {
                if (j < 4) { sc[j] = (byte)ls[j]; sc[j + 4] = (byte)lm[j]; }
                else { sc[j + 4] = (byte)((ls[j] & 0xF) | ((lm[j] & 0xF) << 4)); sc[j - 4] |= (byte)((ls[j] >> 4) << 6); sc[j] |= (byte)((lm[j] >> 4) << 6); }
            }

            var qs = block.AsSpan(16, 128);
            for (int j = 0; j < 8; j++) for (int i = 0; i < 32; i++)
            {
                int q = (i + j) & 15, at = j / 2 * 32 + i;
                if (j % 2 == 0) qs[at] = (byte)((qs[at] & 0xF0) | q); else qs[at] = (byte)((qs[at] & 0x0F) | q << 4);
            }

            Blocks.DecodeQ4_K(block, v);
            for (int j = 0; j < 8 && ok; j++) for (int i = 0; i < 32; i++)
            {
                float want = 1f * ls[j] * ((i + j) & 15) - 0.5f * lm[j];
                if (v[j * 32 + i] != want) { ok = false; Console.WriteLine($"check block q4_k spec: sub-block {j} value {i} = {v[j * 32 + i]}, expected {want}"); break; }
            }
        }

        // Q6_K: d = 0.125; scales_j = j − 8 (int8); value i of sub-block j (16 values) = d·scale_j·(q − 32) with q = (i·3 + j) & 63 laid into ql / qh.
        {
            var block = new byte[210];
            Half(0.125f, block, 208);
            var sc = block.AsSpan(192, 16);
            for (int j = 0; j < 16; j++) sc[j] = (byte)(sbyte)(j - 8);
            var q = new int[256];
            for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) q[j * 16 + i] = (i * 3 + j) & 63;
            var ql = block.AsSpan(0, 128);
            var qh = block.AsSpan(128, 64);
            for (int half = 0; half < 2; half++)
            {
                int lBase = half * 64, hBase = half * 32, vBase = half * 128;
                for (int l = 0; l < 32; l++)
                {
                    int q1 = q[vBase + l], q2 = q[vBase + l + 32], q3 = q[vBase + l + 64], q4 = q[vBase + l + 96];
                    ql[lBase + l] = (byte)((q1 & 0xF) | ((q3 & 0xF) << 4));
                    ql[lBase + l + 32] = (byte)((q2 & 0xF) | ((q4 & 0xF) << 4));
                    qh[hBase + l] = (byte)(((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6));
                }
            }

            Blocks.DecodeQ6_K(block, v);
            for (int j = 0; j < 16 && ok; j++) for (int i = 0; i < 16; i++)
            {
                float want = 0.125f * (j - 8) * (((i * 3 + j) & 63) - 32);
                if (v[j * 16 + i] != want) { ok = false; Console.WriteLine($"check block q6_k spec: sub-block {j} value {i} = {v[j * 16 + i]}, expected {want}"); break; }
            }
        }

        if (ok) Console.WriteLine("check block specifications   cpu    Q8_0, Q4_0, Q4_K, Q6_K blocks laid out by hand decode by the formats' formulas");
        return ok;
    }

    private sealed class CoreRun(GpuOps? ops, string what, byte[] x, byte[]? x2, byte[] output, int rows, int cols) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            int n = rows * cols;
            if (ops is null)
            {
                if (what == "add") { var add = new Kernels.AddVisitor(x, x2!, output, n); TN.Elementwise(ref add); }
                else if (what == "softmax") { var s = new Kernels.SoftmaxVisitor(x, output, rows, cols); TN.Float(ref s); }
                else { var r = new Kernels.RmsNormVisitor(x, output, rows, cols, 1e-5); TN.Float(ref r); }
                return true;
            }

            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dx2 = x2 is null ? 0 : gpu.Rent(x2.Length), dy = gpu.Rent(output.Length);
            try
            {
                Qiyas.Gpu.Upload<byte>(x, dx);
                if (x2 is not null) Qiyas.Gpu.Upload<byte>(x2, dx2);
                if (what == "add") ops.Add<TN>(dx, dx2, dy, n);
                else if (what == "softmax") ops.Softmax<TN>(dx, dy, rows, cols);
                else ops.RmsNorm<TN>(dx, dy, rows, cols, 1e-5);
                Qiyas.Gpu.Synchronize();
                Qiyas.Gpu.Download<byte>(dy, output);
                return true;
            }
            finally { gpu.Return(dx); if (dx2 != 0) gpu.Return(dx2); gpu.Return(dy); }
        }
    }

    private sealed class WidenRun(GpuOps ops, ulong stored, ulong lanes, int n) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float> { ops.ToLanes<TN>(Lane.F32, stored, lanes, n); return true; }
    }

    private sealed class NarrowRun(GpuOps ops, ulong lanes, ulong stored, int n) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float> { ops.FromLanes<TN>(Lane.F32, lanes, stored, n); return true; }
    }

    // ---------------------------------------------------------------- routing, sampling and the loss on odd shapes

    /// <summary>Every type: top-k (k = 5 of 1001, ties included) exact; pick exact; sampling (float types) exact against
    /// the float64 draw except where the lane's rounding of the running sum crosses the threshold (allowed: the next
    /// column); the route of 1001 indices into 7 buckets exact.</summary>
    public static bool Routings(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        const int rows = 37, cols = 1001, k = 5;
        var random = new Random(98);
        foreach (var type in Bench.Registry.Names)
        {
            var x = Bench.Registry.Make(type, rows * cols, 95, -4, 4);
            var xv = Bench.Registry.Values(type, x, rows * cols);
            var (wantV, wantI) = Routing.TopKReference(xv, rows, cols, k);
            var values = new byte[Bench.Registry.Bytes(type, rows * k)];
            var indices = new int[rows * k];
            Bench.Registry.With(type, new TopKRun(ops, x, rows, cols, k, values, indices, null, null, null, null));
            // Ties (the narrow types have many): the values must match exactly; the indices where the values are distinct.
            ok &= Close(type, "topk values", where, values, wantV, rows * k);
            int bad = 0;
            for (int i = 0; i < rows * k; i++) if (indices[i] != wantI[i] && xv[i / k * cols + indices[i]] != wantV[i]) bad++;
            if (bad > 0) { Console.WriteLine($"check topk indices  {where,-6} {type,-12} DIFFERS at {bad} places"); ok = false; }
            var index = Enumerable.Range(0, rows).Select(_ => random.Next(cols)).ToArray();
            var picked = new byte[Bench.Registry.Bytes(type, rows)];
            Bench.Registry.With(type, new TopKRun(ops, x, rows, cols, 0, null, null, index, picked, null, null));
            ok &= Close(type, "pick", where, picked, Routing.PickReference(xv, rows, cols, index), rows);
            if (Bench.Registry.IsFloat(type))
            {
                var w = Bench.Registry.Make(type, rows * cols, 95, 0, 1);
                var u = Bench.Registry.Make(type, rows, 96, 0, 1);
                var wantS = Routing.SampleReference(Bench.Registry.Values(type, w, rows * cols), rows, cols, Bench.Registry.Values(type, u, rows));
                var picks = new int[rows];
                Bench.Registry.WithFloat(type, new SampleRun(ops, w, rows, cols, u, picks));
                int off = 0;
                for (int r = 0; r < rows; r++) if (Math.Abs(picks[r] - wantS[r]) > 1) off++;
                if (off > 0) { Console.WriteLine($"check sample        {where,-6} {type,-12} DIFFERS at {off} rows by more than one column"); ok = false; }
                // The cross-entropy loss per row: the type's step at the loss plus the lane's rounding of a sum of cols terms.
                var loss = new byte[Bench.Registry.Bytes(type, rows)];
                using (var ce = new Bench.QiyasSubject().CrossEntropy(ops is null ? Bench.Device.Cpu : Bench.Device.Cuda, type, x, rows, cols, index, loss)!) { ce.Run(); ce.Collect(); }
                ok &= Close(type, "cross_entropy", where, loss, Routing.CrossEntropyReference(xv, rows, cols, index), rows, terms: cols);
            }
        }

        {
            var index = Enumerable.Range(0, 1001).Select(_ => random.Next(7)).ToArray();
            var (wc, wo, worder) = Routing.RouteReference(index, 7);
            var counts = new int[7];
            var offsets = new int[8];
            var order = new int[1001];
            var subject = new Bench.QiyasSubject();
            using var run = subject.Route(ops is null ? Bench.Device.Cpu : Bench.Device.Cuda, index, 7, counts, offsets, order)!;
            run.Run(); run.Collect();
            if (!counts.SequenceEqual(wc) || !offsets.SequenceEqual(wo) || !order.SequenceEqual(worder)) { Console.WriteLine($"check route         {where,-6} DIFFERS"); ok = false; }
        }

        // The ordering at size (item 4f: the device's radix sort and multi-block scan): order, counts and offsets equal to
        // the stable reference — many sources into few targets, into many (three radix passes, a scan over 98 blocks),
        // all into one, and descending indices (every tile's pairs reversed).
        // Under --repeat the device's cases run Stress / 20 passes (the radix place's barriers and shared counts: a race shows).
        for (int pass = 0, passes = ops is null ? 1 : Math.Max(1, Stress / 20); pass < passes && ok; pass++)
        foreach (var (n, buckets, kind) in new[] { (1 << 20, 64, "random"), (1_000_003, 100_000, "random"), (5_000, 1, "one"), (70_001, 65_536, "descending") })
        {
            var index = kind switch
            {
                "one" => new int[n],
                "descending" => Enumerable.Range(0, n).Select(i => (n - 1 - i) % buckets).ToArray(),
                _ => Enumerable.Range(0, n).Select(_ => random.Next(buckets)).ToArray(),
            };
            var (wc, wo, worder) = Routing.RouteReference(index, buckets);
            var counts = new int[buckets];
            var offsets = new int[buckets + 1];
            var order = new int[n];
            using var run = new Bench.QiyasSubject().Route(ops is null ? Bench.Device.Cpu : Bench.Device.Cuda, index, buckets, counts, offsets, order)!;
            run.Run(); run.Collect();
            if (!counts.SequenceEqual(wc) || !offsets.SequenceEqual(wo) || !order.SequenceEqual(worder)) { Console.WriteLine($"check route {n} into {buckets} ({kind}) {where,-6} DIFFERS"); ok = false; }
        }

        if (ok) Console.WriteLine($"check routing                {where,-6} every type: top-k, pick, sampling and the cross-entropy loss (float types) and the route as the references (also 1M into 64, 1 000 003 into 100 000, all into one, descending into 65 536)");
        return ok;
    }

    private sealed class TopKRun(GpuOps? ops, byte[] x, int rows, int cols, int k, byte[]? values, int[]? indices, int[]? index, byte[]? picked, object? u1, object? u2) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            var subject = new Bench.QiyasSubject();
            var device = ops is null ? Bench.Device.Cpu : Bench.Device.Cuda;
            using var run = values is not null ? subject.TopK(device, TN.Name, x, rows, cols, k, values, indices!)! : subject.Pick(device, TN.Name, x, rows, cols, index!, picked!)!;
            run.Run(); run.Collect();
            return true;
        }
    }

    private sealed class SampleRun(GpuOps? ops, byte[] x, int rows, int cols, byte[] u, int[] picks) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            var subject = new Bench.QiyasSubject();
            using var run = subject.Sample(ops is null ? Bench.Device.Cpu : Bench.Device.Cuda, TN.Name, x, rows, cols, u, picks)!;
            run.Run(); run.Collect();
            return true;
        }
    }

    // ---------------------------------------------------------------- recurrent steps on odd shapes

    /// <summary>Every float type: LSTM and GRU steps over 37 rows (input 29, hidden 53) and 5 rows (input 7, hidden 11), forward
    /// (h', c') and backward (dx, dh, dc, dW, dU, db, dbh) against float64 within the products' rounding.</summary>
    public static bool Recurrents(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            foreach (var kind in new[] { "lstm", "gru" })
            {
                foreach (var (B, D, H) in new[] { (37, 29, 53), (5, 7, 11) })
                {
                    var a = new CellArgs(kind, B, D, H);
                    foreach (bool backward in new[] { false, true })
                    {
                        var k = Bench.RecurrentCase.Make(a, type, backward);
                        int GH = a.Gates * H;
                        if (!backward)
                        {
                            var hOut = new byte[k.H.Length];
                            Bench.Registry.WithFloat(type, new RecurrentRun(ops, a, k, hOut, null));
                            ok &= Close(type, $"{kind}_step h", where, hOut, k.WantH, B * H, D + H);
                            if (k.COut is not null) ok &= Close(type, $"{kind}_step c", where, k.COut, k.WantC!, B * H, D + H);
                            continue;
                        }

                        var dx = new byte[k.X.Length];
                        Bench.Registry.WithFloat(type, new RecurrentRun(ops, a, k, dx, null));
                        ok &= Close(type, $"{kind}_step_bwd dx", where, dx, k.WantDx!, B * D, 4 * (D + H + GH));
                        ok &= Close(type, $"{kind}_step_bwd dh", where, k.Dh!, k.WantDh!, B * H, 4 * (D + H + GH));
                        if (k.Dc is not null) ok &= Close(type, $"{kind}_step_bwd dc", where, k.Dc, k.WantDc!, B * H, 4 * (D + H));
                        ok &= Close(type, $"{kind}_step_bwd dw", where, k.Dw!, k.WantDw!, GH * D, 4 * (D + H + B));
                        ok &= Close(type, $"{kind}_step_bwd du", where, k.Du!, k.WantDu!, GH * H, 4 * (D + H + B));
                        ok &= Close(type, $"{kind}_step_bwd db", where, k.Db!, k.WantDb!, GH, 4 * (D + H + B));
                        if (k.Dbh is not null) ok &= Close(type, $"{kind}_step_bwd dbh", where, k.Dbh, k.WantDbh!, GH, 4 * (D + H + B));
                    }
                }
            }
        }

        // Sequences: 3 steps of 5 × 7 → 11, forward (h, c every step) and backward (dx, dh0, dc0, dW, dU, db), for the plain float types.
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || Bench.Registry.IsBlock(type)) continue;
            var a = new CellArgs("lstm", 5, 7, 11);
            const int T = 3;
            int GH = a.Gates * a.H;
            double scale = 1 / Math.Sqrt(a.D + a.H);
            var x = Bench.Registry.Make(type, T * a.B * a.D, 81, -1, 1); var h0 = Bench.Registry.Make(type, a.B * a.H, 82, -1, 1); var c0 = Bench.Registry.Make(type, a.B * a.H, 83, -1, 1);
            var w = Bench.Registry.Make(type, GH * a.D, 84, -scale, scale); var u = Bench.Registry.Make(type, GH * a.H, 85, -scale, scale); var b = Bench.Registry.Make(type, GH, 86, -0.1, 0.1);
            double[] V(byte[] s, int n) => Bench.Registry.Values(type, s, n);
            var (wantH, wantC) = Bench.RecurrentCase.LstmSequenceReference(a, T, V(x, T * a.B * a.D), V(h0, a.B * a.H), V(c0, a.B * a.H), V(w, GH * a.D), V(u, GH * a.H), V(b, GH));
            var hOut = new byte[Bench.Registry.Bytes(type, T * a.B * a.H)]; var cOut = new byte[hOut.Length];
            var subject = new Bench.QiyasSubject();
            var dev = ops is null ? Bench.Device.Cpu : Bench.Device.Cuda;
            using (var run = subject.LstmSequence(dev, type, a, T, x, h0, c0, w, u, b, hOut, cOut, null, null, null, null, null, null)!) { run.Run(); run.Collect(); }
            ok &= Close(type, "lstm_seq h", where, hOut, wantH, T * a.B * a.H, T * (a.D + a.H));
            ok &= Close(type, "lstm_seq c", where, cOut, wantC, T * a.B * a.H, T * (a.D + a.H));
            var dhOut = Bench.Registry.Make(type, T * a.B * a.H, 88, -1, 1); var dcLast = Bench.Registry.Make(type, a.B * a.H, 89, -1, 1);
            var (wdx, wdh0, wdc0, wdw, wdu, wdb) = Bench.RecurrentCase.LstmSequenceBackwardReference(a, T, V(x, T * a.B * a.D), V(h0, a.B * a.H), V(c0, a.B * a.H), V(w, GH * a.D), V(u, GH * a.H), V(b, GH), V(dhOut, T * a.B * a.H), V(dcLast, a.B * a.H));
            var dx = new byte[x.Length]; var dh0 = new byte[h0.Length]; var dc0 = new byte[c0.Length]; var dw = new byte[w.Length]; var du = new byte[u.Length]; var db = new byte[b.Length];
            using (var run = subject.LstmSequence(dev, type, a, T, x, h0, c0, w, u, b, dhOut, dcLast, dx, dh0, dc0, dw, du, db)!) { run.Run(); run.Collect(); }
            int terms = 8 * T * (a.D + a.H + GH + a.B);
            ok &= Close(type, "lstm_seq_bwd dx", where, dx, wdx, T * a.B * a.D, terms);
            ok &= Close(type, "lstm_seq_bwd dh0", where, dh0, wdh0, a.B * a.H, terms);
            ok &= Close(type, "lstm_seq_bwd dc0", where, dc0, wdc0, a.B * a.H, terms);
            ok &= Close(type, "lstm_seq_bwd dw", where, dw, wdw, GH * a.D, terms);
            ok &= Close(type, "lstm_seq_bwd du", where, du, wdu, GH * a.H, terms);
            ok &= Close(type, "lstm_seq_bwd db", where, db, wdb, GH, terms);
        }

        // GRU sequences: 3 steps of 5 × 7 → 11 (h every step), every float type, the block formats through lane storage.
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            var a = new CellArgs("gru", 5, 7, 11);
            const int T = 3;
            int GH = a.Gates * a.H;
            double scale = 1 / Math.Sqrt(a.D + a.H);
            var x = Bench.Registry.Make(type, T * a.B * a.D, 91, -1, 1); var h0 = Bench.Registry.Make(type, a.B * a.H, 92, -1, 1);
            var w = Bench.Registry.Make(type, GH * a.D, 94, -scale, scale); var u = Bench.Registry.Make(type, GH * a.H, 95, -scale, scale);
            var b = Bench.Registry.Make(type, GH, 96, -0.1, 0.1); var bh = Bench.Registry.Make(type, GH, 97, -0.1, 0.1);
            double[] V(byte[] s, int n) => Bench.Registry.Values(type, s, n);
            var wantH = Bench.RecurrentCase.GruSequenceReference(a, T, V(x, T * a.B * a.D), V(h0, a.B * a.H), V(w, GH * a.D), V(u, GH * a.H), V(b, GH), V(bh, GH));
            var hOut = new byte[Bench.Registry.Bytes(type, T * a.B * a.H)];
            var dev = ops is null ? Bench.Device.Cpu : Bench.Device.Cuda;
            using (var run = new Bench.QiyasSubject().GruSequence(dev, type, a, T, x, h0, w, u, b, bh, hOut)!) { run.Run(); run.Collect(); }
            ok &= Close(type, "gru_seq h", where, hOut, wantH, T * a.B * a.H, T * (a.D + a.H));
        }

        // The GRU sequence backward's reference against central differences of the forward reference (the formula, independently
        // of any kernel): the loss Σ dhOut · h over every step, its gradient in every input (x, h0, W, U, b, bh).
        {
            var a = new CellArgs("gru", 2, 3, 4);
            const int T = 3;
            int GH = a.Gates * a.H;
            double[] R(int n, int seed, double lo, double hi) => Bench.Registry.Values("float64", Bench.Registry.Make("float64", n, seed, lo, hi), n);
            var ins = new[] { R(T * a.B * a.D, 91, -1, 1), R(a.B * a.H, 92, -1, 1), R(GH * a.D, 94, -0.5, 0.5), R(GH * a.H, 95, -0.5, 0.5), R(GH, 96, -0.1, 0.1), R(GH, 97, -0.1, 0.1) };
            var dhOut = R(T * a.B * a.H, 98, -1, 1);
            var g = Bench.RecurrentCase.GruSequenceBackwardReference(a, T, ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], dhOut);
            var grads = new[] { g.DX, g.DH0, g.DW, g.DU, g.DB, g.DBH };
            double Loss()
            {
                var hs = Bench.RecurrentCase.GruSequenceReference(a, T, ins[0], ins[1], ins[2], ins[3], ins[4], ins[5]);
                double s = 0;
                for (int i = 0; i < hs.Length; i++) s += dhOut[i] * hs[i];
                return s;
            }

            double worst = 0;
            for (int v = 0; v < ins.Length; v++) for (int i = 0; i < ins[v].Length; i++)
            {
                const double step = 1e-5;
                double keep = ins[v][i];
                ins[v][i] = keep + step; double up = Loss();
                ins[v][i] = keep - step; double down = Loss();
                ins[v][i] = keep;
                worst = Math.Max(worst, Math.Abs((up - down) / (2 * step) - grads[v][i]));
            }

            if (worst > 1e-7) { Console.WriteLine($"check gru_seq_bwd reference: the gradient differs from the loss's differences by {worst:R}"); ok = false; }
        }

        // GRU sequence backward: 3 steps of 5 × 7 → 11 (dx, dh0, dW, dU, db, dbh), every float type, the block formats through lane storage.
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            var a = new CellArgs("gru", 5, 7, 11);
            const int T = 3;
            int GH = a.Gates * a.H;
            double scale = 1 / Math.Sqrt(a.D + a.H);
            var x = Bench.Registry.Make(type, T * a.B * a.D, 91, -1, 1); var h0 = Bench.Registry.Make(type, a.B * a.H, 92, -1, 1);
            var w = Bench.Registry.Make(type, GH * a.D, 94, -scale, scale); var u = Bench.Registry.Make(type, GH * a.H, 95, -scale, scale);
            var b = Bench.Registry.Make(type, GH, 96, -0.1, 0.1); var bh = Bench.Registry.Make(type, GH, 97, -0.1, 0.1);
            var dhOut = Bench.Registry.Make(type, T * a.B * a.H, 98, -1, 1);
            double[] V(byte[] s, int n) => Bench.Registry.Values(type, s, n);
            var (wdx, wdh0, wdw, wdu, wdb, wdbh) = Bench.RecurrentCase.GruSequenceBackwardReference(a, T, V(x, T * a.B * a.D), V(h0, a.B * a.H), V(w, GH * a.D), V(u, GH * a.H), V(b, GH), V(bh, GH), V(dhOut, T * a.B * a.H));
            var dx = new byte[x.Length]; var dh0 = new byte[h0.Length]; var dw = new byte[w.Length]; var du = new byte[u.Length]; var db = new byte[b.Length]; var dbh = new byte[bh.Length];
            var dev = ops is null ? Bench.Device.Cpu : Bench.Device.Cuda;
            using (var run = new Bench.QiyasSubject().GruSequenceBackward(dev, type, a, T, x, h0, w, u, b, bh, dhOut, dx, dh0, dw, du, db, dbh)!) { run.Run(); run.Collect(); }
            int terms = 8 * T * (a.D + a.H + GH + a.B);
            ok &= Close(type, "gru_seq_bwd dx", where, dx, wdx, T * a.B * a.D, terms);
            ok &= Close(type, "gru_seq_bwd dh0", where, dh0, wdh0, a.B * a.H, terms);
            ok &= Close(type, "gru_seq_bwd dw", where, dw, wdw, GH * a.D, terms);
            ok &= Close(type, "gru_seq_bwd du", where, du, wdu, GH * a.H, terms);
            ok &= Close(type, "gru_seq_bwd db", where, db, wdb, GH, terms);
            ok &= Close(type, "gru_seq_bwd dbh", where, dbh, wdbh, GH, terms);
        }

        if (ok) Console.WriteLine($"check recurrent              {where,-6} every float type: LSTM and GRU steps, forward and backward, LSTM sequences and their backward, GRU sequences and their backward (block formats too; the backward's reference by central differences), within the products' rounding");
        return ok;
    }

    private sealed class RecurrentRun(GpuOps? ops, CellArgs a, Bench.RecurrentCase k, byte[] output, byte[]? unused) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            bool lstm = a.Kind == "lstm", backward = k.DhOut is not null;
            var subject = new Bench.QiyasSubject();
            using var run = subject.Recurrent(ops is null ? Bench.Device.Cpu : Bench.Device.Cuda, TN.Name, a, k.X, k.H, k.C, k.W, k.U, k.Bias, k.BiasH,
                backward ? k.DhOut! : output, backward ? k.DcOut : k.COut, backward ? output : null, k.Dh, k.Dc, k.Dw, k.Du, k.Db, k.Dbh)!;
            run.Run();
            run.Collect();
            return true;
        }
    }

    // ---------------------------------------------------------------- convolution on odd shapes

    /// <summary>Every type: a 3 × 7 × 9 image by 5 filters of 3 × 2 at stride 2 × 1, pad 1, dilation 1; and 2 × 5 × 5 by 3 of 3 × 3
    /// dilated 2; forward and the backward pair against float64 within the products' rounding. Then the fused forward on
    /// five shapes, and on the CPU the backward with dw's columns read from the image on the same five.</summary>
    public static bool Convs(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in Bench.Registry.Names)
        {
            foreach (var a in new[] { new ConvArgs(3, 7, 9, 5, 3, 2, 2, 1, 1, 1), new ConvArgs(2, 5, 5, 3, 3, 3, 1, 1, 1, 1, 2, 2) })
            {
                var x = Bench.Registry.Make(type, (int)a.XValues, 61, -1, 1);
                var w = Bench.Registry.Make(type, (int)a.WValues, 62, -1, 1);
                var y = new byte[Bench.Registry.Bytes(type, a.YValues)];
                var want = a.Reference(Bench.Registry.Values(type, x, (int)a.XValues), Bench.Registry.Values(type, w, (int)a.WValues));
                Bench.Registry.With(type, new ConvRun(ops, a, x, w, y, null, null));
                ok &= Close(type, $"conv {a.Kh}x{a.Kw} s{a.Sh}", where, y, want, (int)a.YValues, a.Rows);
                if (!Bench.Registry.IsFloat(type)) continue;                 // the backward needs the float lane's storage
                var dy = Bench.Registry.Make(type, (int)a.YValues, 63, -1, 1);
                var (wantX, wantW) = a.ReferenceBackward(Bench.Registry.Values(type, x, (int)a.XValues), Bench.Registry.Values(type, w, (int)a.WValues), Bench.Registry.Values(type, dy, (int)a.YValues));
                var dx = new byte[x.Length];
                var dw = new byte[w.Length];
                Bench.Registry.WithFloat(type, new ConvBackwardRun(ops, a, x, w, dy, dx, dw));
                ok &= Close(type, $"conv_bwd dx {a.Kh}x{a.Kw}", where, dx, wantX, (int)a.XValues, a.OC * a.Kh * a.Kw);
                ok &= Close(type, $"conv_bwd dw {a.Kh}x{a.Kw}", where, dw, wantW, (int)a.WValues, a.Cols);
            }
        }

        // The fused forward (item 4g, implicit GEMM): forced on, within the products' rounding, and equal bit for bit to the
        // composition through the same kernel (the same packed values): the composition itself on the CPU and on the
        // device's tensor and dot routes; on the device's plain float lanes im2col then the same plain kernel (the plain
        // composition may take another kernel); the integer lanes are exact either way. Block formats: the composition
        // re-encodes the columns into blocks, the fused path reads the image's values: rounding only. Shapes: the two
        // above, a 3 × 3 at 16 × 20 × 20 → 32 (padded tiles), 1 × 1 at 128 × 16 × 16 → 128 (whole tiles), and a strided,
        // dilated one with an even width (sub-byte types in place).
        foreach (var type in Bench.Registry.Names)
        {
            foreach (var a in new[] { new ConvArgs(3, 7, 9, 5, 3, 2, 2, 1, 1, 1), new ConvArgs(2, 5, 5, 3, 3, 3, 1, 1, 1, 1, 2, 2), new ConvArgs(16, 20, 20, 32, 3, 3, 1, 1, 1, 1), new ConvArgs(128, 16, 16, 128, 1, 1), new ConvArgs(4, 9, 12, 6, 3, 3, 2, 2, 1, 1, 2, 2) })
            {
                var x = Bench.Registry.Make(type, (int)a.XValues, 64, -1, 1);
                var w = Bench.Registry.Make(type, (int)a.WValues, 65, -1, 1);
                var want = a.Reference(Bench.Registry.Values(type, x, (int)a.XValues), Bench.Registry.Values(type, w, (int)a.WValues));
                var fused = new byte[Bench.Registry.Bytes(type, a.YValues)];
                var composed = new byte[fused.Length];
                var run = new ConvModeRun(ops, a, x, w, fused, composed);
                Bench.Registry.With(type, run);
                string name = $"conv fused {a.C}x{a.H}x{a.W}>{a.OC} {a.Kh}x{a.Kw} s{a.Sh} d{a.Dh}";
                ok &= Close(type, name, where, fused, want, (int)a.YValues, a.Rows);
                if (!Bench.Registry.IsBlock(type) && !fused.AsSpan().SequenceEqual(composed))
                {
                    Console.WriteLine($"check {name} {where,-6} {type,-12} DIFFERS from the composition through the same kernel ({run.Lane})");
                    ok = false;
                }

                // The CPU's backward: dw with the columns read from the image (forced on) against float64, and dx and dw
                // equal bit for bit to the composition's (the columns written, then the same kernel).
                if (ops is not null || !Bench.Registry.IsFloat(type)) continue;
                var dy = Bench.Registry.Make(type, (int)a.YValues, 66, -1, 1);
                var (_, wantW) = a.ReferenceBackward(Bench.Registry.Values(type, x, (int)a.XValues), Bench.Registry.Values(type, w, (int)a.WValues), Bench.Registry.Values(type, dy, (int)a.YValues));
                byte[] dxF = new byte[x.Length], dwF = new byte[w.Length], dxC = new byte[x.Length], dwC = new byte[w.Length];
                Bench.Registry.WithFloat(type, new ConvBackwardModeRun(a, x, w, dy, dxF, dwF, dxC, dwC));
                string back = $"conv_bwd fused {a.C}x{a.H}x{a.W}>{a.OC} {a.Kh}x{a.Kw} s{a.Sh} d{a.Dh}";
                ok &= Close(type, back, where, dwF, wantW, (int)a.WValues, a.Cols);
                if (!dxF.AsSpan().SequenceEqual(dxC) || !dwF.AsSpan().SequenceEqual(dwC))
                {
                    Console.WriteLine($"check {back} {where,-6} {type,-12} DIFFERS from the composition through the same kernel");
                    ok = false;
                }
            }
        }

        if (ok) Console.WriteLine($"check convolution            {where,-6} every type: forward and the backward pair within the products' rounding; the fused forward equal to the composition through the same kernel{(ops is null ? ", and the backward with dw's columns from the image" : "")}");
        return ok;
    }

    // The fused forward forced on (into fused), then its composition through the same kernel (into composed).
    private sealed class ConvModeRun(GpuOps? ops, ConvArgs a, byte[] x, byte[] w, byte[] fused, byte[] composed) : Bench.ITypeVisitor<bool>
    {
        public string Lane = "";

        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null)
            {
                Kernels.FusedConv = null;
                Lane = Kernels.Conv<TN>(a, x, w, fused);
                Kernels.FusedConv = false;
                Kernels.Conv<TN>(a, x, w, composed);
                Kernels.FusedConv = null;
                return true;
            }

            var gpu = ops.Gpu;
            ulong dX = gpu.Rent(x.Length), dW = gpu.Rent(w.Length), dY = gpu.Rent(fused.Length), dC = gpu.Rent(composed.Length);
            Qiyas.Gpu.Upload<byte>(x, dX); Qiyas.Gpu.Upload<byte>(w, dW);
            ops.FusedConv = true;
            Lane = ops.Conv<TN>(a, dX, dW, dY);
            bool packed = Lane.EndsWith("mma", StringComparison.Ordinal) || Lane.EndsWith("dp", StringComparison.Ordinal);
            if (!packed && Lane is "f32" or "f64" && Bench.Registry.IsFloat(TN.Name)) ops.ConvPlainComposed<TN>(a, dX, dW, dC, Lane);
            else { ops.FusedConv = false; ops.Conv<TN>(a, dX, dW, dC); }
            ops.FusedConv = null;
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dY, fused);
            Qiyas.Gpu.Download<byte>(dC, composed);
            gpu.Return(dX); gpu.Return(dW); gpu.Return(dY); gpu.Return(dC);
            return true;
        }
    }

    // The CPU's backward with dw's columns read from the image (into dxF, dwF), then by the composition (into dxC, dwC).
    private sealed class ConvBackwardModeRun(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dxF, byte[] dwF, byte[] dxC, byte[] dwC) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            Kernels.FusedConv = null;
            Kernels.ConvBackward<TN>(a, x, w, dy, dxF, dwF);
            Kernels.FusedConv = false;
            Kernels.ConvBackward<TN>(a, x, w, dy, dxC, dwC);
            Kernels.FusedConv = null;
            return true;
        }
    }

    private sealed class ConvRun(GpuOps? ops, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.Conv<TN>(a, x, w, y); return true; }
            var gpu = ops.Gpu;
            ulong dX = gpu.Rent(x.Length), dW = gpu.Rent(w.Length), dY = gpu.Rent(y.Length);
            Qiyas.Gpu.Upload<byte>(x, dX); Qiyas.Gpu.Upload<byte>(w, dW);
            ops.Conv<TN>(a, dX, dW, dY);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dY, y);
            gpu.Return(dX); gpu.Return(dW); gpu.Return(dY);
            return true;
        }
    }

    private sealed class ConvBackwardRun(GpuOps? ops, ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.ConvBackward<TN>(a, x, w, dy, dx, dw); return true; }
            var gpu = ops.Gpu;
            ulong dX = gpu.Rent(x.Length), dW = gpu.Rent(w.Length), dY = gpu.Rent(dy.Length), gx = gpu.Rent(dx.Length), gw = gpu.Rent(dw.Length);
            Qiyas.Gpu.Upload<byte>(x, dX); Qiyas.Gpu.Upload<byte>(w, dW); Qiyas.Gpu.Upload<byte>(dy, dY);
            ops.ConvBackward<TN>(a, dX, dW, dY, gx, gw);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(gx, dx); Qiyas.Gpu.Download<byte>(gw, dw);
            gpu.Return(dX); gpu.Return(dW); gpu.Return(dY); gpu.Return(gx); gpu.Return(gw);
            return true;
        }
    }

    // ---------------------------------------------------------------- optimizer steps on odd sizes

    /// <summary>Every float type, 1001 and 37 values, each step at step counts 1 and 7: p against float64 within the lane's
    /// rounding of the update (several operations: 16 lane ulps of the update's magnitude), m and v within 4 lane ulps.</summary>
    public static bool Optimizers(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            int size = type == "float64" ? 8 : 4;
            foreach (var name in OptimizerArgs.Names)
            {
                foreach (var (n, step) in new[] { (1001, 1), (37, 7) })
                {
                    var a = new OptimizerArgs(name, 0.01, 0.9, 0.999, 1e-8, 0.01, step);
                    var p = Bench.Registry.Make(type, n, 71, -1, 1);
                    var g = Bench.Registry.Make(type, n, 72, -1, 1);
                    var random = new Random(73);
                    var mv = new double[n];
                    var vv = a.Moments > 1 ? new double[n] : null;
                    for (int i = 0; i < n; i++) { mv[i] = random.NextDouble() - 0.5; if (vv is not null) vv[i] = random.NextDouble(); }
                    var m = Lanes(size, mv);
                    var v = vv is null ? null : Lanes(size, vv);
                    var want = Bench.Registry.Values(type, p, n);
                    var wantM = (double[])mv.Clone();
                    var wantV = vv is null ? null : (double[])vv.Clone();
                    a.Reference(want, Bench.Registry.Values(type, g, n), wantM, wantV);
                    Bench.Registry.WithFloat(type, new OptimizerRun(ops, a, p, g, m, v, n));
                    ok &= Close(type, $"optimizer {name} t{step}", where, p, want, n, 16);
                    ok &= LanesClose(type, $"optimizer {name} m t{step}", where, m, wantM, size);
                    if (v is not null) ok &= LanesClose(type, $"optimizer {name} v t{step}", where, v, wantV!, size);
                }
            }
        }

        if (ok) Console.WriteLine($"check optimizers             {where,-6} every float type at 1001 and 37 values: p, m, v within the lane's rounding");
        return ok;
    }

    private static byte[] Lanes(int size, double[] values)
    {
        var bytes = new byte[values.Length * size];
        if (size == 8) System.Runtime.InteropServices.MemoryMarshal.Cast<double, byte>(values.AsSpan()).CopyTo(bytes);
        else for (int i = 0; i < values.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), (float)values[i]);
        return bytes;
    }

    private static bool LanesClose(string type, string what, string where, byte[] lanes, double[] want, int size)
    {
        double eps = size == 8 ? Math.Pow(2, -52) : Math.Pow(2, -23), worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double got = size == 8 ? BitConverter.ToDouble(lanes, i * 8) : BitConverter.ToSingle(lanes, i * 4);
            double tol = 4 * eps * (Math.Abs(want[i]) + 1);
            if (Math.Abs(got - want[i]) > tol) worst = Math.Max(worst, Math.Abs(got - want[i]) / tol);
        }

        if (worst > 0) Console.WriteLine($"check {what,-28} {where,-6} {type,-12} DIFFERS by {worst:0.##} tolerances");
        return worst == 0;
    }

    private sealed class OptimizerRun(GpuOps? ops, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.Optimizer<TN>(a, p, g, m, v, n); return true; }
            var gpu = ops.Gpu;
            ulong dp = gpu.Rent(p.Length), dg = gpu.Rent(g.Length), dm = gpu.Rent(m.Length), dv = v is null ? 0 : gpu.Rent(v.Length);
            Qiyas.Gpu.Upload<byte>(p, dp); Qiyas.Gpu.Upload<byte>(g, dg); Qiyas.Gpu.Upload<byte>(m, dm);
            if (v is not null) Qiyas.Gpu.Upload<byte>(v, dv);
            ops.Optimizer<TN>(a, dp, dg, dm, dv, n);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dp, p); Qiyas.Gpu.Download<byte>(dm, m);
            if (v is not null) Qiyas.Gpu.Download<byte>(dv, v);
            gpu.Return(dp); gpu.Return(dg); gpu.Return(dm);
            if (v is not null) gpu.Return(dv);
            return true;
        }
    }

    // ---------------------------------------------------------------- attention on odd shapes

    /// <summary>Every float type: 2 heads, 37 queries over 53 keys at d = 24 and d = 17 (plain, causal, spans), decode
    /// (3 queries over 1001 keys), and the backward pass, against float64 within the lane's rounding over the keys.</summary>
    public static bool Attentions(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            var cases = new List<(AttentionArgs A, bool Decode)>
            {
                (new(2, 37, 53, 24, 0.2, false), false), (new(2, 37, 53, 24, 0.2, true), false), (new(2, 37, 53, 17, 0.2, false), false),
                (new(2, 37, 53, 24, 0.2, false, Enumerable.Range(0, 37).Select(i => i * 3 % 60).ToArray()), false),
                (new(2, 37, 53, 24, 0.2, false, Enumerable.Range(0, 37).Select(i => 20 + i % 30).ToArray(), Enumerable.Range(0, 37).Select(i => i % 25).ToArray()), false),
                (new(2, 3, 1001, 24, 0.2, false, null, [5, 300, 900]), true),
                (new(2, 3, 1001, 24, 0.2, false), true), (new(2, 5, 1001, 17, 0.2, true), true),
            };
            foreach (var (a, decode) in cases)
            {
                var q = Bench.Registry.Make(type, (int)a.QValues, 41, -1, 1);
                var k = Bench.Registry.Make(type, (int)a.KValues, 42, -1, 1);
                var v = Bench.Registry.Make(type, (int)a.KValues, 43, -1, 1);
                var o = new byte[q.Length];
                var want = a.Reference(Bench.Registry.Values(type, q, (int)a.QValues), Bench.Registry.Values(type, k, (int)a.KValues), Bench.Registry.Values(type, v, (int)a.KValues));
                Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, o, decode, null, null, null));
                ok &= Close(type, $"attention{(decode ? " decode" : "")}{(a.Causal ? " causal" : "")}{(a.Spans is not null ? " spans" : "")} d{a.D}", where, o, want, (int)a.QValues, a.Lk + a.D);
            }

            // The fused rows forced, at every head-dimension capacity (32, 64, 128) and tile edge: queries and keys not in
            // whole tiles, causal, spans with starts, a row block whose rows see no key.
            var fusedCases = new List<AttentionArgs>
            {
                new(2, 37, 53, 24, 0.2, false), new(2, 37, 53, 17, 0.2, true), new(3, 37, 100, 64, 0.1, false), new(2, 33, 71, 128, 0.08, true),
                new(2, 37, 53, 120, 0.1, false, Enumerable.Range(0, 37).Select(i => 20 + i % 30).ToArray(), Enumerable.Range(0, 37).Select(i => i % 25).ToArray()),
                new(2, 40, 53, 24, 0.2, false, Enumerable.Range(0, 40).Select(i => i < 16 ? 0 : 1 + i).ToArray()),
            };
            foreach (var a in fusedCases)
            {
                var q = Bench.Registry.Make(type, (int)a.QValues, 41, -1, 1);
                var k = Bench.Registry.Make(type, (int)a.KValues, 42, -1, 1);
                var v = Bench.Registry.Make(type, (int)a.KValues, 43, -1, 1);
                var o = new byte[q.Length];
                var want = a.Reference(Bench.Registry.Values(type, q, (int)a.QValues), Bench.Registry.Values(type, k, (int)a.KValues), Bench.Registry.Values(type, v, (int)a.KValues));
                Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, o, false, null, null, null, fused: true));
                ok &= Close(type, $"attention fused{(a.Causal ? " causal" : "")}{(a.Spans is not null ? " spans" : "")}{(a.SpanStarts is not null ? " starts" : "")} {a.Heads}h {a.Lq}x{a.Lk} d{a.D}", where, o, want, (int)a.QValues, a.Lk + a.D);
            }

            {
                var a = new AttentionArgs(2, 37, 53, 24, 0.2, true);
                var q = Bench.Registry.Make(type, (int)a.QValues, 41, -1, 1);
                var k = Bench.Registry.Make(type, (int)a.KValues, 42, -1, 1);
                var v = Bench.Registry.Make(type, (int)a.KValues, 43, -1, 1);
                var dO = Bench.Registry.Make(type, (int)a.QValues, 44, -1, 1);
                var (wq, wk, wv) = a.ReferenceBackward(Bench.Registry.Values(type, q, (int)a.QValues), Bench.Registry.Values(type, k, (int)a.KValues), Bench.Registry.Values(type, v, (int)a.KValues), Bench.Registry.Values(type, dO, (int)a.QValues));
                var dq = new byte[q.Length];
                var dk = new byte[k.Length];
                var dv = new byte[v.Length];
                Bench.Registry.WithFloat(type, new AttentionRun(ops, a, q, k, v, dO, false, dq, dk, dv));
                ok &= Close(type, "attention_bwd dQ", where, dq, wq, (int)a.QValues, 2 * (a.Lk + a.D));
                ok &= Close(type, "attention_bwd dK", where, dk, wk, (int)a.KValues, 2 * (a.Lq + a.D));
                ok &= Close(type, "attention_bwd dV", where, dv, wv, (int)a.KValues, a.Lq + a.D);
            }
        }

        if (ok) Console.WriteLine($"check attention              {where,-6} every float type: rows (composed and fused), causal, spans, decode and the backward pass within the lane's rounding");
        return ok;
    }

    private sealed class AttentionRun(GpuOps? ops, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, bool decode, byte[]? dq, byte[]? dk, byte[]? dv, bool fused = false) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            bool backward = dq is not null;
            if (ops is null)
            {
                if (backward) Kernels.AttentionBackward<TN>(a, q, k, v, o, dq!, dk!, dv!);
                else if (decode) Kernels.AttentionDecode<TN>(a, q, k, v, o);
                else if (fused) Kernels.AttentionRows<TN>(a, q, k, v, o);
                else Kernels.Attention<TN>(a, q, k, v, o);
                return true;
            }

            var gpu = ops.Gpu;
            ulong dQ = gpu.Rent(q.Length), dK = gpu.Rent(k.Length), dV = gpu.Rent(v.Length), dO = gpu.Rent(o.Length);
            Qiyas.Gpu.Upload<byte>(q, dQ); Qiyas.Gpu.Upload<byte>(k, dK); Qiyas.Gpu.Upload<byte>(v, dV);
            if (backward)
            {
                Qiyas.Gpu.Upload<byte>(o, dO);
                ulong gq = gpu.Rent(dq!.Length), gk = gpu.Rent(dk!.Length), gv = gpu.Rent(dv!.Length);
                ops.AttentionBackward<TN>(a, dQ, dK, dV, dO, gq, gk, gv);
                Qiyas.Gpu.Synchronize();
                Qiyas.Gpu.Download<byte>(gq, dq); Qiyas.Gpu.Download<byte>(gk, dk); Qiyas.Gpu.Download<byte>(gv, dv);
                gpu.Return(gq); gpu.Return(gk); gpu.Return(gv);
            }
            else
            {
                if (decode) ops.AttentionDecode<TN>(a, dQ, dK, dV, dO); else if (fused) ops.AttentionRows<TN>(a, dQ, dK, dV, dO); else ops.Attention<TN>(a, dQ, dK, dV, dO);
                Qiyas.Gpu.Synchronize();
                Qiyas.Gpu.Download<byte>(dO, o);
            }

            gpu.Return(dQ); gpu.Return(dK); gpu.Return(dV); gpu.Return(dO);
            return true;
        }
    }

    // ---------------------------------------------------------------- ctc

    /// <summary>Every float type: the ctc loss (forward alone and with the gradient) and the gradient against the float64
    /// reference on odd shapes: sequences shorter than T, an empty label, a label repeated next to itself (the blank
    /// between them is required) and apart, a blank other than class 0. The recursion runs 2T steps over log values up to
    /// |log P|, so the tolerance takes T · |log P| lane steps. A sequence no alignment fits (types with an infinity): loss
    /// +∞, gradient 0. The gradient of each frame sums to 0 (float32, float64).</summary>
    public static bool Ctcs(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        var cases = new List<(CtcArgs A, bool Impossible)>
        {
            (new(11, 4, 7, 0, [11, 9, 11, 7], [3, 4, 0, 3], [1, 2, 3, 2, 2, 4, 5, 3, 1, 3]), false),
            (new(13, 3, 5, 4, [13, 6, 12], [5, 3, 1], [0, 1, 2, 3, 0, 1, 1, 1, 2]), false),
            (new(5, 2, 4, 0, [5, 2], [2, 2], [1, 2, 3, 3]), true),
        };
        // The reference's gradient against its own loss by central differences (the formula, independently of any kernel).
        foreach (var (a, impossible) in cases)
        {
            if (impossible) continue;
            var x = Bench.Registry.Values("float64", Bench.Registry.Make("float64", (int)a.Values, 61, -2, 2), (int)a.Values);
            var (_, g) = a.Reference(x);
            double worst = 0;
            for (long i = 0; i < a.Values; i++)
            {
                int bb = (int)(i / a.C % a.B);
                const double h = 1e-5;
                double keep = x[i];
                x[i] = keep + h; double up = a.Reference(x).Loss[bb];
                x[i] = keep - h; double down = a.Reference(x).Loss[bb];
                x[i] = keep;
                worst = Math.Max(worst, Math.Abs((up - down) / (2 * h) - g[i]));
            }

            if (worst > 1e-7) { Console.WriteLine($"check ctc reference T{a.T} B{a.B} C{a.C}: the gradient differs from the loss's differences by {worst:R}"); ok = false; }
        }

        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            bool infinity = type is "float32" or "float64" or "float16" or "bfloat16";
            foreach (var (a, impossible) in cases)
            {
                if (impossible && !infinity) continue;
                var x = Bench.Registry.Make(type, (int)a.Values, 61, -2, 2);
                var (wantLoss, wantGrad) = a.Reference(Bench.Registry.Values(type, x, (int)a.Values));
                double logP = wantLoss.Where(double.IsFinite).Max(Math.Abs);
                int terms = a.T * (int)Math.Ceiling(logP + 1);
                string name = $"ctc T{a.T} B{a.B} C{a.C}{(impossible ? " no alignment" : "")}";
                int scored = impossible ? 1 : a.B;                            // the impossible case: its first sequence scored, the second checked for +∞
                var loss = new byte[Bench.Registry.Bytes(type, a.B)];
                Bench.Registry.WithFloat(type, new CtcRun(ops, a, x, loss, null));
                ok &= Close(type, name + " loss", where, loss, wantLoss, scored, terms);
                var loss2 = new byte[loss.Length];
                var grad = new byte[Bench.Registry.Bytes(type, a.Values)];
                Bench.Registry.WithFloat(type, new CtcRun(ops, a, x, loss2, grad));
                ok &= Close(type, name + " loss (bwd)", where, loss2, wantLoss, scored, terms);
                ok &= Close(type, name + " grad", where, grad, wantGrad, (int)a.Values, terms);
                if (impossible)
                {
                    foreach (var l in new[] { loss, loss2 })
                    {
                        double got = Bench.Registry.Values(type, l, a.B)[1];
                        if (!double.IsPositiveInfinity(got)) { Console.WriteLine($"check ctc no alignment {where,-6} {type,-12} loss {got:R}, not +∞"); ok = false; }
                    }
                }

                if (type is "float32" or "float64")
                {
                    var g = Bench.Registry.Values(type, grad, (int)a.Values);
                    double eps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23), worst = 0;
                    for (int t = 0; t < a.T; t++)
                    {
                        for (int bb = 0; bb < a.B; bb++)
                        {
                            double sum = 0;
                            for (int c = 0; c < a.C; c++) sum += g[((long)t * a.B + bb) * a.C + c];
                            worst = Math.Max(worst, Math.Abs(sum));
                        }
                    }

                    if (worst > 4 * (a.C + terms) * eps) { Console.WriteLine($"check ctc grad sum {where,-6} {type,-12} a frame's gradient sums to {worst:R}, not 0"); ok = false; }
                }
            }
        }

        if (ok) Console.WriteLine($"check ctc                    {where,-6} every float type: loss and gradient (the reference's gradient by central differences), short sequences, empty and repeated labels, another blank, no alignment, within the recursion's rounding");
        return ok;
    }

    private sealed class CtcRun(GpuOps? ops, CtcArgs a, byte[] x, byte[] loss, byte[]? grad) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.Ctc<TN>(a, x, loss, grad); return true; }
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dl = gpu.Rent(loss.Length), dg = grad is null ? 0 : gpu.Rent(grad.Length);
            Qiyas.Gpu.Upload<byte>(x, dx);
            ops.Ctc<TN>(a, dx, dl, dg);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dl, loss);
            if (grad is not null) { Qiyas.Gpu.Download<byte>(dg, grad); gpu.Return(dg); }
            gpu.Return(dx); gpu.Return(dl);
            return true;
        }
    }

    // ---------------------------------------------------------------- products on views and the matmul backward on odd shapes

    /// <summary>Every type: 37 × 29 × 53 with each transposition, three batches of it at contiguous strides, and the
    /// backward pair, against float64 within the product's rounding over k terms.</summary>
    public static bool Products(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        const int M = 37, N = 29, K = 53;
        // On the device, every tensor kind it accepts: the measured choice, then the 4-bit kinds forced on and off with the
        // 32-deep 8-bit instruction forced on and off (a timing may pick any of them on a run).
        var kinds = ops is null ? [(null, null, "")] : new (bool? four, bool? k32, string name)[] { (null, null, ""), (true, true, " [4-bit, k32]"), (true, false, " [4-bit, k16]"), (false, true, " [8-bit, k32]"), (false, false, " [8-bit, k16]") };
        // The device's pass repeated under --repeat (Stress / 20 passes): a fault that depends on the buffers' history shows.
        for (int pass = 0, passes = ops is null ? 1 : Math.Max(1, Stress / 20); pass < passes && ok; pass++)
        {
            foreach (var (four, k32, kindName) in kinds)
            {
                if (ops is not null) { ops.FourBit = four; ops.K32 = k32; }
                ok &= ProductsWith(ops, where, M, N, K, pass == 0 ? kindName : $"{kindName} pass {pass + 1}");
            }
        }

        if (ops is not null) { ops.FourBit = null; ops.K32 = null; }
        // The packing reading the views' operands where they are stored (Aᵀ, Bᵀ, batch offsets): shapes that take that
        // route. CPU: two panels of k with a partial last group, edge rows and strips; and batch strides on whole blocks
        // (the block formats in place too). Device: whole tiles (the tensor and dot routes).
        if (ops is null)
        {
            ok &= ProductsWith(null, where, 64, 40, 330, " [in place]");
            ok &= ProductsWith(null, where, 64, 48, 320, " [in place]");
        }
        else ok &= ProductsWith(ops, where, 128, 256, 128, " [in place]");
        if (ops is not null) ok &= TensorConfigs(ops);
        ok &= BlockProducts(ops, where);
        if (ok) Console.WriteLine($"check products               {where,-6} every type at 37 x 29 x 53, transposed, batched, and the backward pair{(ops is null ? "" : ", under every tensor kind")}; read in place at {(ops is null ? "64 x 40 x 330 and 64 x 48 x 320" : "128 x 256 x 128")}: within the product's rounding");
        return ok;
    }

    /// <summary>The block-scaled products where the device has them: A·Bᵀ of two Q8_0 / Q4_0 operands at 37 × 29 with k = 64
    /// and 320 (whole blocks), the block product forced, against the double-double reference of the stored values; gemv's
    /// block path at 37 rows of 64 (the reductions' gate has 1001 columns, never a whole number of blocks).</summary>
    private static bool BlockProducts(GpuOps? ops, string where)
    {
        bool available = ops is null ? Kernels.BlockProductAvailable : ops.BlockProductAvailable;
        if (!available)
        {
            Console.WriteLine($"check block products         {where,-6} not available here (the {(ops is null ? "s8 dot lane" : "s8 tensor instruction at k = 32")}): the decoded route stands");
            return true;
        }

        bool ok = true;
        const int M = 37, N = 29;
        if (ops is null) Kernels.BlockProducts = true; else ops.BlockProducts = true;
        try
        {
            foreach (var type in new[] { "q8_0", "q4_0" })
            {
                foreach (int K in new[] { 64, 320 })
                {
                    var v = new ProductView(M, N, K, false, true);
                    var a = Bench.Registry.Make(type, M * K, 34, -1, 1);
                    var b = Bench.Registry.Make(type, N * K, 35, -1, 1);
                    var c = new byte[Bench.Registry.Bytes(type, M * N)];
                    var want = v.Reference(Bench.Registry.Values(type, a, M * K), Bench.Registry.Values(type, b, N * K));
                    Bench.Registry.With(type, new ProductRun(ops, v, a, b, c, null, null));
                    ok &= Close(type, $"product Bᵀ blocks k{K}", where, c, want, M * N, K);
                }

                var ga = Bench.Registry.Make(type, M * 64, 44, -1, 1);
                var gx = Bench.Registry.Make(type, 64, 45, -1, 1);
                var gy = new byte[Bench.Registry.Bytes(type, M)];
                var gwant = References.Gemv(Bench.Registry.Values(type, ga, M * 64), Bench.Registry.Values(type, gx, 64), M, 64, false);
                Bench.Registry.With(type, new GemvRun(ops, ga, gx, gy, M, 64, false));
                ok &= Close(type, "gemv blocks k64", where, gy, gwant, M, 64);
            }
        }
        finally
        {
            if (ops is null) Kernels.BlockProducts = null; else ops.BlockProducts = null;
        }

        if (ok) Console.WriteLine($"check block products         {where,-6} Q8_0, Q4_0: A·Bᵀ at 37 × 29 × 64 and × 320, the block-scaled product forced, and gemv at 37 × 64: within the product's rounding");
        return ok;
    }

    /// <summary>Every tensor configuration the device accepts, forced in turn, under every tensor kind: the product of every
    /// type a tensor kind takes at k = 53 and k = 300 (neither a whole 128 nor 256 values) against float64.</summary>
    private static bool TensorConfigs(GpuOps ops)
    {
        bool ok = true;
        int tried = 0;
        var kinds = new (bool four, bool k32, string name)[] { (true, true, "4-bit, k32"), (true, false, "4-bit, k16"), (false, true, "8-bit, k32"), (false, false, "8-bit, k16") };
        foreach (var cfg in GpuOps.MmaConfigs)
        {
            ops.ForcedShape = cfg;
            foreach (var (four, k32, kindName) in kinds)
            {
                ops.FourBit = four; ops.K32 = k32;
                foreach (var type in Bench.Registry.Names)
                {
                    if (Bench.Registry.IsBlock(type) || type is "float32" or "float64") continue;
                    foreach (int K in new[] { 53, 300 })
                    {
                        const int M = 37, N = 29;
                        var v = new ProductView(M, N, K, false, false);
                        var a = Bench.Registry.Make(type, M * K, 34, -1, 1);
                        var b = Bench.Registry.Make(type, K * N, 35, -1, 1);
                        var c = new byte[Bench.Registry.Bytes(type, M * N)];
                        var want = v.Reference(Bench.Registry.Values(type, a, M * K), Bench.Registry.Values(type, b, K * N));
                        Bench.Registry.With(type, new ProductRun(ops, v, a, b, c, null, null));
                        ok &= Close(type, $"product k{K} [{kindName}; rows {cfg.RowBytes} B x {cfg.Stages} stages x {cfg.Bn} wide{(cfg.Swizzle ? " swizzled" : "")}]", "device", c, want, M * N, K);
                        tried++;
                    }
                }
            }
        }

        ops.ForcedShape = null; ops.FourBit = null; ops.K32 = null;
        if (ok) Console.WriteLine($"check tensor configurations  device {GpuOps.MmaConfigs.Length} configurations x 4 kind choices: {tried} products at k = 53 and 300 as the references");
        return ok;
    }

    private static bool ProductsWith(GpuOps? ops, string where, int M, int N, int K, string kindName)
    {
        bool ok = true;
        foreach (var type in Bench.Registry.Names)
        {
            var views = new List<ProductView>();
            foreach (bool ta in new[] { false, true }) foreach (bool tb in new[] { false, true }) views.Add(new ProductView(M, N, K, ta, tb));
            views.Add(new ProductView(M, N, K, false, true, 3, M * K, N * K, M * N));
            views.Add(new ProductView(M, N, K, true, false, 3, M * K, 0, M * N));
            views.Add(new ProductView(M, N, K, false, false, 2, M * K, K * N, M * N));     // batched as stored (4e: batch 0 at offset 0 is still a view)
            foreach (var v in views)
            {
                var a = Bench.Registry.Make(type, (int)v.ValuesA, 34, -1, 1);
                var b = Bench.Registry.Make(type, (int)v.ValuesB, 35, -1, 1);
                var c = new byte[Bench.Registry.Bytes(type, v.ValuesC)];
                var want = v.Reference(Bench.Registry.Values(type, a, (int)v.ValuesA), Bench.Registry.Values(type, b, (int)v.ValuesB));
                Bench.Registry.With(type, new ProductRun(ops, v, a, b, c, null, null));
                ok &= Close(type, $"product{(v.TransA ? " Aᵀ" : "")}{(v.TransB ? " Bᵀ" : "")}{(v.Batch > 1 ? $" x{v.Batch}" : "")}{kindName}", where, c, want, (int)v.ValuesC, K);
            }

            {
                var a = Bench.Registry.Make(type, M * K, 34, -1, 1);
                var b = Bench.Registry.Make(type, K * N, 35, -1, 1);
                var dc = Bench.Registry.Make(type, M * N, 36, -1, 1);
                var da = new byte[Bench.Registry.Bytes(type, M * K)];
                var db = new byte[Bench.Registry.Bytes(type, K * N)];
                var wantA = new ProductView(M, K, N, false, true).Reference(Bench.Registry.Values(type, dc, M * N), Bench.Registry.Values(type, b, K * N));
                var wantB = new ProductView(K, N, M, true, false).Reference(Bench.Registry.Values(type, a, M * K), Bench.Registry.Values(type, dc, M * N));
                Bench.Registry.With(type, new ProductRun(ops, new ProductView(M, N, K, false, false), a, b, dc, da, db));
                ok &= Close(type, "matmul_bwd dA" + kindName, where, da, wantA, M * K, N);
                ok &= Close(type, "matmul_bwd dB" + kindName, where, db, wantB, K * N, M);
            }
        }

        return ok;
    }

    private sealed class ProductRun(GpuOps? ops, ProductView v, byte[] a, byte[] b, byte[] c, byte[]? da, byte[]? db) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            bool backward = da is not null;
            if (ops is null)
            {
                if (backward) Kernels.MatMulBackward<TN>(a, b, c, da!, db!, v.M, v.N, v.K); else Kernels.MatMulView<TN>(v, a, b, c);
                return true;
            }

            var gpu = ops.Gpu;
            ulong dA = gpu.Rent(a.Length), dB = gpu.Rent(b.Length), dC = gpu.Rent(c.Length);
            Qiyas.Gpu.Upload<byte>(a, dA); Qiyas.Gpu.Upload<byte>(b, dB);
            if (backward)
            {
                Qiyas.Gpu.Upload<byte>(c, dC);
                ulong dDa = gpu.Rent(da!.Length), dDb = gpu.Rent(db!.Length);
                ops.MatMulBackward<TN>(dA, dB, dC, dDa, dDb, v.M, v.N, v.K);
                Qiyas.Gpu.Synchronize();
                Qiyas.Gpu.Download<byte>(dDa, da); Qiyas.Gpu.Download<byte>(dDb, db);
                gpu.Return(dDa); gpu.Return(dDb);
            }
            else
            {
                ops.MatMulView<TN>(v, dA, dB, dC);
                Qiyas.Gpu.Synchronize();
                Qiyas.Gpu.Download<byte>(dC, c);
            }

            gpu.Return(dA); gpu.Return(dB); gpu.Return(dC);
            return true;
        }
    }

    // ---------------------------------------------------------------- layout and indexing on odd shapes

    /// <summary>Every type, every layout operation on small odd shapes (LayoutCases.Gate): exact for the moves (the value
    /// moves as stored), the sums within the lane's rounding over their terms, bilinear within 4 lane ulps.</summary>
    public static bool Layouts(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in Bench.Registry.Names)
        {
            foreach (var name in LayoutArgs.Names)
            {
                if (!LayoutArgs.Applies(name, Bench.Registry.IsFloat(type))) continue;
                foreach (var k in Bench.LayoutCases.Gate(name, type))
                {
                    if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1") Console.WriteLine($"  layout {name} {type} {Bench.LayoutCases.Shape(k)} ({where})");
                    var reference = LayoutArgs.Reference(name, Bench.Registry.Values(type, k.X, k.InputCount), k.Index, k.P, k.Count, k.X2 is null ? null : Bench.Registry.Values(type, k.X2, Bench.LayoutCases.SecondCount(k)));
                    Bench.Registry.With(type, new LayoutRun(ops, k));
                    // Sums round over their terms; a bilinear value is four weighted terms of the inputs' magnitude (not the result's); a pool's mean over its window.
                    int terms = name is "scatter_add_rows" or "col2im" or "maxpool_bwd" or "depthwise_conv" or "adaptive_maxpool_bwd" ? LayoutArgs.Terms(name, k.P) : name == "interpolate_bilinear" ? 4 : name is "avgpool" or "avgpool_bwd" ? k.P[3] * k.P[4] : name is "adaptive_avgpool" or "adaptive_avgpool_bwd" ? (k.P[1] / k.P[3] + 2) * (k.P[2] / k.P[4] + 2) : 0;
                    ok &= Close(type, $"layout {name}", where, k.Output, reference, k.Count, terms);
                }
            }
        }

        if (ok) Console.WriteLine($"check layouts                {where,-6} every type: moves exact, sums and resizes within the lane's rounding");
        return ok;
    }

    private sealed class LayoutRun(GpuOps? ops, LayoutArgs k) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.Layout<TN>(k); return true; }
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(k.X.Length), dout = gpu.Rent(k.Output.Length), didx = k.Index is null ? 0 : gpu.Rent(k.Index.Length * 4L), dx2 = k.X2 is null ? 0 : gpu.Rent(k.X2.Length);
            Qiyas.Gpu.Upload<byte>(k.X, dx);
            if (k.Index is not null) Qiyas.Gpu.Upload<int>(k.Index, didx);
            if (k.X2 is not null) Qiyas.Gpu.Upload<byte>(k.X2, dx2);
            ops.Layout<TN>(k, dx, didx, dout, dx2);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dout, k.Output);
            gpu.Return(dx); gpu.Return(dout);
            if (didx != 0) gpu.Return(didx);
            if (dx2 != 0) gpu.Return(dx2);
            return true;
        }
    }

    // ---------------------------------------------------------------- whole sums of float types on the device

    /// <summary>The device's whole sum and sum of squares of a narrow float type over 2^20 + 3 values, and its row sums over
    /// 3 rows of 2^18 + 5 and 2 rows of 2^18 + 4 (whole words a row: the word kernel): each within one f32 lane step of the exact sum (compensated sums joined in float64, rounded once).
    /// The plain f32 lane drifts past it.</summary>
    private static bool WholeSums(GpuOps ops)
    {
        bool ok = true;
        foreach (var (axis, rows, cols) in new[] { (Axis.Whole, 1, (1 << 20) + 3), (Axis.Rows, 3, (1 << 18) + 5), (Axis.Rows, 2, (1 << 18) + 4) })
        foreach (var type in new[] { "float32", "float16", "bfloat16", "q8_0" })
        {
            var x = Bench.Registry.Make(type, rows * cols, 41, -1, 1);
            var all = Bench.Registry.Values(type, x, rows * cols);
            foreach (var op in new[] { "sum", "sumsq" })
            {
                var (got, lane) = Bench.Registry.With(type, new ReduceRun(ops, op, axis, x, rows, cols));
                var values = Bench.LaneValues.Decode(lane, got, rows);
                for (int r = 0; r < rows; r++)
                {
                var v = all.AsSpan(r * cols, cols);
                double exact = 0, c = 0;
                foreach (double t in v)
                {
                    // Neumaier's sum in float64: exact to far below the f32 step for these terms.
                    double term = op == "sumsq" ? t * t : t, s = exact + term;
                    c += Math.Abs(exact) >= Math.Abs(term) ? (exact - s) + term : (term - s) + exact;
                    exact = s;
                }

                exact += c;
                double value = values[r];
                double step = Math.Pow(2, Math.Floor(Math.Log2(Math.Abs(exact))) - 23);
                if (Math.Abs(value - exact) > step) { Console.WriteLine($"check {axis} {op} {type} device row {r}: {value:R} against {exact:R} ({Math.Abs(value - exact) / step:0.##} f32 steps, lane {lane})"); ok = false; }
                }
            }
        }

        if (ok) Console.WriteLine("check whole sums             device float32, float16, bfloat16, q8_0: sum and sum of squares of 2^20 + 3 values, and of rows of 2^18 + 5 and 2^18 + 4, within one f32 step of the exact sum");
        return ok;
    }

    // ---------------------------------------------------------------- the buffer arena and the product's workspace

    /// <summary>The device: a transposed product, attention and an LSTM step (float16, float32), each called once outside a scope
    /// (the pool) and then in scopes: from the second scoped call no pool rent, no driver allocation, no rent past the region,
    /// nothing left unreturned or returned late, and the results equal to the pool's bit for bit. The CPU: the product and the
    /// LSTM step reach a call with no workspace allocation and no pool fallback within a few calls, the results unchanged.</summary>
    public static bool Arenas(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in new[] { "float16", "float32" })
        {
            var cases = new List<(string Name, byte[][] Inputs, int OutBytes, Func<ulong[], ulong, string>? Device, Action<byte[][], byte[]>? Cpu)>();
            {
                var v = new ProductView(192, 160, 300, true, false);
                var a = Bench.Registry.Make(type, 300 * 192, 71, -1, 1); var b = Bench.Registry.Make(type, 300 * 160, 72, -1, 1);
                int outBytes = (int)Bench.Registry.Bytes(type, 192 * 160);
                cases.Add(("matmul_tn", [a, b], outBytes,
                    (d, o) => Bench.Registry.WithFloat(type, new ArenaProduct(ops, v, d[0], d[1], o, null, null, null)),
                    (x, o) => Bench.Registry.WithFloat(type, new ArenaProduct(null, v, 0, 0, 0, x[0], x[1], o))));
            }

            if (ops is not null)
            {
                var at = new AttentionArgs(2, 37, 53, 24, 0.2, true);
                var q = Bench.Registry.Make(type, (int)at.QValues, 73, -1, 1); var k = Bench.Registry.Make(type, (int)at.KValues, 74, -1, 1); var vv = Bench.Registry.Make(type, (int)at.KValues, 75, -1, 1);
                cases.Add(("attention", [q, k, vv], (int)Bench.Registry.Bytes(type, at.QValues),
                    (d, o) => Bench.Registry.WithFloat(type, new ArenaAttention(ops, at, d[0], d[1], d[2], o)), null));
            }

            {
                var ca = new CellArgs("lstm", 5, 7, 11);
                int GH = ca.Gates * ca.H;
                byte[] M(int n, int seed, double lo, double hi) => Bench.Registry.Make(type, n, seed, lo, hi);
                byte[][] ins = [M(ca.B * ca.D, 81, -1, 1), M(ca.B * ca.H, 82, -1, 1), M(ca.B * ca.H, 83, -1, 1), M(GH * ca.D, 84, -0.3, 0.3), M(GH * ca.H, 85, -0.3, 0.3), M(GH, 86, -0.1, 0.1)];
                int hBytes = (int)Bench.Registry.Bytes(type, ca.B * ca.H), cOff = (hBytes + 255) / 256 * 256;
                cases.Add(("lstm_step", ins, cOff + hBytes,
                    (d, o) => Bench.Registry.WithFloat(type, new ArenaLstm(ops, ca, d, o, hBytes, cOff, null, null)),
                    (x, o) => Bench.Registry.WithFloat(type, new ArenaLstm(null, ca, null, 0, hBytes, cOff, x, o))));
            }

            foreach (var (name, inputs, outBytes, device, cpu) in cases)
            {
                if (ops is null)
                {
                    var first = new byte[outBytes];
                    cpu!(inputs, first);
                    bool steady = false;
                    for (int call = 0; call < 8 && !steady; call++)
                    {
                        var again = new byte[outBytes];
                        var (_, g0, p0) = Workspace.Counts;
                        cpu(inputs, again);
                        var (_, g1, p1) = Workspace.Counts;
                        steady = g1 == g0 && p1 == p0;
                        if (!again.AsSpan().SequenceEqual(first)) { Console.WriteLine($"check arena {name} {type} cpu: call {call + 2} differs from the first"); ok = false; break; }
                    }

                    if (!steady) { Console.WriteLine($"check arena {name} {type} cpu: still allocating after 8 calls"); ok = false; }
                    continue;
                }

                var gpu = ops.Gpu;
                var d = inputs.Select(x => { ulong p = gpu.Rent(x.Length); Qiyas.Gpu.Upload<byte>(x, p); return p; }).ToArray();
                ulong dout = gpu.Rent(outBytes);
                try
                {
                    var pooled = new byte[outBytes];
                    Qiyas.Gpu.Clear(dout, outBytes);
                    device!(d, dout);
                    Qiyas.Gpu.Synchronize();
                    Qiyas.Gpu.Download<byte>(dout, pooled);
                    for (int call = 0; call < 3; call++)
                    {
                        Qiyas.Gpu.Clear(dout, outBytes);
                        var c0 = gpu.Counts;
                        using (ops.Scope()) device(d, dout);
                        var c1 = gpu.Counts;
                        var got = new byte[outBytes];
                        Qiyas.Gpu.Synchronize();
                        Qiyas.Gpu.Download<byte>(dout, got);
                        if (!got.AsSpan().SequenceEqual(pooled)) { Console.WriteLine($"check arena {name} {type} device: scoped call {call + 1} differs from the pool's"); ok = false; }
                        if (c1.Leaks != c0.Leaks || c1.Strays != c0.Strays) { Console.WriteLine($"check arena {name} {type} device: scoped call {call + 1} left {c1.Leaks - c0.Leaks} buffers unreturned, {c1.Strays - c0.Strays} returned late"); ok = false; }
                        var (high, peak) = gpu.LastScope;
                        if (call >= 1 && high != peak) { Console.WriteLine($"check arena {name} {type} device: scoped call {call + 1} reached {high} bytes of the region for {peak} alive at once (returned buffers not released)"); ok = false; }
                        if (call >= 1 && (c1.PoolRents != c0.PoolRents || c1.DriverAllocations != c0.DriverAllocations || c1.Overflows != c0.Overflows || c1.ArenaRents == c0.ArenaRents))
                        {
                            Console.WriteLine($"check arena {name} {type} device: scoped call {call + 1}: {c1.ArenaRents - c0.ArenaRents} arena rents, {c1.PoolRents - c0.PoolRents} pool rents, {c1.DriverAllocations - c0.DriverAllocations} driver allocations, {c1.Overflows - c0.Overflows} past the region");
                            ok = false;
                        }
                    }
                }
                finally
                {
                    foreach (var p in d) gpu.Return(p);
                    gpu.Return(dout);
                }
            }
        }

        if (ok) Console.WriteLine($"check arena                  {where,-6} float16, float32: {(ops is null ? "the product and the LSTM step reach calls with no workspace allocation, the results unchanged" : "a transposed product, attention, an LSTM step: from the second scoped call no pool rent, no driver allocation, nothing past the region or unreturned, the region's high-water mark the peak alive at once, the results the pool's bit for bit")}");
        return ok;
    }

    private sealed class ArenaProduct(GpuOps? ops, ProductView v, ulong da, ulong db, ulong dc, byte[]? a, byte[]? b, byte[]? c) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float> =>
            ops is null ? Kernels.MatMulView<TN>(v, a!, b!, c!) : ops.MatMulView<TN>(v, da, db, dc);
    }

    private sealed class ArenaAttention(GpuOps ops, AttentionArgs a, ulong q, ulong k, ulong v, ulong o) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float> => ops.Attention<TN>(a, q, k, v, o);
    }

    /// <summary>The CPU's narrow-type steps keep their weights widened per array (Recurrent.cs `Weights`): an LSTM and a GRU step
    /// (float16, bfloat16), called, then called again after W and U were changed in place (the same arrays), equal bit for
    /// bit to the step on fresh copies of the changed weights; and called a third time unchanged, equal again.</summary>
    private static bool WeightsInPlace()
    {
        bool ok = true;
        foreach (var type in new[] { "float16", "bfloat16" })
        foreach (var kind in new[] { "lstm", "gru" })
        {
            var a = new CellArgs(kind, 5, 7, 11);
            int GH = a.Gates * a.H;
            byte[] M(int n, int seed, double lo, double hi) => Bench.Registry.Make(type, n, seed, lo, hi);
            byte[] x = M(a.B * a.D, 91, -1, 1), h = M(a.B * a.H, 92, -1, 1), c = M(a.B * a.H, 93, -1, 1), w = M(GH * a.D, 94, -0.3, 0.3), u = M(GH * a.H, 95, -0.3, 0.3), b = M(GH, 96, -0.1, 0.1), bh = M(GH, 97, -0.1, 0.1);
            byte[] Step(byte[] ww, byte[] uu)
            {
                var hOut = new byte[h.Length]; var cOut = new byte[h.Length];
                Bench.Registry.WithFloat(type, new WeightsStep(a, x, h, c, ww, uu, b, bh, hOut, cOut));
                return [.. hOut, .. cOut];
            }

            Step(w, u);
            M(GH * a.D, 98, -0.3, 0.3).CopyTo(w, 0);
            M(GH * a.H, 99, -0.3, 0.3).CopyTo(u, 0);
            var changed = Step(w, u);
            var fresh = Step((byte[])w.Clone(), (byte[])u.Clone());
            var again = Step(w, u);
            if (!changed.AsSpan().SequenceEqual(fresh) || !again.AsSpan().SequenceEqual(fresh))
            {
                Console.WriteLine($"check weights in place  cpu  {type} {kind}: DIFFERS from the step on fresh weights");
                ok = false;
            }
        }

        if (ok) Console.WriteLine("check weights in place       cpu    float16, bfloat16: LSTM and GRU steps after W and U changed in place equal the steps on fresh copies");
        return ok;
    }

    private sealed class WeightsStep(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] hOut, byte[] cOut) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float> =>
            a.Kind == "lstm" ? Kernels.LstmStep<TN>(a, x, h, c, w, u, b, hOut, cOut) : Kernels.GruStep<TN>(a, x, h, w, u, b, bh, hOut);
    }

    // The LSTM step's h' then c' into one output (c' at cOff, a whole 256 bytes in).
    private sealed class ArenaLstm(GpuOps? ops, CellArgs a, ulong[]? d, ulong o, int hBytes, int cOff, byte[][]? x, byte[]? output) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            if (ops is not null) return ops.LstmStep<TN>(a, d![0], d[1], d[2], d[3], d[4], d[5], o, o + (ulong)cOff);
            byte[] h = new byte[hBytes], c = new byte[hBytes];
            string lane = Kernels.LstmStep<TN>(a, x![0], x[1], x[2], x[3], x[4], x[5], h, c);
            h.CopyTo(output!, 0); c.CopyTo(output!, cOff);
            return lane;
        }
    }

    // ---------------------------------------------------------------- the row operations on odd shapes

    /// <summary>Every float type, three shapes (37 × 1001; 5 × 64, a short row; 3 × 17, rows not ending on a word for the
    /// narrow floats): every row operation against float64 within the type's step plus the operation's ulps of its error
    /// scale; the parameter backward (both statistics) against float64 within its scale.</summary>
    public static bool Rows(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || Bench.Registry.IsBlock(type)) continue;
            foreach (var (rows, cols) in new[] { (37, 1001), (5, 64), (3, 17), (3, 20011), (2, 16384), (2, 32768) })
            {
                foreach (var op in RowOps.All)
                {
                    var k = Bench.RowCase.Make(op, type, rows, cols, positions: true);
                    var y = new byte[k.X.Length];
                    var y2 = op.SecondOut ? new byte[k.X.Length] : null;
                    Bench.Registry.WithFloat(type, new RowOpRun(ops, op, k, y, y2));
                    double worst = Math.Max(k.Worst(y), y2 is null ? 0 : k.Worst(y2, second: true));
                    if (worst > 0) { Console.WriteLine($"check row {op.Name,-18} {where,-6} {type,-12} DIFFERS by {worst:0.##} steps ({rows} x {cols})"); ok = false; }
                }

                foreach (bool centered in new[] { false, true })
                {
                    var x = Bench.Registry.Make(type, rows * cols, 51, -4, 4);
                    var dy = Bench.Registry.Make(type, rows * cols, 52, -1, 1);
                    var dgamma = new double[cols];
                    var dbeta = new double[cols];
                    double scale = RowOps.NormParamsReference(Bench.Registry.Values(type, x, rows * cols), Bench.Registry.Values(type, dy, rows * cols), rows, cols, 1e-5, centered, dgamma, dbeta);
                    var (got, lane) = Bench.Registry.WithFloat(type, new NormParamsRun(ops, x, dy, rows, cols, centered));
                    var g = Bench.LaneValues.Decode(lane, got, 2 * cols);
                    // The row statistic sums cols terms: its rounding, times the row's values, reaches every column sum.
                    double eps = lane == "f64" ? Math.Pow(2, -52) : Math.Pow(2, -23), tol = eps * scale * (8 + cols), worst = 0;
                    for (int i = 0; i < 2 * cols; i++) { double want = i < cols ? dgamma[i] : dbeta[i - cols]; if (Math.Abs(g[i] - want) > tol) worst = Math.Max(worst, Math.Abs(g[i] - want) / tol); }
                    if (worst > 0) { Console.WriteLine($"check norm_params_bwd:{(centered ? "layer" : "rms"),-6} {where,-6} {type,-12} DIFFERS by {worst:0.##} tolerances ({rows} x {cols}, lane {lane})"); ok = false; }
                }

                if (cols is 16384 or 32768)
                {
                    // The core softmax and rms_norm at the wide rows too: their row-in-registers kernels past a block's registers loop instead.
                    var x = Bench.Registry.Make(type, rows * cols, 61, -2, 2);
                    var vx = Bench.Registry.Values(type, x, rows * cols);
                    var output = new byte[x.Length];
                    Bench.Registry.WithFloat(type, new CoreRun(ops, "softmax", x, null, output, rows, cols));
                    ok &= Close(type, "softmax", where, output, References.Softmax(vx, rows, cols), rows * cols, terms: cols);
                    output = new byte[x.Length];
                    Bench.Registry.WithFloat(type, new CoreRun(ops, "rms_norm", x, null, output, rows, cols));
                    ok &= Close(type, "rms_norm", where, output, References.RmsNorm(vx, rows, cols, 1e-5), rows * cols, terms: cols);
                }
            }
        }

        if (ok) Console.WriteLine($"check rows                   {where,-6} every float type at 37 x 1001, 5 x 64, 3 x 17, 3 x 20011, 2 x 16384, 2 x 32768 (rows of 16 words a thread past a block's registers take the segmented shape): within the lane's rounding of the references");
        return ok;
    }

    private sealed class RowOpRun(GpuOps? ops, RowOp op, Bench.RowCase k, byte[] y, byte[]? y2) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.RowOp<TN>(op, k.X, k.X2, k.G, k.Bt, k.S, k.Pos, y, y2, k.Rows, k.Cols, op.A, op.B); return true; }
            var gpu = ops.Gpu;
            var rented = new List<ulong>();
            ulong Up(byte[]? a) { if (a is null) return 0; ulong d = gpu.Rent(a.Length); rented.Add(d); Qiyas.Gpu.Upload<byte>(a, d); return d; }
            ulong dx = Up(k.X), dx2 = Up(k.X2), dg = Up(k.G), dbt = Up(k.Bt), ds = Up(k.S), dpos = 0;
            if (k.Pos is not null) { dpos = gpu.Rent(k.Pos.Length * 4L); rented.Add(dpos); Qiyas.Gpu.Upload<int>(k.Pos, dpos); }
            ulong dy = gpu.Rent(y.Length), dy2 = y2 is null ? 0 : gpu.Rent(y2.Length);
            rented.Add(dy);
            if (dy2 != 0) rented.Add(dy2);
            ops.RowOp<TN>(op, dx, dx2, dg, dbt, ds, dpos, dy, dy2, k.Rows, k.Cols, op.A, op.B);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dy, y);
            if (y2 is not null) Qiyas.Gpu.Download<byte>(dy2, y2);
            foreach (var p in rented) gpu.Return(p);
            return true;
        }
    }

    private sealed class NormParamsRun(GpuOps? ops, byte[] x, byte[] dy, int rows, int cols, bool centered) : Bench.IFloatTypeVisitor<(byte[], string)>
    {
        public (byte[], string) Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            int size = TN.Bits == 64 ? 8 : 4;
            var dgamma = new byte[cols * size];
            var dbeta = new byte[cols * size];
            string lane;
            if (ops is null) lane = Kernels.NormParamsBwd<TN>(x, dy, dgamma, dbeta, rows, cols, 1e-5, centered);
            else
            {
                var gpu = ops.Gpu;
                ulong dx = gpu.Rent(x.Length), ddy = gpu.Rent(dy.Length), dg = gpu.Rent(dgamma.Length), db = gpu.Rent(dbeta.Length);
                Qiyas.Gpu.Upload<byte>(x, dx); Qiyas.Gpu.Upload<byte>(dy, ddy);
                lane = ops.NormParamsBwd<TN>(dx, ddy, dg, db, rows, cols, 1e-5, centered);
                Qiyas.Gpu.Synchronize();
                Qiyas.Gpu.Download<byte>(dg, dgamma); Qiyas.Gpu.Download<byte>(db, dbeta);
                gpu.Return(dx); gpu.Return(ddy); gpu.Return(dg); gpu.Return(db);
            }

            var output = new byte[2 * cols * size];
            dgamma.CopyTo(output, 0);
            dbeta.CopyTo(output, cols * size);
            return (output, lane);
        }
    }

    // ---------------------------------------------------------------- reductions, broadcasts, transposes, matrix · vector on odd shapes

    /// <summary>Every type, 37 × 1001 (rows and columns that end on no word for the packed types): reductions against float64
    /// sums (exact in the lane by the rule: integer results must match exactly, float ones within the lane's rounding),
    /// broadcasts and transposes against the type's rounding, gemv against float64 within the type's step.</summary>
    /// <summary>Repetitions of the gates that once failed intermittently (a schedule-dependent store): raised by --repeat.</summary>
    public static int Stress = 20;

    public static bool Group2(GpuOps? ops)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        const int R = 37, C = 1001;
        foreach (var type in Bench.Registry.Names)
        {
            var data = Bench.Registry.Make(type, R * C, 41, -1, 1);
            var values = Bench.Registry.Values(type, data, R * C);
            bool integer = !Bench.Registry.IsFloat(type);
            // Whole reductions also over a long odd length (97 × 4099: more parts than the team, parts not ending on a word);
            // its columns too (several strips and row blocks of the column tiles, the last of each partial).
            const int LR = 97, LC = 4099;
            var longData = Bench.Registry.Make(type, LR * LC, 44, -1, 1);
            var longValues = Bench.Registry.Values(type, longData, LR * LC);
            foreach (string op in new[] { "sum", "sumsq", "max", "min", "argmax" })
            {
                foreach (var (axis, r, c, d, vals) in new[] { (Axis.Whole, R, C, data, values), (Axis.Rows, R, C, data, values), (Axis.Cols, R, C, data, values), (Axis.Whole, LR, LC, longData, longValues), (Axis.Cols, LR, LC, longData, longValues) })
                {

                    int count = axis == Axis.Whole ? 1 : axis == Axis.Rows ? r : c;
                    var want = References.Reduce(op, axis, vals, r, c);
                    var (got, lane) = Bench.Registry.With(type, new ReduceRun(ops, op, axis, d, r, c));
                    var g = Bench.LaneValues.Decode(lane, got, count);
                    double worst = 0;
                    int terms = axis == Axis.Whole ? r * c : axis == Axis.Rows ? c : r;
                    for (int i = 0; i < count; i++)
                    {
                        // A float sum differs from the float64 reference by the lane's rounding over its terms (any order).
                        double eps = lane == "f32" ? 5.97e-8 : 1.12e-16;
                        double tol = integer || op is "max" or "min" or "argmax" ? 0 : (Math.Abs(want[i]) + 1) * eps * terms;
                        if (Math.Abs(g[i] - want[i]) > tol) worst = Math.Max(worst, Math.Abs(g[i] - want[i]));
                    }

                    if (worst > 0) { Console.WriteLine($"check {op}:{axis.ToString().ToLowerInvariant(),-6} {where,-6} {type,-12} DIFFERS by {worst:0.###e+0} (lane {lane}, {r} x {c})"); ok = false; }
                }
            }

            // Broadcasts (two formulas), transpose, gemv both ways.
            foreach (var opName in new[] { "add", "mul" })
            {
                var op = Qiyas.Ops.Find(opName);
                var v = Bench.Registry.Make(type, C, 42, -1, 1);
                var want = References.MapRow(type, op, values, Bench.Registry.Values(type, v, C), R, C);
                var output = new byte[data.Length];
                Bench.Registry.With(type, new MapRowRun(ops, op, data, v, output, R, C));
                ok &= Close(type, $"maprow:{opName}", where, output, want, R * C);
            }

            // The transpose of a block format repeated: its column tasks once shared blocks and re-encoded them in schedule order.
            for (int rep = 0, reps = Bench.Registry.IsBlock(type) ? Stress : 1; rep < reps; rep++)
            {
                var output = new byte[data.Length];
                Bench.Registry.With(type, new TransposeRun(ops, data, output, R, C));
                if (!Close(type, rep == 0 ? "transpose" : $"transpose (repetition {rep + 1})", where, output, References.Transpose(values, R, C), R * C)) { ok = false; break; }
            }

            foreach (bool t in new[] { false, true })
            {
                var x = Bench.Registry.Make(type, t ? R : C, 43, -1, 1);
                var want = References.Gemv(values, Bench.Registry.Values(type, x, t ? R : C), R, C, t);
                var y = new byte[Bench.Registry.Bytes(type, t ? C : R)];
                Bench.Registry.With(type, new GemvRun(ops, data, x, y, R, C, t));
                ok &= Close(type, t ? "gemv_t" : "gemv", where, y, want, t ? C : R, terms: t ? R : C);
            }
        }

        if (ok) Console.WriteLine($"check reductions, broadcasts, transposes, gemv {where,-6} every type at 37 × 1001: as the references");
        return ok;
    }

    // Stored output against a float64 reference rounded as the type rounds: integers exact, floats within one step of the type plus lane ulps.
    private static bool Close(string type, string what, string where, byte[] output, double[] reference, int n, int terms = 0)
    {
        var expected = Bench.Registry.Values(type, Bench.Registry.Encode(type, reference), n);
        var got = Bench.Registry.Values(type, output, n);
        int mant = Bench.Registry.Mantissa(type), laneMant = type == "float64" ? 52 : 23;
        int blockValues = type switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 0 };
        double[]? blockMax = null;
        if (blockValues > 0)
        {
            // A block format's step is set by its block's largest value: the tolerance is the block's, and a block whose values
            // were produced one rounding apart may take the next scale: two steps.
            blockMax = new double[(n + blockValues - 1) / blockValues];
            for (int i = 0; i < n; i++) blockMax[i / blockValues] = Math.Max(blockMax[i / blockValues], Math.Abs(reference[i]));
        }

        double worst = 0;
        for (int i = 0; i < n; i++)
        {
            double want = expected[i], v = got[i];
            bool debug = Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1";
            if (mant == 0)
            {
                if (v != want)
                {
                    if (debug && worst == 0) Console.WriteLine($"  {what} {type} [{i}] got {v:R} want {want:R} ref {reference[i]:R}");
                    worst = Math.Max(worst, Math.Abs(v - want));
                }

                continue;
            }

            if (blockMax is not null)
            {
                double bstep = (blockMax[i / blockValues] == 0 ? 1 : blockMax[i / blockValues]) * Math.Pow(2, -mant);
                double btol = 2 * bstep + 4 * Math.Abs(reference[i]) * Math.Pow(2, -laneMant);
                if (Math.Abs(v - want) > btol)
                {
                    if (debug && worst == 0)
                    {
                        Console.WriteLine($"  {what} {type} [{i}] got {v:R} want {want:R} ref {reference[i]:R} block max {blockMax[i / blockValues]:R}");
                        for (int j = Math.Max(0, i - 12); j < Math.Min(n, i + 36); j++) Console.WriteLine($"    [{j}] got {got[j]:R} want {expected[j]:R}");
                    }
                    worst = Math.Max(worst, Math.Abs(v - want) / bstep);
                }

                continue;
            }

            double scale = reference[i] == 0 ? 1 : Math.Pow(2, Math.Floor(Math.Log2(Math.Abs(reference[i]))));
            // A matrix · vector result is a sum of `terms` products of magnitude at most 1 in the lane: its rounding is the sum's, not the result's.
            double tol = Math.Max(scale * Math.Pow(2, -mant), Bench.Registry.MinStep(type)) + 4 * scale * Math.Pow(2, -laneMant) + terms * Math.Pow(2, -laneMant) * 4;
            if (Math.Abs(v - want) > tol)
            {
                if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1" && worst == 0) Console.WriteLine($"  {what} {type} [{i}] got {v:R} want {want:R} ref {reference[i]:R}");
                worst = Math.Max(worst, Math.Abs(v - want) / (scale * Math.Pow(2, -mant)));
            }
        }

        if (worst > 0) Console.WriteLine($"check {what,-12} {where,-6} {type,-12} DIFFERS by {worst:0.##} steps");
        return worst == 0;
    }

    private sealed class ReduceRun(GpuOps? ops, string op, Axis axis, byte[] x, int rows, int cols) : Bench.ITypeVisitor<(byte[], string)>
    {
        public (byte[], string) Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null) return Kernels.Reduce<TN>(op, axis, x, rows, cols);
            var gpu = ops.Gpu;
            int count = axis == Axis.Whole ? 1 : axis == Axis.Rows ? rows : cols;
            ulong dx = gpu.Rent(x.Length), dout = gpu.Rent(count * 8L);
            Qiyas.Gpu.Upload<byte>(x, dx);
            string lane = ops.Reduce<TN>(op, axis, dx, rows, cols, dout);
            Qiyas.Gpu.Synchronize();
            var output = new byte[count * 8];
            Qiyas.Gpu.Download<byte>(dout, output);
            gpu.Return(dx); gpu.Return(dout);
            return (output, lane);
        }
    }

    private sealed class MapRowRun(GpuOps? ops, MapOp op, byte[] x, byte[] v, byte[] output, int rows, int cols) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.MapRow<TN>(op.Name, x, v, output, rows, cols); return true; }
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dv = gpu.Rent(v.Length), dout = gpu.Rent(output.Length);
            Qiyas.Gpu.Upload<byte>(x, dx); Qiyas.Gpu.Upload<byte>(v, dv);
            ops.MapRow<TN>(op, dx, dv, dout, rows, cols);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dout, output);
            gpu.Return(dx); gpu.Return(dv); gpu.Return(dout);
            return true;
        }
    }

    private sealed class TransposeRun(GpuOps? ops, byte[] x, byte[] output, int rows, int cols) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.Transpose<TN>(x, output, rows, cols); return true; }
            var gpu = ops.Gpu;
            ulong dx = gpu.Rent(x.Length), dout = gpu.Rent(output.Length);
            Qiyas.Gpu.Upload<byte>(x, dx);
            Qiyas.Gpu.Clear(dout, output.Length);
            ops.Transpose<TN>(dx, dout, rows, cols);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dout, output);
            gpu.Return(dx); gpu.Return(dout);
            return true;
        }
    }

    private sealed class GemvRun(GpuOps? ops, byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (ops is null) { Kernels.Gemv<TN>(a, x, y, rows, cols, transposed); return true; }
            var gpu = ops.Gpu;
            ulong da = gpu.Rent(a.Length), dx = gpu.Rent(x.Length), dy = gpu.Rent(y.Length);
            Qiyas.Gpu.Upload<byte>(a, da); Qiyas.Gpu.Upload<byte>(x, dx);
            ops.Gemv<TN>(da, dx, dy, rows, cols, transposed);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dy, y);
            gpu.Return(da); gpu.Return(dx); gpu.Return(dy);
            return true;
        }
    }

    // ---------------------------------------------------------------- functions on vectors against scalar references, in ulps

    private static bool Function32(string name, Func<Vector<float>, Vector<float>> f, Func<double, double> reference, double low, double high, double maxUlp, Random random, bool logScale = false)
    {
        int w = Vector<float>.Count, worstAt = 0;
        double worst = 0;
        var x = new float[w * 2048];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(logScale ? Math.Exp(Math.Log(low) + (Math.Log(high) - Math.Log(low)) * random.NextDouble()) : low + (high - low) * random.NextDouble());
        for (int i = 0; i < x.Length; i += w)
        {
            var y = f(new Vector<float>(x.AsSpan(i)));
            for (int l = 0; l < w; l++)
            {
                double want = reference(x[i + l]);
                float wantF = (float)want;
                double ulp = Math.Abs(MathF.BitIncrement(Math.Abs(wantF)) - Math.Abs(wantF));
                double err = wantF == 0 ? Math.Abs(y[l]) / float.Epsilon : Math.Abs(y[l] - want) / ulp;
                if (err > worst) { worst = err; worstAt = i + l; }
            }
        }

        Console.WriteLine($"check {name + " (ulps)",-36} {x.Length,9} values: worst {worst:0.00} ulp at x = {x[worstAt]:R}{(worst <= maxUlp ? "" : "  TOO FAR")}");
        return worst <= maxUlp;
    }

    private static bool Function64(string name, Func<Vector<double>, Vector<double>> f, Func<double, double> reference, double low, double high, double maxUlp, Random random, bool logScale = false)
    {
        int w = Vector<double>.Count, worstAt = 0;
        double worst = 0;
        var x = new double[w * 2048];
        for (int i = 0; i < x.Length; i++) x[i] = logScale ? Math.Exp(Math.Log(low) + (Math.Log(high) - Math.Log(low)) * random.NextDouble()) : low + (high - low) * random.NextDouble();
        for (int i = 0; i < x.Length; i += w)
        {
            var y = f(new Vector<double>(x.AsSpan(i)));
            for (int l = 0; l < w; l++)
            {
                double want = reference(x[i + l]);
                double ulp = Math.BitIncrement(Math.Abs(want)) - Math.Abs(want);
                double err = want == 0 ? Math.Abs(y[l]) / double.Epsilon : Math.Abs(y[l] - want) / ulp;
                if (err > worst) { worst = err; worstAt = i + l; }
            }
        }

        Console.WriteLine($"check {name + " (ulps)",-36} {x.Length,9} values: worst {worst:0.00} ulp at x = {x[worstAt]:R}{(worst <= maxUlp ? "" : "  TOO FAR")}");
        return worst <= maxUlp;
    }

    // ---------------------------------------------------------------- every element-wise formula on every type it applies to

    /// <summary>Each formula over 1001 values of each of its types, on the CPU (ops null) or the device, against the float64
    /// reference rounded as the type rounds: integer types exact; float types within one step of the type plus a few ulps of
    /// the lane (the transcendental functions' own error).</summary>
    public static bool Maps(GpuOps? ops)
    {
        const int N = 1001;
        bool ok = true;
        string where = ops is null ? "cpu" : "device";
        foreach (var op in Ops.All)
        {
            double worst = 0;
            string worstType = "";
            bool failed = false;
            foreach (var type in Bench.Registry.Names)
            {
                if (Bench.Registry.IsBlock(type)) continue;                    // measured by the block formats' gate (block steps)
                if (type == "bool" && op.Name == "div") continue;              // bool: a divisor of 0 is half its values: no division (n/a)
                if (op.FloatOnly && !Bench.Registry.IsFloat(type)) continue;
                byte[]? x = op.Inputs >= 1 ? Bench.Registry.Make(type, N, 31, op.XRange.Low, op.XRange.High) : null;
                byte[]? y = op.Inputs >= 2 ? Bench.Registry.Make(type, N, 32, op.YRange.Low, op.YRange.High) : null;
                byte[]? z = op.Inputs >= 3 ? Bench.Registry.Make(type, N, 33, op.YRange.Low, op.YRange.High) : null;
                var output = new byte[Bench.Registry.Bytes(type, N)];
                var reference = References.Map(type, op, x, y, z, N, wrap: false);
                if (ops is null) Bench.Registry.With(type, new CpuMap(op, x, y, z, output, N));
                else Bench.Registry.With(type, new DeviceMap(ops, op, x, y, z, output, N));
                int mant = Bench.Registry.Mantissa(type);
                int laneMant = type == "float64" ? 52 : 23;
                if (mant == 0)
                {
                    // Integers: exact, modulo 2^bits, compared in 128-bit integers (64-bit values read straight from the bytes).
                    int bits = Bench.Registry.Bits(type);
                    bool signed = type.StartsWith("int", StringComparison.Ordinal);
                    var got64 = bits == 64 ? (signed ? MemoryMarshal.Cast<byte, long>(output).ToArray().Select(v => (Int128)v).ToArray() : MemoryMarshal.Cast<byte, ulong>(output).ToArray().Select(v => (Int128)v).ToArray()) : null;
                    var gotValues = got64 ?? Bench.Registry.Values(type, output, N).Select(v => (Int128)v).ToArray();
                    Int128 m = Int128.One << bits;
                    for (int i = 0; i < N; i++)
                    {
                        Int128 want = (Int128)reference[i];
                        want = ((want % m) + m) % m;
                        if (signed && want >= m / 2) want -= m;
                        if (gotValues[i] != want && worst == 0) { worst = (double)(gotValues[i] - want); worstType = type; }
                    }

                    continue;
                }

                var expected = Bench.Registry.Values(type, Bench.Registry.Encode(type, reference), N);
                var got = Bench.Registry.Values(type, output, N);
                for (int i = 0; i < N; i++)
                {
                    double e, want = expected[i], v = got[i];
                    if (double.IsNaN(want) || double.IsInfinity(want)) { e = (double.IsNaN(v) && double.IsNaN(want)) || v == want ? 0 : double.PositiveInfinity; }
                    else
                    {
                        double scale = Math.Abs(reference[i]) == 0 ? 1 : Math.Pow(2, Math.Floor(Math.Log2(Math.Abs(reference[i]))));
                        double stepType = scale * Math.Pow(2, -mant), stepLane = scale * Math.Pow(2, -laneMant);
                        double tol = stepType + op.Ulps * stepLane;
                        e = Math.Abs(v - want) <= tol ? 0 : Math.Abs(v - want) / stepType;
                    }

                    if (e > worst)
                    {
                        worst = e; worstType = type;
                        if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1") Console.WriteLine($"  {op.Name} {type} [{i}] x {(x is null ? 0 : Bench.Registry.Values(type, x, N)[i]):R} got {v:R} want {want:R} ref {reference[i]:R}: {e:0.##} steps");
                    }
                }

                if (worst > 0 && !failed) failed = true;
            }

            Console.WriteLine($"check map {op.Name,-18} {where,-6} {(worst == 0 ? "every type exact to its step" : $"DIFFERS by {worst:0.##} steps in {worstType}")}");
            ok &= worst == 0;
        }

        return ok;
    }

    private sealed class CpuMap(MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            if (Bench.Registry.IsFloat(TN.Name)) Bench.Registry.WithFloat(TN.Name, new CpuFloatMap(op, x, y, z, output, n));
            else Kernels.Map<TN>(op.Name, x, y, z, output, n, op.A, op.B);
            return true;
        }
    }

    private sealed class CpuFloatMap(MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasFloat<TN>, IQiyasNumber<TN, float>
        {
            Kernels.MapFloat<TN>(op.Name, x, y, z, output, n, op.A, op.B);
            return true;
        }
    }

    private sealed class DeviceMap(GpuOps ops, MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            var gpu = ops.Gpu;
            var rented = new List<ulong>();
            ulong Up(byte[]? array)
            {
                if (array is null) return 0;
                ulong d = gpu.Rent(array.Length);
                rented.Add(d);
                Qiyas.Gpu.Upload<byte>(array, d);
                return d;
            }

            ulong dx = Up(x), dy = Up(y), dz = Up(z), dout = gpu.Rent(output.Length);
            rented.Add(dout);
            ops.Map<TN>(op, dx, dy, dz, dout, n, op.A, op.B);
            Qiyas.Gpu.Synchronize();
            Qiyas.Gpu.Download<byte>(dout, output);
            foreach (var p in rented) gpu.Return(p);
            return true;
        }
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
        ok &= Sliced(ops);
        ok &= Maps(ops);
        ok &= Group2(ops);
        ok &= Rows(ops);
        ok &= Layouts(ops);
        ok &= Products(ops);
        ok &= Attentions(ops);
        ok &= Optimizers(ops);
        ok &= Convs(ops);
        ok &= Recurrents(ops);
        ok &= Routings(ops);
        ok &= Ctcs(ops);
        ok &= BlockFormats(ops);
        ok &= Arenas(ops);
        ok &= WholeSums(ops);
        return ok;
    }

    // The sliced float64 product (the 8-bit tensor lane) against float64 math accumulated in double-double on the CPU:
    // the worst error in ulps of the result, over values of one scale and over rows and columns of scales 2^-20 .. 2^20.
    private static bool Sliced(GpuOps ops)
    {
        const int M = 96, N = 80, K = 1000;
        var random = new Random(23);
        bool ok = true;
        foreach (bool scaled in new[] { false, true })
        {
            var a = new double[M * K];
            var b = new double[K * N];
            for (int i = 0; i < M; i++)
            {
                double s = scaled ? Math.Pow(2, i % 41 - 20) : 1;
                for (int p = 0; p < K; p++) a[i * K + p] = (random.NextDouble() * 2 - 1) * s;
            }

            for (int p = 0; p < K; p++)
            {
                for (int j = 0; j < N; j++) b[p * N + j] = (random.NextDouble() * 2 - 1) * (scaled ? Math.Pow(2, j % 37 - 18) : 1);
            }

            var gpu = ops.Gpu;
            ulong da = gpu.Rent(a.Length * 8L), db = gpu.Rent(b.Length * 8L), dc = gpu.Rent((long)M * N * 8);
            Qiyas.Gpu.Upload<double>(a, da);
            Qiyas.Gpu.Upload<double>(b, db);
            bool ran = ops.MatMulSliced(da, db, dc, M, N, K);
            Qiyas.Gpu.Synchronize();
            var got = new double[M * N];
            if (ran) Qiyas.Gpu.Download<double>(dc, got);
            gpu.Return(da); gpu.Return(db); gpu.Return(dc);
            if (!ran)
            {
                Console.WriteLine($"check {"sliced float64 product",-36} not on this device: never chosen");
                return true;
            }

            double worst = 0;
            int worstAt = 0;
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double hi = 0, lo = 0;
                    for (int p = 0; p < K; p++)
                    {
                        double x = a[i * K + p], y = b[p * N + j];
                        double ph = x * y, pl = Math.FusedMultiplyAdd(x, y, -ph);
                        double sum = hi + ph, bv = sum - hi, err = (hi - (sum - bv)) + (ph - bv);
                        hi = sum;
                        lo += err + pl;
                    }

                    double reference = hi + lo, value = got[i * N + j];
                    double ulp = reference == 0 ? double.Epsilon : Math.BitIncrement(Math.Abs(reference)) - Math.Abs(reference);
                    double e = Math.Abs(value - reference) / ulp;
                    if (e > worst) { worst = e; worstAt = i * N + j; }
                }
            }

            string name = scaled ? "sliced float64 (rows, columns scaled)" : "sliced float64 product";
            Console.WriteLine($"check {name,-36} {M * N,9} values: worst {worst:0.000} ulp at {worstAt}{(worst <= 1 ? "" : "  TOO FAR")}");
            ok &= worst <= 1;
        }

        ops.SlicedProven = ok;
        return ok;
    }

    private static bool Device<TN, TL>(GpuOps ops, Lane lane) where TN : struct, IQiyasNumber<TN>, IQiyasNumber<TN, TL> where TL : unmanaged
    {
        const int N = 1 << 16;
        var random = new Random(11);
        var stored = new byte[TN.Bytes(N)];
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
