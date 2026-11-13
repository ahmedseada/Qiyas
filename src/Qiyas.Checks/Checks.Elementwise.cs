// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The element-wise operations past the formulas, on the CPU against their references and on a device against the CPU:
// dropout from a seed (every type but bool, refused), the conversion between every pair of types (against an exact
// reference: the source's value, as
// a dyadic rational or an integer, rounded once into the target by the format's definition), and a float formula over a
// row vector.
using System.Numerics;
using System.Runtime.InteropServices;

namespace Qiyas;

internal static partial class Checks
{
    public static bool Elementwise(IDevice<ulong, ulong>? ops, string device = "device")
    {
        bool ok = true;
        ok &= Dropouts(ops, device);
        ok &= Converts(ops, device);
        ok &= RowFloats(ops, device);
        return ok;
    }

    // ---------------------------------------------------------------- dropout

    /// <summary>Every type but bool at 1001 and 4099 values, with and without a residual: the CPU within the type's rounding of
    /// keep(i) ? x/(1 − p) + r : r in float64 (an integer's kept value within half a step plus the float lane's rounding, a
    /// dropped one the residual exactly as stored, 64-bit integers past 2^53), the device the CPU's values; the draws' rate
    /// and independence.</summary>
    private static bool Dropouts(IDevice<ulong, ulong>? ops, string device)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        const double P = 0.3;
        const ulong Seed = 0x1234_5678_9ABC_DEF0;
        if (ops is null)
        {
            // The draws: the kept share within 5σ of 1 − p; the masks of seeds one apart in either word, and of neighbouring
            // values, agreeing as independent ones would; a sampler's streams one step apart the same.
            const int M = 1 << 20;
            var k1 = Draws.Key(Seed);
            DrawKey k2 = Draws.Key(Seed + 1), k3 = Draws.Key(Seed + (1UL << 32)), s1 = Draws.Stream(k1, 7), s2 = Draws.Stream(k1, 8);
            uint thr = Draws.Threshold(P);
            int kept = 0;
            var agree = new int[4];
            for (uint i = 0; i < M; i++)
            {
                bool a = Draws.Keep(k1, i, thr);
                if (a) kept++;
                if (a == Draws.Keep(k2, i, thr)) agree[0]++;
                if (a == Draws.Keep(k3, i, thr)) agree[1]++;
                if (a == Draws.Keep(k1, i + 1, thr)) agree[2]++;
                if (Draws.Keep(s1, i, thr) == Draws.Keep(s2, i, thr)) agree[3]++;
            }

            double share = kept / (double)M, sigma = Math.Sqrt(P * (1 - P) / M), pSame = P * P + (1 - P) * (1 - P), sameSigma = Math.Sqrt(pSame * (1 - pSame) / M);
            string[] pairs = ["seeds one apart", "seeds 2^32 apart", "neighbouring values", "streams one step apart"];
            for (int t = 0; t < 4; t++)
            {
                double same = agree[t] / (double)M;
                if (Math.Abs(same - pSame) <= 5 * sameSigma) continue;
                Console.WriteLine($"check dropout draws          cpu    DIFFERS: {pairs[t]} agree {same:0.#####} (want {pSame:0.#####})");
                ok = false;
            }

            if (Math.Abs(share - (1 - P)) > 5 * sigma)
            {
                Console.WriteLine($"check dropout draws          cpu    DIFFERS: kept {share:0.#####} (want {1 - P})");
                ok = false;
            }

            // Over a key's 2^32 draws exactly 2^32 − threshold are kept (Bits a bijection of the counter).
            long all = 0;
            Parallel.For(0, 1 << 16, () => 0L, (hi, _, sum) =>
            {
                uint at = (uint)hi << 16;
                for (uint lo = 0; lo < 1u << 16; lo++) if (Draws.Keep(k1, at + lo, thr)) sum++;
                return sum;
            }, sum => Interlocked.Add(ref all, sum));
            if (all != (1L << 32) - thr)
            {
                Console.WriteLine($"check dropout draws          cpu    DIFFERS: {all} of 2^32 draws kept (want {(1L << 32) - thr})");
                ok = false;
            }
        }

        foreach (var type in Bench.Registry.Names)
        {
            if (Bench.Registry.Bits(type) == 1) continue;
            bool integer = !Bench.Registry.IsFloat(type), wide = integer && Bench.Registry.Bits(type) == 64;
            foreach (int n in new[] { 1001, 4099 })
                foreach (bool residual in new[] { false, true })
                {
                    var x = wide ? Wide64(type, n, 71) : Bench.Registry.Make(type, n, 71, -4, 4);
                    var r = !residual ? null : wide ? Wide64(type, n, 72) : Bench.Registry.Make(type, n, 72, -2, 2);
                    var cpu = Bench.Registry.Poisoned(type, n);
                    Bench.Registry.With(type, new DropoutRun(null, x, r, cpu, n, P, Seed));
                    string what = residual ? "dropout+residual" : "dropout";
                    if (ops is null)
                    {
                        var xv = Bench.Registry.Values(type, x, n);
                        var rv = r is null ? null : Bench.Registry.Values(type, r, n);
                        var key = Draws.Key(Seed);
                        uint thr = Draws.Threshold(P);
                        var want = new double[n];
                        var terms = new double[n];
                        for (int i = 0; i < n; i++)
                        {
                            double kept = Draws.Keep(key, (uint)i, thr) ? xv[i] / (1 - P) : 0;
                            want[i] = kept + (rv?[i] ?? 0);
                            terms[i] = Math.Abs(kept) + Math.Abs(rv?[i] ?? 0);
                        }

                        // Within the type's step and two lane steps of the terms: s read into the lane, and the sum's rounding
                        // (an integer's within half a step); a dropped value the residual as stored.
                        double worst = integer ? IntegerExcess(type, cpu, want, i => terms[i], 2) : Excess(type, what, cpu, want, i => terms[i], 2);
                        if (worst > 0) { Console.WriteLine($"check {what} n={n,-12} {where,-6} {type,-12} DIFFERS by {worst:0.##} steps"); ok = false; }
                        if (integer)
                        {
                            int size = Bench.Registry.Bits(type) / 8;
                            var got = Bench.Registry.Values(type, cpu, n);
                            for (int i = 0; i < n; i++)
                            {
                                if (Draws.Keep(key, (uint)i, thr)) continue;
                                bool same = size > 0 && r is not null ? cpu.AsSpan(i * size, size).SequenceEqual(r.AsSpan(i * size, size)) : got[i] == (rv?[i] ?? 0);
                                if (same) continue;
                                Console.WriteLine($"check {what} n={n,-12} {where,-6} {type,-12} DIFFERS: dropped value {i} is not the residual as stored");
                                ok = false;
                                break;
                            }
                        }

                        continue;
                    }

                    var dev = Bench.Registry.Poisoned(type, n);
                    Bench.Registry.With(type, new DropoutRun(ops, x, r, dev, n, P, Seed));
                    ok &= SameValues(type, $"{what} n={n}", where, dev, cpu, n);
                }
        }

        if (ok) Console.WriteLine($"check dropout                {where,-6} every type but bool, 1001 and 4099 values, with and without a residual: {(ops is null ? "within the type's rounding of keep(i) ? x/(1 − p) + r : r (a dropped integer the residual as stored, 64-bit integers past 2^53); the draws' rate and independence" : "the CPU's values")}");
        return ok;
    }

    private sealed class DropoutRun(IDevice<ulong, ulong>? ops, byte[] x, byte[]? r, byte[] output, int n, double p, ulong seed) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IAdad<TN, float>
        {
            if (ops is null) { CpuDevice.Device.Dropout<TN>(x, r, output, n, p, seed); return true; }
            ulong dx = ops.Rent(x.Length + 16), dr = r is null ? 0 : ops.Rent(r.Length + 16), dout = ops.Rent(output.Length + 16);
            try
            {
                ops.Upload(x, dx); ops.Upload(output, dout);
                if (r is not null) ops.Upload(r, dr);
                ops.Dropout<TN>(dx, dr, dout, n, p, seed);
                ops.Synchronize();
                ops.Download(dout, output);
                return true;
            }
            finally { ops.Return(dx); if (dr != 0) ops.Return(dr); ops.Return(dout); }
        }
    }

    // A device's stored values against the CPU's: the same values (a NaN any NaN), a block format's bytes.
    private static bool SameValues(string type, string what, string where, byte[] got, byte[] want, int n)
    {
        if (Bench.Registry.IsBlock(type))
        {
            int bytes = (int)Bench.Registry.Bytes(type, n), diff = 0;
            for (int i = 0; i < bytes; i++) if (got[i] != want[i]) diff++;
            if (diff == 0) return true;
            Console.WriteLine($"check {what,-24} {where,-6} {type,-12} DIFFERS from the CPU at {diff} bytes");
            return false;
        }

        var g = Exacts(type, got, n);
        var w = Exacts(type, want, n);
        for (int i = 0; i < n; i++)
        {
            if (g[i].Equals(w[i])) continue;
            Console.WriteLine($"check {what,-24} {where,-6} {type,-12} DIFFERS from the CPU at [{i}] of {n}: {g[i]} against {w[i]}");
            return false;
        }

        return true;
    }

    // ---------------------------------------------------------------- conversion

    // A stored value exactly: NaN, ±∞, or (−1)^Neg · M · 2^E.
    private readonly record struct ExactValue(bool NaN, bool Inf, bool Neg, BigInteger M, int E)
    {
        public static ExactValue Of(double d)
        {
            if (double.IsNaN(d)) return new(true, false, false, 0, 0);
            bool neg = double.IsNegative(d);
            if (double.IsInfinity(d)) return new(false, true, neg, 0, 0);
            long bits = BitConverter.DoubleToInt64Bits(d);
            int e = (int)((bits >> 52) & 0x7FF);
            long m = bits & 0xFFFFFFFFFFFFFL;
            return e == 0 ? new(false, false, neg, m, -1074) : new(false, false, neg, m | (1L << 52), e - 1075);
        }

        public static ExactValue Of(Int128 v) => new(false, false, v < 0, (BigInteger)(v < 0 ? -v : v), 0);

        // Equal values: NaNs alike, zeros by sign, numbers by value.
        public bool Equals(ExactValue o)
        {
            if (NaN || o.NaN) return NaN && o.NaN;
            if (Inf || o.Inf) return Inf == o.Inf && Neg == o.Neg;
            if (M.IsZero || o.M.IsZero) return M.IsZero && o.M.IsZero && Neg == o.Neg;
            int e = Math.Min(E, o.E);
            return Neg == o.Neg && (M << (E - e)) == (o.M << (o.E - e));
        }

        public override int GetHashCode() => 0;

        public override string ToString() => NaN ? "NaN" : Inf ? (Neg ? "-inf" : "inf") : $"{(Neg ? "-" : "")}{(double)M * Math.Pow(2, E):R}";

        public Int128 Integer => Neg ? -(Int128)(M << E) : (Int128)(M << E);    // a whole value's integer (E ≥ 0)
    }

    // The values stored, exactly (64-bit integers from their bytes; every other type is exact as float64).
    private static ExactValue[] Exacts(string type, byte[] stored, int n)
    {
        if (type == "int64") return [.. MemoryMarshal.Cast<byte, long>(stored.AsSpan(0, n * 8)).ToArray().Select(v => ExactValue.Of((Int128)v))];
        if (type == "uint64") return [.. MemoryMarshal.Cast<byte, ulong>(stored.AsSpan(0, n * 8)).ToArray().Select(v => ExactValue.Of((Int128)v))];
        var d = Bench.Registry.Values(type, stored, n);
        return Bench.Registry.IsFloat(type) ? [.. d.Select(ExactValue.Of)] : [.. d.Select(v => ExactValue.Of((Int128)v))];
    }

    // A float format: significand bits (with the leading one), the smallest normal exponent, the largest finite value,
    // and whether past it (from its rounding point, max + ulp/2) it is NaN (E4M3) rather than ∞.
    private static (int P, int EMin, double Max, bool NanPast) FloatFormat(string type) => type switch
    {
        "float64" => (53, -1022, double.MaxValue, false),
        "float32" => (24, -126, float.MaxValue, false),
        "float16" => (11, -14, 65504, false),
        "bfloat16" => (8, -126, 3.3895313892515355e38, false),
        "float8_e4m3" => (4, -6, 448, true),
        "float8_e5m2" => (3, -14, 57344, false),
        _ => throw new ArgumentException(type),
    };

    // x rounded once into a float format: the nearest value of P significant bits (ties to the even significand),
    // subnormal at 2^(EMin − P + 1); from max + ulp(max)/2 on ∞ (or NaN).
    private static ExactValue RoundFloat(string type, ExactValue x)
    {
        var f = FloatFormat(type);
        if (x.NaN) return x;
        if (x.Inf) return f.NanPast ? new(true, false, false, 0, 0) : x;
        if (x.M.IsZero) return x;
        int top = (int)(x.M.GetBitLength() - 1) + x.E;
        int q = Math.Max(top, f.EMin) - (f.P - 1);
        BigInteger r;
        if (x.E >= q) r = x.M << (x.E - q);
        else
        {
            int sh = q - x.E;
            r = x.M >> sh;
            var rem = x.M - (r << sh);
            var half = BigInteger.One << (sh - 1);
            if (rem > half || (rem == half && !r.IsEven)) r += 1;
        }

        // The overflow point: max + ulp(max)/2 (E4M3's max is not all ones: its last code is NaN).
        int maxTop = Math.ILogB(f.Max), halfUlp = maxTop - f.P;
        var mx = ExactValue.Of(f.Max);
        int ce = Math.Min(Math.Min(x.E, mx.E), halfUlp);
        if ((x.M << (x.E - ce)) >= (mx.M << (mx.E - ce)) + (BigInteger.One << (halfUlp - ce))) return f.NanPast ? new(true, false, false, 0, 0) : new(false, true, x.Neg, 0, 0);
        return new(false, false, x.Neg, r, q);
    }

    // x rounded once into an integer type: NaN 0, otherwise to the nearest integer (ties to even), held to the range.
    private static ExactValue RoundInteger(string type, ExactValue x)
    {
        int bits = Bench.Registry.Bits(type);
        bool signed = type.StartsWith("int", StringComparison.Ordinal);
        Int128 lo = signed ? -(Int128.One << (bits - 1)) : 0, hi = signed ? (Int128.One << (bits - 1)) - 1 : (Int128.One << bits) - 1;
        if (x.NaN) return ExactValue.Of(Int128.Zero);
        if (x.Inf) return ExactValue.Of(x.Neg ? lo : hi);
        BigInteger r;
        if (x.E >= 0) r = x.M << x.E;
        else
        {
            int sh = -x.E;
            r = x.M >> sh;
            var rem = x.M - (r << sh);
            var half = BigInteger.One << (sh - 1);
            if (rem > half || (rem == half && !r.IsEven)) r += 1;
        }

        var v = x.Neg ? -r : r;
        if (v < (BigInteger)lo) v = (BigInteger)lo;
        if (v > (BigInteger)hi) v = (BigInteger)hi;
        return ExactValue.Of((Int128)v);
    }

    // Source values of a type: its whole range with the special values and the targets' rounding points (finite: magnitudes
    // a block format's scales hold, no specials).
    private static byte[] ConvertSource(string type, int n, int seed, bool finite)
    {
        var random = new Random(seed);
        if (Bench.Registry.IsBlock(type)) return Bench.Registry.Make(type, n, seed, -4, 4);
        if (type is "int64" or "uint64")
        {
            var u = new ulong[n];
            ulong[] special = [0, 1, ulong.MaxValue, (1UL << 53) + 1, (1UL << 53) - 1, (1UL << 24) + 1, (1UL << 24) + 3, (1UL << 63) - 1, 1UL << 63, (1UL << 63) + 1,
                0x7FFFFF8000000000, 0x7FFFFF7FFFFFFFFF, 0xFFFFFFFFFFFFFC00, unchecked((ulong)-((1L << 24) + 1)), unchecked((ulong)-((1L << 40) + 12345)), 65520, 65519, 464, 463];
            for (int i = 0; i < n; i++)
            {
                if (!finite && i < special.Length) { u[i] = special[i]; continue; }
                int kind = random.Next(4);
                u[i] = kind switch
                {
                    0 => (ulong)random.NextInt64(long.MinValue, long.MaxValue),
                    1 => (ulong)random.NextInt64(-1000, 1000),
                    2 => (ulong)(random.NextInt64(1L << 20, 1L << 60) * (random.Next(2) == 0 ? 1 : -1)),
                    _ => (ulong)random.NextInt64(-(1L << 30), 1L << 30),
                };
                if (type == "uint64" && kind == 1) u[i] = (ulong)random.Next(0, 1000);
                if (finite) u[i] = (ulong)((long)u[i] % 10000);
            }

            return MemoryMarshal.AsBytes(u.AsSpan()).ToArray();
        }

        var v = new double[n];
        if (Bench.Registry.IsFloat(type))
        {
            double[] special = [0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN, double.Epsilon, -double.Epsilon, double.MaxValue, 2.2250738585072014e-308,
                464, 463.99999999, 448, 456, 65520, 65519.99999, 65504, 61440, 61439.999, 57344, 3.4028235677973366e38, 3.40282357e38, 3.3895313892515355e38, 3.396e38,
                1.401298464324817e-45, 7.006492321624085e-46, 5.960464477539063e-8, 2.9802322387695312e-8, 1 + 1.0 / 2048, 1 + 3.0 / 2048, 1 + 1.0 / 256, 1 + 1.0 / 16, 1 + 1.0 / 8, 16777217, 2147483648.5, -2147483648.5];
            for (int i = 0; i < n; i++)
            {
                if (!finite && i < special.Length) { v[i] = special[i]; continue; }
                double mag = (1 + random.NextDouble()) * Math.Pow(2, finite ? random.Next(-12, 8) : type == "float64" && random.Next(4) == 0 ? random.Next(-1070, 1020) : random.Next(-40, 40));
                v[i] = random.Next(2) == 0 ? mag : -mag;
            }
        }
        else
        {
            int bits = Bench.Registry.Bits(type);
            bool signed = type.StartsWith("int", StringComparison.Ordinal);
            double lo = type == "bool" ? 0 : signed ? -Math.Pow(2, bits - 1) : 0, hi = type == "bool" ? 1 : signed ? Math.Pow(2, bits - 1) - 1 : Math.Pow(2, bits) - 1;
            for (int i = 0; i < n; i++)
            {
                v[i] = i switch { 0 => lo, 1 => hi, 2 => 0, 3 => Math.Min(hi, 1), _ => Math.Floor(lo + random.NextDouble() * (hi - lo + 1)) };
                if (random.Next(3) == 0) v[i] = Math.Clamp(random.Next(-300, 300), lo, hi);
                if (finite) v[i] = Math.Clamp(v[i], -10000, 10000);
            }
        }

        return Bench.Registry.Encode(type, v);
    }

    /// <summary>Every pair of the types: the CPU against the exact reference (each value rounded once into the target, held to
    /// its range; a block format the blocks the CPU's encoder makes of the values rounded to float32), the device the CPU's.</summary>
    private static bool Converts(IDevice<ulong, ulong>? ops, string device)
    {
        const int N = 1001;
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        int pairs = 0;
        foreach (var from in Bench.Registry.Names)
            foreach (var to in Bench.Registry.Names)
            {
                pairs++;
                var x = ConvertSource(from, N, 90 + pairs, finite: Bench.Registry.IsBlock(to));
                var cpu = Bench.Registry.Poisoned(to, N);
                Bench.Registry.With(from, new ConvertRun(null, to, x, cpu, N));
                if (ops is not null)
                {
                    var dev = Bench.Registry.Poisoned(to, N);
                    Bench.Registry.With(from, new ConvertRun(ops, to, x, dev, N));
                    if (!SameValues(to, $"convert {from}->{to}", where, dev, cpu, N)) ok = false;
                    continue;
                }

                var src = Exacts(from, x, N);
                if (Bench.Registry.IsBlock(to))
                {
                    // The block's encoder of the values rounded to nearest float32 (the format's definition; a 64-bit integer by
                    // the runtime's conversion, rounded once).
                    var f32 = from == "int64" ? MemoryMarshal.Cast<byte, long>(x.AsSpan()).ToArray().Select(l => (double)(float)l).ToArray()
                        : from == "uint64" ? MemoryMarshal.Cast<byte, ulong>(x.AsSpan()).ToArray().Select(l => (double)(float)l).ToArray()
                        : Bench.Registry.Values(from, x, N).Select(d => (double)(float)d).ToArray();
                    var want = Bench.Registry.Encode(to, f32);
                    if (!cpu.AsSpan().SequenceEqual(want.AsSpan(0, cpu.Length))) { Console.WriteLine($"check convert {from}->{to,-14} cpu    DIFFERS: the blocks are not the encoder's of the values as float32"); ok = false; }
                    continue;
                }

                var got = Exacts(to, cpu, N);
                for (int i = 0; i < N; i++)
                {
                    var want = Bench.Registry.IsFloat(to) ? RoundFloat(to, src[i]) : RoundInteger(to, src[i]);
                    if (got[i].Equals(want)) continue;
                    Console.WriteLine($"check convert {from}->{to,-14} cpu    DIFFERS at [{i}]: {src[i]} became {got[i]}, rounded once it is {want}");
                    ok = false;
                    break;
                }
            }

        if (ok) Console.WriteLine($"check convert                {where,-6} every pair of the {Bench.Registry.Names.Length} types ({pairs}), 1001 values with the special values and the targets' rounding points: {(ops is null ? "each rounded once into the target, held to its range, as the exact reference" : "the CPU's values")}");
        return ok;
    }

    private sealed class ConvertRun(IDevice<ulong, ulong>? ops, string to, byte[] x, byte[] output, int n) : Bench.ITypeVisitor<bool>
    {
        public bool Visit<TF>() where TF : struct, IAdad<TF, float> => Bench.Registry.With(to, new Into<TF>(ops, x, output, n));

        private sealed class Into<TF>(IDevice<ulong, ulong>? ops, byte[] x, byte[] output, int n) : Bench.ITypeVisitor<bool> where TF : struct, IAdad<TF, float>
        {
            public bool Visit<TT>() where TT : struct, IAdad<TT, float>
            {
                if (ops is null) { CpuDevice.Device.Convert<TF, TT>(x, output, n); return true; }
                ulong dx = ops.Rent(x.Length + 16), dout = ops.Rent(output.Length + 16);
                try
                {
                    ops.Upload(x, dx); ops.Upload(output, dout);
                    ops.Convert<TF, TT>(dx, dout, n);
                    ops.Synchronize();
                    ops.Download(dout, output);
                    return true;
                }
                finally { ops.Return(dx); ops.Return(dout); }
            }
        }
    }

    // ---------------------------------------------------------------- a float formula over a row vector

    /// <summary>Every float type at 37 × 1001: MapRowFloat(add_gelu, x, v) the values MapFloat(add_gelu) gives with v repeated
    /// on every row, on the same device (the formula itself is the map check's).</summary>
    private static bool RowFloats(IDevice<ulong, ulong>? ops, string device)
    {
        const int R = 37, C = 1001;
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        var op = Ops.Find("add_gelu");
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            var x = Bench.Registry.Make(type, R * C, 85, -3, 3);
            var v = Bench.Registry.Make(type, C, 86, -1, 1);
            var vv = Bench.Registry.Values(type, v, C);
            var repeated = Bench.Registry.Encode(type, Enumerable.Range(0, R * C).Select(i => vv[i % C]).ToArray());
            var row = Bench.Registry.Poisoned(type, R * C);
            var whole = Bench.Registry.Poisoned(type, R * C);
            Bench.Registry.WithFloat(type, new RowFloatRun(ops, op, x, v, repeated, row, whole, R, C));
            // A block format: v repeated and encoded again is not v (its blocks fall elsewhere), and the CPU stores a row at a
            // time (a block two rows share encoded again with its kept values): against the reference, within the block's steps.
            if (Bench.Registry.IsBlock(type))
            {
                var xv = Bench.Registry.Values(type, x, R * C);
                ok &= Close(type, "maprow add_gelu", where, row, [.. Enumerable.Range(0, R * C).Select(i => op.Reference(xv[i], vv[i % C], 0, 0, 0))], R * C);
            }
            else if (!SameValues(type, "maprow add_gelu", where, row, whole, R * C)) ok = false;
        }

        if (ok) Console.WriteLine($"check maprow float           {where,-6} every float type at 37 × 1001: add_gelu over a row vector as over the vector repeated");
        return ok;
    }

    private sealed class RowFloatRun(IDevice<ulong, ulong>? ops, MapOp op, byte[] x, byte[] v, byte[] repeated, byte[] row, byte[] whole, int rows, int cols) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            int n = rows * cols;
            if (ops is null)
            {
                CpuDevice.Device.MapRowFloat<TN>(op, x, v, row, rows, cols);
                CpuDevice.Device.MapFloat<TN>(op, x, repeated, null, whole, n, 0, 0);
                return true;
            }

            ulong dx = ops.Rent(x.Length + 16), dv = ops.Rent(v.Length + 16), dr = ops.Rent(repeated.Length + 16), d1 = ops.Rent(row.Length + 16), d2 = ops.Rent(whole.Length + 16);
            try
            {
                ops.Upload(x, dx);
                ops.Upload(v, dv);
                ops.Upload(repeated, dr);
                ops.Upload(row, d1); ops.Upload(whole, d2);
                ops.MapRowFloat<TN>(op, dx, dv, d1, rows, cols);
                ops.MapFloat<TN>(op, dx, dr, 0, d2, n, 0, 0);
                ops.Synchronize();
                ops.Download(d1, row);
                ops.Download(d2, whole);
                return true;
            }
            finally { ops.Return(dx); ops.Return(dv); ops.Return(dr); ops.Return(d1); ops.Return(d2); }
        }
    }
}
