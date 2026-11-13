// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The scaled weight's quantization (Products.cs ScaledWeight; IProducts.Quantize) on every device, every float type of the
// values and scales but the block formats: codes of int8 and int4 (with zero points and without), uint8 and uint2 (with
// zero points), float8 e4m3 and float16, a scale for the tensor, a row and groups of a row (a row of zeros among them, its
// scales 0). Each scale within the type's rounding of the float64 reference's; each zero point the nearest integer to
// min − lo / scale by the stored scale, as the type stores it; each code the nearest code of x / scale (+ the zero point) by
// the stored scale and zero point, either way at a tie within the lane's rounding, held to the code type's range.
namespace Qiyas;

internal static partial class Checks
{
    public static bool QuantizeCheck(IDevice<ulong, ulong>? ops, string device = "device")
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        const int n = 37, k = 67;
        var layouts = new (string Name, ScaledWeight W)[] { ("a scale for the tensor", ScaledWeight.PerTensor(n, k)), ("a scale a row", ScaledWeight.PerRow(n, k)), ("groups of 32", ScaledWeight.Grouped(n, k, 32)) };
        var codeCases = new (string Type, bool Zeros)[] { ("int8", false), ("int8", true), ("int4", false), ("int4", true), ("uint8", true), ("uint2", true), ("float8_e4m3", false), ("float16", false) };
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || Bench.Registry.IsBlock(type)) continue;
            double laneEps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23);
            var values = Bench.Registry.Values(type, Bench.Registry.Make(type, n * k, 120, -2, 1.25), n * k);
            for (int c = 0; c < k; c++) values[3 * k + c] = 0;                              // a row of zeros
            var x = Bench.Registry.Encode(type, values);
            foreach (var (layout, w) in layouts)
            foreach (var (codeType, zeros) in codeCases)
            {
                string what = $"quantize {codeType}{(zeros ? " with zeros" : "")}, {layout}";
                var (min, max) = Bench.Registry.With(codeType, new CodeRange());
                byte[] codes = Bench.Registry.Poisoned(codeType, w.Values), scales = Bench.Registry.Poisoned(type, w.Scales);
                byte[]? zp = zeros ? Bench.Registry.Poisoned(type, w.Scales) : null;
                Bench.Registry.WithFloat(type, new QuantizeRun(ops, codeType, w, x, codes, scales, zp));
                var (wantScales, lows) = w.QuantizeReference(values, min, max, zeros);
                double bad = Excess(type, what + " scales", scales, wantScales, i => Math.Abs(wantScales[i]), 2);
                if (bad > 0) { Console.WriteLine($"check {what,-50} {where,-6} {type,-12} scales DIFFER by {bad:0.##}"); ok = false; continue; }

                double[] s = Bench.Registry.Values(type, scales, (int)w.Scales), z = zp is null ? new double[w.Scales] : Bench.Registry.Values(type, zp, (int)w.Scales);
                double TypeRound(double v) => Bench.Registry.Values(type, Bench.Registry.Encode(type, [v]), 1)[0];
                bool Tie(double q) => Math.Abs(Math.Abs(q - Math.Floor(q)) - 0.5) < 64 * laneEps * Math.Max(1, Math.Abs(q));
                string? wrong = null;
                for (int si = 0; si < w.Scales && wrong is null && zeros; si++)
                {
                    double zq = s[si] > 0 ? min - lows[si] / s[si] : 0, want = Math.Clamp(Math.Round(zq, MidpointRounding.ToEven), min, max);
                    bool near = z[si] == TypeRound(want) || (Tie(zq) && (z[si] == TypeRound(Math.Clamp(Math.Floor(zq), min, max)) || z[si] == TypeRound(Math.Clamp(Math.Ceiling(zq), min, max))));
                    if (!near) wrong = $"zero point {si}: {z[si]:R}, not {want:R} (min − lo / scale = {zq:R})";
                }

                var got = Bench.Registry.Values(codeType, codes, (int)w.Values);
                bool floatCodes = codeType.StartsWith("float", StringComparison.Ordinal);
                for (int i = 0; i < w.Values && wrong is null; i++)
                {
                    int si = w.ScaleAt(i / k, i % k);
                    double q = s[si] > 0 ? values[i] / s[si] : 0, code = got[i];
                    if (code < min || code > max || double.IsNaN(code)) { wrong = $"code {i}: {code:R} outside [{min}, {max}]"; break; }
                    if (floatCodes)
                    {
                        // The nearest code of q held to the range: no farther from q than its own rounding, within the lane's.
                        double held = Math.Clamp(q, min, max), nearest = Bench.Registry.Values(codeType, Bench.Registry.Encode(codeType, [held]), 1)[0];
                        if (Math.Abs(code - held) > Math.Abs(nearest - held) + 4 * laneEps * Math.Abs(held)) wrong = $"code {i}: {code:R}, not {nearest:R} (x / scale = {q:R})";
                    }
                    else
                    {
                        double target = q + z[si], want = Math.Clamp(Math.Round(q, MidpointRounding.ToEven) + z[si], min, max);
                        bool near = code == want || (Tie(q) && Math.Abs(code - Math.Clamp(target, min, max)) <= 0.5 + 64 * laneEps * Math.Max(1, Math.Abs(q)));
                        if (!near) wrong = $"code {i}: {code:R}, not {want:R} (x / scale + zero = {target:R})";
                    }
                }

                if (wrong is not null) { Console.WriteLine($"check {what,-50} {where,-6} {type,-12} DIFFERS: {wrong}"); ok = false; }
            }
        }

        if (ok) Console.WriteLine($"check quantize               {where,-6} every float type but the block formats: codes of int8, int4 (zero points and none), uint8, uint2 (zero points), float8 and float16 for the tensor, a row and groups of a row: scales within the type's rounding, zero points and codes the nearest by the stored scales");
        return ok;
    }

    private sealed class CodeRange : Bench.ITypeVisitor<(double Min, double Max)>
    {
        public (double Min, double Max) Visit<TW>() where TW : struct, IAdad<TW, float> => (TW.Min, TW.Max);
    }

    private sealed record QuantizeRun(IDevice<ulong, ulong>? Ops, string CodesType, ScaledWeight W, byte[] X, byte[] Codes, byte[] Scales, byte[]? Zeros) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float> => Bench.Registry.With(CodesType, new CodesOf<TN>(this));

        private sealed class CodesOf<TN>(QuantizeRun r) : Bench.ITypeVisitor<string> where TN : struct, IFloatAdad<TN>
        {
            public string Visit<TW>() where TW : struct, IAdad<TW, float>
            {
                if (r.Ops is null) return CpuDevice.Device.Quantize<TN, TW>(r.W, r.X, r.Codes, r.Scales, r.Zeros);
                using var st = new Staged(r.Ops);
                string lane = r.Ops.Quantize<TN, TW>(r.W, st.In(r.X), st.Out(r.Codes), st.Out(r.Scales), st.Out(r.Zeros));
                st.Collect();
                return lane;
            }
        }
    }
}
