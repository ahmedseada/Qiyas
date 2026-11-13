// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A step across several tensors under one clipping (IOptimizers.OptimizerMany) on every device, every float type (the
// block formats among them): five tensors (1001, 37, none, 4099 and 258 values), every step, with no clipping (the sum
// of the gradients' squares asked for), clipped to a norm below the gradients' (every gradient scaled) and to one above
// it (the factor 1). The sum within the lane's rounding of the float64 sum over all the tensors; each tensor's p, m and
// v against the float64 reference of the step with its gradients times the factor, as the one-tensor step's are held.
namespace Qiyas;

internal static partial class Checks
{
    private static bool OptimizersMany(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        int size = type == "float64" ? 8 : 4;
        double laneEps = size == 8 ? Math.Pow(2, -52) : Math.Pow(2, -23);
        int[] counts = [1001, 37, 0, 4099, 258];
        foreach (var name in OptimizerArgs.Names)
        {
            foreach (var (maxNorm, withSum) in new[] { (0.0, true), (5.0, false), (5.0, true), (1e6, false) })
            {
                var a = new OptimizerArgs(name, 0.01, 0.9, 0.999, 1e-8, 0.01, 3) { MaxNorm = maxNorm };
                string what = $"optimizer_many {name} norm {maxNorm}";
                int tensors = counts.Length;
                var p = new byte[tensors][]; var g = new byte[tensors][]; var m = new byte[]?[tensors]; var v = new byte[]?[tensors];
                var wantP = new double[tensors][]; var gv = new double[tensors][]; var wantM = new double[]?[tensors]; var wantV = new double[]?[tensors];
                var random = new Random(75);
                for (int t = 0; t < tensors; t++)
                {
                    int n = counts[t];
                    p[t] = Bench.Registry.Make(type, n, 71 + t, -1, 1);
                    g[t] = Bench.Registry.Make(type, n, 81 + t, -1, 1);
                    wantP[t] = Bench.Registry.Values(type, p[t], n);
                    gv[t] = Bench.Registry.Values(type, g[t], n);
                    if (a.Moments > 0) { wantM[t] = new double[n]; for (int i = 0; i < n; i++) wantM[t]![i] = random.NextDouble() - 0.5; m[t] = Lanes(size, wantM[t]!); }
                    if (a.Moments > 1) { wantV[t] = new double[n]; for (int i = 0; i < n; i++) wantV[t]![i] = random.NextDouble(); v[t] = Lanes(size, wantV[t]!); }
                }

                double wantSum = a.ReferenceMany(wantP, gv, wantM, wantV);
                var sum = withSum ? Bench.Registry.Poisoned(size) : null;
                Bench.Registry.WithFloat(type, new OptimizerManyRun(ops, a, p, g, m, v, counts, sum));
                if (sum is not null)
                {
                    double got = size == 8 ? BitConverter.ToDouble(sum) : BitConverter.ToSingle(sum);
                    if (!(Math.Abs(got - wantSum) <= 64 * laneEps * wantSum))
                    {
                        Console.WriteLine($"check {what,-40} {where,-6} {type,-12} the sum of squares DIFFERS: {got:R}, not {wantSum:R}");
                        ok = false;
                    }
                }

                for (int t = 0; t < tensors; t++)
                {
                    int n = counts[t];
                    if (n == 0) continue;
                    ok &= Close(type, $"{what} p[{t}]", where, p[t], wantP[t], n, 16);
                    if (m[t] is not null) ok &= LanesClose(type, $"{what} m[{t}]", where, m[t]!, wantM[t]!, size);
                    if (v[t] is not null) ok &= LanesClose(type, $"{what} v[{t}]", where, v[t]!, wantV[t]!, size);
                }
            }
        }

        return ok;
    }

    private sealed class OptimizerManyRun(IDevice<ulong, ulong>? ops, OptimizerArgs a, byte[][] p, byte[][] g, byte[]?[] m, byte[]?[] v, int[] counts, byte[]? sum) : Bench.IFloatTypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null)
            {
                var host = new OptimizerTensor<byte[]>[p.Length];
                for (int t = 0; t < p.Length; t++) host[t] = new(p[t], g[t], m[t], v[t], counts[t]);
                CpuDevice.Device.OptimizerMany<TN>(a, host, sum);
                return true;
            }

            using var st = new Staged(ops);
            var device = new OptimizerTensor<ulong>[p.Length];
            for (int t = 0; t < p.Length; t++) device[t] = new(st.Out(p[t]), st.In(g[t]), st.Out(m[t]), st.Out(v[t]), counts[t]);
            ops.OptimizerMany<TN>(a, device, st.Out(sum));
            st.Collect();
            return true;
        }
    }
}
