// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The normalizations past the rows, on every device against their float64 references (RowOps): group normalization over
// batch, group, instance and a layer's layouts (GroupShape), with and without its gain and bias and by given statistics, and
// its backward (dx, dgamma, dbeta); the rotary embedding by tables over half and interleaved pairs, a partial rotation and
// the inverse, alone and after an RMS normalization by an offset gain. Every float type, the block formats among them.
namespace Qiyas;

internal static partial class Checks
{
    public static bool Norms(IDevice<ulong, ulong>? ops, string device = "device")
    {
        bool ok = GroupNorms(ops, device);
        ok &= Ropes(ops, device);
        return ok;
    }

    // ---------------------------------------------------------------- group normalization

    private static readonly (string Name, GroupShape Shape)[] GroupShapes =
    [
        ("batch 4x6x50", GroupShape.Batch(4, 6, 50)),
        ("group 2x12x31/4", GroupShape.Group(2, 12, 31, 4)),
        ("instance 2x5x33", GroupShape.Group(2, 5, 33, 5)),
        ("layer 3x4x7", GroupShape.Group(3, 4, 7, 1)),
        ("group 1x64x1/32", GroupShape.Group(1, 64, 1, 32)),
        ("batch 8x3x4099", GroupShape.Batch(8, 3, 4099)),
    ];

    /// <summary>Every float type over the shapes, affine or not, computed or given statistics: y within the type's step plus 8 lane
    /// ulps of each group's error scale, the statistics and the parameters' gradients within the lane's rounding of their sums.</summary>
    private static bool GroupNorms(IDevice<ulong, ulong>? ops, string device)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        const double Eps = 1e-5;
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            string lane = type == "float64" ? "f64" : "f32";
            double laneEps = lane == "f64" ? Math.Pow(2, -52) : Math.Pow(2, -23);
            foreach (var (name, s) in GroupShapes)
            {
                int n = s.Count, m = s.PerGroup;
                var x = Bench.Registry.Make(type, n, 71, -4, 4);
                var dy = Bench.Registry.Make(type, n, 74, -1, 1);
                var gamma = Bench.Registry.Make(type, s.Channels, 72, 0.5, 2);
                var beta = Bench.Registry.Make(type, s.Channels, 73, -1, 1);
                var xv = Bench.Registry.Values(type, x, n);
                var dyv = Bench.Registry.Values(type, dy, n);
                var gv = Bench.Registry.Values(type, gamma, s.Channels);
                var bv = Bench.Registry.Values(type, beta, s.Channels);
                double xMax = xv.Max(Math.Abs);
                // Running statistics for the given case, as the lane stores them.
                var runMean = LaneRound(lane, Enumerable.Range(0, s.Groups).Select(g => ((g * 7) % 11 - 5) / 5.0).ToArray());
                var runVar = LaneRound(lane, Enumerable.Range(0, s.Groups).Select(g => 0.5 + g % 5 * 0.4).ToArray());
                foreach (bool given in new[] { false, true })
                    foreach (bool affine in new[] { true, false })
                    {
                        string what = $"group_norm {name}{(affine ? "" : " plain")}{(given ? " given" : "")}";
                        var mean = given ? LaneBytes(lane, runMean) : new byte[s.Groups * (lane == "f64" ? 8 : 4)];
                        var variance = given ? LaneBytes(lane, runVar) : new byte[mean.Length];
                        double[] rm = given ? (double[])runMean.Clone() : new double[s.Groups], rv = given ? (double[])runVar.Clone() : new double[s.Groups];
                        var ry = new double[n];
                        var scale = RowOps.GroupNormReference(xv, affine ? gv : null, affine ? bv : null, s, Eps, given, rm, rv, ry);
                        var y = new byte[x.Length];
                        Bench.Registry.WithFloat(type, new GroupNormRun(ops, x, affine ? gamma : null, affine ? beta : null, mean, variance, y, s, Eps, given));
                        double worst = Excess(type, what, y, ry, i => scale[s.GroupOf(i)], 8);
                        if (!given)
                        {
                            // A group's sums of m terms: the lane's rounding of each, every term up to max |x| (and its square).
                            var gm = Bench.LaneValues.Decode(lane, mean, s.Groups);
                            var gvar = Bench.LaneValues.Decode(lane, variance, s.Groups);
                            for (int g = 0; g < s.Groups; g++)
                            {
                                double tm = laneEps * (16 + m) * xMax, tv = laneEps * (16 + m) * 4 * xMax * xMax;
                                worst = Math.Max(worst, Over(gm[g], rm[g], tm));
                                worst = Math.Max(worst, Over(gvar[g], rv[g], tv));
                            }
                        }

                        if (worst > 0) { Console.WriteLine($"check {what,-40} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }

                        // The backward from dy: dx, dgamma, dbeta.
                        what = $"group_norm_bwd {name}{(affine ? "" : " plain")}{(given ? " given" : "")}";
                        double[] bm = given ? (double[])runMean.Clone() : new double[s.Groups], bvr = given ? (double[])runVar.Clone() : new double[s.Groups];
                        var rdx = new double[n];
                        var rdg = new double[s.Channels];
                        var rdb = new double[s.Channels];
                        var (sdx, sp) = RowOps.GroupNormBwdReference(xv, dyv, affine ? gv : null, s, Eps, given, bm, bvr, rdx, rdg, rdb);
                        var dx = new byte[x.Length];
                        var dg = new byte[s.Channels * (lane == "f64" ? 8 : 4)];
                        var db = new byte[dg.Length];
                        Bench.Registry.WithFloat(type, new GroupNormBwdRun(ops, x, dy, affine ? gamma : null, given ? mean : null, given ? variance : null, dx, dg, db, s, Eps, given));
                        worst = Excess(type, what, dx, rdx, i => sdx[s.GroupOf(i)], 8);
                        var gdg = Bench.LaneValues.Decode(lane, dg, s.Channels);
                        var gdb = Bench.LaneValues.Decode(lane, db, s.Channels);
                        for (int c = 0; c < s.Channels; c++)
                        {
                            // The group statistics' rounding (m terms) reaches every x̂ the channel sums.
                            double tol = laneEps * sp[c] * (16 + m);
                            worst = Math.Max(worst, Over(gdg[c], rdg[c], tol));
                            worst = Math.Max(worst, Over(gdb[c], rdb[c], tol));
                        }

                        if (worst > 0) { Console.WriteLine($"check {what,-40} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
                    }
            }
        }

        if (ok) Console.WriteLine($"check group norms            {where,-6} every float type: batch, group, instance and layer layouts, affine or plain, computed or given statistics, and the backward: within the lane's rounding of the references");
        return ok;
    }

    // How far a lane value is past its tolerance (in tolerances; 0 within).
    private static double Over(double got, double want, double tol)
    {
        double err = Math.Abs(got - want);
        if (double.IsNaN(err)) return double.PositiveInfinity;
        return err > tol ? err / Math.Max(tol, double.Epsilon) : 0;
    }

    private static double[] LaneRound(string lane, double[] v) => lane == "f64" ? v : v.Select(t => (double)(float)t).ToArray();

    private static byte[] LaneBytes(string lane, double[] v)
    {
        if (lane == "f64") return System.Runtime.InteropServices.MemoryMarshal.AsBytes(v.AsSpan()).ToArray();
        var f = v.Select(t => (float)t).ToArray();
        return System.Runtime.InteropServices.MemoryMarshal.AsBytes(f.AsSpan()).ToArray();
    }

    // The worst excess of stored values over their float64 reference in steps of the type past the tolerance (0 when all
    // within): a value's tolerance is the type's step at it (a block format's: two steps of its block's largest) plus ulps
    // lane ulps of its error scale.
    private static double Excess(string type, string what, byte[] got, double[] reference, Func<int, double> scale, int ulps)
    {
        int n = reference.Length;
        var expected = Bench.Registry.Values(type, Bench.Registry.Encode(type, reference), n);
        var values = Bench.Registry.Values(type, got, n);
        int mant = Bench.Registry.Mantissa(type);
        double laneEps = type == "float64" ? Math.Pow(2, -52) : Math.Pow(2, -23), minStep = Bench.Registry.MinStep(type);
        int blockValues = type switch { "q8_0" or "q4_0" => 32, "q4_k" or "q6_k" => 256, _ => 0 };
        double[]? blockMax = null;
        if (blockValues > 0)
        {
            blockMax = new double[(n + blockValues - 1) / blockValues];
            for (int i = 0; i < n; i++) blockMax[i / blockValues] = Math.Max(blockMax[i / blockValues], Math.Abs(reference[i]));
        }

        double worst = 0;
        for (int i = 0; i < n; i++)
        {
            double want = expected[i], v = got.Length == 0 ? 0 : values[i];
            double step = blockMax is not null
                ? 2 * Math.Max(blockMax[i / blockValues], minStep) * Math.Pow(2, -mant)
                : Math.Max(minStep, reference[i] == 0 ? Math.Pow(2, -mant) : Math.Pow(2, Math.Floor(Math.Log2(Math.Abs(reference[i]))) - mant));
            double err = Math.Abs(v - want), tol = step + ulps * laneEps * scale(i);
            if (double.IsNaN(err) || err > tol)
            {
                if (Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1" && worst == 0) Console.WriteLine($"  {what} {type} [{i}] got {v:R} want {want:R} ref {reference[i]:R} tol {tol:E2}");
                worst = Math.Max(worst, double.IsNaN(err) ? double.PositiveInfinity : err / step);
            }
        }

        return worst;
    }

    // Buffers on a device for one call: inputs uploaded, outputs downloaded back into their host arrays.
    private sealed class Staged(IDevice<ulong, ulong> ops) : IDisposable
    {
        private readonly List<(ulong Ptr, byte[]? Host)> _buffers = [];
        private readonly List<ulong> _indices = [];
        private readonly List<(ulong Ptr, int[] Host)> _indicesBack = [];

        public ulong In(byte[]? a) => Put(a, false);

        /// <summary>Uploaded, and downloaded back (an output that may also be read).</summary>
        public ulong Out(byte[]? a) => Put(a, true);

        public ulong Indices(int[] a)
        {
            ulong d = ops.RentIndices(a.Length);
            ops.UploadIndices(a, d);
            _indices.Add(d);
            return d;
        }

        /// <summary>Indices uploaded, and downloaded back (an output).</summary>
        public ulong IndicesOut(int[] a)
        {
            ulong d = Indices(a);
            _indicesBack.Add((d, a));
            return d;
        }

        private ulong Put(byte[]? a, bool back)
        {
            if (a is null) return 0;
            ulong d = ops.Rent(a.Length + 16);
            ops.Upload(a, d);
            _buffers.Add((d, back ? a : null));
            return d;
        }

        public void Collect()
        {
            ops.Synchronize();
            foreach (var (p, h) in _buffers) if (h is not null) ops.Download(p, h);
            foreach (var (p, h) in _indicesBack) ops.DownloadIndices(p, h);
        }

        public void Dispose()
        {
            foreach (var (p, _) in _buffers) ops.Return(p);
            foreach (var p in _indices) ops.ReturnIndices(p);
        }
    }

    private sealed class GroupNormRun(IDevice<ulong, ulong>? ops, byte[] x, byte[]? gamma, byte[]? beta, byte[] mean, byte[] variance, byte[] y, GroupShape s, double eps, bool given)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.GroupNorm<TN>(x, gamma, beta, mean, variance, y, s, eps, given);
            using var st = new Staged(ops);
            var lane = ops.GroupNorm<TN>(st.In(x), st.In(gamma), st.In(beta), st.Out(mean), st.Out(variance), st.Out(y), s, eps, given);
            st.Collect();
            return lane;
        }
    }

    private sealed class GroupNormBwdRun(IDevice<ulong, ulong>? ops, byte[] x, byte[] dy, byte[]? gamma, byte[]? mean, byte[]? variance, byte[] dx, byte[] dgamma, byte[] dbeta, GroupShape s, double eps, bool given)
        : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.GroupNormBwd<TN>(x, dy, gamma, mean, variance, dx, dgamma, dbeta, s, eps, given);
            using var st = new Staged(ops);
            var lane = ops.GroupNormBwd<TN>(st.In(x), st.In(dy), st.In(gamma), st.In(mean), st.In(variance), st.Out(dx), st.Out(dgamma), st.Out(dbeta), s, eps, given);
            st.Collect();
            return lane;
        }
    }

    // ---------------------------------------------------------------- the rotary embedding by tables

    /// <summary>Every float type over half and interleaved pairs, a partial rotation, an odd row, the inverse, and the RMS
    /// normalization by an offset gain first: within the type's step plus 8 lane ulps of each row's error scale.</summary>
    private static bool Ropes(IDevice<ulong, ulong>? ops, string device)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        const int MaxPos = 64;
        var cases = new (int Batch, int Steps, int Heads, int Cols, int Half, bool Interleaved, bool Inverse, bool Gain)[]
        {
            (2, 5, 3, 64, 32, false, false, false),
            (2, 5, 3, 64, 16, true, false, false),
            (1, 7, 2, 17, 8, false, true, false),
            (2, 3, 4, 128, 32, false, false, true),
            (1, 4, 3, 40, 20, true, true, true),
        };

        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type)) continue;
            string lane = type == "float64" ? "f64" : "f32";
            foreach (var k in cases)
            {
                int rows = k.Batch * k.Steps * k.Heads;
                var a = new RopeArgs(rows, k.Cols, k.Heads, k.Steps, k.Half, k.Interleaved, k.Inverse, 1e-6, k.Gain ? 1 : 0);
                var cos = new double[MaxPos * k.Half];
                var sin = new double[MaxPos * k.Half];
                for (int p = 0; p < MaxPos; p++)
                    for (int i = 0; i < k.Half; i++)
                    {
                        double theta = p * Math.Pow(10000, -(double)i / k.Half);
                        cos[p * k.Half + i] = Math.Cos(theta);
                        sin[p * k.Half + i] = Math.Sin(theta);
                    }

                cos = LaneRound(lane, cos);
                sin = LaneRound(lane, sin);
                var positions = Enumerable.Range(0, k.Steps).Select(t => t * 7 % MaxPos).ToArray();
                var x = Bench.Registry.Make(type, rows * k.Cols, 81, -4, 4);
                var gain = k.Gain ? Bench.Registry.Make(type, k.Cols, 82, 0.5, 2) : null;
                var ry = new double[rows * k.Cols];
                var scale = RowOps.RopeReference(Bench.Registry.Values(type, x, rows * k.Cols), gain is null ? null : Bench.Registry.Values(type, gain, k.Cols), cos, sin, positions, a, ry);
                var y = new byte[x.Length];
                Bench.Registry.WithFloat(type, new RopeRun(ops, a, x, gain, LaneBytes(lane, cos), LaneBytes(lane, sin), positions, y));
                string what = $"rope {rows}x{k.Cols} half {k.Half}{(k.Interleaved ? " interleaved" : "")}{(k.Inverse ? " inverse" : "")}{(k.Gain ? " rms" : "")}";
                double worst = Excess(type, what, y, ry, i => scale[i / k.Cols], 8);
                if (worst > 0) { Console.WriteLine($"check {what,-40} {where,-6} {type,-12} DIFFERS by {worst:0.##}"); ok = false; }
            }
        }

        if (ok) Console.WriteLine($"check ropes                  {where,-6} every float type: half and interleaved pairs, partial, odd rows, inverse, and after an RMS normalization: within the lane's rounding of the references");
        return ok;
    }

    private sealed class RopeRun(IDevice<ulong, ulong>? ops, RopeArgs a, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] y) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.Rope<TN>(a, x, gain, cos, sin, positions, y);
            using var st = new Staged(ops);
            var lane = ops.Rope<TN>(a, st.In(x), st.In(gain), st.In(cos), st.In(sin), st.Indices(positions), st.Out(y));
            st.Collect();
            return lane;
        }
    }
}
