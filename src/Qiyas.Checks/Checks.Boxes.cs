// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The box overlap losses (Boxes.cs) on every device against their float64 reference: IoU, GIoU, DIoU and CIoU over every
// float type (the block formats among them), pairs overlapping, apart, enclosing, equal and with tied corners (a tie's
// gradient shared in halves); the loss within the lane's rounding of its terms (Boxes.Reference's scale) plus the type's
// step, the backward's components the same with their own scales; and first the reference's gradient against its own loss
// by central differences (CIoU's α held at its value, as the gradient takes it; away from the ties).
namespace Qiyas;

internal static partial class Checks
{
    private static readonly BoxOverlap[] Overlaps = [BoxOverlap.IoU, BoxOverlap.GIoU, BoxOverlap.DIoU, BoxOverlap.CIoU];

    public static bool BoxLosses(IDevice<ulong, ulong>? ops, string device = "device", bool blocks = true)
    {
        bool ok = true;
        string where = ops is null ? "cpu" : device;
        const int count = 1001;
        var (pv, tv) = BoxPairs(count);
        if (ops is null) ok &= BoxGradients(pv, tv, count);
        foreach (var type in Bench.Registry.Names)
        {
            if (!Bench.Registry.IsFloat(type) || (!blocks && Bench.Registry.IsBlock(type))) continue;
            var p = Bench.Registry.Encode(type, pv);
            var t = Bench.Registry.Encode(type, tv);
            var dl = Bench.Registry.Make(type, count, 75, -1, 1);
            double[] ps = Bench.Registry.Values(type, p, 4 * count), ts = Bench.Registry.Values(type, t, 4 * count), ds = Bench.Registry.Values(type, dl, count);
            foreach (var overlap in Overlaps)
            {
                string name = Boxes.Name(overlap);
                var want = new double[count];
                var scale = new double[count];
                var wantGrad = new double[4 * count];
                var gradScale = new double[4 * count];
                for (int i = 0; i < count; i++)
                {
                    var g = wantGrad.AsSpan(4 * i, 4);
                    (want[i], scale[i]) = Boxes.Reference(overlap, ps.AsSpan(4 * i, 4), ts.AsSpan(4 * i, 4), Boxes.Eps, g, gradScale.AsSpan(4 * i, 4));
                    for (int k = 0; k < 4; k++) { g[k] *= ds[i]; gradScale[4 * i + k] *= Math.Abs(ds[i]); }
                }

                var loss = new byte[Bench.Registry.Bytes(type, count)];
                Bench.Registry.WithFloat(type, new BoxRun(ops, overlap, p, t, null, loss, count));
                double bad = Excess(type, $"box {name}", loss, want, i => scale[i], 16);
                if (bad > 0) { Console.WriteLine($"check box {name,-5} loss {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
                var grad = new byte[Bench.Registry.Bytes(type, 4 * count)];
                Bench.Registry.WithFloat(type, new BoxRun(ops, overlap, p, t, dl, grad, count));
                bad = Excess(type, $"box {name} bwd", grad, wantGrad, i => gradScale[i], 32);
                if (bad > 0) { Console.WriteLine($"check box {name,-5} grad {where,-6} {type,-12} DIFFERS by {bad:0.##}"); ok = false; }
            }
        }

        if (ok) Console.WriteLine($"check box losses             {where,-6} every float type{(blocks ? "" : " but the block formats")}: iou, giou, diou, ciou and their gradients over overlapping, apart, enclosing, equal and tied pairs, within the lane's rounding of their terms{(ops is null ? "; the reference's gradient by central differences" : "")}");
        return ok;
    }

    // Pairs of corner boxes within (−8, 8), every side at least 2 (a side stays positive at every float type's rounding):
    // overlapping (most), the target the prediction itself, apart, enclosing it, and sharing its top-left corner.
    private static (double[] P, double[] T) BoxPairs(int count)
    {
        var r = new Random(74);
        var p = new double[4 * count];
        var t = new double[4 * count];
        double U(double lo, double hi) => lo + (hi - lo) * r.NextDouble();
        for (int i = 0; i < count; i++)
        {
            int kind = i % 8;
            double x = U(-6, 1), y = U(-6, 1), w = U(2, 4), h = U(2, 4);
            if (kind == 2) { x = U(-6, -4); w = U(2, 2.5); }
            p[4 * i] = x; p[4 * i + 1] = y; p[4 * i + 2] = x + w; p[4 * i + 3] = y + h;
            double tx, ty, tw, th;
            switch (kind)
            {
                case 1: (tx, ty, tw, th) = (x, y, w, h); break;                                   // equal
                case 2: (tx, ty, tw, th) = (U(0.5, 1.5), y, U(2, 3), h); break;                   // apart
                case 3: (tx, ty, tw, th) = (x - 1, y - 1, w + 2, h + 2); break;                   // enclosing
                case 4: (tx, ty, tw, th) = (x, y, U(2, 4), U(2, 4)); break;                       // a shared corner
                default: (tx, ty, tw, th) = (x + U(-2, 2), y + U(-2, 2), U(2, 4), U(2, 4)); break;
            }

            t[4 * i] = tx; t[4 * i + 1] = ty; t[4 * i + 2] = tx + tw; t[4 * i + 3] = ty + th;
        }

        return (p, t);
    }

    // The reference's gradient against central differences of its loss (CIoU's α held at the pair's value), over the pairs
    // whose corners and overlap edges are apart (a tie's gradient is the halves', not a derivative).
    private static bool BoxGradients(double[] pv, double[] tv, int count)
    {
        bool ok = true;
        const double h = 1e-6;
        Span<double> g = stackalloc double[4], gs = stackalloc double[4], q = stackalloc double[4];
        foreach (var overlap in Overlaps)
        {
            double worst = 0;
            for (int i = 0; i < count; i++)
            {
                var p = pv.AsSpan(4 * i, 4);
                var t = tv.AsSpan(4 * i, 4);
                bool near = false;
                for (int k = 0; k < 4; k++) near |= Math.Abs(p[k] - t[k]) < 1e-3;
                near |= Math.Abs(Math.Min(p[2], t[2]) - Math.Max(p[0], t[0])) < 1e-3 || Math.Abs(Math.Min(p[3], t[3]) - Math.Max(p[1], t[1])) < 1e-3;
                if (near) continue;
                Boxes.Reference(overlap, p, t, Boxes.Eps, g, gs);
                double alpha = Boxes.CiouAlpha(p, t, Boxes.Eps);
                for (int k = 0; k < 4; k++)
                {
                    p.CopyTo(q);
                    q[k] = p[k] + h;
                    double up = Boxes.Reference(overlap, q, t, Boxes.Eps, fixedAlpha: alpha).Loss;
                    q[k] = p[k] - h;
                    double down = Boxes.Reference(overlap, q, t, Boxes.Eps, fixedAlpha: alpha).Loss;
                    worst = Math.Max(worst, Math.Abs((up - down) / (2 * h) - g[k]) / (1 + gs[k]));
                }
            }

            if (worst > 1e-7) { Console.WriteLine($"check box {Boxes.Name(overlap)} reference: the gradient differs from the loss's differences by {worst:R}"); ok = false; }
        }

        return ok;
    }

    private sealed class BoxRun(IDevice<ulong, ulong>? ops, BoxOverlap overlap, byte[] p, byte[] t, byte[]? dl, byte[] y, int count) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null)
                return dl is null ? CpuDevice.Device.BoxLoss<TN>(overlap, p, t, y, count, Boxes.Eps) : CpuDevice.Device.BoxLossBackward<TN>(overlap, p, t, dl, y, count, Boxes.Eps);
            using var st = new Staged(ops);
            var lane = dl is null
                ? ops.BoxLoss<TN>(overlap, st.In(p), st.In(t), st.Out(y), count, Boxes.Eps)
                : ops.BoxLossBackward<TN>(overlap, st.In(p), st.In(t), st.In(dl), st.Out(y), count, Boxes.Eps);
            st.Collect();
            return lane;
        }
    }
}
