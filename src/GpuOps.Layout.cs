// The device's layout operations (Layout.cs): the lane by the operation (the type's own for moves, the float lane for
// bilinear interpolation, the sum lane by the bounds for scatter-add and col2im), scatter-add's ordering built on the
// device (counts, offsets, ordered sources), then one kernel with a thread per output word.
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>A layout operation (LayoutArgs.Names) from x (and the index array) into output, all on the device. Returns the lane's name.</summary>
    public string Layout<TN>(LayoutArgs k, ulong x, ulong idx, ulong output, ulong x2 = 0) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                ulong lx = Lanes(f, Lane.F32, x, k.InputCount, r), lx2 = x2 == 0 ? 0 : Lanes(f, Lane.F32, x2, Bench.LayoutCases.SecondCount(k), r), lo = gpu.Rent(k.Count * 4L);
                r.Add(lo);
                Layout<Float32>(k, lx, idx, lo, lx2);
                Back(f, Lane.F32, lo, output, k.Count);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        var (rows, cols) = LayoutArgs.OutputRows(k);
        Lane lane;
        if (LayoutArgs.Sums(k.Name))
        {
            var (lo, hi) = Range<TN>(x, k.InputCount);
            var (lo2, hi2) = x2 != 0 && k.Name == "depthwise_conv" ? Range<TN>(x2, k.P[0] * k.P[3] * k.P[4]) : (1, 1);
            lane = SumLane(f, new Bounds(lo, hi, lo2, hi2, LayoutArgs.Terms(k.Name, k.P)));
            if (!f.IsFloat && lane.IsFloat) lane = f.Signed ? Lane.S32 : Lane.U32;   // an integer sum stays an integer (its store converts to the type)
        }
        else lane = LayoutArgs.FloatLane(k.Name) ? RowLane(f) : f.Natural;

        ulong aux = 0;
        try
        {
            if (k.Name == "scatter_add_rows")
            {
                int n = k.P[0], targets = k.P[2];
                ulong counts = gpu.Rent(targets * 4L);
                aux = gpu.Rent((targets + 1 + n) * 4L);
                try
                {
                    // The rows' offsets and the sources in order of their row (stable): the radix ordering (GpuOps.Radix.cs).
                    OrderByTarget(idx, n, targets, counts, aux, aux + (ulong)((targets + 1) * 4L));
                }
                finally
                {
                    gpu.Return(counts);
                }
            }

            var fn = gpu.Cached($"layout:{k.Name}:{f.Name}:{lane.Name}", (out string e) => Ptx.Layout(gpu, f, lane, k.Name, out e));
            var args = new ulong[8 + 12];
            args[0] = x; args[1] = x2; args[2] = idx; args[3] = aux; args[4] = output; args[5] = Gpu.U(k.Count); args[6] = Gpu.U(rows); args[7] = Gpu.U(cols);
            for (int i = 0; i < 12; i++) args[8 + i] = Gpu.U(i < k.P.Length ? k.P[i] : 0);
            Gpu.Launch(fn, Grid(f.Words(k.Count), 256), 1, 256, 1, 0, args);
        }
        finally
        {
            if (aux != 0) gpu.Return(aux);
        }

        return lane.Name;
    }
}
