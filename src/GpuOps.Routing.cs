// The device's routing, sampling and losses (Routing.cs).
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>top-k of each row: values as the type (rows × k), indices s32; returns the lane's name.</summary>
    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = f.Natural;
        if (f.IsBlock)
        {
            // A block format's rows decoded into float32 storage, the top-k there, the values encoded by blocks.
            var r = new List<ulong>();
            try
            {
                ulong lx = Lanes(f, Lane.F32, x, (long)rows * cols, r), lv = gpu.Rent((long)rows * k * 4);
                r.Add(lv);
                TopK<Float32>(lx, rows, cols, k, lv, indices);
                Back(f, Lane.F32, lv, values, (long)rows * k);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var q in r) gpu.Return(q); }
        }

        if (k > 256) throw new NotSupportedException($"top-k: k = {k} exceeds the kernel's passes (256); a partial sort is the step");
        var fn = gpu.Cached($"topk:{f.Name}", (out string e) => Ptx.TopKRows(gpu, f, lane, out e));
        if (f.Bits < 8) Gpu.Clear(values, f.Words((long)rows * k) * 4);           // sub-byte values are OR-ed into their words
        Gpu.Launch(fn, (uint)rows, 1, 256, 1, 0, x, values, indices, Gpu.U(cols), Gpu.U(k));
        return lane.Name;
    }

    /// <summary>out[r] = x[r][index[r]] as the type.</summary>
    public string Pick<TN>(ulong x, int rows, int cols, ulong index, ulong output) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            var r = new List<ulong>();
            try
            {
                ulong lx = Lanes(f, Lane.F32, x, (long)rows * cols, r), lo = gpu.Rent(rows * 4L);
                r.Add(lo);
                Pick<Float32>(lx, rows, cols, index, lo);
                Back(f, Lane.F32, lo, output, rows);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var q in r) gpu.Return(q); }
        }

        var fn = gpu.Cached($"pick:{f.Name}", (out string e) => Ptx.PickRows(gpu, f, f.Natural, out e));
        if (f.Bits < 8) Gpu.Clear(output, f.Words(rows) * 4);
        Gpu.Launch(fn, Grid(rows, 256), 1, 256, 1, 0, x, index, output, Gpu.U(rows), Gpu.U(cols));
        return f.Natural.Name;
    }

    /// <summary>The sampled column of each row from weights x and a uniform draw u per row (as the type); in the float lane.</summary>
    public string Sample<TN>(ulong x, int rows, int cols, ulong u, ulong picks) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var fn = gpu.Cached($"sample:{f.Name}", (out string e) => Ptx.SampleRows(gpu, f, lane, out e));
        Gpu.Launch(fn, (uint)rows, 1, 256, 1, 0, x, u, picks, Gpu.U(cols));
        return lane.Name;
    }

    /// <summary>counts[b], offsets[b + 1] and the sources' order by bucket, from n indices into `buckets` (the scatter-add's ordering kernels).</summary>
    public void Route(ulong index, int n, int buckets, ulong counts, ulong offsets, ulong order)
    {
        Gpu.Clear(counts, buckets * 4L);
        var count = gpu.Cached("scatter_count", (out string e) => Ptx.ScatterCount(gpu, out e));
        var scan = gpu.Cached("scatter_scan", (out string e) => Ptx.ScatterScan(gpu, out e));
        var place = gpu.Cached("scatter_place", (out string e) => Ptx.ScatterPlace(gpu, out e));
        Gpu.Launch(count, Grid(n, 256), 1, 256, 1, 0, index, counts, Gpu.U(n));
        Gpu.Launch(scan, 1, 1, 32, 1, 0, counts, offsets, Gpu.U(buckets));
        Gpu.Launch(place, Grid(n, 256), 1, 256, 1, 0, index, offsets, order, Gpu.U(n));
    }
}
