// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's routing, sampling and losses (Routing.cs).
namespace Adad;

public sealed partial class GpuOps
{
    /// <summary>top-k of each row: values as the type (rows × k), indices s32; returns the lane's name.</summary>
    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = f.Natural;
        if (f.IsBlock) lane = Lane.F32;                                           // a block format: its rows decoded in place, in f32
        if (k > 256) throw new NotSupportedException($"top-k: k = {k} exceeds the kernel's passes (256); a partial sort is the step");
        var fn = gpu.Cached($"topk:{f.Name}", (out string e) => Ptx.TopKRows(gpu, f, lane, out e));
        if (f.Bits < 8) Gpu.Clear(values, f.Words((long)rows * k) * 4);           // sub-byte values are OR-ed into their words
        Gpu.Launch(fn, (uint)rows, 1, 256, 1, 0, x, values, indices, Gpu.U(cols), Gpu.U(k));
        if (f.IsBlock)
        {
            // The values: each row's chosen columns gathered where they are stored and their blocks encoded.
            var gather = gpu.Cached($"gather_rows_blocks:{f.Name}", (out string e) => Ptx.GatherRowsBlocks(gpu, f, out e));
            Gpu.Launch(gather, BlockGrid(f, (long)rows * k), 1, 256, 1, 0, x, indices, values, Gpu.U(rows), Gpu.U(cols), Gpu.U(k));
            return "f32 (blocks in place)";
        }

        return lane.Name;
    }

    /// <summary>out[r] = x[r][index[r]] as the type.</summary>
    public string Pick<TN>(ulong x, int rows, int cols, ulong index, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            // Each row's value gathered where it is stored, the blocks encoded.
            var gather = gpu.Cached($"gather_rows_blocks:{f.Name}", (out string e) => Ptx.GatherRowsBlocks(gpu, f, out e));
            Gpu.Launch(gather, BlockGrid(f, rows), 1, 256, 1, 0, x, index, output, Gpu.U(rows), Gpu.U(cols), Gpu.U(1));
            return "f32 (blocks in place)";
        }

        var fn = gpu.Cached($"pick:{f.Name}", (out string e) => Ptx.PickRows(gpu, f, f.Natural, out e));
        if (f.Bits < 8) Gpu.Clear(output, f.Words(rows) * 4);
        Gpu.Launch(fn, Grid(rows, 256), 1, 256, 1, 0, x, index, output, Gpu.U(rows), Gpu.U(cols));
        return f.Natural.Name;
    }

    /// <summary>loss[r] = log Σ exp(x[r]) − x[r][t[r]] as the type (t: s32 targets), in the type's float lane: a block per row.</summary>
    public string CrossEntropy<TN>(ulong x, int rows, int cols, ulong targets, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var fn = gpu.Cached($"cross_entropy:{f.Name}", (out string e) => Ptx.CrossEntropyRows(gpu, f, lane, out e));
        if (f.IsBlock)
        {
            // A block format: the rows read in place, each row's loss in f32 (one a row), those encoded by blocks.
            ulong losses = gpu.Rent(rows * 4L);
            try
            {
                Gpu.Launch(fn, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, targets, losses, Gpu.U(cols));
                Back(f, Lane.F32, losses, output, rows);
            }
            finally { gpu.Return(losses); }
            return "f32 (blocks in place)";
        }

        if (f.Bits < 8) Gpu.Clear(output, f.Words(rows) * 4);
        Gpu.Launch(fn, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, targets, output, Gpu.U(cols));
        return lane.Name;
    }

    /// <summary>The sampled column of each row from weights x and a uniform draw u per row (as the type); in the float lane.</summary>
    public string Sample<TN>(ulong x, int rows, int cols, ulong u, ulong picks) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var fn = gpu.Cached($"sample:{f.Name}", (out string e) => Ptx.SampleRows(gpu, f, lane, out e));
        Gpu.Launch(fn, (uint)rows, 1, 256, 1, 0, x, u, picks, Gpu.U(cols));
        return lane.Name;
    }

    /// <summary>counts[b], offsets[b + 1] and the sources' order by bucket, from n indices into `buckets` (scatter-add's
    /// ordering: the radix sort of GpuOps.Radix.cs).</summary>
    public void Route(ulong index, int n, int buckets, ulong counts, ulong offsets, ulong order) => OrderByTarget(index, n, buckets, counts, offsets, order);
}
