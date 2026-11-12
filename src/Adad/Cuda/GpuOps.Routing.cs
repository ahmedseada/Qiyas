// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's routing, sampling and losses (Routing.cs).
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>top-k of each row in Adad's order: values as the type (rows × k), indices s32 (k at most cols); returns the
    /// lane's name.</summary>
    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var f = Fmt.Of<TN>();
        var lane = f.IsBlock ? Lane.F32 : f.Natural;                              // a block format: its rows decoded in place, in f32
        string name = f.IsBlock ? "f32 (blocks in place)" : lane.Name;
        if (k > 256) throw new NotSupportedException($"top-k: k = {k} exceeds the kernel's passes (256); a partial sort is the step");
        if (rows == 0 || k == 0) return name;
        TopKIndices(f, lane, x, indices, rows, cols, k);
        GatherRows(f, x, indices, values, rows, cols, k);
        return name;
    }

    // Each row's k first columns in Adad's order: the one-pass kernel holding the fewest that hold k (Ptx.TopKHeld), else
    // the k passes.
    private void TopKIndices(Fmt f, Lane lane, ulong x, ulong indices, int rows, int cols, int k)
    {
        int held = Array.Find(Ptx.TopKHeld, h => h >= k);
        var fn = held > 0
            ? gpu.Cached($"topk_held:{held}:{f.Name}:{lane.Name}", (out string e) => Ptx.TopKRowsHeld(gpu, f, lane, held, out e))
            : gpu.Cached($"topk:{f.Name}", (out string e) => Ptx.TopKRows(gpu, f, lane, out e));
        Gpu.Launch(fn, (uint)rows, 1, 256, 1, 0, x, indices, Gpu.U(cols), Gpu.U(k));
    }

    // out[r·k + j] = x[r][index[r·k + j]] as the type: a thread a stored word (a block format's blocks by its encoder).
    private void GatherRows(Fmt f, ulong x, ulong index, ulong output, int rows, int cols, int k)
    {
        long n = (long)rows * k;
        if (n == 0) return;
        if (n > uint.MaxValue) throw new NotSupportedException($"a gather of {n} values past 2^32");
        var fn = gpu.Cached($"gather_rows:{f.Name}", (out string e) => Ptx.GatherRows(gpu, f, out e));
        LaunchEach(fn, f, n, x, index, output, Gpu.U((int)n), Gpu.U(cols), Gpu.U(k));
    }

    /// <summary>The top-k of each row as log-probabilities (IRouting.LogSoftmaxTopK): TopK's indices, then Ptx.LogProbRows
    /// (the row's largest the first chosen, Σ exp(x − largest) by the block's reduction, a thread a chosen column storing
    /// (x − m) − log Σ as a lane value); stored as the type once after. Returns the lane's name.</summary>
    public string LogSoftmaxTopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IFloatAdad<TN>
    {
        Routing.ValidateTopK(rows, cols, k);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (rows == 0 || k == 0) return lane.Name;
        if (k > 256) throw new NotSupportedException($"top-k: k = {k} exceeds the kernel's passes (256); a partial sort is the step");
        // TopK's indices.
        TopKIndices(f, f.IsBlock ? Lane.F32 : f.Natural, x, indices, rows, cols, k);
        bool own = f.Is("float32") || f.Is("float64");
        ulong lanes = own ? values : gpu.Rent(Math.Max(16, (long)rows * k * lane.Size));
        try
        {
            var fn = gpu.Cached($"log_probs:{f.Name}", (out string e) => Ptx.LogProbRows(gpu, f, lane, out e));
            Gpu.Launch(fn, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, indices, lanes, Gpu.U(cols), Gpu.U(k));
            FromLane(f, lanes, values, (long)rows * k);
        }
        finally
        {
            if (!own) gpu.Return(lanes);
        }

        return lane.Name;
    }

    /// <summary>out[r] = x[r][index[r]] as the type (a column outside the row: 0): TopK's gather with k = 1.</summary>
    public string Pick<TN>(ulong x, int rows, int cols, ulong index, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        GatherRows(f, x, index, output, rows, cols, 1);
        return f.IsBlock ? "f32 (blocks in place)" : f.Natural.Name;
    }


    /// <summary>loss[r] = w[r]·(log Σ exp(x[r]) − x[r][t[r]]) as the type (t: s32 targets; weights as the type, 0 for none), in
    /// the type's float lane: a block per row.</summary>
    public string CrossEntropy<TN>(ulong x, int rows, int cols, ulong targets, ulong weights, ulong output) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        bool weighted = weights != 0;
        var fn = gpu.Cached($"cross_entropy:{f.Name}{(weighted ? ":w" : "")}", (out string e) => Ptx.CrossEntropyRows(gpu, f, lane, weighted, out e));
        if (f.IsBlock)
        {
            // A block format: the rows read in place, each row's loss in f32 (one a row), those encoded by blocks.
            ulong losses = gpu.Rent(rows * 4L);
            try
            {
                Gpu.Launch(fn, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, targets, losses, Gpu.U(cols), weights);
                Back(f, Lane.F32, losses, output, rows);
            }
            finally { gpu.Return(losses); }
            return "f32 (blocks in place)";
        }

        Gpu.Launch(fn, (uint)rows, 1, Ptx.ReduceThreads, 1, 0, x, targets, output, Gpu.U(cols), weights);
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
