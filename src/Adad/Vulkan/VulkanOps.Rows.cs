// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The row operations, softmax and rms_norm among them, and the normalization parameters' backward on Vulkan
// (Kernels.Rows.cs): the rows' statistics by a workgroup a row, then the outputs a thread per stored word. Float types
// in the f32 lane (float64 in f64), as on CUDA; the block formats have no Vulkan encoders yet (refused).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    private static Lane RowLane(Fmt f) => f.Bits == 64 ? Lane.F64 : Lane.F32;

    public string Softmax<TN>(ulong x, ulong y, int rows, int cols) where TN : struct, IFloatAdad<TN> =>
        Row(Fmt.Of<TN>(), VkKernels.SoftmaxRow, x, 0, 0, 0, 0, 0, y, 0, rows, cols, 0, 0);

    public string RmsNorm<TN>(ulong x, ulong y, int rows, int cols, double eps) where TN : struct, IFloatAdad<TN> =>
        Row(Fmt.Of<TN>(), VkKernels.RmsNormRow, x, 0, 0, 0, 0, 0, y, 0, rows, cols, eps, 0);

    /// <summary>A row operation (RowOps.All) over x (rows × cols) into y (and y2); null inputs are 0 addresses. Returns the lane's name.</summary>
    public string RowOp<TN>(RowOp op, ulong x, ulong x2, ulong g, ulong bt, ulong s, ulong pos, ulong y, ulong y2, int rows, int cols, double a, double b)
        where TN : struct, IFloatAdad<TN> =>
        Row(Fmt.Of<TN>(), op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b);

    private string Row(Fmt f, RowOp op, ulong x, ulong x2, ulong g, ulong bt, ulong s, ulong pos, ulong y, ulong y2, int rows, int cols, double a, double b)
    {
        if (f.IsBlock) throw Missing($"{op.Name} of {f.Name} (a block format)");
        var lane = RowLane(f);
        long n = (long)rows * cols;
        if (n == 0) return lane.Name;
        bool second = op.Second && x2 != 0, out2 = op.SecondOut && y2 != 0;
        ulong stats = VkKernels.RowReduces(op) ? vk.Rent((long)rows * 4 * lane.Size) : 0;
        try
        {
            ulong[] args = [x, x2, g, bt, s, pos, y, y2, Vulkan.U(rows), Vulkan.U(cols), Slot(lane, a), Slot(lane, b), stats];
            if (stats != 0)
            {
                var st = vk.Cached($"rowstats:{op.Name}:{f.Name}:{second}", () => VkKernels.RowStats(vk, f, lane, op, second));
                vk.Launch(st, (uint)rows, 1, args);
            }

            var replay = vk.Cached($"rowreplay:{op.Name}:{f.Name}:{second}:{out2}", () => VkKernels.RowReplay(vk, f, lane, op, second, out2));
            vk.Launch(replay, Groups(f.Words(n), VkKernels.Threads), 1, args);
        }
        finally
        {
            if (stats != 0) vk.Return(stats);
        }

        return lane.Name;
    }

    /// <summary>dgamma[c] = Σ_r dy·x̂, dbeta[c] = Σ_r dy (lane values, cols each); x̂ by rms or, centered, layer statistics. Returns the lane's name.</summary>
    public string NormParamsBwd<TN>(ulong x, ulong dy, ulong dgamma, ulong dbeta, int rows, int cols, double eps, bool centered) where TN : struct, IFloatAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"norm parameters' backward of {f.Name} (a block format)");
        var lane = RowLane(f);
        if (cols == 0) return lane.Name;
        var stats = vk.Cached($"norm_stats:{centered}:{f.Name}", () => VkKernels.NormStats(vk, f, lane, centered));
        var columns = vk.Cached($"norm_params_cols:{f.Name}:{lane.Name}", () => VkKernels.NormParamsCols(vk, f, lane));
        ulong st = vk.Rent(Math.Max(1, (long)rows) * 2 * lane.Size);
        int splits = Splits(rows, cols), per = (rows + splits - 1) / splits;
        ulong pg = splits == 1 ? dgamma : vk.Rent((long)splits * cols * lane.Size), pb = splits == 1 ? dbeta : vk.Rent((long)splits * cols * lane.Size);
        try
        {
            if (rows > 0) vk.Launch(stats, (uint)rows, 1, x, 0, 0, 0, 0, 0, st, 0, Vulkan.U(rows), Vulkan.U(cols), Slot(lane, eps), Slot(lane, 0), 0);
            vk.Launch(columns, Groups(cols, VkKernels.Threads), (uint)splits, x, dy, st, pg, pb, Vulkan.U(rows), Vulkan.U(cols), Vulkan.U(per));
            if (splits > 1)
            {
                var lf = VkKernels.LaneFormat(lane);
                var join = vk.Cached($"reduce_cols:sum:False:{lf.Name}:{lane.Name}", () => VkKernels.ReduceCols(vk, lf, lane, "sum", false));
                vk.Launch(join, Groups(cols, VkKernels.ReduceThreads), 1, pg, dgamma, Vulkan.U(splits), Vulkan.U(cols), Vulkan.U(splits));
                vk.Launch(join, Groups(cols, VkKernels.ReduceThreads), 1, pb, dbeta, Vulkan.U(splits), Vulkan.U(cols), Vulkan.U(splits));
            }
        }
        finally
        {
            vk.Return(st);
            if (splits > 1) { vk.Return(pg); vk.Return(pb); }
        }

        return lane.Name;
    }
}
