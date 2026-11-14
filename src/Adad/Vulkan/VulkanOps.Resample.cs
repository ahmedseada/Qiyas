// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Resampling with a per-channel scale and shift on Vulkan (Kernels.Resample.cs), CUDA's plan (Cuda/GpuOps.Resample.cs): the
// horizontal pass into a rented plane of lane values where it runs, then the outputs a word an invocation
// (a block format's by its encoder, from lane storage where the encoder is staged), the vertical pass reading the
// horizontal one's values (or the input) where they are.
// Where both passes run in the f32 lane into a plain format of eight or more values a word: both in one kernel by tiles
// (VkKernels.ResampleTiled), the middle plane's values kept in workgroup memory and a value an invocation.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    public string Resample<TIn, TN>(ResampleArgs a, ulong x, ulong bounds, ulong weights, ulong values, ulong y) where TIn : struct, IAdad<TIn> where TN : struct, IAdad<TN>
    {
        a.Validate();
        Fmt fi = Fmt.Of<TIn>(), f = Fmt.Of<TN>();
        var lane = f.FloatLane;
        string name = lane.Name;
        if (a.Count == 0) return name;
        // The tiles where both passes run, the output is a plain format of eight or more values a word whose rows are whole
        // words, the lane is f32, and the tile's rows fit workgroup memory. Measured: the two kernels' output a word an
        // invocation, its 8 to 32 values one after another, is what the tiles' value an invocation saves (int4 0.717 ms
        // against 0.850, uint2 0.677 against 0.855); where a word holds 4 or fewer, the tiles lose more than the middle
        // plane's traffic saves (float16 0.742 against 0.525, int8 0.724 against 0.674: a tile's rows made again by the
        // next tile's first rows, a workgroup's memory held). The f64 lane's arithmetic is its cost.
        if (a.XTaps > 0 && a.YTaps > 0 && !f.IsBlock && !lane.Wide && f.Per >= 8 && a.OutWidth % f.Per == 0)
        {
            int rows = VkKernels.ResampleTileRows(a);
            uint tilesX = Groups(a.OutWidth, VkKernels.ResampleTileX), tilesY = Groups(a.OutHeight, VkKernels.ResampleTileY);
            if (VkKernels.ResampleTiledBytes(f, lane, rows) <= vk.MaxSharedBytes && (long)a.Planes * tilesY <= vk.MaxGroupsY)
            {
                bool normed = values != 0;
                var tiled = vk.Cached($"resample_tiled:{f.Name}:{fi.Name}:{lane.Name}:{rows}:{normed}", () => VkKernels.ResampleTiled(vk, f, fi, lane, normed, rows));
                vk.Launch(tiled, tilesX, (uint)(a.Planes * tilesY), x, bounds, weights, values, y, Vulkan.U(a.OutHeight), Vulkan.U(a.OutWidth),
                    Vulkan.U(a.Height), Vulkan.U(a.Width), Vulkan.U(a.XTaps), Vulkan.U(a.YTaps), Vulkan.U(a.RowBounds), Vulkan.U(a.RowWeights),
                    Vulkan.U(a.Channels), Vulkan.U((int)tilesY));
                return name;
            }
        }

        var midFmt = lane.Wide ? new Fmt("float64", 64, true, true) : new Fmt("float32", 32, true, true);
        ulong mid = 0;
        try
        {
            if (a.XTaps > 0)
            {
                long n = a.MiddleCount;
                mid = vk.Rent(n * lane.Size);
                var rows = vk.Cached($"resample_rows:{fi.Name}:{lane.Name}", () => VkKernels.ResampleRows(vk, fi, lane));
                vk.Launch(rows, Groups(n, VkKernels.Threads), 1, x, bounds, weights, mid, Vulkan.U((int)n), Vulkan.U(a.Width), Vulkan.U(a.OutWidth), Vulkan.U(a.XTaps));
            }

            var src = a.XTaps > 0 ? midFmt : fi;
            ulong s = a.XTaps > 0 ? mid : x;
            bool vertical = a.YTaps > 0, norm = values != 0;
            if (f.IsBlock && !VkKernels.EncodesDirect(vk, f))
            {
                // A block format whose encoder is staged (Q4_0: an invocation a block): its values by the plain kernel into
                // lane storage (a word an invocation, each value its taps once), then the encoder from them (BlockFromLanes),
                // not each block's 32 values' taps one after another in the encoder's source (q4_0 1.645 ms against 0.658;
                // a direct encoder's teams make a value an invocation already, q8_0 0.531 against 0.636 through lanes). The
                // same values (ResampleValue in the f32 lane either way): the same blocks.
                var lf = VkKernels.LaneFormat(Lane.F32);
                ulong lanes = vk.Rent(a.Count * 4);
                try
                {
                    var k = vk.Cached($"resample_out:{lf.Name}:{src.Name}:{Lane.F32.Name}:{vertical}:{norm}", () => VkKernels.ResampleOut(vk, lf, src, Lane.F32, vertical, norm));
                    vk.Launch(k, Groups(lf.Words(a.Count), VkKernels.Threads), 1, s, bounds, weights, values, lanes, Vulkan.U(a.Count), Vulkan.U(a.OutHeight), Vulkan.U(a.OutWidth),
                        Vulkan.U(a.Height), Vulkan.U(a.Across), Vulkan.U(a.YTaps), Vulkan.U(a.RowBounds), Vulkan.U(a.RowWeights), Vulkan.U(a.Channels));
                    Encode(f, a.Count, vk.Cached($"block_lanes:{Lane.F32.Name}:{f.Name}", () => VkKernels.BlockFromLanes(vk, Lane.F32, f)), y, lanes);
                }
                finally
                {
                    vk.Return(lanes);
                }
            }
            else if (f.IsBlock)
                Encode(f, a.Count, vk.Cached($"block_resample_out:{f.Name}:{src.Name}:{vertical}:{norm}", () => VkKernels.BlockResampleOut(vk, f, src, vertical, norm)), y,
                    s, bounds, weights, values, Vulkan.U(a.OutHeight), Vulkan.U(a.OutWidth), Vulkan.U(a.Height), Vulkan.U(a.Across), Vulkan.U(a.YTaps), Vulkan.U(a.RowBounds),
                    Vulkan.U(a.RowWeights), Vulkan.U(a.Channels));
            else
            {
                var k = vk.Cached($"resample_out:{f.Name}:{src.Name}:{lane.Name}:{vertical}:{norm}", () => VkKernels.ResampleOut(vk, f, src, lane, vertical, norm));
                vk.Launch(k, Groups(f.Words(a.Count), VkKernels.Threads), 1, s, bounds, weights, values, y, Vulkan.U(a.Count), Vulkan.U(a.OutHeight), Vulkan.U(a.OutWidth),
                    Vulkan.U(a.Height), Vulkan.U(a.Across), Vulkan.U(a.YTaps), Vulkan.U(a.RowBounds), Vulkan.U(a.RowWeights), Vulkan.U(a.Channels));
            }
        }
        finally
        {
            if (mid != 0) vk.Return(mid);
        }

        return name;
    }
}
