// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's steps on Vulkan (Decoding.cs; the kernels in Kernels.Decoding.cs): the ring and its length stay on the
// device, every operation reading the length where it is.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>history[r·capacity + length[0] mod capacity] = ids[r] (VkKernels.HistoryPush).</summary>
    public void HistoryPush(ulong ids, ulong history, ulong length, int rows, int capacity)
    {
        if (rows < 0 || capacity < 1) throw new ArgumentException($"history: {rows} rows, a ring of {capacity}");
        if (rows == 0) return;
        vk.Launch(vk.Cached("history_push", () => VkKernels.HistoryPush(vk)), Groups(rows, VkKernels.Threads), 1, ids, history, length, Vulkan.U(rows), Vulkan.U(capacity));
    }

    /// <summary>The rows' scores with their windows' tokens penalized (IDecoding.Penalize): each row's window ordered into
    /// a table with its mask (VkKernels.PenaltyWindow), then the store (VkKernels.PenalizeRows; a block format's by its
    /// encoder), or — contiguous rows of whole stored words — the rows copied and the words that hold a window's token
    /// alone (VkKernels.PenalizeSparse). Returns the lane's name.</summary>
    public string Penalize<TN>(PenaltyArgs a, ulong x, ulong history, ulong length, ulong output) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.Values == 0) return lane.Name;
        int slots = VkKernels.PenaltySlots(a.MaxWindow);
        ulong win = vk.Rent(Math.Max(16, (long)a.Rows * VkKernels.PenaltyRowWords(slots) * 4));
        try
        {
            vk.Launch(vk.Cached($"penalty_window:{slots}", () => VkKernels.PenaltyWindow(vk, slots)), Groups((long)a.Rows * slots, VkKernels.Threads), 1,
                history, length, win, Vulkan.U(a.Rows), Vulkan.U(a.Capacity), Vulkan.U(a.MaxWindow));
            if (f.IsBlock)
            {
                Encode(f, a.Values, vk.Cached($"block_penalize_rows:{f.Name}:{slots}", () => VkKernels.BlockPenalizeRows(vk, f, slots)), output,
                    x, win, Vulkan.U(a.Cols), Vulkan.U(a.Stride), Vulkan.U(a.RowOffset), Vulkan.F((float)a.Repeat), Vulkan.F((float)a.Presence), Vulkan.F((float)a.Frequency));
                return lane.Name;
            }

            if (a.Stride == a.Cols && a.RowOffset == 0 && (long)a.Cols * f.Bits % f.WordBits == 0)
            {
                // Contiguous rows of whole stored words: the rows copied as they are stored, then only the words that hold a
                // window's token stored (an invocation a slot of the tables).
                if (output != x) vk.Copy(x, output, a.Values * f.Bits / 8);
                vk.Launch(vk.Cached($"penalize_sparse:{f.Name}:{lane.Name}:{slots}", () => VkKernels.PenalizeSparse(vk, f, lane, slots)), Groups((long)a.Rows * slots, VkKernels.Threads), 1,
                    x, win, output, Vulkan.U(a.Rows), Vulkan.U(a.Cols), Vulkan.U(a.Stride), Vulkan.U(a.RowOffset), Slot(lane, a.Repeat), Slot(lane, a.Presence), Slot(lane, a.Frequency));
                return lane.Name;
            }

            var fn = vk.Cached($"penalize_rows:{f.Name}:{lane.Name}:{slots}", () => VkKernels.PenalizeRows(vk, f, lane, slots));
            vk.Launch(fn, Groups(f.Words(a.Values), VkKernels.Threads), 1, x, win, output, Vulkan.U((int)a.Values), Vulkan.U(a.Cols), Vulkan.U(a.Stride), Vulkan.U(a.RowOffset),
                Slot(lane, a.Repeat), Slot(lane, a.Presence), Slot(lane, a.Frequency));
        }
        finally { vk.Return(win); }

        return lane.Name;
    }
}
