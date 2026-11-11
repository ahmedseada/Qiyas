// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's steps on the device (Decoding.cs; the kernels in Ptx.Decoding.cs): the ring and its length stay on the
// device, every operation reading the length where it is.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>history[r·capacity + length[0] mod capacity] = ids[r] (Ptx.HistoryPush).</summary>
    public void HistoryPush(ulong ids, ulong history, ulong length, int rows, int capacity)
    {
        if (rows < 0 || capacity < 1) throw new ArgumentException($"history: {rows} rows, a ring of {capacity}");
        if (rows == 0) return;
        var fn = gpu.Cached("history_push", (out string e) => Ptx.HistoryPush(gpu, out e));
        Gpu.Launch(fn, Grid(rows, 256), 1, 256, 1, 0, ids, history, length, Gpu.U(rows), Gpu.U(capacity));
    }

    /// <summary>The rows' scores with their windows' tokens penalized (IDecoding.Penalize): each row's window ordered into
    /// a table with its mask (Ptx.PenaltyWindow), then the store (Ptx.PenalizeRows): every stored word, or — contiguous
    /// rows of whole stored words — the rows copied and the words that hold a window's token alone. Returns the lane's name.</summary>
    public string Penalize<TN>(PenaltyArgs a, ulong x, ulong history, ulong length, ulong output) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.Values == 0) return lane.Name;
        int slots = Ptx.PenaltySlots(a.MaxWindow);
        ulong win = gpu.Rent(Math.Max(16, (long)a.Rows * Ptx.PenaltyRowWords(slots) * 4));
        try
        {
            var window = gpu.Cached($"penalty_window:{slots}", (out string e) => Ptx.PenaltyWindow(gpu, slots, out e));
            Gpu.Launch(window, Grid((long)a.Rows * slots, 256), 1, 256, 1, 0, history, length, win, Gpu.U(a.Rows), Gpu.U(a.Capacity), Gpu.U(a.MaxWindow));
            ulong Slot(double v) => lane.Wide ? Gpu.D(v) : Gpu.F((float)v);
            // Contiguous rows of whole stored words: the rows copied as they are stored, then only the words that hold a
            // window's token stored (a thread a slot of the tables).
            bool sparse = !f.IsBlock && a.Stride == a.Cols && a.RowOffset == 0 && (long)a.Cols * f.Bits % f.WordBits == 0;
            var fn = gpu.Cached($"penalize_rows:{f.Name}:{lane.Name}:{slots}:{sparse}", (out string e) => Ptx.PenalizeRows(gpu, f, lane, slots, sparse, out e));
            if (sparse)
            {
                if (output != x) Gpu.Copy(x, output, a.Values * f.Bits / 8);
                Gpu.Launch(fn, Grid((long)a.Rows * slots, 256), 1, 256, 1, 0, x, win, output, Gpu.U(a.Rows), Gpu.U(a.Cols), Gpu.U(a.Stride), Gpu.U(a.RowOffset), Slot(a.Repeat), Slot(a.Presence), Slot(a.Frequency));
            }
            else LaunchEach(fn, f, a.Values, x, win, output, Gpu.U((int)a.Values), Gpu.U(a.Cols), Gpu.U(a.Stride), Gpu.U(a.RowOffset), Slot(a.Repeat), Slot(a.Presence), Slot(a.Frequency));
        }
        finally { gpu.Return(win); }

        return lane.Name;
    }
}
