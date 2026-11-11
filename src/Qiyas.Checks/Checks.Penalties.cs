// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's ring and penalties (IDecoding) on every device:
//   history   rows pushed step after step into rings of 5 with the length advanced on the device (13 steps: the ring
//             wraps twice): the rings and the length exactly the reference's.
//   penalize  every float type (the block formats among them): rows of 601 scores (past a block of threads) and rows a
//             stride apart from an offset, rings empty, part filled and wrapped, the whole ring and its last tokens, tokens
//             that repeat, one below the row and one past it; each penalized score within the type's step plus the lane's
//             rounding of its terms of the float64 reference, every other score x's own value exactly (a block format's
//             within its block's step: a block holding a penalized score is encoded anew).
namespace Qiyas;

internal static partial class Checks
{
    private static bool HistoryRings(IDevice<ulong, ulong>? ops, string where)
    {
        const int rows = 37, capacity = 5, steps = 13;
        var history = Enumerable.Repeat(-7, rows * capacity).ToArray();
        var want = (int[])history.Clone();
        var length = new[] { 0 };
        var pushes = Enumerable.Range(0, steps).Select(s => Enumerable.Range(0, rows).Select(r => s * 100 + r).ToArray()).ToArray();
        for (int s = 0; s < steps; s++) want = PenaltyArgs.PushReference(want, pushes[s], s, rows, capacity);
        if (ops is null)
        {
            for (int s = 0; s < steps; s++)
            {
                CpuDevice.Device.HistoryPush(pushes[s], history, length, rows, capacity);
                CpuDevice.Device.PositionAdvance(length, 1);
            }
        }
        else
        {
            using var st = new Staged(ops);
            ulong dh = st.IndicesOut(history), dl = st.IndicesOut(length);
            for (int s = 0; s < steps; s++)
            {
                ops.HistoryPush(st.Indices(pushes[s]), dh, dl, rows, capacity);
                ops.PositionAdvance(dl, 1);
            }

            st.Collect();
        }

        if (history.SequenceEqual(want) && length[0] == steps) return true;
        Console.WriteLine($"check history ring {where,-6} DIFFERS: length {length[0]} (of {steps}), {Enumerable.Range(0, history.Length).Count(i => history[i] != want[i])} slots");
        return false;
    }

    private static bool Penalties(IDevice<ulong, ulong>? ops, string where, string type)
    {
        bool ok = true;
        var cases = new (string Name, PenaltyArgs A, int Length)[]
        {
            ("an empty ring", new PenaltyArgs(5, 601, 9, 0, 1.3, 0.5, 0.25), 0),
            ("a ring part filled", new PenaltyArgs(5, 601, 9, 0, 1.3, 0.5, 0.25), 4),
            ("a wrapped ring", new PenaltyArgs(5, 601, 9, 0, 1.3, 0.5, 0.25), 23),
            ("the last 4 of a wrapped ring", new PenaltyArgs(5, 601, 9, 4, 1.3, 0, 0.25), 23),
            ("the last 64 of a ring of 100, rows apart", new PenaltyArgs(3, 53, 100, 64, 1.1, 0.25, 0.125) { RowStride = 61, RowOffset = 3 }, 257),
            ("the repeat alone", new PenaltyArgs(7, 37, 5, 0, 2, 0, 0), 5),
            ("rows of whole blocks", new PenaltyArgs(3, 4352, 9, 0, 1.3, 0.5, 0.25), 23),
        };
        foreach (var (name, a, length) in cases)
        {
            string what = $"penalize {name}";
            int xn = (a.Rows - 1) * a.Stride + a.RowOffset + a.Cols;
            var x = Bench.Registry.Make(type, xn, 93, -4, 4);
            var xv = Bench.Registry.Values(type, x, xn);
            // The rings: tokens that repeat within a row, the row's first and last columns, one below the row and one past it.
            var random = new Random(94);
            var history = new int[a.Rows * a.Capacity];
            for (int i = 0; i < history.Length; i++) history[i] = i % 3 == 2 ? history[i - 1] : i % 7 == 4 ? history[i - 4] : random.Next(a.Cols);
            for (int r = 0; r < a.Rows; r++)
            {
                history[r * a.Capacity] = a.Cols - 1;
                history[r * a.Capacity + 1] = r % 2 == 0 ? -1 : a.Cols + 3;
                history[r * a.Capacity + 3] = 0;
            }

            var want = a.Reference(xv, history, length);
            var y = new byte[Bench.Registry.Bytes(type, a.Values)];
            Bench.Registry.WithFloat(type, new PenalizeRun(ops, a, x, history, length, y));
            // A penalized score's terms: the score scaled, the presence and the frequency times its occurrences.
            double terms = 4 * Math.Max(a.Repeat, 1 / a.Repeat) + Math.Abs(a.Presence) + Math.Abs(a.Frequency) * a.Capacity;
            double bad = Excess(type, what, y, want, _ => terms, 4);
            string? wrong = bad > 0 ? $"DIFFERS by {bad:0.##}" : null;
            if (wrong is null && !Bench.Registry.IsBlock(type))
            {
                // A score outside the window: x's own value as stored.
                var got = Bench.Registry.Values(type, y, (int)a.Values);
                for (int i = 0; i < a.Values && wrong is null; i++)
                {
                    double source = xv[i / a.Cols * a.Stride + a.RowOffset + i % a.Cols];
                    if (want[i].Equals(source) && !got[i].Equals(source)) wrong = $"score [{i}] outside the window {got[i]:R}, not x's {source:R}";
                }
            }

            if (wrong is not null) { Console.WriteLine($"check {what,-50} {where,-6} {type,-12} {wrong}"); ok = false; }
        }

        return ok;
    }

    private sealed class PenalizeRun(IDevice<ulong, ulong>? ops, PenaltyArgs a, byte[] x, int[] history, int length, byte[] y) : Bench.IFloatTypeVisitor<string>
    {
        public string Visit<TN>() where TN : struct, IFloatAdad<TN>, IAdad<TN, float>
        {
            if (ops is null) return CpuDevice.Device.Penalize<TN>(a, x, history, [length], y);
            using var st = new Staged(ops);
            string lane = ops.Penalize<TN>(a, st.In(x), st.Indices(history), st.Indices([length]), st.Out(y));
            st.Collect();
            return lane;
        }
    }
}
