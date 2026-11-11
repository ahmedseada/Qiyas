// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's steps on the CPU (Decoding.cs): the ring's push, and the penalties — a task a run of rows; a row's window
// read from its ring and ordered once, then the row's scores a tile at a time through the float lane, the window's tokens
// that fall in the tile penalized there (a run of equal tokens once, its length the occurrences), the tile stored as the
// type (a block format's rows that share blocks as the lane's values, encoded once after).
using System.Buffers;
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>history[r·capacity + length[0] mod capacity] = ids[r].</summary>
    public static void HistoryPush(int[] ids, int[] history, int[] length, int rows, int capacity)
    {
        if (rows < 0 || capacity < 1) throw new ArgumentException($"history: {rows} rows, a ring of {capacity}");
        int slot = (int)((uint)length[0] % (uint)capacity);
        for (int r = 0; r < rows; r++) history[(long)r * capacity + slot] = ids[r];
    }

    /// <summary>The rows' scores with their windows' tokens penalized (IDecoding.Penalize); returns the lane's name.</summary>
    public static string Penalize<TN>(PenaltyArgs a, byte[] x, int[] history, int[] length, byte[] output) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        return Mixed<TN, PenalizeCall>(new PenalizeCall(a, x, history, length[0], output));
    }

    private readonly struct PenalizeCall(PenaltyArgs a, byte[] x, int[] history, int length, byte[] output) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.Values == 0) return TL.Name;
            int chunk = Math.Max(1, a.Rows / (Threads * 4));
            if (TN.Granularity <= 1 || (a.Cols % TN.Granularity == 0 && Tile % TN.Granularity == 0))
            {
                // A row is whole storage units (a block format's: whole blocks, a tile of them at a time): stored as the type.
                Team.For(a.Rows, chunk, a.Values * 2, new PenalizeBody<TN, TN, TLane, TL>(a, x, history, length, output));
                return TL.Name;
            }

            // A block format whose rows share blocks: the rows' values in the lane's own storage, encoded once.
            var lanes = new byte[a.Values * Unsafe.SizeOf<TLane>()];
            Team.For(a.Rows, chunk, a.Values * 2, new PenalizeBody<TN, TS, TLane, TL>(a, x, history, length, lanes));
            StoreLanes<TN, TLane>(lanes, output, 0, a.Values);
            return TL.Name;
        }
    }

    private readonly struct PenalizeBody<TN, TO, TLane, TL>(PenaltyArgs a, byte[] x, int[] history, int length, byte[] output) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TO : struct, IAdad<TO, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int n = a.Window(length), cols = a.Cols;
            var tokens = ArrayPool<int>.Shared.Rent(Math.Max(1, n));
            Span<TLane> v = stackalloc TLane[Tile];
            TLane repeat = TL.From(a.Repeat), presence = TL.From(a.Presence), frequency = TL.From(a.Frequency);
            try
            {
                for (int r = first; r < last; r++)
                {
                    // The window: the last n tokens of the row's ring, ordered (equal tokens adjacent, those below the row first).
                    for (int j = 0; j < n; j++) tokens[j] = history[(long)r * a.Capacity + (length - 1 - j) % a.Capacity];
                    tokens.AsSpan(0, n).Sort();
                    int t = 0;
                    while (t < n && tokens[t] < 0) t++;
                    for (int at = 0; at < cols; at += Tile)
                    {
                        int len = Math.Min(Tile, cols - at);
                        TN.Load(x, (int)((long)r * a.Stride + a.RowOffset + at), v[..len]);
                        while (t < n && tokens[t] < at + len)
                        {
                            int token = tokens[t], count = 1;
                            while (t + count < n && tokens[t + count] == token) count++;
                            t += count;
                            ref TLane s = ref v[token - at];
                            TLane y = Widths.Compare(s, TL.Zero) > 0 ? TL.Div(s, repeat) : TL.Mul(s, repeat);
                            s = TL.Sub(y, TL.Fma(frequency, TL.From(count), presence));
                        }

                        TO.Store(v[..len], output, (int)((long)r * cols + at));
                    }
                }
            }
            finally { ArrayPool<int>.Shared.Return(tokens); }
        }
    }
}
