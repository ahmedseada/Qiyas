// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A decoder's steps on the CPU (Decoding.cs): the ring's push, and the penalties — a task a run of rows; a row's window
// read from its ring and ordered once, then the row's scores a tile at a time in the type's natural lane, the window's
// tokens that fall in the tile penalized there in the float lane (a run of equal tokens once, its length the occurrences)
// and taken back through the type's conversion, the tile stored as the type (rows that share storage units — a block
// format's blocks, packed integers' bytes — as natural lane values, stored once after).
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
    public static string Penalize<TN>(PenaltyArgs a, byte[] x, int[] history, int[] length, byte[] output) where TN : struct, IAdad<TN>
    {
        a.Validate();
        if (TN.Bits == 1) throw new NotSupportedException($"penalties: {TN.Name} holds truth values, not scores");
        return InFloatLane<TN, PenalizeCall>(new PenalizeCall(a, x, history, length[0], output));
    }

    private readonly struct PenalizeCall(PenaltyArgs a, byte[] x, int[] history, int length, byte[] output) : IFloatOf
    {
        public string Run<TN, TE, TLane, TL>() where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.Values == 0) return TL.Name;
            int chunk = Math.Max(1, a.Rows / (Threads * 4));
            if (TN.Granularity <= 1 || (a.Cols % TN.Granularity == 0 && Tile % TN.Granularity == 0))
            {
                // A row is whole storage units (a block format's: whole blocks, a tile of them at a time): stored as the type.
                Team.For(a.Rows, chunk, a.Values * 2, new PenalizeBody<TN, TE, TLane, TL>(a, x, history, length, output, null));
                return TL.Name;
            }

            // Rows that share storage units (a block or a byte of packed values): the rows' values in TN's natural lane, stored
            // once.
            var values = new TE[a.Values];
            Team.For(a.Rows, chunk, a.Values * 2, new PenalizeBody<TN, TE, TLane, TL>(a, x, history, length, output, values));
            TN.Store(values, output, 0);
            return TL.Name;
        }
    }

    private readonly struct PenalizeBody<TN, TE, TLane, TL>(PenaltyArgs a, byte[] x, int[] history, int length, byte[] output, TE[]? values) : Team.IBody
        where TN : struct, IAdad<TN, TE> where TE : unmanaged where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            int n = a.Tokens(length), cols = a.Cols;
            var tokens = ArrayPool<int>.Shared.Rent(Math.Max(1, n));
            Span<TE> v = stackalloc TE[Tile];
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
                            // The score in the float lane, penalized, and back through the type's conversion.
                            TLane s = ToFloatLane<TE, TLane>(v[token - at]);
                            TLane y = Widths.Compare(s, TL.Zero) > 0 ? TL.Div(s, repeat) : TL.Mul(s, repeat);
                            v[token - at] = FromFloatLane<TLane, TE>(TL.Sub(y, TL.Fma(frequency, TL.From(count), presence)));
                        }

                        if (values is null) TN.Store(v[..len], output, (int)((long)r * cols + at));
                        else v[..len].CopyTo(values.AsSpan((int)((long)r * cols + at), len));
                    }
                }
            }
            finally { ArrayPool<int>.Shared.Return(tokens); }
        }
    }
}
