// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts on the CPU (Experts.cs): the routing (each token's k experts by their rank among the token's
// probabilities, in the float lane, then the route of the slots, Routing.cs), and the experts' products over the rows
// grouped by expert: each expert's rows of x times its weights (read where they are stored, transposed or not) into the
// float lane's sums, then one store adding each row's expert's bias and applying the gate (a formula of Map, in the lane),
// rounded to the type once.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    public static string RouteTopK<TN>(byte[] p, int tokens, int experts, int k, bool normalize, byte[] weights, int[] offsets, int[] tokenOfRow, int[] slotRow) where TN : struct, IFloatAdad<TN>
    {
        Experts.ValidateRoute(tokens, experts, k);
        return Mixed<TN, RouteCall>(new RouteCall(p, tokens, experts, k, normalize, weights, offsets, tokenOfRow, slotRow));
    }

    public static string ExpertMatMul<TN>(ExpertView v, byte[] x, int[] offsets, byte[] w, byte[]? bias, byte[] y) where TN : struct, IFloatAdad<TN>
    {
        v.Validate();
        return Mixed<TN, ExpertCall>(new ExpertCall(v, null, x, offsets, w, bias, null, null, y));
    }

    public static string ExpertGatedPair<TN>(ExpertView v, string gate, byte[] x, int[] offsets, byte[] gw, byte[]? gb, byte[] uw, byte[]? ub, byte[] hidden) where TN : struct, IFloatAdad<TN>
    {
        v.Validate();
        Experts.Gate(gate);
        return Mixed<TN, ExpertCall>(new ExpertCall(v, gate, x, offsets, gw, gb, uw, ub, hidden));
    }

    private readonly struct RouteCall(byte[] p, int tokens, int experts, int k, bool normalize, byte[] weights, int[] offsets, int[] tokenOfRow, int[] slotRow) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            int slots = tokens * k;
            var w = new TLane[slots];
            var chosen = new int[slots];
            if (tokens > 0) Team.For(tokens, Math.Max(1, tokens / (Threads * 4)), (long)tokens * experts * experts, new PicksBody<TN, TLane, TL>(p, experts, k, normalize, w, chosen));
            TN.Store(w, weights, 0);
            var order = new int[slots];
            Route(chosen, experts, new int[experts], offsets, order);
            for (int r = 0; r < slots; r++)
            {
                tokenOfRow[r] = order[r] / k;
                slotRow[order[r]] = r;
            }

            return TL.Name;
        }
    }

    // Each token's k experts: those of rank below k (the experts more probable, or as probable at a lower index), in
    // increasing expert order; their probabilities, normalized by their sum in that order.
    private readonly struct PicksBody<TN, TLane, TL>(byte[] p, int experts, int k, bool normalize, TLane[] w, int[] chosen) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var row = ArrayPool<TLane>.Shared.Rent(experts);
            try
            {
                for (int t = first; t < last; t++)
                {
                    TN.Load(p, t * experts, row.AsSpan(0, experts));
                    int s = 0;
                    TLane sum = TL.Zero;
                    for (int e = 0; e < experts; e++)
                    {
                        int rank = 0;
                        for (int q = 0; q < experts && rank < k; q++)
                        {
                            int c = Widths.Compare(row[q], row[e]);
                            if (c > 0 || (c == 0 && q < e)) rank++;
                        }

                        if (rank >= k) continue;
                        chosen[t * k + s] = e;
                        w[t * k + s] = row[e];
                        sum = TL.Add(sum, row[e]);
                        s++;
                    }

                    if (normalize) for (int i = 0; i < k; i++) w[t * k + i] = TL.Div(w[t * k + i], sum);
                }
            }
            finally { ArrayPool<TLane>.Shared.Return(row); }
        }
    }

    private readonly struct ExpertCall(ExpertView v, string? gate, byte[] x, int[] offsets, byte[] gw, byte[]? gb, byte[]? uw, byte[]? ub, byte[] y) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (v.YValues == 0) return TL.Name;
            if (offsets.Length < v.Experts + 1) throw new ArgumentException($"experts: {offsets.Length} offsets for {v.Experts} experts");
            var gs = ExpertSums<TN, TS, TLane, TL>(v, x, offsets, gw);
            var us = uw is null ? null : ExpertSums<TN, TS, TLane, TL>(v, x, offsets, uw);
            // Each row's expert (Experts: a row outside every expert's range).
            var rowExpert = new int[v.Rows];
            Array.Fill(rowExpert, v.Experts);
            for (int e = 0; e < v.Experts; e++) for (int r = Math.Max(0, offsets[e]); r < Math.Min(v.Rows, offsets[e + 1]); r++) rowExpert[r] = e;
            static TLane[]? Bias(byte[]? b, long count)
            {
                if (b is null) return null;
                var l = new TLane[count];
                TN.Load(b, 0, l);
                return l;
            }

            var use = new ExpertStoreUse<TN, TLane, TL>(v, gs, Bias(gb, v.BiasValues), us, Bias(ub, v.BiasValues), rowExpert, y);
            string op = gate ?? "copy";
            if (!MapOps.FloatOp<TLane, TL, ExpertStoreUse<TN, TLane, TL>>(op, ref use) && !MapOps.AnyOp<TLane, TL, ExpertStoreUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException($"experts: no formula {op}");
            return TL.Name;
        }
    }

    // Each expert's rows of x times its weights into the lane's sums (Rows × N, lane storage; rows outside every expert's 0).
    private static byte[] ExpertSums<TN, TS, TLane, TL>(ExpertView v, byte[] x, int[] offsets, byte[] w)
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        var sums = new byte[v.YValues * Unsafe.SizeOf<TLane>()];
        for (int e = 0; e < v.Experts; e++)
        {
            int r0 = offsets[e], r1 = offsets[e + 1];
            if (r0 < 0 || r1 > v.Rows || r1 < r0) throw new ArgumentException($"experts: expert {e}'s rows {r0} .. {r1} outside the {v.Rows} rows");
            if (r1 == r0 || v.N == 0) continue;
            MatMul<TN, TN, TS, TLane, TLane, TL>(x, w, sums, r1 - r0, v.N, v.K, new Operands(AOff: r0 * v.K, BOff: (int)(e * (long)v.K * v.N), TransB: v.TransW, COff: r0 * v.N));
        }

        return sums;
    }

    private readonly struct ExpertStoreUse<TN, TLane, TL>(ExpertView v, byte[] gs, TLane[]? gb, byte[]? us, TLane[]? ub, int[] rowExpert, byte[] y) : IOpUse<TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Use<TOp>() where TOp : struct, IMapOp<TLane, TL>
        {
            int n = (int)v.YValues, size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, (long)n * 4, new ExpertStoreBody<TN, TLane, TL, TOp>(v, gs, gb, us, ub, rowExpert, y));
        }
    }

    // y = gate(g + gb[e], u + ub[e]) (no up: g + gb[e]) of every output, in aligned tiles of the index, 0 outside every expert.
    private readonly struct ExpertStoreBody<TN, TLane, TL, TOp>(ExpertView v, byte[] gs, TLane[]? gb, byte[]? us, TLane[]? ub, int[] rowExpert, byte[] y) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        public void Run(int first, int last)
        {
            var g = MemoryMarshal.Cast<byte, TLane>(gs.AsSpan());
            var u = us is null ? default : MemoryMarshal.Cast<byte, TLane>(us.AsSpan());
            int lanes = Vector<TLane>.Count, room = Tile + lanes, N = v.N;
            Span<TLane> all = stackalloc TLane[2 * room];
            Span<TLane> a = all[..room], b = all.Slice(room, room);
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at), full = (len + lanes - 1) / lanes * lanes;
                for (int i = 0; i < len; i++)
                {
                    int r = (at + i) / N, c = (at + i) % N, e = rowExpert[r];
                    bool inside = e < v.Experts;
                    a[i] = !inside ? TL.Zero : gb is null ? g[at + i] : TL.Add(g[at + i], gb[e * N + c]);
                    b[i] = us is null || !inside ? a[i] : ub is null ? u[at + i] : TL.Add(u[at + i], ub[e * N + c]);
                }

                a[len..full].Clear();
                b[len..full].Clear();
                for (int i = 0; i < len; i += lanes)
                {
                    var va = new Vector<TLane>(a.Slice(i, lanes));
                    var vb = new Vector<TLane>(b.Slice(i, lanes));
                    TOp.Apply(va, vb, vb, Vector<TLane>.Zero, Vector<TLane>.Zero).CopyTo(a.Slice(i, lanes));
                }

                for (int i = 0; i < len; i++) if (rowExpert[(at + i) / N] >= v.Experts) a[i] = TL.Zero;
                TN.Store(a[..len], y, at);
            }
        }
    }
}
