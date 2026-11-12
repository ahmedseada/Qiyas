// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The mixture of experts on the CPU (Experts.cs): the routing (each token's k experts at the slots of their ranks among
// the token's probabilities in TopK's order, in the float lane, then the route of the slots, Routing.cs), and the experts'
// products over the rows grouped by expert: each expert's rows of x times its weights (read where they are stored,
// transposed or not) into the lane's sums, then one store adding each row's expert's bias and applying the gate (a
// formula of Map, in the lane), rounded to the type once. A float type sums in its float lane; another type exactly, in
// the lane its products take by the bounds of every value the store makes (ExpertView.IntegerSums), the sums kept as the
// lane's own type and the store in its plain lane (ILane.Kept), its gate an any-type formula.

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

    public static string ExpertMatMul<TN>(ExpertView v, byte[] x, int[] offsets, byte[] w, byte[]? bias, byte[] y) where TN : struct, IAdad<TN>
    {
        v.Validate();
        return ExpertsOf<TN>(new ExpertArgs(v, null, x, offsets, w, bias, null, null, y));
    }

    public static string ExpertGatedPair<TN>(ExpertView v, string gate, byte[] x, int[] offsets, byte[] gw, byte[]? gb, byte[] uw, byte[]? ub, byte[] hidden) where TN : struct, IAdad<TN>
    {
        v.Validate();
        Experts.Gate(gate);
        return ExpertsOf<TN>(new ExpertArgs(v, gate, x, offsets, gw, gb, uw, ub, hidden));
    }

    // A float type in its float lane; another in the lane its products take by the bounds of every value the store makes.
    private static string ExpertsOf<TN>(ExpertArgs a) where TN : struct, IAdad<TN>
    {
        if (TN.IsFloat)
        {
            var call = new MixedVisitor<ExpertCall>(new ExpertCall(a));
            TN.TryFloat(ref call);
            return call.Lane;
        }

        var v = a.V;
        (double, double) R(byte[] b, long n) => TN.Range(b, (int)n);
        var (bounds, _) = v.IntegerSums(TN.Name, TN.Min < 0, a.Gate, R(a.X, v.XValues), R(a.Gw, v.WeightValues), a.Gb is null ? null : R(a.Gb, v.BiasValues),
            a.Uw is null ? null : R(a.Uw, v.WeightValues), a.Ub is null ? null : R(a.Ub, v.BiasValues));
        var sum = new ExpertSum(a);
        TN.Sum(ref sum, bounds);
        return sum.Lane;
    }

    private readonly record struct ExpertArgs(ExpertView V, string? Gate, byte[] X, int[] Offsets, byte[] Gw, byte[]? Gb, byte[]? Uw, byte[]? Ub, byte[] Y);

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

    // Each token's k experts: those of rank below k (the experts before it in TopK's order, Routing.Precedes), each at the
    // slot of its rank; their probabilities, normalized by their sum in slot order.
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
                    for (int e = 0; e < experts; e++)
                    {
                        int rank = 0;
                        for (int q = 0; q < experts && rank < k; q++) if (Precedes(row[q], q, row[e], e)) rank++;
                        if (rank >= k) continue;
                        chosen[t * k + rank] = e;
                        w[t * k + rank] = row[e];
                    }

                    if (!normalize) continue;
                    TLane sum = TL.Zero;
                    for (int i = 0; i < k; i++) sum = TL.Add(sum, w[t * k + i]);
                    for (int i = 0; i < k; i++) w[t * k + i] = TL.Div(w[t * k + i], sum);
                }
            }
            finally { ArrayPool<TLane>.Shared.Return(row); }
        }

        // TopK's order in the lane (Routing.Precedes; Widths.Compare: a NaN above every number, two NaNs equal, −0 = +0),
        // an equal value by the lower index.
        private static bool Precedes(TLane a, int ia, TLane b, int ib)
        {
            int c = Widths.Compare(a, b);
            return c > 0 || (c == 0 && ia < ib);
        }
    }

    // A float type: the sums in its float lane (stored as the lane's own type), the gate a float or an any-type formula.
    private readonly struct ExpertCall(ExpertArgs a) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (a.V.YValues == 0) return TL.Name;
            var use = ExpertStoreOf<TN, TS, TLane, TLane, TL, TL>(a);
            string op = a.Gate ?? "copy";
            if (!MapOps.FloatOp<TLane, TL, ExpertStoreUse<TN, TLane, TL>>(op, ref use) && !MapOps.AnyOp<TLane, TL, ExpertStoreUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException($"experts: no formula {op}");
            return TL.Name;
        }
    }

    // A type that is not float: the lane TN.Sum takes by the bounds, its sums kept as the lane's own type.
    private struct ExpertSum(ExpertArgs a) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            var kept = new ExpertKept<TN, TLane, TLane, TL>(a);
            TL.Kept(ref kept);
            Lane = TL.Name;
        }

        public void RunDot<TN, TIn, TAcc, TD>() where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            var kept = new ExpertKept<TN, TIn, TAcc, TD>(a);
            TD.Kept(ref kept);
            Lane = TD.Name;
        }
    }

    // The products in TL (TIn read, TAcc summed) into sums stored as TS, the store in TP with the any-type gate.
    private readonly struct ExpertKept<TN, TIn, TAcc, TL>(ExpertArgs a) : IKeptVisitor<TAcc>
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        public void Run<TS, TP>() where TS : struct, IAdad<TS, TAcc> where TP : struct, ILane<TP, TAcc>
        {
            if (a.V.YValues == 0) return;
            var use = ExpertStoreOf<TN, TS, TIn, TAcc, TL, TP>(a);
            string op = a.Gate ?? "copy";
            if (!MapOps.AnyOp<TAcc, TP, ExpertStoreUse<TN, TAcc, TP>>(op, ref use)) throw new NotSupportedException($"experts: {op} is not a formula every lane has");
        }
    }

    // The experts' sums (the gate's, and the up's with a gate) and the store's inputs: each row's expert (Experts: a row
    // outside every expert's range) and the biases in the lane.
    private static ExpertStoreUse<TN, TAcc, TP> ExpertStoreOf<TN, TS, TIn, TAcc, TL, TP>(ExpertArgs a)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TS : struct, IAdad<TS, TAcc> where TIn : unmanaged where TAcc : unmanaged
        where TL : struct, ILane<TL, TAcc> where TP : struct, ILane<TP, TAcc>
    {
        var v = a.V;
        if (a.Offsets.Length < v.Experts + 1) throw new ArgumentException($"experts: {a.Offsets.Length} offsets for {v.Experts} experts");
        var gs = ExpertSums<TN, TS, TIn, TAcc, TL>(v, a.X, a.Offsets, a.Gw, Scratch.Sums);
        var us = a.Uw is null ? null : ExpertSums<TN, TS, TIn, TAcc, TL>(v, a.X, a.Offsets, a.Uw, Scratch.SecondSums);
        var rowExpert = new int[v.Rows];
        Array.Fill(rowExpert, v.Experts);
        for (int e = 0; e < v.Experts; e++) for (int r = Math.Max(0, a.Offsets[e]); r < Math.Min(v.Rows, a.Offsets[e + 1]); r++) rowExpert[r] = e;
        static TAcc[]? Bias(byte[]? b, long count)
        {
            if (b is null) return null;
            var l = new TAcc[count];
            TN.Load(b, 0, l);
            return l;
        }

        return new ExpertStoreUse<TN, TAcc, TP>(v, gs, Bias(a.Gb, v.BiasValues), us, Bias(a.Ub, v.BiasValues), rowExpert, a.Y);
    }

    // Each expert's rows of x times its weights into the lane's sums stored as TS (Rows × N; rows outside every expert's
    // not written: the store reads none of them), in the calling thread's scratch slot (kept from call to call).
    private static byte[] ExpertSums<TN, TS, TIn, TAcc, TL>(ExpertView v, byte[] x, int[] offsets, byte[] w, int slot)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TS : struct, IAdad<TS, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        var sums = Scratch.Get(slot, v.YValues * Unsafe.SizeOf<TAcc>());
        for (int e = 0; e < v.Experts; e++)
        {
            int r0 = offsets[e], r1 = offsets[e + 1];
            if (r0 < 0 || r1 > v.Rows || r1 < r0) throw new ArgumentException($"experts: expert {e}'s rows {r0} .. {r1} outside the {v.Rows} rows");
            if (r1 == r0 || v.N == 0) continue;
            MatMul<TN, TN, TS, TIn, TAcc, TL>(x, w, sums, r1 - r0, v.N, v.K, new Operands(AOff: r0 * v.K, BOff: (int)(e * (long)v.K * v.N), TransB: v.TransW, COff: r0 * v.N));
        }

        return sums;
    }

    private readonly struct ExpertStoreUse<TN, TLane, TL>(ExpertView v, byte[] gs, TLane[]? gb, byte[]? us, TLane[]? ub, int[] rowExpert, byte[] y) : IOpUse<TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Use<TOp>() where TOp : struct, IMapOp<TLane, TL>
        {
            int n = (int)v.YValues, size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, (long)n * 4, new ExpertStoreBody<TN, TLane, TL, TOp>(v, gs, gb, us, ub, rowExpert, y));
        }
    }

    // y = gate(g + gb[e], u + ub[e]) (no up: g + gb[e]) of every output, in aligned tiles of the index, 0 outside every expert.
    private readonly struct ExpertStoreBody<TN, TLane, TL, TOp>(ExpertView v, byte[] gs, TLane[]? gb, byte[]? us, TLane[]? ub, int[] rowExpert, byte[] y) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
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
