// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention rows fused (flash): O for rows of queries without the Lq × Lk scores in memory. Per task a head's block of
// query rows; the keys walked in tiles: the tile's scores in the float lane (Kᵀ's tile laid so a key run is a vector:
// S[r][c..c+w) = Σ_t q[r][t]·Kᵀ[t][c..c+w), no horizontal sums), the softmax online (the running max m and sum l per
// row; the row's accumulator rescaled by exp(m_old − m_new) when the max moves), then O[r] += Σ_c p_c·V[c] along d.
// Two rows at a time in registers: the scores of a row pair over four key vectors (eight accumulators, each key vector
// loaded once for both rows), the context of a row pair over four dimension vectors (each value vector loaded once).
// The keys a row sees are [Start(i), Limit(i)) as in the composition; a block's tiles span the union of its rows',
// starting at a whole tile.
//
// Every operand read where it is stored (the type's own lane values in place when its bytes are the lane's): a task
// loads its block of Q rows into lanes once, each tile of Kᵀ and of V as it reaches it, and stores its block of O as the
// type. Kᵀ's tiles are laid once per call, in the type itself (item 6i: every block of a head reads the same tiles), for
// a type of whole values; a block format's are decoded and laid per task from K's rows (its tasks take more rows, so a
// tile serves more of them), and its O is the lane's values encoded once where a task's rows are not whole blocks.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    private const int FusedRows = 16, FusedKeys = 64;                       // a task's query rows; a key tile (L1: 64 × d lanes, twice)

    /// <summary>O for rows of queries by the fused rows (no scores in memory); returns the lane's name.</summary>
    public static string AttentionRows<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IFloatAdad<TN>
    {
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionComposed<TN>(a, q, k, v, o);
        var vis = new FusedVisitor(a, q, k, v, o);
        TN.Float(ref vis);
        return vis.Lane;
    }

    // A task's query rows: 16 where the operands are read in place; a type that loads its tiles into lanes (a narrow float,
    // a block format) more, so each tile it loads serves them all, while the tasks still cover the team.
    private static int RowsPerTask<TN, TLane>(AttentionArgs a) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        if (TN.Direct) return FusedRows;
        int rows = 4 * FusedRows;
        while (rows > FusedRows && (long)a.Heads * ((a.Lq + rows - 1) / rows) < 2 * Threads) rows /= 2;
        return rows;
    }

    // Whether each task's block of rows (rows × d values from a head's start) is whole storage units of TN, so a task
    // stores its own rows; otherwise (a block format) the rows' lanes are kept whole and encoded once.
    private static bool WholeRows<TN>(int rows, int length, int d) where TN : struct, IAdad<TN> =>
        TN.Granularity == 1 || ((long)rows * d % TN.Granularity == 0 && (long)length * d % TN.Granularity == 0);

    /// <summary>Kᵀ's tiles of x (heads × L × d) in TN itself: tile (h, j) at (h·tiles + j)·d·bc, value t·bc + c = x[h][j·bc + c][t],
    /// zero past L; null for a block format (its tiles are laid per task).</summary>
    private static byte[]? KeyTiles<TN, TLane>(AttentionArgs a, byte[] x, int slot) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        if (TN.Granularity != 1) return null;
        int tiles = (a.Lk + FusedKeys - 1) / FusedKeys;
        var kt = Scratch.Get(slot, TN.Bytes((long)a.Heads * tiles * a.D * FusedKeys));
        Team.For(a.Heads * tiles, 1, (long)a.Heads * a.Lk * a.D, new KeyTilesBody<TN, TLane>(a, x, kt, tiles));
        return kt;
    }

    private struct FusedVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            int rows = RowsPerTask<TN, TLane>(a), blocks = (a.Lq + rows - 1) / rows, tiles = (a.Lk + FusedKeys - 1) / FusedKeys;
            var kt = KeyTiles<TN, TLane>(a, k, Scratch.AttnKt);
            var lo = WholeRows<TN>(rows, a.Lq, a.D) ? null : new byte[a.QValues * Unsafe.SizeOf<TLane>()];
            Team.For(a.Heads * blocks, 1, (long)a.Heads * a.Lq * a.Lk * a.D, new FusedBody<TN, TLane, TL>(a, q, kt, k, v, o, lo, rows, blocks, tiles));
            if (lo is not null) StoreLanes<TN, TLane>(lo, o, 0, a.QValues);
        }
    }

    // Kᵀ's tiles (a tile a task): each row of the tile's keys loaded once, its values written down the tile's columns,
    // the tile stored as the type.
    private readonly struct KeyTilesBody<TN, TLane>(AttentionArgs a, byte[] x, byte[] kt, int tiles) : Team.IBody where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        public void Run(int first, int last)
        {
            int d = a.D, bc = FusedKeys;
            var pool = ArrayPool<TLane>.Shared;
            var tile = pool.Rent(d * bc);
            var rows = pool.Rent(bc * d);
            for (int task = first; task < last; task++)
            {
                int h = task / tiles, t0 = task % tiles * bc, n = Math.Min(bc, a.Lk - t0);
                TN.Load(x, (int)((long)h * a.Lk * d + (long)t0 * d), rows.AsSpan(0, n * d));
                Lay(rows, n, d, tile);
                TN.Store(tile.AsSpan(0, d * bc), kt, task * d * bc);
            }

            pool.Return(tile); pool.Return(rows);
        }
    }

    // A tile of Kᵀ in lanes from n key rows (n × d, contiguous): tile[t·bc + c] = rows[c·d + t], zero for c ≥ n.
    private static void Lay<TLane>(TLane[] rows, int n, int d, TLane[] tile) where TLane : unmanaged
    {
        int bc = FusedKeys;
        if (n < bc) tile.AsSpan(0, d * bc).Clear();
        for (int c = 0; c < n; c++)
        {
            var row = rows.AsSpan(c * d, d);
            for (int t = 0; t < d; t++) tile[t * bc + c] = row[t];
        }
    }

    // A task's operands as lanes: in place where TN's bytes are the lane's, else loaded into the task's buffers.
    private static class Operand<TN, TLane> where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        public static ReadOnlySpan<TLane> Of(byte[] x, long at, int count, TLane[]? buffer, int bufferAt = 0)
        {
            if (TN.Direct) return MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice((int)at, count);
            TN.Load(x, (int)at, buffer.AsSpan(bufferAt, count));
            return buffer.AsSpan(bufferAt, count);
        }
    }

    private readonly struct FusedBody<TN, TLane, TL>(AttentionArgs a, byte[] q, byte[]? kts, byte[] k, byte[] v, byte[] o, byte[]? lo, int rowsPer, int blocks, int tiles) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int d = a.D, w = Vector<TLane>.Count, bc = FusedKeys, group = 4 * w;
            bool direct = TN.Direct;
            var pool = ArrayPool<TLane>.Shared;
            var s = pool.Rent(rowsPer * bc);
            var acc = pool.Rent(rowsPer * d);
            TLane[]? qb = direct ? null : pool.Rent(rowsPer * d);
            TLane[]? ktb = direct ? null : pool.Rent(d * bc);
            TLane[]? vb = direct ? null : pool.Rent(bc * d);
            TLane[]? kr = kts is null ? pool.Rent(bc * d) : null;
            var m = new TLane[rowsPer];
            var l = new TLane[rowsPer];
            var alpha = new TLane[rowsPer];
            var seen = new bool[rowsPer];
            TLane scale = TL.From(a.Scale), one = TL.From(1);
            var O = lo is not null ? MemoryMarshal.Cast<byte, TLane>(lo.AsSpan()) : direct ? MemoryMarshal.Cast<byte, TLane>(o.AsSpan()) : default;
            try
            {
                for (int task = first; task < last; task++)
                {
                    int h = task / blocks, i0 = task % blocks * rowsPer, rows = Math.Min(rowsPer, a.Lq - i0);
                    int lo0 = int.MaxValue, hi = 0;
                    for (int r = 0; r < rows; r++) { lo0 = Math.Min(lo0, a.Start(i0 + r)); hi = Math.Max(hi, a.Limit(i0 + r)); }
                    acc.AsSpan(0, rows * d).Clear();
                    Array.Clear(seen);
                    long kHead = (long)h * a.Lk * d, qHead = (long)h * a.Lq * d;
                    var Q = Operand<TN, TLane>.Of(q, qHead + (long)i0 * d, rows * d, qb);
                    for (int t0 = hi <= lo0 ? hi : lo0 / bc * bc; t0 < hi; t0 += bc)
                    {
                        // Kᵀ's tile (zero past Lk, so the groups past n score 0: outside every row's keys).
                        int n = Math.Min(bc, hi - t0), ng = (n + group - 1) / group * group;
                        int at = (h * tiles + t0 / bc) * d * bc;
                        ReadOnlySpan<TLane> kt;
                        if (kts is not null) kt = Operand<TN, TLane>.Of(kts, at, d * bc, ktb);
                        else
                        {
                            int keys = Math.Min(bc, a.Lk - t0);
                            TN.Load(k, (int)(kHead + (long)t0 * d), kr.AsSpan(0, keys * d));
                            Lay(kr!, keys, d, ktb!);
                            kt = ktb.AsSpan(0, d * bc);
                        }

                        // Scores, two rows at a time; then each row's online softmax over its keys of the tile.
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            Scores(Q.Slice(r * d, d), Q.Slice(r1 * d, d), kt, bc, ng, d, s.AsSpan(r * bc, bc), s.AsSpan(r1 * bc, bc));
                        }

                        int cLo = int.MaxValue, cHi = 0;
                        for (int r = 0; r < rows; r++)
                        {
                            int i = i0 + r, from = Math.Max(a.Start(i), t0) - t0, to = Math.Min(a.Limit(i), t0 + n) - t0;
                            var row = s.AsSpan(r * bc, bc);
                            if (to <= from) { row[..n].Clear(); alpha[r] = one; continue; }
                            var span = row[from..to];
                            for (int c = 0; c < span.Length; c++) span[c] = TL.Mul(span[c], scale);
                            TLane tmax = RowMath<TLane, TL>.Max(span);
                            TLane mNew = seen[r] ? TL.Max(m[r], tmax) : tmax;
                            alpha[r] = seen[r] ? TL.Exp(TL.Sub(m[r], mNew)) : TL.Zero;
                            TLane sum = RowMath<TLane, TL>.ExpSum(span, mNew, span);
                            row[..from].Clear();
                            row[to..n].Clear();
                            l[r] = seen[r] ? TL.Fma(l[r], alpha[r], sum) : sum;
                            m[r] = mNew;
                            seen[r] = true;
                            cLo = Math.Min(cLo, from); cHi = Math.Max(cHi, to);
                        }

                        if (cHi <= cLo) continue;
                        // V's rows the tile's weights reach (in place, or loaded at their places in the tile's buffer).
                        ReadOnlySpan<TLane> V;
                        long vBase;
                        if (direct) { V = MemoryMarshal.Cast<byte, TLane>(v.AsSpan()); vBase = kHead + (long)t0 * d; }
                        else { Operand<TN, TLane>.Of(v, kHead + (long)(t0 + cLo) * d, (cHi - cLo) * d, vb, cLo * d); V = vb; vBase = 0; }

                        // The context, two rows at a time: acc = alpha·acc + Σ_c p_c·V[c] (weights outside a row's keys are 0).
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            Context(s.AsSpan(r * bc, bc), s.AsSpan(r1 * bc, bc), alpha[r], alpha[r1], V, vBase, cLo, cHi, d,
                                acc.AsSpan(r * d, d), acc.AsSpan(r1 * d, d), r1 == r);
                        }
                    }

                    // O's rows: in place (lanes), or the block's rows stored as the type at once.
                    for (int r = 0; r < rows; r++)
                    {
                        var orow = acc.AsSpan(r * d, d);
                        TLane inv = seen[r] ? TL.Div(one, l[r]) : TL.Zero;
                        RowMath<TLane, TL>.Scale(orow, TL.Zero, inv, O.IsEmpty ? orow : O.Slice((int)(qHead + (long)(i0 + r) * d), d));
                    }

                    if (O.IsEmpty) TN.Store(acc.AsSpan(0, rows * d), o, (int)(qHead + (long)i0 * d));
                }
            }
            finally
            {
                pool.Return(s); pool.Return(acc);
                if (qb is not null) pool.Return(qb);
                if (ktb is not null) pool.Return(ktb);
                if (vb is not null) pool.Return(vb);
                if (kr is not null) pool.Return(kr);
            }
        }

        // The backward's rows and keys passes (AttentionBackRows.cs) take the same two routines.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ScoresOf(ReadOnlySpan<TLane> q0, ReadOnlySpan<TLane> q1, ReadOnlySpan<TLane> kt, int bc, int ng, int d, Span<TLane> s0, Span<TLane> s1) =>
            Scores(q0, q1, kt, bc, ng, d, s0, s1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ContextOf(ReadOnlySpan<TLane> p0, ReadOnlySpan<TLane> p1, TLane alpha0, TLane alpha1, ReadOnlySpan<TLane> V, long vBase, int cLo, int cHi, int d,
            Span<TLane> o0, Span<TLane> o1, bool single) => Context(p0, p1, alpha0, alpha1, V, vBase, cLo, cHi, d, o0, o1, single);

        // s0[c], s1[c] = q0 · Kᵀ[·][c], q1 · Kᵀ[·][c] for c < ng (ng whole groups of four vectors).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Scores(ReadOnlySpan<TLane> q0, ReadOnlySpan<TLane> q1, ReadOnlySpan<TLane> kt, int bc, int ng, int d, Span<TLane> s0, Span<TLane> s1)
        {
            int w = Vector<TLane>.Count;
            ref TLane rk = ref MemoryMarshal.GetReference(kt);
            ref TLane rq0 = ref MemoryMarshal.GetReference(q0);
            ref TLane rq1 = ref MemoryMarshal.GetReference(q1);
            ref TLane r0 = ref MemoryMarshal.GetReference(s0);
            ref TLane r1 = ref MemoryMarshal.GetReference(s1);
            for (int g = 0; g < ng; g += 4 * w)
            {
                Vector<TLane> a0 = Vector<TLane>.Zero, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                for (int t = 0; t < d; t++)
                {
                    nuint at = (nuint)(t * bc + g);
                    var k0 = Vector.LoadUnsafe(ref rk, at); var k1 = Vector.LoadUnsafe(ref rk, at + (nuint)w);
                    var k2 = Vector.LoadUnsafe(ref rk, at + (nuint)(2 * w)); var k3 = Vector.LoadUnsafe(ref rk, at + (nuint)(3 * w));
                    var x0 = new Vector<TLane>(Unsafe.Add(ref rq0, t));
                    var x1 = new Vector<TLane>(Unsafe.Add(ref rq1, t));
                    a0 = TL.Fma(x0, k0, a0); a1 = TL.Fma(x0, k1, a1); a2 = TL.Fma(x0, k2, a2); a3 = TL.Fma(x0, k3, a3);
                    b0 = TL.Fma(x1, k0, b0); b1 = TL.Fma(x1, k1, b1); b2 = TL.Fma(x1, k2, b2); b3 = TL.Fma(x1, k3, b3);
                }

                a0.StoreUnsafe(ref r0, (nuint)g); a1.StoreUnsafe(ref r0, (nuint)(g + w)); a2.StoreUnsafe(ref r0, (nuint)(g + 2 * w)); a3.StoreUnsafe(ref r0, (nuint)(g + 3 * w));
                b0.StoreUnsafe(ref r1, (nuint)g); b1.StoreUnsafe(ref r1, (nuint)(g + w)); b2.StoreUnsafe(ref r1, (nuint)(g + 2 * w)); b3.StoreUnsafe(ref r1, (nuint)(g + 3 * w));
            }
        }

        // o0 = alpha0·o0 + Σ_{c in [cLo, cHi)} p0[c]·V[c], the same for o1 (one row when single): four dimension vectors at a time.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Context(ReadOnlySpan<TLane> p0, ReadOnlySpan<TLane> p1, TLane alpha0, TLane alpha1, ReadOnlySpan<TLane> V, long vBase, int cLo, int cHi, int d,
            Span<TLane> o0, Span<TLane> o1, bool single)
        {
            int w = Vector<TLane>.Count, t = 0;
            ref TLane rv = ref MemoryMarshal.GetReference(V);
            ref TLane rp0 = ref MemoryMarshal.GetReference(p0);
            ref TLane rp1 = ref MemoryMarshal.GetReference(p1);
            ref TLane ro0 = ref MemoryMarshal.GetReference(o0);
            ref TLane ro1 = ref MemoryMarshal.GetReference(o1);
            var al0 = new Vector<TLane>(alpha0);
            var al1 = new Vector<TLane>(alpha1);
            for (; t + 4 * w <= d; t += 4 * w)
            {
                nuint tt = (nuint)t;
                var a0 = Vector.LoadUnsafe(ref ro0, tt) * al0; var a1 = Vector.LoadUnsafe(ref ro0, tt + (nuint)w) * al0;
                var a2 = Vector.LoadUnsafe(ref ro0, tt + (nuint)(2 * w)) * al0; var a3 = Vector.LoadUnsafe(ref ro0, tt + (nuint)(3 * w)) * al0;
                var b0 = Vector.LoadUnsafe(ref ro1, tt) * al1; var b1 = Vector.LoadUnsafe(ref ro1, tt + (nuint)w) * al1;
                var b2 = Vector.LoadUnsafe(ref ro1, tt + (nuint)(2 * w)) * al1; var b3 = Vector.LoadUnsafe(ref ro1, tt + (nuint)(3 * w)) * al1;
                for (int c = cLo; c < cHi; c++)
                {
                    nuint at = (nuint)(vBase + (long)c * d + t);
                    var v0 = Vector.LoadUnsafe(ref rv, at); var v1 = Vector.LoadUnsafe(ref rv, at + (nuint)w);
                    var v2 = Vector.LoadUnsafe(ref rv, at + (nuint)(2 * w)); var v3 = Vector.LoadUnsafe(ref rv, at + (nuint)(3 * w));
                    var x0 = new Vector<TLane>(Unsafe.Add(ref rp0, c));
                    var x1 = new Vector<TLane>(Unsafe.Add(ref rp1, c));
                    a0 = TL.Fma(x0, v0, a0); a1 = TL.Fma(x0, v1, a1); a2 = TL.Fma(x0, v2, a2); a3 = TL.Fma(x0, v3, a3);
                    b0 = TL.Fma(x1, v0, b0); b1 = TL.Fma(x1, v1, b1); b2 = TL.Fma(x1, v2, b2); b3 = TL.Fma(x1, v3, b3);
                }

                a0.StoreUnsafe(ref ro0, tt); a1.StoreUnsafe(ref ro0, tt + (nuint)w); a2.StoreUnsafe(ref ro0, tt + (nuint)(2 * w)); a3.StoreUnsafe(ref ro0, tt + (nuint)(3 * w));
                if (!single) { b0.StoreUnsafe(ref ro1, tt); b1.StoreUnsafe(ref ro1, tt + (nuint)w); b2.StoreUnsafe(ref ro1, tt + (nuint)(2 * w)); b3.StoreUnsafe(ref ro1, tt + (nuint)(3 * w)); }
            }

            // Single vectors, then the dimensions past them value by value.
            for (; t + w <= d; t += w)
            {
                nuint tt = (nuint)t;
                var a0 = Vector.LoadUnsafe(ref ro0, tt) * al0;
                var b0 = Vector.LoadUnsafe(ref ro1, tt) * al1;
                for (int c = cLo; c < cHi; c++)
                {
                    var v0 = Vector.LoadUnsafe(ref rv, (nuint)(vBase + (long)c * d + t));
                    a0 = TL.Fma(new Vector<TLane>(Unsafe.Add(ref rp0, c)), v0, a0);
                    b0 = TL.Fma(new Vector<TLane>(Unsafe.Add(ref rp1, c)), v0, b0);
                }

                a0.StoreUnsafe(ref ro0, tt);
                if (!single) b0.StoreUnsafe(ref ro1, tt);
            }

            for (; t < d; t++)
            {
                TLane x = TL.Mul(Unsafe.Add(ref ro0, t), alpha0), y = TL.Mul(Unsafe.Add(ref ro1, t), alpha1);
                for (int c = cLo; c < cHi; c++)
                {
                    TLane vv = Unsafe.Add(ref rv, (nint)(vBase + (long)c * d + t));
                    x = TL.Fma(Unsafe.Add(ref rp0, c), vv, x);
                    y = TL.Fma(Unsafe.Add(ref rp1, c), vv, y);
                }

                Unsafe.Add(ref ro0, t) = x;
                if (!single) Unsafe.Add(ref ro1, t) = y;
            }
        }
    }
}
