// Attention rows fused (flash): O for rows of queries without the Lq × Lk scores in memory. Per task a head's block of
// query rows; the keys walked in tiles: the tile's scores in the float lane (Kᵀ's tile laid so a key run is a vector:
// S[r][c..c+w) = Σ_t q[r][t]·Kᵀ[t][c..c+w), no horizontal sums), the softmax online (the running max m and sum l per
// row; the row's accumulator rescaled by exp(m_old − m_new) when the max moves), then O[r] += Σ_c p_c·V[c] along d.
// Two rows at a time in registers: the scores of a row pair over four key vectors (eight accumulators, each key vector
// loaded once for both rows), the context of a row pair over four dimension vectors (each value vector loaded once).
// The keys a row sees are [Start(i), Limit(i)) as in the composition; a block's tiles span the union of its rows',
// starting at a whole tile. Kᵀ's tiles are laid once per call (a pass over K before the tasks, item 6i): every block of
// a head reads the same tiles (before, each block of 16 rows transposed each tile again: 128 times a head at 2048).
// A narrow type is widened to its float lane's storage once (Q, K, V) and O narrowed once, as the composition does.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Qiyas;

internal static partial class Kernels
{
    private const int FusedRows = 16, FusedKeys = 64;                       // a task's query rows; a key tile (L1: 64 × d lanes, twice)

    /// <summary>O for rows of queries by the fused rows (no scores in memory); returns the lane's name.</summary>
    public static string AttentionRows<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) where TN : struct, IQiyasFloat<TN>
    {
        bool own = TN.Name is "float32" or "float64";
        var lq = ToLane<TN>(q, a.QValues);
        var lk = ToLane<TN>(k, a.KValues);
        var lv = ToLane<TN>(v, a.KValues);
        if (Wide<TN>())
        {
            var lo = own ? o : new byte[a.QValues * 8];
            var vis = new FusedVisitor(a, lq, lk, lv, lo);
            Float64.Float(ref vis);
            if (!own) FromLane<TN>(lo, o, a.QValues);
            return "f64";
        }
        else
        {
            var lo = own ? o : new byte[a.QValues * 4];
            var vis = new FusedVisitor(a, lq, lk, lv, lo);
            Float32.Float(ref vis);
            if (!own) FromLane<TN>(lo, o, a.QValues);
            return "f32";
        }
    }

    private readonly struct FusedVisitor(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) : IFloatLaneVisitor
    {
        public void Run<TS, TLane, TL>() where TS : struct, IQiyasNumber<TS, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            int blocks = (a.Lq + FusedRows - 1) / FusedRows, tiles = (a.Lk + FusedKeys - 1) / FusedKeys;
            // Kᵀ's tiles: kt[(h·tiles + j)·d·bc + t·bc + c] = K[h][j·bc + c][t], zero past Lk.
            var kt = Scratch.Get(Scratch.AttnKt, (long)a.Heads * tiles * a.D * FusedKeys * Unsafe.SizeOf<TLane>());
            Team.For(a.Heads * tiles, 1, (long)a.Heads * a.Lk * a.D, new KeyTilesBody<TLane>(a, k, kt, tiles));
            Team.For(a.Heads * blocks, 1, (long)a.Heads * a.Lq * a.Lk * a.D, new FusedBody<TLane, TL>(a, q, kt, v, o, blocks, tiles));
        }
    }

    // Kᵀ's tiles of the heads (a tile a task): each key row read once, its values written down the tile's columns.
    private readonly struct KeyTilesBody<TLane>(AttentionArgs a, byte[] k, byte[] kt, int tiles) : Team.IBody where TLane : unmanaged
    {
        public void Run(int first, int last)
        {
            int d = a.D, bc = FusedKeys;
            var K = MemoryMarshal.Cast<byte, TLane>(k.AsSpan());
            var T = MemoryMarshal.Cast<byte, TLane>(kt.AsSpan());
            for (int task = first; task < last; task++)
            {
                int h = task / tiles, t0 = task % tiles * bc, n = Math.Min(bc, a.Lk - t0);
                var tile = T.Slice(task * d * bc, d * bc);
                if (n < bc) tile.Clear();
                for (int c = 0; c < n; c++)
                {
                    var row = K.Slice((int)((long)h * a.Lk * d + (long)(t0 + c) * d), d);
                    for (int t = 0; t < d; t++) tile[t * bc + c] = row[t];
                }
            }
        }
    }

    private readonly struct FusedBody<TLane, TL>(AttentionArgs a, byte[] q, byte[] kts, byte[] v, byte[] o, int blocks, int tiles) : Team.IBody
        where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int d = a.D, w = Vector<TLane>.Count, bc = FusedKeys, group = 4 * w;
            var Q = MemoryMarshal.Cast<byte, TLane>(q.AsSpan());
            var KT = MemoryMarshal.Cast<byte, TLane>(kts.AsSpan());
            var V = MemoryMarshal.Cast<byte, TLane>(v.AsSpan());
            var O = MemoryMarshal.Cast<byte, TLane>(o.AsSpan());
            var pool = ArrayPool<TLane>.Shared;
            var s = pool.Rent(FusedRows * bc);
            var acc = pool.Rent(FusedRows * d);
            var m = new TLane[FusedRows];
            var l = new TLane[FusedRows];
            var alpha = new TLane[FusedRows];
            var seen = new bool[FusedRows];
            TLane scale = TL.From(a.Scale), one = TL.From(1);
            try
            {
                for (int task = first; task < last; task++)
                {
                    int h = task / blocks, i0 = task % blocks * FusedRows, rows = Math.Min(FusedRows, a.Lq - i0);
                    int lo = int.MaxValue, hi = 0;
                    for (int r = 0; r < rows; r++) { lo = Math.Min(lo, a.Start(i0 + r)); hi = Math.Max(hi, a.Limit(i0 + r)); }
                    acc.AsSpan(0, rows * d).Clear();
                    Array.Clear(seen);
                    long kHead = (long)h * a.Lk * d, qHead = (long)h * a.Lq * d;
                    for (int t0 = hi <= lo ? hi : lo / bc * bc; t0 < hi; t0 += bc)
                    {
                        // Kᵀ's tile (zero past Lk, so the groups past n score 0: outside every row's keys).
                        int n = Math.Min(bc, hi - t0), ng = (n + group - 1) / group * group;
                        var kt = KT.Slice((h * tiles + t0 / bc) * d * bc, d * bc);

                        // Scores, two rows at a time; then each row's online softmax over its keys of the tile.
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            Scores(Q.Slice((int)(qHead + (long)(i0 + r) * d), d), Q.Slice((int)(qHead + (long)(i0 + r1) * d), d), kt, bc, ng, d, s.AsSpan(r * bc, bc), s.AsSpan(r1 * bc, bc));
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
                        // The context, two rows at a time: acc = alpha·acc + Σ_c p_c·V[c] (weights outside a row's keys are 0).
                        for (int r = 0; r < rows; r += 2)
                        {
                            int r1 = Math.Min(r + 1, rows - 1);
                            Context(s.AsSpan(r * bc, bc), s.AsSpan(r1 * bc, bc), alpha[r], alpha[r1], V, kHead + (long)t0 * d, cLo, cHi, d,
                                acc.AsSpan(r * d, d), acc.AsSpan(r1 * d, d), r1 == r);
                        }
                    }

                    for (int r = 0; r < rows; r++)
                    {
                        var orow = acc.AsSpan(r * d, d);
                        TLane inv = seen[r] ? TL.Div(one, l[r]) : TL.Zero;
                        RowMath<TLane, TL>.Scale(orow, TL.Zero, inv, O.Slice((int)(qHead + (long)(i0 + r) * d), d));
                    }
                }
            }
            finally
            {
                pool.Return(s); pool.Return(acc);
            }
        }

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
