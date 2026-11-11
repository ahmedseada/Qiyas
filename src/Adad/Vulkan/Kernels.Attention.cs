// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention on Vulkan (Attention.cs; CUDA's Ptx.AttentionRows.cs, Ptx.Attention.cs, Ptx.AttentionBackward.cs), in the
// row family's lane (f32, f64 for float64). Q, K, V and dO are read where they are stored, decoded on their load into
// workgroup memory; the outputs are lane values (the type's own storage for float32 and float64, else lane storage the
// host stores as the type once after: no sub-word stores).
//
//   AttentionRows      the fused rows (flash): a workgroup per (Br query rows, head); the rows' queries in workgroup
//                      memory once; the keys walked in tiles of Bc (K, then V in the same space); the scores, the online
//                      max and sum and the context without the scores leaving the workgroup. A row's tpr = 256 / Br
//                      invocations share it: each scores Bc / tpr keys of the tile and owns d / tpr of the row's output
//                      dimensions (sub, sub + tpr, …). The row's max and sum: its invocations' values in workgroup memory,
//                      each invocation joining them in the same order (the same result for all of them).
//   AttentionDecode    a workgroup per (head, query); the query in workgroup memory; the keys in chunks of 256, an
//                      invocation a key: its score, the chunk's max and sum by the workgroup tree with the running
//                      softmax rescaled, then an invocation per output dimension accumulating p_j · v_j over the chunk.
//   AttentionBackRows  (dQ) a workgroup per (Br query rows, head): pass 1 the forward itself → the row's m, 1/l and
//                      D = dO · O; pass 2 over the same key tiles: P = e^(s − m)/l, dP = dO·Vᵀ, dS = scale·P·(dP − D),
//                      dQ += dS·K. m, 1/l, D stored per row.
//   AttentionBackKeys  (dK dV) a workgroup per (Bc keys, head), the queries walked in tiles (those whose keys meet the
//                      workgroup's): P and dS recomputed from the rows' m, 1/l, D; dV += Pᵀ·dO, dK += dSᵀ·Q in registers.
// No atomics: every output value is written by one invocation, the same result every run. The mask: key j of query i
// is seen when start(i) ≤ j < limit(i) (Attention.cs); a row that sees no key gives O = 0 and no gradient.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int AttnThreads = 256, DecodeThreads = 256, FusedHeadMax = 128;

    /// <summary>Rows of queries and keys a workgroup's tile holds: (32, 32) in f32, (16, 16) in f64.</summary>
    public static (int Br, int Bc) AttentionTile(Lane lane) => lane.Wide ? (16, 16) : (32, 32);

    /// <summary>The fused kernels' head-dimension capacity: 32, 64 or 128.</summary>
    public static int AttentionCap(int d) => d <= 32 ? 32 : d <= 64 ? 64 : 128;

    // A tile's row in workgroup memory: dCap values and one more (consecutive rows on other banks).
    private static int AttnStride(int dCap) => dCap + 1;

    /// <summary>Workgroup bytes of the fused rows: the queries, the key/value tile, the tile's weights, two reductions.</summary>
    public static int AttentionRowsShared(Lane lane, int dCap)
    {
        var (br, bc) = AttentionTile(lane);
        return ((br + bc) * AttnStride(dCap) + br * (bc + 1) + 2 * AttnThreads) * lane.Size;
    }

    /// <summary>Workgroup bytes of the backward's rows kernel: Q and dO of the rows, the K and V tiles, dS, two reductions.</summary>
    public static int AttentionBackRowsShared(Lane lane, int dCap)
    {
        var (br, bc) = AttentionTile(lane);
        return ((2 * br + 2 * bc) * AttnStride(dCap) + br * (bc + 1) + 2 * AttnThreads) * lane.Size;
    }

    /// <summary>Workgroup bytes of the backward's keys kernel: K and V of the keys, the Q and dO tiles, P and dS, the rows'
    /// statistics, limits and starts.</summary>
    public static int AttentionBackKeysShared(Lane lane, int dCap)
    {
        var (br, bc) = AttentionTile(lane);
        return ((2 * bc + 2 * br) * AttnStride(dCap) + 2 * bc * (br + 1) + 3 * br) * lane.Size + 2 * br * 4;
    }

    /// <summary>Workgroup bytes of the decode kernel: the query, the chunk's weights, two reductions.</summary>
    public static int DecodeShared(Lane lane, int dims) => ((dims + DecodeThreads - 1) / DecodeThreads * DecodeThreads + 3 * DecodeThreads) * lane.Size;

    // The pieces the fused kernels share: tile loads into workgroup memory, a row's reduction, dots, accumulation.
    private sealed class AttnPieces(Spv b, Lane lane, int dCap, V tid, V d)
    {
        private readonly K k = Kinds.Of(lane);
        public readonly int Stride = AttnStride(dCap);

        public V Zero => b.C(k, 0);

        /// <summary>Rows [first, first + count) of a tensor stored as f (rows of d values from value `head`; rows past
        /// rowLimit and dimensions past d as 0) into dst[row · Stride + column]. An invocation's column is fixed
        /// (tid mod dCap), its rows step by 256 / dCap.</summary>
        public void LoadTile(Arr dst, int count, V head, V first, V rowLimit, V src, Fmt f)
        {
            int perRow = AttnThreads / dCap, logCap = System.Numerics.BitOperations.Log2((uint)dCap);
            var col = tid & (ulong)(dCap - 1);
            var row0 = tid >> logCap;
            for (int l = 0; l < count / perRow; l++)
            {
                var row = row0 + b.U(l * perRow);
                var g = first + row;
                var inside = (g < rowLimit) & (col < d);
                var index = head + b.Select(inside, g * d + col, b.U(0));
                dst[row * b.U(Stride) + col] = b.Select(inside, Codec.UnpackAt(b, f, src, index, lane), Zero);
            }
        }

        /// <summary>The join of v over a group's `width` invocations (tid = group · width + sub): each invocation's value in
        /// scratch, then every invocation of the group joins them in order. Uniform control flow; ends with a barrier
        /// (the scratch free for the next use).</summary>
        public V GroupJoin(Arr scratch, V group, int width, V v, Func<V, V, V> join)
        {
            scratch[tid] = v;
            b.Barrier();
            var at = group * b.U(width);
            var acc = scratch[at];
            for (int t = 1; t < width; t++) acc = join(acc, scratch[at + b.U(t)]);
            b.Barrier();
            return acc;
        }

        /// <summary>dst[jj] = Σ over the padded row (dCap values, in order) of rowA's values · rowsB[jj]'s values.</summary>
        public void Dots(Var[] dst, Arr a, V rowA, Arr bArr, V[] rowsB)
        {
            foreach (var x in dst) x.V = Zero;
            b.For(b.U(0), b.U(dCap), t =>
            {
                var av = a[rowA + t];
                for (int jj = 0; jj < dst.Length; jj++) dst[jj].V = b.Fma(av, bArr[rowsB[jj] + t], dst[jj].V);
            });
        }

        /// <summary>acc[j] += Σ_c w[wRow + c] · rows[c · Stride + sub + width · j], c over count, in order.</summary>
        public void Accumulate(Var[] acc, int count, Arr w, V wRow, Arr rows, V sub, int width)
        {
            b.For(b.U(0), b.U(count), c =>
            {
                var wc = w[wRow + c];
                var at = c * b.U(Stride) + sub;
                for (int j = 0; j < acc.Length; j++) acc[j].V = b.Fma(wc, rows[at + b.U(width * j)], acc[j].V);
            });
        }

        public Var[] Zeros(int n)
        {
            var a = new Var[n];
            for (int i = 0; i < n; i++) a[i] = b.Local(Zero);
            return a;
        }

        /// <summary>Values acc[j] of a row (dimension sub + width · j) to dst[row · d + dimension] as lane values (where the
        /// row is valid and the dimension below d).</summary>
        public void StoreRow(Var[] acc, V dst, V rowBase, V sub, int width, V valid, V? scale = null)
        {
            for (int j = 0; j < acc.Length; j++)
            {
                var dim = sub + b.U(width * j);
                var value = scale is { } s ? acc[j].V * s : acc[j].V;
                b.If(valid & (dim < d), () => b.Store(b.At(dst, rowBase + dim, lane.Size), value));
            }
        }
    }

    // This invocation's row and its keys: the row's query index i (valid below lq), [start, limit) (limit 0 past the queries).
    private static (V R, V Sub, V I, V Valid, V Start, V Limit) AttnRow(Spv b, V tid, int br, int tpr, V qb, V lq, V lk, V lim, V starts)
    {
        var r = tid >> System.Numerics.BitOperations.Log2((uint)tpr);
        var sub = tid & (ulong)(tpr - 1);
        var i = qb * b.U(br) + r;
        var valid = i < lq;
        var clamped = b.Select(valid, i, lq - b.U(1));
        var limit = b.Select(valid, b.Min(b.Load(K.U32, b.At(lim, clamped, 4)), lk), b.U(0));
        var start = b.Load(K.U32, b.At(starts, clamped, 4));
        return (r, sub, i, valid, start, limit);
    }

    /// <summary>The fused rows. Arguments: q, k, v, o (lane values), limits, starts, ranges (each row block's keys
    /// [lo, hi)), lq, lk, d, scale (a lane slot); workgroups (⌈lq / Br⌉, heads).</summary>
    public static SpvKernel AttentionRows(Vulkan vk, Fmt f, Lane lane, int dCap)
    {
        var k = Kinds.Of(lane);
        var (Br, Bc) = AttentionTile(lane);
        int tpr = AttnThreads / Br, spt = Bc / tpr, nacc = dCap / tpr;
        var b = new Spv($"attention_rows_{f.Name}_{lane.Name}_d{dCap}", AttnThreads, vk.Controls);
        var q = b.Arg(0);
        var kp = b.Arg(1);
        var vp = b.Arg(2);
        var o = b.Arg(3);
        var lim = b.Arg(4);
        var starts = b.Arg(5);
        var ranges = b.Arg(6);
        var lq = b.Arg(7, K.U32);
        var lk = b.Arg(8, K.U32);
        var d = b.Arg(9, K.U32);
        var scale = b.Arg(10, k);
        var tid = b.LocalIdX;
        var qb = b.GroupX;
        var h = b.GroupY;
        var P = new AttnPieces(b, lane, dCap, tid, d);
        var negInf = b.C(k, double.NegativeInfinity);
        var (r, sub, i, rowValid, start, limit) = AttnRow(b, tid, Br, tpr, qb, lq, lk, lim, starts);
        var qHead = h * lq * d;
        var kHead = h * lk * d;
        var lo = b.Load(K.U32, b.At(ranges, qb * b.U(2), 4));
        var hi = b.Load(K.U32, b.At(ranges, qb * b.U(2) + b.U(1), 4));
        var sQ = b.Shared(k, Br * P.Stride);
        var sKV = b.Shared(k, Bc * P.Stride);
        var sP = b.Shared(k, Br * (Bc + 1));
        var red1 = b.Shared(k, AttnThreads);
        var red2 = b.Shared(k, AttnThreads);
        P.LoadTile(sQ, Br, qHead, qb * b.U(Br), lq, q, f);
        b.Barrier();
        var m = b.Local(negInf);
        var l = b.Local(P.Zero);
        var acc = P.Zeros(nacc);
        var qRow = r * b.U(P.Stride);
        var pRow = r * b.U(Bc + 1);
        var c = new V[spt];
        var kRow = new V[spt];
        for (int jj = 0; jj < spt; jj++)
        {
            c[jj] = sub + b.U(tpr * jj);
            kRow[jj] = c[jj] * b.U(P.Stride);
        }

        b.For(lo, hi, b.U(Bc), kt =>
        {
            P.LoadTile(sKV, Bc, kHead, kt, lk, kp, f);
            b.Barrier();
            var s = P.Zeros(spt);
            P.Dots(s, sQ, qRow, sKV, kRow);
            // The mask, the tile's max, the new running max (0 in the exponent while no key is seen).
            var valid = new V[spt];
            var sc = new V[spt];
            var tmax = negInf;
            for (int jj = 0; jj < spt; jj++)
            {
                var key = kt + c[jj];
                valid[jj] = (key >= start) & (key < limit);
                sc[jj] = b.Select(valid[jj], s[jj].V * scale, negInf);
                tmax = b.Max(tmax, sc[jj]);
            }

            tmax = P.GroupJoin(red1, r, tpr, tmax, b.Max);
            var mNew = b.Max(m.V, tmax);
            var mUse = b.Select(mNew == double.NegativeInfinity, P.Zero, mNew);
            var alpha = Fn.Exp(b, m.V - mUse);
            var psum = P.Zero;
            for (int jj = 0; jj < spt; jj++)
            {
                var p = b.Select(valid[jj], Fn.Exp(b, sc[jj] - mUse), P.Zero);
                sP[pRow + c[jj]] = p;
                psum = psum + p;
            }

            psum = P.GroupJoin(red2, r, tpr, psum, static (x, y) => x + y);
            l.V = b.Fma(l.V, alpha, psum);
            m.V = mNew;
            P.LoadTile(sKV, Bc, kHead, kt, lk, vp, f);
            b.Barrier();
            // The context: acc = alpha·acc + Σ_c p_c · v_c (this invocation's dimensions of the row).
            foreach (var a in acc) a.V = a.V * alpha;
            P.Accumulate(acc, Bc, sP, pRow, sKV, sub, tpr);
            b.Barrier();
        });

        // O = acc / l (0 for a row that sees no key).
        var inv = b.Select(l.V > 0, b.C(k, 1) / l.V, P.Zero);
        P.StoreRow(acc, o, qHead + i * d, sub, tpr, rowValid, inv);
        return b.Build();
    }

    /// <summary>The decode kernel over caches at a position the device holds (ICache.AttentionCached): K and V stored as Fc,
    /// each row's scale (the queries' type, at head·lk + row) multiplied into its values as they are read, the scores capped.</summary>
    internal sealed record DecodeCache(Fmt Fc, bool KScales, bool VScales, bool Capped)
    {
        public string Key => $"{Fc.Name}{(KScales ? "_k" : "")}{(VScales ? "_v" : "")}{(Capped ? "_cap" : "")}";
    }

    /// <summary>The decode kernel. Arguments: q, k, v, o (lane values), limits, starts, lq, lk, d, scale (a lane slot);
    /// workgroups heads · lq. Over caches: lq is the rows a head of K and V serves (a group's query heads × lq0 queries),
    /// limits the position (an s32 at the device): row r's query i = r mod lq0 sees keys [max(0, limit − window) when a
    /// window, limit), limit = p + i + 1 held to [0, lk]; then the K and V row scales, lq0, the window, the cap (a lane slot).</summary>
    /// runs (K and V a block format, d a multiple of 8): a key's row read in runs of 8 values (Codec.UnpackRun), each
    /// value's arithmetic and the sum's order as without.
    public static SpvKernel AttentionDecode(Vulkan vk, Fmt f, Lane lane, int dims, DecodeCache? cache = null, bool runs = false)
    {
        var k = Kinds.Of(lane);
        int nd = (dims + DecodeThreads - 1) / DecodeThreads, cap = nd * DecodeThreads;
        var fk = cache?.Fc ?? f;                                                 // K and V's stored format
        var b = new Spv($"attention_decode_{f.Name}_{lane.Name}_d{cap}{(cache is null ? "" : "_at_" + cache.Key)}{(runs ? "_runs" : "")}", DecodeThreads, vk.Controls);
        var q = b.Arg(0);
        var kp = b.Arg(1);
        var vp = b.Arg(2);
        var o = b.Arg(3);
        var lim = b.Arg(4);
        var starts = b.Arg(5);
        var lq = b.Arg(6, K.U32);
        var lk = b.Arg(7, K.U32);
        var d = b.Arg(8, K.U32);
        var scale = b.Arg(9, k);
        var tid = b.LocalIdX;
        var hq = b.GroupX;
        var h = hq / lq;
        var i = hq - h * lq;
        V limit, start;
        V? kScales = null, vScales = null, capAt = null;
        if (cache is null)
        {
            limit = b.Min(b.Load(K.U32, b.At(lim, i, 4)), lk);
            start = b.Load(K.U32, b.At(starts, i, 4));
        }
        else
        {
            kScales = b.Arg(10);
            vScales = b.Arg(11);
            V lq0 = b.Arg(12, K.U32), window = b.Arg(13, K.U32);
            capAt = b.Arg(14, k);
            var seen = b.Min(b.Max(b.Load(K.S32, lim) + b.Convert(i % lq0, K.S32) + 1.0, b.S(0)), b.Convert(lk, K.S32));
            limit = b.Convert(seen, K.U32);
            start = b.Select(window == 0.0, b.U(0), b.Convert(b.Max(seen - b.Convert(window, K.S32), b.S(0)), K.U32));
        }

        var qOff = hq * d;
        var kHead = h * lk * d;
        var zero = b.C(k, 0);
        var negInf = b.C(k, double.NegativeInfinity);
        var sq = b.Shared(k, cap);
        var red1 = b.Shared(k, DecodeThreads);
        var red2 = b.Shared(k, DecodeThreads);
        var ps = b.Shared(k, DecodeThreads);
        Arr? pv = cache is { VScales: true } ? b.Shared(k, DecodeThreads) : null;      // the chunk's V row scales
        b.For(tid, b.U(cap), b.U(DecodeThreads), t => sq[t] = b.Select(t < d, Codec.UnpackAt(b, f, q, qOff + b.Select(t < d, t, b.U(0)), lane), zero));
        b.Barrier();
        var m = b.Local(negInf);
        var l = b.Local(zero);
        var acc = new Var[nd];
        for (int j = 0; j < nd; j++) acc[j] = b.Local(zero);
        b.For(start, limit, b.U(DecodeThreads), c0 =>
        {
            // This invocation's key: its score.
            var key = c0 + tid;
            var valid = key < limit;
            var row = kHead + b.Select(valid, key, b.U(0)) * d;
            var keyRow = h * lk + b.Select(valid, key, b.U(0));
            var dot = b.Local(zero);
            if (cache is { KScales: true })
            {
                var rowScale = Codec.UnpackAt(b, f, kScales!.Value, keyRow, lane);                 // the row's values: code·scale
                b.For(b.U(0), d, t => dot.V = b.Fma(sq[t], Codec.UnpackAt(b, fk, kp, row + t, lane) * rowScale, dot.V));
            }
            else if (runs)
            {
                b.For(b.U(0), d, b.U(8), t =>
                {
                    var keys = Codec.UnpackRun(b, fk, kp, row + t, 8, lane);
                    for (int r = 0; r < 8; r++) dot.V = b.Fma(sq[t + r], keys[r], dot.V);
                });
            }
            else b.For(b.U(0), d, t => dot.V = b.Fma(sq[t], Codec.UnpackAt(b, fk, kp, row + t, lane), dot.V));

            // The score: scaled, or the map's softcap (cap·tanh(scale·s / cap)).
            var score = cache is { Capped: true } ? capAt!.Value * Fn.Tanh(b, (scale * dot.V) / capAt!.Value) : dot.V * scale;
            var sj = b.Select(valid, score, negInf);
            var cmax = GroupReduce(b, red1, tid, sj, b.Max, DecodeThreads);
            var mNew = b.Max(m.V, cmax);
            var mUse = b.Select(mNew == double.NegativeInfinity, zero, mNew);
            var alpha = Fn.Exp(b, m.V - mUse);
            var p = b.Select(valid, Fn.Exp(b, sj - mUse), zero);
            ps[tid] = p;
            if (pv is { } vRows) vRows[tid] = Codec.UnpackAt(b, f, vScales!.Value, keyRow, lane);
            var csum = GroupReduce(b, red2, tid, p, static (x, y) => x + y, DecodeThreads);
            l.V = b.Fma(l.V, alpha, csum);
            m.V = mNew;
            // The context: an invocation per dimension over the chunk's keys, in order.
            var count = b.Min(limit - c0, b.U(DecodeThreads));
            for (int j = 0; j < nd; j++)
            {
                var dim = tid + b.U(DecodeThreads * j);
                var inside = dim < d;
                var col = b.Select(inside, dim, b.U(0));
                var a = acc[j];
                a.V = a.V * alpha;
                if (pv is { } rowScales) b.For(b.U(0), count, cc => a.V = b.Fma(ps[cc], Codec.UnpackAt(b, fk, vp, kHead + (c0 + cc) * d + col, lane) * rowScales[cc], a.V));
                else b.For(b.U(0), count, cc => a.V = b.Fma(ps[cc], Codec.UnpackAt(b, fk, vp, kHead + (c0 + cc) * d + col, lane), a.V));
            }

            b.Barrier();
        });

        var inv = b.Select(l.V > 0, b.C(k, 1) / l.V, zero);
        for (int j = 0; j < nd; j++)
        {
            var dim = tid + b.U(DecodeThreads * j);
            var value = acc[j].V * inv;
            b.If(dim < d, () => b.Store(b.At(o, qOff + dim, lane.Size), value));
        }

        return b.Build();
    }

    /// <summary>The backward's rows kernel (dQ; each row's m, 1/l, D into stats). Arguments: q, k, v, dO, dq (lane values),
    /// stats, limits, starts, ranges, lq, lk, d, scale (a lane slot); workgroups (⌈lq / Br⌉, heads).</summary>
    public static SpvKernel AttentionBackRows(Vulkan vk, Fmt f, Lane lane, int dCap)
    {
        var k = Kinds.Of(lane);
        var (Br, Bc) = AttentionTile(lane);
        int tpr = AttnThreads / Br, spt = Bc / tpr, nacc = dCap / tpr;
        var b = new Spv($"attention_back_rows_{f.Name}_{lane.Name}_d{dCap}", AttnThreads, vk.Controls);
        var q = b.Arg(0);
        var kp = b.Arg(1);
        var vp = b.Arg(2);
        var dO = b.Arg(3);
        var dq = b.Arg(4);
        var stats = b.Arg(5);
        var lim = b.Arg(6);
        var starts = b.Arg(7);
        var ranges = b.Arg(8);
        var lq = b.Arg(9, K.U32);
        var lk = b.Arg(10, K.U32);
        var d = b.Arg(11, K.U32);
        var scale = b.Arg(12, k);
        var tid = b.LocalIdX;
        var qb = b.GroupX;
        var h = b.GroupY;
        var P = new AttnPieces(b, lane, dCap, tid, d);
        var negInf = b.C(k, double.NegativeInfinity);
        var (r, sub, i, rowValid, start, limit) = AttnRow(b, tid, Br, tpr, qb, lq, lk, lim, starts);
        var qHead = h * lq * d;
        var kHead = h * lk * d;
        var lo = b.Load(K.U32, b.At(ranges, qb * b.U(2), 4));
        var hi = b.Load(K.U32, b.At(ranges, qb * b.U(2) + b.U(1), 4));
        var sQ = b.Shared(k, Br * P.Stride);
        var sDO = b.Shared(k, Br * P.Stride);
        var sK = b.Shared(k, Bc * P.Stride);
        var sV = b.Shared(k, Bc * P.Stride);
        var sP = b.Shared(k, Br * (Bc + 1));
        var red1 = b.Shared(k, AttnThreads);
        var red2 = b.Shared(k, AttnThreads);
        P.LoadTile(sQ, Br, qHead, qb * b.U(Br), lq, q, f);
        P.LoadTile(sDO, Br, qHead, qb * b.U(Br), lq, dO, f);
        b.Barrier();                                                                   // dO is read after the tiles (a row block that sees no key too)
        var qRow = r * b.U(P.Stride);
        var pRow = r * b.U(Bc + 1);
        var c = new V[spt];
        var kRow = new V[spt];
        for (int jj = 0; jj < spt; jj++)
        {
            c[jj] = sub + b.U(tpr * jj);
            kRow[jj] = c[jj] * b.U(P.Stride);
        }

        // The tile's scores with the scale and the mask (masked: −∞), valid[jj] the keys this row sees.
        (V[] S, V[] Valid) Scores(V kt)
        {
            var s = P.Zeros(spt);
            P.Dots(s, sQ, qRow, sK, kRow);
            var sc = new V[spt];
            var valid = new V[spt];
            for (int jj = 0; jj < spt; jj++)
            {
                var key = kt + c[jj];
                valid[jj] = (key >= start) & (key < limit);
                sc[jj] = b.Select(valid[jj], s[jj].V * scale, negInf);
            }

            return (sc, valid);
        }

        // Pass 1: the forward (online softmax, the context in this invocation's dimensions).
        var m = b.Local(negInf);
        var l = b.Local(P.Zero);
        var acc = P.Zeros(nacc);
        b.For(lo, hi, b.U(Bc), kt =>
        {
            P.LoadTile(sK, Bc, kHead, kt, lk, kp, f);
            P.LoadTile(sV, Bc, kHead, kt, lk, vp, f);
            b.Barrier();
            var (sc, valid) = Scores(kt);
            var tmax = negInf;
            for (int jj = 0; jj < spt; jj++) tmax = b.Max(tmax, sc[jj]);
            tmax = P.GroupJoin(red1, r, tpr, tmax, b.Max);
            var mNew = b.Max(m.V, tmax);
            var mUse = b.Select(mNew == double.NegativeInfinity, P.Zero, mNew);
            var alpha = Fn.Exp(b, m.V - mUse);
            var psum = P.Zero;
            for (int jj = 0; jj < spt; jj++)
            {
                var p = b.Select(valid[jj], Fn.Exp(b, sc[jj] - mUse), P.Zero);
                sP[pRow + c[jj]] = p;
                psum = psum + p;
            }

            psum = P.GroupJoin(red2, r, tpr, psum, static (x, y) => x + y);
            l.V = b.Fma(l.V, alpha, psum);
            m.V = mNew;
            foreach (var a in acc) a.V = a.V * alpha;
            P.Accumulate(acc, Bc, sP, pRow, sV, sub, tpr);
            b.Barrier();
        });

        // The row's m (0 while no key: P is 0 there), 1/l (0 when none) and D = Σ_dims dO · O.
        var inv = b.Select(l.V > 0, b.C(k, 1) / l.V, P.Zero);
        var mUseF = b.Select(m.V == double.NegativeInfinity, P.Zero, m.V);
        var dsum = P.Zero;
        for (int j = 0; j < nacc; j++) dsum = b.Fma(sDO[qRow + sub + b.U(tpr * j)], acc[j].V * inv, dsum);
        dsum = P.GroupJoin(red1, r, tpr, dsum, static (x, y) => x + y);
        b.If((sub == 0) & rowValid, () =>
        {
            var at = (h * lq + i) * b.U(3);
            b.Store(b.At(stats, at, lane.Size), mUseF);
            b.Store(b.At(stats, at + b.U(1), lane.Size), inv);
            b.Store(b.At(stats, at + b.U(2), lane.Size), dsum);
        });

        // Pass 2: dS = scale·P·(dP − D) per key, dQ += dS·K.
        var dqAcc = P.Zeros(nacc);
        b.For(lo, hi, b.U(Bc), kt =>
        {
            P.LoadTile(sK, Bc, kHead, kt, lk, kp, f);
            P.LoadTile(sV, Bc, kHead, kt, lk, vp, f);
            b.Barrier();
            var (sc, valid) = Scores(kt);
            var dp = P.Zeros(spt);
            P.Dots(dp, sDO, qRow, sV, kRow);
            for (int jj = 0; jj < spt; jj++)
            {
                var p = b.Select(valid[jj], Fn.Exp(b, sc[jj] - mUseF) * inv, P.Zero);
                var w = (dp[jj].V - dsum) * p;
                sP[pRow + c[jj]] = w * scale;
            }

            b.Barrier();
            P.Accumulate(dqAcc, Bc, sP, pRow, sK, sub, tpr);
            b.Barrier();
        });

        P.StoreRow(dqAcc, dq, qHead + i * d, sub, tpr, rowValid);
        return b.Build();
    }

    /// <summary>The backward's keys kernel (dK, dV). Arguments: q, k, v, dO, stats, dk, dv (lane values), limits, starts,
    /// ranges (each key block's queries [lo, hi)), lq, lk, d, scale (a lane slot); workgroups (⌈lk / Bc⌉, heads).</summary>
    public static SpvKernel AttentionBackKeys(Vulkan vk, Fmt f, Lane lane, int dCap)
    {
        var k = Kinds.Of(lane);
        var (Br, Bc) = AttentionTile(lane);
        int tpk = AttnThreads / Bc, spt = Br / tpk, nacc = dCap / tpk;
        var b = new Spv($"attention_back_keys_{f.Name}_{lane.Name}_d{dCap}", AttnThreads, vk.Controls);
        var q = b.Arg(0);
        var kp = b.Arg(1);
        var vp = b.Arg(2);
        var dO = b.Arg(3);
        var stats = b.Arg(4);
        var dk = b.Arg(5);
        var dv = b.Arg(6);
        var lim = b.Arg(7);
        var starts = b.Arg(8);
        var ranges = b.Arg(9);
        var lq = b.Arg(10, K.U32);
        var lk = b.Arg(11, K.U32);
        var d = b.Arg(12, K.U32);
        var scale = b.Arg(13, k);
        var tid = b.LocalIdX;
        var kb = b.GroupX;
        var h = b.GroupY;
        var P = new AttnPieces(b, lane, dCap, tid, d);
        // This invocation's key and place in it.
        var ck = tid >> System.Numerics.BitOperations.Log2((uint)tpk);
        var sub = tid & (ulong)(tpk - 1);
        var key = kb * b.U(Bc) + ck;
        var keyValid = key < lk;
        var qHead = h * lq * d;
        var kHead = h * lk * d;
        var statHead = h * lq * b.U(3);
        var lo = b.Load(K.U32, b.At(ranges, kb * b.U(2), 4));
        var hi = b.Load(K.U32, b.At(ranges, kb * b.U(2) + b.U(1), 4));
        var sK = b.Shared(k, Bc * P.Stride);
        var sV = b.Shared(k, Bc * P.Stride);
        var sQ = b.Shared(k, Br * P.Stride);
        var sDO = b.Shared(k, Br * P.Stride);
        var sP = b.Shared(k, Bc * (Br + 1));
        var sS = b.Shared(k, Bc * (Br + 1));
        var sStat = b.Shared(k, 3 * Br);
        var sLim = b.Shared(K.U32, 2 * Br);
        P.LoadTile(sK, Bc, kHead, kb * b.U(Bc), lk, kp, f);
        P.LoadTile(sV, Bc, kHead, kb * b.U(Bc), lk, vp, f);
        b.Barrier();
        var kRow = ck * b.U(P.Stride);
        var pRow = ck * b.U(Br + 1);
        var rr = new V[spt];
        var qRows = new V[spt];
        for (int jj = 0; jj < spt; jj++)
        {
            rr[jj] = sub + b.U(tpk * jj);
            qRows[jj] = rr[jj] * b.U(P.Stride);
        }

        var dkAcc = P.Zeros(nacc);
        var dvAcc = P.Zeros(nacc);
        b.For(lo, hi, b.U(Br), qt =>
        {
            P.LoadTile(sQ, Br, qHead, qt, lq, q, f);
            P.LoadTile(sDO, Br, qHead, qt, lq, dO, f);
            // The tile's rows' m, 1/l, D and keys [start, limit) (0, 0 past the queries): invocations 0 .. Br − 1.
            b.If(tid < Br, () =>
            {
                var row = qt + tid;
                var rv = row < lq;
                var idx = b.Select(rv, row, b.U(0));
                var at = statHead + idx * b.U(3);
                for (int t = 0; t < 3; t++) sStat[tid * b.U(3) + b.U(t)] = b.Load(k, b.At(stats, at + b.U(t), lane.Size));
                sLim[tid * b.U(2)] = b.Select(rv, b.Min(b.Load(K.U32, b.At(lim, idx, 4)), lk), b.U(0));
                sLim[tid * b.U(2) + b.U(1)] = b.Load(K.U32, b.At(starts, idx, 4));
            });
            b.Barrier();
            // P and dS of this key for the rows this invocation takes.
            var s = P.Zeros(spt);
            var dp = P.Zeros(spt);
            b.For(b.U(0), b.U(dCap), t =>
            {
                var kv = sK[kRow + t];
                var vv = sV[kRow + t];
                for (int jj = 0; jj < spt; jj++)
                {
                    s[jj].V = b.Fma(sQ[qRows[jj] + t], kv, s[jj].V);
                    dp[jj].V = b.Fma(sDO[qRows[jj] + t], vv, dp[jj].V);
                }
            });
            for (int jj = 0; jj < spt; jj++)
            {
                var lv = sLim[rr[jj] * b.U(2)];
                var sv = sLim[rr[jj] * b.U(2) + b.U(1)];
                var pv = (key >= sv) & (key < lv);
                var mm = sStat[rr[jj] * b.U(3)];
                var il = sStat[rr[jj] * b.U(3) + b.U(1)];
                var dd = sStat[rr[jj] * b.U(3) + b.U(2)];
                var x = b.Select(pv, s[jj].V * scale - mm, P.Zero);                 // no exponent of a masked −∞ or garbage
                var p = b.Select(pv, Fn.Exp(b, x) * il, P.Zero);
                var w = (dp[jj].V - dd) * p;
                sP[pRow + rr[jj]] = p;
                sS[pRow + rr[jj]] = w * scale;
            }

            b.Barrier();
            P.Accumulate(dvAcc, Br, sP, pRow, sDO, sub, tpk);
            P.Accumulate(dkAcc, Br, sS, pRow, sQ, sub, tpk);
            b.Barrier();
        });

        P.StoreRow(dkAcc, dk, kHead + key * d, sub, tpk, keyValid);
        P.StoreRow(dvAcc, dv, kHead + key * d, sub, tpk, keyValid);
        return b.Build();
    }
}
