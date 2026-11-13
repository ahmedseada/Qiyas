// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The fused attention rows on the matrix units (CUDA's Ptx.AttentionMma.cs, on cooperative matrices): Q, K and V in the
// unit's 16-bit kind (exact: the float16, bfloat16 and 8-bit float formats' values are the kind's), the scores and the
// context accumulated in f32. A workgroup of four subgroups per (head, 4·M query rows), each subgroup M rows (the unit's
// M); the keys walked in tiles of 32 through workgroup memory (K, then V in the same space).
//
//   S = Q·Kᵀ        Q's fragments loaded once (from the subgroup's staging); K's from the tile, read by columns (Kᵀ)
//   softmax online  the scores stored to the subgroup's staging; each row's invocations (subgroup / M of them) take its
//                   columns in parts: the row's max and sum joined from the parts in order, the running max, sum and
//                   the context rescaled as the f32 fused rows do (Fn.Exp, the same masks)
//   O += P·V        P as two slices of the kind (hi = P rounded, lo = P − hi rounded: P to 22 bits in f16, 16 in bf16, so
//                   the context keeps the f32 lane's rounding), [hi | lo] against [V; V] on the units into a tile's
//                   context; the tile's context stored to the staging and joined into each row's context, held by the
//                   row's invocations in registers (a cooperative matrix's rows cannot be scaled one by one).
//
// Loads decode value by value (Codec.UnpackAt) into the kind's bits, two values a 32-bit word of workgroup memory. O is
// lane values (f32) stored value by value, as AttentionRows'.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Subgroups a units attention workgroup holds, and the keys a tile of it.</summary>
    public const int UnitsAttnWarps = 4, UnitsAttnKeys = 32;

    /// <summary>The unit's 16-bit kind whose values hold a format's exactly (null: none, the FMA rows).</summary>
    public static string? AttentionKind(Fmt f) => f.Name switch
    {
        "float16" or "float8_e4m3" or "float8_e5m2" => "f16",
        "bfloat16" => "bf16",
        _ => null,
    };

    // A 32-bit word of the K/V tile's row: dCap halves and 4 words of padding.
    private static int UnitsKvStride(int dCap) => dCap / 2 + 4;

    /// <summary>Workgroup bytes of the units attention rows: the K/V tile, each subgroup's staging (its rows' Q, scores,
    /// P's two slices or a tile's context), the row parts' joins.</summary>
    public static int AttentionUnitsShared(CooperativeMatrix unit, int dCap, int subgroup) =>
        4 * (UnitsAttnKeys * UnitsKvStride(dCap) + UnitsAttnWarps * unit.M * Math.Max(2 * UnitsAttnKeys, dCap) + UnitsAttnWarps * subgroup);

    /// <summary>Whether the units attention runs with this unit, head capacity and subgroup size (whole fragments, a row's
    /// invocations a whole share of the subgroup).</summary>
    public static bool AttentionUnitsFit(CooperativeMatrix unit, int dCap, int subgroup) =>
        unit.M <= subgroup && subgroup % unit.M == 0 && dCap % unit.K == 0 && dCap % unit.N == 0 && UnitsAttnKeys % unit.N == 0 && UnitsAttnKeys % unit.K == 0
        && UnitsAttnKeys % (subgroup / unit.M) == 0 && dCap % (subgroup / unit.M) == 0 && (UnitsAttnKeys / (subgroup / unit.M)) % 2 == 0;

    /// <summary>The fused rows on the units. Arguments: q, k, v, o (lane values, f32), limits, starts, ranges (each row
    /// block's keys [lo, hi)), lq, lk, d, scale (f32); workgroups (⌈lq / (4·M)⌉, heads).</summary>
    public static SpvKernel AttentionRowsUnits(Vulkan vk, Fmt f, CooperativeMatrix unit, int dCap)
    {
        int sg = vk.SubgroupSize, threads = UnitsAttnWarps * sg, rm = unit.M, br = UnitsAttnWarps * rm, bc = UnitsAttnKeys;
        if (!AttentionUnitsFit(unit, dCap, sg)) throw new NotSupportedException($"vulkan: attention on the units {unit} at d {dCap} with subgroups of {sg}");
        int tpr = sg / rm, cpt = bc / tpr, opt = dCap / tpr, kvStride = UnitsKvStride(dCap), region = rm * Math.Max(2 * bc, dCap);
        K ek = UnitKind(unit.A);
        var b = new Spv($"attention_rows_units_{f.Name}_{unit.A}_{unit.M}x{unit.N}x{unit.K}_d{dCap}", threads, vk.Controls);
        V q = b.Arg(0), kp = b.Arg(1), vp = b.Arg(2), o = b.Arg(3), lim = b.Arg(4), starts = b.Arg(5), ranges = b.Arg(6);
        V lq = b.Arg(7, K.U32), lk = b.Arg(8, K.U32), d = b.Arg(9, K.U32), scale = b.Arg(10, K.F32);
        var tid = b.LocalIdX;
        var warp = b.SubgroupId;
        var lane = tid - warp * (uint)sg;
        var qb = b.GroupX;
        var head = b.GroupY;
        var kv = b.Shared(K.U32, bc * kvStride);
        var stage = b.Shared(K.U32, UnitsAttnWarps * region);
        var joins = b.Shared(K.F32, threads);
        var mine = warp * (uint)region;                                          // this subgroup's staging
        var zero = b.F(0);
        var negInf = b.F(double.NegativeInfinity);

        // This invocation's row (rows of the subgroup: tpr invocations each) and its part of the row's columns.
        var r = lane / (uint)tpr;
        var part = lane - r * (uint)tpr;
        var i = qb * (uint)br + warp * (uint)rm + r;
        var valid = i < lq;
        var clamped = b.Select(valid, i, lq - 1);
        var limit = b.Select(valid, b.Min(b.Load(K.U32, b.At(lim, clamped, 4)), lk), b.U(0));
        var start = b.Load(K.U32, b.At(starts, clamped, 4));
        var qHead = head * lq * d;
        var kHead = head * lk * d;

        // Value (row, dim) of a tensor stored as f, as the kind's bits (0 past the rows or d).
        V Bits(V src, V rowBase, V row, V rows, V dim)
        {
            var inside = (row < rows) & (dim < d);
            var v = Codec.UnpackAt(b, f, src, rowBase + b.Select(inside, row * d + dim, b.U(0)), Lane.F32);
            var bits = ek == K.F16 ? Codec.FloatToHalf(b, v) : b.Bits(v) >> 16;
            return b.Select(inside, bits, b.U(0));
        }

        // Rows [first, first + count) of K or V (of `rows`) into the tile, two values a word, by the workgroup.
        void LoadTile(V src, V first, V rows)
        {
            int words = bc * dCap / 2;
            for (int it = 0; it < (words + threads - 1) / threads; it++)
            {
                var w = tid + it * threads;
                b.If(w < words, () =>
                {
                    var row = w / (uint)(dCap / 2);
                    var col = (w - row * (uint)(dCap / 2)) * 2;
                    var g = first + row;
                    kv[row * (uint)kvStride + col / 2] = Bits(src, kHead, g, rows, col) | Bits(src, kHead, g, rows, col + 1) << 16;
                });
            }
        }

        // Q's rows of this subgroup into its staging (rm × dCap halves), then its fragments.
        var ta = b.Coop(ek, rm, unit.K, Use.A);
        var tb = b.Coop(ek, unit.K, unit.N, Use.B);
        var tc = b.Coop(K.F32, rm, unit.N, Use.Accumulator);
        int qWords = rm * dCap / 2;
        for (int it = 0; it < (qWords + sg - 1) / sg; it++)
        {
            var w = lane + it * sg;
            b.If(w < qWords, () =>
            {
                var row = w / (uint)(dCap / 2);
                var col = (w - row * (uint)(dCap / 2)) * 2;
                var g = qb * (uint)br + warp * (uint)rm + row;
                stage[mine + w] = Bits(q, qHead, g, lq, col) | Bits(q, qHead, g, lq, col + 1) << 16;
            });
        }

        b.SubgroupBarrier();
        var qf = new M[dCap / unit.K];
        for (int dc = 0; dc < qf.Length; dc++) qf[dc] = b.Load(ta, stage, mine + (uint)(dc * unit.K / 2), b.U(dCap / 2));
        b.SubgroupBarrier();

        var m = b.Local(negInf);
        var l = b.Local(zero);
        var ov = new Var[opt];
        for (int j = 0; j < opt; j++) ov[j] = b.Local(zero);
        var lo = b.Load(K.U32, b.At(ranges, qb * 2, 4));
        var hi = b.Load(K.U32, b.At(ranges, qb * 2 + 1, 4));
        var tile0 = lo / (double)bc * bc;
        int sRegion = 0, pRegion = rm * bc;                                     // the scores (f32) and P's slices (halves)

        b.For(tile0, hi, b.U(bc), t0 =>
        {
            b.Barrier();
            LoadTile(kp, t0, lk);
            b.Barrier();

            // S = Q·Kᵀ: bc / N blocks of keys, each over d by the units; into the staging as f32.
            for (int jb = 0; jb < bc / unit.N; jb++)
            {
                var acc = b.Fill(tc, 0);
                for (int dc = 0; dc < dCap / unit.K; dc++)
                {
                    var kf = b.Load(tb, kv, b.U(jb * unit.N * kvStride + dc * unit.K / 2), b.U(kvStride), columnMajor: true);
                    acc = b.MulAdd(qf[dc], kf, acc);
                }

                b.Store(acc, stage, mine + (uint)(sRegion + jb * unit.N), b.U(bc));
            }

            b.SubgroupBarrier();

            // The row's scores of this invocation's columns: masked, the row's max and sum by the parts in order.
            var s = new V[cpt];
            var ok = new V[cpt];
            var localMax = negInf;
            for (int c = 0; c < cpt; c++)
            {
                var col = part * (uint)cpt + (uint)c;
                var key = t0 + col;
                ok[c] = valid & (key >= start) & (key < limit);
                s[c] = b.Bitcast(stage[mine + (uint)sRegion + r * (uint)bc + col], K.F32) * scale;
                localMax = b.Select(ok[c], b.Max(localMax, s[c]), localMax);
            }

            joins[tid] = localMax;
            b.SubgroupBarrier();
            var rowBase = tid - part;
            var rowMax = joins[rowBase];
            for (int p = 1; p < tpr; p++) rowMax = b.Max(rowMax, joins[rowBase + (uint)p]);
            b.SubgroupBarrier();
            var mNew = b.Max(m.V, rowMax);
            var mUse = b.Select(mNew == double.NegativeInfinity, zero, mNew);
            var alpha = Fn.Exp(b, m.V - mUse);
            var localSum = zero;
            var pv = new V[cpt];
            for (int c = 0; c < cpt; c++)
            {
                pv[c] = b.Select(ok[c], Fn.Exp(b, s[c] - mUse), zero);
                localSum = localSum + pv[c];
            }

            joins[tid] = localSum;
            b.SubgroupBarrier();
            var rowSum = joins[rowBase];
            for (int p = 1; p < tpr; p++) rowSum = rowSum + joins[rowBase + (uint)p];
            b.SubgroupBarrier();
            l.V = b.Fma(l.V, alpha, rowSum);
            m.V = mNew;

            // P's two slices of the kind: hi = P rounded, lo = (P − hi) rounded, [hi | lo] a row of 2·bc halves.
            for (int c = 0; c < cpt; c += 2)
            {
                var col = part * (uint)cpt + (uint)c;
                var (h0, l0) = HiLo(b, ek, pv[c]);
                var (h1, l1) = HiLo(b, ek, pv[c + 1]);
                stage[mine + (uint)pRegion + r * (uint)bc + col / 2] = h0 | h1 << 16;
                stage[mine + (uint)pRegion + r * (uint)bc + (uint)(bc / 2) + col / 2] = l0 | l1 << 16;
            }

            b.SubgroupBarrier();
            b.Barrier();
            LoadTile(vp, t0, lk);
            b.Barrier();

            // The tile's context [hi | lo]·[V; V] by the units, each dimension block into its accumulator.
            var tcs = new M[dCap / unit.N];
            for (int dc = 0; dc < tcs.Length; dc++)
            {
                var acc = b.Fill(tc, 0);
                for (int kc = 0; kc < 2 * bc / unit.K; kc++)
                {
                    var pf = b.Load(ta, stage, mine + (uint)(pRegion + kc * unit.K / 2), b.U(bc));
                    var vf = b.Load(tb, kv, b.U(kc % (bc / unit.K) * unit.K * kvStride + dc * unit.N / 2), b.U(kvStride));
                    acc = b.MulAdd(pf, vf, acc);
                }

                tcs[dc] = acc;
            }

            b.SubgroupBarrier();
            for (int dc = 0; dc < tcs.Length; dc++) b.Store(tcs[dc], stage, mine + (uint)(dc * unit.N), b.U(dCap));
            b.SubgroupBarrier();
            for (int j = 0; j < opt; j++)
            {
                var col = part * (uint)opt + (uint)j;
                ov[j].V = b.Fma(ov[j].V, alpha, b.Bitcast(stage[mine + r * (uint)dCap + col], K.F32));
            }

            b.SubgroupBarrier();
        });

        var inv = b.Select(l.V > 0, b.F(1) / l.V, zero);
        for (int j = 0; j < opt; j++)
        {
            var dim = part * (uint)opt + (uint)j;
            var value = ov[j].V * inv;
            b.If(valid & (dim < d), () => b.Store(b.At(o, qHead + i * d + dim, 4), value));
        }

        return b.Build();
    }

    // p's two slices in the kind: hi = p rounded to nearest even, lo = (p − hi) rounded (bits in the low 16 of a u32).
    private static (V Hi, V Lo) HiLo(Spv b, K kind, V p)
    {
        if (kind == K.F16)
        {
            var hiBits = Codec.FloatToHalf(b, p);
            var loBits = Codec.FloatToHalf(b, p - Codec.HalfToFloat(b, hiBits));
            return (hiBits, loBits);
        }

        V Bf16(V x)
        {
            var bits = b.Bits(x);
            return (bits + 0x7FFFUL + ((bits >> 16) & 1UL)) >> 16;
        }

        var h = Bf16(p);
        var lo = Bf16(p - b.Bitcast(h << 16, K.F32));
        return (h, lo);
    }
}
