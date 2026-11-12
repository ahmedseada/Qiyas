// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention on Vulkan (Attention.cs, Kernels.Attention.cs), CUDA's routes on the FMA lanes: the fused rows or the
// composition for rows of queries (whichever this device measures faster; the composition past the fused kernel's head
// dimension), the fused decode kernel, the backward's fused kernels where both fit the device's workgroup memory or its
// composition. Q, K, V and dO are read where they are stored by every route; the scores and their gradients are lane
// storage (intermediates), the outputs lane values: the type's own storage for float32 and float64, else lane storage
// stored as the type once after (a block format's by its encoder). Grouped heads run as KvHeads heads of Group·Lq rows
// (AttentionArgs.Grouped) by every route; a cap on the scores takes the composition (the map's softcap, softcap_bwd).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The gate's override: false takes the backward's composition, null the fused kernels where they fit.</summary>
    public bool? FusedBackward;

    /// <summary>The backward's last route ("fused", "composed"), for the gate.</summary>
    public string? BackwardRoute;

    private readonly Dictionary<string, bool> _fusedFaster = [];

    private static Fmt AttentionFormat<TN>(string what) where TN : struct, IAdad<TN> => Fmt.Of<TN>();

    private static bool Own(Fmt f) => f.Is("float32") || f.Is("float64");

    // An int per query on the device.
    private ulong PerQuery(AttentionArgs a, Func<int, int> value, List<ulong> rented)
    {
        var host = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) host[i] = value(i);
        ulong d = vk.Rent(Math.Max(1, a.Lq) * 4L);
        rented.Add(d);
        vk.Upload<int>(host, d);
        return d;
    }

    // Each block of `br` query rows: the keys its rows see, [lowest start, highest limit) (0, 0 when none).
    private ulong KeyRanges(AttentionArgs a, int br, List<ulong> rented)
    {
        int blocks = (a.Lq + br - 1) / br;
        var range = new int[2 * blocks];
        for (int qb = 0; qb < blocks; qb++)
        {
            int lo = int.MaxValue, hi = 0;
            for (int i = qb * br; i < Math.Min(a.Lq, qb * br + br); i++) { lo = Math.Min(lo, a.Start(i)); hi = Math.Max(hi, a.Limit(i)); }
            if (hi <= lo) lo = hi = 0;
            range[2 * qb] = lo; range[2 * qb + 1] = hi;
        }

        ulong d = vk.Rent(Math.Max(2, range.Length) * 4L);
        rented.Add(d);
        vk.Upload<int>(range, d);
        return d;
    }

    // Each block of `bc` keys: the queries that see any of them, [first, last + 1) (0, 0 when none).
    private ulong QueryRanges(AttentionArgs a, int bc, List<ulong> rented)
    {
        int blocks = (a.Lk + bc - 1) / bc;
        var range = new int[2 * blocks];
        for (int kb = 0; kb < blocks; kb++)
        {
            int k0 = kb * bc, k1 = Math.Min(a.Lk, k0 + bc), lo = int.MaxValue, hi = 0;
            for (int i = 0; i < a.Lq; i++)
            {
                if (Math.Max(a.Start(i), k0) < Math.Min(a.Limit(i), k1)) { lo = Math.Min(lo, i); hi = i + 1; }
            }

            if (hi <= lo) lo = hi = 0;
            range[2 * kb] = lo; range[2 * kb + 1] = hi;
        }

        ulong d = vk.Rent(Math.Max(2, range.Length) * 4L);
        rented.Add(d);
        vk.Upload<int>(range, d);
        return d;
    }

    // ---- rows of queries

    /// <summary>O for rows of queries: the fused rows or the composition, whichever this device measures faster at this
    /// shape and type (once); the composition past the fused kernel's head dimension. Returns the lane's name.</summary>
    public string Attention<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN>
    {
        // Tensors through views: every form by AttentionLayout's strided copies.
        if (a.Laid) return AttentionLayout.Forward<ulong, ulong, TN>(this, a, q, k, v, o, static (d, h, q, k, v, o) => d.Attention<TN>(h, q, k, v, o));
        var f = AttentionFormat<TN>("attention");
        a = a.Grouped();
        if (a.D > VkKernels.FusedHeadMax || !RowsFit(RowLane(f), a.D) || a.Softcap > 0) return AttentionComposed<TN>(a, q, k, v, o);
        string key = $"{f.Name}:{a.Heads}x{a.Lq}x{a.Lk}x{a.D}:{a.Causal}:{a.Spans is not null}:{a.HasStarts}";
        if (!_fusedFaster.TryGetValue(key, out bool fused))
        {
            var (tf, tc) = Measure.Faster(() => AttentionRows<TN>(a, q, k, v, o), () => AttentionComposed<TN>(a, q, k, v, o), vk.Synchronize);
            fused = _fusedFaster[key] = tf <= tc;
        }

        return fused ? AttentionRows<TN>(a, q, k, v, o) : AttentionComposed<TN>(a, q, k, v, o);
    }

    private bool RowsFit(Lane lane, int d) => VkKernels.AttentionRowsShared(lane, VkKernels.AttentionCap(d)) <= vk.MaxSharedBytes;

    /// <summary>O for rows of queries by the fused rows (no scores in memory); d ≤ 128. Returns the lane's name.</summary>
    public string AttentionRows<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return AttentionLayout.Forward<ulong, ulong, TN>(this, a, q, k, v, o, static (d, h, q, k, v, o) => d.AttentionRows<TN>(h, q, k, v, o));
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionComposed<TN>(a, q, k, v, o);
        var f = AttentionFormat<TN>("attention rows");
        var lane = RowLane(f);
        if (a.D > VkKernels.FusedHeadMax) throw new NotSupportedException($"vulkan: attention rows with a head dimension of {a.D} exceed the fused kernel's {VkKernels.FusedHeadMax}");
        int dCap = VkKernels.AttentionCap(a.D);
        if (!RowsFit(lane, a.D)) throw new NotSupportedException($"vulkan: the fused rows at d {a.D} exceed the workgroup memory ({vk.MaxSharedBytes} bytes)");
        if (a.QValues == 0) return lane.Name;
        var rented = new List<ulong>();
        try
        {
            ulong lo = Own(f) ? o : vk.Rent(a.QValues * lane.Size);
            if (!Own(f)) rented.Add(lo);
            var (br, _) = VkKernels.AttentionTile(lane);
            ulong limits = PerQuery(a, a.Limit, rented), starts = PerQuery(a, a.Start, rented), ranges = KeyRanges(a, br, rented);
            var fn = vk.Cached($"attention_rows:{f.Name}:{dCap}", () => VkKernels.AttentionRows(vk, f, lane, dCap));
            vk.Launch(fn, (uint)((a.Lq + br - 1) / br), (uint)a.Heads, q, k, v, lo, limits, starts, ranges, Vulkan.U(a.Lq), Vulkan.U(a.Lk), Vulkan.U(a.D), Slot(lane, a.Scale));
            if (!Own(f)) Back(f, lane, lo, o, a.QValues);
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }

    /// <summary>O for rows of queries by the composition: per head S = Q·Kᵀ (lane storage; capped when there is a cap), the
    /// scaled masked softmax, O = P·V, the products reading Q, K, V where they are stored. Returns the lane's name.</summary>
    public string AttentionComposed<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> =>
        a.Laid ? AttentionLayout.Forward<ulong, ulong, TN>(this, a, q, k, v, o, static (d, h, q, k, v, o) => d.AttentionComposed<TN>(h, q, k, v, o)) : Composed<TN>(a, q, k, v, o, 0);

    /// <summary>O and each query's log-sum-exp of its scores (heads × Lq, as the type): the composition, the softmax's row
    /// values (lane storage) stored as the type after. Returns the lane's name.</summary>
    public string AttentionLse<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o, ulong lse) where TN : struct, IFloatAdad<TN> =>
        a.Laid ? AttentionLayout.Forward<ulong, ulong, TN>(this, a, q, k, v, o, (d, h, q, k, v, o) => d.AttentionLse<TN>(h, q, k, v, o, lse)) : Composed<TN>(a, q, k, v, o, lse);

    private string Composed<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o, ulong lse) where TN : struct, IFloatAdad<TN>
    {
        var f = AttentionFormat<TN>("attention (composed)");
        var lane = RowLane(f);
        a = a.Grouped();
        if (a.QValues == 0) return lane.Name;
        var lf = VkKernels.LaneFormat(lane);
        var rented = new List<ulong>();
        try
        {
            ulong Rent(long bytes) { ulong p = vk.Rent(bytes); rented.Add(p); return p; }
            ulong s = Rent(Math.Max(1, (long)a.Lq * a.Lk) * lane.Size), lo = Own(f) ? o : Rent(a.QValues * lane.Size);
            ulong ll = lse == 0 ? 0 : Own(f) ? lse : Rent((long)a.Heads * a.Lq * lane.Size);
            // positions = the last visible key of each query (limit − 1): scale_mask_softmax's causal rule excludes the keys past it.
            ulong last = PerQuery(a, i => a.Limit(i) - 1, rented), mask = SpanMask(a, lane, rented);
            var softmax = RowOps.Find(lse != 0 ? "scale_mask_softmax_lse" : "scale_mask_softmax");
            bool capped = a.Softcap > 0;
            int pairs = (int)((long)a.Lq * a.Lk);
            for (int h = 0; h < a.Heads; h++)
            {
                long qo = (long)h * a.Lq * a.D, ko = (long)h * a.Lk * a.D;
                Product(new ProductView(a.Lq, a.Lk, a.D, false, true), lane, q, f, qo, k, f, ko, s);
                if (capped) LaneMap(lane, "softcap", s, 0, s, pairs, a.Softcap, a.Scale);                  // cap·tanh(scale·S / cap)
                Row(lf, softmax, s, mask, 0, 0, 0, last, s, ll == 0 ? 0 : ll + (ulong)((long)h * a.Lq * lane.Size), a.Lq, a.Lk, capped ? 1 : a.Scale, 1);
                Product(new ProductView(a.Lq, a.D, a.Lk, false, false), lane, s, lf, 0, v, f, ko, lo + (ulong)(qo * lane.Size));
            }

            if (!Own(f)) Back(f, lane, lo, o, a.QValues);
            if (ll != 0 && !Own(f)) Back(f, lane, ll, lse, (long)a.Heads * a.Lq);
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }

    // C (lane values at c) = op(A)·op(B), A stored as fa from value aOffset, B as fb from bOffset: one view (no batches).
    private void Product(ProductView v, Lane lane, ulong a, Fmt fa, long aOffset, ulong b, Fmt fb, long bOffset, ulong c)
    {
        if (v.M == 0 || v.N == 0) return;
        if (aOffset > uint.MaxValue || bOffset > uint.MaxValue) throw new NotSupportedException("vulkan: a product's operand past 2^32 values");
        var fn = vk.Cached($"matmul:{fa.Name}:{fb.Name}:{lane.Name}:{v.TransA}:{v.TransB}", () => VkKernels.MatMul(vk, fa, fb, lane, v.TransA, v.TransB));
        int tilesM = (v.M + VkKernels.TileM - 1) / VkKernels.TileM;
        vk.Launch(fn, Groups(v.N, VkKernels.TileN), (uint)tilesM, a, b, c, Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K), 0, 0, 0, Vulkan.U(tilesM),
            Vulkan.U(aOffset), Vulkan.U(bOffset));
    }

    // A formula of Map over lane values (the lane's own type).
    private void LaneMap(Lane lane, string op, ulong x, ulong y, ulong output, int n, double a, double b)
    {
        var m = Ops.Find(op);
        if (lane.Wide) Map<Float64>(m, x, y, 0, output, n, a, b); else Map<Float32>(m, x, y, 0, output, n, a, b);
    }

    /// <summary>The additive mask excluding the keys below each query's start (span starts, a window; lane storage) by the
    /// excluded score (CacheView.Excluded of the lane's own type: −∞, so such a key takes no weight and a query left with
    /// none 0), 0 when none.</summary>
    private ulong SpanMask(AttentionArgs a, Lane lane, List<ulong> rented)
    {
        if (!a.HasStarts) return 0;
        double excluded = lane.Wide ? CacheView.Excluded<Float64>() : CacheView.Excluded<Float32>();
        var m = new double[(long)a.Lq * a.Lk];
        for (int i = 0; i < a.Lq; i++) for (int j = 0; j < a.Start(i); j++) m[(long)i * a.Lk + j] = excluded;
        var bytes = lane.Wide ? Stored.StoreValues<Float64>(m) : Stored.StoreValues<Float32>(m);
        ulong d = vk.Rent(bytes.Length);
        rented.Add(d);
        vk.Upload<byte>(bytes, d);
        return d;
    }

    // ---- decode

    /// <summary>O for a few queries against a long cache, a workgroup per query, K and V read once; returns the lane's name.</summary>
    public string AttentionDecode<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return AttentionLayout.Forward<ulong, ulong, TN>(this, a, q, k, v, o, static (d, h, q, k, v, o) => d.AttentionDecode<TN>(h, q, k, v, o));
        a = a.Grouped();
        if (a.Softcap > 0) return AttentionComposed<TN>(a, q, k, v, o);
        var f = AttentionFormat<TN>("attention decode");
        var lane = RowLane(f);
        if (a.D > 4 * VkKernels.DecodeThreads) throw new NotSupportedException($"vulkan: attention decode with a head dimension of {a.D} exceeds {4 * VkKernels.DecodeThreads}");
        if (a.QValues == 0) return lane.Name;
        int dims = (a.D + VkKernels.DecodeThreads - 1) / VkKernels.DecodeThreads * VkKernels.DecodeThreads;
        var rented = new List<ulong>();
        try
        {
            ulong lo = Own(f) ? o : vk.Rent(a.QValues * lane.Size);
            if (!Own(f)) rented.Add(lo);
            ulong limits = PerQuery(a, a.Limit, rented), starts = PerQuery(a, a.Start, rented);
            bool runs = f.IsBlock && a.D % 8 == 0;                              // a block format's keys in runs of 8 values
            var fn = vk.Cached($"attention_decode:{f.Name}:{dims}{(runs ? ":runs" : "")}", () => VkKernels.AttentionDecode(vk, f, lane, dims, null, runs));
            vk.Launch(fn, (uint)(a.Heads * a.Lq), 1, q, k, v, lo, limits, starts, Vulkan.U(a.Lq), Vulkan.U(a.Lk), Vulkan.U(a.D), Slot(lane, a.Scale));
            if (!Own(f)) Back(f, lane, lo, o, a.QValues);
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }

    // ---- backward

    /// <summary>dQ, dK, dV from dO: the fused kernels where both fit the workgroup memory (d ≤ 128), else the composition;
    /// returns the lane's name.</summary>
    public string AttentionBackward<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv) where TN : struct, IFloatAdad<TN>
    {
        if (a.Laid) return AttentionLayout.Backward<ulong, ulong, TN>(this, a, q, k, v, dO, dq, dk, dv);
        var f = AttentionFormat<TN>("attention backward");
        var lane = RowLane(f);
        a = a.Grouped();
        var rented = new List<ulong>();
        try
        {
            ulong Rent(long bytes) { ulong p = vk.Rent(bytes); rented.Add(p); return p; }
            bool own = Own(f);
            ulong ldq = own ? dq : Rent(Math.Max(1, a.QValues) * lane.Size), ldk = own ? dk : Rent(Math.Max(1, a.KValues) * lane.Size), ldv = own ? dv : Rent(Math.Max(1, a.KValues) * lane.Size);
            if (FusedBackward != false && a.D <= VkKernels.FusedHeadMax && a.Softcap == 0 && BackwardFits(lane, a.D)) { BackwardFused(f, lane, a, q, k, v, dO, ldq, ldk, ldv, rented); BackwardRoute = "fused"; }
            else { BackwardComposed(f, lane, a, q, k, v, dO, ldq, ldk, ldv, rented); BackwardRoute = "composed"; }
            if (!own)
            {
                Back(f, lane, ldq, dq, a.QValues);
                Back(f, lane, ldk, dk, a.KValues);
                Back(f, lane, ldv, dv, a.KValues);
            }

            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }

    private bool BackwardFits(Lane lane, int d)
    {
        int dCap = VkKernels.AttentionCap(d);
        return VkKernels.AttentionBackRowsShared(lane, dCap) <= vk.MaxSharedBytes && VkKernels.AttentionBackKeysShared(lane, dCap) <= vk.MaxSharedBytes;
    }

    // The rows kernel (dQ, and each row's m, 1/l, D), then the keys kernel (dK, dV); the gradients lane values.
    private void BackwardFused(Fmt f, Lane lane, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented)
    {
        if (a.QValues == 0 && a.KValues == 0) return;
        int dCap = VkKernels.AttentionCap(a.D);
        var (br, bc) = VkKernels.AttentionTile(lane);
        ulong limits = PerQuery(a, a.Limit, rented), starts = PerQuery(a, a.Start, rented), rowRanges = KeyRanges(a, br, rented), keyRanges = QueryRanges(a, bc, rented);
        ulong stats = vk.Rent(Math.Max(1, (long)a.Heads * a.Lq) * 3 * lane.Size);
        rented.Add(stats);
        var scale = Slot(lane, a.Scale);
        var fr = vk.Cached($"attention_back_rows:{f.Name}:{dCap}", () => VkKernels.AttentionBackRows(vk, f, lane, dCap));
        if (a.Lq > 0) vk.Launch(fr, (uint)((a.Lq + br - 1) / br), (uint)a.Heads, q, k, v, dO, dq, stats, limits, starts, rowRanges, Vulkan.U(a.Lq), Vulkan.U(a.Lk), Vulkan.U(a.D), scale);
        var fk = vk.Cached($"attention_back_keys:{f.Name}:{dCap}", () => VkKernels.AttentionBackKeys(vk, f, lane, dCap));
        if (a.Lk > 0) vk.Launch(fk, (uint)((a.Lk + bc - 1) / bc), (uint)a.Heads, q, k, v, dO, stats, dk, dv, limits, starts, keyRanges, Vulkan.U(a.Lq), Vulkan.U(a.Lk), Vulkan.U(a.D), scale);
    }

    // The backward's composition per head: P (lane storage) as the forward's, dV = Pᵀ·dO, dP = dO·Vᵀ, dS = P·(dP − Σ P·dP)
    // · scale, dQ = dS·K, dK = dSᵀ·Q; Q, K, V, dO read where they are stored, the gradients lane values.
    private void BackwardComposed(Fmt f, Lane lane, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented)
    {
        if (a.Lq == 0 || a.Lk == 0)
        {
            // No pairs: the gradients are 0.
            if (a.QValues > 0) vk.Clear(dq, a.QValues * lane.Size);
            if (a.KValues > 0) { vk.Clear(dk, a.KValues * lane.Size); vk.Clear(dv, a.KValues * lane.Size); }
            return;
        }

        var lf = VkKernels.LaneFormat(lane);
        bool capped = a.Softcap > 0;
        long sBytes = (long)a.Lq * a.Lk * lane.Size;
        ulong p = vk.Rent(sBytes), dp = vk.Rent(sBytes), raw = capped ? vk.Rent(sBytes) : 0;
        rented.Add(p); rented.Add(dp);
        if (capped) rented.Add(raw);
        ulong last = PerQuery(a, i => a.Limit(i) - 1, rented), mask = SpanMask(a, lane, rented);
        var softmax = RowOps.Find("scale_mask_softmax");
        var softmaxBwd = RowOps.Find("softmax_bwd");
        var scale = Ops.Find("scale");
        int pairs = (int)((long)a.Lq * a.Lk);
        for (int h = 0; h < a.Heads; h++)
        {
            long qo = (long)h * a.Lq * a.D, ko = (long)h * a.Lk * a.D;
            ulong dqh = dq + (ulong)(qo * lane.Size), dkh = dk + (ulong)(ko * lane.Size), dvh = dv + (ulong)(ko * lane.Size);
            Product(new ProductView(a.Lq, a.Lk, a.D, false, true), lane, q, f, qo, k, f, ko, capped ? raw : p);
            if (capped) LaneMap(lane, "softcap", raw, 0, p, pairs, a.Softcap, a.Scale);                       // cap·tanh(scale·S / cap)
            Row(lf, softmax, p, mask, 0, 0, 0, last, p, 0, a.Lq, a.Lk, capped ? 1 : a.Scale, 1);
            Product(new ProductView(a.Lk, a.D, a.Lq, true, false), lane, p, lf, 0, dO, f, qo, dvh);            // dV = Pᵀ·dO
            Product(new ProductView(a.Lq, a.Lk, a.D, false, true), lane, dO, f, qo, v, f, ko, dp);             // dP = dO·Vᵀ
            Row(lf, softmaxBwd, p, dp, 0, 0, 0, 0, dp, 0, a.Lq, a.Lk, 0, 0);                                   // dS = P·(dP − Σ P·dP)
            if (capped) LaneMap(lane, "softcap_bwd", raw, dp, dp, pairs, a.Softcap, a.Scale);                 // · the cap's slope
            else if (lane.Wide) Map<Float64>(scale, dp, 0, 0, dp, pairs, a.Scale, 0); else Map<Float32>(scale, dp, 0, 0, dp, pairs, a.Scale, 0);   // · scale
            Product(new ProductView(a.Lq, a.D, a.Lk, false, false), lane, dp, lf, 0, k, f, ko, dqh);           // dQ = dS·K
            Product(new ProductView(a.Lk, a.D, a.Lq, true, false), lane, dp, lf, 0, q, f, qo, dkh);            // dK = dSᵀ·Q
        }
    }
}
