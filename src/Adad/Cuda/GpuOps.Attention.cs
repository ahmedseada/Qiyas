// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Adad.Abstraction;
// The device's attention (Attention.cs): the fused rows (the tensor rows for the types with a tensor kind, else the
// FMA rows) or the composition for rows of queries (the view products and the scaled masked softmax per head), the fused
// decode kernel for a few queries against a long cache, the backward's fused kernels (d ≤ 128) or its composition. Q, K,
// V and dO are read where they are stored, as the type, by every route: the scores and their gradients (intermediates)
// are lane storage, the outputs stored as the type (a block format's and a sub-byte one's: the lane's values, encoded
// by blocks or words after).
namespace Adad.Cuda;

public sealed partial class GpuOps
{
    private static bool Wide(Fmt f) => f.Bits == 64;

    // ---- lane storage on the device: a lane's values into the type's storage (the outputs the kernels do not store as the type)

    private void FromLane<TN>(ulong lanes, ulong output, long count) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.Is("float32") || f.Is("float64")) return;
        var lane = RowLane(f);
        if (f.IsBlock) { Back(f, lane, lanes, output, count); return; }
        var fn = gpu.Cached($"narrow:{f.Name}:{lane.Name}", (out string e) => Ptx.Narrow(gpu, f, lane, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, lanes, output, Gpu.U((int)count));
    }

    private ulong Limits(AttentionArgs a, List<ulong> rented)
    {
        var limits = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) limits[i] = a.Limit(i);
        ulong d = gpu.Rent(a.Lq * 4L);
        rented.Add(d);
        Gpu.Upload<int>(limits, d);
        return d;
    }

    private ulong Starts(AttentionArgs a, List<ulong> rented)
    {
        var starts = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) starts[i] = a.Start(i);
        ulong d = gpu.Rent(a.Lq * 4L);
        rented.Add(d);
        Gpu.Upload<int>(starts, d);
        return d;
    }

    // ---- rows of queries: the fused rows, or the composition where the device measures it faster

    private readonly Dictionary<string, bool> _fusedFaster = [];

    /// <summary>O for rows of queries: the fused rows (no scores in memory) or the composition, whichever this device
    /// measures faster at this shape and type (once); the composition for head dimensions past the fused kernel's 128.
    /// Returns the lane's name.</summary>
    public string Attention<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (a.D > FusedHeadMax) return AttentionComposed<TN>(a, q, k, v, o);
        string key = $"{f.Name}:{a.Heads}x{a.Lq}x{a.Lk}x{a.D}:{a.Causal}:{a.Spans is not null}:{a.SpanStarts is not null}";
        if (!_fusedFaster.TryGetValue(key, out bool fused))
        {
            // As Measure.Faster settles it: a device still raising its clocks (the first timings of a run) keeps the
            // alternating rounds going instead of deciding against the path timed first.
            var (tf, tc) = Measure.Faster(() => AttentionRows<TN>(a, q, k, v, o), () => AttentionComposed<TN>(a, q, k, v, o), Gpu.Synchronize);

            fused = _fusedFaster[key] = tf <= tc;
            Note($"attention {key}: fused rows {tf:0.000} ms, composition {tc:0.000} ms: {(fused ? "fused rows" : "composition")}");
        }

        return fused ? AttentionRows<TN>(a, q, k, v, o) : AttentionComposed<TN>(a, q, k, v, o);
    }

    /// <summary>The fused kernel's largest head dimension.</summary>
    public const int FusedHeadMax = 128;

    /// <summary>The gate's override: false keeps the fused rows of a type with a tensor kind on the FMA lane.</summary>
    public bool? MmaRows;

    // Each block of `br` query rows: the keys its rows see, [lowest start, highest limit) (0, 0 when none), on the device.
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

        ulong d = gpu.Rent(range.Length * 4L);
        rented.Add(d);
        Gpu.Upload<int>(range, d);
        return d;
    }

    /// <summary>O for rows of queries by the fused rows: a block per (head, query tile), the keys in tiles through shared
    /// memory, the softmax online; no scores in memory. d ≤ 128. Returns the lane's name.</summary>
    public string AttentionRows<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.D > FusedHeadMax) throw new NotSupportedException($"attention rows: a head dimension of {a.D} exceeds the fused kernel's {FusedHeadMax}");
        int dCap = a.D <= 32 ? 32 : a.D <= 64 ? 64 : 128;
        var rented = new List<ulong>();
        try
        {
            // A block format: Q, K, V decoded where they are stored (the kernel's loads), O the rows' f32 values (a block of O
            // may span the rows of several blocks of queries), encoded by blocks after.
            ulong lo = f.IsBlock ? gpu.Rent(a.QValues * 4) : o;
            if (f.IsBlock) rented.Add(lo);
            var kind = MmaRows != false && !f.IsBlock ? Ptx.AttentionKind(f) : null;
            char mLoad = f.Bits == 16 && a.D % 8 == 0 && (q | k | v) % 16 == 0 ? 'c' : f.Bits == 8 && a.D % 4 == 0 && (q | k | v) % 4 == 0 ? 'w' : 'v';
            if (kind is not null && !(Ptx.AttentionMmaShared(dCap) <= AttnSharedLimit
                && AttnAccepts($"attention_rows_mma:{f.Name}:{dCap}:{mLoad}", (out string e) => Ptx.AttentionRowsMma(gpu, f, kind, dCap, mLoad, out e)))) kind = null;
            if (kind is not null)
            {
                // The tensor rows (Ptx.AttentionMma.cs): Q, K, V exact in the kind; 16-byte copies or 32-bit words where the
                // rows allow, else value by value.
                int mBlocks = (a.Lq + Ptx.MmaRowsBr - 1) / Ptx.MmaRowsBr;
                ulong mLimits = Limits(a, rented), mStarts = Starts(a, rented), mRanges = KeyRanges(a, Ptx.MmaRowsBr, rented);
                char load = f.Bits == 16 && a.D % 8 == 0 && (q | k | v) % 16 == 0 ? 'c' : f.Bits == 8 && a.D % 4 == 0 && (q | k | v) % 4 == 0 ? 'w' : 'v';
                var mfn = gpu.Cached($"attention_rows_mma:{f.Name}:{dCap}:{load}", (out string e) => Ptx.AttentionRowsMma(gpu, f, kind, dCap, load, out e));
                Gpu.Launch(mfn, (uint)mBlocks, (uint)a.Heads, Ptx.MmaRowsThreads, 1, 0, q, k, v, o, mLimits, mStarts, mRanges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), Gpu.F((float)a.Scale));
                return kind.Name + "mma";
            }

            var (br, _) = Ptx.AttentionTile(lane);
            int blocks = (a.Lq + br - 1) / br;
            ulong limits = Limits(a, rented), starts = Starts(a, rented), ranges = KeyRanges(a, br, rented);
            bool runs = f.IsBlock && a.D % 8 == 0;
            var fn = gpu.Cached($"attention_rows:{f.Name}:{dCap}:{runs}", (out string e) => Ptx.AttentionRows(gpu, f, lane, dCap, out e, f.IsBlock ? Fmt.Of<Float32>() : null, runs));
            Gpu.Launch(fn, (uint)blocks, (uint)a.Heads, Ptx.RowsThreads, 1, 0, q, k, v, lo, limits, starts, ranges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), lane.Wide ? Gpu.D(a.Scale) : Gpu.F((float)a.Scale));
            if (!f.IsBlock) return lane.Name;
            Back(f, lane, lo, o, a.QValues);
            return "f32 (blocks in place)";
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>O for rows of queries by the composition; returns the lane's name.</summary>
    public string AttentionComposed<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var rented = new List<ulong>();
        try
        {
            if (f.Is("float64")) Compose<Float64>(a, q, k, v, o, rented);
            else if (f.Is("float32")) Compose<Float32>(a, q, k, v, o, rented);
            else ComposeStored(f, a, q, k, v, o, rented);
            return RowLane(f).Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    // The composition on the type's own tensors (f32 lane): S = Q·Kᵀ and O = P·V with Q, K, V read where they are stored
    // (ProductStored), the scores and their softmax in f32 lane storage, O stored as the type.
    private void ComposeStored(Fmt f, AttentionArgs a, ulong q, ulong k, ulong v, ulong o, List<ulong> rented)
    {
        var lane = Lane.F32;
        var f32 = Fmt.Of<Float32>();
        bool typed = StoredC(f);
        ulong s = gpu.Rent((long)a.Lq * a.Lk * 4), lo = typed ? o : gpu.Rent(a.QValues * 4);
        rented.Add(s);
        if (!typed) rented.Add(lo);
        ulong limits = Limits(a, rented);
        var softmax = RowOps.Find("scale_mask_softmax");
        var last = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) last[i] = a.Limit(i) - 1;
        Gpu.Upload<int>(last, limits);
        ulong mask = SpanMask<Float32>(a, rented);
        for (int h = 0; h < a.Heads; h++)
        {
            long qo = (long)h * a.Lq * a.D, ko = (long)h * a.Lk * a.D;
            ProductStored(new ProductView(a.Lq, a.Lk, a.D, false, true), lane, q, f, qo, k, f, ko, s, null);
            RowOp<Float32>(softmax, s, mask, 0, 0, 0, limits, s, 0, a.Lq, a.Lk, a.Scale, 1);
            ProductStored(new ProductView(a.Lq, a.D, a.Lk, false, false), lane, s, f32, 0, v, f, ko, Out(o, lo, qo, f, typed), typed ? f : null);
        }

        if (!typed) Back(f, lane, lo, o, a.QValues);
    }

    // Value `at` of an output: of the type's storage where the kernels store it (typed), else of its f32 lane storage.
    private static ulong Out(ulong typedPtr, ulong lanePtr, long at, Fmt f, bool typed) => typed ? typedPtr + (ulong)(at * f.Bits / 8) : lanePtr + (ulong)(at * 4);

    private void Compose<TS>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o, List<ulong> rented) where TS : struct, IFloatAdad<TS>
    {
        int size = TS.Bits / 8;
        ulong s = gpu.Rent((long)a.Lq * a.Lk * size);
        rented.Add(s);
        ulong limits = Limits(a, rented);
        var softmax = RowOps.Find("scale_mask_softmax");
        // positions = the last visible key of each query (limit − 1): scale_mask_softmax's causal rule excludes the keys past it.
        var last = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) last[i] = a.Limit(i) - 1;
        Gpu.Upload<int>(last, limits);
        ulong mask = SpanMask<TS>(a, rented);
        for (int h = 0; h < a.Heads; h++)
        {
            ulong qh = q + (ulong)((long)h * a.Lq * a.D * size), kh = k + (ulong)((long)h * a.Lk * a.D * size), vh = v + (ulong)((long)h * a.Lk * a.D * size), oh = o + (ulong)((long)h * a.Lq * a.D * size);
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), qh, kh, s);
            RowOp<TS>(softmax, s, mask, 0, 0, 0, limits, s, 0, a.Lq, a.Lk, a.Scale, 1);
            MatMulView<TS>(new ProductView(a.Lq, a.D, a.Lk, false, false), s, vh, oh);
        }
    }

    /// <summary>The additive mask excluding the keys below each query's span start (lane storage on the device), 0 when no span starts.</summary>
    private ulong SpanMask<TS>(AttentionArgs a, List<ulong> rented) where TS : struct, IFloatAdad<TS>
    {
        if (a.SpanStarts is null) return 0;
        var m = new double[(long)a.Lq * a.Lk];
        for (int i = 0; i < a.Lq; i++) for (int j = 0; j < a.Start(i); j++) m[(long)i * a.Lk + j] = -400;
        var bytes = Stored.StoreValues<TS>(m);
        ulong d = gpu.Rent(bytes.Length);
        rented.Add(d);
        Gpu.Upload<byte>(bytes, d);
        return d;
    }

    // ---- decode: the fused kernel

    /// <summary>O for a few queries against a long cache, a block per query, K and V read once; returns the lane's name.</summary>
    public string AttentionDecode<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.D > 4 * Ptx.DecodeThreads) throw new NotSupportedException($"attention_decode: a head dimension of {a.D} exceeds {4 * Ptx.DecodeThreads}");
        int dCap = (a.D + 127) / 128 * 128;
        var rented = new List<ulong>();
        try
        {
            // A block format: Q, K, V decoded where they are stored, O the queries' f32 values, encoded by blocks after.
            ulong lo = f.IsBlock ? gpu.Rent(a.QValues * 4) : o;
            if (f.IsBlock) rented.Add(lo);
            ulong limits = Limits(a, rented), starts = Starts(a, rented);
            var fn = gpu.Cached($"attention_decode:{f.Name}:{dCap}", (out string e) => Ptx.AttentionDecode(gpu, f, lane, dCap, out e, f.IsBlock ? Fmt.Of<Float32>() : null));
            Gpu.Launch(fn, (uint)(a.Heads * a.Lq), 1, Ptx.DecodeThreads, 1, 0, q, k, v, lo, limits, starts, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), lane.Wide ? Gpu.D(a.Scale) : Gpu.F((float)a.Scale));
            if (!f.IsBlock) return lane.Name;
            Back(f, lane, lo, o, a.QValues);
            return "f32 (blocks in place)";
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    // ---- backward: the fused kernels (Ptx.AttentionBackward.cs), or the composition

    // The rows kernel (dQ, and each row's m, 1/l, D), then the keys kernel (dK, dV), over lane storage (f null) or the
    // type's tensors (the gradients as the type where the kernels store it, else lane storage).
    private void BackwardFused(Lane lane, Fmt? f, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented)
    {
        bool runs = f is { IsBlock: true } && a.D % 8 == 0;
        string key = f is null ? lane.Name : $"{lane.Name}:{f.Name}:{runs}";
        int dCap = a.D <= 32 ? 32 : a.D <= 64 ? 64 : 128;
        var (br, bc) = Ptx.AttentionTile(lane);
        ulong limits = Limits(a, rented), starts = Starts(a, rented), rowRanges = KeyRanges(a, br, rented), keyRanges = QueryRanges(a, bc, rented);
        ulong stats = gpu.Rent((long)a.Heads * a.Lq * 3 * lane.Size);
        rented.Add(stats);
        var scale = lane.Wide ? Gpu.D(a.Scale) : Gpu.F((float)a.Scale);
        int rowsShared = Ptx.AttentionBackRowsShared(lane, dCap), keysShared = Ptx.AttentionBackKeysShared(lane, dCap);
        var fr = gpu.Cached($"attention_back_rows:{key}:{dCap}", (out string e) => Ptx.AttentionBackRows(gpu, lane, dCap, out e, f, runs));
        gpu.AllowShared(fr, rowsShared);
        Gpu.Launch(fr, (uint)((a.Lq + br - 1) / br), (uint)a.Heads, Ptx.RowsThreads, 1, Dynamic(rowsShared), q, k, v, dO, dq, stats, limits, starts, rowRanges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), scale);
        var fk = gpu.Cached($"attention_back_keys:{key}:{dCap}", (out string e) => Ptx.AttentionBackKeys(gpu, lane, dCap, out e, f, runs));
        gpu.AllowShared(fk, keysShared);
        Gpu.Launch(fk, (uint)((a.Lk + bc - 1) / bc), (uint)a.Heads, Ptx.RowsThreads, 1, Dynamic(keysShared), q, k, v, dO, stats, dk, dv, limits, starts, keyRanges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), scale);
    }

    // Each block of `bc` keys: the queries that see any of them, [first, last + 1) (0, 0 when none), on the device.
    private ulong QueryRanges(AttentionArgs a, int bc, List<ulong> rented)
    {
        int blocks = (a.Lk + bc - 1) / bc;
        var range = new int[2 * blocks];
        for (int kb = 0; kb < blocks; kb++)
        {
            int k0 = kb * bc, k1 = Math.Min(a.Lk, k0 + bc), lo = int.MaxValue, hi = 0;
            for (int i = 0; i < a.Lq; i++)
            {
                if (Math.Max(a.Start(i), k0) < Math.Min(Math.Min(a.Limit(i), a.Lk), k1)) { lo = Math.Min(lo, i); hi = i + 1; }
            }

            if (hi <= lo) lo = hi = 0;
            range[2 * kb] = lo; range[2 * kb + 1] = hi;
        }

        ulong d = gpu.Rent(range.Length * 4L);
        rented.Add(d);
        Gpu.Upload<int>(range, d);
        return d;
    }

    // ---- backward: the composition

    /// <summary>dQ, dK, dV from dO; returns the lane's name.</summary>
    public string AttentionBackward<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var rented = new List<ulong>();
        try
        {
            // A format the tensor kinds hold exactly: the tensor backward (Ptx.AttentionBackMma.cs) on the stored values.
            var kind = FusedBackward != false && MmaBackward != false && a.D <= FusedHeadMax && !f.IsBlock ? Ptx.AttentionKind(f) : null;
            if (kind is not null && MmaBackwardUsable(f, kind, a, q, k, v, dO)) { BackwardMma(f, kind, a, q, k, v, dO, dq, dk, dv, rented); BackwardRoute = "mma"; return kind.Name + "mma"; }
            var lane = RowLane(f);
            if (f.Is("float64")) { BackwardIn<Float64>(a, q, k, v, dO, dq, dk, dv, rented); return lane.Name; }
            if (f.Is("float32")) { BackwardIn<Float32>(a, q, k, v, dO, dq, dk, dv, rented); return lane.Name; }
            // The type's own tensors, read where they are stored; the gradients stored as the type, or as the lane's values
            // encoded after.
            bool typed = StoredC(f);
            ulong ldq = typed ? dq : gpu.Rent(a.QValues * 4), ldk = typed ? dk : gpu.Rent(a.KValues * 4), ldv = typed ? dv : gpu.Rent(a.KValues * 4);
            if (!typed) { rented.Add(ldq); rented.Add(ldk); rented.Add(ldv); }
            if (FusedBackward != false && a.D <= FusedHeadMax && FusedBackwardFits(lane, a.D)) { BackwardFused(lane, f, a, q, k, v, dO, ldq, ldk, ldv, rented); BackwardRoute = "fused"; }
            else { BackwardStored(f, a, q, k, v, dO, ldq, ldk, ldv, typed, rented); BackwardRoute = "composed"; }
            if (!typed)
            {
                Back(f, lane, ldq, dq, a.QValues);
                Back(f, lane, ldk, dk, a.KValues);
                Back(f, lane, ldv, dv, a.KValues);
            }

            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>The gate's override: false takes the backward's composition, null the fused kernels where they apply (d ≤ 128).</summary>
    public bool? FusedBackward;

    /// <summary>The gate's override: false keeps a type with a tensor kind on the FMA fused backward.</summary>
    public bool? MmaBackward;

    /// <summary>The gate's override of the device's shared-memory limit a block may opt into, for the attention routes only
    /// (null: the device's): a kernel whose shared memory exceeds it is not taken, as on a device with that limit.</summary>
    public int? AttentionSharedLimit;

    /// <summary>The backward's last route ("mma", "fused", "composed"), for the gate.</summary>
    public string? BackwardRoute;

    private int AttnSharedLimit => AttentionSharedLimit ?? gpu.SharedOptIn;

    private readonly Dictionary<string, bool> _attnAccepted = [];

    // Whether the device builds a kernel (once per key; a kernel the device refuses is never launched).
    private bool AttnAccepts(string key, Gpu.Maker make)
    {
        if (_attnAccepted.TryGetValue(key, out bool known)) return known;
        string ptx = make(out string entry);
        return _attnAccepted[key] = Accepts(ptx, entry);
    }

    // The tensor backward where both kernels' shared memory fits the limit and the device builds both.
    private bool MmaBackwardUsable(Fmt f, Ptx.Kind kind, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO)
    {
        int dCap = a.D <= 32 ? 32 : a.D <= 64 ? 64 : 128;
        if (Ptx.AttentionMmaShared(dCap) > AttnSharedLimit || Ptx.AttentionBackKeysMmaShared(dCap) > AttnSharedLimit) return false;
        char load = BackLoad(f, a, q, k, v, dO);
        return AttnAccepts($"attention_back_rows_mma:{f.Name}:{dCap}:{load}", (out string e) => Ptx.AttentionBackRowsMma(gpu, f, kind, dCap, load, out e))
            && AttnAccepts($"attention_back_keys_mma:{f.Name}:{dCap}:{load}", (out string e) => Ptx.AttentionBackKeysMma(gpu, f, kind, dCap, load, out e));
    }

    private static char BackLoad(Fmt f, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO) =>
        f.Bits == 16 && a.D % 8 == 0 && (q | k | v | dO) % 16 == 0 ? 'c' : f.Bits == 8 && a.D % 4 == 0 && (q | k | v | dO) % 4 == 0 ? 'w' : 'v';

    // The FMA fused backward where both kernels' shared memory fits the limit (the lane's kernels build on every device).
    private bool FusedBackwardFits(Lane lane, int d)
    {
        int dCap = d <= 32 ? 32 : d <= 64 ? 64 : 128;
        return Ptx.AttentionBackRowsShared(lane, dCap) <= AttnSharedLimit && Ptx.AttentionBackKeysShared(lane, dCap) <= AttnSharedLimit;
    }

    // The tensor backward: the rows kernel (dQ, each row's m, 1/l, D in f32), then the keys kernel (dK, dV), on the stored values.
    private void BackwardMma(Fmt f, Ptx.Kind kind, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented)
    {
        int dCap = a.D <= 32 ? 32 : a.D <= 64 ? 64 : 128;
        ulong limits = Limits(a, rented), starts = Starts(a, rented), rowRanges = KeyRanges(a, Ptx.MmaRowsBr, rented), keyRanges = QueryRanges(a, Ptx.MmaRowsBc, rented);
        ulong stats = gpu.Rent((long)a.Heads * a.Lq * 12);
        rented.Add(stats);
        char load = BackLoad(f, a, q, k, v, dO);
        var scale = Gpu.F((float)a.Scale);
        var fr = gpu.Cached($"attention_back_rows_mma:{f.Name}:{dCap}:{load}", (out string e) => Ptx.AttentionBackRowsMma(gpu, f, kind, dCap, load, out e));
        Gpu.Launch(fr, (uint)((a.Lq + Ptx.MmaRowsBr - 1) / Ptx.MmaRowsBr), (uint)a.Heads, Ptx.MmaRowsThreads, 1, 0, q, k, v, dO, dq, stats, limits, starts, rowRanges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), scale);
        int keysShared = Ptx.AttentionBackKeysMmaShared(dCap);
        var fk = gpu.Cached($"attention_back_keys_mma:{f.Name}:{dCap}:{load}", (out string e) => Ptx.AttentionBackKeysMma(gpu, f, kind, dCap, load, out e));
        gpu.AllowShared(fk, keysShared);
        Gpu.Launch(fk, (uint)((a.Lk + Ptx.MmaRowsBc - 1) / Ptx.MmaRowsBc), (uint)a.Heads, Ptx.MmaRowsThreads, 1, Dynamic(keysShared), q, k, v, dO, stats, dk, dv, limits, starts, keyRanges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), scale);
        Gpu.Flush();
    }

    private void BackwardIn<TS>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented) where TS : struct, IFloatAdad<TS>
    {
        var tsLane = TS.Bits == 64 ? Lane.F64 : Lane.F32;
        if (FusedBackward != false && a.D <= FusedHeadMax && FusedBackwardFits(tsLane, a.D)) { BackwardFused(tsLane, null, a, q, k, v, dO, dq, dk, dv, rented); BackwardRoute = "fused"; return; }
        BackwardRoute = "composed";
        int size = TS.Bits / 8;
        long sBytes = (long)a.Lq * a.Lk * size;
        ulong p = gpu.Rent(sBytes), dp = gpu.Rent(sBytes);
        rented.Add(p); rented.Add(dp);
        ulong limits = Limits(a, rented);
        var last = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) last[i] = a.Limit(i) - 1;
        Gpu.Upload<int>(last, limits);
        ulong mask = SpanMask<TS>(a, rented);
        var softmax = RowOps.Find("scale_mask_softmax");
        var softmaxBwd = RowOps.Find("softmax_bwd");
        var scale = Ops.Find("scale");
        for (int h = 0; h < a.Heads; h++)
        {
            long qo = (long)h * a.Lq * a.D * size, ko = (long)h * a.Lk * a.D * size;
            ulong qh = q + (ulong)qo, kh = k + (ulong)ko, vh = v + (ulong)ko, doh = dO + (ulong)qo, dqh = dq + (ulong)qo, dkh = dk + (ulong)ko, dvh = dv + (ulong)ko;
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), qh, kh, p);
            RowOp<TS>(softmax, p, mask, 0, 0, 0, limits, p, 0, a.Lq, a.Lk, a.Scale, 1);
            MatMulView<TS>(new ProductView(a.Lk, a.D, a.Lq, true, false), p, doh, dvh);            // dV = Pᵀ·dO
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), doh, vh, dp);             // dP = dO·Vᵀ
            RowOp<TS>(softmaxBwd, p, dp, 0, 0, 0, 0, dp, 0, a.Lq, a.Lk, 0, 0);                      // dS = P·(dP − Σ P·dP)
            Map<TS>(scale, dp, 0, 0, dp, (int)((long)a.Lq * a.Lk), a.Scale, 0);                      // · scale
            MatMulView<TS>(new ProductView(a.Lq, a.D, a.Lk, false, false), dp, kh, dqh);            // dQ = dS·K
            MatMulView<TS>(new ProductView(a.Lk, a.D, a.Lq, true, false), dp, qh, dkh);             // dK = dSᵀ·Q
        }
    }

    // The backward's composition on the type's own tensors (f32 lane): every product reads Q, K, V, dO where they are
    // stored and P, dS as f32 lane storage (ProductStored); the gradients at dq, dk, dv as the type (typed) or f32 lanes.
    private void BackwardStored(Fmt f, AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, bool typed, List<ulong> rented)
    {
        var lane = Lane.F32;
        var f32 = Fmt.Of<Float32>();
        var fc = typed ? f : null;
        long sBytes = (long)a.Lq * a.Lk * 4;
        ulong p = gpu.Rent(sBytes), dp = gpu.Rent(sBytes);
        rented.Add(p); rented.Add(dp);
        ulong limits = Limits(a, rented);
        var last = new int[a.Lq];
        for (int i = 0; i < a.Lq; i++) last[i] = a.Limit(i) - 1;
        Gpu.Upload<int>(last, limits);
        ulong mask = SpanMask<Float32>(a, rented);
        var softmax = RowOps.Find("scale_mask_softmax");
        var softmaxBwd = RowOps.Find("softmax_bwd");
        var scale = Ops.Find("scale");
        for (int h = 0; h < a.Heads; h++)
        {
            long qo = (long)h * a.Lq * a.D, ko = (long)h * a.Lk * a.D;
            ProductStored(new ProductView(a.Lq, a.Lk, a.D, false, true), lane, q, f, qo, k, f, ko, p, null);
            RowOp<Float32>(softmax, p, mask, 0, 0, 0, limits, p, 0, a.Lq, a.Lk, a.Scale, 1);
            ProductStored(new ProductView(a.Lk, a.D, a.Lq, true, false), lane, p, f32, 0, dO, f, qo, Out(dv, dv, ko, f, typed), fc);    // dV = Pᵀ·dO
            ProductStored(new ProductView(a.Lq, a.Lk, a.D, false, true), lane, dO, f, qo, v, f, ko, dp, null);                          // dP = dO·Vᵀ
            RowOp<Float32>(softmaxBwd, p, dp, 0, 0, 0, 0, dp, 0, a.Lq, a.Lk, 0, 0);                                                   // dS = P·(dP − Σ P·dP)
            Map<Float32>(scale, dp, 0, 0, dp, (int)((long)a.Lq * a.Lk), a.Scale, 0);                                                   // · scale
            ProductStored(new ProductView(a.Lq, a.D, a.Lk, false, false), lane, dp, f32, 0, k, f, ko, Out(dq, dq, qo, f, typed), fc);  // dQ = dS·K
            ProductStored(new ProductView(a.Lk, a.D, a.Lq, true, false), lane, dp, f32, 0, q, f, qo, Out(dk, dk, ko, f, typed), fc);   // dK = dSᵀ·Q
        }
    }
}
