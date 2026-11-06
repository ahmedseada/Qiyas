using System.Diagnostics;
// The device's attention (Attention.cs): the fused rows (the tensor rows for the types with a tensor kind, else the
// FMA rows) or the composition in lane storage for rows of queries (widen once, the view products and the scaled masked
// softmax per head, narrow once), the fused decode kernel for a few queries against a long cache, the backward's fused
// kernels (d ≤ 128) or its composition.
namespace Qiyas;

public sealed partial class GpuOps
{
    private static bool Wide(Fmt f) => f.Bits == 64;

    // ---- lane storage on the device

    /// <summary>x (format TN, count values) as lane storage: x itself when the type is the lane's, else a rented buffer (returned by the caller).</summary>
    private ulong ToLane<TN>(ulong x, long count, List<ulong> rented) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.Is("float32") || f.Is("float64")) return x;
        var lane = RowLane(f);
        ulong wide = gpu.Rent(count * lane.Size);
        rented.Add(wide);
        var fn = gpu.Cached($"widen:{f.Name}:{lane.Name}", (out string e) => Ptx.Widen(gpu, f, lane, out e));
        Gpu.Launch(fn, Grid(count, 256), 1, 256, 1, 0, x, wide, Gpu.U((int)count));
        return wide;
    }

    private void FromLane<TN>(ulong lanes, ulong output, long count) where TN : struct, IQiyasNumber<TN>
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
    public string Attention<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IQiyasNumber<TN>
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
    public string AttentionRows<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.D > FusedHeadMax) throw new NotSupportedException($"attention rows: a head dimension of {a.D} exceeds the fused kernel's {FusedHeadMax}");
        int dCap = a.D <= 32 ? 32 : a.D <= 64 ? 64 : 128;
        var rented = new List<ulong>();
        try
        {
            if (f.IsBlock)
            {
                ulong bq = ToLane<TN>(q, a.QValues, rented), bk = ToLane<TN>(k, a.KValues, rented), bv = ToLane<TN>(v, a.KValues, rented), bo = gpu.Rent(a.QValues * 4);
                rented.Add(bo);
                AttentionRows<Float32>(a, bq, bk, bv, bo);
                FromLane<TN>(bo, o, a.QValues);
                return "f32 (blocks through lane storage)";
            }

            var kind = MmaRows != false ? Ptx.AttentionKind(f) : null;
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
            var fn = gpu.Cached($"attention_rows:{f.Name}:{dCap}", (out string e) => Ptx.AttentionRows(gpu, f, lane, dCap, out e));
            Gpu.Launch(fn, (uint)blocks, (uint)a.Heads, Ptx.RowsThreads, 1, 0, q, k, v, o, limits, starts, ranges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), lane.Wide ? Gpu.D(a.Scale) : Gpu.F((float)a.Scale));
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>O for rows of queries by the composition in lane storage; returns the lane's name.</summary>
    public string AttentionComposed<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var rented = new List<ulong>();
        try
        {
            ulong lq = ToLane<TN>(q, a.QValues, rented), lk = ToLane<TN>(k, a.KValues, rented), lv = ToLane<TN>(v, a.KValues, rented);
            bool own = f.Is("float32") || f.Is("float64");
            ulong lo = own ? o : gpu.Rent(a.QValues * RowLane(f).Size);
            if (!own) rented.Add(lo);
            if (Wide(f)) Compose<Float64>(a, lq, lk, lv, lo, rented); else Compose<Float32>(a, lq, lk, lv, lo, rented);
            FromLane<TN>(lo, o, a.QValues);
            return RowLane(f).Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    private void Compose<TS>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o, List<ulong> rented) where TS : struct, IQiyasFloat<TS>
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
        bool debug = Environment.GetEnvironmentVariable("QIYAS_MAP_DEBUG") == "1";
        for (int h = 0; h < a.Heads; h++)
        {
            ulong qh = q + (ulong)((long)h * a.Lq * a.D * size), kh = k + (ulong)((long)h * a.Lk * a.D * size), vh = v + (ulong)((long)h * a.Lk * a.D * size), oh = o + (ulong)((long)h * a.Lq * a.D * size);
            MatMulView<TS>(new ProductView(a.Lq, a.Lk, a.D, false, true), qh, kh, s);
            if (debug) { Gpu.Synchronize(); Console.WriteLine($"  attention head {h}: scores done"); }
            RowOp<TS>(softmax, s, mask, 0, 0, 0, limits, s, 0, a.Lq, a.Lk, a.Scale, 1);
            if (debug) { Gpu.Synchronize(); Console.WriteLine($"  attention head {h}: softmax done"); }
            MatMulView<TS>(new ProductView(a.Lq, a.D, a.Lk, false, false), s, vh, oh);
            if (debug) { Gpu.Synchronize(); Console.WriteLine($"  attention head {h}: context done"); }
        }
    }

    /// <summary>The additive mask excluding the keys below each query's span start (lane storage on the device), 0 when no span starts.</summary>
    private ulong SpanMask<TS>(AttentionArgs a, List<ulong> rented) where TS : struct, IQiyasFloat<TS>
    {
        if (a.SpanStarts is null) return 0;
        var m = new double[(long)a.Lq * a.Lk];
        for (int i = 0; i < a.Lq; i++) for (int j = 0; j < a.Start(i); j++) m[(long)i * a.Lk + j] = -400;
        var bytes = Bench.Registry.Encode(TS.Name, m);
        ulong d = gpu.Rent(bytes.Length);
        rented.Add(d);
        Gpu.Upload<byte>(bytes, d);
        return d;
    }

    // ---- decode: the fused kernel

    /// <summary>O for a few queries against a long cache, a block per query, K and V read once; returns the lane's name.</summary>
    public string AttentionDecode<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (a.D > 4 * Ptx.DecodeThreads) throw new NotSupportedException($"attention_decode: a head dimension of {a.D} exceeds {4 * Ptx.DecodeThreads}");
        int dCap = (a.D + 127) / 128 * 128;
        var rented = new List<ulong>();
        try
        {
            if (f.IsBlock)
            {
                ulong lq = ToLane<TN>(q, a.QValues, rented), lk = ToLane<TN>(k, a.KValues, rented), lv = ToLane<TN>(v, a.KValues, rented), lo = gpu.Rent(a.QValues * 4);
                rented.Add(lo);
                AttentionDecode<Float32>(a, lq, lk, lv, lo);
                FromLane<TN>(lo, o, a.QValues);
                return "f32 (blocks through lane storage)";
            }

            ulong limits = Limits(a, rented), starts = Starts(a, rented);
            var fn = gpu.Cached($"attention_decode:{f.Name}:{dCap}", (out string e) => Ptx.AttentionDecode(gpu, f, lane, dCap, out e));
            Gpu.Launch(fn, (uint)(a.Heads * a.Lq), 1, Ptx.DecodeThreads, 1, 0, q, k, v, o, limits, starts, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), lane.Wide ? Gpu.D(a.Scale) : Gpu.F((float)a.Scale));
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    // ---- backward: the fused kernels (Ptx.AttentionBackward.cs), or the composition

    // The rows kernel (dQ, and each row's m, 1/l, D), then the keys kernel (dK, dV), over lane storage.
    private void BackwardFused<TS>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented) where TS : struct, IQiyasFloat<TS>
    {
        var lane = TS.Bits == 64 ? Lane.F64 : Lane.F32;
        int dCap = a.D <= 32 ? 32 : a.D <= 64 ? 64 : 128;
        var (br, bc) = Ptx.AttentionTile(lane);
        ulong limits = Limits(a, rented), starts = Starts(a, rented), rowRanges = KeyRanges(a, br, rented), keyRanges = QueryRanges(a, bc, rented);
        ulong stats = gpu.Rent((long)a.Heads * a.Lq * 3 * lane.Size);
        rented.Add(stats);
        var scale = lane.Wide ? Gpu.D(a.Scale) : Gpu.F((float)a.Scale);
        int rowsShared = Ptx.AttentionBackRowsShared(lane, dCap), keysShared = Ptx.AttentionBackKeysShared(lane, dCap);
        var fr = gpu.Cached($"attention_back_rows:{lane.Name}:{dCap}", (out string e) => Ptx.AttentionBackRows(gpu, lane, dCap, out e));
        gpu.AllowShared(fr, rowsShared);
        Gpu.Launch(fr, (uint)((a.Lq + br - 1) / br), (uint)a.Heads, Ptx.RowsThreads, 1, Dynamic(rowsShared), q, k, v, dO, dq, stats, limits, starts, rowRanges, Gpu.U(a.Lq), Gpu.U(a.Lk), Gpu.U(a.D), scale);
        var fk = gpu.Cached($"attention_back_keys:{lane.Name}:{dCap}", (out string e) => Ptx.AttentionBackKeys(gpu, lane, dCap, out e));
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
    public string AttentionBackward<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var rented = new List<ulong>();
        try
        {
            var lane = RowLane(f);
            bool own = f.Is("float32") || f.Is("float64");
            ulong lq = ToLane<TN>(q, a.QValues, rented), lk = ToLane<TN>(k, a.KValues, rented), lv = ToLane<TN>(v, a.KValues, rented), ldo = ToLane<TN>(dO, a.QValues, rented);
            ulong ldq = own ? dq : gpu.Rent(a.QValues * lane.Size), ldk = own ? dk : gpu.Rent(a.KValues * lane.Size), ldv = own ? dv : gpu.Rent(a.KValues * lane.Size);
            if (!own) { rented.Add(ldq); rented.Add(ldk); rented.Add(ldv); }
            if (Wide(f)) BackwardIn<Float64>(a, lq, lk, lv, ldo, ldq, ldk, ldv, rented); else BackwardIn<Float32>(a, lq, lk, lv, ldo, ldq, ldk, ldv, rented);
            FromLane<TN>(ldq, dq, a.QValues);
            FromLane<TN>(ldk, dk, a.KValues);
            FromLane<TN>(ldv, dv, a.KValues);
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>The gate's override: false takes the backward's composition, null the fused kernels where they apply (d ≤ 128).</summary>
    public bool? FusedBackward;

    private void BackwardIn<TS>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv, List<ulong> rented) where TS : struct, IQiyasFloat<TS>
    {
        if (FusedBackward != false && a.D <= FusedHeadMax) { BackwardFused<TS>(a, q, k, v, dO, dq, dk, dv, rented); return; }
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
        var scale = Qiyas.Ops.Find("scale");
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
}
