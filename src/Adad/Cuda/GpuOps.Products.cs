// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's products on views (Products.cs): where the product takes the tensor or dot route at whole tiles, each
// batch's operands are read where they are stored (Aᵀ and Bᵀ by the packing's two orientations, a batch's offset on the
// pointer, C written in place); otherwise each operand of a batch made contiguous by the strided-copy kernel (exact,
// any format), the product kernel on it, the result copied to its place (a contiguous slice: a device copy when the
// slice ends on a byte; otherwise the strided copy with a word-aligned output is the step past this).
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>C = op(A)·op(B) over the view's batches; returns the product's lane. A·Bᵀ of two Q8_0 / Q4_0 operands with k
    /// in whole blocks takes the block-scaled product where it measures faster (MatMulBlocks.cs).</summary>
    public string MatMulView<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        if (BlockView<TN>(v) && UseBlocks<TN>(v, a, b, c)) return MatMulViewBlocks<TN>(v, a, b, c);
        return MatMulViewRoute<TN>(v, a, b, c);
    }

    /// <summary>C = op(A)·op(B): each batch's operands read where they are stored by the packing (the tensor and dot
    /// routes), or made contiguous by the strided copies; then the product kernel.</summary>
    private string MatMulViewRoute<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        string lane = "";
        bool direct = v.Batch == 1 && !v.TransA && !v.TransB;
        // A block format's values are addressed by blocks (a copy of a view's operand would re-encode them): every batch's
        // operands read where they are stored, transposed or at their offsets, by the product kernel itself.
        if (f.IsBlock) return MatMulStored<TN>(v, a, b, c, Lane.F32);
        if (MatMulBatched<TN>(v, a, b, c, out string batched)) return batched;

        long bytesA = f.Words((long)v.M * v.K) * (f.WordBits / 8), bytesB = f.Words((long)v.K * v.N) * (f.WordBits / 8), bytesC = f.Words((long)v.M * v.N) * (f.WordBits / 8);
        // A batch's operands read where they are stored (Aᵀ, Bᵀ, a batch's offset on the pointer) where the product's packing
        // reads them: the tensor and dot routes at whole tiles, each offset on 16 bytes (the kernels' widest loads).
        bool Aligned(long values) => values * f.Bits % 128 == 0;
        bool placeable = !direct && Aligned(v.StrideAOrDefault) && Aligned(v.StrideBOrDefault) && Aligned(v.Batch > 1 ? v.StrideCOrDefault : 0);
        ulong ca = 0, cb = 0, cc = 0;
        // Sub-byte results at batch offsets that do not end on a byte are OR-ed into a zeroed C (the bits placed one value at a time).
        bool bitwise = !direct && f.Bits < 8 && (v.StrideCOrDefault * f.Bits % 8 != 0 || (long)v.M * v.N * f.Bits % 8 != 0);
        bool cleared = false;
        try
        {
            for (int i = 0; i < v.Batch; i++)
            {
                if (placeable)
                {
                    ulong pa = a + (ulong)(i * v.StrideAOrDefault * f.Bits / 8), pb = b + (ulong)(i * v.StrideBOrDefault * f.Bits / 8);
                    var ((pLo, pHi), (qLo, qHi)) = RangePair<TN>(pa, v.M * v.K, pb, v.K * v.N);
                    var placed = new Bounds(pLo, pHi, qLo, qHi, v.K);
                    if (PackedRoute<TN>(placed, v.M, v.N, v.K))
                    {
                        lane = MatMul<TN>(pa, pb, c + (ulong)(i * v.StrideCOrDefault * f.Bits / 8), v.M, v.N, v.K, placed, v.TransA, v.TransB);
                        continue;
                    }
                }

                if (direct) { ca = a; cb = b; cc = c; }
                else if (ca == 0) { ca = gpu.Rent(bytesA); cb = gpu.Rent(bytesB); cc = gpu.Rent(bytesC); }
                if (bitwise && !cleared) { Gpu.Clear(c, f.Words(v.ValuesC) * (f.WordBits / 8)); cleared = true; }
                if (!direct)
                {
                    Layout<TN>(new LayoutArgs { Name = "strided", X = [], P = v.ViewA(i), Output = [], Count = v.M * v.K, InputCount = (int)v.ValuesA }, a, 0, ca);
                    Layout<TN>(new LayoutArgs { Name = "strided", X = [], P = v.ViewB(i), Output = [], Count = v.K * v.N, InputCount = (int)v.ValuesB }, b, 0, cb);
                }

                var ((aLo, aHi), (bLo, bHi)) = RangePair<TN>(ca, v.M * v.K, cb, v.K * v.N);
                lane = MatMul<TN>(ca, cb, cc, v.M, v.N, v.K, new Bounds(aLo, aHi, bLo, bHi, v.K));
                if (!direct)
                {
                    long offsetBits = i * v.StrideCOrDefault * f.Bits, lengthBits = (long)v.M * v.N * f.Bits;
                    if (bitwise)
                    {
                        var place = gpu.Cached($"place_bits:{f.Name}", (out string e) => Ptx.PlaceBits(gpu, f, out e));
                        Gpu.Launch(place, Grid((long)v.M * v.N, 256), 1, 256, 1, 0, cc, c, Gpu.U(v.M * v.N), Gpu.U((int)(i * v.StrideCOrDefault)));
                    }
                    else Gpu.Copy(cc, c + (ulong)(offsetBits / 8), lengthBits / 8);
                }
            }
        }
        finally
        {
            if (!direct && ca != 0) { gpu.Return(ca); gpu.Return(cb); gpu.Return(cc); }
        }

        return lane;
    }

    /// <summary>Every batch of the view in one launch (grid z the batches), each batch's A, B and C read and written where
    /// they are stored, as the per-batch launches read them: the staged kernel (Ptx.Gemm2) for a float type whose product
    /// lane is its own storage (Aᵀ read as stored too, by its k-major tile; a single batch so as well, where its tiles fill
    /// the device without split-k: Aᵀ in place and Bᵀ alone copied into a k × n B, where the copies route copied both
    /// operands and C), or the tensor kernel
    /// (Ptx.GemmMma) for a 16-bit float in its own kind at whole tiles (operands as stored, Aᵀ and B k-major by its
    /// in-place reads, C by its epilogue). The batches' C apart (no batch overwrites another's: an order one launch would
    /// not keep) and every batch's operands on 16 bytes (the kernels' copies). One launch at a time, a small product's tiles
    /// leave most of the device idle (16 batches of a 512² C: 16 tiles a launch on 70 multiprocessors); together they fill
    /// it. The same kernel on the same tiles, so the same bits. False where the view is not such a one.</summary>
    private bool MatMulBatched<TN>(ProductView v, ulong a, ulong b, ulong c, out string laneName) where TN : struct, IAdad<TN>
    {
        laneName = "";
        var f = Fmt.Of<TN>();
        if ((v.Batch < 2 && !v.TransA && !v.TransB) || v.Batch > 65535 || !f.IsFloat || f.IsBlock || v.M == 0 || v.N == 0 || v.K == 0 || v.StrideCOrDefault < (long)v.M * v.N) return false;
        long bytesA = v.StrideAOrDefault * f.Bits / 8, bytesB = v.StrideBOrDefault * f.Bits / 8, bytesC = v.StrideCOrDefault * f.Bits / 8;
        if ((a | b | c) % 16 != 0 || bytesA % 16 != 0 || bytesB % 16 != 0 || bytesC % 16 != 0) return false;
        var ((aLo, aHi), (bLo, bHi)) = RangePair<TN>(a, v.M * v.K, b, v.K * v.N);
        string name = ProductLane<TN>(new Bounds(aLo, aHi, bLo, bHi, v.K));
        if (name.EndsWith("mma", StringComparison.Ordinal))
        {
            if (v.Batch < 2) return false;                                    // a single batch: MatMul's own route reads it
            // As MatMul's in-place 16-bit route: whole tiles, the type the kind's own, no padding of k.
            if (v.M % 128 != 0 || v.N % 256 != 0 || v.K % 128 != 0 || InPlace16 == false || Epilogue == false) return false;
            var kind = KindOf(name, v.K);
            if (kind.Bits != 16 || KPadded(kind, v.K) != v.K || !f.Is(kind == Ptx.Kind.F16 ? "float16" : "bfloat16")) return false;
            var mfn = MmaFunction(kind, out uint shared, out int bn, f, v.TransA, !v.TransB, batched: true);
            Gpu.LaunchZ(mfn, Grid(v.N, bn), Grid(v.M, 128), (uint)v.Batch, Ptx.GemmThreads, 1, shared, a, b, c, Gpu.U(v.M), Gpu.U(v.N), Gpu.U(v.K), Gpu.U(v.K),
                (ulong)bytesA, (ulong)bytesB, (ulong)bytesC);
            Gpu.Flush();
            laneName = name;
            return true;
        }

        var lane = LaneOf(name);
        if ((v.TransB && v.Batch > 1) || Staged == false || v.K % 4 != 0 || v.N % 4 != 0 || (v.TransA && v.M % 4 != 0)) return false;
        if (lane.Wide || f.Natural != lane || f.Bits != lane.Size * 8 || f.IsFloat != lane.IsFloat) return false;
        if (v.Batch < 2 && SplitK(v.M, v.N, v.K, out _) > 1) return false;     // too few tiles: the split-k route, on copies
        int stages = Gemm2Stages(lane, a, b, c, v.M, v.N, v.K);
        if (stages == 0) return false;
        bool ta = v.TransA;
        int sharedBytes = Ptx.Gemm2Shared(stages, ta);
        var fn = gpu.Cached($"gemm2:{lane.Name}:{stages}:b:{ta}", (out string e) => Ptx.Gemm2(gpu, lane, stages, out e, batched: true, transA: ta));
        gpu.AllowShared(fn, sharedBytes);
        // Bᵀ (a single batch): copied into a k × n B first, its tile's copies taking 16 bytes of a k-row each.
        ulong bk = v.TransB ? gpu.Rent(Math.Max(16, (long)v.K * v.N * lane.Size)) : 0;
        try
        {
            if (bk != 0) Layout<TN>(new LayoutArgs { Name = "strided", X = [], P = v.ViewB(0), Output = [], Count = v.K * v.N, InputCount = (int)v.ValuesB }, b, 0, bk);
            Gpu.LaunchZ(fn, Grid(v.N, 128), Grid(v.M, 128), (uint)v.Batch, Ptx.GemmThreads, 1, Dynamic(sharedBytes), a, bk != 0 ? bk : b, c, Gpu.U(v.M), Gpu.U(v.N), Gpu.U(v.K),
                (ulong)bytesA, (ulong)bytesB, (ulong)bytesC);
            Gpu.Flush();
        }
        finally
        {
            if (bk != 0) gpu.Return(bk);
        }

        laneName = name;
        return true;
    }

    /// <summary>C = the epilogue's y of op(A)·op(B) (ProductEpilogue): every batch's product into the lane's sums (the
    /// operands read where they are stored), the second product's into sums of its own, then the store (Ptx.ProductStore;
    /// the sums kept first where they are). A float type's lane is its float lane; another type's its products' by the
    /// bounds of every value the store makes (EpilogueLane); returns the lane's name.</summary>
    public string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IAdad<TN>
    {
        var terms = t.Validate(e, v);
        var f = Fmt.Of<TN>();
        e.ValidateType(f.Name, f.IsFloat);
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return f.IsFloat ? RowLane(f).Name : f.Natural.Name;
        var lane = f.IsFloat ? RowLane(f) : EpilogueLane<TN>(v, e, terms, a, b, c, t);
        ulong sums = gpu.Rent(Math.Max(16, v.ValuesC * lane.Size));
        try
        {
            ProductStored(v, lane, a, f, 0, b, f, 0, sums, null);
            EpilogueStore(f, lane, v, e, terms, t, sums, c, 1, v.N);
        }
        finally { gpu.Return(sums); }

        return lane.Name;
    }

    /// <summary>y = the epilogue's y of x · Wᵀ, W the scaled weight's values made in the lane as the stored GEMM reads each
    /// code (GemmStored.Scales); returns the lane's name.</summary>
    public string MatMulScaled<TN, TW>(ScaledWeight w, int m, ProductEpilogue e, ulong x, ulong codes, ulong scales, ulong zeros, ulong y, ProductTerms<ulong> t)
        where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        var v = w.View(m);
        var terms = t.Validate(e, v);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (v.M == 0 || v.N == 0) return lane.Name;
        ulong sums = gpu.Rent(Math.Max(16, v.ValuesC * lane.Size));
        try
        {
            ProductStored(v, lane, x, f, 0, codes, Fmt.Of<TW>(), 0, sums, null, new GemmScale(scales, zeros, f, w.BlockRows, w.BlockColumns, w.ScaleColumns));
            EpilogueStore(f, lane, v, e, terms, t, sums, y, 1, v.N);
        }
        finally { gpu.Return(sums); }

        return lane.Name;
    }

    /// <summary>output = the scaled weight's values as TN (Ptx.ExpandScaled); returns the lane's name.</summary>
    public string ExpandScaled<TN, TW>(ScaledWeight w, ulong codes, ulong scales, ulong zeros, ulong output) where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        Fmt f = Fmt.Of<TN>(), fw = Fmt.Of<TW>();
        var lane = RowLane(f);
        var fn = gpu.Cached($"expand_scaled:{fw.Name}:{f.Name}:{lane.Name}:{zeros != 0}", (out string e) => Ptx.ExpandScaled(gpu, f, fw, lane, zeros != 0, out e));
        LaunchEach(fn, f, w.Values, codes, scales, zeros, output, Gpu.U((int)w.Values), Gpu.U(w.K), Gpu.U(w.BlockRows), Gpu.U(w.BlockColumns), Gpu.U(w.ScaleColumns));
        return lane.Name;
    }

    /// <summary>The scaled weight's codes, scales and zero points from its values (IProducts.Quantize; Ptx.QuantizeRange,
    /// QuantizeZeros, QuantizeCodes): the scales and zero points lane values, stored as TN before the next pass reads them.
    /// Returns the lane's name.</summary>
    public string Quantize<TN, TW>(ScaledWeight w, ulong values, ulong codes, ulong scales, ulong zeros) where TN : struct, IFloatAdad<TN> where TW : struct, IAdad<TW>
    {
        w.Validate();
        Fmt f = Fmt.Of<TN>(), fw = Fmt.Of<TW>();
        ScaledWeight.ValidateCodes(fw, zeros != 0);
        var lane = RowLane(f);
        if (w.Values == 0) return lane.Name;
        bool z = zeros != 0, own = f.Is("float32") || f.Is("float64");
        ulong Min() => lane.Wide ? Gpu.D(TW.Min) : Gpu.F((float)TW.Min);
        ulong Max() => lane.Wide ? Gpu.D(TW.Max) : Gpu.F((float)TW.Max);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = gpu.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong sl = own ? scales : Rent(w.Scales * lane.Size), lows = z ? Rent(w.Scales * lane.Size) : 0;
            var range = gpu.Cached($"quantize_range:{f.Name}:{lane.Name}:{z}", (out string e) => Ptx.QuantizeRange(gpu, f, lane, z, out e));
            Gpu.Launch(range, (uint)w.Scales, 1, 256, 1, 0, values, sl, lows, Gpu.U(w.N), Gpu.U(w.K), Gpu.U(w.BlockRows), Gpu.U(w.BlockColumns), Gpu.U(w.ScaleColumns), Min(), Max());
            FromLane(f, sl, scales, w.Scales);
            if (z)
            {
                ulong zl = own ? zeros : Rent(w.Scales * lane.Size);
                var zfn = gpu.Cached($"quantize_zeros:{f.Name}:{lane.Name}", (out string e) => Ptx.QuantizeZeros(gpu, f, lane, out e));
                Gpu.Launch(zfn, Grid(w.Scales, 256), 1, 256, 1, 0, scales, lows, zl, Gpu.U((int)w.Scales), Min(), Max());
                FromLane(f, zl, zeros, w.Scales);
            }

            var cfn = gpu.Cached($"quantize_codes:{f.Name}:{fw.Name}:{lane.Name}:{z}", (out string e) => Ptx.QuantizeCodes(gpu, f, fw, lane, z, out e));
            LaunchEach(cfn, fw, w.Values, values, scales, zeros, codes, Gpu.U((int)w.Values), Gpu.U(w.K), Gpu.U(w.BlockRows), Gpu.U(w.BlockColumns), Gpu.U(w.ScaleColumns), Min(), Max());
        }
        finally { foreach (var p in rented) gpu.Return(p); }

        return lane.Name;
    }

    // After a product's sums (lane storage, the view's C contiguous): the second product's into sums of its own (every
    // batch's operands read where they are stored), then the store (the sums kept first where they are).
    // A type that is not float: its products' lane by the bounds of every value the store makes (ProductEpilogue.IntegerSums,
    // the ranges read on the device); an unsigned type whose values may fall below 0 takes the float64 lane (ExactSums).
    private Lane EpilogueLane<TN>(ProductView v, ProductEpilogue e, StoreTerms terms, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        (double, double) R(ulong x, long n) => Range<TN>(x, (int)n);
        var w = e.Second;
        var (bounds, negative) = e.IntegerSums(f.Name, f.Signed, v.K, R(a, v.ValuesA), R(b, v.ValuesB), w is null ? null : R(t.SecondA, w.ValuesA),
            w is null ? null : R(t.SecondB, w.ValuesB), terms.Bias ? R(t.Bias, v.N) : null, terms.Beta ? R(c, v.ValuesC) : null,
            terms.Residual ? R(t.Residual, v.ValuesC) : null, terms.Grad is not null ? R(t.ActInputs, v.ValuesC) : null);
        return negative && !f.Signed ? Lane.F64 : SumLane(f, bounds);
    }

    private void EpilogueStore(Fmt f, Lane lane, ProductView v, ProductEpilogue e, StoreTerms terms, ProductTerms<ulong> t, ulong sums, ulong c, int cols, int oc)
    {
        ulong second = 0;
        try
        {
            if (terms.Second)
            {
                second = gpu.Rent(Math.Max(16, v.ValuesC * lane.Size));
                ProductStored(e.Second!, lane, t.SecondA, f, 0, t.SecondB, f, 0, second, null);
            }

            if (terms.Keep) ProductStore(f, lane, terms.KeepPass, sums, second, t.Bias, 0, 0, t.Sums, v.ValuesC, cols, oc, e.ActA, e.Beta, e.ActB, before: c);
            ProductStore(f, lane, terms, sums, second, t.Bias, t.Residual, t.ActInputs, c, v.ValuesC, cols, oc, e.ActA, e.Beta, e.ActB);
        }
        finally
        {
            if (second != 0) gpu.Return(second);
        }
    }

    /// <summary>y (count values of f) = the store terms' value of the lane sums (Ptx.ProductStore): the bias the channel
    /// (at / cols) mod oc's, an absent term's pointer 0; the formula's parameters and β as lane values.</summary>
    private void ProductStore(Fmt f, Lane lane, StoreTerms t, ulong sums, ulong second, ulong bias, ulong residual, ulong actInputs, ulong y, long count, int cols, int oc,
        double actA, double beta, double actB = 0, ulong before = 0)
    {
        var fn = gpu.Cached($"product_store:{f.Name}:{lane.Name}:{t.Key}", (out string e) => Ptx.ProductStore(gpu, f, lane, t, out e));
        LaunchEach(fn, f, count, sums, second, bias, residual, actInputs, y, before == 0 ? y : before, Gpu.U((int)count), Gpu.U(cols), Gpu.U(oc), LaneScalar(lane, actA), LaneScalar(lane, actB),
            LaneScalar(lane, beta));
    }

    // A scalar as a lane's value, a kernel's .{lane.Mem} parameter (an integer lane's: the integer, by its bits).
    private static ulong LaneScalar(Lane lane, double v) => lane.IsFloat ? (lane.Wide ? Gpu.D(v) : Gpu.F((float)v)) : lane.Wide ? (ulong)(long)v : (uint)(int)(long)v;

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }
}
