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

    /// <summary>C = the epilogue's y of op(A)·op(B) (ProductEpilogue): every batch's product into the float lane's sums
    /// (the operands read where they are stored), the low-rank term's into sums of its own, then the store
    /// (Ptx.ProductStore; aux first where it is written); returns the lane's name.</summary>
    public string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, ulong a, ulong b, ulong c, ProductTerms<ulong> t) where TN : struct, IFloatAdad<TN>
    {
        var terms = t.Validate(e, v);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return lane.Name;
        long span = (long)v.M * v.N;
        ulong sums = gpu.Rent(Math.Max(16, v.ValuesC * lane.Size)), low = 0;
        try
        {
            ProductStored(v, lane, a, f, 0, b, f, 0, sums, null);
            if (terms.Low)
            {
                low = gpu.Rent(Math.Max(16, span * lane.Size));
                ProductStored(new ProductView(v.M, v.N, e.Rank, false, e.TransV), lane, t.U, f, 0, t.V, f, 0, low, null);
            }

            if (terms.Aux) ProductStore(f, lane, terms.Input, sums, low, t.Bias, 0, 0, t.Aux, v.ValuesC, 1, v.N, e.ActA, e.Beta);
            ProductStore(f, lane, terms, sums, low, t.Bias, t.Residual, t.Aux, c, v.ValuesC, 1, v.N, e.ActA, e.Beta);
        }
        finally
        {
            gpu.Return(sums);
            if (low != 0) gpu.Return(low);
        }

        return lane.Name;
    }

    /// <summary>y (count values of f) = the store terms' value of the lane sums (Ptx.ProductStore): the bias the channel
    /// (at / cols) mod oc's, an absent term's pointer 0.</summary>
    private void ProductStore(Fmt f, Lane lane, StoreTerms t, ulong sums, ulong low, ulong bias, ulong residual, ulong aux, ulong y, long count, int cols, int oc, double actA, double beta)
    {
        var fn = gpu.Cached($"product_store:{f.Name}:{lane.Name}:{t.Key}", (out string e) => Ptx.ProductStore(gpu, f, lane, t, out e));
        LaunchEach(fn, f, count, sums, low, bias, residual, aux, y, Gpu.U((int)count), Gpu.U(cols), Gpu.U(oc),
            lane.Wide ? Gpu.D(actA) : Gpu.F((float)actA), lane.Wide ? Gpu.D(beta) : Gpu.F((float)beta));
    }

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IAdad<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }
}
