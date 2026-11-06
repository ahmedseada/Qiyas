// The device's products on views (Products.cs): each operand of a batch made contiguous by the strided-copy kernel
// (exact, any format), the product kernel on it, the result copied to its place (a contiguous slice: a device copy
// when the slice ends on a byte; otherwise the strided copy with a word-aligned output is the step past this).
namespace Qiyas;

public sealed partial class GpuOps
{
    /// <summary>C = op(A)·op(B) over the view's batches; returns the product's lane. A·Bᵀ of two Q8_0 / Q4_0 operands with k
    /// in whole blocks takes the block-scaled product where it measures faster (MatMulBlocks.cs).</summary>
    public string MatMulView<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IQiyasNumber<TN>
    {
        if (BlockView<TN>(v) && UseBlocks<TN>(v, a, b, c)) return MatMulViewBlocks<TN>(v, a, b, c);
        return MatMulViewRoute<TN>(v, a, b, c);
    }

    /// <summary>C = op(A)·op(B) by the strided copies and the product kernel.</summary>
    private string MatMulViewRoute<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        string lane = "";
        bool direct = v.Batch == 1 && !v.TransA && !v.TransB;
        if (f.IsBlock && !direct)
        {
            // A block format's values are addressed by blocks: the views are taken in float32 storage (the values decoded
            // once, no re-encoding of the operands' copies), the result encoded by blocks.
            var r = new List<ulong>();
            try
            {
                ulong la = Lanes(f, Lane.F32, a, v.ValuesA, r), lb = Lanes(f, Lane.F32, b, v.ValuesB, r), lc = gpu.Rent(v.ValuesC * 4);
                r.Add(lc);
                MatMulView<Float32>(v, la, lb, lc);
                Back(f, Lane.F32, lc, c, v.ValuesC);
                return "f32 (blocks through lane storage)";
            }
            finally { foreach (var p in r) gpu.Return(p); }
        }

        long bytesA = f.Words((long)v.M * v.K) * (f.WordBits / 8), bytesB = f.Words((long)v.K * v.N) * (f.WordBits / 8), bytesC = f.Words((long)v.M * v.N) * (f.WordBits / 8);
        ulong ca = direct ? a : gpu.Rent(bytesA), cb = direct ? b : gpu.Rent(bytesB), cc = direct ? c : gpu.Rent(bytesC);
        // Sub-byte results at batch offsets that do not end on a byte are OR-ed into a zeroed C (the bits placed one value at a time).
        bool bitwise = !direct && f.Bits < 8 && (v.StrideCOrDefault * f.Bits % 8 != 0 || (long)v.M * v.N * f.Bits % 8 != 0);
        if (bitwise) Gpu.Clear(c, f.Words(v.ValuesC) * (f.WordBits / 8));
        try
        {
            for (int i = 0; i < v.Batch; i++)
            {
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
            if (!direct) { gpu.Return(ca); gpu.Return(cb); gpu.Return(cc); }
        }

        return lane;
    }

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IQiyasNumber<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }
}
