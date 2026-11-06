// Products on views: C = op(A)·op(B) over batches, where op transposes a stored operand, and each batch is at a
// stride in values; the matmul backward passes dA = dC·Bᵀ and dB = Aᵀ·dC. A view is made contiguous by the strided
// copy (Layout.cs: exact in any type, any offset) and the product is Qiyas's product kernel on it — the same lanes,
// the same bounds rule, the same proofs. Fusing the transposition into the product's packing is the step past this.
using System.Buffers;

namespace Qiyas;

/// <summary>A product on views: A stored m × k (k × m when TransA), B stored k × n (n × k when TransB), C m × n;
/// `Batch` products at strides StrideA, StrideB, StrideC (in values; 0 broadcasts the operand across the batch).</summary>
public sealed record ProductView(int M, int N, int K, bool TransA, bool TransB, int Batch = 1, long StrideA = 0, long StrideB = 0, long StrideC = 0)
{
    public long StrideAOrDefault => Batch > 1 ? StrideA : 0;
    public long StrideBOrDefault => Batch > 1 ? StrideB : 0;
    public long StrideCOrDefault => Batch > 1 ? StrideC : (long)M * N;

    /// <summary>The strided view that makes operand A of batch i contiguous (m × k).</summary>
    public int[] ViewA(int i) => TransA ? [1, 1, M, K, 0, 0, 1, M, (int)(i * StrideAOrDefault)] : [1, 1, M, K, 0, 0, K, 1, (int)(i * StrideAOrDefault)];

    /// <summary>The strided view that makes operand B of batch i contiguous (k × n).</summary>
    public int[] ViewB(int i) => TransB ? [1, 1, K, N, 0, 0, 1, K, (int)(i * StrideBOrDefault)] : [1, 1, K, N, 0, 0, N, 1, (int)(i * StrideBOrDefault)];

    public long ValuesA => (TransA ? (long)K * M : (long)M * K) + (Batch - 1) * StrideAOrDefault;
    public long ValuesB => (TransB ? (long)N * K : (long)K * N) + (Batch - 1) * StrideBOrDefault;
    public long ValuesC => (long)M * N + (Batch - 1) * StrideCOrDefault;

    /// <summary>The float64 reference: every batch's product of the operands as viewed.</summary>
    public double[] Reference(double[] a, double[] b)
    {
        var c = new double[ValuesC];
        for (int i = 0; i < Batch; i++)
        {
            var ai = new double[M * K];
            var bi = new double[K * N];
            long baseA = i * StrideAOrDefault, baseB = i * StrideBOrDefault;
            for (int r = 0; r < M; r++) for (int q = 0; q < K; q++) ai[r * K + q] = TransA ? a[baseA + q * M + r] : a[baseA + r * K + q];
            for (int q = 0; q < K; q++) for (int s = 0; s < N; s++) bi[q * N + s] = TransB ? b[baseB + s * K + q] : b[baseB + q * N + s];
            var ci = global::References.MatMul(ai, bi, M, N, K);
            Array.Copy(ci, 0, c, i * StrideCOrDefault, M * N);
        }

        return c;
    }
}

internal static partial class Kernels
{
    private static int BytesOf<TN>(long values) where TN : struct, IQiyasNumber<TN> => (int)((values * TN.Bits + 7) / 8);

    /// <summary>C = op(A)·op(B) over the view's batches on the CPU; returns the product's lane.</summary>
    public static string MatMulView<TN>(ProductView v, byte[] a, byte[] b, byte[] c) where TN : struct, IQiyasNumber<TN>
    {
        string lane = "";
        bool direct = v.Batch == 1 && !v.TransA && !v.TransB;
        var ca = direct ? a : new byte[BytesOf<TN>((long)v.M * v.K)];
        var cb = direct ? b : new byte[BytesOf<TN>((long)v.K * v.N)];
        var cc = direct ? c : new byte[BytesOf<TN>((long)v.M * v.N)];
        for (int i = 0; i < v.Batch; i++)
        {
            if (!direct)
            {
                Layout<TN>(new LayoutArgs { Name = "strided", X = a, P = v.ViewA(i), Output = ca, Count = v.M * v.K, InputCount = (int)v.ValuesA });
                Layout<TN>(new LayoutArgs { Name = "strided", X = b, P = v.ViewB(i), Output = cb, Count = v.K * v.N, InputCount = (int)v.ValuesB });
            }

            var (aLo, aHi) = TN.Range(ca, v.M * v.K);
            var (bLo, bHi) = TN.Range(cb, v.K * v.N);
            var product = new MatMulVisitor(ca, cb, cc, v.M, v.N, v.K);
            TN.Sum(ref product, new Bounds(aLo, aHi, bLo, bHi, v.K));
            lane = product.Lane;
            if (!direct) CopyValues<TN>(cc, 0, c, i * v.StrideCOrDefault, (long)v.M * v.N);
        }

        return lane;
    }

    /// <summary>dA = dC·Bᵀ (m × k) and dB = Aᵀ·dC (k × n) for C = A·B (A m × k, B k × n, dC m × n); returns dA's lane.</summary>
    public static string MatMulBackward<TN>(byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) where TN : struct, IQiyasNumber<TN>
    {
        string lane = MatMulView<TN>(new ProductView(m, k, n, false, true), dc, b, da);
        MatMulView<TN>(new ProductView(k, n, m, true, false), a, dc, db);
        return lane;
    }

    /// <summary>count values of src from srcOffset into dst at dstOffset (any offsets, through the lane).</summary>
    public static void CopyValues<TN>(byte[] src, long srcOffset, byte[] dst, long dstOffset, long count) where TN : struct, IQiyasNumber<TN>
    {
        var v = new CopyVisitor(src, srcOffset, dst, dstOffset, count);
        TN.Elementwise(ref v);
    }

    private struct CopyVisitor(byte[] src, long srcOffset, byte[] dst, long dstOffset, long count) : ILaneVisitor
    {
        public readonly void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            var buffer = ArrayPool<TLane>.Shared.Rent(4096);
            try
            {
                for (long at = 0; at < count; at += 4096)
                {
                    int len = (int)Math.Min(4096, count - at);
                    TN.Load(src, (int)(srcOffset + at), buffer.AsSpan(0, len));
                    TN.Store(buffer.AsSpan(0, len), dst, (int)(dstOffset + at));
                }
            }
            finally
            {
                ArrayPool<TLane>.Shared.Return(buffer);
            }
        }
    }
}
