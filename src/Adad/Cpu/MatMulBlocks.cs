// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The block-scaled tiled product: C = A·Bᵀ for A (m × k) and B (n × k) both Q8_0 or both Q4_0, k in whole blocks of 32
// along each row (a linear layer's weights stored n × k). C_ij = Σ_blocks dA·dB·(Σ_32 qA·qB): every 32-term dot an
// exact integer, the block scales applied in float32 — gemv's semantics on a tile, nothing decoded to a float per value.
//   CPU: A's and B's blocks unpacked once into int8 rows (Q4_0: q − 8) and per-4-byte scales (the scale of the block
//        each int lane of an s8 dot covers, so any vector width works); tasks of 4 rows, a 4 × 2 register tile of float
//        vectors, per vector of k one s8 dot, converted, times sA·sB, added. Where the runtime has the s8 dot lane.
//   CUDA: the s8 tensor instruction at k = 32 (one per block per 16 × 8 tile): a block of 128 threads per 64 × 64 of C, a
//        warp 32 × 32; per k-block A's and B's 64 blocks staged into shared memory by 16-bit loads (a 34-byte Q8_0
//        block is 2-byte aligned only; Q4_0's nibbles widened to int8 − 8 on the way), rows 48 bytes apart (the
//        fragments' 32-bit loads free of bank conflicts), the scales in f32; each tile's s32 result converted and added
//        as acc += d·sA_row·sB_col.
// C comes out as float32 lanes; the callers encode it to the type by blocks.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>Whether the CPU has the block product (the s8 dot lane).</summary>
    public static bool BlockProductAvailable => S8Dot.IsSupported;

    /// <summary>C (float32 lanes in cLanes, m × n) = A·Bᵀ for A (m × k), B (n × k) both Q8_0 or both Q4_0, k a multiple of 32.</summary>
    public static string MatMulBlocks(string format, byte[] a, byte[] b, byte[] cLanes, int m, int n, int k)
    {
        if (format is not ("q8_0" or "q4_0") || k % 32 != 0) throw new NotSupportedException($"{format}: the block product takes Q8_0 or Q4_0 with k in whole blocks");
        if (!S8Dot.IsSupported) throw new NotSupportedException("the block product needs the s8 dot lane");
        var (qa, sa) = UnpackRows(format, a, m, k);
        var (qb, sb) = UnpackRows(format, b, n, k);
        Team.For((m + 3) / 4, 1, (long)m * n * k, new MatMulBlocksBody(qa, sa, qb, sb, cLanes, m, n, k));
        return "s8dp blocks";
    }

    /// <summary>The gate's override: true takes the block product wherever the view allows it, false never, null as measured.</summary>
    internal static bool? BlockProducts;

    private static readonly Dictionary<string, bool> s_blocksFaster = [];

    private static bool BlockView<TN>(ProductView v) where TN : struct, IAdad<TN> =>
        TN.Name is "q8_0" or "q4_0" && v.Batch == 1 && !v.TransA && v.TransB && v.K % 32 == 0 && BlockProductAvailable;

    /// <summary>The block product for the view (A·Bᵀ), C encoded to the type by blocks.</summary>
    public static string MatMulViewBlocks<TN>(ProductView v, byte[] a, byte[] b, byte[] c) where TN : struct, IAdad<TN>
    {
        var lanes = new byte[(long)v.M * v.N * 4];
        string lane = MatMulBlocks(TN.Name, a, b, lanes, v.M, v.N, v.K);
        var store = new BlocksStore(lanes, c, (long)v.M * v.N);
        TN.Elementwise(ref store);
        return lane;
    }

    // float32 lanes encoded into the block format on the team, a run of whole blocks a task.
    private readonly struct BlocksStore(byte[] lanes, byte[] output, long count) : ILaneVisitor
    {
        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            int per = TN.Granularity, chunk = 4096 / per * per;
            Team.For((int)((count + chunk - 1) / chunk), 1, count, new NarrowBody<TN, TLane>(lanes, output, count, chunk));
        }
    }

    // The block product or the strided route, whichever this processor measures faster at the shape and format (once,
    // as Measure.Faster settles it).
    private static bool UseBlocks<TN>(ProductView v, byte[] a, byte[] b, byte[] c) where TN : struct, IAdad<TN>
    {
        if (BlockProducts is bool forced) return forced;
        string key = $"{TN.Name}:{v.M}x{v.N}x{v.K}";
        lock (s_blocksFaster)
        {
            if (s_blocksFaster.TryGetValue(key, out bool blocks)) return blocks;
            var scratch = new byte[c.Length];
            var (tb, tr) = Measure.Faster(() => MatMulViewBlocks<TN>(v, a, b, scratch), () => MatMulViewRoute<TN>(v, a, b, scratch));
            blocks = s_blocksFaster[key] = tb <= tr;
            Note($"product {key} A·Bᵀ: block-scaled {tb:0.000} ms, decoded {tr:0.000} ms: {(blocks ? "block-scaled" : "decoded")}");
            return blocks;
        }
    }

    // Rows of blocks as int8 values and per-4-byte scales.
    private static (sbyte[] Q, float[] S) UnpackRows(string format, byte[] x, int rows, int k)
    {
        var q = new sbyte[(long)rows * k];
        var s = new float[(long)rows * k / 4];
        Team.For(rows, Math.Max(1, rows / (Threads * 4)), (long)rows * k, new UnpackRowsBody(format, x, q, s, k));
        return (q, s);
    }

    // Both bodies compiled at the final tier from the first call: the runtime's first tier ran the product 20× slower,
    // for the first rounds of a run, until the counted calls promoted it.
    private readonly struct UnpackRowsBody(string format, byte[] x, sbyte[] q, float[] s, int k) : Team.IBody
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int blocks = k / 32, bb = format == "q8_0" ? 34 : 18;
            for (int r = first; r < last; r++)
            {
                for (int bl = 0; bl < blocks; bl++)
                {
                    long at = (long)r * blocks + bl;
                    UnpackBlock(format, x.AsSpan((int)(at * bb), bb), out float d, q.AsSpan((int)(at * 32), 32));
                    s.AsSpan((int)(at * 8), 8).Fill(d);
                }
            }
        }
    }

    private readonly struct MatMulBlocksBody(sbyte[] qa, float[] sa, sbyte[] qb, float[] sb, byte[] cLanes, int m, int n, int k) : Team.IBody
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            int w = Vector<int>.Count, k4 = k / 4;
            var QA = MemoryMarshal.Cast<sbyte, int>(qa.AsSpan());
            var QB = MemoryMarshal.Cast<sbyte, int>(qb.AsSpan());
            var C = MemoryMarshal.Cast<byte, float>(cLanes.AsSpan());
            for (int t = first; t < last; t++)
            {
                int i0 = t * 4;
                int r0 = i0, r1 = Math.Min(i0 + 1, m - 1), r2 = Math.Min(i0 + 2, m - 1), r3 = Math.Min(i0 + 3, m - 1);
                for (int j0 = 0; j0 < n; j0 += 2)
                {
                    int c0 = j0, c1 = Math.Min(j0 + 1, n - 1);
                    Vector<float> a00 = default, a01 = default, a10 = default, a11 = default, a20 = default, a21 = default, a30 = default, a31 = default;
                    int v = 0;
                    for (; v + w <= k4; v += w)
                    {
                        var x0 = new Vector<int>(QA.Slice(r0 * k4 + v, w)); var x1 = new Vector<int>(QA.Slice(r1 * k4 + v, w));
                        var x2 = new Vector<int>(QA.Slice(r2 * k4 + v, w)); var x3 = new Vector<int>(QA.Slice(r3 * k4 + v, w));
                        var s0 = new Vector<float>(sa.AsSpan(r0 * k4 + v, w)); var s1 = new Vector<float>(sa.AsSpan(r1 * k4 + v, w));
                        var s2 = new Vector<float>(sa.AsSpan(r2 * k4 + v, w)); var s3 = new Vector<float>(sa.AsSpan(r3 * k4 + v, w));
                        var y0 = new Vector<int>(QB.Slice(c0 * k4 + v, w)); var y1 = new Vector<int>(QB.Slice(c1 * k4 + v, w));
                        var t0 = new Vector<float>(sb.AsSpan(c0 * k4 + v, w)); var t1 = new Vector<float>(sb.AsSpan(c1 * k4 + v, w));
                        a00 += Vector.ConvertToSingle(S8Dot.Fma(x0, y0, Vector<int>.Zero)) * (s0 * t0);
                        a01 += Vector.ConvertToSingle(S8Dot.Fma(x0, y1, Vector<int>.Zero)) * (s0 * t1);
                        a10 += Vector.ConvertToSingle(S8Dot.Fma(x1, y0, Vector<int>.Zero)) * (s1 * t0);
                        a11 += Vector.ConvertToSingle(S8Dot.Fma(x1, y1, Vector<int>.Zero)) * (s1 * t1);
                        a20 += Vector.ConvertToSingle(S8Dot.Fma(x2, y0, Vector<int>.Zero)) * (s2 * t0);
                        a21 += Vector.ConvertToSingle(S8Dot.Fma(x2, y1, Vector<int>.Zero)) * (s2 * t1);
                        a30 += Vector.ConvertToSingle(S8Dot.Fma(x3, y0, Vector<int>.Zero)) * (s3 * t0);
                        a31 += Vector.ConvertToSingle(S8Dot.Fma(x3, y1, Vector<int>.Zero)) * (s3 * t1);
                    }

                    Span<float> acc = [Vector.Sum(a00), Vector.Sum(a01), Vector.Sum(a10), Vector.Sum(a11), Vector.Sum(a20), Vector.Sum(a21), Vector.Sum(a30), Vector.Sum(a31)];
                    for (; v < k4; v++)                                         // a vector wider than what is left of k
                    {
                        int[] rows = [r0, r1, r2, r3], cols = [c0, c1];
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < 2; j++)
                            {
                                int dot = S8Dot.Fma(QA[rows[i] * k4 + v], QB[cols[j] * k4 + v], 0);
                                acc[i * 2 + j] += dot * (sa[rows[i] * k4 + v] * sb[cols[j] * k4 + v]);
                            }
                        }
                    }

                    for (int i = 0; i < 4 && i0 + i < m; i++)
                    {
                        for (int j = 0; j < 2 && j0 + j < n; j++) C[(i0 + i) * n + j0 + j] = acc[i * 2 + j];
                    }
                }
            }
        }
    }
}
