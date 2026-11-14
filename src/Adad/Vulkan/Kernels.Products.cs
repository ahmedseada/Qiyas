// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The products on Vulkan: C = op(A)·op(B) over a view's batches by workgroup tiles. A workgroup of 16 × 16 invocations
// holds a 64 × 64 tile of C, each invocation 4 × 4 of it, or (micro 8) a 128 × 128 tile, 8 × 8 an invocation (where both
// sides of C span a large tile and the lane is 32-bit: VulkanOps.Micro); A's and B's 16-deep slices are read where they are stored
// (Codec.UnpackAt: transposed or at a batch's offset by the index alone) into workgroup memory in the product's lane,
// and each value of C accumulates its k terms in order by fused multiply-adds (the integer lanes exactly). C is the
// lane's values: written as C where the lane is the type's storage, otherwise into lane storage and stored as the type
// once (FromLanesKeep, which keeps the values between the batches' regions). The convolution's product is the same tile
// loop with B gathered from the image (MatMulConv).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int TileM = 64, TileN = 64, TileK = 16, TileSide = 16;

    /// <summary>The side of C's tile a workgroup holds for <paramref name="micro"/> values a side an invocation.</summary>
    public static int Tile(int micro) => TileSide * micro;

    /// <summary>The product kernel, A and B of one format. Arguments: a, b, c (lane values at C's positions), m, n, k,
    /// strideA, strideB, strideC (values), tilesM, offsetA, offsetB (values: A's and B's first), and with split kslice;
    /// workgroups (⌈n/tile⌉, ⌈m/tile⌉ · batch), tile = Tile(micro). split (one product, k in slices): a batch is a slice of
    /// k, [z·kslice, (z + 1)·kslice), its partial C at z·strideC (JoinSlices adds the slices in order).</summary>
    public static SpvKernel MatMul(Vulkan vk, Fmt f, Lane lane, bool transA, bool transB, int micro = 4, bool split = false) => MatMul(vk, f, f, lane, transA, transB, micro, split);

    /// <summary>The product kernel with A stored as fa and B as fb (attention's P in lane storage against V as stored). runsA,
    /// runsB (a block format, its stored rows, offsets and strides in whole runs of 8): the operand's slice read as runs of
    /// an invocation's values consecutive where it is stored, each decoded with one read of its block's scales
    /// (Codec.UnpackRun: the same values as one at a time).</summary>
    public static SpvKernel MatMul(Vulkan vk, Fmt fa, Fmt fb, Lane lane, bool transA, bool transB, int micro = 4, bool split = false, bool runsA = false, bool runsB = false)
    {
        string formats = fa.Name == fb.Name ? fa.Name : $"{fa.Name}_{fb.Name}";
        var b = new Spv($"matmul_{formats}_{lane.Name}{(transA ? "_ta" : "")}{(transB ? "_tb" : "")}{(micro == 4 ? "" : $"_x{micro}")}{(split ? "_k" : "")}{(runsA ? "_ra" : "")}{(runsB ? "_rb" : "")}",
            TileSide, vk.Controls, TileSide);
        var pa = b.Arg(0);
        var pb = b.Arg(1);
        var pc = b.Arg(2);
        var M = b.Arg(3, K.U32);
        var N = b.Arg(4, K.U32);
        var Kd = b.Arg(5, K.U32);
        var strideA = b.Arg(6, K.U32);
        var strideB = b.Arg(7, K.U32);
        var strideC = b.Arg(8, K.U32);
        var tilesM = b.Arg(9, K.U32);
        var offsetA = b.Arg(10, K.U32);
        var offsetB = b.Arg(11, K.U32);
        var gy = b.GroupY;
        var batch = gy / tilesM;
        var row0 = (gy - batch * tilesM) * Tile(micro);
        var baseA = split ? offsetA : batch * strideA + offsetA;
        var baseB = split ? offsetB : batch * strideB + offsetB;
        var kslice = split ? b.Arg(12, K.U32) : Kd;
        var kBegin = split ? batch * kslice : b.U(0);
        var kEnd = split ? b.Min(Kd, kBegin + kslice) : Kd;
        int run = TileK * Tile(micro) / (TileSide * TileSide);
        ProductTiles(b, lane, transA, transB, M, N, kBegin, kEnd, row0, pc, batch * strideC,
            (gr, gk, inside) => Codec.UnpackAt(b, fa, pa, baseA + b.Select(inside, transA ? gk * M + gr : gr * Kd + gk, b.U(0)), lane),
            (gk, gc, inside) => Codec.UnpackAt(b, fb, pb, baseB + b.Select(inside, transB ? gc * Kd + gk : gk * N + gc, b.U(0)), lane), micro,
            runsA ? (gr, gk, inside) => Run(b, fa, pa, baseA + b.Select(inside, transA ? gk * M + gr : gr * Kd + gk, b.U(0)), run, lane) : null,
            runsB ? (gk, gc, inside) => Run(b, fb, pb, baseB + b.Select(inside, transB ? gc * Kd + gk : gk * N + gc, b.U(0)), run, lane) : null);
        return b.Build();
    }

    /// <summary>Whether a format's values are read as runs by RunsWords: a type of one value a 32- or 64-bit word.</summary>
    public static bool RunsWords(Fmt f) => !f.IsBlock && f.Per == 1 && (f.WordBits == 32 || f.WordBits == 64);

    // `count` consecutive values from index (a multiple of count, the array's runs aligned to their bytes) as lane values: a
    // block format's by Codec.UnpackRun; a type of one value a word by loads of 4 words (16 or 32 bytes) at a time.
    private static V[] Run(Spv b, Fmt f, V basePtr, V index, int count, Lane lane)
    {
        if (f.IsBlock) return Codec.UnpackRun(b, f, basePtr, index, count, lane);
        int wb = f.WordBits / 8;
        var kind = f.WordBits == 64 ? K.U64 : K.U32;
        var values = new V[count];
        for (int at = 0; at < count; at += 4)
        {
            int n = Math.Min(4, count - at);
            var words = n == 1 ? [b.Load(kind, b.At(basePtr, index + (double)at, wb))] : b.LoadN(kind, n, b.At(basePtr, index + (double)at, wb));
            for (int i = 0; i < n; i++) values[at + i] = Codec.Unpack(b, f, words[i], 0, lane);
        }

        return values;
    }

    /// <summary>The slices' partial products added in order into C: c[i] = Σ_z parts[z·count + i] in the lane. Arguments:
    /// parts, c, count, slices; a thread per value.</summary>
    public static SpvKernel JoinSlices(Vulkan vk, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"join_slices_{lane.Name}", Threads, vk.Controls);
        var parts = b.Arg(0);
        var pc = b.Arg(1);
        var count = b.Arg(2, K.U32);
        var slices = b.Arg(3, K.U32);
        var gid = b.GlobalX;
        b.If(gid < count, () =>
        {
            var sum = b.Local(b.Load(k, b.At(parts, gid, lane.Size)));
            b.For(b.U(1), slices, z => sum.V = sum.V + b.Load(k, b.At(parts, b.Convert(z, K.U64) * b.Convert(count, K.U64) + b.Convert(gid, K.U64), lane.Size)));
            b.Store(b.At(pc, gid, lane.Size), sum.V);
        });
        return b.Build();
    }

    /// <summary>The convolution's product (implicit GEMM: no columns in memory): A (m × k, stored as fa) times the image's
    /// columns (im2col of x, stored as fb, gathered by the index map on the load: Im2colAt), or, transB, times their
    /// transpose (dw = dy · colsᵀ). Arguments: a, x, c (lane values), m, n, k, then im2col's parameters as the kernels take
    /// them (LayoutArgs.KernelP of ConvArgs.P: the four pads, the dilations, OH and OW) two a slot, then tilesM and kslice;
    /// workgroups (⌈n/tile⌉, ⌈m/tile⌉ · slices): slice z's k in [z·kslice, (z + 1)·kslice), its partial C at z·m·n
    /// (one slice: C itself).</summary>
    public static SpvKernel MatMulConv(Vulkan vk, Fmt fa, Fmt fb, Lane lane, bool transB, int micro = 4)
    {
        string formats = fa.Name == fb.Name ? fa.Name : $"{fa.Name}_{fb.Name}";
        var b = new Spv($"matmul_conv_{formats}_{lane.Name}{(transB ? "_tb" : "")}{(micro == 4 ? "" : $"_x{micro}")}", TileSide, vk.Controls, TileSide);
        var pa = b.Arg(0);
        var px = b.Arg(1);
        var pc = b.Arg(2);
        var M = b.Arg(3, K.U32);
        var N = b.Arg(4, K.U32);
        var Kd = b.Arg(5, K.U32);
        var p = new V[15];
        for (int i = 0; i < 15; i++)
        {
            var raw = b.Arg(6 + i / 2);
            p[i] = b.Convert(i % 2 == 0 ? raw : raw >> 32, K.U32);
        }

        var zero = b.C(Kinds.Of(lane), 0);
        var tilesM = b.Arg(14, K.U32);
        var kslice = b.Arg(15, K.U32);
        var z = b.GroupY / tilesM;
        var kBegin = z * kslice;
        ProductTiles(b, lane, false, transB, M, N, kBegin, b.Min(Kd, kBegin + kslice), (b.GroupY - z * tilesM) * Tile(micro), pc, z * M * N,
            (gr, gk, inside) => Codec.UnpackAt(b, fa, pa, b.Select(inside, gr * Kd + gk, b.U(0)), lane),
            (gk, gc, inside) =>
            {
                // B(k, c) = cols[k][c], or for transB cols[c][k]: column row r, column (pixel) q.
                var (src, image) = Im2colAt(b, p, transB ? gc : gk, transB ? gk : gc);
                var ok = inside & image;
                return b.Select(ok, Codec.UnpackAt(b, fb, px, b.Select(ok, src, b.U(0)), lane), zero);
            }, micro);
        return b.Build();
    }

    // C(r, c) = Σ_k A(r, k)·B(k, c) for the workgroup's tile (rows from row0, columns from the workgroup's x), into lane
    // values at pc + (baseC + r·N + c)·size: A's and B's 16-deep slices loaded through loadA(row, k, inside) and loadB(k,
    // column, inside) (a value at a safe place when not inside; 0 is taken there) into workgroup memory, consecutive
    // invocations along the operand's stored rows, the next slice's values read into registers while the current one's
    // multiply-adds run; each value of C its k terms in order by fused multiply-adds.
    // runA, runB (null: a value at a time): an invocation's `loads` values of the slice as one run along the operand's stored
    // rows (A: along k, or along m for Aᵀ; B: along n, or along k for Bᵀ), from its first (row, k) or (k, column); the run
    // inside or outside as a whole (the caller's rows, offsets and strides in whole runs).
    private static void ProductTiles(Spv b, Lane lane, bool transA, bool transB, V M, V N, V kBegin, V kEnd, V row0, V pc, V baseC,
        Func<V, V, V, V> loadA, Func<V, V, V, V> loadB, int micro = 4, Func<V, V, V, V[]>? runA = null, Func<V, V, V, V[]>? runB = null)
    {
        var k = Kinds.Of(lane);
        int tile = Tile(micro), shift = System.Numerics.BitOperations.Log2((uint)tile);
        var tx = b.LocalIdX;
        var ty = b.LocalIdY;
        var tid = ty * TileSide + tx;
        var col0 = b.GroupX * tile;
        // A slice's rows of the stage a tile and 4 values apart (an invocation's run of rows stays 16-byte aligned; the 16 k
        // of a stored row of A, or of Bᵀ, land on other banks).
        int pitch = tile + 4;
        var sa = b.Shared(k, TileK * pitch);                                         // sa[kk·pitch + i] = A(row0 + i, k0 + kk)
        var sb = b.Shared(k, TileK * pitch);                                         // sb[kk·pitch + j] = B(k0 + kk, col0 + j)
        var acc = new Var[micro, micro];
        for (int i = 0; i < micro; i++) for (int j = 0; j < micro; j++) acc[i, j] = b.Local(b.C(k, 0));
        var zero = b.C(k, 0);
        int threads = TileSide * TileSide;
        // The invocation's values of C: micro rows from ty·micro and micro columns from tx·micro (each slice's values a run
        // of the stage, read in one access where the device merges them).
        var rows = ty * micro;
        var cols = tx * micro;
        int loads = TileK * tile / threads;
        // This invocation's values of a slice: (row or column, k in the slice) and their places in the stage, the same
        // every slice (consecutive invocations along the operand's stored rows: k for A, i for Aᵀ).
        var aRow = new V[loads];
        var aK = new V[loads];
        var bK = new V[loads];
        var bCol = new V[loads];
        for (int l = 0; l < loads; l++)
        {
            var e = tid + l * threads;
            aRow[l] = transA ? e & (ulong)(tile - 1) : e >> 4;
            aK[l] = transA ? e >> shift : e & (TileK - 1);
            bK[l] = transB ? e & (TileK - 1) : e >> shift;
            bCol[l] = transB ? e >> 4 : e & (ulong)(tile - 1);
        }

        // A run's values (along the stored rows): the invocation's first and the step to the next, of rows and of k.
        if (runA is not null)
        {
            int per = transA ? tile / loads : TileK / loads;
            var lead = tid / (uint)per;
            var first = (tid - lead * (uint)per) * (uint)loads;
            for (int l = 0; l < loads; l++)
            {
                aRow[l] = transA ? first + (uint)l : lead;
                aK[l] = transA ? lead : first + (uint)l;
            }
        }

        if (runB is not null)
        {
            int per = transB ? TileK / loads : tile / loads;
            var lead = tid / (uint)per;
            var first = (tid - lead * (uint)per) * (uint)loads;
            for (int l = 0; l < loads; l++)
            {
                bK[l] = transB ? first + (uint)l : lead;
                bCol[l] = transB ? lead : first + (uint)l;
            }
        }

        V[] Fetch(V k0)
        {
            var got = new V[2 * loads];
            if (runA is not null)
            {
                var gr = row0 + aRow[0];
                var gk = k0 + aK[0];
                var inside = (gr < M) & (gk < kEnd);
                var values = runA(gr, gk, inside);
                for (int l = 0; l < loads; l++) got[l] = b.Select(inside, values[l], zero);
            }
            else
            {
                for (int l = 0; l < loads; l++)
                {
                    var gr = row0 + aRow[l];
                    var gk = k0 + aK[l];
                    var inside = (gr < M) & (gk < kEnd);
                    got[l] = b.Select(inside, loadA(gr, gk, inside), zero);
                }
            }

            if (runB is not null)
            {
                var gk = k0 + bK[0];
                var gc = col0 + bCol[0];
                var inside = (gk < kEnd) & (gc < N);
                var values = runB(gk, gc, inside);
                for (int l = 0; l < loads; l++) got[loads + l] = b.Select(inside, values[l], zero);
            }
            else
            {
                for (int l = 0; l < loads; l++)
                {
                    var gk = k0 + bK[l];
                    var gc = col0 + bCol[l];
                    var inside = (gk < kEnd) & (gc < N);
                    got[loads + l] = b.Select(inside, loadB(gk, gc, inside), zero);
                }
            }

            return got;
        }

        void Put(V[] got)
        {
            for (int l = 0; l < loads; l++) sa[aK[l] * pitch + aRow[l]] = got[l];
            for (int l = 0; l < loads; l++) sb[bK[l] * pitch + bCol[l]] = got[loads + l];
        }

        Put(Fetch(kBegin));
        b.Barrier();
        b.For(kBegin, kEnd, b.U(TileK), k0 =>
        {
            var next = Fetch(k0 + TileK);                                            // past kEnd: zeros, never read
            for (int q = 0; q < TileK; q++)
            {
                var av = new V[micro];
                var bv = new V[micro];
                for (int i = 0; i < micro; i++) av[i] = sa[rows + (q * pitch + i)];
                for (int j = 0; j < micro; j++) bv[j] = sb[cols + (q * pitch + j)];
                for (int i = 0; i < micro; i++) for (int j = 0; j < micro; j++) acc[i, j].V = b.Fma(av[i], bv[j], acc[i, j].V);
            }

            b.Barrier();
            Put(next);
            b.Barrier();
        });

        for (int i = 0; i < micro; i++)
        {
            for (int j = 0; j < micro; j++)
            {
                var r = row0 + rows + i;
                var c = col0 + cols + j;
                var value = acc[i, j].V;
                b.If((r < M) & (c < N), () => b.Store(b.At(pc, baseC + r * N + c, lane.Size), value));
            }
        }
    }

    /// <summary>A lane array's n values into a stored array a thread per word, the values at positions p with p mod period
    /// ≥ span (between a batch's region and the next) kept as they are stored. Arguments: lanes, stored, n, period, span.</summary>
    public static SpvKernel FromLanesKeep(Vulkan vk, Lane lane, Fmt f)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"back_keep_{lane.Name}_{f.Name}", Threads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var n = b.Arg(2, K.U32);
        var period = b.Arg(3, K.U32);
        var span = b.Arg(4, K.U32);
        var gid = b.GlobalX;
        var wk = Codec.WordKind(f);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var address = b.At(dst, gid, f.WordBits / 8);
            var old = b.Load(wk, address);
            var word = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                var idx = gid * (double)f.Per + i;
                var fresh = idx < n & idx % period < span;
                var v = b.Local(b.C(k, 0));
                b.If(fresh, () => v.V = b.Load(k, b.At(src, idx, lane.Size)));
                var packed = Codec.Pack(b, f, v.V, lane, i, b.CBits(wk, 0));
                ulong mask = f.WordBits == 64 ? ulong.MaxValue : ((1UL << f.Bits) - 1) << (i * f.Bits);
                word = word | b.Select(fresh, packed, old & mask);
            }

            b.Store(address, word);
        });
        return b.Build();
    }
}
