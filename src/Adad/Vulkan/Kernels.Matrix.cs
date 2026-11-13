// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The products on the device's matrix units (Spv.Units.cs: cooperative matrices, a subgroup's), for the operands whose
// values the units hold exactly: 16-bit floats (float16 and the 8-bit floats as f16, bfloat16 as bf16) into f32 sums, and
// integers of 8 bits or 8-bit range into s32 or u32 sums, the kinds and shapes as the device reports them. A workgroup
// holds a BM × BN tile of C (UnitsConfig: its invocations, the tile, its subgroups' grid; which configuration a kind takes
// is measured on the device, VulkanOps.Matrix.cs), each subgroup a block of it, every accumulator one of the units'
// M × N; A's and B's slices of KBytes of k a row are staged in workgroup memory, 16 bytes a load, in two stages (the
// next slice's loads in flight while the units read the current one). The operands are read in either orientation as
// their rows lie in memory (A by rows of k or by columns of k, B by rows of n or rows of k), the type's own bytes where it
// is the kind (in place) or rows packed into the kind first (PackUnits: any type, any view, zeros past m, n and k). C is
// the sums in their lane at C's positions: a whole M × N block stored by the units, a block past m or n through the
// subgroup's workgroup memory, each value inside written alone.
using Adad.Abstraction;

namespace Adad.Vulkan;

/// <summary>A units product's shape: invocations a workgroup, C's tile (BM × BN), the subgroups' rows (WarpsM: the tile's
/// rows split among them, the rest of the subgroups along its columns), and the bytes of k a stage holds.</summary>
internal readonly record struct UnitsConfig(int Threads, int BM, int BN, int WarpsM, int KBytes)
{
    public override string ToString() => $"{Threads}t {BM}x{BN} {WarpsM}r k{KBytes}";
}

internal static partial class VkKernels
{
    /// <summary>The shapes a kind's units product is measured over (VulkanOps.Matrix.cs): two stages each, within the
    /// workgroup memory the device has (UnitsShared).</summary>
    public static readonly UnitsConfig[] UnitsConfigs =
    [
        new(256, 128, 128, 2, 64), new(128, 128, 128, 2, 64), new(256, 128, 256, 2, 32), new(256, 256, 128, 4, 32), new(128, 128, 128, 2, 32),
    ];

    /// <summary>The workgroup memory of the units' product: two stages of A's and B's slices.</summary>
    public static int UnitsShared(UnitsConfig c, int eb, bool aRows, bool bRows) => 2 * 16 * (UnitsVectors(c, eb, aRows, c.BM) + UnitsVectors(c, eb, !bRows, c.BN));

    // The 16-byte vectors of one stage of an operand's slice: rows of k (KBytes and a vector of padding) or rows of the
    // tile's side (side·eb bytes and a vector), kRows says which.
    private static int UnitsVectors(UnitsConfig c, int eb, bool kRows, int side) =>
        kRows ? side * (c.KBytes / 16 + 1) : c.KBytes / eb * (side * eb / 16 + 1);

    /// <summary>Whether the configuration tiles with the unit and the device's subgroups (whole fragments, whole loads).</summary>
    public static bool UnitsTiles(UnitsConfig c, CooperativeMatrix unit, int subgroup, int eb)
    {
        int warps = c.Threads / subgroup;
        if (warps < c.WarpsM || warps % c.WarpsM != 0 || c.Threads % subgroup != 0) return false;
        int wn = c.BN / (warps / c.WarpsM), wm = c.BM / c.WarpsM, bk = c.KBytes / eb;
        int vecA = c.BM * c.KBytes / 16, vecB = c.BN * c.KBytes / 16;
        return wm % unit.M == 0 && wn % unit.N == 0 && bk % unit.K == 0 && vecA % c.Threads == 0 && vecB % c.Threads == 0 && wm > 0 && wn > 0;
    }

    /// <summary>The element kind of a unit's operand or sum by its name.</summary>
    public static K UnitKind(string name) => name switch
    {
        "f16" => K.F16,
        "bf16" => K.BF16,
        "s8" => K.S8,
        "u8" => K.U8,
        "f32" => K.F32,
        "s32" => K.S32,
        _ => K.U32,
    };

    /// <summary>C (the unit's sums, its lane, m × n a batch) = A·B on the matrix units. Arguments: a, b, c (addresses), lda, ldb
    /// (elements between A's and B's stored rows), m, n, k (k whole slices: a multiple of KBytes of the kind), strideA,
    /// strideB, strideC (elements between batches), tilesM; workgroups (⌈n/BN⌉, ⌈m/BM⌉ · batches). A's rows past m and
    /// B's past n are read as stored (whole tiles, or packed rows padded with zeros), C's written only inside.
    /// aRows: A stored by rows of k (else by columns: Aᵀ's rows); bRows: B stored by rows of n (else by rows of k: Bᵀ's).</summary>
    public static SpvKernel UnitsMatMul(Vulkan vk, CooperativeMatrix unit, UnitsConfig cfg, bool aRows, bool bRows)
    {
        K ek = UnitKind(unit.A), ck = UnitKind(unit.C);
        int eb = ek.Bytes(), bk = cfg.KBytes / eb, sg = vk.SubgroupSize, warps = cfg.Threads / sg, rowVec = cfg.KBytes / 16;
        if (!UnitsTiles(cfg, unit, sg, eb)) throw new NotSupportedException($"vulkan: the matrix units' {unit} do not tile {cfg} with subgroups of {sg}");
        int warpsM = cfg.WarpsM, warpsN = warps / warpsM, wm = cfg.BM / warpsM, wn = cfg.BN / warpsN, fm = wm / unit.M, fn = wn / unit.N;
        var b = new Spv($"units_matmul_{unit.A}_{unit.C}_{unit.M}x{unit.N}x{unit.K}_{cfg.Threads}_{cfg.BM}x{cfg.BN}_{warpsM}_{cfg.KBytes}{(aRows ? "" : "_ta")}{(bRows ? "" : "_tb")}",
            cfg.Threads, vk.Controls);
        var pa = b.Arg(0);
        var pb = b.Arg(1);
        var pc = b.Arg(2);
        var lda = b.Arg(3, K.U32);
        var ldb = b.Arg(4, K.U32);
        var M = b.Arg(5, K.U32);
        var N = b.Arg(6, K.U32);
        var Kd = b.Arg(7, K.U32);
        var strideA = b.Arg(8);
        var strideB = b.Arg(9);
        var strideC = b.Arg(10);
        var tilesM = b.Arg(11, K.U32);

        // The staged slices, 16-byte vectors: rows of k (KBytes and a vector of padding) or rows of the tile's side (and a
        // vector of padding), two stages of A and of B.
        int aVec = UnitsVectors(cfg, eb, aRows, cfg.BM), bVec = UnitsVectors(cfg, eb, !bRows, cfg.BN);
        int aStride = aRows ? rowVec + 1 : cfg.BM * eb / 16 + 1, bStride = bRows ? cfg.BN * eb / 16 + 1 : rowVec + 1;
        var stage = b.SharedVectors(K.U32, 4, 2 * (aVec + bVec));
        var tid = b.LocalIdX;
        var gy = b.GroupY;
        var batch = gy / tilesM;
        var row0 = (gy - batch * tilesM) * cfg.BM;
        var col0 = b.GroupX * cfg.BN;
        var b64 = b.Convert(batch, K.U64);
        var baseA = pa + b64 * strideA * (ulong)eb;
        var baseB = pb + b64 * strideB * (ulong)eb;

        // Thread t's vectors of an operand's slice: vector v = t + threads·l, as (row, vector in the row) of the stored
        // slice, its address past the slice's start and its place in the stage.
        (V Global, V Local)[] Plan(bool kRows, V ld, V first, int side, int stride)
        {
            int count = side * cfg.KBytes / 16 / cfg.Threads, rowVectors = kRows ? rowVec : side * eb / 16;
            var plan = new (V, V)[count];
            for (int l = 0; l < count; l++)
            {
                var v = tid + l * cfg.Threads;
                var row = v / (uint)rowVectors;
                var part = v - row * (uint)rowVectors;
                // A row of k: the row is the m (n) index, its bytes the slice's k; a row of m (n): the row is k.
                var global = kRows ? b.Convert(first + row, K.U64) * b.Convert(ld, K.U64) * (ulong)eb + b.Convert(part, K.U64) * 16
                                   : b.Convert(row, K.U64) * b.Convert(ld, K.U64) * (ulong)eb + b.Convert(first, K.U64) * (ulong)eb + b.Convert(part, K.U64) * 16;
                plan[l] = (global, row * (uint)stride + part);
            }

            return plan;
        }

        var planA = Plan(aRows, lda, row0, cfg.BM, aStride);
        var planB = Plan(!bRows, ldb, col0, cfg.BN, bStride);
        // Slice k0's bytes: along a row of k, k0·eb; across rows of k, k0 rows.
        V SliceA(V k0) => aRows ? b.Convert(k0, K.U64) * (ulong)eb : b.Convert(k0, K.U64) * b.Convert(lda, K.U64) * (ulong)eb;
        V SliceB(V k0) => bRows ? b.Convert(k0, K.U64) * b.Convert(ldb, K.U64) * (ulong)eb : b.Convert(k0, K.U64) * (ulong)eb;
        V[][] Fetch(V k0)
        {
            var got = new V[planA.Length + planB.Length][];
            var sa = baseA + SliceA(k0);
            var sb = baseB + SliceB(k0);
            for (int l = 0; l < planA.Length; l++) got[l] = b.LoadN(K.U32, 4, sa + planA[l].Global);
            for (int l = 0; l < planB.Length; l++) got[planA.Length + l] = b.LoadN(K.U32, 4, sb + planB[l].Global);
            return got;
        }

        void Put(V[][] got, V at)
        {
            for (int l = 0; l < planA.Length; l++) stage.Store(at + planA[l].Local, got[l]);
            for (int l = 0; l < planB.Length; l++) stage.Store(at + (uint)aVec + planB[l].Local, got[planA.Length + l]);
        }

        var warp = b.SubgroupId;
        var wr = warp / (uint)warpsN;
        var wc = warp - wr * (uint)warpsN;
        var ta = b.Coop(ek, unit.M, unit.K, Use.A);
        var tb = b.Coop(ek, unit.K, unit.N, Use.B);
        var tc = b.Coop(ck, unit.M, unit.N, Use.Accumulator);
        var acc = new MVar[fm, fn];
        var zero = b.Fill(tc, 0);
        for (int i = 0; i < fm; i++) for (int j = 0; j < fn; j++) acc[i, j] = b.Local(zero);

        var slices = Kd / (uint)bk;
        Put(Fetch(b.U(0)), b.U(0));
        b.Barrier();
        b.For(b.U(0), slices, s =>
        {
            var cur = (s & 1UL) * (uint)(aVec + bVec);
            var next = b.Min(s + 1, slices - 1);
            var got = Fetch(next * (uint)bk);
            var aAt = cur;
            var bAt = cur + (uint)aVec;
            for (int kk = 0; kk < bk; kk += unit.K)
            {
                var af = new M[fm];
                var bf = new M[fn];
                for (int i = 0; i < fm; i++)
                {
                    var r = wr * (uint)wm + (uint)(i * unit.M);
                    af[i] = aRows ? b.Load(ta, stage, aAt + r * (uint)aStride + (uint)(kk * eb / 16), b.U(aStride))
                                  : b.Load(ta, stage, aAt + (uint)(kk * aStride) + r * (uint)eb / 16, b.U(aStride), columnMajor: true);
                }

                for (int j = 0; j < fn; j++)
                {
                    var c = wc * (uint)wn + (uint)(j * unit.N);
                    bf[j] = bRows ? b.Load(tb, stage, bAt + (uint)(kk * bStride) + c * (uint)eb / 16, b.U(bStride))
                                  : b.Load(tb, stage, bAt + c * (uint)bStride + (uint)(kk * eb / 16), b.U(bStride), columnMajor: true);
                }

                for (int i = 0; i < fm; i++) for (int j = 0; j < fn; j++) acc[i, j].M = b.MulAdd(af[i], bf[j], acc[i, j].M);
            }

            Put(got, b.U(aVec + bVec) - cur);
            b.Barrier();
        });

        // C: a whole block by the units; a block past m or n through the subgroup's 16-byte vectors of workgroup memory
        // (the stage's first vectors, free now), each value inside written by one invocation.
        var lane = tid - warp * (uint)sg;
        var baseC = pc + b64 * strideC * 4;
        int blockVec = unit.M * unit.N / 4;
        var scratch = warp * (uint)blockVec;
        for (int i = 0; i < fm; i++)
        {
            for (int j = 0; j < fn; j++)
            {
                var r = row0 + wr * (uint)wm + (uint)(i * unit.M);
                var c = col0 + wc * (uint)wn + (uint)(j * unit.N);
                var m = acc[i, j].M;
                var at = baseC + (b.Convert(r, K.U64) * b.Convert(N, K.U64) + b.Convert(c, K.U64)) * 4;
                b.If((r + (uint)unit.M <= M) & (c + (uint)unit.N <= N), () => b.Store(m, ck, at, N, false, 4), () =>
                {
                    b.Store(m, stage, scratch, b.U(unit.N / 4));
                    b.SubgroupBarrier();
                    for (int e = 0; e * sg < unit.M * unit.N; e++)
                    {
                        var idx = lane + (uint)(e * sg);
                        var rr = idx / (uint)unit.N;
                        var cc = idx - rr * (uint)unit.N;
                        b.If((r + rr < M) & (c + cc < N), () =>
                        {
                            var vec = stage.Load(scratch + (idx >> 2));
                            var w = vec[0];
                            for (int q = 1; q < 4; q++) w = b.Select((idx & 3UL) == (ulong)q, vec[q], w);
                            b.Store(at + (b.Convert(rr, K.U64) * b.Convert(N, K.U64) + b.Convert(cc, K.U64)) * 4, b.Bitcast(w, ck));
                        });
                    }

                    b.SubgroupBarrier();
                });
            }
        }

        return b.Build();
    }

    /// <summary>Rows of a product's operand packed into the units' kind: rows × kp values (kp whole slices), row r's value p
    /// the stored value at first + r·rowStride + p·colStride for r below rows and p below k, else 0; each value made in the
    /// kind exactly (a 16-bit float's bits, an 8-bit float as f16, an integer's low byte: the route takes only the values the
    /// kind holds). A thread per 32-bit word. Arguments: src, dst, rows, k, rowStride, colStride, wordsPerRow, words, first.</summary>
    public static SpvKernel PackUnits(Vulkan vk, Fmt f, string kind)
    {
        K ek = UnitKind(kind);
        int per = 4 / ek.Bytes(), bits = 32 / per;
        var lane = ek.IsFloat() ? Lane.F32 : f.Signed ? Lane.S32 : Lane.U32;
        var b = new Spv($"pack_units_{f.Name}_{kind}", Threads, vk.Controls);
        var src = b.Arg(0);
        var dst = b.Arg(1);
        var rows = b.Arg(2, K.U32);
        var Kd = b.Arg(3, K.U32);
        var rowStride = b.Arg(4, K.U32);
        var colStride = b.Arg(5, K.U32);
        var wordsPerRow = b.Arg(6, K.U32);
        var words = b.Arg(7, K.U32);
        var first = b.Arg(8, K.U32);
        var gid = b.GlobalX;
        b.If(gid < words, () =>
        {
            var row = gid / wordsPerRow;
            var p0 = (gid - row * wordsPerRow) * (uint)per;
            var word = b.U(0);
            for (int e = 0; e < per; e++)
            {
                var p = p0 + (uint)e;
                var inside = (row < rows) & (p < Kd);
                var v = Codec.UnpackAt(b, f, src, first + b.Select(inside, row * rowStride + p * colStride, b.U(0)), lane);
                V code = ek switch
                {
                    K.F16 => Codec.FloatToHalf(b, v),
                    K.BF16 => b.Bits(v) >> 16,
                    _ => b.Bitcast(v, K.U32) & 0xFFUL,
                };
                word = word | b.Select(inside, code, b.U(0)) << (e * bits);
            }

            b.PutWord(dst, gid, word);
        });
        return b.Build();
    }
}
