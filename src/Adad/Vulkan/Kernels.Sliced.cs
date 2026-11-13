// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// float64 products on the 8-bit matrix units, exactly: CUDA's sliced float64 (Cuda/Ptx.cs) step for step, so the same
// bits on both devices. Each value is scaled by the power of two of its row's (A) or column's (B) largest magnitude and
// rounded to a 62-bit integer, written as eight signed 8-bit digits (balanced base 256). A product of two such values is a
// sum of digit products weighted by powers of 2^-8; all pairs of one weight (a diagonal d = t + u) form one integer product
// over a k (d + 1) times the original (A's slices against B's stacked last first), on the s8 units into s32 sums, which
// cannot overflow (|digit| ≤ 128: a product ≤ 2^14, and k' ≤ 2^17 terms). Twelve diagonals are kept (the fifteen less the
// three below 2^-96 of the scale), combined in 128-bit integers and rounded once, to nearest even, into the float64 result.
// The operands are read where they are stored, by a row and a column stride each (a view's transposes).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    /// <summary>Digits a value is sliced into (8 bits each: 64 bits, 62 of them significant).</summary>
    public const int Slices = 8;

    /// <summary>Diagonals (digit-product weights) kept: d = t + u from 0 to Diagonals − 1.</summary>
    public const int Diagonals = 12;

    /// <summary>An exponent past every scale: a row or column holding a value that is not finite (the f64 lane takes it).</summary>
    public const int SliceRefused = 4000;

    // floor(log2 x) + 3 of a finite magnitude (a subnormal or zero as the smallest normal), held to at least −959; a value
    // that is not finite gives SliceRefused.
    private static V SliceExponent(Spv b, V absMax, V finite)
    {
        var exp = b.Bitcast(b.Convert((b.Bits(absMax) >> 52) & 0x7FFUL, K.U32), K.S32);
        exp = b.Select(exp == 0, b.S(1), exp);
        return b.Select(finite, b.Max(exp - 1020, b.S(-959)), b.S(SliceRefused));
    }

    // Value (i, p) of an operand read where it is stored: first + i·rowStride + p·colStride values from x.
    private static V Strided(Spv b, V x, V first, V i, V p, V rowStride, V colStride) =>
        b.Load(K.F64, b.At(x, b.Convert(first, K.U64) + b.Convert(i, K.U64) * b.Convert(rowStride, K.U64) + b.Convert(p, K.U64) * b.Convert(colStride, K.U64), 8));

    /// <summary>e[i] for each row of A (m × k float64): the slicing exponent of the row's largest magnitude; a workgroup a
    /// row. Arguments: a, e, k, rowStride, colStride, first.</summary>
    public static SpvKernel RowExponents(Vulkan vk)
    {
        var b = new Spv("slice_rowexp_f64", ReduceThreads, vk.Controls);
        var pa = b.Arg(0);
        var pe = b.Arg(1);
        var Kd = b.Arg(2, K.U32);
        var rs = b.Arg(3, K.U32);
        var cs = b.Arg(4, K.U32);
        var first = b.Arg(5, K.U32);
        var tid = b.LocalIdX;
        var row = b.GroupX;
        var max = b.Local(b.D(0));
        var finite = b.Local(b.U(1));
        b.For(tid, Kd, b.U(ReduceThreads), p =>
        {
            var v = b.Abs(Strided(b, pa, first, row, p, rs, cs));
            finite.V = finite.V & b.Select(b.IsInf(v) | b.IsNan(v), b.U(0), b.U(1));
            max.V = b.Max(max.V, v);
        });
        var all = GroupReduce(b, b.Shared(K.F64, ReduceThreads), tid, max.V, b.Max, ReduceThreads);
        var allFinite = GroupReduce(b, b.Shared(K.U32, ReduceThreads), tid, finite.V, (x, y) => x & y, ReduceThreads);
        b.If(tid == 0, () => b.Store(b.At(pe, row, 4), SliceExponent(b, all, allFinite != 0)));
        return b.Build();
    }

    /// <summary>f[j] for each column of B (k × n float64): a thread per column. Arguments: b, f, k, n, rowStride, colStride, first.</summary>
    public static SpvKernel ColumnExponents(Vulkan vk)
    {
        var b = new Spv("slice_colexp_f64", Threads, vk.Controls);
        var pb = b.Arg(0);
        var pf = b.Arg(1);
        var Kd = b.Arg(2, K.U32);
        var N = b.Arg(3, K.U32);
        var rs = b.Arg(4, K.U32);
        var cs = b.Arg(5, K.U32);
        var first = b.Arg(6, K.U32);
        var j = b.GlobalX;
        b.If(j < N, () =>
        {
            var max = b.Local(b.D(0));
            var finite = b.Local(b.Bool(true));
            b.For(b.U(0), Kd, p =>
            {
                var v = b.Abs(Strided(b, pb, first, p, j, rs, cs));
                finite.V = finite.V & !(b.IsInf(v) | b.IsNan(v));
                max.V = b.Max(max.V, v);
            });
            b.Store(b.At(pf, j, 4), SliceExponent(b, max.V, finite.V));
        });
        return b.Build();
    }

    // The value's 62-bit integer Y = round(v · 2^(64 − e)) and its eight balanced digits, the most significant first.
    private static V[] Digits(Spv b, V v, V e)
    {
        var scale = b.Bitcast(b.Convert(b.S(1087) - e, K.U64) << 52, K.F64);
        var y = b.Convert(b.RoundEven(v * scale), K.S64);
        var digits = new V[Slices];
        for (int t = Slices - 1; t >= 0; t--)
        {
            var d = ((y + 128) & 255UL) - 128;
            y = (y - d) >> 8;
            digits[t] = d;
        }

        return digits;
    }

    /// <summary>A's values into packed rows of digits: S[i][t·kpad + p] = digit t of A[i][p] (t = 0 the most significant),
    /// rows past m and values past k 0; a thread per four values of a row (a word of each slice). Arguments: a, e, s, m, k,
    /// kpad, rowStride, colStride, rows (whole tiles), first.</summary>
    public static SpvKernel SliceRows(Vulkan vk)
    {
        var b = new Spv("slice_rows_f64", Threads, vk.Controls);
        var pa = b.Arg(0);
        var pe = b.Arg(1);
        var ps = b.Arg(2);
        var M = b.Arg(3, K.U32);
        var Kd = b.Arg(4, K.U32);
        var kpad = b.Arg(5, K.U32);
        var rs = b.Arg(6, K.U32);
        var cs = b.Arg(7, K.U32);
        var rows = b.Arg(8, K.U32);
        var first = b.Arg(9, K.U32);
        var gid = b.GlobalX;
        var quads = kpad >> 2;
        b.If(gid < rows * quads, () =>
        {
            var i = gid / quads;
            var p0 = (gid - i * quads) * 4;
            var inRow = i < M;
            var e = b.Select(inRow, b.Load(K.S32, b.At(pe, b.Select(inRow, i, b.U(0)), 4)), b.S(0));
            var words = new V[Slices];
            for (int t = 0; t < Slices; t++) words[t] = b.U(0);
            for (int q = 0; q < 4; q++)
            {
                var p = p0 + (uint)q;
                var inside = inRow & (p < Kd);
                var v = b.Select(inside, Strided(b, pa, first, b.Select(inside, i, b.U(0)), b.Select(inside, p, b.U(0)), rs, cs), b.D(0));
                var digits = Digits(b, v, e);
                for (int t = 0; t < Slices; t++) words[t] = words[t] | (b.Convert(digits[t], K.U32) & 0xFFUL) << (8 * q);
            }

            var rowWords = b.Convert(i, K.U64) * b.Convert(kpad * (Slices / 4.0), K.U64);
            for (int t = 0; t < Slices; t++)
                b.Store(b.At(ps, rowWords + b.Convert((kpad >> 2) * t + (p0 >> 2), K.U64), 4), words[t]);
        });
        return b.Build();
    }

    /// <summary>B's values into packed rows of its columns' digits, stacked most significant last: Bt[j][(Slices − 1 − t)·kpad
    /// + p] = digit t of B[p][j], columns past n and values past k 0; a thread per four values of a column. Arguments: b, f,
    /// bt, n, k, kpad, rowStride, colStride, columns (whole tiles), first.</summary>
    public static SpvKernel SliceColumns(Vulkan vk)
    {
        var b = new Spv("slice_cols_f64", Threads, vk.Controls);
        var pb = b.Arg(0);
        var pf = b.Arg(1);
        var pt = b.Arg(2);
        var N = b.Arg(3, K.U32);
        var Kd = b.Arg(4, K.U32);
        var kpad = b.Arg(5, K.U32);
        var rs = b.Arg(6, K.U32);
        var cs = b.Arg(7, K.U32);
        var cols = b.Arg(8, K.U32);
        var first = b.Arg(9, K.U32);
        var gid = b.GlobalX;
        var quads = kpad >> 2;
        b.If(gid < cols * quads, () =>
        {
            // The column fastest (consecutive invocations read consecutive values of a row of B).
            var w = gid / cols;
            var j = gid - w * cols;
            var p0 = w * 4;
            var inCol = j < N;
            var f = b.Select(inCol, b.Load(K.S32, b.At(pf, b.Select(inCol, j, b.U(0)), 4)), b.S(0));
            var words = new V[Slices];
            for (int t = 0; t < Slices; t++) words[t] = b.U(0);
            for (int q = 0; q < 4; q++)
            {
                var p = p0 + (uint)q;
                var inside = inCol & (p < Kd);
                var v = b.Select(inside, Strided(b, pb, first, b.Select(inside, p, b.U(0)), b.Select(inside, j, b.U(0)), rs, cs), b.D(0));
                var digits = Digits(b, v, f);
                for (int t = 0; t < Slices; t++) words[t] = words[t] | (b.Convert(digits[t], K.U32) & 0xFFUL) << (8 * q);
            }

            var rowWords = b.Convert(j, K.U64) * b.Convert(kpad * (Slices / 4.0), K.U64);
            for (int t = 0; t < Slices; t++)
                b.Store(b.At(pt, rowWords + b.Convert((kpad >> 2) * (Slices - 1 - t) + w, K.U64), 4), words[t]);
        });
        return b.Build();
    }

    // The 64-bit value's highest set bit (−1 for 0), as an s32.
    private static V Msb64(Spv b, V x)
    {
        var hi = b.HighBit(b.Convert(x >> 32, K.U32));
        var lo = b.HighBit(b.Convert(x, K.U32));
        return b.Select(hi >= 0, hi + 32, lo);
    }

    /// <summary>C[i][j] = round(Σ_d D_d[i][j] · 2^(8(Diagonals−1−d))) · 2^(e_i + f_j − 16 − 8(Diagonals−1)): the diagonals'
    /// integers summed in 128 bits, rounded once to nearest even into float64, scaled by the exact power of two; a thread per
    /// value. Arguments: d (the diagonals' s32 sums, m·n apart), e, f, c, m, n.</summary>
    public static SpvKernel CombineSlices(Vulkan vk)
    {
        var b = new Spv("slice_combine_f64", Threads, vk.Controls);
        var pd = b.Arg(0);
        var pe = b.Arg(1);
        var pf = b.Arg(2);
        var pc = b.Arg(3);
        var M = b.Arg(4, K.U32);
        var N = b.Arg(5, K.U32);
        var gid = b.GlobalX;
        var total = M * N;
        b.If(gid < total, () =>
        {
            var i = gid / N;
            var j = gid - i * N;
            var e = b.Load(K.S32, b.At(pe, i, 4));
            var f = b.Load(K.S32, b.At(pf, j, 4));

            // S = Σ_d D_d << 8(Diagonals − 1 − d), in 128 bits (lo unsigned, hi two's complement), carries by comparison.
            V lo = b.UL(0), hi = b.UL(0);
            for (int d = 0; d < Diagonals; d++)
            {
                var v = b.Convert(b.Load(K.S32, b.At(pd, b.Convert(gid, K.U64) + b.Convert(total, K.U64) * (ulong)d, 4)), K.S64);
                int sh = 8 * (Diagonals - 1 - d);
                V tlo, thi;
                if (sh == 0) { tlo = b.Bits(v); thi = b.Bits(v >> 63); }
                else if (sh < 64) { tlo = b.Bits(v << sh); thi = b.Bits(v >> (64 - sh)); }
                else { tlo = b.UL(0); thi = b.Bits(v << (sh - 64)); }
                var sum = lo + tlo;
                hi = hi + thi + b.Select(sum < lo, b.UL(1), b.UL(0));
                lo = sum;
            }

            // Sign and magnitude.
            var neg = b.Bitcast(hi, K.S64) < 0;
            var nlo = ~lo + 1;
            var nhi = ~hi + b.Select(nlo == 0, b.UL(1), b.UL(0));
            lo = b.Select(neg, nlo, lo);
            hi = b.Select(neg, nhi, hi);

            // The magnitude's bits; its top 53 with the rest rounded to nearest even (every shift held below 64).
            var nbits = b.Select(hi != 0, Msb64(b, hi) + 65, Msb64(b, lo) + 1);
            var shift = b.Bitcast(b.Max(nbits - 53, b.S(0)), K.U32);
            var low = shift < 64;
            var s = b.Min(shift, b.U(63));                                           // in 0..63
            var s1 = b.Max(s, b.U(1)) - 1;                                           // shift − 1 (low, shift ≥ 1)
            var inv = b.U(64) - b.Max(s, b.U(1));                                    // 64 − shift (low, shift ≥ 1)
            V S(V x) => b.Convert(x, K.U64);
            var mantLow = b.Select(s == 0, lo, hi.Shl(S(inv)) | lo.Shr(S(s)));
            var roundLow = lo.Shr(S(s1)) & 1UL;
            var stickyLow = lo & (b.UL(1).Shl(S(s1)) - 1);
            var s2 = b.Min(shift - b.Min(shift, b.U(64)), b.U(63));                  // shift − 64 (wide)
            var s3 = b.Max(s2, b.U(1)) - 1;
            var mantWide = hi.Shr(S(s2));
            var roundWide = b.Select(s2 == 0, lo >> 63, hi.Shr(S(s3)) & 1UL);
            var stickyWide = b.Select(s2 == 0, lo & 0x7FFFFFFFFFFFFFFFUL, (hi & (b.UL(1).Shl(S(s3)) - 1)) | lo);
            var mant = b.Select(low, mantLow, mantWide);
            var round = b.Select(low, roundLow, roundWide);
            var sticky = b.Select(low, stickyLow, stickyWide);
            var up = round & b.Select(sticky != 0, b.UL(1), mant & 1UL);
            mant = b.Select(shift == 0, lo, mant + up);

            var expo = e + f + b.Bitcast(shift, K.S32) + (1023 - 16 - 8 * (Diagonals - 1));
            var scale = b.Bitcast(b.Convert(expo, K.U64) << 52, K.F64);
            var value = b.Convert(mant, K.F64) * scale;
            b.Store(b.At(pc, gid, 8), b.Select(neg, -value, value));
        });
        return b.Build();
    }
}
