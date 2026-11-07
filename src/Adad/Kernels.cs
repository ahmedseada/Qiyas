// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The CPU kernels over Adad numbers: each written once over <TN, TLane, TL>, run on the Team (struct bodies: nothing
// allocated per call), in place where the stored bytes are the lanes, through tiles on the stack otherwise.
//
// The product is GotoBLAS-shaped: B packed once into strips of two vectors' width (contiguous over k), the output cut
// into tiles of rows × columns spread across the team, each tile walking k in panels: A's 6-row blocks packed per
// panel, a 6 × (2 vectors) block of C held in registers through the panel. A dot lane runs the same kernel with k
// taken in groups: each 32-bit accumulator lane holds four 8-bit (two 16-bit) inputs, packed that way for B once and
// for A per panel, and Fma is the processor's widening multiply-add. Loading, packing and storing are parts of the
// same passes: B's rows load as they pack, A's as they block, C's tile stores the moment its last panel ends.
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Adad;

internal sealed partial class Kernels
{
    private const int Tile = 2048;                                           // lanes an element-wise tile holds (L1)

    private static int Threads => Team.Size;

    /// <summary>Rows a task takes when tasks write whole rows of `cols` values of TN: about `wanted`, rounded up to a count
    /// whose row span is whole storage units (a packed byte, a block), so two tasks never write one unit.</summary>
    internal static int RowChunk<TN>(int rows, int cols, int wanted) where TN : struct, IAdad<TN>
    {
        int g = TN.Granularity;
        if (g <= 1 || cols % g == 0) return Math.Max(1, wanted);
        int unit = g / Gcd(g, cols);                                       // rows per whole unit boundary
        return Math.Max(unit, (Math.Max(1, wanted) + unit - 1) / unit * unit);
    }

    private static int Gcd(int a, int b) { while (b != 0) (a, b) = (b, a % b); return a; }

    /// <summary>When set (profile runs only), MatMul adds each phase's milliseconds: load, pack, product, store.</summary>
    public static double[]? Phases;

    private static long Mark(long since, int phase)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (Phases is not null) Phases[phase] += System.Diagnostics.Stopwatch.GetElapsedTime(since, now).TotalMilliseconds;
        return now;
    }

    // ------------------------------------------------------------------ c = a + b

    public struct AddVisitor(byte[] a, byte[] b, byte[] c, int n) : ILaneVisitor
    {
        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> =>
            Add<TN, TLane, TL>(a, b, c, n);
    }

    private struct AddBody<TN, TLane, TL>(byte[] a, byte[] b, byte[] c) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            if (TN.Direct)
            {
                AddLanes<TLane, TL>(MemoryMarshal.Cast<byte, TLane>(a.AsSpan()).Slice(first, last - first),
                    MemoryMarshal.Cast<byte, TLane>(b.AsSpan()).Slice(first, last - first), MemoryMarshal.Cast<byte, TLane>(c.AsSpan()).Slice(first, last - first));
                return;
            }

            Span<TLane> x = stackalloc TLane[Tile];
            Span<TLane> y = stackalloc TLane[Tile];
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at);
                TN.Load(a, at, x[..len]);
                TN.Load(b, at, y[..len]);
                AddLanes<TLane, TL>(x[..len], y[..len], x[..len]);
                TN.Store(x[..len], c, at);
            }
        }
    }

    public static void Add<TN, TLane, TL>(byte[] a, byte[] b, byte[] c, int n)
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        // Chunks of whole tiles (whole bytes for packed values), about four per thread.
        int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
        Team.For(n, size, n, new AddBody<TN, TLane, TL>(a, b, c));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddLanes<TLane, TL>(ReadOnlySpan<TLane> x, ReadOnlySpan<TLane> y, Span<TLane> z) where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int i = 0, w = Vector<TLane>.Count;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        ref TLane rz = ref MemoryMarshal.GetReference(z);
        for (; i + 4 * w <= x.Length; i += 4 * w)
        {
            TL.Add(Vector.LoadUnsafe(ref rx, (nuint)i), Vector.LoadUnsafe(ref ry, (nuint)i)).StoreUnsafe(ref rz, (nuint)i);
            TL.Add(Vector.LoadUnsafe(ref rx, (nuint)(i + w)), Vector.LoadUnsafe(ref ry, (nuint)(i + w))).StoreUnsafe(ref rz, (nuint)(i + w));
            TL.Add(Vector.LoadUnsafe(ref rx, (nuint)(i + 2 * w)), Vector.LoadUnsafe(ref ry, (nuint)(i + 2 * w))).StoreUnsafe(ref rz, (nuint)(i + 2 * w));
            TL.Add(Vector.LoadUnsafe(ref rx, (nuint)(i + 3 * w)), Vector.LoadUnsafe(ref ry, (nuint)(i + 3 * w))).StoreUnsafe(ref rz, (nuint)(i + 3 * w));
        }

        for (; i + w <= x.Length; i += w) TL.Add(Vector.LoadUnsafe(ref rx, (nuint)i), Vector.LoadUnsafe(ref ry, (nuint)i)).StoreUnsafe(ref rz, (nuint)i);
        for (; i < x.Length; i++) z[i] = TL.Add(x[i], y[i]);
    }

    // ------------------------------------------------------------------ C = A·B

    private const int R = 6, KC = 256;                                       // KC in input values (a multiple of every group)

    /// <summary>Where a product's operands are stored (in values): A from AOff, stored k × m (Aᵀ) when TransA; B from BOff,
    /// stored n × k (Bᵀ) when TransB; C from COff. The packing reads them there: no copy of a transposed or batched
    /// operand. Each offset on whole storage units, and every row starting on a whole byte.</summary>
    public readonly record struct Operands(int AOff = 0, bool TransA = false, int BOff = 0, bool TransB = false, int COff = 0, bool Viewed = true, ConvArgs? ImageB = null)
    {
        // ImageB (item 4g): B is the convolution's columns (C·kh·kw × OH·OW) read from the image itself, the stored b
        // being the image (C × H × W): the packing builds each row of B from it (the implicit im2col); with TransB, Bᵀ
        // is those columns (the convolution's dw = dy · colsᵀ), the transposed packing taking each row's k-range from it.
        /// <summary>The plain product (no view: C exactly m × n); any constructed Operands is a view, even batch 0's.</summary>
        public bool Plain => !Viewed;
    }

    public struct MatMulVisitor(byte[] a, byte[] b, byte[] c, int m, int n, int k) : ISumVisitor
    {
        public string Lane = "";
        public Operands Where;

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            Lane = TL.Name;
            MatMul<TN, TLane, TLane, TL>(a, b, c, m, n, k, Where);
        }

        public void RunDot<TN, TIn, TAcc, TD>()
            where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            Lane = TD.Name;
            MatMul<TN, TIn, TAcc, TD>(a, b, c, m, n, k, Where);
        }
    }

    // A stored matrix into lanes, whole (only for rows that do not start on whole bytes).
    private struct LoadBody<TN, TLane>(byte[] stored, TLane[] lanes) : Team.IBody where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        public readonly void Run(int first, int last) => TN.Load(stored, first, lanes.AsSpan(first, last - first));
    }

    private struct StoreBody<TN, TLane>(TLane[] lanes, byte[] stored, int offset = 0) : Team.IBody where TN : struct, IAdad<TN, TLane> where TLane : unmanaged
    {
        public readonly void Run(int first, int last) => TN.Store(lanes.AsSpan(first, last - first), stored, offset + first);
    }

    // G rows of inputs (each n long, contiguous) into n accumulator-sized units: unit j holds the G values of column j,
    // the first in the low bits (the order the widening multiply-add reads).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Interleave<TIn, TAcc>(ReadOnlySpan<TIn> rows, Span<TAcc> units, int n) where TIn : unmanaged where TAcc : unmanaged
    {
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>();
        if (g == 1)
        {
            MemoryMarshal.Cast<TIn, TAcc>(rows)[..n].CopyTo(units);
            return;
        }

        var outBytes = MemoryMarshal.AsBytes(units);
        if (g == 4)
        {
            var bytes = MemoryMarshal.AsBytes(rows);
            var u = MemoryMarshal.Cast<byte, uint>(outBytes);
            int i = 0, w = Vector<byte>.Count, q = w / 4;
            for (; i + w <= n; i += w)
            {
                Four(bytes, i, n, out var a0, out var a1, out var a2, out var a3);
                Four(bytes[n..], i, n, out var b0, out var b1, out var b2, out var b3);
                Four(bytes[(2 * n)..], i, n, out var c0, out var c1, out var c2, out var c3);
                Four(bytes[(3 * n)..], i, n, out var d0, out var d1, out var d2, out var d3);
                (a0 | (b0 << 8) | (c0 << 16) | (d0 << 24)).CopyTo(u[i..]);
                (a1 | (b1 << 8) | (c1 << 16) | (d1 << 24)).CopyTo(u[(i + q)..]);
                (a2 | (b2 << 8) | (c2 << 16) | (d2 << 24)).CopyTo(u[(i + 2 * q)..]);
                (a3 | (b3 << 8) | (c3 << 16) | (d3 << 24)).CopyTo(u[(i + 3 * q)..]);
            }

            for (; i < n; i++) u[i] = bytes[i] | (uint)bytes[n + i] << 8 | (uint)bytes[2 * n + i] << 16 | (uint)bytes[3 * n + i] << 24;
            return;
        }

        if (g == 2)
        {
            var shorts = MemoryMarshal.Cast<TIn, ushort>(rows);
            var u = MemoryMarshal.Cast<byte, uint>(outBytes);
            int i = 0, w = Vector<ushort>.Count, h = w / 2;
            for (; i + w <= n; i += w)
            {
                Vector.Widen(new Vector<ushort>(shorts[i..]), out Vector<uint> a0, out Vector<uint> a1);
                Vector.Widen(new Vector<ushort>(shorts[(n + i)..]), out Vector<uint> b0, out Vector<uint> b1);
                (a0 | (b0 << 16)).CopyTo(u[i..]);
                (a1 | (b1 << 16)).CopyTo(u[(i + h)..]);
            }

            for (; i < n; i++) u[i] = shorts[i] | (uint)shorts[n + i] << 16;
            return;
        }

        throw new NotSupportedException($"{typeof(TIn).Name} in {typeof(TAcc).Name} lanes");

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Four(ReadOnlySpan<byte> row, int at, int n, out Vector<uint> a, out Vector<uint> b, out Vector<uint> c, out Vector<uint> d)
        {
            Vector.Widen(new Vector<byte>(row[at..]), out Vector<ushort> lo, out Vector<ushort> hi);
            Vector.Widen(lo, out a, out b);
            Vector.Widen(hi, out c, out d);
        }
    }

    // B packed: strip s (columns s·2w ..), group g (rows g·G ..) at (s·kg + g)·2w, the last strip's missing columns and
    // the last group's missing rows zero. Each group's rows are read where they are stored (or loaded into lanes once)
    // and interleaved into units, then scattered into the strips: the load and the packing one pass.
    // Bᵀ stored (TransB): column j of B is row j of Bᵀ, so a group's unit of column j is g values contiguous there; a task
    // reads each Bᵀ row's span of its groups once and lays the units into the strips.
    private struct PackBody<TN, TIn, TAcc>(byte[] b, TIn[]? bLanes, TAcc[] packed, int n, int k, int strip, int kg, int bOff = 0, bool transB = false, ConvArgs? image = null) : Team.IBody
        where TN : struct, IAdad<TN, TIn> where TIn : unmanaged where TAcc : unmanaged
    {
        public readonly void Run(int first, int last)
        {
            if (transB) { RunTransposed(first, last); return; }
            int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>(), strips = (n + strip - 1) / strip;
            bool direct = Lanes.Direct<TN, TIn>();
            var pool = ArrayPool<byte>.Shared;
            int rowsBytes = g * n * Unsafe.SizeOf<TIn>(), unitsBytes = n * Unsafe.SizeOf<TAcc>();
            byte[]? rented = rowsBytes + unitsBytes <= 32768 ? null : pool.Rent(rowsBytes + unitsBytes);
            Span<byte> scratch = rented is null ? stackalloc byte[rowsBytes + unitsBytes] : rented.AsSpan(0, rowsBytes + unitsBytes);
            var rowsBuffer = MemoryMarshal.Cast<byte, TIn>(scratch[..rowsBytes]);
            var units = MemoryMarshal.Cast<byte, TAcc>(scratch[rowsBytes..]);
            ReadOnlySpan<TIn> stored = direct ? MemoryMarshal.Cast<byte, TIn>(b.AsSpan()) : bLanes;
            TIn[]? line = image is not null && !direct ? ArrayPool<TIn>.Shared.Rent(image.W) : null;
            for (int grp = first; grp < last; grp++)
            {
                int row0 = grp * g, rows = Math.Min(g, k - row0);
                ReadOnlySpan<TIn> group = rowsBuffer;
                if (image is not null)
                {
                    for (int t = 0; t < g; t++)
                    {
                        var dst = rowsBuffer.Slice(t * n, n);
                        if (t >= rows) dst.Clear();
                        else ImageRow<TN, TIn>(b, direct ? stored : default, line, image, row0 + t, dst);
                    }
                }
                else if (rows == g && !stored.IsEmpty) group = stored.Slice(bOff + row0 * n, g * n);
                else
                {
                    for (int t = 0; t < g; t++)
                    {
                        var dst = rowsBuffer.Slice(t * n, n);
                        if (t >= rows) dst.Clear();
                        else if (!stored.IsEmpty) stored.Slice(bOff + (row0 + t) * n, n).CopyTo(dst);
                        else TN.Load(b, bOff + (row0 + t) * n, dst);
                    }
                }

                Interleave(group, units, n);
                for (int s = 0; s < strips; s++)
                {
                    var dst = packed.AsSpan((s * kg + grp) * strip, strip);
                    int j = s * strip, len = Math.Min(strip, n - j);
                    units.Slice(j, len).CopyTo(dst);
                    if (len < strip) dst[len..].Clear();
                }
            }

            if (rented is not null) pool.Return(rented);
            if (line is not null) ArrayPool<TIn>.Shared.Return(line);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private readonly void RunTransposed(int first, int last)
        {
            int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>(), strips = (n + strip - 1) / strip, groups = last - first;
            int p0 = first * g, len = Math.Min(k, last * g) - p0;
            ReadOnlySpan<TIn> stored = Lanes.Direct<TN, TIn>() ? MemoryMarshal.Cast<byte, TIn>(b.AsSpan()) : default;
            TIn[]? line = image is not null && stored.IsEmpty ? ArrayPool<TIn>.Shared.Rent(image.W) : null;
            int bytes = groups * g * Unsafe.SizeOf<TIn>();
            byte[]? rented = bytes <= 16384 ? null : ArrayPool<byte>.Shared.Rent(bytes);
            Span<byte> scratch = rented is null ? stackalloc byte[bytes] : rented.AsSpan(0, bytes);
            var row = MemoryMarshal.Cast<byte, TIn>(scratch);
            var units = MemoryMarshal.Cast<byte, TAcc>(scratch);
            row[len..].Clear();                                                // the last group's rows past k
            ref TAcc dst = ref MemoryMarshal.GetArrayDataReference(packed);
            for (int j = 0; j < strips * strip; j++)
            {
                nint at = (nint)(j / strip * kg + first) * strip + j % strip;
                if (j >= n)
                {
                    for (int q = 0; q < groups; q++) Unsafe.Add(ref dst, at + (nint)q * strip) = default;
                    continue;
                }

                if (image is not null) ImageRange<TN, TIn>(b, stored, line, image, j, p0, row[..len]);   // Bᵀ's row j: the columns' row j
                else if (!stored.IsEmpty) stored.Slice(bOff + j * k + p0, len).CopyTo(row);
                else TN.Load(b, bOff + j * k + p0, row[..len]);
                for (int q = 0; q < groups; q++) Unsafe.Add(ref dst, at + (nint)q * strip) = units[q];
            }

            if (rented is not null) ArrayPool<byte>.Shared.Return(rented);
            if (line is not null) ArrayPool<TIn>.Shared.Return(line);
        }
    }

    // One tile of C: rows [r0, r1), strips [s0, s1). A's 6-row blocks come from the stored bytes per panel (in place when
    // they are lanes, else loaded row by row), or from aLanes when A's rows do not start on whole bytes. When C is not
    // stored as lanes, the tile's values are stored the moment its last panel ends.
    // Aᵀ stored (TransA): the panel's k-row p of Aᵀ holds the tile's rows contiguously, so each block takes R values a
    // k-row straight into its [group][R] layout (no row-major staging, no transpose step); types whose stored bytes are
    // not the lane load the tile's span of each k-row once per panel.
    private struct ProductBody<TN, TIn, TAcc, TL>(byte[] a, TIn[]? aLanes, TAcc[] packed, TAcc[] cLanes, byte[] c, bool storeTiles,
        int m, int n, int k, int kg, int tileRows, int tileStrips, int rowTiles, int aOff = 0, bool transA = false, int cLanesOff = 0, int cOff = 0)
        : Team.IBody where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>(), kcg0 = KC / g;
            int w = Vector<TAcc>.Count, strip = 2 * w, strips = (n + strip - 1) / strip;
            // The tile's A panel: every 6-row block of this panel of k packed first (it stays in L2), then each B strip
            // (L1-resident) swept over all the blocks.
            int blocksPerTile = (tileRows + R - 1) / R;
            var panel = Workspace<TAcc>.Rent(Workspace.Panel, blocksPerTile * kcg0 * R);
            Span<TIn> rowsIn = stackalloc TIn[KC * R];
            Span<TAcc> spill = stackalloc TAcc[R * strip];
            ReadOnlySpan<TIn> direct = Lanes.Direct<TN, TIn>() ? MemoryMarshal.Cast<byte, TIn>(a.AsSpan()) : aLanes;
            TIn[]? kRows = transA && direct.IsEmpty ? Workspace<TIn>.Rent(Workspace.KRows, KC * tileRows) : null;
            for (int t = first; t < last; t++)
            {
                int r0 = t % rowTiles * tileRows, r1 = Math.Min(m, r0 + tileRows);
                int s0 = t / rowTiles * tileStrips, s1 = Math.Min(strips, s0 + tileStrips);
                for (int pc = 0; pc < k; pc += KC)
                {
                    int kc = Math.Min(KC, k - pc), kcg = (kc + g - 1) / g, padded = kcg * g;
                    int blocks = (r1 - r0 + R - 1) / R;
                    if (transA)
                    {
                        // The tile's rows of each k-row of the panel: in place, or loaded once.
                        ReadOnlySpan<TIn> src = direct;
                        int at = aOff + pc * m + r0, pitch = m;
                        if (kRows is not null)
                        {
                            for (int p = 0; p < kc; p++) TN.Load(a, aOff + (pc + p) * m + r0, kRows.AsSpan(p * tileRows, r1 - r0));
                            src = kRows;
                            at = 0;
                            pitch = tileRows;
                        }

                        for (int bi = 0; bi < blocks; bi++)
                        {
                            int rows = Math.Min(R, r1 - r0 - bi * R);
                            var block = MemoryMarshal.Cast<TAcc, TIn>(panel.AsSpan(bi * kcg0 * R, kcg * R));
                            PanelOfTransposed(src, at + bi * R, pitch, block, kc, padded, rows, g);
                        }
                    }
                    else
                    {
                        for (int bi = 0; bi < blocks; bi++)
                        {
                            int i = r0 + bi * R, rows = Math.Min(R, r1 - i);
                            // A's block, [group][R]: each row's panel slice read (or loaded) contiguously, padded to whole
                            // groups, then laid across; the rows past the last zero.
                            for (int r = 0; r < R; r++)
                            {
                                var dst = rowsIn.Slice(r * KC, padded);
                                if (r >= rows) { dst.Clear(); continue; }
                                if (!direct.IsEmpty) direct.Slice(aOff + (i + r) * k + pc, kc).CopyTo(dst);
                                else TN.Load(a, aOff + (i + r) * k + pc, dst[..kc]);
                                if (padded > kc) dst[kc..].Clear();
                            }

                            Transpose(MemoryMarshal.Cast<TIn, TAcc>(rowsIn), panel.AsSpan(bi * kcg0 * R, kcg0 * R), kcg, kcg0);
                        }
                    }

                    for (int s = s0; s < s1; s++)
                    {
                        var bstrip = packed.AsSpan((s * kg + pc / g) * strip, kcg * strip);
                        for (int bi = 0; bi < blocks; bi++)
                        {
                            int i = r0 + bi * R, rows = Math.Min(R, r1 - i);
                            Micro<TAcc, TL>(panel.AsSpan(bi * kcg0 * R, kcg * R), bstrip, cLanes, i, rows, s * strip, n, kcg, pc == 0, spill, cLanesOff);
                        }
                    }
                }

                if (storeTiles)
                {
                    int j0 = s0 * strip, j1 = Math.Min(n, s1 * strip);
                    for (int r = r0; r < r1; r++) TN.Store(cLanes.AsSpan(r * n + j0, j1 - j0), c, cOff + r * n + j0);
                }
            }

            Workspace<TAcc>.Return(Workspace.Panel, panel);
            if (kRows is not null) Workspace<TIn>.Return(Workspace.KRows, kRows);
        }
    }

    // Row p = (c, kh, kw) of the convolution's columns from the image (C × H × W): dst[oh·OW + ow] =
    // x[c][oh·sh − ph + kh·dh][ow·sw − pw + kw·dw], 0 off the image; an image row read in place (lanes) or loaded once.
    private static void ImageRow<TN, TIn>(byte[] x, ReadOnlySpan<TIn> stored, TIn[]? line, ConvArgs a, int p, Span<TIn> dst)
        where TN : struct, IAdad<TN, TIn> where TIn : unmanaged
    {
        int khw = a.Kh * a.Kw, c = p / khw, kh = p % khw / a.Kw, kw = p % a.Kw;
        var (oh1, ow1) = a.Out;
        int shift = kw * a.Dw - a.Pw;                                           // iw = ow·sw + shift
        for (int oh = 0; oh < oh1; oh++)
        {
            var row = dst.Slice(oh * ow1, ow1);
            int ih = oh * a.Sh - a.Ph + kh * a.Dh;
            if ((uint)ih >= (uint)a.H) { row.Clear(); continue; }
            ReadOnlySpan<TIn> src;
            if (!stored.IsEmpty) src = stored.Slice((c * a.H + ih) * a.W, a.W);
            else { TN.Load(x, (c * a.H + ih) * a.W, line.AsSpan(0, a.W)); src = line.AsSpan(0, a.W); }
            if (a.Sw == 1)
            {
                int lo = Math.Clamp(-shift, 0, ow1), hi = Math.Clamp(a.W - shift, lo, ow1);
                row[..lo].Clear();
                src.Slice(lo + shift, hi - lo).CopyTo(row[lo..]);
                row[hi..].Clear();
                continue;
            }

            for (int ow = 0; ow < ow1; ow++)
            {
                int iw = ow * a.Sw + shift;
                row[ow] = (uint)iw < (uint)a.W ? src[iw] : default;
            }
        }
    }

    // Columns from .. from + dst.Length of that row (dst[q − from] = the row's value q), for the transposed packing (a
    // k-range of the row a thread packs): the run's whole and partial output rows each taken as ImageRow takes a whole one.
    private static void ImageRange<TN, TIn>(byte[] x, ReadOnlySpan<TIn> stored, TIn[]? line, ConvArgs a, int p, int from, Span<TIn> dst)
        where TN : struct, IAdad<TN, TIn> where TIn : unmanaged
    {
        int khw = a.Kh * a.Kw, c = p / khw, kh = p % khw / a.Kw, kw = p % a.Kw;
        int ow1 = a.Out.OW;
        int shift = kw * a.Dw - a.Pw;                                           // iw = ow·sw + shift
        for (int q = from, end = from + dst.Length; q < end;)
        {
            int oh = q / ow1, ow0 = q % ow1, ow2 = Math.Min(ow1, ow0 + (end - q));
            var row = dst.Slice(q - from, ow2 - ow0);
            q += ow2 - ow0;
            int ih = oh * a.Sh - a.Ph + kh * a.Dh;
            if ((uint)ih >= (uint)a.H) { row.Clear(); continue; }
            ReadOnlySpan<TIn> src;
            if (!stored.IsEmpty) src = stored.Slice((c * a.H + ih) * a.W, a.W);
            else { TN.Load(x, (c * a.H + ih) * a.W, line.AsSpan(0, a.W)); src = line.AsSpan(0, a.W); }
            if (a.Sw == 1)
            {
                int lo = Math.Clamp(-shift, ow0, ow2), hi = Math.Clamp(a.W - shift, lo, ow2);
                row[..(lo - ow0)].Clear();
                src.Slice(lo + shift, hi - lo).CopyTo(row[(lo - ow0)..]);
                row[(hi - ow0)..].Clear();
                continue;
            }

            for (int ow = ow0; ow < ow2; ow++)
            {
                int iw = ow * a.Sw + shift;
                row[ow - ow0] = (uint)iw < (uint)a.W ? src[iw] : default;
            }
        }
    }

    // A block [group][R] (as input values: unit (q, r) holds k-values q·g .. q·g + g − 1 of row r) from Aᵀ: k-row p's R rows
    // at src[at + p·pitch ..]; k-values past kc and rows past `rows` zero.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PanelOfTransposed<TIn>(ReadOnlySpan<TIn> src, int at, int pitch, Span<TIn> block, int kc, int padded, int rows, int g) where TIn : unmanaged
    {
        if (g == 1)
        {
            for (int p = 0; p < kc; p++)
            {
                var dst = block.Slice(p * R, R);
                if (rows == R) src.Slice(at + p * pitch, R).CopyTo(dst);
                else { src.Slice(at + p * pitch, rows).CopyTo(dst); dst[rows..].Clear(); }
            }

            return;
        }

        for (int p = 0; p < padded; p++)
        {
            int q = p / g, e = p % g;
            for (int r = 0; r < R; r++) block[(q * R + r) * g + e] = p < kc && r < rows ? src[at + p * pitch + r] : default;
        }
    }

    // rows [r][stride] (kcg of each used) into block [p][R].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose<TAcc>(ReadOnlySpan<TAcc> rows, Span<TAcc> block, int kcg, int stride) where TAcc : unmanaged
    {
        ref TAcc src = ref MemoryMarshal.GetReference(rows);
        ref TAcc dst = ref MemoryMarshal.GetReference(block);
        for (int p = 0; p < kcg; p++)
        {
            nint d = p * R;
            Unsafe.Add(ref dst, d) = Unsafe.Add(ref src, p);
            Unsafe.Add(ref dst, d + 1) = Unsafe.Add(ref src, stride + p);
            Unsafe.Add(ref dst, d + 2) = Unsafe.Add(ref src, 2 * stride + p);
            Unsafe.Add(ref dst, d + 3) = Unsafe.Add(ref src, 3 * stride + p);
            Unsafe.Add(ref dst, d + 4) = Unsafe.Add(ref src, 4 * stride + p);
            Unsafe.Add(ref dst, d + 5) = Unsafe.Add(ref src, 5 * stride + p);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Micro<TLane, TL>(ReadOnlySpan<TLane> ablock, ReadOnlySpan<TLane> bstrip, TLane[] c, int i, int rows, int j, int n, int kc, bool first, Span<TLane> spill, int cOff = 0)
        where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int w = Vector<TLane>.Count;
        bool whole = rows == R && j + 2 * w <= n;
        ref TLane ra = ref MemoryMarshal.GetReference(ablock);
        ref TLane rb = ref MemoryMarshal.GetReference(bstrip);
        ref TLane rc = ref MemoryMarshal.GetArrayDataReference(c);
        Vector<TLane> x00, x01, x10, x11, x20, x21, x30, x31, x40, x41, x50, x51;
        if (first || !whole)
        {
            x00 = x01 = x10 = x11 = x20 = x21 = x30 = x31 = x40 = x41 = x50 = x51 = Vector<TLane>.Zero;
        }
        else
        {
            nuint c0 = (nuint)(cOff + i * n + j);
            x00 = Vector.LoadUnsafe(ref rc, c0); x01 = Vector.LoadUnsafe(ref rc, c0 + (nuint)w);
            x10 = Vector.LoadUnsafe(ref rc, c0 + (nuint)n); x11 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(n + w));
            x20 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(2 * n)); x21 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(2 * n + w));
            x30 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(3 * n)); x31 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(3 * n + w));
            x40 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(4 * n)); x41 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(4 * n + w));
            x50 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(5 * n)); x51 = Vector.LoadUnsafe(ref rc, c0 + (nuint)(5 * n + w));
        }

        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector.LoadUnsafe(ref rb, (nuint)(p * 2 * w));
            var b1 = Vector.LoadUnsafe(ref rb, (nuint)(p * 2 * w + w));
            nuint ap = (nuint)(p * R);
            var a0 = new Vector<TLane>(Unsafe.Add(ref ra, ap));
            x00 = TL.Fma(a0, b0, x00); x01 = TL.Fma(a0, b1, x01);
            var a1 = new Vector<TLane>(Unsafe.Add(ref ra, ap + 1));
            x10 = TL.Fma(a1, b0, x10); x11 = TL.Fma(a1, b1, x11);
            var a2 = new Vector<TLane>(Unsafe.Add(ref ra, ap + 2));
            x20 = TL.Fma(a2, b0, x20); x21 = TL.Fma(a2, b1, x21);
            var a3 = new Vector<TLane>(Unsafe.Add(ref ra, ap + 3));
            x30 = TL.Fma(a3, b0, x30); x31 = TL.Fma(a3, b1, x31);
            var a4 = new Vector<TLane>(Unsafe.Add(ref ra, ap + 4));
            x40 = TL.Fma(a4, b0, x40); x41 = TL.Fma(a4, b1, x41);
            var a5 = new Vector<TLane>(Unsafe.Add(ref ra, ap + 5));
            x50 = TL.Fma(a5, b0, x50); x51 = TL.Fma(a5, b1, x51);
        }

        if (whole)
        {
            nuint c0 = (nuint)(cOff + i * n + j);
            x00.StoreUnsafe(ref rc, c0); x01.StoreUnsafe(ref rc, c0 + (nuint)w);
            x10.StoreUnsafe(ref rc, c0 + (nuint)n); x11.StoreUnsafe(ref rc, c0 + (nuint)(n + w));
            x20.StoreUnsafe(ref rc, c0 + (nuint)(2 * n)); x21.StoreUnsafe(ref rc, c0 + (nuint)(2 * n + w));
            x30.StoreUnsafe(ref rc, c0 + (nuint)(3 * n)); x31.StoreUnsafe(ref rc, c0 + (nuint)(3 * n + w));
            x40.StoreUnsafe(ref rc, c0 + (nuint)(4 * n)); x41.StoreUnsafe(ref rc, c0 + (nuint)(4 * n + w));
            x50.StoreUnsafe(ref rc, c0 + (nuint)(5 * n)); x51.StoreUnsafe(ref rc, c0 + (nuint)(5 * n + w));
            return;
        }

        // An edge block: through the spill, its valid rows and columns added to (or, the first panel, written over) C.
        ref TLane rs = ref MemoryMarshal.GetReference(spill);
        x00.StoreUnsafe(ref rs, 0); x01.StoreUnsafe(ref rs, (nuint)w);
        x10.StoreUnsafe(ref rs, (nuint)(2 * w)); x11.StoreUnsafe(ref rs, (nuint)(3 * w));
        x20.StoreUnsafe(ref rs, (nuint)(4 * w)); x21.StoreUnsafe(ref rs, (nuint)(5 * w));
        x30.StoreUnsafe(ref rs, (nuint)(6 * w)); x31.StoreUnsafe(ref rs, (nuint)(7 * w));
        x40.StoreUnsafe(ref rs, (nuint)(8 * w)); x41.StoreUnsafe(ref rs, (nuint)(9 * w));
        x50.StoreUnsafe(ref rs, (nuint)(10 * w)); x51.StoreUnsafe(ref rs, (nuint)(11 * w));
        int cols = Math.Min(2 * w, n - j);
        for (int r = 0; r < rows; r++)
        {
            for (int q = 0; q < cols; q++)
            {
                ref TLane dst = ref c[cOff + (i + r) * n + j + q];
                dst = first ? spill[r * 2 * w + q] : TL.Add(dst, spill[r * 2 * w + q]);
            }
        }
    }

    public static void MatMul<TN, TIn, TAcc, TL>(byte[] a, byte[] b, byte[] c, int m, int n, int k)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc> =>
        MatMul<TN, TIn, TAcc, TL>(a, b, c, m, n, k, default);

    /// <summary>C = A·B with the operands where `o` says they are stored (transposed, at offsets): the packing reads them there.</summary>
    public static void MatMul<TN, TIn, TAcc, TL>(byte[] a, byte[] b, byte[] c, int m, int n, int k, Operands o)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int g = Unsafe.SizeOf<TAcc>() / Unsafe.SizeOf<TIn>(), kg = (k + g - 1) / g;
        int w = Vector<TAcc>.Count, strip = 2 * w, strips = (n + strip - 1) / strip;
        bool aDirect = Lanes.Direct<TN, TIn>(), cDirect = Lanes.Direct<TN, TAcc>();
        if (o.Plain && cDirect && c.Length / Unsafe.SizeOf<TAcc>() != m * n) throw new ArgumentException("C must be exactly m·n values.");
        if (!o.Plain && cDirect && c.Length / Unsafe.SizeOf<TAcc>() < (long)o.COff + m * n) throw new ArgumentException("C must hold m·n values from its offset.");
        // Values smaller than a byte: rows read where they are stored need every row to start on a whole byte.
        int per = TN.Bits >= 8 ? 1 : 8 / TN.Bits;
        bool aRows = aDirect || k % per == 0, nRows = aDirect || n % per == 0;
        if (!o.Plain && !(aDirect || (m % per == 0 && n % per == 0 && k % per == 0))) throw new ArgumentException("operands read where they are stored need every row on a whole byte.");
        var inPool = ArrayPool<TIn>.Shared;
        var accPool = ArrayPool<TAcc>.Shared;
        TIn[]? al = null, bl = null;
        TAcc[]? cl = null;
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var packed = Workspace<TAcc>.Rent(Workspace.Packed, strips * strip * kg);
        try
        {
            // Rows that do not start on whole bytes: that matrix loaded whole first.
            if (!aRows)
            {
                al = Workspace<TIn>.Rent(Workspace.ALanes, m * k);
                Team.For(m * k, Math.Max(4096, m * k / (Threads * 2)) / 64 * 64, m * k, new LoadBody<TN, TIn>(a, al));
            }

            if (!nRows && o.ImageB is null)
            {
                bl = Workspace<TIn>.Rent(Workspace.BLanes, k * n);
                Team.For(k * n, Math.Max(4096, k * n / (Threads * 2)) / 64 * 64, k * n, new LoadBody<TN, TIn>(b, bl));
            }

            t0 = Mark(t0, 0);
            Team.For(kg, Math.Max(1, kg / (Threads * 2)), (long)k * n, new PackBody<TN, TIn, TAcc>(b, bl, packed, n, k, strip, kg, o.BOff, o.TransB, o.ImageB));
            t0 = Mark(t0, 1);

            // C in lanes (the stored bytes themselves when they are lanes).
            cl = cDirect ? null : Workspace<TAcc>.Rent(Workspace.CLanes, m * n);
            var cLanes = cl ?? Unsafe.As<TAcc[]>(c);

            // Tiles of C: about six per thread (cores of unequal speed balance through the shared counter), rows in whole blocks, columns in whole strips.
            int tileStrips = Math.Max(1, Math.Min(strips, 256 / strip * 2));
            int colTiles = (strips + tileStrips - 1) / tileStrips;
            int wantRows = Math.Max(1, Threads * 6 / colTiles);
            int tileRows = Math.Max(R, (m + wantRows - 1) / wantRows + R - 1) / R * R;
            int rowTiles = (m + tileRows - 1) / tileRows;
            bool storeTiles = cl is not null && nRows && strip % TN.Granularity == 0 && n % TN.Granularity == 0;
            Team.For(rowTiles * colTiles, 1, (long)m * n * k,
                new ProductBody<TN, TIn, TAcc, TL>(a, al, packed, cLanes, c, storeTiles, m, n, k, kg, tileRows, tileStrips, rowTiles, o.AOff, o.TransA, cl is null ? o.COff : 0, o.COff));
            t0 = Mark(t0, 2);

            if (cl is not null && !storeTiles)
            {
                int unit = Math.Max(64, TN.Granularity);                          // chunks of whole storage units: two tasks never share a block
                Team.For(m * n, Math.Max(4096, m * n / (Threads * 2)) / unit * unit, m * n, new StoreBody<TN, TAcc>(cl, c, o.COff));
            }

            Mark(t0, 3);
        }
        finally
        {
            Workspace<TAcc>.Return(Workspace.Packed, packed);
            if (al is not null) Workspace<TIn>.Return(Workspace.ALanes, al);
            if (bl is not null) Workspace<TIn>.Return(Workspace.BLanes, bl);
            if (cl is not null) Workspace<TAcc>.Return(Workspace.CLanes, cl);
        }
    }

    // ------------------------------------------------------------------ softmax and rms_norm over rows

    public struct SoftmaxVisitor(byte[] x, byte[] y, int rows, int cols) : IFloatLaneVisitor
    {
        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), (long)rows * cols * 8, new SoftmaxBody<TN, TLane, TL>(x, y, cols));
    }

    private struct SoftmaxBody<TN, TLane, TL>(byte[] x, byte[] y, int cols) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            TLane[]? rented = TN.Direct || cols <= 4096 ? null : ArrayPool<TLane>.Shared.Rent(cols);
            Span<TLane> row = TN.Direct ? default : rented is null ? stackalloc TLane[cols] : rented.AsSpan(0, cols);
            try
            {
                for (int r = first; r < last; r++)
                {
                    Span<TLane> v = row;
                    if (TN.Direct)
                    {
                        var src = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols, cols);
                        v = MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(r * cols, cols);
                        Softmax<TLane, TL>(src, v);
                    }
                    else
                    {
                        v = row;
                        TN.Load(x, r * cols, v);
                        Softmax<TLane, TL>(v, v);
                        TN.Store(v, y, r * cols);
                    }
                }
            }
            finally
            {
                if (rented is not null) ArrayPool<TLane>.Shared.Return(rented);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Softmax<TLane, TL>(ReadOnlySpan<TLane> x, Span<TLane> y) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int w = Vector<TLane>.Count, vec = x.Length / w * w;
        ref TLane rx = ref MemoryMarshal.GetReference(x);
        ref TLane ry = ref MemoryMarshal.GetReference(y);
        TLane max = TL.NegativeInfinity;
        if (vec > 0)
        {
            var m0 = new Vector<TLane>(max);
            var m1 = m0;
            int i = 0;
            for (; i + 2 * w <= vec; i += 2 * w)
            {
                m0 = Vector.Max(m0, Vector.LoadUnsafe(ref rx, (nuint)i));
                m1 = Vector.Max(m1, Vector.LoadUnsafe(ref rx, (nuint)(i + w)));
            }

            for (; i < vec; i += w) m0 = Vector.Max(m0, Vector.LoadUnsafe(ref rx, (nuint)i));
            m0 = Vector.Max(m0, m1);
            for (int l = 0; l < w; l++) max = TL.Max(max, m0[l]);
        }

        for (int i = vec; i < x.Length; i++) max = TL.Max(max, x[i]);
        var mv = new Vector<TLane>(max);
        var s0 = Vector<TLane>.Zero;
        var s1 = s0;
        int j = 0;
        for (; j + 2 * w <= vec; j += 2 * w)
        {
            var e0 = TL.Exp(Vector.LoadUnsafe(ref rx, (nuint)j) - mv);
            var e1 = TL.Exp(Vector.LoadUnsafe(ref rx, (nuint)(j + w)) - mv);
            e0.StoreUnsafe(ref ry, (nuint)j);
            e1.StoreUnsafe(ref ry, (nuint)(j + w));
            s0 += e0;
            s1 += e1;
        }

        for (; j < vec; j += w)
        {
            var e0 = TL.Exp(Vector.LoadUnsafe(ref rx, (nuint)j) - mv);
            e0.StoreUnsafe(ref ry, (nuint)j);
            s0 += e0;
        }

        TLane sum = Vector.Sum(s0 + s1);
        for (int i = vec; i < x.Length; i++)
        {
            var e = TL.Exp(TL.Sub(x[i], max));
            y[i] = e;
            sum = TL.Add(sum, e);
        }

        TLane scale = TL.Div(TL.From(1), sum);
        var sv = new Vector<TLane>(scale);
        int q = 0;
        for (; q < vec; q += w) (Vector.LoadUnsafe(ref ry, (nuint)q) * sv).StoreUnsafe(ref ry, (nuint)q);
        for (; q < y.Length; q++) y[q] = TL.Mul(y[q], scale);
    }

    public struct RmsNormVisitor(byte[] x, byte[] y, int rows, int cols, double eps) : IFloatLaneVisitor
    {
        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            Team.For(rows, RowChunk<TN>(rows, cols, rows / (Threads * 4)), (long)rows * cols * 2, new RmsBody<TN, TLane, TL>(x, y, cols, TL.From(eps)));
    }

    private struct RmsBody<TN, TLane, TL>(byte[] x, byte[] y, int cols, TLane eps) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Run(int first, int last)
        {
            TLane[]? rented = TN.Direct || cols <= 4096 ? null : ArrayPool<TLane>.Shared.Rent(cols);
            Span<TLane> row = TN.Direct ? default : rented is null ? stackalloc TLane[cols] : rented.AsSpan(0, cols);
            TLane count = TL.From(cols);
            try
            {
                for (int r = first; r < last; r++)
                {
                    ReadOnlySpan<TLane> src = row;
                    Span<TLane> dst = row;
                    if (TN.Direct)
                    {
                        src = MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(r * cols, cols);
                        dst = MemoryMarshal.Cast<byte, TLane>(y.AsSpan()).Slice(r * cols, cols);
                    }
                    else
                    {
                        TN.Load(x, r * cols, row);
                        src = dst = row;
                    }

                    int w = Vector<TLane>.Count, vec = cols / w * w, i = 0;
                    ref TLane rs = ref MemoryMarshal.GetReference(src);
                    ref TLane rd = ref MemoryMarshal.GetReference(dst);
                    var s0 = Vector<TLane>.Zero;
                    var s1 = s0;
                    for (; i + 2 * w <= vec; i += 2 * w)
                    {
                        var v0 = Vector.LoadUnsafe(ref rs, (nuint)i);
                        var v1 = Vector.LoadUnsafe(ref rs, (nuint)(i + w));
                        s0 = TL.Fma(v0, v0, s0);
                        s1 = TL.Fma(v1, v1, s1);
                    }

                    for (; i < vec; i += w)
                    {
                        var v0 = Vector.LoadUnsafe(ref rs, (nuint)i);
                        s0 = TL.Fma(v0, v0, s0);
                    }

                    TLane sum = Vector.Sum(s0 + s1);
                    for (; i < cols; i++) sum = TL.Fma(src[i], src[i], sum);
                    TLane scale = TL.Div(TL.From(1), TL.Sqrt(TL.Add(TL.Div(sum, count), eps)));
                    var sv = new Vector<TLane>(scale);
                    int q = 0;
                    for (; q < vec; q += w) (Vector.LoadUnsafe(ref rs, (nuint)q) * sv).StoreUnsafe(ref rd, (nuint)q);
                    for (; q < cols; q++) dst[q] = TL.Mul(src[q], scale);
                    if (!TN.Direct) TN.Store(row, y, r * cols);
                }
            }
            finally
            {
                if (rented is not null) ArrayPool<TLane>.Shared.Return(rented);
            }
        }
    }
}
