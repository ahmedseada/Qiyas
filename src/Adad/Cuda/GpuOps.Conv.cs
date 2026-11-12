// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's convolution (Conv.cs). The forward is fused (item 4g, implicit GEMM: no columns in memory):
//   the tensor and dot routes   the image packed straight into the kind's packed rows (Ptx.PackTransposed with conv: the
//                               index map in place of the load), the weights packed by rows, the same kernel, padding
//                               and C path as the composition: the same result bit for bit;
//   the plain lanes (f32, f64, the integer lanes, block formats in f32)   the plain product kernel with its B tile
//                               gathered from the image (Ptx.Gemm with conv), or the composition (im2col, then the
//                               product: Gemm2, the sliced float64 route) where this device measures that faster, once
//                               per shape and type by Measure.Faster, the decision in --report.
// The backward is the composition (im2col and col2im around the products), on the type's own tensors: the columns of
// dw's product gathered from the image by the product kernel (transposed), dx's columns an intermediate in lane storage.
// Every type: dx's sums (w·dy, OCg·Kh·Kw terms a value) and dw's (dy·x over the images' positions; db's dy's alone) each in
// the lane SumLane gives for their bounds (a float type's float lane; an integer type's exact one), each output stored as
// the type once (BackwardLanes).
// Images, groups or any term of the product's epilogue (the float types) take the general path (Conv.cs): the columns of
// every image's channels into lane storage (im2col reading the image where it is stored), a stored-mode product a group
// over the images into the lane's sums, then the epilogue's store over the output (EpilogueStore), y stored once. Its
// backward (every type): dcols = w_gᵀ·dy_g, dx = col2im(dcols); dy and the columns side by side over the images in the
// lane (exact moves), dw_g = dy_g·cols_gᵀ, db = dy's rows times ones; dw and db stored as the type once.
using Adad.Abstraction;
// The integer types by their names here, not System's.
using Int32 = Adad.Abstraction.Int32;
using Int64 = Adad.Abstraction.Int64;
using UInt32 = Adad.Abstraction.UInt32;
using UInt64 = Adad.Abstraction.UInt64;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>The gate's override: false the composition, true the fused path everywhere, null fused (the plain lanes as measured).</summary>
    internal bool? FusedConv;

    private readonly Dictionary<string, bool> _convFused = [];

    /// <summary>y = the epilogue's y of conv(x, w) (IConvolution.Conv; ConvArgs.Single with no term: the one-image path);
    /// returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ProductEpilogue e, ulong x, ulong w, ulong y, ProductTerms<ulong> t) where TN : struct, IAdad<TN>
    {
        a.Validate();
        var terms = t.Validate(e, a.Output);
        if (a.Single && terms == default) return Conv<TN>(a, x, w, y);
        var f = Fmt.Of<TN>();
        if (!f.IsFloat) throw new NotSupportedException($"conv: images, groups or an epilogue term take a float type, not {f.Name}");
        var lane = RowLane(f);
        if (a.YValues == 0) return lane.Name;
        a.ColumnsFit();
        var lf = Ptx.LaneFormat(lane);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = gpu.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong cols = Rent(a.ColValues * lane.Size), sums = Rent(a.YValues * lane.Size);
            LaneLayout(lane, new LayoutArgs { Name = "im2col", X = [], P = a.ImagesP, Output = [], Count = (int)a.ColValues, InputCount = (int)a.XValues }, x, cols, f);
            for (int g = 0; g < a.Groups; g++)
                ProductStored(new ProductView(a.OCg, a.Cols, a.Rows, false, false, a.N, 0, (long)a.Groups * a.Rows * a.Cols, (long)a.OC * a.Cols), lane,
                    w, f, (long)g * a.OCg * a.Rows, cols, lf, (long)g * a.Rows * a.Cols, sums + (ulong)((long)g * a.OCg * a.Cols * lane.Size), null);
            EpilogueStore(f, lane, a.Output, e, terms, t, sums, y, a.Cols, a.OC);
        }
        finally { foreach (var p in rented) gpu.Return(p); }

        return lane.Name;
    }

    /// <summary>A Ptx.StoreEach kernel over count values of f: a thread per stored word, a warp (a CTA) per block.</summary>
    private static void LaunchEach(IntPtr fn, Fmt f, long count, params ulong[] args)
    {
        if (count == 0) return;
        if (f.IsBlock && Ptx.BlockByCta(f)) Gpu.Launch(fn, (uint)((count + f.BlockValues - 1) / f.BlockValues), 1, (uint)f.BlockValues, 1, 0, args);
        else if (f.IsBlock) Gpu.Launch(fn, BlockGrid(f, count), 1, 256, 1, 0, args);
        else Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, args);
    }

    // A layout into lane storage (the lane's own type), its input read as inFmt where it is stored (null: lane storage too).
    private void LaneLayout(Lane lane, LayoutArgs k, ulong x, ulong output, Fmt? inFmt)
    {
        if (lane == Lane.F32) Layout<Float32>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.F64) Layout<Float64>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.S32) Layout<Int32>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.U32) Layout<UInt32>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.S64) Layout<Int64>(k, x, 0, output, 0, inFmt);
        else Layout<UInt64>(k, x, 0, output, 0, inFmt);
    }

    // count ones in the lane's storage.
    private static byte[] LaneOnes(Lane lane, int count)
    {
        var one = new double[count];
        Array.Fill(one, 1.0);
        if (lane == Lane.F32) return Stored.StoreValues<Float32>(one);
        if (lane == Lane.F64) return Stored.StoreValues<Float64>(one);
        var bytes = new byte[(long)count * lane.Size];
        if (lane.Size == 4) System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(bytes.AsSpan()).Fill(1);
        else System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ulong>(bytes.AsSpan()).Fill(1);
        return bytes;
    }

    /// <summary>The backward's lanes (SumLane by the bounds): dx's (w·dy: OCg·Kh·Kw terms a value at most) and dw's and db's
    /// (dy·x and dy·1 over the images' positions); a float type's float lane for both.</summary>
    private (Lane DX, Lane DW) BackwardLanes<TN>(ConvArgs a, ulong x, ulong w, ulong dy) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsFloat) return (RowLane(f), RowLane(f));
        var ((wLo, wHi), (xLo, xHi)) = RangePair<TN>(w, (int)a.WValues, x, (int)a.XValues);
        var (dLo, dHi) = Range<TN>(dy, (int)a.YValues);
        return (SumLane(f, new Bounds(wLo, wHi, dLo, dHi, (long)a.OCg * a.Kh * a.Kw)),
                SumLane(f, new Bounds(dLo, dHi, Math.Min(xLo, 0), Math.Max(xHi, 1), (long)a.N * a.Cols)));
    }

    // The lane names a backward returns: dx's, and dw's after it where it differs.
    private static string LanesName(Lane dx, Lane dw) => dx == dw ? dx.Name : $"{dx.Name}; dw in {dw.Name}";

    /// <summary>The general backward (dx, dw, db: see the file's note); returns the lanes' names.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw, ulong db) where TN : struct, IAdad<TN>
    {
        a.Validate();
        if (a.Single && db == 0) return ConvBackward<TN>(a, x, w, dy, dx, dw);
        a.ColumnsFit();
        var f = Fmt.Of<TN>();
        var (lx, lw) = BackwardLanes<TN>(a, x, w, dy);
        Fmt fx = Ptx.LaneFormat(lx), fw = Ptx.LaneFormat(lw);
        int G = a.Groups, R = a.Rows, P = a.Cols, NP = a.N * a.Cols;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = gpu.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong cols = Rent(a.ColValues * lw.Size), dcols = Rent(a.ColValues * lx.Size), dyp = Rent((long)a.OC * NP * lw.Size), colsp = Rent(a.ColValues * lw.Size);
            ulong ldw = Rent(a.WValues * lw.Size);
            LaneLayout(lw, new LayoutArgs { Name = "im2col", X = [], P = a.ImagesP, Output = [], Count = (int)a.ColValues, InputCount = (int)a.XValues }, x, cols, f);
            for (int g = 0; g < G; g++)
                ProductStored(new ProductView(R, P, a.OCg, true, false, a.N, 0, (long)a.OC * P, (long)G * R * P), lx,
                    w, f, (long)g * a.OCg * R, dy, f, (long)g * a.OCg * P, dcols + (ulong)((long)g * R * P * lx.Size), null);                  // w_gᵀ · dy_g
            Layout<TN>(new LayoutArgs { Name = "col2im", X = [], P = a.ImagesP, Output = [], Count = (int)a.XValues, InputCount = (int)a.ColValues }, dcols, 0, dx, 0, fx);
            LaneLayout(lw, new LayoutArgs { Name = "strided", X = [], P = [1, a.OC, a.N, P, 0, P, a.OC * P, 1, 0], Output = [], Count = a.OC * NP, InputCount = (int)a.YValues }, dy, dyp, f);
            LaneLayout(lw, new LayoutArgs { Name = "strided", X = [], P = [1, G * R, a.N, P, 0, P, G * R * P, 1, 0], Output = [], Count = (int)a.ColValues, InputCount = (int)a.ColValues }, cols, colsp, null);
            for (int g = 0; g < G; g++)
                ProductStored(new ProductView(a.OCg, R, NP, false, true), lw, dyp, fw, (long)g * a.OCg * NP, colsp, fw, (long)g * R * NP,
                    ldw + (ulong)((long)g * a.OCg * R * lw.Size), null);                                                                    // dy_g · cols_gᵀ
            Back(f, lw, ldw, dw, a.WValues);
            if (db != 0)
            {
                // Each channel's Σ dy: its row of dy (every image's positions) times ones.
                ulong ones = Rent((long)NP * lw.Size), ldb = Rent((long)a.OC * lw.Size);
                Gpu.Upload<byte>(LaneOnes(lw, NP), ones);
                ProductStored(new ProductView(a.OC, 1, NP, false, false), lw, dyp, fw, 0, ones, fw, 0, ldb, null);
                Back(f, lw, ldb, db, a.OC);
            }
        }
        finally { foreach (var p in rented) gpu.Return(p); }

        return LanesName(lx, lw);
    }

    /// <summary>y = conv(x, w) of one image of one group; returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        if (FusedConv == false) return ConvComposed<TN>(a, x, w, y);
        var f = Fmt.Of<TN>();
        // The columns' values are the image's and 0 (padding): the image's range, with 0, bounds them.
        var ((wLo, wHi), (xLo, xHi)) = RangePair<TN>(w, (int)a.WValues, x, (int)a.XValues);
        var bounds = new Bounds(wLo, wHi, Math.Min(xLo, 0), Math.Max(xHi, 0), a.Rows);
        string laneName = ProductLane<TN>(bounds);
        if (!f.IsBlock && (laneName.EndsWith("mma", StringComparison.Ordinal) || laneName.EndsWith("dp", StringComparison.Ordinal)))
            return ConvPacked<TN>(a, x, w, y, laneName);
        var lane = LaneOf(laneName);
        // A block format's composition would encode the columns as blocks (each window's values quantized again): fused.
        // The composition's columns past 2^31 values: fused.
        if (FusedConv == true || f.IsBlock || a.ColValues > int.MaxValue) return ConvPlain<TN>(a, x, w, y, lane);
        string key = $"{f.Name}: {a.Shape}";
        if (!_convFused.TryGetValue(key, out bool fused))
        {
            ulong scratch = gpu.Rent(f.Words(a.YValues) * (f.WordBits / 8) + 64);
            try
            {
                var (tf, tc) = Measure.Faster(() => ConvPlain<TN>(a, x, w, scratch, lane), () => ConvComposed<TN>(a, x, w, scratch), Gpu.Synchronize);
                fused = _convFused[key] = tf <= tc;
                Note($"conv {key}: fused {tf:0.000} ms, im2col and the product {tc:0.000} ms: {(fused ? "fused" : "im2col")}");
            }
            finally { gpu.Return(scratch); }
        }

        return fused ? ConvPlain<TN>(a, x, w, y, lane) : ConvComposed<TN>(a, x, w, y);
    }

    /// <summary>y = conv(x, w) by the composition: im2col, then the product.</summary>
    private string ConvComposed<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        a.ColumnsFit();
        var f = Fmt.Of<TN>();
        ulong cols = gpu.Rent(f.Words((long)a.Rows * a.Cols) * (f.WordBits / 8));
        try
        {
            Layout<TN>(new LayoutArgs { Name = "im2col", X = [], P = a.P, Output = [], Count = a.Rows * a.Cols, InputCount = (int)a.XValues }, x, 0, cols);
            return MatMulView<TN>(new ProductView(a.OC, a.Cols, a.Rows, false, false), w, cols, y);
        }
        finally
        {
            gpu.Return(cols);
        }
    }

    private static ulong[] ConvArgsOf(ConvArgs a) => [Gpu.U(a.H), Gpu.U(a.W), Gpu.U(a.Kh), Gpu.U(a.Kw), Gpu.U(a.Sh), Gpu.U(a.Sw), Gpu.U(a.PadTop), Gpu.U(a.PadLeft), Gpu.U(a.Dh), Gpu.U(a.Dw), Gpu.U(a.Out.OW)];

    // The tensor / dot route with B packed from the image: the composition's steps (MatMul, MatMulPadded) with the columns'
    // packing reading the image: the same packed values, kernel, padding and C path, so the same result.
    private string ConvPacked<TN>(ConvArgs a, ulong x, ulong w, ulong y, string laneName) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = LaneOf(laneName);
        int m = a.OC, n = a.Cols, k = a.Rows;
        bool whole = m % 128 == 0 && n % 256 == 0 && k % 128 == 0;
        int M = whole ? m : (m + 127) / 128 * 128, N = whole ? n : (n + 255) / 256 * 256, K = whole ? k : (k + 127) / 128 * 128;
        var kind = KindOf(laneName, K);
        int kp = KPadded(kind, K);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = gpu.Rent(bytes); rented.Add(p); return p; }
        try
        {
            long aBytes = (long)M * kp * kind.Bits / 8, bBytes = (long)N * kp * kind.Bits / 8;
            ulong pa = Rent(aBytes), pb = Rent(bBytes);
            Gpu.Clear(pa, aBytes);
            Gpu.Clear(pb, bBytes);
            var rowsFn = gpu.Cached($"pack:{f.Name}:{kind.Name}:n", (out string e) => Ptx.PackRows(gpu, f, kind, false, out e));
            Gpu.Launch(rowsFn, Grid((long)m * kp / kind.Per, 256), 1, 256, 1, 0, w, pa, Gpu.U(m), Gpu.U(k), Gpu.U(kp), Gpu.U(k));
            var convFn = gpu.Cached($"packconv:{f.Name}:{kind.Name}", (out string e) => Ptx.PackTransposed(gpu, f, kind, out e, conv: true));
            Gpu.Launch(convFn, Grid(n, 64), (uint)(kp / 32), 256, 1, 0, [x, pb, Gpu.U(n), Gpu.U(k), Gpu.U(kp), .. ConvArgsOf(a)]);
            // C as MatMul writes it (in place when the lane is the type's storage), at the padded shape when padded.
            ulong typed = whole ? y : Rent(f.Words((long)M * N) * (f.WordBits / 8));
            ulong lc = PackedProduct(f, kind, laneName, pa, pb, typed, M, N, kp, rented);
            if (lc != typed) Back(f, lane, lc, typed, (long)M * N);
            if (!whole) Layout<TN>(new LayoutArgs { Name = "strided", X = [], P = [1, 1, m, n, 0, 0, N, 1, 0], Output = [], Count = m * n, InputCount = M * N }, typed, 0, y);
            return laneName;
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    // The plain lanes with B gathered from the image by the product kernel itself (Ptx.Gemm with conv); where the lane's
    // storage is not the type's, the image and the weights decoded where they are stored (the stored mode) and y stored as
    // the type where its values are whole bytes and not blocks, else the lane's sums encoded after.
    private string ConvPlain<TN>(ConvArgs a, ulong x, ulong w, ulong y, Lane lane) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        bool direct = !f.IsBlock && f.Natural == lane && f.Bits == lane.Size * 8 && f.IsFloat == lane.IsFloat;
        if (direct) { GemmConv(a, lane, w, x, y); return lane.Name; }
        bool typedC = StoredC(f);
        int micro = lane.Wide ? 4 : 8, tile = micro * 16;
        bool runsA = f.IsBlock && a.Rows % micro == 0;                                    // the weights' rows as runs of their blocks
        var stored = new Ptx.GemmStored(f, f, typedC ? f : null);
        var fn = gpu.Cached($"gemmconv_stored:{f.Name}:{lane.Name}:{micro}:{runsA}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e, conv: true, stored: stored, runsA: runsA));
        ulong lc = typedC ? y : gpu.Rent(a.YValues * lane.Size);
        try
        {
            Gpu.Launch(fn, Grid(a.Cols, tile), Grid(a.OC, tile), Ptx.GemmThreads, 1, 0, [w, x, lc, Gpu.U(a.OC), Gpu.U(a.Cols), Gpu.U(a.Rows), .. ConvArgsOf(a), 0, 0]);
            if (!typedC) Back(f, lane, lc, y, a.YValues);
        }
        finally
        {
            if (!typedC) gpu.Return(lc);
        }

        return lane.Name;
    }

    private void GemmConv(ConvArgs a, Lane lane, ulong lw, ulong lx, ulong lc, bool composed = false)
    {
        int micro = lane.Wide ? 4 : 8, tile = micro * 16;
        var fn = gpu.Cached($"gemm{(composed ? "" : "conv")}:{lane.Name}:{micro}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e, conv: !composed));
        ulong[] head = [lw, lx, lc, Gpu.U(a.OC), Gpu.U(a.Cols), Gpu.U(a.Rows)];
        Gpu.Launch(fn, Grid(a.Cols, tile), Grid(a.OC, tile), Ptx.GemmThreads, 1, 0, composed ? head : [.. head, .. ConvArgsOf(a)]);
    }

    /// <summary>For the gate: the plain lanes' composition through the same kernel as the fused path (im2col in the lane's
    /// storage, then Ptx.Gemm at the same tile): the fused path must equal it bit for bit.</summary>
    internal void ConvPlainComposed<TN>(ConvArgs a, ulong x, ulong w, ulong y, string laneName) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = LaneOf(laneName);
        bool direct = !f.IsBlock && f.Natural == lane && f.Bits == lane.Size * 8 && f.IsFloat == lane.IsFloat;
        var rented = new List<ulong>();
        try
        {
            ulong lx = direct ? x : Lanes(f, lane, x, a.XValues, rented), lw = direct ? w : Lanes(f, lane, w, a.WValues, rented);
            ulong cols = gpu.Rent((long)a.Rows * a.Cols * lane.Size);
            rented.Add(cols);
            var args = new LayoutArgs { Name = "im2col", X = [], P = a.P, Output = [], Count = a.Rows * a.Cols, InputCount = (int)a.XValues };
            if (lane == Lane.F32) Layout<Float32>(args, lx, 0, cols);
            else if (lane == Lane.F64) Layout<Float64>(args, lx, 0, cols);
            else if (lane == Lane.S32) Layout<Int32>(args, lx, 0, cols);
            else if (lane == Lane.U32) Layout<UInt32>(args, lx, 0, cols);
            else if (lane == Lane.S64) Layout<Int64>(args, lx, 0, cols);
            else Layout<UInt64>(args, lx, 0, cols);
            ulong lc = direct ? y : gpu.Rent(a.YValues * lane.Size);
            if (!direct) rented.Add(lc);
            GemmConv(a, lane, lw, cols, lc, composed: true);
            if (!direct) Back(f, lane, lc, y, a.YValues);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    /// <summary>dx and dw from dy (one image of one group) in the backward's lanes (BackwardLanes); returns their names.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        // The type's storage its float lane's own: the composition on the type's tensors.
        if (f.Is("float64")) { ConvBackwardIn<Float64>(a, x, w, dy, dx, dw); return Lane.F64.Name; }
        if (f.Is("float32")) { ConvBackwardIn<Float32>(a, x, w, dy, dx, dw); return Lane.F32.Name; }
        var (lx, lw) = BackwardLanes<TN>(a, x, w, dy);
        ConvBackwardStored<TN>(a, x, w, dy, dx, dw, lx, lw);
        return LanesName(lx, lw);
    }

    // The backward on the type's own tensors, none widened in memory: dcols = wᵀ·dy (an intermediate in dx's lane storage)
    // with w and dy read where they are stored, dx = col2im(dcols) stored as the type; dw = dy · colsᵀ in dw's lane with the
    // columns gathered from the image where it is stored (Ptx.Gemm's conv mode, transposed), stored as the type or as the
    // lane's sums encoded after.
    private void ConvBackwardStored<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw, Lane lx, Lane lane) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        bool typed = StoredC(f);
        ulong dcols = gpu.Rent(Math.Max(16, (long)a.Rows * a.Cols * lx.Size)), ldw = typed ? dw : gpu.Rent(Math.Max(16, a.WValues * lane.Size));
        try
        {
            ProductStored(new ProductView(a.Rows, a.Cols, a.OC, true, false), lx, w, f, 0, dy, f, 0, dcols, null);
            Layout<TN>(new LayoutArgs { Name = "col2im", X = [], P = a.P, Output = [], Count = (int)a.XValues, InputCount = a.Rows * a.Cols }, dcols, 0, dx, 0, Ptx.LaneFormat(lx));
            var view = new ProductView(a.OC, a.Rows, a.Cols, false, true);
            int micro = !lane.Wide && a.OC >= 128 && a.Rows >= 128 ? 8 : 4, tile = micro * 16;
            bool runsA = f.IsBlock && a.Cols % micro == 0;                                // dy's rows as runs of their blocks
            uint mb = Grid(a.OC, tile);
            if (ProductSplit(view, lane, tile, ldw, typed ? f : null, out int slices, out int sliceBlocks))
            {
                // Split-k (the columns are long, dw small): the slices' partial dw joined in order, then stored as the type.
                var st = new Ptx.GemmStored(f, f, null);
                var fs = gpu.Cached($"gemmconvt_stored:{f.Name}:{lane.Name}:{micro}:{runsA}:k", (out string e) => Ptx.Gemm(gpu, lane, micro, out e, conv: true, stored: st, transB: true, runsA: runsA, split: true));
                ulong parts = gpu.Rent((long)slices * a.WValues * lane.Size), joined = typed ? gpu.Rent(a.WValues * lane.Size) : ldw;
                try
                {
                    Gpu.Launch(fs, Grid(a.Rows, tile), mb * (uint)slices, Ptx.GemmThreads, 1, 0, [dy, x, parts, Gpu.U(a.OC), Gpu.U(a.Rows), Gpu.U(a.Cols), .. ConvArgsOf(a), 0, 0, mb, Gpu.U(sliceBlocks)]);
                    var join = gpu.Cached($"sum_slices:{lane.Name}", (out string e) => Ptx.SumSlices(gpu, lane, out e));
                    Gpu.Launch(join, Grid(a.WValues / 4, 256), 1, 256, 1, 0, parts, joined, Gpu.U((int)a.WValues), Gpu.U(slices));
                    Back(f, lane, joined, dw, a.WValues);
                }
                finally
                {
                    gpu.Return(parts);
                    if (typed) gpu.Return(joined);
                }
            }
            else
            {
                var stored = new Ptx.GemmStored(f, f, typed ? f : null);
                var fn = gpu.Cached($"gemmconvt_stored:{f.Name}:{lane.Name}:{micro}:{runsA}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e, conv: true, stored: stored, transB: true, runsA: runsA));
                Gpu.Launch(fn, Grid(a.Rows, tile), mb, Ptx.GemmThreads, 1, 0, [dy, x, ldw, Gpu.U(a.OC), Gpu.U(a.Rows), Gpu.U(a.Cols), .. ConvArgsOf(a), 0, 0]);
                if (!typed) Back(f, lane, ldw, dw, a.WValues);
            }
        }
        finally
        {
            gpu.Return(dcols);
            if (!typed) gpu.Return(ldw);
        }
    }

    private void ConvBackwardIn<TS>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TS : struct, IFloatAdad<TS>
    {
        var f = Fmt.Of<TS>();
        long colBytes = f.Words((long)a.Rows * a.Cols) * (f.WordBits / 8);
        ulong cols = gpu.Rent(colBytes), dcols = gpu.Rent(colBytes);
        try
        {
            Layout<TS>(new LayoutArgs { Name = "im2col", X = [], P = a.P, Output = [], Count = a.Rows * a.Cols, InputCount = (int)a.XValues }, x, 0, cols);
            MatMulView<TS>(new ProductView(a.Rows, a.Cols, a.OC, true, false), w, dy, dcols);
            Layout<TS>(new LayoutArgs { Name = "col2im", X = [], P = a.P, Output = [], Count = (int)a.XValues, InputCount = a.Rows * a.Cols }, dcols, 0, dx);
            MatMulView<TS>(new ProductView(a.OC, a.Rows, a.Cols, false, true), dy, cols, dw);
        }
        finally
        {
            gpu.Return(cols); gpu.Return(dcols);
        }
    }
}
