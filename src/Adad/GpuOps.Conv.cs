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
// The backward keeps the composition.
namespace Adad;

public sealed partial class GpuOps
{
    /// <summary>The gate's override: false the composition, true the fused path everywhere, null fused (the plain lanes as measured).</summary>
    internal bool? FusedConv;

    private readonly Dictionary<string, bool> _convFused = [];

    /// <summary>y = conv(x, w); returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        if (FusedConv == false || a.XValues > int.MaxValue || a.WValues > int.MaxValue) return ConvComposed<TN>(a, x, w, y);
        var f = Fmt.Of<TN>();
        // The columns' values are the image's and 0 (padding): the image's range, with 0, bounds them.
        var ((wLo, wHi), (xLo, xHi)) = RangePair<TN>(w, (int)a.WValues, x, (int)a.XValues);
        var bounds = new Bounds(wLo, wHi, Math.Min(xLo, 0), Math.Max(xHi, 0), a.Rows);
        string laneName = ProductLane<TN>(bounds);
        if (!f.IsBlock && (laneName.EndsWith("mma", StringComparison.Ordinal) || laneName.EndsWith("dp", StringComparison.Ordinal)))
            return ConvPacked<TN>(a, x, w, y, laneName);
        var lane = LaneOf(laneName);
        // A block format's composition would encode the columns as blocks (each window's values quantized again): fused.
        if (FusedConv == true || f.IsBlock) return ConvPlain<TN>(a, x, w, y, lane);
        string key = $"{f.Name}:{a.C}x{a.H}x{a.W}>{a.OC}:{a.Kh}x{a.Kw}:s{a.Sh}x{a.Sw}:p{a.Ph}x{a.Pw}:d{a.Dh}x{a.Dw}";
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
    public string ConvComposed<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
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

    private static ulong[] ConvArgsOf(ConvArgs a) => [Gpu.U(a.H), Gpu.U(a.W), Gpu.U(a.Kh), Gpu.U(a.Kw), Gpu.U(a.Sh), Gpu.U(a.Sw), Gpu.U(a.Ph), Gpu.U(a.Pw), Gpu.U(a.Dh), Gpu.U(a.Dw), Gpu.U(a.Out.OW)];

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
        bool typedC = !f.IsBlock && f.Bits >= 8;
        int micro = lane.Wide ? 4 : 8, tile = micro * 16;
        bool runsA = f.IsBlock && a.Rows % micro == 0;                                    // the weights' rows as runs of their blocks
        var fn = gpu.Cached($"gemmconv_stored:{f.Name}:{lane.Name}:{micro}:{runsA}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e, conv: true, stored: f, runsA: runsA));
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

    /// <summary>dx and dw from dy, in the type's float lane storage; returns the lane's name.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        bool own = f.Is("float32") || f.Is("float64");
        var rented = new List<ulong>();
        try
        {
            ulong lx = ToLane<TN>(x, a.XValues, rented), lw = ToLane<TN>(w, a.WValues, rented), ldy = ToLane<TN>(dy, a.YValues, rented);
            ulong ldx = own ? dx : gpu.Rent(a.XValues * lane.Size), ldw = own ? dw : gpu.Rent(a.WValues * lane.Size);
            if (!own) { rented.Add(ldx); rented.Add(ldw); }
            if (lane.Wide) ConvBackwardIn<Float64>(a, lx, lw, ldy, ldx, ldw); else ConvBackwardIn<Float32>(a, lx, lw, ldy, ldx, ldw);
            FromLane<TN>(ldx, dx, a.XValues);
            FromLane<TN>(ldw, dw, a.WValues);
            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
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
