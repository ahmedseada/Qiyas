// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The convolution on Vulkan (Conv.cs), CUDA's design on the plain lanes: the forward fused (implicit GEMM: the product
// kernel with its B tile gathered from the image, Kernels.Products.cs MatMulConv; no columns in memory) or the
// composition (im2col, then the product) where this device measures that faster, once per shape and type; both the same
// tile loop over the same values, so the same result. The lane is the product's by the bounds of the weights and the
// image's values with 0 (the padding). The backward on the type's own tensors, every type: dcols = wᵀ·dy (an intermediate:
// lane storage) with w and dy read where they are stored, dx = col2im(dcols) stored as the type once; dw = dy · colsᵀ with
// the columns gathered from the image where it is stored, the lane's values (the type's own storage where the lane's is,
// else stored as the type once after; a block format's by its encoder). dx's sums and dw's (and db's) each in the lane
// SumLane gives for their bounds: a float type's float lane, an integer type's exact one (BackwardLanes). A block
// format's forward is fused: its composition would encode the columns as blocks (each window's values quantized again).
// Images, groups or any term of the product's epilogue take the general path (Conv.cs), CUDA's: the columns of every
// image's channels into lane storage, a product an image and group into the lane's sums, then the epilogue's store over
// the output (EpilogueStore), y stored once; a float type's float lane, any other type's the exact one for the bounds of
// every value the store makes (ConvLane); its backward: dcols = w_gᵀ·dy_g, dx = col2im(dcols); dy and
// the columns side by side over the images in the lane, dw_g = dy_g·cols_gᵀ, db = dy's rows times ones, stored once.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The gate's override: false the composition, true the fused path, null as this device measures.</summary>
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
        e.ValidateType(f.Name, f.IsFloat);
        if (a.YValues == 0) return f.IsFloat ? RowLane(f).Name : f.Natural.Name;
        var lane = f.IsFloat ? RowLane(f) : ConvLane<TN>(a, e, terms, x, w, y, t);
        a.ColumnsFit();
        var lf = VkKernels.LaneFormat(lane);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong cols = Rent(a.ColValues * lane.Size), sums = Rent(a.YValues * lane.Size);
            LaneLayout(lane, new LayoutArgs { Name = "im2col", X = [], P = a.ImagesP, Output = [], Count = (int)a.ColValues, InputCount = (int)a.XValues }, x, cols, f);
            for (int n = 0; n < a.N; n++)
                for (int g = 0; g < a.Groups; g++)
                    Product(new ProductView(a.OCg, a.Cols, a.Rows, false, false), lane, w, f, (long)g * a.OCg * a.Rows, cols, lf, (long)(n * a.Groups + g) * a.Rows * a.Cols,
                        sums + (ulong)(((long)n * a.OC + g * a.OCg) * a.Cols * lane.Size));
            EpilogueStore(f, lane, a.Output, e, terms, t, sums, y, a.Cols, a.OC, Rent);
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    // A type that is not float: the lane its sums take by the bounds of every value the store makes (ConvArgs.IntegerSums);
    // an unsigned type whose values may go negative F64's.
    private Lane ConvLane<TN>(ConvArgs a, ProductEpilogue e, StoreTerms terms, ulong x, ulong w, ulong y, ProductTerms<ulong> t) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var (bounds, negative) = a.IntegerSums(e, terms, f.Name, f.Signed, (p, n) => Range<TN>(p, (int)n), x, w, y, t);
        return negative && !f.Signed ? Lane.F64 : SumLane(f, bounds);
    }

    // A layout into lane storage (the lane's own type), its input read as inFmt where it is stored (null: lane storage too).
    private void LaneLayout(Lane lane, LayoutArgs k, ulong x, ulong output, Fmt? inFmt)
    {
        if (lane == Lane.F32) Layout<Float32>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.F64) Layout<Float64>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.S32) Layout<Adad.Abstraction.Int32>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.U32) Layout<Adad.Abstraction.UInt32>(k, x, 0, output, 0, inFmt);
        else if (lane == Lane.S64) Layout<Adad.Abstraction.Int64>(k, x, 0, output, 0, inFmt);
        else Layout<Adad.Abstraction.UInt64>(k, x, 0, output, 0, inFmt);
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
        var (wLo, wHi) = Range<TN>(w, (int)a.WValues);
        var (xLo, xHi) = Range<TN>(x, (int)a.XValues);
        var (dLo, dHi) = Range<TN>(dy, (int)a.YValues);
        return (SumLane(f, new Bounds(wLo, wHi, dLo, dHi, (long)a.OCg * a.Kh * a.Kw)),
                SumLane(f, new Bounds(dLo, dHi, Math.Min(xLo, 0), Math.Max(xHi, 1), (long)a.N * a.Cols)));
    }

    // The lane names a backward returns: dx's, and dw's after it where it differs.
    private static string LanesName(Lane dx, Lane dw) => dx == dw ? dx.Name : $"{dx.Name}; dw in {dw.Name}";

    // Whether the lane's values are the type's own storage (a product writes them in place).
    private static bool Direct(Fmt f, Lane lane) => !f.IsBlock && f.Natural.Name == lane.Name && f.Bits == lane.Size * 8;

    /// <summary>The general backward (dx, dw, db: see the file's note); returns the lanes' names.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw, ulong db) where TN : struct, IAdad<TN>
    {
        a.Validate();
        if (a.Single && db == 0) return ConvBackward<TN>(a, x, w, dy, dx, dw);
        a.ColumnsFit();
        var f = Fmt.Of<TN>();
        var (lx, lw) = BackwardLanes<TN>(a, x, w, dy);
        Fmt fx = VkKernels.LaneFormat(lx), fw = VkKernels.LaneFormat(lw);
        int G = a.Groups, R = a.Rows, P = a.Cols, NP = a.N * a.Cols;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong cols = Rent(a.ColValues * lw.Size), dcols = Rent(a.ColValues * lx.Size), dyp = Rent((long)a.OC * NP * lw.Size), colsp = Rent(a.ColValues * lw.Size);
            ulong ldw = Rent(a.WValues * lw.Size);
            LaneLayout(lw, new LayoutArgs { Name = "im2col", X = [], P = a.ImagesP, Output = [], Count = (int)a.ColValues, InputCount = (int)a.XValues }, x, cols, f);
            for (int n = 0; n < a.N; n++)
                for (int g = 0; g < G; g++)
                    Product(new ProductView(R, P, a.OCg, true, false), lx, w, f, (long)g * a.OCg * R, dy, f, ((long)n * a.OC + g * a.OCg) * P,
                        dcols + (ulong)((long)(n * G + g) * R * P * lx.Size));                                                                    // w_gᵀ · dy_g
            Layout<TN>(new LayoutArgs { Name = "col2im", X = [], P = a.ImagesP, Output = [], Count = (int)a.XValues, InputCount = (int)a.ColValues }, dcols, 0, dx, 0, fx);
            LaneLayout(lw, new LayoutArgs { Name = "strided", X = [], P = [1, a.OC, a.N, P, 0, P, a.OC * P, 1, 0], Output = [], Count = a.OC * NP, InputCount = (int)a.YValues }, dy, dyp, f);
            LaneLayout(lw, new LayoutArgs { Name = "strided", X = [], P = [1, G * R, a.N, P, 0, P, G * R * P, 1, 0], Output = [], Count = (int)a.ColValues, InputCount = (int)a.ColValues }, cols, colsp, null);
            for (int g = 0; g < G; g++)
                Product(new ProductView(a.OCg, R, NP, false, true), lw, dyp, fw, (long)g * a.OCg * NP, colsp, fw, (long)g * R * NP, ldw + (ulong)((long)g * a.OCg * R * lw.Size));   // dy_g · cols_gᵀ
            Back(f, lw, ldw, dw, a.WValues);
            if (db != 0)
            {
                // Each channel's Σ dy: its row of dy (every image's positions) times ones.
                ulong ones = Rent((long)NP * lw.Size), ldb = Rent((long)a.OC * lw.Size);
                vk.Upload<byte>(LaneOnes(lw, NP), ones);
                Product(new ProductView(a.OC, 1, NP, false, false), lw, dyp, fw, 0, ones, fw, 0, ldb);
                Back(f, lw, ldb, db, a.OC);
            }
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return LanesName(lx, lw);
    }

    /// <summary>y = conv(x, w) of one image of one group; returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (FusedConv == false) return ConvComposed<TN>(a, x, w, y);
        // The columns' values are the image's and 0 (padding): the image's range, with 0, bounds them.
        var (wLo, wHi) = Range<TN>(w, (int)a.WValues);
        var (xLo, xHi) = Range<TN>(x, (int)a.XValues);
        var bounds = new Bounds(wLo, wHi, Math.Min(xLo, 0), Math.Max(xHi, 0), a.Rows);
        var lane = SumLane(f, bounds);
        // A product on the matrix units, or float64's that may take the sliced product: the composition (the columns as the
        // type, then the product's measured route), whichever mode (the fused path and the composition then the same sums).
        if (!f.IsBlock && a.ColValues <= int.MaxValue && (UnitsKind(f, bounds) is not null || (f.Is("float64") && SlicedProven))) return ConvComposed<TN>(a, x, w, y);
        // The composition's columns past 2^31 values: fused.
        if (FusedConv == true || f.IsBlock || a.ColValues > int.MaxValue) return ConvFused(f, a, x, w, y, lane);
        string key = $"{f.Name}: {a.Shape}";
        if (!_convFused.TryGetValue(key, out bool fused))
        {
            ulong scratch = vk.Rent(f.Words(a.YValues) * (f.WordBits / 8) + 64);
            try
            {
                var (tf, tc) = Measure.Faster(() => ConvFused(f, a, x, w, scratch, lane), () => ConvComposed<TN>(a, x, w, scratch), vk.Synchronize);
                fused = _convFused[key] = tf <= tc;
            }
            finally { vk.Return(scratch); }
        }

        return fused ? ConvFused(f, a, x, w, y, lane) : ConvComposed<TN>(a, x, w, y);
    }

    /// <summary>y = conv(x, w) by the composition: im2col (the columns as the type), then the product.</summary>
    private string ConvComposed<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        a.ColumnsFit();
        var f = Fmt.Of<TN>();
        ulong cols = vk.Rent(Math.Max(1, f.Words((long)a.Rows * a.Cols)) * (f.WordBits / 8));
        try
        {
            Layout<TN>(new LayoutArgs { Name = "im2col", X = [], P = a.P, Output = [], Count = a.Rows * a.Cols, InputCount = (int)a.XValues }, x, 0, cols, 0);
            return MatMulView<TN>(new ProductView(a.OC, a.Cols, a.Rows, false, false), w, cols, y);
        }
        finally
        {
            vk.Return(cols);
        }
    }

    // The fused forward: y's lane values from the product with the columns gathered from the image, stored as MatMulView
    // stores C (in place where the lane is the type's storage, else lane storage stored as the type once, the bits of y's
    // last word past its values kept): the composition's y, bit for bit.
    private string ConvFused(Fmt f, ConvArgs a, ulong x, ulong w, ulong y, Lane lane)
    {
        if (a.YValues == 0) return lane.Name;
        bool direct = f.Natural.Name == lane.Name && f.Bits == lane.Size * 8;
        ulong lc = direct ? y : vk.Rent(a.YValues * lane.Size);
        try
        {
            ConvProduct(f, f, lane, w, x, lc, a.OC, a.Cols, a.Rows, a, transB: false);
            if (!direct) BackKeep(f, lane, lc, y, a.YValues, a.YValues, a.YValues);
        }
        finally
        {
            if (!direct) vk.Return(lc);
        }

        return lane.Name;
    }

    // C (lane values at c, m × n) = A (m × k, stored as fa) times the image's columns (k × n), or for transB their
    // transpose (the columns n × k): Kernels.Products.cs MatMulConv.
    private void ConvProduct(Fmt fa, Fmt fb, Lane lane, ulong a, ulong x, ulong c, int m, int n, int k, ConvArgs conv, bool transB)
    {
        if (m == 0 || n == 0) return;
        var t = Tiling(lane, m, n, k, 1, tt => ConvLaunch(tt, fa, fb, lane, a, x, c, m, n, k, conv, transB));
        ConvLaunch(t, fa, fb, lane, a, x, c, m, n, k, conv, transB);
    }

    // ConvProduct at a tiling (the plain product's: Tiling); split, each slice's partial C into lane storage, then added in order.
    private void ConvLaunch(PlainTiling t, Fmt fa, Fmt fb, Lane lane, ulong a, ulong x, ulong c, int m, int n, int k, ConvArgs conv, bool transB)
    {
        var fn = vk.Cached($"matmul_conv:{fa.Name}:{fb.Name}:{lane.Name}:{transB}:{t.Micro}", () => VkKernels.MatMulConv(vk, fa, fb, lane, transB, t.Micro));
        int tile = VkKernels.Tile(t.Micro), tilesM = (m + tile - 1) / tile;
        if (k == 0) t = t with { Slices = 1 };
        int kslice = t.Slices > 1 ? ((k + t.Slices - 1) / t.Slices + VkKernels.TileK - 1) / VkKernels.TileK * VkKernels.TileK : Math.Max(1, k);
        int slices = t.Slices > 1 ? (k + kslice - 1) / kslice : 1;
        long mn = (long)m * n;
        ulong parts = slices > 1 ? vk.Rent(slices * mn * lane.Size) : 0;
        try
        {
            // im2col's parameters as the kernels take them (every pad, the dilations, OH and OW) two a slot; then tilesM, kslice.
            var p = LayoutArgs.KernelP("im2col", conv.P);
            var args = new ulong[6 + (p.Length + 1) / 2 + 2];
            args[0] = a; args[1] = x; args[2] = slices > 1 ? parts : c;
            args[3] = Vulkan.U(m); args[4] = Vulkan.U(n); args[5] = Vulkan.U(k);
            for (int j = 0; 2 * j < p.Length; j++) args[6 + j] = (uint)p[2 * j] | (2 * j + 1 < p.Length ? (ulong)(uint)p[2 * j + 1] << 32 : 0);
            args[^2] = Vulkan.U(tilesM);
            args[^1] = Vulkan.U(kslice);
            vk.Launch(fn, Groups(n, tile), (uint)(tilesM * slices), args);
            if (slices > 1) JoinSlices(lane, parts, c, mn, slices);
        }
        finally { if (parts != 0) vk.Return(parts); }
    }

    /// <summary>dx and dw from dy (one image of one group) in the backward's lanes (BackwardLanes); returns their names.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var (lx, lw) = BackwardLanes<TN>(a, x, w, dy);
        bool direct = Direct(f, lw);
        var rented = new List<ulong>();
        try
        {
            ulong Rent(long bytes) { ulong p = vk.Rent(bytes); rented.Add(p); return p; }
            ulong dcols = Rent(Math.Max(1, (long)a.Rows * a.Cols) * lx.Size);
            ulong ldw = direct ? dw : Rent(Math.Max(1, a.WValues) * lw.Size);
            // dcols = wᵀ·dy (an intermediate) in dx's lane; dx = col2im(dcols), its sums in that lane, stored as the type once.
            Product(new ProductView(a.Rows, a.Cols, a.OC, true, false), lx, w, f, 0, dy, f, 0, dcols);
            Layout<TN>(new LayoutArgs { Name = "col2im", X = [], P = a.P, Output = [], Count = (int)a.XValues, InputCount = a.Rows * a.Cols }, dcols, 0, dx, 0, VkKernels.LaneFormat(lx));
            // dw = dy · colsᵀ in dw's lane, the columns gathered from the image where it is stored.
            ConvProduct(f, f, lw, dy, x, ldw, a.OC, a.Rows, a.Cols, a, transB: true);
            if (!direct) Back(f, lw, ldw, dw, a.WValues);
            return LanesName(lx, lw);
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }
}
