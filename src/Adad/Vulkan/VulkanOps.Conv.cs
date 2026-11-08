// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The convolution on Vulkan (Conv.cs), CUDA's design on the plain lanes: the forward fused (implicit GEMM: the product
// kernel with its B tile gathered from the image, Kernels.Products.cs MatMulConv; no columns in memory) or the
// composition (im2col, then the product) where this device measures that faster, once per shape and type; both the same
// tile loop over the same values, so the same result. The lane is the product's by the bounds of the weights and the
// image's values with 0 (the padding). The backward in the type's float lane, on the type's own tensors: dcols = wᵀ·dy
// (an intermediate: lane storage) with w and dy read where they are stored, dx = col2im(dcols); dw = dy · colsᵀ with the
// columns gathered from the image where it is stored; the gradients lane values (the type's own storage for float32 and
// float64, else stored as the type once after; a block format's by its encoder). A block format's forward is fused: its
// composition would encode the columns as blocks (each window's values quantized again).
// Images, groups, the bottom and right pads, a bias and an activation (the float types) take the general path (Conv.cs),
// CUDA's: the columns of every image's channels into lane storage, a product an image and group into the lane's sums, then
// y = act(sums + bias[f]) stored once (Kernels.ConvStore.cs); its backward: dcols = w_gᵀ·dy_g, dx = col2im(dcols); dy and
// the columns side by side over the images in the lane, dw_g = dy_g·cols_gᵀ, db = dy's rows times ones, stored once.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The gate's override: false the composition, true the fused path, null as this device measures.</summary>
    internal bool? FusedConv;

    private readonly Dictionary<string, bool> _convFused = [];

    private static Fmt ConvFormat<TN>(string what) where TN : struct, IAdad<TN> => Fmt.Of<TN>();

    /// <summary>y = act(conv(x, w) + bias) (ConvArgs.Single without a bias: the one-image path); returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong bias, ulong y) where TN : struct, IAdad<TN>
    {
        a.Validate();
        if (a.Single && bias == 0) return Conv<TN>(a, x, w, y);
        var f = Fmt.Of<TN>();
        if (!f.IsFloat) throw new NotSupportedException($"conv: images, groups, uneven pads, a bias or an activation take a float type, not {f.Name}");
        var lane = RowLane(f);
        if (a.YValues == 0) return lane.Name;
        var lf = LaneFormat(lane);
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
            bool b = bias != 0;
            if (f.IsBlock)
                Encode(f, a.YValues, vk.Cached($"block_conv_store:{f.Name}:{a.Act}:{b}", () => VkKernels.BlockConvStore(vk, f, a.Act, b)), y,
                    sums, bias, Vulkan.U(a.Cols), Vulkan.U(a.OC), Vulkan.F((float)a.ActA));
            else
            {
                var k = vk.Cached($"conv_store:{f.Name}:{lane.Name}:{a.Act}:{b}", () => VkKernels.ConvStore(vk, f, lane, a.Act, b));
                vk.Launch(k, Groups(f.Words(a.YValues), VkKernels.Threads), 1, sums, bias, y, Vulkan.U((int)a.YValues), Vulkan.U(a.Cols), Vulkan.U(a.OC), Slot(lane, a.ActA));
            }
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    // The float lane's own format.
    private static Fmt LaneFormat(Lane lane) => lane.Wide ? new Fmt("float64", 64, true, true) : new Fmt("float32", 32, true, true);

    // A layout into lane storage (the lane's own type), its input read as inFmt where it is stored (null: lane storage too).
    private void LaneLayout(Lane lane, LayoutArgs k, ulong x, ulong output, Fmt? inFmt)
    {
        if (lane.Wide) Layout<Float64>(k, x, 0, output, 0, inFmt);
        else Layout<Float32>(k, x, 0, output, 0, inFmt);
    }

    /// <summary>The general backward (dx, dw, db: see the file's note); returns the lane's name.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw, ulong db) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        if (a.Single && db == 0) return ConvBackward<TN>(a, x, w, dy, dx, dw);
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        var lf = LaneFormat(lane);
        int G = a.Groups, R = a.Rows, P = a.Cols, NP = a.N * a.Cols;
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong cols = Rent(a.ColValues * lane.Size), dcols = Rent(a.ColValues * lane.Size), dyp = Rent((long)a.OC * NP * lane.Size), colsp = Rent(a.ColValues * lane.Size);
            ulong ldw = Rent(a.WValues * lane.Size);
            LaneLayout(lane, new LayoutArgs { Name = "im2col", X = [], P = a.ImagesP, Output = [], Count = (int)a.ColValues, InputCount = (int)a.XValues }, x, cols, f);
            for (int n = 0; n < a.N; n++)
                for (int g = 0; g < G; g++)
                    Product(new ProductView(R, P, a.OCg, true, false), lane, w, f, (long)g * a.OCg * R, dy, f, ((long)n * a.OC + g * a.OCg) * P,
                        dcols + (ulong)((long)(n * G + g) * R * P * lane.Size));                                                                  // w_gᵀ · dy_g
            Layout<TN>(new LayoutArgs { Name = "col2im", X = [], P = a.ImagesP, Output = [], Count = (int)a.XValues, InputCount = (int)a.ColValues }, dcols, 0, dx, 0, lf);
            LaneLayout(lane, new LayoutArgs { Name = "strided", X = [], P = [1, a.OC, a.N, P, 0, P, a.OC * P, 1, 0], Output = [], Count = a.OC * NP, InputCount = (int)a.YValues }, dy, dyp, f);
            LaneLayout(lane, new LayoutArgs { Name = "strided", X = [], P = [1, G * R, a.N, P, 0, P, G * R * P, 1, 0], Output = [], Count = (int)a.ColValues, InputCount = (int)a.ColValues }, cols, colsp, null);
            for (int g = 0; g < G; g++)
                Product(new ProductView(a.OCg, R, NP, false, true), lane, dyp, lf, (long)g * a.OCg * NP, colsp, lf, (long)g * R * NP, ldw + (ulong)((long)g * a.OCg * R * lane.Size));   // dy_g · cols_gᵀ
            Back(f, lane, ldw, dw, a.WValues);
            if (db != 0)
            {
                // Each channel's Σ dy: its row of dy (every image's positions) times ones.
                ulong ones = Rent((long)NP * lane.Size), ldb = Rent((long)a.OC * lane.Size);
                var one = new double[NP];
                Array.Fill(one, 1.0);
                vk.Upload<byte>(lane.Wide ? Stored.StoreValues<Float64>(one) : Stored.StoreValues<Float32>(one), ones);
                Product(new ProductView(a.OC, 1, NP, false, false), lane, dyp, lf, 0, ones, lf, 0, ldb);
                Back(f, lane, ldb, db, a.OC);
            }
        }
        finally { foreach (var p in rented) vk.Return(p); }

        return lane.Name;
    }

    /// <summary>y = conv(x, w) of one image of one group; returns the product's lane.</summary>
    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        var f = ConvFormat<TN>("conv");
        if (FusedConv == false || a.XValues > int.MaxValue || a.WValues > int.MaxValue || a.YValues > int.MaxValue) return ConvComposed<TN>(a, x, w, y);
        // The columns' values are the image's and 0 (padding): the image's range, with 0, bounds them.
        var (wLo, wHi) = Range<TN>(w, (int)a.WValues);
        var (xLo, xHi) = Range<TN>(x, (int)a.XValues);
        var lane = SumLane(f, new Bounds(wLo, wHi, Math.Min(xLo, 0), Math.Max(xHi, 0), a.Rows));
        if (FusedConv == true || f.IsBlock) return ConvFused(f, a, x, w, y, lane);
        string key = $"{f.Name}:{a.C}x{a.H}x{a.W}>{a.OC}:{a.Kh}x{a.Kw}:s{a.Sh}x{a.Sw}:p{a.Ph}x{a.Pw}:d{a.Dh}x{a.Dw}";
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
    public string ConvComposed<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN>
    {
        var f = ConvFormat<TN>("conv (composed)");
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
        var fn = vk.Cached($"matmul_conv:{fa.Name}:{fb.Name}:{lane.Name}:{transB}", () => VkKernels.MatMulConv(vk, fa, fb, lane, transB));
        var args = new ulong[12];
        args[0] = a; args[1] = x; args[2] = c;
        args[3] = Vulkan.U(m); args[4] = Vulkan.U(n); args[5] = Vulkan.U(k);
        var p = conv.P;
        for (int j = 0; j < 6; j++) args[6 + j] = (uint)p[2 * j] | (ulong)(uint)p[2 * j + 1] << 32;
        vk.Launch(fn, Groups(n, VkKernels.TileN), (uint)((m + VkKernels.TileM - 1) / VkKernels.TileM), args);
    }

    /// <summary>dx and dw from dy in the type's float lane; returns the lane's name.</summary>
    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TN : struct, IFloatAdad<TN>
    {
        var f = ConvFormat<TN>("conv backward");
        var lane = RowLane(f);
        bool own = Own(f);
        var rented = new List<ulong>();
        try
        {
            ulong Rent(long bytes) { ulong p = vk.Rent(bytes); rented.Add(p); return p; }
            ulong dcols = Rent(Math.Max(1, (long)a.Rows * a.Cols) * lane.Size);
            ulong ldx = own ? dx : Rent(Math.Max(1, a.XValues) * lane.Size), ldw = own ? dw : Rent(Math.Max(1, a.WValues) * lane.Size);
            // dcols = wᵀ·dy (an intermediate), dx = col2im(dcols) in the lane.
            Product(new ProductView(a.Rows, a.Cols, a.OC, true, false), lane, w, f, 0, dy, f, 0, dcols);
            var col2im = new LayoutArgs { Name = "col2im", X = [], P = a.P, Output = [], Count = (int)a.XValues, InputCount = a.Rows * a.Cols };
            if (lane.Wide) Layout<Float64>(col2im, dcols, 0, ldx, 0); else Layout<Float32>(col2im, dcols, 0, ldx, 0);
            // dw = dy · colsᵀ, the columns gathered from the image where it is stored.
            ConvProduct(f, f, lane, dy, x, ldw, a.OC, a.Rows, a.Cols, a, transB: true);
            if (!own)
            {
                Back(f, lane, ldx, dx, a.XValues);
                Back(f, lane, ldw, dw, a.WValues);
            }

            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }
}
