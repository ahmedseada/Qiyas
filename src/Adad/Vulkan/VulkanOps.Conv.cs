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
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The gate's override: false the composition, true the fused path, null as this device measures.</summary>
    internal bool? FusedConv;

    private readonly Dictionary<string, bool> _convFused = [];

    private static Fmt ConvFormat<TN>(string what) where TN : struct, IAdad<TN> => Fmt.Of<TN>();

    /// <summary>y = conv(x, w); returns the product's lane.</summary>
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
