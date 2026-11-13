// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The INumberBase<T> version's product cells on Vulkan (GenericSpv.Products.cs), run as NumberSubject runs its CUDA
// cells: the same buffers, the same steps and the same measured choice of the product's tile. The convolution as the
// generic code writes it: im2col, then the product; the backward wᵀ·dy, col2im for dx; im2col, its transpose, then
// dy·colsᵀ for dw (the columns and the backward's transposes the run's: Held). The box losses a thread a pair; ctc a
// workgroup a sequence with its α, β and log-sum-exp scratch (Held). Their gate cases (ProductCases): each cell on a
// small case on every type its dispatcher takes, against the CPU's version (Agree).
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Vulkan;

namespace Qiyas.Bench;

internal static partial class NumberVulkan
{
    private static uint Groups(long items, int threads) => (uint)((items + threads - 1) / threads);

    /// <summary>Every kernel of the product cells, each cell on each type its dispatcher takes (the gate builds them all).</summary>
    private static IEnumerable<(string, string, Vulkan.Maker)> Products() =>
        ProductEvery<double>("float64").Concat(ProductEvery<float>("float32")).Concat(ProductEvery<Half>("float16"))
            .Concat(ProductEvery<long>("int64")).Concat(ProductEvery<int>("int32")).Concat(ProductEvery<short>("int16")).Concat(ProductEvery<sbyte>("int8"))
            .Concat(ProductEvery<ulong>("uint64")).Concat(ProductEvery<uint>("uint32")).Concat(ProductEvery<ushort>("uint16")).Concat(ProductEvery<byte>("uint8"))
            .Concat(ProductFloats<double>("float64")).Concat(ProductFloats<float>("float32")).Concat(ProductFloats<Half>("float16"));

    // The cells every type takes: matmul (both tiles), conv (im2col, col2im, the transpose; its products matmul's).
    private static IEnumerable<(string, string, Vulkan.Maker)> ProductEvery<T>(string type) where T : unmanaged, INumberBase<T>
    {
        foreach (int micro in new[] { 8, 4 }) yield return ("matmul", type, () => GenericSpv.MatMul<T>(Vk, micro));
        yield return ("conv", type, () => GenericSpv.Im2col<T>(Vk));
        yield return ("conv", type, () => GenericSpv.Col2im<T>(Vk));
        yield return ("conv", type, () => GenericSpv.Transpose<T>(Vk));
    }

    // The float types' cells: the box losses (every overlap, forward and backward) and ctc.
    private static IEnumerable<(string, string, Vulkan.Maker)> ProductFloats<T>(string type) where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (var overlap in Enum.GetValues<BoxOverlap>())
            foreach (bool backward in new[] { false, true }) yield return ("box_loss", type, () => GenericSpv.BoxLoss<T>(Vk, overlap, backward));
        if (CtcRuns<T>()) yield return ("ctc", type, () => GenericSpv.Ctc<T>(Vk));
    }

    // ctc's gradient adds atomically: a float64 cell where the device has the f64 atomic add (the others by a 32-bit
    // compare-and-swap where it has none).
    private static bool CtcRuns<T>() => typeof(T) != typeof(double) || Vk.AtomicAdd64;

    // ---------------------------------------------------------------- matmul

    public static IBenchRun? MatMul<T>(byte[] x, byte[] y, byte[] z, int m, int n, int k) where T : unmanaged, INumberBase<T>
    {
        var bufs = new Buffers();
        ulong da = bufs.In(x), db = bufs.In(y), dc = bufs.Rent(z.Length);
        var product = Product<T>(da, db, dc, m, n, k);
        return new BenchRun(() => { product.Launch(da, db, dc, m, n, k); Vk.Synchronize(); },
            () => Vk.Download<byte>(dc, z), bufs.Free, () => product.Entry, () => new(0, 0));
    }

    // The generic product's kernel for an m × n × k shape: its 64 and 128 tiles both measured once on these operands, the
    // faster kept (NumberSubject's CUDA product's rule).
    private sealed record GenericProduct((ulong Pipeline, SpvKernel Kernel) Function, int Tile)
    {
        public string Entry => Function.Kernel.Name;

        public void Launch(ulong a, ulong b, ulong c, int m, int n, int k) =>
            Vk.Launch(Function, Groups(n, Tile), Groups(m, Tile), a, b, c, Vulkan.U(m), Vulkan.U(n), Vulkan.U(k));
    }

    private static GenericProduct Product<T>(ulong da, ulong db, ulong dc, int m, int n, int k) where T : unmanaged, INumberBase<T>
    {
        GenericProduct? best = null;
        double bestMs = double.MaxValue;
        foreach (int micro in new[] { 8, 4 })
        {
            var p = new GenericProduct(Kernel(() => GenericSpv.MatMul<T>(Vk, micro)), 16 * micro);
            p.Launch(da, db, dc, m, n, k);
            Vk.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) p.Launch(da, db, dc, m, n, k);
            Vk.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            if (ms < bestMs) (bestMs, best) = (ms, p);
        }

        return best!;
    }

    // ---------------------------------------------------------------- conv

    public static IBenchRun? Conv<T>(ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) where T : unmanaged, INumberBase<T>
    {
        int rows = a.Rows, cols = a.Cols, oc = a.OC, size = Marshal.SizeOf<T>();
        var (oh, ow) = a.Out;
        ulong[] p = [Vulkan.U(a.H), Vulkan.U(a.W), Vulkan.U(a.Kh), Vulkan.U(a.Kw), Vulkan.U(a.Sh), Vulkan.U(a.Sw), Vulkan.U(a.PadTop), Vulkan.U(a.PadLeft),
            Vulkan.U(a.Dh), Vulkan.U(a.Dw), Vulkan.U(oh), Vulkan.U(ow)];
        var im2col = Kernel(() => GenericSpv.Im2col<T>(Vk));
        var bufs = new Buffers();
        ulong dX = bufs.In(x), dW = bufs.In(w), dCols = bufs.Rent((long)rows * cols * size);
        void Columns() => Vk.Launch(im2col, Groups((long)rows * cols, 256), 1, [dX, dCols, Vulkan.U(rows * cols), .. p]);
        string ie = im2col.Kernel.Name;
        if (dx is null)
        {
            ulong dY = bufs.Rent(y.Length);
            Columns();
            var product = Product<T>(dW, dCols, dY, oc, cols, rows);
            return new BenchRun(() => { Columns(); product.Launch(dW, dCols, dY, oc, cols, rows); Vk.Synchronize(); },
                () => Vk.Download<byte>(dY, y), bufs.Free, () => $"{ie} + {product.Entry}", () => new(0, (long)rows * cols * size));
        }

        var col2im = Kernel(() => GenericSpv.Col2im<T>(Vk));
        var transpose = Kernel(() => GenericSpv.Transpose<T>(Vk));
        ulong dDy = bufs.In(y), dWt = bufs.Rent((long)rows * oc * size), dDc = bufs.Rent((long)rows * cols * size), dCt = bufs.Rent((long)cols * rows * size);
        ulong dDx = bufs.Rent(dx.Length), dDw = bufs.Rent(dw!.Length);
        void Transpose(ulong from, ulong to, int r, int c) => Vk.Launch(transpose, Groups((long)r * c, 256), 1, from, to, Vulkan.U(r), Vulkan.U(c));
        Transpose(dW, dWt, oc, rows);
        Columns();
        Transpose(dCols, dCt, rows, cols);
        var dcols = Product<T>(dWt, dDy, dDc, rows, cols, oc);
        var dweights = Product<T>(dDy, dCt, dDw, oc, rows, cols);
        return new BenchRun(() =>
        {
            Transpose(dW, dWt, oc, rows);
            dcols.Launch(dWt, dDy, dDc, rows, cols, oc);                                  // dcols = wᵀ·dy
            Vk.Launch(col2im, Groups(a.XValues, 256), 1, [dDc, dDx, Vulkan.U((int)a.XValues), .. p]);
            Columns();
            Transpose(dCols, dCt, rows, cols);
            dweights.Launch(dDy, dCt, dDw, oc, rows, cols);                               // dw = dy·colsᵀ
            Vk.Synchronize();
        }, () => { Vk.Download<byte>(dDx, dx); Vk.Download<byte>(dDw, dw); }, bufs.Free, () => $"{ie}, {transpose.Kernel.Name}, {col2im.Kernel.Name} + {dcols.Entry}",
            () => new(0, ((long)rows * oc + 3L * rows * cols) * size));
    }

    // ---------------------------------------------------------------- box losses

    public static IBenchRun? BoxLoss<T>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var bufs = new Buffers();
        ulong dp = bufs.In(predicted), dt = bufs.In(target), dl = dloss is null ? 0 : bufs.In(dloss), dy = bufs.Rent(output.Length);
        bool backward = dloss is not null;
        var f = Kernel(() => GenericSpv.BoxLoss<T>(Vk, overlap, backward));
        return new BenchRun(() =>
        {
            Vk.Launch(f, Groups(count, 256), 1, dp, dt, dl, dy, Vulkan.U(count));
            Vk.Synchronize();
        }, () => Vk.Download<byte>(dy, output), bufs.Free, () => f.Kernel.Name, () => new(0, 0));
    }

    // ---------------------------------------------------------------- ctc

    // A workgroup a sequence (GenericSpv.Ctc), the scratch α (β) of T × the most states and the log-sum-exp of T frames a sequence.
    public static IBenchRun? Ctc<T>(CtcArgs a, int[] offsets, byte[] logits, byte[] loss, byte[]? grad) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (!CtcRuns<T>()) return null;
        int states = 2 * a.LabelLengths.Max() + 1;
        long lseBytes = (long)a.B * a.T * Marshal.SizeOf<T>(), alphaBytes = lseBytes * states;
        var bufs = new Buffers();
        ulong dx = bufs.In(logits), dloss = bufs.Rent(loss.Length), dgrad = grad is null ? 0 : bufs.Rent(grad.Length);
        ulong dlse = bufs.Rent(lseBytes), dalpha = bufs.Rent(alphaBytes), dbeta = grad is null ? 0 : bufs.Rent(alphaBytes);
        ulong dframes = bufs.Rent(a.B * 4L), dlengths = bufs.Rent(a.B * 4L), doffsets = bufs.Rent(a.B * 4L), dlabels = bufs.Rent(Math.Max(1, a.Labels.Length) * 4L);
        Vk.Upload<int>(a.InputLengths, dframes);
        Vk.Upload<int>(a.LabelLengths, dlengths);
        Vk.Upload<int>(offsets, doffsets);
        Vk.Upload<int>(a.Labels, dlabels);
        var f = Kernel(() => GenericSpv.Ctc<T>(Vk));
        return new BenchRun(() =>
        {
            Vk.Launch(f, (uint)a.B, 1, dx, dloss, dgrad, dlse, dalpha, dbeta, dframes, dlengths, doffsets, dlabels,
                Vulkan.U(a.T), Vulkan.U(a.B), Vulkan.U(a.C), Vulkan.U(a.Blank), Vulkan.U(states), Vulkan.U(grad is null ? 0 : 1));
            Vk.Synchronize();
        }, () =>
        {
            Vk.Download<byte>(dloss, loss);
            if (grad is not null) Vk.Download<byte>(dgrad, grad);
        }, bufs.Free, () => f.Kernel.Name, () => new(0, lseBytes + alphaBytes * (grad is null ? 1 : 2)));
    }

    // ---------------------------------------------------------------- the gate's cases

    // The product cells' small cases, each against the CPU's version (Agree): matmul at 37 × 29 × 20 on every type; conv
    // and its backward at a padded, strided 3 × 3 (one image of one group: the cell's shape); the box losses over 300
    // pairs, every overlap, forward and backward; ctc and its backward over three sequences (one with no labels; float16's
    // gradient not held, see there).
    private static IEnumerable<(string Cell, string Type, Func<string?> Run)> ProductCases()
    {
        var s = new NumberSubject();
        foreach (string type in Plain)
        {
            const int m = 37, n = 29, k = 20;
            byte[] a = Registry.Make(type, m * k, 11, -1, 1), b = Registry.Make(type, k * n, 12, -1, 1);
            yield return ("matmul", type, () => Agree(d => { var c = Registry.Poisoned(type, m * n); return (s.MatMul(d, type, a, b, c, m, n, k), [(type, m * n, c)]); }));

            var conv = new ConvArgs(3, 9, 10, 4, 3, 3) { Sh = 2, PadTop = 1, PadLeft = 1, PadBottom = 1, PadRight = 1 };
            byte[] x = Registry.Make(type, (int)conv.XValues, 13, -1, 1), w = Registry.Make(type, (int)conv.WValues, 14, -1, 1);
            byte[] dy = Registry.Make(type, (int)conv.YValues, 15, -1, 1);
            yield return ("conv", type, () => Agree(d =>
            {
                var y = Registry.Poisoned(type, conv.YValues);
                return (s.Conv(d, type, conv, x, w, y, null, null), [(type, (int)conv.YValues, y)]);
            }));
            yield return ("conv_bwd", type, () => Agree(d =>
            {
                byte[] gx = Registry.Poisoned(type, conv.XValues), gw = Registry.Poisoned(type, conv.WValues);
                return (s.Conv(d, type, conv, x, w, dy, gx, gw), [(type, (int)conv.XValues, gx), (type, (int)conv.WValues, gw)]);
            }));
        }

        foreach (string type in new[] { "float64", "float32", "float16" })
        {
            const int pairs = 300;
            byte[] p = Registry.Make(type, 4 * pairs, 16, 0, 4), t = Registry.Make(type, 4 * pairs, 17, 0, 4), dl = Registry.Make(type, pairs, 18, -1, 1);
            foreach (var overlap in Enum.GetValues<BoxOverlap>())
            {
                yield return ("box_loss:" + Boxes.Name(overlap), type, () => Agree(d =>
                {
                    var y = Registry.Poisoned(type, pairs);
                    return (s.BoxLoss(d, type, overlap, p, t, null, y, pairs), [(type, pairs, y)]);
                }));
                yield return ("box_loss_bwd:" + Boxes.Name(overlap), type, () => Agree(d =>
                {
                    var y = Registry.Poisoned(type, 4 * pairs);
                    return (s.BoxLoss(d, type, overlap, p, t, dl, y, pairs), [(type, 4 * pairs, y)]);
                }));
            }

            if (type == "float64" && !Vk.AtomicAdd64) continue;
            var ctc = new CtcArgs(12, 3, 7, 0, [12, 9, 11], [4, 2, 0], [1, 3, 3, 5, 2, 6]);
            int values = ctc.T * ctc.B * ctc.C;
            byte[] logits = Registry.Make(type, values, 19, -2, 2);
            yield return ("ctc", type, () => Agree(d =>
            {
                var loss = Registry.Poisoned(type, ctc.B);
                return (s.Ctc(d, type, ctc, logits, loss, null), [(type, ctc.B, loss)]);
            }));
            // float16's gradient is the loss's only where it is held: both versions sum α + β − log y − log P in f16 at
            // magnitudes near 50 (an f16 step of 0.03 there), so their exps differ by a few percent (their errors from
            // float64 reach 1e3 on CUDA too); its loss and every value of the other types are held.
            yield return ("ctc_bwd", type, () => Agree(d =>
            {
                byte[] loss = Registry.Poisoned(type, ctc.B), grad = Registry.Poisoned(type, values);
                return (s.Ctc(d, type, ctc, logits, loss, grad), type == "float16" ? [(type, ctc.B, loss)] : [(type, ctc.B, loss), (type, values, grad)]);
            }));
        }
    }
}
