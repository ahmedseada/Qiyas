// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's layout operations (Layout.cs): the lane by the operation (the type's own for moves, the float lane for
// bilinear interpolation, the sum lane by the bounds for scatter-add and col2im), scatter-add's ordering built on the
// device (counts, offsets, ordered sources), then one kernel with a thread per output word.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>The gate's override of the per-value layout kernel for packed types: true always, false never (the word
    /// kernel), null by the output's size.</summary>
    internal bool? LayoutPerValue;

    /// <summary>A packed type's layout takes a thread per value (Ptx.LayoutPerValue) when a thread per word would not fill
    /// the device: fewer words than its multiprocessors hold threads.</summary>
    private bool PerValueLayout(Fmt f, int count) =>
        Ptx.PerValueFits(f) && (LayoutPerValue ?? (long)f.Words(count) < (long)gpu.Multiprocessors * gpu.ThreadsPerMultiprocessor);

    /// <summary>A canonical strided view (LayoutArgs.StridedKernel: d0 … d(n−1) | s0 … s(n−1) | base) whose input runs along
    /// a dimension j (stride 1) other than the last, the last not (stride ≠ 1): its output rows read across the input's runs
    /// a value each, every warp's reads 32 apart. Taken as a batch of tiled transposes (Ptx.Transpose, strided): A[r][c]
    /// with r the last dimension (input stride s(n−1)) and c dimension j, B[c][r] at the output's stride of j; the other
    /// dimensions the batch. Whole output words only (the last dimension a multiple of the values a word holds), at most
    /// TransposeBatchDims others and 65535 batches; false (nothing launched) otherwise.</summary>
    private bool StridedTransposed(Fmt f, Lane lane, int[] p, ulong x, ulong output)
    {
        int n = p.Length / 2;
        if (f.IsBlock || n < 2 || p[2 * n - 1] == 1 || p[n - 1] % f.Per != 0 || p[n - 1] < 2) return false;
        int j = -1;
        for (int i = 0; i < n - 1; i++) if (p[n + i] == 1 && p[i] >= 2) { j = i; break; }
        if (j < 0 || n - 2 > Ptx.TransposeBatchDims) return false;
        var outStride = new long[n];
        long acc = 1;
        for (int i = n - 1; i >= 0; i--) { outStride[i] = acc; acc *= p[i]; }
        long batches = 1;
        var others = new List<int>();
        for (int i = 0; i < n - 1; i++) if (i != j) { others.Add(i); batches *= p[i]; }
        if (batches == 0 || batches > 65535 || acc > uint.MaxValue) return false;
        var z = new ulong[3 * Ptx.TransposeBatchDims];
        int slot0 = Ptx.TransposeBatchDims - others.Count;
        for (int s = 0; s < Ptx.TransposeBatchDims; s++) z[3 * s] = 1;
        for (int o = 0; o < others.Count; o++)
        {
            int dim = others[o], s = slot0 + o;
            z[3 * s] = Gpu.U(p[dim]);
            z[3 * s + 1] = Gpu.U(p[n + dim]);
            z[3 * s + 2] = (ulong)(uint)outStride[dim];
        }

        bool valueWise = false;
        var fn = gpu.Cached($"transpose:{f.Name}:{valueWise}:s", (out string e) => Ptx.Transpose(gpu, f, lane, valueWise, out e, strided: true));
        int rows = p[n - 1], cols = p[j];
        Gpu.LaunchZ(fn, Grid(cols, Ptx.TransposeTile), Grid(rows, Ptx.TransposeTile), (uint)batches, 256, 1, 0,
            [x, output, Gpu.U(rows), Gpu.U(cols), Gpu.U(p[2 * n - 1]), (ulong)(uint)outStride[j], Gpu.U(p[2 * n]), .. z]);
        return true;
    }

    /// <summary>A layout operation (LayoutArgs.Names) from x (and the index array) into output, all on the device. Returns the lane's name.</summary>
    public string Layout<TN>(LayoutArgs k, ulong x, ulong idx, ulong output, ulong x2 = 0) where TN : struct, IAdad<TN> => Layout<TN>(k, x, idx, output, x2, null);

    /// <summary>As Layout, x (and x2) of inFmt where it is not TN's: a float composition's f32 lane storage (the lane f32).</summary>
    internal string Layout<TN>(LayoutArgs k, ulong x, ulong idx, ulong output, ulong x2, Fmt? inFmt) where TN : struct, IAdad<TN>
    {
        k = k.ForKernels();
        var f = Fmt.Of<TN>();
        // A strided view takes the kernel of its canonical rank (a division a dimension a value).
        string kn = LayoutArgs.KernelName(k.Name, k.P);
        // A block format takes the same lanes (f32: a float type's) with its inputs decoded where they are stored, and its
        // output written by a warp or a CTA per block (Ptx.LayoutBlocks).
        var (rows, cols) = LayoutArgs.OutputRows(k);
        Lane lane;
        if (inFmt is not null) lane = inFmt.Natural;
        else if (LayoutArgs.Sums(k.Name))
        {
            var (lo, hi) = Range<TN>(x, k.InputCount);
            var (lo2, hi2) = x2 != 0 && k.Name == "depthwise_conv" ? Range<TN>(x2, k.P[0] * k.P[3] * k.P[4]) : (1, 1);
            lane = SumLane(f, new Bounds(lo, hi, lo2, hi2, LayoutArgs.Terms(k.Name, k.P)));
            if (!f.IsFloat && lane.IsFloat) lane = f.Signed ? Lane.S32 : Lane.U32;   // an integer sum stays an integer (its store converts to the type)
        }
        else lane = LayoutArgs.FloatLane(k.Name) ? RowLane(f) : f.Natural;

        if (k.Count == 0) return lane.Name;                                         // no outputs (a window longer than its image)
        ulong aux = 0;
        try
        {
            if (k.Name == "scatter_add_rows")
            {
                int n = k.P[0], targets = k.P[2];
                ulong counts = gpu.Rent(targets * 4L);
                aux = gpu.Rent((targets + 1 + n) * 4L);
                try
                {
                    // The rows' offsets and the sources in order of their row (stable): the radix ordering (GpuOps.Radix.cs).
                    OrderByTarget(idx, n, targets, counts, aux, aux + (ulong)((targets + 1) * 4L));
                }
                finally
                {
                    gpu.Return(counts);
                }
            }

            var args = new ulong[8 + Ptx.LayoutParamCount + Ptx.LayoutDivisorArgs];
            for (int i = 0; i < Ptx.LayoutParamCount; i++) args[8 + i] = Gpu.U(i < k.P.Length ? k.P[i] : 0);
            if (k.Name is "maxpool_bwd" or "adaptive_maxpool_bwd")
            {
                // Each window's first-ranked of the forward input in Adad's order, once (Ptx.LayoutWindowArgmax), into aux.
                var (OH, OW) = k.Name == "maxpool_bwd" ? LayoutArgs.PoolOut(k.P) : (k.P[3], k.P[4]);
                int windows = k.P[0] * OH * OW;
                aux = gpu.Rent(Math.Max(1, windows) * 4L);
                // An adaptive pool's large windows a warp each (Ptx.AdaptiveByWarp), else a thread each.
                bool byWarp = k.Name == "adaptive_maxpool_bwd" && Ptx.AdaptiveByWarp(k.P);
                var argmax = byWarp
                    ? gpu.Cached($"layout_argmax_warp:{f.Name}:{lane.Name}", (out string e) => Ptx.LayoutWindowArgmaxWarp(gpu, f, lane, out e))
                    : gpu.Cached($"layout_argmax:{k.Name}:{f.Name}:{lane.Name}", (out string e) => Ptx.LayoutWindowArgmax(gpu, f, lane, k.Name, out e));
                args[1] = x2; args[4] = aux; args[5] = Gpu.U(windows);
                Ptx.LayoutDivisors(args);
                if (windows > 0) Gpu.Launch(argmax, Grid(windows * (byWarp ? 32L : 1L), 256), 1, 256, 1, 0, args);
            }

            args[0] = x; args[1] = x2; args[2] = idx; args[3] = aux; args[4] = output; args[5] = Gpu.U(k.Count); args[6] = Gpu.U(rows); args[7] = Gpu.U(cols);
            Ptx.LayoutDivisors(args);
            if (f.IsBlock)
            {
                var fb = gpu.Cached($"layout_blocks:{kn}:{f.Name}:{inFmt?.Name}", (out string e) => Ptx.LayoutBlocks(gpu, f, kn, out e, inFmt));
                if (k.Count > 0 && Ptx.BlockByCta(f)) Gpu.Launch(fb, (uint)((k.Count + f.BlockValues - 1) / f.BlockValues), 1, (uint)f.BlockValues, 1, 0, args);
                else if (k.Count > 0) Gpu.Launch(fb, BlockGrid(f, k.Count), 1, 256, 1, 0, args);
                return "f32 (blocks in place)";
            }

            if (kn.StartsWith("strided", StringComparison.Ordinal) && inFmt is null && StridedTransposed(f, lane, k.P, x, output)) return lane.Name;
            if (k.Name == "adaptive_maxpool" && inFmt is null && Ptx.AdaptiveByWarp(k.P))
            {
                // Large windows: a warp each (the word's values' windows one after another), the order's max the same.
                var fw = gpu.Cached($"layout_adaptive_max_warp:{f.Name}:{lane.Name}", (out string e) => Ptx.AdaptiveMaxWarp(gpu, f, lane, out e));
                Gpu.Launch(fw, Grid(f.Words(k.Count) * 32L, 256), 1, 256, 1, 0, args);
                return lane.Name;
            }

            bool perValue = inFmt is null && PerValueLayout(f, k.Count);
            var fn = gpu.Cached($"layout:{kn}:{f.Name}:{lane.Name}{(perValue ? ":v" : "")}:{inFmt?.Name}", (out string e) => Ptx.Layout(gpu, f, lane, kn, perValue, out e, inFmt));
            Gpu.Launch(fn, Grid(perValue ? (long)f.Words(k.Count) * f.Per : f.Words(k.Count), 256), 1, 256, 1, 0, args);
        }
        finally
        {
            if (aux != 0) gpu.Return(aux);
        }

        return lane.Name;
    }
}
