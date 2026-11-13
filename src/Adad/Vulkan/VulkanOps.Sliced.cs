// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// float64 products on the s8 matrix units, exactly (Kernels.Sliced.cs; CUDA's sliced float64, the same definition and
// bits): where the device proved the route (SlicedProven, set by its gate), has s8 units, the sums of a diagonal fit s32
// (k ≤ 2^14) and the route measures faster than the f64 lane at the shape (once per shape). Each batch's rows and columns
// sliced by their own exponents (read back together: one wait), its twelve diagonals' integer products on the units, then
// combined into C. A batch whose exponents are past the slicing scale (or holding a value that is not finite) refuses the
// route before anything is computed: the f64 lane takes the product.
using System.Diagnostics;
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>Set by the device proof (Checks): the sliced product reproduces float64 math to within an ulp on this device.</summary>
    public bool SlicedProven { get; set; }

    /// <summary>The checks' override of the measured choice (true: the sliced product wherever it is eligible); null: measured.</summary>
    public bool? SlicedChoice { get; set; }

    private readonly Dictionary<string, bool> _slicedFaster = [];

    /// <summary>The longest k the sliced product takes: a diagonal's k' = 8·k terms of digit products up to 2^14 within s32.</summary>
    public const int SlicedMaxK = 1 << 14;

    // Whether the float64 product of the view (A from value aOff, B from bOff) took the sliced route (and computed C).
    private bool Sliced(ProductView v, ulong a, long aOff, ulong b, long bOff, ulong c, Action plain)
    {
        if (!SlicedProven || v.K == 0 || v.K > SlicedMaxK || Unit("s8") is null) return false;
        if (SlicedChoice is bool chosen) return chosen && MatMulSliced(v, a, aOff, b, bOff, c);
        string key = $"{v.M}x{v.N}x{v.K}:{v.TransA}:{v.TransB}:{v.Batch}";
        if (!_slicedFaster.TryGetValue(key, out bool faster))
        {
            double Time(Func<bool> run)
            {
                if (!run()) return double.MaxValue;
                vk.Synchronize();
                long start = Stopwatch.GetTimestamp();
                for (int i = 0; i < 3; i++) run();
                vk.Synchronize();
                return Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            }

            double lane = Time(() => { plain(); return true; }), sliced = Time(() => MatMulSliced(v, a, aOff, b, bOff, c));
            faster = _slicedFaster[key] = sliced < lane;
            _unitsTimes.Add($"float64 {key}: f64 lane {lane:0.000} ms, sliced {(sliced == double.MaxValue ? "refused" : $"{sliced:0.000} ms")}");
        }

        return faster && MatMulSliced(v, a, aOff, b, bOff, c);
    }

    /// <summary>C = op(A)·op(B) of float64 over the view's batches on the s8 units, exactly (Kernels.Sliced.cs). False when the
    /// device has no s8 units, k is past SlicedMaxK, or a batch's exponents are past the slicing scale: nothing was computed.</summary>
    public bool MatMulSliced(ProductView v, ulong a, ulong b, ulong c) => MatMulSliced(v, a, 0, b, 0, c);

    /// <summary>MatMulSliced with A's first value at aOff and B's at bOff (a composition's operands inside larger tensors).</summary>
    public bool MatMulSliced(ProductView v, ulong a, long aOff, ulong b, long bOff, ulong c)
    {
        if (aOff + (v.Batch - 1) * v.StrideAOrDefault > uint.MaxValue || bOff + (v.Batch - 1) * v.StrideBOrDefault > uint.MaxValue) return false;
        if (v.K == 0 || v.K > SlicedMaxK || Unit("s8") is not CooperativeMatrix unit) return false;
        if (v.M == 0 || v.N == 0 || v.Batch == 0) return true;
        var cfg = Shape("s8", unit);
        int m = v.M, n = v.N, k = v.K, batches = v.Batch;
        int kpad = (k + 63) / 64 * 64, rowLen = VkKernels.Slices * kpad;
        int tilesM = (m + cfg.BM - 1) / cfg.BM, mp = tilesM * cfg.BM, np = (n + cfg.BN - 1) / cfg.BN * cfg.BN;
        long mn = (long)m * n;
        // A's value (i, p) at first + i·rs + p·cs, B's (p, j) at first + p·rs + j·cs, as the view stores them.
        uint aRs = (uint)(v.TransA ? 1 : k), aCs = (uint)(v.TransA ? m : 1), bRs = (uint)(v.TransB ? 1 : n), bCs = (uint)(v.TransB ? k : 1);
        var rented = new List<ulong>();
        ulong Rent(long bytes) { ulong p = vk.Rent(Math.Max(16, bytes)); rented.Add(p); return p; }
        try
        {
            ulong e = Rent((long)batches * m * 4), f = Rent((long)batches * n * 4);
            var rowExp = vk.Cached("slice_rowexp_f64", () => VkKernels.RowExponents(vk));
            var colExp = vk.Cached("slice_colexp_f64", () => VkKernels.ColumnExponents(vk));
            for (int z = 0; z < batches; z++)
            {
                vk.Launch(rowExp, (uint)m, 1, a, e + (ulong)(z * m * 4L), Vulkan.U(k), aRs, aCs, Vulkan.U(aOff + z * v.StrideAOrDefault));
                vk.Launch(colExp, Groups(n, VkKernels.Threads), 1, b, f + (ulong)(z * n * 4L), Vulkan.U(k), Vulkan.U(n), bRs, bCs, Vulkan.U(bOff + z * v.StrideBOrDefault));
            }

            // The exponents must keep every scale a normal float64 (the slicing scale 2^(64 − e), the result's 2^(e + f + shift − 104)).
            var ea = new int[batches * m];
            var fa = new int[batches * n];
            vk.Download<int>(e, ea);
            vk.Download<int>(f, fa);
            // A row or column of zeros has no exponent (far below any scale): it takes 0, any finite scale slicing it to zeros.
            bool zeroRows = false, zeroCols = false;
            for (int i = 0; i < ea.Length; i++) if (ea[i] < -900) { ea[i] = 0; zeroRows = true; }
            for (int i = 0; i < fa.Length; i++) if (fa[i] < -900) { fa[i] = 0; zeroCols = true; }
            for (int z = 0; z < batches; z++)
            {
                int eMin = int.MaxValue, eMax = int.MinValue, fMin = int.MaxValue, fMax = int.MinValue;
                foreach (int x in ea.AsSpan(z * m, m)) { eMin = Math.Min(eMin, x); eMax = Math.Max(eMax, x); }
                foreach (int x in fa.AsSpan(z * n, n)) { fMin = Math.Min(fMin, x); fMax = Math.Max(fMax, x); }
                if (eMax > 1086 || fMax > 1086 || eMin + fMin < -918 || eMax + fMax > 1057) return false;
            }

            if (zeroRows) vk.Upload<int>(ea, e);
            if (zeroCols) vk.Upload<int>(fa, f);
            ulong sa = Rent((long)mp * rowLen), bt = Rent((long)np * rowLen), d = Rent(VkKernels.Diagonals * mn * 4);
            var sliceRows = vk.Cached("slice_rows_f64", () => VkKernels.SliceRows(vk));
            var sliceCols = vk.Cached("slice_cols_f64", () => VkKernels.SliceColumns(vk));
            var combine = vk.Cached("slice_combine_f64", () => VkKernels.CombineSlices(vk));
            var product = vk.Cached($"units_matmul:s8:{cfg}:True:False", () => VkKernels.UnitsMatMul(vk, unit, cfg, true, false));
            uint groupsX = Groups(n, cfg.BN);
            for (int z = 0; z < batches; z++)
            {
                ulong ez = e + (ulong)(z * m * 4L), fz = f + (ulong)(z * n * 4L);
                vk.Launch(sliceRows, Groups((long)mp * kpad / 4, VkKernels.Threads), 1, a, ez, sa, Vulkan.U(m), Vulkan.U(k), Vulkan.U(kpad), aRs, aCs, Vulkan.U(mp),
                    Vulkan.U(aOff + z * v.StrideAOrDefault));
                vk.Launch(sliceCols, Groups((long)np * kpad / 4, VkKernels.Threads), 1, b, fz, bt, Vulkan.U(n), Vulkan.U(k), Vulkan.U(kpad), bRs, bCs, Vulkan.U(np),
                    Vulkan.U(bOff + z * v.StrideBOrDefault));
                // One product per diagonal: the digit pairs (t, u) with t + u = diag. B's digits are stacked most significant
                // last, so for diag below the slice count A's first diag + 1 slices meet B's last diag + 1; above it, A's last
                // 15 − diag meet B's first.
                for (int diag = 0; diag < VkKernels.Diagonals; diag++)
                {
                    int s = VkKernels.Slices, pairs = diag < s ? diag + 1 : 2 * s - 1 - diag, extent = pairs * kpad;
                    ulong pa = diag < s ? sa : sa + (ulong)((diag - s + 1) * kpad);
                    ulong pb = diag < s ? bt + (ulong)((s - 1 - diag) * kpad) : bt;
                    vk.Launch(product, groupsX, (uint)tilesM, pa, pb, d + (ulong)(diag * mn * 4), Vulkan.U(rowLen), Vulkan.U(rowLen), Vulkan.U(m), Vulkan.U(n),
                        Vulkan.U(extent), 0, 0, 0, Vulkan.U(tilesM));
                }

                vk.Launch(combine, Groups(mn, VkKernels.Threads), 1, d, ez, fz, c + (ulong)(z * v.StrideCOrDefault * 8L), Vulkan.U(m), Vulkan.U(n));
            }

            return true;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }
}
