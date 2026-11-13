// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The products on the device's matrix units (Kernels.Matrix.cs), CUDA's rule for which product takes them: a 16-bit float
// (float16 and the 8-bit floats in f16, bfloat16 in bf16) into f32 sums, an integer type of 8 bits or of values within 8
// bits into s32 or u32 sums while the bound of its sums fits them, each where the device reports a unit of those element
// types (Vulkan.CooperativeMatrices). The lane says the unit (f16mma, bf16mma, s8mma, u8mma: CUDA's names). The kernel's
// shape (UnitsConfig) is the one this device runs fastest, measured once a kind on a 2048³ product of packed rows. The
// operands are read where they are stored when the type is the unit's own and the shape whole tiles; otherwise their rows
// are packed into the kind first (any type, any view). C is the sums' lane at C's positions, the type's own storage where
// that is the lane (int32 and uint32 by s32 and u32 sums), else stored as the type once after.
using System.Diagnostics;
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The gate's override: false keeps every product off the matrix units, true takes them wherever the rule allows
    /// (null: the same; kept for a measured choice).</summary>
    public bool? Units;

    /// <summary>The gate's override of the measured shape of the units' kernel (null: measured).</summary>
    internal UnitsConfig? UnitsShape;

    /// <summary>The products' last route on the units ("in place", "packed"), for the gate.</summary>
    public string? UnitsRoute;

    private readonly Dictionary<string, UnitsConfig> _unitsShapes = [];

    // The unit's element kind (f16, bf16, s8, u8) a product of f takes at these bounds, or null for the plain lanes.
    private string? UnitsKind(Fmt f, in Bounds bounds)
    {
        if (Units == false) return null;
        string kind;
        if (f.IsFloat)
        {
            if (f.Bits >= 32 || f.IsBlock) return null;                    // f64, f32 and the blocks' values (exact in f32, not in 16 bits)
            kind = f.Is("bfloat16") ? "bf16" : "f16";
        }
        else
        {
            bool narrow = f.Bits <= 8 || bounds.Within(f.Signed ? sbyte.MinValue : 0, f.Signed ? sbyte.MaxValue : byte.MaxValue);
            if (!narrow || bounds.Bound > (f.Signed ? int.MaxValue : uint.MaxValue)) return null;
            kind = f.Signed ? "s8" : "u8";
        }

        return Unit(kind) is null ? null : kind;
    }

    // The device's unit for the kind (its sums f32, s32, u32), where some shape of the kernel tiles it within the device's
    // workgroup memory.
    private CooperativeMatrix? Unit(string kind)
    {
        var unit = vk.Cooperative(kind, kind is "f16" or "bf16" ? "f32" : kind == "s8" ? "s32" : "u32");
        if (unit is null) return null;
        return Shapes(kind, unit.Value).Any() ? unit : null;
    }

    // The shapes of the kernel this device can run for the kind: tiling its unit and subgroups, two stages within its
    // workgroup memory in every orientation.
    private IEnumerable<UnitsConfig> Shapes(string kind, CooperativeMatrix unit)
    {
        int eb = VkKernels.UnitKind(kind).Bytes();
        foreach (var c in VkKernels.UnitsConfigs)
        {
            if (c.Threads > vk.MaxWorkgroup || !VkKernels.UnitsTiles(c, unit, vk.SubgroupSize, eb)) continue;
            bool fits = true;
            foreach (bool ar in new[] { true, false }) foreach (bool br in new[] { true, false }) fits &= VkKernels.UnitsShared(c, eb, ar, br) <= vk.MaxSharedBytes;
            if (fits) yield return c;
        }
    }

    /// <summary>The units kernel's shape for the kind: the gate's when it forces one, else the fastest of the shapes this
    /// device runs, measured once on a 2048³ product of packed rows (A by rows of k, Bᵀ by rows of k).</summary>
    private UnitsConfig Shape(string kind, CooperativeMatrix unit)
    {
        if (UnitsShape is UnitsConfig forced && Shapes(kind, unit).Contains(forced)) return forced;
        if (_unitsShapes.TryGetValue(kind, out var known)) return known;
        var shapes = Shapes(kind, unit).ToArray();
        if (shapes.Length == 1) return _unitsShapes[kind] = shapes[0];
        const int Size = 2048;
        int eb = VkKernels.UnitKind(kind).Bytes();
        ulong a = vk.Rent((long)Size * Size * eb), b = vk.Rent((long)Size * Size * eb), c = vk.Rent((long)Size * Size * 4);
        try
        {
            vk.Clear(a, (long)Size * Size * eb);
            vk.Clear(b, (long)Size * Size * eb);
            UnitsConfig best = shapes[0];
            double bestMs = double.MaxValue;
            foreach (var cfg in shapes)
            {
                var fn = vk.Cached($"units_matmul:{kind}:{cfg}:True:False", () => VkKernels.UnitsMatMul(vk, unit, cfg, true, false));
                void Run() => vk.Launch(fn, Groups(Size, cfg.BN), Groups(Size, cfg.BM), a, b, c, Vulkan.U(Size), Vulkan.U(Size), Vulkan.U(Size), Vulkan.U(Size),
                    Vulkan.U(Size), 0, 0, 0, Vulkan.U((Size + cfg.BM - 1) / cfg.BM));
                Run();
                vk.Synchronize();
                long start = Stopwatch.GetTimestamp();
                for (int i = 0; i < 3; i++) Run();
                vk.Synchronize();
                double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
                _unitsTimes.Add($"{kind} {cfg}: {ms:0.000} ms");
                if (ms < bestMs) (bestMs, best) = (ms, cfg);
            }

            return _unitsShapes[kind] = best;
        }
        finally
        {
            vk.Return(a);
            vk.Return(b);
            vk.Return(c);
        }
    }

    private readonly List<string> _unitsTimes = [];

    /// <summary>Every shape of the units' kernel this device runs for some kind it has units of (the gate checks each).</summary>
    internal IEnumerable<UnitsConfig> UnitsShapes() =>
        new[] { "f16", "bf16", "s8", "u8" }.Select(k => (k, u: Unit(k))).Where(p => p.u is not null).SelectMany(p => Shapes(p.k, p.u!.Value)).Distinct();

    /// <summary>The units kernel's shapes as measured on this device (each kind's candidates and the one taken), for --report.</summary>
    public string UnitsMeasured() => _unitsTimes.Count == 0 ? "" :
        "matrix units' kernel shapes measured (2048³): " + string.Join("; ", _unitsTimes) + "; taken: " + string.Join(", ", _unitsShapes.Select(p => $"{p.Key} {p.Value}"));

    private static Lane UnitsLane(string kind) => kind is "f16" or "bf16" ? Lane.F32 : kind == "s8" ? Lane.S32 : Lane.U32;

    /// <summary>C = op(A)·op(B) over the view's batches on the matrix units of the kind; returns the lane's name.</summary>
    private string MatMulUnits<TN>(ProductView v, string kind, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = UnitsLane(kind);
        bool direct = Direct(f, lane);
        ulong lanes = direct ? c : vk.Rent(v.ValuesC * lane.Size);
        try
        {
            UnitsProduct(v, kind, a, f, 0, b, f, 0, lanes);
            if (!direct)
            {
                long span = (long)v.M * v.N, period = v.Batch > 1 ? Math.Max(span, v.StrideCOrDefault) : span;
                BackKeep(f, lane, lanes, c, v.ValuesC, period, span);
            }
        }
        finally
        {
            if (!direct) vk.Return(lanes);
        }

        return kind + "mma";
    }

    // The units' kind both operands' float formats hold exactly (float16 and the 8-bit floats: f16; bfloat16: bf16), where
    // the device has its unit; null otherwise (a lane's own storage among them: the composition's intermediates).
    private string? UnitsKindOf(Fmt fa, Fmt fb)
    {
        static string? KindOf(Fmt f) => f.IsBlock ? null : f.Is("float16") || f.Is("float8_e4m3") || f.Is("float8_e5m2") ? "f16" : f.Is("bfloat16") ? "bf16" : null;
        return Units != false && KindOf(fa) is string ka && KindOf(fb) == ka && Unit(ka) is not null ? ka : null;
    }

    /// <summary>The units' sums (the kind's lane at c, C's batches strideC apart) of op(A)·op(B) over the view's batches,
    /// A stored as fa from value aOff, B as fb from bOff: read in place where both are the kind's own type, the shape whole
    /// tiles and the operands 16-byte aligned, else their rows packed into the kind (PackUnits).</summary>
    private void UnitsProduct(ProductView v, string kind, ulong a, Fmt fa, long aOff, ulong b, Fmt fb, long bOff, ulong lanes)
    {
        var unit = Unit(kind)!.Value;
        var cfg = Shape(kind, unit);
        int eb = VkKernels.UnitKind(kind).Bytes(), bk = cfg.KBytes / eb;
        static bool OwnKind(Fmt f, string kind) => (f.Is("float16") && kind == "f16") || (f.Is("bfloat16") && kind == "bf16") || (f.Is("int8") && kind == "s8") || (f.Is("uint8") && kind == "u8");
        ulong at = a + (ulong)(aOff * eb), bt = b + (ulong)(bOff * eb);
        bool whole = v.M % cfg.BM == 0 && v.N % cfg.BN == 0 && v.K > 0 && v.K % bk == 0 && at % 16 == 0 && bt % 16 == 0
            && (v.Batch == 1 || (v.StrideAOrDefault * eb % 16 == 0 && v.StrideBOrDefault * eb % 16 == 0));
        var rented = new List<ulong>();
        try
        {
            int tilesM = (v.M + cfg.BM - 1) / cfg.BM;
            uint groupsX = Groups(v.N, cfg.BN), groupsY = (uint)((long)tilesM * v.Batch);
            if (OwnKind(fa, kind) && OwnKind(fb, kind) && whole)
            {
                var fn = vk.Cached($"units_matmul:{kind}:{cfg}:{!v.TransA}:{!v.TransB}", () => VkKernels.UnitsMatMul(vk, unit, cfg, !v.TransA, !v.TransB));
                vk.Launch(fn, groupsX, groupsY, at, bt, lanes, Vulkan.U(v.TransA ? v.M : v.K), Vulkan.U(v.TransB ? v.K : v.N), Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(v.K),
                    Vulkan.U(v.StrideAOrDefault), Vulkan.U(v.StrideBOrDefault), Vulkan.U(v.StrideCOrDefault), Vulkan.U(tilesM));
                UnitsRoute = "in place";
                return;
            }

            // A's rows and Bᵀ's rows packed into the kind: whole tiles of rows, whole slices of k, zeros past them.
            if (aOff + (v.Batch - 1) * v.StrideAOrDefault > uint.MaxValue || bOff + (v.Batch - 1) * v.StrideBOrDefault > uint.MaxValue)
                throw new NotSupportedException("vulkan: a product's operand past 2^32 values");
            int mp = tilesM * cfg.BM, np = (int)groupsX * cfg.BN, kp = Math.Max(bk, (v.K + bk - 1) / bk * bk);
            long rowBytes = (long)kp * eb, aBytes = mp * rowBytes, bBytes = np * rowBytes;
            ulong pa = vk.Rent(v.Batch * aBytes), pb = vk.Rent(v.Batch * bBytes);
            rented.Add(pa);
            rented.Add(pb);
            var packA = vk.Cached($"pack_units:{fa.Name}:{kind}", () => VkKernels.PackUnits(vk, fa, kind));
            var packB = vk.Cached($"pack_units:{fb.Name}:{kind}", () => VkKernels.PackUnits(vk, fb, kind));
            int wordsPerRow = (int)(rowBytes / 4);
            for (int z = 0; z < v.Batch; z++)
            {
                // A (m × k) by rows: stored by rows of k (row stride k), or as Aᵀ (k × m: row stride 1, k's stride m).
                vk.Launch(packA, Groups((long)mp * wordsPerRow, VkKernels.Threads), 1, a, pa + (ulong)(z * aBytes), Vulkan.U(v.M), Vulkan.U(v.K),
                    Vulkan.U(v.TransA ? 1 : v.K), Vulkan.U(v.TransA ? v.M : 1), Vulkan.U(wordsPerRow), Vulkan.U(mp * wordsPerRow), Vulkan.U(aOff + z * v.StrideAOrDefault));
                // Bᵀ (n × k) by rows: B stored k × n (column j: row stride 1, k's stride n), or as Bᵀ (row stride k).
                vk.Launch(packB, Groups((long)np * wordsPerRow, VkKernels.Threads), 1, b, pb + (ulong)(z * bBytes), Vulkan.U(v.N), Vulkan.U(v.K),
                    Vulkan.U(v.TransB ? v.K : 1), Vulkan.U(v.TransB ? 1 : v.N), Vulkan.U(wordsPerRow), Vulkan.U(np * wordsPerRow), Vulkan.U(bOff + z * v.StrideBOrDefault));
            }

            var fp = vk.Cached($"units_matmul:{kind}:{cfg}:True:False", () => VkKernels.UnitsMatMul(vk, unit, cfg, true, false));
            vk.Launch(fp, groupsX, groupsY, pa, pb, lanes, Vulkan.U(kp), Vulkan.U(kp), Vulkan.U(v.M), Vulkan.U(v.N), Vulkan.U(kp),
                Vulkan.U(aBytes / eb), Vulkan.U(bBytes / eb), Vulkan.U(v.StrideCOrDefault), Vulkan.U(tilesM));
            UnitsRoute = "packed";
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }
}
