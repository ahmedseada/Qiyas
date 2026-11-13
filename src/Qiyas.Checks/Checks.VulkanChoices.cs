// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The Vulkan products under every choice a run's measurement may take. On the matrix units, every shape of the kernel the
// device runs (VulkanOps.UnitsShape forced, as CUDA's tensor configurations are): each shape's products of every type, at
// whole tiles of the largest shape (the operands read in place for the units' own types) and at sizes off every tile (the
// packed rows), transposed, batched and the backward pair, within the product's rounding. On the plain kernel, each
// tiling (VulkanOps.TilingForced: 8 or 4 values a side an invocation, k whole or split in slices added in order) at a
// long k and at sizes off every tile, every type; the experts' grouped products at each value a side, and the experts'
// products an expert a units product (VulkanOps.ExpertsOnUnits forced).
using Adad.Vulkan;

namespace Qiyas;

internal static partial class Checks
{
    private static bool VulkanChoices(VulkanOps ops)
    {
        bool ok = true;
        var tilings = new VulkanOps.PlainTiling[] { new(8, 1), new(4, 1), new(8, 3), new(4, 4) };
        try
        {
            foreach (var t in tilings)
            {
                ops.TilingForced = t;
                ok &= ProductsWith(ops, "vulkan", 37, 29, 1100, $" [plain {t.Micro}x{t.Micro}, {t.Slices} slices]");
                ok &= ProductsWith(ops, "vulkan", 130, 140, 53, $" [plain {t.Micro}x{t.Micro}, {t.Slices} slices]");
                if (t.Slices == 1) ok &= ExpertsCheck(ops, "vulkan");                // the experts' grouped products at its values a side
            }
        }
        finally { ops.TilingForced = null; }
        if (ok) Console.WriteLine($"check {"products' plain tilings",-36} vulkan every type under each tiling (8 or 4 values a side; k whole, in 3 or 4 slices) at 37 x 29 x 1100 and 130 x 140 x 53, within the product's rounding");

        var shapes = ops.UnitsShapes().ToArray();
        if (shapes.Length == 0)
        {
            Console.WriteLine($"check {"products on the matrix units",-36} not on this device: the plain lanes");
            return ok;
        }

        try
        {
            foreach (var shape in shapes)
            {
                ops.UnitsShape = shape;
                ok &= ProductsWith(ops, "vulkan", 256, 256, 128, $" [units {shape}, in place]");
                ok &= ProductsWith(ops, "vulkan", 37, 29, 53, $" [units {shape}]");
            }
        }
        finally { ops.UnitsShape = null; }
        if (ok) Console.WriteLine($"check {"products on the matrix units",-36} vulkan every type under each of the kernel's {shapes.Length} shapes ({string.Join(", ", shapes)}): in place at 256 x 256 x 128 and packed at 37 x 29 x 53, within the product's rounding");

        // The experts' products an expert a units product wherever the type and lane allow (the measurement's other choice).
        try
        {
            ops.ExpertsOnUnits = true;
            ok &= ExpertsCheck(ops, "vulkan");
        }
        finally { ops.ExpertsOnUnits = null; }
        if (ok) Console.WriteLine($"check {"experts on the matrix units",-36} vulkan every type with the experts' products on the units forced where they apply");
        return ok;
    }
}
