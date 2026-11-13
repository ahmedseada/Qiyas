// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The INumberBase<T> version's product cells on Vulkan (matmul, the convolution, the box losses, ctc): not written yet,
// so their cells are blank (null) on Vulkan.
using System.Numerics;
using Adad.Vulkan;

namespace Qiyas.Bench;

internal static partial class NumberVulkan
{
    private static IEnumerable<(string, string, Vulkan.Maker)> Products() => [];

    public static IBenchRun? MatMul<T>(byte[] x, byte[] y, byte[] z, int m, int n, int k) where T : unmanaged, INumberBase<T> => null;

    public static IBenchRun? Conv<T>(ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) where T : unmanaged, INumberBase<T> => null;

    public static IBenchRun? BoxLoss<T>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[]? dloss, byte[] output, int count) where T : unmanaged, IFloatingPointIeee754<T> => null;

    public static IBenchRun? Ctc<T>(CtcArgs a, int[] offsets, byte[] logits, byte[] loss, byte[]? grad) where T : unmanaged, IFloatingPointIeee754<T> => null;
}
