// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad.Abstraction;

/// <summary>The box overlap losses (Boxes.cs) over count pairs of corner boxes (x1, y1, x2, y2): predicted and target count × 4
/// values as the type, read where they are stored; each pair in the type's float lane.</summary>
public interface IBoxes<TBuffer>
{
    /// <summary>loss[i] (count values as the type) of pair i by the overlap's formula; eps keeps the divisions finite
    /// (torchvision's 1e-7, Boxes.Eps).</summary>
    string BoxLoss<TN>(BoxOverlap overlap, TBuffer predicted, TBuffer target, TBuffer loss, int count, double eps) where TN : struct, IFloatAdad<TN>;

    /// <summary>dpredicted[i] = dloss[i] · ∂loss[i] / ∂predicted[i] (count × 4 values as the type, written), the targets
    /// constants and CIoU's α a constant; a tie of a minimum or a maximum shares its gradient in halves.</summary>
    string BoxLossBackward<TN>(BoxOverlap overlap, TBuffer predicted, TBuffer target, TBuffer dloss, TBuffer dpredicted, int count, double eps) where TN : struct, IFloatAdad<TN>;
}
