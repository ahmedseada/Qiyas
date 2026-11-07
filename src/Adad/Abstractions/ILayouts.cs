// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Adad;

/// <summary>Layout and indexing (Layout.cs): data moved by index arithmetic, exact in every type.</summary>
public interface ILayouts<TBuffer, TIndex>
{
    /// <summary>The layout operation k.Name over k's shape (k.P, k.Count, k.InputCount): x and the optional second input x2
    /// (k.SecondCount values) and index buffer into output.</summary>
    string Layout<TN>(LayoutArgs k, TBuffer x, TIndex? index, TBuffer output, TBuffer? x2) where TN : struct, IAdad<TN>;
}
