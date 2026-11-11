// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A device: Adad's operations over one device's memory. Every operation is written once over the number type TN, under
// the operation's own constraint (IAdad for every type, IFloatAdad for the float formulas); each device implements
// the families on its own buffers (TBuffer: the CPU's byte arrays, CUDA's device pointers) and index buffers (TIndex:
// int arrays, device pointers), with the lanes its hardware reports for each type. A device may implement a family at a
// time; IDevice is all of them. Operations return the name of the lane they ran in.
namespace Adad.Abstraction;

public interface IDevice<TBuffer, TIndex> : IMemory<TBuffer, TIndex>, IElementwise<TBuffer>, IReductions<TBuffer>, IRows<TBuffer, TIndex>,
    ILayouts<TBuffer, TIndex>, IProducts<TBuffer>, IAttention<TBuffer>, IOptimizers<TBuffer>, IConvolution<TBuffer>, IRecurrent<TBuffer>,
    IRouting<TBuffer, TIndex>, ICtc<TBuffer>, IBoxes<TBuffer>, IExperts<TBuffer, TIndex>, ICache<TBuffer, TIndex>
{
    /// <summary>What the kernels were chosen by on this device (measured, or as the hardware reports it).</summary>
    string Report();

    /// <summary>The decisions taken from measurements so far.</summary>
    string Measured();
}
