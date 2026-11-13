// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's product units in a kernel, as the device reports them (Vulkan.CooperativeMatrices, Vulkan.IntDot8Signed /
// IntDot8Unsigned): a subgroup's cooperative matrices (SPV_KHR_cooperative_matrix: loaded from workgroup memory or memory
// by address, multiplied and added by the matrix units, stored), the packed 4 × 8-bit integer dot product
// (SPV_KHR_integer_dot_product), and workgroup arrays of vectors (one load or store of 8 or 16 bytes, as the units'
// operands are staged). A matrix's elements are the subgroup's, in the device's order: they are reached only through
// loads, stores, the product and element-wise arithmetic, never one by one.
namespace Adad.Vulkan;

/// <summary>What a cooperative matrix is for: the product's A (M × K), B (K × N), or the accumulator (M × N).</summary>
internal enum Use
{
    A = 0,
    B = 1,
    Accumulator = 2,
}

/// <summary>A cooperative matrix type: rows × columns of <see cref="Component"/>, for <see cref="Use"/>.</summary>
internal readonly record struct CoopType(uint Id, K Component, int Rows, int Columns, Use Use);

/// <summary>A cooperative matrix value of a subgroup.</summary>
internal readonly struct M(Spv b, uint id, CoopType type)
{
    public readonly Spv B = b;
    public readonly uint Id = id;
    public readonly CoopType Type = type;
}

/// <summary>A function variable holding a cooperative matrix (an accumulator across a loop's iterations).</summary>
internal readonly struct MVar(Spv b, uint pointer, CoopType type)
{
    public M M
    {
        get => b.LoadMatrix(pointer, type);
        set => b.StoreMatrix(pointer, value);
    }
}

/// <summary>A workgroup array of vectors of <see cref="Count"/> values of <see cref="Kind"/> (8 or 16 bytes an element).</summary>
internal readonly struct VArr(Spv b, uint variable, uint vectorType, K kind, int count, int length)
{
    public readonly K Kind = kind;
    public readonly int Count = count;
    public readonly int Length = length;
    internal readonly uint Variable = variable;
    internal readonly uint VectorType = vectorType;

    /// <summary>The vector at <paramref name="index"/>, its values.</summary>
    public V[] Load(V index) => b.VectorLoad(this, index);

    /// <summary>Stores <paramref name="values"/> (Count of them, of Kind) as the vector at <paramref name="index"/>.</summary>
    public void Store(V index, params V[] values) => b.VectorStore(this, index, values);
}

internal sealed partial class Spv
{
    private V? _subgroupId;

    /// <summary>The invocation's subgroup within its workgroup (SubgroupId: every invocation of a subgroup has the same).</summary>
    public V SubgroupId
    {
        get
        {
            if (_subgroupId is V known) return known;
            _m.Capability(Cap.GroupNonUniform);
            uint u32 = Type(K.U32);
            uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Input, u32), StorageClass.Input);
            _m.Decorate(variable, Decoration.BuiltIn, (uint)BuiltIn.SubgroupId);
            var v = new V(this, _m.PrologueValue(SpirvOp.Load, u32, variable), K.U32);
            _subgroupId = v;
            return v;
        }
    }

    /// <summary>Every invocation of the subgroup waits here; their workgroup-memory writes before it are visible after it
    /// (a barrier a subgroup may reach alone, as one taking a path of its own).</summary>
    public void SubgroupBarrier() => _m.Code(SpirvOp.ControlBarrier, Const(K.U32, 3), Const(K.U32, 3), Const(K.U32, 0x108));

    // ---------------------------------------------------------------- cooperative matrices

    /// <summary>The cooperative matrix type of rows × columns of <paramref name="component"/> (f16, bf16, f32, s8, u8, s32, u32) for
    /// <paramref name="use"/>, a subgroup's.</summary>
    public CoopType Coop(K component, int rows, int columns, Use use)
    {
        if (component == K.BF16) _m.Capability(Cap.BFloat16CooperativeMatrix);
        return new CoopType(_m.TypeCooperativeMatrix(Type(component), rows, columns, (int)use), component, rows, columns, use);
    }

    /// <summary>Every element <paramref name="value"/> (a constant of the component's kind).</summary>
    public M Fill(CoopType type, double value) => new(this, _m.ConstantComposite(type.Id, C(type.Component, value).Id), type);

    /// <summary>A matrix loaded from a workgroup array of vectors: its first row (column, columnMajor) at vector
    /// <paramref name="index"/>, the next <paramref name="stride"/> vectors on (both in the array's vectors, whatever the
    /// matrix's components).</summary>
    public M Load(CoopType type, VArr array, V index, V stride, bool columnMajor = false)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, array.VectorType), array.Variable, index.Id);
        return new M(this, _m.Value(SpirvOp.CooperativeMatrixLoadKHR, type.Id, pointer, Const(K.U32, columnMajor ? 1UL : 0), stride.Id), type);
    }

    /// <summary>A matrix loaded from a workgroup array of scalars (its first row, or column, at element <paramref name="index"/>,
    /// the next <paramref name="stride"/> elements on: both in the array's elements, whatever the matrix's components).</summary>
    public M Load(CoopType type, Arr array, V index, V stride, bool columnMajor = false)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(array.Storage, Type(array.Kind)), array.Variable, index.Id);
        return new M(this, _m.Value(SpirvOp.CooperativeMatrixLoadKHR, type.Id, pointer, Const(K.U32, columnMajor ? 1UL : 0), stride.Id), type);
    }

    /// <summary>Stores a matrix into a workgroup array of scalars (rows, or columns, <paramref name="stride"/> elements apart).</summary>
    public void Store(M matrix, Arr array, V index, V stride, bool columnMajor = false)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(array.Storage, Type(array.Kind)), array.Variable, index.Id);
        _m.Code(SpirvOp.CooperativeMatrixStoreKHR, pointer, matrix.Id, Const(K.U32, columnMajor ? 1UL : 0), stride.Id);
    }

    /// <summary>A matrix loaded from memory by address: values of <paramref name="element"/> (the matrix's own component, or
    /// words of them), its first row (column) at <paramref name="address"/>, the next <paramref name="stride"/> elements on;
    /// <paramref name="align"/> the bytes every row's start is aligned to.</summary>
    public M Load(CoopType type, K element, V address, V stride, bool columnMajor, int align)
    {
        uint pointer = ArrayPointer(element, address);
        return new M(this, _m.Value(SpirvOp.CooperativeMatrixLoadKHR, type.Id, pointer, Const(K.U32, columnMajor ? 1UL : 0), stride.Id, 0x2, (uint)align), type);
    }

    /// <summary>Stores a matrix into memory by address as values of <paramref name="element"/> (rows, or columns for
    /// columnMajor, <paramref name="stride"/> elements apart from <paramref name="address"/>).</summary>
    public void Store(M matrix, K element, V address, V stride, bool columnMajor, int align)
    {
        uint pointer = ArrayPointer(element, address);
        _m.Code(SpirvOp.CooperativeMatrixStoreKHR, pointer, matrix.Id, Const(K.U32, columnMajor ? 1UL : 0), stride.Id, 0x2, (uint)align);
    }

    /// <summary>Stores a matrix into a workgroup array of vectors (rows, or columns, <paramref name="stride"/> vectors apart).</summary>
    public void Store(M matrix, VArr array, V index, V stride, bool columnMajor = false)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, array.VectorType), array.Variable, index.Id);
        _m.Code(SpirvOp.CooperativeMatrixStoreKHR, pointer, matrix.Id, Const(K.U32, columnMajor ? 1UL : 0), stride.Id);
    }

    // A pointer into memory by address as the matrix units' loads and stores take it: into an array of the element.
    private uint ArrayPointer(K element, V address)
    {
        if (address.Kind != K.U64) throw new InvalidOperationException($"{_name}: an address is a u64, not {address.Kind}");
        Narrow(element.Width());
        uint elementType = Type(element);
        uint block = _m.TypeBufferArray(elementType, element.Bytes());
        uint pointer = _m.Value(SpirvOp.ConvertUToPtr, _m.TypePointer(StorageClass.PhysicalStorageBuffer, block), address.Id);
        return _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.PhysicalStorageBuffer, elementType), pointer, Const(K.S32, 0), Const(K.U32, 0));
    }

    /// <summary>a·b + c by the matrix units (the integer components' signedness from their kinds).</summary>
    public M MulAdd(M a, M b, M c)
    {
        uint operands = (a.Type.Component.IsSigned() ? 0x1u : 0) | (b.Type.Component.IsSigned() ? 0x2u : 0) | (c.Type.Component.IsSigned() ? 0x4u | 0x8u : 0);
        return new M(this, _m.Value(SpirvOp.CooperativeMatrixMulAddKHR, c.Type.Id, a.Id, b.Id, c.Id, operands), c.Type);
    }

    /// <summary>The matrix's elements as <paramref name="to"/>'s component (a conversion of each, as Convert's rules).</summary>
    public M Convert(M m, CoopType to)
    {
        K from = m.Type.Component, k = to.Component;
        if (from == k) return m;
        SpirvOp op = from.IsFloat() && k.IsFloat() ? SpirvOp.FConvert
            : !from.IsFloat() && k.IsFloat() ? (from.IsSigned() ? SpirvOp.ConvertSToF : SpirvOp.ConvertUToF)
            : from.IsFloat() ? (k.IsSigned() ? SpirvOp.ConvertFToS : SpirvOp.ConvertFToU)
            : from.Width() == k.Width() ? SpirvOp.Bitcast : from.IsSigned() ? SpirvOp.SConvert : SpirvOp.UConvert;
        return new M(this, _m.Value(op, to.Id, m.Id), to);
    }

    /// <summary>Each element times <paramref name="scalar"/> (of the component's kind).</summary>
    public M Scale(M m, V scalar)
    {
        if (scalar.Kind != m.Type.Component) throw new InvalidOperationException($"{_name}: a {m.Type.Component} matrix scaled by a {scalar.Kind}");
        var v = new M(this, _m.Value(SpirvOp.MatrixTimesScalar, m.Type.Id, m.Id, scalar.Id), m.Type);
        _m.Decorate(v.Id, Decoration.NoContraction);
        return v;
    }

    /// <summary>a · b element by element (float components).</summary>
    public M Mul(M a, M b)
    {
        var v = new M(this, _m.Value(SpirvOp.FMul, a.Type.Id, a.Id, b.Id), a.Type);
        _m.Decorate(v.Id, Decoration.NoContraction);
        return v;
    }

    /// <summary>a + b element by element.</summary>
    public M Add(M a, M b)
    {
        bool f = a.Type.Component.IsFloat();
        var v = new M(this, _m.Value(f ? SpirvOp.FAdd : SpirvOp.IAdd, a.Type.Id, a.Id, b.Id), a.Type);
        if (f) _m.Decorate(v.Id, Decoration.NoContraction);
        return v;
    }

    /// <summary>A variable holding a matrix, starting at <paramref name="initial"/>.</summary>
    public MVar Local(M initial)
    {
        var v = new MVar(this, _m.LocalVariable(_m.TypePointer(StorageClass.Function, initial.Type.Id)), initial.Type);
        v.M = initial;
        return v;
    }

    internal M LoadMatrix(uint pointer, CoopType type) => new(this, _m.Value(SpirvOp.Load, type.Id, pointer), type);

    internal void StoreMatrix(uint pointer, M value) => _m.Code(SpirvOp.Store, pointer, value.Id);

    // ---------------------------------------------------------------- the packed 8-bit dot product

    /// <summary>The dot product of the four bytes of a and of b (u32 words; each byte signed or not as said), as an s32 (or a
    /// u32 when both are unsigned), by the device's packed dot product instruction.</summary>
    public V Dot4x8(V a, V b, bool signedA, bool signedB)
    {
        _m.Capability(Cap.DotProduct);
        _m.Capability(Cap.DotProductInput4x8BitPacked);
        _m.Extension("SPV_KHR_integer_dot_product");
        var op = signedA && signedB ? SpirvOp.SDot : !signedA && !signedB ? SpirvOp.UDot : SpirvOp.SUDot;
        var (x, y) = signedA || !signedB ? (a, b) : (b, a);                   // SUDot: the signed operand first
        var k = signedA || signedB ? K.S32 : K.U32;
        return new V(this, _m.Value(op, Type(k), Bits(x).Id, Bits(y).Id, 0), k);
    }

    // ---------------------------------------------------------------- workgroup arrays of vectors

    /// <summary>A workgroup array of <paramref name="length"/> vectors of <paramref name="count"/> (2 to 4) values of <paramref name="kind"/>.</summary>
    public VArr SharedVectors(K kind, int count, int length)
    {
        uint vector = _m.TypeVector(Type(kind), (uint)count);
        uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Workgroup, _m.TypeArray(vector, (uint)length)), StorageClass.Workgroup);
        _sharedBytes += kind.Bytes() * count * length;
        return new VArr(this, variable, vector, kind, count, length);
    }

    internal V[] VectorLoad(VArr array, V index)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, array.VectorType), array.Variable, index.Id);
        uint vector = _m.Value(SpirvOp.Load, array.VectorType, pointer);
        var values = new V[array.Count];
        for (int i = 0; i < array.Count; i++) values[i] = new V(this, _m.Value(SpirvOp.CompositeExtract, Type(array.Kind), vector, (uint)i), array.Kind);
        return values;
    }

    internal void VectorStore(VArr array, V index, V[] values)
    {
        if (values.Length != array.Count) throw new InvalidOperationException($"{_name}: {values.Length} values for vectors of {array.Count}");
        foreach (var v in values) if (v.Kind != array.Kind) throw new InvalidOperationException($"{_name}: a {v.Kind} into vectors of {array.Kind}");
        uint vector = _m.Value(SpirvOp.CompositeConstruct, array.VectorType, [.. values.Select(v => v.Id)]);
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(StorageClass.Workgroup, array.VectorType), array.Variable, index.Id);
        _m.Code(SpirvOp.Store, pointer, vector);
    }
}
