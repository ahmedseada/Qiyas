// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A compute kernel written in C# that reads like the code it generates, over the lanes (Abstraction/Formats.cs) as
// SPIR-V types: values are SSA results of a kind (bool, s32, u32, s64, u64, f32, f64) with the kind's arithmetic as
// operators; memory is reached by device address (a u64, PhysicalStorageBuffer pointers made at each access, aligned
// to the access's size); the kernel's arguments are 64-bit push-constant slots (addresses and values alike, as CUDA's).
//
// Every float addition, subtraction and multiplication is rounded on its own (NoContraction: a driver never fuses it
// into an FMA the kernel did not write), and division and square root are correctly rounded (the device's are not
// required to be: a quotient or root refined by one FMA step), so a kernel's arithmetic is the CPU's and CUDA's,
// operation for operation. The float controls the device grants (round to nearest even, denormals and ±0/∞/NaN kept)
// are asked for on every kernel.
namespace Adad.Vulkan;

/// <summary>The kind of a value: a lane's register type, or a condition; the 16- and 8-bit kinds for values of those widths
/// (float16 arithmetic, narrow loads and stores, the matrix units' elements; bfloat16 for conversions and matrices only).</summary>
internal enum K : byte
{
    Bool,
    U32,
    S32,
    U64,
    S64,
    F32,
    F64,
    F16,
    BF16,
    U16,
    S16,
    U8,
    S8,
}

internal static class Kinds
{
    public static bool IsFloat(this K k) => k is K.F32 or K.F64 or K.F16 or K.BF16;

    public static bool IsSigned(this K k) => k is K.S32 or K.S64 or K.S16 or K.S8;

    public static bool Wide(this K k) => k is K.U64 or K.S64 or K.F64;

    /// <summary>The kind's width in bits (a condition's as a 32-bit word's).</summary>
    public static int Width(this K k) => k switch { K.U64 or K.S64 or K.F64 => 64, K.F16 or K.BF16 or K.U16 or K.S16 => 16, K.U8 or K.S8 => 8, _ => 32 };

    public static int Bytes(this K k) => k.Width() / 8;

    /// <summary>The integer kind of this width and signedness.</summary>
    public static K Integer(int width, bool signed) => width switch
    {
        64 => signed ? K.S64 : K.U64,
        16 => signed ? K.S16 : K.U16,
        8 => signed ? K.S8 : K.U8,
        _ => signed ? K.S32 : K.U32,
    };

    /// <summary>The kind of a lane.</summary>
    public static K Of(Abstraction.Lane lane) => lane.Name switch
    {
        "f32" => K.F32,
        "f64" => K.F64,
        "s32" => K.S32,
        "u32" => K.U32,
        "s64" => K.S64,
        "u64" => K.U64,
        _ => throw new NotSupportedException(lane.Name),
    };

    /// <summary>The unsigned kind of the same width (a value's bits).</summary>
    public static K BitsKind(this K k) => Integer(k.Width(), false);
}

/// <summary>A generated kernel: its SPIR-V words, workgroup size, argument slots and workgroup memory.</summary>
internal sealed record SpvKernel(string Name, uint[] Words, int LocalX, int LocalY, int Slots, int SharedBytes);

/// <summary>A value of a kernel: valid in the block that made it and the blocks it dominates (a Var carries one past an If or a loop).</summary>
internal readonly struct V(Spv b, uint id, K kind)
{
    public readonly Spv B = b;
    public readonly uint Id = id;
    public readonly K Kind = kind;

    public static V operator +(V a, V b) => a.B.Arith(a, b, SpirvOp.IAdd, SpirvOp.FAdd);
    public static V operator -(V a, V b) => a.B.Arith(a, b, SpirvOp.ISub, SpirvOp.FSub);
    public static V operator *(V a, V b) => a.B.Arith(a, b, SpirvOp.IMul, SpirvOp.FMul);
    public static V operator /(V a, V b) => a.B.Quotient(a, b);
    public static V operator %(V a, V b) => a.B.Op2(a, b, a.Kind.IsSigned() ? SpirvOp.SRem : a.Kind.IsFloat() ? SpirvOp.FRem : SpirvOp.UMod);
    public static V operator &(V a, V b) => a.Kind == K.Bool ? a.B.Op2(a, b, SpirvOp.LogicalAnd) : a.B.Op2(a, b, SpirvOp.BitwiseAnd);
    public static V operator |(V a, V b) => a.Kind == K.Bool ? a.B.Op2(a, b, SpirvOp.LogicalOr) : a.B.Op2(a, b, SpirvOp.BitwiseOr);
    public static V operator ^(V a, V b) => a.B.Op2(a, b, SpirvOp.BitwiseXor);
    public static V operator -(V a) => a.B.Op1(a, a.Kind.IsFloat() ? SpirvOp.FNegate : SpirvOp.SNegate);
    public static V operator ~(V a) => a.B.Op1(a, SpirvOp.Not);
    public static V operator !(V a) => a.B.Op1(a, SpirvOp.LogicalNot);
    public static V operator <<(V a, int s) => a.B.Shift(a, a.B.Same(a, s), SpirvOp.ShiftLeftLogical);
    public static V operator >>(V a, int s) => a.B.Shift(a, a.B.Same(a, s), a.Kind.IsSigned() ? SpirvOp.ShiftRightArithmetic : SpirvOp.ShiftRightLogical);
    public V Shl(V s) => B.Shift(this, s, SpirvOp.ShiftLeftLogical);
    public V Shr(V s) => B.Shift(this, s, Kind.IsSigned() ? SpirvOp.ShiftRightArithmetic : SpirvOp.ShiftRightLogical);

    public static V operator <(V a, V b) => a.B.Compare(a, b, SpirvOp.SLessThan, SpirvOp.ULessThan, SpirvOp.FOrdLessThan);
    public static V operator >(V a, V b) => a.B.Compare(a, b, SpirvOp.SGreaterThan, SpirvOp.UGreaterThan, SpirvOp.FOrdGreaterThan);
    public static V operator <=(V a, V b) => a.B.Compare(a, b, SpirvOp.SLessThanEqual, SpirvOp.ULessThanEqual, SpirvOp.FOrdLessThanEqual);
    public static V operator >=(V a, V b) => a.B.Compare(a, b, SpirvOp.SGreaterThanEqual, SpirvOp.UGreaterThanEqual, SpirvOp.FOrdGreaterThanEqual);
    public static V operator ==(V a, V b) => a.Kind == K.Bool ? a.B.Op2(a, b, SpirvOp.LogicalEqual, K.Bool) : a.B.Compare(a, b, SpirvOp.IEqual, SpirvOp.IEqual, SpirvOp.FOrdEqual);
    // Not equal as C# has it: true when either is NaN.
    public static V operator !=(V a, V b) => a.Kind == K.Bool ? a.B.Op2(a, b, SpirvOp.LogicalNotEqual, K.Bool) : a.B.Compare(a, b, SpirvOp.INotEqual, SpirvOp.INotEqual, SpirvOp.FUnordNotEqual);

    public static V operator +(V a, double c) => a + a.B.Same(a, c);
    public static V operator -(V a, double c) => a - a.B.Same(a, c);
    public static V operator *(V a, double c) => a * a.B.Same(a, c);
    public static V operator /(V a, double c) => a / a.B.Same(a, c);
    public static V operator %(V a, double c) => a % a.B.Same(a, c);
    public static V operator &(V a, ulong c) => a & a.B.Same(a, c);
    public static V operator |(V a, ulong c) => a | a.B.Same(a, c);
    public static V operator ^(V a, ulong c) => a ^ a.B.Same(a, c);
    public static V operator <(V a, double c) => a < a.B.Same(a, c);
    public static V operator >(V a, double c) => a > a.B.Same(a, c);
    public static V operator <=(V a, double c) => a <= a.B.Same(a, c);
    public static V operator >=(V a, double c) => a >= a.B.Same(a, c);
    public static V operator ==(V a, double c) => a == a.B.Same(a, c);
    public static V operator !=(V a, double c) => a != a.B.Same(a, c);

    public override bool Equals(object? obj) => obj is V v && v.Id == Id;

    public override int GetHashCode() => (int)Id;

    /// <summary>The value as kind <paramref name="to"/> (Spv.Convert).</summary>
    public V To(K to) => B.Convert(this, to);
}

/// <summary>A function variable: a value that lives across Ifs and loop iterations.</summary>
internal readonly struct Var(Spv b, uint pointer, K kind)
{
    public readonly K Kind = kind;

    public V V
    {
        get => b.LoadVar(pointer, Kind);
        set => b.StoreVar(pointer, value);
    }
}

/// <summary>A workgroup (or function) array of one kind.</summary>
internal readonly struct Arr(Spv b, uint variable, K kind, int length, StorageClass storage)
{
    public readonly int Length = length;
    public readonly K Kind = kind;
    internal readonly uint Variable = variable;
    internal readonly StorageClass Space = storage;

    public V this[V index]
    {
        get => b.ArrayLoad(Variable, Space, Kind, index);
        set => b.ArrayStore(Variable, Space, index, value);
    }

    public V this[int index]
    {
        get => this[b.U(index)];
        set => this[b.U(index)] = value;
    }
}

/// <summary>What the device grants a kernel's float arithmetic (the float controls it reports) and the widths it may declare
/// (f64, f16, 8- and 16-bit integers, 8- and 16-bit loads and stores of memory, bfloat16), set by the device once.</summary>
internal readonly record struct FloatControls(bool Rte32, bool Rte64, bool Denorm32, bool Denorm64, bool Preserve32, bool Preserve64, bool Float64, bool Int64,
    bool Float16 = false, bool Denorm16 = false, bool Preserve16 = false, bool Int8 = false, bool Int16 = false, bool Storage8 = false, bool Storage16 = false,
    bool Bfloat16 = false, bool AtomicAdd32 = false, bool AtomicAdd64 = false);

internal sealed partial class Spv
{
    private readonly SpirvModule _m = new();
    private readonly string _name;
    private readonly int _localX, _localY;
    private readonly FloatControls _controls;
    private readonly uint _pushVariable;
    private readonly uint _pushStruct;
    private readonly int _slots;
    private readonly Dictionary<BuiltIn, uint> _builtIns = [];
    private readonly Dictionary<(BuiltIn, int), V> _builtInValues = [];
    private int _sharedBytes;
    private bool _usesF64, _usesF16;

    /// <summary>The push-constant slots every kernel declares (its arguments are the first ones it reads).</summary>
    public const int MaxSlots = 16;

    public Spv(string name, int localX, FloatControls controls, int localY = 1, int slots = MaxSlots)
    {
        _name = name;
        (_localX, _localY) = (localX, localY);
        _controls = controls;
        _slots = slots;
        _m.Name(_m.Function, name);
        uint u64 = Type(K.U64);
        var members = new uint[slots];
        Array.Fill(members, u64);
        _pushStruct = _m.TypeStruct(members);
        _m.Decorate(_pushStruct, Decoration.Block);
        for (int i = 0; i < slots; i++) _m.MemberDecorate(_pushStruct, (uint)i, Decoration.Offset, (uint)(8 * i));
        _pushVariable = _m.GlobalVariable(_m.TypePointer(StorageClass.PushConstant, _pushStruct), StorageClass.PushConstant);
    }

    public string Name => _name;

    public int LocalX => _localX;

    public int LocalY => _localY;

    // ---------------------------------------------------------------- types

    internal uint Type(K k) => k switch
    {
        K.Bool => _m.TypeBool(),
        K.U32 => _m.TypeInt(32, false),
        K.S32 => _m.TypeInt(32, true),
        K.U64 => _m.TypeInt(64, false),
        K.S64 => _m.TypeInt(64, true),
        K.F32 => _m.TypeFloat(32),
        K.F64 => F64Type(),
        K.F16 => Granted(_controls.Float16, "f16 (shaderFloat16)", () => { _usesF16 = true; return _m.TypeFloat(16); }),
        K.BF16 => Granted(_controls.Bfloat16, "bfloat16 (VK_KHR_shader_bfloat16)", _m.TypeBfloat16),
        K.U16 or K.S16 => Granted(_controls.Int16, "16-bit integers (shaderInt16)", () => _m.TypeInt(16, k == K.S16)),
        _ => Granted(_controls.Int8, "8-bit integers (shaderInt8)", () => _m.TypeInt(8, k == K.S8)),
    };

    private uint Granted(bool granted, string what, Func<uint> type) =>
        granted ? type() : throw new NotSupportedException($"{_name}: the device has no {what}");

    private uint F64Type()
    {
        if (!_controls.Float64) throw new NotSupportedException($"{_name}: the device has no f64 lane (shaderFloat64)");
        _usesF64 = true;
        return _m.TypeFloat(64);
    }

    // ---------------------------------------------------------------- arguments

    /// <summary>Argument slot <paramref name="slot"/> as kind <paramref name="kind"/>: an address or a 64-bit value as it is,
    /// a 32-bit value from its low bits, a float from its bits (Vulkan.F, Vulkan.D put them there).</summary>
    public V Arg(int slot, K kind = K.U64)
    {
        if (slot >= _slots) throw new InvalidOperationException($"{_name}: argument {slot} past the kernel's {_slots} slots");
        uint u64 = Type(K.U64);
        uint pointer = _m.PrologueValue(SpirvOp.AccessChain, _m.TypePointer(StorageClass.PushConstant, u64), _pushVariable, _m.Constant(Type(K.S32), (uint)slot));
        var raw = new V(this, _m.PrologueValue(SpirvOp.Load, u64, pointer), K.U64);
        return kind switch
        {
            K.U64 => raw,
            K.S64 or K.F64 => new V(this, _m.PrologueValue(SpirvOp.Bitcast, Type(kind), raw.Id), kind),
            K.U32 => new V(this, _m.PrologueValue(SpirvOp.UConvert, Type(K.U32), raw.Id), K.U32),
            K.S32 or K.F32 => new V(this, _m.PrologueValue(SpirvOp.Bitcast, Type(kind), _m.PrologueValue(SpirvOp.UConvert, Type(K.U32), raw.Id)), kind),
            _ => throw new NotSupportedException(kind.ToString()),
        };
    }

    // ---------------------------------------------------------------- memory by address

    /// <summary>The address of element <paramref name="index"/> of <paramref name="bytes"/> bytes from <paramref name="address"/>.</summary>
    public V At(V address, V index, int bytes)
    {
        var i = index.Kind == K.U64 ? index : Convert(index, index.Kind.IsSigned() ? K.S64 : K.U64).AsBits();
        return address + (bytes == 1 ? i : i * (ulong)bytes);
    }

    public V At(V address, V index, int bytes, long offset) => At(address, index, bytes) + (ulong)offset;

    /// <summary>A value of <paramref name="kind"/> at <paramref name="address"/> (aligned to its size).</summary>
    public V Load(K kind, V address, int align = 0)
    {
        uint pointer = Pointer(kind, address);
        return new V(this, _m.Value(SpirvOp.Load, Type(kind), pointer, 0x2, (uint)(align > 0 ? align : kind.Bytes())), kind);
    }

    public void Store(V address, V value, int align = 0)
    {
        uint pointer = Pointer(value.Kind, address);
        _m.Code(SpirvOp.Store, pointer, value.Id, 0x2, (uint)(align > 0 ? align : value.Kind.Bytes()));
    }

    /// <summary>A u32 word at index <paramref name="index"/> of the words at <paramref name="address"/>.</summary>
    public V Word(V address, V index) => Load(K.U32, At(address, index, 4));

    public void PutWord(V address, V index, V word) => Store(At(address, index, 4), word);

    /// <summary><paramref name="count"/> values of <paramref name="kind"/> at <paramref name="address"/> in one access (a vector
    /// of them, the address aligned to their whole size: 8 or 16 bytes), each as a value.</summary>
    public V[] LoadN(K kind, int count, V address)
    {
        uint vector = _m.TypeVector(Type(kind), (uint)count);
        uint loaded = _m.Value(SpirvOp.Load, vector, VectorPointer(vector, address), 0x2, (uint)(count * kind.Bytes()));
        var values = new V[count];
        for (int i = 0; i < count; i++) values[i] = new V(this, _m.Value(SpirvOp.CompositeExtract, Type(kind), loaded, (uint)i), kind);
        return values;
    }

    /// <summary>The values (of one kind) at <paramref name="address"/> in one access, as LoadN reads them.</summary>
    public void StoreN(V address, params V[] values)
    {
        foreach (var v in values) Check(values[0], v);
        uint vector = _m.TypeVector(Type(values[0].Kind), (uint)values.Length);
        uint built = _m.Value(SpirvOp.CompositeConstruct, vector, [.. values.Select(v => v.Id)]);
        _m.Code(SpirvOp.Store, VectorPointer(vector, address), built, 0x2, (uint)(values.Length * values[0].Kind.Bytes()));
    }

    private uint VectorPointer(uint vector, V address)
    {
        if (address.Kind != K.U64) throw new InvalidOperationException($"{_name}: an address is a u64, not {address.Kind}");
        return _m.Value(SpirvOp.ConvertUToPtr, _m.TypePointer(StorageClass.PhysicalStorageBuffer, vector), address.Id);
    }

    private uint Pointer(K kind, V address)
    {
        if (address.Kind != K.U64) throw new InvalidOperationException($"{_name}: an address is a u64, not {address.Kind}");
        Narrow(kind.Width());
        uint type = _m.TypePointer(StorageClass.PhysicalStorageBuffer, Type(kind));
        return _m.Value(SpirvOp.ConvertUToPtr, type, address.Id);
    }

    // A load or store of memory of 8 or 16 bits: the device's storage capability for it.
    private void Narrow(int width)
    {
        if (width == 8)
        {
            if (!_controls.Storage8) throw new NotSupportedException($"{_name}: the device has no 8-bit loads and stores (storageBuffer8BitAccess)");
            _m.Capability(Cap.StorageBuffer8BitAccess);
        }
        else if (width == 16)
        {
            if (!_controls.Storage16) throw new NotSupportedException($"{_name}: the device has no 16-bit loads and stores (storageBuffer16BitAccess)");
            _m.Capability(Cap.StorageBuffer16BitAccess);
        }
    }

    /// <summary>Atomically adds <paramref name="value"/> (u32 or s32) to the word at <paramref name="address"/>; returns the old value.</summary>
    public V AtomicAdd(V address, V value)
    {
        uint pointer = Pointer(value.Kind, address);
        return new V(this, _m.Value(SpirvOp.AtomicIAdd, Type(value.Kind), pointer, Const(K.U32, 1), Const(K.U32, 0), value.Id), value.Kind);
    }

    /// <summary>Atomically replaces the word at <paramref name="address"/> by <paramref name="value"/> when it equals
    /// <paramref name="comparator"/>; returns the old value.</summary>
    public V AtomicCompareExchange(V address, V value, V comparator)
    {
        uint pointer = Pointer(value.Kind, address);
        return new V(this, _m.Value(SpirvOp.AtomicCompareExchange, Type(value.Kind), pointer, Const(K.U32, 1), Const(K.U32, 0), Const(K.U32, 0), value.Id, comparator.Id), value.Kind);
    }

    /// <summary>Atomically adds the float <paramref name="value"/> (f32 or f64, where the device has the float atomic add:
    /// Vulkan.AtomicAdd32, AtomicAdd64) to the value at <paramref name="address"/>; returns the old value.</summary>
    public V AtomicFAdd(V address, V value)
    {
        bool wide = value.Kind == K.F64;
        if (value.Kind is not (K.F32 or K.F64) || !(wide ? _controls.AtomicAdd64 : _controls.AtomicAdd32))
            throw new NotSupportedException($"{_name}: the device has no {value.Kind} atomic add (VK_EXT_shader_atomic_float)");
        _m.Capability(wide ? Cap.AtomicFloat64Add : Cap.AtomicFloat32Add);
        _m.Extension("SPV_EXT_shader_atomic_float_add");
        uint pointer = Pointer(value.Kind, address);
        return new V(this, _m.Value(SpirvOp.AtomicFAddEXT, Type(value.Kind), pointer, Const(K.U32, 1), Const(K.U32, 0), value.Id), value.Kind);
    }

    // ---------------------------------------------------------------- built-ins (u32)

    public V GlobalX => BuiltInValue(BuiltIn.GlobalInvocationId, 0);
    public V GlobalY => BuiltInValue(BuiltIn.GlobalInvocationId, 1);
    public V LocalIdX => BuiltInValue(BuiltIn.LocalInvocationId, 0);
    public V LocalIdY => BuiltInValue(BuiltIn.LocalInvocationId, 1);
    public V GroupX => BuiltInValue(BuiltIn.WorkgroupId, 0);
    public V GroupY => BuiltInValue(BuiltIn.WorkgroupId, 1);
    public V GroupZ => BuiltInValue(BuiltIn.WorkgroupId, 2);
    public V GroupsX => BuiltInValue(BuiltIn.NumWorkgroups, 0);
    public V GroupsY => BuiltInValue(BuiltIn.NumWorkgroups, 1);

    /// <summary>Invocations of the whole dispatch along x (a grid-stride loop's step).</summary>
    public V StrideX => GroupsX * _localX;

    private V BuiltInValue(BuiltIn builtIn, int component)
    {
        if (_builtInValues.TryGetValue((builtIn, component), out var cached)) return cached;
        uint u32 = Type(K.U32), uint3 = _m.TypeVector(u32, 3);
        if (!_builtIns.TryGetValue(builtIn, out uint loaded))
        {
            uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Input, uint3), StorageClass.Input);
            _m.Decorate(variable, Decoration.BuiltIn, (uint)builtIn);
            loaded = _m.PrologueValue(SpirvOp.Load, uint3, variable);
            _builtIns[builtIn] = loaded;
        }

        var value = new V(this, _m.PrologueValue(SpirvOp.CompositeExtract, u32, loaded, (uint)component), K.U32);
        _builtInValues[(builtIn, component)] = value;
        return value;
    }

    // ---------------------------------------------------------------- constants

    // A constant's bits as SPIR-V takes them: a kind narrower than 32 bits in the low bits of its word, a signed one's sign
    // extended, the others' high bits 0.
    internal uint Const(K k, ulong bits) => k switch
    {
        K.Bool => _m.ConstantBool(bits != 0),
        _ when k.Wide() => _m.Constant(Type(k), bits, 2),
        _ when k.Width() < 32 => _m.Constant(Type(k), k.IsSigned() ? (uint)((long)(bits << (64 - k.Width())) >> (64 - k.Width())) : bits & ((1UL << k.Width()) - 1)),
        _ => _m.Constant(Type(k), bits & 0xFFFFFFFF),
    };

    public V U(long value) => new(this, Const(K.U32, (ulong)value), K.U32);
    public V S(long value) => new(this, Const(K.S32, (ulong)value), K.S32);
    public V UL(ulong value) => new(this, Const(K.U64, value), K.U64);
    public V SL(long value) => new(this, Const(K.S64, (ulong)value), K.S64);
    public V F(double value) => new(this, Const(K.F32, BitConverter.SingleToUInt32Bits((float)value)), K.F32);
    public V D(double value) => new(this, Const(K.F64, BitConverter.DoubleToUInt64Bits(value)), K.F64);
    public V Bool(bool value) => new(this, Const(K.Bool, value ? 1UL : 0), K.Bool);

    /// <summary>A constant of kind <paramref name="k"/> with value <paramref name="value"/> (a float's rounded to the kind).</summary>
    public V C(K k, double value) => k switch
    {
        K.F32 => F(value),
        K.F64 => D(value),
        K.F16 => new(this, Const(K.F16, BitConverter.HalfToUInt16Bits((Half)value)), K.F16),
        K.BF16 => new(this, Const(K.BF16, Bf16Bits((float)value)), K.BF16),
        K.U32 => U((long)value),
        K.S32 => S((long)value),
        K.U64 => UL((ulong)value),
        K.S64 => SL((long)value),
        K.U16 or K.U8 => new(this, Const(k, (ulong)value), k),
        K.S16 or K.S8 => new(this, Const(k, (ulong)(long)value), k),
        _ => Bool(value != 0),
    };

    // A float's nearest bfloat16 (ties to even; a NaN stays a NaN).
    private static ushort Bf16Bits(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return (ushort)((bits >> 16) | 0x40);
        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    /// <summary>The integer constant <paramref name="bits"/> as kind <paramref name="k"/> (its bits, any kind).</summary>
    public V CBits(K k, ulong bits) => new(this, Const(k, bits), k);

    internal V Same(V like, double c) => C(like.Kind, c);
    internal V Same(V like, ulong c) => like.Kind.IsFloat() ? throw new InvalidOperationException("a bit mask of a float") : CBits(like.Kind, c);
    internal V Same(V like, int c) => C(like.Kind, c);

    // ---------------------------------------------------------------- arithmetic

    internal V Arith(V a, V b, SpirvOp integer, SpirvOp floating)
    {
        Check(a, b);
        if (!a.Kind.IsFloat()) return Op2(a, b, integer);
        var v = Op2(a, b, floating);
        _m.Decorate(v.Id, Decoration.NoContraction);
        return v;
    }

    internal V Op2(V a, V b, SpirvOp op, K? result = null)
    {
        Check(a, b);
        var k = result ?? a.Kind;
        return new V(this, _m.Value(op, Type(k), a.Id, b.Id), k);
    }

    internal V Op1(V a, SpirvOp op) => new(this, _m.Value(op, Type(a.Kind), a.Id), a.Kind);

    internal V Shift(V a, V s, SpirvOp op) => new(this, _m.Value(op, Type(a.Kind), a.Id, s.Id), a.Kind);

    internal V Compare(V a, V b, SpirvOp signed, SpirvOp unsigned, SpirvOp floating)
    {
        Check(a, b);
        var op = a.Kind.IsFloat() ? floating : a.Kind.IsSigned() ? signed : unsigned;
        return new V(this, _m.Value(op, Type(K.Bool), a.Id, b.Id), K.Bool);
    }

    private void Check(V a, V b)
    {
        if (a.Kind != b.Kind) throw new InvalidOperationException($"{_name}: {a.Kind} with {b.Kind}");
    }

    /// <summary>a / b: an integer's truncating quotient, a float's correctly rounded (the device's quotient refined).</summary>
    internal V Quotient(V a, V b)
    {
        Check(a, b);
        if (!a.Kind.IsFloat()) return Op2(a, b, a.Kind.IsSigned() ? SpirvOp.SDiv : SpirvOp.UDiv);
        // q from the device; r = a − q·b exactly (an FMA); q + r·(1/b) rounded once. Where q is 0, ±∞ or NaN (b zero, an
        // overflow, a NaN) the device's quotient stands; a residual outside the normal range (q near the denormals) too.
        var q = Op2(a, b, SpirvOp.FDiv);
        var rcp = Op2(C(a.Kind, 1), b, SpirvOp.FDiv);
        var e = Fma(-b, rcp, C(a.Kind, 1));
        rcp = Fma(rcp, e, rcp);
        var r = Fma(-q, b, a);
        var refined = Fma(r, rcp, q);
        var finite = !IsNan(q) & !IsInf(q) & (q != 0.0);
        var normal = Abs(q) >= (a.Kind switch { K.F32 => 1.1754943508222875e-38, K.F16 => 6.103515625e-05, _ => 2.2250738585072014e-308 } * 4);
        return Select(finite & normal & !IsInf(rcp) & !IsNan(refined), refined, q);
    }

    /// <summary>√x correctly rounded: the device's root refined (s + r·(0.5/s), r = x − s² exactly).</summary>
    /// <summary>2^x and log2 x by the device's instructions (GLSL.std.450 Exp2, Log2: an f32's or an f16's, the precision the
    /// device gives them).</summary>
    public V Exp2(V x) => Ext(Glsl.Exp2, x);

    public V Log2(V x) => Ext(Glsl.Log2, x);

    public V Sqrt(V x)
    {
        var s = Ext(Glsl.Sqrt, x);
        var half = C(x.Kind, 0.5) / s;
        var r = Fma(-s, s, x);
        var refined = Fma(r, half, s);
        var ok = (x > 0.0) & !IsInf(x) & !IsNan(refined);
        return Select(ok, refined, s);
    }

    public V Fma(V a, V b, V c)
    {
        Check(a, b);
        Check(a, c);
        if (!a.Kind.IsFloat()) return a * b + c;
        return Ext(Glsl.Fma, a, b, c);
    }

    /// <summary>The smaller (a NaN operand gives the other, as PTX's min).</summary>
    public V Min(V a, V b) => Ext(a.Kind.IsFloat() ? Glsl.NMin : a.Kind.IsSigned() ? Glsl.SMin : Glsl.UMin, a, b);

    public V Max(V a, V b) => Ext(a.Kind.IsFloat() ? Glsl.NMax : a.Kind.IsSigned() ? Glsl.SMax : Glsl.UMax, a, b);

    public V Abs(V a) => a.Kind.IsFloat() ? Ext(Glsl.FAbs, a) : a.Kind.IsSigned() ? Ext(Glsl.SAbs, a) : a;

    public V RoundEven(V a) => Ext(Glsl.RoundEven, a);
    public V Floor(V a) => Ext(Glsl.Floor, a);
    public V Ceil(V a) => Ext(Glsl.Ceil, a);
    public V Trunc(V a) => Ext(Glsl.Trunc, a);

    /// <summary>The index of the highest set bit of a u32 (−1 for 0), as an s32.</summary>
    public V HighBit(V a) => new(this, _m.Ext(Glsl.FindUMsb, Type(K.S32), a.Id), K.S32);

    public V IsNan(V a) => new(this, _m.Value(SpirvOp.IsNan, Type(K.Bool), a.Id), K.Bool);
    public V IsInf(V a) => new(this, _m.Value(SpirvOp.IsInf, Type(K.Bool), a.Id), K.Bool);

    public V Select(V condition, V a, V b)
    {
        Check(a, b);
        return new V(this, _m.Value(SpirvOp.Select, Type(a.Kind), condition.Id, a.Id, b.Id), a.Kind);
    }

    /// <summary>bits [offset, offset + count) of a, sign-extended for a signed kind.</summary>
    public V Extract(V a, V offset, V count) =>
        new(this, _m.Value(a.Kind.IsSigned() ? SpirvOp.BitFieldSExtract : SpirvOp.BitFieldUExtract, Type(a.Kind), a.Id, offset.Id, count.Id), a.Kind);

    public V Extract(V a, int offset, int count) => Extract(a, U(offset), U(count));

    /// <summary>a with bits [offset, offset + count) replaced by the low bits of insert.</summary>
    public V Insert(V a, V insert, V offset, V count) => new(this, _m.Value(SpirvOp.BitFieldInsert, Type(a.Kind), a.Id, insert.Id, offset.Id, count.Id), a.Kind);

    public V Insert(V a, V insert, int offset, int count) => Insert(a, insert, U(offset), U(count));

    private V Ext(Glsl op, params V[] args)
    {
        foreach (var a in args) Check(args[0], a);
        return new V(this, _m.Ext(op, Type(args[0].Kind), [.. args.Select(a => a.Id)]), args[0].Kind);
    }

    // ---------------------------------------------------------------- conversions

    /// <summary>The value's bits as the unsigned kind of its width.</summary>
    public V Bits(V a) => a.Kind is K.U32 or K.U64 ? a : Bitcast(a, a.Kind.BitsKind());

    public V Bitcast(V a, K to) => a.Kind == to ? a : new V(this, _m.Value(SpirvOp.Bitcast, Type(to), a.Id), to);

    /// <summary>A value as kind <paramref name="to"/>: integers widened by their own sign and narrowed to their low bits,
    /// integers to floats rounded to nearest, floats widened exactly and narrowed rounded to nearest, floats to integers
    /// truncated (in range only: Pack holds a value to its range first).</summary>
    public V Convert(V a, K to)
    {
        if (a.Kind == to) return a;
        var from = a.Kind;
        if (from == K.Bool) return Select(a, C(to, 1), C(to, 0));
        if (to == K.Bool) return a != 0.0;
        if (from.IsFloat() && to.IsFloat()) return new V(this, _m.Value(SpirvOp.FConvert, Type(to), a.Id), to);
        if (!from.IsFloat() && to.IsFloat()) return new V(this, _m.Value(from.IsSigned() ? SpirvOp.ConvertSToF : SpirvOp.ConvertUToF, Type(to), a.Id), to);
        if (from.IsFloat()) return new V(this, _m.Value(to.IsSigned() ? SpirvOp.ConvertFToS : SpirvOp.ConvertFToU, Type(to), a.Id), to);
        // Integers: the width by the source's sign, then the signedness by the bits.
        if (from.Width() == to.Width()) return Bitcast(a, to);
        var wid = Kinds.Integer(to.Width(), from.IsSigned());
        var widened = new V(this, _m.Value(from.IsSigned() ? SpirvOp.SConvert : SpirvOp.UConvert, Type(wid), a.Id), wid);
        return Bitcast(widened, to);
    }

    // ---------------------------------------------------------------- variables and arrays

    public Var Local(K kind) => new(this, _m.LocalVariable(_m.TypePointer(StorageClass.Function, Type(kind))), kind);

    public Var Local(V initial)
    {
        var v = Local(initial.Kind);
        v.V = initial;
        return v;
    }

    internal V LoadVar(uint pointer, K kind) => new(this, _m.Value(SpirvOp.Load, Type(kind), pointer), kind);

    internal void StoreVar(uint pointer, V value) => _m.Code(SpirvOp.Store, pointer, value.Id);

    /// <summary>A workgroup array of <paramref name="length"/> values of <paramref name="kind"/>.</summary>
    public Arr Shared(K kind, int length)
    {
        uint type = _m.TypeArray(Type(kind), (uint)length);
        uint variable = _m.GlobalVariable(_m.TypePointer(StorageClass.Workgroup, type), StorageClass.Workgroup);
        _sharedBytes += kind.Bytes() * length;
        return new Arr(this, variable, kind, length, StorageClass.Workgroup);
    }

    /// <summary>An array of the invocation's own (function memory).</summary>
    public Arr Private(K kind, int length)
    {
        uint type = _m.TypeArray(Type(kind), (uint)length);
        return new Arr(this, _m.LocalVariable(_m.TypePointer(StorageClass.Function, type)), kind, length, StorageClass.Function);
    }

    internal V ArrayLoad(uint variable, StorageClass storage, K kind, V index)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(storage, Type(kind)), variable, index.Id);
        return new V(this, _m.Value(SpirvOp.Load, Type(kind), pointer), kind);
    }

    internal void ArrayStore(uint variable, StorageClass storage, V index, V value)
    {
        uint pointer = _m.Value(SpirvOp.AccessChain, _m.TypePointer(storage, Type(value.Kind)), variable, index.Id);
        _m.Code(SpirvOp.Store, pointer, value.Id);
    }

    // ---------------------------------------------------------------- control flow

    public void If(V condition, Action then, Action? otherwise = null)
    {
        uint thenLabel = _m.Id(), merge = _m.Id(), elseLabel = otherwise is null ? merge : _m.Id();
        _m.Code(SpirvOp.SelectionMerge, merge, 0);
        _m.Terminate(SpirvOp.BranchConditional, condition.Id, thenLabel, elseLabel);
        _m.Label(thenLabel);
        then();
        if (!_m.Terminated) _m.Terminate(SpirvOp.Branch, merge);
        if (otherwise is not null)
        {
            _m.Label(elseLabel);
            otherwise();
            if (!_m.Terminated) _m.Terminate(SpirvOp.Branch, merge);
        }

        _m.Label(merge);
    }

    /// <summary>for (i = start; i &lt; end; i += step) body(i), over a variable of start's kind.</summary>
    public void For(V start, V end, V step, Action<V> body)
    {
        var i = Local(start);
        Loop(() => i.V < end, () => body(i.V), () => i.V = i.V + step);
    }

    public void For(V start, V end, Action<V> body) => For(start, end, Same(start, 1), body);

    /// <summary>for (i = 0; i &lt; count; i++) body(i) unrolled at build time (count a constant).</summary>
    public static void Unroll(int count, Action<int> body)
    {
        for (int i = 0; i < count; i++) body(i);
    }

    public void While(Func<V> condition, Action body) => Loop(condition, body, static () => { });

    private void Loop(Func<V> condition, Action body, Action next)
    {
        uint header = _m.Id(), test = _m.Id(), bodyLabel = _m.Id(), continueLabel = _m.Id(), merge = _m.Id();
        _m.Terminate(SpirvOp.Branch, header);
        _m.Label(header);
        _m.Code(SpirvOp.LoopMerge, merge, continueLabel, 0);
        _m.Terminate(SpirvOp.Branch, test);
        _m.Label(test);
        var c = condition();
        _m.Terminate(SpirvOp.BranchConditional, c.Id, bodyLabel, merge);
        _m.Label(bodyLabel);
        body();
        if (!_m.Terminated) _m.Terminate(SpirvOp.Branch, continueLabel);
        _m.Label(continueLabel);
        next();
        _m.Terminate(SpirvOp.Branch, header);
        _m.Label(merge);
    }

    /// <summary>Every invocation of the workgroup waits here; their workgroup-memory writes before it are visible after it.</summary>
    public void Barrier() => _m.Code(SpirvOp.ControlBarrier, Const(K.U32, 2), Const(K.U32, 2), Const(K.U32, 0x108));

    /// <summary>Barrier, also ordering the device memory the workgroup wrote before it (read by others of it after).</summary>
    public void MemoryBarrierAll() => _m.Code(SpirvOp.ControlBarrier, Const(K.U32, 2), Const(K.U32, 1), Const(K.U32, 0x148));

    // ---------------------------------------------------------------- the module

    public SpvKernel Build()
    {
        // The float controls the device grants, on the widths the kernel computes in.
        AddControls(32, _controls.Rte32, _controls.Denorm32, _controls.Preserve32);
        if (_usesF64) AddControls(64, _controls.Rte64, _controls.Denorm64, _controls.Preserve64);
        if (_usesF16) AddControls(16, false, _controls.Denorm16, _controls.Preserve16);
        var words = _m.Finish((uint)_localX, (uint)_localY, 1);
        return new SpvKernel(_name, words, _localX, _localY, _slots, _sharedBytes);
    }

    // RoundingModeRTE is not declared even where the device grants it: NVIDIA's compiler aborts the process on a module
    // with that mode and any OpFDiv (measured on an RTX 5070 Ti, Vulkan 1.4). The device's float arithmetic rounds to
    // nearest even without it, and the checks hold every kernel's results to the CPU's bits.
    private void AddControls(uint width, bool rte, bool denorm, bool preserve)
    {
        _ = rte;
        if (denorm) { _m.Capability(Cap.DenormPreserve); _m.Mode(FloatMode.DenormPreserve, width); }
        if (preserve) { _m.Capability(Cap.SignedZeroInfNanPreserve); _m.Mode(FloatMode.SignedZeroInfNanPreserve, width); }
    }
}

internal static class VExtensions
{
    /// <summary>A signed 64-bit value as its unsigned bits (address arithmetic).</summary>
    public static V AsBits(this V v) => v.B.Bits(v);
}
