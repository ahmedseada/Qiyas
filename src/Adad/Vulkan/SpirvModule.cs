// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// One SPIR-V 1.5 compute module (Vulkan 1.2): the PhysicalStorageBuffer64 / GLSL450 memory model (memory reached by
// device addresses), the GLSL.std.450 instructions, one entry point "main" with its local size and the float controls
// the device grants, and a single function. Types and constants are made once and shared; every other instruction goes
// into its section in the order the specification requires. Integers of 8, 16, 32 and 64 bits and floats of 16, 32 and
// 64 bits: a kernel declares the widths it uses (and the capabilities follow).
using System.Text;

namespace Adad.Vulkan;

internal sealed class SpirvModule
{
    public const uint Version = 0x00010500;

    private const uint Magic = 0x07230203;

    private readonly List<uint> _annotations = [];
    private readonly List<uint> _debug = [];
    private readonly List<uint> _globals = [];
    private readonly List<uint> _variables = [];
    private readonly List<uint> _prologue = [];
    private readonly List<uint> _body = [];
    private readonly Dictionary<string, uint> _types = [];
    private readonly Dictionary<(uint Type, ulong Bits), uint> _constants = [];
    private readonly List<uint> _interface = [];
    private readonly SortedSet<uint> _capabilities = [];
    private readonly SortedSet<string> _extensions = new(StringComparer.Ordinal);
    private readonly List<(uint Mode, uint Operand)> _modes = [];
    private uint _next = 1;
    private bool _terminated;

    public SpirvModule()
    {
        GlslSet = Id();
        Function = Id();
        EntryLabel = Id();
        Capability(Cap.PhysicalStorageBufferAddresses);
        Capability(Cap.Int64);                                               // addresses are 64-bit integers
    }

    public uint GlslSet { get; }

    public uint Function { get; }

    public uint EntryLabel { get; }

    public bool Terminated => _terminated;

    public uint Id() => _next++;

    public void Capability(uint capability) => _capabilities.Add(capability);

    public void Extension(string name) => _extensions.Add(name);

    /// <summary>An execution mode of the entry point with one literal operand (the float controls: a width).</summary>
    public void Mode(uint mode, uint operand)
    {
        if (!_modes.Contains((mode, operand))) _modes.Add((mode, operand));
    }

    // ---------------------------------------------------------------- types

    public uint TypeVoid() => Type("void", () => Emit(_globals, SpirvOp.TypeVoid, Id()));

    public uint TypeBool() => Type("bool", () => Emit(_globals, SpirvOp.TypeBool, Id()));

    public uint TypeInt(int bits, bool signed) => Type($"int{bits}:{signed}", () =>
    {
        if (bits == 64) Capability(Cap.Int64);
        else if (bits == 16) Capability(Cap.Int16);
        else if (bits == 8) Capability(Cap.Int8);
        return Emit(_globals, SpirvOp.TypeInt, Id(), (uint)bits, signed ? 1u : 0u);
    });

    public uint TypeFloat(int bits) => Type($"float{bits}", () =>
    {
        if (bits == 64) Capability(Cap.Float64);
        else if (bits == 16) Capability(Cap.Float16);
        return Emit(_globals, SpirvOp.TypeFloat, Id(), (uint)bits);
    });

    public uint TypeVector(uint component, uint count) => Type($"vec:{component}:{count}", () => Emit(_globals, SpirvOp.TypeVector, Id(), component, count));

    public uint TypeArray(uint element, uint length) => Type($"array:{element}:{length}", () =>
        Emit(_globals, SpirvOp.TypeArray, Id(), element, Constant(TypeInt(32, false), length)));

    public uint TypeStruct(params uint[] members) => Emit(_globals, SpirvOp.TypeStruct, [Id(), .. members]);

    public uint TypePointer(StorageClass storage, uint type) =>
        Type($"ptr:{(uint)storage}:{type}", () => Emit(_globals, SpirvOp.TypePointer, Id(), (uint)storage, type));

    public uint TypeFunction(uint result) => Type($"fn:{result}", () => Emit(_globals, SpirvOp.TypeFunction, Id(), result));

    private uint Type(string key, Func<uint> create)
    {
        if (!_types.TryGetValue(key, out uint id))
        {
            id = create();
            _types[key] = id;
        }

        return id;
    }

    // ---------------------------------------------------------------- constants

    /// <summary>A constant of <paramref name="type"/> (<paramref name="words"/> 1 or 2) with these bits.</summary>
    public uint Constant(uint type, ulong bits, int words = 1)
    {
        if (!_constants.TryGetValue((type, bits), out uint id))
        {
            id = words == 2
                ? Emit(_globals, SpirvOp.Constant, type, Id(), (uint)bits, (uint)(bits >> 32))
                : Emit(_globals, SpirvOp.Constant, type, Id(), (uint)bits);
            _constants[(type, bits)] = id;
        }

        return id;
    }

    public uint ConstantBool(bool value) => Type(value ? "true" : "false",
        () => Emit(_globals, value ? SpirvOp.ConstantTrue : SpirvOp.ConstantFalse, TypeBool(), Id()));

    // ---------------------------------------------------------------- globals, decorations, names

    /// <summary>A module-scope variable (every one is in the entry point's interface, as SPIR-V 1.4 on requires).</summary>
    public uint GlobalVariable(uint pointerType, StorageClass storage)
    {
        uint id = Id();
        Emit(_globals, SpirvOp.Variable, pointerType, id, (uint)storage);
        _interface.Add(id);
        return id;
    }

    public void Decorate(uint target, Decoration decoration, params uint[] operands) =>
        Emit(_annotations, SpirvOp.Decorate, [target, (uint)decoration, .. operands]);

    public void MemberDecorate(uint structType, uint member, Decoration decoration, params uint[] operands) =>
        Emit(_annotations, SpirvOp.MemberDecorate, [structType, member, (uint)decoration, .. operands]);

    public void Name(uint target, string name) => Emit(_debug, SpirvOp.Name, [target, .. Text(name)]);

    // ---------------------------------------------------------------- function code

    public uint LocalVariable(uint pointerType) => Emit(_variables, SpirvOp.Variable, pointerType, Id(), (uint)StorageClass.Function);

    /// <summary>An instruction placed at the start of the first block (after the variables): its value dominates all the code.</summary>
    public uint PrologueValue(SpirvOp op, uint type, params uint[] operands)
    {
        uint id = Id();
        Emit(_prologue, op, [type, id, .. operands]);
        return id;
    }

    public void Code(SpirvOp op, params uint[] operands)
    {
        if (_terminated) throw new InvalidOperationException($"SPIR-V: {op} after the block ended.");
        Emit(_body, op, operands);
    }

    public uint Value(SpirvOp op, uint type, params uint[] operands)
    {
        uint id = Id();
        Code(op, [type, id, .. operands]);
        return id;
    }

    public uint Ext(Glsl instruction, uint type, params uint[] operands) => Value(SpirvOp.ExtInst, type, [GlslSet, (uint)instruction, .. operands]);

    public void Terminate(SpirvOp op, params uint[] operands)
    {
        Code(op, operands);
        _terminated = true;
    }

    public void Label(uint label)
    {
        if (!_terminated) throw new InvalidOperationException("SPIR-V: a block starts before the previous one ended.");
        _terminated = false;
        Emit(_body, SpirvOp.Label, label);
    }

    // ---------------------------------------------------------------- the module

    public uint[] Finish(uint localX, uint localY, uint localZ)
    {
        if (!_terminated) Terminate(SpirvOp.Return);
        uint voidType = TypeVoid();
        uint functionType = TypeFunction(voidType);
        var words = new List<uint>(_globals.Count + _body.Count + 128) { Magic, Version, 0, 0, 0 };
        Emit(words, SpirvOp.Capability, Cap.Shader);
        foreach (uint capability in _capabilities) Emit(words, SpirvOp.Capability, capability);
        foreach (string extension in _extensions) Emit(words, SpirvOp.Extension, Text(extension));
        Emit(words, SpirvOp.ExtInstImport, [GlslSet, .. Text("GLSL.std.450")]);
        Emit(words, SpirvOp.MemoryModel, 5348, 1);                           // PhysicalStorageBuffer64, GLSL450
        Emit(words, SpirvOp.EntryPoint, [5, Function, .. Text("main"), .. _interface]);   // GLCompute
        Emit(words, SpirvOp.ExecutionMode, Function, 17, localX, localY, localZ);         // LocalSize
        foreach (var (mode, operand) in _modes) Emit(words, SpirvOp.ExecutionMode, Function, mode, operand);
        words.AddRange(_debug);
        words.AddRange(_annotations);
        words.AddRange(_globals);
        Emit(words, SpirvOp.Function, voidType, Function, 0, functionType);
        Emit(words, SpirvOp.Label, EntryLabel);
        words.AddRange(_variables);
        words.AddRange(_prologue);
        words.AddRange(_body);
        Emit(words, SpirvOp.FunctionEnd);
        words[3] = _next;
        return [.. words];
    }

    private static uint Emit(List<uint> section, SpirvOp op, params uint[] operands)
    {
        section.Add((uint)(operands.Length + 1) << 16 | (uint)op);
        section.AddRange(operands);
        return operands.Length switch
        {
            0 => 0,
            _ when op is SpirvOp.Constant or SpirvOp.ConstantComposite or SpirvOp.ConstantTrue or SpirvOp.ConstantFalse or SpirvOp.Variable => operands[1],
            _ => operands[0],
        };
    }

    private static uint[] Text(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        var words = new uint[bytes.Length / 4 + 1];
        for (int i = 0; i < bytes.Length; i++) words[i / 4] |= (uint)bytes[i] << (8 * (i % 4));
        return words;
    }
}

/// <summary>The capabilities a module may declare.</summary>
internal static class Cap
{
    public const uint Shader = 1, Float16 = 9, Float64 = 10, Int64 = 11, Int64Atomics = 12, Int16 = 22, Int8 = 39,
        GroupNonUniform = 61, GroupNonUniformVote = 62, GroupNonUniformArithmetic = 63, GroupNonUniformBallot = 64, GroupNonUniformShuffle = 65,
        DenormPreserve = 4464, SignedZeroInfNanPreserve = 4466, RoundingModeRte = 4467, PhysicalStorageBufferAddresses = 5347;
}

/// <summary>The execution modes of the float controls (each with a width operand).</summary>
internal static class FloatMode
{
    public const uint DenormPreserve = 4459, SignedZeroInfNanPreserve = 4461, RoundingModeRte = 4462;
}

internal enum SpirvOp : ushort
{
    Name = 5,
    Extension = 10,
    ExtInstImport = 11,
    ExtInst = 12,
    MemoryModel = 14,
    EntryPoint = 15,
    ExecutionMode = 16,
    Capability = 17,
    TypeVoid = 19,
    TypeBool = 20,
    TypeInt = 21,
    TypeFloat = 22,
    TypeVector = 23,
    TypeArray = 28,
    TypeRuntimeArray = 29,
    TypeStruct = 30,
    TypePointer = 32,
    TypeFunction = 33,
    ConstantTrue = 41,
    ConstantFalse = 42,
    Constant = 43,
    ConstantComposite = 44,
    Function = 54,
    FunctionEnd = 56,
    Variable = 59,
    Load = 61,
    Store = 62,
    AccessChain = 65,
    Decorate = 71,
    MemberDecorate = 72,
    CompositeConstruct = 80,
    CompositeExtract = 81,
    ConvertFToU = 109,
    ConvertFToS = 110,
    ConvertSToF = 111,
    ConvertUToF = 112,
    UConvert = 113,
    SConvert = 114,
    FConvert = 115,
    ConvertUToPtr = 120,
    Bitcast = 124,
    SNegate = 126,
    FNegate = 127,
    IAdd = 128,
    FAdd = 129,
    ISub = 130,
    FSub = 131,
    IMul = 132,
    FMul = 133,
    UDiv = 134,
    SDiv = 135,
    FDiv = 136,
    UMod = 137,
    SRem = 138,
    SMod = 139,
    FRem = 140,
    UMulExtended = 151,
    SMulExtended = 152,
    IsNan = 156,
    IsInf = 157,
    LogicalEqual = 164,
    LogicalNotEqual = 165,
    LogicalOr = 166,
    LogicalAnd = 167,
    LogicalNot = 168,
    Select = 169,
    IEqual = 170,
    INotEqual = 171,
    UGreaterThan = 172,
    SGreaterThan = 173,
    UGreaterThanEqual = 174,
    SGreaterThanEqual = 175,
    ULessThan = 176,
    SLessThan = 177,
    ULessThanEqual = 178,
    SLessThanEqual = 179,
    FOrdEqual = 180,
    FUnordEqual = 181,
    FOrdNotEqual = 182,
    FUnordNotEqual = 183,
    FOrdLessThan = 184,
    FUnordLessThan = 185,
    FOrdGreaterThan = 186,
    FUnordGreaterThan = 187,
    FOrdLessThanEqual = 188,
    FUnordLessThanEqual = 189,
    FOrdGreaterThanEqual = 190,
    FUnordGreaterThanEqual = 191,
    ShiftRightLogical = 194,
    ShiftRightArithmetic = 195,
    ShiftLeftLogical = 196,
    BitwiseOr = 197,
    BitwiseXor = 198,
    BitwiseAnd = 199,
    Not = 200,
    BitFieldInsert = 201,
    BitFieldSExtract = 202,
    BitFieldUExtract = 203,
    BitReverse = 204,
    BitCount = 205,
    ControlBarrier = 224,
    MemoryBarrier = 225,
    AtomicLoad = 227,
    AtomicStore = 228,
    AtomicExchange = 229,
    AtomicCompareExchange = 230,
    AtomicIAdd = 234,
    AtomicUMax = 240,
    AtomicOr = 242,
    LoopMerge = 246,
    SelectionMerge = 247,
    Label = 248,
    Branch = 249,
    BranchConditional = 250,
    Return = 253,
    GroupNonUniformBroadcast = 337,
    GroupNonUniformShuffle = 345,
    GroupNonUniformShuffleXor = 346,
    GroupNonUniformShuffleDown = 348,
}

internal enum StorageClass : uint
{
    Input = 1,
    Workgroup = 4,
    Function = 7,
    PushConstant = 9,
    PhysicalStorageBuffer = 5349,
}

internal enum Decoration : uint
{
    Block = 2,
    ArrayStride = 6,
    BuiltIn = 11,
    NoContraction = 42,
    Offset = 35,
    Restrict = 19,
    Aliased = 20,
    AliasedPointer = 5356,
}

internal enum BuiltIn : uint
{
    NumWorkgroups = 24,
    WorkgroupId = 26,
    LocalInvocationId = 27,
    GlobalInvocationId = 28,
    LocalInvocationIndex = 29,
    SubgroupSize = 36,
    SubgroupLocalInvocationId = 41,
}

internal enum Glsl : uint
{
    RoundEven = 2,
    Trunc = 3,
    FAbs = 4,
    SAbs = 5,
    Floor = 8,
    Ceil = 9,
    Fract = 10,
    Sqrt = 31,
    InverseSqrt = 32,
    FMin = 37,
    UMin = 38,
    SMin = 39,
    FMax = 40,
    UMax = 41,
    SMax = 42,
    FClamp = 43,
    UClamp = 44,
    SClamp = 45,
    Fma = 50,
    Ldexp = 53,
    FindILsb = 73,
    FindSMsb = 74,
    FindUMsb = 75,
    NMin = 79,
    NMax = 80,
}
