// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// One Vulkan device: its facts as it reports them (the features and float controls the kernels are built against, the
// subgroup and workgroup limits), its kernels (SPIR-V made once per key, a compute pipeline each, one layout for all:
// 16 push-constant slots of 64 bits, no descriptors), its memory (device-local buffers by device address, a free-list
// by size so a buffer of a size seen before costs no driver call, one staging buffer for the host's copies), and its
// queue: dispatches recorded in order into command buffers, each followed by a barrier (every write before it visible
// to every read and write after it, as one CUDA stream), handed to the queue in batches and waited for on Synchronize.
//
// A device refuses what it has not: no bufferDeviceAddress or shaderInt64 and there is no Vulkan device for Adad; no
// shaderFloat64 and every kernel in the f64 lane is refused (Spv throws), never run in another lane.
using System.Runtime.InteropServices;
using System.Text;

namespace Adad.Vulkan;

public sealed unsafe class Vulkan : IDisposable
{
    private readonly IntPtr _instance, _physical, _device, _queue;
    private readonly uint _family;
    private readonly ulong _layout, _pool;
    private readonly VkPhysicalDeviceMemoryProperties _memory;
    private readonly Dictionary<string, ulong> _pipelines = [];
    private readonly List<ulong> _modules = [];
    private readonly Dictionary<long, Stack<ulong>> _free = [];
    private readonly SortedList<ulong, Allocation> _allocations = [];
    private readonly delegate* unmanaged<IntPtr, VkBufferDeviceAddressInfo*, ulong> _address;

    private const int Ring = 4, BatchCommands = 256;
    private readonly IntPtr[] _commands = new IntPtr[Ring];
    private readonly ulong[] _fences = new ulong[Ring];
    private readonly bool[] _pending = new bool[Ring];
    private int _current = -1, _recorded;

    private Allocation _staging;

    private readonly struct Allocation(ulong buffer, ulong memory, ulong address, long size, void* mapped)
    {
        public readonly ulong Buffer = buffer, Memory = memory, Address = address;
        public readonly long Size = size;
        public readonly void* Mapped = mapped;
    }

    /// <summary>Whether this machine has a Vulkan device Adad can use (the reason when not).</summary>
    public static bool TryAvailable(out string reason)
    {
        if (!VkApi.TryLoad(out reason)) return false;
        try
        {
            using var probe = new Vulkan();
            return true;
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or DllNotFoundException or EntryPointNotFoundException)
        {
            reason = e.Message;
            return false;
        }
    }

    public Vulkan()
    {
        if (!VkApi.TryLoad(out string why)) throw new NotSupportedException(why);
        _instance = CreateInstance();
        (_physical, _family) = Choose(_instance);

        VkPhysicalDeviceProperties properties;
        VkApi.vkGetPhysicalDeviceProperties(_physical, &properties);
        Name = Marshal.PtrToStringUTF8((IntPtr)properties.DeviceName) ?? "vulkan";
        ApiVersion = $"{properties.ApiVersion >> 22}.{(properties.ApiVersion >> 12) & 0x3FF}";
        MaxSharedBytes = (int)properties.MaxComputeSharedMemorySize;
        MaxWorkgroup = (int)properties.MaxComputeWorkGroupInvocations;
        MaxGroupsX = properties.MaxComputeWorkGroupCountX;
        MaxGroupsY = properties.MaxComputeWorkGroupCountY;
        if (properties.MaxPushConstantsSize < Spv.MaxSlots * 8) throw new NotSupportedException($"{Name}: {properties.MaxPushConstantsSize} bytes of push constants (Adad's kernels take {Spv.MaxSlots * 8})");
        Discrete = properties.DeviceType == VkApi.DeviceTypeDiscreteGpu;

        // The extensions whose features the kernels may use: the matrix units (cooperative matrices), the packed 8-bit
        // integer dot product, bfloat16 elements of the matrix units.
        var extensions = DeviceExtensions(_physical);
        bool hasCoop = extensions.Contains("VK_KHR_cooperative_matrix"), hasDot = extensions.Contains("VK_KHR_shader_integer_dot_product"),
            hasBf16 = extensions.Contains("VK_KHR_shader_bfloat16"), hasAtomic = extensions.Contains("VK_EXT_shader_atomic_float");

        var subgroup = new VkPhysicalDeviceSubgroupProperties { SType = VkApi.StructurePhysicalDeviceSubgroupProperties };
        var controls = new VkPhysicalDeviceFloatControlsProperties { SType = VkApi.StructurePhysicalDeviceFloatControlsProperties, PNext = &subgroup };
        var dot = new VkIntegerDotProductProperties { SType = VkApi.StructurePhysicalDeviceIntegerDotProductProperties };
        if (hasDot) subgroup.PNext = &dot;
        var properties2 = new VkPhysicalDeviceProperties2 { SType = VkApi.StructurePhysicalDeviceProperties2, PNext = &controls };
        VkApi.vkGetPhysicalDeviceProperties2(_physical, &properties2);
        SubgroupSize = (int)subgroup.SubgroupSize;

        var features12 = new VkPhysicalDeviceVulkan12Features { SType = VkApi.StructurePhysicalDeviceVulkan12Features };
        var features11 = new VkPhysicalDeviceVulkan11Features { SType = VkApi.StructurePhysicalDeviceVulkan11Features, PNext = &features12 };
        var coopFeatures = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceCooperativeMatrixFeatures };
        var dotFeatures = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceIntegerDotProductFeatures };
        var bf16Features = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceBfloat16Features };
        var atomicFeatures = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceAtomicFloatFeatures };
        void* chain = null;
        if (hasAtomic) { atomicFeatures.PNext = chain; chain = &atomicFeatures; }
        if (hasCoop) { coopFeatures.PNext = chain; chain = &coopFeatures; }
        if (hasDot) { dotFeatures.PNext = chain; chain = &dotFeatures; }
        if (hasBf16) { bf16Features.PNext = chain; chain = &bf16Features; }
        features12.PNext = chain;
        var features = new VkPhysicalDeviceFeatures2 { SType = VkApi.StructurePhysicalDeviceFeatures2, PNext = &features11 };
        VkApi.vkGetPhysicalDeviceFeatures2(_physical, &features);
        bool f64 = features.Features[VkPhysicalDeviceFeatures2.ShaderFloat64] != 0, i64 = features.Features[VkPhysicalDeviceFeatures2.ShaderInt64] != 0;
        bool i16 = features.Features[VkPhysicalDeviceFeatures2.ShaderInt16] != 0, i8 = features12.Features[VkPhysicalDeviceVulkan12Features.ShaderInt8] != 0;
        bool f16 = features12.Features[VkPhysicalDeviceVulkan12Features.ShaderFloat16] != 0;
        bool store8 = features12.Features[VkPhysicalDeviceVulkan12Features.StorageBuffer8BitAccess] != 0;
        bool store16 = features11.Features[VkPhysicalDeviceVulkan11Features.StorageBuffer16BitAccess] != 0;
        bool coop = hasCoop && coopFeatures.Features[0] != 0, dots = hasDot && dotFeatures.Features[0] != 0;
        bool bf16 = hasBf16 && bf16Features.Features[0] != 0, bf16Coop = bf16 && bf16Features.Features[2] != 0;
        bool add32 = hasAtomic && atomicFeatures.Features[1] != 0, add64 = hasAtomic && f64 && atomicFeatures.Features[3] != 0;
        if (features12.Features[VkPhysicalDeviceVulkan12Features.BufferDeviceAddress] == 0) throw new NotSupportedException($"{Name}: no bufferDeviceAddress (Adad's kernels reach memory by address)");
        if (!i64) throw new NotSupportedException($"{Name}: no shaderInt64 (addresses are 64-bit integers)");
        Controls = new FloatControls(controls.RoundingModeRteFloat32 != 0, controls.RoundingModeRteFloat64 != 0, controls.DenormPreserveFloat32 != 0,
            controls.DenormPreserveFloat64 != 0, controls.SignedZeroInfNanPreserveFloat32 != 0, controls.SignedZeroInfNanPreserveFloat64 != 0, f64, i64,
            f16, f16 && controls.DenormPreserveFloat16 != 0, f16 && controls.SignedZeroInfNanPreserveFloat16 != 0, i8, i16, store8 && i8, store16, bf16, add32, add64);
        Int16 = i16;
        Int8 = i8;
        Float16 = f16;
        Storage8 = store8 && i8;
        Storage16 = store16;
        IntDot8Signed = dots && dot.Accelerated[VkIntegerDotProductProperties.Packed4x8Signed] != 0;
        IntDot8Unsigned = dots && dot.Accelerated[VkIntegerDotProductProperties.Packed4x8Unsigned] != 0;
        CooperativeMatrices = coop ? CooperativeKinds(_instance, _physical, f16, i8, bf16Coop) : [];

        // The device: one compute queue; the features the kernels use, enabled only where reported.
        var enable12 = new VkPhysicalDeviceVulkan12Features { SType = VkApi.StructurePhysicalDeviceVulkan12Features };
        enable12.Features[VkPhysicalDeviceVulkan12Features.BufferDeviceAddress] = 1;
        if (i8) enable12.Features[VkPhysicalDeviceVulkan12Features.ShaderInt8] = 1;
        if (f16) enable12.Features[VkPhysicalDeviceVulkan12Features.ShaderFloat16] = 1;
        if (Storage8) enable12.Features[VkPhysicalDeviceVulkan12Features.StorageBuffer8BitAccess] = 1;
        var enable11 = new VkPhysicalDeviceVulkan11Features { SType = VkApi.StructurePhysicalDeviceVulkan11Features, PNext = &enable12 };
        if (store16) enable11.Features[VkPhysicalDeviceVulkan11Features.StorageBuffer16BitAccess] = 1;
        var enableCoop = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceCooperativeMatrixFeatures };
        var enableDot = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceIntegerDotProductFeatures };
        var enableBf16 = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceBfloat16Features };
        var enableAtomic = new VkFeatureFlags { SType = VkApi.StructurePhysicalDeviceAtomicFloatFeatures };
        var names = new List<string>();
        void* enabled = null;
        if (coop) { enableCoop.Features[0] = 1; enableCoop.PNext = enabled; enabled = &enableCoop; names.Add("VK_KHR_cooperative_matrix"); }
        if (dots) { enableDot.Features[0] = 1; enableDot.PNext = enabled; enabled = &enableDot; names.Add("VK_KHR_shader_integer_dot_product"); }
        if (bf16) { enableBf16.Features[0] = 1; enableBf16.Features[2] = bf16Coop ? 1u : 0; enableBf16.PNext = enabled; enabled = &enableBf16; names.Add("VK_KHR_shader_bfloat16"); }
        if (add32 || add64)
        {
            enableAtomic.Features[0] = enableAtomic.Features[1] = add32 ? 1u : 0;
            enableAtomic.Features[2] = enableAtomic.Features[3] = add64 ? 1u : 0;
            enableAtomic.PNext = enabled;
            enabled = &enableAtomic;
            names.Add("VK_EXT_shader_atomic_float");
        }

        enable12.PNext = enabled;
        var enable = new VkPhysicalDeviceFeatures2 { SType = VkApi.StructurePhysicalDeviceFeatures2, PNext = &enable11 };
        enable.Features[VkPhysicalDeviceFeatures2.ShaderInt64] = 1;
        if (f64) enable.Features[VkPhysicalDeviceFeatures2.ShaderFloat64] = 1;
        if (i16) enable.Features[VkPhysicalDeviceFeatures2.ShaderInt16] = 1;
        float priority = 1;
        var queueInfo = new VkDeviceQueueCreateInfo { SType = VkApi.StructureDeviceQueueCreateInfo, QueueFamilyIndex = _family, QueueCount = 1, QueuePriorities = &priority };
        var nameBytes = names.Select(n => Marshal.StringToCoTaskMemUTF8(n)).ToArray();
        try
        {
            fixed (IntPtr* namePointers = nameBytes)
            {
                var deviceInfo = new VkDeviceCreateInfo
                {
                    SType = VkApi.StructureDeviceCreateInfo, PNext = &enable, QueueCreateInfoCount = 1, QueueCreateInfos = &queueInfo,
                    EnabledExtensionCount = (uint)nameBytes.Length, EnabledExtensionNames = (byte**)namePointers,
                };
                VkApi.Check(VkApi.vkCreateDevice(_physical, &deviceInfo, null, out _device), "vkCreateDevice");
            }
        }
        finally
        {
            foreach (var p in nameBytes) Marshal.FreeCoTaskMem(p);
        }

        Bfloat16 = bf16;
        VkApi.vkGetDeviceQueue(_device, _family, 0, out _queue);

        fixed (byte* name = "vkGetBufferDeviceAddress\0"u8)
        {
            var proc = VkApi.vkGetDeviceProcAddr(_device, name);
            if (proc == IntPtr.Zero) throw new NotSupportedException($"{Name}: vkGetBufferDeviceAddress is missing");
            _address = (delegate* unmanaged<IntPtr, VkBufferDeviceAddressInfo*, ulong>)proc;
        }

        VkPhysicalDeviceMemoryProperties memory;
        VkApi.vkGetPhysicalDeviceMemoryProperties(_physical, &memory);
        _memory = memory;

        var range = new VkPushConstantRange { StageFlags = VkApi.ShaderStageCompute, Offset = 0, Size = Spv.MaxSlots * 8 };
        var layoutInfo = new VkPipelineLayoutCreateInfo { SType = VkApi.StructurePipelineLayoutCreateInfo, PushConstantRangeCount = 1, PushConstantRanges = &range };
        VkApi.Check(VkApi.vkCreatePipelineLayout(_device, &layoutInfo, null, out _layout), "vkCreatePipelineLayout");

        var poolInfo = new VkCommandPoolCreateInfo { SType = VkApi.StructureCommandPoolCreateInfo, Flags = VkApi.CommandPoolResetCommandBuffer, QueueFamilyIndex = _family };
        VkApi.Check(VkApi.vkCreateCommandPool(_device, &poolInfo, null, out _pool), "vkCreateCommandPool");
        var allocInfo = new VkCommandBufferAllocateInfo { SType = VkApi.StructureCommandBufferAllocateInfo, CommandPool = _pool, Level = VkApi.CommandBufferLevelPrimary, CommandBufferCount = Ring };
        fixed (IntPtr* buffers = _commands) VkApi.Check(VkApi.vkAllocateCommandBuffers(_device, &allocInfo, buffers), "vkAllocateCommandBuffers");
        for (int i = 0; i < Ring; i++)
        {
            var fenceInfo = new VkFenceCreateInfo { SType = VkApi.StructureFenceCreateInfo };
            VkApi.Check(VkApi.vkCreateFence(_device, &fenceInfo, null, out _fences[i]), "vkCreateFence");
        }

        _staging = Allocate(1 << 22, host: true);
    }

    public string Name { get; }

    public string ApiVersion { get; }

    public bool Discrete { get; }

    public int SubgroupSize { get; }

    public int MaxSharedBytes { get; }

    public int MaxWorkgroup { get; }

    public uint MaxGroupsX { get; }

    public uint MaxGroupsY { get; }

    /// <summary>What the kernels' float arithmetic is built with (the device's float controls and lanes).</summary>
    internal FloatControls Controls { get; }

    public bool Float64 => Controls.Float64;

    public bool Int16 { get; }

    public bool Int8 { get; }

    /// <summary>float16 arithmetic in kernels (shaderFloat16).</summary>
    public bool Float16 { get; }

    /// <summary>bfloat16 values in kernels (VK_KHR_shader_bfloat16's shaderBFloat16Type).</summary>
    public bool Bfloat16 { get; private set; }

    /// <summary>8-bit and 16-bit loads and stores of device memory (storageBuffer8BitAccess, storageBuffer16BitAccess).</summary>
    public bool Storage8 { get; }

    public bool Storage16 { get; }

    /// <summary>The packed 4 × 8-bit integer dot product, accelerated by the device (VK_KHR_shader_integer_dot_product), for
    /// signed and for unsigned bytes.</summary>
    public bool IntDot8Signed { get; }

    public bool IntDot8Unsigned { get; }

    /// <summary>The atomic add of an f32 and of an f64 in memory (VK_EXT_shader_atomic_float).</summary>
    public bool AtomicAdd32 => Controls.AtomicAdd32;

    public bool AtomicAdd64 => Controls.AtomicAdd64;

    /// <summary>The matrix units' shapes as the device reports them (VK_KHR_cooperative_matrix, a subgroup's matrices), the
    /// kinds whose element types its kernels can declare; none when it has no matrix units.</summary>
    public IReadOnlyList<CooperativeMatrix> CooperativeMatrices { get; }

    /// <summary>The first shape with these element types (A and B, the accumulator C and the result), or null.</summary>
    public CooperativeMatrix? Cooperative(string ab, string c) => CooperativeMatrices.FirstOrDefault(m => m.A == ab && m.B == ab && m.C == c && m.Result == c) is { M: > 0 } k ? k : null;

    /// <summary>The device's facts in one line.</summary>
    public string Facts =>
        $"{Name} (Vulkan {ApiVersion}{(Discrete ? ", discrete" : "")}), subgroup {SubgroupSize}, workgroup up to {MaxWorkgroup}, shared {MaxSharedBytes / 1024} KB, " +
        $"f64 {(Float64 ? "yes" : "no")}, f16 {(Float16 ? "yes" : "no")}, rounding the driver's (RTE not declared: see Spv.AddControls), denormals f32 {Yes(Controls.Denorm32)}, inf/nan f32 {Yes(Controls.Preserve32)}, " +
        $"8-bit dot {(IntDot8Signed ? "s8" : "")}{(IntDot8Signed && IntDot8Unsigned ? "/" : "")}{(IntDot8Unsigned ? "u8" : "")}{(IntDot8Signed || IntDot8Unsigned ? "" : "no")}, " +
        $"matrix units {(CooperativeMatrices.Count == 0 ? "none" : string.Join(" ", CooperativeMatrices.Select(m => m.ToString())))}";

    private static string Yes(bool b) => b ? "kept" : "no";

    private static HashSet<string> DeviceExtensions(IntPtr physical)
    {
        uint count = 0;
        VkApi.Check(VkApi.vkEnumerateDeviceExtensionProperties(physical, null, ref count, null), "vkEnumerateDeviceExtensionProperties");
        var list = new VkExtensionProperties[count];
        fixed (VkExtensionProperties* p = list) VkApi.Check(VkApi.vkEnumerateDeviceExtensionProperties(physical, null, ref count, p), "vkEnumerateDeviceExtensionProperties");
        var names = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < count; i++)
        {
            fixed (byte* name = list[i].ExtensionName) names.Add(Marshal.PtrToStringUTF8((IntPtr)name) ?? "");
        }

        return names;
    }

    // The matrix units' subgroup shapes whose element types a kernel can declare here (16-bit floats with shaderFloat16, bfloat16
    // with its cooperative matrices, 8-bit integers with shaderInt8), as vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR lists them.
    private static CooperativeMatrix[] CooperativeKinds(IntPtr instance, IntPtr physical, bool f16, bool i8, bool bf16)
    {
        IntPtr proc;
        fixed (byte* name = "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0"u8) proc = VkApi.vkGetInstanceProcAddr(instance, name);
        if (proc == IntPtr.Zero) return [];
        var get = (delegate* unmanaged<IntPtr, uint*, VkCooperativeMatrixProperties*, int>)proc;
        uint count = 0;
        if (get(physical, &count, null) != VkApi.Success || count == 0) return [];
        var list = new VkCooperativeMatrixProperties[count];
        for (int i = 0; i < count; i++) list[i].SType = VkApi.StructureCooperativeMatrixProperties;
        fixed (VkCooperativeMatrixProperties* p = list) VkApi.Check(get(physical, &count, p), "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        bool Declarable(string t) => t switch { "f16" => f16, "bf16" => bf16 && f16, "s8" or "u8" => i8, "f32" or "s32" or "u32" => true, _ => false };
        var kinds = new List<CooperativeMatrix>();
        foreach (var p in list.AsSpan(0, (int)count))
        {
            if (p.Scope != VkApi.ScopeSubgroup) continue;
            var k = new CooperativeMatrix((int)p.MSize, (int)p.NSize, (int)p.KSize, Component(p.AType), Component(p.BType), Component(p.CType), Component(p.ResultType), p.SaturatingAccumulation != 0);
            if (Declarable(k.A) && Declarable(k.B) && Declarable(k.C) && Declarable(k.Result) && !kinds.Contains(k)) kinds.Add(k);
        }

        return [.. kinds];
    }

    private static string Component(uint type) => type switch
    {
        VkApi.ComponentFloat16 => "f16",
        VkApi.ComponentFloat32 => "f32",
        VkApi.ComponentFloat64 => "f64",
        VkApi.ComponentSint8 => "s8",
        VkApi.ComponentSint16 => "s16",
        VkApi.ComponentSint32 => "s32",
        VkApi.ComponentSint64 => "s64",
        VkApi.ComponentUint8 => "u8",
        VkApi.ComponentUint16 => "u16",
        VkApi.ComponentUint32 => "u32",
        VkApi.ComponentUint64 => "u64",
        VkApi.ComponentBfloat16 => "bf16",
        VkApi.ComponentFloat8E4M3 => "e4m3",
        VkApi.ComponentFloat8E5M2 => "e5m2",
        _ => $"type{type}",
    };

    private static IntPtr CreateInstance()
    {
        uint version = VkApi.MakeVersion(1, 0);
        try { _ = VkApi.vkEnumerateInstanceVersion(out version); } catch (EntryPointNotFoundException) { }
        if (version < VkApi.MakeVersion(1, 2)) throw new NotSupportedException($"the Vulkan loader is {version >> 22}.{(version >> 12) & 0x3FF}; Adad needs 1.2");
        fixed (byte* name = "Adad\0"u8)
        {
            var app = new VkApplicationInfo { SType = VkApi.StructureApplicationInfo, ApplicationName = name, EngineName = name, ApiVersion = VkApi.MakeVersion(1, 2) };
            var info = new VkInstanceCreateInfo { SType = VkApi.StructureInstanceCreateInfo, ApplicationInfo = &app };
            VkApi.Check(VkApi.vkCreateInstance(&info, null, out var instance), "vkCreateInstance");
            return instance;
        }
    }

    // The device: ADAD_VULKAN_DEVICE's index when set, else the first discrete GPU of Vulkan 1.2 with a compute queue,
    // else the first such device.
    private static (IntPtr Device, uint Family) Choose(IntPtr instance)
    {
        uint count = 0;
        VkApi.Check(VkApi.vkEnumeratePhysicalDevices(instance, ref count, null), "vkEnumeratePhysicalDevices");
        if (count == 0) throw new NotSupportedException("the Vulkan loader reports no device");
        var devices = new IntPtr[count];
        fixed (IntPtr* p = devices) VkApi.Check(VkApi.vkEnumeratePhysicalDevices(instance, ref count, p), "vkEnumeratePhysicalDevices");
        var candidates = new List<(IntPtr Device, uint Family, bool Discrete)>();
        foreach (var d in devices)
        {
            VkPhysicalDeviceProperties props;
            VkApi.vkGetPhysicalDeviceProperties(d, &props);
            if (props.ApiVersion < VkApi.MakeVersion(1, 2) || props.DeviceType == VkApi.DeviceTypeCpu) continue;
            uint families = 0;
            VkApi.vkGetPhysicalDeviceQueueFamilyProperties(d, ref families, null);
            var list = new VkQueueFamilyProperties[families];
            fixed (VkQueueFamilyProperties* f = list) VkApi.vkGetPhysicalDeviceQueueFamilyProperties(d, ref families, f);
            for (uint i = 0; i < families; i++)
            {
                if ((list[i].QueueFlags & VkApi.QueueCompute) == 0) continue;
                candidates.Add((d, i, props.DeviceType == VkApi.DeviceTypeDiscreteGpu));
                break;
            }
        }

        if (candidates.Count == 0) throw new NotSupportedException("no Vulkan 1.2 GPU with a compute queue");
        if (Environment.GetEnvironmentVariable("ADAD_VULKAN_DEVICE") is { Length: > 0 } chosen && int.TryParse(chosen, out int at) && at >= 0 && at < candidates.Count)
            return (candidates[at].Device, candidates[at].Family);
        var pick = candidates.FirstOrDefault(c => c.Discrete);
        return pick.Device != IntPtr.Zero ? (pick.Device, pick.Family) : (candidates[0].Device, candidates[0].Family);
    }

    // ---------------------------------------------------------------- kernels

    /// <summary>A kernel's SPIR-V, made on demand.</summary>
    internal delegate SpvKernel Maker();

    private readonly Dictionary<string, (ulong Pipeline, SpvKernel Kernel)> _byKey = [];

    /// <summary>The pipeline for <paramref name="key"/>: its kernel made and compiled the first time, then looked up.</summary>
    internal (ulong Pipeline, SpvKernel Kernel) Cached(string key, Maker make)
    {
        if (_byKey.TryGetValue(key, out var known)) return known;
        var kernel = make();
        if (kernel.SharedBytes > MaxSharedBytes) throw new NotSupportedException($"{kernel.Name}: {kernel.SharedBytes} bytes of workgroup memory (the device has {MaxSharedBytes})");
        if (Environment.GetEnvironmentVariable("ADAD_SPIRV_DUMP") is { Length: > 0 } dir)
        {
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, kernel.Name + ".spv"), MemoryMarshal.AsBytes(kernel.Words.AsSpan()).ToArray());
        }

        var entry = (Pipeline(kernel), kernel);
        _byKey[key] = entry;
        return entry;
    }

    private ulong Pipeline(SpvKernel kernel)
    {
        ulong module;
        fixed (uint* code = kernel.Words)
        {
            var info = new VkShaderModuleCreateInfo { SType = VkApi.StructureShaderModuleCreateInfo, CodeSize = (nuint)(kernel.Words.Length * 4), Code = code };
            VkApi.Check(VkApi.vkCreateShaderModule(_device, &info, null, out module), $"vkCreateShaderModule {kernel.Name}");
        }

        _modules.Add(module);
        ulong pipeline;
        fixed (byte* main = "main\0"u8)
        {
            var stage = new VkPipelineShaderStageCreateInfo { SType = VkApi.StructurePipelineShaderStageCreateInfo, Stage = VkApi.ShaderStageCompute, Module = module, Name = main };
            var info = new VkComputePipelineCreateInfo { SType = VkApi.StructureComputePipelineCreateInfo, Stage = stage, Layout = _layout, BasePipelineIndex = -1 };
            VkApi.Check(VkApi.vkCreateComputePipelines(_device, 0, 1, &info, null, &pipeline), $"vkCreateComputePipelines {kernel.Name}");
        }

        return pipeline;
    }

    /// <summary>Records a dispatch of <paramref name="kernel"/>: groupsX × groupsY workgroups, the arguments in its slots.</summary>
    internal void Launch((ulong Pipeline, SpvKernel Kernel) kernel, uint groupsX, uint groupsY, params ReadOnlySpan<ulong> args)
    {
        if (args.Length > kernel.Kernel.Slots) throw new InvalidOperationException($"{kernel.Kernel.Name}: {args.Length} arguments for {kernel.Kernel.Slots} slots");
        if (groupsX == 0 || groupsY == 0) return;
        if (groupsX > MaxGroupsX || groupsY > MaxGroupsY) throw new NotSupportedException($"{kernel.Kernel.Name}: {groupsX} × {groupsY} workgroups (the device takes {MaxGroupsX} × {MaxGroupsY})");
        var cb = Recording();
        VkApi.vkCmdBindPipeline(cb, VkApi.PipelineBindPointCompute, kernel.Pipeline);
        var slots = stackalloc ulong[Spv.MaxSlots];
        for (int i = 0; i < args.Length; i++) slots[i] = args[i];
        VkApi.vkCmdPushConstants(cb, _layout, VkApi.ShaderStageCompute, 0, (uint)(args.Length * 8), slots);
        VkApi.vkCmdDispatch(cb, groupsX, groupsY, 1);
        Barrier(cb);
        if (++_recorded >= BatchCommands) Flush();
    }

    public static ulong U(int value) => (uint)value;

    public static ulong U(long value) => (ulong)value;

    public static ulong F(float value) => BitConverter.SingleToUInt32Bits(value);

    public static ulong D(double value) => BitConverter.DoubleToUInt64Bits(value);

    // ---------------------------------------------------------------- the queue

    private IntPtr Recording()
    {
        if (_current >= 0) return _commands[_current];
        // The next command buffer of the ring, once the work it last held is done.
        int next = (_last + 1) % Ring;
        if (_pending[next]) Wait(next);
        var cb = _commands[next];
        VkApi.Check(VkApi.vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
        var begin = new VkCommandBufferBeginInfo { SType = VkApi.StructureCommandBufferBeginInfo, Flags = VkApi.CommandBufferOneTimeSubmit };
        VkApi.Check(VkApi.vkBeginCommandBuffer(cb, &begin), "vkBeginCommandBuffer");
        _current = next;
        _last = next;
        _recorded = 0;
        return cb;
    }

    private int _last = Ring - 1;

    // Every write of the commands before it visible to every read and write of the commands after it (submission order
    // carries it across command buffers and batches).
    private static void Barrier(IntPtr cb)
    {
        var barrier = new VkMemoryBarrier
        {
            SType = VkApi.StructureMemoryBarrier,
            SourceAccessMask = VkApi.AccessShaderWrite | VkApi.AccessTransferWrite,
            DestinationAccessMask = VkApi.AccessShaderRead | VkApi.AccessShaderWrite | VkApi.AccessTransferRead | VkApi.AccessTransferWrite,
        };
        VkApi.vkCmdPipelineBarrier(cb, VkApi.StageComputeShader | VkApi.StageTransfer, VkApi.StageComputeShader | VkApi.StageTransfer, 0, 1, &barrier, 0, null, 0, null);
    }

    /// <summary>Hands the commands recorded so far to the queue without waiting for them.</summary>
    public void Flush()
    {
        if (_current < 0) return;
        var cb = _commands[_current];
        VkApi.Check(VkApi.vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        var submit = new VkSubmitInfo { SType = VkApi.StructureSubmitInfo, CommandBufferCount = 1, CommandBuffers = &cb };
        VkApi.Check(VkApi.vkQueueSubmit(_queue, 1, &submit, _fences[_current]), "vkQueueSubmit");
        _pending[_current] = true;
        _current = -1;
    }

    /// <summary>Waits for everything recorded so far.</summary>
    public void Synchronize()
    {
        Flush();
        for (int i = 0; i < Ring; i++)
        {
            if (_pending[i]) Wait(i);
        }
    }

    private void Wait(int i)
    {
        ulong fence = _fences[i];
        VkApi.Check(VkApi.vkWaitForFences(_device, 1, &fence, 1, ulong.MaxValue), "vkWaitForFences");
        VkApi.Check(VkApi.vkResetFences(_device, 1, &fence), "vkResetFences");
        _pending[i] = false;
    }

    // ---------------------------------------------------------------- memory

    /// <summary>Device memory of at least <paramref name="bytes"/>, by its device address (from the free-list when a buffer of that size was returned).</summary>
    public ulong Rent(long bytes)
    {
        long size = Math.Max(256, (bytes + 255) / 256 * 256);
        if (_free.TryGetValue(size, out var stack) && stack.Count > 0) return stack.Pop();
        var a = Allocate(size, host: false);
        _allocations[a.Address] = a;
        return a.Address;
    }

    public void Return(ulong address)
    {
        var a = _allocations[address];
        if (!_free.TryGetValue(a.Size, out var stack)) _free[a.Size] = stack = new Stack<ulong>();
        stack.Push(address);
    }

    /// <summary>A rented buffer given back to the driver instead of kept for a later rent of its size (a buffer only a passing
    /// measurement wants, as the bench's limits: copies of up to gigabytes). Waits for the work recorded so far.</summary>
    public void Release(ulong address)
    {
        Synchronize();
        var a = _allocations[address];
        _allocations.Remove(address);
        Free(a);
    }

    private Allocation Allocate(long size, bool host)
    {
        uint usage = VkApi.BufferStorageBuffer | VkApi.BufferTransferSource | VkApi.BufferTransferDestination | (host ? 0 : VkApi.BufferDeviceAddress);
        var info = new VkBufferCreateInfo { SType = VkApi.StructureBufferCreateInfo, Size = (ulong)size, Usage = usage };
        VkApi.Check(VkApi.vkCreateBuffer(_device, &info, null, out ulong buffer), "vkCreateBuffer");
        VkMemoryRequirements req;
        VkApi.vkGetBufferMemoryRequirements(_device, buffer, &req);
        uint want = host ? VkApi.MemoryHostVisible | VkApi.MemoryHostCoherent : VkApi.MemoryDeviceLocal;
        int type = MemoryType(req.MemoryTypeBits, want, host ? VkApi.MemoryHostCached : 0);
        var flags = new VkMemoryAllocateFlagsInfo { SType = VkApi.StructureMemoryAllocateFlagsInfo, Flags = VkApi.MemoryAllocateDeviceAddress };
        var alloc = new VkMemoryAllocateInfo { SType = VkApi.StructureMemoryAllocateInfo, PNext = host ? null : &flags, AllocationSize = req.Size, MemoryTypeIndex = (uint)type };
        int result = VkApi.vkAllocateMemory(_device, &alloc, null, out ulong memory);
        if (result != VkApi.Success)
        {
            VkApi.vkDestroyBuffer(_device, buffer, null);
            VkApi.Check(result, $"vkAllocateMemory {size} bytes");
        }

        VkApi.Check(VkApi.vkBindBufferMemory(_device, buffer, memory, 0), "vkBindBufferMemory");
        void* mapped = null;
        if (host) VkApi.Check(VkApi.vkMapMemory(_device, memory, 0, ulong.MaxValue, 0, &mapped), "vkMapMemory");
        ulong address = 0;
        if (!host)
        {
            var addressInfo = new VkBufferDeviceAddressInfo { SType = VkApi.StructureBufferDeviceAddressInfo, Buffer = buffer };
            address = _address(_device, &addressInfo);
        }

        return new Allocation(buffer, memory, address, size, mapped);
    }

    private int MemoryType(uint bits, uint want, uint prefer)
    {
        int found = -1;
        for (int i = 0; i < _memory.MemoryTypeCount; i++)
        {
            if ((bits & (1u << i)) == 0 || (_memory.TypeFlags(i) & want) != want) continue;
            if ((_memory.TypeFlags(i) & prefer) == prefer) return i;
            if (found < 0) found = i;
        }

        if (found < 0) throw new NotSupportedException($"{Name}: no memory type with flags {want:X}");
        return found;
    }

    // The allocation holding a device address, and the offset of the address in it.
    private (Allocation A, ulong Offset) Find(ulong address)
    {
        var keys = _allocations.Keys;
        int lo = 0, hi = keys.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (keys[mid] <= address) lo = mid + 1;
            else hi = mid - 1;
        }

        if (hi < 0) throw new ArgumentException($"no device buffer holds address {address:X}");
        var a = _allocations.Values[hi];
        if (address >= a.Address + (ulong)a.Size) throw new ArgumentException($"no device buffer holds address {address:X}");
        return (a, address - a.Address);
    }

    private void Staging(long bytes)
    {
        if (_staging.Size >= bytes) return;
        Synchronize();
        Free(_staging);
        long size = _staging.Size;
        while (size < bytes) size *= 2;
        _staging = Allocate(size, host: true);
    }

    private void Free(Allocation a)
    {
        if (a.Mapped is not null) VkApi.vkUnmapMemory(_device, a.Memory);
        VkApi.vkDestroyBuffer(_device, a.Buffer, null);
        VkApi.vkFreeMemory(_device, a.Memory, null);
    }

    /// <summary>The host's bytes into device memory at <paramref name="destination"/> (ordered after the work recorded before it; returns when copied).</summary>
    public void Upload<T>(ReadOnlySpan<T> source, ulong destination) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(source);
        if (bytes.Length == 0) return;
        Staging(bytes.Length);
        Synchronize();
        bytes.CopyTo(new Span<byte>(_staging.Mapped, bytes.Length));
        var (a, offset) = Find(destination);
        var region = new VkBufferCopy { SourceOffset = 0, DestinationOffset = offset, Size = (ulong)bytes.Length };
        var cb = Recording();
        VkApi.vkCmdCopyBuffer(cb, _staging.Buffer, a.Buffer, 1, &region);
        Barrier(cb);
        Synchronize();
    }

    /// <summary>Device memory at <paramref name="source"/> into the host's bytes (after the work recorded before it).</summary>
    public void Download<T>(ulong source, Span<T> destination) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(destination);
        if (bytes.Length == 0) return;
        Staging(bytes.Length);
        var (a, offset) = Find(source);
        var region = new VkBufferCopy { SourceOffset = offset, DestinationOffset = 0, Size = (ulong)bytes.Length };
        var cb = Recording();
        VkApi.vkCmdCopyBuffer(cb, a.Buffer, _staging.Buffer, 1, &region);
        var barrier = new VkMemoryBarrier { SType = VkApi.StructureMemoryBarrier, SourceAccessMask = VkApi.AccessTransferWrite, DestinationAccessMask = VkApi.AccessHostRead };
        VkApi.vkCmdPipelineBarrier(cb, VkApi.StageTransfer, VkApi.StageHost, 0, 1, &barrier, 0, null, 0, null);
        Synchronize();
        new ReadOnlySpan<byte>(_staging.Mapped, bytes.Length).CopyTo(bytes);
    }

    /// <summary>Bytes from one device address to another, recorded in order with the dispatches.</summary>
    public void Copy(ulong source, ulong destination, long bytes)
    {
        if (bytes == 0) return;
        var (s, so) = Find(source);
        var (d, dOffset) = Find(destination);
        var region = new VkBufferCopy { SourceOffset = so, DestinationOffset = dOffset, Size = (ulong)bytes };
        var cb = Recording();
        VkApi.vkCmdCopyBuffer(cb, s.Buffer, d.Buffer, 1, &region);
        Barrier(cb);
    }

    /// <summary>Zero bytes at a device address (a multiple of 4 bytes from a 4-byte-aligned address), recorded in order.</summary>
    public void Clear(ulong destination, long bytes)
    {
        if (bytes == 0) return;
        var (d, offset) = Find(destination);
        var cb = Recording();
        VkApi.vkCmdFillBuffer(cb, d.Buffer, offset, (ulong)((bytes + 3) / 4 * 4), 0);
        Barrier(cb);
    }

    public void Dispose()
    {
        try { Synchronize(); } catch (InvalidOperationException) { }
        foreach (var (pipeline, _) in _byKey.Values) VkApi.vkDestroyPipeline(_device, pipeline, null);
        foreach (var module in _modules) VkApi.vkDestroyShaderModule(_device, module, null);
        foreach (var a in _allocations.Values) Free(a);
        Free(_staging);
        foreach (var f in _fences) VkApi.vkDestroyFence(_device, f, null);
        VkApi.vkDestroyCommandPool(_device, _pool, null);
        VkApi.vkDestroyPipelineLayout(_device, _layout, null);
        VkApi.vkDestroyDevice(_device, null);
        VkApi.vkDestroyInstance(_instance, null);
    }

    public override string ToString() => new StringBuilder(Name).Append(" (Vulkan)").ToString();
}

/// <summary>One shape of the device's matrix units: a subgroup multiplies an M × K matrix of A by a K × N matrix of B into
/// an M × N accumulator; the element types by name (f16, bf16, f32, s8, u8, s32, u32, …).</summary>
public readonly record struct CooperativeMatrix(int M, int N, int K, string A, string B, string C, string Result, bool Saturating)
{
    public override string ToString() => $"{A}{(B == A ? "" : "x" + B)}->{Result} {M}x{N}x{K}{(C == Result ? "" : $" (C {C})")}{(Saturating ? " sat" : "")}";
}
