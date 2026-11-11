// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Adad's own Vulkan binding: the loader (vulkan-1 on Windows, libvulkan.so.1 elsewhere) that ships with the graphics
// driver, and the core functions a compute device needs — no SDK, no layers. Buffers reach the kernels by their device
// addresses (Vulkan 1.2's bufferDeviceAddress): a kernel's arguments are 64-bit slots of push constants, addresses and
// values alike, as CUDA's are, so there are no descriptors. Dispatchable handles are pointers; the others 64-bit numbers.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace Adad.Vulkan;

internal static unsafe partial class VkApi
{
    private const string Library = "vulkan-1";

    private static IntPtr s_handle;

    static VkApi()
    {
        // The CUDA binding owns this assembly's import resolver, so the loader is found through the load context's
        // fallback instead (asked only for names the default search did not find).
        var context = AssemblyLoadContext.GetLoadContext(typeof(VkApi).Assembly) ?? AssemblyLoadContext.Default;
        context.ResolvingUnmanagedDll += Resolve;
    }

    private static IntPtr Resolve(Assembly assembly, string name) =>
        name == Library && assembly == typeof(VkApi).Assembly && TryLoad(out _) ? s_handle : IntPtr.Zero;

    private static string[] Candidates => OperatingSystem.IsWindows() ? ["vulkan-1.dll"] : OperatingSystem.IsAndroid() ? ["libvulkan.so"] : ["libvulkan.so.1", "libvulkan.so"];

    /// <summary>Loads the Vulkan loader; false (with the reason) when there is none.</summary>
    public static bool TryLoad(out string reason)
    {
        reason = "";
        if (s_handle != IntPtr.Zero) return true;
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS())
        {
            reason = "no Vulkan loader on this operating system";
            return false;
        }

        foreach (var name in Candidates)
        {
            if (NativeLibrary.TryLoad(name, out var handle))
            {
                s_handle = handle;
                return true;
            }
        }

        reason = $"the Vulkan loader ({string.Join(" / ", Candidates)}) was not found";
        return false;
    }

    public static void Check(int result, string call)
    {
        if (result != Success) throw new InvalidOperationException($"{call}: {Describe(result)}");
    }

    public static string Describe(int result) => result switch
    {
        Success => "VK_SUCCESS",
        NotReady => "VK_NOT_READY",
        Timeout => "VK_TIMEOUT",
        -1 => "VK_ERROR_OUT_OF_HOST_MEMORY",
        -2 => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        -3 => "VK_ERROR_INITIALIZATION_FAILED",
        -4 => "VK_ERROR_DEVICE_LOST",
        -5 => "VK_ERROR_MEMORY_MAP_FAILED",
        -7 => "VK_ERROR_EXTENSION_NOT_PRESENT",
        -8 => "VK_ERROR_FEATURE_NOT_PRESENT",
        -9 => "VK_ERROR_INCOMPATIBLE_DRIVER",
        -13 => "VK_ERROR_UNKNOWN",
        -1000012000 => "VK_ERROR_INVALID_SHADER_NV",
        _ => "VK_ERROR",
    } + $" ({result})";

    public const int Success = 0, NotReady = 1, Timeout = 2;

    // Structure types.
    public const uint StructureApplicationInfo = 0, StructureInstanceCreateInfo = 1, StructureDeviceQueueCreateInfo = 2,
        StructureDeviceCreateInfo = 3, StructureSubmitInfo = 4, StructureMemoryAllocateInfo = 5, StructureFenceCreateInfo = 8,
        StructureBufferCreateInfo = 12, StructureShaderModuleCreateInfo = 16, StructurePipelineShaderStageCreateInfo = 18,
        StructureComputePipelineCreateInfo = 29, StructurePipelineLayoutCreateInfo = 30, StructureCommandPoolCreateInfo = 39,
        StructureCommandBufferAllocateInfo = 40, StructureCommandBufferBeginInfo = 42, StructureMemoryBarrier = 46,
        StructurePhysicalDeviceVulkan12Features = 51, StructurePhysicalDeviceFeatures2 = 1000059000,
        StructurePhysicalDeviceProperties2 = 1000059001, StructureMemoryAllocateFlagsInfo = 1000060000,
        StructurePhysicalDeviceSubgroupProperties = 1000094000, StructurePhysicalDeviceFloatControlsProperties = 1000197000,
        StructureBufferDeviceAddressInfo = 1000244001;

    public static uint MakeVersion(uint major, uint minor) => (major << 22) | (minor << 12);

    public const uint DeviceTypeIntegratedGpu = 1, DeviceTypeDiscreteGpu = 2, DeviceTypeVirtualGpu = 3, DeviceTypeCpu = 4;

    public const uint QueueCompute = 0x2;

    public const uint MemoryDeviceLocal = 0x1, MemoryHostVisible = 0x2, MemoryHostCoherent = 0x4, MemoryHostCached = 0x8;

    public const uint BufferTransferSource = 0x1, BufferTransferDestination = 0x2, BufferStorageBuffer = 0x20, BufferDeviceAddress = 0x20000;
    public const uint MemoryAllocateDeviceAddress = 0x2;

    public const uint ShaderStageCompute = 0x20;
    public const uint PipelineBindPointCompute = 1;

    public const uint CommandPoolResetCommandBuffer = 0x2;
    public const uint CommandBufferLevelPrimary = 0;
    public const uint CommandBufferOneTimeSubmit = 0x1;

    public const uint StageComputeShader = 0x800, StageTransfer = 0x1000, StageHost = 0x4000;
    public const uint AccessShaderRead = 0x20, AccessShaderWrite = 0x40, AccessTransferRead = 0x800, AccessTransferWrite = 0x1000,
        AccessHostRead = 0x2000, AccessHostWrite = 0x4000;

    // ---------------------------------------------------------------- instance and devices

    [LibraryImport(Library)] public static partial int vkEnumerateInstanceVersion(out uint version);
    [LibraryImport(Library)] public static partial int vkCreateInstance(VkInstanceCreateInfo* info, void* allocator, out IntPtr instance);
    [LibraryImport(Library)] public static partial void vkDestroyInstance(IntPtr instance, void* allocator);
    [LibraryImport(Library)] public static partial int vkEnumeratePhysicalDevices(IntPtr instance, ref uint count, IntPtr* devices);
    [LibraryImport(Library)] public static partial void vkGetPhysicalDeviceProperties(IntPtr physicalDevice, VkPhysicalDeviceProperties* properties);
    [LibraryImport(Library)] public static partial void vkGetPhysicalDeviceProperties2(IntPtr physicalDevice, VkPhysicalDeviceProperties2* properties);
    [LibraryImport(Library)] public static partial void vkGetPhysicalDeviceFeatures2(IntPtr physicalDevice, VkPhysicalDeviceFeatures2* features);
    [LibraryImport(Library)] public static partial void vkGetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, VkPhysicalDeviceMemoryProperties* properties);
    [LibraryImport(Library)] public static partial void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint count, VkQueueFamilyProperties* properties);
    [LibraryImport(Library)] public static partial int vkCreateDevice(IntPtr physicalDevice, VkDeviceCreateInfo* info, void* allocator, out IntPtr device);
    [LibraryImport(Library)] public static partial void vkDestroyDevice(IntPtr device, void* allocator);
    [LibraryImport(Library)] public static partial void vkGetDeviceQueue(IntPtr device, uint family, uint index, out IntPtr queue);
    [LibraryImport(Library)] public static partial int vkDeviceWaitIdle(IntPtr device);
    [LibraryImport(Library)] public static partial IntPtr vkGetDeviceProcAddr(IntPtr device, byte* name);

    // ---------------------------------------------------------------- memory and buffers

    [LibraryImport(Library)] public static partial int vkCreateBuffer(IntPtr device, VkBufferCreateInfo* info, void* allocator, out ulong buffer);
    [LibraryImport(Library)] public static partial void vkDestroyBuffer(IntPtr device, ulong buffer, void* allocator);
    [LibraryImport(Library)] public static partial void vkGetBufferMemoryRequirements(IntPtr device, ulong buffer, VkMemoryRequirements* requirements);
    [LibraryImport(Library)] public static partial int vkAllocateMemory(IntPtr device, VkMemoryAllocateInfo* info, void* allocator, out ulong memory);
    [LibraryImport(Library)] public static partial void vkFreeMemory(IntPtr device, ulong memory, void* allocator);
    [LibraryImport(Library)] public static partial int vkBindBufferMemory(IntPtr device, ulong buffer, ulong memory, ulong offset);
    [LibraryImport(Library)] public static partial int vkMapMemory(IntPtr device, ulong memory, ulong offset, ulong size, uint flags, void** data);
    [LibraryImport(Library)] public static partial void vkUnmapMemory(IntPtr device, ulong memory);

    // ---------------------------------------------------------------- pipelines

    [LibraryImport(Library)] public static partial int vkCreatePipelineLayout(IntPtr device, VkPipelineLayoutCreateInfo* info, void* allocator, out ulong layout);
    [LibraryImport(Library)] public static partial void vkDestroyPipelineLayout(IntPtr device, ulong layout, void* allocator);
    [LibraryImport(Library)] public static partial int vkCreateShaderModule(IntPtr device, VkShaderModuleCreateInfo* info, void* allocator, out ulong module);
    [LibraryImport(Library)] public static partial void vkDestroyShaderModule(IntPtr device, ulong module, void* allocator);
    [LibraryImport(Library)] public static partial int vkCreateComputePipelines(IntPtr device, ulong cache, uint count, VkComputePipelineCreateInfo* infos, void* allocator, ulong* pipelines);
    [LibraryImport(Library)] public static partial void vkDestroyPipeline(IntPtr device, ulong pipeline, void* allocator);

    // ---------------------------------------------------------------- commands

    [LibraryImport(Library)] public static partial int vkCreateCommandPool(IntPtr device, VkCommandPoolCreateInfo* info, void* allocator, out ulong pool);
    [LibraryImport(Library)] public static partial void vkDestroyCommandPool(IntPtr device, ulong pool, void* allocator);
    [LibraryImport(Library)] public static partial int vkAllocateCommandBuffers(IntPtr device, VkCommandBufferAllocateInfo* info, IntPtr* buffers);
    [LibraryImport(Library)] public static partial int vkResetCommandBuffer(IntPtr commandBuffer, uint flags);
    [LibraryImport(Library)] public static partial int vkBeginCommandBuffer(IntPtr commandBuffer, VkCommandBufferBeginInfo* info);
    [LibraryImport(Library)] public static partial int vkEndCommandBuffer(IntPtr commandBuffer);
    [LibraryImport(Library)] public static partial void vkCmdBindPipeline(IntPtr commandBuffer, uint bindPoint, ulong pipeline);
    [LibraryImport(Library)] public static partial void vkCmdPushConstants(IntPtr commandBuffer, ulong layout, uint stages, uint offset, uint size, void* values);
    [LibraryImport(Library)] public static partial void vkCmdDispatch(IntPtr commandBuffer, uint groupsX, uint groupsY, uint groupsZ);
    [LibraryImport(Library)] public static partial void vkCmdPipelineBarrier(IntPtr commandBuffer, uint sourceStages, uint destinationStages, uint dependencyFlags,
        uint memoryBarrierCount, VkMemoryBarrier* memoryBarriers, uint bufferBarrierCount, void* bufferBarriers, uint imageBarrierCount, void* imageBarriers);
    [LibraryImport(Library)] public static partial void vkCmdCopyBuffer(IntPtr commandBuffer, ulong source, ulong destination, uint regionCount, VkBufferCopy* regions);
    [LibraryImport(Library)] public static partial void vkCmdFillBuffer(IntPtr commandBuffer, ulong buffer, ulong offset, ulong size, uint data);

    // ---------------------------------------------------------------- synchronization

    [LibraryImport(Library)] public static partial int vkCreateFence(IntPtr device, VkFenceCreateInfo* info, void* allocator, out ulong fence);
    [LibraryImport(Library)] public static partial void vkDestroyFence(IntPtr device, ulong fence, void* allocator);
    [LibraryImport(Library)] public static partial int vkResetFences(IntPtr device, uint count, ulong* fences);
    [LibraryImport(Library)] public static partial int vkWaitForFences(IntPtr device, uint count, ulong* fences, uint waitAll, ulong timeout);
    [LibraryImport(Library)] public static partial int vkQueueSubmit(IntPtr queue, uint count, VkSubmitInfo* submits, ulong fence);
}

// The structures, laid out as in vulkan_core.h (natural C alignment; 64-bit handles; pointers as void*).

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkApplicationInfo
{
    public uint SType;
    public void* PNext;
    public byte* ApplicationName;
    public uint ApplicationVersion;
    public byte* EngineName;
    public uint EngineVersion;
    public uint ApiVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkInstanceCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public VkApplicationInfo* ApplicationInfo;
    public uint EnabledLayerCount;
    public byte** EnabledLayerNames;
    public uint EnabledExtensionCount;
    public byte** EnabledExtensionNames;
}

/// <summary>VkPhysicalDeviceProperties (824 bytes): the fields read here, at their offsets; the limits start at 296.</summary>
[StructLayout(LayoutKind.Explicit, Size = 824)]
internal unsafe struct VkPhysicalDeviceProperties
{
    [FieldOffset(0)] public uint ApiVersion;
    [FieldOffset(4)] public uint DriverVersion;
    [FieldOffset(8)] public uint VendorId;
    [FieldOffset(12)] public uint DeviceId;
    [FieldOffset(16)] public uint DeviceType;
    [FieldOffset(20)] public fixed byte DeviceName[256];
    [FieldOffset(296 + 28)] public uint MaxStorageBufferRange;
    [FieldOffset(296 + 32)] public uint MaxPushConstantsSize;
    [FieldOffset(296 + 216)] public uint MaxComputeSharedMemorySize;
    [FieldOffset(296 + 220)] public uint MaxComputeWorkGroupCountX;
    [FieldOffset(296 + 224)] public uint MaxComputeWorkGroupCountY;
    [FieldOffset(296 + 228)] public uint MaxComputeWorkGroupCountZ;
    [FieldOffset(296 + 232)] public uint MaxComputeWorkGroupInvocations;
    [FieldOffset(296 + 236)] public uint MaxComputeWorkGroupSizeX;
    [FieldOffset(296 + 240)] public uint MaxComputeWorkGroupSizeY;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceProperties2
{
    public uint SType;
    public void* PNext;
    public VkPhysicalDeviceProperties Properties;
}

/// <summary>VkPhysicalDeviceSubgroupProperties (Vulkan 1.1).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceSubgroupProperties
{
    public uint SType;
    public void* PNext;
    public uint SubgroupSize;
    public uint SupportedStages;
    public uint SupportedOperations;
    public uint QuadOperationsInAllStages;
}

/// <summary>VkPhysicalDeviceFloatControlsProperties (Vulkan 1.2): what a kernel may ask of the device's float arithmetic.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceFloatControlsProperties
{
    public uint SType;
    public void* PNext;
    public uint DenormBehaviorIndependence;
    public uint RoundingModeIndependence;
    public uint SignedZeroInfNanPreserveFloat16;
    public uint SignedZeroInfNanPreserveFloat32;
    public uint SignedZeroInfNanPreserveFloat64;
    public uint DenormPreserveFloat16;
    public uint DenormPreserveFloat32;
    public uint DenormPreserveFloat64;
    public uint DenormFlushToZeroFloat16;
    public uint DenormFlushToZeroFloat32;
    public uint DenormFlushToZeroFloat64;
    public uint RoundingModeRteFloat16;
    public uint RoundingModeRteFloat32;
    public uint RoundingModeRteFloat64;
    public uint RoundingModeRtzFloat16;
    public uint RoundingModeRtzFloat32;
    public uint RoundingModeRtzFloat64;
}

/// <summary>VkPhysicalDeviceFeatures2: the core 1.0 features, 55 booleans (shaderFloat64 is the 40th, shaderInt64 the 41st, shaderInt16 the 42nd).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceFeatures2
{
    public uint SType;
    public void* PNext;
    public fixed uint Features[55];

    public const int ShaderFloat64 = 39, ShaderInt64 = 40, ShaderInt16 = 41;
}

/// <summary>VkPhysicalDeviceVulkan12Features: 47 booleans (shaderFloat16 the 8th, shaderInt8 the 9th, bufferDeviceAddress the 39th).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceVulkan12Features
{
    public uint SType;
    public void* PNext;
    public fixed uint Features[47];

    public const int ShaderFloat16 = 7, ShaderInt8 = 8, BufferDeviceAddress = 38;
}

/// <summary>VkPhysicalDeviceMemoryProperties: 32 memory types (flags, heap) and 16 heaps (size, flags).</summary>
[StructLayout(LayoutKind.Explicit, Size = 520)]
internal unsafe struct VkPhysicalDeviceMemoryProperties
{
    [FieldOffset(0)] public uint MemoryTypeCount;
    [FieldOffset(4)] public fixed uint MemoryTypes[64];
    [FieldOffset(260)] public uint MemoryHeapCount;
    [FieldOffset(264)] public fixed ulong MemoryHeaps[32];

    public uint TypeFlags(int type) => MemoryTypes[2 * type];

    public int TypeHeap(int type) => (int)MemoryTypes[2 * type + 1];

    public ulong HeapSize(int heap) => MemoryHeaps[2 * heap];
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkQueueFamilyProperties
{
    public uint QueueFlags;
    public uint QueueCount;
    public uint TimestampValidBits;
    public uint GranularityWidth, GranularityHeight, GranularityDepth;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDeviceQueueCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint QueueFamilyIndex;
    public uint QueueCount;
    public float* QueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDeviceCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint QueueCreateInfoCount;
    public VkDeviceQueueCreateInfo* QueueCreateInfos;
    public uint EnabledLayerCount;
    public byte** EnabledLayerNames;
    public uint EnabledExtensionCount;
    public byte** EnabledExtensionNames;
    public void* EnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkBufferCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public ulong Size;
    public uint Usage;
    public uint SharingMode;
    public uint QueueFamilyIndexCount;
    public uint* QueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryRequirements
{
    public ulong Size;
    public ulong Alignment;
    public uint MemoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkMemoryAllocateInfo
{
    public uint SType;
    public void* PNext;
    public ulong AllocationSize;
    public uint MemoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkMemoryAllocateFlagsInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint DeviceMask;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkBufferDeviceAddressInfo
{
    public uint SType;
    public void* PNext;
    public ulong Buffer;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPushConstantRange
{
    public uint StageFlags;
    public uint Offset;
    public uint Size;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPipelineLayoutCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint SetLayoutCount;
    public ulong* SetLayouts;
    public uint PushConstantRangeCount;
    public VkPushConstantRange* PushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkShaderModuleCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public nuint CodeSize;
    public uint* Code;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPipelineShaderStageCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint Stage;
    public ulong Module;
    public byte* Name;
    public void* SpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkComputePipelineCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public VkPipelineShaderStageCreateInfo Stage;
    public ulong Layout;
    public ulong BasePipelineHandle;
    public int BasePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCommandPoolCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint QueueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCommandBufferAllocateInfo
{
    public uint SType;
    public void* PNext;
    public ulong CommandPool;
    public uint Level;
    public uint CommandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCommandBufferBeginInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public void* InheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkMemoryBarrier
{
    public uint SType;
    public void* PNext;
    public uint SourceAccessMask;
    public uint DestinationAccessMask;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCopy
{
    public ulong SourceOffset;
    public ulong DestinationOffset;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkFenceCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkSubmitInfo
{
    public uint SType;
    public void* PNext;
    public uint WaitSemaphoreCount;
    public ulong* WaitSemaphores;
    public uint* WaitDestinationStageMask;
    public uint CommandBufferCount;
    public IntPtr* CommandBuffers;
    public uint SignalSemaphoreCount;
    public ulong* SignalSemaphores;
}
