// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Qiyas's own CUDA driver binding: the driver API functions the kernels need, the device's facts as it reports them
// (compute capability, multiprocessors), PTX modules JIT-compiled once per text, device memory with a free-list so a
// buffer of a size seen before costs no driver call, and launches with every argument in a 64-bit slot.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace Qiyas;

internal static unsafe partial class Driver
{
    // "nvcuda" is the driver's name on Windows; elsewhere a resolver maps it to libcuda (set only where no other
    // binding in this assembly has set one: the bench also carries the lib's).
    private const string Library = "nvcuda";

    static Driver()
    {
        if (OperatingSystem.IsWindows()) return;
        try { NativeLibrary.SetDllImportResolver(typeof(Driver).Assembly, Resolve); }
        catch (InvalidOperationException) { }
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? path)
    {
        if (name != Library) return IntPtr.Zero;
        foreach (var candidate in new[] { "libcuda.so.1", "libcuda.so" })
        {
            if (NativeLibrary.TryLoad(candidate, out var handle)) return handle;
        }

        return IntPtr.Zero;
    }

    public static bool Available
    {
        get
        {
            try { return cuInit(0) == 0; }
            catch (DllNotFoundException) { return false; }
        }
    }

    [LibraryImport(Library)] public static partial int cuInit(uint flags);
    [LibraryImport(Library)] public static partial int cuDeviceGet(out int device, int ordinal);
    [LibraryImport(Library)] public static partial int cuDeviceGetAttribute(out int value, int attribute, int device);
    [LibraryImport(Library)] public static partial int cuDeviceGetName(byte* name, int length, int device);
    [LibraryImport(Library)] public static partial int cuDevicePrimaryCtxRetain(out IntPtr context, int device);
    [LibraryImport(Library)] public static partial int cuCtxSetCurrent(IntPtr context);
    [LibraryImport(Library)] public static partial int cuCtxSynchronize();
    [LibraryImport(Library)] public static partial int cuStreamQuery(IntPtr stream);
    [LibraryImport(Library)] public static partial int cuModuleLoadDataEx(out IntPtr module, byte* image, uint options, int* keys, void** values);
    [LibraryImport(Library)] public static partial int cuModuleGetFunction(out IntPtr function, IntPtr module, byte* name);
    [LibraryImport(Library)] public static partial int cuModuleUnload(IntPtr module);
    [LibraryImport(Library, EntryPoint = "cuMemAlloc_v2")] public static partial int cuMemAlloc(out ulong pointer, nuint bytes);
    [LibraryImport(Library, EntryPoint = "cuMemFree_v2")] public static partial int cuMemFree(ulong pointer);
    [LibraryImport(Library, EntryPoint = "cuMemcpyHtoD_v2")] public static partial int cuMemcpyHtoD(ulong destination, void* source, nuint bytes);
    [LibraryImport(Library, EntryPoint = "cuMemcpyDtoH_v2")] public static partial int cuMemcpyDtoH(void* destination, ulong source, nuint bytes);
    [LibraryImport(Library, EntryPoint = "cuMemcpyDtoD_v2")] public static partial int cuMemcpyDtoD(ulong destination, ulong source, nuint bytes);
    [LibraryImport(Library, EntryPoint = "cuMemsetD8_v2")] public static partial int cuMemsetD8(ulong destination, byte value, nuint bytes);
    [LibraryImport(Library)] public static partial int cuLaunchKernel(IntPtr function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, uint sharedBytes, IntPtr stream, void** parameters, void** extra);
    [LibraryImport(Library)] public static partial int cuGetErrorString(int error, out IntPtr text);
    [LibraryImport(Library)] public static partial int cuFuncSetAttribute(IntPtr function, int attribute, int value);
    [LibraryImport(Library)] public static partial int cuFuncGetAttribute(out int value, int attribute, IntPtr function);
    public const int FunctionAttributeMaxThreadsPerBlock = 0;

    public const int AttributeMultiprocessors = 16, AttributeMajor = 75, AttributeMinor = 76, AttributeSharedPerBlock = 8, AttributeSharedOptIn = 97;
    public const int AttributeThreadsPerMultiprocessor = 39;
    public const int FunctionAttributeMaxDynamicShared = 8;
    public const int JitErrorLogBuffer = 5, JitErrorLogBufferSizeBytes = 6;

    public static void Check(int result, string call)
    {
        if (result != 0) throw new InvalidOperationException($"{call}: {Describe(result)}");
    }

    public static string Describe(int result) => cuGetErrorString(result, out var text) == 0 ? Marshal.PtrToStringAnsi(text) ?? result.ToString() : result.ToString();
}

/// <summary>One CUDA device: its facts, its compiled kernels and its memory.</summary>
public sealed unsafe partial class Gpu : IDisposable
{
    private readonly Dictionary<string, IntPtr> _modules = [];
    private readonly Dictionary<(IntPtr, string), IntPtr> _functions = [];
    private readonly Dictionary<long, Stack<ulong>> _free = [];
    private readonly Dictionary<ulong, long> _sizes = [];
    private readonly HashSet<string> _refused = [];

    public Gpu()
    {
        Driver.Check(Driver.cuInit(0), "cuInit");
        Driver.Check(Driver.cuDeviceGet(out int device, 0), "cuDeviceGet");
        Driver.Check(Driver.cuDevicePrimaryCtxRetain(out var context, device), "cuDevicePrimaryCtxRetain");
        Driver.Check(Driver.cuCtxSetCurrent(context), "cuCtxSetCurrent");
        Driver.Check(Driver.cuDeviceGetAttribute(out int sms, Driver.AttributeMultiprocessors, device), "cuDeviceGetAttribute");
        Driver.Check(Driver.cuDeviceGetAttribute(out int major, Driver.AttributeMajor, device), "cuDeviceGetAttribute");
        Driver.Check(Driver.cuDeviceGetAttribute(out int minor, Driver.AttributeMinor, device), "cuDeviceGetAttribute");
        Driver.Check(Driver.cuDeviceGetAttribute(out int shared, Driver.AttributeSharedPerBlock, device), "cuDeviceGetAttribute");
        int optIn = shared;
        if (Driver.cuDeviceGetAttribute(out int optInValue, Driver.AttributeSharedOptIn, device) == 0) optIn = Math.Max(shared, optInValue);
        SharedOptIn = optIn;
        // The threads a multiprocessor holds at once (2048 assumed where the driver does not say).
        ThreadsPerMultiprocessor = Driver.cuDeviceGetAttribute(out int perSm, Driver.AttributeThreadsPerMultiprocessor, device) == 0 && perSm > 0 ? perSm : 2048;
        var name = stackalloc byte[256];
        Driver.Check(Driver.cuDeviceGetName(name, 256, device), "cuDeviceGetName");
        Name = Marshal.PtrToStringAnsi((IntPtr)name) ?? "cuda";
        Multiprocessors = sms;
        Capability = major * 10 + minor;
        SharedPerBlock = shared;
    }

    public string Name { get; }

    public int Multiprocessors { get; }

    /// <summary>The most threads a multiprocessor holds resident, as the device reports.</summary>
    public int ThreadsPerMultiprocessor { get; }

    /// <summary>The compute capability the device reports, as major·10 + minor.</summary>
    public int Capability { get; }

    public int SharedPerBlock { get; }

    /// <summary>The most shared memory a block may ask for, as the device reports (its opt-in limit).</summary>
    public int SharedOptIn { get; }

    private readonly HashSet<IntPtr> _widened = [];

    /// <summary>Lets <paramref name="function"/> take <paramref name="bytes"/> of dynamic shared memory (beyond the static limit).</summary>
    public void AllowShared(IntPtr function, int bytes)
    {
        if (bytes <= 48 * 1024 || !_widened.Add(function)) return;
        Driver.Check(Driver.cuFuncSetAttribute(function, Driver.FunctionAttributeMaxDynamicShared, bytes), "cuFuncSetAttribute");
    }

    /// <summary>The PTX target line: the device's own capability, held to what PTX ISA 7.0 names (the JIT builds for the real device).</summary>
    public string Target => $".version 7.0\n.target sm_{Math.Min(Capability, 80)}\n.address_size 64\n";

    /// <summary>The function <paramref name="entry"/> of the module built from <paramref name="ptx"/> (JIT-compiled once per text).</summary>
    public IntPtr Function(string ptx, string entry)
    {
        if (Environment.GetEnvironmentVariable("QIYAS_PTX_DUMP") is { Length: > 0 } dir && !_modules.ContainsKey(ptx))
        {
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, entry + ".ptx"), ptx);
        }

        if (!_modules.TryGetValue(ptx, out var module))
        {
            byte[] image = Encoding.UTF8.GetBytes(ptx + "\0");
            var log = new byte[16384];
            fixed (byte* p = image)
            fixed (byte* l = log)
            {
                int* keys = stackalloc int[] { Driver.JitErrorLogBuffer, Driver.JitErrorLogBufferSizeBytes };
                void** values = stackalloc void*[] { l, (void*)(nint)log.Length };
                int result = Driver.cuModuleLoadDataEx(out module, p, 2, keys, values);
                if (result != 0)
                {
                    string text = Encoding.UTF8.GetString(log).TrimEnd('\0');
                    throw new InvalidOperationException($"cuModuleLoadDataEx failed: {Driver.Describe(result)}\n{text}\n{Numbered(ptx)}");
                }
            }

            _modules.Add(ptx, module);
        }

        if (!_functions.TryGetValue((module, entry), out var function))
        {
            byte[] bytes = Encoding.UTF8.GetBytes(entry + "\0");
            fixed (byte* b = bytes)
            {
                Driver.Check(Driver.cuModuleGetFunction(out function, module, b), "cuModuleGetFunction " + entry);
            }

            _functions.Add((module, entry), function);
        }

        return function;
    }

    /// <summary>A kernel's text, made on demand (its entry name through the out parameter).</summary>
    public delegate string Maker(out string entry);

    private readonly Dictionary<string, IntPtr> _byKey = [];

    /// <summary>The function for <paramref name="key"/>: made and compiled the first time, then looked up without regenerating its text.</summary>
    public IntPtr Cached(string key, Maker make)
    {
        if (_byKey.TryGetValue(key, out var function)) return function;
        string ptx = make(out string entry);
        function = Function(ptx, entry);
        _byKey[key] = function;
        return function;
    }

    /// <summary>Whether the device's JIT accepts <paramref name="ptx"/> (a kernel using an instruction the device may lack); the answer is remembered.
    /// <paramref name="reason"/>: the JIT's first lines when it refuses the kernel now, null when accepted or refused before.</summary>
    public bool Accepts(string ptx, string entry, out string? reason)
    {
        reason = null;
        if (_refused.Contains(entry)) return false;
        try
        {
            Function(ptx, entry);
            return true;
        }
        catch (InvalidOperationException e)
        {
            _refused.Add(entry);
            reason = string.Join(" | ", e.Message.Split((char)10).Take(3));
            return false;
        }
    }

    private static string Numbered(string ptx)
    {
        var sb = new StringBuilder();
        int line = 1;
        foreach (var l in ptx.Split('\n')) sb.Append(line++).Append(": ").Append(l).Append('\n');
        return sb.ToString();
    }

    /// <summary>Device memory of at least <paramref name="bytes"/>: inside a scope from the arena (src/Arena.cs), else from the
    /// free-list when a buffer of that size was released before.</summary>
    public ulong Rent(long bytes) => _depth > 0 ? ArenaRent(bytes) : PoolRent(bytes);

    public void Return(ulong pointer)
    {
        if (ArenaReturn(pointer)) return;
        PoolReturn(pointer);
    }

    private ulong PoolRent(long bytes)
    {
        long size = Math.Max(256, (bytes + 255) / 256 * 256);
        _poolRents++;
        if (_free.TryGetValue(size, out var stack) && stack.Count > 0) return stack.Pop();
        Driver.Check(Driver.cuMemAlloc(out ulong pointer, (nuint)size), "cuMemAlloc");
        _driverAllocations++;
        _sizes[pointer] = size;
        return pointer;
    }

    private void PoolReturn(ulong pointer)
    {
        long size = _sizes[pointer];
        if (!_free.TryGetValue(size, out var stack)) _free[size] = stack = new Stack<ulong>();
        stack.Push(pointer);
    }

    public static void Upload<T>(ReadOnlySpan<T> source, ulong destination) where T : unmanaged
    {
        fixed (T* p = source) Driver.Check(Driver.cuMemcpyHtoD(destination, p, (nuint)(source.Length * sizeof(T))), "cuMemcpyHtoD");
    }

    public static void Download<T>(ulong source, Span<T> destination) where T : unmanaged
    {
        fixed (T* p = destination) Driver.Check(Driver.cuMemcpyDtoH(p, source, (nuint)(destination.Length * sizeof(T))), "cuMemcpyDtoH");
    }

    public static void Copy(ulong source, ulong destination, long bytes) => Driver.Check(Driver.cuMemcpyDtoD(destination, source, (nuint)bytes), "cuMemcpyDtoD");

    public static void Clear(ulong destination, long bytes) => Driver.Check(Driver.cuMemsetD8(destination, 0, (nuint)bytes), "cuMemsetD8");

    public static void Synchronize() => Driver.Check(Driver.cuCtxSynchronize(), "cuCtxSynchronize");

    /// <summary>Hands the launches queued so far to the device without waiting for them (a query of the default stream:
    /// on Windows (WDDM) the driver batches launches and submits them late otherwise — item 6j: a 4096³ float16 product
    /// whose launches waited queued read 1.1 ms for 0.5). Its answer (done or still running) is not used.</summary>
    public static void Flush() => _ = Driver.cuStreamQuery(IntPtr.Zero);

    /// <summary>The most threads a block of <paramref name="function"/> can be launched with, as the driver reports for this
    /// device (the kernel's registers per thread against the multiprocessor's, and the device's own limit).</summary>
    public static int MaxThreads(IntPtr function)
    {
        lock (s_maxThreads)
        {
            // Asked of the driver once per function (a launch path asks every call).
            if (s_maxThreads.TryGetValue(function, out int known)) return known;
            Driver.Check(Driver.cuFuncGetAttribute(out int value, Driver.FunctionAttributeMaxThreadsPerBlock, function), "cuFuncGetAttribute");
            s_maxThreads[function] = value;
            return value;
        }
    }

    private static readonly Dictionary<IntPtr, int> s_maxThreads = [];

    public static void Launch(IntPtr function, uint gridX, uint gridY, uint blockX, uint blockY, uint sharedBytes, params ReadOnlySpan<ulong> args)
    {
        var slots = stackalloc ulong[args.Length];
        var pointers = stackalloc void*[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            slots[i] = args[i];
            pointers[i] = &slots[i];
        }

        Driver.Check(Driver.cuLaunchKernel(function, gridX, gridY, 1, blockX, blockY, 1, sharedBytes, IntPtr.Zero, pointers, null), "cuLaunchKernel");
    }

    public static ulong U(int value) => (uint)value;

    public static ulong F(float value) => BitConverter.SingleToUInt32Bits(value);

    public static ulong D(double value) => BitConverter.DoubleToUInt64Bits(value);

    public void Dispose()
    {
        foreach (var module in _modules.Values) Driver.cuModuleUnload(module);
        foreach (var pointer in _sizes.Keys) Driver.cuMemFree(pointer);
        if (_region != 0) Driver.cuMemFree(_region);
    }
}
