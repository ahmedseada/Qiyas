// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Stores that bypass the caches. A pass whose operands and result together are well past every cache the processor
// reports (each level's caches summed: what all the cores hold at once) evicts its own result before anything could read
// it from a cache, and a plain store first reads each line it writes (a write miss allocates the line): the result's bytes
// cross the memory bus twice. Such a pass computes its tiles in place on the stack and copies each into the result with
// non-temporal stores (the line written whole, not read first), then fences them before its task ends. The caches are the
// system's own report (Windows' logical processor information, Linux's /sys/devices/system/cpu); where none can be read,
// nothing streams. "Well past" is 1.5 times them, measured once: passes of 1.7 and 2.5 times the caches ran 0.72 to 0.80
// of their time streamed, one of 1.2 times (float16 add, 96 MB against 80 MB) 1.10 (part of it stays cached between calls).
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Adad.Cpu;

internal static unsafe class Streams
{
    /// <summary>Every data cache the processor reports, summed (0: none could be read).</summary>
    public static readonly long CacheBytes = Caches();

    /// <summary>Whether a pass moving this many bytes (read and written) streams its result past the caches: past 1.5 times
    /// them (above).</summary>
    public static bool Past(long bytes) => Force ?? (CacheBytes > 0 && bytes > CacheBytes + CacheBytes / 2);

    /// <summary>The gate's override: true streams every pass that can, false none, null by the bytes (above).</summary>
    internal static bool? Force;

    /// <summary>dst = src: the whole aligned vectors of dst by non-temporal stores, its edges by plain ones.</summary>
    public static void Copy(ReadOnlySpan<byte> src, Span<byte> dst)
    {
        int n = src.Length, i = 0, w = Vector<byte>.Count;
        fixed (byte* s = src)
        fixed (byte* d = dst)
        {
            int head = (int)Math.Min(n, (uint)(w - (int)((nuint)d & (nuint)(w - 1))) & (uint)(w - 1));
            for (; i < head; i++) d[i] = s[i];
            for (; i + w <= n; i += w)
            {
                var v = Vector.Load(s + i);
                if (w == 32) Vector256.StoreAlignedNonTemporal(v.AsVector256(), d + i);
                else if (w == 64) Vector512.StoreAlignedNonTemporal(v.AsVector512(), d + i);
                else Vector128.StoreAlignedNonTemporal(v.AsVector128(), d + i);
            }

            for (; i < n; i++) d[i] = s[i];
        }
    }

    /// <summary>The streamed stores made visible to the other threads (before a task's end is counted).</summary>
    public static void Fence()
    {
        if (Sse.IsSupported) Sse.StoreFence();
        else Interlocked.MemoryBarrier();
    }

    private static long Caches()
    {
        try
        {
            var found = OperatingSystem.IsWindows() ? WindowsCaches() : OperatingSystem.IsLinux() ? LinuxCaches() : [];
            long total = 0;
            foreach (var (_, bytes, _) in found.DistinctBy(c => (c.Level, c.Key))) total += bytes;
            return total;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or FormatException or EntryPointNotFoundException or DllNotFoundException)
        {
            return 0;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(int relationship, byte* buffer, ref int length);

    // RelationCache (2): a record a cache, its level, size and type (data or unified kept) at fixed offsets.
    private static List<(int Level, long Bytes, string Key)> WindowsCaches()
    {
        var list = new List<(int, long, string)>();
        int length = 0;
        GetLogicalProcessorInformationEx(2, null, ref length);
        if (length <= 0) return list;
        var buffer = new byte[length];
        fixed (byte* b = buffer)
        {
            if (!GetLogicalProcessorInformationEx(2, b, ref length)) return list;
            for (int at = 0, n = 0; at + 24 <= length; n++)
            {
                int size = *(int*)(b + at + 4), level = b[at + 8], type = *(int*)(b + at + 16);
                long bytes = (uint)*(int*)(b + at + 12);
                if (type is 0 or 2) list.Add((level, bytes, $"w{n}"));
                if (size <= 0) break;
                at += size;
            }
        }

        return list;
    }

    // Each processor's cache directories: one cache a level and set of processors sharing it.
    private static List<(int Level, long Bytes, string Key)> LinuxCaches()
    {
        var list = new List<(int, long, string)>();
        foreach (var cpu in Directory.GetDirectories("/sys/devices/system/cpu", "cpu*"))
        {
            string dir = Path.Combine(cpu, "cache");
            if (!Directory.Exists(dir)) continue;
            foreach (var index in Directory.GetDirectories(dir, "index*"))
            {
                string Read(string f) => File.ReadAllText(Path.Combine(index, f)).Trim();
                if (Read("type") is not ("Data" or "Unified")) continue;
                string text = Read("size");
                long unit = text.EndsWith('K') ? 1L << 10 : text.EndsWith('M') ? 1L << 20 : text.EndsWith('G') ? 1L << 30 : 1;
                long bytes = long.Parse(text.TrimEnd('K', 'M', 'G'), CultureInfo.InvariantCulture) * unit;
                list.Add((int.Parse(Read("level"), CultureInfo.InvariantCulture), bytes, Read("shared_cpu_list")));
            }
        }

        return list;
    }
}
