// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The GPU's clock state as the driver's management library reports it (NVML, where the driver installs it: nvml.dll,
// libnvidia-ml.so.1): the performance state and the memory clock of the device CUDA runs on (found by its PCI address),
// read once at the end of each subject's timing of a row on CUDA. A driver that holds a GPU's memory clock down between
// bursts of work (a lower performance state: half the memory clock, or less) halves a memory-bound row's rate, and the
// row says so. No library, or no device found: no reading, nothing said.
using System.Globalization;
using System.Runtime.InteropServices;
using Adad.Cuda;

namespace Qiyas.Bench;

internal static unsafe class GpuClock
{
    private const int MemoryClock = 2;                                       // NVML_CLOCK_MEM

    private static bool s_tried;
    private static IntPtr s_device;
    private static delegate* unmanaged<IntPtr, int, uint*, int> s_clock;
    private static delegate* unmanaged<IntPtr, int*, int> s_state;

    /// <summary>"P1 mem 13801 MHz", or null where it cannot be read.</summary>
    public static string? Now()
    {
        if (!s_tried) Open();
        if (s_device == IntPtr.Zero) return null;
        uint mhz;
        int state;
        if (s_clock(s_device, MemoryClock, &mhz) != 0 || s_state(s_device, &state) != 0) return null;
        return string.Create(CultureInfo.InvariantCulture, $"P{state} mem {mhz} MHz");
    }

    /// <summary>The memory clock in a reading of Now (0 where none).</summary>
    public static long Mhz(string? reading)
    {
        if (reading is null) return 0;
        int at = reading.IndexOf("mem ", StringComparison.Ordinal);
        return at < 0 ? 0 : long.Parse(reading[(at + 4)..reading.IndexOf(' ', at + 4)], CultureInfo.InvariantCulture);
    }

    private static void Open()
    {
        s_tried = true;
        IntPtr lib = IntPtr.Zero;
        foreach (var name in new[] { "nvml.dll", "libnvidia-ml.so.1", "libnvidia-ml.so" })
        {
            if (NativeLibrary.TryLoad(name, out lib)) break;
        }

        if (lib == IntPtr.Zero) return;
        try
        {
            var init = (delegate* unmanaged<int>)NativeLibrary.GetExport(lib, "nvmlInit_v2");
            var byBus = (delegate* unmanaged<byte*, IntPtr*, int>)NativeLibrary.GetExport(lib, "nvmlDeviceGetHandleByPciBusId_v2");
            s_clock = (delegate* unmanaged<IntPtr, int, uint*, int>)NativeLibrary.GetExport(lib, "nvmlDeviceGetClockInfo");
            s_state = (delegate* unmanaged<IntPtr, int*, int>)NativeLibrary.GetExport(lib, "nvmlDeviceGetPerformanceState");
            if (init() != 0) return;
            // CUDA's device 0 (the one Adad runs on) by its PCI domain, bus and device numbers.
            if (Driver.cuDeviceGet(out int device, 0) != 0) return;
            if (Driver.cuDeviceGetAttribute(out int bus, 33, device) != 0) return;            // CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
            if (Driver.cuDeviceGetAttribute(out int slot, 34, device) != 0) return;           // CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
            if (Driver.cuDeviceGetAttribute(out int domain, 50, device) != 0) return;         // CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
            var address = System.Text.Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"{domain:X8}:{bus:X2}:{slot:X2}.0\0"));
            IntPtr handle;
            fixed (byte* a = address)
            {
                if (byBus(a, &handle) != 0) return;
            }

            s_device = handle;
        }
        catch (EntryPointNotFoundException)
        {
            s_device = IntPtr.Zero;
        }
    }
}
