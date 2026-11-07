// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The CPU product's workspace: its packed B, the loaded A and B, and every task's A panel and tile of C in lanes, kept per
// thread and per element type, grown to the largest shape seen, so a call of a shape seen before allocates nothing and asks no pool.
// A slot already in use on the thread (a product inside a product) falls back to ArrayPool.
using System.Buffers;

namespace Adad;

internal static class Workspace
{
    // The product's slots.
    public const int Packed = 0, ALanes = 1, BLanes = 2, CLanes = 3, Panel = 4, KRows = 5, Slots = 6;

    private static long s_rents, s_grows, s_pooled;

    /// <summary>Cumulative counts: rents served, arrays grown (allocations), rents that fell back to ArrayPool.</summary>
    public static (long Rents, long Grows, long Pooled) Counts => (Interlocked.Read(ref s_rents), Interlocked.Read(ref s_grows), Interlocked.Read(ref s_pooled));

    internal static void Rented() => Interlocked.Increment(ref s_rents);

    internal static void Grown() => Interlocked.Increment(ref s_grows);

    internal static void Pooled() => Interlocked.Increment(ref s_pooled);
}

internal static class Workspace<T> where T : unmanaged
{
    [ThreadStatic] private static T[]?[]? s_slots;
    [ThreadStatic] private static int s_busy;

    /// <summary>An array of at least <paramref name="length"/> values (contents left as they were) for slot <paramref name="slot"/>.</summary>
    public static T[] Rent(int slot, int length)
    {
        Workspace.Rented();
        int bit = 1 << slot;
        if ((s_busy & bit) != 0)
        {
            Workspace.Pooled();
            return ArrayPool<T>.Shared.Rent(length);
        }

        var slots = s_slots ??= new T[Workspace.Slots][];
        var array = slots[slot];
        if (array is null || array.Length < length)
        {
            // Grown with a quarter's headroom, so shapes that creep up do not reallocate every call.
            array = slots[slot] = GC.AllocateUninitializedArray<T>(length + length / 4);
            Workspace.Grown();
        }

        s_busy |= bit;
        return array;
    }

    public static void Return(int slot, T[] array)
    {
        int bit = 1 << slot;
        if ((s_busy & bit) != 0 && ReferenceEquals(s_slots![slot], array))
        {
            s_busy &= ~bit;
            return;
        }

        ArrayPool<T>.Shared.Return(array);
    }
}
