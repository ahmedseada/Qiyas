// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The CPU product's workspace: its packed panel of B, and every task's A panel and tile of C in lanes, kept per
// thread and per element type, grown to the largest shape seen, so a call of a shape seen before allocates nothing and asks no pool.
// A slot already in use on the thread (a product inside a product) falls back to ArrayPool.
// The bytes these kept buffers lend a call are counted (Held), so the bench's peak working memory sees them: a rent that
// falls back to ArrayPool is the pool's to count, not counted here. Each thread keeps its own count (one writer a count,
// no atomics); the reading sums the threads' peaks after the call.
using System.Buffers;
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal static class Workspace
{
    // The product's slots.
    public const int Packed = 0, CLanes = 1, Panel = 2, KRows = 3, Slots = 4;

    private static long s_rents, s_grows, s_pooled;

    /// <summary>Cumulative counts: rents served, arrays grown (allocations), rents that fell back to ArrayPool.</summary>
    public static (long Rents, long Grows, long Pooled) Counts => (Interlocked.Read(ref s_rents), Interlocked.Read(ref s_grows), Interlocked.Read(ref s_pooled));

    internal static void Rented() => Interlocked.Increment(ref s_rents);

    internal static void Grown() => Interlocked.Increment(ref s_grows);

    internal static void Pooled() => Interlocked.Increment(ref s_pooled);

    // A thread's count of the kept bytes it has lent: written by its thread alone, started again when the thread first
    // lends in a new count (ResetHeld moves the generation).
    private sealed class HeldCount
    {
        public long Generation, Held, Peak;
    }

    [ThreadStatic] private static HeldCount? t_held;
    private static readonly List<HeldCount> s_counts = [];                  // every thread's (added once, under the lock)
    private static long s_generation = 1;

    /// <summary>The bytes of kept buffers (this workspace's slots, the sequences' scratch) lent since ResetHeld: each
    /// thread's most at once, summed — a bound on the most lent at once across the threads (two threads' peaks may fall at
    /// different moments). Read after a call.</summary>
    public static long HeldPeak
    {
        get
        {
            long sum = 0, generation = Generation;
            lock (s_counts)
            {
                foreach (var c in s_counts)
                {
                    if (Volatile.Read(ref c.Generation) == generation) sum += c.Peak;
                }
            }

            return sum;
        }
    }

    /// <summary>Starts a count of the kept buffers' bytes lent (the scratch's slots are counted again from here); called
    /// between calls, never during one.</summary>
    public static void ResetHeld() => Volatile.Write(ref s_generation, Generation + 1);

    /// <summary>The count a kept buffer handed out without a return (the scratch's) is taken once in: moved by ResetHeld.</summary>
    internal static long Generation => Volatile.Read(ref s_generation);

    // This thread's count, started again in a new generation.
    private static HeldCount Mine()
    {
        var c = t_held;
        if (c is null)
        {
            t_held = c = new HeldCount();
            lock (s_counts) s_counts.Add(c);
        }

        long generation = Generation;
        if (c.Generation != generation)
        {
            c.Held = 0;
            c.Peak = 0;
            Volatile.Write(ref c.Generation, generation);
        }

        return c;
    }

    /// <summary>A kept buffer of <paramref name="bytes"/> lent (one add a buffer, nothing a value).</summary>
    internal static void Hold(long bytes)
    {
        var c = Mine();
        c.Held += bytes;
        if (c.Held > c.Peak) c.Peak = c.Held;
    }

    /// <summary>A kept buffer of <paramref name="bytes"/> given back.</summary>
    internal static void Release(long bytes) => Mine().Held -= bytes;
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
        Workspace.Hold((long)array.Length * Unsafe.SizeOf<T>());
        return array;
    }

    public static void Return(int slot, T[] array)
    {
        int bit = 1 << slot;
        if ((s_busy & bit) != 0 && ReferenceEquals(s_slots![slot], array))
        {
            s_busy &= ~bit;
            Workspace.Release((long)array.Length * Unsafe.SizeOf<T>());
            return;
        }

        ArrayPool<T>.Shared.Return(array);
    }
}
