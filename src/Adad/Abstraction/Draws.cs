// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Random draws without state: a seed's key, and the 32 bits of draw i under it, the same integer arithmetic on every device
// (32-bit multiplies, shifts and xors), so a mask or a sample is regenerated where it is needed instead of stored or sent —
// dropout's backward from its seed, a sampler's uniform from the seed and the step.
namespace Adad.Abstraction;

public static class Draws
{
    /// <summary>A 32-bit avalanche (each input bit moves about half of the output's).</summary>
    public static uint Mix(uint h)
    {
        h ^= h >> 16;
        h *= 0x7FEB352Du;
        h ^= h >> 15;
        h *= 0x846CA68Bu;
        h ^= h >> 16;
        return h;
    }

    /// <summary>The key of a seed: both halves mixed (computed once, on the host).</summary>
    public static uint Key(ulong seed) => Mix(Mix((uint)seed) ^ (uint)(seed >> 32) ^ 0x9E3779B9u);

    /// <summary>Draw i's 32 bits under a key.</summary>
    public static uint Bits(uint key, uint i) => Mix(Mix(i ^ key) + key);

    /// <summary>Dropout's rate p as a threshold on the bits: draw i keeps its value when its bits reach it (p in steps of 2^−32).</summary>
    public static uint Threshold(double p)
    {
        if (!(p >= 0 && p < 1)) throw new ArgumentOutOfRangeException(nameof(p), p, "a dropout rate in [0, 1)");
        return (uint)Math.Min(uint.MaxValue, Math.Round(p * 4294967296.0));
    }

    public static bool Keep(uint key, uint i, uint threshold) => Bits(key, i) >= threshold;

    /// <summary>Draw i as a uniform value in [0, 1): its top 24 bits over 2^24 (exact in float32).</summary>
    public static double Uniform(uint key, uint i) => (Bits(key, i) >> 8) * (1.0 / 16777216);

    /// <summary>A sampler's uniform value in [0, 1) at (step, row): draw `row` under the key's draw `step`.</summary>
    public static double Uniform(uint key, uint step, uint row) => Uniform(Bits(key, step), row);
}
