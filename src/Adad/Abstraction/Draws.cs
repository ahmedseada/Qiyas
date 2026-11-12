// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Adad's random draws. A draw is 32 bits made from a seed and a counter alone, by 32-bit integer multiplies, shifts, xors
// and adds — operations every device runs exactly and at full rate — so every device makes the same draw for the same
// seed and counter in any order and any number of threads, and a mask or a sample is made again where it is needed
// instead of stored or sent: dropout's backward from its seed, a sampler's draw from its seed and its step.
//
//   Mix(h)          h ^= h >> 16; h *= M1; h ^= h >> S; h *= M2; h ^= h >> 16. Each step is a bijection of the 32-bit
//                   values (a xor with a right shift of itself; a product by an odd number modulo 2^32), so Mix is one.
//   Key(seed)       A = Mix(seed's low 32 bits), B = Mix(seed's high 32 bits ^ A): a bijection of the 64-bit seeds, so two
//                   seeds never share a key.
//   Bits(key, i)    Mix(Mix(i ^ A) + B): for each key a bijection of the counter, so over the 2^32 counters each 32-bit value
//                   comes exactly once.
//   Stream(key, j)  (Bits(key, j), B): the key of stream j under a key (a sampler's step).
//   Uniform(bits)   bits · 2^−32 in [0, 1), exact in float64; a lane holds it rounded to its own precision.
//
// M1, M2 and S are measured, not taken: M1 and M2 from the odd numbers made of the first 32 bits of the fractional part of
// √p for each of the first 64 primes p, S from 13 to 17; of those 64 · 63 · 5 mixes, the one whose avalanche bias was the
// lowest measured: the root mean square over input bit j and output bit k of 2·P(k flips when j flips) − 1, over 2^24
// random inputs, the sampling noise taken out. Qiyas's dropout check measures the draws' rate and independence.
namespace Adad.Abstraction;

/// <summary>A seed's key (Draws.Key): two 32-bit words.</summary>
public readonly record struct DrawKey(uint A, uint B);

public static class Draws
{
    /// <summary>The measured mix's constants: M1 from √193, M2 from √127, S = 15 (avalanche bias 2.8e−4 measured).</summary>
    public const uint M1 = 0xE4773595u, M2 = 0x44F93635u;
    public const int S = 15;

    /// <summary>The 32-bit mix: a bijection, each input bit moving about half of the output's.</summary>
    public static uint Mix(uint h)
    {
        h ^= h >> 16;
        h *= M1;
        h ^= h >> S;
        h *= M2;
        h ^= h >> 16;
        return h;
    }

    /// <summary>The key of a seed (computed once, on the host).</summary>
    public static DrawKey Key(ulong seed)
    {
        uint a = Mix((uint)seed);
        return new(a, Mix((uint)(seed >> 32) ^ a));
    }

    /// <summary>Draw i's 32 bits under a key.</summary>
    public static uint Bits(DrawKey key, uint i) => Mix(Mix(i ^ key.A) + key.B);

    /// <summary>The key of stream j under a key.</summary>
    public static DrawKey Stream(DrawKey key, uint j) => new(Bits(key, j), key.B);

    /// <summary>A rate p as a threshold on the bits, round(p · 2^32) (at most 2^32 − 1): a draw is kept when its bits reach
    /// it, so of a key's 2^32 draws exactly 2^32 − threshold are kept.</summary>
    public static uint Threshold(double p)
    {
        if (!(p >= 0 && p < 1)) throw new ArgumentOutOfRangeException(nameof(p), p, "a rate in [0, 1)");
        return (uint)Math.Min(uint.MaxValue, Math.Round(p * 4294967296.0));
    }

    public static bool Keep(DrawKey key, uint i, uint threshold) => Bits(key, i) >= threshold;

    /// <summary>A draw's bits as a value in [0, 1): bits · 2^−32, exact.</summary>
    public static double Uniform(uint bits) => bits * (1.0 / 4294967296.0);

    /// <summary>A sampler's value at (step, row): draw `row` of the key's stream `step`.</summary>
    public static double Uniform(DrawKey key, uint step, uint row) => Uniform(Bits(Stream(key, step), row));
}
