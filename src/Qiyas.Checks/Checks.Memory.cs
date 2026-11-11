// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A device's memory by ranges (IMemory) on every device, byte for byte against the same steps on host arrays: a buffer
// rented zeroed where the pool hands back one that held other bytes; bytes uploaded at an offset that is no whole word;
// ranges cleared within a word, across two, and over many (odd offsets and lengths); a range copied to another buffer
// and to another place of its own; a range downloaded; and every byte outside each range as it was.
namespace Qiyas;

internal static partial class Checks
{
    public static bool MemoryCheck<TB, TI>(IDevice<TB, TI> d, string where)
    {
        const int n = 1001;
        var random = new Random(41);
        byte[] a = new byte[n], b = new byte[n], patch = new byte[37];
        random.NextBytes(a); random.NextBytes(b); random.NextBytes(patch);
        byte[] ma = (byte[])a.Clone(), mb = (byte[])b.Clone();
        string? wrong = null;
        TB da = d.Rent(n + 16), db = d.Rent(n + 16);
        d.Upload(a, da);
        d.Upload(b, db);

        d.Upload(patch, da, 3);
        patch.CopyTo(ma, 3);
        foreach (var (offset, bytes) in new[] { (5, 1), (130, 2), (200, 3), (301, 7), (402, 64), (513, 133), (996, 5), (700, 0) })
        {
            d.Clear(db, offset, bytes);
            mb.AsSpan(offset, bytes).Clear();
        }

        d.Copy(da, 11, db, 701, 129);
        ma.AsSpan(11, 129).CopyTo(mb.AsSpan(701));
        d.Copy(da, 600, da, 50, 77);
        ma.AsSpan(600, 77).CopyTo(ma.AsSpan(50));
        d.Synchronize();

        byte[] ga = new byte[n], gb = new byte[n], part = new byte[333];
        d.Download(da, ga);
        d.Download(db, gb);
        d.Download(da, 9, part);
        wrong ??= FirstDifference("the buffer written by ranges", ga, ma);
        wrong ??= FirstDifference("the buffer cleared and copied into by ranges", gb, mb);
        wrong ??= FirstDifference("a range downloaded", part, ma.AsSpan(9, part.Length).ToArray());

        // A zeroed rent of a buffer the pool held with other bytes in it.
        var ones = new byte[n];
        Array.Fill(ones, (byte)0xFF);
        d.Upload(ones, da);
        d.Upload(ones, db);
        d.Synchronize();
        d.Return(da);
        d.Return(db);
        TB z = d.RentZeroed(n);
        var gz = new byte[n];
        Array.Fill(gz, (byte)0x55);
        d.Synchronize();
        d.Download(z, gz);
        wrong ??= FirstDifference("a buffer rented zeroed", gz, new byte[n]);
        d.Return(z);

        if (wrong is not null) { Console.WriteLine($"check memory                 {where,-6} DIFFERS: {wrong}"); return false; }
        Console.WriteLine($"check memory                 {where,-6} ranges of a buffer byte for byte: uploaded at an offset, cleared (within a word, across words, none), copied to another buffer and within one, downloaded; a buffer rented zeroed after the pool held it with other bytes");
        return true;
    }

    private static string? FirstDifference(string what, byte[] got, byte[] want)
    {
        for (int i = 0; i < want.Length; i++)
            if (got[i] != want[i]) return $"{what}: byte {i} is {got[i]}, not {want[i]}";
        return null;
    }
}
