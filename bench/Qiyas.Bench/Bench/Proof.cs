// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The checks' proof for a build. A run whose checks pass on a device records it beside the build (proof-<device>.txt in
// the assemblies' directory): the build (SHA-256 over Adad.dll, Qiyas.Checks.dll and Qiyas.Bench.dll, the bytes this run
// loaded), the device, the checks' stress and when. A run given --proven hashes its own assemblies and skips the checks
// only where every device it benches has a proof of that very build at its stress or harder; otherwise the checks run as
// before. A copy of the directory (a snapshot) takes its proofs along, and its assemblies hash the same.
//
// What a device's proof covers: cpu, the checks every run makes first (Checks.Run); cuda and vulkan, those and the
// device's own (Checks.Gpu, Checks.Vulkan), all in one process. The proof says only that this build passed them on this
// machine: the build alone decides what runs (the PTX and the shaders are made from Adad's code at run time).
//
// Passed means the checks returned true and the run printed no failing line before the proof (Watched: the gates'
// pattern, DIFFER|FAIL|[Cc]heck failed|Unhandled|Exception|DEVICE_LOST|illegal memory, over everything the run wrote to
// the console): a failure a check reports in words its result missed still keeps the proof unwritten.
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Qiyas.Bench;

internal sealed class Proof
{
    private static readonly string[] Assemblies = ["Adad.dll", "Qiyas.Checks.dll", "Qiyas.Bench.dll"];

    private readonly string _directory;

    /// <summary>The build: SHA-256 over the assemblies' names and bytes, in hex.</summary>
    public string Build { get; }

    /// <summary>The build's first 16 hex digits, for the lines a run prints.</summary>
    public string Short => Build[..16];

    private Proof(string directory, string build) { _directory = directory; Build = build; }

    /// <summary>The build in `directory` (the running bench's own: AppContext.BaseDirectory).</summary>
    public static Proof Of(string directory)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var name in Assemblies)
        {
            sha.AppendData(Encoding.UTF8.GetBytes(name + "\n"));
            sha.AppendData(File.ReadAllBytes(Path.Combine(directory, name)));
        }

        return new Proof(directory, Convert.ToHexStringLower(sha.GetHashAndReset()));
    }

    private string FileOf(Device device) => Path.Combine(_directory, $"proof-{Name(device)}.txt");

    private static string Name(Device device) => device.ToString().ToLowerInvariant();

    /// <summary>Whether every device has a proof of this build at `stress` or harder; `said` names each proof found, or the
    /// devices without one.</summary>
    public bool Covers(IEnumerable<Device> devices, int stress, out string said)
    {
        var found = new List<string>();
        var missing = new List<string>();
        foreach (var device in devices.Distinct())
        {
            if (Read(device) is { } p && p.Build == Build && p.Stress >= stress) found.Add($"{Name(device)} (passed {p.When}, stress {p.Stress})");
            else missing.Add(Name(device));
        }

        said = missing.Count == 0 ? string.Join(", ", found) : string.Join(", ", missing);
        return missing.Count == 0;
    }

    /// <summary>Records that this build passed the device's checks at `stress`. A proof that cannot be written is only said:
    /// the next run's checks then run.</summary>
    public void Record(Device device, int stress)
    {
        if (Watched.Failed)
        {
            Console.WriteLine($"proof: not recorded for {Name(device)}: the run printed a failing line");
            return;
        }

        string line = string.Create(CultureInfo.InvariantCulture, $"build {Build} device {Name(device)} stress {stress} passed {DateTime.Now:yyyy-MM-ddTHH:mm:ss} pid {Environment.ProcessId}");
        string path = FileOf(device), temp = $"{path}.{Environment.ProcessId}.tmp";
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                File.WriteAllText(temp, line + Environment.NewLine);
                File.Move(temp, path, overwrite: true);                    // whole or not at all: parallel gates on one build
                Console.WriteLine($"proof: recorded for {Name(device)}, build {Short}");
                return;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                if (attempt < 5) { Thread.Sleep(50); continue; }             // another gate's move of the same file
                Console.WriteLine($"proof: not recorded for {Name(device)} ({e.Message.Trim()})");
                try { File.Delete(temp); } catch (Exception d) when (d is IOException or UnauthorizedAccessException) { }
                return;
            }
        }
    }

    private (string Build, int Stress, string When)? Read(Device device)
    {
        string path = FileOf(device);
        if (!File.Exists(path)) return null;
        try
        {
            var f = File.ReadAllText(path).Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (f.Length < 8 || f[0] != "build" || f[2] != "device" || f[3] != Name(device) || f[4] != "stress" || f[6] != "passed") return null;
            return (f[1], int.Parse(f[5], CultureInfo.InvariantCulture), f[7]);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }
}

/// <summary>The console's output passed through while each line is matched against the gates' failing pattern: once
/// one matches, no proof is recorded for the rest of the run.</summary>
internal sealed class Watched : TextWriter
{
    private static readonly Regex Failing = new("DIFFER|FAIL|[Cc]heck failed|Unhandled|Exception|DEVICE_LOST|illegal memory", RegexOptions.Compiled);
    private static volatile bool s_failed;
    private readonly TextWriter _inner;
    private readonly StringBuilder _line = new();

    private Watched(TextWriter inner) => _inner = inner;

    /// <summary>Whether the run has printed a failing line.</summary>
    public static bool Failed => s_failed;

    /// <summary>Puts the watch on the console's output (writers on several threads in turn: synchronized).</summary>
    public static void Install() => Console.SetOut(TextWriter.Synchronized(new Watched(Console.Out)));

    public override Encoding Encoding => _inner.Encoding;

    public override void Write(char value)
    {
        _inner.Write(value);
        Take(value);
    }

    public override void Write(string? value)
    {
        if (value is null) return;
        _inner.Write(value);
        foreach (char c in value) Take(c);
    }

    public override void Write(char[] buffer, int index, int count)
    {
        _inner.Write(buffer, index, count);
        for (int i = index; i < index + count; i++) Take(buffer[i]);
    }

    public override void Flush() => _inner.Flush();

    private void Take(char c)
    {
        if (c != '\n') { _line.Append(c); return; }
        if (Failing.IsMatch(_line.ToString())) s_failed = true;
        _line.Clear();
    }
}
