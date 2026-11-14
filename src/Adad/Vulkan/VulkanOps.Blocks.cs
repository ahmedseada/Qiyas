// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The block formats' outputs on Vulkan (Kernels.Blocks.cs): an encoder over its source (lane storage, or the operation's
// formula over its inputs decoded where they are stored). Where the device stores 8-bit words, a workgroup's teams a
// block write each block where it lies (the encoders VkKernels.EncodesDirect chooses); an output that is one of its inputs (any argument in
// the output's buffer) is encoded into a buffer of its own and copied, so no block is written before every source has
// read it. Otherwise a thread a block into word-aligned staging, then compacted into the output's words, every encoder
// before any compaction (an output may be one of its inputs: the old blocks are read whole before they are written).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>An encoder's job: its kernel, the output it writes, its source's arguments (slots 0 .. 13).</summary>
    private readonly record struct BlockJob((ulong Pipeline, SpvKernel Kernel) Encoder, ulong Output, ulong[] Args);

    // Values 0 .. count − 1 of each job's output (of the block format f): every encoder, then every compaction.
    private void Encode(Fmt f, long count, params BlockJob[] jobs)
    {
        if (count <= 0) return;
        if (count > uint.MaxValue) throw new NotSupportedException($"vulkan: {count} values of {f.Name} past 2^32");
        long blocks = (count + f.BlockValues - 1) / f.BlockValues;
        bool direct = VkKernels.IsDirect(jobs[0].Encoder.Kernel);
        if (jobs.Any(j => VkKernels.IsDirect(j.Encoder.Kernel) != direct)) throw new InvalidOperationException($"vulkan: {f.Name}'s jobs mix direct and staged encoders");
        if (direct)
        {
            EncodeDirect(f, count, blocks, jobs);
            return;
        }

        var staging = new ulong[jobs.Length];
        try
        {
            for (int j = 0; j < jobs.Length; j++)
            {
                staging[j] = vk.Rent(blocks * VkKernels.BlockSlotWords(f) * 4L);
                if (jobs[j].Args.Length > VkKernels.StagingSlot) throw new ArgumentException($"{jobs[j].Encoder.Kernel.Name}: {jobs[j].Args.Length} source arguments");
                var slots = new ulong[Spv.MaxSlots];
                jobs[j].Args.CopyTo(slots, 0);
                slots[VkKernels.StagingSlot] = staging[j];
                slots[VkKernels.CountSlot] = (ulong)count;
                vk.Launch(jobs[j].Encoder, Groups(blocks, VkKernels.Threads), 1, slots);
            }

            var compact = vk.Cached($"block_compact:{f.Name}", () => VkKernels.BlockCompact(vk, f));
            for (int j = 0; j < jobs.Length; j++) vk.Launch(compact, Groups(f.Words(count), VkKernels.Threads), 1, staging[j], jobs[j].Output, Vulkan.U(blocks));
        }
        finally
        {
            foreach (var s in staging) if (s != 0) vk.Return(s);
        }
    }

    // Each job's encoder over the workgroups that hold its blocks, writing them where they lie; where an output's buffer
    // holds any job's argument, every job into a buffer of its own first, then each copied to its output.
    private void EncodeDirect(Fmt f, long count, long blocks, BlockJob[] jobs)
    {
        bool aliased = jobs.Any(o => jobs.Any(j => j.Args.Any(a => vk.SameBuffer(a, o.Output))));
        long bytes = blocks * f.BlockBytes;
        var targets = new ulong[jobs.Length];
        try
        {
            for (int j = 0; j < jobs.Length; j++)
            {
                if (jobs[j].Args.Length > VkKernels.StagingSlot) throw new ArgumentException($"{jobs[j].Encoder.Kernel.Name}: {jobs[j].Args.Length} source arguments");
                targets[j] = aliased ? vk.Rent(bytes) : jobs[j].Output;
                var slots = new ulong[Spv.MaxSlots];
                jobs[j].Args.CopyTo(slots, 0);
                slots[VkKernels.StagingSlot] = targets[j];
                slots[VkKernels.CountSlot] = (ulong)count;
                vk.Launch(jobs[j].Encoder, Groups(blocks, VkKernels.DirectBlocks(f)), 1, slots);
            }

            if (aliased) for (int j = 0; j < jobs.Length; j++) vk.Copy(targets[j], jobs[j].Output, bytes);
        }
        finally
        {
            if (aliased) foreach (var t in targets) if (t != 0) vk.Return(t);
        }
    }

    private void Encode(Fmt f, long count, (ulong Pipeline, SpvKernel Kernel) encoder, ulong output, params ulong[] args) =>
        Encode(f, count, new BlockJob(encoder, output, args));
}
