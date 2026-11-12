// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Vulkan as a device: Adad's operations over the contract (Abstraction/), on device addresses — buffers of stored values
// and of indices alike — with the same lane rules as the CPU and CUDA: a type's own lane, its float lane for the float
// formulas. Every family of the contract has its Vulkan kernels, the block formats among the types (their outputs by the
// block encoders, VulkanOps.Blocks.cs); a shape past a kernel's limits is refused (NotSupportedException, naming it): no
// operation runs in another lane, or on another device, in its place.
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps(Vulkan vk) : IDevice<ulong, ulong>
{
    public Vulkan Vk => vk;

    private static uint Groups(long items, int threads) => (uint)((items + threads - 1) / threads);

    private static NotSupportedException Missing(string what) => new($"vulkan: {what} has no Vulkan kernels yet");

    // A lane parameter in its 64-bit slot: a float's bits, an integer's two's complement.
    private static ulong Slot(Lane lane, double value) => lane.IsFloat
        ? lane.Wide ? Vulkan.D(value) : Vulkan.F((float)value)
        : lane.Wide ? (ulong)(long)value : (uint)(int)value;

    // ---------------------------------------------------------------- memory

    public ulong Rent(long bytes) => vk.Rent(bytes);

    public ulong RentZeroed(long bytes)
    {
        ulong buffer = vk.Rent(bytes);
        vk.Clear(buffer, bytes);                 // whole 32-bit words: a buffer's size is one
        return buffer;
    }

    // A range of a buffer is its address past the buffer's (the device finds the buffer holding an address). The device's
    // fill takes whole 32-bit words from an aligned address: the bytes of a range before its first whole word and after
    // its last are copied from a buffer of zeros kept for it.
    private ulong _zeros;

    public void Clear(ulong buffer, long offset, long bytes)
    {
        if (bytes <= 0) return;
        ulong at = buffer + (ulong)offset, end = at + (ulong)bytes, first = (at + 3) & ~3UL, last = end & ~3UL;
        if (first >= last)
        {
            // Within a word or two: no whole word of the fill's.
            ClearBytes(at, bytes);
            return;
        }

        ClearBytes(at, (long)(first - at));
        vk.Clear(first, (long)(last - first));
        ClearBytes(last, (long)(end - last));
    }

    private void ClearBytes(ulong at, long bytes)
    {
        if (bytes <= 0) return;
        if (_zeros == 0) { _zeros = vk.Rent(256); vk.Clear(_zeros, 256); }
        vk.Copy(_zeros, at, bytes);
    }

    public void Upload(ReadOnlySpan<byte> source, ulong destination, long offset) => vk.Upload(source, destination + (ulong)offset);

    public void Download(ulong source, long offset, Span<byte> destination) => vk.Download(source + (ulong)offset, destination);

    /// <summary>The bytes of a copy below which the device's own copy command is taken whole.</summary>
    private const int CopyByKernel = 1 << 16;

    // A long range whose source and destination are the same bytes past a 32-bit word: the bytes up to its first whole
    // word and after its last by the device's copy command, its whole words by the copy kernel (the map's copy over
    // 32-bit words: the device's pace for a kernel, several times the command's on a range off a word). Any other range:
    // the command.
    public void Copy(ulong source, long sourceOffset, ulong destination, long destinationOffset, long bytes)
    {
        if (bytes <= 0) return;
        ulong s = source + (ulong)sourceOffset, d = destination + (ulong)destinationOffset;
        if (((s ^ d) & 3) != 0 || bytes < CopyByKernel) { vk.Copy(s, d, bytes); return; }
        long head = (long)((4 - (d & 3)) & 3), tail = (bytes - head) & 3, middle = bytes - head - tail;
        if (middle / 4 > int.MaxValue) { vk.Copy(s, d, bytes); return; }
        if (head > 0) vk.Copy(s, d, head);
        Map<Adad.Abstraction.UInt32>(Ops.Find("copy"), s + (ulong)head, 0, 0, d + (ulong)head, (int)(middle / 4), 0, 0);
        if (tail > 0) vk.Copy(s + (ulong)(head + middle), d + (ulong)(head + middle), tail);
    }

    public void Return(ulong buffer) => vk.Return(buffer);

    public void Upload(ReadOnlySpan<byte> source, ulong destination) => vk.Upload(source, destination);

    public void Download(ulong source, Span<byte> destination) => vk.Download(source, destination);

    public void Copy(ulong source, ulong destination, long bytes) => Copy(source, 0, destination, 0, bytes);

    public ulong RentIndices(int count) => vk.Rent(count * 4L);

    public void ReturnIndices(ulong indices) => vk.Return(indices);

    public void UploadIndices(ReadOnlySpan<int> source, ulong destination) => vk.Upload(source, destination);

    public void DownloadIndices(ulong source, Span<int> destination) => vk.Download(source, destination);

    public void Synchronize() => vk.Synchronize();

    public string Report() => vk.Facts;

    public string Measured() => "";

    // ---------------------------------------------------------------- element-wise

    public void Add<TN>(ulong a, ulong b, ulong c, int n) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock)
        {
            // A block format: each block encoded from the decoded sums (f32).
            Encode(f, n, vk.Cached($"block_add:{f.Name}", () => VkKernels.BlockAdd(vk, f)), c, a, b);
            return;
        }

        var k = vk.Cached($"add:{f.Name}", () => VkKernels.Add(vk, f));
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, a, b, c, Vulkan.U(n));
    }

    public string Map<TN>(MapOp op, ulong x, ulong y, ulong z, ulong output, int n, double a, double b) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (op.FloatOnly && !f.IsFloat) throw new NotSupportedException($"{op.Name}: a float formula; {f.Name} is not a float type");
        var lane = f.Natural;
        (a, b) = Stored.Held<TN>(a, b);
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_map:{op.Name}:{f.Name}:{lane.Name}", () => VkKernels.BlockMap(vk, f, lane, op)), output, x, y, z, Slot(lane, a), Slot(lane, b));
            return lane.Name;
        }

        var k = vk.Cached($"map:{op.Name}:{f.Name}:{lane.Name}", () => VkKernels.Map(vk, f, lane, op));
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, x, y, z, output, Vulkan.U(n), Slot(lane, a), Slot(lane, b));
        return lane.Name;
    }

    public string MapFloat<TN>(MapOp op, ulong x, ulong y, ulong z, ulong output, int n, double a, double b) where TN : struct, IFloatAdad<TN> =>
        Map<TN>(op, x, y, z, output, n, a, b);

    public string MapRow<TN>(MapOp op, ulong x, ulong v, ulong output, int rows, int cols) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = f.Natural;
        int n = rows * cols;
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_maprow:{op.Name}:{f.Name}", () => VkKernels.BlockMapRow(vk, f, lane, op)), output, x, v, Vulkan.U(cols));
            return lane.Name;
        }

        var k = vk.Cached($"maprow:{op.Name}:{f.Name}", () => VkKernels.MapRow(vk, f, lane, op));
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, x, v, output, Vulkan.U(n), Vulkan.U(cols));
        return lane.Name;
    }

    public string MapRowFloat<TN>(MapOp op, ulong x, ulong v, ulong output, int rows, int cols) where TN : struct, IFloatAdad<TN> =>
        MapRow<TN>(op, x, v, output, rows, cols);

    /// <summary>Dropout from the seed's draws (VkKernels.Dropout): a thread per word in the float lane, or a block format's
    /// blocks by their encoder (the residual and the output may be x).</summary>
    public string Dropout<TN>(ulong x, ulong residual, ulong output, int n, double p, ulong seed) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.Bits == 1) throw new NotSupportedException($"dropout: {f.Name} holds truth values, which 1/(1 − p) cannot scale");
        var key = Draws.Key(seed);
        ulong thr = Draws.Threshold(p);
        double s = 1 / (1 - p);
        bool res = residual != 0;
        ulong r = res ? residual : x;
        if (f.IsBlock)
        {
            Encode(f, n, vk.Cached($"block_dropout:{f.Name}:{res}", () => VkKernels.BlockDropout(vk, f, res)), output, x, r, key.A, thr, Vulkan.F((float)s), key.B);
            return "f32 (blocks in place)";
        }

        var lane = f.FloatLane;
        var k = vk.Cached($"dropout:{f.Name}:{res}", () => VkKernels.Dropout(vk, f, lane, res));
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, x, r, output, Vulkan.U(n), key.A, thr, Slot(lane, s), key.B);
        return lane.Name;
    }

    /// <summary>Each value of TFrom rounded once into TTo (VkKernels.Convert, Fmt.ConvertPlan): a thread per output word, or a
    /// block format's blocks by their encoder from the values as float32.</summary>
    public string Convert<TFrom, TTo>(ulong x, ulong output, int n) where TFrom : struct, IAdad<TFrom> where TTo : struct, IAdad<TTo>
    {
        Fmt ff = Fmt.Of<TFrom>(), ft = Fmt.Of<TTo>();
        var (lane, toOdd) = Fmt.ConvertPlan(ff, ft);
        if (ft.IsBlock)
        {
            Encode(ft, n, vk.Cached($"block_convert:{ff.Name}:{ft.Name}", () => VkKernels.BlockConvert(vk, ff, ft)), output, x);
            return "f32 (blocks encoded)";
        }

        var k = vk.Cached($"convert:{ff.Name}:{ft.Name}", () => VkKernels.Convert(vk, ff, ft));
        vk.Launch(k, Groups(ft.Words(n), VkKernels.Threads), 1, x, output, Vulkan.U(n));
        return lane.Name + (toOdd ? " (to odd at f32 first)" : "");
    }
}
