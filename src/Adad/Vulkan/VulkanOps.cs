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

    public void Return(ulong buffer) => vk.Return(buffer);

    public void Upload(ReadOnlySpan<byte> source, ulong destination) => vk.Upload(source, destination);

    public void Download(ulong source, Span<byte> destination) => vk.Download(source, destination);

    public void Copy(ulong source, ulong destination, long bytes) => vk.Copy(source, destination, bytes);

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
}
