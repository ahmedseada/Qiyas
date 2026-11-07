// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Vulkan as a device: Adad's operations over the contract (Abstraction/), on device addresses — buffers of stored values
// and of indices alike — with the same lane rules as the CPU and CUDA: a type's own lane, its float lane for the float
// formulas. A family whose kernels are not written for this device yet refuses (NotSupportedException, naming it): no
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
        if (f.IsBlock) throw Missing($"add of {f.Name} (a block format)");
        var k = vk.Cached($"add:{f.Name}", () => VkKernels.Add(vk, f));
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, a, b, c, Vulkan.U(n));
    }

    public string Map<TN>(MapOp op, ulong x, ulong y, ulong z, ulong output, int n, double a, double b) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"map {op.Name} of {f.Name} (a block format)");
        if (op.FloatOnly && !f.IsFloat) throw new NotSupportedException($"{op.Name}: a float formula; {f.Name} is not a float type");
        var lane = f.Natural;
        (a, b) = Stored.Held<TN>(a, b);
        var k = vk.Cached($"map:{op.Name}:{f.Name}:{lane.Name}", () => VkKernels.Map(vk, f, lane, op));
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, x, y, z, output, Vulkan.U(n), Slot(lane, a), Slot(lane, b));
        return lane.Name;
    }

    public string MapFloat<TN>(MapOp op, ulong x, ulong y, ulong z, ulong output, int n, double a, double b) where TN : struct, IFloatAdad<TN> =>
        Map<TN>(op, x, y, z, output, n, a, b);

    public string MapRow<TN>(MapOp op, ulong x, ulong v, ulong output, int rows, int cols) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"map_row {op.Name} of {f.Name} (a block format)");
        var lane = f.Natural;
        var k = vk.Cached($"maprow:{op.Name}:{f.Name}", () => VkKernels.MapRow(vk, f, lane, op));
        int n = rows * cols;
        vk.Launch(k, Groups(f.Words(n), VkKernels.Threads), 1, x, v, output, Vulkan.U(n), Vulkan.U(cols));
        return lane.Name;
    }

    // ---------------------------------------------------------------- the families not on Vulkan yet

    public string Softmax<TN>(ulong x, ulong y, int rows, int cols) where TN : struct, IFloatAdad<TN> => throw Missing("softmax");

    public string RmsNorm<TN>(ulong x, ulong y, int rows, int cols, double eps) where TN : struct, IFloatAdad<TN> => throw Missing("rms_norm");

    public string RowOp<TN>(RowOp op, ulong x, ulong x2, ulong g, ulong bt, ulong s, ulong pos, ulong y, ulong y2, int rows, int cols, double a, double b) where TN : struct, IFloatAdad<TN> =>
        throw Missing($"row op {op}");

    public string NormParamsBwd<TN>(ulong x, ulong dy, ulong dgamma, ulong dbeta, int rows, int cols, double eps, bool centered) where TN : struct, IFloatAdad<TN> =>
        throw Missing("norm parameters' backward");

    public string Layout<TN>(LayoutArgs k, ulong x, ulong index, ulong output, ulong x2) where TN : struct, IAdad<TN> => throw Missing($"layout {k.Name}");

    public string MatMul<TN>(ulong a, ulong b, ulong c, int m, int n, int k) where TN : struct, IAdad<TN> => throw Missing("matmul");

    public string MatMulView<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN> => throw Missing("matmul (view)");

    public string MatMulBackward<TN>(ulong a, ulong b, ulong dc, ulong da, ulong db, int m, int n, int k) where TN : struct, IAdad<TN> => throw Missing("matmul backward");

    public string MatMulViewBlocks<TN>(ProductView v, ulong a, ulong b, ulong c) where TN : struct, IAdad<TN> => throw Missing("block-scaled matmul");

    public bool BlockProductAvailable => false;

    public string Attention<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> => throw Missing("attention");

    public string AttentionComposed<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> => throw Missing("attention (composed)");

    public string AttentionRows<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> => throw Missing("attention (fused rows)");

    public string AttentionDecode<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) where TN : struct, IFloatAdad<TN> => throw Missing("attention decode");

    public string AttentionBackward<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv) where TN : struct, IFloatAdad<TN> =>
        throw Missing("attention backward");

    public string Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) where TN : struct, IFloatAdad<TN> => throw Missing("optimizers");

    public string Conv<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN> => throw Missing("conv");

    public string ConvComposed<TN>(ConvArgs a, ulong x, ulong w, ulong y) where TN : struct, IAdad<TN> => throw Missing("conv (composed)");

    public string ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw) where TN : struct, IFloatAdad<TN> => throw Missing("conv backward");

    public void CellGates<TN>(CellArgs a, ulong x, ulong w, ulong h, ulong u, ulong bias, ulong g) where TN : struct, IAdad<TN> => throw Missing("recurrent gates");

    public string LstmCell<TN>(CellArgs a, ulong g, ulong c, ulong hOut, ulong cOut) where TN : struct, IFloatAdad<TN> => throw Missing("lstm cell");

    public string LstmCellBackward<TN>(CellArgs a, ulong g, ulong c, ulong dh, ulong dcNext, ulong dg, ulong dc) where TN : struct, IFloatAdad<TN> => throw Missing("lstm cell backward");

    public string GruCell<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong hOut) where TN : struct, IFloatAdad<TN> => throw Missing("gru cell");

    public string GruCellBackward<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong dhOut, ulong dgx, ulong dgh, ulong dh) where TN : struct, IFloatAdad<TN> =>
        throw Missing("gru cell backward");

    public string LstmStep<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IFloatAdad<TN> => throw Missing("lstm step");

    public string LstmStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcOut, ulong dx, ulong dh, ulong dc, ulong dw, ulong du, ulong db)
        where TN : struct, IFloatAdad<TN> => throw Missing("lstm step backward");

    public string GruStep<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IFloatAdad<TN> => throw Missing("gru step");

    public string GruStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh, ulong dw, ulong du, ulong db, ulong dbh)
        where TN : struct, IFloatAdad<TN> => throw Missing("gru step backward");

    public string LstmSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IFloatAdad<TN> =>
        throw Missing("lstm sequence");

    public string LstmSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcLast, ulong dx, ulong dh0, ulong dc0, ulong dw, ulong du, ulong db)
        where TN : struct, IFloatAdad<TN> => throw Missing("lstm sequence backward");

    public string GruSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IFloatAdad<TN> =>
        throw Missing("gru sequence");

    public string GruSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh0, ulong dw, ulong du, ulong db, ulong dbh)
        where TN : struct, IFloatAdad<TN> => throw Missing("gru sequence backward");

    public string TopK<TN>(ulong x, int rows, int cols, int k, ulong values, ulong indices) where TN : struct, IAdad<TN> => throw Missing("topk");

    public string Pick<TN>(ulong x, int rows, int cols, ulong index, ulong output) where TN : struct, IAdad<TN> => throw Missing("pick");

    public string Sample<TN>(ulong x, int rows, int cols, ulong u, ulong picks) where TN : struct, IFloatAdad<TN> => throw Missing("sample");

    public string CrossEntropy<TN>(ulong x, int rows, int cols, ulong targets, ulong output) where TN : struct, IFloatAdad<TN> => throw Missing("cross entropy");

    public void Route(ulong index, int count, int buckets, ulong counts, ulong offsets, ulong order) => throw Missing("route");

    public string Ctc<TN>(CtcArgs a, ulong x, ulong loss, ulong grad) where TN : struct, IFloatAdad<TN> => throw Missing("ctc");
}
