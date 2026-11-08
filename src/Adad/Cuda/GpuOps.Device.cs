// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// CUDA as a device: the device operations over the contract (Abstractions/), on device pointers — buffers of stored values
// and of indices alike. The operations whose signature is the contract's implement it as they are; the float formulas
// (their constraint IFloatAdad in the contract) and the memory are here.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps : IDevice<ulong, ulong>
{
    // ---------------------------------------------------------------- memory

    ulong IMemory<ulong, ulong>.Rent(long bytes) => gpu.Rent(bytes);

    void IMemory<ulong, ulong>.Return(ulong buffer) => gpu.Return(buffer);

    void IMemory<ulong, ulong>.Upload(ReadOnlySpan<byte> source, ulong destination) => Gpu.Upload(source, destination);

    void IMemory<ulong, ulong>.Download(ulong source, Span<byte> destination) => Gpu.Download(source, destination);

    void IMemory<ulong, ulong>.Copy(ulong source, ulong destination, long bytes) => Gpu.Copy(source, destination, bytes);

    ulong IMemory<ulong, ulong>.RentIndices(int count) => gpu.Rent(count * 4L);

    void IMemory<ulong, ulong>.ReturnIndices(ulong indices) => gpu.Return(indices);

    void IMemory<ulong, ulong>.UploadIndices(ReadOnlySpan<int> source, ulong destination) => Gpu.Upload(source, destination);

    void IMemory<ulong, ulong>.DownloadIndices(ulong source, Span<int> destination) => Gpu.Download(source, destination);

    void IMemory<ulong, ulong>.Synchronize() => Gpu.Synchronize();

    // ---------------------------------------------------------------- the float formulas

    string IElementwise<ulong>.MapFloat<TN>(MapOp op, ulong x, ulong y, ulong z, ulong output, int n, double a, double b) => Map<TN>(op, x, y, z, output, n, a, b);

    string IRows<ulong, ulong>.Softmax<TN>(ulong x, ulong y, int rows, int cols) => Softmax<TN>(x, y, rows, cols);

    string IRows<ulong, ulong>.RmsNorm<TN>(ulong x, ulong y, int rows, int cols, double eps) => RmsNorm<TN>(x, y, rows, cols, eps);

    string IRows<ulong, ulong>.RowOp<TN>(RowOp op, ulong x, ulong x2, ulong g, ulong bt, ulong s, ulong pos, ulong y, ulong y2, int rows, int cols, double a, double b) =>
        RowOp<TN>(op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b);

    string IRows<ulong, ulong>.NormParamsBwd<TN>(ulong x, ulong dy, ulong dgamma, ulong dbeta, int rows, int cols, double eps, bool centered) =>
        NormParamsBwd<TN>(x, dy, dgamma, dbeta, rows, cols, eps, centered);

    string IAttention<ulong>.Attention<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) => Attention<TN>(a, q, k, v, o);

    string IAttention<ulong>.AttentionComposed<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) => AttentionComposed<TN>(a, q, k, v, o);

    string IAttention<ulong>.AttentionRows<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) => AttentionRows<TN>(a, q, k, v, o);

    string IAttention<ulong>.AttentionDecode<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o) => AttentionDecode<TN>(a, q, k, v, o);

    string IAttention<ulong>.AttentionLse<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong o, ulong lse) => AttentionLse<TN>(a, q, k, v, o, lse);

    string IAttention<ulong>.AttentionBackward<TN>(AttentionArgs a, ulong q, ulong k, ulong v, ulong dO, ulong dq, ulong dk, ulong dv) => AttentionBackward<TN>(a, q, k, v, dO, dq, dk, dv);

    string IOptimizers<ulong>.Optimizer<TN>(OptimizerArgs a, ulong p, ulong g, ulong m, ulong v, int n) => Optimizer<TN>(a, p, g, m, v, n);

    string IConvolution<ulong>.ConvBackward<TN>(ConvArgs a, ulong x, ulong w, ulong dy, ulong dx, ulong dw, ulong db) => ConvBackward<TN>(a, x, w, dy, dx, dw, db);

    string ICtc<ulong>.Ctc<TN>(CtcArgs a, ulong x, ulong loss, ulong grad) => Ctc<TN>(a, x, loss, grad);

    string IRouting<ulong, ulong>.Sample<TN>(ulong x, int rows, int cols, ulong u, ulong picks) => Sample<TN>(x, rows, cols, u, picks);

    string IRouting<ulong, ulong>.CrossEntropy<TN>(ulong x, int rows, int cols, ulong targets, ulong weights, ulong output) => CrossEntropy<TN>(x, rows, cols, targets, weights, output);

    string IBoxes<ulong>.BoxLoss<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong loss, int count, double eps) => BoxLoss<TN>(overlap, predicted, target, loss, count, eps);

    string IBoxes<ulong>.BoxLossBackward<TN>(BoxOverlap overlap, ulong predicted, ulong target, ulong dloss, ulong dpredicted, int count, double eps) =>
        BoxLossBackward<TN>(overlap, predicted, target, dloss, dpredicted, count, eps);

    string IRecurrent<ulong>.LstmCell<TN>(CellArgs a, ulong g, ulong c, ulong hOut, ulong cOut) => LstmCell<TN>(a, g, c, hOut, cOut);

    string IRecurrent<ulong>.LstmCellBackward<TN>(CellArgs a, ulong g, ulong c, ulong dh, ulong dcNext, ulong dg, ulong dc) => LstmCellBackward<TN>(a, g, c, dh, dcNext, dg, dc);

    string IRecurrent<ulong>.GruCell<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong hOut) => GruCell<TN>(a, gx, gh, h, hOut);

    string IRecurrent<ulong>.GruCellBackward<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong dhOut, ulong dgx, ulong dgh, ulong dh) =>
        GruCellBackward<TN>(a, gx, gh, h, dhOut, dgx, dgh, dh);

    string IRecurrent<ulong>.LstmStep<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) =>
        LstmStep<TN>(a, x, h, c, w, u, bias, hOut, cOut);

    string IRecurrent<ulong>.LstmStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcOut,
        ulong dx, ulong dh, ulong dc, ulong dw, ulong du, ulong db) => LstmStepBackward<TN>(a, x, h, c, w, u, bias, dhOut, dcOut, dx, dh, dc, dw, du, db);

    string IRecurrent<ulong>.GruStep<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) =>
        GruStep<TN>(a, x, h, w, u, bias, biasH, hOut);

    string IRecurrent<ulong>.GruStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut,
        ulong dx, ulong dh, ulong dw, ulong du, ulong db, ulong dbh) => GruStepBackward<TN>(a, x, h, w, u, bias, biasH, dhOut, dx, dh, dw, du, db, dbh);

    string IRecurrent<ulong>.LstmSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) =>
        LstmSequence<TN>(a, T, x, h0, c0, w, u, bias, hOut, cOut);

    string IRecurrent<ulong>.LstmSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcLast,
        ulong dx, ulong dh0, ulong dc0, ulong dw, ulong du, ulong db) => LstmSequenceBackward<TN>(a, T, x, h0, c0, w, u, bias, dhOut, dcLast, dx, dh0, dc0, dw, du, db);

    string IRecurrent<ulong>.GruSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) =>
        GruSequence<TN>(a, T, x, h0, w, u, bias, biasH, hOut);

    string IRecurrent<ulong>.GruSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut,
        ulong dx, ulong dh0, ulong dw, ulong du, ulong db, ulong dbh) => GruSequenceBackward<TN>(a, T, x, h0, w, u, bias, biasH, dhOut, dx, dh0, dw, du, db, dbh);

    // ---------------------------------------------------------------- the product with the bounds measured on the operands

    string IProducts<ulong>.MatMul<TN>(ulong a, ulong b, ulong c, int m, int n, int k)
    {
        var ((aLow, aHigh), (bLow, bHigh)) = RangePair<TN>(a, m * k, b, k * n);
        return MatMul<TN>(a, b, c, m, n, k, new Bounds(aLow, aHigh, bLow, bHigh, k));
    }
}
