// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The CPU as a device: the kernels over the device contract (Abstractions/), on the host's memory — byte arrays of stored
// values, int arrays of indices; every operation runs on the Team and is done when it returns.
using Adad.Abstraction;

namespace Adad.Cpu;

/// <summary>The CPU device.</summary>
public static class CpuDevice
{
    public static IDevice<byte[], int[]> Device { get; } = new Kernels();
}

internal sealed partial class Kernels : IDevice<byte[], int[]>
{
    // ---------------------------------------------------------------- memory

    byte[] IMemory<byte[], int[]>.Rent(long bytes) => new byte[bytes];

    void IMemory<byte[], int[]>.Return(byte[] buffer) { }

    void IMemory<byte[], int[]>.Upload(ReadOnlySpan<byte> source, byte[] destination) => source.CopyTo(destination);

    void IMemory<byte[], int[]>.Download(byte[] source, Span<byte> destination) => source.AsSpan(0, destination.Length).CopyTo(destination);

    void IMemory<byte[], int[]>.Copy(byte[] source, byte[] destination, long bytes) => Array.Copy(source, destination, bytes);

    int[] IMemory<byte[], int[]>.RentIndices(int count) => new int[count];

    void IMemory<byte[], int[]>.ReturnIndices(int[] indices) { }

    void IMemory<byte[], int[]>.UploadIndices(ReadOnlySpan<int> source, int[] destination) => source.CopyTo(destination);

    void IMemory<byte[], int[]>.DownloadIndices(int[] source, Span<int> destination) => source.AsSpan(0, destination.Length).CopyTo(destination);

    void IMemory<byte[], int[]>.Synchronize() { }

    string IDevice<byte[], int[]>.Report() => Report();

    string IDevice<byte[], int[]>.Measured() => Measured();

    // ---------------------------------------------------------------- element-wise

    void IElementwise<byte[]>.Add<TN>(byte[] a, byte[] b, byte[] c, int n)
    {
        var add = new AddVisitor(a, b, c, n);
        TN.Elementwise(ref add);
    }

    string IElementwise<byte[]>.Map<TN>(MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) => Map<TN>(op.Name, x, y, z, output, n, a, b);

    string IElementwise<byte[]>.MapFloat<TN>(MapOp op, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) => MapFloat<TN>(op.Name, x, y, z, output, n, a, b);

    string IElementwise<byte[]>.MapRow<TN>(MapOp op, byte[] x, byte[] v, byte[] output, int rows, int cols) => MapRow<TN>(op.Name, x, v, output, rows, cols);

    string IElementwise<byte[]>.MapRowFloat<TN>(MapOp op, byte[] x, byte[] v, byte[] output, int rows, int cols) => MapRowFloat<TN>(op.Name, x, v, output, rows, cols);

    string IElementwise<byte[]>.Dropout<TN>(byte[] x, byte[]? residual, byte[] output, int n, double p, ulong seed) => Dropout<TN>(x, residual, output, n, p, seed);

    string IElementwise<byte[]>.Convert<TFrom, TTo>(byte[] x, byte[] output, int n) => Convert<TFrom, TTo>(x, output, n);

    // ---------------------------------------------------------------- reductions

    string IReductions<byte[]>.Reduce<TN>(string op, Axis axis, byte[] x, int rows, int cols, byte[] output)
    {
        var (result, lane) = Reduce<TN>(op, axis, x, rows, cols);
        result.CopyTo(output.AsSpan());
        return lane;
    }

    string IReductions<byte[]>.Transpose<TN>(byte[] x, byte[] output, int rows, int cols) => Transpose<TN>(x, output, rows, cols);

    string IReductions<byte[]>.Gemv<TN>(byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) => Gemv<TN>(a, x, y, rows, cols, transposed);

    // ---------------------------------------------------------------- rows

    string IRows<byte[], int[]>.Softmax<TN>(byte[] x, byte[] y, int rows, int cols)
    {
        var softmax = new SoftmaxVisitor(x, y, rows, cols);
        TN.Float(ref softmax);
        return FloatLane<TN>();
    }

    string IRows<byte[], int[]>.RmsNorm<TN>(byte[] x, byte[] y, int rows, int cols, double eps)
    {
        var rms = new RmsNormVisitor(x, y, rows, cols, eps);
        TN.Float(ref rms);
        return FloatLane<TN>();
    }

    string IRows<byte[], int[]>.RowOp<TN>(RowOp op, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) =>
        RowOp<TN>(op, x, x2, g, bt, s, pos, y, y2, rows, cols, a, b);

    string IRows<byte[], int[]>.NormParamsBwd<TN>(byte[] x, byte[] dy, byte[] dgamma, byte[] dbeta, int rows, int cols, double eps, bool centered) =>
        NormParamsBwd<TN>(x, dy, dgamma, dbeta, rows, cols, eps, centered);

    string IRows<byte[], int[]>.GroupNorm<TN>(byte[] x, byte[]? gamma, byte[]? beta, byte[]? mean, byte[]? variance, byte[] y, GroupShape shape, double eps, bool given) =>
        GroupNorm<TN>(x, gamma, beta, mean, variance, y, shape, eps, given);

    string IRows<byte[], int[]>.GroupNormBwd<TN>(byte[] x, byte[] dy, byte[]? gamma, byte[]? mean, byte[]? variance, byte[]? dx, byte[]? dgamma, byte[]? dbeta, GroupShape shape, double eps, bool given) =>
        GroupNormBwd<TN>(x, dy, gamma, mean, variance, dx, dgamma, dbeta, shape, eps, given);

    string IRows<byte[], int[]>.Rope<TN>(RopeArgs a, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] y) => Rope<TN>(a, x, gain, cos, sin, positions, y);

    // ---------------------------------------------------------------- layouts

    string ILayouts<byte[], int[]>.Layout<TN>(LayoutArgs k, byte[] x, int[]? index, byte[] output, byte[]? x2) => Layout<TN>(new LayoutArgs
    {
        Name = k.Name, X = x, X2 = x2, Index = index, P = k.P, Output = output, Count = k.Count, InputCount = k.InputCount,
    });

    string ILayouts<byte[], int[]>.Resample<TIn, TN>(ResampleArgs a, byte[] x, int[] bounds, byte[] weights, byte[]? values, byte[] y) =>
        Resample<TIn, TN>(a, x, bounds, weights, values, y);

    // ---------------------------------------------------------------- products

    string IProducts<byte[]>.MatMul<TN>(byte[] a, byte[] b, byte[] c, int m, int n, int k)
    {
        var (aLow, aHigh) = TN.Range(a, m * k);
        var (bLow, bHigh) = TN.Range(b, k * n);
        var product = new MatMulVisitor(a, b, c, m, n, k);
        TN.Sum(ref product, new Bounds(aLow, aHigh, bLow, bHigh, k));
        return product.Lane;
    }

    string IProducts<byte[]>.MatMulView<TN>(ProductView v, byte[] a, byte[] b, byte[] c) => MatMulView<TN>(v, a, b, c);

    string IProducts<byte[]>.MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) => MatMulEpilogue<TN>(v, e, a, b, c, t);

    string IProducts<byte[]>.MatMulScaled<TN, TW>(ScaledWeight w, int m, ProductEpilogue e, byte[] x, byte[] codes, byte[] scales, byte[]? zeros, byte[] y, ProductTerms<byte[]> t) =>
        MatMulScaled<TN, TW>(w, m, e, x, codes, scales, zeros, y, t);

    string IProducts<byte[]>.ExpandScaled<TN, TW>(ScaledWeight w, byte[] codes, byte[] scales, byte[]? zeros, byte[] output) => ExpandScaled<TN, TW>(w, codes, scales, zeros, output);

    string IProducts<byte[]>.Quantize<TN, TW>(ScaledWeight w, byte[] values, byte[] codes, byte[] scales, byte[]? zeros) => Quantize<TN, TW>(w, values, codes, scales, zeros);

    // ---------------------------------------------------------------- experts

    string IExperts<byte[], int[]>.RouteTopK<TN>(byte[] p, int tokens, int experts, int k, bool normalize, byte[] weights, int[] offsets, int[] tokenOfRow, int[] slotRow) =>
        RouteTopK<TN>(p, tokens, experts, k, normalize, weights, offsets, tokenOfRow, slotRow);

    string IExperts<byte[], int[]>.ExpertMatMul<TN>(ExpertView v, byte[] x, int[] offsets, byte[] weights, byte[]? bias, byte[] y) => ExpertMatMul<TN>(v, x, offsets, weights, bias, y);

    string IExperts<byte[], int[]>.ExpertGatedPair<TN>(ExpertView v, string gate, byte[] x, int[] offsets, byte[] gateWeights, byte[]? gateBias, byte[] upWeights, byte[]? upBias, byte[] hidden) =>
        ExpertGatedPair<TN>(v, gate, x, offsets, gateWeights, gateBias, upWeights, upBias, hidden);

    string IProducts<byte[]>.MatMulBackward<TN>(byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) => MatMulBackward<TN>(a, b, dc, da, db, m, n, k);

    string IProducts<byte[]>.MatMulViewBlocks<TN>(ProductView v, byte[] a, byte[] b, byte[] c) => MatMulViewBlocks<TN>(v, a, b, c);

    bool IProducts<byte[]>.BlockProductAvailable => BlockProductAvailable;

    // ---------------------------------------------------------------- attention

    string IAttention<byte[]>.Attention<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => Attention<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionComposed<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => AttentionComposed<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionRows<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => AttentionRows<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionDecode<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => AttentionDecode<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionLse<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, byte[] lse) => AttentionLse<TN>(a, q, k, v, o, lse);

    string IAttention<byte[]>.AttentionBackward<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) => AttentionBackward<TN>(a, q, k, v, dO, dq, dk, dv);

    // ---------------------------------------------------------------- caches

    string ICache<byte[], int[]>.CacheWrite<TN, TC>(CacheView c, int steps, int[] position, byte[] source, byte[] cache, byte[]? scales) =>
        CacheWrite<TN, TC>(c, steps, position, source, cache, scales);

    string ICache<byte[], int[]>.CacheWriteRope<TN, TC>(CacheView c, RopeArgs a, int[] position, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] cache, byte[]? scales) =>
        CacheWriteRope<TN, TC>(c, a, position, x, gain, cos, sin, positions, cache, scales);

    string ICache<byte[], int[]>.AttentionCached<TN, TC>(AttentionArgs a, int[] position, byte[] q, byte[] k, byte[]? kScales, byte[] v, byte[]? vScales, byte[] o) =>
        AttentionCached<TN, TC>(a, position, q, k, kScales, v, vScales, o);

    string ICache<byte[], int[]>.DecoderMask<TN>(int rows, int capacity, int window, int[] position, byte[] mask) => DecoderMask<TN>(rows, capacity, window, position, mask);

    void ICache<byte[], int[]>.PositionAdvance(int[] position, int steps) => PositionAdvance(position, steps);

    // ---------------------------------------------------------------- optimizers, convolution, ctc

    string IOptimizers<byte[]>.Optimizer<TN>(OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) => Optimizer<TN>(a, p, g, m, v, n);

    string IConvolution<byte[]>.Conv<TN>(ConvArgs a, byte[] x, byte[] w, byte[]? bias, byte[] y) => Conv<TN>(a, x, w, bias, y);

    string IConvolution<byte[]>.ConvComposed<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) => ConvComposed<TN>(a, x, w, y);

    string IConvolution<byte[]>.ConvBackward<TN>(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw, byte[]? db) => ConvBackward<TN>(a, x, w, dy, dx, dw, db);

    string ICtc<byte[]>.Ctc<TN>(CtcArgs a, byte[] x, byte[] loss, byte[]? grad) => Ctc<TN>(a, x, loss, grad);

    // ---------------------------------------------------------------- recurrent

    void IRecurrent<byte[]>.CellGates<TN>(CellArgs a, byte[] x, byte[] w, byte[]? h, byte[]? u, byte[] bias, byte[] g) => CellGates<TN>(a, x, w, h, u, bias, g);

    string IRecurrent<byte[]>.LstmCell<TN>(CellArgs a, byte[] g, byte[] c, byte[] hOut, byte[] cOut) => LstmCell<TN>(a, g, c, hOut, cOut);

    string IRecurrent<byte[]>.LstmCellBackward<TN>(CellArgs a, byte[] g, byte[] c, byte[] dh, byte[] dcNext, byte[] dg, byte[] dc) => LstmCellBackward<TN>(a, g, c, dh, dcNext, dg, dc);

    string IRecurrent<byte[]>.GruCell<TN>(CellArgs a, byte[] gx, byte[] gh, byte[] h, byte[] hOut) => GruCell<TN>(a, gx, gh, h, hOut);

    string IRecurrent<byte[]>.GruCellBackward<TN>(CellArgs a, byte[] gx, byte[] gh, byte[] h, byte[] dhOut, byte[] dgx, byte[] dgh, byte[] dh) =>
        GruCellBackward<TN>(a, gx, gh, h, dhOut, dgx, dgh, dh);

    string IRecurrent<byte[]>.LstmStep<TN>(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut) =>
        LstmStep<TN>(a, x, h, c, w, u, bias, hOut, cOut);

    string IRecurrent<byte[]>.LstmStepBackward<TN>(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcOut,
        byte[] dx, byte[] dh, byte[] dc, byte[] dw, byte[] du, byte[] db) => LstmStepBackward<TN>(a, x, h, c, w, u, bias, dhOut, dcOut, dx, dh, dc, dw, du, db);

    string IRecurrent<byte[]>.GruStep<TN>(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut) =>
        GruStep<TN>(a, x, h, w, u, bias, biasH, hOut);

    string IRecurrent<byte[]>.GruStepBackward<TN>(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut,
        byte[] dx, byte[] dh, byte[] dw, byte[] du, byte[] db, byte[] dbh) => GruStepBackward<TN>(a, x, h, w, u, bias, biasH, dhOut, dx, dh, dw, du, db, dbh);

    string IRecurrent<byte[]>.LstmSequence<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut) =>
        LstmSequence<TN>(a, T, x, h0, c0, w, u, bias, hOut, cOut);

    string IRecurrent<byte[]>.LstmSequenceBackward<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcLast,
        byte[] dx, byte[] dh0, byte[] dc0, byte[] dw, byte[] du, byte[] db) => LstmSequenceBackward<TN>(a, T, x, h0, c0, w, u, bias, dhOut, dcLast, dx, dh0, dc0, dw, du, db);

    string IRecurrent<byte[]>.GruSequence<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut) =>
        GruSequence<TN>(a, T, x, h0, w, u, bias, biasH, hOut);

    string IRecurrent<byte[]>.GruSequenceBackward<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut,
        byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh) => GruSequenceBackward<TN>(a, T, x, h0, w, u, bias, biasH, dhOut, dx, dh0, dw, du, db, dbh);

    // ---------------------------------------------------------------- routing

    string IRouting<byte[], int[]>.TopK<TN>(byte[] x, int rows, int cols, int k, byte[] values, int[] indices) => TopK<TN>(x, rows, cols, k, values, indices);

    string IRouting<byte[], int[]>.Pick<TN>(byte[] x, int rows, int cols, int[] index, byte[] output) => Pick<TN>(x, rows, cols, index, output);

    string IRouting<byte[], int[]>.Sample<TN>(byte[] x, int rows, int cols, byte[] u, int[] picks) => Sample<TN>(x, rows, cols, u, picks);

    string IRouting<byte[], int[]>.CrossEntropy<TN>(byte[] x, int rows, int cols, int[] targets, byte[]? weights, byte[] output) => CrossEntropy<TN>(x, rows, cols, targets, weights, output);

    string IBoxes<byte[]>.BoxLoss<TN>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[] loss, int count, double eps) => BoxLoss<TN>(overlap, predicted, target, loss, count, eps);

    string IBoxes<byte[]>.BoxLossBackward<TN>(BoxOverlap overlap, byte[] predicted, byte[] target, byte[] dloss, byte[] dpredicted, int count, double eps) =>
        BoxLossBackward<TN>(overlap, predicted, target, dloss, dpredicted, count, eps);

    void IRouting<byte[], int[]>.Route(int[] index, int count, int buckets, int[] counts, int[] offsets, int[] order) =>
        Route(count == index.Length ? index : index[..count], buckets, counts, offsets, order);

    string IRouting<byte[], int[]>.ClassMatch<TN>(byte[] predictions, byte[] targets, byte[] output, int rows, int cols, double threshold) =>
        ClassMatch<TN>(predictions, targets, output, rows, cols, threshold);

    // ---------------------------------------------------------------- the lanes

    /// <summary>The name of TN's float lane.</summary>
    internal static string FloatLane<TN>() where TN : struct, IFloatAdad<TN>
    {
        var probe = new FloatLaneName();
        TN.Float(ref probe);
        return probe.Name;
    }

    private struct FloatLaneName : IFloatLaneVisitor
    {
        public string Name;

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> => Name = TL.Name;
    }
}
