// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The CPU as a device: the kernels over the device contract (Abstractions/), on the host's memory — byte arrays of stored
// values, int arrays of indices; every operation runs on the Team and is done when it returns.
namespace Qiyas;

/// <summary>The CPU device.</summary>
public static class Cpu
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

    // ---------------------------------------------------------------- layouts

    string ILayouts<byte[], int[]>.Layout<TN>(LayoutArgs k, byte[] x, int[]? index, byte[] output, byte[]? x2) => Layout<TN>(new LayoutArgs
    {
        Name = k.Name, X = x, X2 = x2, Index = index, P = k.P, Output = output, Count = k.Count, InputCount = k.InputCount,
    });

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

    string IProducts<byte[]>.MatMulBackward<TN>(byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) => MatMulBackward<TN>(a, b, dc, da, db, m, n, k);

    string IProducts<byte[]>.MatMulViewBlocks<TN>(ProductView v, byte[] a, byte[] b, byte[] c) => MatMulViewBlocks<TN>(v, a, b, c);

    bool IProducts<byte[]>.BlockProductAvailable => BlockProductAvailable;

    // ---------------------------------------------------------------- attention

    string IAttention<byte[]>.Attention<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => Attention<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionComposed<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => AttentionComposed<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionRows<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => AttentionRows<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionDecode<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => AttentionDecode<TN>(a, q, k, v, o);

    string IAttention<byte[]>.AttentionBackward<TN>(AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) => AttentionBackward<TN>(a, q, k, v, dO, dq, dk, dv);

    // ---------------------------------------------------------------- optimizers, convolution, ctc

    string IOptimizers<byte[]>.Optimizer<TN>(OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) => Optimizer<TN>(a, p, g, m, v, n);

    string IConvolution<byte[]>.Conv<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) => Conv<TN>(a, x, w, y);

    string IConvolution<byte[]>.ConvComposed<TN>(ConvArgs a, byte[] x, byte[] w, byte[] y) => ConvComposed<TN>(a, x, w, y);

    string IConvolution<byte[]>.ConvBackward<TN>(ConvArgs a, byte[] x, byte[] w, byte[] dy, byte[] dx, byte[] dw) => ConvBackward<TN>(a, x, w, dy, dx, dw);

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

    string IRouting<byte[], int[]>.CrossEntropy<TN>(byte[] x, int rows, int cols, int[] targets, byte[] output) => CrossEntropy<TN>(x, rows, cols, targets, output);

    void IRouting<byte[], int[]>.Route(int[] index, int count, int buckets, int[] counts, int[] offsets, int[] order) =>
        Route(count == index.Length ? index : index[..count], buckets, counts, offsets, order);

    // ---------------------------------------------------------------- the lanes

    /// <summary>The name of TN's float lane.</summary>
    internal static string FloatLane<TN>() where TN : struct, IQiyasFloat<TN>
    {
        var probe = new FloatLaneName();
        TN.Float(ref probe);
        return probe.Name;
    }

    private struct FloatLaneName : IFloatLaneVisitor
    {
        public string Name;

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane> => Name = TL.Name;
    }
}
