// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The bench contract. A subject is one implementation under measurement; the harness knows nothing about how it
// computes. It hands the subject stored bytes of a named type (made by Qiyas's proven conversions, so every subject
// sees the same bits), times Run, reads the output after Collect, and scores it against float64 math.
//
// To add an implementation: one class implementing IBenchSubject with a parameterless constructor, anywhere in this
// assembly or in an assembly loaded next to it; it is found by reflection and becomes a column of the report. Each
// method returns null for anything the implementation does not do (the report shows —); never an approximation.
//
//   - inputs and outputs are stored bytes of the named type (Registry.Names), values packed as the type packs them;
//   - Run is what is timed: it must include the device synchronization that makes the result ready;
//   - Collect writes the result into the output bytes the subject was given; Dispose frees what Prepare took;
//   - Lane says what the subject computed in (a lane, a kernel name) and is shown beside its time.
namespace Qiyas.Bench;

public enum Device
{
    Cpu,
    Cuda,
    Vulkan,
}

/// <summary>One prepared operation.</summary>
public interface IBenchRun : IDisposable
{
    void Run();

    void Collect();

    string Lane { get; }
}

/// <summary>An implementation under measurement.</summary>
public interface IBenchSubject
{
    /// <summary>The report's column.</summary>
    string Name { get; }

    /// <summary>A line about the subject on this device (threads, kernels, what it reports), or null.</summary>
    string? Describe(Device device) => null;

    /// <summary>c = a + b over n values.</summary>
    IBenchRun? Add(Device device, string type, byte[] a, byte[] b, byte[] c, int n);

    /// <summary>C = A·B, A m×k and B k×n row-major, C m×n.</summary>
    IBenchRun? MatMul(Device device, string type, byte[] a, byte[] b, byte[] c, int m, int n, int k);

    /// <summary>Softmax over each row of x (rows × cols) into y.</summary>
    IBenchRun? Softmax(Device device, string type, byte[] x, byte[] y, int rows, int cols);

    /// <summary>y = x / sqrt(mean(x²) + eps) over each row.</summary>
    IBenchRun? RmsNorm(Device device, string type, byte[] x, byte[] y, int rows, int cols, double eps);

    /// <summary>output = op(x, y, z; a, b) over n values (Ops.All names the formulas; inputs past op.Inputs are null).</summary>
    IBenchRun? Map(Device device, MapOp op, string type, byte[]? x, byte[]? y, byte[]? z, byte[] output, int n, double a, double b) => null;

    /// <summary>A reduction (sum, sumsq, max, min, argmax) of x (rows × cols) along the axis: the result is written as the
    /// subject's lane values (Lane names it: f32, f64, s32, u32, s64, u64, or "s32 index"); output holds 8 bytes per result.</summary>
    IBenchRun? Reduce(Device device, string op, Axis axis, string type, byte[] x, int rows, int cols, byte[] output) => null;

    /// <summary>output[r][c] = op(x[r][c], v[c]) for a two-input formula (add, sub, mul, div, max, min).</summary>
    IBenchRun? MapRow(Device device, MapOp op, string type, byte[] x, byte[] v, byte[] output, int rows, int cols) => null;

    /// <summary>output (cols × rows) = xᵀ.</summary>
    IBenchRun? Transpose(Device device, string type, byte[] x, byte[] output, int rows, int cols) => null;

    /// <summary>y = A·x (A rows × cols, x cols) or, transposed, y = Aᵀ·x (x rows); y stored as the type.</summary>
    IBenchRun? Gemv(Device device, string type, byte[] a, byte[] x, byte[] y, int rows, int cols, bool transposed) => null;

    /// <summary>A row operation (RowOps.All) over x (rows × cols): the second row x2, the per-column vectors g and bt, the
    /// per-row scalars s and positions pos are null when the operation has none; y2 is the second output when it has one.</summary>
    IBenchRun? RowOp(Device device, RowOp op, string type, byte[] x, byte[]? x2, byte[]? g, byte[]? bt, byte[]? s, int[]? pos, byte[] y, byte[]? y2, int rows, int cols, double a, double b) => null;

    /// <summary>dgamma[c] = Σ_r dy·x̂, dbeta[c] = Σ_r dy (x̂ by rms or, centered, layer statistics with eps): output holds dgamma then
    /// dbeta as the subject's lane values (8 bytes a slot, 2·cols slots).</summary>
    IBenchRun? NormParamsBwd(Device device, string type, byte[] x, byte[] dy, byte[] output, int rows, int cols, double eps, bool centered) => null;

    /// <summary>A layout operation (LayoutArgs.Names): the input, the index array and the parameters in k; the output in k.Output (stored as the type).</summary>
    IBenchRun? Layout(Device device, string type, LayoutArgs k) => null;

    /// <summary>C = op(A)·op(B) over a view's batches (ProductView: transposed operands, batch strides); c stored as the type.</summary>
    IBenchRun? MatMulView(Device device, string type, ProductView v, byte[] a, byte[] b, byte[] c) => null;

    /// <summary>dA = dC·Bᵀ and dB = Aᵀ·dC for C = A·B (A m × k, B k × n, dC m × n), both stored as the type.</summary>
    IBenchRun? MatMulBackward(Device device, string type, byte[] a, byte[] b, byte[] dc, byte[] da, byte[] db, int m, int n, int k) => null;

    /// <summary>O = softmax(scale·Q·Kᵀ, masked)·V per head for rows of queries (AttentionArgs); decode: a few queries against a long cache.</summary>
    IBenchRun? Attention(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o, bool decode) => null;

    /// <summary>O for rows of queries by a fused kernel (no Lq × Lk scores in memory), when the subject has one.</summary>
    IBenchRun? AttentionFused(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => null;

    /// <summary>dQ, dK, dV from dO.</summary>
    IBenchRun? AttentionBackward(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] dO, byte[] dq, byte[] dk, byte[] dv) => null;

    /// <summary>The source's rows (c.Heads × steps × c.D of the type) into the cache (of cacheType) at the position
    /// (ICache.CacheWrite); with scales (of the type) given: the rows' codes and their scales.</summary>
    IBenchRun? CacheWrite(Device device, string type, string cacheType, CacheView c, int steps, int position, byte[] source, byte[] cache, byte[]? scales) => null;

    /// <summary>O for the queries at the position against caches of cacheType (ICache.AttentionCached), their row scales (of
    /// the type) when given.</summary>
    IBenchRun? AttentionCached(Device device, string type, string cacheType, AttentionArgs a, int position, byte[] q, byte[] k, byte[]? kScales, byte[] v, byte[]? vScales, byte[] o) => null;

    /// <summary>The rotary embedding by tables (IRows.Rope): x's rows normalized (a gain given) and rotated into y (the type),
    /// y's rows a head at a time when a.HeadsFirst; cos and sin the float lane's storage.</summary>
    IBenchRun? Rope(Device device, string type, RopeArgs a, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] y) => null;

    /// <summary>A layer's rows normalized, rotated and written into the cache (of cacheType) at the position (ICache.CacheWriteRope).</summary>
    IBenchRun? CacheWriteRope(Device device, string type, string cacheType, CacheView c, RopeArgs a, int position, byte[] x, byte[]? gain, byte[] cos, byte[] sin, int[] positions, byte[] cache, byte[]? scales) => null;

    /// <summary>O for Q, K, V and O a step at a time (AttentionArgs.StepsFirst).</summary>
    IBenchRun? AttentionStepsFirst(Device device, string type, AttentionArgs a, byte[] q, byte[] k, byte[] v, byte[] o) => null;

    /// <summary>The mask of rows decoder steps over capacity positions at the position (ICache.DecoderMask), as the type.</summary>
    IBenchRun? DecoderMask(Device device, string type, int rows, int capacity, int window, int position, byte[] mask) => null;

    /// <summary>The codes (of codesType) and scales (of the type) of a scaled weight's values (IProducts.Quantize, no zero points).</summary>
    IBenchRun? Quantize(Device device, string type, string codesType, ScaledWeight w, byte[] values, byte[] codes, byte[] scales) => null;

    /// <summary>An optimizer step (OptimizerArgs.Names) over n values: p (updated in place) and g as the type, the moments m, v as
    /// the subject's lane storage (float32 or float64 bytes; v null for one moment), updated in place.</summary>
    IBenchRun? Optimizer(Device device, string type, OptimizerArgs a, byte[] p, byte[] g, byte[] m, byte[]? v, int n) => null;

    /// <summary>y = conv(x, w) for one image (ConvArgs); backward: dx and dw from dy when dx is given (y is then dy).</summary>
    IBenchRun? Conv(Device device, string type, ConvArgs a, byte[] x, byte[] w, byte[] y, byte[]? dx, byte[]? dw) => null;

    /// <summary>top-k of each row of x (rows × cols): values (rows × k, as the type) and indices (s32), largest first.</summary>
    IBenchRun? TopK(Device device, string type, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) => null;

    /// <summary>The rows' scores with the tokens of each row's window penalized (IDecoding.Penalize): the rings (a.Rows ×
    /// a.Capacity) at the length, output a.Rows × a.Cols as the type.</summary>
    IBenchRun? Penalize(Device device, string type, PenaltyArgs a, byte[] x, int[] history, int length, byte[] output) => null;

    /// <summary>A column of each row drawn from its kept weights (IDecoding.SampleFiltered) at a step: picks (s32) and each
    /// pick's probability (rows, as the type).</summary>
    IBenchRun? SampleFiltered(Device device, string type, SamplingArgs a, byte[] x, int step, int[] picks, byte[] probability) => null;

    /// <summary>top-k of each row as log-probabilities (IRouting.LogSoftmaxTopK): values (rows × k, as the type) and indices.</summary>
    IBenchRun? LogSoftmaxTopK(Device device, string type, byte[] x, int rows, int cols, int k, byte[] values, int[] indices) => null;

    /// <summary>loss[r] = log Σ exp(x[r]) − x[r][targets[r]] as the type (the cross-entropy of each row's logits; null: the
    /// bench composes log_softmax, pick and neg).</summary>
    IBenchRun? CrossEntropy(Device device, string type, byte[] x, int rows, int cols, int[] targets, byte[] loss) => null;

    /// <summary>output[r] = x[r][index[r]] as the type.</summary>
    IBenchRun? Pick(Device device, string type, byte[] x, int rows, int cols, int[] index, byte[] output) => null;

    /// <summary>The ctc loss per sequence (B values as the type) from logits (T × B × C, CtcArgs); with grad given, the
    /// gradient with respect to the logits too (T × B × C as the type).</summary>
    IBenchRun? Ctc(Device device, string type, CtcArgs a, byte[] logits, byte[] loss, byte[]? grad) => null;

    /// <summary>picks[r] = the first column whose running sum of row r reaches u[r] · Σ row (u as the type, one per row).</summary>
    IBenchRun? Sample(Device device, string type, byte[] x, int rows, int cols, byte[] u, int[] picks) => null;

    /// <summary>counts[b], offsets[b + 1], order (the sources grouped by bucket, in source order) from n indices into buckets.</summary>
    IBenchRun? Route(Device device, int[] index, int buckets, int[] counts, int[] offsets, int[] order) => null;

    /// <summary>An LSTM sequence of T steps (x: T × B × D; h0, c0; h, c out: T × B × H), or with dx given its backward (hOut is then dh
    /// per step, cOut the last step's dc; dx, dh0, dc0, dw, du, db the gradients).</summary>
    IBenchRun? LstmSequence(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] b, byte[] hOut, byte[] cOut,
        byte[]? dx, byte[]? dh0, byte[]? dc0, byte[]? dw, byte[]? du, byte[]? db) => null;

    /// <summary>C = A·Bᵀ (A m × k, B n × k, C m × n) by the block-scaled product of two Q8_0 / Q4_0 operands (k in whole blocks).</summary>
    IBenchRun? MatMulBlocks(Device device, string type, int m, int n, int k, byte[] a, byte[] b, byte[] c) => null;

    /// <summary>A GRU sequence of T steps (x: T × B × D; h0; W, U, b, bh as the step's; h out: T × B × H).</summary>
    IBenchRun? GruSequence(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] hOut) => null;

    /// <summary>The GRU sequence's backward: from dhOut (T × B × H, the loss's gradient at every step) the gradients dx (T × B × D),
    /// dh0 (B × H), dw, du, db, dbh (summed over the steps).</summary>
    IBenchRun? GruSequenceBackward(Device device, string type, CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] b, byte[] bh, byte[] dhOut,
        byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh) => null;

    /// <summary>A recurrent step (CellArgs: lstm or gru): h' (and c' for LSTM) from x, h (, c) and the weights W (GH × D), U (GH × H),
    /// b (GH), bh (GRU's second bias, GH); when dx is given the backward pass instead: hOut (, cOut) are then dh', dc', and the
    /// gradients go to dx, dh, dc, dw, du, db, dbh.</summary>
    IBenchRun? Recurrent(Device device, string type, CellArgs a, byte[] x, byte[] h, byte[]? c, byte[] w, byte[] u, byte[] b, byte[]? bh, byte[] hOut, byte[]? cOut,
        byte[]? dx, byte[]? dh, byte[]? dc, byte[]? dw, byte[]? du, byte[]? db, byte[]? dbh) => null;
}

/// <summary>Lane values as float64 (what a reduction returns).</summary>
public static class LaneValues
{
    public static double[] Decode(string lane, byte[] bytes, int count)
    {
        var d = new double[count];
        // A subject may add a note after the lane name ("f32; converted on the device"): the name is what is decoded.
        int note = lane.IndexOf(';');
        if (note > 0) lane = lane[..note].Trim();
        switch (lane)
        {
            case "s8": for (int i = 0; i < count; i++) d[i] = (sbyte)bytes[i]; break;
            case "u8": for (int i = 0; i < count; i++) d[i] = bytes[i]; break;
            case "s16": for (int i = 0; i < count; i++) d[i] = BitConverter.ToInt16(bytes, i * 2); break;
            case "u16": for (int i = 0; i < count; i++) d[i] = BitConverter.ToUInt16(bytes, i * 2); break;
            case "f32" or "f32 index": for (int i = 0; i < count; i++) d[i] = BitConverter.ToSingle(bytes, i * 4); break;
            case "f64": for (int i = 0; i < count; i++) d[i] = BitConverter.ToDouble(bytes, i * 8); break;
            case "s32" or "s8dp" or "s16dp" or "s32 index": for (int i = 0; i < count; i++) d[i] = BitConverter.ToInt32(bytes, i * 4); break;
            case "u32" or "u8dp" or "u16dp": for (int i = 0; i < count; i++) d[i] = BitConverter.ToUInt32(bytes, i * 4); break;
            case "s64": for (int i = 0; i < count; i++) d[i] = BitConverter.ToInt64(bytes, i * 8); break;
            case "u64": for (int i = 0; i < count; i++) d[i] = BitConverter.ToUInt64(bytes, i * 8); break;
            default: throw new ArgumentException(lane);
        }

        return d;
    }
}

/// <summary>An IBenchRun from delegates.</summary>
public sealed class BenchRun(Action run, Action? collect = null, Action? dispose = null, Func<string>? lane = null) : IBenchRun
{
    public void Run() => run();

    public void Collect() => collect?.Invoke();

    public string Lane => lane?.Invoke() ?? "";

    public void Dispose() => dispose?.Invoke();
}

/// <summary>The GPU devices the subjects share (each created on first use): CUDA, and Vulkan.</summary>
public static class Devices
{
    private static GpuOps? s_ops;
    private static Adad.Vulkan.VulkanOps? s_vk;
    private static bool? s_hasVulkan;
    private static string s_vulkanReason = "";

    public static bool HasCuda => Driver.Available;

    public static GpuOps Ops => s_ops ??= new GpuOps(new Gpu());

    public static Gpu Gpu => Ops.Gpu;

    /// <summary>Whether this machine has a Vulkan device Adad can use (VulkanReason says why not).</summary>
    public static bool HasVulkan => s_hasVulkan ??= Adad.Vulkan.Vulkan.TryAvailable(out s_vulkanReason);

    public static string VulkanReason => s_vulkanReason;

    public static Adad.Vulkan.VulkanOps Vk => s_vk ??= new Adad.Vulkan.VulkanOps(new Adad.Vulkan.Vulkan());

    /// <summary>The device a GPU subject runs on.</summary>
    public static IDevice<ulong, ulong> Of(Device device) => device == Device.Vulkan ? Vk : Ops;
}
