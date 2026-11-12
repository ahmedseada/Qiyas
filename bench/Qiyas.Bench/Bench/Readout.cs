// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The readout of a row: what its operation moves and computes by its own definition (Work), and the memory a subject's
// run held beyond its operands (HostMeter for what a run takes from the runtime on the host; the subject's own count for
// its kept buffers and its device buffers, IBenchRun.Held). With the device's limits measured at the start of the run
// (Limits.cs), each cell reads as a fraction of the bound the hardware sets for the row.
using System.Collections.Concurrent;
using System.Diagnostics.Tracing;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Qiyas.Bench;

/// <summary>What a row's operation moves and computes by its own definition: the bytes read and written at the stored widths
/// (every operand read once and every result written once — the traffic no implementation avoids; lane-storage operands at
/// their lane's width) and its arithmetic operations (each addition, multiplication, division, comparison, square root or
/// elementary function one; a multiply-add two).</summary>
internal readonly record struct Work(double Bytes, double Ops)
{
    /// <summary>Of the bytes, those written (the rest read); null where the row does not say (its bytes then held to the copy's
    /// mix). Measure sets it to the row's output.</summary>
    public double? Written { get; init; }

    public static Work operator +(Work a, Work b) => new(a.Bytes + b.Bytes, a.Ops + b.Ops) { Written = a.Written + b.Written };

    public static Work operator *(double k, Work a) => new(k * a.Bytes, k * a.Ops) { Written = k * a.Written };

    /// <summary>The stored bytes of values of the type.</summary>
    public static double B(string type, long values) => Registry.Bytes(type, values);

    /// <summary>Values of the type read (or written) and nothing computed.</summary>
    public static Work Of(string type, long values) => new(B(type, values), 0);

    /// <summary>The time the row takes at the device's limits: its bytes at the memory's rates for their size (Limits.MoveSeconds:
    /// reads, writes and copies as measured; "copy" where the row does not say which bytes it writes, "memory" where it
    /// does), or its operations at the lane's peak, the larger, and which one binds.</summary>
    public (double Ms, string By) Bound(Limits limits, string lane)
    {
        double move = limits.MoveSeconds(Bytes, Written) * 1e3;
        double ops = limits.Ops(lane) is { } peak && Ops > 0 ? Ops / peak * 1e3 : 0;
        return ops > move ? (ops, lane) : (move, Written is null ? "copy" : "memory");
    }

    public static string Size(double v, string unit)
    {
        string[] prefix = ["", "k", "M", "G", "T"];
        int i = 0;
        while (Math.Abs(v) >= 1000 && i < prefix.Length - 1) { v /= 1000; i++; }
        return string.Create(CultureInfo.InvariantCulture, $"{v:0.##} {prefix[i]}{unit}");
    }
}

/// <summary>Each family's work by its definition.</summary>
internal static class Works
{
    private static double B(string type, long values) => Work.B(type, values);

    /// <summary>An element-wise formula over n values: its inputs and its output, and its operations a value.</summary>
    public static Work Map(MapOp op, string type, long n) => new((op.Inputs + 1) * B(type, n), MapOps(op.Name) * (double)n);

    // A formula's operations a value, by the formula as Map.cs defines it (an elementary function one; sigmoid
    // 1/(1 + exp(−x)) four; gelu's tanh form nine; their derivatives as written).
    private static double MapOps(string name) => name switch
    {
        "copy" or "fill" => 0,
        "neg" or "abs" or "sign" or "square" or "relu" or "add_scalar" or "scale" or "add" or "sub" or "mul" or "div" or "max" or "min" or "where"
            or "relu_bwd" or "where_bwd" or "max_bwd_x" or "max_bwd_y" or "min_bwd_x" or "min_bwd_y" or "sqrt" or "exp" or "expm1" or "log" or "sin"
            or "cos" or "tanh" or "atan" or "erf" or "pow" or "log_bwd" => 1,
        "affine" or "clamp" or "axpy" or "muladd" or "square_bwd" or "abs_bwd" or "clamp_bwd" or "gated_relu" or "gated_relu_bwd_x" or "gated_relu_bwd_y"
            or "leaky_relu" or "dropout" or "exp_bwd" or "sin_bwd" or "cos_bwd" or "leaky_relu_bwd" => 2,
        "inv_sqrt" or "sqrt_bwd" or "pow_bwd" or "softcap" => 3,
        "sigmoid" or "gelu_erf" or "inv_sqrt_bwd" or "tanh_bwd" or "clip_factor" => 4,
        "silu" => 5,
        "gated_silu" or "gated_silu_bwd_y" or "gated_tanh_sigmoid" or "softcap_bwd" => 6,
        "sigmoid_bwd" => 7,
        "gelu" or "silu_bwd" or "gated_tanh_sigmoid_bwd_x" or "gated_tanh_sigmoid_bwd_y" => 9,
        "gated_gelu" or "gated_gelu_bwd_y" or "gated_silu_bwd_x" or "gelu_erf_bwd" or "add_gelu" => 10,
        "gelu_bwd" => 18,
        "gated_gelu_bwd_x" => 19,
        "sigmoid_focal" => 22,
        "sigmoid_focal_bwd" => 36,
        _ => 1,
    };

    /// <summary>A row operation over rows × cols: its inputs and outputs as the case gives them, its operations a value.</summary>
    public static Work Row(RowOp op, RowCase k, string type, int rows, int cols)
    {
        long n = (long)rows * cols;
        double bytes = B(type, n) + (k.X2 is null ? 0 : B(type, n)) + (k.G is null ? 0 : B(type, cols)) + (k.Bt is null ? 0 : B(type, cols))
            + (k.S is null ? 0 : B(type, rows)) + (k.Pos is null ? 0 : 4.0 * rows) + B(type, n) + (op.SecondOut ? B(type, n) : op.RowOut ? B(type, rows) : 0);
        double per = op.Name switch
        {
            "softmax_bwd" or "log_softmax_bwd" => 4,
            "rms_norm_affine" => 4,
            "rms_norm_bwd" or "weight_norm_bwd" => 7,
            "layer_norm" or "scale_mask_softmax" or "scale_mask_softmax_lse" or "weighted_cross_entropy_bwd" => 7,
            "layer_norm_bwd" => 10,
            "weight_norm" => 3,
            "add_rms_norm" or "log_softmax" or "rope" or "cross_entropy_bwd" => 5,
            "add_layer_norm" => 8,
            _ => 5,
        };
        return new(bytes, per * n);
    }

    /// <summary>The reduction of rows × cols along the axis into count lane values (8 bytes each).</summary>
    public static Work Reduce(string op, string type, long values, int count) => new(B(type, values) + 8.0 * count, (op == "sumsq" ? 2 : 1) * (double)values);

    /// <summary>A layout operation: its input(s), index and output at the type; the operations its definition computes.</summary>
    public static Work Layout(string name, LayoutArgs k, string type)
    {
        double bytes = B(type, k.InputCount) + (k.X2 is null ? 0 : B(type, k.SecondCount)) + 4.0 * (k.Index?.Length ?? 0) + B(type, k.Count);
        int[] p = k.P;
        double ops = name switch
        {
            "scatter_add_rows" or "col2im" or "interpolate_nearest_bwd" or "adaptive_maxpool" or "adaptive_avgpool" => k.InputCount,
            "interpolate_bilinear" => 7.0 * k.Count,                                   // four weights times four values, three sums
            "interpolate_bilinear_bwd" => 8.0 * k.InputCount,                          // each gradient into four inputs
            "maxpool" or "avgpool" => (double)p[3] * p[4] * k.Count,                   // a window's comparisons or sums an output
            "maxpool_bwd" or "avgpool_bwd" => (double)p[3] * p[4] * k.InputCount,
            "adaptive_maxpool_bwd" or "adaptive_avgpool_bwd" => k.Count,
            "depthwise_conv" => 2.0 * p[3] * p[4] * k.Count,
            _ => 0,
        };
        return new(bytes, ops);
    }

    /// <summary>C = A·B over m × n × k (a batch of them): the operands and the result once, a multiply-add a term.</summary>
    public static Work Product(string type, long a, long b, long c, double terms) => new(B(type, a) + B(type, b) + B(type, c), 2 * terms);

    /// <summary>Attention: Q, K, V read and O written; per query and key it attends to, the score and the weighted value (a
    /// multiply-add a dimension each) and the softmax's five operations; the backward reads dO and writes dQ, dK, dV and
    /// takes five products a pair (the scores again, dV, dP, dQ, dK).</summary>
    public static Work Attention(AttentionArgs a, string type, bool backward, string? cacheType = null)
    {
        double pairs = 0;
        for (int i = 0; i < a.Lq; i++) pairs += a.Limit(i);
        pairs *= a.Heads;
        string kv = cacheType ?? type;
        double q = B(type, a.QValues), k = B(kv, a.KValues);
        if (backward) return new(3 * q + 3 * k + q, pairs * (10.0 * a.D + 8));
        return new(2 * q + 2 * k, pairs * (4.0 * a.D + 5));
    }

    /// <summary>A convolution: x, w read, y written, a multiply-add a term; the backward reads x, w, dy and writes dx, dw.</summary>
    public static Work Conv(ConvArgs a, string type, bool backward)
    {
        double terms = (double)a.YValues * a.Rows;
        double x = B(type, a.XValues), w = B(type, a.WValues), y = B(type, a.YValues);
        return backward ? new(2 * x + 2 * w + y, 4 * terms) : new(x + w + y, 2 * terms);
    }

    /// <summary>A recurrent cell over T steps: the products x·Wᵀ and h·Uᵀ a step (a multiply-add a term) and the gates' own
    /// operations a hidden value (LSTM: the biases 8, three sigmoids and a tanh 13, the cell and its output 6; GRU: the
    /// biases 6, two sigmoids and a tanh 9, the reset and the blend 8); the backward four products a step and twice the
    /// gates' operations.</summary>
    public static Work Recurrent(CellArgs a, int T, string type, bool backward)
    {
        int g = a.Gates;
        double terms = (double)T * a.B * g * a.H * (a.D + a.H);
        double gates = (double)T * a.B * a.H * (g == 4 ? 27 : 23);
        double x = B(type, (long)T * a.B * a.D), h = B(type, (long)T * a.B * a.H), state = B(type, (long)a.B * a.H);
        double weights = B(type, (long)g * a.H * (a.D + a.H)) + B(type, (long)g * a.H * (g == 3 ? 2 : 1));
        double cells = g == 4 ? 2 : 1;                                                 // LSTM keeps c beside h
        if (!backward) return new(x + weights + cells * state + cells * h, 2 * terms + gates);
        return new(2 * x + 2 * weights + 2 * cells * state + 2 * cells * h, 4 * terms + 2 * gates);
    }

    /// <summary>An optimizer step over n values: p and g as the type, the moments in lane storage of laneBytes, p and the
    /// moments written back; the step's operations a value as OptimizerArgs defines them.</summary>
    public static Work Optimizer(OptimizerArgs a, string type, long n, int laneBytes, bool many)
    {
        double moments = a.Moments * (double)laneBytes * n;
        double per = a.Name switch { "sgd" => 4, "sgd_momentum" => 6, "adam" => 14, _ => 16 };
        if (many) per += 3;                                                            // the gradient's square summed, then scaled by the factor
        return new(3 * B(type, n) + 2 * moments, per * n);
    }

    /// <summary>ctc over T × B × C logits: read once, the loss per sequence written (with the gradient, the gradient too); per
    /// frame of each sequence the logits' log-softmax (five a class) and the alignment's states (a log-sum-exp of up to
    /// three terms and the emission: eight a state; the backward's second pass as many and the gradient's three a class).</summary>
    public static Work Ctc(CtcArgs a, string type, bool grad)
    {
        double states = 0, frames = 0;
        for (int b = 0; b < a.B; b++) { states += (double)a.InputLengths[b] * (2 * a.LabelLengths[b] + 1); frames += a.InputLengths[b]; }
        double bytes = B(type, a.Values) + 4.0 * a.Labels.Length + 8.0 * a.B + B(type, a.B) + (grad ? B(type, a.Values) : 0);
        double ops = 5.0 * frames * a.C + 8 * states + (grad ? 8 * states + 3.0 * frames * a.C : 0);
        return new(bytes, ops);
    }

    /// <summary>The rotary embedding by tables: x read and y written, the tables' entries the rows' positions use (each step's
    /// half row of cosines and of sines, in lane storage), the positions; a pair's rotation six operations.</summary>
    public static Work Rope(RopeArgs a, string type, int laneBytes)
    {
        long n = (long)a.Rows * a.Cols;
        return new(2 * B(type, n) + 2.0 * a.Steps * a.Half * laneBytes + 4.0 * a.Steps, 6.0 * a.Rows * a.Half);
    }
}

/// <summary>The host bytes a run takes from the runtime: the most of the shared array pools' buffers out at once (their rents
/// and returns as the pools report them through their event source, listened to only while a run is measured) plus what
/// the run allocates otherwise (the GC's count less the pools' own new buffers, which are counted once). The GC's count
/// is taken over a run of its own with the events off (an event's report is itself an allocation).</summary>
internal sealed class HostMeter : EventListener
{
    private EventSource? _pools;
    private readonly object _gate = new();
    private readonly ConcurrentDictionary<int, int> _elementBytes = new();
    private long _out, _peak, _pooledNew;
    private bool _registered;

    protected override void OnEventSourceCreated(EventSource source)
    {
        if (source.Name == "System.Buffers.ArrayPoolEventSource") _pools = source;
    }

    protected override void OnEventWritten(EventWrittenEventArgs e)
    {
        if (e.Payload is not { Count: >= 3 } p) return;
        long bytes = Convert.ToInt64(p[1], CultureInfo.InvariantCulture) * ElementBytes(Convert.ToInt32(p[2], CultureInfo.InvariantCulture));
        lock (_gate)
        {
            switch (e.EventId)
            {
                case 1: _out += bytes; _peak = Math.Max(_peak, _out); break;          // rented
                case 2: _pooledNew += bytes; break;                                    // a new buffer made for a rent
                case 3: _out -= bytes; break;                                          // returned
            }
        }
    }

    /// <summary>The run made twice (the GC's count over the first, the pools' over the second), with the host bytes it took.</summary>
    public long Measure(Action run)
    {
        if (!_registered) Register();
        long allocated = GC.GetTotalAllocatedBytes(true);
        run();
        allocated = GC.GetTotalAllocatedBytes(true) - allocated;
        lock (_gate) { _out = 0; _peak = 0; _pooledNew = 0; }
        if (_pools is not null) EnableEvents(_pools, EventLevel.Verbose);
        try { run(); }
        finally
        {
            if (_pools is not null) DisableEvents(_pools);
        }

        lock (_gate) return _peak + Math.Max(0, allocated - _pooledNew);
    }

    // A pool's id is its shared instance's hash: every unmanaged value type of the runtime's primitives and of the loaded
    // Adad and Qiyas assemblies is registered by its element size (an unknown pool's element counted as 8 bytes).
    private int ElementBytes(int pool) => _elementBytes.TryGetValue(pool, out int size) ? size : 8;

    private void Register()
    {
        _registered = true;
        var types = new List<Type> { typeof(byte), typeof(sbyte), typeof(short), typeof(ushort), typeof(int), typeof(uint), typeof(long), typeof(ulong),
            typeof(float), typeof(double), typeof(Half), typeof(char), typeof(bool), typeof(nint), typeof(nuint), typeof(decimal), typeof(Int128), typeof(UInt128) };
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            string name = assembly.GetName().Name ?? "";
            if (!name.StartsWith("Adad", StringComparison.Ordinal) && !name.StartsWith("Qiyas", StringComparison.Ordinal)) continue;
            try { types.AddRange(assembly.GetTypes().Where(t => t.IsValueType && !t.ContainsGenericParameters && !t.IsByRefLike && !t.IsEnum)); }
            catch (ReflectionTypeLoadException) { }
        }

        var sizeOf = typeof(Unsafe).GetMethod(nameof(Unsafe.SizeOf))!;
        var refs = typeof(RuntimeHelpers).GetMethod(nameof(RuntimeHelpers.IsReferenceOrContainsReferences))!;
        foreach (var t in types)
        {
            try
            {
                if ((bool)refs.MakeGenericMethod(t).Invoke(null, null)!) continue;
                var pool = typeof(System.Buffers.ArrayPool<>).MakeGenericType(t).GetProperty("Shared")!.GetValue(null)!;
                _elementBytes[RuntimeHelpers.GetHashCode(pool)] = (int)sizeOf.MakeGenericMethod(t).Invoke(null, null)!;
            }
            catch (ArgumentException) { }
            catch (TargetInvocationException) { }
            catch (NotSupportedException) { }
        }
    }
}
