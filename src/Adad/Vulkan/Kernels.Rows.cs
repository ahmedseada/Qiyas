// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The row operations on Vulkan (Rows.cs; CUDA's Ptx.Rows.cs, whose operations EmitRowOp repeats instruction for
// instruction). An operation is written once and run as two kernels, CUDA's shape for a block format's rows:
//   RowStats   a workgroup per row: each of the operation's reductions over the row's values in workgroup strides,
//              joined by the tree in workgroup memory, its result kept as the row's statistic stats[row·4 + k];
//   RowReplay  a thread per stored output word: each of its values the operation replayed for that one value with its
//              row's statistics read back (the same instructions after the reductions, so the same bits as one kernel
//              holding the row), the word packed and stored once. A word's values may belong to two rows (rows need not
//              end on a word), so no two invocations store into one word.
// The parameters' backward (dgamma, dbeta) is CUDA's: the rows' statistics, then a thread per column over a split of
// the rows, the splits' partials joined by the column reduction.
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int RowThreads = 256;

    /// <summary>Softmax and rms_norm as row operations (rms_norm's eps is `a`): their arithmetic is CUDA's own kernels'.</summary>
    internal static readonly RowOp SoftmaxRow = new("softmax", false, false, false, false, false, false, 0, 0, static (_, _, _, _, _, _, _, _, _, _, _) => 0);
    internal static readonly RowOp RmsNormRow = new("rms_norm", false, false, false, false, false, false, 0, 0, static (_, _, _, _, _, _, _, _, _, _, _) => 0);

    /// <summary>Whether the operation reduces its rows (rope only rotates: no statistics kernel).</summary>
    public static bool RowReduces(RowOp op) => op.Name != "rope";

    // The row kernels' arguments: x, x2, g, bt, s, pos, y, y2, rows, cols, a, b, stats (null inputs are 0 addresses).
    private const int ArgX = 0, ArgX2 = 1, ArgG = 2, ArgBt = 3, ArgS = 4, ArgPos = 5, ArgY = 6, ArgY2 = 7, ArgRows = 8, ArgCols = 9, ArgA = 10, ArgB = 11, ArgStats = 12;

    /// <summary>The operation's reductions over rows of f (a workgroup a row) into stats (lane values, 4 a row).</summary>
    public static SpvKernel RowStats(Vulkan vk, Fmt f, Lane lane, RowOp op, bool second)
    {
        var b = new Spv($"rowstats_{op.Name}_{f.Name}_{lane.Name}{(second ? "" : "_nosecond")}", RowThreads, vk.Controls);
        var c = new RowCtx(b, f, lane, RowMode.Stats, op.Second && second);
        EmitRowOp(c, op);
        return b.Build();
    }

    /// <summary>The operation's outputs (y, and y2 when out2) a thread per stored word, each value replayed with its row's statistics.</summary>
    public static SpvKernel RowReplay(Vulkan vk, Fmt f, Lane lane, RowOp op, bool second, bool out2)
    {
        var b = new Spv($"rowreplay_{op.Name}_{f.Name}_{lane.Name}{(second ? "" : "_nosecond")}{(out2 ? "_y2" : "")}", Threads, vk.Controls);
        var c = new RowCtx(b, f, lane, RowMode.Replay, op.Second && second);
        var gid = b.GlobalX;
        var n = c.Rows * c.Cols;
        var wk = Codec.WordKind(f);
        var zero = b.C(c.K, 0);
        b.If(gid < WordCount(b, f, n), () =>
        {
            var word = b.CBits(wk, 0);
            var word2 = b.CBits(wk, 0);
            for (int i = 0; i < f.Per; i++)
            {
                // A value past the last (in the last word) replays the last value and stores zero.
                var idx = gid * (double)f.Per + i;
                c.BeginValue(b.Min(idx, n - 1));
                EmitRowOp(c, op);
                var inside = idx < n;
                word = Codec.Pack(b, f, b.Select(inside, c.Captured ?? throw new InvalidOperationException($"{op.Name}: no output"), zero), lane, i, word);
                if (out2) word2 = Codec.Pack(b, f, b.Select(inside, c.Captured2 ?? throw new InvalidOperationException($"{op.Name}: no second output"), zero), lane, i, word2);
            }

            b.Store(b.At(c.Dst, gid, f.WordBits / 8), word);
            if (out2) b.Store(b.At(c.Dst2, gid, f.WordBits / 8), word2);
        });
        return b.Build();
    }

    /// <summary>st[2·row] = mean (0 when not centered), st[2·row + 1] = 1/√(mean (x − mean)² + eps), lane values; st is
    /// argument y, eps argument a.</summary>
    public static SpvKernel NormStats(Vulkan vk, Fmt f, Lane lane, bool centered)
    {
        var b = new Spv($"norm_stats_{(centered ? "c" : "r")}_{f.Name}_{lane.Name}", RowThreads, vk.Controls);
        var c = new RowCtx(b, f, lane, RowMode.Stats, false) { Keep = false };
        var n = c.N;
        var mean = centered ? c.Reduce("add", c.X) / n : b.C(c.K, 0);
        var ss = c.Reduce("add", () => { var d = c.X() - mean; return d * d; });
        var rstd = b.C(c.K, 1) / b.Sqrt(ss / n + c.A);
        b.If(c.Tid == 0, () =>
        {
            b.Store(b.At(c.Dst, c.Row * 2, lane.Size), mean);
            b.Store(b.At(c.Dst, c.Row * 2 + 1, lane.Size), rstd);
        });
        return b.Build();
    }

    /// <summary>A thread per column over rows [split·per, …) (split the workgroup's y): partials_g[split][c] = Σ dy·(x − mean_r)·rstd_r,
    /// partials_b[split][c] = Σ dy (lane values). Arguments: x, dy, stats, partials_g, partials_b, rows, cols, per.</summary>
    public static SpvKernel NormParamsCols(Vulkan vk, Fmt f, Lane lane)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"norm_params_cols_{f.Name}_{lane.Name}", Threads, vk.Controls);
        var px = b.Arg(0);
        var pdy = b.Arg(1);
        var stats = b.Arg(2);
        var og = b.Arg(3);
        var ob = b.Arg(4);
        var rows = b.Arg(5, K.U32);
        var cols = b.Arg(6, K.U32);
        var per = b.Arg(7, K.U32);
        var col = b.GlobalX;
        var split = b.GroupY;
        b.If(col < cols, () =>
        {
            var accG = b.Local(b.C(k, 0));
            var accB = b.Local(b.C(k, 0));
            var r0 = split * per;
            b.For(r0, b.Min(r0 + per, rows), r =>
            {
                var idx = r * cols + col;
                var xv = Codec.UnpackAt(b, f, px, idx, lane);
                var dv = Codec.UnpackAt(b, f, pdy, idx, lane);
                var mean = b.Load(k, b.At(stats, r * 2, lane.Size));
                var rstd = b.Load(k, b.At(stats, r * 2 + 1, lane.Size));
                var h = (xv - mean) * rstd;
                accG.V = b.Fma(dv, h, accG.V);
                accB.V = accB.V + dv;
            });
            var at = split * cols + col;
            b.Store(b.At(og, at, lane.Size), accG.V);
            b.Store(b.At(ob, at, lane.Size), accB.V);
        });
        return b.Build();
    }

    // ---------------------------------------------------------------- the row context: values, statistics, outputs

    private enum RowMode { Stats, Replay }

    /// <summary>Stats: each reduction over the row (the workgroup's), its result kept (Keep) at stats[row·4 + k] by
    /// invocation 0 and returned to all; stores are not made. Replay: one value at a time (BeginValue), each reduction read
    /// back from stats, the stored value captured. Terms and outputs are of the current value (X, X2, G, Bt, Col).</summary>
    private sealed class RowCtx
    {
        private readonly Spv b;
        private readonly Fmt f;
        private readonly Lane lane;
        private readonly RowMode mode;
        private readonly bool second;
        private readonly V src, src2, pg, pb, ps, ppos, stats;
        private readonly Arr shared;                                         // Stats only
        private int k;

        public readonly K K;
        public readonly V Rows, Cols, ColsM1, A, Bs, N, Tid, Dst, Dst2;

        /// <summary>Stats: keep each reduction's result in stats (false for a kernel that stores its own statistics).</summary>
        public bool Keep { get; init; } = true;

        // The current value: its row, column, linear index and the inputs read for it (cleared with each value).
        public V Row { get; private set; }

        private V col, at;
        private V? x, x2, g, bt, s, posInt, pos;

        public V? Captured, Captured2;

        public RowCtx(Spv b, Fmt f, Lane lane, RowMode mode, bool second)
        {
            this.b = b;
            this.f = f;
            this.lane = lane;
            this.mode = mode;
            this.second = second;
            K = Kinds.Of(lane);
            src = b.Arg(ArgX);
            src2 = b.Arg(ArgX2);
            pg = b.Arg(ArgG);
            pb = b.Arg(ArgBt);
            ps = b.Arg(ArgS);
            ppos = b.Arg(ArgPos);
            Dst = b.Arg(ArgY);
            Dst2 = b.Arg(ArgY2);
            Rows = b.Arg(ArgRows, K.U32);
            Cols = b.Arg(ArgCols, K.U32);
            A = b.Arg(ArgA, K);
            Bs = b.Arg(ArgB, K);
            stats = b.Arg(ArgStats);
            ColsM1 = Cols - 1;
            N = b.Convert(Cols, K);
            Tid = b.LocalIdX;
            if (mode == RowMode.Stats)
            {
                // The workgroup's row; its scalar and position read once (they dominate every loop of the kernel).
                Row = b.GroupX;
                shared = b.Shared(K, RowThreads);
                s = LoadS();
                posInt = LoadPos();
                pos = b.Convert(posInt.Value, K);
            }
        }

        /// <summary>Replay: the value at linear index `index` (row index / cols): its inputs read as used.</summary>
        public void BeginValue(V index)
        {
            Row = index / Cols;
            at = index;
            col = index - Row * Cols;
            x = x2 = g = bt = s = posInt = pos = null;
            k = 0;
            Captured = Captured2 = null;
        }

        // Stats: value `c` of the workgroup's row.
        private void BeginColumn(V c)
        {
            col = c;
            at = Row * Cols + c;
            x = x2 = g = bt = null;
        }

        public V X() => x ??= Codec.UnpackAt(b, f, src, at, lane);

        /// <summary>The second row's value; zero when the operation's second input was not given (an absent mask).</summary>
        public V X2() => x2 ??= second ? Codec.UnpackAt(b, f, src2, at, lane) : b.C(K, 0);

        /// <summary>The per-column vectors' values at the current column: gamma, beta.</summary>
        public V G() => g ??= Codec.UnpackAt(b, f, pg, col, lane);

        public V Bt() => bt ??= Codec.UnpackAt(b, f, pb, col, lane);

        public V Col => col;

        /// <summary>The row's scalar (p_s; 1 when null).</summary>
        public V S() => s ??= LoadS();

        /// <summary>The row's position as s32 (p_pos; the row's index when null).</summary>
        public V PosInt() => posInt ??= LoadPos();

        /// <summary>The row's position as a lane value.</summary>
        public V Pos() => pos ??= b.Convert(PosInt(), K);

        private V LoadS()
        {
            var v = b.Local(b.C(K, 1));
            var row = Row;
            b.If(ps != 0UL, () => v.V = Codec.UnpackAt(b, f, ps, row, lane));
            return v.V;
        }

        private V LoadPos()
        {
            var v = b.Local(b.Bitcast(Row, K.S32));
            var row = Row;
            b.If(ppos != 0UL, () => v.V = b.Load(K.S32, b.At(ppos, row, 4)));
            return v.V;
        }

        /// <summary>The row's reduction ("add", "sumsq" or "max") of term over its values.</summary>
        public V Reduce(string kind, Func<V> term)
        {
            int slot = k++;
            if (mode == RowMode.Replay) return b.Load(K, b.At(stats, Row * 4 + slot, lane.Size));

            var acc = b.Local(b.C(K, kind == "max" ? double.NegativeInfinity : 0));
            b.For(Tid, Cols, b.U(RowThreads), c =>
            {
                BeginColumn(c);
                var t = term();
                acc.V = kind switch
                {
                    "max" => b.Max(acc.V, t),
                    "sumsq" => b.Fma(t, t, acc.V),                                   // Σ t² by fma
                    _ => acc.V + t,
                };
            });
            x = x2 = g = bt = null;
            Func<V, V, V> join = kind == "max" ? (p, q) => b.Max(p, q) : static (p, q) => p + q;
            var result = GroupReduce(b, shared, Tid, acc.V, join, RowThreads);
            if (Keep)
            {
                var row = Row;
                b.If(Tid == 0, () => b.Store(b.At(stats, row * 4 + slot, lane.Size), result));
            }

            return result;
        }

        /// <summary>The row's one value (RowOut: a lane value at y2[row], when y2 is given) by invocation 0 of the statistics pass;
        /// a replay makes none.</summary>
        public void StoreRow(V v)
        {
            if (mode == RowMode.Replay) return;
            var row = Row;
            b.If((Tid == 0) & (Dst2 != 0UL), () => b.Store(b.At(Dst2, row, lane.Size), v));
        }

        /// <summary>The output (or the second output) from value(): captured in a replay, not made in the statistics pass.</summary>
        public void Store(Func<V> value, bool second = false)
        {
            if (mode == RowMode.Stats) return;
            if (second) Captured2 = value();
            else Captured = value();
        }

        public V Lit(double v) => b.C(K, v);

        public V Exp(V v) => Fn.Exp(b, v);

        public V Log(V v) => Fn.Log(b, v);

        public V Sqrt(V v) => b.Sqrt(v);

        public V Sin(V v) => Fn.Sin(b, v);

        public V Cos(V v) => Fn.Cos(b, v);

        public V Fma(V p, V q, V r) => b.Fma(p, q, r);

        public V Max(V p, V q) => b.Max(p, q);

        public V Sel(V condition, V p, V q) => b.Select(condition, p, q);

        public V Rcp(V v) => Lit(1) / v;

        /// <summary>x / y from y's rounded reciprocal r: q = x·r, then q + (x − q·y)·r (the residual exact by fma).</summary>
        public V Quotient(V p, V q, V r)
        {
            var t = p * r;
            return Fma(Fma(-t, q, p), r, t);
        }

        /// <summary>The partner value of a rotation: x at (row, column) of the current row.</summary>
        public V XAt(V column) => Codec.UnpackAt(b, f, src, Row * Cols + column, lane);

        public Spv B => b;
    }

    // ---------------------------------------------------------------- the operations (CUDA's EmitRowOp, instruction for instruction)

    private static V Rstd(RowCtx c, V sumSq, V n, V eps) => c.Rcp(c.Sqrt(sumSq / n + eps));

    // The gain of an operation that scales by g + b (RowOp.Offset).
    private static V Gain(RowCtx c) => c.G() + c.Bs;

    private static void EmitRowOp(RowCtx c, RowOp op)
    {
        var n = c.N;
        switch (op.Name)
        {
            case "softmax":
            {
                var m = c.Reduce("max", c.X);
                var sum = c.Reduce("add", () => c.Exp(c.X() - m));
                var scale = c.Rcp(sum);
                c.Store(() => c.Quotient(c.Exp(c.X() - m), sum, scale));
                return;
            }

            case "rms_norm":
            {
                var scale = c.Rcp(c.Sqrt(c.Reduce("sumsq", c.X) / n + c.A));
                c.Store(() => c.X() * scale);
                return;
            }

            case "log_softmax":
            {
                var m = c.Reduce("max", c.X);
                var sum = c.Reduce("add", () => c.Exp(c.X() - m));
                var ls = m + c.Log(sum);
                c.Store(() => c.X() - ls);
                return;
            }

            case "softmax_bwd":
            {
                var d = c.Reduce("add", () => c.X() * c.X2());
                c.Store(() => c.X() * (c.X2() - d));
                return;
            }

            case "log_softmax_bwd":
            {
                var sd = c.Reduce("add", c.X2);
                c.Store(() => c.X2() - c.Exp(c.X()) * sd);
                return;
            }

            case "rms_norm_affine":
            {
                var rstd = Rstd(c, c.Reduce("add", () => c.X() * c.X()), n, c.A);
                c.Store(() => c.X() * rstd * Gain(c));
                return;
            }

            case "rms_norm_bwd":
            {
                var ss = c.Reduce("add", () => c.X() * c.X());
                var t = c.Reduce("add", () => Gain(c) * c.X2() * c.X());
                var rstd = Rstd(c, ss, n, c.A);
                var k2 = rstd * (rstd * t / n);                                          // rstd · mean(γ·dy·x̂)
                c.Store(() => (Gain(c) * c.X2() - c.X() * k2) * rstd);
                return;
            }

            case "layer_norm":
            {
                var mu = c.Reduce("add", c.X) / n;
                var v = c.Reduce("add", () => { var d = c.X() - mu; return d * d; });
                var rstd = Rstd(c, v, n, c.A);
                c.Store(() => c.Fma((c.X() - mu) * rstd, c.G(), c.Bt()));
                return;
            }

            case "layer_norm_bwd":
            {
                var mu = c.Reduce("add", c.X) / n;
                var v = c.Reduce("add", () => { var d = c.X() - mu; return d * d; });
                var sgd = c.Reduce("add", () => c.G() * c.X2());
                var sgdx = c.Reduce("add", () => c.G() * c.X2() * (c.X() - mu));
                var rstd = Rstd(c, v, n, c.A);
                var m1 = sgd / n;
                var k2 = rstd * (rstd * sgdx / n);
                c.Store(() => (c.G() * c.X2() - m1 - (c.X() - mu) * k2) * rstd);
                return;
            }

            case "weight_norm_bwd":
            {
                // dx = (s/n)·(dy − x·k), k = Σ dy·x / (Σ x² + a), n = √(Σ x² + a); the row's value (Σ dy·x)/n.
                var ss = c.Reduce("add", () => c.X() * c.X());
                var dot = c.Reduce("add", () => c.X2() * c.X());
                var nn = ss + c.A;
                var nrm = c.Sqrt(nn);
                var k = dot / nn;
                var fct = c.S() / nrm;
                c.Store(() => (c.X2() - c.X() * k) * fct);
                c.StoreRow(dot / nrm);
                return;
            }

            case "weight_norm":
            {
                var ss = c.Reduce("add", () => c.X() * c.X());
                var k = c.S() / c.Sqrt(ss + c.A);
                c.Store(() => c.X() * k);
                return;
            }

            case "rope":
            {
                // θ = pos · exp(i · k), k = −2 ln a / n; the partner value (i ± half) read from the row.
                var b = c.B;
                var half = c.Cols >> 1;
                var twice = half << 1;
                var k = c.Lit(-2) / n * c.Log(c.A);
                c.Store(() =>
                {
                    var pos = c.Pos();
                    var halfF = b.Convert(half, c.K);
                    var col = c.Col;
                    var low = col < half;
                    var pair = col < twice;
                    var colF = b.Convert(col, c.K);
                    var iF = c.Sel(low, colF, colF - halfF);
                    var theta = pos * c.Exp(iF * k);
                    var cs = c.Cos(theta);
                    var sn = c.Sin(theta);
                    var partnerCol = b.Min(c.Sel(low, col + half, col - half), c.ColsM1);
                    var partner = c.XAt(partnerCol);
                    var x = c.X();
                    var lowValue = x * cs - partner * sn;                               // x[i]·cos − x[i+half]·sin
                    var highValue = c.Fma(partner, sn, x * cs);                         // x[i]·sin + x[i+half]·cos (partner = x[i])
                    return c.Sel(pair, c.Sel(low, lowValue, highValue), x);
                });
                return;
            }

            case "scale_mask_softmax":
            {
                // z = a·x + mask; causal (b ≠ 0): columns past the row's position excluded; excluded values give 0 (a fully excluded row too).
                var b = c.B;
                var causal = c.Bs > 0;
                var negInf = c.Lit(double.NegativeInfinity);
                V Keep() => !(causal & (b.Bitcast(c.Col, K.S32) > c.PosInt()));
                V Z() => c.Sel(Keep(), c.Fma(c.X(), c.A, c.X2()), negInf);
                var m = c.Reduce("max", Z);
                var any = m > negInf;
                var sum = c.Reduce("add", () => c.Sel(Keep(), c.Exp(Z() - m), c.Lit(0)));
                var scale = c.Sel(any, c.Rcp(sum), c.Lit(0));
                c.Store(() => c.Sel(Keep(), c.Exp(Z() - m) * scale, c.Lit(0)));
                return;
            }

            case "cross_entropy_bwd":
            case "weighted_cross_entropy_bwd":
            {
                // Weighted: (softmax − one_hot)·(a·s), the target's p − 1 before the scaling.
                var b = c.B;
                var m = c.Reduce("max", c.X);
                var sum = c.Reduce("add", () => c.Exp(c.X() - m));
                var inv = c.Rcp(sum);
                V? factor = op.Name == "weighted_cross_entropy_bwd" ? c.A * c.S() : null;
                c.Store(() =>
                {
                    var p = c.Exp(c.X() - m) * inv;
                    var d = c.Sel(b.Bitcast(c.Col, K.S32) == c.PosInt(), p - c.Lit(1), p);
                    return factor is null ? d : d * factor.Value;
                });
                return;
            }

            case "add_rms_norm":
            {
                V H() => c.X() + c.X2();
                c.Store(H, second: true);
                // y = h·g / s, s = √(mean h² + eps): the quotient by its reciprocal and one residual step (h·g − y·s, exact
                // by fma) — near-correctly rounded, two roundings fewer than h · (1/s) · g.
                var s = c.Sqrt(c.Reduce("add", () => { var h = H(); return h * h; }) / n + c.A);
                var rstd = c.Rcp(s);
                c.Store(() => c.Quotient(H() * Gain(c), s, rstd));
                return;
            }

            case "add_layer_norm":
            {
                V H() => c.X() + c.X2();
                c.Store(H, second: true);
                var mu = c.Reduce("add", H) / n;
                var v = c.Reduce("add", () => { var d = H() - mu; return d * d; });
                var rstd = Rstd(c, v, n, c.A);
                c.Store(() => c.Fma((H() - mu) * rstd, c.G(), c.Bt()));
                return;
            }

            default: throw new NotSupportedException($"vulkan: row operation {op.Name} has no Vulkan kernels yet");
        }
    }
}
