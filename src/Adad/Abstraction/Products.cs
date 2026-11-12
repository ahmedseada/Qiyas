// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Products: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Products.cs; Cuda/GpuOps.Products.cs with its kernels in Cuda/Ptx.cs.

using System.Buffers;

namespace Adad.Abstraction;

/// <summary>A product on views: A stored m × k (k × m when TransA), B stored k × n (n × k when TransB), C m × n;
/// `Batch` products at strides StrideA, StrideB, StrideC (in values; 0 broadcasts the operand across the batch).</summary>
public sealed record ProductView(int M, int N, int K, bool TransA, bool TransB, int Batch = 1, long StrideA = 0, long StrideB = 0, long StrideC = 0)
{
    public long StrideAOrDefault => Batch > 1 ? StrideA : 0;
    public long StrideBOrDefault => Batch > 1 ? StrideB : 0;
    public long StrideCOrDefault => Batch > 1 ? StrideC : (long)M * N;

    /// <summary>The strided view that makes operand A of batch i contiguous (m × k).</summary>
    public int[] ViewA(int i) => TransA ? [1, 1, M, K, 0, 0, 1, M, (int)(i * StrideAOrDefault)] : [1, 1, M, K, 0, 0, K, 1, (int)(i * StrideAOrDefault)];

    /// <summary>The strided view that makes operand B of batch i contiguous (k × n).</summary>
    public int[] ViewB(int i) => TransB ? [1, 1, K, N, 0, 0, 1, K, (int)(i * StrideBOrDefault)] : [1, 1, K, N, 0, 0, N, 1, (int)(i * StrideBOrDefault)];

    public long ValuesA => (TransA ? (long)K * M : (long)M * K) + (Batch - 1) * StrideAOrDefault;
    public long ValuesB => (TransB ? (long)N * K : (long)K * N) + (Batch - 1) * StrideBOrDefault;
    public long ValuesC => (long)M * N + (Batch - 1) * StrideCOrDefault;

    /// <summary>The float64 reference: every batch's product of the operands as viewed.</summary>
    public double[] Reference(double[] a, double[] b)
    {
        var c = new double[ValuesC];
        for (int i = 0; i < Batch; i++)
        {
            var ai = new double[M * K];
            var bi = new double[K * N];
            long baseA = i * StrideAOrDefault, baseB = i * StrideBOrDefault;
            for (int r = 0; r < M; r++) for (int q = 0; q < K; q++) ai[r * K + q] = TransA ? a[baseA + q * M + r] : a[baseA + r * K + q];
            for (int q = 0; q < K; q++) for (int s = 0; s < N; s++) bi[q * N + s] = TransB ? b[baseB + s * K + q] : b[baseB + q * N + s];
            var ci = Reference(ai, bi, M, N, K);
            Array.Copy(ci, 0, c, i * StrideCOrDefault, M * N);
        }

        return c;
    }

    /// <summary>The float64 reference product (A m × k, B k × n), accumulated in double-double (each product exact by a fused
    /// multiply-add, each sum with its error term), so a lane that is exact reads 0 and a float64 lane shows its own rounding.</summary>
    public static double[] Reference(double[] a, double[] b, int m, int n, int k)
    {
        var c = new double[m * n];
        Parallel.For(0, m, i =>
        {
            var hi = new double[n];
            var lo = new double[n];
            for (int p = 0; p < k; p++)
            {
                double x = a[i * k + p];
                var bp = b.AsSpan(p * n, n);
                for (int j = 0; j < n; j++)
                {
                    double y = bp[j], ph = x * y, pl = Math.FusedMultiplyAdd(x, y, -ph);
                    double h = hi[j], sum = h + ph, bv = sum - h, err = (h - (sum - bv)) + (ph - bv);
                    hi[j] = sum;
                    lo[j] += err + pl;
                }
            }

            var row = c.AsSpan(i * n, n);
            for (int j = 0; j < n; j++) row[j] = hi[j] + lo[j];
        });
        return c;
    }
}

/// <summary>What a product's store does with each of its sums s = op(A)·op(B) (the float lane's, every term below added in
/// it) before y is rounded to the type once: s + op(A2)·op(B2) (a second product over the view Second: the product's M, N
/// and batches, its own K, transposes and strides; at a small K, a low-rank term), + bias[col] (n values), + β·y (y's
/// value before, read where it is stored; β 0: y is not read); s stored as the type into Sums where it is given; then
/// y = act(s), or with Grad, y = grad(ActInputs, s) (the gradient through the activation whose inputs ActInputs holds, s
/// the incoming gradient: gelu_bwd: a gelu's backward through the product before it); then y + residual (y's layout).
/// The batches' C contiguous (StrideC m·n), the second product's too.</summary>
public sealed record ProductEpilogue
{
    /// <summary>y's value before, times β, added to the sums (0: y not read).</summary>
    public double Beta { get; init; }

    /// <summary>A one-input formula (Map.cs) applied to the sums, with ActA and ActB its parameters (a and b: affine's
    /// a·x + b, clamp's bounds); null none.</summary>
    public string? Act { get; init; }

    public double ActA { get; init; }

    public double ActB { get; init; }

    /// <summary>A two-input backward formula grad(x, dy) (Map.cs: gelu_bwd, silu_bwd, …) taken as grad(aux, s); null none.</summary>
    public string? Grad { get; init; }

    /// <summary>A second product added to the sums (null: none): its M, N and batches the product's, its C contiguous.</summary>
    public ProductView? Second { get; init; }

    /// <summary>The terms the store takes for the buffers given (sums: s kept, stored into ProductTerms.Sums).</summary>
    public StoreTerms Terms(bool bias, bool residual, bool sums) => new(Second is not null, bias, Beta != 0, Act, Grad, residual) { Keep = sums };

    public void Validate(ProductView v, bool bias, bool residual, bool sums, bool actInputs, bool secondA, bool secondB)
    {
        if (v.Batch > 1 && v.StrideCOrDefault != (long)v.M * v.N) throw new ArgumentException("matmul epilogue: the batches' C must be contiguous (StrideC m·n)");
        if (v.ValuesC > int.MaxValue) throw new ArgumentException("matmul epilogue: C past 2^31 values");
        if (Act is not null && Grad is not null) throw new ArgumentException("matmul epilogue: an activation or a gradient, not both");
        if (Act is not null && Ops.Find(Act).Inputs != 1) throw new ArgumentException($"matmul epilogue: the activation {Act} is not a one-input formula");
        if (Grad is not null && Ops.Find(Grad).Inputs != 2) throw new ArgumentException($"matmul epilogue: the gradient {Grad} is not a two-input formula");
        if (Grad is not null && !actInputs) throw new ArgumentException("matmul epilogue: a gradient reads the activation's inputs (ActInputs)");
        if (Grad is null && actInputs) throw new ArgumentException("matmul epilogue: the activation's inputs given without a gradient to read them");
        if (Second is null)
        {
            if (secondA || secondB) throw new ArgumentException("matmul epilogue: a second product's operands given without its view");
            return;
        }

        var w = Second;
        if (!secondA || !secondB) throw new ArgumentException("matmul epilogue: the second product takes its A and B");
        if (w.M != v.M || w.N != v.N || w.Batch != v.Batch) throw new ArgumentException($"matmul epilogue: a second product of {w.Batch} × {w.M} × {w.N} for a product of {v.Batch} × {v.M} × {v.N}");
        if (w.K < 0) throw new ArgumentException("matmul epilogue: a second product of negative depth");
        if (w.Batch > 1 && w.StrideCOrDefault != (long)w.M * w.N) throw new ArgumentException("matmul epilogue: the second product's C must be contiguous (StrideC m·n)");
        if (w.ValuesA > int.MaxValue || w.ValuesB > int.MaxValue) throw new ArgumentException("matmul epilogue: a second product's operand past 2^31 values");
    }

    /// <summary>Refuses what a type that is not float does not take: a formula only the float types run (MapOp.FloatOnly,
    /// as Map refuses it), a β or a formula's parameter that is not an integer (an integer type's terms are its integers),
    /// and a formula whose value at integers is not an integer (Integral: div).</summary>
    public void ValidateType(string type, bool isFloat)
    {
        if (isFloat) return;
        foreach (var name in new[] { Act, Grad })
            if (name is not null && Ops.Find(name).FloatOnly) throw new NotSupportedException($"matmul epilogue: {name} is a float formula, and {type} is not a float type");
        if (Beta != Math.Floor(Beta) || ActA != Math.Floor(ActA) || ActB != Math.Floor(ActB))
            throw new ArgumentException($"matmul epilogue: {type} takes an integer β and parameters, not {Beta}, {ActA} and {ActB}");
        foreach (var name in new[] { Act, Grad })
            if (name is not null && !Integral(Ops.Find(name), ActA, ActB))
                throw new NotSupportedException($"matmul epilogue: {name}'s value at integers is not always an integer, and {type} is not a float type");
    }

    /// <summary>Whether a formula's value at integers is a finite integer at every point of a grid (each input -3 to 3; a
    /// and b its parameters): what a type that is not float takes where the lane is chosen by bounds, a float lane or an
    /// integer one, either holding such a value exactly. div is not (1 / 2, 1 / 0): its integer form truncates, which a
    /// float lane would not, and its quotient over a range crossing 0 has no bound.</summary>
    public static bool Integral(MapOp op, double a, double b)
    {
        int ny = op.Inputs > 1 ? 3 : 0, nz = op.Inputs > 2 ? 3 : 0;
        for (int x = -3; x <= 3; x++)
            for (int y = -ny; y <= ny; y++)
                for (int z = -nz; z <= nz; z++)
                {
                    double r = op.Reference(x, y, z, a, b);
                    if (!double.IsFinite(r) || r != Math.Floor(r)) return false;
                }

        return true;
    }

    /// <summary>The lane an integer type's epilogue sums in, from the ranges of what it reads ((low, high) each; null where
    /// its term is not taken): every value the store makes, from the products' sums to y before its one rounding, held
    /// exactly (the type's store then saturates once); ExactSums by those values and the operands' ranges (the second
    /// product's with the product's: a dot lane reads both).</summary>
    public (Bounds Bounds, bool Negative) IntegerSums(string type, bool signed, int k, (double Lo, double Hi) a, (double Lo, double Hi) b,
        (double Lo, double Hi)? a2, (double Lo, double Hi)? b2, (double Lo, double Hi)? bias, (double Lo, double Hi)? y0,
        (double Lo, double Hi)? residual, (double Lo, double Hi)? actInputs)
    {
        static (double Lo, double Hi) Products(long k, (double Lo, double Hi) x, (double Lo, double Hi) y)
        {
            if (k == 0) return (0, 0);
            double p = x.Lo * y.Lo, q = x.Lo * y.Hi, r = x.Hi * y.Lo, s = x.Hi * y.Hi;
            return (k * Math.Min(Math.Min(p, q), Math.Min(r, s)), k * Math.Max(Math.Max(p, q), Math.Max(r, s)));
        }

        // The terms of s; each partial sum of them lies within [Σ min(low, 0), Σ max(high, 0)].
        var terms = new List<(double Lo, double Hi)> { Products(k, a, b) };
        if (Second is not null) terms.Add(Products(Second.K, a2!.Value, b2!.Value));
        if (bias is { } bv) terms.Add(bv);
        if (Beta != 0) terms.Add((Math.Min(Beta * y0!.Value.Lo, Beta * y0.Value.Hi), Math.Max(Beta * y0.Value.Lo, Beta * y0.Value.Hi)));
        double sLo = 0, sHi = 0, lo = 0, hi = 0;
        foreach (var (tl, th) in terms) { sLo += tl; sHi += th; lo += Math.Min(tl, 0); hi += Math.Max(th, 0); }

        // The formula over s's range (its ends and 0; a gradient's with its inputs' ends and 0), then the residual.
        var o = (Lo: sLo, Hi: sHi);
        if ((Act ?? Grad) is { } name)
        {
            var f = Ops.Find(name);
            double oLo = double.PositiveInfinity, oHi = double.NegativeInfinity;
            foreach (double x in Grad is null ? [0.0] : new[] { actInputs!.Value.Lo, 0, actInputs.Value.Hi })
                foreach (double s in new[] { sLo, 0, sHi })
                {
                    double r = Grad is null ? f.Reference(s, 0, 0, ActA, ActB) : f.Reference(x, s, 0, ActA, ActB);
                    oLo = Math.Min(oLo, r);
                    oHi = Math.Max(oHi, r);
                }

            o = (oLo, oHi);
        }

        var y = residual is { } rv ? (Lo: o.Lo + rv.Lo, Hi: o.Hi + rv.Hi) : o;
        lo = Math.Min(lo, Math.Min(o.Lo, y.Lo));
        hi = Math.Max(hi, Math.Max(o.Hi, y.Hi));
        // An unsigned lane holds a negative β or parameter only by wrapping: a float lane takes them.
        if (!signed && (Beta < 0 || ActA < 0 || ActB < 0)) lo = Math.Min(lo, -1);
        return ExactSums("matmul epilogue", type, signed, lo, hi, (Math.Min(a.Lo, a2?.Lo ?? a.Lo), Math.Max(a.Hi, a2?.Hi ?? a.Hi)),
            (Math.Min(b.Lo, b2?.Lo ?? b.Lo), Math.Max(b.Hi, b2?.Hi ?? b.Hi)));
    }

    /// <summary>A product's lane for a type that is not float, every value its store makes within [lo, hi] and its operands
    /// within a and b (both products' together): Bounds with those ranges (at least 0 to 1, so a zero operand still
    /// bounds the terms) and a count of terms whose Bound passes |lo| and |hi|; Negative: lo below 0, which an unsigned
    /// type's own lanes do not hold: its Bounds then pass 2^32, past every unsigned dot and 32-bit lane, which leaves the
    /// float64 lane (every device takes it), refused past 2^53, which it does not hold exactly.</summary>
    public static (Bounds Bounds, bool Negative) ExactSums(string what, string type, bool signed, double lo, double hi, (double Lo, double Hi) a, (double Lo, double Hi) b)
    {
        double bound = Math.Max(-lo, hi);
        if (!signed && lo < 0)
        {
            if (bound > Exact.Float64) throw new NotSupportedException($"{what}: {type}'s values reach {lo} and {hi}: below 0 and past 2^53, no lane holds them exactly");
            return (new Bounds(-1, 1, -1, 1, (long)Math.Max(Math.Ceiling(bound), 4294967297.0)), true);
        }

        double aLo = a.Lo, aHi = Math.Max(a.Hi, 1), bLo = b.Lo, bHi = Math.Max(b.Hi, 1);
        double per = Math.Max(-aLo, aHi) * Math.Max(-bLo, bHi);
        return (new Bounds(aLo, aHi, bLo, bHi, (long)Math.Clamp(Math.Ceiling(bound / per), 1, long.MaxValue)), lo < 0);
    }

    /// <summary>The float64 reference: y and the sums s it was made from (y0 y's values before; bias n values; residual and
    /// actInputs y's layout; a2 and b2 the second product's operands as its view stores them).</summary>
    public (double[] Y, double[] S) Reference(ProductView v, double[] a, double[] b, double[] y0, double[]? bias, double[]? residual, double[]? actInputs, double[]? a2, double[]? b2) =>
        Apply(v.Reference(a, b), 1, v.N, y0, bias, residual, actInputs, Second?.Reference(a2!, b2!));

    /// <summary>The store's float64 reference over a product's sums p (y's layout, every batch): y and the sums s, value i's
    /// bias the channel (i / cols) mod oc's (a view's product: cols 1, oc its n columns; a product whose bias is a value a
    /// row of n columns: cols n, oc the rows), second the second product's sums (or none).</summary>
    public (double[] Y, double[] S) Apply(double[] p, int cols, int oc, double[] y0, double[]? bias, double[]? residual, double[]? actInputs, double[]? second)
    {
        var act = Act is null ? null : Ops.Find(Act);
        var grad = Grad is null ? null : Ops.Find(Grad);
        var s = new double[p.Length];
        var y = new double[p.Length];
        for (int i = 0; i < p.Length; i++)
        {
            double t = p[i] + (second is null ? 0 : second[i]) + (bias is null ? 0 : bias[i / cols % oc]) + (Beta == 0 ? 0 : Beta * y0[i]);
            s[i] = t;
            double o = act is not null ? act.Reference(t, 0, 0, ActA, ActB) : grad is not null ? grad.Reference(actInputs![i], t, 0, ActA, ActB) : t;
            y[i] = o + (residual is null ? 0 : residual[i]);
        }

        return (y, s);
    }
}

/// <summary>The terms a product's store adds to its lane sums and what it applies (ProductEpilogue.Terms; the convolution's
/// store takes the bias and the activation): a store kernel's variant. Second: a second product's lane sums added; Keep: a
/// pass stores s before y (KeepPass).</summary>
public readonly record struct StoreTerms(bool Second, bool Bias, bool Beta, string? Act, string? Grad, bool Residual)
{
    public bool Keep { get; init; }

    /// <summary>The pass that stores s (every term before the activation or the gradient, no residual) into the sums kept.</summary>
    public StoreTerms KeepPass => this with { Act = null, Grad = null, Residual = false, Keep = false };

    public string Key => $"{(Second ? "s" : "")}{(Bias ? "b" : "")}{(Beta ? "c" : "")}_{(Grad is null ? Act ?? "none" : "d" + Grad)}{(Residual ? "_r" : "")}";
}

/// <summary>A product epilogue's operands (ProductEpilogue), each absent (null; 0 on a device) where its term is not taken:
/// Bias (n values), Residual (y's layout), Sums (an output, y's layout: s stored as the type), ActInputs (y's layout: the
/// activation's inputs a gradient reads), SecondA and SecondB (the second product's operands, as its view stores them).</summary>
public readonly record struct ProductTerms<TBuffer>(TBuffer? Bias = default, TBuffer? Residual = default, TBuffer? Sums = default, TBuffer? ActInputs = default,
    TBuffer? SecondA = default, TBuffer? SecondB = default)
{
    public static bool Given(TBuffer? x) => !EqualityComparer<TBuffer?>.Default.Equals(x, default);

    public StoreTerms Validate(ProductEpilogue e, ProductView v)
    {
        e.Validate(v, Given(Bias), Given(Residual), Given(Sums), Given(ActInputs), Given(SecondA), Given(SecondB));
        return e.Terms(Given(Bias), Given(Residual), Given(Sums));
    }
}

/// <summary>A weight stored as codes with scales: N rows of K codes (a row an output), value (r, c) = (code − zero)·scale,
/// one scale and zero point (zeros absent: 0) a block of BlockRows rows by BlockColumns columns, at (r / BlockRows)·
/// ScaleColumns + c / BlockColumns: one block for the tensor, one a row, a group of a row's columns, or blocks of rows and
/// columns. The value is made in the float lane as a product reads the code: the code taken into the lane, the zero
/// subtracted, the scale multiplied.</summary>
public sealed record ScaledWeight(int N, int K, int BlockRows, int BlockColumns)
{
    public static ScaledWeight PerTensor(int n, int k) => new(n, k, Math.Max(1, n), Math.Max(1, k));

    public static ScaledWeight PerRow(int n, int k) => new(n, k, 1, Math.Max(1, k));

    public static ScaledWeight Grouped(int n, int k, int group) => new(n, k, 1, group);

    /// <summary>The blocks across a row: ⌈K / BlockColumns⌉ (at least one).</summary>
    public int ScaleColumns => BlockColumns < 1 ? 1 : Math.Max(1, (int)(((long)K + BlockColumns - 1) / BlockColumns));

    public long Values => (long)N * K;

    /// <summary>The scales (and zero points) the layout takes.</summary>
    public long Scales => (long)((N + BlockRows - 1) / BlockRows) * ScaleColumns;

    public int ScaleAt(int r, int c) => r / BlockRows * ScaleColumns + c / BlockColumns;

    /// <summary>The product view the weight takes in y = x · Wᵀ (x m × K).</summary>
    public ProductView View(int m) => new(m, N, K, false, true);

    public void Validate()
    {
        if (N < 0 || K < 0) throw new ArgumentException("scaled weight: negative extents");
        if (BlockRows < 1 || BlockColumns < 1) throw new ArgumentException("scaled weight: a block of no rows or columns");
        if (Values > int.MaxValue || Scales > int.MaxValue) throw new ArgumentException("scaled weight: past 2^31 values");
    }

    /// <summary>The block (rows [R0, R1), columns [C0, C1)) scale si covers.</summary>
    public (int R0, int R1, int C0, int C1) Block(int si)
    {
        int r0 = si / ScaleColumns * BlockRows, c0 = si % ScaleColumns * BlockColumns;
        return (r0, Math.Min(N, r0 + BlockRows), c0, Math.Max(c0, Math.Min(K, c0 + BlockColumns)));
    }

    /// <summary>Whether codes of fw (zero points or not) can be made by Quantize: not a block format; zero points for an integer type only.</summary>
    public static void ValidateCodes(Fmt fw, bool zeros)
    {
        if (fw.IsBlock) throw new ArgumentException($"quantize: codes of the block format {fw.Name}");
        if (zeros && fw.IsFloat) throw new ArgumentException($"quantize: zero points for the float codes of {fw.Name}");
    }

    /// <summary>The float64 reference of each block's range (IProducts.Quantize's rule): lo = min(x, 0) and hi = max(x, 0)
    /// over the block's values, and its scale from them, max(hi, −lo) / max, or with zero points (hi − lo) / (max − min).</summary>
    public (double[] Scales, double[] Lows) QuantizeReference(double[] x, double min, double max, bool zeros)
    {
        var scales = new double[Scales];
        var lows = new double[Scales];
        for (int si = 0; si < Scales; si++)
        {
            var (r0, r1, c0, c1) = Block(si);
            double lo = 0, hi = 0;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++) { double v = x[(long)r * K + c]; lo = Math.Min(lo, v); hi = Math.Max(hi, v); }
            lows[si] = lo;
            scales[si] = zeros ? (hi - lo) / (max - min) : Math.Max(hi, -lo) / max;
        }

        return (scales, lows);
    }

    /// <summary>The float64 reference of the weight's values (N × K).</summary>
    public double[] Reference(double[] codes, double[] scales, double[]? zeros)
    {
        var w = new double[Values];
        for (int r = 0; r < N; r++)
            for (int c = 0; c < K; c++)
            {
                int at = ScaleAt(r, c);
                w[(long)r * K + c] = (codes[(long)r * K + c] - (zeros is null ? 0 : zeros[at])) * scales[at];
            }

        return w;
    }
}
