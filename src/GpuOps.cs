// The operations on the device over the contract: the same lane rules as the CPU (Types.cs), with the device's lanes
// and the device's measured facts (whether its JIT takes dp4a; which product tile is faster for a shape).
using System.Diagnostics;

namespace Qiyas;

public sealed class GpuOps(Gpu gpu)
{
    private readonly Dictionary<string, int> _tiles = [];

    public Gpu Gpu => gpu;

    private static uint Grid(long items, int block) => (uint)((items + block - 1) / block);

    /// <summary>When set (profile runs only), MatMul adds each phase's milliseconds after a synchronization: pack A, pack B, product, back.</summary>
    public static double[]? Phases;

    private static long Mark(long since, int phase)
    {
        if (Phases is null) return since;
        Gpu.Synchronize();
        long now = Stopwatch.GetTimestamp();
        Phases[phase] += Stopwatch.GetElapsedTime(since, now).TotalMilliseconds;
        return now;
    }

    // ---------------------------------------------------------------- add

    public void Add<TN>(ulong a, ulong b, ulong c, int n) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var fn = gpu.Cached($"add:{f.Name}", (out string e) => Ptx.Add(gpu, f, out e));
        long words = f.Words(n);
        int perThread = f.WordBits == 64 ? 2 : 4;
        Gpu.Launch(fn, Grid((words + perThread - 1) / perThread, 256), 1, 256, 1, 0, a, b, c, Gpu.U(n));
    }

    // ---------------------------------------------------------------- the range of stored values

    public (double Min, double Max) Range<TN>(ulong x, int n) where TN : struct, IQiyasNumber<TN> => RangePair<TN>(x, n, 0, 0).A;

    /// <summary>The ranges of two stored arrays (the operands of a product) with one download.</summary>
    public ((double Min, double Max) A, (double Min, double Max) B) RangePair<TN>(ulong x, int nx, ulong y, int ny) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsFloat || f.Bits <= 8) return ((TN.Min, TN.Max), (TN.Min, TN.Max));   // the type's range decides the lane already
        var fn = gpu.Cached($"range:{f.Name}", (out string e) => Ptx.Range(gpu, f, out e));
        int Blocks(int n) => (int)Math.Min(512, Math.Max(1, (f.Words(n) + Ptx.RangeThreads - 1) / Ptx.RangeThreads));
        int bx = Blocks(nx), by = ny > 0 ? Blocks(ny) : 0;
        ulong outBuffer = gpu.Rent((bx + by) * 16L);
        Gpu.Launch(fn, (uint)bx, 1, Ptx.RangeThreads, 1, 0, x, Gpu.U(nx), outBuffer);
        if (by > 0) Gpu.Launch(fn, (uint)by, 1, Ptx.RangeThreads, 1, 0, y, Gpu.U(ny), outBuffer + (ulong)(bx * 16));
        var pairs = new long[(bx + by) * 2];
        Gpu.Download<long>(outBuffer, pairs);
        gpu.Return(outBuffer);
        (long, long) Join(int from, int count)
        {
            long lo = long.MaxValue, hi = long.MinValue;
            for (int i = from; i < from + count; i++) { lo = Math.Min(lo, pairs[2 * i]); hi = Math.Max(hi, pairs[2 * i + 1]); }
            return (lo, hi);
        }

        var ra = Join(0, bx);
        var rb = by > 0 ? Join(bx, by) : (0L, 0L);
        return ((ra.Item1, ra.Item2), (rb.Item1, rb.Item2));
    }

    // ---------------------------------------------------------------- C = A·B

    /// <summary>The lane the product takes for these bounds, by the same rule as the CPU, on this device's lanes: the tensor
    /// instructions first (exact for every input they are chosen for), then the dot instruction, then the plain lanes.</summary>
    public string ProductLane<TN>(in Bounds bounds) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsFloat)
        {
            if (f.Bits == 64) return "f64";
            if (f.Bits == 32) return "f32";
            var kind = f.Is("bfloat16") ? Ptx.Kind.BF16 : Ptx.Kind.F16;
            return MmaSupported(kind) ? kind.Name + "mma" : "f32";
        }

        double bound = bounds.Bound;
        bool narrow = f.Bits <= 8 || bounds.Within(f.Signed ? sbyte.MinValue : 0, f.Signed ? sbyte.MaxValue : byte.MaxValue);
        if (narrow && bound <= (f.Signed ? int.MaxValue : uint.MaxValue))
        {
            // Values that fit 4 bits may take the 4-bit tensor instruction: whichever of the 4- and 8-bit ones the device
            // runs faster (measured once per shape), where it has them.
            bool tiny = f.Bits <= 4 || bounds.Within(f.Signed ? -8 : 0, f.Signed ? 7 : 15);
            var kind = EightBitKind(f.Signed, bounds.Terms);
            var kind4 = f.Signed ? Ptx.Kind.S4 : Ptx.Kind.U4;
            bool has8 = MmaSupported(kind), has4 = tiny && MmaSupported(kind4);
            if (has4 && (!has8 || FasterKind(kind4, kind, bounds.Terms))) return kind4.Name + "mma";
            if (has8) return kind.Name + "mma";
            if (Dp4aSupported(f.Signed)) return kind.Name + "dp";
        }

        if (bound <= Exact.Float32) return "f32";
        if (bound <= (f.Signed ? int.MaxValue : uint.MaxValue)) return f.Signed ? "s32" : "u32";
        if (bound <= Exact.Float64) return "f64";
        return f.Signed ? "s64" : "u64";
    }

    public readonly record struct MmaConfig(int RowBytes, int Stages, int Bn, bool Swizzle);

    private readonly Dictionary<string, MmaConfig?> _shapes = [];

    /// <summary>The tensor kernel's shape for a kind: of the row widths, pipeline depths and block widths the device's
    /// shared memory takes and its JIT accepts, the one it runs fastest (measured once, on a 2048³ product); null when it has none.</summary>
    private MmaConfig? MmaShape(Ptx.Kind kind)
    {
        if (_shapes.TryGetValue(kind.Tag, out var known)) return known;
        MmaConfig? best = null;
        double bestMs = double.MaxValue;
        var report = new List<string>();
        const int Size = 2048;
        int kp = Ptx.Padded(Size);
        ulong x = gpu.Rent((long)Size * kp * kind.Bits / 8), y = gpu.Rent((long)Size * kp * kind.Bits / 8), z = gpu.Rent((long)Size * Size * 4);
        foreach (var cfg in new MmaConfig[]
        {
            new(64, 3, 128, false), new(64, 4, 128, false), new(128, 2, 128, false), new(64, 3, 256, false),
            new(64, 3, 256, true), new(64, 4, 256, true), new(128, 2, 256, true), new(128, 3, 128, true), new(64, 6, 128, true),
        })
        {
            int shared = Ptx.MmaShared(cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle);
            if (shared > gpu.SharedOptIn) continue;
            string ptx = Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out string entry);
            if (!gpu.Accepts(ptx, entry)) continue;
            var fn = gpu.Cached($"mma:{kind.Tag}:{cfg}", (out string e) => Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out e));
            gpu.AllowShared(fn, shared);
            double ms = TimeMma(fn, Dynamic(shared), cfg.Bn, x, y, z, Size, kp);
            report.Add($"rows {cfg.RowBytes} B × {cfg.Stages} stages × {cfg.Bn} wide{(cfg.Swizzle ? " swizzled" : "")} {ms:0.000} ms");
            if (ms < bestMs) (bestMs, best) = (ms, cfg);
        }

        gpu.Return(x); gpu.Return(y); gpu.Return(z);
        if (best is { } shape) Console.WriteLine($"device measured {kind.Tag} tensor kernel at {Size}³: {string.Join(", ", report)}: rows {shape.RowBytes} B × {shape.Stages} stages × {shape.Bn} wide{(shape.Swizzle ? " swizzled" : "")} kept");
        return _shapes[kind.Tag] = best;
    }

    private IntPtr MmaFunction(Ptx.Kind kind, out uint shared, out int bn)
    {
        var cfg = MmaShape(kind) ?? throw new InvalidOperationException($"{kind.Name}: no tensor kernel on this device");
        shared = Dynamic(Ptx.MmaShared(cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle));
        bn = cfg.Bn;
        return gpu.Cached($"mma:{kind.Tag}:{cfg}", (out string e) => Ptx.GemmMma(gpu, kind, cfg.RowBytes, cfg.Stages, cfg.Bn, cfg.Swizzle, out e));
    }

    // Shared bytes to ask for at launch: only what a kernel declares as extern (beyond the static limit).
    private static uint Dynamic(int bytes) => bytes > Ptx.StaticShared ? (uint)bytes : 0;

    private readonly Dictionary<string, bool> _supported = [];

    private bool MmaSupported(Ptx.Kind kind) => MmaShape(kind) is not null;

    private readonly Dictionary<bool, Ptx.Kind> _eightBit = [];

    // The 8-bit tensor instruction the device runs faster: one taking 16 or 32 values of k (measured once).
    private Ptx.Kind EightBitKind(bool signed, long terms)
    {
        if (_eightBit.TryGetValue(signed, out var known)) return known;
        var k16 = signed ? Ptx.Kind.S8 : Ptx.Kind.U8;
        var k32 = signed ? Ptx.Kind.S8K32 : Ptx.Kind.U8K32;
        bool has16 = MmaSupported(k16), has32 = MmaSupported(k32);
        var kind = has32 && (!has16 || FasterKind(k32, k16, terms)) ? k32 : k16;
        return _eightBit[signed] = kind;
    }

    private readonly Dictionary<string, bool> _faster = [];

    // Whether the first kind's kernel beats the second's on a square product of this many terms, measured once.
    private bool FasterKind(Ptx.Kind a, Ptx.Kind b, long terms)
    {
        string key = $"{a.Tag}<{b.Tag}:{terms}";
        if (_faster.TryGetValue(key, out bool faster)) return faster;
        int size = (int)Math.Min(terms, 2048), kp = Ptx.Padded(size);
        int bits = Math.Max(a.Bits, b.Bits);
        ulong x = gpu.Rent((long)size * kp * bits / 8), y = gpu.Rent((long)size * kp * bits / 8), z = gpu.Rent((long)size * size * 4);
        var fa = MmaFunction(a, out uint sa, out int bna);
        var fb = MmaFunction(b, out uint sb, out int bnb);
        double ta = TimeMma(fa, sa, bna, x, y, z, size, kp), tb = TimeMma(fb, sb, bnb, x, y, z, size, kp);
        gpu.Return(x); gpu.Return(y); gpu.Return(z);
        Console.WriteLine($"device measured {a.Tag} tensor kernel {ta:0.000} ms against {b.Tag} {tb:0.000} ms at {size}³: {(ta < tb ? a.Tag : b.Tag)} kept");
        return _faster[key] = ta < tb;
    }

    private double TimeMma(IntPtr fn, uint shared, int bn, ulong x, ulong y, ulong z, int size, int kp)
    {
        Gpu.Launch(fn, Grid(size, bn), Grid(size, 128), Ptx.GemmThreads, 1, shared, x, y, z, Gpu.U(size), Gpu.U(size), Gpu.U(kp), Gpu.U(kp));
        Gpu.Synchronize();
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < 3; i++) Gpu.Launch(fn, Grid(size, bn), Grid(size, 128), Ptx.GemmThreads, 1, shared, x, y, z, Gpu.U(size), Gpu.U(size), Gpu.U(kp), Gpu.U(kp));
        Gpu.Synchronize();
        return Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
    }

    private bool Dp4aSupported(bool signed)
    {
        if (_supported.TryGetValue("dp4a:" + signed, out bool known)) return known;
        return _supported["dp4a:" + signed] = gpu.Accepts(Ptx.GemmDot(gpu, signed, out string entry), entry);
    }

    private static Lane LaneOf(string name) => name switch
    {
        "f32" or "f16mma" or "bf16mma" => Lane.F32,
        "f64" => Lane.F64,
        "s32" or "s8dp" or "s8mma" or "s4mma" => Lane.S32,
        "u32" or "u8dp" or "u8mma" or "u4mma" => Lane.U32,
        "s64" => Lane.S64,
        _ => Lane.U64,
    };

    public string MatMul<TN>(ulong a, ulong b, ulong c, int m, int n, int k, in Bounds bounds) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        string laneName = ProductLane<TN>(bounds);
        var lane = LaneOf(laneName);
        var rented = new List<ulong>();
        try
        {
            if (laneName.EndsWith("mma", StringComparison.Ordinal) || laneName.EndsWith("dp", StringComparison.Ordinal))
            {
                var kind = laneName[..^(laneName.EndsWith("mma", StringComparison.Ordinal) ? 3 : 2)] switch
                {
                    "s8" => laneName.EndsWith("mma", StringComparison.Ordinal) ? EightBitKind(true, k) : Ptx.Kind.S8,
                    "u8" => laneName.EndsWith("mma", StringComparison.Ordinal) ? EightBitKind(false, k) : Ptx.Kind.U8,
                    "s4" => Ptx.Kind.S4, "u4" => Ptx.Kind.U4, "f16" => Ptx.Kind.F16, _ => Ptx.Kind.BF16,
                };
                int kp = Ptx.Padded(k);
                // A's rows as stored already have the kind's layout when the type is the kind and k is whole blocks.
                bool aAsIs = f.Bits == kind.Bits && kp == k && (f.IsFloat == kind.IsFloat) && (!f.IsFloat || f.Is(kind == Ptx.Kind.F16 ? "float16" : "bfloat16"));
                long t0 = Phases is null ? 0 : Stopwatch.GetTimestamp();
                ulong pa = aAsIs ? a : Pack(f, kind, a, m, k, kp, transposed: false, rented);
                t0 = Mark(t0, 0);
                ulong pb = Pack(f, kind, b, n, k, kp, transposed: true, rented);
                t0 = Mark(t0, 1);
                bool cAsIs = f.Bits == 32 && f.IsFloat == lane.IsFloat;
                ulong lc = cAsIs ? c : gpu.Rent((long)m * n * 4);
                if (!cAsIs) rented.Add(lc);
                if (laneName.EndsWith("mma", StringComparison.Ordinal))
                {
                    var fn = MmaFunction(kind, out uint shared, out int bn);
                    Gpu.Launch(fn, Grid(n, bn), Grid(m, 128), Ptx.GemmThreads, 1, shared, pa, pb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(kp), Gpu.U(kp));
                }
                else
                {
                    var fn = gpu.Cached($"dp4a:{f.Signed}", (out string e) => Ptx.GemmDot(gpu, f.Signed, out e));
                    Gpu.Launch(fn, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, 0, pa, pb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(kp / 4));
                }

                t0 = Mark(t0, 2);
                if (!cAsIs) Back(f, lane, lc, c, (long)m * n);
                Mark(t0, 3);
                return laneName;
            }

            if (f.Is("float64") && SlicedProven && SlicedFaster(m, n, k, a, b, c) && MatMulSliced(a, b, c, m, n, k)) return "f64x";
            bool direct = f.Natural == lane && f.Bits == lane.Size * 8 && (f.IsFloat == lane.IsFloat);
            ulong la = direct ? a : Lanes(f, lane, a, (long)m * k, rented);
            ulong lb = direct ? b : Lanes(f, lane, b, (long)k * n, rented);
            ulong lcl = direct ? c : gpu.Rent((long)m * n * lane.Size);
            if (!direct) rented.Add(lcl);
            if (!lane.Wide && k % 4 == 0 && n % 4 == 0)
            {
                // The staged kernel: its pipeline depth measured on the device once per lane.
                int stages = Gemm2Stages(lane, la, lb, lcl, m, n, k);
                var g2 = gpu.Cached($"gemm2:{lane.Name}:{stages}", (out string e) => Ptx.Gemm2(gpu, lane, stages, out e));
                gpu.AllowShared(g2, Ptx.Gemm2Shared(stages));
                Gpu.Launch(g2, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, Dynamic(Ptx.Gemm2Shared(stages)), la, lb, lcl, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            }
            else
            {
                int micro = Tile(lane, m, n, k, la, lb, lcl);
                var g = gpu.Cached($"gemm:{lane.Name}:{micro}", (out string e) => Ptx.Gemm(gpu, lane, micro, out e));
                int tile = micro * 16;
                Gpu.Launch(g, Grid(n, tile), Grid(m, tile), Ptx.GemmThreads, 1, 0, la, lb, lcl, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            }
            if (!direct) Back(f, lane, lcl, c, (long)m * n);
            return laneName;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    // ---------------------------------------------------------------- float64 products through the 8-bit tensor lane

    /// <summary>Set by the device proof (Checks.Gpu): the sliced product reproduces float64 math to within an ulp on this device.</summary>
    public bool SlicedProven { get; set; }

    private readonly Dictionary<string, bool> _slicedFaster = [];

    /// <summary>Whether the sliced product beats the plain float64 kernel at this shape (measured once per shape).</summary>
    private bool SlicedFaster(int m, int n, int k, ulong a, ulong b, ulong c)
    {
        string key = $"{m}x{n}x{k}";
        if (_slicedFaster.TryGetValue(key, out bool known)) return known;
        double Time(Action run)
        {
            run();
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) run();
            Gpu.Synchronize();
            return Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
        }

        double plain = Time(() => PlainF64(a, b, c, m, n, k));
        double sliced = Time(() => { if (!MatMulSliced(a, b, c, m, n, k)) throw new InvalidOperationException("sliced product refused"); });
        Console.WriteLine($"device measured float64 product at {key}: float64 lane {plain:0.000} ms, sliced 8-bit tensor lane {sliced:0.000} ms: {(sliced < plain ? "sliced" : "float64")} kept");
        return _slicedFaster[key] = sliced < plain;
    }

    private void PlainF64(ulong a, ulong b, ulong c, int m, int n, int k)
    {
        var lane = Lane.F64;
        var g = gpu.Cached($"gemm:{lane.Name}:4", (out string e) => Ptx.Gemm(gpu, lane, 4, out e));
        Gpu.Launch(g, Grid(n, 64), Grid(m, 64), Ptx.GemmThreads, 1, 0, a, b, c, Gpu.U(m), Gpu.U(n), Gpu.U(k));
    }

    /// <summary>C = A·B for float64 operands through the 8-bit tensor instructions (see Ptx: sliced float64). False when the
    /// device has no 8-bit tensor kind or the data's exponents are beyond what the slicing scale takes: nothing was computed.</summary>
    public bool MatMulSliced(ulong a, ulong b, ulong c, int m, int n, int k)
    {
        if (!MmaSupported(Ptx.Kind.S8) && !MmaSupported(Ptx.Kind.S8K32)) return false;
        var kind = EightBitKind(true, k);
        int kpad = Ptx.Padded(k), rowLen = Ptx.Slices * kpad;
        long mn = (long)m * n;
        ulong e = gpu.Rent(m * 4L), f = gpu.Rent(n * 4L), sa = gpu.Rent((long)m * rowLen), stack = gpu.Rent((long)rowLen * n), bt = gpu.Rent((long)n * rowLen), d = gpu.Rent(Ptx.Diagonals * mn * 4);
        try
        {
            var rowExp = gpu.Cached("rowexp_f64", (out string x) => Ptx.RowExponents(gpu, out x));
            var colExp = gpu.Cached("colexp_f64", (out string x) => Ptx.ColumnExponents(gpu, out x));
            var sliceRows = gpu.Cached("slice_rows_f64", (out string x) => Ptx.SliceRows(gpu, out x));
            var sliceCols = gpu.Cached("slice_cols_f64", (out string x) => Ptx.SliceColumns(gpu, out x));
            var combine = gpu.Cached("combine_f64x", (out string x) => Ptx.CombineSlices(gpu, out x));
            Gpu.Launch(rowExp, (uint)m, 1, 256, 1, 0, a, e, Gpu.U(k));
            Gpu.Launch(colExp, Grid(n, 256), 1, 256, 1, 0, b, f, Gpu.U(k), Gpu.U(n));
            // The exponents must keep every scale a normal float64 (the slicing scale 2^(64 − e), the result's 2^(e + f + shift − 72)).
            var ea = new int[m];
            var fa = new int[n];
            Gpu.Download<int>(e, ea);
            Gpu.Download<int>(f, fa);
            int eMin = ea.Min(), eMax = ea.Max(), fMin = fa.Min(), fMax = fa.Max();
            if (eMax > 1086 || fMax > 1086 || eMin + fMin < -918 || eMax + fMax > 1057) return false;
            if (kpad != k) Gpu.Clear(sa, (long)m * rowLen);
            Gpu.Clear(stack, (long)rowLen * n);
            Gpu.Launch(sliceRows, Grid((long)m * k, 256), 1, 256, 1, 0, a, e, sa, Gpu.U(m), Gpu.U(k), Gpu.U(kpad));
            Gpu.Launch(sliceCols, Grid((long)k * n, 256), 1, 256, 1, 0, b, f, stack, Gpu.U(k), Gpu.U(n), Gpu.U(kpad));
            // The stacked digits of B (rowLen × n, signed bytes) transposed into packed rows Bt[j][0..rowLen).
            var int8 = new Fmt("int8", 8, false, true);
            var ft = gpu.Cached($"packt:{int8.Name}:{kind.Name}", (out string x) => Ptx.PackTransposed(gpu, int8, kind, out x));
            Gpu.Launch(ft, Grid(n, 64), (uint)(rowLen / 32), 256, 1, 0, stack, bt, Gpu.U(n), Gpu.U(rowLen), Gpu.U(rowLen));
            // One product per diagonal d: the digit pairs (t, u) with t + u = d. B's digits are stacked most significant last,
            // so for d below the slice count A's first d + 1 slices meet B's last d + 1; above it, A's last 15 − d meet B's first.
            var fn = MmaFunction(kind, out uint shared, out int bn);
            for (int diag = 0; diag < Ptx.Diagonals; diag++)
            {
                int pairs = diag < Ptx.Slices ? diag + 1 : 2 * Ptx.Slices - 1 - diag, extent = pairs * kpad;
                ulong pa = diag < Ptx.Slices ? sa : sa + (ulong)((diag - Ptx.Slices + 1) * kpad);
                ulong pb = diag < Ptx.Slices ? bt + (ulong)((Ptx.Slices - 1 - diag) * kpad) : bt;
                ulong pd = d + (ulong)(diag * mn * 4);
                Gpu.Launch(fn, Grid(n, bn), Grid(m, 128), Ptx.GemmThreads, 1, shared, pa, pb, pd, Gpu.U(m), Gpu.U(n), Gpu.U(extent), Gpu.U(rowLen));
            }

            Gpu.Launch(combine, Grid(mn, 256), 1, 256, 1, 0, d, e, f, c, Gpu.U(m), Gpu.U(n));
            return true;
        }
        finally
        {
            foreach (var p in new[] { e, f, sa, stack, bt, d }) gpu.Return(p);
        }
    }

    private ulong Pack(Fmt f, Ptx.Kind kind, ulong stored, int rows, int k, int kp, bool transposed, List<ulong> rented)
    {
        ulong packed = gpu.Rent((long)rows * kp * kind.Bits / 8);
        rented.Add(packed);
        if (transposed)
        {
            var ft = gpu.Cached($"packt:{f.Name}:{kind.Name}", (out string e) => Ptx.PackTransposed(gpu, f, kind, out e));
            Gpu.Launch(ft, Grid(rows, 64), (uint)(kp / 32), 256, 1, 0, stored, packed, Gpu.U(rows), Gpu.U(k), Gpu.U(kp));
            return packed;
        }

        var fn = gpu.Cached($"pack:{f.Name}:{kind.Name}:n", (out string e) => Ptx.PackRows(gpu, f, kind, false, out e));
        long words = (long)rows * kp / kind.Per;
        Gpu.Launch(fn, Grid(words, 256), 1, 256, 1, 0, stored, packed, Gpu.U(rows), Gpu.U(k), Gpu.U(kp), Gpu.U(k));
        return packed;
    }

    private readonly Dictionary<string, int> _gemm2Stages = [];

    // The pipeline depth (2 or 3 stages) the device runs faster for this lane, measured once on the first product's shape.
    private int Gemm2Stages(Lane lane, ulong la, ulong lb, ulong lc, int m, int n, int k)
    {
        if (_gemm2Stages.TryGetValue(lane.Name, out int known)) return known;
        int best = 2;
        double bestMs = double.MaxValue;
        var report = new List<string>();
        foreach (int stages in new[] { 2, 3 })
        {
            int shared = Ptx.Gemm2Shared(stages);
            if (shared > gpu.SharedOptIn) continue;
            string ptx = Ptx.Gemm2(gpu, lane, stages, out string entry);
            if (!gpu.Accepts(ptx, entry)) continue;
            var fn = gpu.Cached($"gemm2:{lane.Name}:{stages}", (out string e) => Ptx.Gemm2(gpu, lane, stages, out e));
            gpu.AllowShared(fn, shared);
            Gpu.Launch(fn, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, Dynamic(shared), la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) Gpu.Launch(fn, Grid(n, 128), Grid(m, 128), Ptx.GemmThreads, 1, Dynamic(shared), la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            report.Add($"{stages} stages {ms:0.000} ms");
            if (ms < bestMs) (bestMs, best) = (ms, stages);
        }

        Console.WriteLine($"device measured {lane.Name} staged product at {m}x{n}x{k}: {string.Join(", ", report)}: {best} stages kept");
        return _gemm2Stages[lane.Name] = best;
    }

    // The faster of the 128 and 64 tiles for this lane and shape, measured on the device once (a 64-bit lane's
    // accumulators do not fit the 128 tile's registers: 64 only).
    private int Tile(Lane lane, int m, int n, int k, ulong la, ulong lb, ulong lc)
    {
        if (lane.Wide) return 4;
        string key = $"{lane.Name}:{m}x{n}x{k}";
        if (_tiles.TryGetValue(key, out int micro)) return micro;
        double best = double.MaxValue;
        foreach (int candidate in new[] { 8, 4 })
        {
            var fn = gpu.Cached($"gemm:{lane.Name}:{candidate}", (out string e) => Ptx.Gemm(gpu, lane, candidate, out e));
            int tile = candidate * 16;
            Gpu.Launch(fn, Grid(n, tile), Grid(m, tile), Ptx.GemmThreads, 1, 0, la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < 3; i++) Gpu.Launch(fn, Grid(n, tile), Grid(m, tile), Ptx.GemmThreads, 1, 0, la, lb, lc, Gpu.U(m), Gpu.U(n), Gpu.U(k));
            Gpu.Synchronize();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds / 3;
            if (ms < best) (best, micro) = (ms, candidate);
        }

        _tiles[key] = micro;
        return micro;
    }

    private ulong Lanes(Fmt f, Lane lane, ulong stored, long count, List<ulong> rented)
    {
        ulong buffer = gpu.Rent(count * lane.Size);
        rented.Add(buffer);
        var fn = gpu.Cached($"to:{f.Name}:{lane.Name}", (out string e) => Ptx.ToLanes(gpu, f, lane, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, stored, buffer, Gpu.U((int)count));
        return buffer;
    }

    private void Back(Fmt f, Lane lane, ulong lanes, ulong stored, long count)
    {
        var fn = gpu.Cached($"back:{lane.Name}:{f.Name}", (out string e) => Ptx.FromLanes(gpu, lane, f, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, lanes, stored, Gpu.U((int)count));
    }

    /// <summary>A stored array into a lane array (for checks).</summary>
    public void ToLanes<TN>(Lane lane, ulong stored, ulong lanes, int count) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var fn = gpu.Cached($"to:{f.Name}:{lane.Name}", (out string e) => Ptx.ToLanes(gpu, f, lane, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, stored, lanes, Gpu.U(count));
    }

    public void FromLanes<TN>(Lane lane, ulong lanes, ulong stored, int count) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var fn = gpu.Cached($"back:{lane.Name}:{f.Name}", (out string e) => Ptx.FromLanes(gpu, lane, f, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, lanes, stored, Gpu.U(count));
    }

    // ---------------------------------------------------------------- rows

    private static Lane RowLane(Fmt f) => f.Bits == 64 ? Lane.F64 : Lane.F32;

    // A row's block: enough threads (128 to 1024) that each holds about four words; the words per thread (a row in
    // registers up to 16; longer rows take the three-pass kernels).
    private static (int Threads, int WordsPerThread) RowShape(Fmt f, int cols)
    {
        long words = f.Words(cols);
        int threads = 128;
        while (threads < 1024 && (words + threads - 1) / threads > 4) threads *= 2;
        return (threads, (int)((words + threads - 1) / threads));
    }

    private static void CheckRows(Fmt f, int cols)
    {
        if ((long)cols * f.Bits % f.WordBits != 0) throw new NotSupportedException($"{f.Name}: a row of {cols} values does not end on a word; the device's row kernels take whole words.");
    }

    public string Softmax<TN>(ulong x, ulong y, int rows, int cols) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        CheckRows(f, cols);
        var lane = RowLane(f);
        var (threads, perThread) = RowShape(f, cols);
        bool vector = f.WordBits == 32 && perThread == 4 && f.Words(cols) % 4 == 0;
        var fn = perThread <= 16
            ? gpu.Cached($"softmax:{f.Name}:{perThread}:{threads}:{vector}", (out string e) => Ptx.SoftmaxRegs(gpu, f, lane, perThread, threads, vector, out e))
            : gpu.Cached($"softmaxloop:{f.Name}:{threads}", (out string e) => Ptx.Softmax(gpu, f, lane, threads, out e));
        Gpu.Launch(fn, (uint)rows, 1, (uint)threads, 1, 0, x, y, Gpu.U(cols));
        return lane.Name;
    }

    public string RmsNorm<TN>(ulong x, ulong y, int rows, int cols, double eps) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        CheckRows(f, cols);
        var lane = RowLane(f);
        var (threads, perThread) = RowShape(f, cols);
        bool vector = f.WordBits == 32 && perThread == 4 && f.Words(cols) % 4 == 0;
        var fn = perThread <= 16
            ? gpu.Cached($"rms:{f.Name}:{perThread}:{threads}:{vector}", (out string e) => Ptx.RmsNormRegs(gpu, f, lane, perThread, threads, vector, out e))
            : gpu.Cached($"rmsloop:{f.Name}:{threads}", (out string e) => Ptx.RmsNorm(gpu, f, lane, threads, out e));
        Gpu.Launch(fn, (uint)rows, 1, (uint)threads, 1, 0, x, y, Gpu.U(cols), lane.Wide ? Gpu.D(eps) : Gpu.F((float)eps));
        return lane.Name;
    }
}
