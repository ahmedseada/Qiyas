// The Qiyas bench: every subject the assembly holds (IBenchSubject: Qiyas, the INumberBase<T> version, and whatever
// else is dropped in), on every stored type, on the CPU and on CUDA, scored against float64 math on the same inputs.
//
//   floats   float64 float32 float16 bfloat16 float8_e4m3 float8_e5m2
//   signed   int64 int32 int16 int8 int4 int2
//   unsigned uint64 uint32 uint16 uint8 uint2
//
// Usage: Qiyas.dll [cpu|cuda|cuda,cpu] [--subjects a,b] [--types t1,t2] [--ops add,matmul,softmax,rms_norm] [--baseline name]
//        Qiyas.dll profile | gpuprofile | team       (Qiyas's own diagnostics)
//
// Every proof in Checks must pass before anything is timed. Times are medians over an adaptive number of runs.
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Qiyas;
using Qiyas.Bench;

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var options = Options.Parse(args);
Bench.Odd = options.Shapes == "odd";
Console.WriteLine($"bench: {Environment.ProcessorCount} logical processors, .NET {Environment.Version}");
if (!Checks.Run())
{
    Console.WriteLine("A check failed: no timing until it is fixed.");
    return 1;
}

if (options.Report) Console.WriteLine(Qiyas.Kernels.Report());

if (options.Mode == "team")
{
    TeamBench.Run();
    return 0;
}

if (options.Mode == "profile")
{
    foreach (var type in options.Types) Diagnostics.Profile(type);
    return 0;
}

if (options.Mode == "gpuprofile")
{
    if (!Devices.HasCuda) { Console.WriteLine("cuda: no CUDA driver on this machine."); return 1; }
    Console.WriteLine($"cuda: {Devices.Gpu.Name}");
    foreach (var type in options.Types) Diagnostics.GpuProfile(type);
    return 0;
}

var subjects = Subjects.Discover(options.Subjects);
Console.WriteLine($"subjects: {string.Join(", ", subjects.Select(s => s.Name))}; baseline: {options.BaselineOf(subjects)}");
var bench = new Bench(subjects, options);
foreach (var device in options.Devices)
{
    if (device == Device.Cuda)
    {
        if (!Devices.HasCuda)
        {
            Console.WriteLine("cuda: no CUDA driver on this machine; skipped.");
            continue;
        }

        if (!Checks.Gpu(Devices.Ops))
        {
            Console.WriteLine("A device check failed: no timing until it is fixed.");
            return 1;
        }

        if (options.Report) Console.WriteLine(Devices.Ops.Report());
    }

    foreach (var s in subjects)
    {
        if (s.Describe(device) is { } line) Console.WriteLine(line);
    }

    foreach (var type in options.Types) bench.Run(device, type);
}

bench.Write(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", $"results-qiyas-{string.Join('-', options.Devices.Select(d => d.ToString().ToLowerInvariant()))}.md"));
if (options.Report)
{
    Console.WriteLine("measured during the run");
    Console.Write(Qiyas.Kernels.Measured());
    if (options.Devices.Contains(Device.Cuda) && Devices.HasCuda) Console.Write(Devices.Ops.Measured());
}
return 0;

// ---------------------------------------------------------------------------------------------- the command line

internal sealed class Options
{
    public string Mode = "bench";
    public List<Device> Devices = [Device.Cpu];
    public List<string> Types = [.. Registry.Names];
    public List<string> Ops = ["add", "matmul", "softmax", "rms_norm"];

    public static readonly string[] Core = ["add", "matmul", "softmax", "rms_norm"];

    public static readonly string[] Group2 =
    [
        "sum:whole", "sum:rows", "sum:cols", "sumsq:rows", "max:whole", "max:rows", "max:cols", "min:rows", "argmax:rows", "argmax:cols",
        "maprow:add", "maprow:mul", "transpose", "gemv", "gemv_t",
    ];
    public static readonly string[] Group3 = [.. RowOps.All.Select(o => "row:" + o.Name), "norm_params_bwd:rms", "norm_params_bwd:layer"];
    public static readonly string[] Group4 = [.. LayoutArgs.Names.Select(n => "layout:" + n)];
    public static readonly string[] Group4b = ["matmul_nt", "matmul_tn", "matmul_tt", "matmul_batched", "matmul_bwd", "matmul_blocks"];
    public static readonly string[] Group5 = ["attention", "attention_causal", "attention_fused", "attention_fused_causal", "attention_decode", "attention_bwd"];
    public static readonly string[] Group7 = [.. OptimizerArgs.Names.Select(n => "optimizer:" + n)];
    public static readonly string[] Group6 = ["conv", "conv_bwd"];
    public static readonly string[] Group8 = ["lstm_step", "gru_step", "lstm_step_bwd", "gru_step_bwd", "lstm_seq", "lstm_seq_bwd", "gru_seq", "gru_seq_bwd"];
    public static readonly string[] Group9 = ["topk_rows", "pick_rows", "sample_rows", "route", "route_wide", "cross_entropy", "ctc", "ctc_bwd"];
    public List<string>? Subjects;
    public string? Baseline;

    /// <summary>"regular" (the default sizes) or "odd": every row's sizes shifted to primes (no dimension ends on a tile, a word or a block).</summary>
    public string Shapes = "regular";

    /// <summary>With --report: print the device's facts and the lanes chosen for the run's operations before the rows.</summary>
    public bool Report;

    /// <summary>--repeat N: every row measured N times over (a stress for intermittent failures; the last measurement is kept).</summary>
    public int Repeat = 1;

    public static Options Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            List<string> Next() => [.. args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
            switch (a)
            {
                case "--subjects": o.Subjects = Next(); break;
                case "--types": o.Types = Next(); break;
                case "--ops":
                    o.Ops = Next();
                    if (o.Ops.Contains("all")) o.Ops = [.. Core, .. Qiyas.Ops.All.Select(m => "map:" + m.Name), .. Group2, .. Group3, .. Group4, .. Group4b, .. Group5, .. Group6, .. Group7, .. Group8, .. Group9];
                    if (o.Ops.Contains("map")) o.Ops = [.. o.Ops.Where(n => n != "map"), .. Qiyas.Ops.All.Select(m => "map:" + m.Name)];
                    if (o.Ops.Contains("reduce")) o.Ops = [.. o.Ops.Where(n => n != "reduce"), .. Group2];
                    if (o.Ops.Contains("rows")) o.Ops = [.. o.Ops.Where(n => n != "rows"), .. Group3];
                    if (o.Ops.Contains("layout")) o.Ops = [.. o.Ops.Where(n => n != "layout"), .. Group4];
                    if (o.Ops.Contains("products")) o.Ops = [.. o.Ops.Where(n => n != "products"), .. Group4b];
                    if (o.Ops.Contains("attention")) o.Ops = [.. o.Ops.Where(n => n != "attention"), .. Group5];
                    if (o.Ops.Contains("optimizers")) o.Ops = [.. o.Ops.Where(n => n != "optimizers"), .. Group7];
                    if (o.Ops.Contains("conv")) o.Ops = [.. o.Ops.Where(n => n != "conv"), .. Group6];
                    if (o.Ops.Contains("recurrent")) o.Ops = [.. o.Ops.Where(n => n != "recurrent"), .. Group8];
                    if (o.Ops.Contains("routing")) o.Ops = [.. o.Ops.Where(n => n != "routing"), .. Group9];
                    break;
                case "--baseline": o.Baseline = args[++i]; break;
                case "--shapes": o.Shapes = args[++i]; if (o.Shapes is not ("regular" or "odd")) throw new ArgumentException($"--shapes takes regular or odd, not {o.Shapes}"); break;
                case "--report": o.Report = true; break;
                case "--repeat": o.Repeat = int.Parse(args[++i]); Qiyas.Checks.Stress = Math.Max(Qiyas.Checks.Stress, o.Repeat); break;
                case "profile" or "gpuprofile" or "team": o.Mode = a; break;
                default:
                    o.Devices = [.. a.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                        .Select(d => d switch { "cpu" => Device.Cpu, "cuda" => Device.Cuda, _ => throw new ArgumentException($"unknown device {d}") })];
                    break;
            }
        }

        foreach (var t in o.Types) if (!Registry.Names.Contains(t)) throw new ArgumentException($"unknown type {t}");
        return o;
    }

    /// <summary>The subject the × column is taken against: the one named, else "lib" when present, else the first.</summary>
    public string BaselineOf(IReadOnlyList<IBenchSubject> subjects) =>
        Baseline ?? (subjects.Any(s => s.Name == "lib") ? "lib" : subjects[0].Name);
}

internal static class Subjects
{
    /// <summary>Every IBenchSubject with a parameterless constructor in the loaded assemblies, by name.</summary>
    public static List<IBenchSubject> Discover(List<string>? only)
    {
        var found = AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(a => { try { return a.GetTypes(); } catch { return []; } })
            .Where(t => !t.IsAbstract && typeof(IBenchSubject).IsAssignableFrom(t) && t.GetConstructor(Type.EmptyTypes) is not null)
            .Select(t => (IBenchSubject)Activator.CreateInstance(t)!)
            .OrderBy(s => s.Name, StringComparer.Ordinal)
            .ToList();
        if (only is not null) found = found.Where(s => only.Contains(s.Name)).ToList();
        if (found.Count == 0) throw new InvalidOperationException("no bench subjects found");
        return found;
    }
}

// ---------------------------------------------------------------------------------------------- the bench

internal sealed record Cell(string Subject, string Lane, double Ms, double Err);

internal sealed record Row(Device Device, string Op, string Type, string Shape, List<Cell?> Cells);

internal sealed class Bench(IReadOnlyList<IBenchSubject> subjects, Options options)
{
    public static int AddCount => Odd ? (1 << 24) - 5 : 1 << 24;
    // The regular sizes; --shapes odd takes the nearest primes below (4093, 4091; 16M − 5; …) so no size ends on a tile, a word or a block.
    public static bool Odd;
    public static int Rows => Odd ? 4093 : 4096;
    public static int Cols => Odd ? 4091 : 4096;
    public static int Prime(int n) => Odd ? NearestPrimeBelow(n) : n;

    private static int NearestPrimeBelow(int n)
    {
        for (int c = n - 1; c > 2; c--)
        {
            bool prime = true;
            for (int d = 2; (long)d * d <= c; d++) if (c % d == 0) { prime = false; break; }
            if (prime) return c;
        }

        return n;
    }
    public const double Eps = 1e-6;
    private readonly List<Row> _rows = [];
    private readonly string _baseline = options.BaselineOf(subjects);

    public static int MatSize(Device device) => Prime(device == Device.Cuda ? 2048 : 1024);

    public void Run(Device device, string type)
    {
        if (options.Ops.Contains("add"))
        {
            var a = Registry.Make(type, AddCount, 1, -1, 1);
            var b = Registry.Make(type, AddCount, 2, -1, 1);
            double[] reference = Registry.Values(type, a, AddCount), vb = Registry.Values(type, b, AddCount);
            for (int i = 0; i < reference.Length; i++) reference[i] += vb[i];
            Measure(device, "add", type, $"{AddCount}", reference, (s, c) => s.Add(device, type, a, b, c, AddCount), a.Length);
        }

        if (options.Ops.Contains("matmul"))
        {
            int s = MatSize(device);
            var a = Registry.Make(type, s * s, 3, -1, 1);
            var b = Registry.Make(type, s * s, 4, -1, 1);
            var reference = References.MatMul(Registry.Values(type, a, s * s), Registry.Values(type, b, s * s), s, s, s);
            for (int rep = 0; rep < options.Repeat; rep++) Measure(device, "matmul", type, $"{s}x{s}x{s}", reference, (subject, c) => subject.MatMul(device, type, a, b, c, s, s, s), a.Length);
        }

        foreach (var op in Qiyas.Ops.All)
        {
            if (!options.Ops.Contains("map:" + op.Name)) continue;
            if (op.FloatOnly && !Registry.IsFloat(type)) continue;
            MapRow(device, type, op);
        }

        foreach (var name in Options.Group2)
        {
            if (options.Ops.Contains(name)) Group2Row(device, type, name);
        }

        foreach (var name in Options.Group3)
        {
            if (options.Ops.Contains(name)) Group3Row(device, type, name);
        }

        foreach (var name in Options.Group4)
        {
            if (options.Ops.Contains(name)) Group4Row(device, type, name[7..]);
        }

        foreach (var name in Options.Group4b)
        {
            if (options.Ops.Contains(name)) ProductRow(device, type, name);
        }

        foreach (var name in Options.Group5)
        {
            if (options.Ops.Contains(name)) AttentionRow(device, type, name);
        }

        foreach (var name in Options.Group7)
        {
            if (options.Ops.Contains(name)) OptimizerRow(device, type, name[10..]);
        }

        foreach (var name in Options.Group6)
        {
            if (options.Ops.Contains(name)) ConvRow(device, type, name);
        }

        foreach (var name in Options.Group8)
        {
            if (options.Ops.Contains(name)) RecurrentRow(device, type, name);
        }

        foreach (var name in Options.Group9)
        {
            if (options.Ops.Contains(name)) RoutingRow(device, type, name);
        }

        if (!Registry.IsFloat(type)) return;
        int count = Rows * Cols;
        if (options.Ops.Contains("softmax"))
        {
            var x = Registry.Make(type, count, 5, -4, 4);
            var reference = References.Softmax(Registry.Values(type, x, count), Rows, Cols);
            Measure(device, "softmax", type, $"{Rows}x{Cols}", reference, (s, y) => s.Softmax(device, type, x, y, Rows, Cols), x.Length);
        }

        if (options.Ops.Contains("rms_norm"))
        {
            var x = Registry.Make(type, count, 6, -2, 2);
            var reference = References.RmsNorm(Registry.Values(type, x, count), Rows, Cols, Eps);
            Measure(device, "rms_norm", type, $"{Rows}x{Cols}", reference, (s, y) => s.RmsNorm(device, type, x, y, Rows, Cols, Eps), x.Length);
        }
    }

    public const int MapCount = 1 << 24;

    // Reductions, broadcasts, transposes at 4096 × 4096; matrix · vector at 8192 × 8192.
    private void Group2Row(Device device, string type, string name)
    {
        int r = Rows, c = Cols;
        if (name.StartsWith("gemv", StringComparison.Ordinal))
        {
            bool t = name == "gemv_t";
            int m = Prime(8192), k = Prime(8192);
            var a = Registry.Make(type, m * k, 21, -1, 1);
            var x = Registry.Make(type, t ? m : k, 22, -1, 1);
            var reference = References.Gemv(Registry.Values(type, a, m * k), Registry.Values(type, x, t ? m : k), m, k, t);
            Measure(device, name, type, $"{m}x{k}", reference, (s, y) => s.Gemv(device, type, a, x, y, m, k, t), (int)Registry.Bytes(type, t ? k : m));
            return;
        }

        var data = Registry.Make(type, r * c, 23, -1, 1);
        var values = Registry.Values(type, data, r * c);
        if (name == "transpose")
        {
            Measure(device, name, type, $"{r}x{c}", References.Transpose(values, r, c), (s, o) => s.Transpose(device, type, data, o, r, c), data.Length);
            return;
        }

        if (name.StartsWith("maprow:", StringComparison.Ordinal))
        {
            var op = Qiyas.Ops.Find(name[7..]);
            var v = Registry.Make(type, c, 24, -1, 1);
            var reference = References.MapRow(type, op, values, Registry.Values(type, v, c), r, c);
            Measure(device, name, type, $"{r}x{c}", reference, (s, o) => s.MapRow(device, op, type, data, v, o, r, c), data.Length);
            return;
        }

        string rop = name[..name.IndexOf(':')];
        var axis = name.EndsWith(":rows", StringComparison.Ordinal) ? Axis.Rows : name.EndsWith(":cols", StringComparison.Ordinal) ? Axis.Cols : Axis.Whole;
        int count = axis == Axis.Whole ? 1 : axis == Axis.Rows ? r : c;
        var reduced = References.Reduce(rop, axis, values, r, c);
        MeasureLanes(device, name, type, $"{r}x{c}", reduced, (s, o) => s.Reduce(device, rop, axis, type, data, r, c, o), count);
    }

    // Routing, sampling and the loss: top-8 of 4096 rows of 32768 (a vocabulary), pick and sample over the same rows, a
    // route of 1M indices into 64 buckets, cross-entropy as log_softmax then pick (the loss row scored: −log p at the target).
    private void RoutingRow(Device device, string type, string name)
    {
        int rows = Prime(4096), cols = Prime(32768), k = 8;
        var random = new Random(97);
        switch (name)
        {
            case "topk_rows":
            {
                var x = Registry.Make(type, rows * cols, 95, -4, 4);
                var (want, _) = Routing.TopKReference(Registry.Values(type, x, rows * cols), rows, cols, k);
                var indices = new int[rows * k];
                Measure(device, name, type, $"{rows}x{cols} k{k}", want, (s, values) => s.TopK(device, type, x, rows, cols, k, values, indices), (int)Registry.Bytes(type, rows * k));
                return;
            }

            case "pick_rows":
            {
                var x = Registry.Make(type, rows * cols, 95, -4, 4);
                var index = Enumerable.Range(0, rows).Select(_ => random.Next(cols)).ToArray();
                var want = Routing.PickReference(Registry.Values(type, x, rows * cols), rows, cols, index);
                Measure(device, name, type, $"{rows}x{cols}", want, (s, o) => s.Pick(device, type, x, rows, cols, index, o), (int)Registry.Bytes(type, rows));
                return;
            }

            case "sample_rows":
            {
                if (!Registry.IsFloat(type)) return;
                var x = Registry.Make(type, rows * cols, 95, 0, 1);
                var u = Registry.Make(type, rows, 96, 0, 1);
                var want = Routing.SampleReference(Registry.Values(type, x, rows * cols), rows, cols, Registry.Values(type, u, rows));
                MeasureIndices(device, name, type, $"{rows}x{cols}", want, (s, picks) => s.Sample(device, type, x, rows, cols, u, picks));
                return;
            }

            case "route":
            case "route_wide":                                               // 4f: many targets (two radix passes)
            {
                if (type != Registry.Names[0]) return;                       // no type: once
                int n = 1 << 20, buckets = name == "route" ? 64 : 65_536;
                var index = Enumerable.Range(0, n).Select(_ => random.Next(buckets)).ToArray();
                var (wantCounts, wantOffsets, wantOrder) = Routing.RouteReference(index, buckets);
                var counts = new int[buckets];
                var offsets = new int[buckets + 1];
                MeasureIndices(device, name, "s32", $"{n} into {buckets}", wantOrder, (s, order) => s.Route(device, index, buckets, counts, offsets, order), extra: () => counts.SequenceEqual(wantCounts) && offsets.SequenceEqual(wantOffsets) ? 0 : 1);
                return;
            }

            case "ctc":
            case "ctc_bwd":
            {
                // ctc over 256 frames of 32 sequences, 1024 classes (blank 0), 64 labels each, the sequences 256, 248, 240,
                // 232 frames in turn: the loss scored (ctc), the gradient scored (ctc_bwd, the loss computed with it).
                if (!Registry.IsFloat(type)) return;
                int T = Prime(256), B = Prime(32), C = Prime(1024), L = Prime(64);
                var labelRandom = new Random(98);
                var a = new CtcArgs(T, B, C, 0, [.. Enumerable.Range(0, B).Select(b => T - b % 4 * 8)], [.. Enumerable.Repeat(L, B)], [.. Enumerable.Range(0, B * L).Select(_ => 1 + labelRandom.Next(C - 1))]);
                var x = Registry.Make(type, (int)a.Values, 99, -4, 4);
                var (wantLoss, wantGrad) = a.Reference(Registry.Values(type, x, (int)a.Values));
                string shape = $"T{T} B{B} C{C} L{L}";
                if (name == "ctc") Measure(device, name, type, shape, wantLoss, (s, o) => s.Ctc(device, type, a, x, o, null), (int)Registry.Bytes(type, B));
                else
                {
                    var loss = new byte[Registry.Bytes(type, B)];
                    Measure(device, name, type, shape, wantGrad, (s, o) => s.Ctc(device, type, a, x, loss, o), (int)Registry.Bytes(type, a.Values));
                }

                return;
            }

            case "cross_entropy":
            {
                if (!Registry.IsFloat(type)) return;
                var x = Registry.Make(type, rows * cols, 95, -4, 4);
                var targets = Enumerable.Range(0, rows).Select(_ => random.Next(cols)).ToArray();
                var xv = Registry.Values(type, x, rows * cols);
                var want = new double[rows];
                for (int r = 0; r < rows; r++)
                {
                    double m = double.NegativeInfinity, sum = 0;
                    for (int c = 0; c < cols; c++) m = Math.Max(m, xv[r * cols + c]);
                    for (int c = 0; c < cols; c++) sum += Math.Exp(xv[r * cols + c] - m);
                    want[r] = m + Math.Log(sum) - xv[r * cols + targets[r]];
                }

                var logp = new byte[x.Length];
                var op = RowOps.Find("log_softmax");
                var neg = Qiyas.Ops.Find("neg");
                Measure(device, name, type, $"{rows}x{cols}", want, (s, o) =>
                {
                    // log_softmax, pick at the target, negate: three of the subject's runs timed as one.
                    var a = s.RowOp(device, op, type, x, null, null, null, null, null, logp, null, rows, cols, 0, 0);
                    var b = s.Pick(device, type, logp, rows, cols, targets, o);
                    var c = s.Map(device, neg, type, o, null, null, o, rows, 0, 0);
                    b?.Dispose(); c?.Dispose();
                    if (a is null || b is null || c is null) { a?.Dispose(); return null; }
                    // The pick and the negation are prepared after the step before them has collected: a device subject uploads
                    // its inputs when a run is prepared, so runs prepared up front read logp and o before log_softmax wrote them.
                    return new BenchRun(() =>
                    {
                        a.Run(); a.Collect();
                        using (var pick = s.Pick(device, type, logp, rows, cols, targets, o)!) { pick.Run(); pick.Collect(); }
                        using (var negate = s.Map(device, neg, type, o, null, null, o, rows, 0, 0)!) { negate.Run(); negate.Collect(); }
                    }, null, () => a.Dispose(), () => a.Lane + " (log_softmax, pick, neg)");
                }, (int)Registry.Bytes(type, rows));
                return;
            }
        }
    }

    // A result of s32 indices, compared exactly with the reference (extra: a further check, 0 when it holds).
    private void MeasureIndices(Device device, string op, string type, string shape, int[] reference, Func<IBenchSubject, int[], IBenchRun?> prepare, Func<int>? extra = null)
    {
        var cells = new List<Cell?>();
        foreach (var subject in subjects)
        {
            var output = new int[reference.Length];
            using var run = prepare(subject, output);
            if (run is null) { cells.Add(null); continue; }
            double ms = Time(Scoped(device, subject, run));
            run.Collect();
            double worst = extra?.Invoke() ?? 0;
            for (int i = 0; i < reference.Length; i++) worst = Math.Max(worst, Math.Abs((double)output[i] - reference[i]));
            cells.Add(new Cell(subject.Name, run.Lane, ms, worst));
        }

        var row = new Row(device, op, type, shape, cells);
        _rows.Add(row);
        Console.WriteLine(Line(row));
    }

    // A recurrent step over a batch of 1024 rows, input 1024, hidden 1024 (h' scored; the backward: dx scored, the rest computed too).
    private void RecurrentRow(Device device, string type, string name)
    {
        if (!Registry.IsFloat(type)) return;
        if (name == "gru_seq")
        {
            // 16 steps of a batch of 256, input 512, hidden 512, as lstm_seq (h at every step scored); the block formats too.
            var ga = new CellArgs("gru", Prime(256), Prime(512), Prime(512));
            const int T = 16;
            int GH = ga.Gates * ga.H;
            double sc = 1 / Math.Sqrt(ga.D + ga.H);
            var x = Registry.Make(type, T * ga.B * ga.D, 91, -1, 1); var h0 = Registry.Make(type, ga.B * ga.H, 92, -1, 1);
            var w = Registry.Make(type, GH * ga.D, 94, -sc, sc); var u = Registry.Make(type, GH * ga.H, 95, -sc, sc);
            var b = Registry.Make(type, GH, 96, -0.1, 0.1); var bh = Registry.Make(type, GH, 97, -0.1, 0.1);
            double[] V(byte[] s, int n) => Registry.Values(type, s, n);
            var wantH = RecurrentCase.GruSequenceReference(ga, T, V(x, T * ga.B * ga.D), V(h0, ga.B * ga.H), V(w, GH * ga.D), V(u, GH * ga.H), V(b, GH), V(bh, GH));
            Measure(device, name, type, $"T{T} B{ga.B} D{ga.D} H{ga.H}", wantH, (s, hOut) => s.GruSequence(device, type, ga, T, x, h0, w, u, b, bh, hOut), (int)Registry.Bytes(type, T * ga.B * ga.H));
            return;
        }

        if (name == "gru_seq_bwd")
        {
            // The GRU sequence's backward at gru_seq's shape (dx scored; dh0, dW, dU, db, dbh computed too); the block formats too.
            var ga = new CellArgs("gru", Prime(256), Prime(512), Prime(512));
            const int T = 16;
            int GH = ga.Gates * ga.H;
            double sc = 1 / Math.Sqrt(ga.D + ga.H);
            var x = Registry.Make(type, T * ga.B * ga.D, 91, -1, 1); var h0 = Registry.Make(type, ga.B * ga.H, 92, -1, 1);
            var w = Registry.Make(type, GH * ga.D, 94, -sc, sc); var u = Registry.Make(type, GH * ga.H, 95, -sc, sc);
            var b = Registry.Make(type, GH, 96, -0.1, 0.1); var bh = Registry.Make(type, GH, 97, -0.1, 0.1);
            var dhOut = Registry.Make(type, T * ga.B * ga.H, 98, -1, 1);
            double[] V(byte[] s, int n) => Registry.Values(type, s, n);
            var (wdx, _, _, _, _, _) = RecurrentCase.GruSequenceBackwardReference(ga, T, V(x, T * ga.B * ga.D), V(h0, ga.B * ga.H), V(w, GH * ga.D), V(u, GH * ga.H), V(b, GH), V(bh, GH), V(dhOut, T * ga.B * ga.H));
            var dh0 = new byte[h0.Length]; var dw = new byte[w.Length]; var du = new byte[u.Length]; var db = new byte[b.Length]; var dbh = new byte[bh.Length];
            Measure(device, name, type, $"T{T} B{ga.B} D{ga.D} H{ga.H}", wdx, (s, dx) => s.GruSequenceBackward(device, type, ga, T, x, h0, w, u, b, bh, dhOut, dx, dh0, dw, du, db, dbh), x.Length);
            return;
        }

        if (name.StartsWith("lstm_seq", StringComparison.Ordinal))
        {
            // 16 steps of a batch of 256, input 512, hidden 512 (h at every step scored; the backward: dx scored).
            if (Registry.IsBlock(type)) return;
            var sa = new CellArgs("lstm", Prime(256), Prime(512), Prime(512));
            const int T = 16;
            int GH = sa.Gates * sa.H;
            double sc = 1 / Math.Sqrt(sa.D + sa.H);
            var x = Registry.Make(type, T * sa.B * sa.D, 81, -1, 1); var h0 = Registry.Make(type, sa.B * sa.H, 82, -1, 1); var c0 = Registry.Make(type, sa.B * sa.H, 83, -1, 1);
            var w = Registry.Make(type, GH * sa.D, 84, -sc, sc); var u = Registry.Make(type, GH * sa.H, 85, -sc, sc); var b = Registry.Make(type, GH, 86, -0.1, 0.1);
            double[] V(byte[] s, int n) => Registry.Values(type, s, n);
            string seqShape = $"T{T} B{sa.B} D{sa.D} H{sa.H}";
            if (name == "lstm_seq")
            {
                var (wantH, _) = RecurrentCase.LstmSequenceReference(sa, T, V(x, T * sa.B * sa.D), V(h0, sa.B * sa.H), V(c0, sa.B * sa.H), V(w, GH * sa.D), V(u, GH * sa.H), V(b, GH));
                var cOut = new byte[Registry.Bytes(type, T * sa.B * sa.H)];
                Measure(device, name, type, seqShape, wantH, (s, hOut) => s.LstmSequence(device, type, sa, T, x, h0, c0, w, u, b, hOut, cOut, null, null, null, null, null, null), (int)Registry.Bytes(type, T * sa.B * sa.H));
                return;
            }

            var dhOut = Registry.Make(type, T * sa.B * sa.H, 88, -1, 1); var dcLast = Registry.Make(type, sa.B * sa.H, 89, -1, 1);
            var (wdx, _, _, _, _, _) = RecurrentCase.LstmSequenceBackwardReference(sa, T, V(x, T * sa.B * sa.D), V(h0, sa.B * sa.H), V(c0, sa.B * sa.H), V(w, GH * sa.D), V(u, GH * sa.H), V(b, GH), V(dhOut, T * sa.B * sa.H), V(dcLast, sa.B * sa.H));
            var dh0 = new byte[h0.Length]; var dc0 = new byte[c0.Length]; var dw = new byte[w.Length]; var du = new byte[u.Length]; var db = new byte[b.Length];
            Measure(device, name, type, seqShape, wdx, (s, dx) => s.LstmSequence(device, type, sa, T, x, h0, c0, w, u, b, dhOut, dcLast, dx, dh0, dc0, dw, du, db), x.Length);
            return;
        }

        bool lstm = name.StartsWith("lstm", StringComparison.Ordinal), backward = name.EndsWith("_bwd", StringComparison.Ordinal);
        var a = new CellArgs(lstm ? "lstm" : "gru", Prime(1024), Prime(1024), Prime(1024));
        var k = RecurrentCase.Make(a, type, backward);
        string shape = $"B{a.B} D{a.D} H{a.H}";
        if (backward)
        {
            Measure(device, name, type, shape, k.WantDx!, (s, dx) => s.Recurrent(device, type, a, k.X, k.H, k.C, k.W, k.U, k.Bias, k.BiasH, k.DhOut!, k.DcOut, dx, k.Dh, k.Dc, k.Dw, k.Du, k.Db, k.Dbh), k.X.Length);
            return;
        }

        Measure(device, name, type, shape, k.WantH, (s, hOut) => s.Recurrent(device, type, a, k.X, k.H, k.C, k.W, k.U, k.Bias, k.BiasH, hOut, k.COut, null, null, null, null, null, null, null), k.H.Length);
    }

    // A convolution of a 64 × 128 × 128 image by 128 filters of 3 × 3 (pad 1); the backward pair (dx scored, dw computed too).
    private void ConvRow(Device device, string type, string name)
    {
        var a = new ConvArgs(Prime(64), Prime(128), Prime(128), Prime(128), 3, 3, 1, 1, 1, 1);
        var x = Registry.Make(type, (int)a.XValues, 61, -1, 1);
        var w = Registry.Make(type, (int)a.WValues, 62, -1, 1);
        string shape = $"{a.C}x{a.H}x{a.W} by {a.OC} of {a.Kh}x{a.Kw}";
        if (name == "conv_bwd")
        {
            if (!Registry.IsFloat(type)) return;
            var dy = Registry.Make(type, (int)a.YValues, 63, -1, 1);
            var (wantX, _) = a.ReferenceBackward(Registry.Values(type, x, (int)a.XValues), Registry.Values(type, w, (int)a.WValues), Registry.Values(type, dy, (int)a.YValues));
            var dw = new byte[w.Length];
            Measure(device, name, type, shape, wantX, (s, dx) => s.Conv(device, type, a, x, w, dy, dx, dw), x.Length);
            return;
        }

        var want = a.Reference(Registry.Values(type, x, (int)a.XValues), Registry.Values(type, w, (int)a.WValues));
        Measure(device, name, type, shape, want, (s, y) => s.Conv(device, type, a, x, w, y, null, null), (int)Registry.Bytes(type, a.YValues));
    }

    // An optimizer step over 16M parameters (the moments in lane storage); p scored against float64 (m, v updated too).
    private void OptimizerRow(Device device, string type, string name)
    {
        if (!Registry.IsFloat(type)) return;
        var a = new OptimizerArgs(name, 1e-3, 0.9, 0.999, 1e-8, 0.01, 7);
        int n = MapCount, size = type == "float64" ? 8 : 4;
        var p = Registry.Make(type, n, 71, -1, 1);
        var g = Registry.Make(type, n, 72, -1, 1);
        var mv = new double[n];
        var vv = a.Moments > 1 ? new double[n] : null;
        var random = new Random(73);
        for (int i = 0; i < n; i++) { mv[i] = random.NextDouble() - 0.5; if (vv is not null) vv[i] = random.NextDouble(); }
        var m = LaneBytes(size, mv);
        var v = vv is null ? null : LaneBytes(size, vv);
        var pv = Registry.Values(type, p, n);
        a.Reference(pv, Registry.Values(type, g, n), (double[])mv.Clone(), vv is null ? null : (double[])vv.Clone());
        Measure(device, "optimizer:" + name, type, $"{n}", pv, (s, o) => { p.CopyTo(o, 0); return s.Optimizer(device, type, a, o, g, (byte[])m.Clone(), v is null ? null : (byte[])v.Clone(), n); }, p.Length);
    }

    private static byte[] LaneBytes(int size, double[] values)
    {
        var bytes = new byte[values.Length * size];
        if (size == 8) System.Runtime.InteropServices.MemoryMarshal.Cast<double, byte>(values.AsSpan()).CopyTo(bytes);
        else for (int i = 0; i < values.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), (float)values[i]);
        return bytes;
    }

    // Attention: 16 heads of 2048 queries over 2048 keys at d = 128 (plain and causal); decode: 32 heads, one query over
    // 4096 keys; backward: 8 heads of 1024 at d = 128 (dQ scored; dK, dV computed too). Float types.
    private void AttentionRow(Device device, string type, string name)
    {
        if (!Registry.IsFloat(type)) return;
        AttentionArgs a = name switch
        {
            "attention" or "attention_fused" => new(16, Prime(2048), Prime(2048), Prime(128), 1 / Math.Sqrt(128), false),
            "attention_causal" or "attention_fused_causal" => new(16, Prime(2048), Prime(2048), Prime(128), 1 / Math.Sqrt(128), true),
            "attention_decode" => new(32, 1, Prime(4096), Prime(128), 1 / Math.Sqrt(128), false),
            _ => new(8, Prime(1024), Prime(1024), Prime(128), 1 / Math.Sqrt(128), true),
        };
        var q = Registry.Make(type, (int)a.QValues, 41, -1, 1);
        var k = Registry.Make(type, (int)a.KValues, 42, -1, 1);
        var v = Registry.Make(type, (int)a.KValues, 43, -1, 1);
        var qv = Registry.Values(type, q, (int)a.QValues);
        var kv = Registry.Values(type, k, (int)a.KValues);
        var vv = Registry.Values(type, v, (int)a.KValues);
        string shape = $"{a.Heads}h {a.Lq}x{a.Lk} d{a.D}";
        if (name == "attention_bwd")
        {
            var dO = Registry.Make(type, (int)a.QValues, 44, -1, 1);
            var (wantQ, _, _) = a.ReferenceBackward(qv, kv, vv, Registry.Values(type, dO, (int)a.QValues));
            var dk = new byte[k.Length];
            var dv = new byte[v.Length];
            Measure(device, name, type, shape, wantQ, (s, dq) => s.AttentionBackward(device, type, a, q, k, v, dO, dq, dk, dv), q.Length);
            return;
        }

        var want = a.Reference(qv, kv, vv);
        if (name.StartsWith("attention_fused", StringComparison.Ordinal)) { Measure(device, name, type, shape, want, (s, o) => s.AttentionFused(device, type, a, q, k, v, o), q.Length); return; }
        Measure(device, name, type, shape, want, (s, o) => s.Attention(device, type, a, q, k, v, o, name == "attention_decode"), q.Length);
    }

    // Products on views at the core product's size: transposed operands, 16 batches of a quarter size, the backward pair (dA and dB; dA scored).
    private void ProductRow(Device device, string type, string name)
    {
        int s = MatSize(device), s4 = s / 4;
        if (name == "matmul_blocks")
        {
            // A·Bᵀ of two Q8_0 / Q4_0 operands by the block-scaled product alone (matmul_nt takes it where measured faster).
            if (type is not ("q8_0" or "q4_0")) return;
            var blockView = new ProductView(s, s, s, false, true);
            var ba = Registry.Make(type, s * s, 31, -1, 1);
            var bb = Registry.Make(type, s * s, 32, -1, 1);
            var bwant = blockView.Reference(Registry.Values(type, ba, s * s), Registry.Values(type, bb, s * s));
            Measure(device, name, type, $"{s}x{s}x{s} Bᵀ", bwant, (subject, c) => subject.MatMulBlocks(device, type, s, s, s, ba, bb, c), (int)Registry.Bytes(type, s * s));
            return;
        }

        if (name == "matmul_bwd")
        {
            var a = Registry.Make(type, s * s, 31, -1, 1);
            var b = Registry.Make(type, s * s, 32, -1, 1);
            var dc = Registry.Make(type, s * s, 33, -1, 1);
            var reference = new ProductView(s, s, s, false, true).Reference(Registry.Values(type, dc, s * s), Registry.Values(type, b, s * s));
            var db = new byte[a.Length];
            Measure(device, name, type, $"{s}x{s}x{s} (dA, dB)", reference, (subject, da) => subject.MatMulBackward(device, type, a, b, dc, da, db, s, s, s), a.Length);
            return;
        }

        ProductView v = name switch
        {
            "matmul_nt" => new(s, s, s, false, true),
            "matmul_tn" => new(s, s, s, true, false),
            "matmul_tt" => new(s, s, s, true, true),
            _ => new(s4, s4, s4, false, false, 16, s4 * s4, s4 * s4, s4 * s4),
        };
        var av = Registry.Make(type, (int)v.ValuesA, 31, -1, 1);
        var bv = Registry.Make(type, (int)v.ValuesB, 32, -1, 1);
        var want = v.Reference(Registry.Values(type, av, (int)v.ValuesA), Registry.Values(type, bv, (int)v.ValuesB));
        string shape = v.Batch > 1 ? $"{v.Batch} x {v.M}x{v.N}x{v.K}" : $"{v.M}x{v.N}x{v.K}{(v.TransA ? " Aᵀ" : "")}{(v.TransB ? " Bᵀ" : "")}";
        Measure(device, name, type, shape, want, (subject, c) => subject.MatMulView(device, type, v, av, bv, c), (int)Registry.Bytes(type, v.ValuesC));
    }

    // A layout operation at a realistic shape (16M outputs for the moves; a 64 × 128 × 128 image, 3 × 3 kernel for the
    // convolution layouts; 16 × 512 × 512 → 1024 × 1024 for the resizes).
    private void Group4Row(Device device, string type, string name)
    {
        if (!LayoutArgs.Applies(name, Registry.IsFloat(type))) return;
        var k = LayoutCases.Bench(name, type);
        var reference = LayoutArgs.Reference(name, Registry.Values(type, k.X, k.InputCount), k.Index, k.P, k.Count, k.X2 is null ? null : Registry.Values(type, k.X2, LayoutCases.SecondCount(k)));
        Measure(device, "layout:" + name, type, LayoutCases.Shape(k), reference, (s, o) => s.Layout(device, type, k.With(o)), (int)Registry.Bytes(type, k.Count));
    }

    // A row operation at the bench's shape (float types only: the row family computes in a float lane); the parameter
    // backward's two column vectors come back as lane values.
    private void Group3Row(Device device, string type, string name)
    {
        if (!Registry.IsFloat(type)) return;
        if (name.StartsWith("norm_params_bwd:", StringComparison.Ordinal))
        {
            bool centered = name.EndsWith(":layer", StringComparison.Ordinal);
            var x = Registry.Make(type, Rows * Cols, 51, -4, 4);
            var dy = Registry.Make(type, Rows * Cols, 52, -1, 1);
            var dgamma = new double[Cols];
            var dbeta = new double[Cols];
            RowOps.NormParamsReference(Registry.Values(type, x, Rows * Cols), Registry.Values(type, dy, Rows * Cols), Rows, Cols, 1e-5, centered, dgamma, dbeta);
            MeasureLanes(device, name, type, $"{Rows}x{Cols}", [.. dgamma, .. dbeta], (s, o) => s.NormParamsBwd(device, type, x, dy, o, Rows, Cols, 1e-5, centered), 2 * Cols);
            return;
        }

        var op = RowOps.Find(name[4..]);
        var k = RowCase.Make(op, type, Rows, Cols, positions: false);
        var y2 = op.SecondOut ? new byte[k.X.Length] : null;
        Measure(device, name, type, $"{Rows}x{Cols}", k.Y, (s, y) => s.RowOp(device, op, type, k.X, k.X2, k.G, k.Bt, k.S, k.Pos, y, y2, Rows, Cols, op.A, op.B), k.X.Length);
    }

    // A reduction's result comes back as lane values: decoded by the lane the subject names.
    private void MeasureLanes(Device device, string op, string type, string shape, double[] reference, Func<IBenchSubject, byte[], IBenchRun?> prepare, int count)
    {
        var cells = new List<Cell?>();
        foreach (var subject in subjects)
        {
            var output = new byte[count * 8];
            using var run = prepare(subject, output);
            if (run is null) { cells.Add(null); continue; }
            double ms = Time(Scoped(device, subject, run));
            run.Collect();
            var got = LaneValues.Decode(run.Lane, output, count);
            double worst = 0;
            for (int i = 0; i < count; i++) worst = Math.Max(worst, Math.Abs(got[i] - reference[i]));
            cells.Add(new Cell(subject.Name, run.Lane, ms, worst));
        }

        var row = new Row(device, op, type, shape, cells);
        _rows.Add(row);
        Console.WriteLine(Line(row));
    }


    // An element-wise formula at 4M values: the inputs the formula's ranges ask for, the reference the formula on their values.
    private void MapRow(Device device, string type, MapOp op)
    {
        byte[]? x = op.Inputs >= 1 ? Registry.Make(type, MapCount, 11, op.XRange.Low, op.XRange.High) : null;
        byte[]? y = op.Inputs >= 2 ? Registry.Make(type, MapCount, 12, op.YRange.Low, op.YRange.High) : null;
        byte[]? z = op.Inputs >= 3 ? Registry.Make(type, MapCount, 13, op.YRange.Low, op.YRange.High) : null;
        var reference = References.Map(type, op, x, y, z, MapCount);
        Measure(device, "map:" + op.Name, type, $"{MapCount}", reference, (s, o) => s.Map(device, op, type, x, y, z, o, MapCount, op.A, op.B), (int)Registry.Bytes(type, MapCount));
    }

    private void Measure(Device device, string op, string type, string shape, double[] reference, Func<IBenchSubject, byte[], IBenchRun?> prepare, int outputBytes)
    {
        var cells = new List<Cell?>();
        foreach (var subject in subjects)
        {
            var output = new byte[outputBytes];
            using var run = prepare(subject, output);
            if (run is null) { cells.Add(null); continue; }
            double ms = Time(Scoped(device, subject, run));
            run.Collect();
            cells.Add(new Cell(subject.Name, run.Lane, ms, Err(type, output, reference)));
        }

        var row = new Row(device, op, type, shape, cells);
        _rows.Add(row);
        Console.WriteLine(Line(row));
    }

    private static double Err(string type, byte[] got, double[] reference)
    {
        var v = Registry.Values(type, got, reference.Length);
        double worst = 0;
        for (int i = 0; i < v.Length; i++)
        {
            double e = Math.Abs(v[i] - reference[i]);
            worst = double.IsNaN(e) ? double.PositiveInfinity : Math.Max(worst, e);
        }

        return worst;
    }

    // Qiyas on the device: each timed call inside one arena scope (src/Arena.cs), so its buffers come from one region with no
    // pool lookup; the buffers prepared outside the call stay the pool's.
    private static Action Scoped(Device device, IBenchSubject subject, IBenchRun run) =>
        device == Device.Cuda && subject is QiyasSubject ? () => { using var scope = Devices.Ops.Scope(); run.Run(); } : run.Run;

    // Median wall time (ms) after one warm-up: at least 7 runs and 300 ms of them (at most 201 runs; past 3 runs, at
    // most 3 s), so a millisecond operation is timed over hundreds of runs and none by a handful.
    public static double Time(Action run)
    {
        run();
        var times = new List<double>();
        double total = 0;
        while (times.Count < 201 && (times.Count < 7 || total < 300) && !(times.Count >= 3 && total > 3000))
        {
            long start = Stopwatch.GetTimestamp();
            run();
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            times.Add(ms);
            total += ms;
        }

        times.Sort();
        return times[times.Count / 2];
    }

    private double? BaselineMs(Row r) => r.Cells.FirstOrDefault(c => c?.Subject == _baseline)?.Ms;

    private string Line(Row r)
    {
        var sb = new StringBuilder(string.Create(CultureInfo.InvariantCulture, $"{r.Device.ToString().ToLowerInvariant(),-4} {r.Op,-8} {r.Type,-12}"));
        double? baseMs = BaselineMs(r);
        foreach (var c in r.Cells)
        {
            if (c is null) { sb.Append("  —"); continue; }
            sb.Append(string.Create(CultureInfo.InvariantCulture, $"  {c.Subject} {c.Ms,8:0.000}"));
            if (baseMs is { } bm && c.Subject != _baseline) sb.Append(string.Create(CultureInfo.InvariantCulture, $" (x{bm / c.Ms:0.00})"));
            sb.Append($" [{c.Lane}]");
        }

        sb.Append("  err");
        foreach (var c in r.Cells) sb.Append(c is null ? " —" : string.Create(CultureInfo.InvariantCulture, $" {c.Err:0.##e+0}"));
        return sb.ToString();
    }

    public void Write(string path)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Qiyas bench");
        sb.AppendLine();
        sb.AppendLine(string.Create(CultureInfo.InvariantCulture,
            $"{DateTime.Now:yyyy-MM-dd HH:mm}. Median wall time (ms); `×` = {_baseline} ÷ subject (above 1: faster than {_baseline}). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in."));
        sb.AppendLine();
        var names = subjects.Select(s => s.Name).ToList();
        sb.Append("| device | operation | type | shape |");
        foreach (var n in names) sb.Append($" {n} ms | {n} × | {n} lane | {n} err |");
        sb.AppendLine();
        sb.Append("|---|---|---|---|");
        foreach (var _ in names) sb.Append("---:|---:|---|---:|");
        sb.AppendLine();
        foreach (var r in _rows)
        {
            sb.Append($"| {r.Device.ToString().ToLowerInvariant()} | {r.Op} | {r.Type} | {r.Shape} |");
            double? baseMs = BaselineMs(r);
            foreach (var c in r.Cells)
            {
                if (c is null) { sb.Append(" — | — | — | — |"); continue; }
                string x = baseMs is { } bm && c.Subject != _baseline ? (bm / c.Ms).ToString("0.00", CultureInfo.InvariantCulture) : "—";
                sb.Append(string.Create(CultureInfo.InvariantCulture, $" {c.Ms:0.000} | {x} | {c.Lane} | {c.Err:0.##e+0} |"));
            }

            sb.AppendLine();
        }

        File.WriteAllText(Path.GetFullPath(path), sb.ToString());
        Console.WriteLine($"results: {Path.GetFullPath(path)}");
    }
}

// Float64 references: what each operation means, computed in float64 on the stored values.
internal static class References
{
    public static double[] Reduce(string op, Axis axis, double[] x, int rows, int cols)
    {
        int r = axis == Axis.Whole ? 1 : rows, c = axis == Axis.Whole ? rows * cols : cols;
        int count = axis == Axis.Cols ? c : r;
        var result = new double[count];
        for (int i = 0; i < count; i++)
        {
            int n = axis == Axis.Cols ? r : c;
            double acc = op == "max" || op == "argmax" ? double.NegativeInfinity : op == "min" ? double.PositiveInfinity : 0;
            int at = -1;
            for (int j = 0; j < n; j++)
            {
                double v = axis == Axis.Cols ? x[j * c + i] : x[i * c + j];
                switch (op)
                {
                    case "sum": acc += v; break;
                    case "sumsq": acc += v * v; break;
                    case "max": acc = Math.Max(acc, v); break;
                    case "min": acc = Math.Min(acc, v); break;
                    case "argmax": if (v > acc) { acc = v; at = j; } break;
                }
            }

            result[i] = op == "argmax" ? at : acc;
        }

        return result;
    }

    public static double[] MapRow(string type, MapOp op, double[] x, double[] v, int rows, int cols)
    {
        var r = new double[rows * cols];
        bool integer = !Registry.IsFloat(type), signed = !type.StartsWith("uint", StringComparison.Ordinal) && type != "bool";
        double m = integer ? Math.Pow(2, Registry.Bits(type)) : 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                double y = op.Reference(x[i * cols + j], v[j], 0, 0, 0);
                if (integer)
                {
                    // The lanes' arithmetic is modular at the type's width (as Map's reference wraps).
                    y = Math.Truncate(y);
                    if (signed) { if (y >= m / 2) y -= m; else if (y < -m / 2) y += m; }
                    else { if (y < 0) y += m; else if (y >= m) y -= m; }
                }

                r[i * cols + j] = y;
            }

        return r;
    }

    public static double[] Transpose(double[] x, int rows, int cols)
    {
        var r = new double[rows * cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) r[j * rows + i] = x[i * cols + j];
        return r;
    }

    public static double[] Gemv(double[] a, double[] x, int rows, int cols, bool transposed)
    {
        var y = new double[transposed ? cols : rows];
        if (transposed) { for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) y[j] += a[i * cols + j] * x[i]; }
        else { for (int i = 0; i < rows; i++) { double s = 0; for (int j = 0; j < cols; j++) s += a[i * cols + j] * x[j]; y[i] = s; } }
        return y;
    }

    /// <summary>The formula on the stored values; integer types truncate a division as their lanes do.</summary>
    public static double[] Map(string type, MapOp op, byte[]? x, byte[]? y, byte[]? z, int n, bool wrap = true)
    {
        double[]? vx = x is null ? null : Registry.Values(type, x, n);
        double[]? vy = y is null ? null : Registry.Values(type, y, n);
        double[]? vz = z is null ? null : Registry.Values(type, z, n);
        bool integer = !Registry.IsFloat(type);
        int bits = Registry.Bits(type);
        bool signed = type.StartsWith("int", StringComparison.Ordinal);
        double a = op.A, b = op.B;
        if (integer)
        {
            double lo = signed ? -Math.Pow(2, bits - 1) : 0, hi = signed ? Math.Pow(2, bits - 1) - 1 : Math.Pow(2, bits) - 1;
            a = Math.Clamp(a, lo, hi);
            b = Math.Clamp(b, lo, hi);
        }

        var r = new double[n];
        Parallel.For(0, n, i =>
        {
            double v = op.Reference(vx?[i] ?? 0, vy?[i] ?? 0, vz?[i] ?? 0, a, b);
            if (integer)
            {
                // The lanes' arithmetic is modular at the type's width.
                v = Math.Truncate(v);
                if (wrap)
                {
                    double m = Math.Pow(2, bits);
                    if (signed) { if (v >= m / 2) v -= m; else if (v < -m / 2) v += m; }
                    else { if (v < 0) v += m; else if (v >= m) v -= m; }
                }
            }

            r[i] = v;
        });

        return r;
    }

    /// <summary>The float64 reference product, accumulated in double-double (each product exact by a fused multiply-add,
    /// each sum with its error term), so a lane that is exact reads 0 and a float64 lane shows its own rounding.</summary>
    public static double[] MatMul(double[] a, double[] b, int m, int n, int k)
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

    public static double[] Softmax(double[] x, int rows, int cols)
    {
        var y = new double[x.Length];
        Parallel.For(0, rows, r =>
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int j = 0; j < cols; j++) max = Math.Max(max, x[r * cols + j]);
            for (int j = 0; j < cols; j++) sum += y[r * cols + j] = Math.Exp(x[r * cols + j] - max);
            for (int j = 0; j < cols; j++) y[r * cols + j] /= sum;
        });
        return y;
    }

    public static double[] RmsNorm(double[] x, int rows, int cols, double eps)
    {
        var y = new double[x.Length];
        Parallel.For(0, rows, r =>
        {
            double sum = 0;
            for (int j = 0; j < cols; j++) sum += x[r * cols + j] * x[r * cols + j];
            double scale = 1 / Math.Sqrt(sum / cols + eps);
            for (int j = 0; j < cols; j++) y[r * cols + j] = x[r * cols + j] * scale;
        });
        return y;
    }
}

// ---------------------------------------------------------------------------------------------- Qiyas's own diagnostics

internal static class Diagnostics
{
    /// <summary>Matmul phases on the CPU over 100 runs (after 200, past the JIT's tiering): the range scan, load, pack, product, store.</summary>
    public static void Profile(string type) => Registry.With(type, new CpuProfile());

    public static void GpuProfile(string type) => Registry.With(type, new DeviceProfile());

    private sealed class CpuProfile : ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            int s = Bench.MatSize(Device.Cpu);
            var ma = Registry.Make(TN.Name, s * s, 3, -1, 1);
            var mb = Registry.Make(TN.Name, s * s, 4, -1, 1);
            var qys = new byte[ma.Length];
            string lane = "";
            var phases = new double[5];
            var whole = new double[100];
            for (int run = -200; run < whole.Length; run++)
            {
                if (run == 0) { Array.Clear(phases); Kernels.Phases = phases; }
                long start = Stopwatch.GetTimestamp();
                var (aLow, aHigh) = TN.Range(ma, s * s);
                var (bLow, bHigh) = TN.Range(mb, s * s);
                var bounds = new Bounds(aLow, aHigh, bLow, bHigh, s);
                if (run >= 0) phases[4] += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                var product = new Kernels.MatMulVisitor(ma, mb, qys, s, s, s);
                TN.Sum(ref product, in bounds);
                lane = product.Lane;
                if (run >= 0) whole[run] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }

            Kernels.Phases = null;
            Array.Sort(whole);
            int r = whole.Length;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"profile matmul {TN.Name,-12} lane {lane,-6} min {whole[0],7:0.000} p10 {whole[r / 10],7:0.000} median {whole[r / 2],7:0.000} p90 {whole[r * 9 / 10],7:0.000} ms | " +
                $"range {phases[4] / r,6:0.000} load {phases[0] / r,6:0.000} pack {phases[1] / r,6:0.000} product {phases[2] / r,6:0.000} store {phases[3] / r,6:0.000}"));
            return true;
        }
    }

    /// <summary>The device product's phases over 20 runs (after 5): pack A, pack B, product, back; and the range scans.</summary>
    private sealed class DeviceProfile : ITypeVisitor<bool>
    {
        public bool Visit<TN>() where TN : struct, IQiyasNumber<TN, float>
        {
            var ops = Devices.Ops;
            var gpu = ops.Gpu;
            int s = Bench.MatSize(Device.Cuda);
            var ma = Registry.Make(TN.Name, s * s, 3, -1, 1);
            var mb = Registry.Make(TN.Name, s * s, 4, -1, 1);
            ulong da = gpu.Rent(ma.Length), db = gpu.Rent(mb.Length), dc = gpu.Rent(ma.Length);
            Gpu.Upload<byte>(ma, da);
            Gpu.Upload<byte>(mb, db);
            var phases = new double[5];
            string lane = "";
            double whole = 0;
            for (int run = -5; run < 20; run++)
            {
                if (run == 0) { Array.Clear(phases); GpuOps.Phases = phases; }
                long start = Stopwatch.GetTimestamp();
                var ((aLow, aHigh), (bLow, bHigh)) = ops.RangePair<TN>(da, s * s, db, s * s);
                Gpu.Synchronize();
                if (run >= 0) phases[4] += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                var bounds = new Bounds(aLow, aHigh, bLow, bHigh, s);
                lane = ops.MatMul<TN>(da, db, dc, s, s, s, in bounds);
                Gpu.Synchronize();
                if (run >= 0) whole += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }

            GpuOps.Phases = null;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"gpu profile matmul {TN.Name,-12} lane {lane,-7} whole {whole / 20,7:0.000} ms | range {phases[4] / 20,6:0.000} packA {phases[0] / 20,6:0.000} packB {phases[1] / 20,6:0.000} product {phases[2] / 20,6:0.000} back {phases[3] / 20,6:0.000}"));
            foreach (var p in new[] { da, db, dc }) gpu.Return(p);
            return true;
        }
    }
}

// The team's dispatch cost: an empty job of 48 chunks, back to back and after a pause that lets the workers sleep.
internal static class TeamBench
{
    private readonly struct Nothing : Team.IBody
    {
        public void Run(int first, int last) { }
    }

    public static void Run()
    {
        Measure("back to back", 0);
        Measure("after 5 ms pauses", 5);
        Measure("after 50 ms pauses", 50);
    }

    private static void Measure(string name, int pauseMs)
    {
        var times = new double[pauseMs == 0 ? 2000 : 200];
        for (int i = -20; i < times.Length; i++)
        {
            if (pauseMs > 0) Thread.Sleep(pauseMs);
            long start = Stopwatch.GetTimestamp();
            Team.For(48, 1, 1 << 20, new Nothing());
            if (i >= 0) times[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds * 1000;
        }

        Array.Sort(times);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"team dispatch {name,-20} min {times[0],8:0.0} median {times[times.Length / 2],8:0.0} p90 {times[times.Length * 9 / 10],8:0.0} p99 {times[times.Length * 99 / 100],8:0.0} max {times[^1],8:0.0} µs"));
    }
}
