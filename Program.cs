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
Console.WriteLine($"bench: {Environment.ProcessorCount} logical processors, .NET {Environment.Version}");
if (!Checks.Run())
{
    Console.WriteLine("A check failed: no timing until it is fixed.");
    return 1;
}

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
    }

    foreach (var s in subjects)
    {
        if (s.Describe(device) is { } line) Console.WriteLine(line);
    }

    foreach (var type in options.Types) bench.Run(device, type);
}

bench.Write(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", $"results-qiyas-{string.Join('-', options.Devices.Select(d => d.ToString().ToLowerInvariant()))}.md"));
return 0;

// ---------------------------------------------------------------------------------------------- the command line

internal sealed class Options
{
    public string Mode = "bench";
    public List<Device> Devices = [Device.Cpu];
    public List<string> Types = [.. Registry.Names];
    public List<string> Ops = ["add", "matmul", "softmax", "rms_norm"];
    public List<string>? Subjects;
    public string? Baseline;

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
                case "--ops": o.Ops = Next(); break;
                case "--baseline": o.Baseline = args[++i]; break;
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
    public const int AddCount = 1 << 24;
    public const int Rows = 4096, Cols = 4096;
    public const double Eps = 1e-6;
    private readonly List<Row> _rows = [];
    private readonly string _baseline = options.BaselineOf(subjects);

    public static int MatSize(Device device) => device == Device.Cuda ? 2048 : 1024;

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
            Measure(device, "matmul", type, $"{s}x{s}x{s}", reference, (subject, c) => subject.MatMul(device, type, a, b, c, s, s, s), a.Length);
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

    private void Measure(Device device, string op, string type, string shape, double[] reference, Func<IBenchSubject, byte[], IBenchRun?> prepare, int outputBytes)
    {
        var cells = new List<Cell?>();
        foreach (var subject in subjects)
        {
            var output = new byte[outputBytes];
            using var run = prepare(subject, output);
            if (run is null) { cells.Add(null); continue; }
            double ms = Time(run.Run);
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
    public static double[] MatMul(double[] a, double[] b, int m, int n, int k)
    {
        var c = new double[m * n];
        Parallel.For(0, m, i =>
        {
            var row = c.AsSpan(i * n, n);
            for (int p = 0; p < k; p++)
            {
                double x = a[i * k + p];
                var bp = b.AsSpan(p * n, n);
                for (int j = 0; j < n; j++) row[j] += x * bp[j];
            }
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
