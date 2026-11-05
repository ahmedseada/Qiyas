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
}

/// <summary>An IBenchRun from delegates.</summary>
public sealed class BenchRun(Action run, Action? collect = null, Action? dispose = null, Func<string>? lane = null) : IBenchRun
{
    public void Run() => run();

    public void Collect() => collect?.Invoke();

    public string Lane => lane?.Invoke() ?? "";

    public void Dispose() => dispose?.Invoke();
}

/// <summary>The CUDA device the subjects share (created on first use; null when the machine has no driver).</summary>
public static class Devices
{
    private static GpuOps? s_ops;

    public static bool HasCuda => Driver.Available;

    public static GpuOps Ops => s_ops ??= new GpuOps(new Gpu());

    public static Gpu Gpu => Ops.Gpu;
}
