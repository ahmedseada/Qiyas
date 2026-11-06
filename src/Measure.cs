// The one way a choice between two paths is measured, on either device: which of two runs that give the same result is
// faster on this machine at this shape. A single timed call says little: code the runtime has not yet compiled to its
// final tier, a processor or device still raising its clocks, a team of threads waking, the timer's own grain at short
// runs. So: each path warmed once; a round times a path over enough calls to fill a couple of milliseconds (the count
// set from the warm call); the paths timed in alternating rounds, the minimum per call of each kept; at least three
// rounds, and on until neither minimum has improved by more than 2 % and the runtime has compiled nothing for three rounds
// in a row (a path that is still speeding up — a tier arriving, clocks rising — keeps the rounds going), at most 24
// rounds. Found by the block product's choice: its body still in the runtime's first tier (20× slower) for the first
// rounds of a run, two single calls a side chose the decoded route; the bodies of a choice's paths are also compiled at
// the final tier from the first call (AggressiveOptimization), as the hot kernels are.
using System.Diagnostics;

namespace Qiyas;

internal static class Measure
{
    private const double RoundMs = 2.0, Gain = 0.98;
    private const int MinRounds = 3, Settled = 3, MaxRounds = 24, MaxCalls = 64;

    /// <summary>The settled time per call of each path, in ms. sync (the device's) runs after a path's calls and is timed with them.</summary>
    public static (double First, double Second) Faster(Action first, Action second, Action? sync = null)
    {
        double Once(Action run, int calls)
        {
            sync?.Invoke();                                                    // nothing earlier still queued in the timing
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < calls; i++) run();
            sync?.Invoke();
            return Stopwatch.GetElapsedTime(t0).TotalMilliseconds / calls;
        }

        int Calls(double ms) => (int)Math.Clamp(Math.Ceiling(RoundMs / Math.Max(ms, 1e-3)), 1, MaxCalls);

        int n1 = Calls(Once(first, 1)), n2 = Calls(Once(second, 1));
        double t1 = double.MaxValue, t2 = double.MaxValue;
        int still = 0;
        for (int round = 0; round < MaxRounds; round++)
        {
            long compiled = System.Runtime.JitInfo.GetCompiledMethodCount();
            double a = Once(first, n1), b = Once(second, n2);
            bool gained = a < t1 * Gain || b < t2 * Gain || System.Runtime.JitInfo.GetCompiledMethodCount() != compiled;
            t1 = Math.Min(t1, a);
            t2 = Math.Min(t2, b);
            still = gained ? 0 : still + 1;
            if (round + 1 >= MinRounds && still >= Settled) break;
        }

        return (t1, t2);
    }
}
