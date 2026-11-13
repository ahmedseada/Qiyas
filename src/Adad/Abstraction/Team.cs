// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Adad's own worker team: one thread per logical processor less the caller, kept for the process. A job is a struct
// body cut into chunks; the caller and the workers take chunks from one counter, and the caller returns the moment
// the last chunk completes: a worker that wakes late finds no chunk left and delays nothing. Waiting workers spin
// (a job that follows another starts at once), for up to 20 ms, then sleep on their own event; a job wakes the
// spinning workers first, and a waiting worker sits at a lower priority so that whatever else must run takes a
// spinner's core rather than a working thread's. Nothing allocates per job: the body is passed by address and called through a function
// pointer of its type, and the body is read only by a worker that has taken a chunk, so it is never read after the
// caller returns. A job is closed when its last chunk completes (its chunk counter set past any count), so a worker
// that read the old generation's counter and then the next job's chunk count cannot take a chunk under a description
// half overwritten (seen once as an out-of-range product tile).
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

internal static unsafe class Team
{
    public interface IBody
    {
        void Run(int first, int last);
    }

    /// <summary>Threads a job runs on (the caller included).</summary>
    public static readonly int Size = Environment.ProcessorCount;

    private const long InlineWork = 1 << 15;                                 // jobs this small run on the caller
    private static readonly long s_spinBudget = Stopwatch.Frequency / 50;    // 20 ms of spinning before a worker sleeps

    private static readonly object s_gate = new();
    private static Worker[]? s_workers;
    private static delegate*<void*, int, int, void> s_invoke;
    private static void* s_body;
    private static int s_count, s_size, s_chunks;
    private static int s_generation;

    // The counters every thread writes, each on 128 bytes of its own (a line and the line the processor fetches with it):
    // the job's state (its generation in the high 32 bits, the next chunk in the low), taken by a compare-exchange a chunk,
    // and the chunks done, counted by each thread and read by the caller as it waits. As plain statics they fell wherever
    // the process had placed Team's statics, beside each other and other hot fields: an empty 48-chunk job took 13-15 us
    // in most processes and 6 in some (with or without a CUDA part before, the team made first or last), and the rows
    // that dispatch often followed it (quantize_rows float16 2.35-2.55 ms or 1.11). Apart they take 4.1-4.6 us in every
    // process (quantize_rows float16 0.80-0.91).
    [StructLayout(LayoutKind.Explicit, Size = 384)]
    private struct Counters
    {
        [FieldOffset(128)] public long State;
        [FieldOffset(256)] public int Done;
    }

    private static readonly Counters[] s_counters = new Counters[1];
    private static ref long s_state => ref s_counters[0].State;
    private static ref int s_done => ref s_counters[0].Done;
    private static ExceptionDispatchInfo? s_error;

    [ThreadStatic]
    private static bool t_inside;

    /// <summary>Runs <paramref name="body"/> over [0, count) in chunks of <paramref name="size"/>.</summary>
    public static void For<TBody>(int count, int size, long work, TBody body) where TBody : struct, IBody
    {
        int chunks = (count + size - 1) / size;
        if (chunks <= 1 || work < InlineWork || t_inside || Size == 1)
        {
            body.Run(0, count);
            return;
        }

        lock (s_gate)
        {
            var workers = s_workers ??= Start();
            s_invoke = &Invoke<TBody>;
            s_body = Unsafe.AsPointer(ref body);
            s_count = count;
            s_size = size;
            s_chunks = chunks;
            s_error = null;
            Volatile.Write(ref s_done, 0);
            int generation = Interlocked.Increment(ref s_generation);
            Interlocked.Exchange(ref s_state, (long)generation << 32);       // published last: a chunk taken sees the job above
            int wanted = Math.Min(workers.Length, chunks - 1), woken = 0;
            for (int pass = 0; pass < 2 && woken < wanted; pass++)          // the spinning workers first, then sleepers
            {
                for (int i = 0; i < workers.Length && woken < wanted; i++)
                {
                    if (workers[i].Wake(generation, sleepersToo: pass == 1)) woken++;
                }
            }

            t_inside = true;
            try
            {
                Take(generation);
            }
            finally
            {
                t_inside = false;
            }

            var spin = new SpinWait();
            while (Volatile.Read(ref s_done) < chunks) spin.SpinOnce(-1);
            // The job closed: its next chunk set past any count, so a worker that read this job's generation before the
            // next job is described can take nothing under that description (its exchange finds the counter changed).
            Interlocked.Exchange(ref s_state, (long)generation << 32 | int.MaxValue);
            s_body = null;
            s_error?.Throw();
        }
    }

    private static void Invoke<TBody>(void* body, int first, int last) where TBody : struct, IBody => Unsafe.AsRef<TBody>(body).Run(first, last);

    // Chunks of one job only: the take compares the generation it was handed with the one in the counter, so a worker
    // arriving after its job ended (or during the next) never runs a chunk under another job's description.
    private static void Take(int generation)
    {
        while (true)
        {
            long state = Volatile.Read(ref s_state);
            int c = (int)state;
            if ((int)(state >> 32) != generation || c >= s_chunks) return;
            if (Interlocked.CompareExchange(ref s_state, state + 1, state) != state) continue;
            try
            {
                int first = c * s_size;
                s_invoke(s_body, first, Math.Min(s_count, first + s_size));
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref s_error, ExceptionDispatchInfo.Capture(e), null);
            }

            Interlocked.Increment(ref s_done);
        }
    }

    private static Worker[] Start()
    {
        var workers = new Worker[Size - 1];
        for (int i = 0; i < workers.Length; i++) workers[i] = new Worker(i);
        return workers;
    }

    // A worker asks the system not to throttle its execution speed. Windows' power throttling (EcoQoS) may run a thread it
    // judges a background one on efficiency cores at low clocks: a team ran at one of two speeds a process (measured:
    // products, convolutions, maps and norms 1.3-2x apart, the same build); opted out, one speed and the faster. THREAD_POWER_THROTTLING_STATE {Version 1, ControlMask EXECUTION_SPEED, StateMask 0}: the
    // thread manages its own speed, unthrottled. Elsewhere nothing to ask. The caller's thread is the caller's: untouched.
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    private static extern bool SetThreadInformation(nint thread, int infoClass, ref ThrottlingState info, int size);

    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    private static extern nint GetCurrentThread();

    private struct ThrottlingState { public uint Version, ControlMask, StateMask; }

    private static void FullSpeed()
    {
        if (!OperatingSystem.IsWindows()) return;
        try
        {
            var state = new ThrottlingState { Version = 1, ControlMask = 1, StateMask = 0 };
            SetThreadInformation(GetCurrentThread(), 3, ref state, 12);           // ThreadPowerThrottling
        }
        catch (Exception e) when (e is EntryPointNotFoundException or DllNotFoundException) { }
    }

    private sealed class Worker
    {
        private readonly ManualResetEventSlim _wake = new(false, 0);
        // The generation this worker is to run and whether it sleeps, on 128 bytes of their own: the caller writes every
        // worker's in turn while each spins reading its own.
        [StructLayout(LayoutKind.Explicit, Size = 256)]
        private struct Flags
        {
            [FieldOffset(128)] public int Job;
            [FieldOffset(132)] public int Sleeping;
        }

        private Flags _flags;
        private ref int _job => ref _flags.Job;
        private ref int _sleeping => ref _flags.Sleeping;

        public Worker(int index)
        {
            var thread = new Thread(Loop, 256 * 1024) { IsBackground = true, Name = $"adad team {index + 1}" };
            thread.Start();
        }

        /// <summary>Hands the worker the job; a sleeping worker only when asked. True when it was handed.</summary>
        public bool Wake(int generation, bool sleepersToo)
        {
            if (Volatile.Read(ref _job) == generation) return false;         // already handed this job
            bool sleeping = Volatile.Read(ref _sleeping) == 1;
            if (sleeping && !sleepersToo) return false;
            Volatile.Write(ref _job, generation);
            if (Interlocked.CompareExchange(ref _sleeping, 0, 0) == 1) _wake.Set();
            return true;
        }

        private void Loop()
        {
            int done = 0;
            t_inside = true;
            var thread = Thread.CurrentThread;
            FullSpeed();
            while (true)
            {
                // Waiting at a lower priority: when something else must run, the scheduler takes a spinner's core, not
                // a working thread's.
                thread.Priority = ThreadPriority.BelowNormal;
                long deadline = Stopwatch.GetTimestamp() + s_spinBudget;
                int spins = 0;
                while (Volatile.Read(ref _job) == done)
                {
                    Thread.SpinWait(8);
                    if ((++spins & 63) != 0 || Stopwatch.GetTimestamp() < deadline) continue;
                    _wake.Reset();
                    Interlocked.Exchange(ref _sleeping, 1);
                    if (Volatile.Read(ref _job) == done) _wake.Wait();
                    Interlocked.Exchange(ref _sleeping, 0);
                    deadline = Stopwatch.GetTimestamp() + s_spinBudget;
                }

                done = Volatile.Read(ref _job);
                thread.Priority = ThreadPriority.Normal;
                Take(done);
            }
        }
    }
}
