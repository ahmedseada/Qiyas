// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's ctc (Ctc.cs, Ptx.Ctc.cs): a block per sequence in the float lane, the logits read where they are
// stored; the loss and the gradient stored as the type (a block format's, a sub-byte one's: the lane's values, encoded
// once). The rows' log-sum-exp and the α recursion are f32 intermediates.
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    /// <summary>The gate's override of the one-state ctc kernel (Ptx.Ctc oneState): false takes the general kernel at any
    /// length; null takes the one-state kernel where every sequence's states fit a thread each.</summary>
    internal bool? CtcOneState;

    /// <summary>The ctc loss per sequence (B values as the type) and, when grad is not 0, the gradient with respect to the
    /// logits (T × B × C as the type); returns the lane's name.</summary>
    public string Ctc<TN>(CtcArgs a, ulong x, ulong loss, ulong grad) where TN : struct, IAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        int smax = a.MaxStates, cap = (smax + 63) / 64 * 64;
        if (Ptx.CtcShared(lane, cap) > 48 * 1024) throw new NotSupportedException($"ctc: {smax} extended states exceed the kernel's shared memory");
        if (a.Values > uint.MaxValue || (long)a.B * a.T * smax > uint.MaxValue) throw new NotSupportedException("ctc: more than 2^32 values");
        var rented = new List<ulong>();
        try
        {
            bool own = f.Is("float32") || f.Is("float64");
            // The logits read where they are stored (fx); the outputs as the type where the kernels store it (fo).
            Fmt? fx = own ? null : f, fo = own || !StoredC(f) ? null : f;
            bool viaLane = !own && fo is null;
            ulong Rent(long bytes) { ulong p = gpu.Rent(bytes); rented.Add(p); return p; }
            ulong lloss = viaLane ? Rent(a.B * (long)lane.Size) : loss;
            ulong lgrad = grad == 0 ? 0 : viaLane ? Rent(a.Values * lane.Size) : grad;
            ulong lse = Rent((long)a.T * a.B * lane.Size), alpha = Rent((long)a.B * a.T * smax * lane.Size);
            var (cls, flags) = a.States();
            // One upload: the classes, the flags, the frames and the states per sequence.
            var host = new int[cls.Length + flags.Length + 2 * a.B];
            cls.CopyTo(host, 0);
            flags.CopyTo(host, cls.Length);
            a.InputLengths.CopyTo(host, cls.Length + flags.Length);
            for (int i = 0; i < a.B; i++) host[cls.Length + flags.Length + a.B + i] = 2 * a.LabelLengths[i] + 1;
            ulong dc = Rent(host.Length * 4L), df = dc + (ulong)(cls.Length * 4L), di = df + (ulong)(flags.Length * 4L), ds = di + (ulong)(a.B * 4L);
            Gpu.Upload<int>(host, dc);
            bool backward = grad != 0;
            var rowsFn = gpu.Cached($"ctc_rows:{lane.Name}:{backward}:{fx?.Name}:{fo?.Name}", (out string e) => Ptx.CtcRows(gpu, lane, backward, out e, fx, fo));
            var (ts, bs) = a.RowStrides;
            Gpu.Launch(rowsFn, Grid((long)a.T * a.B, Ptx.CtcThreads / 32), 1, Ptx.CtcThreads, 1, 0, x, lse, di, lgrad, Gpu.U(a.T), Gpu.U(a.B), Gpu.U(a.C), Gpu.U(ts), Gpu.U(bs));
            bool one = cap <= Ptx.CtcThreads && CtcOneState != false;
            var fn = gpu.Cached($"ctc:{lane.Name}:{backward}:{cap}{(one ? ":1" : "")}:{fx?.Name}:{fo?.Name}", (out string e) => Ptx.Ctc(gpu, lane, cap, backward, one, out e, fx, fo));
            Gpu.Launch(fn, (uint)a.B, 1, Ptx.CtcThreads, 1, 0, x, lse, alpha, dc, df, di, ds, lloss, lgrad, Gpu.U(a.T), Gpu.U(a.B), Gpu.U(a.C), Gpu.U(smax),
                Gpu.U(ts), Gpu.U(bs), Gpu.U(a.ZeroInfinity ? 1 : 0));
            if (viaLane)
            {
                FromLane<TN>(lloss, loss, a.B);
                if (backward) FromLane<TN>(lgrad, grad, a.Values);
            }

            return f.IsBlock ? $"{lane.Name} (blocks in place)" : lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }
}
