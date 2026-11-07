// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's ctc (Ctc.cs, Ptx.Ctc.cs): the logits widened to the float lane's storage once, a block per sequence,
// the loss and the gradient narrowed once.
namespace Adad;

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
            ulong lx = ToLane<TN>(x, a.Values, rented);
            ulong Rent(long bytes) { ulong p = gpu.Rent(bytes); rented.Add(p); return p; }
            ulong lloss = own ? loss : Rent(a.B * (long)lane.Size);
            ulong lgrad = grad == 0 ? 0 : own ? grad : Rent(a.Values * lane.Size);
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
            var rowsFn = gpu.Cached($"ctc_rows:{lane.Name}:{backward}", (out string e) => Ptx.CtcRows(gpu, lane, backward, out e));
            Gpu.Launch(rowsFn, Grid((long)a.T * a.B, Ptx.CtcThreads / 32), 1, Ptx.CtcThreads, 1, 0, lx, lse, di, lgrad, Gpu.U(a.T), Gpu.U(a.B), Gpu.U(a.C));
            bool one = cap <= Ptx.CtcThreads && CtcOneState != false;
            var fn = gpu.Cached($"ctc:{lane.Name}:{backward}:{cap}{(one ? ":1" : "")}", (out string e) => Ptx.Ctc(gpu, lane, cap, backward, one, out e));
            Gpu.Launch(fn, (uint)a.B, 1, Ptx.CtcThreads, 1, 0, lx, lse, alpha, dc, df, di, ds, lloss, lgrad, Gpu.U(a.T), Gpu.U(a.B), Gpu.U(a.C), Gpu.U(smax));
            if (!own)
            {
                FromLane<TN>(lloss, loss, a.B);
                if (backward) FromLane<TN>(lgrad, grad, a.Values);
            }

            return f.IsBlock ? $"{lane.Name} (blocks through lane storage)" : lane.Name;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }
}
