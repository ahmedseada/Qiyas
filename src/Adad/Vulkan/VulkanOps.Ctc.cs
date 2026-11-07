// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The ctc loss on Vulkan (Ctc.cs, Kernels.Ctc.cs): a workgroup per sequence in the float lane (CUDA's), the logits read
// where they are stored; the loss and the gradient on lane storage, the type's own for float32 and float64, else stored
// as the type once after (a block format's by its encoder).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>The ctc loss per sequence (B values as the type) and, when grad is not 0, the gradient with respect to the
    /// logits (T × B × C as the type); returns the lane's name.</summary>
    public string Ctc<TN>(CtcArgs a, ulong x, ulong loss, ulong grad) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        int smax = a.MaxStates, cap = (smax + 63) / 64 * 64;
        if (VkKernels.CtcShared(lane, cap) > vk.MaxSharedBytes) throw new NotSupportedException($"vulkan: ctc's {smax} extended states exceed the workgroup memory ({vk.MaxSharedBytes} bytes)");
        if (a.Values > uint.MaxValue || (long)a.B * a.T * smax > uint.MaxValue) throw new NotSupportedException("vulkan: ctc of more than 2^32 values");
        var rented = new List<ulong>();
        try
        {
            ulong Rent(long bytes) { ulong p = vk.Rent(bytes); rented.Add(p); return p; }
            bool own = f.Is("float32") || f.Is("float64");
            bool backward = grad != 0;
            ulong lloss = own ? loss : Rent(a.B * (long)lane.Size);
            ulong lgrad = !backward ? 0 : own ? grad : Rent(a.Values * lane.Size);
            ulong lse = Rent((long)a.T * a.B * lane.Size), alpha = Rent((long)a.B * a.T * smax * lane.Size);
            var (cls, flags) = a.States();
            // One upload: the classes, the flags, the frames and the states per sequence.
            var host = new int[cls.Length + flags.Length + 2 * a.B];
            cls.CopyTo(host, 0);
            flags.CopyTo(host, cls.Length);
            a.InputLengths.CopyTo(host, cls.Length + flags.Length);
            for (int i = 0; i < a.B; i++) host[cls.Length + flags.Length + a.B + i] = 2 * a.LabelLengths[i] + 1;
            ulong dc = Rent(host.Length * 4L), df = dc + (ulong)(cls.Length * 4L), di = df + (ulong)(flags.Length * 4L), ds = di + (ulong)(a.B * 4L);
            vk.Upload<int>(host, dc);
            var rows = vk.Cached($"ctc_rows:{f.Name}:{backward}", () => VkKernels.CtcRows(vk, f, lane, backward));
            vk.Launch(rows, (uint)((long)a.T * a.B), 1, x, lse, di, lgrad, Vulkan.U(a.T), Vulkan.U(a.B), Vulkan.U(a.C));
            var fn = vk.Cached($"ctc:{f.Name}:{backward}:{cap}", () => VkKernels.Ctc(vk, f, lane, cap, backward));
            vk.Launch(fn, (uint)a.B, 1, x, lse, alpha, dc, df, di, ds, lloss, lgrad, Vulkan.U(a.T), Vulkan.U(a.B), Vulkan.U(a.C), Vulkan.U(smax));
            if (!own)
            {
                Back(f, lane, lloss, loss, a.B);
                if (backward) Back(f, lane, lgrad, grad, a.Values);
            }

            return lane.Name;
        }
        finally
        {
            foreach (var p in rented) vk.Return(p);
        }
    }
}
