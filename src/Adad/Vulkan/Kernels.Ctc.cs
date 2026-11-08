// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The ctc loss on Vulkan (Ctc.cs; CUDA's Ptx.Ctc.cs, its general kernel), on lane storage (f32; f64 for float64):
//   1. CtcRows: each (frame, sequence) row's log-sum-exp over the classes, a workgroup a row (the classes strided over it,
//      max and sum by the tree); for the gradient the row's softmax y_t(c), and 0 for the rows past the sequence's frames;
//   2. Ctc, a workgroup a sequence: α forward over the extended states, the previous frame's α in workgroup memory (two
//      buffers, two −∞ pads before state 0), a barrier a frame, α of every frame kept in global scratch; invocation 0
//      takes log P and stores the loss; then β backward in the same buffers (pads after the last state), each state's
//      exp(α + β − log y − log P) into workgroup memory, and the first state of each class (the host marks it) sums its
//      class's states in state order and writes y − sum: no atomics, the same order on every run. A sequence without an
//      alignment: loss +∞, gradient 0.
// The logits are read where they are stored; the loss and the gradient are lane values (stored as the type after).
using Adad.Abstraction;

namespace Adad.Vulkan;

internal static partial class VkKernels
{
    public const int CtcThreads = 256, CtcRowThreads = 64;

    /// <summary>Workgroup bytes of the ctc kernel for states up to cap: two state buffers with pads, the per-state terms, log P.</summary>
    public static int CtcShared(Lane lane, int cap) => (3 * cap + 5) * lane.Size;

    /// <summary>Arguments: x, lse, ilen, grad (0 for the loss alone), T, B, C, the row strides of a frame and of a sequence
    /// (frame-major B and 1, batch-first 1 and T); a workgroup per row (T·B of them).</summary>
    public static SpvKernel CtcRows(Vulkan vk, Fmt f, Lane lane, bool backward)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"ctc_rows_{f.Name}_{lane.Name}_{(backward ? "bwd" : "fwd")}", CtcRowThreads, vk.Controls);
        var x = b.Arg(0);
        var lse = b.Arg(1);
        var ilen = b.Arg(2);
        var grad = b.Arg(3);
        var T = b.Arg(4, K.U32);
        var B = b.Arg(5, K.U32);
        var C = b.Arg(6, K.U32);
        var ts = b.Arg(7, K.U32);
        var bs = b.Arg(8, K.U32);
        var tid = b.LocalIdX;
        var r = b.GroupX;
        var t = r / ts % T;
        var seq = r / bs % B;
        var tb = b.Load(K.U32, b.At(ilen, seq, 4));
        var row = r * C;
        var past = t >= tb;
        if (backward)
        {
            // Past the sequence's frames: 0.
            b.If(past, () => b.For(tid, C, b.U(CtcRowThreads), c => b.Store(b.At(grad, row + c, lane.Size), b.C(k, 0))));
        }

        var sh = b.Shared(k, CtcRowThreads);
        b.If(!past, () =>
        {
            var m = b.Local(b.C(k, double.NegativeInfinity));
            b.For(tid, C, b.U(CtcRowThreads), c => m.V = b.Max(m.V, Codec.UnpackAt(b, f, x, row + c, lane)));
            var max = GroupReduce(b, sh, tid, m.V, b.Max, CtcRowThreads);
            var s = b.Local(b.C(k, 0));
            b.For(tid, C, b.U(CtcRowThreads), c => s.V = s.V + Fn.Exp(b, Codec.UnpackAt(b, f, x, row + c, lane) - max));
            var sum = GroupReduce(b, sh, tid, s.V, static (p, q) => p + q, CtcRowThreads);
            var res = max + Fn.Log(b, sum);
            b.If(tid == 0, () => b.Store(b.At(lse, r, lane.Size), res));
            if (backward) b.For(tid, C, b.U(CtcRowThreads), c => b.Store(b.At(grad, row + c, lane.Size), Fn.Exp(b, Codec.UnpackAt(b, f, x, row + c, lane) - res)));
        });
        return b.Build();
    }

    /// <summary>Arguments: x, lse, alpha, cls, flags, ilen, slen, loss, grad (0 for the loss alone), T, B, C, smax, the row
    /// strides of a frame and of a sequence, zero-infinity (1: a sequence no alignment fits counts 0); a workgroup per
    /// sequence.</summary>
    public static SpvKernel Ctc(Vulkan vk, Fmt f, Lane lane, int cap, bool backward)
    {
        var k = Kinds.Of(lane);
        var b = new Spv($"ctc_{f.Name}_{lane.Name}_{(backward ? "bwd" : "fwd")}_{cap}", CtcThreads, vk.Controls);
        var x = b.Arg(0);
        var lseG = b.Arg(1);
        var alphaG = b.Arg(2);
        var clsG = b.Arg(3);
        var flagsG = b.Arg(4);
        var ilen = b.Arg(5);
        var slen = b.Arg(6);
        var loss = b.Arg(7);
        var grad = b.Arg(8);
        var T = b.Arg(9, K.U32);
        var B = b.Arg(10, K.U32);
        var C = b.Arg(11, K.U32);
        var smax = b.Arg(12, K.U32);
        var ts = b.Arg(13, K.U32);
        var bs = b.Arg(14, K.U32);
        var zinf = b.Arg(15, K.U32);
        var tid = b.LocalIdX;
        var seq = b.GroupX;
        var negInf = b.C(k, double.NegativeInfinity);
        var zero = b.C(k, 0);
        int stride = cap + 2;
        var buf = b.Shared(k, 2 * stride);
        var gsh = b.Shared(k, cap);
        var logPs = b.Shared(k, 1);

        // The sequence's frames, states and rows.
        var tb = b.Load(K.U32, b.At(ilen, seq, 4));
        var S = b.Load(K.U32, b.At(slen, seq, 4));
        var flagRow = (smax + 2) * seq;
        var alphaRow = seq * T;
        V U(V basePtr, V index) => b.Load(K.U32, b.At(basePtr, index, 4));
        var seqRow = seq * bs;                                                  // the sequence's row at frame 0; a frame on, ts rows
        V RowBase(V t) => (t * ts + seqRow) * C;
        V LseAt(V t) => b.Load(k, b.At(lseG, t * ts + seqRow, lane.Size));
        V Base(V t) => (t & 1UL) * stride;
        // log y_t(l'_s) = x[t][seq][cls(s)] − lse_t.
        V Emit(V row, V lseT, V s) => Codec.UnpackAt(b, f, x, row + U(clsG, flagRow + s), lane) - lseT;

        V Lse3(V p, V q, V r)
        {
            var m = b.Max(b.Max(p, q), r);
            var none = m == double.NegativeInfinity;
            var ms = b.Select(none, zero, m);
            var sum = Fn.Exp(b, p - ms) + Fn.Exp(b, q - ms);
            sum = sum + Fn.Exp(b, r - ms);
            return b.Select(none, negInf, ms + Fn.Log(b, sum));
        }

        // Both state buffers −∞ (the pads included).
        void ClearBuffers()
        {
            b.For(tid, b.U(2 * stride), b.U(CtcThreads), s => buf[s] = negInf);
            b.Barrier();
        }

        // α forward.
        ClearBuffers();
        b.For(b.U(0), tb, t =>
        {
            var row = RowBase(t);
            var lseT = LseAt(t);
            var cur = Base(t);
            var prev = Base(t - 1);
            var first = t == 0;
            b.For(tid, S, b.U(CtcThreads), s =>
            {
                var e = Emit(row, lseT, s);
                var p = buf[prev + s + 2];
                var q = buf[prev + s + 1];
                var r = b.Select((U(flagsG, flagRow + s) & 1UL) == 0, negInf, buf[prev + s]);
                var v = Lse3(p, q, r) + e;
                // The first frame: state 0 and 1 take their emissions, the rest −∞.
                v = b.Select(first, b.Select(s < 2, e, negInf), v);
                buf[cur + s + 2] = v;
                b.Store(b.At(alphaG, (alphaRow + t) * smax + s, lane.Size), v);
            });
            b.Barrier();
        });

        // log P from the last frame's last two states (index S + 1 and S: the pad when S = 1); the loss.
        b.If(tid == 0, () =>
        {
            var last = Base(tb - 1);
            var logP = Lse3(buf[last + S + 1], buf[last + S], negInf);
            logPs[0] = logP;
            b.Store(b.At(loss, seq, lane.Size), b.Select((zinf != 0) & (logP == negInf), zero, -logP));
        });
        b.Barrier();
        if (!backward) return b.Build();

        var logPv = logPs[0];
        var noAlignment = logPv == double.NegativeInfinity;
        // No alignment: the sequence's frames' rows 0 too (the rows past its frames are the rows kernel's).
        b.If(noAlignment, () => b.For(b.U(0), tb, t =>
        {
            var row = RowBase(t);
            b.For(tid, C, b.U(CtcThreads), c => b.Store(b.At(grad, row + c, lane.Size), zero));
        }));
        b.If(!noAlignment, () =>
        {
            ClearBuffers();
            var tLast = tb - 1;
            b.For(b.U(0), tb, i =>
            {
                var t = tLast - i;
                var row = RowBase(t);
                var lseT = LseAt(t);
                var cur = Base(t);
                var prev = Base(t + 1);
                var isLast = t == tLast;
                // β_t over the states; each state's term into workgroup memory.
                b.For(tid, S, b.U(CtcThreads), s =>
                {
                    var e = Emit(row, lseT, s);
                    var p = buf[prev + s];
                    var q = buf[prev + s + 1];
                    var r = b.Select((U(flagsG, flagRow + s + 2) & 1UL) == 0, negInf, buf[prev + s + 2]);
                    var v = Lse3(p, q, r) + e;
                    v = b.Select(isLast, b.Select(s + 2 >= S, e, negInf), v);
                    buf[cur + s] = v;
                    var av = b.Load(k, b.At(alphaG, (alphaRow + t) * smax + s, lane.Size));
                    gsh[s] = Fn.Exp(b, av + v - e - logPv);
                });
                b.Barrier();
                // Each class's first state: y − Σ of its class's terms, in state order.
                b.For(tid, S, b.U(CtcThreads), s =>
                {
                    b.If((U(flagsG, flagRow + s) & 2UL) != 0, () =>
                    {
                        var myc = U(clsG, flagRow + s);
                        var sum = b.Local(zero);
                        b.For(s, S, b.U(2), q => sum.V = b.Select(U(clsG, flagRow + q) == myc, sum.V + gsh[q], sum.V));
                        var ci = row + myc;
                        var y = Fn.Exp(b, Codec.UnpackAt(b, f, x, ci, lane) - lseT) - sum.V;
                        b.Store(b.At(grad, ci, lane.Size), y);
                    });
                });
                b.Barrier();
            });
        });
        return b.Build();
    }
}
