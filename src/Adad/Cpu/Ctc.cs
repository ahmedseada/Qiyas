// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// ctc: the connectionist temporal classification loss and its gradient, for a batch of B sequences of T frames over C
// classes (one of them the blank). The input is logits (T × B × C, frame-major: logits in another layout are moved by the
// strided copy, ILayouts); log_softmax is taken inside. Sequence b sees its first InputLengths[b] frames and its LabelLengths[b] labels (Labels flat, sequences one
// after another, none the blank). The extended labels l' put a blank before, between and after them (S = 2L + 1
// states); in log space α_t(s) and β_t(s), both including frame t's emission y_t(l'_s):
//   α_0(0) = log y_0(blank), α_0(1) = log y_0(l_1); α_t(s) = lse(α_{t−1}(s), α_{t−1}(s−1), α_{t−1}(s−2) when l'_s is not
//   the blank and differs from l'_{s−2}) + log y_t(l'_s); β the same backwards from the last frame's last two states.
//   loss_b = −log P, log P = lse(α_{T_b−1}(S−1), α_{T_b−1}(S−2)).
//   d loss / d logit_t(c) = y_t(c) − Σ_{s: l'_s = c} exp(α_t(s) + β_t(s) − log y_t(l'_s) − log P); 0 past T_b.
// A sequence no alignment fits (too few frames for its labels and their repeats) has loss +∞ and a gradient of 0.
// The lane is the float lane (f32; f64 for float64). Three passes on the team, so that every processor has work however
// few the sequences: every frame's row of every sequence (the bulk of the work: the row's log-sum-exp and its states'
// emissions log y_t(l'_s), into one table of a row of states a frame), then each sequence's recursions over its states
// (α; for the gradient β, and each state's share γ_t(s) = exp(α + β − log y − log P) written over its emission once β
// has passed the frame), then for the gradient every row again (its softmax less its states' shares). A three-term
// log-sum-exp takes two exponentials (the largest term's is 1). Rows are read where they are stored (float32 and float64
// in place, a narrower type's row converted into lanes); the loss and the gradient stored as the type (a block format's:
// the lane's values, encoded once — rows of a sequence do not start on blocks).

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>The ctc loss per sequence (B values as the type) and, when grad is given, the gradient with respect to the
    /// logits (T × B × C as the type); returns the lane's name.</summary>
    public static string Ctc<TN>(CtcArgs a, byte[] x, byte[] loss, byte[]? grad) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        var vis = new CtcVisitor(a, x, loss, grad);
        TN.Float(ref vis);
        return vis.Lane;
    }

    private struct CtcVisitor(CtcArgs a, byte[] x, byte[] loss, byte[]? grad) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TS, TLane, TL>() where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            int rows = a.T * a.B, w = Vector<TLane>.Count, size = Unsafe.SizeOf<TLane>();
            int width = (a.MaxStates + w - 1) / w * w;                       // a frame's row of states in the table
            // A block format: the loss and the gradient as the lane's values, encoded once.
            bool lanesOut = TS.Granularity != 1;
            byte[] lloss = lanesOut ? new byte[a.B * size] : loss;
            byte[]? lgrad = grad is null ? null : lanesOut ? new byte[a.Values * size] : grad;
            // Each sequence's states' classes (a row of width a sequence).
            var cls = new int[a.B * width];
            for (int b = 0, off = 0; b < a.B; off += a.LabelLengths[b], b++)
                for (int s = 0; s < 2 * a.LabelLengths[b] + 1; s++) cls[b * width + s] = (s & 1) == 0 ? a.Blank : a.Labels[off + (s >> 1)];
            var pool = ArrayPool<TLane>.Shared;
            var lse = pool.Rent(rows);
            var logP = pool.Rent(a.B);
            var states = pool.Rent(rows * width);                           // emissions, then shares: (b·T + t)·width + s
            try
            {
                int chunk = Math.Max(1, rows / (Threads * 8));
                Team.For(rows, chunk, (long)rows * a.C * 2, new CtcRowsBody<TS, TLane, TL>(a, x, cls, width, lse, states));
                Team.For(a.B, 1, (long)a.T * width * (grad is null ? 8 : 24), new CtcSequenceBody<TLane, TL>(a, cls, width, states, logP, grad is not null));
                CtcLoss<TS, TLane, TL>(a, logP, lloss, lanesOut);
                if (lgrad is not null) Team.For(rows, chunk, (long)rows * a.C * 3, new CtcGradientBody<TS, TLane, TL>(a, x, cls, width, lse, logP, states, lgrad, lanesOut));
                if (lanesOut)
                {
                    StoreLanes<TS, TLane>(lloss, loss, 0, a.B);
                    if (grad is not null) StoreLanes<TS, TLane>(lgrad!, grad, 0, a.Values);
                }
            }
            finally
            {
                pool.Return(lse);
                pool.Return(logP);
                pool.Return(states);
            }
        }
    }

    // loss_b = −log P_b as the type (as lanes for a block format).
    private static void CtcLoss<TS, TLane, TL>(CtcArgs a, TLane[] logP, byte[] loss, bool lanesOut)
        where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        var lanes = new TLane[a.B];
        for (int b = 0; b < a.B; b++) lanes[b] = TL.Sub(TL.Zero, logP[b]);
        if (TS.Direct || lanesOut) lanes.AsSpan().CopyTo(MemoryMarshal.Cast<byte, TLane>(loss.AsSpan()));
        else TS.Store(lanes, loss, 0);
    }

    // The row of frame t of sequence b (row q = t·B + b): in place for float32 and float64, else converted into buf.
    private static ReadOnlySpan<TLane> CtcRow<TS, TLane>(byte[] x, int q, int C, TLane[]? buf) where TS : struct, IAdad<TS, TLane> where TLane : unmanaged
    {
        if (TS.Direct) return MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice(q * C, C);
        TS.Load(x, q * C, buf.AsSpan(0, C));
        return buf.AsSpan(0, C);
    }

    // log Σ exp of three vectors of log values: m the largest, the sum 1 + exp of the other two less m; −∞ where all are.
    private static Vector<TLane> CtcLse<TLane, TL>(Vector<TLane> p, Vector<TLane> q, Vector<TLane> r) where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        var negInf = new Vector<TLane>(TL.NegativeInfinity);
        var hi = TL.Max(p, q);
        var lo = TL.Min(p, q);
        var m = TL.Max(hi, r);
        var other = TL.Min(hi, r);
        var none = TL.Equal(m, negInf);
        var ms = TL.Select(none, Vector<TLane>.Zero, m);
        var sum = TL.Add(TL.Add(new Vector<TLane>(TL.From(1)), TL.Exp(TL.Sub(other, ms))), TL.Exp(TL.Sub(lo, ms)));
        return TL.Select(none, negInf, TL.Add(ms, TL.Log(sum)));
    }

    // Each row's log-sum-exp over the classes (lse[t·B + b]) and its states' emissions x_t(l'_s) − lse_t, −∞ past the
    // sequence's states; the rows past a sequence's frames are not read later.
    private readonly struct CtcRowsBody<TS, TLane, TL>(CtcArgs a, byte[] x, int[] cls, int width, TLane[] lse, TLane[] states) : Team.IBody
        where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int from, int to)
        {
            int C = a.C, B = a.B, w = Vector<TLane>.Count;
            var buf = TS.Direct ? null : ArrayPool<TLane>.Shared.Rent(C);
            try
            {
                for (int q = from; q < to; q++)
                {
                    int t = q / B, b = q % B;
                    if (t >= a.InputLengths[b]) continue;
                    var row = CtcRow<TS, TLane>(x, q, C, buf);
                    var mv = new Vector<TLane>(TL.NegativeInfinity);
                    int c = 0;
                    for (; c + w <= C; c += w) mv = TL.Max(mv, new Vector<TLane>(row.Slice(c, w)));
                    var m = TL.NegativeInfinity;
                    for (int i = 0; i < w; i++) m = TL.Max(m, mv[i]);
                    for (; c < C; c++) m = TL.Max(m, row[c]);
                    var vm = new Vector<TLane>(m);
                    var sv = Vector<TLane>.Zero;
                    for (c = 0; c + w <= C; c += w) sv = TL.Add(sv, TL.Exp(TL.Sub(new Vector<TLane>(row.Slice(c, w)), vm)));
                    var sum = TL.Zero;
                    for (int i = 0; i < w; i++) sum = TL.Add(sum, sv[i]);
                    for (; c < C; c++) sum = TL.Add(sum, TL.Exp(TL.Sub(row[c], m)));
                    var l = TL.Add(m, TL.Log(new Vector<TLane>(sum))[0]);
                    lse[q] = l;
                    int S = 2 * a.LabelLengths[b] + 1;
                    var e = states.AsSpan((b * a.T + t) * width, width);
                    var k = cls.AsSpan(b * width, S);
                    for (int s = 0; s < S; s++) e[s] = TL.Sub(row[k[s]], l);
                    e[S..].Fill(TL.NegativeInfinity);
                }
            }
            finally
            {
                if (buf is not null) ArrayPool<TLane>.Shared.Return(buf);
            }
        }
    }

    // A sequence's recursions over its states from the emissions in the table (a vector of states at a time): α forward,
    // log P; for the gradient β backward, each frame's states' shares γ_t(s) written over their emissions as β leaves it.
    private readonly struct CtcSequenceBody<TLane, TL>(CtcArgs a, int[] cls, int width, TLane[] states, TLane[] logP, bool grad) : Team.IBody
        where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int from, int to)
        {
            for (int b = from; b < to; b++) Sequence(b);
        }

        private void Sequence(int b)
        {
            int tb = a.InputLengths[b], S = 2 * a.LabelLengths[b] + 1, w = Vector<TLane>.Count;
            int Sp = (S + w - 1) / w * w, stride = Sp + 2 * w;              // states rounded to whole vectors; pads around them
            var negInf = TL.NegativeInfinity;
            var pool = ArrayPool<TLane>.Shared;
            var skipA = pool.Rent(Sp);                                      // 0 where α may skip from s − 2, −∞ where not
            var skipB = pool.Rent(Sp);                                      // 0 where β may skip from s + 2
            var alpha = pool.Rent(tb * stride);                             // α_t(s) at t·stride + w + s; −∞ in the pads
            var beta = pool.Rent(2 * stride);
            try
            {
                var k = cls.AsSpan(b * width, S);
                int blank = a.Blank;
                for (int s = 0; s < Sp; s++)
                {
                    skipA[s] = s >= 2 && s < S && k[s] != blank && k[s] != k[s - 2] ? TL.Zero : negInf;
                    skipB[s] = s + 2 < S && k[s + 2] != blank && k[s + 2] != k[s] ? TL.Zero : negInf;
                }

                var emit = states.AsSpan(b * a.T * width, tb * width);      // frame t's emissions at t·width
                // α forward: a vector of states at a time from the previous frame's s, s − 1 and s − 2.
                alpha.AsSpan(0, tb * stride).Fill(negInf);
                alpha[w] = emit[0];
                if (S > 1) alpha[w + 1] = emit[1];
                for (int t = 1; t < tb; t++)
                {
                    var prev = alpha.AsSpan((t - 1) * stride, stride);
                    var cur = alpha.AsSpan(t * stride, stride);
                    for (int s = 0; s < Sp; s += w)
                    {
                        var p = new Vector<TLane>(prev.Slice(w + s, w));
                        var q = new Vector<TLane>(prev.Slice(w + s - 1, w));
                        var r = TL.Add(new Vector<TLane>(prev.Slice(w + s - 2, w)), new Vector<TLane>(skipA.AsSpan(s, w)));
                        TL.Add(CtcLse<TLane, TL>(p, q, r), new Vector<TLane>(emit.Slice(t * width + s, w))).CopyTo(cur.Slice(w + s, w));
                    }

                    cur.Slice(w + S, stride - w - S).Fill(negInf);
                }

                var last = alpha.AsSpan((tb - 1) * stride + w, S);
                var lp = CtcLse<TLane, TL>(new Vector<TLane>(last[S - 1]), new Vector<TLane>(S >= 2 ? last[S - 2] : negInf), new Vector<TLane>(negInf))[0];
                logP[b] = lp;
                if (!grad || EqualityComparer<TLane>.Default.Equals(lp, negInf)) return;

                // β backward; each frame's shares over its emissions once β_t is made (frame t's emissions not read again).
                var vLogP = new Vector<TLane>(lp);
                beta.AsSpan(0, 2 * stride).Fill(negInf);
                for (int t = tb - 1; t >= 0; t--)
                {
                    var cur = beta.AsSpan((t & 1) * stride, stride);
                    var e = emit.Slice(t * width, width);
                    if (t == tb - 1)
                    {
                        cur[w + S - 1] = e[S - 1];
                        if (S >= 2) cur[w + S - 2] = e[S - 2];
                    }
                    else
                    {
                        var prev = beta.AsSpan(((t + 1) & 1) * stride, stride);
                        for (int s = 0; s < Sp; s += w)
                        {
                            var p = new Vector<TLane>(prev.Slice(w + s, w));
                            var q = new Vector<TLane>(prev.Slice(w + s + 1, w));
                            var r = TL.Add(new Vector<TLane>(prev.Slice(w + s + 2, w)), new Vector<TLane>(skipB.AsSpan(s, w)));
                            TL.Add(CtcLse<TLane, TL>(p, q, r), new Vector<TLane>(e.Slice(s, w))).CopyTo(cur.Slice(w + s, w));
                        }

                        cur.Slice(w + S, stride - w - S).Fill(negInf);       // past the states: −∞ again for the next frame
                    }

                    var at = alpha.AsSpan(t * stride + w, Sp);
                    int s2 = 0;
                    for (; s2 + w <= S; s2 += w)
                    {
                        var g = TL.Sub(TL.Sub(TL.Add(new Vector<TLane>(at.Slice(s2, w)), new Vector<TLane>(cur.Slice(w + s2, w))), new Vector<TLane>(e.Slice(s2, w))), vLogP);
                        TL.Exp(g).CopyTo(e.Slice(s2, w));
                    }

                    if (s2 < S)
                    {
                        var g = TL.Exp(TL.Sub(TL.Sub(TL.Add(new Vector<TLane>(at.Slice(s2, w)), new Vector<TLane>(cur.Slice(w + s2, w))), new Vector<TLane>(e.Slice(s2, w))), vLogP));
                        for (int i = 0; s2 + i < S; i++) e[s2 + i] = g[i];
                    }
                }
            }
            finally
            {
                pool.Return(skipA); pool.Return(skipB); pool.Return(alpha); pool.Return(beta);
            }
        }
    }

    // The gradient's rows: exp(x − lse) less the row's states' shares at their classes; 0 past a sequence's frames and
    // where no alignment fits.
    private readonly struct CtcGradientBody<TS, TLane, TL>(CtcArgs a, byte[] x, int[] cls, int width, TLane[] lse, TLane[] logP, TLane[] states, byte[] grad, bool lanesOut) : Team.IBody
        where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int from, int to)
        {
            int C = a.C, B = a.B, w = Vector<TLane>.Count;
            bool inPlace = TS.Direct || lanesOut;
            var pool = ArrayPool<TLane>.Shared;
            var buf = TS.Direct ? null : pool.Rent(C);
            var outBuf = inPlace ? null : pool.Rent(C);
            try
            {
                for (int q = from; q < to; q++)
                {
                    int t = q / B, b = q % B;
                    Span<TLane> gs = inPlace ? MemoryMarshal.Cast<byte, TLane>(grad.AsSpan()).Slice(q * C, C) : outBuf.AsSpan(0, C);
                    if (t >= a.InputLengths[b] || EqualityComparer<TLane>.Default.Equals(logP[b], TL.NegativeInfinity))
                    {
                        gs.Clear();
                    }
                    else
                    {
                        var row = CtcRow<TS, TLane>(x, q, C, buf);
                        var vl = new Vector<TLane>(lse[q]);
                        int c = 0;
                        for (; c + w <= C; c += w) TL.Exp(TL.Sub(new Vector<TLane>(row.Slice(c, w)), vl)).CopyTo(gs.Slice(c, w));
                        for (; c < C; c++) gs[c] = TL.Exp(TL.Sub(row[c], lse[q]));
                        int S = 2 * a.LabelLengths[b] + 1;
                        var share = states.AsSpan((b * a.T + t) * width, S);
                        var k = cls.AsSpan(b * width, S);
                        for (int s = 0; s < S; s++) gs[k[s]] = TL.Sub(gs[k[s]], share[s]);
                    }

                    if (!inPlace) TS.Store(gs, grad, q * C);
                }
            }
            finally
            {
                if (buf is not null) pool.Return(buf);
                if (outBuf is not null) pool.Return(outBuf);
            }
        }
    }
}
