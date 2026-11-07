// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// ctc: the connectionist temporal classification loss and its gradient, for a batch of B sequences of T frames over C
// classes (one of them the blank). The input is logits (T × B × C, frame-major as the usual layout); log_softmax is taken
// inside. Sequence b sees its first InputLengths[b] frames and its LabelLengths[b] labels (Labels flat, sequences one
// after another, none the blank). The extended labels l' put a blank before, between and after them (S = 2L + 1
// states); in log space α_t(s) and β_t(s), both including frame t's emission y_t(l'_s):
//   α_0(0) = log y_0(blank), α_0(1) = log y_0(l_1); α_t(s) = lse(α_{t−1}(s), α_{t−1}(s−1), α_{t−1}(s−2) when l'_s is not
//   the blank and differs from l'_{s−2}) + log y_t(l'_s); β the same backwards from the last frame's last two states.
//   loss_b = −log P, log P = lse(α_{T_b−1}(S−1), α_{T_b−1}(S−2)).
//   d loss / d logit_t(c) = y_t(c) − Σ_{s: l'_s = c} exp(α_t(s) + β_t(s) − log y_t(l'_s) − log P); 0 past T_b.
// A sequence no alignment fits (too few frames for its labels and their repeats) has loss +∞ and a gradient of 0.
// The lane is the float lane (f32; f64 for float64), a narrow type widened once and the results narrowed once.
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Adad;

/// <summary>A ctc problem: T frames, B sequences, C classes; the blank class; per sequence its frames and its labels.</summary>
public sealed record CtcArgs(int T, int B, int C, int Blank, int[] InputLengths, int[] LabelLengths, int[] Labels)
{
    public long Values => (long)T * B * C;

    /// <summary>The first label of sequence b in Labels.</summary>
    public int Offset(int b)
    {
        int at = 0;
        for (int i = 0; i < b; i++) at += LabelLengths[i];
        return at;
    }

    /// <summary>The class of extended state s of sequence b (the blank at even s).</summary>
    public int Class(int b, int s) => (s & 1) == 0 ? Blank : Labels[Offset(b) + (s >> 1)];

    /// <summary>The longest extended label sequence: 2·max L + 1.</summary>
    public int MaxStates => 2 * LabelLengths.Max() + 1;

    /// <summary>Refuses what the definition does not cover: a sequence of no frames or more than T, a label that is the
    /// blank or not a class, label counts that do not add up to Labels.</summary>
    public void Validate()
    {
        if (T < 1 || B < 1 || C < 2 || Blank < 0 || Blank >= C) throw new ArgumentException($"ctc: T {T}, B {B}, C {C}, blank {Blank}");
        if (InputLengths.Length != B || LabelLengths.Length != B) throw new ArgumentException("ctc: a length per sequence");
        if (InputLengths.Any(t => t < 1 || t > T)) throw new ArgumentException("ctc: every sequence has 1 to T frames");
        if (LabelLengths.Any(l => l < 0) || LabelLengths.Sum() != Labels.Length) throw new ArgumentException("ctc: the label counts add up to the labels given");
        if (Labels.Any(l => l < 0 || l >= C || l == Blank)) throw new ArgumentException("ctc: labels are classes other than the blank");
    }

    /// <summary>Per sequence (rows of MaxStates + 2, zeros past its states): the class of each extended state, and its
    /// flags: bit 0 where α may come from s − 2 (a label differing from the label before it), bit 1 on the first state
    /// of each class, bits 2 up the next state of its class (0: none).</summary>
    public (int[] Classes, int[] Flags) States()
    {
        // One walk per sequence (its labels from a running offset): a class's states share a parity (the blank the even
        // states, the labels the odd), so its first state is the first seen, and its next the last seen walking back.
        int stride = MaxStates + 2;
        var cls = new int[B * stride];
        var flags = new int[B * stride];
        var seen = new HashSet<int>();
        var next = new Dictionary<int, int>();
        for (int b = 0, off = 0; b < B; off += LabelLengths[b], b++)
        {
            int S = 2 * LabelLengths[b] + 1, row = b * stride;
            seen.Clear();
            next.Clear();
            for (int s = 0; s < S; s++)
            {
                int c = (s & 1) == 0 ? Blank : Labels[off + (s >> 1)];
                cls[row + s] = c;
                bool skip = s >= 2 && c != Blank && c != cls[row + s - 2];
                bool first = seen.Add(c);
                flags[row + s] = (skip ? 1 : 0) | (first ? 2 : 0);
            }

            for (int s = S - 1; s >= 0; s--)
            {
                int c = cls[row + s];
                if (next.TryGetValue(c, out int k)) flags[row + s] |= k << 2;
                next[c] = s;
            }
        }

        return (cls, flags);
    }

    /// <summary>The float64 reference: the loss per sequence and the gradient with respect to the logits (T × B × C).</summary>
    public (double[] Loss, double[] Grad) Reference(double[] x)
    {
        var loss = new double[B];
        var grad = new double[Values];
        for (int b = 0; b < B; b++)
        {
            int tb = InputLengths[b], S = 2 * LabelLengths[b] + 1;
            var logy = new double[tb, C];
            for (int t = 0; t < tb; t++)
            {
                long row = ((long)t * B + b) * C;
                double m = double.NegativeInfinity, sum = 0;
                for (int c = 0; c < C; c++) m = Math.Max(m, x[row + c]);
                for (int c = 0; c < C; c++) sum += Math.Exp(x[row + c] - m);
                double lse = m + Math.Log(sum);
                for (int c = 0; c < C; c++) logy[t, c] = x[row + c] - lse;
            }

            var cls = new int[S];
            for (int s = 0; s < S; s++) cls[s] = Class(b, s);
            bool Skip(int s) => s >= 2 && cls[s] != Blank && cls[s] != cls[s - 2];
            var alpha = new double[tb, S];
            var beta = new double[tb, S];
            for (int s = 0; s < S; s++) alpha[0, s] = s < 2 ? logy[0, cls[s]] : double.NegativeInfinity;
            for (int t = 1; t < tb; t++)
            {
                for (int s = 0; s < S; s++)
                {
                    double v = Lse(alpha[t - 1, s], s >= 1 ? alpha[t - 1, s - 1] : double.NegativeInfinity, Skip(s) ? alpha[t - 1, s - 2] : double.NegativeInfinity);
                    alpha[t, s] = v + logy[t, cls[s]];
                }
            }

            double logP = Lse(alpha[tb - 1, S - 1], S >= 2 ? alpha[tb - 1, S - 2] : double.NegativeInfinity, double.NegativeInfinity);
            loss[b] = -logP;
            if (double.IsNegativeInfinity(logP)) continue;                   // no alignment: the gradient stays 0
            for (int s = 0; s < S; s++) beta[tb - 1, s] = s >= S - 2 ? logy[tb - 1, cls[s]] : double.NegativeInfinity;
            for (int t = tb - 2; t >= 0; t--)
            {
                for (int s = 0; s < S; s++)
                {
                    double v = Lse(beta[t + 1, s], s + 1 < S ? beta[t + 1, s + 1] : double.NegativeInfinity, s + 2 < S && Skip(s + 2) ? beta[t + 1, s + 2] : double.NegativeInfinity);
                    beta[t, s] = v + logy[t, cls[s]];
                }
            }

            for (int t = 0; t < tb; t++)
            {
                long row = ((long)t * B + b) * C;
                for (int c = 0; c < C; c++) grad[row + c] = Math.Exp(logy[t, c]);
                for (int s = 0; s < S; s++) grad[row + cls[s]] -= Math.Exp(alpha[t, s] + beta[t, s] - logy[t, cls[s]] - logP);
            }
        }

        return (loss, grad);
    }

    private static double Lse(double a, double b, double c)
    {
        double m = Math.Max(a, Math.Max(b, c));
        return double.IsNegativeInfinity(m) ? m : m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m) + Math.Exp(c - m));
    }
}

internal sealed partial class Kernels
{
    /// <summary>The ctc loss per sequence (B values as the type) and, when grad is given, the gradient with respect to the
    /// logits (T × B × C as the type); returns the lane's name.</summary>
    public static string Ctc<TN>(CtcArgs a, byte[] x, byte[] loss, byte[]? grad) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        if (TN.Granularity > 8)
        {
            // A block format: the logits decoded into float32 storage once, the results encoded by blocks once (a sequence's
            // rows do not start on blocks).
            var lx = ToLane<TN>(x, a.Values);
            var lloss = new byte[a.B * 4];
            var lgrad = grad is null ? null : new byte[a.Values * 4];
            var lanes = new CtcVisitor(a, lx, lloss, lgrad);
            Float32.Float(ref lanes);
            FromLane<TN>(lloss, loss, a.B);
            if (grad is not null) FromLane<TN>(lgrad!, grad, a.Values);
            return "f32";
        }

        // Any other float type read and written as stored, a row at a time inside its sequence's task (no widening pass).
        var vis = new CtcVisitor(a, x, loss, grad);
        TN.Float(ref vis);
        return Wide<TN>() ? "f64" : "f32";
    }

    private readonly struct CtcVisitor(CtcArgs a, byte[] x, byte[] loss, byte[]? grad) : IFloatLaneVisitor
    {
        public void Run<TS, TLane, TL>() where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            Team.For(a.B, 1, (long)a.T * a.C * (grad is null ? 2 : 4) + (long)a.T * a.MaxStates * 16, new CtcBody<TS, TLane, TL>(a, x, loss, grad));
        }
    }

    // A sequence: its rows' log-sum-exp and the states' emissions; α forward (a vector of states at a time), log P and
    // the loss; for the gradient β backward and, per frame, the row's softmax less each state's exp(α + β − log y − log P).
    private readonly struct CtcBody<TS, TLane, TL>(CtcArgs a, byte[] x, byte[] loss, byte[]? grad) : Team.IBody
        where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            for (int b = first; b < last; b++) Sequence(b);
        }

        private static ArrayPool<TLane> pool0() => ArrayPool<TLane>.Shared;

        private static TLane Log(TLane v) => TL.Log(new Vector<TLane>(v))[0];

        private static bool IsNegInf(TLane v) => EqualityComparer<TLane>.Default.Equals(v, TL.NegativeInfinity);

        // lse of three vectors of log values: −∞ where all three are.
        private static Vector<TLane> Lse(Vector<TLane> p, Vector<TLane> q, Vector<TLane> r)
        {
            var negInf = new Vector<TLane>(TL.NegativeInfinity);
            var m = TL.Max(p, TL.Max(q, r));
            var none = TL.Equal(m, negInf);
            var ms = TL.Select(none, Vector<TLane>.Zero, m);
            var sum = TL.Add(TL.Add(TL.Exp(TL.Sub(p, ms)), TL.Exp(TL.Sub(q, ms))), TL.Exp(TL.Sub(r, ms)));
            return TL.Select(none, negInf, TL.Add(ms, TL.Log(sum)));
        }

        private void Sequence(int b)
        {
            int B = a.B, C = a.C, tb = a.InputLengths[b], S = 2 * a.LabelLengths[b] + 1, w = Vector<TLane>.Count;
            int Sp = (S + w - 1) / w * w, stride = Sp + 2 * w;              // states rounded to whole vectors; pads around them
            // float32 / float64 rows read and written where they are stored; a narrower type's rows converted into lanes here.
            bool own = TS.Name is "float32" or "float64";
            var X = own ? null : pool0().Rent(tb * C);
            if (!own) for (int t = 0; t < tb; t++) TS.Load(x, (int)(((long)t * B + b) * C), X.AsSpan(t * C, C));
            var negInf = TL.NegativeInfinity;
            var pool = ArrayPool<TLane>.Shared;
            var lse = pool.Rent(tb);
            var emit = pool.Rent(tb * Sp);                                  // log y_t(l'_s), −∞ past S
            var skipA = pool.Rent(Sp);                                      // 0 where α may skip from s − 2, −∞ where not
            var skipB = pool.Rent(Sp);                                      // 0 where β may skip from s + 2
            var alpha = pool.Rent(tb * stride);                             // α_t(s) at t·stride + w + s; −∞ in the pads
            var beta = pool.Rent(2 * stride);
            var gamma = pool.Rent(Sp);
            var cls = ArrayPool<int>.Shared.Rent(S);
            try
            {
                int off = a.Offset(b);
                for (int s = 0; s < S; s++) cls[s] = (s & 1) == 0 ? a.Blank : a.Labels[off + (s >> 1)];
                int blank = a.Blank;
                bool Skip(int s) => s >= 2 && s < S && cls[s] != blank && cls[s] != cls[s - 2];
                for (int s = 0; s < Sp; s++)
                {
                    skipA[s] = Skip(s) ? TL.Zero : negInf;
                    skipB[s] = Skip(s + 2) ? TL.Zero : negInf;
                }

                // Each frame's log-sum-exp over the classes, and the states' emissions.
                for (int t = 0; t < tb; t++)
                {
                    Span<TLane> row = own ? MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice((int)(((long)t * B + b) * C), C) : X.AsSpan(t * C, C);
                    var mv = new Vector<TLane>(negInf);
                    int c = 0;
                    for (; c + w <= C; c += w) mv = TL.Max(mv, new Vector<TLane>(row.Slice(c, w)));
                    var m = negInf;
                    for (int i = 0; i < w; i++) m = TL.Max(m, mv[i]);
                    for (; c < C; c++) m = TL.Max(m, row[c]);
                    var vm = new Vector<TLane>(m);
                    var sv = Vector<TLane>.Zero;
                    for (c = 0; c + w <= C; c += w) sv = TL.Add(sv, TL.Exp(TL.Sub(new Vector<TLane>(row.Slice(c, w)), vm)));
                    var sum = TL.Zero;
                    for (int i = 0; i < w; i++) sum = TL.Add(sum, sv[i]);
                    for (; c < C; c++) sum = TL.Add(sum, TL.Exp(TL.Sub(row[c], m)));
                    lse[t] = TL.Add(m, Log(sum));
                    for (int s = 0; s < Sp; s++) emit[t * Sp + s] = s < S ? TL.Sub(row[cls[s]], lse[t]) : negInf;
                }

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
                        TL.Add(Lse(p, q, r), new Vector<TLane>(emit.AsSpan(t * Sp + s, w))).CopyTo(cur.Slice(w + s, w));
                    }

                    cur.Slice(w + S, stride - w - S).Fill(negInf);
                }

                var last = alpha.AsSpan((tb - 1) * stride + w, S);
                var logP = Lse(new Vector<TLane>(last[S - 1]), new Vector<TLane>(S >= 2 ? last[S - 2] : negInf), new Vector<TLane>(negInf))[0];
                Span<TLane> one = [TL.Sub(TL.Zero, logP)];
                TS.Store(one, loss, b);
                if (grad is null) return;

                var g = pool.Rent(C);
                bool none = IsNegInf(logP);
                g.AsSpan(0, C).Clear();
                for (int t = none ? 0 : tb; t < a.T; t++)
                {
                    int at0 = (int)(((long)t * B + b) * C);
                    if (own) MemoryMarshal.Cast<byte, TLane>(grad.AsSpan()).Slice(at0, C).Clear(); else TS.Store(g.AsSpan(0, C), grad, at0);
                }

                if (none) { pool.Return(g); return; }

                // β backward, the gradient of each frame as β reaches it.
                var vLogP = new Vector<TLane>(logP);
                beta.AsSpan(0, 2 * stride).Fill(negInf);
                for (int t = tb - 1; t >= 0; t--)
                {
                    var cur = beta.AsSpan((t & 1) * stride, stride);
                    if (t == tb - 1)
                    {
                        cur[w + S - 1] = emit[t * Sp + S - 1];
                        if (S >= 2) cur[w + S - 2] = emit[t * Sp + S - 2];
                    }
                    else
                    {
                        var prev = beta.AsSpan(((t + 1) & 1) * stride, stride);
                        for (int s = 0; s < Sp; s += w)
                        {
                            var p = new Vector<TLane>(prev.Slice(w + s, w));
                            var q = new Vector<TLane>(prev.Slice(w + s + 1, w));
                            var r = TL.Add(new Vector<TLane>(prev.Slice(w + s + 2, w)), new Vector<TLane>(skipB.AsSpan(s, w)));
                            TL.Add(Lse(p, q, r), new Vector<TLane>(emit.AsSpan(t * Sp + s, w))).CopyTo(cur.Slice(w + s, w));
                        }

                        cur.Slice(w + S, stride - w - S).Fill(negInf);       // past the states: −∞ again for the next frame
                    }

                    var at = alpha.AsSpan(t * stride + w, Sp);
                    for (int s = 0; s < Sp; s += w)
                    {
                        var e = TL.Sub(TL.Sub(TL.Add(new Vector<TLane>(at.Slice(s, w)), new Vector<TLane>(cur.Slice(w + s, w))), new Vector<TLane>(emit.AsSpan(t * Sp + s, w))), vLogP);
                        TL.Exp(e).CopyTo(gamma.AsSpan(s, w));
                    }

                    Span<TLane> row = own ? MemoryMarshal.Cast<byte, TLane>(x.AsSpan()).Slice((int)(((long)t * B + b) * C), C) : X.AsSpan(t * C, C);
                    var vl = new Vector<TLane>(lse[t]);
                    int at1 = (int)(((long)t * B + b) * C), c = 0;
                    Span<TLane> gs = own ? MemoryMarshal.Cast<byte, TLane>(grad.AsSpan()).Slice(at1, C) : g.AsSpan(0, C);
                    for (; c + w <= C; c += w) TL.Exp(TL.Sub(new Vector<TLane>(row.Slice(c, w)), vl)).CopyTo(gs.Slice(c, w));
                    for (; c < C; c++) gs[c] = TL.Exp(TL.Sub(row[c], lse[t]));
                    for (int s = 0; s < S; s++) gs[cls[s]] = TL.Sub(gs[cls[s]], gamma[s]);
                    if (!own) TS.Store(gs, grad, at1);
                }

                pool.Return(g);
            }
            finally
            {
                if (X is not null) pool.Return(X);
                pool.Return(lse); pool.Return(emit); pool.Return(skipA); pool.Return(skipB); pool.Return(alpha); pool.Return(beta); pool.Return(gamma);
                ArrayPool<int>.Shared.Return(cls);
            }
        }
    }
}
