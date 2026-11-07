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
// The lane is the float lane (f32; f64 for float64): a sequence's task loads its rows into lanes (in place for float32
// and float64) and stores its loss and gradient as the type (a block format's: the lane's values, encoded once — a
// sequence's rows do not start on blocks).

using System.Buffers;
using System.Numerics;
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
            long work = (long)a.T * a.C * (grad is null ? 2 : 4) + (long)a.T * a.MaxStates * 16;
            if (TS.Granularity == 1)
            {
                Team.For(a.B, 1, work, new CtcBody<TS, TLane, TL>(a, x, loss, grad, false));
                return;
            }

            // A block format: the loss and the gradient as the lane's values, encoded once.
            int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>();
            var lloss = new byte[a.B * size];
            var lgrad = grad is null ? null : new byte[a.Values * size];
            Team.For(a.B, 1, work, new CtcBody<TS, TLane, TL>(a, x, lloss, lgrad, true));
            StoreLanes<TS, TLane>(lloss, loss, 0, a.B);
            if (grad is not null) StoreLanes<TS, TLane>(lgrad!, grad, 0, a.Values);
        }
    }

    // A sequence: its rows' log-sum-exp and the states' emissions; α forward (a vector of states at a time), log P and
    // the loss; for the gradient β backward and, per frame, the row's softmax less each state's exp(α + β − log y − log P).
    // lanesOut: loss and grad are the lane's values (a block format's, encoded after).
    private readonly struct CtcBody<TS, TLane, TL>(CtcArgs a, byte[] x, byte[] loss, byte[]? grad, bool lanesOut) : Team.IBody
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
            // float32 / float64 rows read and written where they are stored; a narrower type's rows converted into lanes here
            // (own), its outputs stored as the type, or written as lanes (lanesOut).
            bool own = TS.Direct, ownOut = own || lanesOut;
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
                if (ownOut) MemoryMarshal.Cast<byte, TLane>(loss.AsSpan())[b] = one[0]; else TS.Store(one, loss, b);
                if (grad is null) return;

                var g = pool.Rent(C);
                bool none = IsNegInf(logP);
                g.AsSpan(0, C).Clear();
                for (int t = none ? 0 : tb; t < a.T; t++)
                {
                    int at0 = (int)(((long)t * B + b) * C);
                    if (ownOut) MemoryMarshal.Cast<byte, TLane>(grad.AsSpan()).Slice(at0, C).Clear(); else TS.Store(g.AsSpan(0, C), grad, at0);
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
                    Span<TLane> gs = ownOut ? MemoryMarshal.Cast<byte, TLane>(grad.AsSpan()).Slice(at1, C) : g.AsSpan(0, C);
                    for (; c + w <= C; c += w) TL.Exp(TL.Sub(new Vector<TLane>(row.Slice(c, w)), vl)).CopyTo(gs.Slice(c, w));
                    for (; c < C; c++) gs[c] = TL.Exp(TL.Sub(row[c], lse[t]));
                    for (int s = 0; s < S; s++) gs[cls[s]] = TL.Sub(gs[cls[s]], gamma[s]);
                    if (!ownOut) TS.Store(gs, grad, at1);
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
