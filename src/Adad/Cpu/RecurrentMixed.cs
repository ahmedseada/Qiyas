// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The recurrent cells of a narrower float or a block format (float32 and float64 take Recurrent.cs's own-lane path):
// the same steps and sequences, every input read where it is stored. The gates, the states across a sequence's
// steps and the gates' gradients are intermediates in the float lane's own type (TS, Mixed.cs); x, h0, c0, W, U and the
// biases are read as the type by the products, the cells and the adds; each output — h', c', dx, dh, dc, dW, dU, db —
// is stored as the type by the kernel that computes it (a block format's cell outputs: the lane's values, encoded once).
using System.Runtime.CompilerServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    private readonly struct LstmStepCall(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.LstmStep(a, x, h, c, w, u, bias, hOut, cOut);
    }

    private readonly struct LstmStepBackwardCall(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcOut,
        byte[] dx, byte[] dh, byte[] dc, byte[] dw, byte[] du, byte[] db) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.LstmStepBackward(a, x, h, c, w, u, bias, dhOut, dcOut, dx, dh, dc, dw, du, db);
    }

    private readonly struct GruStepCall(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.GruStep(a, x, h, w, u, bias, biasH, hOut);
    }

    private readonly struct GruStepBackwardCall(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut,
        byte[] dx, byte[] dh, byte[] dw, byte[] du, byte[] db, byte[] dbh) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.GruStepBackward(a, x, h, w, u, bias, biasH, dhOut, dx, dh, dw, du, db, dbh);
    }

    private readonly struct LstmSequenceCall(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.LstmSequence(a, T, x, h0, c0, w, u, bias, hOut, cOut);
    }

    private readonly struct LstmSequenceBackwardCall(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcLast,
        byte[] dx, byte[] dh0, byte[] dc0, byte[] dw, byte[] du, byte[] db) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.LstmSequenceBackward(a, T, x, h0, c0, w, u, bias, dhOut, dcLast, dx, dh0, dc0, dw, du, db);
    }

    private readonly struct GruSequenceCall(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.GruSequence(a, T, x, h0, w, u, bias, biasH, hOut);
    }

    private readonly struct GruSequenceBackwardCall(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut,
        byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> =>
            RecurrentMixed<TN, TS, TLane, TL>.GruSequenceBackward(a, T, x, h0, w, u, bias, biasH, dhOut, dx, dh0, dw, du, db, dbh);
    }

    private static class RecurrentMixed<TN, TS, TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        private static int Size => Unsafe.SizeOf<TLane>();

        // The cells store their outputs as the type where it is whole values; a block format's are the lane's values,
        // encoded once after (a block straddles rows).
        private static bool Typed => TN.Granularity == 1;

        private static Arg Lanes(long values) => new(new byte[Math.Max(1, values) * Size], 0, true);

        private static Arg ScratchLanes(int slot, long values) => new(Scratch.Get(slot, Math.Max(1, values) * Size), 0, true);

        // A cell's output: the type's own array (typed), or lanes to encode after (Done).
        private static Arg Out(byte[] output, long values) => Typed ? new Arg(output) : Lanes(values);

        private static void Done(Arg o, byte[] output, long values)
        {
            if (o.Lanes) StoreLanes<TN, TLane>(o.X!, output, 0, values);
        }

        private static void Product(Arg a, bool transA, Arg b, bool transB, Arg c, int m, int n, int k) => Prod<TN, TS, TLane, TL>(a, transA, b, transB, c, m, n, k);

        private static void Add(Arg x, Arg y, Arg o, int n, int cols = 0) => AddMixed<TN, TS, TLane, TL>(x, y, o, n, cols);

        private static void Cell(CellArgs a, Arg g, Arg g2, Arg s, Arg o1, Arg o2, Arg m1 = default, Arg m2 = default) =>
            Team.For(a.B, RowChunk<TN>(a.B, a.H, a.B / (Threads * 4)), (long)a.B * a.H * 8, new CellBody<TN, TS, TLane, TL>(a, g, g2, s, o1, o2, m1, m2));

        private static void CellBack(CellArgs a, Arg g, Arg g2, Arg s, Arg d1, Arg d2, Arg o1, Arg o2, Arg o3) =>
            Team.For(a.B, RowChunk<TN>(a.B, a.Gates * a.H, a.B / (Threads * 4)), (long)a.B * a.H * 16, new CellBackwardBody<TN, TS, TLane, TL>(a, g, g2, s, d1, d2, o1, o2, o3));

        // The gates (TS, rows × GH, at g) = x·Wᵀ (+ h·Uᵀ) + b.
        private static void Gates(CellArgs a, Arg x, Arg w, Arg h, Arg u, Arg bias, Arg g, int rows)
        {
            int GH = a.Gates * a.H;
            Product(x, false, w, true, g, rows, GH, a.D);                                       // x · Wᵀ (W is GH × D)
            if (h.X is not null)
            {
                var gh = Lanes((long)rows * GH);
                Product(h, false, u, true, gh, rows, GH, a.H);                                  // h · Uᵀ (U is GH × H)
                Add(g, gh, g, rows * GH);
            }

            Add(g, bias, g, rows * GH, GH);
        }

        // From dg (TS, rows × GH): dx = dg·W, dh = dg·U, dU = dgᵀ·h, dW = dgᵀ·x (the type's), db = dg's column sums.
        private static void GatesBackward(CellArgs a, int rows, Arg dg, Arg x, Arg h, Arg w, Arg u, byte[]? dx, byte[]? dh, byte[] dw, byte[]? du, byte[] db)
        {
            int GH = a.Gates * a.H;
            if (dx is not null) Product(dg, false, w, false, new Arg(dx), rows, a.D, GH);       // dg · W
            if (dh is not null) Product(dg, false, u, false, new Arg(dh), rows, a.H, GH);       // dg · U
            if (du is not null) Product(dg, true, h, false, new Arg(du), GH, a.H, rows);        // dgᵀ · h
            Product(dg, true, x, false, new Arg(dw), GH, a.D, rows);                            // dgᵀ · x
            ColumnSums(dg, rows, GH, db);
        }

        // An output of the type: the column sums of dg (TS, rows × cols from its offset 0), in the sum lane.
        private static void ColumnSums(Arg dg, int rows, int cols, byte[] db)
        {
            var (sum, _) = Reduce<TS>("sum", Axis.Cols, dg.X!, rows, cols);
            CopyLanes<TN>(sum, db, cols);
        }

        // dU (the type's) = Σ_t dg_tᵀ · the step's input state: h0 (as the type) for the first step taken's rows, the states
        // hs (TS) of the steps before them for the others' (CellArgs.Shift).
        private static void ShiftedWeights(CellArgs a, int T, Arg dgAll, byte[] h0, Arg hs, byte[] du)
        {
            int GH = a.Gates * a.H, bg = a.B * GH, bh = a.B * a.H;
            long values = (long)GH * a.H;
            if (T == 1) { Product(dgAll, true, new Arg(h0), false, new Arg(du), GH, a.H, a.B); return; }
            var (first, others, input) = a.Shift(T);
            var d0 = Lanes(values);
            var d1 = Lanes(values);
            Product(dgAll.Plus(first * bg), true, new Arg(h0), false, d0, GH, a.H, a.B);
            Product(dgAll.Plus(others * bg), true, hs.Plus(input * bh), false, d1, GH, a.H, (T - 1) * a.B);
            Add(d0, d1, new Arg(du), (int)values);
        }

        // ---- the steps

        public static string LstmStep(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut)
        {
            long bh = (long)a.B * a.H;
            var g = Lanes(a.GateValues);
            Gates(a, new(x), new(w), new(h), new(u), new(bias), g, a.B);
            Arg ho = Out(hOut, bh), co = Out(cOut, bh);
            Cell(a, g, default, new(c), ho, co);
            Done(ho, hOut, bh); Done(co, cOut, bh);
            return TL.Name;
        }

        public static string LstmStepBackward(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcOut,
            byte[] dx, byte[] dh, byte[] dc, byte[] dw, byte[] du, byte[] db)
        {
            long bh = (long)a.B * a.H;
            var g = Lanes(a.GateValues);
            Gates(a, new(x), new(w), new(h), new(u), new(bias), g, a.B);
            var dg = Lanes(a.GateValues);
            var dco = Out(dc, bh);
            CellBack(a, g, default, new(c), new(dhOut), new(dcOut), dg, dco, default);
            Done(dco, dc, bh);
            GatesBackward(a, a.B, dg, new(x), new(h), new(w), new(u), dx, dh, dw, du, db);
            return TL.Name;
        }

        public static string GruStep(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut)
        {
            long bh = (long)a.B * a.H;
            var hArgs = a with { D = a.H };
            var gx = Lanes(a.GateValues);
            var gh = Lanes(a.GateValues);
            Gates(a, new(x), new(w), default, default, new(bias), gx, a.B);
            Gates(hArgs, new(h), new(u), default, default, new(biasH), gh, a.B);
            var ho = Out(hOut, bh);
            Cell(a, gx, gh, new(h), ho, default);
            Done(ho, hOut, bh);
            return TL.Name;
        }

        public static string GruStepBackward(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut,
            byte[] dx, byte[] dh, byte[] dw, byte[] du, byte[] db, byte[] dbh)
        {
            int GH = a.Gates * a.H;
            long bh = (long)a.B * a.H;
            var hArgs = a with { D = a.H };
            var gx = Lanes(a.GateValues);
            var gh = Lanes(a.GateValues);
            Gates(a, new(x), new(w), default, default, new(bias), gx, a.B);
            Gates(hArgs, new(h), new(u), default, default, new(biasH), gh, a.B);
            var dgx = Lanes(a.GateValues);
            var dgh = Lanes(a.GateValues);
            var dhCell = Lanes(bh);
            CellBack(a, gx, gh, new(h), new(dhOut), default, dgx, dgh, dhCell);
            GatesBackward(a, a.B, dgx, new(x), default, new(w), default, dx, null, dw, null, db);
            var dhGates = Lanes(bh);
            Product(dgh, false, new(u), false, dhGates, a.B, a.H, GH);                          // dgh · U
            Add(dhGates, dhCell, new Arg(dh), (int)bh);
            Product(dgh, true, new(h), false, new Arg(du), GH, a.H, a.B);                       // dghᵀ · h
            ColumnSums(dgh, a.B, GH, dbh);
            return TL.Name;
        }

        // ---- the sequences

        // x · Wᵀ + b of every step at once (TS, T·B × GH).
        private static Arg InputGates(CellArgs a, int T, byte[] x, byte[] w, byte[] bias)
        {
            int GH = a.Gates * a.H, rows = T * a.B;
            var gx = ScratchLanes(Scratch.Gx, (long)rows * GH);
            Product(new(x), false, new(w), true, gx, rows, GH, a.D);
            Add(gx, new(bias), gx, rows * GH, GH);
            return gx;
        }

        // The LSTM over the steps. hs, cs: each step's h, c (T × B × H) — lanes, the next step reading its state there; or
        // the type's outputs, the cell then mirroring its state into two alternating lane buffers the next step reads.
        // gates: each step's gates kept (TS, T × B × GH) for the backward.
        private static void LstmForward(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, Arg hs, Arg cs, byte[]? gates)
        {
            int GH = a.Gates * a.H, bh = a.B * a.H, bg = a.B * GH;
            var gx = InputGates(a, T, x, w, bias);
            var g = gates is null ? ScratchLanes(Scratch.StepG, bg) : default;
            var gh = ScratchLanes(Scratch.StepG2, bg);
            Arg[] hm = hs.Lanes ? [] : [ScratchLanes(Scratch.H0, bh), ScratchLanes(Scratch.H1, bh)];
            Arg[] cm = cs.Lanes ? [] : [ScratchLanes(Scratch.C0, bh), ScratchLanes(Scratch.C1, bh)];
            Arg hPrev = new(h0), cPrev = new(c0);
            for (int t = 0; t < T; t++)
            {
                int step = a.Step(t, T);
                var gt = gates is null ? g : new Arg(gates, step * bg, true);
                Product(hPrev, false, new(u), true, gh, a.B, GH, a.H);                          // h · Uᵀ
                Add(gx.Plus(step * bg), gh, gt, bg);
                Arg ht = hs.Plus(step * bh), ct = cs.Plus(step * bh);
                if (hs.Lanes) { Cell(a, gt, default, cPrev, ht, ct); hPrev = ht; cPrev = ct; }
                else { Cell(a, gt, default, cPrev, ht, ct, hm[t & 1], cm[t & 1]); hPrev = hm[t & 1]; cPrev = cm[t & 1]; }
            }
        }

        // The GRU over the steps: gx of all the steps (TS) returned; a step's gh = h·Uᵀ + bh kept into ghAll when given.
        // hs as the LSTM's.
        private static Arg GruForward(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, Arg hs, byte[]? ghAll)
        {
            int GH = a.Gates * a.H, bh = a.B * a.H, bg = a.B * GH;
            var gx = InputGates(a, T, x, w, bias);
            var ghStep = ghAll is null ? ScratchLanes(Scratch.StepG2, bg) : default;
            Arg[] hm = hs.Lanes ? [] : [ScratchLanes(Scratch.H0, bh), ScratchLanes(Scratch.H1, bh)];
            Arg hPrev = new(h0);
            for (int t = 0; t < T; t++)
            {
                int step = a.Step(t, T);
                var gh = ghAll is null ? ghStep : new Arg(ghAll, step * bg, true);
                Product(hPrev, false, new(u), true, gh, a.B, GH, a.H);                          // h · Uᵀ
                Add(gh, new(biasH), gh, bg, GH);
                var ht = hs.Plus(step * bh);
                if (hs.Lanes) { Cell(a, gx.Plus(step * bg), gh, hPrev, ht, default); hPrev = ht; }
                else { Cell(a, gx.Plus(step * bg), gh, hPrev, ht, default, hm[t & 1]); hPrev = hm[t & 1]; }
            }

            return gx;
        }

        public static string LstmSequence(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut)
        {
            long values = (long)T * a.B * a.H;
            Arg hs = Out(hOut, values), cs = Out(cOut, values);
            LstmForward(a, T, x, h0, c0, w, u, bias, hs, cs, null);
            Done(hs, hOut, values); Done(cs, cOut, values);
            return TL.Name;
        }

        public static string GruSequence(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut)
        {
            long values = (long)T * a.B * a.H;
            var hs = Out(hOut, values);
            GruForward(a, T, x, h0, w, u, bias, biasH, hs, null);
            Done(hs, hOut, values);
            return TL.Name;
        }

        // The forward recomputed once, each step's gates and states kept (TS); per step back only the cell's backward and
        // dh = dg · U (the chain); dx, dW, dU, db over all the steps at once.
        public static string LstmSequenceBackward(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcLast,
            byte[] dx, byte[] dh0, byte[] dc0, byte[] dw, byte[] du, byte[] db)
        {
            int GH = a.Gates * a.H, bh = a.B * a.H, bg = a.B * GH;
            var hs = ScratchLanes(Scratch.Hs, (long)T * bh);
            var cs = ScratchLanes(Scratch.Cs, (long)T * bh);
            var gates = ScratchLanes(Scratch.Gates, (long)T * bg);
            LstmForward(a, T, x, h0, c0, w, u, bias, hs, cs, gates.X);
            var dgAll = ScratchLanes(Scratch.Dg, (long)T * bg);
            var dht = ScratchLanes(Scratch.Dh, bh);
            Arg[] dhb = [ScratchLanes(Scratch.Dh0, bh), ScratchLanes(Scratch.Dh1, bh)], dcb = [ScratchLanes(Scratch.Dc0, bh), ScratchLanes(Scratch.Dc1, bh)];
            Arg dhNext = default, dcNext = new(dcLast);
            for (int t = T - 1; t >= 0; t--)
            {
                int step = a.Step(t, T);
                Add(new Arg(dhOut, step * bh), dhNext, dht, bh);                                  // the loss's gradient plus the next step's
                var dg = dgAll.Plus(step * bg);
                CellBack(a, gates.Plus(step * bg), default, t == 0 ? new Arg(c0) : cs.Plus(a.Step(t - 1, T) * bh), dht, dcNext, dg, dcb[t & 1], default);
                Product(dg, false, new(u), false, dhb[t & 1], a.B, a.H, GH);                   // dg · U
                dhNext = dhb[t & 1]; dcNext = dcb[t & 1];
            }

            int rows = T * a.B;
            Product(dgAll, false, new(w), false, new Arg(dx), rows, a.D, GH);                  // dx = dg · W
            Product(dgAll, true, new(x), false, new Arg(dw), GH, a.D, rows);                   // dW = dgᵀ · x
            ShiftedWeights(a, T, dgAll, h0, hs, du);
            ColumnSums(dgAll, rows, GH, db);
            StoreLanes<TN, TLane>(dhNext.X!, dh0, 0, bh);
            StoreLanes<TN, TLane>(dcNext.X!, dc0, 0, bh);
            return TL.Name;
        }

        // The forward recomputed once, gx (all steps), each step's gh and state kept; per step back only the cell's
        // backward and dh = dgh · U + the cell's direct part; dx, dW, db from dgx and dU, dbh from dgh over all the steps.
        public static string GruSequenceBackward(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut,
            byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh)
        {
            int GH = a.Gates * a.H, bh = a.B * a.H, bg = a.B * GH;
            var hs = ScratchLanes(Scratch.Hs, (long)T * bh);
            var ghAll = ScratchLanes(Scratch.Gates, (long)T * bg);
            var gx = GruForward(a, T, x, h0, w, u, bias, biasH, hs, ghAll.X);
            var dgxAll = ScratchLanes(Scratch.Dg, (long)T * bg);
            var dghAll = ScratchLanes(Scratch.Dg2, (long)T * bg);
            var dht = ScratchLanes(Scratch.Dh, bh);
            var dhCell = ScratchLanes(Scratch.DhCell, bh);
            Arg[] dhb = [ScratchLanes(Scratch.Dh0, bh), ScratchLanes(Scratch.Dh1, bh)];
            Arg dhNext = default;
            for (int t = T - 1; t >= 0; t--)
            {
                int step = a.Step(t, T);
                Add(new Arg(dhOut, step * bh), dhNext, dht, bh);                                  // the loss's gradient plus the next step's
                var dgh = dghAll.Plus(step * bg);
                CellBack(a, gx.Plus(step * bg), ghAll.Plus(step * bg), t == 0 ? new Arg(h0) : hs.Plus(a.Step(t - 1, T) * bh), dht, default, dgxAll.Plus(step * bg), dgh, dhCell);
                var dhp = dhb[t & 1];
                Product(dgh, false, new(u), false, dhp, a.B, a.H, GH);                         // dgh · U
                Add(dhp, dhCell, dhp, bh);
                dhNext = dhp;
            }

            int rows = T * a.B;
            var hArgs = a with { D = a.H };
            Product(dgxAll, false, new(w), false, new Arg(dx), rows, a.D, GH);                 // dx = dgx · W
            Product(dgxAll, true, new(x), false, new Arg(dw), GH, a.D, rows);                  // dW = dgxᵀ · x
            ColumnSums(dgxAll, rows, GH, db);
            ShiftedWeights(hArgs, T, dghAll, h0, hs, du);
            ColumnSums(dghAll, rows, GH, dbh);
            StoreLanes<TN, TLane>(dhNext.X!, dh0, 0, bh);
            return TL.Name;
        }
    }
}
