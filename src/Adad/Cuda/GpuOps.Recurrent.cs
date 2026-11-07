// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's recurrent cells and steps (Recurrent.cs): the cells by one kernel each, the gates by the products on
// views and the broadcast add, the steps and backward passes as the same compositions on device buffers: the type's own
// for float32 and float64, else in the f32 lane on the type's tensors read where they are stored (GpuOps.RecurrentStored.cs).
using Adad.Abstraction;

namespace Adad.Cuda;

public sealed partial class GpuOps
{
    private void LaunchCell<TN>(CellArgs a, bool backward, ulong g, ulong g2, ulong state, ulong d1, ulong d2, ulong o1, ulong o2, ulong o3) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (f.IsBlock)
        {
            // A block format's cell: the inputs decoded where they are stored (a mixed cell), the outputs the cell's f32
            // values encoded by blocks (a block may span rows of the cell's threads).
            long gates = a.GateValues, bh = (long)a.B * a.H;
            long o1n = backward ? gates : bh, o2n = backward && a.Gates == 3 ? gates : bh;
            var r = new List<ulong>();
            Buf In(ulong x) => x == 0 ? default : new(x, f);
            ulong Out(ulong x, long count) { if (x == 0) return 0; ulong l = gpu.Rent(count * 4); r.Add(l); return l; }
            try
            {
                ulong lo1 = Out(o1, o1n), lo2 = Out(o2, o2n), lo3 = Out(o3, bh);
                CellMixed(a, backward, In(g), In(g2), In(state), In(d1), In(d2), lo1, lo2, lo3);
                if (o1 != 0) Back(f, Lane.F32, lo1, o1, o1n);
                if (o2 != 0) Back(f, Lane.F32, lo2, o2, o2n);
                if (o3 != 0) Back(f, Lane.F32, lo3, o3, bh);
                return;
            }
            finally { foreach (var q in r) gpu.Return(q); }
        }

        var fn = gpu.Cached($"cell:{a.Kind}:{backward}:{f.Name}", (out string e) => Ptx.Cell(gpu, f, lane, a.Kind, backward, out e));
        Gpu.Launch(fn, Grid(f.Words((long)a.B * a.H), 256), 1, 256, 1, 0, g, g2, state, d1, d2, o1, o2, o3, Gpu.U(a.B), Gpu.U(a.H));
    }

    /// <summary>LSTM: (h', c') from the gates (B × 4H) and c; returns the lane's name.</summary>
    public string LstmCell<TN>(CellArgs a, ulong g, ulong c, ulong hOut, ulong cOut) where TN : struct, IAdad<TN>
    {
        LaunchCell<TN>(a, false, g, 0, c, 0, 0, hOut, cOut, 0);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>GRU: h' from gx, gh (B × 3H) and h.</summary>
    public string GruCell<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong hOut) where TN : struct, IAdad<TN>
    {
        LaunchCell<TN>(a, false, gx, gh, h, 0, 0, hOut, 0, 0);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>LSTM backward: dg (B × 4H) and dc from dh', dc' (and the gates, c).</summary>
    public string LstmCellBackward<TN>(CellArgs a, ulong g, ulong c, ulong dh, ulong dcNext, ulong dg, ulong dc) where TN : struct, IAdad<TN>
    {
        LaunchCell<TN>(a, true, g, 0, c, dh, dcNext, dg, dc, 0);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>GRU backward: dgx, dgh (B × 3H) and dh from dh' (and gx, gh, h).</summary>
    public string GruCellBackward<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong dhOut, ulong dgx, ulong dgh, ulong dh) where TN : struct, IAdad<TN>
    {
        LaunchCell<TN>(a, true, gx, gh, h, dhOut, 0, dgx, dgh, dh);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>The gates g = x·Wᵀ (+ h·Uᵀ) + b (B × GH) as stored values of the type.</summary>
    public void CellGates<TN>(CellArgs a, ulong x, ulong w, ulong h, ulong u, ulong bias, ulong g) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int GH = a.Gates * a.H;
        MatMulView<TN>(new ProductView(a.B, GH, a.D, false, true), x, w, g);
        if (h != 0 && u != 0)
        {
            ulong gh = gpu.Rent(f.Words(a.GateValues) * (f.WordBits / 8));
            try
            {
                MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), h, u, gh);
                Map<TN>(Ops.Find("add"), g, gh, 0, g, (int)a.GateValues, 0, 0);
            }
            finally
            {
                gpu.Return(gh);
            }
        }

        MapRow<TN>(Ops.Find("add"), g, bias, g, a.B, GH);
    }

    private bool OwnLane(Fmt f) => f.Is("float32") || f.Is("float64");

    /// <summary>An LSTM step: (h', c') from x, h, c and W (4H × D), U (4H × H), b (4H).</summary>
    public string LstmStep<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return LstmStepMixed(f, a, x, h, c, w, u, bias, hOut, cOut);

        ulong g = gpu.Rent(f.Words(a.GateValues) * (f.WordBits / 8));
        try
        {
            CellGates<TN>(a, x, w, h, u, bias, g);
            return LstmCell<TN>(a, g, c, hOut, cOut);
        }
        finally
        {
            gpu.Return(g);
        }
    }

    private void GatesBackward<TN>(CellArgs a, ulong dg, ulong x, ulong h, ulong w, ulong u, ulong dx, ulong dh, ulong dw, ulong du, ulong db) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int GH = a.Gates * a.H;
        MatMulView<TN>(new ProductView(a.B, a.D, GH, false, false), dg, w, dx);
        if (h != 0 && u != 0 && dh != 0 && du != 0)
        {
            MatMulView<TN>(new ProductView(a.B, a.H, GH, false, false), dg, u, dh);
            MatMulView<TN>(new ProductView(GH, a.H, a.B, true, false), dg, h, du);
        }

        MatMulView<TN>(new ProductView(GH, a.D, a.B, true, false), dg, x, dw);
        // db = the column sums of dg, as lane values, narrowed into the type.
        var lane = RowLane(f);
        ulong sums = gpu.Rent((long)GH * 8);
        try
        {
            string sumLane = Reduce<TN>("sum", Axis.Cols, dg, a.B, GH, sums);
            var sl = LaneOf(sumLane);
            if (sl.IsFloat) { var narrow = gpu.Cached($"narrow:{f.Name}:{sl.Name}", (out string e) => Ptx.Narrow(gpu, f, sl, out e)); Gpu.Launch(narrow, Grid(f.Words(GH), 256), 1, 256, 1, 0, sums, db, Gpu.U(GH)); }
            else throw new NotSupportedException($"{f.Name}: a bias gradient in an integer lane");
        }
        finally
        {
            gpu.Return(sums);
        }
    }

    /// <summary>An LSTM step's backward: dx, dh, dc, dW, dU, db from dh', dc'.</summary>
    public string LstmStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcOut, ulong dx, ulong dh, ulong dc, ulong dw, ulong du, ulong db) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return LstmStepBackwardMixed(f, a, x, h, c, w, u, bias, dhOut, dcOut, dx, dh, dc, dw, du, db);

        long bytes = f.Words(a.GateValues) * (f.WordBits / 8);
        ulong g = gpu.Rent(bytes), dg = gpu.Rent(bytes);
        try
        {
            CellGates<TN>(a, x, w, h, u, bias, g);
            string lane = LstmCellBackward<TN>(a, g, c, dhOut, dcOut, dg, dc);
            GatesBackward<TN>(a, dg, x, h, w, u, dx, dh, dw, du, db);
            return lane;
        }
        finally
        {
            gpu.Return(g); gpu.Return(dg);
        }
    }

    /// <summary>A GRU step's backward: dx, dh, dW, dU, db, dbh from dh'.</summary>
    public string GruStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh, ulong dw, ulong du, ulong db, ulong dbh) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return GruStepBackwardMixed(f, a, x, h, w, u, bias, biasH, dhOut, dx, dh, dw, du, db, dbh);

        long bytes = f.Words(a.GateValues) * (f.WordBits / 8), hBytes = f.Words((long)a.B * a.H) * (f.WordBits / 8);
        ulong gx = gpu.Rent(bytes), gh = gpu.Rent(bytes), dgx = gpu.Rent(bytes), dgh = gpu.Rent(bytes), dhCell = gpu.Rent(hBytes), dhGates = gpu.Rent(hBytes);
        try
        {
            var hArgs = new CellArgs("gru", a.B, a.H, a.H);
            CellGates<TN>(a, x, w, 0, 0, bias, gx);
            CellGates<TN>(hArgs, h, u, 0, 0, biasH, gh);
            string lane = GruCellBackward<TN>(a, gx, gh, h, dhOut, dgx, dgh, dhCell);
            GatesBackward<TN>(a, dgx, x, 0, w, 0, dx, 0, dw, 0, db);
            GatesBackward<TN>(hArgs, dgh, h, 0, u, 0, dhGates, 0, du, 0, dbh);
            Map<TN>(Ops.Find("add"), dhGates, dhCell, 0, dh, (int)((long)a.B * a.H), 0, 0);
            return lane;
        }
        finally
        {
            gpu.Return(gx); gpu.Return(gh); gpu.Return(dgx); gpu.Return(dgh); gpu.Return(dhCell); gpu.Return(dhGates);
        }
    }

    /// <summary>An LSTM over T steps on the device: x (T × B × D), h0, c0 → h, c (T × B × H each).</summary>
    public string LstmSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return LstmSequenceMixed(f, a, T, x, h0, c0, w, u, bias, hOut, cOut);

        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        return LstmForward<TN>(a, T, x, h0, c0, w, u, bias, hOut, cOut, 0);
    }

    // x · Wᵀ + b of every step at once (T·B × GH): one product over all the steps' rows, the bias added once.
    private ulong InputGates<TN>(CellArgs a, int T, ulong x, ulong w, ulong bias) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int GH = a.Gates * a.H, rows = T * a.B;
        ulong gx = gpu.Rent(f.Words((long)rows * GH) * (f.WordBits / 8));
        MatMulView<TN>(new ProductView(rows, GH, a.D, false, true), x, w, gx);
        MapRow<TN>(Ops.Find("add"), gx, bias, gx, rows, GH);
        return gx;
    }

    // The LSTM over the steps from the input gates of all steps: per step h · Uᵀ, the add (into `gates`' step when given:
    // kept for the backward) and the cell; h, c of every step into hOut, cOut.
    private string LstmForward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut, ulong gates) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int GH = a.Gates * a.H;
        long gStep = (long)a.B * GH * f.Bits / 8, hStep = (long)a.B * a.H * f.Bits / 8;
        var add = Ops.Find("add");
        ulong gx = InputGates<TN>(a, T, x, w, bias), gh = gpu.Rent(gStep + 4), g = gates == 0 ? gpu.Rent(gStep + 4) : 0;
        try
        {
            ulong hPrev = h0, cPrev = c0;
            string lane = "";
            for (int t = 0; t < T; t++)
            {
                ulong gt = gates == 0 ? g : gates + (ulong)(t * gStep), ht = hOut + (ulong)(t * hStep), ct = cOut + (ulong)(t * hStep);
                MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), hPrev, u, gh);                  // h · Uᵀ
                Map<TN>(add, gx + (ulong)(t * gStep), gh, 0, gt, a.B * GH, 0, 0);
                lane = LstmCell<TN>(a, gt, cPrev, ht, ct);
                hPrev = ht; cPrev = ct;
            }

            return lane;
        }
        finally
        {
            if (g != 0) gpu.Return(g);
            gpu.Return(gh); gpu.Return(gx);
        }
    }

    // The GRU over the steps: gx of all steps at once (returned; the caller returns it), a step's gh = h · Uᵀ + bh (into
    // `ghAll`'s step when given: kept for the backward) and the cell.
    private (string Lane, ulong Gx) GruForward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut, ulong ghAll) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int GH = a.Gates * a.H;
        long gStep = (long)a.B * GH * f.Bits / 8, hStep = (long)a.B * a.H * f.Bits / 8;
        var add = Ops.Find("add");
        ulong gx = InputGates<TN>(a, T, x, w, bias), gh = ghAll == 0 ? gpu.Rent(gStep + 4) : 0;
        try
        {
            ulong hPrev = h0;
            string lane = "";
            for (int t = 0; t < T; t++)
            {
                ulong ght = ghAll == 0 ? gh : ghAll + (ulong)(t * gStep), ht = hOut + (ulong)(t * hStep);
                MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), hPrev, u, ght);                 // h · Uᵀ
                MapRow<TN>(add, ght, biasH, ght, a.B, GH);
                lane = GruCell<TN>(a, gx + (ulong)(t * gStep), ght, hPrev, ht);
                hPrev = ht;
            }

            return (lane, gx);
        }
        finally
        {
            if (gh != 0) gpu.Return(gh);
        }
    }

    // The weights' gradients of all steps at once from the gates' gradients dg (rows = T·B, × GH): dx = dg · W (when dx is
    // given), dW = dgᵀ · x, dU = dgᵀ · hPrev (when given: each step's input state, rows × H), db = dg's column sums.
    private void SequenceWeights<TN>(CellArgs a, int rows, ulong dg, ulong x, ulong hPrev, ulong w, ulong dx, ulong dw, ulong du, ulong db) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        int GH = a.Gates * a.H;
        if (dx != 0) MatMulView<TN>(new ProductView(rows, a.D, GH, false, false), dg, w, dx);
        MatMulView<TN>(new ProductView(GH, a.D, rows, true, false), dg, x, dw);
        if (hPrev != 0) MatMulView<TN>(new ProductView(GH, a.H, rows, true, false), dg, hPrev, du);
        ulong sums = gpu.Rent((long)GH * 8);
        try
        {
            string sumLane = Reduce<TN>("sum", Axis.Cols, dg, rows, GH, sums);
            var sl = LaneOf(sumLane);
            if (!sl.IsFloat) throw new NotSupportedException($"{f.Name}: a bias gradient in an integer lane");
            var narrow = gpu.Cached($"narrow:{f.Name}:{sl.Name}", (out string e) => Ptx.Narrow(gpu, f, sl, out e));
            Gpu.Launch(narrow, Grid(f.Words(GH), 256), 1, 256, 1, 0, sums, db, Gpu.U(GH));
        }
        finally
        {
            gpu.Return(sums);
        }
    }

    // Each step's input state, T·B × H: h0, then h of steps 0 … T − 2.
    private ulong Shifted(ulong h0, ulong hs, long hStep, int T)
    {
        ulong prev = gpu.Rent(hStep * T + 4);
        Gpu.Copy(h0, prev, hStep);
        if (T > 1) Gpu.Copy(hs, prev + (ulong)hStep, hStep * (T - 1));
        return prev;
    }

    /// <summary>A GRU over T steps on the device.</summary>
    public string GruSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return GruSequenceMixed(f, a, T, x, h0, w, u, bias, biasH, hOut);

        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        var (lane, gx) = GruForward<TN>(a, T, x, h0, w, u, bias, biasH, hOut, 0);
        gpu.Return(gx);
        return lane;
    }

    /// <summary>The LSTM sequence's backward on the device (the forward recomputed; the weights' gradients summed over the steps).</summary>
    public string LstmSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcLast, ulong dx, ulong dh0, ulong dc0, ulong dw, ulong du, ulong db) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return LstmSequenceBackwardMixed(f, a, T, x, h0, c0, w, u, bias, dhOut, dcLast, dx, dh0, dc0, dw, du, db);

        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        long xStep = (long)a.B * a.D * f.Bits / 8, hStep = (long)a.B * a.H * f.Bits / 8;
        // The forward recomputed once keeping every step's gates; per step back only the cell's backward and dh = dg · U (the
        // chain, dh and dc alternating between two buffers); dx, dW, dU, db over all the steps at once (SequenceWeights).
        int GW = a.Gates * a.H;
        long gStep = (long)a.B * GW * f.Bits / 8;
        _ = xStep;
        ulong hs = gpu.Rent(hStep * T + 4), cs = gpu.Rent(hStep * T + 4);
        var rented = new List<ulong> { hs, cs };
        ulong R(long n) { ulong p = gpu.Rent(n); rented.Add(p); return p; }
        try
        {
            ulong gates = R(gStep * T + 4), dgAll = R(gStep * T + 4);
            LstmForward<TN>(a, T, x, h0, c0, w, u, bias, hs, cs, gates);
            ulong dht = R(hStep + 4);
            ulong[] dhb = [R(hStep + 4), R(hStep + 4)], dcb = [R(hStep + 4), R(hStep + 4)];
            ulong dhNext = R(hStep + 4), dcNext = R(hStep + 4);
            Gpu.Clear(dhNext, hStep); Gpu.Copy(dcLast, dcNext, hStep);
            var add = Ops.Find("add");
            string lane = "";
            for (int t = T - 1; t >= 0; t--)
            {
                Map<TN>(add, dhOut + (ulong)(t * hStep), dhNext, 0, dht, (int)((long)a.B * a.H), 0, 0);
                ulong dg = dgAll + (ulong)(t * gStep), dhp = dhb[t & 1], dcp = dcb[t & 1];
                lane = LstmCellBackward<TN>(a, gates + (ulong)(t * gStep), t == 0 ? c0 : cs + (ulong)((t - 1) * hStep), dht, dcNext, dg, dcp);
                MatMulView<TN>(new ProductView(a.B, a.H, GW, false, false), dg, u, dhp);                // dg · U
                dhNext = dhp; dcNext = dcp;
            }

            ulong prev = Shifted(h0, hs, hStep, T);
            rented.Add(prev);
            SequenceWeights<TN>(a, T * a.B, dgAll, x, prev, w, dx, dw, du, db);
            Gpu.Copy(dhNext, dh0, hStep); Gpu.Copy(dcNext, dc0, hStep);
            return lane;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>The GRU sequence's backward on the device: dx, dh0, dW, dU, db, dbh from dh per step (the forward recomputed;
    /// the weights' gradients summed over the steps).</summary>
    public string GruSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh0, ulong dw, ulong du, ulong db, ulong dbh) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return GruSequenceBackwardMixed(f, a, T, x, h0, w, u, bias, biasH, dhOut, dx, dh0, dw, du, db, dbh);

        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        long xStep = (long)a.B * a.D * f.Bits / 8, hStep = (long)a.B * a.H * f.Bits / 8;
        // The forward recomputed once keeping gx (all steps) and every step's gh; per step back only the cell's backward and
        // dh = dgh · U + the cell's direct part (alternating buffers); dx, dW, db from dgx and dU, dbh from dgh over all the
        // steps at once.
        int GW = a.Gates * a.H;
        long gStep = (long)a.B * GW * f.Bits / 8;
        _ = xStep;
        ulong hs = gpu.Rent(hStep * T + 4);
        var rented = new List<ulong> { hs };
        ulong R(long n) { ulong p = gpu.Rent(n); rented.Add(p); return p; }
        try
        {
            ulong ghAll = R(gStep * T + 4), dgxAll = R(gStep * T + 4), dghAll = R(gStep * T + 4);
            var (_, gx) = GruForward<TN>(a, T, x, h0, w, u, bias, biasH, hs, ghAll);
            rented.Add(gx);
            ulong dht = R(hStep + 4), dhCell = R(hStep + 4), dhNext = R(hStep + 4);
            ulong[] dhb = [R(hStep + 4), R(hStep + 4)];
            Gpu.Clear(dhNext, hStep);
            var add = Ops.Find("add");
            string lane = "";
            for (int t = T - 1; t >= 0; t--)
            {
                Map<TN>(add, dhOut + (ulong)(t * hStep), dhNext, 0, dht, (int)((long)a.B * a.H), 0, 0);
                ulong dgx = dgxAll + (ulong)(t * gStep), dgh = dghAll + (ulong)(t * gStep), dhp = dhb[t & 1];
                lane = GruCellBackward<TN>(a, gx + (ulong)(t * gStep), ghAll + (ulong)(t * gStep), t == 0 ? h0 : hs + (ulong)((t - 1) * hStep), dht, dgx, dgh, dhCell);
                MatMulView<TN>(new ProductView(a.B, a.H, GW, false, false), dgh, u, dhp);               // dgh · U
                Map<TN>(add, dhp, dhCell, 0, dhp, (int)((long)a.B * a.H), 0, 0);
                dhNext = dhp;
            }

            ulong prev = Shifted(h0, hs, hStep, T);
            rented.Add(prev);
            SequenceWeights<TN>(a, T * a.B, dgxAll, x, 0, w, dx, dw, 0, db);
            SequenceWeights<TN>(new CellArgs("gru", a.B, a.H, a.H), T * a.B, dghAll, prev, 0, u, 0, du, 0, dbh);
            Gpu.Copy(dhNext, dh0, hStep);
            return lane;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>A GRU step: h' from x, h and W (3H × D), U (3H × H), b (3H), bh (3H).</summary>
    public string GruStep<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f)) return GruStepMixed(f, a, x, h, w, u, bias, biasH, hOut);

        long bytes = f.Words(a.GateValues) * (f.WordBits / 8);
        ulong gx = gpu.Rent(bytes), gh = gpu.Rent(bytes);
        try
        {
            CellGates<TN>(a, x, w, 0, 0, bias, gx);
            CellGates<TN>(new CellArgs("gru", a.B, a.H, a.H), h, u, 0, 0, biasH, gh);
            return GruCell<TN>(a, gx, gh, h, hOut);
        }
        finally
        {
            gpu.Return(gx); gpu.Return(gh);
        }
    }
}
