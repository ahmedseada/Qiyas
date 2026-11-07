// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device's recurrent steps and sequences for a type whose storage is not its lane's (every type but float32 and
// float64; block formats among them): the compositions of GpuOps.Recurrent.cs in the f32 lane on the type's own tensors.
// No tensor of the type is widened in memory: the inputs (x, the states, the weights, the biases, the incoming
// gradients) are read where they are stored by the products (ProductStored), the adds (Ptx.MapMixed) and the cells (a
// mixed cell); the gates, the recurrence's states across the steps and every gradient of them are f32 intermediates;
// the outputs are stored as the type by the kernels that compute them (the products, the cells, the adds, the bias sums)
// where its values are whole bytes and not blocks, else encoded from the lane's values once (a block of a block format
// may span the rows of several threads). A sequence's states stay f32 across its steps (the next step's input).
namespace Adad;

public sealed partial class GpuOps
{
    /// <summary>A tensor of a composition: device pointer, its format, a value offset into it.</summary>
    private readonly record struct Buf(ulong P, Fmt F, long Off = 0)
    {
        public bool None => P == 0;
    }

    private static Fmt F32Fmt => Fmt.Of<Float32>();

    private static Buf Lanes32(ulong p) => new(p, F32Fmt);

    // C (lane storage) = op(A)·op(B), each read as its own format.
    private void Prod(ProductView v, Buf a, Buf b, ulong c) => ProductStored(v, Lane.F32, a.P, a.F, a.Off, b.P, b.F, b.Off, c, null);

    // An output of the type from a product: stored as f by the kernel where it can, else the lane's sums encoded after.
    private void ProdOut(ProductView v, Buf a, Buf b, ulong output, Fmt f)
    {
        if (StoredC(f)) { ProductStored(v, Lane.F32, a.P, a.F, a.Off, b.P, b.F, b.Off, output, f); return; }
        ulong lc = gpu.Rent(v.ValuesC * 4);
        try
        {
            Prod(v, a, b, lc);
            Back(f, Lane.F32, lc, output, v.ValuesC);
        }
        finally { gpu.Return(lc); }
    }

    // o (f32 lanes, or fo) = x + y over n values, each read as its own format (row: y a row vector of `cols`).
    private void AddMixed(Buf x, Buf y, ulong o, long n, int cols = 0, Fmt? fo = null)
    {
        var add = Adad.Ops.Find("add");
        bool row = cols > 0;
        var outFmt = fo ?? F32Fmt;
        var fn = gpu.Cached($"mapmix:add:{x.F.Name}:{y.F.Name}:{outFmt.Name}:{row}", (out string e) => Ptx.MapMixed(gpu, x.F, y.F, outFmt, Lane.F32, add, row, out e));
        if (n > 0) Gpu.Launch(fn, Grid(n, 256), 1, 256, 1, 0, x.P, y.P, o, Gpu.U((int)n), Gpu.U(Math.Max(1, cols)), Gpu.U((int)x.Off), Gpu.U((int)y.Off), Gpu.F(0), Gpu.F(0));
    }

    // A cell with each input read as its own format; its state outputs (h', c'; dc; dh) stored as `o` (null: f32 lanes),
    // mirrored as f32 lanes into m4, m5 when given; the gates' gradients f32 lanes.
    private void CellMixed(CellArgs a, bool backward, Buf g, Buf g2, Buf s, Buf d1, Buf d2, ulong o1, ulong o2, ulong o3, Fmt? o = null, ulong m4 = 0, ulong m5 = 0)
    {
        foreach (var t in new[] { g, g2, s, d1, d2 }) if (t.Off != 0) throw new ArgumentException("a cell's inputs from their first value");
        static Fmt Of(Buf t) => t.None ? F32Fmt : t.F;
        var formats = new Ptx.CellFormats(Of(g), Of(g2), Of(s), Of(d1), Of(d2), o, m4 != 0 || m5 != 0);
        var fn = gpu.Cached($"cell:{a.Kind}:{backward}:mixed:{formats.Name}", (out string e) => Ptx.Cell(gpu, F32Fmt, Lane.F32, a.Kind, backward, out e, formats));
        Gpu.Launch(fn, Grid((long)a.B * a.H, 256), 1, 256, 1, 0, g.P, g2.P, s.P, d1.P, d2.P, o1, o2, o3, m4, m5, Gpu.U(a.B), Gpu.U(a.H));
    }

    // Where a step's state output goes: the type's output itself when the cell stores it (whole bytes, not blocks), else f32
    // lanes to encode after.
    private ulong StateOut(Fmt f, ulong output, long values, List<ulong> rented) => StoredC(f) ? output : RentLanes(rented, values);

    private void StateDone(Fmt f, ulong lanes, ulong output, long values) { if (!StoredC(f)) Narrowed(f, lanes, output, values); }

    // f32 lane values into an output of the type (once): a block format's by blocks.
    private void Narrowed(Fmt f, ulong lanes, ulong output, long count)
    {
        if (count == 0) return;
        if (f.IsBlock) { Back(f, Lane.F32, lanes, output, count); return; }
        var fn = gpu.Cached($"narrow:{f.Name}:f32", (out string e) => Ptx.Narrow(gpu, f, Lane.F32, out e));
        Gpu.Launch(fn, Grid(f.Words(count), 256), 1, 256, 1, 0, lanes, output, Gpu.U((int)count));
    }

    // An output of the type: the column sums of an f32 intermediate (rows × cols), in the sum lane, stored as f.
    private void ColumnSums(Fmt f, ulong x, int rows, int cols, ulong output)
    {
        ulong sums = gpu.Rent((long)cols * 8);
        try
        {
            var sl = LaneOf(Reduce<Float32>("sum", Axis.Cols, x, rows, cols, sums));
            if (f.IsBlock) { Back(f, sl, sums, output, cols); return; }
            var narrow = gpu.Cached($"narrow:{f.Name}:{sl.Name}", (out string e) => Ptx.Narrow(gpu, f, sl, out e));
            Gpu.Launch(narrow, Grid(f.Words(cols), 256), 1, 256, 1, 0, sums, output, Gpu.U(cols));
        }
        finally { gpu.Return(sums); }
    }

    // The gates (f32 lanes, B × GH) = x·Wᵀ (+ h·Uᵀ) + b.
    private void GatesMixed(CellArgs a, Buf x, Buf w, Buf h, Buf u, Buf bias, ulong g)
    {
        int GH = a.Gates * a.H;
        Prod(new ProductView(a.B, GH, a.D, false, true), x, w, g);
        if (!h.None && !u.None)
        {
            ulong gh = gpu.Rent(a.GateValues * 4);
            try
            {
                Prod(new ProductView(a.B, GH, a.H, false, true), h, u, gh);
                Map<Float32>(Adad.Ops.Find("add"), g, gh, 0, g, (int)a.GateValues, 0, 0);
            }
            finally { gpu.Return(gh); }
        }

        AddMixed(Lanes32(g), bias, g, a.GateValues, GH);
    }

    // The gates' backward from dg (f32 lanes, rows × GH): dx = dg·W, dh = dg·U, dU = dgᵀ·h, dW = dgᵀ·x, db = dg's column
    // sums, the outputs of the type (dh: lane storage when lanes, a step's sum still to come).
    private void GatesBackwardMixed(Fmt f, CellArgs a, int rows, ulong dg, Buf x, Buf h, Buf w, Buf u, ulong dx, ulong dhLanes, ulong dw, ulong du, ulong db)
    {
        int GH = a.Gates * a.H;
        var g = Lanes32(dg);
        if (dx != 0) ProdOut(new ProductView(rows, a.D, GH, false, false), g, w, dx, f);
        if (!h.None && !u.None)
        {
            Prod(new ProductView(rows, a.H, GH, false, false), g, u, dhLanes);
            ProdOut(new ProductView(GH, a.H, rows, true, false), g, h, du, f);
        }

        ProdOut(new ProductView(GH, a.D, rows, true, false), g, x, dw, f);
        ColumnSums(f, dg, rows, GH, db);
    }

    // dU (an output of the type) = dgᵀ · each step's input state: h0 for step 0's rows, the f32 states hs of steps
    // 0 … T − 2 for the rest.
    private void ShiftedWeights(Fmt f, CellArgs a, int T, ulong dg, Buf h0, ulong hs, ulong du)
    {
        int GH = a.Gates * a.H;
        long values = (long)GH * a.H;
        ulong acc = gpu.Rent(values * 4), rest = T > 1 ? gpu.Rent(values * 4) : 0;
        try
        {
            Prod(new ProductView(GH, a.H, a.B, true, false), Lanes32(dg), h0, acc);
            if (T > 1)
            {
                Prod(new ProductView(GH, a.H, (T - 1) * a.B, true, false), Lanes32(dg + (ulong)((long)a.B * GH * 4)), Lanes32(hs), rest);
                Map<Float32>(Adad.Ops.Find("add"), acc, rest, 0, acc, (int)values, 0, 0);
            }

            Narrowed(f, acc, du, values);
        }
        finally
        {
            gpu.Return(acc);
            if (rest != 0) gpu.Return(rest);
        }
    }

    private ulong RentLanes(List<ulong> rented, long values) { ulong p = gpu.Rent(Math.Max(1, values) * 4); rented.Add(p); return p; }

    // ---- the steps

    private string LstmStepMixed(Fmt f, CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut)
    {
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong g = RentLanes(rented, a.GateValues), lh = StateOut(f, hOut, bh, rented), lc = StateOut(f, cOut, bh, rented);
            GatesMixed(a, new(x, f), new(w, f), new(h, f), new(u, f), new(bias, f), g);
            CellMixed(a, false, Lanes32(g), default, new(c, f), default, default, lh, lc, 0, StoredC(f) ? f : null);
            StateDone(f, lh, hOut, bh);
            StateDone(f, lc, cOut, bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private string LstmStepBackwardMixed(Fmt f, CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcOut, ulong dx, ulong dh, ulong dc, ulong dw, ulong du, ulong db)
    {
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong g = RentLanes(rented, a.GateValues), dg = RentLanes(rented, a.GateValues), ldc = StateOut(f, dc, bh, rented);
            GatesMixed(a, new(x, f), new(w, f), new(h, f), new(u, f), new(bias, f), g);
            CellMixed(a, true, Lanes32(g), default, new(c, f), new(dhOut, f), new(dcOut, f), dg, ldc, 0, StoredC(f) ? f : null);
            GatesBackwardMixed(f, a, a.B, dg, new(x, f), default, new(w, f), default, dx, 0, dw, 0, db);
            ProdOut(new ProductView(a.B, a.H, a.Gates * a.H, false, false), Lanes32(dg), new(u, f), dh, f);              // dh = dg · U
            ProdOut(new ProductView(a.Gates * a.H, a.H, a.B, true, false), Lanes32(dg), new(h, f), du, f);               // dU = dgᵀ · h
            StateDone(f, ldc, dc, bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private string GruStepMixed(Fmt f, CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut)
    {
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong gx = RentLanes(rented, a.GateValues), gh = RentLanes(rented, a.GateValues), lh = StateOut(f, hOut, bh, rented);
            GatesMixed(a, new(x, f), new(w, f), default, default, new(bias, f), gx);
            GatesMixed(new CellArgs("gru", a.B, a.H, a.H), new(h, f), new(u, f), default, default, new(biasH, f), gh);
            CellMixed(a, false, Lanes32(gx), Lanes32(gh), new(h, f), default, default, lh, 0, 0, StoredC(f) ? f : null);
            StateDone(f, lh, hOut, bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private string GruStepBackwardMixed(Fmt f, CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh, ulong dw, ulong du, ulong db, ulong dbh)
    {
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            var hArgs = new CellArgs("gru", a.B, a.H, a.H);
            ulong gx = RentLanes(rented, a.GateValues), gh = RentLanes(rented, a.GateValues), dgx = RentLanes(rented, a.GateValues), dgh = RentLanes(rented, a.GateValues);
            ulong dhCell = RentLanes(rented, bh), dhGates = RentLanes(rented, bh);
            GatesMixed(a, new(x, f), new(w, f), default, default, new(bias, f), gx);
            GatesMixed(hArgs, new(h, f), new(u, f), default, default, new(biasH, f), gh);
            CellMixed(a, true, Lanes32(gx), Lanes32(gh), new(h, f), new(dhOut, f), default, dgx, dgh, dhCell);
            GatesBackwardMixed(f, a, a.B, dgx, new(x, f), default, new(w, f), default, dx, 0, dw, 0, db);
            // dh = dgh · U (the gates' part, f32) + the cell's direct part, stored once.
            Prod(new ProductView(a.B, a.H, a.Gates * a.H, false, false), Lanes32(dgh), new(u, f), dhGates);
            ProdOut(new ProductView(a.Gates * a.H, a.H, a.B, true, false), Lanes32(dgh), new(h, f), du, f);
            ColumnSums(f, dgh, a.B, a.Gates * a.H, dbh);
            if (StoredC(f)) AddMixed(Lanes32(dhGates), Lanes32(dhCell), dh, bh, 0, f);
            else
            {
                Map<Float32>(Adad.Ops.Find("add"), dhGates, dhCell, 0, dhGates, (int)bh, 0, 0);
                Narrowed(f, dhGates, dh, bh);
            }
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private static string MixedName(Fmt f) => f.IsBlock ? "f32 (blocks in place)" : "f32";

    // ---- the sequences

    // x · Wᵀ + b of every step at once (f32 lanes, T·B × GH).
    private ulong InputGatesMixed(CellArgs a, int T, Buf x, Buf w, Buf bias, List<ulong> rented)
    {
        int GH = a.Gates * a.H, rows = T * a.B;
        ulong gx = RentLanes(rented, (long)rows * GH);
        Prod(new ProductView(rows, GH, a.D, false, true), x, w, gx);
        AddMixed(Lanes32(gx), bias, gx, (long)rows * GH, GH);
        return gx;
    }

    // The LSTM over the steps: the states h, c of every step (f32 lanes) into hs, cs; each step's gates into `gates` when
    // given (kept for the backward). typedOut (hs, cs then the type's hOut, cOut: whole bytes, not blocks): the cell stores
    // each step's h, c as the type and its f32 state into two alternating step buffers.
    private void LstmForwardMixed(Fmt f, CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hs, ulong cs, ulong gates, List<ulong> rented, bool typedOut = false)
    {
        int GH = a.Gates * a.H;
        long gStep = (long)a.B * GH * 4, hStep = (long)a.B * a.H * 4;
        var add = Adad.Ops.Find("add");
        ulong gx = InputGatesMixed(a, T, new(x, f), new(w, f), new(bias, f), rented), gh = RentLanes(rented, (long)a.B * GH), g = gates == 0 ? RentLanes(rented, (long)a.B * GH) : 0;
        long bh = (long)a.B * a.H, typedStep = bh * f.Bits / 8;
        ulong[] hb = typedOut ? [RentLanes(rented, bh), RentLanes(rented, bh)] : [], cb = typedOut ? [RentLanes(rented, bh), RentLanes(rented, bh)] : [];
        Buf hPrev = new(h0, f), cPrev = new(c0, f);
        for (int t = 0; t < T; t++)
        {
            ulong gt = gates == 0 ? g : gates + (ulong)(t * gStep);
            Prod(new ProductView(a.B, GH, a.H, false, true), hPrev, new(u, f), gh);                          // h · Uᵀ
            Map<Float32>(add, gx + (ulong)(t * gStep), gh, 0, gt, a.B * GH, 0, 0);
            if (typedOut)
            {
                ulong ht = hb[t & 1], ct = cb[t & 1];
                CellMixed(a, false, Lanes32(gt), default, cPrev, default, default, hs + (ulong)(t * typedStep), cs + (ulong)(t * typedStep), 0, f, ht, ct);
                hPrev = Lanes32(ht); cPrev = Lanes32(ct);
            }
            else
            {
                ulong ht = hs + (ulong)(t * hStep), ct = cs + (ulong)(t * hStep);
                CellMixed(a, false, Lanes32(gt), default, cPrev, default, default, ht, ct, 0);
                hPrev = Lanes32(ht); cPrev = Lanes32(ct);
            }
        }
    }

    // The GRU over the steps: the states of every step (f32 lanes) into hs; each step's gh into ghAll when given; gx of all
    // the steps returned (rented). typedOut: hs is the type's hOut, each step's h stored as the type by the cell, its f32
    // state into two alternating step buffers.
    private ulong GruForwardMixed(Fmt f, CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hs, ulong ghAll, List<ulong> rented, bool typedOut = false)
    {
        int GH = a.Gates * a.H;
        long gStep = (long)a.B * GH * 4, hStep = (long)a.B * a.H * 4;
        ulong gx = InputGatesMixed(a, T, new(x, f), new(w, f), new(bias, f), rented), gh = ghAll == 0 ? RentLanes(rented, (long)a.B * GH) : 0;
        long bh = (long)a.B * a.H, typedStep = bh * f.Bits / 8;
        ulong[] hb = typedOut ? [RentLanes(rented, bh), RentLanes(rented, bh)] : [];
        Buf hPrev = new(h0, f);
        for (int t = 0; t < T; t++)
        {
            ulong ght = ghAll == 0 ? gh : ghAll + (ulong)(t * gStep);
            Prod(new ProductView(a.B, GH, a.H, false, true), hPrev, new(u, f), ght);                         // h · Uᵀ
            AddMixed(Lanes32(ght), new(biasH, f), ght, (long)a.B * GH, GH);
            ulong ht = typedOut ? hb[t & 1] : hs + (ulong)(t * hStep);
            if (typedOut) CellMixed(a, false, Lanes32(gx + (ulong)(t * gStep)), Lanes32(ght), hPrev, default, default, hs + (ulong)(t * typedStep), 0, 0, f, ht);
            else CellMixed(a, false, Lanes32(gx + (ulong)(t * gStep)), Lanes32(ght), hPrev, default, default, ht, 0, 0);
            hPrev = Lanes32(ht);
        }

        return gx;
    }

    private string LstmSequenceMixed(Fmt f, CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut)
    {
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            if (StoredC(f)) { LstmForwardMixed(f, a, T, x, h0, c0, w, u, bias, hOut, cOut, 0, rented, typedOut: true); return MixedName(f); }
            ulong hs = RentLanes(rented, T * bh), cs = RentLanes(rented, T * bh);
            LstmForwardMixed(f, a, T, x, h0, c0, w, u, bias, hs, cs, 0, rented);
            Narrowed(f, hs, hOut, T * bh);
            Narrowed(f, cs, cOut, T * bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private string GruSequenceMixed(Fmt f, CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut)
    {
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            if (StoredC(f)) { GruForwardMixed(f, a, T, x, h0, w, u, bias, biasH, hOut, 0, rented, typedOut: true); return MixedName(f); }
            ulong hs = RentLanes(rented, T * bh);
            GruForwardMixed(f, a, T, x, h0, w, u, bias, biasH, hs, 0, rented);
            Narrowed(f, hs, hOut, T * bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private string LstmSequenceBackwardMixed(Fmt f, CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcLast, ulong dx, ulong dh0, ulong dc0, ulong dw, ulong du, ulong db)
    {
        var rented = new List<ulong>();
        try
        {
            int GW = a.Gates * a.H;
            long bh = (long)a.B * a.H, gStep = (long)a.B * GW * 4, hStep = bh * 4;
            ulong hs = RentLanes(rented, T * bh), cs = RentLanes(rented, T * bh), gates = RentLanes(rented, T * (long)a.B * GW), dgAll = RentLanes(rented, T * (long)a.B * GW);
            LstmForwardMixed(f, a, T, x, h0, c0, w, u, bias, hs, cs, gates, rented);
            ulong dht = RentLanes(rented, bh), zero = RentLanes(rented, bh);
            ulong[] dhb = [RentLanes(rented, bh), RentLanes(rented, bh)], dcb = [RentLanes(rented, bh), RentLanes(rented, bh)];
            Gpu.Clear(zero, hStep);
            Buf dhNext = Lanes32(zero), dcNext = new(dcLast, f);
            for (int t = T - 1; t >= 0; t--)
            {
                AddMixed(new(dhOut, f, t * bh), dhNext, dht, bh);
                ulong dg = dgAll + (ulong)(t * gStep), dhp = dhb[t & 1], dcp = dcb[t & 1];
                CellMixed(a, true, Lanes32(gates + (ulong)(t * gStep)), default, t == 0 ? new(c0, f) : Lanes32(cs + (ulong)((t - 1) * hStep)), Lanes32(dht), dcNext, dg, dcp, 0);
                Prod(new ProductView(a.B, a.H, GW, false, false), Lanes32(dg), new(u, f), dhp);                  // dg · U
                dhNext = Lanes32(dhp); dcNext = Lanes32(dcp);
            }

            int rows = T * a.B;
            ProdOut(new ProductView(rows, a.D, GW, false, false), Lanes32(dgAll), new(w, f), dx, f);
            ProdOut(new ProductView(GW, a.D, rows, true, false), Lanes32(dgAll), new(x, f), dw, f);
            ShiftedWeights(f, a, T, dgAll, new(h0, f), hs, du);
            ColumnSums(f, dgAll, rows, GW, db);
            Narrowed(f, dhNext.P, dh0, bh);
            Narrowed(f, dcNext.P, dc0, bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }

    private string GruSequenceBackwardMixed(Fmt f, CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh0, ulong dw, ulong du, ulong db, ulong dbh)
    {
        var rented = new List<ulong>();
        try
        {
            int GW = a.Gates * a.H;
            long bh = (long)a.B * a.H, gStep = (long)a.B * GW * 4, hStep = bh * 4;
            ulong hs = RentLanes(rented, T * bh), ghAll = RentLanes(rented, T * (long)a.B * GW), dgxAll = RentLanes(rented, T * (long)a.B * GW), dghAll = RentLanes(rented, T * (long)a.B * GW);
            ulong gx = GruForwardMixed(f, a, T, x, h0, w, u, bias, biasH, hs, ghAll, rented);
            ulong dht = RentLanes(rented, bh), dhCell = RentLanes(rented, bh), zero = RentLanes(rented, bh);
            ulong[] dhb = [RentLanes(rented, bh), RentLanes(rented, bh)];
            Gpu.Clear(zero, hStep);
            Buf dhNext = Lanes32(zero);
            var add = Adad.Ops.Find("add");
            for (int t = T - 1; t >= 0; t--)
            {
                AddMixed(new(dhOut, f, t * bh), dhNext, dht, bh);
                ulong dgx = dgxAll + (ulong)(t * gStep), dgh = dghAll + (ulong)(t * gStep), dhp = dhb[t & 1];
                CellMixed(a, true, Lanes32(gx + (ulong)(t * gStep)), Lanes32(ghAll + (ulong)(t * gStep)), t == 0 ? new(h0, f) : Lanes32(hs + (ulong)((t - 1) * hStep)), Lanes32(dht), default, dgx, dgh, dhCell);
                Prod(new ProductView(a.B, a.H, GW, false, false), Lanes32(dgh), new(u, f), dhp);                 // dgh · U
                Map<Float32>(add, dhp, dhCell, 0, dhp, (int)bh, 0, 0);
                dhNext = Lanes32(dhp);
            }

            int rows = T * a.B;
            ProdOut(new ProductView(rows, a.D, GW, false, false), Lanes32(dgxAll), new(w, f), dx, f);
            ProdOut(new ProductView(GW, a.D, rows, true, false), Lanes32(dgxAll), new(x, f), dw, f);
            ColumnSums(f, dgxAll, rows, GW, db);
            ShiftedWeights(f, new CellArgs("gru", a.B, a.H, a.H), T, dghAll, new(h0, f), hs, du);
            ColumnSums(f, dghAll, rows, GW, dbh);
            Narrowed(f, dhNext.P, dh0, bh);
            return MixedName(f);
        }
        finally { foreach (var p in rented) gpu.Return(p); }
    }
}
