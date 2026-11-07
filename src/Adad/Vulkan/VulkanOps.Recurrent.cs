// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The recurrent cells, steps and sequences on Vulkan (Recurrent.cs, Kernels.Recurrent.cs): CUDA's compositions in the
// type's float lane (f32, f64 for float64) on the type's own tensors. No tensor of the type is widened in memory: the
// inputs (x, the states, the weights, the biases, the incoming gradients) are read where they are stored by the products
// (Product: a format per operand), the adds (the mixed add) and the cells (each input its own format); the gates, the
// recurrence's states across the steps and every gradient of them are lane intermediates; the outputs are lane values,
// the type's own storage for float32 and float64, else stored as the type once after. The gates (CellGates) are the
// type's own operations on any type. The block formats are refused (no Vulkan block encoder yet).
using Adad.Abstraction;

namespace Adad.Vulkan;

public sealed partial class VulkanOps
{
    /// <summary>A tensor of a composition: device pointer, its format, a value offset into it.</summary>
    private readonly record struct Buf(ulong P, Fmt F, long Off = 0)
    {
        public bool None => P == 0;
    }

    private static Buf Lanes(Lane lane, ulong p) => new(p, VkKernels.LaneFormat(lane));

    private Fmt RecurrentFormat<TN>(string what) where TN : struct, IAdad<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock) throw Missing($"{what} of {f.Name} (a block format)");
        return f;
    }

    private ulong RentLanes(Lane lane, List<ulong> rented, long values)
    {
        ulong p = vk.Rent(Math.Max(1, values) * lane.Size);
        rented.Add(p);
        return p;
    }

    // C (lane values) = op(A)·op(B), each read as its own format.
    private void Prod(Lane lane, ProductView v, Buf a, Buf b, ulong c) => Product(v, lane, a.P, a.F, a.Off, b.P, b.F, b.Off, c);

    // An output of the type from a product: in place for the type's own lane, else lane storage stored as the type once.
    private void ProdOut(Fmt f, Lane lane, ProductView v, Buf a, Buf b, ulong output)
    {
        if (Own(f)) { Prod(lane, v, a, b, output); return; }
        ulong lc = vk.Rent(Math.Max(1, v.ValuesC) * lane.Size);
        try
        {
            Prod(lane, v, a, b, lc);
            Back(f, lane, lc, output, v.ValuesC);
        }
        finally { vk.Return(lc); }
    }

    // o (lane values) = x + y over n values, each read as its own format (cols > 0: y a row vector of cols).
    private void AddMixed(Lane lane, Buf x, Buf y, ulong o, long n, int cols = 0)
    {
        if (n == 0) return;
        if (x.Off > uint.MaxValue || y.Off > uint.MaxValue) throw new NotSupportedException("vulkan: an add's operand past 2^32 values");
        bool row = cols > 0;
        var fn = vk.Cached($"add_mixed:{x.F.Name}:{y.F.Name}:{lane.Name}:{row}", () => VkKernels.AddMixed(vk, x.F, y.F, lane, row));
        vk.Launch(fn, Groups(n, VkKernels.Threads), 1, x.P, y.P, o, Vulkan.U(n), Vulkan.U(Math.Max(1, cols)), Vulkan.U(x.Off), Vulkan.U(y.Off));
    }

    // o = x + y on lane storage.
    private void LaneAdd(Lane lane, ulong x, ulong y, ulong o, long n)
    {
        var add = Ops.Find("add");
        if (lane.Wide) Map<Float64>(add, x, y, 0, o, (int)n, 0, 0); else Map<Float32>(add, x, y, 0, o, (int)n, 0, 0);
    }

    // A cell with each input read as its own format (none: 0), its outputs lane values.
    private void CellMixed(Lane lane, CellArgs a, bool backward, Buf g, Buf g2, Buf s, Buf d1, Buf d2, ulong o1, ulong o2, ulong o3)
    {
        foreach (var t in new[] { g, g2, s, d1, d2 }) if (t.Off != 0) throw new ArgumentException("a cell's inputs from their first value");
        var lf = VkKernels.LaneFormat(lane);
        Fmt Of(Buf t) => t.None ? lf : t.F;
        var formats = new CellFormats(Of(g), Of(g2), Of(s), Of(d1), Of(d2));
        long count = (long)a.B * a.H;
        if (count == 0) return;
        var fn = vk.Cached($"cell:{a.Kind}:{backward}:{lane.Name}:{formats.Name}", () => VkKernels.Cell(vk, lane, a.Kind, backward, formats));
        vk.Launch(fn, Groups(count, VkKernels.Threads), 1, g.P, g2.P, s.P, d1.P, d2.P, o1, o2, o3, Vulkan.U(a.B), Vulkan.U(a.H));
    }

    // Where an output goes: the type's output itself in its own lane, else lane storage to store after (StateDone).
    private ulong StateOut(Fmt f, Lane lane, ulong output, long values, List<ulong> rented) => Own(f) ? output : RentLanes(lane, rented, values);

    private void StateDone(Fmt f, Lane lane, ulong lanes, ulong output, long values)
    {
        if (!Own(f) && values > 0) Back(f, lane, lanes, output, values);
    }

    // Lane values into an output of the type (a copy in the type's own lane).
    private void Narrowed(Fmt f, Lane lane, ulong lanes, ulong output, long count)
    {
        if (count > 0) Back(f, lane, lanes, output, count);
    }

    // An output of the type: the column sums of a lane intermediate (rows × cols), in the sum lane, stored as f.
    private void ColumnSums(Fmt f, Lane lane, ulong x, int rows, int cols, ulong output)
    {
        if (cols == 0) return;
        ulong sums = vk.Rent((long)cols * 8);
        try
        {
            string name = lane.Wide ? Reduce<Float64>("sum", Axis.Cols, x, rows, cols, sums) : Reduce<Float32>("sum", Axis.Cols, x, rows, cols, sums);
            Back(f, name == "f64" ? Lane.F64 : Lane.F32, sums, output, cols);
        }
        finally { vk.Return(sums); }
    }

    // ---- the cells (the type's tensors in, the type's outputs)

    public string LstmCell<TN>(CellArgs a, ulong g, ulong c, ulong hOut, ulong cOut) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("lstm cell");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong lh = StateOut(f, lane, hOut, bh, rented), lc = StateOut(f, lane, cOut, bh, rented);
            CellMixed(lane, a, false, new(g, f), default, new(c, f), default, default, lh, lc, 0);
            StateDone(f, lane, lh, hOut, bh);
            StateDone(f, lane, lc, cOut, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    public string LstmCellBackward<TN>(CellArgs a, ulong g, ulong c, ulong dh, ulong dcNext, ulong dg, ulong dc) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("lstm cell backward");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong ldg = StateOut(f, lane, dg, a.GateValues, rented), ldc = StateOut(f, lane, dc, bh, rented);
            CellMixed(lane, a, true, new(g, f), default, new(c, f), new(dh, f), new(dcNext, f), ldg, ldc, 0);
            StateDone(f, lane, ldg, dg, a.GateValues);
            StateDone(f, lane, ldc, dc, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    public string GruCell<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong hOut) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("gru cell");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong lh = StateOut(f, lane, hOut, bh, rented);
            CellMixed(lane, a, false, new(gx, f), new(gh, f), new(h, f), default, default, lh, 0, 0);
            StateDone(f, lane, lh, hOut, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    public string GruCellBackward<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong dhOut, ulong dgx, ulong dgh, ulong dh) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("gru cell backward");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong lx = StateOut(f, lane, dgx, a.GateValues, rented), lg = StateOut(f, lane, dgh, a.GateValues, rented), ld = StateOut(f, lane, dh, bh, rented);
            CellMixed(lane, a, true, new(gx, f), new(gh, f), new(h, f), new(dhOut, f), default, lx, lg, ld);
            StateDone(f, lane, lx, dgx, a.GateValues);
            StateDone(f, lane, lg, dgh, a.GateValues);
            StateDone(f, lane, ld, dh, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>The gates g = x·Wᵀ (+ h·Uᵀ) + b (B × GH) as stored values of the type: the type's own products and adds.</summary>
    public void CellGates<TN>(CellArgs a, ulong x, ulong w, ulong h, ulong u, ulong bias, ulong g) where TN : struct, IAdad<TN>
    {
        var f = RecurrentFormat<TN>("recurrent gates");
        int GH = a.Gates * a.H;
        MatMulView<TN>(new ProductView(a.B, GH, a.D, false, true), x, w, g);
        if (h != 0 && u != 0)
        {
            ulong gh = vk.Rent(Math.Max(1, f.Words(a.GateValues)) * (f.WordBits / 8));
            try
            {
                MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), h, u, gh);
                Map<TN>(Ops.Find("add"), g, gh, 0, g, (int)a.GateValues, 0, 0);
            }
            finally { vk.Return(gh); }
        }

        MapRow<TN>(Ops.Find("add"), g, bias, g, a.B, GH);
    }

    // ---- the steps

    // The gates (lane values, B × GH) = x·Wᵀ (+ h·Uᵀ) + b.
    private void GatesMixed(Lane lane, CellArgs a, Buf x, Buf w, Buf h, Buf u, Buf bias, ulong g)
    {
        int GH = a.Gates * a.H;
        Prod(lane, new ProductView(a.B, GH, a.D, false, true), x, w, g);
        if (!h.None && !u.None)
        {
            ulong gh = vk.Rent(Math.Max(1, a.GateValues) * lane.Size);
            try
            {
                Prod(lane, new ProductView(a.B, GH, a.H, false, true), h, u, gh);
                LaneAdd(lane, g, gh, g, a.GateValues);
            }
            finally { vk.Return(gh); }
        }

        AddMixed(lane, Lanes(lane, g), bias, g, a.GateValues, GH);
    }

    // The gates' backward from dg (lane values, rows × GH): dx = dg·W, dh = dg·U (lane values), dU = dgᵀ·h, dW = dgᵀ·x,
    // db = dg's column sums; the outputs of the type.
    private void GatesBackwardMixed(Fmt f, Lane lane, CellArgs a, int rows, ulong dg, Buf x, Buf h, Buf w, Buf u, ulong dx, ulong dhLanes, ulong dw, ulong du, ulong db)
    {
        int GH = a.Gates * a.H;
        var g = Lanes(lane, dg);
        if (dx != 0) ProdOut(f, lane, new ProductView(rows, a.D, GH, false, false), g, w, dx);
        if (!h.None && !u.None)
        {
            Prod(lane, new ProductView(rows, a.H, GH, false, false), g, u, dhLanes);
            ProdOut(f, lane, new ProductView(GH, a.H, rows, true, false), g, h, du);
        }

        ProdOut(f, lane, new ProductView(GH, a.D, rows, true, false), g, x, dw);
        ColumnSums(f, lane, dg, rows, GH, db);
    }

    /// <summary>An LSTM step: (h', c') from x, h, c and W (4H × D), U (4H × H), b (4H).</summary>
    public string LstmStep<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("lstm step");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong g = RentLanes(lane, rented, a.GateValues), lh = StateOut(f, lane, hOut, bh, rented), lc = StateOut(f, lane, cOut, bh, rented);
            GatesMixed(lane, a, new(x, f), new(w, f), new(h, f), new(u, f), new(bias, f), g);
            CellMixed(lane, a, false, Lanes(lane, g), default, new(c, f), default, default, lh, lc, 0);
            StateDone(f, lane, lh, hOut, bh);
            StateDone(f, lane, lc, cOut, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>An LSTM step's backward: dx, dh, dc, dW, dU, db from dh', dc'.</summary>
    public string LstmStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcOut, ulong dx, ulong dh, ulong dc, ulong dw, ulong du, ulong db)
        where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("lstm step backward");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong g = RentLanes(lane, rented, a.GateValues), dg = RentLanes(lane, rented, a.GateValues), ldc = StateOut(f, lane, dc, bh, rented);
            GatesMixed(lane, a, new(x, f), new(w, f), new(h, f), new(u, f), new(bias, f), g);
            CellMixed(lane, a, true, Lanes(lane, g), default, new(c, f), new(dhOut, f), new(dcOut, f), dg, ldc, 0);
            GatesBackwardMixed(f, lane, a, a.B, dg, new(x, f), default, new(w, f), default, dx, 0, dw, 0, db);
            ProdOut(f, lane, new ProductView(a.B, a.H, a.Gates * a.H, false, false), Lanes(lane, dg), new(u, f), dh);           // dh = dg · U
            ProdOut(f, lane, new ProductView(a.Gates * a.H, a.H, a.B, true, false), Lanes(lane, dg), new(h, f), du);            // dU = dgᵀ · h
            StateDone(f, lane, ldc, dc, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>A GRU step: h' from x, h and W (3H × D), U (3H × H), b (3H), bh (3H).</summary>
    public string GruStep<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("gru step");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            ulong gx = RentLanes(lane, rented, a.GateValues), gh = RentLanes(lane, rented, a.GateValues), lh = StateOut(f, lane, hOut, bh, rented);
            GatesMixed(lane, a, new(x, f), new(w, f), default, default, new(bias, f), gx);
            GatesMixed(lane, new CellArgs("gru", a.B, a.H, a.H), new(h, f), new(u, f), default, default, new(biasH, f), gh);
            CellMixed(lane, a, false, Lanes(lane, gx), Lanes(lane, gh), new(h, f), default, default, lh, 0, 0);
            StateDone(f, lane, lh, hOut, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>A GRU step's backward: dx, dh, dW, dU, db, dbh from dh'.</summary>
    public string GruStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh, ulong dw, ulong du, ulong db, ulong dbh)
        where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("gru step backward");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long bh = (long)a.B * a.H;
            var hArgs = new CellArgs("gru", a.B, a.H, a.H);
            ulong gx = RentLanes(lane, rented, a.GateValues), gh = RentLanes(lane, rented, a.GateValues), dgx = RentLanes(lane, rented, a.GateValues), dgh = RentLanes(lane, rented, a.GateValues);
            ulong dhCell = RentLanes(lane, rented, bh), dhGates = RentLanes(lane, rented, bh), ldh = StateOut(f, lane, dh, bh, rented);
            GatesMixed(lane, a, new(x, f), new(w, f), default, default, new(bias, f), gx);
            GatesMixed(lane, hArgs, new(h, f), new(u, f), default, default, new(biasH, f), gh);
            CellMixed(lane, a, true, Lanes(lane, gx), Lanes(lane, gh), new(h, f), new(dhOut, f), default, dgx, dgh, dhCell);
            GatesBackwardMixed(f, lane, a, a.B, dgx, new(x, f), default, new(w, f), default, dx, 0, dw, 0, db);
            // dh = dgh · U (the gates' part) + the cell's direct part, stored once.
            Prod(lane, new ProductView(a.B, a.H, a.Gates * a.H, false, false), Lanes(lane, dgh), new(u, f), dhGates);
            ProdOut(f, lane, new ProductView(a.Gates * a.H, a.H, a.B, true, false), Lanes(lane, dgh), new(h, f), du);
            ColumnSums(f, lane, dgh, a.B, a.Gates * a.H, dbh);
            LaneAdd(lane, dhGates, dhCell, ldh, bh);
            StateDone(f, lane, ldh, dh, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    // ---- the sequences

    // x · Wᵀ + b of every step at once (lane values, T·B × GH).
    private ulong InputGatesMixed(Lane lane, CellArgs a, int T, Buf x, Buf w, Buf bias, List<ulong> rented)
    {
        int GH = a.Gates * a.H, rows = T * a.B;
        ulong gx = RentLanes(lane, rented, (long)rows * GH);
        Prod(lane, new ProductView(rows, GH, a.D, false, true), x, w, gx);
        AddMixed(lane, Lanes(lane, gx), bias, gx, (long)rows * GH, GH);
        return gx;
    }

    // The LSTM over the steps: the states h, c of every step (lane values) into hs, cs; each step's gates into `gates` when
    // given (kept for the backward).
    private void LstmForwardMixed(Fmt f, Lane lane, CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hs, ulong cs, ulong gates, List<ulong> rented)
    {
        int GH = a.Gates * a.H;
        long gStep = (long)a.B * GH * lane.Size, hStep = (long)a.B * a.H * lane.Size;
        ulong gx = InputGatesMixed(lane, a, T, new(x, f), new(w, f), new(bias, f), rented), gh = RentLanes(lane, rented, (long)a.B * GH), g = gates == 0 ? RentLanes(lane, rented, (long)a.B * GH) : 0;
        Buf hPrev = new(h0, f), cPrev = new(c0, f);
        for (int t = 0; t < T; t++)
        {
            ulong gt = gates == 0 ? g : gates + (ulong)(t * gStep), ht = hs + (ulong)(t * hStep), ct = cs + (ulong)(t * hStep);
            Prod(lane, new ProductView(a.B, GH, a.H, false, true), hPrev, new(u, f), gh);                    // h · Uᵀ
            LaneAdd(lane, gx + (ulong)(t * gStep), gh, gt, (long)a.B * GH);
            CellMixed(lane, a, false, Lanes(lane, gt), default, cPrev, default, default, ht, ct, 0);
            hPrev = Lanes(lane, ht); cPrev = Lanes(lane, ct);
        }
    }

    // The GRU over the steps: the states of every step (lane values) into hs; each step's gh into ghAll when given; gx of all
    // the steps returned (rented).
    private ulong GruForwardMixed(Fmt f, Lane lane, CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hs, ulong ghAll, List<ulong> rented)
    {
        int GH = a.Gates * a.H;
        long gStep = (long)a.B * GH * lane.Size, hStep = (long)a.B * a.H * lane.Size;
        ulong gx = InputGatesMixed(lane, a, T, new(x, f), new(w, f), new(bias, f), rented), gh = ghAll == 0 ? RentLanes(lane, rented, (long)a.B * GH) : 0;
        Buf hPrev = new(h0, f);
        for (int t = 0; t < T; t++)
        {
            ulong ght = ghAll == 0 ? gh : ghAll + (ulong)(t * gStep), ht = hs + (ulong)(t * hStep);
            Prod(lane, new ProductView(a.B, GH, a.H, false, true), hPrev, new(u, f), ght);                   // h · Uᵀ
            AddMixed(lane, Lanes(lane, ght), new(biasH, f), ght, (long)a.B * GH, GH);
            CellMixed(lane, a, false, Lanes(lane, gx + (ulong)(t * gStep)), Lanes(lane, ght), hPrev, default, default, ht, 0, 0);
            hPrev = Lanes(lane, ht);
        }

        return gx;
    }

    // dU (an output of the type) = dgᵀ · each step's input state: h0 for step 0's rows, the lane states hs of steps
    // 0 … T − 2 for the rest.
    private void ShiftedWeights(Fmt f, Lane lane, CellArgs a, int T, ulong dg, Buf h0, ulong hs, ulong du)
    {
        int GH = a.Gates * a.H;
        long values = (long)GH * a.H;
        ulong acc = vk.Rent(Math.Max(1, values) * lane.Size), rest = T > 1 ? vk.Rent(Math.Max(1, values) * lane.Size) : 0;
        try
        {
            Prod(lane, new ProductView(GH, a.H, a.B, true, false), Lanes(lane, dg), h0, acc);
            if (T > 1)
            {
                Prod(lane, new ProductView(GH, a.H, (T - 1) * a.B, true, false), Lanes(lane, dg + (ulong)((long)a.B * GH * lane.Size)), Lanes(lane, hs), rest);
                LaneAdd(lane, acc, rest, acc, values);
            }

            Narrowed(f, lane, acc, du, values);
        }
        finally
        {
            vk.Return(acc);
            if (rest != 0) vk.Return(rest);
        }
    }

    /// <summary>An LSTM over T steps: x (T × B × D), h0, c0 → h, c (T × B × H each).</summary>
    public string LstmSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("lstm sequence");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long values = T * (long)a.B * a.H;
            ulong hs = StateOut(f, lane, hOut, values, rented), cs = StateOut(f, lane, cOut, values, rented);
            LstmForwardMixed(f, lane, a, T, x, h0, c0, w, u, bias, hs, cs, 0, rented);
            StateDone(f, lane, hs, hOut, values);
            StateDone(f, lane, cs, cOut, values);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>A GRU over T steps: x (T × B × D), h0 → h (T × B × H).</summary>
    public string GruSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("gru sequence");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            long values = T * (long)a.B * a.H;
            ulong hs = StateOut(f, lane, hOut, values, rented);
            GruForwardMixed(f, lane, a, T, x, h0, w, u, bias, biasH, hs, 0, rented);
            StateDone(f, lane, hs, hOut, values);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>The LSTM sequence's backward (the forward recomputed keeping every step's gates; per step back the cell's
    /// backward and dh = dg · U; dx, dW, dU, db over all the steps at once).</summary>
    public string LstmSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcLast, ulong dx, ulong dh0, ulong dc0, ulong dw, ulong du, ulong db)
        where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("lstm sequence backward");
        if (T < 1) throw new ArgumentException("a sequence of at least one step");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            int GW = a.Gates * a.H;
            long bh = (long)a.B * a.H, gStep = (long)a.B * GW * lane.Size, hStep = bh * lane.Size;
            ulong hs = RentLanes(lane, rented, T * bh), cs = RentLanes(lane, rented, T * bh), gates = RentLanes(lane, rented, T * (long)a.B * GW), dgAll = RentLanes(lane, rented, T * (long)a.B * GW);
            LstmForwardMixed(f, lane, a, T, x, h0, c0, w, u, bias, hs, cs, gates, rented);
            ulong dht = RentLanes(lane, rented, bh), zero = RentLanes(lane, rented, bh);
            ulong[] dhb = [RentLanes(lane, rented, bh), RentLanes(lane, rented, bh)], dcb = [RentLanes(lane, rented, bh), RentLanes(lane, rented, bh)];
            vk.Clear(zero, Math.Max(1, bh) * lane.Size);
            Buf dhNext = Lanes(lane, zero), dcNext = new(dcLast, f);
            for (int t = T - 1; t >= 0; t--)
            {
                AddMixed(lane, new(dhOut, f, t * bh), dhNext, dht, bh);
                ulong dg = dgAll + (ulong)(t * gStep), dhp = dhb[t & 1], dcp = dcb[t & 1];
                CellMixed(lane, a, true, Lanes(lane, gates + (ulong)(t * gStep)), default, t == 0 ? new(c0, f) : Lanes(lane, cs + (ulong)((t - 1) * hStep)), Lanes(lane, dht), dcNext, dg, dcp, 0);
                Prod(lane, new ProductView(a.B, a.H, GW, false, false), Lanes(lane, dg), new(u, f), dhp);         // dg · U
                dhNext = Lanes(lane, dhp); dcNext = Lanes(lane, dcp);
            }

            int rows = T * a.B;
            ProdOut(f, lane, new ProductView(rows, a.D, GW, false, false), Lanes(lane, dgAll), new(w, f), dx);
            ProdOut(f, lane, new ProductView(GW, a.D, rows, true, false), Lanes(lane, dgAll), new(x, f), dw);
            ShiftedWeights(f, lane, a, T, dgAll, new(h0, f), hs, du);
            ColumnSums(f, lane, dgAll, rows, GW, db);
            Narrowed(f, lane, dhNext.P, dh0, bh);
            Narrowed(f, lane, dcNext.P, dc0, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }

    /// <summary>The GRU sequence's backward: dx, dh0, dW, dU, db, dbh from dh per step (the forward recomputed keeping gx and
    /// every step's gh; per step back the cell's backward and dh = dgh · U + the cell's direct part).</summary>
    public string GruSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh0, ulong dw, ulong du, ulong db, ulong dbh)
        where TN : struct, IFloatAdad<TN>
    {
        var f = RecurrentFormat<TN>("gru sequence backward");
        if (T < 1) throw new ArgumentException("a sequence of at least one step");
        var lane = RowLane(f);
        var rented = new List<ulong>();
        try
        {
            int GW = a.Gates * a.H;
            long bh = (long)a.B * a.H, gStep = (long)a.B * GW * lane.Size, hStep = bh * lane.Size;
            ulong hs = RentLanes(lane, rented, T * bh), ghAll = RentLanes(lane, rented, T * (long)a.B * GW), dgxAll = RentLanes(lane, rented, T * (long)a.B * GW), dghAll = RentLanes(lane, rented, T * (long)a.B * GW);
            ulong gx = GruForwardMixed(f, lane, a, T, x, h0, w, u, bias, biasH, hs, ghAll, rented);
            ulong dht = RentLanes(lane, rented, bh), dhCell = RentLanes(lane, rented, bh), zero = RentLanes(lane, rented, bh);
            ulong[] dhb = [RentLanes(lane, rented, bh), RentLanes(lane, rented, bh)];
            vk.Clear(zero, Math.Max(1, bh) * lane.Size);
            Buf dhNext = Lanes(lane, zero);
            for (int t = T - 1; t >= 0; t--)
            {
                AddMixed(lane, new(dhOut, f, t * bh), dhNext, dht, bh);
                ulong dgx = dgxAll + (ulong)(t * gStep), dgh = dghAll + (ulong)(t * gStep), dhp = dhb[t & 1];
                CellMixed(lane, a, true, Lanes(lane, gx + (ulong)(t * gStep)), Lanes(lane, ghAll + (ulong)(t * gStep)), t == 0 ? new(h0, f) : Lanes(lane, hs + (ulong)((t - 1) * hStep)), Lanes(lane, dht), default, dgx, dgh, dhCell);
                Prod(lane, new ProductView(a.B, a.H, GW, false, false), Lanes(lane, dgh), new(u, f), dhp);        // dgh · U
                LaneAdd(lane, dhp, dhCell, dhp, bh);
                dhNext = Lanes(lane, dhp);
            }

            int rows = T * a.B;
            ProdOut(f, lane, new ProductView(rows, a.D, GW, false, false), Lanes(lane, dgxAll), new(w, f), dx);
            ProdOut(f, lane, new ProductView(GW, a.D, rows, true, false), Lanes(lane, dgxAll), new(x, f), dw);
            ColumnSums(f, lane, dgxAll, rows, GW, db);
            ShiftedWeights(f, lane, new CellArgs("gru", a.B, a.H, a.H), T, dghAll, new(h0, f), hs, du);
            ColumnSums(f, lane, dghAll, rows, GW, dbh);
            Narrowed(f, lane, dhNext.P, dh0, bh);
            return lane.Name;
        }
        finally { foreach (var p in rented) vk.Return(p); }
    }
}
