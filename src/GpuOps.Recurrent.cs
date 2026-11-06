// The device's recurrent cells and steps (Recurrent.cs): the cells by one kernel each, the gates by the products on
// views and the broadcast add, the steps and backward passes as the same compositions on device buffers.
namespace Qiyas;

public sealed partial class GpuOps
{
    private void LaunchCell<TN>(CellArgs a, bool backward, ulong g, ulong g2, ulong state, ulong d1, ulong d2, ulong o1, ulong o2, ulong o3) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        var lane = RowLane(f);
        if (f.IsBlock)
        {
            // A block format's cell: the inputs decoded into float32 storage, the cell there, the outputs encoded by blocks.
            long gates = a.GateValues, bh = (long)a.B * a.H;
            long o1n = backward ? gates : bh, o2n = backward && a.Gates == 3 ? gates : bh;
            var r = new List<ulong>();
            ulong In(ulong x, long count) => x == 0 ? 0 : Lanes(f, Lane.F32, x, count, r);
            ulong Out(ulong x, long count) { if (x == 0) return 0; ulong l = gpu.Rent(count * 4); r.Add(l); return l; }
            try
            {
                ulong lg = In(g, gates), lg2 = In(g2, gates), ls = In(state, bh), ld1 = In(d1, bh), ld2 = In(d2, bh), lo1 = Out(o1, o1n), lo2 = Out(o2, o2n), lo3 = Out(o3, bh);
                LaunchCell<Float32>(a, backward, lg, lg2, ls, ld1, ld2, lo1, lo2, lo3);
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
    public string LstmCell<TN>(CellArgs a, ulong g, ulong c, ulong hOut, ulong cOut) where TN : struct, IQiyasNumber<TN>
    {
        LaunchCell<TN>(a, false, g, 0, c, 0, 0, hOut, cOut, 0);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>GRU: h' from gx, gh (B × 3H) and h.</summary>
    public string GruCell<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong hOut) where TN : struct, IQiyasNumber<TN>
    {
        LaunchCell<TN>(a, false, gx, gh, h, 0, 0, hOut, 0, 0);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>LSTM backward: dg (B × 4H) and dc from dh', dc' (and the gates, c).</summary>
    public string LstmCellBackward<TN>(CellArgs a, ulong g, ulong c, ulong dh, ulong dcNext, ulong dg, ulong dc) where TN : struct, IQiyasNumber<TN>
    {
        LaunchCell<TN>(a, true, g, 0, c, dh, dcNext, dg, dc, 0);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>GRU backward: dgx, dgh (B × 3H) and dh from dh' (and gx, gh, h).</summary>
    public string GruCellBackward<TN>(CellArgs a, ulong gx, ulong gh, ulong h, ulong dhOut, ulong dgx, ulong dgh, ulong dh) where TN : struct, IQiyasNumber<TN>
    {
        LaunchCell<TN>(a, true, gx, gh, h, dhOut, 0, dgx, dgh, dh);
        return RowLane(Fmt.Of<TN>()).Name;
    }

    /// <summary>The gates g = x·Wᵀ (+ h·Uᵀ) + b (B × GH) as stored values of the type.</summary>
    public void CellGates<TN>(CellArgs a, ulong x, ulong w, ulong h, ulong u, ulong bias, ulong g) where TN : struct, IQiyasNumber<TN>
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
                Map<TN>(Qiyas.Ops.Find("add"), g, gh, 0, g, (int)a.GateValues, 0, 0);
            }
            finally
            {
                gpu.Return(gh);
            }
        }

        MapRow<TN>(Qiyas.Ops.Find("add"), g, bias, g, a.B, GH);
    }

    private bool OwnLane(Fmt f) => f.Is("float32") || f.Is("float64");

    /// <summary>An LSTM step: (h', c') from x, h, c and W (4H × D), U (4H × H), b (4H); a narrow float's in its lane's storage.</summary>
    public string LstmStep<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f))
        {
            var lane = RowLane(f);
            var rented = new List<ulong>();
            try
            {
                long bh = (long)a.B * a.H, GH = a.Gates * a.H;
                ulong lh = gpu.Rent(bh * lane.Size), lc = gpu.Rent(bh * lane.Size);
                rented.Add(lh); rented.Add(lc);
                ulong lx = ToLane<TN>(x, (long)a.B * a.D, rented), lhh = ToLane<TN>(h, bh, rented), lcc = ToLane<TN>(c, bh, rented), lw = ToLane<TN>(w, GH * a.D, rented), lu = ToLane<TN>(u, GH * a.H, rented), lb = ToLane<TN>(bias, GH, rented);
                string name = lane.Wide ? LstmStep<Float64>(a, lx, lhh, lcc, lw, lu, lb, lh, lc) : LstmStep<Float32>(a, lx, lhh, lcc, lw, lu, lb, lh, lc);
                FromLane<TN>(lh, hOut, bh); FromLane<TN>(lc, cOut, bh);
                return name;
            }
            finally
            {
                foreach (var p in rented) gpu.Return(p);
            }
        }

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

    private void GatesBackward<TN>(CellArgs a, ulong dg, ulong x, ulong h, ulong w, ulong u, ulong dx, ulong dh, ulong dw, ulong du, ulong db) where TN : struct, IQiyasNumber<TN>
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

    /// <summary>An LSTM step's backward: dx, dh, dc, dW, dU, db from dh', dc'; a narrow float's in its lane's storage.</summary>
    public string LstmStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong c, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcOut, ulong dx, ulong dh, ulong dc, ulong dw, ulong du, ulong db) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f))
        {
            var lane = RowLane(f);
            var rented = new List<ulong>();
            try
            {
                long bh = (long)a.B * a.H, bd = (long)a.B * a.D, GH = a.Gates * a.H;
                ulong R(long n) { ulong p = gpu.Rent(n * lane.Size); rented.Add(p); return p; }
                ulong ldx = R(bd), ldh = R(bh), ldc = R(bh), ldw = R(GH * a.D), ldu = R(GH * a.H), ldb = R(GH);
                ulong lx = ToLane<TN>(x, bd, rented), lhh = ToLane<TN>(h, bh, rented), lcc = ToLane<TN>(c, bh, rented), lw = ToLane<TN>(w, GH * a.D, rented), lu = ToLane<TN>(u, GH * a.H, rented), lb = ToLane<TN>(bias, GH, rented);
                ulong ldho = ToLane<TN>(dhOut, bh, rented), ldco = ToLane<TN>(dcOut, bh, rented);
                string name = lane.Wide ? LstmStepBackward<Float64>(a, lx, lhh, lcc, lw, lu, lb, ldho, ldco, ldx, ldh, ldc, ldw, ldu, ldb) : LstmStepBackward<Float32>(a, lx, lhh, lcc, lw, lu, lb, ldho, ldco, ldx, ldh, ldc, ldw, ldu, ldb);
                FromLane<TN>(ldx, dx, bd); FromLane<TN>(ldh, dh, bh); FromLane<TN>(ldc, dc, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH);
                return name;
            }
            finally
            {
                foreach (var p in rented) gpu.Return(p);
            }
        }

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

    /// <summary>A GRU step's backward: dx, dh, dW, dU, db, dbh from dh'; a narrow float's in its lane's storage.</summary>
    public string GruStepBackward<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong dhOut, ulong dx, ulong dh, ulong dw, ulong du, ulong db, ulong dbh) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f))
        {
            var lane = RowLane(f);
            var rented = new List<ulong>();
            try
            {
                long bh = (long)a.B * a.H, bd = (long)a.B * a.D, GH = a.Gates * a.H;
                ulong R(long n) { ulong p = gpu.Rent(n * lane.Size); rented.Add(p); return p; }
                ulong ldx = R(bd), ldh = R(bh), ldw = R(GH * a.D), ldu = R(GH * a.H), ldb = R(GH), ldbh = R(GH);
                ulong lx = ToLane<TN>(x, bd, rented), lhh = ToLane<TN>(h, bh, rented), lw = ToLane<TN>(w, GH * a.D, rented), lu = ToLane<TN>(u, GH * a.H, rented), lb = ToLane<TN>(bias, GH, rented), lbh = ToLane<TN>(biasH, GH, rented), ldho = ToLane<TN>(dhOut, bh, rented);
                string name = lane.Wide ? GruStepBackward<Float64>(a, lx, lhh, lw, lu, lb, lbh, ldho, ldx, ldh, ldw, ldu, ldb, ldbh) : GruStepBackward<Float32>(a, lx, lhh, lw, lu, lb, lbh, ldho, ldx, ldh, ldw, ldu, ldb, ldbh);
                FromLane<TN>(ldx, dx, bd); FromLane<TN>(ldh, dh, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH); FromLane<TN>(ldbh, dbh, GH);
                return name;
            }
            finally
            {
                foreach (var p in rented) gpu.Return(p);
            }
        }

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
            Map<TN>(Qiyas.Ops.Find("add"), dhGates, dhCell, 0, dh, (int)((long)a.B * a.H), 0, 0);
            return lane;
        }
        finally
        {
            gpu.Return(gx); gpu.Return(gh); gpu.Return(dgx); gpu.Return(dgh); gpu.Return(dhCell); gpu.Return(dhGates);
        }
    }

    /// <summary>An LSTM over T steps on the device: x (T × B × D), h0, c0 → h, c (T × B × H each).</summary>
    public string LstmSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong hOut, ulong cOut) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f) && !f.IsBlock)
        {
            var ln = RowLane(f);
            var rented = new List<ulong>();
            try
            {
                long bd = (long)a.B * a.D, bh = (long)a.B * a.H, GH = a.Gates * a.H;
                ulong lh = gpu.Rent(T * bh * ln.Size), lc = gpu.Rent(T * bh * ln.Size);
                rented.Add(lh); rented.Add(lc);
                ulong lx = ToLane<TN>(x, T * bd, rented), lh0 = ToLane<TN>(h0, bh, rented), lc0 = ToLane<TN>(c0, bh, rented), lw = ToLane<TN>(w, GH * a.D, rented), lu = ToLane<TN>(u, GH * a.H, rented), lb = ToLane<TN>(bias, GH, rented);
                string name = ln.Wide ? LstmSequence<Float64>(a, T, lx, lh0, lc0, lw, lu, lb, lh, lc) : LstmSequence<Float32>(a, T, lx, lh0, lc0, lw, lu, lb, lh, lc);
                FromLane<TN>(lh, hOut, T * bh); FromLane<TN>(lc, cOut, T * bh);
                return name;
            }
            finally
            {
                foreach (var p in rented) gpu.Return(p);
            }
        }

        long xBytes = f.Words((long)a.B * a.D) * (f.WordBits / 8), hBytes = f.Words((long)a.B * a.H) * (f.WordBits / 8);
        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        ulong hPrev = h0, cPrev = c0;
        string lane = "";
        for (int t = 0; t < T; t++)
        {
            ulong xt = x + (ulong)(t * (long)a.B * a.D * f.Bits / 8), ht = hOut + (ulong)(t * (long)a.B * a.H * f.Bits / 8), ct = cOut + (ulong)(t * (long)a.B * a.H * f.Bits / 8);
            lane = LstmStep<TN>(a, xt, hPrev, cPrev, w, u, bias, ht, ct);
            hPrev = ht; cPrev = ct;
        }

        _ = xBytes; _ = hBytes;
        return lane;
    }

    /// <summary>A GRU over T steps on the device.</summary>
    public string GruSequence<TN>(CellArgs a, int T, ulong x, ulong h0, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        ulong hPrev = h0;
        string lane = "";
        for (int t = 0; t < T; t++)
        {
            ulong xt = x + (ulong)(t * (long)a.B * a.D * f.Bits / 8), ht = hOut + (ulong)(t * (long)a.B * a.H * f.Bits / 8);
            lane = GruStep<TN>(a, xt, hPrev, w, u, bias, biasH, ht);
            hPrev = ht;
        }

        return lane;
    }

    /// <summary>The LSTM sequence's backward on the device (the forward recomputed; the weights' gradients summed over the steps).</summary>
    public string LstmSequenceBackward<TN>(CellArgs a, int T, ulong x, ulong h0, ulong c0, ulong w, ulong u, ulong bias, ulong dhOut, ulong dcLast, ulong dx, ulong dh0, ulong dc0, ulong dw, ulong du, ulong db) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f) && !f.IsBlock)
        {
            var ln = RowLane(f);
            var rented2 = new List<ulong>();
            try
            {
                long bd = (long)a.B * a.D, bh = (long)a.B * a.H, GH = a.Gates * a.H;
                ulong R2(long n) { ulong p = gpu.Rent(n * ln.Size); rented2.Add(p); return p; }
                ulong ldx = R2(T * bd), ldh0 = R2(bh), ldc0 = R2(bh), ldw = R2(GH * a.D), ldu = R2(GH * a.H), ldb = R2(GH);
                ulong lx = ToLane<TN>(x, T * bd, rented2), lh0 = ToLane<TN>(h0, bh, rented2), lc0 = ToLane<TN>(c0, bh, rented2), lw = ToLane<TN>(w, GH * a.D, rented2), lu = ToLane<TN>(u, GH * a.H, rented2), lb = ToLane<TN>(bias, GH, rented2), ldho = ToLane<TN>(dhOut, T * bh, rented2), ldcl = ToLane<TN>(dcLast, bh, rented2);
                string name = ln.Wide ? LstmSequenceBackward<Float64>(a, T, lx, lh0, lc0, lw, lu, lb, ldho, ldcl, ldx, ldh0, ldc0, ldw, ldu, ldb) : LstmSequenceBackward<Float32>(a, T, lx, lh0, lc0, lw, lu, lb, ldho, ldcl, ldx, ldh0, ldc0, ldw, ldu, ldb);
                FromLane<TN>(ldx, dx, T * bd); FromLane<TN>(ldh0, dh0, bh); FromLane<TN>(ldc0, dc0, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH);
                return name;
            }
            finally
            {
                foreach (var p in rented2) gpu.Return(p);
            }
        }

        if (f.IsBlock || ((long)a.B * a.D * f.Bits % 8 != 0) || ((long)a.B * a.H * f.Bits % 8 != 0)) throw new NotSupportedException($"{f.Name}: a sequence's steps must end on a byte");
        long xStep = (long)a.B * a.D * f.Bits / 8, hStep = (long)a.B * a.H * f.Bits / 8;
        long dwBytes = f.Words((long)a.Gates * a.H * a.D) * (f.WordBits / 8), duBytes = f.Words((long)a.Gates * a.H * a.H) * (f.WordBits / 8), dbBytes = f.Words(a.Gates * a.H) * (f.WordBits / 8);
        ulong hs = gpu.Rent(hStep * T + 4), cs = gpu.Rent(hStep * T + 4);
        var rented = new List<ulong> { hs, cs };
        ulong R(long n) { ulong p = gpu.Rent(n); rented.Add(p); return p; }
        try
        {
            ulong hPrev = h0, cPrev = c0;
            for (int t = 0; t < T; t++)
            {
                ulong ht = hs + (ulong)(t * hStep), ct = cs + (ulong)(t * hStep);
                LstmStep<TN>(a, x + (ulong)(t * xStep), hPrev, cPrev, w, u, bias, ht, ct);
                hPrev = ht; cPrev = ct;
            }

            ulong dhNext = R(hStep + 4), dcNext = R(hStep + 4), dht = R(hStep + 4), dhp = R(hStep + 4), dcp = R(hStep + 4), dwt = R(dwBytes), dut = R(duBytes), dbt = R(dbBytes);
            Gpu.Clear(dhNext, hStep); Gpu.Copy(dcLast, dcNext, hStep);
            Gpu.Clear(dw, dwBytes); Gpu.Clear(du, duBytes); Gpu.Clear(db, dbBytes);
            var add = Qiyas.Ops.Find("add");
            string lane = "";
            for (int t = T - 1; t >= 0; t--)
            {
                Map<TN>(add, dhOut + (ulong)(t * hStep), dhNext, 0, dht, (int)((long)a.B * a.H), 0, 0);
                lane = LstmStepBackward<TN>(a, x + (ulong)(t * xStep), t == 0 ? h0 : hs + (ulong)((t - 1) * hStep), t == 0 ? c0 : cs + (ulong)((t - 1) * hStep), w, u, bias, dht, dcNext, dx + (ulong)(t * xStep), dhp, dcp, dwt, dut, dbt);
                Map<TN>(add, dw, dwt, 0, dw, (int)((long)a.Gates * a.H * a.D), 0, 0);
                Map<TN>(add, du, dut, 0, du, (int)((long)a.Gates * a.H * a.H), 0, 0);
                Map<TN>(add, db, dbt, 0, db, a.Gates * a.H, 0, 0);
                Gpu.Copy(dhp, dhNext, hStep); Gpu.Copy(dcp, dcNext, hStep);
            }

            Gpu.Copy(dhNext, dh0, hStep); Gpu.Copy(dcNext, dc0, hStep);
            return lane;
        }
        finally
        {
            foreach (var p in rented) gpu.Return(p);
        }
    }

    /// <summary>A GRU step: h' from x, h and W (3H × D), U (3H × H), b (3H), bh (3H); a narrow float's in its lane's storage.</summary>
    public string GruStep<TN>(CellArgs a, ulong x, ulong h, ulong w, ulong u, ulong bias, ulong biasH, ulong hOut) where TN : struct, IQiyasNumber<TN>
    {
        var f = Fmt.Of<TN>();
        if (!OwnLane(f))
        {
            var lane = RowLane(f);
            var rented = new List<ulong>();
            try
            {
                long bh = (long)a.B * a.H, GH = a.Gates * a.H;
                ulong lh = gpu.Rent(bh * lane.Size);
                rented.Add(lh);
                ulong lx = ToLane<TN>(x, (long)a.B * a.D, rented), lhh = ToLane<TN>(h, bh, rented), lw = ToLane<TN>(w, GH * a.D, rented), lu = ToLane<TN>(u, GH * a.H, rented), lb = ToLane<TN>(bias, GH, rented), lbh = ToLane<TN>(biasH, GH, rented);
                string name = lane.Wide ? GruStep<Float64>(a, lx, lhh, lw, lu, lb, lbh, lh) : GruStep<Float32>(a, lx, lhh, lw, lu, lb, lbh, lh);
                FromLane<TN>(lh, hOut, bh);
                return name;
            }
            finally
            {
                foreach (var p in rented) gpu.Return(p);
            }
        }

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
