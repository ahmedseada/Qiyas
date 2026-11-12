// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention over tensors laid through views (AttentionArgs.QRows, KRows, VRows, ORows: HeadRows, a strided view of a head's
// rows), written once for every device over the device contract: each input taken a head at a time by the device's own
// strided copy (ILayouts' strided view: any offset and strides), the form run on them, and each output moved into its view
// the same way (an output's view covers its tensor's values once, so the copy is a gather in the view's own order). The
// copies move the values as the type stores them: a row of whole storage units (any type of 8 bits or more; a block format
// whose rows, strides and offset are whole blocks) is moved as it is, a block format's bytes as bytes; only a block format
// whose rows share blocks is re-encoded by its blocks, as every layout's output of it is.
// A device whose kernels read the views where they are replaces the copies in its own entry; until then this is every
// form's route.
namespace Adad.Abstraction;

/// <summary>Where a tensor's head rows lie (a strided view, as the strided layout takes): row i of head h at Offset + (h /
/// Heads)·BatchStride + (h % Heads)·HeadStride + i·RowStride values, its D values after one another; Heads the heads of a
/// batch (the tensor's heads a whole multiple of it). An output's view covers its tensor's values once: its strides a
/// dense order of its batches, heads, rows and D values, from offset 0.</summary>
public sealed record HeadRows(int Heads, long HeadStride, long RowStride, long BatchStride = 0, long Offset = 0)
{
    /// <summary>Heads after one another, each its rows after one another (no view).</summary>
    public static HeadRows HeadsFirst(int heads, int rows, int d) => new(Math.Max(1, heads), (long)rows * d, d, 0);

    /// <summary>[batches, rows, heads, d]: each row's heads side by side, the layout a layer's projections come in (heads
    /// the tensor's, a whole multiple of batches).</summary>
    public static HeadRows StepsFirst(int heads, int rows, int d, int batches = 1)
    {
        int hb = Math.Max(1, heads / Math.Max(1, batches));
        return new(hb, d, (long)hb * d, (long)rows * hb * d);
    }

    /// <summary>The value row i of head h starts at.</summary>
    public long At(int h, int i) => Offset + h / Heads * BatchStride + h % Heads * HeadStride + i * RowStride;

    /// <summary>The last value the view reaches, plus one (heads × rows of d).</summary>
    public long Extent(int heads, int rows, int d) => heads == 0 || rows == 0 || d == 0 ? Offset : At(heads - 1, rows - 1) + d;

    public void Validate(string what, int heads, int rows, int d, bool output)
    {
        if (Heads < 1 || heads % Heads != 0 || HeadStride < 0 || RowStride < 0 || BatchStride < 0 || Offset < 0)
            throw new ArgumentException($"attention: {what}'s view of {Heads} heads a batch, strides {BatchStride}, {HeadStride}, {RowStride} from {Offset}, over {heads} heads");
        if (output && !Dense(heads, rows, d)) throw new ArgumentException($"attention: {what}'s view does not cover its {heads} × {rows} × {d} values once (an output's view is a dense order)");
    }

    // The dimensions of a view of heads × rows of d: (extent, stride) of the batches, the heads of a batch, the rows.
    private (int N, long S)[] Dims(int heads, int rows) => [(heads / Heads, BatchStride), (Heads, HeadStride), (rows, RowStride)];

    /// <summary>Whether the view covers heads × rows × d values once from 0: its dimensions of more than one value, by stride,
    /// each the product of the ones inside it (d's stride 1).</summary>
    public bool Dense(int heads, int rows, int d)
    {
        if (Offset != 0) return false;
        long next = d;
        foreach (var (n, s) in Dims(heads, rows).Where(x => x.N > 1).OrderBy(x => x.S))
        {
            if (s != next) return false;
            next *= n;
        }

        return true;
    }

    /// <summary>The strided layout's parameters that gather the view's values into heads × rows of d after one another
    /// (gather: true), or, for a dense view, the values after one another into the view's order (gather: false). Units: the
    /// copy moves `unit` stored units a value (bytes as a whole-unit row's: w per row), strides counted in them.</summary>
    public int[] Strided(int heads, int rows, int d, int w, Func<long, long> units, bool gather)
    {
        var dims = Dims(heads, rows);
        if (gather) return [dims[0].N, dims[1].N, dims[2].N, w, Int(units(dims[0].S)), Int(units(dims[1].S)), Int(units(dims[2].S)), 1, Int(units(Offset))];
        // The view's memory order, outermost first; each dimension read from heads × rows of d after one another.
        long[] contiguous = [units((long)Heads * rows * d), units((long)rows * d), units(d)];
        var order = new[] { 0, 1, 2 }.OrderByDescending(i => dims[i].N > 1 ? dims[i].S : -1).ThenBy(i => i).ToArray();
        return [dims[order[0]].N, dims[order[1]].N, dims[order[2]].N, w, Int(contiguous[order[0]]), Int(contiguous[order[1]]), Int(contiguous[order[2]]), 1, 0];
    }

    private static int Int(long v) => v <= int.MaxValue ? (int)v : throw new NotSupportedException($"attention: a view's stride of {v} values past 2^31");
}

public static class AttentionLayout
{
    // A tensor of heads × rows of d through its view into rented storage (toHeads), or from storage after one another into
    // its view (back).
    private static void Move<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, HeadRows view, TBuffer source, TBuffer destination, int heads, int rows, int d, bool toHeads)
        where TN : struct, IAdad<TN>
    {
        int n = heads * rows * d;
        if (n == 0) return;
        // Rows of whole blocks (and strides and offset whole blocks): their bytes moved as bytes.
        bool bytes = WholeUnits<TN>(view, d) && TN.Granularity > 1;
        int w = bytes ? (int)TN.Bytes(d) : d;
        Func<long, long> units = bytes ? static x => TN.Bytes(x) : static x => x;
        n = heads * rows * w;
        // The values the copy reads: a gather's, the view's extent (in whole storage units of the source, which holds
        // them); a dense view's from storage, its n.
        int g = bytes ? 1 : TN.Granularity;
        long extent = toHeads ? (units(view.Extent(heads, rows, d)) + g - 1) / g * g : n;
        if (extent > int.MaxValue) throw new NotSupportedException($"attention: a view reaching {extent} values past 2^31");
        var args = new LayoutArgs { Name = "strided", X = [], P = view.Strided(heads, rows, d, w, units, toHeads), Output = [], Count = n, InputCount = (int)extent };
        if (bytes) device.Layout<UInt8>(args, source, default, destination, default);
        else device.Layout<TN>(args, source, default, destination, default);
    }

    /// <summary>Whether a view's rows of d values of TN, its strides and its offset are whole storage units (so rows move
    /// without sharing one): every type of 8 bits or more, a block format when they are whole blocks.</summary>
    public static bool WholeUnits<TN>(HeadRows? view, int d) where TN : struct, IAdad<TN>
    {
        int g = TN.Granularity;
        return d % g == 0 && (view is null || (view.HeadStride % g == 0 && view.RowStride % g == 0 && view.BatchStride % g == 0 && view.Offset % g == 0));
    }

    /// <summary>Whether every view of a's tensors moves its rows as whole storage units (WholeUnits).</summary>
    public static bool WholeUnits<TN>(AttentionArgs a) where TN : struct, IAdad<TN> =>
        WholeUnits<TN>(a.QRows, a.D) && WholeUnits<TN>(a.KRows, a.D) && WholeUnits<TN>(a.VRows, a.D) && WholeUnits<TN>(a.ORows, a.D);

    private static TBuffer Rent<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, long values) where TN : struct, IAdad<TN> => device.Rent(TN.Bytes(values) + 16);

    // A tensor through its view a head at a time, in a rented buffer (its own buffer when it has no view).
    private static (TBuffer Buffer, bool Rented) In<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, HeadRows? view, TBuffer source, int heads, int rows, int d) where TN : struct, IAdad<TN>
    {
        if (view is null) return (source, false);
        var rented = Rent<TBuffer, TIndex, TN>(device, (long)heads * rows * d);
        Move<TBuffer, TIndex, TN>(device, view, source, rented, heads, rows, d, true);
        return (rented, true);
    }

    // The buffer for an output a head at a time (its own when it has no view).
    private static (TBuffer Buffer, bool Rented) Out<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, HeadRows? view, TBuffer output, long values) where TN : struct, IAdad<TN> =>
        view is null ? (output, false) : (Rent<TBuffer, TIndex, TN>(device, values), true);

    private static void Back<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, HeadRows? view, (TBuffer Buffer, bool Rented) heads, TBuffer output, int count, int rows, int d) where TN : struct, IAdad<TN>
    {
        if (view is not null) Move<TBuffer, TIndex, TN>(device, view, heads.Buffer, output, count, rows, d, false);
    }

    private static void Return<TBuffer, TIndex>(IDevice<TBuffer, TIndex> device, params (TBuffer Buffer, bool Rented)[] buffers)
    {
        foreach (var (buffer, rented) in buffers) if (rented) device.Return(buffer);
    }

    /// <summary>O of a form (run: the form over tensors a head at a time) for Q, K, V and O through their views.</summary>
    public static string Forward<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer o,
        Func<IDevice<TBuffer, TIndex>, AttentionArgs, TBuffer, TBuffer, TBuffer, TBuffer, string> run) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        int hk = a.KvHeadCount;
        var qh = In<TBuffer, TIndex, TN>(device, a.QRows, q, a.Heads, a.Lq, a.D);
        var kh = In<TBuffer, TIndex, TN>(device, a.KRows, k, hk, a.Lk, a.D);
        var vh = In<TBuffer, TIndex, TN>(device, a.VRows, v, hk, a.Lk, a.D);
        var oh = Out<TBuffer, TIndex, TN>(device, a.ORows, o, a.QValues);
        try
        {
            string lane = run(device, a.HeadMajor, qh.Buffer, kh.Buffer, vh.Buffer, oh.Buffer);
            Back<TBuffer, TIndex, TN>(device, a.ORows, oh, o, a.Heads, a.Lq, a.D);
            return lane;
        }
        finally { Return(device, qh, kh, vh, oh); }
    }

    /// <summary>O of a form over caches (run: the form over Q and O a head at a time) for Q and O through their views.</summary>
    public static string Queries<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, AttentionArgs a, TBuffer q, TBuffer o,
        Func<IDevice<TBuffer, TIndex>, AttentionArgs, TBuffer, TBuffer, string> run) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        if (a.KRows is not null || a.VRows is not null) throw new ArgumentException("attention over a cache: K and V are the cache's rows (no views)");
        var qh = In<TBuffer, TIndex, TN>(device, a.QRows, q, a.Heads, a.Lq, a.D);
        var oh = Out<TBuffer, TIndex, TN>(device, a.ORows, o, a.QValues);
        try
        {
            string lane = run(device, a.HeadMajor, qh.Buffer, oh.Buffer);
            Back<TBuffer, TIndex, TN>(device, a.ORows, oh, o, a.Heads, a.Lq, a.D);
            return lane;
        }
        finally { Return(device, qh, oh); }
    }

    /// <summary>dQ, dK and dV (through Q's, K's and V's views) from dO (through O's), Q, K and V through theirs.</summary>
    public static string Backward<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer dO, TBuffer dq, TBuffer dk, TBuffer dv)
        where TN : struct, IFloatAdad<TN>
    {
        a.ValidateBackward();
        int hk = a.KvHeadCount;
        var qh = In<TBuffer, TIndex, TN>(device, a.QRows, q, a.Heads, a.Lq, a.D);
        var kh = In<TBuffer, TIndex, TN>(device, a.KRows, k, hk, a.Lk, a.D);
        var vh = In<TBuffer, TIndex, TN>(device, a.VRows, v, hk, a.Lk, a.D);
        var gh = In<TBuffer, TIndex, TN>(device, a.ORows, dO, a.Heads, a.Lq, a.D);
        var dqh = Out<TBuffer, TIndex, TN>(device, a.QRows, dq, a.QValues);
        var dkh = Out<TBuffer, TIndex, TN>(device, a.KRows, dk, a.KValues);
        var dvh = Out<TBuffer, TIndex, TN>(device, a.VRows, dv, a.KValues);
        try
        {
            string lane = device.AttentionBackward<TN>(a.HeadMajor, qh.Buffer, kh.Buffer, vh.Buffer, gh.Buffer, dqh.Buffer, dkh.Buffer, dvh.Buffer);
            Back<TBuffer, TIndex, TN>(device, a.QRows, dqh, dq, a.Heads, a.Lq, a.D);
            Back<TBuffer, TIndex, TN>(device, a.KRows, dkh, dk, hk, a.Lk, a.D);
            Back<TBuffer, TIndex, TN>(device, a.VRows, dvh, dv, hk, a.Lk, a.D);
            return lane;
        }
        finally { Return(device, qh, kh, vh, gh, dqh, dkh, dvh); }
    }

    /// <summary>A tensor's heads × rows of d through its view (null: as it is), a head at a time, on the host (the checks'
    /// references).</summary>
    public static double[] ToHeads(double[] x, HeadRows? view, int heads, int rows, int d)
    {
        if (view is null) return (double[])x.Clone();
        var y = new double[(long)heads * rows * d];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < rows; i++)
                Array.Copy(x, view.At(h, i), y, ((long)h * rows + i) * d, d);
        return y;
    }

    /// <summary>Heads × rows of d a head at a time into a tensor through its view (the values outside it kept), on the host.</summary>
    public static double[] FromHeads(double[] x, HeadRows? view, int heads, int rows, int d, double[]? into = null)
    {
        if (view is null) return (double[])x.Clone();
        var y = into ?? new double[view.Extent(heads, rows, d)];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < rows; i++)
                Array.Copy(x, ((long)h * rows + i) * d, y, view.At(h, i), d);
        return y;
    }
}
