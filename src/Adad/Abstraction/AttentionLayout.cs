// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Attention over tensors laid a step at a time (AttentionArgs.StepsFirst: [batch, steps, heads · d], the layout a layer's
// projections come in), written once for every device over the device contract: Q, K and V (and dO) taken a head at a
// time by the device's own strided copies (ILayouts' strided view), the form run on them, and the outputs moved back the
// same way. The copies move the values as the type stores them: a row of whole storage units (any type of 8 bits or more;
// a block format whose row is whole blocks) is moved as it is, a block format's bytes as bytes; only a block format whose
// rows share blocks is re-encoded by its blocks, as every layout's output of it is.
// A device whose kernels read the strided layout where it is stored replaces the copies in its own entry; until then
// this is every form's route.
namespace Adad.Abstraction;

public static class AttentionLayout
{
    // count values of [batch, steps, heads, d] (toHeads) or [batch, heads, steps, d] as the other order: a strided view.
    private static void Move<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, TBuffer source, TBuffer destination, int batch, int steps, int heads, int d, bool toHeads)
        where TN : struct, IAdad<TN>
    {
        int n = batch * steps * heads * d;
        if (n == 0) return;
        // A row of whole blocks: its bytes moved as bytes (a row of d values is w bytes of the uint8 view).
        bool bytes = WholeUnits<TN>(d) && TN.Granularity > 1;
        int w = bytes ? (int)TN.Bytes(d) : d;
        n = batch * steps * heads * w;
        int[] p = toHeads ? [batch, heads, steps, w, steps * heads * w, w, heads * w, 1, 0] : [batch, steps, heads, w, heads * steps * w, w, steps * w, 1, 0];
        var args = new LayoutArgs { Name = "strided", X = [], P = p, Output = [], Count = n, InputCount = n };
        if (bytes) device.Layout<UInt8>(args, source, default, destination, default);
        else device.Layout<TN>(args, source, default, destination, default);
    }

    /// <summary>Whether a row of d values of TN is whole storage units (so rows move without sharing one): every type of 8
    /// bits or more, a block format when d is whole blocks.</summary>
    public static bool WholeUnits<TN>(int d) where TN : struct, IAdad<TN> => d % TN.Granularity == 0;

    private static TBuffer Rent<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, long values) where TN : struct, IAdad<TN> => device.Rent(TN.Bytes(values) + 16);

    // A tensor of the layout a head at a time, in a rented buffer.
    private static TBuffer ToHeads<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, TBuffer source, int batch, int steps, int heads, int d) where TN : struct, IAdad<TN>
    {
        var rented = Rent<TBuffer, TIndex, TN>(device, (long)batch * steps * heads * d);
        Move<TBuffer, TIndex, TN>(device, source, rented, batch, steps, heads, d, true);
        return rented;
    }

    /// <summary>O of a form (run: the form over tensors a head at a time) for Q, K, V and O a step at a time.</summary>
    public static string Forward<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer o,
        Func<IDevice<TBuffer, TIndex>, AttentionArgs, TBuffer, TBuffer, TBuffer, TBuffer, string> run) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        int b = a.BatchCount, hq = a.Heads / b, hk = a.KvHeadCount / b;
        TBuffer qh = ToHeads<TBuffer, TIndex, TN>(device, q, b, a.Lq, hq, a.D), kh = ToHeads<TBuffer, TIndex, TN>(device, k, b, a.Lk, hk, a.D),
            vh = ToHeads<TBuffer, TIndex, TN>(device, v, b, a.Lk, hk, a.D), oh = Rent<TBuffer, TIndex, TN>(device, a.QValues);
        try
        {
            string lane = run(device, a with { StepsFirst = false }, qh, kh, vh, oh);
            Move<TBuffer, TIndex, TN>(device, oh, o, b, a.Lq, hq, a.D, false);
            return lane;
        }
        finally
        {
            device.Return(qh); device.Return(kh); device.Return(vh); device.Return(oh);
        }
    }

    /// <summary>O of a form over caches (run: the form over Q and O a head at a time) for Q and O a step at a time.</summary>
    public static string Queries<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, AttentionArgs a, TBuffer q, TBuffer o,
        Func<IDevice<TBuffer, TIndex>, AttentionArgs, TBuffer, TBuffer, string> run) where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        int b = a.BatchCount, hq = a.Heads / b;
        TBuffer qh = ToHeads<TBuffer, TIndex, TN>(device, q, b, a.Lq, hq, a.D), oh = Rent<TBuffer, TIndex, TN>(device, a.QValues);
        try
        {
            string lane = run(device, a with { StepsFirst = false }, qh, oh);
            Move<TBuffer, TIndex, TN>(device, oh, o, b, a.Lq, hq, a.D, false);
            return lane;
        }
        finally
        {
            device.Return(qh); device.Return(oh);
        }
    }

    /// <summary>dQ, dK and dV (laid as Q, K and V) from dO (laid as O), all a step at a time.</summary>
    public static string Backward<TBuffer, TIndex, TN>(IDevice<TBuffer, TIndex> device, AttentionArgs a, TBuffer q, TBuffer k, TBuffer v, TBuffer dO, TBuffer dq, TBuffer dk, TBuffer dv)
        where TN : struct, IFloatAdad<TN>
    {
        a.Validate();
        int b = a.BatchCount, hq = a.Heads / b, hk = a.KvHeadCount / b;
        TBuffer qh = ToHeads<TBuffer, TIndex, TN>(device, q, b, a.Lq, hq, a.D), kh = ToHeads<TBuffer, TIndex, TN>(device, k, b, a.Lk, hk, a.D),
            vh = ToHeads<TBuffer, TIndex, TN>(device, v, b, a.Lk, hk, a.D), gh = ToHeads<TBuffer, TIndex, TN>(device, dO, b, a.Lq, hq, a.D);
        TBuffer dqh = Rent<TBuffer, TIndex, TN>(device, a.QValues), dkh = Rent<TBuffer, TIndex, TN>(device, a.KValues), dvh = Rent<TBuffer, TIndex, TN>(device, a.KValues);
        try
        {
            string lane = device.AttentionBackward<TN>(a with { StepsFirst = false }, qh, kh, vh, gh, dqh, dkh, dvh);
            Move<TBuffer, TIndex, TN>(device, dqh, dq, b, a.Lq, hq, a.D, false);
            Move<TBuffer, TIndex, TN>(device, dkh, dk, b, a.Lk, hk, a.D, false);
            Move<TBuffer, TIndex, TN>(device, dvh, dv, b, a.Lk, hk, a.D, false);
            return lane;
        }
        finally
        {
            device.Return(qh); device.Return(kh); device.Return(vh); device.Return(gh); device.Return(dqh); device.Return(dkh); device.Return(dvh);
        }
    }

    /// <summary>A tensor of [batch, steps, heads, d] a head at a time, on the host (the checks' references).</summary>
    public static double[] ToHeads(double[] x, int batch, int steps, int heads, int d)
    {
        var y = new double[x.Length];
        for (int b = 0; b < batch; b++)
        for (int s = 0; s < steps; s++)
        for (int h = 0; h < heads; h++)
            Array.Copy(x, (((long)b * steps + s) * heads + h) * d, y, (((long)b * heads + h) * steps + s) * d, d);
        return y;
    }

    /// <summary>A tensor of [batch, heads, steps, d] a step at a time, on the host.</summary>
    public static double[] FromHeads(double[] x, int batch, int steps, int heads, int d)
    {
        var y = new double[x.Length];
        for (int b = 0; b < batch; b++)
        for (int s = 0; s < steps; s++)
        for (int h = 0; h < heads; h++)
            Array.Copy(x, (((long)b * heads + h) * steps + s) * d, y, (((long)b * steps + s) * heads + h) * d, d);
        return y;
    }
}
