// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Compositions over two types: TN, the type the caller stores, read and written where it is stored; TS, its float
// lane's own type (float32 for the narrow floats and the block formats, float64 for float64), which holds a
// composition's intermediates — scores, gates, states across steps, gradients of intermediates. Nothing of TN is
// widened whole: products read their operands as their types (MatMul), element-wise passes load a chunk into lanes
// and store it as the output's type.
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>An operand of a composition: its bytes, a value offset, and whether it is TS (the lane's own type) or TN.</summary>
    internal readonly record struct Arg(byte[]? X, int At = 0, bool Lanes = false)
    {
        public Arg Plus(int values) => this with { At = At + values };
    }

    // count values of an operand from value `at` past its offset, into lanes / from lanes.
    private static void LoadArg<TN, TS, TLane>(in Arg x, int at, Span<TLane> lanes) where TN : struct, IAdad<TN, TLane> where TS : struct, IAdad<TS, TLane> where TLane : unmanaged
    {
        if (x.Lanes) TS.Load(x.X!, x.At + at, lanes);
        else TN.Load(x.X!, x.At + at, lanes);
    }

    private static void StoreArg<TN, TS, TLane>(in Arg x, ReadOnlySpan<TLane> lanes, int at) where TN : struct, IAdad<TN, TLane> where TS : struct, IAdad<TS, TLane> where TLane : unmanaged
    {
        if (x.Lanes) TS.Store(lanes, x.X!, x.At + at);
        else TN.Store(lanes, x.X!, x.At + at);
    }

    /// <summary>A composition run with TN's float lane and that lane's own type.</summary>
    internal interface IMixed
    {
        string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane>
            where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>;
    }

    /// <summary>Runs <paramref name="call"/> with TN's float lane and its own type; returns what it returns (the lane's name).</summary>
    internal static string Mixed<TN, TM>(TM call) where TN : struct, IFloatAdad<TN> where TM : struct, IMixed
    {
        var v = new MixedVisitor<TM>(call);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct MixedVisitor<TM>(TM call) : IFloatLaneVisitor where TM : struct, IMixed
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            var own = new MixedOwn<TN, TLane, TL, TM>(call);
            TL.Own(ref own);
            Lane = own.Lane;
        }
    }

    private struct MixedOwn<TN, TLane, TL, TM>(TM call) : IOwnVisitor<TLane>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> where TM : struct, IMixed
    {
        public string Lane = "";

        public void Run<TS>() where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> => Lane = call.Run<TN, TS, TLane, TL>();
    }

    /// <summary>o = x + y over n values, each operand TN or TS (Arg.Lanes); cols &gt; 0: y is a row of cols values added to
    /// every row of x; no y (its bytes null): o = x, converted. A chunk of whole rows loaded into lanes at a time.</summary>
    internal static void AddMixed<TN, TS, TLane, TL>(Arg x, Arg y, Arg o, int n, int cols = 0)
        where TN : struct, IAdad<TN, TLane> where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        int unit = o.Lanes ? 1 : TN.Granularity;
        int chunk = cols > 0 ? Math.Max(1, 4096 / cols) * cols : 4096;
        if (chunk % unit != 0 || o.At % unit != 0) chunk = n;                        // an output's storage units never split between tasks
        Team.For((n + chunk - 1) / chunk, 1, n, new AddMixedBody<TN, TS, TLane, TL>(x, y, o, n, cols, chunk));
    }

    private readonly struct AddMixedBody<TN, TS, TLane, TL>(Arg x, Arg y, Arg o, int n, int cols, int chunk) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Run(int first, int last)
        {
            var pool = ArrayPool<TLane>.Shared;
            int size = Math.Min(chunk, n);
            var xl = pool.Rent(size);
            var yl = pool.Rent(cols > 0 ? cols : size);
            if (cols > 0) LoadArg<TN, TS, TLane>(y, 0, yl.AsSpan(0, cols));
            for (int c = first; c < last; c++)
            {
                int at = c * chunk, len = Math.Min(chunk, n - at);
                var xs = xl.AsSpan(0, len);
                LoadArg<TN, TS, TLane>(x, at, xs);
                if (cols > 0)
                {
                    for (int r = 0; r < len; r += cols) Add(xs.Slice(r, cols), yl.AsSpan(0, cols));
                }
                else if (y.X is not null)
                {
                    LoadArg<TN, TS, TLane>(y, at, yl.AsSpan(0, len));
                    Add(xs, yl.AsSpan(0, len));
                }

                StoreArg<TN, TS, TLane>(o, xs, at);
            }

            pool.Return(xl); pool.Return(yl);
        }

        // x += y.
        private static void Add(Span<TLane> x, ReadOnlySpan<TLane> y)
        {
            int w = Vector<TLane>.Count, i = 0;
            ref TLane rx = ref MemoryMarshal.GetReference(x);
            ref TLane ry = ref MemoryMarshal.GetReference(y);
            for (; i + w <= x.Length; i += w) TL.Add(Vector.LoadUnsafe(ref rx, (nuint)i), Vector.LoadUnsafe(ref ry, (nuint)i)).StoreUnsafe(ref rx, (nuint)i);
            for (; i < x.Length; i++) x[i] = TL.Add(x[i], y[i]);
        }
    }

    /// <summary>C = op(A)·op(B) in TL, each operand TN or TS (Arg.Lanes), C at its offset: the product kernel reading each
    /// where it is stored.</summary>
    internal static void Prod<TN, TS, TLane, TL>(Arg a, bool transA, Arg b, bool transB, Arg c, int m, int n, int k)
        where TN : struct, IAdad<TN, TLane> where TS : struct, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        var o = new Operands(a.At, transA, b.At, transB, c.At);
        switch ((a.Lanes, b.Lanes, c.Lanes))
        {
            case (false, false, false): MatMul<TN, TN, TN, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            case (false, false, true): MatMul<TN, TN, TS, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            case (true, false, false): MatMul<TS, TN, TN, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            case (true, false, true): MatMul<TS, TN, TS, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            case (false, true, false): MatMul<TN, TS, TN, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            case (false, true, true): MatMul<TN, TS, TS, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            case (true, true, false): MatMul<TS, TS, TN, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
            default: MatMul<TS, TS, TS, TLane, TLane, TL>(a.X!, b.X!, c.X!, m, n, k, o); break;
        }
    }
}
