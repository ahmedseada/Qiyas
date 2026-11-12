// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A product's epilogue on the CPU (Products.cs ProductEpilogue): the product (every batch's operands read where they are
// stored, transposed or at their offsets) into the float lane's sums, the second product's into sums of its own (the same
// way), then the store (ProductStore): each value's sums, bias, β·y (y read where it is stored, a tile before it is
// written), the activation or the gradient through one (the formula's, Map.cs, in the lane) and the residual, all in the
// lane, y rounded to the type once; the sums kept by a pass of their own first. The convolution's general path (Conv.cs)
// takes the same store with its bias and activation.

using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>C = the epilogue's y of op(A)·op(B) on the CPU; returns the lane's name.</summary>
    public static string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) where TN : struct, IFloatAdad<TN>
    {
        var terms = t.Validate(e, v);
        return Mixed<TN, EpilogueCall>(new EpilogueCall(v, e, terms, a, b, c, t));
    }

    private readonly struct EpilogueCall(ProductView v, ProductEpilogue e, StoreTerms terms, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (v.M == 0 || v.N == 0 || v.Batch == 0) return TL.Name;
            var sums = new byte[v.ValuesC * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>()];
            ViewSums<TN, TS, TLane, TL>(v, a, b, sums);
            EpilogueStore<TN, TS, TLane, TL>(v, e, terms, t, sums, c);
            return TL.Name;
        }
    }

    // Every batch's op(A)·op(B) of the view into lane sums (the batches' C contiguous, m·n apart), each operand read where
    // it is stored.
    private static void ViewSums<TN, TS, TLane, TL>(ProductView v, byte[] a, byte[] b, byte[] sums)
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        int span = v.M * v.N;
        for (int i = 0; i < v.Batch; i++)
            MatMul<TN, TN, TS, TLane, TLane, TL>(a, b, sums, v.M, v.N, v.K,
                new Operands(AOff: (int)(i * v.StrideAOrDefault), TransA: v.TransA, BOff: (int)(i * v.StrideBOrDefault), TransB: v.TransB, COff: i * span));
    }

    // After a product's sums (lane storage, the view's C contiguous): the second product's into sums of its own, then the
    // store (the sums kept first where they are).
    private static void EpilogueStore<TN, TS, TLane, TL>(ProductView v, ProductEpilogue e, StoreTerms terms, ProductTerms<byte[]> t, byte[] sums, byte[] c)
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        byte[]? second = null;
        if (terms.Second)
        {
            second = new byte[v.ValuesC * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>()];
            ViewSums<TN, TS, TLane, TL>(e.Second!, t.SecondA!, t.SecondB!, second);
        }

        var src = new StoreSources(sums, second, t.Bias, t.Residual, t.ActInputs, 1, v.N, e.ActA, e.Beta);
        if (terms.Keep) ProductStore<TN, TLane, TL>(terms.KeepPass, src, t.Sums!, (int)v.ValuesC);
        ProductStore<TN, TLane, TL>(terms, src, c, (int)v.ValuesC);
    }

    /// <summary>A store's inputs: the sums (lane storage), the second product's (lane storage, or none), the bias (the type: a
    /// value an output channel, the channel (at / Cols) mod OC), the residual and the activation's inputs (the type, y's
    /// layout), the activation's parameter and β.</summary>
    private readonly record struct StoreSources(byte[] Sums, byte[]? Second, byte[]? Bias, byte[]? Residual, byte[]? ActInputs, int Cols, int OC, double ActA, double Beta);

    // y = the terms' value of every output (n of them), in aligned tiles of the index (a stored word or block written by one
    // task, y's values before read by the same task first), the formula (Map.cs) in the lane, stored as the type once.
    private static void ProductStore<TN, TLane, TL>(StoreTerms terms, StoreSources src, byte[] y, int n) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        if (n == 0) return;
        var bl = src.Bias is null || !terms.Bias ? null : new TLane[src.OC];
        if (bl is not null) TN.Load(src.Bias, 0, bl);
        var use = new StoreUse<TN, TLane, TL>(terms, src, bl, y, n);
        string op = terms.Grad ?? terms.Act ?? "copy";
        if (!MapOps.FloatOp<TLane, TL, StoreUse<TN, TLane, TL>>(op, ref use) && !MapOps.AnyOp<TLane, TL, StoreUse<TN, TLane, TL>>(op, ref use)) throw new ArgumentException($"product store: no formula {op}");
    }

    private readonly struct StoreUse<TN, TLane, TL>(StoreTerms terms, StoreSources src, TLane[]? bias, byte[] y, int n) : IOpUse<TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        public void Use<TOp>() where TOp : struct, IMapOp<TLane, TL>
        {
            int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, (long)n * 4, new StoreBody<TN, TLane, TL, TOp>(terms, src, bias, y));
        }
    }

    private readonly struct StoreBody<TN, TLane, TL, TOp>(StoreTerms terms, StoreSources src, TLane[]? bias, byte[] y) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        public void Run(int first, int last)
        {
            var s = MemoryMarshal.Cast<byte, TLane>(src.Sums.AsSpan());
            var l = terms.Second ? MemoryMarshal.Cast<byte, TLane>(src.Second.AsSpan()) : default;
            int lanes = Vector<TLane>.Count, P = src.Cols, room = Tile + lanes;
            Span<TLane> all = stackalloc TLane[3 * room];
            Span<TLane> v = all[..room], x = all.Slice(room, room), r = all.Slice(2 * room, room);
            var va = new Vector<TLane>(TL.From(src.ActA));
            var beta = TL.From(src.Beta);
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at), full = (len + lanes - 1) / lanes * lanes;
                for (int i = 0; i < len; i++) v[i] = s[at + i];
                if (terms.Second) for (int i = 0; i < len; i++) v[i] = TL.Add(v[i], l[at + i]);
                if (bias is not null) for (int i = 0; i < len; i++) v[i] = TL.Add(v[i], bias[(at + i) / P % src.OC]);
                if (terms.Beta)
                {
                    TN.Load(y, at, r[..len]);
                    for (int i = 0; i < len; i++) v[i] = TL.Fma(beta, r[i], v[i]);
                }

                v[len..full].Clear();
                if (terms.Grad is not null)
                {
                    TN.Load(src.ActInputs!, at, x[..len]);
                    x[len..full].Clear();
                }

                for (int i = 0; i < len; i += lanes)
                {
                    var vi = new Vector<TLane>(v.Slice(i, lanes));
                    var xi = terms.Grad is null ? vi : new Vector<TLane>(x.Slice(i, lanes));
                    TOp.Apply(xi, vi, vi, va, Vector<TLane>.Zero).CopyTo(v.Slice(i, lanes));
                }

                if (terms.Residual)
                {
                    TN.Load(src.Residual!, at, r[..len]);
                    for (int i = 0; i < len; i++) v[i] = TL.Add(v[i], r[i]);
                }

                TN.Store(v[..len], y, at);
            }
        }
    }
}
