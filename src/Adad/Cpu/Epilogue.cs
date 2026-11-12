// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A product's epilogue on the CPU (Products.cs ProductEpilogue): the product (every batch's operands read where they are
// stored, transposed or at their offsets) into the lane's sums, the second product's into sums of its own (the same
// way), then the store (ProductStore): each value's sums, bias, β·y (y read where it is stored, a tile before it is
// written), the activation or the gradient through one (the formula's, Map.cs, in the lane) and the residual, all in the
// lane, y rounded to the type once; the sums kept by a pass of their own first. A float type sums in its float lane; a
// type that is not float in the lane its products take by the bounds of every value the store makes
// (ProductEpilogue.IntegerSums: exact), the sums kept as the lane's own type and the terms added in its plain lane
// (ILane.Kept: a dot lane's accumulator's), its formulas Map's any-type ones. The convolution's general path (Conv.cs)
// takes the same store over its output (ConvArgs.Output), the bias a value an output channel.

using System.Numerics;
using System.Runtime.InteropServices;
using Adad.Abstraction;

namespace Adad.Cpu;

internal sealed partial class Kernels
{
    /// <summary>C = the epilogue's y of op(A)·op(B) on the CPU; returns the lane's name.</summary>
    public static string MatMulEpilogue<TN>(ProductView v, ProductEpilogue e, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) where TN : struct, IAdad<TN>
    {
        var terms = t.Validate(e, v);
        e.ValidateType(TN.Name, TN.IsFloat);
        if (TN.IsFloat)
        {
            var call = new MixedVisitor<EpilogueCall>(new EpilogueCall(v, e, terms, a, b, c, t));
            TN.TryFloat(ref call);
            return call.Lane;
        }

        (double, double) R(byte[] x, long n) => TN.Range(x, (int)n);
        var w = e.Second;
        var (bounds, _) = e.IntegerSums(TN.Name, TN.Min < 0, v.K, R(a, v.ValuesA), R(b, v.ValuesB), w is null ? null : R(t.SecondA!, w.ValuesA),
            w is null ? null : R(t.SecondB!, w.ValuesB), terms.Bias ? R(t.Bias!, v.N) : null, terms.Beta ? R(c, v.ValuesC) : null,
            terms.Residual ? R(t.Residual!, v.ValuesC) : null, terms.Grad is not null ? R(t.ActInputs!, v.ValuesC) : null);
        var sum = new EpilogueSum(v, e, terms, a, b, c, t);
        TN.Sum(ref sum, bounds);
        return sum.Lane;
    }

    // A type that is not float: the lane TN.Sum takes by the bounds, its sums kept as the lane's own type.
    private struct EpilogueSum(ProductView v, ProductEpilogue e, StoreTerms terms, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) : ISumVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
        {
            var kept = new EpilogueKept<TN, TLane, TLane, TL>(v, e, terms, a, b, c, t);
            TL.Kept(ref kept);
            Lane = TL.Name;
        }

        public void RunDot<TN, TIn, TAcc, TD>() where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TD : struct, IDotLane<TD, TIn, TAcc>
        {
            var kept = new EpilogueKept<TN, TIn, TAcc, TD>(v, e, terms, a, b, c, t);
            TD.Kept(ref kept);
            Lane = TD.Name;
        }
    }

    // The products in TL (TIn read, TAcc summed) into sums stored as TS, the store's terms in TP.
    private readonly struct EpilogueKept<TN, TIn, TAcc, TL>(ProductView v, ProductEpilogue e, StoreTerms terms, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) : IKeptVisitor<TAcc>
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        public void Run<TS, TP>() where TS : struct, IAdad<TS, TAcc> where TP : struct, ILane<TP, TAcc>
        {
            if (v.M == 0 || v.N == 0 || v.Batch == 0) return;
            int size = System.Runtime.CompilerServices.Unsafe.SizeOf<TAcc>();
            var sums = new byte[v.ValuesC * size];
            ViewSums<TN, TS, TIn, TAcc, TL>(v, a, b, sums);
            byte[]? second = null;
            if (terms.Second)
            {
                second = new byte[v.ValuesC * size];
                ViewSums<TN, TS, TIn, TAcc, TL>(e.Second!, t.SecondA!, t.SecondB!, second);
            }

            var src = new StoreSources(sums, second, t.Bias, t.Residual, t.ActInputs, 1, v.N, e.ActA, e.Beta) { ActB = e.ActB, Before = c };
            if (terms.Keep) ProductStoreAny<TN, TAcc, TP>(terms.KeepPass, src, t.Sums!, (int)v.ValuesC);
            ProductStoreAny<TN, TAcc, TP>(terms, src, c, (int)v.ValuesC);
        }
    }

    private readonly struct EpilogueCall(ProductView v, ProductEpilogue e, StoreTerms terms, byte[] a, byte[] b, byte[] c, ProductTerms<byte[]> t) : IMixed
    {
        public string Run<TN, TS, TLane, TL>() where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
        {
            if (v.M == 0 || v.N == 0 || v.Batch == 0) return TL.Name;
            var sums = new byte[v.ValuesC * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>()];
            ViewSums<TN, TS, TLane, TLane, TL>(v, a, b, sums);
            EpilogueStore<TN, TS, TLane, TL>(v, e, terms, t, sums, c, 1, v.N);
            return TL.Name;
        }
    }

    // Every batch's op(A)·op(B) of the view into lane sums stored as TS (the batches' C contiguous, m·n apart), each
    // operand read where it is stored (TIn, summed in TL's TAcc).
    private static void ViewSums<TN, TS, TIn, TAcc, TL>(ProductView v, byte[] a, byte[] b, byte[] sums)
        where TN : struct, IAdad<TN, TIn>, IAdad<TN, TAcc> where TS : struct, IAdad<TS, TAcc> where TIn : unmanaged where TAcc : unmanaged where TL : struct, ILane<TL, TAcc>
    {
        int span = v.M * v.N;
        for (int i = 0; i < v.Batch; i++)
            MatMul<TN, TN, TS, TIn, TAcc, TL>(a, b, sums, v.M, v.N, v.K,
                new Operands(AOff: (int)(i * v.StrideAOrDefault), TransA: v.TransA, BOff: (int)(i * v.StrideBOrDefault), TransB: v.TransB, COff: i * span));
    }

    // After a product's sums (lane storage, the view's C contiguous): the second product's into sums of its own, then the
    // store (the sums kept first where they are), value i's bias the channel (i / cols) mod oc's (a view's: 1, n).
    private static void EpilogueStore<TN, TS, TLane, TL>(ProductView v, ProductEpilogue e, StoreTerms terms, ProductTerms<byte[]> t, byte[] sums, byte[] c, int cols, int oc)
        where TN : struct, IAdad<TN, TLane> where TS : struct, IFloatAdad<TS>, IAdad<TS, TLane> where TLane : unmanaged where TL : struct, IFloatLane<TL, TLane>
    {
        byte[]? second = null;
        if (terms.Second)
        {
            second = new byte[v.ValuesC * System.Runtime.CompilerServices.Unsafe.SizeOf<TLane>()];
            ViewSums<TN, TS, TLane, TLane, TL>(e.Second!, t.SecondA!, t.SecondB!, second);
        }

        var src = new StoreSources(sums, second, t.Bias, t.Residual, t.ActInputs, cols, oc, e.ActA, e.Beta) { ActB = e.ActB, Before = c };
        if (terms.Keep) ProductStore<TN, TLane, TL>(terms.KeepPass, src, t.Sums!, (int)v.ValuesC);
        ProductStore<TN, TLane, TL>(terms, src, c, (int)v.ValuesC);
    }

    /// <summary>A store's inputs: the sums (lane storage), the second product's (lane storage, or none), the bias (the type: a
    /// value an output channel, the channel (at / Cols) mod OC), the residual and the activation's inputs (the type, y's
    /// layout), the activation's parameter and β.</summary>
    private readonly record struct StoreSources(byte[] Sums, byte[]? Second, byte[]? Bias, byte[]? Residual, byte[]? ActInputs, int Cols, int OC, double ActA, double Beta)
    {
        /// <summary>The formula's second parameter (b).</summary>
        public double ActB { get; init; }

        /// <summary>y's values before, β's term (the type, y's layout): the product's y, also where a pass stores elsewhere
        /// (the sums kept); null: the buffer the pass stores into.</summary>
        public byte[]? Before { get; init; }
    }

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

    // The store in a lane of any kind (a type that is not float: its any-type formulas, ProductEpilogue.ValidateType).
    private static void ProductStoreAny<TN, TLane, TL>(StoreTerms terms, StoreSources src, byte[] y, int n) where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        if (n == 0) return;
        var bl = src.Bias is null || !terms.Bias ? null : new TLane[src.OC];
        if (bl is not null) TN.Load(src.Bias, 0, bl);
        var use = new StoreUse<TN, TLane, TL>(terms, src, bl, y, n);
        string op = terms.Grad ?? terms.Act ?? "copy";
        if (!MapOps.AnyOp<TLane, TL, StoreUse<TN, TLane, TL>>(op, ref use)) throw new NotSupportedException($"product store: {op} is not a formula every lane has");
    }

    private readonly struct StoreUse<TN, TLane, TL>(StoreTerms terms, StoreSources src, TLane[]? bias, byte[] y, int n) : IOpUse<TLane, TL>
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane>
    {
        public void Use<TOp>() where TOp : struct, IMapOp<TLane, TL>
        {
            int size = Math.Max(Tile, (n / (Threads * 4) + Tile - 1) / Tile * Tile);
            Team.For(n, size, (long)n * 4, new StoreBody<TN, TLane, TL, TOp>(terms, src, bias, y));
        }
    }

    private readonly struct StoreBody<TN, TLane, TL, TOp>(StoreTerms terms, StoreSources src, TLane[]? bias, byte[] y) : Team.IBody
        where TN : struct, IAdad<TN, TLane> where TLane : unmanaged where TL : struct, ILane<TL, TLane> where TOp : struct, IMapOp<TLane, TL>
    {
        public void Run(int first, int last)
        {
            var s = MemoryMarshal.Cast<byte, TLane>(src.Sums.AsSpan());
            var l = terms.Second ? MemoryMarshal.Cast<byte, TLane>(src.Second.AsSpan()) : default;
            int lanes = Vector<TLane>.Count, P = src.Cols, room = Tile + lanes;
            Span<TLane> all = stackalloc TLane[3 * room];
            Span<TLane> v = all[..room], x = all.Slice(room, room), r = all.Slice(2 * room, room);
            Vector<TLane> va = new(TL.From(src.ActA)), vb = new(TL.From(src.ActB));
            var beta = TL.From(src.Beta);
            for (int at = first; at < last; at += Tile)
            {
                int len = Math.Min(Tile, last - at), full = (len + lanes - 1) / lanes * lanes;
                for (int i = 0; i < len; i++) v[i] = s[at + i];
                if (terms.Second) for (int i = 0; i < len; i++) v[i] = TL.Add(v[i], l[at + i]);
                if (bias is not null) for (int i = 0; i < len; i++) v[i] = TL.Add(v[i], bias[(at + i) / P % src.OC]);
                if (terms.Beta)
                {
                    TN.Load(src.Before ?? y, at, r[..len]);
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
                    TOp.Apply(xi, vi, vi, va, vb).CopyTo(v.Slice(i, lanes));
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
