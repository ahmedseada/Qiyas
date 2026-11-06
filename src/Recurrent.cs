// Recurrent cells: LSTM and GRU over rows (a batch of B rows, hidden size H, input size D), in the type's float lane.
//
//   The gates by Qiyas's products on views and the row-vector broadcast: g = x·Wᵀ + h·Uᵀ + b (LSTM: 4H columns in the
//   order i, f, g, o; GRU: gx = x·Wᵀ + b and gh = h·Uᵀ + bh kept apart, 3H columns in the order r, z, n).
//   The cell is one pass over the rows (thread per hidden unit on the device):
//     LSTM  i = σ(g_i), f = σ(g_f), ĝ = tanh(g_g), o = σ(g_o);  c' = f·c + i·ĝ;  h' = o·tanh(c')
//     GRU   r = σ(gx_r + gh_r), z = σ(gx_z + gh_z), n = tanh(gx_n + r·gh_n);  h' = (1 − z)·n + z·h
//   A step is the gates then the cell; a sequence is T steps (the cell's inputs of each step kept for the backward).
//   The cell backward is one pass (dgates, dc or dh from dh', dc'); the step backward the products' backward
//   (dx = dg·W, dh = dg·U, dW = dgᵀ·x, dU = dgᵀ·h, db = Σ rows of dg).
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Qiyas;

/// <summary>B rows, input size D, hidden size H; the kind ("lstm" or "gru").</summary>
public sealed record CellArgs(string Kind, int B, int D, int H)
{
    public int Gates => Kind == "lstm" ? 4 : 3;
    public long GateValues => (long)B * Gates * H;

    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    /// <summary>The float64 reference of the LSTM cell: (h', c') from the gates (B × 4H) and c (B × H).</summary>
    public void LstmCell(double[] g, double[] c, double[] hOut, double[] cOut)
    {
        for (int b = 0; b < B; b++) for (int j = 0; j < H; j++)
        {
            long row = (long)b * 4 * H;
            double i = Sigmoid(g[row + j]), f = Sigmoid(g[row + H + j]), gg = Math.Tanh(g[row + 2 * H + j]), o = Sigmoid(g[row + 3 * H + j]);
            double cn = f * c[b * H + j] + i * gg;
            cOut[b * H + j] = cn;
            hOut[b * H + j] = o * Math.Tanh(cn);
        }
    }

    /// <summary>The float64 reference of the GRU cell: h' from gx, gh (B × 3H) and h (B × H).</summary>
    public void GruCell(double[] gx, double[] gh, double[] h, double[] hOut)
    {
        for (int b = 0; b < B; b++) for (int j = 0; j < H; j++)
        {
            long row = (long)b * 3 * H;
            double r = Sigmoid(gx[row + j] + gh[row + j]), z = Sigmoid(gx[row + H + j] + gh[row + H + j]);
            double n = Math.Tanh(gx[row + 2 * H + j] + r * gh[row + 2 * H + j]);
            hOut[b * H + j] = (1 - z) * n + z * h[b * H + j];
        }
    }

    /// <summary>The LSTM cell backward: from dh', dc' (and the gates, c): dgates (B × 4H) and dc (B × H).</summary>
    public void LstmCellBackward(double[] g, double[] c, double[] dh, double[] dcNext, double[] dg, double[] dc)
    {
        for (int b = 0; b < B; b++) for (int j = 0; j < H; j++)
        {
            long row = (long)b * 4 * H;
            double i = Sigmoid(g[row + j]), f = Sigmoid(g[row + H + j]), gg = Math.Tanh(g[row + 2 * H + j]), o = Sigmoid(g[row + 3 * H + j]);
            double cn = f * c[b * H + j] + i * gg, tc = Math.Tanh(cn);
            double dcn = dcNext[b * H + j] + dh[b * H + j] * o * (1 - tc * tc);
            dg[row + 3 * H + j] = dh[b * H + j] * tc * o * (1 - o);
            dg[row + j] = dcn * gg * i * (1 - i);
            dg[row + H + j] = dcn * c[b * H + j] * f * (1 - f);
            dg[row + 2 * H + j] = dcn * i * (1 - gg * gg);
            dc[b * H + j] = dcn * f;
        }
    }

    /// <summary>The GRU cell backward: from dh' (and gx, gh, h): dgx, dgh (B × 3H) and dh (B × H).</summary>
    public void GruCellBackward(double[] gx, double[] gh, double[] h, double[] dhOut, double[] dgx, double[] dgh, double[] dh)
    {
        for (int b = 0; b < B; b++) for (int j = 0; j < H; j++)
        {
            long row = (long)b * 3 * H;
            double r = Sigmoid(gx[row + j] + gh[row + j]), z = Sigmoid(gx[row + H + j] + gh[row + H + j]);
            double n = Math.Tanh(gx[row + 2 * H + j] + r * gh[row + 2 * H + j]);
            double d = dhOut[b * H + j];
            double dn = d * (1 - z), dz = d * (h[b * H + j] - n);
            double dpre = dn * (1 - n * n);
            dgx[row + 2 * H + j] = dpre;
            dgh[row + 2 * H + j] = dpre * r;
            double dr = dpre * gh[row + 2 * H + j];
            dgx[row + j] = dgh[row + j] = dr * r * (1 - r);
            dgx[row + H + j] = dgh[row + H + j] = dz * z * (1 - z);
            dh[b * H + j] = d * z;
        }
    }
}

internal static partial class Kernels
{
    /// <summary>The LSTM cell on the CPU: h', c' from the gates and c (lane arithmetic); returns the lane's name.</summary>
    public static string LstmCell<TN>(CellArgs a, byte[] g, byte[] c, byte[] hOut, byte[] cOut) where TN : struct, IQiyasFloat<TN>
    {
        var v = new CellVisitor(a, g, null, c, hOut, cOut);
        TN.Float(ref v);
        return v.Lane;
    }

    /// <summary>The GRU cell on the CPU: h' from gx, gh and h.</summary>
    public static string GruCell<TN>(CellArgs a, byte[] gx, byte[] gh, byte[] h, byte[] hOut) where TN : struct, IQiyasFloat<TN>
    {
        var v = new CellVisitor(a, gx, gh, h, hOut, null);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct CellVisitor(CellArgs a, byte[] g, byte[]? g2, byte[] state, byte[] hOut, byte[]? cOut) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(a.B, RowChunk<TN>(a.B, a.H, a.B / (Threads * 4)), (long)a.B * a.H * 8, new CellBody<TN, TLane, TL>(a, g, g2, state, hOut, cOut));
        }
    }

    private struct CellBody<TN, TLane, TL>(CellArgs a, byte[] g, byte[]? g2, byte[] state, byte[] hOut, byte[]? cOut) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            int H = a.H, G = a.Gates;
            var pool = ArrayPool<TLane>.Shared;
            var gl = pool.Rent(G * H);
            var g2l = pool.Rent(G * H);
            var sl = pool.Rent(H);
            var ho = pool.Rent(H);
            var co = pool.Rent(H);
            try
            {
                int w = Vector<TLane>.Count, vec = H / w * w;
                var one = new Vector<TLane>(TL.From(1));
                for (int b = first; b < last; b++)
                {
                    TN.Load(g, b * G * H, gl.AsSpan(0, G * H));
                    if (g2 is not null) TN.Load(g2, b * G * H, g2l.AsSpan(0, G * H));
                    TN.Load(state, b * H, sl.AsSpan(0, H));
                    if (a.Kind == "lstm")
                    {
                        int j = 0;
                        for (; j < vec; j += w)
                        {
                            var i = TL.Sigmoid(new Vector<TLane>(gl.AsSpan(j)));
                            var f = TL.Sigmoid(new Vector<TLane>(gl.AsSpan(H + j)));
                            var gg = TL.Tanh(new Vector<TLane>(gl.AsSpan(2 * H + j)));
                            var o = TL.Sigmoid(new Vector<TLane>(gl.AsSpan(3 * H + j)));
                            var cn = TL.Fma(f, new Vector<TLane>(sl.AsSpan(j)), i * gg);
                            cn.CopyTo(co.AsSpan(j));
                            (o * TL.Tanh(cn)).CopyTo(ho.AsSpan(j));
                        }

                        for (; j < H; j++)
                        {
                            TLane i = TL.Sigmoid(new Vector<TLane>(gl[j]))[0], f = TL.Sigmoid(new Vector<TLane>(gl[H + j]))[0], gg = TL.Tanh(new Vector<TLane>(gl[2 * H + j]))[0], o = TL.Sigmoid(new Vector<TLane>(gl[3 * H + j]))[0];
                            TLane cn = TL.Fma(f, sl[j], TL.Mul(i, gg));
                            co[j] = cn;
                            ho[j] = TL.Mul(o, TL.Tanh(new Vector<TLane>(cn))[0]);
                        }

                        TN.Store(ho.AsSpan(0, H), hOut, b * H);
                        TN.Store(co.AsSpan(0, H), cOut!, b * H);
                    }
                    else
                    {
                        int j = 0;
                        for (; j < vec; j += w)
                        {
                            var r = TL.Sigmoid(new Vector<TLane>(gl.AsSpan(j)) + new Vector<TLane>(g2l.AsSpan(j)));
                            var z = TL.Sigmoid(new Vector<TLane>(gl.AsSpan(H + j)) + new Vector<TLane>(g2l.AsSpan(H + j)));
                            var n = TL.Tanh(TL.Fma(r, new Vector<TLane>(g2l.AsSpan(2 * H + j)), new Vector<TLane>(gl.AsSpan(2 * H + j))));
                            var hv = new Vector<TLane>(sl.AsSpan(j));
                            TL.Fma(z, hv, (one - z) * n).CopyTo(ho.AsSpan(j));
                        }

                        for (; j < H; j++)
                        {
                            TLane r = TL.Sigmoid(new Vector<TLane>(TL.Add(gl[j], g2l[j])))[0], z = TL.Sigmoid(new Vector<TLane>(TL.Add(gl[H + j], g2l[H + j])))[0];
                            TLane n = TL.Tanh(new Vector<TLane>(TL.Fma(r, g2l[2 * H + j], gl[2 * H + j])))[0];
                            ho[j] = TL.Fma(z, sl[j], TL.Mul(TL.Sub(TL.From(1), z), n));
                        }

                        TN.Store(ho.AsSpan(0, H), hOut, b * H);
                    }
                }
            }
            finally
            {
                pool.Return(gl); pool.Return(g2l); pool.Return(sl); pool.Return(ho); pool.Return(co);
            }
        }
    }

    /// <summary>The gates g = x·Wᵀ + h·Uᵀ + b (B × GH) in lane storage of the type, by the products on views and the broadcast add.</summary>
    public static void CellGates<TN>(CellArgs a, byte[] x, byte[] w, byte[]? h, byte[]? u, byte[] bias, byte[] g) where TN : struct, IQiyasNumber<TN>
    {
        int GH = a.Gates * a.H;
        MatMulView<TN>(new ProductView(a.B, GH, a.D, false, true), x, w, g);                       // x · Wᵀ (W is GH × D)
        if (h is not null && u is not null)
        {
            var gh = new byte[g.Length];
            MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), h, u, gh);                  // h · Uᵀ (U is GH × H)
            Map<TN>("add", g, gh, null, g, (int)a.GateValues, 0, 0);
        }

        MapRow<TN>("add", g, bias, g, a.B, GH);
    }

    private static bool OwnLane<TN>() where TN : struct, IQiyasFloat<TN> => TN.Name is "float32" or "float64";

    // A narrow float's step runs in its lane's storage: the gates are sums of products the type cannot hold between the passes.
    private static byte[] L<TN>(byte[] x, long count) where TN : struct, IQiyasFloat<TN> => ToLane<TN>(x, count);
    private static byte[] LOut<TN>(long count) where TN : struct, IQiyasFloat<TN> => new byte[count * (TN.Bits == 64 ? 8 : 4)];

    /// <summary>An LSTM step: (h', c') from x, h, c and the weights W (4H × D), U (4H × H), b (4H).</summary>
    public static string LstmStep<TN>(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut) where TN : struct, IQiyasFloat<TN>
    {
        if (!OwnLane<TN>())
        {
            long bh = (long)a.B * a.H;
            var lh = LOut<TN>(bh); var lc = LOut<TN>(bh);
            string lane = TN.Bits == 64
                ? LstmStep<Float64>(a, L<TN>(x, (long)a.B * a.D), L<TN>(h, bh), L<TN>(c, bh), L<TN>(w, a.GateValues / a.B * a.D), L<TN>(u, a.GateValues / a.B * a.H), L<TN>(bias, a.Gates * a.H), lh, lc)
                : LstmStep<Float32>(a, L<TN>(x, (long)a.B * a.D), L<TN>(h, bh), L<TN>(c, bh), L<TN>(w, a.GateValues / a.B * a.D), L<TN>(u, a.GateValues / a.B * a.H), L<TN>(bias, a.Gates * a.H), lh, lc);
            FromLane<TN>(lh, hOut, bh); FromLane<TN>(lc, cOut, bh);
            return lane;
        }

        var g = new byte[BytesOf<TN>(a.GateValues)];
        CellGates<TN>(a, x, w, h, u, bias, g);
        return LstmCell<TN>(a, g, c, hOut, cOut);
    }

    /// <summary>The LSTM cell backward on the CPU: dg (B × 4H) and dc from dh', dc'.</summary>
    public static string LstmCellBackward<TN>(CellArgs a, byte[] g, byte[] c, byte[] dh, byte[] dcNext, byte[] dg, byte[] dc) where TN : struct, IQiyasFloat<TN>
    {
        var v = new CellBackwardVisitor(a, g, null, c, dh, dcNext, dg, dc, null);
        TN.Float(ref v);
        return v.Lane;
    }

    /// <summary>The GRU cell backward on the CPU: dgx, dgh (B × 3H) and dh from dh'.</summary>
    public static string GruCellBackward<TN>(CellArgs a, byte[] gx, byte[] gh, byte[] h, byte[] dhOut, byte[] dgx, byte[] dgh, byte[] dh) where TN : struct, IQiyasFloat<TN>
    {
        var v = new CellBackwardVisitor(a, gx, gh, h, dhOut, null, dgx, dgh, dh);
        TN.Float(ref v);
        return v.Lane;
    }

    private struct CellBackwardVisitor(CellArgs a, byte[] g, byte[]? g2, byte[] state, byte[] d1, byte[]? d2, byte[] o1, byte[] o2, byte[]? o3) : IFloatLaneVisitor
    {
        public string Lane = "";

        public void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
        {
            Lane = TL.Name;
            Team.For(a.B, RowChunk<TN>(a.B, a.Gates * a.H, a.B / (Threads * 4)), (long)a.B * a.H * 16, new CellBackwardBody<TN, TLane, TL>(a, g, g2, state, d1, d2, o1, o2, o3));
        }
    }

    private struct CellBackwardBody<TN, TLane, TL>(CellArgs a, byte[] g, byte[]? g2, byte[] state, byte[] d1, byte[]? d2, byte[] o1, byte[] o2, byte[]? o3) : Team.IBody
        where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasFloatLane<TL, TLane>
    {
        public readonly void Run(int first, int last)
        {
            int H = a.H, G = a.Gates;
            var pool = ArrayPool<TLane>.Shared;
            var gl = pool.Rent(G * H);
            var g2l = pool.Rent(G * H);
            var sl = pool.Rent(H);
            var d1l = pool.Rent(H);
            var d2l = pool.Rent(H);
            var dg = pool.Rent(G * H);
            var dg2 = pool.Rent(G * H);
            var ds = pool.Rent(H);
            try
            {
                TLane one = TL.From(1);
                TLane Sig(TLane x) => TL.Sigmoid(new Vector<TLane>(x))[0];
                TLane Tanh(TLane x) => TL.Tanh(new Vector<TLane>(x))[0];
                for (int b = first; b < last; b++)
                {
                    TN.Load(g, b * G * H, gl.AsSpan(0, G * H));
                    if (g2 is not null) TN.Load(g2, b * G * H, g2l.AsSpan(0, G * H));
                    TN.Load(state, b * H, sl.AsSpan(0, H));
                    TN.Load(d1, b * H, d1l.AsSpan(0, H));
                    if (d2 is not null) TN.Load(d2, b * H, d2l.AsSpan(0, H));
                    // Whole vectors of the row first (the same formulas, a vector's sigmoid / tanh a vector of values), the tail
                    // one value at a time.
                    int w = Vector<TLane>.Count, vec = H / w * w, j0 = 0;
                    var onev = new Vector<TLane>(one);
                    Vector<TLane> V(TLane[] arr, int at) => new(arr.AsSpan(at));
                    for (; j0 < vec; j0 += w)
                    {
                        if (a.Kind == "lstm")
                        {
                            var i = TL.Sigmoid(V(gl, j0)); var f = TL.Sigmoid(V(gl, H + j0)); var gg = TL.Tanh(V(gl, 2 * H + j0)); var o = TL.Sigmoid(V(gl, 3 * H + j0));
                            var s = V(sl, j0); var d = V(d1l, j0);
                            var cn = TL.Fma(f, s, i * gg); var tc = TL.Tanh(cn);
                            var dcn = TL.Fma(d * o, onev - tc * tc, V(d2l, j0));
                            (d * tc * (o * (onev - o))).CopyTo(dg.AsSpan(3 * H + j0));
                            (dcn * gg * (i * (onev - i))).CopyTo(dg.AsSpan(j0));
                            (dcn * s * (f * (onev - f))).CopyTo(dg.AsSpan(H + j0));
                            (dcn * i * (onev - gg * gg)).CopyTo(dg.AsSpan(2 * H + j0));
                            (dcn * f).CopyTo(ds.AsSpan(j0));
                        }
                        else
                        {
                            var r = TL.Sigmoid(V(gl, j0) + V(g2l, j0)); var z = TL.Sigmoid(V(gl, H + j0) + V(g2l, H + j0));
                            var hn = V(g2l, 2 * H + j0);
                            var n = TL.Tanh(TL.Fma(r, hn, V(gl, 2 * H + j0)));
                            var d = V(d1l, j0);
                            var dn = d * (onev - z); var dz = d * (V(sl, j0) - n);
                            var dpre = dn * (onev - n * n);
                            var dr = dpre * hn;
                            dpre.CopyTo(dg.AsSpan(2 * H + j0));
                            (dpre * r).CopyTo(dg2.AsSpan(2 * H + j0));
                            var drr = dr * (r * (onev - r)); drr.CopyTo(dg.AsSpan(j0)); drr.CopyTo(dg2.AsSpan(j0));
                            var dzz = dz * (z * (onev - z)); dzz.CopyTo(dg.AsSpan(H + j0)); dzz.CopyTo(dg2.AsSpan(H + j0));
                            (d * z).CopyTo(ds.AsSpan(j0));
                        }
                    }

                    for (int j = vec; j < H; j++)
                    {
                        if (a.Kind == "lstm")
                        {
                            TLane i = Sig(gl[j]), f = Sig(gl[H + j]), gg = Tanh(gl[2 * H + j]), o = Sig(gl[3 * H + j]);
                            TLane cn = TL.Fma(f, sl[j], TL.Mul(i, gg)), tc = Tanh(cn);
                            TLane dcn = TL.Fma(TL.Mul(d1l[j], o), TL.Sub(one, TL.Mul(tc, tc)), d2l[j]);
                            dg[3 * H + j] = TL.Mul(TL.Mul(d1l[j], tc), TL.Mul(o, TL.Sub(one, o)));
                            dg[j] = TL.Mul(TL.Mul(dcn, gg), TL.Mul(i, TL.Sub(one, i)));
                            dg[H + j] = TL.Mul(TL.Mul(dcn, sl[j]), TL.Mul(f, TL.Sub(one, f)));
                            dg[2 * H + j] = TL.Mul(TL.Mul(dcn, i), TL.Sub(one, TL.Mul(gg, gg)));
                            ds[j] = TL.Mul(dcn, f);
                        }
                        else
                        {
                            TLane r = Sig(TL.Add(gl[j], g2l[j])), z = Sig(TL.Add(gl[H + j], g2l[H + j]));
                            TLane n = Tanh(TL.Fma(r, g2l[2 * H + j], gl[2 * H + j]));
                            TLane d = d1l[j];
                            TLane dn = TL.Mul(d, TL.Sub(one, z)), dz = TL.Mul(d, TL.Sub(sl[j], n));
                            TLane dpre = TL.Mul(dn, TL.Sub(one, TL.Mul(n, n)));
                            TLane dr = TL.Mul(dpre, g2l[2 * H + j]);
                            dg[2 * H + j] = dpre;
                            dg2[2 * H + j] = TL.Mul(dpre, r);
                            dg[j] = dg2[j] = TL.Mul(dr, TL.Mul(r, TL.Sub(one, r)));
                            dg[H + j] = dg2[H + j] = TL.Mul(dz, TL.Mul(z, TL.Sub(one, z)));
                            ds[j] = TL.Mul(d, z);
                        }
                    }

                    TN.Store(dg.AsSpan(0, G * H), o1, b * G * H);
                    if (a.Kind == "lstm") TN.Store(ds.AsSpan(0, H), o2, b * H);
                    else { TN.Store(dg2.AsSpan(0, G * H), o2, b * G * H); TN.Store(ds.AsSpan(0, H), o3!, b * H); }
                }
            }
            finally
            {
                pool.Return(gl); pool.Return(g2l); pool.Return(sl); pool.Return(d1l); pool.Return(d2l); pool.Return(dg); pool.Return(dg2); pool.Return(ds);
            }
        }
    }

    /// <summary>The products' backward of a step's gates: from dg (B × GH): dx += dg·W, dh += dg·U, dW = dgᵀ·x, dU = dgᵀ·h, db = Σ rows of dg.</summary>
    public static void GatesBackward<TN>(CellArgs a, byte[] dg, byte[] x, byte[]? h, byte[] w, byte[]? u, byte[] dx, byte[]? dh, byte[] dw, byte[]? du, byte[] db) where TN : struct, IQiyasNumber<TN>
    {
        int GH = a.Gates * a.H;
        MatMulView<TN>(new ProductView(a.B, a.D, GH, false, false), dg, w, dx);                         // dg · W
        if (h is not null && u is not null && dh is not null && du is not null)
        {
            MatMulView<TN>(new ProductView(a.B, a.H, GH, false, false), dg, u, dh);                     // dg · U
            MatMulView<TN>(new ProductView(GH, a.H, a.B, true, false), dg, h, du);                      // dgᵀ · h
        }

        MatMulView<TN>(new ProductView(GH, a.D, a.B, true, false), dg, x, dw);                          // dgᵀ · x
        var (sum, _) = Reduce<TN>("sum", Axis.Cols, dg, a.B, GH);
        CopyLanes<TN>(sum, db, GH);
    }

    // Lane values (the reductions' results: float32 / float64 bytes) stored as the type.
    private static void CopyLanes<TN>(byte[] lanes, byte[] output, int count) where TN : struct, IQiyasNumber<TN>
    {
        var v = new LanesToTypeVisitor(lanes, output, count);
        TN.Elementwise(ref v);
    }

    private struct LanesToTypeVisitor(byte[] lanes, byte[] output, int count) : ILaneVisitor
    {
        public readonly void Run<TN, TLane, TL>() where TN : struct, IQiyasNumber<TN, TLane> where TLane : unmanaged where TL : struct, IQiyasLane<TL, TLane>
        {
            // The sum lane may be wider than TLane (f32 sums of a narrow float, s32 of an integer type): read by its size.
            var values = new TLane[count];
            int size = lanes.Length / count;
            for (int i = 0; i < count; i++)
            {
                double d = size == 8 ? (typeof(TLane) == typeof(double) || TL.Name == "f64" ? BitConverter.ToDouble(lanes, i * 8) : BitConverter.ToInt64(lanes, i * 8)) : TL.Name[0] == 'f' ? BitConverter.ToSingle(lanes, i * 4) : BitConverter.ToInt32(lanes, i * 4);
                values[i] = TL.From(d);
            }

            TN.Store(values, output, 0);
        }
    }

    /// <summary>An LSTM step's backward: from dh', dc' (and x, h, c, the weights): dx, dh, dc, dW, dU, db.</summary>
    public static string LstmStepBackward<TN>(CellArgs a, byte[] x, byte[] h, byte[] c, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcOut, byte[] dx, byte[] dh, byte[] dc, byte[] dw, byte[] du, byte[] db) where TN : struct, IQiyasFloat<TN>
    {
        if (!OwnLane<TN>())
        {
            long bh = (long)a.B * a.H, bd = (long)a.B * a.D, GH = a.Gates * a.H;
            var ldx = LOut<TN>(bd); var ldh = LOut<TN>(bh); var ldc = LOut<TN>(bh); var ldw = LOut<TN>(GH * a.D); var ldu = LOut<TN>(GH * a.H); var ldb = LOut<TN>(GH);
            string name = TN.Bits == 64
                ? LstmStepBackward<Float64>(a, L<TN>(x, bd), L<TN>(h, bh), L<TN>(c, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(dhOut, bh), L<TN>(dcOut, bh), ldx, ldh, ldc, ldw, ldu, ldb)
                : LstmStepBackward<Float32>(a, L<TN>(x, bd), L<TN>(h, bh), L<TN>(c, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(dhOut, bh), L<TN>(dcOut, bh), ldx, ldh, ldc, ldw, ldu, ldb);
            FromLane<TN>(ldx, dx, bd); FromLane<TN>(ldh, dh, bh); FromLane<TN>(ldc, dc, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH);
            return name;
        }

        var g = new byte[BytesOf<TN>(a.GateValues)];
        CellGates<TN>(a, x, w, h, u, bias, g);
        var dg = new byte[g.Length];
        string lane = LstmCellBackward<TN>(a, g, c, dhOut, dcOut, dg, dc);
        GatesBackward<TN>(a, dg, x, h, w, u, dx, dh, dw, du, db);
        return lane;
    }

    /// <summary>A GRU step's backward: from dh' (and x, h, the weights): dx, dh, dW, dU, db, dbh.</summary>
    public static string GruStepBackward<TN>(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut, byte[] dx, byte[] dh, byte[] dw, byte[] du, byte[] db, byte[] dbh) where TN : struct, IQiyasFloat<TN>
    {
        if (!OwnLane<TN>())
        {
            long bh = (long)a.B * a.H, bd = (long)a.B * a.D, GH = a.Gates * a.H;
            var ldx = LOut<TN>(bd); var ldh = LOut<TN>(bh); var ldw = LOut<TN>(GH * a.D); var ldu = LOut<TN>(GH * a.H); var ldb = LOut<TN>(GH); var ldbh = LOut<TN>(GH);
            string name = TN.Bits == 64
                ? GruStepBackward<Float64>(a, L<TN>(x, bd), L<TN>(h, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), L<TN>(dhOut, bh), ldx, ldh, ldw, ldu, ldb, ldbh)
                : GruStepBackward<Float32>(a, L<TN>(x, bd), L<TN>(h, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), L<TN>(dhOut, bh), ldx, ldh, ldw, ldu, ldb, ldbh);
            FromLane<TN>(ldx, dx, bd); FromLane<TN>(ldh, dh, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH); FromLane<TN>(ldbh, dbh, GH);
            return name;
        }

        var gx = new byte[BytesOf<TN>(a.GateValues)];
        var gh = new byte[gx.Length];
        CellGates<TN>(a, x, w, null, null, bias, gx);
        var hArgs = new CellArgs("gru", a.B, a.H, a.H);
        CellGates<TN>(hArgs, h, u, null, null, biasH, gh);
        var dgx = new byte[gx.Length];
        var dgh = new byte[gx.Length];
        var dhCell = new byte[dh.Length];
        string lane = GruCellBackward<TN>(a, gx, gh, h, dhOut, dgx, dgh, dhCell);
        var dxOnly = new byte[dx.Length];
        GatesBackward<TN>(a, dgx, x, null, w, null, dx, null, dw, null, db);
        var dhFromGates = new byte[dh.Length];
        GatesBackward<TN>(hArgs, dgh, h, null, u, null, dhFromGates, null, du, null, dbh);
        Map<TN>("add", dhFromGates, dhCell, null, dh, (int)((long)a.B * a.H), 0, 0);
        return lane;
    }

    /// <summary>An LSTM over T steps: x (T × B × D), h0, c0 → h (T × B × H), c (T × B × H); returns the lane's name.</summary>
    public static string LstmSequence<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut) where TN : struct, IQiyasFloat<TN>
    {
        long bd = (long)a.B * a.D, bh = (long)a.B * a.H;
        if (!OwnLane<TN>())
        {
            long GH = a.Gates * a.H;
            var lh = LOut<TN>(T * bh); var lc = LOut<TN>(T * bh);
            string name = TN.Bits == 64
                ? LstmSequence<Float64>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(c0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), lh, lc)
                : LstmSequence<Float32>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(c0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), lh, lc);
            FromLane<TN>(lh, hOut, T * bh); FromLane<TN>(lc, cOut, T * bh);
            return name;
        }

        return LstmForward<TN>(a, T, x, h0, c0, w, u, bias, hOut, cOut, null);
    }

    // x · Wᵀ + b for every step at once (T·B × GH): one product over all the steps' rows, the bias added once; a step then
    // adds only h · Uᵀ.
    private static byte[] InputGates<TN>(CellArgs a, int T, byte[] x, byte[] w, byte[] bias) where TN : struct, IQiyasFloat<TN>
    {
        int GH = a.Gates * a.H, rows = T * a.B;
        var gx = Scratch.Get(Scratch.Gx, BytesOf<TN>((long)rows * GH));
        MatMulView<TN>(new ProductView(rows, GH, a.D, false, true), x, w, gx);
        MapRow<TN>("add", gx, bias, gx, rows, GH);
        return gx;
    }

    // The LSTM over the steps from the input gates of all steps at once (h, c of every step into hOut, cOut); with gates
    // given, each step's gates (pre-activation, T × B × GH) kept for the backward.
    private static string LstmForward<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] hOut, byte[] cOut, byte[]? gates) where TN : struct, IQiyasFloat<TN>
    {
        int GH = a.Gates * a.H, hBytes = (int)TN.Bytes((long)a.B * a.H), gBytes = BytesOf<TN>((long)a.B * GH);
        var gx = InputGates<TN>(a, T, x, w, bias);
        var hPrev = h0; var cPrev = c0;
        // The step's buffers allocated once (megabytes each: a fresh array a step is a large-object allocation and its
        // zeroing); h and c alternate between two (the step reads the previous one).
        var g = Scratch.Get(Scratch.StepG, gBytes); var gh = Scratch.Get(Scratch.StepG2, gBytes);
        byte[][] hb = [Scratch.Get(Scratch.H0, hBytes), Scratch.Get(Scratch.H1, hBytes)], cb = [Scratch.Get(Scratch.C0, hBytes), Scratch.Get(Scratch.C1, hBytes)];
        string lane = "";
        for (int t = 0; t < T; t++)
        {
            Buffer.BlockCopy(gx, t * gBytes, g, 0, gBytes);
            MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), hPrev, u, gh);                // h · Uᵀ
            Map<TN>("add", g, gh, null, g, a.B * GH, 0, 0);
            if (gates is not null) Buffer.BlockCopy(g, 0, gates, t * gBytes, gBytes);
            var ht = hb[t & 1]; var ct = cb[t & 1];
            lane = LstmCell<TN>(a, g, cPrev, ht, ct);
            Buffer.BlockCopy(ht, 0, hOut, t * hBytes, hBytes);
            Buffer.BlockCopy(ct, 0, cOut, t * hBytes, hBytes);
            hPrev = ht; cPrev = ct;
        }

        return lane;
    }

    // The GRU over the steps: gx of all steps at once, a step's gh = h · Uᵀ + bh; with the arrays given, each step's gh kept
    // (gx is returned) for the backward.
    private static (string Lane, byte[] Gx) GruForward<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut, byte[]? ghAll) where TN : struct, IQiyasFloat<TN>
    {
        int GH = a.Gates * a.H, hBytes = (int)TN.Bytes((long)a.B * a.H), gBytes = BytesOf<TN>((long)a.B * GH);
        var gx = InputGates<TN>(a, T, x, w, bias);
        var hPrev = h0;
        var gxt = Scratch.Get(Scratch.StepG, gBytes); var gh = Scratch.Get(Scratch.StepG2, gBytes);
        byte[][] hb = [Scratch.Get(Scratch.H0, hBytes), Scratch.Get(Scratch.H1, hBytes)];
        string lane = "";
        for (int t = 0; t < T; t++)
        {
            Buffer.BlockCopy(gx, t * gBytes, gxt, 0, gBytes);
            MatMulView<TN>(new ProductView(a.B, GH, a.H, false, true), hPrev, u, gh);                // h · Uᵀ
            MapRow<TN>("add", gh, biasH, gh, a.B, GH);
            if (ghAll is not null) Buffer.BlockCopy(gh, 0, ghAll, t * gBytes, gBytes);
            var ht = hb[t & 1];
            lane = GruCell<TN>(a, gxt, gh, hPrev, ht);
            Buffer.BlockCopy(ht, 0, hOut, t * hBytes, hBytes);
            hPrev = ht;
        }

        return (lane, gx);
    }

    // The weights' gradients of all steps at once from the gates' gradients dg (T·B × GH): dx = dg · W (T·B × D),
    // dW = dgᵀ · x, dU = dgᵀ · hPrev (hPrev: each step's input state, T·B × H), db = the column sums of dg.
    private static void SequenceWeights<TN>(CellArgs a, int T, byte[] dg, byte[] x, byte[]? hPrev, byte[] w, byte[]? dx, byte[] dw, byte[]? du, byte[] db) where TN : struct, IQiyasFloat<TN>
    {
        int GH = a.Gates * a.H, rows = T * a.B;
        if (dx is not null) MatMulView<TN>(new ProductView(rows, a.D, GH, false, false), dg, w, dx);   // dg · W
        MatMulView<TN>(new ProductView(GH, a.D, rows, true, false), dg, x, dw);                          // dgᵀ · x
        if (hPrev is not null && du is not null) MatMulView<TN>(new ProductView(GH, a.H, rows, true, false), dg, hPrev, du);   // dgᵀ · hPrev
        var (sum, _) = Reduce<TN>("sum", Axis.Cols, dg, rows, GH);
        CopyLanes<TN>(sum, db, GH);
    }

    // The sequences' buffers kept per thread at their exact length (tens of megabytes for all the steps' gates: a fresh
    // array a call is a large-object allocation, its zeroing and the collections it brings); every one is written whole
    // before it is read. A slot a call site.
    private static class Scratch
    {
        public const int Gx = 0, StepG = 1, StepG2 = 2, H0 = 3, H1 = 4, C0 = 5, C1 = 6, Prev = 7, Hs = 8, Cs = 9, Gates = 10, Dg = 11, Dg2 = 12,
            Dh = 13, BackG = 14, BackG2 = 15, BackDg = 16, BackDg2 = 17, PrevState = 18, Dh0 = 19, Dh1 = 20, Dc0 = 21, Dc1 = 22, DhCell = 23, Count = 24;

        [ThreadStatic] private static byte[]?[]? s_slots;

        public static byte[] Get(int slot, long bytes)
        {
            var slots = s_slots ??= new byte[Count][];
            var array = slots[slot];
            if (array is null || array.LongLength != bytes) slots[slot] = array = GC.AllocateUninitializedArray<byte>(checked((int)bytes));
            return array;
        }
    }

    // Each step's input state: h0, then h of steps 0 … T − 2 (T·B × H).
    private static byte[] Shifted(byte[] h0, byte[] hs, int hBytes, int T)
    {
        var prev = Scratch.Get(Scratch.Prev, (long)T * hBytes);
        Buffer.BlockCopy(h0, 0, prev, 0, hBytes);
        Buffer.BlockCopy(hs, 0, prev, hBytes, (T - 1) * hBytes);
        return prev;
    }

    /// <summary>A GRU over T steps: x (T × B × D), h0 → h (T × B × H).</summary>
    public static string GruSequence<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut) where TN : struct, IQiyasFloat<TN>
    {
        long bd = (long)a.B * a.D, bh = (long)a.B * a.H;
        if (!OwnLane<TN>())
        {
            // A narrower type or a block format: widened once, h kept in the float lane across the steps, narrowed once.
            long GH = a.Gates * a.H;
            var lh = LOut<TN>(T * bh);
            string name = TN.Bits == 64
                ? GruSequence<Float64>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), lh)
                : GruSequence<Float32>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), lh);
            FromLane<TN>(lh, hOut, T * bh);
            return name;
        }

        return GruForward<TN>(a, T, x, h0, w, u, bias, biasH, hOut, null).Lane;
    }

    /// <summary>The LSTM sequence's backward: from dh (T × B × H, the loss's gradient at every step) and dc at the last step:
    /// dx (T × B × D), dh0, dc0, dW, dU, db (summed over the steps). The forward's h, c are recomputed from h0, c0.</summary>
    public static string LstmSequenceBackward<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] c0, byte[] w, byte[] u, byte[] bias, byte[] dhOut, byte[] dcLast, byte[] dx, byte[] dh0, byte[] dc0, byte[] dw, byte[] du, byte[] db) where TN : struct, IQiyasFloat<TN>
    {
        long bd = (long)a.B * a.D, bh = (long)a.B * a.H;
        if (!OwnLane<TN>())
        {
            long GH = a.Gates * a.H;
            var ldx = LOut<TN>(T * bd); var ldh0 = LOut<TN>(bh); var ldc0 = LOut<TN>(bh); var ldw = LOut<TN>(GH * a.D); var ldu = LOut<TN>(GH * a.H); var ldb = LOut<TN>(GH);
            string name = TN.Bits == 64
                ? LstmSequenceBackward<Float64>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(c0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(dhOut, T * bh), L<TN>(dcLast, bh), ldx, ldh0, ldc0, ldw, ldu, ldb)
                : LstmSequenceBackward<Float32>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(c0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(dhOut, T * bh), L<TN>(dcLast, bh), ldx, ldh0, ldc0, ldw, ldu, ldb);
            FromLane<TN>(ldx, dx, T * bd); FromLane<TN>(ldh0, dh0, bh); FromLane<TN>(ldc0, dc0, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH);
            return name;
        }

        // The forward recomputed once, each step's gates kept; per step back only the cell's backward and dh = dg · U (the
        // chain); dx, dW, dU, db over all the steps at once (SequenceWeights).
        int GW = a.Gates * a.H, hBytes = (int)TN.Bytes(bh), gBytes = BytesOf<TN>((long)a.B * GW);
        var hs = Scratch.Get(Scratch.Hs, (long)T * hBytes); var cs = Scratch.Get(Scratch.Cs, (long)T * hBytes);
        var gates = Scratch.Get(Scratch.Gates, (long)T * gBytes);
        LstmForward<TN>(a, T, x, h0, c0, w, u, bias, hs, cs, gates);
        var dgAll = Scratch.Get(Scratch.Dg, (long)T * gBytes);
        var dhNext = new byte[hBytes]; var dcNext = (byte[])dcLast.Clone();
        // The step's buffers once; dh and dc alternate between two (a step reads the next step's).
        var dht = Scratch.Get(Scratch.Dh, hBytes); var g = Scratch.Get(Scratch.BackG, gBytes); var dg = Scratch.Get(Scratch.BackDg, gBytes); var cPrev = Scratch.Get(Scratch.PrevState, hBytes);
        byte[][] dhb = [Scratch.Get(Scratch.Dh0, hBytes), Scratch.Get(Scratch.Dh1, hBytes)], dcb = [Scratch.Get(Scratch.Dc0, hBytes), Scratch.Get(Scratch.Dc1, hBytes)];
        string lane = "";
        for (int t = T - 1; t >= 0; t--)
        {
            Buffer.BlockCopy(dhOut, t * hBytes, dht, 0, hBytes);
            Map<TN>("add", dht, dhNext, null, dht, (int)bh, 0, 0);                        // the loss's gradient plus the next step's
            Buffer.BlockCopy(gates, t * gBytes, g, 0, gBytes);
            if (t > 0) Buffer.BlockCopy(cs, (t - 1) * hBytes, cPrev, 0, hBytes);
            var dcp = dcb[t & 1];
            lane = LstmCellBackward<TN>(a, g, t == 0 ? c0 : cPrev, dht, dcNext, dg, dcp);
            Buffer.BlockCopy(dg, 0, dgAll, t * gBytes, gBytes);
            var dhp = dhb[t & 1];
            MatMulView<TN>(new ProductView(a.B, a.H, GW, false, false), dg, u, dhp);         // dg · U
            dhNext = dhp; dcNext = dcp;
        }

        SequenceWeights<TN>(a, T, dgAll, x, Shifted(h0, hs, hBytes, T), w, dx, dw, du, db);
        dhNext.CopyTo(dh0, 0); dcNext.CopyTo(dc0, 0);
        return lane;
    }

    /// <summary>The GRU sequence's backward: from dh (T × B × H, the loss's gradient at every step): dx (T × B × D), dh0, dW, dU,
    /// db, dbh (summed over the steps). The forward's h are recomputed from h0; a narrower type or a block format widened once.</summary>
    public static string GruSequenceBackward<TN>(CellArgs a, int T, byte[] x, byte[] h0, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] dhOut, byte[] dx, byte[] dh0, byte[] dw, byte[] du, byte[] db, byte[] dbh) where TN : struct, IQiyasFloat<TN>
    {
        long bd = (long)a.B * a.D, bh = (long)a.B * a.H;
        if (!OwnLane<TN>())
        {
            long GH = a.Gates * a.H;
            var ldx = LOut<TN>(T * bd); var ldh0 = LOut<TN>(bh); var ldw = LOut<TN>(GH * a.D); var ldu = LOut<TN>(GH * a.H); var ldb = LOut<TN>(GH); var ldbh = LOut<TN>(GH);
            string name = TN.Bits == 64
                ? GruSequenceBackward<Float64>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), L<TN>(dhOut, T * bh), ldx, ldh0, ldw, ldu, ldb, ldbh)
                : GruSequenceBackward<Float32>(a, T, L<TN>(x, T * bd), L<TN>(h0, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), L<TN>(dhOut, T * bh), ldx, ldh0, ldw, ldu, ldb, ldbh);
            FromLane<TN>(ldx, dx, T * bd); FromLane<TN>(ldh0, dh0, bh); FromLane<TN>(ldw, dw, GH * a.D); FromLane<TN>(ldu, du, GH * a.H); FromLane<TN>(ldb, db, GH); FromLane<TN>(ldbh, dbh, GH);
            return name;
        }

        // The forward recomputed once, gx (all steps) and each step's gh kept; per step back only the cell's backward and
        // dh = dgh · U + the cell's direct part; dx, dW, db from dgx and dU, dbh from dgh over all the steps at once.
        int GW = a.Gates * a.H, hBytes = (int)TN.Bytes(bh), gBytes = BytesOf<TN>((long)a.B * GW);
        var hs = Scratch.Get(Scratch.Hs, (long)T * hBytes);
        var ghAll = Scratch.Get(Scratch.Gates, (long)T * gBytes);
        var (_, gx) = GruForward<TN>(a, T, x, h0, w, u, bias, biasH, hs, ghAll);
        var dgxAll = Scratch.Get(Scratch.Dg, (long)T * gBytes); var dghAll = Scratch.Get(Scratch.Dg2, (long)T * gBytes);
        var dhNext = new byte[hBytes];
        // The step's buffers once; dh alternates between two (a step reads the next step's).
        var dht = Scratch.Get(Scratch.Dh, hBytes); var gxt = Scratch.Get(Scratch.BackG, gBytes); var ght = Scratch.Get(Scratch.BackG2, gBytes); var hPrev = Scratch.Get(Scratch.PrevState, hBytes);
        var dgx = Scratch.Get(Scratch.BackDg, gBytes); var dgh = Scratch.Get(Scratch.BackDg2, gBytes); var dhCell = Scratch.Get(Scratch.DhCell, hBytes);
        byte[][] dhb = [Scratch.Get(Scratch.Dh0, hBytes), Scratch.Get(Scratch.Dh1, hBytes)];
        string lane = "";
        for (int t = T - 1; t >= 0; t--)
        {
            Buffer.BlockCopy(dhOut, t * hBytes, dht, 0, hBytes);
            Map<TN>("add", dht, dhNext, null, dht, (int)bh, 0, 0);                        // the loss's gradient plus the next step's
            Buffer.BlockCopy(gx, t * gBytes, gxt, 0, gBytes);
            Buffer.BlockCopy(ghAll, t * gBytes, ght, 0, gBytes);
            if (t > 0) Buffer.BlockCopy(hs, (t - 1) * hBytes, hPrev, 0, hBytes);
            lane = GruCellBackward<TN>(a, gxt, ght, t == 0 ? h0 : hPrev, dht, dgx, dgh, dhCell);
            Buffer.BlockCopy(dgx, 0, dgxAll, t * gBytes, gBytes);
            Buffer.BlockCopy(dgh, 0, dghAll, t * gBytes, gBytes);
            var dhp = dhb[t & 1];
            MatMulView<TN>(new ProductView(a.B, a.H, GW, false, false), dgh, u, dhp);        // dgh · U
            Map<TN>("add", dhp, dhCell, null, dhp, (int)bh, 0, 0);
            dhNext = dhp;
        }

        var hArgs = new CellArgs("gru", a.B, a.H, a.H);
        SequenceWeights<TN>(a, T, dgxAll, x, null, w, dx, dw, null, db);
        SequenceWeights<TN>(hArgs, T, dghAll, Shifted(h0, hs, hBytes, T), null, u, null, du, null, dbh);
        dhNext.CopyTo(dh0, 0);
        return lane;
    }

    /// <summary>A GRU step: h' from x, h and W (3H × D), U (3H × H), b (3H), bh (3H).</summary>
    public static string GruStep<TN>(CellArgs a, byte[] x, byte[] h, byte[] w, byte[] u, byte[] bias, byte[] biasH, byte[] hOut) where TN : struct, IQiyasFloat<TN>
    {
        if (!OwnLane<TN>())
        {
            long bh = (long)a.B * a.H, GH = a.Gates * a.H;
            var lh = LOut<TN>(bh);
            string lane = TN.Bits == 64
                ? GruStep<Float64>(a, L<TN>(x, (long)a.B * a.D), L<TN>(h, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), lh)
                : GruStep<Float32>(a, L<TN>(x, (long)a.B * a.D), L<TN>(h, bh), L<TN>(w, GH * a.D), L<TN>(u, GH * a.H), L<TN>(bias, GH), L<TN>(biasH, GH), lh);
            FromLane<TN>(lh, hOut, bh);
            return lane;
        }

        var gx = new byte[BytesOf<TN>(a.GateValues)];
        var gh = new byte[gx.Length];
        CellGates<TN>(a, x, w, null, null, bias, gx);
        CellGates<TN>(new CellArgs("gru", a.B, a.H, a.H), h, u, null, null, biasH, gh);
        return GruCell<TN>(a, gx, gh, h, hOut);
    }
}
