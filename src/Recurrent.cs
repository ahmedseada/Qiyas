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
                    for (int j = 0; j < H; j++)
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

        int xBytes = (int)TN.Bytes(bd), hBytes = (int)TN.Bytes(bh);
        var hPrev = h0; var cPrev = c0;
        string lane = "";
        for (int t = 0; t < T; t++)
        {
            var xt = new byte[xBytes]; Array.Copy(x, (long)t * xBytes, xt, 0, xBytes);
            var ht = new byte[hBytes]; var ct = new byte[hBytes];
            lane = LstmStep<TN>(a, xt, hPrev, cPrev, w, u, bias, ht, ct);
            Array.Copy(ht, 0, hOut, (long)t * hBytes, hBytes);
            Array.Copy(ct, 0, cOut, (long)t * hBytes, hBytes);
            hPrev = ht; cPrev = ct;
        }

        return lane;
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

        int xBytes = (int)TN.Bytes(bd), hBytes = (int)TN.Bytes(bh);
        var hPrev = h0;
        string lane = "";
        for (int t = 0; t < T; t++)
        {
            var xt = new byte[xBytes]; Array.Copy(x, (long)t * xBytes, xt, 0, xBytes);
            var ht = new byte[hBytes];
            lane = GruStep<TN>(a, xt, hPrev, w, u, bias, biasH, ht);
            Array.Copy(ht, 0, hOut, (long)t * hBytes, hBytes);
            hPrev = ht;
        }

        return lane;
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

        int xBytes = (int)TN.Bytes(bd), hBytes = (int)TN.Bytes(bh);
        var hs = new byte[T][]; var cs = new byte[T][];
        var hPrev = h0; var cPrev = c0;
        for (int t = 0; t < T; t++)
        {
            var xt = new byte[xBytes]; Array.Copy(x, (long)t * xBytes, xt, 0, xBytes);
            hs[t] = new byte[hBytes]; cs[t] = new byte[hBytes];
            LstmStep<TN>(a, xt, hPrev, cPrev, w, u, bias, hs[t], cs[t]);
            hPrev = hs[t]; cPrev = cs[t];
        }

        var dhNext = new byte[hBytes]; var dcNext = (byte[])dcLast.Clone();
        var dwSum = new byte[dw.Length]; var duSum = new byte[du.Length]; var dbSum = new byte[db.Length];
        string lane = "";
        for (int t = T - 1; t >= 0; t--)
        {
            var xt = new byte[xBytes]; Array.Copy(x, (long)t * xBytes, xt, 0, xBytes);
            var dht = new byte[hBytes]; Array.Copy(dhOut, (long)t * hBytes, dht, 0, hBytes);
            Map<TN>("add", dht, dhNext, null, dht, (int)bh, 0, 0);                        // the loss's gradient plus the next step's
            var dxt = new byte[xBytes]; var dhp = new byte[hBytes]; var dcp = new byte[hBytes];
            var dwt = new byte[dw.Length]; var dut = new byte[du.Length]; var dbt = new byte[db.Length];
            lane = LstmStepBackward<TN>(a, xt, t == 0 ? h0 : hs[t - 1], t == 0 ? c0 : cs[t - 1], w, u, bias, dht, dcNext, dxt, dhp, dcp, dwt, dut, dbt);
            Array.Copy(dxt, 0, dx, (long)t * xBytes, xBytes);
            Map<TN>("add", dwSum, dwt, null, dwSum, (int)(a.Gates * a.H * (long)a.D), 0, 0);
            Map<TN>("add", duSum, dut, null, duSum, (int)(a.Gates * a.H * (long)a.H), 0, 0);
            Map<TN>("add", dbSum, dbt, null, dbSum, a.Gates * a.H, 0, 0);
            dhNext = dhp; dcNext = dcp;
        }

        dhNext.CopyTo(dh0, 0); dcNext.CopyTo(dc0, 0); dwSum.CopyTo(dw, 0); duSum.CopyTo(du, 0); dbSum.CopyTo(db, 0);
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
