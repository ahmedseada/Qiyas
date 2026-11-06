// A recurrent step's inputs and float64 references (the gate's and the bench's).
namespace Qiyas.Bench;

public sealed class RecurrentCase
{
    public required byte[] X, H, W, U, Bias;
    public byte[]? C, BiasH, COut, DhOut, DcOut, Dh, Dc, Dw, Du, Db, Dbh;
    public required double[] WantH;
    public double[]? WantC, WantDx, WantDh, WantDc, WantDw, WantDu, WantDb, WantDbh;

    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    /// <summary>The float64 gates of one step: gx = x·Wᵀ + b, gh = h·Uᵀ + bh (bh zero when null).</summary>
    public static (double[] Gx, double[] Gh) Gates(CellArgs a, double[] xv, double[] hv, double[] wv, double[] uv, double[] bv, double[]? bhv)
    {
        int GH = a.Gates * a.H;
        var gx = new double[(long)a.B * GH];
        var gh = new double[(long)a.B * GH];
        for (int b = 0; b < a.B; b++) for (int k = 0; k < GH; k++)
        {
            double sx = bv[k], sh = bhv is null ? 0 : bhv[k];
            for (int d = 0; d < a.D; d++) sx += xv[b * a.D + d] * wv[k * a.D + d];
            for (int j = 0; j < a.H; j++) sh += hv[b * a.H + j] * uv[k * a.H + j];
            gx[(long)b * GH + k] = sx; gh[(long)b * GH + k] = sh;
        }

        return (gx, gh);
    }

    /// <summary>A GRU sequence in float64: h per step (T × B × H), the step's gates gx = x·Wᵀ + b, gh = h·Uᵀ + bh, then the cell.</summary>
    public static double[] GruSequenceReference(CellArgs a, int T, double[] x, double[] h0, double[] w, double[] u, double[] b, double[] bh)
    {
        int bd = a.B * a.D, bhn = a.B * a.H;
        var hs = new double[T * bhn];
        var hPrev = h0;
        for (int t = 0; t < T; t++)
        {
            var (gx, gh) = Gates(a, x.AsSpan(t * bd, bd).ToArray(), hPrev, w, u, b, bh);
            var ht = new double[bhn];
            a.GruCell(gx, gh, hPrev, ht);
            Array.Copy(ht, 0, hs, t * bhn, bhn);
            hPrev = ht;
        }

        return hs;
    }

    /// <summary>An LSTM sequence in float64: h, c per step; and its backward from dh per step and dc at the last: dx, dh0, dc0, dW, dU, db.</summary>
    public static (double[] H, double[] C) LstmSequenceReference(CellArgs a, int T, double[] x, double[] h0, double[] c0, double[] w, double[] u, double[] b)
    {
        int bd = a.B * a.D, bh = a.B * a.H;
        var hs = new double[T * bh]; var cs = new double[T * bh];
        var hPrev = h0; var cPrev = c0;
        for (int t = 0; t < T; t++)
        {
            var xt = x.AsSpan(t * bd, bd).ToArray();
            var (gx, gh) = Gates(a, xt, hPrev, w, u, b, null);
            var g = new double[gx.Length];
            for (int i = 0; i < g.Length; i++) g[i] = gx[i] + gh[i];
            var ht = new double[bh]; var ct = new double[bh];
            a.LstmCell(g, cPrev, ht, ct);
            Array.Copy(ht, 0, hs, t * bh, bh); Array.Copy(ct, 0, cs, t * bh, bh);
            hPrev = ht; cPrev = ct;
        }

        return (hs, cs);
    }

    public static (double[] DX, double[] DH0, double[] DC0, double[] DW, double[] DU, double[] DB) LstmSequenceBackwardReference(CellArgs a, int T, double[] x, double[] h0, double[] c0, double[] w, double[] u, double[] b, double[] dhOut, double[] dcLast)
    {
        int bd = a.B * a.D, bh = a.B * a.H, GH = a.Gates * a.H;
        var (hs, cs) = LstmSequenceReference(a, T, x, h0, c0, w, u, b);
        var dx = new double[T * bd]; var dw = new double[GH * a.D]; var du = new double[GH * a.H]; var db = new double[GH];
        var dhNext = new double[bh]; var dcNext = (double[])dcLast.Clone();
        for (int t = T - 1; t >= 0; t--)
        {
            var xt = x.AsSpan(t * bd, bd).ToArray();
            var hPrev = t == 0 ? h0 : hs.AsSpan((t - 1) * bh, bh).ToArray();
            var cPrev = t == 0 ? c0 : cs.AsSpan((t - 1) * bh, bh).ToArray();
            var (gx, gh) = Gates(a, xt, hPrev, w, u, b, null);
            var g = new double[gx.Length];
            for (int i = 0; i < g.Length; i++) g[i] = gx[i] + gh[i];
            var dht = new double[bh];
            for (int i = 0; i < bh; i++) dht[i] = dhOut[t * bh + i] + dhNext[i];
            var dg = new double[g.Length]; var dc = new double[bh];
            a.LstmCellBackward(g, cPrev, dht, dcNext, dg, dc);
            var dhp = new double[bh];
            for (int bb = 0; bb < a.B; bb++) for (int k = 0; k < GH; k++)
            {
                double v = dg[(long)bb * GH + k];
                for (int d = 0; d < a.D; d++) { dx[t * bd + bb * a.D + d] += v * w[k * a.D + d]; dw[k * a.D + d] += v * xt[bb * a.D + d]; }
                for (int j = 0; j < a.H; j++) { dhp[bb * a.H + j] += v * u[k * a.H + j]; du[k * a.H + j] += v * hPrev[bb * a.H + j]; }
                db[k] += v;
            }

            dhNext = dhp; dcNext = dc;
        }

        return (dx, dhNext, dcNext, dw, du, db);
    }

    /// <summary>The GRU sequence's backward in float64, from dh per step (T × B × H): dx (T × B × D), dh0, dW, dU, db, dbh
    /// (summed over the steps). Per step: gx = x·Wᵀ + b, gh = h·Uᵀ + bh; the cell's backward gives dgx, dgh and its direct dh;
    /// dx = dgx·W, dW += dgxᵀ·x, db += Σ dgx; dh_prev = dh_cell + dgh·U, dU += dghᵀ·h, dbh += Σ dgh.</summary>
    public static (double[] DX, double[] DH0, double[] DW, double[] DU, double[] DB, double[] DBH) GruSequenceBackwardReference(CellArgs a, int T, double[] x, double[] h0, double[] w, double[] u, double[] b, double[] bh, double[] dhOut)
    {
        int bd = a.B * a.D, bhn = a.B * a.H, GH = a.Gates * a.H;
        var hs = GruSequenceReference(a, T, x, h0, w, u, b, bh);
        var dx = new double[T * bd]; var dw = new double[GH * a.D]; var du = new double[GH * a.H]; var db = new double[GH]; var dbh = new double[GH];
        var dhNext = new double[bhn];
        for (int t = T - 1; t >= 0; t--)
        {
            var xt = x.AsSpan(t * bd, bd).ToArray();
            var hPrev = t == 0 ? h0 : hs.AsSpan((t - 1) * bhn, bhn).ToArray();
            var (gx, gh) = Gates(a, xt, hPrev, w, u, b, bh);
            var dht = new double[bhn];
            for (int i = 0; i < bhn; i++) dht[i] = dhOut[t * bhn + i] + dhNext[i];
            var dgx = new double[gx.Length]; var dgh = new double[gx.Length]; var dhp = new double[bhn];
            a.GruCellBackward(gx, gh, hPrev, dht, dgx, dgh, dhp);
            for (int bb = 0; bb < a.B; bb++) for (int k = 0; k < GH; k++)
            {
                double v = dgx[(long)bb * GH + k], v2 = dgh[(long)bb * GH + k];
                for (int d = 0; d < a.D; d++) { dx[t * bd + bb * a.D + d] += v * w[k * a.D + d]; dw[k * a.D + d] += v * xt[bb * a.D + d]; }
                for (int j = 0; j < a.H; j++) { dhp[bb * a.H + j] += v2 * u[k * a.H + j]; du[k * a.H + j] += v2 * hPrev[bb * a.H + j]; }
                db[k] += v; dbh[k] += v2;
            }

            dhNext = dhp;
        }

        return (dx, dhNext, dw, du, db, dbh);
    }

    public static RecurrentCase Make(CellArgs a, string type, bool backward)
    {
        bool lstm = a.Kind == "lstm";
        int GH = a.Gates * a.H;
        var x = Registry.Make(type, a.B * a.D, 81, -1, 1);
        var h = Registry.Make(type, a.B * a.H, 82, -1, 1);
        var c = lstm ? Registry.Make(type, a.B * a.H, 83, -1, 1) : null;
        // Weights scaled so the gates stay in the sigmoids' useful range.
        double scale = 1 / Math.Sqrt(a.D + a.H);
        var w = Registry.Make(type, GH * a.D, 84, -scale, scale);
        var u = Registry.Make(type, GH * a.H, 85, -scale, scale);
        var bias = Registry.Make(type, GH, 86, -0.1, 0.1);
        var biasH = lstm ? null : Registry.Make(type, GH, 87, -0.1, 0.1);
        var xv = Registry.Values(type, x, a.B * a.D); var hv = Registry.Values(type, h, a.B * a.H); var cv = c is null ? null : Registry.Values(type, c, a.B * a.H);
        var wv = Registry.Values(type, w, GH * a.D); var uv = Registry.Values(type, u, GH * a.H); var bv = Registry.Values(type, bias, GH); var bhv = biasH is null ? null : Registry.Values(type, biasH, GH);
        // The gates in float64.
        var gx = new double[(long)a.B * GH];
        var gh = new double[(long)a.B * GH];
        for (int b = 0; b < a.B; b++) for (int k = 0; k < GH; k++)
        {
            double sx = bv[k], sh = bhv is null ? 0 : bhv[k];
            for (int d = 0; d < a.D; d++) sx += xv[b * a.D + d] * wv[k * a.D + d];
            for (int j = 0; j < a.H; j++) sh += hv[b * a.H + j] * uv[k * a.H + j];
            gx[(long)b * GH + k] = sx; gh[(long)b * GH + k] = sh;
        }

        var g = new double[gx.Length];
        for (int i = 0; i < g.Length; i++) g[i] = gx[i] + gh[i];
        var wantH = new double[a.B * a.H];
        var wantC = lstm ? new double[a.B * a.H] : null;
        if (lstm) a.LstmCell(g, cv!, wantH, wantC!); else a.GruCell(gx, gh, hv, wantH);
        var k0 = new RecurrentCase { X = x, H = h, C = c, W = w, U = u, Bias = bias, BiasH = biasH, WantH = wantH, WantC = wantC, COut = lstm ? new byte[h.Length] : null };
        if (!backward) return k0;
        var dhOut = Registry.Make(type, a.B * a.H, 88, -1, 1);
        var dcOut = lstm ? Registry.Make(type, a.B * a.H, 89, -1, 1) : null;
        var dhv = Registry.Values(type, dhOut, a.B * a.H); var dcv = dcOut is null ? null : Registry.Values(type, dcOut, a.B * a.H);
        var dg = new double[g.Length];
        var dg2 = new double[g.Length];
        var dstate = new double[a.B * a.H];
        if (lstm) a.LstmCellBackward(g, cv!, dhv, dcv!, dg, dstate); else a.GruCellBackward(gx, gh, hv, dhv, dg, dg2, dstate);
        // dx = dg·W, dh = dg·U (+ the GRU cell's dh), dW = dgᵀ·x, dU = dgᵀ·h (GRU: dg2 for the h side), db = Σ rows.
        var dx = new double[a.B * a.D]; var dh = new double[a.B * a.H]; var dw = new double[GH * a.D]; var du = new double[GH * a.H]; var db = new double[GH]; var dbh = lstm ? null : new double[GH];
        var dgh = lstm ? dg : dg2;
        for (int b = 0; b < a.B; b++) for (int k = 0; k < GH; k++)
        {
            double v = dg[(long)b * GH + k], v2 = dgh[(long)b * GH + k];
            for (int d = 0; d < a.D; d++) { dx[b * a.D + d] += v * wv[k * a.D + d]; dw[k * a.D + d] += v * xv[b * a.D + d]; }
            for (int j = 0; j < a.H; j++) { dh[b * a.H + j] += v2 * uv[k * a.H + j]; du[k * a.H + j] += v2 * hv[b * a.H + j]; }
            db[k] += v;
            if (dbh is not null) dbh[k] += v2;
        }

        if (!lstm) for (int i = 0; i < dh.Length; i++) dh[i] += dstate[i];
        k0.DhOut = dhOut; k0.DcOut = dcOut;
        k0.Dh = new byte[h.Length]; k0.Dc = lstm ? new byte[h.Length] : null; k0.Dw = new byte[w.Length]; k0.Du = new byte[u.Length]; k0.Db = new byte[bias.Length]; k0.Dbh = lstm ? null : new byte[bias.Length];
        k0.WantDx = dx; k0.WantDh = dh; k0.WantDc = lstm ? dstate : null; k0.WantDw = dw; k0.WantDu = du; k0.WantDb = db; k0.WantDbh = dbh;
        return k0;
    }
}
