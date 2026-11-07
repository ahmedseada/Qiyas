// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device-neutral part of Recurrent: its arguments, catalogue and references. The devices implement it in their
// folders: Cpu/Recurrent.cs; Cuda/GpuOps.Recurrent*.cs with its kernels in Cuda/Ptx.Recurrent.cs.

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Adad.Abstraction;

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
