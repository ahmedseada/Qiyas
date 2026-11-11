// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The device side of reductions, row-vector broadcasts, transposes and matrix-vector products. Values are addressed
// by index (Codec.UnpackAt), so every shape works for every format, packed ones included; rows need not end on a word.
using Adad.Abstraction;

namespace Adad.Cuda;

internal static partial class Ptx
{
    public const int ReduceThreads = 256;

    /// <summary>A lane's values as a stored format (partial results live in lane arrays).</summary>
    public static Fmt LaneFormat(Lane lane) => lane.Name switch
    {
        "f32" => new Fmt("float32", 32, true, true),
        "f64" => new Fmt("float64", 64, true, true),
        "s32" => new Fmt("int32", 32, false, true),
        "u32" => new Fmt("uint32", 32, false, false),
        "s64" => new Fmt("int64", 64, false, true),
        _ => new Fmt("uint64", 64, false, false),
    };

    private static string Combine(Lane lane, string op, string d, string a, string b) => op switch
    {
        "max" => $"max.{lane.Op} {d}, {a}, {b};",
        "min" => $"min.{lane.Op} {d}, {a}, {b};",
        _ => lane.Add(d, a, b),
    };

    private static string Identity(Lane lane, string op) => ReduceIdentity(lane, op);

    // ---------------------------------------------------------------- per row (a block a row) and whole (blocks over chunks)

    /// <summary>out[row] = op over x[row·cols + first .. ] for `count` values (a block per row; rows = %ctaid.x); the
    /// block's chunk when p_chunk > 0: row r covers values [r·chunk, min(n, (r+1)·chunk)) of one long row.</summary>
    /// <param name="compensated">A float sum's chunk pass (lane f32): each thread's running sum compensated (Kahan: the
    /// rounding lost by every add kept and taken back), the threads' sums joined in float64 and stored as f64 partials.</param>
    /// <param name="storeF32">With <paramref name="compensated"/>: the threads' sums joined as exact f32 pairs (no f64) and
    /// rounded once to the f32 lane (a row's sum, not a partial).</param>
    /// <param name="table">The chunk pass of several arrays in one launch: p_x is a table on the device (Ptx.TableRow) of
    /// p_cols rows — an array's first block, its values, its chunk, where it is — and the block is the array's its index
    /// falls in; the partials at p_out in block order.</param>
    /// <summary>A sum table's row: the array's first block, its values, its chunk, where it is.</summary>
    public const int SumRow = 4;

    public static string ReduceRows(Gpu gpu, Fmt f, Lane lane, string op, bool squares, out string entry, bool compensated = false, bool storeF32 = false, bool table = false)
    {
        entry = $"reduce_rows_{op}{(squares ? "sq" : "")}_{f.Name}_{lane.Name}{(compensated ? "_kahan" : "")}{(storeF32 ? "_f32" : "")}{(table ? "_tab" : "")}";
        if (storeF32 && !compensated) throw new ArgumentException("the f32 store is the compensated pass's");
        if (table && storeF32) throw new ArgumentException("the table is the chunk pass's");
        if (compensated && (lane.Name != "f32" || op != "sum")) throw new ArgumentException("the compensated pass is a float sum in the f32 lane");
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_out"), cols = Param32(b, "p_cols"), chunk = Param32(b, "p_chunk");
        if (table)
        {
            var (tr, local) = TableRow(b, src, cols, row, SumRow);
            string t = b.RD();
            b.L($"ld.global.u64 {t}, [{tr}];");
            b.L($"mul.lo.u64 {t}, {t}, {(compensated ? 8 : lane.Size)};");
            b.L($"add.u64 {dst}, {dst}, {t};");
            b.L($"ld.global.u64 {t}, [{tr}+8];");
            b.L($"cvt.u32.u64 {cols}, {t};");
            b.L($"ld.global.u64 {t}, [{tr}+16];");
            b.L($"cvt.u32.u64 {chunk}, {t};");
            b.L($"ld.global.u64 {src}, [{tr}+24];");
            b.L($"mov.u32 {row}, {local};");
        }

        string start = b.R(), count = b.R(), pChunk = b.P(), end = b.R();
        b.L($"setp.ne.u32 {pChunk}, {chunk}, 0;");
        b.L($"mul.lo.u32 {start}, {row}, {cols};");
        b.L($"mov.u32 {count}, {cols};");
        string cs = b.R(), ce = b.R(), cc = b.R();
        b.L($"mul.lo.u32 {cs}, {row}, {chunk};");
        b.L($"add.u32 {ce}, {cs}, {chunk};");
        b.L($"min.u32 {ce}, {ce}, {cols};");
        b.L($"sub.u32 {cc}, {ce}, {cs};");
        b.L($"selp.u32 {start}, {cs}, {start}, {pChunk};");
        b.L($"selp.u32 {count}, {cc}, {count}, {pChunk};");
        string acc = lane.NewReg(b), i = b.R(), pEnd = b.P();
        string comp = compensated ? b.F() : "", ky = compensated ? b.F() : "", kt = compensated ? b.F() : "";
        b.L($"mov.{lane.Reg} {acc}, {Identity(lane, op)};");
        if (compensated) b.L($"mov.f32 {comp}, 0f00000000;");
        b.L($"mov.u32 {i}, {tid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {i}, {count};");
        b.L($"@{pEnd} bra DONE;");
        string idx = b.R();
        b.L($"add.u32 {idx}, {start}, {i};");
        string v = Codec.UnpackAt(b, f, src, idx, lane);
        if (compensated)
        {
            // y = term − c; t = acc + y; c = (t − acc) − y; acc = t (each step rounded as written: no contraction).
            if (squares) b.L($"mul.rn.f32 {ky}, {v}, {v};");
            else b.L($"mov.f32 {ky}, {v};");
            b.L($"sub.rn.f32 {ky}, {ky}, {comp};");
            b.L($"add.rn.f32 {kt}, {acc}, {ky};");
            b.L($"sub.rn.f32 {comp}, {kt}, {acc};");
            b.L($"sub.rn.f32 {comp}, {comp}, {ky};");
            b.L($"mov.f32 {acc}, {kt};");
        }
        else if (squares) b.L(lane.Fma(acc, v, v, acc));
        else b.L(Combine(lane, op, acc, acc, v));
        b.L($"add.u32 {i}, {i}, {ReduceThreads};");
        b.L("bra LOOP;");
        b.Label("DONE");
        _ = end;
        if (storeF32)
        {
            // A row's sum: the threads' pairs acc − c joined exactly in f32 pairs, rounded once to f32.
            string negComp = b.F();
            b.L($"neg.f32 {negComp}, {comp};");
            var (ph, pl) = BlockReducePair(b, tid, acc, negComp, ReduceThreads);
            string rowSum = b.F(), pz = b.P(), rAddr = b.RD();
            b.L($"add.rn.f32 {rowSum}, {ph}, {pl};");
            b.L($"setp.ne.u32 {pz}, {tid}, 0;");
            b.L($"@{pz} bra OUT;");
            b.L($"mul.wide.u32 {rAddr}, {row}, 4;");
            b.L($"add.u64 {rAddr}, {dst}, {rAddr};");
            b.L($"st.global.f32 [{rAddr}], {rowSum};");
            b.Label("OUT");
            b.L("ret;");
            return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_cols, .param .u32 p_chunk", ReduceBytes);
        }

        if (compensated)
        {
            // The thread's sum in float64: acc − c, then the block's sum in float64, stored as an f64 partial.
            string wideAcc = b.FD(), wideComp = b.FD();
            b.L($"cvt.f64.f32 {wideAcc}, {acc};");
            b.L($"cvt.f64.f32 {wideComp}, {comp};");
            b.L($"sub.rn.f64 {wideAcc}, {wideAcc}, {wideComp};");
            acc = wideAcc;
            lane = Lane.F64;
        }

        string total = BlockReduce(b, lane, tid, acc, op == "max" ? "max" : op == "min" ? "min" : "add", ReduceThreads);
        string p0 = b.P(), oAddr = b.RD();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra OUT;");
        b.L($"mul.wide.u32 {oAddr}, {row}, {lane.Size};");
        b.L($"add.u64 {oAddr}, {dst}, {oAddr};");
        b.L($"st.global.{lane.Mem} [{oAddr}], {total};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_cols, .param .u32 p_chunk", ReduceBytes);
    }

    /// <summary>out[row] = the sum (of squares) of row `row` of a float format in 32-bit words, cols a whole number of words:
    /// a block a row, each thread over whole words (one load for every value the word holds), its running sum compensated
    /// (Kahan), the threads' sums joined as exact f32 pairs and rounded once to f32.</summary>
    public static string SumRowsWords(Gpu gpu, Fmt f, bool squares, out string entry)
    {
        entry = $"sum_rows_words_{(squares ? "sq" : "")}_{f.Name}";
        if (!f.IsFloat || f.IsBlock || f.WordBits != 32) throw new ArgumentException("a plain float format in 32-bit words");
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_out"), cols = Param32(b, "p_cols");
        _ = Param32(b, "p_chunk");
        string words = b.R(), w = b.R(), pEnd = b.P(), rowBase = b.RD(), off = b.RD();
        b.L($"div.u32 {words}, {cols}, {f.Per};");
        b.L($"mul.wide.u32 {rowBase}, {row}, {words};");
        b.L($"shl.b64 {rowBase}, {rowBase}, 2;");
        b.L($"add.u64 {rowBase}, {src}, {rowBase};");
        string acc = b.F(), comp = b.F(), ky = b.F(), kt = b.F();
        b.L($"mov.f32 {acc}, 0f00000000;");
        b.L($"mov.f32 {comp}, 0f00000000;");
        b.L($"mov.u32 {w}, {tid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {w}, {words};");
        b.L($"@{pEnd} bra DONE;");
        string word = b.R();
        b.L($"mul.wide.u32 {off}, {w}, 4;");
        b.L($"add.u64 {off}, {rowBase}, {off};");
        b.L($"ld.global.b32 {word}, [{off}];");
        for (int i = 0; i < f.Per; i++)
        {
            string v = Codec.Unpack(b, f, word, i, Lane.F32);
            // y = term − c; t = acc + y; c = (t − acc) − y; acc = t.
            if (squares) b.L($"mul.rn.f32 {ky}, {v}, {v};");
            else b.L($"mov.f32 {ky}, {v};");
            b.L($"sub.rn.f32 {ky}, {ky}, {comp};");
            b.L($"add.rn.f32 {kt}, {acc}, {ky};");
            b.L($"sub.rn.f32 {comp}, {kt}, {acc};");
            b.L($"sub.rn.f32 {comp}, {comp}, {ky};");
            b.L($"mov.f32 {acc}, {kt};");
        }

        b.L($"add.u32 {w}, {w}, {ReduceThreads};");
        b.L("bra LOOP;");
        b.Label("DONE");
        string negComp = b.F();
        b.L($"neg.f32 {negComp}, {comp};");
        var (ph, pl) = BlockReducePair(b, tid, acc, negComp, ReduceThreads);
        string rowSum = b.F(), pz = b.P(), rAddr = b.RD();
        b.L($"add.rn.f32 {rowSum}, {ph}, {pl};");
        b.L($"setp.ne.u32 {pz}, {tid}, 0;");
        b.L($"@{pz} bra OUT;");
        b.L($"mul.wide.u32 {rAddr}, {row}, 4;");
        b.L($"add.u64 {rAddr}, {dst}, {rAddr};");
        b.L($"st.global.f32 [{rAddr}], {rowSum};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_cols, .param .u32 p_chunk", ReduceBytes);
    }

    /// <summary>out[col] = the row index of the column's first largest value (s32), a thread per column.</summary>
    public static string ArgmaxCols(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"argmax_cols_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), col = b.R(), src = Param64(b, "p_x"), dst = Param64(b, "p_out"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {col}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {col}, {cols};");
        b.L($"@{pEnd} bra DONE;");
        string r = b.R(), best = lane.NewReg(b), bestAt = b.R(), idx = b.R(), pR = b.P(), pGt = b.P(), pFirst = b.P();
        b.L($"mov.{lane.Reg} {best}, {Identity(lane, "max")};");
        b.L($"mov.u32 {bestAt}, 0xFFFFFFFF;");
        b.L($"mov.u32 {r}, 0;");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pR}, {r}, {rows};");
        b.L($"@{pR} bra END;");
        b.L($"mad.lo.u32 {idx}, {r}, {cols}, {col};");
        string v = Codec.UnpackAt(b, f, src, idx, lane);
        b.L($"setp.gt.{lane.Op} {pGt}, {v}, {best};");
        b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
        b.L($"or.pred {pGt}, {pGt}, {pFirst};");
        b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {best}, {v}, {best}, {pGt};");
        b.L($"selp.u32 {bestAt}, {r}, {bestAt}, {pGt};");
        b.L($"add.u32 {r}, {r}, 1;");
        b.L("bra LOOP;");
        b.Label("END");
        string addr = b.RD();
        b.L($"mul.wide.u32 {addr}, {col}, 4;");
        b.L($"add.u64 {addr}, {dst}, {addr};");
        b.L($"st.global.s32 [{addr}], {bestAt};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_rows, .param .u32 p_cols");
    }

    /// <summary>out[row] = the index of the first largest value of the row (s32), a block per row.</summary>
    public static string ArgmaxRows(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"argmax_rows_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {row}, %ctaid.x;");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_out"), cols = Param32(b, "p_cols");
        string start = b.R(), best = lane.NewReg(b), bestAt = b.R(), i = b.R(), pEnd = b.P();
        b.L($"mul.lo.u32 {start}, {row}, {cols};");
        b.L($"mov.{lane.Reg} {best}, {Identity(lane, "max")};");
        b.L($"mov.u32 {bestAt}, 0xFFFFFFFF;");
        b.L($"mov.u32 {i}, {tid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {i}, {cols};");
        b.L($"@{pEnd} bra DONE;");
        string idx = b.R(), pGt = b.P(), pFirst = b.P();
        b.L($"add.u32 {idx}, {start}, {i};");
        string v = Codec.UnpackAt(b, f, src, idx, lane);
        b.L($"setp.gt.{lane.Op} {pGt}, {v}, {best};");
        b.L($"setp.eq.u32 {pFirst}, {bestAt}, 0xFFFFFFFF;");
        b.L($"or.pred {pGt}, {pGt}, {pFirst};");
        b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {best}, {v}, {best}, {pGt};");
        b.L($"selp.u32 {bestAt}, {i}, {bestAt}, {pGt};");
        b.L($"add.u32 {i}, {i}, {ReduceThreads};");
        b.L("bra LOOP;");
        b.Label("DONE");
        // The block's best: a shared tree over (value, index) pairs; ties keep the lower index.
        string smem = b.R(), mine = b.R();
        b.L($"mov.u32 {smem}, smem;");
        b.L($"mad.lo.u32 {mine}, {tid}, 16, {smem};");
        b.L($"st.shared.{lane.Mem} [{mine}], {best};");
        b.L($"st.shared.u32 [{mine}+8], {bestAt};");
        b.L("bar.sync 0;");
        for (int s = ReduceThreads / 2; s >= 1; s /= 2)
        {
            string p = b.P(), other = b.R(), ov = lane.NewReg(b), oi = b.R(), pTake = b.P(), pEq = b.P(), pLess = b.P(), pNone = b.P(), pMineNone = b.P();
            b.L($"setp.ge.u32 {p}, {tid}, {s};");
            b.L($"@{p} bra SKIP{s};");
            b.L($"add.u32 {other}, {mine}, {s * 16};");
            b.L($"ld.shared.{lane.Mem} {ov}, [{other}];");
            b.L($"ld.shared.u32 {oi}, [{other}+8];");
            b.L($"setp.ne.u32 {pNone}, {oi}, 0xFFFFFFFF;");
            b.L($"setp.eq.u32 {pMineNone}, {bestAt}, 0xFFFFFFFF;");
            b.L($"setp.gt.{lane.Op} {pTake}, {ov}, {best};");
            b.L($"setp.eq.{lane.Op} {pEq}, {ov}, {best};");
            b.L($"setp.lt.u32 {pLess}, {oi}, {bestAt};");
            b.L($"and.pred {pEq}, {pEq}, {pLess};");
            b.L($"or.pred {pTake}, {pTake}, {pEq};");
            b.L($"or.pred {pTake}, {pTake}, {pMineNone};");
            b.L($"and.pred {pTake}, {pTake}, {pNone};");
            b.L($"selp.{(lane.IsFloat ? lane.Op : lane.Reg)} {best}, {ov}, {best}, {pTake};");
            b.L($"selp.u32 {bestAt}, {oi}, {bestAt}, {pTake};");
            b.L($"st.shared.{lane.Mem} [{mine}], {best};");
            b.L($"st.shared.u32 [{mine}+8], {bestAt};");
            b.Label($"SKIP{s}");
            b.L("bar.sync 0;");
        }

        string p0 = b.P(), oAddr = b.RD();
        b.L($"setp.ne.u32 {p0}, {tid}, 0;");
        b.L($"@{p0} bra OUT;");
        b.L($"mul.wide.u32 {oAddr}, {row}, 4;");
        b.L($"add.u64 {oAddr}, {dst}, {oAddr};");
        b.L($"st.global.u32 [{oAddr}], {bestAt};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_cols", ReduceThreads * 16);
    }

    /// <summary>out[col] = op over the column's rows (a thread per column).</summary>
    /// <summary>out[split][col] = op over the column's rows in split %ctaid.y (rows [split·per, min(rows, (split+1)·per))); a thread per column.</summary>
    public static string ReduceCols(Gpu gpu, Fmt f, Lane lane, string op, bool squares, out string entry)
    {
        entry = $"reduce_cols_{op}{(squares ? "sq" : "")}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string src = Param64(b, "p_x"), dst = Param64(b, "p_out"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols"), per = Param32(b, "p_per");
        string pEnd = b.P(), acc = lane.NewReg(b), r = b.R(), idx = b.R(), pLoop = b.P(), split = b.R(), rEnd = b.R();
        b.L($"mov.u32 {split}, %ctaid.y;");
        b.L($"setp.ge.u32 {pEnd}, {gid}, {cols};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"mov.{lane.Reg} {acc}, {Identity(lane, op)};");
        b.L($"mul.lo.u32 {r}, {split}, {per};");
        b.L($"add.u32 {rEnd}, {r}, {per};");
        b.L($"min.u32 {rEnd}, {rEnd}, {rows};");
        b.L($"mad.lo.u32 {idx}, {r}, {cols}, {gid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pLoop}, {r}, {rEnd};");
        b.L($"@{pLoop} bra DONE;");
        string v = Codec.UnpackAt(b, f, src, idx, lane);
        if (squares) b.L(lane.Fma(acc, v, v, acc));
        else b.L(Combine(lane, op, acc, acc, v));
        b.L($"add.u32 {idx}, {idx}, {cols};");
        b.L($"add.u32 {r}, {r}, 1;");
        b.L("bra LOOP;");
        b.Label("DONE");
        string oAddr = b.RD(), oIdx = b.R();
        b.L($"mad.lo.u32 {oIdx}, {split}, {cols}, {gid};");
        b.L($"mul.wide.u32 {oAddr}, {oIdx}, {lane.Size};");
        b.L($"add.u64 {oAddr}, {dst}, {oAddr};");
        b.L($"st.global.{lane.Mem} [{oAddr}], {acc};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_per");
    }

    // ---------------------------------------------------------------- output[r][c] = op(x[r][c], v[c])

    public static string MapRow(Gpu gpu, Fmt f, Lane lane, MapOp op, out string entry)
    {
        entry = $"maprow_{op.Name}_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string px = Param64(b, "p_x"), pv = Param64(b, "p_v"), po = Param64(b, "p_o"), n = Param32(b, "p_n"), cols = Param32(b, "p_cols");
        string zero = lane.NewReg(b);
        b.L($"mov.{lane.Reg} {zero}, {lane.Zero};");
        string words = b.R(), p = b.P();
        if (f.WordBits == 64) b.L($"mov.u32 {words}, {n};");
        else
        {
            string bits = b.RD(), wq = b.RD();
            b.L($"mul.wide.u32 {bits}, {n}, {f.Bits};");
            b.L($"add.u64 {bits}, {bits}, 31;");
            b.L($"shr.u64 {wq}, {bits}, 5;");
            b.L($"cvt.u32.u64 {words}, {wq};");
        }

        b.L($"setp.ge.u32 {p}, {gid}, {words};");
        b.L($"@{p} bra DONE;");
        string wt = f.WordBits == 64 ? "b64" : "b32";
        string off = b.RD(), ax = b.RD(), ao = b.RD(), wx = f.WordBits == 64 ? b.RD() : b.R(), wo = f.WordBits == 64 ? b.RD() : b.R(), first = b.R();
        b.L($"mul.wide.u32 {off}, {gid}, {f.WordBits / 8};");
        b.L($"add.u64 {ax}, {px}, {off};");
        b.L($"add.u64 {ao}, {po}, {off};");
        b.L($"ld.global.{wt} {wx}, [{ax}];");
        if (f.WordBits == 32) b.L($"mov.b32 {wo}, 0;");
        b.L($"mul.lo.u32 {first}, {gid}, {f.Per};");
        // The word's first column by one remainder, the next values' by a step that wraps at the row's end (item 6j: a
        // remainder a value before).
        string col = b.R(), pWrap = b.P();
        b.L($"rem.u32 {col}, {first}, {cols};");
        for (int i = 0; i < f.Per; i++)
        {
            string x = Codec.Unpack(b, f, wx, i, lane);
            if (i > 0)
            {
                b.L($"add.u32 {col}, {col}, 1;");
                b.L($"setp.ge.u32 {pWrap}, {col}, {cols};");
                b.L($"@{pWrap} mov.u32 {col}, 0;");
            }

            string y = Codec.UnpackAt(b, f, pv, col, lane);
            string r = Formula(b, lane, op.Name, x, y, zero, zero, zero);
            Codec.Pack(b, f, r, lane, i, wo, wrap: !f.IsFloat);
        }

        b.L($"st.global.{wt} [{ao}], {wo};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_v, .param .u64 p_o, .param .u32 p_n, .param .u32 p_cols");
    }

    // ---------------------------------------------------------------- transpose through a 32 × 32 tile

    public const int TransposeTile = 32;

    /// <summary>output (cols × rows) = xᵀ: a block per 32 × 32 tile (256 threads), the tile read along the rows into shared
    /// memory as lane values, then each thread writes whole words of the output rows.</summary>
    public static string Transpose(Gpu gpu, Fmt f, Lane lane, bool valueWise, out string entry)
    {
        entry = $"transpose_{f.Name}{(valueWise ? "_v" : "")}";
        const int T = TransposeTile;
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), tx = b.R(), ty = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"and.b32 {tx}, {tid}, 31;");
        b.L($"shr.u32 {ty}, {tid}, 5;");
        string src = Param64(b, "p_x"), dst = Param64(b, "p_o"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols");
        string r0 = b.R(), c0 = b.R(), smem = b.R();
        b.L($"mul.lo.u32 {r0}, {by}, {T};");
        b.L($"mul.lo.u32 {c0}, {bx}, {T};");
        b.L($"mov.u32 {smem}, smem;");
        // Read: rows r0 + ty + 8k, column c0 + tx, into tile[row][col] (row stride T + 1 lanes against bank conflicts).
        for (int k = 0; k < T / 8; k++)
        {
            string r = b.R(), c = b.R(), pr = b.P(), pc = b.P(), idx = b.R(), sAddr = b.R();
            b.L($"add.u32 {r}, {r0}, {ty};");
            b.L($"add.u32 {r}, {r}, {k * 8};");
            b.L($"add.u32 {c}, {c0}, {tx};");
            b.L($"setp.lt.u32 {pr}, {r}, {rows};");
            b.L($"setp.lt.u32 {pc}, {c}, {cols};");
            b.L($"and.pred {pr}, {pr}, {pc};");
            b.L($"@!{pr} bra NEXT{k};");
            b.L($"mad.lo.u32 {idx}, {r}, {cols}, {c};");
            string v = Codec.UnpackAt(b, f, src, idx, lane);
            b.L($"mad.lo.u32 {sAddr}, {ty}, {(T + 1) * lane.Size}, {smem};");
            b.L($"add.u32 {sAddr}, {sAddr}, {k * 8 * (T + 1) * lane.Size};");
            b.L($"mad.lo.u32 {sAddr}, {tx}, {lane.Size}, {sAddr};");
            b.L($"st.shared.{lane.Mem} [{sAddr}], {v};");
            b.Label($"NEXT{k}");
        }

        b.L("bar.sync 0;");
        if (valueWise)
        {
            // Rows do not end on words: each value written on its own (16- and 8-bit types as such; narrower ones ORed into a
            // zeroed output, since two threads share a byte).
            for (int k = 0; k < T / 8; k++)
            {
                string rr = b.R(), cc = b.R(), pr = b.P(), pc = b.P(), sAddr = b.R(), v = lane.NewReg(b), vIdx = b.RD(), addr = b.RD(), rows64 = b.RD(), t64 = b.RD();
                b.L($"add.u32 {cc}, {ty}, {k * 8};");                       // the tile column this thread writes (an output row)
                b.L($"add.u32 {rr}, {r0}, {tx};");                         // the input row (an output column)
                b.L($"add.u32 {cc}, {c0}, {cc};");
                b.L($"setp.lt.u32 {pr}, {rr}, {rows};");
                b.L($"setp.lt.u32 {pc}, {cc}, {cols};");
                b.L($"and.pred {pr}, {pr}, {pc};");
                b.L($"@!{pr} bra VNEXT{k};");
                b.L($"mad.lo.u32 {sAddr}, {tx}, {(T + 1) * lane.Size}, {smem};");
                b.L($"add.u32 {sAddr}, {sAddr}, {k * 8 * lane.Size};");
                b.L($"mad.lo.u32 {sAddr}, {ty}, {lane.Size}, {sAddr};");
                b.L($"ld.shared.{lane.Mem} {v}, [{sAddr}];");
                b.L($"cvt.u64.u32 {rows64}, {rows};");
                b.L($"cvt.u64.u32 {vIdx}, {cc};");
                b.L($"mul.lo.u64 {vIdx}, {vIdx}, {rows64};");
                b.L($"cvt.u64.u32 {t64}, {rr};");
                b.L($"add.u64 {vIdx}, {vIdx}, {t64};");
                if (f.Bits >= 8)
                {
                    string code = f.WordBits == 64 ? b.RD() : b.R();
                    if (f.WordBits == 32) b.L($"mov.b32 {code}, 0;");
                    Codec.Pack(b, f, v, lane, 0, code);
                    b.L($"mul.lo.u64 {addr}, {vIdx}, {f.Bits / 8};");
                    b.L($"add.u64 {addr}, {dst}, {addr};");
                    if (f.Bits == 64) b.L($"st.global.b64 [{addr}], {code};");
                    else if (f.Bits == 32) b.L($"st.global.b32 [{addr}], {code};");
                    else if (f.Bits == 16) { string h = b.H(); b.L($"cvt.u16.u32 {h}, {code};"); b.L($"st.global.b16 [{addr}], {h};"); }
                    else { string h = b.H(); b.L($"cvt.u16.u32 {h}, {code};"); b.L($"st.global.b8 [{addr}], {h};"); }
                }
                else
                {
                    string code = b.R(), shift = b.RD(), shift32 = b.R(), wordIdx = b.RD(), shifted = b.R();
                    b.L($"mov.b32 {code}, 0;");
                    Codec.Pack(b, f, v, lane, 0, code);
                    b.L($"mul.lo.u64 {shift}, {vIdx}, {f.Bits};");
                    b.L($"shr.u64 {wordIdx}, {shift}, 5;");
                    b.L($"and.b64 {shift}, {shift}, 31;");
                    b.L($"cvt.u32.u64 {shift32}, {shift};");
                    b.L($"shl.b32 {shifted}, {code}, {shift32};");
                    b.L($"shl.b64 {addr}, {wordIdx}, 2;");
                    b.L($"add.u64 {addr}, {dst}, {addr};");
                    b.L($"atom.global.or.b32 {code}, [{addr}], {shifted};");
                }

                b.Label($"VNEXT{k}");
            }

            b.L("ret;");
            return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_o, .param .u32 p_rows, .param .u32 p_cols", T * (T + 1) * lane.Size);
        }

        // Write: output row c0 + j (an input column), values r0 .. r0 + 32 (input rows) = 32 / Per words; thread tid handles
        // word (tid % wordsPerRow) of output row tid / wordsPerRow, for rows tid / wordsPerRow + 256 / wordsPerRow · k.
        int wordsPerRow = T / f.Per, rowsPerPass = 256 / wordsPerRow;
        string wIdx = b.R(), jBase = b.R();
        b.L($"rem.u32 {wIdx}, {tid}, {wordsPerRow};");
        b.L($"div.u32 {jBase}, {tid}, {wordsPerRow};");
        for (int pass = 0; pass * rowsPerPass < T; pass++)
        {
            string j = b.R(), pj = b.P(), outRow = b.R(), word = f.WordBits == 64 ? b.RD() : b.R(), rStart = b.R();
            b.L($"add.u32 {j}, {jBase}, {pass * rowsPerPass};");
            b.L($"add.u32 {outRow}, {c0}, {j};");
            b.L($"setp.lt.u32 {pj}, {outRow}, {cols};");
            b.L($"setp.lt.u32 {pj}, {j}, {T};");
            b.L($"@!{pj} bra WNEXT{pass};");
            string pcol = b.P();
            b.L($"setp.lt.u32 {pcol}, {outRow}, {cols};");
            b.L($"@!{pcol} bra WNEXT{pass};");
            if (f.WordBits == 32) b.L($"mov.b32 {word}, 0;");
            b.L($"mul.lo.u32 {rStart}, {wIdx}, {f.Per};");
            for (int i = 0; i < f.Per; i++)
            {
                string rr = b.R(), sAddr = b.R(), v = lane.NewReg(b), pIn = b.P(), rg = b.R();
                b.L($"add.u32 {rr}, {rStart}, {i};");
                b.L($"add.u32 {rg}, {r0}, {rr};");
                b.L($"setp.lt.u32 {pIn}, {rg}, {rows};");
                b.L($"mov.{lane.Reg} {v}, {lane.Zero};");
                b.L($"mad.lo.u32 {sAddr}, {rr}, {(T + 1) * lane.Size}, {smem};");
                b.L($"mad.lo.u32 {sAddr}, {j}, {lane.Size}, {sAddr};");
                b.L($"@{pIn} ld.shared.{lane.Mem} {v}, [{sAddr}];");
                Codec.Pack(b, f, v, lane, i, word);
            }

            // The output word index: outRow · rows + r0 + rStart values, a whole word since r0 and rStart are multiples of Per.
            string vIdx = b.RD(), wAddr = b.RD(), rows64 = b.RD(), add64 = b.RD();
            b.L($"cvt.u64.u32 {rows64}, {rows};");
            b.L($"cvt.u64.u32 {vIdx}, {outRow};");
            b.L($"mul.lo.u64 {vIdx}, {vIdx}, {rows64};");
            b.L($"cvt.u64.u32 {add64}, {r0};");
            b.L($"add.u64 {vIdx}, {vIdx}, {add64};");
            b.L($"cvt.u64.u32 {add64}, {rStart};");
            b.L($"add.u64 {vIdx}, {vIdx}, {add64};");
            if (f.WordBits == 64) b.L($"shl.b64 {wAddr}, {vIdx}, 3;");
            else
            {
                b.L($"mul.lo.u64 {wAddr}, {vIdx}, {f.Bits};");
                b.L($"shr.u64 {wAddr}, {wAddr}, 3;");
            }

            b.L($"add.u64 {wAddr}, {dst}, {wAddr};");
            // Only whole words: a word past the output row's end (rows not a multiple of Per) is written with its zero tail.
            string pEnd = b.P(), rNext = b.R();
            b.L($"add.u32 {rNext}, {r0}, {rStart};");
            b.L($"setp.lt.u32 {pEnd}, {rNext}, {rows};");
            b.L($"@{pEnd} st.global.{(f.WordBits == 64 ? "b64" : "b32")} [{wAddr}], {word};");
            b.Label($"WNEXT{pass}");
        }

        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_o, .param .u32 p_rows, .param .u32 p_cols", T * (T + 1) * lane.Size);
    }

    // ---------------------------------------------------------------- y = A·x (a warp per row), y = Aᵀ·x (a thread per column)

    public static string GemvRows(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"gemv_rows_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), ctaid = b.R(), laneId = b.R(), warp = b.R(), row = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {ctaid}, %ctaid.x;");
        b.L($"and.b32 {laneId}, {tid}, 31;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"mad.lo.u32 {row}, {ctaid}, 8, {warp};");
        string pa = Param64(b, "p_a"), px = Param64(b, "p_x"), py = Param64(b, "p_y"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols");
        string pEnd = b.P(), acc = lane.NewReg(b), i = b.R(), start = b.R(), pLoop = b.P();
        b.L($"setp.ge.u32 {pEnd}, {row}, {rows};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
        b.L($"mul.lo.u32 {start}, {row}, {cols};");
        b.L($"mov.u32 {i}, {laneId};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pLoop}, {i}, {cols};");
        b.L($"@{pLoop} bra DONE;");
        string idx = b.R();
        b.L($"add.u32 {idx}, {start}, {i};");
        string av = Codec.UnpackAt(b, f, pa, idx, lane);
        string xv = Codec.UnpackAt(b, f, px, i, lane);
        b.L(lane.Fma(acc, av, xv, acc));
        b.L($"add.u32 {i}, {i}, 32;");
        b.L("bra LOOP;");
        b.Label("DONE");
        WarpReduce(b, lane, acc, "add");
        string p0 = b.P(), yAddr = b.RD();
        b.L($"setp.ne.u32 {p0}, {laneId}, 0;");
        b.L($"@{p0} bra OUT;");
        b.L($"mul.wide.u32 {yAddr}, {row}, {lane.Size};");
        b.L($"add.u64 {yAddr}, {py}, {yAddr};");
        b.L($"st.global.{lane.Mem} [{yAddr}], {acc};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_cols");
    }

    /// <summary>y[split][col] = Σ over the rows of split %ctaid.y of A[r][col]·x[r]; a thread per column.</summary>
    public static string GemvCols(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"gemv_cols_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pa = Param64(b, "p_a"), px = Param64(b, "p_x"), py = Param64(b, "p_y"), rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols"), per = Param32(b, "p_per");
        string pEnd = b.P(), acc = lane.NewReg(b), r = b.R(), idx = b.R(), pLoop = b.P(), split = b.R(), rEnd = b.R();
        b.L($"mov.u32 {split}, %ctaid.y;");
        b.L($"setp.ge.u32 {pEnd}, {gid}, {cols};");
        b.L($"@{pEnd} bra OUT;");
        b.L($"mov.{lane.Reg} {acc}, {lane.Zero};");
        b.L($"mul.lo.u32 {r}, {split}, {per};");
        b.L($"add.u32 {rEnd}, {r}, {per};");
        b.L($"min.u32 {rEnd}, {rEnd}, {rows};");
        b.L($"mad.lo.u32 {idx}, {r}, {cols}, {gid};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pLoop}, {r}, {rEnd};");
        b.L($"@{pLoop} bra DONE;");
        string av = Codec.UnpackAt(b, f, pa, idx, lane);
        string xv = Codec.UnpackAt(b, f, px, r, lane);
        b.L(lane.Fma(acc, av, xv, acc));
        b.L($"add.u32 {idx}, {idx}, {cols};");
        b.L($"add.u32 {r}, {r}, 1;");
        b.L("bra LOOP;");
        b.Label("DONE");
        string yAddr = b.RD(), oIdx = b.R();
        b.L($"mad.lo.u32 {oIdx}, {split}, {cols}, {gid};");
        b.L($"mul.wide.u32 {yAddr}, {oIdx}, {lane.Size};");
        b.L($"add.u64 {yAddr}, {py}, {yAddr};");
        b.L($"st.global.{lane.Mem} [{yAddr}], {acc};");
        b.Label("OUT");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_a, .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_per");
    }
}
