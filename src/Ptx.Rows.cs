// The device's row operations (Rows.cs): one block per row, the row's values in registers (each thread `perThread`
// words at tid, tid + threads, …, as the softmax kernel holds them), the row's statistics by block reductions (up to
// two rounds, each of up to three sums or a max), the outputs packed and stored once. Rows that do not end on a word
// (16- and 8-bit floats with such a column count) are read and written value by value (`valueWise`).
//
// The parameter backward (dgamma, dbeta) is two kernels: the rows' statistics (mean, rstd as lane values), then a
// thread per column over a split of the rows (partials joined by the column-reduction kernel).
namespace Qiyas;

internal static partial class Ptx
{
    public static string RowOp(Gpu gpu, Fmt f, Lane lane, RowOp op, int perThread, int threads, bool vector, bool valueWise, bool second, out string entry, bool segmented = false)
    {
        entry = $"row_{op.Name}_{f.Name}_{lane.Name}_r{perThread}_t{threads}{(vector ? "_v" : "")}{(valueWise ? "_w" : "")}{(second ? "" : "_nosecond")}{(segmented ? "_seg" : "")}";
        var c = new RowCtx(new PtxBuilder(), f, lane, perThread, threads, vector, valueWise, segmented);
        c.Prologue(op.Second && second);
        EmitRowOp(c, op);
        c.B.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return c.B.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_x2, .param .u64 p_g, .param .u64 p_bt, .param .u64 p_s, .param .u64 p_pos, .param .u64 p_y, .param .u64 p_y2, .param .u32 p_rows, .param .u32 p_cols, .param .{pa} p_a, .param .{pa} p_b",
            ReduceBytes);
    }

    /// <summary>stats[2·row] = mean (0 when not centered), stats[2·row + 1] = 1/√(mean (x − mean)² + eps), lane values.</summary>
    public static string NormStats(Gpu gpu, Fmt f, Lane lane, bool centered, int perThread, int threads, bool vector, bool valueWise, out string entry, bool segmented = false)
    {
        entry = $"norm_stats_{(centered ? "c" : "r")}_{f.Name}_{lane.Name}_r{perThread}_t{threads}{(vector ? "_v" : "")}{(valueWise ? "_w" : "")}{(segmented ? "_seg" : "")}";
        var c = new RowCtx(new PtxBuilder(), f, lane, perThread, threads, vector, valueWise, segmented);
        c.Prologue(false);
        var b = c.B;
        string n = c.Count();
        string mean = centered ? c.Div(c.Reduce("add", (q, i) => c.X(q, i)), n) : c.Lit(0);
        string ss = c.Reduce("add", (q, i) => { string d = c.Sub(c.X(q, i), mean); return c.Mul(d, d); });
        string rstd = c.Rcp(c.Sqrt(c.Add(c.Div(ss, n), c.A)));
        string p0 = b.P(), addr = b.RD();
        b.L($"setp.ne.u32 {p0}, {c.Tid}, 0;");
        b.L($"@{p0} bra OUT;");
        b.L($"mul.wide.u32 {addr}, {c.Row}, {2 * lane.Size};");
        b.L($"add.u64 {addr}, {c.Dst}, {addr};");
        b.L($"st.global.{lane.Mem} [{addr}], {mean};");
        b.L($"st.global.{lane.Mem} [{addr}+{lane.Size}], {rstd};");
        b.Label("OUT");
        b.L("ret;");
        string pa = lane.Wide ? "f64" : "f32";
        return b.Finish(Header(gpu), entry,
            $".param .u64 p_x, .param .u64 p_x2, .param .u64 p_g, .param .u64 p_bt, .param .u64 p_s, .param .u64 p_pos, .param .u64 p_y, .param .u64 p_y2, .param .u32 p_rows, .param .u32 p_cols, .param .{pa} p_a, .param .{pa} p_b",
            ReduceBytes);
    }

    /// <summary>A thread per column over rows [split·per, …): partials_g[split][c] = Σ dy·(x − mean_r)·rstd_r, partials_b[split][c] = Σ dy (lane values).</summary>
    public static string NormParamsCols(Gpu gpu, Fmt f, Lane lane, out string entry)
    {
        entry = $"norm_params_cols_{f.Name}_{lane.Name}";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), split = b.R(), col = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mov.u32 {split}, %ctaid.y;");
        b.L($"mad.lo.u32 {col}, {bid}, 256, {tid};");
        string x = Param64(b, "p_x"), dy = Param64(b, "p_dy"), stats = Param64(b, "p_stats"), og = Param64(b, "p_g"), ob = Param64(b, "p_b");
        string rows = Param32(b, "p_rows"), cols = Param32(b, "p_cols"), per = Param32(b, "p_per");
        string pOut = b.P();
        b.L($"setp.ge.u32 {pOut}, {col}, {cols};");
        b.L($"@{pOut} bra DONE;");
        string r = b.R(), rEnd = b.R(), accG = lane.NewReg(b), accB = lane.NewReg(b), pEnd = b.P();
        b.L($"mul.lo.u32 {r}, {split}, {per};");
        b.L($"add.u32 {rEnd}, {r}, {per};");
        b.L($"min.u32 {rEnd}, {rEnd}, {rows};");
        b.L($"mov.{lane.Reg} {accG}, {lane.Zero};");
        b.L($"mov.{lane.Reg} {accB}, {lane.Zero};");
        b.Label("LOOP");
        b.L($"setp.ge.u32 {pEnd}, {r}, {rEnd};");
        b.L($"@{pEnd} bra END;");
        string idx = b.R(), sAddr = b.RD(), mean = lane.NewReg(b), rstd = lane.NewReg(b);
        b.L($"mad.lo.u32 {idx}, {r}, {cols}, {col};");
        string xv = Codec.UnpackAt(b, f, x, idx, lane), dv = Codec.UnpackAt(b, f, dy, idx, lane);
        b.L($"mul.wide.u32 {sAddr}, {r}, {2 * lane.Size};");
        b.L($"add.u64 {sAddr}, {stats}, {sAddr};");
        b.L($"ld.global.{lane.Mem} {mean}, [{sAddr}];");
        b.L($"ld.global.{lane.Mem} {rstd}, [{sAddr}+{lane.Size}];");
        string h = lane.NewReg(b);
        b.L($"sub.rn.{lane.Op} {h}, {xv}, {mean};");
        b.L(lane.Mul(h, h, rstd));
        b.L(lane.Fma(accG, dv, h, accG));
        b.L(lane.Add(accB, accB, dv));
        b.L($"add.u32 {r}, {r}, 1;");
        b.L("bra LOOP;");
        b.Label("END");
        string oIdx = b.R(), oAddr = b.RD();
        b.L($"mad.lo.u32 {oIdx}, {split}, {cols}, {col};");
        b.L($"mul.wide.u32 {oAddr}, {oIdx}, {lane.Size};");
        string ag = b.RD(), ab = b.RD();
        b.L($"add.u64 {ag}, {og}, {oAddr};");
        b.L($"add.u64 {ab}, {ob}, {oAddr};");
        b.L($"st.global.{lane.Mem} [{ag}], {accG};");
        b.L($"st.global.{lane.Mem} [{ab}], {accB};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_x, .param .u64 p_dy, .param .u64 p_stats, .param .u64 p_g, .param .u64 p_b, .param .u32 p_rows, .param .u32 p_cols, .param .u32 p_per");
    }

    // ---------------------------------------------------------------- the row context: loads, lane arithmetic, reductions, stores

    private sealed class RowCtx(PtxBuilder b, Fmt f, Lane lane, int perThread, int threads, bool vector, bool valueWise, bool segmented = false)
    {
        public PtxBuilder B => b;
        public bool Segmented => segmented;
        private string _segBase = "", _segs = "", _seg = "";
        private int _segLoops;
        private bool _second;
        public Lane Lane => lane;
        public Fmt F => f;
        public string Tid = "", Row = "", Cols = "", ColsM1 = "", Rows = "", Src = "", Dst = "", A = "", Bs = "";
        private string _words = "", _rowSrc = "", _rowDst = "", _rowSrc2 = "", _rowDst2 = "", _src2 = "", _dst2 = "", _pg = "", _pb = "", _ps = "", _ppos = "", _base = "";
        private string[] _present = [], _inWords = [];
        private string[][] _x = [], _x2 = [], _col = [], _g = [], _bt = [];
        private string[] _pv = [];
        private string? _pos, _s, _n;

        public int Per => valueWise ? 1 : f.Per;
        public int Words => perThread;
        public bool ValueWise => valueWise;

        /// <summary>Every (word, value) of this thread.</summary>
        public IEnumerable<(int q, int i)> Each()
        {
            for (int q = 0; q < perThread; q++) for (int i = 0; i < Per; i++) yield return (q, i);
        }

        public void Prologue(bool second)
        {
            Tid = b.R(); Row = b.R(); Cols = b.R(); Rows = b.R();
            b.L($"mov.u32 {Tid}, %tid.x;");
            b.L($"mov.u32 {Row}, %ctaid.x;");
            b.L($"ld.param.u32 {Cols}, [p_cols];");
            b.L($"ld.param.u32 {Rows}, [p_rows];");
            ColsM1 = b.R();
            b.L($"sub.u32 {ColsM1}, {Cols}, 1;");
            Src = Param64(b, "p_x"); Dst = Param64(b, "p_y"); _src2 = Param64(b, "p_x2"); _dst2 = Param64(b, "p_y2");
            _pg = Param64(b, "p_g"); _pb = Param64(b, "p_bt"); _ps = Param64(b, "p_s"); _ppos = Param64(b, "p_pos");
            A = lane.NewReg(b); Bs = lane.NewReg(b);
            b.L($"ld.param.{lane.Op} {A}, [p_a];");
            b.L($"ld.param.{lane.Op} {Bs}, [p_b];");
            _x = new string[perThread][]; _col = new string[perThread][]; _pv = new string[perThread];
            _second = second;
            if (!valueWise)
            {
                _words = b.R();
                string off = b.RD();
                if (f.WordBits == 64) b.L($"mov.u32 {_words}, {Cols};");
                else { b.L($"mul.lo.u32 {_words}, {Cols}, {f.Bits};"); b.L($"shr.u32 {_words}, {_words}, 5;"); }
                b.L($"mul.wide.u32 {off}, {Row}, {_words};");
                b.L($"mul.lo.u64 {off}, {off}, {f.WordBits / 8};");
                _rowSrc = b.RD(); _rowDst = b.RD(); _rowSrc2 = b.RD(); _rowDst2 = b.RD();
                b.L($"add.u64 {_rowSrc}, {Src}, {off};");
                b.L($"add.u64 {_rowDst}, {Dst}, {off};");
                b.L($"add.u64 {_rowSrc2}, {_src2}, {off};");
                b.L($"add.u64 {_rowDst2}, {_dst2}, {off};");
                if (segmented)
                {
                    // The row in segments of perThread·threads words: the registers below are reloaded for each segment in every phase.
                    _segs = b.R(); _segBase = b.R(); _seg = b.R();
                    b.L($"add.u32 {_segs}, {_words}, {perThread * threads - 1};");
                    b.L($"div.u32 {_segs}, {_segs}, {perThread * threads};");
                    _present = new string[perThread]; _inWords = new string[perThread]; _x = new string[perThread][]; _x2 = new string[perThread][];
                    for (int q = 0; q < perThread; q++)
                    {
                        _present[q] = b.P(); _inWords[q] = b.R(); _x[q] = new string[f.Per]; _x2[q] = new string[f.Per]; _col[q] = new string[f.Per]; _pv[q] = _present[q];
                        for (int i = 0; i < f.Per; i++) { _x[q][i] = lane.NewReg(b); if (second) _x2[q][i] = lane.NewReg(b); _col[q][i] = b.R(); }
                        if (!second) _x2[q] = null!;                                        // X2 reads zero
                    }

                    return;
                }

                (_present, _inWords, _x) = RowLoad(b, f, lane, Tid, _words, _rowSrc, perThread, lane.Zero, threads, vector);
                if (second) (_, _, _x2) = RowLoad(b, f, lane, Tid, _words, _rowSrc2, perThread, lane.Zero, threads, vector);
                for (int q = 0; q < perThread; q++)
                {
                    _pv[q] = _present[q];
                    _col[q] = new string[f.Per];
                    for (int i = 0; i < f.Per; i++)
                    {
                        _col[q][i] = b.R();
                        b.L($"mad.lo.u32 {_col[q][i]}, {_inWords[q]}, {f.Per}, {i};");
                    }
                }

                return;
            }

            // Value by value: value c = tid + q·threads of the row, at index row·cols + c.
            _base = b.R();
            b.L($"mul.lo.u32 {_base}, {Row}, {Cols};");
            _x2 = new string[perThread][];
            if (segmented)
            {
                // The row in segments of perThread·threads values: the registers below are reloaded for each segment in every phase.
                _segs = b.R(); _segBase = b.R(); _seg = b.R();
                b.L($"add.u32 {_segs}, {Cols}, {perThread * threads - 1};");
                b.L($"div.u32 {_segs}, {_segs}, {perThread * threads};");
                _present = new string[perThread];
                for (int q = 0; q < perThread; q++)
                {
                    _present[q] = b.P(); _pv[q] = _present[q]; _col[q] = [b.R()]; _x[q] = [lane.NewReg(b)]; if (second) _x2[q] = [lane.NewReg(b)];
                }

                return;
            }

            for (int q = 0; q < perThread; q++)
            {
                string c = b.R(), p = b.P(), idx = b.R();
                b.L($"add.u32 {c}, {Tid}, {q * threads};");
                b.L($"setp.lt.u32 {p}, {c}, {Cols};");
                string cs = b.R();
                b.L($"min.u32 {cs}, {c}, {ColsM1};");
                b.L($"add.u32 {idx}, {_base}, {cs};");
                _col[q] = [c]; _pv[q] = p;
                _x[q] = [Codec.UnpackAt(b, f, Src, idx, lane)];
                if (second) _x2[q] = [Codec.UnpackAt(b, f, _src2, idx, lane)];
            }
        }

        public string X(int q, int i) => _x[q][i];

        /// <summary>The second row's value; zero when the operation's second input was not given (an absent mask).</summary>
        public string X2(int q, int i)
        {
            if (_x2.Length > 0 && _x2[q] is not null) return _x2[q][i];
            string z = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {z}, {lane.Zero};");
            return z;
        }
        public string Col(int q, int i) => _col[q][i];
        public string Pred(int q) => _pv[q];

        /// <summary>The per-column vector's value at this (q, i): gamma or beta, loaded once.</summary>
        public string G(int q, int i) => Vec(ref _g, _pg, q, i);
        public string Bt(int q, int i) => Vec(ref _bt, _pb, q, i);

        private string Vec(ref string[][] cache, string ptr, int q, int i)
        {
            if (cache.Length == 0) { cache = new string[perThread][]; for (int k = 0; k < perThread; k++) cache[k] = new string[Per]; }
            if (cache[q][i] is not null) return cache[q][i];
            string cs = b.R();
            b.L($"min.u32 {cs}, {Col(q, i)}, {ColsM1};");   // the padding lanes read the last value (dropped on the store)
            return cache[q][i] = Codec.UnpackAt(b, f, ptr, cs, lane);
        }

        /// <summary>The row's scalar (p_s; 1 when null).</summary>
        public string S()
        {
            if (_s is not null) return _s;
            string pz = b.P(), v = lane.NewReg(b);
            b.L($"setp.eq.u64 {pz}, {_ps}, 0;");
            b.L($"mov.{lane.Reg} {v}, {lane.Literal(1)};");
            b.L($"@{pz} bra S_DONE;");
            string loaded = Codec.UnpackAt(b, f, _ps, Row, lane);
            b.L($"mov.{lane.Reg} {v}, {loaded};");
            b.Label("S_DONE");
            return _s = v;
        }

        private string? _posInt;

        /// <summary>The row's position as an s32 register (p_pos; the row's index when null).</summary>
        public string PosInt()
        {
            if (_posInt is not null) return _posInt;
            string pz = b.P(), pi = b.R(), addr = b.RD();
            b.L($"setp.eq.u64 {pz}, {_ppos}, 0;");
            b.L($"mov.u32 {pi}, {Row};");
            b.L($"mul.wide.u32 {addr}, {Row}, 4;");
            b.L($"add.u64 {addr}, {_ppos}, {addr};");
            b.L($"@!{pz} ld.global.s32 {pi}, [{addr}];");
            return _posInt = pi;
        }

        /// <summary>The row's position as a lane value.</summary>
        public string Pos()
        {
            if (_pos is not null) return _pos;
            string v = lane.NewReg(b);
            b.L($"cvt.rn.{lane.Op}.s32 {v}, {PosInt()};");
            return _pos = v;
        }

        public string GtS(string x, string y) { string p = b.P(); b.L($"setp.gt.s32 {p}, {x}, {y};"); return p; }

        /// <summary>cols as a lane value.</summary>
        public string Count()
        {
            if (_n is not null) return _n;
            string v = lane.NewReg(b);
            b.L($"cvt.rn.{lane.Op}.u32 {v}, {Cols};");
            return _n = v;
        }

        // ---- segments: a loop over the row's segments with this thread's words of the segment loaded into the value registers
        public void BeginSegments()
        {
            if (!segmented) return;
            string tag = $"SEG{_segLoops++}";
            b.L($"mov.u32 {_seg}, 0;");
            b.Label(tag);
            string pEnd = b.P();
            b.L($"setp.ge.u32 {pEnd}, {_seg}, {_segs};");
            b.L($"@{pEnd} bra {tag}_END;");
            b.L($"mul.lo.u32 {_segBase}, {_seg}, {perThread * threads};");
            _g = []; _bt = [];                                                 // the per-column vectors are read per segment
            if (valueWise)
            {
                for (int q = 0; q < perThread; q++)
                {
                    string cs = b.R(), idx = b.R();
                    b.L($"add.u32 {_col[q][0]}, {Tid}, {q * threads};");
                    b.L($"add.u32 {_col[q][0]}, {_col[q][0]}, {_segBase};");
                    b.L($"setp.lt.u32 {_present[q]}, {_col[q][0]}, {Cols};");
                    b.L($"min.u32 {cs}, {_col[q][0]}, {ColsM1};");
                    b.L($"add.u32 {idx}, {_base}, {cs};");
                    string v = Codec.UnpackAt(b, f, Src, idx, lane);
                    b.L($"mov.{lane.Reg} {_x[q][0]}, {v};");
                    if (_second) { string v2 = Codec.UnpackAt(b, f, _src2, idx, lane); b.L($"mov.{lane.Reg} {_x2[q][0]}, {v2};"); }
                }

                _segTag = tag;
                return;
            }

            string wt = f.WordBits == 64 ? "b64" : "b32";
            for (int q = 0; q < perThread; q++)
            {
                string off = b.RD(), addr = b.RD(), word = f.WordBits == 64 ? b.RD() : b.R(), word2 = f.WordBits == 64 ? b.RD() : b.R();
                b.L($"add.u32 {_inWords[q]}, {Tid}, {q * threads};");
                b.L($"add.u32 {_inWords[q]}, {_inWords[q]}, {_segBase};");
                b.L($"setp.lt.u32 {_present[q]}, {_inWords[q]}, {_words};");
                b.L($"mul.wide.u32 {off}, {_inWords[q]}, {f.WordBits / 8};");
                b.L($"add.u64 {addr}, {_rowSrc}, {off};");
                b.L($"mov.{wt} {word}, 0;");
                b.L($"@{_present[q]} ld.global.{wt} {word}, [{addr}];");
                if (_second)
                {
                    string addr2 = b.RD();
                    b.L($"add.u64 {addr2}, {_rowSrc2}, {off};");
                    b.L($"mov.{wt} {word2}, 0;");
                    b.L($"@{_present[q]} ld.global.{wt} {word2}, [{addr2}];");
                }

                for (int i = 0; i < f.Per; i++)
                {
                    string v = Codec.Unpack(b, f, word, i, lane);
                    b.L($"mov.{lane.Reg} {_x[q][i]}, {v};");
                    if (_second) { string v2 = Codec.Unpack(b, f, word2, i, lane); b.L($"mov.{lane.Reg} {_x2[q][i]}, {v2};"); }
                    b.L($"mad.lo.u32 {_col[q][i]}, {_inWords[q]}, {f.Per}, {i};");
                }
            }

            _segTag = tag;
        }

        private string _segTag = "";

        public void EndSegments()
        {
            if (!segmented) return;
            b.L($"add.u32 {_seg}, {_seg}, 1;");
            b.L($"bra {_segTag};");
            b.Label($"{_segTag}_END");
        }

        // ---- lane arithmetic
        public string Lit(double v) => lane.Literal(v);
        public string Add(string x, string y) { string d = lane.NewReg(b); b.L(lane.Add(d, x, y)); return d; }
        public string Sub(string x, string y) { string d = lane.NewReg(b); b.L($"sub.rn.{lane.Op} {d}, {x}, {y};"); return d; }
        public string Mul(string x, string y) { string d = lane.NewReg(b); b.L(lane.Mul(d, x, y)); return d; }
        public string Fma(string x, string y, string z) { string d = lane.NewReg(b); b.L(lane.Fma(d, x, y, z)); return d; }
        public string Div(string x, string y) { string d = lane.NewReg(b); b.L($"div.rn.{lane.Op} {d}, {x}, {y};"); return d; }
        public string Rcp(string x) => Div(Lit(1), x);
        public string Neg(string x) { string d = lane.NewReg(b); b.L($"neg.{lane.Op} {d}, {x};"); return d; }
        public string Max(string x, string y) { string d = lane.NewReg(b); b.L($"max.{lane.Op} {d}, {x}, {y};"); return d; }
        public string Exp(string x) => Codec.Exp(b, lane, x);
        public string Log(string x) => Fn.Log(b, lane, x);
        public string Sqrt(string x) => Fn.Sqrt(b, lane, x);
        public string Sin(string x) => Fn.Sin(b, lane, x);
        public string Cos(string x) => Fn.Cos(b, lane, x);
        public string Sel(string p, string x, string y) { string d = lane.NewReg(b); b.L($"selp.{lane.Op} {d}, {x}, {y}, {p};"); return d; }
        public string Gt(string x, string y) { string p = b.P(); b.L($"setp.gt.{lane.Op} {p}, {x}, {y};"); return p; }
        public string GtU(string x, string y) { string p = b.P(); b.L($"setp.gt.u32 {p}, {x}, {y};"); return p; }
        public string LtU(string x, string y) { string p = b.P(); b.L($"setp.lt.u32 {p}, {x}, {y};"); return p; }
        public string And(string p, string q) { string d = b.P(); b.L($"and.pred {d}, {p}, {q};"); return d; }
        public string Not(string p) { string d = b.P(); b.L($"not.pred {d}, {p};"); return d; }
        public string ColF(int q, int i) { string v = lane.NewReg(b); b.L($"cvt.rn.{lane.Op}.u32 {v}, {Col(q, i)};"); return v; }

        /// <summary>The block's reduction ("add" or "max") of term(q, i) over this thread's present values.</summary>
        public string Reduce(string kind, Func<int, int, string> term)
        {
            string acc = lane.NewReg(b);
            b.L($"mov.{lane.Reg} {acc}, {(kind == "max" ? lane.Literal(double.NegativeInfinity) : lane.Zero)};");
            BeginSegments();
            foreach (var (q, i) in Each())
            {
                string t = term(q, i);
                b.L(kind == "max" ? $"@{Pred(q)} max.{lane.Op} {acc}, {acc}, {t};" : $"@{Pred(q)} {lane.Add(acc, acc, t)}");
            }

            EndSegments();
            return BlockReduce(b, lane, Tid, acc, kind, threads);
        }

        /// <summary>The output row (or the second one) from value(q, i).</summary>
        public void Store(Func<int, int, string> value, bool second = false)
        {
            if (!valueWise)
            {
                BeginSegments();
                RowStore(b, f, lane, _present, _inWords, second ? _rowDst2 : _rowDst, perThread, value, _words, vector && !segmented);
                EndSegments();
                return;
            }

            string dst = second ? _dst2 : Dst;
            BeginSegments();
            for (int q = 0; q < perThread; q++)
            {
                string code = b.R(), idx = b.R(), addr = b.RD(), h = b.H();
                b.L($"mov.b32 {code}, 0;");
                Codec.Pack(b, f, value(q, 0), lane, 0, code);
                b.L($"add.u32 {idx}, {_base}, {Col(q, 0)};");
                b.L($"mul.wide.u32 {addr}, {idx}, {f.Bits / 8};");
                b.L($"add.u64 {addr}, {dst}, {addr};");
                b.L($"cvt.u16.u32 {h}, {code};");
                b.L($"@{Pred(q)} st.global.b{f.Bits} [{addr}], {h};");
            }

            EndSegments();
        }
    }

    // ---------------------------------------------------------------- the operations

    private static void EmitRowOp(RowCtx c, RowOp op)
    {
        string n = c.Count();
        switch (op.Name)
        {
            case "log_softmax":
            {
                string m = c.Reduce("max", c.X);
                string sum = c.Reduce("add", (q, i) => c.Exp(c.Sub(c.X(q, i), m)));
                string ls = c.Add(m, c.Log(sum));
                c.Store((q, i) => c.Sub(c.X(q, i), ls));
                return;
            }

            case "softmax_bwd":
            {
                string d = c.Reduce("add", (q, i) => c.Mul(c.X(q, i), c.X2(q, i)));
                c.Store((q, i) => c.Mul(c.X(q, i), c.Sub(c.X2(q, i), d)));
                return;
            }

            case "log_softmax_bwd":
            {
                string sd = c.Reduce("add", c.X2);
                c.Store((q, i) => c.Sub(c.X2(q, i), c.Mul(c.Exp(c.X(q, i)), sd)));
                return;
            }

            case "rms_norm_affine":
            {
                string rstd = Rstd(c, c.Reduce("add", (q, i) => c.Mul(c.X(q, i), c.X(q, i))), n, c.A);
                c.Store((q, i) => c.Mul(c.Mul(c.X(q, i), rstd), c.G(q, i)));
                return;
            }

            case "rms_norm_bwd":
            {
                string ss = c.Reduce("add", (q, i) => c.Mul(c.X(q, i), c.X(q, i)));
                string t = c.Reduce("add", (q, i) => c.Mul(c.Mul(c.G(q, i), c.X2(q, i)), c.X(q, i)));
                string rstd = Rstd(c, ss, n, c.A);
                string k2 = c.Mul(rstd, c.Div(c.Mul(rstd, t), n));                      // rstd · mean(g·dy·x̂)
                c.Store((q, i) => c.Mul(c.Sub(c.Mul(c.G(q, i), c.X2(q, i)), c.Mul(c.X(q, i), k2)), rstd));
                return;
            }

            case "layer_norm":
            {
                string mu = c.Div(c.Reduce("add", c.X), n);
                string v = c.Reduce("add", (q, i) => { string d = c.Sub(c.X(q, i), mu); return c.Mul(d, d); });
                string rstd = Rstd(c, v, n, c.A);
                c.Store((q, i) => c.Fma(c.Mul(c.Sub(c.X(q, i), mu), rstd), c.G(q, i), c.Bt(q, i)));
                return;
            }

            case "layer_norm_bwd":
            {
                string mu = c.Div(c.Reduce("add", c.X), n);
                string v = c.Reduce("add", (q, i) => { string d = c.Sub(c.X(q, i), mu); return c.Mul(d, d); });
                string sgd = c.Reduce("add", (q, i) => c.Mul(c.G(q, i), c.X2(q, i)));
                string sgdx = c.Reduce("add", (q, i) => c.Mul(c.Mul(c.G(q, i), c.X2(q, i)), c.Sub(c.X(q, i), mu)));
                string rstd = Rstd(c, v, n, c.A);
                string m1 = c.Div(sgd, n), k2 = c.Mul(rstd, c.Div(c.Mul(rstd, sgdx), n));
                c.Store((q, i) => c.Mul(c.Sub(c.Sub(c.Mul(c.G(q, i), c.X2(q, i)), m1), c.Mul(c.Sub(c.X(q, i), mu), k2)), rstd));
                return;
            }

            case "weight_norm":
            {
                string ss = c.Reduce("add", (q, i) => c.Mul(c.X(q, i), c.X(q, i)));
                string k = c.Div(c.S(), c.Sqrt(c.Add(ss, c.A)));
                c.Store((q, i) => c.Mul(c.X(q, i), k));
                return;
            }

            case "rope":
            {
                // θ = pos · exp(i · k), k = −2 ln a / n; the partner value (i ± half) read from the row.
                var b = c.B;
                string half = b.R(), twice = b.R();
                b.L($"shr.u32 {half}, {c.Cols}, 1;");
                b.L($"shl.b32 {twice}, {half}, 1;");
                string k = c.Mul(c.Div(c.Lit(-2), n), c.Log(c.A)), pos = c.Pos();
                string halfF = c.Lane.NewReg(b);
                b.L($"cvt.rn.{c.Lane.Op}.u32 {halfF}, {half};");
                c.Store((q, i) =>
                {
                    string col = c.Col(q, i), pLow = c.LtU(col, half), pPair = c.LtU(col, twice);
                    string colF = c.ColF(q, i), iF = c.Sel(pLow, colF, c.Sub(colF, halfF));
                    string theta = c.Mul(pos, c.Exp(c.Mul(iF, k)));
                    string cs = c.Cos(theta), sn = c.Sin(theta);
                    string partnerCol = b.R(), lowPartner = b.R(), pIdx = b.R();
                    b.L($"add.u32 {partnerCol}, {col}, {half};");
                    b.L($"sub.u32 {lowPartner}, {col}, {half};");
                    b.L($"selp.u32 {partnerCol}, {partnerCol}, {lowPartner}, {pLow};");
                    b.L($"min.u32 {partnerCol}, {partnerCol}, {c.ColsM1};");
                    b.L($"mad.lo.u32 {pIdx}, {c.Row}, {c.Cols}, {partnerCol};");
                    string partner = Codec.UnpackAt(b, c.F, c.Src, pIdx, c.Lane), x = c.X(q, i);
                    string low = c.Sub(c.Mul(x, cs), c.Mul(partner, sn));        // x[i]·cos − x[i+half]·sin
                    string high = c.Fma(partner, sn, c.Mul(x, cs));              // x[i]·sin + x[i+half]·cos (partner = x[i])
                    return c.Sel(pPair, c.Sel(pLow, low, high), x);
                });
                return;
            }

            case "scale_mask_softmax":
            {
                // z = a·x + mask; causal (b ≠ 0): columns past the row's position excluded; excluded values give 0 (a fully excluded row too).
                string pCausal = c.Gt(c.Bs, c.Lit(0)), negInf = c.Lit(double.NegativeInfinity), posInt = c.PosInt();
                string Keep(int q, int i) => c.Not(c.And(pCausal, c.GtS(c.Col(q, i), posInt)));
                string Z(int q, int i) => c.Sel(Keep(q, i), c.Fma(c.X(q, i), c.A, c.X2(q, i)), negInf);
                string m = c.Reduce("max", Z);
                string pAny = c.Gt(m, negInf);
                string sum = c.Reduce("add", (q, i) => c.Sel(Keep(q, i), c.Exp(c.Sub(Z(q, i), m)), c.Lit(0)));
                string scale = c.Sel(pAny, c.Rcp(sum), c.Lit(0));
                c.Store((q, i) => c.Sel(Keep(q, i), c.Mul(c.Exp(c.Sub(Z(q, i), m)), scale), c.Lit(0)));
                return;
            }

            case "cross_entropy_bwd":
            {
                string m = c.Reduce("max", c.X);
                string sum = c.Reduce("add", (q, i) => c.Exp(c.Sub(c.X(q, i), m)));
                string inv = c.Rcp(sum), posInt = c.PosInt();
                c.Store((q, i) =>
                {
                    string p = c.Mul(c.Exp(c.Sub(c.X(q, i), m)), inv), pTarget = c.B.P();
                    c.B.L($"setp.eq.s32 {pTarget}, {c.Col(q, i)}, {posInt};");
                    return c.Sel(pTarget, c.Sub(p, c.Lit(1)), p);
                });
                return;
            }

            case "add_rms_norm":
            {
                string H(int q, int i) => c.Add(c.X(q, i), c.X2(q, i));
                c.Store(H, second: true);
                string rstd = Rstd(c, c.Reduce("add", (q, i) => { string h = H(q, i); return c.Mul(h, h); }), n, c.A);
                c.Store((q, i) => c.Mul(c.Mul(H(q, i), rstd), c.G(q, i)));
                return;
            }

            case "add_layer_norm":
            {
                string H(int q, int i) => c.Add(c.X(q, i), c.X2(q, i));
                c.Store(H, second: true);
                string mu = c.Div(c.Reduce("add", H), n);
                string v = c.Reduce("add", (q, i) => { string d = c.Sub(H(q, i), mu); return c.Mul(d, d); });
                string rstd = Rstd(c, v, n, c.A);
                c.Store((q, i) => c.Fma(c.Mul(c.Sub(H(q, i), mu), rstd), c.G(q, i), c.Bt(q, i)));
                return;
            }

            default: throw new NotSupportedException($"row operation {op.Name} on the device");
        }
    }

    private static string Rstd(RowCtx c, string sumSq, string n, string eps) => c.Rcp(c.Sqrt(c.Add(c.Div(sumSq, n), eps)));
}
