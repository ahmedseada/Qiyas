// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The generic GPU side: one PTX generator per operation, written once and specialized by T at run time (the GPU's
// counterpart of a C# generic the JIT specializes per T). T's PTX names come from its description (PtxType.Of<T>), never
// from a per-type copy of the kernel. Where PTX has no instruction for T (no f64 exp, no f16 div/sqrt/rcp/min/max, no atan
// at all), the generator says so in Gaps: those steps run in f32 registers for that T. Where an operation's state comes in
// another type (an optimizer's moments, rope's tables), its arithmetic is in that type, T converted on load and store.
using System.Globalization;
using System.Numerics;
using System.Text;

namespace Qiyas;

/// <summary>T as PTX sees it: the register type, the memory type, the arithmetic suffix, its size and literals.</summary>
internal sealed record PtxType(string Name, string Reg, string Mem, string Op, int Size, bool IsFloat, string ZeroLiteral, string? NegInfLiteral, string? Log2ELiteral)
{
    public static PtxType Of<T>() where T : INumberBase<T>
    {
        if (typeof(T) == typeof(double)) return new("f64", "f64", "f64", "f64", 8, true, "0d0000000000000000", "0dFFF0000000000000", "0d3FF71547652B82FE");
        if (typeof(T) == typeof(float)) return new("f32", "f32", "f32", "f32", 4, true, "0f00000000", "0fFF800000", "0f3FB8AA3B");
        if (typeof(T) == typeof(Half)) return new("f16", "b16", "b16", "f16", 2, true, "0x0000", "0xFC00", "0x3DC5");
        if (typeof(T) == typeof(long)) return new("s64", "s64", "s64", "s64", 8, false, "0", null, null);
        if (typeof(T) == typeof(int)) return new("s32", "s32", "s32", "s32", 4, false, "0", null, null);
        if (typeof(T) == typeof(short)) return new("s16", "s16", "s16", "s16", 2, false, "0", null, null);
        if (typeof(T) == typeof(ushort)) return new("u16", "u16", "u16", "u16", 2, false, "0", null, null);
        if (typeof(T) == typeof(sbyte)) return new("s8", "s16", "s8", "s16", 1, false, "0", null, null);
        if (typeof(T) == typeof(ulong)) return new("u64", "u64", "u64", "u64", 8, false, "0", null, null);
        if (typeof(T) == typeof(uint)) return new("u32", "u32", "u32", "u32", 4, false, "0", null, null);
        if (typeof(T) == typeof(byte)) return new("u8", "u16", "u8", "u16", 1, false, "0", null, null);
        throw new NotSupportedException(typeof(T).Name);
    }

    public bool Half => Op == "f16";

    // A move of a literal into a register of this type (.b16 registers take mov.b16).
    public string MovLiteral(string target, string literal) => $"mov.{(Half ? "b16" : Reg)} {target}, {literal};";

    public string MulAdd(string d, string a, string b, string c) => IsFloat ? $"fma.rn.{Op} {d}, {a}, {b}, {c};" : $"mad.lo.{Op} {d}, {a}, {b}, {c};";

    // The literal of a float type's nearest value to v (its bits in hex).
    public string Literal(double v) => Name switch
    {
        "f64" => "0d" + BitConverter.DoubleToInt64Bits(v).ToString("X16", CultureInfo.InvariantCulture),
        "f32" => "0f" + BitConverter.SingleToInt32Bits((float)v).ToString("X8", CultureInfo.InvariantCulture),
        "f16" => "0x" + BitConverter.HalfToInt16Bits((Half)v).ToString("X4", CultureInfo.InvariantCulture),
        _ => throw new NotSupportedException(Name),
    };

    public string Header => Half ? ".version 7.0\n.target sm_75\n.address_size 64\n" : ".version 7.0\n.target sm_50\n.address_size 64\n";
}

internal static class GenericPtx
{
    private static string Shift(int size) => size switch { 1 => "0", 2 => "1", 4 => "2", _ => "3" };

    // c[i] = a[i] + b[i], a thread per element.
    public static string Add<T>(out string entry) where T : INumberBase<T>
    {
        var t = PtxType.Of<T>();
        entry = "add_" + t.Name;
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_n)
            {
                .reg .pred %p<2>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<8>;
                .reg .{{t.Reg}} %v<4>;
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ntid.x;
                mov.u32 %r3, %tid.x;
                mad.lo.u32 %r4, %r1, %r2, %r3;
                ld.param.u32 %r5, [p_n];
                setp.ge.u32 %p1, %r4, %r5;
                @%p1 bra DONE;
                mul.wide.u32 %rd1, %r4, {{t.Size}};
                ld.param.u64 %rd2, [p_a];
                ld.param.u64 %rd3, [p_b];
                ld.param.u64 %rd4, [p_c];
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                add.u64 %rd2, %rd2, %rd1;
                add.u64 %rd3, %rd3, %rd1;
                add.u64 %rd4, %rd4, %rd1;
                ld.global.{{t.Mem}} %v1, [%rd2];
                ld.global.{{t.Mem}} %v2, [%rd3];
                add.{{t.Op}} %v3, %v1, %v2;
                st.global.{{t.Mem}} [%rd4], %v3;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>
    /// C = A·B (row-major, [m,k]·[k,n]), sums in T: blocks of 16×16 threads, each thread a micro×micro tile of C in
    /// registers, the block a (16·micro)² tile; A and B through shared memory 16 columns of k at a time.
    /// </summary>
    public static string MatMul<T>(int micro, out string entry) where T : INumberBase<T>
    {
        var t = PtxType.Of<T>();
        int tile = 16 * micro, loads = tile * 16 / 256;                       // elements of each operand a thread loads per k step
        entry = $"gemm{tile}_{t.Name}";
        var s = new StringBuilder(t.Header);
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{entry}}(.param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_k)
            {
                .reg .pred %p<4>;
                .reg .b32 %r<40>;
                .reg .b64 %rd<12>;
                .reg .{{t.Reg}} %acc<{{micro * micro}}>;
                .reg .{{t.Reg}} %ra<{{micro}}>;
                .reg .{{t.Reg}} %rb<{{micro}}>;
                .reg .{{t.Reg}} %v;
                .reg .u32 %tx, %ty, %lin, %m, %n, %k, %t, %row0, %col0, %sa, %sb, %ga, %gk, %gc, %idx, %sh;
                .shared .align 8 .b8 As[{{16 * tile * t.Size}}];
                .shared .align 8 .b8 Bs[{{16 * tile * t.Size}}];
                mov.u32 %tx, %tid.x;
                mov.u32 %ty, %tid.y;
                mad.lo.u32 %lin, %ty, 16, %tx;
                mov.u32 %r1, %ctaid.x;
                mul.lo.u32 %col0, %r1, {{tile}};
                mov.u32 %r1, %ctaid.y;
                mul.lo.u32 %row0, %r1, {{tile}};
                ld.param.u32 %m, [p_m];
                ld.param.u32 %n, [p_n];
                ld.param.u32 %k, [p_k];
                ld.param.u64 %rd1, [p_a];
                cvta.to.global.u64 %rd1, %rd1;
                ld.param.u64 %rd2, [p_b];
                cvta.to.global.u64 %rd2, %rd2;
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd3, %rd3;
                mov.u32 %sa, As;
                mov.u32 %sb, Bs;
            """);
        for (int i = 0; i < micro * micro; i++)
        {
            s.AppendLine("    " + t.MovLiteral($"%acc{i}", t.ZeroLiteral));
        }

        s.AppendLine("""
                mov.u32 %t, 0;
            TILE:
                setp.ge.u32 %p1, %t, %k;
                @%p1 bra STORE;
            """);

        // The block's tiles of A (tile × 16, stored k-major: As[kc][row]) and B (16 × tile: Bs[kr][col]).
        for (int e = 0; e < loads; e++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %idx, %lin, {{e * 256}};
                    shr.u32 %r2, %idx, 4;
                    and.b32 %r3, %idx, 15;
                    add.u32 %ga, %row0, %r2;
                    add.u32 %gk, %t, %r3;
                    {{t.MovLiteral("%v", t.ZeroLiteral)}}
                    setp.lt.u32 %p2, %ga, %m;
                    setp.lt.and.u32 %p2, %gk, %k, %p2;
                    mad.lo.u32 %r4, %ga, %k, %gk;
                    mul.wide.u32 %rd4, %r4, {{t.Size}};
                    add.u64 %rd5, %rd1, %rd4;
                    @%p2 ld.global.{{t.Mem}} %v, [%rd5];
                    mad.lo.u32 %r5, %r3, {{tile}}, %r2;
                    shl.b32 %r5, %r5, {{Shift(t.Size)}};
                    add.u32 %sh, %sa, %r5;
                    st.shared.{{t.Mem}} [%sh], %v;
                    shr.u32 %r2, %idx, {{BitOperations.Log2((uint)tile)}};
                    and.b32 %r3, %idx, {{tile - 1}};
                    add.u32 %gk, %t, %r2;
                    add.u32 %gc, %col0, %r3;
                    {{t.MovLiteral("%v", t.ZeroLiteral)}}
                    setp.lt.u32 %p2, %gk, %k;
                    setp.lt.and.u32 %p2, %gc, %n, %p2;
                    mad.lo.u32 %r4, %gk, %n, %gc;
                    mul.wide.u32 %rd4, %r4, {{t.Size}};
                    add.u64 %rd5, %rd2, %rd4;
                    @%p2 ld.global.{{t.Mem}} %v, [%rd5];
                    mad.lo.u32 %r5, %r2, {{tile}}, %r3;
                    shl.b32 %r5, %r5, {{Shift(t.Size)}};
                    add.u32 %sh, %sb, %r5;
                    st.shared.{{t.Mem}} [%sh], %v;
                """);
        }

        s.AppendLine("    bar.sync 0;");
        // This thread's rows ty·micro.. and columns tx·micro..: base shared addresses, then 16 steps of k unrolled.
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                mul.lo.u32 %r6, %ty, {micro * t.Size};
                add.u32 %r6, %r6, %sa;
                mul.lo.u32 %r7, %tx, {micro * t.Size};
                add.u32 %r7, %r7, %sb;
            """);
        for (int kk = 0; kk < 16; kk++)
        {
            for (int i = 0; i < micro; i++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.{t.Mem} %ra{i}, [%r6+{(kk * tile + i) * t.Size}];");
            }

            for (int j = 0; j < micro; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.{t.Mem} %rb{j}, [%r7+{(kk * tile + j) * t.Size}];");
            }

            for (int i = 0; i < micro; i++)
            {
                for (int j = 0; j < micro; j++)
                {
                    s.AppendLine("    " + t.MulAdd($"%acc{i * micro + j}", $"%ra{i}", $"%rb{j}", $"%acc{i * micro + j}"));
                }
            }
        }

        s.AppendLine("""
                bar.sync 0;
                add.u32 %t, %t, 16;
                bra TILE;
            STORE:
            """);
        for (int i = 0; i < micro; i++)
        {
            for (int j = 0; j < micro; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"""
                        mad.lo.u32 %r8, %ty, {micro}, {i};
                        add.u32 %r8, %r8, %row0;
                        mad.lo.u32 %r9, %tx, {micro}, {j};
                        add.u32 %r9, %r9, %col0;
                        setp.lt.u32 %p3, %r8, %m;
                        setp.lt.and.u32 %p3, %r9, %n, %p3;
                        mad.lo.u32 %r10, %r8, %n, %r9;
                        mul.wide.u32 %rd6, %r10, {t.Size};
                        add.u64 %rd7, %rd3, %rd6;
                        @%p3 st.global.{t.Mem} [%rd7], %acc{i * micro + j};
                    """);
            }
        }

        s.AppendLine("    ret;\n}");
        return s.ToString();
    }

    // Steps of T for which PTX has no instruction, run in f32 registers by the row kernels below.
    public static string Gaps<T>(string operation) where T : INumberBase<T>
    {
        var t = PtxType.Of<T>();
        return (operation, t.Name) switch
        {
            ("softmax", "f64") => "exp in f32 (PTX has no f64 exp)",
            ("softmax", "f16") => "1/sum in f32 (PTX has no f16 rcp)",
            ("rms_norm", "f16") => "mean, sqrt and 1/x in f32 (PTX has no f16 div, sqrt or rcp)",
            ("optimizer", "f16") => "the step in f32, the moments' type",
            ("rope", "f16") => "the rotation in f32, the tables' type",
            ("box_loss", "f64") => "atan in f32 (PTX has no atan: a polynomial)",
            ("box_loss", "f32") => "atan by a polynomial (PTX has none)",
            ("box_loss", "f16") => "div, min, max and atan in f32 (PTX has no f16 div or atan, nor f16 min or max before sm_80)",
            ("ctc", _) => "exp and log in f32 (PTX's ex2 and lg2 approximations)",
            _ => "",
        };
    }

    // Block reduction of %acc over the block's 256 threads through shared memory in T (add or max), the result in %acc
    // for every thread.
    private static string Reduce(PtxType t, string label, bool max) => $$"""
            st.shared.{{t.Mem}} [%mine], %acc;
            bar.sync 0;
            mov.u32 %s, 128;
        {{label}}:
            setp.ge.u32 %p2, %tx, %s;
            @%p2 bra {{label}}_SKIP;
            shl.b32 %r9, %s, {{Shift(t.Size)}};
            add.u32 %r9, %r9, %mine;
            ld.shared.{{t.Mem}} %w, [%r9];
            ld.shared.{{t.Mem}} %acc, [%mine];
            {{(max ? $"setp.gt.{t.Op} %p3, %w, %acc;\n    selp.b{t.Size * 8} %acc, %w, %acc, %p3;" : $"add.{t.Op} %acc, %acc, %w;")}}
            st.shared.{{t.Mem}} [%mine], %acc;
        {{label}}_SKIP:
            bar.sync 0;
            shr.u32 %s, %s, 1;
            setp.ne.u32 %p2, %s, 0;
            @%p2 bra {{label}};
            ld.shared.{{t.Mem}} %acc, [%red];
            bar.sync 0;
        """;

    // Loop of this thread over columns tx, tx + 256, … of the row: %rd5 = the column's address in x, %x its value, %rd6
    // its address in y.
    private static string Columns(PtxType t, string label, string body, string source = "%rd5") => $$"""
            mov.u32 %c, %tx;
        {{label}}:
            setp.ge.u32 %p1, %c, %cols;
            @%p1 bra {{label}}_END;
            mul.wide.u32 %rd4, %c, {{t.Size}};
            add.u64 %rd5, %rd1, %rd4;
            add.u64 %rd6, %rd2, %rd4;
            ld.global.{{t.Mem}} %x, [{{source}}];
            {{body}}
            add.u32 %c, %c, 256;
            bra {{label}};
        {{label}}_END:
        """;

    private static string RowStart(PtxType t, string entry, string extraParams) => t.Header + $$"""
        .visible .entry {{entry}}(.param .u64 p_x, .param .u64 p_y{{extraParams}}, .param .u32 p_cols, .param .f32 p_eps)
        {
            .reg .pred %p<4>;
            .reg .b32 %r<12>;
            .reg .b64 %rd<10>;
            .reg .f32 %f<6>;
            .reg .f64 %d<2>;
            .reg .{{t.Reg}} %acc, %acc_max, %w, %x, %e, %scale;
            .reg .u32 %tx, %row, %cols, %c, %s, %mine, %red;
            .shared .align 8 .b8 red[{{256 * t.Size}}];
            mov.u32 %tx, %tid.x;
            mov.u32 %row, %ctaid.x;
            ld.param.u32 %cols, [p_cols];
            mov.u32 %red, red;
            shl.b32 %r1, %tx, {{Shift(t.Size)}};
            add.u32 %mine, %red, %r1;
            mul.lo.u32 %r2, %row, %cols;
            mul.wide.u32 %rd3, %r2, {{t.Size}};
            ld.param.u64 %rd1, [p_x];
            cvta.to.global.u64 %rd1, %rd1;
            add.u64 %rd1, %rd1, %rd3;
            ld.param.u64 %rd2, [p_y];
            cvta.to.global.u64 %rd2, %rd2;
            add.u64 %rd2, %rd2, %rd3;
        """;

    // e = exp(x - max) in T where PTX has the instruction (f32: ex2.approx.ftz.f32; f16: ex2.approx.f16, the instructions PTX has for exp in T).
    private static string Exp(PtxType t) => t.Name switch
    {
        "f32" => $"sub.f32 %e, %x, %acc_max;\n    mul.f32 %e, %e, {t.Log2ELiteral};\n    ex2.approx.ftz.f32 %e, %e;",
        "f16" => $"sub.f16 %e, %x, %acc_max;\n    mov.b16 %w, {t.Log2ELiteral};\n    mul.f16 %e, %e, %w;\n    ex2.approx.f16 %e, %e;",
        _ => "sub.f64 %e, %x, %acc_max;\n    cvt.rn.f32.f64 %f1, %e;\n    mul.f32 %f1, %f1, 0f3FB8AA3B;\n    ex2.approx.ftz.f32 %f1, %f1;\n    cvt.f64.f32 %e, %f1;",
    };

    // ------------------------------------------------------------------ convolution: im2col, col2im, transpose

    // The convolution's parameters as the kernels take them: H W kh kw sh sw pt pl dh dw OH OW (twelve u32).
    private const string ConvParams = ".param .u32 p_h, .param .u32 p_w, .param .u32 p_kh, .param .u32 p_kw, .param .u32 p_sh, .param .u32 p_sw, .param .u32 p_pt, .param .u32 p_pl, .param .u32 p_dh, .param .u32 p_dw, .param .u32 p_oh, .param .u32 p_ow";

    /// <summary>The columns of one image: a thread per column value (row r = (c, a, b), column q = (i, j)), x's value at
    /// (c, i·sh − pt + a·dh, j·sw − pl + b·dw) or 0 off the image. Arguments: x, cols, count (rows·OH·OW), then ConvParams.</summary>
    public static string Im2col<T>(out string entry) where T : INumberBase<T>
    {
        var t = PtxType.Of<T>();
        entry = "im2col_" + t.Name;
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_x, .param .u64 p_c, .param .u32 p_n, {{ConvParams}})
            {
                .reg .pred %p<4>;
                .reg .b32 %r<40>;
                .reg .b64 %rd<8>;
                .reg .{{t.Reg}} %v<2>;
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ntid.x;
                mov.u32 %r3, %tid.x;
                mad.lo.u32 %r4, %r1, %r2, %r3;
                ld.param.u32 %r5, [p_n];
                setp.ge.u32 %p1, %r4, %r5;
                @%p1 bra DONE;
                ld.param.u32 %r6, [p_h];
                ld.param.u32 %r7, [p_w];
                ld.param.u32 %r8, [p_kh];
                ld.param.u32 %r9, [p_kw];
                ld.param.u32 %r10, [p_sh];
                ld.param.u32 %r11, [p_sw];
                ld.param.u32 %r12, [p_pt];
                ld.param.u32 %r13, [p_pl];
                ld.param.u32 %r14, [p_dh];
                ld.param.u32 %r15, [p_dw];
                ld.param.u32 %r16, [p_oh];
                ld.param.u32 %r17, [p_ow];
                mul.lo.u32 %r18, %r16, %r17;          // OH·OW
                div.u32 %r19, %r4, %r18;              // r
                rem.u32 %r20, %r4, %r18;              // q
                div.u32 %r21, %r20, %r17;             // i
                rem.u32 %r22, %r20, %r17;             // j
                rem.u32 %r23, %r19, %r9;              // b
                div.u32 %r24, %r19, %r9;
                rem.u32 %r25, %r24, %r8;              // a
                div.u32 %r26, %r24, %r8;              // c
                mul.lo.u32 %r27, %r21, %r10;
                sub.u32 %r27, %r27, %r12;
                mad.lo.u32 %r27, %r25, %r14, %r27;    // ih (above the image: past H unsigned)
                mul.lo.u32 %r28, %r22, %r11;
                sub.u32 %r28, %r28, %r13;
                mad.lo.u32 %r28, %r23, %r15, %r28;    // iw
                {{t.MovLiteral("%v1", t.ZeroLiteral)}}
                setp.ge.u32 %p2, %r27, %r6;
                setp.ge.u32 %p3, %r28, %r7;
                or.pred %p2, %p2, %p3;
                @%p2 bra STORE;
                mad.lo.u32 %r29, %r26, %r6, %r27;
                mad.lo.u32 %r29, %r29, %r7, %r28;
                mul.wide.u32 %rd1, %r29, {{t.Size}};
                ld.param.u64 %rd2, [p_x];
                cvta.to.global.u64 %rd2, %rd2;
                add.u64 %rd2, %rd2, %rd1;
                ld.global.{{t.Mem}} %v1, [%rd2];
            STORE:
                mul.wide.u32 %rd3, %r4, {{t.Size}};
                ld.param.u64 %rd4, [p_c];
                cvta.to.global.u64 %rd4, %rd4;
                add.u64 %rd4, %rd4, %rd3;
                st.global.{{t.Mem}} [%rd4], %v1;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>The image's gradient from the columns' (col2im) as a gather: a thread per value of dx (c, ih, iw), the sum in T
    /// of every column value its position fed, taps in order. Arguments: dcols, dx, count (C·H·W), then ConvParams.</summary>
    public static string Col2im<T>(out string entry) where T : INumberBase<T>
    {
        var t = PtxType.Of<T>();
        entry = "col2im_" + t.Name;
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_c, .param .u64 p_x, .param .u32 p_n, {{ConvParams}})
            {
                .reg .pred %p<8>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<8>;
                .reg .{{t.Reg}} %v<4>;
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ntid.x;
                mov.u32 %r3, %tid.x;
                mad.lo.u32 %r4, %r1, %r2, %r3;
                ld.param.u32 %r5, [p_n];
                setp.ge.u32 %p1, %r4, %r5;
                @%p1 bra DONE;
                ld.param.u32 %r6, [p_h];
                ld.param.u32 %r7, [p_w];
                ld.param.u32 %r8, [p_kh];
                ld.param.u32 %r9, [p_kw];
                ld.param.u32 %r10, [p_sh];
                ld.param.u32 %r11, [p_sw];
                ld.param.u32 %r12, [p_pt];
                ld.param.u32 %r13, [p_pl];
                ld.param.u32 %r14, [p_dh];
                ld.param.u32 %r15, [p_dw];
                ld.param.u32 %r16, [p_oh];
                ld.param.u32 %r17, [p_ow];
                mul.lo.u32 %r18, %r6, %r7;
                div.u32 %r19, %r4, %r18;              // c
                rem.u32 %r20, %r4, %r18;
                div.u32 %r21, %r20, %r7;              // ih
                rem.u32 %r22, %r20, %r7;              // iw
                mul.lo.u32 %r23, %r16, %r17;          // OH·OW
                ld.param.u64 %rd1, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                {{t.MovLiteral("%v1", t.ZeroLiteral)}}
                mov.u32 %r24, 0;                      // a
            ROW:
                setp.ge.u32 %p2, %r24, %r8;
                @%p2 bra STORE;
                add.u32 %r25, %r21, %r12;
                mul.lo.u32 %r26, %r24, %r14;
                setp.lt.u32 %p3, %r25, %r26;
                @%p3 bra NEXTROW;
                sub.u32 %r25, %r25, %r26;             // ih + pt − a·dh
                rem.u32 %r27, %r25, %r10;
                setp.ne.u32 %p3, %r27, 0;
                @%p3 bra NEXTROW;
                div.u32 %r28, %r25, %r10;             // i
                setp.ge.u32 %p3, %r28, %r16;
                @%p3 bra NEXTROW;
                mov.u32 %r29, 0;                      // b
            COL:
                setp.ge.u32 %p4, %r29, %r9;
                @%p4 bra NEXTROW;
                add.u32 %r30, %r22, %r13;
                mul.lo.u32 %r31, %r29, %r15;
                setp.lt.u32 %p5, %r30, %r31;
                @%p5 bra NEXTCOL;
                sub.u32 %r30, %r30, %r31;
                rem.u32 %r32, %r30, %r11;
                setp.ne.u32 %p5, %r32, 0;
                @%p5 bra NEXTCOL;
                div.u32 %r33, %r30, %r11;             // j
                setp.ge.u32 %p5, %r33, %r17;
                @%p5 bra NEXTCOL;
                mad.lo.u32 %r34, %r19, %r8, %r24;
                mad.lo.u32 %r34, %r34, %r9, %r29;     // the columns' row (c, a, b)
                mad.lo.u32 %r35, %r28, %r17, %r33;    // the column (i, j)
                mad.lo.u32 %r34, %r34, %r23, %r35;
                mul.wide.u32 %rd2, %r34, {{t.Size}};
                add.u64 %rd2, %rd1, %rd2;
                ld.global.{{t.Mem}} %v2, [%rd2];
                add.{{t.Op}} %v1, %v1, %v2;
            NEXTCOL:
                add.u32 %r29, %r29, 1;
                bra COL;
            NEXTROW:
                add.u32 %r24, %r24, 1;
                bra ROW;
            STORE:
                mul.wide.u32 %rd3, %r4, {{t.Size}};
                ld.param.u64 %rd4, [p_x];
                cvta.to.global.u64 %rd4, %rd4;
                add.u64 %rd4, %rd4, %rd3;
                st.global.{{t.Mem}} [%rd4], %v1;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>t = xᵀ (x rows × cols): a thread per value. Arguments: x, t, rows, cols.</summary>
    public static string Transpose<T>(out string entry) where T : INumberBase<T>
    {
        var t = PtxType.Of<T>();
        entry = "transpose_" + t.Name;
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_x, .param .u64 p_t, .param .u32 p_rows, .param .u32 p_cols)
            {
                .reg .pred %p<2>;
                .reg .b32 %r<12>;
                .reg .b64 %rd<8>;
                .reg .{{t.Reg}} %v<2>;
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ntid.x;
                mov.u32 %r3, %tid.x;
                mad.lo.u32 %r4, %r1, %r2, %r3;
                ld.param.u32 %r5, [p_rows];
                ld.param.u32 %r6, [p_cols];
                mul.lo.u32 %r7, %r5, %r6;
                setp.ge.u32 %p1, %r4, %r7;
                @%p1 bra DONE;
                div.u32 %r8, %r4, %r6;                // i
                rem.u32 %r9, %r4, %r6;                // j
                mul.wide.u32 %rd1, %r4, {{t.Size}};
                ld.param.u64 %rd2, [p_x];
                cvta.to.global.u64 %rd2, %rd2;
                add.u64 %rd2, %rd2, %rd1;
                ld.global.{{t.Mem}} %v1, [%rd2];
                mad.lo.u32 %r10, %r9, %r5, %r8;       // j·rows + i
                mul.wide.u32 %rd3, %r10, {{t.Size}};
                ld.param.u64 %rd4, [p_t];
                cvta.to.global.u64 %rd4, %rd4;
                add.u64 %rd4, %rd4, %rd3;
                st.global.{{t.Mem}} [%rd4], %v1;
            DONE:
                ret;
            }

            """;
    }

    // Softmax over each row, one block of 256 threads a row: max, Σ exp(x - max) with exp stored in y, then y · 1/Σ.
    public static string Softmax<T>(out string entry) where T : IFloatingPointIeee754<T>
    {
        var t = PtxType.Of<T>();
        entry = "softmax_" + t.Name;
        string reciprocal = t.Name switch
        {
            "f16" => "cvt.f32.f16 %f1, %acc;\n    rcp.rn.f32 %f1, %f1;\n    cvt.rn.f16.f32 %scale, %f1;",
            _ => $"rcp.rn.{t.Op} %scale, %acc;",
        };
        return RowStart(t, entry, "") + $"""
                {t.MovLiteral("%acc", t.NegInfLiteral!)}
            """ + "\n" + Columns(t, "MAX", $"setp.gt.{t.Op} %p3, %x, %acc;\n    selp.b{t.Size * 8} %acc, %x, %acc, %p3;") + "\n"
            + Reduce(t, "RMAX", max: true) + "\n" + $"""
                mov.b{t.Size * 8} %acc_max, %acc;
                {t.MovLiteral("%acc", t.ZeroLiteral)}
            """ + "\n" + Columns(t, "EXP", Exp(t) + $"\n    st.global.{t.Mem} [%rd6], %e;\n    add.{t.Op} %acc, %acc, %e;") + "\n"
            + Reduce(t, "RSUM", max: false) + "\n" + reciprocal + "\n"
            + Columns(t, "NORM", $"mul.{t.Op} %x, %x, %scale;\n    st.global.{t.Mem} [%rd6], %x;", source: "%rd6")
            + "\n    ret;\n}\n";
    }

    // y = x · 1/sqrt(mean(x²) + eps) per row (inv stored), one block of 256 threads a row: Σ x² in T.
    public static string RmsNorm<T>(out string entry) where T : IFloatingPointIeee754<T>
    {
        var t = PtxType.Of<T>();
        entry = "rms_norm_" + t.Name;
        string scale = t.Name switch
        {
            "f64" => "cvt.rn.f64.u32 %d1, %cols;\n    div.rn.f64 %acc, %acc, %d1;\n    ld.param.f32 %f2, [p_eps];\n    cvt.f64.f32 %d1, %f2;\n    add.f64 %acc, %acc, %d1;\n    sqrt.rn.f64 %acc, %acc;\n    rcp.rn.f64 %scale, %acc;",
            "f32" => "cvt.rn.f32.u32 %f1, %cols;\n    div.rn.f32 %acc, %acc, %f1;\n    ld.param.f32 %f2, [p_eps];\n    add.f32 %acc, %acc, %f2;\n    sqrt.rn.f32 %acc, %acc;\n    rcp.rn.f32 %scale, %acc;",
            _ => "cvt.f32.f16 %f3, %acc;\n    cvt.rn.f32.u32 %f1, %cols;\n    div.rn.f32 %f3, %f3, %f1;\n    ld.param.f32 %f2, [p_eps];\n    add.f32 %f3, %f3, %f2;\n    sqrt.rn.f32 %f3, %f3;\n    rcp.rn.f32 %f3, %f3;\n    cvt.rn.f16.f32 %scale, %f3;",
        };
        return RowStart(t, entry, ", .param .u64 p_inv") + "\n    " + t.MovLiteral("%acc", t.ZeroLiteral) + "\n"
            + Columns(t, "SQ", t.MulAdd("%acc", "%x", "%x", "%acc")) + "\n"
            + Reduce(t, "RSUM", max: false) + "\n" + scale + $"""

                setp.ne.u32 %p1, %tx, 0;
                @%p1 bra ROWS;
                ld.param.u64 %rd7, [p_inv];
                cvta.to.global.u64 %rd7, %rd7;
                mul.wide.u32 %rd8, %row, {t.Size};
                add.u64 %rd7, %rd7, %rd8;
                st.global.{t.Mem} [%rd7], %scale;
            ROWS:
            """ + "\n" + Columns(t, "RW", $"mul.{t.Op} %x, %x, %scale;\n    st.global.{t.Mem} [%rd6], %x;") + "\n    ret;\n}\n";
    }

    // The thread's element: %r4 = ctaid·ntid + tid; past p_n it is done.
    private const string Element = """
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mov.u32 %r3, %tid.x;
            mad.lo.u32 %r4, %r1, %r2, %r3;
            ld.param.u32 %r5, [p_n];
            setp.ge.u32 %p1, %r4, %r5;
            @%p1 bra DONE;
        """;

    // A value of T in register from into a register of the computing type m (the same, or f16 widened to f32), and back.
    private static string Widen(PtxType t, PtxType m, string to, string from) => t.Name == m.Name ? $"mov.{m.Reg} {to}, {from};" : $"cvt.{m.Op}.{t.Op} {to}, {from};";

    private static string Narrow(PtxType t, PtxType m, string to, string from) => t.Name == m.Name ? $"mov.{m.Reg} {to}, {from};" : $"cvt.rn.{t.Op}.{m.Op} {to}, {from};";

    /// <summary>
    /// One optimizer step (sgd, sgd_momentum, adam, adamw; CpuGeneric.Step's formulas), a thread per value: p and g of T,
    /// the moments m, v of TM, the step in TM; its scalars (lr, b1, b2, eps, wd, c1, c2, 1 − b1, 1 − b2) and the gradient's
    /// scale as TM parameters.
    /// </summary>
    public static string Optimizer<T, TM>(string name, out string entry) where T : INumberBase<T> where TM : INumberBase<TM>
    {
        PtxType t = PtxType.Of<T>(), m = PtxType.Of<TM>();
        entry = $"{name}_{t.Name}_{m.Name}";
        string o = m.Op, scalars = string.Concat(new[] { "lr", "b1", "b2", "eps", "wd", "c1", "c2", "b1c", "b2c", "k" }.Select(s => $", .param .{o} p_{s}"));
        string loads = string.Concat(new[] { "lr", "b1", "b2", "eps", "wd", "c1", "c2", "b1c", "b2c", "k" }.Select(s => $"    ld.param.{o} %{s}, [p_{s}];\n"));
        string step = name switch
        {
            "sgd" => $"""
                    mul.rn.{o} %u, %wd, %pp;
                    add.rn.{o} %u, %gg, %u;
                    mul.rn.{o} %u, %lr, %u;
                    sub.rn.{o} %pp, %pp, %u;
                """,
            "sgd_momentum" => $"""
                    ld.global.{o} %m, [%rd5];
                    mul.rn.{o} %m, %b1, %m;
                    add.rn.{o} %m, %m, %gg;
                    mul.rn.{o} %u, %wd, %pp;
                    add.rn.{o} %m, %m, %u;
                    st.global.{o} [%rd5], %m;
                    mul.rn.{o} %u, %lr, %m;
                    sub.rn.{o} %pp, %pp, %u;
                """,
            _ => $"""
                    ld.global.{o} %m, [%rd5];
                    ld.global.{o} %v, [%rd6];
                    mul.rn.{o} %m, %b1, %m;
                    mul.rn.{o} %u, %b1c, %gg;
                    add.rn.{o} %m, %m, %u;
                    mul.rn.{o} %v, %b2, %v;
                    mul.rn.{o} %u, %b2c, %gg;
                    mul.rn.{o} %u, %u, %gg;
                    add.rn.{o} %v, %v, %u;
                    st.global.{o} [%rd5], %m;
                    st.global.{o} [%rd6], %v;
                    div.rn.{o} %u, %m, %c1;
                    mul.rn.{o} %u, %lr, %u;
                    div.rn.{o} %w, %v, %c2;
                    sqrt.rn.{o} %w, %w;
                    add.rn.{o} %w, %w, %eps;
                    div.rn.{o} %u, %u, %w;
                """ + (name == "adamw" ? $"""

                    mul.rn.{o} %w, %lr, %wd;
                    mul.rn.{o} %w, %w, %pp;
                    sub.rn.{o} %pp, %pp, %w;
                """ : "") + $"""

                    sub.rn.{o} %pp, %pp, %u;
                """,
        };
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_p, .param .u64 p_g, .param .u64 p_m, .param .u64 p_v, .param .u32 p_n{{scalars}})
            {
                .reg .pred %p<2>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<8>;
                .reg .{{t.Reg}} %x;
                .reg .{{o}} %lr, %b1, %b2, %eps, %wd, %c1, %c2, %b1c, %b2c, %k, %pp, %gg, %m, %v, %u, %w;
            {{Element}}
                mul.wide.u32 %rd1, %r4, {{t.Size}};
                mul.wide.u32 %rd2, %r4, {{m.Size}};
                ld.param.u64 %rd3, [p_p];
                cvta.to.global.u64 %rd3, %rd3;
                add.u64 %rd3, %rd3, %rd1;
                ld.param.u64 %rd4, [p_g];
                cvta.to.global.u64 %rd4, %rd4;
                add.u64 %rd4, %rd4, %rd1;
                ld.param.u64 %rd5, [p_m];
                cvta.to.global.u64 %rd5, %rd5;
                add.u64 %rd5, %rd5, %rd2;
                ld.param.u64 %rd6, [p_v];
                cvta.to.global.u64 %rd6, %rd6;
                add.u64 %rd6, %rd6, %rd2;
            {{loads}}
                ld.global.{{t.Mem}} %x, [%rd3];
                {{Widen(t, m, "%pp", "%x")}}
                ld.global.{{t.Mem}} %x, [%rd4];
                {{Widen(t, m, "%gg", "%x")}}
                mul.rn.{{o}} %gg, %gg, %k;
            {{step}}
                {{Narrow(t, m, "%x", "%pp")}}
                st.global.{{t.Mem}} [%rd3], %x;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>Σ g² of T in TM: each block of 256 threads over the values ctaid·256 + tid, + nctaid·256, …, its sum (a
    /// tree in shared memory) into p_out[ctaid].</summary>
    public static string SumSquares<T, TM>(out string entry) where T : INumberBase<T> where TM : INumberBase<TM>
    {
        PtxType t = PtxType.Of<T>(), m = PtxType.Of<TM>();
        entry = $"sumsq_{t.Name}_{m.Name}";
        string o = m.Op;
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_g, .param .u64 p_out, .param .u32 p_n)
            {
                .reg .pred %p<3>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<6>;
                .reg .{{t.Reg}} %x;
                .reg .{{o}} %acc, %y, %w;
                .reg .u32 %tx, %i, %stride, %n, %s, %mine, %red;
                .shared .align 8 .b8 red[{{256 * m.Size}}];
                mov.u32 %tx, %tid.x;
                mov.u32 %r1, %ctaid.x;
                mad.lo.u32 %i, %r1, 256, %tx;
                mov.u32 %r2, %nctaid.x;
                mul.lo.u32 %stride, %r2, 256;
                ld.param.u32 %n, [p_n];
                ld.param.u64 %rd1, [p_g];
                cvta.to.global.u64 %rd1, %rd1;
                {{m.MovLiteral("%acc", m.ZeroLiteral)}}
            LOOP:
                setp.ge.u32 %p1, %i, %n;
                @%p1 bra REDUCE;
                mul.wide.u32 %rd2, %i, {{t.Size}};
                add.u64 %rd3, %rd1, %rd2;
                ld.global.{{t.Mem}} %x, [%rd3];
                {{Widen(t, m, "%y", "%x")}}
                fma.rn.{{o}} %acc, %y, %y, %acc;
                add.u32 %i, %i, %stride;
                bra LOOP;
            REDUCE:
                mov.u32 %red, red;
                shl.b32 %r3, %tx, {{Shift(m.Size)}};
                add.u32 %mine, %red, %r3;
                st.shared.{{o}} [%mine], %acc;
                bar.sync 0;
                mov.u32 %s, 128;
            RLOOP:
                setp.ge.u32 %p2, %tx, %s;
                @%p2 bra RSKIP;
                shl.b32 %r4, %s, {{Shift(m.Size)}};
                add.u32 %r4, %r4, %mine;
                ld.shared.{{o}} %w, [%r4];
                ld.shared.{{o}} %acc, [%mine];
                add.rn.{{o}} %acc, %acc, %w;
                st.shared.{{o}} [%mine], %acc;
            RSKIP:
                bar.sync 0;
                shr.u32 %s, %s, 1;
                setp.ne.u32 %p2, %s, 0;
                @%p2 bra RLOOP;
                setp.ne.u32 %p2, %tx, 0;
                @%p2 bra DONE;
                ld.shared.{{o}} %acc, [%red];
                ld.param.u64 %rd4, [p_out];
                cvta.to.global.u64 %rd4, %rd4;
                mul.wide.u32 %rd5, %r1, {{m.Size}};
                add.u64 %rd4, %rd4, %rd5;
                st.global.{{o}} [%rd4], %acc;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>
    /// The rotary embedding by tables (CpuGeneric.Rope's definition), a thread per output value: column c of row r below
    /// 2·half is its pair's rotation (the pair and its side from c: interleaved (2p, 2p + 1), else (p, p + half)) by the
    /// tables of TL at the row's position, in TL; past 2·half the value copied. p_interleaved, p_inverse: 0 or 1.
    /// </summary>
    public static string Rope<T, TL>(out string entry) where T : INumberBase<T> where TL : INumberBase<TL>
    {
        PtxType t = PtxType.Of<T>(), l = PtxType.Of<TL>();
        entry = $"rope_{t.Name}_{l.Name}";
        string o = l.Op;
        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_x, .param .u64 p_cos, .param .u64 p_sin, .param .u64 p_pos, .param .u64 p_y, .param .u32 p_n,
                .param .u32 p_cols, .param .u32 p_heads, .param .u32 p_steps, .param .u32 p_half, .param .u32 p_interleaved, .param .u32 p_inverse)
            {
                .reg .pred %p<6>;
                .reg .b32 %r<32>;
                .reg .b64 %rd<16>;
                .reg .{{t.Reg}} %x;
                .reg .{{o}} %xi, %xj, %c, %s, %lo, %hi, %u;
            {{Element}}
                ld.param.u32 %r6, [p_cols];
                div.u32 %r7, %r4, %r6;
                mul.lo.u32 %r8, %r7, %r6;
                sub.u32 %r8, %r4, %r8;
                ld.param.u64 %rd1, [p_x];
                cvta.to.global.u64 %rd1, %rd1;
                ld.param.u64 %rd2, [p_y];
                cvta.to.global.u64 %rd2, %rd2;
                mul.wide.u32 %rd3, %r4, {{t.Size}};
                add.u64 %rd4, %rd2, %rd3;
                ld.param.u32 %r9, [p_half];
                shl.b32 %r10, %r9, 1;
                setp.lt.u32 %p2, %r8, %r10;
                @%p2 bra ROTATE;
                add.u64 %rd5, %rd1, %rd3;
                ld.global.{{t.Mem}} %x, [%rd5];
                st.global.{{t.Mem}} [%rd4], %x;
                bra DONE;
            ROTATE:
                ld.param.u32 %r11, [p_heads];
                ld.param.u32 %r12, [p_steps];
                div.u32 %r13, %r7, %r11;
                rem.u32 %r13, %r13, %r12;
                ld.param.u64 %rd6, [p_pos];
                cvta.to.global.u64 %rd6, %rd6;
                mul.wide.u32 %rd7, %r13, 4;
                add.u64 %rd6, %rd6, %rd7;
                ld.global.u32 %r14, [%rd6];
                ld.param.u32 %r15, [p_interleaved];
                setp.ne.u32 %p3, %r15, 0;
                shr.u32 %r16, %r8, 1;
                and.b32 %r17, %r8, 1;
                setp.ne.u32 %p4, %r17, 0;
                setp.ge.u32 %p5, %r8, %r9;
                sub.u32 %r18, %r8, %r9;
                selp.u32 %r18, %r18, %r8, %p5;
                selp.u32 %r19, %r16, %r18, %p3;
                @%p3 mov.pred %p5, %p4;
                shl.b32 %r20, %r19, 1;
                selp.u32 %r21, %r20, %r19, %p3;
                add.u32 %r22, %r20, 1;
                add.u32 %r23, %r19, %r9;
                selp.u32 %r22, %r22, %r23, %p3;
                sub.u32 %r24, %r4, %r8;
                add.u32 %r25, %r24, %r21;
                mul.wide.u32 %rd8, %r25, {{t.Size}};
                add.u64 %rd8, %rd1, %rd8;
                ld.global.{{t.Mem}} %x, [%rd8];
                {{Widen(t, l, "%xi", "%x")}}
                add.u32 %r26, %r24, %r22;
                mul.wide.u32 %rd9, %r26, {{t.Size}};
                add.u64 %rd9, %rd1, %rd9;
                ld.global.{{t.Mem}} %x, [%rd9];
                {{Widen(t, l, "%xj", "%x")}}
                mad.lo.u32 %r27, %r14, %r9, %r19;
                mul.wide.u32 %rd10, %r27, {{l.Size}};
                ld.param.u64 %rd11, [p_cos];
                cvta.to.global.u64 %rd11, %rd11;
                add.u64 %rd11, %rd11, %rd10;
                ld.global.{{o}} %c, [%rd11];
                ld.param.u64 %rd12, [p_sin];
                cvta.to.global.u64 %rd12, %rd12;
                add.u64 %rd12, %rd12, %rd10;
                ld.global.{{o}} %s, [%rd12];
                ld.param.u32 %r28, [p_inverse];
                setp.ne.u32 %p1, %r28, 0;
                @%p1 neg.{{o}} %s, %s;
                mul.rn.{{o}} %lo, %xi, %c;
                mul.rn.{{o}} %u, %xj, %s;
                sub.rn.{{o}} %lo, %lo, %u;
                mul.rn.{{o}} %hi, %xi, %s;
                mul.rn.{{o}} %u, %xj, %c;
                add.rn.{{o}} %hi, %hi, %u;
                selp.b{{l.Size * 8}} %lo, %hi, %lo, %p5;
                {{Narrow(t, l, "%x", "%lo")}}
                st.global.{{t.Mem}} [%rd4], %x;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>
    /// The box overlap loss of a pair (backward: its gradient times dloss), a thread per pair, in T (CpuGeneric.BoxLoss's
    /// formulas in their order): the steps PTX lacks for T in f32 registers (Gaps), atan by a polynomial (PTX has none).
    /// </summary>
    public static string BoxLoss<T>(BoxOverlap overlap, bool backward, out string entry) where T : IFloatingPointIeee754<T>
    {
        var t = PtxType.Of<T>();
        entry = $"box_{overlap.ToString().ToLowerInvariant()}{(backward ? "_bwd" : "")}_{t.Name}";
        var e = new Scalar(t);
        double D(T value) => double.CreateTruncating(value);
        string zero = e.Const(0), one = e.Const(1), two = e.Const(2), half = e.Const(0.5);
        string eps = e.Const(D(T.BitIncrement(T.One) - T.One)), aspect = e.Const(D(T.CreateTruncating(4) / (T.Pi * T.Pi)));
        int z = t.Size;
        string x1 = e.Load("%rd2"), y1 = e.Load($"%rd2+{z}"), x2 = e.Load($"%rd2+{2 * z}"), y2 = e.Load($"%rd2+{3 * z}");
        string gx1 = e.Load("%rd3"), gy1 = e.Load($"%rd3+{z}"), gx2 = e.Load($"%rd3+{2 * z}"), gy2 = e.Load($"%rd3+{3 * z}");
        string iw = e.Sub(e.Min(x2, gx2), e.Max(x1, gx1)), ih = e.Sub(e.Min(y2, gy2), e.Max(y1, gy1));
        string overlaps = e.And(e.Gt(iw, zero), e.Gt(ih, zero));
        string inter = e.Select(overlaps, e.Mul(iw, ih), zero);
        string wp = e.Sub(x2, x1), hp = e.Sub(y2, y1), wg = e.Sub(gx2, gx1), hg = e.Sub(gy2, gy1);
        string union = e.Sub(e.Add(e.Mul(wp, hp), e.Mul(wg, hg)), inter), ue = e.Add(union, eps), iou = e.Div(inter, ue);
        string cw = "", ch = "";
        if (overlap != BoxOverlap.IoU)
        {
            cw = e.Sub(e.Max(x2, gx2), e.Min(x1, gx1));
            ch = e.Sub(e.Max(y2, gy2), e.Min(y1, gy1));
        }

        if (!backward)
        {
            string loss = e.Sub(one, iou);
            if (overlap == BoxOverlap.GIoU)
            {
                string area = e.Mul(cw, ch);
                loss = e.Add(loss, e.Div(e.Sub(area, union), e.Add(area, eps)));
            }
            else if (overlap != BoxOverlap.IoU)
            {
                string diag = e.Add(e.Add(e.Mul(cw, cw), e.Mul(ch, ch)), eps);
                string dx = e.Mul(e.Add(e.Sub(x1, gx1), e.Sub(x2, gx2)), half), dy = e.Mul(e.Add(e.Sub(y1, gy1), e.Sub(y2, gy2)), half);
                loss = e.Add(loss, e.Div(e.Add(e.Mul(dx, dx), e.Mul(dy, dy)), diag));
                if (overlap == BoxOverlap.CIoU)
                {
                    string diff = e.Sub(e.Atan(e.Div(wg, hg)), e.Atan(e.Div(wp, hp)));
                    string v = e.Mul(e.Mul(aspect, diff), diff), alpha = e.Div(v, e.Add(e.Add(e.Sub(one, iou), v), eps));
                    loss = e.Add(loss, e.Mul(alpha, v));
                }
            }

            e.Store("%rd6", loss);
        }
        else
        {
            string dInter = e.Div(e.Const(-1), ue), dUnion = e.Div(iou, ue), dCw = zero, dCh = zero, dPx = zero, dPy = zero, dW = zero, dH = zero;
            if (overlap == BoxOverlap.GIoU)
            {
                string area = e.Mul(cw, ch), ae = e.Add(area, eps);
                dUnion = e.Sub(dUnion, e.Div(one, ae));
                string dArea = e.Div(ue, e.Mul(ae, ae));
                dCw = e.Mul(dArea, ch);
                dCh = e.Mul(dArea, cw);
            }
            else if (overlap != BoxOverlap.IoU)
            {
                string diag = e.Add(e.Add(e.Mul(cw, cw), e.Mul(ch, ch)), eps);
                string dx = e.Mul(e.Add(e.Sub(x1, gx1), e.Sub(x2, gx2)), half), dy = e.Mul(e.Add(e.Sub(y1, gy1), e.Sub(y2, gy2)), half);
                string rho = e.Add(e.Mul(dx, dx), e.Mul(dy, dy));
                dPx = e.Div(e.Mul(two, dx), diag);
                dPy = e.Div(e.Mul(two, dy), diag);
                string dDiag = e.Neg(e.Div(rho, e.Mul(diag, diag)));
                dCw = e.Mul(e.Mul(two, dDiag), cw);
                dCh = e.Mul(e.Mul(two, dDiag), ch);
                if (overlap == BoxOverlap.CIoU)
                {
                    string diff = e.Sub(e.Atan(e.Div(wg, hg)), e.Atan(e.Div(wp, hp)));
                    string v = e.Mul(e.Mul(aspect, diff), diff), alpha = e.Div(v, e.Add(e.Add(e.Sub(one, iou), v), eps));
                    string held = e.Sub(one, e.Mul(alpha, alpha));
                    dInter = e.Mul(dInter, held);
                    dUnion = e.Mul(dUnion, held);
                    string dv = e.Mul(alpha, e.Sub(two, alpha)), r2 = e.Add(e.Mul(wp, wp), e.Mul(hp, hp));
                    string k = e.Div(e.Mul(e.Mul(e.Mul(two, aspect), dv), diff), r2);
                    dW = e.Neg(e.Mul(k, hp));
                    dH = e.Mul(k, wp);
                }
            }

            dInter = e.Sub(dInter, dUnion);
            dW = e.Add(dW, e.Mul(dUnion, hp));
            dH = e.Add(dH, e.Mul(dUnion, wp));
            string ix = e.Select(overlaps, e.Mul(ih, dInter), zero), iy = e.Select(overlaps, e.Mul(iw, dInter), zero);
            string Above(string a, string b) => e.Select(e.Gt(a, b), one, e.Select(e.Eq(a, b), one, zero));      // a tie: the predicted corner takes it all (Boxes.Above)
            string Below(string a, string b) => e.Select(e.Lt(a, b), one, e.Select(e.Eq(a, b), one, zero));      // (Boxes.Below)
            string dl = e.Load("%rd5");
            e.Store("%rd6", e.Mul(e.Sub(e.Sub(e.Add(e.Neg(dW), e.Mul(dPx, half)), e.Mul(ix, Above(x1, gx1))), e.Mul(dCw, Below(x1, gx1))), dl));
            e.Store($"%rd6+{z}", e.Mul(e.Sub(e.Sub(e.Add(e.Neg(dH), e.Mul(dPy, half)), e.Mul(iy, Above(y1, gy1))), e.Mul(dCh, Below(y1, gy1))), dl));
            e.Store($"%rd6+{2 * z}", e.Mul(e.Add(e.Add(e.Add(dW, e.Mul(dPx, half)), e.Mul(ix, Below(x2, gx2))), e.Mul(dCw, Above(x2, gx2))), dl));
            e.Store($"%rd6+{3 * z}", e.Mul(e.Add(e.Add(e.Add(dH, e.Mul(dPy, half)), e.Mul(iy, Below(y2, gy2))), e.Mul(dCh, Above(y2, gy2))), dl));
        }

        return t.Header + $$"""
            .visible .entry {{entry}}(.param .u64 p_p, .param .u64 p_t, .param .u64 p_dl, .param .u64 p_y, .param .u32 p_n)
            {
                .reg .pred %p<2>;
                .reg .b32 %r<6>;
                .reg .b64 %rd<8>;
                {{e.Declarations}}
            {{Element}}
                mul.wide.u32 %rd1, %r4, {{4 * z}};
                mul.wide.u32 %rd4, %r4, {{z}};
                ld.param.u64 %rd2, [p_p];
                cvta.to.global.u64 %rd2, %rd2;
                add.u64 %rd2, %rd2, %rd1;
                ld.param.u64 %rd3, [p_t];
                cvta.to.global.u64 %rd3, %rd3;
                add.u64 %rd3, %rd3, %rd1;
                ld.param.u64 %rd5, [p_dl];
                cvta.to.global.u64 %rd5, %rd5;
                add.u64 %rd5, %rd5, %rd4;
                ld.param.u64 %rd6, [p_y];
                cvta.to.global.u64 %rd6, %rd6;
                add.u64 %rd6, %rd6, {{(backward ? "%rd1" : "%rd4")}};
            {{e.Code}}
            DONE:
                ret;
            }

            """;
    }

    /// <summary>
    /// ctc (CpuGeneric.Ctc's definition and order), a block of 256 threads per sequence b, the frame loops inside: each
    /// frame's log-sum-exp over its C logits (block reductions in T) into p_lse; α frame by frame, a thread per state, into
    /// p_alpha; the loss by thread 0; with p_withgrad, β back from the last frame into p_beta, then per frame the row's
    /// exp(x − lse) (0 past the sequence's frames) less each state's exp(α + β − log y − log P), added by atomic adds (states
    /// share a class). The scratch is the sequence's at b·T (lse) and b·T·p_states (α, β). exp and log in f32 (Gaps);
    /// the f64 kernel needs sm_60 for its atomic add.
    /// </summary>
    public static string Ctc<T>(out string entry) where T : IFloatingPointIeee754<T>
    {
        var t = PtxType.Of<T>();
        entry = "ctc_" + t.Name;
        string o = t.Op, mem = t.Mem, z = t.Size.ToString(CultureInfo.InvariantCulture), bits = (t.Size * 8).ToString(CultureInfo.InvariantCulture);
        string header = t.Name == "f64" ? ".version 7.0\n.target sm_60\n.address_size 64\n" : t.Header;
        string Widen32(string s) => t.Name switch { "f32" => $"mov.f32 %fe, {s};", "f64" => $"cvt.rn.f32.f64 %fe, {s};", _ => $"cvt.f32.f16 %fe, {s};" };
        string Narrow32(string d) => t.Name switch { "f32" => $"mov.f32 {d}, %fe;", "f64" => $"cvt.f64.f32 {d}, %fe;", _ => $"cvt.rn.f16.f32 {d}, %fe;" };
        string Exp(string d, string s) => $"{Widen32(s)}\n    mul.f32 %fe, %fe, 0f3FB8AA3B;\n    ex2.approx.ftz.f32 %fe, %fe;\n    {Narrow32(d)}";
        string Log(string d, string s) => $"{Widen32(s)}\n    lg2.approx.ftz.f32 %fe, %fe;\n    mul.f32 %fe, %fe, 0f3F317218;\n    {Narrow32(d)}";
        string red = t.Name == "f16" ? "red.global.add.noftz.f16" : $"red.global.add.{o}";

        // %a0..%a2 → %res = log(e^a0 + e^a1 + e^a2) by the largest (−∞ when all are).
        string lse3 = $"""
                setp.gt.{o} %q7, %a2, %a1;
                selp.b{bits} %mx, %a2, %a1, %q7;
                setp.gt.{o} %q7, %mx, %a0;
                selp.b{bits} %mx, %mx, %a0, %q7;
                sub.rn.{o} %e, %a0, %mx;
                {Exp("%sum", "%e")}
                sub.rn.{o} %e, %a1, %mx;
                {Exp("%e", "%e")}
                add.rn.{o} %sum, %sum, %e;
                sub.rn.{o} %e, %a2, %mx;
                {Exp("%e", "%e")}
                add.rn.{o} %sum, %sum, %e;
                {Log("%res", "%sum")}
                add.rn.{o} %res, %mx, %res;
                setp.eq.{o} %q7, %mx, %ninf;
                selp.b{bits} %res, %mx, %res, %q7;
            """;

        // The class of state %st into %cls (the blank at even states).
        const string Class = """
                and.b32 %r1, %st, 1;
                setp.eq.u32 %q3, %r1, 0;
                mov.u32 %cls, %blank;
                shr.u32 %r2, %st, 1;
                add.u32 %r2, %r2, %off;
                mul.wide.u32 %rd7, %r2, 4;
                add.u64 %rd7, %rdlab, %rd7;
                @!%q3 ld.global.u32 %cls, [%rd7];
            """;

        // log y of frame f at class %cls into %ly: x[(f·B + b)·C + cls] − lse[f].
        string LogY(string f) => $"""
                mad.lo.u32 %r3, {f}, %nB, %b;
                mad.lo.u32 %r3, %r3, %nC, %cls;
                mul.wide.u32 %rd8, %r3, {z};
                add.u64 %rd8, %rdx, %rd8;
                ld.global.{mem} %x, [%rd8];
                mul.wide.u32 %rd9, {f}, {z};
                add.u64 %rd9, %rdlse, %rd9;
                ld.global.{mem} %lse, [%rd9];
                sub.rn.{o} %ly, %x, %lse;
            """;

        // %q4: state s may come from s − 2 (odd s ≥ 3 whose label differs from the label before it).
        string Skip(string s) => $"""
                and.b32 %r4, {s}, 1;
                setp.ne.u32 %q4, %r4, 0;
                setp.ge.and.u32 %q4, {s}, 3, %q4;
                shr.u32 %r5, {s}, 1;
                add.u32 %r5, %r5, %off;
                mul.wide.u32 %rd10, %r5, 4;
                add.u64 %rd10, %rdlab, %rd10;
                mov.u32 %r6, 0;
                mov.u32 %r7, 0;
                @%q4 ld.global.u32 %r6, [%rd10];
                @%q4 ld.global.u32 %r7, [%rd10+-4];
                setp.ne.and.u32 %q4, %r6, %r7, %q4;
            """;

        // The address of (frame f, state s) of α or β (base) into rd.
        string At(string b, string f, string s, string rd) => $"""
                mad.lo.u32 %r10, {f}, %states, {s};
                mul.wide.u32 {rd}, %r10, {z};
                add.u64 {rd}, {b}, {rd};
            """;

        return header + $$"""
            .visible .entry {{entry}}(.param .u64 p_x, .param .u64 p_loss, .param .u64 p_grad, .param .u64 p_lse, .param .u64 p_alpha, .param .u64 p_beta,
                .param .u64 p_frames, .param .u64 p_lengths, .param .u64 p_offsets, .param .u64 p_labels,
                .param .u32 p_T, .param .u32 p_B, .param .u32 p_C, .param .u32 p_blank, .param .u32 p_states, .param .u32 p_withgrad)
            {
                .reg .pred %p<4>;
                .reg .pred %q<8>;
                .reg .b32 %r<16>;
                .reg .b64 %rd<24>;
                .reg .b64 %rdx, %rdlse, %rdal, %rdbe, %rdlab, %rdg;
                .reg .f32 %fe;
                .reg .{{t.Reg}} %acc, %w, %x, %m, %e, %lse, %a0, %a1, %a2, %mx, %sum, %ly, %ninf, %zero, %logp, %res;
                .reg .u32 %tx, %s, %mine, %red, %b, %frames, %S, %off, %nT, %nB, %nC, %blank, %states, %ft, %fp, %fl, %c, %st, %st2, %cls, %row, %flag;
                .shared .align 8 .b8 red[{{256 * t.Size}}];
                mov.u32 %tx, %tid.x;
                mov.u32 %b, %ctaid.x;
                mov.u32 %red, red;
                shl.b32 %r1, %tx, {{Shift(t.Size)}};
                add.u32 %mine, %red, %r1;
                ld.param.u32 %nT, [p_T];
                ld.param.u32 %nB, [p_B];
                ld.param.u32 %nC, [p_C];
                ld.param.u32 %blank, [p_blank];
                ld.param.u32 %states, [p_states];
                {{t.MovLiteral("%ninf", t.NegInfLiteral!)}}
                {{t.MovLiteral("%zero", t.ZeroLiteral)}}
                mul.wide.u32 %rd1, %b, 4;
                ld.param.u64 %rd2, [p_frames];
                cvta.to.global.u64 %rd2, %rd2;
                add.u64 %rd2, %rd2, %rd1;
                ld.global.u32 %frames, [%rd2];
                ld.param.u64 %rd2, [p_lengths];
                cvta.to.global.u64 %rd2, %rd2;
                add.u64 %rd2, %rd2, %rd1;
                ld.global.u32 %S, [%rd2];
                shl.b32 %S, %S, 1;
                add.u32 %S, %S, 1;
                ld.param.u64 %rd2, [p_offsets];
                cvta.to.global.u64 %rd2, %rd2;
                add.u64 %rd2, %rd2, %rd1;
                ld.global.u32 %off, [%rd2];
                ld.param.u64 %rdlab, [p_labels];
                cvta.to.global.u64 %rdlab, %rdlab;
                ld.param.u64 %rdx, [p_x];
                cvta.to.global.u64 %rdx, %rdx;
                ld.param.u64 %rdg, [p_grad];
                cvta.to.global.u64 %rdg, %rdg;
                mul.lo.u32 %r1, %b, %nT;
                mul.wide.u32 %rd3, %r1, {{z}};
                ld.param.u64 %rdlse, [p_lse];
                cvta.to.global.u64 %rdlse, %rdlse;
                add.u64 %rdlse, %rdlse, %rd3;
                mul.lo.u32 %r1, %r1, %states;
                mul.wide.u32 %rd3, %r1, {{z}};
                ld.param.u64 %rdal, [p_alpha];
                cvta.to.global.u64 %rdal, %rdal;
                add.u64 %rdal, %rdal, %rd3;
                ld.param.u64 %rdbe, [p_beta];
                cvta.to.global.u64 %rdbe, %rdbe;
                add.u64 %rdbe, %rdbe, %rd3;

                // each frame's log-sum-exp
                mov.u32 %ft, 0;
            LSE:
                setp.ge.u32 %q1, %ft, %frames;
                @%q1 bra ALPHA;
                mad.lo.u32 %row, %ft, %nB, %b;
                mul.lo.u32 %row, %row, %nC;
                mul.wide.u32 %rd4, %row, {{z}};
                add.u64 %rd4, %rdx, %rd4;
                mov.b{{bits}} %acc, %ninf;
                mov.u32 %c, %tx;
            LMAX:
                setp.ge.u32 %q1, %c, %nC;
                @%q1 bra LMAX_END;
                mul.wide.u32 %rd5, %c, {{z}};
                add.u64 %rd5, %rd4, %rd5;
                ld.global.{{mem}} %x, [%rd5];
                setp.gt.{{o}} %q2, %x, %acc;
                selp.b{{bits}} %acc, %x, %acc, %q2;
                add.u32 %c, %c, 256;
                bra LMAX;
            LMAX_END:
            {{Reduce(t, "RMAX", max: true)}}
                mov.b{{bits}} %m, %acc;
                mov.b{{bits}} %acc, %zero;
                mov.u32 %c, %tx;
            LSUM:
                setp.ge.u32 %q1, %c, %nC;
                @%q1 bra LSUM_END;
                mul.wide.u32 %rd5, %c, {{z}};
                add.u64 %rd5, %rd4, %rd5;
                ld.global.{{mem}} %x, [%rd5];
                sub.rn.{{o}} %e, %x, %m;
                {{Exp("%e", "%e")}}
                add.rn.{{o}} %acc, %acc, %e;
                add.u32 %c, %c, 256;
                bra LSUM;
            LSUM_END:
            {{Reduce(t, "RSUM", max: false)}}
                {{Log("%lse", "%acc")}}
                add.rn.{{o}} %lse, %m, %lse;
                setp.ne.u32 %q1, %tx, 0;
                @%q1 bra LSE_NEXT;
                mul.wide.u32 %rd6, %ft, {{z}};
                add.u64 %rd6, %rdlse, %rd6;
                st.global.{{mem}} [%rd6], %lse;
            LSE_NEXT:
                add.u32 %ft, %ft, 1;
                bra LSE;

                // alpha at frame 0, then frame by frame
            ALPHA:
                bar.sync 0;
                mov.u32 %ft, 0;
                mov.u32 %st, %tx;
            A0:
                setp.ge.u32 %q1, %st, %S;
                @%q1 bra A0_END;
            {{Class}}
            {{LogY("%ft")}}
                setp.lt.u32 %q2, %st, 2;
                selp.b{{bits}} %res, %ly, %ninf, %q2;
            {{At("%rdal", "%ft", "%st", "%rd11")}}
                st.global.{{mem}} [%rd11], %res;
                add.u32 %st, %st, 256;
                bra A0;
            A0_END:
                bar.sync 0;
                mov.u32 %ft, 1;
            AT:
                setp.ge.u32 %q1, %ft, %frames;
                @%q1 bra LOSS;
                sub.u32 %fp, %ft, 1;
                mov.u32 %st, %tx;
            AS:
                setp.ge.u32 %q1, %st, %S;
                @%q1 bra AS_END;
            {{At("%rdal", "%fp", "%st", "%rd11")}}
                ld.global.{{mem}} %a0, [%rd11];
                mov.b{{bits}} %a1, %ninf;
                setp.ge.u32 %q2, %st, 1;
                @%q2 ld.global.{{mem}} %a1, [%rd11+-{{z}}];
            {{Skip("%st")}}
                mov.b{{bits}} %a2, %ninf;
                @%q4 ld.global.{{mem}} %a2, [%rd11+-{{2 * t.Size}}];
            {{lse3}}
            {{Class}}
            {{LogY("%ft")}}
                add.rn.{{o}} %res, %res, %ly;
            {{At("%rdal", "%ft", "%st", "%rd12")}}
                st.global.{{mem}} [%rd12], %res;
                add.u32 %st, %st, 256;
                bra AS;
            AS_END:
                bar.sync 0;
                add.u32 %ft, %ft, 1;
                bra AT;

                // the loss
            LOSS:
                sub.u32 %fl, %frames, 1;
                sub.u32 %r11, %S, 1;
            {{At("%rdal", "%fl", "%r11", "%rd11")}}
                ld.global.{{mem}} %a0, [%rd11];
                mov.b{{bits}} %a1, %ninf;
                setp.ge.u32 %q2, %S, 2;
                @%q2 ld.global.{{mem}} %a1, [%rd11+-{{z}}];
                mov.b{{bits}} %a2, %ninf;
            {{lse3}}
                mov.b{{bits}} %logp, %res;
                setp.ne.u32 %q1, %tx, 0;
                @%q1 bra LOSS_DONE;
                neg.{{o}} %res, %logp;
                mul.wide.u32 %rd13, %b, {{z}};
                ld.param.u64 %rd14, [p_loss];
                cvta.to.global.u64 %rd14, %rd14;
                add.u64 %rd14, %rd14, %rd13;
                st.global.{{mem}} [%rd14], %res;
            LOSS_DONE:
                ld.param.u32 %flag, [p_withgrad];
                setp.eq.u32 %q1, %flag, 0;
                @%q1 bra DONE;
                setp.eq.{{o}} %q1, %logp, %ninf;
                @%q1 bra ZERO;

                // beta at the last frame, then back
                mov.u32 %st, %tx;
            B0:
                setp.ge.u32 %q1, %st, %S;
                @%q1 bra B0_END;
            {{Class}}
            {{LogY("%fl")}}
                add.u32 %r11, %st, 2;
                setp.ge.u32 %q2, %r11, %S;
                selp.b{{bits}} %res, %ly, %ninf, %q2;
            {{At("%rdbe", "%fl", "%st", "%rd11")}}
                st.global.{{mem}} [%rd11], %res;
                add.u32 %st, %st, 256;
                bra B0;
            B0_END:
                bar.sync 0;
                mov.u32 %ft, %fl;
            BT:
                setp.eq.u32 %q1, %ft, 0;
                @%q1 bra GRAD;
                sub.u32 %fp, %ft, 1;
                mov.u32 %st, %tx;
            BS:
                setp.ge.u32 %q1, %st, %S;
                @%q1 bra BS_END;
            {{At("%rdbe", "%ft", "%st", "%rd11")}}
                ld.global.{{mem}} %a0, [%rd11];
                mov.b{{bits}} %a1, %ninf;
                add.u32 %r11, %st, 1;
                setp.lt.u32 %q2, %r11, %S;
                @%q2 ld.global.{{mem}} %a1, [%rd11+{{z}}];
                add.u32 %st2, %st, 2;
            {{Skip("%st2")}}
                setp.lt.and.u32 %q4, %st2, %S, %q4;
                mov.b{{bits}} %a2, %ninf;
                @%q4 ld.global.{{mem}} %a2, [%rd11+{{2 * t.Size}}];
            {{lse3}}
            {{Class}}
            {{LogY("%fp")}}
                add.rn.{{o}} %res, %res, %ly;
            {{At("%rdbe", "%fp", "%st", "%rd12")}}
                st.global.{{mem}} [%rd12], %res;
                add.u32 %st, %st, 256;
                bra BS;
            BS_END:
                bar.sync 0;
                mov.u32 %ft, %fp;
                bra BT;

                // the gradient: per frame exp(x - lse) (0 past the frames), less each state's share
            GRAD:
                mov.u32 %ft, 0;
            GT:
                setp.ge.u32 %q1, %ft, %nT;
                @%q1 bra DONE;
                setp.lt.u32 %q5, %ft, %frames;
                mad.lo.u32 %row, %ft, %nB, %b;
                mul.lo.u32 %row, %row, %nC;
                mul.wide.u32 %rd4, %row, {{z}};
                add.u64 %rd15, %rdx, %rd4;
                add.u64 %rd16, %rdg, %rd4;
                mul.wide.u32 %rd6, %ft, {{z}};
                add.u64 %rd6, %rdlse, %rd6;
                mov.b{{bits}} %lse, %zero;
                @%q5 ld.global.{{mem}} %lse, [%rd6];
                mov.u32 %c, %tx;
            GC:
                setp.ge.u32 %q1, %c, %nC;
                @%q1 bra GC_END;
                mul.wide.u32 %rd5, %c, {{z}};
                add.u64 %rd17, %rd15, %rd5;
                ld.global.{{mem}} %x, [%rd17];
                sub.rn.{{o}} %e, %x, %lse;
                {{Exp("%e", "%e")}}
                selp.b{{bits}} %e, %e, %zero, %q5;
                add.u64 %rd18, %rd16, %rd5;
                st.global.{{mem}} [%rd18], %e;
                add.u32 %c, %c, 256;
                bra GC;
            GC_END:
                bar.sync 0;
                @!%q5 bra G_NEXT;
                mov.u32 %st, %tx;
            GS:
                setp.ge.u32 %q1, %st, %S;
                @%q1 bra G_NEXT;
            {{At("%rdal", "%ft", "%st", "%rd11")}}
                ld.global.{{mem}} %a0, [%rd11];
            {{At("%rdbe", "%ft", "%st", "%rd12")}}
                ld.global.{{mem}} %a1, [%rd12];
            {{Class}}
            {{LogY("%ft")}}
                add.rn.{{o}} %e, %a0, %a1;
                sub.rn.{{o}} %e, %e, %ly;
                sub.rn.{{o}} %e, %e, %logp;
                {{Exp("%e", "%e")}}
                neg.{{o}} %e, %e;
                mul.wide.u32 %rd19, %cls, {{z}};
                add.u64 %rd19, %rd16, %rd19;
                {{red}} [%rd19], %e;
                add.u32 %st, %st, 256;
                bra GS;
            G_NEXT:
                add.u32 %ft, %ft, 1;
                bra GT;

                // no alignment: the gradient 0 on every frame
            ZERO:
                mov.u32 %ft, 0;
            ZT:
                setp.ge.u32 %q1, %ft, %nT;
                @%q1 bra DONE;
                mad.lo.u32 %row, %ft, %nB, %b;
                mul.lo.u32 %row, %row, %nC;
                mul.wide.u32 %rd4, %row, {{z}};
                add.u64 %rd16, %rdg, %rd4;
                mov.u32 %c, %tx;
            ZC:
                setp.ge.u32 %q1, %c, %nC;
                @%q1 bra ZC_END;
                mul.wide.u32 %rd5, %c, {{z}};
                add.u64 %rd18, %rd16, %rd5;
                st.global.{{mem}} [%rd18], %zero;
                add.u32 %c, %c, 256;
                bra ZC;
            ZC_END:
                add.u32 %ft, %ft, 1;
                bra ZT;
            DONE:
                ret;
            }

            """;
    }

    /// <summary>Straight-line code over registers of T (%v), a fresh register a result: the steps PTX has for T in T (add,
    /// sub, mul, neg, compare, select), the rest in f32 registers (%f): f16's div, min and max (PTX has no f16 div, nor f16
    /// min or max before sm_80), and every type's atan, by the polynomial atan(x) = base + x + x·z·P(z), z = x² after the
    /// reduction |x| > tan(3π/8): base π/2, x → −1/x; |x| > tan(π/8): base π/4, x → (x − 1)/(x + 1).</summary>
    private sealed class Scalar(PtxType t)
    {
        private readonly StringBuilder _code = new();
        private int _v, _f, _q;

        public string Code => _code.ToString();

        public string Declarations => $".reg .{t.Reg} %v<{Math.Max(1, _v)}>;\n    .reg .f32 %f<{Math.Max(1, _f)}>;\n    .reg .pred %q<{Math.Max(1, _q)}>;";

        private void Line(string line) => _code.Append("    ").Append(line).Append('\n');

        private string V() => $"%v{_v++}";

        private string F() => $"%f{_f++}";

        private string Q() => $"%q{_q++}";

        public string Const(double value)
        {
            string r = V();
            Line(t.MovLiteral(r, t.Literal(value)));
            return r;
        }

        public string Load(string address)
        {
            string r = V();
            Line($"ld.global.{t.Mem} {r}, [{address}];");
            return r;
        }

        public void Store(string address, string value) => Line($"st.global.{t.Mem} [{address}], {value};");

        private string Native(string op, string a, string b)
        {
            string r = V();
            Line($"{op}.{t.Op} {r}, {a}, {b};");
            return r;
        }

        public string Add(string a, string b) => Native("add.rn", a, b);

        public string Sub(string a, string b) => Native("sub.rn", a, b);

        public string Mul(string a, string b) => Native("mul.rn", a, b);

        public string Div(string a, string b) => t.Half ? In32("div.rn", a, b) : Native("div.rn", a, b);

        public string Min(string a, string b) => t.Half ? In32("min", a, b) : Native("min", a, b);

        public string Max(string a, string b) => t.Half ? In32("max", a, b) : Native("max", a, b);

        public string Neg(string a)
        {
            string r = V();
            Line($"neg.{t.Op} {r}, {a};");
            return r;
        }

        private string In32(string op, string a, string b)
        {
            string fa = F(), fb = F(), r = V();
            Line($"cvt.f32.f16 {fa}, {a};");
            Line($"cvt.f32.f16 {fb}, {b};");
            Line($"{op}.f32 {fa}, {fa}, {fb};");
            Line($"cvt.rn.f16.f32 {r}, {fa};");
            return r;
        }

        private string Compare(string how, string a, string b)
        {
            string q = Q();
            Line($"setp.{how}.{t.Op} {q}, {a}, {b};");
            return q;
        }

        public string Gt(string a, string b) => Compare("gt", a, b);

        public string Lt(string a, string b) => Compare("lt", a, b);

        public string Eq(string a, string b) => Compare("eq", a, b);

        public string And(string a, string b)
        {
            string q = Q();
            Line($"and.pred {q}, {a}, {b};");
            return q;
        }

        public string Select(string q, string a, string b)
        {
            string r = V();
            Line($"selp.b{t.Size * 8} {r}, {a}, {b}, {q};");
            return r;
        }

        public string Atan(string a)
        {
            string x = F(), ax = F(), one = F(), inv = F(), num = F(), den = F(), mid = F(), xr = F(), quarter = F(), halfPi = F(), b = F(), zz = F(), p = F(), y = F(), ny = F();
            string big = Q(), middle = Q(), negative = Q(), r = V();
            Line(t.Name switch { "f32" => $"mov.f32 {x}, {a};", "f64" => $"cvt.rn.f32.f64 {x}, {a};", _ => $"cvt.f32.f16 {x}, {a};" });
            Line($"abs.f32 {ax}, {x};");
            Line($"setp.gt.f32 {big}, {ax}, 0f401A827A;");                  // tan(3π/8)
            Line($"setp.gt.f32 {middle}, {ax}, 0f3ED413CD;");               // tan(π/8)
            Line($"mov.f32 {one}, 0f3F800000;");
            Line($"mov.f32 {inv}, 0fBF800000;");
            Line($"div.rn.f32 {inv}, {inv}, {ax};");
            Line($"sub.rn.f32 {num}, {ax}, {one};");
            Line($"add.rn.f32 {den}, {ax}, {one};");
            Line($"div.rn.f32 {mid}, {num}, {den};");
            Line($"selp.f32 {xr}, {mid}, {ax}, {middle};");
            Line($"selp.f32 {xr}, {inv}, {xr}, {big};");
            Line($"mov.f32 {quarter}, 0f3F490FDB;");                         // π/4
            Line($"mov.f32 {halfPi}, 0f3FC90FDB;");                          // π/2
            Line($"mov.f32 {b}, 0f00000000;");
            Line($"selp.f32 {b}, {quarter}, {b}, {middle};");
            Line($"selp.f32 {b}, {halfPi}, {b}, {big};");
            Line($"mul.rn.f32 {zz}, {xr}, {xr};");
            Line($"mov.f32 {p}, 0f3DA4F0D1;");                               // 8.05374449538e-2
            Line($"fma.rn.f32 {p}, {p}, {zz}, 0fBE0E1B85;");                 // − 1.38776856032e-1
            Line($"fma.rn.f32 {p}, {p}, {zz}, 0f3E4C925F;");                 // + 1.99777106478e-1
            Line($"fma.rn.f32 {p}, {p}, {zz}, 0fBEAAAA2A;");                 // − 3.33329491539e-1
            Line($"mul.rn.f32 {p}, {p}, {zz};");
            Line($"fma.rn.f32 {p}, {p}, {xr}, {xr};");
            Line($"add.rn.f32 {y}, {b}, {p};");
            Line($"setp.lt.f32 {negative}, {x}, 0f00000000;");
            Line($"neg.f32 {ny}, {y};");
            Line($"selp.f32 {y}, {ny}, {y}, {negative};");
            Line(t.Name switch { "f32" => $"mov.f32 {r}, {y};", "f64" => $"cvt.f64.f32 {r}, {y};", _ => $"cvt.rn.f16.f32 {r}, {y};" });
            return r;
        }
    }
}
