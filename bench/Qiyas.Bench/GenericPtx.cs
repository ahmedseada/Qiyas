// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The generic GPU side: one PTX generator per operation, written once and specialized by T at run time (the GPU's
// counterpart of a C# generic the JIT specializes per T). T's PTX names come from its description (PtxType.Of<T>), never
// from a per-type copy of the kernel. Where PTX has no instruction for T (no f64 exp, no f16 div/sqrt/rcp), the generator
// says so in Gaps: those steps run in f32 registers for that T.
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
}
