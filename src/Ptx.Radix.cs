// Scatter-add's ordering at any size (item 4f): the sources' order by target as a stable LSD radix sort of the
// (target, source) pairs, 8-bit digits; and the exclusive scan over many blocks that the offsets and the radix passes
// need. Every step is deterministic: the same order as the CPU's stable counting sort, no atomics on the order.
//   scan_tile   a block of 256 threads scans 1024 values (4 a thread, a warp scan by shuffles, the warps' sums through
//               shared memory) and writes its total; scan_add adds each block's scanned total back (the host recurses
//               on the totals).
//   radix_count per tile of 2048 sources (256 threads × 8, item-major: item t covers tile + t·256 + tid), the counts
//               of each digit (shared-memory counts) into table[digit][tile] — digit-major, so its exclusive scan is
//               each (digit, tile)'s first place.
//   radix_place the same tile, items in order: each warp's peers (lanes with the same digit) from eight ballots, a
//               lane's rank among its earlier peers by popc; the warps' per-digit counts through shared memory give
//               the earlier warps' share; a running count per digit carries the earlier items. Stable by construction.
namespace Qiyas;

internal static partial class Ptx
{
    public const int ScanBlock = 1024, RadixTile = 2048, RadixItems = 8;

    /// <summary>out[i] = Σ in[0..i) within each block of 1024 values; totals[block] = the block's sum (when p_totals ≠ 0).</summary>
    public static string ScanTile(Gpu gpu, out string entry)
    {
        entry = "scan_tile";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), sm = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mov.u32 {sm}, smem;");
        string pin = Param64(b, "p_in"), pout = Param64(b, "p_out"), ptot = Param64(b, "p_totals"), n = Param32(b, "p_n");
        string first = b.R();
        b.L($"mul.lo.u32 {first}, {bid}, {ScanBlock};");
        b.L($"mad.lo.u32 {first}, {tid}, 4, {first};");
        var v = new string[4];
        var valid = new string[4];
        var addr = new string[4];
        string s = b.R();
        b.L($"mov.u32 {s}, 0;");
        for (int j = 0; j < 4; j++)
        {
            string idx = b.R();
            v[j] = b.R(); valid[j] = b.P(); addr[j] = b.RD();
            b.L($"add.u32 {idx}, {first}, {j};");
            b.L($"setp.lt.u32 {valid[j]}, {idx}, {n};");
            b.L($"mul.wide.u32 {addr[j]}, {idx}, 4;");
            b.L($"mov.u32 {v[j]}, 0;");
            string a = b.RD();
            b.L($"add.u64 {a}, {pin}, {addr[j]};");
            b.L($"@{valid[j]} ld.global.u32 {v[j]}, [{a}];");
            b.L($"add.u32 {s}, {s}, {v[j]};");
        }

        // The warp's inclusive scan of the threads' sums.
        string x = b.R(), lane = b.R(), warp = b.R();
        b.L($"mov.u32 {x}, {s};");
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        foreach (int off in new[] { 1, 2, 4, 8, 16 })
        {
            string y = b.R(), py = b.P();
            b.L($"shfl.sync.up.b32 {y}|{py}, {x}, {off}, 0, 0xffffffff;");
            b.L($"@{py} add.u32 {x}, {x}, {y};");
        }

        string last = b.P(), wa = b.R();
        b.L($"setp.eq.u32 {last}, {lane}, 31;");
        b.L($"mad.lo.u32 {wa}, {warp}, 4, {sm};");
        b.L($"@{last} st.shared.u32 [{wa}], {x};");
        b.L("bar.sync 0;");
        string pre = b.R(), total = b.R();
        b.L($"mov.u32 {pre}, 0;");
        b.L($"mov.u32 {total}, 0;");
        for (int w = 0; w < 8; w++)
        {
            string ws = b.R(), pw = b.P();
            b.L($"ld.shared.u32 {ws}, [{sm}+{w * 4}];");
            b.L($"add.u32 {total}, {total}, {ws};");
            b.L($"setp.lt.u32 {pw}, {w}, {warp};");
            b.L($"@{pw} add.u32 {pre}, {pre}, {ws};");
        }

        string run = b.R();
        b.L($"add.u32 {run}, {pre}, {x};");
        b.L($"sub.u32 {run}, {run}, {s};");
        for (int j = 0; j < 4; j++)
        {
            string a = b.RD();
            b.L($"add.u64 {a}, {pout}, {addr[j]};");
            b.L($"@{valid[j]} st.global.u32 [{a}], {run};");
            b.L($"add.u32 {run}, {run}, {v[j]};");
        }

        string pt = b.P(), ta = b.RD();
        b.L($"setp.eq.u32 {pt}, {tid}, 0;");
        b.L($"setp.ne.and.u64 {pt}, {ptot}, 0, {pt};");
        b.L($"mul.wide.u32 {ta}, {bid}, 4;");
        b.L($"add.u64 {ta}, {ptot}, {ta};");
        b.L($"@{pt} st.global.u32 [{ta}], {total};");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_in, .param .u64 p_out, .param .u64 p_totals, .param .u32 p_n", 64);
    }

    /// <summary>out[i] += add[i / 1024] over n values (a thread a value).</summary>
    public static string ScanAdd(Gpu gpu, out string entry)
    {
        entry = "scan_add";
        var b = new PtxBuilder();
        string gid = GlobalIndex(b);
        string pout = Param64(b, "p_out"), padd = Param64(b, "p_add"), n = Param32(b, "p_n"), pEnd = b.P();
        b.L($"setp.ge.u32 {pEnd}, {gid}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string blk = b.R(), aa = b.RD(), oa = b.RD(), add = b.R(), val = b.R();
        b.L($"shr.u32 {blk}, {gid}, 10;");
        b.L($"mul.wide.u32 {aa}, {blk}, 4;");
        b.L($"add.u64 {aa}, {padd}, {aa};");
        b.L($"ld.global.u32 {add}, [{aa}];");
        b.L($"mul.wide.u32 {oa}, {gid}, 4;");
        b.L($"add.u64 {oa}, {pout}, {oa};");
        b.L($"ld.global.u32 {val}, [{oa}];");
        b.L($"add.u32 {val}, {val}, {add};");
        b.L($"st.global.u32 [{oa}], {val};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_out, .param .u64 p_add, .param .u32 p_n");
    }

    // The tile's item loop shared by both radix kernels: i = tile·2048 + t·256 + tid for t in 0..8.
    private static (string I, string Valid, string Key, string Digit) RadixItem(PtxBuilder b, string tileFirst, string t, string tid, string n, string keys, string shift)
    {
        string i = b.R(), valid = b.P(), key = b.R(), digit = b.R(), ka = b.RD();
        b.L($"mad.lo.u32 {i}, {t}, 256, {tileFirst};");
        b.L($"add.u32 {i}, {i}, {tid};");
        b.L($"setp.lt.u32 {valid}, {i}, {n};");
        b.L($"mov.u32 {key}, 0;");
        b.L($"mul.wide.u32 {ka}, {i}, 4;");
        b.L($"add.u64 {ka}, {keys}, {ka};");
        b.L($"@{valid} ld.global.u32 {key}, [{ka}];");
        b.L($"shr.u32 {digit}, {key}, {shift};");
        b.L($"and.b32 {digit}, {digit}, 255;");
        return (i, valid, key, digit);
    }

    /// <summary>table[digit · tiles + tile] = the count of digit ((key >> shift) &amp; 255) among the tile's 2048 keys.</summary>
    public static string RadixCount(Gpu gpu, out string entry)
    {
        entry = "radix_count";
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), sm = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mov.u32 {sm}, smem;");
        string keys = Param64(b, "p_keys"), table = Param64(b, "p_table"), n = Param32(b, "p_n"), tiles = Param32(b, "p_tiles"), shift = Param32(b, "p_shift");
        string mine = b.R(), zero = b.R(), tileFirst = b.R();
        b.L($"mad.lo.u32 {mine}, {tid}, 4, {sm};");
        b.L($"mov.u32 {zero}, 0;");
        b.L($"st.shared.u32 [{mine}], {zero};");
        b.L($"mul.lo.u32 {tileFirst}, {bid}, {RadixTile};");
        b.L("bar.sync 0;");
        string t = b.R(), pt = b.P();
        b.L($"mov.u32 {t}, 0;");
        b.Label("ITEM");
        var (_, valid, _, digit) = RadixItem(b, tileFirst, t, tid, n, keys, shift);
        string da = b.R(), old = b.R();
        b.L($"mad.lo.u32 {da}, {digit}, 4, {sm};");
        b.L($"@{valid} atom.shared.add.u32 {old}, [{da}], 1;");
        b.L($"add.u32 {t}, {t}, 1;");
        b.L($"setp.lt.u32 {pt}, {t}, {RadixItems};");
        b.L($"@{pt} bra ITEM;");
        b.L("bar.sync 0;");
        string count = b.R(), at = b.R(), ta = b.RD();
        b.L($"ld.shared.u32 {count}, [{mine}];");
        b.L($"mad.lo.u32 {at}, {tid}, {tiles}, {bid};");
        b.L($"mul.wide.u32 {ta}, {at}, 4;");
        b.L($"add.u64 {ta}, {table}, {ta};");
        b.L($"st.global.u32 [{ta}], {count};");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_keys, .param .u64 p_table, .param .u32 p_n, .param .u32 p_tiles, .param .u32 p_shift", 1024);
    }

    /// <summary>The tile's pairs placed stably by digit: keysOut[r] = key, valsOut[r] = value (the source's index i when
    /// p_vals is 0), r = base[digit · tiles + tile] + the earlier same-digit pairs of the tile.</summary>
    public static string RadixPlace(Gpu gpu, out string entry)
    {
        entry = "radix_place";
        const int Running = 0, Warps = 1024;                                  // shared: running[256], then warpCount[8][256]
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), sm = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mov.u32 {sm}, smem;");
        string keys = Param64(b, "p_keys"), vals = Param64(b, "p_vals"), keysOut = Param64(b, "p_keys_out"), valsOut = Param64(b, "p_vals_out"), bases = Param64(b, "p_base");
        string n = Param32(b, "p_n"), tiles = Param32(b, "p_tiles"), shift = Param32(b, "p_shift");
        string lane = b.R(), warp = b.R(), tileFirst = b.R(), mine = b.R();
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"mul.lo.u32 {tileFirst}, {bid}, {RadixTile};");
        b.L($"mad.lo.u32 {mine}, {tid}, 4, {sm};");
        {
            // Thread d: running[d] = the digit's first place for this tile; its column of the warps' counts cleared.
            string at = b.R(), ba = b.RD(), first = b.R(), zero = b.R();
            b.L($"mad.lo.u32 {at}, {tid}, {tiles}, {bid};");
            b.L($"mul.wide.u32 {ba}, {at}, 4;");
            b.L($"add.u64 {ba}, {bases}, {ba};");
            b.L($"ld.global.u32 {first}, [{ba}];");
            b.L($"st.shared.u32 [{mine}+{Running}], {first};");
            b.L($"mov.u32 {zero}, 0;");
            for (int w = 0; w < 8; w++) b.L($"st.shared.u32 [{mine}+{Warps + w * 1024}], {zero};");
        }

        b.L("bar.sync 0;");
        string t = b.R(), pt = b.P(), lt = b.R();
        b.L($"mov.u32 {lt}, %lanemask_lt;");
        b.L($"mov.u32 {t}, 0;");
        b.Label("ITEM");
        var (i, valid, key, digit) = RadixItem(b, tileFirst, t, tid, n, keys, shift);
        // The lanes holding a valid pair with this lane's digit.
        string peers = b.R();
        b.L($"vote.sync.ballot.b32 {peers}, {valid}, 0xffffffff;");
        for (int bit = 0; bit < 8; bit++)
        {
            string bitv = b.R(), pb = b.P(), bal = b.R(), nbal = b.R(), sel = b.R();
            b.L($"bfe.u32 {bitv}, {digit}, {bit}, 1;");
            b.L($"setp.ne.u32 {pb}, {bitv}, 0;");
            b.L($"vote.sync.ballot.b32 {bal}, {pb}, 0xffffffff;");
            b.L($"not.b32 {nbal}, {bal};");
            b.L($"selp.b32 {sel}, {bal}, {nbal}, {pb};");
            b.L($"and.b32 {peers}, {peers}, {sel};");
        }

        string below = b.R(), local = b.R(), size = b.R(), lead = b.P(), wc = b.R();
        b.L($"and.b32 {below}, {peers}, {lt};");
        b.L($"popc.b32 {local}, {below};");
        b.L($"popc.b32 {size}, {peers};");
        b.L($"setp.eq.and.u32 {lead}, {local}, 0, {valid};");
        b.L($"mad.lo.u32 {wc}, {warp}, 1024, {sm};");
        b.L($"mad.lo.u32 {wc}, {digit}, 4, {wc};");
        b.L($"@{lead} st.shared.u32 [{wc}+{Warps}], {size};");
        b.L("bar.sync 0;");
        {
            // rank = running[digit] + the earlier warps' counts of the digit + the earlier peers in this warp.
            string rank = b.R(), da = b.R();
            b.L($"mad.lo.u32 {da}, {digit}, 4, {sm};");
            b.L($"ld.shared.u32 {rank}, [{da}+{Running}];");
            for (int w = 0; w < 7; w++)
            {
                string c = b.R(), pw = b.P();
                b.L($"ld.shared.u32 {c}, [{da}+{Warps + w * 1024}];");
                b.L($"setp.lt.u32 {pw}, {w}, {warp};");
                b.L($"@{pw} add.u32 {rank}, {rank}, {c};");
            }

            b.L($"add.u32 {rank}, {rank}, {local};");
            string value = b.R(), pv = b.P(), va = b.RD(), ka = b.RD(), oa = b.RD();
            b.L($"mov.u32 {value}, {i};");
            b.L($"setp.ne.and.u64 {pv}, {vals}, 0, {valid};");
            b.L($"mul.wide.u32 {va}, {i}, 4;");
            b.L($"add.u64 {va}, {vals}, {va};");
            b.L($"@{pv} ld.global.u32 {value}, [{va}];");
            b.L($"mul.wide.u32 {oa}, {rank}, 4;");
            b.L($"add.u64 {ka}, {keysOut}, {oa};");
            b.L($"@{valid} st.global.u32 [{ka}], {key};");
            b.L($"add.u64 {oa}, {valsOut}, {oa};");
            b.L($"@{valid} st.global.u32 [{oa}], {value};");
        }

        b.L("bar.sync 0;");
        {
            // Thread d carries the item's count of digit d into running[d] and clears its column.
            string sum = b.R(), zero = b.R(), r = b.R();
            b.L($"mov.u32 {sum}, 0;");
            b.L($"mov.u32 {zero}, 0;");
            for (int w = 0; w < 8; w++)
            {
                string c = b.R();
                b.L($"ld.shared.u32 {c}, [{mine}+{Warps + w * 1024}];");
                b.L($"add.u32 {sum}, {sum}, {c};");
                b.L($"st.shared.u32 [{mine}+{Warps + w * 1024}], {zero};");
            }

            b.L($"ld.shared.u32 {r}, [{mine}+{Running}];");
            b.L($"add.u32 {r}, {r}, {sum};");
            b.L($"st.shared.u32 [{mine}+{Running}], {r};");
        }

        b.L("bar.sync 0;");
        b.L($"add.u32 {t}, {t}, 1;");
        b.L($"setp.lt.u32 {pt}, {t}, {RadixItems};");
        b.L($"@{pt} bra ITEM;");
        b.L("ret;");
        return b.Finish(Header(gpu), entry, ".param .u64 p_keys, .param .u64 p_vals, .param .u64 p_keys_out, .param .u64 p_vals_out, .param .u64 p_base, .param .u32 p_n, .param .u32 p_tiles, .param .u32 p_shift", Warps + 8 * 1024);
    }
}
