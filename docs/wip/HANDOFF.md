# Handoff: items 4–6 of the coverage stretch

Written 2026-11-06 at the end of item 3 and kept current through item 4, for a session with none of this one's
context. Read `COVERAGE.md` ("State" first) and `README.md` before anything (the repository has no CLAUDE.md; the
standing rules are the next sentence). Standing rules: Idrak is an application of Qiyas, never
its guide (nothing from `Lib/` in repository files; `sh guard.sh` zero hits); card-agnostic; no fixed type; every
operation written once over `<TN, TLane, TL>` on both devices with a float64 reference, an error column and a gate in
`src/Checks.cs`; nothing removed, renamed or weakened; the tree builds and the gates are green at every stop; no commits
or pushes (the coordinator commits on the owner's go); report after each item with (a) what changed, (b) the exact
commands and pass lines, (c) the grid delta, (d) `git status --short`, (e) README/COVERAGE updated, guard zero hits,
Release build 0 errors — then stop editing and wait for "go".

## Tools in docs/wip and the commands that matter

| what | how |
|---|---|
| Build | `dotnet build -c Release` (from `D:\Projects\Qiyas`); `sh guard.sh` before every build. A build fails while a bench process holds the dll. |
| Full run (both devices' gates, then the core rows of every type) | `dotnet bin/Release/net10.0/Qiyas.dll cuda,cpu --report` (~25 min; writes `results-qiyas-cuda-cpu.md`). `cuda` alone also runs the CPU gates first. |
| Regression gate | per item (from 4c on): the gates plus only the rows of the family the item changed, qiyas only (e.g. `dotnet bin/Release/net10.0/Qiyas.dll cuda,cpu --subjects qiyas --ops recurrent` for the GRU row; every gate runs at startup), then `python docs/wip/compare.py docs/wip/baseline.md results-qiyas-cuda-cpu.md` — compare.py finds each subject's columns by the header's names, so a qiyas-only file compares with the full baseline; the full `cuda,cpu --report` run with every column only at the end of each whole item (after item 4, after item 5, after item 6; never after a piece inside an item): the README Results tables are regenerated from it (`docs/wip/readme_results.py`) and it becomes `docs/wip/baseline.md`. compare.py prints every cell with base/new ms, ratio, lanes, errors; flags `SLOWER` (> 5 %) and `ERR UP`; "new row" for rows the baseline lacks. `baseline.md` is item 4's `cuda,cpu --report --ops all` run of
19:11 (4216 rows, every family, every column; stopped during the CPU's q4_k rows, so the CPU's q6_k and bool rows are
missing; converted from the console lines by the scratchpad script `log2results.py`: no shape column except where the
4a baseline had the row). Until item 4 it was the run of item 4a (128 core rows). An earlier commit (c68ef15) held a matmul re-timing run there by mistake; the 12:38 run of item 3 survives only as the README Results tables, which `python docs/wip/compare_readme.py README.md results-qiyas-cuda-cpu.md <an older full run for the error column>` compares against cell by cell. Some CPU cells of the 4a run caught this laptop's slow level; re-time before calling a cell slower. Re-time a flagged cell several times before calling it swing: this laptop's sub-2 ms cells move ±20–40 % run to run; an A/B against a build of the baseline commit (`git worktree add <dir> c73a251`, build it, run both dlls back to back) separates code from machine state. |
| README Results tables | `python docs/wip/readme_results.py results-qiyas-cuda-cpu.md > tables.md` — the two tables (lib / number / qiyas / ×). The script sets stdout to UTF-8; redirect to a file and splice (the Bash console's code page mangles — × ³ otherwise). |
| Gate-only quick runs | `Qiyas.dll cpu --types int8 --ops add` (every CPU gate, one row); `Qiyas.dll cuda --types float32 --ops add` (every gate on both devices). `--types` only limits the bench rows; the gates always cover every type. |
| Stress | `--repeat N`: repeats the matmul rows N times and raises `Checks.Stress` (default 20) to N: the block-format transpose is repeated `Stress` times in Group2, the team gate runs `Stress × 500` jobs, the device products gate runs `Stress / 20` passes under every tensor kind. Used: `cpu --types q6_k,q4_k,q4_0,q8_0 --ops reduce --repeat 100` (the transpose race), `cpu --types int4 --ops matmul --repeat 30` ×5 (the team race), `cuda --types uint16 --ops matmul --repeat 600` (30 passes of the device products gate). |
| Debug prints | `QIYAS_MAP_DEBUG=1` prints the first failing values of a `Close` check with their neighbours; `QIYAS_PTX_DUMP` dumps generated PTX. |
| Options | `--ops map|reduce|rows|layout|products|attention|optimizers|conv|recurrent|routing|all` or single names (`row:rope`, `layout:im2col`, `attention_decode`); `--shapes odd` (primes below the regular sizes); `--subjects`, `--types`, `--baseline name`. |

One bench at a time (they share the GPU and the CPU team; timings are meaningless otherwise). Run long ones in the
background and wait for the notification. Patch scripts with quotes inside heredocs break the Bash tool: write the
script to a file and run `python script.py .` from the repository root.

## Pitfalls already known

- Block formats (Q8_0, Q4_0, Q4_K, Q6_K): the lane is float32; `Blocks.Store` re-encodes a partial block keeping its
  other values, so the block's scale then covers them — an output buffer must be one whose blocks the output covers whole
  or whose other values belong with it (never a buffer holding stale larger values: they quantize the new values to 0).
  Gates use fresh buffers per check. CPU tasks must never share a block: chunk rows with `Kernels.RowChunk<TN>` (whole
  storage units) and value chunks in multiples of `TN.Granularity`. The device writes block types through float32
  storage (`GpuOps.Lanes`/`Back`) and encodes by blocks (`Ptx.LanesToBlock`, all four formats).
- bool is a 1-bit packed type held to 0/1; `div` is n/a; its row tasks need `RowChunk` too (8 values a byte).
- Device row kernels: rows longer than 16 words a thread are segmented at 256 threads × 8 words (1024 × 16 f64
  overflows a block's registers: "too many resources requested for launch"); the value-wise variant (rows not ending on
  a word) is segmented the same (`RowCtx` in `src/Ptx.Rows.cs`, `RowKernelShape` in `src/GpuOps.Rows.cs`). Any new
  row kernel must go through `RowCtx.Reduce`/`Store` so segments work.
- Device products: a block format's values are exact in f32, not in the f16 tensor lane (`ProductLane` returns "f32"
  for blocks). Tensor-kind choices are measured timings (`FasterKind`) and can flip between runs; the products gate
  forces every kind (`GpuOps.FourBit`, `GpuOps.K32`) so all are checked regardless.
- The thread team (`src/Team.cs`): a job is closed when its last chunk completes; keep it so (the int4 product fault).
- `References.MatMul` (Program.cs) is double-double: an exact lane reads 0; a float64 lane shows ~1e-13 at 2048³.
- Measured choices (one path or another, per shape) go through `Measure.Faster` (`src/Measure.cs`) on both devices,
  and the bodies of both paths carry AggressiveOptimization: a single timed call each, or a body still in the runtime's
  first tier, chose the 20× slower path (the block product at 4d).
- The open intermittent (COVERAGE.md State, item 4 there): `matmul_bwd dB device` unsigned types wrong by 138 steps,
  seen once, not reproduced in 150 gate passes. If it shows again, `QIYAS_MAP_DEBUG=1` on the run captures the values.

## Bench runs (the owner's rules, from item 4b on)

Do not overuse bench runs: run one only when it is needed, and as small as the question allows.

1. Per item: the gates (they all run at startup) plus only the rows of the family the item changed, qiyas only, e.g.
   `--subjects qiyas --ops recurrent` for the GRU row.
2. A/B or re-timing only for a slower cell whose code the item changed: at most once per cell, both orders in a single
   comparison. A slower cell in code the item did not touch is machine swing by definition: list it, no A/B.
3. Stress (`--repeat`) only when the item touches scheduling, chunking, shared buffers or a race-prone path.
4. One full run with every column at the end of each whole item: once after item 4 (all seven todos gone, no todo
   left in COVERAGE.md), once after item 5 (the arena), once after item 6 (the known-slow list). Each regenerates the
   README Results tables and becomes `docs/wip/baseline.md`. Never after a piece inside an item (4b, the GRU row, the
   tiled product, ...): a piece gets only the gates plus its own family's rows (rule 1).
5. No bench to check a doc or a script change.
6. The full runs at the end of items 5 and 6 (the coordinator, after item 4's `--ops all` run took 92 minutes: the
   length came from the map family's formulas across all 21 types, not the number column). They time:
   - the families the item changed, at every type;
   - every other family at a fixed representative set of types: float32, float16, float64, int8, int4, q4_k;
   - the map family with a fixed handful of formulas: add, mul, exp, gelu, erf and one backward (e.g. gelu_bwd).
   Columns: qiyas and lib everywhere they have rows, number on the core ops only (it implements nothing else). The CLI
   takes one `--types` and one `--ops` per invocation, so this is several invocations, their results merged into the
   one baseline. Estimate the length from the baseline's per-row times (plus a per-row overhead for inputs, references
   and the repeated timing) and send the estimate to the coordinator before launching; the ceiling is 40 minutes, aim
   well under. Every bench runs on the owner's PC: anything over 5 minutes is estimated first, over 15 minutes asked
   first; gate-only runs (`cuda --subjects qiyas --types float32 --ops add`, ~30 s, every gate) are preferred, and no
   polling waiters or monitors.

## Item 4 progress (update as it moves)

Item 4 = every remaining **todo** in COVERAGE.md gone: 4a fused attention rows, 4b ctc, then 4c the GRU sequence bench
row, 4d the block-scaled tiled product, 4e the transposition fused into the product's packing, 4f the radix step of
scatter-add, 4g the fused direct convolution. Each may be reported as its own boundary.

- 4a: done, committed as c35fd15 (the full run of 4a is docs/wip/baseline.md). What landed:
  - device kernel `Ptx.AttentionRows` (`src/Ptx.AttentionRows.cs`): a block of 256 threads per (head, Br query rows),
    Br = Bc = 32 in f32 and 16 in f64 (`Ptx.AttentionTile`), dCap ∈ {32, 64, 128}; K and V tiles through one shared
    buffer (`AttentionRowsShared`), 16-byte shared loads, group max/sum by `shfl.sync.bfly`, P through shared memory;
    per-block key ranges [lo, hi) computed on the host (`p_range`), so causal and span tiles outside are skipped.
    FMA lane: the tensor lanes inside it are the known-slow step.
  - CPU `Kernels.AttentionRows` (`src/AttentionRows.cs`): tasks of 16 query rows, key tiles of 64 laid transposed,
    row pairs in register accumulators.
  - `Attention<TN>` on both devices is now the measured choice between `AttentionRows<TN>` and `AttentionComposed<TN>`
    (the old body, renamed) per shape and type: both warmed, then alternating rounds, the minimum of each (a first run
    while the GPU raises its clocks timed the fused rows at 23 ms against 3.8 warm and chose wrong before that).
    Decisions: `GpuOps.Note` / `Kernels.Note` (new, static); `--report` prints them again at the end of the run
    (`Kernels.Measured`, `GpuOps.Measured`).
  - gate: `Checks.Attentions` runs every case composed and fused, plus d 17 / 64 / 128, causal, spans with starts, a
    block whose rows see no key; bench rows `attention_fused`, `attention_fused_causal` (`IBench.AttentionFused`).
  - also in 4a: the device products intermittent settled (`GpuOps.KPadded`, gate `Checks.TensorConfigs`; COVERAGE
    "Settled").
  - measured (16 heads, 2048², d 128): device f32 3.79 ms (causal 2.12) from 6.1 by the composition; f16 3.82, fp8
    4.72, q8_0 3.95; f64 non-causal keeps the composition (59 vs 61), causal fused 31.7 vs 59. CPU f32 60–80 ms (causal
    33–38) from 100–148.
- 4b: done, committed as 168059c. What landed:
  - `src/Ctc.cs`: `CtcArgs(T, B, C, Blank, InputLengths, LabelLengths, Labels)` with `Validate`, `States` (per sequence,
    rows of MaxStates + 2: the class of each extended state; flags bit 0 = α may skip from s − 2, bit 1 = the class's
    first state) and the float64 `Reference(x)` → (loss, grad); logits in, log_softmax inside.
  - CPU `Kernels.Ctc<TN>`: a task per sequence (`CtcBody<TS, TLane, TL>`), the states a vector at a time with the
    lane's exp/log; float32/float64 rows in place, narrower floats loaded a row at a time in the task (`TS.Load`), block
    formats through float32 storage (`ToLane`/`FromLane`). Tried and dropped: the rows' lse and softmax as a pass over
    all T × B rows (slower: the emissions became single-value loads).
  - device `GpuOps.Ctc<TN>` (`src/GpuOps.Ctc.cs`) on lane storage: `Ptx.CtcRows` (a warp per row: lse, and for the
    gradient the softmax row, 0 past the frames), then `Ptx.Ctc` (a block per sequence, 256 threads over the states,
    α in two shared buffers with 2 front pads, α of every frame in global scratch, thread 0 takes log P; backward: β in
    the same buffers with pads after S, per frame the states' terms into shared memory and each class's first state
    writes y − Σ; no atomics). Shared bytes `CtcShared(lane, cap)`, cap = MaxStates rounded to 64; refuses past 48 KB.
  - gate `Checks.Ctcs` (both devices, every float type): the reference's gradient by central differences; T11 B4 C7
    (short sequences, an empty label, an adjacent and a non-adjacent repeat), T13 B3 C5 with blank 4, T5 B2 C4 with a
    sequence no alignment fits (types with an infinity: loss +∞, gradient 0); per-frame gradient sums to 0 (f32/f64).
    A mutation (the skip rule off) fails it on both devices by millions of steps.
  - contract `IBenchSubject.Ctc(device, type, args, logits, loss, grad)`; bench rows `ctc`, `ctc_bwd` in Group9
    (`--ops routing`).
  - measured (T 256, B 32, C 1024, 64 labels): device f32 0.54 / 1.18 ms (fwd / bwd), f64 2.35 / 5.25; CPU f32
    1.8 / 3.6–3.8, f16 2.4 / 4.4, f64 4.0 / 8.0. The device recursion is latency-bound (COVERAGE "Known slow").
- 4c: done, reported as its own boundary (uncommitted until the coordinator's go). What landed:
  - `RecurrentCase.GruSequenceReference` (the step's `Gates` and `CellArgs.GruCell` looped in float64).
  - `Kernels.GruSequence<TN>` and `GpuOps.GruSequence<TN>`: a narrower type or a block format widened once, h kept in
    the float lane across the steps, narrowed once (the LSTM sequence's wrapper; on the device block formats too,
    where the GRU sequence refused them before; on the CPU narrow types rounded h to the type every step before).
  - contract `IBenchSubject.GruSequence(...)`, `QiyasSubject.GruSequenceVisitor`; gate in `Checks.Recurrents` (3 steps
    of 5 × 7 → 11, every float type incl. blocks); bench row `gru_seq` in Group8 (T16 B256 D512 H512).
  - measured: device f32 2.55 ms, f16 2.77, f64 28.5, q8_0 2.70; CPU f32 71, f16 62, f64 126, q8_0 55.
  - no GRU sequence backward (not asked for); COVERAGE says so. Committed as bb5ab2c.
- 4d: done, reported as its own boundary (uncommitted until the coordinator's go). What landed (design in "4d as built"):
  - `src/MatMulBlocks.cs`: CPU `Kernels.MatMulBlocks` (A and B unpacked once to int8 rows and per-4-byte scales, tasks
    of 4 rows, 4 × 2 tiles of float vectors, `S8Dot.Fma` per vector), `MatMulViewBlocks` (C encoded by blocks on the
    team: `BlocksStore` over `NarrowBody`; the serial `CopyLanes` took 15 of 18 ms); device `Ptx.MatMulBlocks`
    (128 threads per 64 × 64, m16n8k32 s8 per block per 16 × 8 tile, 4 k-blocks staged a time: element i of the tile's
    64 × 4·halves 16-bit loads to thread i % 128, so a row's contiguous blocks are read by consecutive lanes;
    the per-halfword staging before that took 3.2 / 1.4 ms) and `GpuOps.MatMulBlocks<TN>` / `MatMulViewBlocks<TN>`.
  - `MatMulView` on both devices: `BlockView` (A·Bᵀ, batch 1, Q8_0/Q4_0, k % 32 = 0, the lane available) then
    `UseBlocks` (measured once per shape and format; override `BlockProducts` for the gate); the old body is
    `MatMulViewRoute`.
  - fixed on the way: the device gemv's block path (`Ptx.GemvBlocks`) loaded 32-bit words from 2-byte-aligned blocks
    (misaligned address on every other block); now two 16-bit halves. Its first gate at a whole number of blocks.
  - gate `Checks.BlockProducts` (from `Products`): q8_0, q4_0 A·Bᵀ at 37 × 29 × 64 and × 320 forced, gemv at 37 × 64,
    both devices; a mutation of the device scales fails it by 6–9 steps.
  - contract `IBenchSubject.MatMulBlocks(device, type, m, n, k, a, b, c)`; bench row `matmul_blocks` in Group4b.
  - measured: device 2048³ q8_0 0.54 ms, q4_0 0.37 (matmul_nt by the decoded route 3.8 / 0.89); CPU 1024³ q8_0 2.8,
    q4_0 2.7–3.0 (decoded 19 / 17). 4h (Idrak in every family's lib column) was cancelled by the owner.
  - fixed before the go (the coordinator's run chose the decoded route on the CPU for q8_0: block-scaled 66 ms against
    a steady 2.8): the cause was the runtime's first compilation tier, not memory (no collections in those calls). Q8_0's
    first block calls ran 68, 68, 54, then 10, 3.4 ms as the optimized body arrived; Q4_0, measured next on the same
    body, ran 2.8 from the first round; the decoded route was already optimized by earlier work. Fixed both ways:
    the cost (`MatMulBlocksBody.Run`, `UnpackRowsBody.Run` compiled at the final tier from the first call,
    AggressiveOptimization, as the hot kernels already are) and the measurement (`src/Measure.cs`, `Measure.Faster`:
    rounds of ≥ 2 ms of calls, alternating, at least 3, on until neither minimum improves by 2 % and the runtime
    compiles nothing for 3 rounds in a row, at most 24). All four measured choices use it: block product and attention,
    CPU and device (CPU attention at the gates' shapes flipped between fused rows and composition run to run before).
    Proof: `cuda,cpu --report --subjects qiyas --types q8_0,q4_0 --ops matmul_nt,matmul_blocks` 3 times, block-scaled
    chosen for both formats on both devices every time, matmul_nt on the lane of matmul_blocks. One A/B (both orders)
    of the attribute: the steady CPU block row 3.5–3.65 ms with it, 3.15–3.18 without; profile-guided optimization off
    left the build without it unchanged, so the gap is within this laptop's swing or a real ~10 % — re-check at item 4's
    full run.
- 4d committed as a784145 (the coordinator: keep AggressiveOptimization; the full run settles the ~10 %).
- 2 steps left in item 4 after 4g: the GRU sequence backward (below), then item 4's full run.
- GRU sequence backward: done, reported as its own boundary (uncommitted until the coordinator's go). See its entry
  below. 1 step left in item 4: the full run.
- GRU sequence backward committed as 1606800 (verified by the coordinator, pushed, main fast-forwarded).
- Item 4's full run (the coordinator: `cuda,cpu --report --ops all`, every column, the baseline for every family). The
  number subject implements the four core ops only (src/Bench/NumberSubject.cs) and the lib subject add, matmul,
  softmax, rms_norm, map, row, reduce, transpose, so `--ops all` gives every family qiyas (and lib where it has the op)
  and the number column on the core ops only, as the coordinator's fallback would. Two faults the first launch hit:
  - the device row kernels: `cross_entropy` float16 (log_softmax over 4096 × 32768: 1024 threads × 16 words) failed to
    launch ("too many resources requested": the row in registers needs more of them than a block of 1024 has on this
    card; float64 at 16384 the same). Fixed card-agnostically: `Gpu.MaxThreads(function)` (cuFuncGetAttribute,
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) is asked once the kernel is compiled; past it `RowOp` and the norm
    statistics (GpuOps.Rows.cs) take the segmented shape (256 × 8, as rows past 16 words do), `Softmax` and `RmsNorm`
    (GpuOps.cs) their looping kernel. Gate `Checks.Rows` gains 2 × 16384 and 2 × 32768 for the row family and softmax,
    rms_norm. Mutations: the fallback off in the row ops → the gate crashes at float64 log_softmax 2 × 16384; off in
    softmax / rms_norm only → the gate crashes too.
  - the `cross_entropy` bench row on the device read err 16 (the CPU 2e-6): it prepared the pick and neg runs before
    log_softmax ran, and a device subject uploads its inputs when a run is prepared, so pick read stale logp. Fixed in
    Program.cs: pick and neg prepared inside the timed run after the step before collected (the device time is now the
    host round trips: a Known slow row and item 6 below).
  - the run: 19:11:48 to the coordinator's stop at 20:44 (92 minutes; CUDA about 45, the CPU about 47 up to q4_k);
    0 crash, 0 failed check (the gates and the 4216 rows). The coordinator stopped it for its length and closed item
    4 with it: the rows became `docs/wip/baseline.md` (partial on the CPU: q6_k and bool missing), the README Results
    regenerated from them (the CPU q6_k and bool core rows kept from the earlier tables, marked "earlier run") with
    the per-family table `readme_results.py` now prints.
  - compare.py against the 4a baseline: 122 cells compared, 4096 new rows, 0 error grown, 13 slower: in code item 4
    did not change (swing, listed only) CUDA add float8_e5m2 1.35, int16 1.79, int8 1.52, int2 1.06, uint8 1.83, CPU
    add int8 1.09, rms_norm float16 1.49, q8_0 1.19; in code item 4 changed (not re-timed, the coordinator's order:
    item 6) CPU matmul float64 1.16, float16 1.38, float8_e5m2 1.41 (4e's kernel), CUDA matmul int4 1.15 (4e), CUDA
    softmax float16 1.12 (the launch-limit query, a driver call per launch: now asked once per kernel, cached in
    `Gpu.MaxThreads`).
  - the README's error sentence was wrong across the families: ten CUDA rows have a larger error than the lib's
    (float32 sin, cos, gelu, silu, add_rms_norm up to 1.4×; float16 softmax 9.54e-7 vs 9.53e-7; sum:whole float32
    5.98e-4 vs 1.95e-4, float8_e5m2 3.66e-4 vs 6.1e-5; sum:rows float16, bfloat16 up to 1.3×). The README now says so;
    item 6 below.
- Item 4: done (closed by the coordinator with the stopped run). Next: item 5, the arena, on the coordinator's go.
- Item 4 committed as c936dca.

## Item 5 progress

- Done, reported (uncommitted until the coordinator's go). What landed:
  - `src/Arena.cs` (Gpu is partial now): `Gpu.Scope()` / `GpuOps.Scope()`; inside a scope `Rent` bumps in one region and
    `Return` releases from the top (a buffer returned below the top waits until everything above it is); a rent past the
    region comes from the pool for that call, and the outermost scope's end grows the region (synchronize, free, allocate,
    rounded to 1 MB) to the call's peak computed as if every rent had been bumped, so the next call fits. Counts:
    `Gpu.Counts` (arena rents, pool rents, driver allocations, overflows, leaks, strays), `Gpu.LastScope` (high-water
    mark, peak alive). The pool (`PoolRent` / `PoolReturn`) is the old Rent / Return, unchanged, for rents outside scopes.
  - the bench (`Program.Scoped`): every timed Qiyas call on the device inside one scope; the buffers prepared outside
    stay the pool's.
  - `src/Workspace.cs`: `Workspace<T>.Rent(slot, n)` / `Return` — per-thread, per-type arrays grown with a quarter's
    headroom; a slot busy on the thread falls back to ArrayPool. `Kernels.MatMul`'s packed, al, bl, cl and
    `ProductBody`'s panel and kRows use it.
  - `--report`: pool pair 653 ns, arena pair 14 ns; a transposed float16 256³ product's second scoped call 3 arena
    rents, 0 pool rents, 0 driver allocations (region 2 MB); CPU: a float16 product's second call 34 workspace rents, 0
    allocations, 0 pool fallbacks.
  - gate `Checks.Arenas` (both devices, float16 / float32): device — a transposed product, attention, an LSTM step,
    once from the pool then three scoped calls: equal bit for bit, no leak or stray, and from the second scoped call no
    pool rent, driver allocation or overflow and high-water mark == peak alive; CPU — the product and the LSTM step
    reach a call with no workspace allocation or pool fallback within 8 calls, results unchanged. Mutations: the
    workspace never freeing its slot (CPU: 4 failures), the bump never moving (device: results differ), the top never
    released (device: lstm_step high-water 668160 for 337920 alive).
  - timing (the coordinator's one run: `cuda,cpu --subjects qiyas --ops add,matmul,softmax,rms_norm,products`, 217 s)
    against item 4's baseline: geomean new/base CUDA 0.997 (176 cells: 15 faster, 21 slower), CPU 1.009 (158: 41
    faster, 42 slower); untouched cells moved as much (CPU add float16 1.41, CUDA add uint64 1.92): swing. One device
    cluster is consistent: the products with Aᵀ (tn / tt) float16 1.11–1.14, bfloat16 1.10–1.14, int32 / uint32 1.17–1.30,
    int64 / uint64 tt 1.59, batched int4 / int2 1.39–1.45, while nn / nt did not move: a Known slow row and item 6 below.
    The sub-ms device cells expected to gain a few µs read softmax float16 0.084 → 0.075, bfloat16 0.082 → 0.077,
    rms_norm bfloat16 0.081 → 0.075.
- 4g: done, committed as d9939ff. What landed:
  - CPU: `Kernels.Operands.ImageB` (a ConvArgs): `PackBody` builds each row of the columns from the image
    (`ImageRow`: row p = (c, kh, kw), an image row read in place or loaded once, stride 1 by a contiguous copy with the
    padding cleared, otherwise gathered) — the micro-kernel, A and C untouched. `Kernels.Conv` takes it where
    `ConvInPlace` (rows on whole bytes) holds, bounds = the image's range with 0; `ConvComposed` is the old body;
    `Kernels.FusedConv` (false: the composition) for the gate.
  - device: `Ptx.PackTransposed(…, conv: true)` (entry packconv_…: the load's index replaced by the image map, 0 off the
    image) and `Ptx.Gemm(…, conv: true)` (gemmconv…: the B tile gathered from the image in the lane's storage).
    `GpuOps.Conv`: the tensor / dot lanes → `ConvPacked` (the composition's steps: padded M, N, K as MatMulPadded, the
    weights by `PackRows` into a cleared padded buffer, the image by the conv packer, `MmaFunction` / dp4a, `Back`, the
    strided copy out — the same values, so the same bits); the plain lanes → `ConvPlain` (gemmconv at tile 128, 64 for
    wide lanes) or `ConvComposed`, measured once per shape and type by `Measure.Faster` (`GpuOps.FusedConv` forces).
    `GpuOps.KindOf` factored out of `MatMul` (the kind of a packed lane), unchanged behaviour.
  - the coordinator's question (the plain lanes cannot silently keep the columns): they have the fused path
    (gemmconv) and a measured choice against the composition. At the bench shape (64 × 128 × 128 by 128 of 3 × 3):
    float32 fused 0.137 ms vs 0.352, q8_0 0.213 vs 1.30, q4_k 0.235 vs 1.61 → fused; float64 fused 3.40 vs 2.55 by the
    composition through the sliced f64x route → the composition (two runs). In the full products run the sliced
    product measured 3.56 instead of 2.15 (`SlicedFaster`: one warm call, three timed), the choice took fused and the
    row read 3.41: a Known slow row and item 6 below.
  - gate (`Checks.Convs`): the fused forward forced on, within rounding, and equal bit for bit to the composition
    through the same kernel (CPU: the composition; device tensor / dot: the composition; device plain float lanes:
    `GpuOps.ConvPlainComposed` = im2col in lane storage + the same plain kernel; integer lanes exact either way; block
    formats rounding only — the composition re-encodes the columns into blocks), at five shapes: the two old ones, 16 ×
    20 × 20 → 32 of 3 × 3 (padded tiles), 1 × 1 at 128 × 16 × 16 → 128 (whole tiles), 4 × 9 × 12 → 6, stride 2,
    dilation 2. Mutations (the padding's sign): CPU `ImageRow` fails 164 checks; device packconv and gemmconv fail the
    fused checks of every tensor / dot type and float32 / float64.
  - measured `--subjects qiyas --ops conv` against 70f687d (a worktree): CPU conv 0.05–0.51× (float32 8.87 → 2.31 ms,
    float16 14.1 → 3.53, int8 2.96 → 0.86, q4_k 126 → 8.4, q6_k 189 → 10.2); device 0.15–0.90× (float32 0.358 → 0.143,
    float16 0.218 → 0.138, int64 0.724 → 0.200, q8_0 1.15 → 0.218); float64 as above. conv_bwd unchanged (the backward
    keeps the composition).
  - also in 4g's edits (the coordinator, from 4f): Known slow rows and item 6 entries for the device route with few
    targets and the CPU scatter_add_rows on block formats.
- 4f: done, committed as 70f687d. What landed:
  - `src/Ptx.Radix.cs`: `ScanTile` (a block of 256 scans 1024 values: shuffles within a warp, the warps' sums in shared
    memory; writes the block's total) and `ScanAdd`; `RadixCount` (per tile of 2048 sources, item-major, the digit
    counts by shared atomics into table[digit][tile]) and `RadixPlace` (the same tile in order: a warp's peers with the
    same digit from eight `vote.sync.ballot`s, the rank among earlier peers by popc of the lanemask, the earlier warps'
    counts through shared memory, a running count per digit across items; stable, any card).
  - `src/GpuOps.Radix.cs`: `ExclusiveScan` (recursive over the blocks' totals; in place allowed), `RadixOrder` (passes =
    ⌈bits(targets − 1) / 8⌉, ping-pong buffers, the last pass's values written into the caller's order), and
    `OrderByTarget` (counts by `ScatterCount`, offsets by the scan, offsets[targets] = n, the order by the radix sort),
    used by `scatter_add_rows` (GpuOps.Layout.cs) and `Route`. `ScatterScan` / `ScatterPlace` stay in Ptx.Layout.cs,
    unused.
  - gate (`Checks.Routings`): also 1M into 64, 1 000 003 into 100 000 (three passes, a scan over 98 blocks), all into
    one, descending into 65 536 — order, counts, offsets equal to the stable reference, both devices. Mutations: the
    scan's totals not added back fails the three multi-block cases; the earlier warps' share dropped corrupts the
    order, which the layouts gate's scatter-add then reads as source indices (an illegal-address fault: the run stops).
    Under `--repeat` the device runs these cases Stress / 20 passes (fresh indices each); `--repeat 400` (20 passes): equal.
  - bench: new row `route_wide` (1M into 65 536) in Group9. Measured against fd60205 (a worktree): device route 1M into 64
    216.7 → 0.274 ms; route_wide 0.186 (fewer atomics per counter than 64 buckets: the counts' contention is the larger
    part of `route` now — the radix table's totals could give the counts instead, not taken); device scatter_add_rows
    (4096 into 4096 × 4096) 0.15 ms less on every type (float32 0.457 → 0.306, int8 0.325 → 0.172); int64 read 0.827 →
    1.163 once, A/B the other order 0.675 new vs 0.826 old: swing. CPU unchanged (route 1.05 → 0.99; route_wide 2.22).
  - also in 4f's edits (the owner's rule for open timing questions): the CPU sub-byte products after 4e as a Known slow
    row in COVERAGE and under item 6 here.
- The two open timing questions (the CPU block product with AggressiveOptimization; device float64 non-causal attention,
  the composition timed at 89 ms by the helper) moved to item 6 by the owner: COVERAGE Known slow rows, and below
  under item 6. Item 4's full run only records them.
- 4e: done, committed as fd60205. What landed:
  - CPU `Kernels.Operands(AOff, TransA, BOff, TransB, COff, Viewed)` and `MatMul<TN, TIn, TAcc, TL>(…, Operands)` (the old
    signature calls it with `default`, a plain product): `PackBody` takes Bᵀ by `RunTransposed` (a task reads each Bᵀ
    row's span of its k-groups once — a unit of g values is contiguous there — and lays the units into the strips,
    zero past n and past k); `ProductBody` takes Aᵀ by `PanelOfTransposed` (k-row p of Aᵀ holds the tile's rows
    contiguously: R values a k-row straight into the [group][R] block; no staging or transpose step; non-lane types
    load the tile's span of each k-row once per panel); every index offset by the batch's base; `Micro` and the stores
    take C's offset. `Products.cs`: `InPlace<TN>` (rows on whole bytes: a byte or more, or a block format at even m, n,
    k; strides on whole storage units) → the in-place route per batch, ranges taken on the stored slice; otherwise
    `MatMulViewCopies` (the old body).
  - device: `GpuOps.PackedRoute<TN>` (tensor or dot lane at whole tiles) and `MatMul<TN>(…, transA, transB)`: Aᵀ by
    `Pack(…, transposed: true)` (the transposed packer, rows = m), Bᵀ by `Pack(…, transposed: false)` (the row packer);
    `MatMulViewRoute` takes a batch in place when every offset is on 16 bytes and the lane is packed, else the copies
    (rented only when a batch needs them). The plain f32 / f64 lanes and the block formats keep the copies (their
    kernels read row-major operands; a strided `Gemm` / `Lanes` is the step past this).
  - a bug found on the way by the bench: batch 0 of an untransposed batch has an all-zero Operands, which read as the
    plain product and failed C's exact-size check; `Viewed` marks a view. The gate had no untransposed batch: added.
  - gate: `ProductsWith` also at 64 × 40 × 330 and 64 × 48 × 320 (CPU) and 128 × 256 × 128 (device), and a batch of two
    untransposed products in every shape. Mutations: CPU Bᵀ units (one group for all) and Aᵀ rows (shifted by one) fail
    the CPU checks across types; device Bᵀ packed as if stored k × n fails the device in-place checks on every tensor and
    dot type (int4 included).
  - measured, `--subjects qiyas --ops products` against the same command on a784145 (a worktree; the baseline holds no
    product rows): CPU 0.07–0.87× on every byte-wide and block type (float32 matmul_nt 11.0 → 1.96 ms, float16 10.9 →
    2.16, int64 11.8 → 0.81, q4_k 34.4 → 9.6, q6_k 50.1 → 10.9; block errors lower: no re-encoding); device 0.38–0.87×
    on the tensor and dot types (int8 matmul_tt 0.274 → 0.103, float16 0.409 → 0.269), unchanged on f32 / f64 / blocks.
    Sub-byte types (copy route as before) flagged slower in three cells; one A/B both orders: int4 matmul_tn and int2
    matmul_nt slower in both (7–26 %), uint2 mixed — the shared kernel only gained offsets; re-check in the full run.
- GRU sequence backward (the owner's addition; after 4g): dx, dh0, dW, dU, db of the GRU sequence, on both devices, built
  the way the LSTM sequence backward is (float64 reference looped over the steps, gate in `Checks.Recurrents` on every
  float type incl. blocks, contract method, bench row `gru_seq_bwd` in Group8). "A complete core cannot leave it out."
  When it lands, COVERAGE's GRU sequences row stops saying the backward is not provided.
  Done. What landed:
  - `RecurrentCase.GruSequenceBackwardReference(a, T, x, h0, w, u, b, bh, dhOut)` → (dx, dh0, dW, dU, db, dbh): the
    forward by `GruSequenceReference`, then per step back: dh_t = dhOut[t] + the next step's dh; the step's gates;
    `CellArgs.GruCellBackward` → dgx, dgh, the cell's direct dh; dx_t = dgx·W, dW += dgxᵀ·x_t, db += Σ dgx; dh_prev =
    dh_cell + dgh·U, dU += dghᵀ·h_prev, dbh += Σ dgh. The gate checks it against central differences of the forward
    reference (loss Σ dhOut · h, every input, T 3, B 2, D 3, H 4, worst ≤ 1e-7).
  - `Kernels.GruSequenceBackward<TN>` (src/Recurrent.cs) and `GpuOps.GruSequenceBackward<TN>`
    (src/GpuOps.Recurrent.cs): the LSTM sequence backward's shape with `GruStepBackward` per step and dbh summed too; a
    narrower type or a block format widened once (lane storage; on the device block formats included, as the GRU
    forward).
  - contract `IBenchSubject.GruSequenceBackward(device, type, a, T, x, h0, w, u, b, bh, dhOut, dx, dh0, dw, du, db,
    dbh)` (default null), `QiyasSubject.GruSequenceBackwardVisitor`; gate in `Checks.Recurrents` (3 steps of 5 × 7 →
    11, every float type incl. blocks, both devices, all six gradients); bench row `gru_seq_bwd` (dx scored).
  - mutation (dh's carry from the next step dropped): the CPU gate fails 55 checks, the device gate 55, on every float type.
  - measured `cuda,cpu --subjects qiyas --ops gru_seq,gru_seq_bwd` (new rows, no before), float32: device 15.8 / 12.4 ms
    (two runs), f16 12.7 / 14.3, f64 111, q8_0 12.7; CPU 114 / 102, f16 119 / 89, f64 214, q8_0 97. 4.5–6× the forward:
    a Known slow row and item 6 below.

## Item 4a — flash-fused attention rows

Today: `Attention<TN>` is the composition (scores per head in lane storage: view products, `scale_mask_softmax`,
product with V); `AttentionDecode<TN>` is fused for a few queries (a block per query, K and V read once;
`Ptx.AttentionDecode`, `Ptx.DecodeThreads`, `dCap` ≤ 4 × threads); `AttentionBackward<TN>` is a composition.
CPU: `src/Attention.cs` (`Attention<TN>`, `AttentionDecode<TN>`, `AttentionBackward<TN>`, `AttentionArgs(Heads, Lq,
Lk, D, Scale, Causal, Spans, SpanStarts)`, the float64 reference `AttentionArgs.Reference(q, k, v, p?)`). Device:
`src/GpuOps.Attention.cs` (ToLane/FromLane lane storage; `Limits`/`Starts` per row from Spans), kernels in
`src/Ptx.Attention.cs`. COVERAGE row: "fused rows (flash: tiles of keys through the tensor lanes without the scores in
memory) | todo | todo".

Design intended:
- Device kernel `Ptx.AttentionRows(gpu, f, lane, dCap, tile)`: a block per (head, tile of 64 or 128 query rows); the
  keys walked in tiles of 64 through shared memory; per query row the online softmax (running max m, running sum l,
  accumulator o[d] rescaled by exp(m_old − m_new) when the max moves); S = Q·Kᵀ tile and O += P·V tile on the tensor
  instructions where the lane allows (f16/bf16 types: the f16 kind, `Ptx.GemmMma`'s fragment layout is the pattern;
  float32/float64 and the block formats: the plain FMA lane — the lane rule is the row family's "float"), the mask from
  `Limits`/`Starts` (causal: key limit = row position + 1; spans: [start, limit)), the scale folded into Q. No scores in
  memory. Register budget: 256 threads, each 1 query row × d ≤ 128 of O in registers in f32 (the decode kernel's
  register pattern); d > 128 falls back to the composition (say so in the note).
- Entry: `GpuOps.Attention<TN>` takes the fused kernel when `a.D <= 128` and Lq ≥ the tile (a `Note` records the
  choice); block formats through ToLane/FromLane as now; the composition stays as the fallback and the reference path.
- CPU: `Kernels.AttentionRows<TN, TLane, TL>` — per (head, row block of R rows) in a task: the online softmax over key
  tiles that fit L1 (keys × d lane values), the same recurrence, vectors along d; chunk tasks by `RowChunk<TN>` on the
  output rows (block formats and bool). Lane: the type's float lane (f32 / f64).
- Reference and gate: `AttentionArgs.Reference` (float64) is already there; extend `Checks.Attentions` to run the fused
  path explicitly on both devices (a flag on the call or an `AttentionArgs` option `Fused`), at the gate's shapes
  (heads 2–3, Lq/Lk odd such as 37/53, d 17/64/128, causal, spans with starts) against the reference within the lane's
  rounding (`Close` with terms = Lk, as the composition's check does). Also a check that fused == composition within
  f32 rounding at one shape.
- Bench rows: `attention_fused`, `attention_fused_causal` (16 heads, 2048², d 128 — the same shapes as `attention`/
  `attention_causal`, so the Known-slow rows "CUDA attention by the composition 6.1 ms" and "CPU attention" are
  measured against them), added to `Program.Group5` and `Program.cs` case list near line 544; `--shapes odd` sizes by
  `Prime()`.
- COVERAGE: the "fused rows" row → ✓ float / ✓ float with the lane note; Known-slow rows for attention updated with the
  measured numbers; README: the bench-options paragraph lists the two rows.

## Item 4b — ctc loss (forward + backward)

Today: `cross_entropy` is a composition (log_softmax → pick_rows; backward = row op `cross_entropy_bwd` in
`src/Rows.cs`). COVERAGE row "ctc | todo | todo".

Design intended:
- `src/Ctc.cs` (new): `CtcArgs(int T, int B, int C, int blank, int[] inputLengths, int[] labelLengths, int[] labels
  (flat, per batch), int maxLabel)`; the extended label sequence l' (length 2L+1 with blanks); `Reference` in float64:
  log-space α/β recursions, loss = −log(α_T(L') + α_T(L'−1)), gradient dlogits = softmax − (1/P) Σ over s with l'_s = c
  of α_t(s)β_t(s) (the standard formula, in log space with log-sum-exp).
- CPU: `Kernels.Ctc<TN, TLane, TL>(args, byte[] logProbs (T × B × C, log_softmax already applied by the caller or
  applied inside from logits), byte[] loss (B), byte[] grad (T × B × C))`: a task per batch element (B tasks, chunked by
  `RowChunk` for packed types on the grad rows); α and β in lane storage (f32 or f64, the float lane), log-sum-exp
  with the lane's exp/log (`IQiyasFloatLane`).
- Device: `Ptx.Ctc(gpu, f, lane)`: a block per batch element; threads over s (the 2L+1 states, ≤ 1024); α forward in
  time with a block barrier per t, then β backward, then the gradient per (t, c) by atomics-free accumulation (each
  thread owns states; the gradient per class gathered by a second pass over s per class, or shared-memory per-class
  sums with one thread per class); the result through lane storage for narrow types (`ToLane`/`FromLane` as attention).
- Gate: `Checks.Ctcs` — float types on both devices, small odd shapes (T 11, B 3, C 7, labels of length 2–4 incl. a
  repeated label, which exercises the blank rule), loss and gradient against the reference within the lane's rounding
  times T (the recursion's depth); an exactness check that the gradient sums to 0 per (t, b) within rounding.
- Contract: `IBenchSubject.Ctc(device, type, args, logits, loss, grad)` in `src/Bench/Contract.cs`, implemented in
  `src/Bench/QiyasSubject.cs` (the lib has no ctc kernel: lib column —; the `number` subject may skip: null).
- Bench rows: `ctc` and `ctc_bwd` (T 256, B 32, C 1024, labels 64) in a new `Program.Group10` (`--ops routing` or a
  new `--ops losses`), the README options paragraph updated; COVERAGE "ctc" row → ✓ float / ✓ float.

## Item 4c — the GRU sequence bench row

Today: `GruSequence<TN>` exists on both devices (`src/Recurrent.cs` ~453, `src/GpuOps.Recurrent.cs` ~255) and the gate
covers LSTM sequences only; the bench has `lstm_seq`, `lstm_seq_bwd` (`Program.Group8`, `Program.cs` ~449).
Design: a float64 reference `RecurrentCase.GruSequenceReference(a, T, x, h0, w, u, b, bh)` in
`src/Bench/RecurrentCases.cs` (the step reference looped, as `LstmSequenceReference`); `Checks.Recurrents` adds GRU
sequences (T 3, the gate's odd shapes) on both devices; the contract gets `IBenchSubject.GruSequence(...)` (default
null) implemented in `QiyasSubject.cs`; bench row `gru_seq` (16 steps of 256 × 512 → 512, as `lstm_seq`) in Group8.
The device's narrow types need the lane-storage wrapper `LstmSequence` has (the `!OwnLane(f) && !f.IsBlock` branch):
copy it for GRU; block formats go through it too (remove the `f.IsBlock` throw by routing blocks through the wrapper).
A GRU sequence backward is not asked for (the step backward exists); say so in the COVERAGE row.

## Item 4d — the block-scaled tiled product

Today: a product with a Q8_0/Q4_0 operand decodes to f32 and multiplies in f32 (both devices); gemv has the
block-scaled integer dot (`src/BlockProducts.cs`: `Kernels.GemvBlocks`, `Ptx.GemvBlocks`, `GpuOps.GemvBlocks<TN>`).
Design: for C = A·B with A in Q8_0/Q4_0 (weights) and B any float type: quantize B's columns per 32-block of k to s8
with an f32 scale (the activation side, as GGUF's Q8_1-style dot does), then C[i][j] = Σ_blocks dA_ib · dB_bj · (s8 dot
of 32 terms) — exact integer dots, the scales' product in f32. CPU: `Kernels.MatMulBlocks` — the product's tiling
(`ProductBody` pattern: tiles of 6 rows × strips) with the inner kernel an `S8Dot` lane dot per block
(`Vector`-widened multiply-add of sbyte pairs into int32, then one f32 FMA with the two scales); device:
`Ptx.MatMulBlocks` with dp4a per 4 values (a 32-block = 8 dp4a) per (row, column) tile from shared memory, or the s8
tensor kind (`s8mma`) per 32-block with the per-block scale applied to the int32 tile before the f32 accumulate.
Lane name `s8dp blocks` (as gemv). Note the error: B's quantization adds B's block step — the reference is A's decoded
values times B's stored values, so the gate's tolerance must include B's quantization step (or quantize B in the
reference too and say so); keep the f32 route as the exact one and choose between them only where the caller allows the
activation quantization (an option, default off: the plain product stays exact as today). Gate: `Checks.Products`
block types with the option on, against the reference with B quantized. Bench row `matmul_blocks` (q8_0, q4_0;
2048³ cuda / 1024³ cpu). Q4_K/Q6_K: their sub-block scales make the same scheme work per 32 values (Q4_K's scale × min
term needs the column sums of B per block: Σ dB·sum(qB)); do Q8_0/Q4_0 first, K formats if time allows, and say which.

### 4d as built (supersedes the draft above; written before building)

The COVERAGE todo is gemv's semantics carried to the tiled product: both operands in the same block format, the blocks
along k, the integer dots exact, the block scales applied in f32 — no activation quantization and no option. That is
C = A·Bᵀ (`ProductView` with TransB, batch 1): A (M × K) and B (N × K) both Q8_0 or both Q4_0, K a multiple of 32 —
the weights-times-activations layout (a linear layer's W stored N × K). A plain A·B cannot take it: B's blocks run along
N, and re-blocking along k would re-quantize. Q4_K / Q6_K are not taken (sub-block scales and Q4_K's min term need B's
per-block sums): say so in COVERAGE.

C_ij = Σ_blocks dA_ib · dB_jb · (Σ_32 qA·qB), the result in float32 lanes, encoded to the type by blocks (as gemv).
- CPU `Kernels.MatMulBlocks(format, a, b, float[] c, m, n, k)` (`src/BlockProducts.cs`): A's and B's blocks unpacked
  once into int8 rows (Q4_0: q − 8) and per-4-byte expanded scales (K/4 floats a row: the scale of the block each int
  lane of an s8 dot covers, so any vector width works). Tasks of 4 rows; a 4 × 4 register tile of Vector<float>
  accumulators; per vector of k: `S8Dot.Fma` (the s8 dot lane), converted to float, times sA·sB, added. Offered only
  where the runtime reports the s8 dot lane (else the f32 route stays). Lane `s8dp blocks`.
- device `Ptx.MatMulBlocks` / `GpuOps.MatMulBlocks<TN>`: the s8 tensor kind with k = 32 (`mma.m16n8k32.s8`, one
  instruction per block) where the JIT accepts it, else the f32 route. A block of 128 threads (4 warps) per 64 × 64 of
  C, a warp 32 × 32 (2 × 4 tiles); per k-block A's and B's 64 blocks staged into shared memory by 16-bit loads (a
  34-byte Q8_0 block is 2-byte aligned only; Q4_0's nibbles expanded to int8 − 8 on the way), rows 48 bytes apart (no
  bank conflicts for the fragment loads), the scales in f32 beside them; fragments by 32-bit shared loads in the
  m16n8k32 layout; each tile's s32 result converted and added as acc += c·(sA_row·sB_col). Lane `s8 mma blocks`.
- `MatMulView` on both devices takes it for that view and those formats when measured faster than the f32 route
  (once per shape and format, both warmed, alternating rounds; the decision noted, as attention's).
- gate (`Checks.Products`): q8_0 and q4_0 at 37 × 29 with K = 64 and 320, the block path forced on both devices,
  against the double-double reference of the stored values; the device gemv's block path at K = 64 too (its gate so
  far had 1001 columns, never a whole number of blocks).
- contract `IBenchSubject.MatMulBlocks(device, type, m, n, k, a, b, c)`; bench row `matmul_blocks` (q8_0, q4_0;
  C = A·Bᵀ at 2048³ on the device, 1024³ on the CPU) in Group4b beside `matmul_nt`.

## Item 4e — the transposition fused into the product's packing

Today: `MatMulView` makes each transposed or batched operand contiguous by the strided copy, then the product
(`src/Products.cs` CPU; `src/GpuOps.Products.cs` device). Design: pass the operand's layout into the packing instead.
CPU: `Kernels.MatMul` packs B by groups of k rows (`PackBody`) and reads A by rows (`ProductBody`): add a `transB`
flag so `PackBody` reads B's column j of row p from Bᵀ storage (the strip is then a contiguous read of Bᵀ's rows —
cheaper than today), and a `transA` flag so the A-panel load reads Aᵀ's columns (gather into `rowsIn`); the batch's
offsets passed as base indices (no copy). Device: `Pack(f, kind, …, transposed)` already exists for B in both
orientations (`Ptx.PackRows` with `transposed`, `Ptx.PackTransposed`): call it with the view's orientation and the
batch offset on the stored pointer (byte offset = values · bits / 8 when whole bytes; sub-byte offsets keep the copy);
the plain-lane kernels (`Gemm`, `Gemm2`) need a strided A/B variant or the lane conversion (`Lanes`) done strided — the
`to:` kernel with a stride. Gate: `Checks.Products` already covers Aᵀ, Bᵀ, AᵀBᵀ, batched — unchanged, must stay green;
bench rows `matmul_nt`, `matmul_tn`, `matmul_tt`, `matmul_batched` show the gain (compare.py on `--ops products`).

As built (see the progress entry 4e): the CPU as designed, every type whose rows start on whole bytes; the device in the
tensor and dot routes only (the packers' two orientations, offsets on the pointer) — the plain f32 / f64 lanes and the
block formats keep the copies (a strided `Gemm` / `Lanes` is the step past this, not taken in 4e).

## Item 4f — the radix step of scatter-add

Today: the device orders scatter-add sources by `scatter_count` (atomic counts) → `scatter_scan` (one warp scans the
counts) → `scatter_place` (each source's rank = the number of earlier sources with the same target: a loop over all
earlier sources — quadratic in n) (`src/Ptx.Layout.cs` ~727–794, `GpuOps.Layout.cs` ~40, the same three kernels in
`GpuOps.Route`). Design: a stable counting sort by blocks: (1) per block of 256 sources, per-target counts in shared
memory is too large for many targets, so instead: each block computes its sources' local ranks among equal targets
inside the block (a block-wide match: sort the 256 (target, index) pairs in shared memory by a bitonic sort, which is
stable when the index is the tie key), writes block-local counts per target as (target, block, count) runs; (2) a scan
over blocks per target gives each block's base offset per target (`blockBase[target][block]` is too big for many
targets — use the sorted runs: an exclusive scan of run counts in (target, block) order via a segmented scan, i.e.
sort the runs by target with a device radix sort on the target key (LSD, 8 bits a pass, stable)); simplest correct
version: a full LSD radix sort of the (target, sourceIndex) pairs — 8-bit digits, per pass a block-local histogram,
a scan of histograms across blocks (digit-major), a stable scatter — giving `order` directly; `counts`/`offsets` from
the sorted targets by a boundary scan. The scan kernel must be multi-block for large target counts (`scatter_scan`
today is one warp: also linear in targets). Gate: `Checks.Layouts` scatter_add cases + `Checks.Routings` route, plus a
case with 1M sources into 64 and into 100 000 targets (exact order equality with the CPU's stable order). Bench rows
`layout:scatter_add_rows` and `route` (1M into 64 buckets) show it; add `route` at 1M into 65 536.

## Item 4g — the fused direct convolution

Today: conv forward = im2col → product (both devices; `src/Conv.cs`, `src/GpuOps.Conv.cs`), the im2col matrix
(C·KH·KW × OH·OW) materialized. Design: an implicit-GEMM product — the product kernel reads B (the im2col matrix)
through an index map instead of from memory. CPU: in `Kernels.MatMul`'s `PackBody`, a mode where row p of B (one
(c, kh, kw)) and column j (one output pixel) are read from the image at (c, oh·sh − ph + kh·dh, ow·sw − pw + kw·dw) or
0 outside — the packing is the only reader of B, so the micro-kernel is untouched and the im2col buffer disappears.
Device: the same in the packing kernel for the tensor kinds (`Ptx.PackTransposed` for B: the index map in place of
the load) and in the plain lanes' B load. Lanes and bounds: the product's (the image's range bounds the im2col's).
Gate: `Checks.Convs` (strides, pads, dilations; forward and backward) unchanged and green; a check that the fused path
equals the composition bit for bit (same products, same order). Bench row `conv` shows the gain; COVERAGE conv row:
"im2col fused into the product's packing (implicit GEMM)".

## Item 5 — the buffer arena

Today: `Gpu.Rent(bytes)`/`Return(pointer)` is a size-keyed free list (`src/Gpu.cs` ~line 214: sizes rounded to 256
bytes; after a shape's first call no driver allocation; `--report` measures a rent/return pair at ~570 ns). Every
device operation rents its lane buffers and packed operands per call (search `gpu.Rent(` in `src/GpuOps*.cs`: Lanes,
Back, Pack, MatMulPadded, MatMulView, the attention/recurrent/conv wrappers). The CPU side rents from
`ArrayPool<T>.Shared` per call (`src/Kernels.cs` MatMul: `al`, `bl`, `cl`, `packed`, the per-task `panel`; Map/Rows use
stackalloc or ArrayPool for long rows).

Design intended:
- `GpuOps.Arena` (new file `src/Arena.cs`): a scope object `using var scope = ops.Scope()` holding a bump pointer over
  one device region reserved from the shapes of the first call (`Reserve(bytes)`; grows by reallocating a larger region
  between calls, never inside one); `Rent` inside a scope bumps, `Return` is a no-op; the scope's end resets the bump.
  The existing pool stays as the fallback when no scope is open (nothing removed). The per-call allocation count
  (`Arena.Rents`, `Pool.Rents`, `Pool.DriverAllocations`) exposed and printed by `--report` ("steady state: 0 rents,
  0 driver allocations per call") and checked by a gate: run `matmul` (view), `attention`, `lstm_step` twice in a scope
  and require the second call's driver allocations == 0 and pool rents == 0.
- The bench's `BenchRun` for the device wraps `Run` in a scope so the timed region has no pool lookups; the lane
  buffers for the core rows that are rented *outside* Run today stay as they are.
- CPU: `Kernels.Workspace` — the product's `al`/`bl`/`cl`/`packed` and the per-task `panel` kept in a per-thread
  workspace sized by the largest shape seen (`[ThreadStatic]` arrays grown on demand), replacing the per-call
  `ArrayPool.Rent/Return` on the hot path; the ArrayPool stays for the rest.
- Timing effect: measure before/after on the core rows and the view products (`--ops products`) with compare.py;
  expect the sub-ms device cells (16-bit adds, softmax f16 at 0.08 ms) to move a few µs; report both.
- COVERAGE: the paragraph "Device buffers come from a size-keyed pool…" rewritten to describe the arena, with the
  measured counts; README "--report" sentence updated.

## Item 6 — the known-slow list (COVERAGE.md "Known slow")

Each: measure against the baseline cell, change, re-measure (the full run or the family's `--ops`), compare.py.
- CPU column-wise reductions (max:cols 3.68 ms vs max:rows 0.96; sum:cols 1.44 vs 0.34): `src/Reduce.cs` the column
  bodies (lines ~270–300, `Team.For(c, …)` with `PartCompareBody`/`PartBody`): a column pass currently walks rows per
  column chunk; tile it as the row-wise kernels do — a task owns a strip of columns and sweeps all rows with vectors
  along the columns (the strided loads become contiguous), accumulators in registers per vector of columns. Same for
  `norm_params_bwd`'s column sums (`src/Rows.cs` `NormParamsBody`) and `gemv_t`.
- CPU attention by the composition (and CUDA 6.1 ms): item 4a's fused rows are the fix; measure the Known-slow rows
  against the fused rows.
- Device `lstm_step` 5.1 ms anomaly vs `gru_step`: `src/GpuOps.Recurrent.cs` `LstmStep<TN>` → `CellGates` (two view
  products + an add + a MapRow) + `LstmCell`; profile with `gpuprofile`; the suspect is the second product's operand
  copy through `MatMulView` (TransB → strided copy of U every call) — cache the transposed packed weights across calls
  (the arena of item 5 makes that natural: a packed-operand cache keyed by pointer + shape), or fuse the two products
  into one (x|h)·(W|U)ᵀ product (concatenate in lane storage once per step).
- CPU erf (17.8 ms per 16M float32; gelu_erf 41 ms): `src/Functions.cs` `Erf(Vector<float>)`: the current formula
  (see its comment) uses a division and exp; replace with the piecewise polynomial in |x| with the `Vector` FMA
  (Abramowitz–Stegun 7.1.26-class accuracy is not enough for the 2-ulp gate — use a higher-order minimax in two
  ranges, |x| < 1 and 1 ≤ |x| < 4, saturating to ±1 beyond), keep the ulps check (`check erf f32 (ulps)` must stay
  ≤ its current 2.13 ulp or tighten); the device emitter shares the coefficient tables (`src/Functions.cs` ↔
  `src/Ptx.Map.cs` `Fn.Erf`), so change both from the same table.
- CPU strided copy (16 ms vs 1.8 plain), pools (5–18 ms), top-k (113 ms, k passes): tile the gather by the source's
  rows (`src/Layout.cs` `MoveBody`); a partial sort (a per-row heap of k) for top-k in `src/Routing.cs` `TopKBody`.
- When a cell lands: update its Known-slow row with the new measurement (or remove the row and say so in the item
  report); README Results regenerated from the full run that includes it.
- CPU block-scaled product with AggressiveOptimization (the owner's, from 4d): steady 3.5–3.65 ms with the attribute,
  3.15–3.18 without, in one A/B (1024³ q8_0). Re-measure both orders; if the gap holds, find a way to keep first-call
  optimization (the reason: the first tier ran it 20× slower for the first rounds) without the steady-state cost.
  Stable on re-measurement: mark the Known-slow row done.
- Device float64 non-causal attention (the owner's, from 4d): 62.6 ms on the fused rows now, 59 by the composition at
  4a, and `Measure.Faster` timed the composition at 89. Re-measure both paths (the `attention` row; the composition
  forced through `AttentionComposed`); if the composition is really faster, explain the 89 and fix the measurement.
- Device route with few targets (the coordinator, from 4f): 0.275 ms for 1M into 64 against 0.185 into 65 536 — the
  counts' 1M atomics on 64 counters (`ScatterCount`) dominate. Take the counts from the radix table's totals instead
  (after the last pass's `RadixCount`, or a per-tile count summed by the scan: counts[t] = offsets[t + 1] − offsets[t]
  once the offsets come from the sorted keys' boundaries); `OrderByTarget` in src/GpuOps.Radix.cs.
- CPU scatter_add_rows on block formats (the coordinator, from 4f): q8_0 83.9 ms against float32 1.94 on the same shape
  (4096 into 4096 × 4096); 4f's run read q8_0 92.5, q4_0 86.0, q4_k 202, q6_k 336. Probably a per-row re-encode
  (`Blocks.Store` of a partial block re-encodes the whole block) or a serial path: find it (src/Layout.cs, the
  scatter_add_rows case) and fix.
- CPU sub-byte products on views (the owner's rule for open timing questions, from 4e): int4 matmul_tn, int2 matmul_nt,
  uint2 matmul_nt slower by 7–26 % after 4e in one A/B (uint2 mixed); the coordinator's run read int4 tn 1.88 ms, int2 nt
  2.22. They keep the copy route (`MatMulViewCopies`); the shared kernel gained offsets (`Operands`, `Micro`'s cOff,
  `PackBody`'s transB branch). Re-measure both orders against fd60205's parent a784145; if it holds, give sub-byte types
  the in-place route (loads at even indices: `TN.Load` at whole bytes) or remove what the offsets cost them.
- Device float64 conv (from 4g): the composition through the sliced route 2.55 ms against the fused plain kernel 3.40;
  `SlicedFaster` (GpuOps.cs: one warm call and three timed) read the sliced product at 3.56 ms instead of 2.15 in one
  run and flipped the conv choice. Time `SlicedFaster` by `Measure.Faster`; consider the image packed straight into the
  sliced route's digits (`Ptx.SliceColumns` reading through the conv index map) so float64 conv is fused there too.
- CPU recurrent sequences (the coordinator's observation at 4c): gru_seq 65–71 ms, lstm_seq 83 ms against ~15–20 ms
  implied by the f32 product rate (COVERAGE Known slow). The GRU sequence backward piece's runs read CPU gru_seq float32
  at 20.7 and 22.1 ms (after 4e's in-place products): re-measure lstm_seq too; the row may be done.
- Device products with Aᵀ after item 5's arena (one run, no A/B at the coordinator's order): tn / tt float16
  1.11–1.14×, bfloat16 1.10–1.14×, int32 / uint32 1.17–1.30×, int64 / uint64 tt 1.59×, batched int4 / int2 1.39–1.45×
  against item 4's baseline, nn / nt unchanged. The Aᵀ route packs A into a rented buffer: compare the pool and the
  arena on that path (`GpuOps.FusedConv`-style switch: the bench's scope off) once; if the arena costs it, find why
  (the packed A adjacent to B in one region, or the growth's synchronize on the first scoped calls).
- Item 4's full run, slower cells in code item 4 changed (not re-timed at the coordinator's order): CPU matmul
  float64 1.16×, float16 1.38×, float8_e5m2 1.41× (4e's product kernel: `Operands`, `Micro`'s cOff, the in-place
  route), CUDA matmul int4 1.15× (4e), CUDA softmax float16 1.12× (the launch-limit query, cached since). Re-time
  each once, both orders, against a build of d9939ff's parent where the change landed; fix what holds.
- Item 4's full run, errors above the lib's (accuracy, not time): CUDA float32 sum:whole 5.98e-4 against the lib's
  1.95e-4 and float8_e5m2 3.66e-4 against 6.1e-5 (the order of the device's whole-array sum: pairwise or compensated
  partials are the step); float32 sin, cos, gelu, silu, add_rms_norm up to 1.4×; float16 softmax and sum:rows float16,
  bfloat16 up to 1.3×. Find each one's cause; bring them to the lib's or below, and the README sentence with them.
- Device cross_entropy bench row (from item 4's full run): 46 ms float16, 95 float32 against the CPU's 18–23 at 4096 ×
  32768 — the row chains three subject runs through host arrays (`Program.cs` RoutingRow "cross_entropy"), so the
  device time is the 512 MB upload of logp and the downloads, not the kernels (log_softmax alone 0.075 ms float16).
  Give the contract a device-resident chaining (or a fused cross-entropy loss row on both devices) and re-measure.
- GRU sequence backward (the owner's rule for open timing questions, from its piece): 4.5–6× the forward on both
  devices where the products imply ~4× (float32 device 12.4–15.8 ms vs 2.55–2.76; CPU 102–114 vs 21–22; two runs).
  Per step `GruStepBackward` recomputes gx, gh that the forward recompute already made, and dW, dU, db, dbh are summed by
  separate `Map add`s (CPU: fresh arrays per step). Re-measure; if it holds, keep the forward's gates (T × B × 3H each)
  and accumulate the weights' gradients in place (the products' C += form), for the LSTM sequence backward too.

## End of item 6 — the final readiness check (the owner's addition)

The goal: Qiyas ready to replace Idrak's kernels entirely once items 4–6 are done. After item 6's full run, compare
the list of Idrak's operation names against COVERAGE.md, reading names only and never design (the standing rule:
nothing of Lib's code or design enters the repository). Report every operation an application would need that Qiyas
lacks, with what it would take; nothing is added without the coordinator's go.
