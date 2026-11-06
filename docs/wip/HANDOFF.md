# Handoff: items 4–6 of the coverage stretch

Written 2026-11-06 at the end of item 3, for a session with none of this one's context. Read `CLAUDE.md`,
`COVERAGE.md` ("State" first) and `README.md` before anything. Standing rules: Idrak is an application of Qiyas, never
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
| Regression gate | `python docs/wip/compare.py docs/wip/baseline.md results-qiyas-cuda-cpu.md` — prints every cell with base/new ms, ratio, lanes, errors; flags `SLOWER` (> 5 %) and `ERR UP`; "new row" for rows the baseline lacks. `baseline.md` is the run of 2026-11-06 00:51 (the state before the coverage work; no block/bool rows). Re-time a flagged cell several times before calling it swing: this laptop's sub-2 ms cells move ±20–40 % run to run; an A/B against a build of the baseline commit (`git worktree add <dir> c73a251`, build it, run both dlls back to back) separates code from machine state. |
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
- The open intermittent (COVERAGE.md State, item 4 there): `matmul_bwd dB device` unsigned types wrong by 138 steps,
  seen once, not reproduced in 150 gate passes. If it shows again, `QIYAS_MAP_DEBUG=1` on the run captures the values.

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
