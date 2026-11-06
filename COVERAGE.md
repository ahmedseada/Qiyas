# Coverage

What Qiyas computes, in its own terms: one row per operation, one column per device. A cell is **✓** with the lane
rule the operation follows, **n/a** with the reason, or **todo**. Every ✓ has a float64 reference and an error column
in the bench, a correctness gate in `Checks` that runs over every type the operation applies to on both devices before
anything is timed, and a bench row at a realistic shape.

The type set: float64, float32, float16, bfloat16, float8 e4m3, float8 e5m2; int64, int32, int16, int8, int4, int2;
uint64, uint32, uint16, uint8, uint2; bool (1 bit); the block formats Q8_0, Q4_0, Q4_K, Q6_K (the GGUF layouts: values
in blocks with shared f16 scales). A block format's lane is float32; it is read a value at a time anywhere (both devices
decode by index) and written a block at a time: on the CPU every store re-encodes the blocks it touches with the
block's other values kept (tasks never share a block; the block's scale then covers the kept values, so an output buffer
of a block format is one whose blocks the output covers whole or whose other values belong with it, not one holding stale
values of another scale); on the device the operations that store value by value run in float32 storage and encode the
result by blocks (all four formats have device encoders, byte for byte the CPU's).

Lane rules, named once here and referred to below:

- **own** — the type's element-wise lane: integers their own width (modular arithmetic at that width), the 16- and
  8-bit floats f32, float64 f64. Scalars are converted as the type stores them.
- **float** — the type's float lane: f32 for every float narrower than 32 bits, f64 for float64.
- **sum** — the cheapest lane exact for the bounds measured on the data: the device's dot-product or tensor
  instructions for 8-bit-ranged inputs, f32 to 2^24, s32/u32, then f64 to 2^53 and s64/u64 on the CPU, s64/u64 at
  once on the device (exact further than f64 and several times faster where f64 runs at a fraction of the f32 rate);
  float64 products may take the sliced 8-bit tensor route (`f64x`, exact) where the device measures it faster. Where measuring would cost a pass as
  long as the operation and could not buy a faster lane (16- and 32-bit integer sums on the device), the type's range is
  the bound.

Applications map onto these operations; where an application's kernel list has a name, the note says which Qiyas
operation covers it. Nothing here mirrors any application's structure.

Every bench row has a shape option (`--shapes odd`: the nearest primes below the regular sizes) and the run has a
report option (`--report`: the device's and the CPU's facts and the lanes chosen, measured — nothing assumed about a
vendor, a vector width or a capability; a kind the JIT refuses falls back to the plain lanes).

Device buffers: inside an arena scope (`GpuOps.Scope()`, src/Arena.cs; the bench opens one around every timed Qiyas call
on the device) every rent bumps a pointer in one region and a return releases from the top (a buffer returned below the
top once everything above it is), so a call's buffers are a stack in one allocation: no pool lookup, no driver call. A
rent past the region comes from the pool for that call, and the scope's end grows the region to the call's peak (never
inside a call), so the next call of the same shapes bumps everything. Outside a scope the size-keyed pool serves as
before. `--report` measures a rent/return pair (pool 653 ns, arena 14 ns on this machine) and a transposed float16 256³
product's second call in a scope (0 pool rents, 0 driver allocations). Gate (`Checks.Arenas`): a transposed product,
attention and an LSTM step in float16 and float32: from the second scoped call no pool rent, no driver allocation,
nothing past the region, unreturned or returned late, the region's high-water mark equal to the peak alive at once, and
the results the pool's bit for bit. The CPU product's packed B, loaded A / B, C lanes, A panels and transposed k-rows come
from a per-thread workspace (src/Workspace.cs) grown to the largest shape seen (a slot in use on the thread falls back to
ArrayPool); the gate requires the product and the LSTM step to reach calls with no workspace allocation, results
unchanged.

## State (2026-11-06, branch `coverage-wip`)

Written so the next session continues from the files alone. The tree builds (Release, 0 errors) and `guard.sh` reports
zero hits.

**Green, confirmed in a full gate run on both devices for every type (the 17 plain types, bool, and Q8_0, Q4_0, Q4_K,
Q6_K):** every gate in `Checks` — element-wise, reductions, rows (37 × 1001, 5 × 64, 3 × 17, 3 × 20011), layout,
view products and the backward pair, attention with spans and decode, optimizers, convolution, recurrent incl. LSTM
sequences, routing, argmax per column, adaptive-pool backward — the block specifications (`BlockSpecifications`: blocks
laid out by hand decode by the formulas of all four formats), the block round trips, the core operations (add, softmax,
rms_norm over 37 × 1001) on the block formats on both devices, and the device's decode and encode of all four formats
byte for byte as the CPU's (`BlockFormats`), the sliced f64 proofs, `--report` on both devices. The
transpose of a block format is repeated 20 times in the gate (`Checks.Stress`, raised by `--repeat N`): the schedule-
dependent re-encode it once showed (reproduced at repetition 25 of 40 without the fix) is gone with the column tasks
aligned to whole blocks; 100 repetitions × 4 formats are green.

The drafted patch `docs/wip/k-block-encoders.py` of the previous state is applied and removed; `docs/wip/` keeps the
regression comparer and the baseline run.

**Open, in the order they are taken:**

1. The todo rows below (item 4, in this order: fused attention rows — done, ctc — done, the GRU sequence bench row — done, the block-scaled
   tiled product — done, the transposition fused into the product's packing — done, the radix step of scatter-add — done, the fused direct
   convolution — done); the designs and the progress are in `docs/wip/HANDOFF.md`.
2. A bump-allocated scope past the pool (buffers reserved once from the shapes; zero rents on the hot path) — done
   (item 5: the device arena and the CPU product workspace, above).
3. The known-slow list.

**Settled — the device products intermittent** (`matmul_bwd dB device uint64 / uint32 / uint16 DIFFERS by 138 steps`,
seen once in item 3's first full run; again in item 4 as `matmul_bwd dA … by 123` and, under the forced 4-bit kind,
every product of the types whose values fit 4 bits). Cause: the tensor kernel's configuration is measured per kind at
2048³ (`GpuOps.MmaShape`), and a configuration with 128-byte rows walks k in chunks of 128 bytes — 256 values of a
4-bit kind — while the product padded k to whole 128 values only; at a k of 128 (any padded k below 256) each row's
chunk read the next row's values. The 4-bit kinds' timing picks a 128-byte configuration on some runs only (`u4k32 ->
rows 128 B x 3 stages x 128 wide swizzled` in both failing runs; `rows 64 B …` in the passing ones), hence the
intermittence. Fixed: k is padded to whole chunks of the configuration chosen (`GpuOps.KPadded`), in the product and in
the kinds' timing. Gate: `check tensor configurations` forces each of the 9 configurations in turn under each of the
4 kind choices and runs the product of every type a tensor kind takes at k = 53 and k = 300 (1152 products); with the
fix reverted it fails 102 of them, all under the three 128-byte configurations with a 4-bit kind.

**Settled — the regression gate for the stretch (items 1–3).** The full `cuda,cpu --report` run of 2026-11-06 12:38
(every type, every gate green on both devices, 128 rows) against `docs/wip/baseline.md` (00:51): 36 new rows (the block
formats and bool on both devices), no error grown except the float64 product rows' expected change under the double-
double reference (cpu matmul float64 6.4e-14 → 1.1e-13: the f64 lane's own rounding, formerly hidden), and 15 cells
beyond 5 % — all sub-2 ms cells re-timed three times and, for the two consistent ones (CUDA softmax/rms_norm of the
16-bit floats 0.077 → 0.081 ms, CPU matmul float16 1.45 → 1.75 ms), timed back to back against a build of the baseline
commit (c73a251): the old build measures the same today (0.083–0.085 and 1.76–2.62 ms), so the difference is the
machine's state since 00:51, not the code. The CUDA float64 product took the sliced route (f64x: 4.19 ms against 22.7,
exact under the double-double reference).

**Settled — the float64 product reference.** `References.MatMul` (the bench's and the gates' float64 reference for every
product, view and backward) accumulates in double-double: each product exact by a fused multiply-add, each sum with its
error term. A lane that is exact now reads 0 (the sliced f64 route, the integer lanes) and a float64 lane shows its own
rounding (~1e-13 at 2048³, which the sequential reference used to hide by rounding the same way).

**Settled — the CPU int4 matmul `ArgumentOutOfRangeException`.** Reproduced: `cpu --types int4 --ops matmul --repeat 30`,
the third of three runs, at `Kernels.cs` line 306 of `ProductBody.Run` (the tile's store, the first range-checked access
of the tile: the panel, transpose and micro-kernel before it address by reference without checks). The site is the
thread team, not the product: `Team.Take` read the job's counter (generation, next chunk) and then the chunk count as
two reads; a worker that read the counter at the end of job G and the chunk count after the caller had begun describing
job G+1 (a larger count, the description half written: body pointer, invoke, sizes) could take chunk `chunks(G)` under
that mixed description — a tile of garbage fields, and a chunk of G+1 accounted as done without running. Fixed in
`Team.For`: a job is closed when its last chunk completes (its chunk counter set past any count), so the worker's
exchange finds the counter changed and takes nothing. The gate `TeamJobs` (`check team`) runs 20 000 jobs (raised by
`--repeat`) of alternating chunk counts — a chunk per thread, then 256 chunks — and requires every chunk within its job
and every index run exactly once; it did not reproduce the window on the unfixed team in 60 000 jobs (the window is a
few nanoseconds wide unless a thread is preempted inside it), so the fix rests on the analysis above and on the
reproduction not recurring: 150 repeats of the int4 product after the fix.

## Known slow

Measured cells that are correct but below what the device or the lanes allow; each is addressed at a natural point
and measured against the baseline when it is.

| where | what | measured |
|---|---|---|
| CPU | column-wise reductions and compares against row-wise | max:cols 3.68 ms vs max:rows 0.96 (float32, 4096 × 4096); sum:cols 1.44 vs 0.34; int8 max:cols 1.12 vs 0.074. 6h: the column passes as tiles of a strip of columns (about 8 KB of accumulators) by a block of rows, at least four a thread, each block's result in its own slot joined in block order; values read in place where stored as the lane; argmax by a vector compare a row (4-byte lanes take value and row by its mask). float32 sum:cols 1.46 → 0.60 ms, max:cols 2.30 → 0.54, argmax:cols 1.87 → 1.24; float16 0.96 → 0.59, 3.73 → 0.57, 1.37 → 1.25; int8 0.36 → 0.30, 1.17 → 0.071, 4.39 → 1.38; float64 2.45 → 1.47, 3.80 → 1.47, 2.33 → 1.94 (baseline → one run). float32 sum:cols error 3.3e-4 → 3.8e-5 (shorter sums); compares exact |
| CPU | max / min / argmax slower than sum over the same bytes | float32 max:rows 0.96 vs sum:rows 0.34; min:rows 1.30 6j: max and min (their result the value) no longer search the row a second time for the first position, and the vectors run in four chains joined lane by lane (each lane the extreme of the same positions: the lane's Max / Min are IEEE maximum / minimum, checked commutative for ±0 and NaN, integers exact): float32 max:rows 0.811 → 0.385 ms, min:rows 0.997 → 0.385, max:whole 0.782 → 0.386 (sum:rows 0.352 in the same run); float16 1.315 → 0.470, 1.586 → 0.704, 0.755 → 0.479; float64 1.659 → 1.276, 1.580 → 1.265, 1.748 → 1.412; int8 within swing (0.068 → 0.071); argmax:rows keeps the position search (float32 0.939 → 0.686, the chains alone) (item 4's baseline → one run; errors 0) |
| CPU | erf and the formulas built on it | erf 17.8 ms, gelu_erf 41 ms, gelu_erf_bwd 28.5 ms per 16M float32 values (exp 1.74). 6g: the f32 lane's tail (\|x\| ≥ 0.75) is e^(−x²)·t·S(t), t = 1/x, S a degree-16 polynomial fitted to 3.2e-9 (Horner in a shifted variable), for the continued fraction's 48–80 serial divisions; the f64 lane's fraction as a ratio of two recurrences (a division every 16 levels). float32 erf 22.6 → 6.7 ms, gelu_erf 25.8 → 6.0, gelu_erf_bwd 28.5 → 8.0; float16 23.3 → 7.8, 37.5 → 4.8, 44.4 → 9.7; float64 71.5 → 65.6, 339 → 221, 382 → 226 (baseline → one run). Errors unchanged; `check erf f32 (ulps)` 2.13 as before, new `check erfc f32 / f64 (ulps)` 3.80 / 3.00 |
| CPU | gemv_t (columns) against gemv (rows) | 9.4 vs 2.9 ms float32, 10.7 vs 1.9 float16 (8192 × 8192). 6h: the column tiles: float32 6.65 → 3.19 ms (gemv 2.98), float16 7.17 → 2.42 (1.85), int8 2.40 → 1.50 (0.80), float64 14.0 → 6.98 (6.38) (baseline → one run). Errors: float32 4.2e-4 → 4.7e-5; float64 0 → 6.0e-13 (the baseline summed in the reference's own row order; gemv reads 6.1e-13) |
| CUDA | 16-bit broadcasts below bandwidth | int16 maprow:mul 0.146 ms (440 GB/s) against int8 0.035 (900 GB/s) and int32 0.173 (740 GB/s). 6j: the card's memory clock (P1 / P3 / P5 within a run), not the kernel: int16 0.087 and float16 0.086 ms as the minimum of three runs (int16's baseline 0.088). One `rem` a word for the column (a `rem` a value before): int4 0.037 → 0.028, int8 0.040 → 0.031 |
| CUDA | function-bearing two-input formulas in float16 | exp_bwd, tanh_bwd … 0.238 ms against add 0.126 (compute-bound f32 functions). 6j: 0.124–0.129 ms with the card at P1, equal to add: the memory clock, closed with no change |
| CPU | rope, scale_mask_softmax, log_softmax against the plain row operations | rope 2.9 ms, scale_mask_softmax 3.3, log_softmax 2.5 against rms_norm_affine 1.7 (float32, 4096 × 4096); float16 log_softmax 3.6 against layer_norm 0.9. 6j: rope's sine and cosine of each angle from one reduction (`SinCos`, equal bit for bit to Sin and Cos: each computed both polynomials and the reduction) and the pairs' frequencies kept per thread for the call's (n, a) (an exp a pair a row before): float32 4.70 → 1.82 ms, float16 5.99 → 1.30 (layer_norm 1.72 / 0.95 in the same run). scale_mask_softmax reads x and the mask only over the columns a causal row keeps (they were scaled across the whole row, the rest then overwritten): float32 3.47 → 2.57, float16 3.16 → 2.98 (the driver widens whole rows of the narrow types). log_softmax 1.8–2.4 float32, 2.1 float16 with unchanged code in later runs: layer_norm plus one exp a value, closed with no change. Errors equal |
| CPU | norm_params_bwd (column sums with row statistics) | 3.0–3.8 ms against 1.7 for a row pass over the same bytes. 6h: the column tiles: float32 rms 3.33 → 1.99 ms, layer 3.37 → 2.01; float16 4.73 → 1.60, 4.80 → 1.79; float64 6.80 → 4.69, 6.78 → 4.80 (baseline → one run; the row statistics pass included). Errors float32 3.1e-4 → 4.5e-5 / 3.2e-5 |
| CUDA | three-stream row operations in float32 in a full run | scale_mask_softmax 0.50 ms in the group's run, 0.26 alone (a run-order effect also seen on add float8_e5m2) |
| CPU | the strided copy (a 3-d permutation of 16M values) | 16.2 ms float32, 7.3 int8 against 1.8 for a plain copy: a value-at-a-time gather (tiling by the source's rows is the step). 6g: when one outer dimension has stride 1 (a transposition), tiles of output rows along it: each column's values read as one contiguous line into a tile, each row then stored whole. float32 15.6 → 2.7 ms, float16 31.4 → 3.6, float64 21.2 → 5.3, int8 3.7 → 0.65, q8_0 42.0 → 7.6 (baseline → one run); exact; two gate cases added (the tiled dimension the first and the second) |
| CPU | the pools and the depthwise convolution | maxpool_bwd 5.1 ms, adaptive_maxpool 5.8, depthwise_conv 4.8 (float32, 64 × 256 × 256); int8 adaptive_maxpool 18.6, depthwise 24.4 (value-at-a-time loads of a packed type). 6j: the team measured a layout job by its outputs, so an adaptive pool's 3136 outputs (each a 37 × 37 window) ran on the caller alone; the measure is now outputs × the values each reads (`LayoutArgs.ValueReads`): adaptive_maxpool float32 8.53 → 0.62 ms, float16 7.98 → 0.92, int8 15.4 → 1.06; adaptive_avgpool float32 → 0.29. depthwise_conv at column stride 1: each tap a vector pass along the output row (each output's sum the same terms in the same order): float32 5.34 → 1.07, float16 6.14 → 1.38, int8 6.76 → 0.74. adaptive_avgpool_bwd: each window's share added over its columns (the windows' bounds were recomputed for every pixel): float32 8.28 → 0.28, float16 9.99 → 0.42. maxpool_bwd 2.7–4.3 float32 across runs, unchanged code. Errors equal; gate: two depthwise cases (a column stride of 2; rows of 37 with a column dilation of 3) |
| CUDA | float64 conv: which path | the composition through the sliced float64 tensor route 2.55 ms against the fused plain kernel 3.40 (item 4g); in one full products run the sliced product's own measurement (`SlicedFaster`: one warm call, three timed) read 3.56 ms instead of 2.15, the conv choice took the fused path and the row read 3.41. To time `SlicedFaster` by `Measure.Faster` in item 6, and to consider the image packed into the sliced route's digits (the fused path for the sliced lane) |
| CPU | the convolution backward | 33 ms against 6.9 forward (64 × 128 × 128 by 128 of 3 × 3): three passes in lane storage. 6g: dw = dy·colsᵀ reads the columns from the image in the transposed packing (no im2col pass, no columns in memory; the same packed values, so equal bit for bit), and the columns' gradient for dx in a buffer kept per thread (two fresh 41 MB arrays a call before). conv_bwd float32 16.1 → 8.7 ms, float16 17.6 → 12.8, float64 30.1 → 16.3 (baseline → one run), errors unchanged. The forward read 3.4 ms float32 in the same run against 2.3 in the baseline (its code unchanged; it has read 2.3–6.9 across runs) |
| CUDA | attention rows | 3.8 ms (causal 2.1) for 16 heads of 2048² at d = 128 by the fused rows in the FMA lane, from 6.1 by the composition (since item 4a); the tensor lanes inside the fused kernel are the step. float64: see the next row. 6i: the tensor rows (`Ptx.AttentionMma.cs`) for float16, bfloat16 and the 8-bit floats (their values exact in f16 / bf16): S = Q·Kᵀ and O += P·V by m16n8k16, P as two slices of the kind (hi + lo: the f32 lane's rounding kept; the lo slice dropped fails the gate by 80–416 steps). attention_fused float16 3.79 → 0.81 ms, bfloat16 3.79 → 0.81, float8_e4m3 4.72 → 1.22, float8_e5m2 4.81 → 1.23; causal 2.15 → 0.50 (16-bit), 2.65 → 0.76 (8-bit); errors equal. float32, float64 and the block formats keep the FMA rows (no exact kind): float32 3.78 |
| CUDA | float64 non-causal attention: which path | 62.6 ms on the fused rows since item 4d, against 59 by the composition at 4a; the shared measurement (`Measure.Faster`) timed the composition at 89 this time and chose the fused rows. To re-measure both paths in item 6 (the `attention` row, the composition forced); if the composition is really faster, explain the 89. 6i (one run): the measured choice took the composition, `attention` 59.5 ms against `attention_fused` 60.8: the two paths within 2 %, the measurement consistent this time (the 89 not seen again) |
| CUDA | route with few targets | 0.275 ms for 1M into 64 against 0.185 into 65 536 (item 4f): the counts' 1M atomics on 64 counters dominate. 6j: up to 256 targets the counts and offsets come from the one radix pass's scanned table (no atomics): 0.277 → 0.088 ms, into 65 536 unchanged (0.186); every route case gated both ways |
| CPU | scatter_add_rows on block formats — done (item 6) | was q8_0 84.8 ms, q4_k 180 against float32 2.35: a float type's input range was read serially before the sum lane was chosen (block formats decode every block for it; a float type's lane does not depend on it), and every value was loaded one at a time. Now q8_0 5.8–7.0 ms, q4_k 5.3–8.3. The same fix made the CPU layout family 7–140× faster on narrow floats, packed and block types (the input decoded once by the team), and the max-pool backward reads each window's first largest from one pass (adaptive_maxpool_bwd float32 685 ms → 0.23, q4_k 52 s → 1.25) |
| CPU | sub-byte products on views after item 4e | int4 matmul_tn, int2 matmul_nt, uint2 matmul_nt slower by 7–26 % after 4e in one A/B (both orders; uint2 mixed); the coordinator's run read int4 tn 1.88 ms, int2 nt 2.22 (1024³). These types keep the copy route; the shared product kernel gained offsets. To re-measure in item 6; if it holds, give sub-byte types the in-place route or remove what the offsets cost them |
| CPU | the block-scaled product with AggressiveOptimization | steady 3.5–3.65 ms with the attribute against 3.15–3.18 without in one A/B (1024³ q8_0, item 4d); the attribute keeps the first calls off the runtime's first tier. To re-measure in item 6 in both orders; if the gap holds, keep first-call optimization without the steady-state cost |
| CPU | attention rows | 60–80 ms float32 (causal 33–38) for 16 heads of 2048² at d = 128 by the fused rows, from 100–148 by the composition (since item 4a). 6i: Kᵀ's tiles laid once per call (each 16-row task transposed every key tile again: 128 times a head): attention_fused float32 78.2 → 39.5 ms, causal 40.1 → 20.2; float16 81.2 → 51.4; float64 200 → 68.1, causal 99.9 → 33.9 (baseline → one run); `attention` (the measured choice) float32 51.3 → 39.1. Errors equal |
| CUDA | attention backward by the composition | 4.0 ms for 8 heads of 1024². 6i: the composition's time was its three products with k = 1024 (dV, dQ, dK: 3.5 of 4.75 ms in a phase probe, 1024 × 128 outputs on 8 tiles of the f32 product kernel: about 2 TFLOPS). The fused backward (`Ptx.AttentionBackward.cs`, d ≤ 128): float32 3.92 → 1.56 ms, float16 4.29 → 1.72, bfloat16 6.73 → 1.61, float8_e5m2 4.08 → 1.64, q4_0 4.07 → 1.64, float64 31.9 → 16.9; errors equal (float64 1.1e-16 → 1.5e-16). The FMA lane for every type (a tensor backward for the 16-bit kinds is a further step). 6j: the tensor backward (`Ptx.AttentionBackMma.cs`) for the formats a tensor kind holds exactly: float16 1.72 → 0.495 ms, bfloat16 1.61 → 0.472, float8_e5m2 1.64 → 0.677, float8_e4m3 4.65 → 0.674 (one run; errors equal to the baseline's). 6j: the f32 product kernel on few-tile shapes takes split-k (slices of k in grid y, the partial C summed in slice order): the composition's k = 1024 products are such shapes (not re-timed here); the fused backward does not use them |
| CUDA | int8 adaptive_maxpool | 0.265 ms against float32 0.079 (per-value unpacking of 4 values a word in the window loop). 6j: the cause was a thread per word over 784 words (4 windows each, serially); a packed type whose output has fewer words than the device holds threads takes a thread per value, the word joined by shuffles: int8 0.266 → 0.051 ms, int4 0.423 → 0.048, float16 0.145 → 0.067 (adaptive_avgpool float16 0.144 → 0.067). The max-pool backwards read each window's first largest from one pass (it was rescanned for every pixel): adaptive_maxpool_bwd float32 6.99 → 0.230 ms, float16 11.4 → 0.220, int8 23.7 → 0.218, int4 41.5 → 0.235; maxpool_bwd float32 0.122 → 0.089, int8 0.099 → 0.088 (one run each; errors equal) |
| CPU | top-k per row — done (item 6) | was 113 ms float32, 305 int8 (k passes over the row); now one pass keeping the k largest in order (the same values and indices: the comparison is a total order): float16 81 → 7.3–7.8 ms, int8 313 → 26; block formats widened by the team (q8_0 278 → 62) |
| CUDA | ctc's recursion | 0.54 ms forward, 1.18 backward (float32; T 256, B 32, C 1024, 64 labels), of which the rows kernel is a small part: a block per sequence (32 of the device's blocks) and a barrier per frame, each step waiting on global loads of the state's class, flags and emission; the emissions computed beforehand and the next frame's loads issued before the barrier are the step. 6j: up to 256 states a thread a state, its class and flags in registers, the next frame's loads issued before the barrier, the blank's class sum by warp shuffles and a label's by next-state links; the host's state table in one walk (about 310 µs of host time a call before) and one upload: float32 0.544 → 0.206 ms, backward 1.182 → 0.499; float16 0.518 → 0.208, 1.125 → 0.643 (one run; errors equal). Kernel time at the bench shape (Nsight Systems): rows 25 µs and recursion 76 µs forward, 85 and 320 backward |
| CPU | ctc | 1.8 ms forward, 3.6–3.8 backward (float32, the same shape): a task per sequence (32 sequences on 24 threads, the recursion serial within one); moving the rows' work into a pass over all rows measured slower (single-value loads of the emissions), so it was not kept. 6j, measured, not kept: the three passes (all rows' log-sum-exp and emissions by the team, the recursions per sequence, the gradient rows by the team; every value by the same steps) read 2.48 / 3.67 and 2.48 / 3.79 ms against the committed code's 1.74 / 3.47 and 2.36 / 3.60 in alternating `--report` runs (20 s cool-down). A probe of that version: rows 1.19 ms, the sequences 1.49, the gradient rows 1.03. One sequence's α recursion takes 0.153 ms on an idle thread and 0.21–0.32 with all 24 threads busy (a scratch program on the same lane code), against 0.44–0.5 inside the bench run; the exp's −∞ inputs (a subnormal product before the select) were not the cause (no change when removed) |
| CPU | LSTM / GRU sequences | gru_seq 65–71 ms, lstm_seq 83 ms float32 (16 steps of 256 × 512 → 512, about 13 GFLOP of products) where the CPU's float32 product rate implies 15–20 ms: per step the gate products, the cell and the copies go through separate passes and allocations. Item 6 (6f): x·Wᵀ of all steps in one product, the step's buffers kept: float32 lstm_seq 25.3, gru_seq 17.9 ms (item 4's baseline 27.2, 28.9) |
| CUDA, CPU | GRU sequence backward | 4.5–6× the forward on both devices (16 steps of 256 × 512 → 512, float32: device 12.4–15.8 ms against 2.55–2.76, CPU 102–114 against 21–22) where the products imply about 4×: the step's backward recomputes the gates the forward's recompute already made, and each step's dW, dU, db, dbh are summed by separate adds (on the CPU into fresh arrays). To re-measure in item 6; keeping the recomputed gates and accumulating into the sums in place are the steps. CPU done in 6f: the recompute keeps every step's gates, per step only the cell's backward and dg·U, dx / dW / dU / db over all the steps in one product each: float32 gru_seq_bwd 109 → 57 ms, lstm_seq_bwd 150 → 69 (3.2× and 2.7× the forward). The device in 6f2, the same restructure: float32 gru_seq_bwd 12.5 → 5.5 ms, lstm_seq_bwd 26.0 → 6.6 (3.3× and 3.4× the forward). The device's float64 lstm_seq_bwd error 4.4e-16 → 2.0e-15 since 6f2 is the float64 lane's rounding, not a regression: the float64 product's route is measured per shape, and the new shapes take the plain float64 lane where the per-step shapes took the sliced route; both within the product's proof. 6j: the f32 product on few-tile shapes by split-k (the recurrent backwards' k-long products): device float32 lstm_seq_bwd 6.57 → 4.29 ms, gru_seq_bwd 5.51 → 3.87 (one run) |
| CUDA | products with Aᵀ after the arena (item 5) | tn / tt float16 1.11–1.14×, bfloat16 1.10–1.14×, int32 / uint32 1.17–1.30×, int64 / uint64 tt 1.59×, batched int4 / int2 1.39–1.45× against item 4's baseline in one run, nn / nt unchanged. Item 6 (6e): not the arena (the pool reads the same); int32, uint32, batched back within swing; the 16-bit transposing pack word-wide (0.025 → 0.021 ms a 2048² operand, at the bandwidth); float16 tn / tt now nn + that pack (0.285 against 0.260 in the profile). Left: the pack itself (an Aᵀ read in the product kernel) and C through f32 (0.018 ms, a 16-bit epilogue); int64 2048³ reads 0.36 or 0.57 ms by run in any orientation (nn too), not Aᵀ. 6j: both done for float16 / bfloat16 — C stored as the type by the tensor kernel, and the operands read as stored (Aᵀ and B k-major by ldmatrix.trans on row-swizzled k-rows, Bᵀ as rows): float16 matmul 0.243 → 0.229 ms, nt 0.246 → 0.220, tn 0.263 → 0.234, tt 0.267 → 0.236, batched 0.627 → 0.520, matmul_bwd 0.504 → 0.444; bfloat16 matmul 0.244 → 0.229, tn 0.262 → 0.233, batched 0.624 → 0.522, matmul_bwd 0.503 → 0.444 (item 4's baseline → one run; errors equal). A k-major operand costs about 4 % of the kernel on the measured configuration, so nn at 2048³ only breaks even; the pass it replaces is fixed per operand, and smaller m gains more |
| CPU, CUDA | item 4's full run: slower cells in changed code | CPU matmul float64 1.16×, float16 1.38×, float8_e5m2 1.41×, CUDA matmul int4 1.15× against item 4a's baseline (4e changed the product kernels); CUDA softmax float16 1.12× (the launch-limit query, cached since). Not re-timed: item 6 re-times each once, both orders |
| CUDA | the cross_entropy bench row | done in item 6 (6d): the fused loss row, 0.65 ms float32 and 0.64 float16 against 95 and 48 through the composition's host round trips; float64 12.8 ms (the device's f64 exp rate) |
| CPU | LSTM / GRU steps | 20–31 ms forward, 56–84 backward at B = D = H = 1024: the products dominate (four gates, two operands); a fused gates product is the step. 6f: float32 lstm_step 14.9, backward 41.6 ms — at the product rate (17 and 51 GFLOP); the cell's backward vectorized. float16 22.9 / 62.6: the step widens W and U to f32 every call (the API passes the type's bytes). 6f2: W and U kept widened per array (checked unchanged by their bytes each call); the swing itself was the bench's first row measured under tiered compilation (50.7 ms first, 19.2 with it off). 6g: the bench's per-row timing settles as `Measure.Faster` does (warm until the runtime compiles nothing and the run stops getting faster, capped; tiered compilation and PGO stay on): bfloat16 lstm_step first in a run 50.7 → 22.3 ms |

## Element-wise (map)

One kernel over `<TN, TLane, TL>` per device runs every formula; a formula takes up to three inputs and two scalars.
Backward formulas are formulas. Lane: **own** for the formulas every lane has, **float** for the rest.

| operation | CPU | CUDA | note |
|---|---|---|---|
| copy, fill, neg, abs, sign, square, relu, add_scalar, scale, affine, clamp | ✓ own | ✓ own | integer types modular at their width |
| add, sub, mul, div, max, min, axpy, muladd, where | ✓ own | ✓ own | `where`: x ≠ 0 takes y, else z; integer div truncates |
| square_bwd, abs_bwd, relu_bwd, clamp_bwd, where_bwd | ✓ own | ✓ own | (x, dy) → dx |
| sqrt, inv_sqrt, exp, expm1, log, sin, cos, tanh, sigmoid, erf | ✓ float | ✓ float | functions proven ≤ 4 ulp (exp 1, log 2, sin/cos 2, tanh/sigmoid 3, erf 4) |
| gelu (tanh form), gelu_erf, silu, leaky_relu, pow, dropout, gated_silu, gated_gelu | ✓ float | ✓ float | gelu as x·σ(2u), gelu_erf as x·Φ(x) by erfc: no cancellation; on the device's f32 lane gelu and silu as x / (1 + e^(−t)) (gelu's t carried as an exact pair), sin / cos on the reduced argument as a pair |
| sqrt_bwd, inv_sqrt_bwd, exp_bwd, log_bwd, sin_bwd, cos_bwd, tanh_bwd, sigmoid_bwd, silu_bwd, gelu_bwd, gelu_erf_bwd, leaky_relu_bwd, pow_bwd | ✓ float | ✓ float | sech² and σ(x)σ(−x) forms |
| gated_silu_bwd_x/y, gated_gelu_bwd_x/y | ✓ float | ✓ float | (x, y, dz) → dx or dy |
| row-vector broadcast: output[r][c] = op(x[r][c], v[c]) for add, sub, mul, div, max, min | ✓ own | ✓ own | `MapRow` |
| convert (any type to any type) | ✓ | ✓ | every load and store, proven bit for bit (`Checks`) |

Applications' names covered: abs, add, add_scalar, affine, axpy, clamp, cos, dropout (mask given), exp, fill, gated
activations, gelu, gelu_erf, inv_sqrt, leaky_relu, log, mul, muladd, pow, relu, sigmoid, sign, silu, sin, sqrt,
square, sub, tanh, where, and each one's backward; softmax (log), softmax_bwd, rms_norm_affine, rms_norm_backward,
layernorm_fused / layernorm_train / layernorm_bwd / layernorm_bwd_params, weight_norm, rope, scale_mask_softmax,
add_rms_norm_affine → the row family; add_rowvec / broadcast_axis → row-vector broadcast; sum, sum_rows,
sum_axis, sumsq, max, min, argmax → the reductions; transpose; gemv_nn / gemv_nt / the quantized gemv families → gemv
(the stored type is the type; no separate kernel per format).

## Products

| operation | CPU | CUDA | note |
|---|---|---|---|
| matmul C = A·B | ✓ sum | ✓ sum | tensor/dot lanes first; float64 by `f64x` where faster (proven 0 ulp; shapes in multiples of 8, others in the f64 lane); on the device a shape not in the kernels' tiles (m, k in 128s, n in 256s) is padded with zeros (pad2d) and C taken back by the strided copy |
| gemv: y = A·x, y = Aᵀ·x | ✓ sum | ✓ sum | a warp per row / a thread per column on the device; dot lanes on the CPU (rows are groups as stored) |
| multi-vector products | ✓ sum | ✓ sum | matmul with a narrow B |
| batched, strided, transposed operands (`ProductView`) | ✓ sum | ✓ sum | the product kernel reads the operands where they are stored (same lanes, bounds, proofs): CPU every type whose rows start on whole bytes (a byte or more, and the block formats at even shapes; batch offsets on whole storage units) — Aᵀ's k-rows straight into the A blocks, Bᵀ's rows into the packed strips, a batch's offset as a base index; the device's tensor and dot routes at whole tiles (Aᵀ by the transposed packer, Bᵀ by the row packer, batch offsets on 16 bytes as pointer offsets; since 6j a float16 / bfloat16 operand is not packed at all: the tensor kernel reads Aᵀ and B k-major by ldmatrix.trans and Bᵀ as rows, and stores C as the type). The rest (sub-byte types on the CPU; on the device the plain f32 / f64 lanes, the block formats through lane storage, shapes off the tiles) makes each operand of a batch contiguous by the exact strided copy first. A transposed block-format operand is no longer re-encoded through a copy (its error dropped, e.g. CPU q4_k matmul_nt 4.65 → 2.73). Gate: also at 64 × 40 × 330 and 64 × 48 × 320 (CPU) and 128 × 256 × 128 (device), where the views take the in-place route, plus a batch of untransposed operands |
| matmul backward (dA = dC·Bᵀ, dB = Aᵀ·dC) | ✓ sum | ✓ sum | two products on views |

## Rows and normalization

| operation | CPU | CUDA | note |
|---|---|---|---|
| softmax | ✓ float | ✓ float | exp proven; rows in registers on the device up to 16 words a thread |
| rms_norm | ✓ float | ✓ float | |
| log_softmax; softmax_bwd (y, dy); log_softmax_bwd (log y, dy) | ✓ float | ✓ float | `RowOps`: one kernel shape per device runs every row formula (up to two rounds of block reductions) |
| rms_norm_affine (x, g); rms_norm_bwd (x, dy, g) → dx | ✓ float | ✓ float | x̂ = x·rstd; dx = rstd·(g·dy − x̂·mean(g·dy·x̂)) |
| layer_norm (x, g, bt); layer_norm_bwd (x, dy, g) → dx | ✓ float | ✓ float | two-pass variance (mean, then Σ(x − μ)²): no cancellation |
| norm_params_bwd: dgamma = Σ_r dy·x̂, dbeta = Σ_r dy (rms or layer statistics) | ✓ float | ✓ float | results as lane values; row statistics then a thread per column over row splits |
| weight_norm (x, s per row) | ✓ float | ✓ float | y = x·s/‖x‖ |
| rope (x, positions per row or the row's index; base a) | ✓ float | ✓ float | pairs (i, i + cols/2); θ = pos·a^(−2i/cols); sin/cos proven ≤ 2 ulp; an odd row's last value kept |
| scale_mask_softmax (x, additive mask; scale a; causal when b ≠ 0: columns past the row's position, the index or the one given) | ✓ float | ✓ float | excluded values give 0, a fully excluded row too; the position per row makes a key span |
| add_rms_norm, add_layer_norm (x, residual, g[, bt]) → y and h = x + residual | ✓ float | ✓ float | fused residual add + normalization, the sum written once |
| the row family on integer types | n/a | n/a | the row formulas are float formulas: a stored integer type has no float lane here |
| rows longer than 16 words a thread of 1024 (16384 float32 values) in the row family | ✓ | ✓ | the row in segments of 256 threads × 8 words (or 8 values, for rows not ending on a word), reloaded in every phase (the per-value temporaries recomputed); the statistics kernel of norm_params_bwd the same |
| a row kernel whose registers a block of its planned threads cannot hold (on this card: 1024 threads × 16 words, e.g. float16 log_softmax of 32768 or float64 of 16384) | n/a | ✓ | the driver's per-kernel limit (`Gpu.MaxThreads`: the kernel's maximum threads per block) is asked once the kernel is compiled; past it the row operation and the norm statistics take the segmented shape, softmax and rms_norm their looping kernel; gate: the row family, softmax and rms_norm at 2 × 16384 and 2 × 32768, every float type |

## Reductions

| operation | CPU | CUDA | note |
|---|---|---|---|
| sum, sum of squares: whole, per row, per column | ✓ sum | ✓ sum | results as lane values; on the CPU a dot lane sums packed groups against ones; a whole sum on the device takes the narrower lane its chunks' bound allows and the wide lane over the partials |
| max, min: whole, per row, per column | ✓ own | ✓ own | results as lane values |
| argmax: whole, per row | ✓ own | ✓ own | the first largest; s32 indices |
| argmax: per column | ✓ own | ✓ own | the first largest down the column; s32 indices (a thread per column on the device) |

## Layout and indexing

| operation | CPU | CUDA | note |
|---|---|---|---|
| transpose | ✓ own | ✓ own | 32 × 32 tiles through shared memory; rows not ending on a word written value-wise |
| strided copy (permute, transposed and sliced views, broadcasts with a zero stride; up to 4 dims) | ✓ own | ✓ own | `Layout`: a thread per output word, every value gathered by index: no atomics in any format |
| one_hot, gather rows, gather columns, shift (zero fill), pad2d (rows widened with zeros) | ✓ own | ✓ own | exact moves; pad2d also pads a product's operands to the kernels' alignment |
| scatter-add rows (the embedding gradient) | ✓ sum | ✓ sum | each target row sums its sources in source order (deterministic); the device orders sources by counts (integer atomics), offsets (a multi-block exclusive scan) and a stable LSD radix sort of the (target, source) pairs, 8-bit digits, a pass per byte of the target count (`GpuOps.OrderByTarget`, `src/Ptx.Radix.cs`; ranks within a warp from eight ballots, so no instruction a card may lack): linear in the sources, the same order as the CPU's |
| im2col / col2im | ✓ own / ✓ sum | ✓ own / ✓ sum | strides, pads, dilations; col2im gathers each input pixel's column entries in kernel order |
| interpolate: nearest, bilinear (half-pixel centers) | ✓ own / ✓ float | ✓ own / ✓ float | nearest by integer arithmetic (exact); bilinear in the float lane |

## Attention

| operation | CPU | CUDA | note |
|---|---|---|---|
| rows of queries: O = softmax(scale·Q·Kᵀ, causal or spans)·V per head | ✓ float | ✓ float | the fused rows or the composition, whichever the device measures faster at the shape and type (once, by `Measure.Faster`: both warmed, then alternating rounds of several calls, the minimum of each, on until both settle and the runtime compiles nothing; the decision in `--report`); the composition for d > 128. The composition: Qiyas's own operations in lane storage (a narrow type widened once, O narrowed once): the view product, scale_mask_softmax with the key limit per row, the view product |
| decode: a few queries over a long cache | ✓ float | ✓ float | the device's fused kernel: a block per query, keys in chunks, the softmax online, K and V read once; the CPU per query in the lane |
| backward: dQ, dK, dV | ✓ float | ✓ float | the composition: P recomputed, dV = Pᵀ·dO, dP = dO·Vᵀ, dS = softmax_bwd, dQ = scale·dS·K, dK = scale·dSᵀ·Q. CUDA tensor backward (6j; float16, the 8-bit floats in f16, bfloat16 in bf16, d ≤ 128): the same two kernels on m16n8k16 with f32 accumulators, 4 warps per 64 query rows (Q's and dO's fragments in registers; the forward's pass for m, 1/l, D, then S, dP, dS and dQ += dS·K 32 keys at a time) and per 64 keys (Sᵀ = K·Qᵀ, dPᵀ = V·dOᵀ, dV += Pᵀ·dO, dK += dSᵀ·Q); P and dS as the A operand in two slices of the kind (22 bits in f16, 16 in bf16); the gradients stored as the type. Each route only where the device takes it: the tensor kernels where both kernels' shared memory (34.8 and 70.9 KB at d = 128) fits the device's opt-in limit and the device builds both, else the FMA fused kernels where theirs fits (f32 71.8 and 76.7 KB at d = 128), else the composition; the forward's tensor rows also only where the device builds them. Gate: the backward's seven shapes also on the FMA fused kernels (`GpuOps.MmaBackward` false) and as on a device limited to 48 KB (`GpuOps.AttentionSharedLimit`: d ≤ 64 keeps the tensor or FMA kernels, d = 120 / 128 takes the composition, each route asserted and green); a mutation dropping the lo slice fails at 22 cells, one ignoring the limit at 8. CPU (6j): two fused passes where measured faster than the composition (once per shape and type): rows (a task per head and 16 query rows: the scores over the keys the rows see by Kᵀ's tiles, P, dP = dO·Vᵀ by Vᵀ's tiles, dS = P·(dP − Σ P·dP)·scale, dQ = dS·K; Pᵀ and dSᵀ written for the block's keys in runs of 16) then keys (a task per head and 16 keys: dV = Pᵀ·dO, dK = dSᵀ·Q over the row blocks that see them); causal and span masks skip the keys no row sees; no reductions across tasks; Pᵀ and dSᵀ for a group of heads within 32 MB. float32 attention_bwd 16.3 → 6.8 ms, float64 27.4 → 12.6, float16 18.2 → 9.4 (8 heads of 1024², causal; item 4's baseline → one run); float32 error 1.47e-7 → 1.72e-7 (another summation order, within the lane's rounding). CUDA, d ≤ 128: two fused kernels in the lane (f32, f64), no scores in memory, no atomics: the rows kernel (a block per 32 query rows: the forward recomputed for each row's max, 1/l and D = dO·O, then dQ += scale·P·(dP − D)·K over the same key tiles) and the keys kernel (a block per 32 keys, the query tiles that see them: dV += Pᵀ·dO, dK += dSᵀ·Q in registers). Gate: six shapes (the composition's case and the fused rows' cases: d 17 / 24 / 64 / 120 / 128, causal, spans with starts, rows that see no key), the fused kernels and the composition each |
| fused rows (flash: tiles of keys without the scores in memory), d ≤ 128 | ✓ float | ✓ float | the softmax online (running max and sum, the context rescaled when the max rises); block formats through lane storage. CPU: tasks of 16 query rows, keys in tiles of 64 laid transposed for vector scores, row pairs in registers. CUDA: a block of 256 threads per (head, 32 query rows; 16 in f64), keys in tiles of 32 (16) through shared memory with 16-byte loads, group reductions by shuffles, causal and span tiles skipped per block; FMA lane for float32, float64 and the block formats. float16, bfloat16 and the 8-bit floats: the tensor rows (4 warps per 64 query rows, key tiles of 64; S and P·V by m16n8k16 with f32 accumulators, P as hi + lo slices of the kind; K and V by 16-byte asynchronous copies for the 16-bit types, 32-bit words for the 8-bit, else value by value). Gate: the composition's cases plus d 17 / 64 / 128, causal, spans with starts, rows that see no key; bench rows `attention_fused`, `attention_fused_causal` |
| spans as a key range per query: [start, limit) | ✓ | ✓ | the limit by scale_mask_softmax's position, the start by an additive mask row (the composition) or the decode kernel's first chunk |

## Convolution and pooling

| operation | CPU | CUDA | note |
|---|---|---|---|
| conv forward (one image; strides, pads, dilations) | ✓ sum | ✓ sum | im2col fused into the product's packing (implicit GEMM; item 4g): no columns in memory; the product's lanes, bounds (the image's range with 0) and kernels. CPU: the packing builds each row of the columns from the image (every type whose rows start on whole bytes; otherwise the composition im2col → product). Device: the tensor and dot routes pack the image straight into the kind's rows (the composition's padding, kernel and C path: the same result bit for bit); the plain lanes (f32, f64, the integer lanes, block formats through f32) run the plain product kernel with its B tile gathered from the image, or the composition where measured faster (once per shape and type, `Measure.Faster`, in `--report`): float64 at 64 × 128 × 128 by 128 of 3 × 3 keeps the composition through the sliced float64 tensor route (2.55 ms against 3.40 fused), float32 and the block formats take the fused path (0.14 ms against 0.35). Block formats: the fused path reads the image's values where the composition re-encoded the columns into blocks. Gate: the fused path forced, equal bit for bit to the composition through the same kernel on both devices, five shapes (strided, dilated, padded and whole tiles) |
| conv backward input, backward weight | ✓ float | ✓ float | the composition in the float lane's storage (the columns' gradient is a sum of products no narrow type holds between the passes); on the CPU dw's columns read from the image by the transposed packing (gate: equal bit for bit to the composition on the fused forward's five shapes, and within the products' rounding of float64); integer types: n/a, a gradient needs a wider type |
| depthwise convolution | ✓ sum | ✓ sum | a layout operation: a gather with the channel's weights |
| maxpool, avgpool (strides, pads: a window's count excludes the pads), their backward passes | ✓ own / ✓ float | ✓ own / ✓ float | layout operations; maxpool backward recomputes each window's first largest (deterministic) |
| adaptive max / average pools | ✓ own / ✓ float | ✓ own / ✓ float | windows [⌊i·n/o⌋, ⌈(i + 1)·n/o⌉) |
| adaptive max / average pool backward | ✓ sum / ✓ float | ✓ sum / ✓ float | layout operations: each input pixel gathers the windows holding it |

## Optimizers

| operation | CPU | CUDA | note |
|---|---|---|---|
| sgd_momentum, adam, adamw (p, g as the type; m, v in the float lane's storage) | ✓ float | ✓ float | one pass over the arrays; Adam's bias correction by the step count |
| clip by the global norm | ✓ | ✓ | the composition: sum of squares (whole) → the factor → scale (both operations in the bench) |

## Recurrent

| operation | CPU | CUDA | note |
|---|---|---|---|
| LSTM and GRU cells (h', c' from the gates), their backward (the gates' gradients) | ✓ float | ✓ float | one pass over the rows (a thread per hidden unit); sigmoid and tanh proven |
| LSTM and GRU steps (x, h → h', c') and their backward (dx, dh, dc, dW, dU, db, dbh) | ✓ float | ✓ float | the gates by the products on views and the broadcast add, in the float lane's storage for a narrow type (the gates are sums of products); the backward by the cell backward and the products' backward |
| LSTM sequences (T steps: h, c every step) and their backward (dx, dh0, dc0, dW, dU, db) | ✓ float | ✓ float | a loop of steps; the backward recomputes the forward and walks the steps back, the weights' gradients summed |
| GRU sequences (T steps: h every step) and their backward (dx, dh0, dW, dU, db, dbh) | ✓ float | ✓ float | the loop of the GRU step; a narrower type or a block format widened once, h kept in the float lane across the steps, narrowed once (both devices, block formats included); the backward recomputes the forward and walks the steps back through the step's backward, the weights' and both biases' gradients summed, in the same lane storage; gate: 3 steps of 5 × 7 → 11 against the step reference looped in float64 (the backward's reference checked against central differences of the forward's), every float type; bench rows `gru_seq`, `gru_seq_bwd` (16 steps of 256 × 512 → 512) |

## Quantized and block formats

| operation | CPU | CUDA | note |
|---|---|---|---|
| Q8_0, Q4_0, Q4_K, Q6_K as types (Load decodes blocks into the f32 lane, Store encodes blocks) | ✓ | ✓ | the GGUF layouts, the codecs proven against blocks laid out by hand; every operation above runs on them (the device's value-wise stores through float32 storage and a block encode) |
| products with a block-format operand | ✓ sum | ✓ sum | matmul through the f32 lane on both devices (a block's values, an f16 scale times an integer, are exact in f32 and not in the f16 tensor lane); gemv on Q8_0 / Q4_0 by the block-scaled integer dot (32-term s8 dots per block, the block scales' product in f32: the format's own fast path, `s8dp blocks`); the tiled product C = A·Bᵀ of two Q8_0 / Q4_0 operands (k in whole blocks, the weights-times-activations layout) by the same block-scaled dot where measured faster than the decoded route (once per shape and format by `Measure.Faster`, the decision in `--report`): CPU the s8 dot lane in 4 × 2 register tiles (`s8dp blocks`), the device the s8 tensor instruction at k = 32, one per block (`s8 mma blocks`, four k-blocks staged a time by coalesced 16-bit loads); a plain A·B keeps the decoded route (B's blocks run along n), and Q4_K / Q6_K keep it too (sub-block scales and Q4_K's min term). The device gemv's block path read its values by 32-bit loads that a 34- or 18-byte block leaves 2-byte aligned (every other block faulted; no gate reached it: the reductions' gate has 1001 columns): now by 16-bit halves, and gated at 37 × 64. Gate: A·Bᵀ at 37 × 29 × 64 and × 320, the block product forced, both devices; bench row `matmul_blocks` |

## Masks

| operation | CPU | CUDA | note |
|---|---|---|---|
| bool as a type (1 bit, 8 a byte); and / or / not are mul / max / (1 − x) in its lane; mask-select is `where` | ✓ own | ✓ own | the packed-integer machinery at 1 bit; `div` is n/a on bool (a divisor of 0 is half its values) |

## Routing, sampling, losses

| operation | CPU | CUDA | note |
|---|---|---|---|
| top-k per row (values as the type, indices s32; largest first, ties by the lower index) | ✓ own | ✓ own | the CPU: one pass keeping the k largest in order (a value enters when strictly larger than the k-th; ties keep the lower index); the device: k passes of the row's argmax (k ≤ 256); the CPU's row tasks chunked on whole storage units of the k values (bool, the packed integers), the device's block formats through lane storage |
| pick per row (x[r][index[r]]) | ✓ own | ✓ own | |
| sampling per row (the first column whose running sum reaches u·Σ) | ✓ float | ✓ float | chunk sums, their prefix, the least crossing |
| route: counts, offsets, the sources ordered by bucket | ✓ | ✓ | the scatter-add's ordering (deterministic): on the device the radix sort above — 1M into 64 buckets 0.27 ms (216.7 by the quadratic ranks before item 4f), 1M into 65 536 0.19 (`route_wide`). Gate: also 1M into 64, 1 000 003 into 100 000, all into one, descending into 65 536, each equal to the stable reference |
| cross-entropy rows: the loss per row (one pass: running max and Σ exp, log Σ − x[target]; also the composition log_softmax → pick → neg), its backward (softmax − one_hot, a row operation) | ✓ float | ✓ float | the loss gated against float64 on both devices (`check routing`) |
| expert products (a product per bucket over the routed rows) | ✓ | ✓ | gather_rows by the order, the batched product on views, scatter-add rows back: the composition |
| ctc: the loss per sequence and its gradient with respect to the logits (T × B × C; log_softmax inside; per sequence its frames and labels, any blank class) | ✓ float | ✓ float | log-space α/β recursions over the 2L + 1 extended states; the gradient y − Σ exp(α + β − log y − log P) per class; a sequence no alignment fits: loss +∞, gradient 0. CPU: a task per sequence, the states a vector at a time (the lane's exp and log), float32/float64 rows in place, narrower floats a row at a time in the task, block formats through float32 storage. CUDA: a warp per (frame, sequence) row for the log-sum-exp and the softmax row, then a block per sequence: the states over its threads, a barrier per frame, each class's states summed by the class's first state (no atomics). Gate: the reference's gradient against its own loss by central differences; odd shapes with short sequences, empty and repeated labels, a blank other than 0, no alignment; the gradient of each frame sums to 0. float8_e4m3 cannot hold a loss past 448 (the bench row's ≈ 1800 store as NaN, its error column ∞); its gradient row is within the type |
