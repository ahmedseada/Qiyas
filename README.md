# Qiyas

Qiyas (قياس, "measure") is a generic number contract and the kernels over it: every stored type — float64, float32,
float16, bfloat16, float8 (e4m3, e5m2), int64 down to int2, uint64 down to uint2, bool (one bit), and the block formats
Q8_0, Q4_0, Q4_K, Q6_K (the GGUF layouts: values in blocks with shared f16 scales) — computed in the lane the hardware
does best for it, chosen per operation from the data itself. No type is fixed anywhere: not for values, not for sums.

- **One definition per type.** A type is one struct: its description (name, bits, range) and the lanes it offers, each
  lane a vectorized load and store. Kernels are written once over `<TN, TLane, TL>`; adding a type adds no kernel.
- **The lane is a measured choice.** Element-wise work runs in the type's own lane. A product scans its operands'
  range and takes the cheapest lane in which every sum is exact: the processor's dot-product or tensor instructions
  for 8-bit (or 8-bit-ranged) integers and for 16-bit and 8-bit floats, f32 while the bound fits 24 bits, s32, f64,
  s64. Which instructions exist is asked of the device at run time; nothing names a card or a processor.
- **CPU and CUDA from the same contract.** The CPU side runs on its own thread team and `Vector<T>`; the CUDA side
  generates PTX per (type, lane) with its own driver binding — tensor-core, dot-product and plain GEMMs, row kernels,
  conversions — all from the type's description.
- **Proven, not assumed.** Before anything is timed, every conversion is checked bit for bit against a scalar
  reference written from the format's definition, the dot lanes against scalar sums, exp against the runtime's, and
  the device's conversions against the CPU's; then every operation of every family runs over every type it applies to
  on both devices, on small odd shapes, against float64 references (`src/Checks.cs`); the block codecs against blocks
  laid out by hand from the formats' specifications; the block-scaled products of Q8_0 / Q4_0 (A·Bᵀ and gemv, k in whole
  blocks) forced on both devices; the products on views at shapes where the packing reads transposed and batched
  operands where they are stored, on both devices; the products on the device under every tensor kind the device
  accepts (the measured choice, then the 4-bit kinds and the 32-deep 8-bit instruction forced on and off), and under
  every tensor-kernel configuration the device can run with every kind choice (k = 53 and 300, every integer and
  narrow-float type); attention rows by both the composition and the fused rows (causal, spans with starts, head
  dimensions up to 128, rows that see no key); ctc's loss and gradient (the float64 reference's gradient checked
  against its own loss by central differences; short sequences, empty and repeated labels, another blank, a sequence
  no alignment fits); the thread
  team under 20 000 jobs of alternating chunk counts (every chunk within its job, every index once). The float64
  reference of every product is accumulated in double-double, so an exact lane reads 0. The bench refuses to time while
  any gate fails.
- **The operation surface.** `COVERAGE.md` is the list, one row per operation with the lane rule on each device:
  element-wise formulas and their backward passes; reductions, broadcasts, transposes, gemv; the row family
  (softmax, log_softmax, the norms and their backward passes, rope, masked softmax, cross-entropy's backward); layout
  and indexing (strided copies, one-hot, gathers, scatter-add, im2col/col2im, interpolation, shift, pad, the pools and
  their backward, depthwise convolution); products on views (transposed, batched) and the matmul backward; attention
  (rows by the composition or the fused rows, whichever is measured faster; causal, spans, decode, backward); convolution forward and backward; the optimizer steps; LSTM and GRU cells and
  steps with their backward passes, LSTM and GRU sequences with theirs; top-k, pick, sampling, routing, cross-entropy and ctc (loss and
  gradient); and the block formats and bool
  through all of it. What is still open is listed in COVERAGE.md's "State" and marked **todo** in its rows.

## The bench

`Qiyas.dll [cpu|cuda|cuda,cpu] [--subjects a,b] [--types ...] [--ops add,matmul,softmax,rms_norm,map|reduce|rows|layout|products|attention|optimizers|conv|recurrent|routing|all] [--baseline name] [--shapes regular|odd] [--report] [--repeat N]`

`--subjects` picks the columns (`qiyas`, `number`, and what you add), `--types` the stored types (the seventeen plain
ones, `bool`, `q8_0`, `q4_0`, `q4_k`, `q6_k`), `--ops` the rows, `--baseline` a named earlier results file to
compare against. `--repeat N` repeats the matmul rows N times and raises the repetitions of the gates that once
failed intermittently (the block transpose; 20 by default): a stress for schedule-dependent faults.

`--shapes odd` runs every row at the nearest primes below its regular sizes (4093 × 4091 instead of 4096², 16M − 5,
…) so no dimension ends on a tile, a word or a block: the padded, value-wise and segmented paths are what is timed.
`--report` prints, after the gates, the facts the run's kernels were chosen by (the team and vector widths, the dot
lanes the runtime reports, the tensor kinds the JIT accepted, the sliced-float64 proof and every measured decision),
and the buffers: the pool's and the arena's rent/return cost and a product's second call in an arena scope (its pool
rents and driver allocations), the CPU product workspace's the same.

`--ops map` adds a row per element-wise formula (`map:<name>`, 16M values); `reduce` adds the reductions (`sum:whole`,
`sum:rows`, `sum:cols`, `sumsq:rows`, `max:whole`, `max:rows`, `max:cols`, `min:rows`, `argmax:rows`), the row-vector
broadcasts (`maprow:add`, `maprow:mul`), `transpose` (4096 × 4096) and `gemv`, `gemv_t` (8192 × 8192); `all` is the four
core operations plus every formula, every reduction and every row operation; `rows` adds the row family (`row:<name>`
for each of `RowOps.All` — log_softmax, the softmax backward passes, rms_norm_affine and its backward, layer_norm and
its backward, weight_norm, rope, scale_mask_softmax, add_rms_norm, add_layer_norm — and `norm_params_bwd:rms|layer`,
4096 × 4096, float types); `layout` adds the layout operations (`layout:<name>` for each of `LayoutArgs.Names`: strided,
one_hot, gather_rows, gather_cols, scatter_add_rows, im2col, col2im, interpolate_nearest, interpolate_bilinear, shift).
`products` adds the products on views (`matmul_nt`, `matmul_tn`, `matmul_tt` at 4096³, `matmul_batched`: 16 × 1024³,
`matmul_bwd`: dA and dB at 4096³, `matmul_blocks`: A·Bᵀ of two Q8_0 / Q4_0 operands by the block-scaled product alone);
`attention` adds `attention`, `attention_causal` (16 heads, 2048 × 2048, d = 128),
`attention_fused`, `attention_fused_causal` (the same shapes, the fused rows alone, for comparing with the measured
choice), `attention_decode` (32 heads, one query over 4096 keys) and `attention_bwd` (8 heads of 1024). The decisions
taken from measurements during a run (attention's fused-or-composed choice per shape and type, among others) are printed
at its end with `--report`. A single name works too
(`--ops map:erf,sum:rows,row:rope,layout:im2col,attention_decode`); `optimizers` adds `optimizer:sgd_momentum`,
`optimizer:adam`, `optimizer:adamw` over 16M parameters (the moments in lane storage; each run restores p, m, v first,
named in the lane column); `conv` adds `conv` and `conv_bwd` (a 64 × 128 × 128 image by 128 filters of 3 × 3); the
pools and the depthwise convolution are layout operations (`layout:maxpool`, `layout:avgpool`, their `_bwd`,
`layout:adaptive_maxpool`, `layout:adaptive_avgpool`, `layout:depthwise_conv`); `recurrent` adds `lstm_step`, `gru_step`
and their `_bwd` (a batch of 1024, input 1024, hidden 1024), `lstm_seq`, `lstm_seq_bwd`, `gru_seq` and `gru_seq_bwd` (16 steps of 256 × 512 → 512); `routing` adds `topk_rows`, `pick_rows`, `sample_rows`
(4096 rows of 32768), `route` (1M indices into 64 buckets), `route_wide` (1M into 65 536), `cross_entropy` (the loss by log_softmax, pick, neg), and
`ctc`, `ctc_bwd` (256 frames of 32 sequences over 1024 classes, 64 labels each; the loss scored, then the gradient).
`COVERAGE.md` lists every operation with its lane rule on each device. The block formats and bool run through every
family: a block format's lane is float32, read a value at a time anywhere (both devices decode by index) and written a
block at a time (the CPU re-encodes the blocks a store touches, tasks never sharing a block; the device computes
value-wise stores in float32 storage and encodes by blocks — all four formats have device encoders, byte for byte the
CPU's); bool is a one-bit packed type held to 0/1 (division is n/a). Products with a Q8_0 or Q4_0 operand take the
block-scaled integer dot (`s8dp blocks`).

Every implementation in the assembly that implements `IBenchSubject` (`src/Bench/Contract.cs`) is found by reflection
and becomes a column: `qiyas`, `number` (one implementation over .NET's `INumberBase<T>`, for the types .NET has a
`T` for), and whatever you add. The harness makes the inputs (stored bytes of each type, from Qiyas's proven
conversions, so every subject sees the same bits), times `Run` (medians over an adaptive number of runs), reads the
output after `Collect`, and scores it against float64 math on the same inputs. Results go to `results-qiyas-<devices>.md`.

To add an implementation: one class with a parameterless constructor implementing the four methods (and `Map`, `Reduce`,
`MapRow`, `Transpose`, `Gemv` for the other families, which default to `null`), each returning `null` for anything it does not do (shown as —),
never an approximation. Inputs and outputs are stored bytes of the
named type; `Run` must include your device synchronization; `Collect` writes the output; `Lane` says what you computed in.

`profile`, `gpuprofile` and `team` are Qiyas's own diagnostics (matmul phases on each device; the thread team's dispatch cost).

## Building

`dotnet build -c Release` (net10.0). The gates run at the start of every bench invocation; the regression gate is
`python docs/wip/compare.py docs/wip/baseline.md results-qiyas-cuda-cpu.md` after a full `cuda,cpu` run (a cell more
than 5 % slower, or an error that grew, is a regression). `sh guard.sh` proves that no file in the repository names anything of the library
Qiyas is measured against; run it before every build. That library's column (`lib`) exists only in the author's local
checkout (files listed in `.gitignore`); a clone runs the other subjects.

## Layout

| file | what |
|---|---|
| `src/Numbers.cs` | the contract: `IQiyasNumber<TSelf>`, lanes, dot lanes, the visitors; exp on vectors |
| `src/Types.cs` | the seventeen stored types and their lane rules |
| `src/Blocks.cs`, `src/Ptx.Blocks.cs`, `src/BlockProducts.cs` | the block formats Q8_0, Q4_0, Q4_K, Q6_K and bool: the codecs (CPU and device encoders, a thread per block), the block-scaled integer gemv |
| `src/MatMulBlocks.cs` | the block-scaled tiled product A·Bᵀ of Q8_0 / Q4_0 on the CPU (the s8 dot lane) and the device (the s8 tensor instruction at k = 32), and its measured choice against the decoded route |
| `src/Report.cs` | `--report`: the facts and measured decisions of each device, the buffers' costs and steady state |
| `src/Arena.cs` | the device's buffer arena: a scope's buffers bumped in one region, released from the top, the region grown between calls |
| `src/Workspace.cs` | the CPU product's per-thread workspace (packed B, loaded A / B, C lanes, A panels) |
| `docs/wip/` | the regression comparer (`compare.py`), the baseline run it compares against (`baseline.md`), the README Results generator (`readme_results.py`) |
| `src/Convert.cs` | every load and store on vectors; `Widths`, any lane to any lane |
| `src/Kernels.cs` | the CPU kernels: add, product (GotoBLAS-shaped, dot lanes in groups), softmax, rms_norm |
| `src/Map.cs`, `src/Ptx.Map.cs` | the element-wise family: the catalogue of formulas (with float64 references), the CPU kernel, the device kernel |
| `src/Rows.cs`, `src/Ptx.Rows.cs`, `src/GpuOps.Rows.cs`, `src/Bench/RowInputs.cs` | the row family: the catalogue (references, error scales), the CPU kernels, the device kernels, the inputs and tolerance shared by the checks and the bench |
| `src/Routing.cs`, `src/Ptx.Routing.cs`, `src/GpuOps.Routing.cs` | top-k, pick, sampling, the route; cross-entropy's backward is a row operation |
| `src/Ptx.Radix.cs`, `src/GpuOps.Radix.cs` | the device's ordering by target for scatter-add and the route: counts, the multi-block exclusive scan, the stable LSD radix sort |
| `src/Ctc.cs`, `src/Ptx.Ctc.cs`, `src/GpuOps.Ctc.cs` | ctc: the arguments and the float64 reference, the CPU kernel, the device kernels (rows, then a block per sequence) |
| `src/Recurrent.cs`, `src/Ptx.Recurrent.cs`, `src/GpuOps.Recurrent.cs`, `src/Bench/RecurrentCases.cs` | LSTM and GRU cells and steps, forward and backward |
| `src/Conv.cs`, `src/GpuOps.Conv.cs` | convolution: the forward fused into the product's packing (implicit GEMM: the image read through the index map; on the device's plain lanes a measured choice against im2col), the backward as im2col / product / col2im compositions |
| `src/Optimizers.cs`, `src/Ptx.Optimizers.cs`, `src/GpuOps.Optimizers.cs` | optimizer steps: sgd with momentum, adam, adamw |
| `src/Attention.cs`, `src/Ptx.Attention.cs`, `src/GpuOps.Attention.cs` | attention: the measured choice, the composition in lane storage, the fused decode kernel, the backward composition; the lane-storage conversions |
| `src/AttentionRows.cs`, `src/Ptx.AttentionRows.cs` | attention's fused rows (flash: keys in tiles, the softmax online, no scores in memory) on the CPU and the device |
| `src/Products.cs`, `src/GpuOps.Products.cs` | products on views (transposed operands, batches) and the matmul backward: the product's packing reads the operands where they are stored (`Kernels.Operands`; the device's tensor and dot routes), otherwise through the strided copy |
| `src/Layout.cs`, `src/Ptx.Layout.cs`, `src/GpuOps.Layout.cs`, `src/Bench/LayoutCases.cs` | layout and indexing: the references, the CPU kernels, the device kernels, the gate's and the bench's cases |
| `src/Reduce.cs`, `src/Ptx.Reduce.cs` | reductions (sum, sum of squares, max, min, argmax: whole, rows, columns), row-vector broadcasts, transpose, gemv: the CPU kernels, the device kernels |
| `src/Functions.cs` | log, sin, cos, expm1, tanh, sigmoid, erf, erfc on vectors: coefficient tables shared with the device emitter |
| `src/Team.cs` | the CPU thread team (jobs closed on completion, so a late worker never takes a chunk under the next job's half-written description) |
| `src/Gpu.cs`, `src/Ptx.cs`, `src/GpuOps.cs` | the CUDA side: driver binding, PTX generation, the operations |
| `src/Checks.cs` | the proofs |
| `src/CpuGeneric.cs`, `src/GenericPtx.cs` | the `INumberBase<T>` version (the `number` column) |
| `src/Bench/` | the bench contract, the type registry, the subjects |
| `Program.cs` | the harness |

## Results

Measured on one machine, 2026-11-06 (item 4's `cuda,cpu --report --ops all` run of 19:11, every family, stopped
during the CPU's last two types: the CPU rows of q6_k and bool below are the earlier run's; compared cell by cell
against item 4a's baseline by `docs/wip/compare.py`: no error grown, 13 of 122 cells slower, listed in
`COVERAGE.md`'s "Known slow" for re-measurement): Intel Core Ultra 9 285 (24 logical processors, 256-bit
vectors, the s8/u8/s16/u16 dot lanes — as the runtime reports), 64 GB; NVIDIA GeForce RTX 5070 Ti (70 multiprocessors,
compute capability 12.0, 16 GB, driver 610.88; PTX target sm_80); Windows 11, .NET 10.0.12. The full table with every
subject's lane and error is `results-qiyas-cuda-cpu.md`; the three columns are the library Qiyas is measured against
(`lib`, present only in the author's checkout), the `INumberBase<T>` version (`number`, — where .NET has no such type)
and Qiyas. Times are medians in ms. Every subject's error against float64 math (products in double-double) is in the
full table (`docs/wip/baseline.md`); Qiyas's is equal to or below the library's on every row the two share except
eight CUDA rows of item 4's run: float32 sin, cos, gelu, silu and add_rms_norm (up to 1.4× the library's), float16
softmax (9.54e-7 against 9.53e-7), and sum:rows float16 and bfloat16 (up to 1.3×). The CUDA whole sums of the narrow
float types now sum each thread's values compensated and join the partials in float64 (float32 4.87e-5 against the
library's 1.95e-4, float8_e5m2 0 against 6.1e-5; a gate holds them within one f32 step of the exact sum). It is exact (0) wherever the lane is exact — the
CUDA float64 product now takes the sliced 8-bit tensor route (`f64x`, proven exact, 4.18 ms against 23.1 in the
float64 lane). The block formats and bool have no lib or number column (—): the lib has no such types.

**CUDA (2048³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 124.683 / 1.000 / **0.521** / 239.31× | 31.448 / 23.081 / **4.175** / 7.53× | 79.646 / 0.396 / **1.746** / 45.62× | 89.619 / 0.372 / **0.552** / 162.35× |
| s32 |  |  |  |  |
| float32 | 0.262 / 0.262 / **0.263** / 1.00× | 1.042 / 0.839 / **0.685** / 1.52× | 0.183 / 0.181 / **0.181** / 1.01× | 0.184 / 0.182 / **0.180** / 1.02× |
| float16 | 0.661 / 0.139 / **0.123** / 5.37× | 1.028 / 0.657 / **0.243** / 4.23× | 0.443 / 0.090 / **0.084** / 5.27× | 0.442 / 0.088 / **0.078** / 5.67× |
| bfloat16 | 0.662 / — / **0.126** / 5.25× | 1.018 / — / **0.244** / 4.17× | 0.444 / — / **0.082** / 5.41× | 0.443 / — / **0.081** / 5.47× |
| float8_e4m3 | 494.600 / — / **0.050** / 9892.00× | 130.558 / — / **0.259** / 504.08× | 107.532 / — / **0.067** / 1604.96× | 444.869 / — / **0.041** / 10850.46× |
| float8_e5m2 | 472.262 / — / **1.107** / 426.61× | 110.644 / — / **0.261** / 423.92× | 295.054 / — / **0.069** / 4276.14× | 340.592 / — / **0.047** / 7246.64× |
| int64 | 117.636 / 0.520 / **0.520** / 226.22× | 29.637 / 4.427 / **0.356** / 83.25× |  |  |
| int32 | 0.787 / 0.262 / **0.263** / 2.99× | 1.141 / 1.106 / **0.209** / 5.46× |  |  |
| int16 | 72.698 / 0.259 / **0.224** / 324.54× | 18.072 / 1.591 / **0.175** / 103.27× |  |  |
| int8 | 64.100 / 0.098 / **0.035** / 1831.43× | 21.436 / 1.554 / **0.094** / 228.04× |  |  |
| int4 | 78.638 / — / **0.018** / 4368.78× | 44.193 / — / **0.118** / 374.52× |  |  |
| int2 | 75.790 / — / **0.019** / 3988.95× | 19.817 / — / **0.111** / 178.53× |  |  |
| uint64 | 134.628 / 0.999 / **0.520** / 258.90× | 35.524 / 4.370 / **0.355** / 100.07× |  |  |
| uint32 | 122.715 / 0.502 / **0.263** / 466.60× | 23.266 / 1.091 / **0.211** / 110.27× |  |  |
| uint16 | 74.622 / 0.140 / **0.124** / 601.79× | 21.982 / 1.330 / **0.170** / 129.31× |  |  |
| uint8 | 64.427 / 0.107 / **0.042** / 1533.98× | 16.605 / 1.286 / **0.093** / 178.55× |  |  |
| uint2 | 73.139 / — / **0.017** / 4302.29× | 17.703 / — / **0.109** / 162.41× |  |  |
| q8_0 | — / — / **0.983** / — | — / — / **0.860** / — | — / — / **0.782** / — | — / — / **0.776** / — |
| q4_0 | — / — / **0.660** / — | — / — / **0.780** / — | — / — / **0.460** / — | — / — / **0.457** / — |
| q4_k | — / — / **1.196** / — | — / — / **0.871** / — | — / — / **0.988** / — | — / — / **0.988** / — |
| q6_k | — / — / **1.402** / — | — / — / **0.957** / — | — / — / **1.192** / — | — / — / **1.193** / — |
| bool | — / — / **0.017** / — | — / — / **0.125** / — |  |  |

**CPU (1024³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 6.307 / 6.883 / **6.724** / 0.94× | 5.783 / 15.855 / **3.364** / 1.72× | 31.985 / 6.392 / **4.812** / 6.65× | 31.035 / 4.078 / **4.664** / 6.65× |
| s32 |  |  |  |  |
| float32 | 3.240 / 3.042 / **2.936** / 1.10× | 2.703 / 4.105 / **1.676** / 1.61× | 2.502 / 4.244 / **1.806** / 1.39× | 2.256 / 1.860 / **1.649** / 1.37× |
| float16 | 2.908 / 7.633 / **1.242** / 2.34× | 7.070 / 881.784 / **1.731** / 4.08× | 44.974 / 58.020 / **2.219** / 20.27× | 44.632 / 25.095 / **1.231** / 36.26× |
| bfloat16 | 1.137 / — / **0.821** / 1.38× | 4.560 / — / **1.176** / 3.88× | 23.581 / — / **1.768** / 13.34× | 30.452 / — / **0.404** / 75.38× |
| float8_e4m3 | 28.600 / — / **1.968** / 14.53× | 35.141 / — / **1.753** / 20.05× | 90.060 / — / **2.565** / 35.11× | 429.597 / — / **1.342** / 320.12× |
| float8_e5m2 | 17.993 / — / **1.972** / 9.12× | 31.074 / — / **1.781** / 17.45× | 279.161 / — / **3.231** / 86.40× | 314.909 / — / **1.711** / 184.05× |
| int64 | 25.096 / 6.621 / **6.691** / 3.75× | 6.394 / 32.382 / **0.601** / 10.64× |  |  |
| int32 | 20.738 / 3.034 / **2.913** / 7.12× | 6.005 / 5.932 / **0.573** / 10.48× |  |  |
| int16 | 1.397 / 1.047 / **0.831** / 1.68× | 5.175 / 1.845 / **0.408** / 12.68× |  |  |
| int8 | 1.543 / 0.420 / **0.144** / 10.72× | 5.905 / 2.205 / **0.393** / 15.03× |  |  |
| int4 | 2.872 / — / **0.107** / 26.84× | 11.391 / — / **0.351** / 32.45× |  |  |
| int2 | 2.357 / — / **0.182** / 12.95× | 6.646 / — / **0.360** / 18.46× |  |  |
| uint64 | 12.511 / 7.053 / **6.979** / 1.79× | 6.756 / 33.098 / **0.663** / 10.19× |  |  |
| uint32 | 12.454 / 3.022 / **2.868** / 4.34× | 6.307 / 6.386 / **0.504** / 12.51× |  |  |
| uint16 | 1.441 / 1.241 / **0.791** / 1.82× | 5.436 / 2.004 / **0.356** / 15.27× |  |  |
| uint8 | 1.902 / 0.478 / **0.127** / 14.98× | 6.172 / 4.583 / **0.485** / 12.73× |  |  |
| uint2 | 2.731 / — / **0.244** / 11.19× | 6.749 / — / **0.525** / 12.86× |  |  |
| q8_0 | — / — / **6.140** / — | — / — / **9.056** / — | — / — / **5.664** / — | — / — / **5.398** / — |
| q4_0 | — / — / **3.530** / — | — / — / **8.086** / — | — / — / **4.072** / — | — / — / **2.661** / — |
| q4_k | — / — / **5.031** / — | — / — / **8.777** / — |  |  |
| q6_k (earlier run) | — / — / **10.020** / — | — / — / **10.080** / — | — / — / **11.789** / — | — / — / **7.230** / — |
| bool (earlier run) | — / — / **4.882** / — | — / — / **2.150** / — |  |  |

**Every family** — rows measured (operation × type) and, where the lib has the row, the geometric mean of lib ÷ qiyas

| family | operations | CUDA rows | CUDA vs lib | CPU rows | CPU vs lib |
|---|---:|---:|---|---:|---|
| core | 4 | 64 | 103.31× (ahead on 45 of 46) | 56 | 9.09× (ahead on 45 of 46) |
| map (element-wise formulas) | 60 | 899 | 197.58× (ahead on 285 of 304) | 815 | — |
| reductions, broadcasts, transpose, gemv | 15 | 330 | 172.68× (ahead on 84 of 85) | 300 | — |
| rows | 15 | 150 | 63.38× (ahead on 42 of 42) | 135 | — |
| layout and indexing | 20 | 380 | — | 345 | — |
| products on views | 6 | 112 | — | 102 | — |
| attention | 6 | 60 | — | 54 | — |
| optimizers | 3 | 30 | — | 27 | — |
| convolution | 2 | 32 | — | 29 | — |
| recurrent | 8 | 72 | — | 65 | — |
| routing, losses | 8 | 86 | — | 72 | — |

Where Qiyas is not ahead: the memory-bound adds of 32- and 64-bit types (the same bytes moved, within a few percent).
The CUDA float64 product is exact where the library computes it in float32 (its error 1.6e-4), through the sliced
8-bit tensor route where the device measures it faster than its float64 lane.

The core tables above and the family table are item 4's run (`docs/wip/readme_results.py`); the regression baseline
`docs/wip/baseline.md` is that run's every row (4216: every CUDA row, the CPU's up to q4_k), so every family is compared
from now on. The lib has rows for the core ops, the element-wise formulas, the reductions and the row operations on
CUDA, and the core ops on the CPU; elsewhere the family table has no lib column (—). `COVERAGE.md`'s "Known slow"
names the measured cells that are correct but below what the device allows.

## Open

What the gates have not confirmed, or is not done, is in `COVERAGE.md` ("State" for the order, **todo** in the rows):
the known-slow
cells (among them the tensor lanes inside the fused attention kernel, which runs in the FMA lane). The README claims nothing beyond what the gates and the regression
comparison have confirmed.
