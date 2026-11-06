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
  laid out by hand from the formats' specifications; the products on the device under every tensor kind the device
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
  steps with their backward passes, LSTM sequences with theirs, GRU sequences; top-k, pick, sampling, routing, cross-entropy and ctc (loss and
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
lanes the runtime reports, the tensor kinds the JIT accepted, the sliced-float64 proof and every measured decision).

`--ops map` adds a row per element-wise formula (`map:<name>`, 16M values); `reduce` adds the reductions (`sum:whole`,
`sum:rows`, `sum:cols`, `sumsq:rows`, `max:whole`, `max:rows`, `max:cols`, `min:rows`, `argmax:rows`), the row-vector
broadcasts (`maprow:add`, `maprow:mul`), `transpose` (4096 × 4096) and `gemv`, `gemv_t` (8192 × 8192); `all` is the four
core operations plus every formula, every reduction and every row operation; `rows` adds the row family (`row:<name>`
for each of `RowOps.All` — log_softmax, the softmax backward passes, rms_norm_affine and its backward, layer_norm and
its backward, weight_norm, rope, scale_mask_softmax, add_rms_norm, add_layer_norm — and `norm_params_bwd:rms|layer`,
4096 × 4096, float types); `layout` adds the layout operations (`layout:<name>` for each of `LayoutArgs.Names`: strided,
one_hot, gather_rows, gather_cols, scatter_add_rows, im2col, col2im, interpolate_nearest, interpolate_bilinear, shift).
`products` adds the products on views (`matmul_nt`, `matmul_tn`, `matmul_tt` at 4096³, `matmul_batched`: 16 × 1024³,
`matmul_bwd`: dA and dB at 4096³); `attention` adds `attention`, `attention_causal` (16 heads, 2048 × 2048, d = 128),
`attention_fused`, `attention_fused_causal` (the same shapes, the fused rows alone, for comparing with the measured
choice), `attention_decode` (32 heads, one query over 4096 keys) and `attention_bwd` (8 heads of 1024). The decisions
taken from measurements during a run (attention's fused-or-composed choice per shape and type, among others) are printed
at its end with `--report`. A single name works too
(`--ops map:erf,sum:rows,row:rope,layout:im2col,attention_decode`); `optimizers` adds `optimizer:sgd_momentum`,
`optimizer:adam`, `optimizer:adamw` over 16M parameters (the moments in lane storage; each run restores p, m, v first,
named in the lane column); `conv` adds `conv` and `conv_bwd` (a 64 × 128 × 128 image by 128 filters of 3 × 3); the
pools and the depthwise convolution are layout operations (`layout:maxpool`, `layout:avgpool`, their `_bwd`,
`layout:adaptive_maxpool`, `layout:adaptive_avgpool`, `layout:depthwise_conv`); `recurrent` adds `lstm_step`, `gru_step`
and their `_bwd` (a batch of 1024, input 1024, hidden 1024), `lstm_seq`, `lstm_seq_bwd` and `gru_seq` (16 steps of 256 × 512 → 512); `routing` adds `topk_rows`, `pick_rows`, `sample_rows`
(4096 rows of 32768), `route` (1M indices into 64 buckets), `cross_entropy` (the loss by log_softmax, pick, neg), and
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
| `src/Report.cs` | `--report`: the facts and measured decisions of each device, the pool's rent/return cost |
| `docs/wip/` | the regression comparer (`compare.py`), the baseline run it compares against (`baseline.md`), the README Results generator (`readme_results.py`) |
| `src/Convert.cs` | every load and store on vectors; `Widths`, any lane to any lane |
| `src/Kernels.cs` | the CPU kernels: add, product (GotoBLAS-shaped, dot lanes in groups), softmax, rms_norm |
| `src/Map.cs`, `src/Ptx.Map.cs` | the element-wise family: the catalogue of formulas (with float64 references), the CPU kernel, the device kernel |
| `src/Rows.cs`, `src/Ptx.Rows.cs`, `src/GpuOps.Rows.cs`, `src/Bench/RowInputs.cs` | the row family: the catalogue (references, error scales), the CPU kernels, the device kernels, the inputs and tolerance shared by the checks and the bench |
| `src/Routing.cs`, `src/Ptx.Routing.cs`, `src/GpuOps.Routing.cs` | top-k, pick, sampling, the route; cross-entropy's backward is a row operation |
| `src/Ctc.cs`, `src/Ptx.Ctc.cs`, `src/GpuOps.Ctc.cs` | ctc: the arguments and the float64 reference, the CPU kernel, the device kernels (rows, then a block per sequence) |
| `src/Recurrent.cs`, `src/Ptx.Recurrent.cs`, `src/GpuOps.Recurrent.cs`, `src/Bench/RecurrentCases.cs` | LSTM and GRU cells and steps, forward and backward |
| `src/Conv.cs`, `src/GpuOps.Conv.cs` | convolution forward and backward as im2col / product / col2im compositions |
| `src/Optimizers.cs`, `src/Ptx.Optimizers.cs`, `src/GpuOps.Optimizers.cs` | optimizer steps: sgd with momentum, adam, adamw |
| `src/Attention.cs`, `src/Ptx.Attention.cs`, `src/GpuOps.Attention.cs` | attention: the measured choice, the composition in lane storage, the fused decode kernel, the backward composition; the lane-storage conversions |
| `src/AttentionRows.cs`, `src/Ptx.AttentionRows.cs` | attention's fused rows (flash: keys in tiles, the softmax online, no scores in memory) on the CPU and the device |
| `src/Products.cs`, `src/GpuOps.Products.cs` | products on views (transposed operands, batches) and the matmul backward, through the strided copy and the product kernel |
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

Measured on one machine, 2026-11-06 (the full `cuda,cpu --report` run of 12:38, compared cell by cell against the
baseline run of 00:51 in `docs/wip/baseline.md` by `docs/wip/compare.py`: no cell slower beyond this laptop's swing,
no error grown): Intel Core Ultra 9 285 (24 logical processors, 256-bit
vectors, the s8/u8/s16/u16 dot lanes — as the runtime reports), 64 GB; NVIDIA GeForce RTX 5070 Ti (70 multiprocessors,
compute capability 12.0, 16 GB, driver 610.88; PTX target sm_80); Windows 11, .NET 10.0.12. The full table with every
subject's lane and error is `results-qiyas-cuda-cpu.md`; the three columns are the library Qiyas is measured against
(`lib`, present only in the author's checkout), the `INumberBase<T>` version (`number`, — where .NET has no such type)
and Qiyas. Times are medians in ms. Every subject's error against float64 math (products in double-double) is in the
full table; Qiyas's is equal to or below the library's on every row, and exact (0) wherever the lane is exact — the
CUDA float64 product now takes the sliced 8-bit tensor route (`f64x`, proven exact, 4.19 ms against 22.7 in the
float64 lane). The block formats and bool have no lib or number column (—): the lib has no such types.

**CUDA (2048³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 108.071 / 1.000 / **0.520** / 207.83× | 30.089 / 22.694 / **4.188** / 7.18× | 82.790 / 0.396 / **1.745** / 47.44× | 81.723 / 0.372 / **0.554** / 147.51× |
| float32 | 0.263 / 0.263 / **0.263** / 1.00× | 0.990 / 0.843 / **0.688** / 1.44× | 0.181 / 0.182 / **0.179** / 1.01× | 0.184 / 0.183 / **0.180** / 1.02× |
| float16 | 0.662 / 0.139 / **0.126** / 5.25× | 1.025 / 0.661 / **0.245** / 4.18× | 0.443 / 0.092 / **0.083** / 5.34× | 0.443 / 0.089 / **0.080** / 5.54× |
| bfloat16 | 0.659 / — / **0.125** / 5.27× | 1.019 / — / **0.245** / 4.16× | 0.444 / — / **0.082** / 5.41× | 0.442 / — / **0.079** / 5.59× |
| float8_e4m3 | 489.377 / — / **0.051** / 9595.63× | 131.827 / — / **0.260** / 507.03× | 105.340 / — / **0.069** / 1526.67× | 444.929 / — / **0.046** / 9672.37× |
| float8_e5m2 | 475.449 / — / **0.877** / 542.13× | 116.661 / — / **0.264** / 441.90× | 296.343 / — / **0.070** / 4233.47× | 361.874 / — / **0.050** / 7237.48× |
| int64 | 114.436 / 0.520 / **0.520** / 220.07× | 29.298 / 4.449 / **0.376** / 77.92× |  |  |
| int32 | 0.788 / 0.263 / **0.263** / 3.00× | 1.148 / 1.099 / **0.214** / 5.36× |  |  |
| int16 | 66.940 / 0.139 / **0.125** / 535.52× | 18.546 / 1.592 / **0.174** / 106.59× |  |  |
| int8 | 62.942 / 0.078 / **0.023** / 2736.61× | 21.669 / 1.554 / **0.094** / 230.52× |  |  |
| int4 | 78.370 / — / **0.018** / 4353.89× | 38.037 / — / **0.103** / 369.29× |  |  |
| int2 | 77.427 / — / **0.018** / 4301.50× | 23.917 / — / **0.113** / 211.65× |  |  |
| uint64 | 125.069 / 0.522 / **0.520** / 240.52× | 32.828 / 4.386 / **0.361** / 90.94× |  |  |
| uint32 | 103.178 / 0.264 / **0.263** / 392.31× | 23.652 / 1.101 / **0.214** / 110.52× |  |  |
| uint16 | 67.499 / 0.139 / **0.126** / 535.71× | 17.226 / 1.335 / **0.178** / 96.78× |  |  |
| uint8 | 64.178 / 0.074 / **0.023** / 2790.35× | 16.759 / 1.301 / **0.094** / 178.29× |  |  |
| uint2 | 71.151 / — / **0.017** / 4185.35× | 18.032 / — / **0.114** / 158.18× |  |  |
| q8_0 | — / — / **1.409** / — | — / — / **0.864** / — | — / — / **0.779** / — | — / — / **0.775** / — |
| q4_0 | — / — / **0.660** / — | — / — / **0.788** / — | — / — / **0.457** / — | — / — / **0.454** / — |
| q4_k | — / — / **1.191** / — | — / — / **0.868** / — | — / — / **0.988** / — | — / — / **0.984** / — |
| q6_k | — / — / **1.401** / — | — / — / **0.958** / — | — / — / **1.190** / — | — / — / **1.193** / — |
| bool | — / — / **0.017** / — | — / — / **0.122** / — |  |  |

**CPU (1024³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 6.471 / 6.750 / **6.629** / 0.98× | 5.946 / 14.260 / **3.114** / 1.91× | 31.524 / 6.596 / **4.857** / 6.49× | 30.759 / 4.253 / **4.640** / 6.63× |
| float32 | 2.965 / 2.957 / **2.871** / 1.03× | 1.851 / 4.715 / **1.769** / 1.05× | 3.282 / 4.028 / **1.876** / 1.75× | 2.481 / 2.137 / **1.635** / 1.52× |
| float16 | 3.085 / 7.432 / **1.738** / 1.78× | 6.819 / 910.367 / **1.741** / 3.92× | 46.245 / 49.926 / **2.218** / 20.85× | 45.339 / 20.933 / **0.825** / 54.96× |
| bfloat16 | 1.114 / — / **0.807** / 1.38× | 4.882 / — / **1.207** / 4.04× | 24.582 / — / **1.765** / 13.93× | 23.979 / — / **0.405** / 59.21× |
| float8_e4m3 | 26.617 / — / **2.120** / 12.56× | 27.486 / — / **1.762** / 15.60× | 92.300 / — / **3.075** / 30.02× | 397.020 / — / **1.612** / 246.29× |
| float8_e5m2 | 22.414 / — / **2.358** / 9.51× | 24.259 / — / **1.793** / 13.53× | 280.242 / — / **3.885** / 72.13× | 317.788 / — / **1.515** / 209.76× |
| int64 | 25.181 / 6.761 / **6.569** / 3.83× | 6.275 / 21.757 / **0.887** / 7.07× |  |  |
| int32 | 25.915 / 3.092 / **2.943** / 8.81× | 5.700 / 5.119 / **0.513** / 11.11× |  |  |
| int16 | 1.610 / 1.323 / **1.184** / 1.36× | 5.307 / 2.096 / **0.531** / 9.99× |  |  |
| int8 | 1.833 / 0.338 / **0.129** / 14.21× | 6.091 / 2.981 / **0.529** / 11.51× |  |  |
| int4 | 2.926 / — / **0.144** / 20.32× | 12.123 / — / **0.484** / 25.05× |  |  |
| int2 | 2.865 / — / **0.284** / 10.09× | 6.908 / — / **0.471** / 14.67× |  |  |
| uint64 | 10.966 / 6.742 / **6.656** / 1.65× | 6.664 / 21.610 / **0.896** / 7.44× |  |  |
| uint32 | 12.747 / 3.093 / **2.846** / 4.48× | 6.038 / 4.366 / **0.542** / 11.14× |  |  |
| uint16 | 1.557 / 1.425 / **0.885** / 1.76× | 5.437 / 3.124 / **0.531** / 10.24× |  |  |
| uint8 | 1.612 / 0.398 / **0.139** / 11.60× | 5.828 / 2.116 / **0.399** / 14.61× |  |  |
| uint2 | 2.673 / — / **0.160** / 16.71× | 6.525 / — / **0.486** / 13.43× |  |  |
| q8_0 | — / — / **6.415** / — | — / — / **8.297** / — | — / — / **7.484** / — | — / — / **4.521** / — |
| q4_0 | — / — / **4.287** / — | — / — / **8.595** / — | — / — / **5.381** / — | — / — / **3.241** / — |
| q4_k | — / — / **5.039** / — | — / — / **9.111** / — | — / — / **6.232** / — | — / — / **4.498** / — |
| q6_k | — / — / **10.020** / — | — / — / **10.080** / — | — / — / **11.789** / — | — / — / **7.230** / — |
| bool | — / — / **4.882** / — | — / — / **2.150** / — |  |  |

Where Qiyas is not ahead: the memory-bound adds of 32- and 64-bit types (the same bytes moved, within a few percent).
The CUDA float64 product is exact where the library computes it in float32 (its error 1.6e-4), through the sliced
8-bit tensor route where the device measures it faster than its float64 lane.

The tables above are the four core rows of the full `cuda,cpu` run of 12:38, which the regression gate compares
against (`docs/wip/compare_readme.py` compares a run with these tables); the regression baseline `docs/wip/baseline.md` is the
full `cuda,cpu --report` run of item 4a (128 rows, every type); the other families' rows, the block formats and bool are in `results-qiyas-cuda-cpu.md` of
the latest run, and `COVERAGE.md`'s "Known slow" names the measured cells that are correct but below what the device
allows.

## Open

What the gates have not confirmed, or is not done, is in `COVERAGE.md` ("State" for the order, **todo** in the rows):
the block-scaled tiled product, the transposition fused into the product's packing, the
radix step of scatter-add, a fused direct convolution, a bump-allocated scope past the buffer pool, and the known-slow
cells (among them the tensor lanes inside the fused attention kernel, which runs in the FMA lane). The README claims nothing beyond what the gates and the regression
comparison have confirmed.
