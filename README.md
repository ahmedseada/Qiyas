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
  laid out by hand from the formats' specifications. The bench refuses to time while any gate fails.
- **The operation surface.** `COVERAGE.md` is the list, one row per operation with the lane rule on each device:
  element-wise formulas and their backward passes; reductions, broadcasts, transposes, gemv; the row family
  (softmax, log_softmax, the norms and their backward passes, rope, masked softmax, cross-entropy's backward); layout
  and indexing (strided copies, one-hot, gathers, scatter-add, im2col/col2im, interpolation, shift, pad, the pools and
  their backward, depthwise convolution); products on views (transposed, batched) and the matmul backward; attention
  (rows, causal, spans, decode, backward); convolution forward and backward; the optimizer steps; LSTM and GRU cells,
  steps and sequences with their backward passes; top-k, pick, sampling, routing; and the block formats and bool
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
`attention_decode` (32 heads, one query over 4096 keys) and `attention_bwd` (8 heads of 1024). A single name works too
(`--ops map:erf,sum:rows,row:rope,layout:im2col,attention_decode`); `optimizers` adds `optimizer:sgd_momentum`,
`optimizer:adam`, `optimizer:adamw` over 16M parameters (the moments in lane storage; each run restores p, m, v first,
named in the lane column); `conv` adds `conv` and `conv_bwd` (a 64 × 128 × 128 image by 128 filters of 3 × 3); the
pools and the depthwise convolution are layout operations (`layout:maxpool`, `layout:avgpool`, their `_bwd`,
`layout:adaptive_maxpool`, `layout:adaptive_avgpool`, `layout:depthwise_conv`); `recurrent` adds `lstm_step`, `gru_step`
and their `_bwd` (a batch of 1024, input 1024, hidden 1024), `lstm_seq` and `lstm_seq_bwd` (16 steps of 256 × 512 → 512); `routing` adds `topk_rows`, `pick_rows`, `sample_rows`
(4096 rows of 32768), `route` (1M indices into 64 buckets) and `cross_entropy` (the loss by log_softmax, pick, neg).
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
| `docs/wip/` | the regression comparer (`compare.py`) and the baseline run it compares against (`baseline.md`) |
| `src/Convert.cs` | every load and store on vectors; `Widths`, any lane to any lane |
| `src/Kernels.cs` | the CPU kernels: add, product (GotoBLAS-shaped, dot lanes in groups), softmax, rms_norm |
| `src/Map.cs`, `src/Ptx.Map.cs` | the element-wise family: the catalogue of formulas (with float64 references), the CPU kernel, the device kernel |
| `src/Rows.cs`, `src/Ptx.Rows.cs`, `src/GpuOps.Rows.cs`, `src/Bench/RowInputs.cs` | the row family: the catalogue (references, error scales), the CPU kernels, the device kernels, the inputs and tolerance shared by the checks and the bench |
| `src/Routing.cs`, `src/Ptx.Routing.cs`, `src/GpuOps.Routing.cs` | top-k, pick, sampling, the route; cross-entropy's backward is a row operation |
| `src/Recurrent.cs`, `src/Ptx.Recurrent.cs`, `src/GpuOps.Recurrent.cs`, `src/Bench/RecurrentCases.cs` | LSTM and GRU cells and steps, forward and backward |
| `src/Conv.cs`, `src/GpuOps.Conv.cs` | convolution forward and backward as im2col / product / col2im compositions |
| `src/Optimizers.cs`, `src/Ptx.Optimizers.cs`, `src/GpuOps.Optimizers.cs` | optimizer steps: sgd with momentum, adam, adamw |
| `src/Attention.cs`, `src/Ptx.Attention.cs`, `src/GpuOps.Attention.cs` | attention: the composition in lane storage, the fused decode kernel, the backward composition; the lane-storage conversions |
| `src/Products.cs`, `src/GpuOps.Products.cs` | products on views (transposed operands, batches) and the matmul backward, through the strided copy and the product kernel |
| `src/Layout.cs`, `src/Ptx.Layout.cs`, `src/GpuOps.Layout.cs`, `src/Bench/LayoutCases.cs` | layout and indexing: the references, the CPU kernels, the device kernels, the gate's and the bench's cases |
| `src/Reduce.cs`, `src/Ptx.Reduce.cs` | reductions (sum, sum of squares, max, min, argmax: whole, rows, columns), row-vector broadcasts, transpose, gemv: the CPU kernels, the device kernels |
| `src/Functions.cs` | log, sin, cos, expm1, tanh, sigmoid, erf, erfc on vectors: coefficient tables shared with the device emitter |
| `src/Team.cs` | the CPU thread team |
| `src/Gpu.cs`, `src/Ptx.cs`, `src/GpuOps.cs` | the CUDA side: driver binding, PTX generation, the operations |
| `src/Checks.cs` | the proofs |
| `src/CpuGeneric.cs`, `src/GenericPtx.cs` | the `INumberBase<T>` version (the `number` column) |
| `src/Bench/` | the bench contract, the type registry, the subjects |
| `Program.cs` | the harness |

## Results

Measured on one machine, 2026-11-06: Intel Core Ultra 9 285 (24 cores, AVX2 with AVX-VNNI), 64 GB; NVIDIA GeForce RTX 5070 Ti
(70 multiprocessors, compute capability 12.0, 16 GB, driver 610.88); Windows 11, .NET 10.0.12. The full table with every
subject's lane and error is `results-qiyas-cuda-cpu.md`; the three columns are the library Qiyas is measured against
(`lib`, present only in the author's checkout), the `INumberBase<T>` version (`number`, — where .NET has no such type)
and Qiyas. Times are medians in ms. Every subject's error against float64 math is in the full table; Qiyas's is equal
to or below the library's on every row, and exact (0) wherever the lane is exact.

**CUDA (2048³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 125.849 / 0.520 / **0.521** / 241.60× | 30.943 / 22.802 / **22.710** / 1.36× | 78.319 / 0.393 / **1.738** / 45.07× | 77.926 / 0.373 / **0.549** / 141.84× |
| float32 | 0.264 / 0.262 / **0.263** / 1.00× | 0.951 / 0.834 / **0.683** / 1.39× | 0.183 / 0.181 / **0.181** / 1.01× | 0.183 / 0.181 / **0.179** / 1.02× |
| float16 | 0.662 / 0.140 / **0.125** / 5.30× | 1.014 / 0.657 / **0.287** / 3.53× | 0.442 / 0.090 / **0.078** / 5.68× | 0.441 / 0.088 / **0.076** / 5.79× |
| bfloat16 | 0.662 / — / **0.124** / 5.34× | 1.003 / — / **0.284** / 3.53× | 0.441 / — / **0.077** / 5.75× | 0.442 / — / **0.077** / 5.76× |
| float8_e4m3 | 489.831 / — / **0.050** / 9835.96× | 130.881 / — / **0.301** / 434.68× | 109.814 / — / **0.068** / 1624.47× | 462.157 / — / **0.046** / 10046.90× |
| float8_e5m2 | 481.180 / — / **0.847** / 567.90× | 117.019 / — / **0.301** / 389.41× | 299.496 / — / **0.069** / 4372.20× | 350.423 / — / **0.050** / 7022.50× |
| int64 | 120.177 / 0.520 / **0.520** / 231.11× | 30.041 / 4.418 / **0.421** / 71.41× |  |  |
| int32 | 0.787 / 0.263 / **0.263** / 2.99× | 1.144 / 1.106 / **0.279** / 4.11× |  |  |
| int16 | 67.399 / 0.139 / **0.124** / 543.98× | 17.225 / 1.585 / **0.231** / 74.44× |  |  |
| int8 | 65.073 / 0.082 / **0.023** / 2780.90× | 21.861 / 1.548 / **0.214** / 102.20× |  |  |
| int4 | 81.782 / — / **0.018** / 4444.68× | 38.206 / — / **0.221** / 173.03× |  |  |
| int2 | 76.942 / — / **0.018** / 4181.61× | 18.738 / — / **0.232** / 80.80× |  |  |
| uint64 | 143.842 / 1.000 / **0.520** / 276.57× | 32.614 / 4.359 / **0.416** / 78.46× |  |  |
| uint32 | 103.208 / 0.263 / **0.263** / 392.58× | 22.980 / 1.076 / **0.271** / 84.67× |  |  |
| uint16 | 68.127 / 0.139 / **0.124** / 547.64× | 17.984 / 1.325 / **0.225** / 79.82× |  |  |
| uint8 | 65.224 / 0.074 / **0.022** / 3033.67× | 18.122 / 1.280 / **0.212** / 85.44× |  |  |
| uint2 | 68.037 / — / **0.017** / 3955.65× | 17.815 / — / **0.226** / 78.72× |  |  |

**CPU (1024³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 7.344 / 6.823 / **6.714** / 1.09× | 6.378 / 15.905 / **3.286** / 1.94× | 31.993 / 6.411 / **5.565** / 5.75× | 30.889 / 4.377 / **4.605** / 6.71× |
| float32 | 3.048 / 3.005 / **2.922** / 1.04× | 2.713 / 5.988 / **1.625** / 1.67× | 3.531 / 4.294 / **2.922** / 1.21× | 2.394 / 1.894 / **1.596** / 1.50× |
| float16 | 2.808 / 7.501 / **1.742** / 1.61× | 6.838 / 765.105 / **1.454** / 4.70× | 45.965 / 55.323 / **2.241** / 20.52× | 45.370 / 23.947 / **0.827** / 54.87× |
| bfloat16 | 1.261 / — / **0.915** / 1.38× | 4.736 / — / **1.386** / 3.42× | 23.833 / — / **1.786** / 13.35× | 23.501 / — / **0.474** / 49.55× |
| float8_e4m3 | 26.282 / — / **2.222** / 11.83× | 37.162 / — / **1.477** / 25.17× | 89.788 / — / **3.686** / 24.36× | 429.956 / — / **1.914** / 224.65× |
| float8_e5m2 | 21.284 / — / **2.422** / 8.79× | 29.022 / — / **1.491** / 19.47× | 283.307 / — / **3.878** / 73.05× | 313.137 / — / **2.083** / 150.31× |
| int64 | 19.776 / 6.763 / **6.615** / 2.99× | 6.126 / 24.649 / **0.876** / 7.00× |  |  |
| int32 | 20.288 / 2.959 / **2.948** / 6.88× | 5.923 / 6.204 / **0.607** / 9.76× |  |  |
| int16 | 1.639 / 1.425 / **0.840** / 1.95× | 5.658 / 2.371 / **0.992** / 5.71× |  |  |
| int8 | 1.564 / 0.432 / **0.154** / 10.14× | 6.123 / 2.507 / **0.597** / 10.26× |  |  |
| int4 | 2.437 / — / **0.101** / 24.11× | 11.926 / — / **0.532** / 22.43× |  |  |
| int2 | 2.868 / — / **0.267** / 10.74× | 6.899 / — / **0.514** / 13.42× |  |  |
| uint64 | 6.599 / 6.738 / **6.705** / 0.98× | 6.337 / 22.120 / **0.810** / 7.82× |  |  |
| uint32 | 5.172 / 2.943 / **2.933** / 1.76× | 6.090 / 3.922 / **0.429** / 14.18× |  |  |
| uint16 | 1.375 / 1.283 / **0.796** / 1.73× | 5.178 / 1.925 / **0.698** / 7.42× |  |  |
| uint8 | 1.630 / 0.423 / **0.130** / 12.59× | 6.154 / 2.198 / **0.569** / 10.81× |  |  |
| uint2 | 2.761 / — / **0.232** / 11.91× | 6.861 / — / **0.402** / 17.08× |  |  |

Where Qiyas is not ahead: the memory-bound adds of 32- and 64-bit types (the same bytes moved, within a few percent),
and CUDA float64 matmul, which runs at the device's float64 rate and exact where the library computes it in float32.

The tables above are the four core rows of the full `cuda,cpu` run the regression gate compares against
(`docs/wip/baseline.md`); the other families' rows, the block formats and bool are in `results-qiyas-cuda-cpu.md` of
the latest run, and `COVERAGE.md`'s "Known slow" names the measured cells that are correct but below what the device
allows.

## Open

What the gates have not confirmed, or is not done, is in `COVERAGE.md` ("State" for the order, **todo** in the rows):
the flash-fused attention rows, ctc, a GRU sequence bench row, the block-scaled tiled product, a bump-allocated scope
past the buffer pool, and the known-slow cells. The README claims nothing beyond what the gates and the regression
comparison have confirmed.
