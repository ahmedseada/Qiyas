# Qiyas

Qiyas (قياس, "measure") is the bench and the proofs of **Adad** (عدد, "number"), the library in `src/Adad`. Adad is a
generic number contract and the kernels over it: every stored type — float64, float32,
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
- **One device contract.** Each operation family is an interface generic over the number type and the device's buffers
  (`src/Adad/Abstraction/`, namespace `Adad.Abstraction`: `IDevice<TBuffer, TIndex>` and its families, each operation
  under its own constraint — `IAdad<TN>` for every type, `IFloatAdad<TN>` for the float formulas — with the types,
  formats and everything the devices share). The CPU (`Adad.Cpu`, `CpuDevice.Device`: byte arrays) and CUDA
  (`Adad.Cuda`, `GpuOps`: device pointers) implement it, each in its own folder and namespace with its own lanes; a
  device may implement a family at a time.
- **Proven, not assumed.** Before anything is timed, every conversion is checked bit for bit against a scalar
  reference written from the format's definition, the dot lanes against scalar sums, exp against the runtime's, and
  the device's conversions against the CPU's; then every operation of every family runs over every type it applies to
  on both devices, on small odd shapes, against float64 references (`src/Qiyas.Checks/Checks.cs`); the block codecs against blocks
  laid out by hand from the formats' specifications; the block-scaled products of Q8_0 / Q4_0 (A·Bᵀ and gemv, k in whole
  blocks) forced on both devices; the products on views at shapes where the packing reads transposed and batched
  operands where they are stored, on both devices; the products on the device under every tensor kind the device
  accepts (the measured choice, then the 4-bit kinds and the 32-deep 8-bit instruction forced on and off), and under
  every tensor-kernel configuration the device can run with every kind choice (k = 53 and 300, every integer and
  narrow-float type); attention rows by both the composition and the fused rows (causal, spans with starts, head
  dimensions up to 128, rows that see no key), and the backward by both its fused kernels and its composition on the
  same shapes; attention's options (grouped heads, multi-query, a window, a cap, the log-sum-exp) by every form; the optimizer steps and the step across five tensors under one clipping (none, below the gradients' norm, above it); a device's memory by ranges, byte for byte against the same steps on host arrays (odd offsets and lengths, a zeroed rent of a buffer the pool held with other bytes); the caches' writes (inside, past and before the capacity; the rows outside a write as they were; caches of each block format with rows of whole blocks and of shared ones), attention over them against the reference of the values as stored, and the decoder's mask (the excluded score −∞ as the type stores it); attention over tensors through views of their rows by every form (a step at a time in batches, grouped heads and a window; Q, K and V at their offsets in one projection's rows; the backward; a cache's queries; a block format's moved outputs held to both encodings' steps); the top-k as log-probabilities (TopK's indices exactly, rows wider than a block of threads, k the whole row), top-k in Adad's order on every device (rows of NaN, ±0, ±∞ and ties: the columns exactly), top-k past the row refused and pick outside the row (0, nothing read); a decoder's rings pushed and advanced on the device and the penalties of their windows (an empty, part filled and wrapped ring, its last tokens, tokens that repeat and tokens outside the row, rows a stride apart; every score outside the window x's own), and the sampler's draw from the kept columns (a temperature, top-k, the nucleus, min-p and all three; a row of +∞ scores, a NaN among the scores, a row of −∞; each row's column held to the float64 rule within the lane's rounding at the seed's draw for the device's step); ctc's loss and gradient (the float64 reference's gradient checked
  against its own loss by central differences; short sequences, empty and repeated labels, another blank, a sequence
  no alignment fits); the thread
  team under 20 000 jobs of alternating chunk counts (every chunk within its job, every index once). The float64
  reference of every product is accumulated in double-double, so an exact lane reads 0. The bench refuses to time while
  any gate fails.
- **The operation surface.** `COVERAGE.md` is the list, one row per operation with the lane rule on each device:
  element-wise formulas and their backward passes (the sigmoid focal loss among them); reductions, broadcasts, transposes, gemv; the row family
  (softmax, log_softmax, the norms and their backward passes, rope, masked softmax, cross-entropy's backward, weighted too); layout
  and indexing (strided copies of up to six dimensions, one-hot, gathers, scatter-add, im2col/col2im, interpolation
  (half-pixel, align-corners or asymmetric, given scales) and its backward, shift, pad, the pools (ceil mode, counting the
  pads) and their backward, depthwise convolution); image resampling by separable filters (box, triangle, Hamming, cubic, Lanczos; antialiased when shrinking) from any input type into any type in the output's float lane, with a scale and a shift a channel; products on views (transposed, batched) and the matmul backward, and the product's epilogue (a bias, an activation keeping its input or the gradient through one, β·C, a residual, a low-rank term; one rounding); products with a weight stored as codes of any type with scales and zero points (for the tensor, a row, groups of a row, blocks), each weight's value made in the lane as the product reads its code, and its expansion; a mixture of experts (the top-k routing of tokens to experts, and the experts' products and gated pairs over the rows grouped by expert, the offsets read where they are on the device); attention
  (rows by the composition or the fused rows, whichever is measured faster; causal, spans, decode, backward; grouped heads and multi-query by every form, a sliding window, a cap on the scores, each query's log-sum-exp); key and value caches at a position the device holds (rows written there as any type by one rounding, a compact cache a cache of a compact type: a block format's by its blocks, a row of whole blocks a block at a time, rows that share blocks by re-encoding the blocks a write touches; attention over the rows filled, each row read where it is stored; the decoder's mask; the position advanced on the device); a layer's projections into the layouts attention reads (the rotary embedding by tables: half or interleaved pairs, partial, inverse; every attention form over tensors through strided views of their rows: a step at a time in batches, Q, K and V at their offsets in one projection's rows); convolution forward and backward on every type (images, groups, a pad a side; y the product's epilogue over the output, the bias a value an output channel; dx, dw, db, each in the exact lane for its bounds where the type is not float); a device's memory by ranges (a buffer rented zeroed; a range of bytes at an offset cleared, uploaded, downloaded, copied to another buffer or within its own); the optimizer steps (plain sgd, sgd with momentum, adam, adamw), and a step across several tensors under one clipping of all their gradients (the squares summed as one sum across the tensors, the factor made on the device, each gradient scaled as the step reads it); LSTM and GRU cells and
  steps with their backward passes, LSTM and GRU sequences with theirs (either direction); top-k (the values, or their log-probabilities), pick, sampling, routing, cross-entropy and ctc (loss and
  gradient: the logits frame-major, the labels one sequence after another, +∞ where no alignment fits); the box overlap losses (IoU, GIoU, DIoU, CIoU; the type's precision added to each divisor that can be 0)
  and their gradients (the loss's derivative, CIoU's α included; a tie of a minimum or a maximum shared in halves); a decoder's steps on the device (each row's ring of the tokens it took, pushed where the length the device holds says, the repetition, presence and frequency penalties of the ring's window on the row's scores, and the row's next token drawn from softmax(scores / temperature) held to its top-k columns, to the nucleus and to min-p by a seeded draw at the step the device holds); and the block formats and bool
  through all of it. What is still open is listed in COVERAGE.md's "State" and marked **todo** in its rows.

## The bench

`Qiyas.Bench.dll [cpu|cuda|cuda,cpu] [--subjects a,b] [--types ...] [--ops add,matmul,softmax,rms_norm,map|reduce|rows|layout|products|attention|optimizers|conv|recurrent|routing|cache|heads|decoding|memory|experts|all] [--baseline name] [--shapes regular|odd] [--report] [--repeat N]`

`--subjects` picks the columns (`--subjects adad,number`: Adad and the `INumberBase<T>` version), `--types` the stored types (the seventeen plain
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
its backward, weight_norm and its backward, rope, scale_mask_softmax, add_rms_norm, add_layer_norm — and `norm_params_bwd:rms|layer`,
4096 × 4096, float types); `layout` adds the layout operations (`layout:<name>` for each of `LayoutArgs.Names`: strided,
one_hot, gather_rows, gather_cols, scatter_add_rows, im2col, col2im, interpolate_nearest, interpolate_bilinear, their `_bwd`, shift).
`products` adds the products on views (`matmul_nt`, `matmul_tn`, `matmul_tt` at 4096³, `matmul_batched`: 16 × 1024³,
`matmul_bwd`: dA and dB at 4096³, `matmul_blocks`: A·Bᵀ of two Q8_0 / Q4_0 operands by the block-scaled product alone);
`attention` adds `attention`, `attention_causal` (16 heads, 2048 × 2048, d = 128),
`attention_fused`, `attention_fused_causal` (the same shapes, the fused rows alone, for comparing with the measured
choice), `attention_decode` (32 heads, one query over 4096 keys) and `attention_bwd` (8 heads of 1024). The decisions
taken from measurements during a run (attention's fused-or-composed choice per shape and type, among others) are printed
at its end with `--report`. A single name works too
(`--ops map:erf,sum:rows,row:rope,layout:im2col,attention_decode`); `optimizers` adds `optimizer:sgd`, `optimizer:sgd_momentum`,
`optimizer:adam`, `optimizer:adamw` over 16M parameters (the moments in lane storage; each run restores p, m, v first,
named in the lane column) and `optimizer_many` (an adamw step across 64 tensors of those 16M parameters, their gradients
clipped to a norm of 1; `QIYAS_MANY_TENSORS` sets another count of tensors: at 1 the row is `optimizer:adamw` with the
sum's pass; the CPU's restore is a copy a tensor, timed); `conv` adds `conv` and `conv_bwd` (a 64 × 128 × 128 image by 128 filters of 3 × 3); the
pools and the depthwise convolution are layout operations (`layout:maxpool`, `layout:avgpool`, their `_bwd`,
`layout:adaptive_maxpool`, `layout:adaptive_avgpool`, `layout:depthwise_conv`); `recurrent` adds `lstm_step`, `gru_step`
and their `_bwd` (a batch of 1024, input 1024, hidden 1024), `lstm_seq`, `lstm_seq_bwd`, `gru_seq` and `gru_seq_bwd` (16 steps of 256 × 512 → 512); `routing` adds `topk_rows`, `log_softmax_topk` (the top 8 of each row as log-probabilities), `pick_rows`, `sample_rows`
(4096 rows of 32768), `route` (1M indices into 64 buckets), `route_wide` (1M into 65 536), `cross_entropy` (the loss per row: the subject's fused loss where it has one, else log_softmax, pick, neg), and
`ctc`, `ctc_bwd` (256 frames of 32 sequences over 1024 classes, 64 labels each; the loss scored, then the gradient);
`cache` adds the key and value caches' rows (float types): `cache_write` (16 heads of 512 rows of d = 128 into a cache of
2048 rows at row 1000, as the type itself — a block format's rows whole blocks for Q8_0 and Q4_0, shared blocks for Q4_K and Q6_K — the cache scored), `quantize_rows` (the same
rows through Quantize as int8 codes with a scale a row, the scales scored), `attention_cached` (`attention_decode`'s shape,
32 heads and one query at the last of 4096 rows, over a cache of the type) and `decoder_mask` (512 steps over 4096
positions). `--shapes odd` keeps a row of the cache whole 32-bit words (d = 124);
`heads` adds a layer's projections into the layouts attention reads: `rope_table` (the table rotary embedding over 16
heads of 512 steps of d = 128) and `attention_steps` (`attention_causal`'s shape with Q, K, V and O a step at a time).
`decoding` adds a decoder's steps (every type but bool): `penalize` (512 rows of 32768 scores, a map row's 16M values, each row's
window the last 64 tokens of a ring of 256 that has wrapped), `sample_plain` and `sample_filtered` (`sample_rows`'s
4096 rows of 32768 scores: a column of each row drawn from every column, and from the top 40, the nucleus of 0.95 and
min-p 0.02 at a temperature of 0.8). `memory` adds a device's memory by a range (the `uint8` row; 16M bytes but the
first and last three, a range off a word at both ends): `memory:copy` (into another buffer), `memory:clear` and
`memory:rent_zeroed` (a zeroed buffer of the 16M bytes rented and returned); `map:copy` over uint8 is the copy's
nearest equivalent. `experts` adds a mixture of experts (float types): `route_topk` (4096 tokens' 8 first of 64
experts in TopK's order, normalized; the weights scored), `expert_matmul` and `expert_gated` (gated_silu: two products
a row) over 8 experts with biases, the core product's size grouped as routing half as many tokens to 2 experts each
groups them, the weights within ±√(3 / K) (the sums of order one, inside every type's range). The sampling rows' error is not a distance between columns: it is the probability the worst
row's pick is off the float64 rule by (`SamplingArgs.PickErrors`): 0 where the seed's draw lies in the pick's interval of
the kept weights' running sum, the lane's rounding where a sum's rounding moved the pick to the next kept column, and up
to a column's probability where the lane holds two weights at a threshold as one and keeps them together; with
`QIYAS_MAP_DEBUG=1` each such row is printed with the running sum on both sides of its draw.
`COVERAGE.md` lists every operation with its lane rule on each device. The block formats and bool run through every
family: a block format's lane is float32, read a value at a time anywhere (both devices decode by index) and written a
block at a time (the CPU re-encodes the blocks a store touches, tasks never sharing a block; the device computes
value-wise stores in float32 storage and encodes by blocks — all four formats have device encoders, byte for byte the
CPU's); bool is a one-bit packed type held to 0/1 (division is n/a). Products with a Q8_0 or Q4_0 operand take the
block-scaled integer dot (`s8dp blocks`).

Every implementation in the loaded assemblies that implements `IBenchSubject` (`src/Qiyas.Checks/Bench/Contract.cs`) is
found by reflection and becomes a column: `adad` (`src/Qiyas.Checks/Bench/AdadSubject.cs`: every operation once, over the
device contract), `number` (one implementation over .NET's `INumberBase<T>`, for the types .NET has a
`T` for), and whatever you add. The harness makes the inputs (stored bytes of each type, from Adad's proven
conversions, so every subject sees the same bits), times `Run` (warmed until the runtime compiles nothing more and the
time stops falling, then medians over an adaptive number of runs), reads the
output after `Collect`, and scores it against float64 math on the same inputs. Results go to `results-qiyas-<devices>.md`.

To add an implementation: one class with a parameterless constructor implementing the four methods (and `Map`, `Reduce`,
`MapRow`, `Transpose`, `Gemv` for the other families, which default to `null`), each returning `null` for anything it does not do (shown as —),
never an approximation. Inputs and outputs are stored bytes of the
named type; `Run` must include your device synchronization; `Collect` writes the output; `Lane` says what you computed in.

`profile`, `gpuprofile` and `team` are the bench's own diagnostics (matmul phases on each device; the thread team's dispatch cost).

## Building

`dotnet build Qiyas.sln -c Release` (net10.0) builds three projects: `src/Adad` (the library), `src/Qiyas.Checks` (the
proofs, and what they share with the bench: the subject they run Adad through, the float64 references, the stored types
by name, the cases) and `bench/Qiyas.Bench` (the harness: `dotnet bench/Qiyas.Bench/bin/Release/net10.0/Qiyas.Bench.dll`,
results written at the repository's root). The gates run at the start of every bench invocation; the regression gate is
`python docs/wip/compare.py docs/wip/baseline.md results-qiyas-cuda-cpu.md` after a full `cuda,cpu` run (a cell more
than 5 % slower, or an error that grew, is a regression). `sh guard.sh` checks every C# file of the repository (the
names it must not use, the license header); run it before every build. A clone runs every subject named here.

## Layout

| file | what |
|---|---|
| `src/Adad/Abstraction/`, `src/Adad/Cpu/`, `src/Adad/Cuda/` | a folder and a namespace each: the shared contracts, types and references (`Adad.Abstraction`); the CPU (`Adad.Cpu`); CUDA (`Adad.Cuda`). An operation family's arguments, catalogue and references sit in `Abstraction/`, its kernels in each device's folder |
| `src/Adad/Abstraction/` | namespace `Adad.Abstraction`, what every device shares. The device contract: `IDevice<TBuffer, TIndex>` and its families (memory, element-wise, reductions, rows, layouts, products, attention, optimizers, convolution, recurrent, routing, ctc, boxes, experts, caches, decoding); `Fmt` and `Lane`, the stored formats and lanes described once for every device |
| `src/Adad/Cpu/Kernels.Device.cs`, `src/Adad/Cuda/GpuOps.Device.cs` | the CPU and CUDA as devices: the contract on byte arrays and on device pointers |
| `src/Adad/Abstraction/IMemory.cs`, `src/Qiyas.Checks/Checks.Memory.cs` | a device's memory: buffers rented (zeroed when asked), returned, uploaded, downloaded, copied and cleared, whole or a range of bytes at an offset. The CPU moves a long range across its team; CUDA takes a range as an address into the buffer for the driver's fill and copies; Vulkan fills whole words by its fill command and copies a long range's whole words by the copy kernel, the bytes off a word at either end by its copy command |
| `src/Adad/Abstraction/Numbers.cs` | the contract: `IAdad<TSelf>`, lanes, dot lanes, the visitors; exp on vectors |
| `src/Adad/Abstraction/Types.cs` | the seventeen stored types and their lane rules |
| `src/Adad/Abstraction/Blocks.cs`, `src/Adad/Cuda/Ptx.Blocks.cs`, `src/Adad/Cpu/BlockProducts.cs`, `src/Adad/Cuda/BlockProducts.cs` | the block formats Q8_0, Q4_0, Q4_K, Q6_K and bool: the codecs (CPU and device encoders, a thread per block), the block-scaled integer gemv |
| `src/Adad/Cpu/MatMulBlocks.cs`, `src/Adad/Cuda/MatMulBlocks.cs` | the block-scaled tiled product A·Bᵀ of Q8_0 / Q4_0 on the CPU (the s8 dot lane) and the device (the s8 tensor instruction at k = 32), and its measured choice against the decoded route |
| `src/Adad/Cpu/Report.cs`, `src/Adad/Cuda/Report.cs` | `--report`: the facts and measured decisions of each device, the buffers' costs and steady state |
| `src/Adad/Cuda/Arena.cs` | the device's buffer arena: a scope's buffers bumped in one region, released from the top, the region grown between calls |
| `src/Adad/Cpu/Workspace.cs` | the CPU product's per-thread workspace (packed B, loaded A / B, C lanes, A panels) |
| `docs/wip/` | the regression comparer (`compare.py`), the baseline run it compares against (`baseline.md`), the README Results generator (`readme_results.py`) |
| `src/Adad/Abstraction/Convert.cs` | every load and store on vectors; `Widths`, any lane to any lane |
| `src/Adad/Cpu/Kernels.cs` | the CPU kernels: add, product (GotoBLAS-shaped, dot lanes in groups), softmax, rms_norm |
| `src/Adad/Abstraction/Map.cs`, `src/Adad/Cpu/Map.cs`, `src/Adad/Cuda/Ptx.Map.cs` | the element-wise family: the catalogue of formulas (with float64 references), the CPU kernel, the device kernel |
| `src/Adad/Abstraction/Rows.cs`, `src/Adad/Cpu/Rows.cs`, `src/Adad/Cuda/Ptx.Rows.cs`, `src/Adad/Cuda/GpuOps.Rows.cs`, `src/Qiyas.Checks/Bench/RowInputs.cs` | the row family: the catalogue (references, error scales), the CPU kernels, the device kernels, the inputs and tolerance shared by the checks and the bench |
| `src/Adad/Cpu/Norms.cs`, `src/Adad/Cuda/Ptx.Norms.cs`, `src/Adad/Cuda/GpuOps.Norms.cs`, `src/Adad/Vulkan/Kernels.Norms.cs`, `src/Adad/Vulkan/VulkanOps.Norms.cs`, `src/Qiyas.Checks/Checks.Norms.cs` | group normalization over `GroupShape` (batch, group, instance, a layer's over channels and positions; by given statistics for inference) and its backward, and the rotary embedding by tables (`RopeArgs`: half or interleaved pairs, partial, inverse); their references sit in `Abstraction/Rows.cs` |
| `src/Adad/Abstraction/Resample.cs`, `src/Adad/Cpu/Resample.cs`, `src/Adad/Cuda/Ptx.Resample.cs`, `src/Adad/Cuda/GpuOps.Resample.cs`, `src/Adad/Vulkan/Kernels.Resample.cs`, `src/Adad/Vulkan/VulkanOps.Resample.cs`, `src/Qiyas.Checks/Checks.Resample.cs` | image resampling (`ResampleArgs`, the plan by `Resampling.Plan`: the box, triangle, Hamming, cubic and Lanczos kernels, separable, the kernel stretched when shrinking; the weights made in float64 and rounded once to the lane): each pass Σ x·w in the output's float lane from any input type read where it is stored, then a scale and a shift a channel, one rounding into any type (an integer's or bool's by the type's own conversion: to nearest, ties to even, held to the range) |
| `src/Adad/Abstraction/Boxes.cs`, `src/Adad/Cpu/Boxes.cs`, `src/Adad/Cuda/Ptx.Boxes.cs`, `src/Adad/Vulkan/Kernels.Boxes.cs` | the box overlap losses (IoU, GIoU, DIoU, CIoU, `BoxOverlap`) and their gradients (the loss's derivative): each output word computed where it is stored from the corners read where they are |
| `src/Adad/Abstraction/Routing.cs`, `src/Adad/Cpu/Routing.cs`, `src/Adad/Cuda/Ptx.Routing.cs`, `src/Adad/Cuda/GpuOps.Routing.cs` | top-k in Adad's order (`Routing.Precedes`: larger first, a NaN above every number, an equal value by the lower column; the values gathered a stored word a thread), the top-k as log-probabilities (TopK's indices, then each chosen value less the row's first-ranked and the log of Σ exp(x − it), one rounding), pick, sampling, the route, the cross-entropy loss per row (one pass; a weight a row when given); its backward is a row operation (weighted: `weighted_cross_entropy_bwd`) |
| `src/Adad/Cuda/Ptx.Radix.cs`, `src/Adad/Cuda/GpuOps.Radix.cs` | the device's ordering by target for scatter-add and the route: counts, the multi-block exclusive scan, the stable LSD radix sort |
| `src/Adad/Abstraction/Ctc.cs`, `src/Adad/Cpu/Ctc.cs`, `src/Adad/Cuda/Ptx.Ctc.cs`, `src/Adad/Cuda/GpuOps.Ctc.cs` | ctc: the arguments and the float64 reference, the CPU kernel, the device kernels (rows, then a block per sequence) |
| `src/Adad/Abstraction/Recurrent.cs`, `src/Adad/Cpu/Recurrent.cs`, `src/Adad/Cuda/Ptx.Recurrent.cs`, `src/Adad/Cuda/GpuOps.Recurrent.cs`, `src/Qiyas.Checks/Bench/RecurrentCases.cs` | LSTM and GRU cells and steps, forward and backward |
| `src/Adad/Abstraction/Conv.cs`, `src/Adad/Cpu/Conv.cs`, `src/Adad/Cuda/GpuOps.Conv.cs`, `src/Adad/Vulkan/VulkanOps.Conv.cs` | convolution, its definition and float64 reference in `ConvArgs` (weights [OC, C / groups, kh, kw], a pad a side): one image of one group fused into the product's packing (implicit GEMM: the image read through the index map; on the device's plain lanes a measured choice against im2col); images, groups and the epilogue's terms by the general path (the columns in lane storage, a product an image and group, the epilogue's store over the output); the backward as im2col / product / col2im compositions; a type that is not float in the exact lane its bounds give |
| `src/Adad/Abstraction/Optimizers.cs`, `src/Adad/Cpu/Optimizers.cs`, `src/Adad/Cuda/Ptx.Optimizers.cs`, `src/Adad/Cuda/GpuOps.Optimizers.cs`, `src/Adad/Vulkan/Kernels.Optimizers.cs`, `src/Adad/Vulkan/VulkanOps.Optimizers.cs`, `src/Qiyas.Checks/Checks.Optimizers.cs` | optimizer steps: plain sgd, sgd with momentum, adam, adamw; and the step across several tensors under one clipping (`OptimizerMany`, `OptimizerArgs.MaxNorm`): one whole sum of the squares of all the tensors' gradients by the device's own rule for a whole sum, the factor by the map's `clip_factor` in the lane, the steps reading it where it is; on the devices the sum's pass and the step are a launch each for all the tensors, the kernels finding their tensor in a table on the device (`TableRow`; a block format's steps a launch a tensor) |
| `src/Adad/Abstraction/Attention.cs`, `src/Adad/Cpu/Attention.cs`, `src/Adad/Cuda/Ptx.Attention.cs`, `src/Adad/Cuda/GpuOps.Attention.cs` | attention: the measured choice, the composition in lane storage, the fused decode kernel, the backward composition; the lane-storage conversions |
| `src/Adad/Cpu/AttentionRows.cs`, `src/Adad/Cuda/Ptx.AttentionRows.cs`, `src/Adad/Cuda/Ptx.AttentionMma.cs` | attention's fused rows (flash: keys in tiles, the softmax online, no scores in memory) on the CPU and the device; on the device the tensor instructions for float16, bfloat16 and the 8-bit floats |
| `src/Adad/Cuda/Ptx.AttentionBackward.cs` | attention's fused backward on the device (dQ by the rows, dK and dV by the keys) |
| `src/Adad/Abstraction/Products.cs`, `src/Adad/Cpu/Products.cs`, `src/Adad/Cuda/GpuOps.Products.cs` | products on views (transposed operands, batches) and the matmul backward: the product's packing reads the operands where they are stored (`Kernels.Operands`; the device's tensor and dot routes), otherwise through the strided copy |
| `src/Adad/Cpu/Scaled.cs`, `src/Adad/Cuda/Ptx.Scaled.cs`, `src/Adad/Vulkan/Kernels.Scaled.cs` | a scaled weight's products (`ScaledWeight`: codes of any type, scales and zero points by blocks of rows and columns): each value (code − zero)·scale made in the float lane as the product reads the code (the CPU packing's panel source, the stored GEMM's B loads, the Vulkan product's tiles), never the weight expanded; then the epilogue's store |
| `src/Adad/Abstraction/AttentionLayout.cs` | attention over tensors through views (`AttentionArgs.QRows`, `KRows`, `VRows`, `ORows`: `HeadRows`, where a tensor's head rows lie by its batch, head and row strides and an offset; an output's view a dense order), written once over the device contract: Q, K, V and dO taken a head at a time by the device's strided copies (a row of whole storage units moved as it is, a block format's bytes as bytes; a block format whose rows share blocks re-encoded), the form run, the outputs moved back; the CPU moves whole-unit rows as bytes into pooled buffers |
| `src/Adad/Abstraction/Experts.cs`, `src/Adad/Cpu/Experts.cs`, `src/Adad/Cuda/GpuOps.Experts.cs`, `src/Adad/Vulkan/VulkanOps.Experts.cs` | a mixture of experts (`IExperts`): the routing (each token's k experts at the slots of their ranks in TopK's order, the weights normalized in the lane; the slots routed by the device's ordering) and the experts' products over the rows grouped by expert in one launch (CUDA: the stored GEMM's grouped mode; Vulkan: the grouped product's tiles), each row's expert's bias and the gate (Map's formula) in the store |
| `src/Adad/Abstraction/Cache.cs`, `src/Adad/Cpu/Cache.cs`, `src/Adad/Cuda/GpuOps.Cache.cs`, `src/Adad/Cuda/Ptx.Cache.cs`, `src/Adad/Vulkan/VulkanOps.Cache.cs`, `src/Adad/Vulkan/Kernels.Cache.cs` | key and value caches (`ICache`): rows written at the position the device holds, each value as the cache's type by one rounding (a stored word of the cache its own writer; a block format cache's rows a warp or an invocation a block of the written rows where a row is whole blocks, else by re-encoding the blocks the write touches, their other values kept), attention over the rows filled by the decode kernel (the limits from the position on the device, K and V read where they are stored as the cache's type, the cap in the kernel), the decoder's mask (the excluded score `CacheView.Excluded`: −∞ as the type stores it, else its lowest), the position advanced |
| `src/Adad/Abstraction/Decoding.cs`, `src/Adad/Cpu/Decoding.cs`, `src/Adad/Cuda/GpuOps.Decoding.cs`, `src/Adad/Cuda/Ptx.Decoding.cs`, `src/Adad/Vulkan/VulkanOps.Decoding.cs`, `src/Adad/Vulkan/Kernels.Decoding.cs`, `src/Qiyas.Checks/Checks.Penalties.cs` | a decoder's steps (`IDecoding`): each row's ring of tokens (`HistoryPush` at the length the device holds, advanced by `PositionAdvance`), and `Penalize` on every type but bool (`PenaltyArgs`: the window the ring's last tokens, 0 to the ring's capacity; a token that occurs n times there once, (x above 0 ? x / repeat : x · repeat) − (presence + frequency · n) in the float lane, every other score x's value; the scores read where they are, rows a stride apart). The CPU orders a row's window and walks the row a tile at a time; the devices order each window into a table (a thread a slot, its rank by (token, slot)) with a mask of its tokens' bits, then copy contiguous rows of whole stored words as they are and store only the words that hold a window's token (a thread a slot, the first slot of a word alone), or, for a block format and rows a stride apart, store every word, a score whose column's bit is clear x's and the others sought in the table by halvings |
| `src/Adad/Abstraction/Sampling.cs`, `src/Adad/Cpu/Sampling.cs`, `src/Adad/Cuda/GpuOps.Sampling.cs`, `src/Adad/Cuda/Ptx.Sampling.cs`, `src/Adad/Vulkan/VulkanOps.Sampling.cs`, `src/Adad/Vulkan/Kernels.Sampling.cs`, `src/Qiyas.Checks/Checks.Sampling.cs` | a decoder's sampling (`IDecoding.SampleFiltered`, `SamplingArgs`), every type but bool: each rule keeps a prefix of the row's scores as stored in Adad's one order (`Routing.Precedes`), within the one before it: top-k the first TopK columns (equal scores at the cut by column), the nucleus the shortest prefix of those whose weights reach TopP of their sum, min-p the columns whose weight is at least MinP; the weights exp((x − m) / temperature) in the float lane, m the first-ranked score (a row whose first-ranked score is a NaN or +∞, or a row of −∞, gives −1 and a NaN probability); found without ordering the row by narrowing the range of the scores' order keys (`Routing.OrderKey`: a stored value as an unsigned integer in Adad's order), a pass of counts or of sums a step (the CPU by halves until top-k leaves few columns, which it gathers and orders; the devices sixteen ways a pass, the nucleus over top-k's columns gathered into shared memory where k is 256 or fewer); the draw `Draws.Uniform(key, step, row)` with the step read where the device holds it, the kept column whose interval of the running sum holds it; each pick's probability among the kept, stored as the float lane holds it. The devices keep the weights in lane storage (one kernel reads the type, the other is the lane's alone) |
| `src/Adad/Cpu/Epilogue.cs`, `src/Adad/Cuda/Ptx.ProductStore.cs`, `src/Adad/Vulkan/Kernels.ProductStore.cs` | the product's epilogue (`ProductEpilogue`): the product and the low-rank term into the float lane's sums, then one store adding the bias, β·C (read where it is stored) and the residual around the activation or the gradient through one (Map's formulas), y rounded to the type once; the general convolution's store is the same kernel |
| `src/Adad/Abstraction/Layout.cs`, `src/Adad/Cpu/Layout.cs`, `src/Adad/Cuda/Ptx.Layout.cs`, `src/Adad/Cuda/GpuOps.Layout.cs`, `src/Qiyas.Checks/Bench/LayoutCases.cs` | layout and indexing: the references, the CPU kernels, the device kernels, the gate's and the bench's cases |
| `src/Adad/Abstraction/Reduce.cs`, `src/Adad/Cpu/Reduce.cs`, `src/Adad/Cuda/Ptx.Reduce.cs` | reductions (sum, sum of squares, max, min, argmax: whole, rows, columns), row-vector broadcasts, transpose, gemv: the CPU kernels, the device kernels |
| `src/Adad/Abstraction/Functions.cs` | log, sin, cos, expm1, tanh, sigmoid, erf, erfc on vectors: coefficient tables shared with the device emitter |
| `src/Adad/Abstraction/Team.cs` | the host thread team, shared: the CPU's kernels and the shared conversions run on it (jobs closed on completion, so a late worker never takes a chunk under the next job's half-written description) |
| `src/Adad/Cuda/Gpu.cs`, `src/Adad/Cuda/Ptx.cs`, `src/Adad/Cuda/GpuOps.cs` | the CUDA side: driver binding, PTX generation, the operations |
| `src/Qiyas.Checks/Checks.cs`, `src/Qiyas.Checks/References.cs` | the proofs; the float64 references the proofs and the bench score against |
| `bench/Qiyas.Bench/CpuGeneric.cs`, `bench/Qiyas.Bench/GenericPtx.cs` | the `INumberBase<T>` version (the `number` column) |
| `src/Qiyas.Checks/Bench/` | the bench contract, the type registry, the Adad subject |
| `bench/Qiyas.Bench/Program.cs` | the harness |

## Results

Measured on one machine (Intel Core Ultra 9 285: 24 logical processors, 256-bit vectors, the s8/u8/s16/u16 dot
lanes, as the runtime reports; 64 GB; NVIDIA GeForce RTX 5070 Ti: 70 multiprocessors, compute capability 12.0, 16 GB,
driver 610.88, PTX target sm_80; Windows 11, .NET 10.0.12). The tables are assembled from several runs on that machine
(2026-11-07, the end of item 6): item 6's capped full run, stopped at the owner's request (two of its invocations
complete, the third partial), and the single-family runs of
items 5 and 6, each cell taking each subject's newest reading of code that is in main and unchanged since; a cell none
of those runs measured keeps item 4's reading (3512 of the 4234 rows' Adad cells). `docs/wip/baseline.md` names each
row's run in its source column. Compared cell by cell against item 4's baseline by `docs/wip/compare.py`: 93 cells
slower by more than 5 %, 6 errors grown (each a lane change recorded in `COVERAGE.md`), 18 new rows (the CPU's q6_k
and bool, q4_k's softmax and rms_norm); the slower cells are single readings, listed in `COVERAGE.md`'s "Known slow".
Device bandwidth rows follow the driver's memory P-state (a row timed at P3 reads up to about 1.9× its P1 time);
setting "Prefer maximum performance" for dotnet.exe in the NVIDIA control panel gives stable readings.

The full table with every subject's lane and error is `docs/wip/baseline.md`; the columns shown here are the
`INumberBase<T>` version (`number`, — where .NET has no such type) and Adad (`qiyas` in that file: the library's name
before it was Adad). Times are medians in ms. Every subject's error is against float64 math (products in
double-double). On the core rows Adad's error is equal to or below `number`'s except two: CUDA float32 rms_norm
(2.97e-7 against 2.93e-7) and CPU float64 softmax (1.08e-17 against 4.34e-19, both far below float64's step). Adad's
error is exact (0) wherever the lane is exact; the CUDA float64 product takes the sliced 8-bit tensor route (`f64x`,
proven exact, 4.15 ms against 23.1 in the float64 lane). The narrow floats, the sub-byte integers, the block formats
and bool have no `number` column (—): .NET has no such types.

**CUDA (2048³ product, 16M-value add, 4096×4096 rows)** — number / **adad** / × (× = number ÷ adad)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 1.000 / **0.520** / 1.92× | 23.142 / **4.149** / 5.58× | 0.394 / **1.826** / 0.22× | 0.372 / **0.549** / 0.68× |
| s32 |  |  |  |  |
| float32 | 0.263 / **0.263** / 1.00× | 0.842 / **0.681** / 1.24× | 0.181 / **0.180** / 1.01× | 0.184 / **0.180** / 1.02× |
| float16 | 0.139 / **0.125** / 1.11× | 0.654 / **0.229** / 2.86× | 0.092 / **0.079** / 1.16× | 0.089 / **0.084** / 1.06× |
| bfloat16 | — / **0.125** / — | — / **0.228** / — | — / **0.082** / — | — / **0.074** / — |
| float8_e4m3 | — / **0.051** / — | — / **0.271** / — | — / **0.067** / — | — / **0.041** / — |
| float8_e5m2 | — / **0.056** / — | — / **0.258** / — | — / **0.069** / — | — / **0.044** / — |
| int64 | 0.520 / **0.519** / 1.00× | 4.427 / **0.358** / 12.37× |  |  |
| int32 | 0.262 / **0.262** / 1.00× | 1.106 / **0.217** / 5.10× |  |  |
| int16 | 0.259 / **0.125** / 2.07× | 1.591 / **0.167** / 9.53× |  |  |
| int8 | 0.080 / **0.023** / 3.48× | 1.548 / **0.094** / 16.47× |  |  |
| int4 | — / **0.018** / — | — / **0.103** / — |  |  |
| int2 | — / **0.018** / — | — / **0.107** / — |  |  |
| uint64 | 0.999 / **0.519** / 1.92× | 4.370 / **0.362** / 12.07× |  |  |
| uint32 | 0.502 / **0.263** / 1.91× | 1.091 / **0.230** / 4.74× |  |  |
| uint16 | 0.140 / **0.125** / 1.12× | 1.330 / **0.181** / 7.35× |  |  |
| uint8 | 0.107 / **0.029** / 3.69× | 1.286 / **0.093** / 13.83× |  |  |
| uint2 | — / **0.017** / — | — / **0.109** / — |  |  |
| q8_0 | — / **0.986** / — | — / **0.861** / — | — / **0.777** / — | — / **0.778** / — |
| q4_0 | — / **0.658** / — | — / **0.784** / — | — / **0.456** / — | — / **0.456** / — |
| q4_k | — / **1.191** / — | — / **0.868** / — | — / **0.983** / — | — / **0.983** / — |
| q6_k | — / **1.398** / — | — / **0.955** / — | — / **1.196** / — | — / **1.198** / — |
| bool | — / **0.017** / — | — / **0.122** / — |  |  |

**CPU (1024³ product, 16M-value add, 4096×4096 rows)** — number / **adad** / × (× = number ÷ adad)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 6.950 / **6.910** / 1.01× | 18.234 / **3.290** / 5.54× | 6.203 / **5.593** / 1.11× | 4.141 / **4.703** / 0.88× |
| s32 |  |  |  |  |
| float32 | 3.039 / **3.033** / 1.00× | 3.977 / **1.598** / 2.49× | 4.239 / **1.860** / 2.28× | 1.807 / **1.675** / 1.08× |
| float16 | 7.566 / **1.166** / 6.49× | 867.638 / **1.247** / 695.78× | 54.353 / **3.318** / 16.38× | 18.926 / **1.231** / 15.37× |
| bfloat16 | — / **1.172** / — | — / **1.805** / — | — / **2.638** / — | — / **0.633** / — |
| float8_e4m3 | — / **1.800** / — | — / **1.235** / — | — / **3.090** / — | — / **1.357** / — |
| float8_e5m2 | — / **1.797** / — | — / **1.910** / — | — / **3.901** / — | — / **1.782** / — |
| int64 | 6.621 / **6.893** / 0.96× | 32.382 / **0.757** / 42.78× |  |  |
| int32 | 3.034 / **3.012** / 1.01× | 5.932 / **0.722** / 8.22× |  |  |
| int16 | 1.047 / **0.900** / 1.16× | 1.845 / **0.450** / 4.10× |  |  |
| int8 | 0.483 / **0.173** / 2.79× | 2.502 / **0.505** / 4.95× |  |  |
| int4 | — / **0.101** / — | — / **0.350** / — |  |  |
| int2 | — / **0.189** / — | — / **0.354** / — |  |  |
| uint64 | 7.053 / **6.893** / 1.02× | 33.098 / **0.822** / 40.27× |  |  |
| uint32 | 3.022 / **3.014** / 1.00× | 6.386 / **0.576** / 11.09× |  |  |
| uint16 | 1.241 / **0.861** / 1.44× | 2.004 / **0.471** / 4.25× |  |  |
| uint8 | 0.478 / **0.127** / 3.76× | 4.583 / **0.546** / 8.39× |  |  |
| uint2 | — / **0.242** / — | — / **0.477** / — |  |  |
| q8_0 | — / **5.194** / — | — / **8.913** / — | — / **6.232** / — | — / **6.215** / — |
| q4_0 | — / **4.253** / — | — / **8.700** / — | — / **4.079** / — | — / **2.673** / — |
| q4_k | — / **6.242** / — | — / **8.752** / — | — / **6.236** / — | — / **6.261** / — |
| q6_k | — / **10.344** / — | — / **10.107** / — | — / **11.383** / — | — / **5.618** / — |
| bool | — / **4.923** / — | — / **1.358** / — |  |  |

**Every family** — rows measured (operation × type)

| family | operations | CUDA rows | CPU rows |
|---|---:|---:|---:|
| core | 4 | 64 | 64 |
| map (element-wise formulas) | 60 | 899 | 815 |
| reductions, broadcasts, transpose, gemv | 15 | 330 | 300 |
| rows | 15 | 150 | 135 |
| layout and indexing | 20 | 380 | 345 |
| products on views | 6 | 112 | 112 |
| attention | 6 | 60 | 54 |
| optimizers | 3 | 30 | 27 |
| convolution | 2 | 32 | 29 |
| recurrent | 8 | 72 | 65 |
| routing, losses | 8 | 86 | 72 |

Where Adad is not ahead of `number` on the core rows: the memory-bound adds that move the same bytes (CUDA float32
and int32, 0.263 and 0.262 against the same; CPU int64, 6.89 against 6.62); and three float64 rows: CUDA softmax
(1.826 against 0.394, its error 1.13e-17 against 1.88e-10), CUDA rms_norm (0.549 against 0.372) and CPU rms_norm
(4.70 against 4.14). The CUDA float64 product is exact, through the sliced 8-bit tensor route where the device
measures it faster than its float64 lane.

The core tables above and the family table are taken from `docs/wip/baseline.md` (its `number` and Adad columns).
That file is the assembled bench and the regression baseline from now on: 4234 rows, every CUDA row and every CPU row
including q6_k and bool, each with its source. `COVERAGE.md`'s "Known slow" names the measured cells that are correct
but below what the device allows.

## Open

What the gates have not confirmed, or is not done, is in `COVERAGE.md` ("State" for the order, **todo** in the rows):
the known-slow
cells. The README claims nothing beyond what the gates and the regression
comparison have confirmed.

## License

Qiyas and Adad are licensed under the [Apache License, Version 2.0](LICENSE); see also [NOTICE](NOTICE).
Copyright (c) 2026 Ahmed Seada. Contributions are accepted under the [Contributor License Agreement](CLA.md);
[CONTRIBUTING.md](CONTRIBUTING.md) explains how to sign it and what a pull request must pass.
