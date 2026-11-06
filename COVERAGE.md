# Coverage

What Qiyas computes, in its own terms: one row per operation, one column per device. A cell is **✓** with the lane
rule the operation follows, **n/a** with the reason, or **todo**. Every ✓ has a float64 reference and an error column
in the bench, a correctness gate in `Checks` that runs over every type the operation applies to on both devices before
anything is timed, and a bench row at a realistic shape.

The type set: float64, float32, float16, bfloat16, float8 e4m3, float8 e5m2; int64, int32, int16, int8, int4, int2;
uint64, uint32, uint16, uint8, uint2; bool (1 bit); the block formats Q8_0, Q4_0, Q4_K, Q6_K (the GGUF layouts: values
in blocks with shared f16 scales). A block format's lane is float32; it is read a value at a time anywhere (both devices
decode by index) and written a block at a time: on the CPU every store re-encodes the blocks it touches; on the device
the operations that store value by value run in float32 storage and encode the result by blocks (Q8_0 and Q4_0 have
device encoders; Q4_K and Q6_K are encoded on the CPU, so on the device they are inputs — products, reductions, the
row statistics — with float32 or another type's outputs: their device encoders are todo).

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

Device buffers come from a size-keyed pool: after a shape's first call no driver allocation happens; `--report` measures
the cost of a rent/return pair (tens of nanoseconds). That pool is the arena the plan asked for; a bump-allocated scope
per operation would save the remaining lookup and is the step past it.

## Known slow

Measured cells that are correct but below what the device or the lanes allow; each is addressed at a natural point
and measured against the baseline when it is.

| where | what | measured |
|---|---|---|
| CPU | column-wise reductions and compares against row-wise | max:cols 3.68 ms vs max:rows 0.96 (float32, 4096 × 4096); sum:cols 1.44 vs 0.34; int8 max:cols 1.12 vs 0.074 |
| CPU | max / min / argmax slower than sum over the same bytes | float32 max:rows 0.96 vs sum:rows 0.34; min:rows 1.30 |
| CPU | erf and the formulas built on it | erf 17.8 ms, gelu_erf 41 ms, gelu_erf_bwd 28.5 ms per 16M float32 values (exp 1.74) |
| CPU | gemv_t (columns) against gemv (rows) | 9.4 vs 2.9 ms float32, 10.7 vs 1.9 float16 (8192 × 8192) |
| CUDA | 16-bit broadcasts below bandwidth | int16 maprow:mul 0.146 ms (440 GB/s) against int8 0.035 (900 GB/s) and int32 0.173 (740 GB/s) |
| CUDA | function-bearing two-input formulas in float16 | exp_bwd, tanh_bwd … 0.238 ms against add 0.126 (compute-bound f32 functions) |
| CPU | rope, scale_mask_softmax, log_softmax against the plain row operations | rope 2.9 ms, scale_mask_softmax 3.3, log_softmax 2.5 against rms_norm_affine 1.7 (float32, 4096 × 4096); float16 log_softmax 3.6 against layer_norm 0.9 |
| CPU | norm_params_bwd (column sums with row statistics) | 3.0–3.8 ms against 1.7 for a row pass over the same bytes |
| CUDA | three-stream row operations in float32 in a full run | scale_mask_softmax 0.50 ms in the group's run, 0.26 alone (a run-order effect also seen on add float8_e5m2) |
| CPU | the strided copy (a 3-d permutation of 16M values) | 16.2 ms float32, 7.3 int8 against 1.8 for a plain copy: a value-at-a-time gather (tiling by the source's rows is the step) |
| CPU | the pools and the depthwise convolution | maxpool_bwd 5.1 ms, adaptive_maxpool 5.8, depthwise_conv 4.8 (float32, 64 × 256 × 256); int8 adaptive_maxpool 18.6, depthwise 24.4 (value-at-a-time loads of a packed type) |
| CPU | the convolution backward | 33 ms against 6.9 forward (64 × 128 × 128 by 128 of 3 × 3): three passes in lane storage |
| CUDA | attention by the composition | 6.1 ms for 16 heads of 2048² at d = 128 (the scores through memory; a fused tile kernel through the tensor lanes is the step) |
| CUDA | attention backward by the composition | 4.0 ms for 8 heads of 1024² |
| CUDA | int8 adaptive_maxpool | 0.265 ms against float32 0.079 (per-value unpacking of 4 values a word in the window loop) |
| CPU | top-k per row | 113 ms float32, 305 int8 for top-8 of 4096 × 32768: k passes over the row (a partial sort is the step) |
| CPU | LSTM / GRU steps | 20–31 ms forward, 56–84 backward at B = D = H = 1024: the products dominate (four gates, two operands); a fused gates product is the step |

## Element-wise (map)

One kernel over `<TN, TLane, TL>` per device runs every formula; a formula takes up to three inputs and two scalars.
Backward formulas are formulas. Lane: **own** for the formulas every lane has, **float** for the rest.

| operation | CPU | CUDA | note |
|---|---|---|---|
| copy, fill, neg, abs, sign, square, relu, add_scalar, scale, affine, clamp | ✓ own | ✓ own | integer types modular at their width |
| add, sub, mul, div, max, min, axpy, muladd, where | ✓ own | ✓ own | `where`: x ≠ 0 takes y, else z; integer div truncates |
| square_bwd, abs_bwd, relu_bwd, clamp_bwd, where_bwd | ✓ own | ✓ own | (x, dy) → dx |
| sqrt, inv_sqrt, exp, expm1, log, sin, cos, tanh, sigmoid, erf | ✓ float | ✓ float | functions proven ≤ 4 ulp (exp 1, log 2, sin/cos 2, tanh/sigmoid 3, erf 4) |
| gelu (tanh form), gelu_erf, silu, leaky_relu, pow, dropout, gated_silu, gated_gelu | ✓ float | ✓ float | gelu as x·σ(2u), gelu_erf as x·Φ(x) by erfc: no cancellation |
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
| batched, strided, transposed operands (`ProductView`) | ✓ sum | ✓ sum | each operand of a batch made contiguous by the exact strided copy, then the product kernel (same lanes, bounds, proofs); fusing the transposition into the packing: todo |
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
| rows longer than 16 words a thread of 1024 (16384 float32 values) in the row family | ✓ | ✓ | the row in segments of the registers' span, reloaded in every phase (the per-value temporaries recomputed) |

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
| scatter-add rows (the embedding gradient) | ✓ sum | ✓ sum | each target row sums its sources in source order (deterministic); the device orders sources by counts, offsets and ranks (quadratic in the source count: a radix step for millions is todo) |
| im2col / col2im | ✓ own / ✓ sum | ✓ own / ✓ sum | strides, pads, dilations; col2im gathers each input pixel's column entries in kernel order |
| interpolate: nearest, bilinear (half-pixel centers) | ✓ own / ✓ float | ✓ own / ✓ float | nearest by integer arithmetic (exact); bilinear in the float lane |

## Attention

| operation | CPU | CUDA | note |
|---|---|---|---|
| rows of queries: O = softmax(scale·Q·Kᵀ, causal or spans)·V per head | ✓ float | ✓ float | the composition of Qiyas's own operations in lane storage (a narrow type widened once, O narrowed once): the view product, scale_mask_softmax with the key limit per row, the view product |
| decode: a few queries over a long cache | ✓ float | ✓ float | the device's fused kernel: a block per query, keys in chunks, the softmax online, K and V read once; the CPU per query in the lane |
| backward: dQ, dK, dV | ✓ float | ✓ float | the composition: P recomputed, dV = Pᵀ·dO, dP = dO·Vᵀ, dS = softmax_bwd, dQ = scale·dS·K, dK = scale·dSᵀ·Q |
| fused rows (flash: tiles of keys through the tensor lanes without the scores in memory) | todo | todo | the composition keeps Lq × Lk scores per head in lane storage |
| spans as a key range per query: [start, limit) | ✓ | ✓ | the limit by scale_mask_softmax's position, the start by an additive mask row (the composition) or the decode kernel's first chunk |

## Convolution and pooling

| operation | CPU | CUDA | note |
|---|---|---|---|
| conv forward (one image; strides, pads, dilations) | ✓ sum | ✓ sum | the composition im2col → product: the product's lanes and bounds; a fused direct convolution: todo |
| conv backward input, backward weight | ✓ float | ✓ float | the composition in the float lane's storage (the columns' gradient is a sum of products no narrow type holds between the passes); integer types: n/a, a gradient needs a wider type |
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
| LSTM sequences (T steps: h, c every step) and their backward (dx, dh0, dc0, dW, dU, db) | ✓ float | ✓ float | a loop of steps; the backward recomputes the forward and walks the steps back, the weights' gradients summed; a GRU sequence is the same loop of its step (its bench row: todo) |

## Quantized and block formats

| operation | CPU | CUDA | note |
|---|---|---|---|
| Q8_0, Q4_0, Q4_K, Q6_K as types (Load decodes blocks into the f32 lane, Store encodes blocks) | ✓ | ✓ decode; encode Q8_0, Q4_0 | the GGUF layouts; every operation above runs on them (the device's value-wise stores through float32 storage and a block encode) |
| products with a block-format operand | ✓ sum | ✓ sum | matmul through the f32 lane (the values decoded); gemv on Q8_0 / Q4_0 by the block-scaled integer dot (32-term s8 dots per block, the block scales' product in f32: the format's own fast path, `s8dp blocks`); the same for the tiled product: todo |

## Masks

| operation | CPU | CUDA | note |
|---|---|---|---|
| bool as a type (1 bit, 8 a byte); and / or / not are mul / max / (1 − x) in its lane; mask-select is `where` | ✓ own | ✓ own | the packed-integer machinery at 1 bit; `div` is n/a on bool (a divisor of 0 is half its values) |

## Routing, sampling, losses

| operation | CPU | CUDA | note |
|---|---|---|---|
| top-k per row (values as the type, indices s32; largest first, ties by the lower index) | ✓ own | ✓ own | k passes of the row's argmax (k ≤ 256 on the device; a partial sort is the step for large k) |
| pick per row (x[r][index[r]]) | ✓ own | ✓ own | |
| sampling per row (the first column whose running sum reaches u·Σ) | ✓ float | ✓ float | chunk sums, their prefix, the least crossing |
| route: counts, offsets, the sources ordered by bucket | ✓ | ✓ | the scatter-add's ordering kernels (deterministic; quadratic in the source count on the device: a radix step for millions) |
| cross-entropy rows: the loss (log_softmax → pick → neg), its backward (softmax − one_hot, a row operation) | ✓ float | ✓ float | |
| expert products (a product per bucket over the routed rows) | ✓ | ✓ | gather_rows by the order, the batched product on views, scatter-add rows back: the composition |
| ctc | todo | todo | |
