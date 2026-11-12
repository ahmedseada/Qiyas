# Qiyas bench

Assembled 2026-11-07 for the end of item 6. Median wall time (ms). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

The capped end-of-item-6 full run was stopped at the owner's request at 03:09 (invocations 1 and 2 complete, invocation 3
partial, invocation 4 not run). This file takes each subject's newest reading of each cell from that run and from the
session's single-family runs of code that is now in main (a reading of code changed after it is not used); every cell
no such run measured keeps the item 4 run's reading, marked "from the item 4 run" in the source column. Readings from
different runs on one machine (Intel Core Ultra 9 285, RTX 5070 Ti); device bandwidth rows follow the driver's P-state
(COVERAGE: the three-stream row).

| device | operation | type | shape | number ms | number lane | number err | qiyas ms | qiyas lane | qiyas err | source |
|---|---|---|---|---:|---|---:|---:|---|---:|---|
| cuda | add | float64 | 16777216 | 1.000 | add_f64 | 0e+0 | 0.520 | f64 | 0e+0 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | float64 | 2048x2048x2048 | 23.142 | gemm64_f64 | 2.59e-13 | 4.149 | f64x | 0e+0 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:fill | float64 | — | — | — | — | 0.165 | f64 | 0e+0 | from the item 4 run |
| cuda | map:neg | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:abs | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:sign | float64 | — | — | — | — | 0.359 | f64 | 0e+0 | from the item 4 run |
| cuda | map:square | float64 | — | — | — | — | 0.359 | f64 | 0e+0 | from the item 4 run |
| cuda | map:relu | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:scale | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:affine | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:clamp | float64 | — | — | — | — | 0.359 | f64 | 0e+0 | from the item 4 run |
| cuda | map:add | float64 | — | — | — | — | 0.520 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:sub | float64 | — | — | — | — | 1.000 | f64 | 0e+0 | from the item 4 run |
| cuda | map:mul | float64 | — | — | — | — | 0.999 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:div | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:max | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:min | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:axpy | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:muladd | float64 | — | — | — | — | 1.311 | f64 | 0e+0 | from the item 4 run |
| cuda | map:where | float64 | — | — | — | — | 0.683 | f64 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | float64 | — | — | — | — | 0.520 | f64 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | float64 | — | — | — | — | 0.520 | f64 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | float64 | — | — | — | — | 0.520 | f64 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | float64 | — | — | — | — | 0.520 | f64 | 0e+0 | from the item 4 run |
| cuda | map:sqrt | float64 | — | — | — | — | 0.399 | f64 | 0e+0 | from the item 4 run |
| cuda | map:inv_sqrt | float64 | — | — | — | — | 0.836 | f64 | 0e+0 | from the item 4 run |
| cuda | map:exp | float64 | — | — | — | — | 1.231 | f64 | 7.11e-15 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:expm1 | float64 | — | — | — | — | 1.226 | f64 | 1.11e-16 | from the item 4 run |
| cuda | map:log | float64 | — | — | — | — | 1.670 | f64 | 2.22e-16 | from the item 4 run |
| cuda | map:sin | float64 | — | — | — | — | 1.536 | f64 | 2.22e-16 | from the item 4 run |
| cuda | map:cos | float64 | — | — | — | — | 1.587 | f64 | 1.11e-16 | from the item 4 run |
| cuda | map:tanh | float64 | — | — | — | — | 1.814 | f64 | 3.33e-16 | from the item 4 run |
| cuda | map:sigmoid | float64 | — | — | — | — | 1.721 | f64 | 2.22e-16 | from the item 4 run |
| cuda | map:erf | float64 | — | — | — | — | 47.679 | f64 | 6.66e-16 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu | float64 | — | — | — | — | 1.971 | f64 | 8.88e-16 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu_erf | float64 | — | — | — | — | 177.879 | f64 | 4.44e-16 | from the item 4 run |
| cuda | map:silu | float64 | — | — | — | — | 1.770 | f64 | 8.88e-16 | from the item 4 run |
| cuda | map:leaky_relu | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | map:pow | float64 | — | — | — | — | 3.388 | f64 | 2.84e-14 | from the item 4 run |
| cuda | map:dropout | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:gated_silu | float64 | — | — | — | — | 1.823 | f64 | 3.55e-15 | from the item 4 run |
| cuda | map:gated_gelu | float64 | — | — | — | — | 2.024 | f64 | 3.55e-15 | from the item 4 run |
| cuda | map:sqrt_bwd | float64 | — | — | — | — | 0.823 | f64 | 0e+0 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | float64 | — | — | — | — | 1.310 | f64 | 1.78e-15 | from the item 4 run |
| cuda | map:exp_bwd | float64 | — | — | — | — | 1.283 | f64 | 2.84e-14 | from the item 4 run |
| cuda | map:log_bwd | float64 | — | — | — | — | 0.521 | f64 | 0e+0 | from the item 4 run |
| cuda | map:sin_bwd | float64 | — | — | — | — | 1.638 | f64 | 4.44e-16 | from the item 4 run |
| cuda | map:cos_bwd | float64 | — | — | — | — | 1.636 | f64 | 4.44e-16 | from the item 4 run |
| cuda | map:tanh_bwd | float64 | — | — | — | — | 3.684 | f64 | 3.11e-15 | from the item 4 run |
| cuda | map:sigmoid_bwd | float64 | — | — | — | — | 3.479 | f64 | 4.44e-16 | from the item 4 run |
| cuda | map:silu_bwd | float64 | — | — | — | — | 3.526 | f64 | 2.66e-15 | from the item 4 run |
| cuda | map:gelu_bwd | float64 | — | — | — | — | 3.921 | f64 | 1.78e-15 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu_erf_bwd | float64 | — | — | — | — | 181.093 | f64 | 1.78e-15 | from the item 4 run |
| cuda | map:leaky_relu_bwd | float64 | — | — | — | — | 0.520 | f64 | 0e+0 | from the item 4 run |
| cuda | map:pow_bwd | float64 | — | — | — | — | 3.540 | f64 | 4.26e-14 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | float64 | — | — | — | — | 3.576 | f64 | 7.11e-15 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | float64 | — | — | — | — | 1.822 | f64 | 3.55e-15 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | float64 | — | — | — | — | 3.974 | f64 | 7.11e-15 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | float64 | — | — | — | — | 2.024 | f64 | 3.55e-15 | from the item 4 run |
| cuda | sum:whole | float64 | — | — | — | — | 0.175 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sum:rows | float64 | — | — | — | — | 0.171 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sum:cols | float64 | — | — | — | — | 0.336 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sumsq:rows | float64 | — | — | — | — | 0.171 | f64 | 1.98e-11 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:whole | float64 | — | — | — | — | 0.175 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:rows | float64 | — | — | — | — | 0.171 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:cols | float64 | — | — | — | — | 0.337 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | min:rows | float64 | — | — | — | — | 0.171 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | argmax:rows | float64 | — | — | — | — | 0.171 | s32 index | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | argmax:cols | float64 | — | — | — | — | 0.632 | s32 index | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | maprow:add | float64 | — | — | — | — | 0.357 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | maprow:mul | float64 | — | — | — | — | 0.357 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | transpose | float64 | — | — | — | — | 0.360 | f64 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | gemv | float64 | — | — | — | — | 0.656 | f64 | 6.11e-13 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | gemv_t | float64 | — | — | — | — | 0.655 | f64 | 6.39e-13 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:log_softmax | float64 | — | — | — | — | 2.071 | f64 | 7.11e-15 | from the item 4 run |
| cuda | row:softmax_bwd | float64 | — | — | — | — | 0.521 | f64 | 3.41e-13 | from the item 4 run |
| cuda | row:log_softmax_bwd | float64 | — | — | — | — | 1.529 | f64 | 2.84e-14 | from the item 4 run |
| cuda | row:rms_norm_affine | float64 | — | — | — | — | 0.652 | f64 | 2.22e-14 | from the item 4 run |
| cuda | row:rms_norm_bwd | float64 | — | — | — | — | 1.135 | f64 | 5.88e-15 | from the item 4 run |
| cuda | row:layer_norm | float64 | — | — | — | — | 0.968 | f64 | 1.15e-14 | from the item 4 run |
| cuda | row:layer_norm_bwd | float64 | — | — | — | — | 1.742 | f64 | 2.78e-15 | from the item 4 run |
| cuda | row:weight_norm | float64 | — | — | — | — | 0.492 | f64 | 3.12e-16 | from the item 4 run |
| cuda | row:rope | float64 | — | — | — | — | 3.887 | f64 | 2.51e-12 | from the item 4 run |
| cuda | row:scale_mask_softmax | float64 | — | — | — | — | 3.183 | f64 | 5.55e-17 | from the item 4 run |
| cuda | row:add_rms_norm | float64 | — | — | — | — | 1.364 | f64 | 2.49e-14 | from the item 4 run |
| cuda | row:add_layer_norm | float64 | — | — | — | — | 1.150 | f64 | 1.42e-14 | from the item 4 run |
| cuda | row:cross_entropy_bwd | float64 | — | — | — | — | 3.061 | f64 | 1.11e-16 | from the item 4 run |
| cuda | norm_params_bwd:rms | float64 | — | — | — | — | 0.991 | f64 | 7.18e-13 | from the item 4 run |
| cuda | norm_params_bwd:layer | float64 | — | — | — | — | 1.214 | f64 | 5.12e-13 | from the item 4 run |
| cuda | layout:strided | float64 | — | — | — | — | 0.698 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | float64 | — | — | — | — | 0.176 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | float64 | — | — | — | — | 0.341 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | float64 | — | — | — | — | 0.383 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | float64 | — | — | — | — | 0.460 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | float64 | — | — | — | — | 0.116 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | float64 | — | — | — | — | 0.105 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | float64 | — | — | — | — | 0.241 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_bilinear | float64 | — | — | — | — | 2.189 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:shift | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | float64 | — | — | — | — | 0.358 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | float64 | — | — | — | — | 0.023 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool | float64 | — | — | — | — | 0.049 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | float64 | — | — | — | — | 0.155 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool_bwd | float64 | — | — | — | — | 0.253 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | float64 | — | — | — | — | 0.134 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_avgpool | float64 | — | — | — | — | 0.104 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | float64 | — | — | — | — | 0.124 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | float64 | — | — | — | — | 34.208 | f64 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | float64 | — | — | — | — | 0.231 | f64 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | float64 | 2048x2048x2048 Bᵀ | — | — | — | 4.386 | f64x | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | float64 | 2048x2048x2048 Aᵀ | — | — | — | 4.345 | f64x | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | float64 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 4.416 | f64x | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | float64 | 16 x 512x512x512 | — | — | — | 6.563 | f64 | 6.04e-14 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | float64 | 2048x2048x2048 (dA, dB) | — | — | — | 8.749 | f64x | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | float64 | 16h 2048x2048 d128 | — | — | — | 60.967 | f64 | 4.58e-16 | 6k CUDA float64 attention / conv |
| cuda | attention_causal | float64 | 16h 2048x2048 d128 | — | — | — | 32.024 | f64 | 4.51e-16 | 6k CUDA float64 attention / conv |
| cuda | attention_fused | float64 | 16h 2048x2048 d128 | — | — | — | 60.868 | f64 | 4.58e-16 | 6k CUDA float64 attention / conv |
| cuda | attention_fused_causal | float64 | 16h 2048x2048 d128 | — | — | — | 31.942 | f64 | 4.51e-16 | 6k CUDA float64 attention / conv |
| cuda | attention_decode | float64 | 32h 1x4096 d128 | — | — | — | 1.169 | f64 | 2.15e-16 | 6k CUDA float64 attention / conv |
| cuda | attention_bwd | float64 | 8h 1024x1024 d128 | — | — | — | 16.555 | f64 | 1.46e-16 | 6k CUDA float64 attention / conv |
| cuda | optimizer:sgd_momentum | float64 | — | — | — | — | 1.575 | f64; + restore of p, m, v | 1.11e-16 | from the item 4 run |
| cuda | optimizer:adam | float64 | — | — | — | — | 2.342 | f64; + restore of p, m, v | 1.11e-16 | from the item 4 run |
| cuda | optimizer:adamw | float64 | — | — | — | — | 2.448 | f64; + restore of p, m, v | 1.11e-16 | from the item 4 run |
| cuda | conv | float64 | 64x128x128 by 128 of 3x3 | — | — | — | 2.542 | f64x | 0e+0 | 6k CUDA float64 attention / conv |
| cuda | conv_bwd | float64 | 64x128x128 by 128 of 3x3 | — | — | — | 15.717 | f64 | 0e+0 | 6k CUDA float64 attention / conv |
| cuda | lstm_step | float64 | — | — | — | — | 6.120 | f64 | 8.05e-16 | from the item 4 run |
| cuda | gru_step | float64 | — | — | — | — | 6.036 | f64 | 1.21e-15 | from the item 4 run |
| cuda | lstm_step_bwd | float64 | — | — | — | — | 19.450 | f64 | 3.44e-15 | from the item 4 run |
| cuda | gru_step_bwd | float64 | — | — | — | — | 16.438 | f64 | 1.44e-15 | from the item 4 run |
| cuda | lstm_seq | float64 | — | — | — | — | 23.812 | f64 | 4.09e-16 | from the item 4 run |
| cuda | lstm_seq_bwd | float64 | — | — | — | — | 113.888 | f64 | 4.44e-16 | from the item 4 run |
| cuda | gru_seq | float64 | — | — | — | — | 22.446 | f64 | 8.33e-16 | from the item 4 run |
| cuda | gru_seq_bwd | float64 | — | — | — | — | 98.492 | f64 | 1.17e-15 | from the item 4 run |
| cuda | topk_rows | float64 | — | — | — | — | 20.735 | f64 | 0e+0 | from the item 4 run |
| cuda | pick_rows | float64 | — | — | — | — | 0.009 | f64 | 0e+0 | from the item 4 run |
| cuda | sample_rows | float64 | — | — | — | — | 7.621 | f64 | 0e+0 | from the item 4 run |
| cuda | route | s32 | — | — | — | — | 0.275 | s32 | 0e+0 | from the item 4 run |
| cuda | route_wide | s32 | — | — | — | — | 0.186 | s32 | 0e+0 | from the item 4 run |
| cuda | cross_entropy | float64 | — | — | — | — | 175.411 | f64 (log_softmax, pick, neg) | 1.78e-14 | from the item 4 run |
| cuda | ctc | float64 | — | — | — | — | 2.398 | f64 | 4.55e-13 | from the item 4 run |
| cuda | ctc_bwd | float64 | — | — | — | — | 5.242 | f64 | 6.8e-13 | from the item 4 run |
| cuda | softmax | float64 | 4096x4096 | 0.394 | softmax_f64 | 1.88e-10 | 1.826 | f64 | 1.13e-17 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | float64 | 4096x4096 | 0.372 | rms_norm_f64 | 1.24e-14 | 0.549 | f64 | 1.07e-14 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | float32 | 16777216 | 0.263 | add_f32 | 5.96e-8 | 0.263 | f32 | 5.96e-8 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | float32 | 2048x2048x2048 | 0.842 | gemm128_f32 | 1.62e-4 | 0.681 | f32 | 1.62e-4 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | float32 | — | — | — | — | 0.182 | f32 | 0e+0 | from the item 4 run |
| cuda | map:fill | float32 | — | — | — | — | 0.084 | f32 | 0e+0 | from the item 4 run |
| cuda | map:neg | float32 | — | — | — | — | 0.180 | f32 | 0e+0 | from the item 4 run |
| cuda | map:abs | float32 | — | — | — | — | 0.179 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sign | float32 | — | — | — | — | 0.181 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square | float32 | — | — | — | — | 0.181 | f32 | 4.77e-7 | from the item 4 run |
| cuda | map:relu | float32 | — | — | — | — | 0.181 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | float32 | — | — | — | — | 0.182 | f32 | 2.38e-7 | from the item 4 run |
| cuda | map:scale | float32 | — | — | — | — | 0.181 | f32 | 4.77e-7 | from the item 4 run |
| cuda | map:affine | float32 | — | — | — | — | 0.179 | f32 | 4.77e-7 | from the item 4 run |
| cuda | map:clamp | float32 | — | — | — | — | 0.181 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add | float32 | — | — | — | — | 0.262 | f32 | 2.38e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:sub | float32 | — | — | — | — | 0.263 | f32 | 2.38e-7 | from the item 4 run |
| cuda | map:mul | float32 | — | — | — | — | 0.263 | f32 | 4.77e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:div | float32 | — | — | — | — | 0.262 | f32 | 1.19e-7 | from the item 4 run |
| cuda | map:max | float32 | — | — | — | — | 0.262 | f32 | 0e+0 | from the item 4 run |
| cuda | map:min | float32 | — | — | — | — | 0.262 | f32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | float32 | — | — | — | — | 0.262 | f32 | 4.77e-7 | from the item 4 run |
| cuda | map:muladd | float32 | — | — | — | — | 0.343 | f32 | 9.54e-7 | from the item 4 run |
| cuda | map:where | float32 | — | — | — | — | 0.262 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | float32 | — | — | — | — | 0.263 | f32 | 9.54e-7 | from the item 4 run |
| cuda | map:abs_bwd | float32 | — | — | — | — | 0.262 | f32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | float32 | — | — | — | — | 0.262 | f32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | float32 | — | — | — | — | 0.262 | f32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | float32 | — | — | — | — | 0.502 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sqrt | float32 | — | — | — | — | 0.177 | f32 | 5.96e-8 | from the item 4 run |
| cuda | map:inv_sqrt | float32 | — | — | — | — | 0.180 | f32 | 1.37e-7 | from the item 4 run |
| cuda | map:exp | float32 | — | — | — | — | 0.178 | f32 | 3.42e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:expm1 | float32 | — | — | — | — | 0.344 | f32 | 6.84e-8 | from the item 4 run |
| cuda | map:log | float32 | — | — | — | — | 0.181 | f32 | 1.26e-7 | from the item 4 run |
| cuda | map:sin | float32 | — | — | — | — | 0.181 | f32 | 7.99e-8 | from the item 4 run |
| cuda | map:cos | float32 | — | — | — | — | 0.181 | f32 | 7.96e-8 | from the item 4 run |
| cuda | map:tanh | float32 | — | — | — | — | 0.343 | f32 | 1.49e-7 | from the item 4 run |
| cuda | map:sigmoid | float32 | — | — | — | — | 0.181 | f32 | 8.77e-8 | from the item 4 run |
| cuda | map:erf | float32 | — | — | — | — | 0.609 | f32 | 1.45e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu | float32 | — | — | — | — | 0.175 | f32 | 2.42e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu_erf | float32 | — | — | — | — | 0.987 | f32 | 2.37e-7 | from the item 4 run |
| cuda | map:silu | float32 | — | — | — | — | 0.181 | f32 | 4.58e-7 | from the item 4 run |
| cuda | map:leaky_relu | float32 | — | — | — | — | 0.181 | f32 | 1.79e-8 | from the item 4 run |
| cuda | map:pow | float32 | — | — | — | — | 0.177 | f32 | 9.47e-6 | from the item 4 run |
| cuda | map:dropout | float32 | — | — | — | — | 0.263 | f32 | 2.38e-7 | from the item 4 run |
| cuda | map:gated_silu | float32 | — | — | — | — | 0.263 | f32 | 2.24e-6 | from the item 4 run |
| cuda | map:gated_gelu | float32 | — | — | — | — | 0.502 | f32 | 2.04e-6 | from the item 4 run |
| cuda | map:sqrt_bwd | float32 | — | — | — | — | 0.261 | f32 | 2.29e-7 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | float32 | — | — | — | — | 0.261 | f32 | 1.04e-6 | from the item 4 run |
| cuda | map:exp_bwd | float32 | — | — | — | — | 0.262 | f32 | 1.95e-5 | from the item 4 run |
| cuda | map:log_bwd | float32 | — | — | — | — | 0.262 | f32 | 2.38e-7 | from the item 4 run |
| cuda | map:sin_bwd | float32 | — | — | — | — | 0.262 | f32 | 4.04e-7 | from the item 4 run |
| cuda | map:cos_bwd | float32 | — | — | — | — | 0.262 | f32 | 4e-7 | from the item 4 run |
| cuda | map:tanh_bwd | float32 | — | — | — | — | 0.262 | f32 | 9.3e-7 | from the item 4 run |
| cuda | map:sigmoid_bwd | float32 | — | — | — | — | 0.502 | f32 | 2.31e-7 | from the item 4 run |
| cuda | map:silu_bwd | float32 | — | — | — | — | 0.263 | f32 | 9.79e-7 | from the item 4 run |
| cuda | map:gelu_bwd | float32 | — | — | — | — | 0.261 | f32 | 8.92e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu_erf_bwd | float32 | — | — | — | — | 1.008 | f32 | 8.35e-7 | from the item 4 run |
| cuda | map:leaky_relu_bwd | float32 | — | — | — | — | 0.262 | f32 | 1.79e-8 | from the item 4 run |
| cuda | map:pow_bwd | float32 | — | — | — | — | 0.262 | f32 | 2.48e-5 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | float32 | — | — | — | — | 0.345 | f32 | 4.11e-6 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | float32 | — | — | — | — | 0.262 | f32 | 2.12e-6 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | float32 | — | — | — | — | 0.345 | f32 | 2.92e-6 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | float32 | — | — | — | — | 0.262 | f32 | 2.07e-6 | from the item 4 run |
| cuda | sum:whole | float32 | — | — | — | — | 0.099 | f32 | 4.87e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sum:rows | float32 | — | — | — | — | 0.089 | f32 | 5.64e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sum:cols | float32 | — | — | — | — | 0.096 | f32 | 3.81e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sumsq:rows | float32 | — | — | — | — | 0.089 | f32 | 6.22e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:whole | float32 | — | — | — | — | 0.093 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:rows | float32 | — | — | — | — | 0.089 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:cols | float32 | — | — | — | — | 0.096 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | min:rows | float32 | — | — | — | — | 0.089 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | argmax:rows | float32 | — | — | — | — | 0.089 | s32 index | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | argmax:cols | float32 | — | — | — | — | 0.510 | s32 index | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | maprow:add | float32 | — | — | — | — | 0.174 | f32 | 5.96e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | maprow:mul | float32 | — | — | — | — | 0.182 | f32 | 2.98e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | transpose | float32 | — | — | — | — | 0.192 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | gemv | float32 | — | — | — | — | 0.336 | f32 | 3.96e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | gemv_t | float32 | — | — | — | — | 0.335 | f32 | 4.99e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:log_softmax | float32 | — | — | — | — | 0.179 | f32 | 1.27e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:softmax_bwd | float32 | — | — | — | — | 0.263 | f32 | 1.2e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:log_softmax_bwd | float32 | — | — | — | — | 0.262 | f32 | 2.27e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:rms_norm_affine | float32 | — | — | — | — | 0.179 | f32 | 6.2e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:rms_norm_bwd | float32 | — | — | — | — | 0.263 | f32 | 1.72e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:layer_norm | float32 | — | — | — | — | 0.182 | f32 | 7.26e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:layer_norm_bwd | float32 | — | — | — | — | 0.262 | f32 | 1.75e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:weight_norm | float32 | — | — | — | — | 0.180 | f32 | 7.22e-9 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:rope | float32 | — | — | — | — | 0.181 | f32 | 1.35e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:scale_mask_softmax | float32 | — | — | — | — | 0.263 | f32 | 4.81e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:add_rms_norm | float32 | — | — | — | — | 0.358 | f32 | 9.44e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:add_layer_norm | float32 | — | — | — | — | 0.359 | f32 | 1.13e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:cross_entropy_bwd | float32 | — | — | — | — | 0.175 | f32 | 2.98e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | norm_params_bwd:rms | float32 | — | — | — | — | 0.260 | f32 | 3.16e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | norm_params_bwd:layer | float32 | — | — | — | — | 0.265 | f32 | 3.51e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:strided | float32 | — | — | — | — | 0.430 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:one_hot | float32 | — | — | — | — | 0.134 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:gather_rows | float32 | — | — | — | — | 0.195 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:gather_cols | float32 | — | — | — | — | 0.253 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:scatter_add_rows | float32 | — | — | — | — | 0.305 | f32 | 4.77e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:im2col | float32 | — | — | — | — | 0.087 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:col2im | float32 | — | — | — | — | 0.048 | f32 | 1.08e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:interpolate_nearest | float32 | — | — | — | — | 0.151 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:interpolate_bilinear | float32 | — | — | — | — | 0.245 | f32 | 1.19e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:shift | float32 | — | — | — | — | 0.343 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:pad2d | float32 | — | — | — | — | 0.182 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:maxpool | float32 | — | — | — | — | 0.021 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:avgpool | float32 | — | — | — | — | 0.021 | f32 | 7.45e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:maxpool_bwd | float32 | — | — | — | — | 0.089 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:avgpool_bwd | float32 | — | — | — | — | 0.110 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_maxpool | float32 | — | — | — | — | 0.079 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_avgpool | float32 | — | — | — | — | 0.079 | f32 | 7.4e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:depthwise_conv | float32 | — | — | — | — | 0.095 | f32 | 6.56e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_maxpool_bwd | float32 | — | — | — | — | 0.230 | f32 | 5.96e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_avgpool_bwd | float32 | — | — | — | — | 0.161 | f32 | 2.08e-10 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | matmul_nt | float32 | 2048x2048x2048 Bᵀ | — | — | — | 0.822 | f32 | 1.57e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | float32 | 2048x2048x2048 Aᵀ | — | — | — | 0.835 | f32 | 1.51e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | float32 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.862 | f32 | 1.54e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | float32 | 16 x 512x512x512 | — | — | — | 0.969 | f32 | 3.9e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | float32 | 2048x2048x2048 (dA, dB) | — | — | — | 1.653 | f32 | 1.54e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | float32 | — | — | — | — | 3.792 | f32 | 1.39e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_causal | float32 | — | — | — | — | 2.106 | f32 | 1.46e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_fused | float32 | — | — | — | — | 3.789 | f32 | 1.39e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_fused_causal | float32 | — | — | — | — | 2.097 | f32 | 1.46e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_decode | float32 | — | — | — | — | 0.608 | f32 | 7.05e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_bwd | float32 | — | — | — | — | 1.557 | f32 | 1.47e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | optimizer:sgd_momentum | float32 | — | — | — | — | 0.790 | f32; + restore of p, m, v | 5.98e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | optimizer:adam | float32 | — | — | — | — | 1.141 | f32; + restore of p, m, v | 5.95e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | optimizer:adamw | float32 | — | — | — | — | 1.140 | f32; + restore of p, m, v | 7.29e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | conv | float32 | — | — | — | — | 0.141 | f32 | 4.36e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | conv_bwd | float32 | — | — | — | — | 1.068 | f32 | 1.56e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | lstm_step | float32 | — | — | — | — | 1.068 | f32 | 5.76e-7 | from the item 4 run |
| cuda | gru_step | float32 | — | — | — | — | 0.803 | f32 | 7.78e-7 | from the item 4 run |
| cuda | lstm_step_bwd | float32 | — | — | — | — | 2.699 | f32 | 1.81e-6 | from the item 4 run |
| cuda | gru_step_bwd | float32 | — | — | — | — | 2.093 | f32 | 8.33e-7 | from the item 4 run |
| cuda | lstm_seq | float32 | — | — | — | — | 2.744 | f32 | 2.76e-7 | from the item 4 run |
| cuda | lstm_seq_bwd | float32 | — | — | — | — | 26.048 | f32 | 9.1e-7 | from the item 4 run |
| cuda | gru_seq | float32 | — | — | — | — | 2.543 | f32 | 5.31e-7 | from the item 4 run |
| cuda | gru_seq_bwd | float32 | — | — | — | — | 12.485 | f32 | 6.81e-7 | from the item 4 run |
| cuda | topk_rows | float32 | — | — | — | — | 4.251 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | pick_rows | float32 | — | — | — | — | 0.011 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sample_rows | float32 | — | — | — | — | 1.697 | f32 | 1e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | cross_entropy | float32 | — | — | — | — | 1.266 | f32 | 1.57e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | ctc | float32 | — | — | — | — | 0.204 | f32 | 1.36e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | ctc_bwd | float32 | — | — | — | — | 0.421 | f32 | 1.67e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | softmax | float32 | 4096x4096 | 0.181 | softmax_f32 | 5.11e-10 | 0.180 | f32 | 3.68e-10 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | float32 | 4096x4096 | 0.184 | rms_norm_f32 | 2.93e-7 | 0.180 | f32 | 2.97e-7 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | float16 | 16777216 | 0.139 | add_f16 | 4.88e-4 | 0.125 | f32 | 4.88e-4 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | float16 | 2048x2048x2048 | 0.654 | gemm128_f16 | 1.51e+0 | 0.229 | f16mma | 3.09e-2 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | float16 | — | — | — | — | 0.082 | f32 | 0e+0 | from the item 4 run |
| cuda | map:fill | float16 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | map:neg | float16 | — | — | — | — | 0.084 | f32 | 0e+0 | from the item 4 run |
| cuda | map:abs | float16 | — | — | — | — | 0.085 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sign | float16 | — | — | — | — | 0.085 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square | float16 | — | — | — | — | 0.085 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:relu | float16 | — | — | — | — | 0.084 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | float16 | — | — | — | — | 0.084 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:scale | float16 | — | — | — | — | 0.085 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:affine | float16 | — | — | — | — | 0.087 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:clamp | float16 | — | — | — | — | 0.085 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add | float16 | — | — | — | — | 0.125 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:sub | float16 | — | — | — | — | 0.125 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:mul | float16 | — | — | — | — | 0.126 | f32 | 3.91e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:div | float16 | — | — | — | — | 0.126 | f32 | 9.76e-4 | from the item 4 run |
| cuda | map:max | float16 | — | — | — | — | 0.126 | f32 | 0e+0 | from the item 4 run |
| cuda | map:min | float16 | — | — | — | — | 0.126 | f32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | float16 | — | — | — | — | 0.126 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:muladd | float16 | — | — | — | — | 0.169 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:where | float16 | — | — | — | — | 0.169 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | float16 | — | — | — | — | 0.126 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:abs_bwd | float16 | — | — | — | — | 0.126 | f32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | float16 | — | — | — | — | 0.126 | f32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | float16 | — | — | — | — | 0.238 | f32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | float16 | — | — | — | — | 0.235 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sqrt | float16 | — | — | — | — | 0.086 | f32 | 4.88e-4 | from the item 4 run |
| cuda | map:inv_sqrt | float16 | — | — | — | — | 0.089 | f32 | 4.88e-4 | from the item 4 run |
| cuda | map:exp | float16 | — | — | — | — | 0.086 | f32 | 1.55e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:expm1 | float16 | — | — | — | — | 0.087 | f32 | 2.44e-4 | from the item 4 run |
| cuda | map:log | float16 | — | — | — | — | 0.095 | f32 | 4.88e-4 | from the item 4 run |
| cuda | map:sin | float16 | — | — | — | — | 0.086 | f32 | 2.44e-4 | from the item 4 run |
| cuda | map:cos | float16 | — | — | — | — | 0.086 | f32 | 2.44e-4 | from the item 4 run |
| cuda | map:tanh | float16 | — | — | — | — | 0.095 | f32 | 2.44e-4 | from the item 4 run |
| cuda | map:sigmoid | float16 | — | — | — | — | 0.094 | f32 | 2.44e-4 | from the item 4 run |
| cuda | map:erf | float16 | — | — | — | — | 0.611 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu | float16 | — | — | — | — | 0.098 | f32 | 9.73e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu_erf | float16 | — | — | — | — | 0.985 | f32 | 9.76e-4 | from the item 4 run |
| cuda | map:silu | float16 | — | — | — | — | 0.095 | f32 | 9.75e-4 | from the item 4 run |
| cuda | map:leaky_relu | float16 | — | — | — | — | 0.082 | f32 | 9.77e-5 | from the item 4 run |
| cuda | map:pow | float16 | — | — | — | — | 0.102 | f32 | 7.8e-3 | from the item 4 run |
| cuda | map:dropout | float16 | — | — | — | — | 0.126 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:gated_silu | float16 | — | — | — | — | 0.126 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:gated_gelu | float16 | — | — | — | — | 0.127 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:sqrt_bwd | float16 | — | — | — | — | 0.235 | f32 | 9.77e-4 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | float16 | — | — | — | — | 0.238 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:exp_bwd | float16 | — | — | — | — | 0.124 | f32 | 6.25e-2 | small cells 1: CUDA float16 formulas |
| cuda | map:log_bwd | float16 | — | — | — | — | 0.237 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:sin_bwd | float16 | — | — | — | — | 0.238 | f32 | 9.77e-4 | from the item 4 run |
| cuda | map:cos_bwd | float16 | — | — | — | — | 0.237 | f32 | 9.77e-4 | from the item 4 run |
| cuda | map:tanh_bwd | float16 | — | — | — | — | 0.129 | f32 | 9.77e-4 | small cells 1: CUDA float16 formulas |
| cuda | map:sigmoid_bwd | float16 | — | — | — | — | 0.238 | f32 | 2.44e-4 | from the item 4 run |
| cuda | map:silu_bwd | float16 | — | — | — | — | 0.235 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:gelu_bwd | float16 | — | — | — | — | 0.129 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | map:gelu_erf_bwd | float16 | — | — | — | — | 1.010 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:leaky_relu_bwd | float16 | — | — | — | — | 0.126 | f32 | 9.77e-5 | from the item 4 run |
| cuda | map:pow_bwd | float16 | — | — | — | — | 0.127 | f32 | 3.13e-2 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | float16 | — | — | — | — | 0.170 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | float16 | — | — | — | — | 0.235 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | float16 | — | — | — | — | 0.321 | f32 | 3.91e-3 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | float16 | — | — | — | — | 0.236 | f32 | 3.91e-3 | from the item 4 run |
| cuda | sum:whole | float16 | — | — | — | — | 0.033 | f32 | 2.97e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sum:rows | float16 | — | — | — | — | 0.023 | f32 | 3.81e-6 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sum:cols | float16 | — | — | — | — | 0.029 | f32 | 2.87e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sumsq:rows | float16 | — | — | — | — | 0.023 | f32 | 6.35e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:whole | float16 | — | — | — | — | 0.025 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:rows | float16 | — | — | — | — | 0.023 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | max:cols | float16 | — | — | — | — | 0.029 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | min:rows | float16 | — | — | — | — | 0.023 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | argmax:rows | float16 | — | — | — | — | 0.027 | s32 index | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | argmax:cols | float16 | — | — | — | — | 0.189 | s32 index | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | maprow:add | float16 | — | — | — | — | 0.086 | f32 | 4.88e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | maprow:mul | float16 | — | — | — | — | 0.087 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | transpose | float16 | — | — | — | — | 0.090 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | gemv | float16 | — | — | — | — | 0.176 | f32 | 3.12e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | gemv_t | float16 | — | — | — | — | 0.185 | f32 | 5.38e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:log_softmax | float16 | — | — | — | — | 0.078 | f32 | 3.91e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:softmax_bwd | float16 | — | — | — | — | 0.125 | f32 | 3.12e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:log_softmax_bwd | float16 | — | — | — | — | 0.125 | f32 | 3.13e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:rms_norm_affine | float16 | — | — | — | — | 0.077 | f32 | 9.77e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:rms_norm_bwd | float16 | — | — | — | — | 0.126 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:layer_norm | float16 | — | — | — | — | 0.081 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:layer_norm_bwd | float16 | — | — | — | — | 0.127 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:weight_norm | float16 | — | — | — | — | 0.077 | f32 | 1.53e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:rope | float16 | — | — | — | — | 0.093 | f32 | 2.53e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:scale_mask_softmax | float16 | — | — | — | — | 0.127 | f32 | 2.39e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:add_rms_norm | float16 | — | — | — | — | 0.182 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:add_layer_norm | float16 | — | — | — | — | 0.183 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | row:cross_entropy_bwd | float16 | — | — | — | — | 0.084 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | norm_params_bwd:rms | float16 | — | — | — | — | 0.102 | f32 | 3.15e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | norm_params_bwd:layer | float16 | — | — | — | — | 0.108 | f32 | 3.44e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:strided | float16 | — | — | — | — | 0.437 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:one_hot | float16 | — | — | — | — | 0.048 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:gather_rows | float16 | — | — | — | — | 0.090 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:gather_cols | float16 | — | — | — | — | 0.131 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:scatter_add_rows | float16 | — | — | — | — | 0.210 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:im2col | float16 | — | — | — | — | 0.067 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:col2im | float16 | — | — | — | — | 0.048 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:interpolate_nearest | float16 | — | — | — | — | 0.093 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:interpolate_bilinear | float16 | — | — | — | — | 0.138 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:shift | float16 | — | — | — | — | 0.161 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:pad2d | float16 | — | — | — | — | 0.159 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:maxpool | float16 | — | — | — | — | 0.021 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:avgpool | float16 | — | — | — | — | 0.021 | f32 | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:maxpool_bwd | float16 | — | — | — | — | 0.083 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:avgpool_bwd | float16 | — | — | — | — | 0.085 | f32 | 2.98e-8 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_maxpool | float16 | — | — | — | — | 0.067 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_avgpool | float16 | — | — | — | — | 0.067 | f32 | 1.53e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:depthwise_conv | float16 | — | — | — | — | 0.100 | f32 | 1.95e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_maxpool_bwd | float16 | — | — | — | — | 0.220 | f32 | 4.88e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | layout:adaptive_avgpool_bwd | float16 | — | — | — | — | 0.169 | f32 | 9.44e-7 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | matmul_nt | float16 | 2048x2048x2048 Bᵀ | — | — | — | 0.219 | f16mma | 3.09e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | float16 | 2048x2048x2048 Aᵀ | — | — | — | 0.231 | f16mma | 3.08e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | float16 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.233 | f16mma | 3.11e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | float16 | 16 x 512x512x512 | — | — | — | 0.515 | f16mma | 1.56e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | float16 | 2048x2048x2048 (dA, dB) | — | — | — | 0.443 | f16mma | 3.1e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | float16 | — | — | — | — | 0.818 | f16mma | 2.27e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_causal | float16 | — | — | — | — | 0.505 | f16mma | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_fused | float16 | — | — | — | — | 0.817 | f16mma | 2.27e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_fused_causal | float16 | — | — | — | — | 0.508 | f16mma | 2.44e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_decode | float16 | — | — | — | — | 0.608 | f32 | 1.34e-5 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | attention_bwd | float16 | — | — | — | — | 0.469 | f16mma | 1.2e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | optimizer:sgd_momentum | float16 | — | — | — | — | 0.572 | f32; + restore of p, m, v | 4.88e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | optimizer:adam | float16 | — | — | — | — | 0.923 | f32; + restore of p, m, v | 4.88e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | optimizer:adamw | float16 | — | — | — | — | 0.924 | f32; + restore of p, m, v | 4.87e-4 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | conv | float16 | — | — | — | — | 0.138 | f16mma | 1.55e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | conv_bwd | float16 | — | — | — | — | 1.099 | f32 | 1.56e-2 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | lstm_step | float16 | — | — | — | — | 1.179 | f32 | 2.04e-4 | from the item 4 run |
| cuda | gru_step | float16 | — | — | — | — | 0.980 | f32 | 2.44e-4 | from the item 4 run |
| cuda | lstm_step_bwd | float16 | — | — | — | — | 2.901 | f32 | 2.43e-4 | from the item 4 run |
| cuda | gru_step_bwd | float16 | — | — | — | — | 2.506 | f32 | 2.41e-4 | from the item 4 run |
| cuda | lstm_seq | float16 | — | — | — | — | 3.787 | f32 | 1.7e-4 | from the item 4 run |
| cuda | lstm_seq_bwd | float16 | — | — | — | — | 14.141 | f32 | 2.39e-4 | from the item 4 run |
| cuda | gru_seq | float16 | — | — | — | — | 2.610 | f32 | 2.44e-4 | from the item 4 run |
| cuda | gru_seq_bwd | float16 | — | — | — | — | 18.643 | f32 | 2.44e-4 | from the item 4 run |
| cuda | topk_rows | float16 | — | — | — | — | 3.479 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | pick_rows | float16 | — | — | — | — | 0.007 | f32 | 0e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | sample_rows | float16 | — | — | — | — | 1.436 | f32 | 1e+0 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | cross_entropy | float16 | — | — | — | — | 0.641 | f32 | 7.8e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | ctc | float16 | — | — | — | — | 0.183 | f32 | 4.96e-1 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | ctc_bwd | float16 | — | — | — | — | 0.455 | f32 | 1.6e-3 | end-of-item-6 run, invocation 3 (stopped at 03:09: CUDA float32, float16, part of float64) |
| cuda | softmax | float16 | 4096x4096 | 0.092 | softmax_f16 | 3.3e-6 | 0.079 | f32 | 9.54e-7 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | float16 | 4096x4096 | 0.089 | rms_norm_f16 | 1.64e-3 | 0.084 | f32 | 4.88e-4 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | bfloat16 | 16777216 | — | — | — | 0.125 | f32 | 3.91e-3 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | bfloat16 | 2048x2048x2048 | — | — | — | 0.228 | bf16mma | 2.5e-1 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | bfloat16 | — | — | — | — | 0.085 | f32 | 0e+0 | from the item 4 run |
| cuda | map:fill | bfloat16 | — | — | — | — | 0.031 | f32 | 0e+0 | from the item 4 run |
| cuda | map:neg | bfloat16 | — | — | — | — | 0.087 | f32 | 0e+0 | from the item 4 run |
| cuda | map:abs | bfloat16 | — | — | — | — | 0.086 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sign | bfloat16 | — | — | — | — | 0.084 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square | bfloat16 | — | — | — | — | 0.084 | f32 | 3.1e-2 | from the item 4 run |
| cuda | map:relu | bfloat16 | — | — | — | — | 0.085 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | bfloat16 | — | — | — | — | 0.083 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:scale | bfloat16 | — | — | — | — | 0.084 | f32 | 3.13e-2 | from the item 4 run |
| cuda | map:affine | bfloat16 | — | — | — | — | 0.085 | f32 | 3.13e-2 | from the item 4 run |
| cuda | map:clamp | bfloat16 | — | — | — | — | 0.084 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add | bfloat16 | — | — | — | — | 0.233 | f32 | 1.56e-2 | small cells 1: CUDA bfloat16 formulas |
| cuda | map:sub | bfloat16 | — | — | — | — | 0.125 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:mul | bfloat16 | — | — | — | — | 0.126 | f32 | 3.13e-2 | from the item 4 run |
| cuda | map:div | bfloat16 | — | — | — | — | 0.127 | f32 | 7.78e-3 | from the item 4 run |
| cuda | map:max | bfloat16 | — | — | — | — | 0.127 | f32 | 0e+0 | from the item 4 run |
| cuda | map:min | bfloat16 | — | — | — | — | 0.127 | f32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | bfloat16 | — | — | — | — | 0.127 | f32 | 3.13e-2 | from the item 4 run |
| cuda | map:muladd | bfloat16 | — | — | — | — | 0.169 | f32 | 6.25e-2 | from the item 4 run |
| cuda | map:where | bfloat16 | — | — | — | — | 0.169 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | bfloat16 | — | — | — | — | 0.125 | f32 | 6.25e-2 | from the item 4 run |
| cuda | map:abs_bwd | bfloat16 | — | — | — | — | 0.127 | f32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | bfloat16 | — | — | — | — | 0.126 | f32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | bfloat16 | — | — | — | — | 0.241 | f32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | bfloat16 | — | — | — | — | 0.237 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sqrt | bfloat16 | — | — | — | — | 0.085 | f32 | 3.9e-3 | from the item 4 run |
| cuda | map:inv_sqrt | bfloat16 | — | — | — | — | 0.091 | f32 | 3.9e-3 | from the item 4 run |
| cuda | map:exp | bfloat16 | — | — | — | — | 0.088 | f32 | 1.21e-1 | from the item 4 run |
| cuda | map:expm1 | bfloat16 | — | — | — | — | 0.089 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:log | bfloat16 | — | — | — | — | 0.100 | f32 | 3.84e-3 | from the item 4 run |
| cuda | map:sin | bfloat16 | — | — | — | — | 0.087 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:cos | bfloat16 | — | — | — | — | 0.089 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:tanh | bfloat16 | — | — | — | — | 0.095 | f32 | 1.94e-3 | from the item 4 run |
| cuda | map:sigmoid | bfloat16 | — | — | — | — | 0.095 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:erf | bfloat16 | — | — | — | — | 0.615 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:gelu | bfloat16 | — | — | — | — | 0.099 | f32 | 7.75e-3 | from the item 4 run |
| cuda | map:gelu_erf | bfloat16 | — | — | — | — | 0.991 | f32 | 7.77e-3 | from the item 4 run |
| cuda | map:silu | bfloat16 | — | — | — | — | 0.097 | f32 | 7.79e-3 | from the item 4 run |
| cuda | map:leaky_relu | bfloat16 | — | — | — | — | 0.085 | f32 | 7.81e-4 | from the item 4 run |
| cuda | map:pow | bfloat16 | — | — | — | — | 0.110 | f32 | 6.17e-2 | from the item 4 run |
| cuda | map:dropout | bfloat16 | — | — | — | — | 0.127 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:gated_silu | bfloat16 | — | — | — | — | 0.128 | f32 | 3.12e-2 | from the item 4 run |
| cuda | map:gated_gelu | bfloat16 | — | — | — | — | 0.127 | f32 | 3.12e-2 | from the item 4 run |
| cuda | map:sqrt_bwd | bfloat16 | — | — | — | — | 0.235 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | bfloat16 | — | — | — | — | 0.239 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:exp_bwd | bfloat16 | — | — | — | — | 0.241 | f32 | 5e-1 | small cells 1: CUDA bfloat16 formulas |
| cuda | map:log_bwd | bfloat16 | — | — | — | — | 0.240 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:sin_bwd | bfloat16 | — | — | — | — | 0.236 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:cos_bwd | bfloat16 | — | — | — | — | 0.240 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:tanh_bwd | bfloat16 | — | — | — | — | 0.237 | f32 | 7.81e-3 | from the item 4 run |
| cuda | map:sigmoid_bwd | bfloat16 | — | — | — | — | 0.241 | f32 | 1.95e-3 | from the item 4 run |
| cuda | map:silu_bwd | bfloat16 | — | — | — | — | 0.239 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:gelu_bwd | bfloat16 | — | — | — | — | 0.239 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:gelu_erf_bwd | bfloat16 | — | — | — | — | 1.009 | f32 | 1.56e-2 | from the item 4 run |
| cuda | map:leaky_relu_bwd | bfloat16 | — | — | — | — | 0.127 | f32 | 7.81e-4 | from the item 4 run |
| cuda | map:pow_bwd | bfloat16 | — | — | — | — | 0.127 | f32 | 2.5e-1 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | bfloat16 | — | — | — | — | 0.171 | f32 | 6.23e-2 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | bfloat16 | — | — | — | — | 0.237 | f32 | 3.12e-2 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | bfloat16 | — | — | — | — | 0.323 | f32 | 3.12e-2 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | bfloat16 | — | — | — | — | 0.237 | f32 | 3.12e-2 | from the item 4 run |
| cuda | sum:whole | bfloat16 | — | — | — | — | 0.025 | f32 | 2.48e-4 | from the item 4 run |
| cuda | sum:rows | bfloat16 | — | — | — | — | 0.023 | f32 | 1e-5 | from the item 4 run |
| cuda | sum:cols | bfloat16 | — | — | — | — | 0.029 | f32 | 1.86e-5 | from the item 4 run |
| cuda | sumsq:rows | bfloat16 | — | — | — | — | 0.023 | f32 | 1.79e-4 | from the item 4 run |
| cuda | max:whole | bfloat16 | — | — | — | — | 0.025 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | bfloat16 | — | — | — | — | 0.023 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | bfloat16 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | bfloat16 | — | — | — | — | 0.023 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | bfloat16 | — | — | — | — | 0.027 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | bfloat16 | — | — | — | — | 0.189 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | bfloat16 | — | — | — | — | 0.143 | f32 | 3.91e-3 | from the item 4 run |
| cuda | maprow:mul | bfloat16 | — | — | — | — | 0.156 | f32 | 1.95e-3 | from the item 4 run |
| cuda | transpose | bfloat16 | — | — | — | — | 0.090 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv | bfloat16 | — | — | — | — | 0.179 | f32 | 2.5e-1 | from the item 4 run |
| cuda | gemv_t | bfloat16 | — | — | — | — | 0.185 | f32 | 2.49e-1 | from the item 4 run |
| cuda | row:log_softmax | bfloat16 | — | — | — | — | 0.080 | f32 | 3.13e-2 | from the item 4 run |
| cuda | row:softmax_bwd | bfloat16 | — | — | — | — | 0.124 | f32 | 2.5e-1 | from the item 4 run |
| cuda | row:log_softmax_bwd | bfloat16 | — | — | — | — | 0.124 | f32 | 2.5e-1 | from the item 4 run |
| cuda | row:rms_norm_affine | bfloat16 | — | — | — | — | 0.079 | f32 | 7.81e-3 | from the item 4 run |
| cuda | row:rms_norm_bwd | bfloat16 | — | — | — | — | 0.125 | f32 | 1.95e-3 | from the item 4 run |
| cuda | row:layer_norm | bfloat16 | — | — | — | — | 0.081 | f32 | 1.56e-2 | from the item 4 run |
| cuda | row:layer_norm_bwd | bfloat16 | — | — | — | — | 0.127 | f32 | 1.95e-3 | from the item 4 run |
| cuda | row:weight_norm | bfloat16 | — | — | — | — | 0.081 | f32 | 1.22e-4 | from the item 4 run |
| cuda | row:rope | bfloat16 | — | — | — | — | 0.089 | f32 | 1.61e-2 | from the item 4 run |
| cuda | row:scale_mask_softmax | bfloat16 | — | — | — | — | 0.126 | f32 | 1.34e-3 | from the item 4 run |
| cuda | row:add_rms_norm | bfloat16 | — | — | — | — | 0.182 | f32 | 1.56e-2 | from the item 4 run |
| cuda | row:add_layer_norm | bfloat16 | — | — | — | — | 0.182 | f32 | 1.56e-2 | from the item 4 run |
| cuda | row:cross_entropy_bwd | bfloat16 | — | — | — | — | 0.085 | f32 | 1.95e-3 | from the item 4 run |
| cuda | norm_params_bwd:rms | bfloat16 | — | — | — | — | 0.103 | f32 | 3.44e-5 | from the item 4 run |
| cuda | norm_params_bwd:layer | bfloat16 | — | — | — | — | 0.110 | f32 | 4.58e-5 | from the item 4 run |
| cuda | layout:strided | bfloat16 | — | — | — | — | 0.439 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | bfloat16 | — | — | — | — | 0.052 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | bfloat16 | — | — | — | — | 0.089 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | bfloat16 | — | — | — | — | 0.129 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | bfloat16 | — | — | — | — | 0.209 | f32 | 1.56e-2 | from the item 4 run |
| cuda | layout:im2col | bfloat16 | — | — | — | — | 0.068 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | bfloat16 | — | — | — | — | 0.048 | f32 | 1.56e-2 | from the item 4 run |
| cuda | layout:interpolate_nearest | bfloat16 | — | — | — | — | 0.098 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_bilinear | bfloat16 | — | — | — | — | 0.140 | f32 | 1.95e-3 | from the item 4 run |
| cuda | layout:shift | bfloat16 | — | — | — | — | 0.158 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | bfloat16 | — | — | — | — | 0.156 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | bfloat16 | — | — | — | — | 0.021 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool | bfloat16 | — | — | — | — | 0.023 | f32 | 1.95e-3 | from the item 4 run |
| cuda | layout:maxpool_bwd | bfloat16 | — | — | — | — | 0.094 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool_bwd | bfloat16 | — | — | — | — | 0.086 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | bfloat16 | — | — | — | — | 0.144 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_avgpool | bfloat16 | — | — | — | — | 0.144 | f32 | 1.22e-4 | from the item 4 run |
| cuda | layout:depthwise_conv | bfloat16 | — | — | — | — | 0.101 | f32 | 1.55e-2 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | bfloat16 | — | — | — | — | 11.218 | f32 | 3.91e-3 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | bfloat16 | — | — | — | — | 0.169 | f32 | 7.52e-6 | from the item 4 run |
| cuda | matmul_nt | bfloat16 | 2048x2048x2048 Bᵀ | — | — | — | 0.218 | bf16mma | 2.49e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | bfloat16 | 2048x2048x2048 Aᵀ | — | — | — | 0.230 | bf16mma | 2.49e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | bfloat16 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.232 | bf16mma | 2.44e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | bfloat16 | 16 x 512x512x512 | — | — | — | 0.520 | bf16mma | 1.21e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | bfloat16 | 2048x2048x2048 (dA, dB) | — | — | — | 0.441 | bf16mma | 2.49e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | bfloat16 | — | — | — | — | 3.801 | f32 | 1.42e-4 | from the item 4 run |
| cuda | attention_causal | bfloat16 | — | — | — | — | 2.150 | f32 | 1.95e-3 | from the item 4 run |
| cuda | attention_fused | bfloat16 | — | — | — | — | 3.794 | f32 | 1.42e-4 | from the item 4 run |
| cuda | attention_fused_causal | bfloat16 | — | — | — | — | 2.153 | f32 | 1.95e-3 | from the item 4 run |
| cuda | attention_decode | bfloat16 | — | — | — | — | 0.607 | f32 | 1.15e-4 | from the item 4 run |
| cuda | attention_bwd | bfloat16 | — | — | — | — | 6.727 | f32 | 9.66e-4 | from the item 4 run |
| cuda | optimizer:sgd_momentum | bfloat16 | — | — | — | — | 0.571 | f32; + restore of p, m, v | 1.46e-3 | from the item 4 run |
| cuda | optimizer:adam | bfloat16 | — | — | — | — | 0.926 | f32; + restore of p, m, v | 3.68e-3 | from the item 4 run |
| cuda | optimizer:adamw | bfloat16 | — | — | — | — | 0.925 | f32; + restore of p, m, v | 3.69e-3 | from the item 4 run |
| cuda | conv | bfloat16 | — | — | — | — | 0.138 | bf16mma | 1.24e-1 | from the item 4 run |
| cuda | conv_bwd | bfloat16 | — | — | — | — | 2.222 | f32 | 1.25e-1 | from the item 4 run |
| cuda | lstm_step | bfloat16 | — | — | — | — | 1.187 | f32 | 1.92e-3 | from the item 4 run |
| cuda | gru_step | bfloat16 | — | — | — | — | 0.989 | f32 | 1.95e-3 | from the item 4 run |
| cuda | lstm_step_bwd | bfloat16 | — | — | — | — | 2.889 | f32 | 1.94e-3 | from the item 4 run |
| cuda | gru_step_bwd | bfloat16 | — | — | — | — | 2.263 | f32 | 1.81e-3 | from the item 4 run |
| cuda | lstm_seq | bfloat16 | — | — | — | — | 2.847 | f32 | 1.46e-3 | from the item 4 run |
| cuda | lstm_seq_bwd | bfloat16 | — | — | — | — | 14.053 | f32 | 1.75e-3 | from the item 4 run |
| cuda | gru_seq | bfloat16 | — | — | — | — | 2.608 | f32 | 1.95e-3 | from the item 4 run |
| cuda | gru_seq_bwd | bfloat16 | — | — | — | — | 16.897 | f32 | 1.95e-3 | from the item 4 run |
| cuda | topk_rows | bfloat16 | — | — | — | — | 3.588 | f32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | bfloat16 | — | — | — | — | 0.008 | f32 | 0e+0 | from the item 4 run |
| cuda | sample_rows | bfloat16 | — | — | — | — | 1.430 | f32 | 1e+0 | from the item 4 run |
| cuda | cross_entropy | bfloat16 | — | — | — | — | 46.829 | f32 (log_softmax, pick, neg) | 6.19e-2 | from the item 4 run |
| cuda | ctc | bfloat16 | — | — | — | — | 0.530 | f32 | 3.95e+0 | from the item 4 run |
| cuda | ctc_bwd | bfloat16 | — | — | — | — | 1.288 | f32 | 2.93e-3 | from the item 4 run |
| cuda | softmax | bfloat16 | 4096x4096 | — | — | — | 0.082 | f32 | 7.63e-6 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | bfloat16 | 4096x4096 | — | — | — | 0.074 | f32 | 3.91e-3 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | float8_e4m3 | 16777216 | — | — | — | 0.051 | f32 | 6.25e-2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | — | — | — | 0.271 | f16mma | 3.99e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | float8_e4m3 | — | — | — | — | 0.037 | f32 | 0e+0 | from the item 4 run |
| cuda | map:fill | float8_e4m3 | — | — | — | — | 0.021 | f32 | 0e+0 | from the item 4 run |
| cuda | map:neg | float8_e4m3 | — | — | — | — | 0.039 | f32 | 0e+0 | from the item 4 run |
| cuda | map:abs | float8_e4m3 | — | — | — | — | 0.042 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sign | float8_e4m3 | — | — | — | — | 0.043 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square | float8_e4m3 | — | — | — | — | 0.042 | f32 | 4.38e-1 | from the item 4 run |
| cuda | map:relu | float8_e4m3 | — | — | — | — | 0.042 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | float8_e4m3 | — | — | — | — | 0.042 | f32 | 2.5e-1 | from the item 4 run |
| cuda | map:scale | float8_e4m3 | — | — | — | — | 0.042 | f32 | 5e-1 | from the item 4 run |
| cuda | map:affine | float8_e4m3 | — | — | — | — | 0.044 | f32 | 5e-1 | from the item 4 run |
| cuda | map:clamp | float8_e4m3 | — | — | — | — | 0.044 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add | float8_e4m3 | — | — | — | — | 0.481 | f32 | 2.5e-1 | from the item 4 run |
| cuda | map:sub | float8_e4m3 | — | — | — | — | 0.817 | f32 | 2.5e-1 | from the item 4 run |
| cuda | map:mul | float8_e4m3 | — | — | — | — | 0.443 | f32 | 5e-1 | from the item 4 run |
| cuda | map:div | float8_e4m3 | — | — | — | — | 0.058 | f32 | 1.17e-1 | from the item 4 run |
| cuda | map:max | float8_e4m3 | — | — | — | — | 0.055 | f32 | 0e+0 | from the item 4 run |
| cuda | map:min | float8_e4m3 | — | — | — | — | 0.978 | f32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | float8_e4m3 | — | — | — | — | 0.055 | f32 | 5e-1 | from the item 4 run |
| cuda | map:muladd | float8_e4m3 | — | — | — | — | 0.078 | f32 | 1e+0 | from the item 4 run |
| cuda | map:where | float8_e4m3 | — | — | — | — | 0.075 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | float8_e4m3 | — | — | — | — | 0.052 | f32 | 1e+0 | from the item 4 run |
| cuda | map:abs_bwd | float8_e4m3 | — | — | — | — | 0.057 | f32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | float8_e4m3 | — | — | — | — | 0.054 | f32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | float8_e4m3 | — | — | — | — | 0.056 | f32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | float8_e4m3 | — | — | — | — | 0.055 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sqrt | float8_e4m3 | — | — | — | — | 0.043 | f32 | 6.15e-2 | from the item 4 run |
| cuda | map:inv_sqrt | float8_e4m3 | — | — | — | — | 0.056 | f32 | 5.6e-2 | from the item 4 run |
| cuda | map:exp | float8_e4m3 | — | — | — | — | 0.060 | f32 | 1.48e+0 | from the item 4 run |
| cuda | map:expm1 | float8_e4m3 | — | — | — | — | 0.062 | f32 | 2.7e-2 | from the item 4 run |
| cuda | map:log | float8_e4m3 | — | — | — | — | 0.094 | f32 | 5.37e-2 | from the item 4 run |
| cuda | map:sin | float8_e4m3 | — | — | — | — | 0.060 | f32 | 3.05e-2 | from the item 4 run |
| cuda | map:cos | float8_e4m3 | — | — | — | — | 0.061 | f32 | 3.11e-2 | from the item 4 run |
| cuda | map:tanh | float8_e4m3 | — | — | — | — | 0.071 | f32 | 3.01e-2 | from the item 4 run |
| cuda | map:sigmoid | float8_e4m3 | — | — | — | — | 0.069 | f32 | 3.12e-2 | from the item 4 run |
| cuda | map:erf | float8_e4m3 | — | — | — | — | 0.652 | f32 | 3.02e-2 | from the item 4 run |
| cuda | map:gelu | float8_e4m3 | — | — | — | — | 0.073 | f32 | 5.7e-2 | from the item 4 run |
| cuda | map:gelu_erf | float8_e4m3 | — | — | — | — | 1.126 | f32 | 5.7e-2 | from the item 4 run |
| cuda | map:silu | float8_e4m3 | — | — | — | — | 0.067 | f32 | 1.21e-1 | from the item 4 run |
| cuda | map:leaky_relu | float8_e4m3 | — | — | — | — | 0.044 | f32 | 1.25e-2 | from the item 4 run |
| cuda | map:pow | float8_e4m3 | — | — | — | — | 0.120 | f32 | 9.58e-1 | from the item 4 run |
| cuda | map:dropout | float8_e4m3 | — | — | — | — | 0.925 | f32 | 1.88e-1 | from the item 4 run |
| cuda | map:gated_silu | float8_e4m3 | — | — | — | — | 0.078 | f32 | 4.94e-1 | from the item 4 run |
| cuda | map:gated_gelu | float8_e4m3 | — | — | — | — | 0.080 | f32 | 4.98e-1 | from the item 4 run |
| cuda | map:sqrt_bwd | float8_e4m3 | — | — | — | — | 0.064 | f32 | 1.22e-1 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | float8_e4m3 | — | — | — | — | 0.075 | f32 | 2.43e-1 | from the item 4 run |
| cuda | map:exp_bwd | float8_e4m3 | — | — | — | — | 0.071 | f32 | 7.5e+0 | from the item 4 run |
| cuda | map:log_bwd | float8_e4m3 | — | — | — | — | 0.061 | f32 | 2.33e-1 | from the item 4 run |
| cuda | map:sin_bwd | float8_e4m3 | — | — | — | — | 0.070 | f32 | 1.25e-1 | from the item 4 run |
| cuda | map:cos_bwd | float8_e4m3 | — | — | — | — | 0.071 | f32 | 1.24e-1 | from the item 4 run |
| cuda | map:tanh_bwd | float8_e4m3 | — | — | — | — | 0.111 | f32 | 1.22e-1 | from the item 4 run |
| cuda | map:sigmoid_bwd | float8_e4m3 | — | — | — | — | 0.108 | f32 | 3.06e-2 | from the item 4 run |
| cuda | map:silu_bwd | float8_e4m3 | — | — | — | — | 0.109 | f32 | 2.45e-1 | from the item 4 run |
| cuda | map:gelu_bwd | float8_e4m3 | — | — | — | — | 0.115 | f32 | 2.4e-1 | from the item 4 run |
| cuda | map:gelu_erf_bwd | float8_e4m3 | — | — | — | — | 1.215 | f32 | 2.37e-1 | from the item 4 run |
| cuda | map:leaky_relu_bwd | float8_e4m3 | — | — | — | — | 0.052 | f32 | 1.25e-2 | from the item 4 run |
| cuda | map:pow_bwd | float8_e4m3 | — | — | — | — | 0.128 | f32 | 3.92e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | float8_e4m3 | — | — | — | — | 0.122 | f32 | 9.79e-1 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | float8_e4m3 | — | — | — | — | 0.081 | f32 | 4.94e-1 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | float8_e4m3 | — | — | — | — | 0.142 | f32 | 4.97e-1 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | float8_e4m3 | — | — | — | — | 0.084 | f32 | 4.98e-1 | from the item 4 run |
| cuda | sum:whole | float8_e4m3 | — | — | — | — | 0.042 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | float8_e4m3 | — | — | — | — | 0.037 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | float8_e4m3 | — | — | — | — | 0.043 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | float8_e4m3 | — | — | — | — | 0.038 | f32 | 1.64e-4 | from the item 4 run |
| cuda | max:whole | float8_e4m3 | — | — | — | — | 0.043 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | float8_e4m3 | — | — | — | — | 0.039 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | float8_e4m3 | — | — | — | — | 0.048 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | float8_e4m3 | — | — | — | — | 0.041 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | float8_e4m3 | — | — | — | — | 0.043 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | float8_e4m3 | — | — | — | — | 0.245 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | float8_e4m3 | — | — | — | — | 0.071 | f32 | 6.25e-2 | from the item 4 run |
| cuda | maprow:mul | float8_e4m3 | — | — | — | — | 0.072 | f32 | 3.13e-2 | from the item 4 run |
| cuda | transpose | float8_e4m3 | — | — | — | — | 0.075 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv | float8_e4m3 | — | — | — | — | 0.337 | f32 | 3.99e+0 | from the item 4 run |
| cuda | gemv_t | float8_e4m3 | — | — | — | — | 0.349 | f32 | 7.52e+0 | from the item 4 run |
| cuda | row:log_softmax | float8_e4m3 | — | — | — | — | 0.067 | f32 | 5e-1 | from the item 4 run |
| cuda | row:softmax_bwd | float8_e4m3 | — | — | — | — | 0.890 | f32 | 3.99e+0 | from the item 4 run |
| cuda | row:log_softmax_bwd | float8_e4m3 | — | — | — | — | 0.885 | f32 | 4e+0 | from the item 4 run |
| cuda | row:rms_norm_affine | float8_e4m3 | — | — | — | — | 0.068 | f32 | 1.25e-1 | from the item 4 run |
| cuda | row:rms_norm_bwd | float8_e4m3 | — | — | — | — | 0.089 | f32 | 3.12e-2 | from the item 4 run |
| cuda | row:layer_norm | float8_e4m3 | — | — | — | — | 0.095 | f32 | 2.5e-1 | from the item 4 run |
| cuda | row:layer_norm_bwd | float8_e4m3 | — | — | — | — | 0.841 | f32 | 3.13e-2 | from the item 4 run |
| cuda | row:weight_norm | float8_e4m3 | — | — | — | — | 0.043 | f32 | 1.95e-3 | from the item 4 run |
| cuda | row:rope | float8_e4m3 | — | — | — | — | 0.111 | f32 | 2.5e-1 | from the item 4 run |
| cuda | row:scale_mask_softmax | float8_e4m3 | — | — | — | — | 0.124 | f32 | 2.93e-2 | from the item 4 run |
| cuda | row:add_rms_norm | float8_e4m3 | — | — | — | — | 0.097 | f32 | 2.5e-1 | from the item 4 run |
| cuda | row:add_layer_norm | float8_e4m3 | — | — | — | — | 0.138 | f32 | 2.5e-1 | from the item 4 run |
| cuda | row:cross_entropy_bwd | float8_e4m3 | — | — | — | — | 0.081 | f32 | 2.07e-3 | from the item 4 run |
| cuda | norm_params_bwd:rms | float8_e4m3 | — | — | — | — | 0.101 | f32 | 3.71e-5 | from the item 4 run |
| cuda | norm_params_bwd:layer | float8_e4m3 | — | — | — | — | 0.103 | f32 | 3.75e-5 | from the item 4 run |
| cuda | layout:strided | float8_e4m3 | — | — | — | — | 0.312 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | float8_e4m3 | — | — | — | — | 0.044 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | float8_e4m3 | — | — | — | — | 0.063 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | float8_e4m3 | — | — | — | — | 0.064 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | float8_e4m3 | — | — | — | — | 0.183 | f32 | 2.5e-1 | from the item 4 run |
| cuda | layout:im2col | float8_e4m3 | — | — | — | — | 0.066 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | float8_e4m3 | — | — | — | — | 0.056 | f32 | 2.5e-1 | from the item 4 run |
| cuda | layout:interpolate_nearest | float8_e4m3 | — | — | — | — | 0.099 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_bilinear | float8_e4m3 | — | — | — | — | 0.198 | f32 | 3.13e-2 | from the item 4 run |
| cuda | layout:shift | float8_e4m3 | — | — | — | — | 0.062 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | float8_e4m3 | — | — | — | — | 0.061 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | float8_e4m3 | — | — | — | — | 0.026 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool | float8_e4m3 | — | — | — | — | 0.028 | f32 | 3.13e-2 | from the item 4 run |
| cuda | layout:maxpool_bwd | float8_e4m3 | — | — | — | — | 0.114 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool_bwd | float8_e4m3 | — | — | — | — | 0.090 | f32 | 9.77e-4 | from the item 4 run |
| cuda | layout:adaptive_maxpool | float8_e4m3 | — | — | — | — | 0.322 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_avgpool | float8_e4m3 | — | — | — | — | 0.319 | f32 | 1.95e-3 | from the item 4 run |
| cuda | layout:depthwise_conv | float8_e4m3 | — | — | — | — | 0.159 | f32 | 2.5e-1 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | float8_e4m3 | — | — | — | — | 41.685 | f32 | 6.25e-2 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | float8_e4m3 | — | — | — | — | 0.194 | f32 | 9.76e-4 | from the item 4 run |
| cuda | matmul_nt | float8_e4m3 | 2048x2048x2048 Bᵀ | — | — | — | 0.266 | f16mma | 3.95e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | float8_e4m3 | 2048x2048x2048 Aᵀ | — | — | — | 0.279 | f16mma | 3.97e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | float8_e4m3 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.270 | f16mma | 3.95e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | float8_e4m3 | 16 x 512x512x512 | — | — | — | 0.697 | f16mma | 1.97e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | float8_e4m3 | 2048x2048x2048 (dA, dB) | — | — | — | 0.524 | f16mma | 3.99e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | float8_e4m3 | — | — | — | — | 4.724 | f32 | 1.95e-3 | from the item 4 run |
| cuda | attention_causal | float8_e4m3 | — | — | — | — | 2.657 | f32 | 3.12e-2 | from the item 4 run |
| cuda | attention_fused | float8_e4m3 | — | — | — | — | 4.723 | f32 | 1.95e-3 | from the item 4 run |
| cuda | attention_fused_causal | float8_e4m3 | — | — | — | — | 2.654 | f32 | 3.12e-2 | from the item 4 run |
| cuda | attention_decode | float8_e4m3 | — | — | — | — | 0.431 | f32 | 1.93e-3 | from the item 4 run |
| cuda | attention_bwd | float8_e4m3 | — | — | — | — | 4.653 | f32 | 1.53e-2 | from the item 4 run |
| cuda | optimizer:sgd_momentum | float8_e4m3 | — | — | — | — | 0.459 | f32; + restore of p, m, v | 1.46e-3 | from the item 4 run |
| cuda | optimizer:adam | float8_e4m3 | — | — | — | — | 0.819 | f32; + restore of p, m, v | 1.81e-2 | from the item 4 run |
| cuda | optimizer:adamw | float8_e4m3 | — | — | — | — | 0.820 | f32; + restore of p, m, v | 1.81e-2 | from the item 4 run |
| cuda | conv | float8_e4m3 | — | — | — | — | 0.144 | f16mma | 2e+0 | from the item 4 run |
| cuda | conv_bwd | float8_e4m3 | — | — | — | — | 2.219 | f32 | 2e+0 | from the item 4 run |
| cuda | lstm_step | float8_e4m3 | — | — | — | — | 1.344 | f32 | 3.08e-2 | from the item 4 run |
| cuda | gru_step | float8_e4m3 | — | — | — | — | 0.957 | f32 | 3.12e-2 | from the item 4 run |
| cuda | lstm_step_bwd | float8_e4m3 | — | — | — | — | 2.863 | f32 | 3.09e-2 | from the item 4 run |
| cuda | gru_step_bwd | float8_e4m3 | — | — | — | — | 2.232 | f32 | 3.05e-2 | from the item 4 run |
| cuda | lstm_seq | float8_e4m3 | — | — | — | — | 2.852 | f32 | 1.77e-2 | from the item 4 run |
| cuda | lstm_seq_bwd | float8_e4m3 | — | — | — | — | 14.270 | f32 | 2.45e-2 | from the item 4 run |
| cuda | gru_seq | float8_e4m3 | — | — | — | — | 2.622 | f32 | 3.12e-2 | from the item 4 run |
| cuda | gru_seq_bwd | float8_e4m3 | — | — | — | — | 16.900 | f32 | 3.11e-2 | from the item 4 run |
| cuda | topk_rows | float8_e4m3 | — | — | — | — | 4.076 | f32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | float8_e4m3 | — | — | — | — | 0.009 | f32 | 0e+0 | from the item 4 run |
| cuda | sample_rows | float8_e4m3 | — | — | — | — | 1.447 | f32 | 1e+0 | from the item 4 run |
| cuda | cross_entropy | float8_e4m3 | — | — | — | — | 18.723 | f32 (log_softmax, pick, neg) | 5e-1 | from the item 4 run |
| cuda | ctc | float8_e4m3 | — | — | — | — | 0.528 | f32 | Infinity | from the item 4 run |
| cuda | ctc_bwd | float8_e4m3 | — | — | — | — | 1.243 | f32 | 3.22e-2 | from the item 4 run |
| cuda | softmax | float8_e4m3 | 4096x4096 | — | — | — | 0.067 | f32 | 9.76e-4 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | — | — | — | 0.041 | f32 | 6.25e-2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | float8_e5m2 | 16777216 | — | — | — | 0.056 | f32 | 1.25e-1 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | — | — | — | 0.258 | f16mma | 7.71e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | float8_e5m2 | — | — | — | — | 0.039 | f32 | 0e+0 | from the item 4 run |
| cuda | map:fill | float8_e5m2 | — | — | — | — | 0.019 | f32 | 0e+0 | from the item 4 run |
| cuda | map:neg | float8_e5m2 | — | — | — | — | 0.041 | f32 | 0e+0 | from the item 4 run |
| cuda | map:abs | float8_e5m2 | — | — | — | — | 0.041 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sign | float8_e5m2 | — | — | — | — | 0.045 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square | float8_e5m2 | — | — | — | — | 0.047 | f32 | 1e+0 | from the item 4 run |
| cuda | map:relu | float8_e5m2 | — | — | — | — | 0.046 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | float8_e5m2 | — | — | — | — | 0.046 | f32 | 5e-1 | from the item 4 run |
| cuda | map:scale | float8_e5m2 | — | — | — | — | 0.046 | f32 | 1e+0 | from the item 4 run |
| cuda | map:affine | float8_e5m2 | — | — | — | — | 0.046 | f32 | 1e+0 | from the item 4 run |
| cuda | map:clamp | float8_e5m2 | — | — | — | — | 0.047 | f32 | 0e+0 | from the item 4 run |
| cuda | map:add | float8_e5m2 | — | — | — | — | 0.606 | f32 | 5e-1 | from the item 4 run |
| cuda | map:sub | float8_e5m2 | — | — | — | — | 0.657 | f32 | 5e-1 | from the item 4 run |
| cuda | map:mul | float8_e5m2 | — | — | — | — | 0.063 | f32 | 1e+0 | from the item 4 run |
| cuda | map:div | float8_e5m2 | — | — | — | — | 0.588 | f32 | 2.14e-1 | from the item 4 run |
| cuda | map:max | float8_e5m2 | — | — | — | — | 0.066 | f32 | 0e+0 | from the item 4 run |
| cuda | map:min | float8_e5m2 | — | — | — | — | 0.572 | f32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | float8_e5m2 | — | — | — | — | 0.066 | f32 | 1e+0 | from the item 4 run |
| cuda | map:muladd | float8_e5m2 | — | — | — | — | 0.075 | f32 | 2e+0 | from the item 4 run |
| cuda | map:where | float8_e5m2 | — | — | — | — | 0.088 | f32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | float8_e5m2 | — | — | — | — | 0.057 | f32 | 2e+0 | from the item 4 run |
| cuda | map:abs_bwd | float8_e5m2 | — | — | — | — | 0.061 | f32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | float8_e5m2 | — | — | — | — | 0.057 | f32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | float8_e5m2 | — | — | — | — | 0.058 | f32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | float8_e5m2 | — | — | — | — | 0.064 | f32 | 0e+0 | from the item 4 run |
| cuda | map:sqrt | float8_e5m2 | — | — | — | — | 0.048 | f32 | 1.21e-1 | from the item 4 run |
| cuda | map:inv_sqrt | float8_e5m2 | — | — | — | — | 0.059 | f32 | 9.53e-2 | from the item 4 run |
| cuda | map:exp | float8_e5m2 | — | — | — | — | 0.065 | f32 | 1.4e+0 | from the item 4 run |
| cuda | map:expm1 | float8_e5m2 | — | — | — | — | 0.065 | f32 | 4.88e-2 | from the item 4 run |
| cuda | map:log | float8_e5m2 | — | — | — | — | 0.097 | f32 | 1.14e-1 | from the item 4 run |
| cuda | map:sin | float8_e5m2 | — | — | — | — | 0.063 | f32 | 5.66e-2 | from the item 4 run |
| cuda | map:cos | float8_e5m2 | — | — | — | — | 0.065 | f32 | 6.15e-2 | from the item 4 run |
| cuda | map:tanh | float8_e5m2 | — | — | — | — | 0.073 | f32 | 5.86e-2 | from the item 4 run |
| cuda | map:sigmoid | float8_e5m2 | — | — | — | — | 0.071 | f32 | 6.22e-2 | from the item 4 run |
| cuda | map:erf | float8_e5m2 | — | — | — | — | 0.633 | f32 | 4.79e-2 | from the item 4 run |
| cuda | map:gelu | float8_e5m2 | — | — | — | — | 0.072 | f32 | 1.18e-1 | from the item 4 run |
| cuda | map:gelu_erf | float8_e5m2 | — | — | — | — | 1.113 | f32 | 1.18e-1 | from the item 4 run |
| cuda | map:silu | float8_e5m2 | — | — | — | — | 0.069 | f32 | 1.9e-1 | from the item 4 run |
| cuda | map:leaky_relu | float8_e5m2 | — | — | — | — | 0.048 | f32 | 2.5e-2 | from the item 4 run |
| cuda | map:pow | float8_e5m2 | — | — | — | — | 0.122 | f32 | 1.08e+0 | from the item 4 run |
| cuda | map:dropout | float8_e5m2 | — | — | — | — | 1.202 | f32 | 3.75e-1 | from the item 4 run |
| cuda | map:gated_silu | float8_e5m2 | — | — | — | — | 0.082 | f32 | 7.59e-1 | from the item 4 run |
| cuda | map:gated_gelu | float8_e5m2 | — | — | — | — | 0.084 | f32 | 9.89e-1 | from the item 4 run |
| cuda | map:sqrt_bwd | float8_e5m2 | — | — | — | — | 0.068 | f32 | 2.14e-1 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | float8_e5m2 | — | — | — | — | 0.078 | f32 | 3.43e-1 | from the item 4 run |
| cuda | map:exp_bwd | float8_e5m2 | — | — | — | — | 0.075 | f32 | 8.5e+0 | from the item 4 run |
| cuda | map:log_bwd | float8_e5m2 | — | — | — | — | 0.070 | f32 | 4.29e-1 | from the item 4 run |
| cuda | map:sin_bwd | float8_e5m2 | — | — | — | — | 0.074 | f32 | 2.46e-1 | from the item 4 run |
| cuda | map:cos_bwd | float8_e5m2 | — | — | — | — | 0.074 | f32 | 2.39e-1 | from the item 4 run |
| cuda | map:tanh_bwd | float8_e5m2 | — | — | — | — | 0.115 | f32 | 2.47e-1 | from the item 4 run |
| cuda | map:sigmoid_bwd | float8_e5m2 | — | — | — | — | 0.113 | f32 | 6.19e-2 | from the item 4 run |
| cuda | map:silu_bwd | float8_e5m2 | — | — | — | — | 0.113 | f32 | 3.98e-1 | from the item 4 run |
| cuda | map:gelu_bwd | float8_e5m2 | — | — | — | — | 0.119 | f32 | 4.9e-1 | from the item 4 run |
| cuda | map:gelu_erf_bwd | float8_e5m2 | — | — | — | — | 1.145 | f32 | 4.91e-1 | from the item 4 run |
| cuda | map:leaky_relu_bwd | float8_e5m2 | — | — | — | — | 0.058 | f32 | 2.5e-2 | from the item 4 run |
| cuda | map:pow_bwd | float8_e5m2 | — | — | — | — | 0.132 | f32 | 6e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | float8_e5m2 | — | — | — | — | 0.126 | f32 | 1.12e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | float8_e5m2 | — | — | — | — | 0.084 | f32 | 7.59e-1 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | float8_e5m2 | — | — | — | — | 0.146 | f32 | 9.93e-1 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | float8_e5m2 | — | — | — | — | 0.087 | f32 | 9.89e-1 | from the item 4 run |
| cuda | sum:whole | float8_e5m2 | — | — | — | — | 0.044 | f32 | 3.66e-4 | from the item 4 run |
| cuda | sum:rows | float8_e5m2 | — | — | — | — | 0.039 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | float8_e5m2 | — | — | — | — | 0.045 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | float8_e5m2 | — | — | — | — | 0.039 | f32 | 1.71e-4 | from the item 4 run |
| cuda | max:whole | float8_e5m2 | — | — | — | — | 0.045 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | float8_e5m2 | — | — | — | — | 0.040 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | float8_e5m2 | — | — | — | — | 0.049 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | float8_e5m2 | — | — | — | — | 0.043 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | float8_e5m2 | — | — | — | — | 0.045 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | float8_e5m2 | — | — | — | — | 0.251 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | float8_e5m2 | — | — | — | — | 0.074 | f32 | 1.25e-1 | from the item 4 run |
| cuda | maprow:mul | float8_e5m2 | — | — | — | — | 0.074 | f32 | 6.25e-2 | from the item 4 run |
| cuda | transpose | float8_e5m2 | — | — | — | — | 0.077 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv | float8_e5m2 | — | — | — | — | 0.335 | f32 | 7.95e+0 | from the item 4 run |
| cuda | gemv_t | float8_e5m2 | — | — | — | — | 0.357 | f32 | 1.15e+1 | from the item 4 run |
| cuda | row:log_softmax | float8_e5m2 | 4096x4096 | — | — | — | 0.069 | f32 | 1e+0 | qiyas: 6k CUDA rows |
| cuda | row:softmax_bwd | float8_e5m2 | 4096x4096 | — | — | — | 0.062 | f32 | 7.49e+0 | qiyas: 6k CUDA rows |
| cuda | row:log_softmax_bwd | float8_e5m2 | 4096x4096 | — | — | — | 0.079 | f32 | 8e+0 | qiyas: 6k CUDA rows |
| cuda | row:rms_norm_affine | float8_e5m2 | 4096x4096 | — | — | — | 0.073 | f32 | 2.5e-1 | qiyas: 6k CUDA rows |
| cuda | row:rms_norm_bwd | float8_e5m2 | 4096x4096 | — | — | — | 0.097 | f32 | 6.25e-2 | 6k CUDA rows |
| cuda | row:layer_norm | float8_e5m2 | 4096x4096 | — | — | — | 0.096 | f32 | 5e-1 | qiyas: 6k CUDA rows |
| cuda | row:layer_norm_bwd | float8_e5m2 | 4096x4096 | — | — | — | 0.105 | f32 | 6.25e-2 | 6k CUDA rows |
| cuda | row:weight_norm | float8_e5m2 | 4096x4096 | — | — | — | 0.046 | f32 | 3.91e-3 | qiyas: 6k CUDA rows |
| cuda | row:rope | float8_e5m2 | 4096x4096 | — | — | — | 0.122 | f32 | 5e-1 | 6k CUDA rows |
| cuda | row:scale_mask_softmax | float8_e5m2 | 4096x4096 | — | — | — | 0.133 | f32 | 5.42e-2 | 6k CUDA rows |
| cuda | row:add_rms_norm | float8_e5m2 | 4096x4096 | — | — | — | 0.153 | f32 | 5e-1 | qiyas: 6k CUDA rows |
| cuda | row:add_layer_norm | float8_e5m2 | 4096x4096 | — | — | — | 0.162 | f32 | 5e-1 | 6k CUDA rows |
| cuda | row:cross_entropy_bwd | float8_e5m2 | 4096x4096 | — | — | — | 0.085 | f32 | 2.05e-3 | 6k CUDA rows |
| cuda | norm_params_bwd:rms | float8_e5m2 | 4096x4096 | — | — | — | 0.112 | f32 | 3.62e-5 | 6k CUDA rows |
| cuda | norm_params_bwd:layer | float8_e5m2 | 4096x4096 | — | — | — | 0.114 | f32 | 3.56e-5 | 6k CUDA rows |
| cuda | layout:strided | float8_e5m2 | — | — | — | — | 0.311 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | float8_e5m2 | — | — | — | — | 0.045 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | float8_e5m2 | — | — | — | — | 0.066 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | float8_e5m2 | — | — | — | — | 0.066 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | float8_e5m2 | — | — | — | — | 0.184 | f32 | 5e-1 | from the item 4 run |
| cuda | layout:im2col | float8_e5m2 | — | — | — | — | 0.067 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | float8_e5m2 | — | — | — | — | 0.056 | f32 | 5e-1 | from the item 4 run |
| cuda | layout:interpolate_nearest | float8_e5m2 | — | — | — | — | 0.099 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_bilinear | float8_e5m2 | — | — | — | — | 0.200 | f32 | 6.25e-2 | from the item 4 run |
| cuda | layout:shift | float8_e5m2 | — | — | — | — | 0.067 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | float8_e5m2 | — | — | — | — | 0.066 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | float8_e5m2 | — | — | — | — | 0.027 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool | float8_e5m2 | — | — | — | — | 0.028 | f32 | 6.25e-2 | from the item 4 run |
| cuda | layout:maxpool_bwd | float8_e5m2 | — | — | — | — | 0.116 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:avgpool_bwd | float8_e5m2 | — | — | — | — | 0.091 | f32 | 7.63e-6 | from the item 4 run |
| cuda | layout:adaptive_maxpool | float8_e5m2 | — | — | — | — | 0.330 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_avgpool | float8_e5m2 | — | — | — | — | 0.323 | f32 | 3.9e-3 | from the item 4 run |
| cuda | layout:depthwise_conv | float8_e5m2 | — | — | — | — | 0.167 | f32 | 4.92e-1 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | float8_e5m2 | — | — | — | — | 44.348 | f32 | 1.25e-1 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | float8_e5m2 | — | — | — | — | 0.196 | f32 | 2.34e-4 | from the item 4 run |
| cuda | matmul_nt | float8_e5m2 | 2048x2048x2048 Bᵀ | — | — | — | 0.256 | f16mma | 7.97e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | float8_e5m2 | 2048x2048x2048 Aᵀ | — | — | — | 0.260 | f16mma | 7.66e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | float8_e5m2 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.258 | f16mma | 7.96e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | float8_e5m2 | 16 x 512x512x512 | — | — | — | 0.699 | f16mma | 3.65e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | float8_e5m2 | 2048x2048x2048 (dA, dB) | — | — | — | 0.523 | f16mma | 7.27e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | float8_e5m2 | — | — | — | — | 5.670 | f32 | 3.91e-3 | from the item 4 run |
| cuda | attention_causal | float8_e5m2 | — | — | — | — | 2.698 | f32 | 6.25e-2 | from the item 4 run |
| cuda | attention_fused | float8_e5m2 | — | — | — | — | 4.808 | f32 | 3.91e-3 | from the item 4 run |
| cuda | attention_fused_causal | float8_e5m2 | — | — | — | — | 2.704 | f32 | 6.25e-2 | from the item 4 run |
| cuda | attention_decode | float8_e5m2 | — | — | — | — | 0.434 | f32 | 2.98e-3 | from the item 4 run |
| cuda | attention_bwd | float8_e5m2 | — | — | — | — | 4.078 | f32 | 3.06e-2 | from the item 4 run |
| cuda | optimizer:sgd_momentum | float8_e5m2 | — | — | — | — | 0.459 | f32; + restore of p, m, v | 1.46e-3 | from the item 4 run |
| cuda | optimizer:adam | float8_e5m2 | — | — | — | — | 0.820 | f32; + restore of p, m, v | 2.59e-2 | from the item 4 run |
| cuda | optimizer:adamw | float8_e5m2 | — | — | — | — | 0.818 | f32; + restore of p, m, v | 2.59e-2 | from the item 4 run |
| cuda | conv | float8_e5m2 | — | — | — | — | 0.144 | f16mma | 3.99e+0 | from the item 4 run |
| cuda | conv_bwd | float8_e5m2 | — | — | — | — | 2.218 | f32 | 4e+0 | from the item 4 run |
| cuda | lstm_step | float8_e5m2 | — | — | — | — | 1.167 | f32 | 4.62e-2 | from the item 4 run |
| cuda | gru_step | float8_e5m2 | — | — | — | — | 0.959 | f32 | 6.25e-2 | from the item 4 run |
| cuda | lstm_step_bwd | float8_e5m2 | — | — | — | — | 2.859 | f32 | 6.16e-2 | from the item 4 run |
| cuda | gru_step_bwd | float8_e5m2 | — | — | — | — | 2.226 | f32 | 5.97e-2 | from the item 4 run |
| cuda | lstm_seq | float8_e5m2 | — | — | — | — | 2.844 | f32 | 5.59e-2 | from the item 4 run |
| cuda | lstm_seq_bwd | float8_e5m2 | — | — | — | — | 13.998 | f32 | 5.98e-2 | from the item 4 run |
| cuda | gru_seq | float8_e5m2 | — | — | — | — | 2.605 | f32 | 6.25e-2 | from the item 4 run |
| cuda | gru_seq_bwd | float8_e5m2 | — | — | — | — | 12.669 | f32 | 6.24e-2 | from the item 4 run |
| cuda | topk_rows | float8_e5m2 | — | — | — | — | 4.122 | f32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | float8_e5m2 | — | — | — | — | 0.007 | f32 | 0e+0 | from the item 4 run |
| cuda | sample_rows | float8_e5m2 | — | — | — | — | 1.580 | f32 | 1e+0 | from the item 4 run |
| cuda | cross_entropy | float8_e5m2 | — | — | — | — | 18.233 | f32 (log_softmax, pick, neg) | 9.84e-1 | from the item 4 run |
| cuda | ctc | float8_e5m2 | — | — | — | — | 0.528 | f32 | 1.27e+2 | from the item 4 run |
| cuda | ctc_bwd | float8_e5m2 | — | — | — | — | 1.244 | f32 | 6.33e-2 | from the item 4 run |
| cuda | softmax | float8_e5m2 | 4096x4096 | — | — | — | 0.069 | f32 | 1.62e-4 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | — | — | — | 0.044 | f32 | 1.08e-1 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | int64 | 16777216 | 0.520 | add_s64 | 0e+0 | 0.519 | s64 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | int64 | 2048x2048x2048 | 4.427 | gemm64_s64 | 0e+0 | 0.358 | s8mma | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:fill | int64 | — | — | — | — | 0.165 | s64 | 0e+0 | from the item 4 run |
| cuda | map:neg | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:abs | int64 | — | — | — | — | 0.359 | s64 | 0e+0 | from the item 4 run |
| cuda | map:sign | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:square | int64 | — | — | — | — | 0.685 | s64 | 0e+0 | from the item 4 run |
| cuda | map:relu | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:scale | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:affine | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | map:clamp | int64 | — | — | — | — | 0.359 | s64 | 0e+0 | from the item 4 run |
| cuda | map:add | int64 | — | — | — | — | 0.523 | s64 | 0e+0 | from the item 4 run |
| cuda | map:sub | int64 | — | — | — | — | 0.553 | s64 | 0e+0 | from the item 4 run |
| cuda | map:mul | int64 | — | — | — | — | 0.521 | s64 | 0e+0 | from the item 4 run |
| cuda | map:div | int64 | — | — | — | — | 0.521 | s64 | 0e+0 | from the item 4 run |
| cuda | map:max | int64 | — | — | — | — | 0.520 | s64 | 0e+0 | from the item 4 run |
| cuda | map:min | int64 | — | — | — | — | 0.520 | s64 | 0e+0 | from the item 4 run |
| cuda | map:axpy | int64 | — | — | — | — | 0.521 | s64 | 0e+0 | from the item 4 run |
| cuda | map:muladd | int64 | — | — | — | — | 0.686 | s64 | 0e+0 | from the item 4 run |
| cuda | map:where | int64 | — | — | — | — | 0.617 | s64 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | int64 | — | — | — | — | 0.520 | s64 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | int64 | — | — | — | — | 0.519 | s64 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | int64 | — | — | — | — | 0.521 | s64 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | int64 | — | — | — | — | 0.520 | s64 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | int64 | — | — | — | — | 0.520 | s64 | 0e+0 | from the item 4 run |
| cuda | sum:whole | int64 | — | — | — | — | 0.395 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | int64 | — | — | — | — | 0.390 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | int64 | — | — | — | — | 0.393 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | int64 | — | — | — | — | 0.391 | f32 | 0e+0 | from the item 4 run |
| cuda | max:whole | int64 | — | — | — | — | 0.174 | s64 | 0e+0 | from the item 4 run |
| cuda | max:rows | int64 | — | — | — | — | 0.170 | s64 | 0e+0 | from the item 4 run |
| cuda | max:cols | int64 | — | — | — | — | 0.177 | s64 | 0e+0 | from the item 4 run |
| cuda | min:rows | int64 | — | — | — | — | 0.170 | s64 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | int64 | — | — | — | — | 0.327 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | int64 | — | — | — | — | 0.392 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | int64 | — | — | — | — | 0.357 | s64 | 0e+0 | from the item 4 run |
| cuda | transpose | int64 | — | — | — | — | 0.361 | s64 | 0e+0 | from the item 4 run |
| cuda | gemv | int64 | — | — | — | — | 1.421 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv_t | int64 | — | — | — | — | 1.473 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:strided | int64 | — | — | — | — | 0.691 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | int64 | — | — | — | — | 0.175 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | int64 | — | — | — | — | 0.343 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | int64 | — | — | — | — | 0.382 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | int64 | — | — | — | — | 0.675 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | int64 | — | — | — | — | 0.116 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | int64 | — | — | — | — | 0.254 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | int64 | — | — | — | — | 0.244 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:shift | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | int64 | — | — | — | — | 0.358 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | int64 | — | — | — | — | 0.021 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | int64 | — | — | — | — | 0.193 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | int64 | — | — | — | — | 0.088 | s64 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | int64 | — | — | — | — | 0.235 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | int64 | — | — | — | — | 6.897 | s32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | int64 | 2048x2048x2048 Bᵀ | — | — | — | 0.367 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | int64 | 2048x2048x2048 Aᵀ | — | — | — | 0.365 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | int64 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.355 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | int64 | 16 x 512x512x512 | — | — | — | 1.743 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | int64 | 2048x2048x2048 (dA, dB) | — | — | — | 0.720 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | int64 | — | — | — | — | 0.195 | s8mma | 0e+0 | from the item 4 run |
| cuda | topk_rows | int64 | — | — | — | — | 16.244 | s64 | 0e+0 | from the item 4 run |
| cuda | pick_rows | int64 | — | — | — | — | 0.009 | s64 | 0e+0 | from the item 4 run |
| cuda | add | int32 | 16777216 | 0.262 | add_s32 | 0e+0 | 0.262 | s32 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | int32 | 2048x2048x2048 | 1.106 | gemm128_s32 | 0e+0 | 0.217 | s8mma | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | int32 | — | — | — | — | 0.181 | s32 | 0e+0 | from the item 4 run |
| cuda | map:fill | int32 | — | — | — | — | 0.083 | s32 | 0e+0 | from the item 4 run |
| cuda | map:neg | int32 | — | — | — | — | 0.182 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs | int32 | — | — | — | — | 0.182 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sign | int32 | — | — | — | — | 0.180 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square | int32 | — | — | — | — | 0.181 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu | int32 | — | — | — | — | 0.183 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | int32 | — | — | — | — | 0.182 | s32 | 0e+0 | from the item 4 run |
| cuda | map:scale | int32 | — | — | — | — | 0.181 | s32 | 0e+0 | from the item 4 run |
| cuda | map:affine | int32 | — | — | — | — | 0.182 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | int32 | — | — | — | — | 0.181 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add | int32 | — | — | — | — | 0.261 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sub | int32 | — | — | — | — | 0.263 | s32 | 0e+0 | from the item 4 run |
| cuda | map:mul | int32 | — | — | — | — | 0.263 | s32 | 0e+0 | from the item 4 run |
| cuda | map:div | int32 | — | — | — | — | 0.261 | s32 | 0e+0 | from the item 4 run |
| cuda | map:max | int32 | — | — | — | — | 0.263 | s32 | 0e+0 | from the item 4 run |
| cuda | map:min | int32 | — | — | — | — | 0.262 | s32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | int32 | — | — | — | — | 0.262 | s32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | int32 | — | — | — | — | 0.344 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where | int32 | — | — | — | — | 0.329 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | int32 | — | — | — | — | 0.263 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | int32 | — | — | — | — | 0.262 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | int32 | — | — | — | — | 0.263 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | int32 | — | — | — | — | 0.262 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | int32 | — | — | — | — | 0.263 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | int32 | — | — | — | — | 0.093 | s64 | 0e+0 | from the item 4 run |
| cuda | sum:rows | int32 | — | — | — | — | 0.089 | s64 | 0e+0 | from the item 4 run |
| cuda | sum:cols | int32 | — | — | — | — | 0.099 | s64 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | int32 | — | — | — | — | 0.089 | s64 | 0e+0 | from the item 4 run |
| cuda | max:whole | int32 | — | — | — | — | 0.093 | s32 | 0e+0 | from the item 4 run |
| cuda | max:rows | int32 | — | — | — | — | 0.089 | s32 | 0e+0 | from the item 4 run |
| cuda | max:cols | int32 | — | — | — | — | 0.096 | s32 | 0e+0 | from the item 4 run |
| cuda | min:rows | int32 | — | — | — | — | 0.089 | s32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | int32 | — | — | — | — | 0.089 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | int32 | — | — | — | — | 0.398 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | int32 | — | — | — | — | 0.180 | s32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | int32 | — | — | — | — | 0.183 | s32 | 0e+0 | from the item 4 run |
| cuda | transpose | int32 | — | — | — | — | 0.192 | s32 | 0e+0 | from the item 4 run |
| cuda | gemv | int32 | — | — | — | — | 0.814 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv_t | int32 | — | — | — | — | 0.814 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:strided | int32 | — | — | — | — | 0.428 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | int32 | — | — | — | — | 0.134 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | int32 | — | — | — | — | 0.195 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | int32 | — | — | — | — | 0.253 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | int32 | — | — | — | — | 0.473 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | int32 | — | — | — | — | 0.087 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | int32 | — | — | — | — | 0.119 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | int32 | — | — | — | — | 0.151 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | int32 | — | — | — | — | 0.181 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | int32 | — | — | — | — | 0.181 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | int32 | — | — | — | — | 0.021 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | int32 | — | — | — | — | 0.172 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | int32 | — | — | — | — | 0.081 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | int32 | — | — | — | — | 0.207 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | int32 | — | — | — | — | 7.045 | s32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | int32 | 2048x2048x2048 Bᵀ | — | — | — | 0.276 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | int32 | 2048x2048x2048 Aᵀ | — | — | — | 0.274 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | int32 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.273 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | int32 | 16 x 512x512x512 | — | — | — | 1.497 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | int32 | 2048x2048x2048 (dA, dB) | — | — | — | 0.516 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | int32 | — | — | — | — | 0.174 | s8mma | 0e+0 | from the item 4 run |
| cuda | topk_rows | int32 | — | — | — | — | 4.613 | s32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | int32 | — | — | — | — | 0.009 | s32 | 0e+0 | from the item 4 run |
| cuda | add | int16 | 16777216 | 0.259 | add_s16 | 0e+0 | 0.125 | s16 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | int16 | 2048x2048x2048 | 1.591 | gemm128_s16 | 0e+0 | 0.167 | s8mma | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | int16 | — | — | — | — | 0.086 | s32 | 0e+0 | from the item 4 run |
| cuda | map:fill | int16 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | map:neg | int16 | — | — | — | — | 0.082 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs | int16 | — | — | — | — | 0.090 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sign | int16 | — | — | — | — | 0.141 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square | int16 | — | — | — | — | 0.141 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu | int16 | — | — | — | — | 0.141 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | int16 | — | — | — | — | 0.145 | s32 | 0e+0 | from the item 4 run |
| cuda | map:scale | int16 | — | — | — | — | 0.141 | s32 | 0e+0 | from the item 4 run |
| cuda | map:affine | int16 | — | — | — | — | 0.141 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | int16 | — | — | — | — | 0.141 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add | int16 | — | — | — | — | 0.242 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sub | int16 | — | — | — | — | 0.241 | s32 | 0e+0 | from the item 4 run |
| cuda | map:mul | int16 | — | — | — | — | 0.241 | s32 | 0e+0 | from the item 4 run |
| cuda | map:div | int16 | — | — | — | — | 0.233 | s32 | 0e+0 | from the item 4 run |
| cuda | map:max | int16 | — | — | — | — | 0.235 | s32 | 0e+0 | from the item 4 run |
| cuda | map:min | int16 | — | — | — | — | 0.233 | s32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | int16 | — | — | — | — | 0.233 | s32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | int16 | — | — | — | — | 0.321 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where | int16 | — | — | — | — | 0.321 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | int16 | — | — | — | — | 0.241 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | int16 | — | — | — | — | 0.237 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | int16 | — | — | — | — | 0.239 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | int16 | — | — | — | — | 0.232 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | int16 | — | — | — | — | 0.241 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | int16 | — | — | — | — | 0.029 | s64 | 0e+0 | from the item 4 run |
| cuda | sum:rows | int16 | — | — | — | — | 0.027 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | int16 | — | — | — | — | 0.031 | s32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | int16 | — | — | — | — | 0.031 | s64 | 0e+0 | from the item 4 run |
| cuda | max:whole | int16 | — | — | — | — | 0.031 | s32 | 0e+0 | from the item 4 run |
| cuda | max:rows | int16 | — | — | — | — | 0.028 | s32 | 0e+0 | from the item 4 run |
| cuda | max:cols | int16 | — | — | — | — | 0.031 | s32 | 0e+0 | from the item 4 run |
| cuda | min:rows | int16 | — | — | — | — | 0.027 | s32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | int16 | — | — | — | — | 0.031 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | int16 | — | — | — | — | 0.194 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | int16 | — | — | — | — | 1.550 | s32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | int16 | — | — | — | — | 0.088 | s32 | 0e+0 | from the item 4 run |
| cuda | transpose | int16 | — | — | — | — | 0.091 | s32 | 0e+0 | from the item 4 run |
| cuda | gemv | int16 | — | — | — | — | 0.744 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv_t | int16 | — | — | — | — | 0.459 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:strided | int16 | — | — | — | — | 0.355 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | int16 | — | — | — | — | 0.048 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | int16 | — | — | — | — | 0.092 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | int16 | — | — | — | — | 0.130 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | int16 | — | — | — | — | 0.292 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | int16 | — | — | — | — | 0.070 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | int16 | — | — | — | — | 0.112 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | int16 | — | — | — | — | 0.096 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | int16 | — | — | — | — | 0.156 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | int16 | — | — | — | — | 0.159 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | int16 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | int16 | — | — | — | — | 0.158 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | int16 | — | — | — | — | 0.145 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | int16 | — | — | — | — | 0.227 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | int16 | — | — | — | — | 14.406 | s32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | int16 | 2048x2048x2048 Bᵀ | — | — | — | 0.167 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | int16 | 2048x2048x2048 Aᵀ | — | — | — | 0.180 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | int16 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.177 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | int16 | 16 x 512x512x512 | — | — | — | 1.538 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | int16 | 2048x2048x2048 (dA, dB) | — | — | — | 0.417 | s8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | int16 | — | — | — | — | 0.201 | s8mma | 0e+0 | from the item 4 run |
| cuda | topk_rows | int16 | — | — | — | — | 3.549 | s32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | int16 | — | — | — | — | 0.008 | s32 | 0e+0 | from the item 4 run |
| cuda | add | int8 | 16777216 | 0.080 | add_s8 | 0e+0 | 0.023 | s8 | 0e+0 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | int8 | 2048x2048x2048 | 1.548 | gemm128_s8 | 5.12e+2 | 0.094 | s8mma | 3.23e+2 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:fill | int8 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:neg | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sign | int8 | — | — | — | — | 0.024 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square | int8 | — | — | — | — | 0.024 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:scale | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:affine | int8 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | int8 | — | — | — | — | 0.026 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add | int8 | — | — | — | — | 0.539 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sub | int8 | — | — | — | — | 0.034 | s32 | 0e+0 | from the item 4 run |
| cuda | map:mul | int8 | — | — | — | — | 0.047 | s32 | 0e+0 | from the item 4 run |
| cuda | map:div | int8 | — | — | — | — | 0.058 | s32 | 0e+0 | from the item 4 run |
| cuda | map:max | int8 | — | — | — | — | 0.051 | s32 | 0e+0 | from the item 4 run |
| cuda | map:min | int8 | — | — | — | — | 0.049 | s32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | int8 | — | — | — | — | 0.049 | s32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | int8 | — | — | — | — | 0.074 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where | int8 | — | — | — | — | 0.074 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | int8 | — | — | — | — | 0.047 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | int8 | — | — | — | — | 0.049 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | int8 | — | — | — | — | 0.047 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | int8 | — | — | — | — | 0.049 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | int8 | — | — | — | — | 0.047 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | int8 | — | — | — | — | 0.029 | s64 | 0e+0 | from the item 4 run |
| cuda | sum:rows | int8 | — | — | — | — | 0.027 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | int8 | — | — | — | — | 0.033 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | int8 | — | — | — | — | 0.027 | s32 | 0e+0 | from the item 4 run |
| cuda | max:whole | int8 | — | — | — | — | 0.033 | s32 | 0e+0 | from the item 4 run |
| cuda | max:rows | int8 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | max:cols | int8 | — | — | — | — | 0.034 | s32 | 0e+0 | from the item 4 run |
| cuda | min:rows | int8 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | int8 | — | — | — | — | 0.033 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | int8 | — | — | — | — | 0.197 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | int8 | — | — | — | — | 0.040 | s32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | int8 | — | — | — | — | 0.040 | s32 | 0e+0 | from the item 4 run |
| cuda | transpose | int8 | — | — | — | — | 0.046 | s32 | 0e+0 | from the item 4 run |
| cuda | gemv | int8 | — | — | — | — | 0.130 | s32 | 5.72e+2 | from the item 4 run |
| cuda | gemv_t | int8 | — | — | — | — | 0.148 | s32 | 6.32e+2 | from the item 4 run |
| cuda | layout:strided | int8 | 1x1024x64x256 | — | — | — | 0.311 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:one_hot | int8 | 16384x1024 | — | — | — | 0.038 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:gather_rows | int8 | 4096 of 4096x4096 | — | — | — | 0.046 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:gather_cols | int8 | 4096 of 4096x4096 | — | — | — | 0.047 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:scatter_add_rows | int8 | 4096 into 4096x4096 | — | — | — | 0.172 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:im2col | int8 | 64x128x128 k3x3 | — | — | — | 0.056 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:col2im | int8 | 64x128x128 k3x3 | — | — | — | 0.050 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:interpolate_nearest | int8 | 16x512x512 to 1024x1024 | — | — | — | 0.079 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:shift | int8 | 4096x4096 by 3 | — | — | — | 0.046 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:pad2d | int8 | 4096x4093 to 4096 | — | — | — | 0.046 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:maxpool | int8 | 64x256x256 k2x2 s2 p0 | — | — | — | 0.024 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:maxpool_bwd | int8 | 64x256x256 k2x2 s2 p0 | — | — | — | 0.088 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:adaptive_maxpool | int8 | 64x256x256 to 7x7 | — | — | — | 0.051 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:depthwise_conv | int8 | 64x256x256 k3x3 s1 p1 | — | — | — | 0.118 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:adaptive_maxpool_bwd | int8 | 64x256x256 to 7x7 | — | — | — | 0.218 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | matmul_nt | int8 | 2048x2048x2048 Bᵀ | — | — | — | 0.092 | s8mma | 3.98e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | int8 | 2048x2048x2048 Aᵀ | — | — | — | 0.102 | s8mma | 3.42e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | int8 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.103 | s8mma | 3.77e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | int8 | 16 x 512x512x512 | — | — | — | 0.433 | s8mma | 1.27e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | int8 | 2048x2048x2048 (dA, dB) | — | — | — | 0.187 | s8mma | 3.31e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | int8 | — | — | — | — | 0.103 | s8mma | 1.27e+2 | from the item 4 run |
| cuda | topk_rows | int8 | — | — | — | — | 6.531 | s32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | int8 | — | — | — | — | 0.008 | s32 | 0e+0 | from the item 4 run |
| cuda | add | int4 | 16777216 | — | — | — | 0.018 | s8 | 0e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | int4 | 2048x2048x2048 | — | — | — | 0.103 | s8mma | 4.43e+2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | int4 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:fill | int4 | — | — | — | — | 0.016 | s32 | 0e+0 | from the item 4 run |
| cuda | map:neg | int4 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sign | int4 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:scale | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:affine | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sub | int4 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:mul | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:div | int4 | — | — | — | — | 0.046 | s32 | 0e+0 | from the item 4 run |
| cuda | map:max | int4 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:min | int4 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | int4 | — | — | — | — | 0.026 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where | int4 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | int4 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | int4 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | int4 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | int4 | — | — | — | — | 0.021 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | int4 | — | — | — | — | 0.023 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | int4 | — | — | — | — | 0.033 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | int4 | — | — | — | — | 0.032 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | int4 | — | — | — | — | 0.036 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | int4 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | max:whole | int4 | — | — | — | — | 0.033 | s32 | 0e+0 | from the item 4 run |
| cuda | max:rows | int4 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | max:cols | int4 | — | — | — | — | 0.034 | s32 | 0e+0 | from the item 4 run |
| cuda | min:rows | int4 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | int4 | — | — | — | — | 0.033 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | int4 | — | — | — | — | 0.194 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | int4 | — | — | — | — | 0.037 | s32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | int4 | — | — | — | — | 0.037 | s32 | 0e+0 | from the item 4 run |
| cuda | transpose | int4 | — | — | — | — | 0.045 | s32 | 0e+0 | from the item 4 run |
| cuda | gemv | int4 | — | — | — | — | 0.139 | f32 | 6.92e+2 | from the item 4 run |
| cuda | gemv_t | int4 | — | — | — | — | 0.155 | f32 | 7.52e+2 | from the item 4 run |
| cuda | layout:strided | int4 | 1x1024x64x256 | — | — | — | 0.316 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:one_hot | int4 | 16384x1024 | — | — | — | 0.036 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:gather_rows | int4 | 4096 of 4096x4096 | — | — | — | 0.042 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:gather_cols | int4 | 4096 of 4096x4096 | — | — | — | 0.046 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:scatter_add_rows | int4 | 4096 into 4096x4096 | — | — | — | 0.166 | s32 | 3e+0 | small cells 1: CUDA layout |
| cuda | layout:im2col | int4 | 64x128x128 k3x3 | — | — | — | 0.052 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:col2im | int4 | 64x128x128 k3x3 | — | — | — | 0.054 | s32 | 1.1e+1 | small cells 1: CUDA layout |
| cuda | layout:interpolate_nearest | int4 | 16x512x512 to 1024x1024 | — | — | — | 0.073 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:shift | int4 | 4096x4096 by 3 | — | — | — | 0.041 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:pad2d | int4 | 4096x4093 to 4096 | — | — | — | 0.040 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:maxpool | int4 | 64x256x256 k2x2 s2 p0 | — | — | — | 0.024 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:maxpool_bwd | int4 | 64x256x256 k2x2 s2 p0 | — | — | — | 0.089 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:adaptive_maxpool | int4 | 64x256x256 to 7x7 | — | — | — | 0.048 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | layout:depthwise_conv | int4 | 64x256x256 k3x3 s1 p1 | — | — | — | 0.121 | s32 | 2.2e+1 | small cells 1: CUDA layout |
| cuda | layout:adaptive_maxpool_bwd | int4 | 64x256x256 to 7x7 | — | — | — | 0.235 | s32 | 0e+0 | small cells 1: CUDA layout |
| cuda | matmul_nt | int4 | 2048x2048x2048 Bᵀ | — | — | — | 0.101 | s8mma | 5.18e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | int4 | 2048x2048x2048 Aᵀ | — | — | — | 0.110 | s8mma | 4.62e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | int4 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.104 | s8mma | 4.97e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | int4 | 16 x 512x512x512 | — | — | — | 0.589 | s8mma | 2.47e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | int4 | 2048x2048x2048 (dA, dB) | — | — | — | 0.197 | s8mma | 4.51e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | int4 | — | — | — | — | 0.107 | s8mma | 2.47e+2 | from the item 4 run |
| cuda | topk_rows | int4 | — | — | — | — | 3.593 | s32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | int4 | — | — | — | — | 0.013 | s32 | 0e+0 | from the item 4 run |
| cuda | add | int2 | 16777216 | — | — | — | 0.018 | s8 | 0e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | int2 | 2048x2048x2048 | — | — | — | 0.107 | s8mma | 6.16e+2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | int2 | — | — | — | — | 0.014 | s32 | 0e+0 | from the item 4 run |
| cuda | map:fill | int2 | — | — | — | — | 0.014 | s32 | 0e+0 | from the item 4 run |
| cuda | map:neg | int2 | — | — | — | — | 0.016 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs | int2 | — | — | — | — | 0.017 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sign | int2 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square | int2 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu | int2 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | int2 | — | — | — | — | 0.020 | s32 | 0e+0 | from the item 4 run |
| cuda | map:scale | int2 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:affine | int2 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | int2 | — | — | — | — | 0.017 | s32 | 0e+0 | from the item 4 run |
| cuda | map:add | int2 | — | — | — | — | 0.018 | s32 | 0e+0 | from the item 4 run |
| cuda | map:sub | int2 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:mul | int2 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:div | int2 | — | — | — | — | 0.042 | s32 | 0e+0 | from the item 4 run |
| cuda | map:max | int2 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:min | int2 | — | — | — | — | 0.022 | s32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | int2 | — | — | — | — | 0.018 | s32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | int2 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where | int2 | — | — | — | — | 0.019 | s32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | int2 | — | — | — | — | 0.016 | s32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | int2 | — | — | — | — | 0.018 | s32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | int2 | — | — | — | — | 0.017 | s32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | int2 | — | — | — | — | 0.018 | s32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | int2 | — | — | — | — | 0.017 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | int2 | — | — | — | — | 0.033 | s32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | int2 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | int2 | — | — | — | — | 0.037 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | int2 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | max:whole | int2 | — | — | — | — | 0.033 | s32 | 0e+0 | from the item 4 run |
| cuda | max:rows | int2 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | max:cols | int2 | — | — | — | — | 0.035 | s32 | 0e+0 | from the item 4 run |
| cuda | min:rows | int2 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | int2 | — | — | — | — | 0.033 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | int2 | — | — | — | — | 0.191 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | int2 | — | — | — | — | 0.035 | s32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | int2 | — | — | — | — | 0.036 | s32 | 0e+0 | from the item 4 run |
| cuda | transpose | int2 | — | — | — | — | 0.041 | s32 | 0e+0 | from the item 4 run |
| cuda | gemv | int2 | — | — | — | — | 0.138 | f32 | 2.15e+3 | from the item 4 run |
| cuda | gemv_t | int2 | — | — | — | — | 0.152 | f32 | 2.15e+3 | from the item 4 run |
| cuda | layout:strided | int2 | — | — | — | — | 0.318 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | int2 | — | — | — | — | 0.036 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | int2 | — | — | — | — | 0.041 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | int2 | — | — | — | — | 0.066 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | int2 | — | — | — | — | 0.188 | s32 | 3e+0 | from the item 4 run |
| cuda | layout:im2col | int2 | — | — | — | — | 0.053 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | int2 | — | — | — | — | 0.064 | s32 | 7e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | int2 | — | — | — | — | 0.079 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | int2 | — | — | — | — | 0.044 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | int2 | — | — | — | — | 0.044 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | int2 | — | — | — | — | 0.029 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | int2 | — | — | — | — | 0.168 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | int2 | — | — | — | — | 0.669 | s32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | int2 | — | — | — | — | 0.122 | s32 | 7e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | int2 | — | — | — | — | 47.309 | s32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | int2 | 2048x2048x2048 Bᵀ | — | — | — | 0.114 | s8mma | 6.08e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | int2 | 2048x2048x2048 Aᵀ | — | — | — | 0.112 | s8mma | 6.14e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | int2 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.111 | s8mma | 6.11e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | int2 | 16 x 512x512x512 | — | — | — | 0.458 | s8mma | 1.83e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | int2 | 2048x2048x2048 (dA, dB) | — | — | — | 0.213 | s8mma | 6.24e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | int2 | — | — | — | — | 0.124 | s8mma | 2.04e+2 | from the item 4 run |
| cuda | topk_rows | int2 | — | — | — | — | 3.300 | s32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | int2 | — | — | — | — | 0.013 | s32 | 0e+0 | from the item 4 run |
| cuda | add | uint64 | 16777216 | 0.999 | add_u64 | 0e+0 | 0.519 | u64 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | uint64 | 2048x2048x2048 | 4.370 | gemm64_u64 | 0e+0 | 0.362 | u8mma | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | map:fill | uint64 | — | — | — | — | 0.165 | u64 | 0e+0 | from the item 4 run |
| cuda | map:neg | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | map:abs | uint64 | — | — | — | — | 0.685 | u64 | 0e+0 | from the item 4 run |
| cuda | map:sign | uint64 | — | — | — | — | 0.359 | u64 | 0e+0 | from the item 4 run |
| cuda | map:square | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | map:relu | uint64 | — | — | — | — | 0.685 | u64 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | map:scale | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | map:affine | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | map:clamp | uint64 | — | — | — | — | 0.683 | u64 | 0e+0 | from the item 4 run |
| cuda | map:add | uint64 | — | — | — | — | 0.520 | u64 | 0e+0 | from the item 4 run |
| cuda | map:sub | uint64 | — | — | — | — | 1.001 | u64 | 0e+0 | from the item 4 run |
| cuda | map:mul | uint64 | — | — | — | — | 1.001 | u64 | 0e+0 | from the item 4 run |
| cuda | map:div | uint64 | — | — | — | — | 0.521 | u64 | 0e+0 | from the item 4 run |
| cuda | map:max | uint64 | — | — | — | — | 0.521 | u64 | 0e+0 | from the item 4 run |
| cuda | map:min | uint64 | — | — | — | — | 0.997 | u64 | 0e+0 | from the item 4 run |
| cuda | map:axpy | uint64 | — | — | — | — | 0.521 | u64 | 0e+0 | from the item 4 run |
| cuda | map:muladd | uint64 | — | — | — | — | 0.686 | u64 | 0e+0 | from the item 4 run |
| cuda | map:where | uint64 | — | — | — | — | 0.633 | u64 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | uint64 | — | — | — | — | 0.521 | u64 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | uint64 | — | — | — | — | 0.521 | u64 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | uint64 | — | — | — | — | 0.521 | u64 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | uint64 | — | — | — | — | 0.520 | u64 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | uint64 | — | — | — | — | 0.520 | u64 | 0e+0 | from the item 4 run |
| cuda | sum:whole | uint64 | — | — | — | — | 0.393 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | uint64 | — | — | — | — | 0.389 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | uint64 | — | — | — | — | 0.393 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | uint64 | — | — | — | — | 0.390 | f32 | 0e+0 | from the item 4 run |
| cuda | max:whole | uint64 | — | — | — | — | 0.174 | u64 | 0e+0 | from the item 4 run |
| cuda | max:rows | uint64 | — | — | — | — | 0.170 | u64 | 0e+0 | from the item 4 run |
| cuda | max:cols | uint64 | — | — | — | — | 0.177 | u64 | 0e+0 | from the item 4 run |
| cuda | min:rows | uint64 | — | — | — | — | 0.170 | u64 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | uint64 | — | — | — | — | 0.326 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | uint64 | — | — | — | — | 0.525 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | uint64 | — | — | — | — | 0.357 | u64 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | uint64 | — | — | — | — | 0.358 | u64 | 0e+0 | from the item 4 run |
| cuda | transpose | uint64 | — | — | — | — | 0.680 | u64 | 0e+0 | from the item 4 run |
| cuda | gemv | uint64 | — | — | — | — | 1.425 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv_t | uint64 | — | — | — | — | 1.478 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:strided | uint64 | — | — | — | — | 0.698 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | uint64 | — | — | — | — | 0.175 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | uint64 | — | — | — | — | 0.342 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | uint64 | — | — | — | — | 0.384 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | uint64 | — | — | — | — | 0.683 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | uint64 | — | — | — | — | 0.116 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | uint64 | — | — | — | — | 0.254 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | uint64 | — | — | — | — | 0.241 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:shift | uint64 | — | — | — | — | 0.357 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | uint64 | — | — | — | — | 0.357 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | uint64 | — | — | — | — | 0.021 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | uint64 | — | — | — | — | 0.198 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | uint64 | — | — | — | — | 0.089 | u64 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | uint64 | — | — | — | — | 0.236 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | uint64 | — | — | — | — | 6.890 | u32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | uint64 | 2048x2048x2048 Bᵀ | — | — | — | 0.356 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | uint64 | 2048x2048x2048 Aᵀ | — | — | — | 0.355 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | uint64 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.564 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | uint64 | 16 x 512x512x512 | — | — | — | 1.615 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | uint64 | 2048x2048x2048 (dA, dB) | — | — | — | 0.704 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | uint64 | — | — | — | — | 0.205 | u8mma | 0e+0 | from the item 4 run |
| cuda | topk_rows | uint64 | — | — | — | — | 10.672 | u64 | 0e+0 | from the item 4 run |
| cuda | pick_rows | uint64 | — | — | — | — | 0.008 | u64 | 0e+0 | from the item 4 run |
| cuda | add | uint32 | 16777216 | 0.502 | add_u32 | 0e+0 | 0.263 | u32 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | uint32 | 2048x2048x2048 | 1.091 | gemm128_u32 | 0e+0 | 0.230 | u8mma | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | uint32 | — | — | — | — | 0.183 | u32 | 0e+0 | from the item 4 run |
| cuda | map:fill | uint32 | — | — | — | — | 0.083 | u32 | 0e+0 | from the item 4 run |
| cuda | map:neg | uint32 | — | — | — | — | 0.175 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs | uint32 | — | — | — | — | 0.344 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sign | uint32 | — | — | — | — | 0.343 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square | uint32 | — | — | — | — | 0.344 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu | uint32 | — | — | — | — | 0.343 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | uint32 | — | — | — | — | 0.343 | u32 | 0e+0 | from the item 4 run |
| cuda | map:scale | uint32 | — | — | — | — | 0.181 | u32 | 0e+0 | from the item 4 run |
| cuda | map:affine | uint32 | — | — | — | — | 0.344 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | uint32 | — | — | — | — | 0.344 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add | uint32 | — | — | — | — | 0.501 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sub | uint32 | — | — | — | — | 0.264 | u32 | 0e+0 | from the item 4 run |
| cuda | map:mul | uint32 | — | — | — | — | 0.501 | u32 | 0e+0 | from the item 4 run |
| cuda | map:div | uint32 | — | — | — | — | 0.263 | u32 | 0e+0 | from the item 4 run |
| cuda | map:max | uint32 | — | — | — | — | 0.501 | u32 | 0e+0 | from the item 4 run |
| cuda | map:min | uint32 | — | — | — | — | 0.262 | u32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | uint32 | — | — | — | — | 0.502 | u32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | uint32 | — | — | — | — | 0.345 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where | uint32 | — | — | — | — | 0.643 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | uint32 | — | — | — | — | 0.262 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | uint32 | — | — | — | — | 0.262 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | uint32 | — | — | — | — | 0.261 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | uint32 | — | — | — | — | 0.263 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | uint32 | — | — | — | — | 0.261 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | uint32 | — | — | — | — | 0.173 | u64 | 0e+0 | from the item 4 run |
| cuda | sum:rows | uint32 | — | — | — | — | 0.169 | u64 | 0e+0 | from the item 4 run |
| cuda | sum:cols | uint32 | — | — | — | — | 0.176 | u64 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | uint32 | — | — | — | — | 0.169 | u64 | 0e+0 | from the item 4 run |
| cuda | max:whole | uint32 | — | — | — | — | 0.173 | u32 | 0e+0 | from the item 4 run |
| cuda | max:rows | uint32 | — | — | — | — | 0.169 | u32 | 0e+0 | from the item 4 run |
| cuda | max:cols | uint32 | — | — | — | — | 0.175 | u32 | 0e+0 | from the item 4 run |
| cuda | min:rows | uint32 | — | — | — | — | 0.169 | u32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | uint32 | — | — | — | — | 0.169 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | uint32 | — | — | — | — | 0.513 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | uint32 | — | — | — | — | 0.182 | u32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | uint32 | — | — | — | — | 0.181 | u32 | 0e+0 | from the item 4 run |
| cuda | transpose | uint32 | — | — | — | — | 0.350 | u32 | 0e+0 | from the item 4 run |
| cuda | gemv | uint32 | — | — | — | — | 1.431 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv_t | uint32 | — | — | — | — | 0.813 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:strided | uint32 | — | — | — | — | 0.451 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | uint32 | — | — | — | — | 0.134 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | uint32 | — | — | — | — | 0.195 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | uint32 | — | — | — | — | 0.253 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | uint32 | — | — | — | — | 0.472 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | uint32 | — | — | — | — | 0.087 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | uint32 | — | — | — | — | 0.119 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | uint32 | — | — | — | — | 0.152 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | uint32 | — | — | — | — | 0.182 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | uint32 | — | — | — | — | 0.181 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | uint32 | — | — | — | — | 0.021 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | uint32 | — | — | — | — | 0.172 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | uint32 | — | — | — | — | 0.079 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | uint32 | — | — | — | — | 0.208 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | uint32 | — | — | — | — | 7.050 | u32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | uint32 | 2048x2048x2048 Bᵀ | — | — | — | 0.215 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | uint32 | 2048x2048x2048 Aᵀ | — | — | — | 0.269 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | uint32 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.272 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | uint32 | 16 x 512x512x512 | — | — | — | 1.489 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | uint32 | 2048x2048x2048 (dA, dB) | — | — | — | 0.423 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | uint32 | — | — | — | — | 0.173 | u8mma | 0e+0 | from the item 4 run |
| cuda | topk_rows | uint32 | — | — | — | — | 4.331 | u32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | uint32 | — | — | — | — | 0.008 | u32 | 0e+0 | from the item 4 run |
| cuda | add | uint16 | 16777216 | 0.140 | add_u16 | 0e+0 | 0.125 | u16 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | uint16 | 2048x2048x2048 | 1.330 | gemm128_u16 | 0e+0 | 0.181 | u8mma | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | uint16 | — | — | — | — | 0.082 | u32 | 0e+0 | from the item 4 run |
| cuda | map:fill | uint16 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | map:neg | uint16 | — | — | — | — | 0.084 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs | uint16 | — | — | — | — | 0.084 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sign | uint16 | — | — | — | — | 0.141 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square | uint16 | — | — | — | — | 0.141 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu | uint16 | — | — | — | — | 0.140 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | uint16 | — | — | — | — | 0.142 | u32 | 0e+0 | from the item 4 run |
| cuda | map:scale | uint16 | — | — | — | — | 0.140 | u32 | 0e+0 | from the item 4 run |
| cuda | map:affine | uint16 | — | — | — | — | 0.141 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | uint16 | — | — | — | — | 0.141 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add | uint16 | — | — | — | — | 0.238 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sub | uint16 | — | — | — | — | 0.240 | u32 | 0e+0 | from the item 4 run |
| cuda | map:mul | uint16 | — | — | — | — | 0.239 | u32 | 0e+0 | from the item 4 run |
| cuda | map:div | uint16 | — | — | — | — | 0.237 | u32 | 0e+0 | from the item 4 run |
| cuda | map:max | uint16 | — | — | — | — | 0.237 | u32 | 0e+0 | from the item 4 run |
| cuda | map:min | uint16 | — | — | — | — | 0.236 | u32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | uint16 | — | — | — | — | 0.235 | u32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | uint16 | — | — | — | — | 0.323 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where | uint16 | — | — | — | — | 0.323 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | uint16 | — | — | — | — | 0.239 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | uint16 | — | — | — | — | 0.234 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | uint16 | — | — | — | — | 0.238 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | uint16 | — | — | — | — | 0.235 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | uint16 | — | — | — | — | 0.237 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | uint16 | — | — | — | — | 0.029 | u64 | 0e+0 | from the item 4 run |
| cuda | sum:rows | uint16 | — | — | — | — | 0.027 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | uint16 | — | — | — | — | 0.031 | u32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | uint16 | — | — | — | — | 0.029 | u64 | 0e+0 | from the item 4 run |
| cuda | max:whole | uint16 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | max:rows | uint16 | — | — | — | — | 0.027 | u32 | 0e+0 | from the item 4 run |
| cuda | max:cols | uint16 | — | — | — | — | 0.031 | u32 | 0e+0 | from the item 4 run |
| cuda | min:rows | uint16 | — | — | — | — | 0.027 | u32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | uint16 | — | — | — | — | 0.031 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | uint16 | — | — | — | — | 0.194 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | uint16 | — | — | — | — | 0.088 | u32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | uint16 | — | — | — | — | 0.087 | u32 | 0e+0 | from the item 4 run |
| cuda | transpose | uint16 | — | — | — | — | 0.091 | u32 | 0e+0 | from the item 4 run |
| cuda | gemv | uint16 | — | — | — | — | 0.742 | f32 | 0e+0 | from the item 4 run |
| cuda | gemv_t | uint16 | — | — | — | — | 0.453 | f32 | 0e+0 | from the item 4 run |
| cuda | layout:strided | uint16 | — | — | — | — | 0.353 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | uint16 | — | — | — | — | 0.047 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | uint16 | — | — | — | — | 0.090 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | uint16 | — | — | — | — | 0.129 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | uint16 | — | — | — | — | 0.281 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | uint16 | — | — | — | — | 0.068 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | uint16 | — | — | — | — | 0.109 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | uint16 | — | — | — | — | 0.095 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | uint16 | — | — | — | — | 0.158 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | uint16 | — | — | — | — | 0.155 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | uint16 | — | — | — | — | 0.021 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | uint16 | — | — | — | — | 0.158 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | uint16 | — | — | — | — | 0.145 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | uint16 | — | — | — | — | 0.220 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | uint16 | — | — | — | — | 14.175 | u32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | uint16 | 2048x2048x2048 Bᵀ | — | — | — | 0.178 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | uint16 | 2048x2048x2048 Aᵀ | — | — | — | 0.192 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | uint16 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.179 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | uint16 | 16 x 512x512x512 | — | — | — | 1.551 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | uint16 | 2048x2048x2048 (dA, dB) | — | — | — | 0.415 | u8mma | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | uint16 | — | — | — | — | 0.204 | u8mma | 0e+0 | from the item 4 run |
| cuda | topk_rows | uint16 | — | — | — | — | 3.546 | u32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | uint16 | — | — | — | — | 0.009 | u32 | 0e+0 | from the item 4 run |
| cuda | add | uint8 | 16777216 | 0.107 | add_u8 | 0e+0 | 0.029 | u8 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | uint8 | 2048x2048x2048 | 1.286 | gemm128_u8 | 5.12e+3 | 0.093 | u8mma | 4.96e+3 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:fill | uint8 | — | — | — | — | 0.020 | u32 | 0e+0 | from the item 4 run |
| cuda | map:neg | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sign | uint8 | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square | uint8 | — | — | — | — | 0.026 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:scale | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:affine | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | uint8 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add | uint8 | — | — | — | — | 0.558 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sub | uint8 | — | — | — | — | 0.053 | u32 | 0e+0 | from the item 4 run |
| cuda | map:mul | uint8 | — | — | — | — | 0.055 | u32 | 0e+0 | from the item 4 run |
| cuda | map:div | uint8 | — | — | — | — | 0.058 | u32 | 0e+0 | from the item 4 run |
| cuda | map:max | uint8 | — | — | — | — | 0.052 | u32 | 0e+0 | from the item 4 run |
| cuda | map:min | uint8 | — | — | — | — | 0.052 | u32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | uint8 | — | — | — | — | 0.037 | u32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | uint8 | — | — | — | — | 0.074 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where | uint8 | — | — | — | — | 0.135 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | uint8 | — | — | — | — | 0.055 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | uint8 | — | — | — | — | 0.053 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | uint8 | — | — | — | — | 0.054 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | uint8 | — | — | — | — | 0.052 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | uint8 | — | — | — | — | 0.053 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | uint8 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | uint8 | — | — | — | — | 0.027 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | uint8 | — | — | — | — | 0.036 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | uint8 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | max:whole | uint8 | — | — | — | — | 0.033 | u32 | 0e+0 | from the item 4 run |
| cuda | max:rows | uint8 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | max:cols | uint8 | — | — | — | — | 0.034 | u32 | 0e+0 | from the item 4 run |
| cuda | min:rows | uint8 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | uint8 | — | — | — | — | 0.036 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | uint8 | — | — | — | — | 0.196 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | uint8 | — | — | — | — | 0.040 | u32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | uint8 | — | — | — | — | 0.040 | u32 | 0e+0 | from the item 4 run |
| cuda | transpose | uint8 | — | — | — | — | 0.046 | u32 | 0e+0 | from the item 4 run |
| cuda | gemv | uint8 | — | — | — | — | 0.129 | u32 | 1.9e+4 | from the item 4 run |
| cuda | gemv_t | uint8 | — | — | — | — | 0.149 | u32 | 1.91e+4 | from the item 4 run |
| cuda | layout:strided | uint8 | — | — | — | — | 0.312 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | uint8 | — | — | — | — | 0.036 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | uint8 | — | — | — | — | 0.045 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | uint8 | — | — | — | — | 0.046 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | uint8 | — | — | — | — | 0.173 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:im2col | uint8 | — | — | — | — | 0.056 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | uint8 | — | — | — | — | 0.050 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | uint8 | — | — | — | — | 0.079 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | uint8 | — | — | — | — | 0.043 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | uint8 | — | — | — | — | 0.043 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | uint8 | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | uint8 | — | — | — | — | 0.097 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | uint8 | — | — | — | — | 0.266 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | uint8 | — | — | — | — | 0.116 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | uint8 | — | — | — | — | 23.184 | u32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | uint8 | 2048x2048x2048 Bᵀ | — | — | — | 0.092 | u8mma | 5.04e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | uint8 | 2048x2048x2048 Aᵀ | — | — | — | 0.102 | u8mma | 5.01e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | uint8 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.103 | u8mma | 5.01e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | uint8 | 16 x 512x512x512 | — | — | — | 0.422 | u8mma | 1.24e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | uint8 | 2048x2048x2048 (dA, dB) | — | — | — | 0.186 | u8mma | 4.98e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | uint8 | — | — | — | — | 0.107 | u8mma | 1.35e+3 | from the item 4 run |
| cuda | topk_rows | uint8 | — | — | — | — | 3.542 | u32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | uint8 | — | — | — | — | 0.010 | u32 | 0e+0 | from the item 4 run |
| cuda | add | uint2 | 16777216 | — | — | — | 0.017 | u8 | 0e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | uint2 | 2048x2048x2048 | — | — | — | 0.109 | u8mma | 6.11e+2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | uint2 | — | — | — | — | 0.014 | u32 | 0e+0 | from the item 4 run |
| cuda | map:fill | uint2 | — | — | — | — | 0.012 | u32 | 0e+0 | from the item 4 run |
| cuda | map:neg | uint2 | — | — | — | — | 0.014 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs | uint2 | — | — | — | — | 0.014 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sign | uint2 | — | — | — | — | 0.017 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square | uint2 | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu | uint2 | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | uint2 | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cuda | map:scale | uint2 | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cuda | map:affine | uint2 | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | uint2 | — | — | — | — | 0.017 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add | uint2 | — | — | — | — | 0.018 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sub | uint2 | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:mul | uint2 | — | — | — | — | 0.018 | u32 | 0e+0 | from the item 4 run |
| cuda | map:div | uint2 | — | — | — | — | 0.035 | u32 | 0e+0 | from the item 4 run |
| cuda | map:max | uint2 | — | — | — | — | 0.018 | u32 | 0e+0 | from the item 4 run |
| cuda | map:min | uint2 | — | — | — | — | 0.018 | u32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | uint2 | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | uint2 | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where | uint2 | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | uint2 | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | uint2 | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | uint2 | — | — | — | — | 0.021 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | uint2 | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | uint2 | — | — | — | — | 0.021 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | uint2 | — | — | — | — | 0.035 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | uint2 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | uint2 | — | — | — | — | 0.036 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | uint2 | — | — | — | — | 0.029 | f32 | 0e+0 | from the item 4 run |
| cuda | max:whole | uint2 | — | — | — | — | 0.033 | u32 | 0e+0 | from the item 4 run |
| cuda | max:rows | uint2 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | max:cols | uint2 | — | — | — | — | 0.035 | u32 | 0e+0 | from the item 4 run |
| cuda | min:rows | uint2 | — | — | — | — | 0.029 | u32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | uint2 | — | — | — | — | 0.035 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | uint2 | — | — | — | — | 0.195 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | uint2 | — | — | — | — | 0.038 | u32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | uint2 | — | — | — | — | 0.039 | u32 | 0e+0 | from the item 4 run |
| cuda | transpose | uint2 | — | — | — | — | 0.041 | u32 | 0e+0 | from the item 4 run |
| cuda | gemv | uint2 | — | — | — | — | 0.138 | f32 | 2.19e+3 | from the item 4 run |
| cuda | gemv_t | uint2 | — | — | — | — | 0.152 | f32 | 2.19e+3 | from the item 4 run |
| cuda | layout:strided | uint2 | — | — | — | — | 0.318 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | uint2 | — | — | — | — | 0.033 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | uint2 | — | — | — | — | 0.042 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | uint2 | — | — | — | — | 0.066 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | uint2 | — | — | — | — | 0.156 | u32 | 2e+0 | from the item 4 run |
| cuda | layout:im2col | uint2 | — | — | — | — | 0.052 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | uint2 | — | — | — | — | 0.063 | u32 | 6e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | uint2 | — | — | — | — | 0.078 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | uint2 | — | — | — | — | 0.042 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | uint2 | — | — | — | — | 0.042 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | uint2 | — | — | — | — | 0.028 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | uint2 | — | — | — | — | 0.167 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | uint2 | — | — | — | — | 0.668 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | uint2 | — | — | — | — | 0.122 | u32 | 4e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | uint2 | — | — | — | — | 46.729 | u32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | uint2 | 2048x2048x2048 Bᵀ | — | — | — | 0.109 | u8mma | 6.12e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | uint2 | 2048x2048x2048 Aᵀ | — | — | — | 0.114 | u8mma | 6.19e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | uint2 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.114 | u8mma | 6.11e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | uint2 | 16 x 512x512x512 | — | — | — | 0.646 | u8mma | 1.78e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | uint2 | 2048x2048x2048 (dA, dB) | — | — | — | 0.208 | u8mma | 6.16e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | uint2 | — | — | — | — | 0.123 | u8mma | 1.93e+2 | from the item 4 run |
| cuda | topk_rows | uint2 | — | — | — | — | 3.309 | u32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | uint2 | — | — | — | — | 0.013 | u32 | 0e+0 | from the item 4 run |
| cuda | add | q8_0 | 16777216 | — | — | — | 0.986 | f32 | 7.87e-3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | q8_0 | 2048x2048x2048 | — | — | — | 0.861 | f32 | 3.09e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:fill | q8_0 | — | — | — | — | 0.551 | f32 (blocks through lane storage) | 1.83e-4 | from the item 4 run |
| cuda | map:neg | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:abs | q8_0 | — | — | — | — | 0.783 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:sign | q8_0 | — | — | — | — | 0.780 | f32 (blocks through lane storage) | 6.1e-5 | from the item 4 run |
| cuda | map:square | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 6.26e-2 | from the item 4 run |
| cuda | map:relu | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 1.56e-2 | from the item 4 run |
| cuda | map:add_scalar | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 2.36e-2 | from the item 4 run |
| cuda | map:scale | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 3.88e-3 | from the item 4 run |
| cuda | map:affine | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 3.53e-2 | from the item 4 run |
| cuda | map:clamp | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 7.87e-3 | from the item 4 run |
| cuda | map:add | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 3.15e-2 | from the item 4 run |
| cuda | map:sub | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 3.15e-2 | from the item 4 run |
| cuda | map:mul | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 6.29e-2 | from the item 4 run |
| cuda | map:div | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.59e-2 | from the item 4 run |
| cuda | map:max | q8_0 | — | — | — | — | 0.986 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:min | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:axpy | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 4.71e-2 | from the item 4 run |
| cuda | map:muladd | q8_0 | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 7.85e-2 | from the item 4 run |
| cuda | map:where | q8_0 | — | — | — | — | 1.115 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:square_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.26e-1 | from the item 4 run |
| cuda | map:abs_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.56e-2 | from the item 4 run |
| cuda | map:relu_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.56e-2 | from the item 4 run |
| cuda | map:clamp_bwd | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.56e-2 | from the item 4 run |
| cuda | map:where_bwd | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.56e-2 | from the item 4 run |
| cuda | map:sqrt | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 7.83e-3 | from the item 4 run |
| cuda | map:inv_sqrt | q8_0 | — | — | — | — | 0.782 | f32 (blocks through lane storage) | 5.64e-3 | from the item 4 run |
| cuda | map:exp | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 2.14e-1 | from the item 4 run |
| cuda | map:expm1 | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 2.54e-3 | from the item 4 run |
| cuda | map:log | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 5.45e-3 | from the item 4 run |
| cuda | map:sin | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | map:cos | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | map:tanh | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 3.93e-3 | from the item 4 run |
| cuda | map:sigmoid | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 3.86e-3 | from the item 4 run |
| cuda | map:erf | q8_0 | — | — | — | — | 1.408 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | map:gelu | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:gelu_erf | q8_0 | — | — | — | — | 1.990 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:silu | q8_0 | — | — | — | — | 0.781 | f32 (blocks through lane storage) | 1.54e-2 | from the item 4 run |
| cuda | map:leaky_relu | q8_0 | — | — | — | — | 0.782 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:pow | q8_0 | — | — | — | — | 0.783 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | map:dropout | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.95e-2 | from the item 4 run |
| cuda | map:gated_silu | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 6.16e-2 | from the item 4 run |
| cuda | map:gated_gelu | q8_0 | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 6.29e-2 | from the item 4 run |
| cuda | map:sqrt_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.12e-2 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 2.28e-2 | from the item 4 run |
| cuda | map:exp_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 8.57e-1 | from the item 4 run |
| cuda | map:log_bwd | q8_0 | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 3.22e-2 | from the item 4 run |
| cuda | map:sin_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:cos_bwd | q8_0 | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:tanh_bwd | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:sigmoid_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 3.93e-3 | from the item 4 run |
| cuda | map:silu_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.73e-2 | from the item 4 run |
| cuda | map:gelu_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.78e-2 | from the item 4 run |
| cuda | map:gelu_erf_bwd | q8_0 | — | — | — | — | 2.230 | f32 (blocks through lane storage) | 1.78e-2 | from the item 4 run |
| cuda | map:leaky_relu_bwd | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | map:pow_bwd | q8_0 | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 3.15e-1 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | q8_0 | — | — | — | — | 1.192 | f32 (blocks through lane storage) | 6.63e-2 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | q8_0 | — | — | — | — | 1.110 | f32 (blocks through lane storage) | 6.17e-2 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | q8_0 | — | — | — | — | 1.193 | f32 (blocks through lane storage) | 6.28e-2 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | q8_0 | — | — | — | — | 1.110 | f32 (blocks through lane storage) | 6.29e-2 | from the item 4 run |
| cuda | sum:whole | q8_0 | — | — | — | — | 0.033 | f32 | 3.2e-4 | from the item 4 run |
| cuda | sum:rows | q8_0 | — | — | — | — | 0.031 | f32 | 7.63e-6 | from the item 4 run |
| cuda | sum:cols | q8_0 | — | — | — | — | 0.036 | f32 | 2.67e-5 | from the item 4 run |
| cuda | sumsq:rows | q8_0 | — | — | — | — | 0.031 | f32 | 1.74e-4 | from the item 4 run |
| cuda | max:whole | q8_0 | — | — | — | — | 0.033 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | q8_0 | — | — | — | — | 0.031 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | q8_0 | — | — | — | — | 0.035 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | q8_0 | — | — | — | — | 0.031 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | q8_0 | — | — | — | — | 0.037 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | q8_0 | — | — | — | — | 0.228 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | q8_0 | — | — | — | — | 1.040 | f32 (blocks through lane storage) | 7.87e-3 | from the item 4 run |
| cuda | maprow:mul | q8_0 | — | — | — | — | 0.784 | f32 (blocks through lane storage) | 3.93e-3 | from the item 4 run |
| cuda | transpose | q8_0 | — | — | — | — | 0.788 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | gemv | q8_0 | — | — | — | — | 0.447 | s8dp blocks | 4.75e-1 | from the item 4 run |
| cuda | gemv_t | q8_0 | — | — | — | — | 0.847 | f32 (blocks through lane storage) | 5.89e-1 | from the item 4 run |
| cuda | row:log_softmax | q8_0 | — | — | — | — | 0.782 | f32 (blocks through lane storage) | 5.63e-2 | from the item 4 run |
| cuda | row:softmax_bwd | q8_0 | — | — | — | — | 0.984 | f32 (blocks through lane storage) | 3.11e-1 | from the item 4 run |
| cuda | row:log_softmax_bwd | q8_0 | — | — | — | — | 0.986 | f32 (blocks through lane storage) | 5.06e-1 | from the item 4 run |
| cuda | row:rms_norm_affine | q8_0 | — | — | — | — | 0.782 | f32 (blocks through lane storage) | 1.38e-2 | from the item 4 run |
| cuda | row:rms_norm_bwd | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 3.46e-3 | from the item 4 run |
| cuda | row:layer_norm | q8_0 | — | — | — | — | 0.786 | f32 (blocks through lane storage) | 1.75e-2 | from the item 4 run |
| cuda | row:layer_norm_bwd | q8_0 | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 3.54e-3 | from the item 4 run |
| cuda | row:weight_norm | q8_0 | — | — | — | — | 0.783 | f32 (blocks through lane storage) | 1.51e-5 | from the item 4 run |
| cuda | row:rope | q8_0 | — | — | — | — | 0.780 | f32 (blocks through lane storage) | 2.23e-2 | from the item 4 run |
| cuda | row:scale_mask_softmax | q8_0 | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.42e-3 | from the item 4 run |
| cuda | row:add_rms_norm | q8_0 | — | — | — | — | 1.557 | f32 (blocks through lane storage) | 1.94e-2 | from the item 4 run |
| cuda | row:add_layer_norm | q8_0 | — | — | — | — | 1.558 | f32 (blocks through lane storage) | 2.26e-2 | from the item 4 run |
| cuda | row:cross_entropy_bwd | q8_0 | — | — | — | — | 0.783 | f32 (blocks through lane storage) | 2.08e-3 | from the item 4 run |
| cuda | norm_params_bwd:rms | q8_0 | — | — | — | — | 0.106 | f32 | NaN | from the item 4 run |
| cuda | norm_params_bwd:layer | q8_0 | — | — | — | — | 0.113 | f32 | NaN | from the item 4 run |
| cuda | layout:strided | q8_0 | — | — | — | — | 1.006 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | layout:one_hot | q8_0 | — | — | — | — | 0.621 | f32 (blocks through lane storage) | 6.1e-5 | from the item 4 run |
| cuda | layout:gather_rows | q8_0 | — | — | — | — | 0.796 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | q8_0 | — | — | — | — | 0.855 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | layout:scatter_add_rows | q8_0 | — | — | — | — | 0.885 | f32 (blocks through lane storage) | 1.68e-2 | from the item 4 run |
| cuda | layout:im2col | q8_0 | — | — | — | — | 0.368 | f32 (blocks through lane storage) | 3.93e-3 | from the item 4 run |
| cuda | layout:col2im | q8_0 | — | — | — | — | 0.150 | f32 (blocks through lane storage) | 2.95e-2 | from the item 4 run |
| cuda | layout:interpolate_nearest | q8_0 | — | — | — | — | 0.649 | f32 (blocks through lane storage) | 3.91e-3 | from the item 4 run |
| cuda | layout:interpolate_bilinear | q8_0 | — | — | — | — | 0.669 | f32 (blocks through lane storage) | 3.92e-3 | from the item 4 run |
| cuda | layout:shift | q8_0 | — | — | — | — | 0.785 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | layout:pad2d | q8_0 | — | — | — | — | 0.786 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | layout:maxpool | q8_0 | — | — | — | — | 0.074 | f32 (blocks through lane storage) | 3.94e-3 | from the item 4 run |
| cuda | layout:avgpool | q8_0 | — | — | — | — | 0.075 | f32 (blocks through lane storage) | 3.79e-3 | from the item 4 run |
| cuda | layout:maxpool_bwd | q8_0 | — | — | — | — | 0.275 | f32 (blocks through lane storage) | 3.91e-3 | from the item 4 run |
| cuda | layout:avgpool_bwd | q8_0 | — | — | — | — | 0.261 | f32 (blocks through lane storage) | 9.77e-4 | from the item 4 run |
| cuda | layout:adaptive_maxpool | q8_0 | — | — | — | — | 0.103 | f32 (blocks through lane storage) | 3.88e-3 | from the item 4 run |
| cuda | layout:adaptive_avgpool | q8_0 | — | — | — | — | 0.102 | f32 (blocks through lane storage) | 2.32e-4 | from the item 4 run |
| cuda | layout:depthwise_conv | q8_0 | — | — | — | — | 0.240 | f32 (blocks through lane storage) | 1.95e-2 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | q8_0 | — | — | — | — | 7.166 | f32 (blocks through lane storage) | 2.85e-3 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | q8_0 | — | — | — | — | 0.309 | f32 (blocks through lane storage) | 7.6e-6 | from the item 4 run |
| cuda | matmul_nt | q8_0 | 2048x2048x2048 Bᵀ | — | — | — | 0.537 | s8 mma blocks | 3.22e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | q8_0 | 2048x2048x2048 Aᵀ | — | — | — | 1.006 | f32 (blocks through lane storage) | 3.1e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | q8_0 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 1.046 | f32 (blocks through lane storage) | 3.09e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | q8_0 | 16 x 512x512x512 | — | — | — | 1.127 | f32 (blocks through lane storage) | 1.58e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | q8_0 | 2048x2048x2048 (dA, dB) | — | — | — | 1.518 | s8 mma blocks | 3.05e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_blocks | q8_0 | 2048x2048x2048 Bᵀ | — | — | — | 0.534 | s8 mma blocks | 3.22e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | q8_0 | — | — | — | — | 3.951 | f32 (blocks through lane storage) | 2.44e-4 | from the item 4 run |
| cuda | attention_causal | q8_0 | — | — | — | — | 2.319 | f32 (blocks through lane storage) | 3.73e-3 | from the item 4 run |
| cuda | attention_fused | q8_0 | — | — | — | — | 3.952 | f32 (blocks through lane storage) | 2.44e-4 | from the item 4 run |
| cuda | attention_fused_causal | q8_0 | — | — | — | — | 2.322 | f32 (blocks through lane storage) | 3.73e-3 | from the item 4 run |
| cuda | attention_decode | q8_0 | — | — | — | — | 0.884 | f32 (blocks through lane storage) | 1.39e-4 | from the item 4 run |
| cuda | attention_bwd | q8_0 | — | — | — | — | 4.137 | f32 | 1.56e-3 | from the item 4 run |
| cuda | optimizer:sgd_momentum | q8_0 | — | — | — | — | 1.347 | f32 (blocks through lane storage); + restore of p, m, v | 3.34e-3 | from the item 4 run |
| cuda | optimizer:adam | q8_0 | — | — | — | — | 1.695 | f32 (blocks through lane storage); + restore of p, m, v | 3.92e-3 | from the item 4 run |
| cuda | optimizer:adamw | q8_0 | — | — | — | — | 1.698 | f32 (blocks through lane storage); + restore of p, m, v | 3.93e-3 | from the item 4 run |
| cuda | conv | q8_0 | — | — | — | — | 0.220 | f32 | 1.67e-1 | from the item 4 run |
| cuda | conv_bwd | q8_0 | — | — | — | — | 2.257 | f32 | 2.23e-1 | from the item 4 run |
| cuda | lstm_step | q8_0 | — | — | — | — | 1.404 | f32 | 2.12e-3 | from the item 4 run |
| cuda | gru_step | q8_0 | — | — | — | — | 0.999 | f32 | 3.58e-3 | from the item 4 run |
| cuda | lstm_step_bwd | q8_0 | — | — | — | — | 3.161 | f32 | 2.42e-3 | from the item 4 run |
| cuda | gru_step_bwd | q8_0 | — | — | — | — | 2.429 | f32 | 2.33e-3 | from the item 4 run |
| cuda | gru_seq | q8_0 | — | — | — | — | 2.676 | f32 (blocks through lane storage) | 3.34e-3 | from the item 4 run |
| cuda | gru_seq_bwd | q8_0 | — | — | — | — | 16.686 | f32 (blocks through lane storage) | 2.64e-3 | from the item 4 run |
| cuda | topk_rows | q8_0 | — | — | — | — | 8.356 | f32 (blocks through lane storage) | 3.88e-3 | from the item 4 run |
| cuda | pick_rows | q8_0 | — | — | — | — | 1.806 | f32 (blocks through lane storage) | 1.57e-2 | from the item 4 run |
| cuda | sample_rows | q8_0 | — | — | — | — | 2.419 | f32 | 1e+0 | from the item 4 run |
| cuda | cross_entropy | q8_0 | — | — | — | — | 31.682 | f32 (blocks through lane storage) (log_softmax, pick, neg) | 1.25e-1 | from the item 4 run |
| cuda | ctc | q8_0 | — | — | — | — | 0.526 | f32 (blocks through lane storage) | 6.44e+0 | from the item 4 run |
| cuda | ctc_bwd | q8_0 | — | — | — | — | 1.328 | f32 (blocks through lane storage) | 3.91e-3 | from the item 4 run |
| cuda | softmax | q8_0 | 4096x4096 | — | — | — | 0.777 | f32 (blocks through lane storage) | 8.37e-6 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | q8_0 | 4096x4096 | — | — | — | 0.778 | f32 (blocks through lane storage) | 4.85e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | q4_0 | 16777216 | — | — | — | 0.658 | f32 | 2.34e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | q4_0 | 2048x2048x2048 | — | — | — | 0.784 | f32 | 6.51e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:fill | q4_0 | — | — | — | — | 0.235 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:neg | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:abs | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:sign | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | map:square | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 1e+0 | from the item 4 run |
| cuda | map:relu | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 1.88e-1 | from the item 4 run |
| cuda | map:add_scalar | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 3.61e-1 | from the item 4 run |
| cuda | map:scale | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 3.91e-3 | from the item 4 run |
| cuda | map:affine | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 8.75e-1 | from the item 4 run |
| cuda | map:clamp | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | map:add | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 9.37e-1 | from the item 4 run |
| cuda | map:sub | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 9.38e-1 | from the item 4 run |
| cuda | map:mul | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 1.75e+0 | from the item 4 run |
| cuda | map:div | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 5.06e-1 | from the item 4 run |
| cuda | map:max | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:min | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:axpy | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 1.38e+0 | from the item 4 run |
| cuda | map:muladd | q4_0 | — | — | — | — | 0.864 | f32 (blocks through lane storage) | 2.18e+0 | from the item 4 run |
| cuda | map:where | q4_0 | — | — | — | — | 0.818 | f32 (blocks through lane storage) | 5e-1 | from the item 4 run |
| cuda | map:square_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 3.5e+0 | from the item 4 run |
| cuda | map:abs_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 5e-1 | from the item 4 run |
| cuda | map:relu_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:clamp_bwd | q4_0 | — | — | — | — | 0.664 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:where_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:sqrt | q4_0 | — | — | — | — | 0.460 | f32 (blocks through lane storage) | 1.21e-1 | from the item 4 run |
| cuda | map:inv_sqrt | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 8.82e-2 | from the item 4 run |
| cuda | map:exp | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 2.72e+0 | from the item 4 run |
| cuda | map:expm1 | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 4.04e-2 | from the item 4 run |
| cuda | map:log | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 9.81e-2 | from the item 4 run |
| cuda | map:sin | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | map:cos | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | map:tanh | q4_0 | — | — | — | — | 0.462 | f32 (blocks through lane storage) | 1.24e-1 | from the item 4 run |
| cuda | map:sigmoid | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 6.13e-2 | from the item 4 run |
| cuda | map:erf | q4_0 | — | — | — | — | 2.027 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | map:gelu | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 2.05e-1 | from the item 4 run |
| cuda | map:gelu_erf | q4_0 | — | — | — | — | 3.255 | f32 (blocks through lane storage) | 2.04e-1 | from the item 4 run |
| cuda | map:silu | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 2.44e-1 | from the item 4 run |
| cuda | map:leaky_relu | q4_0 | — | — | — | — | 0.460 | f32 (blocks through lane storage) | 2.5e-1 | from the item 4 run |
| cuda | map:pow | q4_0 | — | — | — | — | 0.461 | f32 (blocks through lane storage) | 1.88e+0 | from the item 4 run |
| cuda | map:dropout | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 5.48e-1 | from the item 4 run |
| cuda | map:gated_silu | q4_0 | — | — | — | — | 0.664 | f32 (blocks through lane storage) | 1.72e+0 | from the item 4 run |
| cuda | map:gated_gelu | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 1.75e+0 | from the item 4 run |
| cuda | map:sqrt_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 3.36e-1 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 8.24e-1 | from the item 4 run |
| cuda | map:exp_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 2.39e+1 | from the item 4 run |
| cuda | map:log_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 1.04e+0 | from the item 4 run |
| cuda | map:sin_bwd | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 5e-1 | from the item 4 run |
| cuda | map:cos_bwd | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 5e-1 | from the item 4 run |
| cuda | map:tanh_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:sigmoid_bwd | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 1.09e-1 | from the item 4 run |
| cuda | map:silu_bwd | q4_0 | — | — | — | — | 0.664 | f32 (blocks through lane storage) | 4.82e-1 | from the item 4 run |
| cuda | map:gelu_bwd | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 4.94e-1 | from the item 4 run |
| cuda | map:gelu_erf_bwd | q4_0 | — | — | — | — | 3.476 | f32 (blocks through lane storage) | 4.94e-1 | from the item 4 run |
| cuda | map:leaky_relu_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 4.38e-1 | from the item 4 run |
| cuda | map:pow_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 8.77e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | q4_0 | — | — | — | — | 0.868 | f32 (blocks through lane storage) | 1.9e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | q4_0 | — | — | — | — | 0.783 | f32 (blocks through lane storage) | 1.72e+0 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | q4_0 | — | — | — | — | 0.868 | f32 (blocks through lane storage) | 1.77e+0 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | q4_0 | — | — | — | — | 0.782 | f32 (blocks through lane storage) | 1.75e+0 | from the item 4 run |
| cuda | sum:whole | q4_0 | — | — | — | — | 0.037 | f32 | 1.83e-4 | from the item 4 run |
| cuda | sum:rows | q4_0 | — | — | — | — | 0.033 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | q4_0 | — | — | — | — | 0.039 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | q4_0 | — | — | — | — | 0.033 | f32 | 1.7e-4 | from the item 4 run |
| cuda | max:whole | q4_0 | — | — | — | — | 0.037 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | q4_0 | — | — | — | — | 0.033 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | q4_0 | — | — | — | — | 0.039 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | q4_0 | — | — | — | — | 0.033 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | q4_0 | — | — | — | — | 0.039 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | q4_0 | — | — | — | — | 0.224 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | q4_0 | — | — | — | — | 0.772 | f32 (blocks through lane storage) | 2.34e-1 | from the item 4 run |
| cuda | maprow:mul | q4_0 | — | — | — | — | 0.462 | f32 (blocks through lane storage) | 1.09e-1 | from the item 4 run |
| cuda | transpose | q4_0 | — | — | — | — | 0.464 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | gemv | q4_0 | — | — | — | — | 0.243 | s32 blocks | 1.03e+1 | from the item 4 run |
| cuda | gemv_t | q4_0 | — | — | — | — | 0.830 | f32 (blocks through lane storage) | 1.14e+1 | from the item 4 run |
| cuda | row:log_softmax | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 8.92e-1 | from the item 4 run |
| cuda | row:softmax_bwd | q4_0 | — | — | — | — | 0.661 | f32 (blocks through lane storage) | 1.89e+0 | from the item 4 run |
| cuda | row:log_softmax_bwd | q4_0 | — | — | — | — | 0.660 | f32 (blocks through lane storage) | 8.32e+0 | from the item 4 run |
| cuda | row:rms_norm_affine | q4_0 | — | — | — | — | 0.460 | f32 (blocks through lane storage) | 3.89e-1 | from the item 4 run |
| cuda | row:rms_norm_bwd | q4_0 | — | — | — | — | 0.662 | f32 (blocks through lane storage) | 9.72e-2 | from the item 4 run |
| cuda | row:layer_norm | q4_0 | — | — | — | — | 0.463 | f32 (blocks through lane storage) | 4.9e-1 | from the item 4 run |
| cuda | row:layer_norm_bwd | q4_0 | — | — | — | — | 0.663 | f32 (blocks through lane storage) | 9.7e-2 | from the item 4 run |
| cuda | row:weight_norm | q4_0 | — | — | — | — | 0.461 | f32 (blocks through lane storage) | 1.53e-5 | from the item 4 run |
| cuda | row:rope | q4_0 | — | — | — | — | 0.458 | f32 (blocks through lane storage) | 6.94e-1 | from the item 4 run |
| cuda | row:scale_mask_softmax | q4_0 | — | — | — | — | 0.661 | f32 (blocks through lane storage) | 3.11e-2 | from the item 4 run |
| cuda | row:add_rms_norm | q4_0 | — | — | — | — | 0.917 | f32 (blocks through lane storage) | 5.78e-1 | from the item 4 run |
| cuda | row:add_layer_norm | q4_0 | — | — | — | — | 0.918 | f32 (blocks through lane storage) | 6.37e-1 | from the item 4 run |
| cuda | row:cross_entropy_bwd | q4_0 | — | — | — | — | 0.459 | f32 (blocks through lane storage) | 2.14e-3 | from the item 4 run |
| cuda | norm_params_bwd:rms | q4_0 | — | — | — | — | 0.106 | f32 | NaN | from the item 4 run |
| cuda | norm_params_bwd:layer | q4_0 | — | — | — | — | 0.110 | f32 | NaN | from the item 4 run |
| cuda | layout:strided | q4_0 | — | — | — | — | 1.005 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | layout:one_hot | q4_0 | — | — | — | — | 0.303 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | q4_0 | — | — | — | — | 0.470 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | q4_0 | — | — | — | — | 0.533 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | layout:scatter_add_rows | q4_0 | — | — | — | — | 0.566 | f32 (blocks through lane storage) | 4.39e-1 | from the item 4 run |
| cuda | layout:im2col | q4_0 | — | — | — | — | 0.189 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | layout:col2im | q4_0 | — | — | — | — | 0.100 | f32 (blocks through lane storage) | 6.46e-1 | from the item 4 run |
| cuda | layout:interpolate_nearest | q4_0 | — | — | — | — | 0.328 | f32 (blocks through lane storage) | 1.09e-1 | from the item 4 run |
| cuda | layout:interpolate_bilinear | q4_0 | — | — | — | — | 0.348 | f32 (blocks through lane storage) | 1.12e-1 | from the item 4 run |
| cuda | layout:shift | q4_0 | — | — | — | — | 0.461 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | layout:pad2d | q4_0 | — | — | — | — | 0.463 | f32 (blocks through lane storage) | 1.25e-1 | from the item 4 run |
| cuda | layout:maxpool | q4_0 | — | — | — | — | 0.054 | f32 (blocks through lane storage) | 1.09e-1 | from the item 4 run |
| cuda | layout:avgpool | q4_0 | — | — | — | — | 0.056 | f32 (blocks through lane storage) | 1.06e-1 | from the item 4 run |
| cuda | layout:maxpool_bwd | q4_0 | — | — | — | — | 0.196 | f32 (blocks through lane storage) | 1.09e-1 | from the item 4 run |
| cuda | layout:avgpool_bwd | q4_0 | — | — | — | — | 0.191 | f32 (blocks through lane storage) | 2.73e-2 | from the item 4 run |
| cuda | layout:adaptive_maxpool | q4_0 | — | — | — | — | 0.102 | f32 (blocks through lane storage) | 1.17e-2 | from the item 4 run |
| cuda | layout:adaptive_avgpool | q4_0 | — | — | — | — | 0.102 | f32 (blocks through lane storage) | 4.53e-3 | from the item 4 run |
| cuda | layout:depthwise_conv | q4_0 | — | — | — | — | 0.163 | f32 (blocks through lane storage) | 4.7e-1 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | q4_0 | — | — | — | — | 7.088 | f32 (blocks through lane storage) | 1.09e-1 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | q4_0 | — | — | — | — | 0.232 | f32 (blocks through lane storage) | 1.35e-4 | from the item 4 run |
| cuda | matmul_nt | q4_0 | 2048x2048x2048 Bᵀ | — | — | — | 0.374 | s8 mma blocks | 7.47e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | q4_0 | 2048x2048x2048 Aᵀ | — | — | — | 0.927 | f32 (blocks through lane storage) | 6.72e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | q4_0 | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.965 | f32 (blocks through lane storage) | 6.17e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | q4_0 | 16 x 512x512x512 | — | — | — | 1.043 | f32 (blocks through lane storage) | 3.32e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | q4_0 | 2048x2048x2048 (dA, dB) | — | — | — | 1.274 | s8 mma blocks | 6.37e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_blocks | q4_0 | 2048x2048x2048 Bᵀ | — | — | — | 0.372 | s8 mma blocks | 7.47e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | q4_0 | — | — | — | — | 3.944 | f32 (blocks through lane storage) | 6.18e-3 | from the item 4 run |
| cuda | attention_causal | q4_0 | — | — | — | — | 2.238 | f32 (blocks through lane storage) | 1.06e-1 | from the item 4 run |
| cuda | attention_fused | q4_0 | — | — | — | — | 3.871 | f32 (blocks through lane storage) | 6.18e-3 | from the item 4 run |
| cuda | attention_fused_causal | q4_0 | — | — | — | — | 2.241 | f32 (blocks through lane storage) | 1.06e-1 | from the item 4 run |
| cuda | attention_decode | q4_0 | — | — | — | — | 0.876 | f32 (blocks through lane storage) | 3e-3 | from the item 4 run |
| cuda | attention_bwd | q4_0 | — | — | — | — | 4.071 | f32 | 3.43e-2 | from the item 4 run |
| cuda | optimizer:sgd_momentum | q4_0 | — | — | — | — | 1.007 | f32 (blocks through lane storage); + restore of p, m, v | 2.92e-3 | from the item 4 run |
| cuda | optimizer:adam | q4_0 | — | — | — | — | 1.363 | f32 (blocks through lane storage); + restore of p, m, v | 3.16e-2 | from the item 4 run |
| cuda | optimizer:adamw | q4_0 | — | — | — | — | 1.360 | f32 (blocks through lane storage); + restore of p, m, v | 3.16e-2 | from the item 4 run |
| cuda | conv | q4_0 | — | — | — | — | 0.179 | f32 | 3.39e+0 | from the item 4 run |
| cuda | conv_bwd | q4_0 | — | — | — | — | 2.229 | f32 | 4.35e+0 | from the item 4 run |
| cuda | lstm_step | q4_0 | — | — | — | — | 1.365 | f32 | 5.17e-2 | from the item 4 run |
| cuda | gru_step | q4_0 | — | — | — | — | 0.878 | f32 | 9.63e-2 | from the item 4 run |
| cuda | lstm_step_bwd | q4_0 | — | — | — | — | 2.960 | f32 | 5.03e-2 | from the item 4 run |
| cuda | gru_step_bwd | q4_0 | — | — | — | — | 2.284 | f32 | 4.61e-2 | from the item 4 run |
| cuda | gru_seq | q4_0 | — | — | — | — | 2.604 | f32 (blocks through lane storage) | 1e-1 | from the item 4 run |
| cuda | gru_seq_bwd | q4_0 | — | — | — | — | 14.923 | f32 (blocks through lane storage) | 5.31e-2 | from the item 4 run |
| cuda | topk_rows | q4_0 | — | — | — | — | 7.700 | f32 (blocks through lane storage) | 3.91e-3 | from the item 4 run |
| cuda | pick_rows | q4_0 | — | — | — | — | 0.975 | f32 (blocks through lane storage) | 4.98e-1 | from the item 4 run |
| cuda | sample_rows | q4_0 | — | — | — | — | 0.577 | f32 | 1e+0 | from the item 4 run |
| cuda | cross_entropy | q4_0 | — | — | — | — | 20.375 | f32 (blocks through lane storage) (log_softmax, pick, neg) | 1.96e+0 | from the item 4 run |
| cuda | ctc | q4_0 | — | — | — | — | 0.520 | f32 (blocks through lane storage) | 1.07e+2 | from the item 4 run |
| cuda | ctc_bwd | q4_0 | — | — | — | — | 1.212 | f32 (blocks through lane storage) | 5.63e-2 | from the item 4 run |
| cuda | softmax | q4_0 | 4096x4096 | — | — | — | 0.456 | f32 (blocks through lane storage) | 1.11e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | q4_0 | 4096x4096 | — | — | — | 0.456 | f32 (blocks through lane storage) | 4.88e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | q4_k | 16777216 | — | — | — | 1.191 | f32 | 1.33e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | q4_k | 2048x2048x2048 | — | — | — | 0.868 | f32 | 3.83e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | q4_k | — | — | — | — | 0.991 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:fill | q4_k | — | — | — | — | 0.740 | f32 (blocks through lane storage) | 7.32e-4 | from the item 4 run |
| cuda | map:neg | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 3.24e-2 | from the item 4 run |
| cuda | map:abs | q4_k | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.36e-1 | from the item 4 run |
| cuda | map:sign | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 6.61e-2 | from the item 4 run |
| cuda | map:square | q4_k | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 5.55e-1 | from the item 4 run |
| cuda | map:relu | q4_k | — | — | — | — | 0.992 | f32 (blocks through lane storage) | 1.36e-1 | from the item 4 run |
| cuda | map:add_scalar | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.99e-1 | from the item 4 run |
| cuda | map:scale | q4_k | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 1.11e-2 | from the item 4 run |
| cuda | map:affine | q4_k | — | — | — | — | 0.986 | f32 (blocks through lane storage) | 5.47e-2 | from the item 4 run |
| cuda | map:clamp | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1e-1 | from the item 4 run |
| cuda | map:add | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 5.31e-1 | from the item 4 run |
| cuda | map:sub | q4_k | — | — | — | — | 1.193 | f32 (blocks through lane storage) | 5.31e-1 | from the item 4 run |
| cuda | map:mul | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 1.07e+0 | from the item 4 run |
| cuda | map:div | q4_k | — | — | — | — | 1.193 | f32 (blocks through lane storage) | 2.7e-1 | from the item 4 run |
| cuda | map:max | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 2.66e-1 | from the item 4 run |
| cuda | map:min | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 2.66e-1 | from the item 4 run |
| cuda | map:axpy | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 7.81e-1 | from the item 4 run |
| cuda | map:muladd | q4_k | — | — | — | — | 1.401 | f32 (blocks through lane storage) | 1.32e+0 | from the item 4 run |
| cuda | map:where | q4_k | — | — | — | — | 1.345 | f32 (blocks through lane storage) | 2.68e-1 | from the item 4 run |
| cuda | map:square_bwd | q4_k | — | — | — | — | 1.193 | f32 (blocks through lane storage) | 2.15e+0 | from the item 4 run |
| cuda | map:abs_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 2.73e-1 | from the item 4 run |
| cuda | map:relu_bwd | q4_k | — | — | — | — | 1.194 | f32 (blocks through lane storage) | 2.65e-1 | from the item 4 run |
| cuda | map:clamp_bwd | q4_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 2.65e-1 | from the item 4 run |
| cuda | map:where_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 2.65e-1 | from the item 4 run |
| cuda | map:sqrt | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 6.6e-2 | from the item 4 run |
| cuda | map:inv_sqrt | q4_k | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 5.22e-2 | from the item 4 run |
| cuda | map:exp | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.96e+0 | from the item 4 run |
| cuda | map:expm1 | q4_k | — | — | — | — | 0.986 | f32 (blocks through lane storage) | 3.47e-2 | from the item 4 run |
| cuda | map:log | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 6.72e-2 | from the item 4 run |
| cuda | map:sin | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 6.67e-2 | from the item 4 run |
| cuda | map:cos | q4_k | — | — | — | — | 0.986 | f32 (blocks through lane storage) | 6.67e-2 | from the item 4 run |
| cuda | map:tanh | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 6.66e-2 | from the item 4 run |
| cuda | map:sigmoid | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 3.28e-2 | from the item 4 run |
| cuda | map:erf | q4_k | — | — | — | — | 1.411 | f32 (blocks through lane storage) | 6.67e-2 | from the item 4 run |
| cuda | map:gelu | q4_k | — | — | — | — | 0.986 | f32 (blocks through lane storage) | 1.41e-1 | from the item 4 run |
| cuda | map:gelu_erf | q4_k | — | — | — | — | 1.793 | f32 (blocks through lane storage) | 1.42e-1 | from the item 4 run |
| cuda | map:silu | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 1.43e-1 | from the item 4 run |
| cuda | map:leaky_relu | q4_k | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 1.48e-1 | from the item 4 run |
| cuda | map:pow | q4_k | — | — | — | — | 0.989 | f32 (blocks through lane storage) | 1.04e+0 | from the item 4 run |
| cuda | map:dropout | q4_k | — | — | — | — | 1.199 | f32 (blocks through lane storage) | 3.32e-1 | from the item 4 run |
| cuda | map:gated_silu | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 1.05e+0 | from the item 4 run |
| cuda | map:gated_gelu | q4_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.07e+0 | from the item 4 run |
| cuda | map:sqrt_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 1.91e-1 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | q4_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 4.15e-1 | from the item 4 run |
| cuda | map:exp_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 1.51e+1 | from the item 4 run |
| cuda | map:log_bwd | q4_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 5.64e-1 | from the item 4 run |
| cuda | map:sin_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 2.71e-1 | from the item 4 run |
| cuda | map:cos_bwd | q4_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 2.69e-1 | from the item 4 run |
| cuda | map:tanh_bwd | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 2.66e-1 | from the item 4 run |
| cuda | map:sigmoid_bwd | q4_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 6.64e-2 | from the item 4 run |
| cuda | map:silu_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 2.93e-1 | from the item 4 run |
| cuda | map:gelu_bwd | q4_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 3e-1 | from the item 4 run |
| cuda | map:gelu_erf_bwd | q4_k | — | — | — | — | 2.203 | f32 (blocks through lane storage) | 3e-1 | from the item 4 run |
| cuda | map:leaky_relu_bwd | q4_k | — | — | — | — | 1.200 | f32 (blocks through lane storage) | 2.64e-1 | from the item 4 run |
| cuda | map:pow_bwd | q4_k | — | — | — | — | 1.194 | f32 (blocks through lane storage) | 5.3e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | q4_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 1.12e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | q4_k | — | — | — | — | 1.321 | f32 (blocks through lane storage) | 1.05e+0 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | q4_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 1.07e+0 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | q4_k | — | — | — | — | 1.321 | f32 (blocks through lane storage) | 1.07e+0 | from the item 4 run |
| cuda | sum:whole | q4_k | — | — | — | — | 0.065 | f32 | 8.3e-5 | from the item 4 run |
| cuda | sum:rows | q4_k | — | — | — | — | 0.058 | f32 | 9.54e-6 | from the item 4 run |
| cuda | sum:cols | q4_k | — | — | — | — | 0.068 | f32 | 3.62e-5 | from the item 4 run |
| cuda | sumsq:rows | q4_k | — | — | — | — | 0.057 | f32 | 1.88e-4 | from the item 4 run |
| cuda | max:whole | q4_k | — | — | — | — | 0.064 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | q4_k | — | — | — | — | 0.057 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | q4_k | — | — | — | — | 0.068 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | q4_k | — | — | — | — | 0.057 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | q4_k | — | — | — | — | 0.061 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | q4_k | — | — | — | — | 0.814 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | q4_k | — | — | — | — | 0.990 | f32 (blocks through lane storage) | 1.32e-1 | from the item 4 run |
| cuda | maprow:mul | q4_k | — | — | — | — | 0.992 | f32 (blocks through lane storage) | 6.65e-2 | from the item 4 run |
| cuda | transpose | q4_k | — | — | — | — | 0.969 | f32 (blocks through lane storage) | 6.73e-2 | from the item 4 run |
| cuda | gemv | q4_k | — | — | — | — | 0.848 | f32 (blocks through lane storage) | 6.29e+0 | from the item 4 run |
| cuda | gemv_t | q4_k | — | — | — | — | 0.859 | f32 (blocks through lane storage) | 6.57e+0 | from the item 4 run |
| cuda | row:log_softmax | q4_k | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 1.14e-1 | from the item 4 run |
| cuda | row:softmax_bwd | q4_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 2.49e+0 | from the item 4 run |
| cuda | row:log_softmax_bwd | q4_k | — | — | — | — | 1.194 | f32 (blocks through lane storage) | 4.44e+0 | from the item 4 run |
| cuda | row:rms_norm_affine | q4_k | — | — | — | — | 0.990 | f32 (blocks through lane storage) | 2.33e-1 | from the item 4 run |
| cuda | row:rms_norm_bwd | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 5.79e-2 | from the item 4 run |
| cuda | row:layer_norm | q4_k | — | — | — | — | 0.994 | f32 (blocks through lane storage) | 2.86e-1 | from the item 4 run |
| cuda | row:layer_norm_bwd | q4_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 5.79e-2 | from the item 4 run |
| cuda | row:weight_norm | q4_k | — | — | — | — | 0.988 | f32 (blocks through lane storage) | 4.3e-5 | from the item 4 run |
| cuda | row:rope | q4_k | — | — | — | — | 0.987 | f32 (blocks through lane storage) | 3.73e-1 | from the item 4 run |
| cuda | row:scale_mask_softmax | q4_k | — | — | — | — | 1.204 | f32 (blocks through lane storage) | 1.75e-2 | from the item 4 run |
| cuda | row:add_rms_norm | q4_k | — | — | — | — | 1.980 | f32 (blocks through lane storage) | 3.11e-1 | from the item 4 run |
| cuda | row:add_layer_norm | q4_k | — | — | — | — | 1.984 | f32 (blocks through lane storage) | 3.56e-1 | from the item 4 run |
| cuda | row:cross_entropy_bwd | q4_k | — | — | — | — | 0.990 | f32 (blocks through lane storage) | 2.52e-3 | from the item 4 run |
| cuda | norm_params_bwd:rms | q4_k | — | — | — | — | 0.142 | f32 | NaN | from the item 4 run |
| cuda | norm_params_bwd:layer | q4_k | — | — | — | — | 0.146 | f32 | NaN | from the item 4 run |
| cuda | layout:strided | q4_k | — | — | — | — | 1.210 | f32 (blocks through lane storage) | 6.72e-2 | from the item 4 run |
| cuda | layout:one_hot | q4_k | — | — | — | — | 0.807 | f32 (blocks through lane storage) | 3.57e-4 | from the item 4 run |
| cuda | layout:gather_rows | q4_k | — | — | — | — | 1.000 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | q4_k | — | — | — | — | 1.061 | f32 (blocks through lane storage) | 6.72e-2 | from the item 4 run |
| cuda | layout:scatter_add_rows | q4_k | — | — | — | — | 1.092 | f32 (blocks through lane storage) | 2.77e-1 | from the item 4 run |
| cuda | layout:im2col | q4_k | — | — | — | — | 0.553 | f32 (blocks through lane storage) | 6.67e-2 | from the item 4 run |
| cuda | layout:col2im | q4_k | — | — | — | — | 0.189 | f32 (blocks through lane storage) | 4.18e-1 | from the item 4 run |
| cuda | layout:interpolate_nearest | q4_k | — | — | — | — | 0.843 | f32 (blocks through lane storage) | 6.2e-2 | from the item 4 run |
| cuda | layout:interpolate_bilinear | q4_k | — | — | — | — | 0.861 | f32 (blocks through lane storage) | 6.66e-2 | from the item 4 run |
| cuda | layout:shift | q4_k | — | — | — | — | 0.995 | f32 (blocks through lane storage) | 6.72e-2 | from the item 4 run |
| cuda | layout:pad2d | q4_k | — | — | — | — | 0.996 | f32 (blocks through lane storage) | 6.7e-2 | from the item 4 run |
| cuda | layout:maxpool | q4_k | — | — | — | — | 0.131 | f32 (blocks through lane storage) | 6.36e-2 | from the item 4 run |
| cuda | layout:avgpool | q4_k | — | — | — | — | 0.132 | f32 (blocks through lane storage) | 5.84e-2 | from the item 4 run |
| cuda | layout:maxpool_bwd | q4_k | — | — | — | — | 0.286 | f32 (blocks through lane storage) | 6.63e-2 | from the item 4 run |
| cuda | layout:avgpool_bwd | q4_k | — | — | — | — | 0.272 | f32 (blocks through lane storage) | 1.55e-2 | from the item 4 run |
| cuda | layout:adaptive_maxpool | q4_k | — | — | — | — | 0.134 | f32 (blocks through lane storage) | 2.97e-2 | from the item 4 run |
| cuda | layout:adaptive_avgpool | q4_k | — | — | — | — | 0.134 | f32 (blocks through lane storage) | 3.06e-3 | from the item 4 run |
| cuda | layout:depthwise_conv | q4_k | — | — | — | — | 0.249 | f32 (blocks through lane storage) | 2.95e-1 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | q4_k | — | — | — | — | 7.180 | f32 (blocks through lane storage) | 5.43e-2 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | q4_k | — | — | — | — | 0.320 | f32 (blocks through lane storage) | 6.65e-5 | from the item 4 run |
| cuda | matmul_nt | q4_k | 2048x2048x2048 Bᵀ | — | — | — | 0.998 | f32 (blocks through lane storage) | 4.49e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | q4_k | 2048x2048x2048 Aᵀ | — | — | — | 1.011 | f32 (blocks through lane storage) | 3.89e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | q4_k | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 1.053 | f32 (blocks through lane storage) | 4.31e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | q4_k | 16 x 512x512x512 | — | — | — | 1.126 | f32 (blocks through lane storage) | 1.92e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | q4_k | 2048x2048x2048 (dA, dB) | — | — | — | 2.000 | f32 (blocks through lane storage) | 3.92e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | q4_k | — | — | — | — | 3.977 | f32 (blocks through lane storage) | 3.47e-3 | from the item 4 run |
| cuda | attention_causal | q4_k | — | — | — | — | 2.327 | f32 (blocks through lane storage) | 5.93e-2 | from the item 4 run |
| cuda | attention_fused | q4_k | — | — | — | — | 3.961 | f32 (blocks through lane storage) | 3.47e-3 | from the item 4 run |
| cuda | attention_fused_causal | q4_k | — | — | — | — | 2.327 | f32 (blocks through lane storage) | 5.93e-2 | from the item 4 run |
| cuda | attention_decode | q4_k | — | — | — | — | 0.898 | f32 (blocks through lane storage) | 2.05e-3 | from the item 4 run |
| cuda | attention_bwd | q4_k | — | — | — | — | 4.314 | f32 | 2.51e-2 | from the item 4 run |
| cuda | optimizer:sgd_momentum | q4_k | — | — | — | — | 1.547 | f32 (blocks through lane storage); + restore of p, m, v | 5.87e-3 | from the item 4 run |
| cuda | optimizer:adam | q4_k | — | — | — | — | 1.900 | f32 (blocks through lane storage); + restore of p, m, v | 2.26e-2 | from the item 4 run |
| cuda | optimizer:adamw | q4_k | — | — | — | — | 1.901 | f32 (blocks through lane storage); + restore of p, m, v | 2.26e-2 | from the item 4 run |
| cuda | conv | q4_k | — | — | — | — | 0.246 | f32 | 2.08e+0 | from the item 4 run |
| cuda | conv_bwd | q4_k | — | — | — | — | 2.382 | f32 | 2.95e+0 | from the item 4 run |
| cuda | lstm_step | q4_k | — | — | — | — | 1.508 | f32 | 2.95e-2 | from the item 4 run |
| cuda | gru_step | q4_k | — | — | — | — | 1.049 | f32 | 5.6e-2 | from the item 4 run |
| cuda | lstm_step_bwd | q4_k | — | — | — | — | 3.742 | f32 | 3.21e-2 | from the item 4 run |
| cuda | gru_step_bwd | q4_k | — | — | — | — | 2.607 | f32 | 3.05e-2 | from the item 4 run |
| cuda | gru_seq | q4_k | — | — | — | — | 2.671 | f32 (blocks through lane storage) | 5.58e-2 | from the item 4 run |
| cuda | gru_seq_bwd | q4_k | — | — | — | — | 12.964 | f32 (blocks through lane storage) | 3.51e-2 | from the item 4 run |
| cuda | topk_rows | q4_k | — | — | — | — | 5.539 | f32 (blocks through lane storage) | 5.87e-2 | from the item 4 run |
| cuda | pick_rows | q4_k | — | — | — | — | 1.016 | f32 (blocks through lane storage) | 2.64e-1 | from the item 4 run |
| cuda | sample_rows | q4_k | — | — | — | — | 2.657 | f32 | 2e+0 | from the item 4 run |
| cuda | cross_entropy | q4_k | — | — | — | — | 24.634 | f32 (blocks through lane storage) (log_softmax, pick, neg) | 9.02e-1 | from the item 4 run |
| cuda | ctc | q4_k | — | — | — | — | 0.560 | f32 (blocks through lane storage) | 5.55e+1 | from the item 4 run |
| cuda | ctc_bwd | q4_k | — | — | — | — | 1.455 | f32 (blocks through lane storage) | 3.06e-2 | from the item 4 run |
| cuda | softmax | q4_k | 4096x4096 | — | — | — | 0.983 | f32 (blocks through lane storage) | 7.31e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | q4_k | 4096x4096 | — | — | — | 0.983 | f32 (blocks through lane storage) | 1.37e-3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | q6_k | 16777216 | — | — | — | 1.398 | f32 | 3.22e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | q6_k | 2048x2048x2048 | — | — | — | 0.955 | f32 | 1.25e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | q6_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:fill | q6_k | — | — | — | — | 0.960 | f32 (blocks through lane storage) | 6.2e-5 | from the item 4 run |
| cuda | map:neg | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:abs | q6_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | map:sign | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 3.34e-4 | from the item 4 run |
| cuda | map:square | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 2.54e-1 | from the item 4 run |
| cuda | map:relu | q6_k | — | — | — | — | 1.199 | f32 (blocks through lane storage) | 6.19e-2 | from the item 4 run |
| cuda | map:add_scalar | q6_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 9.58e-2 | from the item 4 run |
| cuda | map:scale | q6_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 3.75e-3 | from the item 4 run |
| cuda | map:affine | q6_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 1.44e-1 | from the item 4 run |
| cuda | map:clamp | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 3.22e-2 | from the item 4 run |
| cuda | map:add | q6_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 1.29e-1 | from the item 4 run |
| cuda | map:sub | q6_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 1.29e-1 | from the item 4 run |
| cuda | map:mul | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 2.53e-1 | from the item 4 run |
| cuda | map:div | q6_k | — | — | — | — | 1.405 | f32 (blocks through lane storage) | 6.78e-2 | from the item 4 run |
| cuda | map:max | q6_k | — | — | — | — | 1.405 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:min | q6_k | — | — | — | — | 1.405 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:axpy | q6_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 1.92e-1 | from the item 4 run |
| cuda | map:muladd | q6_k | — | — | — | — | 1.613 | f32 (blocks through lane storage) | 3.22e-1 | from the item 4 run |
| cuda | map:where | q6_k | — | — | — | — | 1.541 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:square_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 5.07e-1 | from the item 4 run |
| cuda | map:abs_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 6.11e-2 | from the item 4 run |
| cuda | map:relu_bwd | q6_k | — | — | — | — | 1.408 | f32 (blocks through lane storage) | 6.19e-2 | from the item 4 run |
| cuda | map:clamp_bwd | q6_k | — | — | — | — | 1.408 | f32 (blocks through lane storage) | 6.19e-2 | from the item 4 run |
| cuda | map:where_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 6.18e-2 | from the item 4 run |
| cuda | map:sqrt | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 3.22e-2 | from the item 4 run |
| cuda | map:inv_sqrt | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 2.41e-2 | from the item 4 run |
| cuda | map:exp | q6_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 8.79e-1 | from the item 4 run |
| cuda | map:expm1 | q6_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 1.04e-2 | from the item 4 run |
| cuda | map:log | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 2.23e-2 | from the item 4 run |
| cuda | map:sin | q6_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | map:cos | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | map:tanh | q6_k | — | — | — | — | 1.198 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | map:sigmoid | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.58e-2 | from the item 4 run |
| cuda | map:erf | q6_k | — | — | — | — | 2.209 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | map:gelu | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 6.43e-2 | from the item 4 run |
| cuda | map:gelu_erf | q6_k | — | — | — | — | 3.248 | f32 (blocks through lane storage) | 6.43e-2 | from the item 4 run |
| cuda | map:silu | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 6.21e-2 | from the item 4 run |
| cuda | map:leaky_relu | q6_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:pow | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 5.04e-1 | from the item 4 run |
| cuda | map:dropout | q6_k | — | — | — | — | 1.407 | f32 (blocks through lane storage) | 7.66e-2 | from the item 4 run |
| cuda | map:gated_silu | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 2.53e-1 | from the item 4 run |
| cuda | map:gated_gelu | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 2.58e-1 | from the item 4 run |
| cuda | map:sqrt_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 4.78e-2 | from the item 4 run |
| cuda | map:inv_sqrt_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 1.06e-1 | from the item 4 run |
| cuda | map:exp_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 3.51e+0 | from the item 4 run |
| cuda | map:log_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 1.43e-1 | from the item 4 run |
| cuda | map:sin_bwd | q6_k | — | — | — | — | 1.405 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:cos_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:tanh_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:sigmoid_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | map:silu_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 7.09e-2 | from the item 4 run |
| cuda | map:gelu_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 7.28e-2 | from the item 4 run |
| cuda | map:gelu_erf_bwd | q6_k | — | — | — | — | 3.483 | f32 (blocks through lane storage) | 7.28e-2 | from the item 4 run |
| cuda | map:leaky_relu_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | map:pow_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 1.29e+0 | from the item 4 run |
| cuda | map:gated_silu_bwd_x | q6_k | — | — | — | — | 1.615 | f32 (blocks through lane storage) | 2.71e-1 | from the item 4 run |
| cuda | map:gated_silu_bwd_y | q6_k | — | — | — | — | 1.530 | f32 (blocks through lane storage) | 2.52e-1 | from the item 4 run |
| cuda | map:gated_gelu_bwd_x | q6_k | — | — | — | — | 1.616 | f32 (blocks through lane storage) | 2.58e-1 | from the item 4 run |
| cuda | map:gated_gelu_bwd_y | q6_k | — | — | — | — | 1.531 | f32 (blocks through lane storage) | 2.57e-1 | from the item 4 run |
| cuda | sum:whole | q6_k | — | — | — | — | 0.060 | f32 | 9.95e-5 | from the item 4 run |
| cuda | sum:rows | q6_k | — | — | — | — | 0.054 | f32 | 1.31e-5 | from the item 4 run |
| cuda | sum:cols | q6_k | — | — | — | — | 0.066 | f32 | 3.08e-5 | from the item 4 run |
| cuda | sumsq:rows | q6_k | — | — | — | — | 0.054 | f32 | 1.83e-4 | from the item 4 run |
| cuda | max:whole | q6_k | — | — | — | — | 0.060 | f32 | 0e+0 | from the item 4 run |
| cuda | max:rows | q6_k | — | — | — | — | 0.054 | f32 | 0e+0 | from the item 4 run |
| cuda | max:cols | q6_k | — | — | — | — | 0.066 | f32 | 0e+0 | from the item 4 run |
| cuda | min:rows | q6_k | — | — | — | — | 0.054 | f32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | q6_k | — | — | — | — | 0.058 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | q6_k | — | — | — | — | 0.818 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | q6_k | — | — | — | — | 1.196 | f32 (blocks through lane storage) | 3.22e-2 | from the item 4 run |
| cuda | maprow:mul | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.58e-2 | from the item 4 run |
| cuda | transpose | q6_k | — | — | — | — | 1.181 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | gemv | q6_k | — | — | — | — | 0.870 | f32 (blocks through lane storage) | 1.89e+0 | from the item 4 run |
| cuda | gemv_t | q6_k | — | — | — | — | 0.883 | f32 (blocks through lane storage) | 2.16e+0 | from the item 4 run |
| cuda | row:log_softmax | q6_k | — | — | — | — | 1.191 | f32 (blocks through lane storage) | 2.31e-1 | from the item 4 run |
| cuda | row:softmax_bwd | q6_k | — | — | — | — | 1.406 | f32 (blocks through lane storage) | 1.27e+0 | from the item 4 run |
| cuda | row:log_softmax_bwd | q6_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 2.05e+0 | from the item 4 run |
| cuda | row:rms_norm_affine | q6_k | — | — | — | — | 1.195 | f32 (blocks through lane storage) | 5.53e-2 | from the item 4 run |
| cuda | row:rms_norm_bwd | q6_k | — | — | — | — | 1.404 | f32 (blocks through lane storage) | 1.43e-2 | from the item 4 run |
| cuda | row:layer_norm | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 7.14e-2 | from the item 4 run |
| cuda | row:layer_norm_bwd | q6_k | — | — | — | — | 1.407 | f32 (blocks through lane storage) | 1.43e-2 | from the item 4 run |
| cuda | row:weight_norm | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.17e-4 | from the item 4 run |
| cuda | row:rope | q6_k | — | — | — | — | 1.192 | f32 (blocks through lane storage) | 9.1e-2 | from the item 4 run |
| cuda | row:scale_mask_softmax | q6_k | — | — | — | — | 1.410 | f32 (blocks through lane storage) | 5.7e-3 | from the item 4 run |
| cuda | row:add_rms_norm | q6_k | — | — | — | — | 2.396 | f32 (blocks through lane storage) | 7.84e-2 | from the item 4 run |
| cuda | row:add_layer_norm | q6_k | — | — | — | — | 2.399 | f32 (blocks through lane storage) | 9.2e-2 | from the item 4 run |
| cuda | row:cross_entropy_bwd | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 2.1e-3 | from the item 4 run |
| cuda | norm_params_bwd:rms | q6_k | — | — | — | — | 0.123 | f32 | NaN | from the item 4 run |
| cuda | norm_params_bwd:layer | q6_k | — | — | — | — | 0.126 | f32 | NaN | from the item 4 run |
| cuda | layout:strided | q6_k | — | — | — | — | 1.427 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:one_hot | q6_k | — | — | — | — | 1.014 | f32 (blocks through lane storage) | 3.34e-4 | from the item 4 run |
| cuda | layout:gather_rows | q6_k | — | — | — | — | 1.207 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | q6_k | — | — | — | — | 1.269 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:scatter_add_rows | q6_k | — | — | — | — | 1.301 | f32 (blocks through lane storage) | 6.79e-2 | from the item 4 run |
| cuda | layout:im2col | q6_k | — | — | — | — | 0.674 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:col2im | q6_k | — | — | — | — | 0.262 | f32 (blocks through lane storage) | 1.19e-1 | from the item 4 run |
| cuda | layout:interpolate_nearest | q6_k | — | — | — | — | 1.044 | f32 (blocks through lane storage) | 1.52e-2 | from the item 4 run |
| cuda | layout:interpolate_bilinear | q6_k | — | — | — | — | 1.064 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:shift | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:pad2d | q6_k | — | — | — | — | 1.197 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:maxpool | q6_k | — | — | — | — | 0.195 | f32 (blocks through lane storage) | 1.61e-2 | from the item 4 run |
| cuda | layout:avgpool | q6_k | — | — | — | — | 0.195 | f32 (blocks through lane storage) | 1.56e-2 | from the item 4 run |
| cuda | layout:maxpool_bwd | q6_k | — | — | — | — | 0.368 | f32 (blocks through lane storage) | 1.55e-2 | from the item 4 run |
| cuda | layout:avgpool_bwd | q6_k | — | — | — | — | 0.358 | f32 (blocks through lane storage) | 3.81e-3 | from the item 4 run |
| cuda | layout:adaptive_maxpool | q6_k | — | — | — | — | 0.152 | f32 (blocks through lane storage) | 1.07e-2 | from the item 4 run |
| cuda | layout:adaptive_avgpool | q6_k | — | — | — | — | 0.152 | f32 (blocks through lane storage) | 9.08e-4 | from the item 4 run |
| cuda | layout:depthwise_conv | q6_k | — | — | — | — | 0.334 | f32 (blocks through lane storage) | 7.98e-2 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | q6_k | — | — | — | — | 7.262 | f32 (blocks through lane storage) | 1.33e-2 | from the item 4 run |
| cuda | layout:adaptive_avgpool_bwd | q6_k | — | — | — | — | 0.404 | f32 (blocks through lane storage) | 1.17e-4 | from the item 4 run |
| cuda | matmul_nt | q6_k | 2048x2048x2048 Bᵀ | — | — | — | 1.086 | f32 (blocks through lane storage) | 1.24e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | q6_k | 2048x2048x2048 Aᵀ | — | — | — | 1.099 | f32 (blocks through lane storage) | 1.22e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | q6_k | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 1.141 | f32 (blocks through lane storage) | 1.22e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | q6_k | 16 x 512x512x512 | — | — | — | 1.219 | f32 (blocks through lane storage) | 6.5e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | q6_k | 2048x2048x2048 (dA, dB) | — | — | — | 2.176 | f32 (blocks through lane storage) | 1.29e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | attention | q6_k | — | — | — | — | 4.058 | f32 (blocks through lane storage) | 1.02e-3 | from the item 4 run |
| cuda | attention_causal | q6_k | — | — | — | — | 2.413 | f32 (blocks through lane storage) | 1.51e-2 | from the item 4 run |
| cuda | attention_fused | q6_k | — | — | — | — | 4.062 | f32 (blocks through lane storage) | 1.02e-3 | from the item 4 run |
| cuda | attention_fused_causal | q6_k | — | — | — | — | 2.412 | f32 (blocks through lane storage) | 1.51e-2 | from the item 4 run |
| cuda | attention_decode | q6_k | — | — | — | — | 0.917 | f32 (blocks through lane storage) | 5.47e-4 | from the item 4 run |
| cuda | attention_bwd | q6_k | — | — | — | — | 4.509 | f32 | 6.59e-3 | from the item 4 run |
| cuda | optimizer:sgd_momentum | q6_k | — | — | — | — | 1.766 | f32 (blocks through lane storage); + restore of p, m, v | 3.34e-3 | from the item 4 run |
| cuda | optimizer:adam | q6_k | — | — | — | — | 2.119 | f32 (blocks through lane storage); + restore of p, m, v | 1.44e-2 | from the item 4 run |
| cuda | optimizer:adamw | q6_k | — | — | — | — | 2.115 | f32 (blocks through lane storage); + restore of p, m, v | 1.44e-2 | from the item 4 run |
| cuda | conv | q6_k | — | — | — | — | 0.311 | f32 | 6.79e-1 | from the item 4 run |
| cuda | conv_bwd | q6_k | — | — | — | — | 2.473 | f32 | 8.92e-1 | from the item 4 run |
| cuda | lstm_step | q6_k | — | — | — | — | 1.648 | f32 | 8.59e-3 | from the item 4 run |
| cuda | gru_step | q6_k | — | — | — | — | 1.025 | f32 | 1.42e-2 | from the item 4 run |
| cuda | lstm_step_bwd | q6_k | — | — | — | — | 3.741 | f32 | 9.36e-3 | from the item 4 run |
| cuda | gru_step_bwd | q6_k | — | — | — | — | 2.888 | f32 | 9.11e-3 | from the item 4 run |
| cuda | gru_seq | q6_k | — | — | — | — | 2.743 | f32 (blocks through lane storage) | 1.39e-2 | from the item 4 run |
| cuda | gru_seq_bwd | q6_k | — | — | — | — | 13.105 | f32 (blocks through lane storage) | 1.05e-2 | from the item 4 run |
| cuda | topk_rows | q6_k | — | — | — | — | 7.946 | f32 (blocks through lane storage) | 0e+0 | from the item 4 run |
| cuda | pick_rows | q6_k | — | — | — | — | 1.053 | f32 (blocks through lane storage) | 6.45e-2 | from the item 4 run |
| cuda | sample_rows | q6_k | — | — | — | — | 8.404 | f32 | 1e+0 | from the item 4 run |
| cuda | cross_entropy | q6_k | — | — | — | — | 34.853 | f32 (blocks through lane storage) (log_softmax, pick, neg) | 5.08e-1 | from the item 4 run |
| cuda | ctc | q6_k | — | — | — | — | 0.565 | f32 (blocks through lane storage) | 2.76e+1 | from the item 4 run |
| cuda | ctc_bwd | q6_k | — | — | — | — | 1.567 | f32 (blocks through lane storage) | 1.54e-2 | from the item 4 run |
| cuda | softmax | q6_k | 4096x4096 | — | — | — | 1.196 | f32 (blocks through lane storage) | 1.17e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | rms_norm | q6_k | 4096x4096 | — | — | — | 1.198 | f32 (blocks through lane storage) | 4.69e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | add | bool | 16777216 | — | — | — | 0.017 | u8 | 1e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | 0.122 | u8mma | 1.27e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | map:copy | bool | — | — | — | — | 0.013 | u32 | 0e+0 | from the item 4 run |
| cuda | map:fill | bool | — | — | — | — | 0.010 | u32 | 0e+0 | from the item 4 run |
| cuda | map:neg | bool | — | — | — | — | 0.015 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs | bool | — | — | — | — | 0.013 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sign | bool | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square | bool | — | — | — | — | 0.014 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu | bool | — | — | — | — | 0.015 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add_scalar | bool | — | — | — | — | 0.015 | u32 | 0e+0 | from the item 4 run |
| cuda | map:scale | bool | — | — | — | — | 0.014 | u32 | 0e+0 | from the item 4 run |
| cuda | map:affine | bool | — | — | — | — | 0.014 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp | bool | — | — | — | — | 0.015 | u32 | 0e+0 | from the item 4 run |
| cuda | map:add | bool | — | — | — | — | 0.017 | u32 | 0e+0 | from the item 4 run |
| cuda | map:sub | bool | — | — | — | — | 0.017 | u32 | 0e+0 | from the item 4 run |
| cuda | map:mul | bool | — | — | — | — | 0.018 | u32 | 0e+0 | from the item 4 run |
| cuda | map:div | bool | — | — | — | — | — | — | — | — |
| cuda | map:max | bool | — | — | — | — | 0.019 | u32 | 0e+0 | from the item 4 run |
| cuda | map:min | bool | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:axpy | bool | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:muladd | bool | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where | bool | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:square_bwd | bool | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:abs_bwd | bool | — | — | — | — | 0.024 | u32 | 0e+0 | from the item 4 run |
| cuda | map:relu_bwd | bool | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | map:clamp_bwd | bool | — | — | — | — | 0.023 | u32 | 0e+0 | from the item 4 run |
| cuda | map:where_bwd | bool | — | — | — | — | 0.022 | u32 | 0e+0 | from the item 4 run |
| cuda | sum:whole | bool | — | — | — | — | 0.037 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:rows | bool | — | — | — | — | 0.032 | f32 | 0e+0 | from the item 4 run |
| cuda | sum:cols | bool | — | — | — | — | 0.038 | f32 | 0e+0 | from the item 4 run |
| cuda | sumsq:rows | bool | — | — | — | — | 0.032 | f32 | 0e+0 | from the item 4 run |
| cuda | max:whole | bool | — | — | — | — | 0.036 | u32 | 0e+0 | from the item 4 run |
| cuda | max:rows | bool | — | — | — | — | 0.031 | u32 | 0e+0 | from the item 4 run |
| cuda | max:cols | bool | — | — | — | — | 0.036 | u32 | 0e+0 | from the item 4 run |
| cuda | min:rows | bool | — | — | — | — | 0.032 | u32 | 0e+0 | from the item 4 run |
| cuda | argmax:rows | bool | — | — | — | — | 0.034 | s32 index | 0e+0 | from the item 4 run |
| cuda | argmax:cols | bool | — | — | — | — | 0.199 | s32 index | 0e+0 | from the item 4 run |
| cuda | maprow:add | bool | — | — | — | — | 0.039 | u32 | 0e+0 | from the item 4 run |
| cuda | maprow:mul | bool | — | — | — | — | 0.037 | u32 | 0e+0 | from the item 4 run |
| cuda | transpose | bool | — | — | — | — | 0.044 | u32 | 0e+0 | from the item 4 run |
| cuda | gemv | bool | — | — | — | — | 0.138 | f32 | 4.78e+3 | from the item 4 run |
| cuda | gemv_t | bool | — | — | — | — | 0.155 | f32 | 4.8e+3 | from the item 4 run |
| cuda | layout:strided | bool | — | — | — | — | 0.286 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:one_hot | bool | — | — | — | — | 0.036 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_rows | bool | — | — | — | — | 0.040 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:gather_cols | bool | — | — | — | — | 0.113 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:scatter_add_rows | bool | — | — | — | — | 0.158 | u32 | 4e+0 | from the item 4 run |
| cuda | layout:im2col | bool | — | — | — | — | 0.054 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:col2im | bool | — | — | — | — | 0.075 | u32 | 8e+0 | from the item 4 run |
| cuda | layout:interpolate_nearest | bool | — | — | — | — | 0.074 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:shift | bool | — | — | — | — | 0.042 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:pad2d | bool | — | — | — | — | 0.041 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool | bool | — | — | — | — | 0.033 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:maxpool_bwd | bool | — | — | — | — | 0.197 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool | bool | — | — | — | — | 0.930 | u32 | 0e+0 | from the item 4 run |
| cuda | layout:depthwise_conv | bool | — | — | — | — | 0.130 | u32 | 8e+0 | from the item 4 run |
| cuda | layout:adaptive_maxpool_bwd | bool | — | — | — | — | 54.973 | u32 | 0e+0 | from the item 4 run |
| cuda | matmul_nt | bool | 2048x2048x2048 Bᵀ | — | — | — | 0.123 | u8mma | 1.27e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tn | bool | 2048x2048x2048 Aᵀ | — | — | — | 0.123 | u8mma | 1.26e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_tt | bool | 2048x2048x2048 Aᵀ Bᵀ | — | — | — | 0.123 | u8mma | 1.26e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_batched | bool | 16 x 512x512x512 | — | — | — | 0.472 | u8mma | 3.48e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | matmul_bwd | bool | 2048x2048x2048 (dA, dB) | — | — | — | 0.231 | u8mma | 1.25e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cuda | conv | bool | — | — | — | — | 0.114 | u8mma | 3.73e+2 | from the item 4 run |
| cuda | topk_rows | bool | — | — | — | — | 3.061 | u32 | 0e+0 | from the item 4 run |
| cuda | pick_rows | bool | — | — | — | — | 0.016 | u32 | 0e+0 | from the item 4 run |
| cpu | add | float64 | 16777216 | 6.950 | Vector<Double> | 0e+0 | 6.910 | f64 | 0e+0 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | float64 | 1024x1024x1024 | 18.234 | Vector<Double> | 1.14e-13 | 3.290 | f64 | 1.14e-13 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | float64 | — | — | — | — | 4.750 | f64 | 0e+0 | from the item 4 run |
| cpu | map:fill | float64 | — | — | — | — | 2.483 | f64 | 0e+0 | from the item 4 run |
| cpu | map:neg | float64 | — | — | — | — | 4.677 | f64 | 0e+0 | from the item 4 run |
| cpu | map:abs | float64 | — | — | — | — | 4.744 | f64 | 0e+0 | from the item 4 run |
| cpu | map:sign | float64 | — | — | — | — | 4.692 | f64 | 0e+0 | from the item 4 run |
| cpu | map:square | float64 | — | — | — | — | 4.949 | f64 | 0e+0 | from the item 4 run |
| cpu | map:relu | float64 | — | — | — | — | 4.625 | f64 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | float64 | — | — | — | — | 4.886 | f64 | 0e+0 | from the item 4 run |
| cpu | map:scale | float64 | — | — | — | — | 4.714 | f64 | 0e+0 | from the item 4 run |
| cpu | map:affine | float64 | — | — | — | — | 4.704 | f64 | 0e+0 | from the item 4 run |
| cpu | map:clamp | float64 | — | — | — | — | 4.669 | f64 | 0e+0 | from the item 4 run |
| cpu | map:add | float64 | — | — | — | — | 6.692 | f64 | 0e+0 | from the item 4 run |
| cpu | map:sub | float64 | — | — | — | — | 6.820 | f64 | 0e+0 | from the item 4 run |
| cpu | map:mul | float64 | — | — | — | — | 6.680 | f64 | 0e+0 | from the item 4 run |
| cpu | map:div | float64 | — | — | — | — | 6.674 | f64 | 0e+0 | from the item 4 run |
| cpu | map:max | float64 | — | — | — | — | 6.617 | f64 | 0e+0 | from the item 4 run |
| cpu | map:min | float64 | — | — | — | — | 6.640 | f64 | 0e+0 | from the item 4 run |
| cpu | map:axpy | float64 | — | — | — | — | 6.710 | f64 | 0e+0 | from the item 4 run |
| cpu | map:muladd | float64 | — | — | — | — | 8.615 | f64 | 0e+0 | from the item 4 run |
| cpu | map:where | float64 | — | — | — | — | 8.531 | f64 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | float64 | — | — | — | — | 6.615 | f64 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | float64 | — | — | — | — | 6.748 | f64 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | float64 | — | — | — | — | 6.751 | f64 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | float64 | — | — | — | — | 6.913 | f64 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | float64 | — | — | — | — | 6.899 | f64 | 0e+0 | from the item 4 run |
| cpu | map:sqrt | float64 | — | — | — | — | 4.834 | f64 | 0e+0 | from the item 4 run |
| cpu | map:inv_sqrt | float64 | — | — | — | — | 4.870 | f64 | 0e+0 | from the item 4 run |
| cpu | map:exp | float64 | — | — | — | — | 4.771 | f64 | 7.11e-15 | small cells 2: CPU map / rows |
| cpu | map:expm1 | float64 | — | — | — | — | 4.926 | f64 | 1.11e-16 | from the item 4 run |
| cpu | map:log | float64 | — | — | — | — | 5.037 | f64 | 2.22e-16 | from the item 4 run |
| cpu | map:sin | float64 | — | — | — | — | 4.903 | f64 | 2.22e-16 | from the item 4 run |
| cpu | map:cos | float64 | — | — | — | — | 4.956 | f64 | 1.11e-16 | from the item 4 run |
| cpu | map:tanh | float64 | — | — | — | — | 5.149 | f64 | 3.33e-16 | from the item 4 run |
| cpu | map:sigmoid | float64 | — | — | — | — | 4.812 | f64 | 2.22e-16 | small cells 2: CPU map / rows |
| cpu | map:erf | float64 | — | — | — | — | 71.471 | f64 | 6.66e-16 | from the item 4 run |
| cpu | map:gelu | float64 | — | — | — | — | 4.924 | f64 | 8.88e-16 | from the item 4 run |
| cpu | map:gelu_erf | float64 | — | — | — | — | 338.764 | f64 | 4.44e-16 | from the item 4 run |
| cpu | map:silu | float64 | — | — | — | — | 4.847 | f64 | 8.88e-16 | from the item 4 run |
| cpu | map:leaky_relu | float64 | — | — | — | — | 4.703 | f64 | 0e+0 | from the item 4 run |
| cpu | map:pow | float64 | — | — | — | — | 8.774 | f64 | 2.84e-14 | from the item 4 run |
| cpu | map:dropout | float64 | — | — | — | — | 6.882 | f64 | 0e+0 | from the item 4 run |
| cpu | map:gated_silu | float64 | — | — | — | — | 6.541 | f64 | 3.55e-15 | from the item 4 run |
| cpu | map:gated_gelu | float64 | — | — | — | — | 6.570 | f64 | 3.55e-15 | from the item 4 run |
| cpu | map:sqrt_bwd | float64 | — | — | — | — | 6.665 | f64 | 0e+0 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | float64 | — | — | — | — | 6.877 | f64 | 1.78e-15 | from the item 4 run |
| cpu | map:exp_bwd | float64 | — | — | — | — | 6.602 | f64 | 2.84e-14 | from the item 4 run |
| cpu | map:log_bwd | float64 | — | — | — | — | 6.723 | f64 | 0e+0 | from the item 4 run |
| cpu | map:sin_bwd | float64 | — | — | — | — | 6.985 | f64 | 4.44e-16 | from the item 4 run |
| cpu | map:cos_bwd | float64 | — | — | — | — | 6.837 | f64 | 4.44e-16 | from the item 4 run |
| cpu | map:tanh_bwd | float64 | — | — | — | — | 8.707 | f64 | 3.11e-15 | from the item 4 run |
| cpu | map:sigmoid_bwd | float64 | — | — | — | — | 8.935 | f64 | 4.44e-16 | from the item 4 run |
| cpu | map:silu_bwd | float64 | — | — | — | — | 9.030 | f64 | 2.66e-15 | from the item 4 run |
| cpu | map:gelu_bwd | float64 | — | — | — | — | 9.731 | f64 | 1.78e-15 | from the item 4 run |
| cpu | map:gelu_erf_bwd | float64 | — | — | — | — | 382.496 | f64 | 1.78e-15 | from the item 4 run |
| cpu | map:leaky_relu_bwd | float64 | — | — | — | — | 6.675 | f64 | 0e+0 | from the item 4 run |
| cpu | map:pow_bwd | float64 | — | — | — | — | 14.030 | f64 | 4.26e-14 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | float64 | — | — | — | — | 9.446 | f64 | 7.11e-15 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | float64 | — | — | — | — | 8.562 | f64 | 3.55e-15 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | float64 | — | — | — | — | 9.795 | f64 | 7.11e-15 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | float64 | — | — | — | — | 8.592 | f64 | 3.55e-15 | from the item 4 run |
| cpu | sum:whole | float64 | — | — | — | — | 1.318 | f64 | 0e+0 | 6j part 4: CPU reductions |
| cpu | sum:rows | float64 | — | — | — | — | 1.291 | f64 | 0e+0 | 6j part 4: CPU reductions |
| cpu | sum:cols | float64 | — | — | — | — | 1.546 | f64 | 0e+0 | 6j part 4: CPU reductions |
| cpu | sumsq:rows | float64 | — | — | — | — | 1.303 | f64 | 2.14e-11 | 6j part 4: CPU reductions |
| cpu | max:whole | float64 | — | — | — | — | 1.748 | f64 | 0e+0 | from the item 4 run |
| cpu | max:rows | float64 | — | — | — | — | 1.659 | f64 | 0e+0 | from the item 4 run |
| cpu | max:cols | float64 | — | — | — | — | 3.802 | f64 | 0e+0 | from the item 4 run |
| cpu | min:rows | float64 | — | — | — | — | 1.580 | f64 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | float64 | — | — | — | — | 1.477 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | float64 | — | — | — | — | 2.325 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | float64 | — | — | — | — | 4.671 | f64 | 0e+0 | 6j part 4: CPU reductions |
| cpu | maprow:mul | float64 | — | — | — | — | 4.635 | f64 | 0e+0 | 6j part 4: CPU reductions |
| cpu | transpose | float64 | — | — | — | — | 6.053 | f64 | 0e+0 | 6j part 4: CPU reductions |
| cpu | gemv | float64 | — | — | — | — | 6.421 | f64 | 6.11e-13 | 6j part 4: CPU reductions |
| cpu | gemv_t | float64 | — | — | — | — | 7.168 | f64 | 5.97e-13 | 6j part 4: CPU reductions |
| cpu | row:log_softmax | float64 | — | — | — | — | 4.709 | f64 | 7.11e-15 | small cells 2: CPU map / rows |
| cpu | row:softmax_bwd | float64 | — | — | — | — | 6.542 | f64 | 3.27e-13 | from the item 4 run |
| cpu | row:log_softmax_bwd | float64 | — | — | — | — | 6.775 | f64 | 2.84e-14 | from the item 4 run |
| cpu | row:rms_norm_affine | float64 | — | — | — | — | 4.732 | f64 | 2.62e-14 | from the item 4 run |
| cpu | row:rms_norm_bwd | float64 | — | — | — | — | 6.648 | f64 | 6.88e-15 | from the item 4 run |
| cpu | row:layer_norm | float64 | 4096x4096 | — | — | — | 4.801 | f64 | 1.15e-14 | small cells 2: CPU rows |
| cpu | row:layer_norm_bwd | float64 | — | — | — | — | 6.876 | f64 | 3.11e-15 | from the item 4 run |
| cpu | row:weight_norm | float64 | — | — | — | — | 4.908 | f64 | 3.75e-16 | from the item 4 run |
| cpu | row:rope | float64 | 4096x4096 | — | — | — | 4.739 | f64 | 2.51e-12 | small cells 2: CPU rows |
| cpu | row:scale_mask_softmax | float64 | 4096x4096 | — | — | — | 5.054 | f64 | 5.55e-17 | small cells 2: CPU rows |
| cpu | row:add_rms_norm | float64 | — | — | — | — | 10.696 | f64 | 2.58e-14 | from the item 4 run |
| cpu | row:add_layer_norm | float64 | — | — | — | — | 10.884 | f64 | 1.51e-14 | from the item 4 run |
| cpu | row:cross_entropy_bwd | float64 | — | — | — | — | 5.648 | f64 | 1.11e-16 | from the item 4 run |
| cpu | norm_params_bwd:rms | float64 | — | — | — | — | 6.803 | f64 | 9.24e-13 | from the item 4 run |
| cpu | norm_params_bwd:layer | float64 | — | — | — | — | 6.775 | f64 | 3.41e-13 | from the item 4 run |
| cpu | layout:strided | float64 | — | — | — | — | 21.177 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | float64 | — | — | — | — | 2.549 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | float64 | — | — | — | — | 3.663 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | float64 | — | — | — | — | 4.778 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | float64 | — | — | — | — | 4.994 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | float64 | — | — | — | — | 1.577 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | float64 | — | — | — | — | 2.081 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | float64 | — | — | — | — | 3.448 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_bilinear | float64 | — | — | — | — | 7.903 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:shift | float64 | — | — | — | — | 3.956 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | float64 | — | — | — | — | 3.777 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | float64 | — | — | — | — | 1.644 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool | float64 | — | — | — | — | 0.625 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | float64 | — | — | — | — | 6.806 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool_bwd | float64 | — | — | — | — | 2.519 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | float64 | — | — | — | — | 9.078 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_avgpool | float64 | — | — | — | — | 5.754 | f64 | 6.94e-18 | from the item 4 run |
| cpu | layout:depthwise_conv | float64 | — | — | — | — | 4.762 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | float64 | — | — | — | — | 720.755 | f64 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | float64 | — | — | — | — | 9.096 | f64 | 4.34e-19 | from the item 4 run |
| cpu | matmul_nt | float64 | 1024x1024x1024 Bᵀ | — | — | — | 3.664 | f64 | 1.42e-13 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | float64 | 1024x1024x1024 Aᵀ | — | — | — | 3.945 | f64 | 1.28e-13 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | float64 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 3.943 | f64 | 1.21e-13 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | float64 | 16 x 256x256x256 | — | — | — | 1.494 | f64 | 3.2e-14 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | float64 | 1024x1024x1024 (dA, dB) | — | — | — | 7.649 | f64 | 1.28e-13 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | float64 | — | — | — | — | 99.003 | f64 | 3.75e-16 | from the item 4 run |
| cpu | attention_causal | float64 | — | — | — | — | 88.724 | f64 | 4.02e-16 | from the item 4 run |
| cpu | attention_fused | float64 | — | — | — | — | 200.052 | f64 | 4.51e-16 | from the item 4 run |
| cpu | attention_fused_causal | float64 | — | — | — | — | 99.921 | f64 | 4.3e-16 | from the item 4 run |
| cpu | attention_decode | float64 | — | — | — | — | 3.449 | f64 | 2.22e-16 | from the item 4 run |
| cpu | attention_bwd | float64 | — | — | — | — | 12.607 | f64 | 1.39e-16 | 6j part 2: CPU attention backward |
| cpu | optimizer:sgd_momentum | float64 | — | — | — | — | 19.925 | f64; + restore of p, m, v | 1.11e-16 | from the item 4 run |
| cpu | optimizer:adam | float64 | — | — | — | — | 30.329 | f64; + restore of p, m, v | 1.11e-16 | from the item 4 run |
| cpu | optimizer:adamw | float64 | — | — | — | — | 30.213 | f64; + restore of p, m, v | 1.11e-16 | from the item 4 run |
| cpu | conv | float64 | — | — | — | — | 6.921 | f64 | 7.11e-14 | from the item 4 run |
| cpu | conv_bwd | float64 | — | — | — | — | 30.051 | f64 | 2.13e-14 | from the item 4 run |
| cpu | lstm_step | float64 | — | — | — | — | 31.472 | f64 | 7.22e-16 | from the item 4 run |
| cpu | gru_step | float64 | — | — | — | — | 21.152 | f64 | 9.99e-16 | from the item 4 run |
| cpu | lstm_step_bwd | float64 | — | — | — | — | 88.303 | f64 | 1.67e-15 | from the item 4 run |
| cpu | gru_step_bwd | float64 | — | — | — | — | 72.014 | f64 | 7.77e-16 | from the item 4 run |
| cpu | lstm_seq | float64 | — | — | — | — | 63.800 | f64 | 3.61e-16 | from the item 4 run |
| cpu | lstm_seq_bwd | float64 | — | — | — | — | 258.586 | f64 | 1.33e-15 | from the item 4 run |
| cpu | gru_seq | float64 | — | — | — | — | 44.408 | f64 | 6.66e-16 | from the item 4 run |
| cpu | gru_seq_bwd | float64 | — | — | — | — | 231.356 | f64 | 1.33e-15 | from the item 4 run |
| cpu | topk_rows | float64 | — | — | — | — | 82.086 | f64 | 0e+0 | from the item 4 run |
| cpu | pick_rows | float64 | — | — | — | — | 0.029 | f64 | 0e+0 | from the item 4 run |
| cpu | sample_rows | float64 | — | — | — | — | 13.632 | f64 | 0e+0 | from the item 4 run |
| cpu | route | s32 | — | — | — | — | 1.171 | s32 | 0e+0 | from the item 4 run |
| cpu | route_wide | s32 | — | — | — | — | 2.639 | s32 | 0e+0 | from the item 4 run |
| cpu | cross_entropy | float64 | — | — | — | — | 45.986 | f64 (log_softmax, pick, neg) | 1.78e-14 | from the item 4 run |
| cpu | ctc | float64 | — | — | — | — | 4.675 | f64 | 4.55e-13 | from the item 4 run |
| cpu | ctc_bwd | float64 | — | — | — | — | 8.043 | f64 | 6.81e-13 | from the item 4 run |
| cpu | softmax | float64 | 4096x4096 | 6.203 | Vector<Double> | 4.34e-19 | 5.593 | f64 | 1.08e-17 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | float64 | 4096x4096 | 4.141 | Vector<Double> | 1.33e-14 | 4.703 | f64 | 1.27e-14 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | float32 | 16777216 | 3.039 | Vector<Single> | 5.96e-8 | 3.033 | f32 | 5.96e-8 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | float32 | 1024x1024x1024 | 3.977 | Vector<Single> | 8.34e-5 | 1.598 | f32 | 7.96e-5 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | float32 | — | — | — | — | 1.664 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | float32 | — | — | — | — | 0.771 | f32 | 0e+0 | from the item 4 run |
| cpu | map:neg | float32 | — | — | — | — | 1.679 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | float32 | — | — | — | — | 1.644 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | float32 | — | — | — | — | 1.653 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square | float32 | — | — | — | — | 1.647 | f32 | 4.77e-7 | from the item 4 run |
| cpu | map:relu | float32 | — | — | — | — | 1.884 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | float32 | — | — | — | — | 1.708 | f32 | 2.38e-7 | from the item 4 run |
| cpu | map:scale | float32 | — | — | — | — | 1.671 | f32 | 4.77e-7 | from the item 4 run |
| cpu | map:affine | float32 | — | — | — | — | 1.648 | f32 | 4.77e-7 | from the item 4 run |
| cpu | map:clamp | float32 | — | — | — | — | 1.634 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add | float32 | — | — | — | — | 3.005 | f32 | 2.38e-7 | from the item 4 run |
| cpu | map:sub | float32 | — | — | — | — | 3.000 | f32 | 2.38e-7 | from the item 4 run |
| cpu | map:mul | float32 | — | — | — | — | 2.933 | f32 | 4.77e-7 | from the item 4 run |
| cpu | map:div | float32 | — | — | — | — | 3.006 | f32 | 1.19e-7 | from the item 4 run |
| cpu | map:max | float32 | — | — | — | — | 2.896 | f32 | 0e+0 | from the item 4 run |
| cpu | map:min | float32 | — | — | — | — | 2.873 | f32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | float32 | — | — | — | — | 2.884 | f32 | 4.77e-7 | from the item 4 run |
| cpu | map:muladd | float32 | — | — | — | — | 4.004 | f32 | 9.54e-7 | from the item 4 run |
| cpu | map:where | float32 | — | — | — | — | 4.006 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | float32 | — | — | — | — | 2.923 | f32 | 9.54e-7 | from the item 4 run |
| cpu | map:abs_bwd | float32 | — | — | — | — | 2.892 | f32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | float32 | — | — | — | — | 2.950 | f32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | float32 | — | — | — | — | 3.291 | f32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | float32 | — | — | — | — | 2.934 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sqrt | float32 | — | — | — | — | 1.741 | f32 | 5.96e-8 | from the item 4 run |
| cpu | map:inv_sqrt | float32 | — | — | — | — | 1.764 | f32 | 1.37e-7 | from the item 4 run |
| cpu | map:exp | float32 | — | — | — | — | 2.030 | f32 | 3.42e-6 | small cells 2: CPU map / rows |
| cpu | map:expm1 | float32 | — | — | — | — | 1.800 | f32 | 6.84e-8 | from the item 4 run |
| cpu | map:log | float32 | — | — | — | — | 1.667 | f32 | 1.26e-7 | from the item 4 run |
| cpu | map:sin | float32 | — | — | — | — | 1.969 | f32 | 7.99e-8 | from the item 4 run |
| cpu | map:cos | float32 | — | — | — | — | 2.632 | f32 | 7.96e-8 | from the item 4 run |
| cpu | map:tanh | float32 | — | — | — | — | 2.828 | f32 | 1.49e-7 | from the item 4 run |
| cpu | map:sigmoid | float32 | — | — | — | — | 1.698 | f32 | 8.77e-8 | small cells 2: CPU map / rows |
| cpu | map:erf | float32 | — | — | — | — | 22.573 | f32 | 1.45e-7 | from the item 4 run |
| cpu | map:gelu | float32 | — | — | — | — | 1.752 | f32 | 4.28e-7 | from the item 4 run |
| cpu | map:gelu_erf | float32 | — | — | — | — | 25.798 | f32 | 2.37e-7 | from the item 4 run |
| cpu | map:silu | float32 | — | — | — | — | 1.727 | f32 | 4.58e-7 | from the item 4 run |
| cpu | map:leaky_relu | float32 | — | — | — | — | 1.658 | f32 | 1.79e-8 | from the item 4 run |
| cpu | map:pow | float32 | — | — | — | — | 5.379 | f32 | 9.47e-6 | from the item 4 run |
| cpu | map:dropout | float32 | — | — | — | — | 3.022 | f32 | 2.38e-7 | from the item 4 run |
| cpu | map:gated_silu | float32 | — | — | — | — | 3.182 | f32 | 2.24e-6 | from the item 4 run |
| cpu | map:gated_gelu | float32 | — | — | — | — | 2.868 | f32 | 2.04e-6 | from the item 4 run |
| cpu | map:sqrt_bwd | float32 | — | — | — | — | 2.917 | f32 | 2.29e-7 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | float32 | — | — | — | — | 3.023 | f32 | 1.04e-6 | from the item 4 run |
| cpu | map:exp_bwd | float32 | — | — | — | — | 2.952 | f32 | 1.95e-5 | from the item 4 run |
| cpu | map:log_bwd | float32 | — | — | — | — | 3.069 | f32 | 2.38e-7 | from the item 4 run |
| cpu | map:sin_bwd | float32 | — | — | — | — | 3.195 | f32 | 4.04e-7 | from the item 4 run |
| cpu | map:cos_bwd | float32 | — | — | — | — | 3.123 | f32 | 4e-7 | from the item 4 run |
| cpu | map:tanh_bwd | float32 | — | — | — | — | 4.195 | f32 | 9.3e-7 | from the item 4 run |
| cpu | map:sigmoid_bwd | float32 | — | — | — | — | 3.278 | f32 | 2.31e-7 | from the item 4 run |
| cpu | map:silu_bwd | float32 | — | — | — | — | 3.328 | f32 | 9.79e-7 | from the item 4 run |
| cpu | map:gelu_bwd | float32 | — | — | — | — | 4.593 | f32 | 8.92e-7 | from the item 4 run |
| cpu | map:gelu_erf_bwd | float32 | — | — | — | — | 28.510 | f32 | 8.35e-7 | from the item 4 run |
| cpu | map:leaky_relu_bwd | float32 | — | — | — | — | 2.886 | f32 | 1.79e-8 | from the item 4 run |
| cpu | map:pow_bwd | float32 | — | — | — | — | 3.677 | f32 | 2.48e-5 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | float32 | — | — | — | — | 3.928 | f32 | 4.11e-6 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | float32 | — | — | — | — | 3.910 | f32 | 2.12e-6 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | float32 | — | — | — | — | 3.804 | f32 | 2.92e-6 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | float32 | — | — | — | — | 3.904 | f32 | 2.07e-6 | from the item 4 run |
| cpu | sum:whole | float32 | — | — | — | — | 0.333 | f32 | 3.71e-3 | 6j part 4: CPU reductions |
| cpu | sum:rows | float32 | — | — | — | — | 0.352 | f32 | 3.58e-5 | 6j part 4: CPU reductions |
| cpu | sum:cols | float32 | — | — | — | — | 0.524 | f32 | 3.77e-5 | 6j part 4: CPU reductions |
| cpu | sumsq:rows | float32 | — | — | — | — | 0.342 | f32 | 2.46e-4 | 6j part 4: CPU reductions |
| cpu | max:whole | float32 | — | — | — | — | 0.386 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | max:rows | float32 | — | — | — | — | 0.385 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | max:cols | float32 | — | — | — | — | 0.538 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | min:rows | float32 | — | — | — | — | 0.385 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | argmax:rows | float32 | — | — | — | — | 0.686 | s32 index | 0e+0 | 6j part 4: CPU reductions |
| cpu | argmax:cols | float32 | — | — | — | — | 0.714 | s32 index | 0e+0 | 6j part 4: CPU reductions |
| cpu | maprow:add | float32 | — | — | — | — | 1.720 | f32 | 5.96e-8 | 6j part 4: CPU reductions |
| cpu | maprow:mul | float32 | — | — | — | — | 1.702 | f32 | 2.98e-8 | 6j part 4: CPU reductions |
| cpu | transpose | float32 | — | — | — | — | 3.158 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | gemv | float32 | — | — | — | — | 2.935 | f32 | 5.19e-5 | 6j part 4: CPU reductions |
| cpu | gemv_t | float32 | — | — | — | — | 3.188 | f32 | 4.74e-5 | 6j part 4: CPU reductions |
| cpu | row:log_softmax | float32 | — | — | — | — | 2.562 | f32 | 1.36e-6 | small cells 2: CPU map / rows |
| cpu | row:softmax_bwd | float32 | — | — | — | — | 2.881 | f32 | 3.02e-5 | from the item 4 run |
| cpu | row:log_softmax_bwd | float32 | — | — | — | — | 2.975 | f32 | 4.67e-5 | from the item 4 run |
| cpu | row:rms_norm_affine | float32 | 4096x4096 | — | — | — | 1.990 | f32 | 8.11e-7 | small cells 2: CPU rows / ctc |
| cpu | row:rms_norm_bwd | float32 | — | — | — | — | 3.026 | f32 | 2.07e-7 | from the item 4 run |
| cpu | row:layer_norm | float32 | 4096x4096 | — | — | — | 1.708 | f32 | 8.74e-7 | small cells 2: CPU rows |
| cpu | row:layer_norm_bwd | float32 | — | — | — | — | 2.884 | f32 | 2.11e-7 | from the item 4 run |
| cpu | row:weight_norm | float32 | — | — | — | — | 1.696 | f32 | 9.55e-9 | from the item 4 run |
| cpu | row:rope | float32 | 4096x4096 | — | — | — | 1.822 | f32 | 1.35e-3 | small cells 2: CPU rows |
| cpu | row:scale_mask_softmax | float32 | 4096x4096 | — | — | — | 2.570 | f32 | 4.81e-8 | small cells 2: CPU rows |
| cpu | row:add_rms_norm | float32 | — | — | — | — | 4.909 | f32 | 1.22e-6 | from the item 4 run |
| cpu | row:add_layer_norm | float32 | — | — | — | — | 4.901 | f32 | 1.28e-6 | from the item 4 run |
| cpu | row:cross_entropy_bwd | float32 | — | — | — | — | 1.811 | f32 | 2.98e-8 | from the item 4 run |
| cpu | norm_params_bwd:rms | float32 | — | — | — | — | 3.329 | f32 | 3.09e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | float32 | — | — | — | — | 3.373 | f32 | 3.09e-4 | from the item 4 run |
| cpu | layout:strided | float32 | 1x1024x64x256 | — | — | — | 2.388 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:one_hot | float32 | 16384x1024 | — | — | — | 0.581 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:gather_rows | float32 | 4096 of 4096x4096 | — | — | — | 1.458 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:gather_cols | float32 | 4096 of 4096x4096 | — | — | — | 2.924 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:scatter_add_rows | float32 | 4096 into 4096x4096 | — | — | — | 1.897 | f32 | 4.77e-7 | small cells 2: CPU layout |
| cpu | layout:im2col | float32 | 64x128x128 k3x3 | — | — | — | 0.904 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:col2im | float32 | 64x128x128 k3x3 | — | — | — | 2.036 | f32 | 1.08e-6 | small cells 2: CPU layout |
| cpu | layout:interpolate_nearest | float32 | 16x512x512 to 1024x1024 | — | — | — | 2.155 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:interpolate_bilinear | float32 | 16x512x512 to 1024x1024 | — | — | — | 6.039 | f32 | 1.19e-7 | small cells 2: CPU layout |
| cpu | layout:shift | float32 | 4096x4096 by 3 | — | — | — | 1.708 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:pad2d | float32 | 4096x4093 to 4096 | — | — | — | 1.596 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:maxpool | float32 | 64x256x256 k2x2 s2 p0 | — | — | — | 1.226 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:avgpool | float32 | 64x256x256 k2x2 s2 p0 | — | — | — | 0.536 | f32 | 7.45e-8 | small cells 2: CPU layout |
| cpu | layout:maxpool_bwd | float32 | 64x256x256 k2x2 s2 p0 | — | — | — | 4.336 | f32 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:avgpool_bwd | float32 | 64x256x256 k2x2 s2 p0 | — | — | — | 3.216 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:adaptive_maxpool | float32 | 64x256x256 to 7x7 | — | — | — | 0.752 | f32 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:adaptive_avgpool | float32 | 64x256x256 to 7x7 | — | — | — | 0.287 | f32 | 7.4e-8 | small cells 2: CPU layout |
| cpu | layout:depthwise_conv | float32 | 64x256x256 k3x3 s1 p1 | — | — | — | 1.074 | f32 | 6.56e-7 | small cells 2: CPU pools |
| cpu | layout:adaptive_maxpool_bwd | float32 | 64x256x256 to 7x7 | — | — | — | 0.307 | f32 | 5.96e-8 | small cells 2: CPU layout |
| cpu | layout:adaptive_avgpool_bwd | float32 | 64x256x256 to 7x7 | — | — | — | 0.279 | f32 | 2.08e-10 | small cells 2: CPU pools |
| cpu | matmul_nt | float32 | 1024x1024x1024 Bᵀ | — | — | — | 1.708 | f32 | 8.59e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | float32 | 1024x1024x1024 Aᵀ | — | — | — | 1.532 | f32 | 7.82e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | float32 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 1.893 | f32 | 7.81e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | float32 | 16 x 256x256x256 | — | — | — | 0.748 | f32 | 1.92e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | float32 | 1024x1024x1024 (dA, dB) | — | — | — | 3.763 | f32 | 8.05e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | float32 | — | — | — | — | 51.273 | f32 | 1.63e-7 | from the item 4 run |
| cpu | attention_causal | float32 | — | — | — | — | 40.174 | f32 | 1.46e-7 | from the item 4 run |
| cpu | attention_fused | float32 | — | — | — | — | 78.230 | f32 | 1.39e-7 | from the item 4 run |
| cpu | attention_fused_causal | float32 | — | — | — | — | 40.100 | f32 | 1.46e-7 | from the item 4 run |
| cpu | attention_decode | float32 | — | — | — | — | 1.729 | f32 | 6.32e-8 | from the item 4 run |
| cpu | attention_bwd | float32 | — | — | — | — | 6.769 | f32 | 1.72e-7 | 6j part 2: CPU attention backward |
| cpu | optimizer:sgd_momentum | float32 | — | — | — | — | 10.869 | f32; + restore of p, m, v | 5.98e-8 | from the item 4 run |
| cpu | optimizer:adam | float32 | — | — | — | — | 14.829 | f32; + restore of p, m, v | 5.95e-8 | from the item 4 run |
| cpu | optimizer:adamw | float32 | — | — | — | — | 14.106 | f32; + restore of p, m, v | 7.29e-8 | from the item 4 run |
| cpu | conv | float32 | — | — | — | — | 2.310 | f32 | 4.36e-5 | from the item 4 run |
| cpu | conv_bwd | float32 | — | — | — | — | 16.128 | f32 | 1.56e-5 | from the item 4 run |
| cpu | lstm_step | float32 | — | — | — | — | 13.904 | f32 | 5.76e-7 | 6k CPU recurrent steps (cooled) |
| cpu | gru_step | float32 | — | — | — | — | 13.457 | f32 | 7.78e-7 | 6k CPU recurrent steps (cooled) |
| cpu | lstm_step_bwd | float32 | B1024 D1024 H1024 | — | — | — | 41.883 | f32 | 1.81e-6 | 6k CPU recurrent (cooled) |
| cpu | gru_step_bwd | float32 | B1024 D1024 H1024 | — | — | — | 34.564 | f32 | 8.33e-7 | 6k CPU recurrent (cooled) |
| cpu | lstm_seq | float32 | T16 B256 D512 H512 | — | — | — | 20.211 | f32 | 2.85e-7 | 6k CPU recurrent (cooled) |
| cpu | lstm_seq_bwd | float32 | T16 B256 D512 H512 | — | — | — | 57.738 | f32 | 9.4e-7 | 6k CPU recurrent (cooled) |
| cpu | gru_seq | float32 | T16 B256 D512 H512 | — | — | — | 14.490 | f32 | 5.31e-7 | 6k CPU recurrent (cooled) |
| cpu | gru_seq_bwd | float32 | T16 B256 D512 H512 | — | — | — | 46.181 | f32 | 6.81e-7 | 6k CPU recurrent (cooled) |
| cpu | topk_rows | float32 | — | — | — | — | 87.360 | f32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | float32 | — | — | — | — | 0.020 | f32 | 0e+0 | from the item 4 run |
| cpu | sample_rows | float32 | — | — | — | — | 7.096 | f32 | 2e+0 | from the item 4 run |
| cpu | cross_entropy | float32 | — | — | — | — | 21.872 | f32 (log_softmax, pick, neg) | 1.96e-6 | from the item 4 run |
| cpu | ctc | float32 | — | — | — | — | 2.362 | f32 | 1.36e-3 | small cells 2: CPU ctc (committed code, cooled) |
| cpu | ctc_bwd | float32 | — | — | — | — | 3.601 | f32 | 1.67e-3 | small cells 2: CPU ctc (committed code, cooled) |
| cpu | softmax | float32 | 4096x4096 | 4.239 | Vector<Single> | 5.91e-9 | 1.860 | f32 | 6.21e-10 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | float32 | 4096x4096 | 1.807 | Vector<Single> | 5.16e-7 | 1.675 | f32 | 3.68e-7 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | float16 | 16777216 | 7.566 | scalar | 4.88e-4 | 1.166 | f32 | 4.88e-4 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | float16 | 1024x1024x1024 | 867.638 | scalar | 6.08e-1 | 1.247 | f32 | 1.56e-2 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | float16 | — | — | — | — | 1.249 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | float16 | — | — | — | — | 0.377 | f32 | 0e+0 | from the item 4 run |
| cpu | map:neg | float16 | — | — | — | — | 1.262 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | float16 | — | — | — | — | 1.278 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | float16 | — | — | — | — | 1.428 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square | float16 | — | — | — | — | 1.270 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:relu | float16 | — | — | — | — | 1.302 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | float16 | — | — | — | — | 1.293 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:scale | float16 | — | — | — | — | 1.271 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:affine | float16 | — | — | — | — | 1.271 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:clamp | float16 | — | — | — | — | 1.572 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add | float16 | — | — | — | — | 1.788 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:sub | float16 | — | — | — | — | 1.196 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:mul | float16 | — | — | — | — | 1.194 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:div | float16 | — | — | — | — | 1.094 | f32 | 9.76e-4 | from the item 4 run |
| cpu | map:max | float16 | — | — | — | — | 1.259 | f32 | 0e+0 | from the item 4 run |
| cpu | map:min | float16 | — | — | — | — | 1.257 | f32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | float16 | — | — | — | — | 1.204 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:muladd | float16 | — | — | — | — | 2.383 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:where | float16 | — | — | — | — | 1.658 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | float16 | — | — | — | — | 1.196 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:abs_bwd | float16 | — | — | — | — | 1.246 | f32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | float16 | — | — | — | — | 1.220 | f32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | float16 | — | — | — | — | 1.216 | f32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | float16 | — | — | — | — | 1.210 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sqrt | float16 | — | — | — | — | 0.762 | f32 | 4.88e-4 | from the item 4 run |
| cpu | map:inv_sqrt | float16 | — | — | — | — | 0.910 | f32 | 4.88e-4 | from the item 4 run |
| cpu | map:exp | float16 | — | — | — | — | 1.521 | f32 | 1.55e-2 | from the item 4 run |
| cpu | map:expm1 | float16 | — | — | — | — | 3.364 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:log | float16 | — | — | — | — | 1.613 | f32 | 4.88e-4 | from the item 4 run |
| cpu | map:sin | float16 | — | — | — | — | 4.113 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:cos | float16 | — | — | — | — | 3.798 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:tanh | float16 | — | — | — | — | 3.978 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:sigmoid | float16 | — | — | — | — | 3.077 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:erf | float16 | — | — | — | — | 23.263 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:gelu | float16 | — | — | — | — | 3.436 | f32 | 9.73e-4 | from the item 4 run |
| cpu | map:gelu_erf | float16 | — | — | — | — | 37.528 | f32 | 9.76e-4 | from the item 4 run |
| cpu | map:silu | float16 | — | — | — | — | 3.127 | f32 | 9.75e-4 | from the item 4 run |
| cpu | map:leaky_relu | float16 | — | — | — | — | 1.397 | f32 | 9.77e-5 | from the item 4 run |
| cpu | map:pow | float16 | — | — | — | — | 6.531 | f32 | 7.8e-3 | from the item 4 run |
| cpu | map:dropout | float16 | — | — | — | — | 1.807 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:gated_silu | float16 | — | — | — | — | 2.111 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:gated_gelu | float16 | — | — | — | — | 4.048 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:sqrt_bwd | float16 | — | — | — | — | 2.324 | f32 | 9.77e-4 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | float16 | — | — | — | — | 1.499 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:exp_bwd | float16 | — | — | — | — | 1.912 | f32 | 6.25e-2 | from the item 4 run |
| cpu | map:log_bwd | float16 | — | — | — | — | 1.094 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:sin_bwd | float16 | — | — | — | — | 2.923 | f32 | 9.77e-4 | from the item 4 run |
| cpu | map:cos_bwd | float16 | — | — | — | — | 4.469 | f32 | 9.77e-4 | from the item 4 run |
| cpu | map:tanh_bwd | float16 | — | — | — | — | 5.763 | f32 | 9.77e-4 | from the item 4 run |
| cpu | map:sigmoid_bwd | float16 | — | — | — | — | 5.830 | f32 | 2.44e-4 | from the item 4 run |
| cpu | map:silu_bwd | float16 | — | — | — | — | 5.820 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:gelu_bwd | float16 | — | — | — | — | 4.147 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:gelu_erf_bwd | float16 | — | — | — | — | 44.438 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:leaky_relu_bwd | float16 | — | — | — | — | 1.227 | f32 | 9.77e-5 | from the item 4 run |
| cpu | map:pow_bwd | float16 | — | — | — | — | 4.769 | f32 | 3.13e-2 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | float16 | — | — | — | — | 3.613 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | float16 | — | — | — | — | 2.855 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | float16 | — | — | — | — | 4.466 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | float16 | — | — | — | — | 3.027 | f32 | 3.91e-3 | from the item 4 run |
| cpu | sum:whole | float16 | — | — | — | — | 0.409 | f32 | 1.53e-4 | 6k CPU sums A/B (HEAD) |
| cpu | sum:rows | float16 | — | — | — | — | 0.399 | f32 | 1.05e-5 | 6k CPU sums A/B (HEAD) |
| cpu | sum:cols | float16 | 4096x4096 | — | — | — | 0.817 | f32 | 3.39e-5 | 6k CPU sums (cooled) |
| cpu | sumsq:rows | float16 | — | — | — | — | 0.400 | f32 | 2.62e-4 | 6k CPU sums A/B (HEAD) |
| cpu | max:whole | float16 | — | — | — | — | 0.479 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | max:rows | float16 | 4096x4096 | — | — | — | 0.497 | f32 | 0e+0 | 6k CPU sums (cooled) |
| cpu | max:cols | float16 | — | — | — | — | 0.809 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | min:rows | float16 | — | — | — | — | 0.704 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | argmax:rows | float16 | — | — | — | — | 1.160 | s32 index | 0e+0 | 6j part 4: CPU reductions |
| cpu | argmax:cols | float16 | — | — | — | — | 0.744 | s32 index | 0e+0 | 6j part 4: CPU reductions |
| cpu | maprow:add | float16 | — | — | — | — | 1.538 | f32 | 4.88e-4 | 6j part 4: CPU reductions |
| cpu | maprow:mul | float16 | — | — | — | — | 0.875 | f32 | 2.44e-4 | 6j part 4: CPU reductions |
| cpu | transpose | float16 | — | — | — | — | 1.504 | f32 | 0e+0 | 6j part 4: CPU reductions |
| cpu | gemv | float16 | — | — | — | — | 1.860 | f32 | 3.12e-2 | 6j part 4: CPU reductions |
| cpu | gemv_t | float16 | — | — | — | — | 2.430 | f32 | 5.38e-2 | 6j part 4: CPU reductions |
| cpu | row:log_softmax | float16 | 4096x4096 | — | — | — | 2.091 | f32 | 3.91e-3 | small cells 2: CPU rows |
| cpu | row:softmax_bwd | float16 | — | — | — | — | 1.936 | f32 | 3.12e-2 | from the item 4 run |
| cpu | row:log_softmax_bwd | float16 | — | — | — | — | 1.830 | f32 | 3.13e-2 | from the item 4 run |
| cpu | row:rms_norm_affine | float16 | 4096x4096 | — | — | — | 1.526 | f32 | 9.77e-4 | small cells 2: CPU rows / ctc |
| cpu | row:rms_norm_bwd | float16 | — | — | — | — | 1.350 | f32 | 2.44e-4 | from the item 4 run |
| cpu | row:layer_norm | float16 | 4096x4096 | — | — | — | 0.949 | f32 | 1.95e-3 | small cells 2: CPU rows |
| cpu | row:layer_norm_bwd | float16 | — | — | — | — | 1.436 | f32 | 2.44e-4 | from the item 4 run |
| cpu | row:weight_norm | float16 | — | — | — | — | 0.851 | f32 | 1.53e-5 | from the item 4 run |
| cpu | row:rope | float16 | 4096x4096 | — | — | — | 1.299 | f32 | 2.53e-3 | small cells 2: CPU rows |
| cpu | row:scale_mask_softmax | float16 | 4096x4096 | — | — | — | 2.982 | f32 | 2.39e-4 | small cells 2: CPU rows |
| cpu | row:add_rms_norm | float16 | — | — | — | — | 1.961 | f32 | 1.95e-3 | from the item 4 run |
| cpu | row:add_layer_norm | float16 | — | — | — | — | 2.103 | f32 | 1.95e-3 | from the item 4 run |
| cpu | row:cross_entropy_bwd | float16 | — | — | — | — | 3.108 | f32 | 2.44e-4 | from the item 4 run |
| cpu | norm_params_bwd:rms | float16 | — | — | — | — | 4.734 | f32 | 2.92e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | float16 | — | — | — | — | 4.796 | f32 | 3.75e-4 | from the item 4 run |
| cpu | layout:strided | float16 | 1x1024x64x256 | — | — | — | 3.774 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:one_hot | float16 | 16384x1024 | — | — | — | 0.376 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:gather_rows | float16 | 4096 of 4096x4096 | — | — | — | 0.593 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:gather_cols | float16 | 4096 of 4096x4096 | — | — | — | 3.152 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:scatter_add_rows | float16 | 4096 into 4096x4096 | — | — | — | 2.982 | f32 | 1.95e-3 | small cells 2: CPU layout |
| cpu | layout:im2col | float16 | 64x128x128 k3x3 | — | — | — | 1.364 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:col2im | float16 | 64x128x128 k3x3 | — | — | — | 2.367 | f32 | 1.95e-3 | small cells 2: CPU layout |
| cpu | layout:interpolate_nearest | float16 | 16x512x512 to 1024x1024 | — | — | — | 3.579 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:interpolate_bilinear | float16 | 16x512x512 to 1024x1024 | — | — | — | 10.273 | f32 | 2.44e-4 | small cells 2: CPU layout |
| cpu | layout:shift | float16 | 4096x4096 by 3 | — | — | — | 2.714 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:pad2d | float16 | 4096x4093 to 4096 | — | — | — | 0.785 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:maxpool | float16 | 64x256x256 k2x2 s2 p0 | — | — | — | 1.471 | f32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:avgpool | float16 | 64x256x256 k2x2 s2 p0 | — | — | — | 0.732 | f32 | 2.44e-4 | small cells 2: CPU layout |
| cpu | layout:maxpool_bwd | float16 | 64x256x256 k2x2 s2 p0 | — | — | — | 4.004 | f32 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:avgpool_bwd | float16 | 64x256x256 k2x2 s2 p0 | — | — | — | 3.680 | f32 | 2.98e-8 | small cells 2: CPU layout |
| cpu | layout:adaptive_maxpool | float16 | 64x256x256 to 7x7 | — | — | — | 0.624 | f32 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:adaptive_avgpool | float16 | 64x256x256 to 7x7 | — | — | — | 0.424 | f32 | 1.53e-5 | small cells 2: CPU layout |
| cpu | layout:depthwise_conv | float16 | 64x256x256 k3x3 s1 p1 | — | — | — | 1.382 | f32 | 1.95e-3 | small cells 2: CPU pools |
| cpu | layout:adaptive_maxpool_bwd | float16 | 64x256x256 to 7x7 | — | — | — | 0.546 | f32 | 4.88e-4 | small cells 2: CPU layout |
| cpu | layout:adaptive_avgpool_bwd | float16 | 64x256x256 to 7x7 | — | — | — | 0.417 | f32 | 9.44e-7 | small cells 2: CPU pools |
| cpu | matmul_nt | float16 | 1024x1024x1024 Bᵀ | — | — | — | 1.857 | f32 | 1.56e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | float16 | 1024x1024x1024 Aᵀ | — | — | — | 2.026 | f32 | 1.56e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | float16 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 2.124 | f32 | 1.56e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | float16 | 16 x 256x256x256 | — | — | — | 0.736 | f32 | 7.81e-3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | float16 | 1024x1024x1024 (dA, dB) | — | — | — | 4.111 | f32 | 1.56e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | float16 | — | — | — | — | 64.564 | f32 | 2.27e-5 | from the item 4 run |
| cpu | attention_causal | float16 | — | — | — | — | 44.280 | f32 | 2.44e-4 | from the item 4 run |
| cpu | attention_fused | float16 | — | — | — | — | 81.181 | f32 | 2.27e-5 | from the item 4 run |
| cpu | attention_fused_causal | float16 | — | — | — | — | 43.941 | f32 | 2.44e-4 | from the item 4 run |
| cpu | attention_decode | float16 | — | — | — | — | 1.549 | f32 | 1.34e-5 | from the item 4 run |
| cpu | attention_bwd | float16 | — | — | — | — | 9.357 | f32 | 1.2e-4 | 6j part 2: CPU attention backward |
| cpu | optimizer:sgd_momentum | float16 | — | — | — | — | 6.913 | f32; + restore of p, m, v | 4.88e-4 | from the item 4 run |
| cpu | optimizer:adam | float16 | — | — | — | — | 11.580 | f32; + restore of p, m, v | 4.88e-4 | from the item 4 run |
| cpu | optimizer:adamw | float16 | — | — | — | — | 11.735 | f32; + restore of p, m, v | 4.87e-4 | from the item 4 run |
| cpu | conv | float16 | — | — | — | — | 2.565 | f32 | 1.55e-2 | from the item 4 run |
| cpu | conv_bwd | float16 | — | — | — | — | 17.602 | f32 | 1.56e-2 | from the item 4 run |
| cpu | lstm_step | float16 | — | — | — | — | 20.452 | f32 | 2.04e-4 | 6k CPU recurrent steps (cooled) |
| cpu | gru_step | float16 | — | — | — | — | 14.637 | f32 | 2.44e-4 | 6k CPU recurrent steps (cooled) |
| cpu | lstm_step_bwd | float16 | B1024 D1024 H1024 | — | — | — | 49.868 | f32 | 2.43e-4 | 6k CPU recurrent (cooled) |
| cpu | gru_step_bwd | float16 | B1024 D1024 H1024 | — | — | — | 33.121 | f32 | 2.41e-4 | 6k CPU recurrent (cooled) |
| cpu | lstm_seq | float16 | T16 B256 D512 H512 | — | — | — | 22.909 | f32 | 1.7e-4 | 6k CPU recurrent (cooled) |
| cpu | lstm_seq_bwd | float16 | T16 B256 D512 H512 | — | — | — | 51.173 | f32 | 2.39e-4 | 6k CPU recurrent (cooled) |
| cpu | gru_seq | float16 | T16 B256 D512 H512 | — | — | — | 17.006 | f32 | 2.44e-4 | 6k CPU recurrent (cooled) |
| cpu | gru_seq_bwd | float16 | T16 B256 D512 H512 | — | — | — | 51.806 | f32 | 2.44e-4 | 6k CPU recurrent (cooled) |
| cpu | topk_rows | float16 | — | — | — | — | 81.487 | f32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | float16 | — | — | — | — | 0.087 | f32 | 0e+0 | from the item 4 run |
| cpu | sample_rows | float16 | — | — | — | — | 6.307 | f32 | 2e+0 | from the item 4 run |
| cpu | cross_entropy | float16 | — | — | — | — | 18.073 | f32 (log_softmax, pick, neg) | 7.8e-3 | from the item 4 run |
| cpu | ctc | float16 | — | — | — | — | 2.188 | f32 | 4.96e-1 | small cells 2: CPU ctc (committed code, cooled) |
| cpu | ctc_bwd | float16 | — | — | — | — | 4.344 | f32 | 1.6e-3 | small cells 2: CPU ctc (committed code, cooled) |
| cpu | softmax | float16 | 4096x4096 | 54.353 | scalar | 1.57e-4 | 3.318 | f32 | 9.54e-7 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | float16 | 4096x4096 | 18.926 | scalar | 7.06e-2 | 1.231 | f32 | 4.89e-4 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | bfloat16 | 16777216 | — | — | — | 1.172 | f32 | 3.91e-3 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | bfloat16 | 1024x1024x1024 | — | — | — | 1.805 | f32 | 1.25e-1 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | bfloat16 | — | — | — | — | 0.390 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | bfloat16 | — | — | — | — | 0.198 | f32 | 0e+0 | from the item 4 run |
| cpu | map:neg | bfloat16 | — | — | — | — | 0.401 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | bfloat16 | — | — | — | — | 0.403 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | bfloat16 | — | — | — | — | 0.483 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square | bfloat16 | — | — | — | — | 0.402 | f32 | 3.1e-2 | from the item 4 run |
| cpu | map:relu | bfloat16 | — | — | — | — | 0.418 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | bfloat16 | — | — | — | — | 0.403 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:scale | bfloat16 | — | — | — | — | 0.401 | f32 | 3.13e-2 | from the item 4 run |
| cpu | map:affine | bfloat16 | — | — | — | — | 0.407 | f32 | 3.13e-2 | from the item 4 run |
| cpu | map:clamp | bfloat16 | — | — | — | — | 0.588 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add | bfloat16 | — | — | — | — | 0.821 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:sub | bfloat16 | — | — | — | — | 0.817 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:mul | bfloat16 | — | — | — | — | 0.823 | f32 | 3.13e-2 | from the item 4 run |
| cpu | map:div | bfloat16 | — | — | — | — | 0.864 | f32 | 7.78e-3 | from the item 4 run |
| cpu | map:max | bfloat16 | — | — | — | — | 0.836 | f32 | 0e+0 | from the item 4 run |
| cpu | map:min | bfloat16 | — | — | — | — | 0.846 | f32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | bfloat16 | — | — | — | — | 1.187 | f32 | 3.13e-2 | from the item 4 run |
| cpu | map:muladd | bfloat16 | — | — | — | — | 1.379 | f32 | 6.25e-2 | from the item 4 run |
| cpu | map:where | bfloat16 | — | — | — | — | 1.409 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | bfloat16 | — | — | — | — | 0.810 | f32 | 6.25e-2 | from the item 4 run |
| cpu | map:abs_bwd | bfloat16 | — | — | — | — | 0.829 | f32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | bfloat16 | — | — | — | — | 0.807 | f32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | bfloat16 | — | — | — | — | 0.823 | f32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | bfloat16 | — | — | — | — | 0.812 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sqrt | bfloat16 | — | — | — | — | 0.429 | f32 | 3.9e-3 | from the item 4 run |
| cpu | map:inv_sqrt | bfloat16 | — | — | — | — | 0.555 | f32 | 3.9e-3 | from the item 4 run |
| cpu | map:exp | bfloat16 | — | — | — | — | 2.253 | f32 | 1.21e-1 | from the item 4 run |
| cpu | map:expm1 | bfloat16 | — | — | — | — | 1.609 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:log | bfloat16 | — | — | — | — | 1.407 | f32 | 3.84e-3 | from the item 4 run |
| cpu | map:sin | bfloat16 | — | — | — | — | 2.129 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:cos | bfloat16 | — | — | — | — | 3.007 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:tanh | bfloat16 | — | — | — | — | 3.240 | f32 | 1.94e-3 | from the item 4 run |
| cpu | map:sigmoid | bfloat16 | — | — | — | — | 2.640 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:erf | bfloat16 | — | — | — | — | 22.000 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:gelu | bfloat16 | — | — | — | — | 2.720 | f32 | 7.75e-3 | from the item 4 run |
| cpu | map:gelu_erf | bfloat16 | — | — | — | — | 37.271 | f32 | 7.77e-3 | from the item 4 run |
| cpu | map:silu | bfloat16 | — | — | — | — | 2.412 | f32 | 7.79e-3 | from the item 4 run |
| cpu | map:leaky_relu | bfloat16 | — | — | — | — | 0.741 | f32 | 7.81e-4 | from the item 4 run |
| cpu | map:pow | bfloat16 | — | — | — | — | 5.795 | f32 | 6.17e-2 | from the item 4 run |
| cpu | map:dropout | bfloat16 | — | — | — | — | 1.198 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:gated_silu | bfloat16 | — | — | — | — | 2.738 | f32 | 3.12e-2 | from the item 4 run |
| cpu | map:gated_gelu | bfloat16 | — | — | — | — | 3.014 | f32 | 3.12e-2 | from the item 4 run |
| cpu | map:sqrt_bwd | bfloat16 | — | — | — | — | 1.515 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | bfloat16 | — | — | — | — | 1.108 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:exp_bwd | bfloat16 | — | — | — | — | 1.688 | f32 | 5e-1 | from the item 4 run |
| cpu | map:log_bwd | bfloat16 | — | — | — | — | 0.853 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:sin_bwd | bfloat16 | — | — | — | — | 2.226 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:cos_bwd | bfloat16 | — | — | — | — | 3.502 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:tanh_bwd | bfloat16 | — | — | — | — | 4.668 | f32 | 7.81e-3 | from the item 4 run |
| cpu | map:sigmoid_bwd | bfloat16 | — | — | — | — | 4.627 | f32 | 1.95e-3 | from the item 4 run |
| cpu | map:silu_bwd | bfloat16 | — | — | — | — | 4.743 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:gelu_bwd | bfloat16 | — | — | — | — | 5.060 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:gelu_erf_bwd | bfloat16 | — | — | — | — | 43.443 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:leaky_relu_bwd | bfloat16 | — | — | — | — | 1.197 | f32 | 7.81e-4 | from the item 4 run |
| cpu | map:pow_bwd | bfloat16 | — | — | — | — | 4.012 | f32 | 2.5e-1 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | bfloat16 | — | — | — | — | 3.350 | f32 | 6.23e-2 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | bfloat16 | — | — | — | — | 2.073 | f32 | 3.12e-2 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | bfloat16 | — | — | — | — | 3.563 | f32 | 3.12e-2 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | bfloat16 | — | — | — | — | 2.236 | f32 | 3.12e-2 | from the item 4 run |
| cpu | sum:whole | bfloat16 | — | — | — | — | 0.178 | f32 | 1.18e-4 | from the item 4 run |
| cpu | sum:rows | bfloat16 | — | — | — | — | 0.180 | f32 | 8.23e-6 | from the item 4 run |
| cpu | sum:cols | bfloat16 | — | — | — | — | 0.908 | f32 | 1.86e-5 | from the item 4 run |
| cpu | sumsq:rows | bfloat16 | — | — | — | — | 0.178 | f32 | 3.08e-4 | from the item 4 run |
| cpu | max:whole | bfloat16 | — | — | — | — | 0.443 | f32 | 0e+0 | from the item 4 run |
| cpu | max:rows | bfloat16 | — | — | — | — | 0.712 | f32 | 0e+0 | from the item 4 run |
| cpu | max:cols | bfloat16 | — | — | — | — | 3.278 | f32 | 0e+0 | from the item 4 run |
| cpu | min:rows | bfloat16 | — | — | — | — | 0.781 | f32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | bfloat16 | — | — | — | — | 0.823 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | bfloat16 | — | — | — | — | 1.308 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | bfloat16 | — | — | — | — | 0.873 | f32 | 3.91e-3 | from the item 4 run |
| cpu | maprow:mul | bfloat16 | — | — | — | — | 0.851 | f32 | 1.95e-3 | from the item 4 run |
| cpu | transpose | bfloat16 | — | — | — | — | 1.324 | f32 | 0e+0 | from the item 4 run |
| cpu | gemv | bfloat16 | — | — | — | — | 1.478 | f32 | 2.5e-1 | from the item 4 run |
| cpu | gemv_t | bfloat16 | — | — | — | — | 6.166 | f32 | 2.49e-1 | from the item 4 run |
| cpu | row:log_softmax | bfloat16 | — | — | — | — | 1.644 | f32 | 3.13e-2 | from the item 4 run |
| cpu | row:softmax_bwd | bfloat16 | — | — | — | — | 0.906 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:log_softmax_bwd | bfloat16 | — | — | — | — | 1.664 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:rms_norm_affine | bfloat16 | — | — | — | — | 0.470 | f32 | 7.81e-3 | from the item 4 run |
| cpu | row:rms_norm_bwd | bfloat16 | — | — | — | — | 0.956 | f32 | 1.95e-3 | from the item 4 run |
| cpu | row:layer_norm | bfloat16 | — | — | — | — | 0.526 | f32 | 1.56e-2 | from the item 4 run |
| cpu | row:layer_norm_bwd | bfloat16 | — | — | — | — | 1.012 | f32 | 1.95e-3 | from the item 4 run |
| cpu | row:weight_norm | bfloat16 | — | — | — | — | 0.430 | f32 | 1.22e-4 | from the item 4 run |
| cpu | row:rope | bfloat16 | — | — | — | — | 3.054 | f32 | 1.61e-2 | from the item 4 run |
| cpu | row:scale_mask_softmax | bfloat16 | — | — | — | — | 2.443 | f32 | 1.34e-3 | from the item 4 run |
| cpu | row:add_rms_norm | bfloat16 | — | — | — | — | 1.757 | f32 | 1.56e-2 | from the item 4 run |
| cpu | row:add_layer_norm | bfloat16 | — | — | — | — | 1.806 | f32 | 1.56e-2 | from the item 4 run |
| cpu | row:cross_entropy_bwd | bfloat16 | — | — | — | — | 1.630 | f32 | 1.95e-3 | from the item 4 run |
| cpu | norm_params_bwd:rms | bfloat16 | — | — | — | — | 3.450 | f32 | 2.62e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | bfloat16 | — | — | — | — | 2.308 | f32 | 3.76e-4 | from the item 4 run |
| cpu | layout:strided | bfloat16 | — | — | — | — | 8.221 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | bfloat16 | — | — | — | — | 0.197 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | bfloat16 | — | — | — | — | 0.338 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | bfloat16 | — | — | — | — | 3.195 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | bfloat16 | — | — | — | — | 2.789 | f32 | 1.56e-2 | from the item 4 run |
| cpu | layout:im2col | bfloat16 | — | — | — | — | 1.684 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | bfloat16 | — | — | — | — | 3.527 | f32 | 1.56e-2 | from the item 4 run |
| cpu | layout:interpolate_nearest | bfloat16 | — | — | — | — | 5.352 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_bilinear | bfloat16 | — | — | — | — | 18.040 | f32 | 1.95e-3 | from the item 4 run |
| cpu | layout:shift | bfloat16 | — | — | — | — | 3.750 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | bfloat16 | — | — | — | — | 0.371 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | bfloat16 | — | — | — | — | 1.970 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool | bfloat16 | — | — | — | — | 1.296 | f32 | 1.95e-3 | from the item 4 run |
| cpu | layout:maxpool_bwd | bfloat16 | — | — | — | — | 16.960 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool_bwd | bfloat16 | — | — | — | — | 4.135 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | bfloat16 | — | — | — | — | 22.525 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_avgpool | bfloat16 | — | — | — | — | 19.019 | f32 | 1.22e-4 | from the item 4 run |
| cpu | layout:depthwise_conv | bfloat16 | — | — | — | — | 11.805 | f32 | 1.55e-2 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | bfloat16 | — | — | — | — | 1313.007 | f32 | 3.91e-3 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | bfloat16 | — | — | — | — | 10.031 | f32 | 7.52e-6 | from the item 4 run |
| cpu | matmul_nt | bfloat16 | 1024x1024x1024 Bᵀ | — | — | — | 1.684 | f32 | 1.25e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | bfloat16 | 1024x1024x1024 Aᵀ | — | — | — | 1.966 | f32 | 1.25e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | bfloat16 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 1.975 | f32 | 1.25e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | bfloat16 | 16 x 256x256x256 | — | — | — | 0.762 | f32 | 6.25e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | bfloat16 | 1024x1024x1024 (dA, dB) | — | — | — | 3.795 | f32 | 1.25e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | bfloat16 | — | — | — | — | 52.704 | f32 | 1.42e-4 | from the item 4 run |
| cpu | attention_causal | bfloat16 | — | — | — | — | 43.340 | f32 | 1.95e-3 | from the item 4 run |
| cpu | attention_fused | bfloat16 | — | — | — | — | 80.966 | f32 | 1.42e-4 | from the item 4 run |
| cpu | attention_fused_causal | bfloat16 | — | — | — | — | 44.384 | f32 | 1.95e-3 | from the item 4 run |
| cpu | attention_decode | bfloat16 | — | — | — | — | 0.845 | f32 | 1.15e-4 | from the item 4 run |
| cpu | attention_bwd | bfloat16 | — | — | — | — | 20.219 | f32 | 9.66e-4 | from the item 4 run |
| cpu | optimizer:sgd_momentum | bfloat16 | — | — | — | — | 6.860 | f32; + restore of p, m, v | 1.46e-3 | from the item 4 run |
| cpu | optimizer:adam | bfloat16 | — | — | — | — | 11.875 | f32; + restore of p, m, v | 3.68e-3 | from the item 4 run |
| cpu | optimizer:adamw | bfloat16 | — | — | — | — | 11.443 | f32; + restore of p, m, v | 3.69e-3 | from the item 4 run |
| cpu | conv | bfloat16 | — | — | — | — | 3.053 | f32 | 1.24e-1 | from the item 4 run |
| cpu | conv_bwd | bfloat16 | — | — | — | — | 20.683 | f32 | 1.25e-1 | from the item 4 run |
| cpu | lstm_step | bfloat16 | — | — | — | — | 19.539 | f32 | 1.92e-3 | from the item 4 run |
| cpu | gru_step | bfloat16 | — | — | — | — | 12.352 | f32 | 1.95e-3 | from the item 4 run |
| cpu | lstm_step_bwd | bfloat16 | — | — | — | — | 54.399 | f32 | 1.94e-3 | from the item 4 run |
| cpu | gru_step_bwd | bfloat16 | — | — | — | — | 45.092 | f32 | 1.81e-3 | from the item 4 run |
| cpu | lstm_seq | bfloat16 | — | — | — | — | 32.280 | f32 | 1.46e-3 | from the item 4 run |
| cpu | lstm_seq_bwd | bfloat16 | — | — | — | — | 139.338 | f32 | 1.75e-3 | from the item 4 run |
| cpu | gru_seq | bfloat16 | — | — | — | — | 22.748 | f32 | 1.95e-3 | from the item 4 run |
| cpu | gru_seq_bwd | bfloat16 | — | — | — | — | 92.641 | f32 | 1.95e-3 | from the item 4 run |
| cpu | topk_rows | bfloat16 | — | — | — | — | 80.310 | f32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | bfloat16 | — | — | — | — | 0.033 | f32 | 0e+0 | from the item 4 run |
| cpu | sample_rows | bfloat16 | — | — | — | — | 4.779 | f32 | 2e+0 | from the item 4 run |
| cpu | cross_entropy | bfloat16 | — | — | — | — | 15.717 | f32 (log_softmax, pick, neg) | 6.19e-2 | from the item 4 run |
| cpu | ctc | bfloat16 | — | — | — | — | 1.911 | f32 | 3.95e+0 | from the item 4 run |
| cpu | ctc_bwd | bfloat16 | — | — | — | — | 3.846 | f32 | 2.93e-3 | from the item 4 run |
| cpu | softmax | bfloat16 | 4096x4096 | — | — | — | 2.638 | f32 | 7.63e-6 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | bfloat16 | 4096x4096 | — | — | — | 0.633 | f32 | 3.91e-3 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | float8_e4m3 | 16777216 | — | — | — | 1.800 | f32 | 6.25e-2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | — | — | — | 1.235 | f32 | 2e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | float8_e4m3 | — | — | — | — | 1.644 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | float8_e4m3 | — | — | — | — | 1.200 | f32 | 0e+0 | from the item 4 run |
| cpu | map:neg | float8_e4m3 | — | — | — | — | 1.680 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | float8_e4m3 | — | — | — | — | 1.669 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | float8_e4m3 | — | — | — | — | 1.818 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square | float8_e4m3 | — | — | — | — | 1.685 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:relu | float8_e4m3 | — | — | — | — | 1.717 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | float8_e4m3 | — | — | — | — | 1.665 | f32 | 2.5e-1 | from the item 4 run |
| cpu | map:scale | float8_e4m3 | — | — | — | — | 1.668 | f32 | 5e-1 | from the item 4 run |
| cpu | map:affine | float8_e4m3 | — | — | — | — | 1.395 | f32 | 5e-1 | from the item 4 run |
| cpu | map:clamp | float8_e4m3 | — | — | — | — | 1.582 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add | float8_e4m3 | — | — | — | — | 1.818 | f32 | 2.5e-1 | from the item 4 run |
| cpu | map:sub | float8_e4m3 | — | — | — | — | 2.150 | f32 | 2.5e-1 | from the item 4 run |
| cpu | map:mul | float8_e4m3 | — | — | — | — | 1.824 | f32 | 5e-1 | from the item 4 run |
| cpu | map:div | float8_e4m3 | — | — | — | — | 1.985 | f32 | 1.17e-1 | from the item 4 run |
| cpu | map:max | float8_e4m3 | — | — | — | — | 1.876 | f32 | 0e+0 | from the item 4 run |
| cpu | map:min | float8_e4m3 | — | — | — | — | 1.889 | f32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | float8_e4m3 | — | — | — | — | 1.824 | f32 | 5e-1 | from the item 4 run |
| cpu | map:muladd | float8_e4m3 | — | — | — | — | 2.207 | f32 | 1e+0 | from the item 4 run |
| cpu | map:where | float8_e4m3 | — | — | — | — | 2.245 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | float8_e4m3 | — | — | — | — | 1.833 | f32 | 1e+0 | from the item 4 run |
| cpu | map:abs_bwd | float8_e4m3 | — | — | — | — | 1.872 | f32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | float8_e4m3 | — | — | — | — | 1.804 | f32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | float8_e4m3 | — | — | — | — | 2.200 | f32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | float8_e4m3 | — | — | — | — | 2.181 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sqrt | float8_e4m3 | — | — | — | — | 1.712 | f32 | 6.15e-2 | from the item 4 run |
| cpu | map:inv_sqrt | float8_e4m3 | — | — | — | — | 2.215 | f32 | 5.6e-2 | from the item 4 run |
| cpu | map:exp | float8_e4m3 | — | — | — | — | 3.365 | f32 | 1.48e+0 | from the item 4 run |
| cpu | map:expm1 | float8_e4m3 | — | — | — | — | 2.771 | f32 | 2.7e-2 | from the item 4 run |
| cpu | map:log | float8_e4m3 | — | — | — | — | 2.665 | f32 | 5.37e-2 | from the item 4 run |
| cpu | map:sin | float8_e4m3 | — | — | — | — | 3.603 | f32 | 3.05e-2 | from the item 4 run |
| cpu | map:cos | float8_e4m3 | — | — | — | — | 3.226 | f32 | 3.11e-2 | from the item 4 run |
| cpu | map:tanh | float8_e4m3 | — | — | — | — | 3.382 | f32 | 3.01e-2 | from the item 4 run |
| cpu | map:sigmoid | float8_e4m3 | — | — | — | — | 2.729 | f32 | 3.12e-2 | from the item 4 run |
| cpu | map:erf | float8_e4m3 | — | — | — | — | 14.075 | f32 | 3.02e-2 | from the item 4 run |
| cpu | map:gelu | float8_e4m3 | — | — | — | — | 2.924 | f32 | 5.7e-2 | from the item 4 run |
| cpu | map:gelu_erf | float8_e4m3 | — | — | — | — | 24.009 | f32 | 5.7e-2 | from the item 4 run |
| cpu | map:silu | float8_e4m3 | — | — | — | — | 2.740 | f32 | 1.21e-1 | from the item 4 run |
| cpu | map:leaky_relu | float8_e4m3 | — | — | — | — | 1.649 | f32 | 1.25e-2 | from the item 4 run |
| cpu | map:pow | float8_e4m3 | — | — | — | — | 4.690 | f32 | 9.58e-1 | from the item 4 run |
| cpu | map:dropout | float8_e4m3 | — | — | — | — | 1.846 | f32 | 1.88e-1 | from the item 4 run |
| cpu | map:gated_silu | float8_e4m3 | — | — | — | — | 3.249 | f32 | 4.94e-1 | from the item 4 run |
| cpu | map:gated_gelu | float8_e4m3 | — | — | — | — | 4.389 | f32 | 4.98e-1 | from the item 4 run |
| cpu | map:sqrt_bwd | float8_e4m3 | — | — | — | — | 2.258 | f32 | 1.22e-1 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | float8_e4m3 | — | — | — | — | 2.354 | f32 | 2.43e-1 | from the item 4 run |
| cpu | map:exp_bwd | float8_e4m3 | — | — | — | — | 2.900 | f32 | 7.5e+0 | from the item 4 run |
| cpu | map:log_bwd | float8_e4m3 | — | — | — | — | 1.985 | f32 | 2.33e-1 | from the item 4 run |
| cpu | map:sin_bwd | float8_e4m3 | — | — | — | — | 4.708 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:cos_bwd | float8_e4m3 | — | — | — | — | 4.891 | f32 | 1.24e-1 | from the item 4 run |
| cpu | map:tanh_bwd | float8_e4m3 | — | — | — | — | 4.765 | f32 | 1.22e-1 | from the item 4 run |
| cpu | map:sigmoid_bwd | float8_e4m3 | — | — | — | — | 6.157 | f32 | 3.06e-2 | from the item 4 run |
| cpu | map:silu_bwd | float8_e4m3 | — | — | — | — | 6.206 | f32 | 2.45e-1 | from the item 4 run |
| cpu | map:gelu_bwd | float8_e4m3 | — | — | — | — | 6.515 | f32 | 2.4e-1 | from the item 4 run |
| cpu | map:gelu_erf_bwd | float8_e4m3 | — | — | — | — | 45.437 | f32 | 2.37e-1 | from the item 4 run |
| cpu | map:leaky_relu_bwd | float8_e4m3 | — | — | — | — | 1.877 | f32 | 1.25e-2 | from the item 4 run |
| cpu | map:pow_bwd | float8_e4m3 | — | — | — | — | 7.716 | f32 | 3.92e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | float8_e4m3 | — | — | — | — | 4.532 | f32 | 9.79e-1 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | float8_e4m3 | — | — | — | — | 4.588 | f32 | 4.94e-1 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | float8_e4m3 | — | — | — | — | 7.037 | f32 | 4.97e-1 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | float8_e4m3 | — | — | — | — | 4.850 | f32 | 4.98e-1 | from the item 4 run |
| cpu | sum:whole | float8_e4m3 | — | — | — | — | 0.378 | f32 | 0e+0 | from the item 4 run |
| cpu | sum:rows | float8_e4m3 | — | — | — | — | 0.376 | f32 | 0e+0 | from the item 4 run |
| cpu | sum:cols | float8_e4m3 | — | — | — | — | 0.854 | f32 | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | float8_e4m3 | — | — | — | — | 0.380 | f32 | 1.6e-4 | from the item 4 run |
| cpu | max:whole | float8_e4m3 | — | — | — | — | 0.596 | f32 | 0e+0 | from the item 4 run |
| cpu | max:rows | float8_e4m3 | — | — | — | — | 1.587 | f32 | 0e+0 | from the item 4 run |
| cpu | max:cols | float8_e4m3 | — | — | — | — | 2.840 | f32 | 0e+0 | from the item 4 run |
| cpu | min:rows | float8_e4m3 | — | — | — | — | 1.592 | f32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | float8_e4m3 | — | — | — | — | 0.904 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | float8_e4m3 | — | — | — | — | 1.847 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | float8_e4m3 | — | — | — | — | 1.658 | f32 | 6.25e-2 | from the item 4 run |
| cpu | maprow:mul | float8_e4m3 | — | — | — | — | 1.659 | f32 | 3.13e-2 | from the item 4 run |
| cpu | transpose | float8_e4m3 | — | — | — | — | 2.454 | f32 | 0e+0 | from the item 4 run |
| cpu | gemv | float8_e4m3 | — | — | — | — | 1.760 | f32 | 3.99e+0 | from the item 4 run |
| cpu | gemv_t | float8_e4m3 | — | — | — | — | 5.411 | f32 | 7.52e+0 | from the item 4 run |
| cpu | row:log_softmax | float8_e4m3 | — | — | — | — | 3.528 | f32 | 5e-1 | from the item 4 run |
| cpu | row:softmax_bwd | float8_e4m3 | — | — | — | — | 2.245 | f32 | 3.99e+0 | from the item 4 run |
| cpu | row:log_softmax_bwd | float8_e4m3 | — | — | — | — | 3.119 | f32 | 4e+0 | from the item 4 run |
| cpu | row:rms_norm_affine | float8_e4m3 | — | — | — | — | 1.671 | f32 | 1.25e-1 | from the item 4 run |
| cpu | row:rms_norm_bwd | float8_e4m3 | — | — | — | — | 1.965 | f32 | 3.12e-2 | from the item 4 run |
| cpu | row:layer_norm | float8_e4m3 | — | — | — | — | 1.741 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:layer_norm_bwd | float8_e4m3 | — | — | — | — | 2.076 | f32 | 3.13e-2 | from the item 4 run |
| cpu | row:weight_norm | float8_e4m3 | — | — | — | — | 1.666 | f32 | 1.95e-3 | from the item 4 run |
| cpu | row:rope | float8_e4m3 | — | — | — | — | 5.545 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:scale_mask_softmax | float8_e4m3 | — | — | — | — | 3.357 | f32 | 2.93e-2 | from the item 4 run |
| cpu | row:add_rms_norm | float8_e4m3 | — | — | — | — | 3.338 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:add_layer_norm | float8_e4m3 | — | — | — | — | 3.384 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:cross_entropy_bwd | float8_e4m3 | — | — | — | — | 2.909 | f32 | 2.07e-3 | from the item 4 run |
| cpu | norm_params_bwd:rms | float8_e4m3 | — | — | — | — | 2.458 | f32 | 3.55e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | float8_e4m3 | — | — | — | — | 2.530 | f32 | 3.51e-4 | from the item 4 run |
| cpu | layout:strided | float8_e4m3 | — | — | — | — | 23.516 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | float8_e4m3 | — | — | — | — | 1.199 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | float8_e4m3 | — | — | — | — | 1.552 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | float8_e4m3 | — | — | — | — | 22.627 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | float8_e4m3 | — | — | — | — | 22.802 | f32 | 2.5e-1 | from the item 4 run |
| cpu | layout:im2col | float8_e4m3 | — | — | — | — | 15.961 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | float8_e4m3 | — | — | — | — | 14.043 | f32 | 2.5e-1 | from the item 4 run |
| cpu | layout:interpolate_nearest | float8_e4m3 | — | — | — | — | 30.431 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_bilinear | float8_e4m3 | — | — | — | — | 121.080 | f32 | 3.13e-2 | from the item 4 run |
| cpu | layout:shift | float8_e4m3 | — | — | — | — | 26.917 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | float8_e4m3 | — | — | — | — | 1.554 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | float8_e4m3 | — | — | — | — | 7.756 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool | float8_e4m3 | — | — | — | — | 7.376 | f32 | 3.13e-2 | from the item 4 run |
| cpu | layout:maxpool_bwd | float8_e4m3 | — | — | — | — | 41.494 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool_bwd | float8_e4m3 | — | — | — | — | 7.704 | f32 | 9.77e-4 | from the item 4 run |
| cpu | layout:adaptive_maxpool | float8_e4m3 | — | — | — | — | 115.765 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_avgpool | float8_e4m3 | — | — | — | — | 112.363 | f32 | 1.95e-3 | from the item 4 run |
| cpu | layout:depthwise_conv | float8_e4m3 | — | — | — | — | 91.236 | f32 | 2.5e-1 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | float8_e4m3 | — | — | — | — | 10102.242 | f32 | 6.25e-2 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | float8_e4m3 | — | — | — | — | 15.524 | f32 | 9.76e-4 | from the item 4 run |
| cpu | matmul_nt | float8_e4m3 | 1024x1024x1024 Bᵀ | — | — | — | 1.694 | f32 | 2e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | float8_e4m3 | 1024x1024x1024 Aᵀ | — | — | — | 2.058 | f32 | 2e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | float8_e4m3 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 2.162 | f32 | 2e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | float8_e4m3 | 16 x 256x256x256 | — | — | — | 0.796 | f32 | 9.99e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | float8_e4m3 | 1024x1024x1024 (dA, dB) | — | — | — | 4.053 | f32 | 2e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | float8_e4m3 | — | — | — | — | 55.461 | f32 | 1.95e-3 | from the item 4 run |
| cpu | attention_causal | float8_e4m3 | — | — | — | — | 43.789 | f32 | 3.12e-2 | from the item 4 run |
| cpu | attention_fused | float8_e4m3 | — | — | — | — | 81.278 | f32 | 1.95e-3 | from the item 4 run |
| cpu | attention_fused_causal | float8_e4m3 | — | — | — | — | 44.011 | f32 | 3.12e-2 | from the item 4 run |
| cpu | attention_decode | float8_e4m3 | — | — | — | — | 1.586 | f32 | 1.93e-3 | from the item 4 run |
| cpu | attention_bwd | float8_e4m3 | — | — | — | — | 23.113 | f32 | 1.53e-2 | from the item 4 run |
| cpu | optimizer:sgd_momentum | float8_e4m3 | — | — | — | — | 6.213 | f32; + restore of p, m, v | 1.46e-3 | from the item 4 run |
| cpu | optimizer:adam | float8_e4m3 | — | — | — | — | 10.387 | f32; + restore of p, m, v | 1.81e-2 | from the item 4 run |
| cpu | optimizer:adamw | float8_e4m3 | — | — | — | — | 10.489 | f32; + restore of p, m, v | 1.81e-2 | from the item 4 run |
| cpu | conv | float8_e4m3 | — | — | — | — | 3.399 | f32 | 2e+0 | from the item 4 run |
| cpu | conv_bwd | float8_e4m3 | — | — | — | — | 17.999 | f32 | 2e+0 | from the item 4 run |
| cpu | lstm_step | float8_e4m3 | — | — | — | — | 25.359 | f32 | 3.08e-2 | from the item 4 run |
| cpu | gru_step | float8_e4m3 | — | — | — | — | 12.216 | f32 | 3.12e-2 | from the item 4 run |
| cpu | lstm_step_bwd | float8_e4m3 | — | — | — | — | 52.703 | f32 | 3.09e-2 | from the item 4 run |
| cpu | gru_step_bwd | float8_e4m3 | — | — | — | — | 50.530 | f32 | 3.05e-2 | from the item 4 run |
| cpu | lstm_seq | float8_e4m3 | — | — | — | — | 38.168 | f32 | 1.77e-2 | from the item 4 run |
| cpu | lstm_seq_bwd | float8_e4m3 | — | — | — | — | 120.524 | f32 | 2.45e-2 | from the item 4 run |
| cpu | gru_seq | float8_e4m3 | — | — | — | — | 21.722 | f32 | 3.12e-2 | from the item 4 run |
| cpu | gru_seq_bwd | float8_e4m3 | — | — | — | — | 121.726 | f32 | 3.11e-2 | from the item 4 run |
| cpu | topk_rows | float8_e4m3 | — | — | — | — | 89.190 | f32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | float8_e4m3 | — | — | — | — | 0.135 | f32 | 0e+0 | from the item 4 run |
| cpu | sample_rows | float8_e4m3 | — | — | — | — | 6.393 | f32 | 1e+0 | from the item 4 run |
| cpu | cross_entropy | float8_e4m3 | — | — | — | — | 28.917 | f32 (log_softmax, pick, neg) | 5e-1 | from the item 4 run |
| cpu | ctc | float8_e4m3 | — | — | — | — | 2.098 | f32 | Infinity | from the item 4 run |
| cpu | ctc_bwd | float8_e4m3 | — | — | — | — | 4.475 | f32 | 3.22e-2 | from the item 4 run |
| cpu | softmax | float8_e4m3 | 4096x4096 | — | — | — | 3.090 | f32 | 9.76e-4 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | — | — | — | 1.357 | f32 | 6.25e-2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | float8_e5m2 | 16777216 | — | — | — | 1.797 | f32 | 1.25e-1 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | — | — | — | 1.910 | f32 | 4e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | float8_e5m2 | — | — | — | — | 1.512 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | float8_e5m2 | — | — | — | — | 1.342 | f32 | 0e+0 | from the item 4 run |
| cpu | map:neg | float8_e5m2 | — | — | — | — | 1.877 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | float8_e5m2 | — | — | — | — | 1.862 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | float8_e5m2 | — | — | — | — | 2.000 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square | float8_e5m2 | — | — | — | — | 1.855 | f32 | 1e+0 | from the item 4 run |
| cpu | map:relu | float8_e5m2 | — | — | — | — | 1.900 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | float8_e5m2 | — | — | — | — | 1.874 | f32 | 5e-1 | from the item 4 run |
| cpu | map:scale | float8_e5m2 | — | — | — | — | 1.848 | f32 | 1e+0 | from the item 4 run |
| cpu | map:affine | float8_e5m2 | — | — | — | — | 1.847 | f32 | 1e+0 | from the item 4 run |
| cpu | map:clamp | float8_e5m2 | — | — | — | — | 2.190 | f32 | 0e+0 | from the item 4 run |
| cpu | map:add | float8_e5m2 | — | — | — | — | 2.013 | f32 | 5e-1 | from the item 4 run |
| cpu | map:sub | float8_e5m2 | — | — | — | — | 2.395 | f32 | 5e-1 | from the item 4 run |
| cpu | map:mul | float8_e5m2 | — | — | — | — | 2.007 | f32 | 1e+0 | from the item 4 run |
| cpu | map:div | float8_e5m2 | — | — | — | — | 2.166 | f32 | 2.14e-1 | from the item 4 run |
| cpu | map:max | float8_e5m2 | — | — | — | — | 2.085 | f32 | 0e+0 | from the item 4 run |
| cpu | map:min | float8_e5m2 | — | — | — | — | 2.081 | f32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | float8_e5m2 | — | — | — | — | 2.027 | f32 | 1e+0 | from the item 4 run |
| cpu | map:muladd | float8_e5m2 | — | — | — | — | 2.467 | f32 | 2e+0 | from the item 4 run |
| cpu | map:where | float8_e5m2 | — | — | — | — | 2.491 | f32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | float8_e5m2 | — | — | — | — | 2.380 | f32 | 2e+0 | from the item 4 run |
| cpu | map:abs_bwd | float8_e5m2 | — | — | — | — | 2.507 | f32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | float8_e5m2 | — | — | — | — | 2.432 | f32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | float8_e5m2 | — | — | — | — | 2.431 | f32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | float8_e5m2 | — | — | — | — | 2.029 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sqrt | float8_e5m2 | — | — | — | — | 1.777 | f32 | 1.21e-1 | from the item 4 run |
| cpu | map:inv_sqrt | float8_e5m2 | — | — | — | — | 1.989 | f32 | 9.53e-2 | from the item 4 run |
| cpu | map:exp | float8_e5m2 | — | — | — | — | 2.726 | f32 | 1.4e+0 | from the item 4 run |
| cpu | map:expm1 | float8_e5m2 | — | — | — | — | 2.881 | f32 | 4.88e-2 | from the item 4 run |
| cpu | map:log | float8_e5m2 | — | — | — | — | 2.785 | f32 | 1.14e-1 | from the item 4 run |
| cpu | map:sin | float8_e5m2 | — | — | — | — | 3.752 | f32 | 5.66e-2 | from the item 4 run |
| cpu | map:cos | float8_e5m2 | — | — | — | — | 3.335 | f32 | 6.15e-2 | from the item 4 run |
| cpu | map:tanh | float8_e5m2 | — | — | — | — | 3.382 | f32 | 5.86e-2 | from the item 4 run |
| cpu | map:sigmoid | float8_e5m2 | — | — | — | — | 2.826 | f32 | 6.22e-2 | from the item 4 run |
| cpu | map:erf | float8_e5m2 | — | — | — | — | 14.446 | f32 | 4.79e-2 | from the item 4 run |
| cpu | map:gelu | float8_e5m2 | — | — | — | — | 3.021 | f32 | 1.18e-1 | from the item 4 run |
| cpu | map:gelu_erf | float8_e5m2 | — | — | — | — | 27.505 | f32 | 1.18e-1 | from the item 4 run |
| cpu | map:silu | float8_e5m2 | — | — | — | — | 2.852 | f32 | 1.9e-1 | from the item 4 run |
| cpu | map:leaky_relu | float8_e5m2 | — | — | — | — | 1.745 | f32 | 2.5e-2 | from the item 4 run |
| cpu | map:pow | float8_e5m2 | — | — | — | — | 5.041 | f32 | 1.08e+0 | from the item 4 run |
| cpu | map:dropout | float8_e5m2 | — | — | — | — | 2.016 | f32 | 3.75e-1 | from the item 4 run |
| cpu | map:gated_silu | float8_e5m2 | — | — | — | — | 3.141 | f32 | 7.59e-1 | from the item 4 run |
| cpu | map:gated_gelu | float8_e5m2 | — | — | — | — | 4.695 | f32 | 9.89e-1 | from the item 4 run |
| cpu | map:sqrt_bwd | float8_e5m2 | — | — | — | — | 2.456 | f32 | 2.14e-1 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | float8_e5m2 | — | — | — | — | 2.712 | f32 | 3.43e-1 | from the item 4 run |
| cpu | map:exp_bwd | float8_e5m2 | — | — | — | — | 4.172 | f32 | 8.5e+0 | from the item 4 run |
| cpu | map:log_bwd | float8_e5m2 | — | — | — | — | 2.602 | f32 | 4.29e-1 | from the item 4 run |
| cpu | map:sin_bwd | float8_e5m2 | — | — | — | — | 4.949 | f32 | 2.46e-1 | from the item 4 run |
| cpu | map:cos_bwd | float8_e5m2 | — | — | — | — | 3.978 | f32 | 2.39e-1 | from the item 4 run |
| cpu | map:tanh_bwd | float8_e5m2 | — | — | — | — | 6.392 | f32 | 2.47e-1 | from the item 4 run |
| cpu | map:sigmoid_bwd | float8_e5m2 | — | — | — | — | 4.285 | f32 | 6.19e-2 | from the item 4 run |
| cpu | map:silu_bwd | float8_e5m2 | — | — | — | — | 4.445 | f32 | 3.98e-1 | from the item 4 run |
| cpu | map:gelu_bwd | float8_e5m2 | — | — | — | — | 4.732 | f32 | 4.9e-1 | from the item 4 run |
| cpu | map:gelu_erf_bwd | float8_e5m2 | — | — | — | — | 30.145 | f32 | 4.91e-1 | from the item 4 run |
| cpu | map:leaky_relu_bwd | float8_e5m2 | — | — | — | — | 2.443 | f32 | 2.5e-2 | from the item 4 run |
| cpu | map:pow_bwd | float8_e5m2 | — | — | — | — | 5.407 | f32 | 6e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | float8_e5m2 | — | — | — | — | 4.720 | f32 | 1.12e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | float8_e5m2 | — | — | — | — | 3.457 | f32 | 7.59e-1 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | float8_e5m2 | — | — | — | — | 4.943 | f32 | 9.93e-1 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | float8_e5m2 | — | — | — | — | 3.723 | f32 | 9.89e-1 | from the item 4 run |
| cpu | sum:whole | float8_e5m2 | — | — | — | — | 0.622 | f32 | 3.05e-4 | from the item 4 run |
| cpu | sum:rows | float8_e5m2 | — | — | — | — | 0.729 | f32 | 0e+0 | from the item 4 run |
| cpu | sum:cols | float8_e5m2 | — | — | — | — | 1.347 | f32 | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | float8_e5m2 | — | — | — | — | 0.732 | f32 | 1.88e-4 | from the item 4 run |
| cpu | max:whole | float8_e5m2 | — | — | — | — | 1.133 | f32 | 0e+0 | from the item 4 run |
| cpu | max:rows | float8_e5m2 | — | — | — | — | 1.750 | f32 | 0e+0 | from the item 4 run |
| cpu | max:cols | float8_e5m2 | — | — | — | — | 3.218 | f32 | 0e+0 | from the item 4 run |
| cpu | min:rows | float8_e5m2 | — | — | — | — | 1.740 | f32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | float8_e5m2 | — | — | — | — | 1.726 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | float8_e5m2 | — | — | — | — | 2.058 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | float8_e5m2 | — | — | — | — | 1.876 | f32 | 1.25e-1 | from the item 4 run |
| cpu | maprow:mul | float8_e5m2 | — | — | — | — | 1.852 | f32 | 6.25e-2 | from the item 4 run |
| cpu | transpose | float8_e5m2 | — | — | — | — | 2.461 | f32 | 0e+0 | from the item 4 run |
| cpu | gemv | float8_e5m2 | — | — | — | — | 1.890 | f32 | 7.95e+0 | from the item 4 run |
| cpu | gemv_t | float8_e5m2 | — | — | — | — | 5.705 | f32 | 1.15e+1 | from the item 4 run |
| cpu | row:log_softmax | float8_e5m2 | — | — | — | — | 3.097 | f32 | 1e+0 | from the item 4 run |
| cpu | row:softmax_bwd | float8_e5m2 | — | — | — | — | 2.100 | f32 | 7.49e+0 | from the item 4 run |
| cpu | row:log_softmax_bwd | float8_e5m2 | — | — | — | — | 3.383 | f32 | 8e+0 | from the item 4 run |
| cpu | row:rms_norm_affine | float8_e5m2 | — | — | — | — | 1.769 | f32 | 2.5e-1 | from the item 4 run |
| cpu | row:rms_norm_bwd | float8_e5m2 | — | — | — | — | 2.194 | f32 | 6.25e-2 | from the item 4 run |
| cpu | row:layer_norm | float8_e5m2 | — | — | — | — | 1.832 | f32 | 5e-1 | from the item 4 run |
| cpu | row:layer_norm_bwd | float8_e5m2 | — | — | — | — | 2.304 | f32 | 6.25e-2 | from the item 4 run |
| cpu | row:weight_norm | float8_e5m2 | — | — | — | — | 1.860 | f32 | 3.91e-3 | from the item 4 run |
| cpu | row:rope | float8_e5m2 | — | — | — | — | 5.739 | f32 | 5e-1 | from the item 4 run |
| cpu | row:scale_mask_softmax | float8_e5m2 | — | — | — | — | 3.622 | f32 | 5.42e-2 | from the item 4 run |
| cpu | row:add_rms_norm | float8_e5m2 | — | — | — | — | 3.743 | f32 | 5e-1 | from the item 4 run |
| cpu | row:add_layer_norm | float8_e5m2 | — | — | — | — | 3.798 | f32 | 5e-1 | from the item 4 run |
| cpu | row:cross_entropy_bwd | float8_e5m2 | — | — | — | — | 3.707 | f32 | 2.05e-3 | from the item 4 run |
| cpu | norm_params_bwd:rms | float8_e5m2 | — | — | — | — | 4.073 | f32 | 2.41e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | float8_e5m2 | — | — | — | — | 2.550 | f32 | 3.2e-4 | from the item 4 run |
| cpu | layout:strided | float8_e5m2 | — | — | — | — | 25.488 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | float8_e5m2 | — | — | — | — | 1.318 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | float8_e5m2 | — | — | — | — | 1.656 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | float8_e5m2 | — | — | — | — | 24.231 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | float8_e5m2 | — | — | — | — | 24.328 | f32 | 5e-1 | from the item 4 run |
| cpu | layout:im2col | float8_e5m2 | — | — | — | — | 16.828 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | float8_e5m2 | — | — | — | — | 18.152 | f32 | 5e-1 | from the item 4 run |
| cpu | layout:interpolate_nearest | float8_e5m2 | — | — | — | — | 31.154 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_bilinear | float8_e5m2 | — | — | — | — | 129.252 | f32 | 6.25e-2 | from the item 4 run |
| cpu | layout:shift | float8_e5m2 | — | — | — | — | 29.107 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | float8_e5m2 | — | — | — | — | 1.675 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | float8_e5m2 | — | — | — | — | 8.537 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool | float8_e5m2 | — | — | — | — | 6.182 | f32 | 6.25e-2 | from the item 4 run |
| cpu | layout:maxpool_bwd | float8_e5m2 | — | — | — | — | 43.425 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:avgpool_bwd | float8_e5m2 | — | — | — | — | 11.023 | f32 | 7.63e-6 | from the item 4 run |
| cpu | layout:adaptive_maxpool | float8_e5m2 | — | — | — | — | 122.159 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_avgpool | float8_e5m2 | — | — | — | — | 118.057 | f32 | 3.9e-3 | from the item 4 run |
| cpu | layout:depthwise_conv | float8_e5m2 | — | — | — | — | 105.132 | f32 | 4.92e-1 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | float8_e5m2 | — | — | — | — | 10570.257 | f32 | 1.25e-1 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | float8_e5m2 | — | — | — | — | 16.005 | f32 | 2.34e-4 | from the item 4 run |
| cpu | matmul_nt | float8_e5m2 | 1024x1024x1024 Bᵀ | — | — | — | 1.736 | f32 | 4e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | float8_e5m2 | 1024x1024x1024 Aᵀ | — | — | — | 2.111 | f32 | 4e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | float8_e5m2 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 2.096 | f32 | 4e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | float8_e5m2 | 16 x 256x256x256 | — | — | — | 0.884 | f32 | 2e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | float8_e5m2 | 1024x1024x1024 (dA, dB) | — | — | — | 4.116 | f32 | 4e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | float8_e5m2 | — | — | — | — | 54.473 | f32 | 3.91e-3 | from the item 4 run |
| cpu | attention_causal | float8_e5m2 | — | — | — | — | 43.520 | f32 | 6.25e-2 | from the item 4 run |
| cpu | attention_fused | float8_e5m2 | — | — | — | — | 89.692 | f32 | 3.91e-3 | from the item 4 run |
| cpu | attention_fused_causal | float8_e5m2 | — | — | — | — | 44.370 | f32 | 6.25e-2 | from the item 4 run |
| cpu | attention_decode | float8_e5m2 | — | — | — | — | 1.962 | f32 | 2.98e-3 | from the item 4 run |
| cpu | attention_bwd | float8_e5m2 | — | — | — | — | 21.749 | f32 | 3.06e-2 | from the item 4 run |
| cpu | optimizer:sgd_momentum | float8_e5m2 | — | — | — | — | 6.745 | f32; + restore of p, m, v | 1.46e-3 | from the item 4 run |
| cpu | optimizer:adam | float8_e5m2 | — | — | — | — | 10.456 | f32; + restore of p, m, v | 2.59e-2 | from the item 4 run |
| cpu | optimizer:adamw | float8_e5m2 | — | — | — | — | 10.594 | f32; + restore of p, m, v | 2.59e-2 | from the item 4 run |
| cpu | conv | float8_e5m2 | — | — | — | — | 3.415 | f32 | 3.99e+0 | from the item 4 run |
| cpu | conv_bwd | float8_e5m2 | — | — | — | — | 18.536 | f32 | 4e+0 | from the item 4 run |
| cpu | lstm_step | float8_e5m2 | — | — | — | — | 28.894 | f32 | 4.62e-2 | from the item 4 run |
| cpu | gru_step | float8_e5m2 | — | — | — | — | 15.798 | f32 | 6.25e-2 | from the item 4 run |
| cpu | lstm_step_bwd | float8_e5m2 | — | — | — | — | 53.860 | f32 | 6.16e-2 | from the item 4 run |
| cpu | gru_step_bwd | float8_e5m2 | — | — | — | — | 45.032 | f32 | 5.97e-2 | from the item 4 run |
| cpu | lstm_seq | float8_e5m2 | — | — | — | — | 32.572 | f32 | 5.59e-2 | from the item 4 run |
| cpu | lstm_seq_bwd | float8_e5m2 | — | — | — | — | 132.114 | f32 | 5.98e-2 | from the item 4 run |
| cpu | gru_seq | float8_e5m2 | — | — | — | — | 20.450 | f32 | 6.25e-2 | from the item 4 run |
| cpu | gru_seq_bwd | float8_e5m2 | — | — | — | — | 99.344 | f32 | 6.24e-2 | from the item 4 run |
| cpu | topk_rows | float8_e5m2 | — | — | — | — | 102.985 | f32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | float8_e5m2 | — | — | — | — | 0.145 | f32 | 0e+0 | from the item 4 run |
| cpu | sample_rows | float8_e5m2 | — | — | — | — | 6.319 | f32 | 1e+0 | from the item 4 run |
| cpu | cross_entropy | float8_e5m2 | — | — | — | — | 24.899 | f32 (log_softmax, pick, neg) | 9.84e-1 | from the item 4 run |
| cpu | ctc | float8_e5m2 | — | — | — | — | 2.010 | f32 | 1.27e+2 | from the item 4 run |
| cpu | ctc_bwd | float8_e5m2 | — | — | — | — | 4.412 | f32 | 6.31e-2 | from the item 4 run |
| cpu | softmax | float8_e5m2 | 4096x4096 | — | — | — | 3.901 | f32 | 1.62e-4 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | — | — | — | 1.782 | f32 | 1.08e-1 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | int64 | 16777216 | 6.621 | Vector<Int64> | 0e+0 | 6.893 | s64 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | int64 | 1024x1024x1024 | 32.382 | Vector<Int64> | 0e+0 | 0.757 | s8dp | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | int64 | — | — | — | — | 4.739 | s64 | 0e+0 | from the item 4 run |
| cpu | map:fill | int64 | — | — | — | — | 2.649 | s64 | 0e+0 | from the item 4 run |
| cpu | map:neg | int64 | — | — | — | — | 4.716 | s64 | 0e+0 | from the item 4 run |
| cpu | map:abs | int64 | — | — | — | — | 4.698 | s64 | 0e+0 | from the item 4 run |
| cpu | map:sign | int64 | — | — | — | — | 4.648 | s64 | 0e+0 | from the item 4 run |
| cpu | map:square | int64 | — | — | — | — | 4.672 | s64 | 0e+0 | from the item 4 run |
| cpu | map:relu | int64 | — | — | — | — | 4.754 | s64 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | int64 | — | — | — | — | 4.726 | s64 | 0e+0 | from the item 4 run |
| cpu | map:scale | int64 | — | — | — | — | 4.666 | s64 | 0e+0 | from the item 4 run |
| cpu | map:affine | int64 | — | — | — | — | 5.041 | s64 | 0e+0 | from the item 4 run |
| cpu | map:clamp | int64 | — | — | — | — | 4.661 | s64 | 0e+0 | from the item 4 run |
| cpu | map:add | int64 | — | — | — | — | 6.621 | s64 | 0e+0 | from the item 4 run |
| cpu | map:sub | int64 | — | — | — | — | 6.706 | s64 | 0e+0 | from the item 4 run |
| cpu | map:mul | int64 | — | — | — | — | 6.659 | s64 | 0e+0 | from the item 4 run |
| cpu | map:div | int64 | — | — | — | — | 6.721 | s64 | 0e+0 | from the item 4 run |
| cpu | map:max | int64 | — | — | — | — | 6.695 | s64 | 0e+0 | from the item 4 run |
| cpu | map:min | int64 | — | — | — | — | 6.855 | s64 | 0e+0 | from the item 4 run |
| cpu | map:axpy | int64 | — | — | — | — | 6.707 | s64 | 0e+0 | from the item 4 run |
| cpu | map:muladd | int64 | — | — | — | — | 8.689 | s64 | 0e+0 | from the item 4 run |
| cpu | map:where | int64 | — | — | — | — | 9.305 | s64 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | int64 | — | — | — | — | 6.832 | s64 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | int64 | — | — | — | — | 6.883 | s64 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | int64 | — | — | — | — | 6.823 | s64 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | int64 | — | — | — | — | 6.910 | s64 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | int64 | — | — | — | — | 6.864 | s64 | 0e+0 | from the item 4 run |
| cpu | sum:whole | int64 | — | — | — | — | 2.551 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | int64 | — | — | — | — | 2.557 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | int64 | — | — | — | — | 4.281 | s8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | int64 | — | — | — | — | 2.606 | s8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | int64 | — | — | — | — | 1.269 | s64 | 0e+0 | from the item 4 run |
| cpu | max:rows | int64 | — | — | — | — | 1.290 | s64 | 0e+0 | from the item 4 run |
| cpu | max:cols | int64 | — | — | — | — | 6.777 | s64 | 0e+0 | from the item 4 run |
| cpu | min:rows | int64 | — | — | — | — | 1.418 | s64 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | int64 | — | — | — | — | 1.356 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | int64 | — | — | — | — | 4.994 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | int64 | — | — | — | — | 4.707 | s64 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | int64 | — | — | — | — | 4.702 | s64 | 0e+0 | from the item 4 run |
| cpu | transpose | int64 | — | — | — | — | 6.263 | s64 | 0e+0 | from the item 4 run |
| cpu | gemv | int64 | — | — | — | — | 12.635 | s8dp | 0e+0 | from the item 4 run |
| cpu | gemv_t | int64 | — | — | — | — | 16.259 | s8dp | 0e+0 | from the item 4 run |
| cpu | layout:strided | int64 | — | — | — | — | 20.259 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | int64 | — | — | — | — | 2.465 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | int64 | — | — | — | — | 4.185 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | int64 | — | — | — | — | 5.517 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | int64 | — | — | — | — | 7.256 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | int64 | — | — | — | — | 1.578 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | int64 | — | — | — | — | 3.489 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | int64 | — | — | — | — | 3.429 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:shift | int64 | — | — | — | — | 4.623 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | int64 | — | — | — | — | 3.821 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | int64 | — | — | — | — | 1.991 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | int64 | — | — | — | — | 11.345 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | int64 | — | — | — | — | 14.725 | s64 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | int64 | — | — | — | — | 35.110 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | int64 | — | — | — | — | 2889.760 | s32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | int64 | 1024x1024x1024 Bᵀ | — | — | — | 0.681 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | int64 | 1024x1024x1024 Aᵀ | — | — | — | 0.742 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | int64 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.994 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | int64 | 16 x 256x256x256 | — | — | — | 1.063 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | int64 | 1024x1024x1024 (dA, dB) | — | — | — | 1.865 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | int64 | — | — | — | — | 1.346 | s8dp | 0e+0 | from the item 4 run |
| cpu | topk_rows | int64 | — | — | — | — | 302.096 | s64 | 0e+0 | from the item 4 run |
| cpu | pick_rows | int64 | — | — | — | — | 0.028 | s64 | 0e+0 | from the item 4 run |
| cpu | add | int32 | 16777216 | 3.034 | Vector<Int32> | 0e+0 | 3.012 | s32 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | int32 | 1024x1024x1024 | 5.932 | Vector<Int32> | 0e+0 | 0.722 | s8dp | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | int32 | — | — | — | — | 1.689 | s32 | 0e+0 | from the item 4 run |
| cpu | map:fill | int32 | — | — | — | — | 0.360 | s32 | 0e+0 | from the item 4 run |
| cpu | map:neg | int32 | — | — | — | — | 1.796 | s32 | 0e+0 | from the item 4 run |
| cpu | map:abs | int32 | — | — | — | — | 1.636 | s32 | 0e+0 | from the item 4 run |
| cpu | map:sign | int32 | — | — | — | — | 1.756 | s32 | 0e+0 | from the item 4 run |
| cpu | map:square | int32 | — | — | — | — | 1.948 | s32 | 0e+0 | from the item 4 run |
| cpu | map:relu | int32 | — | — | — | — | 1.716 | s32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | int32 | — | — | — | — | 1.891 | s32 | 0e+0 | from the item 4 run |
| cpu | map:scale | int32 | — | — | — | — | 1.747 | s32 | 0e+0 | from the item 4 run |
| cpu | map:affine | int32 | — | — | — | — | 1.910 | s32 | 0e+0 | from the item 4 run |
| cpu | map:clamp | int32 | — | — | — | — | 1.809 | s32 | 0e+0 | from the item 4 run |
| cpu | map:add | int32 | — | — | — | — | 2.993 | s32 | 0e+0 | from the item 4 run |
| cpu | map:sub | int32 | — | — | — | — | 2.991 | s32 | 0e+0 | from the item 4 run |
| cpu | map:mul | int32 | — | — | — | — | 3.001 | s32 | 0e+0 | from the item 4 run |
| cpu | map:div | int32 | — | — | — | — | 2.984 | s32 | 0e+0 | from the item 4 run |
| cpu | map:max | int32 | — | — | — | — | 3.102 | s32 | 0e+0 | from the item 4 run |
| cpu | map:min | int32 | — | — | — | — | 3.032 | s32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | int32 | — | — | — | — | 2.998 | s32 | 0e+0 | from the item 4 run |
| cpu | map:muladd | int32 | — | — | — | — | 4.097 | s32 | 0e+0 | from the item 4 run |
| cpu | map:where | int32 | — | — | — | — | 4.110 | s32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | int32 | — | — | — | — | 2.884 | s32 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | int32 | — | — | — | — | 2.891 | s32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | int32 | — | — | — | — | 2.961 | s32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | int32 | — | — | — | — | 2.910 | s32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | int32 | — | — | — | — | 2.875 | s32 | 0e+0 | from the item 4 run |
| cpu | sum:whole | int32 | — | — | — | — | 0.764 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | int32 | — | — | — | — | 0.778 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | int32 | — | — | — | — | 1.713 | s8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | int32 | — | — | — | — | 0.796 | s8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | int32 | — | — | — | — | 0.377 | s32 | 0e+0 | from the item 4 run |
| cpu | max:rows | int32 | — | — | — | — | 0.391 | s32 | 0e+0 | from the item 4 run |
| cpu | max:cols | int32 | — | — | — | — | 2.329 | s32 | 0e+0 | from the item 4 run |
| cpu | min:rows | int32 | — | — | — | — | 0.384 | s32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | int32 | — | — | — | — | 0.378 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | int32 | — | — | — | — | 5.220 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | int32 | — | — | — | — | 1.745 | s32 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | int32 | — | — | — | — | 1.727 | s32 | 0e+0 | from the item 4 run |
| cpu | transpose | int32 | — | — | — | — | 2.925 | s32 | 0e+0 | from the item 4 run |
| cpu | gemv | int32 | — | — | — | — | 5.842 | s8dp | 0e+0 | from the item 4 run |
| cpu | gemv_t | int32 | — | — | — | — | 8.337 | s8dp | 0e+0 | from the item 4 run |
| cpu | layout:strided | int32 | — | — | — | — | 14.035 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | int32 | — | — | — | — | 0.523 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | int32 | — | — | — | — | 1.446 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | int32 | — | — | — | — | 2.230 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | int32 | — | — | — | — | 2.613 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | int32 | — | — | — | — | 1.040 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | int32 | — | — | — | — | 2.502 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | int32 | — | — | — | — | 2.284 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:shift | int32 | — | — | — | — | 2.216 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | int32 | — | — | — | — | 1.951 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | int32 | — | — | — | — | 1.943 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | int32 | — | — | — | — | 7.700 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | int32 | — | — | — | — | 14.880 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | int32 | — | — | — | — | 5.207 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | int32 | — | — | — | — | 1399.303 | s32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | int32 | 1024x1024x1024 Bᵀ | — | — | — | 0.585 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | int32 | 1024x1024x1024 Aᵀ | — | — | — | 0.803 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | int32 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.943 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | int32 | 16 x 256x256x256 | — | — | — | 0.517 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | int32 | 1024x1024x1024 (dA, dB) | — | — | — | 1.526 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | int32 | — | — | — | — | 0.940 | s8dp | 0e+0 | from the item 4 run |
| cpu | topk_rows | int32 | — | — | — | — | 304.829 | s32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | int32 | — | — | — | — | 0.027 | s32 | 0e+0 | from the item 4 run |
| cpu | add | int16 | 16777216 | 1.047 | Vector<Int16> | 0e+0 | 0.900 | s16 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | int16 | 1024x1024x1024 | 1.845 | Vector<Int16> | 0e+0 | 0.450 | s8dp | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | int16 | — | — | — | — | 0.271 | s16 | 0e+0 | from the item 4 run |
| cpu | map:fill | int16 | — | — | — | — | 0.129 | s16 | 0e+0 | from the item 4 run |
| cpu | map:neg | int16 | — | — | — | — | 0.277 | s16 | 0e+0 | from the item 4 run |
| cpu | map:abs | int16 | — | — | — | — | 0.315 | s16 | 0e+0 | from the item 4 run |
| cpu | map:sign | int16 | — | — | — | — | 0.326 | s16 | 0e+0 | from the item 4 run |
| cpu | map:square | int16 | — | — | — | — | 0.267 | s16 | 0e+0 | from the item 4 run |
| cpu | map:relu | int16 | — | — | — | — | 0.259 | s16 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | int16 | — | — | — | — | 0.322 | s16 | 0e+0 | from the item 4 run |
| cpu | map:scale | int16 | — | — | — | — | 0.281 | s16 | 0e+0 | from the item 4 run |
| cpu | map:affine | int16 | — | — | — | — | 0.276 | s16 | 0e+0 | from the item 4 run |
| cpu | map:clamp | int16 | — | — | — | — | 0.347 | s16 | 0e+0 | from the item 4 run |
| cpu | map:add | int16 | — | — | — | — | 0.790 | s16 | 0e+0 | from the item 4 run |
| cpu | map:sub | int16 | — | — | — | — | 0.874 | s16 | 0e+0 | from the item 4 run |
| cpu | map:mul | int16 | — | — | — | — | 0.793 | s16 | 0e+0 | from the item 4 run |
| cpu | map:div | int16 | — | — | — | — | 1.419 | s16 | 0e+0 | from the item 4 run |
| cpu | map:max | int16 | — | — | — | — | 0.789 | s16 | 0e+0 | from the item 4 run |
| cpu | map:min | int16 | — | — | — | — | 0.850 | s16 | 0e+0 | from the item 4 run |
| cpu | map:axpy | int16 | — | — | — | — | 0.783 | s16 | 0e+0 | from the item 4 run |
| cpu | map:muladd | int16 | — | — | — | — | 1.390 | s16 | 0e+0 | from the item 4 run |
| cpu | map:where | int16 | — | — | — | — | 1.321 | s16 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | int16 | — | — | — | — | 0.803 | s16 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | int16 | — | — | — | — | 0.817 | s16 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | int16 | — | — | — | — | 0.789 | s16 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | int16 | — | — | — | — | 0.793 | s16 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | int16 | — | — | — | — | 0.922 | s16 | 0e+0 | from the item 4 run |
| cpu | sum:whole | int16 | — | — | — | — | 0.188 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | int16 | — | — | — | — | 0.193 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | int16 | — | — | — | — | 0.687 | s8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | int16 | — | — | — | — | 0.183 | s8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | int16 | — | — | — | — | 0.094 | s16 | 0e+0 | from the item 4 run |
| cpu | max:rows | int16 | — | — | — | — | 0.103 | s16 | 0e+0 | from the item 4 run |
| cpu | max:cols | int16 | — | — | — | — | 1.898 | s16 | 0e+0 | from the item 4 run |
| cpu | min:rows | int16 | — | — | — | — | 0.128 | s16 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | int16 | — | — | — | — | 0.105 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | int16 | — | — | — | — | 4.699 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | int16 | — | — | — | — | 0.292 | s16 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | int16 | — | — | — | — | 0.306 | s16 | 0e+0 | from the item 4 run |
| cpu | transpose | int16 | — | — | — | — | 0.966 | s16 | 0e+0 | from the item 4 run |
| cpu | gemv | int16 | — | — | — | — | 2.519 | s8dp | 0e+0 | from the item 4 run |
| cpu | gemv_t | int16 | — | — | — | — | 5.225 | s8dp | 0e+0 | from the item 4 run |
| cpu | layout:strided | int16 | — | — | — | — | 6.018 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | int16 | — | — | — | — | 0.118 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | int16 | — | — | — | — | 0.335 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | int16 | — | — | — | — | 0.935 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | int16 | — | — | — | — | 3.591 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | int16 | — | — | — | — | 0.990 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | int16 | — | — | — | — | 3.078 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | int16 | — | — | — | — | 1.829 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:shift | int16 | — | — | — | — | 1.049 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | int16 | — | — | — | — | 0.647 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | int16 | — | — | — | — | 1.617 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | int16 | — | — | — | — | 11.946 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | int16 | — | — | — | — | 15.253 | s16 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | int16 | — | — | — | — | 18.564 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | int16 | — | — | — | — | 3718.135 | s32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | int16 | 1024x1024x1024 Bᵀ | — | — | — | 0.493 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | int16 | 1024x1024x1024 Aᵀ | — | — | — | 0.788 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | int16 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.958 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | int16 | 16 x 256x256x256 | — | — | — | 0.403 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | int16 | 1024x1024x1024 (dA, dB) | — | — | — | 1.109 | s8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | int16 | — | — | — | — | 0.980 | s8dp | 0e+0 | from the item 4 run |
| cpu | topk_rows | int16 | — | — | — | — | 313.804 | s16 | 0e+0 | from the item 4 run |
| cpu | pick_rows | int16 | — | — | — | — | 0.025 | s16 | 0e+0 | from the item 4 run |
| cpu | add | int8 | 16777216 | 0.483 | Vector<SByte> | 0e+0 | 0.173 | s8 | 0e+0 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | int8 | 1024x1024x1024 | 2.502 | Vector<SByte> | 2.56e+2 | 0.505 | s8dp | 1.76e+2 | number: end-of-item-6 run, invocation 2 (core and number); qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | int8 | — | — | — | — | 0.078 | s8 | 0e+0 | from the item 4 run |
| cpu | map:fill | int8 | — | — | — | — | 0.082 | s8 | 0e+0 | from the item 4 run |
| cpu | map:neg | int8 | — | — | — | — | 0.126 | s8 | 0e+0 | from the item 4 run |
| cpu | map:abs | int8 | — | — | — | — | 0.079 | s8 | 0e+0 | from the item 4 run |
| cpu | map:sign | int8 | — | — | — | — | 0.080 | s8 | 0e+0 | from the item 4 run |
| cpu | map:square | int8 | — | — | — | — | 0.078 | s8 | 0e+0 | from the item 4 run |
| cpu | map:relu | int8 | — | — | — | — | 0.118 | s8 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | int8 | — | — | — | — | 0.080 | s8 | 0e+0 | from the item 4 run |
| cpu | map:scale | int8 | — | — | — | — | 0.080 | s8 | 0e+0 | from the item 4 run |
| cpu | map:affine | int8 | — | — | — | — | 0.079 | s8 | 0e+0 | from the item 4 run |
| cpu | map:clamp | int8 | — | — | — | — | 0.122 | s8 | 0e+0 | from the item 4 run |
| cpu | map:add | int8 | — | — | — | — | 0.212 | s8 | 0e+0 | from the item 4 run |
| cpu | map:sub | int8 | — | — | — | — | 0.200 | s8 | 0e+0 | from the item 4 run |
| cpu | map:mul | int8 | — | — | — | — | 0.159 | s8 | 0e+0 | from the item 4 run |
| cpu | map:div | int8 | — | — | — | — | 1.441 | s8 | 0e+0 | from the item 4 run |
| cpu | map:max | int8 | — | — | — | — | 0.176 | s8 | 0e+0 | from the item 4 run |
| cpu | map:min | int8 | — | — | — | — | 0.178 | s8 | 0e+0 | from the item 4 run |
| cpu | map:axpy | int8 | — | — | — | — | 0.196 | s8 | 0e+0 | from the item 4 run |
| cpu | map:muladd | int8 | — | — | — | — | 0.307 | s8 | 0e+0 | from the item 4 run |
| cpu | map:where | int8 | — | — | — | — | 0.297 | s8 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | int8 | — | — | — | — | 0.218 | s8 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | int8 | — | — | — | — | 0.196 | s8 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | int8 | — | — | — | — | 0.176 | s8 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | int8 | — | — | — | — | 0.191 | s8 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | int8 | — | — | — | — | 0.170 | s8 | 0e+0 | from the item 4 run |
| cpu | sum:whole | int8 | — | — | — | — | 0.073 | s8dp | 0e+0 | 6k CPU sums A/B (HEAD) |
| cpu | sum:rows | int8 | — | — | — | — | 0.137 | s8dp | 0e+0 | 6k CPU sums A/B (HEAD) |
| cpu | sum:cols | int8 | 4096x4096 | — | — | — | 0.233 | s8dp | 0e+0 | 6k CPU sums (cooled) |
| cpu | sumsq:rows | int8 | — | — | — | — | 0.032 | s8dp | 0e+0 | 6k CPU sums A/B (HEAD) |
| cpu | max:whole | int8 | — | — | — | — | 0.039 | s8 | 0e+0 | 6j part 4: CPU reductions |
| cpu | max:rows | int8 | 4096x4096 | — | — | — | 0.070 | s8 | 0e+0 | 6k CPU sums (cooled) |
| cpu | max:cols | int8 | — | — | — | — | 0.096 | s8 | 0e+0 | 6j part 4: CPU reductions |
| cpu | min:rows | int8 | — | — | — | — | 0.075 | s8 | 0e+0 | 6j part 4: CPU reductions |
| cpu | argmax:rows | int8 | — | — | — | — | 0.075 | s32 index | 0e+0 | 6j part 4: CPU reductions |
| cpu | argmax:cols | int8 | — | — | — | — | 1.423 | s32 index | 0e+0 | 6j part 4: CPU reductions |
| cpu | maprow:add | int8 | — | — | — | — | 0.081 | s8 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | int8 | — | — | — | — | 0.084 | s8 | 0e+0 | from the item 4 run |
| cpu | transpose | int8 | — | — | — | — | 0.560 | s8 | 0e+0 | from the item 4 run |
| cpu | gemv | int8 | — | — | — | — | 0.769 | s8dp | 5.72e+2 | from the item 4 run |
| cpu | gemv_t | int8 | — | — | — | — | 2.401 | s8dp | 6.32e+2 | from the item 4 run |
| cpu | layout:strided | int8 | 1x1024x64x256 | — | — | — | 0.595 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:one_hot | int8 | 16384x1024 | — | — | — | 0.662 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:gather_rows | int8 | 4096 of 4096x4096 | — | — | — | 0.088 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:gather_cols | int8 | 4096 of 4096x4096 | — | — | — | 0.791 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:scatter_add_rows | int8 | 4096 into 4096x4096 | — | — | — | 2.807 | s32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:im2col | int8 | 64x128x128 k3x3 | — | — | — | 1.114 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:col2im | int8 | 64x128x128 k3x3 | — | — | — | 1.867 | s32 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:interpolate_nearest | int8 | 16x512x512 to 1024x1024 | — | — | — | 1.985 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:shift | int8 | 4096x4096 by 3 | — | — | — | 0.795 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:pad2d | int8 | 4096x4093 to 4096 | — | — | — | 0.089 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:maxpool | int8 | 64x256x256 k2x2 s2 p0 | — | — | — | 1.731 | s8 | 0e+0 | small cells 2: CPU layout |
| cpu | layout:maxpool_bwd | int8 | 64x256x256 k2x2 s2 p0 | — | — | — | 5.170 | s32 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:adaptive_maxpool | int8 | 64x256x256 to 7x7 | — | — | — | 0.911 | s8 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:depthwise_conv | int8 | 64x256x256 k3x3 s1 p1 | — | — | — | 0.736 | s32 | 0e+0 | small cells 2: CPU pools |
| cpu | layout:adaptive_maxpool_bwd | int8 | 64x256x256 to 7x7 | — | — | — | 1.317 | s32 | 0e+0 | small cells 2: CPU layout |
| cpu | matmul_nt | int8 | 1024x1024x1024 Bᵀ | — | — | — | 0.502 | s8dp | 1.88e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | int8 | 1024x1024x1024 Aᵀ | — | — | — | 0.635 | s8dp | 1.98e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | int8 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.615 | s8dp | 1.85e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | int8 | 16 x 256x256x256 | — | — | — | 0.371 | s8dp | 2.6e+1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | int8 | 1024x1024x1024 (dA, dB) | — | — | — | 1.128 | s8dp | 2.22e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | int8 | — | — | — | — | 0.931 | s8dp | 1.27e+2 | from the item 4 run |
| cpu | topk_rows | int8 | — | — | — | — | 313.203 | s8 | 0e+0 | from the item 4 run |
| cpu | pick_rows | int8 | — | — | — | — | 0.024 | s8 | 0e+0 | from the item 4 run |
| cpu | add | int4 | 16777216 | — | — | — | 0.101 | s8 | 0e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | int4 | 1024x1024x1024 | — | — | — | 0.350 | s8dp | 2.96e+2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | int4 | — | — | — | — | 0.288 | s8 | 0e+0 | from the item 4 run |
| cpu | map:fill | int4 | — | — | — | — | 0.250 | s8 | 0e+0 | from the item 4 run |
| cpu | map:neg | int4 | — | — | — | — | 0.290 | s8 | 0e+0 | from the item 4 run |
| cpu | map:abs | int4 | — | — | — | — | 0.278 | s8 | 0e+0 | from the item 4 run |
| cpu | map:sign | int4 | — | — | — | — | 0.271 | s8 | 0e+0 | from the item 4 run |
| cpu | map:square | int4 | — | — | — | — | 0.288 | s8 | 0e+0 | from the item 4 run |
| cpu | map:relu | int4 | — | — | — | — | 0.291 | s8 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | int4 | — | — | — | — | 0.276 | s8 | 0e+0 | from the item 4 run |
| cpu | map:scale | int4 | — | — | — | — | 0.290 | s8 | 0e+0 | from the item 4 run |
| cpu | map:affine | int4 | — | — | — | — | 0.308 | s8 | 0e+0 | from the item 4 run |
| cpu | map:clamp | int4 | — | — | — | — | 0.306 | s8 | 0e+0 | from the item 4 run |
| cpu | map:add | int4 | — | — | — | — | 0.294 | s8 | 0e+0 | from the item 4 run |
| cpu | map:sub | int4 | — | — | — | — | 0.298 | s8 | 0e+0 | from the item 4 run |
| cpu | map:mul | int4 | — | — | — | — | 0.298 | s8 | 0e+0 | from the item 4 run |
| cpu | map:div | int4 | — | — | — | — | 1.324 | s8 | 0e+0 | from the item 4 run |
| cpu | map:max | int4 | — | — | — | — | 0.295 | s8 | 0e+0 | from the item 4 run |
| cpu | map:min | int4 | — | — | — | — | 0.301 | s8 | 0e+0 | from the item 4 run |
| cpu | map:axpy | int4 | — | — | — | — | 0.298 | s8 | 0e+0 | from the item 4 run |
| cpu | map:muladd | int4 | — | — | — | — | 0.341 | s8 | 0e+0 | from the item 4 run |
| cpu | map:where | int4 | — | — | — | — | 0.326 | s8 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | int4 | — | — | — | — | 0.325 | s8 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | int4 | — | — | — | — | 0.310 | s8 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | int4 | — | — | — | — | 0.298 | s8 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | int4 | — | — | — | — | 0.301 | s8 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | int4 | — | — | — | — | 0.305 | s8 | 0e+0 | from the item 4 run |
| cpu | sum:whole | int4 | — | — | — | — | 0.042 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | int4 | — | — | — | — | 0.059 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | int4 | — | — | — | — | 0.521 | s8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | int4 | — | — | — | — | 0.060 | s8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | int4 | — | — | — | — | 0.103 | s8 | 0e+0 | from the item 4 run |
| cpu | max:rows | int4 | — | — | — | — | 0.143 | s8 | 0e+0 | from the item 4 run |
| cpu | max:cols | int4 | — | — | — | — | 1.806 | s8 | 0e+0 | from the item 4 run |
| cpu | min:rows | int4 | — | — | — | — | 0.113 | s8 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | int4 | — | — | — | — | 0.110 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | int4 | — | — | — | — | 4.479 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | int4 | — | — | — | — | 0.745 | s8 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | int4 | — | — | — | — | 0.193 | s8 | 0e+0 | from the item 4 run |
| cpu | transpose | int4 | — | — | — | — | 0.910 | s8 | 0e+0 | from the item 4 run |
| cpu | gemv | int4 | — | — | — | — | 0.322 | s8dp | 6.92e+2 | from the item 4 run |
| cpu | gemv_t | int4 | — | — | — | — | 2.959 | s8dp | 7.52e+2 | from the item 4 run |
| cpu | layout:strided | int4 | — | — | — | — | 7.027 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | int4 | — | — | — | — | 0.088 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | int4 | — | — | — | — | 0.094 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | int4 | — | — | — | — | 12.205 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | int4 | — | — | — | — | 31.326 | s32 | 3e+0 | from the item 4 run |
| cpu | layout:im2col | int4 | — | — | — | — | 4.656 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | int4 | — | — | — | — | 23.844 | s32 | 1.1e+1 | from the item 4 run |
| cpu | layout:interpolate_nearest | int4 | — | — | — | — | 14.842 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:shift | int4 | — | — | — | — | 14.407 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | int4 | — | — | — | — | 0.100 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | int4 | — | — | — | — | 4.814 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | int4 | — | — | — | — | 50.713 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | int4 | — | — | — | — | 41.506 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | int4 | — | — | — | — | 167.001 | s32 | 2.2e+1 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | int4 | — | — | — | — | 13605.755 | s32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | int4 | 1024x1024x1024 Bᵀ | — | — | — | 1.691 | s8dp | 3.08e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | int4 | 1024x1024x1024 Aᵀ | — | — | — | 1.039 | s8dp | 3.18e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | int4 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 1.155 | s8dp | 3.05e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | int4 | 16 x 256x256x256 | — | — | — | 1.139 | s8dp | 1.46e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | int4 | 1024x1024x1024 (dA, dB) | — | — | — | 2.192 | s8dp | 3.42e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | int4 | — | — | — | — | 0.954 | s8dp | 2.47e+2 | from the item 4 run |
| cpu | topk_rows | int4 | — | — | — | — | 307.286 | s8 | 0e+0 | from the item 4 run |
| cpu | pick_rows | int4 | — | — | — | — | 0.032 | s8 | 0e+0 | from the item 4 run |
| cpu | add | int2 | 16777216 | — | — | — | 0.189 | s8 | 0e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | int2 | 1024x1024x1024 | — | — | — | 0.354 | s8dp | 3.26e+2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | int2 | — | — | — | — | 0.309 | s8 | 0e+0 | from the item 4 run |
| cpu | map:fill | int2 | — | — | — | — | 0.290 | s8 | 0e+0 | from the item 4 run |
| cpu | map:neg | int2 | — | — | — | — | 0.310 | s8 | 0e+0 | from the item 4 run |
| cpu | map:abs | int2 | — | — | — | — | 0.309 | s8 | 0e+0 | from the item 4 run |
| cpu | map:sign | int2 | — | — | — | — | 0.310 | s8 | 0e+0 | from the item 4 run |
| cpu | map:square | int2 | — | — | — | — | 0.302 | s8 | 0e+0 | from the item 4 run |
| cpu | map:relu | int2 | — | — | — | — | 0.341 | s8 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | int2 | — | — | — | — | 0.306 | s8 | 0e+0 | from the item 4 run |
| cpu | map:scale | int2 | — | — | — | — | 0.305 | s8 | 0e+0 | from the item 4 run |
| cpu | map:affine | int2 | — | — | — | — | 0.316 | s8 | 0e+0 | from the item 4 run |
| cpu | map:clamp | int2 | — | — | — | — | 0.310 | s8 | 0e+0 | from the item 4 run |
| cpu | map:add | int2 | — | — | — | — | 0.340 | s8 | 0e+0 | from the item 4 run |
| cpu | map:sub | int2 | — | — | — | — | 0.337 | s8 | 0e+0 | from the item 4 run |
| cpu | map:mul | int2 | — | — | — | — | 0.361 | s8 | 0e+0 | from the item 4 run |
| cpu | map:div | int2 | — | — | — | — | 1.549 | s8 | 0e+0 | from the item 4 run |
| cpu | map:max | int2 | — | — | — | — | 0.468 | s8 | 0e+0 | from the item 4 run |
| cpu | map:min | int2 | — | — | — | — | 0.467 | s8 | 0e+0 | from the item 4 run |
| cpu | map:axpy | int2 | — | — | — | — | 0.361 | s8 | 0e+0 | from the item 4 run |
| cpu | map:muladd | int2 | — | — | — | — | 0.416 | s8 | 0e+0 | from the item 4 run |
| cpu | map:where | int2 | — | — | — | — | 0.391 | s8 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | int2 | — | — | — | — | 0.407 | s8 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | int2 | — | — | — | — | 0.380 | s8 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | int2 | — | — | — | — | 0.337 | s8 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | int2 | — | — | — | — | 0.338 | s8 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | int2 | — | — | — | — | 0.333 | s8 | 0e+0 | from the item 4 run |
| cpu | sum:whole | int2 | — | — | — | — | 0.067 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | int2 | — | — | — | — | 0.067 | s8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | int2 | — | — | — | — | 0.592 | s8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | int2 | — | — | — | — | 0.069 | s8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | int2 | — | — | — | — | 0.147 | s8 | 0e+0 | from the item 4 run |
| cpu | max:rows | int2 | — | — | — | — | 0.224 | s8 | 0e+0 | from the item 4 run |
| cpu | max:cols | int2 | — | — | — | — | 1.818 | s8 | 0e+0 | from the item 4 run |
| cpu | min:rows | int2 | — | — | — | — | 0.152 | s8 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | int2 | — | — | — | — | 0.256 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | int2 | — | — | — | — | 7.130 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | int2 | — | — | — | — | 0.263 | s8 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | int2 | — | — | — | — | 0.267 | s8 | 0e+0 | from the item 4 run |
| cpu | transpose | int2 | — | — | — | — | 3.190 | s8 | 0e+0 | from the item 4 run |
| cpu | gemv | int2 | — | — | — | — | 0.552 | s8dp | 2.15e+3 | from the item 4 run |
| cpu | gemv_t | int2 | — | — | — | — | 3.892 | s8dp | 2.15e+3 | from the item 4 run |
| cpu | layout:strided | int2 | — | — | — | — | 12.733 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | int2 | — | — | — | — | 0.171 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | int2 | — | — | — | — | 0.214 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | int2 | — | — | — | — | 17.705 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | int2 | — | — | — | — | 29.820 | s32 | 3e+0 | from the item 4 run |
| cpu | layout:im2col | int2 | — | — | — | — | 12.634 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | int2 | — | — | — | — | 24.171 | s32 | 7e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | int2 | — | — | — | — | 19.459 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:shift | int2 | — | — | — | — | 18.943 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | int2 | — | — | — | — | 0.257 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | int2 | — | — | — | — | 7.371 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | int2 | — | — | — | — | 44.028 | s32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | int2 | — | — | — | — | 33.679 | s8 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | int2 | — | — | — | — | 172.204 | s32 | 7e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | int2 | — | — | — | — | 14421.366 | s32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | int2 | 1024x1024x1024 Bᵀ | — | — | — | 1.588 | s8dp | 3.25e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | int2 | 1024x1024x1024 Aᵀ | — | — | — | 0.820 | s8dp | 3.24e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | int2 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 1.327 | s8dp | 3.2e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | int2 | 16 x 256x256x256 | — | — | — | 1.372 | s8dp | 9.7e+1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | int2 | 1024x1024x1024 (dA, dB) | — | — | — | 2.272 | s8dp | 3.21e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | int2 | — | — | — | — | 0.995 | s8dp | 2.04e+2 | from the item 4 run |
| cpu | topk_rows | int2 | — | — | — | — | 432.060 | s8 | 0e+0 | from the item 4 run |
| cpu | pick_rows | int2 | — | — | — | — | 0.033 | s8 | 0e+0 | from the item 4 run |
| cpu | add | uint64 | 16777216 | 7.053 | Vector<UInt64> | 0e+0 | 6.893 | u64 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | uint64 | 1024x1024x1024 | 33.098 | Vector<UInt64> | 0e+0 | 0.822 | u8dp | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | uint64 | — | — | — | — | 4.700 | u64 | 0e+0 | from the item 4 run |
| cpu | map:fill | uint64 | — | — | — | — | 2.626 | u64 | 0e+0 | from the item 4 run |
| cpu | map:neg | uint64 | — | — | — | — | 4.692 | u64 | 0e+0 | from the item 4 run |
| cpu | map:abs | uint64 | — | — | — | — | 4.689 | u64 | 0e+0 | from the item 4 run |
| cpu | map:sign | uint64 | — | — | — | — | 4.725 | u64 | 0e+0 | from the item 4 run |
| cpu | map:square | uint64 | — | — | — | — | 4.651 | u64 | 0e+0 | from the item 4 run |
| cpu | map:relu | uint64 | — | — | — | — | 4.712 | u64 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | uint64 | — | — | — | — | 4.641 | u64 | 0e+0 | from the item 4 run |
| cpu | map:scale | uint64 | — | — | — | — | 4.655 | u64 | 0e+0 | from the item 4 run |
| cpu | map:affine | uint64 | — | — | — | — | 4.798 | u64 | 0e+0 | from the item 4 run |
| cpu | map:clamp | uint64 | — | — | — | — | 4.864 | u64 | 0e+0 | from the item 4 run |
| cpu | map:add | uint64 | — | — | — | — | 6.711 | u64 | 0e+0 | from the item 4 run |
| cpu | map:sub | uint64 | — | — | — | — | 6.700 | u64 | 0e+0 | from the item 4 run |
| cpu | map:mul | uint64 | — | — | — | — | 6.751 | u64 | 0e+0 | from the item 4 run |
| cpu | map:div | uint64 | — | — | — | — | 6.578 | u64 | 0e+0 | from the item 4 run |
| cpu | map:max | uint64 | — | — | — | — | 6.701 | u64 | 0e+0 | from the item 4 run |
| cpu | map:min | uint64 | — | — | — | — | 6.869 | u64 | 0e+0 | from the item 4 run |
| cpu | map:axpy | uint64 | — | — | — | — | 6.708 | u64 | 0e+0 | from the item 4 run |
| cpu | map:muladd | uint64 | — | — | — | — | 8.460 | u64 | 0e+0 | from the item 4 run |
| cpu | map:where | uint64 | — | — | — | — | 8.400 | u64 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | uint64 | — | — | — | — | 6.772 | u64 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | uint64 | — | — | — | — | 6.846 | u64 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | uint64 | — | — | — | — | 6.909 | u64 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | uint64 | — | — | — | — | 6.834 | u64 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | uint64 | — | — | — | — | 6.754 | u64 | 0e+0 | from the item 4 run |
| cpu | sum:whole | uint64 | — | — | — | — | 2.660 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | uint64 | — | — | — | — | 2.580 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | uint64 | — | — | — | — | 7.290 | u8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | uint64 | — | — | — | — | 2.927 | u8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | uint64 | — | — | — | — | 1.408 | u64 | 0e+0 | from the item 4 run |
| cpu | max:rows | uint64 | — | — | — | — | 1.467 | u64 | 0e+0 | from the item 4 run |
| cpu | max:cols | uint64 | — | — | — | — | 4.790 | u64 | 0e+0 | from the item 4 run |
| cpu | min:rows | uint64 | — | — | — | — | 1.306 | u64 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | uint64 | — | — | — | — | 1.312 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | uint64 | — | — | — | — | 4.584 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | uint64 | — | — | — | — | 4.776 | u64 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | uint64 | — | — | — | — | 4.749 | u64 | 0e+0 | from the item 4 run |
| cpu | transpose | uint64 | — | — | — | — | 6.218 | u64 | 0e+0 | from the item 4 run |
| cpu | gemv | uint64 | — | — | — | — | 12.619 | u8dp | 0e+0 | from the item 4 run |
| cpu | gemv_t | uint64 | — | — | — | — | 17.178 | u8dp | 0e+0 | from the item 4 run |
| cpu | layout:strided | uint64 | — | — | — | — | 20.365 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | uint64 | — | — | — | — | 2.417 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | uint64 | — | — | — | — | 3.446 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | uint64 | — | — | — | — | 4.886 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | uint64 | — | — | — | — | 7.239 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | uint64 | — | — | — | — | 1.252 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | uint64 | — | — | — | — | 3.643 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | uint64 | — | — | — | — | 3.503 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:shift | uint64 | — | — | — | — | 3.947 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | uint64 | — | — | — | — | 3.683 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | uint64 | — | — | — | — | 1.797 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | uint64 | — | — | — | — | 11.450 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | uint64 | — | — | — | — | 8.614 | u64 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | uint64 | — | — | — | — | 44.786 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | uint64 | — | — | — | — | 2719.826 | u32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | uint64 | 1024x1024x1024 Bᵀ | — | — | — | 0.584 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | uint64 | 1024x1024x1024 Aᵀ | — | — | — | 0.683 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | uint64 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.819 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | uint64 | 16 x 256x256x256 | — | — | — | 1.140 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | uint64 | 1024x1024x1024 (dA, dB) | — | — | — | 1.959 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | uint64 | — | — | — | — | 1.471 | u8dp | 0e+0 | from the item 4 run |
| cpu | topk_rows | uint64 | — | — | — | — | 233.075 | u64 | 0e+0 | from the item 4 run |
| cpu | pick_rows | uint64 | — | — | — | — | 0.029 | u64 | 0e+0 | from the item 4 run |
| cpu | add | uint32 | 16777216 | 3.022 | Vector<UInt32> | 0e+0 | 3.014 | u32 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.386 | Vector<UInt32> | 0e+0 | 0.576 | u8dp | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | uint32 | — | — | — | — | 1.652 | u32 | 0e+0 | from the item 4 run |
| cpu | map:fill | uint32 | — | — | — | — | 0.409 | u32 | 0e+0 | from the item 4 run |
| cpu | map:neg | uint32 | — | — | — | — | 1.927 | u32 | 0e+0 | from the item 4 run |
| cpu | map:abs | uint32 | — | — | — | — | 1.887 | u32 | 0e+0 | from the item 4 run |
| cpu | map:sign | uint32 | — | — | — | — | 1.917 | u32 | 0e+0 | from the item 4 run |
| cpu | map:square | uint32 | — | — | — | — | 1.914 | u32 | 0e+0 | from the item 4 run |
| cpu | map:relu | uint32 | — | — | — | — | 1.961 | u32 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | uint32 | — | — | — | — | 1.912 | u32 | 0e+0 | from the item 4 run |
| cpu | map:scale | uint32 | — | — | — | — | 1.926 | u32 | 0e+0 | from the item 4 run |
| cpu | map:affine | uint32 | — | — | — | — | 1.919 | u32 | 0e+0 | from the item 4 run |
| cpu | map:clamp | uint32 | — | — | — | — | 1.915 | u32 | 0e+0 | from the item 4 run |
| cpu | map:add | uint32 | — | — | — | — | 2.999 | u32 | 0e+0 | from the item 4 run |
| cpu | map:sub | uint32 | — | — | — | — | 3.034 | u32 | 0e+0 | from the item 4 run |
| cpu | map:mul | uint32 | — | — | — | — | 3.098 | u32 | 0e+0 | from the item 4 run |
| cpu | map:div | uint32 | — | — | — | — | 2.906 | u32 | 0e+0 | from the item 4 run |
| cpu | map:max | uint32 | — | — | — | — | 3.170 | u32 | 0e+0 | from the item 4 run |
| cpu | map:min | uint32 | — | — | — | — | 2.943 | u32 | 0e+0 | from the item 4 run |
| cpu | map:axpy | uint32 | — | — | — | — | 3.057 | u32 | 0e+0 | from the item 4 run |
| cpu | map:muladd | uint32 | — | — | — | — | 4.096 | u32 | 0e+0 | from the item 4 run |
| cpu | map:where | uint32 | — | — | — | — | 4.097 | u32 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | uint32 | — | — | — | — | 2.911 | u32 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | uint32 | — | — | — | — | 2.856 | u32 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | uint32 | — | — | — | — | 2.939 | u32 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | uint32 | — | — | — | — | 2.944 | u32 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | uint32 | — | — | — | — | 2.922 | u32 | 0e+0 | from the item 4 run |
| cpu | sum:whole | uint32 | — | — | — | — | 0.761 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | uint32 | — | — | — | — | 0.773 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | uint32 | — | — | — | — | 1.555 | u8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | uint32 | — | — | — | — | 0.766 | u8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | uint32 | — | — | — | — | 0.381 | u32 | 0e+0 | from the item 4 run |
| cpu | max:rows | uint32 | — | — | — | — | 0.389 | u32 | 0e+0 | from the item 4 run |
| cpu | max:cols | uint32 | — | — | — | — | 2.264 | u32 | 0e+0 | from the item 4 run |
| cpu | min:rows | uint32 | — | — | — | — | 0.392 | u32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | uint32 | — | — | — | — | 0.376 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | uint32 | — | — | — | — | 5.300 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | uint32 | — | — | — | — | 1.723 | u32 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | uint32 | — | — | — | — | 1.679 | u32 | 0e+0 | from the item 4 run |
| cpu | transpose | uint32 | — | — | — | — | 2.866 | u32 | 0e+0 | from the item 4 run |
| cpu | gemv | uint32 | — | — | — | — | 5.891 | u8dp | 0e+0 | from the item 4 run |
| cpu | gemv_t | uint32 | — | — | — | — | 9.020 | u8dp | 0e+0 | from the item 4 run |
| cpu | layout:strided | uint32 | — | — | — | — | 15.764 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | uint32 | — | — | — | — | 0.776 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | uint32 | — | — | — | — | 1.475 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | uint32 | — | — | — | — | 2.823 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | uint32 | — | — | — | — | 3.211 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | uint32 | — | — | — | — | 1.029 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | uint32 | — | — | — | — | 2.006 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | uint32 | — | — | — | — | 1.807 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:shift | uint32 | — | — | — | — | 2.140 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | uint32 | — | — | — | — | 1.683 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | uint32 | — | — | — | — | 1.715 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | uint32 | — | — | — | — | 6.867 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | uint32 | — | — | — | — | 9.390 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | uint32 | — | — | — | — | 5.753 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | uint32 | — | — | — | — | 1423.310 | u32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | uint32 | 1024x1024x1024 Bᵀ | — | — | — | 0.454 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | uint32 | 1024x1024x1024 Aᵀ | — | — | — | 0.593 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | uint32 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.965 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | uint32 | 16 x 256x256x256 | — | — | — | 0.503 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | uint32 | 1024x1024x1024 (dA, dB) | — | — | — | 1.371 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | uint32 | — | — | — | — | 0.984 | u8dp | 0e+0 | from the item 4 run |
| cpu | topk_rows | uint32 | — | — | — | — | 332.775 | u32 | 0e+0 | from the item 4 run |
| cpu | pick_rows | uint32 | — | — | — | — | 0.027 | u32 | 0e+0 | from the item 4 run |
| cpu | add | uint16 | 16777216 | 1.241 | Vector<UInt16> | 0e+0 | 0.861 | u16 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | uint16 | 1024x1024x1024 | 2.004 | Vector<UInt16> | 0e+0 | 0.471 | u8dp | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | uint16 | — | — | — | — | 0.320 | u16 | 0e+0 | from the item 4 run |
| cpu | map:fill | uint16 | — | — | — | — | 0.129 | u16 | 0e+0 | from the item 4 run |
| cpu | map:neg | uint16 | — | — | — | — | 0.265 | u16 | 0e+0 | from the item 4 run |
| cpu | map:abs | uint16 | — | — | — | — | 0.265 | u16 | 0e+0 | from the item 4 run |
| cpu | map:sign | uint16 | — | — | — | — | 0.268 | u16 | 0e+0 | from the item 4 run |
| cpu | map:square | uint16 | — | — | — | — | 0.310 | u16 | 0e+0 | from the item 4 run |
| cpu | map:relu | uint16 | — | — | — | — | 0.296 | u16 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | uint16 | — | — | — | — | 0.281 | u16 | 0e+0 | from the item 4 run |
| cpu | map:scale | uint16 | — | — | — | — | 0.264 | u16 | 0e+0 | from the item 4 run |
| cpu | map:affine | uint16 | — | — | — | — | 0.277 | u16 | 0e+0 | from the item 4 run |
| cpu | map:clamp | uint16 | — | — | — | — | 0.276 | u16 | 0e+0 | from the item 4 run |
| cpu | map:add | uint16 | — | — | — | — | 0.767 | u16 | 0e+0 | from the item 4 run |
| cpu | map:sub | uint16 | — | — | — | — | 0.823 | u16 | 0e+0 | from the item 4 run |
| cpu | map:mul | uint16 | — | — | — | — | 0.842 | u16 | 0e+0 | from the item 4 run |
| cpu | map:div | uint16 | — | — | — | — | 1.405 | u16 | 0e+0 | from the item 4 run |
| cpu | map:max | uint16 | — | — | — | — | 0.801 | u16 | 0e+0 | from the item 4 run |
| cpu | map:min | uint16 | — | — | — | — | 0.799 | u16 | 0e+0 | from the item 4 run |
| cpu | map:axpy | uint16 | — | — | — | — | 0.807 | u16 | 0e+0 | from the item 4 run |
| cpu | map:muladd | uint16 | — | — | — | — | 1.366 | u16 | 0e+0 | from the item 4 run |
| cpu | map:where | uint16 | — | — | — | — | 1.340 | u16 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | uint16 | — | — | — | — | 0.795 | u16 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | uint16 | — | — | — | — | 0.771 | u16 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | uint16 | — | — | — | — | 0.862 | u16 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | uint16 | — | — | — | — | 0.859 | u16 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | uint16 | — | — | — | — | 0.849 | u16 | 0e+0 | from the item 4 run |
| cpu | sum:whole | uint16 | — | — | — | — | 0.257 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | uint16 | — | — | — | — | 0.272 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | uint16 | — | — | — | — | 0.878 | u8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | uint16 | — | — | — | — | 0.268 | u8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | uint16 | — | — | — | — | 0.127 | u16 | 0e+0 | from the item 4 run |
| cpu | max:rows | uint16 | — | — | — | — | 0.137 | u16 | 0e+0 | from the item 4 run |
| cpu | max:cols | uint16 | — | — | — | — | 1.728 | u16 | 0e+0 | from the item 4 run |
| cpu | min:rows | uint16 | — | — | — | — | 0.134 | u16 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | uint16 | — | — | — | — | 0.135 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | uint16 | — | — | — | — | 5.874 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | uint16 | — | — | — | — | 0.354 | u16 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | uint16 | — | — | — | — | 0.301 | u16 | 0e+0 | from the item 4 run |
| cpu | transpose | uint16 | — | — | — | — | 0.913 | u16 | 0e+0 | from the item 4 run |
| cpu | gemv | uint16 | — | — | — | — | 2.537 | u8dp | 0e+0 | from the item 4 run |
| cpu | gemv_t | uint16 | — | — | — | — | 5.186 | u8dp | 0e+0 | from the item 4 run |
| cpu | layout:strided | uint16 | — | — | — | — | 6.030 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | uint16 | — | — | — | — | 0.127 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | uint16 | — | — | — | — | 0.329 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | uint16 | — | — | — | — | 0.873 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | uint16 | — | — | — | — | 3.605 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | uint16 | — | — | — | — | 0.946 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | uint16 | — | — | — | — | 3.225 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | uint16 | — | — | — | — | 2.136 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:shift | uint16 | — | — | — | — | 1.146 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | uint16 | — | — | — | — | 0.441 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | uint16 | — | — | — | — | 2.117 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | uint16 | — | — | — | — | 11.919 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | uint16 | — | — | — | — | 15.349 | u16 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | uint16 | — | — | — | — | 16.180 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | uint16 | — | — | — | — | 3633.520 | u32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | uint16 | 1024x1024x1024 Bᵀ | — | — | — | 0.510 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | uint16 | 1024x1024x1024 Aᵀ | — | — | — | 0.548 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | uint16 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.645 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | uint16 | 16 x 256x256x256 | — | — | — | 0.428 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | uint16 | 1024x1024x1024 (dA, dB) | — | — | — | 1.277 | u8dp | 0e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | uint16 | — | — | — | — | 0.919 | u8dp | 0e+0 | from the item 4 run |
| cpu | topk_rows | uint16 | — | — | — | — | 357.098 | u16 | 0e+0 | from the item 4 run |
| cpu | pick_rows | uint16 | — | — | — | — | 0.027 | u16 | 0e+0 | from the item 4 run |
| cpu | add | uint8 | 16777216 | 0.478 | Vector<Byte> | 0e+0 | 0.127 | u8 | 0e+0 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | uint8 | 1024x1024x1024 | 4.583 | Vector<Byte> | 2.56e+3 | 0.546 | u8dp | 2.48e+3 | number: from the item 4 run; qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | uint8 | — | — | — | — | 0.084 | u8 | 0e+0 | from the item 4 run |
| cpu | map:fill | uint8 | — | — | — | — | 0.083 | u8 | 0e+0 | from the item 4 run |
| cpu | map:neg | uint8 | — | — | — | — | 0.120 | u8 | 0e+0 | from the item 4 run |
| cpu | map:abs | uint8 | — | — | — | — | 0.100 | u8 | 0e+0 | from the item 4 run |
| cpu | map:sign | uint8 | — | — | — | — | 0.109 | u8 | 0e+0 | from the item 4 run |
| cpu | map:square | uint8 | — | — | — | — | 0.123 | u8 | 0e+0 | from the item 4 run |
| cpu | map:relu | uint8 | — | — | — | — | 0.121 | u8 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | uint8 | — | — | — | — | 0.116 | u8 | 0e+0 | from the item 4 run |
| cpu | map:scale | uint8 | — | — | — | — | 0.127 | u8 | 0e+0 | from the item 4 run |
| cpu | map:affine | uint8 | — | — | — | — | 0.125 | u8 | 0e+0 | from the item 4 run |
| cpu | map:clamp | uint8 | — | — | — | — | 0.103 | u8 | 0e+0 | from the item 4 run |
| cpu | map:add | uint8 | — | — | — | — | 0.240 | u8 | 0e+0 | from the item 4 run |
| cpu | map:sub | uint8 | — | — | — | — | 0.159 | u8 | 0e+0 | from the item 4 run |
| cpu | map:mul | uint8 | — | — | — | — | 0.210 | u8 | 0e+0 | from the item 4 run |
| cpu | map:div | uint8 | — | — | — | — | 1.181 | u8 | 0e+0 | from the item 4 run |
| cpu | map:max | uint8 | — | — | — | — | 0.237 | u8 | 0e+0 | from the item 4 run |
| cpu | map:min | uint8 | — | — | — | — | 0.162 | u8 | 0e+0 | from the item 4 run |
| cpu | map:axpy | uint8 | — | — | — | — | 0.151 | u8 | 0e+0 | from the item 4 run |
| cpu | map:muladd | uint8 | — | — | — | — | 0.304 | u8 | 0e+0 | from the item 4 run |
| cpu | map:where | uint8 | — | — | — | — | 0.285 | u8 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | uint8 | — | — | — | — | 0.144 | u8 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | uint8 | — | — | — | — | 0.139 | u8 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | uint8 | — | — | — | — | 0.129 | u8 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | uint8 | — | — | — | — | 0.156 | u8 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | uint8 | — | — | — | — | 0.132 | u8 | 0e+0 | from the item 4 run |
| cpu | sum:whole | uint8 | — | — | — | — | 0.075 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | uint8 | — | — | — | — | 0.074 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | uint8 | — | — | — | — | 0.304 | u8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | uint8 | — | — | — | — | 0.072 | u8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | uint8 | — | — | — | — | 0.037 | u8 | 0e+0 | from the item 4 run |
| cpu | max:rows | uint8 | — | — | — | — | 0.054 | u8 | 0e+0 | from the item 4 run |
| cpu | max:cols | uint8 | — | — | — | — | 1.489 | u8 | 0e+0 | from the item 4 run |
| cpu | min:rows | uint8 | — | — | — | — | 0.059 | u8 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | uint8 | — | — | — | — | 0.055 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | uint8 | — | — | — | — | 4.369 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | uint8 | — | — | — | — | 0.086 | u8 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | uint8 | — | — | — | — | 0.144 | u8 | 0e+0 | from the item 4 run |
| cpu | transpose | uint8 | — | — | — | — | 0.759 | u8 | 0e+0 | from the item 4 run |
| cpu | gemv | uint8 | — | — | — | — | 0.895 | u8dp | 1.9e+4 | from the item 4 run |
| cpu | gemv_t | uint8 | — | — | — | — | 3.234 | u8dp | 1.91e+4 | from the item 4 run |
| cpu | layout:strided | uint8 | — | — | — | — | 4.209 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | uint8 | — | — | — | — | 0.075 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | uint8 | — | — | — | — | 0.115 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | uint8 | — | — | — | — | 0.748 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | uint8 | — | — | — | — | 5.003 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:im2col | uint8 | — | — | — | — | 1.065 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | uint8 | — | — | — | — | 3.506 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | uint8 | — | — | — | — | 1.609 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:shift | uint8 | — | — | — | — | 0.861 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | uint8 | — | — | — | — | 0.094 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | uint8 | — | — | — | — | 1.865 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | uint8 | — | — | — | — | 11.661 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | uint8 | — | — | — | — | 14.468 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | uint8 | — | — | — | — | 22.209 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | uint8 | — | — | — | — | 2542.219 | u32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | uint8 | 1024x1024x1024 Bᵀ | — | — | — | 0.508 | u8dp | 2.48e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | uint8 | 1024x1024x1024 Aᵀ | — | — | — | 0.404 | u8dp | 2.43e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | uint8 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 0.399 | u8dp | 2.48e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | uint8 | 16 x 256x256x256 | — | — | — | 0.337 | u8dp | 5.3e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | uint8 | 1024x1024x1024 (dA, dB) | — | — | — | 1.128 | u8dp | 2.49e+3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | uint8 | — | — | — | — | 0.866 | u8dp | 1.35e+3 | from the item 4 run |
| cpu | topk_rows | uint8 | — | — | — | — | 296.490 | u8 | 0e+0 | from the item 4 run |
| cpu | pick_rows | uint8 | — | — | — | — | 0.024 | u8 | 0e+0 | from the item 4 run |
| cpu | add | uint2 | 16777216 | — | — | — | 0.242 | u8 | 0e+0 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | uint2 | 1024x1024x1024 | — | — | — | 0.477 | u8dp | 3.18e+2 | qiyas: end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | uint2 | — | — | — | — | 0.360 | u8 | 0e+0 | from the item 4 run |
| cpu | map:fill | uint2 | — | — | — | — | 0.322 | u8 | 0e+0 | from the item 4 run |
| cpu | map:neg | uint2 | — | — | — | — | 0.335 | u8 | 0e+0 | from the item 4 run |
| cpu | map:abs | uint2 | — | — | — | — | 0.264 | u8 | 0e+0 | from the item 4 run |
| cpu | map:sign | uint2 | — | — | — | — | 0.286 | u8 | 0e+0 | from the item 4 run |
| cpu | map:square | uint2 | — | — | — | — | 0.283 | u8 | 0e+0 | from the item 4 run |
| cpu | map:relu | uint2 | — | — | — | — | 0.286 | u8 | 0e+0 | from the item 4 run |
| cpu | map:add_scalar | uint2 | — | — | — | — | 0.291 | u8 | 0e+0 | from the item 4 run |
| cpu | map:scale | uint2 | — | — | — | — | 0.286 | u8 | 0e+0 | from the item 4 run |
| cpu | map:affine | uint2 | — | — | — | — | 0.293 | u8 | 0e+0 | from the item 4 run |
| cpu | map:clamp | uint2 | — | — | — | — | 0.271 | u8 | 0e+0 | from the item 4 run |
| cpu | map:add | uint2 | — | — | — | — | 0.305 | u8 | 0e+0 | from the item 4 run |
| cpu | map:sub | uint2 | — | — | — | — | 0.300 | u8 | 0e+0 | from the item 4 run |
| cpu | map:mul | uint2 | — | — | — | — | 0.313 | u8 | 0e+0 | from the item 4 run |
| cpu | map:div | uint2 | — | — | — | — | 1.326 | u8 | 0e+0 | from the item 4 run |
| cpu | map:max | uint2 | — | — | — | — | 0.315 | u8 | 0e+0 | from the item 4 run |
| cpu | map:min | uint2 | — | — | — | — | 0.313 | u8 | 0e+0 | from the item 4 run |
| cpu | map:axpy | uint2 | — | — | — | — | 0.310 | u8 | 0e+0 | from the item 4 run |
| cpu | map:muladd | uint2 | — | — | — | — | 0.367 | u8 | 0e+0 | from the item 4 run |
| cpu | map:where | uint2 | — | — | — | — | 0.481 | u8 | 0e+0 | from the item 4 run |
| cpu | map:square_bwd | uint2 | — | — | — | — | 0.502 | u8 | 0e+0 | from the item 4 run |
| cpu | map:abs_bwd | uint2 | — | — | — | — | 0.477 | u8 | 0e+0 | from the item 4 run |
| cpu | map:relu_bwd | uint2 | — | — | — | — | 0.396 | u8 | 0e+0 | from the item 4 run |
| cpu | map:clamp_bwd | uint2 | — | — | — | — | 0.399 | u8 | 0e+0 | from the item 4 run |
| cpu | map:where_bwd | uint2 | — | — | — | — | 0.413 | u8 | 0e+0 | from the item 4 run |
| cpu | sum:whole | uint2 | — | — | — | — | 0.085 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:rows | uint2 | — | — | — | — | 0.086 | u8dp | 0e+0 | from the item 4 run |
| cpu | sum:cols | uint2 | — | — | — | — | 0.639 | u8dp | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | uint2 | — | — | — | — | 0.065 | u8dp | 0e+0 | from the item 4 run |
| cpu | max:whole | uint2 | — | — | — | — | 0.095 | u8 | 0e+0 | from the item 4 run |
| cpu | max:rows | uint2 | — | — | — | — | 0.219 | u8 | 0e+0 | from the item 4 run |
| cpu | max:cols | uint2 | — | — | — | — | 1.804 | u8 | 0e+0 | from the item 4 run |
| cpu | min:rows | uint2 | — | — | — | — | 0.219 | u8 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | uint2 | — | — | — | — | 0.218 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | uint2 | — | — | — | — | 6.872 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | uint2 | — | — | — | — | 0.979 | u8 | 0e+0 | from the item 4 run |
| cpu | maprow:mul | uint2 | — | — | — | — | 0.242 | u8 | 0e+0 | from the item 4 run |
| cpu | transpose | uint2 | — | — | — | — | 3.440 | u8 | 0e+0 | from the item 4 run |
| cpu | gemv | uint2 | — | — | — | — | 0.405 | u8dp | 2.19e+3 | from the item 4 run |
| cpu | gemv_t | uint2 | — | — | — | — | 3.400 | u8dp | 2.19e+3 | from the item 4 run |
| cpu | layout:strided | uint2 | — | — | — | — | 5.592 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:one_hot | uint2 | — | — | — | — | 0.112 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | uint2 | — | — | — | — | 0.160 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | uint2 | — | — | — | — | 6.425 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:scatter_add_rows | uint2 | — | — | — | — | 35.422 | u32 | 2e+0 | from the item 4 run |
| cpu | layout:im2col | uint2 | — | — | — | — | 6.015 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:col2im | uint2 | — | — | — | — | 14.867 | u32 | 6e+0 | from the item 4 run |
| cpu | layout:interpolate_nearest | uint2 | — | — | — | — | 11.109 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:shift | uint2 | — | — | — | — | 10.371 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:pad2d | uint2 | — | — | — | — | 0.200 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool | uint2 | — | — | — | — | 3.692 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:maxpool_bwd | uint2 | — | — | — | — | 48.865 | u32 | 0e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool | uint2 | — | — | — | — | 37.810 | u8 | 0e+0 | from the item 4 run |
| cpu | layout:depthwise_conv | uint2 | — | — | — | — | 169.680 | u32 | 4e+0 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | uint2 | — | — | — | — | 12629.176 | u32 | 0e+0 | from the item 4 run |
| cpu | matmul_nt | uint2 | 1024x1024x1024 Bᵀ | — | — | — | 1.593 | u8dp | 3.23e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | uint2 | 1024x1024x1024 Aᵀ | — | — | — | 1.195 | u8dp | 3.19e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | uint2 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 1.317 | u8dp | 3.27e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | uint2 | 16 x 256x256x256 | — | — | — | 1.475 | u8dp | 9.6e+1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | uint2 | 1024x1024x1024 (dA, dB) | — | — | — | 2.131 | u8dp | 3.22e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | conv | uint2 | — | — | — | — | 0.894 | u8dp | 1.93e+2 | from the item 4 run |
| cpu | topk_rows | uint2 | — | — | — | — | 430.433 | u8 | 0e+0 | from the item 4 run |
| cpu | pick_rows | uint2 | — | — | — | — | 0.029 | u8 | 0e+0 | from the item 4 run |
| cpu | add | q8_0 | 16777216 | — | — | — | 5.194 | f32 | 7.87e-3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | q8_0 | 1024x1024x1024 | — | — | — | 8.913 | f32 | 2.03e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | q8_0 | — | — | — | — | 5.508 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | q8_0 | — | — | — | — | 4.828 | f32 | 1.83e-4 | from the item 4 run |
| cpu | map:neg | q8_0 | — | — | — | — | 5.532 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | q8_0 | — | — | — | — | 5.517 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | q8_0 | — | — | — | — | 5.666 | f32 | 6.1e-5 | from the item 4 run |
| cpu | map:square | q8_0 | — | — | — | — | 4.610 | f32 | 6.26e-2 | from the item 4 run |
| cpu | map:relu | q8_0 | — | — | — | — | 4.644 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:add_scalar | q8_0 | — | — | — | — | 4.701 | f32 | 2.36e-2 | from the item 4 run |
| cpu | map:scale | q8_0 | — | — | — | — | 5.531 | f32 | 3.88e-3 | from the item 4 run |
| cpu | map:affine | q8_0 | — | — | — | — | 5.573 | f32 | 3.53e-2 | from the item 4 run |
| cpu | map:clamp | q8_0 | — | — | — | — | 6.088 | f32 | 7.87e-3 | from the item 4 run |
| cpu | map:add | q8_0 | — | — | — | — | 5.189 | f32 | 3.15e-2 | from the item 4 run |
| cpu | map:sub | q8_0 | — | — | — | — | 6.264 | f32 | 3.15e-2 | from the item 4 run |
| cpu | map:mul | q8_0 | — | — | — | — | 6.293 | f32 | 6.29e-2 | from the item 4 run |
| cpu | map:div | q8_0 | — | — | — | — | 6.557 | f32 | 1.59e-2 | from the item 4 run |
| cpu | map:max | q8_0 | — | — | — | — | 6.454 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:min | q8_0 | — | — | — | — | 5.252 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:axpy | q8_0 | — | — | — | — | 5.179 | f32 | 4.71e-2 | from the item 4 run |
| cpu | map:muladd | q8_0 | — | — | — | — | 5.787 | f32 | 7.85e-2 | from the item 4 run |
| cpu | map:where | q8_0 | — | — | — | — | 5.814 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:square_bwd | q8_0 | — | — | — | — | 5.198 | f32 | 1.26e-1 | from the item 4 run |
| cpu | map:abs_bwd | q8_0 | — | — | — | — | 5.269 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:relu_bwd | q8_0 | — | — | — | — | 5.205 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:clamp_bwd | q8_0 | — | — | — | — | 6.213 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:where_bwd | q8_0 | — | — | — | — | 5.224 | f32 | 1.56e-2 | from the item 4 run |
| cpu | map:sqrt | q8_0 | — | — | — | — | 5.781 | f32 | 7.83e-3 | from the item 4 run |
| cpu | map:inv_sqrt | q8_0 | — | — | — | — | 6.219 | f32 | 5.64e-3 | from the item 4 run |
| cpu | map:exp | q8_0 | — | — | — | — | 7.526 | f32 | 2.14e-1 | from the item 4 run |
| cpu | map:expm1 | q8_0 | — | — | — | — | 7.356 | f32 | 2.54e-3 | from the item 4 run |
| cpu | map:log | q8_0 | — | — | — | — | 7.394 | f32 | 5.45e-3 | from the item 4 run |
| cpu | map:sin | q8_0 | — | — | — | — | 6.818 | f32 | 3.94e-3 | from the item 4 run |
| cpu | map:cos | q8_0 | — | — | — | — | 8.020 | f32 | 3.94e-3 | from the item 4 run |
| cpu | map:tanh | q8_0 | — | — | — | — | 8.263 | f32 | 3.93e-3 | from the item 4 run |
| cpu | map:sigmoid | q8_0 | — | — | — | — | 7.321 | f32 | 3.86e-3 | from the item 4 run |
| cpu | map:erf | q8_0 | — | — | — | — | 30.789 | f32 | 3.94e-3 | from the item 4 run |
| cpu | map:gelu | q8_0 | — | — | — | — | 7.965 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:gelu_erf | q8_0 | — | — | — | — | 31.950 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:silu | q8_0 | — | — | — | — | 7.406 | f32 | 1.54e-2 | from the item 4 run |
| cpu | map:leaky_relu | q8_0 | — | — | — | — | 5.674 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:pow | q8_0 | — | — | — | — | 11.154 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:dropout | q8_0 | — | — | — | — | 6.323 | f32 | 1.95e-2 | from the item 4 run |
| cpu | map:gated_silu | q8_0 | — | — | — | — | 6.766 | f32 | 6.16e-2 | from the item 4 run |
| cpu | map:gated_gelu | q8_0 | — | — | — | — | 6.997 | f32 | 6.29e-2 | from the item 4 run |
| cpu | map:sqrt_bwd | q8_0 | — | — | — | — | 5.625 | f32 | 1.12e-2 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | q8_0 | — | — | — | — | 5.874 | f32 | 2.28e-2 | from the item 4 run |
| cpu | map:exp_bwd | q8_0 | — | — | — | — | 6.629 | f32 | 8.57e-1 | from the item 4 run |
| cpu | map:log_bwd | q8_0 | — | — | — | — | 5.339 | f32 | 3.22e-2 | from the item 4 run |
| cpu | map:sin_bwd | q8_0 | — | — | — | — | 7.297 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:cos_bwd | q8_0 | — | — | — | — | 7.453 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:tanh_bwd | q8_0 | — | — | — | — | 7.598 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:sigmoid_bwd | q8_0 | — | — | — | — | 8.416 | f32 | 3.93e-3 | from the item 4 run |
| cpu | map:silu_bwd | q8_0 | — | — | — | — | 7.994 | f32 | 1.73e-2 | from the item 4 run |
| cpu | map:gelu_bwd | q8_0 | — | — | — | — | 7.837 | f32 | 1.78e-2 | from the item 4 run |
| cpu | map:gelu_erf_bwd | q8_0 | — | — | — | — | 32.745 | f32 | 1.78e-2 | from the item 4 run |
| cpu | map:leaky_relu_bwd | q8_0 | — | — | — | — | 5.222 | f32 | 1.57e-2 | from the item 4 run |
| cpu | map:pow_bwd | q8_0 | — | — | — | — | 8.792 | f32 | 3.15e-1 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | q8_0 | — | — | — | — | 8.418 | f32 | 6.63e-2 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | q8_0 | — | — | — | — | 8.946 | f32 | 6.17e-2 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | q8_0 | — | — | — | — | 8.527 | f32 | 6.28e-2 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | q8_0 | — | — | — | — | 7.585 | f32 | 6.29e-2 | from the item 4 run |
| cpu | sum:whole | q8_0 | — | — | — | — | 56.213 | f32 | 7.78e-4 | from the item 4 run |
| cpu | sum:rows | q8_0 | — | — | — | — | 55.945 | f32 | 7.63e-6 | from the item 4 run |
| cpu | sum:cols | q8_0 | — | — | — | — | 56.940 | f32 | 2.48e-4 | from the item 4 run |
| cpu | sumsq:rows | q8_0 | — | — | — | — | 56.259 | f32 | 2.43e-4 | from the item 4 run |
| cpu | max:whole | q8_0 | — | — | — | — | 0.958 | f32 | 0e+0 | from the item 4 run |
| cpu | max:rows | q8_0 | — | — | — | — | 2.207 | f32 | 0e+0 | from the item 4 run |
| cpu | max:cols | q8_0 | — | — | — | — | 3.528 | f32 | 0e+0 | from the item 4 run |
| cpu | min:rows | q8_0 | — | — | — | — | 2.359 | f32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | q8_0 | — | — | — | — | 1.969 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | q8_0 | — | — | — | — | 2.010 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | q8_0 | — | — | — | — | 6.302 | f32 | 7.87e-3 | from the item 4 run |
| cpu | maprow:mul | q8_0 | — | — | — | — | 6.276 | f32 | 3.93e-3 | from the item 4 run |
| cpu | transpose | q8_0 | — | — | — | — | 7.193 | f32 | 3.94e-3 | from the item 4 run |
| cpu | gemv | q8_0 | — | — | — | — | 2.011 | s8dp blocks | 4.75e-1 | from the item 4 run |
| cpu | gemv_t | q8_0 | — | — | — | — | 228.393 | f32 | 5.89e-1 | from the item 4 run |
| cpu | row:log_softmax | q8_0 | — | — | — | — | 6.726 | f32 | 5.63e-2 | from the item 4 run |
| cpu | row:softmax_bwd | q8_0 | — | — | — | — | 5.229 | f32 | 3.11e-1 | from the item 4 run |
| cpu | row:log_softmax_bwd | q8_0 | — | — | — | — | 6.454 | f32 | 5.06e-1 | from the item 4 run |
| cpu | row:rms_norm_affine | q8_0 | — | — | — | — | 4.626 | f32 | 1.38e-2 | from the item 4 run |
| cpu | row:rms_norm_bwd | q8_0 | — | — | — | — | 5.302 | f32 | 3.46e-3 | from the item 4 run |
| cpu | row:layer_norm | q8_0 | — | — | — | — | 4.663 | f32 | 1.75e-2 | from the item 4 run |
| cpu | row:layer_norm_bwd | q8_0 | — | — | — | — | 5.405 | f32 | 3.54e-3 | from the item 4 run |
| cpu | row:weight_norm | q8_0 | — | — | — | — | 4.667 | f32 | 1.51e-5 | from the item 4 run |
| cpu | row:rope | q8_0 | — | — | — | — | 7.729 | f32 | 2.23e-2 | from the item 4 run |
| cpu | row:scale_mask_softmax | q8_0 | — | — | — | — | 7.029 | f32 | 1.42e-3 | from the item 4 run |
| cpu | row:add_rms_norm | q8_0 | — | — | — | — | 9.161 | f32 | 1.94e-2 | from the item 4 run |
| cpu | row:add_layer_norm | q8_0 | — | — | — | — | 9.259 | f32 | 2.26e-2 | from the item 4 run |
| cpu | row:cross_entropy_bwd | q8_0 | — | — | — | — | 6.051 | f32 | 2.08e-3 | from the item 4 run |
| cpu | norm_params_bwd:rms | q8_0 | — | — | — | — | 3.847 | f32 | 3.08e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | q8_0 | — | — | — | — | 3.797 | f32 | 2.54e-4 | from the item 4 run |
| cpu | layout:strided | q8_0 | — | — | — | — | 42.040 | f32 | 3.94e-3 | from the item 4 run |
| cpu | layout:one_hot | q8_0 | — | — | — | — | 3.743 | f32 | 6.1e-5 | from the item 4 run |
| cpu | layout:gather_rows | q8_0 | — | — | — | — | 5.298 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | q8_0 | — | — | — | — | 30.450 | f32 | 3.94e-3 | from the item 4 run |
| cpu | layout:scatter_add_rows | q8_0 | — | — | — | — | 84.836 | f32 | 1.68e-2 | from the item 4 run |
| cpu | layout:im2col | q8_0 | — | — | — | — | 20.979 | f32 | 3.93e-3 | from the item 4 run |
| cpu | layout:col2im | q8_0 | — | — | — | — | 52.140 | f32 | 2.95e-2 | from the item 4 run |
| cpu | layout:interpolate_nearest | q8_0 | — | — | — | — | 32.560 | f32 | 3.91e-3 | from the item 4 run |
| cpu | layout:interpolate_bilinear | q8_0 | — | — | — | — | 149.004 | f32 | 3.92e-3 | from the item 4 run |
| cpu | layout:shift | q8_0 | — | — | — | — | 30.118 | f32 | 3.94e-3 | from the item 4 run |
| cpu | layout:pad2d | q8_0 | — | — | — | — | 4.447 | f32 | 3.94e-3 | from the item 4 run |
| cpu | layout:maxpool | q8_0 | — | — | — | — | 8.020 | f32 | 3.94e-3 | from the item 4 run |
| cpu | layout:avgpool | q8_0 | — | — | — | — | 9.535 | f32 | 3.79e-3 | from the item 4 run |
| cpu | layout:maxpool_bwd | q8_0 | — | — | — | — | 57.820 | f32 | 3.91e-3 | from the item 4 run |
| cpu | layout:avgpool_bwd | q8_0 | — | — | — | — | 11.419 | f32 | 9.77e-4 | from the item 4 run |
| cpu | layout:adaptive_maxpool | q8_0 | — | — | — | — | 147.010 | f32 | 3.88e-3 | from the item 4 run |
| cpu | layout:adaptive_avgpool | q8_0 | — | — | — | — | 145.586 | f32 | 2.32e-4 | from the item 4 run |
| cpu | layout:depthwise_conv | q8_0 | — | — | — | — | 171.131 | f32 | 1.95e-2 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | q8_0 | — | — | — | — | 13738.767 | f32 | 2.85e-3 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | q8_0 | — | — | — | — | 18.404 | f32 | 7.6e-6 | from the item 4 run |
| cpu | matmul_nt | q8_0 | 1024x1024x1024 Bᵀ | — | — | — | 2.333 | f32 | 2.08e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | q8_0 | 1024x1024x1024 Aᵀ | — | — | — | 2.603 | f32 | 2.22e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | q8_0 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 2.642 | f32 | 1.93e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | q8_0 | 16 x 256x256x256 | — | — | — | 1.484 | f32 | 9.95e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | q8_0 | 1024x1024x1024 (dA, dB) | — | — | — | 4.774 | f32 | 2.17e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_blocks | q8_0 | 1024x1024x1024 Bᵀ | — | — | — | 2.725 | s8dp blocks | 2.08e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | q8_0 | — | — | — | — | 64.650 | f32 | 2.44e-4 | from the item 4 run |
| cpu | attention_causal | q8_0 | — | — | — | — | 44.849 | f32 | 3.73e-3 | from the item 4 run |
| cpu | attention_fused | q8_0 | — | — | — | — | 81.537 | f32 | 2.44e-4 | from the item 4 run |
| cpu | attention_fused_causal | q8_0 | — | — | — | — | 45.468 | f32 | 3.73e-3 | from the item 4 run |
| cpu | attention_decode | q8_0 | — | — | — | — | 2.101 | f32 | 1.39e-4 | from the item 4 run |
| cpu | attention_bwd | q8_0 | — | — | — | — | 17.682 | f32 | 1.56e-3 | from the item 4 run |
| cpu | optimizer:sgd_momentum | q8_0 | — | — | — | — | 9.079 | f32; + restore of p, m, v | 3.34e-3 | from the item 4 run |
| cpu | optimizer:adam | q8_0 | — | — | — | — | 12.640 | f32; + restore of p, m, v | 3.92e-3 | from the item 4 run |
| cpu | optimizer:adamw | q8_0 | — | — | — | — | 16.081 | f32; + restore of p, m, v | 3.93e-3 | from the item 4 run |
| cpu | conv | q8_0 | — | — | — | — | 7.825 | f32 | 1.67e-1 | from the item 4 run |
| cpu | conv_bwd | q8_0 | — | — | — | — | 20.194 | f32 | 2.23e-1 | from the item 4 run |
| cpu | lstm_step | q8_0 | — | — | — | — | 26.065 | f32 | 2.12e-3 | from the item 4 run |
| cpu | gru_step | q8_0 | — | — | — | — | 19.956 | f32 | 3.58e-3 | from the item 4 run |
| cpu | lstm_step_bwd | q8_0 | — | — | — | — | 60.820 | f32 | 2.42e-3 | from the item 4 run |
| cpu | gru_step_bwd | q8_0 | — | — | — | — | 46.274 | f32 | 2.33e-3 | from the item 4 run |
| cpu | gru_seq | q8_0 | — | — | — | — | 31.337 | f32 | 3.34e-3 | from the item 4 run |
| cpu | gru_seq_bwd | q8_0 | — | — | — | — | 116.469 | f32 | 2.64e-3 | from the item 4 run |
| cpu | topk_rows | q8_0 | — | — | — | — | 278.453 | f32 | 3.88e-3 | from the item 4 run |
| cpu | pick_rows | q8_0 | — | — | — | — | 0.193 | f32 | 1.57e-2 | from the item 4 run |
| cpu | sample_rows | q8_0 | — | — | — | — | 11.287 | f32 | 2e+0 | from the item 4 run |
| cpu | cross_entropy | q8_0 | — | — | — | — | 59.806 | f32 (log_softmax, pick, neg) | 1.25e-1 | from the item 4 run |
| cpu | ctc | q8_0 | — | — | — | — | 10.526 | f32 | 6.44e+0 | from the item 4 run |
| cpu | ctc_bwd | q8_0 | — | — | — | — | 11.706 | f32 | 3.91e-3 | from the item 4 run |
| cpu | softmax | q8_0 | 4096x4096 | — | — | — | 6.232 | f32 | 8.37e-6 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | q8_0 | 4096x4096 | — | — | — | 6.215 | f32 | 4.85e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | q4_0 | 16777216 | — | — | — | 4.253 | f32 | 2.34e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | q4_0 | 1024x1024x1024 | — | — | — | 8.700 | f32 | 4.06e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | q4_0 | — | — | — | — | 2.718 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | q4_0 | — | — | — | — | 1.584 | f32 | 0e+0 | from the item 4 run |
| cpu | map:neg | q4_0 | — | — | — | — | 2.742 | f32 | 0e+0 | from the item 4 run |
| cpu | map:abs | q4_0 | — | — | — | — | 3.356 | f32 | 0e+0 | from the item 4 run |
| cpu | map:sign | q4_0 | — | — | — | — | 6.930 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:square | q4_0 | — | — | — | — | 3.325 | f32 | 1e+0 | from the item 4 run |
| cpu | map:relu | q4_0 | — | — | — | — | 3.506 | f32 | 1.88e-1 | from the item 4 run |
| cpu | map:add_scalar | q4_0 | — | — | — | — | 3.417 | f32 | 3.61e-1 | from the item 4 run |
| cpu | map:scale | q4_0 | — | — | — | — | 3.325 | f32 | 3.91e-3 | from the item 4 run |
| cpu | map:affine | q4_0 | — | — | — | — | 3.396 | f32 | 8.75e-1 | from the item 4 run |
| cpu | map:clamp | q4_0 | — | — | — | — | 3.359 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:add | q4_0 | — | — | — | — | 3.580 | f32 | 9.37e-1 | from the item 4 run |
| cpu | map:sub | q4_0 | — | — | — | — | 4.367 | f32 | 9.38e-1 | from the item 4 run |
| cpu | map:mul | q4_0 | — | — | — | — | 3.529 | f32 | 1.75e+0 | from the item 4 run |
| cpu | map:div | q4_0 | — | — | — | — | 3.614 | f32 | 5.06e-1 | from the item 4 run |
| cpu | map:max | q4_0 | — | — | — | — | 3.407 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:min | q4_0 | — | — | — | — | 3.405 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:axpy | q4_0 | — | — | — | — | 3.634 | f32 | 1.38e+0 | from the item 4 run |
| cpu | map:muladd | q4_0 | — | — | — | — | 4.109 | f32 | 2.18e+0 | from the item 4 run |
| cpu | map:where | q4_0 | — | — | — | — | 3.877 | f32 | 5e-1 | from the item 4 run |
| cpu | map:square_bwd | q4_0 | — | — | — | — | 3.551 | f32 | 3.5e+0 | from the item 4 run |
| cpu | map:abs_bwd | q4_0 | — | — | — | — | 3.492 | f32 | 5e-1 | from the item 4 run |
| cpu | map:relu_bwd | q4_0 | — | — | — | — | 3.480 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:clamp_bwd | q4_0 | — | — | — | — | 3.494 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:where_bwd | q4_0 | — | — | — | — | 3.275 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:sqrt | q4_0 | — | — | — | — | 2.632 | f32 | 1.21e-1 | from the item 4 run |
| cpu | map:inv_sqrt | q4_0 | — | — | — | — | 2.708 | f32 | 8.82e-2 | from the item 4 run |
| cpu | map:exp | q4_0 | — | — | — | — | 3.463 | f32 | 2.72e+0 | from the item 4 run |
| cpu | map:expm1 | q4_0 | — | — | — | — | 5.669 | f32 | 4.04e-2 | from the item 4 run |
| cpu | map:log | q4_0 | — | — | — | — | 3.874 | f32 | 9.81e-2 | from the item 4 run |
| cpu | map:sin | q4_0 | — | — | — | — | 7.429 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:cos | q4_0 | — | — | — | — | 6.544 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:tanh | q4_0 | — | — | — | — | 9.813 | f32 | 1.24e-1 | from the item 4 run |
| cpu | map:sigmoid | q4_0 | — | — | — | — | 5.198 | f32 | 6.13e-2 | from the item 4 run |
| cpu | map:erf | q4_0 | — | — | — | — | 32.702 | f32 | 1.25e-1 | from the item 4 run |
| cpu | map:gelu | q4_0 | — | — | — | — | 5.540 | f32 | 2.05e-1 | from the item 4 run |
| cpu | map:gelu_erf | q4_0 | — | — | — | — | 44.007 | f32 | 2.04e-1 | from the item 4 run |
| cpu | map:silu | q4_0 | — | — | — | — | 3.526 | f32 | 2.44e-1 | from the item 4 run |
| cpu | map:leaky_relu | q4_0 | — | — | — | — | 3.482 | f32 | 2.5e-1 | from the item 4 run |
| cpu | map:pow | q4_0 | — | — | — | — | 5.875 | f32 | 1.88e+0 | from the item 4 run |
| cpu | map:dropout | q4_0 | — | — | — | — | 4.194 | f32 | 5.48e-1 | from the item 4 run |
| cpu | map:gated_silu | q4_0 | — | — | — | — | 6.196 | f32 | 1.72e+0 | from the item 4 run |
| cpu | map:gated_gelu | q4_0 | — | — | — | — | 6.494 | f32 | 1.75e+0 | from the item 4 run |
| cpu | map:sqrt_bwd | q4_0 | — | — | — | — | 4.804 | f32 | 3.36e-1 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | q4_0 | — | — | — | — | 3.572 | f32 | 8.24e-1 | from the item 4 run |
| cpu | map:exp_bwd | q4_0 | — | — | — | — | 4.157 | f32 | 2.39e+1 | from the item 4 run |
| cpu | map:log_bwd | q4_0 | — | — | — | — | 3.441 | f32 | 1.04e+0 | from the item 4 run |
| cpu | map:sin_bwd | q4_0 | — | — | — | — | 5.047 | f32 | 5e-1 | from the item 4 run |
| cpu | map:cos_bwd | q4_0 | — | — | — | — | 5.284 | f32 | 5e-1 | from the item 4 run |
| cpu | map:tanh_bwd | q4_0 | — | — | — | — | 5.704 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:sigmoid_bwd | q4_0 | — | — | — | — | 5.574 | f32 | 1.09e-1 | from the item 4 run |
| cpu | map:silu_bwd | q4_0 | — | — | — | — | 5.648 | f32 | 4.82e-1 | from the item 4 run |
| cpu | map:gelu_bwd | q4_0 | — | — | — | — | 5.910 | f32 | 4.94e-1 | from the item 4 run |
| cpu | map:gelu_erf_bwd | q4_0 | — | — | — | — | 31.365 | f32 | 4.94e-1 | from the item 4 run |
| cpu | map:leaky_relu_bwd | q4_0 | — | — | — | — | 3.468 | f32 | 4.38e-1 | from the item 4 run |
| cpu | map:pow_bwd | q4_0 | — | — | — | — | 6.552 | f32 | 8.77e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | q4_0 | — | — | — | — | 6.226 | f32 | 1.9e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | q4_0 | — | — | — | — | 4.822 | f32 | 1.72e+0 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | q4_0 | — | — | — | — | 6.323 | f32 | 1.77e+0 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | q4_0 | — | — | — | — | 5.009 | f32 | 1.75e+0 | from the item 4 run |
| cpu | sum:whole | q4_0 | — | — | — | — | 54.582 | f32 | 6.1e-5 | from the item 4 run |
| cpu | sum:rows | q4_0 | — | — | — | — | 54.775 | f32 | 0e+0 | from the item 4 run |
| cpu | sum:cols | q4_0 | — | — | — | — | 55.584 | f32 | 0e+0 | from the item 4 run |
| cpu | sumsq:rows | q4_0 | — | — | — | — | 54.834 | f32 | 3.97e-4 | from the item 4 run |
| cpu | max:whole | q4_0 | — | — | — | — | 0.922 | f32 | 0e+0 | from the item 4 run |
| cpu | max:rows | q4_0 | — | — | — | — | 1.782 | f32 | 0e+0 | from the item 4 run |
| cpu | max:cols | q4_0 | — | — | — | — | 3.313 | f32 | 0e+0 | from the item 4 run |
| cpu | min:rows | q4_0 | — | — | — | — | 2.480 | f32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | q4_0 | — | — | — | — | 1.989 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | q4_0 | — | — | — | — | 1.773 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | q4_0 | — | — | — | — | 3.110 | f32 | 2.34e-1 | from the item 4 run |
| cpu | maprow:mul | q4_0 | — | — | — | — | 3.592 | f32 | 1.09e-1 | from the item 4 run |
| cpu | transpose | q4_0 | — | — | — | — | 5.455 | f32 | 1.25e-1 | from the item 4 run |
| cpu | gemv | q4_0 | — | — | — | — | 1.952 | s8dp blocks | 1.03e+1 | from the item 4 run |
| cpu | gemv_t | q4_0 | — | — | — | — | 224.006 | f32 | 1.14e+1 | from the item 4 run |
| cpu | row:log_softmax | q4_0 | — | — | — | — | 3.930 | f32 | 8.92e-1 | from the item 4 run |
| cpu | row:softmax_bwd | q4_0 | — | — | — | — | 4.880 | f32 | 1.89e+0 | from the item 4 run |
| cpu | row:log_softmax_bwd | q4_0 | — | — | — | — | 5.833 | f32 | 8.32e+0 | from the item 4 run |
| cpu | row:rms_norm_affine | q4_0 | — | — | — | — | 3.674 | f32 | 3.89e-1 | from the item 4 run |
| cpu | row:rms_norm_bwd | q4_0 | — | — | — | — | 3.768 | f32 | 9.72e-2 | from the item 4 run |
| cpu | row:layer_norm | q4_0 | — | — | — | — | 3.163 | f32 | 4.9e-1 | from the item 4 run |
| cpu | row:layer_norm_bwd | q4_0 | — | — | — | — | 3.861 | f32 | 9.7e-2 | from the item 4 run |
| cpu | row:weight_norm | q4_0 | — | — | — | — | 2.703 | f32 | 1.53e-5 | from the item 4 run |
| cpu | row:rope | q4_0 | — | — | — | — | 6.278 | f32 | 6.94e-1 | from the item 4 run |
| cpu | row:scale_mask_softmax | q4_0 | — | — | — | — | 4.629 | f32 | 3.11e-2 | from the item 4 run |
| cpu | row:add_rms_norm | q4_0 | — | — | — | — | 6.086 | f32 | 5.78e-1 | from the item 4 run |
| cpu | row:add_layer_norm | q4_0 | — | — | — | — | 6.155 | f32 | 6.37e-1 | from the item 4 run |
| cpu | row:cross_entropy_bwd | q4_0 | — | — | — | — | 3.960 | f32 | 2.14e-3 | from the item 4 run |
| cpu | norm_params_bwd:rms | q4_0 | — | — | — | — | 3.073 | f32 | 2.74e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | q4_0 | — | — | — | — | 3.041 | f32 | 3.15e-4 | from the item 4 run |
| cpu | layout:strided | q4_0 | — | — | — | — | 37.520 | f32 | 1.25e-1 | from the item 4 run |
| cpu | layout:one_hot | q4_0 | — | — | — | — | 1.614 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_rows | q4_0 | — | — | — | — | 2.272 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | q4_0 | — | — | — | — | 36.152 | f32 | 1.25e-1 | from the item 4 run |
| cpu | layout:scatter_add_rows | q4_0 | — | — | — | — | 87.141 | f32 | 4.39e-1 | from the item 4 run |
| cpu | layout:im2col | q4_0 | — | — | — | — | 18.945 | f32 | 1.25e-1 | from the item 4 run |
| cpu | layout:col2im | q4_0 | — | — | — | — | 51.294 | f32 | 6.46e-1 | from the item 4 run |
| cpu | layout:interpolate_nearest | q4_0 | — | — | — | — | 34.807 | f32 | 1.09e-1 | from the item 4 run |
| cpu | layout:interpolate_bilinear | q4_0 | — | — | — | — | 130.866 | f32 | 1.12e-1 | from the item 4 run |
| cpu | layout:shift | q4_0 | — | — | — | — | 30.697 | f32 | 1.25e-1 | from the item 4 run |
| cpu | layout:pad2d | q4_0 | — | — | — | — | 2.812 | f32 | 1.25e-1 | from the item 4 run |
| cpu | layout:maxpool | q4_0 | — | — | — | — | 8.702 | f32 | 1.09e-1 | from the item 4 run |
| cpu | layout:avgpool | q4_0 | — | — | — | — | 8.536 | f32 | 1.06e-1 | from the item 4 run |
| cpu | layout:maxpool_bwd | q4_0 | — | — | — | — | 49.759 | f32 | 1.09e-1 | from the item 4 run |
| cpu | layout:avgpool_bwd | q4_0 | — | — | — | — | 10.050 | f32 | 2.73e-2 | from the item 4 run |
| cpu | layout:adaptive_maxpool | q4_0 | — | — | — | — | 135.532 | f32 | 1.17e-2 | from the item 4 run |
| cpu | layout:adaptive_avgpool | q4_0 | — | — | — | — | 131.701 | f32 | 4.53e-3 | from the item 4 run |
| cpu | layout:depthwise_conv | q4_0 | — | — | — | — | 148.913 | f32 | 4.7e-1 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | q4_0 | — | — | — | — | 12037.421 | f32 | 1.09e-1 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | q4_0 | — | — | — | — | 16.579 | f32 | 1.35e-4 | from the item 4 run |
| cpu | matmul_nt | q4_0 | 1024x1024x1024 Bᵀ | — | — | — | 2.064 | f32 | 4.02e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | q4_0 | 1024x1024x1024 Aᵀ | — | — | — | 2.281 | f32 | 4.36e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | q4_0 | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 2.398 | f32 | 4.29e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | q4_0 | 16 x 256x256x256 | — | — | — | 1.159 | f32 | 2.28e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | q4_0 | 1024x1024x1024 (dA, dB) | — | — | — | 4.514 | f32 | 4.55e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_blocks | q4_0 | 1024x1024x1024 Bᵀ | — | — | — | 2.793 | s8dp blocks | 4.02e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | q4_0 | — | — | — | — | 52.830 | f32 | 6.18e-3 | from the item 4 run |
| cpu | attention_causal | q4_0 | — | — | — | — | 45.848 | f32 | 1.06e-1 | from the item 4 run |
| cpu | attention_fused | q4_0 | — | — | — | — | 82.441 | f32 | 6.18e-3 | from the item 4 run |
| cpu | attention_fused_causal | q4_0 | — | — | — | — | 43.311 | f32 | 1.06e-1 | from the item 4 run |
| cpu | attention_decode | q4_0 | — | — | — | — | 2.039 | f32 | 3e-3 | from the item 4 run |
| cpu | attention_bwd | q4_0 | — | — | — | — | 15.120 | f32 | 3.43e-2 | from the item 4 run |
| cpu | optimizer:sgd_momentum | q4_0 | — | — | — | — | 6.857 | f32; + restore of p, m, v | 2.92e-3 | from the item 4 run |
| cpu | optimizer:adam | q4_0 | — | — | — | — | 10.706 | f32; + restore of p, m, v | 3.16e-2 | from the item 4 run |
| cpu | optimizer:adamw | q4_0 | — | — | — | — | 10.528 | f32; + restore of p, m, v | 3.16e-2 | from the item 4 run |
| cpu | conv | q4_0 | — | — | — | — | 6.554 | f32 | 3.39e+0 | from the item 4 run |
| cpu | conv_bwd | q4_0 | — | — | — | — | 17.861 | f32 | 4.35e+0 | from the item 4 run |
| cpu | lstm_step | q4_0 | — | — | — | — | 24.555 | f32 | 5.17e-2 | from the item 4 run |
| cpu | gru_step | q4_0 | — | — | — | — | 15.706 | f32 | 9.63e-2 | from the item 4 run |
| cpu | lstm_step_bwd | q4_0 | — | — | — | — | 50.975 | f32 | 5.03e-2 | from the item 4 run |
| cpu | gru_step_bwd | q4_0 | — | — | — | — | 35.910 | f32 | 4.61e-2 | from the item 4 run |
| cpu | gru_seq | q4_0 | — | — | — | — | 25.982 | f32 | 1e-1 | from the item 4 run |
| cpu | gru_seq_bwd | q4_0 | — | — | — | — | 106.265 | f32 | 5.31e-2 | from the item 4 run |
| cpu | topk_rows | q4_0 | — | — | — | — | 264.549 | f32 | 3.91e-3 | from the item 4 run |
| cpu | pick_rows | q4_0 | — | — | — | — | 0.165 | f32 | 4.98e-1 | from the item 4 run |
| cpu | sample_rows | q4_0 | — | — | — | — | 10.644 | f32 | 3e+0 | from the item 4 run |
| cpu | cross_entropy | q4_0 | — | — | — | — | 31.037 | f32 (log_softmax, pick, neg) | 1.96e+0 | from the item 4 run |
| cpu | ctc | q4_0 | — | — | — | — | 9.929 | f32 | 1.07e+2 | from the item 4 run |
| cpu | ctc_bwd | q4_0 | — | — | — | — | 14.316 | f32 | 5.63e-2 | from the item 4 run |
| cpu | softmax | q4_0 | 4096x4096 | — | — | — | 4.079 | f32 | 1.11e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | q4_0 | 4096x4096 | — | — | — | 2.673 | f32 | 4.89e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | q4_k | 16777216 | — | — | — | 6.242 | f32 | 1.33e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | q4_k | 1024x1024x1024 | — | — | — | 8.752 | f32 | 2.71e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | map:copy | q4_k | — | — | — | — | 4.607 | f32 | 0e+0 | from the item 4 run |
| cpu | map:fill | q4_k | — | — | — | — | 4.215 | f32 | 7.32e-4 | from the item 4 run |
| cpu | map:neg | q4_k | — | — | — | — | 4.634 | f32 | 3.24e-2 | from the item 4 run |
| cpu | map:abs | q4_k | — | — | — | — | 4.615 | f32 | 1.36e-1 | from the item 4 run |
| cpu | map:sign | q4_k | — | — | — | — | 4.741 | f32 | 6.61e-2 | from the item 4 run |
| cpu | map:square | q4_k | — | — | — | — | 4.622 | f32 | 5.55e-1 | from the item 4 run |
| cpu | map:relu | q4_k | — | — | — | — | 4.642 | f32 | 1.36e-1 | from the item 4 run |
| cpu | map:add_scalar | q4_k | — | — | — | — | 5.553 | f32 | 1.99e-1 | from the item 4 run |
| cpu | map:scale | q4_k | — | — | — | — | 5.556 | f32 | 1.11e-2 | from the item 4 run |
| cpu | map:affine | q4_k | — | — | — | — | 4.630 | f32 | 5.47e-2 | from the item 4 run |
| cpu | map:clamp | q4_k | — | — | — | — | 4.848 | f32 | 1e-1 | from the item 4 run |
| cpu | map:add | q4_k | — | — | — | — | 5.062 | f32 | 5.31e-1 | from the item 4 run |
| cpu | map:sub | q4_k | — | — | — | — | 5.137 | f32 | 5.31e-1 | from the item 4 run |
| cpu | map:mul | q4_k | — | — | — | — | 5.040 | f32 | 1.07e+0 | from the item 4 run |
| cpu | map:div | q4_k | — | — | — | — | 6.268 | f32 | 2.7e-1 | from the item 4 run |
| cpu | map:max | q4_k | — | — | — | — | 6.173 | f32 | 2.66e-1 | from the item 4 run |
| cpu | map:min | q4_k | — | — | — | — | 5.169 | f32 | 2.66e-1 | from the item 4 run |
| cpu | map:axpy | q4_k | — | — | — | — | 5.085 | f32 | 7.81e-1 | from the item 4 run |
| cpu | map:muladd | q4_k | — | — | — | — | 5.557 | f32 | 1.32e+0 | from the item 4 run |
| cpu | map:where | q4_k | — | — | — | — | 5.554 | f32 | 2.68e-1 | from the item 4 run |
| cpu | map:square_bwd | q4_k | — | — | — | — | 6.094 | f32 | 2.15e+0 | from the item 4 run |
| cpu | map:abs_bwd | q4_k | — | — | — | — | 6.151 | f32 | 2.73e-1 | from the item 4 run |
| cpu | map:relu_bwd | q4_k | — | — | — | — | 5.063 | f32 | 2.65e-1 | from the item 4 run |
| cpu | map:clamp_bwd | q4_k | — | — | — | — | 5.076 | f32 | 2.65e-1 | from the item 4 run |
| cpu | map:where_bwd | q4_k | — | — | — | — | 5.066 | f32 | 2.65e-1 | from the item 4 run |
| cpu | map:sqrt | q4_k | — | — | — | — | 4.767 | f32 | 6.6e-2 | from the item 4 run |
| cpu | map:inv_sqrt | q4_k | — | — | — | — | 4.616 | f32 | 5.22e-2 | from the item 4 run |
| cpu | map:exp | q4_k | — | — | — | — | 5.194 | f32 | 1.96e+0 | from the item 4 run |
| cpu | map:expm1 | q4_k | — | — | — | — | 7.419 | f32 | 3.47e-2 | from the item 4 run |
| cpu | map:log | q4_k | — | — | — | — | 5.683 | f32 | 6.72e-2 | from the item 4 run |
| cpu | map:sin | q4_k | — | — | — | — | 8.169 | f32 | 6.67e-2 | from the item 4 run |
| cpu | map:cos | q4_k | — | — | — | — | 5.781 | f32 | 6.67e-2 | from the item 4 run |
| cpu | map:tanh | q4_k | — | — | — | — | 5.865 | f32 | 6.66e-2 | from the item 4 run |
| cpu | map:sigmoid | q4_k | — | — | — | — | 5.285 | f32 | 3.28e-2 | from the item 4 run |
| cpu | map:erf | q4_k | — | — | — | — | 17.471 | f32 | 6.67e-2 | from the item 4 run |
| cpu | map:gelu | q4_k | — | — | — | — | 5.611 | f32 | 1.41e-1 | from the item 4 run |
| cpu | map:gelu_erf | q4_k | — | — | — | — | 29.250 | f32 | 1.42e-1 | from the item 4 run |
| cpu | map:silu | q4_k | — | — | — | — | 5.315 | f32 | 1.43e-1 | from the item 4 run |
| cpu | map:leaky_relu | q4_k | — | — | — | — | 4.719 | f32 | 1.48e-1 | from the item 4 run |
| cpu | map:pow | q4_k | — | — | — | — | 7.359 | f32 | 1.04e+0 | from the item 4 run |
| cpu | map:dropout | q4_k | — | — | — | — | 5.067 | f32 | 3.32e-1 | from the item 4 run |
| cpu | map:gated_silu | q4_k | — | — | — | — | 5.761 | f32 | 1.05e+0 | from the item 4 run |
| cpu | map:gated_gelu | q4_k | — | — | — | — | 5.920 | f32 | 1.07e+0 | from the item 4 run |
| cpu | map:sqrt_bwd | q4_k | — | — | — | — | 5.058 | f32 | 1.91e-1 | from the item 4 run |
| cpu | map:inv_sqrt_bwd | q4_k | — | — | — | — | 5.154 | f32 | 4.15e-1 | from the item 4 run |
| cpu | map:exp_bwd | q4_k | — | — | — | — | 5.615 | f32 | 1.51e+1 | from the item 4 run |
| cpu | map:log_bwd | q4_k | — | — | — | — | 5.112 | f32 | 5.64e-1 | from the item 4 run |
| cpu | map:sin_bwd | q4_k | — | — | — | — | 6.284 | f32 | 2.71e-1 | from the item 4 run |
| cpu | map:cos_bwd | q4_k | — | — | — | — | 6.566 | f32 | 2.69e-1 | from the item 4 run |
| cpu | map:tanh_bwd | q4_k | — | — | — | — | 6.903 | f32 | 2.66e-1 | from the item 4 run |
| cpu | map:sigmoid_bwd | q4_k | — | — | — | — | 6.891 | f32 | 6.64e-2 | from the item 4 run |
| cpu | map:silu_bwd | q4_k | — | — | — | — | 6.961 | f32 | 2.93e-1 | from the item 4 run |
| cpu | map:gelu_bwd | q4_k | — | — | — | — | 7.317 | f32 | 3e-1 | from the item 4 run |
| cpu | map:gelu_erf_bwd | q4_k | — | — | — | — | 32.570 | f32 | 3e-1 | from the item 4 run |
| cpu | map:leaky_relu_bwd | q4_k | — | — | — | — | 5.090 | f32 | 2.64e-1 | from the item 4 run |
| cpu | map:pow_bwd | q4_k | — | — | — | — | 8.096 | f32 | 5.3e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_x | q4_k | — | — | — | — | 7.391 | f32 | 1.12e+0 | from the item 4 run |
| cpu | map:gated_silu_bwd_y | q4_k | — | — | — | — | 6.257 | f32 | 1.05e+0 | from the item 4 run |
| cpu | map:gated_gelu_bwd_x | q4_k | — | — | — | — | 7.675 | f32 | 1.07e+0 | from the item 4 run |
| cpu | map:gated_gelu_bwd_y | q4_k | — | — | — | — | 6.324 | f32 | 1.07e+0 | from the item 4 run |
| cpu | sum:whole | q4_k | — | — | — | — | 56.902 | f32 | 1.61e-4 | from the item 4 run |
| cpu | sum:rows | q4_k | — | — | — | — | 56.675 | f32 | 9.54e-6 | from the item 4 run |
| cpu | sum:cols | q4_k | — | — | — | — | 58.935 | f32 | 2.17e-4 | from the item 4 run |
| cpu | sumsq:rows | q4_k | — | — | — | — | 56.783 | f32 | 3.36e-4 | from the item 4 run |
| cpu | max:whole | q4_k | — | — | — | — | 1.940 | f32 | 0e+0 | from the item 4 run |
| cpu | max:rows | q4_k | — | — | — | — | 2.053 | f32 | 0e+0 | from the item 4 run |
| cpu | max:cols | q4_k | — | — | — | — | 3.974 | f32 | 0e+0 | from the item 4 run |
| cpu | min:rows | q4_k | — | — | — | — | 1.986 | f32 | 0e+0 | from the item 4 run |
| cpu | argmax:rows | q4_k | — | — | — | — | 1.925 | s32 index | 0e+0 | from the item 4 run |
| cpu | argmax:cols | q4_k | — | — | — | — | 2.444 | s32 index | 0e+0 | from the item 4 run |
| cpu | maprow:add | q4_k | — | — | — | — | 5.465 | f32 | 1.32e-1 | from the item 4 run |
| cpu | maprow:mul | q4_k | — | — | — | — | 5.469 | f32 | 6.65e-2 | from the item 4 run |
| cpu | transpose | q4_k | — | — | — | — | 24.786 | f32 | 1e+0 | from the item 4 run |
| cpu | gemv | q4_k | — | — | — | — | 226.217 | f32 | 6.29e+0 | from the item 4 run |
| cpu | gemv_t | q4_k | — | — | — | — | 232.515 | f32 | 6.57e+0 | from the item 4 run |
| cpu | row:log_softmax | q4_k | — | — | — | — | 6.081 | f32 | 1.14e-1 | from the item 4 run |
| cpu | row:softmax_bwd | q4_k | — | — | — | — | 5.060 | f32 | 2.49e+0 | from the item 4 run |
| cpu | row:log_softmax_bwd | q4_k | — | — | — | — | 6.320 | f32 | 4.44e+0 | from the item 4 run |
| cpu | row:rms_norm_affine | q4_k | — | — | — | — | 4.598 | f32 | 2.33e-1 | from the item 4 run |
| cpu | row:rms_norm_bwd | q4_k | — | — | — | — | 5.170 | f32 | 5.79e-2 | from the item 4 run |
| cpu | row:layer_norm | q4_k | — | — | — | — | 4.668 | f32 | 2.86e-1 | from the item 4 run |
| cpu | row:layer_norm_bwd | q4_k | — | — | — | — | 5.353 | f32 | 5.79e-2 | from the item 4 run |
| cpu | row:weight_norm | q4_k | — | — | — | — | 5.477 | f32 | 4.3e-5 | from the item 4 run |
| cpu | row:rope | q4_k | — | — | — | — | 7.750 | f32 | 3.73e-1 | from the item 4 run |
| cpu | row:scale_mask_softmax | q4_k | — | — | — | — | 6.341 | f32 | 1.75e-2 | from the item 4 run |
| cpu | row:add_rms_norm | q4_k | — | — | — | — | 9.162 | f32 | 3.11e-1 | from the item 4 run |
| cpu | row:add_layer_norm | q4_k | — | — | — | — | 9.264 | f32 | 3.56e-1 | from the item 4 run |
| cpu | row:cross_entropy_bwd | q4_k | — | — | — | — | 6.037 | f32 | 2.52e-3 | from the item 4 run |
| cpu | norm_params_bwd:rms | q4_k | — | — | — | — | 3.820 | f32 | 2.8e-4 | from the item 4 run |
| cpu | norm_params_bwd:layer | q4_k | — | — | — | — | 3.948 | f32 | 2.65e-4 | from the item 4 run |
| cpu | layout:strided | q4_k | — | — | — | — | 159.167 | f32 | 6.72e-2 | from the item 4 run |
| cpu | layout:one_hot | q4_k | — | — | — | — | 4.363 | f32 | 3.57e-4 | from the item 4 run |
| cpu | layout:gather_rows | q4_k | — | — | — | — | 4.446 | f32 | 0e+0 | from the item 4 run |
| cpu | layout:gather_cols | q4_k | — | — | — | — | 149.465 | f32 | 6.72e-2 | from the item 4 run |
| cpu | layout:scatter_add_rows | q4_k | — | — | — | — | 180.432 | f32 | 2.77e-1 | from the item 4 run |
| cpu | layout:im2col | q4_k | — | — | — | — | 87.492 | f32 | 6.67e-2 | from the item 4 run |
| cpu | layout:col2im | q4_k | — | — | — | — | 99.963 | f32 | 4.18e-1 | from the item 4 run |
| cpu | layout:interpolate_nearest | q4_k | — | — | — | — | 163.332 | f32 | 6.2e-2 | from the item 4 run |
| cpu | layout:interpolate_bilinear | q4_k | — | — | — | — | 586.920 | f32 | 6.66e-2 | from the item 4 run |
| cpu | layout:shift | q4_k | — | — | — | — | 158.628 | f32 | 6.72e-2 | from the item 4 run |
| cpu | layout:pad2d | q4_k | — | — | — | — | 4.485 | f32 | 6.7e-2 | from the item 4 run |
| cpu | layout:maxpool | q4_k | — | — | — | — | 38.368 | f32 | 6.36e-2 | from the item 4 run |
| cpu | layout:avgpool | q4_k | — | — | — | — | 39.655 | f32 | 5.84e-2 | from the item 4 run |
| cpu | layout:maxpool_bwd | q4_k | — | — | — | — | 203.073 | f32 | 6.63e-2 | from the item 4 run |
| cpu | layout:avgpool_bwd | q4_k | — | — | — | — | 34.708 | f32 | 1.55e-2 | from the item 4 run |
| cpu | layout:adaptive_maxpool | q4_k | — | — | — | — | 568.769 | f32 | 2.97e-2 | from the item 4 run |
| cpu | layout:adaptive_avgpool | q4_k | — | — | — | — | 568.130 | f32 | 3.06e-3 | from the item 4 run |
| cpu | layout:depthwise_conv | q4_k | — | — | — | — | 678.618 | f32 | 2.95e-1 | from the item 4 run |
| cpu | layout:adaptive_maxpool_bwd | q4_k | — | — | — | — | 52113.482 | f32 | 5.43e-2 | from the item 4 run |
| cpu | layout:adaptive_avgpool_bwd | q4_k | — | — | — | — | 45.936 | f32 | 6.65e-5 | from the item 4 run |
| cpu | matmul_nt | q4_k | 1024x1024x1024 Bᵀ | — | — | — | 1.912 | f32 | 2.73e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | q4_k | 1024x1024x1024 Aᵀ | — | — | — | 3.179 | f32 | 2.8e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | q4_k | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 3.766 | f32 | 2.66e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | q4_k | 16 x 256x256x256 | — | — | — | 1.459 | f32 | 1.36e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | q4_k | 1024x1024x1024 (dA, dB) | — | — | — | 5.889 | f32 | 2.73e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | attention | q4_k | — | — | — | — | 55.099 | f32 | 3.47e-3 | from the item 4 run |
| cpu | attention_causal | q4_k | — | — | — | — | 44.892 | f32 | 5.93e-2 | from the item 4 run |
| cpu | attention_fused | q4_k | — | — | — | — | 81.701 | f32 | 3.47e-3 | from the item 4 run |
| cpu | attention_fused_causal | q4_k | — | — | — | — | 45.252 | f32 | 5.93e-2 | from the item 4 run |
| cpu | attention_decode | q4_k | — | — | — | — | 3.654 | f32 | 2.05e-3 | from the item 4 run |
| cpu | attention_bwd | q4_k | — | — | — | — | 18.661 | f32 | 2.51e-2 | from the item 4 run |
| cpu | optimizer:sgd_momentum | q4_k | — | — | — | — | 8.390 | f32; + restore of p, m, v | 5.87e-3 | from the item 4 run |
| cpu | optimizer:adam | q4_k | — | — | — | — | 13.262 | f32; + restore of p, m, v | 2.26e-2 | from the item 4 run |
| cpu | optimizer:adamw | q4_k | — | — | — | — | 14.083 | f32; + restore of p, m, v | 2.26e-2 | from the item 4 run |
| cpu | conv | q4_k | — | — | — | — | 8.071 | f32 | 2.08e+0 | from the item 4 run |
| cpu | conv_bwd | q4_k | — | — | — | — | 18.053 | f32 | 2.95e+0 | from the item 4 run |
| cpu | lstm_step | q4_k | — | — | — | — | 23.954 | f32 | 2.95e-2 | from the item 4 run |
| cpu | gru_step | q4_k | — | — | — | — | 14.571 | f32 | 5.6e-2 | from the item 4 run |
| cpu | lstm_step_bwd | q4_k | — | — | — | — | 59.980 | f32 | 3.21e-2 | from the item 4 run |
| cpu | gru_step_bwd | q4_k | — | — | — | — | 45.604 | f32 | 3.05e-2 | from the item 4 run |
| cpu | gru_seq | q4_k | — | — | — | — | 20.433 | f32 | 5.58e-2 | from the item 4 run |
| cpu | softmax | q4_k | 4096x4096 | — | — | — | 6.236 | f32 | 7.31e-5 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | q4_k | 4096x4096 | — | — | — | 6.261 | f32 | 1.37e-3 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | q6_k | 16777216 | — | — | — | 10.344 | f32 | 3.22e-2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | q6_k | 1024x1024x1024 | — | — | — | 10.107 | f32 | 8.19e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_nt | q6_k | 1024x1024x1024 Bᵀ | — | — | — | 3.569 | f32 | 8.57e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | q6_k | 1024x1024x1024 Aᵀ | — | — | — | 3.878 | f32 | 8.99e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | q6_k | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 4.912 | f32 | 7.73e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | q6_k | 16 x 256x256x256 | — | — | — | 1.502 | f32 | 4.11e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | q6_k | 1024x1024x1024 (dA, dB) | — | — | — | 7.408 | f32 | 8.68e-1 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | softmax | q6_k | 4096x4096 | — | — | — | 11.383 | f32 | 1.17e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | rms_norm | q6_k | 4096x4096 | — | — | — | 5.618 | f32 | 4.69e-4 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | add | bool | 16777216 | — | — | — | 4.923 | u8 | 1e+0 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | 1.358 | f32 | 6.46e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_nt | bool | 1024x1024x1024 Bᵀ | — | — | — | 3.627 | f32 | 6.44e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tn | bool | 1024x1024x1024 Aᵀ | — | — | — | 2.910 | f32 | 6.5e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_tt | bool | 1024x1024x1024 Aᵀ Bᵀ | — | — | — | 3.047 | f32 | 6.53e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_batched | bool | 16 x 256x256x256 | — | — | — | 4.167 | f32 | 1.8e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
| cpu | matmul_bwd | bool | 1024x1024x1024 (dA, dB) | — | — | — | 7.323 | f32 | 6.53e+2 | end-of-item-6 run, invocation 1 (core and products, every type) |
