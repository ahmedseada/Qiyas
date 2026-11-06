# Qiyas bench

2026-11-06 19:11. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

Item 4's `cuda,cpu --report --ops all` run (every column), stopped by the coordinator during the CPU's last types: every
CUDA row; the CPU rows up to q4_k `gru_seq` (CPU q6_k and bool missing). Converted from the run's console lines (they
carry no shape: the shape is the earlier baseline's for its rows, — for the rest).

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | add | float64 | 16777216 | 124.683 | — | add_f32; converted through the host | 5.96e-8 | 1.000 | 124.73 | add_f64 | 0e+0 | 0.521 | 239.27 | f64 | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 31.448 | — | gemm128_f32; converted through the host | 1.62e-4 | 23.081 | 1.36 | gemm64_f64 | 2.59e-13 | 4.175 | 7.53 | f64x | 0e+0 |
| cuda | map:copy | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | map:fill | float64 | — | 42.456 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.165 | 257.62 | f64 | 0e+0 |
| cuda | map:neg | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | map:abs | float64 | — | 85.519 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 238.61 | f64 | 0e+0 |
| cuda | map:sign | float64 | — | 90.860 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.359 | 252.95 | f64 | 0e+0 |
| cuda | map:square | float64 | — | 90.022 | — | square_f32; converted through the host | 4.77e-7 | — | — | — | — | 0.359 | 250.69 | f64 | 0e+0 |
| cuda | map:relu | float64 | — | 83.957 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 234.32 | f64 | 0e+0 |
| cuda | map:add_scalar | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | map:scale | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | map:affine | float64 | — | 83.598 | — | affine_f32; converted through the host | 4.77e-7 | — | — | — | — | 0.358 | 233.45 | f64 | 0e+0 |
| cuda | map:clamp | float64 | — | 83.531 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.359 | 232.81 | f64 | 0e+0 |
| cuda | map:add | float64 | — | — | — | — | — | — | — | — | — | 0.520 | — | f64 | 0e+0 |
| cuda | map:sub | float64 | — | 124.892 | — | sub_f32; converted through the host | 2.38e-7 | — | — | — | — | 1.000 | 124.88 | f64 | 0e+0 |
| cuda | map:mul | float64 | — | 127.381 | — | mul_f32; converted through the host | 4.77e-7 | — | — | — | — | 0.521 | 244.68 | f64 | 0e+0 |
| cuda | map:div | float64 | — | — | — | — | — | — | — | — | — | 0.521 | — | f64 | 0e+0 |
| cuda | map:max | float64 | — | 125.616 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.521 | 241.20 | f64 | 0e+0 |
| cuda | map:min | float64 | — | 126.108 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.521 | 242.10 | f64 | 0e+0 |
| cuda | map:axpy | float64 | — | 124.311 | — | axpy_f32; converted through the host; + copy of y | 4.77e-7 | — | — | — | — | 0.521 | 238.78 | f64 | 0e+0 |
| cuda | map:muladd | float64 | — | 166.675 | — | muladd_f32; converted through the host; + copy of z | 9.54e-7 | — | — | — | — | 1.311 | 127.18 | f64 | 0e+0 |
| cuda | map:where | float64 | — | 177.774 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.683 | 260.36 | f64 | 0e+0 |
| cuda | map:square_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.520 | — | f64 | 0e+0 |
| cuda | map:abs_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.521 | — | f64 | 0e+0 |
| cuda | map:relu_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.520 | — | f64 | 0e+0 |
| cuda | map:clamp_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.520 | — | f64 | 0e+0 |
| cuda | map:where_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.520 | — | f64 | 0e+0 |
| cuda | map:sqrt | float64 | — | 89.047 | — | sqrt_f32; converted through the host | 5.96e-8 | — | — | — | — | 0.399 | 223.29 | f64 | 0e+0 |
| cuda | map:inv_sqrt | float64 | — | 84.343 | — | inv_sqrt_f32; converted through the host | 1.37e-7 | — | — | — | — | 0.836 | 100.90 | f64 | 0e+0 |
| cuda | map:exp | float64 | — | 91.481 | — | exp_f32; converted through the host | 1.36e-5 | — | — | — | — | 1.232 | 74.27 | f64 | 7.11e-15 |
| cuda | map:expm1 | float64 | — | — | — | — | — | — | — | — | — | 1.226 | — | f64 | 1.11e-16 |
| cuda | map:log | float64 | — | 83.314 | — | log_f32; converted through the host | 1.74e-7 | — | — | — | — | 1.670 | 49.90 | f64 | 2.22e-16 |
| cuda | map:sin | float64 | — | 83.004 | — | sin_f32; converted through the host | 7.09e-8 | — | — | — | — | 1.536 | 54.05 | f64 | 2.22e-16 |
| cuda | map:cos | float64 | — | 89.351 | — | cos_f32; converted through the host | 7.07e-8 | — | — | — | — | 1.587 | 56.32 | f64 | 1.11e-16 |
| cuda | map:tanh | float64 | — | 83.450 | — | tanh_f32; converted through the host | 2.77e-7 | — | — | — | — | 1.814 | 46.02 | f64 | 3.33e-16 |
| cuda | map:sigmoid | float64 | — | 83.646 | — | sigmoid_f32; converted through the host | 9.45e-8 | — | — | — | — | 1.721 | 48.61 | f64 | 2.22e-16 |
| cuda | map:erf | float64 | — | — | — | — | — | — | — | — | — | 47.760 | — | f64 | 6.66e-16 |
| cuda | map:gelu | float64 | — | 81.148 | — | gelu_f32; converted through the host | 3.13e-7 | — | — | — | — | 1.971 | 41.17 | f64 | 8.88e-16 |
| cuda | map:gelu_erf | float64 | — | — | — | — | — | — | — | — | — | 177.879 | — | f64 | 4.44e-16 |
| cuda | map:silu | float64 | — | 84.428 | — | silu_f32; converted through the host | 4.57e-7 | — | — | — | — | 1.770 | 47.69 | f64 | 8.88e-16 |
| cuda | map:leaky_relu | float64 | — | 84.517 | — | leaky_relu_f32; converted through the host | 1.79e-8 | — | — | — | — | 0.358 | 236.15 | f64 | 0e+0 |
| cuda | map:pow | float64 | — | — | — | — | — | — | — | — | — | 3.388 | — | f64 | 2.84e-14 |
| cuda | map:dropout | float64 | — | — | — | — | — | — | — | — | — | 0.521 | — | f64 | 0e+0 |
| cuda | map:gated_silu | float64 | — | — | — | — | — | — | — | — | — | 1.823 | — | f64 | 3.55e-15 |
| cuda | map:gated_gelu | float64 | — | — | — | — | — | — | — | — | — | 2.024 | — | f64 | 3.55e-15 |
| cuda | map:sqrt_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.823 | — | f64 | 0e+0 |
| cuda | map:inv_sqrt_bwd | float64 | — | — | — | — | — | — | — | — | — | 1.310 | — | f64 | 1.78e-15 |
| cuda | map:exp_bwd | float64 | — | — | — | — | — | — | — | — | — | 1.283 | — | f64 | 2.84e-14 |
| cuda | map:log_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.521 | — | f64 | 0e+0 |
| cuda | map:sin_bwd | float64 | — | — | — | — | — | — | — | — | — | 1.638 | — | f64 | 4.44e-16 |
| cuda | map:cos_bwd | float64 | — | — | — | — | — | — | — | — | — | 1.636 | — | f64 | 4.44e-16 |
| cuda | map:tanh_bwd | float64 | — | — | — | — | — | — | — | — | — | 3.684 | — | f64 | 3.11e-15 |
| cuda | map:sigmoid_bwd | float64 | — | — | — | — | — | — | — | — | — | 3.479 | — | f64 | 4.44e-16 |
| cuda | map:silu_bwd | float64 | — | — | — | — | — | — | — | — | — | 3.526 | — | f64 | 2.66e-15 |
| cuda | map:gelu_bwd | float64 | — | — | — | — | — | — | — | — | — | 3.923 | — | f64 | 1.78e-15 |
| cuda | map:gelu_erf_bwd | float64 | — | — | — | — | — | — | — | — | — | 181.093 | — | f64 | 1.78e-15 |
| cuda | map:leaky_relu_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.520 | — | f64 | 0e+0 |
| cuda | map:pow_bwd | float64 | — | — | — | — | — | — | — | — | — | 3.540 | — | f64 | 4.26e-14 |
| cuda | map:gated_silu_bwd_x | float64 | — | — | — | — | — | — | — | — | — | 3.576 | — | f64 | 7.11e-15 |
| cuda | map:gated_silu_bwd_y | float64 | — | — | — | — | — | — | — | — | — | 1.822 | — | f64 | 3.55e-15 |
| cuda | map:gated_gelu_bwd_x | float64 | — | — | — | — | — | — | — | — | — | 3.974 | — | f64 | 7.11e-15 |
| cuda | map:gated_gelu_bwd_y | float64 | — | — | — | — | — | — | — | — | — | 2.024 | — | f64 | 3.55e-15 |
| cuda | sum:whole | float64 | — | 42.527 | — | f32; converted through the host | 1.95e-4 | — | — | — | — | 0.176 | 242.18 | f64 | 0e+0 |
| cuda | sum:rows | float64 | — | 42.995 | — | f32; converted through the host | 3.58e-5 | — | — | — | — | 0.171 | 251.43 | f64 | 0e+0 |
| cuda | sum:cols | float64 | — | 42.653 | — | f32; converted through the host; + zero of y | 3.28e-4 | — | — | — | — | 0.178 | 239.49 | f64 | 0e+0 |
| cuda | sumsq:rows | float64 | — | — | — | — | — | — | — | — | — | 0.327 | — | f64 | 1.98e-11 |
| cuda | max:whole | float64 | — | — | — | — | — | — | — | — | — | 0.175 | — | f64 | 0e+0 |
| cuda | max:rows | float64 | — | — | — | — | — | — | — | — | — | 0.171 | — | f64 | 0e+0 |
| cuda | max:cols | float64 | — | — | — | — | — | — | — | — | — | 0.337 | — | f64 | 0e+0 |
| cuda | min:rows | float64 | — | — | — | — | — | — | — | — | — | 0.171 | — | f64 | 0e+0 |
| cuda | argmax:rows | float64 | — | 42.408 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.171 | 248.00 | s32 index | 0e+0 |
| cuda | argmax:cols | float64 | — | — | — | — | — | — | — | — | — | 0.635 | — | s32 index | 0e+0 |
| cuda | maprow:add | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | maprow:mul | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | transpose | float64 | — | 82.822 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.361 | 229.49 | f64 | 0e+0 |
| cuda | gemv | float64 | — | — | — | — | — | — | — | — | — | 0.656 | — | f64 | 6.11e-13 |
| cuda | gemv_t | float64 | — | — | — | — | — | — | — | — | — | 0.654 | — | f64 | 6.39e-13 |
| cuda | row:log_softmax | float64 | — | 82.021 | — | softmax_f32; converted through the host | 1.66e-6 | — | — | — | — | 2.071 | 39.60 | f64 | 7.11e-15 |
| cuda | row:softmax_bwd | float64 | — | 128.066 | — | softmax_bwd_f32; converted through the host; + zero of dx | 2.04e-5 | — | — | — | — | 0.521 | 246.04 | f64 | 3.41e-13 |
| cuda | row:log_softmax_bwd | float64 | — | 128.295 | — | softmax_bwd_f32; converted through the host; + zero of dx | 2.45e-5 | — | — | — | — | 1.529 | 83.91 | f64 | 2.84e-14 |
| cuda | row:rms_norm_affine | float64 | — | 84.783 | — | rms_norm_affine_f32; converted through the host | 6.37e-7 | — | — | — | — | 0.652 | 129.96 | f64 | 2.22e-14 |
| cuda | row:rms_norm_bwd | float64 | — | — | — | — | — | — | — | — | — | 1.135 | — | f64 | 5.88e-15 |
| cuda | row:layer_norm | float64 | — | 83.852 | — | layernorm_fused_f32; converted through the host | 8.31e-7 | — | — | — | — | 0.968 | 86.61 | f64 | 1.15e-14 |
| cuda | row:layer_norm_bwd | float64 | — | — | — | — | — | — | — | — | — | 1.742 | — | f64 | 2.78e-15 |
| cuda | row:weight_norm | float64 | — | 83.080 | — | weight_norm_f32; converted through the host | 9.76e-9 | — | — | — | — | 0.492 | 168.76 | f64 | 3.12e-16 |
| cuda | row:rope | float64 | — | — | — | — | — | — | — | — | — | 3.887 | — | f64 | 2.51e-12 |
| cuda | row:scale_mask_softmax | float64 | — | — | — | — | — | — | — | — | — | 3.183 | — | f64 | 5.55e-17 |
| cuda | row:add_rms_norm | float64 | — | 168.423 | — | add_rms_norm_affine_f32; converted through the host | 1.11e-6 | — | — | — | — | 1.364 | 123.49 | f64 | 2.49e-14 |
| cuda | row:add_layer_norm | float64 | — | — | — | — | — | — | — | — | — | 1.150 | — | f64 | 1.42e-14 |
| cuda | row:cross_entropy_bwd | float64 | — | — | — | — | — | — | — | — | — | 3.061 | — | f64 | 1.11e-16 |
| cuda | norm_params_bwd:rms | float64 | — | — | — | — | — | — | — | — | — | 0.991 | — | f64 | 7.18e-13 |
| cuda | norm_params_bwd:layer | float64 | — | — | — | — | — | — | — | — | — | 1.214 | — | f64 | 5.12e-13 |
| cuda | layout:strided | float64 | — | — | — | — | — | — | — | — | — | 0.698 | — | f64 | 0e+0 |
| cuda | layout:one_hot | float64 | — | — | — | — | — | — | — | — | — | 0.176 | — | f64 | 0e+0 |
| cuda | layout:gather_rows | float64 | — | — | — | — | — | — | — | — | — | 0.341 | — | f64 | 0e+0 |
| cuda | layout:gather_cols | float64 | — | — | — | — | — | — | — | — | — | 0.383 | — | f64 | 0e+0 |
| cuda | layout:scatter_add_rows | float64 | — | — | — | — | — | — | — | — | — | 0.460 | — | f64 | 0e+0 |
| cuda | layout:im2col | float64 | — | — | — | — | — | — | — | — | — | 0.116 | — | f64 | 0e+0 |
| cuda | layout:col2im | float64 | — | — | — | — | — | — | — | — | — | 0.105 | — | f64 | 0e+0 |
| cuda | layout:interpolate_nearest | float64 | — | — | — | — | — | — | — | — | — | 0.241 | — | f64 | 0e+0 |
| cuda | layout:interpolate_bilinear | float64 | — | — | — | — | — | — | — | — | — | 2.189 | — | f64 | 0e+0 |
| cuda | layout:shift | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | layout:pad2d | float64 | — | — | — | — | — | — | — | — | — | 0.358 | — | f64 | 0e+0 |
| cuda | layout:maxpool | float64 | — | — | — | — | — | — | — | — | — | 0.023 | — | f64 | 0e+0 |
| cuda | layout:avgpool | float64 | — | — | — | — | — | — | — | — | — | 0.049 | — | f64 | 0e+0 |
| cuda | layout:maxpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.155 | — | f64 | 0e+0 |
| cuda | layout:avgpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.253 | — | f64 | 0e+0 |
| cuda | layout:adaptive_maxpool | float64 | — | — | — | — | — | — | — | — | — | 0.134 | — | f64 | 0e+0 |
| cuda | layout:adaptive_avgpool | float64 | — | — | — | — | — | — | — | — | — | 0.104 | — | f64 | 0e+0 |
| cuda | layout:depthwise_conv | float64 | — | — | — | — | — | — | — | — | — | 0.124 | — | f64 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 34.208 | — | f64 | 0e+0 |
| cuda | layout:adaptive_avgpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 0.231 | — | f64 | 0e+0 |
| cuda | matmul_nt | float64 | — | — | — | — | — | — | — | — | — | 4.396 | — | f64x | 0e+0 |
| cuda | matmul_tn | float64 | — | — | — | — | — | — | — | — | — | 4.377 | — | f64x | 0e+0 |
| cuda | matmul_tt | float64 | — | — | — | — | — | — | — | — | — | 4.453 | — | f64x | 0e+0 |
| cuda | matmul_batched | float64 | — | — | — | — | — | — | — | — | — | 6.592 | — | f64 | 6.04e-14 |
| cuda | matmul_bwd | float64 | — | — | — | — | — | — | — | — | — | 8.949 | — | f64x | 0e+0 |
| cuda | attention | float64 | — | — | — | — | — | — | — | — | — | 60.294 | — | f64 | 3.82e-16 |
| cuda | attention_causal | float64 | — | — | — | — | — | — | — | — | — | 32.421 | — | f64 | 4.51e-16 |
| cuda | attention_fused | float64 | — | — | — | — | — | — | — | — | — | 61.653 | — | f64 | 4.58e-16 |
| cuda | attention_fused_causal | float64 | — | — | — | — | — | — | — | — | — | 32.386 | — | f64 | 4.51e-16 |
| cuda | attention_decode | float64 | — | — | — | — | — | — | — | — | — | 1.156 | — | f64 | 2.15e-16 |
| cuda | attention_bwd | float64 | — | — | — | — | — | — | — | — | — | 31.879 | — | f64 | 1.11e-16 |
| cuda | optimizer:sgd_momentum | float64 | — | — | — | — | — | — | — | — | — | 1.575 | — | f64; + restore of p, m, v | 1.11e-16 |
| cuda | optimizer:adam | float64 | — | — | — | — | — | — | — | — | — | 2.342 | — | f64; + restore of p, m, v | 1.11e-16 |
| cuda | optimizer:adamw | float64 | — | — | — | — | — | — | — | — | — | 2.448 | — | f64; + restore of p, m, v | 1.11e-16 |
| cuda | conv | float64 | — | — | — | — | — | — | — | — | — | 2.551 | — | f64x | 0e+0 |
| cuda | conv_bwd | float64 | — | — | — | — | — | — | — | — | — | 15.882 | — | f64 | 0e+0 |
| cuda | lstm_step | float64 | — | — | — | — | — | — | — | — | — | 6.120 | — | f64 | 8.05e-16 |
| cuda | gru_step | float64 | — | — | — | — | — | — | — | — | — | 6.036 | — | f64 | 1.21e-15 |
| cuda | lstm_step_bwd | float64 | — | — | — | — | — | — | — | — | — | 19.450 | — | f64 | 3.44e-15 |
| cuda | gru_step_bwd | float64 | — | — | — | — | — | — | — | — | — | 16.438 | — | f64 | 1.44e-15 |
| cuda | lstm_seq | float64 | — | — | — | — | — | — | — | — | — | 23.812 | — | f64 | 4.09e-16 |
| cuda | lstm_seq_bwd | float64 | — | — | — | — | — | — | — | — | — | 113.888 | — | f64 | 4.44e-16 |
| cuda | gru_seq | float64 | — | — | — | — | — | — | — | — | — | 22.446 | — | f64 | 8.33e-16 |
| cuda | gru_seq_bwd | float64 | — | — | — | — | — | — | — | — | — | 98.492 | — | f64 | 1.17e-15 |
| cuda | topk_rows | float64 | — | — | — | — | — | — | — | — | — | 20.735 | — | f64 | 0e+0 |
| cuda | pick_rows | float64 | — | — | — | — | — | — | — | — | — | 0.009 | — | f64 | 0e+0 |
| cuda | sample_rows | float64 | — | — | — | — | — | — | — | — | — | 7.621 | — | f64 | 0e+0 |
| cuda | route | s32 | — | — | — | — | — | — | — | — | — | 0.275 | — | s32 | 0e+0 |
| cuda | route_wide | s32 | — | — | — | — | — | — | — | — | — | 0.186 | — | s32 | 0e+0 |
| cuda | cross_entropy | float64 | — | — | — | — | — | — | — | — | — | 175.411 | — | f64 (log_softmax, pick, neg) | 1.78e-14 |
| cuda | ctc | float64 | — | — | — | — | — | — | — | — | — | 2.398 | — | f64 | 4.55e-13 |
| cuda | ctc_bwd | float64 | — | — | — | — | — | — | — | — | — | 5.242 | — | f64 | 6.8e-13 |
| cuda | softmax | float64 | 4096x4096 | 79.646 | — | softmax_f32; converted through the host | 5.29e-10 | 0.396 | 201.33 | softmax_f64 | 1.88e-10 | 1.746 | 45.62 | f64 | 1.13e-17 |
| cuda | rms_norm | float64 | 4096x4096 | 89.619 | — | rms_norm_f32; converted through the host | 3.1e-7 | 0.372 | 241.17 | rms_norm_f64 | 1.24e-14 | 0.552 | 162.35 | f64 | 1.07e-14 |
| cuda | add | float32 | 16777216 | 0.262 | — | add_f32 | 5.96e-8 | 0.262 | 1.00 | add_f32 | 5.96e-8 | 0.263 | 1.00 | f32 | 5.96e-8 |
| cuda | matmul | float32 | 2048x2048x2048 | 1.042 | — | gemm64_f32 | 1.62e-4 | 0.839 | 1.24 | gemm128_f32 | 1.62e-4 | 0.685 | 1.52 | f32 | 1.62e-4 |
| cuda | map:copy | float32 | — | — | — | — | — | — | — | — | — | 0.182 | — | f32 | 0e+0 |
| cuda | map:fill | float32 | — | 0.084 | — | fill_f32 | 0e+0 | — | — | — | — | 0.084 | 0.99 | f32 | 0e+0 |
| cuda | map:neg | float32 | — | — | — | — | — | — | — | — | — | 0.180 | — | f32 | 0e+0 |
| cuda | map:abs | float32 | — | 0.179 | — | abs_f32 | 0e+0 | — | — | — | — | 0.179 | 1.00 | f32 | 0e+0 |
| cuda | map:sign | float32 | — | 0.181 | — | sign_f32 | 0e+0 | — | — | — | — | 0.181 | 1.00 | f32 | 0e+0 |
| cuda | map:square | float32 | — | 0.176 | — | square_f32 | 4.77e-7 | — | — | — | — | 0.181 | 0.97 | f32 | 4.77e-7 |
| cuda | map:relu | float32 | — | 0.181 | — | relu_f32 | 0e+0 | — | — | — | — | 0.181 | 1.00 | f32 | 0e+0 |
| cuda | map:add_scalar | float32 | — | — | — | — | — | — | — | — | — | 0.182 | — | f32 | 2.38e-7 |
| cuda | map:scale | float32 | — | — | — | — | — | — | — | — | — | 0.181 | — | f32 | 4.77e-7 |
| cuda | map:affine | float32 | — | 0.181 | — | affine_f32 | 4.77e-7 | — | — | — | — | 0.179 | 1.01 | f32 | 4.77e-7 |
| cuda | map:clamp | float32 | — | 0.180 | — | clamp_f32 | 0e+0 | — | — | — | — | 0.181 | 0.99 | f32 | 0e+0 |
| cuda | map:add | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 2.38e-7 |
| cuda | map:sub | float32 | — | 0.262 | — | sub_f32 | 2.38e-7 | — | — | — | — | 0.263 | 1.00 | f32 | 2.38e-7 |
| cuda | map:mul | float32 | — | 0.501 | — | mul_f32 | 4.77e-7 | — | — | — | — | 0.263 | 1.91 | f32 | 4.77e-7 |
| cuda | map:div | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 1.19e-7 |
| cuda | map:max | float32 | — | 0.262 | — | max_f32 | 0e+0 | — | — | — | — | 0.262 | 1.00 | f32 | 0e+0 |
| cuda | map:min | float32 | — | 0.261 | — | min_f32 | 0e+0 | — | — | — | — | 0.262 | 1.00 | f32 | 0e+0 |
| cuda | map:axpy | float32 | — | 0.438 | — | axpy_f32; + copy of y | 4.77e-7 | — | — | — | — | 0.262 | 1.67 | f32 | 4.77e-7 |
| cuda | map:muladd | float32 | — | 0.521 | — | muladd_f32; + copy of z | 9.54e-7 | — | — | — | — | 0.343 | 1.52 | f32 | 9.54e-7 |
| cuda | map:where | float32 | — | 0.262 | — | where_f32 | 0e+0 | — | — | — | — | 0.262 | 1.00 | f32 | 0e+0 |
| cuda | map:square_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.263 | — | f32 | 9.54e-7 |
| cuda | map:abs_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 0e+0 |
| cuda | map:relu_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 0e+0 |
| cuda | map:clamp_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 0e+0 |
| cuda | map:where_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.502 | — | f32 | 0e+0 |
| cuda | map:sqrt | float32 | — | 0.180 | — | sqrt_f32 | 5.96e-8 | — | — | — | — | 0.177 | 1.02 | f32 | 5.96e-8 |
| cuda | map:inv_sqrt | float32 | — | 0.174 | — | inv_sqrt_f32 | 1.37e-7 | — | — | — | — | 0.180 | 0.97 | f32 | 1.37e-7 |
| cuda | map:exp | float32 | — | 0.181 | — | exp_f32 | 1.36e-5 | — | — | — | — | 0.178 | 1.02 | f32 | 3.42e-6 |
| cuda | map:expm1 | float32 | — | — | — | — | — | — | — | — | — | 0.344 | — | f32 | 6.84e-8 |
| cuda | map:log | float32 | — | 0.181 | — | log_f32 | 1.74e-7 | — | — | — | — | 0.181 | 1.00 | f32 | 1.26e-7 |
| cuda | map:sin | float32 | — | 0.181 | — | sin_f32 | 7.09e-8 | — | — | — | — | 0.181 | 1.00 | f32 | 7.99e-8 |
| cuda | map:cos | float32 | — | 0.181 | — | cos_f32 | 7.07e-8 | — | — | — | — | 0.181 | 1.00 | f32 | 7.96e-8 |
| cuda | map:tanh | float32 | — | 0.181 | — | tanh_f32 | 2.77e-7 | — | — | — | — | 0.343 | 0.53 | f32 | 1.49e-7 |
| cuda | map:sigmoid | float32 | — | 0.181 | — | sigmoid_f32 | 9.45e-8 | — | — | — | — | 0.181 | 1.00 | f32 | 8.77e-8 |
| cuda | map:erf | float32 | — | — | — | — | — | — | — | — | — | 0.612 | — | f32 | 1.45e-7 |
| cuda | map:gelu | float32 | — | 0.176 | — | gelu_f32 | 3.13e-7 | — | — | — | — | 0.176 | 1.00 | f32 | 4.28e-7 |
| cuda | map:gelu_erf | float32 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 | 2.37e-7 |
| cuda | map:silu | float32 | — | 0.177 | — | silu_f32 | 4.57e-7 | — | — | — | — | 0.181 | 0.98 | f32 | 4.58e-7 |
| cuda | map:leaky_relu | float32 | — | 0.181 | — | leaky_relu_f32 | 1.79e-8 | — | — | — | — | 0.181 | 1.00 | f32 | 1.79e-8 |
| cuda | map:pow | float32 | — | — | — | — | — | — | — | — | — | 0.177 | — | f32 | 9.47e-6 |
| cuda | map:dropout | float32 | — | — | — | — | — | — | — | — | — | 0.263 | — | f32 | 2.38e-7 |
| cuda | map:gated_silu | float32 | — | — | — | — | — | — | — | — | — | 0.263 | — | f32 | 2.24e-6 |
| cuda | map:gated_gelu | float32 | — | — | — | — | — | — | — | — | — | 0.502 | — | f32 | 2.04e-6 |
| cuda | map:sqrt_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.261 | — | f32 | 2.29e-7 |
| cuda | map:inv_sqrt_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.261 | — | f32 | 1.04e-6 |
| cuda | map:exp_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 1.95e-5 |
| cuda | map:log_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 2.38e-7 |
| cuda | map:sin_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 4.04e-7 |
| cuda | map:cos_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 4e-7 |
| cuda | map:tanh_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 9.3e-7 |
| cuda | map:sigmoid_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.502 | — | f32 | 2.31e-7 |
| cuda | map:silu_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.263 | — | f32 | 9.79e-7 |
| cuda | map:gelu_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 8.92e-7 |
| cuda | map:gelu_erf_bwd | float32 | — | — | — | — | — | — | — | — | — | 1.008 | — | f32 | 8.35e-7 |
| cuda | map:leaky_relu_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 1.79e-8 |
| cuda | map:pow_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 2.48e-5 |
| cuda | map:gated_silu_bwd_x | float32 | — | — | — | — | — | — | — | — | — | 0.345 | — | f32 | 4.11e-6 |
| cuda | map:gated_silu_bwd_y | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 2.12e-6 |
| cuda | map:gated_gelu_bwd_x | float32 | — | — | — | — | — | — | — | — | — | 0.345 | — | f32 | 2.92e-6 |
| cuda | map:gated_gelu_bwd_y | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 2.07e-6 |
| cuda | sum:whole | float32 | — | 0.102 | — | f32 | 1.95e-4 | — | — | — | — | 0.093 | 1.10 | f32 | 5.98e-4 |
| cuda | sum:rows | float32 | — | 0.089 | — | f32 | 3.58e-5 | — | — | — | — | 0.089 | 1.00 | f32 | 1.77e-5 |
| cuda | sum:cols | float32 | — | 0.487 | — | f32; + zero of y | 3.28e-4 | — | — | — | — | 0.175 | 2.78 | f32 | 3.81e-5 |
| cuda | sumsq:rows | float32 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 1.85e-4 |
| cuda | max:whole | float32 | — | — | — | — | — | — | — | — | — | 0.093 | — | f32 | 0e+0 |
| cuda | max:rows | float32 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 0e+0 |
| cuda | max:cols | float32 | — | — | — | — | — | — | — | — | — | 0.096 | — | f32 | 0e+0 |
| cuda | min:rows | float32 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 0e+0 |
| cuda | argmax:rows | float32 | — | 0.091 | — | f32 index | 0e+0 | — | — | — | — | 0.089 | 1.03 | s32 index | 0e+0 |
| cuda | argmax:cols | float32 | — | — | — | — | — | — | — | — | — | 0.502 | — | s32 index | 0e+0 |
| cuda | maprow:add | float32 | — | — | — | — | — | — | — | — | — | 0.175 | — | f32 | 5.96e-8 |
| cuda | maprow:mul | float32 | — | — | — | — | — | — | — | — | — | 0.181 | — | f32 | 2.98e-8 |
| cuda | transpose | float32 | — | 0.350 | — | transpose_f32 | 0e+0 | — | — | — | — | 0.192 | 1.83 | f32 | 0e+0 |
| cuda | gemv | float32 | — | — | — | — | — | — | — | — | — | 0.337 | — | f32 | 3.96e-5 |
| cuda | gemv_t | float32 | — | — | — | — | — | — | — | — | — | 0.650 | — | f32 | 4.99e-5 |
| cuda | row:log_softmax | float32 | — | 0.183 | — | softmax_f32 | 1.66e-6 | — | — | — | — | 0.179 | 1.02 | f32 | 1.27e-6 |
| cuda | row:softmax_bwd | float32 | — | 0.415 | — | softmax_bwd_f32; + zero of dx | 2.04e-5 | — | — | — | — | 0.261 | 1.59 | f32 | 1.2e-5 |
| cuda | row:log_softmax_bwd | float32 | — | 0.415 | — | softmax_bwd_f32; + zero of dx | 2.45e-5 | — | — | — | — | 0.262 | 1.58 | f32 | 2.27e-5 |
| cuda | row:rms_norm_affine | float32 | — | 0.183 | — | rms_norm_affine_f32 | 6.37e-7 | — | — | — | — | 0.179 | 1.02 | f32 | 6.2e-7 |
| cuda | row:rms_norm_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 1.72e-7 |
| cuda | row:layer_norm | float32 | — | 0.183 | — | layernorm_fused_f32 | 8.31e-7 | — | — | — | — | 0.177 | 1.04 | f32 | 7.26e-7 |
| cuda | row:layer_norm_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.261 | — | f32 | 1.75e-7 |
| cuda | row:weight_norm | float32 | — | 0.183 | — | weight_norm_f32 | 9.76e-9 | — | — | — | — | 0.179 | 1.02 | f32 | 7.22e-9 |
| cuda | row:rope | float32 | — | — | — | — | — | — | — | — | — | 0.181 | — | f32 | 1.35e-3 |
| cuda | row:scale_mask_softmax | float32 | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 | 4.81e-8 |
| cuda | row:add_rms_norm | float32 | — | 0.364 | — | add_rms_norm_affine_f32 | 1.11e-6 | — | — | — | — | 0.357 | 1.02 | f32 | 1.12e-6 |
| cuda | row:add_layer_norm | float32 | — | — | — | — | — | — | — | — | — | 0.359 | — | f32 | 1.13e-6 |
| cuda | row:cross_entropy_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.177 | — | f32 | 2.98e-8 |
| cuda | norm_params_bwd:rms | float32 | — | — | — | — | — | — | — | — | — | 0.259 | — | f32 | 3.16e-5 |
| cuda | norm_params_bwd:layer | float32 | — | — | — | — | — | — | — | — | — | 0.264 | — | f32 | 3.51e-5 |
| cuda | layout:strided | float32 | — | — | — | — | — | — | — | — | — | 0.450 | — | f32 | 0e+0 |
| cuda | layout:one_hot | float32 | — | — | — | — | — | — | — | — | — | 0.134 | — | f32 | 0e+0 |
| cuda | layout:gather_rows | float32 | — | — | — | — | — | — | — | — | — | 0.196 | — | f32 | 0e+0 |
| cuda | layout:gather_cols | float32 | — | — | — | — | — | — | — | — | — | 0.255 | — | f32 | 0e+0 |
| cuda | layout:scatter_add_rows | float32 | — | — | — | — | — | — | — | — | — | 0.309 | — | f32 | 4.77e-7 |
| cuda | layout:im2col | float32 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 0e+0 |
| cuda | layout:col2im | float32 | — | — | — | — | — | — | — | — | — | 0.047 | — | f32 | 1.08e-6 |
| cuda | layout:interpolate_nearest | float32 | — | — | — | — | — | — | — | — | — | 0.151 | — | f32 | 0e+0 |
| cuda | layout:interpolate_bilinear | float32 | — | — | — | — | — | — | — | — | — | 0.166 | — | f32 | 1.19e-7 |
| cuda | layout:shift | float32 | — | — | — | — | — | — | — | — | — | 0.344 | — | f32 | 0e+0 |
| cuda | layout:pad2d | float32 | — | — | — | — | — | — | — | — | — | 0.181 | — | f32 | 0e+0 |
| cuda | layout:maxpool | float32 | — | — | — | — | — | — | — | — | — | 0.021 | — | f32 | 0e+0 |
| cuda | layout:avgpool | float32 | — | — | — | — | — | — | — | — | — | 0.021 | — | f32 | 7.45e-8 |
| cuda | layout:maxpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.122 | — | f32 | 0e+0 |
| cuda | layout:avgpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.110 | — | f32 | 0e+0 |
| cuda | layout:adaptive_maxpool | float32 | — | — | — | — | — | — | — | — | — | 0.079 | — | f32 | 0e+0 |
| cuda | layout:adaptive_avgpool | float32 | — | — | — | — | — | — | — | — | — | 0.079 | — | f32 | 7.4e-8 |
| cuda | layout:depthwise_conv | float32 | — | — | — | — | — | — | — | — | — | 0.095 | — | f32 | 6.56e-7 |
| cuda | layout:adaptive_maxpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 7.161 | — | f32 | 5.96e-8 |
| cuda | layout:adaptive_avgpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 0.163 | — | f32 | 2.08e-10 |
| cuda | matmul_nt | float32 | — | — | — | — | — | — | — | — | — | 0.841 | — | f32 | 1.57e-4 |
| cuda | matmul_tn | float32 | — | — | — | — | — | — | — | — | — | 0.843 | — | f32 | 1.51e-4 |
| cuda | matmul_tt | float32 | — | — | — | — | — | — | — | — | — | 0.892 | — | f32 | 1.54e-4 |
| cuda | matmul_batched | float32 | — | — | — | — | — | — | — | — | — | 0.969 | — | f32 | 3.9e-5 |
| cuda | matmul_bwd | float32 | — | — | — | — | — | — | — | — | — | 1.687 | — | f32 | 1.54e-4 |
| cuda | attention | float32 | — | — | — | — | — | — | — | — | — | 3.820 | — | f32 | 1.39e-7 |
| cuda | attention_causal | float32 | — | — | — | — | — | — | — | — | — | 2.122 | — | f32 | 1.46e-7 |
| cuda | attention_fused | float32 | — | — | — | — | — | — | — | — | — | 3.803 | — | f32 | 1.39e-7 |
| cuda | attention_fused_causal | float32 | — | — | — | — | — | — | — | — | — | 2.112 | — | f32 | 1.46e-7 |
| cuda | attention_decode | float32 | — | — | — | — | — | — | — | — | — | 0.611 | — | f32 | 7.05e-8 |
| cuda | attention_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.924 | — | f32 | 1.47e-7 |
| cuda | optimizer:sgd_momentum | float32 | — | — | — | — | — | — | — | — | — | 0.789 | — | f32; + restore of p, m, v | 5.98e-8 |
| cuda | optimizer:adam | float32 | — | — | — | — | — | — | — | — | — | 1.143 | — | f32; + restore of p, m, v | 5.95e-8 |
| cuda | optimizer:adamw | float32 | — | — | — | — | — | — | — | — | — | 1.143 | — | f32; + restore of p, m, v | 7.29e-8 |
| cuda | conv | float32 | — | — | — | — | — | — | — | — | — | 0.142 | — | f32 | 4.36e-5 |
| cuda | conv_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.183 | — | f32 | 1.56e-5 |
| cuda | lstm_step | float32 | — | — | — | — | — | — | — | — | — | 1.068 | — | f32 | 5.76e-7 |
| cuda | gru_step | float32 | — | — | — | — | — | — | — | — | — | 0.803 | — | f32 | 7.78e-7 |
| cuda | lstm_step_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.699 | — | f32 | 1.81e-6 |
| cuda | gru_step_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.093 | — | f32 | 8.33e-7 |
| cuda | lstm_seq | float32 | — | — | — | — | — | — | — | — | — | 2.744 | — | f32 | 2.76e-7 |
| cuda | lstm_seq_bwd | float32 | — | — | — | — | — | — | — | — | — | 26.048 | — | f32 | 9.1e-7 |
| cuda | gru_seq | float32 | — | — | — | — | — | — | — | — | — | 2.543 | — | f32 | 5.31e-7 |
| cuda | gru_seq_bwd | float32 | — | — | — | — | — | — | — | — | — | 12.485 | — | f32 | 6.81e-7 |
| cuda | topk_rows | float32 | — | — | — | — | — | — | — | — | — | 6.873 | — | f32 | 0e+0 |
| cuda | pick_rows | float32 | — | — | — | — | — | — | — | — | — | 0.009 | — | f32 | 0e+0 |
| cuda | sample_rows | float32 | — | — | — | — | — | — | — | — | — | 1.702 | — | f32 | 1e+0 |
| cuda | cross_entropy | float32 | — | — | — | — | — | — | — | — | — | 95.124 | — | f32 (log_softmax, pick, neg) | 1.68e-6 |
| cuda | ctc | float32 | — | — | — | — | — | — | — | — | — | 0.524 | — | f32 | 1.36e-3 |
| cuda | ctc_bwd | float32 | — | — | — | — | — | — | — | — | — | 1.179 | — | f32 | 1.67e-3 |
| cuda | softmax | float32 | 4096x4096 | 0.183 | — | softmax_f32 | 5.29e-10 | 0.181 | 1.01 | softmax_f32 | 5.11e-10 | 0.181 | 1.01 | f32 | 4.52e-10 |
| cuda | rms_norm | float32 | 4096x4096 | 0.184 | — | rms_norm_f32 | 3.1e-7 | 0.182 | 1.01 | rms_norm_f32 | 2.93e-7 | 0.180 | 1.02 | f32 | 2.97e-7 |
| cuda | add | float16 | 16777216 | 0.661 | — | add_f32; converted on the device | 4.88e-4 | 0.139 | 4.74 | add_f16 | 4.88e-4 | 0.123 | 5.36 | f32 | 4.88e-4 |
| cuda | matmul | float16 | 2048x2048x2048 | 1.028 | — | gemm128_f32; converted on the device | 3.09e-2 | 0.657 | 1.56 | gemm128_f16 | 1.51e+0 | 0.243 | 4.24 | f16mma | 3.09e-2 |
| cuda | map:copy | float16 | — | — | — | — | — | — | — | — | — | 0.082 | — | f32 | 0e+0 |
| cuda | map:fill | float16 | — | 0.208 | — | fill_f32; converted on the device | 0e+0 | — | — | — | — | 0.029 | 7.27 | f32 | 0e+0 |
| cuda | map:neg | float16 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 0e+0 |
| cuda | map:abs | float16 | — | 0.442 | — | abs_f32; converted on the device | 0e+0 | — | — | — | — | 0.085 | 5.20 | f32 | 0e+0 |
| cuda | map:sign | float16 | — | 0.441 | — | sign_f32; converted on the device | 0e+0 | — | — | — | — | 0.085 | 5.21 | f32 | 0e+0 |
| cuda | map:square | float16 | — | 0.441 | — | square_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.085 | 5.19 | f32 | 3.91e-3 |
| cuda | map:relu | float16 | — | 0.440 | — | relu_f32; converted on the device | 0e+0 | — | — | — | — | 0.084 | 5.21 | f32 | 0e+0 |
| cuda | map:add_scalar | float16 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 1.95e-3 |
| cuda | map:scale | float16 | — | — | — | — | — | — | — | — | — | 0.085 | — | f32 | 3.91e-3 |
| cuda | map:affine | float16 | — | 0.441 | — | affine_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.087 | 5.05 | f32 | 3.91e-3 |
| cuda | map:clamp | float16 | — | 0.439 | — | clamp_f32; converted on the device | 0e+0 | — | — | — | — | 0.085 | 5.14 | f32 | 0e+0 |
| cuda | map:add | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 1.95e-3 |
| cuda | map:sub | float16 | — | 0.661 | — | sub_f32; converted on the device | 1.95e-3 | — | — | — | — | 0.125 | 5.28 | f32 | 1.95e-3 |
| cuda | map:mul | float16 | — | 0.661 | — | mul_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.125 | 5.28 | f32 | 3.91e-3 |
| cuda | map:div | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 9.76e-4 |
| cuda | map:max | float16 | — | 0.663 | — | max_f32; converted on the device | 0e+0 | — | — | — | — | 0.126 | 5.25 | f32 | 0e+0 |
| cuda | map:min | float16 | — | 0.660 | — | min_f32; converted on the device | 0e+0 | — | — | — | — | 0.126 | 5.25 | f32 | 0e+0 |
| cuda | map:axpy | float16 | — | 0.838 | — | axpy_f32; converted on the device; + copy of y | 3.91e-3 | — | — | — | — | 0.126 | 6.66 | f32 | 3.91e-3 |
| cuda | map:muladd | float16 | — | 1.055 | — | muladd_f32; converted on the device; + copy of z | 7.81e-3 | — | — | — | — | 0.169 | 6.24 | f32 | 7.81e-3 |
| cuda | map:where | float16 | — | 0.798 | — | where_f32; converted on the device | 0e+0 | — | — | — | — | 0.169 | 4.72 | f32 | 0e+0 |
| cuda | map:square_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 7.81e-3 |
| cuda | map:abs_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 0e+0 |
| cuda | map:relu_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 0e+0 |
| cuda | map:clamp_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.238 | — | f32 | 0e+0 |
| cuda | map:where_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 | 0e+0 |
| cuda | map:sqrt | float16 | — | 0.848 | — | sqrt_f32; converted on the device | 4.88e-4 | — | — | — | — | 0.086 | 9.86 | f32 | 4.88e-4 |
| cuda | map:inv_sqrt | float16 | — | 0.441 | — | inv_sqrt_f32; converted on the device | 4.88e-4 | — | — | — | — | 0.089 | 4.96 | f32 | 4.88e-4 |
| cuda | map:exp | float16 | — | 0.442 | — | exp_f32; converted on the device | 1.55e-2 | — | — | — | — | 0.086 | 5.11 | f32 | 1.55e-2 |
| cuda | map:expm1 | float16 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 2.44e-4 |
| cuda | map:log | float16 | — | 0.442 | — | log_f32; converted on the device | 4.88e-4 | — | — | — | — | 0.095 | 4.64 | f32 | 4.88e-4 |
| cuda | map:sin | float16 | — | 0.444 | — | sin_f32; converted on the device | 2.44e-4 | — | — | — | — | 0.086 | 5.16 | f32 | 2.44e-4 |
| cuda | map:cos | float16 | — | 0.444 | — | cos_f32; converted on the device | 2.44e-4 | — | — | — | — | 0.086 | 5.16 | f32 | 2.44e-4 |
| cuda | map:tanh | float16 | — | 0.443 | — | tanh_f32; converted on the device | 2.44e-4 | — | — | — | — | 0.095 | 4.66 | f32 | 2.44e-4 |
| cuda | map:sigmoid | float16 | — | 0.444 | — | sigmoid_f32; converted on the device | 2.44e-4 | — | — | — | — | 0.094 | 4.70 | f32 | 2.44e-4 |
| cuda | map:erf | float16 | — | — | — | — | — | — | — | — | — | 0.612 | — | f32 | 2.44e-4 |
| cuda | map:gelu | float16 | — | 0.442 | — | gelu_f32; converted on the device | 9.73e-4 | — | — | — | — | 0.095 | 4.68 | f32 | 9.73e-4 |
| cuda | map:gelu_erf | float16 | — | — | — | — | — | — | — | — | — | 0.985 | — | f32 | 9.76e-4 |
| cuda | map:silu | float16 | — | 0.444 | — | silu_f32; converted on the device | 9.75e-4 | — | — | — | — | 0.095 | 4.68 | f32 | 9.75e-4 |
| cuda | map:leaky_relu | float16 | — | 0.443 | — | leaky_relu_f32; converted on the device | 9.77e-5 | — | — | — | — | 0.082 | 5.38 | f32 | 9.77e-5 |
| cuda | map:pow | float16 | — | — | — | — | — | — | — | — | — | 0.102 | — | f32 | 7.8e-3 |
| cuda | map:dropout | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 1.95e-3 |
| cuda | map:gated_silu | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 3.91e-3 |
| cuda | map:gated_gelu | float16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 3.91e-3 |
| cuda | map:sqrt_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 | 9.77e-4 |
| cuda | map:inv_sqrt_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.238 | — | f32 | 1.95e-3 |
| cuda | map:exp_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.234 | — | f32 | 6.25e-2 |
| cuda | map:log_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 1.95e-3 |
| cuda | map:sin_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.238 | — | f32 | 9.77e-4 |
| cuda | map:cos_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 9.77e-4 |
| cuda | map:tanh_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 9.77e-4 |
| cuda | map:sigmoid_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.238 | — | f32 | 2.44e-4 |
| cuda | map:silu_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 | 1.95e-3 |
| cuda | map:gelu_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.240 | — | f32 | 1.95e-3 |
| cuda | map:gelu_erf_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.010 | — | f32 | 1.95e-3 |
| cuda | map:leaky_relu_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 9.77e-5 |
| cuda | map:pow_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 3.13e-2 |
| cuda | map:gated_silu_bwd_x | float16 | — | — | — | — | — | — | — | — | — | 0.170 | — | f32 | 7.81e-3 |
| cuda | map:gated_silu_bwd_y | float16 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 | 3.91e-3 |
| cuda | map:gated_gelu_bwd_x | float16 | — | — | — | — | — | — | — | — | — | 0.321 | — | f32 | 3.91e-3 |
| cuda | map:gated_gelu_bwd_y | float16 | — | — | — | — | — | — | — | — | — | 0.236 | — | f32 | 3.91e-3 |
| cuda | sum:whole | float16 | — | 0.398 | — | f32; converted on the device | 9.08e-5 | — | — | — | — | 0.025 | 15.97 | f32 | 3.13e-5 |
| cuda | sum:rows | float16 | — | 0.360 | — | f32; converted on the device | 1.04e-5 | — | — | — | — | 0.023 | 16.01 | f32 | 1.16e-5 |
| cuda | sum:cols | float16 | — | 0.533 | — | f32; converted on the device; + zero of y | 6.44e-5 | — | — | — | — | 0.029 | 18.62 | f32 | 2.87e-5 |
| cuda | sumsq:rows | float16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 2.2e-4 |
| cuda | max:whole | float16 | — | — | — | — | — | — | — | — | — | 0.025 | — | f32 | 0e+0 |
| cuda | max:rows | float16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 0e+0 |
| cuda | max:cols | float16 | — | — | — | — | — | — | — | — | — | 0.029 | — | f32 | 0e+0 |
| cuda | min:rows | float16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 0e+0 |
| cuda | argmax:rows | float16 | — | 0.230 | — | f32 index; converted on the device | 0e+0 | — | — | — | — | 0.027 | 8.55 | s32 index | 0e+0 |
| cuda | argmax:cols | float16 | — | — | — | — | — | — | — | — | — | 0.189 | — | s32 index | 0e+0 |
| cuda | maprow:add | float16 | — | — | — | — | — | — | — | — | — | 0.086 | — | f32 | 4.88e-4 |
| cuda | maprow:mul | float16 | — | — | — | — | — | — | — | — | — | 0.086 | — | f32 | 2.44e-4 |
| cuda | transpose | float16 | — | 0.828 | — | transpose_f32; converted on the device | 0e+0 | — | — | — | — | 0.091 | 9.11 | f32 | 0e+0 |
| cuda | gemv | float16 | — | — | — | — | — | — | — | — | — | 0.178 | — | f32 | 3.12e-2 |
| cuda | gemv_t | float16 | — | — | — | — | — | — | — | — | — | 0.185 | — | f32 | 5.38e-2 |
| cuda | row:log_softmax | float16 | — | 0.441 | — | softmax_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.079 | 5.62 | f32 | 3.91e-3 |
| cuda | row:softmax_bwd | float16 | — | 0.816 | — | softmax_bwd_f32; converted on the device; + zero of dx | 3.12e-2 | — | — | — | — | 0.123 | 6.61 | f32 | 3.12e-2 |
| cuda | row:log_softmax_bwd | float16 | — | 0.815 | — | softmax_bwd_f32; converted on the device; + zero of dx | 3.13e-2 | — | — | — | — | 0.124 | 6.58 | f32 | 3.13e-2 |
| cuda | row:rms_norm_affine | float16 | — | 0.445 | — | rms_norm_affine_f32; converted on the device | 9.77e-4 | — | — | — | — | 0.075 | 5.91 | f32 | 9.77e-4 |
| cuda | row:rms_norm_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 2.44e-4 |
| cuda | row:layer_norm | float16 | — | 0.446 | — | layernorm_fused_f32; converted on the device | 1.95e-3 | — | — | — | — | 0.083 | 5.41 | f32 | 1.95e-3 |
| cuda | row:layer_norm_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.125 | — | f32 | 2.44e-4 |
| cuda | row:weight_norm | float16 | — | 0.445 | — | weight_norm_f32; converted on the device | 1.53e-5 | — | — | — | — | 0.081 | 5.51 | f32 | 1.53e-5 |
| cuda | row:rope | float16 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 2.53e-3 |
| cuda | row:scale_mask_softmax | float16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 2.39e-4 |
| cuda | row:add_rms_norm | float16 | — | 0.893 | — | add_rms_norm_affine_f32; converted on the device | 1.95e-3 | — | — | — | — | 0.182 | 4.90 | f32 | 1.95e-3 |
| cuda | row:add_layer_norm | float16 | — | — | — | — | — | — | — | — | — | 0.181 | — | f32 | 1.95e-3 |
| cuda | row:cross_entropy_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 2.44e-4 |
| cuda | norm_params_bwd:rms | float16 | — | — | — | — | — | — | — | — | — | 0.101 | — | f32 | 3.15e-5 |
| cuda | norm_params_bwd:layer | float16 | — | — | — | — | — | — | — | — | — | 0.110 | — | f32 | 3.44e-5 |
| cuda | layout:strided | float16 | — | — | — | — | — | — | — | — | — | 0.451 | — | f32 | 0e+0 |
| cuda | layout:one_hot | float16 | — | — | — | — | — | — | — | — | — | 0.048 | — | f32 | 0e+0 |
| cuda | layout:gather_rows | float16 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 0e+0 |
| cuda | layout:gather_cols | float16 | — | — | — | — | — | — | — | — | — | 0.130 | — | f32 | 0e+0 |
| cuda | layout:scatter_add_rows | float16 | — | — | — | — | — | — | — | — | — | 0.205 | — | f32 | 1.95e-3 |
| cuda | layout:im2col | float16 | — | — | — | — | — | — | — | — | — | 0.067 | — | f32 | 0e+0 |
| cuda | layout:col2im | float16 | — | — | — | — | — | — | — | — | — | 0.048 | — | f32 | 1.95e-3 |
| cuda | layout:interpolate_nearest | float16 | — | — | — | — | — | — | — | — | — | 0.093 | — | f32 | 0e+0 |
| cuda | layout:interpolate_bilinear | float16 | — | — | — | — | — | — | — | — | — | 0.138 | — | f32 | 2.44e-4 |
| cuda | layout:shift | float16 | — | — | — | — | — | — | — | — | — | 0.157 | — | f32 | 0e+0 |
| cuda | layout:pad2d | float16 | — | — | — | — | — | — | — | — | — | 0.156 | — | f32 | 0e+0 |
| cuda | layout:maxpool | float16 | — | — | — | — | — | — | — | — | — | 0.021 | — | f32 | 0e+0 |
| cuda | layout:avgpool | float16 | — | — | — | — | — | — | — | — | — | 0.021 | — | f32 | 2.44e-4 |
| cuda | layout:maxpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.093 | — | f32 | 0e+0 |
| cuda | layout:avgpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.085 | — | f32 | 2.98e-8 |
| cuda | layout:adaptive_maxpool | float16 | — | — | — | — | — | — | — | — | — | 0.144 | — | f32 | 0e+0 |
| cuda | layout:adaptive_avgpool | float16 | — | — | — | — | — | — | — | — | — | 0.144 | — | f32 | 1.53e-5 |
| cuda | layout:depthwise_conv | float16 | — | — | — | — | — | — | — | — | — | 0.100 | — | f32 | 1.95e-3 |
| cuda | layout:adaptive_maxpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 11.387 | — | f32 | 4.88e-4 |
| cuda | layout:adaptive_avgpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.168 | — | f32 | 9.44e-7 |
| cuda | matmul_nt | float16 | — | — | — | — | — | — | — | — | — | 0.246 | — | f16mma | 3.09e-2 |
| cuda | matmul_tn | float16 | — | — | — | — | — | — | — | — | — | 0.263 | — | f16mma | 3.08e-2 |
| cuda | matmul_tt | float16 | — | — | — | — | — | — | — | — | — | 0.267 | — | f16mma | 3.11e-2 |
| cuda | matmul_batched | float16 | — | — | — | — | — | — | — | — | — | 0.627 | — | f16mma | 1.56e-2 |
| cuda | matmul_bwd | float16 | — | — | — | — | — | — | — | — | — | 0.504 | — | f16mma | 3.1e-2 |
| cuda | attention | float16 | — | — | — | — | — | — | — | — | — | 3.810 | — | f32 | 2.27e-5 |
| cuda | attention_causal | float16 | — | — | — | — | — | — | — | — | — | 2.148 | — | f32 | 2.44e-4 |
| cuda | attention_fused | float16 | — | — | — | — | — | — | — | — | — | 3.788 | — | f32 | 2.27e-5 |
| cuda | attention_fused_causal | float16 | — | — | — | — | — | — | — | — | — | 2.146 | — | f32 | 2.44e-4 |
| cuda | attention_decode | float16 | — | — | — | — | — | — | — | — | — | 0.610 | — | f32 | 1.34e-5 |
| cuda | attention_bwd | float16 | — | — | — | — | — | — | — | — | — | 4.293 | — | f32 | 1.2e-4 |
| cuda | optimizer:sgd_momentum | float16 | — | — | — | — | — | — | — | — | — | 0.573 | — | f32; + restore of p, m, v | 4.88e-4 |
| cuda | optimizer:adam | float16 | — | — | — | — | — | — | — | — | — | 0.925 | — | f32; + restore of p, m, v | 4.88e-4 |
| cuda | optimizer:adamw | float16 | — | — | — | — | — | — | — | — | — | 0.925 | — | f32; + restore of p, m, v | 4.87e-4 |
| cuda | conv | float16 | — | — | — | — | — | — | — | — | — | 0.138 | — | f16mma | 1.55e-2 |
| cuda | conv_bwd | float16 | — | — | — | — | — | — | — | — | — | 2.222 | — | f32 | 1.56e-2 |
| cuda | lstm_step | float16 | — | — | — | — | — | — | — | — | — | 1.179 | — | f32 | 2.04e-4 |
| cuda | gru_step | float16 | — | — | — | — | — | — | — | — | — | 0.980 | — | f32 | 2.44e-4 |
| cuda | lstm_step_bwd | float16 | — | — | — | — | — | — | — | — | — | 2.901 | — | f32 | 2.43e-4 |
| cuda | gru_step_bwd | float16 | — | — | — | — | — | — | — | — | — | 2.506 | — | f32 | 2.41e-4 |
| cuda | lstm_seq | float16 | — | — | — | — | — | — | — | — | — | 3.787 | — | f32 | 1.7e-4 |
| cuda | lstm_seq_bwd | float16 | — | — | — | — | — | — | — | — | — | 14.141 | — | f32 | 2.39e-4 |
| cuda | gru_seq | float16 | — | — | — | — | — | — | — | — | — | 2.610 | — | f32 | 2.44e-4 |
| cuda | gru_seq_bwd | float16 | — | — | — | — | — | — | — | — | — | 18.643 | — | f32 | 2.44e-4 |
| cuda | topk_rows | float16 | — | — | — | — | — | — | — | — | — | 3.588 | — | f32 | 0e+0 |
| cuda | pick_rows | float16 | — | — | — | — | — | — | — | — | — | 0.008 | — | f32 | 0e+0 |
| cuda | sample_rows | float16 | — | — | — | — | — | — | — | — | — | 1.437 | — | f32 | 1e+0 |
| cuda | cross_entropy | float16 | — | — | — | — | — | — | — | — | — | 47.812 | — | f32 (log_softmax, pick, neg) | 7.8e-3 |
| cuda | ctc | float16 | — | — | — | — | — | — | — | — | — | 0.515 | — | f32 | 4.96e-1 |
| cuda | ctc_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.281 | — | f32 | 1.6e-3 |
| cuda | softmax | float16 | 4096x4096 | 0.443 | — | softmax_f32; converted on the device | 9.53e-7 | 0.090 | 4.92 | softmax_f16 | 3.3e-6 | 0.084 | 5.30 | f32 | 9.54e-7 |
| cuda | rms_norm | float16 | 4096x4096 | 0.442 | — | rms_norm_f32; converted on the device | 4.88e-4 | 0.088 | 5.01 | rms_norm_f16 | 1.64e-3 | 0.078 | 5.64 | f32 | 4.88e-4 |
| cuda | add | bfloat16 | 16777216 | 0.662 | — | add_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.126 | 5.24 | f32 | 3.91e-3 |
| cuda | matmul | bfloat16 | 2048x2048x2048 | 1.018 | — | gemm128_f32; converted on the device | 2.5e-1 | — | — | — | — | 0.244 | 4.17 | bf16mma | 2.5e-1 |
| cuda | map:copy | bfloat16 | — | — | — | — | — | — | — | — | — | 0.085 | — | f32 | 0e+0 |
| cuda | map:fill | bfloat16 | — | 0.211 | — | fill_f32; converted on the device | 0e+0 | — | — | — | — | 0.031 | 6.83 | f32 | 0e+0 |
| cuda | map:neg | bfloat16 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 0e+0 |
| cuda | map:abs | bfloat16 | — | 0.445 | — | abs_f32; converted on the device | 0e+0 | — | — | — | — | 0.086 | 5.17 | f32 | 0e+0 |
| cuda | map:sign | bfloat16 | — | 0.442 | — | sign_f32; converted on the device | 0e+0 | — | — | — | — | 0.084 | 5.27 | f32 | 0e+0 |
| cuda | map:square | bfloat16 | — | 0.442 | — | square_f32; converted on the device | 3.1e-2 | — | — | — | — | 0.084 | 5.26 | f32 | 3.1e-2 |
| cuda | map:relu | bfloat16 | — | 0.442 | — | relu_f32; converted on the device | 0e+0 | — | — | — | — | 0.085 | 5.20 | f32 | 0e+0 |
| cuda | map:add_scalar | bfloat16 | — | — | — | — | — | — | — | — | — | 0.083 | — | f32 | 1.56e-2 |
| cuda | map:scale | bfloat16 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 3.13e-2 |
| cuda | map:affine | bfloat16 | — | 0.442 | — | affine_f32; converted on the device | 3.13e-2 | — | — | — | — | 0.085 | 5.20 | f32 | 3.13e-2 |
| cuda | map:clamp | bfloat16 | — | 0.441 | — | clamp_f32; converted on the device | 0e+0 | — | — | — | — | 0.084 | 5.23 | f32 | 0e+0 |
| cuda | map:add | bfloat16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 1.56e-2 |
| cuda | map:sub | bfloat16 | — | 0.660 | — | sub_f32; converted on the device | 1.56e-2 | — | — | — | — | 0.125 | 5.27 | f32 | 1.56e-2 |
| cuda | map:mul | bfloat16 | — | 0.661 | — | mul_f32; converted on the device | 3.13e-2 | — | — | — | — | 0.126 | 5.27 | f32 | 3.13e-2 |
| cuda | map:div | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 7.78e-3 |
| cuda | map:max | bfloat16 | — | 0.658 | — | max_f32; converted on the device | 0e+0 | — | — | — | — | 0.127 | 5.19 | f32 | 0e+0 |
| cuda | map:min | bfloat16 | — | 0.660 | — | min_f32; converted on the device | 0e+0 | — | — | — | — | 0.127 | 5.20 | f32 | 0e+0 |
| cuda | map:axpy | bfloat16 | — | 0.834 | — | axpy_f32; converted on the device; + copy of y | 3.13e-2 | — | — | — | — | 0.127 | 6.57 | f32 | 3.13e-2 |
| cuda | map:muladd | bfloat16 | — | 1.055 | — | muladd_f32; converted on the device; + copy of z | 6.25e-2 | — | — | — | — | 0.169 | 6.24 | f32 | 6.25e-2 |
| cuda | map:where | bfloat16 | — | 0.798 | — | where_f32; converted on the device | 0e+0 | — | — | — | — | 0.169 | 4.73 | f32 | 0e+0 |
| cuda | map:square_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.125 | — | f32 | 6.25e-2 |
| cuda | map:abs_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 0e+0 |
| cuda | map:relu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 0e+0 |
| cuda | map:clamp_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.241 | — | f32 | 0e+0 |
| cuda | map:where_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 0e+0 |
| cuda | map:sqrt | bfloat16 | — | 0.849 | — | sqrt_f32; converted on the device | 3.9e-3 | — | — | — | — | 0.085 | 9.98 | f32 | 3.9e-3 |
| cuda | map:inv_sqrt | bfloat16 | — | 0.443 | — | inv_sqrt_f32; converted on the device | 3.9e-3 | — | — | — | — | 0.091 | 4.87 | f32 | 3.9e-3 |
| cuda | map:exp | bfloat16 | — | 0.442 | — | exp_f32; converted on the device | 1.21e-1 | — | — | — | — | 0.088 | 5.02 | f32 | 1.21e-1 |
| cuda | map:expm1 | bfloat16 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 1.95e-3 |
| cuda | map:log | bfloat16 | — | 0.440 | — | log_f32; converted on the device | 3.84e-3 | — | — | — | — | 0.100 | 4.39 | f32 | 3.84e-3 |
| cuda | map:sin | bfloat16 | — | 0.441 | — | sin_f32; converted on the device | 1.95e-3 | — | — | — | — | 0.087 | 5.05 | f32 | 1.95e-3 |
| cuda | map:cos | bfloat16 | — | 0.437 | — | cos_f32; converted on the device | 1.95e-3 | — | — | — | — | 0.089 | 4.94 | f32 | 1.95e-3 |
| cuda | map:tanh | bfloat16 | — | 0.441 | — | tanh_f32; converted on the device | 1.94e-3 | — | — | — | — | 0.095 | 4.65 | f32 | 1.94e-3 |
| cuda | map:sigmoid | bfloat16 | — | 0.439 | — | sigmoid_f32; converted on the device | 1.95e-3 | — | — | — | — | 0.095 | 4.61 | f32 | 1.95e-3 |
| cuda | map:erf | bfloat16 | — | — | — | — | — | — | — | — | — | 0.615 | — | f32 | 1.95e-3 |
| cuda | map:gelu | bfloat16 | — | 0.442 | — | gelu_f32; converted on the device | 7.75e-3 | — | — | — | — | 0.099 | 4.47 | f32 | 7.75e-3 |
| cuda | map:gelu_erf | bfloat16 | — | — | — | — | — | — | — | — | — | 0.991 | — | f32 | 7.77e-3 |
| cuda | map:silu | bfloat16 | — | 0.440 | — | silu_f32; converted on the device | 7.79e-3 | — | — | — | — | 0.097 | 4.56 | f32 | 7.79e-3 |
| cuda | map:leaky_relu | bfloat16 | — | 0.441 | — | leaky_relu_f32; converted on the device | 7.81e-4 | — | — | — | — | 0.085 | 5.21 | f32 | 7.81e-4 |
| cuda | map:pow | bfloat16 | — | — | — | — | — | — | — | — | — | 0.110 | — | f32 | 6.17e-2 |
| cuda | map:dropout | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 1.56e-2 |
| cuda | map:gated_silu | bfloat16 | — | — | — | — | — | — | — | — | — | 0.128 | — | f32 | 3.12e-2 |
| cuda | map:gated_gelu | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 3.12e-2 |
| cuda | map:sqrt_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 | 7.81e-3 |
| cuda | map:inv_sqrt_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.239 | — | f32 | 1.56e-2 |
| cuda | map:exp_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 | 5e-1 |
| cuda | map:log_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.240 | — | f32 | 1.56e-2 |
| cuda | map:sin_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.236 | — | f32 | 7.81e-3 |
| cuda | map:cos_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.240 | — | f32 | 7.81e-3 |
| cuda | map:tanh_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 7.81e-3 |
| cuda | map:sigmoid_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.241 | — | f32 | 1.95e-3 |
| cuda | map:silu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.239 | — | f32 | 1.56e-2 |
| cuda | map:gelu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.239 | — | f32 | 1.56e-2 |
| cuda | map:gelu_erf_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.009 | — | f32 | 1.56e-2 |
| cuda | map:leaky_relu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 7.81e-4 |
| cuda | map:pow_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 2.5e-1 |
| cuda | map:gated_silu_bwd_x | bfloat16 | — | — | — | — | — | — | — | — | — | 0.171 | — | f32 | 6.23e-2 |
| cuda | map:gated_silu_bwd_y | bfloat16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 3.12e-2 |
| cuda | map:gated_gelu_bwd_x | bfloat16 | — | — | — | — | — | — | — | — | — | 0.323 | — | f32 | 3.12e-2 |
| cuda | map:gated_gelu_bwd_y | bfloat16 | — | — | — | — | — | — | — | — | — | 0.237 | — | f32 | 3.12e-2 |
| cuda | sum:whole | bfloat16 | — | 0.397 | — | f32; converted on the device | 6.14e-4 | — | — | — | — | 0.025 | 15.96 | f32 | 2.48e-4 |
| cuda | sum:rows | bfloat16 | — | 0.195 | — | f32; converted on the device | 7.63e-6 | — | — | — | — | 0.023 | 8.68 | f32 | 1e-5 |
| cuda | sum:cols | bfloat16 | — | 0.780 | — | f32; converted on the device; + zero of y | 1.86e-5 | — | — | — | — | 0.029 | 27.19 | f32 | 1.86e-5 |
| cuda | sumsq:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 1.79e-4 |
| cuda | max:whole | bfloat16 | — | — | — | — | — | — | — | — | — | 0.025 | — | f32 | 0e+0 |
| cuda | max:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 0e+0 |
| cuda | max:cols | bfloat16 | — | — | — | — | — | — | — | — | — | 0.029 | — | f32 | 0e+0 |
| cuda | min:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 0e+0 |
| cuda | argmax:rows | bfloat16 | — | 0.224 | — | f32 index; converted on the device | 0e+0 | — | — | — | — | 0.027 | 8.37 | s32 index | 0e+0 |
| cuda | argmax:cols | bfloat16 | — | — | — | — | — | — | — | — | — | 0.189 | — | s32 index | 0e+0 |
| cuda | maprow:add | bfloat16 | — | — | — | — | — | — | — | — | — | 0.143 | — | f32 | 3.91e-3 |
| cuda | maprow:mul | bfloat16 | — | — | — | — | — | — | — | — | — | 0.156 | — | f32 | 1.95e-3 |
| cuda | transpose | bfloat16 | — | 0.826 | — | transpose_f32; converted on the device | 0e+0 | — | — | — | — | 0.090 | 9.16 | f32 | 0e+0 |
| cuda | gemv | bfloat16 | — | — | — | — | — | — | — | — | — | 0.179 | — | f32 | 2.5e-1 |
| cuda | gemv_t | bfloat16 | — | — | — | — | — | — | — | — | — | 0.185 | — | f32 | 2.49e-1 |
| cuda | row:log_softmax | bfloat16 | — | 0.444 | — | softmax_f32; converted on the device | 3.13e-2 | — | — | — | — | 0.080 | 5.53 | f32 | 3.13e-2 |
| cuda | row:softmax_bwd | bfloat16 | — | 0.816 | — | softmax_bwd_f32; converted on the device; + zero of dx | 2.5e-1 | — | — | — | — | 0.124 | 6.59 | f32 | 2.5e-1 |
| cuda | row:log_softmax_bwd | bfloat16 | — | 0.815 | — | softmax_bwd_f32; converted on the device; + zero of dx | 2.5e-1 | — | — | — | — | 0.124 | 6.58 | f32 | 2.5e-1 |
| cuda | row:rms_norm_affine | bfloat16 | — | 0.446 | — | rms_norm_affine_f32; converted on the device | 7.81e-3 | — | — | — | — | 0.079 | 5.64 | f32 | 7.81e-3 |
| cuda | row:rms_norm_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.125 | — | f32 | 1.95e-3 |
| cuda | row:layer_norm | bfloat16 | — | 0.448 | — | layernorm_fused_f32; converted on the device | 1.56e-2 | — | — | — | — | 0.081 | 5.55 | f32 | 1.56e-2 |
| cuda | row:layer_norm_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.127 | — | f32 | 1.95e-3 |
| cuda | row:weight_norm | bfloat16 | — | 0.446 | — | weight_norm_f32; converted on the device | 1.22e-4 | — | — | — | — | 0.081 | 5.54 | f32 | 1.22e-4 |
| cuda | row:rope | bfloat16 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 1.61e-2 |
| cuda | row:scale_mask_softmax | bfloat16 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 1.34e-3 |
| cuda | row:add_rms_norm | bfloat16 | — | 0.893 | — | add_rms_norm_affine_f32; converted on the device | 1.56e-2 | — | — | — | — | 0.182 | 4.90 | f32 | 1.56e-2 |
| cuda | row:add_layer_norm | bfloat16 | — | — | — | — | — | — | — | — | — | 0.182 | — | f32 | 1.56e-2 |
| cuda | row:cross_entropy_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.085 | — | f32 | 1.95e-3 |
| cuda | norm_params_bwd:rms | bfloat16 | — | — | — | — | — | — | — | — | — | 0.103 | — | f32 | 3.44e-5 |
| cuda | norm_params_bwd:layer | bfloat16 | — | — | — | — | — | — | — | — | — | 0.110 | — | f32 | 4.58e-5 |
| cuda | layout:strided | bfloat16 | — | — | — | — | — | — | — | — | — | 0.439 | — | f32 | 0e+0 |
| cuda | layout:one_hot | bfloat16 | — | — | — | — | — | — | — | — | — | 0.052 | — | f32 | 0e+0 |
| cuda | layout:gather_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 0e+0 |
| cuda | layout:gather_cols | bfloat16 | — | — | — | — | — | — | — | — | — | 0.129 | — | f32 | 0e+0 |
| cuda | layout:scatter_add_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.209 | — | f32 | 1.56e-2 |
| cuda | layout:im2col | bfloat16 | — | — | — | — | — | — | — | — | — | 0.068 | — | f32 | 0e+0 |
| cuda | layout:col2im | bfloat16 | — | — | — | — | — | — | — | — | — | 0.048 | — | f32 | 1.56e-2 |
| cuda | layout:interpolate_nearest | bfloat16 | — | — | — | — | — | — | — | — | — | 0.098 | — | f32 | 0e+0 |
| cuda | layout:interpolate_bilinear | bfloat16 | — | — | — | — | — | — | — | — | — | 0.140 | — | f32 | 1.95e-3 |
| cuda | layout:shift | bfloat16 | — | — | — | — | — | — | — | — | — | 0.158 | — | f32 | 0e+0 |
| cuda | layout:pad2d | bfloat16 | — | — | — | — | — | — | — | — | — | 0.156 | — | f32 | 0e+0 |
| cuda | layout:maxpool | bfloat16 | — | — | — | — | — | — | — | — | — | 0.021 | — | f32 | 0e+0 |
| cuda | layout:avgpool | bfloat16 | — | — | — | — | — | — | — | — | — | 0.023 | — | f32 | 1.95e-3 |
| cuda | layout:maxpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.094 | — | f32 | 0e+0 |
| cuda | layout:avgpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.086 | — | f32 | 0e+0 |
| cuda | layout:adaptive_maxpool | bfloat16 | — | — | — | — | — | — | — | — | — | 0.144 | — | f32 | 0e+0 |
| cuda | layout:adaptive_avgpool | bfloat16 | — | — | — | — | — | — | — | — | — | 0.144 | — | f32 | 1.22e-4 |
| cuda | layout:depthwise_conv | bfloat16 | — | — | — | — | — | — | — | — | — | 0.101 | — | f32 | 1.55e-2 |
| cuda | layout:adaptive_maxpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 11.218 | — | f32 | 3.91e-3 |
| cuda | layout:adaptive_avgpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.169 | — | f32 | 7.52e-6 |
| cuda | matmul_nt | bfloat16 | — | — | — | — | — | — | — | — | — | 0.244 | — | bf16mma | 2.49e-1 |
| cuda | matmul_tn | bfloat16 | — | — | — | — | — | — | — | — | — | 0.262 | — | bf16mma | 2.49e-1 |
| cuda | matmul_tt | bfloat16 | — | — | — | — | — | — | — | — | — | 0.269 | — | bf16mma | 2.44e-1 |
| cuda | matmul_batched | bfloat16 | — | — | — | — | — | — | — | — | — | 0.624 | — | bf16mma | 1.21e-1 |
| cuda | matmul_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.503 | — | bf16mma | 2.49e-1 |
| cuda | attention | bfloat16 | — | — | — | — | — | — | — | — | — | 3.801 | — | f32 | 1.42e-4 |
| cuda | attention_causal | bfloat16 | — | — | — | — | — | — | — | — | — | 2.150 | — | f32 | 1.95e-3 |
| cuda | attention_fused | bfloat16 | — | — | — | — | — | — | — | — | — | 3.794 | — | f32 | 1.42e-4 |
| cuda | attention_fused_causal | bfloat16 | — | — | — | — | — | — | — | — | — | 2.153 | — | f32 | 1.95e-3 |
| cuda | attention_decode | bfloat16 | — | — | — | — | — | — | — | — | — | 0.607 | — | f32 | 1.15e-4 |
| cuda | attention_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 6.727 | — | f32 | 9.66e-4 |
| cuda | optimizer:sgd_momentum | bfloat16 | — | — | — | — | — | — | — | — | — | 0.571 | — | f32; + restore of p, m, v | 1.46e-3 |
| cuda | optimizer:adam | bfloat16 | — | — | — | — | — | — | — | — | — | 0.926 | — | f32; + restore of p, m, v | 3.68e-3 |
| cuda | optimizer:adamw | bfloat16 | — | — | — | — | — | — | — | — | — | 0.925 | — | f32; + restore of p, m, v | 3.69e-3 |
| cuda | conv | bfloat16 | — | — | — | — | — | — | — | — | — | 0.138 | — | bf16mma | 1.24e-1 |
| cuda | conv_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 2.222 | — | f32 | 1.25e-1 |
| cuda | lstm_step | bfloat16 | — | — | — | — | — | — | — | — | — | 1.187 | — | f32 | 1.92e-3 |
| cuda | gru_step | bfloat16 | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 | 1.95e-3 |
| cuda | lstm_step_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 2.889 | — | f32 | 1.94e-3 |
| cuda | gru_step_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 2.263 | — | f32 | 1.81e-3 |
| cuda | lstm_seq | bfloat16 | — | — | — | — | — | — | — | — | — | 2.847 | — | f32 | 1.46e-3 |
| cuda | lstm_seq_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 14.053 | — | f32 | 1.75e-3 |
| cuda | gru_seq | bfloat16 | — | — | — | — | — | — | — | — | — | 2.608 | — | f32 | 1.95e-3 |
| cuda | gru_seq_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 16.897 | — | f32 | 1.95e-3 |
| cuda | topk_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 3.588 | — | f32 | 0e+0 |
| cuda | pick_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.008 | — | f32 | 0e+0 |
| cuda | sample_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 1.430 | — | f32 | 1e+0 |
| cuda | cross_entropy | bfloat16 | — | — | — | — | — | — | — | — | — | 46.829 | — | f32 (log_softmax, pick, neg) | 6.19e-2 |
| cuda | ctc | bfloat16 | — | — | — | — | — | — | — | — | — | 0.530 | — | f32 | 3.95e+0 |
| cuda | ctc_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.288 | — | f32 | 2.93e-3 |
| cuda | softmax | bfloat16 | 4096x4096 | 0.444 | — | softmax_f32; converted on the device | 7.63e-6 | — | — | — | — | 0.082 | 5.42 | f32 | 7.63e-6 |
| cuda | rms_norm | bfloat16 | 4096x4096 | 0.443 | — | rms_norm_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.081 | 5.45 | f32 | 3.91e-3 |
| cuda | add | float8_e4m3 | 16777216 | 494.600 | — | add_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.050 | 9852.59 | f32 | 6.25e-2 |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | 130.558 | — | gemm128_f32; converted through the host | 3.99e+0 | — | — | — | — | 0.259 | 505.06 | f16mma | 3.99e+0 |
| cuda | map:copy | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.037 | — | f32 | 0e+0 |
| cuda | map:fill | float8_e4m3 | — | 89.024 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.021 | 4342.62 | f32 | 0e+0 |
| cuda | map:neg | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.039 | — | f32 | 0e+0 |
| cuda | map:abs | float8_e4m3 | — | 456.475 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.042 | 10868.45 | f32 | 0e+0 |
| cuda | map:sign | float8_e4m3 | — | 153.720 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.043 | 3574.89 | f32 | 0e+0 |
| cuda | map:square | float8_e4m3 | — | 460.408 | — | square_f32; converted through the host | 4.38e-1 | — | — | — | — | 0.042 | 10962.10 | f32 | 4.38e-1 |
| cuda | map:relu | float8_e4m3 | — | 334.471 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.042 | 7963.59 | f32 | 0e+0 |
| cuda | map:add_scalar | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.042 | — | f32 | 2.5e-1 |
| cuda | map:scale | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.042 | — | f32 | 5e-1 |
| cuda | map:affine | float8_e4m3 | — | 463.035 | — | affine_f32; converted through the host | 5e-1 | — | — | — | — | 0.044 | 10547.49 | f32 | 5e-1 |
| cuda | map:clamp | float8_e4m3 | — | 312.891 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.044 | 7111.15 | f32 | 0e+0 |
| cuda | map:add | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.481 | — | f32 | 2.5e-1 |
| cuda | map:sub | float8_e4m3 | — | 540.570 | — | sub_f32; converted through the host | 2.5e-1 | — | — | — | — | 0.817 | 661.90 | f32 | 2.5e-1 |
| cuda | map:mul | float8_e4m3 | — | 556.257 | — | mul_f32; converted through the host | 5e-1 | — | — | — | — | 0.443 | 1254.81 | f32 | 5e-1 |
| cuda | map:div | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.058 | — | f32 | 1.17e-1 |
| cuda | map:max | float8_e4m3 | — | 456.290 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.055 | 8236.27 | f32 | 0e+0 |
| cuda | map:min | float8_e4m3 | — | 476.357 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.978 | 487.17 | f32 | 0e+0 |
| cuda | map:axpy | float8_e4m3 | — | 501.713 | — | axpy_f32; converted through the host; + copy of y | 5e-1 | — | — | — | — | 0.055 | 9072.56 | f32 | 5e-1 |
| cuda | map:muladd | float8_e4m3 | — | 603.590 | — | muladd_f32; converted through the host; + copy of z | 1e+0 | — | — | — | — | 0.078 | 7758.23 | f32 | 1e+0 |
| cuda | map:where | float8_e4m3 | — | 474.117 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.075 | 6363.99 | f32 | 0e+0 |
| cuda | map:square_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.052 | — | f32 | 1e+0 |
| cuda | map:abs_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.057 | — | f32 | 0e+0 |
| cuda | map:relu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.054 | — | f32 | 0e+0 |
| cuda | map:clamp_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.056 | — | f32 | 0e+0 |
| cuda | map:where_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.055 | — | f32 | 0e+0 |
| cuda | map:sqrt | float8_e4m3 | — | 368.144 | — | sqrt_f32; converted through the host | 6.15e-2 | — | — | — | — | 0.043 | 8561.49 | f32 | 6.15e-2 |
| cuda | map:inv_sqrt | float8_e4m3 | — | 393.115 | — | inv_sqrt_f32; converted through the host | 5.6e-2 | — | — | — | — | 0.056 | 6982.51 | f32 | 5.6e-2 |
| cuda | map:exp | float8_e4m3 | — | 531.737 | — | exp_f32; converted through the host | 1.48e+0 | — | — | — | — | 0.060 | 8803.59 | f32 | 1.48e+0 |
| cuda | map:expm1 | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.062 | — | f32 | 2.7e-2 |
| cuda | map:log | float8_e4m3 | — | 421.769 | — | log_f32; converted through the host | 5.37e-2 | — | — | — | — | 0.094 | 4506.08 | f32 | 5.37e-2 |
| cuda | map:sin | float8_e4m3 | — | 376.902 | — | sin_f32; converted through the host | 3.05e-2 | — | — | — | — | 0.060 | 6240.10 | f32 | 3.05e-2 |
| cuda | map:cos | float8_e4m3 | — | 380.531 | — | cos_f32; converted through the host | 3.11e-2 | — | — | — | — | 0.061 | 6248.46 | f32 | 3.11e-2 |
| cuda | map:tanh | float8_e4m3 | — | 327.387 | — | tanh_f32; converted through the host | 3.01e-2 | — | — | — | — | 0.071 | 4637.21 | f32 | 3.01e-2 |
| cuda | map:sigmoid | float8_e4m3 | — | 476.986 | — | sigmoid_f32; converted through the host | 3.12e-2 | — | — | — | — | 0.069 | 6953.15 | f32 | 3.12e-2 |
| cuda | map:erf | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.652 | — | f32 | 3.02e-2 |
| cuda | map:gelu | float8_e4m3 | — | 466.040 | — | gelu_f32; converted through the host | 5.7e-2 | — | — | — | — | 0.073 | 6410.45 | f32 | 5.7e-2 |
| cuda | map:gelu_erf | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.126 | — | f32 | 5.7e-2 |
| cuda | map:silu | float8_e4m3 | — | 441.284 | — | silu_f32; converted through the host | 1.21e-1 | — | — | — | — | 0.067 | 6635.86 | f32 | 1.21e-1 |
| cuda | map:leaky_relu | float8_e4m3 | — | 507.662 | — | leaky_relu_f32; converted through the host | 1.25e-2 | — | — | — | — | 0.044 | 11537.77 | f32 | 1.25e-2 |
| cuda | map:pow | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.120 | — | f32 | 9.58e-1 |
| cuda | map:dropout | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.925 | — | f32 | 1.88e-1 |
| cuda | map:gated_silu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.078 | — | f32 | 4.94e-1 |
| cuda | map:gated_gelu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.080 | — | f32 | 4.98e-1 |
| cuda | map:sqrt_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.064 | — | f32 | 1.22e-1 |
| cuda | map:inv_sqrt_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.075 | — | f32 | 2.43e-1 |
| cuda | map:exp_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.071 | — | f32 | 7.5e+0 |
| cuda | map:log_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.061 | — | f32 | 2.33e-1 |
| cuda | map:sin_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.070 | — | f32 | 1.25e-1 |
| cuda | map:cos_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.071 | — | f32 | 1.24e-1 |
| cuda | map:tanh_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.111 | — | f32 | 1.22e-1 |
| cuda | map:sigmoid_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.108 | — | f32 | 3.06e-2 |
| cuda | map:silu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.109 | — | f32 | 2.45e-1 |
| cuda | map:gelu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.115 | — | f32 | 2.4e-1 |
| cuda | map:gelu_erf_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.215 | — | f32 | 2.37e-1 |
| cuda | map:leaky_relu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.052 | — | f32 | 1.25e-2 |
| cuda | map:pow_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.128 | — | f32 | 3.92e+0 |
| cuda | map:gated_silu_bwd_x | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.122 | — | f32 | 9.79e-1 |
| cuda | map:gated_silu_bwd_y | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.081 | — | f32 | 4.94e-1 |
| cuda | map:gated_gelu_bwd_x | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.142 | — | f32 | 4.97e-1 |
| cuda | map:gated_gelu_bwd_y | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 4.98e-1 |
| cuda | sum:whole | float8_e4m3 | — | 18.539 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.042 | 445.66 | f32 | 0e+0 |
| cuda | sum:rows | float8_e4m3 | — | 18.233 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.037 | 487.51 | f32 | 0e+0 |
| cuda | sum:cols | float8_e4m3 | — | 19.547 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.043 | 454.57 | f32 | 0e+0 |
| cuda | sumsq:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.038 | — | f32 | 1.64e-4 |
| cuda | max:whole | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.043 | — | f32 | 0e+0 |
| cuda | max:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.039 | — | f32 | 0e+0 |
| cuda | max:cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.048 | — | f32 | 0e+0 |
| cuda | min:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.041 | — | f32 | 0e+0 |
| cuda | argmax:rows | float8_e4m3 | — | 26.767 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.043 | 622.49 | s32 index | 0e+0 |
| cuda | argmax:cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.245 | — | s32 index | 0e+0 |
| cuda | maprow:add | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.071 | — | f32 | 6.25e-2 |
| cuda | maprow:mul | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.072 | — | f32 | 3.13e-2 |
| cuda | transpose | float8_e4m3 | — | 476.746 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.075 | 6382.14 | f32 | 0e+0 |
| cuda | gemv | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.337 | — | f32 | 3.99e+0 |
| cuda | gemv_t | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.349 | — | f32 | 7.52e+0 |
| cuda | row:log_softmax | float8_e4m3 | — | 388.943 | — | softmax_f32; converted through the host | 5e-1 | — | — | — | — | 0.067 | 5848.76 | f32 | 5e-1 |
| cuda | row:softmax_bwd | float8_e4m3 | — | 592.648 | — | softmax_bwd_f32; converted through the host; + zero of dx | 3.99e+0 | — | — | — | — | 0.890 | 665.67 | f32 | 3.99e+0 |
| cuda | row:log_softmax_bwd | float8_e4m3 | — | 601.949 | — | softmax_bwd_f32; converted through the host; + zero of dx | 4e+0 | — | — | — | — | 0.885 | 680.55 | f32 | 4e+0 |
| cuda | row:rms_norm_affine | float8_e4m3 | — | 495.661 | — | rms_norm_affine_f32; converted through the host | 1.25e-1 | — | — | — | — | 0.068 | 7299.86 | f32 | 1.25e-1 |
| cuda | row:rms_norm_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.089 | — | f32 | 3.12e-2 |
| cuda | row:layer_norm | float8_e4m3 | — | 527.703 | — | layernorm_fused_f32; converted through the host | 2.5e-1 | — | — | — | — | 0.095 | 5543.10 | f32 | 2.5e-1 |
| cuda | row:layer_norm_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.841 | — | f32 | 3.13e-2 |
| cuda | row:weight_norm | float8_e4m3 | — | 427.633 | — | weight_norm_f32; converted through the host | 1.95e-3 | — | — | — | — | 0.043 | 9944.95 | f32 | 1.95e-3 |
| cuda | row:rope | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.111 | — | f32 | 2.5e-1 |
| cuda | row:scale_mask_softmax | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.124 | — | f32 | 2.93e-2 |
| cuda | row:add_rms_norm | float8_e4m3 | — | 1061.833 | — | add_rms_norm_affine_f32; converted through the host | 2.5e-1 | — | — | — | — | 0.097 | 10901.77 | f32 | 2.5e-1 |
| cuda | row:add_layer_norm | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.138 | — | f32 | 2.5e-1 |
| cuda | row:cross_entropy_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.081 | — | f32 | 2.07e-3 |
| cuda | norm_params_bwd:rms | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.101 | — | f32 | 3.71e-5 |
| cuda | norm_params_bwd:layer | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.103 | — | f32 | 3.75e-5 |
| cuda | layout:strided | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.312 | — | f32 | 0e+0 |
| cuda | layout:one_hot | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.044 | — | f32 | 0e+0 |
| cuda | layout:gather_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.063 | — | f32 | 0e+0 |
| cuda | layout:gather_cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.064 | — | f32 | 0e+0 |
| cuda | layout:scatter_add_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.183 | — | f32 | 2.5e-1 |
| cuda | layout:im2col | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.066 | — | f32 | 0e+0 |
| cuda | layout:col2im | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.056 | — | f32 | 2.5e-1 |
| cuda | layout:interpolate_nearest | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.099 | — | f32 | 0e+0 |
| cuda | layout:interpolate_bilinear | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.198 | — | f32 | 3.13e-2 |
| cuda | layout:shift | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.062 | — | f32 | 0e+0 |
| cuda | layout:pad2d | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.061 | — | f32 | 0e+0 |
| cuda | layout:maxpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.026 | — | f32 | 0e+0 |
| cuda | layout:avgpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.028 | — | f32 | 3.13e-2 |
| cuda | layout:maxpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.114 | — | f32 | 0e+0 |
| cuda | layout:avgpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.090 | — | f32 | 9.77e-4 |
| cuda | layout:adaptive_maxpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.322 | — | f32 | 0e+0 |
| cuda | layout:adaptive_avgpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.319 | — | f32 | 1.95e-3 |
| cuda | layout:depthwise_conv | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.159 | — | f32 | 2.5e-1 |
| cuda | layout:adaptive_maxpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 41.685 | — | f32 | 6.25e-2 |
| cuda | layout:adaptive_avgpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.194 | — | f32 | 9.76e-4 |
| cuda | matmul_nt | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.256 | — | f16mma | 3.95e+0 |
| cuda | matmul_tn | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.260 | — | f16mma | 3.97e+0 |
| cuda | matmul_tt | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.258 | — | f16mma | 3.95e+0 |
| cuda | matmul_batched | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.712 | — | f16mma | 1.97e+0 |
| cuda | matmul_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.525 | — | f16mma | 3.99e+0 |
| cuda | attention | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.724 | — | f32 | 1.95e-3 |
| cuda | attention_causal | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.657 | — | f32 | 3.12e-2 |
| cuda | attention_fused | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.723 | — | f32 | 1.95e-3 |
| cuda | attention_fused_causal | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.654 | — | f32 | 3.12e-2 |
| cuda | attention_decode | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.431 | — | f32 | 1.93e-3 |
| cuda | attention_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.653 | — | f32 | 1.53e-2 |
| cuda | optimizer:sgd_momentum | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32; + restore of p, m, v | 1.46e-3 |
| cuda | optimizer:adam | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.819 | — | f32; + restore of p, m, v | 1.81e-2 |
| cuda | optimizer:adamw | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.820 | — | f32; + restore of p, m, v | 1.81e-2 |
| cuda | conv | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.144 | — | f16mma | 2e+0 |
| cuda | conv_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.219 | — | f32 | 2e+0 |
| cuda | lstm_step | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.344 | — | f32 | 3.08e-2 |
| cuda | gru_step | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.957 | — | f32 | 3.12e-2 |
| cuda | lstm_step_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.863 | — | f32 | 3.09e-2 |
| cuda | gru_step_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.232 | — | f32 | 3.05e-2 |
| cuda | lstm_seq | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.852 | — | f32 | 1.77e-2 |
| cuda | lstm_seq_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 14.270 | — | f32 | 2.45e-2 |
| cuda | gru_seq | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.622 | — | f32 | 3.12e-2 |
| cuda | gru_seq_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 16.900 | — | f32 | 3.11e-2 |
| cuda | topk_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.076 | — | f32 | 0e+0 |
| cuda | pick_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.009 | — | f32 | 0e+0 |
| cuda | sample_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.447 | — | f32 | 1e+0 |
| cuda | cross_entropy | float8_e4m3 | — | — | — | — | — | — | — | — | — | 18.723 | — | f32 (log_softmax, pick, neg) | 5e-1 |
| cuda | ctc | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.528 | — | f32 | Infinity |
| cuda | ctc_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.243 | — | f32 | 3.22e-2 |
| cuda | softmax | float8_e4m3 | 4096x4096 | 107.532 | — | softmax_f32; converted through the host | 9.76e-4 | — | — | — | — | 0.067 | 1617.02 | f32 | 9.76e-4 |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | 444.869 | — | rms_norm_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.041 | 10797.79 | f32 | 6.25e-2 |
| cuda | add | float8_e5m2 | 16777216 | 472.262 | — | add_f32; converted through the host | 1.25e-1 | — | — | — | — | 1.107 | 426.65 | f32 | 1.25e-1 |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | 110.644 | — | gemm128_f32; converted through the host | 7.71e+0 | — | — | — | — | 0.261 | 423.76 | f16mma | 7.71e+0 |
| cuda | map:copy | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.039 | — | f32 | 0e+0 |
| cuda | map:fill | float8_e5m2 | — | 88.659 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.019 | 4570.04 | f32 | 0e+0 |
| cuda | map:neg | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.041 | — | f32 | 0e+0 |
| cuda | map:abs | float8_e5m2 | — | 348.782 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.041 | 8527.68 | f32 | 0e+0 |
| cuda | map:sign | float8_e5m2 | — | 111.657 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.045 | 2459.41 | f32 | 0e+0 |
| cuda | map:square | float8_e5m2 | — | 392.103 | — | square_f32; converted through the host | 1e+0 | — | — | — | — | 0.047 | 8432.32 | f32 | 1e+0 |
| cuda | map:relu | float8_e5m2 | — | 305.778 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.046 | 6632.92 | f32 | 0e+0 |
| cuda | map:add_scalar | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.046 | — | f32 | 5e-1 |
| cuda | map:scale | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.046 | — | f32 | 1e+0 |
| cuda | map:affine | float8_e5m2 | — | 411.205 | — | affine_f32; converted through the host | 1e+0 | — | — | — | — | 0.046 | 8919.85 | f32 | 1e+0 |
| cuda | map:clamp | float8_e5m2 | — | 280.039 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.047 | 5970.99 | f32 | 0e+0 |
| cuda | map:add | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.606 | — | f32 | 5e-1 |
| cuda | map:sub | float8_e5m2 | — | 414.959 | — | sub_f32; converted through the host | 5e-1 | — | — | — | — | 0.657 | 631.79 | f32 | 5e-1 |
| cuda | map:mul | float8_e5m2 | — | 455.474 | — | mul_f32; converted through the host | 1e+0 | — | — | — | — | 0.063 | 7195.48 | f32 | 1e+0 |
| cuda | map:div | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.588 | — | f32 | 2.14e-1 |
| cuda | map:max | float8_e5m2 | — | 374.292 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.066 | 5714.38 | f32 | 0e+0 |
| cuda | map:min | float8_e5m2 | — | 375.287 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.572 | 655.64 | f32 | 0e+0 |
| cuda | map:axpy | float8_e5m2 | — | 424.974 | — | axpy_f32; converted through the host; + copy of y | 1e+0 | — | — | — | — | 0.066 | 6488.15 | f32 | 1e+0 |
| cuda | map:muladd | float8_e5m2 | — | 498.473 | — | muladd_f32; converted through the host; + copy of z | 2e+0 | — | — | — | — | 0.075 | 6664.07 | f32 | 2e+0 |
| cuda | map:where | float8_e5m2 | — | 389.321 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.088 | 4424.10 | f32 | 0e+0 |
| cuda | map:square_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.057 | — | f32 | 2e+0 |
| cuda | map:abs_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.061 | — | f32 | 0e+0 |
| cuda | map:relu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.057 | — | f32 | 0e+0 |
| cuda | map:clamp_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.058 | — | f32 | 0e+0 |
| cuda | map:where_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.064 | — | f32 | 0e+0 |
| cuda | map:sqrt | float8_e5m2 | — | 326.941 | — | sqrt_f32; converted through the host | 1.21e-1 | — | — | — | — | 0.048 | 6868.51 | f32 | 1.21e-1 |
| cuda | map:inv_sqrt | float8_e5m2 | — | 348.557 | — | inv_sqrt_f32; converted through the host | 9.53e-2 | — | — | — | — | 0.059 | 5948.08 | f32 | 9.53e-2 |
| cuda | map:exp | float8_e5m2 | — | 457.886 | — | exp_f32; converted through the host | 1.4e+0 | — | — | — | — | 0.065 | 7099.00 | f32 | 1.4e+0 |
| cuda | map:expm1 | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.065 | — | f32 | 4.88e-2 |
| cuda | map:log | float8_e5m2 | — | 339.663 | — | log_f32; converted through the host | 1.14e-1 | — | — | — | — | 0.097 | 3498.07 | f32 | 1.14e-1 |
| cuda | map:sin | float8_e5m2 | — | 306.285 | — | sin_f32; converted through the host | 5.66e-2 | — | — | — | — | 0.063 | 4900.56 | f32 | 5.66e-2 |
| cuda | map:cos | float8_e5m2 | — | 322.854 | — | cos_f32; converted through the host | 6.15e-2 | — | — | — | — | 0.065 | 5005.48 | f32 | 6.15e-2 |
| cuda | map:tanh | float8_e5m2 | — | 252.696 | — | tanh_f32; converted through the host | 5.86e-2 | — | — | — | — | 0.073 | 3475.87 | f32 | 5.86e-2 |
| cuda | map:sigmoid | float8_e5m2 | — | 392.929 | — | sigmoid_f32; converted through the host | 6.22e-2 | — | — | — | — | 0.071 | 5549.85 | f32 | 6.22e-2 |
| cuda | map:erf | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.633 | — | f32 | 4.79e-2 |
| cuda | map:gelu | float8_e5m2 | — | 374.770 | — | gelu_f32; converted through the host | 1.18e-1 | — | — | — | — | 0.072 | 5226.92 | f32 | 1.18e-1 |
| cuda | map:gelu_erf | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.113 | — | f32 | 1.18e-1 |
| cuda | map:silu | float8_e5m2 | — | 359.355 | — | silu_f32; converted through the host | 1.9e-1 | — | — | — | — | 0.069 | 5178.02 | f32 | 1.9e-1 |
| cuda | map:leaky_relu | float8_e5m2 | — | 422.658 | — | leaky_relu_f32; converted through the host | 2.5e-2 | — | — | — | — | 0.048 | 8787.07 | f32 | 2.5e-2 |
| cuda | map:pow | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.122 | — | f32 | 1.08e+0 |
| cuda | map:dropout | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.202 | — | f32 | 3.75e-1 |
| cuda | map:gated_silu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.082 | — | f32 | 7.59e-1 |
| cuda | map:gated_gelu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 9.89e-1 |
| cuda | map:sqrt_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.068 | — | f32 | 2.14e-1 |
| cuda | map:inv_sqrt_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.078 | — | f32 | 3.43e-1 |
| cuda | map:exp_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.075 | — | f32 | 8.5e+0 |
| cuda | map:log_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.070 | — | f32 | 4.29e-1 |
| cuda | map:sin_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.074 | — | f32 | 2.46e-1 |
| cuda | map:cos_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.074 | — | f32 | 2.39e-1 |
| cuda | map:tanh_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.115 | — | f32 | 2.47e-1 |
| cuda | map:sigmoid_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.113 | — | f32 | 6.19e-2 |
| cuda | map:silu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.113 | — | f32 | 3.98e-1 |
| cuda | map:gelu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.119 | — | f32 | 4.9e-1 |
| cuda | map:gelu_erf_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.145 | — | f32 | 4.91e-1 |
| cuda | map:leaky_relu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.058 | — | f32 | 2.5e-2 |
| cuda | map:pow_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.132 | — | f32 | 6e+0 |
| cuda | map:gated_silu_bwd_x | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | 1.12e+0 |
| cuda | map:gated_silu_bwd_y | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.084 | — | f32 | 7.59e-1 |
| cuda | map:gated_gelu_bwd_x | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.146 | — | f32 | 9.93e-1 |
| cuda | map:gated_gelu_bwd_y | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 9.89e-1 |
| cuda | sum:whole | float8_e5m2 | — | 18.164 | — | f32; converted through the host | 6.1e-5 | — | — | — | — | 0.044 | 412.81 | f32 | 3.66e-4 |
| cuda | sum:rows | float8_e5m2 | — | 18.628 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.039 | 478.86 | f32 | 0e+0 |
| cuda | sum:cols | float8_e5m2 | — | 18.889 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.045 | 419.74 | f32 | 0e+0 |
| cuda | sumsq:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.039 | — | f32 | 1.71e-4 |
| cuda | max:whole | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.045 | — | f32 | 0e+0 |
| cuda | max:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.040 | — | f32 | 0e+0 |
| cuda | max:cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.049 | — | f32 | 0e+0 |
| cuda | min:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.043 | — | f32 | 0e+0 |
| cuda | argmax:rows | float8_e5m2 | — | 26.583 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.045 | 590.74 | s32 index | 0e+0 |
| cuda | argmax:cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.251 | — | s32 index | 0e+0 |
| cuda | maprow:add | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.074 | — | f32 | 1.25e-1 |
| cuda | maprow:mul | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.074 | — | f32 | 6.25e-2 |
| cuda | transpose | float8_e5m2 | — | 367.736 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.077 | 4763.41 | f32 | 0e+0 |
| cuda | gemv | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.335 | — | f32 | 7.95e+0 |
| cuda | gemv_t | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.357 | — | f32 | 1.15e+1 |
| cuda | row:log_softmax | float8_e5m2 | — | 338.132 | — | softmax_f32; converted through the host | 1e+0 | — | — | — | — | 0.069 | 4929.04 | f32 | 1e+0 |
| cuda | row:softmax_bwd | float8_e5m2 | — | 472.214 | — | softmax_bwd_f32; converted through the host; + zero of dx | 7.49e+0 | — | — | — | — | 1.096 | 431.01 | f32 | 7.49e+0 |
| cuda | row:log_softmax_bwd | float8_e5m2 | — | 496.682 | — | softmax_bwd_f32; converted through the host; + zero of dx | 8e+0 | — | — | — | — | 0.079 | 6303.07 | f32 | 8e+0 |
| cuda | row:rms_norm_affine | float8_e5m2 | — | 401.888 | — | rms_norm_affine_f32; converted through the host | 2.5e-1 | — | — | — | — | 0.077 | 5219.33 | f32 | 2.5e-1 |
| cuda | row:rms_norm_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.573 | — | f32 | 6.25e-2 |
| cuda | row:layer_norm | float8_e5m2 | — | 426.207 | — | layernorm_fused_f32; converted through the host | 5e-1 | — | — | — | — | 0.095 | 4476.96 | f32 | 5e-1 |
| cuda | row:layer_norm_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.105 | — | f32 | 6.25e-2 |
| cuda | row:weight_norm | float8_e5m2 | — | 373.440 | — | weight_norm_f32; converted through the host | 3.91e-3 | — | — | — | — | 0.050 | 7424.25 | f32 | 3.91e-3 |
| cuda | row:rope | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.120 | — | f32 | 5e-1 |
| cuda | row:scale_mask_softmax | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.133 | — | f32 | 5.42e-2 |
| cuda | row:add_rms_norm | float8_e5m2 | — | 837.895 | — | add_rms_norm_affine_f32; converted through the host | 5e-1 | — | — | — | — | 0.108 | 7794.37 | f32 | 5e-1 |
| cuda | row:add_layer_norm | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.146 | — | f32 | 5e-1 |
| cuda | row:cross_entropy_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.085 | — | f32 | 2.05e-3 |
| cuda | norm_params_bwd:rms | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.112 | — | f32 | 3.62e-5 |
| cuda | norm_params_bwd:layer | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.114 | — | f32 | 3.56e-5 |
| cuda | layout:strided | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.311 | — | f32 | 0e+0 |
| cuda | layout:one_hot | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.045 | — | f32 | 0e+0 |
| cuda | layout:gather_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.066 | — | f32 | 0e+0 |
| cuda | layout:gather_cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.066 | — | f32 | 0e+0 |
| cuda | layout:scatter_add_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.184 | — | f32 | 5e-1 |
| cuda | layout:im2col | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.067 | — | f32 | 0e+0 |
| cuda | layout:col2im | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.056 | — | f32 | 5e-1 |
| cuda | layout:interpolate_nearest | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.099 | — | f32 | 0e+0 |
| cuda | layout:interpolate_bilinear | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.200 | — | f32 | 6.25e-2 |
| cuda | layout:shift | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.067 | — | f32 | 0e+0 |
| cuda | layout:pad2d | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.066 | — | f32 | 0e+0 |
| cuda | layout:maxpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.027 | — | f32 | 0e+0 |
| cuda | layout:avgpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.028 | — | f32 | 6.25e-2 |
| cuda | layout:maxpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.116 | — | f32 | 0e+0 |
| cuda | layout:avgpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.091 | — | f32 | 7.63e-6 |
| cuda | layout:adaptive_maxpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.330 | — | f32 | 0e+0 |
| cuda | layout:adaptive_avgpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.323 | — | f32 | 3.9e-3 |
| cuda | layout:depthwise_conv | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.167 | — | f32 | 4.92e-1 |
| cuda | layout:adaptive_maxpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 44.348 | — | f32 | 1.25e-1 |
| cuda | layout:adaptive_avgpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.196 | — | f32 | 2.34e-4 |
| cuda | matmul_nt | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.258 | — | f16mma | 7.97e+0 |
| cuda | matmul_tn | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.263 | — | f16mma | 7.66e+0 |
| cuda | matmul_tt | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.262 | — | f16mma | 7.96e+0 |
| cuda | matmul_batched | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.732 | — | f16mma | 3.65e+0 |
| cuda | matmul_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.532 | — | f16mma | 7.27e+0 |
| cuda | attention | float8_e5m2 | — | — | — | — | — | — | — | — | — | 5.670 | — | f32 | 3.91e-3 |
| cuda | attention_causal | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.698 | — | f32 | 6.25e-2 |
| cuda | attention_fused | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.808 | — | f32 | 3.91e-3 |
| cuda | attention_fused_causal | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.704 | — | f32 | 6.25e-2 |
| cuda | attention_decode | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.434 | — | f32 | 2.98e-3 |
| cuda | attention_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.078 | — | f32 | 3.06e-2 |
| cuda | optimizer:sgd_momentum | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32; + restore of p, m, v | 1.46e-3 |
| cuda | optimizer:adam | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.820 | — | f32; + restore of p, m, v | 2.59e-2 |
| cuda | optimizer:adamw | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.818 | — | f32; + restore of p, m, v | 2.59e-2 |
| cuda | conv | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.144 | — | f16mma | 3.99e+0 |
| cuda | conv_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.218 | — | f32 | 4e+0 |
| cuda | lstm_step | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.167 | — | f32 | 4.62e-2 |
| cuda | gru_step | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.959 | — | f32 | 6.25e-2 |
| cuda | lstm_step_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.859 | — | f32 | 6.16e-2 |
| cuda | gru_step_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.226 | — | f32 | 5.97e-2 |
| cuda | lstm_seq | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.844 | — | f32 | 5.59e-2 |
| cuda | lstm_seq_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 13.998 | — | f32 | 5.98e-2 |
| cuda | gru_seq | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.605 | — | f32 | 6.25e-2 |
| cuda | gru_seq_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 12.669 | — | f32 | 6.24e-2 |
| cuda | topk_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.122 | — | f32 | 0e+0 |
| cuda | pick_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.007 | — | f32 | 0e+0 |
| cuda | sample_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.580 | — | f32 | 1e+0 |
| cuda | cross_entropy | float8_e5m2 | — | — | — | — | — | — | — | — | — | 18.233 | — | f32 (log_softmax, pick, neg) | 9.84e-1 |
| cuda | ctc | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.528 | — | f32 | 1.27e+2 |
| cuda | ctc_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.244 | — | f32 | 6.33e-2 |
| cuda | softmax | float8_e5m2 | 4096x4096 | 295.054 | — | softmax_f32; converted through the host | 1.62e-4 | — | — | — | — | 0.069 | 4307.35 | f32 | 1.62e-4 |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | 340.592 | — | rms_norm_f32; converted through the host | 1.08e-1 | — | — | — | — | 0.047 | 7277.62 | f32 | 1.08e-1 |
| cuda | add | int64 | 16777216 | 117.636 | — | add_f32; converted through the host | 0e+0 | 0.520 | 226.27 | add_s64 | 0e+0 | 0.520 | 226.14 | s64 | 0e+0 |
| cuda | matmul | int64 | 2048x2048x2048 | 29.637 | — | gemm64_f32; converted through the host | 0e+0 | 4.427 | 6.70 | gemm64_s64 | 0e+0 | 0.356 | 83.27 | s8mma | 0e+0 |
| cuda | map:copy | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | map:fill | int64 | — | 46.083 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.165 | 279.63 | s64 | 0e+0 |
| cuda | map:neg | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | map:abs | int64 | — | 81.796 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.359 | 227.72 | s64 | 0e+0 |
| cuda | map:sign | int64 | — | 81.961 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 229.00 | s64 | 0e+0 |
| cuda | map:square | int64 | — | 81.589 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.685 | 119.16 | s64 | 0e+0 |
| cuda | map:relu | int64 | — | 78.936 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 220.25 | s64 | 0e+0 |
| cuda | map:add_scalar | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | map:scale | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | map:affine | int64 | — | 84.677 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 236.26 | s64 | 0e+0 |
| cuda | map:clamp | int64 | — | 82.319 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.359 | 229.49 | s64 | 0e+0 |
| cuda | map:add | int64 | — | — | — | — | — | — | — | — | — | 0.523 | — | s64 | 0e+0 |
| cuda | map:sub | int64 | — | 120.851 | — | sub_f32; converted through the host | 0e+0 | — | — | — | — | 0.553 | 218.70 | s64 | 0e+0 |
| cuda | map:mul | int64 | — | 125.555 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.521 | 240.94 | s64 | 0e+0 |
| cuda | map:div | int64 | — | — | — | — | — | — | — | — | — | 0.521 | — | s64 | 0e+0 |
| cuda | map:max | int64 | — | 121.563 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.520 | 233.60 | s64 | 0e+0 |
| cuda | map:min | int64 | — | 120.621 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.520 | 231.79 | s64 | 0e+0 |
| cuda | map:axpy | int64 | — | 121.397 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.521 | 233.14 | s64 | 0e+0 |
| cuda | map:muladd | int64 | — | 160.435 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.686 | 233.87 | s64 | 0e+0 |
| cuda | map:where | int64 | — | 159.329 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.617 | 258.32 | s64 | 0e+0 |
| cuda | map:square_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.520 | — | s64 | 0e+0 |
| cuda | map:abs_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.519 | — | s64 | 0e+0 |
| cuda | map:relu_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.521 | — | s64 | 0e+0 |
| cuda | map:clamp_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.520 | — | s64 | 0e+0 |
| cuda | map:where_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.520 | — | s64 | 0e+0 |
| cuda | sum:whole | int64 | — | 39.135 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.395 | 99.00 | s32 | 0e+0 |
| cuda | sum:rows | int64 | — | 38.685 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.390 | 99.14 | f32 | 0e+0 |
| cuda | sum:cols | int64 | — | 42.957 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.393 | 109.25 | f32 | 0e+0 |
| cuda | sumsq:rows | int64 | — | — | — | — | — | — | — | — | — | 0.391 | — | f32 | 0e+0 |
| cuda | max:whole | int64 | — | — | — | — | — | — | — | — | — | 0.174 | — | s64 | 0e+0 |
| cuda | max:rows | int64 | — | — | — | — | — | — | — | — | — | 0.170 | — | s64 | 0e+0 |
| cuda | max:cols | int64 | — | — | — | — | — | — | — | — | — | 0.177 | — | s64 | 0e+0 |
| cuda | min:rows | int64 | — | — | — | — | — | — | — | — | — | 0.170 | — | s64 | 0e+0 |
| cuda | argmax:rows | int64 | — | 39.111 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.327 | 119.64 | s32 index | 0e+0 |
| cuda | argmax:cols | int64 | — | — | — | — | — | — | — | — | — | 0.392 | — | s32 index | 0e+0 |
| cuda | maprow:add | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | maprow:mul | int64 | — | — | — | — | — | — | — | — | — | 0.357 | — | s64 | 0e+0 |
| cuda | transpose | int64 | — | 83.550 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.361 | 231.70 | s64 | 0e+0 |
| cuda | gemv | int64 | — | — | — | — | — | — | — | — | — | 1.421 | — | f32 | 0e+0 |
| cuda | gemv_t | int64 | — | — | — | — | — | — | — | — | — | 1.473 | — | f32 | 0e+0 |
| cuda | layout:strided | int64 | — | — | — | — | — | — | — | — | — | 0.691 | — | s64 | 0e+0 |
| cuda | layout:one_hot | int64 | — | — | — | — | — | — | — | — | — | 0.175 | — | s64 | 0e+0 |
| cuda | layout:gather_rows | int64 | — | — | — | — | — | — | — | — | — | 0.343 | — | s64 | 0e+0 |
| cuda | layout:gather_cols | int64 | — | — | — | — | — | — | — | — | — | 0.382 | — | s64 | 0e+0 |
| cuda | layout:scatter_add_rows | int64 | — | — | — | — | — | — | — | — | — | 0.675 | — | s32 | 0e+0 |
| cuda | layout:im2col | int64 | — | — | — | — | — | — | — | — | — | 0.116 | — | s64 | 0e+0 |
| cuda | layout:col2im | int64 | — | — | — | — | — | — | — | — | — | 0.254 | — | s32 | 0e+0 |
| cuda | layout:interpolate_nearest | int64 | — | — | — | — | — | — | — | — | — | 0.244 | — | s64 | 0e+0 |
| cuda | layout:shift | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | layout:pad2d | int64 | — | — | — | — | — | — | — | — | — | 0.358 | — | s64 | 0e+0 |
| cuda | layout:maxpool | int64 | — | — | — | — | — | — | — | — | — | 0.021 | — | s64 | 0e+0 |
| cuda | layout:maxpool_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.193 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool | int64 | — | — | — | — | — | — | — | — | — | 0.088 | — | s64 | 0e+0 |
| cuda | layout:depthwise_conv | int64 | — | — | — | — | — | — | — | — | — | 0.235 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | int64 | — | — | — | — | — | — | — | — | — | 6.897 | — | s32 | 0e+0 |
| cuda | matmul_nt | int64 | — | — | — | — | — | — | — | — | — | 0.353 | — | s8mma | 0e+0 |
| cuda | matmul_tn | int64 | — | — | — | — | — | — | — | — | — | 0.356 | — | s8mma | 0e+0 |
| cuda | matmul_tt | int64 | — | — | — | — | — | — | — | — | — | 0.356 | — | s8mma | 0e+0 |
| cuda | matmul_batched | int64 | — | — | — | — | — | — | — | — | — | 1.614 | — | s8mma | 0e+0 |
| cuda | matmul_bwd | int64 | — | — | — | — | — | — | — | — | — | 0.720 | — | s8mma | 0e+0 |
| cuda | conv | int64 | — | — | — | — | — | — | — | — | — | 0.195 | — | s8mma | 0e+0 |
| cuda | topk_rows | int64 | — | — | — | — | — | — | — | — | — | 16.244 | — | s64 | 0e+0 |
| cuda | pick_rows | int64 | — | — | — | — | — | — | — | — | — | 0.009 | — | s64 | 0e+0 |
| cuda | add | int32 | 16777216 | 0.787 | — | add_f32; converted on the device | 0e+0 | 0.262 | 3.01 | add_s32 | 0e+0 | 0.263 | 2.99 | s32 | 0e+0 |
| cuda | matmul | int32 | 2048x2048x2048 | 1.141 | — | gemm128_f32; converted on the device | 0e+0 | 1.106 | 1.03 | gemm128_s32 | 0e+0 | 0.209 | 5.46 | s8mma | 0e+0 |
| cuda | map:copy | int32 | — | — | — | — | — | — | — | — | — | 0.181 | — | s32 | 0e+0 |
| cuda | map:fill | int32 | — | 0.262 | — | fill_f32; converted on the device | 0e+0 | — | — | — | — | 0.083 | 3.14 | s32 | 0e+0 |
| cuda | map:neg | int32 | — | — | — | — | — | — | — | — | — | 0.182 | — | s32 | 0e+0 |
| cuda | map:abs | int32 | — | 0.533 | — | abs_f32; converted on the device | 0e+0 | — | — | — | — | 0.182 | 2.92 | s32 | 0e+0 |
| cuda | map:sign | int32 | — | 0.534 | — | sign_f32; converted on the device | 0e+0 | — | — | — | — | 0.180 | 2.97 | s32 | 0e+0 |
| cuda | map:square | int32 | — | 0.532 | — | square_f32; converted on the device | 0e+0 | — | — | — | — | 0.181 | 2.93 | s32 | 0e+0 |
| cuda | map:relu | int32 | — | 0.532 | — | relu_f32; converted on the device | 0e+0 | — | — | — | — | 0.183 | 2.91 | s32 | 0e+0 |
| cuda | map:add_scalar | int32 | — | — | — | — | — | — | — | — | — | 0.182 | — | s32 | 0e+0 |
| cuda | map:scale | int32 | — | — | — | — | — | — | — | — | — | 0.181 | — | s32 | 0e+0 |
| cuda | map:affine | int32 | — | 0.533 | — | affine_f32; converted on the device | 0e+0 | — | — | — | — | 0.182 | 2.93 | s32 | 0e+0 |
| cuda | map:clamp | int32 | — | 0.534 | — | clamp_f32; converted on the device | 0e+0 | — | — | — | — | 0.181 | 2.94 | s32 | 0e+0 |
| cuda | map:add | int32 | — | — | — | — | — | — | — | — | — | 0.261 | — | s32 | 0e+0 |
| cuda | map:sub | int32 | — | 0.788 | — | sub_f32; converted on the device | 0e+0 | — | — | — | — | 0.263 | 3.00 | s32 | 0e+0 |
| cuda | map:mul | int32 | — | 0.788 | — | mul_f32; converted on the device | 0e+0 | — | — | — | — | 0.263 | 3.00 | s32 | 0e+0 |
| cuda | map:div | int32 | — | — | — | — | — | — | — | — | — | 0.261 | — | s32 | 0e+0 |
| cuda | map:max | int32 | — | 0.788 | — | max_f32; converted on the device | 0e+0 | — | — | — | — | 0.263 | 3.00 | s32 | 0e+0 |
| cuda | map:min | int32 | — | 0.789 | — | min_f32; converted on the device | 0e+0 | — | — | — | — | 0.262 | 3.01 | s32 | 0e+0 |
| cuda | map:axpy | int32 | — | 0.963 | — | axpy_f32; converted on the device; + copy of y | 0e+0 | — | — | — | — | 0.262 | 3.67 | s32 | 0e+0 |
| cuda | map:muladd | int32 | — | 1.221 | — | muladd_f32; converted on the device; + copy of z | 0e+0 | — | — | — | — | 0.344 | 3.55 | s32 | 0e+0 |
| cuda | map:where | int32 | — | 1.027 | — | where_f32; converted on the device | 0e+0 | — | — | — | — | 0.329 | 3.12 | s32 | 0e+0 |
| cuda | map:square_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.263 | — | s32 | 0e+0 |
| cuda | map:abs_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.262 | — | s32 | 0e+0 |
| cuda | map:relu_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.263 | — | s32 | 0e+0 |
| cuda | map:clamp_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.262 | — | s32 | 0e+0 |
| cuda | map:where_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.263 | — | s32 | 0e+0 |
| cuda | sum:whole | int32 | — | 0.262 | — | f32; converted on the device | 0e+0 | — | — | — | — | 0.093 | 2.81 | s64 | 0e+0 |
| cuda | sum:rows | int32 | — | 0.233 | — | f32; converted on the device | 0e+0 | — | — | — | — | 0.089 | 2.61 | s64 | 0e+0 |
| cuda | sum:cols | int32 | — | 0.557 | — | f32; converted on the device; + zero of y | 0e+0 | — | — | — | — | 0.099 | 5.61 | s64 | 0e+0 |
| cuda | sumsq:rows | int32 | — | — | — | — | — | — | — | — | — | 0.089 | — | s64 | 0e+0 |
| cuda | max:whole | int32 | — | — | — | — | — | — | — | — | — | 0.093 | — | s32 | 0e+0 |
| cuda | max:rows | int32 | — | — | — | — | — | — | — | — | — | 0.089 | — | s32 | 0e+0 |
| cuda | max:cols | int32 | — | — | — | — | — | — | — | — | — | 0.096 | — | s32 | 0e+0 |
| cuda | min:rows | int32 | — | — | — | — | — | — | — | — | — | 0.089 | — | s32 | 0e+0 |
| cuda | argmax:rows | int32 | — | 0.259 | — | f32 index; converted on the device | 0e+0 | — | — | — | — | 0.089 | 2.91 | s32 index | 0e+0 |
| cuda | argmax:cols | int32 | — | — | — | — | — | — | — | — | — | 0.398 | — | s32 index | 0e+0 |
| cuda | maprow:add | int32 | — | — | — | — | — | — | — | — | — | 0.180 | — | s32 | 0e+0 |
| cuda | maprow:mul | int32 | — | — | — | — | — | — | — | — | — | 0.183 | — | s32 | 0e+0 |
| cuda | transpose | int32 | — | 1.005 | — | transpose_f32; converted on the device | 0e+0 | — | — | — | — | 0.192 | 5.25 | s32 | 0e+0 |
| cuda | gemv | int32 | — | — | — | — | — | — | — | — | — | 0.814 | — | f32 | 0e+0 |
| cuda | gemv_t | int32 | — | — | — | — | — | — | — | — | — | 0.814 | — | f32 | 0e+0 |
| cuda | layout:strided | int32 | — | — | — | — | — | — | — | — | — | 0.428 | — | s32 | 0e+0 |
| cuda | layout:one_hot | int32 | — | — | — | — | — | — | — | — | — | 0.134 | — | s32 | 0e+0 |
| cuda | layout:gather_rows | int32 | — | — | — | — | — | — | — | — | — | 0.195 | — | s32 | 0e+0 |
| cuda | layout:gather_cols | int32 | — | — | — | — | — | — | — | — | — | 0.253 | — | s32 | 0e+0 |
| cuda | layout:scatter_add_rows | int32 | — | — | — | — | — | — | — | — | — | 0.473 | — | s32 | 0e+0 |
| cuda | layout:im2col | int32 | — | — | — | — | — | — | — | — | — | 0.087 | — | s32 | 0e+0 |
| cuda | layout:col2im | int32 | — | — | — | — | — | — | — | — | — | 0.119 | — | s32 | 0e+0 |
| cuda | layout:interpolate_nearest | int32 | — | — | — | — | — | — | — | — | — | 0.151 | — | s32 | 0e+0 |
| cuda | layout:shift | int32 | — | — | — | — | — | — | — | — | — | 0.181 | — | s32 | 0e+0 |
| cuda | layout:pad2d | int32 | — | — | — | — | — | — | — | — | — | 0.181 | — | s32 | 0e+0 |
| cuda | layout:maxpool | int32 | — | — | — | — | — | — | — | — | — | 0.021 | — | s32 | 0e+0 |
| cuda | layout:maxpool_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.172 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool | int32 | — | — | — | — | — | — | — | — | — | 0.081 | — | s32 | 0e+0 |
| cuda | layout:depthwise_conv | int32 | — | — | — | — | — | — | — | — | — | 0.207 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | int32 | — | — | — | — | — | — | — | — | — | 7.045 | — | s32 | 0e+0 |
| cuda | matmul_nt | int32 | — | — | — | — | — | — | — | — | — | 0.212 | — | s8mma | 0e+0 |
| cuda | matmul_tn | int32 | — | — | — | — | — | — | — | — | — | 0.210 | — | s8mma | 0e+0 |
| cuda | matmul_tt | int32 | — | — | — | — | — | — | — | — | — | 0.270 | — | s8mma | 0e+0 |
| cuda | matmul_batched | int32 | — | — | — | — | — | — | — | — | — | 1.528 | — | s8mma | 0e+0 |
| cuda | matmul_bwd | int32 | — | — | — | — | — | — | — | — | — | 0.397 | — | s8mma | 0e+0 |
| cuda | conv | int32 | — | — | — | — | — | — | — | — | — | 0.174 | — | s8mma | 0e+0 |
| cuda | topk_rows | int32 | — | — | — | — | — | — | — | — | — | 4.613 | — | s32 | 0e+0 |
| cuda | pick_rows | int32 | — | — | — | — | — | — | — | — | — | 0.009 | — | s32 | 0e+0 |
| cuda | add | int16 | 16777216 | 72.698 | — | add_f32; converted through the host | 0e+0 | 0.259 | 280.69 | add_s16 | 0e+0 | 0.224 | 324.11 | s16 | 0e+0 |
| cuda | matmul | int16 | 2048x2048x2048 | 18.072 | — | gemm64_f32; converted through the host | 0e+0 | 1.591 | 11.36 | gemm128_s16 | 0e+0 | 0.175 | 103.21 | s8mma | 0e+0 |
| cuda | map:copy | int16 | — | — | — | — | — | — | — | — | — | 0.086 | — | s32 | 0e+0 |
| cuda | map:fill | int16 | — | 32.622 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.029 | 1117.19 | s32 | 0e+0 |
| cuda | map:neg | int16 | — | — | — | — | — | — | — | — | — | 0.082 | — | s32 | 0e+0 |
| cuda | map:abs | int16 | — | 52.926 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.090 | 586.11 | s32 | 0e+0 |
| cuda | map:sign | int16 | — | 50.140 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 355.35 | s32 | 0e+0 |
| cuda | map:square | int16 | — | 51.828 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 366.79 | s32 | 0e+0 |
| cuda | map:relu | int16 | — | 50.851 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 361.16 | s32 | 0e+0 |
| cuda | map:add_scalar | int16 | — | — | — | — | — | — | — | — | — | 0.145 | — | s32 | 0e+0 |
| cuda | map:scale | int16 | — | — | — | — | — | — | — | — | — | 0.141 | — | s32 | 0e+0 |
| cuda | map:affine | int16 | — | 48.420 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 344.62 | s32 | 0e+0 |
| cuda | map:clamp | int16 | — | 48.855 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 347.47 | s32 | 0e+0 |
| cuda | map:add | int16 | — | — | — | — | — | — | — | — | — | 0.242 | — | s32 | 0e+0 |
| cuda | map:sub | int16 | — | 67.688 | — | sub_f32; converted through the host | 0e+0 | — | — | — | — | 0.241 | 281.33 | s32 | 0e+0 |
| cuda | map:mul | int16 | — | 70.492 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.241 | 292.98 | s32 | 0e+0 |
| cuda | map:div | int16 | — | — | — | — | — | — | — | — | — | 0.233 | — | s32 | 0e+0 |
| cuda | map:max | int16 | — | 67.132 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.235 | 285.67 | s32 | 0e+0 |
| cuda | map:min | int16 | — | 70.431 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.233 | 301.76 | s32 | 0e+0 |
| cuda | map:axpy | int16 | — | 68.073 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.233 | 292.03 | s32 | 0e+0 |
| cuda | map:muladd | int16 | — | 84.424 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.321 | 263.41 | s32 | 0e+0 |
| cuda | map:where | int16 | — | 86.729 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.321 | 270.61 | s32 | 0e+0 |
| cuda | map:square_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.241 | — | s32 | 0e+0 |
| cuda | map:abs_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.237 | — | s32 | 0e+0 |
| cuda | map:relu_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.239 | — | s32 | 0e+0 |
| cuda | map:clamp_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.232 | — | s32 | 0e+0 |
| cuda | map:where_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.241 | — | s32 | 0e+0 |
| cuda | sum:whole | int16 | — | 19.731 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.029 | 675.70 | s64 | 0e+0 |
| cuda | sum:rows | int16 | — | 19.652 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.027 | 738.80 | s32 | 0e+0 |
| cuda | sum:cols | int16 | — | 19.007 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.031 | 619.11 | s32 | 0e+0 |
| cuda | sumsq:rows | int16 | — | — | — | — | — | — | — | — | — | 0.031 | — | s64 | 0e+0 |
| cuda | max:whole | int16 | — | — | — | — | — | — | — | — | — | 0.031 | — | s32 | 0e+0 |
| cuda | max:rows | int16 | — | — | — | — | — | — | — | — | — | 0.028 | — | s32 | 0e+0 |
| cuda | max:cols | int16 | — | — | — | — | — | — | — | — | — | 0.031 | — | s32 | 0e+0 |
| cuda | min:rows | int16 | — | — | — | — | — | — | — | — | — | 0.027 | — | s32 | 0e+0 |
| cuda | argmax:rows | int16 | — | 19.540 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.031 | 624.27 | s32 index | 0e+0 |
| cuda | argmax:cols | int16 | — | — | — | — | — | — | — | — | — | 0.194 | — | s32 index | 0e+0 |
| cuda | maprow:add | int16 | — | — | — | — | — | — | — | — | — | 1.550 | — | s32 | 0e+0 |
| cuda | maprow:mul | int16 | — | — | — | — | — | — | — | — | — | 0.088 | — | s32 | 0e+0 |
| cuda | transpose | int16 | — | 49.731 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.091 | 548.91 | s32 | 0e+0 |
| cuda | gemv | int16 | — | — | — | — | — | — | — | — | — | 0.744 | — | f32 | 0e+0 |
| cuda | gemv_t | int16 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 | 0e+0 |
| cuda | layout:strided | int16 | — | — | — | — | — | — | — | — | — | 0.355 | — | s32 | 0e+0 |
| cuda | layout:one_hot | int16 | — | — | — | — | — | — | — | — | — | 0.048 | — | s32 | 0e+0 |
| cuda | layout:gather_rows | int16 | — | — | — | — | — | — | — | — | — | 0.092 | — | s32 | 0e+0 |
| cuda | layout:gather_cols | int16 | — | — | — | — | — | — | — | — | — | 0.130 | — | s32 | 0e+0 |
| cuda | layout:scatter_add_rows | int16 | — | — | — | — | — | — | — | — | — | 0.292 | — | s32 | 0e+0 |
| cuda | layout:im2col | int16 | — | — | — | — | — | — | — | — | — | 0.070 | — | s32 | 0e+0 |
| cuda | layout:col2im | int16 | — | — | — | — | — | — | — | — | — | 0.112 | — | s32 | 0e+0 |
| cuda | layout:interpolate_nearest | int16 | — | — | — | — | — | — | — | — | — | 0.096 | — | s32 | 0e+0 |
| cuda | layout:shift | int16 | — | — | — | — | — | — | — | — | — | 0.156 | — | s32 | 0e+0 |
| cuda | layout:pad2d | int16 | — | — | — | — | — | — | — | — | — | 0.159 | — | s32 | 0e+0 |
| cuda | layout:maxpool | int16 | — | — | — | — | — | — | — | — | — | 0.023 | — | s32 | 0e+0 |
| cuda | layout:maxpool_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.158 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool | int16 | — | — | — | — | — | — | — | — | — | 0.145 | — | s32 | 0e+0 |
| cuda | layout:depthwise_conv | int16 | — | — | — | — | — | — | — | — | — | 0.227 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | int16 | — | — | — | — | — | — | — | — | — | 14.406 | — | s32 | 0e+0 |
| cuda | matmul_nt | int16 | — | — | — | — | — | — | — | — | — | 0.170 | — | s8mma | 0e+0 |
| cuda | matmul_tn | int16 | — | — | — | — | — | — | — | — | — | 0.171 | — | s8mma | 0e+0 |
| cuda | matmul_tt | int16 | — | — | — | — | — | — | — | — | — | 0.184 | — | s8mma | 0e+0 |
| cuda | matmul_batched | int16 | — | — | — | — | — | — | — | — | — | 1.520 | — | s8mma | 0e+0 |
| cuda | matmul_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.376 | — | s8mma | 0e+0 |
| cuda | conv | int16 | — | — | — | — | — | — | — | — | — | 0.201 | — | s8mma | 0e+0 |
| cuda | topk_rows | int16 | — | — | — | — | — | — | — | — | — | 3.549 | — | s32 | 0e+0 |
| cuda | pick_rows | int16 | — | — | — | — | — | — | — | — | — | 0.008 | — | s32 | 0e+0 |
| cuda | add | int8 | 16777216 | 64.100 | — | add_f32; converted through the host | 0e+0 | 0.098 | 652.08 | add_s8 | 0e+0 | 0.035 | 1841.95 | s8 | 0e+0 |
| cuda | matmul | int8 | 2048x2048x2048 | 21.436 | — | gemm64_f32; converted through the host | 3.23e+2 | 1.554 | 13.80 | gemm128_s8 | 5.12e+2 | 0.094 | 227.56 | s8mma | 3.23e+2 |
| cuda | map:copy | int8 | — | — | — | — | — | — | — | — | — | 0.023 | — | s32 | 0e+0 |
| cuda | map:fill | int8 | — | 27.447 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.020 | 1407.53 | s32 | 0e+0 |
| cuda | map:neg | int8 | — | — | — | — | — | — | — | — | — | 0.023 | — | s32 | 0e+0 |
| cuda | map:abs | int8 | — | 46.204 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 2008.87 | s32 | 0e+0 |
| cuda | map:sign | int8 | — | 46.468 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.024 | 1960.68 | s32 | 0e+0 |
| cuda | map:square | int8 | — | 45.640 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.024 | 1885.96 | s32 | 0e+0 |
| cuda | map:relu | int8 | — | 55.606 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 2376.30 | s32 | 0e+0 |
| cuda | map:add_scalar | int8 | — | — | — | — | — | — | — | — | — | 0.023 | — | s32 | 0e+0 |
| cuda | map:scale | int8 | — | — | — | — | — | — | — | — | — | 0.023 | — | s32 | 0e+0 |
| cuda | map:affine | int8 | — | 45.822 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 2036.54 | s32 | 0e+0 |
| cuda | map:clamp | int8 | — | 56.192 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.026 | 2128.48 | s32 | 0e+0 |
| cuda | map:add | int8 | — | — | — | — | — | — | — | — | — | 0.539 | — | s32 | 0e+0 |
| cuda | map:sub | int8 | — | 61.831 | — | sub_f32; converted through the host | 0e+0 | — | — | — | — | 0.034 | 1829.33 | s32 | 0e+0 |
| cuda | map:mul | int8 | — | 64.587 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.047 | 1371.28 | s32 | 0e+0 |
| cuda | map:div | int8 | — | — | — | — | — | — | — | — | — | 0.058 | — | s32 | 0e+0 |
| cuda | map:max | int8 | — | 65.284 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.051 | 1292.76 | s32 | 0e+0 |
| cuda | map:min | int8 | — | 62.543 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.049 | 1268.62 | s32 | 0e+0 |
| cuda | map:axpy | int8 | — | 64.949 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.049 | 1314.75 | s32 | 0e+0 |
| cuda | map:muladd | int8 | — | 94.786 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.074 | 1286.10 | s32 | 0e+0 |
| cuda | map:where | int8 | — | 82.033 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.074 | 1113.06 | s32 | 0e+0 |
| cuda | map:square_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.047 | — | s32 | 0e+0 |
| cuda | map:abs_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.049 | — | s32 | 0e+0 |
| cuda | map:relu_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.047 | — | s32 | 0e+0 |
| cuda | map:clamp_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.049 | — | s32 | 0e+0 |
| cuda | map:where_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.047 | — | s32 | 0e+0 |
| cuda | sum:whole | int8 | — | 17.379 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.029 | 593.14 | s64 | 0e+0 |
| cuda | sum:rows | int8 | — | 18.121 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.027 | 676.15 | f32 | 0e+0 |
| cuda | sum:cols | int8 | — | 18.229 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.033 | 557.46 | f32 | 0e+0 |
| cuda | sumsq:rows | int8 | — | — | — | — | — | — | — | — | — | 0.027 | — | s32 | 0e+0 |
| cuda | max:whole | int8 | — | — | — | — | — | — | — | — | — | 0.033 | — | s32 | 0e+0 |
| cuda | max:rows | int8 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | max:cols | int8 | — | — | — | — | — | — | — | — | — | 0.034 | — | s32 | 0e+0 |
| cuda | min:rows | int8 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | argmax:rows | int8 | — | 26.559 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.033 | 812.20 | s32 index | 0e+0 |
| cuda | argmax:cols | int8 | — | — | — | — | — | — | — | — | — | 0.197 | — | s32 index | 0e+0 |
| cuda | maprow:add | int8 | — | — | — | — | — | — | — | — | — | 0.040 | — | s32 | 0e+0 |
| cuda | maprow:mul | int8 | — | — | — | — | — | — | — | — | — | 0.040 | — | s32 | 0e+0 |
| cuda | transpose | int8 | — | 83.639 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.046 | 1806.45 | s32 | 0e+0 |
| cuda | gemv | int8 | — | — | — | — | — | — | — | — | — | 0.130 | — | s32 | 5.72e+2 |
| cuda | gemv_t | int8 | — | — | — | — | — | — | — | — | — | 0.148 | — | s32 | 6.32e+2 |
| cuda | layout:strided | int8 | — | — | — | — | — | — | — | — | — | 0.314 | — | s32 | 0e+0 |
| cuda | layout:one_hot | int8 | — | — | — | — | — | — | — | — | — | 0.038 | — | s32 | 0e+0 |
| cuda | layout:gather_rows | int8 | — | — | — | — | — | — | — | — | — | 0.045 | — | s32 | 0e+0 |
| cuda | layout:gather_cols | int8 | — | — | — | — | — | — | — | — | — | 0.047 | — | s32 | 0e+0 |
| cuda | layout:scatter_add_rows | int8 | — | — | — | — | — | — | — | — | — | 0.173 | — | s32 | 0e+0 |
| cuda | layout:im2col | int8 | — | — | — | — | — | — | — | — | — | 0.057 | — | s32 | 0e+0 |
| cuda | layout:col2im | int8 | — | — | — | — | — | — | — | — | — | 0.050 | — | s32 | 0e+0 |
| cuda | layout:interpolate_nearest | int8 | — | — | — | — | — | — | — | — | — | 0.079 | — | s32 | 0e+0 |
| cuda | layout:shift | int8 | — | — | — | — | — | — | — | — | — | 0.044 | — | s32 | 0e+0 |
| cuda | layout:pad2d | int8 | — | — | — | — | — | — | — | — | — | 0.043 | — | s32 | 0e+0 |
| cuda | layout:maxpool | int8 | — | — | — | — | — | — | — | — | — | 0.022 | — | s32 | 0e+0 |
| cuda | layout:maxpool_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.097 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool | int8 | — | — | — | — | — | — | — | — | — | 0.263 | — | s32 | 0e+0 |
| cuda | layout:depthwise_conv | int8 | — | — | — | — | — | — | — | — | — | 0.116 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | int8 | — | — | — | — | — | — | — | — | — | 23.266 | — | s32 | 0e+0 |
| cuda | matmul_nt | int8 | — | — | — | — | — | — | — | — | — | 0.093 | — | s8mma | 3.98e+2 |
| cuda | matmul_tn | int8 | — | — | — | — | — | — | — | — | — | 0.102 | — | s8mma | 3.42e+2 |
| cuda | matmul_tt | int8 | — | — | — | — | — | — | — | — | — | 0.103 | — | s8mma | 3.77e+2 |
| cuda | matmul_batched | int8 | — | — | — | — | — | — | — | — | — | 0.457 | — | s8mma | 1.27e+2 |
| cuda | matmul_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.188 | — | s8mma | 3.31e+2 |
| cuda | conv | int8 | — | — | — | — | — | — | — | — | — | 0.103 | — | s8mma | 1.27e+2 |
| cuda | topk_rows | int8 | — | — | — | — | — | — | — | — | — | 6.531 | — | s32 | 0e+0 |
| cuda | pick_rows | int8 | — | — | — | — | — | — | — | — | — | 0.008 | — | s32 | 0e+0 |
| cuda | add | int4 | 16777216 | 78.638 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4273.80 | s8 | 0e+0 |
| cuda | matmul | int4 | 2048x2048x2048 | 44.193 | — | gemm64_f32; converted through the host | 4.43e+2 | — | — | — | — | 0.118 | 373.57 | s8mma | 4.43e+2 |
| cuda | map:copy | int4 | — | — | — | — | — | — | — | — | — | 0.019 | — | s32 | 0e+0 |
| cuda | map:fill | int4 | — | 40.607 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.016 | 2553.91 | s32 | 0e+0 |
| cuda | map:neg | int4 | — | — | — | — | — | — | — | — | — | 0.019 | — | s32 | 0e+0 |
| cuda | map:abs | int4 | — | 91.472 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 4254.52 | s32 | 0e+0 |
| cuda | map:sign | int4 | — | 76.630 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.020 | 3756.36 | s32 | 0e+0 |
| cuda | map:square | int4 | — | 67.109 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 3121.35 | s32 | 0e+0 |
| cuda | map:relu | int4 | — | 92.161 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 4227.56 | s32 | 0e+0 |
| cuda | map:add_scalar | int4 | — | — | — | — | — | — | — | — | — | 0.022 | — | s32 | 0e+0 |
| cuda | map:scale | int4 | — | — | — | — | — | — | — | — | — | 0.022 | — | s32 | 0e+0 |
| cuda | map:affine | int4 | — | 91.641 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 4262.38 | s32 | 0e+0 |
| cuda | map:clamp | int4 | — | 91.141 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 4239.11 | s32 | 0e+0 |
| cuda | map:add | int4 | — | — | — | — | — | — | — | — | — | 0.022 | — | s32 | 0e+0 |
| cuda | map:sub | int4 | — | 131.273 | — | sub_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 5782.97 | s32 | 0e+0 |
| cuda | map:mul | int4 | — | 115.342 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 5364.76 | s32 | 0e+0 |
| cuda | map:div | int4 | — | — | — | — | — | — | — | — | — | 0.046 | — | s32 | 0e+0 |
| cuda | map:max | int4 | — | 132.479 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.020 | 6690.86 | s32 | 0e+0 |
| cuda | map:min | int4 | — | 96.457 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.020 | 4871.59 | s32 | 0e+0 |
| cuda | map:axpy | int4 | — | 101.124 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.022 | 4575.76 | s32 | 0e+0 |
| cuda | map:muladd | int4 | — | 178.922 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.026 | 7016.56 | s32 | 0e+0 |
| cuda | map:where | int4 | — | 132.808 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 5902.60 | s32 | 0e+0 |
| cuda | map:square_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.020 | — | s32 | 0e+0 |
| cuda | map:abs_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.022 | — | s32 | 0e+0 |
| cuda | map:relu_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.020 | — | s32 | 0e+0 |
| cuda | map:clamp_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.021 | — | s32 | 0e+0 |
| cuda | map:where_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.023 | — | s32 | 0e+0 |
| cuda | sum:whole | int4 | — | 43.665 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.033 | 1315.21 | s32 | 0e+0 |
| cuda | sum:rows | int4 | — | 30.114 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.032 | 949.97 | f32 | 0e+0 |
| cuda | sum:cols | int4 | — | 31.963 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.036 | 892.81 | f32 | 0e+0 |
| cuda | sumsq:rows | int4 | — | — | — | — | — | — | — | — | — | 0.029 | — | f32 | 0e+0 |
| cuda | max:whole | int4 | — | — | — | — | — | — | — | — | — | 0.033 | — | s32 | 0e+0 |
| cuda | max:rows | int4 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | max:cols | int4 | — | — | — | — | — | — | — | — | — | 0.034 | — | s32 | 0e+0 |
| cuda | min:rows | int4 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | argmax:rows | int4 | — | 30.312 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.033 | 926.98 | s32 index | 0e+0 |
| cuda | argmax:cols | int4 | — | — | — | — | — | — | — | — | — | 0.194 | — | s32 index | 0e+0 |
| cuda | maprow:add | int4 | — | — | — | — | — | — | — | — | — | 0.037 | — | s32 | 0e+0 |
| cuda | maprow:mul | int4 | — | — | — | — | — | — | — | — | — | 0.037 | — | s32 | 0e+0 |
| cuda | transpose | int4 | — | 68.763 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.045 | 1538.33 | s32 | 0e+0 |
| cuda | gemv | int4 | — | — | — | — | — | — | — | — | — | 0.139 | — | f32 | 6.92e+2 |
| cuda | gemv_t | int4 | — | — | — | — | — | — | — | — | — | 0.155 | — | f32 | 7.52e+2 |
| cuda | layout:strided | int4 | — | — | — | — | — | — | — | — | — | 0.320 | — | s32 | 0e+0 |
| cuda | layout:one_hot | int4 | — | — | — | — | — | — | — | — | — | 0.036 | — | s32 | 0e+0 |
| cuda | layout:gather_rows | int4 | — | — | — | — | — | — | — | — | — | 0.043 | — | s32 | 0e+0 |
| cuda | layout:gather_cols | int4 | — | — | — | — | — | — | — | — | — | 0.050 | — | s32 | 0e+0 |
| cuda | layout:scatter_add_rows | int4 | — | — | — | — | — | — | — | — | — | 0.154 | — | s32 | 3e+0 |
| cuda | layout:im2col | int4 | — | — | — | — | — | — | — | — | — | 0.056 | — | s32 | 0e+0 |
| cuda | layout:col2im | int4 | — | — | — | — | — | — | — | — | — | 0.058 | — | s32 | 1.1e+1 |
| cuda | layout:interpolate_nearest | int4 | — | — | — | — | — | — | — | — | — | 0.077 | — | s32 | 0e+0 |
| cuda | layout:shift | int4 | — | — | — | — | — | — | — | — | — | 0.048 | — | s32 | 0e+0 |
| cuda | layout:pad2d | int4 | — | — | — | — | — | — | — | — | — | 0.045 | — | s32 | 0e+0 |
| cuda | layout:maxpool | int4 | — | — | — | — | — | — | — | — | — | 0.028 | — | s32 | 0e+0 |
| cuda | layout:maxpool_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.104 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool | int4 | — | — | — | — | — | — | — | — | — | 0.429 | — | s32 | 0e+0 |
| cuda | layout:depthwise_conv | int4 | — | — | — | — | — | — | — | — | — | 0.124 | — | s32 | 2.2e+1 |
| cuda | layout:adaptive_maxpool_bwd | int4 | — | — | — | — | — | — | — | — | — | 42.030 | — | s32 | 0e+0 |
| cuda | matmul_nt | int4 | — | — | — | — | — | — | — | — | — | 0.104 | — | s8mma | 5.18e+2 |
| cuda | matmul_tn | int4 | — | — | — | — | — | — | — | — | — | 0.105 | — | s8mma | 4.62e+2 |
| cuda | matmul_tt | int4 | — | — | — | — | — | — | — | — | — | 0.105 | — | s8mma | 4.97e+2 |
| cuda | matmul_batched | int4 | — | — | — | — | — | — | — | — | — | 0.456 | — | s8mma | 2.47e+2 |
| cuda | matmul_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.197 | — | s8mma | 4.51e+2 |
| cuda | conv | int4 | — | — | — | — | — | — | — | — | — | 0.107 | — | s8mma | 2.47e+2 |
| cuda | topk_rows | int4 | — | — | — | — | — | — | — | — | — | 3.593 | — | s32 | 0e+0 |
| cuda | pick_rows | int4 | — | — | — | — | — | — | — | — | — | 0.013 | — | s32 | 0e+0 |
| cuda | add | int2 | 16777216 | 75.790 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.019 | 4074.70 | s8 | 0e+0 |
| cuda | matmul | int2 | 2048x2048x2048 | 19.817 | — | gemm128_f32; converted through the host | 6.16e+2 | — | — | — | — | 0.111 | 179.17 | s8mma | 6.16e+2 |
| cuda | map:copy | int2 | — | — | — | — | — | — | — | — | — | 0.014 | — | s32 | 0e+0 |
| cuda | map:fill | int2 | — | 32.006 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.014 | 2370.79 | s32 | 0e+0 |
| cuda | map:neg | int2 | — | — | — | — | — | — | — | — | — | 0.016 | — | s32 | 0e+0 |
| cuda | map:abs | int2 | — | 65.545 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 3788.73 | s32 | 0e+0 |
| cuda | map:sign | int2 | — | 65.881 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 3064.23 | s32 | 0e+0 |
| cuda | map:square | int2 | — | 87.726 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.019 | 4521.95 | s32 | 0e+0 |
| cuda | map:relu | int2 | — | 87.172 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.020 | 4402.62 | s32 | 0e+0 |
| cuda | map:add_scalar | int2 | — | — | — | — | — | — | — | — | — | 0.020 | — | s32 | 0e+0 |
| cuda | map:scale | int2 | — | — | — | — | — | — | — | — | — | 0.019 | — | s32 | 0e+0 |
| cuda | map:affine | int2 | — | 86.873 | — | affine_f32; converted through the host | 1e+0 | — | — | — | — | 0.019 | 4477.97 | s32 | 0e+0 |
| cuda | map:clamp | int2 | — | 86.588 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 5034.19 | s32 | 0e+0 |
| cuda | map:add | int2 | — | — | — | — | — | — | — | — | — | 0.018 | — | s32 | 0e+0 |
| cuda | map:sub | int2 | — | 115.991 | — | sub_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 5394.92 | s32 | 0e+0 |
| cuda | map:mul | int2 | — | 95.990 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 4464.65 | s32 | 0e+0 |
| cuda | map:div | int2 | — | — | — | — | — | — | — | — | — | 0.042 | — | s32 | 0e+0 |
| cuda | map:max | int2 | — | 127.964 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 5951.83 | s32 | 0e+0 |
| cuda | map:min | int2 | — | 94.080 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.022 | 4375.82 | s32 | 0e+0 |
| cuda | map:axpy | int2 | — | 165.496 | — | axpy_f32; converted through the host; + copy of y | 1e+0 | — | — | — | — | 0.018 | 8994.32 | s32 | 0e+0 |
| cuda | map:muladd | int2 | — | 97.543 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.019 | 5161.01 | s32 | 0e+0 |
| cuda | map:where | int2 | — | 100.244 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.019 | 5276.01 | s32 | 0e+0 |
| cuda | map:square_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.016 | — | s32 | 0e+0 |
| cuda | map:abs_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.018 | — | s32 | 0e+0 |
| cuda | map:relu_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.017 | — | s32 | 0e+0 |
| cuda | map:clamp_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.018 | — | s32 | 0e+0 |
| cuda | map:where_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.017 | — | s32 | 0e+0 |
| cuda | sum:whole | int2 | — | 27.279 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.033 | 829.14 | s32 | 0e+0 |
| cuda | sum:rows | int2 | — | 27.047 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.029 | 945.71 | f32 | 0e+0 |
| cuda | sum:cols | int2 | — | 28.725 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.037 | 782.69 | f32 | 0e+0 |
| cuda | sumsq:rows | int2 | — | — | — | — | — | — | — | — | — | 0.029 | — | f32 | 0e+0 |
| cuda | max:whole | int2 | — | — | — | — | — | — | — | — | — | 0.033 | — | s32 | 0e+0 |
| cuda | max:rows | int2 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | max:cols | int2 | — | — | — | — | — | — | — | — | — | 0.035 | — | s32 | 0e+0 |
| cuda | min:rows | int2 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | argmax:rows | int2 | — | 29.242 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.033 | 894.25 | s32 index | 0e+0 |
| cuda | argmax:cols | int2 | — | — | — | — | — | — | — | — | — | 0.191 | — | s32 index | 0e+0 |
| cuda | maprow:add | int2 | — | — | — | — | — | — | — | — | — | 0.035 | — | s32 | 0e+0 |
| cuda | maprow:mul | int2 | — | — | — | — | — | — | — | — | — | 0.036 | — | s32 | 0e+0 |
| cuda | transpose | int2 | — | 68.815 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.041 | 1678.42 | s32 | 0e+0 |
| cuda | gemv | int2 | — | — | — | — | — | — | — | — | — | 0.138 | — | f32 | 2.15e+3 |
| cuda | gemv_t | int2 | — | — | — | — | — | — | — | — | — | 0.152 | — | f32 | 2.15e+3 |
| cuda | layout:strided | int2 | — | — | — | — | — | — | — | — | — | 0.318 | — | s32 | 0e+0 |
| cuda | layout:one_hot | int2 | — | — | — | — | — | — | — | — | — | 0.036 | — | s32 | 0e+0 |
| cuda | layout:gather_rows | int2 | — | — | — | — | — | — | — | — | — | 0.041 | — | s32 | 0e+0 |
| cuda | layout:gather_cols | int2 | — | — | — | — | — | — | — | — | — | 0.066 | — | s32 | 0e+0 |
| cuda | layout:scatter_add_rows | int2 | — | — | — | — | — | — | — | — | — | 0.188 | — | s32 | 3e+0 |
| cuda | layout:im2col | int2 | — | — | — | — | — | — | — | — | — | 0.053 | — | s32 | 0e+0 |
| cuda | layout:col2im | int2 | — | — | — | — | — | — | — | — | — | 0.064 | — | s32 | 7e+0 |
| cuda | layout:interpolate_nearest | int2 | — | — | — | — | — | — | — | — | — | 0.079 | — | s32 | 0e+0 |
| cuda | layout:shift | int2 | — | — | — | — | — | — | — | — | — | 0.044 | — | s32 | 0e+0 |
| cuda | layout:pad2d | int2 | — | — | — | — | — | — | — | — | — | 0.044 | — | s32 | 0e+0 |
| cuda | layout:maxpool | int2 | — | — | — | — | — | — | — | — | — | 0.029 | — | s32 | 0e+0 |
| cuda | layout:maxpool_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.168 | — | s32 | 0e+0 |
| cuda | layout:adaptive_maxpool | int2 | — | — | — | — | — | — | — | — | — | 0.669 | — | s32 | 0e+0 |
| cuda | layout:depthwise_conv | int2 | — | — | — | — | — | — | — | — | — | 0.122 | — | s32 | 7e+0 |
| cuda | layout:adaptive_maxpool_bwd | int2 | — | — | — | — | — | — | — | — | — | 47.309 | — | s32 | 0e+0 |
| cuda | matmul_nt | int2 | — | — | — | — | — | — | — | — | — | 0.109 | — | s8mma | 6.08e+2 |
| cuda | matmul_tn | int2 | — | — | — | — | — | — | — | — | — | 0.112 | — | s8mma | 6.14e+2 |
| cuda | matmul_tt | int2 | — | — | — | — | — | — | — | — | — | 0.111 | — | s8mma | 6.11e+2 |
| cuda | matmul_batched | int2 | — | — | — | — | — | — | — | — | — | 0.452 | — | s8mma | 1.83e+2 |
| cuda | matmul_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.207 | — | s8mma | 6.24e+2 |
| cuda | conv | int2 | — | — | — | — | — | — | — | — | — | 0.124 | — | s8mma | 2.04e+2 |
| cuda | topk_rows | int2 | — | — | — | — | — | — | — | — | — | 3.300 | — | s32 | 0e+0 |
| cuda | pick_rows | int2 | — | — | — | — | — | — | — | — | — | 0.013 | — | s32 | 0e+0 |
| cuda | add | uint64 | 16777216 | 134.628 | — | add_f32; converted through the host | 0e+0 | 0.999 | 134.80 | add_u64 | 0e+0 | 0.520 | 258.90 | u64 | 0e+0 |
| cuda | matmul | uint64 | 2048x2048x2048 | 35.524 | — | gemm128_f32; converted through the host | 0e+0 | 4.370 | 8.13 | gemm64_u64 | 0e+0 | 0.355 | 99.95 | u8mma | 0e+0 |
| cuda | map:copy | uint64 | — | — | — | — | — | — | — | — | — | 0.358 | — | u64 | 0e+0 |
| cuda | map:fill | uint64 | — | 50.507 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.165 | 306.29 | u64 | 0e+0 |
| cuda | map:neg | uint64 | — | — | — | — | — | — | — | — | — | 0.358 | — | u64 | 0e+0 |
| cuda | map:abs | uint64 | — | 122.613 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.685 | 179.13 | u64 | 0e+0 |
| cuda | map:sign | uint64 | — | 123.836 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.359 | 345.14 | u64 | 0e+0 |
| cuda | map:square | uint64 | — | 124.373 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 347.02 | u64 | 0e+0 |
| cuda | map:relu | uint64 | — | 121.051 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.685 | 176.72 | u64 | 0e+0 |
| cuda | map:add_scalar | uint64 | — | — | — | — | — | — | — | — | — | 0.358 | — | u64 | 0e+0 |
| cuda | map:scale | uint64 | — | — | — | — | — | — | — | — | — | 0.358 | — | u64 | 0e+0 |
| cuda | map:affine | uint64 | — | 90.761 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.358 | 253.24 | u64 | 0e+0 |
| cuda | map:clamp | uint64 | — | 120.274 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.683 | 176.12 | u64 | 0e+0 |
| cuda | map:add | uint64 | — | — | — | — | — | — | — | — | — | 0.520 | — | u64 | 0e+0 |
| cuda | map:sub | uint64 | — | 176.444 | — | sub_f32; converted through the host | 1.84e+19 | — | — | — | — | 1.001 | 176.25 | u64 | 0e+0 |
| cuda | map:mul | uint64 | — | 185.427 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 1.001 | 185.30 | u64 | 0e+0 |
| cuda | map:div | uint64 | — | — | — | — | — | — | — | — | — | 0.521 | — | u64 | 0e+0 |
| cuda | map:max | uint64 | — | 139.268 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.521 | 267.46 | u64 | 0e+0 |
| cuda | map:min | uint64 | — | 182.154 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.997 | 182.65 | u64 | 0e+0 |
| cuda | map:axpy | uint64 | — | 137.456 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.521 | 263.73 | u64 | 0e+0 |
| cuda | map:muladd | uint64 | — | 182.586 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.686 | 266.35 | u64 | 0e+0 |
| cuda | map:where | uint64 | — | 206.019 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.633 | 325.26 | u64 | 0e+0 |
| cuda | map:square_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.521 | — | u64 | 0e+0 |
| cuda | map:abs_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.521 | — | u64 | 0e+0 |
| cuda | map:relu_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.521 | — | u64 | 0e+0 |
| cuda | map:clamp_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.520 | — | u64 | 0e+0 |
| cuda | map:where_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.520 | — | u64 | 0e+0 |
| cuda | sum:whole | uint64 | — | 40.930 | — | f32; converted through the host | 1e+0 | — | — | — | — | 0.393 | 104.09 | u32 | 0e+0 |
| cuda | sum:rows | uint64 | — | 41.032 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.389 | 105.45 | f32 | 0e+0 |
| cuda | sum:cols | uint64 | — | 40.927 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.393 | 104.09 | f32 | 0e+0 |
| cuda | sumsq:rows | uint64 | — | — | — | — | — | — | — | — | — | 0.390 | — | f32 | 0e+0 |
| cuda | max:whole | uint64 | — | — | — | — | — | — | — | — | — | 0.174 | — | u64 | 0e+0 |
| cuda | max:rows | uint64 | — | — | — | — | — | — | — | — | — | 0.170 | — | u64 | 0e+0 |
| cuda | max:cols | uint64 | — | — | — | — | — | — | — | — | — | 0.177 | — | u64 | 0e+0 |
| cuda | min:rows | uint64 | — | — | — | — | — | — | — | — | — | 0.170 | — | u64 | 0e+0 |
| cuda | argmax:rows | uint64 | — | 41.091 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.326 | 126.01 | s32 index | 0e+0 |
| cuda | argmax:cols | uint64 | — | — | — | — | — | — | — | — | — | 0.525 | — | s32 index | 0e+0 |
| cuda | maprow:add | uint64 | — | — | — | — | — | — | — | — | — | 0.357 | — | u64 | 0e+0 |
| cuda | maprow:mul | uint64 | — | — | — | — | — | — | — | — | — | 0.358 | — | u64 | 0e+0 |
| cuda | transpose | uint64 | — | 121.193 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.680 | 178.25 | u64 | 0e+0 |
| cuda | gemv | uint64 | — | — | — | — | — | — | — | — | — | 1.425 | — | f32 | 0e+0 |
| cuda | gemv_t | uint64 | — | — | — | — | — | — | — | — | — | 1.478 | — | f32 | 0e+0 |
| cuda | layout:strided | uint64 | — | — | — | — | — | — | — | — | — | 0.698 | — | u64 | 0e+0 |
| cuda | layout:one_hot | uint64 | — | — | — | — | — | — | — | — | — | 0.175 | — | u64 | 0e+0 |
| cuda | layout:gather_rows | uint64 | — | — | — | — | — | — | — | — | — | 0.342 | — | u64 | 0e+0 |
| cuda | layout:gather_cols | uint64 | — | — | — | — | — | — | — | — | — | 0.384 | — | u64 | 0e+0 |
| cuda | layout:scatter_add_rows | uint64 | — | — | — | — | — | — | — | — | — | 0.683 | — | u32 | 0e+0 |
| cuda | layout:im2col | uint64 | — | — | — | — | — | — | — | — | — | 0.116 | — | u64 | 0e+0 |
| cuda | layout:col2im | uint64 | — | — | — | — | — | — | — | — | — | 0.254 | — | u32 | 0e+0 |
| cuda | layout:interpolate_nearest | uint64 | — | — | — | — | — | — | — | — | — | 0.241 | — | u64 | 0e+0 |
| cuda | layout:shift | uint64 | — | — | — | — | — | — | — | — | — | 0.357 | — | u64 | 0e+0 |
| cuda | layout:pad2d | uint64 | — | — | — | — | — | — | — | — | — | 0.357 | — | u64 | 0e+0 |
| cuda | layout:maxpool | uint64 | — | — | — | — | — | — | — | — | — | 0.021 | — | u64 | 0e+0 |
| cuda | layout:maxpool_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.198 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool | uint64 | — | — | — | — | — | — | — | — | — | 0.089 | — | u64 | 0e+0 |
| cuda | layout:depthwise_conv | uint64 | — | — | — | — | — | — | — | — | — | 0.236 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | uint64 | — | — | — | — | — | — | — | — | — | 6.890 | — | u32 | 0e+0 |
| cuda | matmul_nt | uint64 | — | — | — | — | — | — | — | — | — | 0.355 | — | u8mma | 0e+0 |
| cuda | matmul_tn | uint64 | — | — | — | — | — | — | — | — | — | 0.355 | — | u8mma | 0e+0 |
| cuda | matmul_tt | uint64 | — | — | — | — | — | — | — | — | — | 0.356 | — | u8mma | 0e+0 |
| cuda | matmul_batched | uint64 | — | — | — | — | — | — | — | — | — | 1.596 | — | u8mma | 0e+0 |
| cuda | matmul_bwd | uint64 | — | — | — | — | — | — | — | — | — | 0.719 | — | u8mma | 0e+0 |
| cuda | conv | uint64 | — | — | — | — | — | — | — | — | — | 0.205 | — | u8mma | 0e+0 |
| cuda | topk_rows | uint64 | — | — | — | — | — | — | — | — | — | 10.672 | — | u64 | 0e+0 |
| cuda | pick_rows | uint64 | — | — | — | — | — | — | — | — | — | 0.008 | — | u64 | 0e+0 |
| cuda | add | uint32 | 16777216 | 122.715 | — | add_f32; converted through the host | 0e+0 | 0.502 | 244.60 | add_u32 | 0e+0 | 0.263 | 466.60 | u32 | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 23.266 | — | gemm128_f32; converted through the host | 0e+0 | 1.091 | 21.33 | gemm128_u32 | 0e+0 | 0.211 | 110.27 | u8mma | 0e+0 |
| cuda | map:copy | uint32 | — | — | — | — | — | — | — | — | — | 0.183 | — | u32 | 0e+0 |
| cuda | map:fill | uint32 | — | 37.488 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.083 | 450.03 | u32 | 0e+0 |
| cuda | map:neg | uint32 | — | — | — | — | — | — | — | — | — | 0.175 | — | u32 | 0e+0 |
| cuda | map:abs | uint32 | — | 98.807 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.344 | 287.65 | u32 | 0e+0 |
| cuda | map:sign | uint32 | — | 97.374 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.343 | 283.56 | u32 | 0e+0 |
| cuda | map:square | uint32 | — | 98.305 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.344 | 285.69 | u32 | 0e+0 |
| cuda | map:relu | uint32 | — | 96.707 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.343 | 281.62 | u32 | 0e+0 |
| cuda | map:add_scalar | uint32 | — | — | — | — | — | — | — | — | — | 0.343 | — | u32 | 0e+0 |
| cuda | map:scale | uint32 | — | — | — | — | — | — | — | — | — | 0.181 | — | u32 | 0e+0 |
| cuda | map:affine | uint32 | — | 63.729 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.344 | 185.21 | u32 | 0e+0 |
| cuda | map:clamp | uint32 | — | 90.588 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.344 | 263.72 | u32 | 0e+0 |
| cuda | map:add | uint32 | — | — | — | — | — | — | — | — | — | 0.501 | — | u32 | 0e+0 |
| cuda | map:sub | uint32 | — | 137.615 | — | sub_f32; converted through the host | 4.29e+9 | — | — | — | — | 0.264 | 521.47 | u32 | 0e+0 |
| cuda | map:mul | uint32 | — | 146.615 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.501 | 292.59 | u32 | 0e+0 |
| cuda | map:div | uint32 | — | — | — | — | — | — | — | — | — | 0.263 | — | u32 | 0e+0 |
| cuda | map:max | uint32 | — | 99.335 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.501 | 198.15 | u32 | 0e+0 |
| cuda | map:min | uint32 | — | 148.008 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.262 | 564.91 | u32 | 0e+0 |
| cuda | map:axpy | uint32 | — | 100.208 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.502 | 199.50 | u32 | 0e+0 |
| cuda | map:muladd | uint32 | — | 128.098 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.345 | 371.73 | u32 | 0e+0 |
| cuda | map:where | uint32 | — | 149.853 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.643 | 233.20 | u32 | 0e+0 |
| cuda | map:square_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.262 | — | u32 | 0e+0 |
| cuda | map:abs_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.262 | — | u32 | 0e+0 |
| cuda | map:relu_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.261 | — | u32 | 0e+0 |
| cuda | map:clamp_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.263 | — | u32 | 0e+0 |
| cuda | map:where_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.261 | — | u32 | 0e+0 |
| cuda | sum:whole | uint32 | — | 26.864 | — | f32; converted through the host | 1e+0 | — | — | — | — | 0.173 | 155.55 | u64 | 0e+0 |
| cuda | sum:rows | uint32 | — | 27.021 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.169 | 159.79 | u64 | 0e+0 |
| cuda | sum:cols | uint32 | — | 27.187 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.176 | 154.38 | u64 | 0e+0 |
| cuda | sumsq:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.169 | — | u64 | 0e+0 |
| cuda | max:whole | uint32 | — | — | — | — | — | — | — | — | — | 0.173 | — | u32 | 0e+0 |
| cuda | max:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.169 | — | u32 | 0e+0 |
| cuda | max:cols | uint32 | — | — | — | — | — | — | — | — | — | 0.175 | — | u32 | 0e+0 |
| cuda | min:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.169 | — | u32 | 0e+0 |
| cuda | argmax:rows | uint32 | — | 26.816 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.169 | 158.49 | s32 index | 0e+0 |
| cuda | argmax:cols | uint32 | — | — | — | — | — | — | — | — | — | 0.513 | — | s32 index | 0e+0 |
| cuda | maprow:add | uint32 | — | — | — | — | — | — | — | — | — | 0.182 | — | u32 | 0e+0 |
| cuda | maprow:mul | uint32 | — | — | — | — | — | — | — | — | — | 0.181 | — | u32 | 0e+0 |
| cuda | transpose | uint32 | — | 94.673 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.350 | 270.80 | u32 | 0e+0 |
| cuda | gemv | uint32 | — | — | — | — | — | — | — | — | — | 1.431 | — | f32 | 0e+0 |
| cuda | gemv_t | uint32 | — | — | — | — | — | — | — | — | — | 0.813 | — | f32 | 0e+0 |
| cuda | layout:strided | uint32 | — | — | — | — | — | — | — | — | — | 0.451 | — | u32 | 0e+0 |
| cuda | layout:one_hot | uint32 | — | — | — | — | — | — | — | — | — | 0.134 | — | u32 | 0e+0 |
| cuda | layout:gather_rows | uint32 | — | — | — | — | — | — | — | — | — | 0.195 | — | u32 | 0e+0 |
| cuda | layout:gather_cols | uint32 | — | — | — | — | — | — | — | — | — | 0.253 | — | u32 | 0e+0 |
| cuda | layout:scatter_add_rows | uint32 | — | — | — | — | — | — | — | — | — | 0.472 | — | u32 | 0e+0 |
| cuda | layout:im2col | uint32 | — | — | — | — | — | — | — | — | — | 0.087 | — | u32 | 0e+0 |
| cuda | layout:col2im | uint32 | — | — | — | — | — | — | — | — | — | 0.119 | — | u32 | 0e+0 |
| cuda | layout:interpolate_nearest | uint32 | — | — | — | — | — | — | — | — | — | 0.152 | — | u32 | 0e+0 |
| cuda | layout:shift | uint32 | — | — | — | — | — | — | — | — | — | 0.182 | — | u32 | 0e+0 |
| cuda | layout:pad2d | uint32 | — | — | — | — | — | — | — | — | — | 0.181 | — | u32 | 0e+0 |
| cuda | layout:maxpool | uint32 | — | — | — | — | — | — | — | — | — | 0.021 | — | u32 | 0e+0 |
| cuda | layout:maxpool_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.172 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool | uint32 | — | — | — | — | — | — | — | — | — | 0.079 | — | u32 | 0e+0 |
| cuda | layout:depthwise_conv | uint32 | — | — | — | — | — | — | — | — | — | 0.208 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | uint32 | — | — | — | — | — | — | — | — | — | 7.050 | — | u32 | 0e+0 |
| cuda | matmul_nt | uint32 | — | — | — | — | — | — | — | — | — | 0.210 | — | u8mma | 0e+0 |
| cuda | matmul_tn | uint32 | — | — | — | — | — | — | — | — | — | 0.225 | — | u8mma | 0e+0 |
| cuda | matmul_tt | uint32 | — | — | — | — | — | — | — | — | — | 0.261 | — | u8mma | 0e+0 |
| cuda | matmul_batched | uint32 | — | — | — | — | — | — | — | — | — | 1.525 | — | u8mma | 0e+0 |
| cuda | matmul_bwd | uint32 | — | — | — | — | — | — | — | — | — | 0.510 | — | u8mma | 0e+0 |
| cuda | conv | uint32 | — | — | — | — | — | — | — | — | — | 0.173 | — | u8mma | 0e+0 |
| cuda | topk_rows | uint32 | — | — | — | — | — | — | — | — | — | 4.331 | — | u32 | 0e+0 |
| cuda | pick_rows | uint32 | — | — | — | — | — | — | — | — | — | 0.008 | — | u32 | 0e+0 |
| cuda | add | uint16 | 16777216 | 74.622 | — | add_f32; converted through the host | 0e+0 | 0.140 | 531.88 | add_u16 | 0e+0 | 0.124 | 599.86 | u16 | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 21.982 | — | gemm64_f32; converted through the host | 0e+0 | 1.330 | 16.53 | gemm128_u16 | 0e+0 | 0.170 | 129.16 | u8mma | 0e+0 |
| cuda | map:copy | uint16 | — | — | — | — | — | — | — | — | — | 0.082 | — | u32 | 0e+0 |
| cuda | map:fill | uint16 | — | 30.282 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.029 | 1030.00 | u32 | 0e+0 |
| cuda | map:neg | uint16 | — | — | — | — | — | — | — | — | — | 0.084 | — | u32 | 0e+0 |
| cuda | map:abs | uint16 | — | 47.626 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.084 | 566.97 | u32 | 0e+0 |
| cuda | map:sign | uint16 | — | 48.010 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 340.02 | u32 | 0e+0 |
| cuda | map:square | uint16 | — | 48.120 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 342.00 | u32 | 0e+0 |
| cuda | map:relu | uint16 | — | 46.961 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.140 | 334.72 | u32 | 0e+0 |
| cuda | map:add_scalar | uint16 | — | — | — | — | — | — | — | — | — | 0.142 | — | u32 | 0e+0 |
| cuda | map:scale | uint16 | — | — | — | — | — | — | — | — | — | 0.140 | — | u32 | 0e+0 |
| cuda | map:affine | uint16 | — | 48.171 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 342.85 | u32 | 0e+0 |
| cuda | map:clamp | uint16 | — | 46.777 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.141 | 332.93 | u32 | 0e+0 |
| cuda | map:add | uint16 | — | — | — | — | — | — | — | — | — | 0.238 | — | u32 | 0e+0 |
| cuda | map:sub | uint16 | — | 117.441 | — | sub_f32; converted through the host | 6.55e+4 | — | — | — | — | 0.240 | 490.36 | u32 | 0e+0 |
| cuda | map:mul | uint16 | — | 66.869 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.239 | 280.25 | u32 | 0e+0 |
| cuda | map:div | uint16 | — | — | — | — | — | — | — | — | — | 0.237 | — | u32 | 0e+0 |
| cuda | map:max | uint16 | — | 66.827 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.237 | 281.97 | u32 | 0e+0 |
| cuda | map:min | uint16 | — | 66.548 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.236 | 282.46 | u32 | 0e+0 |
| cuda | map:axpy | uint16 | — | 68.139 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.235 | 289.95 | u32 | 0e+0 |
| cuda | map:muladd | uint16 | — | 86.935 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.323 | 269.56 | u32 | 0e+0 |
| cuda | map:where | uint16 | — | 87.924 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.323 | 272.21 | u32 | 0e+0 |
| cuda | map:square_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.239 | — | u32 | 0e+0 |
| cuda | map:abs_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.234 | — | u32 | 0e+0 |
| cuda | map:relu_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.238 | — | u32 | 0e+0 |
| cuda | map:clamp_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.235 | — | u32 | 0e+0 |
| cuda | map:where_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.237 | — | u32 | 0e+0 |
| cuda | sum:whole | uint16 | — | 19.071 | — | f32; converted through the host | 1e+0 | — | — | — | — | 0.029 | 650.87 | u64 | 0e+0 |
| cuda | sum:rows | uint16 | — | 18.604 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.027 | 699.40 | u32 | 0e+0 |
| cuda | sum:cols | uint16 | — | 19.953 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.031 | 649.94 | u32 | 0e+0 |
| cuda | sumsq:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.029 | — | u64 | 0e+0 |
| cuda | max:whole | uint16 | — | — | — | — | — | — | — | — | — | 0.029 | — | u32 | 0e+0 |
| cuda | max:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.027 | — | u32 | 0e+0 |
| cuda | max:cols | uint16 | — | — | — | — | — | — | — | — | — | 0.031 | — | u32 | 0e+0 |
| cuda | min:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.027 | — | u32 | 0e+0 |
| cuda | argmax:rows | uint16 | — | 19.462 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.031 | 633.95 | s32 index | 0e+0 |
| cuda | argmax:cols | uint16 | — | — | — | — | — | — | — | — | — | 0.194 | — | s32 index | 0e+0 |
| cuda | maprow:add | uint16 | — | — | — | — | — | — | — | — | — | 0.088 | — | u32 | 0e+0 |
| cuda | maprow:mul | uint16 | — | — | — | — | — | — | — | — | — | 0.087 | — | u32 | 0e+0 |
| cuda | transpose | uint16 | — | 49.725 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.091 | 549.45 | u32 | 0e+0 |
| cuda | gemv | uint16 | — | — | — | — | — | — | — | — | — | 0.742 | — | f32 | 0e+0 |
| cuda | gemv_t | uint16 | — | — | — | — | — | — | — | — | — | 0.453 | — | f32 | 0e+0 |
| cuda | layout:strided | uint16 | — | — | — | — | — | — | — | — | — | 0.353 | — | u32 | 0e+0 |
| cuda | layout:one_hot | uint16 | — | — | — | — | — | — | — | — | — | 0.047 | — | u32 | 0e+0 |
| cuda | layout:gather_rows | uint16 | — | — | — | — | — | — | — | — | — | 0.090 | — | u32 | 0e+0 |
| cuda | layout:gather_cols | uint16 | — | — | — | — | — | — | — | — | — | 0.129 | — | u32 | 0e+0 |
| cuda | layout:scatter_add_rows | uint16 | — | — | — | — | — | — | — | — | — | 0.281 | — | u32 | 0e+0 |
| cuda | layout:im2col | uint16 | — | — | — | — | — | — | — | — | — | 0.068 | — | u32 | 0e+0 |
| cuda | layout:col2im | uint16 | — | — | — | — | — | — | — | — | — | 0.109 | — | u32 | 0e+0 |
| cuda | layout:interpolate_nearest | uint16 | — | — | — | — | — | — | — | — | — | 0.095 | — | u32 | 0e+0 |
| cuda | layout:shift | uint16 | — | — | — | — | — | — | — | — | — | 0.158 | — | u32 | 0e+0 |
| cuda | layout:pad2d | uint16 | — | — | — | — | — | — | — | — | — | 0.155 | — | u32 | 0e+0 |
| cuda | layout:maxpool | uint16 | — | — | — | — | — | — | — | — | — | 0.021 | — | u32 | 0e+0 |
| cuda | layout:maxpool_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.158 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool | uint16 | — | — | — | — | — | — | — | — | — | 0.145 | — | u32 | 0e+0 |
| cuda | layout:depthwise_conv | uint16 | — | — | — | — | — | — | — | — | — | 0.220 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | uint16 | — | — | — | — | — | — | — | — | — | 14.175 | — | u32 | 0e+0 |
| cuda | matmul_nt | uint16 | — | — | — | — | — | — | — | — | — | 0.170 | — | u8mma | 0e+0 |
| cuda | matmul_tn | uint16 | — | — | — | — | — | — | — | — | — | 0.170 | — | u8mma | 0e+0 |
| cuda | matmul_tt | uint16 | — | — | — | — | — | — | — | — | — | 0.189 | — | u8mma | 0e+0 |
| cuda | matmul_batched | uint16 | — | — | — | — | — | — | — | — | — | 1.562 | — | u8mma | 0e+0 |
| cuda | matmul_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.377 | — | u8mma | 0e+0 |
| cuda | conv | uint16 | — | — | — | — | — | — | — | — | — | 0.204 | — | u8mma | 0e+0 |
| cuda | topk_rows | uint16 | — | — | — | — | — | — | — | — | — | 3.546 | — | u32 | 0e+0 |
| cuda | pick_rows | uint16 | — | — | — | — | — | — | — | — | — | 0.009 | — | u32 | 0e+0 |
| cuda | add | uint8 | 16777216 | 64.427 | — | add_f32; converted through the host | 0e+0 | 0.107 | 601.00 | add_u8 | 0e+0 | 0.042 | 1530.34 | u8 | 0e+0 |
| cuda | matmul | uint8 | 2048x2048x2048 | 16.605 | — | gemm64_f32; converted through the host | 4.96e+3 | 1.286 | 12.91 | gemm128_u8 | 5.12e+3 | 0.093 | 178.16 | u8mma | 4.96e+3 |
| cuda | map:copy | uint8 | — | — | — | — | — | — | — | — | — | 0.023 | — | u32 | 0e+0 |
| cuda | map:fill | uint8 | — | 27.183 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.020 | 1393.99 | u32 | 0e+0 |
| cuda | map:neg | uint8 | — | — | — | — | — | — | — | — | — | 0.023 | — | u32 | 0e+0 |
| cuda | map:abs | uint8 | — | 45.714 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 2031.75 | u32 | 0e+0 |
| cuda | map:sign | uint8 | — | 43.742 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.024 | 1845.65 | u32 | 0e+0 |
| cuda | map:square | uint8 | — | 43.905 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.026 | 1695.19 | u32 | 0e+0 |
| cuda | map:relu | uint8 | — | 55.880 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 2408.60 | u32 | 0e+0 |
| cuda | map:add_scalar | uint8 | — | — | — | — | — | — | — | — | — | 0.023 | — | u32 | 0e+0 |
| cuda | map:scale | uint8 | — | — | — | — | — | — | — | — | — | 0.023 | — | u32 | 0e+0 |
| cuda | map:affine | uint8 | — | 46.880 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 2083.55 | u32 | 0e+0 |
| cuda | map:clamp | uint8 | — | 43.878 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.023 | 1883.18 | u32 | 0e+0 |
| cuda | map:add | uint8 | — | — | — | — | — | — | — | — | — | 0.558 | — | u32 | 0e+0 |
| cuda | map:sub | uint8 | — | 117.814 | — | sub_f32; converted through the host | 2.55e+2 | — | — | — | — | 0.053 | 2210.40 | u32 | 0e+0 |
| cuda | map:mul | uint8 | — | 63.071 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.055 | 1146.75 | u32 | 0e+0 |
| cuda | map:div | uint8 | — | — | — | — | — | — | — | — | — | 0.058 | — | u32 | 0e+0 |
| cuda | map:max | uint8 | — | 82.401 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.052 | 1572.54 | u32 | 0e+0 |
| cuda | map:min | uint8 | — | 65.254 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.052 | 1252.47 | u32 | 0e+0 |
| cuda | map:axpy | uint8 | — | 83.139 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.037 | 2271.55 | u32 | 0e+0 |
| cuda | map:muladd | uint8 | — | 81.411 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.074 | 1104.63 | u32 | 0e+0 |
| cuda | map:where | uint8 | — | 80.977 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.135 | 599.38 | u32 | 0e+0 |
| cuda | map:square_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.055 | — | u32 | 0e+0 |
| cuda | map:abs_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.053 | — | u32 | 0e+0 |
| cuda | map:relu_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.054 | — | u32 | 0e+0 |
| cuda | map:clamp_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.052 | — | u32 | 0e+0 |
| cuda | map:where_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.053 | — | u32 | 0e+0 |
| cuda | sum:whole | uint8 | — | 18.690 | — | f32; converted through the host | 1e+0 | — | — | — | — | 0.029 | 644.48 | u32 | 0e+0 |
| cuda | sum:rows | uint8 | — | 18.148 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.027 | 682.25 | f32 | 0e+0 |
| cuda | sum:cols | uint8 | — | 27.062 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.036 | 755.93 | f32 | 0e+0 |
| cuda | sumsq:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.029 | — | u32 | 0e+0 |
| cuda | max:whole | uint8 | — | — | — | — | — | — | — | — | — | 0.033 | — | u32 | 0e+0 |
| cuda | max:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.029 | — | u32 | 0e+0 |
| cuda | max:cols | uint8 | — | — | — | — | — | — | — | — | — | 0.034 | — | u32 | 0e+0 |
| cuda | min:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.029 | — | u32 | 0e+0 |
| cuda | argmax:rows | uint8 | — | 27.271 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.036 | 761.76 | s32 index | 0e+0 |
| cuda | argmax:cols | uint8 | — | — | — | — | — | — | — | — | — | 0.196 | — | s32 index | 0e+0 |
| cuda | maprow:add | uint8 | — | — | — | — | — | — | — | — | — | 0.040 | — | u32 | 0e+0 |
| cuda | maprow:mul | uint8 | — | — | — | — | — | — | — | — | — | 0.040 | — | u32 | 0e+0 |
| cuda | transpose | uint8 | — | 71.083 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.046 | 1541.93 | u32 | 0e+0 |
| cuda | gemv | uint8 | — | — | — | — | — | — | — | — | — | 0.129 | — | u32 | 1.9e+4 |
| cuda | gemv_t | uint8 | — | — | — | — | — | — | — | — | — | 0.149 | — | u32 | 1.91e+4 |
| cuda | layout:strided | uint8 | — | — | — | — | — | — | — | — | — | 0.312 | — | u32 | 0e+0 |
| cuda | layout:one_hot | uint8 | — | — | — | — | — | — | — | — | — | 0.036 | — | u32 | 0e+0 |
| cuda | layout:gather_rows | uint8 | — | — | — | — | — | — | — | — | — | 0.045 | — | u32 | 0e+0 |
| cuda | layout:gather_cols | uint8 | — | — | — | — | — | — | — | — | — | 0.046 | — | u32 | 0e+0 |
| cuda | layout:scatter_add_rows | uint8 | — | — | — | — | — | — | — | — | — | 0.173 | — | u32 | 0e+0 |
| cuda | layout:im2col | uint8 | — | — | — | — | — | — | — | — | — | 0.056 | — | u32 | 0e+0 |
| cuda | layout:col2im | uint8 | — | — | — | — | — | — | — | — | — | 0.050 | — | u32 | 0e+0 |
| cuda | layout:interpolate_nearest | uint8 | — | — | — | — | — | — | — | — | — | 0.079 | — | u32 | 0e+0 |
| cuda | layout:shift | uint8 | — | — | — | — | — | — | — | — | — | 0.043 | — | u32 | 0e+0 |
| cuda | layout:pad2d | uint8 | — | — | — | — | — | — | — | — | — | 0.043 | — | u32 | 0e+0 |
| cuda | layout:maxpool | uint8 | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | layout:maxpool_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.097 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool | uint8 | — | — | — | — | — | — | — | — | — | 0.266 | — | u32 | 0e+0 |
| cuda | layout:depthwise_conv | uint8 | — | — | — | — | — | — | — | — | — | 0.116 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool_bwd | uint8 | — | — | — | — | — | — | — | — | — | 23.184 | — | u32 | 0e+0 |
| cuda | matmul_nt | uint8 | — | — | — | — | — | — | — | — | — | 0.092 | — | u8mma | 5.04e+3 |
| cuda | matmul_tn | uint8 | — | — | — | — | — | — | — | — | — | 0.103 | — | u8mma | 5.01e+3 |
| cuda | matmul_tt | uint8 | — | — | — | — | — | — | — | — | — | 0.103 | — | u8mma | 5.01e+3 |
| cuda | matmul_batched | uint8 | — | — | — | — | — | — | — | — | — | 0.459 | — | u8mma | 1.24e+3 |
| cuda | matmul_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.185 | — | u8mma | 4.98e+3 |
| cuda | conv | uint8 | — | — | — | — | — | — | — | — | — | 0.107 | — | u8mma | 1.35e+3 |
| cuda | topk_rows | uint8 | — | — | — | — | — | — | — | — | — | 3.542 | — | u32 | 0e+0 |
| cuda | pick_rows | uint8 | — | — | — | — | — | — | — | — | — | 0.010 | — | u32 | 0e+0 |
| cuda | add | uint2 | 16777216 | 73.139 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 4353.52 | u8 | 0e+0 |
| cuda | matmul | uint2 | 2048x2048x2048 | 17.703 | — | gemm64_f32; converted through the host | 6.11e+2 | — | — | — | — | 0.109 | 163.16 | u8mma | 6.11e+2 |
| cuda | map:copy | uint2 | — | — | — | — | — | — | — | — | — | 0.014 | — | u32 | 0e+0 |
| cuda | map:fill | uint2 | — | 35.321 | — | fill_f32; converted through the host | 0e+0 | — | — | — | — | 0.012 | 3071.37 | u32 | 0e+0 |
| cuda | map:neg | uint2 | — | — | — | — | — | — | — | — | — | 0.014 | — | u32 | 0e+0 |
| cuda | map:abs | uint2 | — | 54.516 | — | abs_f32; converted through the host | 0e+0 | — | — | — | — | 0.014 | 3812.29 | u32 | 0e+0 |
| cuda | map:sign | uint2 | — | 53.683 | — | sign_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 3157.82 | u32 | 0e+0 |
| cuda | map:square | uint2 | — | 62.904 | — | square_f32; converted through the host | 0e+0 | — | — | — | — | 0.016 | 3907.09 | u32 | 0e+0 |
| cuda | map:relu | uint2 | — | 62.302 | — | relu_f32; converted through the host | 0e+0 | — | — | — | — | 0.016 | 3798.87 | u32 | 0e+0 |
| cuda | map:add_scalar | uint2 | — | — | — | — | — | — | — | — | — | 0.016 | — | u32 | 0e+0 |
| cuda | map:scale | uint2 | — | — | — | — | — | — | — | — | — | 0.016 | — | u32 | 0e+0 |
| cuda | map:affine | uint2 | — | 69.468 | — | affine_f32; converted through the host | 0e+0 | — | — | — | — | 0.016 | 4314.78 | u32 | 0e+0 |
| cuda | map:clamp | uint2 | — | 61.404 | — | clamp_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 3721.42 | u32 | 0e+0 |
| cuda | map:add | uint2 | — | — | — | — | — | — | — | — | — | 0.018 | — | u32 | 0e+0 |
| cuda | map:sub | uint2 | — | 157.624 | — | sub_f32; converted through the host | 3e+0 | — | — | — | — | 0.022 | 7331.34 | u32 | 0e+0 |
| cuda | map:mul | uint2 | — | 122.515 | — | mul_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 6658.42 | u32 | 0e+0 |
| cuda | map:div | uint2 | — | — | — | — | — | — | — | — | — | 0.035 | — | u32 | 0e+0 |
| cuda | map:max | uint2 | — | 122.059 | — | max_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 6633.64 | u32 | 0e+0 |
| cuda | map:min | uint2 | — | 91.573 | — | min_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4976.77 | u32 | 0e+0 |
| cuda | map:axpy | uint2 | — | 95.271 | — | axpy_f32; converted through the host; + copy of y | 0e+0 | — | — | — | — | 0.022 | 4431.22 | u32 | 0e+0 |
| cuda | map:muladd | uint2 | — | 121.690 | — | muladd_f32; converted through the host; + copy of z | 0e+0 | — | — | — | — | 0.024 | 5156.36 | u32 | 0e+0 |
| cuda | map:where | uint2 | — | 143.620 | — | where_f32; converted through the host | 0e+0 | — | — | — | — | 0.024 | 6034.44 | u32 | 0e+0 |
| cuda | map:square_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | map:abs_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.024 | — | u32 | 0e+0 |
| cuda | map:relu_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.021 | — | u32 | 0e+0 |
| cuda | map:clamp_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.023 | — | u32 | 0e+0 |
| cuda | map:where_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.021 | — | u32 | 0e+0 |
| cuda | sum:whole | uint2 | — | 38.196 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.035 | 1097.59 | u32 | 0e+0 |
| cuda | sum:rows | uint2 | — | 37.823 | — | f32; converted through the host | 0e+0 | — | — | — | — | 0.029 | 1317.88 | f32 | 0e+0 |
| cuda | sum:cols | uint2 | — | 39.239 | — | f32; converted through the host; + zero of y | 0e+0 | — | — | — | — | 0.036 | 1093.02 | f32 | 0e+0 |
| cuda | sumsq:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.029 | — | f32 | 0e+0 |
| cuda | max:whole | uint2 | — | — | — | — | — | — | — | — | — | 0.033 | — | u32 | 0e+0 |
| cuda | max:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.029 | — | u32 | 0e+0 |
| cuda | max:cols | uint2 | — | — | — | — | — | — | — | — | — | 0.035 | — | u32 | 0e+0 |
| cuda | min:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.029 | — | u32 | 0e+0 |
| cuda | argmax:rows | uint2 | — | 26.373 | — | f32 index; converted through the host | 0e+0 | — | — | — | — | 0.035 | 757.85 | s32 index | 0e+0 |
| cuda | argmax:cols | uint2 | — | — | — | — | — | — | — | — | — | 0.195 | — | s32 index | 0e+0 |
| cuda | maprow:add | uint2 | — | — | — | — | — | — | — | — | — | 0.038 | — | u32 | 0e+0 |
| cuda | maprow:mul | uint2 | — | — | — | — | — | — | — | — | — | 0.039 | — | u32 | 0e+0 |
| cuda | transpose | uint2 | — | 85.758 | — | transpose_f32; converted through the host | 0e+0 | — | — | — | — | 0.041 | 2096.76 | u32 | 0e+0 |
| cuda | gemv | uint2 | — | — | — | — | — | — | — | — | — | 0.138 | — | f32 | 2.19e+3 |
| cuda | gemv_t | uint2 | — | — | — | — | — | — | — | — | — | 0.152 | — | f32 | 2.19e+3 |
| cuda | layout:strided | uint2 | — | — | — | — | — | — | — | — | — | 0.318 | — | u32 | 0e+0 |
| cuda | layout:one_hot | uint2 | — | — | — | — | — | — | — | — | — | 0.033 | — | u32 | 0e+0 |
| cuda | layout:gather_rows | uint2 | — | — | — | — | — | — | — | — | — | 0.042 | — | u32 | 0e+0 |
| cuda | layout:gather_cols | uint2 | — | — | — | — | — | — | — | — | — | 0.066 | — | u32 | 0e+0 |
| cuda | layout:scatter_add_rows | uint2 | — | — | — | — | — | — | — | — | — | 0.156 | — | u32 | 2e+0 |
| cuda | layout:im2col | uint2 | — | — | — | — | — | — | — | — | — | 0.052 | — | u32 | 0e+0 |
| cuda | layout:col2im | uint2 | — | — | — | — | — | — | — | — | — | 0.063 | — | u32 | 6e+0 |
| cuda | layout:interpolate_nearest | uint2 | — | — | — | — | — | — | — | — | — | 0.078 | — | u32 | 0e+0 |
| cuda | layout:shift | uint2 | — | — | — | — | — | — | — | — | — | 0.042 | — | u32 | 0e+0 |
| cuda | layout:pad2d | uint2 | — | — | — | — | — | — | — | — | — | 0.042 | — | u32 | 0e+0 |
| cuda | layout:maxpool | uint2 | — | — | — | — | — | — | — | — | — | 0.028 | — | u32 | 0e+0 |
| cuda | layout:maxpool_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.167 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool | uint2 | — | — | — | — | — | — | — | — | — | 0.668 | — | u32 | 0e+0 |
| cuda | layout:depthwise_conv | uint2 | — | — | — | — | — | — | — | — | — | 0.122 | — | u32 | 4e+0 |
| cuda | layout:adaptive_maxpool_bwd | uint2 | — | — | — | — | — | — | — | — | — | 46.729 | — | u32 | 0e+0 |
| cuda | matmul_nt | uint2 | — | — | — | — | — | — | — | — | — | 0.109 | — | u8mma | 6.12e+2 |
| cuda | matmul_tn | uint2 | — | — | — | — | — | — | — | — | — | 0.114 | — | u8mma | 6.19e+2 |
| cuda | matmul_tt | uint2 | — | — | — | — | — | — | — | — | — | 0.110 | — | u8mma | 6.11e+2 |
| cuda | matmul_batched | uint2 | — | — | — | — | — | — | — | — | — | 0.540 | — | u8mma | 1.78e+2 |
| cuda | matmul_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.211 | — | u8mma | 6.16e+2 |
| cuda | conv | uint2 | — | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.93e+2 |
| cuda | topk_rows | uint2 | — | — | — | — | — | — | — | — | — | 3.309 | — | u32 | 0e+0 |
| cuda | pick_rows | uint2 | — | — | — | — | — | — | — | — | — | 0.013 | — | u32 | 0e+0 |
| cuda | add | q8_0 | 16777216 | — | — | — | — | — | — | — | — | 0.983 | — | f32 | 7.87e-3 |
| cuda | matmul | q8_0 | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.860 | — | f32 | 3.09e-1 |
| cuda | map:copy | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:fill | q8_0 | — | — | — | — | — | — | — | — | — | 0.551 | — | f32 (blocks through lane storage) | 1.83e-4 |
| cuda | map:neg | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:abs | q8_0 | — | — | — | — | — | — | — | — | — | 0.783 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:sign | q8_0 | — | — | — | — | — | — | — | — | — | 0.780 | — | f32 (blocks through lane storage) | 6.1e-5 |
| cuda | map:square | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 6.26e-2 |
| cuda | map:relu | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 1.56e-2 |
| cuda | map:add_scalar | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 2.36e-2 |
| cuda | map:scale | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 3.88e-3 |
| cuda | map:affine | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 3.53e-2 |
| cuda | map:clamp | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 7.87e-3 |
| cuda | map:add | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 3.15e-2 |
| cuda | map:sub | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 3.15e-2 |
| cuda | map:mul | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 6.29e-2 |
| cuda | map:div | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.59e-2 |
| cuda | map:max | q8_0 | — | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:min | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:axpy | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 4.71e-2 |
| cuda | map:muladd | q8_0 | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 7.85e-2 |
| cuda | map:where | q8_0 | — | — | — | — | — | — | — | — | — | 1.115 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:square_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.26e-1 |
| cuda | map:abs_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.56e-2 |
| cuda | map:relu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.56e-2 |
| cuda | map:clamp_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.56e-2 |
| cuda | map:where_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.56e-2 |
| cuda | map:sqrt | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 7.83e-3 |
| cuda | map:inv_sqrt | q8_0 | — | — | — | — | — | — | — | — | — | 0.782 | — | f32 (blocks through lane storage) | 5.64e-3 |
| cuda | map:exp | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 2.14e-1 |
| cuda | map:expm1 | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 2.54e-3 |
| cuda | map:log | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 5.45e-3 |
| cuda | map:sin | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | map:cos | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | map:tanh | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 3.93e-3 |
| cuda | map:sigmoid | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 3.86e-3 |
| cuda | map:erf | q8_0 | — | — | — | — | — | — | — | — | — | 1.408 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | map:gelu | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:gelu_erf | q8_0 | — | — | — | — | — | — | — | — | — | 1.990 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:silu | q8_0 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 (blocks through lane storage) | 1.54e-2 |
| cuda | map:leaky_relu | q8_0 | — | — | — | — | — | — | — | — | — | 0.782 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:pow | q8_0 | — | — | — | — | — | — | — | — | — | 0.783 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | map:dropout | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.95e-2 |
| cuda | map:gated_silu | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 6.16e-2 |
| cuda | map:gated_gelu | q8_0 | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 6.29e-2 |
| cuda | map:sqrt_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.12e-2 |
| cuda | map:inv_sqrt_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 2.28e-2 |
| cuda | map:exp_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 8.57e-1 |
| cuda | map:log_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 3.22e-2 |
| cuda | map:sin_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:cos_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:tanh_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:sigmoid_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 3.93e-3 |
| cuda | map:silu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.73e-2 |
| cuda | map:gelu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.78e-2 |
| cuda | map:gelu_erf_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 2.230 | — | f32 (blocks through lane storage) | 1.78e-2 |
| cuda | map:leaky_relu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | map:pow_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 3.15e-1 |
| cuda | map:gated_silu_bwd_x | q8_0 | — | — | — | — | — | — | — | — | — | 1.192 | — | f32 (blocks through lane storage) | 6.63e-2 |
| cuda | map:gated_silu_bwd_y | q8_0 | — | — | — | — | — | — | — | — | — | 1.110 | — | f32 (blocks through lane storage) | 6.17e-2 |
| cuda | map:gated_gelu_bwd_x | q8_0 | — | — | — | — | — | — | — | — | — | 1.193 | — | f32 (blocks through lane storage) | 6.28e-2 |
| cuda | map:gated_gelu_bwd_y | q8_0 | — | — | — | — | — | — | — | — | — | 1.110 | — | f32 (blocks through lane storage) | 6.29e-2 |
| cuda | sum:whole | q8_0 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 3.2e-4 |
| cuda | sum:rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.031 | — | f32 | 7.63e-6 |
| cuda | sum:cols | q8_0 | — | — | — | — | — | — | — | — | — | 0.036 | — | f32 | 2.67e-5 |
| cuda | sumsq:rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.031 | — | f32 | 1.74e-4 |
| cuda | max:whole | q8_0 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 0e+0 |
| cuda | max:rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.031 | — | f32 | 0e+0 |
| cuda | max:cols | q8_0 | — | — | — | — | — | — | — | — | — | 0.035 | — | f32 | 0e+0 |
| cuda | min:rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.031 | — | f32 | 0e+0 |
| cuda | argmax:rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.037 | — | s32 index | 0e+0 |
| cuda | argmax:cols | q8_0 | — | — | — | — | — | — | — | — | — | 0.228 | — | s32 index | 0e+0 |
| cuda | maprow:add | q8_0 | — | — | — | — | — | — | — | — | — | 1.040 | — | f32 (blocks through lane storage) | 7.87e-3 |
| cuda | maprow:mul | q8_0 | — | — | — | — | — | — | — | — | — | 0.784 | — | f32 (blocks through lane storage) | 3.93e-3 |
| cuda | transpose | q8_0 | — | — | — | — | — | — | — | — | — | 0.788 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | gemv | q8_0 | — | — | — | — | — | — | — | — | — | 0.447 | — | s8dp blocks | 4.75e-1 |
| cuda | gemv_t | q8_0 | — | — | — | — | — | — | — | — | — | 0.847 | — | f32 (blocks through lane storage) | 5.89e-1 |
| cuda | row:log_softmax | q8_0 | — | — | — | — | — | — | — | — | — | 0.782 | — | f32 (blocks through lane storage) | 5.63e-2 |
| cuda | row:softmax_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.984 | — | f32 (blocks through lane storage) | 3.11e-1 |
| cuda | row:log_softmax_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 5.06e-1 |
| cuda | row:rms_norm_affine | q8_0 | — | — | — | — | — | — | — | — | — | 0.782 | — | f32 (blocks through lane storage) | 1.38e-2 |
| cuda | row:rms_norm_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 3.46e-3 |
| cuda | row:layer_norm | q8_0 | — | — | — | — | — | — | — | — | — | 0.786 | — | f32 (blocks through lane storage) | 1.75e-2 |
| cuda | row:layer_norm_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 3.54e-3 |
| cuda | row:weight_norm | q8_0 | — | — | — | — | — | — | — | — | — | 0.783 | — | f32 (blocks through lane storage) | 1.51e-5 |
| cuda | row:rope | q8_0 | — | — | — | — | — | — | — | — | — | 0.780 | — | f32 (blocks through lane storage) | 2.23e-2 |
| cuda | row:scale_mask_softmax | q8_0 | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.42e-3 |
| cuda | row:add_rms_norm | q8_0 | — | — | — | — | — | — | — | — | — | 1.557 | — | f32 (blocks through lane storage) | 1.94e-2 |
| cuda | row:add_layer_norm | q8_0 | — | — | — | — | — | — | — | — | — | 1.558 | — | f32 (blocks through lane storage) | 2.26e-2 |
| cuda | row:cross_entropy_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.783 | — | f32 (blocks through lane storage) | 2.08e-3 |
| cuda | norm_params_bwd:rms | q8_0 | — | — | — | — | — | — | — | — | — | 0.106 | — | f32 | NaN |
| cuda | norm_params_bwd:layer | q8_0 | — | — | — | — | — | — | — | — | — | 0.113 | — | f32 | NaN |
| cuda | layout:strided | q8_0 | — | — | — | — | — | — | — | — | — | 1.006 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | layout:one_hot | q8_0 | — | — | — | — | — | — | — | — | — | 0.621 | — | f32 (blocks through lane storage) | 6.1e-5 |
| cuda | layout:gather_rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.796 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | layout:gather_cols | q8_0 | — | — | — | — | — | — | — | — | — | 0.855 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | layout:scatter_add_rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.885 | — | f32 (blocks through lane storage) | 1.68e-2 |
| cuda | layout:im2col | q8_0 | — | — | — | — | — | — | — | — | — | 0.368 | — | f32 (blocks through lane storage) | 3.93e-3 |
| cuda | layout:col2im | q8_0 | — | — | — | — | — | — | — | — | — | 0.150 | — | f32 (blocks through lane storage) | 2.95e-2 |
| cuda | layout:interpolate_nearest | q8_0 | — | — | — | — | — | — | — | — | — | 0.649 | — | f32 (blocks through lane storage) | 3.91e-3 |
| cuda | layout:interpolate_bilinear | q8_0 | — | — | — | — | — | — | — | — | — | 0.669 | — | f32 (blocks through lane storage) | 3.92e-3 |
| cuda | layout:shift | q8_0 | — | — | — | — | — | — | — | — | — | 0.785 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | layout:pad2d | q8_0 | — | — | — | — | — | — | — | — | — | 0.786 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | layout:maxpool | q8_0 | — | — | — | — | — | — | — | — | — | 0.074 | — | f32 (blocks through lane storage) | 3.94e-3 |
| cuda | layout:avgpool | q8_0 | — | — | — | — | — | — | — | — | — | 0.075 | — | f32 (blocks through lane storage) | 3.79e-3 |
| cuda | layout:maxpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.275 | — | f32 (blocks through lane storage) | 3.91e-3 |
| cuda | layout:avgpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.261 | — | f32 (blocks through lane storage) | 9.77e-4 |
| cuda | layout:adaptive_maxpool | q8_0 | — | — | — | — | — | — | — | — | — | 0.103 | — | f32 (blocks through lane storage) | 3.88e-3 |
| cuda | layout:adaptive_avgpool | q8_0 | — | — | — | — | — | — | — | — | — | 0.102 | — | f32 (blocks through lane storage) | 2.32e-4 |
| cuda | layout:depthwise_conv | q8_0 | — | — | — | — | — | — | — | — | — | 0.240 | — | f32 (blocks through lane storage) | 1.95e-2 |
| cuda | layout:adaptive_maxpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 7.166 | — | f32 (blocks through lane storage) | 2.85e-3 |
| cuda | layout:adaptive_avgpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 0.309 | — | f32 (blocks through lane storage) | 7.6e-6 |
| cuda | matmul_nt | q8_0 | — | — | — | — | — | — | — | — | — | 0.540 | — | s8 mma blocks | 3.22e-1 |
| cuda | matmul_tn | q8_0 | — | — | — | — | — | — | — | — | — | 1.004 | — | f32 (blocks through lane storage) | 3.1e-1 |
| cuda | matmul_tt | q8_0 | — | — | — | — | — | — | — | — | — | 1.051 | — | f32 (blocks through lane storage) | 3.09e-1 |
| cuda | matmul_batched | q8_0 | — | — | — | — | — | — | — | — | — | 1.132 | — | f32 (blocks through lane storage) | 1.58e-1 |
| cuda | matmul_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 1.526 | — | s8 mma blocks | 3.05e-1 |
| cuda | matmul_blocks | q8_0 | — | — | — | — | — | — | — | — | — | 0.539 | — | s8 mma blocks | 3.22e-1 |
| cuda | attention | q8_0 | — | — | — | — | — | — | — | — | — | 3.951 | — | f32 (blocks through lane storage) | 2.44e-4 |
| cuda | attention_causal | q8_0 | — | — | — | — | — | — | — | — | — | 2.319 | — | f32 (blocks through lane storage) | 3.73e-3 |
| cuda | attention_fused | q8_0 | — | — | — | — | — | — | — | — | — | 3.952 | — | f32 (blocks through lane storage) | 2.44e-4 |
| cuda | attention_fused_causal | q8_0 | — | — | — | — | — | — | — | — | — | 2.322 | — | f32 (blocks through lane storage) | 3.73e-3 |
| cuda | attention_decode | q8_0 | — | — | — | — | — | — | — | — | — | 0.884 | — | f32 (blocks through lane storage) | 1.39e-4 |
| cuda | attention_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 4.137 | — | f32 | 1.56e-3 |
| cuda | optimizer:sgd_momentum | q8_0 | — | — | — | — | — | — | — | — | — | 1.347 | — | f32 (blocks through lane storage); + restore of p, m, v | 3.34e-3 |
| cuda | optimizer:adam | q8_0 | — | — | — | — | — | — | — | — | — | 1.695 | — | f32 (blocks through lane storage); + restore of p, m, v | 3.92e-3 |
| cuda | optimizer:adamw | q8_0 | — | — | — | — | — | — | — | — | — | 1.698 | — | f32 (blocks through lane storage); + restore of p, m, v | 3.93e-3 |
| cuda | conv | q8_0 | — | — | — | — | — | — | — | — | — | 0.220 | — | f32 | 1.67e-1 |
| cuda | conv_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 2.257 | — | f32 | 2.23e-1 |
| cuda | lstm_step | q8_0 | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 | 2.12e-3 |
| cuda | gru_step | q8_0 | — | — | — | — | — | — | — | — | — | 0.999 | — | f32 | 3.58e-3 |
| cuda | lstm_step_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 3.161 | — | f32 | 2.42e-3 |
| cuda | gru_step_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 2.429 | — | f32 | 2.33e-3 |
| cuda | gru_seq | q8_0 | — | — | — | — | — | — | — | — | — | 2.676 | — | f32 (blocks through lane storage) | 3.34e-3 |
| cuda | gru_seq_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 16.686 | — | f32 (blocks through lane storage) | 2.64e-3 |
| cuda | topk_rows | q8_0 | — | — | — | — | — | — | — | — | — | 8.356 | — | f32 (blocks through lane storage) | 3.88e-3 |
| cuda | pick_rows | q8_0 | — | — | — | — | — | — | — | — | — | 1.806 | — | f32 (blocks through lane storage) | 1.57e-2 |
| cuda | sample_rows | q8_0 | — | — | — | — | — | — | — | — | — | 2.419 | — | f32 | 1e+0 |
| cuda | cross_entropy | q8_0 | — | — | — | — | — | — | — | — | — | 31.682 | — | f32 (blocks through lane storage) (log_softmax, pick, neg) | 1.25e-1 |
| cuda | ctc | q8_0 | — | — | — | — | — | — | — | — | — | 0.526 | — | f32 (blocks through lane storage) | 6.44e+0 |
| cuda | ctc_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 1.328 | — | f32 (blocks through lane storage) | 3.91e-3 |
| cuda | softmax | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.782 | — | f32 (blocks through lane storage) | 8.37e-6 |
| cuda | rms_norm | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.776 | — | f32 (blocks through lane storage) | 4.85e-4 |
| cuda | add | q4_0 | 16777216 | — | — | — | — | — | — | — | — | 0.660 | — | f32 | 2.34e-1 |
| cuda | matmul | q4_0 | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.780 | — | f32 | 6.51e+0 |
| cuda | map:copy | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:fill | q4_0 | — | — | — | — | — | — | — | — | — | 0.235 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:neg | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:abs | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:sign | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | map:square | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 1e+0 |
| cuda | map:relu | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 1.88e-1 |
| cuda | map:add_scalar | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 3.61e-1 |
| cuda | map:scale | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 3.91e-3 |
| cuda | map:affine | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 8.75e-1 |
| cuda | map:clamp | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | map:add | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 9.37e-1 |
| cuda | map:sub | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 9.38e-1 |
| cuda | map:mul | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 1.75e+0 |
| cuda | map:div | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 5.06e-1 |
| cuda | map:max | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:min | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:axpy | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 1.38e+0 |
| cuda | map:muladd | q4_0 | — | — | — | — | — | — | — | — | — | 0.864 | — | f32 (blocks through lane storage) | 2.18e+0 |
| cuda | map:where | q4_0 | — | — | — | — | — | — | — | — | — | 0.818 | — | f32 (blocks through lane storage) | 5e-1 |
| cuda | map:square_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 3.5e+0 |
| cuda | map:abs_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 5e-1 |
| cuda | map:relu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:clamp_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.664 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:where_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:sqrt | q4_0 | — | — | — | — | — | — | — | — | — | 0.460 | — | f32 (blocks through lane storage) | 1.21e-1 |
| cuda | map:inv_sqrt | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 8.82e-2 |
| cuda | map:exp | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 2.72e+0 |
| cuda | map:expm1 | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 4.04e-2 |
| cuda | map:log | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 9.81e-2 |
| cuda | map:sin | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | map:cos | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | map:tanh | q4_0 | — | — | — | — | — | — | — | — | — | 0.462 | — | f32 (blocks through lane storage) | 1.24e-1 |
| cuda | map:sigmoid | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 6.13e-2 |
| cuda | map:erf | q4_0 | — | — | — | — | — | — | — | — | — | 2.027 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | map:gelu | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 2.05e-1 |
| cuda | map:gelu_erf | q4_0 | — | — | — | — | — | — | — | — | — | 3.255 | — | f32 (blocks through lane storage) | 2.04e-1 |
| cuda | map:silu | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 2.44e-1 |
| cuda | map:leaky_relu | q4_0 | — | — | — | — | — | — | — | — | — | 0.460 | — | f32 (blocks through lane storage) | 2.5e-1 |
| cuda | map:pow | q4_0 | — | — | — | — | — | — | — | — | — | 0.461 | — | f32 (blocks through lane storage) | 1.88e+0 |
| cuda | map:dropout | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 5.48e-1 |
| cuda | map:gated_silu | q4_0 | — | — | — | — | — | — | — | — | — | 0.664 | — | f32 (blocks through lane storage) | 1.72e+0 |
| cuda | map:gated_gelu | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 1.75e+0 |
| cuda | map:sqrt_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 3.36e-1 |
| cuda | map:inv_sqrt_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 8.24e-1 |
| cuda | map:exp_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 2.39e+1 |
| cuda | map:log_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 1.04e+0 |
| cuda | map:sin_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 5e-1 |
| cuda | map:cos_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 5e-1 |
| cuda | map:tanh_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:sigmoid_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 1.09e-1 |
| cuda | map:silu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.664 | — | f32 (blocks through lane storage) | 4.82e-1 |
| cuda | map:gelu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 4.94e-1 |
| cuda | map:gelu_erf_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.476 | — | f32 (blocks through lane storage) | 4.94e-1 |
| cuda | map:leaky_relu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 4.38e-1 |
| cuda | map:pow_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 8.77e+0 |
| cuda | map:gated_silu_bwd_x | q4_0 | — | — | — | — | — | — | — | — | — | 0.868 | — | f32 (blocks through lane storage) | 1.9e+0 |
| cuda | map:gated_silu_bwd_y | q4_0 | — | — | — | — | — | — | — | — | — | 0.783 | — | f32 (blocks through lane storage) | 1.72e+0 |
| cuda | map:gated_gelu_bwd_x | q4_0 | — | — | — | — | — | — | — | — | — | 0.868 | — | f32 (blocks through lane storage) | 1.77e+0 |
| cuda | map:gated_gelu_bwd_y | q4_0 | — | — | — | — | — | — | — | — | — | 0.782 | — | f32 (blocks through lane storage) | 1.75e+0 |
| cuda | sum:whole | q4_0 | — | — | — | — | — | — | — | — | — | 0.037 | — | f32 | 1.83e-4 |
| cuda | sum:rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 0e+0 |
| cuda | sum:cols | q4_0 | — | — | — | — | — | — | — | — | — | 0.039 | — | f32 | 0e+0 |
| cuda | sumsq:rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 1.7e-4 |
| cuda | max:whole | q4_0 | — | — | — | — | — | — | — | — | — | 0.037 | — | f32 | 0e+0 |
| cuda | max:rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 0e+0 |
| cuda | max:cols | q4_0 | — | — | — | — | — | — | — | — | — | 0.039 | — | f32 | 0e+0 |
| cuda | min:rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 0e+0 |
| cuda | argmax:rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.039 | — | s32 index | 0e+0 |
| cuda | argmax:cols | q4_0 | — | — | — | — | — | — | — | — | — | 0.224 | — | s32 index | 0e+0 |
| cuda | maprow:add | q4_0 | — | — | — | — | — | — | — | — | — | 0.772 | — | f32 (blocks through lane storage) | 2.34e-1 |
| cuda | maprow:mul | q4_0 | — | — | — | — | — | — | — | — | — | 0.462 | — | f32 (blocks through lane storage) | 1.09e-1 |
| cuda | transpose | q4_0 | — | — | — | — | — | — | — | — | — | 0.464 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | gemv | q4_0 | — | — | — | — | — | — | — | — | — | 0.243 | — | s32 blocks | 1.03e+1 |
| cuda | gemv_t | q4_0 | — | — | — | — | — | — | — | — | — | 0.830 | — | f32 (blocks through lane storage) | 1.14e+1 |
| cuda | row:log_softmax | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 8.92e-1 |
| cuda | row:softmax_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.661 | — | f32 (blocks through lane storage) | 1.89e+0 |
| cuda | row:log_softmax_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.660 | — | f32 (blocks through lane storage) | 8.32e+0 |
| cuda | row:rms_norm_affine | q4_0 | — | — | — | — | — | — | — | — | — | 0.460 | — | f32 (blocks through lane storage) | 3.89e-1 |
| cuda | row:rms_norm_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.662 | — | f32 (blocks through lane storage) | 9.72e-2 |
| cuda | row:layer_norm | q4_0 | — | — | — | — | — | — | — | — | — | 0.463 | — | f32 (blocks through lane storage) | 4.9e-1 |
| cuda | row:layer_norm_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.663 | — | f32 (blocks through lane storage) | 9.7e-2 |
| cuda | row:weight_norm | q4_0 | — | — | — | — | — | — | — | — | — | 0.461 | — | f32 (blocks through lane storage) | 1.53e-5 |
| cuda | row:rope | q4_0 | — | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 6.94e-1 |
| cuda | row:scale_mask_softmax | q4_0 | — | — | — | — | — | — | — | — | — | 0.661 | — | f32 (blocks through lane storage) | 3.11e-2 |
| cuda | row:add_rms_norm | q4_0 | — | — | — | — | — | — | — | — | — | 0.917 | — | f32 (blocks through lane storage) | 5.78e-1 |
| cuda | row:add_layer_norm | q4_0 | — | — | — | — | — | — | — | — | — | 0.918 | — | f32 (blocks through lane storage) | 6.37e-1 |
| cuda | row:cross_entropy_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.459 | — | f32 (blocks through lane storage) | 2.14e-3 |
| cuda | norm_params_bwd:rms | q4_0 | — | — | — | — | — | — | — | — | — | 0.106 | — | f32 | NaN |
| cuda | norm_params_bwd:layer | q4_0 | — | — | — | — | — | — | — | — | — | 0.110 | — | f32 | NaN |
| cuda | layout:strided | q4_0 | — | — | — | — | — | — | — | — | — | 1.005 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | layout:one_hot | q4_0 | — | — | — | — | — | — | — | — | — | 0.303 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | layout:gather_rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.470 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | layout:gather_cols | q4_0 | — | — | — | — | — | — | — | — | — | 0.533 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | layout:scatter_add_rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.566 | — | f32 (blocks through lane storage) | 4.39e-1 |
| cuda | layout:im2col | q4_0 | — | — | — | — | — | — | — | — | — | 0.189 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | layout:col2im | q4_0 | — | — | — | — | — | — | — | — | — | 0.100 | — | f32 (blocks through lane storage) | 6.46e-1 |
| cuda | layout:interpolate_nearest | q4_0 | — | — | — | — | — | — | — | — | — | 0.328 | — | f32 (blocks through lane storage) | 1.09e-1 |
| cuda | layout:interpolate_bilinear | q4_0 | — | — | — | — | — | — | — | — | — | 0.348 | — | f32 (blocks through lane storage) | 1.12e-1 |
| cuda | layout:shift | q4_0 | — | — | — | — | — | — | — | — | — | 0.461 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | layout:pad2d | q4_0 | — | — | — | — | — | — | — | — | — | 0.463 | — | f32 (blocks through lane storage) | 1.25e-1 |
| cuda | layout:maxpool | q4_0 | — | — | — | — | — | — | — | — | — | 0.054 | — | f32 (blocks through lane storage) | 1.09e-1 |
| cuda | layout:avgpool | q4_0 | — | — | — | — | — | — | — | — | — | 0.056 | — | f32 (blocks through lane storage) | 1.06e-1 |
| cuda | layout:maxpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.196 | — | f32 (blocks through lane storage) | 1.09e-1 |
| cuda | layout:avgpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.191 | — | f32 (blocks through lane storage) | 2.73e-2 |
| cuda | layout:adaptive_maxpool | q4_0 | — | — | — | — | — | — | — | — | — | 0.102 | — | f32 (blocks through lane storage) | 1.17e-2 |
| cuda | layout:adaptive_avgpool | q4_0 | — | — | — | — | — | — | — | — | — | 0.102 | — | f32 (blocks through lane storage) | 4.53e-3 |
| cuda | layout:depthwise_conv | q4_0 | — | — | — | — | — | — | — | — | — | 0.163 | — | f32 (blocks through lane storage) | 4.7e-1 |
| cuda | layout:adaptive_maxpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 7.088 | — | f32 (blocks through lane storage) | 1.09e-1 |
| cuda | layout:adaptive_avgpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 0.232 | — | f32 (blocks through lane storage) | 1.35e-4 |
| cuda | matmul_nt | q4_0 | — | — | — | — | — | — | — | — | — | 0.375 | — | s8 mma blocks | 7.47e+0 |
| cuda | matmul_tn | q4_0 | — | — | — | — | — | — | — | — | — | 0.924 | — | f32 (blocks through lane storage) | 6.72e+0 |
| cuda | matmul_tt | q4_0 | — | — | — | — | — | — | — | — | — | 0.972 | — | f32 (blocks through lane storage) | 6.17e+0 |
| cuda | matmul_batched | q4_0 | — | — | — | — | — | — | — | — | — | 1.046 | — | f32 (blocks through lane storage) | 3.32e+0 |
| cuda | matmul_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 1.280 | — | s8 mma blocks | 6.37e+0 |
| cuda | matmul_blocks | q4_0 | — | — | — | — | — | — | — | — | — | 0.376 | — | s8 mma blocks | 7.47e+0 |
| cuda | attention | q4_0 | — | — | — | — | — | — | — | — | — | 3.944 | — | f32 (blocks through lane storage) | 6.18e-3 |
| cuda | attention_causal | q4_0 | — | — | — | — | — | — | — | — | — | 2.238 | — | f32 (blocks through lane storage) | 1.06e-1 |
| cuda | attention_fused | q4_0 | — | — | — | — | — | — | — | — | — | 3.871 | — | f32 (blocks through lane storage) | 6.18e-3 |
| cuda | attention_fused_causal | q4_0 | — | — | — | — | — | — | — | — | — | 2.241 | — | f32 (blocks through lane storage) | 1.06e-1 |
| cuda | attention_decode | q4_0 | — | — | — | — | — | — | — | — | — | 0.876 | — | f32 (blocks through lane storage) | 3e-3 |
| cuda | attention_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 4.071 | — | f32 | 3.43e-2 |
| cuda | optimizer:sgd_momentum | q4_0 | — | — | — | — | — | — | — | — | — | 1.007 | — | f32 (blocks through lane storage); + restore of p, m, v | 2.92e-3 |
| cuda | optimizer:adam | q4_0 | — | — | — | — | — | — | — | — | — | 1.363 | — | f32 (blocks through lane storage); + restore of p, m, v | 3.16e-2 |
| cuda | optimizer:adamw | q4_0 | — | — | — | — | — | — | — | — | — | 1.360 | — | f32 (blocks through lane storage); + restore of p, m, v | 3.16e-2 |
| cuda | conv | q4_0 | — | — | — | — | — | — | — | — | — | 0.179 | — | f32 | 3.39e+0 |
| cuda | conv_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 2.229 | — | f32 | 4.35e+0 |
| cuda | lstm_step | q4_0 | — | — | — | — | — | — | — | — | — | 1.365 | — | f32 | 5.17e-2 |
| cuda | gru_step | q4_0 | — | — | — | — | — | — | — | — | — | 0.878 | — | f32 | 9.63e-2 |
| cuda | lstm_step_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 2.960 | — | f32 | 5.03e-2 |
| cuda | gru_step_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 2.284 | — | f32 | 4.61e-2 |
| cuda | gru_seq | q4_0 | — | — | — | — | — | — | — | — | — | 2.604 | — | f32 (blocks through lane storage) | 1e-1 |
| cuda | gru_seq_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 14.923 | — | f32 (blocks through lane storage) | 5.31e-2 |
| cuda | topk_rows | q4_0 | — | — | — | — | — | — | — | — | — | 7.700 | — | f32 (blocks through lane storage) | 3.91e-3 |
| cuda | pick_rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.975 | — | f32 (blocks through lane storage) | 4.98e-1 |
| cuda | sample_rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.577 | — | f32 | 1e+0 |
| cuda | cross_entropy | q4_0 | — | — | — | — | — | — | — | — | — | 20.375 | — | f32 (blocks through lane storage) (log_softmax, pick, neg) | 1.96e+0 |
| cuda | ctc | q4_0 | — | — | — | — | — | — | — | — | — | 0.520 | — | f32 (blocks through lane storage) | 1.07e+2 |
| cuda | ctc_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 1.212 | — | f32 (blocks through lane storage) | 5.63e-2 |
| cuda | softmax | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.460 | — | f32 (blocks through lane storage) | 1.11e-4 |
| cuda | rms_norm | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.457 | — | f32 (blocks through lane storage) | 4.88e-4 |
| cuda | add | q4_k | 16777216 | — | — | — | — | — | — | — | — | 1.196 | — | f32 | 1.33e-1 |
| cuda | matmul | q4_k | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.871 | — | f32 | 3.83e+0 |
| cuda | map:copy | q4_k | — | — | — | — | — | — | — | — | — | 0.991 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:fill | q4_k | — | — | — | — | — | — | — | — | — | 0.740 | — | f32 (blocks through lane storage) | 7.32e-4 |
| cuda | map:neg | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 3.24e-2 |
| cuda | map:abs | q4_k | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.36e-1 |
| cuda | map:sign | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 6.61e-2 |
| cuda | map:square | q4_k | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 5.55e-1 |
| cuda | map:relu | q4_k | — | — | — | — | — | — | — | — | — | 0.992 | — | f32 (blocks through lane storage) | 1.36e-1 |
| cuda | map:add_scalar | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.99e-1 |
| cuda | map:scale | q4_k | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 1.11e-2 |
| cuda | map:affine | q4_k | — | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 5.47e-2 |
| cuda | map:clamp | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1e-1 |
| cuda | map:add | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 5.31e-1 |
| cuda | map:sub | q4_k | — | — | — | — | — | — | — | — | — | 1.193 | — | f32 (blocks through lane storage) | 5.31e-1 |
| cuda | map:mul | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 1.07e+0 |
| cuda | map:div | q4_k | — | — | — | — | — | — | — | — | — | 1.193 | — | f32 (blocks through lane storage) | 2.7e-1 |
| cuda | map:max | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 2.66e-1 |
| cuda | map:min | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 2.66e-1 |
| cuda | map:axpy | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 7.81e-1 |
| cuda | map:muladd | q4_k | — | — | — | — | — | — | — | — | — | 1.401 | — | f32 (blocks through lane storage) | 1.32e+0 |
| cuda | map:where | q4_k | — | — | — | — | — | — | — | — | — | 1.345 | — | f32 (blocks through lane storage) | 2.68e-1 |
| cuda | map:square_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.193 | — | f32 (blocks through lane storage) | 2.15e+0 |
| cuda | map:abs_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 2.73e-1 |
| cuda | map:relu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.194 | — | f32 (blocks through lane storage) | 2.65e-1 |
| cuda | map:clamp_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 2.65e-1 |
| cuda | map:where_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 2.65e-1 |
| cuda | map:sqrt | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 6.6e-2 |
| cuda | map:inv_sqrt | q4_k | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 5.22e-2 |
| cuda | map:exp | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.96e+0 |
| cuda | map:expm1 | q4_k | — | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 3.47e-2 |
| cuda | map:log | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 6.72e-2 |
| cuda | map:sin | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 6.67e-2 |
| cuda | map:cos | q4_k | — | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 6.67e-2 |
| cuda | map:tanh | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 6.66e-2 |
| cuda | map:sigmoid | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 3.28e-2 |
| cuda | map:erf | q4_k | — | — | — | — | — | — | — | — | — | 1.411 | — | f32 (blocks through lane storage) | 6.67e-2 |
| cuda | map:gelu | q4_k | — | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 1.41e-1 |
| cuda | map:gelu_erf | q4_k | — | — | — | — | — | — | — | — | — | 1.793 | — | f32 (blocks through lane storage) | 1.42e-1 |
| cuda | map:silu | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 1.43e-1 |
| cuda | map:leaky_relu | q4_k | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 1.48e-1 |
| cuda | map:pow | q4_k | — | — | — | — | — | — | — | — | — | 0.989 | — | f32 (blocks through lane storage) | 1.04e+0 |
| cuda | map:dropout | q4_k | — | — | — | — | — | — | — | — | — | 1.199 | — | f32 (blocks through lane storage) | 3.32e-1 |
| cuda | map:gated_silu | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 1.05e+0 |
| cuda | map:gated_gelu | q4_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.07e+0 |
| cuda | map:sqrt_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 1.91e-1 |
| cuda | map:inv_sqrt_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 4.15e-1 |
| cuda | map:exp_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 1.51e+1 |
| cuda | map:log_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 5.64e-1 |
| cuda | map:sin_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 2.71e-1 |
| cuda | map:cos_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 2.69e-1 |
| cuda | map:tanh_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 2.66e-1 |
| cuda | map:sigmoid_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 6.64e-2 |
| cuda | map:silu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 2.93e-1 |
| cuda | map:gelu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 3e-1 |
| cuda | map:gelu_erf_bwd | q4_k | — | — | — | — | — | — | — | — | — | 2.203 | — | f32 (blocks through lane storage) | 3e-1 |
| cuda | map:leaky_relu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.200 | — | f32 (blocks through lane storage) | 2.64e-1 |
| cuda | map:pow_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.194 | — | f32 (blocks through lane storage) | 5.3e+0 |
| cuda | map:gated_silu_bwd_x | q4_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 1.12e+0 |
| cuda | map:gated_silu_bwd_y | q4_k | — | — | — | — | — | — | — | — | — | 1.321 | — | f32 (blocks through lane storage) | 1.05e+0 |
| cuda | map:gated_gelu_bwd_x | q4_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 1.07e+0 |
| cuda | map:gated_gelu_bwd_y | q4_k | — | — | — | — | — | — | — | — | — | 1.321 | — | f32 (blocks through lane storage) | 1.07e+0 |
| cuda | sum:whole | q4_k | — | — | — | — | — | — | — | — | — | 0.065 | — | f32 | 8.3e-5 |
| cuda | sum:rows | q4_k | — | — | — | — | — | — | — | — | — | 0.058 | — | f32 | 9.54e-6 |
| cuda | sum:cols | q4_k | — | — | — | — | — | — | — | — | — | 0.068 | — | f32 | 3.62e-5 |
| cuda | sumsq:rows | q4_k | — | — | — | — | — | — | — | — | — | 0.057 | — | f32 | 1.88e-4 |
| cuda | max:whole | q4_k | — | — | — | — | — | — | — | — | — | 0.064 | — | f32 | 0e+0 |
| cuda | max:rows | q4_k | — | — | — | — | — | — | — | — | — | 0.057 | — | f32 | 0e+0 |
| cuda | max:cols | q4_k | — | — | — | — | — | — | — | — | — | 0.068 | — | f32 | 0e+0 |
| cuda | min:rows | q4_k | — | — | — | — | — | — | — | — | — | 0.057 | — | f32 | 0e+0 |
| cuda | argmax:rows | q4_k | — | — | — | — | — | — | — | — | — | 0.061 | — | s32 index | 0e+0 |
| cuda | argmax:cols | q4_k | — | — | — | — | — | — | — | — | — | 0.814 | — | s32 index | 0e+0 |
| cuda | maprow:add | q4_k | — | — | — | — | — | — | — | — | — | 0.990 | — | f32 (blocks through lane storage) | 1.32e-1 |
| cuda | maprow:mul | q4_k | — | — | — | — | — | — | — | — | — | 0.992 | — | f32 (blocks through lane storage) | 6.65e-2 |
| cuda | transpose | q4_k | — | — | — | — | — | — | — | — | — | 0.969 | — | f32 (blocks through lane storage) | 6.73e-2 |
| cuda | gemv | q4_k | — | — | — | — | — | — | — | — | — | 0.848 | — | f32 (blocks through lane storage) | 6.29e+0 |
| cuda | gemv_t | q4_k | — | — | — | — | — | — | — | — | — | 0.859 | — | f32 (blocks through lane storage) | 6.57e+0 |
| cuda | row:log_softmax | q4_k | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.14e-1 |
| cuda | row:softmax_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 2.49e+0 |
| cuda | row:log_softmax_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.194 | — | f32 (blocks through lane storage) | 4.44e+0 |
| cuda | row:rms_norm_affine | q4_k | — | — | — | — | — | — | — | — | — | 0.990 | — | f32 (blocks through lane storage) | 2.33e-1 |
| cuda | row:rms_norm_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 5.79e-2 |
| cuda | row:layer_norm | q4_k | — | — | — | — | — | — | — | — | — | 0.994 | — | f32 (blocks through lane storage) | 2.86e-1 |
| cuda | row:layer_norm_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 5.79e-2 |
| cuda | row:weight_norm | q4_k | — | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 4.3e-5 |
| cuda | row:rope | q4_k | — | — | — | — | — | — | — | — | — | 0.987 | — | f32 (blocks through lane storage) | 3.73e-1 |
| cuda | row:scale_mask_softmax | q4_k | — | — | — | — | — | — | — | — | — | 1.204 | — | f32 (blocks through lane storage) | 1.75e-2 |
| cuda | row:add_rms_norm | q4_k | — | — | — | — | — | — | — | — | — | 1.980 | — | f32 (blocks through lane storage) | 3.11e-1 |
| cuda | row:add_layer_norm | q4_k | — | — | — | — | — | — | — | — | — | 1.984 | — | f32 (blocks through lane storage) | 3.56e-1 |
| cuda | row:cross_entropy_bwd | q4_k | — | — | — | — | — | — | — | — | — | 0.990 | — | f32 (blocks through lane storage) | 2.52e-3 |
| cuda | norm_params_bwd:rms | q4_k | — | — | — | — | — | — | — | — | — | 0.142 | — | f32 | NaN |
| cuda | norm_params_bwd:layer | q4_k | — | — | — | — | — | — | — | — | — | 0.146 | — | f32 | NaN |
| cuda | layout:strided | q4_k | — | — | — | — | — | — | — | — | — | 1.210 | — | f32 (blocks through lane storage) | 6.72e-2 |
| cuda | layout:one_hot | q4_k | — | — | — | — | — | — | — | — | — | 0.807 | — | f32 (blocks through lane storage) | 3.57e-4 |
| cuda | layout:gather_rows | q4_k | — | — | — | — | — | — | — | — | — | 1.000 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | layout:gather_cols | q4_k | — | — | — | — | — | — | — | — | — | 1.061 | — | f32 (blocks through lane storage) | 6.72e-2 |
| cuda | layout:scatter_add_rows | q4_k | — | — | — | — | — | — | — | — | — | 1.092 | — | f32 (blocks through lane storage) | 2.77e-1 |
| cuda | layout:im2col | q4_k | — | — | — | — | — | — | — | — | — | 0.553 | — | f32 (blocks through lane storage) | 6.67e-2 |
| cuda | layout:col2im | q4_k | — | — | — | — | — | — | — | — | — | 0.189 | — | f32 (blocks through lane storage) | 4.18e-1 |
| cuda | layout:interpolate_nearest | q4_k | — | — | — | — | — | — | — | — | — | 0.843 | — | f32 (blocks through lane storage) | 6.2e-2 |
| cuda | layout:interpolate_bilinear | q4_k | — | — | — | — | — | — | — | — | — | 0.861 | — | f32 (blocks through lane storage) | 6.66e-2 |
| cuda | layout:shift | q4_k | — | — | — | — | — | — | — | — | — | 0.995 | — | f32 (blocks through lane storage) | 6.72e-2 |
| cuda | layout:pad2d | q4_k | — | — | — | — | — | — | — | — | — | 0.996 | — | f32 (blocks through lane storage) | 6.7e-2 |
| cuda | layout:maxpool | q4_k | — | — | — | — | — | — | — | — | — | 0.131 | — | f32 (blocks through lane storage) | 6.36e-2 |
| cuda | layout:avgpool | q4_k | — | — | — | — | — | — | — | — | — | 0.132 | — | f32 (blocks through lane storage) | 5.84e-2 |
| cuda | layout:maxpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 0.286 | — | f32 (blocks through lane storage) | 6.63e-2 |
| cuda | layout:avgpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 0.272 | — | f32 (blocks through lane storage) | 1.55e-2 |
| cuda | layout:adaptive_maxpool | q4_k | — | — | — | — | — | — | — | — | — | 0.134 | — | f32 (blocks through lane storage) | 2.97e-2 |
| cuda | layout:adaptive_avgpool | q4_k | — | — | — | — | — | — | — | — | — | 0.134 | — | f32 (blocks through lane storage) | 3.06e-3 |
| cuda | layout:depthwise_conv | q4_k | — | — | — | — | — | — | — | — | — | 0.249 | — | f32 (blocks through lane storage) | 2.95e-1 |
| cuda | layout:adaptive_maxpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 7.180 | — | f32 (blocks through lane storage) | 5.43e-2 |
| cuda | layout:adaptive_avgpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 0.320 | — | f32 (blocks through lane storage) | 6.65e-5 |
| cuda | matmul_nt | q4_k | — | — | — | — | — | — | — | — | — | 1.003 | — | f32 (blocks through lane storage) | 4.49e+0 |
| cuda | matmul_tn | q4_k | — | — | — | — | — | — | — | — | — | 1.011 | — | f32 (blocks through lane storage) | 3.89e+0 |
| cuda | matmul_tt | q4_k | — | — | — | — | — | — | — | — | — | 1.057 | — | f32 (blocks through lane storage) | 4.31e+0 |
| cuda | matmul_batched | q4_k | — | — | — | — | — | — | — | — | — | 1.147 | — | f32 (blocks through lane storage) | 1.92e+0 |
| cuda | matmul_bwd | q4_k | — | — | — | — | — | — | — | — | — | 2.005 | — | f32 (blocks through lane storage) | 3.92e+0 |
| cuda | attention | q4_k | — | — | — | — | — | — | — | — | — | 3.977 | — | f32 (blocks through lane storage) | 3.47e-3 |
| cuda | attention_causal | q4_k | — | — | — | — | — | — | — | — | — | 2.327 | — | f32 (blocks through lane storage) | 5.93e-2 |
| cuda | attention_fused | q4_k | — | — | — | — | — | — | — | — | — | 3.961 | — | f32 (blocks through lane storage) | 3.47e-3 |
| cuda | attention_fused_causal | q4_k | — | — | — | — | — | — | — | — | — | 2.327 | — | f32 (blocks through lane storage) | 5.93e-2 |
| cuda | attention_decode | q4_k | — | — | — | — | — | — | — | — | — | 0.898 | — | f32 (blocks through lane storage) | 2.05e-3 |
| cuda | attention_bwd | q4_k | — | — | — | — | — | — | — | — | — | 4.314 | — | f32 | 2.51e-2 |
| cuda | optimizer:sgd_momentum | q4_k | — | — | — | — | — | — | — | — | — | 1.547 | — | f32 (blocks through lane storage); + restore of p, m, v | 5.87e-3 |
| cuda | optimizer:adam | q4_k | — | — | — | — | — | — | — | — | — | 1.900 | — | f32 (blocks through lane storage); + restore of p, m, v | 2.26e-2 |
| cuda | optimizer:adamw | q4_k | — | — | — | — | — | — | — | — | — | 1.901 | — | f32 (blocks through lane storage); + restore of p, m, v | 2.26e-2 |
| cuda | conv | q4_k | — | — | — | — | — | — | — | — | — | 0.246 | — | f32 | 2.08e+0 |
| cuda | conv_bwd | q4_k | — | — | — | — | — | — | — | — | — | 2.382 | — | f32 | 2.95e+0 |
| cuda | lstm_step | q4_k | — | — | — | — | — | — | — | — | — | 1.508 | — | f32 | 2.95e-2 |
| cuda | gru_step | q4_k | — | — | — | — | — | — | — | — | — | 1.049 | — | f32 | 5.6e-2 |
| cuda | lstm_step_bwd | q4_k | — | — | — | — | — | — | — | — | — | 3.742 | — | f32 | 3.21e-2 |
| cuda | gru_step_bwd | q4_k | — | — | — | — | — | — | — | — | — | 2.607 | — | f32 | 3.05e-2 |
| cuda | gru_seq | q4_k | — | — | — | — | — | — | — | — | — | 2.671 | — | f32 (blocks through lane storage) | 5.58e-2 |
| cuda | gru_seq_bwd | q4_k | — | — | — | — | — | — | — | — | — | 12.964 | — | f32 (blocks through lane storage) | 3.51e-2 |
| cuda | topk_rows | q4_k | — | — | — | — | — | — | — | — | — | 5.539 | — | f32 (blocks through lane storage) | 5.87e-2 |
| cuda | pick_rows | q4_k | — | — | — | — | — | — | — | — | — | 1.016 | — | f32 (blocks through lane storage) | 2.64e-1 |
| cuda | sample_rows | q4_k | — | — | — | — | — | — | — | — | — | 2.657 | — | f32 | 2e+0 |
| cuda | cross_entropy | q4_k | — | — | — | — | — | — | — | — | — | 24.634 | — | f32 (blocks through lane storage) (log_softmax, pick, neg) | 9.02e-1 |
| cuda | ctc | q4_k | — | — | — | — | — | — | — | — | — | 0.560 | — | f32 (blocks through lane storage) | 5.55e+1 |
| cuda | ctc_bwd | q4_k | — | — | — | — | — | — | — | — | — | 1.455 | — | f32 (blocks through lane storage) | 3.06e-2 |
| cuda | softmax | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 7.31e-5 |
| cuda | rms_norm | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 0.988 | — | f32 (blocks through lane storage) | 1.37e-3 |
| cuda | add | q6_k | 16777216 | — | — | — | — | — | — | — | — | 1.402 | — | f32 | 3.22e-2 |
| cuda | matmul | q6_k | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.957 | — | f32 | 1.25e+0 |
| cuda | map:copy | q6_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:fill | q6_k | — | — | — | — | — | — | — | — | — | 0.960 | — | f32 (blocks through lane storage) | 6.2e-5 |
| cuda | map:neg | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:abs | q6_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | map:sign | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 3.34e-4 |
| cuda | map:square | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 2.54e-1 |
| cuda | map:relu | q6_k | — | — | — | — | — | — | — | — | — | 1.199 | — | f32 (blocks through lane storage) | 6.19e-2 |
| cuda | map:add_scalar | q6_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 9.58e-2 |
| cuda | map:scale | q6_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 3.75e-3 |
| cuda | map:affine | q6_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 1.44e-1 |
| cuda | map:clamp | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 3.22e-2 |
| cuda | map:add | q6_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 1.29e-1 |
| cuda | map:sub | q6_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 1.29e-1 |
| cuda | map:mul | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 2.53e-1 |
| cuda | map:div | q6_k | — | — | — | — | — | — | — | — | — | 1.405 | — | f32 (blocks through lane storage) | 6.78e-2 |
| cuda | map:max | q6_k | — | — | — | — | — | — | — | — | — | 1.405 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:min | q6_k | — | — | — | — | — | — | — | — | — | 1.405 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:axpy | q6_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 1.92e-1 |
| cuda | map:muladd | q6_k | — | — | — | — | — | — | — | — | — | 1.613 | — | f32 (blocks through lane storage) | 3.22e-1 |
| cuda | map:where | q6_k | — | — | — | — | — | — | — | — | — | 1.541 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:square_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 5.07e-1 |
| cuda | map:abs_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 6.11e-2 |
| cuda | map:relu_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.408 | — | f32 (blocks through lane storage) | 6.19e-2 |
| cuda | map:clamp_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.408 | — | f32 (blocks through lane storage) | 6.19e-2 |
| cuda | map:where_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 6.18e-2 |
| cuda | map:sqrt | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 3.22e-2 |
| cuda | map:inv_sqrt | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 2.41e-2 |
| cuda | map:exp | q6_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 8.79e-1 |
| cuda | map:expm1 | q6_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 1.04e-2 |
| cuda | map:log | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 2.23e-2 |
| cuda | map:sin | q6_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | map:cos | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | map:tanh | q6_k | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | map:sigmoid | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.58e-2 |
| cuda | map:erf | q6_k | — | — | — | — | — | — | — | — | — | 2.209 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | map:gelu | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 6.43e-2 |
| cuda | map:gelu_erf | q6_k | — | — | — | — | — | — | — | — | — | 3.248 | — | f32 (blocks through lane storage) | 6.43e-2 |
| cuda | map:silu | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 6.21e-2 |
| cuda | map:leaky_relu | q6_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:pow | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 5.04e-1 |
| cuda | map:dropout | q6_k | — | — | — | — | — | — | — | — | — | 1.407 | — | f32 (blocks through lane storage) | 7.66e-2 |
| cuda | map:gated_silu | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 2.53e-1 |
| cuda | map:gated_gelu | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 2.58e-1 |
| cuda | map:sqrt_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 4.78e-2 |
| cuda | map:inv_sqrt_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 1.06e-1 |
| cuda | map:exp_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 3.51e+0 |
| cuda | map:log_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 1.43e-1 |
| cuda | map:sin_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.405 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:cos_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:tanh_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:sigmoid_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | map:silu_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 7.09e-2 |
| cuda | map:gelu_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 7.28e-2 |
| cuda | map:gelu_erf_bwd | q6_k | — | — | — | — | — | — | — | — | — | 3.483 | — | f32 (blocks through lane storage) | 7.28e-2 |
| cuda | map:leaky_relu_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | map:pow_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 1.29e+0 |
| cuda | map:gated_silu_bwd_x | q6_k | — | — | — | — | — | — | — | — | — | 1.615 | — | f32 (blocks through lane storage) | 2.71e-1 |
| cuda | map:gated_silu_bwd_y | q6_k | — | — | — | — | — | — | — | — | — | 1.530 | — | f32 (blocks through lane storage) | 2.52e-1 |
| cuda | map:gated_gelu_bwd_x | q6_k | — | — | — | — | — | — | — | — | — | 1.616 | — | f32 (blocks through lane storage) | 2.58e-1 |
| cuda | map:gated_gelu_bwd_y | q6_k | — | — | — | — | — | — | — | — | — | 1.531 | — | f32 (blocks through lane storage) | 2.57e-1 |
| cuda | sum:whole | q6_k | — | — | — | — | — | — | — | — | — | 0.060 | — | f32 | 9.95e-5 |
| cuda | sum:rows | q6_k | — | — | — | — | — | — | — | — | — | 0.054 | — | f32 | 1.31e-5 |
| cuda | sum:cols | q6_k | — | — | — | — | — | — | — | — | — | 0.066 | — | f32 | 3.08e-5 |
| cuda | sumsq:rows | q6_k | — | — | — | — | — | — | — | — | — | 0.054 | — | f32 | 1.83e-4 |
| cuda | max:whole | q6_k | — | — | — | — | — | — | — | — | — | 0.060 | — | f32 | 0e+0 |
| cuda | max:rows | q6_k | — | — | — | — | — | — | — | — | — | 0.054 | — | f32 | 0e+0 |
| cuda | max:cols | q6_k | — | — | — | — | — | — | — | — | — | 0.066 | — | f32 | 0e+0 |
| cuda | min:rows | q6_k | — | — | — | — | — | — | — | — | — | 0.054 | — | f32 | 0e+0 |
| cuda | argmax:rows | q6_k | — | — | — | — | — | — | — | — | — | 0.058 | — | s32 index | 0e+0 |
| cuda | argmax:cols | q6_k | — | — | — | — | — | — | — | — | — | 0.818 | — | s32 index | 0e+0 |
| cuda | maprow:add | q6_k | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 (blocks through lane storage) | 3.22e-2 |
| cuda | maprow:mul | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.58e-2 |
| cuda | transpose | q6_k | — | — | — | — | — | — | — | — | — | 1.181 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | gemv | q6_k | — | — | — | — | — | — | — | — | — | 0.870 | — | f32 (blocks through lane storage) | 1.89e+0 |
| cuda | gemv_t | q6_k | — | — | — | — | — | — | — | — | — | 0.883 | — | f32 (blocks through lane storage) | 2.16e+0 |
| cuda | row:log_softmax | q6_k | — | — | — | — | — | — | — | — | — | 1.191 | — | f32 (blocks through lane storage) | 2.31e-1 |
| cuda | row:softmax_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.406 | — | f32 (blocks through lane storage) | 1.27e+0 |
| cuda | row:log_softmax_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 2.05e+0 |
| cuda | row:rms_norm_affine | q6_k | — | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 5.53e-2 |
| cuda | row:rms_norm_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.404 | — | f32 (blocks through lane storage) | 1.43e-2 |
| cuda | row:layer_norm | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 7.14e-2 |
| cuda | row:layer_norm_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.407 | — | f32 (blocks through lane storage) | 1.43e-2 |
| cuda | row:weight_norm | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.17e-4 |
| cuda | row:rope | q6_k | — | — | — | — | — | — | — | — | — | 1.192 | — | f32 (blocks through lane storage) | 9.1e-2 |
| cuda | row:scale_mask_softmax | q6_k | — | — | — | — | — | — | — | — | — | 1.410 | — | f32 (blocks through lane storage) | 5.7e-3 |
| cuda | row:add_rms_norm | q6_k | — | — | — | — | — | — | — | — | — | 2.396 | — | f32 (blocks through lane storage) | 7.84e-2 |
| cuda | row:add_layer_norm | q6_k | — | — | — | — | — | — | — | — | — | 2.399 | — | f32 (blocks through lane storage) | 9.2e-2 |
| cuda | row:cross_entropy_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 2.1e-3 |
| cuda | norm_params_bwd:rms | q6_k | — | — | — | — | — | — | — | — | — | 0.123 | — | f32 | NaN |
| cuda | norm_params_bwd:layer | q6_k | — | — | — | — | — | — | — | — | — | 0.126 | — | f32 | NaN |
| cuda | layout:strided | q6_k | — | — | — | — | — | — | — | — | — | 1.427 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:one_hot | q6_k | — | — | — | — | — | — | — | — | — | 1.014 | — | f32 (blocks through lane storage) | 3.34e-4 |
| cuda | layout:gather_rows | q6_k | — | — | — | — | — | — | — | — | — | 1.207 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | layout:gather_cols | q6_k | — | — | — | — | — | — | — | — | — | 1.269 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:scatter_add_rows | q6_k | — | — | — | — | — | — | — | — | — | 1.301 | — | f32 (blocks through lane storage) | 6.79e-2 |
| cuda | layout:im2col | q6_k | — | — | — | — | — | — | — | — | — | 0.674 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:col2im | q6_k | — | — | — | — | — | — | — | — | — | 0.262 | — | f32 (blocks through lane storage) | 1.19e-1 |
| cuda | layout:interpolate_nearest | q6_k | — | — | — | — | — | — | — | — | — | 1.044 | — | f32 (blocks through lane storage) | 1.52e-2 |
| cuda | layout:interpolate_bilinear | q6_k | — | — | — | — | — | — | — | — | — | 1.064 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:shift | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:pad2d | q6_k | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:maxpool | q6_k | — | — | — | — | — | — | — | — | — | 0.195 | — | f32 (blocks through lane storage) | 1.61e-2 |
| cuda | layout:avgpool | q6_k | — | — | — | — | — | — | — | — | — | 0.195 | — | f32 (blocks through lane storage) | 1.56e-2 |
| cuda | layout:maxpool_bwd | q6_k | — | — | — | — | — | — | — | — | — | 0.368 | — | f32 (blocks through lane storage) | 1.55e-2 |
| cuda | layout:avgpool_bwd | q6_k | — | — | — | — | — | — | — | — | — | 0.358 | — | f32 (blocks through lane storage) | 3.81e-3 |
| cuda | layout:adaptive_maxpool | q6_k | — | — | — | — | — | — | — | — | — | 0.152 | — | f32 (blocks through lane storage) | 1.07e-2 |
| cuda | layout:adaptive_avgpool | q6_k | — | — | — | — | — | — | — | — | — | 0.152 | — | f32 (blocks through lane storage) | 9.08e-4 |
| cuda | layout:depthwise_conv | q6_k | — | — | — | — | — | — | — | — | — | 0.334 | — | f32 (blocks through lane storage) | 7.98e-2 |
| cuda | layout:adaptive_maxpool_bwd | q6_k | — | — | — | — | — | — | — | — | — | 7.262 | — | f32 (blocks through lane storage) | 1.33e-2 |
| cuda | layout:adaptive_avgpool_bwd | q6_k | — | — | — | — | — | — | — | — | — | 0.404 | — | f32 (blocks through lane storage) | 1.17e-4 |
| cuda | matmul_nt | q6_k | — | — | — | — | — | — | — | — | — | 1.093 | — | f32 (blocks through lane storage) | 1.24e+0 |
| cuda | matmul_tn | q6_k | — | — | — | — | — | — | — | — | — | 1.098 | — | f32 (blocks through lane storage) | 1.22e+0 |
| cuda | matmul_tt | q6_k | — | — | — | — | — | — | — | — | — | 1.147 | — | f32 (blocks through lane storage) | 1.22e+0 |
| cuda | matmul_batched | q6_k | — | — | — | — | — | — | — | — | — | 1.219 | — | f32 (blocks through lane storage) | 6.5e-1 |
| cuda | matmul_bwd | q6_k | — | — | — | — | — | — | — | — | — | 2.185 | — | f32 (blocks through lane storage) | 1.29e+0 |
| cuda | attention | q6_k | — | — | — | — | — | — | — | — | — | 4.058 | — | f32 (blocks through lane storage) | 1.02e-3 |
| cuda | attention_causal | q6_k | — | — | — | — | — | — | — | — | — | 2.413 | — | f32 (blocks through lane storage) | 1.51e-2 |
| cuda | attention_fused | q6_k | — | — | — | — | — | — | — | — | — | 4.062 | — | f32 (blocks through lane storage) | 1.02e-3 |
| cuda | attention_fused_causal | q6_k | — | — | — | — | — | — | — | — | — | 2.412 | — | f32 (blocks through lane storage) | 1.51e-2 |
| cuda | attention_decode | q6_k | — | — | — | — | — | — | — | — | — | 0.917 | — | f32 (blocks through lane storage) | 5.47e-4 |
| cuda | attention_bwd | q6_k | — | — | — | — | — | — | — | — | — | 4.509 | — | f32 | 6.59e-3 |
| cuda | optimizer:sgd_momentum | q6_k | — | — | — | — | — | — | — | — | — | 1.766 | — | f32 (blocks through lane storage); + restore of p, m, v | 3.34e-3 |
| cuda | optimizer:adam | q6_k | — | — | — | — | — | — | — | — | — | 2.119 | — | f32 (blocks through lane storage); + restore of p, m, v | 1.44e-2 |
| cuda | optimizer:adamw | q6_k | — | — | — | — | — | — | — | — | — | 2.115 | — | f32 (blocks through lane storage); + restore of p, m, v | 1.44e-2 |
| cuda | conv | q6_k | — | — | — | — | — | — | — | — | — | 0.311 | — | f32 | 6.79e-1 |
| cuda | conv_bwd | q6_k | — | — | — | — | — | — | — | — | — | 2.473 | — | f32 | 8.92e-1 |
| cuda | lstm_step | q6_k | — | — | — | — | — | — | — | — | — | 1.648 | — | f32 | 8.59e-3 |
| cuda | gru_step | q6_k | — | — | — | — | — | — | — | — | — | 1.025 | — | f32 | 1.42e-2 |
| cuda | lstm_step_bwd | q6_k | — | — | — | — | — | — | — | — | — | 3.741 | — | f32 | 9.36e-3 |
| cuda | gru_step_bwd | q6_k | — | — | — | — | — | — | — | — | — | 2.888 | — | f32 | 9.11e-3 |
| cuda | gru_seq | q6_k | — | — | — | — | — | — | — | — | — | 2.743 | — | f32 (blocks through lane storage) | 1.39e-2 |
| cuda | gru_seq_bwd | q6_k | — | — | — | — | — | — | — | — | — | 13.105 | — | f32 (blocks through lane storage) | 1.05e-2 |
| cuda | topk_rows | q6_k | — | — | — | — | — | — | — | — | — | 7.946 | — | f32 (blocks through lane storage) | 0e+0 |
| cuda | pick_rows | q6_k | — | — | — | — | — | — | — | — | — | 1.053 | — | f32 (blocks through lane storage) | 6.45e-2 |
| cuda | sample_rows | q6_k | — | — | — | — | — | — | — | — | — | 8.404 | — | f32 | 1e+0 |
| cuda | cross_entropy | q6_k | — | — | — | — | — | — | — | — | — | 34.853 | — | f32 (blocks through lane storage) (log_softmax, pick, neg) | 5.08e-1 |
| cuda | ctc | q6_k | — | — | — | — | — | — | — | — | — | 0.565 | — | f32 (blocks through lane storage) | 2.76e+1 |
| cuda | ctc_bwd | q6_k | — | — | — | — | — | — | — | — | — | 1.567 | — | f32 (blocks through lane storage) | 1.54e-2 |
| cuda | softmax | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 1.192 | — | f32 (blocks through lane storage) | 1.17e-4 |
| cuda | rms_norm | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 1.193 | — | f32 (blocks through lane storage) | 4.69e-4 |
| cuda | add | bool | 16777216 | — | — | — | — | — | — | — | — | 0.017 | — | u8 | 1e+0 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.125 | — | u8mma | 1.27e+3 |
| cuda | map:copy | bool | — | — | — | — | — | — | — | — | — | 0.013 | — | u32 | 0e+0 |
| cuda | map:fill | bool | — | — | — | — | — | — | — | — | — | 0.010 | — | u32 | 0e+0 |
| cuda | map:neg | bool | — | — | — | — | — | — | — | — | — | 0.015 | — | u32 | 0e+0 |
| cuda | map:abs | bool | — | — | — | — | — | — | — | — | — | 0.013 | — | u32 | 0e+0 |
| cuda | map:sign | bool | — | — | — | — | — | — | — | — | — | 0.016 | — | u32 | 0e+0 |
| cuda | map:square | bool | — | — | — | — | — | — | — | — | — | 0.014 | — | u32 | 0e+0 |
| cuda | map:relu | bool | — | — | — | — | — | — | — | — | — | 0.015 | — | u32 | 0e+0 |
| cuda | map:add_scalar | bool | — | — | — | — | — | — | — | — | — | 0.015 | — | u32 | 0e+0 |
| cuda | map:scale | bool | — | — | — | — | — | — | — | — | — | 0.014 | — | u32 | 0e+0 |
| cuda | map:affine | bool | — | — | — | — | — | — | — | — | — | 0.014 | — | u32 | 0e+0 |
| cuda | map:clamp | bool | — | — | — | — | — | — | — | — | — | 0.015 | — | u32 | 0e+0 |
| cuda | map:add | bool | — | — | — | — | — | — | — | — | — | 0.017 | — | u32 | 0e+0 |
| cuda | map:sub | bool | — | — | — | — | — | — | — | — | — | 0.017 | — | u32 | 0e+0 |
| cuda | map:mul | bool | — | — | — | — | — | — | — | — | — | 0.018 | — | u32 | 0e+0 |
| cuda | map:div | bool | — | — | — | — | — | — | — | — | — | — | — | — | — |
| cuda | map:max | bool | — | — | — | — | — | — | — | — | — | 0.019 | — | u32 | 0e+0 |
| cuda | map:min | bool | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | map:axpy | bool | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | map:muladd | bool | — | — | — | — | — | — | — | — | — | 0.024 | — | u32 | 0e+0 |
| cuda | map:where | bool | — | — | — | — | — | — | — | — | — | 0.024 | — | u32 | 0e+0 |
| cuda | map:square_bwd | bool | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | map:abs_bwd | bool | — | — | — | — | — | — | — | — | — | 0.024 | — | u32 | 0e+0 |
| cuda | map:relu_bwd | bool | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | map:clamp_bwd | bool | — | — | — | — | — | — | — | — | — | 0.023 | — | u32 | 0e+0 |
| cuda | map:where_bwd | bool | — | — | — | — | — | — | — | — | — | 0.022 | — | u32 | 0e+0 |
| cuda | sum:whole | bool | — | — | — | — | — | — | — | — | — | 0.037 | — | f32 | 0e+0 |
| cuda | sum:rows | bool | — | — | — | — | — | — | — | — | — | 0.032 | — | f32 | 0e+0 |
| cuda | sum:cols | bool | — | — | — | — | — | — | — | — | — | 0.038 | — | f32 | 0e+0 |
| cuda | sumsq:rows | bool | — | — | — | — | — | — | — | — | — | 0.032 | — | f32 | 0e+0 |
| cuda | max:whole | bool | — | — | — | — | — | — | — | — | — | 0.036 | — | u32 | 0e+0 |
| cuda | max:rows | bool | — | — | — | — | — | — | — | — | — | 0.031 | — | u32 | 0e+0 |
| cuda | max:cols | bool | — | — | — | — | — | — | — | — | — | 0.036 | — | u32 | 0e+0 |
| cuda | min:rows | bool | — | — | — | — | — | — | — | — | — | 0.032 | — | u32 | 0e+0 |
| cuda | argmax:rows | bool | — | — | — | — | — | — | — | — | — | 0.034 | — | s32 index | 0e+0 |
| cuda | argmax:cols | bool | — | — | — | — | — | — | — | — | — | 0.199 | — | s32 index | 0e+0 |
| cuda | maprow:add | bool | — | — | — | — | — | — | — | — | — | 0.039 | — | u32 | 0e+0 |
| cuda | maprow:mul | bool | — | — | — | — | — | — | — | — | — | 0.037 | — | u32 | 0e+0 |
| cuda | transpose | bool | — | — | — | — | — | — | — | — | — | 0.044 | — | u32 | 0e+0 |
| cuda | gemv | bool | — | — | — | — | — | — | — | — | — | 0.138 | — | f32 | 4.78e+3 |
| cuda | gemv_t | bool | — | — | — | — | — | — | — | — | — | 0.155 | — | f32 | 4.8e+3 |
| cuda | layout:strided | bool | — | — | — | — | — | — | — | — | — | 0.286 | — | u32 | 0e+0 |
| cuda | layout:one_hot | bool | — | — | — | — | — | — | — | — | — | 0.036 | — | u32 | 0e+0 |
| cuda | layout:gather_rows | bool | — | — | — | — | — | — | — | — | — | 0.040 | — | u32 | 0e+0 |
| cuda | layout:gather_cols | bool | — | — | — | — | — | — | — | — | — | 0.113 | — | u32 | 0e+0 |
| cuda | layout:scatter_add_rows | bool | — | — | — | — | — | — | — | — | — | 0.158 | — | u32 | 4e+0 |
| cuda | layout:im2col | bool | — | — | — | — | — | — | — | — | — | 0.054 | — | u32 | 0e+0 |
| cuda | layout:col2im | bool | — | — | — | — | — | — | — | — | — | 0.075 | — | u32 | 8e+0 |
| cuda | layout:interpolate_nearest | bool | — | — | — | — | — | — | — | — | — | 0.074 | — | u32 | 0e+0 |
| cuda | layout:shift | bool | — | — | — | — | — | — | — | — | — | 0.042 | — | u32 | 0e+0 |
| cuda | layout:pad2d | bool | — | — | — | — | — | — | — | — | — | 0.041 | — | u32 | 0e+0 |
| cuda | layout:maxpool | bool | — | — | — | — | — | — | — | — | — | 0.033 | — | u32 | 0e+0 |
| cuda | layout:maxpool_bwd | bool | — | — | — | — | — | — | — | — | — | 0.197 | — | u32 | 0e+0 |
| cuda | layout:adaptive_maxpool | bool | — | — | — | — | — | — | — | — | — | 0.930 | — | u32 | 0e+0 |
| cuda | layout:depthwise_conv | bool | — | — | — | — | — | — | — | — | — | 0.130 | — | u32 | 8e+0 |
| cuda | layout:adaptive_maxpool_bwd | bool | — | — | — | — | — | — | — | — | — | 54.973 | — | u32 | 0e+0 |
| cuda | matmul_nt | bool | — | — | — | — | — | — | — | — | — | 0.121 | — | u8mma | 1.27e+3 |
| cuda | matmul_tn | bool | — | — | — | — | — | — | — | — | — | 0.121 | — | u8mma | 1.26e+3 |
| cuda | matmul_tt | bool | — | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.26e+3 |
| cuda | matmul_batched | bool | — | — | — | — | — | — | — | — | — | 0.537 | — | u8mma | 3.48e+2 |
| cuda | matmul_bwd | bool | — | — | — | — | — | — | — | — | — | 0.232 | — | u8mma | 1.25e+3 |
| cuda | conv | bool | — | — | — | — | — | — | — | — | — | 0.114 | — | u8mma | 3.73e+2 |
| cuda | topk_rows | bool | — | — | — | — | — | — | — | — | — | 3.061 | — | u32 | 0e+0 |
| cuda | pick_rows | bool | — | — | — | — | — | — | — | — | — | 0.016 | — | u32 | 0e+0 |
| cpu | add | float64 | 16777216 | 6.307 | — | typed tile | 0e+0 | 6.883 | 0.92 | Vector<Double> | 0e+0 | 6.724 | 0.94 | f64 | 0e+0 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.783 | — | to f32 and back | 7.96e-5 | 15.855 | 0.36 | Vector<Double> | 1.14e-13 | 3.364 | 1.72 | f64 | 1.14e-13 |
| cpu | map:copy | float64 | — | — | — | — | — | — | — | — | — | 4.750 | — | f64 | 0e+0 |
| cpu | map:fill | float64 | — | — | — | — | — | — | — | — | — | 2.483 | — | f64 | 0e+0 |
| cpu | map:neg | float64 | — | — | — | — | — | — | — | — | — | 4.677 | — | f64 | 0e+0 |
| cpu | map:abs | float64 | — | — | — | — | — | — | — | — | — | 4.744 | — | f64 | 0e+0 |
| cpu | map:sign | float64 | — | — | — | — | — | — | — | — | — | 4.692 | — | f64 | 0e+0 |
| cpu | map:square | float64 | — | — | — | — | — | — | — | — | — | 4.949 | — | f64 | 0e+0 |
| cpu | map:relu | float64 | — | — | — | — | — | — | — | — | — | 4.625 | — | f64 | 0e+0 |
| cpu | map:add_scalar | float64 | — | — | — | — | — | — | — | — | — | 4.886 | — | f64 | 0e+0 |
| cpu | map:scale | float64 | — | — | — | — | — | — | — | — | — | 4.714 | — | f64 | 0e+0 |
| cpu | map:affine | float64 | — | — | — | — | — | — | — | — | — | 4.704 | — | f64 | 0e+0 |
| cpu | map:clamp | float64 | — | — | — | — | — | — | — | — | — | 4.669 | — | f64 | 0e+0 |
| cpu | map:add | float64 | — | — | — | — | — | — | — | — | — | 6.692 | — | f64 | 0e+0 |
| cpu | map:sub | float64 | — | — | — | — | — | — | — | — | — | 6.820 | — | f64 | 0e+0 |
| cpu | map:mul | float64 | — | — | — | — | — | — | — | — | — | 6.680 | — | f64 | 0e+0 |
| cpu | map:div | float64 | — | — | — | — | — | — | — | — | — | 6.674 | — | f64 | 0e+0 |
| cpu | map:max | float64 | — | — | — | — | — | — | — | — | — | 6.617 | — | f64 | 0e+0 |
| cpu | map:min | float64 | — | — | — | — | — | — | — | — | — | 6.640 | — | f64 | 0e+0 |
| cpu | map:axpy | float64 | — | — | — | — | — | — | — | — | — | 6.710 | — | f64 | 0e+0 |
| cpu | map:muladd | float64 | — | — | — | — | — | — | — | — | — | 8.615 | — | f64 | 0e+0 |
| cpu | map:where | float64 | — | — | — | — | — | — | — | — | — | 8.531 | — | f64 | 0e+0 |
| cpu | map:square_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.615 | — | f64 | 0e+0 |
| cpu | map:abs_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.748 | — | f64 | 0e+0 |
| cpu | map:relu_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.751 | — | f64 | 0e+0 |
| cpu | map:clamp_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.913 | — | f64 | 0e+0 |
| cpu | map:where_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.899 | — | f64 | 0e+0 |
| cpu | map:sqrt | float64 | — | — | — | — | — | — | — | — | — | 4.834 | — | f64 | 0e+0 |
| cpu | map:inv_sqrt | float64 | — | — | — | — | — | — | — | — | — | 4.870 | — | f64 | 0e+0 |
| cpu | map:exp | float64 | — | — | — | — | — | — | — | — | — | 5.192 | — | f64 | 7.11e-15 |
| cpu | map:expm1 | float64 | — | — | — | — | — | — | — | — | — | 4.926 | — | f64 | 1.11e-16 |
| cpu | map:log | float64 | — | — | — | — | — | — | — | — | — | 5.037 | — | f64 | 2.22e-16 |
| cpu | map:sin | float64 | — | — | — | — | — | — | — | — | — | 4.903 | — | f64 | 2.22e-16 |
| cpu | map:cos | float64 | — | — | — | — | — | — | — | — | — | 4.956 | — | f64 | 1.11e-16 |
| cpu | map:tanh | float64 | — | — | — | — | — | — | — | — | — | 5.149 | — | f64 | 3.33e-16 |
| cpu | map:sigmoid | float64 | — | — | — | — | — | — | — | — | — | 4.908 | — | f64 | 2.22e-16 |
| cpu | map:erf | float64 | — | — | — | — | — | — | — | — | — | 71.471 | — | f64 | 6.66e-16 |
| cpu | map:gelu | float64 | — | — | — | — | — | — | — | — | — | 4.924 | — | f64 | 8.88e-16 |
| cpu | map:gelu_erf | float64 | — | — | — | — | — | — | — | — | — | 338.764 | — | f64 | 4.44e-16 |
| cpu | map:silu | float64 | — | — | — | — | — | — | — | — | — | 4.847 | — | f64 | 8.88e-16 |
| cpu | map:leaky_relu | float64 | — | — | — | — | — | — | — | — | — | 4.703 | — | f64 | 0e+0 |
| cpu | map:pow | float64 | — | — | — | — | — | — | — | — | — | 8.774 | — | f64 | 2.84e-14 |
| cpu | map:dropout | float64 | — | — | — | — | — | — | — | — | — | 6.882 | — | f64 | 0e+0 |
| cpu | map:gated_silu | float64 | — | — | — | — | — | — | — | — | — | 6.541 | — | f64 | 3.55e-15 |
| cpu | map:gated_gelu | float64 | — | — | — | — | — | — | — | — | — | 6.570 | — | f64 | 3.55e-15 |
| cpu | map:sqrt_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.665 | — | f64 | 0e+0 |
| cpu | map:inv_sqrt_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.877 | — | f64 | 1.78e-15 |
| cpu | map:exp_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.602 | — | f64 | 2.84e-14 |
| cpu | map:log_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.723 | — | f64 | 0e+0 |
| cpu | map:sin_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.985 | — | f64 | 4.44e-16 |
| cpu | map:cos_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.837 | — | f64 | 4.44e-16 |
| cpu | map:tanh_bwd | float64 | — | — | — | — | — | — | — | — | — | 8.707 | — | f64 | 3.11e-15 |
| cpu | map:sigmoid_bwd | float64 | — | — | — | — | — | — | — | — | — | 8.935 | — | f64 | 4.44e-16 |
| cpu | map:silu_bwd | float64 | — | — | — | — | — | — | — | — | — | 9.030 | — | f64 | 2.66e-15 |
| cpu | map:gelu_bwd | float64 | — | — | — | — | — | — | — | — | — | 9.731 | — | f64 | 1.78e-15 |
| cpu | map:gelu_erf_bwd | float64 | — | — | — | — | — | — | — | — | — | 382.496 | — | f64 | 1.78e-15 |
| cpu | map:leaky_relu_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.675 | — | f64 | 0e+0 |
| cpu | map:pow_bwd | float64 | — | — | — | — | — | — | — | — | — | 14.030 | — | f64 | 4.26e-14 |
| cpu | map:gated_silu_bwd_x | float64 | — | — | — | — | — | — | — | — | — | 9.446 | — | f64 | 7.11e-15 |
| cpu | map:gated_silu_bwd_y | float64 | — | — | — | — | — | — | — | — | — | 8.562 | — | f64 | 3.55e-15 |
| cpu | map:gated_gelu_bwd_x | float64 | — | — | — | — | — | — | — | — | — | 9.795 | — | f64 | 7.11e-15 |
| cpu | map:gated_gelu_bwd_y | float64 | — | — | — | — | — | — | — | — | — | 8.592 | — | f64 | 3.55e-15 |
| cpu | sum:whole | float64 | — | — | — | — | — | — | — | — | — | 1.236 | — | f64 | 0e+0 |
| cpu | sum:rows | float64 | — | — | — | — | — | — | — | — | — | 1.248 | — | f64 | 0e+0 |
| cpu | sum:cols | float64 | — | — | — | — | — | — | — | — | — | 2.453 | — | f64 | 0e+0 |
| cpu | sumsq:rows | float64 | — | — | — | — | — | — | — | — | — | 1.229 | — | f64 | 2.14e-11 |
| cpu | max:whole | float64 | — | — | — | — | — | — | — | — | — | 1.748 | — | f64 | 0e+0 |
| cpu | max:rows | float64 | — | — | — | — | — | — | — | — | — | 1.659 | — | f64 | 0e+0 |
| cpu | max:cols | float64 | — | — | — | — | — | — | — | — | — | 3.802 | — | f64 | 0e+0 |
| cpu | min:rows | float64 | — | — | — | — | — | — | — | — | — | 1.580 | — | f64 | 0e+0 |
| cpu | argmax:rows | float64 | — | — | — | — | — | — | — | — | — | 1.477 | — | s32 index | 0e+0 |
| cpu | argmax:cols | float64 | — | — | — | — | — | — | — | — | — | 2.325 | — | s32 index | 0e+0 |
| cpu | maprow:add | float64 | — | — | — | — | — | — | — | — | — | 4.694 | — | f64 | 0e+0 |
| cpu | maprow:mul | float64 | — | — | — | — | — | — | — | — | — | 4.726 | — | f64 | 0e+0 |
| cpu | transpose | float64 | — | — | — | — | — | — | — | — | — | 5.956 | — | f64 | 0e+0 |
| cpu | gemv | float64 | — | — | — | — | — | — | — | — | — | 6.313 | — | f64 | 6.11e-13 |
| cpu | gemv_t | float64 | — | — | — | — | — | — | — | — | — | 13.956 | — | f64 | 0e+0 |
| cpu | row:log_softmax | float64 | — | — | — | — | — | — | — | — | — | 5.139 | — | f64 | 7.11e-15 |
| cpu | row:softmax_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.542 | — | f64 | 3.27e-13 |
| cpu | row:log_softmax_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.775 | — | f64 | 2.84e-14 |
| cpu | row:rms_norm_affine | float64 | — | — | — | — | — | — | — | — | — | 4.732 | — | f64 | 2.62e-14 |
| cpu | row:rms_norm_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.648 | — | f64 | 6.88e-15 |
| cpu | row:layer_norm | float64 | — | — | — | — | — | — | — | — | — | 4.848 | — | f64 | 1.15e-14 |
| cpu | row:layer_norm_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.876 | — | f64 | 3.11e-15 |
| cpu | row:weight_norm | float64 | — | — | — | — | — | — | — | — | — | 4.908 | — | f64 | 3.75e-16 |
| cpu | row:rope | float64 | — | — | — | — | — | — | — | — | — | 8.369 | — | f64 | 2.51e-12 |
| cpu | row:scale_mask_softmax | float64 | — | — | — | — | — | — | — | — | — | 7.091 | — | f64 | 5.55e-17 |
| cpu | row:add_rms_norm | float64 | — | — | — | — | — | — | — | — | — | 10.696 | — | f64 | 2.58e-14 |
| cpu | row:add_layer_norm | float64 | — | — | — | — | — | — | — | — | — | 10.884 | — | f64 | 1.51e-14 |
| cpu | row:cross_entropy_bwd | float64 | — | — | — | — | — | — | — | — | — | 5.648 | — | f64 | 1.11e-16 |
| cpu | norm_params_bwd:rms | float64 | — | — | — | — | — | — | — | — | — | 6.803 | — | f64 | 9.24e-13 |
| cpu | norm_params_bwd:layer | float64 | — | — | — | — | — | — | — | — | — | 6.775 | — | f64 | 3.41e-13 |
| cpu | layout:strided | float64 | — | — | — | — | — | — | — | — | — | 21.177 | — | f64 | 0e+0 |
| cpu | layout:one_hot | float64 | — | — | — | — | — | — | — | — | — | 2.549 | — | f64 | 0e+0 |
| cpu | layout:gather_rows | float64 | — | — | — | — | — | — | — | — | — | 3.663 | — | f64 | 0e+0 |
| cpu | layout:gather_cols | float64 | — | — | — | — | — | — | — | — | — | 4.778 | — | f64 | 0e+0 |
| cpu | layout:scatter_add_rows | float64 | — | — | — | — | — | — | — | — | — | 4.994 | — | f64 | 0e+0 |
| cpu | layout:im2col | float64 | — | — | — | — | — | — | — | — | — | 1.577 | — | f64 | 0e+0 |
| cpu | layout:col2im | float64 | — | — | — | — | — | — | — | — | — | 2.081 | — | f64 | 0e+0 |
| cpu | layout:interpolate_nearest | float64 | — | — | — | — | — | — | — | — | — | 3.448 | — | f64 | 0e+0 |
| cpu | layout:interpolate_bilinear | float64 | — | — | — | — | — | — | — | — | — | 7.903 | — | f64 | 0e+0 |
| cpu | layout:shift | float64 | — | — | — | — | — | — | — | — | — | 3.956 | — | f64 | 0e+0 |
| cpu | layout:pad2d | float64 | — | — | — | — | — | — | — | — | — | 3.777 | — | f64 | 0e+0 |
| cpu | layout:maxpool | float64 | — | — | — | — | — | — | — | — | — | 1.644 | — | f64 | 0e+0 |
| cpu | layout:avgpool | float64 | — | — | — | — | — | — | — | — | — | 0.625 | — | f64 | 0e+0 |
| cpu | layout:maxpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 6.806 | — | f64 | 0e+0 |
| cpu | layout:avgpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 2.519 | — | f64 | 0e+0 |
| cpu | layout:adaptive_maxpool | float64 | — | — | — | — | — | — | — | — | — | 9.078 | — | f64 | 0e+0 |
| cpu | layout:adaptive_avgpool | float64 | — | — | — | — | — | — | — | — | — | 5.754 | — | f64 | 6.94e-18 |
| cpu | layout:depthwise_conv | float64 | — | — | — | — | — | — | — | — | — | 4.762 | — | f64 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 720.755 | — | f64 | 0e+0 |
| cpu | layout:adaptive_avgpool_bwd | float64 | — | — | — | — | — | — | — | — | — | 9.096 | — | f64 | 4.34e-19 |
| cpu | matmul_nt | float64 | — | — | — | — | — | — | — | — | — | 3.618 | — | f64 | 1.42e-13 |
| cpu | matmul_tn | float64 | — | — | — | — | — | — | — | — | — | 3.627 | — | f64 | 1.28e-13 |
| cpu | matmul_tt | float64 | — | — | — | — | — | — | — | — | — | 3.776 | — | f64 | 1.21e-13 |
| cpu | matmul_batched | float64 | — | — | — | — | — | — | — | — | — | 1.350 | — | f64 | 3.2e-14 |
| cpu | matmul_bwd | float64 | — | — | — | — | — | — | — | — | — | 7.498 | — | f64 | 1.28e-13 |
| cpu | attention | float64 | — | — | — | — | — | — | — | — | — | 99.003 | — | f64 | 3.75e-16 |
| cpu | attention_causal | float64 | — | — | — | — | — | — | — | — | — | 88.724 | — | f64 | 4.02e-16 |
| cpu | attention_fused | float64 | — | — | — | — | — | — | — | — | — | 200.052 | — | f64 | 4.51e-16 |
| cpu | attention_fused_causal | float64 | — | — | — | — | — | — | — | — | — | 99.921 | — | f64 | 4.3e-16 |
| cpu | attention_decode | float64 | — | — | — | — | — | — | — | — | — | 3.449 | — | f64 | 2.22e-16 |
| cpu | attention_bwd | float64 | — | — | — | — | — | — | — | — | — | 27.390 | — | f64 | 1.39e-16 |
| cpu | optimizer:sgd_momentum | float64 | — | — | — | — | — | — | — | — | — | 19.925 | — | f64; + restore of p, m, v | 1.11e-16 |
| cpu | optimizer:adam | float64 | — | — | — | — | — | — | — | — | — | 30.329 | — | f64; + restore of p, m, v | 1.11e-16 |
| cpu | optimizer:adamw | float64 | — | — | — | — | — | — | — | — | — | 30.213 | — | f64; + restore of p, m, v | 1.11e-16 |
| cpu | conv | float64 | — | — | — | — | — | — | — | — | — | 6.921 | — | f64 | 7.11e-14 |
| cpu | conv_bwd | float64 | — | — | — | — | — | — | — | — | — | 30.051 | — | f64 | 2.13e-14 |
| cpu | lstm_step | float64 | — | — | — | — | — | — | — | — | — | 31.472 | — | f64 | 7.22e-16 |
| cpu | gru_step | float64 | — | — | — | — | — | — | — | — | — | 21.152 | — | f64 | 9.99e-16 |
| cpu | lstm_step_bwd | float64 | — | — | — | — | — | — | — | — | — | 88.303 | — | f64 | 1.67e-15 |
| cpu | gru_step_bwd | float64 | — | — | — | — | — | — | — | — | — | 72.014 | — | f64 | 7.77e-16 |
| cpu | lstm_seq | float64 | — | — | — | — | — | — | — | — | — | 63.800 | — | f64 | 3.61e-16 |
| cpu | lstm_seq_bwd | float64 | — | — | — | — | — | — | — | — | — | 258.586 | — | f64 | 1.33e-15 |
| cpu | gru_seq | float64 | — | — | — | — | — | — | — | — | — | 44.408 | — | f64 | 6.66e-16 |
| cpu | gru_seq_bwd | float64 | — | — | — | — | — | — | — | — | — | 231.356 | — | f64 | 1.33e-15 |
| cpu | topk_rows | float64 | — | — | — | — | — | — | — | — | — | 82.086 | — | f64 | 0e+0 |
| cpu | pick_rows | float64 | — | — | — | — | — | — | — | — | — | 0.029 | — | f64 | 0e+0 |
| cpu | sample_rows | float64 | — | — | — | — | — | — | — | — | — | 13.632 | — | f64 | 0e+0 |
| cpu | route | s32 | — | — | — | — | — | — | — | — | — | 1.171 | — | s32 | 0e+0 |
| cpu | route_wide | s32 | — | — | — | — | — | — | — | — | — | 2.639 | — | s32 | 0e+0 |
| cpu | cross_entropy | float64 | — | — | — | — | — | — | — | — | — | 45.986 | — | f64 (log_softmax, pick, neg) | 1.78e-14 |
| cpu | ctc | float64 | — | — | — | — | — | — | — | — | — | 4.675 | — | f64 | 4.55e-13 |
| cpu | ctc_bwd | float64 | — | — | — | — | — | — | — | — | — | 8.043 | — | f64 | 6.81e-13 |
| cpu | softmax | float64 | 4096x4096 | 31.985 | — | to f32 and back | 6.21e-10 | 6.392 | 5.00 | Vector<Double> | 4.34e-19 | 4.812 | 6.65 | f64 | 1.08e-17 |
| cpu | rms_norm | float64 | 4096x4096 | 31.035 | — | to f32 and back | 3.73e-6 | 4.078 | 7.61 | Vector<Double> | 1.33e-14 | 4.664 | 6.65 | f64 | 1.27e-14 |
| cpu | add | float32 | 16777216 | 3.240 | — | f32 kernel | 5.96e-8 | 3.042 | 1.07 | Vector<Single> | 5.96e-8 | 2.936 | 1.10 | f32 | 5.96e-8 |
| cpu | matmul | float32 | 1024x1024x1024 | 2.703 | — | f32 kernel | 7.96e-5 | 4.105 | 0.66 | Vector<Single> | 8.34e-5 | 1.676 | 1.61 | f32 | 7.96e-5 |
| cpu | map:copy | float32 | — | — | — | — | — | — | — | — | — | 1.664 | — | f32 | 0e+0 |
| cpu | map:fill | float32 | — | — | — | — | — | — | — | — | — | 0.771 | — | f32 | 0e+0 |
| cpu | map:neg | float32 | — | — | — | — | — | — | — | — | — | 1.679 | — | f32 | 0e+0 |
| cpu | map:abs | float32 | — | — | — | — | — | — | — | — | — | 1.644 | — | f32 | 0e+0 |
| cpu | map:sign | float32 | — | — | — | — | — | — | — | — | — | 1.653 | — | f32 | 0e+0 |
| cpu | map:square | float32 | — | — | — | — | — | — | — | — | — | 1.647 | — | f32 | 4.77e-7 |
| cpu | map:relu | float32 | — | — | — | — | — | — | — | — | — | 1.884 | — | f32 | 0e+0 |
| cpu | map:add_scalar | float32 | — | — | — | — | — | — | — | — | — | 1.708 | — | f32 | 2.38e-7 |
| cpu | map:scale | float32 | — | — | — | — | — | — | — | — | — | 1.671 | — | f32 | 4.77e-7 |
| cpu | map:affine | float32 | — | — | — | — | — | — | — | — | — | 1.648 | — | f32 | 4.77e-7 |
| cpu | map:clamp | float32 | — | — | — | — | — | — | — | — | — | 1.634 | — | f32 | 0e+0 |
| cpu | map:add | float32 | — | — | — | — | — | — | — | — | — | 3.005 | — | f32 | 2.38e-7 |
| cpu | map:sub | float32 | — | — | — | — | — | — | — | — | — | 3.000 | — | f32 | 2.38e-7 |
| cpu | map:mul | float32 | — | — | — | — | — | — | — | — | — | 2.933 | — | f32 | 4.77e-7 |
| cpu | map:div | float32 | — | — | — | — | — | — | — | — | — | 3.006 | — | f32 | 1.19e-7 |
| cpu | map:max | float32 | — | — | — | — | — | — | — | — | — | 2.896 | — | f32 | 0e+0 |
| cpu | map:min | float32 | — | — | — | — | — | — | — | — | — | 2.873 | — | f32 | 0e+0 |
| cpu | map:axpy | float32 | — | — | — | — | — | — | — | — | — | 2.884 | — | f32 | 4.77e-7 |
| cpu | map:muladd | float32 | — | — | — | — | — | — | — | — | — | 4.004 | — | f32 | 9.54e-7 |
| cpu | map:where | float32 | — | — | — | — | — | — | — | — | — | 4.006 | — | f32 | 0e+0 |
| cpu | map:square_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.923 | — | f32 | 9.54e-7 |
| cpu | map:abs_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.892 | — | f32 | 0e+0 |
| cpu | map:relu_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.950 | — | f32 | 0e+0 |
| cpu | map:clamp_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.291 | — | f32 | 0e+0 |
| cpu | map:where_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.934 | — | f32 | 0e+0 |
| cpu | map:sqrt | float32 | — | — | — | — | — | — | — | — | — | 1.741 | — | f32 | 5.96e-8 |
| cpu | map:inv_sqrt | float32 | — | — | — | — | — | — | — | — | — | 1.764 | — | f32 | 1.37e-7 |
| cpu | map:exp | float32 | — | — | — | — | — | — | — | — | — | 1.697 | — | f32 | 3.42e-6 |
| cpu | map:expm1 | float32 | — | — | — | — | — | — | — | — | — | 1.800 | — | f32 | 6.84e-8 |
| cpu | map:log | float32 | — | — | — | — | — | — | — | — | — | 1.667 | — | f32 | 1.26e-7 |
| cpu | map:sin | float32 | — | — | — | — | — | — | — | — | — | 1.969 | — | f32 | 7.99e-8 |
| cpu | map:cos | float32 | — | — | — | — | — | — | — | — | — | 2.632 | — | f32 | 7.96e-8 |
| cpu | map:tanh | float32 | — | — | — | — | — | — | — | — | — | 2.828 | — | f32 | 1.49e-7 |
| cpu | map:sigmoid | float32 | — | — | — | — | — | — | — | — | — | 2.189 | — | f32 | 8.77e-8 |
| cpu | map:erf | float32 | — | — | — | — | — | — | — | — | — | 22.573 | — | f32 | 1.45e-7 |
| cpu | map:gelu | float32 | — | — | — | — | — | — | — | — | — | 1.752 | — | f32 | 4.28e-7 |
| cpu | map:gelu_erf | float32 | — | — | — | — | — | — | — | — | — | 25.798 | — | f32 | 2.37e-7 |
| cpu | map:silu | float32 | — | — | — | — | — | — | — | — | — | 1.727 | — | f32 | 4.58e-7 |
| cpu | map:leaky_relu | float32 | — | — | — | — | — | — | — | — | — | 1.658 | — | f32 | 1.79e-8 |
| cpu | map:pow | float32 | — | — | — | — | — | — | — | — | — | 5.379 | — | f32 | 9.47e-6 |
| cpu | map:dropout | float32 | — | — | — | — | — | — | — | — | — | 3.022 | — | f32 | 2.38e-7 |
| cpu | map:gated_silu | float32 | — | — | — | — | — | — | — | — | — | 3.182 | — | f32 | 2.24e-6 |
| cpu | map:gated_gelu | float32 | — | — | — | — | — | — | — | — | — | 2.868 | — | f32 | 2.04e-6 |
| cpu | map:sqrt_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.917 | — | f32 | 2.29e-7 |
| cpu | map:inv_sqrt_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.023 | — | f32 | 1.04e-6 |
| cpu | map:exp_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.952 | — | f32 | 1.95e-5 |
| cpu | map:log_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.069 | — | f32 | 2.38e-7 |
| cpu | map:sin_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.195 | — | f32 | 4.04e-7 |
| cpu | map:cos_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.123 | — | f32 | 4e-7 |
| cpu | map:tanh_bwd | float32 | — | — | — | — | — | — | — | — | — | 4.195 | — | f32 | 9.3e-7 |
| cpu | map:sigmoid_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.278 | — | f32 | 2.31e-7 |
| cpu | map:silu_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.328 | — | f32 | 9.79e-7 |
| cpu | map:gelu_bwd | float32 | — | — | — | — | — | — | — | — | — | 4.593 | — | f32 | 8.92e-7 |
| cpu | map:gelu_erf_bwd | float32 | — | — | — | — | — | — | — | — | — | 28.510 | — | f32 | 8.35e-7 |
| cpu | map:leaky_relu_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.886 | — | f32 | 1.79e-8 |
| cpu | map:pow_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.677 | — | f32 | 2.48e-5 |
| cpu | map:gated_silu_bwd_x | float32 | — | — | — | — | — | — | — | — | — | 3.928 | — | f32 | 4.11e-6 |
| cpu | map:gated_silu_bwd_y | float32 | — | — | — | — | — | — | — | — | — | 3.910 | — | f32 | 2.12e-6 |
| cpu | map:gated_gelu_bwd_x | float32 | — | — | — | — | — | — | — | — | — | 3.804 | — | f32 | 2.92e-6 |
| cpu | map:gated_gelu_bwd_y | float32 | — | — | — | — | — | — | — | — | — | 3.904 | — | f32 | 2.07e-6 |
| cpu | sum:whole | float32 | — | — | — | — | — | — | — | — | — | 0.339 | — | f32 | 3.71e-3 |
| cpu | sum:rows | float32 | — | — | — | — | — | — | — | — | — | 0.339 | — | f32 | 3.58e-5 |
| cpu | sum:cols | float32 | — | — | — | — | — | — | — | — | — | 1.464 | — | f32 | 3.28e-4 |
| cpu | sumsq:rows | float32 | — | — | — | — | — | — | — | — | — | 0.341 | — | f32 | 2.46e-4 |
| cpu | max:whole | float32 | — | — | — | — | — | — | — | — | — | 0.782 | — | f32 | 0e+0 |
| cpu | max:rows | float32 | — | — | — | — | — | — | — | — | — | 0.811 | — | f32 | 0e+0 |
| cpu | max:cols | float32 | — | — | — | — | — | — | — | — | — | 2.302 | — | f32 | 0e+0 |
| cpu | min:rows | float32 | — | — | — | — | — | — | — | — | — | 0.997 | — | f32 | 0e+0 |
| cpu | argmax:rows | float32 | — | — | — | — | — | — | — | — | — | 0.939 | — | s32 index | 0e+0 |
| cpu | argmax:cols | float32 | — | — | — | — | — | — | — | — | — | 1.872 | — | s32 index | 0e+0 |
| cpu | maprow:add | float32 | — | — | — | — | — | — | — | — | — | 1.644 | — | f32 | 5.96e-8 |
| cpu | maprow:mul | float32 | — | — | — | — | — | — | — | — | — | 1.732 | — | f32 | 2.98e-8 |
| cpu | transpose | float32 | — | — | — | — | — | — | — | — | — | 2.892 | — | f32 | 0e+0 |
| cpu | gemv | float32 | — | — | — | — | — | — | — | — | — | 2.943 | — | f32 | 5.19e-5 |
| cpu | gemv_t | float32 | — | — | — | — | — | — | — | — | — | 6.654 | — | f32 | 4.18e-4 |
| cpu | row:log_softmax | float32 | — | — | — | — | — | — | — | — | — | 1.795 | — | f32 | 1.36e-6 |
| cpu | row:softmax_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.881 | — | f32 | 3.02e-5 |
| cpu | row:log_softmax_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.975 | — | f32 | 4.67e-5 |
| cpu | row:rms_norm_affine | float32 | — | — | — | — | — | — | — | — | — | 1.685 | — | f32 | 8.11e-7 |
| cpu | row:rms_norm_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.026 | — | f32 | 2.07e-7 |
| cpu | row:layer_norm | float32 | — | — | — | — | — | — | — | — | — | 1.705 | — | f32 | 8.74e-7 |
| cpu | row:layer_norm_bwd | float32 | — | — | — | — | — | — | — | — | — | 2.884 | — | f32 | 2.11e-7 |
| cpu | row:weight_norm | float32 | — | — | — | — | — | — | — | — | — | 1.696 | — | f32 | 9.55e-9 |
| cpu | row:rope | float32 | — | — | — | — | — | — | — | — | — | 4.709 | — | f32 | 1.35e-3 |
| cpu | row:scale_mask_softmax | float32 | — | — | — | — | — | — | — | — | — | 3.372 | — | f32 | 4.81e-8 |
| cpu | row:add_rms_norm | float32 | — | — | — | — | — | — | — | — | — | 4.909 | — | f32 | 1.22e-6 |
| cpu | row:add_layer_norm | float32 | — | — | — | — | — | — | — | — | — | 4.901 | — | f32 | 1.28e-6 |
| cpu | row:cross_entropy_bwd | float32 | — | — | — | — | — | — | — | — | — | 1.811 | — | f32 | 2.98e-8 |
| cpu | norm_params_bwd:rms | float32 | — | — | — | — | — | — | — | — | — | 3.329 | — | f32 | 3.09e-4 |
| cpu | norm_params_bwd:layer | float32 | — | — | — | — | — | — | — | — | — | 3.373 | — | f32 | 3.09e-4 |
| cpu | layout:strided | float32 | — | — | — | — | — | — | — | — | — | 15.557 | — | f32 | 0e+0 |
| cpu | layout:one_hot | float32 | — | — | — | — | — | — | — | — | — | 0.490 | — | f32 | 0e+0 |
| cpu | layout:gather_rows | float32 | — | — | — | — | — | — | — | — | — | 1.477 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | float32 | — | — | — | — | — | — | — | — | — | 2.718 | — | f32 | 0e+0 |
| cpu | layout:scatter_add_rows | float32 | — | — | — | — | — | — | — | — | — | 2.354 | — | f32 | 4.77e-7 |
| cpu | layout:im2col | float32 | — | — | — | — | — | — | — | — | — | 1.036 | — | f32 | 0e+0 |
| cpu | layout:col2im | float32 | — | — | — | — | — | — | — | — | — | 2.090 | — | f32 | 1.08e-6 |
| cpu | layout:interpolate_nearest | float32 | — | — | — | — | — | — | — | — | — | 2.301 | — | f32 | 0e+0 |
| cpu | layout:interpolate_bilinear | float32 | — | — | — | — | — | — | — | — | — | 6.063 | — | f32 | 1.19e-7 |
| cpu | layout:shift | float32 | — | — | — | — | — | — | — | — | — | 1.703 | — | f32 | 0e+0 |
| cpu | layout:pad2d | float32 | — | — | — | — | — | — | — | — | — | 1.588 | — | f32 | 0e+0 |
| cpu | layout:maxpool | float32 | — | — | — | — | — | — | — | — | — | 1.500 | — | f32 | 0e+0 |
| cpu | layout:avgpool | float32 | — | — | — | — | — | — | — | — | — | 0.513 | — | f32 | 7.45e-8 |
| cpu | layout:maxpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 7.200 | — | f32 | 0e+0 |
| cpu | layout:avgpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.224 | — | f32 | 0e+0 |
| cpu | layout:adaptive_maxpool | float32 | — | — | — | — | — | — | — | — | — | 7.243 | — | f32 | 0e+0 |
| cpu | layout:adaptive_avgpool | float32 | — | — | — | — | — | — | — | — | — | 4.156 | — | f32 | 7.4e-8 |
| cpu | layout:depthwise_conv | float32 | — | — | — | — | — | — | — | — | — | 3.943 | — | f32 | 6.56e-7 |
| cpu | layout:adaptive_maxpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 685.466 | — | f32 | 5.96e-8 |
| cpu | layout:adaptive_avgpool_bwd | float32 | — | — | — | — | — | — | — | — | — | 7.992 | — | f32 | 2.08e-10 |
| cpu | matmul_nt | float32 | — | — | — | — | — | — | — | — | — | 1.802 | — | f32 | 8.59e-5 |
| cpu | matmul_tn | float32 | — | — | — | — | — | — | — | — | — | 1.511 | — | f32 | 7.82e-5 |
| cpu | matmul_tt | float32 | — | — | — | — | — | — | — | — | — | 1.834 | — | f32 | 7.81e-5 |
| cpu | matmul_batched | float32 | — | — | — | — | — | — | — | — | — | 0.790 | — | f32 | 1.92e-5 |
| cpu | matmul_bwd | float32 | — | — | — | — | — | — | — | — | — | 3.348 | — | f32 | 8.05e-5 |
| cpu | attention | float32 | — | — | — | — | — | — | — | — | — | 51.273 | — | f32 | 1.63e-7 |
| cpu | attention_causal | float32 | — | — | — | — | — | — | — | — | — | 40.174 | — | f32 | 1.46e-7 |
| cpu | attention_fused | float32 | — | — | — | — | — | — | — | — | — | 78.230 | — | f32 | 1.39e-7 |
| cpu | attention_fused_causal | float32 | — | — | — | — | — | — | — | — | — | 40.100 | — | f32 | 1.46e-7 |
| cpu | attention_decode | float32 | — | — | — | — | — | — | — | — | — | 1.729 | — | f32 | 6.32e-8 |
| cpu | attention_bwd | float32 | — | — | — | — | — | — | — | — | — | 16.267 | — | f32 | 1.47e-7 |
| cpu | optimizer:sgd_momentum | float32 | — | — | — | — | — | — | — | — | — | 10.869 | — | f32; + restore of p, m, v | 5.98e-8 |
| cpu | optimizer:adam | float32 | — | — | — | — | — | — | — | — | — | 14.829 | — | f32; + restore of p, m, v | 5.95e-8 |
| cpu | optimizer:adamw | float32 | — | — | — | — | — | — | — | — | — | 14.106 | — | f32; + restore of p, m, v | 7.29e-8 |
| cpu | conv | float32 | — | — | — | — | — | — | — | — | — | 2.310 | — | f32 | 4.36e-5 |
| cpu | conv_bwd | float32 | — | — | — | — | — | — | — | — | — | 16.128 | — | f32 | 1.56e-5 |
| cpu | lstm_step | float32 | — | — | — | — | — | — | — | — | — | 13.957 | — | f32 | 5.76e-7 |
| cpu | gru_step | float32 | — | — | — | — | — | — | — | — | — | 12.267 | — | f32 | 7.78e-7 |
| cpu | lstm_step_bwd | float32 | — | — | — | — | — | — | — | — | — | 47.287 | — | f32 | 1.81e-6 |
| cpu | gru_step_bwd | float32 | — | — | — | — | — | — | — | — | — | 38.303 | — | f32 | 8.33e-7 |
| cpu | lstm_seq | float32 | — | — | — | — | — | — | — | — | — | 27.221 | — | f32 | 2.76e-7 |
| cpu | lstm_seq_bwd | float32 | — | — | — | — | — | — | — | — | — | 149.939 | — | f32 | 9.1e-7 |
| cpu | gru_seq | float32 | — | — | — | — | — | — | — | — | — | 28.874 | — | f32 | 5.31e-7 |
| cpu | gru_seq_bwd | float32 | — | — | — | — | — | — | — | — | — | 109.379 | — | f32 | 6.81e-7 |
| cpu | topk_rows | float32 | — | — | — | — | — | — | — | — | — | 87.360 | — | f32 | 0e+0 |
| cpu | pick_rows | float32 | — | — | — | — | — | — | — | — | — | 0.020 | — | f32 | 0e+0 |
| cpu | sample_rows | float32 | — | — | — | — | — | — | — | — | — | 7.096 | — | f32 | 2e+0 |
| cpu | cross_entropy | float32 | — | — | — | — | — | — | — | — | — | 21.872 | — | f32 (log_softmax, pick, neg) | 1.96e-6 |
| cpu | ctc | float32 | — | — | — | — | — | — | — | — | — | 2.123 | — | f32 | 1.36e-3 |
| cpu | ctc_bwd | float32 | — | — | — | — | — | — | — | — | — | 4.140 | — | f32 | 1.67e-3 |
| cpu | softmax | float32 | 4096x4096 | 2.502 | — | f32 kernel | 6.21e-10 | 4.244 | 0.59 | Vector<Single> | 5.91e-9 | 1.806 | 1.39 | f32 | 6.21e-10 |
| cpu | rms_norm | float32 | 4096x4096 | 2.256 | — | f32 kernel | 3.73e-6 | 1.860 | 1.21 | Vector<Single> | 5.16e-7 | 1.649 | 1.37 | f32 | 3.68e-7 |
| cpu | add | float16 | 16777216 | 2.908 | — | typed tile | 4.88e-4 | 7.633 | 0.38 | scalar | 4.88e-4 | 1.242 | 2.34 | f32 | 4.88e-4 |
| cpu | matmul | float16 | 1024x1024x1024 | 7.070 | — | to f32 and back | 1.56e-2 | 881.784 | 0.01 | scalar | 6.08e-1 | 1.731 | 4.08 | f32 | 1.56e-2 |
| cpu | map:copy | float16 | — | — | — | — | — | — | — | — | — | 1.249 | — | f32 | 0e+0 |
| cpu | map:fill | float16 | — | — | — | — | — | — | — | — | — | 0.377 | — | f32 | 0e+0 |
| cpu | map:neg | float16 | — | — | — | — | — | — | — | — | — | 1.262 | — | f32 | 0e+0 |
| cpu | map:abs | float16 | — | — | — | — | — | — | — | — | — | 1.278 | — | f32 | 0e+0 |
| cpu | map:sign | float16 | — | — | — | — | — | — | — | — | — | 1.428 | — | f32 | 0e+0 |
| cpu | map:square | float16 | — | — | — | — | — | — | — | — | — | 1.270 | — | f32 | 3.91e-3 |
| cpu | map:relu | float16 | — | — | — | — | — | — | — | — | — | 1.302 | — | f32 | 0e+0 |
| cpu | map:add_scalar | float16 | — | — | — | — | — | — | — | — | — | 1.293 | — | f32 | 1.95e-3 |
| cpu | map:scale | float16 | — | — | — | — | — | — | — | — | — | 1.271 | — | f32 | 3.91e-3 |
| cpu | map:affine | float16 | — | — | — | — | — | — | — | — | — | 1.271 | — | f32 | 3.91e-3 |
| cpu | map:clamp | float16 | — | — | — | — | — | — | — | — | — | 1.572 | — | f32 | 0e+0 |
| cpu | map:add | float16 | — | — | — | — | — | — | — | — | — | 1.788 | — | f32 | 1.95e-3 |
| cpu | map:sub | float16 | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 | 1.95e-3 |
| cpu | map:mul | float16 | — | — | — | — | — | — | — | — | — | 1.194 | — | f32 | 3.91e-3 |
| cpu | map:div | float16 | — | — | — | — | — | — | — | — | — | 1.094 | — | f32 | 9.76e-4 |
| cpu | map:max | float16 | — | — | — | — | — | — | — | — | — | 1.259 | — | f32 | 0e+0 |
| cpu | map:min | float16 | — | — | — | — | — | — | — | — | — | 1.257 | — | f32 | 0e+0 |
| cpu | map:axpy | float16 | — | — | — | — | — | — | — | — | — | 1.204 | — | f32 | 3.91e-3 |
| cpu | map:muladd | float16 | — | — | — | — | — | — | — | — | — | 2.383 | — | f32 | 7.81e-3 |
| cpu | map:where | float16 | — | — | — | — | — | — | — | — | — | 1.658 | — | f32 | 0e+0 |
| cpu | map:square_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.196 | — | f32 | 7.81e-3 |
| cpu | map:abs_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.246 | — | f32 | 0e+0 |
| cpu | map:relu_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.220 | — | f32 | 0e+0 |
| cpu | map:clamp_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.216 | — | f32 | 0e+0 |
| cpu | map:where_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.210 | — | f32 | 0e+0 |
| cpu | map:sqrt | float16 | — | — | — | — | — | — | — | — | — | 0.762 | — | f32 | 4.88e-4 |
| cpu | map:inv_sqrt | float16 | — | — | — | — | — | — | — | — | — | 0.910 | — | f32 | 4.88e-4 |
| cpu | map:exp | float16 | — | — | — | — | — | — | — | — | — | 1.521 | — | f32 | 1.55e-2 |
| cpu | map:expm1 | float16 | — | — | — | — | — | — | — | — | — | 3.364 | — | f32 | 2.44e-4 |
| cpu | map:log | float16 | — | — | — | — | — | — | — | — | — | 1.613 | — | f32 | 4.88e-4 |
| cpu | map:sin | float16 | — | — | — | — | — | — | — | — | — | 4.113 | — | f32 | 2.44e-4 |
| cpu | map:cos | float16 | — | — | — | — | — | — | — | — | — | 3.798 | — | f32 | 2.44e-4 |
| cpu | map:tanh | float16 | — | — | — | — | — | — | — | — | — | 3.978 | — | f32 | 2.44e-4 |
| cpu | map:sigmoid | float16 | — | — | — | — | — | — | — | — | — | 3.077 | — | f32 | 2.44e-4 |
| cpu | map:erf | float16 | — | — | — | — | — | — | — | — | — | 23.263 | — | f32 | 2.44e-4 |
| cpu | map:gelu | float16 | — | — | — | — | — | — | — | — | — | 3.436 | — | f32 | 9.73e-4 |
| cpu | map:gelu_erf | float16 | — | — | — | — | — | — | — | — | — | 37.528 | — | f32 | 9.76e-4 |
| cpu | map:silu | float16 | — | — | — | — | — | — | — | — | — | 3.127 | — | f32 | 9.75e-4 |
| cpu | map:leaky_relu | float16 | — | — | — | — | — | — | — | — | — | 1.397 | — | f32 | 9.77e-5 |
| cpu | map:pow | float16 | — | — | — | — | — | — | — | — | — | 6.531 | — | f32 | 7.8e-3 |
| cpu | map:dropout | float16 | — | — | — | — | — | — | — | — | — | 1.807 | — | f32 | 1.95e-3 |
| cpu | map:gated_silu | float16 | — | — | — | — | — | — | — | — | — | 2.111 | — | f32 | 3.91e-3 |
| cpu | map:gated_gelu | float16 | — | — | — | — | — | — | — | — | — | 4.048 | — | f32 | 3.91e-3 |
| cpu | map:sqrt_bwd | float16 | — | — | — | — | — | — | — | — | — | 2.324 | — | f32 | 9.77e-4 |
| cpu | map:inv_sqrt_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.499 | — | f32 | 1.95e-3 |
| cpu | map:exp_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.912 | — | f32 | 6.25e-2 |
| cpu | map:log_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.094 | — | f32 | 1.95e-3 |
| cpu | map:sin_bwd | float16 | — | — | — | — | — | — | — | — | — | 2.923 | — | f32 | 9.77e-4 |
| cpu | map:cos_bwd | float16 | — | — | — | — | — | — | — | — | — | 4.469 | — | f32 | 9.77e-4 |
| cpu | map:tanh_bwd | float16 | — | — | — | — | — | — | — | — | — | 5.763 | — | f32 | 9.77e-4 |
| cpu | map:sigmoid_bwd | float16 | — | — | — | — | — | — | — | — | — | 5.830 | — | f32 | 2.44e-4 |
| cpu | map:silu_bwd | float16 | — | — | — | — | — | — | — | — | — | 5.820 | — | f32 | 1.95e-3 |
| cpu | map:gelu_bwd | float16 | — | — | — | — | — | — | — | — | — | 4.147 | — | f32 | 1.95e-3 |
| cpu | map:gelu_erf_bwd | float16 | — | — | — | — | — | — | — | — | — | 44.438 | — | f32 | 1.95e-3 |
| cpu | map:leaky_relu_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.227 | — | f32 | 9.77e-5 |
| cpu | map:pow_bwd | float16 | — | — | — | — | — | — | — | — | — | 4.769 | — | f32 | 3.13e-2 |
| cpu | map:gated_silu_bwd_x | float16 | — | — | — | — | — | — | — | — | — | 3.613 | — | f32 | 7.81e-3 |
| cpu | map:gated_silu_bwd_y | float16 | — | — | — | — | — | — | — | — | — | 2.855 | — | f32 | 3.91e-3 |
| cpu | map:gated_gelu_bwd_x | float16 | — | — | — | — | — | — | — | — | — | 4.466 | — | f32 | 3.91e-3 |
| cpu | map:gated_gelu_bwd_y | float16 | — | — | — | — | — | — | — | — | — | 3.027 | — | f32 | 3.91e-3 |
| cpu | sum:whole | float16 | — | — | — | — | — | — | — | — | — | 0.401 | — | f32 | 1.53e-4 |
| cpu | sum:rows | float16 | — | — | — | — | — | — | — | — | — | 0.399 | — | f32 | 1.05e-5 |
| cpu | sum:cols | float16 | — | — | — | — | — | — | — | — | — | 0.958 | — | f32 | 6.44e-5 |
| cpu | sumsq:rows | float16 | — | — | — | — | — | — | — | — | — | 0.398 | — | f32 | 2.62e-4 |
| cpu | max:whole | float16 | — | — | — | — | — | — | — | — | — | 0.755 | — | f32 | 0e+0 |
| cpu | max:rows | float16 | — | — | — | — | — | — | — | — | — | 1.315 | — | f32 | 0e+0 |
| cpu | max:cols | float16 | — | — | — | — | — | — | — | — | — | 3.728 | — | f32 | 0e+0 |
| cpu | min:rows | float16 | — | — | — | — | — | — | — | — | — | 1.586 | — | f32 | 0e+0 |
| cpu | argmax:rows | float16 | — | — | — | — | — | — | — | — | — | 1.294 | — | s32 index | 0e+0 |
| cpu | argmax:cols | float16 | — | — | — | — | — | — | — | — | — | 1.366 | — | s32 index | 0e+0 |
| cpu | maprow:add | float16 | — | — | — | — | — | — | — | — | — | 0.858 | — | f32 | 4.88e-4 |
| cpu | maprow:mul | float16 | — | — | — | — | — | — | — | — | — | 0.862 | — | f32 | 2.44e-4 |
| cpu | transpose | float16 | — | — | — | — | — | — | — | — | — | 1.509 | — | f32 | 0e+0 |
| cpu | gemv | float16 | — | — | — | — | — | — | — | — | — | 1.829 | — | f32 | 3.12e-2 |
| cpu | gemv_t | float16 | — | — | — | — | — | — | — | — | — | 7.167 | — | f32 | 5.38e-2 |
| cpu | row:log_softmax | float16 | — | — | — | — | — | — | — | — | — | 3.127 | — | f32 | 3.91e-3 |
| cpu | row:softmax_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.936 | — | f32 | 3.12e-2 |
| cpu | row:log_softmax_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.830 | — | f32 | 3.13e-2 |
| cpu | row:rms_norm_affine | float16 | — | — | — | — | — | — | — | — | — | 0.885 | — | f32 | 9.77e-4 |
| cpu | row:rms_norm_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.350 | — | f32 | 2.44e-4 |
| cpu | row:layer_norm | float16 | — | — | — | — | — | — | — | — | — | 0.942 | — | f32 | 1.95e-3 |
| cpu | row:layer_norm_bwd | float16 | — | — | — | — | — | — | — | — | — | 1.436 | — | f32 | 2.44e-4 |
| cpu | row:weight_norm | float16 | — | — | — | — | — | — | — | — | — | 0.851 | — | f32 | 1.53e-5 |
| cpu | row:rope | float16 | — | — | — | — | — | — | — | — | — | 3.466 | — | f32 | 2.53e-3 |
| cpu | row:scale_mask_softmax | float16 | — | — | — | — | — | — | — | — | — | 3.326 | — | f32 | 2.39e-4 |
| cpu | row:add_rms_norm | float16 | — | — | — | — | — | — | — | — | — | 1.961 | — | f32 | 1.95e-3 |
| cpu | row:add_layer_norm | float16 | — | — | — | — | — | — | — | — | — | 2.103 | — | f32 | 1.95e-3 |
| cpu | row:cross_entropy_bwd | float16 | — | — | — | — | — | — | — | — | — | 3.108 | — | f32 | 2.44e-4 |
| cpu | norm_params_bwd:rms | float16 | — | — | — | — | — | — | — | — | — | 4.734 | — | f32 | 2.92e-4 |
| cpu | norm_params_bwd:layer | float16 | — | — | — | — | — | — | — | — | — | 4.796 | — | f32 | 3.75e-4 |
| cpu | layout:strided | float16 | — | — | — | — | — | — | — | — | — | 31.350 | — | f32 | 0e+0 |
| cpu | layout:one_hot | float16 | — | — | — | — | — | — | — | — | — | 0.556 | — | f32 | 0e+0 |
| cpu | layout:gather_rows | float16 | — | — | — | — | — | — | — | — | — | 1.151 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | float16 | — | — | — | — | — | — | — | — | — | 21.250 | — | f32 | 0e+0 |
| cpu | layout:scatter_add_rows | float16 | — | — | — | — | — | — | — | — | — | 21.339 | — | f32 | 1.95e-3 |
| cpu | layout:im2col | float16 | — | — | — | — | — | — | — | — | — | 11.886 | — | f32 | 0e+0 |
| cpu | layout:col2im | float16 | — | — | — | — | — | — | — | — | — | 12.298 | — | f32 | 1.95e-3 |
| cpu | layout:interpolate_nearest | float16 | — | — | — | — | — | — | — | — | — | 22.955 | — | f32 | 0e+0 |
| cpu | layout:interpolate_bilinear | float16 | — | — | — | — | — | — | — | — | — | 97.396 | — | f32 | 2.44e-4 |
| cpu | layout:shift | float16 | — | — | — | — | — | — | — | — | — | 20.055 | — | f32 | 0e+0 |
| cpu | layout:pad2d | float16 | — | — | — | — | — | — | — | — | — | 1.175 | — | f32 | 0e+0 |
| cpu | layout:maxpool | float16 | — | — | — | — | — | — | — | — | — | 6.148 | — | f32 | 0e+0 |
| cpu | layout:avgpool | float16 | — | — | — | — | — | — | — | — | — | 5.698 | — | f32 | 2.44e-4 |
| cpu | layout:maxpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 31.848 | — | f32 | 0e+0 |
| cpu | layout:avgpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 7.995 | — | f32 | 2.98e-8 |
| cpu | layout:adaptive_maxpool | float16 | — | — | — | — | — | — | — | — | — | 87.676 | — | f32 | 0e+0 |
| cpu | layout:adaptive_avgpool | float16 | — | — | — | — | — | — | — | — | — | 82.989 | — | f32 | 1.53e-5 |
| cpu | layout:depthwise_conv | float16 | — | — | — | — | — | — | — | — | — | 85.897 | — | f32 | 1.95e-3 |
| cpu | layout:adaptive_maxpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 9921.996 | — | f32 | 4.88e-4 |
| cpu | layout:adaptive_avgpool_bwd | float16 | — | — | — | — | — | — | — | — | — | 10.239 | — | f32 | 9.44e-7 |
| cpu | matmul_nt | float16 | — | — | — | — | — | — | — | — | — | 1.343 | — | f32 | 1.56e-2 |
| cpu | matmul_tn | float16 | — | — | — | — | — | — | — | — | — | 2.034 | — | f32 | 1.56e-2 |
| cpu | matmul_tt | float16 | — | — | — | — | — | — | — | — | — | 1.558 | — | f32 | 1.56e-2 |
| cpu | matmul_batched | float16 | — | — | — | — | — | — | — | — | — | 0.786 | — | f32 | 7.81e-3 |
| cpu | matmul_bwd | float16 | — | — | — | — | — | — | — | — | — | 4.100 | — | f32 | 1.56e-2 |
| cpu | attention | float16 | — | — | — | — | — | — | — | — | — | 64.564 | — | f32 | 2.27e-5 |
| cpu | attention_causal | float16 | — | — | — | — | — | — | — | — | — | 44.280 | — | f32 | 2.44e-4 |
| cpu | attention_fused | float16 | — | — | — | — | — | — | — | — | — | 81.181 | — | f32 | 2.27e-5 |
| cpu | attention_fused_causal | float16 | — | — | — | — | — | — | — | — | — | 43.941 | — | f32 | 2.44e-4 |
| cpu | attention_decode | float16 | — | — | — | — | — | — | — | — | — | 1.549 | — | f32 | 1.34e-5 |
| cpu | attention_bwd | float16 | — | — | — | — | — | — | — | — | — | 18.200 | — | f32 | 1.2e-4 |
| cpu | optimizer:sgd_momentum | float16 | — | — | — | — | — | — | — | — | — | 6.913 | — | f32; + restore of p, m, v | 4.88e-4 |
| cpu | optimizer:adam | float16 | — | — | — | — | — | — | — | — | — | 11.580 | — | f32; + restore of p, m, v | 4.88e-4 |
| cpu | optimizer:adamw | float16 | — | — | — | — | — | — | — | — | — | 11.735 | — | f32; + restore of p, m, v | 4.87e-4 |
| cpu | conv | float16 | — | — | — | — | — | — | — | — | — | 2.565 | — | f32 | 1.55e-2 |
| cpu | conv_bwd | float16 | — | — | — | — | — | — | — | — | — | 17.602 | — | f32 | 1.56e-2 |
| cpu | lstm_step | float16 | — | — | — | — | — | — | — | — | — | 25.798 | — | f32 | 2.04e-4 |
| cpu | gru_step | float16 | — | — | — | — | — | — | — | — | — | 12.649 | — | f32 | 2.44e-4 |
| cpu | lstm_step_bwd | float16 | — | — | — | — | — | — | — | — | — | 48.882 | — | f32 | 2.43e-4 |
| cpu | gru_step_bwd | float16 | — | — | — | — | — | — | — | — | — | 44.667 | — | f32 | 2.41e-4 |
| cpu | lstm_seq | float16 | — | — | — | — | — | — | — | — | — | 39.145 | — | f32 | 1.7e-4 |
| cpu | lstm_seq_bwd | float16 | — | — | — | — | — | — | — | — | — | 122.971 | — | f32 | 2.39e-4 |
| cpu | gru_seq | float16 | — | — | — | — | — | — | — | — | — | 29.503 | — | f32 | 2.44e-4 |
| cpu | gru_seq_bwd | float16 | — | — | — | — | — | — | — | — | — | 112.951 | — | f32 | 2.44e-4 |
| cpu | topk_rows | float16 | — | — | — | — | — | — | — | — | — | 81.487 | — | f32 | 0e+0 |
| cpu | pick_rows | float16 | — | — | — | — | — | — | — | — | — | 0.087 | — | f32 | 0e+0 |
| cpu | sample_rows | float16 | — | — | — | — | — | — | — | — | — | 6.307 | — | f32 | 2e+0 |
| cpu | cross_entropy | float16 | — | — | — | — | — | — | — | — | — | 18.073 | — | f32 (log_softmax, pick, neg) | 7.8e-3 |
| cpu | ctc | float16 | — | — | — | — | — | — | — | — | — | 2.034 | — | f32 | 4.96e-1 |
| cpu | ctc_bwd | float16 | — | — | — | — | — | — | — | — | — | 4.332 | — | f32 | 1.6e-3 |
| cpu | softmax | float16 | 4096x4096 | 44.974 | — | to f32 and back | 9.54e-7 | 58.020 | 0.78 | scalar | 1.57e-4 | 2.219 | 20.27 | f32 | 9.54e-7 |
| cpu | rms_norm | float16 | 4096x4096 | 44.632 | — | to f32 and back | 4.93e-4 | 25.095 | 1.78 | scalar | 7.06e-2 | 1.231 | 36.25 | f32 | 4.89e-4 |
| cpu | add | bfloat16 | 16777216 | 1.137 | — | typed tile | 3.91e-3 | — | — | — | — | 0.821 | 1.39 | f32 | 3.91e-3 |
| cpu | matmul | bfloat16 | 1024x1024x1024 | 4.560 | — | to f32 and back | 1.25e-1 | — | — | — | — | 1.176 | 3.88 | f32 | 1.25e-1 |
| cpu | map:copy | bfloat16 | — | — | — | — | — | — | — | — | — | 0.390 | — | f32 | 0e+0 |
| cpu | map:fill | bfloat16 | — | — | — | — | — | — | — | — | — | 0.198 | — | f32 | 0e+0 |
| cpu | map:neg | bfloat16 | — | — | — | — | — | — | — | — | — | 0.401 | — | f32 | 0e+0 |
| cpu | map:abs | bfloat16 | — | — | — | — | — | — | — | — | — | 0.403 | — | f32 | 0e+0 |
| cpu | map:sign | bfloat16 | — | — | — | — | — | — | — | — | — | 0.483 | — | f32 | 0e+0 |
| cpu | map:square | bfloat16 | — | — | — | — | — | — | — | — | — | 0.402 | — | f32 | 3.1e-2 |
| cpu | map:relu | bfloat16 | — | — | — | — | — | — | — | — | — | 0.418 | — | f32 | 0e+0 |
| cpu | map:add_scalar | bfloat16 | — | — | — | — | — | — | — | — | — | 0.403 | — | f32 | 1.56e-2 |
| cpu | map:scale | bfloat16 | — | — | — | — | — | — | — | — | — | 0.401 | — | f32 | 3.13e-2 |
| cpu | map:affine | bfloat16 | — | — | — | — | — | — | — | — | — | 0.407 | — | f32 | 3.13e-2 |
| cpu | map:clamp | bfloat16 | — | — | — | — | — | — | — | — | — | 0.588 | — | f32 | 0e+0 |
| cpu | map:add | bfloat16 | — | — | — | — | — | — | — | — | — | 0.821 | — | f32 | 1.56e-2 |
| cpu | map:sub | bfloat16 | — | — | — | — | — | — | — | — | — | 0.817 | — | f32 | 1.56e-2 |
| cpu | map:mul | bfloat16 | — | — | — | — | — | — | — | — | — | 0.823 | — | f32 | 3.13e-2 |
| cpu | map:div | bfloat16 | — | — | — | — | — | — | — | — | — | 0.864 | — | f32 | 7.78e-3 |
| cpu | map:max | bfloat16 | — | — | — | — | — | — | — | — | — | 0.836 | — | f32 | 0e+0 |
| cpu | map:min | bfloat16 | — | — | — | — | — | — | — | — | — | 0.846 | — | f32 | 0e+0 |
| cpu | map:axpy | bfloat16 | — | — | — | — | — | — | — | — | — | 1.187 | — | f32 | 3.13e-2 |
| cpu | map:muladd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.379 | — | f32 | 6.25e-2 |
| cpu | map:where | bfloat16 | — | — | — | — | — | — | — | — | — | 1.409 | — | f32 | 0e+0 |
| cpu | map:square_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.810 | — | f32 | 6.25e-2 |
| cpu | map:abs_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.829 | — | f32 | 0e+0 |
| cpu | map:relu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.807 | — | f32 | 0e+0 |
| cpu | map:clamp_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.823 | — | f32 | 0e+0 |
| cpu | map:where_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.812 | — | f32 | 0e+0 |
| cpu | map:sqrt | bfloat16 | — | — | — | — | — | — | — | — | — | 0.429 | — | f32 | 3.9e-3 |
| cpu | map:inv_sqrt | bfloat16 | — | — | — | — | — | — | — | — | — | 0.555 | — | f32 | 3.9e-3 |
| cpu | map:exp | bfloat16 | — | — | — | — | — | — | — | — | — | 2.253 | — | f32 | 1.21e-1 |
| cpu | map:expm1 | bfloat16 | — | — | — | — | — | — | — | — | — | 1.609 | — | f32 | 1.95e-3 |
| cpu | map:log | bfloat16 | — | — | — | — | — | — | — | — | — | 1.407 | — | f32 | 3.84e-3 |
| cpu | map:sin | bfloat16 | — | — | — | — | — | — | — | — | — | 2.129 | — | f32 | 1.95e-3 |
| cpu | map:cos | bfloat16 | — | — | — | — | — | — | — | — | — | 3.007 | — | f32 | 1.95e-3 |
| cpu | map:tanh | bfloat16 | — | — | — | — | — | — | — | — | — | 3.240 | — | f32 | 1.94e-3 |
| cpu | map:sigmoid | bfloat16 | — | — | — | — | — | — | — | — | — | 2.640 | — | f32 | 1.95e-3 |
| cpu | map:erf | bfloat16 | — | — | — | — | — | — | — | — | — | 22.000 | — | f32 | 1.95e-3 |
| cpu | map:gelu | bfloat16 | — | — | — | — | — | — | — | — | — | 2.720 | — | f32 | 7.75e-3 |
| cpu | map:gelu_erf | bfloat16 | — | — | — | — | — | — | — | — | — | 37.271 | — | f32 | 7.77e-3 |
| cpu | map:silu | bfloat16 | — | — | — | — | — | — | — | — | — | 2.412 | — | f32 | 7.79e-3 |
| cpu | map:leaky_relu | bfloat16 | — | — | — | — | — | — | — | — | — | 0.741 | — | f32 | 7.81e-4 |
| cpu | map:pow | bfloat16 | — | — | — | — | — | — | — | — | — | 5.795 | — | f32 | 6.17e-2 |
| cpu | map:dropout | bfloat16 | — | — | — | — | — | — | — | — | — | 1.198 | — | f32 | 1.56e-2 |
| cpu | map:gated_silu | bfloat16 | — | — | — | — | — | — | — | — | — | 2.738 | — | f32 | 3.12e-2 |
| cpu | map:gated_gelu | bfloat16 | — | — | — | — | — | — | — | — | — | 3.014 | — | f32 | 3.12e-2 |
| cpu | map:sqrt_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.515 | — | f32 | 7.81e-3 |
| cpu | map:inv_sqrt_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.108 | — | f32 | 1.56e-2 |
| cpu | map:exp_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.688 | — | f32 | 5e-1 |
| cpu | map:log_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.853 | — | f32 | 1.56e-2 |
| cpu | map:sin_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 2.226 | — | f32 | 7.81e-3 |
| cpu | map:cos_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 3.502 | — | f32 | 7.81e-3 |
| cpu | map:tanh_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 4.668 | — | f32 | 7.81e-3 |
| cpu | map:sigmoid_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 4.627 | — | f32 | 1.95e-3 |
| cpu | map:silu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 4.743 | — | f32 | 1.56e-2 |
| cpu | map:gelu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 5.060 | — | f32 | 1.56e-2 |
| cpu | map:gelu_erf_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 43.443 | — | f32 | 1.56e-2 |
| cpu | map:leaky_relu_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.197 | — | f32 | 7.81e-4 |
| cpu | map:pow_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 4.012 | — | f32 | 2.5e-1 |
| cpu | map:gated_silu_bwd_x | bfloat16 | — | — | — | — | — | — | — | — | — | 3.350 | — | f32 | 6.23e-2 |
| cpu | map:gated_silu_bwd_y | bfloat16 | — | — | — | — | — | — | — | — | — | 2.073 | — | f32 | 3.12e-2 |
| cpu | map:gated_gelu_bwd_x | bfloat16 | — | — | — | — | — | — | — | — | — | 3.563 | — | f32 | 3.12e-2 |
| cpu | map:gated_gelu_bwd_y | bfloat16 | — | — | — | — | — | — | — | — | — | 2.236 | — | f32 | 3.12e-2 |
| cpu | sum:whole | bfloat16 | — | — | — | — | — | — | — | — | — | 0.178 | — | f32 | 1.18e-4 |
| cpu | sum:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.180 | — | f32 | 8.23e-6 |
| cpu | sum:cols | bfloat16 | — | — | — | — | — | — | — | — | — | 0.908 | — | f32 | 1.86e-5 |
| cpu | sumsq:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.178 | — | f32 | 3.08e-4 |
| cpu | max:whole | bfloat16 | — | — | — | — | — | — | — | — | — | 0.443 | — | f32 | 0e+0 |
| cpu | max:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.712 | — | f32 | 0e+0 |
| cpu | max:cols | bfloat16 | — | — | — | — | — | — | — | — | — | 3.278 | — | f32 | 0e+0 |
| cpu | min:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.781 | — | f32 | 0e+0 |
| cpu | argmax:rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.823 | — | s32 index | 0e+0 |
| cpu | argmax:cols | bfloat16 | — | — | — | — | — | — | — | — | — | 1.308 | — | s32 index | 0e+0 |
| cpu | maprow:add | bfloat16 | — | — | — | — | — | — | — | — | — | 0.873 | — | f32 | 3.91e-3 |
| cpu | maprow:mul | bfloat16 | — | — | — | — | — | — | — | — | — | 0.851 | — | f32 | 1.95e-3 |
| cpu | transpose | bfloat16 | — | — | — | — | — | — | — | — | — | 1.324 | — | f32 | 0e+0 |
| cpu | gemv | bfloat16 | — | — | — | — | — | — | — | — | — | 1.478 | — | f32 | 2.5e-1 |
| cpu | gemv_t | bfloat16 | — | — | — | — | — | — | — | — | — | 6.166 | — | f32 | 2.49e-1 |
| cpu | row:log_softmax | bfloat16 | — | — | — | — | — | — | — | — | — | 1.644 | — | f32 | 3.13e-2 |
| cpu | row:softmax_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.906 | — | f32 | 2.5e-1 |
| cpu | row:log_softmax_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.664 | — | f32 | 2.5e-1 |
| cpu | row:rms_norm_affine | bfloat16 | — | — | — | — | — | — | — | — | — | 0.470 | — | f32 | 7.81e-3 |
| cpu | row:rms_norm_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 0.956 | — | f32 | 1.95e-3 |
| cpu | row:layer_norm | bfloat16 | — | — | — | — | — | — | — | — | — | 0.526 | — | f32 | 1.56e-2 |
| cpu | row:layer_norm_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.012 | — | f32 | 1.95e-3 |
| cpu | row:weight_norm | bfloat16 | — | — | — | — | — | — | — | — | — | 0.430 | — | f32 | 1.22e-4 |
| cpu | row:rope | bfloat16 | — | — | — | — | — | — | — | — | — | 3.054 | — | f32 | 1.61e-2 |
| cpu | row:scale_mask_softmax | bfloat16 | — | — | — | — | — | — | — | — | — | 2.443 | — | f32 | 1.34e-3 |
| cpu | row:add_rms_norm | bfloat16 | — | — | — | — | — | — | — | — | — | 1.757 | — | f32 | 1.56e-2 |
| cpu | row:add_layer_norm | bfloat16 | — | — | — | — | — | — | — | — | — | 1.806 | — | f32 | 1.56e-2 |
| cpu | row:cross_entropy_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1.630 | — | f32 | 1.95e-3 |
| cpu | norm_params_bwd:rms | bfloat16 | — | — | — | — | — | — | — | — | — | 3.450 | — | f32 | 2.62e-4 |
| cpu | norm_params_bwd:layer | bfloat16 | — | — | — | — | — | — | — | — | — | 2.308 | — | f32 | 3.76e-4 |
| cpu | layout:strided | bfloat16 | — | — | — | — | — | — | — | — | — | 8.221 | — | f32 | 0e+0 |
| cpu | layout:one_hot | bfloat16 | — | — | — | — | — | — | — | — | — | 0.197 | — | f32 | 0e+0 |
| cpu | layout:gather_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.338 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | bfloat16 | — | — | — | — | — | — | — | — | — | 3.195 | — | f32 | 0e+0 |
| cpu | layout:scatter_add_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 2.789 | — | f32 | 1.56e-2 |
| cpu | layout:im2col | bfloat16 | — | — | — | — | — | — | — | — | — | 1.684 | — | f32 | 0e+0 |
| cpu | layout:col2im | bfloat16 | — | — | — | — | — | — | — | — | — | 3.527 | — | f32 | 1.56e-2 |
| cpu | layout:interpolate_nearest | bfloat16 | — | — | — | — | — | — | — | — | — | 5.352 | — | f32 | 0e+0 |
| cpu | layout:interpolate_bilinear | bfloat16 | — | — | — | — | — | — | — | — | — | 18.040 | — | f32 | 1.95e-3 |
| cpu | layout:shift | bfloat16 | — | — | — | — | — | — | — | — | — | 3.750 | — | f32 | 0e+0 |
| cpu | layout:pad2d | bfloat16 | — | — | — | — | — | — | — | — | — | 0.371 | — | f32 | 0e+0 |
| cpu | layout:maxpool | bfloat16 | — | — | — | — | — | — | — | — | — | 1.970 | — | f32 | 0e+0 |
| cpu | layout:avgpool | bfloat16 | — | — | — | — | — | — | — | — | — | 1.296 | — | f32 | 1.95e-3 |
| cpu | layout:maxpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 16.960 | — | f32 | 0e+0 |
| cpu | layout:avgpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 4.135 | — | f32 | 0e+0 |
| cpu | layout:adaptive_maxpool | bfloat16 | — | — | — | — | — | — | — | — | — | 22.525 | — | f32 | 0e+0 |
| cpu | layout:adaptive_avgpool | bfloat16 | — | — | — | — | — | — | — | — | — | 19.019 | — | f32 | 1.22e-4 |
| cpu | layout:depthwise_conv | bfloat16 | — | — | — | — | — | — | — | — | — | 11.805 | — | f32 | 1.55e-2 |
| cpu | layout:adaptive_maxpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 1313.007 | — | f32 | 3.91e-3 |
| cpu | layout:adaptive_avgpool_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 10.031 | — | f32 | 7.52e-6 |
| cpu | matmul_nt | bfloat16 | — | — | — | — | — | — | — | — | — | 1.665 | — | f32 | 1.25e-1 |
| cpu | matmul_tn | bfloat16 | — | — | — | — | — | — | — | — | — | 1.814 | — | f32 | 1.25e-1 |
| cpu | matmul_tt | bfloat16 | — | — | — | — | — | — | — | — | — | 1.841 | — | f32 | 1.25e-1 |
| cpu | matmul_batched | bfloat16 | — | — | — | — | — | — | — | — | — | 0.542 | — | f32 | 6.25e-2 |
| cpu | matmul_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 3.596 | — | f32 | 1.25e-1 |
| cpu | attention | bfloat16 | — | — | — | — | — | — | — | — | — | 52.704 | — | f32 | 1.42e-4 |
| cpu | attention_causal | bfloat16 | — | — | — | — | — | — | — | — | — | 43.340 | — | f32 | 1.95e-3 |
| cpu | attention_fused | bfloat16 | — | — | — | — | — | — | — | — | — | 80.966 | — | f32 | 1.42e-4 |
| cpu | attention_fused_causal | bfloat16 | — | — | — | — | — | — | — | — | — | 44.384 | — | f32 | 1.95e-3 |
| cpu | attention_decode | bfloat16 | — | — | — | — | — | — | — | — | — | 0.845 | — | f32 | 1.15e-4 |
| cpu | attention_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 20.219 | — | f32 | 9.66e-4 |
| cpu | optimizer:sgd_momentum | bfloat16 | — | — | — | — | — | — | — | — | — | 6.860 | — | f32; + restore of p, m, v | 1.46e-3 |
| cpu | optimizer:adam | bfloat16 | — | — | — | — | — | — | — | — | — | 11.875 | — | f32; + restore of p, m, v | 3.68e-3 |
| cpu | optimizer:adamw | bfloat16 | — | — | — | — | — | — | — | — | — | 11.443 | — | f32; + restore of p, m, v | 3.69e-3 |
| cpu | conv | bfloat16 | — | — | — | — | — | — | — | — | — | 3.053 | — | f32 | 1.24e-1 |
| cpu | conv_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 20.683 | — | f32 | 1.25e-1 |
| cpu | lstm_step | bfloat16 | — | — | — | — | — | — | — | — | — | 19.539 | — | f32 | 1.92e-3 |
| cpu | gru_step | bfloat16 | — | — | — | — | — | — | — | — | — | 12.352 | — | f32 | 1.95e-3 |
| cpu | lstm_step_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 54.399 | — | f32 | 1.94e-3 |
| cpu | gru_step_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 45.092 | — | f32 | 1.81e-3 |
| cpu | lstm_seq | bfloat16 | — | — | — | — | — | — | — | — | — | 32.280 | — | f32 | 1.46e-3 |
| cpu | lstm_seq_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 139.338 | — | f32 | 1.75e-3 |
| cpu | gru_seq | bfloat16 | — | — | — | — | — | — | — | — | — | 22.748 | — | f32 | 1.95e-3 |
| cpu | gru_seq_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 92.641 | — | f32 | 1.95e-3 |
| cpu | topk_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 80.310 | — | f32 | 0e+0 |
| cpu | pick_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 0.033 | — | f32 | 0e+0 |
| cpu | sample_rows | bfloat16 | — | — | — | — | — | — | — | — | — | 4.779 | — | f32 | 2e+0 |
| cpu | cross_entropy | bfloat16 | — | — | — | — | — | — | — | — | — | 15.717 | — | f32 (log_softmax, pick, neg) | 6.19e-2 |
| cpu | ctc | bfloat16 | — | — | — | — | — | — | — | — | — | 1.911 | — | f32 | 3.95e+0 |
| cpu | ctc_bwd | bfloat16 | — | — | — | — | — | — | — | — | — | 3.846 | — | f32 | 2.93e-3 |
| cpu | softmax | bfloat16 | 4096x4096 | 23.581 | — | to f32 and back | 7.63e-6 | — | — | — | — | 1.768 | 13.34 | f32 | 7.63e-6 |
| cpu | rms_norm | bfloat16 | 4096x4096 | 30.452 | — | to f32 and back | 3.91e-3 | — | — | — | — | 0.404 | 75.32 | f32 | 3.91e-3 |
| cpu | add | float8_e4m3 | 16777216 | 28.600 | — | typed tile | 6.25e-2 | — | — | — | — | 1.968 | 14.53 | f32 | 6.25e-2 |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | 35.141 | — | to f32 and back | 2e+0 | — | — | — | — | 1.753 | 20.05 | f32 | 2e+0 |
| cpu | map:copy | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.644 | — | f32 | 0e+0 |
| cpu | map:fill | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.200 | — | f32 | 0e+0 |
| cpu | map:neg | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.680 | — | f32 | 0e+0 |
| cpu | map:abs | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.669 | — | f32 | 0e+0 |
| cpu | map:sign | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.818 | — | f32 | 0e+0 |
| cpu | map:square | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.685 | — | f32 | 4.38e-1 |
| cpu | map:relu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.717 | — | f32 | 0e+0 |
| cpu | map:add_scalar | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.665 | — | f32 | 2.5e-1 |
| cpu | map:scale | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.668 | — | f32 | 5e-1 |
| cpu | map:affine | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.395 | — | f32 | 5e-1 |
| cpu | map:clamp | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.582 | — | f32 | 0e+0 |
| cpu | map:add | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.818 | — | f32 | 2.5e-1 |
| cpu | map:sub | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.150 | — | f32 | 2.5e-1 |
| cpu | map:mul | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.824 | — | f32 | 5e-1 |
| cpu | map:div | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.985 | — | f32 | 1.17e-1 |
| cpu | map:max | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.876 | — | f32 | 0e+0 |
| cpu | map:min | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.889 | — | f32 | 0e+0 |
| cpu | map:axpy | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.824 | — | f32 | 5e-1 |
| cpu | map:muladd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.207 | — | f32 | 1e+0 |
| cpu | map:where | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.245 | — | f32 | 0e+0 |
| cpu | map:square_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.833 | — | f32 | 1e+0 |
| cpu | map:abs_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.872 | — | f32 | 0e+0 |
| cpu | map:relu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.804 | — | f32 | 0e+0 |
| cpu | map:clamp_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.200 | — | f32 | 0e+0 |
| cpu | map:where_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.181 | — | f32 | 0e+0 |
| cpu | map:sqrt | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.712 | — | f32 | 6.15e-2 |
| cpu | map:inv_sqrt | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.215 | — | f32 | 5.6e-2 |
| cpu | map:exp | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.365 | — | f32 | 1.48e+0 |
| cpu | map:expm1 | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.771 | — | f32 | 2.7e-2 |
| cpu | map:log | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.665 | — | f32 | 5.37e-2 |
| cpu | map:sin | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.603 | — | f32 | 3.05e-2 |
| cpu | map:cos | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.226 | — | f32 | 3.11e-2 |
| cpu | map:tanh | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.382 | — | f32 | 3.01e-2 |
| cpu | map:sigmoid | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.729 | — | f32 | 3.12e-2 |
| cpu | map:erf | float8_e4m3 | — | — | — | — | — | — | — | — | — | 14.075 | — | f32 | 3.02e-2 |
| cpu | map:gelu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.924 | — | f32 | 5.7e-2 |
| cpu | map:gelu_erf | float8_e4m3 | — | — | — | — | — | — | — | — | — | 24.009 | — | f32 | 5.7e-2 |
| cpu | map:silu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.740 | — | f32 | 1.21e-1 |
| cpu | map:leaky_relu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.649 | — | f32 | 1.25e-2 |
| cpu | map:pow | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.690 | — | f32 | 9.58e-1 |
| cpu | map:dropout | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.846 | — | f32 | 1.88e-1 |
| cpu | map:gated_silu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.249 | — | f32 | 4.94e-1 |
| cpu | map:gated_gelu | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.389 | — | f32 | 4.98e-1 |
| cpu | map:sqrt_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.258 | — | f32 | 1.22e-1 |
| cpu | map:inv_sqrt_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.354 | — | f32 | 2.43e-1 |
| cpu | map:exp_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.900 | — | f32 | 7.5e+0 |
| cpu | map:log_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.985 | — | f32 | 2.33e-1 |
| cpu | map:sin_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.708 | — | f32 | 1.25e-1 |
| cpu | map:cos_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.891 | — | f32 | 1.24e-1 |
| cpu | map:tanh_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.765 | — | f32 | 1.22e-1 |
| cpu | map:sigmoid_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 6.157 | — | f32 | 3.06e-2 |
| cpu | map:silu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 6.206 | — | f32 | 2.45e-1 |
| cpu | map:gelu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 6.515 | — | f32 | 2.4e-1 |
| cpu | map:gelu_erf_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 45.437 | — | f32 | 2.37e-1 |
| cpu | map:leaky_relu_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.877 | — | f32 | 1.25e-2 |
| cpu | map:pow_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 7.716 | — | f32 | 3.92e+0 |
| cpu | map:gated_silu_bwd_x | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.532 | — | f32 | 9.79e-1 |
| cpu | map:gated_silu_bwd_y | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.588 | — | f32 | 4.94e-1 |
| cpu | map:gated_gelu_bwd_x | float8_e4m3 | — | — | — | — | — | — | — | — | — | 7.037 | — | f32 | 4.97e-1 |
| cpu | map:gated_gelu_bwd_y | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.850 | — | f32 | 4.98e-1 |
| cpu | sum:whole | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.378 | — | f32 | 0e+0 |
| cpu | sum:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.376 | — | f32 | 0e+0 |
| cpu | sum:cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.854 | — | f32 | 0e+0 |
| cpu | sumsq:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.380 | — | f32 | 1.6e-4 |
| cpu | max:whole | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.596 | — | f32 | 0e+0 |
| cpu | max:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.587 | — | f32 | 0e+0 |
| cpu | max:cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.840 | — | f32 | 0e+0 |
| cpu | min:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.592 | — | f32 | 0e+0 |
| cpu | argmax:rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.904 | — | s32 index | 0e+0 |
| cpu | argmax:cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.847 | — | s32 index | 0e+0 |
| cpu | maprow:add | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.658 | — | f32 | 6.25e-2 |
| cpu | maprow:mul | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.659 | — | f32 | 3.13e-2 |
| cpu | transpose | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.454 | — | f32 | 0e+0 |
| cpu | gemv | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.760 | — | f32 | 3.99e+0 |
| cpu | gemv_t | float8_e4m3 | — | — | — | — | — | — | — | — | — | 5.411 | — | f32 | 7.52e+0 |
| cpu | row:log_softmax | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.528 | — | f32 | 5e-1 |
| cpu | row:softmax_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.245 | — | f32 | 3.99e+0 |
| cpu | row:log_softmax_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.119 | — | f32 | 4e+0 |
| cpu | row:rms_norm_affine | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.671 | — | f32 | 1.25e-1 |
| cpu | row:rms_norm_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.965 | — | f32 | 3.12e-2 |
| cpu | row:layer_norm | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.741 | — | f32 | 2.5e-1 |
| cpu | row:layer_norm_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.076 | — | f32 | 3.13e-2 |
| cpu | row:weight_norm | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.666 | — | f32 | 1.95e-3 |
| cpu | row:rope | float8_e4m3 | — | — | — | — | — | — | — | — | — | 5.545 | — | f32 | 2.5e-1 |
| cpu | row:scale_mask_softmax | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.357 | — | f32 | 2.93e-2 |
| cpu | row:add_rms_norm | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.338 | — | f32 | 2.5e-1 |
| cpu | row:add_layer_norm | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.384 | — | f32 | 2.5e-1 |
| cpu | row:cross_entropy_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.909 | — | f32 | 2.07e-3 |
| cpu | norm_params_bwd:rms | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.458 | — | f32 | 3.55e-4 |
| cpu | norm_params_bwd:layer | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.530 | — | f32 | 3.51e-4 |
| cpu | layout:strided | float8_e4m3 | — | — | — | — | — | — | — | — | — | 23.516 | — | f32 | 0e+0 |
| cpu | layout:one_hot | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.199 | — | f32 | 0e+0 |
| cpu | layout:gather_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.552 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | float8_e4m3 | — | — | — | — | — | — | — | — | — | 22.627 | — | f32 | 0e+0 |
| cpu | layout:scatter_add_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 22.802 | — | f32 | 2.5e-1 |
| cpu | layout:im2col | float8_e4m3 | — | — | — | — | — | — | — | — | — | 15.961 | — | f32 | 0e+0 |
| cpu | layout:col2im | float8_e4m3 | — | — | — | — | — | — | — | — | — | 14.043 | — | f32 | 2.5e-1 |
| cpu | layout:interpolate_nearest | float8_e4m3 | — | — | — | — | — | — | — | — | — | 30.431 | — | f32 | 0e+0 |
| cpu | layout:interpolate_bilinear | float8_e4m3 | — | — | — | — | — | — | — | — | — | 121.080 | — | f32 | 3.13e-2 |
| cpu | layout:shift | float8_e4m3 | — | — | — | — | — | — | — | — | — | 26.917 | — | f32 | 0e+0 |
| cpu | layout:pad2d | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.554 | — | f32 | 0e+0 |
| cpu | layout:maxpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 7.756 | — | f32 | 0e+0 |
| cpu | layout:avgpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 7.376 | — | f32 | 3.13e-2 |
| cpu | layout:maxpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 41.494 | — | f32 | 0e+0 |
| cpu | layout:avgpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 7.704 | — | f32 | 9.77e-4 |
| cpu | layout:adaptive_maxpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 115.765 | — | f32 | 0e+0 |
| cpu | layout:adaptive_avgpool | float8_e4m3 | — | — | — | — | — | — | — | — | — | 112.363 | — | f32 | 1.95e-3 |
| cpu | layout:depthwise_conv | float8_e4m3 | — | — | — | — | — | — | — | — | — | 91.236 | — | f32 | 2.5e-1 |
| cpu | layout:adaptive_maxpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 10102.242 | — | f32 | 6.25e-2 |
| cpu | layout:adaptive_avgpool_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 15.524 | — | f32 | 9.76e-4 |
| cpu | matmul_nt | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.882 | — | f32 | 2e+0 |
| cpu | matmul_tn | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.047 | — | f32 | 2e+0 |
| cpu | matmul_tt | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.149 | — | f32 | 2e+0 |
| cpu | matmul_batched | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.788 | — | f32 | 9.99e-1 |
| cpu | matmul_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.076 | — | f32 | 2e+0 |
| cpu | attention | float8_e4m3 | — | — | — | — | — | — | — | — | — | 55.461 | — | f32 | 1.95e-3 |
| cpu | attention_causal | float8_e4m3 | — | — | — | — | — | — | — | — | — | 43.789 | — | f32 | 3.12e-2 |
| cpu | attention_fused | float8_e4m3 | — | — | — | — | — | — | — | — | — | 81.278 | — | f32 | 1.95e-3 |
| cpu | attention_fused_causal | float8_e4m3 | — | — | — | — | — | — | — | — | — | 44.011 | — | f32 | 3.12e-2 |
| cpu | attention_decode | float8_e4m3 | — | — | — | — | — | — | — | — | — | 1.586 | — | f32 | 1.93e-3 |
| cpu | attention_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 23.113 | — | f32 | 1.53e-2 |
| cpu | optimizer:sgd_momentum | float8_e4m3 | — | — | — | — | — | — | — | — | — | 6.213 | — | f32; + restore of p, m, v | 1.46e-3 |
| cpu | optimizer:adam | float8_e4m3 | — | — | — | — | — | — | — | — | — | 10.387 | — | f32; + restore of p, m, v | 1.81e-2 |
| cpu | optimizer:adamw | float8_e4m3 | — | — | — | — | — | — | — | — | — | 10.489 | — | f32; + restore of p, m, v | 1.81e-2 |
| cpu | conv | float8_e4m3 | — | — | — | — | — | — | — | — | — | 3.399 | — | f32 | 2e+0 |
| cpu | conv_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 17.999 | — | f32 | 2e+0 |
| cpu | lstm_step | float8_e4m3 | — | — | — | — | — | — | — | — | — | 25.359 | — | f32 | 3.08e-2 |
| cpu | gru_step | float8_e4m3 | — | — | — | — | — | — | — | — | — | 12.216 | — | f32 | 3.12e-2 |
| cpu | lstm_step_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 52.703 | — | f32 | 3.09e-2 |
| cpu | gru_step_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 50.530 | — | f32 | 3.05e-2 |
| cpu | lstm_seq | float8_e4m3 | — | — | — | — | — | — | — | — | — | 38.168 | — | f32 | 1.77e-2 |
| cpu | lstm_seq_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 120.524 | — | f32 | 2.45e-2 |
| cpu | gru_seq | float8_e4m3 | — | — | — | — | — | — | — | — | — | 21.722 | — | f32 | 3.12e-2 |
| cpu | gru_seq_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 121.726 | — | f32 | 3.11e-2 |
| cpu | topk_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 89.190 | — | f32 | 0e+0 |
| cpu | pick_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 0.135 | — | f32 | 0e+0 |
| cpu | sample_rows | float8_e4m3 | — | — | — | — | — | — | — | — | — | 6.393 | — | f32 | 1e+0 |
| cpu | cross_entropy | float8_e4m3 | — | — | — | — | — | — | — | — | — | 28.917 | — | f32 (log_softmax, pick, neg) | 5e-1 |
| cpu | ctc | float8_e4m3 | — | — | — | — | — | — | — | — | — | 2.098 | — | f32 | Infinity |
| cpu | ctc_bwd | float8_e4m3 | — | — | — | — | — | — | — | — | — | 4.475 | — | f32 | 3.22e-2 |
| cpu | softmax | float8_e4m3 | 4096x4096 | 90.060 | — | to f32 and back | 9.76e-4 | — | — | — | — | 2.565 | 35.12 | f32 | 9.76e-4 |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | 429.597 | — | to f32 and back | 6.25e-2 | — | — | — | — | 1.342 | 320.05 | f32 | 6.25e-2 |
| cpu | add | float8_e5m2 | 16777216 | 17.993 | — | typed tile | 1.25e-1 | — | — | — | — | 1.972 | 9.13 | f32 | 1.25e-1 |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | 31.074 | — | to f32 and back | 4e+0 | — | — | — | — | 1.781 | 17.45 | f32 | 4e+0 |
| cpu | map:copy | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.512 | — | f32 | 0e+0 |
| cpu | map:fill | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.342 | — | f32 | 0e+0 |
| cpu | map:neg | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.877 | — | f32 | 0e+0 |
| cpu | map:abs | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.862 | — | f32 | 0e+0 |
| cpu | map:sign | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.000 | — | f32 | 0e+0 |
| cpu | map:square | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.855 | — | f32 | 1e+0 |
| cpu | map:relu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.900 | — | f32 | 0e+0 |
| cpu | map:add_scalar | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.874 | — | f32 | 5e-1 |
| cpu | map:scale | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.848 | — | f32 | 1e+0 |
| cpu | map:affine | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.847 | — | f32 | 1e+0 |
| cpu | map:clamp | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.190 | — | f32 | 0e+0 |
| cpu | map:add | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.013 | — | f32 | 5e-1 |
| cpu | map:sub | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.395 | — | f32 | 5e-1 |
| cpu | map:mul | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.007 | — | f32 | 1e+0 |
| cpu | map:div | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.166 | — | f32 | 2.14e-1 |
| cpu | map:max | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.085 | — | f32 | 0e+0 |
| cpu | map:min | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.081 | — | f32 | 0e+0 |
| cpu | map:axpy | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.027 | — | f32 | 1e+0 |
| cpu | map:muladd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.467 | — | f32 | 2e+0 |
| cpu | map:where | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.491 | — | f32 | 0e+0 |
| cpu | map:square_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.380 | — | f32 | 2e+0 |
| cpu | map:abs_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.507 | — | f32 | 0e+0 |
| cpu | map:relu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.432 | — | f32 | 0e+0 |
| cpu | map:clamp_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.431 | — | f32 | 0e+0 |
| cpu | map:where_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.029 | — | f32 | 0e+0 |
| cpu | map:sqrt | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.777 | — | f32 | 1.21e-1 |
| cpu | map:inv_sqrt | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.989 | — | f32 | 9.53e-2 |
| cpu | map:exp | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.726 | — | f32 | 1.4e+0 |
| cpu | map:expm1 | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.881 | — | f32 | 4.88e-2 |
| cpu | map:log | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.785 | — | f32 | 1.14e-1 |
| cpu | map:sin | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.752 | — | f32 | 5.66e-2 |
| cpu | map:cos | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.335 | — | f32 | 6.15e-2 |
| cpu | map:tanh | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.382 | — | f32 | 5.86e-2 |
| cpu | map:sigmoid | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.826 | — | f32 | 6.22e-2 |
| cpu | map:erf | float8_e5m2 | — | — | — | — | — | — | — | — | — | 14.446 | — | f32 | 4.79e-2 |
| cpu | map:gelu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.021 | — | f32 | 1.18e-1 |
| cpu | map:gelu_erf | float8_e5m2 | — | — | — | — | — | — | — | — | — | 27.505 | — | f32 | 1.18e-1 |
| cpu | map:silu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.852 | — | f32 | 1.9e-1 |
| cpu | map:leaky_relu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.745 | — | f32 | 2.5e-2 |
| cpu | map:pow | float8_e5m2 | — | — | — | — | — | — | — | — | — | 5.041 | — | f32 | 1.08e+0 |
| cpu | map:dropout | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 3.75e-1 |
| cpu | map:gated_silu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.141 | — | f32 | 7.59e-1 |
| cpu | map:gated_gelu | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.695 | — | f32 | 9.89e-1 |
| cpu | map:sqrt_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.456 | — | f32 | 2.14e-1 |
| cpu | map:inv_sqrt_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.712 | — | f32 | 3.43e-1 |
| cpu | map:exp_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.172 | — | f32 | 8.5e+0 |
| cpu | map:log_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.602 | — | f32 | 4.29e-1 |
| cpu | map:sin_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.949 | — | f32 | 2.46e-1 |
| cpu | map:cos_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.978 | — | f32 | 2.39e-1 |
| cpu | map:tanh_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 6.392 | — | f32 | 2.47e-1 |
| cpu | map:sigmoid_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.285 | — | f32 | 6.19e-2 |
| cpu | map:silu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.445 | — | f32 | 3.98e-1 |
| cpu | map:gelu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.732 | — | f32 | 4.9e-1 |
| cpu | map:gelu_erf_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 30.145 | — | f32 | 4.91e-1 |
| cpu | map:leaky_relu_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.443 | — | f32 | 2.5e-2 |
| cpu | map:pow_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 5.407 | — | f32 | 6e+0 |
| cpu | map:gated_silu_bwd_x | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.720 | — | f32 | 1.12e+0 |
| cpu | map:gated_silu_bwd_y | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.457 | — | f32 | 7.59e-1 |
| cpu | map:gated_gelu_bwd_x | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.943 | — | f32 | 9.93e-1 |
| cpu | map:gated_gelu_bwd_y | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.723 | — | f32 | 9.89e-1 |
| cpu | sum:whole | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.622 | — | f32 | 3.05e-4 |
| cpu | sum:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.729 | — | f32 | 0e+0 |
| cpu | sum:cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.347 | — | f32 | 0e+0 |
| cpu | sumsq:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.732 | — | f32 | 1.88e-4 |
| cpu | max:whole | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.133 | — | f32 | 0e+0 |
| cpu | max:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.750 | — | f32 | 0e+0 |
| cpu | max:cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.218 | — | f32 | 0e+0 |
| cpu | min:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.740 | — | f32 | 0e+0 |
| cpu | argmax:rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.726 | — | s32 index | 0e+0 |
| cpu | argmax:cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.058 | — | s32 index | 0e+0 |
| cpu | maprow:add | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.876 | — | f32 | 1.25e-1 |
| cpu | maprow:mul | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.852 | — | f32 | 6.25e-2 |
| cpu | transpose | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.461 | — | f32 | 0e+0 |
| cpu | gemv | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.890 | — | f32 | 7.95e+0 |
| cpu | gemv_t | float8_e5m2 | — | — | — | — | — | — | — | — | — | 5.705 | — | f32 | 1.15e+1 |
| cpu | row:log_softmax | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.097 | — | f32 | 1e+0 |
| cpu | row:softmax_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.100 | — | f32 | 7.49e+0 |
| cpu | row:log_softmax_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.383 | — | f32 | 8e+0 |
| cpu | row:rms_norm_affine | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.769 | — | f32 | 2.5e-1 |
| cpu | row:rms_norm_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.194 | — | f32 | 6.25e-2 |
| cpu | row:layer_norm | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.832 | — | f32 | 5e-1 |
| cpu | row:layer_norm_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.304 | — | f32 | 6.25e-2 |
| cpu | row:weight_norm | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.860 | — | f32 | 3.91e-3 |
| cpu | row:rope | float8_e5m2 | — | — | — | — | — | — | — | — | — | 5.739 | — | f32 | 5e-1 |
| cpu | row:scale_mask_softmax | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.622 | — | f32 | 5.42e-2 |
| cpu | row:add_rms_norm | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.743 | — | f32 | 5e-1 |
| cpu | row:add_layer_norm | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.798 | — | f32 | 5e-1 |
| cpu | row:cross_entropy_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.707 | — | f32 | 2.05e-3 |
| cpu | norm_params_bwd:rms | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.073 | — | f32 | 2.41e-4 |
| cpu | norm_params_bwd:layer | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.550 | — | f32 | 3.2e-4 |
| cpu | layout:strided | float8_e5m2 | — | — | — | — | — | — | — | — | — | 25.488 | — | f32 | 0e+0 |
| cpu | layout:one_hot | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.318 | — | f32 | 0e+0 |
| cpu | layout:gather_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.656 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | float8_e5m2 | — | — | — | — | — | — | — | — | — | 24.231 | — | f32 | 0e+0 |
| cpu | layout:scatter_add_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 24.328 | — | f32 | 5e-1 |
| cpu | layout:im2col | float8_e5m2 | — | — | — | — | — | — | — | — | — | 16.828 | — | f32 | 0e+0 |
| cpu | layout:col2im | float8_e5m2 | — | — | — | — | — | — | — | — | — | 18.152 | — | f32 | 5e-1 |
| cpu | layout:interpolate_nearest | float8_e5m2 | — | — | — | — | — | — | — | — | — | 31.154 | — | f32 | 0e+0 |
| cpu | layout:interpolate_bilinear | float8_e5m2 | — | — | — | — | — | — | — | — | — | 129.252 | — | f32 | 6.25e-2 |
| cpu | layout:shift | float8_e5m2 | — | — | — | — | — | — | — | — | — | 29.107 | — | f32 | 0e+0 |
| cpu | layout:pad2d | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.675 | — | f32 | 0e+0 |
| cpu | layout:maxpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 8.537 | — | f32 | 0e+0 |
| cpu | layout:avgpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 6.182 | — | f32 | 6.25e-2 |
| cpu | layout:maxpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 43.425 | — | f32 | 0e+0 |
| cpu | layout:avgpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 11.023 | — | f32 | 7.63e-6 |
| cpu | layout:adaptive_maxpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 122.159 | — | f32 | 0e+0 |
| cpu | layout:adaptive_avgpool | float8_e5m2 | — | — | — | — | — | — | — | — | — | 118.057 | — | f32 | 3.9e-3 |
| cpu | layout:depthwise_conv | float8_e5m2 | — | — | — | — | — | — | — | — | — | 105.132 | — | f32 | 4.92e-1 |
| cpu | layout:adaptive_maxpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 10570.257 | — | f32 | 1.25e-1 |
| cpu | layout:adaptive_avgpool_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 16.005 | — | f32 | 2.34e-4 |
| cpu | matmul_nt | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.044 | — | f32 | 4e+0 |
| cpu | matmul_tn | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.086 | — | f32 | 4e+0 |
| cpu | matmul_tt | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.163 | — | f32 | 4e+0 |
| cpu | matmul_batched | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.742 | — | f32 | 2e+0 |
| cpu | matmul_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.128 | — | f32 | 4e+0 |
| cpu | attention | float8_e5m2 | — | — | — | — | — | — | — | — | — | 54.473 | — | f32 | 3.91e-3 |
| cpu | attention_causal | float8_e5m2 | — | — | — | — | — | — | — | — | — | 43.520 | — | f32 | 6.25e-2 |
| cpu | attention_fused | float8_e5m2 | — | — | — | — | — | — | — | — | — | 89.692 | — | f32 | 3.91e-3 |
| cpu | attention_fused_causal | float8_e5m2 | — | — | — | — | — | — | — | — | — | 44.370 | — | f32 | 6.25e-2 |
| cpu | attention_decode | float8_e5m2 | — | — | — | — | — | — | — | — | — | 1.962 | — | f32 | 2.98e-3 |
| cpu | attention_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 21.749 | — | f32 | 3.06e-2 |
| cpu | optimizer:sgd_momentum | float8_e5m2 | — | — | — | — | — | — | — | — | — | 6.745 | — | f32; + restore of p, m, v | 1.46e-3 |
| cpu | optimizer:adam | float8_e5m2 | — | — | — | — | — | — | — | — | — | 10.456 | — | f32; + restore of p, m, v | 2.59e-2 |
| cpu | optimizer:adamw | float8_e5m2 | — | — | — | — | — | — | — | — | — | 10.594 | — | f32; + restore of p, m, v | 2.59e-2 |
| cpu | conv | float8_e5m2 | — | — | — | — | — | — | — | — | — | 3.415 | — | f32 | 3.99e+0 |
| cpu | conv_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 18.536 | — | f32 | 4e+0 |
| cpu | lstm_step | float8_e5m2 | — | — | — | — | — | — | — | — | — | 28.894 | — | f32 | 4.62e-2 |
| cpu | gru_step | float8_e5m2 | — | — | — | — | — | — | — | — | — | 15.798 | — | f32 | 6.25e-2 |
| cpu | lstm_step_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 53.860 | — | f32 | 6.16e-2 |
| cpu | gru_step_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 45.032 | — | f32 | 5.97e-2 |
| cpu | lstm_seq | float8_e5m2 | — | — | — | — | — | — | — | — | — | 32.572 | — | f32 | 5.59e-2 |
| cpu | lstm_seq_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 132.114 | — | f32 | 5.98e-2 |
| cpu | gru_seq | float8_e5m2 | — | — | — | — | — | — | — | — | — | 20.450 | — | f32 | 6.25e-2 |
| cpu | gru_seq_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 99.344 | — | f32 | 6.24e-2 |
| cpu | topk_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 102.985 | — | f32 | 0e+0 |
| cpu | pick_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 0.145 | — | f32 | 0e+0 |
| cpu | sample_rows | float8_e5m2 | — | — | — | — | — | — | — | — | — | 6.319 | — | f32 | 1e+0 |
| cpu | cross_entropy | float8_e5m2 | — | — | — | — | — | — | — | — | — | 24.899 | — | f32 (log_softmax, pick, neg) | 9.84e-1 |
| cpu | ctc | float8_e5m2 | — | — | — | — | — | — | — | — | — | 2.010 | — | f32 | 1.27e+2 |
| cpu | ctc_bwd | float8_e5m2 | — | — | — | — | — | — | — | — | — | 4.412 | — | f32 | 6.31e-2 |
| cpu | softmax | float8_e5m2 | 4096x4096 | 279.161 | — | to f32 and back | 1.62e-4 | — | — | — | — | 3.231 | 86.41 | f32 | 1.62e-4 |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | 314.909 | — | to f32 and back | 1.08e-1 | — | — | — | — | 1.711 | 184.04 | f32 | 1.08e-1 |
| cpu | add | int64 | 16777216 | 25.096 | — | typed tile | 0e+0 | 6.621 | 3.79 | Vector<Int64> | 0e+0 | 6.691 | 3.75 | s64 | 0e+0 |
| cpu | matmul | int64 | 1024x1024x1024 | 6.394 | — | to f32 and back | 0e+0 | 32.382 | 0.20 | Vector<Int64> | 0e+0 | 0.601 | 10.65 | s8dp | 0e+0 |
| cpu | map:copy | int64 | — | — | — | — | — | — | — | — | — | 4.739 | — | s64 | 0e+0 |
| cpu | map:fill | int64 | — | — | — | — | — | — | — | — | — | 2.649 | — | s64 | 0e+0 |
| cpu | map:neg | int64 | — | — | — | — | — | — | — | — | — | 4.716 | — | s64 | 0e+0 |
| cpu | map:abs | int64 | — | — | — | — | — | — | — | — | — | 4.698 | — | s64 | 0e+0 |
| cpu | map:sign | int64 | — | — | — | — | — | — | — | — | — | 4.648 | — | s64 | 0e+0 |
| cpu | map:square | int64 | — | — | — | — | — | — | — | — | — | 4.672 | — | s64 | 0e+0 |
| cpu | map:relu | int64 | — | — | — | — | — | — | — | — | — | 4.754 | — | s64 | 0e+0 |
| cpu | map:add_scalar | int64 | — | — | — | — | — | — | — | — | — | 4.726 | — | s64 | 0e+0 |
| cpu | map:scale | int64 | — | — | — | — | — | — | — | — | — | 4.666 | — | s64 | 0e+0 |
| cpu | map:affine | int64 | — | — | — | — | — | — | — | — | — | 5.041 | — | s64 | 0e+0 |
| cpu | map:clamp | int64 | — | — | — | — | — | — | — | — | — | 4.661 | — | s64 | 0e+0 |
| cpu | map:add | int64 | — | — | — | — | — | — | — | — | — | 6.621 | — | s64 | 0e+0 |
| cpu | map:sub | int64 | — | — | — | — | — | — | — | — | — | 6.706 | — | s64 | 0e+0 |
| cpu | map:mul | int64 | — | — | — | — | — | — | — | — | — | 6.659 | — | s64 | 0e+0 |
| cpu | map:div | int64 | — | — | — | — | — | — | — | — | — | 6.721 | — | s64 | 0e+0 |
| cpu | map:max | int64 | — | — | — | — | — | — | — | — | — | 6.695 | — | s64 | 0e+0 |
| cpu | map:min | int64 | — | — | — | — | — | — | — | — | — | 6.855 | — | s64 | 0e+0 |
| cpu | map:axpy | int64 | — | — | — | — | — | — | — | — | — | 6.707 | — | s64 | 0e+0 |
| cpu | map:muladd | int64 | — | — | — | — | — | — | — | — | — | 8.689 | — | s64 | 0e+0 |
| cpu | map:where | int64 | — | — | — | — | — | — | — | — | — | 9.305 | — | s64 | 0e+0 |
| cpu | map:square_bwd | int64 | — | — | — | — | — | — | — | — | — | 6.832 | — | s64 | 0e+0 |
| cpu | map:abs_bwd | int64 | — | — | — | — | — | — | — | — | — | 6.883 | — | s64 | 0e+0 |
| cpu | map:relu_bwd | int64 | — | — | — | — | — | — | — | — | — | 6.823 | — | s64 | 0e+0 |
| cpu | map:clamp_bwd | int64 | — | — | — | — | — | — | — | — | — | 6.910 | — | s64 | 0e+0 |
| cpu | map:where_bwd | int64 | — | — | — | — | — | — | — | — | — | 6.864 | — | s64 | 0e+0 |
| cpu | sum:whole | int64 | — | — | — | — | — | — | — | — | — | 2.551 | — | s8dp | 0e+0 |
| cpu | sum:rows | int64 | — | — | — | — | — | — | — | — | — | 2.557 | — | s8dp | 0e+0 |
| cpu | sum:cols | int64 | — | — | — | — | — | — | — | — | — | 4.281 | — | s8dp | 0e+0 |
| cpu | sumsq:rows | int64 | — | — | — | — | — | — | — | — | — | 2.606 | — | s8dp | 0e+0 |
| cpu | max:whole | int64 | — | — | — | — | — | — | — | — | — | 1.269 | — | s64 | 0e+0 |
| cpu | max:rows | int64 | — | — | — | — | — | — | — | — | — | 1.290 | — | s64 | 0e+0 |
| cpu | max:cols | int64 | — | — | — | — | — | — | — | — | — | 6.777 | — | s64 | 0e+0 |
| cpu | min:rows | int64 | — | — | — | — | — | — | — | — | — | 1.418 | — | s64 | 0e+0 |
| cpu | argmax:rows | int64 | — | — | — | — | — | — | — | — | — | 1.356 | — | s32 index | 0e+0 |
| cpu | argmax:cols | int64 | — | — | — | — | — | — | — | — | — | 4.994 | — | s32 index | 0e+0 |
| cpu | maprow:add | int64 | — | — | — | — | — | — | — | — | — | 4.707 | — | s64 | 0e+0 |
| cpu | maprow:mul | int64 | — | — | — | — | — | — | — | — | — | 4.702 | — | s64 | 0e+0 |
| cpu | transpose | int64 | — | — | — | — | — | — | — | — | — | 6.263 | — | s64 | 0e+0 |
| cpu | gemv | int64 | — | — | — | — | — | — | — | — | — | 12.635 | — | s8dp | 0e+0 |
| cpu | gemv_t | int64 | — | — | — | — | — | — | — | — | — | 16.259 | — | s8dp | 0e+0 |
| cpu | layout:strided | int64 | — | — | — | — | — | — | — | — | — | 20.259 | — | s64 | 0e+0 |
| cpu | layout:one_hot | int64 | — | — | — | — | — | — | — | — | — | 2.465 | — | s64 | 0e+0 |
| cpu | layout:gather_rows | int64 | — | — | — | — | — | — | — | — | — | 4.185 | — | s64 | 0e+0 |
| cpu | layout:gather_cols | int64 | — | — | — | — | — | — | — | — | — | 5.517 | — | s64 | 0e+0 |
| cpu | layout:scatter_add_rows | int64 | — | — | — | — | — | — | — | — | — | 7.256 | — | s32 | 0e+0 |
| cpu | layout:im2col | int64 | — | — | — | — | — | — | — | — | — | 1.578 | — | s64 | 0e+0 |
| cpu | layout:col2im | int64 | — | — | — | — | — | — | — | — | — | 3.489 | — | s32 | 0e+0 |
| cpu | layout:interpolate_nearest | int64 | — | — | — | — | — | — | — | — | — | 3.429 | — | s64 | 0e+0 |
| cpu | layout:shift | int64 | — | — | — | — | — | — | — | — | — | 4.623 | — | s64 | 0e+0 |
| cpu | layout:pad2d | int64 | — | — | — | — | — | — | — | — | — | 3.821 | — | s64 | 0e+0 |
| cpu | layout:maxpool | int64 | — | — | — | — | — | — | — | — | — | 1.991 | — | s64 | 0e+0 |
| cpu | layout:maxpool_bwd | int64 | — | — | — | — | — | — | — | — | — | 11.345 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool | int64 | — | — | — | — | — | — | — | — | — | 14.725 | — | s64 | 0e+0 |
| cpu | layout:depthwise_conv | int64 | — | — | — | — | — | — | — | — | — | 35.110 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | int64 | — | — | — | — | — | — | — | — | — | 2889.760 | — | s32 | 0e+0 |
| cpu | matmul_nt | int64 | — | — | — | — | — | — | — | — | — | 0.832 | — | s8dp | 0e+0 |
| cpu | matmul_tn | int64 | — | — | — | — | — | — | — | — | — | 0.971 | — | s8dp | 0e+0 |
| cpu | matmul_tt | int64 | — | — | — | — | — | — | — | — | — | 1.065 | — | s8dp | 0e+0 |
| cpu | matmul_batched | int64 | — | — | — | — | — | — | — | — | — | 1.203 | — | s8dp | 0e+0 |
| cpu | matmul_bwd | int64 | — | — | — | — | — | — | — | — | — | 1.905 | — | s8dp | 0e+0 |
| cpu | conv | int64 | — | — | — | — | — | — | — | — | — | 1.346 | — | s8dp | 0e+0 |
| cpu | topk_rows | int64 | — | — | — | — | — | — | — | — | — | 302.096 | — | s64 | 0e+0 |
| cpu | pick_rows | int64 | — | — | — | — | — | — | — | — | — | 0.028 | — | s64 | 0e+0 |
| cpu | add | int32 | 16777216 | 20.738 | — | typed tile | 0e+0 | 3.034 | 6.84 | Vector<Int32> | 0e+0 | 2.913 | 7.12 | s32 | 0e+0 |
| cpu | matmul | int32 | 1024x1024x1024 | 6.005 | — | to f32 and back | 0e+0 | 5.932 | 1.01 | Vector<Int32> | 0e+0 | 0.573 | 10.49 | s8dp | 0e+0 |
| cpu | map:copy | int32 | — | — | — | — | — | — | — | — | — | 1.689 | — | s32 | 0e+0 |
| cpu | map:fill | int32 | — | — | — | — | — | — | — | — | — | 0.360 | — | s32 | 0e+0 |
| cpu | map:neg | int32 | — | — | — | — | — | — | — | — | — | 1.796 | — | s32 | 0e+0 |
| cpu | map:abs | int32 | — | — | — | — | — | — | — | — | — | 1.636 | — | s32 | 0e+0 |
| cpu | map:sign | int32 | — | — | — | — | — | — | — | — | — | 1.756 | — | s32 | 0e+0 |
| cpu | map:square | int32 | — | — | — | — | — | — | — | — | — | 1.948 | — | s32 | 0e+0 |
| cpu | map:relu | int32 | — | — | — | — | — | — | — | — | — | 1.716 | — | s32 | 0e+0 |
| cpu | map:add_scalar | int32 | — | — | — | — | — | — | — | — | — | 1.891 | — | s32 | 0e+0 |
| cpu | map:scale | int32 | — | — | — | — | — | — | — | — | — | 1.747 | — | s32 | 0e+0 |
| cpu | map:affine | int32 | — | — | — | — | — | — | — | — | — | 1.910 | — | s32 | 0e+0 |
| cpu | map:clamp | int32 | — | — | — | — | — | — | — | — | — | 1.809 | — | s32 | 0e+0 |
| cpu | map:add | int32 | — | — | — | — | — | — | — | — | — | 2.993 | — | s32 | 0e+0 |
| cpu | map:sub | int32 | — | — | — | — | — | — | — | — | — | 2.991 | — | s32 | 0e+0 |
| cpu | map:mul | int32 | — | — | — | — | — | — | — | — | — | 3.001 | — | s32 | 0e+0 |
| cpu | map:div | int32 | — | — | — | — | — | — | — | — | — | 2.984 | — | s32 | 0e+0 |
| cpu | map:max | int32 | — | — | — | — | — | — | — | — | — | 3.102 | — | s32 | 0e+0 |
| cpu | map:min | int32 | — | — | — | — | — | — | — | — | — | 3.032 | — | s32 | 0e+0 |
| cpu | map:axpy | int32 | — | — | — | — | — | — | — | — | — | 2.998 | — | s32 | 0e+0 |
| cpu | map:muladd | int32 | — | — | — | — | — | — | — | — | — | 4.097 | — | s32 | 0e+0 |
| cpu | map:where | int32 | — | — | — | — | — | — | — | — | — | 4.110 | — | s32 | 0e+0 |
| cpu | map:square_bwd | int32 | — | — | — | — | — | — | — | — | — | 2.884 | — | s32 | 0e+0 |
| cpu | map:abs_bwd | int32 | — | — | — | — | — | — | — | — | — | 2.891 | — | s32 | 0e+0 |
| cpu | map:relu_bwd | int32 | — | — | — | — | — | — | — | — | — | 2.961 | — | s32 | 0e+0 |
| cpu | map:clamp_bwd | int32 | — | — | — | — | — | — | — | — | — | 2.910 | — | s32 | 0e+0 |
| cpu | map:where_bwd | int32 | — | — | — | — | — | — | — | — | — | 2.875 | — | s32 | 0e+0 |
| cpu | sum:whole | int32 | — | — | — | — | — | — | — | — | — | 0.764 | — | s8dp | 0e+0 |
| cpu | sum:rows | int32 | — | — | — | — | — | — | — | — | — | 0.778 | — | s8dp | 0e+0 |
| cpu | sum:cols | int32 | — | — | — | — | — | — | — | — | — | 1.713 | — | s8dp | 0e+0 |
| cpu | sumsq:rows | int32 | — | — | — | — | — | — | — | — | — | 0.796 | — | s8dp | 0e+0 |
| cpu | max:whole | int32 | — | — | — | — | — | — | — | — | — | 0.377 | — | s32 | 0e+0 |
| cpu | max:rows | int32 | — | — | — | — | — | — | — | — | — | 0.391 | — | s32 | 0e+0 |
| cpu | max:cols | int32 | — | — | — | — | — | — | — | — | — | 2.329 | — | s32 | 0e+0 |
| cpu | min:rows | int32 | — | — | — | — | — | — | — | — | — | 0.384 | — | s32 | 0e+0 |
| cpu | argmax:rows | int32 | — | — | — | — | — | — | — | — | — | 0.378 | — | s32 index | 0e+0 |
| cpu | argmax:cols | int32 | — | — | — | — | — | — | — | — | — | 5.220 | — | s32 index | 0e+0 |
| cpu | maprow:add | int32 | — | — | — | — | — | — | — | — | — | 1.745 | — | s32 | 0e+0 |
| cpu | maprow:mul | int32 | — | — | — | — | — | — | — | — | — | 1.727 | — | s32 | 0e+0 |
| cpu | transpose | int32 | — | — | — | — | — | — | — | — | — | 2.925 | — | s32 | 0e+0 |
| cpu | gemv | int32 | — | — | — | — | — | — | — | — | — | 5.842 | — | s8dp | 0e+0 |
| cpu | gemv_t | int32 | — | — | — | — | — | — | — | — | — | 8.337 | — | s8dp | 0e+0 |
| cpu | layout:strided | int32 | — | — | — | — | — | — | — | — | — | 14.035 | — | s32 | 0e+0 |
| cpu | layout:one_hot | int32 | — | — | — | — | — | — | — | — | — | 0.523 | — | s32 | 0e+0 |
| cpu | layout:gather_rows | int32 | — | — | — | — | — | — | — | — | — | 1.446 | — | s32 | 0e+0 |
| cpu | layout:gather_cols | int32 | — | — | — | — | — | — | — | — | — | 2.230 | — | s32 | 0e+0 |
| cpu | layout:scatter_add_rows | int32 | — | — | — | — | — | — | — | — | — | 2.613 | — | s32 | 0e+0 |
| cpu | layout:im2col | int32 | — | — | — | — | — | — | — | — | — | 1.040 | — | s32 | 0e+0 |
| cpu | layout:col2im | int32 | — | — | — | — | — | — | — | — | — | 2.502 | — | s32 | 0e+0 |
| cpu | layout:interpolate_nearest | int32 | — | — | — | — | — | — | — | — | — | 2.284 | — | s32 | 0e+0 |
| cpu | layout:shift | int32 | — | — | — | — | — | — | — | — | — | 2.216 | — | s32 | 0e+0 |
| cpu | layout:pad2d | int32 | — | — | — | — | — | — | — | — | — | 1.951 | — | s32 | 0e+0 |
| cpu | layout:maxpool | int32 | — | — | — | — | — | — | — | — | — | 1.943 | — | s32 | 0e+0 |
| cpu | layout:maxpool_bwd | int32 | — | — | — | — | — | — | — | — | — | 7.700 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool | int32 | — | — | — | — | — | — | — | — | — | 14.880 | — | s32 | 0e+0 |
| cpu | layout:depthwise_conv | int32 | — | — | — | — | — | — | — | — | — | 5.207 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | int32 | — | — | — | — | — | — | — | — | — | 1399.303 | — | s32 | 0e+0 |
| cpu | matmul_nt | int32 | — | — | — | — | — | — | — | — | — | 0.642 | — | s8dp | 0e+0 |
| cpu | matmul_tn | int32 | — | — | — | — | — | — | — | — | — | 0.820 | — | s8dp | 0e+0 |
| cpu | matmul_tt | int32 | — | — | — | — | — | — | — | — | — | 0.943 | — | s8dp | 0e+0 |
| cpu | matmul_batched | int32 | — | — | — | — | — | — | — | — | — | 0.539 | — | s8dp | 0e+0 |
| cpu | matmul_bwd | int32 | — | — | — | — | — | — | — | — | — | 1.515 | — | s8dp | 0e+0 |
| cpu | conv | int32 | — | — | — | — | — | — | — | — | — | 0.940 | — | s8dp | 0e+0 |
| cpu | topk_rows | int32 | — | — | — | — | — | — | — | — | — | 304.829 | — | s32 | 0e+0 |
| cpu | pick_rows | int32 | — | — | — | — | — | — | — | — | — | 0.027 | — | s32 | 0e+0 |
| cpu | add | int16 | 16777216 | 1.397 | — | typed tile | 0e+0 | 1.047 | 1.33 | Vector<Int16> | 0e+0 | 0.831 | 1.68 | s16 | 0e+0 |
| cpu | matmul | int16 | 1024x1024x1024 | 5.175 | — | to f32 and back | 0e+0 | 1.845 | 2.81 | Vector<Int16> | 0e+0 | 0.408 | 12.67 | s8dp | 0e+0 |
| cpu | map:copy | int16 | — | — | — | — | — | — | — | — | — | 0.271 | — | s16 | 0e+0 |
| cpu | map:fill | int16 | — | — | — | — | — | — | — | — | — | 0.129 | — | s16 | 0e+0 |
| cpu | map:neg | int16 | — | — | — | — | — | — | — | — | — | 0.277 | — | s16 | 0e+0 |
| cpu | map:abs | int16 | — | — | — | — | — | — | — | — | — | 0.315 | — | s16 | 0e+0 |
| cpu | map:sign | int16 | — | — | — | — | — | — | — | — | — | 0.326 | — | s16 | 0e+0 |
| cpu | map:square | int16 | — | — | — | — | — | — | — | — | — | 0.267 | — | s16 | 0e+0 |
| cpu | map:relu | int16 | — | — | — | — | — | — | — | — | — | 0.259 | — | s16 | 0e+0 |
| cpu | map:add_scalar | int16 | — | — | — | — | — | — | — | — | — | 0.322 | — | s16 | 0e+0 |
| cpu | map:scale | int16 | — | — | — | — | — | — | — | — | — | 0.281 | — | s16 | 0e+0 |
| cpu | map:affine | int16 | — | — | — | — | — | — | — | — | — | 0.276 | — | s16 | 0e+0 |
| cpu | map:clamp | int16 | — | — | — | — | — | — | — | — | — | 0.347 | — | s16 | 0e+0 |
| cpu | map:add | int16 | — | — | — | — | — | — | — | — | — | 0.790 | — | s16 | 0e+0 |
| cpu | map:sub | int16 | — | — | — | — | — | — | — | — | — | 0.874 | — | s16 | 0e+0 |
| cpu | map:mul | int16 | — | — | — | — | — | — | — | — | — | 0.793 | — | s16 | 0e+0 |
| cpu | map:div | int16 | — | — | — | — | — | — | — | — | — | 1.419 | — | s16 | 0e+0 |
| cpu | map:max | int16 | — | — | — | — | — | — | — | — | — | 0.789 | — | s16 | 0e+0 |
| cpu | map:min | int16 | — | — | — | — | — | — | — | — | — | 0.850 | — | s16 | 0e+0 |
| cpu | map:axpy | int16 | — | — | — | — | — | — | — | — | — | 0.783 | — | s16 | 0e+0 |
| cpu | map:muladd | int16 | — | — | — | — | — | — | — | — | — | 1.390 | — | s16 | 0e+0 |
| cpu | map:where | int16 | — | — | — | — | — | — | — | — | — | 1.321 | — | s16 | 0e+0 |
| cpu | map:square_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.803 | — | s16 | 0e+0 |
| cpu | map:abs_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.817 | — | s16 | 0e+0 |
| cpu | map:relu_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.789 | — | s16 | 0e+0 |
| cpu | map:clamp_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.793 | — | s16 | 0e+0 |
| cpu | map:where_bwd | int16 | — | — | — | — | — | — | — | — | — | 0.922 | — | s16 | 0e+0 |
| cpu | sum:whole | int16 | — | — | — | — | — | — | — | — | — | 0.188 | — | s8dp | 0e+0 |
| cpu | sum:rows | int16 | — | — | — | — | — | — | — | — | — | 0.193 | — | s8dp | 0e+0 |
| cpu | sum:cols | int16 | — | — | — | — | — | — | — | — | — | 0.687 | — | s8dp | 0e+0 |
| cpu | sumsq:rows | int16 | — | — | — | — | — | — | — | — | — | 0.183 | — | s8dp | 0e+0 |
| cpu | max:whole | int16 | — | — | — | — | — | — | — | — | — | 0.094 | — | s16 | 0e+0 |
| cpu | max:rows | int16 | — | — | — | — | — | — | — | — | — | 0.103 | — | s16 | 0e+0 |
| cpu | max:cols | int16 | — | — | — | — | — | — | — | — | — | 1.898 | — | s16 | 0e+0 |
| cpu | min:rows | int16 | — | — | — | — | — | — | — | — | — | 0.128 | — | s16 | 0e+0 |
| cpu | argmax:rows | int16 | — | — | — | — | — | — | — | — | — | 0.105 | — | s32 index | 0e+0 |
| cpu | argmax:cols | int16 | — | — | — | — | — | — | — | — | — | 4.699 | — | s32 index | 0e+0 |
| cpu | maprow:add | int16 | — | — | — | — | — | — | — | — | — | 0.292 | — | s16 | 0e+0 |
| cpu | maprow:mul | int16 | — | — | — | — | — | — | — | — | — | 0.306 | — | s16 | 0e+0 |
| cpu | transpose | int16 | — | — | — | — | — | — | — | — | — | 0.966 | — | s16 | 0e+0 |
| cpu | gemv | int16 | — | — | — | — | — | — | — | — | — | 2.519 | — | s8dp | 0e+0 |
| cpu | gemv_t | int16 | — | — | — | — | — | — | — | — | — | 5.225 | — | s8dp | 0e+0 |
| cpu | layout:strided | int16 | — | — | — | — | — | — | — | — | — | 6.018 | — | s16 | 0e+0 |
| cpu | layout:one_hot | int16 | — | — | — | — | — | — | — | — | — | 0.118 | — | s16 | 0e+0 |
| cpu | layout:gather_rows | int16 | — | — | — | — | — | — | — | — | — | 0.335 | — | s16 | 0e+0 |
| cpu | layout:gather_cols | int16 | — | — | — | — | — | — | — | — | — | 0.935 | — | s16 | 0e+0 |
| cpu | layout:scatter_add_rows | int16 | — | — | — | — | — | — | — | — | — | 3.591 | — | s32 | 0e+0 |
| cpu | layout:im2col | int16 | — | — | — | — | — | — | — | — | — | 0.990 | — | s16 | 0e+0 |
| cpu | layout:col2im | int16 | — | — | — | — | — | — | — | — | — | 3.078 | — | s32 | 0e+0 |
| cpu | layout:interpolate_nearest | int16 | — | — | — | — | — | — | — | — | — | 1.829 | — | s16 | 0e+0 |
| cpu | layout:shift | int16 | — | — | — | — | — | — | — | — | — | 1.049 | — | s16 | 0e+0 |
| cpu | layout:pad2d | int16 | — | — | — | — | — | — | — | — | — | 0.647 | — | s16 | 0e+0 |
| cpu | layout:maxpool | int16 | — | — | — | — | — | — | — | — | — | 1.617 | — | s16 | 0e+0 |
| cpu | layout:maxpool_bwd | int16 | — | — | — | — | — | — | — | — | — | 11.946 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool | int16 | — | — | — | — | — | — | — | — | — | 15.253 | — | s16 | 0e+0 |
| cpu | layout:depthwise_conv | int16 | — | — | — | — | — | — | — | — | — | 18.564 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | int16 | — | — | — | — | — | — | — | — | — | 3718.135 | — | s32 | 0e+0 |
| cpu | matmul_nt | int16 | — | — | — | — | — | — | — | — | — | 0.570 | — | s8dp | 0e+0 |
| cpu | matmul_tn | int16 | — | — | — | — | — | — | — | — | — | 0.836 | — | s8dp | 0e+0 |
| cpu | matmul_tt | int16 | — | — | — | — | — | — | — | — | — | 0.963 | — | s8dp | 0e+0 |
| cpu | matmul_batched | int16 | — | — | — | — | — | — | — | — | — | 0.452 | — | s8dp | 0e+0 |
| cpu | matmul_bwd | int16 | — | — | — | — | — | — | — | — | — | 1.513 | — | s8dp | 0e+0 |
| cpu | conv | int16 | — | — | — | — | — | — | — | — | — | 0.980 | — | s8dp | 0e+0 |
| cpu | topk_rows | int16 | — | — | — | — | — | — | — | — | — | 313.804 | — | s16 | 0e+0 |
| cpu | pick_rows | int16 | — | — | — | — | — | — | — | — | — | 0.025 | — | s16 | 0e+0 |
| cpu | add | int8 | 16777216 | 1.543 | — | typed tile | 0e+0 | 0.420 | 3.68 | Vector<SByte> | 0e+0 | 0.144 | 10.70 | s8 | 0e+0 |
| cpu | matmul | int8 | 1024x1024x1024 | 5.905 | — | to f32 and back | 1.76e+2 | 2.205 | 2.68 | Vector<SByte> | 2.56e+2 | 0.393 | 15.02 | s8dp | 1.76e+2 |
| cpu | map:copy | int8 | — | — | — | — | — | — | — | — | — | 0.078 | — | s8 | 0e+0 |
| cpu | map:fill | int8 | — | — | — | — | — | — | — | — | — | 0.082 | — | s8 | 0e+0 |
| cpu | map:neg | int8 | — | — | — | — | — | — | — | — | — | 0.126 | — | s8 | 0e+0 |
| cpu | map:abs | int8 | — | — | — | — | — | — | — | — | — | 0.079 | — | s8 | 0e+0 |
| cpu | map:sign | int8 | — | — | — | — | — | — | — | — | — | 0.080 | — | s8 | 0e+0 |
| cpu | map:square | int8 | — | — | — | — | — | — | — | — | — | 0.078 | — | s8 | 0e+0 |
| cpu | map:relu | int8 | — | — | — | — | — | — | — | — | — | 0.118 | — | s8 | 0e+0 |
| cpu | map:add_scalar | int8 | — | — | — | — | — | — | — | — | — | 0.080 | — | s8 | 0e+0 |
| cpu | map:scale | int8 | — | — | — | — | — | — | — | — | — | 0.080 | — | s8 | 0e+0 |
| cpu | map:affine | int8 | — | — | — | — | — | — | — | — | — | 0.079 | — | s8 | 0e+0 |
| cpu | map:clamp | int8 | — | — | — | — | — | — | — | — | — | 0.122 | — | s8 | 0e+0 |
| cpu | map:add | int8 | — | — | — | — | — | — | — | — | — | 0.212 | — | s8 | 0e+0 |
| cpu | map:sub | int8 | — | — | — | — | — | — | — | — | — | 0.200 | — | s8 | 0e+0 |
| cpu | map:mul | int8 | — | — | — | — | — | — | — | — | — | 0.159 | — | s8 | 0e+0 |
| cpu | map:div | int8 | — | — | — | — | — | — | — | — | — | 1.441 | — | s8 | 0e+0 |
| cpu | map:max | int8 | — | — | — | — | — | — | — | — | — | 0.176 | — | s8 | 0e+0 |
| cpu | map:min | int8 | — | — | — | — | — | — | — | — | — | 0.178 | — | s8 | 0e+0 |
| cpu | map:axpy | int8 | — | — | — | — | — | — | — | — | — | 0.196 | — | s8 | 0e+0 |
| cpu | map:muladd | int8 | — | — | — | — | — | — | — | — | — | 0.307 | — | s8 | 0e+0 |
| cpu | map:where | int8 | — | — | — | — | — | — | — | — | — | 0.297 | — | s8 | 0e+0 |
| cpu | map:square_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.218 | — | s8 | 0e+0 |
| cpu | map:abs_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.196 | — | s8 | 0e+0 |
| cpu | map:relu_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.176 | — | s8 | 0e+0 |
| cpu | map:clamp_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.191 | — | s8 | 0e+0 |
| cpu | map:where_bwd | int8 | — | — | — | — | — | — | — | — | — | 0.170 | — | s8 | 0e+0 |
| cpu | sum:whole | int8 | — | — | — | — | — | — | — | — | — | 0.104 | — | s8dp | 0e+0 |
| cpu | sum:rows | int8 | — | — | — | — | — | — | — | — | — | 0.103 | — | s8dp | 0e+0 |
| cpu | sum:cols | int8 | — | — | — | — | — | — | — | — | — | 0.361 | — | s8dp | 0e+0 |
| cpu | sumsq:rows | int8 | — | — | — | — | — | — | — | — | — | 0.075 | — | s8dp | 0e+0 |
| cpu | max:whole | int8 | — | — | — | — | — | — | — | — | — | 0.041 | — | s8 | 0e+0 |
| cpu | max:rows | int8 | — | — | — | — | — | — | — | — | — | 0.068 | — | s8 | 0e+0 |
| cpu | max:cols | int8 | — | — | — | — | — | — | — | — | — | 1.170 | — | s8 | 0e+0 |
| cpu | min:rows | int8 | — | — | — | — | — | — | — | — | — | 0.075 | — | s8 | 0e+0 |
| cpu | argmax:rows | int8 | — | — | — | — | — | — | — | — | — | 0.057 | — | s32 index | 0e+0 |
| cpu | argmax:cols | int8 | — | — | — | — | — | — | — | — | — | 4.389 | — | s32 index | 0e+0 |
| cpu | maprow:add | int8 | — | — | — | — | — | — | — | — | — | 0.081 | — | s8 | 0e+0 |
| cpu | maprow:mul | int8 | — | — | — | — | — | — | — | — | — | 0.084 | — | s8 | 0e+0 |
| cpu | transpose | int8 | — | — | — | — | — | — | — | — | — | 0.560 | — | s8 | 0e+0 |
| cpu | gemv | int8 | — | — | — | — | — | — | — | — | — | 0.769 | — | s8dp | 5.72e+2 |
| cpu | gemv_t | int8 | — | — | — | — | — | — | — | — | — | 2.401 | — | s8dp | 6.32e+2 |
| cpu | layout:strided | int8 | — | — | — | — | — | — | — | — | — | 3.672 | — | s8 | 0e+0 |
| cpu | layout:one_hot | int8 | — | — | — | — | — | — | — | — | — | 0.095 | — | s8 | 0e+0 |
| cpu | layout:gather_rows | int8 | — | — | — | — | — | — | — | — | — | 0.099 | — | s8 | 0e+0 |
| cpu | layout:gather_cols | int8 | — | — | — | — | — | — | — | — | — | 0.629 | — | s8 | 0e+0 |
| cpu | layout:scatter_add_rows | int8 | — | — | — | — | — | — | — | — | — | 4.337 | — | s32 | 0e+0 |
| cpu | layout:im2col | int8 | — | — | — | — | — | — | — | — | — | 0.914 | — | s8 | 0e+0 |
| cpu | layout:col2im | int8 | — | — | — | — | — | — | — | — | — | 3.630 | — | s32 | 0e+0 |
| cpu | layout:interpolate_nearest | int8 | — | — | — | — | — | — | — | — | — | 1.702 | — | s8 | 0e+0 |
| cpu | layout:shift | int8 | — | — | — | — | — | — | — | — | — | 0.860 | — | s8 | 0e+0 |
| cpu | layout:pad2d | int8 | — | — | — | — | — | — | — | — | — | 0.122 | — | s8 | 0e+0 |
| cpu | layout:maxpool | int8 | — | — | — | — | — | — | — | — | — | 1.982 | — | s8 | 0e+0 |
| cpu | layout:maxpool_bwd | int8 | — | — | — | — | — | — | — | — | — | 10.045 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool | int8 | — | — | — | — | — | — | — | — | — | 14.455 | — | s8 | 0e+0 |
| cpu | layout:depthwise_conv | int8 | — | — | — | — | — | — | — | — | — | 22.576 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | int8 | — | — | — | — | — | — | — | — | — | 2658.250 | — | s32 | 0e+0 |
| cpu | matmul_nt | int8 | — | — | — | — | — | — | — | — | — | 0.526 | — | s8dp | 1.88e+2 |
| cpu | matmul_tn | int8 | — | — | — | — | — | — | — | — | — | 0.426 | — | s8dp | 1.98e+2 |
| cpu | matmul_tt | int8 | — | — | — | — | — | — | — | — | — | 0.426 | — | s8dp | 1.85e+2 |
| cpu | matmul_batched | int8 | — | — | — | — | — | — | — | — | — | 0.335 | — | s8dp | 2.6e+1 |
| cpu | matmul_bwd | int8 | — | — | — | — | — | — | — | — | — | 1.159 | — | s8dp | 2.22e+2 |
| cpu | conv | int8 | — | — | — | — | — | — | — | — | — | 0.931 | — | s8dp | 1.27e+2 |
| cpu | topk_rows | int8 | — | — | — | — | — | — | — | — | — | 313.203 | — | s8 | 0e+0 |
| cpu | pick_rows | int8 | — | — | — | — | — | — | — | — | — | 0.024 | — | s8 | 0e+0 |
| cpu | add | int4 | 16777216 | 2.872 | — | typed tile | 0e+0 | — | — | — | — | 0.107 | 26.94 | s8 | 0e+0 |
| cpu | matmul | int4 | 1024x1024x1024 | 11.391 | — | to f32 and back | 2.96e+2 | — | — | — | — | 0.351 | 32.47 | s8dp | 2.96e+2 |
| cpu | map:copy | int4 | — | — | — | — | — | — | — | — | — | 0.288 | — | s8 | 0e+0 |
| cpu | map:fill | int4 | — | — | — | — | — | — | — | — | — | 0.250 | — | s8 | 0e+0 |
| cpu | map:neg | int4 | — | — | — | — | — | — | — | — | — | 0.290 | — | s8 | 0e+0 |
| cpu | map:abs | int4 | — | — | — | — | — | — | — | — | — | 0.278 | — | s8 | 0e+0 |
| cpu | map:sign | int4 | — | — | — | — | — | — | — | — | — | 0.271 | — | s8 | 0e+0 |
| cpu | map:square | int4 | — | — | — | — | — | — | — | — | — | 0.288 | — | s8 | 0e+0 |
| cpu | map:relu | int4 | — | — | — | — | — | — | — | — | — | 0.291 | — | s8 | 0e+0 |
| cpu | map:add_scalar | int4 | — | — | — | — | — | — | — | — | — | 0.276 | — | s8 | 0e+0 |
| cpu | map:scale | int4 | — | — | — | — | — | — | — | — | — | 0.290 | — | s8 | 0e+0 |
| cpu | map:affine | int4 | — | — | — | — | — | — | — | — | — | 0.308 | — | s8 | 0e+0 |
| cpu | map:clamp | int4 | — | — | — | — | — | — | — | — | — | 0.306 | — | s8 | 0e+0 |
| cpu | map:add | int4 | — | — | — | — | — | — | — | — | — | 0.294 | — | s8 | 0e+0 |
| cpu | map:sub | int4 | — | — | — | — | — | — | — | — | — | 0.298 | — | s8 | 0e+0 |
| cpu | map:mul | int4 | — | — | — | — | — | — | — | — | — | 0.298 | — | s8 | 0e+0 |
| cpu | map:div | int4 | — | — | — | — | — | — | — | — | — | 1.324 | — | s8 | 0e+0 |
| cpu | map:max | int4 | — | — | — | — | — | — | — | — | — | 0.295 | — | s8 | 0e+0 |
| cpu | map:min | int4 | — | — | — | — | — | — | — | — | — | 0.301 | — | s8 | 0e+0 |
| cpu | map:axpy | int4 | — | — | — | — | — | — | — | — | — | 0.298 | — | s8 | 0e+0 |
| cpu | map:muladd | int4 | — | — | — | — | — | — | — | — | — | 0.341 | — | s8 | 0e+0 |
| cpu | map:where | int4 | — | — | — | — | — | — | — | — | — | 0.326 | — | s8 | 0e+0 |
| cpu | map:square_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.325 | — | s8 | 0e+0 |
| cpu | map:abs_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.310 | — | s8 | 0e+0 |
| cpu | map:relu_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.298 | — | s8 | 0e+0 |
| cpu | map:clamp_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.301 | — | s8 | 0e+0 |
| cpu | map:where_bwd | int4 | — | — | — | — | — | — | — | — | — | 0.305 | — | s8 | 0e+0 |
| cpu | sum:whole | int4 | — | — | — | — | — | — | — | — | — | 0.042 | — | s8dp | 0e+0 |
| cpu | sum:rows | int4 | — | — | — | — | — | — | — | — | — | 0.059 | — | s8dp | 0e+0 |
| cpu | sum:cols | int4 | — | — | — | — | — | — | — | — | — | 0.521 | — | s8dp | 0e+0 |
| cpu | sumsq:rows | int4 | — | — | — | — | — | — | — | — | — | 0.060 | — | s8dp | 0e+0 |
| cpu | max:whole | int4 | — | — | — | — | — | — | — | — | — | 0.103 | — | s8 | 0e+0 |
| cpu | max:rows | int4 | — | — | — | — | — | — | — | — | — | 0.143 | — | s8 | 0e+0 |
| cpu | max:cols | int4 | — | — | — | — | — | — | — | — | — | 1.806 | — | s8 | 0e+0 |
| cpu | min:rows | int4 | — | — | — | — | — | — | — | — | — | 0.113 | — | s8 | 0e+0 |
| cpu | argmax:rows | int4 | — | — | — | — | — | — | — | — | — | 0.110 | — | s32 index | 0e+0 |
| cpu | argmax:cols | int4 | — | — | — | — | — | — | — | — | — | 4.479 | — | s32 index | 0e+0 |
| cpu | maprow:add | int4 | — | — | — | — | — | — | — | — | — | 0.745 | — | s8 | 0e+0 |
| cpu | maprow:mul | int4 | — | — | — | — | — | — | — | — | — | 0.193 | — | s8 | 0e+0 |
| cpu | transpose | int4 | — | — | — | — | — | — | — | — | — | 0.910 | — | s8 | 0e+0 |
| cpu | gemv | int4 | — | — | — | — | — | — | — | — | — | 0.322 | — | s8dp | 6.92e+2 |
| cpu | gemv_t | int4 | — | — | — | — | — | — | — | — | — | 2.959 | — | s8dp | 7.52e+2 |
| cpu | layout:strided | int4 | — | — | — | — | — | — | — | — | — | 7.027 | — | s8 | 0e+0 |
| cpu | layout:one_hot | int4 | — | — | — | — | — | — | — | — | — | 0.088 | — | s8 | 0e+0 |
| cpu | layout:gather_rows | int4 | — | — | — | — | — | — | — | — | — | 0.094 | — | s8 | 0e+0 |
| cpu | layout:gather_cols | int4 | — | — | — | — | — | — | — | — | — | 12.205 | — | s8 | 0e+0 |
| cpu | layout:scatter_add_rows | int4 | — | — | — | — | — | — | — | — | — | 31.326 | — | s32 | 3e+0 |
| cpu | layout:im2col | int4 | — | — | — | — | — | — | — | — | — | 4.656 | — | s8 | 0e+0 |
| cpu | layout:col2im | int4 | — | — | — | — | — | — | — | — | — | 23.844 | — | s32 | 1.1e+1 |
| cpu | layout:interpolate_nearest | int4 | — | — | — | — | — | — | — | — | — | 14.842 | — | s8 | 0e+0 |
| cpu | layout:shift | int4 | — | — | — | — | — | — | — | — | — | 14.407 | — | s8 | 0e+0 |
| cpu | layout:pad2d | int4 | — | — | — | — | — | — | — | — | — | 0.100 | — | s8 | 0e+0 |
| cpu | layout:maxpool | int4 | — | — | — | — | — | — | — | — | — | 4.814 | — | s8 | 0e+0 |
| cpu | layout:maxpool_bwd | int4 | — | — | — | — | — | — | — | — | — | 50.713 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool | int4 | — | — | — | — | — | — | — | — | — | 41.506 | — | s8 | 0e+0 |
| cpu | layout:depthwise_conv | int4 | — | — | — | — | — | — | — | — | — | 167.001 | — | s32 | 2.2e+1 |
| cpu | layout:adaptive_maxpool_bwd | int4 | — | — | — | — | — | — | — | — | — | 13605.755 | — | s32 | 0e+0 |
| cpu | matmul_nt | int4 | — | — | — | — | — | — | — | — | — | 1.689 | — | s8dp | 3.08e+2 |
| cpu | matmul_tn | int4 | — | — | — | — | — | — | — | — | — | 1.783 | — | s8dp | 3.18e+2 |
| cpu | matmul_tt | int4 | — | — | — | — | — | — | — | — | — | 1.805 | — | s8dp | 3.05e+2 |
| cpu | matmul_batched | int4 | — | — | — | — | — | — | — | — | — | 1.694 | — | s8dp | 1.46e+2 |
| cpu | matmul_bwd | int4 | — | — | — | — | — | — | — | — | — | 3.847 | — | s8dp | 3.42e+2 |
| cpu | conv | int4 | — | — | — | — | — | — | — | — | — | 0.954 | — | s8dp | 2.47e+2 |
| cpu | topk_rows | int4 | — | — | — | — | — | — | — | — | — | 307.286 | — | s8 | 0e+0 |
| cpu | pick_rows | int4 | — | — | — | — | — | — | — | — | — | 0.032 | — | s8 | 0e+0 |
| cpu | add | int2 | 16777216 | 2.357 | — | typed tile | 0e+0 | — | — | — | — | 0.182 | 12.95 | s8 | 0e+0 |
| cpu | matmul | int2 | 1024x1024x1024 | 6.646 | — | to f32 and back | 3.26e+2 | — | — | — | — | 0.360 | 18.47 | s8dp | 3.26e+2 |
| cpu | map:copy | int2 | — | — | — | — | — | — | — | — | — | 0.309 | — | s8 | 0e+0 |
| cpu | map:fill | int2 | — | — | — | — | — | — | — | — | — | 0.290 | — | s8 | 0e+0 |
| cpu | map:neg | int2 | — | — | — | — | — | — | — | — | — | 0.310 | — | s8 | 0e+0 |
| cpu | map:abs | int2 | — | — | — | — | — | — | — | — | — | 0.309 | — | s8 | 0e+0 |
| cpu | map:sign | int2 | — | — | — | — | — | — | — | — | — | 0.310 | — | s8 | 0e+0 |
| cpu | map:square | int2 | — | — | — | — | — | — | — | — | — | 0.302 | — | s8 | 0e+0 |
| cpu | map:relu | int2 | — | — | — | — | — | — | — | — | — | 0.341 | — | s8 | 0e+0 |
| cpu | map:add_scalar | int2 | — | — | — | — | — | — | — | — | — | 0.306 | — | s8 | 0e+0 |
| cpu | map:scale | int2 | — | — | — | — | — | — | — | — | — | 0.305 | — | s8 | 0e+0 |
| cpu | map:affine | int2 | — | — | — | — | — | — | — | — | — | 0.316 | — | s8 | 0e+0 |
| cpu | map:clamp | int2 | — | — | — | — | — | — | — | — | — | 0.310 | — | s8 | 0e+0 |
| cpu | map:add | int2 | — | — | — | — | — | — | — | — | — | 0.340 | — | s8 | 0e+0 |
| cpu | map:sub | int2 | — | — | — | — | — | — | — | — | — | 0.337 | — | s8 | 0e+0 |
| cpu | map:mul | int2 | — | — | — | — | — | — | — | — | — | 0.361 | — | s8 | 0e+0 |
| cpu | map:div | int2 | — | — | — | — | — | — | — | — | — | 1.549 | — | s8 | 0e+0 |
| cpu | map:max | int2 | — | — | — | — | — | — | — | — | — | 0.468 | — | s8 | 0e+0 |
| cpu | map:min | int2 | — | — | — | — | — | — | — | — | — | 0.467 | — | s8 | 0e+0 |
| cpu | map:axpy | int2 | — | — | — | — | — | — | — | — | — | 0.361 | — | s8 | 0e+0 |
| cpu | map:muladd | int2 | — | — | — | — | — | — | — | — | — | 0.416 | — | s8 | 0e+0 |
| cpu | map:where | int2 | — | — | — | — | — | — | — | — | — | 0.391 | — | s8 | 0e+0 |
| cpu | map:square_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.407 | — | s8 | 0e+0 |
| cpu | map:abs_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.380 | — | s8 | 0e+0 |
| cpu | map:relu_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.337 | — | s8 | 0e+0 |
| cpu | map:clamp_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.338 | — | s8 | 0e+0 |
| cpu | map:where_bwd | int2 | — | — | — | — | — | — | — | — | — | 0.333 | — | s8 | 0e+0 |
| cpu | sum:whole | int2 | — | — | — | — | — | — | — | — | — | 0.067 | — | s8dp | 0e+0 |
| cpu | sum:rows | int2 | — | — | — | — | — | — | — | — | — | 0.067 | — | s8dp | 0e+0 |
| cpu | sum:cols | int2 | — | — | — | — | — | — | — | — | — | 0.592 | — | s8dp | 0e+0 |
| cpu | sumsq:rows | int2 | — | — | — | — | — | — | — | — | — | 0.069 | — | s8dp | 0e+0 |
| cpu | max:whole | int2 | — | — | — | — | — | — | — | — | — | 0.147 | — | s8 | 0e+0 |
| cpu | max:rows | int2 | — | — | — | — | — | — | — | — | — | 0.224 | — | s8 | 0e+0 |
| cpu | max:cols | int2 | — | — | — | — | — | — | — | — | — | 1.818 | — | s8 | 0e+0 |
| cpu | min:rows | int2 | — | — | — | — | — | — | — | — | — | 0.152 | — | s8 | 0e+0 |
| cpu | argmax:rows | int2 | — | — | — | — | — | — | — | — | — | 0.256 | — | s32 index | 0e+0 |
| cpu | argmax:cols | int2 | — | — | — | — | — | — | — | — | — | 7.130 | — | s32 index | 0e+0 |
| cpu | maprow:add | int2 | — | — | — | — | — | — | — | — | — | 0.263 | — | s8 | 0e+0 |
| cpu | maprow:mul | int2 | — | — | — | — | — | — | — | — | — | 0.267 | — | s8 | 0e+0 |
| cpu | transpose | int2 | — | — | — | — | — | — | — | — | — | 3.190 | — | s8 | 0e+0 |
| cpu | gemv | int2 | — | — | — | — | — | — | — | — | — | 0.552 | — | s8dp | 2.15e+3 |
| cpu | gemv_t | int2 | — | — | — | — | — | — | — | — | — | 3.892 | — | s8dp | 2.15e+3 |
| cpu | layout:strided | int2 | — | — | — | — | — | — | — | — | — | 12.733 | — | s8 | 0e+0 |
| cpu | layout:one_hot | int2 | — | — | — | — | — | — | — | — | — | 0.171 | — | s8 | 0e+0 |
| cpu | layout:gather_rows | int2 | — | — | — | — | — | — | — | — | — | 0.214 | — | s8 | 0e+0 |
| cpu | layout:gather_cols | int2 | — | — | — | — | — | — | — | — | — | 17.705 | — | s8 | 0e+0 |
| cpu | layout:scatter_add_rows | int2 | — | — | — | — | — | — | — | — | — | 29.820 | — | s32 | 3e+0 |
| cpu | layout:im2col | int2 | — | — | — | — | — | — | — | — | — | 12.634 | — | s8 | 0e+0 |
| cpu | layout:col2im | int2 | — | — | — | — | — | — | — | — | — | 24.171 | — | s32 | 7e+0 |
| cpu | layout:interpolate_nearest | int2 | — | — | — | — | — | — | — | — | — | 19.459 | — | s8 | 0e+0 |
| cpu | layout:shift | int2 | — | — | — | — | — | — | — | — | — | 18.943 | — | s8 | 0e+0 |
| cpu | layout:pad2d | int2 | — | — | — | — | — | — | — | — | — | 0.257 | — | s8 | 0e+0 |
| cpu | layout:maxpool | int2 | — | — | — | — | — | — | — | — | — | 7.371 | — | s8 | 0e+0 |
| cpu | layout:maxpool_bwd | int2 | — | — | — | — | — | — | — | — | — | 44.028 | — | s32 | 0e+0 |
| cpu | layout:adaptive_maxpool | int2 | — | — | — | — | — | — | — | — | — | 33.679 | — | s8 | 0e+0 |
| cpu | layout:depthwise_conv | int2 | — | — | — | — | — | — | — | — | — | 172.204 | — | s32 | 7e+0 |
| cpu | layout:adaptive_maxpool_bwd | int2 | — | — | — | — | — | — | — | — | — | 14421.366 | — | s32 | 0e+0 |
| cpu | matmul_nt | int2 | — | — | — | — | — | — | — | — | — | 2.081 | — | s8dp | 3.25e+2 |
| cpu | matmul_tn | int2 | — | — | — | — | — | — | — | — | — | 2.124 | — | s8dp | 3.24e+2 |
| cpu | matmul_tt | int2 | — | — | — | — | — | — | — | — | — | 2.089 | — | s8dp | 3.2e+2 |
| cpu | matmul_batched | int2 | — | — | — | — | — | — | — | — | — | 1.947 | — | s8dp | 9.7e+1 |
| cpu | matmul_bwd | int2 | — | — | — | — | — | — | — | — | — | 4.263 | — | s8dp | 3.21e+2 |
| cpu | conv | int2 | — | — | — | — | — | — | — | — | — | 0.995 | — | s8dp | 2.04e+2 |
| cpu | topk_rows | int2 | — | — | — | — | — | — | — | — | — | 432.060 | — | s8 | 0e+0 |
| cpu | pick_rows | int2 | — | — | — | — | — | — | — | — | — | 0.033 | — | s8 | 0e+0 |
| cpu | add | uint64 | 16777216 | 12.511 | — | typed tile | 0e+0 | 7.053 | 1.77 | Vector<UInt64> | 0e+0 | 6.979 | 1.79 | u64 | 0e+0 |
| cpu | matmul | uint64 | 1024x1024x1024 | 6.756 | — | to f32 and back | 0e+0 | 33.098 | 0.20 | Vector<UInt64> | 0e+0 | 0.663 | 10.19 | u8dp | 0e+0 |
| cpu | map:copy | uint64 | — | — | — | — | — | — | — | — | — | 4.700 | — | u64 | 0e+0 |
| cpu | map:fill | uint64 | — | — | — | — | — | — | — | — | — | 2.626 | — | u64 | 0e+0 |
| cpu | map:neg | uint64 | — | — | — | — | — | — | — | — | — | 4.692 | — | u64 | 0e+0 |
| cpu | map:abs | uint64 | — | — | — | — | — | — | — | — | — | 4.689 | — | u64 | 0e+0 |
| cpu | map:sign | uint64 | — | — | — | — | — | — | — | — | — | 4.725 | — | u64 | 0e+0 |
| cpu | map:square | uint64 | — | — | — | — | — | — | — | — | — | 4.651 | — | u64 | 0e+0 |
| cpu | map:relu | uint64 | — | — | — | — | — | — | — | — | — | 4.712 | — | u64 | 0e+0 |
| cpu | map:add_scalar | uint64 | — | — | — | — | — | — | — | — | — | 4.641 | — | u64 | 0e+0 |
| cpu | map:scale | uint64 | — | — | — | — | — | — | — | — | — | 4.655 | — | u64 | 0e+0 |
| cpu | map:affine | uint64 | — | — | — | — | — | — | — | — | — | 4.798 | — | u64 | 0e+0 |
| cpu | map:clamp | uint64 | — | — | — | — | — | — | — | — | — | 4.864 | — | u64 | 0e+0 |
| cpu | map:add | uint64 | — | — | — | — | — | — | — | — | — | 6.711 | — | u64 | 0e+0 |
| cpu | map:sub | uint64 | — | — | — | — | — | — | — | — | — | 6.700 | — | u64 | 0e+0 |
| cpu | map:mul | uint64 | — | — | — | — | — | — | — | — | — | 6.751 | — | u64 | 0e+0 |
| cpu | map:div | uint64 | — | — | — | — | — | — | — | — | — | 6.578 | — | u64 | 0e+0 |
| cpu | map:max | uint64 | — | — | — | — | — | — | — | — | — | 6.701 | — | u64 | 0e+0 |
| cpu | map:min | uint64 | — | — | — | — | — | — | — | — | — | 6.869 | — | u64 | 0e+0 |
| cpu | map:axpy | uint64 | — | — | — | — | — | — | — | — | — | 6.708 | — | u64 | 0e+0 |
| cpu | map:muladd | uint64 | — | — | — | — | — | — | — | — | — | 8.460 | — | u64 | 0e+0 |
| cpu | map:where | uint64 | — | — | — | — | — | — | — | — | — | 8.400 | — | u64 | 0e+0 |
| cpu | map:square_bwd | uint64 | — | — | — | — | — | — | — | — | — | 6.772 | — | u64 | 0e+0 |
| cpu | map:abs_bwd | uint64 | — | — | — | — | — | — | — | — | — | 6.846 | — | u64 | 0e+0 |
| cpu | map:relu_bwd | uint64 | — | — | — | — | — | — | — | — | — | 6.909 | — | u64 | 0e+0 |
| cpu | map:clamp_bwd | uint64 | — | — | — | — | — | — | — | — | — | 6.834 | — | u64 | 0e+0 |
| cpu | map:where_bwd | uint64 | — | — | — | — | — | — | — | — | — | 6.754 | — | u64 | 0e+0 |
| cpu | sum:whole | uint64 | — | — | — | — | — | — | — | — | — | 2.660 | — | u8dp | 0e+0 |
| cpu | sum:rows | uint64 | — | — | — | — | — | — | — | — | — | 2.580 | — | u8dp | 0e+0 |
| cpu | sum:cols | uint64 | — | — | — | — | — | — | — | — | — | 7.290 | — | u8dp | 0e+0 |
| cpu | sumsq:rows | uint64 | — | — | — | — | — | — | — | — | — | 2.927 | — | u8dp | 0e+0 |
| cpu | max:whole | uint64 | — | — | — | — | — | — | — | — | — | 1.408 | — | u64 | 0e+0 |
| cpu | max:rows | uint64 | — | — | — | — | — | — | — | — | — | 1.467 | — | u64 | 0e+0 |
| cpu | max:cols | uint64 | — | — | — | — | — | — | — | — | — | 4.790 | — | u64 | 0e+0 |
| cpu | min:rows | uint64 | — | — | — | — | — | — | — | — | — | 1.306 | — | u64 | 0e+0 |
| cpu | argmax:rows | uint64 | — | — | — | — | — | — | — | — | — | 1.312 | — | s32 index | 0e+0 |
| cpu | argmax:cols | uint64 | — | — | — | — | — | — | — | — | — | 4.584 | — | s32 index | 0e+0 |
| cpu | maprow:add | uint64 | — | — | — | — | — | — | — | — | — | 4.776 | — | u64 | 0e+0 |
| cpu | maprow:mul | uint64 | — | — | — | — | — | — | — | — | — | 4.749 | — | u64 | 0e+0 |
| cpu | transpose | uint64 | — | — | — | — | — | — | — | — | — | 6.218 | — | u64 | 0e+0 |
| cpu | gemv | uint64 | — | — | — | — | — | — | — | — | — | 12.619 | — | u8dp | 0e+0 |
| cpu | gemv_t | uint64 | — | — | — | — | — | — | — | — | — | 17.178 | — | u8dp | 0e+0 |
| cpu | layout:strided | uint64 | — | — | — | — | — | — | — | — | — | 20.365 | — | u64 | 0e+0 |
| cpu | layout:one_hot | uint64 | — | — | — | — | — | — | — | — | — | 2.417 | — | u64 | 0e+0 |
| cpu | layout:gather_rows | uint64 | — | — | — | — | — | — | — | — | — | 3.446 | — | u64 | 0e+0 |
| cpu | layout:gather_cols | uint64 | — | — | — | — | — | — | — | — | — | 4.886 | — | u64 | 0e+0 |
| cpu | layout:scatter_add_rows | uint64 | — | — | — | — | — | — | — | — | — | 7.239 | — | u32 | 0e+0 |
| cpu | layout:im2col | uint64 | — | — | — | — | — | — | — | — | — | 1.252 | — | u64 | 0e+0 |
| cpu | layout:col2im | uint64 | — | — | — | — | — | — | — | — | — | 3.643 | — | u32 | 0e+0 |
| cpu | layout:interpolate_nearest | uint64 | — | — | — | — | — | — | — | — | — | 3.503 | — | u64 | 0e+0 |
| cpu | layout:shift | uint64 | — | — | — | — | — | — | — | — | — | 3.947 | — | u64 | 0e+0 |
| cpu | layout:pad2d | uint64 | — | — | — | — | — | — | — | — | — | 3.683 | — | u64 | 0e+0 |
| cpu | layout:maxpool | uint64 | — | — | — | — | — | — | — | — | — | 1.797 | — | u64 | 0e+0 |
| cpu | layout:maxpool_bwd | uint64 | — | — | — | — | — | — | — | — | — | 11.450 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool | uint64 | — | — | — | — | — | — | — | — | — | 8.614 | — | u64 | 0e+0 |
| cpu | layout:depthwise_conv | uint64 | — | — | — | — | — | — | — | — | — | 44.786 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | uint64 | — | — | — | — | — | — | — | — | — | 2719.826 | — | u32 | 0e+0 |
| cpu | matmul_nt | uint64 | — | — | — | — | — | — | — | — | — | 0.811 | — | u8dp | 0e+0 |
| cpu | matmul_tn | uint64 | — | — | — | — | — | — | — | — | — | 1.016 | — | u8dp | 0e+0 |
| cpu | matmul_tt | uint64 | — | — | — | — | — | — | — | — | — | 1.081 | — | u8dp | 0e+0 |
| cpu | matmul_batched | uint64 | — | — | — | — | — | — | — | — | — | 1.425 | — | u8dp | 0e+0 |
| cpu | matmul_bwd | uint64 | — | — | — | — | — | — | — | — | — | 1.951 | — | u8dp | 0e+0 |
| cpu | conv | uint64 | — | — | — | — | — | — | — | — | — | 1.471 | — | u8dp | 0e+0 |
| cpu | topk_rows | uint64 | — | — | — | — | — | — | — | — | — | 233.075 | — | u64 | 0e+0 |
| cpu | pick_rows | uint64 | — | — | — | — | — | — | — | — | — | 0.029 | — | u64 | 0e+0 |
| cpu | add | uint32 | 16777216 | 12.454 | — | typed tile | 0e+0 | 3.022 | 4.12 | Vector<UInt32> | 0e+0 | 2.868 | 4.34 | u32 | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.307 | — | to f32 and back | 0e+0 | 6.386 | 0.99 | Vector<UInt32> | 0e+0 | 0.504 | 12.51 | u8dp | 0e+0 |
| cpu | map:copy | uint32 | — | — | — | — | — | — | — | — | — | 1.652 | — | u32 | 0e+0 |
| cpu | map:fill | uint32 | — | — | — | — | — | — | — | — | — | 0.409 | — | u32 | 0e+0 |
| cpu | map:neg | uint32 | — | — | — | — | — | — | — | — | — | 1.927 | — | u32 | 0e+0 |
| cpu | map:abs | uint32 | — | — | — | — | — | — | — | — | — | 1.887 | — | u32 | 0e+0 |
| cpu | map:sign | uint32 | — | — | — | — | — | — | — | — | — | 1.917 | — | u32 | 0e+0 |
| cpu | map:square | uint32 | — | — | — | — | — | — | — | — | — | 1.914 | — | u32 | 0e+0 |
| cpu | map:relu | uint32 | — | — | — | — | — | — | — | — | — | 1.961 | — | u32 | 0e+0 |
| cpu | map:add_scalar | uint32 | — | — | — | — | — | — | — | — | — | 1.912 | — | u32 | 0e+0 |
| cpu | map:scale | uint32 | — | — | — | — | — | — | — | — | — | 1.926 | — | u32 | 0e+0 |
| cpu | map:affine | uint32 | — | — | — | — | — | — | — | — | — | 1.919 | — | u32 | 0e+0 |
| cpu | map:clamp | uint32 | — | — | — | — | — | — | — | — | — | 1.915 | — | u32 | 0e+0 |
| cpu | map:add | uint32 | — | — | — | — | — | — | — | — | — | 2.999 | — | u32 | 0e+0 |
| cpu | map:sub | uint32 | — | — | — | — | — | — | — | — | — | 3.034 | — | u32 | 0e+0 |
| cpu | map:mul | uint32 | — | — | — | — | — | — | — | — | — | 3.098 | — | u32 | 0e+0 |
| cpu | map:div | uint32 | — | — | — | — | — | — | — | — | — | 2.906 | — | u32 | 0e+0 |
| cpu | map:max | uint32 | — | — | — | — | — | — | — | — | — | 3.170 | — | u32 | 0e+0 |
| cpu | map:min | uint32 | — | — | — | — | — | — | — | — | — | 2.943 | — | u32 | 0e+0 |
| cpu | map:axpy | uint32 | — | — | — | — | — | — | — | — | — | 3.057 | — | u32 | 0e+0 |
| cpu | map:muladd | uint32 | — | — | — | — | — | — | — | — | — | 4.096 | — | u32 | 0e+0 |
| cpu | map:where | uint32 | — | — | — | — | — | — | — | — | — | 4.097 | — | u32 | 0e+0 |
| cpu | map:square_bwd | uint32 | — | — | — | — | — | — | — | — | — | 2.911 | — | u32 | 0e+0 |
| cpu | map:abs_bwd | uint32 | — | — | — | — | — | — | — | — | — | 2.856 | — | u32 | 0e+0 |
| cpu | map:relu_bwd | uint32 | — | — | — | — | — | — | — | — | — | 2.939 | — | u32 | 0e+0 |
| cpu | map:clamp_bwd | uint32 | — | — | — | — | — | — | — | — | — | 2.944 | — | u32 | 0e+0 |
| cpu | map:where_bwd | uint32 | — | — | — | — | — | — | — | — | — | 2.922 | — | u32 | 0e+0 |
| cpu | sum:whole | uint32 | — | — | — | — | — | — | — | — | — | 0.761 | — | u8dp | 0e+0 |
| cpu | sum:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.773 | — | u8dp | 0e+0 |
| cpu | sum:cols | uint32 | — | — | — | — | — | — | — | — | — | 1.555 | — | u8dp | 0e+0 |
| cpu | sumsq:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.766 | — | u8dp | 0e+0 |
| cpu | max:whole | uint32 | — | — | — | — | — | — | — | — | — | 0.381 | — | u32 | 0e+0 |
| cpu | max:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.389 | — | u32 | 0e+0 |
| cpu | max:cols | uint32 | — | — | — | — | — | — | — | — | — | 2.264 | — | u32 | 0e+0 |
| cpu | min:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.392 | — | u32 | 0e+0 |
| cpu | argmax:rows | uint32 | — | — | — | — | — | — | — | — | — | 0.376 | — | s32 index | 0e+0 |
| cpu | argmax:cols | uint32 | — | — | — | — | — | — | — | — | — | 5.300 | — | s32 index | 0e+0 |
| cpu | maprow:add | uint32 | — | — | — | — | — | — | — | — | — | 1.723 | — | u32 | 0e+0 |
| cpu | maprow:mul | uint32 | — | — | — | — | — | — | — | — | — | 1.679 | — | u32 | 0e+0 |
| cpu | transpose | uint32 | — | — | — | — | — | — | — | — | — | 2.866 | — | u32 | 0e+0 |
| cpu | gemv | uint32 | — | — | — | — | — | — | — | — | — | 5.891 | — | u8dp | 0e+0 |
| cpu | gemv_t | uint32 | — | — | — | — | — | — | — | — | — | 9.020 | — | u8dp | 0e+0 |
| cpu | layout:strided | uint32 | — | — | — | — | — | — | — | — | — | 15.764 | — | u32 | 0e+0 |
| cpu | layout:one_hot | uint32 | — | — | — | — | — | — | — | — | — | 0.776 | — | u32 | 0e+0 |
| cpu | layout:gather_rows | uint32 | — | — | — | — | — | — | — | — | — | 1.475 | — | u32 | 0e+0 |
| cpu | layout:gather_cols | uint32 | — | — | — | — | — | — | — | — | — | 2.823 | — | u32 | 0e+0 |
| cpu | layout:scatter_add_rows | uint32 | — | — | — | — | — | — | — | — | — | 3.211 | — | u32 | 0e+0 |
| cpu | layout:im2col | uint32 | — | — | — | — | — | — | — | — | — | 1.029 | — | u32 | 0e+0 |
| cpu | layout:col2im | uint32 | — | — | — | — | — | — | — | — | — | 2.006 | — | u32 | 0e+0 |
| cpu | layout:interpolate_nearest | uint32 | — | — | — | — | — | — | — | — | — | 1.807 | — | u32 | 0e+0 |
| cpu | layout:shift | uint32 | — | — | — | — | — | — | — | — | — | 2.140 | — | u32 | 0e+0 |
| cpu | layout:pad2d | uint32 | — | — | — | — | — | — | — | — | — | 1.683 | — | u32 | 0e+0 |
| cpu | layout:maxpool | uint32 | — | — | — | — | — | — | — | — | — | 1.715 | — | u32 | 0e+0 |
| cpu | layout:maxpool_bwd | uint32 | — | — | — | — | — | — | — | — | — | 6.867 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool | uint32 | — | — | — | — | — | — | — | — | — | 9.390 | — | u32 | 0e+0 |
| cpu | layout:depthwise_conv | uint32 | — | — | — | — | — | — | — | — | — | 5.753 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | uint32 | — | — | — | — | — | — | — | — | — | 1423.310 | — | u32 | 0e+0 |
| cpu | matmul_nt | uint32 | — | — | — | — | — | — | — | — | — | 0.649 | — | u8dp | 0e+0 |
| cpu | matmul_tn | uint32 | — | — | — | — | — | — | — | — | — | 0.596 | — | u8dp | 0e+0 |
| cpu | matmul_tt | uint32 | — | — | — | — | — | — | — | — | — | 0.703 | — | u8dp | 0e+0 |
| cpu | matmul_batched | uint32 | — | — | — | — | — | — | — | — | — | 0.532 | — | u8dp | 0e+0 |
| cpu | matmul_bwd | uint32 | — | — | — | — | — | — | — | — | — | 1.522 | — | u8dp | 0e+0 |
| cpu | conv | uint32 | — | — | — | — | — | — | — | — | — | 0.984 | — | u8dp | 0e+0 |
| cpu | topk_rows | uint32 | — | — | — | — | — | — | — | — | — | 332.775 | — | u32 | 0e+0 |
| cpu | pick_rows | uint32 | — | — | — | — | — | — | — | — | — | 0.027 | — | u32 | 0e+0 |
| cpu | add | uint16 | 16777216 | 1.441 | — | typed tile | 0e+0 | 1.241 | 1.16 | Vector<UInt16> | 0e+0 | 0.791 | 1.82 | u16 | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.436 | — | to f32 and back | 0e+0 | 2.004 | 2.71 | Vector<UInt16> | 0e+0 | 0.356 | 15.28 | u8dp | 0e+0 |
| cpu | map:copy | uint16 | — | — | — | — | — | — | — | — | — | 0.320 | — | u16 | 0e+0 |
| cpu | map:fill | uint16 | — | — | — | — | — | — | — | — | — | 0.129 | — | u16 | 0e+0 |
| cpu | map:neg | uint16 | — | — | — | — | — | — | — | — | — | 0.265 | — | u16 | 0e+0 |
| cpu | map:abs | uint16 | — | — | — | — | — | — | — | — | — | 0.265 | — | u16 | 0e+0 |
| cpu | map:sign | uint16 | — | — | — | — | — | — | — | — | — | 0.268 | — | u16 | 0e+0 |
| cpu | map:square | uint16 | — | — | — | — | — | — | — | — | — | 0.310 | — | u16 | 0e+0 |
| cpu | map:relu | uint16 | — | — | — | — | — | — | — | — | — | 0.296 | — | u16 | 0e+0 |
| cpu | map:add_scalar | uint16 | — | — | — | — | — | — | — | — | — | 0.281 | — | u16 | 0e+0 |
| cpu | map:scale | uint16 | — | — | — | — | — | — | — | — | — | 0.264 | — | u16 | 0e+0 |
| cpu | map:affine | uint16 | — | — | — | — | — | — | — | — | — | 0.277 | — | u16 | 0e+0 |
| cpu | map:clamp | uint16 | — | — | — | — | — | — | — | — | — | 0.276 | — | u16 | 0e+0 |
| cpu | map:add | uint16 | — | — | — | — | — | — | — | — | — | 0.767 | — | u16 | 0e+0 |
| cpu | map:sub | uint16 | — | — | — | — | — | — | — | — | — | 0.823 | — | u16 | 0e+0 |
| cpu | map:mul | uint16 | — | — | — | — | — | — | — | — | — | 0.842 | — | u16 | 0e+0 |
| cpu | map:div | uint16 | — | — | — | — | — | — | — | — | — | 1.405 | — | u16 | 0e+0 |
| cpu | map:max | uint16 | — | — | — | — | — | — | — | — | — | 0.801 | — | u16 | 0e+0 |
| cpu | map:min | uint16 | — | — | — | — | — | — | — | — | — | 0.799 | — | u16 | 0e+0 |
| cpu | map:axpy | uint16 | — | — | — | — | — | — | — | — | — | 0.807 | — | u16 | 0e+0 |
| cpu | map:muladd | uint16 | — | — | — | — | — | — | — | — | — | 1.366 | — | u16 | 0e+0 |
| cpu | map:where | uint16 | — | — | — | — | — | — | — | — | — | 1.340 | — | u16 | 0e+0 |
| cpu | map:square_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.795 | — | u16 | 0e+0 |
| cpu | map:abs_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.771 | — | u16 | 0e+0 |
| cpu | map:relu_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.862 | — | u16 | 0e+0 |
| cpu | map:clamp_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.859 | — | u16 | 0e+0 |
| cpu | map:where_bwd | uint16 | — | — | — | — | — | — | — | — | — | 0.849 | — | u16 | 0e+0 |
| cpu | sum:whole | uint16 | — | — | — | — | — | — | — | — | — | 0.257 | — | u8dp | 0e+0 |
| cpu | sum:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.272 | — | u8dp | 0e+0 |
| cpu | sum:cols | uint16 | — | — | — | — | — | — | — | — | — | 0.878 | — | u8dp | 0e+0 |
| cpu | sumsq:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.268 | — | u8dp | 0e+0 |
| cpu | max:whole | uint16 | — | — | — | — | — | — | — | — | — | 0.127 | — | u16 | 0e+0 |
| cpu | max:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.137 | — | u16 | 0e+0 |
| cpu | max:cols | uint16 | — | — | — | — | — | — | — | — | — | 1.728 | — | u16 | 0e+0 |
| cpu | min:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.134 | — | u16 | 0e+0 |
| cpu | argmax:rows | uint16 | — | — | — | — | — | — | — | — | — | 0.135 | — | s32 index | 0e+0 |
| cpu | argmax:cols | uint16 | — | — | — | — | — | — | — | — | — | 5.874 | — | s32 index | 0e+0 |
| cpu | maprow:add | uint16 | — | — | — | — | — | — | — | — | — | 0.354 | — | u16 | 0e+0 |
| cpu | maprow:mul | uint16 | — | — | — | — | — | — | — | — | — | 0.301 | — | u16 | 0e+0 |
| cpu | transpose | uint16 | — | — | — | — | — | — | — | — | — | 0.913 | — | u16 | 0e+0 |
| cpu | gemv | uint16 | — | — | — | — | — | — | — | — | — | 2.537 | — | u8dp | 0e+0 |
| cpu | gemv_t | uint16 | — | — | — | — | — | — | — | — | — | 5.186 | — | u8dp | 0e+0 |
| cpu | layout:strided | uint16 | — | — | — | — | — | — | — | — | — | 6.030 | — | u16 | 0e+0 |
| cpu | layout:one_hot | uint16 | — | — | — | — | — | — | — | — | — | 0.127 | — | u16 | 0e+0 |
| cpu | layout:gather_rows | uint16 | — | — | — | — | — | — | — | — | — | 0.329 | — | u16 | 0e+0 |
| cpu | layout:gather_cols | uint16 | — | — | — | — | — | — | — | — | — | 0.873 | — | u16 | 0e+0 |
| cpu | layout:scatter_add_rows | uint16 | — | — | — | — | — | — | — | — | — | 3.605 | — | u32 | 0e+0 |
| cpu | layout:im2col | uint16 | — | — | — | — | — | — | — | — | — | 0.946 | — | u16 | 0e+0 |
| cpu | layout:col2im | uint16 | — | — | — | — | — | — | — | — | — | 3.225 | — | u32 | 0e+0 |
| cpu | layout:interpolate_nearest | uint16 | — | — | — | — | — | — | — | — | — | 2.136 | — | u16 | 0e+0 |
| cpu | layout:shift | uint16 | — | — | — | — | — | — | — | — | — | 1.146 | — | u16 | 0e+0 |
| cpu | layout:pad2d | uint16 | — | — | — | — | — | — | — | — | — | 0.441 | — | u16 | 0e+0 |
| cpu | layout:maxpool | uint16 | — | — | — | — | — | — | — | — | — | 2.117 | — | u16 | 0e+0 |
| cpu | layout:maxpool_bwd | uint16 | — | — | — | — | — | — | — | — | — | 11.919 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool | uint16 | — | — | — | — | — | — | — | — | — | 15.349 | — | u16 | 0e+0 |
| cpu | layout:depthwise_conv | uint16 | — | — | — | — | — | — | — | — | — | 16.180 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | uint16 | — | — | — | — | — | — | — | — | — | 3633.520 | — | u32 | 0e+0 |
| cpu | matmul_nt | uint16 | — | — | — | — | — | — | — | — | — | 0.573 | — | u8dp | 0e+0 |
| cpu | matmul_tn | uint16 | — | — | — | — | — | — | — | — | — | 0.804 | — | u8dp | 0e+0 |
| cpu | matmul_tt | uint16 | — | — | — | — | — | — | — | — | — | 0.876 | — | u8dp | 0e+0 |
| cpu | matmul_batched | uint16 | — | — | — | — | — | — | — | — | — | 0.451 | — | u8dp | 0e+0 |
| cpu | matmul_bwd | uint16 | — | — | — | — | — | — | — | — | — | 1.410 | — | u8dp | 0e+0 |
| cpu | conv | uint16 | — | — | — | — | — | — | — | — | — | 0.919 | — | u8dp | 0e+0 |
| cpu | topk_rows | uint16 | — | — | — | — | — | — | — | — | — | 357.098 | — | u16 | 0e+0 |
| cpu | pick_rows | uint16 | — | — | — | — | — | — | — | — | — | 0.027 | — | u16 | 0e+0 |
| cpu | add | uint8 | 16777216 | 1.902 | — | typed tile | 0e+0 | 0.478 | 3.98 | Vector<Byte> | 0e+0 | 0.127 | 14.98 | u8 | 0e+0 |
| cpu | matmul | uint8 | 1024x1024x1024 | 6.172 | — | to f32 and back | 2.48e+3 | 4.583 | 1.35 | Vector<Byte> | 2.56e+3 | 0.485 | 12.72 | u8dp | 2.48e+3 |
| cpu | map:copy | uint8 | — | — | — | — | — | — | — | — | — | 0.084 | — | u8 | 0e+0 |
| cpu | map:fill | uint8 | — | — | — | — | — | — | — | — | — | 0.083 | — | u8 | 0e+0 |
| cpu | map:neg | uint8 | — | — | — | — | — | — | — | — | — | 0.120 | — | u8 | 0e+0 |
| cpu | map:abs | uint8 | — | — | — | — | — | — | — | — | — | 0.100 | — | u8 | 0e+0 |
| cpu | map:sign | uint8 | — | — | — | — | — | — | — | — | — | 0.109 | — | u8 | 0e+0 |
| cpu | map:square | uint8 | — | — | — | — | — | — | — | — | — | 0.123 | — | u8 | 0e+0 |
| cpu | map:relu | uint8 | — | — | — | — | — | — | — | — | — | 0.121 | — | u8 | 0e+0 |
| cpu | map:add_scalar | uint8 | — | — | — | — | — | — | — | — | — | 0.116 | — | u8 | 0e+0 |
| cpu | map:scale | uint8 | — | — | — | — | — | — | — | — | — | 0.127 | — | u8 | 0e+0 |
| cpu | map:affine | uint8 | — | — | — | — | — | — | — | — | — | 0.125 | — | u8 | 0e+0 |
| cpu | map:clamp | uint8 | — | — | — | — | — | — | — | — | — | 0.103 | — | u8 | 0e+0 |
| cpu | map:add | uint8 | — | — | — | — | — | — | — | — | — | 0.240 | — | u8 | 0e+0 |
| cpu | map:sub | uint8 | — | — | — | — | — | — | — | — | — | 0.159 | — | u8 | 0e+0 |
| cpu | map:mul | uint8 | — | — | — | — | — | — | — | — | — | 0.210 | — | u8 | 0e+0 |
| cpu | map:div | uint8 | — | — | — | — | — | — | — | — | — | 1.181 | — | u8 | 0e+0 |
| cpu | map:max | uint8 | — | — | — | — | — | — | — | — | — | 0.237 | — | u8 | 0e+0 |
| cpu | map:min | uint8 | — | — | — | — | — | — | — | — | — | 0.162 | — | u8 | 0e+0 |
| cpu | map:axpy | uint8 | — | — | — | — | — | — | — | — | — | 0.151 | — | u8 | 0e+0 |
| cpu | map:muladd | uint8 | — | — | — | — | — | — | — | — | — | 0.304 | — | u8 | 0e+0 |
| cpu | map:where | uint8 | — | — | — | — | — | — | — | — | — | 0.285 | — | u8 | 0e+0 |
| cpu | map:square_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.144 | — | u8 | 0e+0 |
| cpu | map:abs_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.139 | — | u8 | 0e+0 |
| cpu | map:relu_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.129 | — | u8 | 0e+0 |
| cpu | map:clamp_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.156 | — | u8 | 0e+0 |
| cpu | map:where_bwd | uint8 | — | — | — | — | — | — | — | — | — | 0.132 | — | u8 | 0e+0 |
| cpu | sum:whole | uint8 | — | — | — | — | — | — | — | — | — | 0.075 | — | u8dp | 0e+0 |
| cpu | sum:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.074 | — | u8dp | 0e+0 |
| cpu | sum:cols | uint8 | — | — | — | — | — | — | — | — | — | 0.304 | — | u8dp | 0e+0 |
| cpu | sumsq:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.072 | — | u8dp | 0e+0 |
| cpu | max:whole | uint8 | — | — | — | — | — | — | — | — | — | 0.037 | — | u8 | 0e+0 |
| cpu | max:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.054 | — | u8 | 0e+0 |
| cpu | max:cols | uint8 | — | — | — | — | — | — | — | — | — | 1.489 | — | u8 | 0e+0 |
| cpu | min:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.059 | — | u8 | 0e+0 |
| cpu | argmax:rows | uint8 | — | — | — | — | — | — | — | — | — | 0.055 | — | s32 index | 0e+0 |
| cpu | argmax:cols | uint8 | — | — | — | — | — | — | — | — | — | 4.369 | — | s32 index | 0e+0 |
| cpu | maprow:add | uint8 | — | — | — | — | — | — | — | — | — | 0.086 | — | u8 | 0e+0 |
| cpu | maprow:mul | uint8 | — | — | — | — | — | — | — | — | — | 0.144 | — | u8 | 0e+0 |
| cpu | transpose | uint8 | — | — | — | — | — | — | — | — | — | 0.759 | — | u8 | 0e+0 |
| cpu | gemv | uint8 | — | — | — | — | — | — | — | — | — | 0.895 | — | u8dp | 1.9e+4 |
| cpu | gemv_t | uint8 | — | — | — | — | — | — | — | — | — | 3.234 | — | u8dp | 1.91e+4 |
| cpu | layout:strided | uint8 | — | — | — | — | — | — | — | — | — | 4.209 | — | u8 | 0e+0 |
| cpu | layout:one_hot | uint8 | — | — | — | — | — | — | — | — | — | 0.075 | — | u8 | 0e+0 |
| cpu | layout:gather_rows | uint8 | — | — | — | — | — | — | — | — | — | 0.115 | — | u8 | 0e+0 |
| cpu | layout:gather_cols | uint8 | — | — | — | — | — | — | — | — | — | 0.748 | — | u8 | 0e+0 |
| cpu | layout:scatter_add_rows | uint8 | — | — | — | — | — | — | — | — | — | 5.003 | — | u32 | 0e+0 |
| cpu | layout:im2col | uint8 | — | — | — | — | — | — | — | — | — | 1.065 | — | u8 | 0e+0 |
| cpu | layout:col2im | uint8 | — | — | — | — | — | — | — | — | — | 3.506 | — | u32 | 0e+0 |
| cpu | layout:interpolate_nearest | uint8 | — | — | — | — | — | — | — | — | — | 1.609 | — | u8 | 0e+0 |
| cpu | layout:shift | uint8 | — | — | — | — | — | — | — | — | — | 0.861 | — | u8 | 0e+0 |
| cpu | layout:pad2d | uint8 | — | — | — | — | — | — | — | — | — | 0.094 | — | u8 | 0e+0 |
| cpu | layout:maxpool | uint8 | — | — | — | — | — | — | — | — | — | 1.865 | — | u8 | 0e+0 |
| cpu | layout:maxpool_bwd | uint8 | — | — | — | — | — | — | — | — | — | 11.661 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool | uint8 | — | — | — | — | — | — | — | — | — | 14.468 | — | u8 | 0e+0 |
| cpu | layout:depthwise_conv | uint8 | — | — | — | — | — | — | — | — | — | 22.209 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool_bwd | uint8 | — | — | — | — | — | — | — | — | — | 2542.219 | — | u32 | 0e+0 |
| cpu | matmul_nt | uint8 | — | — | — | — | — | — | — | — | — | 0.560 | — | u8dp | 2.48e+3 |
| cpu | matmul_tn | uint8 | — | — | — | — | — | — | — | — | — | 0.607 | — | u8dp | 2.43e+3 |
| cpu | matmul_tt | uint8 | — | — | — | — | — | — | — | — | — | 0.608 | — | u8dp | 2.48e+3 |
| cpu | matmul_batched | uint8 | — | — | — | — | — | — | — | — | — | 0.399 | — | u8dp | 5.3e+2 |
| cpu | matmul_bwd | uint8 | — | — | — | — | — | — | — | — | — | 1.051 | — | u8dp | 2.49e+3 |
| cpu | conv | uint8 | — | — | — | — | — | — | — | — | — | 0.866 | — | u8dp | 1.35e+3 |
| cpu | topk_rows | uint8 | — | — | — | — | — | — | — | — | — | 296.490 | — | u8 | 0e+0 |
| cpu | pick_rows | uint8 | — | — | — | — | — | — | — | — | — | 0.024 | — | u8 | 0e+0 |
| cpu | add | uint2 | 16777216 | 2.731 | — | typed tile | 0e+0 | — | — | — | — | 0.244 | 11.20 | u8 | 0e+0 |
| cpu | matmul | uint2 | 1024x1024x1024 | 6.749 | — | to f32 and back | 3.18e+2 | — | — | — | — | 0.525 | 12.86 | u8dp | 3.18e+2 |
| cpu | map:copy | uint2 | — | — | — | — | — | — | — | — | — | 0.360 | — | u8 | 0e+0 |
| cpu | map:fill | uint2 | — | — | — | — | — | — | — | — | — | 0.322 | — | u8 | 0e+0 |
| cpu | map:neg | uint2 | — | — | — | — | — | — | — | — | — | 0.335 | — | u8 | 0e+0 |
| cpu | map:abs | uint2 | — | — | — | — | — | — | — | — | — | 0.264 | — | u8 | 0e+0 |
| cpu | map:sign | uint2 | — | — | — | — | — | — | — | — | — | 0.286 | — | u8 | 0e+0 |
| cpu | map:square | uint2 | — | — | — | — | — | — | — | — | — | 0.283 | — | u8 | 0e+0 |
| cpu | map:relu | uint2 | — | — | — | — | — | — | — | — | — | 0.286 | — | u8 | 0e+0 |
| cpu | map:add_scalar | uint2 | — | — | — | — | — | — | — | — | — | 0.291 | — | u8 | 0e+0 |
| cpu | map:scale | uint2 | — | — | — | — | — | — | — | — | — | 0.286 | — | u8 | 0e+0 |
| cpu | map:affine | uint2 | — | — | — | — | — | — | — | — | — | 0.293 | — | u8 | 0e+0 |
| cpu | map:clamp | uint2 | — | — | — | — | — | — | — | — | — | 0.271 | — | u8 | 0e+0 |
| cpu | map:add | uint2 | — | — | — | — | — | — | — | — | — | 0.305 | — | u8 | 0e+0 |
| cpu | map:sub | uint2 | — | — | — | — | — | — | — | — | — | 0.300 | — | u8 | 0e+0 |
| cpu | map:mul | uint2 | — | — | — | — | — | — | — | — | — | 0.313 | — | u8 | 0e+0 |
| cpu | map:div | uint2 | — | — | — | — | — | — | — | — | — | 1.326 | — | u8 | 0e+0 |
| cpu | map:max | uint2 | — | — | — | — | — | — | — | — | — | 0.315 | — | u8 | 0e+0 |
| cpu | map:min | uint2 | — | — | — | — | — | — | — | — | — | 0.313 | — | u8 | 0e+0 |
| cpu | map:axpy | uint2 | — | — | — | — | — | — | — | — | — | 0.310 | — | u8 | 0e+0 |
| cpu | map:muladd | uint2 | — | — | — | — | — | — | — | — | — | 0.367 | — | u8 | 0e+0 |
| cpu | map:where | uint2 | — | — | — | — | — | — | — | — | — | 0.481 | — | u8 | 0e+0 |
| cpu | map:square_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.502 | — | u8 | 0e+0 |
| cpu | map:abs_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.477 | — | u8 | 0e+0 |
| cpu | map:relu_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.396 | — | u8 | 0e+0 |
| cpu | map:clamp_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.399 | — | u8 | 0e+0 |
| cpu | map:where_bwd | uint2 | — | — | — | — | — | — | — | — | — | 0.413 | — | u8 | 0e+0 |
| cpu | sum:whole | uint2 | — | — | — | — | — | — | — | — | — | 0.085 | — | u8dp | 0e+0 |
| cpu | sum:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.086 | — | u8dp | 0e+0 |
| cpu | sum:cols | uint2 | — | — | — | — | — | — | — | — | — | 0.639 | — | u8dp | 0e+0 |
| cpu | sumsq:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.065 | — | u8dp | 0e+0 |
| cpu | max:whole | uint2 | — | — | — | — | — | — | — | — | — | 0.095 | — | u8 | 0e+0 |
| cpu | max:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.219 | — | u8 | 0e+0 |
| cpu | max:cols | uint2 | — | — | — | — | — | — | — | — | — | 1.804 | — | u8 | 0e+0 |
| cpu | min:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.219 | — | u8 | 0e+0 |
| cpu | argmax:rows | uint2 | — | — | — | — | — | — | — | — | — | 0.218 | — | s32 index | 0e+0 |
| cpu | argmax:cols | uint2 | — | — | — | — | — | — | — | — | — | 6.872 | — | s32 index | 0e+0 |
| cpu | maprow:add | uint2 | — | — | — | — | — | — | — | — | — | 0.979 | — | u8 | 0e+0 |
| cpu | maprow:mul | uint2 | — | — | — | — | — | — | — | — | — | 0.242 | — | u8 | 0e+0 |
| cpu | transpose | uint2 | — | — | — | — | — | — | — | — | — | 3.440 | — | u8 | 0e+0 |
| cpu | gemv | uint2 | — | — | — | — | — | — | — | — | — | 0.405 | — | u8dp | 2.19e+3 |
| cpu | gemv_t | uint2 | — | — | — | — | — | — | — | — | — | 3.400 | — | u8dp | 2.19e+3 |
| cpu | layout:strided | uint2 | — | — | — | — | — | — | — | — | — | 5.592 | — | u8 | 0e+0 |
| cpu | layout:one_hot | uint2 | — | — | — | — | — | — | — | — | — | 0.112 | — | u8 | 0e+0 |
| cpu | layout:gather_rows | uint2 | — | — | — | — | — | — | — | — | — | 0.160 | — | u8 | 0e+0 |
| cpu | layout:gather_cols | uint2 | — | — | — | — | — | — | — | — | — | 6.425 | — | u8 | 0e+0 |
| cpu | layout:scatter_add_rows | uint2 | — | — | — | — | — | — | — | — | — | 35.422 | — | u32 | 2e+0 |
| cpu | layout:im2col | uint2 | — | — | — | — | — | — | — | — | — | 6.015 | — | u8 | 0e+0 |
| cpu | layout:col2im | uint2 | — | — | — | — | — | — | — | — | — | 14.867 | — | u32 | 6e+0 |
| cpu | layout:interpolate_nearest | uint2 | — | — | — | — | — | — | — | — | — | 11.109 | — | u8 | 0e+0 |
| cpu | layout:shift | uint2 | — | — | — | — | — | — | — | — | — | 10.371 | — | u8 | 0e+0 |
| cpu | layout:pad2d | uint2 | — | — | — | — | — | — | — | — | — | 0.200 | — | u8 | 0e+0 |
| cpu | layout:maxpool | uint2 | — | — | — | — | — | — | — | — | — | 3.692 | — | u8 | 0e+0 |
| cpu | layout:maxpool_bwd | uint2 | — | — | — | — | — | — | — | — | — | 48.865 | — | u32 | 0e+0 |
| cpu | layout:adaptive_maxpool | uint2 | — | — | — | — | — | — | — | — | — | 37.810 | — | u8 | 0e+0 |
| cpu | layout:depthwise_conv | uint2 | — | — | — | — | — | — | — | — | — | 169.680 | — | u32 | 4e+0 |
| cpu | layout:adaptive_maxpool_bwd | uint2 | — | — | — | — | — | — | — | — | — | 12629.176 | — | u32 | 0e+0 |
| cpu | matmul_nt | uint2 | — | — | — | — | — | — | — | — | — | 1.075 | — | u8dp | 3.23e+2 |
| cpu | matmul_tn | uint2 | — | — | — | — | — | — | — | — | — | 1.149 | — | u8dp | 3.19e+2 |
| cpu | matmul_tt | uint2 | — | — | — | — | — | — | — | — | — | 1.089 | — | u8dp | 3.27e+2 |
| cpu | matmul_batched | uint2 | — | — | — | — | — | — | — | — | — | 1.080 | — | u8dp | 9.6e+1 |
| cpu | matmul_bwd | uint2 | — | — | — | — | — | — | — | — | — | 2.356 | — | u8dp | 3.22e+2 |
| cpu | conv | uint2 | — | — | — | — | — | — | — | — | — | 0.894 | — | u8dp | 1.93e+2 |
| cpu | topk_rows | uint2 | — | — | — | — | — | — | — | — | — | 430.433 | — | u8 | 0e+0 |
| cpu | pick_rows | uint2 | — | — | — | — | — | — | — | — | — | 0.029 | — | u8 | 0e+0 |
| cpu | add | q8_0 | 16777216 | — | — | — | — | — | — | — | — | 6.140 | — | f32 | 7.87e-3 |
| cpu | matmul | q8_0 | 1024x1024x1024 | — | — | — | — | — | — | — | — | 9.056 | — | f32 | 2.03e-1 |
| cpu | map:copy | q8_0 | — | — | — | — | — | — | — | — | — | 5.508 | — | f32 | 0e+0 |
| cpu | map:fill | q8_0 | — | — | — | — | — | — | — | — | — | 4.828 | — | f32 | 1.83e-4 |
| cpu | map:neg | q8_0 | — | — | — | — | — | — | — | — | — | 5.532 | — | f32 | 0e+0 |
| cpu | map:abs | q8_0 | — | — | — | — | — | — | — | — | — | 5.517 | — | f32 | 0e+0 |
| cpu | map:sign | q8_0 | — | — | — | — | — | — | — | — | — | 5.666 | — | f32 | 6.1e-5 |
| cpu | map:square | q8_0 | — | — | — | — | — | — | — | — | — | 4.610 | — | f32 | 6.26e-2 |
| cpu | map:relu | q8_0 | — | — | — | — | — | — | — | — | — | 4.644 | — | f32 | 1.56e-2 |
| cpu | map:add_scalar | q8_0 | — | — | — | — | — | — | — | — | — | 4.701 | — | f32 | 2.36e-2 |
| cpu | map:scale | q8_0 | — | — | — | — | — | — | — | — | — | 5.531 | — | f32 | 3.88e-3 |
| cpu | map:affine | q8_0 | — | — | — | — | — | — | — | — | — | 5.573 | — | f32 | 3.53e-2 |
| cpu | map:clamp | q8_0 | — | — | — | — | — | — | — | — | — | 6.088 | — | f32 | 7.87e-3 |
| cpu | map:add | q8_0 | — | — | — | — | — | — | — | — | — | 5.189 | — | f32 | 3.15e-2 |
| cpu | map:sub | q8_0 | — | — | — | — | — | — | — | — | — | 6.264 | — | f32 | 3.15e-2 |
| cpu | map:mul | q8_0 | — | — | — | — | — | — | — | — | — | 6.293 | — | f32 | 6.29e-2 |
| cpu | map:div | q8_0 | — | — | — | — | — | — | — | — | — | 6.557 | — | f32 | 1.59e-2 |
| cpu | map:max | q8_0 | — | — | — | — | — | — | — | — | — | 6.454 | — | f32 | 1.57e-2 |
| cpu | map:min | q8_0 | — | — | — | — | — | — | — | — | — | 5.252 | — | f32 | 1.57e-2 |
| cpu | map:axpy | q8_0 | — | — | — | — | — | — | — | — | — | 5.179 | — | f32 | 4.71e-2 |
| cpu | map:muladd | q8_0 | — | — | — | — | — | — | — | — | — | 5.787 | — | f32 | 7.85e-2 |
| cpu | map:where | q8_0 | — | — | — | — | — | — | — | — | — | 5.814 | — | f32 | 1.57e-2 |
| cpu | map:square_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.198 | — | f32 | 1.26e-1 |
| cpu | map:abs_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.269 | — | f32 | 1.56e-2 |
| cpu | map:relu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.205 | — | f32 | 1.56e-2 |
| cpu | map:clamp_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 6.213 | — | f32 | 1.56e-2 |
| cpu | map:where_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.224 | — | f32 | 1.56e-2 |
| cpu | map:sqrt | q8_0 | — | — | — | — | — | — | — | — | — | 5.781 | — | f32 | 7.83e-3 |
| cpu | map:inv_sqrt | q8_0 | — | — | — | — | — | — | — | — | — | 6.219 | — | f32 | 5.64e-3 |
| cpu | map:exp | q8_0 | — | — | — | — | — | — | — | — | — | 7.526 | — | f32 | 2.14e-1 |
| cpu | map:expm1 | q8_0 | — | — | — | — | — | — | — | — | — | 7.356 | — | f32 | 2.54e-3 |
| cpu | map:log | q8_0 | — | — | — | — | — | — | — | — | — | 7.394 | — | f32 | 5.45e-3 |
| cpu | map:sin | q8_0 | — | — | — | — | — | — | — | — | — | 6.818 | — | f32 | 3.94e-3 |
| cpu | map:cos | q8_0 | — | — | — | — | — | — | — | — | — | 8.020 | — | f32 | 3.94e-3 |
| cpu | map:tanh | q8_0 | — | — | — | — | — | — | — | — | — | 8.263 | — | f32 | 3.93e-3 |
| cpu | map:sigmoid | q8_0 | — | — | — | — | — | — | — | — | — | 7.321 | — | f32 | 3.86e-3 |
| cpu | map:erf | q8_0 | — | — | — | — | — | — | — | — | — | 30.789 | — | f32 | 3.94e-3 |
| cpu | map:gelu | q8_0 | — | — | — | — | — | — | — | — | — | 7.965 | — | f32 | 1.57e-2 |
| cpu | map:gelu_erf | q8_0 | — | — | — | — | — | — | — | — | — | 31.950 | — | f32 | 1.57e-2 |
| cpu | map:silu | q8_0 | — | — | — | — | — | — | — | — | — | 7.406 | — | f32 | 1.54e-2 |
| cpu | map:leaky_relu | q8_0 | — | — | — | — | — | — | — | — | — | 5.674 | — | f32 | 1.57e-2 |
| cpu | map:pow | q8_0 | — | — | — | — | — | — | — | — | — | 11.154 | — | f32 | 1.25e-1 |
| cpu | map:dropout | q8_0 | — | — | — | — | — | — | — | — | — | 6.323 | — | f32 | 1.95e-2 |
| cpu | map:gated_silu | q8_0 | — | — | — | — | — | — | — | — | — | 6.766 | — | f32 | 6.16e-2 |
| cpu | map:gated_gelu | q8_0 | — | — | — | — | — | — | — | — | — | 6.997 | — | f32 | 6.29e-2 |
| cpu | map:sqrt_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.625 | — | f32 | 1.12e-2 |
| cpu | map:inv_sqrt_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.874 | — | f32 | 2.28e-2 |
| cpu | map:exp_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 6.629 | — | f32 | 8.57e-1 |
| cpu | map:log_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.339 | — | f32 | 3.22e-2 |
| cpu | map:sin_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 7.297 | — | f32 | 1.57e-2 |
| cpu | map:cos_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 7.453 | — | f32 | 1.57e-2 |
| cpu | map:tanh_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 7.598 | — | f32 | 1.57e-2 |
| cpu | map:sigmoid_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 8.416 | — | f32 | 3.93e-3 |
| cpu | map:silu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 7.994 | — | f32 | 1.73e-2 |
| cpu | map:gelu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 7.837 | — | f32 | 1.78e-2 |
| cpu | map:gelu_erf_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 32.745 | — | f32 | 1.78e-2 |
| cpu | map:leaky_relu_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.222 | — | f32 | 1.57e-2 |
| cpu | map:pow_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 8.792 | — | f32 | 3.15e-1 |
| cpu | map:gated_silu_bwd_x | q8_0 | — | — | — | — | — | — | — | — | — | 8.418 | — | f32 | 6.63e-2 |
| cpu | map:gated_silu_bwd_y | q8_0 | — | — | — | — | — | — | — | — | — | 8.946 | — | f32 | 6.17e-2 |
| cpu | map:gated_gelu_bwd_x | q8_0 | — | — | — | — | — | — | — | — | — | 8.527 | — | f32 | 6.28e-2 |
| cpu | map:gated_gelu_bwd_y | q8_0 | — | — | — | — | — | — | — | — | — | 7.585 | — | f32 | 6.29e-2 |
| cpu | sum:whole | q8_0 | — | — | — | — | — | — | — | — | — | 56.213 | — | f32 | 7.78e-4 |
| cpu | sum:rows | q8_0 | — | — | — | — | — | — | — | — | — | 55.945 | — | f32 | 7.63e-6 |
| cpu | sum:cols | q8_0 | — | — | — | — | — | — | — | — | — | 56.940 | — | f32 | 2.48e-4 |
| cpu | sumsq:rows | q8_0 | — | — | — | — | — | — | — | — | — | 56.259 | — | f32 | 2.43e-4 |
| cpu | max:whole | q8_0 | — | — | — | — | — | — | — | — | — | 0.958 | — | f32 | 0e+0 |
| cpu | max:rows | q8_0 | — | — | — | — | — | — | — | — | — | 2.207 | — | f32 | 0e+0 |
| cpu | max:cols | q8_0 | — | — | — | — | — | — | — | — | — | 3.528 | — | f32 | 0e+0 |
| cpu | min:rows | q8_0 | — | — | — | — | — | — | — | — | — | 2.359 | — | f32 | 0e+0 |
| cpu | argmax:rows | q8_0 | — | — | — | — | — | — | — | — | — | 1.969 | — | s32 index | 0e+0 |
| cpu | argmax:cols | q8_0 | — | — | — | — | — | — | — | — | — | 2.010 | — | s32 index | 0e+0 |
| cpu | maprow:add | q8_0 | — | — | — | — | — | — | — | — | — | 6.302 | — | f32 | 7.87e-3 |
| cpu | maprow:mul | q8_0 | — | — | — | — | — | — | — | — | — | 6.276 | — | f32 | 3.93e-3 |
| cpu | transpose | q8_0 | — | — | — | — | — | — | — | — | — | 7.193 | — | f32 | 3.94e-3 |
| cpu | gemv | q8_0 | — | — | — | — | — | — | — | — | — | 2.011 | — | s8dp blocks | 4.75e-1 |
| cpu | gemv_t | q8_0 | — | — | — | — | — | — | — | — | — | 228.393 | — | f32 | 5.89e-1 |
| cpu | row:log_softmax | q8_0 | — | — | — | — | — | — | — | — | — | 6.726 | — | f32 | 5.63e-2 |
| cpu | row:softmax_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.229 | — | f32 | 3.11e-1 |
| cpu | row:log_softmax_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 6.454 | — | f32 | 5.06e-1 |
| cpu | row:rms_norm_affine | q8_0 | — | — | — | — | — | — | — | — | — | 4.626 | — | f32 | 1.38e-2 |
| cpu | row:rms_norm_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.302 | — | f32 | 3.46e-3 |
| cpu | row:layer_norm | q8_0 | — | — | — | — | — | — | — | — | — | 4.663 | — | f32 | 1.75e-2 |
| cpu | row:layer_norm_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 5.405 | — | f32 | 3.54e-3 |
| cpu | row:weight_norm | q8_0 | — | — | — | — | — | — | — | — | — | 4.667 | — | f32 | 1.51e-5 |
| cpu | row:rope | q8_0 | — | — | — | — | — | — | — | — | — | 7.729 | — | f32 | 2.23e-2 |
| cpu | row:scale_mask_softmax | q8_0 | — | — | — | — | — | — | — | — | — | 7.029 | — | f32 | 1.42e-3 |
| cpu | row:add_rms_norm | q8_0 | — | — | — | — | — | — | — | — | — | 9.161 | — | f32 | 1.94e-2 |
| cpu | row:add_layer_norm | q8_0 | — | — | — | — | — | — | — | — | — | 9.259 | — | f32 | 2.26e-2 |
| cpu | row:cross_entropy_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 6.051 | — | f32 | 2.08e-3 |
| cpu | norm_params_bwd:rms | q8_0 | — | — | — | — | — | — | — | — | — | 3.847 | — | f32 | 3.08e-4 |
| cpu | norm_params_bwd:layer | q8_0 | — | — | — | — | — | — | — | — | — | 3.797 | — | f32 | 2.54e-4 |
| cpu | layout:strided | q8_0 | — | — | — | — | — | — | — | — | — | 42.040 | — | f32 | 3.94e-3 |
| cpu | layout:one_hot | q8_0 | — | — | — | — | — | — | — | — | — | 3.743 | — | f32 | 6.1e-5 |
| cpu | layout:gather_rows | q8_0 | — | — | — | — | — | — | — | — | — | 5.298 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | q8_0 | — | — | — | — | — | — | — | — | — | 30.450 | — | f32 | 3.94e-3 |
| cpu | layout:scatter_add_rows | q8_0 | — | — | — | — | — | — | — | — | — | 84.836 | — | f32 | 1.68e-2 |
| cpu | layout:im2col | q8_0 | — | — | — | — | — | — | — | — | — | 20.979 | — | f32 | 3.93e-3 |
| cpu | layout:col2im | q8_0 | — | — | — | — | — | — | — | — | — | 52.140 | — | f32 | 2.95e-2 |
| cpu | layout:interpolate_nearest | q8_0 | — | — | — | — | — | — | — | — | — | 32.560 | — | f32 | 3.91e-3 |
| cpu | layout:interpolate_bilinear | q8_0 | — | — | — | — | — | — | — | — | — | 149.004 | — | f32 | 3.92e-3 |
| cpu | layout:shift | q8_0 | — | — | — | — | — | — | — | — | — | 30.118 | — | f32 | 3.94e-3 |
| cpu | layout:pad2d | q8_0 | — | — | — | — | — | — | — | — | — | 4.447 | — | f32 | 3.94e-3 |
| cpu | layout:maxpool | q8_0 | — | — | — | — | — | — | — | — | — | 8.020 | — | f32 | 3.94e-3 |
| cpu | layout:avgpool | q8_0 | — | — | — | — | — | — | — | — | — | 9.535 | — | f32 | 3.79e-3 |
| cpu | layout:maxpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 57.820 | — | f32 | 3.91e-3 |
| cpu | layout:avgpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 11.419 | — | f32 | 9.77e-4 |
| cpu | layout:adaptive_maxpool | q8_0 | — | — | — | — | — | — | — | — | — | 147.010 | — | f32 | 3.88e-3 |
| cpu | layout:adaptive_avgpool | q8_0 | — | — | — | — | — | — | — | — | — | 145.586 | — | f32 | 2.32e-4 |
| cpu | layout:depthwise_conv | q8_0 | — | — | — | — | — | — | — | — | — | 171.131 | — | f32 | 1.95e-2 |
| cpu | layout:adaptive_maxpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 13738.767 | — | f32 | 2.85e-3 |
| cpu | layout:adaptive_avgpool_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 18.404 | — | f32 | 7.6e-6 |
| cpu | matmul_nt | q8_0 | — | — | — | — | — | — | — | — | — | 3.129 | — | s8dp blocks | 2.08e-1 |
| cpu | matmul_tn | q8_0 | — | — | — | — | — | — | — | — | — | 9.048 | — | f32 | 2.22e-1 |
| cpu | matmul_tt | q8_0 | — | — | — | — | — | — | — | — | — | 9.186 | — | f32 | 1.93e-1 |
| cpu | matmul_batched | q8_0 | — | — | — | — | — | — | — | — | — | 8.241 | — | f32 | 9.95e-2 |
| cpu | matmul_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 12.212 | — | s8dp blocks | 2.17e-1 |
| cpu | matmul_blocks | q8_0 | — | — | — | — | — | — | — | — | — | 3.350 | — | s8dp blocks | 2.08e-1 |
| cpu | attention | q8_0 | — | — | — | — | — | — | — | — | — | 64.650 | — | f32 | 2.44e-4 |
| cpu | attention_causal | q8_0 | — | — | — | — | — | — | — | — | — | 44.849 | — | f32 | 3.73e-3 |
| cpu | attention_fused | q8_0 | — | — | — | — | — | — | — | — | — | 81.537 | — | f32 | 2.44e-4 |
| cpu | attention_fused_causal | q8_0 | — | — | — | — | — | — | — | — | — | 45.468 | — | f32 | 3.73e-3 |
| cpu | attention_decode | q8_0 | — | — | — | — | — | — | — | — | — | 2.101 | — | f32 | 1.39e-4 |
| cpu | attention_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 17.682 | — | f32 | 1.56e-3 |
| cpu | optimizer:sgd_momentum | q8_0 | — | — | — | — | — | — | — | — | — | 9.079 | — | f32; + restore of p, m, v | 3.34e-3 |
| cpu | optimizer:adam | q8_0 | — | — | — | — | — | — | — | — | — | 12.640 | — | f32; + restore of p, m, v | 3.92e-3 |
| cpu | optimizer:adamw | q8_0 | — | — | — | — | — | — | — | — | — | 16.081 | — | f32; + restore of p, m, v | 3.93e-3 |
| cpu | conv | q8_0 | — | — | — | — | — | — | — | — | — | 7.825 | — | f32 | 1.67e-1 |
| cpu | conv_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 20.194 | — | f32 | 2.23e-1 |
| cpu | lstm_step | q8_0 | — | — | — | — | — | — | — | — | — | 26.065 | — | f32 | 2.12e-3 |
| cpu | gru_step | q8_0 | — | — | — | — | — | — | — | — | — | 19.956 | — | f32 | 3.58e-3 |
| cpu | lstm_step_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 60.820 | — | f32 | 2.42e-3 |
| cpu | gru_step_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 46.274 | — | f32 | 2.33e-3 |
| cpu | gru_seq | q8_0 | — | — | — | — | — | — | — | — | — | 31.337 | — | f32 | 3.34e-3 |
| cpu | gru_seq_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 116.469 | — | f32 | 2.64e-3 |
| cpu | topk_rows | q8_0 | — | — | — | — | — | — | — | — | — | 278.453 | — | f32 | 3.88e-3 |
| cpu | pick_rows | q8_0 | — | — | — | — | — | — | — | — | — | 0.193 | — | f32 | 1.57e-2 |
| cpu | sample_rows | q8_0 | — | — | — | — | — | — | — | — | — | 11.287 | — | f32 | 2e+0 |
| cpu | cross_entropy | q8_0 | — | — | — | — | — | — | — | — | — | 59.806 | — | f32 (log_softmax, pick, neg) | 1.25e-1 |
| cpu | ctc | q8_0 | — | — | — | — | — | — | — | — | — | 10.526 | — | f32 | 6.44e+0 |
| cpu | ctc_bwd | q8_0 | — | — | — | — | — | — | — | — | — | 11.706 | — | f32 | 3.91e-3 |
| cpu | softmax | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 5.664 | — | f32 | 8.37e-6 |
| cpu | rms_norm | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 5.398 | — | f32 | 4.85e-4 |
| cpu | add | q4_0 | 16777216 | — | — | — | — | — | — | — | — | 3.530 | — | f32 | 2.34e-1 |
| cpu | matmul | q4_0 | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.086 | — | f32 | 4.06e+0 |
| cpu | map:copy | q4_0 | — | — | — | — | — | — | — | — | — | 2.718 | — | f32 | 0e+0 |
| cpu | map:fill | q4_0 | — | — | — | — | — | — | — | — | — | 1.584 | — | f32 | 0e+0 |
| cpu | map:neg | q4_0 | — | — | — | — | — | — | — | — | — | 2.742 | — | f32 | 0e+0 |
| cpu | map:abs | q4_0 | — | — | — | — | — | — | — | — | — | 3.356 | — | f32 | 0e+0 |
| cpu | map:sign | q4_0 | — | — | — | — | — | — | — | — | — | 6.930 | — | f32 | 1.25e-1 |
| cpu | map:square | q4_0 | — | — | — | — | — | — | — | — | — | 3.325 | — | f32 | 1e+0 |
| cpu | map:relu | q4_0 | — | — | — | — | — | — | — | — | — | 3.506 | — | f32 | 1.88e-1 |
| cpu | map:add_scalar | q4_0 | — | — | — | — | — | — | — | — | — | 3.417 | — | f32 | 3.61e-1 |
| cpu | map:scale | q4_0 | — | — | — | — | — | — | — | — | — | 3.325 | — | f32 | 3.91e-3 |
| cpu | map:affine | q4_0 | — | — | — | — | — | — | — | — | — | 3.396 | — | f32 | 8.75e-1 |
| cpu | map:clamp | q4_0 | — | — | — | — | — | — | — | — | — | 3.359 | — | f32 | 1.25e-1 |
| cpu | map:add | q4_0 | — | — | — | — | — | — | — | — | — | 3.580 | — | f32 | 9.37e-1 |
| cpu | map:sub | q4_0 | — | — | — | — | — | — | — | — | — | 4.367 | — | f32 | 9.38e-1 |
| cpu | map:mul | q4_0 | — | — | — | — | — | — | — | — | — | 3.529 | — | f32 | 1.75e+0 |
| cpu | map:div | q4_0 | — | — | — | — | — | — | — | — | — | 3.614 | — | f32 | 5.06e-1 |
| cpu | map:max | q4_0 | — | — | — | — | — | — | — | — | — | 3.407 | — | f32 | 4.38e-1 |
| cpu | map:min | q4_0 | — | — | — | — | — | — | — | — | — | 3.405 | — | f32 | 4.38e-1 |
| cpu | map:axpy | q4_0 | — | — | — | — | — | — | — | — | — | 3.634 | — | f32 | 1.38e+0 |
| cpu | map:muladd | q4_0 | — | — | — | — | — | — | — | — | — | 4.109 | — | f32 | 2.18e+0 |
| cpu | map:where | q4_0 | — | — | — | — | — | — | — | — | — | 3.877 | — | f32 | 5e-1 |
| cpu | map:square_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.551 | — | f32 | 3.5e+0 |
| cpu | map:abs_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.492 | — | f32 | 5e-1 |
| cpu | map:relu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.480 | — | f32 | 4.38e-1 |
| cpu | map:clamp_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.494 | — | f32 | 4.38e-1 |
| cpu | map:where_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.275 | — | f32 | 4.38e-1 |
| cpu | map:sqrt | q4_0 | — | — | — | — | — | — | — | — | — | 2.632 | — | f32 | 1.21e-1 |
| cpu | map:inv_sqrt | q4_0 | — | — | — | — | — | — | — | — | — | 2.708 | — | f32 | 8.82e-2 |
| cpu | map:exp | q4_0 | — | — | — | — | — | — | — | — | — | 3.463 | — | f32 | 2.72e+0 |
| cpu | map:expm1 | q4_0 | — | — | — | — | — | — | — | — | — | 5.669 | — | f32 | 4.04e-2 |
| cpu | map:log | q4_0 | — | — | — | — | — | — | — | — | — | 3.874 | — | f32 | 9.81e-2 |
| cpu | map:sin | q4_0 | — | — | — | — | — | — | — | — | — | 7.429 | — | f32 | 1.25e-1 |
| cpu | map:cos | q4_0 | — | — | — | — | — | — | — | — | — | 6.544 | — | f32 | 1.25e-1 |
| cpu | map:tanh | q4_0 | — | — | — | — | — | — | — | — | — | 9.813 | — | f32 | 1.24e-1 |
| cpu | map:sigmoid | q4_0 | — | — | — | — | — | — | — | — | — | 5.198 | — | f32 | 6.13e-2 |
| cpu | map:erf | q4_0 | — | — | — | — | — | — | — | — | — | 32.702 | — | f32 | 1.25e-1 |
| cpu | map:gelu | q4_0 | — | — | — | — | — | — | — | — | — | 5.540 | — | f32 | 2.05e-1 |
| cpu | map:gelu_erf | q4_0 | — | — | — | — | — | — | — | — | — | 44.007 | — | f32 | 2.04e-1 |
| cpu | map:silu | q4_0 | — | — | — | — | — | — | — | — | — | 3.526 | — | f32 | 2.44e-1 |
| cpu | map:leaky_relu | q4_0 | — | — | — | — | — | — | — | — | — | 3.482 | — | f32 | 2.5e-1 |
| cpu | map:pow | q4_0 | — | — | — | — | — | — | — | — | — | 5.875 | — | f32 | 1.88e+0 |
| cpu | map:dropout | q4_0 | — | — | — | — | — | — | — | — | — | 4.194 | — | f32 | 5.48e-1 |
| cpu | map:gated_silu | q4_0 | — | — | — | — | — | — | — | — | — | 6.196 | — | f32 | 1.72e+0 |
| cpu | map:gated_gelu | q4_0 | — | — | — | — | — | — | — | — | — | 6.494 | — | f32 | 1.75e+0 |
| cpu | map:sqrt_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 4.804 | — | f32 | 3.36e-1 |
| cpu | map:inv_sqrt_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.572 | — | f32 | 8.24e-1 |
| cpu | map:exp_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 4.157 | — | f32 | 2.39e+1 |
| cpu | map:log_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.441 | — | f32 | 1.04e+0 |
| cpu | map:sin_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.047 | — | f32 | 5e-1 |
| cpu | map:cos_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.284 | — | f32 | 5e-1 |
| cpu | map:tanh_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.704 | — | f32 | 4.38e-1 |
| cpu | map:sigmoid_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.574 | — | f32 | 1.09e-1 |
| cpu | map:silu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.648 | — | f32 | 4.82e-1 |
| cpu | map:gelu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.910 | — | f32 | 4.94e-1 |
| cpu | map:gelu_erf_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 31.365 | — | f32 | 4.94e-1 |
| cpu | map:leaky_relu_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.468 | — | f32 | 4.38e-1 |
| cpu | map:pow_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 6.552 | — | f32 | 8.77e+0 |
| cpu | map:gated_silu_bwd_x | q4_0 | — | — | — | — | — | — | — | — | — | 6.226 | — | f32 | 1.9e+0 |
| cpu | map:gated_silu_bwd_y | q4_0 | — | — | — | — | — | — | — | — | — | 4.822 | — | f32 | 1.72e+0 |
| cpu | map:gated_gelu_bwd_x | q4_0 | — | — | — | — | — | — | — | — | — | 6.323 | — | f32 | 1.77e+0 |
| cpu | map:gated_gelu_bwd_y | q4_0 | — | — | — | — | — | — | — | — | — | 5.009 | — | f32 | 1.75e+0 |
| cpu | sum:whole | q4_0 | — | — | — | — | — | — | — | — | — | 54.582 | — | f32 | 6.1e-5 |
| cpu | sum:rows | q4_0 | — | — | — | — | — | — | — | — | — | 54.775 | — | f32 | 0e+0 |
| cpu | sum:cols | q4_0 | — | — | — | — | — | — | — | — | — | 55.584 | — | f32 | 0e+0 |
| cpu | sumsq:rows | q4_0 | — | — | — | — | — | — | — | — | — | 54.834 | — | f32 | 3.97e-4 |
| cpu | max:whole | q4_0 | — | — | — | — | — | — | — | — | — | 0.922 | — | f32 | 0e+0 |
| cpu | max:rows | q4_0 | — | — | — | — | — | — | — | — | — | 1.782 | — | f32 | 0e+0 |
| cpu | max:cols | q4_0 | — | — | — | — | — | — | — | — | — | 3.313 | — | f32 | 0e+0 |
| cpu | min:rows | q4_0 | — | — | — | — | — | — | — | — | — | 2.480 | — | f32 | 0e+0 |
| cpu | argmax:rows | q4_0 | — | — | — | — | — | — | — | — | — | 1.989 | — | s32 index | 0e+0 |
| cpu | argmax:cols | q4_0 | — | — | — | — | — | — | — | — | — | 1.773 | — | s32 index | 0e+0 |
| cpu | maprow:add | q4_0 | — | — | — | — | — | — | — | — | — | 3.110 | — | f32 | 2.34e-1 |
| cpu | maprow:mul | q4_0 | — | — | — | — | — | — | — | — | — | 3.592 | — | f32 | 1.09e-1 |
| cpu | transpose | q4_0 | — | — | — | — | — | — | — | — | — | 5.455 | — | f32 | 1.25e-1 |
| cpu | gemv | q4_0 | — | — | — | — | — | — | — | — | — | 1.952 | — | s8dp blocks | 1.03e+1 |
| cpu | gemv_t | q4_0 | — | — | — | — | — | — | — | — | — | 224.006 | — | f32 | 1.14e+1 |
| cpu | row:log_softmax | q4_0 | — | — | — | — | — | — | — | — | — | 3.930 | — | f32 | 8.92e-1 |
| cpu | row:softmax_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 4.880 | — | f32 | 1.89e+0 |
| cpu | row:log_softmax_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 5.833 | — | f32 | 8.32e+0 |
| cpu | row:rms_norm_affine | q4_0 | — | — | — | — | — | — | — | — | — | 3.674 | — | f32 | 3.89e-1 |
| cpu | row:rms_norm_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.768 | — | f32 | 9.72e-2 |
| cpu | row:layer_norm | q4_0 | — | — | — | — | — | — | — | — | — | 3.163 | — | f32 | 4.9e-1 |
| cpu | row:layer_norm_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.861 | — | f32 | 9.7e-2 |
| cpu | row:weight_norm | q4_0 | — | — | — | — | — | — | — | — | — | 2.703 | — | f32 | 1.53e-5 |
| cpu | row:rope | q4_0 | — | — | — | — | — | — | — | — | — | 6.278 | — | f32 | 6.94e-1 |
| cpu | row:scale_mask_softmax | q4_0 | — | — | — | — | — | — | — | — | — | 4.629 | — | f32 | 3.11e-2 |
| cpu | row:add_rms_norm | q4_0 | — | — | — | — | — | — | — | — | — | 6.086 | — | f32 | 5.78e-1 |
| cpu | row:add_layer_norm | q4_0 | — | — | — | — | — | — | — | — | — | 6.155 | — | f32 | 6.37e-1 |
| cpu | row:cross_entropy_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 3.960 | — | f32 | 2.14e-3 |
| cpu | norm_params_bwd:rms | q4_0 | — | — | — | — | — | — | — | — | — | 3.073 | — | f32 | 2.74e-4 |
| cpu | norm_params_bwd:layer | q4_0 | — | — | — | — | — | — | — | — | — | 3.041 | — | f32 | 3.15e-4 |
| cpu | layout:strided | q4_0 | — | — | — | — | — | — | — | — | — | 37.520 | — | f32 | 1.25e-1 |
| cpu | layout:one_hot | q4_0 | — | — | — | — | — | — | — | — | — | 1.614 | — | f32 | 0e+0 |
| cpu | layout:gather_rows | q4_0 | — | — | — | — | — | — | — | — | — | 2.272 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | q4_0 | — | — | — | — | — | — | — | — | — | 36.152 | — | f32 | 1.25e-1 |
| cpu | layout:scatter_add_rows | q4_0 | — | — | — | — | — | — | — | — | — | 87.141 | — | f32 | 4.39e-1 |
| cpu | layout:im2col | q4_0 | — | — | — | — | — | — | — | — | — | 18.945 | — | f32 | 1.25e-1 |
| cpu | layout:col2im | q4_0 | — | — | — | — | — | — | — | — | — | 51.294 | — | f32 | 6.46e-1 |
| cpu | layout:interpolate_nearest | q4_0 | — | — | — | — | — | — | — | — | — | 34.807 | — | f32 | 1.09e-1 |
| cpu | layout:interpolate_bilinear | q4_0 | — | — | — | — | — | — | — | — | — | 130.866 | — | f32 | 1.12e-1 |
| cpu | layout:shift | q4_0 | — | — | — | — | — | — | — | — | — | 30.697 | — | f32 | 1.25e-1 |
| cpu | layout:pad2d | q4_0 | — | — | — | — | — | — | — | — | — | 2.812 | — | f32 | 1.25e-1 |
| cpu | layout:maxpool | q4_0 | — | — | — | — | — | — | — | — | — | 8.702 | — | f32 | 1.09e-1 |
| cpu | layout:avgpool | q4_0 | — | — | — | — | — | — | — | — | — | 8.536 | — | f32 | 1.06e-1 |
| cpu | layout:maxpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 49.759 | — | f32 | 1.09e-1 |
| cpu | layout:avgpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 10.050 | — | f32 | 2.73e-2 |
| cpu | layout:adaptive_maxpool | q4_0 | — | — | — | — | — | — | — | — | — | 135.532 | — | f32 | 1.17e-2 |
| cpu | layout:adaptive_avgpool | q4_0 | — | — | — | — | — | — | — | — | — | 131.701 | — | f32 | 4.53e-3 |
| cpu | layout:depthwise_conv | q4_0 | — | — | — | — | — | — | — | — | — | 148.913 | — | f32 | 4.7e-1 |
| cpu | layout:adaptive_maxpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 12037.421 | — | f32 | 1.09e-1 |
| cpu | layout:adaptive_avgpool_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 16.579 | — | f32 | 1.35e-4 |
| cpu | matmul_nt | q4_0 | — | — | — | — | — | — | — | — | — | 3.086 | — | s8dp blocks | 4.02e+0 |
| cpu | matmul_tn | q4_0 | — | — | — | — | — | — | — | — | — | 8.775 | — | f32 | 4.36e+0 |
| cpu | matmul_tt | q4_0 | — | — | — | — | — | — | — | — | — | 8.900 | — | f32 | 4.29e+0 |
| cpu | matmul_batched | q4_0 | — | — | — | — | — | — | — | — | — | 7.872 | — | f32 | 2.28e+0 |
| cpu | matmul_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 12.110 | — | s8dp blocks | 4.55e+0 |
| cpu | matmul_blocks | q4_0 | — | — | — | — | — | — | — | — | — | 2.660 | — | s8dp blocks | 4.02e+0 |
| cpu | attention | q4_0 | — | — | — | — | — | — | — | — | — | 52.830 | — | f32 | 6.18e-3 |
| cpu | attention_causal | q4_0 | — | — | — | — | — | — | — | — | — | 45.848 | — | f32 | 1.06e-1 |
| cpu | attention_fused | q4_0 | — | — | — | — | — | — | — | — | — | 82.441 | — | f32 | 6.18e-3 |
| cpu | attention_fused_causal | q4_0 | — | — | — | — | — | — | — | — | — | 43.311 | — | f32 | 1.06e-1 |
| cpu | attention_decode | q4_0 | — | — | — | — | — | — | — | — | — | 2.039 | — | f32 | 3e-3 |
| cpu | attention_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 15.120 | — | f32 | 3.43e-2 |
| cpu | optimizer:sgd_momentum | q4_0 | — | — | — | — | — | — | — | — | — | 6.857 | — | f32; + restore of p, m, v | 2.92e-3 |
| cpu | optimizer:adam | q4_0 | — | — | — | — | — | — | — | — | — | 10.706 | — | f32; + restore of p, m, v | 3.16e-2 |
| cpu | optimizer:adamw | q4_0 | — | — | — | — | — | — | — | — | — | 10.528 | — | f32; + restore of p, m, v | 3.16e-2 |
| cpu | conv | q4_0 | — | — | — | — | — | — | — | — | — | 6.554 | — | f32 | 3.39e+0 |
| cpu | conv_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 17.861 | — | f32 | 4.35e+0 |
| cpu | lstm_step | q4_0 | — | — | — | — | — | — | — | — | — | 24.555 | — | f32 | 5.17e-2 |
| cpu | gru_step | q4_0 | — | — | — | — | — | — | — | — | — | 15.706 | — | f32 | 9.63e-2 |
| cpu | lstm_step_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 50.975 | — | f32 | 5.03e-2 |
| cpu | gru_step_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 35.910 | — | f32 | 4.61e-2 |
| cpu | gru_seq | q4_0 | — | — | — | — | — | — | — | — | — | 25.982 | — | f32 | 1e-1 |
| cpu | gru_seq_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 106.265 | — | f32 | 5.31e-2 |
| cpu | topk_rows | q4_0 | — | — | — | — | — | — | — | — | — | 264.549 | — | f32 | 3.91e-3 |
| cpu | pick_rows | q4_0 | — | — | — | — | — | — | — | — | — | 0.165 | — | f32 | 4.98e-1 |
| cpu | sample_rows | q4_0 | — | — | — | — | — | — | — | — | — | 10.644 | — | f32 | 3e+0 |
| cpu | cross_entropy | q4_0 | — | — | — | — | — | — | — | — | — | 31.037 | — | f32 (log_softmax, pick, neg) | 1.96e+0 |
| cpu | ctc | q4_0 | — | — | — | — | — | — | — | — | — | 9.929 | — | f32 | 1.07e+2 |
| cpu | ctc_bwd | q4_0 | — | — | — | — | — | — | — | — | — | 14.316 | — | f32 | 5.63e-2 |
| cpu | softmax | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 4.072 | — | f32 | 1.11e-4 |
| cpu | rms_norm | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 2.661 | — | f32 | 4.89e-4 |
| cpu | add | q4_k | 16777216 | — | — | — | — | — | — | — | — | 5.031 | — | f32 | 1.33e-1 |
| cpu | matmul | q4_k | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.777 | — | f32 | 2.71e+0 |
| cpu | map:copy | q4_k | — | — | — | — | — | — | — | — | — | 4.607 | — | f32 | 0e+0 |
| cpu | map:fill | q4_k | — | — | — | — | — | — | — | — | — | 4.215 | — | f32 | 7.32e-4 |
| cpu | map:neg | q4_k | — | — | — | — | — | — | — | — | — | 4.634 | — | f32 | 3.24e-2 |
| cpu | map:abs | q4_k | — | — | — | — | — | — | — | — | — | 4.615 | — | f32 | 1.36e-1 |
| cpu | map:sign | q4_k | — | — | — | — | — | — | — | — | — | 4.741 | — | f32 | 6.61e-2 |
| cpu | map:square | q4_k | — | — | — | — | — | — | — | — | — | 4.622 | — | f32 | 5.55e-1 |
| cpu | map:relu | q4_k | — | — | — | — | — | — | — | — | — | 4.642 | — | f32 | 1.36e-1 |
| cpu | map:add_scalar | q4_k | — | — | — | — | — | — | — | — | — | 5.553 | — | f32 | 1.99e-1 |
| cpu | map:scale | q4_k | — | — | — | — | — | — | — | — | — | 5.556 | — | f32 | 1.11e-2 |
| cpu | map:affine | q4_k | — | — | — | — | — | — | — | — | — | 4.630 | — | f32 | 5.47e-2 |
| cpu | map:clamp | q4_k | — | — | — | — | — | — | — | — | — | 4.848 | — | f32 | 1e-1 |
| cpu | map:add | q4_k | — | — | — | — | — | — | — | — | — | 5.062 | — | f32 | 5.31e-1 |
| cpu | map:sub | q4_k | — | — | — | — | — | — | — | — | — | 5.137 | — | f32 | 5.31e-1 |
| cpu | map:mul | q4_k | — | — | — | — | — | — | — | — | — | 5.040 | — | f32 | 1.07e+0 |
| cpu | map:div | q4_k | — | — | — | — | — | — | — | — | — | 6.268 | — | f32 | 2.7e-1 |
| cpu | map:max | q4_k | — | — | — | — | — | — | — | — | — | 6.173 | — | f32 | 2.66e-1 |
| cpu | map:min | q4_k | — | — | — | — | — | — | — | — | — | 5.169 | — | f32 | 2.66e-1 |
| cpu | map:axpy | q4_k | — | — | — | — | — | — | — | — | — | 5.085 | — | f32 | 7.81e-1 |
| cpu | map:muladd | q4_k | — | — | — | — | — | — | — | — | — | 5.557 | — | f32 | 1.32e+0 |
| cpu | map:where | q4_k | — | — | — | — | — | — | — | — | — | 5.554 | — | f32 | 2.68e-1 |
| cpu | map:square_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.094 | — | f32 | 2.15e+0 |
| cpu | map:abs_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.151 | — | f32 | 2.73e-1 |
| cpu | map:relu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.063 | — | f32 | 2.65e-1 |
| cpu | map:clamp_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.076 | — | f32 | 2.65e-1 |
| cpu | map:where_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.066 | — | f32 | 2.65e-1 |
| cpu | map:sqrt | q4_k | — | — | — | — | — | — | — | — | — | 4.767 | — | f32 | 6.6e-2 |
| cpu | map:inv_sqrt | q4_k | — | — | — | — | — | — | — | — | — | 4.616 | — | f32 | 5.22e-2 |
| cpu | map:exp | q4_k | — | — | — | — | — | — | — | — | — | 5.194 | — | f32 | 1.96e+0 |
| cpu | map:expm1 | q4_k | — | — | — | — | — | — | — | — | — | 7.419 | — | f32 | 3.47e-2 |
| cpu | map:log | q4_k | — | — | — | — | — | — | — | — | — | 5.683 | — | f32 | 6.72e-2 |
| cpu | map:sin | q4_k | — | — | — | — | — | — | — | — | — | 8.169 | — | f32 | 6.67e-2 |
| cpu | map:cos | q4_k | — | — | — | — | — | — | — | — | — | 5.781 | — | f32 | 6.67e-2 |
| cpu | map:tanh | q4_k | — | — | — | — | — | — | — | — | — | 5.865 | — | f32 | 6.66e-2 |
| cpu | map:sigmoid | q4_k | — | — | — | — | — | — | — | — | — | 5.285 | — | f32 | 3.28e-2 |
| cpu | map:erf | q4_k | — | — | — | — | — | — | — | — | — | 17.471 | — | f32 | 6.67e-2 |
| cpu | map:gelu | q4_k | — | — | — | — | — | — | — | — | — | 5.611 | — | f32 | 1.41e-1 |
| cpu | map:gelu_erf | q4_k | — | — | — | — | — | — | — | — | — | 29.250 | — | f32 | 1.42e-1 |
| cpu | map:silu | q4_k | — | — | — | — | — | — | — | — | — | 5.315 | — | f32 | 1.43e-1 |
| cpu | map:leaky_relu | q4_k | — | — | — | — | — | — | — | — | — | 4.719 | — | f32 | 1.48e-1 |
| cpu | map:pow | q4_k | — | — | — | — | — | — | — | — | — | 7.359 | — | f32 | 1.04e+0 |
| cpu | map:dropout | q4_k | — | — | — | — | — | — | — | — | — | 5.067 | — | f32 | 3.32e-1 |
| cpu | map:gated_silu | q4_k | — | — | — | — | — | — | — | — | — | 5.761 | — | f32 | 1.05e+0 |
| cpu | map:gated_gelu | q4_k | — | — | — | — | — | — | — | — | — | 5.920 | — | f32 | 1.07e+0 |
| cpu | map:sqrt_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.058 | — | f32 | 1.91e-1 |
| cpu | map:inv_sqrt_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.154 | — | f32 | 4.15e-1 |
| cpu | map:exp_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.615 | — | f32 | 1.51e+1 |
| cpu | map:log_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.112 | — | f32 | 5.64e-1 |
| cpu | map:sin_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.284 | — | f32 | 2.71e-1 |
| cpu | map:cos_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.566 | — | f32 | 2.69e-1 |
| cpu | map:tanh_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.903 | — | f32 | 2.66e-1 |
| cpu | map:sigmoid_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.891 | — | f32 | 6.64e-2 |
| cpu | map:silu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.961 | — | f32 | 2.93e-1 |
| cpu | map:gelu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 7.317 | — | f32 | 3e-1 |
| cpu | map:gelu_erf_bwd | q4_k | — | — | — | — | — | — | — | — | — | 32.570 | — | f32 | 3e-1 |
| cpu | map:leaky_relu_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.090 | — | f32 | 2.64e-1 |
| cpu | map:pow_bwd | q4_k | — | — | — | — | — | — | — | — | — | 8.096 | — | f32 | 5.3e+0 |
| cpu | map:gated_silu_bwd_x | q4_k | — | — | — | — | — | — | — | — | — | 7.391 | — | f32 | 1.12e+0 |
| cpu | map:gated_silu_bwd_y | q4_k | — | — | — | — | — | — | — | — | — | 6.257 | — | f32 | 1.05e+0 |
| cpu | map:gated_gelu_bwd_x | q4_k | — | — | — | — | — | — | — | — | — | 7.675 | — | f32 | 1.07e+0 |
| cpu | map:gated_gelu_bwd_y | q4_k | — | — | — | — | — | — | — | — | — | 6.324 | — | f32 | 1.07e+0 |
| cpu | sum:whole | q4_k | — | — | — | — | — | — | — | — | — | 56.902 | — | f32 | 1.61e-4 |
| cpu | sum:rows | q4_k | — | — | — | — | — | — | — | — | — | 56.675 | — | f32 | 9.54e-6 |
| cpu | sum:cols | q4_k | — | — | — | — | — | — | — | — | — | 58.935 | — | f32 | 2.17e-4 |
| cpu | sumsq:rows | q4_k | — | — | — | — | — | — | — | — | — | 56.783 | — | f32 | 3.36e-4 |
| cpu | max:whole | q4_k | — | — | — | — | — | — | — | — | — | 1.940 | — | f32 | 0e+0 |
| cpu | max:rows | q4_k | — | — | — | — | — | — | — | — | — | 2.053 | — | f32 | 0e+0 |
| cpu | max:cols | q4_k | — | — | — | — | — | — | — | — | — | 3.974 | — | f32 | 0e+0 |
| cpu | min:rows | q4_k | — | — | — | — | — | — | — | — | — | 1.986 | — | f32 | 0e+0 |
| cpu | argmax:rows | q4_k | — | — | — | — | — | — | — | — | — | 1.925 | — | s32 index | 0e+0 |
| cpu | argmax:cols | q4_k | — | — | — | — | — | — | — | — | — | 2.444 | — | s32 index | 0e+0 |
| cpu | maprow:add | q4_k | — | — | — | — | — | — | — | — | — | 5.465 | — | f32 | 1.32e-1 |
| cpu | maprow:mul | q4_k | — | — | — | — | — | — | — | — | — | 5.469 | — | f32 | 6.65e-2 |
| cpu | transpose | q4_k | — | — | — | — | — | — | — | — | — | 24.786 | — | f32 | 1e+0 |
| cpu | gemv | q4_k | — | — | — | — | — | — | — | — | — | 226.217 | — | f32 | 6.29e+0 |
| cpu | gemv_t | q4_k | — | — | — | — | — | — | — | — | — | 232.515 | — | f32 | 6.57e+0 |
| cpu | row:log_softmax | q4_k | — | — | — | — | — | — | — | — | — | 6.081 | — | f32 | 1.14e-1 |
| cpu | row:softmax_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.060 | — | f32 | 2.49e+0 |
| cpu | row:log_softmax_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.320 | — | f32 | 4.44e+0 |
| cpu | row:rms_norm_affine | q4_k | — | — | — | — | — | — | — | — | — | 4.598 | — | f32 | 2.33e-1 |
| cpu | row:rms_norm_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.170 | — | f32 | 5.79e-2 |
| cpu | row:layer_norm | q4_k | — | — | — | — | — | — | — | — | — | 4.668 | — | f32 | 2.86e-1 |
| cpu | row:layer_norm_bwd | q4_k | — | — | — | — | — | — | — | — | — | 5.353 | — | f32 | 5.79e-2 |
| cpu | row:weight_norm | q4_k | — | — | — | — | — | — | — | — | — | 5.477 | — | f32 | 4.3e-5 |
| cpu | row:rope | q4_k | — | — | — | — | — | — | — | — | — | 7.750 | — | f32 | 3.73e-1 |
| cpu | row:scale_mask_softmax | q4_k | — | — | — | — | — | — | — | — | — | 6.341 | — | f32 | 1.75e-2 |
| cpu | row:add_rms_norm | q4_k | — | — | — | — | — | — | — | — | — | 9.162 | — | f32 | 3.11e-1 |
| cpu | row:add_layer_norm | q4_k | — | — | — | — | — | — | — | — | — | 9.264 | — | f32 | 3.56e-1 |
| cpu | row:cross_entropy_bwd | q4_k | — | — | — | — | — | — | — | — | — | 6.037 | — | f32 | 2.52e-3 |
| cpu | norm_params_bwd:rms | q4_k | — | — | — | — | — | — | — | — | — | 3.820 | — | f32 | 2.8e-4 |
| cpu | norm_params_bwd:layer | q4_k | — | — | — | — | — | — | — | — | — | 3.948 | — | f32 | 2.65e-4 |
| cpu | layout:strided | q4_k | — | — | — | — | — | — | — | — | — | 159.167 | — | f32 | 6.72e-2 |
| cpu | layout:one_hot | q4_k | — | — | — | — | — | — | — | — | — | 4.363 | — | f32 | 3.57e-4 |
| cpu | layout:gather_rows | q4_k | — | — | — | — | — | — | — | — | — | 4.446 | — | f32 | 0e+0 |
| cpu | layout:gather_cols | q4_k | — | — | — | — | — | — | — | — | — | 149.465 | — | f32 | 6.72e-2 |
| cpu | layout:scatter_add_rows | q4_k | — | — | — | — | — | — | — | — | — | 180.432 | — | f32 | 2.77e-1 |
| cpu | layout:im2col | q4_k | — | — | — | — | — | — | — | — | — | 87.492 | — | f32 | 6.67e-2 |
| cpu | layout:col2im | q4_k | — | — | — | — | — | — | — | — | — | 99.963 | — | f32 | 4.18e-1 |
| cpu | layout:interpolate_nearest | q4_k | — | — | — | — | — | — | — | — | — | 163.332 | — | f32 | 6.2e-2 |
| cpu | layout:interpolate_bilinear | q4_k | — | — | — | — | — | — | — | — | — | 586.920 | — | f32 | 6.66e-2 |
| cpu | layout:shift | q4_k | — | — | — | — | — | — | — | — | — | 158.628 | — | f32 | 6.72e-2 |
| cpu | layout:pad2d | q4_k | — | — | — | — | — | — | — | — | — | 4.485 | — | f32 | 6.7e-2 |
| cpu | layout:maxpool | q4_k | — | — | — | — | — | — | — | — | — | 38.368 | — | f32 | 6.36e-2 |
| cpu | layout:avgpool | q4_k | — | — | — | — | — | — | — | — | — | 39.655 | — | f32 | 5.84e-2 |
| cpu | layout:maxpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 203.073 | — | f32 | 6.63e-2 |
| cpu | layout:avgpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 34.708 | — | f32 | 1.55e-2 |
| cpu | layout:adaptive_maxpool | q4_k | — | — | — | — | — | — | — | — | — | 568.769 | — | f32 | 2.97e-2 |
| cpu | layout:adaptive_avgpool | q4_k | — | — | — | — | — | — | — | — | — | 568.130 | — | f32 | 3.06e-3 |
| cpu | layout:depthwise_conv | q4_k | — | — | — | — | — | — | — | — | — | 678.618 | — | f32 | 2.95e-1 |
| cpu | layout:adaptive_maxpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 52113.482 | — | f32 | 5.43e-2 |
| cpu | layout:adaptive_avgpool_bwd | q4_k | — | — | — | — | — | — | — | — | — | 45.936 | — | f32 | 6.65e-5 |
| cpu | matmul_nt | q4_k | — | — | — | — | — | — | — | — | — | 9.427 | — | f32 | 2.73e+0 |
| cpu | matmul_tn | q4_k | — | — | — | — | — | — | — | — | — | 9.741 | — | f32 | 2.8e+0 |
| cpu | matmul_tt | q4_k | — | — | — | — | — | — | — | — | — | 10.277 | — | f32 | 2.66e+0 |
| cpu | matmul_batched | q4_k | — | — | — | — | — | — | — | — | — | 8.312 | — | f32 | 1.36e+0 |
| cpu | matmul_bwd | q4_k | — | — | — | — | — | — | — | — | — | 19.327 | — | f32 | 2.73e+0 |
| cpu | attention | q4_k | — | — | — | — | — | — | — | — | — | 55.099 | — | f32 | 3.47e-3 |
| cpu | attention_causal | q4_k | — | — | — | — | — | — | — | — | — | 44.892 | — | f32 | 5.93e-2 |
| cpu | attention_fused | q4_k | — | — | — | — | — | — | — | — | — | 81.701 | — | f32 | 3.47e-3 |
| cpu | attention_fused_causal | q4_k | — | — | — | — | — | — | — | — | — | 45.252 | — | f32 | 5.93e-2 |
| cpu | attention_decode | q4_k | — | — | — | — | — | — | — | — | — | 3.654 | — | f32 | 2.05e-3 |
| cpu | attention_bwd | q4_k | — | — | — | — | — | — | — | — | — | 18.661 | — | f32 | 2.51e-2 |
| cpu | optimizer:sgd_momentum | q4_k | — | — | — | — | — | — | — | — | — | 8.390 | — | f32; + restore of p, m, v | 5.87e-3 |
| cpu | optimizer:adam | q4_k | — | — | — | — | — | — | — | — | — | 13.262 | — | f32; + restore of p, m, v | 2.26e-2 |
| cpu | optimizer:adamw | q4_k | — | — | — | — | — | — | — | — | — | 14.083 | — | f32; + restore of p, m, v | 2.26e-2 |
| cpu | conv | q4_k | — | — | — | — | — | — | — | — | — | 8.071 | — | f32 | 2.08e+0 |
| cpu | conv_bwd | q4_k | — | — | — | — | — | — | — | — | — | 18.053 | — | f32 | 2.95e+0 |
| cpu | lstm_step | q4_k | — | — | — | — | — | — | — | — | — | 23.954 | — | f32 | 2.95e-2 |
| cpu | gru_step | q4_k | — | — | — | — | — | — | — | — | — | 14.571 | — | f32 | 5.6e-2 |
| cpu | lstm_step_bwd | q4_k | — | — | — | — | — | — | — | — | — | 59.980 | — | f32 | 3.21e-2 |
| cpu | gru_step_bwd | q4_k | — | — | — | — | — | — | — | — | — | 45.604 | — | f32 | 3.05e-2 |
| cpu | gru_seq | q4_k | — | — | — | — | — | — | — | — | — | 20.433 | — | f32 | 5.58e-2 |
