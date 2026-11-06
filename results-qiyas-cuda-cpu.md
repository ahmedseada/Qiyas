# Qiyas bench

2026-11-06 17:02. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | add | float64 | 16777216 | 124.425 | — | add_f32; converted through the host | 5.96e-8 | 0.999 | 124.61 | add_f64 | 0e+0 | 0.520 | 239.19 | f64 | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 31.968 | — | gemm128_f32; converted through the host | 1.62e-4 | 23.115 | 1.38 | gemm64_f64 | 2.59e-13 | 4.178 | 7.65 | f64x | 0e+0 |
| cuda | softmax | float64 | 4096x4096 | 81.900 | — | softmax_f32; converted through the host | 5.29e-10 | 0.394 | 207.97 | softmax_f64 | 1.88e-10 | 1.741 | 47.04 | f64 | 1.13e-17 |
| cuda | rms_norm | float64 | 4096x4096 | 83.747 | — | rms_norm_f32; converted through the host | 3.1e-7 | 0.370 | 226.10 | rms_norm_f64 | 1.24e-14 | 0.551 | 152.10 | f64 | 1.07e-14 |
| cuda | add | float32 | 16777216 | 0.262 | — | add_f32 | 5.96e-8 | 0.264 | 0.99 | add_f32 | 5.96e-8 | 0.263 | 1.00 | f32 | 5.96e-8 |
| cuda | matmul | float32 | 2048x2048x2048 | 0.986 | — | gemm128_f32 | 1.62e-4 | 0.840 | 1.17 | gemm128_f32 | 1.62e-4 | 0.684 | 1.44 | f32 | 1.62e-4 |
| cuda | softmax | float32 | 4096x4096 | 0.181 | — | softmax_f32 | 5.29e-10 | 0.181 | 1.00 | softmax_f32 | 5.11e-10 | 0.177 | 1.02 | f32 | 4.52e-10 |
| cuda | rms_norm | float32 | 4096x4096 | 0.183 | — | rms_norm_f32 | 3.1e-7 | 0.181 | 1.01 | rms_norm_f32 | 2.93e-7 | 0.179 | 1.02 | f32 | 2.97e-7 |
| cuda | add | float16 | 16777216 | 0.664 | — | add_f32; converted on the device | 4.88e-4 | 0.140 | 4.73 | add_f16 | 4.88e-4 | 0.125 | 5.31 | f32 | 4.88e-4 |
| cuda | matmul | float16 | 2048x2048x2048 | 1.101 | — | gemm64_f32; converted on the device | 3.09e-2 | 0.657 | 1.68 | gemm128_f16 | 1.51e+0 | 0.243 | 4.54 | f16mma | 3.09e-2 |
| cuda | softmax | float16 | 4096x4096 | 0.443 | — | softmax_f32; converted on the device | 9.53e-7 | 0.092 | 4.81 | softmax_f16 | 3.3e-6 | 0.075 | 5.94 | f32 | 9.54e-7 |
| cuda | rms_norm | float16 | 4096x4096 | 0.445 | — | rms_norm_f32; converted on the device | 4.88e-4 | 0.090 | 4.93 | rms_norm_f16 | 1.64e-3 | 0.081 | 5.49 | f32 | 4.88e-4 |
| cuda | add | bfloat16 | 16777216 | 0.660 | — | add_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.125 | 5.29 | f32 | 3.91e-3 |
| cuda | matmul | bfloat16 | 2048x2048x2048 | 1.096 | — | gemm64_f32; converted on the device | 2.5e-1 | — | — | — | — | 0.243 | 4.52 | bf16mma | 2.5e-1 |
| cuda | softmax | bfloat16 | 4096x4096 | 0.445 | — | softmax_f32; converted on the device | 7.63e-6 | — | — | — | — | 0.084 | 5.31 | f32 | 7.63e-6 |
| cuda | rms_norm | bfloat16 | 4096x4096 | 0.442 | — | rms_norm_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.081 | 5.47 | f32 | 3.91e-3 |
| cuda | add | float8_e4m3 | 16777216 | 507.989 | — | add_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.051 | 9921.66 | f32 | 6.25e-2 |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | 136.133 | — | gemm64_f32; converted through the host | 3.99e+0 | — | — | — | — | 0.278 | 489.69 | f16mma | 3.99e+0 |
| cuda | softmax | float8_e4m3 | 4096x4096 | 107.326 | — | softmax_f32; converted through the host | 9.76e-4 | — | — | — | — | 0.067 | 1613.92 | f32 | 9.76e-4 |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | 421.088 | — | rms_norm_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.046 | 9134.24 | f32 | 6.25e-2 |
| cuda | add | float8_e5m2 | 16777216 | 478.204 | — | add_f32; converted through the host | 1.25e-1 | — | — | — | — | 1.077 | 444.06 | f32 | 1.25e-1 |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | 113.603 | — | gemm64_f32; converted through the host | 7.71e+0 | — | — | — | — | 0.264 | 429.99 | f16mma | 7.71e+0 |
| cuda | softmax | float8_e5m2 | 4096x4096 | 301.963 | — | softmax_f32; converted through the host | 1.62e-4 | — | — | — | — | 0.069 | 4401.79 | f32 | 1.62e-4 |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | 339.917 | — | rms_norm_f32; converted through the host | 1.08e-1 | — | — | — | — | 0.046 | 7341.62 | f32 | 1.08e-1 |
| cuda | add | int64 | 16777216 | 124.529 | — | add_f32; converted through the host | 0e+0 | 0.520 | 239.34 | add_s64 | 0e+0 | 0.520 | 239.39 | s64 | 0e+0 |
| cuda | matmul | int64 | 2048x2048x2048 | 30.643 | — | gemm128_f32; converted through the host | 0e+0 | 4.438 | 6.91 | gemm64_s64 | 0e+0 | 0.357 | 85.93 | s8mma | 0e+0 |
| cuda | add | int32 | 16777216 | 0.788 | — | add_f32; converted on the device | 0e+0 | 0.262 | 3.01 | add_s32 | 0e+0 | 0.261 | 3.02 | s32 | 0e+0 |
| cuda | matmul | int32 | 2048x2048x2048 | 1.139 | — | gemm128_f32; converted on the device | 0e+0 | 1.093 | 1.04 | gemm128_s32 | 0e+0 | 0.202 | 5.63 | s8mma | 0e+0 |
| cuda | add | int16 | 16777216 | 69.511 | — | add_f32; converted through the host | 0e+0 | 0.140 | 496.86 | add_s16 | 0e+0 | 0.125 | 555.65 | s16 | 0e+0 |
| cuda | matmul | int16 | 2048x2048x2048 | 19.441 | — | gemm128_f32; converted through the host | 0e+0 | 1.588 | 12.25 | gemm128_s16 | 0e+0 | 0.174 | 111.66 | s8mma | 0e+0 |
| cuda | add | int8 | 16777216 | 69.664 | — | add_f32; converted through the host | 0e+0 | 0.078 | 895.43 | add_s8 | 0e+0 | 0.023 | 3096.18 | s8 | 0e+0 |
| cuda | matmul | int8 | 2048x2048x2048 | 23.041 | — | gemm128_f32; converted through the host | 3.23e+2 | 1.549 | 14.87 | gemm128_s8 | 5.12e+2 | 0.094 | 244.60 | s8mma | 3.23e+2 |
| cuda | add | int4 | 16777216 | 79.477 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4319.41 | s8 | 0e+0 |
| cuda | matmul | int4 | 2048x2048x2048 | 38.454 | — | gemm128_f32; converted through the host | 4.43e+2 | — | — | — | — | 0.103 | 371.90 | s8mma | 4.43e+2 |
| cuda | add | int2 | 16777216 | 78.877 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4286.80 | s8 | 0e+0 |
| cuda | matmul | int2 | 2048x2048x2048 | 19.498 | — | gemm64_f32; converted through the host | 6.16e+2 | — | — | — | — | 0.111 | 176.45 | s8mma | 6.16e+2 |
| cuda | add | uint64 | 16777216 | 141.532 | — | add_f32; converted through the host | 0e+0 | 1.001 | 141.41 | add_u64 | 0e+0 | 0.521 | 271.86 | u64 | 0e+0 |
| cuda | matmul | uint64 | 2048x2048x2048 | 37.906 | — | gemm128_f32; converted through the host | 0e+0 | 4.369 | 8.68 | gemm64_u64 | 0e+0 | 0.355 | 106.72 | u8mma | 0e+0 |
| cuda | add | uint32 | 16777216 | 106.432 | — | add_f32; converted through the host | 0e+0 | 0.261 | 407.94 | add_u32 | 0e+0 | 0.262 | 406.39 | u32 | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 24.507 | — | gemm64_f32; converted through the host | 0e+0 | 1.095 | 22.39 | gemm128_u32 | 0e+0 | 0.209 | 117.48 | u8mma | 0e+0 |
| cuda | add | uint16 | 16777216 | 68.578 | — | add_f32; converted through the host | 0e+0 | 0.140 | 489.49 | add_u16 | 0e+0 | 0.125 | 547.75 | u16 | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 19.189 | — | gemm64_f32; converted through the host | 0e+0 | 1.329 | 14.44 | gemm128_u16 | 0e+0 | 0.169 | 113.61 | u8mma | 0e+0 |
| cuda | add | uint8 | 16777216 | 64.665 | — | add_f32; converted through the host | 0e+0 | 0.088 | 734.00 | add_u8 | 0e+0 | 0.027 | 2431.03 | u8 | 0e+0 |
| cuda | matmul | uint8 | 2048x2048x2048 | 19.998 | — | gemm128_f32; converted through the host | 4.96e+3 | 1.283 | 15.59 | gemm128_u8 | 5.12e+3 | 0.093 | 214.57 | u8mma | 4.96e+3 |
| cuda | add | uint2 | 16777216 | 72.361 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 4256.54 | u8 | 0e+0 |
| cuda | matmul | uint2 | 2048x2048x2048 | 16.349 | — | gemm128_f32; converted through the host | 6.11e+2 | — | — | — | — | 0.107 | 153.51 | u8mma | 6.11e+2 |
| cuda | add | q8_0 | 16777216 | — | — | — | — | — | — | — | — | 1.411 | — | f32 | 7.87e-3 |
| cuda | matmul | q8_0 | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.860 | — | f32 | 3.09e-1 |
| cuda | softmax | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.779 | — | f32 (blocks through lane storage) | 8.37e-6 |
| cuda | rms_norm | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.776 | — | f32 (blocks through lane storage) | 4.85e-4 |
| cuda | add | q4_0 | 16777216 | — | — | — | — | — | — | — | — | 0.658 | — | f32 | 2.34e-1 |
| cuda | matmul | q4_0 | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.782 | — | f32 | 6.51e+0 |
| cuda | softmax | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 1.11e-4 |
| cuda | rms_norm | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.456 | — | f32 (blocks through lane storage) | 4.88e-4 |
| cuda | add | q4_k | 16777216 | — | — | — | — | — | — | — | — | 1.192 | — | f32 | 1.33e-1 |
| cuda | matmul | q4_k | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.868 | — | f32 | 3.83e+0 |
| cuda | softmax | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 0.983 | — | f32 (blocks through lane storage) | 7.31e-5 |
| cuda | rms_norm | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 0.983 | — | f32 (blocks through lane storage) | 1.37e-3 |
| cuda | add | q6_k | 16777216 | — | — | — | — | — | — | — | — | 1.401 | — | f32 | 3.22e-2 |
| cuda | matmul | q6_k | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.954 | — | f32 | 1.25e+0 |
| cuda | softmax | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 1.195 | — | f32 (blocks through lane storage) | 1.17e-4 |
| cuda | rms_norm | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 1.194 | — | f32 (blocks through lane storage) | 4.69e-4 |
| cuda | add | bool | 16777216 | — | — | — | — | — | — | — | — | 0.017 | — | u8 | 1e+0 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cpu | add | float64 | 16777216 | 6.698 | — | typed tile | 0e+0 | 7.639 | 0.88 | Vector<Double> | 0e+0 | 7.321 | 0.91 | f64 | 0e+0 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.339 | — | to f32 and back | 7.96e-5 | 17.473 | 0.36 | Vector<Double> | 1.14e-13 | 3.235 | 1.96 | f64 | 1.14e-13 |
| cpu | softmax | float64 | 4096x4096 | 30.737 | — | to f32 and back | 6.21e-10 | 5.907 | 5.20 | Vector<Double> | 4.34e-19 | 5.238 | 5.87 | f64 | 1.08e-17 |
| cpu | rms_norm | float64 | 4096x4096 | 30.614 | — | to f32 and back | 3.73e-6 | 4.204 | 7.28 | Vector<Double> | 1.33e-14 | 5.009 | 6.11 | f64 | 1.27e-14 |
| cpu | add | float32 | 16777216 | 3.138 | — | f32 kernel | 5.96e-8 | 3.200 | 0.98 | Vector<Single> | 5.96e-8 | 3.043 | 1.03 | f32 | 5.96e-8 |
| cpu | matmul | float32 | 1024x1024x1024 | 1.937 | — | f32 kernel | 7.96e-5 | 3.620 | 0.54 | Vector<Single> | 8.34e-5 | 1.807 | 1.07 | f32 | 7.96e-5 |
| cpu | softmax | float32 | 4096x4096 | 3.231 | — | f32 kernel | 6.21e-10 | 4.841 | 0.67 | Vector<Single> | 5.91e-9 | 2.159 | 1.50 | f32 | 6.21e-10 |
| cpu | rms_norm | float32 | 4096x4096 | 2.531 | — | f32 kernel | 3.73e-6 | 2.275 | 1.11 | Vector<Single> | 5.16e-7 | 2.027 | 1.25 | f32 | 3.68e-7 |
| cpu | add | float16 | 16777216 | 3.051 | — | typed tile | 4.88e-4 | 7.423 | 0.41 | scalar | 4.88e-4 | 1.167 | 2.62 | f32 | 4.88e-4 |
| cpu | matmul | float16 | 1024x1024x1024 | 6.664 | — | to f32 and back | 1.56e-2 | 907.849 | 0.01 | scalar | 6.08e-1 | 1.773 | 3.76 | f32 | 1.56e-2 |
| cpu | softmax | float16 | 4096x4096 | 45.990 | — | to f32 and back | 9.54e-7 | 47.347 | 0.97 | scalar | 1.57e-4 | 2.220 | 20.72 | f32 | 9.54e-7 |
| cpu | rms_norm | float16 | 4096x4096 | 45.045 | — | to f32 and back | 4.93e-4 | 20.984 | 2.15 | scalar | 7.06e-2 | 0.825 | 54.59 | f32 | 4.89e-4 |
| cpu | add | bfloat16 | 16777216 | 1.172 | — | typed tile | 3.91e-3 | — | — | — | — | 0.861 | 1.36 | f32 | 3.91e-3 |
| cpu | matmul | bfloat16 | 1024x1024x1024 | 4.593 | — | to f32 and back | 1.25e-1 | — | — | — | — | 1.630 | 2.82 | f32 | 1.25e-1 |
| cpu | softmax | bfloat16 | 4096x4096 | 24.028 | — | to f32 and back | 7.63e-6 | — | — | — | — | 2.639 | 9.10 | f32 | 7.63e-6 |
| cpu | rms_norm | bfloat16 | 4096x4096 | 23.230 | — | to f32 and back | 3.91e-3 | — | — | — | — | 0.740 | 31.40 | f32 | 3.91e-3 |
| cpu | add | float8_e4m3 | 16777216 | 22.887 | — | typed tile | 6.25e-2 | — | — | — | — | 1.827 | 12.53 | f32 | 6.25e-2 |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | 36.541 | — | to f32 and back | 2e+0 | — | — | — | — | 1.305 | 28.00 | f32 | 2e+0 |
| cpu | softmax | float8_e4m3 | 4096x4096 | 89.675 | — | to f32 and back | 9.76e-4 | — | — | — | — | 3.089 | 29.03 | f32 | 9.76e-4 |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | 434.466 | — | to f32 and back | 6.25e-2 | — | — | — | — | 1.649 | 263.55 | f32 | 6.25e-2 |
| cpu | add | float8_e5m2 | 16777216 | 17.680 | — | typed tile | 1.25e-1 | — | — | — | — | 1.988 | 8.89 | f32 | 1.25e-1 |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | 29.510 | — | to f32 and back | 4e+0 | — | — | — | — | 1.288 | 22.91 | f32 | 4e+0 |
| cpu | softmax | float8_e5m2 | 4096x4096 | 278.741 | — | to f32 and back | 1.62e-4 | — | — | — | — | 3.885 | 71.75 | f32 | 1.62e-4 |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | 314.923 | — | to f32 and back | 1.08e-1 | — | — | — | — | 1.786 | 176.32 | f32 | 1.08e-1 |
| cpu | add | int64 | 16777216 | 24.938 | — | typed tile | 0e+0 | 7.250 | 3.44 | Vector<Int64> | 0e+0 | 7.171 | 3.48 | s64 | 0e+0 |
| cpu | matmul | int64 | 1024x1024x1024 | 6.059 | — | to f32 and back | 0e+0 | 31.555 | 0.19 | Vector<Int64> | 0e+0 | 1.133 | 5.35 | s8dp | 0e+0 |
| cpu | add | int32 | 16777216 | 27.190 | — | typed tile | 0e+0 | 3.245 | 8.38 | Vector<Int32> | 0e+0 | 3.043 | 8.94 | s32 | 0e+0 |
| cpu | matmul | int32 | 1024x1024x1024 | 5.991 | — | to f32 and back | 0e+0 | 3.753 | 1.60 | Vector<Int32> | 0e+0 | 0.561 | 10.68 | s8dp | 0e+0 |
| cpu | add | int16 | 16777216 | 1.684 | — | typed tile | 0e+0 | 1.226 | 1.37 | Vector<Int16> | 0e+0 | 0.907 | 1.86 | s16 | 0e+0 |
| cpu | matmul | int16 | 1024x1024x1024 | 5.422 | — | to f32 and back | 0e+0 | 1.880 | 2.88 | Vector<Int16> | 0e+0 | 0.394 | 13.77 | s8dp | 0e+0 |
| cpu | add | int8 | 16777216 | 1.906 | — | typed tile | 0e+0 | 0.437 | 4.36 | Vector<SByte> | 0e+0 | 0.136 | 14.05 | s8 | 0e+0 |
| cpu | matmul | int8 | 1024x1024x1024 | 6.391 | — | to f32 and back | 1.76e+2 | 2.710 | 2.36 | Vector<SByte> | 2.56e+2 | 0.414 | 15.44 | s8dp | 1.76e+2 |
| cpu | add | int4 | 16777216 | 3.221 | — | typed tile | 0e+0 | — | — | — | — | 0.144 | 22.38 | s8 | 0e+0 |
| cpu | matmul | int4 | 1024x1024x1024 | 12.242 | — | to f32 and back | 2.96e+2 | — | — | — | — | 0.526 | 23.26 | s8dp | 2.96e+2 |
| cpu | add | int2 | 16777216 | 2.979 | — | typed tile | 0e+0 | — | — | — | — | 0.285 | 10.45 | s8 | 0e+0 |
| cpu | matmul | int2 | 1024x1024x1024 | 6.078 | — | to f32 and back | 3.26e+2 | — | — | — | — | 0.349 | 17.43 | s8dp | 3.26e+2 |
| cpu | add | uint64 | 16777216 | 13.476 | — | typed tile | 0e+0 | 6.973 | 1.93 | Vector<UInt64> | 0e+0 | 7.093 | 1.90 | u64 | 0e+0 |
| cpu | matmul | uint64 | 1024x1024x1024 | 7.039 | — | to f32 and back | 0e+0 | 36.458 | 0.19 | Vector<UInt64> | 0e+0 | 1.173 | 6.00 | u8dp | 0e+0 |
| cpu | add | uint32 | 16777216 | 14.069 | — | typed tile | 0e+0 | 3.134 | 4.49 | Vector<UInt32> | 0e+0 | 3.102 | 4.54 | u32 | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.746 | — | to f32 and back | 0e+0 | 4.420 | 1.30 | Vector<UInt32> | 0e+0 | 0.553 | 10.39 | u8dp | 0e+0 |
| cpu | add | uint16 | 16777216 | 1.748 | — | typed tile | 0e+0 | 1.152 | 1.52 | Vector<UInt16> | 0e+0 | 1.010 | 1.73 | u16 | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.965 | — | to f32 and back | 0e+0 | 2.296 | 2.16 | Vector<UInt16> | 0e+0 | 0.533 | 9.32 | u8dp | 0e+0 |
| cpu | add | uint8 | 16777216 | 1.654 | — | typed tile | 0e+0 | 0.374 | 4.42 | Vector<Byte> | 0e+0 | 0.134 | 12.33 | u8 | 0e+0 |
| cpu | matmul | uint8 | 1024x1024x1024 | 5.219 | — | to f32 and back | 2.48e+3 | 2.166 | 2.41 | Vector<Byte> | 2.56e+3 | 0.415 | 12.57 | u8dp | 2.48e+3 |
| cpu | add | uint2 | 16777216 | 2.183 | — | typed tile | 0e+0 | — | — | — | — | 0.136 | 16.06 | u8 | 0e+0 |
| cpu | matmul | uint2 | 1024x1024x1024 | 5.886 | — | to f32 and back | 3.18e+2 | — | — | — | — | 0.362 | 16.26 | u8dp | 3.18e+2 |
| cpu | add | q8_0 | 16777216 | — | — | — | — | — | — | — | — | 5.240 | — | f32 | 7.87e-3 |
| cpu | matmul | q8_0 | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.460 | — | f32 | 2.03e-1 |
| cpu | softmax | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 6.227 | — | f32 | 8.37e-6 |
| cpu | rms_norm | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 4.558 | — | f32 | 4.85e-4 |
| cpu | add | q4_0 | 16777216 | — | — | — | — | — | — | — | — | 3.584 | — | f32 | 2.34e-1 |
| cpu | matmul | q4_0 | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.230 | — | f32 | 4.06e+0 |
| cpu | softmax | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 4.090 | — | f32 | 1.11e-4 |
| cpu | rms_norm | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 2.728 | — | f32 | 4.89e-4 |
| cpu | add | q4_k | 16777216 | — | — | — | — | — | — | — | — | 5.026 | — | f32 | 1.33e-1 |
| cpu | matmul | q4_k | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.940 | — | f32 | 2.71e+0 |
| cpu | softmax | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 6.220 | — | f32 | 7.31e-5 |
| cpu | rms_norm | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 6.236 | — | f32 | 1.37e-3 |
| cpu | add | q6_k | 16777216 | — | — | — | — | — | — | — | — | 10.262 | — | f32 | 3.22e-2 |
| cpu | matmul | q6_k | 1024x1024x1024 | — | — | — | — | — | — | — | — | 9.924 | — | f32 | 8.19e-1 |
| cpu | softmax | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 11.843 | — | f32 | 1.17e-4 |
| cpu | rms_norm | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 7.628 | — | f32 | 4.69e-4 |
| cpu | add | bool | 16777216 | — | — | — | — | — | — | — | — | 5.694 | — | u8 | 1e+0 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.236 | — | f32 | 6.46e+2 |
