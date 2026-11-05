# Qiyas bench

2026-11-06 00:51. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | add | float64 | 16777216 | 125.849 | — | add_f32; converted through the host | 5.96e-8 | 0.520 | 241.83 | add_f64 | 0e+0 | 0.521 | 241.60 | f64 | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.943 | — | gemm64_f32; converted through the host | 1.62e-4 | 22.802 | 1.36 | gemm64_f64 | 0e+0 | 22.710 | 1.36 | f64 | 0e+0 |
| cuda | softmax | float64 | 4096x4096 | 78.319 | — | softmax_f32; converted through the host | 5.29e-10 | 0.393 | 199.18 | softmax_f64 | 1.88e-10 | 1.738 | 45.07 | f64 | 1.13e-17 |
| cuda | rms_norm | float64 | 4096x4096 | 77.926 | — | rms_norm_f32; converted through the host | 3.1e-7 | 0.373 | 209.03 | rms_norm_f64 | 1.24e-14 | 0.549 | 141.84 | f64 | 1.07e-14 |
| cuda | add | float32 | 16777216 | 0.264 | — | add_f32 | 5.96e-8 | 0.262 | 1.01 | add_f32 | 5.96e-8 | 0.263 | 1.00 | f32 | 5.96e-8 |
| cuda | matmul | float32 | 2048x2048x2048 | 0.951 | — | gemm128_f32 | 1.62e-4 | 0.834 | 1.14 | gemm128_f32 | 1.62e-4 | 0.683 | 1.39 | f32 | 1.62e-4 |
| cuda | softmax | float32 | 4096x4096 | 0.183 | — | softmax_f32 | 5.29e-10 | 0.181 | 1.01 | softmax_f32 | 5.11e-10 | 0.181 | 1.01 | f32 | 4.52e-10 |
| cuda | rms_norm | float32 | 4096x4096 | 0.183 | — | rms_norm_f32 | 3.1e-7 | 0.181 | 1.01 | rms_norm_f32 | 2.93e-7 | 0.179 | 1.02 | f32 | 2.97e-7 |
| cuda | add | float16 | 16777216 | 0.662 | — | add_f32; converted on the device | 4.88e-4 | 0.140 | 4.74 | add_f16 | 4.88e-4 | 0.125 | 5.30 | f32 | 4.88e-4 |
| cuda | matmul | float16 | 2048x2048x2048 | 1.014 | — | gemm128_f32; converted on the device | 3.09e-2 | 0.657 | 1.54 | gemm128_f16 | 1.51e+0 | 0.287 | 3.53 | f16mma | 3.09e-2 |
| cuda | softmax | float16 | 4096x4096 | 0.442 | — | softmax_f32; converted on the device | 9.53e-7 | 0.090 | 4.89 | softmax_f16 | 3.3e-6 | 0.078 | 5.68 | f32 | 9.54e-7 |
| cuda | rms_norm | float16 | 4096x4096 | 0.441 | — | rms_norm_f32; converted on the device | 4.88e-4 | 0.088 | 5.00 | rms_norm_f16 | 1.64e-3 | 0.076 | 5.79 | f32 | 4.88e-4 |
| cuda | add | bfloat16 | 16777216 | 0.662 | — | add_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.124 | 5.34 | f32 | 3.91e-3 |
| cuda | matmul | bfloat16 | 2048x2048x2048 | 1.003 | — | gemm128_f32; converted on the device | 2.5e-1 | — | — | — | — | 0.284 | 3.53 | bf16mma | 2.5e-1 |
| cuda | softmax | bfloat16 | 4096x4096 | 0.441 | — | softmax_f32; converted on the device | 7.63e-6 | — | — | — | — | 0.077 | 5.75 | f32 | 7.63e-6 |
| cuda | rms_norm | bfloat16 | 4096x4096 | 0.442 | — | rms_norm_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.077 | 5.76 | f32 | 3.91e-3 |
| cuda | add | float8_e4m3 | 16777216 | 489.831 | — | add_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.050 | 9835.96 | f32 | 6.25e-2 |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | 130.881 | — | gemm64_f32; converted through the host | 3.99e+0 | — | — | — | — | 0.301 | 434.68 | f16mma | 3.99e+0 |
| cuda | softmax | float8_e4m3 | 4096x4096 | 109.814 | — | softmax_f32; converted through the host | 9.76e-4 | — | — | — | — | 0.068 | 1624.47 | f32 | 9.76e-4 |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | 462.157 | — | rms_norm_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.046 | 10046.90 | f32 | 6.25e-2 |
| cuda | add | float8_e5m2 | 16777216 | 481.180 | — | add_f32; converted through the host | 1.25e-1 | — | — | — | — | 0.847 | 567.90 | f32 | 1.25e-1 |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | 117.019 | — | gemm128_f32; converted through the host | 7.71e+0 | — | — | — | — | 0.301 | 389.41 | f16mma | 7.71e+0 |
| cuda | softmax | float8_e5m2 | 4096x4096 | 299.496 | — | softmax_f32; converted through the host | 1.62e-4 | — | — | — | — | 0.069 | 4372.20 | f32 | 1.62e-4 |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | 350.423 | — | rms_norm_f32; converted through the host | 1.08e-1 | — | — | — | — | 0.050 | 7022.50 | f32 | 1.08e-1 |
| cuda | add | int64 | 16777216 | 120.177 | — | add_f32; converted through the host | 0e+0 | 0.520 | 231.24 | add_s64 | 0e+0 | 0.520 | 231.11 | s64 | 0e+0 |
| cuda | matmul | int64 | 2048x2048x2048 | 30.041 | — | gemm128_f32; converted through the host | 0e+0 | 4.418 | 6.80 | gemm64_s64 | 0e+0 | 0.421 | 71.41 | s8mma | 0e+0 |
| cuda | add | int32 | 16777216 | 0.787 | — | add_f32; converted on the device | 0e+0 | 0.263 | 2.99 | add_s32 | 0e+0 | 0.263 | 2.99 | s32 | 0e+0 |
| cuda | matmul | int32 | 2048x2048x2048 | 1.144 | — | gemm128_f32; converted on the device | 0e+0 | 1.106 | 1.03 | gemm128_s32 | 0e+0 | 0.279 | 4.11 | s8mma | 0e+0 |
| cuda | add | int16 | 16777216 | 67.399 | — | add_f32; converted through the host | 0e+0 | 0.139 | 483.84 | add_s16 | 0e+0 | 0.124 | 543.98 | s16 | 0e+0 |
| cuda | matmul | int16 | 2048x2048x2048 | 17.225 | — | gemm128_f32; converted through the host | 0e+0 | 1.585 | 10.86 | gemm128_s16 | 0e+0 | 0.231 | 74.44 | s8mma | 0e+0 |
| cuda | add | int8 | 16777216 | 65.073 | — | add_f32; converted through the host | 0e+0 | 0.082 | 789.72 | add_s8 | 0e+0 | 0.023 | 2780.90 | s8 | 0e+0 |
| cuda | matmul | int8 | 2048x2048x2048 | 21.861 | — | gemm128_f32; converted through the host | 3.23e+2 | 1.548 | 14.12 | gemm128_s8 | 5.12e+2 | 0.214 | 102.20 | s8mma | 3.23e+2 |
| cuda | add | int4 | 16777216 | 81.782 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4444.68 | s8 | 0e+0 |
| cuda | matmul | int4 | 2048x2048x2048 | 38.206 | — | gemm128_f32; converted through the host | 4.43e+2 | — | — | — | — | 0.221 | 173.03 | s8mma | 4.43e+2 |
| cuda | add | int2 | 16777216 | 76.942 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4181.61 | s8 | 0e+0 |
| cuda | matmul | int2 | 2048x2048x2048 | 18.738 | — | gemm64_f32; converted through the host | 6.16e+2 | — | — | — | — | 0.232 | 80.80 | s8mma | 6.16e+2 |
| cuda | add | uint64 | 16777216 | 143.842 | — | add_f32; converted through the host | 0e+0 | 1.000 | 143.83 | add_u64 | 0e+0 | 0.520 | 276.57 | u64 | 0e+0 |
| cuda | matmul | uint64 | 2048x2048x2048 | 32.614 | — | gemm128_f32; converted through the host | 0e+0 | 4.359 | 7.48 | gemm64_u64 | 0e+0 | 0.416 | 78.46 | u8mma | 0e+0 |
| cuda | add | uint32 | 16777216 | 103.208 | — | add_f32; converted through the host | 0e+0 | 0.263 | 392.73 | add_u32 | 0e+0 | 0.263 | 392.58 | u32 | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.980 | — | gemm128_f32; converted through the host | 0e+0 | 1.076 | 21.35 | gemm128_u32 | 0e+0 | 0.271 | 84.67 | u8mma | 0e+0 |
| cuda | add | uint16 | 16777216 | 68.127 | — | add_f32; converted through the host | 0e+0 | 0.139 | 489.07 | add_u16 | 0e+0 | 0.124 | 547.64 | u16 | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.984 | — | gemm128_f32; converted through the host | 0e+0 | 1.325 | 13.57 | gemm128_u16 | 0e+0 | 0.225 | 79.82 | u8mma | 0e+0 |
| cuda | add | uint8 | 16777216 | 65.224 | — | add_f32; converted through the host | 0e+0 | 0.074 | 884.99 | add_u8 | 0e+0 | 0.022 | 3033.67 | u8 | 0e+0 |
| cuda | matmul | uint8 | 2048x2048x2048 | 18.122 | — | gemm128_f32; converted through the host | 4.96e+3 | 1.280 | 14.16 | gemm128_u8 | 5.12e+3 | 0.212 | 85.44 | u8mma | 4.96e+3 |
| cuda | add | uint2 | 16777216 | 68.037 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 3955.65 | u8 | 0e+0 |
| cuda | matmul | uint2 | 2048x2048x2048 | 17.815 | — | gemm128_f32; converted through the host | 6.11e+2 | — | — | — | — | 0.226 | 78.72 | u8mma | 6.11e+2 |
| cpu | add | float64 | 16777216 | 7.344 | — | typed tile | 0e+0 | 6.823 | 1.08 | Vector<Double> | 0e+0 | 6.714 | 1.09 | f64 | 0e+0 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.378 | — | to f32 and back | 7.96e-5 | 15.905 | 0.40 | Vector<Double> | 0e+0 | 3.286 | 1.94 | f64 | 6.39e-14 |
| cpu | softmax | float64 | 4096x4096 | 31.993 | — | to f32 and back | 6.21e-10 | 6.411 | 4.99 | Vector<Double> | 4.34e-19 | 5.565 | 5.75 | f64 | 1.08e-17 |
| cpu | rms_norm | float64 | 4096x4096 | 30.889 | — | to f32 and back | 3.73e-6 | 4.377 | 7.06 | Vector<Double> | 1.33e-14 | 4.605 | 6.71 | f64 | 1.27e-14 |
| cpu | add | float32 | 16777216 | 3.048 | — | f32 kernel | 5.96e-8 | 3.005 | 1.01 | Vector<Single> | 5.96e-8 | 2.922 | 1.04 | f32 | 5.96e-8 |
| cpu | matmul | float32 | 1024x1024x1024 | 2.713 | — | f32 kernel | 7.96e-5 | 5.988 | 0.45 | Vector<Single> | 8.34e-5 | 1.625 | 1.67 | f32 | 7.96e-5 |
| cpu | softmax | float32 | 4096x4096 | 3.531 | — | f32 kernel | 6.21e-10 | 4.294 | 0.82 | Vector<Single> | 5.91e-9 | 2.922 | 1.21 | f32 | 6.21e-10 |
| cpu | rms_norm | float32 | 4096x4096 | 2.394 | — | f32 kernel | 3.73e-6 | 1.894 | 1.26 | Vector<Single> | 5.16e-7 | 1.596 | 1.50 | f32 | 3.68e-7 |
| cpu | add | float16 | 16777216 | 2.808 | — | typed tile | 4.88e-4 | 7.501 | 0.37 | scalar | 4.88e-4 | 1.742 | 1.61 | f32 | 4.88e-4 |
| cpu | matmul | float16 | 1024x1024x1024 | 6.838 | — | to f32 and back | 1.56e-2 | 765.105 | 0.01 | scalar | 6.08e-1 | 1.454 | 4.70 | f32 | 1.56e-2 |
| cpu | softmax | float16 | 4096x4096 | 45.965 | — | to f32 and back | 9.54e-7 | 55.323 | 0.83 | scalar | 1.57e-4 | 2.241 | 20.52 | f32 | 9.54e-7 |
| cpu | rms_norm | float16 | 4096x4096 | 45.370 | — | to f32 and back | 4.93e-4 | 23.947 | 1.89 | scalar | 7.06e-2 | 0.827 | 54.87 | f32 | 4.89e-4 |
| cpu | add | bfloat16 | 16777216 | 1.261 | — | typed tile | 3.91e-3 | — | — | — | — | 0.915 | 1.38 | f32 | 3.91e-3 |
| cpu | matmul | bfloat16 | 1024x1024x1024 | 4.736 | — | to f32 and back | 1.25e-1 | — | — | — | — | 1.386 | 3.42 | f32 | 1.25e-1 |
| cpu | softmax | bfloat16 | 4096x4096 | 23.833 | — | to f32 and back | 7.63e-6 | — | — | — | — | 1.786 | 13.35 | f32 | 7.63e-6 |
| cpu | rms_norm | bfloat16 | 4096x4096 | 23.501 | — | to f32 and back | 3.91e-3 | — | — | — | — | 0.474 | 49.55 | f32 | 3.91e-3 |
| cpu | add | float8_e4m3 | 16777216 | 26.282 | — | typed tile | 6.25e-2 | — | — | — | — | 2.222 | 11.83 | f32 | 6.25e-2 |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | 37.162 | — | to f32 and back | 2e+0 | — | — | — | — | 1.477 | 25.17 | f32 | 2e+0 |
| cpu | softmax | float8_e4m3 | 4096x4096 | 89.788 | — | to f32 and back | 9.76e-4 | — | — | — | — | 3.686 | 24.36 | f32 | 9.76e-4 |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | 429.956 | — | to f32 and back | 6.25e-2 | — | — | — | — | 1.914 | 224.65 | f32 | 6.25e-2 |
| cpu | add | float8_e5m2 | 16777216 | 21.284 | — | typed tile | 1.25e-1 | — | — | — | — | 2.422 | 8.79 | f32 | 1.25e-1 |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | 29.022 | — | to f32 and back | 4e+0 | — | — | — | — | 1.491 | 19.47 | f32 | 4e+0 |
| cpu | softmax | float8_e5m2 | 4096x4096 | 283.307 | — | to f32 and back | 1.62e-4 | — | — | — | — | 3.878 | 73.05 | f32 | 1.62e-4 |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | 313.137 | — | to f32 and back | 1.08e-1 | — | — | — | — | 2.083 | 150.31 | f32 | 1.08e-1 |
| cpu | add | int64 | 16777216 | 19.776 | — | typed tile | 0e+0 | 6.763 | 2.92 | Vector<Int64> | 0e+0 | 6.615 | 2.99 | s64 | 0e+0 |
| cpu | matmul | int64 | 1024x1024x1024 | 6.126 | — | to f32 and back | 0e+0 | 24.649 | 0.25 | Vector<Int64> | 0e+0 | 0.876 | 7.00 | s8dp | 0e+0 |
| cpu | add | int32 | 16777216 | 20.288 | — | typed tile | 0e+0 | 2.959 | 6.86 | Vector<Int32> | 0e+0 | 2.948 | 6.88 | s32 | 0e+0 |
| cpu | matmul | int32 | 1024x1024x1024 | 5.923 | — | to f32 and back | 0e+0 | 6.204 | 0.95 | Vector<Int32> | 0e+0 | 0.607 | 9.76 | s8dp | 0e+0 |
| cpu | add | int16 | 16777216 | 1.639 | — | typed tile | 0e+0 | 1.425 | 1.15 | Vector<Int16> | 0e+0 | 0.840 | 1.95 | s16 | 0e+0 |
| cpu | matmul | int16 | 1024x1024x1024 | 5.658 | — | to f32 and back | 0e+0 | 2.371 | 2.39 | Vector<Int16> | 0e+0 | 0.992 | 5.71 | s8dp | 0e+0 |
| cpu | add | int8 | 16777216 | 1.564 | — | typed tile | 0e+0 | 0.432 | 3.62 | Vector<SByte> | 0e+0 | 0.154 | 10.14 | s8 | 0e+0 |
| cpu | matmul | int8 | 1024x1024x1024 | 6.123 | — | to f32 and back | 1.76e+2 | 2.507 | 2.44 | Vector<SByte> | 2.56e+2 | 0.597 | 10.26 | s8dp | 1.76e+2 |
| cpu | add | int4 | 16777216 | 2.437 | — | typed tile | 0e+0 | — | — | — | — | 0.101 | 24.11 | s8 | 0e+0 |
| cpu | matmul | int4 | 1024x1024x1024 | 11.926 | — | to f32 and back | 2.96e+2 | — | — | — | — | 0.532 | 22.43 | s8dp | 2.96e+2 |
| cpu | add | int2 | 16777216 | 2.868 | — | typed tile | 0e+0 | — | — | — | — | 0.267 | 10.74 | s8 | 0e+0 |
| cpu | matmul | int2 | 1024x1024x1024 | 6.899 | — | to f32 and back | 3.26e+2 | — | — | — | — | 0.514 | 13.42 | s8dp | 3.26e+2 |
| cpu | add | uint64 | 16777216 | 6.599 | — | typed tile | 0e+0 | 6.738 | 0.98 | Vector<UInt64> | 0e+0 | 6.705 | 0.98 | u64 | 0e+0 |
| cpu | matmul | uint64 | 1024x1024x1024 | 6.337 | — | to f32 and back | 0e+0 | 22.120 | 0.29 | Vector<UInt64> | 0e+0 | 0.810 | 7.82 | u8dp | 0e+0 |
| cpu | add | uint32 | 16777216 | 5.172 | — | typed tile | 0e+0 | 2.943 | 1.76 | Vector<UInt32> | 0e+0 | 2.933 | 1.76 | u32 | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.090 | — | to f32 and back | 0e+0 | 3.922 | 1.55 | Vector<UInt32> | 0e+0 | 0.429 | 14.18 | u8dp | 0e+0 |
| cpu | add | uint16 | 16777216 | 1.375 | — | typed tile | 0e+0 | 1.283 | 1.07 | Vector<UInt16> | 0e+0 | 0.796 | 1.73 | u16 | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.178 | — | to f32 and back | 0e+0 | 1.925 | 2.69 | Vector<UInt16> | 0e+0 | 0.698 | 7.42 | u8dp | 0e+0 |
| cpu | add | uint8 | 16777216 | 1.630 | — | typed tile | 0e+0 | 0.423 | 3.85 | Vector<Byte> | 0e+0 | 0.130 | 12.59 | u8 | 0e+0 |
| cpu | matmul | uint8 | 1024x1024x1024 | 6.154 | — | to f32 and back | 2.48e+3 | 2.198 | 2.80 | Vector<Byte> | 2.56e+3 | 0.569 | 10.81 | u8dp | 2.48e+3 |
| cpu | add | uint2 | 16777216 | 2.761 | — | typed tile | 0e+0 | — | — | — | — | 0.232 | 11.91 | u8 | 0e+0 |
| cpu | matmul | uint2 | 1024x1024x1024 | 6.861 | — | to f32 and back | 3.18e+2 | — | — | — | — | 0.402 | 17.08 | u8dp | 3.18e+2 |
