# Qiyas bench

2026-11-06 07:20. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | add | float64 | 16777216 | 124.132 | — | add_f32; converted through the host | 5.96e-8 | 0.520 | 238.67 | add_f64 | 0e+0 | 0.520 | 238.62 | f64 | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 31.367 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.644 | 1.39 | gemm64_f64 | 0e+0 | 4.167 | 7.53 | f64x | 2.59e-13 |
| cuda | softmax | float64 | 4096x4096 | 88.423 | — | softmax_f32; converted through the host | 5.29e-10 | 0.394 | 224.54 | softmax_f64 | 1.88e-10 | 1.740 | 50.82 | f64 | 1.13e-17 |
| cuda | rms_norm | float64 | 4096x4096 | 82.227 | — | rms_norm_f32; converted through the host | 3.1e-7 | 0.371 | 221.46 | rms_norm_f64 | 1.24e-14 | 0.550 | 149.56 | f64 | 1.07e-14 |
| cuda | add | float32 | 16777216 | 0.263 | — | add_f32 | 5.96e-8 | 0.262 | 1.01 | add_f32 | 5.96e-8 | 0.263 | 1.00 | f32 | 5.96e-8 |
| cuda | matmul | float32 | 2048x2048x2048 | 0.966 | — | gemm128_f32 | 1.62e-4 | 0.844 | 1.14 | gemm128_f32 | 1.62e-4 | 0.683 | 1.41 | f32 | 1.62e-4 |
| cuda | softmax | float32 | 4096x4096 | 0.182 | — | softmax_f32 | 5.29e-10 | 0.181 | 1.00 | softmax_f32 | 5.11e-10 | 0.179 | 1.01 | f32 | 4.52e-10 |
| cuda | rms_norm | float32 | 4096x4096 | 0.183 | — | rms_norm_f32 | 3.1e-7 | 0.182 | 1.00 | rms_norm_f32 | 2.93e-7 | 0.180 | 1.02 | f32 | 2.97e-7 |
| cuda | add | float16 | 16777216 | 0.661 | — | add_f32; converted on the device | 4.88e-4 | 0.139 | 4.75 | add_f16 | 4.88e-4 | 0.125 | 5.30 | f32 | 4.88e-4 |
| cuda | matmul | float16 | 2048x2048x2048 | 1.019 | — | gemm128_f32; converted on the device | 3.09e-2 | 0.658 | 1.55 | gemm128_f16 | 1.51e+0 | 0.245 | 4.16 | f16mma | 3.09e-2 |
| cuda | softmax | float16 | 4096x4096 | 0.443 | — | softmax_f32; converted on the device | 9.53e-7 | 0.089 | 4.95 | softmax_f16 | 3.3e-6 | 0.075 | 5.91 | f32 | 9.54e-7 |
| cuda | rms_norm | float16 | 4096x4096 | 0.442 | — | rms_norm_f32; converted on the device | 4.88e-4 | 0.087 | 5.08 | rms_norm_f16 | 1.64e-3 | 0.077 | 5.77 | f32 | 4.88e-4 |
| cuda | add | bfloat16 | 16777216 | 0.661 | — | add_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.125 | 5.29 | f32 | 3.91e-3 |
| cuda | matmul | bfloat16 | 2048x2048x2048 | 1.004 | — | gemm128_f32; converted on the device | 2.5e-1 | — | — | — | — | 0.243 | 4.14 | bf16mma | 2.5e-1 |
| cuda | softmax | bfloat16 | 4096x4096 | 0.443 | — | softmax_f32; converted on the device | 7.63e-6 | — | — | — | — | 0.077 | 5.77 | f32 | 7.63e-6 |
| cuda | rms_norm | bfloat16 | 4096x4096 | 0.440 | — | rms_norm_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.076 | 5.81 | f32 | 3.91e-3 |
| cuda | add | float8_e4m3 | 16777216 | 495.297 | — | add_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.053 | 9310.09 | f32 | 6.25e-2 |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | 137.725 | — | gemm64_f32; converted through the host | 3.99e+0 | — | — | — | — | 0.272 | 506.71 | f16mma | 3.99e+0 |
| cuda | softmax | float8_e4m3 | 4096x4096 | 108.007 | — | softmax_f32; converted through the host | 9.76e-4 | — | — | — | — | 0.067 | 1624.16 | f32 | 9.76e-4 |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | 472.585 | — | rms_norm_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.046 | 10273.58 | f32 | 6.25e-2 |
| cuda | add | float8_e5m2 | 16777216 | 471.164 | — | add_f32; converted through the host | 1.25e-1 | — | — | — | — | 0.666 | 707.56 | f32 | 1.25e-1 |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | 115.629 | — | gemm64_f32; converted through the host | 7.71e+0 | — | — | — | — | 0.265 | 436.99 | f16mma | 7.71e+0 |
| cuda | softmax | float8_e5m2 | 4096x4096 | 305.112 | — | softmax_f32; converted through the host | 1.62e-4 | — | — | — | — | 0.070 | 4383.80 | f32 | 1.62e-4 |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | 375.077 | — | rms_norm_f32; converted through the host | 1.08e-1 | — | — | — | — | 0.050 | 7486.57 | f32 | 1.08e-1 |
| cuda | add | int64 | 16777216 | 120.079 | — | add_f32; converted through the host | 0e+0 | 0.520 | 230.88 | add_s64 | 0e+0 | 0.520 | 230.83 | s64 | 0e+0 |
| cuda | matmul | int64 | 2048x2048x2048 | 29.443 | — | gemm128_f32; converted through the host | 0e+0 | 4.424 | 6.66 | gemm64_s64 | 0e+0 | 0.359 | 82.04 | s8mma | 0e+0 |
| cuda | add | int32 | 16777216 | 0.783 | — | add_f32; converted on the device | 0e+0 | 0.263 | 2.98 | add_s32 | 0e+0 | 0.263 | 2.98 | s32 | 0e+0 |
| cuda | matmul | int32 | 2048x2048x2048 | 1.116 | — | gemm64_f32; converted on the device | 0e+0 | 1.086 | 1.03 | gemm128_s32 | 0e+0 | 0.218 | 5.12 | s8mma | 0e+0 |
| cuda | add | int16 | 16777216 | 73.475 | — | add_f32; converted through the host | 0e+0 | 0.139 | 527.83 | add_s16 | 0e+0 | 0.124 | 593.02 | s16 | 0e+0 |
| cuda | matmul | int16 | 2048x2048x2048 | 17.580 | — | gemm128_f32; converted through the host | 0e+0 | 1.586 | 11.08 | gemm128_s16 | 0e+0 | 0.177 | 99.32 | s8mma | 0e+0 |
| cuda | add | int8 | 16777216 | 65.141 | — | add_f32; converted through the host | 0e+0 | 0.075 | 872.03 | add_s8 | 0e+0 | 0.023 | 2895.15 | s8 | 0e+0 |
| cuda | matmul | int8 | 2048x2048x2048 | 22.431 | — | gemm128_f32; converted through the host | 3.23e+2 | 1.550 | 14.48 | gemm128_s8 | 5.12e+2 | 0.093 | 240.67 | s8mma | 3.23e+2 |
| cuda | add | int4 | 16777216 | 80.279 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4362.98 | s8 | 0e+0 |
| cuda | matmul | int4 | 2048x2048x2048 | 39.062 | — | gemm128_f32; converted through the host | 4.43e+2 | — | — | — | — | 0.103 | 377.77 | s8mma | 4.43e+2 |
| cuda | add | int2 | 16777216 | 78.323 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4256.67 | s8 | 0e+0 |
| cuda | matmul | int2 | 2048x2048x2048 | 18.817 | — | gemm64_f32; converted through the host | 6.16e+2 | — | — | — | — | 0.112 | 168.16 | s8mma | 6.16e+2 |
| cuda | add | uint64 | 16777216 | 136.900 | — | add_f32; converted through the host | 0e+0 | 0.521 | 262.86 | add_u64 | 0e+0 | 0.520 | 263.47 | u64 | 0e+0 |
| cuda | matmul | uint64 | 2048x2048x2048 | 34.064 | — | gemm64_f32; converted through the host | 0e+0 | 4.359 | 7.82 | gemm64_u64 | 0e+0 | 0.357 | 95.31 | u8mma | 0e+0 |
| cuda | add | uint32 | 16777216 | 101.415 | — | add_f32; converted through the host | 0e+0 | 0.263 | 385.46 | add_u32 | 0e+0 | 0.263 | 385.90 | u32 | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 23.537 | — | gemm64_f32; converted through the host | 0e+0 | 1.099 | 21.41 | gemm128_u32 | 0e+0 | 0.215 | 109.42 | u8mma | 0e+0 |
| cuda | add | uint16 | 16777216 | 68.893 | — | add_f32; converted through the host | 0e+0 | 0.139 | 496.35 | add_u16 | 0e+0 | 0.124 | 554.69 | u16 | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.061 | — | gemm128_f32; converted through the host | 0e+0 | 1.327 | 12.86 | gemm128_u16 | 0e+0 | 0.173 | 98.62 | u8mma | 0e+0 |
| cuda | add | uint8 | 16777216 | 64.411 | — | add_f32; converted through the host | 0e+0 | 0.096 | 669.55 | add_u8 | 0e+0 | 0.032 | 2031.90 | u8 | 0e+0 |
| cuda | matmul | uint8 | 2048x2048x2048 | 17.035 | — | gemm128_f32; converted through the host | 4.96e+3 | 1.284 | 13.27 | gemm128_u8 | 5.12e+3 | 0.093 | 182.97 | u8mma | 4.96e+3 |
| cuda | add | uint2 | 16777216 | 71.768 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.016 | 4376.12 | u8 | 0e+0 |
| cuda | matmul | uint2 | 2048x2048x2048 | 17.723 | — | gemm128_f32; converted through the host | 6.11e+2 | — | — | — | — | 0.108 | 164.41 | u8mma | 6.11e+2 |
| cpu | add | float64 | 16777216 | 7.064 | — | typed tile | 0e+0 | 6.752 | 1.05 | Vector<Double> | 0e+0 | 6.646 | 1.06 | f64 | 0e+0 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.243 | — | to f32 and back | 7.96e-5 | 15.909 | 0.39 | Vector<Double> | 0e+0 | 3.019 | 2.07 | f64 | 6.39e-14 |
| cpu | softmax | float64 | 4096x4096 | 30.929 | — | to f32 and back | 6.21e-10 | 7.623 | 4.06 | Vector<Double> | 4.34e-19 | 5.093 | 6.07 | f64 | 1.08e-17 |
| cpu | rms_norm | float64 | 4096x4096 | 30.874 | — | to f32 and back | 3.73e-6 | 4.061 | 7.60 | Vector<Double> | 1.33e-14 | 4.583 | 6.74 | f64 | 1.27e-14 |
| cpu | add | float32 | 16777216 | 2.953 | — | f32 kernel | 5.96e-8 | 2.951 | 1.00 | Vector<Single> | 5.96e-8 | 2.909 | 1.02 | f32 | 5.96e-8 |
| cpu | matmul | float32 | 1024x1024x1024 | 2.160 | — | f32 kernel | 7.96e-5 | 5.761 | 0.37 | Vector<Single> | 8.34e-5 | 1.361 | 1.59 | f32 | 7.96e-5 |
| cpu | softmax | float32 | 4096x4096 | 2.432 | — | f32 kernel | 6.21e-10 | 4.202 | 0.58 | Vector<Single> | 5.91e-9 | 1.854 | 1.31 | f32 | 6.21e-10 |
| cpu | rms_norm | float32 | 4096x4096 | 3.701 | — | f32 kernel | 3.73e-6 | 1.735 | 2.13 | Vector<Single> | 5.16e-7 | 1.624 | 2.28 | f32 | 3.68e-7 |
| cpu | add | float16 | 16777216 | 2.383 | — | typed tile | 4.88e-4 | 6.715 | 0.35 | scalar | 4.88e-4 | 1.169 | 2.04 | f32 | 4.88e-4 |
| cpu | matmul | float16 | 1024x1024x1024 | 6.771 | — | to f32 and back | 1.56e-2 | 817.620 | 0.01 | scalar | 6.08e-1 | 1.481 | 4.57 | f32 | 1.56e-2 |
| cpu | softmax | float16 | 4096x4096 | 46.889 | — | to f32 and back | 9.54e-7 | 45.650 | 1.03 | scalar | 1.57e-4 | 2.222 | 21.10 | f32 | 9.54e-7 |
| cpu | rms_norm | float16 | 4096x4096 | 45.695 | — | to f32 and back | 4.93e-4 | 20.710 | 2.21 | scalar | 7.06e-2 | 0.825 | 55.39 | f32 | 4.89e-4 |
| cpu | add | bfloat16 | 16777216 | 1.085 | — | typed tile | 3.91e-3 | — | — | — | — | 0.908 | 1.20 | f32 | 3.91e-3 |
| cpu | matmul | bfloat16 | 1024x1024x1024 | 4.846 | — | to f32 and back | 1.25e-1 | — | — | — | — | 1.162 | 4.17 | f32 | 1.25e-1 |
| cpu | softmax | bfloat16 | 4096x4096 | 24.043 | — | to f32 and back | 7.63e-6 | — | — | — | — | 1.773 | 13.56 | f32 | 7.63e-6 |
| cpu | rms_norm | bfloat16 | 4096x4096 | 23.810 | — | to f32 and back | 3.91e-3 | — | — | — | — | 0.402 | 59.30 | f32 | 3.91e-3 |
| cpu | add | float8_e4m3 | 16777216 | 21.216 | — | typed tile | 6.25e-2 | — | — | — | — | 1.822 | 11.64 | f32 | 6.25e-2 |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | 35.324 | — | to f32 and back | 2e+0 | — | — | — | — | 1.242 | 28.44 | f32 | 2e+0 |
| cpu | softmax | float8_e4m3 | 4096x4096 | 91.311 | — | to f32 and back | 9.76e-4 | — | — | — | — | 3.076 | 29.68 | f32 | 9.76e-4 |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | 402.311 | — | to f32 and back | 6.25e-2 | — | — | — | — | 1.620 | 248.29 | f32 | 6.25e-2 |
| cpu | add | float8_e5m2 | 16777216 | 17.483 | — | typed tile | 1.25e-1 | — | — | — | — | 1.995 | 8.77 | f32 | 1.25e-1 |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | 30.426 | — | to f32 and back | 4e+0 | — | — | — | — | 1.780 | 17.10 | f32 | 4e+0 |
| cpu | softmax | float8_e5m2 | 4096x4096 | 292.458 | — | to f32 and back | 1.62e-4 | — | — | — | — | 3.231 | 90.53 | f32 | 1.62e-4 |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | 320.714 | — | to f32 and back | 1.08e-1 | — | — | — | — | 1.701 | 188.53 | f32 | 1.08e-1 |
| cpu | add | int64 | 16777216 | 21.918 | — | typed tile | 0e+0 | 6.747 | 3.25 | Vector<Int64> | 0e+0 | 6.708 | 3.27 | s64 | 0e+0 |
| cpu | matmul | int64 | 1024x1024x1024 | 6.345 | — | to f32 and back | 0e+0 | 20.748 | 0.31 | Vector<Int64> | 0e+0 | 0.685 | 9.26 | s8dp | 0e+0 |
| cpu | add | int32 | 16777216 | 26.485 | — | typed tile | 0e+0 | 2.955 | 8.96 | Vector<Int32> | 0e+0 | 2.940 | 9.01 | s32 | 0e+0 |
| cpu | matmul | int32 | 1024x1024x1024 | 6.059 | — | to f32 and back | 0e+0 | 5.129 | 1.18 | Vector<Int32> | 0e+0 | 0.568 | 10.66 | s8dp | 0e+0 |
| cpu | add | int16 | 16777216 | 1.341 | — | typed tile | 0e+0 | 1.253 | 1.07 | Vector<Int16> | 0e+0 | 0.775 | 1.73 | s16 | 0e+0 |
| cpu | matmul | int16 | 1024x1024x1024 | 5.899 | — | to f32 and back | 0e+0 | 1.507 | 3.92 | Vector<Int16> | 0e+0 | 0.523 | 11.28 | s8dp | 0e+0 |
| cpu | add | int8 | 16777216 | 1.865 | — | typed tile | 0e+0 | 0.514 | 3.63 | Vector<SByte> | 0e+0 | 0.129 | 14.51 | s8 | 0e+0 |
| cpu | matmul | int8 | 1024x1024x1024 | 6.415 | — | to f32 and back | 1.76e+2 | 2.762 | 2.32 | Vector<SByte> | 2.56e+2 | 0.581 | 11.05 | s8dp | 1.76e+2 |
| cpu | add | int4 | 16777216 | 2.933 | — | typed tile | 0e+0 | — | — | — | — | 0.143 | 20.48 | s8 | 0e+0 |
| cpu | matmul | int4 | 1024x1024x1024 | 12.125 | — | to f32 and back | 2.96e+2 | — | — | — | — | 0.481 | 25.22 | s8dp | 2.96e+2 |
| cpu | add | int2 | 16777216 | 2.868 | — | typed tile | 0e+0 | — | — | — | — | 0.275 | 10.44 | s8 | 0e+0 |
| cpu | matmul | int2 | 1024x1024x1024 | 6.946 | — | to f32 and back | 3.26e+2 | — | — | — | — | 0.522 | 13.31 | s8dp | 3.26e+2 |
| cpu | add | uint64 | 16777216 | 12.998 | — | typed tile | 0e+0 | 6.815 | 1.91 | Vector<UInt64> | 0e+0 | 6.719 | 1.93 | u64 | 0e+0 |
| cpu | matmul | uint64 | 1024x1024x1024 | 6.852 | — | to f32 and back | 0e+0 | 33.086 | 0.21 | Vector<UInt64> | 0e+0 | 0.792 | 8.65 | u8dp | 0e+0 |
| cpu | add | uint32 | 16777216 | 14.199 | — | typed tile | 0e+0 | 2.991 | 4.75 | Vector<UInt32> | 0e+0 | 3.013 | 4.71 | u32 | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.167 | — | to f32 and back | 0e+0 | 5.202 | 1.19 | Vector<UInt32> | 0e+0 | 0.543 | 11.36 | u8dp | 0e+0 |
| cpu | add | uint16 | 16777216 | 1.654 | — | typed tile | 0e+0 | 1.385 | 1.19 | Vector<UInt16> | 0e+0 | 0.913 | 1.81 | u16 | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.350 | — | to f32 and back | 0e+0 | 3.418 | 1.57 | Vector<UInt16> | 0e+0 | 0.544 | 9.84 | u8dp | 0e+0 |
| cpu | add | uint8 | 16777216 | 1.988 | — | typed tile | 0e+0 | 0.377 | 5.27 | Vector<Byte> | 0e+0 | 0.145 | 13.67 | u8 | 0e+0 |
| cpu | matmul | uint8 | 1024x1024x1024 | 6.040 | — | to f32 and back | 2.48e+3 | 2.420 | 2.50 | Vector<Byte> | 2.56e+3 | 0.420 | 14.38 | u8dp | 2.48e+3 |
| cpu | add | uint2 | 16777216 | 2.936 | — | typed tile | 0e+0 | — | — | — | — | 0.244 | 12.04 | u8 | 0e+0 |
| cpu | matmul | uint2 | 1024x1024x1024 | 6.643 | — | to f32 and back | 3.18e+2 | — | — | — | — | 0.422 | 15.76 | u8dp | 3.18e+2 |
