# Qiyas bench

2026-11-06 01:28. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | add | float64 | 16777216 | 125.971 | — | add_f32; converted through the host | 5.96e-8 | 0.521 | 241.83 | add_f64 | 0e+0 | 0.521 | 241.79 | f64 | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.939 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.976 | 1.35 | gemm64_f64 | 0e+0 | 4.163 | 7.43 | f64x | 2.59e-13 |
| cuda | softmax | float64 | 4096x4096 | 84.633 | — | softmax_f32; converted through the host | 5.29e-10 | 0.394 | 214.80 | softmax_f64 | 1.88e-10 | 1.740 | 48.65 | f64 | 1.13e-17 |
| cuda | rms_norm | float64 | 4096x4096 | 83.513 | — | rms_norm_f32; converted through the host | 3.1e-7 | 0.371 | 225.41 | rms_norm_f64 | 1.24e-14 | 0.549 | 152.17 | f64 | 1.07e-14 |
| cuda | add | float32 | 16777216 | 0.263 | — | add_f32 | 5.96e-8 | 0.263 | 1.00 | add_f32 | 5.96e-8 | 0.263 | 1.00 | f32 | 5.96e-8 |
| cuda | matmul | float32 | 2048x2048x2048 | 1.019 | — | gemm128_f32 | 1.62e-4 | 0.837 | 1.22 | gemm128_f32 | 1.62e-4 | 0.684 | 1.49 | f32 | 1.62e-4 |
| cuda | softmax | float32 | 4096x4096 | 0.183 | — | softmax_f32 | 5.29e-10 | 0.181 | 1.01 | softmax_f32 | 5.11e-10 | 0.179 | 1.02 | f32 | 4.52e-10 |
| cuda | rms_norm | float32 | 4096x4096 | 0.183 | — | rms_norm_f32 | 3.1e-7 | 0.183 | 1.00 | rms_norm_f32 | 2.93e-7 | 0.179 | 1.02 | f32 | 2.97e-7 |
| cuda | add | float16 | 16777216 | 0.662 | — | add_f32; converted on the device | 4.88e-4 | 0.140 | 4.74 | add_f16 | 4.88e-4 | 0.126 | 5.26 | f32 | 4.88e-4 |
| cuda | matmul | float16 | 2048x2048x2048 | 1.020 | — | gemm128_f32; converted on the device | 3.09e-2 | 0.659 | 1.55 | gemm128_f16 | 1.51e+0 | 0.243 | 4.20 | f16mma | 3.09e-2 |
| cuda | softmax | float16 | 4096x4096 | 0.444 | — | softmax_f32; converted on the device | 9.53e-7 | 0.091 | 4.86 | softmax_f16 | 3.3e-6 | 0.082 | 5.45 | f32 | 9.54e-7 |
| cuda | rms_norm | float16 | 4096x4096 | 0.443 | — | rms_norm_f32; converted on the device | 4.88e-4 | 0.089 | 4.98 | rms_norm_f16 | 1.64e-3 | 0.081 | 5.48 | f32 | 4.88e-4 |
| cuda | add | bfloat16 | 16777216 | 0.662 | — | add_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.126 | 5.25 | f32 | 3.91e-3 |
| cuda | matmul | bfloat16 | 2048x2048x2048 | 1.036 | — | gemm128_f32; converted on the device | 2.5e-1 | — | — | — | — | 0.243 | 4.27 | bf16mma | 2.5e-1 |
| cuda | softmax | bfloat16 | 4096x4096 | 0.443 | — | softmax_f32; converted on the device | 7.63e-6 | — | — | — | — | 0.084 | 5.28 | f32 | 7.63e-6 |
| cuda | rms_norm | bfloat16 | 4096x4096 | 0.442 | — | rms_norm_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.082 | 5.39 | f32 | 3.91e-3 |
| cuda | add | float8_e4m3 | 16777216 | 490.982 | — | add_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.051 | 9589.49 | f32 | 6.25e-2 |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | 130.118 | — | gemm64_f32; converted through the host | 3.99e+0 | — | — | — | — | 0.266 | 489.90 | f16mma | 3.99e+0 |
| cuda | softmax | float8_e4m3 | 4096x4096 | 117.973 | — | softmax_f32; converted through the host | 9.76e-4 | — | — | — | — | 0.069 | 1719.73 | f32 | 9.76e-4 |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | 447.492 | — | rms_norm_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.046 | 9728.08 | f32 | 6.25e-2 |
| cuda | add | float8_e5m2 | 16777216 | 470.551 | — | add_f32; converted through the host | 1.25e-1 | — | — | — | — | 1.229 | 382.97 | f32 | 1.25e-1 |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | 118.262 | — | gemm128_f32; converted through the host | 7.71e+0 | — | — | — | — | 0.260 | 454.50 | f16mma | 7.71e+0 |
| cuda | softmax | float8_e5m2 | 4096x4096 | 297.301 | — | softmax_f32; converted through the host | 1.62e-4 | — | — | — | — | 0.069 | 4333.83 | f32 | 1.62e-4 |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | 343.514 | — | rms_norm_f32; converted through the host | 1.08e-1 | — | — | — | — | 0.044 | 7771.81 | f32 | 1.08e-1 |
| cuda | add | int64 | 16777216 | 126.962 | — | add_f32; converted through the host | 0e+0 | 0.520 | 244.16 | add_s64 | 0e+0 | 0.520 | 244.11 | s64 | 0e+0 |
| cuda | matmul | int64 | 2048x2048x2048 | 30.269 | — | gemm128_f32; converted through the host | 0e+0 | 4.431 | 6.83 | gemm64_s64 | 0e+0 | 0.362 | 83.52 | s8mma | 0e+0 |
| cuda | add | int32 | 16777216 | 0.789 | — | add_f32; converted on the device | 0e+0 | 0.263 | 3.00 | add_s32 | 0e+0 | 0.263 | 3.00 | s32 | 0e+0 |
| cuda | matmul | int32 | 2048x2048x2048 | 1.137 | — | gemm128_f32; converted on the device | 0e+0 | 1.091 | 1.04 | gemm128_s32 | 0e+0 | 0.213 | 5.33 | s8mma | 0e+0 |
| cuda | add | int16 | 16777216 | 69.967 | — | add_f32; converted through the host | 0e+0 | 0.139 | 502.64 | add_s16 | 0e+0 | 0.126 | 553.54 | s16 | 0e+0 |
| cuda | matmul | int16 | 2048x2048x2048 | 17.228 | — | gemm128_f32; converted through the host | 0e+0 | 1.588 | 10.85 | gemm128_s16 | 0e+0 | 0.173 | 99.53 | s8mma | 0e+0 |
| cuda | add | int8 | 16777216 | 65.749 | — | add_f32; converted through the host | 0e+0 | 0.074 | 888.50 | add_s8 | 0e+0 | 0.023 | 2922.17 | s8 | 0e+0 |
| cuda | matmul | int8 | 2048x2048x2048 | 22.246 | — | gemm128_f32; converted through the host | 3.23e+2 | 1.550 | 14.35 | gemm128_s8 | 5.12e+2 | 0.093 | 238.69 | s8mma | 3.23e+2 |
| cuda | add | int4 | 16777216 | 79.988 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4347.19 | s8 | 0e+0 |
| cuda | matmul | int4 | 2048x2048x2048 | 38.302 | — | gemm128_f32; converted through the host | 4.43e+2 | — | — | — | — | 0.103 | 370.43 | s8mma | 4.43e+2 |
| cuda | add | int2 | 16777216 | 76.715 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4169.27 | s8 | 0e+0 |
| cuda | matmul | int2 | 2048x2048x2048 | 18.821 | — | gemm128_f32; converted through the host | 6.16e+2 | — | — | — | — | 0.112 | 168.64 | s8mma | 6.16e+2 |
| cuda | add | uint64 | 16777216 | 142.691 | — | add_f32; converted through the host | 0e+0 | 0.999 | 142.86 | add_u64 | 0e+0 | 0.521 | 274.09 | u64 | 0e+0 |
| cuda | matmul | uint64 | 2048x2048x2048 | 34.938 | — | gemm64_f32; converted through the host | 0e+0 | 4.354 | 8.02 | gemm64_u64 | 0e+0 | 0.568 | 61.53 | u8mma | 0e+0 |
| cuda | add | uint32 | 16777216 | 102.895 | — | add_f32; converted through the host | 0e+0 | 0.503 | 204.68 | add_u32 | 0e+0 | 0.502 | 205.09 | u32 | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 23.765 | — | gemm128_f32; converted through the host | 0e+0 | 1.103 | 21.54 | gemm128_u32 | 0e+0 | 0.217 | 109.77 | u8mma | 0e+0 |
| cuda | add | uint16 | 16777216 | 67.362 | — | add_f32; converted through the host | 0e+0 | 0.139 | 483.57 | add_u16 | 0e+0 | 0.126 | 535.04 | u16 | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.006 | — | gemm64_f32; converted through the host | 0e+0 | 1.328 | 12.81 | gemm128_u16 | 0e+0 | 0.170 | 100.04 | u8mma | 0e+0 |
| cuda | add | uint8 | 16777216 | 64.328 | — | add_f32; converted through the host | 0e+0 | 0.089 | 726.05 | add_u8 | 0e+0 | 0.029 | 2249.22 | u8 | 0e+0 |
| cuda | matmul | uint8 | 2048x2048x2048 | 16.924 | — | gemm128_f32; converted through the host | 4.96e+3 | 1.284 | 13.18 | gemm128_u8 | 5.12e+3 | 0.093 | 181.59 | u8mma | 4.96e+3 |
| cuda | add | uint2 | 16777216 | 72.318 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 4304.65 | u8 | 0e+0 |
| cuda | matmul | uint2 | 2048x2048x2048 | 17.897 | — | gemm128_f32; converted through the host | 6.11e+2 | — | — | — | — | 0.110 | 163.29 | u8mma | 6.11e+2 |
| cpu | add | float64 | 16777216 | 6.411 | — | typed tile | 0e+0 | 6.958 | 0.92 | Vector<Double> | 0e+0 | 6.791 | 0.94 | f64 | 0e+0 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.316 | — | to f32 and back | 7.96e-5 | 17.712 | 0.30 | Vector<Double> | 0e+0 | 3.348 | 1.59 | f64 | 6.39e-14 |
| cpu | softmax | float64 | 4096x4096 | 31.587 | — | to f32 and back | 6.21e-10 | 6.332 | 4.99 | Vector<Double> | 4.34e-19 | 4.818 | 6.56 | f64 | 1.08e-17 |
| cpu | rms_norm | float64 | 4096x4096 | 30.102 | — | to f32 and back | 3.73e-6 | 4.059 | 7.42 | Vector<Double> | 1.33e-14 | 4.847 | 6.21 | f64 | 1.27e-14 |
| cpu | add | float32 | 16777216 | 3.234 | — | f32 kernel | 5.96e-8 | 3.198 | 1.01 | Vector<Single> | 5.96e-8 | 3.025 | 1.07 | f32 | 5.96e-8 |
| cpu | matmul | float32 | 1024x1024x1024 | 1.971 | — | f32 kernel | 7.96e-5 | 3.356 | 0.59 | Vector<Single> | 8.34e-5 | 1.576 | 1.25 | f32 | 7.96e-5 |
| cpu | softmax | float32 | 4096x4096 | 2.609 | — | f32 kernel | 6.21e-10 | 4.363 | 0.60 | Vector<Single> | 5.91e-9 | 1.861 | 1.40 | f32 | 6.21e-10 |
| cpu | rms_norm | float32 | 4096x4096 | 2.485 | — | f32 kernel | 3.73e-6 | 2.292 | 1.08 | Vector<Single> | 5.16e-7 | 1.778 | 1.40 | f32 | 3.68e-7 |
| cpu | add | float16 | 16777216 | 2.926 | — | typed tile | 4.88e-4 | 7.518 | 0.39 | scalar | 4.88e-4 | 1.167 | 2.51 | f32 | 4.88e-4 |
| cpu | matmul | float16 | 1024x1024x1024 | 7.148 | — | to f32 and back | 1.56e-2 | 727.236 | 0.01 | scalar | 6.08e-1 | 1.708 | 4.18 | f32 | 1.56e-2 |
| cpu | softmax | float16 | 4096x4096 | 44.536 | — | to f32 and back | 9.54e-7 | 59.692 | 0.75 | scalar | 1.57e-4 | 3.321 | 13.41 | f32 | 9.54e-7 |
| cpu | rms_norm | float16 | 4096x4096 | 45.864 | — | to f32 and back | 4.93e-4 | 25.053 | 1.83 | scalar | 7.06e-2 | 1.225 | 37.45 | f32 | 4.89e-4 |
| cpu | add | bfloat16 | 16777216 | 1.423 | — | typed tile | 3.91e-3 | — | — | — | — | 1.250 | 1.14 | f32 | 3.91e-3 |
| cpu | matmul | bfloat16 | 1024x1024x1024 | 5.483 | — | to f32 and back | 1.25e-1 | — | — | — | — | 1.805 | 3.04 | f32 | 1.25e-1 |
| cpu | softmax | bfloat16 | 4096x4096 | 23.261 | — | to f32 and back | 7.63e-6 | — | — | — | — | 1.769 | 13.15 | f32 | 7.63e-6 |
| cpu | rms_norm | bfloat16 | 4096x4096 | 23.925 | — | to f32 and back | 3.91e-3 | — | — | — | — | 0.402 | 59.54 | f32 | 3.91e-3 |
| cpu | add | float8_e4m3 | 16777216 | 22.305 | — | typed tile | 6.25e-2 | — | — | — | — | 1.804 | 12.36 | f32 | 6.25e-2 |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | 35.778 | — | to f32 and back | 2e+0 | — | — | — | — | 1.224 | 29.23 | f32 | 2e+0 |
| cpu | softmax | float8_e4m3 | 4096x4096 | 90.450 | — | to f32 and back | 9.76e-4 | — | — | — | — | 3.085 | 29.32 | f32 | 9.76e-4 |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | 402.350 | — | to f32 and back | 6.25e-2 | — | — | — | — | 1.617 | 248.83 | f32 | 6.25e-2 |
| cpu | add | float8_e5m2 | 16777216 | 17.597 | — | typed tile | 1.25e-1 | — | — | — | — | 1.987 | 8.86 | f32 | 1.25e-1 |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | 29.490 | — | to f32 and back | 4e+0 | — | — | — | — | 1.236 | 23.86 | f32 | 4e+0 |
| cpu | softmax | float8_e5m2 | 4096x4096 | 282.744 | — | to f32 and back | 1.62e-4 | — | — | — | — | 3.244 | 87.17 | f32 | 1.62e-4 |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | 313.079 | — | to f32 and back | 1.08e-1 | — | — | — | — | 1.701 | 184.06 | f32 | 1.08e-1 |
| cpu | add | int64 | 16777216 | 27.110 | — | typed tile | 0e+0 | 6.902 | 3.93 | Vector<Int64> | 0e+0 | 6.834 | 3.97 | s64 | 0e+0 |
| cpu | matmul | int64 | 1024x1024x1024 | 6.489 | — | to f32 and back | 0e+0 | 28.839 | 0.23 | Vector<Int64> | 0e+0 | 0.780 | 8.32 | s8dp | 0e+0 |
| cpu | add | int32 | 16777216 | 22.932 | — | typed tile | 0e+0 | 3.047 | 7.53 | Vector<Int32> | 0e+0 | 2.929 | 7.83 | s32 | 0e+0 |
| cpu | matmul | int32 | 1024x1024x1024 | 6.364 | — | to f32 and back | 0e+0 | 4.599 | 1.38 | Vector<Int32> | 0e+0 | 0.380 | 16.74 | s8dp | 0e+0 |
| cpu | add | int16 | 16777216 | 1.428 | — | typed tile | 0e+0 | 1.005 | 1.42 | Vector<Int16> | 0e+0 | 0.889 | 1.61 | s16 | 0e+0 |
| cpu | matmul | int16 | 1024x1024x1024 | 6.047 | — | to f32 and back | 0e+0 | 2.369 | 2.55 | Vector<Int16> | 0e+0 | 0.743 | 8.14 | s8dp | 0e+0 |
| cpu | add | int8 | 16777216 | 1.944 | — | typed tile | 0e+0 | 0.360 | 5.40 | Vector<SByte> | 0e+0 | 0.130 | 14.95 | s8 | 0e+0 |
| cpu | matmul | int8 | 1024x1024x1024 | 6.610 | — | to f32 and back | 1.76e+2 | 3.932 | 1.68 | Vector<SByte> | 2.56e+2 | 0.578 | 11.44 | s8dp | 1.76e+2 |
| cpu | add | int4 | 16777216 | 2.974 | — | typed tile | 0e+0 | — | — | — | — | 0.143 | 20.75 | s8 | 0e+0 |
| cpu | matmul | int4 | 1024x1024x1024 | 11.944 | — | to f32 and back | 2.96e+2 | — | — | — | — | 0.514 | 23.23 | s8dp | 2.96e+2 |
| cpu | add | int2 | 16777216 | 2.904 | — | typed tile | 0e+0 | — | — | — | — | 0.267 | 10.86 | s8 | 0e+0 |
| cpu | matmul | int2 | 1024x1024x1024 | 7.082 | — | to f32 and back | 3.26e+2 | — | — | — | — | 0.506 | 13.99 | s8dp | 3.26e+2 |
| cpu | add | uint64 | 16777216 | 13.256 | — | typed tile | 0e+0 | 6.949 | 1.91 | Vector<UInt64> | 0e+0 | 6.850 | 1.94 | u64 | 0e+0 |
| cpu | matmul | uint64 | 1024x1024x1024 | 7.178 | — | to f32 and back | 0e+0 | 33.600 | 0.21 | Vector<UInt64> | 0e+0 | 0.845 | 8.49 | u8dp | 0e+0 |
| cpu | add | uint32 | 16777216 | 14.566 | — | typed tile | 0e+0 | 3.034 | 4.80 | Vector<UInt32> | 0e+0 | 2.983 | 4.88 | u32 | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.850 | — | to f32 and back | 0e+0 | 6.607 | 1.04 | Vector<UInt32> | 0e+0 | 0.544 | 12.59 | u8dp | 0e+0 |
| cpu | add | uint16 | 16777216 | 1.602 | — | typed tile | 0e+0 | 1.368 | 1.17 | Vector<UInt16> | 0e+0 | 0.874 | 1.83 | u16 | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.232 | — | to f32 and back | 0e+0 | 2.292 | 2.28 | Vector<UInt16> | 0e+0 | 0.702 | 7.45 | u8dp | 0e+0 |
| cpu | add | uint8 | 16777216 | 1.630 | — | typed tile | 0e+0 | 0.436 | 3.74 | Vector<Byte> | 0e+0 | 0.132 | 12.35 | u8 | 0e+0 |
| cpu | matmul | uint8 | 1024x1024x1024 | 5.752 | — | to f32 and back | 2.48e+3 | 2.135 | 2.69 | Vector<Byte> | 2.56e+3 | 0.566 | 10.16 | u8dp | 2.48e+3 |
| cpu | add | uint2 | 16777216 | 2.190 | — | typed tile | 0e+0 | — | — | — | — | 0.155 | 14.09 | u8 | 0e+0 |
| cpu | matmul | uint2 | 1024x1024x1024 | 6.867 | — | to f32 and back | 3.18e+2 | — | — | — | — | 0.524 | 13.11 | u8dp | 3.18e+2 |
