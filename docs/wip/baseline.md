# Qiyas bench

2026-11-06 16:12. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | add | float64 | 16777216 | 138.073 | — | add_f32; converted through the host | 5.96e-8 | 0.521 | 265.27 | add_f64 | 0e+0 | 0.521 | 265.22 | f64 | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 32.791 | — | gemm128_f32; converted through the host | 1.62e-4 | 23.110 | 1.42 | gemm64_f64 | 2.59e-13 | 4.164 | 7.88 | f64x | 0e+0 |
| cuda | softmax | float64 | 4096x4096 | 83.665 | — | softmax_f32; converted through the host | 5.29e-10 | 0.395 | 211.76 | softmax_f64 | 1.88e-10 | 1.745 | 47.96 | f64 | 1.13e-17 |
| cuda | rms_norm | float64 | 4096x4096 | 83.433 | — | rms_norm_f32; converted through the host | 3.1e-7 | 0.370 | 225.56 | rms_norm_f64 | 1.24e-14 | 0.550 | 151.67 | f64 | 1.07e-14 |
| cuda | add | float32 | 16777216 | 0.262 | — | add_f32 | 5.96e-8 | 0.261 | 1.00 | add_f32 | 5.96e-8 | 0.262 | 1.00 | f32 | 5.96e-8 |
| cuda | matmul | float32 | 2048x2048x2048 | 0.978 | — | gemm128_f32 | 1.62e-4 | 0.835 | 1.17 | gemm128_f32 | 1.62e-4 | 0.683 | 1.43 | f32 | 1.62e-4 |
| cuda | softmax | float32 | 4096x4096 | 0.181 | — | softmax_f32 | 5.29e-10 | 0.179 | 1.01 | softmax_f32 | 5.11e-10 | 0.177 | 1.02 | f32 | 4.52e-10 |
| cuda | rms_norm | float32 | 4096x4096 | 0.182 | — | rms_norm_f32 | 3.1e-7 | 0.181 | 1.01 | rms_norm_f32 | 2.93e-7 | 0.178 | 1.02 | f32 | 2.97e-7 |
| cuda | add | float16 | 16777216 | 0.661 | — | add_f32; converted on the device | 4.88e-4 | 0.140 | 4.72 | add_f16 | 4.88e-4 | 0.125 | 5.30 | f32 | 4.88e-4 |
| cuda | matmul | float16 | 2048x2048x2048 | 1.013 | — | gemm128_f32; converted on the device | 3.09e-2 | 0.656 | 1.54 | gemm128_f16 | 1.51e+0 | 0.243 | 4.18 | f16mma | 3.09e-2 |
| cuda | softmax | float16 | 4096x4096 | 0.444 | — | softmax_f32; converted on the device | 9.53e-7 | 0.092 | 4.85 | softmax_f16 | 3.3e-6 | 0.075 | 5.95 | f32 | 9.54e-7 |
| cuda | rms_norm | float16 | 4096x4096 | 0.443 | — | rms_norm_f32; converted on the device | 4.88e-4 | 0.090 | 4.92 | rms_norm_f16 | 1.64e-3 | 0.079 | 5.62 | f32 | 4.88e-4 |
| cuda | add | bfloat16 | 16777216 | 0.661 | — | add_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.125 | 5.31 | f32 | 3.91e-3 |
| cuda | matmul | bfloat16 | 2048x2048x2048 | 1.039 | — | gemm128_f32; converted on the device | 2.5e-1 | — | — | — | — | 0.243 | 4.28 | bf16mma | 2.5e-1 |
| cuda | softmax | bfloat16 | 4096x4096 | 0.445 | — | softmax_f32; converted on the device | 7.63e-6 | — | — | — | — | 0.084 | 5.27 | f32 | 7.63e-6 |
| cuda | rms_norm | bfloat16 | 4096x4096 | 0.443 | — | rms_norm_f32; converted on the device | 3.91e-3 | — | — | — | — | 0.083 | 5.34 | f32 | 3.91e-3 |
| cuda | add | float8_e4m3 | 16777216 | 493.296 | — | add_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.050 | 9826.60 | f32 | 6.25e-2 |
| cuda | matmul | float8_e4m3 | 2048x2048x2048 | 134.323 | — | gemm128_f32; converted through the host | 3.99e+0 | — | — | — | — | 0.268 | 501.77 | f16mma | 3.99e+0 |
| cuda | softmax | float8_e4m3 | 4096x4096 | 104.446 | — | softmax_f32; converted through the host | 9.76e-4 | — | — | — | — | 0.069 | 1522.54 | f32 | 9.76e-4 |
| cuda | rms_norm | float8_e4m3 | 4096x4096 | 451.950 | — | rms_norm_f32; converted through the host | 6.25e-2 | — | — | — | — | 0.046 | 9825.01 | f32 | 6.25e-2 |
| cuda | add | float8_e5m2 | 16777216 | 471.957 | — | add_f32; converted through the host | 1.25e-1 | — | — | — | — | 0.819 | 576.47 | f32 | 1.25e-1 |
| cuda | matmul | float8_e5m2 | 2048x2048x2048 | 114.127 | — | gemm128_f32; converted through the host | 7.71e+0 | — | — | — | — | 0.263 | 433.94 | f16mma | 7.71e+0 |
| cuda | softmax | float8_e5m2 | 4096x4096 | 294.417 | — | softmax_f32; converted through the host | 1.62e-4 | — | — | — | — | 0.069 | 4298.06 | f32 | 1.62e-4 |
| cuda | rms_norm | float8_e5m2 | 4096x4096 | 344.947 | — | rms_norm_f32; converted through the host | 1.08e-1 | — | — | — | — | 0.050 | 6885.17 | f32 | 1.08e-1 |
| cuda | add | int64 | 16777216 | 114.237 | — | add_f32; converted through the host | 0e+0 | 0.520 | 219.90 | add_s64 | 0e+0 | 0.520 | 219.56 | s64 | 0e+0 |
| cuda | matmul | int64 | 2048x2048x2048 | 29.901 | — | gemm128_f32; converted through the host | 0e+0 | 4.432 | 6.75 | gemm64_s64 | 0e+0 | 0.356 | 84.06 | s8mma | 0e+0 |
| cuda | add | int32 | 16777216 | 0.786 | — | add_f32; converted on the device | 0e+0 | 0.262 | 3.00 | add_s32 | 0e+0 | 0.262 | 3.00 | s32 | 0e+0 |
| cuda | matmul | int32 | 2048x2048x2048 | 1.145 | — | gemm128_f32; converted on the device | 0e+0 | 1.087 | 1.05 | gemm128_s32 | 0e+0 | 0.209 | 5.48 | s8mma | 0e+0 |
| cuda | add | int16 | 16777216 | 66.948 | — | add_f32; converted through the host | 0e+0 | 0.139 | 480.60 | add_s16 | 0e+0 | 0.125 | 536.01 | s16 | 0e+0 |
| cuda | matmul | int16 | 2048x2048x2048 | 18.750 | — | gemm128_f32; converted through the host | 0e+0 | 1.587 | 11.81 | gemm128_s16 | 0e+0 | 0.168 | 111.87 | s8mma | 0e+0 |
| cuda | add | int8 | 16777216 | 62.565 | — | add_f32; converted through the host | 0e+0 | 0.084 | 745.71 | add_s8 | 0e+0 | 0.023 | 2780.66 | s8 | 0e+0 |
| cuda | matmul | int8 | 2048x2048x2048 | 21.620 | — | gemm128_f32; converted through the host | 3.23e+2 | 1.550 | 13.95 | gemm128_s8 | 5.12e+2 | 0.094 | 229.51 | s8mma | 3.23e+2 |
| cuda | add | int4 | 16777216 | 78.165 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4248.11 | s8 | 0e+0 |
| cuda | matmul | int4 | 2048x2048x2048 | 39.171 | — | gemm64_f32; converted through the host | 4.43e+2 | — | — | — | — | 0.103 | 378.83 | s8mma | 4.43e+2 |
| cuda | add | int2 | 16777216 | 76.875 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.018 | 4177.99 | s8 | 0e+0 |
| cuda | matmul | int2 | 2048x2048x2048 | 18.882 | — | gemm64_f32; converted through the host | 6.16e+2 | — | — | — | — | 0.110 | 171.35 | s8mma | 6.16e+2 |
| cuda | add | uint64 | 16777216 | 140.024 | — | add_f32; converted through the host | 0e+0 | 0.999 | 140.18 | add_u64 | 0e+0 | 0.519 | 269.74 | u64 | 0e+0 |
| cuda | matmul | uint64 | 2048x2048x2048 | 32.695 | — | gemm128_f32; converted through the host | 0e+0 | 4.362 | 7.50 | gemm64_u64 | 0e+0 | 0.354 | 92.28 | u8mma | 0e+0 |
| cuda | add | uint32 | 16777216 | 99.284 | — | add_f32; converted through the host | 0e+0 | 0.261 | 379.96 | add_u32 | 0e+0 | 0.261 | 380.84 | u32 | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 24.795 | — | gemm128_f32; converted through the host | 0e+0 | 1.085 | 22.86 | gemm128_u32 | 0e+0 | 0.208 | 119.44 | u8mma | 0e+0 |
| cuda | add | uint16 | 16777216 | 66.329 | — | add_f32; converted through the host | 0e+0 | 0.140 | 472.77 | add_u16 | 0e+0 | 0.125 | 531.06 | u16 | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.691 | — | gemm128_f32; converted through the host | 0e+0 | 1.328 | 13.32 | gemm128_u16 | 0e+0 | 0.167 | 105.93 | u8mma | 0e+0 |
| cuda | add | uint8 | 16777216 | 62.460 | — | add_f32; converted through the host | 0e+0 | 0.072 | 866.29 | add_u8 | 0e+0 | 0.023 | 2775.98 | u8 | 0e+0 |
| cuda | matmul | uint8 | 2048x2048x2048 | 16.700 | — | gemm128_f32; converted through the host | 4.96e+3 | 1.294 | 12.91 | gemm128_u8 | 5.12e+3 | 0.094 | 178.42 | u8mma | 4.96e+3 |
| cuda | add | uint2 | 16777216 | 73.026 | — | add_f32; converted through the host | 0e+0 | — | — | — | — | 0.017 | 4399.14 | u8 | 0e+0 |
| cuda | matmul | uint2 | 2048x2048x2048 | 17.772 | — | gemm128_f32; converted through the host | 6.11e+2 | — | — | — | — | 0.110 | 161.27 | u8mma | 6.11e+2 |
| cuda | add | q8_0 | 16777216 | — | — | — | — | — | — | — | — | 0.985 | — | f32 | 7.87e-3 |
| cuda | matmul | q8_0 | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.861 | — | f32 | 3.09e-1 |
| cuda | softmax | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.780 | — | f32 (blocks through lane storage) | 8.37e-6 |
| cuda | rms_norm | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.776 | — | f32 (blocks through lane storage) | 4.85e-4 |
| cuda | add | q4_0 | 16777216 | — | — | — | — | — | — | — | — | 0.659 | — | f32 | 2.34e-1 |
| cuda | matmul | q4_0 | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.784 | — | f32 | 6.51e+0 |
| cuda | softmax | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.458 | — | f32 (blocks through lane storage) | 1.11e-4 |
| cuda | rms_norm | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 0.454 | — | f32 (blocks through lane storage) | 4.88e-4 |
| cuda | add | q4_k | 16777216 | — | — | — | — | — | — | — | — | 1.190 | — | f32 | 1.33e-1 |
| cuda | matmul | q4_k | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.868 | — | f32 | 3.83e+0 |
| cuda | softmax | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 0.986 | — | f32 (blocks through lane storage) | 7.31e-5 |
| cuda | rms_norm | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 0.983 | — | f32 (blocks through lane storage) | 1.37e-3 |
| cuda | add | q6_k | 16777216 | — | — | — | — | — | — | — | — | 1.401 | — | f32 | 3.22e-2 |
| cuda | matmul | q6_k | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.953 | — | f32 | 1.25e+0 |
| cuda | softmax | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 1.190 | — | f32 (blocks through lane storage) | 1.17e-4 |
| cuda | rms_norm | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 1.193 | — | f32 (blocks through lane storage) | 4.69e-4 |
| cuda | add | bool | 16777216 | — | — | — | — | — | — | — | — | 0.017 | — | u8 | 1e+0 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cpu | add | float64 | 16777216 | 7.201 | — | typed tile | 0e+0 | 6.857 | 1.05 | Vector<Double> | 0e+0 | 6.759 | 1.07 | f64 | 0e+0 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.484 | — | to f32 and back | 7.96e-5 | 17.908 | 0.31 | Vector<Double> | 1.14e-13 | 2.911 | 1.88 | f64 | 1.14e-13 |
| cpu | softmax | float64 | 4096x4096 | 31.080 | — | to f32 and back | 6.21e-10 | 6.766 | 4.59 | Vector<Double> | 4.34e-19 | 5.176 | 6.00 | f64 | 1.08e-17 |
| cpu | rms_norm | float64 | 4096x4096 | 31.078 | — | to f32 and back | 3.73e-6 | 4.040 | 7.69 | Vector<Double> | 1.33e-14 | 4.752 | 6.54 | f64 | 1.27e-14 |
| cpu | add | float32 | 16777216 | 3.153 | — | f32 kernel | 5.96e-8 | 3.037 | 1.04 | Vector<Single> | 5.96e-8 | 3.041 | 1.04 | f32 | 5.96e-8 |
| cpu | matmul | float32 | 1024x1024x1024 | 3.137 | — | f32 kernel | 7.96e-5 | 4.273 | 0.73 | Vector<Single> | 8.34e-5 | 1.800 | 1.74 | f32 | 7.96e-5 |
| cpu | softmax | float32 | 4096x4096 | 3.673 | — | f32 kernel | 6.21e-10 | 4.869 | 0.75 | Vector<Single> | 5.91e-9 | 1.878 | 1.96 | f32 | 6.21e-10 |
| cpu | rms_norm | float32 | 4096x4096 | 2.841 | — | f32 kernel | 3.73e-6 | 2.060 | 1.38 | Vector<Single> | 5.16e-7 | 1.666 | 1.71 | f32 | 3.68e-7 |
| cpu | add | float16 | 16777216 | 3.141 | — | typed tile | 4.88e-4 | 7.613 | 0.41 | scalar | 4.88e-4 | 1.746 | 1.80 | f32 | 4.88e-4 |
| cpu | matmul | float16 | 1024x1024x1024 | 4.995 | — | to f32 and back | 1.56e-2 | 887.073 | 0.01 | scalar | 6.08e-1 | 1.250 | 4.00 | f32 | 1.56e-2 |
| cpu | softmax | float16 | 4096x4096 | 46.076 | — | to f32 and back | 9.54e-7 | 47.782 | 0.96 | scalar | 1.57e-4 | 2.231 | 20.66 | f32 | 9.54e-7 |
| cpu | rms_norm | float16 | 4096x4096 | 45.190 | — | to f32 and back | 4.93e-4 | 20.931 | 2.16 | scalar | 7.06e-2 | 0.824 | 54.82 | f32 | 4.89e-4 |
| cpu | add | bfloat16 | 16777216 | 1.353 | — | typed tile | 3.91e-3 | — | — | — | — | 0.908 | 1.49 | f32 | 3.91e-3 |
| cpu | matmul | bfloat16 | 1024x1024x1024 | 3.820 | — | to f32 and back | 1.25e-1 | — | — | — | — | 1.812 | 2.11 | f32 | 1.25e-1 |
| cpu | softmax | bfloat16 | 4096x4096 | 24.537 | — | to f32 and back | 7.63e-6 | — | — | — | — | 3.028 | 8.10 | f32 | 7.63e-6 |
| cpu | rms_norm | bfloat16 | 4096x4096 | 23.852 | — | to f32 and back | 3.91e-3 | — | — | — | — | 0.418 | 57.05 | f32 | 3.91e-3 |
| cpu | add | float8_e4m3 | 16777216 | 27.216 | — | typed tile | 6.25e-2 | — | — | — | — | 2.130 | 12.78 | f32 | 6.25e-2 |
| cpu | matmul | float8_e4m3 | 1024x1024x1024 | 27.213 | — | to f32 and back | 2e+0 | — | — | — | — | 1.774 | 15.34 | f32 | 2e+0 |
| cpu | softmax | float8_e4m3 | 4096x4096 | 99.133 | — | to f32 and back | 9.76e-4 | — | — | — | — | 3.072 | 32.27 | f32 | 9.76e-4 |
| cpu | rms_norm | float8_e4m3 | 4096x4096 | 409.439 | — | to f32 and back | 6.25e-2 | — | — | — | — | 1.633 | 250.73 | f32 | 6.25e-2 |
| cpu | add | float8_e5m2 | 16777216 | 22.739 | — | typed tile | 1.25e-1 | — | — | — | — | 1.894 | 12.01 | f32 | 1.25e-1 |
| cpu | matmul | float8_e5m2 | 1024x1024x1024 | 31.069 | — | to f32 and back | 4e+0 | — | — | — | — | 1.265 | 24.57 | f32 | 4e+0 |
| cpu | softmax | float8_e5m2 | 4096x4096 | 286.265 | — | to f32 and back | 1.62e-4 | — | — | — | — | 3.243 | 88.26 | f32 | 1.62e-4 |
| cpu | rms_norm | float8_e5m2 | 4096x4096 | 314.273 | — | to f32 and back | 1.08e-1 | — | — | — | — | 1.739 | 180.70 | f32 | 1.08e-1 |
| cpu | add | int64 | 16777216 | 17.804 | — | typed tile | 0e+0 | 6.923 | 2.57 | Vector<Int64> | 0e+0 | 6.785 | 2.62 | s64 | 0e+0 |
| cpu | matmul | int64 | 1024x1024x1024 | 6.577 | — | to f32 and back | 0e+0 | 31.348 | 0.21 | Vector<Int64> | 0e+0 | 1.137 | 5.78 | s8dp | 0e+0 |
| cpu | add | int32 | 16777216 | 22.983 | — | typed tile | 0e+0 | 2.989 | 7.69 | Vector<Int32> | 0e+0 | 3.011 | 7.63 | s32 | 0e+0 |
| cpu | matmul | int32 | 1024x1024x1024 | 5.881 | — | to f32 and back | 0e+0 | 4.141 | 1.42 | Vector<Int32> | 0e+0 | 0.836 | 7.04 | s8dp | 0e+0 |
| cpu | add | int16 | 16777216 | 1.627 | — | typed tile | 0e+0 | 1.277 | 1.27 | Vector<Int16> | 0e+0 | 0.963 | 1.69 | s16 | 0e+0 |
| cpu | matmul | int16 | 1024x1024x1024 | 5.447 | — | to f32 and back | 0e+0 | 2.239 | 2.43 | Vector<Int16> | 0e+0 | 0.532 | 10.24 | s8dp | 0e+0 |
| cpu | add | int8 | 16777216 | 1.914 | — | typed tile | 0e+0 | 0.266 | 7.18 | Vector<SByte> | 0e+0 | 0.132 | 14.50 | s8 | 0e+0 |
| cpu | matmul | int8 | 1024x1024x1024 | 6.978 | — | to f32 and back | 1.76e+2 | 5.462 | 1.28 | Vector<SByte> | 2.56e+2 | 0.574 | 12.16 | s8dp | 1.76e+2 |
| cpu | add | int4 | 16777216 | 3.029 | — | typed tile | 0e+0 | — | — | — | — | 0.145 | 20.83 | s8 | 0e+0 |
| cpu | matmul | int4 | 1024x1024x1024 | 12.890 | — | to f32 and back | 2.96e+2 | — | — | — | — | 0.518 | 24.91 | s8dp | 2.96e+2 |
| cpu | add | int2 | 16777216 | 2.874 | — | typed tile | 0e+0 | — | — | — | — | 0.286 | 10.04 | s8 | 0e+0 |
| cpu | matmul | int2 | 1024x1024x1024 | 7.673 | — | to f32 and back | 3.26e+2 | — | — | — | — | 0.522 | 14.70 | s8dp | 3.26e+2 |
| cpu | add | uint64 | 16777216 | 8.148 | — | typed tile | 0e+0 | 6.856 | 1.19 | Vector<UInt64> | 0e+0 | 6.819 | 1.19 | u64 | 0e+0 |
| cpu | matmul | uint64 | 1024x1024x1024 | 7.438 | — | to f32 and back | 0e+0 | 26.237 | 0.28 | Vector<UInt64> | 0e+0 | 1.154 | 6.44 | u8dp | 0e+0 |
| cpu | add | uint32 | 16777216 | 9.692 | — | typed tile | 0e+0 | 2.995 | 3.24 | Vector<UInt32> | 0e+0 | 2.999 | 3.23 | u32 | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.051 | — | to f32 and back | 0e+0 | 4.207 | 1.44 | Vector<UInt32> | 0e+0 | 0.628 | 9.63 | u8dp | 0e+0 |
| cpu | add | uint16 | 16777216 | 1.401 | — | typed tile | 0e+0 | 1.108 | 1.26 | Vector<UInt16> | 0e+0 | 0.845 | 1.66 | u16 | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.094 | — | to f32 and back | 0e+0 | 1.817 | 2.25 | Vector<UInt16> | 0e+0 | 0.367 | 11.14 | u8dp | 0e+0 |
| cpu | add | uint8 | 16777216 | 1.664 | — | typed tile | 0e+0 | 0.312 | 5.34 | Vector<Byte> | 0e+0 | 0.200 | 8.32 | u8 | 0e+0 |
| cpu | matmul | uint8 | 1024x1024x1024 | 6.721 | — | to f32 and back | 2.48e+3 | 4.354 | 1.54 | Vector<Byte> | 2.56e+3 | 0.574 | 11.70 | u8dp | 2.48e+3 |
| cpu | add | uint2 | 16777216 | 2.592 | — | typed tile | 0e+0 | — | — | — | — | 0.237 | 10.94 | u8 | 0e+0 |
| cpu | matmul | uint2 | 1024x1024x1024 | 6.995 | — | to f32 and back | 3.18e+2 | — | — | — | — | 0.525 | 13.33 | u8dp | 3.18e+2 |
| cpu | add | q8_0 | 16777216 | — | — | — | — | — | — | — | — | 6.174 | — | f32 | 7.87e-3 |
| cpu | matmul | q8_0 | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.947 | — | f32 | 2.03e-1 |
| cpu | softmax | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 7.534 | — | f32 | 8.37e-6 |
| cpu | rms_norm | q8_0 | 4096x4096 | — | — | — | — | — | — | — | — | 4.531 | — | f32 | 4.85e-4 |
| cpu | add | q4_0 | 16777216 | — | — | — | — | — | — | — | — | 3.656 | — | f32 | 2.34e-1 |
| cpu | matmul | q4_0 | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.624 | — | f32 | 4.06e+0 |
| cpu | softmax | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 4.245 | — | f32 | 1.11e-4 |
| cpu | rms_norm | q4_0 | 4096x4096 | — | — | — | — | — | — | — | — | 3.433 | — | f32 | 4.89e-4 |
| cpu | add | q4_k | 16777216 | — | — | — | — | — | — | — | — | 5.060 | — | f32 | 1.33e-1 |
| cpu | matmul | q4_k | 1024x1024x1024 | — | — | — | — | — | — | — | — | 8.978 | — | f32 | 2.71e+0 |
| cpu | softmax | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 8.091 | — | f32 | 7.31e-5 |
| cpu | rms_norm | q4_k | 4096x4096 | — | — | — | — | — | — | — | — | 5.405 | — | f32 | 1.37e-3 |
| cpu | add | q6_k | 16777216 | — | — | — | — | — | — | — | — | 11.590 | — | f32 | 3.22e-2 |
| cpu | matmul | q6_k | 1024x1024x1024 | — | — | — | — | — | — | — | — | 10.054 | — | f32 | 8.19e-1 |
| cpu | softmax | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 9.848 | — | f32 | 1.17e-4 |
| cpu | rms_norm | q6_k | 4096x4096 | — | — | — | — | — | — | — | — | 10.029 | — | f32 | 4.69e-4 |
| cpu | add | bool | 16777216 | — | — | — | — | — | — | — | — | 4.958 | — | u8 | 1e+0 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.411 | — | f32 | 6.46e+2 |
