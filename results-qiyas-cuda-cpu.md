# Qiyas bench

2026-11-06 12:55. Median wall time (ms); `×` = lib ÷ subject (above 1: faster than lib). `err` = largest |result − float64 math on the same stored inputs|. `lane` = what the subject computed in.

| device | operation | type | shape | lib ms | lib × | lib lane | lib err | number ms | number × | number lane | number err | qiyas ms | qiyas × | qiyas lane | qiyas err |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| cuda | matmul | float64 | 2048x2048x2048 | 29.586 | — | gemm64_f32; converted through the host | 1.62e-4 | 22.640 | 1.31 | gemm64_f64 | 2.59e-13 | 4.173 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.916 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.641 | 1.32 | gemm64_f64 | 2.59e-13 | 4.166 | 7.18 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.833 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.639 | 1.36 | gemm64_f64 | 2.59e-13 | 4.172 | 7.39 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.022 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.638 | 1.33 | gemm64_f64 | 2.59e-13 | 4.168 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.776 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.638 | 1.32 | gemm64_f64 | 2.59e-13 | 4.171 | 7.14 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.772 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.640 | 1.32 | gemm64_f64 | 2.59e-13 | 4.174 | 7.13 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.615 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.635 | 1.31 | gemm64_f64 | 2.59e-13 | 4.175 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.044 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.635 | 1.33 | gemm64_f64 | 2.59e-13 | 4.174 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.006 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.635 | 1.33 | gemm64_f64 | 2.59e-13 | 4.170 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.933 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.633 | 1.32 | gemm64_f64 | 2.59e-13 | 4.164 | 7.19 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.317 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.639 | 1.30 | gemm64_f64 | 2.59e-13 | 4.175 | 7.02 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.976 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.638 | 1.32 | gemm64_f64 | 2.59e-13 | 4.177 | 7.18 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.350 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.635 | 1.34 | gemm64_f64 | 2.59e-13 | 4.173 | 7.27 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.669 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.632 | 1.31 | gemm64_f64 | 2.59e-13 | 4.173 | 7.11 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.925 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.636 | 1.32 | gemm64_f64 | 2.59e-13 | 4.174 | 7.17 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.801 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.634 | 1.32 | gemm64_f64 | 2.59e-13 | 4.177 | 7.13 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.517 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.633 | 1.30 | gemm64_f64 | 2.59e-13 | 4.164 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.489 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.630 | 1.30 | gemm64_f64 | 2.59e-13 | 4.177 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.578 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.633 | 1.31 | gemm64_f64 | 2.59e-13 | 4.174 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.329 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.624 | 1.30 | gemm64_f64 | 2.59e-13 | 4.175 | 7.02 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.858 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.632 | 1.32 | gemm64_f64 | 2.59e-13 | 4.162 | 7.17 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.821 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.628 | 1.32 | gemm64_f64 | 2.59e-13 | 4.176 | 7.14 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.579 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.628 | 1.31 | gemm64_f64 | 2.59e-13 | 4.173 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.451 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.629 | 1.30 | gemm64_f64 | 2.59e-13 | 4.170 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.659 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.630 | 1.31 | gemm64_f64 | 2.59e-13 | 4.173 | 7.11 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.877 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.632 | 1.32 | gemm64_f64 | 2.59e-13 | 4.174 | 7.16 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.629 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.628 | 1.31 | gemm64_f64 | 2.59e-13 | 4.172 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.703 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.625 | 1.31 | gemm64_f64 | 2.59e-13 | 4.165 | 7.13 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.630 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.633 | 1.31 | gemm64_f64 | 2.59e-13 | 4.175 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.654 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.625 | 1.31 | gemm64_f64 | 2.59e-13 | 4.177 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.863 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.626 | 1.32 | gemm64_f64 | 2.59e-13 | 4.172 | 7.16 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.645 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.627 | 1.31 | gemm64_f64 | 2.59e-13 | 4.176 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.196 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.631 | 1.29 | gemm64_f64 | 2.59e-13 | 4.175 | 6.99 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.892 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.632 | 1.32 | gemm64_f64 | 2.59e-13 | 4.174 | 7.16 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.604 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.629 | 1.31 | gemm64_f64 | 2.59e-13 | 4.170 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.815 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.625 | 1.32 | gemm64_f64 | 2.59e-13 | 4.174 | 7.14 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.660 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.626 | 1.31 | gemm64_f64 | 2.59e-13 | 4.175 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.559 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.629 | 1.31 | gemm64_f64 | 2.59e-13 | 4.176 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.495 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.682 | 1.30 | gemm64_f64 | 2.59e-13 | 4.173 | 7.07 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.663 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.691 | 1.31 | gemm64_f64 | 2.59e-13 | 4.178 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.613 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.687 | 1.31 | gemm64_f64 | 2.59e-13 | 4.176 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.596 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.688 | 1.30 | gemm64_f64 | 2.59e-13 | 4.176 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.069 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.692 | 1.33 | gemm64_f64 | 2.59e-13 | 4.175 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.530 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.691 | 1.30 | gemm64_f64 | 2.59e-13 | 4.177 | 7.07 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.041 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.694 | 1.32 | gemm64_f64 | 2.59e-13 | 4.174 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.552 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.693 | 1.30 | gemm64_f64 | 2.59e-13 | 4.173 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.307 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.691 | 1.29 | gemm64_f64 | 2.59e-13 | 4.175 | 7.02 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.057 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.687 | 1.32 | gemm64_f64 | 2.59e-13 | 4.177 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.810 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.690 | 1.31 | gemm64_f64 | 2.59e-13 | 4.176 | 7.14 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.060 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.690 | 1.32 | gemm64_f64 | 2.59e-13 | 4.173 | 7.20 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.606 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.30 | gemm64_f64 | 2.59e-13 | 4.175 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.590 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.30 | gemm64_f64 | 2.59e-13 | 4.179 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.901 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.683 | 1.32 | gemm64_f64 | 2.59e-13 | 4.173 | 7.17 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.520 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.687 | 1.30 | gemm64_f64 | 2.59e-13 | 4.173 | 7.07 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.731 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.684 | 1.31 | gemm64_f64 | 2.59e-13 | 4.174 | 7.12 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.726 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.682 | 1.31 | gemm64_f64 | 2.59e-13 | 4.178 | 7.11 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.495 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.688 | 1.30 | gemm64_f64 | 2.59e-13 | 4.180 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.631 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.683 | 1.31 | gemm64_f64 | 2.59e-13 | 4.176 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.694 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.683 | 1.31 | gemm64_f64 | 2.59e-13 | 4.180 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.564 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.684 | 1.30 | gemm64_f64 | 2.59e-13 | 4.175 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.632 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.682 | 1.31 | gemm64_f64 | 2.59e-13 | 4.177 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.749 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.679 | 1.31 | gemm64_f64 | 2.59e-13 | 4.182 | 7.11 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.558 | — | gemm64_f32; converted through the host | 1.62e-4 | 22.680 | 1.30 | gemm64_f64 | 2.59e-13 | 4.181 | 7.07 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.457 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.686 | 1.30 | gemm64_f64 | 2.59e-13 | 4.176 | 7.05 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.538 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.684 | 1.30 | gemm64_f64 | 2.59e-13 | 4.187 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.670 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.31 | gemm64_f64 | 2.59e-13 | 4.184 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.546 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.694 | 1.30 | gemm64_f64 | 2.59e-13 | 4.187 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.493 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.686 | 1.30 | gemm64_f64 | 2.59e-13 | 4.187 | 7.04 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.892 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.686 | 1.32 | gemm64_f64 | 2.59e-13 | 4.188 | 7.14 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.856 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.683 | 1.32 | gemm64_f64 | 2.59e-13 | 4.187 | 7.13 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.344 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.697 | 1.29 | gemm64_f64 | 2.59e-13 | 4.175 | 7.03 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.758 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.686 | 1.31 | gemm64_f64 | 2.59e-13 | 4.188 | 7.11 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.635 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.683 | 1.31 | gemm64_f64 | 2.59e-13 | 4.188 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.528 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.30 | gemm64_f64 | 2.59e-13 | 4.186 | 7.05 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.507 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.683 | 1.30 | gemm64_f64 | 2.59e-13 | 4.188 | 7.05 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.559 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.30 | gemm64_f64 | 2.59e-13 | 4.189 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.867 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.682 | 1.32 | gemm64_f64 | 2.59e-13 | 4.184 | 7.14 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.525 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.686 | 1.30 | gemm64_f64 | 2.59e-13 | 4.182 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.698 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.31 | gemm64_f64 | 2.59e-13 | 4.189 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.689 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.31 | gemm64_f64 | 2.59e-13 | 4.182 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.794 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.31 | gemm64_f64 | 2.59e-13 | 4.187 | 7.12 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.893 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.687 | 1.32 | gemm64_f64 | 2.59e-13 | 4.193 | 7.13 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.693 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.688 | 1.31 | gemm64_f64 | 2.59e-13 | 4.185 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.709 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.686 | 1.31 | gemm64_f64 | 2.59e-13 | 4.188 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.672 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.31 | gemm64_f64 | 2.59e-13 | 4.188 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.772 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.682 | 1.31 | gemm64_f64 | 2.59e-13 | 4.191 | 7.10 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.683 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.31 | gemm64_f64 | 2.59e-13 | 4.188 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.538 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.691 | 1.30 | gemm64_f64 | 2.59e-13 | 4.190 | 7.05 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.657 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.31 | gemm64_f64 | 2.59e-13 | 4.187 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.708 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.688 | 1.31 | gemm64_f64 | 2.59e-13 | 4.190 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.577 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.693 | 1.30 | gemm64_f64 | 2.59e-13 | 4.188 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.620 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.31 | gemm64_f64 | 2.59e-13 | 4.190 | 7.07 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.446 | — | gemm64_f32; converted through the host | 1.62e-4 | 22.690 | 1.30 | gemm64_f64 | 2.59e-13 | 4.182 | 7.04 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.624 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.690 | 1.31 | gemm64_f64 | 2.59e-13 | 4.191 | 7.07 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.706 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.699 | 1.31 | gemm64_f64 | 2.59e-13 | 4.187 | 7.09 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 30.220 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.695 | 1.33 | gemm64_f64 | 2.59e-13 | 4.184 | 7.22 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.876 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.689 | 1.32 | gemm64_f64 | 2.59e-13 | 4.188 | 7.13 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.645 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.691 | 1.31 | gemm64_f64 | 2.59e-13 | 4.190 | 7.08 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.587 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.685 | 1.30 | gemm64_f64 | 2.59e-13 | 4.189 | 7.06 | f64x | 0e+0 |
| cuda | matmul | float64 | 2048x2048x2048 | 29.589 | — | gemm128_f32; converted through the host | 1.62e-4 | 22.687 | 1.30 | gemm64_f64 | 2.59e-13 | 4.191 | 7.06 | f64x | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.012 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.74 | gemm128_u16 | 0e+0 | 0.180 | 94.56 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.436 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 13.04 | gemm128_u16 | 0e+0 | 0.169 | 103.23 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.170 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.86 | gemm128_u16 | 0e+0 | 0.170 | 101.30 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 18.042 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 13.51 | gemm128_u16 | 0e+0 | 0.169 | 106.82 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.243 | — | gemm128_f32; converted through the host | 0e+0 | 1.334 | 12.93 | gemm128_u16 | 0e+0 | 0.168 | 102.45 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.648 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.47 | gemm128_u16 | 0e+0 | 0.169 | 98.74 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.728 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.53 | gemm128_u16 | 0e+0 | 0.169 | 99.28 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.839 | — | gemm128_f32; converted through the host | 0e+0 | 1.334 | 12.62 | gemm128_u16 | 0e+0 | 0.169 | 99.70 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.599 | — | gemm128_f32; converted through the host | 0e+0 | 1.344 | 12.35 | gemm128_u16 | 0e+0 | 0.169 | 98.39 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.724 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.52 | gemm128_u16 | 0e+0 | 0.170 | 98.38 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.141 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.83 | gemm128_u16 | 0e+0 | 0.170 | 100.89 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.758 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.56 | gemm128_u16 | 0e+0 | 0.168 | 99.51 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.755 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.55 | gemm128_u16 | 0e+0 | 0.168 | 99.79 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.183 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.87 | gemm128_u16 | 0e+0 | 0.168 | 102.22 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.772 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.55 | gemm128_u16 | 0e+0 | 0.169 | 99.07 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.668 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.49 | gemm128_u16 | 0e+0 | 0.173 | 96.51 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.297 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.21 | gemm128_u16 | 0e+0 | 0.170 | 96.09 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.374 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.26 | gemm128_u16 | 0e+0 | 0.170 | 96.15 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.519 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.38 | gemm128_u16 | 0e+0 | 0.170 | 97.23 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.561 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.40 | gemm128_u16 | 0e+0 | 0.170 | 97.42 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.647 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.46 | gemm128_u16 | 0e+0 | 0.170 | 97.98 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.544 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.39 | gemm128_u16 | 0e+0 | 0.170 | 97.09 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.627 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.46 | gemm128_u16 | 0e+0 | 0.169 | 98.56 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.243 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.08 | gemm128_u16 | 0e+0 | 0.169 | 96.17 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.251 | — | gemm128_f32; converted through the host | 0e+0 | 1.343 | 12.10 | gemm128_u16 | 0e+0 | 0.170 | 95.60 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.246 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.16 | gemm128_u16 | 0e+0 | 0.168 | 96.76 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.224 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.15 | gemm128_u16 | 0e+0 | 0.171 | 94.66 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.239 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.16 | gemm128_u16 | 0e+0 | 0.170 | 95.64 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.751 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.54 | gemm128_u16 | 0e+0 | 0.169 | 99.12 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.205 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.13 | gemm128_u16 | 0e+0 | 0.169 | 95.94 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.797 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.48 | gemm128_u16 | 0e+0 | 0.170 | 98.80 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 17.066 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.79 | gemm128_u16 | 0e+0 | 0.169 | 100.74 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.499 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.26 | gemm128_u16 | 0e+0 | 0.170 | 97.28 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.242 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.16 | gemm128_u16 | 0e+0 | 0.173 | 93.88 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.231 | — | gemm64_f32; converted through the host | 0e+0 | 1.335 | 12.16 | gemm128_u16 | 0e+0 | 0.169 | 96.21 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.541 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 12.38 | gemm128_u16 | 0e+0 | 0.169 | 98.05 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.207 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.13 | gemm128_u16 | 0e+0 | 0.169 | 95.96 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.225 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.16 | gemm128_u16 | 0e+0 | 0.169 | 96.01 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.285 | — | gemm64_f32; converted through the host | 0e+0 | 1.335 | 12.20 | gemm128_u16 | 0e+0 | 0.170 | 95.79 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.232 | — | gemm128_f32; converted through the host | 0e+0 | 1.341 | 12.10 | gemm128_u16 | 0e+0 | 0.170 | 95.48 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.340 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.15 | gemm128_u16 | 0e+0 | 0.169 | 96.92 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.272 | — | gemm128_f32; converted through the host | 0e+0 | 1.334 | 12.20 | gemm128_u16 | 0e+0 | 0.169 | 96.40 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.385 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.19 | gemm128_u16 | 0e+0 | 0.169 | 97.01 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.171 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 12.10 | gemm128_u16 | 0e+0 | 0.171 | 94.68 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.573 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.32 | gemm128_u16 | 0e+0 | 0.168 | 98.47 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.313 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.13 | gemm128_u16 | 0e+0 | 0.170 | 96.19 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.245 | — | gemm128_f32; converted through the host | 0e+0 | 1.342 | 12.10 | gemm128_u16 | 0e+0 | 0.169 | 96.07 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.424 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.20 | gemm128_u16 | 0e+0 | 0.169 | 96.95 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.286 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.11 | gemm128_u16 | 0e+0 | 0.169 | 96.48 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.230 | — | gemm128_f32; converted through the host | 0e+0 | 1.344 | 12.07 | gemm128_u16 | 0e+0 | 0.170 | 95.53 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.194 | — | gemm128_f32; converted through the host | 0e+0 | 1.335 | 12.13 | gemm128_u16 | 0e+0 | 0.171 | 94.92 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.246 | — | gemm128_f32; converted through the host | 0e+0 | 1.344 | 12.08 | gemm128_u16 | 0e+0 | 0.169 | 96.42 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.234 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.07 | gemm128_u16 | 0e+0 | 0.169 | 96.34 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.223 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.06 | gemm128_u16 | 0e+0 | 0.168 | 96.45 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.263 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 12.16 | gemm128_u16 | 0e+0 | 0.169 | 96.29 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.480 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 12.33 | gemm128_u16 | 0e+0 | 0.171 | 96.43 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.212 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.04 | gemm128_u16 | 0e+0 | 0.169 | 95.99 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.348 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.15 | gemm128_u16 | 0e+0 | 0.169 | 96.79 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.290 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.11 | gemm128_u16 | 0e+0 | 0.183 | 89.16 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.745 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 12.53 | gemm128_u16 | 0e+0 | 0.169 | 98.96 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.397 | — | gemm128_f32; converted through the host | 0e+0 | 1.347 | 12.18 | gemm128_u16 | 0e+0 | 0.169 | 96.80 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.407 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.28 | gemm128_u16 | 0e+0 | 0.168 | 97.54 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.370 | — | gemm128_f32; converted through the host | 0e+0 | 1.347 | 12.16 | gemm128_u16 | 0e+0 | 0.170 | 96.58 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.259 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.09 | gemm128_u16 | 0e+0 | 0.168 | 96.55 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.224 | — | gemm128_f32; converted through the host | 0e+0 | 1.338 | 12.13 | gemm128_u16 | 0e+0 | 0.170 | 95.66 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.174 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.02 | gemm128_u16 | 0e+0 | 0.173 | 93.60 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.979 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.62 | gemm128_u16 | 0e+0 | 0.173 | 98.09 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.819 | — | gemm128_f32; converted through the host | 0e+0 | 1.342 | 12.54 | gemm128_u16 | 0e+0 | 0.169 | 99.76 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.559 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.39 | gemm128_u16 | 0e+0 | 0.169 | 97.98 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.764 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.47 | gemm128_u16 | 0e+0 | 0.170 | 98.61 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.419 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.21 | gemm128_u16 | 0e+0 | 0.169 | 97.15 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.769 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.46 | gemm128_u16 | 0e+0 | 0.170 | 98.70 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.765 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.47 | gemm128_u16 | 0e+0 | 0.170 | 98.62 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.826 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.59 | gemm128_u16 | 0e+0 | 0.168 | 100.21 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.489 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.26 | gemm128_u16 | 0e+0 | 0.169 | 97.74 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.209 | — | gemm128_f32; converted through the host | 0e+0 | 1.343 | 12.07 | gemm128_u16 | 0e+0 | 0.170 | 95.57 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.248 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.08 | gemm128_u16 | 0e+0 | 0.169 | 96.25 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.353 | — | gemm128_f32; converted through the host | 0e+0 | 1.344 | 12.17 | gemm128_u16 | 0e+0 | 0.169 | 96.82 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.256 | — | gemm128_f32; converted through the host | 0e+0 | 1.347 | 12.07 | gemm128_u16 | 0e+0 | 0.170 | 95.85 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.390 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.18 | gemm128_u16 | 0e+0 | 0.169 | 97.04 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.241 | — | gemm128_f32; converted through the host | 0e+0 | 1.336 | 12.15 | gemm128_u16 | 0e+0 | 0.169 | 95.93 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.249 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.08 | gemm128_u16 | 0e+0 | 0.169 | 96.38 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.210 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.04 | gemm128_u16 | 0e+0 | 0.169 | 96.20 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.853 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.52 | gemm128_u16 | 0e+0 | 0.169 | 100.02 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.620 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.34 | gemm128_u16 | 0e+0 | 0.170 | 97.93 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.451 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.23 | gemm128_u16 | 0e+0 | 0.169 | 97.52 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.677 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.40 | gemm128_u16 | 0e+0 | 0.170 | 98.33 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.776 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.47 | gemm128_u16 | 0e+0 | 0.168 | 99.86 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.628 | — | gemm128_f32; converted through the host | 0e+0 | 1.337 | 12.44 | gemm128_u16 | 0e+0 | 0.168 | 99.03 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.353 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.15 | gemm128_u16 | 0e+0 | 0.169 | 96.88 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.477 | — | gemm128_f32; converted through the host | 0e+0 | 1.338 | 12.31 | gemm128_u16 | 0e+0 | 0.169 | 97.55 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.534 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.29 | gemm128_u16 | 0e+0 | 0.171 | 96.63 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.580 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.32 | gemm128_u16 | 0e+0 | 0.170 | 97.36 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.314 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.12 | gemm128_u16 | 0e+0 | 0.169 | 96.71 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.230 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.06 | gemm128_u16 | 0e+0 | 0.169 | 95.92 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.248 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.07 | gemm128_u16 | 0e+0 | 0.169 | 96.09 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.216 | — | gemm128_f32; converted through the host | 0e+0 | 1.346 | 12.05 | gemm128_u16 | 0e+0 | 0.171 | 94.83 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.330 | — | gemm128_f32; converted through the host | 0e+0 | 1.345 | 12.14 | gemm128_u16 | 0e+0 | 0.169 | 96.74 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.377 | — | gemm128_f32; converted through the host | 0e+0 | 1.344 | 12.19 | gemm128_u16 | 0e+0 | 0.169 | 96.96 | u8mma | 0e+0 |
| cuda | matmul | uint16 | 2048x2048x2048 | 16.339 | — | gemm128_f32; converted through the host | 0e+0 | 1.344 | 12.16 | gemm128_u16 | 0e+0 | 0.171 | 95.38 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.408 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.35 | gemm128_u32 | 0e+0 | 0.211 | 106.25 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.690 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 20.49 | gemm128_u32 | 0e+0 | 0.208 | 109.14 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.516 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.44 | gemm128_u32 | 0e+0 | 0.210 | 107.22 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.662 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 20.46 | gemm128_u32 | 0e+0 | 0.208 | 109.00 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.523 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 20.44 | gemm128_u32 | 0e+0 | 0.209 | 107.82 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.866 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.75 | gemm128_u32 | 0e+0 | 0.208 | 105.17 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.879 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.88 | gemm128_u32 | 0e+0 | 0.210 | 104.23 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.971 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.85 | gemm128_u32 | 0e+0 | 0.209 | 105.22 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.002 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.97 | gemm128_u32 | 0e+0 | 0.210 | 105.02 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.838 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.73 | gemm128_u32 | 0e+0 | 0.208 | 105.09 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.954 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.94 | gemm128_u32 | 0e+0 | 0.211 | 104.09 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.873 | — | gemm128_f32; converted through the host | 0e+0 | 1.108 | 19.75 | gemm128_u32 | 0e+0 | 0.209 | 104.81 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.033 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 20.01 | gemm128_u32 | 0e+0 | 0.209 | 105.47 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.276 | — | gemm128_f32; converted through the host | 0e+0 | 1.108 | 20.10 | gemm128_u32 | 0e+0 | 0.209 | 106.69 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.898 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.89 | gemm128_u32 | 0e+0 | 0.210 | 104.33 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.865 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.76 | gemm128_u32 | 0e+0 | 0.209 | 104.72 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.167 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 20.12 | gemm128_u32 | 0e+0 | 0.212 | 104.36 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.317 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 20.17 | gemm128_u32 | 0e+0 | 0.209 | 106.93 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.867 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.86 | gemm128_u32 | 0e+0 | 0.210 | 104.23 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.913 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.80 | gemm128_u32 | 0e+0 | 0.208 | 105.45 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.903 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.88 | gemm128_u32 | 0e+0 | 0.210 | 104.35 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.282 | — | gemm128_f32; converted through the host | 0e+0 | 1.108 | 20.12 | gemm128_u32 | 0e+0 | 0.209 | 106.66 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.853 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.85 | gemm128_u32 | 0e+0 | 0.210 | 104.06 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.894 | — | gemm128_f32; converted through the host | 0e+0 | 1.106 | 20.69 | gemm128_u32 | 0e+0 | 0.225 | 101.66 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.683 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.59 | gemm128_u32 | 0e+0 | 0.210 | 108.07 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.983 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.85 | gemm128_u32 | 0e+0 | 0.209 | 105.23 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.966 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.96 | gemm128_u32 | 0e+0 | 0.210 | 104.65 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.006 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.88 | gemm128_u32 | 0e+0 | 0.210 | 105.04 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.911 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.88 | gemm128_u32 | 0e+0 | 0.210 | 104.39 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.821 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.71 | gemm128_u32 | 0e+0 | 0.209 | 104.51 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.887 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.88 | gemm128_u32 | 0e+0 | 0.209 | 104.77 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.881 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.77 | gemm128_u32 | 0e+0 | 0.208 | 105.25 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.923 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.92 | gemm128_u32 | 0e+0 | 0.206 | 106.52 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.837 | — | gemm128_f32; converted through the host | 0e+0 | 1.106 | 19.74 | gemm128_u32 | 0e+0 | 0.208 | 105.03 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.961 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.94 | gemm128_u32 | 0e+0 | 0.229 | 96.11 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.824 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.72 | gemm128_u32 | 0e+0 | 0.208 | 104.97 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.898 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.90 | gemm128_u32 | 0e+0 | 0.211 | 103.88 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.110 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.97 | gemm128_u32 | 0e+0 | 0.210 | 105.54 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.884 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.87 | gemm128_u32 | 0e+0 | 0.211 | 103.91 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.924 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.81 | gemm128_u32 | 0e+0 | 0.208 | 105.45 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.074 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 20.04 | gemm128_u32 | 0e+0 | 0.210 | 105.16 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.918 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.80 | gemm128_u32 | 0e+0 | 0.209 | 104.92 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.907 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.89 | gemm128_u32 | 0e+0 | 0.210 | 104.37 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.915 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.79 | gemm128_u32 | 0e+0 | 0.208 | 105.46 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.980 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.96 | gemm128_u32 | 0e+0 | 0.211 | 104.22 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.897 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.77 | gemm128_u32 | 0e+0 | 0.204 | 107.49 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.886 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.88 | gemm128_u32 | 0e+0 | 0.210 | 104.27 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.906 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.80 | gemm128_u32 | 0e+0 | 0.227 | 96.67 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.916 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.90 | gemm128_u32 | 0e+0 | 0.210 | 104.41 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.846 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.74 | gemm128_u32 | 0e+0 | 0.211 | 103.58 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.067 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 20.04 | gemm128_u32 | 0e+0 | 0.210 | 105.13 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.875 | — | gemm128_f32; converted through the host | 0e+0 | 1.108 | 19.75 | gemm128_u32 | 0e+0 | 0.209 | 104.91 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.844 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.84 | gemm128_u32 | 0e+0 | 0.210 | 104.07 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.938 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.82 | gemm128_u32 | 0e+0 | 0.211 | 103.82 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.135 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.11 | gemm128_u32 | 0e+0 | 0.210 | 105.46 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.984 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.86 | gemm128_u32 | 0e+0 | 0.208 | 105.79 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.899 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.89 | gemm128_u32 | 0e+0 | 0.209 | 104.58 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.820 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.71 | gemm128_u32 | 0e+0 | 0.226 | 96.42 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.918 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.91 | gemm128_u32 | 0e+0 | 0.212 | 103.43 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.905 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.79 | gemm128_u32 | 0e+0 | 0.228 | 95.95 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.876 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.85 | gemm128_u32 | 0e+0 | 0.210 | 104.22 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.878 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.77 | gemm128_u32 | 0e+0 | 0.212 | 103.25 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.859 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.85 | gemm128_u32 | 0e+0 | 0.227 | 96.42 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.970 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.85 | gemm128_u32 | 0e+0 | 0.209 | 104.97 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.928 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.92 | gemm128_u32 | 0e+0 | 0.211 | 103.97 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.897 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.78 | gemm128_u32 | 0e+0 | 0.208 | 105.32 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.973 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.95 | gemm128_u32 | 0e+0 | 0.210 | 104.68 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.927 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.81 | gemm128_u32 | 0e+0 | 0.208 | 105.47 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.831 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.83 | gemm128_u32 | 0e+0 | 0.213 | 102.49 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.856 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.74 | gemm128_u32 | 0e+0 | 0.208 | 105.13 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.812 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.81 | gemm128_u32 | 0e+0 | 0.211 | 103.42 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.863 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.75 | gemm128_u32 | 0e+0 | 0.209 | 104.71 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.061 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.03 | gemm128_u32 | 0e+0 | 0.210 | 105.10 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.974 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.85 | gemm128_u32 | 0e+0 | 0.208 | 105.65 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.951 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.94 | gemm128_u32 | 0e+0 | 0.210 | 104.58 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.882 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.77 | gemm128_u32 | 0e+0 | 0.208 | 105.25 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.943 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.93 | gemm128_u32 | 0e+0 | 0.210 | 104.54 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.058 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.93 | gemm128_u32 | 0e+0 | 0.199 | 111.01 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.968 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.93 | gemm128_u32 | 0e+0 | 0.210 | 104.61 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.150 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 20.01 | gemm128_u32 | 0e+0 | 0.209 | 106.08 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.965 | — | gemm128_f32; converted through the host | 0e+0 | 1.102 | 19.93 | gemm128_u32 | 0e+0 | 0.210 | 104.64 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.836 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.72 | gemm128_u32 | 0e+0 | 0.209 | 104.53 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.969 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.95 | gemm128_u32 | 0e+0 | 0.210 | 104.66 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.961 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.84 | gemm128_u32 | 0e+0 | 0.208 | 105.48 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.898 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.89 | gemm128_u32 | 0e+0 | 0.215 | 102.04 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.122 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.99 | gemm128_u32 | 0e+0 | 0.209 | 105.90 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.856 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.84 | gemm128_u32 | 0e+0 | 0.219 | 100.03 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.957 | — | gemm128_f32; converted through the host | 0e+0 | 1.108 | 19.83 | gemm128_u32 | 0e+0 | 0.208 | 105.61 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.236 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.19 | gemm128_u32 | 0e+0 | 0.228 | 97.35 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.821 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.70 | gemm128_u32 | 0e+0 | 0.208 | 105.11 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 22.259 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 20.21 | gemm128_u32 | 0e+0 | 0.200 | 111.46 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.970 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.84 | gemm128_u32 | 0e+0 | 0.209 | 105.17 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.983 | — | gemm64_f32; converted through the host | 0e+0 | 1.101 | 19.96 | gemm128_u32 | 0e+0 | 0.212 | 103.94 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.851 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.74 | gemm128_u32 | 0e+0 | 0.208 | 105.10 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.901 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.90 | gemm128_u32 | 0e+0 | 0.210 | 104.34 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.932 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.81 | gemm128_u32 | 0e+0 | 0.208 | 105.49 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.811 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.80 | gemm128_u32 | 0e+0 | 0.210 | 103.91 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.934 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.81 | gemm128_u32 | 0e+0 | 0.208 | 105.45 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.990 | — | gemm128_f32; converted through the host | 0e+0 | 1.101 | 19.98 | gemm128_u32 | 0e+0 | 0.210 | 104.76 | u8mma | 0e+0 |
| cuda | matmul | uint32 | 2048x2048x2048 | 21.873 | — | gemm128_f32; converted through the host | 0e+0 | 1.107 | 19.76 | gemm128_u32 | 0e+0 | 0.208 | 105.21 | u8mma | 0e+0 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.126 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.124 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.126 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.126 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.126 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.124 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.123 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cuda | matmul | bool | 2048x2048x2048 | — | — | — | — | — | — | — | — | 0.122 | — | u8mma | 1.27e+3 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.107 | — | to f32 and back | 7.96e-5 | 15.584 | 0.39 | Vector<Double> | 1.14e-13 | 3.309 | 1.85 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.197 | — | to f32 and back | 7.96e-5 | 16.203 | 0.38 | Vector<Double> | 1.14e-13 | 3.097 | 2.00 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.084 | — | to f32 and back | 7.96e-5 | 23.331 | 0.26 | Vector<Double> | 1.14e-13 | 3.268 | 1.86 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.134 | — | to f32 and back | 7.96e-5 | 19.379 | 0.32 | Vector<Double> | 1.14e-13 | 3.285 | 1.87 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.431 | — | to f32 and back | 7.96e-5 | 23.695 | 0.27 | Vector<Double> | 1.14e-13 | 3.287 | 1.96 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.276 | — | to f32 and back | 7.96e-5 | 10.445 | 0.60 | Vector<Double> | 1.14e-13 | 3.413 | 1.84 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.515 | — | to f32 and back | 7.96e-5 | 20.408 | 0.32 | Vector<Double> | 1.14e-13 | 3.560 | 1.83 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.001 | — | to f32 and back | 7.96e-5 | 18.842 | 0.32 | Vector<Double> | 1.14e-13 | 3.319 | 1.81 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.052 | — | to f32 and back | 7.96e-5 | 15.479 | 0.39 | Vector<Double> | 1.14e-13 | 3.203 | 1.89 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.233 | — | to f32 and back | 7.96e-5 | 16.204 | 0.38 | Vector<Double> | 1.14e-13 | 2.869 | 2.17 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.432 | — | to f32 and back | 7.96e-5 | 11.303 | 0.57 | Vector<Double> | 1.14e-13 | 3.222 | 2.00 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.209 | — | to f32 and back | 7.96e-5 | 23.972 | 0.26 | Vector<Double> | 1.14e-13 | 3.399 | 1.83 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.034 | — | to f32 and back | 7.96e-5 | 15.528 | 0.39 | Vector<Double> | 1.14e-13 | 3.163 | 1.91 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.525 | — | to f32 and back | 7.96e-5 | 19.896 | 0.33 | Vector<Double> | 1.14e-13 | 3.394 | 1.92 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.217 | — | to f32 and back | 7.96e-5 | 15.872 | 0.39 | Vector<Double> | 1.14e-13 | 3.237 | 1.92 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.105 | — | to f32 and back | 7.96e-5 | 17.926 | 0.34 | Vector<Double> | 1.14e-13 | 3.454 | 1.77 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.058 | — | to f32 and back | 7.96e-5 | 13.100 | 0.46 | Vector<Double> | 1.14e-13 | 2.801 | 2.16 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.477 | — | to f32 and back | 7.96e-5 | 15.457 | 0.42 | Vector<Double> | 1.14e-13 | 3.340 | 1.94 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.425 | — | to f32 and back | 7.96e-5 | 15.781 | 0.41 | Vector<Double> | 1.14e-13 | 3.380 | 1.90 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.521 | — | to f32 and back | 7.96e-5 | 15.119 | 0.43 | Vector<Double> | 1.14e-13 | 3.493 | 1.87 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.971 | — | to f32 and back | 7.96e-5 | 10.596 | 0.56 | Vector<Double> | 1.14e-13 | 3.162 | 1.89 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.381 | — | to f32 and back | 7.96e-5 | 15.821 | 0.40 | Vector<Double> | 1.14e-13 | 3.292 | 1.94 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.182 | — | to f32 and back | 7.96e-5 | 15.527 | 0.40 | Vector<Double> | 1.14e-13 | 3.534 | 1.75 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.001 | — | to f32 and back | 7.96e-5 | 16.134 | 0.37 | Vector<Double> | 1.14e-13 | 3.279 | 1.83 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.363 | — | to f32 and back | 7.96e-5 | 15.562 | 0.41 | Vector<Double> | 1.14e-13 | 3.484 | 1.83 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.647 | — | to f32 and back | 7.96e-5 | 17.350 | 0.38 | Vector<Double> | 1.14e-13 | 3.585 | 1.85 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.164 | — | to f32 and back | 7.96e-5 | 16.709 | 0.37 | Vector<Double> | 1.14e-13 | 3.351 | 1.84 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.105 | — | to f32 and back | 7.96e-5 | 23.384 | 0.26 | Vector<Double> | 1.14e-13 | 3.067 | 1.99 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.168 | — | to f32 and back | 7.96e-5 | 15.501 | 0.40 | Vector<Double> | 1.14e-13 | 3.384 | 1.82 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.362 | — | to f32 and back | 7.96e-5 | 15.949 | 0.40 | Vector<Double> | 1.14e-13 | 3.260 | 1.95 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.195 | — | to f32 and back | 7.96e-5 | 15.971 | 0.39 | Vector<Double> | 1.14e-13 | 3.088 | 2.01 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.029 | — | to f32 and back | 7.96e-5 | 14.556 | 0.41 | Vector<Double> | 1.14e-13 | 3.105 | 1.94 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.015 | — | to f32 and back | 7.96e-5 | 20.716 | 0.29 | Vector<Double> | 1.14e-13 | 3.311 | 1.82 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.967 | — | to f32 and back | 7.96e-5 | 10.381 | 0.57 | Vector<Double> | 1.14e-13 | 3.610 | 1.65 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.473 | — | to f32 and back | 7.96e-5 | 15.963 | 0.41 | Vector<Double> | 1.14e-13 | 3.373 | 1.92 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.002 | — | to f32 and back | 7.96e-5 | 16.211 | 0.37 | Vector<Double> | 1.14e-13 | 3.369 | 1.78 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.142 | — | to f32 and back | 7.96e-5 | 18.822 | 0.33 | Vector<Double> | 1.14e-13 | 3.109 | 1.98 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.130 | — | to f32 and back | 7.96e-5 | 17.968 | 0.34 | Vector<Double> | 1.14e-13 | 3.175 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.039 | — | to f32 and back | 7.96e-5 | 19.984 | 0.30 | Vector<Double> | 1.14e-13 | 3.362 | 1.80 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.188 | — | to f32 and back | 7.96e-5 | 20.748 | 0.30 | Vector<Double> | 1.14e-13 | 3.355 | 1.84 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.394 | — | to f32 and back | 7.96e-5 | 16.160 | 0.40 | Vector<Double> | 1.14e-13 | 2.945 | 2.17 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.107 | — | to f32 and back | 7.96e-5 | 24.475 | 0.25 | Vector<Double> | 1.14e-13 | 3.311 | 1.84 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.996 | — | to f32 and back | 7.96e-5 | 17.127 | 0.35 | Vector<Double> | 1.14e-13 | 3.462 | 1.73 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.154 | — | to f32 and back | 7.96e-5 | 15.933 | 0.39 | Vector<Double> | 1.14e-13 | 3.556 | 1.73 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.376 | — | to f32 and back | 7.96e-5 | 15.786 | 0.40 | Vector<Double> | 1.14e-13 | 3.327 | 1.92 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.223 | — | to f32 and back | 7.96e-5 | 22.995 | 0.27 | Vector<Double> | 1.14e-13 | 3.401 | 1.83 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.429 | — | to f32 and back | 7.96e-5 | 17.277 | 0.37 | Vector<Double> | 1.14e-13 | 3.542 | 1.82 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.049 | — | to f32 and back | 7.96e-5 | 16.177 | 0.37 | Vector<Double> | 1.14e-13 | 3.344 | 1.81 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.019 | — | to f32 and back | 7.96e-5 | 11.857 | 0.51 | Vector<Double> | 1.14e-13 | 3.207 | 1.88 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.277 | — | to f32 and back | 7.96e-5 | 16.224 | 0.39 | Vector<Double> | 1.14e-13 | 3.229 | 1.94 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.420 | — | to f32 and back | 7.96e-5 | 10.385 | 0.62 | Vector<Double> | 1.14e-13 | 3.284 | 1.95 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.058 | — | to f32 and back | 7.96e-5 | 19.240 | 0.31 | Vector<Double> | 1.14e-13 | 3.372 | 1.80 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.226 | — | to f32 and back | 7.96e-5 | 15.796 | 0.39 | Vector<Double> | 1.14e-13 | 3.167 | 1.97 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.092 | — | to f32 and back | 7.96e-5 | 20.885 | 0.29 | Vector<Double> | 1.14e-13 | 3.546 | 1.72 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.270 | — | to f32 and back | 7.96e-5 | 19.232 | 0.33 | Vector<Double> | 1.14e-13 | 3.184 | 1.97 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.039 | — | to f32 and back | 7.96e-5 | 16.114 | 0.37 | Vector<Double> | 1.14e-13 | 3.316 | 1.82 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.250 | — | to f32 and back | 7.96e-5 | 20.894 | 0.30 | Vector<Double> | 1.14e-13 | 3.319 | 1.88 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.166 | — | to f32 and back | 7.96e-5 | 15.005 | 0.41 | Vector<Double> | 1.14e-13 | 3.192 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.133 | — | to f32 and back | 7.96e-5 | 16.097 | 0.38 | Vector<Double> | 1.14e-13 | 3.308 | 1.85 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.390 | — | to f32 and back | 7.96e-5 | 15.847 | 0.40 | Vector<Double> | 1.14e-13 | 3.161 | 2.02 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.119 | — | to f32 and back | 7.96e-5 | 21.896 | 0.28 | Vector<Double> | 1.14e-13 | 3.577 | 1.71 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.177 | — | to f32 and back | 7.96e-5 | 22.725 | 0.27 | Vector<Double> | 1.14e-13 | 3.299 | 1.87 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.978 | — | to f32 and back | 7.96e-5 | 23.134 | 0.26 | Vector<Double> | 1.14e-13 | 3.321 | 1.80 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.147 | — | to f32 and back | 7.96e-5 | 16.320 | 0.38 | Vector<Double> | 1.14e-13 | 3.193 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.909 | — | to f32 and back | 7.96e-5 | 17.835 | 0.33 | Vector<Double> | 1.14e-13 | 3.404 | 1.74 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.190 | — | to f32 and back | 7.96e-5 | 15.928 | 0.39 | Vector<Double> | 1.14e-13 | 3.241 | 1.91 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.146 | — | to f32 and back | 7.96e-5 | 19.031 | 0.32 | Vector<Double> | 1.14e-13 | 3.408 | 1.80 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.199 | — | to f32 and back | 7.96e-5 | 23.343 | 0.27 | Vector<Double> | 1.14e-13 | 3.357 | 1.85 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.168 | — | to f32 and back | 7.96e-5 | 15.729 | 0.39 | Vector<Double> | 1.14e-13 | 3.158 | 1.95 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.012 | — | to f32 and back | 7.96e-5 | 19.115 | 0.31 | Vector<Double> | 1.14e-13 | 3.326 | 1.81 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.389 | — | to f32 and back | 7.96e-5 | 19.911 | 0.32 | Vector<Double> | 1.14e-13 | 3.536 | 1.81 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.055 | — | to f32 and back | 7.96e-5 | 18.470 | 0.33 | Vector<Double> | 1.14e-13 | 3.344 | 1.81 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.573 | — | to f32 and back | 7.96e-5 | 16.570 | 0.40 | Vector<Double> | 1.14e-13 | 3.096 | 2.12 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.217 | — | to f32 and back | 7.96e-5 | 15.003 | 0.41 | Vector<Double> | 1.14e-13 | 3.014 | 2.06 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.400 | — | to f32 and back | 7.96e-5 | 22.758 | 0.28 | Vector<Double> | 1.14e-13 | 3.295 | 1.94 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.081 | — | to f32 and back | 7.96e-5 | 11.481 | 0.53 | Vector<Double> | 1.14e-13 | 3.614 | 1.68 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.233 | — | to f32 and back | 7.96e-5 | 15.843 | 0.39 | Vector<Double> | 1.14e-13 | 3.148 | 1.98 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.276 | — | to f32 and back | 7.96e-5 | 16.942 | 0.37 | Vector<Double> | 1.14e-13 | 3.126 | 2.01 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.317 | — | to f32 and back | 7.96e-5 | 16.668 | 0.38 | Vector<Double> | 1.14e-13 | 3.140 | 2.01 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.490 | — | to f32 and back | 7.96e-5 | 23.002 | 0.28 | Vector<Double> | 1.14e-13 | 3.621 | 1.79 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.499 | — | to f32 and back | 7.96e-5 | 19.609 | 0.33 | Vector<Double> | 1.14e-13 | 3.372 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.189 | — | to f32 and back | 7.96e-5 | 15.358 | 0.40 | Vector<Double> | 1.14e-13 | 3.369 | 1.84 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.983 | — | to f32 and back | 7.96e-5 | 15.647 | 0.38 | Vector<Double> | 1.14e-13 | 3.419 | 1.75 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.092 | — | to f32 and back | 7.96e-5 | 15.140 | 0.40 | Vector<Double> | 1.14e-13 | 2.835 | 2.15 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.590 | — | to f32 and back | 7.96e-5 | 14.035 | 0.47 | Vector<Double> | 1.14e-13 | 3.412 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.380 | — | to f32 and back | 7.96e-5 | 20.081 | 0.32 | Vector<Double> | 1.14e-13 | 3.516 | 1.81 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.417 | — | to f32 and back | 7.96e-5 | 15.996 | 0.40 | Vector<Double> | 1.14e-13 | 3.147 | 2.04 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.347 | — | to f32 and back | 7.96e-5 | 17.011 | 0.37 | Vector<Double> | 1.14e-13 | 3.163 | 2.01 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.347 | — | to f32 and back | 7.96e-5 | 19.868 | 0.32 | Vector<Double> | 1.14e-13 | 3.367 | 1.89 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.224 | — | to f32 and back | 7.96e-5 | 16.283 | 0.38 | Vector<Double> | 1.14e-13 | 3.132 | 1.99 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 5.997 | — | to f32 and back | 7.96e-5 | 15.958 | 0.38 | Vector<Double> | 1.14e-13 | 3.604 | 1.66 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.396 | — | to f32 and back | 7.96e-5 | 15.668 | 0.41 | Vector<Double> | 1.14e-13 | 3.271 | 1.96 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.349 | — | to f32 and back | 7.96e-5 | 15.419 | 0.41 | Vector<Double> | 1.14e-13 | 3.318 | 1.91 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.096 | — | to f32 and back | 7.96e-5 | 15.645 | 0.39 | Vector<Double> | 1.14e-13 | 3.274 | 1.86 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.572 | — | to f32 and back | 7.96e-5 | 20.868 | 0.31 | Vector<Double> | 1.14e-13 | 3.454 | 1.90 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.465 | — | to f32 and back | 7.96e-5 | 20.654 | 0.31 | Vector<Double> | 1.14e-13 | 3.355 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.206 | — | to f32 and back | 7.96e-5 | 21.708 | 0.29 | Vector<Double> | 1.14e-13 | 3.309 | 1.88 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.402 | — | to f32 and back | 7.96e-5 | 18.806 | 0.34 | Vector<Double> | 1.14e-13 | 3.316 | 1.93 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.087 | — | to f32 and back | 7.96e-5 | 15.299 | 0.40 | Vector<Double> | 1.14e-13 | 3.510 | 1.73 | f64 | 1.14e-13 |
| cpu | matmul | float64 | 1024x1024x1024 | 6.223 | — | to f32 and back | 7.96e-5 | 15.818 | 0.39 | Vector<Double> | 1.14e-13 | 3.368 | 1.85 | f64 | 1.14e-13 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.409 | — | to f32 and back | 0e+0 | 2.297 | 2.35 | Vector<UInt16> | 0e+0 | 0.354 | 15.28 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.035 | — | to f32 and back | 0e+0 | 1.898 | 2.65 | Vector<UInt16> | 0e+0 | 0.350 | 14.38 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.882 | — | to f32 and back | 0e+0 | 2.936 | 1.66 | Vector<UInt16> | 0e+0 | 0.349 | 13.98 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.631 | — | to f32 and back | 0e+0 | 2.005 | 2.31 | Vector<UInt16> | 0e+0 | 0.349 | 13.28 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.474 | — | to f32 and back | 0e+0 | 1.837 | 2.44 | Vector<UInt16> | 0e+0 | 0.350 | 12.78 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.971 | — | to f32 and back | 0e+0 | 1.868 | 2.66 | Vector<UInt16> | 0e+0 | 0.350 | 14.22 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.979 | — | to f32 and back | 0e+0 | 1.663 | 2.99 | Vector<UInt16> | 0e+0 | 0.349 | 14.26 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.759 | — | to f32 and back | 0e+0 | 1.848 | 2.58 | Vector<UInt16> | 0e+0 | 0.373 | 12.75 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.672 | — | to f32 and back | 0e+0 | 3.296 | 1.72 | Vector<UInt16> | 0e+0 | 0.531 | 10.67 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.986 | — | to f32 and back | 0e+0 | 2.211 | 2.26 | Vector<UInt16> | 0e+0 | 0.525 | 9.50 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.927 | — | to f32 and back | 0e+0 | 2.039 | 2.42 | Vector<UInt16> | 0e+0 | 0.478 | 10.32 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.113 | — | to f32 and back | 0e+0 | 2.224 | 2.30 | Vector<UInt16> | 0e+0 | 0.399 | 12.80 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.817 | — | to f32 and back | 0e+0 | 2.193 | 2.20 | Vector<UInt16> | 0e+0 | 0.526 | 9.16 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.892 | — | to f32 and back | 0e+0 | 2.054 | 2.38 | Vector<UInt16> | 0e+0 | 0.499 | 9.81 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.277 | — | to f32 and back | 0e+0 | 3.066 | 1.72 | Vector<UInt16> | 0e+0 | 0.403 | 13.10 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.365 | — | to f32 and back | 0e+0 | 2.493 | 2.15 | Vector<UInt16> | 0e+0 | 0.496 | 10.82 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.359 | — | to f32 and back | 0e+0 | 2.450 | 2.19 | Vector<UInt16> | 0e+0 | 0.522 | 10.26 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.174 | — | to f32 and back | 0e+0 | 2.243 | 2.31 | Vector<UInt16> | 0e+0 | 0.488 | 10.60 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.050 | — | to f32 and back | 0e+0 | 2.114 | 2.39 | Vector<UInt16> | 0e+0 | 0.458 | 11.03 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.112 | — | to f32 and back | 0e+0 | 2.090 | 2.45 | Vector<UInt16> | 0e+0 | 0.529 | 9.67 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.887 | — | to f32 and back | 0e+0 | 2.104 | 2.32 | Vector<UInt16> | 0e+0 | 0.527 | 9.27 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.176 | — | to f32 and back | 0e+0 | 2.213 | 2.34 | Vector<UInt16> | 0e+0 | 0.490 | 10.56 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.967 | — | to f32 and back | 0e+0 | 2.027 | 2.45 | Vector<UInt16> | 0e+0 | 0.350 | 14.20 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.804 | — | to f32 and back | 0e+0 | 1.967 | 2.44 | Vector<UInt16> | 0e+0 | 0.348 | 13.82 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.865 | — | to f32 and back | 0e+0 | 1.860 | 2.62 | Vector<UInt16> | 0e+0 | 0.351 | 13.86 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.048 | — | to f32 and back | 0e+0 | 2.017 | 2.50 | Vector<UInt16> | 0e+0 | 0.350 | 14.43 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.700 | — | to f32 and back | 0e+0 | 1.813 | 2.59 | Vector<UInt16> | 0e+0 | 0.349 | 13.47 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.260 | — | to f32 and back | 0e+0 | 2.238 | 2.35 | Vector<UInt16> | 0e+0 | 0.454 | 11.59 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.958 | — | to f32 and back | 0e+0 | 2.371 | 2.09 | Vector<UInt16> | 0e+0 | 0.460 | 10.78 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.146 | — | to f32 and back | 0e+0 | 2.246 | 2.29 | Vector<UInt16> | 0e+0 | 0.527 | 9.76 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.138 | — | to f32 and back | 0e+0 | 1.954 | 2.63 | Vector<UInt16> | 0e+0 | 0.347 | 14.79 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.519 | — | to f32 and back | 0e+0 | 2.176 | 2.08 | Vector<UInt16> | 0e+0 | 0.349 | 12.94 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.776 | — | to f32 and back | 0e+0 | 1.557 | 3.07 | Vector<UInt16> | 0e+0 | 0.350 | 13.65 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.084 | — | to f32 and back | 0e+0 | 2.296 | 2.21 | Vector<UInt16> | 0e+0 | 0.522 | 9.75 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.167 | — | to f32 and back | 0e+0 | 2.968 | 1.74 | Vector<UInt16> | 0e+0 | 0.460 | 11.23 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.317 | — | to f32 and back | 0e+0 | 2.133 | 2.49 | Vector<UInt16> | 0e+0 | 0.506 | 10.51 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.873 | — | to f32 and back | 0e+0 | 2.033 | 2.40 | Vector<UInt16> | 0e+0 | 0.526 | 9.27 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.997 | — | to f32 and back | 0e+0 | 3.103 | 1.61 | Vector<UInt16> | 0e+0 | 0.450 | 11.10 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.386 | — | to f32 and back | 0e+0 | 2.221 | 2.43 | Vector<UInt16> | 0e+0 | 0.428 | 12.59 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.216 | — | to f32 and back | 0e+0 | 2.217 | 2.35 | Vector<UInt16> | 0e+0 | 0.422 | 12.37 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.343 | — | to f32 and back | 0e+0 | 1.747 | 3.06 | Vector<UInt16> | 0e+0 | 0.467 | 11.45 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.587 | — | to f32 and back | 0e+0 | 2.439 | 2.29 | Vector<UInt16> | 0e+0 | 0.529 | 10.56 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.226 | — | to f32 and back | 0e+0 | 2.381 | 2.19 | Vector<UInt16> | 0e+0 | 0.401 | 13.02 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.282 | — | to f32 and back | 0e+0 | 2.336 | 2.26 | Vector<UInt16> | 0e+0 | 0.490 | 10.79 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.169 | — | to f32 and back | 0e+0 | 1.354 | 3.82 | Vector<UInt16> | 0e+0 | 0.351 | 14.74 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.675 | — | to f32 and back | 0e+0 | 2.190 | 2.13 | Vector<UInt16> | 0e+0 | 0.454 | 10.30 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.137 | — | to f32 and back | 0e+0 | 2.242 | 2.29 | Vector<UInt16> | 0e+0 | 0.548 | 9.38 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.216 | — | to f32 and back | 0e+0 | 3.089 | 1.69 | Vector<UInt16> | 0e+0 | 0.492 | 10.59 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.569 | — | to f32 and back | 0e+0 | 3.168 | 1.76 | Vector<UInt16> | 0e+0 | 0.407 | 13.70 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.833 | — | to f32 and back | 0e+0 | 2.491 | 1.94 | Vector<UInt16> | 0e+0 | 0.526 | 9.18 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.024 | — | to f32 and back | 0e+0 | 1.474 | 3.41 | Vector<UInt16> | 0e+0 | 0.361 | 13.90 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.348 | — | to f32 and back | 0e+0 | 2.218 | 2.41 | Vector<UInt16> | 0e+0 | 0.491 | 10.89 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.962 | — | to f32 and back | 0e+0 | 2.031 | 2.44 | Vector<UInt16> | 0e+0 | 0.528 | 9.40 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.080 | — | to f32 and back | 0e+0 | 1.516 | 3.35 | Vector<UInt16> | 0e+0 | 0.353 | 14.37 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.510 | — | to f32 and back | 0e+0 | 2.077 | 2.17 | Vector<UInt16> | 0e+0 | 0.495 | 9.11 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.015 | — | to f32 and back | 0e+0 | 3.106 | 1.61 | Vector<UInt16> | 0e+0 | 0.487 | 10.30 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.091 | — | to f32 and back | 0e+0 | 3.061 | 1.66 | Vector<UInt16> | 0e+0 | 0.528 | 9.65 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.933 | — | to f32 and back | 0e+0 | 2.172 | 2.27 | Vector<UInt16> | 0e+0 | 0.474 | 10.40 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.903 | — | to f32 and back | 0e+0 | 2.083 | 2.35 | Vector<UInt16> | 0e+0 | 0.492 | 9.96 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.010 | — | to f32 and back | 0e+0 | 2.077 | 2.41 | Vector<UInt16> | 0e+0 | 0.500 | 10.02 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.069 | — | to f32 and back | 0e+0 | 1.971 | 2.57 | Vector<UInt16> | 0e+0 | 0.351 | 14.44 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.688 | — | to f32 and back | 0e+0 | 2.251 | 2.08 | Vector<UInt16> | 0e+0 | 0.452 | 10.37 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.207 | — | to f32 and back | 0e+0 | 3.106 | 1.68 | Vector<UInt16> | 0e+0 | 0.496 | 10.50 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.142 | — | to f32 and back | 0e+0 | 2.444 | 2.10 | Vector<UInt16> | 0e+0 | 0.527 | 9.76 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.987 | — | to f32 and back | 0e+0 | 2.223 | 2.24 | Vector<UInt16> | 0e+0 | 0.500 | 9.97 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.915 | — | to f32 and back | 0e+0 | 2.867 | 1.71 | Vector<UInt16> | 0e+0 | 0.349 | 14.10 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.562 | — | to f32 and back | 0e+0 | 2.541 | 2.19 | Vector<UInt16> | 0e+0 | 0.457 | 12.18 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.936 | — | to f32 and back | 0e+0 | 2.157 | 2.29 | Vector<UInt16> | 0e+0 | 0.458 | 10.79 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.957 | — | to f32 and back | 0e+0 | 2.012 | 2.46 | Vector<UInt16> | 0e+0 | 0.349 | 14.22 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.691 | — | to f32 and back | 0e+0 | 1.227 | 3.82 | Vector<UInt16> | 0e+0 | 0.348 | 13.47 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.842 | — | to f32 and back | 0e+0 | 2.029 | 2.39 | Vector<UInt16> | 0e+0 | 0.349 | 13.87 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.641 | — | to f32 and back | 0e+0 | 2.328 | 2.42 | Vector<UInt16> | 0e+0 | 0.350 | 16.14 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.477 | — | to f32 and back | 0e+0 | 2.339 | 1.91 | Vector<UInt16> | 0e+0 | 0.528 | 8.49 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.120 | — | to f32 and back | 0e+0 | 2.152 | 2.38 | Vector<UInt16> | 0e+0 | 0.522 | 9.81 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.318 | — | to f32 and back | 0e+0 | 3.167 | 1.68 | Vector<UInt16> | 0e+0 | 0.530 | 10.04 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.846 | — | to f32 and back | 0e+0 | 2.206 | 2.20 | Vector<UInt16> | 0e+0 | 0.530 | 9.15 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.077 | — | to f32 and back | 0e+0 | 2.380 | 2.13 | Vector<UInt16> | 0e+0 | 0.499 | 10.18 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.169 | — | to f32 and back | 0e+0 | 2.145 | 2.41 | Vector<UInt16> | 0e+0 | 0.530 | 9.75 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.033 | — | to f32 and back | 0e+0 | 2.011 | 2.50 | Vector<UInt16> | 0e+0 | 0.349 | 14.42 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.470 | — | to f32 and back | 0e+0 | 2.253 | 1.98 | Vector<UInt16> | 0e+0 | 0.455 | 9.83 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.547 | — | to f32 and back | 0e+0 | 2.247 | 2.47 | Vector<UInt16> | 0e+0 | 0.353 | 15.70 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.018 | — | to f32 and back | 0e+0 | 2.269 | 2.21 | Vector<UInt16> | 0e+0 | 0.528 | 9.51 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.056 | — | to f32 and back | 0e+0 | 2.066 | 2.45 | Vector<UInt16> | 0e+0 | 0.529 | 9.57 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.173 | — | to f32 and back | 0e+0 | 2.120 | 2.44 | Vector<UInt16> | 0e+0 | 0.402 | 12.88 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.775 | — | to f32 and back | 0e+0 | 3.210 | 1.49 | Vector<UInt16> | 0e+0 | 0.528 | 9.05 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.231 | — | to f32 and back | 0e+0 | 3.093 | 1.69 | Vector<UInt16> | 0e+0 | 0.506 | 10.35 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.909 | — | to f32 and back | 0e+0 | 2.098 | 2.34 | Vector<UInt16> | 0e+0 | 0.352 | 13.95 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.054 | — | to f32 and back | 0e+0 | 2.215 | 2.28 | Vector<UInt16> | 0e+0 | 0.357 | 14.15 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.822 | — | to f32 and back | 0e+0 | 2.336 | 2.06 | Vector<UInt16> | 0e+0 | 0.409 | 11.80 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.228 | — | to f32 and back | 0e+0 | 2.308 | 2.27 | Vector<UInt16> | 0e+0 | 0.455 | 11.50 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.895 | — | to f32 and back | 0e+0 | 2.082 | 2.35 | Vector<UInt16> | 0e+0 | 0.455 | 10.76 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.960 | — | to f32 and back | 0e+0 | 2.291 | 2.16 | Vector<UInt16> | 0e+0 | 0.492 | 10.08 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.237 | — | to f32 and back | 0e+0 | 2.024 | 2.59 | Vector<UInt16> | 0e+0 | 0.350 | 14.98 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.147 | — | to f32 and back | 0e+0 | 2.087 | 2.47 | Vector<UInt16> | 0e+0 | 0.356 | 14.45 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.676 | — | to f32 and back | 0e+0 | 2.053 | 2.28 | Vector<UInt16> | 0e+0 | 0.349 | 13.39 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 5.092 | — | to f32 and back | 0e+0 | 2.088 | 2.44 | Vector<UInt16> | 0e+0 | 0.348 | 14.62 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.701 | — | to f32 and back | 0e+0 | 2.229 | 2.11 | Vector<UInt16> | 0e+0 | 0.352 | 13.34 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.844 | — | to f32 and back | 0e+0 | 1.884 | 2.57 | Vector<UInt16> | 0e+0 | 0.351 | 13.80 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.706 | — | to f32 and back | 0e+0 | 1.903 | 2.47 | Vector<UInt16> | 0e+0 | 0.349 | 13.50 | u8dp | 0e+0 |
| cpu | matmul | uint16 | 1024x1024x1024 | 4.665 | — | to f32 and back | 0e+0 | 2.930 | 1.59 | Vector<UInt16> | 0e+0 | 0.350 | 13.34 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.879 | — | to f32 and back | 0e+0 | 5.823 | 1.01 | Vector<UInt32> | 0e+0 | 0.786 | 7.48 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.912 | — | to f32 and back | 0e+0 | 5.513 | 1.07 | Vector<UInt32> | 0e+0 | 0.361 | 16.36 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.882 | — | to f32 and back | 0e+0 | 4.280 | 1.37 | Vector<UInt32> | 0e+0 | 0.364 | 16.14 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.764 | — | to f32 and back | 0e+0 | 4.359 | 1.32 | Vector<UInt32> | 0e+0 | 0.388 | 14.85 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.228 | — | to f32 and back | 0e+0 | 4.181 | 1.49 | Vector<UInt32> | 0e+0 | 0.467 | 13.33 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.837 | — | to f32 and back | 0e+0 | 4.267 | 1.37 | Vector<UInt32> | 0e+0 | 0.356 | 16.38 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.141 | — | to f32 and back | 0e+0 | 8.118 | 0.76 | Vector<UInt32> | 0e+0 | 0.356 | 17.26 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.192 | — | to f32 and back | 0e+0 | 5.538 | 1.12 | Vector<UInt32> | 0e+0 | 0.538 | 11.52 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.909 | — | to f32 and back | 0e+0 | 5.220 | 1.13 | Vector<UInt32> | 0e+0 | 0.502 | 11.76 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.067 | — | to f32 and back | 0e+0 | 6.048 | 1.00 | Vector<UInt32> | 0e+0 | 0.503 | 12.07 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.029 | — | to f32 and back | 0e+0 | 4.243 | 1.42 | Vector<UInt32> | 0e+0 | 0.368 | 16.39 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.072 | — | to f32 and back | 0e+0 | 4.248 | 1.43 | Vector<UInt32> | 0e+0 | 0.365 | 16.63 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.715 | — | to f32 and back | 0e+0 | 8.511 | 0.67 | Vector<UInt32> | 0e+0 | 0.507 | 11.28 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.742 | — | to f32 and back | 0e+0 | 4.625 | 1.24 | Vector<UInt32> | 0e+0 | 0.506 | 11.34 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.085 | — | to f32 and back | 0e+0 | 5.378 | 1.13 | Vector<UInt32> | 0e+0 | 0.522 | 11.65 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.011 | — | to f32 and back | 0e+0 | 5.006 | 1.20 | Vector<UInt32> | 0e+0 | 0.526 | 11.44 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.015 | — | to f32 and back | 0e+0 | 9.019 | 0.67 | Vector<UInt32> | 0e+0 | 0.490 | 12.29 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.145 | — | to f32 and back | 0e+0 | 4.184 | 1.47 | Vector<UInt32> | 0e+0 | 0.471 | 13.05 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.987 | — | to f32 and back | 0e+0 | 3.855 | 1.55 | Vector<UInt32> | 0e+0 | 0.475 | 12.62 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.035 | — | to f32 and back | 0e+0 | 6.013 | 1.00 | Vector<UInt32> | 0e+0 | 0.363 | 16.65 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.702 | — | to f32 and back | 0e+0 | 5.975 | 0.95 | Vector<UInt32> | 0e+0 | 0.361 | 15.82 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.596 | — | to f32 and back | 0e+0 | 4.132 | 1.35 | Vector<UInt32> | 0e+0 | 0.361 | 15.52 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.793 | — | to f32 and back | 0e+0 | 4.333 | 1.34 | Vector<UInt32> | 0e+0 | 0.364 | 15.90 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.805 | — | to f32 and back | 0e+0 | 6.381 | 0.91 | Vector<UInt32> | 0e+0 | 0.363 | 15.97 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.798 | — | to f32 and back | 0e+0 | 4.272 | 1.36 | Vector<UInt32> | 0e+0 | 0.365 | 15.89 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.981 | — | to f32 and back | 0e+0 | 4.277 | 1.40 | Vector<UInt32> | 0e+0 | 0.363 | 16.47 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.758 | — | to f32 and back | 0e+0 | 4.136 | 1.39 | Vector<UInt32> | 0e+0 | 0.360 | 15.98 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.692 | — | to f32 and back | 0e+0 | 4.264 | 1.33 | Vector<UInt32> | 0e+0 | 0.362 | 15.71 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.686 | — | to f32 and back | 0e+0 | 6.155 | 0.92 | Vector<UInt32> | 0e+0 | 0.363 | 15.65 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.850 | — | to f32 and back | 0e+0 | 4.314 | 1.36 | Vector<UInt32> | 0e+0 | 0.364 | 16.07 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.770 | — | to f32 and back | 0e+0 | 4.077 | 1.42 | Vector<UInt32> | 0e+0 | 0.357 | 16.18 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.585 | — | to f32 and back | 0e+0 | 4.489 | 1.24 | Vector<UInt32> | 0e+0 | 0.365 | 15.30 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.807 | — | to f32 and back | 0e+0 | 4.472 | 1.30 | Vector<UInt32> | 0e+0 | 0.358 | 16.21 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.630 | — | to f32 and back | 0e+0 | 4.154 | 1.36 | Vector<UInt32> | 0e+0 | 0.364 | 15.49 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.629 | — | to f32 and back | 0e+0 | 6.207 | 0.91 | Vector<UInt32> | 0e+0 | 0.362 | 15.55 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.737 | — | to f32 and back | 0e+0 | 4.264 | 1.35 | Vector<UInt32> | 0e+0 | 0.351 | 16.33 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.750 | — | to f32 and back | 0e+0 | 6.767 | 0.85 | Vector<UInt32> | 0e+0 | 0.359 | 16.02 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.819 | — | to f32 and back | 0e+0 | 5.085 | 1.14 | Vector<UInt32> | 0e+0 | 0.357 | 16.31 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.454 | — | to f32 and back | 0e+0 | 5.864 | 0.93 | Vector<UInt32> | 0e+0 | 0.457 | 11.94 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.094 | — | to f32 and back | 0e+0 | 4.207 | 1.45 | Vector<UInt32> | 0e+0 | 0.466 | 13.09 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.993 | — | to f32 and back | 0e+0 | 7.891 | 0.76 | Vector<UInt32> | 0e+0 | 0.519 | 11.55 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.821 | — | to f32 and back | 0e+0 | 3.965 | 1.47 | Vector<UInt32> | 0e+0 | 0.532 | 10.93 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.365 | — | to f32 and back | 0e+0 | 6.420 | 0.99 | Vector<UInt32> | 0e+0 | 0.487 | 13.08 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.835 | — | to f32 and back | 0e+0 | 4.611 | 1.27 | Vector<UInt32> | 0e+0 | 0.356 | 16.37 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.796 | — | to f32 and back | 0e+0 | 4.257 | 1.36 | Vector<UInt32> | 0e+0 | 0.364 | 15.91 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.776 | — | to f32 and back | 0e+0 | 4.005 | 1.44 | Vector<UInt32> | 0e+0 | 0.361 | 16.01 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.976 | — | to f32 and back | 0e+0 | 4.272 | 1.40 | Vector<UInt32> | 0e+0 | 0.395 | 15.14 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.415 | — | to f32 and back | 0e+0 | 8.225 | 0.78 | Vector<UInt32> | 0e+0 | 0.426 | 15.05 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.784 | — | to f32 and back | 0e+0 | 5.845 | 0.99 | Vector<UInt32> | 0e+0 | 0.364 | 15.91 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.434 | — | to f32 and back | 0e+0 | 4.403 | 1.23 | Vector<UInt32> | 0e+0 | 0.354 | 15.36 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.653 | — | to f32 and back | 0e+0 | 4.249 | 1.33 | Vector<UInt32> | 0e+0 | 0.563 | 10.03 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.225 | — | to f32 and back | 0e+0 | 5.009 | 1.24 | Vector<UInt32> | 0e+0 | 0.544 | 11.44 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.915 | — | to f32 and back | 0e+0 | 6.744 | 0.88 | Vector<UInt32> | 0e+0 | 0.359 | 16.47 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.200 | — | to f32 and back | 0e+0 | 8.039 | 0.77 | Vector<UInt32> | 0e+0 | 0.505 | 12.28 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.079 | — | to f32 and back | 0e+0 | 6.696 | 0.91 | Vector<UInt32> | 0e+0 | 0.514 | 11.83 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.930 | — | to f32 and back | 0e+0 | 4.263 | 1.39 | Vector<UInt32> | 0e+0 | 0.358 | 16.55 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.549 | — | to f32 and back | 0e+0 | 4.186 | 1.33 | Vector<UInt32> | 0e+0 | 0.359 | 15.45 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.778 | — | to f32 and back | 0e+0 | 4.294 | 1.35 | Vector<UInt32> | 0e+0 | 0.384 | 15.04 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.824 | — | to f32 and back | 0e+0 | 7.084 | 0.82 | Vector<UInt32> | 0e+0 | 0.409 | 14.23 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.520 | — | to f32 and back | 0e+0 | 6.720 | 0.82 | Vector<UInt32> | 0e+0 | 0.406 | 13.58 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.576 | — | to f32 and back | 0e+0 | 7.315 | 0.76 | Vector<UInt32> | 0e+0 | 0.415 | 13.43 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.640 | — | to f32 and back | 0e+0 | 6.681 | 0.84 | Vector<UInt32> | 0e+0 | 0.412 | 13.70 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.844 | — | to f32 and back | 0e+0 | 6.446 | 0.91 | Vector<UInt32> | 0e+0 | 0.422 | 13.84 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.534 | — | to f32 and back | 0e+0 | 6.780 | 0.82 | Vector<UInt32> | 0e+0 | 0.411 | 13.47 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.502 | — | to f32 and back | 0e+0 | 6.350 | 0.87 | Vector<UInt32> | 0e+0 | 0.423 | 13.02 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.750 | — | to f32 and back | 0e+0 | 7.668 | 0.75 | Vector<UInt32> | 0e+0 | 0.415 | 13.86 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.622 | — | to f32 and back | 0e+0 | 6.472 | 0.87 | Vector<UInt32> | 0e+0 | 0.416 | 13.50 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.528 | — | to f32 and back | 0e+0 | 6.325 | 0.87 | Vector<UInt32> | 0e+0 | 0.415 | 13.32 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.583 | — | to f32 and back | 0e+0 | 5.062 | 1.10 | Vector<UInt32> | 0e+0 | 0.412 | 13.54 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.943 | — | to f32 and back | 0e+0 | 6.407 | 0.93 | Vector<UInt32> | 0e+0 | 0.408 | 14.58 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.742 | — | to f32 and back | 0e+0 | 6.978 | 0.82 | Vector<UInt32> | 0e+0 | 0.427 | 13.45 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.425 | — | to f32 and back | 0e+0 | 7.388 | 0.73 | Vector<UInt32> | 0e+0 | 0.412 | 13.17 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.616 | — | to f32 and back | 0e+0 | 7.190 | 0.78 | Vector<UInt32> | 0e+0 | 0.411 | 13.68 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.004 | — | to f32 and back | 0e+0 | 6.205 | 0.97 | Vector<UInt32> | 0e+0 | 0.416 | 14.43 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.928 | — | to f32 and back | 0e+0 | 8.268 | 0.72 | Vector<UInt32> | 0e+0 | 0.425 | 13.96 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.590 | — | to f32 and back | 0e+0 | 6.779 | 0.82 | Vector<UInt32> | 0e+0 | 0.415 | 13.47 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.123 | — | to f32 and back | 0e+0 | 4.889 | 1.25 | Vector<UInt32> | 0e+0 | 0.424 | 14.44 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.240 | — | to f32 and back | 0e+0 | 5.633 | 1.11 | Vector<UInt32> | 0e+0 | 0.359 | 17.39 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.503 | — | to f32 and back | 0e+0 | 4.286 | 1.28 | Vector<UInt32> | 0e+0 | 0.365 | 15.07 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.852 | — | to f32 and back | 0e+0 | 5.951 | 0.98 | Vector<UInt32> | 0e+0 | 0.362 | 16.17 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.961 | — | to f32 and back | 0e+0 | 4.288 | 1.39 | Vector<UInt32> | 0e+0 | 0.353 | 16.91 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.822 | — | to f32 and back | 0e+0 | 4.322 | 1.35 | Vector<UInt32> | 0e+0 | 0.399 | 14.61 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.401 | — | to f32 and back | 0e+0 | 5.814 | 1.10 | Vector<UInt32> | 0e+0 | 0.494 | 12.95 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.127 | — | to f32 and back | 0e+0 | 5.464 | 1.12 | Vector<UInt32> | 0e+0 | 0.365 | 16.79 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.948 | — | to f32 and back | 0e+0 | 5.491 | 1.08 | Vector<UInt32> | 0e+0 | 0.541 | 11.00 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.123 | — | to f32 and back | 0e+0 | 6.570 | 0.93 | Vector<UInt32> | 0e+0 | 0.534 | 11.47 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.333 | — | to f32 and back | 0e+0 | 5.285 | 1.20 | Vector<UInt32> | 0e+0 | 0.506 | 12.53 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.189 | — | to f32 and back | 0e+0 | 3.936 | 1.57 | Vector<UInt32> | 0e+0 | 0.545 | 11.35 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.128 | — | to f32 and back | 0e+0 | 8.287 | 0.74 | Vector<UInt32> | 0e+0 | 0.531 | 11.53 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.055 | — | to f32 and back | 0e+0 | 4.366 | 1.39 | Vector<UInt32> | 0e+0 | 0.374 | 16.21 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.230 | — | to f32 and back | 0e+0 | 8.019 | 0.78 | Vector<UInt32> | 0e+0 | 0.504 | 12.37 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.138 | — | to f32 and back | 0e+0 | 6.384 | 0.96 | Vector<UInt32> | 0e+0 | 0.497 | 12.35 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.996 | — | to f32 and back | 0e+0 | 6.792 | 0.88 | Vector<UInt32> | 0e+0 | 0.500 | 12.00 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.874 | — | to f32 and back | 0e+0 | 4.176 | 1.41 | Vector<UInt32> | 0e+0 | 0.357 | 16.45 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 5.819 | — | to f32 and back | 0e+0 | 4.311 | 1.35 | Vector<UInt32> | 0e+0 | 0.356 | 16.34 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.050 | — | to f32 and back | 0e+0 | 4.353 | 1.39 | Vector<UInt32> | 0e+0 | 0.379 | 15.98 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.348 | — | to f32 and back | 0e+0 | 5.484 | 1.16 | Vector<UInt32> | 0e+0 | 0.508 | 12.49 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.020 | — | to f32 and back | 0e+0 | 8.110 | 0.74 | Vector<UInt32> | 0e+0 | 0.494 | 12.19 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.113 | — | to f32 and back | 0e+0 | 5.157 | 1.19 | Vector<UInt32> | 0e+0 | 0.498 | 12.27 | u8dp | 0e+0 |
| cpu | matmul | uint32 | 1024x1024x1024 | 6.112 | — | to f32 and back | 0e+0 | 6.258 | 0.98 | Vector<UInt32> | 0e+0 | 0.522 | 11.71 | u8dp | 0e+0 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.336 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.026 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.275 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.480 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.446 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.771 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.444 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.278 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 1.446 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.021 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.021 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.022 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.014 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.022 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.020 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.014 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.022 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.018 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.019 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.017 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.015 | — | f32 | 6.46e+2 |
| cpu | matmul | bool | 1024x1024x1024 | — | — | — | — | — | — | — | — | 2.016 | — | f32 | 6.46e+2 |
