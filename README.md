# Qiyas

Qiyas (قياس, "measure") is a generic number contract and the kernels over it: every stored type — float64, float32,
float16, bfloat16, float8 (e4m3, e5m2), int64 down to int2, uint64 down to uint2 — computed in the lane the hardware
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
- **Proven, not assumed.** Before anything is timed, every conversion is checked bit for bit against a scalar
  reference written from the format's definition, the dot lanes against scalar sums, exp against the runtime's, and
  the device's conversions against the CPU's.

## The bench

`Qiyas.dll [cpu|cuda|cuda,cpu] [--subjects a,b] [--types ...] [--ops add,matmul,softmax,rms_norm] [--baseline name]`

Every implementation in the assembly that implements `IBenchSubject` (`src/Bench/Contract.cs`) is found by reflection
and becomes a column: `qiyas`, `number` (one implementation over .NET's `INumberBase<T>`, for the types .NET has a
`T` for), and whatever you add. The harness makes the inputs (stored bytes of each type, from Qiyas's proven
conversions, so every subject sees the same bits), times `Run` (medians over an adaptive number of runs), reads the
output after `Collect`, and scores it against float64 math on the same inputs. Results go to `results-qiyas-<devices>.md`.

To add an implementation: one class with a parameterless constructor implementing the four methods, each returning
`null` for anything it does not do (shown as —), never an approximation. Inputs and outputs are stored bytes of the
named type; `Run` must include your device synchronization; `Collect` writes the output; `Lane` says what you computed in.

`profile`, `gpuprofile` and `team` are Qiyas's own diagnostics (matmul phases on each device; the thread team's dispatch cost).

## Building

`dotnet build -c Release` (net10.0). `sh guard.sh` proves that no file in the repository names anything of the library
Qiyas is measured against; run it before every build. That library's column (`lib`) exists only in the author's local
checkout (files listed in `.gitignore`); a clone runs the other subjects.

## Layout

| file | what |
|---|---|
| `src/Numbers.cs` | the contract: `IQiyasNumber<TSelf>`, lanes, dot lanes, the visitors; exp on vectors |
| `src/Types.cs` | the seventeen stored types and their lane rules |
| `src/Convert.cs` | every load and store on vectors; `Widths`, any lane to any lane |
| `src/Kernels.cs` | the CPU kernels: add, product (GotoBLAS-shaped, dot lanes in groups), softmax, rms_norm |
| `src/Team.cs` | the CPU thread team |
| `src/Gpu.cs`, `src/Ptx.cs`, `src/GpuOps.cs` | the CUDA side: driver binding, PTX generation, the operations |
| `src/Checks.cs` | the proofs |
| `src/CpuGeneric.cs`, `src/GenericPtx.cs` | the `INumberBase<T>` version (the `number` column) |
| `src/Bench/` | the bench contract, the type registry, the subjects |
| `Program.cs` | the harness |

## Results

Measured on one machine, 2026-11-06: Intel Core Ultra 9 285 (24 cores, AVX2 with AVX-VNNI), 64 GB; NVIDIA GeForce RTX 5070 Ti
(70 multiprocessors, compute capability 12.0, 16 GB, driver 610.88); Windows 11, .NET 10.0.12. The full table with every
subject's lane and error is `results-qiyas-cuda-cpu.md`; the three columns are the library Qiyas is measured against
(`lib`, present only in the author's checkout), the `INumberBase<T>` version (`number`, — where .NET has no such type)
and Qiyas. Times are medians in ms. Every subject's error against float64 math is in the full table; Qiyas's is equal
to or below the library's on every row, and exact (0) wherever the lane is exact.

**CUDA (2048³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 125.849 / 0.520 / **0.521** / 241.60× | 30.943 / 22.802 / **22.710** / 1.36× | 78.319 / 0.393 / **1.738** / 45.07× | 77.926 / 0.373 / **0.549** / 141.84× |
| float32 | 0.264 / 0.262 / **0.263** / 1.00× | 0.951 / 0.834 / **0.683** / 1.39× | 0.183 / 0.181 / **0.181** / 1.01× | 0.183 / 0.181 / **0.179** / 1.02× |
| float16 | 0.662 / 0.140 / **0.125** / 5.30× | 1.014 / 0.657 / **0.287** / 3.53× | 0.442 / 0.090 / **0.078** / 5.68× | 0.441 / 0.088 / **0.076** / 5.79× |
| bfloat16 | 0.662 / — / **0.124** / 5.34× | 1.003 / — / **0.284** / 3.53× | 0.441 / — / **0.077** / 5.75× | 0.442 / — / **0.077** / 5.76× |
| float8_e4m3 | 489.831 / — / **0.050** / 9835.96× | 130.881 / — / **0.301** / 434.68× | 109.814 / — / **0.068** / 1624.47× | 462.157 / — / **0.046** / 10046.90× |
| float8_e5m2 | 481.180 / — / **0.847** / 567.90× | 117.019 / — / **0.301** / 389.41× | 299.496 / — / **0.069** / 4372.20× | 350.423 / — / **0.050** / 7022.50× |
| int64 | 120.177 / 0.520 / **0.520** / 231.11× | 30.041 / 4.418 / **0.421** / 71.41× |  |  |
| int32 | 0.787 / 0.263 / **0.263** / 2.99× | 1.144 / 1.106 / **0.279** / 4.11× |  |  |
| int16 | 67.399 / 0.139 / **0.124** / 543.98× | 17.225 / 1.585 / **0.231** / 74.44× |  |  |
| int8 | 65.073 / 0.082 / **0.023** / 2780.90× | 21.861 / 1.548 / **0.214** / 102.20× |  |  |
| int4 | 81.782 / — / **0.018** / 4444.68× | 38.206 / — / **0.221** / 173.03× |  |  |
| int2 | 76.942 / — / **0.018** / 4181.61× | 18.738 / — / **0.232** / 80.80× |  |  |
| uint64 | 143.842 / 1.000 / **0.520** / 276.57× | 32.614 / 4.359 / **0.416** / 78.46× |  |  |
| uint32 | 103.208 / 0.263 / **0.263** / 392.58× | 22.980 / 1.076 / **0.271** / 84.67× |  |  |
| uint16 | 68.127 / 0.139 / **0.124** / 547.64× | 17.984 / 1.325 / **0.225** / 79.82× |  |  |
| uint8 | 65.224 / 0.074 / **0.022** / 3033.67× | 18.122 / 1.280 / **0.212** / 85.44× |  |  |
| uint2 | 68.037 / — / **0.017** / 3955.65× | 17.815 / — / **0.226** / 78.72× |  |  |

**CPU (1024³ product, 16M-value add, 4096×4096 rows)** — lib / number / **qiyas** / × (× = lib ÷ qiyas)

| type | add | matmul | softmax | rms_norm |
|---|---|---|---|---|
| float64 | 7.344 / 6.823 / **6.714** / 1.09× | 6.378 / 15.905 / **3.286** / 1.94× | 31.993 / 6.411 / **5.565** / 5.75× | 30.889 / 4.377 / **4.605** / 6.71× |
| float32 | 3.048 / 3.005 / **2.922** / 1.04× | 2.713 / 5.988 / **1.625** / 1.67× | 3.531 / 4.294 / **2.922** / 1.21× | 2.394 / 1.894 / **1.596** / 1.50× |
| float16 | 2.808 / 7.501 / **1.742** / 1.61× | 6.838 / 765.105 / **1.454** / 4.70× | 45.965 / 55.323 / **2.241** / 20.52× | 45.370 / 23.947 / **0.827** / 54.87× |
| bfloat16 | 1.261 / — / **0.915** / 1.38× | 4.736 / — / **1.386** / 3.42× | 23.833 / — / **1.786** / 13.35× | 23.501 / — / **0.474** / 49.55× |
| float8_e4m3 | 26.282 / — / **2.222** / 11.83× | 37.162 / — / **1.477** / 25.17× | 89.788 / — / **3.686** / 24.36× | 429.956 / — / **1.914** / 224.65× |
| float8_e5m2 | 21.284 / — / **2.422** / 8.79× | 29.022 / — / **1.491** / 19.47× | 283.307 / — / **3.878** / 73.05× | 313.137 / — / **2.083** / 150.31× |
| int64 | 19.776 / 6.763 / **6.615** / 2.99× | 6.126 / 24.649 / **0.876** / 7.00× |  |  |
| int32 | 20.288 / 2.959 / **2.948** / 6.88× | 5.923 / 6.204 / **0.607** / 9.76× |  |  |
| int16 | 1.639 / 1.425 / **0.840** / 1.95× | 5.658 / 2.371 / **0.992** / 5.71× |  |  |
| int8 | 1.564 / 0.432 / **0.154** / 10.14× | 6.123 / 2.507 / **0.597** / 10.26× |  |  |
| int4 | 2.437 / — / **0.101** / 24.11× | 11.926 / — / **0.532** / 22.43× |  |  |
| int2 | 2.868 / — / **0.267** / 10.74× | 6.899 / — / **0.514** / 13.42× |  |  |
| uint64 | 6.599 / 6.738 / **6.705** / 0.98× | 6.337 / 22.120 / **0.810** / 7.82× |  |  |
| uint32 | 5.172 / 2.943 / **2.933** / 1.76× | 6.090 / 3.922 / **0.429** / 14.18× |  |  |
| uint16 | 1.375 / 1.283 / **0.796** / 1.73× | 5.178 / 1.925 / **0.698** / 7.42× |  |  |
| uint8 | 1.630 / 0.423 / **0.130** / 12.59× | 6.154 / 2.198 / **0.569** / 10.81× |  |  |
| uint2 | 2.761 / — / **0.232** / 11.91× | 6.861 / — / **0.402** / 17.08× |  |  |

Where Qiyas is not ahead: the memory-bound adds of 32- and 64-bit types (the same bytes moved, within a few percent),
and CUDA float64 matmul, which runs at the device's float64 rate and exact where the library computes it in float32.
