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
