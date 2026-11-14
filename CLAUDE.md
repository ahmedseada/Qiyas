# Qiyas / Adad: project notes

## Adad's targets (the owner's, keep them in view)

1. Generic: an operation is defined once. Adding a type is only defining it: no per-type kernels, no route keyed by a type's name.
2. No type is excluded as awkward.
3. Each device reaches its measured hardware limit. Adad is its own standard.
4. Correct on every device, proven by the gates: bit-identical where defined, err never worse.
5. Published results are reproducible (adad vs number).
6. Order: CPU/CUDA → Vulkan → HIP/GCN → the core library written from Adad's design (the core follows Adad, never the other way).

First principle: per operation, choose the better lane by the data and by measurement.

## The design: how a fast path is chosen

A fast path (block route, matrix units, an integer lane) is never chosen by the type's name. Three checks decide, in this order:

1. **Device capability.** Does the device accept the instruction? (CUDA: the driver accepts the PTX shape, e.g. `MmaShape`/`Accepts`, `Dp4aSupported`; CPU: the ISA flag, e.g. `S8Dot.IsSupported`.)
2. **Format fit and exactness.** Is it allowed for this type and this data? Read from the type's own **description**, not its name:
   - a block-scaled type declares `ScaledBlock(Values, Bytes, CodeBits, CodeBias)` (q8_0 = (32, 34, 8, 0), q4_0 = (32, 18, 4, −8));
   - a float declares `FloatBits = (Exponent, Mantissa)` (float16 = (5, 10), bfloat16 = (8, 7));
   - integers already go by bits and the data's bounds (`ProductLane` / `Bounds`).
3. **Measured speed at the shape.** `Measure.Faster` against the generic path, once per type and shape, cached. A route that passes 1 and 2 but loses the measurement is not taken.

So a new type gets its fast paths by declaring its description; nothing in the kernels or the routes names it.

## The finding (local branch `q8-generic`, 2026-11-14)

Branch `q8-generic` in this repo: commits ea5bb68, ba60001, 423f1d7 on top of 3367cf2. **Local only, not pushed.** It adds `ADAD_ROUTES` (env var) so one build runs four ways:
`name` (today's code, default), `props` (by description), `generic` (no fast paths), `measured` (capability + description + measurement).
Scope: the q8_0/q4_0 block routes (block product, block gemv) on CPU and CUDA, and the 16-bit float matrix units on CUDA.

Medians, ms. Current = `name` at 3367cf2, Measured = `measured`, None = `generic`:

| Type | Device | Op | Current | Measured | None | Measured chose |
|---|---|---|---|---|---|---|
| q8_0 | CUDA | matmul_nt | 0.443 | 0.443 | 1.251 | s8 mma blocks |
| q8_0 | CUDA | gemv | 0.441 | **0.341** | 0.343 | generic |
| q8_0 | CPU | gemv | 1.423 | **0.504** | 1.498 | blocks (after the unpack fix) |
| q4_0 | CUDA | matmul_nt | 0.362 | 0.362 | 1.189 | s8 mma blocks |
| q4_0 | CUDA | gemv | 0.240 | **0.194** | 0.194 | generic |
| q4_0 | CPU | gemv | 1.550 | **1.455** | 1.426 | generic |
| float16, bfloat16 | CUDA | matmul, matmul_nt | ~0.22 | ~0.22 | 0.87–0.91 | matrix units |

float32, int8, bool: unchanged. All modes pass the gates (cpu 278, cuda 539, 0 failing). Err is identical per row.
By name vs by description (`name` vs `props`, ab–ab4, 90 rows over all 8 types and 6 ops): the same lane on all 90 rows, same err; time ratio median 1.000 (range 0.89–1.07, CPU noise).

Conclusions:
1. Choosing by the description is exactly as fast as choosing by name. Name checks buy nothing.
2. The measured design keeps the winners (block matmul_nt 2.8–3.3×, f16/bf16 units 4×) and drops the losers. Today's name-keyed block gemv routes were slower than the generic path and were never measured. Measured makes CUDA block gemv 1.24–1.29× faster and CPU q4_0 gemv 1.07× faster.
3. Separate fix found: in CPU `UnpackBlock`, q8_0 (8-bit codes, bias 0) copies its 32 codes in one move (`MemoryMarshal.Cast` + `CopyTo`). CPU q8_0 gemv 1.423 → 0.504 ms (2.8×).

Open observations (not yet explained or fixed):
- CPU float16/bfloat16 matmul and matmul_nt read 2–7% slower in `measured` mode (float16 matmul 1.278 → 1.377 ms); `generic` on the same build is not slower. No measurement runs for these rows, so the cause is unknown.
- CPU int2 matmul reads 9% slower on the branch build in both `measured` and `generic` (0.330 → 0.362 / 0.366 ms). The routes don't touch int2, so it is the build, not the choice.
- CPU matmul_nt for small types is 4–6× slower than matmul (int2 2.02 vs 0.33 ms; bool 5.9 vs 1.35 ms).

Runs and scripts: `local/q8runs/` (ab.sh … ab5.sh, ab*-dev-variant-round.txt, g*-gate files). Snapshots: `local/snap/q8-base` (3367cf2), `q8-test3` (423f1d7).

## What remains to apply the design across Adad

The branch change is 11 files, +214 / −35. About 60 of those lines are the experiment's mode switch, which goes away once the measured design is kept.

Type-name checks left: **91 lines in 30 files** (CUDA 42, Vulkan 42, shared 5, CPU 2). Find them with:
`git grep -nE '\.Is\("|case "(q8_0|q4_0|q4_k|q6_k|float..|bfloat16)"|Name (is|==) "' -- src/Adad`

| Kind | Lines | What it takes |
|---|---|---|
| Codecs: reading/writing a type's bits in CUDA PTX and Vulkan SPIR-V (float16, bfloat16, float8, q8_0, q4_0, q4_k, q6_k) | ~44 | Floats: generic from (Exponent, Mantissa). q8_0/q4_0: from `ScaledBlock`. **q4_k/q6_k don't fit `ScaledBlock`** (scaled sub-blocks inside a 256-value block): they need a richer description and one generic encoder/decoder per device. Must stay bit-identical under the gates. |
| "float32 or float64" own-lane checks (attention, conv backward, ctc, recurrent, routing) | ~20 | Mechanical: read from the description. |
| float64 sliced route, matrix-unit kind (Vulkan, CUDA attention) | ~17 | Description, plus a measured choice where one doesn't exist yet. |
| Block route gates (Vulkan products, the CPU block-product check), uint64, `BlockValues` | ~10 | Mechanical, as on the branch. |

Every remaining fast path also needs its measured choice, like `GemvBlocksFaster` and `UseUnits` on the branch (~20–30 lines each).

Estimate (a guess; the name-check count was underestimated once already): ~12–20 h in total, of which CPU + CUDA alone is ~6–9 h. The Vulkan half touches the same files as the paused Vulkan agents' uncommitted work (Qiyas-vk1 f088a0c, Qiyas-vk2 c0d8e72), so it waits until that work is committed.

Status: nothing started. Waiting for the owner's go.
