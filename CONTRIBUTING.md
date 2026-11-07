# Contributing to Qiyas

Thank you for helping. Qiyas is licensed under the [Apache License, Version 2.0](LICENSE).

## Contributor License Agreement

Every contributor signs the [Individual Contributor License Agreement](CLA.md) before a pull request can be merged.
You keep the copyright of your work; the agreement grants the copyright and patent licenses the project needs to
distribute it. When you open your first pull request, the CLA Assistant bot posts a link; sign by replying on the
pull request as it asks. One signature covers all your future contributions.

## Before you open a pull request

- Every C# file starts with the license header:

  ```csharp
  // Copyright (c) 2026 Ahmed Seada
  // Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
  ```

- `sh guard.sh` passes: it checks the header and that no file names anything outside the repository's own code.
- `dotnet build -c Release` builds with 0 errors.
- An operation is written once over the number contract, works on every stored type on the CPU and on CUDA, and is
  scored against a float64 reference. It has a gate in `src/Checks.cs` on odd shapes and a bench row.
- The gates pass on both devices: `dotnet bin/Release/net10.0/Qiyas.dll cuda --subjects qiyas --types float32 --ops add`
  runs every gate at startup (about 30 s) and must report no `DIFFERS` or `FAIL`.
- Nothing already there is removed, renamed or weakened. A speed change shows its effect in the bench rows of the
  family it touches.
