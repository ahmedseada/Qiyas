#!/bin/sh
# Proof that the repository's files use nothing of the lib: none may name a lib namespace or type. Run before every build.
cd "$(dirname "$0")"
# Every C# file of the repository: all of src/ (Abstractions/ and Bench/ too) and Program.cs except the three ignored files (the lib column and its host).
files="$(ls src/*.cs src/Abstractions/*.cs src/Bench/*.cs Program.cs | grep -vE '^src/(CpuCurrent|CudaHost|BenchCuda)\.cs$' | tr '
' ' ')"
pattern='Idrak|ElementType|ElementConversions|CpuBackend|CpuMath|CpuMatMul|CpuRanges|CpuTuning|CpuInfo|ComputeResources|HalfFloats|BFloat16Bits|PtxKernels|CudaDriver|Storage\b|CpuCurrent|CudaHost'
if grep -nE "$pattern" $files; then
  echo "guard: Qiyas references the lib (above). Not built."
  exit 1
fi
# Every C# file opens with the license header (two lines; a byte-order mark or CRLF endings are allowed).
h1='// Copyright (c) 2026 Ahmed Seada'
h2='// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.'
missing=""
for f in $files; do
  l1="$(sed -n 1p "$f" | tr -d '\r' | sed 's/^\xEF\xBB\xBF//')"
  l2="$(sed -n 2p "$f" | tr -d '\r')"
  if [ "$l1" != "$h1" ] || [ "$l2" != "$h2" ]; then missing="$missing $f"; fi
done
if [ -n "$missing" ]; then
  echo "guard: these files lack the license header:$missing"
  exit 1
fi
echo "guard: Qiyas files ($files) reference nothing of the lib and carry the license header."
