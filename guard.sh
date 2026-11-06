#!/bin/sh
# Proof that the repository's files use nothing of the lib: none may name a lib namespace or type. Run before every build.
cd "$(dirname "$0")"
# Every C# file of the repository: all of src/ and Program.cs except the three ignored files (the lib column and its host).
files="$(ls src/*.cs src/Bench/*.cs Program.cs | grep -vE '^src/(CpuCurrent|CudaHost|BenchCuda)\.cs$' | tr '
' ' ')"
pattern='Idrak|ElementType|ElementConversions|CpuBackend|CpuMath|CpuMatMul|CpuRanges|CpuTuning|CpuInfo|ComputeResources|HalfFloats|BFloat16Bits|PtxKernels|CudaDriver|Storage\b|CpuCurrent|CudaHost'
if grep -nE "$pattern" $files; then
  echo "guard: Qiyas references the lib (above). Not built."
  exit 1
fi
echo "guard: Qiyas files ($files) reference nothing of the lib."
