#!/bin/sh
# Proof that the repository's files use nothing of the lib: none may name a lib namespace or type. Run before every build.
cd "$(dirname "$0")"
files="src/Numbers.cs src/Convert.cs src/Types.cs src/Kernels.cs src/Checks.cs src/Team.cs src/Gpu.cs src/Ptx.cs src/GpuOps.cs src/CpuGeneric.cs src/GenericPtx.cs src/Bench/*.cs Program.cs"
pattern='Idrak|ElementType|ElementConversions|CpuBackend|CpuMath|CpuMatMul|CpuRanges|CpuTuning|CpuInfo|ComputeResources|HalfFloats|BFloat16Bits|PtxKernels|CudaDriver|Storage\b|CpuCurrent|CudaHost'
if grep -nE "$pattern" $files; then
  echo "guard: Qiyas references the lib (above). Not built."
  exit 1
fi
echo "guard: Qiyas files ($files) reference nothing of the lib."
