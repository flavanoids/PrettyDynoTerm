#!/usr/bin/env bash
set -euo pipefail

# PrettyDynoTerm PTX Builder for NVIDIA Hopper (sm_90)
# Can run on host if nvcc is present, or inside a container (e.g. nvidia/cuda:12.0.1-devel-ubi8)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_CU="${ROOT_DIR}/src/kernels/dyno_kernels.cu"
OUT_PTX="${ROOT_DIR}/src/kernels/blobs/dyno_kernels_sm90.ptx"

mkdir -p "${ROOT_DIR}/src/kernels/blobs"

if command -v nvcc &>/dev/null; then
    echo "▶ Building sm_90 PTX using host nvcc..."
    nvcc -O3 -ptx --gpu-architecture=compute_90 \
        --resource-usage \
        "${SRC_CU}" -o "${OUT_PTX}"
    echo "✓ PTX generated successfully at ${OUT_PTX}"
elif command -v podman &>/dev/null; then
    echo "▶ Building sm_90 PTX using podman (nvidia/cuda:12.0.1-devel-ubi8)..."
    podman run --rm -v "${ROOT_DIR}:/workspace:Z" -w /workspace \
        docker.io/nvidia/cuda:12.0.1-devel-ubi8 \
        nvcc -O3 -ptx --gpu-architecture=compute_90 \
        --resource-usage \
        src/kernels/dyno_kernels.cu -o src/kernels/blobs/dyno_kernels_sm90.ptx
    echo "✓ PTX generated successfully at ${OUT_PTX}"
else
    echo "⚠ Neither nvcc nor podman found; keeping committed precompiled blobs."
fi
