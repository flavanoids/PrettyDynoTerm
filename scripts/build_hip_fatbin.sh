#!/usr/bin/env bash
set -euo pipefail

# PrettyDynoTerm HIP Fatbin/Code Object Builder for AMD CDNA (gfx906, gfx908, gfx90a, gfx942)
# Can run on host if hipcc is present, or inside a container (rocm/dev-ubuntu-22.04:6.3.4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_CU="${ROOT_DIR}/src/kernels/dyno_kernels.cu"
OUT_CO="${ROOT_DIR}/src/kernels/blobs/dyno_kernels_gfx.co"

mkdir -p "${ROOT_DIR}/src/kernels/blobs"

if command -v hipcc &>/dev/null; then
    echo "▶ Building CDNA fatbin using host hipcc..."
    hipcc -O3 -x hip \
        --offload-arch=gfx906,gfx908,gfx90a,gfx942 \
        -Rpass-analysis=kernel-resource-usage \
        "${SRC_CU}" -o "${OUT_CO}"
    echo "✓ Fatbin generated successfully at ${OUT_CO}"
elif command -v podman &>/dev/null; then
    echo "▶ Building CDNA fatbin using podman (rocm/dev-ubuntu-22.04:6.3.4)..."
    podman run --rm -v "${ROOT_DIR}:/workspace:Z" -w /workspace \
        docker.io/rocm/dev-ubuntu-22.04:6.3.4 \
        hipcc -O3 -x hip \
        --offload-arch=gfx906,gfx908,gfx90a,gfx942 \
        -Rpass-analysis=kernel-resource-usage \
        src/kernels/dyno_kernels.cu -o src/kernels/blobs/dyno_kernels_gfx.co
    echo "✓ Fatbin generated successfully at ${OUT_CO}"
else
    echo "⚠ Neither hipcc nor podman found; keeping committed precompiled blobs."
fi
