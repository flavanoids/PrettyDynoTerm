#!/usr/bin/env bash
# Builds PrettyDynoTerm in a Rocky Linux 8 container to ensure glibc <= 2.28 compatibility.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER_TOOL="docker"
if command -v podman >/dev/null 2>&1; then
    CONTAINER_TOOL="podman"
fi

IMAGE_NAME="rockylinux8-rust"

echo "Building Rocky Linux 8 container environment ($IMAGE_NAME) using $CONTAINER_TOOL..."
"$CONTAINER_TOOL" build -t "$IMAGE_NAME" -f "$REPO_ROOT/Containerfile.glibc228" "$REPO_ROOT"

echo "Compiling PrettyDynoTerm inside $IMAGE_NAME..."
mkdir -p "$REPO_ROOT/target/rocky8"

"$CONTAINER_TOOL" run --rm \
    -v "$REPO_ROOT:/workspace:Z" \
    -v "$HOME/.cargo:/root/.cargo:ro" \
    -v "$HOME/.rustup:/root/.rustup:ro" \
    -e PATH="/root/.cargo/bin:$PATH" \
    -e CARGO_TARGET_DIR="/workspace/target/rocky8" \
    -w /workspace \
    "$IMAGE_NAME" \
    cargo build --release

echo "Verifying produced binary..."
cp -f "$REPO_ROOT/target/rocky8/release/dyno_term" "$REPO_ROOT/dyno_term"

"$REPO_ROOT/scripts/check_glibc_symbols.sh" "$REPO_ROOT/dyno_term"

echo ""
echo "Successfully compiled GLIBC 2.28 compatible binary at: $REPO_ROOT/dyno_term"
