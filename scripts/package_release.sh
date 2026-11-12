#!/usr/bin/env bash
# Builds and packages PrettyDynoTerm into a standalone, zero-dependency tarball
# using the host's native musl static target (no Docker, Podman, or Alpine required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="x86_64-unknown-linux-musl"
DIST_DIR="$REPO_ROOT/dist"

echo "=== PrettyDynoTerm Zero-Dependency Release Packaging ==="

# 1. Ensure musl target is installed
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "Adding rustup target $TARGET..."
    rustup target add "$TARGET"
fi

# 2. Compile static release binary
echo "Compiling static binary for $TARGET..."
cargo build --target "$TARGET" --release --manifest-path "$REPO_ROOT/Cargo.toml"

BIN_SRC="$REPO_ROOT/target/$TARGET/release/dyno_term"
if [ ! -f "$BIN_SRC" ]; then
    echo "Error: Compiled binary not found at $BIN_SRC" >&2
    exit 1
fi

# 3. Verify zero glibc dependencies
echo "Verifying symbol isolation..."
"$REPO_ROOT/scripts/check_glibc_symbols.sh" "$BIN_SRC"

# 4. Prepare distribution staging area
VERSION=$(grep '^version = ' "$REPO_ROOT/Cargo.toml" | head -1 | cut -d '"' -f 2)
PACKAGE_NAME="dyno_term-v${VERSION}-linux-x86_64"
STAGE_DIR="$DIST_DIR/$PACKAGE_NAME"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

echo "Staging release payload in $STAGE_DIR..."
cp "$BIN_SRC" "$STAGE_DIR/dyno_term"
if command -v strip >/dev/null 2>&1; then
    echo "Stripping symbols to minimize binary size..."
    strip "$STAGE_DIR/dyno_term"
fi

# Copy smoke test helper
cp "$REPO_ROOT/scripts/node_smoke.sh" "$STAGE_DIR/node_smoke.sh"
cp "$REPO_ROOT/scripts/check_glibc_symbols.sh" "$STAGE_DIR/check_glibc_symbols.sh"
chmod +x "$STAGE_DIR/dyno_term" "$STAGE_DIR/node_smoke.sh" "$STAGE_DIR/check_glibc_symbols.sh"

# 5. Create archive & checksum
ARCHIVE_FILE="$DIST_DIR/${PACKAGE_NAME}.tar.gz"
echo "Creating archive $ARCHIVE_FILE..."
tar -C "$DIST_DIR" -czf "$ARCHIVE_FILE" "$PACKAGE_NAME"

cd "$DIST_DIR"
sha256sum "$(basename "$ARCHIVE_FILE")" > "${PACKAGE_NAME}.tar.gz.sha256"

# Also symlink or copy root dyno_term for immediate convenience
cp -f "$STAGE_DIR/dyno_term" "$REPO_ROOT/dyno_term"

echo ""
echo "========================================================================"
echo " Packaging completed successfully!"
echo " Standalone binary: $STAGE_DIR/dyno_term ($(ls -lh "$STAGE_DIR/dyno_term" | awk '{print $5}'))"
echo " Release tarball:   $ARCHIVE_FILE ($(ls -lh "$ARCHIVE_FILE" | awk '{print $5}'))"
echo " SHA-256 Checksum:  $(cat "${PACKAGE_NAME}.tar.gz.sha256" | awk '{print $1}')"
echo "========================================================================"
echo ""
echo "Deploy this single tarball or binary to any target Linux compute node."
echo "Target nodes require NO Rust, NO GCC, NO glibc version constraints, and NO containers."
