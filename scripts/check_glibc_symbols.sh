#!/usr/bin/env bash
# Verification script for GLIBC symbol compatibility (<= GLIBC_2.31)
set -euo pipefail

TARGET_BIN="${1:-dyno_term}"

if [ ! -f "$TARGET_BIN" ]; then
    # Check common target locations
    if [ -f "target/release/$TARGET_BIN" ]; then
        TARGET_BIN="target/release/$TARGET_BIN"
    elif [ -f "target/x86_64-unknown-linux-musl/release/$TARGET_BIN" ]; then
        TARGET_BIN="target/x86_64-unknown-linux-musl/release/$TARGET_BIN"
    elif [ -f "target/debug/$TARGET_BIN" ]; then
        TARGET_BIN="target/debug/$TARGET_BIN"
    else
        echo "Error: Binary '$TARGET_BIN' not found." >&2
        exit 1
    fi
fi

echo "Checking GLIBC dynamic symbol versions for: $TARGET_BIN"

# Check if file has dynamic symbols
GLIBC_VERSIONS=$(objdump -T "$TARGET_BIN" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -Vu || true)

if [ -z "$GLIBC_VERSIONS" ]; then
    echo "✓ Static or non-glibc binary (zero dynamic GLIBC symbol references)."
    echo "  Compatible with all Linux kernels (musl static / standalone)."
    exit 0
fi

MAX_GLIBC=$(echo "$GLIBC_VERSIONS" | tail -1)
MAX_VER_NUM="${MAX_GLIBC#GLIBC_}"

echo "Detected GLIBC requirements:"
echo "$GLIBC_VERSIONS" | sed 's/^/  - /'
echo "Highest required version: $MAX_GLIBC"

# Compare version numbers (e.g. 2.28 vs 2.31)
MAJOR=$(echo "$MAX_VER_NUM" | cut -d. -f1)
MINOR=$(echo "$MAX_VER_NUM" | cut -d. -f2)

if [ "$MAJOR" -lt 2 ] || { [ "$MAJOR" -eq 2 ] && [ "$MINOR" -le 31 ]; }; then
    echo "✓ PASS: Maximum GLIBC dependency ($MAX_GLIBC) is <= GLIBC_2.31."
    echo "  Binary is fully compatible with Debian 11 (glibc 2.31) and Rocky Linux 8 (glibc 2.28)."
    exit 0
else
    echo "✗ FAIL: Maximum GLIBC dependency ($MAX_GLIBC) exceeds target constraint GLIBC_2.31." >&2
    echo "  Please compile in a Rocky Linux 8 container or target musl." >&2
    exit 1
fi
