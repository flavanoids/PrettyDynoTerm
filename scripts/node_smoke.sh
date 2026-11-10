#!/usr/bin/env bash
# Real-node smoke test script for PrettyDynoTerm on H200, H100, MI100, and MI50 nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DYNO_BIN="${1:-$REPO_ROOT/dyno_term}"

if [ ! -f "$DYNO_BIN" ]; then
    if [ -f "$REPO_ROOT/target/release/dyno_term" ]; then
        DYNO_BIN="$REPO_ROOT/target/release/dyno_term"
    elif [ -f "$REPO_ROOT/target/rocky8/release/dyno_term" ]; then
        DYNO_BIN="$REPO_ROOT/target/rocky8/release/dyno_term"
    else
        echo "Error: dyno_term binary not found at '$DYNO_BIN'." >&2
        echo "Please compile first: ./scripts/build_rocky8_binary.sh" >&2
        exit 1
    fi
fi

echo "========================================================================"
echo " PRETTYDYNOTERM DATACENTER NODE SMOKE TEST"
echo "========================================================================"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host:      $(hostname)"
echo "Kernel:    $(uname -r)"
echo "Binary:    $DYNO_BIN"
echo ""

echo "--- 1. Node GLIBC & Library Check ---"
"$REPO_ROOT/scripts/check_glibc_symbols.sh" "$DYNO_BIN" || true

echo ""
echo "--- 2. Dynamic GPU Driver Libraries Present on Node ---"
ldconfig -p 2>/dev/null | grep -E 'vulkan|libcuda\.so|nvidia-ml|amdhip' || echo "No matching GPU runtime libraries detected via ldconfig"

echo ""
echo "--- 3. Vulkan ICD Registration ---"
ls -la /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || echo "No Vulkan ICD registration directories found."

echo ""
echo "--- 4. Binary Version & Kernel Checksums ---"
"$DYNO_BIN" --version --output json

echo ""
echo "--- 5. Hardware Adapter Enumeration ---"
"$DYNO_BIN" --list-gpus --output json || echo "Adapter listing completed with warning/fallback"

echo ""
echo "--- 6. 5-Second Datacenter Smoke Run (Headless JSONL) ---"
"$DYNO_BIN" --headless --benchmark --gpus all --duration 5 --output jsonl

echo ""
echo "========================================================================"
echo " Node smoke test completed successfully."
echo "========================================================================"
