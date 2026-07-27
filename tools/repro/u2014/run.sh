#!/usr/bin/env bash
# Re-run the rustc 1.96 unicode-escape matrix from docs/notes/u2014_repro.md.
# Compiles each .rs file in this directory and reports rc + stderr.
# Usage: tools/repro/u2014/run.sh  (requires rustc 1.96.0+)
#
# Exit code: 0 if the matrix reproduces as documented, non-zero otherwise.

set -u

cd "$(dirname "$0")"

RUSTC="${RUSTC:-rustc}"
echo "Using rustc: $("$RUSTC" --version)"
echo

PASS=0
FAIL=0
RESULTS=()

for src in case_panic_legacy.rs case_panic_curly.rs case_panic_raw.rs; do
    bin="${src%.rs}"
    out="$("$RUSTC" "$src" -o "$bin" 2>&1)"
    rc=$?
    if [ "$rc" = "0" ]; then
        PASS=$((PASS + 1))
        echo "  [PASS] $src   rc=0"
        RESULTS+=("PASS  $src")
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] $src   rc=$rc"
        echo "    stderr head: $(echo "$out" | head -3 | tr '\n' ' |')"
        RESULTS+=("FAIL  $src   rc=$rc")
    fi
    # clean up the binary so subsequent runs start fresh
    rm -f "$bin" "$bin.exe"
done

echo
echo "Result summary:"
for line in "${RESULTS[@]}"; do
    echo "  $line"
done
echo
echo "$PASS passed, $FAIL failed"

# Documented contract: ALL THREE cases should pass on rustc 1.96 (this only
# nails down the isolation column of the matrix; the "in production panic
# context" column requires `cargo build -p voxel-engine` against real source).
if [ "$FAIL" != "0" ]; then
    echo "FAIL: at least one isolated repro did not compile cleanly."
    echo "(If you're on a different rustc version, the matrix contract above"
    echo " may not hold; see docs/notes/u2014_repro.md for the version statement.)"
    exit 1
fi

exit 0
