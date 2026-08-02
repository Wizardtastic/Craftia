#!/usr/bin/env bash
# Run the core crate test suite (voxel-ecs, voxel-world, voxel-game).
#
# Background: the Freebuff sandbox's user-level cargo config
# (~/.cargo/config.toml) forces `linker = "rust-lld"` plus a lib-compat libc
# shim via `build.rustflags`, which makes dependency build scripts crash with
# SIGSEGV. The overrides below restore the working system defaults. On normal
# development machines these are no-ops (cc is already the default linker and
# no rustflags are needed), so this script is safe everywhere.
#
# In this sandbox the same overrides are also applied automatically by the
# gitignored `.cargo/config.toml`; this script remains the portable, committed
# entry point for other environments and CI.
#
# Usage: ./scripts/test.sh [extra cargo test args...]
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found on PATH (is rustup installed and sourced?)" >&2
    exit 1
fi

export RUSTFLAGS="${RUSTFLAGS:-}"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER:-cc}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-auto}"

cargo test -p voxel-ecs -p voxel-world -p voxel-game "$@"
