#!/usr/bin/env bash
# Builds Craftia release binaries and packages them into dist/.
#
#   Linux:   native cargo build (x86_64-unknown-linux-gnu)
#   Windows: cross-compiled via cargo-xwin (MSVC target) — recommended.
#            Install with:  cargo install cargo-xwin
#            Falls back to the MinGW GNU target if cargo-xwin is missing
#            but x86_64-w64-mingw32-gcc is available.
#
# Prereqs on the build machine:
#   - glslangValidator on PATH (Ubuntu: `apt install glslang-tools`; or the
#     Vulkan SDK). Shaders are compiled to SPIR-V at build time and embedded
#     into the binary, so players never need any SDK.
#
# Usage:
#   scripts/release.sh                # build + package both platforms
#   scripts/release.sh --skip-windows # linux only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_WINDOWS=0
for arg in "$@"; do
  case "$arg" in
    --skip-windows) SKIP_WINDOWS=1 ;;
    *) echo "error: unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# Version from the workspace manifest ([workspace.package] version).
VERSION="$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' Cargo.toml | head -1)"
if [[ -z "$VERSION" ]]; then
  echo "error: could not read version from Cargo.toml" >&2
  exit 1
fi

echo "==> Craftia release v$VERSION"

# --- Linux (native) ---
echo "==> building Linux release"
cargo build --release --bin voxel
bash scripts/package.sh linux "$VERSION" target/release/voxel

# --- Windows (cross-compile) ---
if [[ "$SKIP_WINDOWS" == "1" ]]; then
  echo "==> skipping Windows (--skip-windows)"
else
  if command -v cargo-xwin >/dev/null 2>&1; then
    echo "==> building Windows release via cargo-xwin (MSVC)"
    rustup target add x86_64-pc-windows-msvc
    # cargo-xwin 0.23 auto-accepts the CRT/SDK license on first download.
    cargo xwin build --release --bin voxel --target x86_64-pc-windows-msvc
    bash scripts/package.sh windows "$VERSION" target/x86_64-pc-windows-msvc/release/voxel.exe
  elif command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
    # MinGW cross builds need a linker override; point users at it. This
    # must live in a committed .cargo/config.toml, e.g.:
    #   [target.x86_64-pc-windows-gnu]
    #   linker = "x86_64-w64-mingw32-gcc"
    if ! grep -q 'x86_64-pc-windows-gnu' .cargo/config.toml 2>/dev/null; then
      echo "error: MinGW cross-build requires a linker override in" >&2
      echo "  .cargo/config.toml:" >&2
      echo "    [target.x86_64-pc-windows-gnu]" >&2
      echo "    linker = \"x86_64-w64-mingw32-gcc\"" >&2
      echo "Or install cargo-xwin (cargo install cargo-xwin) instead." >&2
      exit 1
    fi
    echo "==> building Windows release via MinGW (GNU target)"
    rustup target add x86_64-pc-windows-gnu
    cargo build --release --bin voxel --target x86_64-pc-windows-gnu
    bash scripts/package.sh windows "$VERSION" target/x86_64-pc-windows-gnu/release/voxel.exe
  else
    echo "error: neither cargo-xwin nor mingw-w64 found — cannot build Windows." >&2
    echo "Install with: cargo install cargo-xwin" >&2
    exit 1
  fi
fi

# --- Checksums for all archives. Glob only files, never the unpacked
#    directory (sha256sum on a dir would fail and, with set -e, abort). ---
if (cd dist && find . -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.zip' \) \
    -exec sha256sum {} \; 2>/dev/null | sed 's|^\./||' > SHA256SUMS); then
  :
else
  echo "warning: could not generate SHA256SUMS" >&2
fi

echo
echo "==> release artifacts in dist/:"
ls -lh dist/
