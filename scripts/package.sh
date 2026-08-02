#!/usr/bin/env bash
# Shared packaging for Craftia release builds.
#
# Usage:
#   scripts/package.sh <linux|windows> <version> <path-to-binary>
#
# Assembles dist/craftia-<version>-<platform>-x86_64/ with:
#   - the game binary (voxel / voxel.exe)
#   - assets/ (textures, blocks, ...) minus generated caches
#   - a portable config.toml (window size clamped to 1280x720)
#   - README + LICENSE
#   - a launcher (run.sh / Run Craftia.bat) that cd's to the game folder
#     so assets and config resolve no matter where it's started from
#
# Then produces archives in dist/:
#   - Linux:   .tar.gz and .zip
#   - Windows: .zip
#
# This script is used both by scripts/release.sh (local builds) and the
# GitHub Actions release workflow, so the two can never drift apart.
set -euo pipefail

PLATFORM="${1:?usage: package.sh <linux|windows> <version> <binary>}"
VERSION="${2:?missing version}"
BINARY="${3:?missing binary path}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
NAME="craftia-${VERSION}-${PLATFORM}-x86_64"
PKGDIR="$DIST/$NAME"

if [[ "$PLATFORM" != "linux" && "$PLATFORM" != "windows" ]]; then
  echo "error: platform must be 'linux' or 'windows'" >&2
  exit 1
fi

EXE="voxel"
[[ "$PLATFORM" == "windows" ]] && EXE="voxel.exe"

if [[ ! -f "$BINARY" ]]; then
  echo "error: binary not found: $BINARY" >&2
  exit 1
fi

echo "==> packaging $NAME"
rm -rf "$PKGDIR"
mkdir -p "$PKGDIR"

# --- Binary ---
cp "$BINARY" "$PKGDIR/$EXE"

# --- Assets: everything under assets/ except generated caches ---
mkdir -p "$PKGDIR/assets"
cp -r "$ROOT"/assets/. "$PKGDIR/assets/"
find "$PKGDIR/assets" -type d -name '.cache' -exec rm -rf {} + 2>/dev/null || true
find "$PKGDIR/assets" -type f -name '*.cache' -delete 2>/dev/null || true

# --- Config: ship the repo config, but with a portable window size so a
#    user on a smaller monitor isn't greeted by an off-screen window. ---
sed -E \
  -e 's/^width[[:space:]]*=[[:space:]]*[0-9]+/width = 1280/' \
  -e 's/^height[[:space:]]*=[[:space:]]*[0-9]+/height = 720/' \
  "$ROOT/config.toml" > "$PKGDIR/config.toml"

# --- Docs ---
cp "$ROOT/README.md" "$PKGDIR/README.md"
cp "$ROOT/LICENSE" "$PKGDIR/LICENSE" 2>/dev/null || true

# --- Launchers ---
if [[ "$PLATFORM" == "linux" ]]; then
  cat > "$PKGDIR/run.sh" <<'EOF'
#!/usr/bin/env sh
# Launches Craftia from wherever this script lives, so assets/ and
# config.toml resolve correctly regardless of the caller's cwd.
cd "$(dirname "$0")" || exit 1
exec ./voxel "$@"
EOF
  chmod +x "$PKGDIR/run.sh"
else
  cat > "$PKGDIR/Run Craftia.bat" <<'EOF'
@echo off
rem Launch Craftia from the folder this batch file lives in. The console
rem stays attached so crash/log output is visible; full logs also land in
rem logs/latest.log next to the game.
cd /d "%~dp0"
voxel.exe %*
if errorlevel 1 (
  echo.
  echo Craftia exited with an error. Check logs\latest.log for details.
  pause
)
EOF
fi

# --- Archives ---
cd "$DIST"
if [[ "$PLATFORM" == "linux" ]]; then
  rm -f "$NAME.tar.gz" "$NAME.zip"
  tar -czf "$NAME.tar.gz" "$NAME"
  if command -v zip >/dev/null 2>&1; then
    zip -qr "$NAME.zip" "$NAME"
  fi
else
  rm -f "$NAME.zip"
  if command -v zip >/dev/null 2>&1; then
    zip -qr "$NAME.zip" "$NAME"
  elif command -v powershell >/dev/null 2>&1; then
    powershell -NoProfile -Command "Compress-Archive -Force -Path '$NAME' -DestinationPath '$NAME.zip'"
  elif command -v 7z >/dev/null 2>&1; then
    7z a -tzip "$NAME.zip" "$NAME" >/dev/null
  else
    echo "error: no zip tool available" >&2
    exit 1
  fi
fi

echo "==> created:"
ls -lh "$DIST/$NAME".* 2>/dev/null | sed 's/^/   /'
