#!/usr/bin/env bash
# Install pyCircAdapt from the platform wheel published by the CircAdapt team.
#
# Usage:
#   bash scripts/setup_circadapt.sh
#   CIRCADAPT_VERSION=2602 bash scripts/setup_circadapt.sh
#
# The wheel is installed into .venv with --system-site-packages so that the
# system numpy/scipy/matplotlib are reused.  It is not published on PyPI.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${CIRCADAPT_VERSION:-2602}"
BASE="https://framework.circadapt.org/files"
PYTHON="${PYTHON:-python3}"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "$uname_s" in
  Linux)
    if [[ "$uname_m" != "x86_64" ]]; then
      echo "No Linux wheel for architecture '$uname_m' (only x86_64)." >&2
      exit 1
    fi
    WHEEL="circadapt-${VERSION}-py3-none-linux_x86_64.whl"
    VENV_BIN="$ROOT/.venv/bin"
    ;;
  Darwin)
    WHEEL="circadapt-${VERSION}-py3-none-macosx_12_0_universal2.whl"
    VENV_BIN="$ROOT/.venv/bin"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    WHEEL="circadapt-${VERSION}-py3-none-win_amd64.whl"
    VENV_BIN="$ROOT/.venv/Scripts"
    ;;
  *)
    echo "Unsupported platform: $uname_s" >&2
    exit 1
    ;;
esac

URL="${BASE}/${WHEEL}"

if [[ ! -x "$VENV_BIN/python" && ! -x "$VENV_BIN/python.exe" ]]; then
  "$PYTHON" -m venv --system-site-packages "$ROOT/.venv"
fi

INSTALL="$URL"
TMP_WHEEL=""
if [[ -n "${CIRCADAPT_SHA256:-}" ]]; then
  TMP_WHEEL="$ROOT/.venv/${WHEEL}"
  trap 'rm -f "$TMP_WHEEL"' EXIT
  "$PYTHON" - "$URL" "$TMP_WHEEL" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  "$PYTHON" - "$TMP_WHEEL" "$CIRCADAPT_SHA256" <<'PY'
import hashlib, sys
path, expected = sys.argv[1], sys.argv[2].lower()
actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
if actual != expected:
    raise SystemExit(f"checksum mismatch: expected {expected}, got {actual}")
PY
  INSTALL="$TMP_WHEEL"
else
  echo "warning: CIRCADAPT_SHA256 is not set; the wheel checksum is not verified" >&2
fi

"$VENV_BIN/pip" install --no-deps --force-reinstall "$INSTALL"

"$VENV_BIN/python" - <<'PY'
import circadapt
print(f"pyCircAdapt {getattr(circadapt, '__version__', '?')} installed")
PY

echo "Run the experiments with:"
echo "  $VENV_BIN/python scripts/smoke_test.py"
echo "  $VENV_BIN/python scripts/run_phase0.py --tag circadapt"
