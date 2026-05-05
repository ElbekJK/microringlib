#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH

echo "== Reinstall editable package =="
pip uninstall -y microringlib || true
pip install -e .

echo
echo "== Confirm import path =="
python - <<'PY'
import pathlib
import microringlib
print(pathlib.Path(microringlib.__file__).resolve())
PY

echo
echo "== Run unit tests =="
pytest

echo
echo "== Run demo and acceleration checks =="
python tools/check_acceleration_and_demos.py --timeout 180

echo
echo "== Build package =="
python -m build

echo
echo "All pre-commit checks passed."
