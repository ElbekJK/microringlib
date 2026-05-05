# Contributing

Thank you for improving `microringlib`.

## Development setup

```bash
git clone https://github.com/ElbekJK/microringlib.git
cd microringlib

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Optional publication/demo backends:

```bash
python -m pip install -e ".[materials,fde,accel,publication]"
```

## Guidelines

- Keep physical units explicit.
- Add tests for new physics invariants.
- Keep reduced models and proxies clearly labeled.
- Prefer backward-compatible APIs when possible.
- Document assumptions such as fixed `n_eff`, fitted coupling, or surrogate FDTD calibration.
- Do not commit generated publication outputs, logs, build artifacts, caches, or local environments.

## Testing focus

Important invariants:

- couplers satisfy `|t|^2 + |kappa|^2 = 1`;
- passive devices satisfy `P_through + P_drop <= 1` when loss is included;
- resonance metrics should not silently jump orders during sweeps;
- material models return correctly shaped index/loss arrays.
