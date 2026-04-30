# Contributing

Thank you for improving microringlib.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,materials]"
pytest -q
```

## Guidelines

- Keep physical units explicit.
- Add tests for new physics invariants.
- Keep reduced models clearly labeled.
- Prefer backward-compatible APIs when possible.
- For fast helpers, document assumptions such as fixed `n_eff`.

## Testing focus

Important invariants:

- couplers satisfy `|t|^2 + |kappa|^2 = 1`,
- passive add-drop devices satisfy `P_thru + P_drop <= 1`,
- resonance metrics do not silently jump orders during sweeps,
- material models return correctly shaped `n_complex` and `alpha_power` arrays.
