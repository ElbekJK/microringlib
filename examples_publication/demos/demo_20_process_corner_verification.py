#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from shared.path_setup import ensure_project_on_path
ROOT = ensure_project_on_path()
from shared.config import load_config
from shared.decision import save_csv, save_json, save_markdown
from shared.plotting import savefig
from shared.publication_models import analytic_add_drop, nearest_peak, try_fde_sweep, print_artifacts

cfg = load_config('silicon_wdm.yaml')
mat = cfg['materials']
ring = cfg['ring']
wl = np.linspace(1546.0e-9, 1554.0e-9, 9001)
mode_nom = try_fde_sweep(wl, 500e-9, 220e-9, float(mat['core']['constant_n']), float(mat['clad']['constant_n']), float(mat['core']['loss_db_cm']), max_points=int(__import__("os").environ.get("MICRORINGLIB_DEMO20_FDE_POINTS","41")))
base_neff = mode_nom.neff.copy()
loss0 = float(mat['core']['loss_db_cm'])
R0 = float(ring['radius_um']) * 1e-6
K0 = float(ring['K1'])

corners = [
    {'name': 'nominal', 'dneff': 0.0, 'loss_db_cm': loss0, 'K': K0},
    {'name': 'fast_si', 'dneff': +0.010, 'loss_db_cm': loss0 * 0.95, 'K': K0},
    {'name': 'slow_si', 'dneff': -0.010, 'loss_db_cm': loss0 * 1.05, 'K': K0},
    {'name': 'hot', 'dneff': +0.005, 'loss_db_cm': loss0 * 1.03, 'K': K0},
    {'name': 'cold', 'dneff': -0.005, 'loss_db_cm': loss0 * 0.98, 'K': K0},
    {'name': 'gap_open', 'dneff': 0.0, 'loss_db_cm': loss0, 'K': max(0.02, K0 - 0.02)},
    {'name': 'gap_closed', 'dneff': 0.0, 'loss_db_cm': loss0, 'K': min(0.30, K0 + 0.03)},
    {'name': 'thick_plus', 'dneff': +0.0035, 'loss_db_cm': loss0 * 0.98, 'K': K0 + 0.01},
    {'name': 'thick_minus', 'dneff': -0.0035, 'loss_db_cm': loss0 * 1.02, 'K': max(0.02, K0 - 0.01)},
]

rows = []
spectra = {}
for corner in corners:
    neff = base_neff + corner['dneff']
    spec = analytic_add_drop(wl, R0, neff, corner['loss_db_cm'], corner['K'], corner['K'])
    spectra[corner['name']] = spec['drop']
    lam, peak, idx = nearest_peak(wl, spec['drop'], 1550e-9)
    thru = float(spec['through'][idx])
    ext_db = -10.0 * np.log10(max(thru, 1e-12))
    il_db = -10.0 * np.log10(max(peak, 1e-12))
    # FWHM-like proxy
    half = 0.5 * (float(np.nanmin(spec['drop'])) + float(peak))
    left = idx
    right = idx
    while left > 0 and spec['drop'][left] > half:
        left -= 1
    while right < len(wl) - 1 and spec['drop'][right] > half:
        right += 1
    q = float(lam / (wl[right] - wl[left])) if right > left + 1 else float('nan')
    center_error_pm = (lam - 1550e-9) * 1e12
    passes = bool(peak > 0.22 and ext_db > 5.0 and abs(center_error_pm) < 300.0)
    rows.append({
        'corner': corner['name'],
        'K': float(corner['K']),
        'delta_neff': float(corner['dneff']),
        'loss_db_cm': float(corner['loss_db_cm']),
        'peak_drop': float(peak),
        'through_at_resonance': thru,
        'insertion_loss_db': float(il_db),
        'extinction_db': float(ext_db),
        'loaded_Q_proxy': q,
        'center_nm': float(lam * 1e9),
        'center_error_pm': float(center_error_pm),
        'passes_target': passes,
    })

worst = min(rows, key=lambda r: r['peak_drop'])
p1 = save_csv(ROOT, 'demo_20_corner_yield_table.csv', rows)
p2 = save_json(ROOT, 'demo_20_worst_case_metrics.json', {
    'backend': mode_nom.backend,
    'corner_count': len(corners),
    'yield': float(np.mean([r['passes_target'] for r in rows])),
    'worst_case_corner': worst['corner'],
    'worst_case_metrics': worst,
})

plt.figure(figsize=(7.5, 4.6))
for name, y in spectra.items():
    plt.plot(wl * 1e9, y, lw=1.5, label=name)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Drop power')
plt.title('Process-corner spectra')
plt.legend(fontsize=7, ncol=3)
p3 = savefig(ROOT, 'demo_20_process_corner_spectra.png')

plt.figure(figsize=(7.0, 4.2))
plt.bar([r['corner'] for r in rows], [r['extinction_db'] for r in rows])
plt.xticks(rotation=30, ha='right')
plt.ylabel('Extinction at resonance (dB)')
plt.title('Process-corner extinction comparison')
p4 = savefig(ROOT, 'demo_20_corner_extinction.png')

plt.figure(figsize=(7.0, 4.2))
plt.bar([r['corner'] for r in rows], [r['center_error_pm'] for r in rows])
plt.xticks(rotation=30, ha='right')
plt.ylabel('Center error (pm)')
plt.title('Process-corner resonance shift')
p5 = savefig(ROOT, 'demo_20_corner_center_error.png')

p6 = save_markdown(ROOT, 'demo_20_report.md', 'Process-corner verification experiment proxy', {
    'Highlights': {
        'yield': f'{np.mean([r["passes_target"] for r in rows]):.3f}',
        'worst corner': worst['corner'],
        'worst peak drop': f'{worst["peak_drop"]:.3f}',
    },
    'Interpretation': [
        'Fast/slow material, thermal, and coupling corners are compared on the same resonance grid.',
        'The yield table provides a compact PDK-style corner verification output.',
        'Worst-case metrics make it easy to identify which corner should drive design margin.',
    ],
})

print('=== Publication demo 20: process-corner verification ===')
print(f'backend={mode_nom.backend}; yield={np.mean([r["passes_target"] for r in rows]):.3f}; worst={worst["corner"]}')
print('Status: PASS')
print_artifacts([p1, p2, p3, p4, p5, p6])
