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
from shared.publication_models import print_artifacts

cfg = load_config('sic_4h.yaml')
width0 = float(cfg['waveguide']['width_nm'])
thick0 = float(cfg['waveguide']['thickness_nm'])
width_nm = np.linspace(width0 - 70.0, width0 + 70.0, 57)
thick_nm = np.linspace(thick0 - 50.0, thick0 + 50.0, 41)
T_C = np.linspace(20.0, 140.0, 121)

W, H = np.meshgrid(width_nm, thick_nm)
dw = W - width0
dh = H - thick0
# Simple but transparent phase-matching proxy; zero means ideal matching.
delta0 = 1.15e-3 * (dw / 40.0) + 0.95e-3 * (dh / 35.0) + 2.0e-4 * (dw * dh) / (40.0 * 35.0) + 1.0e-4 * (dw / 70.0) ** 2
coeff_T = 2.2e-5
T_required = 25.0 - delta0 / coeff_T
T_clip = np.clip(T_required, T_C.min(), T_C.max())
delta_comp = delta0 + coeff_T * (T_clip - 25.0)
pass_tol = 2.5e-4
mask_uncomp = np.abs(delta0) <= pass_tol
mask_comp = np.abs(delta_comp) <= pass_tol

rows = []
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        rows.append({
            'width_nm': float(W[i, j]),
            'thickness_nm': float(H[i, j]),
            'delta_neff_proxy_uncomp': float(delta0[i, j]),
            'required_temperature_C': float(T_required[i, j]),
            'applied_temperature_C': float(T_clip[i, j]),
            'delta_neff_proxy_comp': float(delta_comp[i, j]),
            'passes_uncompensated': bool(mask_uncomp[i, j]),
            'passes_with_temperature_comp': bool(mask_comp[i, j]),
        })

summary_rows = []
for T in [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]:
    deltaT = delta0 + coeff_T * (T - 25.0)
    summary_rows.append({
        'temperature_C': float(T),
        'phase_matching_yield': float(np.mean(np.abs(deltaT) <= pass_tol)),
        'median_abs_delta_neff_proxy': float(np.median(np.abs(deltaT))),
    })

p1 = save_csv(ROOT, 'demo_24_phase_matching_tolerance_map.csv', rows)
p2 = save_csv(ROOT, 'demo_24_phase_matching_yield_summary.csv', summary_rows)
p3 = save_json(ROOT, 'demo_24_phase_matching_summary.json', {
    'nominal_width_nm': width0,
    'nominal_thickness_nm': thick0,
    'tolerance_abs_delta_neff_proxy': pass_tol,
    'yield_uncompensated': float(np.mean(mask_uncomp)),
    'yield_with_temperature_comp': float(np.mean(mask_comp)),
    'temperature_range_C': [float(T_C.min()), float(T_C.max())],
    'best_required_temperature_C_min': float(np.min(T_required)),
    'best_required_temperature_C_max': float(np.max(T_required)),
})

plt.figure(figsize=(7.4, 4.6))
plt.imshow(np.abs(delta0), origin='lower', aspect='auto', extent=[width_nm.min(), width_nm.max(), thick_nm.min(), thick_nm.max()])
plt.colorbar(label='|Δn_eff| proxy')
plt.xlabel('Width (nm)')
plt.ylabel('Thickness (nm)')
plt.title('Phase-matching tolerance map')
p4 = savefig(ROOT, 'demo_24_phase_matching_tolerance_map.png')

plt.figure(figsize=(7.4, 4.6))
plt.imshow(T_clip, origin='lower', aspect='auto', extent=[width_nm.min(), width_nm.max(), thick_nm.min(), thick_nm.max()])
plt.colorbar(label='Compensation temperature (°C)')
plt.xlabel('Width (nm)')
plt.ylabel('Thickness (nm)')
plt.title('Temperature compensation map')
p5 = savefig(ROOT, 'demo_24_temperature_compensation_map.png')

plt.figure(figsize=(7.4, 4.6))
plt.imshow(np.abs(delta_comp), origin='lower', aspect='auto', extent=[width_nm.min(), width_nm.max(), thick_nm.min(), thick_nm.max()])
plt.colorbar(label='|Δn_eff| proxy after compensation')
plt.xlabel('Width (nm)')
plt.ylabel('Thickness (nm)')
plt.title('Residual mismatch after temperature compensation')
p6 = savefig(ROOT, 'demo_24_phase_matching_residual_map.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot([r['temperature_C'] for r in summary_rows], [r['phase_matching_yield'] for r in summary_rows], marker='o')
plt.xlabel('Applied temperature (°C)')
plt.ylabel('Phase-matching yield')
plt.title('Yield versus global temperature set-point')
p7 = savefig(ROOT, 'demo_24_phase_matching_yield_curve.png')

p8 = save_markdown(ROOT, 'demo_24_report.md', 'Phase-matching tolerance experiment proxy', {
    'Highlights': {
        'yield uncompensated': f'{np.mean(mask_uncomp):.3f}',
        'yield with temperature compensation': f'{np.mean(mask_comp):.3f}',
        'temperature compensation range': f'{np.min(T_required):.1f} to {np.max(T_required):.1f} °C',
    },
    'Interpretation': [
        'Width and thickness errors create a phase-mismatch landscape.',
        'Temperature tuning partially recovers yield across process offsets.',
        'The residual map shows which fabrication points remain challenging even after compensation.',
    ],
})

print('=== Publication demo 24: phase-matching tolerance experiment ===')
print(f'uncompensated_yield={np.mean(mask_uncomp):.3f}, compensated_yield={np.mean(mask_comp):.3f}')
print('Status: PASS')
print_artifacts([p1, p2, p3, p4, p5, p6, p7, p8])
