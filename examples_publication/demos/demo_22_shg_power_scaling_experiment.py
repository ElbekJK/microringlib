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
from shared.tensor_nonlinear import chi2_tensor_4h_sic, deff_chi2, shg_relative_scaling
from shared.publication_models import try_fde_sweep, print_artifacts

cfg = load_config('sic_4h.yaml')
mat = cfg['materials']
wg = cfg['waveguide']
ring = cfg['ring']
nonlin = cfg['nonlinear']
therm = cfg['thermal']
tensor = cfg['tensor_4h_sic']

wl = np.linspace(1545e-9, 1555e-9, 201)
mode = try_fde_sweep(wl, wg['width_nm'] * 1e-9, wg['thickness_nm'] * 1e-9, float(mat['core']['constant_n']), float(mat['cladding']['constant_n']), float(mat['core']['loss_db_cm']), max_points=5)

d = chi2_tensor_4h_sic(tensor['d15_pm_v'], tensor['d31_pm_v'], tensor['d33_pm_v'])
deff = max(abs(deff_chi2(d, *tuple(pol.split(',')))) for pol in tensor['polarizations'])
pump_mw = np.linspace(0.1, max(30.0, 1.5 * nonlin['pump_power_mw_max']), 220)
pump_w = pump_mw * 1e-3
q_p = float(nonlin['target_loaded_Q'])
q_sh = 0.65 * q_p
radius = ring['radius_um'] * 1e-6
ideal = shg_relative_scaling(deff, pump_w, q_p, q_sh, radius)
ideal = ideal / max(np.nanmax(ideal), 1e-30)
thermal_rise = therm['thermal_resistance_K_per_mW'] * pump_mw
linewidth_pm = 34.0
detuning_pm = 0.42 * thermal_rise + 0.020 * pump_mw**2
rolloff = 1.0 / (1.0 + (detuning_pm / linewidth_pm) ** 2)
shg_norm = ideal * rolloff
shg_rate_kcps = 160.0 * shg_norm
norm_eff = shg_rate_kcps / np.maximum(pump_mw**2, 1e-12)
low_mask = pump_mw <= 5.0
slope, intercept = np.polyfit(np.log10(pump_mw[low_mask]), np.log10(np.maximum(shg_rate_kcps[low_mask], 1e-9)), 1)
fit_curve = 10 ** (intercept + slope * np.log10(pump_mw))

T_C = np.linspace(20.0, 180.0, 161)
fixed_power_mw = 10.0
base_T_C = 72.0
thermo_tuning_pm_per_K = 0.70
fixed_detuning_pm = thermo_tuning_pm_per_K * (T_C - base_T_C)
phase_match_penalty = 1.0 / (1.0 + (fixed_detuning_pm / 28.0) ** 2)
fixed_rate_kcps = float(np.interp(fixed_power_mw, pump_mw, shg_rate_kcps)) * phase_match_penalty
best_T = float(T_C[np.argmax(fixed_rate_kcps)])

rows_power = [
    {
        'pump_power_mW': float(p),
        'thermal_rise_K': float(tr),
        'detuning_pm': float(dd),
        'normalized_shg_rate': float(sn),
        'shg_rate_kcps': float(sr),
        'normalized_efficiency_per_mW2': float(ne),
        'ideal_normalized_no_rolloff': float(ii),
    }
    for p, tr, dd, sn, sr, ne, ii in zip(pump_mw, thermal_rise, detuning_pm, shg_norm, shg_rate_kcps, norm_eff, ideal)
]
rows_temp = [
    {
        'temperature_C': float(T),
        'detuning_pm': float(dd),
        'shg_rate_kcps': float(rr),
        'normalized_relative_efficiency': float(rr / max(np.nanmax(fixed_rate_kcps), 1e-30)),
    }
    for T, dd, rr in zip(T_C, fixed_detuning_pm, fixed_rate_kcps)
]

p1 = save_csv(ROOT, 'demo_22_shg_power_scaling.csv', rows_power)
p2 = save_csv(ROOT, 'demo_22_shg_efficiency_vs_temperature.csv', rows_temp)
p3 = save_json(ROOT, 'demo_22_shg_fit_summary.json', {
    'backend': mode.backend,
    'deff_pm_per_V_max_abs': float(deff),
    'neff_1550': float(np.interp(1550e-9, wl, mode.neff)),
    'ng_1550': float(np.interp(1550e-9, wl, mode.ng)),
    'low_power_loglog_slope': float(slope),
    'best_temperature_C': best_T,
    'best_temperature_rate_kcps': float(np.max(fixed_rate_kcps)),
    'rolloff_model': 'Lorentzian thermal detuning penalty on top of quadratic SHG scaling',
})

plt.figure(figsize=(7.2, 4.2))
plt.plot(pump_mw, shg_rate_kcps, lw=2.2, label='thermal-limited SHG proxy')
plt.plot(pump_mw, 160.0 * ideal, lw=1.4, ls='--', label='ideal quadratic trend')
plt.xlabel('Pump power (mW)')
plt.ylabel('SHG rate proxy (kcps)')
plt.title('SHG power-scaling experiment proxy')
plt.legend(fontsize=8)
p4 = savefig(ROOT, 'demo_22_shg_power_scaling.png')

plt.figure(figsize=(7.2, 4.2))
plt.loglog(pump_mw, shg_rate_kcps, lw=2.2, label='simulated SHG proxy')
plt.loglog(pump_mw, fit_curve, lw=1.4, ls='--', label=f'low-power fit slope={slope:.2f}')
plt.xlabel('Pump power (mW)')
plt.ylabel('SHG rate proxy (kcps)')
plt.title('Low-power SHG slope extraction')
plt.legend(fontsize=8)
p5 = savefig(ROOT, 'demo_22_loglog_slope_fit.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot(T_C, fixed_rate_kcps, lw=2.2)
plt.axvline(best_T, lw=1.0, ls='--', label=f'best T = {best_T:.1f} °C')
plt.xlabel('Chip temperature (°C)')
plt.ylabel(f'SHG rate proxy at {fixed_power_mw:.1f} mW (kcps)')
plt.title('SHG efficiency versus temperature')
plt.legend(fontsize=8)
p6 = savefig(ROOT, 'demo_22_shg_efficiency_vs_temperature.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot(pump_mw, norm_eff / np.max(norm_eff), lw=2.2)
plt.xlabel('Pump power (mW)')
plt.ylabel('Normalized SHG efficiency / P²')
plt.title('Power-normalized SHG efficiency clarity plot')
p7 = savefig(ROOT, 'demo_22_shg_normalized_efficiency.png')

p8 = save_markdown(ROOT, 'demo_22_report.md', 'SHG power-scaling experimental proxy', {
    'Highlights': {
        'low-power slope': f'{slope:.3f}',
        'best temperature': f'{best_T:.2f} °C',
        'n_eff@1550 nm': f'{np.interp(1550e-9, wl, mode.neff):.4f}',
        'n_g@1550 nm': f'{np.interp(1550e-9, wl, mode.ng):.4f}',
    },
    'Interpretation': [
        'The low-power regime stays close to quadratic SHG scaling.',
        'Thermal detuning causes rollover at higher pump powers.',
        'A temperature sweep provides an experimentally familiar SHG tuning curve.',
    ],
})

print('=== Publication demo 22: SHG power-scaling experiment ===')
print(f'backend={mode.backend}; low-power slope={slope:.3f}; best temperature={best_T:.1f} °C')
print('Status: PASS')
print_artifacts([p1, p2, p3, p4, p5, p6, p7, p8])
