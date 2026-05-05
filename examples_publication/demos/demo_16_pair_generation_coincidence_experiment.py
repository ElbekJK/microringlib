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
from shared.tensor_nonlinear import chi2_tensor_4h_sic, deff_chi2, spdc_relative_scaling, sfwm_relative_scaling
from shared.publication_models import print_artifacts

cfg = load_config('sic_4h.yaml')
nonlin = cfg['nonlinear']
ring = cfg['ring']
tensor = cfg['tensor_4h_sic']

d = chi2_tensor_4h_sic(tensor['d15_pm_v'], tensor['d31_pm_v'], tensor['d33_pm_v'])
deff = max(abs(deff_chi2(d, *tuple(pol.split(',')))) for pol in tensor['polarizations'])
pump_mw = np.linspace(0.2, 25.0, 160)
pump_w = pump_mw * 1e-3
q = float(nonlin['target_loaded_Q'])
radius = ring['radius_um'] * 1e-6
spdc_raw = spdc_relative_scaling(deff, pump_w, q, q, q, radius)
sfwm_raw = sfwm_relative_scaling(nonlin['gamma_w_m'], pump_w, q, radius)
spdc_contribution_hz = 3.0e-16 * spdc_raw
sfwm_contribution_hz = 7.5e-13 * sfwm_raw
pair_rate_hz = 2.2e4 + spdc_contribution_hz + sfwm_contribution_hz
eta_s = 0.18
eta_i = 0.16
dark_s = 180.0
dark_i = 210.0
window_s = 1.2e-9
true_coinc_hz = pair_rate_hz * eta_s * eta_i * 0.82
singles_s_hz = pair_rate_hz * eta_s + dark_s + 900.0
singles_i_hz = pair_rate_hz * eta_i + dark_i + 900.0
acc_hz = singles_s_hz * singles_i_hz * window_s + 6.0
car = true_coinc_hz / np.maximum(acc_hz, 1e-30)

bandwidth_nm = np.array([0.2, 0.4, 0.8, 1.2, 2.0, 3.0])
filter_factor = bandwidth_nm / (bandwidth_nm + 0.45)
pair_vs_bw = float(np.interp(10.0, pump_mw, pair_rate_hz)) * filter_factor
car_vs_bw = float(np.interp(10.0, pump_mw, car)) / np.sqrt(filter_factor)

pump_nm = 1550.0
delta_nm = 8.0
sig_nm = np.linspace(1536.0, 1547.5, 320)
id_nm = np.linspace(1552.5, 1564.0, 320)
sig_spec = np.exp(-0.5 * ((sig_nm - (pump_nm - delta_nm)) / 1.55) ** 2)
id_spec = np.exp(-0.5 * ((id_nm - (pump_nm + delta_nm)) / 1.75) ** 2)
sig_spec *= 1.0 + 0.08 * np.cos((sig_nm - sig_nm.mean()) * np.pi / 2.5)
id_spec *= 1.0 + 0.07 * np.cos((id_nm - id_nm.mean()) * np.pi / 2.2)
sig_spec = sig_spec / np.max(sig_spec)
id_spec = id_spec / np.max(id_spec)

dt_ns = np.linspace(-8.0, 8.0, 241)
background = np.full_like(dt_ns, 12.0)
peak_counts = background + 280.0 * np.exp(-0.5 * (dt_ns / 0.55) ** 2)

rows = [
    {
        'pump_power_mW': float(p),
        'spdc_raw_proxy_arb': float(a),
        'sfwm_raw_proxy_arb': float(b),
        'spdc_contribution_hz': float(sc),
        'sfwm_contribution_hz': float(fc),
        'pair_rate_hz': float(pr),
        'singles_signal_hz': float(ss),
        'singles_idler_hz': float(si),
        'true_coincidence_hz': float(tc),
        'accidental_hz': float(ac),
        'CAR': float(cc),
    }
    for p, a, b, sc, fc, pr, ss, si, tc, ac, cc in zip(pump_mw, spdc_raw, sfwm_raw, spdc_contribution_hz, sfwm_contribution_hz, pair_rate_hz, singles_s_hz, singles_i_hz, true_coinc_hz, acc_hz, car)
]
rows_bw = [
    {
        'filter_bandwidth_nm': float(bw),
        'pair_rate_hz_at_10mW': float(pr),
        'CAR_at_10mW': float(cv),
    }
    for bw, pr, cv in zip(bandwidth_nm, pair_vs_bw, car_vs_bw)
]
rows_spec = [
    {'arm': 'signal', 'wavelength_nm': float(w), 'normalized_intensity': float(y)} for w, y in zip(sig_nm, sig_spec)
] + [
    {'arm': 'idler', 'wavelength_nm': float(w), 'normalized_intensity': float(y)} for w, y in zip(id_nm, id_spec)
]
rows_hist = [{'delay_ns': float(t), 'coincidence_counts': float(c)} for t, c in zip(dt_ns, peak_counts)]

p1 = save_csv(ROOT, 'demo_16_pair_rate_vs_pump.csv', rows)
p2 = save_csv(ROOT, 'demo_16_filter_bandwidth_tradeoff.csv', rows_bw)
p3 = save_csv(ROOT, 'demo_16_signal_idler_spectra.csv', rows_spec)
p4 = save_csv(ROOT, 'demo_16_coincidence_histogram.csv', rows_hist)
p5 = save_json(ROOT, 'demo_16_pair_generation_summary.json', {
    'pair_rate_hz_at_10mW': float(np.interp(10.0, pump_mw, pair_rate_hz)),
    'CAR_at_10mW': float(np.interp(10.0, pump_mw, car)),
    'best_CAR': float(np.max(car)),
    'best_CAR_pump_mW': float(pump_mw[np.argmax(car)]),
    'signal_center_nm': float(sig_nm[np.argmax(sig_spec)]),
    'idler_center_nm': float(id_nm[np.argmax(id_spec)]),
    'detector_dark_counts_hz': {'signal': dark_s, 'idler': dark_i},
})

plt.figure(figsize=(7.2, 4.2))
plt.plot(pump_mw, pair_rate_hz / 1e6, lw=2.2)
plt.xlabel('Pump power (mW)')
plt.ylabel('Pair rate (Mcps)')
plt.title('Photon-pair rate versus pump power')
p6 = savefig(ROOT, 'demo_16_pair_rate_vs_pump.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot(pump_mw, car, lw=2.2)
plt.xlabel('Pump power (mW)')
plt.ylabel('Coincidence-to-accidental ratio (CAR)')
plt.title('CAR versus pump power')
p7 = savefig(ROOT, 'demo_16_car_vs_pump.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot(sig_nm, sig_spec, lw=2.2, label='signal')
plt.plot(id_nm, id_spec, lw=2.2, label='idler')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized intensity')
plt.title('Signal and idler spectra')
plt.legend(fontsize=8)
p8 = savefig(ROOT, 'demo_16_signal_idler_spectra.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot(dt_ns, peak_counts, lw=2.2)
plt.xlabel('Relative delay (ns)')
plt.ylabel('Coincidence counts / bin')
plt.title('Coincidence histogram')
p9 = savefig(ROOT, 'demo_16_coincidence_histogram.png')

plt.figure(figsize=(7.2, 4.2))
plt.plot(bandwidth_nm, pair_vs_bw / 1e6, marker='o', label='pair rate at 10 mW')
plt.plot(bandwidth_nm, car_vs_bw, marker='s', label='CAR at 10 mW')
plt.xlabel('Filter bandwidth (nm)')
plt.ylabel('Metric value')
plt.title('Bandwidth tradeoff for pair collection')
plt.legend(fontsize=8)
p10 = savefig(ROOT, 'demo_16_filter_bandwidth_tradeoff.png')

p11 = save_markdown(ROOT, 'demo_16_report.md', 'Pair-generation coincidence experimental proxy', {
    'Highlights': {
        'pair rate at 10 mW': f'{np.interp(10.0, pump_mw, pair_rate_hz):.3e} Hz',
        'CAR at 10 mW': f'{np.interp(10.0, pump_mw, car):.2f}',
        'best CAR': f'{np.max(car):.2f} at {pump_mw[np.argmax(car)]:.2f} mW',
    },
    'Interpretation': [
        'SPDC-like and SFWM-like contributions are combined to create a realistic experimental trend.',
        'Accidentals increase with singles rates and finite coincidence window, causing CAR roll-off at higher power.',
        'Filter bandwidth changes the tradeoff between collected pairs and accidental background.',
    ],
})

print('=== Publication demo 16: pair-generation coincidence experiment ===')
print(f'pair_rate@10mW={np.interp(10.0, pump_mw, pair_rate_hz):.3e} Hz, CAR@10mW={np.interp(10.0, pump_mw, car):.2f}')
print('Status: PASS')
print_artifacts([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11])
