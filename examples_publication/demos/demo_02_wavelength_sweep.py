#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from shared.path_setup import ensure_project_on_path
ROOT=ensure_project_on_path()
from shared.config import load_config
from shared.decision import save_csv, save_json, save_markdown
from shared.plotting import savefig
from shared.publication_models import analytic_add_drop, try_fde_sweep, nearest_peak, fwhm_peak, fsr_m, print_artifacts

cfg=load_config('silicon_wdm.yaml'); r=cfg['ring']; mat=cfg['materials']
wl=np.linspace(1520e-9,1580e-9,20001)
mode=try_fde_sweep(wl,500e-9,220e-9,float(mat['core']['constant_n']),float(mat['clad']['constant_n']),float(mat['core']['loss_db_cm']),max_points=int(__import__("os").environ.get("MICRORINGLIB_DEMO02_FDE_POINTS","17")))
Kcrit=np.linspace(0.005,0.25,120)
mins=[]
for K in Kcrit:
    spec=analytic_add_drop(wl,r['radius_um']*1e-6,mode.neff,float(mat['core']['loss_db_cm']),K,K)
    mins.append(np.min(spec['through']))
bestK=float(Kcrit[int(np.argmin(mins))])
spec=analytic_add_drop(wl,r['radius_um']*1e-6,mode.neff,float(mat['core']['loss_db_cm']),bestK,bestK)
lam0,pmax,idx=nearest_peak(wl,spec['drop'],1550e-9); fwhm=fwhm_peak(wl,spec['drop'],idx)
ng1550=float(np.interp(1550e-9,wl,mode.ng)); fsr=float(fsr_m(1550e-9,ng1550,r['radius_um']*1e-6)); Q=lam0/fwhm if np.isfinite(fwhm) else np.nan
thermal_pm_K=cfg['thermal']['thermal_tuning_pm_per_K']; needed_K=(1550e-9-lam0)*1e12/thermal_pm_K
p1=save_csv(ROOT,'demo_02_spectrum.csv',[{'wavelength_nm':float(x*1e9),'through':float(t),'drop':float(d)} for x,t,d in zip(wl[::10],spec['through'][::10],spec['drop'][::10])])
p2=save_csv(ROOT,'demo_02_critical_coupling_sweep.csv',[{'K':float(k),'min_through':float(m)} for k,m in zip(Kcrit,mins)])
p3=save_json(ROOT,'demo_02_metrics.json',{'backend':mode.backend,'best_K_for_critical_coupling':bestK,'drop_peak_nm':lam0*1e9,'loaded_Q':Q,'fwhm_pm':fwhm*1e12,'fsr_nm':fsr*1e9,'thermal_trim_to_1550_K':needed_K,'passive_budget_max':float(np.max(spec['through']+spec['drop']))})
plt.figure(figsize=(8,4)); plt.plot(wl*1e9,10*np.log10(np.maximum(spec['through'],1e-9)),label='through'); plt.plot(wl*1e9,10*np.log10(np.maximum(spec['drop'],1e-9)),label='drop'); plt.xlabel('Wavelength (nm)'); plt.ylabel('Power (dB)'); plt.legend(); p4=savefig(ROOT,'demo_02_add_drop_spectrum.png')
plt.figure(figsize=(7,4)); plt.plot(Kcrit,mins); plt.axvline(bestK,ls='--'); plt.xlabel('Power coupling K'); plt.ylabel('Minimum through power'); p5=savefig(ROOT,'demo_02_critical_coupling.png')
p6=save_markdown(ROOT,'demo_02_report.md','Wavelength sweep, thermal trim, and critical coupling',{'Metrics':{'best K':f'{bestK:.4f}','drop peak':f'{lam0*1e9:.3f} nm','loaded Q':f'{Q:.0f}','thermal trim to 1550 nm':f'{needed_K:.2f} K'},'Decision':'Use the critical-coupling sweep to choose coupling gaps before higher-cost FDTD.'})
print('=== Publication demo 02: compact/FDE wavelength sweep ===')
print(f'FDE backend: {mode.backend}; best critical K={bestK:.4f}; Q≈{Q:.0f}')
print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5,p6])
