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
from shared.publication_models import analytic_through, try_fde_sweep, fsr_m, print_artifacts

cfg=load_config('silicon_wdm.yaml'); v=cfg['vernier']; mat=cfg['materials']
wl=np.linspace(1520e-9,1580e-9,int(v['wavelength_points']))
mode=try_fde_sweep(wl,500e-9,220e-9,float(mat['core']['constant_n']),float(mat['clad']['constant_n']),float(mat['core']['loss_db_cm']),max_points=int(__import__("os").environ.get("MICRORINGLIB_DEMO04_FDE_POINTS","17")))
R1=v['base_radius_um']*1e-6; R2=10.45e-6; K=v['coupling_power']
p1=analytic_through(wl,R1,mode.neff,float(mat['core']['loss_db_cm']),K)
p2=analytic_through(wl,R2,mode.neff,float(mat['core']['loss_db_cm']),K)
combined=p1*p2
dips=np.where((combined[1:-1]<combined[:-2])&(combined[1:-1]<=combined[2:]))[0]+1
selected=dips[np.argsort(combined[dips])[:8]] if dips.size else np.array([int(np.argmin(combined))])
selected=np.sort(selected)
ng=float(np.interp(1550e-9,wl,mode.ng)); fsr1=float(fsr_m(1550e-9,ng,R1)); fsr2=float(fsr_m(1550e-9,ng,R2)); vernier=abs(fsr1*fsr2/(fsr1-fsr2))
rows=[{'dip_nm':float(wl[i]*1e9),'combined_through':float(combined[i])} for i in selected]
pth1=save_csv(ROOT,'demo_04_vernier_dips.csv',rows)
pth2=save_json(ROOT,'demo_04_vernier_summary.json',{'backend':mode.backend,'R1_um':R1*1e6,'R2_um':R2*1e6,'FSR1_nm':fsr1*1e9,'FSR2_nm':fsr2*1e9,'vernier_period_nm':vernier*1e9,'num_dips':int(dips.size),'radius_mismatch_um':float((R2-R1)*1e6),'plot_order':'cascaded -> ring2 -> ring1'})
def _db(x):
    return 10*np.log10(np.maximum(x,1e-9))
main_idx = int(selected[np.argmin(combined[selected])]) if selected.size else int(np.argmin(combined))
zoom_mask = np.abs(wl*1e9 - wl[main_idx]*1e9) <= 2.0

plt.figure(figsize=(8,4))
plt.plot(wl*1e9,_db(combined),label='cascaded',lw=2.4)
plt.plot(wl*1e9,_db(p2),label='ring 2',lw=1.6,ls='--')
plt.plot(wl*1e9,_db(p1),label='ring 1',lw=1.4,ls=':')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Through (dB)'); plt.title(f'Vernier cascade: period≈{vernier*1e9:.1f} nm'); plt.legend(); pth3=savefig(ROOT,'demo_04_vernier_filter.png')

plt.figure(figsize=(8,4))
plt.plot(wl*1e9,_db(combined),label='cascaded',lw=2.4)
plt.plot(wl*1e9,_db(p2),label='ring 2',lw=1.6,ls='--')
plt.plot(wl*1e9,_db(p1),label='ring 1',lw=1.4,ls=':')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Through (dB)'); plt.title('Full-span Vernier response, dB scale'); plt.legend(); pth4=savefig(ROOT,'demo_04_vernier_filter_db_full.png')

plt.figure(figsize=(8,4))
plt.plot(wl*1e9,combined,label='cascaded',lw=2.4)
plt.plot(wl*1e9,p2,label='ring 2',lw=1.6,ls='--')
plt.plot(wl*1e9,p1,label='ring 1',lw=1.4,ls=':')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Through power'); plt.title('Full-span Vernier response, linear scale'); plt.legend(); pth5=savefig(ROOT,'demo_04_vernier_filter_linear_full.png')

plt.figure(figsize=(8,4))
plt.plot(wl[zoom_mask]*1e9,_db(combined[zoom_mask]),label='cascaded',lw=2.4)
plt.plot(wl[zoom_mask]*1e9,_db(p2[zoom_mask]),label='ring 2',lw=1.6,ls='--')
plt.plot(wl[zoom_mask]*1e9,_db(p1[zoom_mask]),label='ring 1',lw=1.4,ls=':')
plt.axvline(wl[main_idx]*1e9,ls='--',lw=1,label='selected dip')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Through (dB)'); plt.title('Zoom around strongest Vernier feature'); plt.legend(fontsize=8); pth6=savefig(ROOT,'demo_04_vernier_zoom_peak.png')

plt.figure(figsize=(8,4))
plt.plot(wl*1e9,_db(combined),label='cascaded',lw=2.2)
for ii,i in enumerate(selected):
    plt.axvline(wl[i]*1e9,alpha=0.35,lw=0.9,ls='--',label='selected dips' if ii==0 else None)
plt.xlabel('Wavelength (nm)'); plt.ylabel('Cascaded through (dB)'); plt.title('Detected Vernier dips'); plt.legend(); pth7=savefig(ROOT,'demo_04_vernier_dip_markers.png')

plt.figure(figsize=(7,4))
labels=['Ring 1 FSR','Ring 2 FSR','Vernier period']; vals=[fsr1*1e9,fsr2*1e9,vernier*1e9]
plt.bar(labels,vals); plt.ylabel('Wavelength scale (nm)'); plt.title('Vernier scale summary')
for x,yv in enumerate(vals): plt.text(x,yv,f'{yv:.2f}',ha='center',va='bottom',fontsize=9)
pth8=savefig(ROOT,'demo_04_vernier_envelope_summary.png')
pth9=save_markdown(ROOT,'demo_04_report.md','Vernier filter from FDE-seeded compact rings',{'Vernier metrics':{'FSR1':f'{fsr1*1e9:.3f} nm','FSR2':f'{fsr2*1e9:.3f} nm','period':f'{vernier*1e9:.2f} nm'},'Backend':mode.backend,'Plot order':'cascaded -> ring 2 -> ring 1'})
print('=== Publication demo 04: Vernier filter ===')
print(f'backend={mode.backend}; Vernier period≈{vernier*1e9:.2f} nm')
print('Status: PASS'); print_artifacts([pth1,pth2,pth3,pth4,pth5,pth6,pth7,pth8,pth9])
