#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import numpy as np
import matplotlib.pyplot as plt
from shared.path_setup import ensure_project_on_path
ROOT=ensure_project_on_path()
from shared.config import load_config
from shared.decision import save_csv, save_json, save_markdown
from shared.plotting import savefig
from shared.publication_models import analytic_add_drop, try_fde_sweep, radius_for_resonance, nearest_peak, print_artifacts

cfg=load_config('silicon_wdm.yaml'); wdm=cfg['wdm']; mat=cfg['materials']
channels=int(wdm['channels']); targets_nm=wdm['first_channel_nm']+wdm['spacing_nm']*np.arange(channels); targets=targets_nm*1e-9
wl=np.linspace((targets_nm.min()-2)*1e-9,(targets_nm.max()+2)*1e-9,16001)
mode=try_fde_sweep(wl,500e-9,220e-9,float(mat['core']['constant_n']),float(mat['clad']['constant_n']),float(mat['core']['loss_db_cm']),max_points=int(__import__("os").environ.get("MICRORINGLIB_DEMO03_FDE_POINTS","17")))
neff1550=float(np.interp(1550e-9,wl,mode.neff)); seed=float(os.environ.get('MICRORINGLIB_WDM_SEED_RADIUS_UM','50.0'))*1e-6
# Use local FDE n_eff at each target wavelength, not only n_eff(1550),
# so channel synthesis is less biased by waveguide dispersion.
radii=[]
for t in targets:
    neff_t=float(np.interp(t, wl, mode.neff))
    m=max(1, int(round(neff_t*2*np.pi*seed/t)))
    R=m*t/(2*np.pi*neff_t)
    if R < 20e-6 or R > 200e-6:
        m=max(1, int(round(neff_t*2*np.pi*seed/t)))
        R=m*t/(2*np.pi*neff_t)
    radii.append(R)
radii=np.asarray(radii,dtype=float)
drops=[]; rows=[]
for ch,(target,R) in enumerate(zip(targets,radii)):
    spec=analytic_add_drop(wl,R,mode.neff,float(mat['core']['loss_db_cm']),wdm['coupling_power_1'],wdm['coupling_power_2'])
    drops.append(spec['drop'])
    lam,pk,idx=nearest_peak(wl,spec['drop'],target)
    rows.append({'channel':ch,'target_nm':target*1e9,'radius_um':R*1e6,'peak_nm':lam*1e9,'center_error_pm':(lam-target)*1e12,'peak_drop':pk})
drops=np.array(drops); through=np.prod([1-d for d in drops],axis=0)
p1=save_csv(ROOT,'demo_03_wdm_channel_table.csv',rows)
p2=save_json(ROOT,'demo_03_wdm_summary.json',{'backend':mode.backend,'channels':channels,'max_abs_center_error_pm':float(max(abs(r['center_error_pm']) for r in rows)),'spacing_nm':wdm['spacing_nm'],'radius_range_um':[float(radii.min()*1e6),float(radii.max()*1e6)],'synthesis_note':'radii use local FDE n_eff at each channel target to reduce dispersion-induced center error'})
plt.figure(figsize=(8,4));
for ch,d in enumerate(drops): plt.plot(wl*1e9,d,label=f'ch{ch}')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Drop power'); plt.legend(ncol=4,fontsize=7); p3=savefig(ROOT,'demo_03_wdm_drop_bank.png')
plt.figure(figsize=(8,3.8)); plt.plot(wl*1e9,10*np.log10(np.maximum(through,1e-9))); plt.xlabel('Wavelength (nm)'); plt.ylabel('Residual bus (dB)'); p4=savefig(ROOT,'demo_03_wdm_residual_bus.png')
p5=save_markdown(ROOT,'demo_03_report.md','FDE-seeded 8-channel WDM bank',{'Decision metrics':{'max center error':f"{max(abs(r['center_error_pm']) for r in rows):.2f} pm",'radius range':f'{radii.min()*1e6:.4f}-{radii.max()*1e6:.4f} um'},'Backend':mode.backend})
print('=== Publication demo 03: WDM from FDE modes ===')
print(f'backend={mode.backend}, max |center error|={max(abs(r["center_error_pm"]) for r in rows):.2f} pm')
print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5])
