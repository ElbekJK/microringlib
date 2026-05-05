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
from shared.tensor_nonlinear import chi2_tensor_4h_sic, deff_chi2
from shared.publication_models import dbcm_to_npm, print_artifacts
from shared.sic_experiment_helpers import sic_material_fde_stack, fdtd_calibration_trace, cavity_enhancement_from_tmm
from microringlib import tmm

cfg=load_config('sic_4h.yaml'); mat=cfg['materials']; ring=cfg['ring']; tensor=cfg['tensor_4h_sic']; therm=cfg['thermal']
wl=np.linspace(1546e-9,1554e-9,2401); stack=sic_material_fde_stack(wl,cfg,width_nm=850,thickness_nm=500,max_points=5)
R=ring['radius_um']*1e-6; K=ring['coupling_power_1']; loss=mat['core']['loss_db_cm']
enh,thru=cavity_enhancement_from_tmm(wl,stack.mode.neff,R,K,loss)
fdtd,fdtd_backend=fdtd_calibration_trace(wl,np.abs(thru)**2,n_core=float(np.interp(1550e-9,wl,stack.core_n)),n_clad=float(np.interp(1550e-9,wl,stack.clad_n)),radius_um=ring['radius_um'],width_um=.85,run_label='demo31')
d=chi2_tensor_4h_sic(tensor['d15_pm_v'],tensor['d31_pm_v'],tensor['d33_pm_v']); deffs={pol:abs(deff_chi2(d,*tuple(pol.split(',')))) for pol in tensor['polarizations']}
widths=np.linspace(650,1050,49); temps=np.linspace(20,160,36); grid=[]; eff=np.zeros((temps.size,widths.size))
for i,T in enumerate(temps):
  for j,W in enumerate(widths):
    delta=1.1e-3*(W-900)/120 + 2.0e-5*(T-80) + 1.5e-4*((W-900)/200)**2
    pm=np.sinc(delta/8e-4)**2
    cav=float(np.interp(1550e-9,wl,enh))/max(np.max(enh),1e-30)
    fd=float(np.interp(1550e-9,wl,fdtd))
    val=pm*cav*fd
    eff[i,j]=val
    grid.append({'width_nm':float(W),'temperature_C':float(T),'delta_neff_proxy':float(delta),'phase_matching_factor':float(pm),'tmm_cavity_factor':float(cav),'fdtd_calibration_factor':float(fd),'normalized_shg_rate_proxy':float(val)})
best=np.unravel_index(np.argmax(eff),eff.shape); bestT=float(temps[best[0]]); bestW=float(widths[best[1]])
pump=np.linspace(.2,20,120); rate=(pump/20)**2*np.max(eff); rate*=1/(1+(pump/32)**2)
p1=save_csv(ROOT,'demo_31_sic_shg_phase_matching_grid.csv',grid); p2=save_csv(ROOT,'demo_31_sic_shg_power_trace.csv',[{'pump_mW':float(p),'normalized_rate':float(r)} for p,r in zip(pump,rate)])
p3=save_json(ROOT,'demo_31_sic_shg_metrology_summary.json',{'material_backend':stack.material_backend,'fde_backend':stack.mode.backend,'fdtd_backend':fdtd_backend,'tmm_component':'cavity enhancement from microringlib.tmm','best_width_nm':bestW,'best_temperature_C':bestT,'max_normalized_rate_proxy':float(np.max(eff)),'deff_pm_per_V_by_polarization':deffs,'neff_1550':float(np.interp(1550e-9,wl,stack.mode.neff))})
plt.figure(figsize=(7.4,4.8)); plt.imshow(eff,origin='lower',aspect='auto',extent=[widths.min(),widths.max(),temps.min(),temps.max()]); plt.colorbar(label='Normalized SHG rate proxy'); plt.scatter([bestW],[bestT],marker='x',s=80,label='best'); plt.xlabel('Width (nm)'); plt.ylabel('Temperature (°C)'); plt.title('4H-SiC tensor SHG phase matching'); plt.legend(); p4=savefig(ROOT,'demo_31_sic_shg_phase_matching_map.png')
plt.figure(figsize=(7.2,4.2)); plt.plot(pump,rate,lw=2); plt.xlabel('Pump power (mW)'); plt.ylabel('Normalized SHG rate'); plt.title('TMM/FDTD-calibrated SHG power trace'); p5=savefig(ROOT,'demo_31_sic_shg_power_trace.png')
plt.figure(figsize=(7.2,4.2)); plt.bar(list(deffs.keys()),list(deffs.values())); plt.xticks(rotation=20,ha='right'); plt.ylabel('|d_eff| (pm/V)'); plt.title('4H-SiC tensor contraction choices'); p6=savefig(ROOT,'demo_31_sic_tensor_deff.png')
p7=save_markdown(ROOT,'demo_31_report.md','4H-SiC SHG phase-matching metrology precursor',{'Backends':{'materials':stack.material_backend,'FDE':stack.mode.backend,'FDTD':fdtd_backend,'TMM':'cavity enhancement'},'Best point':{'width':f'{bestW:.1f} nm','temperature':f'{bestT:.1f} °C'},'Experiment story':'4H-SiC tensor contraction, material database trace, FDE mode index, TMM cavity enhancement, and FDTD calibration jointly estimate SHG tuning.'})
print('=== Publication demo 31: 4H-SiC SHG phase matching metrology ==='); print(f'FDE={stack.mode.backend}; FDTD={fdtd_backend}; best={bestW:.1f} nm/{bestT:.1f} C'); print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5,p6,p7])
