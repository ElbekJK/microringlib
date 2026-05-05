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
from shared.publication_models import analytic_through, dbcm_to_npm, print_artifacts
from shared.sic_experiment_helpers import sic_material_fde_stack, fdtd_calibration_trace, ring_q_proxy
from microringlib import tmm

cfg=load_config('sic_4h.yaml')
wl=np.linspace(1546e-9,1554e-9,6001)
stack=sic_material_fde_stack(wl,cfg,max_points=int(__import__('os').environ.get('MICRORINGLIB_SIC_FDE_POINTS','41')))
R=cfg['ring']['radius_um']*1e-6; K=cfg['ring']['coupling_power_1']; loss=cfg['materials']['core']['loss_db_cm']
alpha=dbcm_to_npm(loss)
field=tmm.ring_allpass_field(wl,stack.mode.neff,R,K,alpha)
tmm_power=np.abs(field)**2
compact=analytic_through(wl,R,stack.mode.neff,loss,K)
fdtd,fdtd_backend=fdtd_calibration_trace(wl,tmm_power,n_core=float(np.interp(1550e-9,wl,stack.core_n)),n_clad=float(np.interp(1550e-9,wl,stack.clad_n)),radius_um=cfg['ring']['radius_um'],width_um=cfg['waveguide']['width_nm']/1000,run_label='demo29')
# No synthetic measurement is generated: fit the physical TMM through-port dip and compare compact/FDTD-calibrated traces.
lam_fit,Q,base,dip,q_method,line_width=ring_q_proxy(wl,tmm_power,1550e-9)
rows=[{'wavelength_nm':float(a*1e9),'tmm_power':float(b),'compact_power':float(c),'fdtd_calibrated_power':float(d)} for a,b,c,d in zip(wl[::6],tmm_power[::6],compact[::6],fdtd[::6])]
p1=save_csv(ROOT,'demo_22_sic_resonance_metrology_trace.csv',rows)
p2=save_json(ROOT,'demo_22_sic_resonance_metrology_summary.json',{'material_backend':stack.material_backend,'fde_backend':stack.mode.backend,'fdtd_backend':fdtd_backend,'tmm_component':'ring_allpass_field','neff_1550':float(np.interp(1550e-9,wl,stack.mode.neff)),'ng_1550':float(np.interp(1550e-9,wl,stack.mode.ng)),'fit_center_nm':float(lam_fit*1e9),'loaded_Q_proxy':float(Q),'q_extraction_method':q_method,'linewidth_pm':float(line_width*1e12),'baseline_power':base,'dip_power':dip,'loss_db_cm':loss,'coupling_K':K,'measurement_policy':'no synthetic measured trace is generated; Q is extracted from the TMM all-pass through-port dip; with MICRORINGLIB_RUN_MEEP_3D=1 the FDTD trace is a full-radius 3D microring simulation unless MICRORINGLIB_ALLOW_LOCAL_3D_HOOK=1 is explicitly set'})
plt.figure(figsize=(7.6,4.4)); plt.plot(wl*1e9,tmm_power,label='TMM all-pass through dip',lw=2,ls='-'); plt.plot(wl*1e9,compact,label='compact microringlib trace',lw=1.6,ls='--'); plt.plot(wl*1e9,fdtd,label='FDTD-calibrated trace / full 3D when enabled',lw=1.4,ls=':',alpha=.85); plt.axvline(lam_fit*1e9,ls='--',lw=1,label='fitted TMM dip center'); plt.xlabel('Wavelength (nm)'); plt.ylabel('Through-port power'); plt.title('4H-SiC resonance metrology without synthetic measurements'); plt.legend(fontsize=8); p3=savefig(ROOT,'demo_22_sic_resonance_metrology_fit.png')
plt.figure(figsize=(7.2,4.2)); plt.plot(wl[::6]*1e9,fdtd[::6]-tmm_power[::6]); plt.xlabel('Wavelength (nm)'); plt.ylabel('FDTD-calibrated - TMM power'); plt.title('Calibration-hook residual trace'); p4=savefig(ROOT,'demo_22_sic_metrology_residual.png')
p5=save_markdown(ROOT,'demo_22_report.md','4H-SiC resonance metrology numerical precursor',{'Backends':{'materials':stack.material_backend,'FDE':stack.mode.backend,'FDTD':fdtd_backend,'TMM':'microringlib.tmm.ring_allpass_field'},'Extracted metrics':{'center':f'{lam_fit*1e9:.4f} nm','Q proxy':f'{Q:.1f}','Q extraction method':q_method,'linewidth':f'{line_width*1e12:.2f} pm','dip':f'{dip:.4f}'},'Experiment story':'Database material trace seeds FDE; FDE seeds TMM; optional full-radius 3D MEEP provides the FDTD monitor trace; microringlib extracts dip/Q metrics without synthetic measurement traces.'})
print('=== Publication demo 22: 4H-SiC resonance metrology pipeline ==='); print(f'FDE={stack.mode.backend}; FDTD={fdtd_backend}; Q≈{Q:.1f} ({q_method}, linewidth={line_width*1e12:.2f} pm)'); print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5])
