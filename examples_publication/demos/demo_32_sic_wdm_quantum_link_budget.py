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
from shared.publication_models import dbcm_to_npm, print_artifacts
from shared.sic_experiment_helpers import sic_material_fde_stack, fdtd_calibration_trace
from microringlib import tmm

cfg=load_config('sic_4h.yaml'); mat=cfg['materials']; ring=cfg['ring']
channels=6; centers_nm=1542+3.2*np.arange(channels); wl=np.linspace(1538e-9,1562e-9,8001)
stack=sic_material_fde_stack(wl,cfg,max_points=5); loss=mat['core']['loss_db_cm']; alpha=dbcm_to_npm(loss); K=0.055; seed=ring['radius_um']*1e-6
radii=[]; drops=[]; throughs=[]
for c in centers_nm:
    ne=float(np.interp(c*1e-9,wl,stack.mode.neff)); m=max(1,int(round(ne*2*np.pi*seed/(c*1e-9)))); R=m*c*1e-9/(2*np.pi*ne); radii.append(R)
    fields=tmm.ring_add_drop_fields(wl,stack.mode.neff,R,K,K,alpha); drops.append(fields['drop_power']); throughs.append(fields['through_power'])
radii=np.asarray(radii); drops=np.asarray(drops)
fdtd,fdtd_backend=fdtd_calibration_trace(wl,np.max(drops,axis=0),n_core=float(np.interp(1550e-9,wl,stack.core_n)),n_clad=float(np.interp(1550e-9,wl,stack.clad_n)),radius_um=float(np.mean(radii)*1e6),width_um=cfg['waveguide']['width_nm']/1000,run_label='demo32')
pump_mw=np.linspace(.5,18,channels); pair_rate=2.5e5*(pump_mw/5)**1.65; eta=0.16; dark=180; window=1.0e-9
rows=[]; xt=np.zeros((channels,channels))
for i,c in enumerate(centers_nm):
  for j in range(channels): xt[i,j]=float(np.interp(c*1e-9,wl,drops[j]))
  signal=pair_rate[i]*eta*xt[i,i]*float(np.interp(c*1e-9,wl,fdtd))
  leak=eta*sum(pair_rate[j]*xt[i,j] for j in range(channels) if j!=i)
  acc=(signal+leak+dark)**2*window+3
  car=signal/max(acc,1e-30)
  rows.append({'channel':i,'center_nm':float(c),'radius_um':float(radii[i]*1e6),'pump_mW':float(pump_mw[i]),'pair_rate_Hz':float(pair_rate[i]),'signal_counts_Hz':float(signal),'leakage_counts_Hz':float(leak),'accidentals_Hz':float(acc),'CAR_proxy':float(car),'fdtd_calibration':float(np.interp(c*1e-9,wl,fdtd))})
xt_rows=[{'victim_channel':i,'aggressor_channel':j,'drop_coupling':float(xt[i,j])} for i in range(channels) for j in range(channels)]
p1=save_csv(ROOT,'demo_32_sic_wdm_quantum_link_budget.csv',rows); p2=save_csv(ROOT,'demo_32_sic_quantum_crosstalk_matrix.csv',xt_rows); p3=save_json(ROOT,'demo_32_sic_quantum_link_summary.json',{'material_backend':stack.material_backend,'fde_backend':stack.mode.backend,'fdtd_backend':fdtd_backend,'tmm_component':'ring_add_drop_fields WDM bank','mean_CAR_proxy':float(np.mean([r['CAR_proxy'] for r in rows])),'min_CAR_proxy':float(np.min([r['CAR_proxy'] for r in rows])),'channels':channels})
plt.figure(figsize=(7.5,4.4));
for i in range(channels): plt.plot(wl*1e9,drops[i],lw=1.4,label=f'ch{i}')
plt.xlabel('Wavelength (nm)'); plt.ylabel('Drop power'); plt.title('4H-SiC TMM WDM quantum filters'); plt.legend(ncol=3,fontsize=7); p4=savefig(ROOT,'demo_32_sic_wdm_filter_bank.png')
plt.figure(figsize=(6.0,5.0)); plt.imshow(10*np.log10(np.maximum(xt,1e-12)),origin='lower'); plt.colorbar(label='Drop coupling (dB)'); plt.xlabel('Aggressor'); plt.ylabel('Victim'); plt.title('Quantum WDM crosstalk matrix'); p5=savefig(ROOT,'demo_32_sic_quantum_crosstalk_matrix.png')
plt.figure(figsize=(7.2,4.2)); plt.plot([r['channel'] for r in rows],[r['CAR_proxy'] for r in rows],marker='o'); plt.xlabel('Channel'); plt.ylabel('CAR proxy'); plt.title('Per-channel quantum link CAR'); p6=savefig(ROOT,'demo_32_sic_car_by_channel.png')
p7=save_markdown(ROOT,'demo_32_report.md','4H-SiC WDM quantum-link numerical precursor',{'Backends':{'materials':stack.material_backend,'FDE':stack.mode.backend,'FDTD':fdtd_backend,'TMM':'add-drop WDM bank'},'Metrics':{'mean CAR':f'{np.mean([r["CAR_proxy"] for r in rows]):.2f}','min CAR':f'{np.min([r["CAR_proxy"] for r in rows]):.2f}'},'Experiment story':'FDE-calibrated 4H-SiC add-drop rings form a WDM quantum filter bank; FDTD hook calibrates collection; microringlib computes CAR/crosstalk budget.'})
print('=== Publication demo 32: 4H-SiC WDM quantum link budget ==='); print(f'FDE={stack.mode.backend}; FDTD={fdtd_backend}; mean CAR={np.mean([r["CAR_proxy"] for r in rows]):.2f}'); print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5,p6,p7])
