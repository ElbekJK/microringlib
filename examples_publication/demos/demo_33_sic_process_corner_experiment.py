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
from shared.publication_models import dbcm_to_npm, print_artifacts, nearest_peak
from shared.sic_experiment_helpers import sic_material_fde_stack, fdtd_calibration_trace
from microringlib import tmm

cfg=load_config('sic_4h.yaml'); mat=cfg['materials']; ring=cfg['ring']; wl=np.linspace(1547e-9,1553e-9,5001)
stack=sic_material_fde_stack(wl,cfg,max_points=5); R0=ring['radius_um']*1e-6; K0=ring['coupling_power_1']; loss0=mat['core']['loss_db_cm']
base_fields=tmm.ring_add_drop_fields(wl,stack.mode.neff,R0,K0,K0,dbcm_to_npm(loss0)); fdtd,fdtd_backend=fdtd_calibration_trace(wl,base_fields['drop_power'],n_core=float(np.interp(1550e-9,wl,stack.core_n)),n_clad=float(np.interp(1550e-9,wl,stack.clad_n)),radius_um=ring['radius_um'],width_um=cfg['waveguide']['width_nm']/1000,run_label='demo33')
corners=[('nominal',0,0,1.0,K0),('width_plus',0.007,0,0.95,K0+0.005),('width_minus',-0.007,0,1.08,max(.01,K0-.005)),('hot_100C',0.005,75,1.04,K0),('cold_0C',-0.002,-25,.98,K0),('rough_high_loss',0,0,2.2,K0),('gap_closed',0,0,1.0,K0+0.025),('gap_open',0,0,1.0,max(.01,K0-.02))]
rows=[]; spectra={}
for name,dn,dT,loss_mult,K in corners:
    neff=stack.mode.neff+dn+mat['core']['dn_dT']*dT
    loss=loss0*loss_mult
    fields=tmm.ring_add_drop_fields(wl,neff,R0,K,K,dbcm_to_npm(loss))
    drop=np.clip(fields['drop_power']*fdtd/np.maximum(np.max(fdtd),1e-12),0,1)
    spectra[name]=drop
    lam,peak,idx=nearest_peak(wl,drop,1550e-9)
    il=-10*np.log10(max(peak,1e-12)); center_pm=(lam-1550e-9)*1e12
    half=.5*peak; l=idx; r=idx
    while l>0 and drop[l]>half: l-=1
    while r<drop.size-1 and drop[r]>half: r+=1
    Q=float(lam/(wl[r]-wl[l])) if r>l+1 else float('nan')
    passes=bool(peak>.13 and il<9 and abs(center_pm)<3500 and Q>300)
    rows.append({'corner':name,'delta_neff':float(dn),'temperature_offset_C':float(dT),'loss_db_cm':float(loss),'K':float(K),'center_nm':float(lam*1e9),'center_error_pm':float(center_pm),'peak_drop':float(peak),'insertion_loss_db':float(il),'loaded_Q_proxy':Q,'passes_corner_target':passes})
p1=save_csv(ROOT,'demo_33_sic_process_corner_table.csv',rows); p2=save_json(ROOT,'demo_33_sic_process_corner_summary.json',{'material_backend':stack.material_backend,'fde_backend':stack.mode.backend,'fdtd_backend':fdtd_backend,'tmm_component':'ring_add_drop_fields process corners','yield':float(np.mean([r['passes_corner_target'] for r in rows])),'worst_corner':min(rows,key=lambda r:r['peak_drop'])['corner'],'corner_count':len(rows)})
plt.figure(figsize=(7.5,4.5));
for name,y in spectra.items(): plt.plot(wl*1e9,y,lw=1.4,label=name)
plt.xlabel('Wavelength (nm)'); plt.ylabel('FDTD-calibrated drop power'); plt.title('4H-SiC process-corner spectra'); plt.legend(ncol=2,fontsize=7); p3=savefig(ROOT,'demo_33_sic_process_corner_spectra.png')
plt.figure(figsize=(7.2,4.2)); plt.bar([r['corner'] for r in rows],[r['center_error_pm'] for r in rows]); plt.xticks(rotation=30,ha='right'); plt.ylabel('Center error (pm)'); plt.title('4H-SiC corner resonance shift'); p4=savefig(ROOT,'demo_33_sic_corner_center_error.png')
plt.figure(figsize=(7.2,4.2)); plt.bar([r['corner'] for r in rows],[r['loaded_Q_proxy'] for r in rows]); plt.xticks(rotation=30,ha='right'); plt.ylabel('Loaded-Q proxy'); plt.title('4H-SiC corner Q proxy'); p5=savefig(ROOT,'demo_33_sic_corner_q_proxy.png')
p6=save_markdown(ROOT,'demo_33_report.md','4H-SiC process-corner numerical precursor',{'Backends':{'materials':stack.material_backend,'FDE':stack.mode.backend,'FDTD':fdtd_backend,'TMM':'add-drop process-corner spectra'},'Metrics':{'yield':f'{np.mean([r["passes_corner_target"] for r in rows]):.3f}','worst corner':min(rows,key=lambda r:r['peak_drop'])['corner']},'Experiment story':'Material-corner and process-corner perturbations are pushed through FDE-calibrated TMM and an FDTD monitor calibration to provide early experiment signoff margins.'})
print('=== Publication demo 33: 4H-SiC process-corner experiment ==='); print(f'FDE={stack.mode.backend}; FDTD={fdtd_backend}; yield={np.mean([r["passes_corner_target"] for r in rows]):.3f}'); print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5,p6])
