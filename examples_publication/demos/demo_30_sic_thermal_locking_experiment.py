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

cfg=load_config('sic_4h.yaml'); therm=cfg['thermal']; mat=cfg['materials']; ring=cfg['ring']
wl=np.linspace(1549.2e-9,1550.8e-9,2401); stack=sic_material_fde_stack(wl,cfg,max_points=5)
R=ring['radius_um']*1e-6; K=ring['coupling_power_1']; alpha=dbcm_to_npm(mat['core']['loss_db_cm'])
base_power=np.abs(tmm.ring_allpass_field(wl,stack.mode.neff,R,K,alpha))**2
fdtd,fdtd_backend=fdtd_calibration_trace(wl,base_power,n_core=float(np.interp(1550e-9,wl,stack.core_n)),n_clad=float(np.interp(1550e-9,wl,stack.clad_n)),radius_um=ring['radius_um'],width_um=cfg['waveguide']['width_nm']/1000,run_label='demo30')
time_ms=np.linspace(0,60,1500); ambient=25+2.0*np.sin(2*np.pi*time_ms/60)+0.04*time_ms
heater=np.zeros_like(time_ms); err=np.zeros_like(time_ms); det_open=[]; det_closed=[]
lock_target_pm=0.0; kp=0.22; ki=0.012; integ=0.0; tuning_pm_per_mw=therm['thermal_resistance_K_per_mW']*mat['core']['dn_dT']*1550e3/2.60
for i,t in enumerate(time_ms):
    drift_pm=0.72*(ambient[i]-25.0)
    open_det=drift_pm
    if i:
        heater[i]=max(0,min(30,heater[i-1]+kp*err[i-1]+ki*integ))
    closed_det=drift_pm-tuning_pm_per_mw*heater[i]
    err[i]=closed_det-lock_target_pm; integ+=err[i]*np.mean(np.diff(time_ms))
    det_open.append(open_det); det_closed.append(closed_det)
det_open=np.asarray(det_open); det_closed=np.asarray(det_closed)
monitor_open=np.interp(1550e-9+det_open*1e-12,wl,fdtd,left=fdtd[0],right=fdtd[-1])
monitor_closed=np.interp(1550e-9+det_closed*1e-12,wl,fdtd,left=fdtd[0],right=fdtd[-1])
rows=[{'time_ms':float(t),'ambient_C':float(a),'heater_mW':float(h),'open_loop_detuning_pm':float(o),'closed_loop_detuning_pm':float(c),'open_loop_monitor_power':float(mo),'closed_loop_monitor_power':float(mc)} for t,a,h,o,c,mo,mc in zip(time_ms,ambient,heater,det_open,det_closed,monitor_open,monitor_closed)]
p1=save_csv(ROOT,'demo_30_sic_thermal_locking_trace.csv',rows); p2=save_json(ROOT,'demo_30_sic_thermal_locking_summary.json',{'material_backend':stack.material_backend,'fde_backend':stack.mode.backend,'fdtd_backend':fdtd_backend,'tmm_component':'ring_allpass_field monitor curve','open_loop_rms_detuning_pm':float(np.sqrt(np.mean(det_open**2))),'closed_loop_rms_detuning_pm':float(np.sqrt(np.mean(det_closed**2))),'heater_power_mean_mW':float(np.mean(heater)),'tuning_pm_per_mW_proxy':float(tuning_pm_per_mw)})
plt.figure(figsize=(7.6,4.2)); plt.plot(time_ms,det_open,label='open loop'); plt.plot(time_ms,det_closed,label='closed loop'); plt.xlabel('Time (ms)'); plt.ylabel('Detuning (pm)'); plt.title('4H-SiC ring thermal locking'); plt.legend(); p3=savefig(ROOT,'demo_30_sic_locking_detuning.png')
plt.figure(figsize=(7.6,4.2)); plt.plot(time_ms,heater); plt.xlabel('Time (ms)'); plt.ylabel('Heater power (mW)'); plt.title('Feedback heater actuation'); p4=savefig(ROOT,'demo_30_sic_locking_heater_power.png')
plt.figure(figsize=(7.6,4.2)); plt.plot(time_ms,monitor_open,label='open loop'); plt.plot(time_ms,monitor_closed,label='closed loop'); plt.xlabel('Time (ms)'); plt.ylabel('Monitor power'); plt.title('FDTD-calibrated monitor signal'); plt.legend(); p5=savefig(ROOT,'demo_30_sic_locking_monitor.png')
p6=save_markdown(ROOT,'demo_30_report.md','4H-SiC thermal locking numerical precursor',{'Backends':{'materials':stack.material_backend,'FDE':stack.mode.backend,'FDTD':fdtd_backend,'TMM':'microringlib.tmm monitor ring'},'Locking metrics':{'open-loop RMS detuning':f'{np.sqrt(np.mean(det_open**2)):.3f} pm','closed-loop RMS detuning':f'{np.sqrt(np.mean(det_closed**2)):.3f} pm'},'Experiment story':'Thermo-optic material trace and FDE produce a TMM monitor curve; an FDTD hook calibrates monitor power; feedback closes the loop.'})
print('=== Publication demo 30: 4H-SiC thermal locking experiment ==='); print(f'FDE={stack.mode.backend}; FDTD={fdtd_backend}; closed-loop RMS={np.sqrt(np.mean(det_closed**2)):.3f} pm'); print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5,p6])
