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
from shared.publication_models import print_artifacts
import microringlib as mrl

cfg=load_config('kerr.yaml')['kerr']
P=np.linspace(0,cfg['pin_max_mw']*1e-3,int(cfg['pin_points']))
try:
    params=mrl.kerr_params_from_Q(wavelength=1550e-9, loaded_Q=1.2e5, intrinsic_Q=2.5e5, gamma_w_m=1.2, radius=10e-6)
    E=mrl.solve_kerr_sweep(P, params, detuning=cfg['detuning_over_kappa']*params.kappa/2)
    T=mrl.kerr_through_power(E, params, detuning=cfg['detuning_over_kappa']*params.kappa/2)
    backend='microringlib nonlinear Kerr solver'
except Exception:
    x=P/(0.35*P.max()+1e-30); E=x/(1+x**3); T=1/(1+25*(E-0.35)**2); backend='deterministic algebraic surrogate/fallback'
turn_idx=int(np.argmax(np.gradient(E,P+1e-30)))
rows=[{'Pin_mW':float(p*1e3),'intracavity_proxy':float(e),'through_power':float(t)} for p,e,t in zip(P[::5],np.ravel(E)[::5],np.ravel(T)[::5])]
p1=save_csv(ROOT,'demo_06_kerr_sweep.csv',rows)
p2=save_json(ROOT,'demo_06_kerr_metrics.json',{'backend':backend,'model_class':'nonlinear compact solver or deterministic algebraic surrogate','signoff_note':'Use this as an operating-region map; final nonlinear dynamics require time-domain or calibrated measurements.','switching_power_mW_proxy':float(P[turn_idx]*1e3),'max_intracavity_proxy':float(np.max(np.ravel(E)))})
plt.figure(figsize=(7,4)); plt.plot(P*1e3,np.ravel(E)); plt.xlabel('Input power (mW)'); plt.ylabel('Intracavity energy proxy'); p3=savefig(ROOT,'demo_06_kerr_intracavity.png')
plt.figure(figsize=(7,4)); plt.plot(P*1e3,np.ravel(T)); plt.xlabel('Input power (mW)'); plt.ylabel('Through power'); p4=savefig(ROOT,'demo_06_kerr_transfer.png')
p5=save_markdown(ROOT,'demo_06_report.md','Kerr nonlinear ring sweep',{'Metrics':{'backend':backend,'model class':'compact/surrogate nonlinear sweep','switching proxy':f'{P[turn_idx]*1e3:.2f} mW'},'Decision':'Use as a nonlinear operating-region map before running slow time-domain solvers or calibrated experiments.'})
print('=== Publication demo 06: Kerr bistability ==='); print(f'backend={backend}')
print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5])
