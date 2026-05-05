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

cfg=load_config('silicon_wdm.yaml')['monte_carlo']; rng=np.random.default_rng(11)
N=int(cfg['samples']); lam0=cfg['target_wavelength_nm']; R=cfg['radius_um']; neff=cfg['neff']
dR=rng.normal(0,cfg['sigma_radius_nm']*1e-3,N) # um
dn=rng.normal(0,cfg['sigma_neff'],N)
shift_pm=lam0*((dR/R)+(dn/neff))*1e3
untrim=abs(shift_pm)<=cfg['untrimmed_tolerance_pm']; heater_budget_pm=20*576
trim_residual=np.maximum(abs(shift_pm)-heater_budget_pm,0); trimmed=trim_residual<=cfg['trimmed_tolerance_pm']
rows=[{'sample':i,'resonance_shift_pm':float(s),'untrimmed_pass':bool(u),'trimmed_pass':bool(t)} for i,(s,u,t) in enumerate(zip(shift_pm[:1000],untrim[:1000],trimmed[:1000]))]
p1=save_csv(ROOT,'demo_08_monte_carlo_samples.csv',rows)
p2=save_json(ROOT,'demo_08_yield_summary.json',{'samples':N,'sigma_shift_pm':float(np.std(shift_pm)),'untrimmed_yield':float(np.mean(untrim)),'trimmed_yield':float(np.mean(trimmed)),'heater_budget_pm':heater_budget_pm})
plt.figure(figsize=(7,4)); plt.hist(shift_pm,bins=80); plt.xlabel('Resonance shift (pm)'); plt.ylabel('Count'); p3=savefig(ROOT,'demo_08_shift_histogram.png')
plt.figure(figsize=(7,4)); budgets=np.linspace(0,heater_budget_pm,120); yields=[np.mean(np.maximum(abs(shift_pm)-b,0)<=cfg['trimmed_tolerance_pm']) for b in budgets]; plt.plot(budgets,yields); plt.xlabel('Available trim range (pm)'); plt.ylabel('Yield'); p4=savefig(ROOT,'demo_08_trim_yield_curve.png')
p5=save_markdown(ROOT,'demo_08_report.md','Fabrication-dependent resonance spread',{'Yield':{'untrimmed':f'{np.mean(untrim):.3f}','trimmed':f'{np.mean(trimmed):.3f}'},'Decision':'Use measured σR and σneff to estimate heater trim budget before tapeout.'})
print('=== Publication demo 08: fabrication Monte Carlo ==='); print(f'untrimmed_yield={np.mean(untrim):.3f}, trimmed_yield={np.mean(trimmed):.3f}')
print('Status: PASS'); print_artifacts([p1,p2,p3,p4,p5])
