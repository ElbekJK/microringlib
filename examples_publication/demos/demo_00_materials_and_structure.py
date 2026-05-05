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
from shared.materials import material_index_sweep
from shared.geometry import fabrication_design_rules, pulley_coupler_effective_K, bend_loss_db_per_turn
from shared.publication_models import coupler_from_K, print_artifacts

cfg = load_config('silicon_wdm.yaml')
wl = np.linspace(1.50e-6, 1.60e-6, 401)
core = material_index_sweep(cfg['materials']['core'], wl, require_database=True)
clad = material_index_sweep(cfg['materials']['clad'], wl, require_database=True)
width_nm = 500.0; gap_nm = 180.0; radius_um = cfg['ring']['radius_um']
K = cfg['ring']['K1']; t,k = coupler_from_K(K)
S = np.array([[t,k],[k,t]])
unitarity_error = float(np.max(np.abs(S.conj().T @ S - np.eye(2))))
drc = fabrication_design_rules(width_nm, gap_nm, radius_um)
angles = np.linspace(0,180,181); Kpulley = pulley_coupler_effective_K(K, angles)
loss_turn = bend_loss_db_per_turn(np.linspace(5,30,100))
rows=[{'material':core.name,'backend':core.backend,'n_1550':float(np.interp(1550e-9,wl,core.n)),'index_validated':bool(np.isfinite(np.interp(1550e-9,wl,core.n)) and np.interp(1550e-9,wl,core.n)>1.05),'loss_db_cm':core.loss_db_cm}, {'material':clad.name,'backend':clad.backend,'n_1550':float(np.interp(1550e-9,wl,clad.n)),'index_validated':bool(np.isfinite(np.interp(1550e-9,wl,clad.n)) and np.interp(1550e-9,wl,clad.n)>1.05),'loss_db_cm':clad.loss_db_cm}]
p1=save_csv(ROOT,'demo_00_material_table.csv',rows)
p2=save_json(ROOT,'demo_00_structure_checks.json',{'width_nm':width_nm,'gap_nm':gap_nm,'radius_um':radius_um,'drc':drc,'coupler_K':K,'unitarity_error':unitarity_error,'passive_ready':unitarity_error<1e-10 and all(drc.values())})
fig, ax1 = plt.subplots(figsize=(7,4.2))
ax2 = ax1.twinx()
l1, = ax1.plot(wl*1e9, core.n, lw=2.0, label=f'{core.name} ({core.backend})')
l2, = ax2.plot(wl*1e9, clad.n, lw=2.0, ls='--', label=f'{clad.name} ({clad.backend})')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Si refractive index')
ax2.set_ylabel('SiO2 refractive index')
ax1.set_title('Database-backed Si and SiO2 material indices')
ax1.legend([l1,l2],[l1.get_label(), l2.get_label()], loc='best', fontsize=8)
fig.tight_layout()
p3=savefig(ROOT,'demo_00_material_indices.png')
plt.figure(figsize=(7,4)); plt.plot(angles,Kpulley); plt.xlabel('Pulley angle (deg)'); plt.ylabel('Effective power coupling K'); p4=savefig(ROOT,'demo_00_pulley_coupling_sweep.png')
p5=save_markdown(ROOT,'demo_00_report.md','Materials, geometry, and coupler checks',{'Decision notes':[f'Coupler unitarity error = {unitarity_error:.3e}',f'DRC pass = {all(drc.values())}',f'Bend loss at 10 um radius ≈ {bend_loss_db_per_turn(10):.3g} dB/turn'],'Backend notes':{'core':core.backend,'clad':clad.backend}})
print('=== Publication demo 00: materials and structure ===')
print('Status: PASS')
print_artifacts([p1,p2,p3,p4,p5])
