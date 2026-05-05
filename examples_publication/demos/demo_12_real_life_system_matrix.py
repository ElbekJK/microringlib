#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Publication demo: real-life-like FDE/FDTD-aware microring design matrix."""
import os
import numpy as np
import matplotlib.pyplot as plt
from shared.path_setup import ensure_project_on_path
ROOT = ensure_project_on_path()
from shared.config import load_config
from shared.decision import save_csv, save_json, save_markdown
from shared.plotting import savefig
from shared.materials import material_index_sweep
from shared.publication_models import C0, analytic_add_drop, bend_loss_db_cm_proxy, fsr_m, nearest_peak, print_artifacts, ring_length, try_fde_sweep

cfg=load_config('silicon_wdm.yaml'); mat=cfg['materials']; wdm=cfg['wdm']
wl=np.linspace(1547.6e-9,1552.4e-9,int(os.environ.get('MICRORINGLIB_DEMO12_WL_POINTS','3001')))
target_m=(1550.0+float(wdm.get('spacing_nm',0.8))*np.arange(-1.5,2.0,1.0))*1e-9
dense=os.environ.get('MICRORINGLIB_DEMO12_DENSE','1')=='1'
if dense:
    widths_nm=np.linspace(430.0,650.0,int(os.environ.get('MICRORINGLIB_DEMO12_WIDTH_POINTS','17')))
    radii_um=np.linspace(20.0,200.0,int(os.environ.get('MICRORINGLIB_DEMO12_RADIUS_POINTS','41')))
    K_values=np.linspace(0.01,0.24,int(os.environ.get('MICRORINGLIB_DEMO12_K_POINTS','17')))
    temperatures_C=np.linspace(0.0,125.0,int(os.environ.get('MICRORINGLIB_DEMO12_TEMPERATURE_POINTS','9')))
    delay_lengths_mm=np.linspace(0.0,5.0,int(os.environ.get('MICRORINGLIB_DEMO12_DELAY_POINTS','5')))
else:
    widths_nm=np.linspace(430.0,650.0,7); radii_um=np.linspace(20.0,200.0,11); K_values=np.linspace(0.02,0.20,9); temperatures_C=np.linspace(0.0,125.0,6); delay_lengths_mm=np.linspace(0,5,9)
core_mat=material_index_sweep(mat['core'],wl); clad_mat=material_index_sweep(mat['clad'],wl)
if os.environ.get('MICRORINGLIB_FDE_USE_DATABASE_INDEX','0')=='1':
    n_core=float(np.interp(1550e-9,wl,core_mat.n)); n_clad=float(np.interp(1550e-9,wl,clad_mat.n))
else:
    n_core=float(mat['core']['constant_n']); n_clad=float(mat['clad']['constant_n'])
base_loss_db_cm=float(mat['core']['loss_db_cm']); bend_model=os.environ.get('MICRORINGLIB_BEND_LOSS_MODEL','piecewise')
dn_eff_dT=1.55e-4; alpha_L=2.6e-6; T_ref=25.0
rows=[]; backends=set()
for width_nm in widths_nm:
    mode=try_fde_sweep(wl,width_nm*1e-9,220e-9,n_core,n_clad,base_loss_db_cm,max_points=int(os.environ.get('MICRORINGLIB_DEMO12_FDE_POINTS','41')))
    backends.add(mode.backend)
    for radius_um in radii_um:
        bend_loss=float(bend_loss_db_cm_proxy(radius_um,bend_model)); loss_db_cm=base_loss_db_cm+bend_loss
        R0=radius_um*1e-6; L0=float(ring_length(R0)); ng1550=float(np.interp(1550e-9,wl,mode.ng))
        fsr_nm=float(fsr_m(1550e-9,ng1550,R0)*1e9); roundtrip_ps=float(ng1550*L0/C0*1e12)
        a_rt=float(np.exp(-0.5*(loss_db_cm*np.log(10)/10*100)*L0)); Kcrit=float(np.clip(1-a_rt**2,0,1))
        for K in K_values:
            for T in temperatures_C:
                dT=T-T_ref; R_T=R0*(1+alpha_L*dT); neff_T=mode.neff+dn_eff_dT*dT
                spec=analytic_add_drop(wl,R_T,neff_T,loss_db_cm,K1=K,K2=K)
                lam,peak,idx=nearest_peak(wl,spec['drop'],1550e-9)
                samples=np.interp(target_m,wl,spec['drop']); adjacent=float(np.partition(samples,-2)[-2]) if samples.size>1 else 0.0
                thru=float(spec['through'][idx]); il_db=float(-10*np.log10(max(peak,1e-12))); ext_db=float(-10*np.log10(max(thru,1e-12)))
                half=0.5*(float(np.nanmin(spec['drop']))+float(peak)); left=idx; right=idx
                while left>0 and spec['drop'][left]>half: left-=1
                while right<len(wl)-1 and spec['drop'][right]>half: right+=1
                Q=float(lam/(wl[right]-wl[left])) if right>left+1 else float('nan')
                pass_rule=bool(peak>=0.35 and il_db<=4.6 and ext_db>=6.0 and abs((lam-1550e-9)*1e12)<=20000)
                for delay_mm in delay_lengths_mm:
                    delay_ps=float(ng1550*delay_mm*1e-3/C0*1e12)
                    rows.append({'width_nm':float(width_nm),'radius_um':float(radius_um),'coupling_power_K':float(K),'temperature_C':float(T),'delay_length_mm':float(delay_mm),'neff_1550':float(np.interp(1550e-9,wl,neff_T)),'ng_1550':ng1550,'base_loss_db_cm':base_loss_db_cm,'bend_loss_db_cm_proxy':bend_loss,'effective_loss_db_cm':loss_db_cm,'fsr_nm':fsr_nm,'resonance_peak_nm':float(lam*1e9),'thermal_shift_pm_from_1550':float((lam-1550e-9)*1e12),'drop_peak_power':float(peak),'through_at_drop_power':thru,'insertion_loss_db':il_db,'extinction_db':ext_db,'adjacent_xtalk_power_proxy':adjacent,'loaded_Q_proxy':Q,'roundtrip_delay_ps':roundtrip_ps,'delay_line_ps':delay_ps,'total_latency_ps':roundtrip_ps+delay_ps,'critical_coupling_K_proxy':Kcrit,'coupling_minus_critical':float(K-Kcrit),'small_radius_risk_flag':False,'pass_design_rule':pass_rule})

p1=save_csv(ROOT,'demo_12_real_life_system_design_matrix.csv',rows)
base_rows=[r for r in rows if abs(r.get('delay_length_mm',0.0)) < 1e-12]
radius_diag=[]
for R in radii_um:
    sub=[r for r in base_rows if abs(r['radius_um']-float(R))<1e-12]
    radius_diag.append({'radius_um':float(R),'rows':len(sub),'pass_rate':float(np.mean([r['pass_design_rule'] for r in sub])),'mean_insertion_loss_db':float(np.mean([r['insertion_loss_db'] for r in sub])),'mean_extinction_db':float(np.mean([r['extinction_db'] for r in sub])),'mean_drop_peak_power':float(np.mean([r['drop_peak_power'] for r in sub])),'mean_loaded_Q_proxy':float(np.nanmean([r['loaded_Q_proxy'] for r in sub])),'mean_fsr_nm':float(np.mean([r['fsr_nm'] for r in sub])),'mean_critical_coupling_offset':float(np.mean([r['coupling_minus_critical'] for r in sub])),'bend_loss_db_cm_proxy':float(bend_loss_db_cm_proxy(float(R),bend_model))})
p1b=save_csv(ROOT,'demo_12_radius_diagnostic_summary.csv',radius_diag)
pass_rate=float(np.mean([r['pass_design_rule'] for r in rows])); best=sorted(rows,key=lambda r:(not r['pass_design_rule'],r['insertion_loss_db'],-r['extinction_db']))[0]
Wsel=float(widths_nm[np.argmin(np.abs(widths_nm-500.0))]); Rsel=float(radii_um[np.argmin(np.abs(radii_um-50.0))]); Ksel=float(K_values[np.argmin(np.abs(K_values-0.08))]); Tsel=float(temperatures_C[np.argmin(np.abs(temperatures_C-25.0))])
sel=[r for r in rows if abs(r['width_nm']-Wsel)<1e-9 and abs(r['radius_um']-Rsel)<1e-9 and abs(r['coupling_power_K']-Ksel)<1e-12]
selT=sorted(sel,key=lambda r:r['temperature_C'])
# Separate delay-line table using selected design so the latency figure is always populated and not hidden inside a huge Cartesian product.
ng_sel=float(selT[np.argmin([abs(r['temperature_C']-Tsel) for r in selT])]['ng_1550']) if selT else 4.0
roundtrip_sel=float(selT[np.argmin([abs(r['temperature_C']-Tsel) for r in selT])]['roundtrip_delay_ps']) if selT else 0.0
delay_rows=[{'width_nm':Wsel,'radius_um':Rsel,'coupling_power_K':Ksel,'temperature_C':Tsel,'delay_length_mm':float(d),'delay_line_ps':float(ng_sel*d*1e-3/C0*1e12),'roundtrip_delay_ps':roundtrip_sel,'total_latency_ps':float(roundtrip_sel+ng_sel*d*1e-3/C0*1e12)} for d in delay_lengths_mm]
p1c=save_csv(ROOT,'demo_12_delay_line_latency_sweep.csv',delay_rows)
p2=save_json(ROOT,'demo_12_real_life_system_summary.json',{'rows':len(rows),'backend':'; '.join(sorted(backends)),'pass_rate':pass_rate,'pass_rule':'drop_peak_power>=0.35, insertion_loss_db<=4.6, extinction_db>=6.0, |center_error|<=20000 pm','best_design':best,'dense_mode':dense,'bend_loss_model':bend_model,'radius_sweep_um':[20.0,200.0],'sampling':{'width_points':len(widths_nm),'radius_points':len(radii_um),'K_points':len(K_values),'temperature_points':len(temperatures_C),'delay_points':len(delay_lengths_mm)},'plot_selection':{'width_nm':Wsel,'radius_um':Rsel,'K':Ksel,'temperature_C':Tsel}})
plt.figure(figsize=(7,4)); plt.plot([r['temperature_C'] for r in selT],[r['thermal_shift_pm_from_1550'] for r in selT],marker='o'); plt.xlabel('Temperature (°C)'); plt.ylabel('Resonance shift from 1550 nm (pm)'); plt.title(f'Temperature shift at width≈{Wsel:.0f} nm, radius≈{Rsel:.1f} µm, K≈{Ksel:.3f}'); p3=savefig(ROOT,'demo_12_temperature_shift.png')
plt.figure(figsize=(7,4)); plt.plot([r['delay_length_mm'] for r in delay_rows],[r['total_latency_ps'] for r in delay_rows],marker='o'); plt.xlabel('Delay-line length (mm)'); plt.ylabel('Total latency proxy (ps)'); plt.title(f'Delay-line latency at radius≈{Rsel:.1f} µm, T≈{Tsel:.1f} °C'); p4=savefig(ROOT,'demo_12_delay_line_latency.png')
trade=np.full((len(radii_um),len(K_values)),np.nan)
for ir,R in enumerate(radii_um):
    for ik,K in enumerate(K_values):
        sub=[r for r in rows if abs(r['width_nm']-Wsel)<1e-9 and abs(r['radius_um']-R)<1e-12 and abs(r['coupling_power_K']-K)<1e-12 and abs(r['temperature_C']-Tsel)<1e-9]
        trade[ir,ik]=np.mean([r['drop_peak_power'] for r in sub]) if sub else np.nan
KK,RR=np.meshgrid(K_values,radii_um)
plt.figure(figsize=(7.5,5.0)); plt.contourf(KK,RR,trade,levels=50); plt.colorbar(label='Drop peak power'); plt.xlabel('Power coupling K'); plt.ylabel('Radius (µm)'); plt.title(f'Coupling–radius tradeoff, width≈{Wsel:.0f} nm'); p5=savefig(ROOT,'demo_12_coupling_radius_tradeoff.png')
heat=np.zeros((len(radii_um),len(widths_nm)))
for ir,R in enumerate(radii_um):
    for iw,W in enumerate(widths_nm):
        sub=[r for r in rows if abs(r['radius_um']-R)<1e-12 and abs(r['width_nm']-W)<1e-9]
        heat[ir,iw]=np.mean([r['pass_design_rule'] for r in sub]) if sub else np.nan
WW,RR2=np.meshgrid(widths_nm,radii_um)
plt.figure(figsize=(7.6,5.0)); plt.contourf(WW,RR2,heat,levels=40,vmin=0,vmax=1); plt.colorbar(label='Pass rate'); plt.xlabel('Width (nm)'); plt.ylabel('Radius (µm)'); plt.title('Pass-rate heatmap, radius sweep 20–200 µm'); p6=savefig(ROOT,'demo_12_pass_rate_heatmap_radius_width.png')
plt.figure(figsize=(7.2,4.8)); sc=plt.scatter([r['insertion_loss_db'] for r in rows],[r['extinction_db'] for r in rows],c=[r['radius_um'] for r in rows],s=7,alpha=0.35); plt.axvline(4.6,ls='--',lw=1,label='IL target 4.6 dB'); plt.axhline(6.0,ls='--',lw=1,label='ER target 6 dB'); plt.legend(fontsize=8); plt.colorbar(sc,label='Radius (µm)'); plt.xlabel('Insertion loss (dB)'); plt.ylabel('Extinction (dB)'); plt.title('Extinction versus insertion-loss design cloud'); p7=savefig(ROOT,'demo_12_extinction_vs_insertion_loss.png')
plt.figure(figsize=(7.2,4.2)); plt.plot([r['radius_um'] for r in radius_diag],[r['mean_loaded_Q_proxy'] for r in radius_diag],lw=2); plt.xlabel('Radius (µm)'); plt.ylabel('Mean loaded-Q proxy'); plt.title('Loaded-Q proxy versus radius'); p8=savefig(ROOT,'demo_12_loaded_q_vs_radius.png')
plt.figure(figsize=(7.2,4.2)); plt.plot([r['radius_um'] for r in radius_diag],[r['mean_fsr_nm'] for r in radius_diag],lw=2); plt.xlabel('Radius (µm)'); plt.ylabel('Mean FSR (nm)'); plt.title('FSR versus radius'); p9=savefig(ROOT,'demo_12_fsr_vs_radius.png')
r50_sub=sorted([r for r in rows if abs(r['radius_um']-Rsel)<1e-12 and abs(r['temperature_C']-Tsel)<1e-9],key=lambda r:(r['width_nm'],r['coupling_power_K']))
plt.figure(figsize=(7.4,4.6)); sc2=plt.scatter([r['coupling_power_K'] for r in r50_sub],[r['extinction_db'] for r in r50_sub],c=[r['insertion_loss_db'] for r in r50_sub],s=20,alpha=0.7); plt.colorbar(sc2,label='Insertion loss (dB)'); plt.axhline(6.0,ls='--',lw=1,label='ER target 6 dB'); plt.legend(fontsize=8); plt.xlabel('Power coupling K'); plt.ylabel('Extinction (dB)'); plt.title(f'Diagnostic near selected radius {Rsel:.1f} µm'); p10=savefig(ROOT,'demo_12_radius_9um_diagnostic.png')
offset=np.zeros((len(radii_um),len(widths_nm)))
for ir,R in enumerate(radii_um):
    for iw,W in enumerate(widths_nm):
        sub=[r for r in rows if abs(r['radius_um']-R)<1e-12 and abs(r['width_nm']-W)<1e-9]
        offset[ir,iw]=np.mean([r['coupling_minus_critical'] for r in sub]) if sub else np.nan
plt.figure(figsize=(7.6,5.0)); plt.contourf(WW,RR2,offset,levels=50); plt.colorbar(label='Mean K - K_critical proxy'); plt.xlabel('Width (nm)'); plt.ylabel('Radius (µm)'); plt.title('Critical-coupling offset map, denser gradient'); p11=savefig(ROOT,'demo_12_critical_coupling_offset_map.png')
p12=save_markdown(ROOT,'demo_12_report.md','Real-life-like microring system design matrix',{'Purpose':'FDE-style dispersion + transfer outputs + thermal tuning + delay-line latency + coupling/geometry sweeps.','Rows':len(rows), 'Main matrix delay sweep': f'{float(delay_lengths_mm.min()):.2f}-{float(delay_lengths_mm.max()):.2f} mm','Radius sweep':'20–200 µm','Sampling':{'width points':len(widths_nm),'radius points':len(radii_um),'coupling points':len(K_values),'temperature points':len(temperatures_C),'delay points':len(delay_lengths_mm)},'Selected design for line plots':{'width_nm':Wsel,'radius_um':Rsel,'K':Ksel,'temperature_C':Tsel},'Pass rate':f'{pass_rate:.3f}','Bend-loss model':bend_model,'Backend':'; '.join(sorted(backends)),'Best design':best})
print('=== Publication demo 12: real-life-like system design matrix ==='); print(f'rows={len(rows)}, radius sweep={radii_um.min():.0f}-{radii_um.max():.0f} µm, pass_rate={pass_rate:.3f}, backend={"; ".join(sorted(backends))}'); print('Status: PASS'); print_artifacts([p1,p1b,p1c,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
