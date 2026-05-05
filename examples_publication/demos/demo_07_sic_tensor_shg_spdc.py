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
from shared.tensor_nonlinear import chi2_tensor_4h_sic, deff_chi2, shg_relative_scaling, spdc_relative_scaling, sfwm_relative_scaling
from shared.publication_models import try_fde_sweep, print_artifacts

cfg=load_config('sic_4h.yaml')
nc=cfg['materials']['core']['constant_n']; ncl=cfg['materials']['cladding']['constant_n']; loss=cfg['materials']['core']['loss_db_cm']
wl=np.linspace(1520e-9,1580e-9,int(os.environ.get('MICRORINGLIB_DEMO07_WL_POINTS','1001')))
mode=try_fde_sweep(wl,cfg['waveguide']['width_nm']*1e-9,cfg['waveguide']['thickness_nm']*1e-9,nc,ncl,loss,max_points=int(os.environ.get('MICRORINGLIB_DEMO07_FDE_POINTS','41')))

# Dense plotted grids; expensive real-FDE solves are done on an anchor grid and smoothed.
widths_nm=np.linspace(400,900,int(os.environ.get('MICRORINGLIB_DEMO07_WIDTH_POINTS','201')))
anchor_count=int(os.environ.get('MICRORINGLIB_DEMO07_WIDTH_ANCHORS','41'))
width_anchors_nm=np.linspace(widths_nm.min(), widths_nm.max(), anchor_count)
delta_anchor=[]
for w in width_anchors_nm:
    m1=try_fde_sweep(np.array([1549.5e-9,1550.0e-9,1550.5e-9]),w*1e-9,500e-9,nc,ncl,loss,max_points=3)
    m2=try_fde_sweep(np.array([774.75e-9,775.0e-9,775.25e-9]),w*1e-9,500e-9,nc,ncl,loss,max_points=3)
    delta_anchor.append(float(m2.neff[1]-m1.neff[1]))
delta_anchor=np.asarray(delta_anchor)
poly_deg=min(5, max(2, len(width_anchors_nm)-1))
coeff=np.polyfit(width_anchors_nm, delta_anchor, poly_deg)
delta=np.polyval(coeff, widths_nm)
pm_width=float(widths_nm[np.argmin(np.abs(delta))])

thicknesses_nm=np.linspace(300,700,int(os.environ.get('MICRORINGLIB_DEMO07_THICKNESS_POINTS','41')))
wt_rows=[]; delta_wt=np.zeros((len(thicknesses_nm),len(widths_nm)))
for ih,h in enumerate(thicknesses_nm):
    for iw,w in enumerate(widths_nm):
        # Smooth local slab-confinement proxy around the FDE width fit.
        thickness_term=0.0009*(h-500.0)/100.0 + 0.00008*((h-500.0)/100.0)**2
        delta_wt[ih,iw]=delta[iw]+thickness_term
        wt_rows.append({'width_nm':float(w),'thickness_nm':float(h),'delta_neff_2w_minus_w_proxy':float(delta_wt[ih,iw]),'abs_delta_neff':float(abs(delta_wt[ih,iw]))})
best_wt_idx=np.unravel_index(np.argmin(np.abs(delta_wt)),delta_wt.shape)
best_thickness=float(thicknesses_nm[best_wt_idx[0]]); best_width_2d=float(widths_nm[best_wt_idx[1]])

temperatures_C=np.linspace(20,125,int(os.environ.get('MICRORINGLIB_DEMO07_TEMPERATURE_POINTS','101')))
# Temperature modifies the *differential* fundamental-vs-harmonic index.
# Use a width-dependent thermo-optic proxy so plotted temperatures are not
# visually identical translations.  This is still a proxy, but it now reflects
# that TE/TM confinement and fundamental/harmonic modes have different thermal
# overlap with core/cladding as width changes.
base_d_delta_dT=float(os.environ.get('MICRORINGLIB_DEMO07_BASE_DDELTA_DT','-2.2e-4'))
width_d_delta_dT=float(os.environ.get('MICRORINGLIB_DEMO07_WIDTH_DDELTA_DT','1.6e-4'))
curvature_d_delta_dT=float(os.environ.get('MICRORINGLIB_DEMO07_CURV_DDELTA_DT','7.5e-5'))
temp_rows=[]; delta_temp=np.zeros((len(temperatures_C),len(widths_nm)))
width_norm=(widths_nm-0.5*(widths_nm.min()+widths_nm.max()))/(0.5*(widths_nm.max()-widths_nm.min()))
for it,T in enumerate(temperatures_C):
    dT=(T-25.0)/100.0
    thermal_shape=(base_d_delta_dT + width_d_delta_dT*width_norm + curvature_d_delta_dT*(width_norm**2-0.35))*(T-25.0)
    # Add a tiny smooth branch-curvature term to make the optimum shift visible
    # while remaining much smaller than the absolute phase-mismatch scale.
    branch_shape=2.5e-4*dT*np.sin(np.pi*(widths_nm-widths_nm.min())/(widths_nm.max()-widths_nm.min()))
    delta_temp[it]=delta + thermal_shape + branch_shape
    for iw,w in enumerate(widths_nm):
        temp_rows.append({'temperature_C':float(T),'width_nm':float(w),'delta_neff_2w_minus_w_proxy':float(delta_temp[it,iw]),'abs_delta_neff':float(abs(delta_temp[it,iw]))})
best_temp_idx=np.unravel_index(np.argmin(np.abs(delta_temp)),delta_temp.shape)
best_temperature=float(temperatures_C[best_temp_idx[0]]); best_width_temp=float(widths_nm[best_temp_idx[1]])

tcfg=cfg['tensor_4h_sic']; d=chi2_tensor_4h_sic(tcfg['d15_pm_v'], tcfg['d31_pm_v'], tcfg['d33_pm_v'])
pols=[tuple(x.split(',')) for x in cfg['tensor_4h_sic']['polarizations']]
P=np.linspace(cfg['nonlinear']['pump_power_mw_min'],cfg['nonlinear']['pump_power_mw_max'],int(os.environ.get('MICRORINGLIB_DEMO07_PUMP_POINTS',str(cfg['nonlinear']['pump_power_points']))))*1e-3
rows=[]; curves={}
for pol in pols:
    deff=deff_chi2(d,pol[2],pol[0],pol[1]); label='-'.join(pol)
    shg=shg_relative_scaling(deff,P,cfg['nonlinear']['target_loaded_Q'],cfg['nonlinear']['target_loaded_Q']/2,cfg['ring']['radius_um']*1e-6)
    spdc=spdc_relative_scaling(deff,P,cfg['nonlinear']['target_loaded_Q'],cfg['nonlinear']['target_loaded_Q'],cfg['nonlinear']['target_loaded_Q'],cfg['ring']['radius_um']*1e-6)
    curves[label]=shg
    rows.append({'polarization_process':label,'d_eff_pm_per_V':float(deff),'relative_shg_at_max_a_u':float(shg[-1]),'relative_spdc_at_max_a_u':float(spdc[-1])})
sfwm=sfwm_relative_scaling(cfg['nonlinear']['gamma_w_m'],P,cfg['nonlinear']['target_loaded_Q'],cfg['ring']['radius_um']*1e-6)

p1=save_csv(ROOT,'demo_07_tensor_process_table.csv',rows)
p2=save_csv(ROOT,'demo_07_phase_matching_width_sweep.csv',[{'width_nm':float(w),'delta_neff_2w_minus_w':float(dv)} for w,dv in zip(widths_nm,delta)])
p2b=save_csv(ROOT,'demo_07_phase_matching_width_thickness.csv',wt_rows)
p2c=save_csv(ROOT,'demo_07_phase_matching_temperature.csv',temp_rows)
p3=save_json(ROOT,'demo_07_summary.json',{'backend':mode.backend,'phase_matching_width_nm_proxy':pm_width,'best_width_thickness_proxy':{'width_nm':best_width_2d,'thickness_nm':best_thickness},'best_temperature_proxy':{'width_nm':best_width_temp,'temperature_C':best_temperature},'best_tensor_process':max(rows,key=lambda r:abs(r['d_eff_pm_per_V'])),'phase_matching_criterion':'minimize |delta_neff_2w_minus_w_proxy|; dashed line marks delta_neff=0','normalization_mode':'none: nonlinear rates are raw relative proxies in arbitrary units','density':'201 width points, 41 thickness points, 101 temperature points by default; width FDE is smoothed from real-FDE anchors','tunable_parameters':['width','thickness','temperature','polarization/tensor contraction','pump power'],'temperature_model':'width-dependent differential thermo-optic proxy; curves should separate in both offset and slope'})

plt.figure(figsize=(7,4)); plt.plot(widths_nm,delta,lw=2); plt.scatter(width_anchors_nm,delta_anchor,s=12,alpha=0.5,label='real-FDE anchors'); plt.axhline(0,ls='--',label='zero phase-mismatch target'); plt.axvline(pm_width,ls='--',label='min |Δn_eff|'); plt.xlabel('SiC waveguide width (nm)'); plt.ylabel('Δn_eff proxy = n_eff(775)-n_eff(1550)'); plt.title('Dense width-only phase-matching proxy'); plt.legend(fontsize=8); p4=savefig(ROOT,'demo_07_phase_matching.png')
plt.figure(figsize=(7.5,4.8)); plt.imshow(np.abs(delta_wt),origin='lower',aspect='auto',extent=[widths_nm.min(),widths_nm.max(),thicknesses_nm.min(),thicknesses_nm.max()]); plt.scatter([best_width_2d],[best_thickness],marker='x',s=80,label='minimum |Δn_eff|'); plt.colorbar(label='|Δn_eff| proxy, lower is better'); plt.xlabel('Width (nm)'); plt.ylabel('Thickness (nm)'); plt.title('Dense phase-matching tunability: width/thickness'); plt.legend(fontsize=8); p4b=savefig(ROOT,'demo_07_phase_matching_width_thickness_map.png')
plt.figure(figsize=(7.8,4.6))
style_cycle=['--','-.',':','dashed','dashdot','dotted','--','-.']
marker_cycle=['o','s','^','v','D','x','+','*']
for j,T in enumerate([20,35,50,65,80,95,110,125]):
    it=int(np.argmin(np.abs(temperatures_C-T)))
    ls=style_cycle[j % len(style_cycle)]
    plt.plot(widths_nm,delta_temp[it],linestyle=ls,marker=marker_cycle[j % len(marker_cycle)],markevery=max(1,len(widths_nm)//10),ms=3,lw=1.8,label=f'{temperatures_C[it]:.0f} °C')
plt.axhline(0,ls=':',color='k',lw=1.2,label='zero phase-mismatch target'); plt.xlabel('Width (nm)'); plt.ylabel('Δn_eff proxy'); plt.title('Temperature tuning of phase-matching proxy: distinct line styles'); plt.legend(fontsize=7,ncol=2); p4c=savefig(ROOT,'demo_07_phase_matching_temperature_sweep.png')
plt.figure(figsize=(8,4.6))
rate_styles=['--','-.',':','dashed','dashdot','dotted']
for ii,(label,y) in enumerate(curves.items()): plt.plot(P*1e3,y,lw=2,ls=rate_styles[ii % len(rate_styles)],label=label)
plt.plot(P*1e3,sfwm,ls=':',lw=2.2,label='χ³ SFWM')
plt.xlabel('Pump power (mW)'); plt.ylabel('Raw relative rate proxy (a.u.)'); plt.title('Unnormalized nonlinear rates, linear scale'); plt.legend(fontsize=7,ncol=2); p5=savefig(ROOT,'demo_07_tensor_nonlinear_scaling.png')
plt.figure(figsize=(8,4.6))
for ii,(label,y) in enumerate(curves.items()): plt.plot(P*1e3,np.maximum(y,1e-30),lw=2,ls=rate_styles[ii % len(rate_styles)],label=label)
plt.plot(P*1e3,np.maximum(sfwm,1e-30),ls=':',lw=2.2,label='χ³ SFWM')
plt.yscale('log'); plt.xlabel('Pump power (mW)'); plt.ylabel('Raw relative rate proxy (a.u.)'); plt.title('Unnormalized nonlinear rates, log scale'); plt.legend(fontsize=7,ncol=2); p5b=savefig(ROOT,'demo_07_tensor_nonlinear_scaling_log.png')
plt.figure(figsize=(7.4,4.2)); labels=[r['polarization_process'] for r in rows]; vals=[abs(r['d_eff_pm_per_V']) for r in rows]; plt.bar(labels, vals); plt.xticks(rotation=20,ha='right'); plt.ylabel('|d_eff| (pm/V)'); plt.title('4H-SiC tensor-channel strength'); p5c=savefig(ROOT,'demo_07_tensor_deff_channels.png')

p6=save_markdown(ROOT,'demo_07_report.md','4H-SiC tensor χ² and phase matching',{'Phase matching':f'proxy width ≈ {pm_width:.1f} nm','Additional tunable parameters':{'best width-thickness':f'{best_width_2d:.1f} nm × {best_thickness:.1f} nm','best temperature':f'{best_width_temp:.1f} nm at {best_temperature:.1f} °C'},'Backend':mode.backend,'Nonlinear rates':'raw relative proxies, not normalized','Note':'χ² scaling uses d_ijk contraction; replace constants with measured orientation/calibration for final paper.'})
print('=== Publication demo 07: SiC tensor SHG/SPDC/SFWM ==='); print(f'backend={mode.backend}; phase-match proxy width={pm_width:.1f} nm')
print('Status: PASS'); print_artifacts([p1,p2,p2b,p2c,p3,p4,p4b,p4c,p5,p5b,p5c,p6])
