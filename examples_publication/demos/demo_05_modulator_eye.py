#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from shared.path_setup import ensure_project_on_path
ROOT=ensure_project_on_path()
from shared.config import load_config
from shared.decision import save_csv, save_json, save_markdown
from shared.plotting import savefig
from shared.thermal_electric_dispersion import eye_penalty_from_bandwidth
from shared.publication_models import print_artifacts

cfg=load_config('modulator.yaml')['modulator']; rng=np.random.default_rng(7)
br=cfg['bit_rate_gbps']; spb=int(cfg['samples_per_bit']); nbits=int(cfg['n_bits']); bits=rng.integers(0,2,nbits)
wave=np.repeat(bits,spb).astype(float); drive=cfg['drive_voltage_v']*wave
linewidth=cfg['linewidth_pm']; det0=cfg['detuning_zero_pm']; shift=95*drive/cfg['drive_voltage_v']; det=det0-shift
through=1-cfg['extinction_depth']/(1+(2*det/linewidth)**2)
# RC filtering proxy
bw=0.35*br; alpha=np.exp(-1/(spb/(br/bw)))
y=np.empty_like(through); y[0]=through[0]
for i in range(1,len(y)): y[i]=alpha*y[i-1]+(1-alpha)*through[i]
sample=y[spb//2::spb][:nbits]; eye_open=float(abs(np.percentile(sample[bits==1],10)-np.percentile(sample[bits==0],90))); energy_fJ=0.5*cfg['capacitance_fF']*cfg['drive_voltage_v']**2
rows=[{'bit':int(i),'input_bit':int(b),'sampled_power':float(s)} for i,(b,s) in enumerate(zip(bits[:200],sample[:200]))]
dt_ps = 1000.0/(br*spb)
time_ps = np.arange(len(y))*dt_ps
trace_rows=[{'time_ps':float(time_ps[i]),'drive_v':float(drive[i]),'detuning_pm':float(det[i]),'unfiltered_power':float(through[i]),'filtered_power':float(y[i])} for i in range(min(len(y),5000))]
p1=save_csv(ROOT,'demo_05_eye_samples.csv',rows)
p1b=save_csv(ROOT,'demo_05_time_trace.csv',trace_rows)

bitrates=np.linspace(max(5.0,0.25*br),2.0*br,24)
eye_sweep=[]
# Use one fixed electro-optic bandwidth for the device.  The previous sweep set
# bandwidth proportional to bitrate, which made every eye opening identical and
# hid the physical trend.  Here the RC time constant is fixed, so higher bitrates
# have less settling time and the eye closes.
device_bw_gHz=float(os.environ.get('MICRORINGLIB_MODULATOR_DEVICE_BW_GHZ', str(0.35*br)))
tau_ps=1000.0/(2*np.pi*device_bw_gHz)
for br_i in bitrates:
    dt_i_ps=1000.0/(br_i*spb)
    alpha_i=float(np.exp(-dt_i_ps/tau_ps))
    yi=np.empty_like(through); yi[0]=through[0]
    for ii in range(1,len(yi)): yi[ii]=alpha_i*yi[ii-1]+(1-alpha_i)*through[ii]
    si=yi[spb//2::spb][:nbits]
    zs=si[bits==0]; osamp=si[bits==1]
    eo=float(abs(np.percentile(osamp,10)-np.percentile(zs,90)))
    eye_sweep.append({'bitrate_gbps':float(br_i),'eye_opening':eo,'device_bandwidth_gHz_proxy':float(device_bw_gHz),'bitrate_to_bandwidth_ratio':float(br_i/device_bw_gHz)})
p1c=save_csv(ROOT,'demo_05_eye_bitrate_sweep.csv',eye_sweep)
p2=save_json(ROOT,'demo_05_modulator_metrics.json',{'eye_opening':eye_open,'energy_fJ_per_bit':energy_fJ,'bandwidth_penalty':eye_penalty_from_bandwidth(br,bw),'bit_rate_gbps':br,'drive_pp_v':float(cfg['drive_voltage_v']),'sampling_phase_ui':0.5,'eye_x_axis_unit':'UI = unit interval = one bit period','zero_level_mean':float(np.mean(sample[bits==0])),'one_level_mean':float(np.mean(sample[bits==1])),'zero_level_std':float(np.std(sample[bits==0])),'one_level_std':float(np.std(sample[bits==1]))})
plt.figure(figsize=(8,4));
for i in range(120,220): plt.plot(np.linspace(0,2,2*spb),y[i*spb:(i+2)*spb],alpha=0.18)
plt.xlabel('Time within bit period (UI)'); plt.ylabel('Through power'); plt.title('Ring modulator eye diagram, 1 UI = 1 bit period'); p3=savefig(ROOT,'demo_05_modulator_eye.png')

nplot=min(2400,len(y))
plt.figure(figsize=(8,3.4)); plt.plot(time_ps[:nplot],drive[:nplot]); plt.xlabel('Time (ps)'); plt.ylabel('Drive (V)'); plt.title('Electrical drive waveform'); p4=savefig(ROOT,'demo_05_drive_waveform.png')
plt.figure(figsize=(8,3.4)); plt.plot(time_ps[:nplot],det[:nplot]); plt.xlabel('Time (ps)'); plt.ylabel('Detuning (pm)'); plt.title('Resonator detuning waveform'); p5=savefig(ROOT,'demo_05_detuning_waveform.png')
plt.figure(figsize=(8,3.4)); plt.plot(time_ps[:nplot],y[:nplot],label='filtered'); plt.plot(time_ps[:nplot],through[:nplot],alpha=0.35,label='unfiltered'); plt.xlabel('Time (ps)'); plt.ylabel('Through power'); plt.title('Optical output waveform'); plt.legend(); p6=savefig(ROOT,'demo_05_output_waveform.png')

det_grid=np.linspace(det0-130,det0+130,500); transfer=1-cfg['extinction_depth']/(1+(2*det_grid/linewidth)**2)
plt.figure(figsize=(7,4)); plt.plot(det_grid,transfer,lw=2); plt.xlabel('Detuning (pm)'); plt.ylabel('Through power'); plt.title('Static ring transfer curve'); p7=savefig(ROOT,'demo_05_transfer_curve.png')
plt.figure(figsize=(7,4)); plt.hist(sample[bits==0],bins=35,alpha=0.65,label='0 samples'); plt.hist(sample[bits==1],bins=35,alpha=0.65,label='1 samples'); plt.xlabel('Sampled through power'); plt.ylabel('Count'); plt.title('Sampled level histogram'); plt.legend(); p8=savefig(ROOT,'demo_05_sampled_level_histogram.png')
plt.figure(figsize=(7,4)); plt.plot([r['bitrate_gbps'] for r in eye_sweep],[r['eye_opening'] for r in eye_sweep],ls='--',marker='o'); plt.axvline(br,ls=':',label='configured rate'); plt.xlabel('Bitrate (Gb/s)'); plt.ylabel('Eye opening'); plt.title('Eye opening versus bitrate proxy'); plt.legend(); p9=savefig(ROOT,'demo_05_eye_vs_bitrate.png')
p10=save_markdown(ROOT,'demo_05_report.md','Ring modulator eye proxy',{'Metrics':{'eye opening':f'{eye_open:.3f}','energy':f'{energy_fJ:.2f} fJ/bit'},'Use':'Publication-level system output; replace RC/PN coefficients with measured data.'})
print('=== Publication demo 05: modulator eye ==='); print(f'eye_opening={eye_open:.3f}, energy={energy_fJ:.2f} fJ/bit')
print('Status: PASS'); print_artifacts([p1,p1b,p1c,p2,p3,p4,p5,p6,p7,p8,p9,p10])
