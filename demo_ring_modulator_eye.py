import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl

np.random.seed(1)

wl = np.linspace(1540e-9, 1560e-9, 30000)

layers_base = [
    mrl.Layer(material="Oxide lower", thickness=2e-6, n=1.444, dn_dT=1e-5, alpha=0),
    mrl.Layer(material="Si core", thickness=220e-9, n=3.476, dn_dT=1.86e-4, alpha=mrl.Layer.dbcm_to_npm(2)),
    mrl.Layer(material="Oxide upper", thickness=2e-6, n=1.444, dn_dT=1e-5, alpha=0),
]

ring = mrl.RingGeometry(radius=10e-6)
c = mrl.Coupler.from_power_coupling(0.01)

# Find a resonance first, then place the laser on the resonance slope.
res0 = mrl.single_mrr_thru(
    wavelengths=wl,
    resonator=ring,
    layers=layers_base,
    t=c.t,
    kappa=c.kappa,
    polarization="TE",
)

m0 = mrl.compute_resonance_metrics(
    wl,
    res0.power,
    target_wavelength=1550e-9,
    warn=False,
)

wl_res = m0["resonance_wavelength"]
fwhm = m0["fwhm"]

# Slope-biased laser wavelength. Change sign if needed.
wl0 = wl_res + 0.5 * fwhm

print("\n=== Static ring setup ===")
print(f"Resonance wavelength: {wl_res * 1e9:.6f} nm")
print(f"FWHM: {fwhm * 1e9:.6f} nm")
print(f"Laser wavelength: {wl0 * 1e9:.6f} nm")
print(f"Loaded Q: {m0['loaded_Q']:.2f}")

bitrate = 25e9
samples_per_bit = 64
n_bits = 256
dt = 1 / bitrate / samples_per_bit

bits = np.random.randint(0, 2, n_bits)
drive = np.repeat(bits, samples_per_bit)

# Effective carrier/plasma/thermal index shift toy model.
dn_off = 0.0
dn_on = -5e-3


def transmission_for_dn(dn):
    layers = [
        layers_base[0],
        mrl.Layer(
            material="Si core",
            thickness=220e-9,
            n=3.476 + dn,
            dn_dT=1.86e-4,
            alpha=mrl.Layer.dbcm_to_npm(2),
        ),
        layers_base[2],
    ]

    res = mrl.single_mrr_thru(
        wavelengths=wl,
        resonator=ring,
        layers=layers,
        t=c.t,
        kappa=c.kappa,
        polarization="TE",
    )

    return float(np.interp(wl0, wl, res.power))


P0 = transmission_for_dn(dn_off)
P1 = transmission_for_dn(dn_on)

optical_power = np.where(drive > 0, P1, P0)

# Simple RC bandwidth limit.
electrical_bw = 18e9
tau_rc = 1 / (2 * np.pi * electrical_bw)
alpha_rc = dt / (tau_rc + dt)

filtered = np.zeros_like(optical_power, dtype=float)
filtered[0] = optical_power[0]

for i in range(1, len(filtered)):
    filtered[i] = filtered[i - 1] + alpha_rc * (optical_power[i] - filtered[i - 1])

noise_std = 0.01 * max(abs(P1 - P0), 1e-3)
signal = filtered + noise_std * np.random.randn(len(filtered))

oma = abs(P1 - P0)
er_db = 10 * np.log10(max(P0, P1) / max(min(P0, P1), 1e-15))

print("\n=== Ring modulator eye demo ===")
print(f"Probe wavelength: {wl0 * 1e9:.6f} nm")
print(f"Off-state transmission: {P0:.8f}")
print(f"On-state transmission:  {P1:.8f}")
print(f"OMA: {oma:.8f}")
print(f"Extinction ratio: {er_db:.3f} dB")
print(f"Bitrate: {bitrate / 1e9:.2f} Gb/s")
print(f"Samples per bit: {samples_per_bit}")
print(f"Electrical bandwidth: {electrical_bw / 1e9:.2f} GHz")

t = np.arange(len(signal)) * dt

plt.figure()
plt.plot(wl * 1e9, res0.power)
plt.axvline(wl_res * 1e9, linestyle="--", label="Resonance")
plt.axvline(wl0 * 1e9, linestyle=":", label="Laser")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Through power")
plt.title("Ring Bias Point")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t[:4000] * 1e9, signal[:4000])
plt.xlabel("Time (ns)")
plt.ylabel("Normalized optical power")
plt.title("Ring Modulator Time Waveform")
plt.grid(True)

plt.figure()
eye_span = 2 * samples_per_bit

for k in range(20, n_bits - 2):
    start = k * samples_per_bit
    seg = signal[start:start + eye_span]
    if len(seg) == eye_span:
        plt.plot(np.arange(eye_span) / samples_per_bit, seg, alpha=0.25)

plt.xlabel("Time (bit periods)")
plt.ylabel("Normalized optical power")
plt.title("Ring Modulator Eye Diagram")
plt.grid(True)

plt.show()