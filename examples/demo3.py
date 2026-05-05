import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def print_tracked_row(row):
    print(
        f"T = {row['parameter']:5.1f} C | "
        f"resonance = {row['resonance_wavelength'] * 1e9:.6f} nm | "
        f"FWHM = {row['fwhm'] * 1e9:.6f} nm | "
        f"Q = {row['loaded_Q']:.2f} | "
        f"FSR = {row['fsr'] * 1e9:.6f} nm | "
        f"finesse = {row['finesse']:.3f} | "
        f"ER = {row['extinction_ratio_db']:.3f} dB | "
        f"resonances = {row['num_resonances_detected']}"
    )


wl = np.linspace(1520e-9, 1580e-9, 20001)

layers = [
    mrl.Layer(
        material="Oxide lower",
        thickness=2e-6,
        n=1.444,
        dn_dT=1.0e-5,
        alpha=0,
    ),
    mrl.Layer(
        material="Si core",
        thickness=220e-9,
        n=3.476,
        dn_dT=1.86e-4,
        alpha=mrl.Layer.dbcm_to_npm(2.0),
    ),
    mrl.Layer(
        material="Oxide upper",
        thickness=2e-6,
        n=1.444,
        dn_dT=1.0e-5,
        alpha=0,
    ),
]

ring = mrl.RingGeometry(radius=10e-6)
c = mrl.Coupler.from_power_coupling(0.10)

temps = [20, 25, 30, 35, 40]

print("\n=== Coupler physics ===")
print(f"|t|^2 = {abs(c.t)**2:.6f}")
print(f"|kappa|^2 = {abs(c.kappa)**2:.6f}")
print(f"|t|^2 + |kappa|^2 = {abs(c.t)**2 + abs(c.kappa)**2:.6f}")

print("\n=== Ring geometry ===")
print(f"Radius: {ring.radius * 1e6:.3f} um")
print(f"Circumference: {mrl.ring_circumference(ring) * 1e6:.3f} um")
print(
    f"Estimated FSR at 1550 nm, ng=4.2: "
    f"{mrl.ring_fsr(1550e-9, 4.2, ring) * 1e9:.3f} nm"
)

spectra = []

for temp in temps:
    res = mrl.single_mrr_thru(
        wavelengths=wl,
        resonator=ring,
        layers=layers,
        T=temp,
        t=c.t,
        kappa=c.kappa,
        polarization="TE",
    )

    spectra.append(res.power)

    print(f"\n=== Raw spectrum check: T = {temp:.1f} C ===")
    print(f"Through min: {np.min(res.power):.6g}")
    print(f"Through max: {np.max(res.power):.6g}")
    print(f"Passive check P <= 1: {np.all(res.power <= 1 + 1e-8)}")

    phase = np.unwrap(np.angle(res.field))
    print(f"Phase span: {float(phase[-1] - phase[0]):.6g} rad")

    plt.plot(wl * 1e9, res.power, label=f"{temp} C")

tracked = mrl.track_resonance_vs_parameter(
    wavelengths=wl,
    spectra=spectra,
    parameter_values=temps,
    initial_target_wavelength=1550e-9,
    resonance_kind="dips",
)

print("\n=== Thermal tuning summary: tracked resonance order ===")
for row in tracked:
    print_tracked_row(row)

tracked_temps = np.array([row["parameter"] for row in tracked], dtype=float)
tracked_wl_nm = np.array(
    [row["resonance_wavelength"] * 1e9 for row in tracked],
    dtype=float,
)

if len(tracked_temps) >= 2:
    slope, intercept = np.polyfit(tracked_temps, tracked_wl_nm, 1)
    print("\n=== Thermal tuning coefficient ===")
    print(f"Approx tracked tuning slope: {slope:.6f} nm/C")
    print(f"Linear fit: lambda_res_nm = {slope:.6f} * T + {intercept:.6f}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Through power")
plt.title("Thermal Resonance Shift")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(tracked_temps, tracked_wl_nm, "o-")
plt.xlabel("Temperature (C)")
plt.ylabel("Tracked resonance wavelength (nm)")
plt.title("Tracked Thermal Tuning")
plt.grid(True)
plt.show()