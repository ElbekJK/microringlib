import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def print_cascade_summary(wl, rings, res):
    thru = res.power[0]
    drop = res.power[1]
    total = thru + drop

    print("\n=== Cascaded add-drop rings ===")

    print("\nRing geometry:")
    for i, ring in enumerate(rings, start=1):
        print(
            f"Ring {i}: radius = {ring.radius * 1e6:.3f} um | "
            f"circumference = {mrl.ring_circumference(ring) * 1e6:.3f} um | "
            f"FSR estimate = {mrl.ring_fsr(1550e-9, 4.2, ring) * 1e9:.3f} nm"
        )

    print("\nPower checks:")
    print(f"Through min/max: {np.min(thru):.6g} / {np.max(thru):.6g}")
    print(f"Drop min/max:    {np.min(drop):.6g} / {np.max(drop):.6g}")
    print(f"Max P_thru + P_drop: {np.max(total):.8f}")
    print(f"Min P_thru + P_drop: {np.min(total):.8f}")
    print(f"Passive cascade check: {np.all(total <= 1 + 1e-8)}")

    try:
        thru_metrics = mrl.compute_resonance_metrics(wl, thru)
        print("\nThrough-port metrics:")
        print(f"Resonance wavelength: {thru_metrics['resonance_wavelength'] * 1e9:.6f} nm")
        print(f"FWHM: {thru_metrics['fwhm'] * 1e9:.6f} nm")
        print(f"FSR: {thru_metrics['fsr'] * 1e9:.6f} nm")
        print(f"Loaded Q: {thru_metrics['loaded_Q']:.3f}")
        print(f"Finesse: {thru_metrics['finesse']:.3f}")
        print(f"Extinction ratio: {thru_metrics['extinction_ratio_db']:.3f} dB")
        print(f"Resonances detected: {thru_metrics.get('num_resonances_detected', 'not reported')}")
    except Exception as e:
        print(f"\nThrough-port metrics skipped: {e}")

    try:
        drop_metrics = mrl.compute_resonance_metrics(wl, drop, resonance_kind="peaks")
        print("\nDrop-port peak metrics:")
        print(f"Peak wavelength: {drop_metrics['resonance_wavelength'] * 1e9:.6f} nm")
        print(f"FWHM: {drop_metrics['fwhm'] * 1e9:.6f} nm")
        print(f"Peak spacing / FSR: {drop_metrics['fsr'] * 1e9:.6f} nm")
        print(f"Loaded Q: {drop_metrics['loaded_Q']:.3f}")
        print(f"Finesse: {drop_metrics['finesse']:.3f}")
        print(f"Peak extinction ratio: {drop_metrics['extinction_ratio_db']:.3f} dB")
        print(f"Peaks detected: {drop_metrics.get('num_resonances_detected', 'not reported')}")
    except Exception as e:
        print(f"\nDrop-port metrics skipped: {e}")


wl = np.linspace(1520e-9, 1580e-9, 20001)

layers = [
    mrl.Layer(material="Oxide lower", thickness=2e-6, n=1.444, alpha=0),
    mrl.Layer(material="Si core", thickness=220e-9, n=3.476, alpha=mrl.Layer.dbcm_to_npm(2.0)),
    mrl.Layer(material="Oxide upper", thickness=2e-6, n=1.444, alpha=0),
]

rings = [
    mrl.RingGeometry(radius=10.0e-6),
    mrl.RingGeometry(radius=10.2e-6),
    mrl.RingGeometry(radius=10.4e-6),
]

res = mrl.cascaded_mrrs_add_drop(
    wavelengths=wl,
    resonators=rings,
    layers=layers,
    coupling={"K": 0.08},
    polarization="TE",
)

print_cascade_summary(wl, rings, res)

thru = res.power[0]
drop = res.power[1]
total = thru + drop

plt.plot(wl * 1e9, thru, label="Through")
plt.plot(wl * 1e9, drop, label="Drop")
plt.plot(wl * 1e9, total, "--", label="Through + Drop")
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.title("Cascaded Rings")
plt.grid(True)
plt.show()