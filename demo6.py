import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def print_physics_summary(name, wl, res):
    print(f"\n=== {name} ===")

    if hasattr(res, "power"):
        power = res.power
        print(f"Power min: {np.min(power):.6g}")
        print(f"Power max: {np.max(power):.6g}")
        print(f"Passive check P <= 1: {np.all(power <= 1 + 1e-8)}")

    if hasattr(res, "field"):
        field = np.asarray(res.field)
    
        if field.ndim == 1:
            phase = np.unwrap(np.angle(field))
            print(f"Phase span: {float(phase[-1] - phase[0]):.6g} rad")
    
        elif field.ndim == 2:
            for i in range(field.shape[0]):
                phase = np.unwrap(np.angle(field[i]))
                print(f"Phase span port {i}: {float(phase[-1] - phase[0]):.6g} rad")

    if hasattr(res, "ports") and "through" in res.ports:
        thru = res.ports["through"]["power"]
        print(f"Through min: {np.min(thru):.6g}")
        print(f"Through max: {np.max(thru):.6g}")

    if hasattr(res, "ports") and "drop" in res.ports:
        drop = res.ports["drop"]["power"]
        print(f"Drop min: {np.min(drop):.6g}")
        print(f"Drop max: {np.max(drop):.6g}")

        total = res.ports["through"]["power"] + drop
        print(f"Max P_thru + P_drop: {np.max(total):.8f}")
        print(f"Passive add-drop check: {np.all(total <= 1 + 1e-8)}")

    try:
        target_power = (
            res.ports["through"]["power"]
            if hasattr(res, "ports") and "through" in res.ports
            else res.power
        )
        metrics = mrl.compute_resonance_metrics(wl, target_power, warn=False)

        print("\nResonance metrics:")
        print(f"Resonance wavelength: {metrics['resonance_wavelength'] * 1e9:.6f} nm")
        print(f"FWHM: {metrics['fwhm'] * 1e9:.6f} nm")
        print(f"FSR: {metrics['fsr'] * 1e9:.6f} nm")
        print(f"Loaded Q: {metrics['loaded_Q']:.3f}")
        print(f"Finesse: {metrics['finesse']:.3f}")
        print(f"Extinction ratio: {metrics['extinction_ratio_db']:.3f} dB")
        print(f"Resonances detected: {metrics.get('num_resonances_detected', 'not reported')}")
    except Exception as e:
        print(f"Resonance metrics skipped: {e}")


wl = np.linspace(1520e-9, 1580e-9, 20001)

layers = [
    mrl.Layer(material="SiO2 lower", thickness=2e-6, n=1.444, alpha=0),
    mrl.Layer(material="Si core", thickness=220e-9, n=3.476, alpha=mrl.Layer.dbcm_to_npm(3.0)),
    mrl.Layer(material="SiO2 upper", thickness=2e-6, n=1.444, alpha=0),
]

ring = mrl.RingGeometry(radius=10e-6)

c1 = mrl.Coupler.from_power_coupling(0.12)
c2 = mrl.Coupler.from_power_coupling(0.12)

print("\n=== Coupler physics ===")
print(f"|t1|^2 = {abs(c1.t)**2:.6f}")
print(f"|kappa1|^2 = {abs(c1.kappa)**2:.6f}")
print(f"|t1|^2 + |kappa1|^2 = {abs(c1.t)**2 + abs(c1.kappa)**2:.6f}")
print(f"|t2|^2 = {abs(c2.t)**2:.6f}")
print(f"|kappa2|^2 = {abs(c2.kappa)**2:.6f}")
print(f"|t2|^2 + |kappa2|^2 = {abs(c2.t)**2 + abs(c2.kappa)**2:.6f}")

print("\n=== Ring geometry ===")
print(f"Radius: {ring.radius * 1e6:.3f} um")
print(f"Circumference: {mrl.ring_circumference(ring) * 1e6:.3f} um")
print(f"Estimated FSR at 1550 nm, ng=4.2: {mrl.ring_fsr(1550e-9, 4.2, ring) * 1e9:.3f} nm")

res = mrl.single_mrr_add_drop(
    wavelengths=wl,
    resonator=ring,
    layers=layers,
    t1=c1.t,
    kappa1=c1.kappa,
    t2=c2.t,
    kappa2=c2.kappa,
    polarization="TE",
    overlap_factors=[0.05, 0.90, 0.05],
)

print_physics_summary("Add-drop microring", wl, res)

thru = res.ports["through"]["power"]
drop = res.ports["drop"]["power"]
total = thru + drop

plt.plot(wl * 1e9, thru, label="Through")
plt.plot(wl * 1e9, drop, label="Drop")
plt.plot(wl * 1e9, total, "--", label="Through + Drop")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power")
plt.title("Add-Drop Microring")
plt.legend()
plt.grid(True)
plt.show()