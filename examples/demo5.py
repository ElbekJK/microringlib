import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


def print_group_delay_summary(wl, c, res, tau):
    print("\n=== Group delay microring ===")

    print("\nCoupler physics:")
    print(f"|t|^2 = {abs(c.t)**2:.6f}")
    print(f"|kappa|^2 = {abs(c.kappa)**2:.6f}")
    print(f"|t|^2 + |kappa|^2 = {abs(c.t)**2 + abs(c.kappa)**2:.6f}")

    print("\nPower checks:")
    print(f"Through min: {np.min(res.power):.6g}")
    print(f"Through max: {np.max(res.power):.6g}")
    print(f"Passive check P <= 1: {np.all(res.power <= 1 + 1e-8)}")

    phase = np.unwrap(np.angle(res.field))
    print("\nPhase:")
    print(f"Phase span: {float(phase[-1] - phase[0]):.6g} rad")

    print("\nGroup delay:")
    print(f"Minimum delay: {np.min(tau) * 1e12:.6f} ps")
    print(f"Maximum delay: {np.max(tau) * 1e12:.6f} ps")
    print(f"Mean delay:    {np.mean(tau) * 1e12:.6f} ps")

    idx_max_delay = int(np.argmax(tau))
    print(f"Max-delay wavelength: {wl[idx_max_delay] * 1e9:.6f} nm")

    try:
        metrics = mrl.compute_resonance_metrics(wl, res.power)

        print("\nResonance metrics:")
        print(f"Resonance wavelength: {metrics['resonance_wavelength'] * 1e9:.6f} nm")
        print(f"FWHM: {metrics['fwhm'] * 1e9:.6f} nm")
        print(f"FSR: {metrics['fsr'] * 1e9:.6f} nm")
        print(f"Loaded Q: {metrics['loaded_Q']:.3f}")
        print(f"Finesse: {metrics['finesse']:.3f}")
        print(f"Extinction ratio: {metrics['extinction_ratio_db']:.3f} dB")
        print(f"Resonances detected: {metrics.get('num_resonances_detected', 'not reported')}")

        delta_nm = abs(wl[idx_max_delay] - metrics["resonance_wavelength"]) * 1e9
        print(f"Delay peak offset from resonance: {delta_nm:.6f} nm")

    except Exception as e:
        print(f"\nResonance metrics skipped: {e}")


wl = np.linspace(1520e-9, 1580e-9, 20001)

layers = [
    mrl.Layer(material="Oxide lower", thickness=2e-6, n=1.444, alpha=0),
    mrl.Layer(material="Si core", thickness=220e-9, n=3.476, alpha=mrl.Layer.dbcm_to_npm(2.0)),
    mrl.Layer(material="Oxide upper", thickness=2e-6, n=1.444, alpha=0),
]

ring = mrl.RingGeometry(radius=10e-6)
c = mrl.Coupler.from_power_coupling(0.1)

print("\n=== Ring geometry ===")
print(f"Radius: {ring.radius * 1e6:.3f} um")
print(f"Circumference: {mrl.ring_circumference(ring) * 1e6:.3f} um")
print(f"Estimated FSR at 1550 nm, ng=4.2: {mrl.ring_fsr(1550e-9, 4.2, ring) * 1e9:.3f} nm")

res = mrl.single_mrr_thru(
    wavelengths=wl,
    resonator=ring,
    layers=layers,
    t=c.t,
    kappa=c.kappa,
    polarization="TE",
)

tau = mrl.compute_group_delay(wl, res.field)

print_group_delay_summary(wl, c, res, tau)

plt.plot(wl * 1e9, res.power)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Through power")
plt.title("Through-Port Transmission")
plt.grid(True)
plt.show()

plt.plot(wl * 1e9, tau * 1e12)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Delay (ps)")
plt.title("Ring Group Delay")
plt.grid(True)
plt.show()