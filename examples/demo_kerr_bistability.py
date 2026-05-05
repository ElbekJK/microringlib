#!/usr/bin/env python3
"""
MicroringLib-integrated Kerr bistability demo.

This demo uses microringlib's nonlinear module for the Kerr cavity parameter
container and through-port transmission calculation, while explicitly solving
the cubic steady-state relation to produce the traditional S-curve and
physical sweep-up / sweep-down hysteresis branches.

Generated figures:
    kerr_steady_state_s_curve.png
    kerr_hysteresis_branches.png
    kerr_through_transmission_hysteresis.png
    kerr_bistability_zoom.png

Important:
    This is a reduced single-mode Kerr model, not a full LLE solver.
"""

import numpy as np
import matplotlib.pyplot as plt
import microringlib as mrl


# ---------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------

Pin = np.linspace(0.0, 80e-3, 800)  # input power in W

# Cavity linewidths
kappa_ex = 40e9 * 2.0 * np.pi
kappa_0 = 20e9 * 2.0 * np.pi
kappa = kappa_ex + kappa_0

# Strong toy Kerr coefficient chosen to visibly show bistability.
# For calibrated prediction, this should come from n2, mode volume, and cavity normalization.
g = 2.5e26

# Red-detuned pump. Bistability appears when nonlinear shift is comparable to linewidth.
detuning = 3.0 * kappa

# MicroringLib nonlinear parameter object.
params = mrl.KerrCavityParams(
    kappa_ex=kappa_ex,
    kappa_0=kappa_0,
    detuning=detuning,
    g=g,
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def pin_from_U(U, p):
    """
    Forward steady-state mapping:

        Pin(U) = U * [ (kappa/2)^2 + (detuning - gU)^2 ] / kappa_ex

    where U is intracavity energy.
    """
    U = np.asarray(U, dtype=float)
    kappa_total = p.kappa_ex + p.kappa_0
    return U * ((kappa_total / 2.0) ** 2 + (p.detuning - p.g * U) ** 2) / p.kappa_ex


def stability_slope(U, p):
    """
    dPin/dU.

    In the standard reduced Kerr bistability picture, positive slope portions
    of the input-output curve are treated as stable and the negative slope
    branch is unstable.
    """
    U = np.asarray(U, dtype=float)
    kappa_total = p.kappa_ex + p.kappa_0

    delta_eff = p.detuning - p.g * U
    A = (kappa_total / 2.0) ** 2 + delta_eff**2
    dA_dU = -2.0 * p.g * delta_eff

    return (A + U * dA_dU) / p.kappa_ex


def roots_for_power(P, p):
    """
    Solve the cubic steady-state equation for U at input power P:

        g^2 U^3
        - 2 detuning g U^2
        + [detuning^2 + (kappa/2)^2] U
        - kappa_ex P = 0

    Returns sorted real nonnegative roots.
    """
    kappa_total = p.kappa_ex + p.kappa_0

    coeffs = [
        p.g**2,
        -2.0 * p.detuning * p.g,
        p.detuning**2 + (kappa_total / 2.0) ** 2,
        -p.kappa_ex * P,
    ]

    roots = np.roots(coeffs)

    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9 * max(1.0, abs(root.real)):
            if root.real >= 0.0:
                real_roots.append(root.real)

    if len(real_roots) == 0:
        return np.array([], dtype=float)

    return np.array(sorted(real_roots), dtype=float)


def stable_roots_for_power(P, p):
    """
    Return only positive-slope roots.
    """
    roots = roots_for_power(P, p)

    if roots.size == 0:
        return roots

    slopes = stability_slope(roots, p)
    return roots[slopes > 0.0]


def through_transmission(U, p):
    """
    Use microringlib's nonlinear helper when available.

    Falls back to the same reduced formula if the installed version of
    microringlib does not expose kerr_through_power with this signature.
    """
    try:
        return mrl.kerr_through_power(U, p)
    except Exception:
        kappa_total = p.kappa_ex + p.kappa_0
        effective_detuning = p.detuning - p.g * np.asarray(U, dtype=float)
        H = 1.0 - p.kappa_ex / (kappa_total / 2.0 - 1j * effective_detuning)
        return np.abs(H) ** 2


# ---------------------------------------------------------------------
# Build steady-state S-curve and turning points
# ---------------------------------------------------------------------

U_grid = np.linspace(0.0, 2.2 * params.detuning / params.g, 20000)
Pin_grid = pin_from_U(U_grid, params)
slope_grid = stability_slope(U_grid, params)

turning_idx = np.where(np.diff(np.sign(slope_grid)) != 0)[0]

if turning_idx.size < 2:
    raise RuntimeError(
        "No bistable region found. Increase detuning or Kerr coefficient."
    )

i1, i2 = turning_idx[0], turning_idx[1]

P_turn_1 = Pin_grid[i1]
P_turn_2 = Pin_grid[i2]

P_lower = min(P_turn_1, P_turn_2)
P_upper = max(P_turn_1, P_turn_2)

stable_mask = slope_grid > 0.0
unstable_mask = ~stable_mask


# ---------------------------------------------------------------------
# Build physical sweep-up and sweep-down branches
# ---------------------------------------------------------------------

U_up = np.full_like(Pin, np.nan, dtype=float)
U_down = np.full_like(Pin, np.nan, dtype=float)

for i, P in enumerate(Pin):
    stable_roots = stable_roots_for_power(P, params)

    if stable_roots.size == 0:
        continue

    # Sweep-up:
    # stay on low branch until the upper switching point,
    # then jump to the high branch.
    if P < P_upper:
        U_up[i] = stable_roots[0]
    else:
        U_up[i] = stable_roots[-1]

    # Sweep-down:
    # stay on high branch until the lower switching point,
    # then jump to the low branch.
    if P > P_lower:
        U_down[i] = stable_roots[-1]
    else:
        U_down[i] = stable_roots[0]


# ---------------------------------------------------------------------
# Derived observables
# ---------------------------------------------------------------------

T_up = through_transmission(U_up, params)
T_down = through_transmission(U_down, params)

hysteresis = np.abs(T_up - T_down)

valid = np.isfinite(hysteresis) & (Pin > 1e-9)
idx_max = np.where(valid)[0][np.argmax(hysteresis[valid])]

Pin_star = Pin[idx_max]
T_up_star = T_up[idx_max]
T_down_star = T_down[idx_max]
U_up_star = U_up[idx_max]
U_down_star = U_down[idx_max]

max_shift_norm = np.nanmax(params.g * U_up / (params.kappa_ex + params.kappa_0))


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------

print("\n=== Kerr nonlinear bistability demo ===")
print("This is a reduced Kerr cavity model, not a full LLE solver.")
print(f"Input power range: {Pin[0] * 1e3:.3f} to {Pin[-1] * 1e3:.3f} mW")
print(f"kappa_ex / 2pi: {params.kappa_ex / (2 * np.pi) / 1e9:.3f} GHz")
print(f"kappa_0 / 2pi:  {params.kappa_0 / (2 * np.pi) / 1e9:.3f} GHz")
print(f"kappa / 2pi:    {(params.kappa_ex + params.kappa_0) / (2 * np.pi) / 1e9:.3f} GHz")
print(f"detuning / kappa: {params.detuning / (params.kappa_ex + params.kappa_0):.3f}")
print(f"Lower switching power: {P_lower * 1e3:.6f} mW")
print(f"Upper switching power: {P_upper * 1e3:.6f} mW")
print(f"Max Kerr shift / kappa, up-sweep: {max_shift_norm:.6f}")
print(f"Max hysteresis transmission difference: {np.nanmax(hysteresis):.6f}")
print(f"Largest hysteresis near Pin = {Pin_star * 1e3:.3f} mW")
print(f"T_up there:   {T_up_star:.6f}")
print(f"T_down there: {T_down_star:.6f}")
print(f"U_up there:   {U_up_star:.6e}")
print(f"U_down there: {U_down_star:.6e}")


# ---------------------------------------------------------------------
# Plot 1: traditional steady-state S-curve
# ---------------------------------------------------------------------

plt.figure(figsize=(7.2, 5.2))
plt.plot(
    Pin_grid[stable_mask] * 1e3,
    U_grid[stable_mask],
    label="Stable branches",
)
plt.plot(
    Pin_grid[unstable_mask] * 1e3,
    U_grid[unstable_mask],
    "--",
    label="Unstable branch",
)
plt.axvline(P_lower * 1e3, linestyle=":", label="Lower switching")
plt.axvline(P_upper * 1e3, linestyle=":", label="Upper switching")
plt.xlabel("Input power (mW)")
plt.ylabel("Intracavity energy U (arb. units)")
plt.title("Kerr Cavity Steady-State S-Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kerr_steady_state_s_curve.png", dpi=250)


# ---------------------------------------------------------------------
# Plot 2: physical sweep-selected branches
# ---------------------------------------------------------------------

plt.figure(figsize=(7.2, 5.2))
plt.plot(Pin * 1e3, U_up, label="Sweep up")
plt.plot(Pin * 1e3, U_down, "--", label="Sweep down")
plt.axvline(P_lower * 1e3, linestyle=":", label="Lower switching")
plt.axvline(P_upper * 1e3, linestyle=":", label="Upper switching")
plt.scatter([Pin_star * 1e3], [U_up_star], s=40)
plt.scatter([Pin_star * 1e3], [U_down_star], s=40)
plt.xlabel("Input power (mW)")
plt.ylabel("Intracavity energy U (arb. units)")
plt.title("Kerr Hysteresis Branches")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kerr_hysteresis_branches.png", dpi=250)


# ---------------------------------------------------------------------
# Plot 3: through-port observable hysteresis
# ---------------------------------------------------------------------

plt.figure(figsize=(7.2, 5.2))
plt.plot(Pin * 1e3, T_up, label="Sweep up")
plt.plot(Pin * 1e3, T_down, "--", label="Sweep down")
plt.axvline(P_lower * 1e3, linestyle=":", label="Lower switching")
plt.axvline(P_upper * 1e3, linestyle=":", label="Upper switching")
plt.scatter([Pin_star * 1e3], [T_up_star], s=40)
plt.scatter([Pin_star * 1e3], [T_down_star], s=40)
plt.xlabel("Input power (mW)")
plt.ylabel("Through transmission")
plt.title("Kerr Through-Transmission Hysteresis")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kerr_through_transmission_hysteresis.png", dpi=250)


# ---------------------------------------------------------------------
# Plot 4: zoom near strongest branch separation
# ---------------------------------------------------------------------

window_mW = 1.5
x0 = Pin_star * 1e3

mask_zoom = (Pin * 1e3 >= x0 - window_mW) & (Pin * 1e3 <= x0 + window_mW)

plt.figure(figsize=(7.2, 5.2))
plt.plot(Pin[mask_zoom] * 1e3, T_up[mask_zoom], label="Sweep up")
plt.plot(Pin[mask_zoom] * 1e3, T_down[mask_zoom], "--", label="Sweep down")
plt.scatter([Pin_star * 1e3], [T_up_star], s=45)
plt.scatter([Pin_star * 1e3], [T_down_star], s=45)
plt.vlines(
    Pin_star * 1e3,
    min(T_up_star, T_down_star),
    max(T_up_star, T_down_star),
    linestyles=":",
    label=f"Max separation = {np.nanmax(hysteresis):.3f}",
)
plt.xlabel("Input power (mW)")
plt.ylabel("Through transmission")
plt.title("Kerr Bistability Zoom Near Strongest Separation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("kerr_bistability_zoom.png", dpi=250)


print("\nSaved:")
print("  kerr_steady_state_s_curve.png")
print("  kerr_hysteresis_branches.png")
print("  kerr_through_transmission_hysteresis.png")
print("  kerr_bistability_zoom.png")

if __name__ == "__main__":
    plt.show()