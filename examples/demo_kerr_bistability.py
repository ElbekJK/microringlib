import numpy as np
import matplotlib.pyplot as plt


# Reduced steady-state Kerr cavity model.
# This is a qualitative bistability demo, not a full nonlinear Maxwell/LLE solver.

Pin = np.linspace(0, 80e-3, 800)  # input power in W

kappa_ex = 40e9 * 2 * np.pi
kappa_0 = 20e9 * 2 * np.pi
kappa = kappa_ex + kappa_0

# Strong toy Kerr coefficient chosen to clearly show bistability.
g = 2.5e26

# Red-detuned pump. Bistability appears when nonlinear shift is comparable to linewidth.
detuning = 3.0 * kappa


def solve_U(P, U_guess):
    """
    Solve intracavity energy U = |a|^2 using fixed-point iteration.
    """
    U = U_guess

    for _ in range(500):
        denom = (kappa / 2) ** 2 + (detuning - g * U) ** 2
        U_new = kappa_ex * P / denom

        # Partial update helps branch following.
        U = 0.5 * U + 0.5 * U_new

    return U


up = []
U = 0.0

for P in Pin:
    U = solve_U(P, U)
    up.append(U)

up = np.array(up)

down = []
U = up[-1]

for P in Pin[::-1]:
    U = solve_U(P, U)
    down.append(U)

down = np.array(down[::-1])


def through_transmission(U):
    effective_detuning = detuning - g * U
    H = 1.0 - kappa_ex / (kappa / 2 - 1j * effective_detuning)
    return np.abs(H) ** 2


T_up = through_transmission(up)
T_down = through_transmission(down)

hysteresis = np.abs(T_up - T_down)

print("\n=== Kerr nonlinear bistability demo ===")
print("This is a reduced Kerr cavity model, not a full LLE solver.")
print(f"Input power range: {Pin[0] * 1e3:.3f} to {Pin[-1] * 1e3:.3f} mW")
print(f"kappa_ex / 2pi: {kappa_ex / (2 * np.pi) / 1e9:.3f} GHz")
print(f"kappa_0 / 2pi:  {kappa_0 / (2 * np.pi) / 1e9:.3f} GHz")
print(f"kappa / 2pi:    {kappa / (2 * np.pi) / 1e9:.3f} GHz")
print(f"detuning / kappa: {detuning / kappa:.3f}")
print(f"Max Kerr shift / kappa, up-sweep: {np.max(g * up) / kappa:.6f}")
print(f"Max hysteresis transmission difference: {np.max(hysteresis):.6f}")

idx = int(np.argmax(hysteresis))
print(f"Largest hysteresis near Pin = {Pin[idx] * 1e3:.3f} mW")
print(f"T_up there:   {T_up[idx]:.6f}")
print(f"T_down there: {T_down[idx]:.6f}")

plt.figure()
plt.plot(Pin * 1e3, T_up, label="Power sweep up")
plt.plot(Pin * 1e3, T_down, "--", label="Power sweep down")
plt.xlabel("Input power (mW)")
plt.ylabel("Through transmission")
plt.title("Kerr Bistability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kerr_bistability.png", dpi=200)

plt.tight_layout()

plt.figure()
plt.plot(Pin * 1e3, g * up / kappa, label="Up sweep")
plt.plot(Pin * 1e3, g * down / kappa, "--", label="Down sweep")
plt.xlabel("Input power (mW)")
plt.ylabel("Normalized Kerr shift gU / kappa")
plt.title("Nonlinear Resonance Shift")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(Pin * 1e3, hysteresis)
plt.xlabel("Input power (mW)")
plt.ylabel("|T_up - T_down|")
plt.title("Hysteresis Difference")
plt.grid(True)
plt.tight_layout()

plt.show()