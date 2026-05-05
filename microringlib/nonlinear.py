from __future__ import annotations

from dataclasses import dataclass
import numpy as np


C0 = 299_792_458.0


@dataclass
class KerrCavityParams:
    kappa_ex: float
    kappa_0: float
    detuning: float
    g: float

    @property
    def kappa(self) -> float:
        return self.kappa_ex + self.kappa_0


def solve_kerr_energy(
    input_power: float,
    params: KerrCavityParams,
    initial_energy: float = 0.0,
    max_iter: int = 500,
    relaxation: float = 0.5,
    rtol: float = 1e-10,
) -> float:
    """
    Fixed-point steady-state Kerr cavity solver.

    Solves:

        U = kappa_ex * Pin / ((kappa/2)^2 + (Delta - gU)^2)

    This is a reduced single-mode model, not a full LLE solver.
    """
    P = float(input_power)
    U = float(initial_energy)

    if P < 0:
        raise ValueError("input_power must be nonnegative")
    if not (0 < relaxation <= 1):
        raise ValueError("relaxation must be in (0, 1]")

    kappa = params.kappa

    for _ in range(max_iter):
        denom = (kappa / 2.0) ** 2 + (params.detuning - params.g * U) ** 2
        U_new = params.kappa_ex * P / denom
        U_next = (1.0 - relaxation) * U + relaxation * U_new

        if abs(U_next - U) <= rtol * max(abs(U_next), 1.0):
            return float(U_next)

        U = U_next

    return float(U)


def solve_kerr_sweep(
    input_powers,
    params: KerrCavityParams,
    direction: str = "up",
    initial_energy: float = 0.0,
    max_iter: int = 500,
    relaxation: float = 0.5,
):
    """
    Track a Kerr steady-state branch over an input-power sweep.

    direction:
        "up"   : use input_powers in given order
        "down" : use input_powers in reverse order and return aligned output
    """
    P = np.asarray(input_powers, dtype=float)

    if P.ndim != 1:
        raise ValueError("input_powers must be 1D")
    if np.any(P < 0):
        raise ValueError("input_powers must be nonnegative")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")

    order = np.arange(P.size)
    if direction == "down":
        order = order[::-1]

    U_out = np.zeros_like(P, dtype=float)
    U = float(initial_energy)

    for idx in order:
        U = solve_kerr_energy(
            P[idx],
            params,
            initial_energy=U,
            max_iter=max_iter,
            relaxation=relaxation,
        )
        U_out[idx] = U

    return U_out


def kerr_effective_detuning(energy, params: KerrCavityParams):
    U = np.asarray(energy, dtype=float)
    return params.detuning - params.g * U


def kerr_through_field(energy, params: KerrCavityParams):
    """
    Single-bus through transfer field:

        H = 1 - kappa_ex / (kappa/2 - i Delta_eff)
    """
    U = np.asarray(energy, dtype=float)
    delta_eff = kerr_effective_detuning(U, params)
    return 1.0 - params.kappa_ex / (params.kappa / 2.0 - 1j * delta_eff)


def kerr_through_power(energy, params: KerrCavityParams):
    H = kerr_through_field(energy, params)
    return np.abs(H) ** 2


def kerr_hysteresis(
    input_powers,
    params: KerrCavityParams,
    max_iter: int = 500,
    relaxation: float = 0.5,
):
    """
    Return up/down Kerr branches and through-port hysteresis.
    """
    P = np.asarray(input_powers, dtype=float)

    U_up = solve_kerr_sweep(
        P,
        params,
        direction="up",
        initial_energy=0.0,
        max_iter=max_iter,
        relaxation=relaxation,
    )

    U_down = solve_kerr_sweep(
        P,
        params,
        direction="down",
        initial_energy=U_up[-1],
        max_iter=max_iter,
        relaxation=relaxation,
    )

    T_up = kerr_through_power(U_up, params)
    T_down = kerr_through_power(U_down, params)

    return {
        "input_power": P,
        "energy_up": U_up,
        "energy_down": U_down,
        "through_up": T_up,
        "through_down": T_down,
        "hysteresis": np.abs(T_up - T_down),
        "max_hysteresis": float(np.max(np.abs(T_up - T_down))),
        "max_kerr_shift_over_kappa_up": float(np.max(params.g * U_up) / params.kappa),
        "max_kerr_shift_over_kappa_down": float(np.max(params.g * U_down) / params.kappa),
    }


def kerr_params_from_Q(
    wavelength: float,
    loaded_Q: float,
    coupling_fraction: float,
    detuning_over_kappa: float,
    g: float,
) -> KerrCavityParams:
    """
    Convenience constructor from optical wavelength and loaded Q.

    coupling_fraction = kappa_ex / kappa.
    """
    if loaded_Q <= 0:
        raise ValueError("loaded_Q must be positive")
    if not (0 <= coupling_fraction <= 1):
        raise ValueError("coupling_fraction must be between 0 and 1")

    omega0 = 2.0 * np.pi * C0 / wavelength
    kappa = omega0 / loaded_Q

    kappa_ex = coupling_fraction * kappa
    kappa_0 = (1.0 - coupling_fraction) * kappa
    detuning = detuning_over_kappa * kappa

    return KerrCavityParams(
        kappa_ex=float(kappa_ex),
        kappa_0=float(kappa_0),
        detuning=float(detuning),
        g=float(g),
    )