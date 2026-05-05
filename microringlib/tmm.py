"""Transfer-matrix and scattering-matrix primitives for integrated photonics.

This module intentionally stays lightweight: it provides first-class matrix
objects, physically checked primitive blocks, and simple composition helpers
that can be used alongside the existing microring compact models.

Conventions
-----------
* Wavelength is in meters.
* Power loss ``alpha_power`` is in Np/m.
* Scattering matrices use ``S[out_port, in_port]`` at each wavelength.
* Two-port cascade uses the Redheffer star-product specialized to two ports.
* ``port 0`` is the left/input side and ``port 1`` is the right/output side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
import numpy as np

C0 = 299_792_458.0

ArrayLike = Any


def _as_wavelengths(wavelengths: ArrayLike) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim == 0:
        wl = wl[None]
    if wl.ndim != 1:
        raise ValueError("wavelengths must be a 1D array or scalar")
    if wl.size == 0 or np.any(~np.isfinite(wl)) or np.any(wl <= 0):
        raise ValueError("wavelengths must be finite and positive")
    return wl


def _broadcast_1d(x: ArrayLike, wavelengths: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else float)
    if arr.ndim == 0:
        return np.full(wavelengths.shape, arr, dtype=arr.dtype)
    if arr.shape != wavelengths.shape:
        raise ValueError(f"{name} must be scalar or have shape {wavelengths.shape}")
    return arr


def dbcm_to_npm(alpha_db_cm: float) -> float:
    """Convert power loss from dB/cm to Np/m."""
    return float(alpha_db_cm * np.log(10.0) / 10.0 * 100.0)


def npm_to_dbcm(alpha_npm: float) -> float:
    """Convert power loss from Np/m to dB/cm."""
    return float(alpha_npm * 10.0 / np.log(10.0) / 100.0)


def propagation_amplitude(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    length: float,
    alpha_power: ArrayLike = 0.0,
) -> np.ndarray:
    """Complex field transmission of a uniform waveguide section.

    The field amplitude is ``exp(-alpha_power*L/2) * exp(-j*beta*L)``.
    """
    wl = _as_wavelengths(wavelengths)
    if length < 0:
        raise ValueError("length must be non-negative")
    neff = _broadcast_1d(n_eff, wl, "n_eff").astype(np.complex128)
    alpha = np.asarray(_broadcast_1d(alpha_power, wl, "alpha_power"), dtype=float)
    if np.any(alpha < 0):
        raise ValueError("alpha_power must be non-negative")
    beta_l = 2.0 * np.pi * neff * float(length) / wl
    return np.exp(-0.5 * alpha * float(length)) * np.exp(-1j * beta_l)


@dataclass(frozen=True)
class TransferMatrix:
    """Wavelength-dependent 2x2 transfer/ABCD matrix.

    ``data`` has shape ``(n_wavelengths, 2, 2)``. The convenience property
    ``through`` assumes the common no-reverse-incident-wave convention and
    returns ``1 / A``.
    """

    wavelengths: np.ndarray
    data: np.ndarray
    ports: tuple[str, str] = ("left", "right")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wl = _as_wavelengths(self.wavelengths)
        data = np.asarray(self.data, dtype=np.complex128)
        if data.shape == (2, 2):
            data = np.broadcast_to(data, (wl.size, 2, 2)).copy()
        if data.shape != (wl.size, 2, 2):
            raise ValueError("data must have shape (N,2,2) or (2,2)")
        object.__setattr__(self, "wavelengths", wl)
        object.__setattr__(self, "data", data)

    def __matmul__(self, other: "TransferMatrix") -> "TransferMatrix":
        return cascade_transfer([self, other])

    @property
    def A(self) -> np.ndarray:
        return self.data[:, 0, 0]

    @property
    def B(self) -> np.ndarray:
        return self.data[:, 0, 1]

    @property
    def C(self) -> np.ndarray:
        return self.data[:, 1, 0]

    @property
    def D(self) -> np.ndarray:
        return self.data[:, 1, 1]

    @property
    def through(self) -> np.ndarray:
        return 1.0 / self.A

    @property
    def power(self) -> np.ndarray:
        return np.abs(self.through) ** 2


@dataclass(frozen=True)
class ScatteringMatrix:
    """Wavelength-dependent scattering matrix.

    ``S`` has shape ``(n_wavelengths, n_ports, n_ports)`` and follows
    ``S[out_port, in_port]`` convention.
    """

    wavelengths: np.ndarray
    S: np.ndarray
    ports: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wl = _as_wavelengths(self.wavelengths)
        S = np.asarray(self.S, dtype=np.complex128)
        if S.ndim == 2:
            S = np.broadcast_to(S, (wl.size,) + S.shape).copy()
        if S.ndim != 3 or S.shape[0] != wl.size or S.shape[1] != S.shape[2]:
            raise ValueError("S must have shape (N,P,P) or (P,P)")
        if len(self.ports) != S.shape[1]:
            raise ValueError("ports length must match S dimensions")
        object.__setattr__(self, "wavelengths", wl)
        object.__setattr__(self, "S", S)

    @property
    def nports(self) -> int:
        return self.S.shape[1]

    def response(self, input_port: int | str = 0, output_port: int | str = 1) -> np.ndarray:
        i = self.port_index(input_port)
        o = self.port_index(output_port)
        return self.S[:, o, i]

    def power(self, input_port: int | str = 0, output_port: int | str = 1) -> np.ndarray:
        return np.abs(self.response(input_port, output_port)) ** 2

    def port_index(self, port: int | str) -> int:
        if isinstance(port, str):
            try:
                return self.ports.index(port)
            except ValueError as exc:
                raise ValueError(f"unknown port {port!r}") from exc
        idx = int(port)
        if idx < 0 or idx >= self.nports:
            raise ValueError("port index out of range")
        return idx

    def passivity_margin(self) -> float:
        return passivity_margin(self.S)

    def is_passive(self, atol: float = 1e-9) -> bool:
        return is_passive(self.S, atol=atol)

    def is_unitary(self, atol: float = 1e-9) -> bool:
        return is_unitary(self.S, atol=atol)

    def is_reciprocal(self, atol: float = 1e-9) -> bool:
        return is_reciprocal(self.S, atol=atol)

    def __matmul__(self, other: "ScatteringMatrix") -> "ScatteringMatrix":
        return cascade_two_ports([self, other])


def identity_transfer(wavelengths: ArrayLike) -> TransferMatrix:
    wl = _as_wavelengths(wavelengths)
    data = np.broadcast_to(np.eye(2, dtype=np.complex128), (wl.size, 2, 2)).copy()
    return TransferMatrix(wl, data, metadata={"type": "identity_transfer"})


def propagation_transfer(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    length: float,
    alpha_power: ArrayLike = 0.0,
) -> TransferMatrix:
    """Transfer matrix of a matched waveguide section.

    With the convention used here, ``through = 1/A`` equals the physical
    propagation amplitude.
    """
    wl = _as_wavelengths(wavelengths)
    p = propagation_amplitude(wl, n_eff, length, alpha_power)
    data = np.zeros((wl.size, 2, 2), dtype=np.complex128)
    data[:, 0, 0] = 1.0 / p
    data[:, 1, 1] = p
    return TransferMatrix(wl, data, metadata={"type": "propagation", "length": float(length)})


def waveguide_smatrix(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    length: float,
    alpha_power: ArrayLike = 0.0,
) -> ScatteringMatrix:
    """Matched two-port scattering matrix of a waveguide section."""
    wl = _as_wavelengths(wavelengths)
    p = propagation_amplitude(wl, n_eff, length, alpha_power)
    S = np.zeros((wl.size, 2, 2), dtype=np.complex128)
    S[:, 1, 0] = p
    S[:, 0, 1] = p
    return ScatteringMatrix(wl, S, ("in", "out"), {"type": "waveguide", "length": float(length)})


def phase_shifter_smatrix(wavelengths: ArrayLike, phase_rad: ArrayLike, loss_power: ArrayLike = 1.0) -> ScatteringMatrix:
    """Matched two-port phase shifter with optional power transmission."""
    wl = _as_wavelengths(wavelengths)
    phase = np.asarray(_broadcast_1d(phase_rad, wl, "phase_rad"), dtype=float)
    loss = np.asarray(_broadcast_1d(loss_power, wl, "loss_power"), dtype=float)
    if np.any(loss < 0) or np.any(loss > 1.0 + 1e-12):
        raise ValueError("loss_power must be between 0 and 1")
    a = np.sqrt(loss) * np.exp(1j * phase)
    S = np.zeros((wl.size, 2, 2), dtype=np.complex128)
    S[:, 1, 0] = a
    S[:, 0, 1] = a
    return ScatteringMatrix(wl, S, ("in", "out"), {"type": "phase_shifter"})


def directional_coupler_smatrix(wavelengths: ArrayLike, K: float, phase: float = np.pi / 2.0) -> ScatteringMatrix:
    """Lossless reciprocal 2x2 directional-coupler model.

    ``K`` is power coupling. The matrix is ``[[t, exp(j*phase) k],
    [exp(j*phase) k, t]]``.
    """
    wl = _as_wavelengths(wavelengths)
    K = float(K)
    if K < 0 or K > 1:
        raise ValueError("K must be in [0,1]")
    t = np.sqrt(1.0 - K)
    k = np.sqrt(K) * np.exp(1j * float(phase))
    S0 = np.array([[t, k], [k, t]], dtype=np.complex128)
    S = np.broadcast_to(S0, (wl.size, 2, 2)).copy()
    return ScatteringMatrix(wl, S, ("bar", "cross"), {"type": "directional_coupler", "K": K})


def splitter_2x2_smatrix(wavelengths: ArrayLike, ratio: float = 0.5) -> ScatteringMatrix:
    """Alias for a lossless 2x2 splitter/coupler."""
    return directional_coupler_smatrix(wavelengths, ratio)


def mzi_smatrix(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    length_1: float,
    length_2: float,
    K1: float = 0.5,
    K2: float = 0.5,
    alpha_power: ArrayLike = 0.0,
    extra_phase_1: ArrayLike = 0.0,
    extra_phase_2: ArrayLike = 0.0,
) -> ScatteringMatrix:
    """Mach-Zehnder interferometer from two couplers and two arms."""
    wl = _as_wavelengths(wavelengths)
    c1 = directional_coupler_smatrix(wl, K1).S
    c2 = directional_coupler_smatrix(wl, K2).S
    p1 = propagation_amplitude(wl, n_eff, length_1, alpha_power) * np.exp(1j * _broadcast_1d(extra_phase_1, wl, "extra_phase_1"))
    p2 = propagation_amplitude(wl, n_eff, length_2, alpha_power) * np.exp(1j * _broadcast_1d(extra_phase_2, wl, "extra_phase_2"))
    S = np.zeros((wl.size, 2, 2), dtype=np.complex128)
    for i in range(wl.size):
        arm = np.diag([p1[i], p2[i]])
        S[i] = c2[i] @ arm @ c1[i]
    return ScatteringMatrix(wl, S, ("in0", "out1"), {"type": "mzi", "length_1": float(length_1), "length_2": float(length_2)})


def ring_roundtrip_amplitude(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    radius: float,
    alpha_power: ArrayLike = 0.0,
) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    return propagation_amplitude(wavelengths, n_eff, 2.0 * np.pi * float(radius), alpha_power)


def ring_allpass_field(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    radius: float,
    K: float,
    alpha_power: ArrayLike = 0.0,
) -> np.ndarray:
    """Through field of an all-pass microring."""
    wl = _as_wavelengths(wavelengths)
    K = float(K)
    if K < 0 or K > 1:
        raise ValueError("K must be in [0,1]")
    t = np.sqrt(1.0 - K)
    a = ring_roundtrip_amplitude(wl, n_eff, radius, alpha_power)
    return (t - a) / (1.0 - t * a)


def ring_allpass_smatrix(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    radius: float,
    K: float,
    alpha_power: ArrayLike = 0.0,
) -> ScatteringMatrix:
    wl = _as_wavelengths(wavelengths)
    h = ring_allpass_field(wl, n_eff, radius, K, alpha_power)
    S = np.zeros((wl.size, 2, 2), dtype=np.complex128)
    S[:, 1, 0] = h
    S[:, 0, 1] = h
    return ScatteringMatrix(wl, S, ("in", "through"), {"type": "ring_allpass", "radius": float(radius), "K": float(K)})


def ring_add_drop_fields(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    radius: float,
    K1: float,
    K2: float | None = None,
    alpha_power: ArrayLike = 0.0,
) -> dict[str, np.ndarray]:
    """Through/drop fields of a two-coupler add-drop ring for a bus input."""
    wl = _as_wavelengths(wavelengths)
    K1 = float(K1)
    K2 = K1 if K2 is None else float(K2)
    if not (0 <= K1 <= 1 and 0 <= K2 <= 1):
        raise ValueError("K1 and K2 must be in [0,1]")
    t1 = np.sqrt(1.0 - K1)
    t2 = np.sqrt(1.0 - K2)
    L = 2.0 * np.pi * float(radius)
    half = propagation_amplitude(wl, n_eff, 0.5 * L, alpha_power)
    rt = half * half
    den = 1.0 - t1 * t2 * rt
    through = (t1 - t2 * rt) / den
    drop = 1j * np.sqrt(K1 * K2) * half / den
    return {"through": through, "drop": drop, "drop_power": np.abs(drop) ** 2, "through_power": np.abs(through) ** 2}


def fabry_perot_field(
    wavelengths: ArrayLike,
    n_eff: ArrayLike,
    length: float,
    r1: complex,
    r2: complex | None = None,
    alpha_power: ArrayLike = 0.0,
) -> np.ndarray:
    """Transmission field of a simple Fabry-Perot cavity."""
    wl = _as_wavelengths(wavelengths)
    r1 = complex(r1)
    r2 = r1 if r2 is None else complex(r2)
    if abs(r1) > 1 or abs(r2) > 1:
        raise ValueError("mirror amplitude reflectivities must have magnitude <= 1")
    t1 = np.sqrt(max(0.0, 1.0 - abs(r1) ** 2))
    t2 = np.sqrt(max(0.0, 1.0 - abs(r2) ** 2))
    p = propagation_amplitude(wl, n_eff, length, alpha_power)
    return t1 * t2 * p / (1.0 - r1 * r2 * p * p)


def cascade_transfer(blocks: Sequence[TransferMatrix]) -> TransferMatrix:
    if not blocks:
        raise ValueError("blocks cannot be empty")
    wl = blocks[0].wavelengths
    total = np.broadcast_to(np.eye(2, dtype=np.complex128), (wl.size, 2, 2)).copy()
    meta = []
    for block in blocks:
        if not np.allclose(block.wavelengths, wl, rtol=0, atol=0):
            raise ValueError("all transfer matrices must use identical wavelength grids")
        total = total @ block.data
        meta.append(block.metadata.get("type", "block"))
    return TransferMatrix(wl, total, metadata={"type": "cascade_transfer", "blocks": meta})


def cascade_two_ports(blocks: Sequence[ScatteringMatrix]) -> ScatteringMatrix:
    """Cascade matched two-port scattering matrices."""
    if not blocks:
        raise ValueError("blocks cannot be empty")
    wl = blocks[0].wavelengths
    total = blocks[0].S.copy()
    meta = [blocks[0].metadata.get("type", "block")]
    for block in blocks[1:]:
        if block.nports != 2 or total.shape[1] != 2:
            raise ValueError("cascade_two_ports only supports 2-port blocks")
        if not np.allclose(block.wavelengths, wl, rtol=0, atol=0):
            raise ValueError("all scattering matrices must use identical wavelength grids")
        A = total
        B = block.S
        den = 1.0 - A[:, 1, 1] * B[:, 0, 0]
        if np.any(np.abs(den) < 1e-14):
            raise ZeroDivisionError("singular two-port cascade denominator")
        S = np.zeros_like(A)
        S[:, 0, 0] = A[:, 0, 0] + A[:, 0, 1] * B[:, 0, 0] * A[:, 1, 0] / den
        S[:, 0, 1] = A[:, 0, 1] * B[:, 0, 1] / den
        S[:, 1, 0] = B[:, 1, 0] * A[:, 1, 0] / den
        S[:, 1, 1] = B[:, 1, 1] + B[:, 1, 0] * A[:, 1, 1] * B[:, 0, 1] / den
        total = S
        meta.append(block.metadata.get("type", "block"))
    return ScatteringMatrix(wl, total, ("in", "out"), {"type": "cascade_two_ports", "blocks": meta})


@dataclass
class Cascade:
    """Small convenience wrapper for sequential two-port S-matrix circuits."""

    blocks: list[ScatteringMatrix]
    name: str = "cascade"

    def solve(self) -> ScatteringMatrix:
        result = cascade_two_ports(self.blocks)
        result.metadata["name"] = self.name
        return result


def passivity_margin(S: ArrayLike) -> float:
    """Return max eigenvalue of SᴴS - I across wavelengths.

    Passive systems have margin <= 0 within numerical tolerance.
    """
    arr = np.asarray(S, dtype=np.complex128)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    worst = -np.inf
    for mat in arr:
        eig = np.linalg.eigvalsh(mat.conj().T @ mat - np.eye(mat.shape[1]))
        worst = max(worst, float(np.max(np.real(eig))))
    return worst


def is_passive(S: ArrayLike, atol: float = 1e-9) -> bool:
    return passivity_margin(S) <= float(atol)


def is_unitary(S: ArrayLike, atol: float = 1e-9) -> bool:
    arr = np.asarray(S, dtype=np.complex128)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    for mat in arr:
        if not np.allclose(mat.conj().T @ mat, np.eye(mat.shape[1]), atol=atol, rtol=0):
            return False
    return True


def is_reciprocal(S: ArrayLike, atol: float = 1e-9) -> bool:
    arr = np.asarray(S, dtype=np.complex128)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return bool(np.allclose(arr, np.swapaxes(arr, -1, -2), atol=atol, rtol=0))


__all__ = [
    "C0",
    "TransferMatrix",
    "ScatteringMatrix",
    "Cascade",
    "dbcm_to_npm",
    "npm_to_dbcm",
    "propagation_amplitude",
    "identity_transfer",
    "propagation_transfer",
    "waveguide_smatrix",
    "phase_shifter_smatrix",
    "directional_coupler_smatrix",
    "splitter_2x2_smatrix",
    "mzi_smatrix",
    "ring_roundtrip_amplitude",
    "ring_allpass_field",
    "ring_allpass_smatrix",
    "ring_add_drop_fields",
    "fabry_perot_field",
    "cascade_transfer",
    "cascade_two_ports",
    "passivity_margin",
    "is_passive",
    "is_unitary",
    "is_reciprocal",
]
