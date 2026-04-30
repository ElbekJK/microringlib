from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union
import numpy as np

ScalarOrArray = Union[float, np.ndarray]
AlphaType = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True, init=False)
class Layer:
    """One vertical material layer in a waveguide stack.

    Backward compatible usage:
        Layer("Si", 220e-9, 3.476, dn_dT=1.86e-4, alpha=...)

    Physics-first material usage:
        Layer("Si", 220e-9, material_model=ConstantMaterial("Si", 3.476))
        Layer("Si", 220e-9, material_model=TabulatedMaterial(...))

    ``alpha`` is an additional power-loss term in Np/m. If the material model
    provides an extinction coefficient ``k``, the total loss is
    ``4*pi*k/lambda + alpha``.
    """

    material: str
    thickness: float
    n: float | None = None
    dn_dT: float = 0.0
    alpha: AlphaType = 0.0
    material_model: object | None = None

    def __init__(
        self,
        material: str,
        thickness: float,
        n: float | None = None,
        dn_dT: float = 0.0,
        alpha: AlphaType = 0.0,
        material_model: object | None = None,
    ) -> None:
        object.__setattr__(self, "material", str(material))
        object.__setattr__(self, "thickness", float(thickness))
        if float(thickness) <= 0:
            raise ValueError("layer thickness must be positive")
        if material_model is None and n is None:
            raise ValueError("Layer requires either n or material_model")
        object.__setattr__(self, "n", None if n is None else float(n))
        object.__setattr__(self, "dn_dT", float(dn_dT))
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "material_model", material_model)

    def n_complex(self, wavelength_m, T: float = 25.0, T_ref: float = 25.0):
        wl = np.asarray(wavelength_m, dtype=float)
        scalar = wl.ndim == 0
        if scalar:
            wl = wl[None]
        if np.any(wl <= 0):
            raise ValueError("wavelength must be positive")
        if self.material_model is not None:
            if not hasattr(self.material_model, "n_complex"):
                raise TypeError("material_model must define n_complex(wavelength_m, T=...)")
            nc = np.asarray(self.material_model.n_complex(wl, T=T), dtype=np.complex128)
            if nc.shape == ():
                nc = np.full_like(wl, complex(nc), dtype=np.complex128)
            if nc.shape != wl.shape:
                raise ValueError("material_model.n_complex returned an incompatible shape")
        else:
            nT = float(self.n) + self.dn_dT * (float(T) - float(T_ref))
            nc = np.full_like(wl, nT, dtype=np.complex128)
        return nc[0] if scalar else nc

    def n_at(self, T: float, T_ref: float = 25.0, wavelength_m: float = 1550e-9) -> float:
        return float(np.real(self.n_complex(wavelength_m, T=T, T_ref=T_ref)))

    @staticmethod
    def dbcm_to_npm(alpha_db_cm: float) -> float:
        return float(alpha_db_cm * np.log(10.0) / 10.0 * 100.0)

    @staticmethod
    def npm_to_dbcm(alpha_npm: float) -> float:
        return float(alpha_npm * 10.0 / np.log(10.0) / 100.0)

    def alpha_at(self, x: Optional[np.ndarray] = None) -> np.ndarray | float:
        if callable(self.alpha):
            if x is None:
                raise ValueError("x must be provided when alpha is callable")
            return np.asarray(self.alpha(np.asarray(x)))
        return self.alpha

    def alpha_power(self, wavelength_m, T: float = 25.0) -> np.ndarray | float:
        wl = np.asarray(wavelength_m, dtype=float)
        scalar = wl.ndim == 0
        if scalar:
            wl = wl[None]
        if np.any(wl <= 0):
            raise ValueError("wavelength must be positive")
        alpha_extra = np.asarray(self.alpha_at(x=wl), dtype=float)
        if alpha_extra.shape == ():
            alpha_extra = np.full_like(wl, float(alpha_extra), dtype=float)
        if alpha_extra.shape != wl.shape:
            raise ValueError("alpha array must match wavelength shape")
        if np.any(alpha_extra < 0):
            raise ValueError("Loss coefficient must be non-negative")
        if self.material_model is not None and hasattr(self.material_model, "alpha_power"):
            alpha_model = np.asarray(self.material_model.alpha_power(wl, T=T), dtype=float)
            if alpha_model.shape == ():
                alpha_model = np.full_like(wl, float(alpha_model), dtype=float)
            if alpha_model.shape != wl.shape:
                raise ValueError("material_model.alpha_power returned an incompatible shape")
        else:
            nc = np.asarray(self.n_complex(wl, T=T), dtype=np.complex128)
            alpha_model = 4.0 * np.pi * np.maximum(np.imag(nc), 0.0) / wl
        out = alpha_model + alpha_extra
        return float(out[0]) if scalar else out

    def effective_alpha(self, confinement: float = 1.0, x: Optional[np.ndarray] = None) -> np.ndarray | float:
        if confinement < 0:
            raise ValueError("confinement must be non-negative")
        return confinement * self.alpha_at(x=x)

    def transmission(self, length: float, confinement: float = 1.0, x: Optional[np.ndarray] = None) -> np.ndarray | float:
        alpha_eff = self.effective_alpha(confinement=confinement, x=x)
        return np.exp(-np.asarray(alpha_eff) * length)


@dataclass(frozen=True, init=False)
class RingGeometry:
    """Geometry descriptor for microring resonators.

    Defaults to a circular ring, so ``RingGeometry(radius=10e-6)`` works.
    Backward-compatible calls such as ``RingGeometry(kind="circular", radius=...)``
    and ``RingGeometry("circular", radius=...)`` are also supported.
    """

    radius: Optional[float] = None
    kind: Literal["circular", "elliptical", "racetrack"] = "circular"
    a: Optional[float] = None
    b: Optional[float] = None
    straight_length: Optional[float] = None

    def __init__(
        self,
        *args,
        radius: Optional[float] = None,
        kind: Literal["circular", "elliptical", "racetrack"] = "circular",
        a: Optional[float] = None,
        b: Optional[float] = None,
        straight_length: Optional[float] = None,
    ) -> None:
        if len(args) > 2:
            raise TypeError("RingGeometry accepts at most two positional arguments")
        if args:
            first = args[0]
            if isinstance(first, str):
                kind = first
                if len(args) == 2:
                    if radius is not None:
                        raise TypeError("radius was provided both positionally and by keyword")
                    radius = float(args[1])
            else:
                if radius is not None:
                    raise TypeError("radius was provided both positionally and by keyword")
                radius = float(first)
                if len(args) == 2:
                    kind = args[1]

        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "straight_length", straight_length)


@dataclass(frozen=True)
class Coupler:
    """Lossless directional coupler using a fixed unitary phase convention."""

    t: complex
    kappa: complex | None = None

    def __post_init__(self) -> None:
        # Preserve historical behavior where validate_lossless() can be called
        # explicitly on obviously invalid |t| > 1 objects in tests/scripts.
        if abs(self.t) > 1.0 + 1e-12:
            return
        k_mag_sq = 1.0 - abs(self.t) ** 2
        if self.kappa is None:
            object.__setattr__(self, "kappa", 1j * np.exp(1j * np.angle(self.t)) * float(np.sqrt(max(0.0, k_mag_sq))))
        else:
            if not np.isclose(abs(self.t) ** 2 + abs(self.kappa) ** 2, 1.0, atol=1e-9):
                return
            S = np.array([[self.t, self.kappa], [self.kappa, self.t]], dtype=np.complex128)
            if not np.allclose(S.conj().T @ S, np.eye(2), atol=1e-8):
                # Preserve the requested coupling magnitude but enforce the unitary phase convention.
                object.__setattr__(self, "kappa", 1j * np.exp(1j * np.angle(self.t)) * abs(self.kappa))
        self.validate_lossless()

    @classmethod
    def from_power_coupling(cls, K: float, through_phase: float = 0.0, cross_phase: float = np.pi / 2) -> "Coupler":
        """Construct a unitary coupler from power coupling K = |kappa|^2."""
        K = float(K)
        if K < 0.0 or K > 1.0:
            raise ValueError("power coupling K must be in [0, 1]")
        t = np.sqrt(1.0 - K) * np.exp(1j * through_phase)
        kappa = np.sqrt(K) * np.exp(1j * cross_phase)
        return cls(t=t, kappa=kappa)

    @property
    def power_coupling(self) -> float:
        return float(abs(self.kappa) ** 2)

    @property
    def through_power(self) -> float:
        return float(abs(self.t) ** 2)

    @property
    def scattering_matrix(self) -> np.ndarray:
        return np.array([[self.t, self.kappa], [self.kappa, self.t]], dtype=np.complex128)

    def validate_lossless(self, atol: float = 1e-9) -> None:
        if abs(self.t) > 1.0 + atol:
            raise ValueError("Coupler transmission magnitude must be <= 1")
        if self.kappa is None:
            raise ValueError("kappa must be internally constructed before validation")
        if not np.isclose(abs(self.t) ** 2 + abs(self.kappa) ** 2, 1.0, atol=atol):
            raise ValueError("Lossless coupler must satisfy |t|^2 + |kappa|^2 = 1")
        S = self.scattering_matrix
        ident = S.conj().T @ S
        # The symmetric two-port convention is unitary when t*conj(k)+k*conj(t)=0.
        if not np.allclose(ident, np.eye(2), atol=1e-8):
            raise ValueError("Coupler scattering matrix is not unitary; use a ±pi/2 cross phase")


@dataclass
class ModeResult:
    wavelength: np.ndarray
    field: np.ndarray
    n_eff: np.ndarray
    polarization: str
    x: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TransmissionResult:
    wavelength: np.ndarray
    field: np.ndarray
    power: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def ports(self) -> dict:
        names = self.metadata.get("ports")
        if not names:
            return {}
        return {
            name: {"field": self.field[i], "power": self.power[i]}
            for i, name in enumerate(names)
        }
