from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

ArrayLike = Any


def _as_wavelength_array(wavelength_m: ArrayLike) -> tuple[np.ndarray, bool]:
    wl = np.asarray(wavelength_m, dtype=float)
    scalar = wl.ndim == 0
    if scalar:
        wl = wl[None]
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive")
    return wl, scalar


def _maybe_scalar(value: np.ndarray, scalar: bool):
    return value[0] if scalar else value


@dataclass(frozen=True)
class ConstantMaterial:
    """Wavelength-independent optical material.

    Parameters
    ----------
    name:
        Human-readable material name.
    n:
        Real refractive index at ``T0``.
    k:
        Extinction coefficient. The corresponding power absorption is
        ``alpha = 4*pi*k/lambda``.
    dn_dT:
        Linear thermo-optic coefficient in 1/K.
    T0:
        Reference temperature in Celsius.
    extra_alpha:
        Additional power loss in Np/m, useful for sidewall/scattering loss.
    """

    name: str
    n: float
    k: float = 0.0
    dn_dT: float = 0.0
    T0: float = 25.0
    extra_alpha: float = 0.0

    def n_complex(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        nT = float(self.n) + float(self.dn_dT) * (float(T) - float(self.T0))
        out = np.full_like(wl, nT + 1j * float(self.k), dtype=np.complex128)
        return _maybe_scalar(out, scalar)

    def alpha_power(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        k = np.imag(np.asarray(self.n_complex(wl, T=T), dtype=np.complex128))
        alpha = 4.0 * np.pi * np.maximum(k, 0.0) / wl + float(self.extra_alpha)
        return _maybe_scalar(alpha.astype(float), scalar)


@dataclass(frozen=True)
class TabulatedMaterial:
    """Material defined by wavelength-dependent n and optional k tables."""

    name: str
    wavelength_m: ArrayLike
    n: ArrayLike
    k: ArrayLike | None = None
    dn_dT: float = 0.0
    T0: float = 25.0
    extra_alpha: float = 0.0

    def __post_init__(self):
        wl = np.asarray(self.wavelength_m, dtype=float)
        n = np.asarray(self.n, dtype=float)
        if wl.ndim != 1 or n.ndim != 1 or wl.shape != n.shape:
            raise ValueError("wavelength_m and n must be 1D arrays with the same shape")
        if np.any(wl <= 0):
            raise ValueError("wavelength_m must be positive")
        order = np.argsort(wl)
        object.__setattr__(self, "wavelength_m", wl[order])
        object.__setattr__(self, "n", n[order])
        if self.k is not None:
            k = np.asarray(self.k, dtype=float)
            if k.shape != wl.shape:
                raise ValueError("k must match wavelength_m shape")
            object.__setattr__(self, "k", k[order])

    def n_complex(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        n = np.interp(wl, self.wavelength_m, self.n)
        if self.k is None:
            k = np.zeros_like(n)
        else:
            k = np.interp(wl, self.wavelength_m, self.k)
        n = n + float(self.dn_dT) * (float(T) - float(self.T0))
        out = n + 1j * k
        return _maybe_scalar(out.astype(np.complex128), scalar)

    def alpha_power(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        k = np.imag(np.asarray(self.n_complex(wl, T=T), dtype=np.complex128))
        alpha = 4.0 * np.pi * np.maximum(k, 0.0) / wl + float(self.extra_alpha)
        return _maybe_scalar(alpha.astype(float), scalar)


@dataclass(frozen=True)
class FunctionMaterial:
    """Material backed by a user-supplied callable returning complex n(lambda, T)."""

    name: str
    n_complex_fn: Callable[[np.ndarray, float], np.ndarray]
    extra_alpha: float = 0.0

    def n_complex(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        out = np.asarray(self.n_complex_fn(wl, float(T)), dtype=np.complex128)
        if out.shape != wl.shape:
            raise ValueError("n_complex_fn must return an array matching wavelength shape")
        return _maybe_scalar(out, scalar)

    def alpha_power(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        k = np.imag(np.asarray(self.n_complex(wl, T=T), dtype=np.complex128))
        alpha = 4.0 * np.pi * np.maximum(k, 0.0) / wl + float(self.extra_alpha)
        return _maybe_scalar(alpha.astype(float), scalar)


@dataclass(frozen=True)
class RefractiveIndexInfoMaterial:
    """Lazy wrapper for the optional ``refractiveindex`` package.

    This wrapper intentionally imports the package only when evaluated, so the
    core library remains lightweight. The exact database identifiers should
    match the installed refractiveindex.info database, for example
    shelf="main", book="Si", page="Green-2008".
    """

    shelf: str
    book: str
    page: str
    name: str | None = None
    dn_dT: float = 0.0
    T0: float = 25.0
    extra_alpha: float = 0.0

    def _material(self):
        try:
            from refractiveindex import RefractiveIndexMaterial as RIMaterial  # type: ignore
            return RIMaterial(self.shelf, self.book, self.page)
        except Exception as first_error:
            try:
                from refractiveindex import RefractiveIndex  # type: ignore
                db = RefractiveIndex()
                return db.getMaterial(self.shelf, self.book, self.page)
            except Exception as second_error:
                raise ImportError(
                    "Install optional dependency 'refractiveindex' to use "
                    "RefractiveIndexInfoMaterial, or use TabulatedMaterial."
                ) from second_error if first_error else second_error

    def n_complex(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        mat = self._material()
        wl_um = wl * 1e6
        if hasattr(mat, "get_refractive_index"):
            n = np.asarray(mat.get_refractive_index(wl_um), dtype=float)
        elif hasattr(mat, "getRefractiveIndex"):
            n = np.asarray(mat.getRefractiveIndex(wl_um), dtype=float)
        else:
            raise AttributeError("Unsupported refractiveindex material API")
        if hasattr(mat, "get_extinction_coefficient"):
            k = np.asarray(mat.get_extinction_coefficient(wl_um), dtype=float)
        elif hasattr(mat, "getExtinctionCoefficient"):
            try:
                k = np.asarray(mat.getExtinctionCoefficient(wl_um), dtype=float)
            except Exception:
                k = np.zeros_like(n)
        else:
            k = np.zeros_like(n)
        n = n + float(self.dn_dT) * (float(T) - float(self.T0))
        out = n + 1j * k
        return _maybe_scalar(out.astype(np.complex128), scalar)

    def alpha_power(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        k = np.imag(np.asarray(self.n_complex(wl, T=T), dtype=np.complex128))
        alpha = 4.0 * np.pi * np.maximum(k, 0.0) / wl + float(self.extra_alpha)
        return _maybe_scalar(alpha.astype(float), scalar)


@dataclass(frozen=True)
class PyOptikMaterial:
    """Adapter for user-created PyOptik material objects.

    PyOptik versions expose slightly different method names. This adapter tries
    common conventions and raises a clear error if the object is unsupported.
    Prefer passing an already-created PyOptik material object via ``backend``.
    """

    backend: Any
    name: str | None = None
    dn_dT: float = 0.0
    T0: float = 25.0
    extra_alpha: float = 0.0

    def n_complex(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        obj = self.backend
        for meth in ("n_complex", "get_n_complex", "refractive_index", "get_refractive_index", "n"):
            if hasattr(obj, meth):
                val = getattr(obj, meth)(wl)
                break
        else:
            raise AttributeError("PyOptik backend has no recognized refractive-index method")
        nc = np.asarray(val, dtype=np.complex128)
        if nc.shape == ():
            nc = np.full_like(wl, complex(nc), dtype=np.complex128)
        if nc.shape != wl.shape:
            raise ValueError("PyOptik backend returned an incompatible shape")
        nc = nc + float(self.dn_dT) * (float(T) - float(self.T0))
        return _maybe_scalar(nc, scalar)

    def alpha_power(self, wavelength_m: ArrayLike, T: float = 25.0):
        wl, scalar = _as_wavelength_array(wavelength_m)
        k = np.imag(np.asarray(self.n_complex(wl, T=T), dtype=np.complex128))
        alpha = 4.0 * np.pi * np.maximum(k, 0.0) / wl + float(self.extra_alpha)
        return _maybe_scalar(alpha.astype(float), scalar)
