from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import os
def _patch_numpy_compat_for_modesolverpy() -> None:
    """
    modesolverpy may use old NumPy dtype string aliases such as 'complex_'.

    NumPy 2.x no longer recognizes some of these aliases when passed as dtype
    strings. This lightweight compatibility patch keeps modesolverpy usable
    without requiring users to downgrade NumPy.
    """
    try:
        np.dtype("complex_")
    except TypeError:
        try:
            np.sctypeDict["complex_"] = np.complex128
        except Exception:
            pass

    try:
        np.dtype("float_")
    except TypeError:
        try:
            np.sctypeDict["float_"] = np.float64
        except Exception:
            pass
def _patch_scipy_compat_for_modesolverpy() -> None:
    """
    modesolverpy may call scipy.interpolate.interp2d, which was removed in
    SciPy 1.14.

    This provides a small regular-grid replacement using RectBivariateSpline.
    It is sufficient for modesolverpy's structure.eps_func use case, where
    x/y are regular grid vectors and z has shape (len(y), len(x)).
    """
    try:
        import scipy.interpolate as spi
        from scipy.interpolate import RectBivariateSpline
    except Exception:
        return

    def _compat_interp2d(
        x,
        y,
        z,
        kind="linear",
        copy=True,
        bounds_error=False,
        fill_value=None,
        *args,
        **kwargs,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z)

        if z.shape != (y.size, x.size):
            z = np.asarray(z).reshape(y.size, x.size)

        degree = {
            "linear": 1,
            "cubic": 3,
            "quintic": 5,
        }.get(kind, 1)

        kx = min(degree, max(1, y.size - 1))
        ky = min(degree, max(1, x.size - 1))

        spline_real = RectBivariateSpline(
            y,
            x,
            np.real(z),
            kx=kx,
            ky=ky,
        )

        if np.iscomplexobj(z):
            spline_imag = RectBivariateSpline(
                y,
                x,
                np.imag(z),
                kx=kx,
                ky=ky,
            )
        else:
            spline_imag = None

        def _call(x_new, y_new):
            x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
            y_new = np.atleast_1d(np.asarray(y_new, dtype=float))

            out = spline_real(y_new, x_new)

            if spline_imag is not None:
                out = out + 1j * spline_imag(y_new, x_new)

            return out

        return _call

    # SciPy >=1.14 still exposes interp2d as a stub that raises
    # NotImplementedError. Replace it unconditionally for modesolverpy.
    spi.interp2d = _compat_interp2d
from .common import require_module


@dataclass
class FDEResult:
    wavelength: float
    neff: complex
    ng: float | None = None
    mode_index: int = 0
    backend: str = "modesolverpy"
    raw: object | None = None


def is_available() -> bool:
    try:
        require_module("modesolverpy", "pip install modesolverpy")
        return True
    except ImportError:
        return False


def _fallback_effective_index(
    *,
    wavelength: float,
    core_width: float,
    core_thickness: float,
    n_core: float,
    n_clad: float,
) -> complex:
    """
    Lightweight fallback approximation.

    This is not a replacement for FDE. It exists so the optional demo can remain
    import- and smoke-testable across modesolverpy API versions.
    """
    wl = float(wavelength)
    w = float(core_width)
    h = float(core_thickness)

    if wl <= 0 or w <= 0 or h <= 0:
        raise ValueError("wavelength, core_width, and core_thickness must be positive")
    if n_core <= n_clad:
        raise ValueError("n_core should be larger than n_clad for a guided waveguide")

    confinement_w = 1.0 - np.exp(-2.0 * w / wl)
    confinement_h = 1.0 - np.exp(-2.0 * h / wl)
    confinement = np.clip(0.5 * (confinement_w + confinement_h), 0.0, 1.0)

    neff = n_clad + confinement * (n_core - n_clad)

    # Keep result inside physical bounds.
    neff = float(np.clip(neff, n_clad + 1e-6, n_core - 1e-6))
    return complex(neff, 0.0)

def _patch_scipy_numpy_aliases_for_modesolverpy() -> None:
    """
    modesolverpy may use old top-level scipy aliases such as scipy.sqrt.

    Modern SciPy removed many NumPy re-exports from the scipy namespace.
    This patch restores the small set commonly used by older modesolverpy code.
    """
    try:
        import scipy
    except Exception:
        return

    aliases = {
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "pi": np.pi,
        "absolute": np.absolute,
        "abs": np.abs,
        "real": np.real,
        "imag": np.imag,
        "conj": np.conj,
        "conjugate": np.conjugate,
        "zeros": np.zeros,
        "ones": np.ones,
        "empty": np.empty,
        "array": np.array,
        "asarray": np.asarray,
        "arange": np.arange,
        "linspace": np.linspace,
        "meshgrid": np.meshgrid,
        "where": np.where,
        "isnan": np.isnan,
        "isfinite": np.isfinite,
        "complex128": np.complex128,
        "float64": np.float64,
    }

    for name, value in aliases.items():
        if not hasattr(scipy, name):
            setattr(scipy, name, value)
            


def _make_modesolver(cls, **kwargs):
    """Instantiate a modesolverpy solver while tolerating API differences."""
    import inspect
    sig = inspect.signature(cls)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**allowed)


def _finite_guided_neff(neffs, *, n_clad: float, n_core: float, mode_index: int) -> complex:
    arr = np.asarray(neffs).reshape(-1)
    arr = arr[np.isfinite(np.real(arr)) & np.isfinite(np.imag(arr))]
    if arr.size == 0:
        raise RuntimeError("modesolverpy returned no finite effective indices")

    # Keep guided candidates.  The upper margin allows weak numerical overshoot.
    guided = arr[(np.real(arr) > float(n_clad) + 1e-5) & (np.real(arr) < float(n_core) + 0.25)]
    if guided.size == 0:
        guided = arr

    # Fundamental TE-like solutions are normally the largest real effective index.
    guided = guided[np.argsort(np.real(guided))[::-1]]
    if guided.size <= mode_index:
        raise RuntimeError(
            f"Requested mode_index={mode_index}, but only {guided.size} finite guided modes were found."
        )
    return complex(guided[mode_index])


def _build_ridge_waveguide(
    structure,
    *,
    wavelength: float,
    core_width: float,
    core_thickness: float,
    n_core: float,
    n_clad: float,
    n_substrate: float,
    n_top: float,
    x_span: float,
    y_span: float,
    dx: float,
    dy: float,
):
    wl_um = float(wavelength) * 1e6
    width_um = float(core_width) * 1e6
    height_um = float(core_thickness) * 1e6
    x_step_um = float(dx) * 1e6
    y_step_um = float(dy) * 1e6
    x_total_um = float(x_span) * 1e6
    y_total_um = float(y_span) * 1e6

    # Keep enough vertical buffer above/below the film.  Very tight windows can
    # make modesolverpy's finite-difference stencil singular on modern SciPy.
    sub_height_um = max(y_total_um / 2.0, 1.0)
    clad_height_um = max(y_total_um / 2.0, 1.0)
    sub_width_um = max(x_total_um, width_um + 2.0)

    return structure.RidgeWaveguide(
        wavelength=wl_um,
        x_step=x_step_um,
        y_step=y_step_um,
        wg_height=height_um,
        wg_width=width_um,
        sub_height=sub_height_um,
        sub_width=sub_width_um,
        clad_height=clad_height_um,
        n_sub=n_substrate,
        n_wg=n_core,
        n_clad=n_top,
        film_thickness=height_um,
    )


def _try_modesolverpy_once(
    *,
    mode_solver,
    structure,
    wavelength: float,
    core_width: float,
    core_thickness: float,
    n_core: float,
    n_clad: float,
    n_substrate: float,
    n_top: float,
    x_span: float,
    y_span: float,
    dx: float,
    dy: float,
    mode_index: int,
    boundary: str,
    tol: float,
    solver_kind: str,
):
    wg = _build_ridge_waveguide(
        structure,
        wavelength=wavelength,
        core_width=core_width,
        core_thickness=core_thickness,
        n_core=n_core,
        n_clad=n_clad,
        n_substrate=n_substrate,
        n_top=n_top,
        x_span=x_span,
        y_span=y_span,
        dx=dx,
        dy=dy,
    )

    n_eigs = max(mode_index + 2, 4)
    tried = []

    if solver_kind in ("auto", "fully-vectorial", "full"):
        cls = getattr(mode_solver, "ModeSolverFullyVectorial", None)
        if cls is not None:
            tried.append(("modesolverpy-fully-vectorial", cls, {
                "n_eigs": n_eigs,
                "tol": tol,
                "boundary": boundary,
            }))

    if solver_kind in ("auto", "semi-vectorial", "semi"):
        cls = getattr(mode_solver, "ModeSolverSemiVectorial", None)
        if cls is not None:
            # modesolverpy versions disagree on the keyword name.  _make_modesolver
            # filters unsupported keywords by introspection.
            tried.append(("modesolverpy-semi-vectorial-Ex", cls, {
                "n_eigs": n_eigs,
                "tol": tol,
                "boundary": boundary,
                "semi_vectorial_method": "Ex",
                "method": "Ex",
            }))
            tried.append(("modesolverpy-semi-vectorial-Ey", cls, {
                "n_eigs": n_eigs,
                "tol": tol,
                "boundary": boundary,
                "semi_vectorial_method": "Ey",
                "method": "Ey",
            }))

    last_exc = None
    for backend, cls, kwargs in tried:
        try:
            solver = _make_modesolver(cls, **kwargs)
            solver.solve(wg)
            neff = _finite_guided_neff(
                getattr(solver, "n_effs", []),
                n_clad=n_clad,
                n_core=n_core,
                mode_index=mode_index,
            )
            return neff, backend, solver
        except Exception as exc:  # pragma: no cover - depends on external solver stability
            last_exc = exc
            continue

    if last_exc is None:
        raise RuntimeError("No compatible modesolverpy solver class was found")
    raise last_exc

def solve_waveguide_mode(
    *,
    wavelength: float,
    core_width: float,
    core_thickness: float,
    n_core: float,
    n_clad: float = 1.444,
    n_substrate: float | None = None,
    n_top: float | None = None,
    x_span: float = 4.0e-6,
    y_span: float = 4.0e-6,
    dx: float = 20e-9,
    dy: float = 20e-9,
    mode_index: int = 0,
    polarization: str = "quasi-TE",
    allow_fallback: bool = True,
    solver_kind: str = "auto",
) -> FDEResult:
    """
    Solve or estimate a rectangular waveguide mode.

    Parameters are SI units.  In real-FDE mode the function tries multiple
    modesolverpy configurations before giving up.  This is necessary because
    modesolverpy/ARPACK can be sensitive to grid spacing, boundary window, and
    SciPy version.  When allow_fallback=False, no analytic effective-index
    surrogate is returned; failures are raised to the caller.
    """
    n_substrate = n_clad if n_substrate is None else n_substrate
    n_top = n_clad if n_top is None else n_top

    try:
        _patch_numpy_compat_for_modesolverpy()
        _patch_scipy_compat_for_modesolverpy()
        _patch_scipy_numpy_aliases_for_modesolverpy()

        require_module("modesolverpy", "pip install modesolverpy")
        from modesolverpy import mode_solver, structure

        # Ordered from accurate/default to more robust.  The semi-vectorial
        # retry is still a real finite-difference eigenmode solve, not a
        # surrogate.  It is useful for high-throughput verification sweeps.
        #
        # Some modesolverpy/SciPy/BLAS builds can abort the Python process in
        # the fully-vectorial ARPACK branch before Python can catch an exception
        # (usually after a LAPACK ZLASCL warning).  For publication sweeps we
        # therefore support a safe semi-vectorial-only mode.  This still uses
        # a real FDE eigenmode solve, but avoids the fragile vectorial branch.
        safe_semi = os.environ.get("MICRORINGLIB_FDE_SAFE_SEMIVECTORIAL", "1") == "1"
        if safe_semi:
            retry_configs = [
                dict(dx=max(dx, 40e-9), dy=max(dy, 40e-9), x_span=max(x_span, 4.0e-6), y_span=max(y_span, 4.0e-6), boundary="0000", tol=1e-6, solver_kind="semi-vectorial"),
                dict(dx=max(dx, 50e-9), dy=max(dy, 50e-9), x_span=max(x_span, 5.0e-6), y_span=max(y_span, 5.0e-6), boundary="0000", tol=1e-5, solver_kind="semi-vectorial"),
                dict(dx=max(dx, 70e-9), dy=max(dy, 70e-9), x_span=max(x_span, 6.0e-6), y_span=max(y_span, 6.0e-6), boundary="0000", tol=1e-5, solver_kind="semi-vectorial"),
            ]
        else:
            retry_configs = [
                dict(dx=dx, dy=dy, x_span=x_span, y_span=y_span, boundary="0000", tol=1e-7, solver_kind=solver_kind),
                dict(dx=max(dx, 30e-9), dy=max(dy, 30e-9), x_span=max(x_span, 4.0e-6), y_span=max(y_span, 4.0e-6), boundary="0000", tol=1e-6, solver_kind=solver_kind),
                dict(dx=max(dx, 50e-9), dy=max(dy, 50e-9), x_span=max(x_span, 5.0e-6), y_span=max(y_span, 5.0e-6), boundary="0000", tol=1e-5, solver_kind="auto"),
                dict(dx=max(dx, 60e-9), dy=max(dy, 60e-9), x_span=max(x_span, 6.0e-6), y_span=max(y_span, 6.0e-6), boundary="0000", tol=1e-5, solver_kind="semi-vectorial"),
            ]

        last_exc = None
        for cfg in retry_configs:
            try:
                neff, backend, solver = _try_modesolverpy_once(
                    mode_solver=mode_solver,
                    structure=structure,
                    wavelength=wavelength,
                    core_width=core_width,
                    core_thickness=core_thickness,
                    n_core=n_core,
                    n_clad=n_clad,
                    n_substrate=n_substrate,
                    n_top=n_top,
                    mode_index=mode_index,
                    **cfg,
                )
                return FDEResult(
                    wavelength=float(wavelength),
                    neff=neff,
                    ng=None,
                    mode_index=mode_index,
                    backend=backend,
                    raw=solver,
                )
            except Exception as exc:  # pragma: no cover - external numerical branch
                last_exc = exc
                continue

        raise RuntimeError("all modesolverpy retry configurations failed") from last_exc

    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "modesolverpy could not solve the waveguide mode after multiple real-FDE retries. "
                "Try a coarser grid, a larger simulation window, Python 3.11/3.12, or allow_fallback=True."
            ) from exc

        neff = _fallback_effective_index(
            wavelength=wavelength,
            core_width=core_width,
            core_thickness=core_thickness,
            n_core=n_core,
            n_clad=n_clad,
        )

        return FDEResult(
            wavelength=float(wavelength),
            neff=neff,
            ng=None,
            mode_index=mode_index,
            backend="effective-index-fallback",
            raw=None,
        )


def solve_dispersion(
    *,
    wavelengths: Sequence[float],
    core_width: float,
    core_thickness: float,
    n_core: float,
    n_clad: float = 1.444,
    mode_index: int = 0,
    **kwargs,
) -> list[FDEResult]:
    """
    Solve or estimate neff over a wavelength sweep.

    Returns a list of FDEResult objects.
    """
    results: list[FDEResult] = []

    for wl in wavelengths:
        results.append(
            solve_waveguide_mode(
                wavelength=float(wl),
                core_width=core_width,
                core_thickness=core_thickness,
                n_core=n_core,
                n_clad=n_clad,
                mode_index=mode_index,
                **kwargs,
            )
        )

    return results
