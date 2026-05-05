from __future__ import annotations

import numpy as np

from .models import Layer, ModeResult
from .utils import layers_signature

_MODE_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def solve_waveguide_modes(
    wavelength,
    layers: list[Layer],
    T: float,
    polarization: str = "TE",
    num_modes: int = 1,
    dx: float = 10e-9,
    disable_cache: bool = False,
) -> ModeResult:
    """
    1D prototype mode solver for vertically layered waveguides.

    This version uses a real symmetric finite-difference operator on the
    interior grid only, which is much more stable than the previous PML-based
    complex eigenproblem for this use case.
    """
    if not layers:
        raise ValueError("layers cannot be empty")
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if polarization not in {"TE", "TM"}:
        raise ValueError("polarization must be 'TE' or 'TM'")

    wl = np.asarray(wavelength, dtype=float)
    scalar = wl.ndim == 0
    if scalar:
        wl = wl[None]
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive")

    boundaries = np.cumsum([0.0] + [layer.thickness for layer in layers])
    total_thickness = float(boundaries[-1])
    if total_thickness <= 0:
        raise ValueError("total layer thickness must be positive")

    n_grid = max(5, int(np.ceil(total_thickness / dx)) + 1)
    x = np.linspace(0.0, total_thickness, n_grid)
    dx = float(x[1] - x[0])

    sig = (
        layers_signature(layers),
        float(T),
        polarization,
        int(num_modes),
        float(dx),
    )

    out_field = np.zeros((len(wl), num_modes, n_grid), dtype=np.complex128)
    out_neff = np.zeros((len(wl), num_modes), dtype=float)

    n_max_phys = max(float(np.real(layer.n_complex(wl, T=T)).max()) for layer in layers) + 1e-6

    for idx, lam in enumerate(wl):
        cache_key = (*sig, float(lam))
        if not disable_cache and cache_key in _MODE_CACHE:
            f, neff = _MODE_CACHE[cache_key]
            out_field[idx] = f
            out_neff[idx] = neff
            continue

        k0 = 2.0 * np.pi / lam

        # refractive-index profile on the full 1D stack
        n_full = np.zeros(n_grid, dtype=float)
        for j, layer in enumerate(layers):
            left, right = boundaries[j], boundaries[j + 1]
            if j == len(layers) - 1:
                mask = (x >= left) & (x <= right)
            else:
                mask = (x >= left) & (x < right)
            n_full[mask] = layer.n_at(T, wavelength_m=float(lam))

        # solve only on interior points; boundaries are Dirichlet (field=0)
        n_int = n_full[1:-1]
        N = len(n_int)
        if N < 3:
            raise RuntimeError("grid too small for mode solve")

        k_search = min(max(num_modes + 6, 8), N - 1)
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        if polarization == "TE":
            # TE scalar equation:
            #   d^2E/dx^2 + (k0^2 n^2 - beta^2) E = 0
            # => (D2 + k0^2 n^2) E = beta^2 E
            main = -2.0 / dx**2 + (k0 * n_int) ** 2
            off = np.full(N - 1, 1.0 / dx**2, dtype=float)

            A = sp.diags(
                diagonals=[off, main, off],
                offsets=[-1, 0, 1],
                shape=(N, N),
                format="csr",
            )

            eigvals, eigvecs = spla.eigsh(A, k=k_search, which="LA")

        else:
            # TM Hy equation with harmonic-averaged inverse permittivity:
            #   d/dx((1/eps) dH/dx) + k0^2 H = beta^2 (1/eps) H
            # This enforces the correct dielectric-interface weighting instead
            # of reusing the TE operator.
            eps = n_int ** 2
            inv_eps = 1.0 / eps
            inv_half = 2.0 * inv_eps[:-1] * inv_eps[1:] / (inv_eps[:-1] + inv_eps[1:])
            left = np.empty(N, dtype=float)
            right = np.empty(N, dtype=float)
            left[0] = inv_eps[0]
            left[1:] = inv_half
            right[-1] = inv_eps[-1]
            right[:-1] = inv_half
            main = -(left + right) / dx**2 + k0**2
            off = inv_half / dx**2
            A = sp.diags([off, main, off], [-1, 0, 1], shape=(N, N), format="csr")
            M = sp.diags(inv_eps, 0, shape=(N, N), format="csr")
            eigvals, eigvecs = spla.eigsh(A, M=M, k=k_search, which="LA")

        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0.0)
        beta = np.sqrt(eigvals)
        neff = beta / k0

        # Keep only physically plausible guided modes
        valid = np.where((neff > 1.0) & (neff < n_max_phys))[0]
        if valid.size == 0:
            raise RuntimeError(
                f"No physical guided mode found at wavelength={lam:.3e} m. "
                f"Candidate n_eff={neff}, physical upper bound={n_max_phys:.6f}"
            )

        valid = valid[np.argsort(neff[valid])[::-1]]

        if valid.size < num_modes:
            raise RuntimeError(
                f"Only {valid.size} physical mode(s) found at wavelength={lam:.3e} m, "
                f"but num_modes={num_modes} was requested. Candidate n_eff={neff}"
            )

        valid = valid[:num_modes]
        neff = neff[valid]
        eigvecs = eigvecs[:, valid]

        field = np.zeros((n_grid, num_modes), dtype=np.complex128)
        field[1:-1, :] = eigvecs

        for m in range(num_modes):
            p = np.trapezoid(np.abs(field[:, m]) ** 2, x=x)
            if p <= 0:
                raise RuntimeError("mode normalization failed")
            field[:, m] /= np.sqrt(p)

        out_field[idx] = field.T
        out_neff[idx] = neff

        if not disable_cache:
            _MODE_CACHE[cache_key] = (field.T.copy(), neff.copy())

    if scalar:
        return ModeResult(
            wavelength=wl,
            field=out_field[0],
            n_eff=out_neff[0],
            polarization=polarization,
            x=x,
            metadata={"solver": "finite_difference_1d"},
        )

    return ModeResult(
        wavelength=wl,
        field=out_field,
        n_eff=out_neff,
        polarization=polarization,
        x=x,
        metadata={"solver": "finite_difference_1d"},
    )


def compute_group_index(wavelength: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelength, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)

    if wl.shape != n_eff.shape:
        raise ValueError("wavelength and n_eff must have the same shape")
    if wl.size < 2:
        raise ValueError("need at least two wavelength samples")

    order = np.argsort(wl)
    wl = wl[order]
    n_eff = n_eff[order]

    if np.any(np.diff(wl) <= 0):
        raise ValueError("wavelength must be strictly increasing")

    dn_dwl = np.gradient(n_eff, wl)
    return n_eff - wl * dn_dwl