from __future__ import annotations
from hashlib import md5
import inspect
import numpy as np
from .models import Layer, AlphaType

def _alpha_key(alpha: AlphaType):
    if callable(alpha):
        try:
            return inspect.getsource(alpha)
        except Exception:
            return f"callable:{id(alpha)}"
    if isinstance(alpha, np.ndarray):
        return ("array", alpha.shape, alpha.dtype.str, md5(alpha.tobytes()).hexdigest())
    return alpha

def _material_model_key(model):
    if model is None:
        return None
    try:
        return (model.__class__.__module__, model.__class__.__qualname__, repr(model))
    except Exception:
        return (model.__class__.__module__, model.__class__.__qualname__, id(model))


def layers_signature(layers: list[Layer]) -> str:
    data = [
        (
            l.material,
            float(l.thickness),
            None if l.n is None else float(l.n),
            float(l.dn_dT),
            _alpha_key(l.alpha),
            _material_model_key(getattr(l, "material_model", None)),
        )
        for l in layers
    ]
    return md5(repr(data).encode()).hexdigest()

def evaluate_alpha(alpha: AlphaType, wl: np.ndarray) -> np.ndarray:
    wl = np.asarray(wl, dtype=float)
    if callable(alpha):
        val = np.asarray(alpha(wl), dtype=float)
    elif np.isscalar(alpha):
        val = np.full_like(wl, float(alpha), dtype=float)
    else:
        val = np.asarray(alpha, dtype=float)
        if val.shape != wl.shape:
            raise ValueError("Alpha array must match wavelength shape")
    if np.any(val < 0):
        raise ValueError("Loss coefficient must be non-negative")
    return val
