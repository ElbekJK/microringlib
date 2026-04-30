from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_transmission(wavelength, through, drop=None, labels=None):
    wl = np.asarray(wavelength, dtype=float) * 1e9
    plt.figure()
    plt.plot(wl, through, label=(labels[0] if labels else "Through"))
    if drop is not None:
        if isinstance(drop, list):
            for i, d in enumerate(drop):
                lab = labels[i + 1] if labels and i + 1 < len(labels) else f"Drop {i + 1}"
                plt.plot(wl, d, label=lab)
        else:
            plt.plot(wl, drop, label=(labels[1] if labels and len(labels) > 1 else "Drop"))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power transmission")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mode_profile(mode_field, layers, T=25.0, labels=None):
    mode_field = np.asarray(mode_field)
    if mode_field.ndim == 1:
        mode_field = mode_field[None, :]
    total_thickness = sum(layer.thickness for layer in layers)
    x = np.linspace(0.0, total_thickness, mode_field.shape[-1])
    fig, ax1 = plt.subplots()
    for i in range(mode_field.shape[0]):
        ax1.plot(x * 1e6, np.abs(mode_field[i]), label=(labels[i] if labels and i < len(labels) else f"Mode {i}"))
    ax1.set_xlabel("Position (µm)")
    ax1.set_ylabel("Field amplitude")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    boundaries = np.cumsum([0.0] + [layer.thickness for layer in layers])
    n = np.zeros_like(x)
    for i, layer in enumerate(layers):
        mask = (x >= boundaries[i]) & (x < boundaries[i + 1])
        n[mask] = layer.n_at(T, wavelength_m=1550e-9)
    ax2.plot(x * 1e6, n, "k--", label="n(x)")
    ax2.set_ylabel("Refractive index")
    for b in boundaries[1:-1]:
        ax1.axvline(b * 1e6, linestyle="--", alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.tight_layout()
    plt.show()
