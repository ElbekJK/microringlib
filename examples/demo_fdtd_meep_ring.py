import matplotlib.pyplot as plt

from microringlib.solvers.fdtd_meep import simulate_ring_resonator_2d


def main():
    result = simulate_ring_resonator_2d(
        wavelength_center=1.55e-6,
        wavelength_span=0.08e-6,
        n_core=3.48,
        n_clad=1.444,
        ring_radius=10e-6,
        waveguide_width=500e-9,
        gap=200e-9,
        resolution=25,
        runtime=400,
    )

    plt.figure()
    plt.plot(result.wavelengths * 1e9, result.transmission)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Raw transmitted flux")
    plt.title("MEEP FDTD ring resonator validation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()