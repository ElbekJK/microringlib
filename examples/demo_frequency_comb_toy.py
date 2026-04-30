import numpy as np
import matplotlib.pyplot as plt

# Simplified split-step LLE-like normalized model.
# This is a qualitative comb demo, not calibrated to a fabricated device.

N = 512
theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
mu = np.fft.fftfreq(N, 1 / N)

dt = 0.005
steps = 6000

alpha = 2.5          # detuning
F = 1.8              # pump
beta2 = -0.015       # anomalous dispersion

A = 0.01 * (np.random.randn(N) + 1j * np.random.randn(N))

linear = -(1 + 1j * alpha) + 1j * beta2 * (mu ** 2)

for step in range(steps):
    A_freq = np.fft.fft(A)
    A_freq *= np.exp(linear * dt / 2)
    A = np.fft.ifft(A_freq)

    A += dt * (1j * np.abs(A) ** 2 * A + F)

    A_freq = np.fft.fft(A)
    A_freq *= np.exp(linear * dt / 2)
    A = np.fft.ifft(A_freq)

spectrum = np.fft.fftshift(np.abs(np.fft.fft(A)) ** 2)
spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)
modes = np.fft.fftshift(mu)

print("\n=== Frequency comb toy demo ===")
print("Qualitative normalized LLE-like simulation.")
print(f"Comb lines above -40 dB: {np.sum(spectrum_db > -40)}")

plt.figure()
plt.plot(theta, np.abs(A) ** 2)
plt.xlabel("Azimuthal angle")
plt.ylabel("Intracavity intensity")
plt.title("Intracavity Pattern")
plt.grid(True)

plt.figure()
plt.stem(modes, spectrum_db, basefmt=" ")
plt.xlabel("Mode number")
plt.ylabel("Power (dB)")
plt.title("Generated Frequency Comb")
plt.ylim(-80, 5)
plt.grid(True)
plt.show()