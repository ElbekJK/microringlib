# Relationship to other photonics tools

`microringlib` is designed as a lightweight, physics-first microring modeling package.
It is complementary to larger photonic design and simulation tools.

| Tool | Main strength | Relationship |
| --- | --- | --- |
| gdsfactory | Python layout, PDK, GDS workflows | Complementary layout-first ecosystem |
| Meep | Open-source FDTD electromagnetic simulation | Higher-fidelity full-wave solver |
| Tidy3D | Python/cloud FDTD workflow | Higher-fidelity commercial/cloud solver |
| SAX / Simphony | S-parameter circuit simulation | More general PIC circuit frameworks |
| microringlib | Ring-focused physics and fast sweeps | Interpretable design exploration layer |

Use `microringlib` early in design for physical insight, sweeps, and teaching; use full-wave solvers and layout tools for final geometry validation and fabrication workflows.
