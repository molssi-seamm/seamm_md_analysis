seamm_md_analysis
=================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/molssi-seamm/seamm_md_analysis/workflows/CI/badge.svg)](https://github.com/molssi-seamm/seamm_md_analysis/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/molssi-seamm/seamm_md_analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/molssi-seamm/seamm_md_analysis/branch/main)

Analysis of molecular dynamics output for SEAMM.

`seamm_md_analysis` is the shared library behind SEAMM's MD analysis steps
(Diffusivity, Thermal Conductivity, and later Viscosity, Structure and the
Dielectric constant). It holds:

- **statistics:** mean ± error over replicas, block averages, statistical
  inefficiency and equilibration detection (pymbar), and formatting of values
  with their errors;
- **correlation functions** of time series, by FFT;
- **transport coefficients:** collective and single-molecule Helfand moments
  and mean square displacements (by FFT), fits of Green–Kubo integrals and
  Helfand moments, and the Yeh–Hummer finite-size correction;
- **plots** of these for SEAMM's `.graph` files;
- **readers** for the time-series and trajectory files the LAMMPS step writes.

The design is in the SEAMM developer guide, campaign 2026-10-14 ("MD analysis:
H5MD output, a shared library and analysis steps"). This is its phase 1.

### Copyright

Copyright (c) 2026, Paul Saxe

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
