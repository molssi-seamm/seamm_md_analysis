=======
History
=======
2026.10.14 -- Initial release: the shared MD analysis library
    * Statistics (mean ± error over replicas, block averages, statistical
      inefficiency and equilibration detection with pymbar, formatting values with
      their errors), correlation functions by FFT, collective and single-molecule
      Helfand moments and mean square displacements by FFT, fits of Green–Kubo
      integrals and Helfand moments, the Yeh–Hummer correction, plots for SEAMM
      .graph files, and readers for the LAMMPS step's time-series, vector and dump
      files.
    * Gathers the code that the Diffusivity and Thermal Conductivity steps each
      had, so both can use one copy. The Helfand moments and MSDs give the same
      results as before, much faster.

