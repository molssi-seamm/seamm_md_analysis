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
    * Corrects defects in that code: the Green–Kubo fit could accept a second
      exponential with a negative amplitude or a time constant beyond the data,
      and gave huge errors when its two time constants were nearly equal; fits
      with zero errors, as for a single run, failed; the slope of a Helfand
      moment was shifted by one sample and its error had the wrong units; and
      the total single-molecule Helfand moment included cross terms between
      directions.

