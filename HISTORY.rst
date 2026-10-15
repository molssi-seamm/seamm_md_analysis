=======
History
=======
2026.10.15 -- SEAMM's H5MD files
    * Writes and reads the H5MD files (``md.h5md``) in which the MD steps store each
      run: the time series, the trajectory, the box, the run's parameters and SEAMM's
      provenance. The layout is described in the new page "The H5MD file".
    * ``open_md`` reads a run, or the list of its segments, and returns the time series
      on their own time axes with their units, streams the frames of a trajectory, and
      builds an MDAnalysis Universe for structural analysis (``seamm-md-analysis[structure]``).
    * Translates H5MD unit strings to and from pint.
    * Streams LAMMPS dump files (gzipped or not, with triclinic boxes) and the LAMMPS
      step's per-molecule files.
    * Needs h5py and pint, which every SEAMM installation already has.

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

