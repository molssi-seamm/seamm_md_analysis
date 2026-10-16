=======
History
=======
2026.10.16 -- Bugfix: fitting a short Green-Kubo integral
    * ``fit_green_kubo_integral`` raised an IndexError for an integral shorter than
      about twice the 0.4 ps window of its moving average. It now raises a ValueError
      saying how many points are needed.

2026.10.15.2 -- Functions for the viscosity
    * ``shear_stress_components`` gives the five independent shear components of a
      pressure tensor, Pxy, Pxz, Pyz, (Pxx-Pyy)/2 and (Pyy-Pzz)/2, which all give the
      viscosity of an isotropic fluid.
    * ``autocorrelation_functions`` and ``component_helfand_moments`` give the
      Green-Kubo and Helfand inputs for independent series such as these. Both can
      leave the mean in, for series whose true mean is zero: subtracting the mean of
      a finite series biases the transport coefficient low.
    * ``publish_viscosity`` and ``preceding_viscosity`` pass a viscosity from the
      Viscosity step to later steps, such as Diffusivity, through the variable
      ``_viscosity``.
    * ``plots.plot_components`` plots curves of any quantity and units, by component.

2026.10.15.1 -- What analyses need from the MD
    * Analysis steps and MD steps can say, before anything runs, what an analysis
      needs (``Requirements``, ``Need``) and what an MD run will sample (``Provides``),
      and ``requirements.check`` reports anything missing or sampled too coarsely.
    * ``select_md_runs`` chooses the earlier MD runs an analysis reads from
      ``_md_runs``: the last run, or the last run of each iteration of a loop.
    * ``step_support`` holds what the analysis steps share to analyze MD run earlier
      in the flowchart, or to run their own: the parameters for the source of the MD,
      the checks of the flowchart, turning sampling on, and finding the files.
    * A new page, "MD steps and analysis steps", describes how they work together.

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

