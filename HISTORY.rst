=======
History
=======
2026.10.17.3 -- Bugfix: nested loops mixed the runs of different systems
    * With a loop over systems around a loop over replicas, "the last MD run of each
      loop iteration" also took the replicas of the systems before: the directories
      of all loops' iterations were treated as the same step. Only the innermost
      loop's iterations are now, so each system's analysis takes its own replicas.

2026.10.17.2 -- Results reported one way: labelled errors, isotropic values, the state
    * ``Estimate``: a value with its standard error over independent runs or blocks;
      its 95% confidence interval uses Student's t for the number of samples (2.78
      for 5 runs, not 2), and it says what its ± is, for a note under a table.
    * ``isotropic_average`` gives one value for a property from its components,
      averaged within each run, with the error of those averages; ``component_check``
      tells whether the components agree (χ²), as a one-line diagnostic.
    * A new module, ``state``: the temperature, pressure, volume and density of runs
      after their equilibration, with errors; for runs at constant volume the
      equilibrium density estimated from the bulk modulus of an NPT run before them
      (``bulk_modulus``, from its volume fluctuations); ``state_table`` for the
      output; and ``isotropy``, whether a run can be treated as isotropic (a cubic
      cell and nothing directional in it).
    * ``step_support.state_of_runs``: the state of the runs an analysis step read,
      using the NPT run before them in the flowchart for the bulk modulus.

2026.10.17.1 -- Fixes offered when the MD does not sample what an analysis needs
    * When the MD before an analysis step does not sample what it needs, or not often
      enough, the problem found before the job runs now comes with a fix that turns
      the sampling on in the MD step, for MD steps that can give one (LAMMPS
      2026.10.17). The editor offers it as a button; ``seamm-flowchart validate
      --fix`` applies it.

2026.10.17 -- Bugfix: only the last loop iteration was analyzed
    * With "the last MD run of each loop iteration", the analysis steps (Viscosity,
      Diffusivity, Liquid Structure, Dielectric Constant, Ionic Conductivity) used only
      the run of the last iteration, because a step's name includes the iteration's
      directory (e.g. 4.iter_2.0.4). The runs of every iteration are now analyzed, as
      independent runs. Results from this choice in earlier versions came from a
      single run.

2026.10.16.1 -- The dielectric constant and the ionic conductivity
    * A new module, ``seamm_md_analysis.electric``, for the static dielectric constant
      from the fluctuations of the dipole of the cell (with errors from blocks, the
      running estimate and the dipole's correlation time), and for the ionic
      conductivity from the collective displacement of the ions' charges
      (Einstein-Helfand) and from their self-diffusion (Nernst-Einstein). It sorts
      the molecules into neutral ones and ions by their charges, so that for an
      electrolyte the dielectric constant is that of the solvent.

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

