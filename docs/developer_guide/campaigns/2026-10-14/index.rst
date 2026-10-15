2026-10-14 MD analysis, phase 1: the shared library
===================================================

The campaign's design is in the SEAMM developer guide:
`MD analysis: H5MD output, a shared library and analysis steps
<https://molssi-seamm.github.io/developer_guide/campaigns/2026-10-14/index.html>`__.
This package is its phase 1.

Phase 1
-------

The code that ``diffusivity_step`` and ``thermal_conductivity_step`` had each
copied and let drift is gathered here, and both steps import it, with no change
for users.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Here
     - Was
   * - ``stats.fmt_err``
     - ``fmt_err`` in both steps. They differed in the decimals used when the
       error is unusable (3 in Diffusivity, 2 in Thermal Conductivity): now the
       ``fallback_decimals`` argument.
   * - ``correlate.ccovf``, ``acf_err``, ``ccf_err``, ``moving_average``,
       ``tensor_correlation_functions``
     - ``thermal_conductivity_step.analysis``: ``statsmodels``' ``ccovf``,
       ``acf_err``, ``ccf_err``, ``moving_average``,
       ``create_correlation_functions``. ``ccovf`` uses the same definition
       and scipy's FFT correlation, so ``statsmodels`` is no longer needed.
   * - ``transport.collective_helfand_moments``
     - ``create_helfand_moments`` in Thermal Conductivity.
   * - ``transport.species_helfand_moments``, ``msd_by_species``
     - ``create_helfand_moments`` and ``compute_msd`` in Diffusivity.
   * - ``transport.fit_green_kubo_integral``, ``fit_helfand_moment``,
       ``helfand_slope``
     - The same in Thermal Conductivity (``get_helfand_slope``).
   * - ``transport.fit_slope``
     - The same in Diffusivity.
   * - ``transport.yeh_hummer_correction``
     - Inline in Diffusivity (twice).
   * - ``plots``
     - The plot functions of both steps, now one helper
       (``add_component_traces``) with thin wrappers.
   * - ``legacy.read_trajectory``, ``read_vector_trajectory``,
       ``read_dump_trajectory``
     - The pandas reader in Thermal Conductivity; the two readers in
       Diffusivity.

Changes in behaviour
--------------------

* **Speed.** The Helfand moments and MSDs are computed with FFTs. The old code
  looped over every time origin in Python, which took O(n m) time. For a heat
  flux of a million points with m = 10,000 that is about 2 minutes, against 0.4 s
  now (this Mac). For the Helfand moments of 512 molecules over 1,001 frames it
  was 3.3 s, against 0.09 s now. The results are the same to rounding: the tests
  check this against the original loops, and on real LAMMPS output from earlier
  jobs they agree to about 1e-15 relative.
* **Precision.** A flux with a non-zero mean has a running integral that drifts
  linearly. The straight line is removed before the FFTs and its contribution
  added back exactly, so the rounding error does not grow with the length of
  the series.
* **Latent bugs fixed:**

  - ``fit_slope`` with weights passed ``sigma[i:]`` with ``y[i:j]``, a shape
    error whenever the fit ended before the last point. No caller used weights.
  - ``fit_green_kubo_integral`` failed without ``sigma``, with a ``NameError``
    on very short series, and when the sampling was coarser than 0.4 ps (a
    zero-width moving average; this is what ended the 2025-11-19 argon demo
    job, Job_003234).
  - ``read_dump_trajectory`` failed on a dump with a single timed frame.
  - The MSD trace had units Å instead of Å^2 in its metadata.

* **Defects corrected** (Paul, 2026-10-14: fix the older problems in phase 1).
  These change results:

  - ``fit_green_kubo_integral`` accepted any second exponential whose plateau
    was positive and less than twice the first. On an OU test series it kept
    one with a negative amplitude and a time constant of 46 ps, beyond the 15 ps
    of data, and gave 0.018 instead of about 0.5. The second exponential is now
    kept only if both amplitudes are positive, the two time constants differ by
    more than 10 %, and three times the slower one lies within the fitted data.
  - Its error was the sum of the errors of the two amplitudes. When the two time
    constants are nearly equal the amplitudes are individually undetermined and
    strongly anticorrelated, so the error came out as large as 10^6 for a value
    of 0.5. The error now includes their covariance.
  - All the fits treated the errors as weights even when they were zero, as
    they are for a single run (the spread over runs). Thermal Conductivity's
    Helfand-moment fit then returned its starting guess, 1.00. A fit with
    missing, zero or non-finite errors is now unweighted.
  - ``helfand_slope`` stored the central difference over points k and k+2 at
    k+2 instead of k+1, which shifted the curve by one sample, and its error was
    the error of the moment, not divided by 2 dt. Both are corrected; the slope
    at the origin is zero, as the running Green–Kubo integral is.
  - The fourth column of ``species_helfand_moments``, the total, was the moment
    of the summed velocity components, (vx + vy + vz)^2. Its cross terms
    average to zero only for an isotropic system, and add noise in any case. It
    is now the sum of the three moments, as for the MSD.

Defects found and corrected in the steps at the same time:

* **LAMMPS Heat Flux sub-step** (``lammps_step``). To remove each atom's average
  stress, the callback passed the average stress, negated, as a virial to
  ``fix external``. ``compute stress/atom`` multiplies a virial by -nktv2p
  (-68,568 in real units), so the stress was changed by +68,568 times its
  average. For argon this made the heat flux about 10^4 times too large and the
  thermal conductivity about 4e7 W/m/K; it is now 0.1 W/m/K, as measured. For
  molecular systems the correction reached neither stress compute, and the
  convective term was counted twice (both ``heat/flux`` computes include it).
* **Diffusivity.** The per-direction coefficients Dx, Dy and Dz used the 1/6
  (MSD) or 1/3 (Helfand) of the total, so they were a third of the diffusion
  coefficient, and they got a third of the Yeh–Hummer correction. They are now
  D_alpha = slope / 2 (MSD) or slope (Helfand moments), and equal D for an
  isotropic system.

Phase 2: H5MD
=============

Done on 2026-10-15, following the design ("The data contract" and "Non-periodic
systems" in the plan).

* **The library** gained ``h5md`` (:class:`~seamm_md_analysis.H5MDWriter`), ``io``
  (:func:`~seamm_md_analysis.open_md`, :class:`~seamm_md_analysis.MDData`,
  :class:`~seamm_md_analysis.TimeSeries`) and ``units`` (H5MD unit strings and
  pint), and streaming readers for LAMMPS dumps and vector trajectories. The layout
  is specified in :ref:`h5md-layout`.
* **The LAMMPS step** writes ``md.h5md`` at the end of each NVE, NVT, NPT and Heat
  Flux sub-step (``lammps_step/md_h5md.py``) and appends the run to ``_md_runs``.
  The native files are kept.
* **Diffusivity and Thermal Conductivity** read ``md.h5md`` and fall back to the
  text files of older runs. Both stop before the MD for a non-periodic system.

Decisions taken while implementing:

* **A fixed box is written as time-dependent edges**, repeated for each frame of
  the positions. MDAnalysis reads ``box/edges/value[frame]`` and fails on the fixed
  form that H5MD also allows.
* **When both atomic positions and a trajectory are sampled**, the atomic positions
  (and velocities) go into ``particles/all``, and the trajectory into its own group,
  ``particles/trajectory``. MDAnalysis reads the first group alphabetically, "all".
* **The off-diagonal pressure** sampled for viscosity (``Pxy Pxz Pyz``) is written as
  ``pressure_offdiagonal`` (attribute ``components`` = "xy xz yz") until phase 4
  replaces it by the full ``pressure_tensor``.
* **Thermal Conductivity averages the temperature over the same samples as the heat
  flux:** the whole run. Before, the temperature came from ``state.json``, which drops
  the equilibration period detected by pymbar and keeps only uncorrelated samples
  (494 of 5,001 in an argon test), while the heat flux was used whole. The two were
  thus taken over different parts of the run. For that argon NVE run the temperature
  changed by 0.03 K, or 0.05% in the thermal conductivity. Dropping an equilibration
  period, if wanted later, must cut the heat flux and the temperature at the same step.

Defects found on the way:

* **LAMMPS, non-periodic systems:** sampling the trajectory of a molecule failed
  ("Variable name sxx ... does not exist"), since the stress columns were added
  regardless of periodicity. The pressure was reported though meaningless.
* **Diffusivity:** without centre-of-mass data it used the atomic positions or
  velocities with the molecules' indices, which is wrong unless every molecule is
  one atom. It now checks the number of particles.
* **Thermal Conductivity:** a missing ``state.json`` gave a ``NameError``; the
  sampling interval was taken as fs even when the file gave ps.

Phase 3: requirements and analysing earlier MD
==============================================

Done on 2026-10-15.

* **The library** gained ``requirements`` (:class:`~seamm_md_analysis.Need`,
  :class:`~seamm_md_analysis.Requirements`, :class:`~seamm_md_analysis.Provides`,
  ``check``), ``select_md_runs``, and ``step_support``, the code the analysis steps
  share. The protocol is described in :ref:`analysis-steps`.
* **seamm**: ``builder.validate`` calls each step's ``check_flowchart(preceding)``;
  the editor validates before submitting. **seamm_exec**: ``run_flowchart`` validates
  before running any step, except in an iteration of a parallel loop, which its
  parent job checked. A failing check is logged, never reported, so a bug in a step
  cannot stop every job.
* **LAMMPS**: ``md_provides`` and ``request_sampling`` on NVE, NVT, NPT and Heat Flux
  (``md_provides.py``); before running, the LAMMPS step calls ``check_configuration``
  on the steps after it up to the next LAMMPS step, following the flowchart's edges
  without calling ``next()``, which a Loop may override.
* **Diffusivity and Thermal Conductivity**: the ``md source``, ``md files`` and
  ``number of blocks`` parameters, both modes, automatic sampling in mode A, and the
  dialogs hide what does not apply, including the sub-flowchart tab in mode B.

Decisions taken while implementing:

* **Mode B reads the last run that samples the first need**, not simply the last
  run, both when checking the flowchart and when running. A LAMMPS step with an NVT
  run sampling centres of mass followed by a Heat Flux run then feeds both
  Diffusivity (the NVT run) and Thermal Conductivity (the Heat Flux run).
* **A single run is divided into blocks** (5 by default) analyzed as independent runs
  for the errors. Thermal Conductivity requires at least 2000 heat-flux samples per
  block; Diffusivity at least 10 frames.
* **Sources:** "the last MD run", "the last MD run of each loop iteration" (runs by
  the same step, i.e. the iterations of a Loop), and files. "All runs of the last MD
  step", for production split over several sub-steps, waits for the segments of
  phase 8.
* **Diffusivity's requirements:** positions of the molecules (or atoms, for monatomic
  systems) at least every 1 ps, turned on at 100 fs; velocities at least every 20 fs,
  turned on at 10 fs. Thermal Conductivity: the heat flux every two steps. Mode A
  turns on what is missing in the last MD sub-step of the sub-flowchart and says so.
* **Thermal Conductivity in mode B** uses the Green–Kubo approach only (the only one
  implemented) at the temperature of the MD; the approach and temperature parameters
  apply only when it runs its own MD.

Found on the way: when the correlation window is shorter than the start of the
Helfand-moment fit (at least 1 ps), Thermal Conductivity silently dropped the Helfand
Moments rows; it now says why.

Phase 4: the viscosity
======================

Done on 2026-10-15.

* **The library** gained ``shear_stress_components``, ``autocorrelation_functions``,
  ``component_helfand_moments``, ``publish_viscosity`` / ``preceding_viscosity``
  (the workspace variable ``_viscosity``) and ``plots.plot_components``.
* **LAMMPS**: the NVE, NVT and NPT sub-steps sample the full pressure tensor ("Sample
  the pressure tensor") instead of its three off-diagonal components. The parameters
  ``shear stress*`` were renamed ``pressure tensor*``; ``NVE_Parameters`` translates
  the old names when a flowchart is read (format 2.0 and 3.0), as NPT already did for
  "keep orthorhombic". The sampled tensor is written to ``md.h5md`` as
  ``pressure_tensor``, replacing the coarser one from the thermodynamic state, so
  ``pressure_offdiagonal`` is no longer written.
* **Viscosity** (new, ``viscosity_step``): Green-Kubo and Helfand moments of the five
  shear components and their average, in both modes; needs ``pressure_tensor`` at
  least every 5 fs and a periodic system. It publishes the Green-Kubo average for
  later steps and stores it as ``viscosity#Green-Kubo#<model>``.
* **Diffusivity**: "Viscosity from" chooses the preceding Viscosity step (the default)
  or a given value for the Yeh-Hummer correction, which is now on by default and
  skipped with a note when there is no viscosity. Older flowcharts, which always
  give the viscosity, keep it.

Decisions taken while implementing:

* **The mean of the shear stress is not subtracted.** For a fluid it is zero, and
  subtracting the sample mean of a block of length T lowers the correlation function
  by about 2 sigma^2 tau / T at every lag. With synthetic stresses of known viscosity
  (0.9 mPa s, 200 ps blocks) the Green-Kubo fit averaged 0.837 with the mean
  subtracted and 0.878 without, over eight seeds.
* **The average is fitted, not averaged.** The Green-Kubo and Helfand values of the
  average are fits to the curve averaged over the five components, which is less
  noisy than any one of them; its error comes from the spread of the fits to each
  run or block.
* **The Viscosity step resets its data at the start of each run**, so it gives the
  right answer inside a loop. Thermal Conductivity does not, which is worth fixing.
