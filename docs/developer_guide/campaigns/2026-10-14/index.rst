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
  - ``fit_green_kubo_integral`` failed without ``sigma``, and with a
    ``NameError`` on very short series.
  - ``read_dump_trajectory`` failed on a dump with a single timed frame.
  - The MSD trace had units Å instead of Å^2 in its metadata.

* **Kept as it was:**

  - ``helfand_slope`` stores the difference between points k and k+2 at k+2
    rather than at the centre, k+1. Changing it would change Thermal
    Conductivity's Helfand-slope results slightly, so it is left for a later,
    deliberate change.
  - The fourth column of ``species_helfand_moments`` uses the sum of the three
    velocity components, not the sum of the three moments.
