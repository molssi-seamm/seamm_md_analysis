Getting Started
===============

``seamm_md_analysis`` is the shared library behind SEAMM's analysis of molecular
dynamics: the Diffusivity, Thermal Conductivity and Viscosity steps now, and
Structure and the Dielectric constant later. The steps run the MD (or read an
earlier run), and this library does the numerical work.

Installing
----------

::

    pip install seamm-md-analysis

It needs numpy, scipy and pymbar.

What is in it
-------------

:mod:`seamm_md_analysis.stats`
    ``mean_and_stderr`` over replicas, ``block_average``,
    ``statistical_inefficiency`` and ``detect_equilibration`` (pymbar, imported
    quietly when first used), ``mean_and_stderr_correlated``, and ``fmt_err``
    to print a value with its error.

:mod:`seamm_md_analysis.correlate`
    ``ccovf``, the cross-covariance of two series by FFT (the definition of
    ``statsmodels``), the errors of correlation functions, and
    ``tensor_correlation_functions`` for the six components of a vector flux
    with their running integrals (the Green–Kubo input), and
    ``autocorrelation_functions`` for independent series such as the components of
    the shear stress.

:mod:`seamm_md_analysis.transport`
    ``collective_helfand_moments`` for a flux such as the heat flux;
    ``shear_stress_components``, the five independent shear components of a
    pressure tensor, and ``component_helfand_moments`` for each of them, for the
    viscosity;
    ``species_helfand_moments`` and ``msd_by_species`` for single-molecule
    diffusion; ``fit_green_kubo_integral``, ``fit_helfand_moment``,
    ``helfand_slope`` and ``fit_slope``; and ``yeh_hummer_correction``.

    The Helfand moments and MSDs are computed with FFTs. They equal the direct
    sums over time origins to rounding, but take O(n log n) rather than O(n m)
    time.

:mod:`seamm_md_analysis.plots`
    Plots of correlation functions, Green–Kubo integrals, Helfand moments and
    MSDs, for ``seamm_util`` figures.

:mod:`seamm_md_analysis.legacy`
    Readers for the files the LAMMPS step writes today: ``!MolSSI trajectory``
    time series (``read_trajectory``), per-molecule ``vector_trajectory`` files
    (``read_vector_trajectory``) and dump files (``read_dump_trajectory``).

Example
-------

The thermal conductivity from a heat flux sampled every ``dt``:

.. code-block:: python

    import numpy as np
    from seamm_md_analysis import (
        collective_helfand_moments,
        fit_green_kubo_integral,
        read_trajectory,
        tensor_correlation_functions,
    )

    metadata, columns = read_trajectory("heat_flux.trj")
    J = np.stack((columns["Jx"], columns["Jy"], columns["Jz"]))
    m = min(J.shape[1] // 10, 10000)

    # Correlation functions and their integrals, per sampling interval
    ccf, integral = tensor_correlation_functions(J, m=m)

    # Helfand moments, per sampling interval squared
    M = collective_helfand_moments(J, m=m)

Multiply by V / (k_B T²) and the sampling interval (Green–Kubo), or
V dt² / (2 k_B T²) (Helfand), to get W/m/K.

The shear viscosity from a pressure tensor ``P`` (n × 3 × 3) works the same way,
averaging the five components:

.. code-block:: python

    from seamm_md_analysis import (
        autocorrelation_functions,
        component_helfand_moments,
        shear_stress_components,
    )

    X = shear_stress_components(P)  # Pxy, Pxz, Pyz, (Pxx-Pyy)/2, (Pyy-Pzz)/2
    acf, integral = autocorrelation_functions(X, m=m)
    M = component_helfand_moments(X, m=m)

Multiply by V / (k_B T) and the sampling interval (Green–Kubo), or
V dt² / (2 k_B T) (Helfand), to get the viscosity.
