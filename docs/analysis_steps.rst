.. _analysis-steps:

===========================
MD steps and analysis steps
===========================

An analysis step, such as Diffusivity or Thermal Conductivity, needs the MD to have
sampled the right things at the right rate: the heat flux every two steps, the
centres of mass of the molecules, and so on. If the MD does not, the analysis fails,
and in the past it failed only after all the MD had run. This page describes how MD
steps and analysis steps tell each other what they provide and need, so that such
mistakes are caught before the MD runs. Any MD step (LAMMPS now, GROMACS later) and
any analysis step can take part by implementing the methods below.

The names are those of the H5MD file (see :ref:`h5md-layout`): an observable such as
``heat_flux`` or ``pressure_tensor``, or a per-particle property as
``<group>/<name>``, e.g. ``molecules/position``.

Where the MD comes from
=======================

An analysis step analyzes MD that ran before it in the flowchart. Its **MD to
analyze** parameter (``md source``) chooses which:

``the last MD run of each loop iteration`` (the default)
    Every run by the same MD step as the last one, which in a Loop are the runs of its
    iterations, analyzed as independent runs. Only the innermost loop around the MD
    counts, so in a loop over temperatures around a loop over replicas, an analysis
    inside the temperature loop takes that temperature's replicas. Without a loop it
    is the one run.
``the last MD run``
    The last MD run earlier in the flowchart that sampled what the step needs, found
    in the workspace variable ``_md_runs``.
``files``
    The ``md.h5md`` files given, each an independent run; ``job://<n>/<path>`` reads
    another job's file.

For independent runs, put the MD in a Loop -- in parallel, each run starting from the
same structure ("Discard the structure" in the MD sub-steps) with new velocities --
and put the analysis after the loop. One set of runs can feed several analyses. When
a single run is analyzed, it is divided into consecutive blocks (``number of
blocks``, 5 by default) for the errors; each block must be much longer than the
correlation time of what is analyzed.

The steps used to run their own MD in a sub-flowchart. That route is gone: a
flowchart still asking for it fails the check before the job runs, with a message
saying what replaces it.

The methods
===========

MD sub-steps
------------

``md_provides()``
    Returns a :class:`~seamm_md_analysis.Provides`: what the run will write to
    ``md.h5md``, each with its sampling :class:`~seamm_md_analysis.Interval` (in time,
    in steps, or both), worked out from the step's parameters before anything runs.
    An interval that cannot be known yet (a variable, or a timestep that depends on
    the structure) is left unknown.

``request_sampling(need)``
    Turns on the sampling a :class:`~seamm_md_analysis.Need` asks for, if it can, and
    returns a description of the change, or None.

Analysis steps
--------------

``md_requirements()``
    Returns :class:`~seamm_md_analysis.Requirements`: what is needed, each with the
    longest acceptable interval (and the interval to use when turning it on), and
    whether the system must be periodic.

``check_flowchart(preceding)``
    Called by ``seamm.builder.validate`` with the steps that run before this one. For
    MD from earlier in the flowchart it compares the requirements with what the last
    MD sub-step that samples the first need provides, and returns the problems
    (:func:`~seamm_md_analysis.step_support.check_md_flowchart`). Validation runs in
    ``seamm-flowchart`` and the MCP tools, when the editor submits a job, and when a
    job starts, which stops before running any step if there are problems.

``check_configuration(configuration)``
    Called by an MD step just before it runs, for each analysis step after it up to
    the next MD step. An analysis that will read this MD and cannot use the structure,
    such as one needing a periodic system, raises an exception, so the job stops
    before the MD. Periodicity is known only once the structure exists, so the
    flowchart check cannot catch it. It may also return warnings, e.g. from
    :func:`~seamm_md_analysis.step_support.size_warnings`, which the MD step prints.

What the MD does not sample is reported by the flowchart check with a fix: the MD
step's ``sampling_edits(need)`` gives the parameter changes that turn the sampling
on, which the editor offers as a button and ``seamm-flowchart validate --fix``
applies.

:mod:`seamm_md_analysis.step_support` holds the code the analysis steps share:
the parameter definitions (``source_parameters``), finding the MD sub-steps, the
checks, finding the files to read (``md_inputs``), and passing results between
steps.

Results for later steps
=======================

An analysis step can publish a result that later steps use. The Viscosity step
calls :func:`~seamm_md_analysis.step_support.publish_viscosity`, which stores the
viscosity (in Pa s), its standard error, the temperature and the method in the
workspace variable ``_viscosity``. Diffusivity reads it with
:func:`~seamm_md_analysis.step_support.preceding_viscosity` for the Yeh–Hummer
correction. A later Viscosity step replaces the record. It uses only the methods of
SEAMM's nodes and flowcharts, so the library does not depend on SEAMM.

How the results are reported
============================

The analysis steps report their results in one way, with the functions of
:mod:`seamm_md_analysis.stats` and :mod:`seamm_md_analysis.state`.

**Errors.** A result's error comes from its spread over independent runs, or over
blocks of a single run. Tables show the 95% confidence interval, Student's t for the
number of runs or blocks times the standard error (2.78 for 5 runs, not 2), and say
so under the table. The results stored for later steps, such as the viscosity a
Diffusivity step uses for the box-size correction, carry the standard error.

**One value for an isotropic system.** For a fluid, a property with components -- the
five shear components of the viscosity, the three diagonal components of the
diffusion coefficient or the thermal conductivity, the trace of the dipole
fluctuations for the dielectric constant -- is reported as one value: the components
are averaged within each run, and the error is that of these averages. The spread of
the components is a check, given as one line ("the components agree: χ²/dof = 0.8"),
not the error. A run is treated as isotropic only when its cell is cubic and nothing
in it is directional (walls, fields, deformation, flow); otherwise the components
are reported. The tables of each component and each run are in the detailed output.

**The state.** Every analysis step reports the state its results belong to: the
temperature, pressure, volume and density, with their errors over the runs. For
NPT runs the density is the equilibrium density at their pressure. For runs at
constant volume the pressure is what the density gives; if an NPT run comes before
them in the flowchart (an equilibration, say), its volume fluctuations give the bulk
modulus K, and the equilibrium density at its pressure is estimated as
ρ (1 + (P_target − P) / K).

