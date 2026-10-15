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

Two ways to get the MD
======================

An analysis step's **MD to analyze** parameter (``md source``) chooses where the MD
comes from:

``run the MD in the sub-flowchart``
    The step runs the MD in its own sub-flowchart a number of times, as independent
    runs, and takes the errors from their spread. This is how the steps have always
    worked.
``the last MD run``
    The step analyzes the last MD run earlier in the flowchart that sampled what it
    needs, found in the workspace variable ``_md_runs``. One production run can then
    feed several analyses.
``the last MD run of each loop iteration``
    Every run by the same MD step as the last one, which in a Loop are the runs of the
    iterations, analyzed as independent runs.
``files``
    The ``md.h5md`` files given, each an independent run; ``job://<n>/<path>`` reads
    another job's file.

When a single run is analyzed, it is divided into consecutive blocks (``number of
blocks``, 5 by default), which are analyzed separately to estimate the errors. Each
block must be much longer than the correlation time of what is analyzed.

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
    flowchart check cannot catch it.

When an analysis runs its own MD, it calls
:func:`~seamm_md_analysis.step_support.request_sampling` on its sub-flowchart first,
which turns on in the last MD sub-step whatever is missing and reports the changes;
anything it cannot turn on stops the step before the MD.

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
