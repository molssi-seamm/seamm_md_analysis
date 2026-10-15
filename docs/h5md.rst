.. _h5md-layout:

=============
The H5MD file
=============

Every MD run in SEAMM writes one file, ``md.h5md``, in the directory of its step. It
holds the trajectory (if one was sampled), every time series the run sampled, the
run's parameters and SEAMM's provenance. The analysis steps read only this file, so
they work the same way whichever code ran the dynamics.

The file is `H5MD 1.1 <https://www.nongnu.org/h5md/h5md.html>`_, written with h5py by
:class:`seamm_md_analysis.H5MDWriter` and read by :func:`seamm_md_analysis.open_md`.
This page is the specification: an MD step that writes this layout can feed every
analysis step.

Writing the file
================

* The MD step writes the file from its code's native output **when the run (or a
  segment of it) ends**. During the run the code writes its usual files, so a crash
  loses nothing and a run can be followed with ``tail``.
* The file is written as ``md.h5md.tmp`` and renamed when it is complete, so an
  interrupted conversion never leaves a damaged file under the real name.
* The HDF5 format is pinned to ``libver=("earliest", "v110")``, so that HDF5 1.10
  and later can read it.
* A run without a trajectory still writes the file, containing only observables.

The layout
==========

.. list-table::
   :header-rows: 1
   :widths: 32 50 18

   * - Path
     - Contents
     - Required
   * - ``h5md/``
     - Attribute ``version`` = [1, 1]. ``creator`` (attributes ``name``, e.g.
       ``SEAMM lammps_step``, and ``version``) and ``author`` (``name``, the job's
       user).
     - yes
   * - ``h5md/modules/units``
     - Declares that ``value`` and ``time`` datasets carry ``unit`` attributes.
     - yes
   * - ``particles/all/box``
     - Attributes ``dimension`` = 3 and ``boundary``, three times ``periodic`` or
       ``none``. For a periodic system, ``edges`` is time-dependent (``value``,
       ``step``, ``time``) with each value a 3×3 matrix whose rows are the cell
       vectors. A box that does not change is repeated for each frame of the
       positions, because MDAnalysis reads the box frame by frame. **A non-periodic
       system has no** ``edges``: MDAnalysis treats any box with edges as periodic.
     - yes
   * - ``particles/all/id``, ``species``, ``mass``, ``charge``
     - Fixed for the run, in the order of the atoms in the configuration: SEAMM's
       atom ids, atomic numbers, masses (``amu``) and charges (``e``). ``charge`` is
       left out when the model has no charges (MLFFs).
     - id, species, mass
   * - ``particles/all/position``
     - **Unwrapped** positions (attribute ``seamm_unwrapped`` = 1, no ``image``),
       float32, in ``Angstrom``.
     - if sampled
   * - ``particles/all/velocity``, ``force``
     - float32.
     - if sampled
   * - ``particles/molecules/...``
     - The centres of mass of the molecules: ``position`` (unwrapped) and
       ``velocity``, with ``id`` the index of each molecule in the configuration.
       ``box`` as for the atoms.
     - if sampled
   * - ``observables/<name>``
     - The time series, each with its own ``step``, ``time`` and ``value``. The
       names are listed below.
     - as sampled
   * - ``observables/user/<name>``
     - The user's own properties.
     - if defined
   * - ``parameters/``
     - Attributes: the ensemble, temperature, pressure, timestep, thermostat and
       barostat with their damping, and the number of steps. Nested dicts are
       groups.
     - yes
   * - ``parameters/seamm/``
     - Provenance: ``layout_version`` (now 1), ``written`` (UTC), ``periodic``, the
       job and step, the configuration, the model chemistry, the code and its
       version, and ``segments``, a group of datasets ``first_step``,
       ``last_step``, ``wall_time`` (s) and ``node``, one entry per segment.
     - yes

Time-dependent elements (positions, velocities, forces, box edges and observables)
store ``step`` (int64) and ``time`` (float64) as arrays, one entry per sample.

Standard observable names
=========================

Analysis steps ask for these names, and the MD steps map their codes' output onto
them. :data:`seamm_md_analysis.h5md.standard_observables` lists them with their
shapes.

.. list-table::
   :header-rows: 1
   :widths: 26 12 22 40

   * - Name
     - Shape
     - Units, e.g.
     - Notes
   * - ``temperature``
     - scalar
     - ``K``
     -
   * - ``pressure``
     - scalar
     - ``atm``
     - Periodic systems only.
   * - ``pressure_tensor``
     - 3×3
     - ``atm``
     - The full symmetric tensor (the pressure, not the stress). Periodic systems
       only.
   * - ``total_energy``, ``kinetic_energy``, ``potential_energy``
     - scalar
     - ``kcal mol-1``
     -
   * - ``molecular_energy``, ``pair_energy``
     - scalar
     - ``kcal mol-1``
     - The bonded and non-bonded parts, where the code reports them.
   * - ``volume``, ``density``
     - scalar
     - ``Angstrom3``, ``g cm-3``
     - Periodic systems only.
   * - ``heat_flux``
     - 3
     - ``W m-2``
     - Periodic systems only.
   * - ``total_dipole``
     - 3
     - ``e Angstrom``
     - Needs charges.
   * - ``hbond_count``, ``hbond_energy``
     - scalar
     - ``kcal mol-1``
     - DREIDING hydrogen bonds, when present.

Units
=====

* Every ``value`` and ``time`` dataset has a ``unit`` attribute, in H5MD's syntax:
  space-separated factors with integer powers and negative powers for the
  denominator, such as ``kcal mol-1`` or ``g cm-3``. An empty string is
  dimensionless.
* Lengths, times, velocities and forces use MDAnalysis's spellings so that it can
  read the trajectory: ``Angstrom``, ``fs``, ``Angstrom ps-1``,
  ``kcal mol-1 Angstrom-1``.
* Values are stored **in the units the code produced**, with no lossy conversion,
  and converted when read. :mod:`seamm_md_analysis.units` translates the strings to
  and from pint.

Rates and averages
==================

Each observable and each particle property has its own sampling rate, given by its
``step`` and ``time``. A value that the code averaged on the fly has the attribute
``seamm_average_window``, the number of MD steps it averages over; it is 1 for
instantaneous values. A correlation function accumulated by the code (e.g. with
LAMMPS's ``fix ave/correlate/long``) has ``seamm_kind`` = ``correlation``.

**Read time series with** :meth:`~seamm_md_analysis.MDData.observable`, on their own
time axis. MDAnalysis attaches observables to trajectory frames by index, ignoring
their steps, so an observable sampled more often than the frames is silently
misaligned there.

Storage
=======

* Positions, velocities and forces are float32, chunked by frame, and not
  compressed by default; gzip gains little on coordinates.
* Observables are float64 with gzip and shuffle.
* Only the standard gzip filter is used, since readers may lack plug-in filters.

Segments
========

A run split into segments (for "run until converged", restarts or preemption) writes
``md.seg-NNNN.h5md`` as each segment ends; when the run ends they are joined into
``md.h5md``. :func:`~seamm_md_analysis.open_md` also accepts the list of segment
files, so a run can be analysed before it finishes. A sample repeated at the
boundary between two segments is used once.

Non-periodic systems
====================

For a non-periodic run the box has ``boundary`` ``none`` and no ``edges``;
``parameters/seamm`` has ``periodic`` = false; and ``pressure``,
``pressure_tensor``, ``volume``, ``density`` and ``heat_flux`` are not written.
:attr:`MDData.periodic <seamm_md_analysis.MDData.periodic>` reports it, and analyses
that need a volume refuse such a run.

File locking
============

Readers open the file read-only. HDF5's file locking works on the file systems we
use (NFS on ARC, ext4 and ZFS on ChemAI). If a site shows "unable to lock file"
errors, set ``HDF5_USE_FILE_LOCKING=FALSE`` in the environment of that
installation.

Reading a file
==============

.. code-block:: python

    from seamm_md_analysis import open_md

    with open_md("md.h5md") as md:
        T = md.observable("temperature")              # a TimeSeries
        P = md.observable("pressure_tensor", units="bar", time_units="ps")
        print(T.values.mean(), T.units, T.dt)

        com = md.particle_series("position", group="molecules", units="Å")

        for steps, times, frames in md.iter_frames(fields=("position",), chunk=200):
            ...  # frames["position"] has the shape (n, n_atoms, 3)

        u = md.universe(configuration)  # MDAnalysis, for structural analysis
