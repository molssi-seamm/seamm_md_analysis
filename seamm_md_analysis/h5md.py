# -*- coding: utf-8 -*-

"""Writing SEAMM's H5MD files.

An MD step writes one H5MD file per run, ``md.h5md``, from its code's native
output when the run ends. :class:`H5MDWriter` holds the layout, so that every MD
step writes the same thing; the layout is described in the documentation
("The H5MD file").

The file is written as ``<name>.tmp`` and renamed when it is complete, so an
interrupted conversion never leaves a damaged file under the real name.
"""

import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np

H5MD_VERSION = (1, 1)
SEAMM_LAYOUT_VERSION = 1

#: The standard observable names and their shapes; MD steps map their output onto
#: these, and analysis steps ask for them.
standard_observables = {
    "temperature": (),
    "pressure": (),
    "pressure_tensor": (3, 3),
    "total_energy": (),
    "kinetic_energy": (),
    "potential_energy": (),
    "molecular_energy": (),
    "pair_energy": (),
    "volume": (),
    "density": (),
    "heat_flux": (3,),
    "total_dipole": (3,),
    "hbond_count": (),
    "hbond_energy": (),
}


def _attr(value):
    """Make a value storable as an HDF5 attribute."""
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind in "biuf":
            return array
    return json.dumps(value, default=str)


class _Appender:
    """Appends frames to a time-dependent H5MD element."""

    def __init__(self, group, shape, dtype, compression):
        self.group = group
        self.n = 0
        chunks = (1,) + tuple(shape)
        kwargs = {}
        if compression:
            kwargs = {"compression": "gzip", "shuffle": True}
        self.value = group.create_dataset(
            "value",
            shape=(0,) + tuple(shape),
            maxshape=(None,) + tuple(shape),
            chunks=chunks,
            dtype=dtype,
            **kwargs,
        )
        self.step = group.create_dataset(
            "step", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="i8"
        )
        self.time = group.create_dataset(
            "time", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="f8"
        )

    def append(self, step, time, value):
        """Add one frame.

        Parameters
        ----------
        step : int
            The MD step.
        time : float
            The time, in the element's time unit.
        value : array_like
            The frame, with the element's shape.
        """
        n = self.n + 1
        self.value.resize(n, axis=0)
        self.step.resize(n, axis=0)
        self.time.resize(n, axis=0)
        self.value[self.n] = value
        self.step[self.n] = step
        self.time[self.n] = time
        self.n = n


class H5MDWriter:
    """Write one MD run as a SEAMM H5MD file.

    Use it as a context manager. The file is renamed to its final name only if the
    block finishes without an exception; otherwise the temporary file is removed.

    >>> with H5MDWriter("md.h5md", creator="SEAMM lammps_step", version="1") as w:
    ...     w.particles(ids=..., species=..., masses=..., cell=..., periodic=True)
    ...     w.observable("temperature", steps, times, T, "K", "fs")

    Parameters
    ----------
    path : str or pathlib.Path
        The final name of the file.
    creator : str
        The program writing the file, e.g. "SEAMM lammps_step".
    version : str
        Its version.
    author : str = None
        The user, by default from the environment.
    compress_particles : bool = False
        Compress positions, velocities and forces with gzip. It gains little on
        coordinates, so it is off by default.
    """

    def __init__(
        self, path, creator="SEAMM", version="", author=None, compress_particles=False
    ):
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        self.compress_particles = compress_particles
        self._fixed_cells = {}
        if author is None:
            author = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
        self.file = h5py.File(self.tmp_path, "w", libver=("earliest", "v110"))
        h5md = self.file.create_group("h5md")
        h5md.attrs["version"] = np.array(H5MD_VERSION)
        h5md.create_group("author").attrs["name"] = author
        group = h5md.create_group("creator")
        group.attrs["name"] = creator
        group.attrs["version"] = str(version)
        units = h5md.create_group("modules").create_group("units")
        units.attrs["version"] = np.array([1, 0])
        units.attrs["system"] = "SI"
        self.file.create_group("parameters/seamm").attrs[
            "layout_version"
        ] = SEAMM_LAYOUT_VERSION
        self.file["parameters/seamm"].attrs["written"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

    def close(self):
        """Finish the file and give it its final name."""
        if self.file is not None:
            self._write_fixed_cells()
            self.file.close()
            self.file = None
            os.replace(self.tmp_path, self.path)

    def abort(self):
        """Discard the file."""
        if self.file is not None:
            self.file.close()
            self.file = None
        self.tmp_path.unlink(missing_ok=True)

    def _write_fixed_cells(self):
        """Write a box that does not change as time-dependent edges.

        MDAnalysis reads the edges frame by frame, so a fixed box is repeated for
        each frame of the positions, or written once at step 0 if there are none.
        """
        for group, (cell, unit) in self._fixed_cells.items():
            g = self.file[f"particles/{group}"]
            if "position" in g:
                steps = g["position/step"][()]
                times = g["position/time"][()]
                time_unit = g["position/time"].attrs["unit"]
            else:
                steps = np.zeros(1, dtype="i8")
                times = np.zeros(1)
                time_unit = "fs"
            edges = g["box"].create_group("edges")
            data = edges.create_dataset(
                "value", data=np.repeat(cell[None], len(steps), 0)
            )
            data.attrs["unit"] = unit
            edges.create_dataset("step", data=steps)
            edges.create_dataset("time", data=times).attrs["unit"] = time_unit
        self._fixed_cells = {}

    # The particles

    def particles(
        self,
        group="all",
        ids=None,
        species=None,
        masses=None,
        charges=None,
        cell=None,
        periodic=True,
        length_unit="Angstrom",
        mass_unit="amu",
        charge_unit="e",
    ):
        """Describe a group of particles and its box.

        Parameters
        ----------
        group : str = "all"
            The particle group: "all" for the atoms, "molecules" for the centres of
            mass of the molecules.
        ids : array_like of int
            The SEAMM ids of the atoms, or the indices of the molecules.
        species : array_like of int = None
            The atomic numbers.
        masses : array_like of float = None
            The masses.
        charges : array_like of float = None
            The charges, omitted when the model has none (MLFFs).
        cell : 3×3 array_like = None
            The cell vectors as rows, for a box that does not change. It is written
            for each frame of the positions when the file is closed, since
            MDAnalysis reads the box frame by frame. Leave it out for a
            non-periodic system, or when the box changes (see
            :meth:`box_appender`).
        periodic : bool = True
            Whether the system is periodic. A non-periodic box has no edges,
            because MDAnalysis treats any box with edges as periodic.
        length_unit, mass_unit, charge_unit : str
            The units, in H5MD syntax.
        """
        g = self.file.require_group(f"particles/{group}")
        box = g.create_group("box")
        box.attrs["dimension"] = 3
        boundary = b"periodic" if periodic else b"none"
        box.attrs["boundary"] = np.array([boundary] * 3)
        if periodic and cell is not None:
            # Written when the file is closed, once the frames are known
            self._fixed_cells[group] = (np.asarray(cell, dtype="f8"), length_unit)
        if ids is not None:
            g.create_dataset("id", data=np.asarray(ids, dtype="i8"))
        if species is not None:
            g.create_dataset("species", data=np.asarray(species, dtype="i4"))
        if masses is not None:
            data = g.create_dataset("mass", data=np.asarray(masses, dtype="f8"))
            data.attrs["unit"] = mass_unit
        if charges is not None:
            data = g.create_dataset("charge", data=np.asarray(charges, dtype="f8"))
            data.attrs["unit"] = charge_unit
        self.file["parameters/seamm"].attrs["periodic"] = bool(periodic)
        return g

    def box_appender(self, group="all", length_unit="Angstrom", time_unit="fs"):
        """An appender for a box that changes with time, as in NPT.

        Returns
        -------
        object
            Call ``append(step, time, cell)`` for each frame, with the cell
            vectors as the rows of a 3×3 array.
        """
        box = self.file[f"particles/{group}/box"]
        self._fixed_cells.pop(group, None)
        if "edges" in box:
            del box["edges"]
        appender = _Appender(box.create_group("edges"), (3, 3), "f8", False)
        appender.value.attrs["unit"] = length_unit
        appender.time.attrs["unit"] = time_unit
        return appender

    def particle_appender(
        self,
        name,
        n_particles,
        unit,
        time_unit="fs",
        group="all",
        attrs=None,
    ):
        """An appender for a per-particle property, frame by frame.

        Positions, velocities and forces are written as float32, chunked by
        frame, so a long trajectory is streamed rather than held in memory.

        Parameters
        ----------
        name : str
            "position", "velocity" or "force".
        n_particles : int
            The number of particles in the group.
        unit, time_unit : str
            The units, in H5MD syntax.
        group : str = "all"
            The particle group.
        attrs : dict = None
            Extra attributes for the value dataset, e.g. {"seamm_unwrapped": 1}.

        Returns
        -------
        object
            Call ``append(step, time, frame)`` with an n_particles × 3 array.
        """
        g = self.file.require_group(f"particles/{group}").create_group(name)
        appender = _Appender(g, (n_particles, 3), "f4", self.compress_particles)
        appender.value.attrs["unit"] = unit
        appender.time.attrs["unit"] = time_unit
        for key, value in (attrs or {}).items():
            appender.value.attrs[key] = _attr(value)
        return appender

    # The time series

    def observable(
        self,
        name,
        steps,
        times,
        values,
        unit,
        time_unit="fs",
        average_window=1,
        kind=None,
        attrs=None,
    ):
        """Write a time series under ``observables/``.

        Parameters
        ----------
        name : str
            The name, a standard one (see :data:`standard_observables`) or, for the
            user's extra properties, "user/<name>".
        steps : array_like of int
            The MD steps.
        times : array_like of float
            The times, in ``time_unit``.
        values : array_like
            The values, with the first index running over the samples.
        unit, time_unit : str
            The units, in H5MD syntax, as the code produced them.
        average_window : int = 1
            The number of MD steps each value averages over; 1 for instantaneous
            values.
        kind : str = None
            "correlation" for a correlation function accumulated by the code.
        attrs : dict = None
            Extra attributes for the value dataset.
        """
        values = np.asarray(values, dtype="f8")
        steps = np.asarray(steps, dtype="i8")
        times = np.asarray(times, dtype="f8")
        if values.shape[0] != steps.shape[0] or times.shape[0] != steps.shape[0]:
            raise ValueError(
                f"Observable {name}: {values.shape[0]} values, {steps.shape[0]} "
                f"steps and {times.shape[0]} times."
            )
        g = self.file.require_group("observables").create_group(name)
        kwargs = {}
        if values.size > 0:
            kwargs = {"compression": "gzip", "shuffle": True}
        data = g.create_dataset("value", data=values, **kwargs)
        data.attrs["unit"] = unit
        data.attrs["seamm_average_window"] = int(average_window)
        if kind is not None:
            data.attrs["seamm_kind"] = kind
        for key, value in (attrs or {}).items():
            data.attrs[key] = _attr(value)
        g.create_dataset("step", data=steps, **kwargs)
        data = g.create_dataset("time", data=times, **kwargs)
        data.attrs["unit"] = time_unit
        return g

    # The parameters and provenance

    def parameters(self, values, group="parameters"):
        """Store the run's parameters as attributes, nesting dicts as groups.

        Parameters
        ----------
        values : dict
            E.g. {"ensemble": "NVT", "temperature": 298.15, "temperature_unit":
            "K", "timestep": 1.0, ...}.
        """
        g = self.file.require_group(group)
        for key, value in values.items():
            if isinstance(value, dict):
                self.parameters(value, group=f"{group}/{key}")
            else:
                g.attrs[key] = _attr(value)

    def provenance(self, values, segments=None):
        """Store SEAMM's provenance under ``parameters/seamm``.

        Parameters
        ----------
        values : dict
            The job and step ids, configuration, model chemistry, code and its
            version, package versions, hashes of model files, ...
        segments : [dict] = None
            One dict per segment of the run, with "first_step", "last_step",
            "wall_time" (s) and "node".
        """
        self.parameters(values, group="parameters/seamm")
        if segments is not None:
            g = self.file.require_group("parameters/seamm/segments")
            for key, dtype in (
                ("first_step", "i8"),
                ("last_step", "i8"),
                ("wall_time", "f8"),
            ):
                data = [s.get(key, -1) for s in segments]
                g.create_dataset(key, data=np.array(data, dtype=dtype))
            nodes = [str(s.get("node", "")) for s in segments]
            g.create_dataset("node", data=np.array(nodes, dtype=h5py.string_dtype()))
