# -*- coding: utf-8 -*-

"""Reading SEAMM's H5MD files.

:func:`open_md` opens one run, given as one file or as the list of its segment
files, and returns an :class:`MDData`. Time series are read with h5py on their
own time axis. MDAnalysis is used only for structural analysis, through
:meth:`MDData.universe`, because it attaches observables to frames by index and
so misaligns any observable sampled more often than the frames.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import h5py
import numpy as np

from .units import to_pint


@dataclass
class TimeSeries:
    """A time series or a per-particle trajectory read from an H5MD file.

    Attributes
    ----------
    name : str
        The observable or property, e.g. "temperature" or "molecules/position".
    values : numpy.ndarray
        The values, with the first index running over the samples.
    times : numpy.ndarray
        The times, in ``time_units``.
    steps : numpy.ndarray
        The MD steps.
    units : pint.Unit
        The units of the values.
    time_units : pint.Unit
        The units of the times.
    average_window : int
        The number of MD steps each value averages over; 1 for instantaneous
        values.
    kind : str or None
        "correlation" for a correlation function accumulated by the code.
    """

    name: str
    values: np.ndarray
    times: np.ndarray
    steps: np.ndarray
    units: object
    time_units: object
    average_window: int = 1
    kind: str = None
    attrs: dict = field(default_factory=dict)

    def __len__(self):
        return self.values.shape[0]

    @property
    def dt(self):
        """The sampling interval as a pint quantity, or None for fewer than 2."""
        if len(self.times) < 2:
            return None
        return float(np.median(np.diff(self.times))) * self.time_units

    def to(self, units=None, time_units=None):
        """A copy in other units.

        Parameters
        ----------
        units : str or pint.Unit = None
            The units of the values; None leaves them as they are.
        time_units : str or pint.Unit = None
            The units of the times; None leaves them as they are.

        Returns
        -------
        TimeSeries
        """
        values, unit = self.values, self.units
        if units is not None:
            q = (values * unit).to(units)
            values, unit = q.magnitude, q.units
        times, tunit = self.times, self.time_units
        if time_units is not None:
            q = (times * tunit).to(time_units)
            times, tunit = q.magnitude, q.units
        return TimeSeries(
            name=self.name,
            values=values,
            times=times,
            steps=self.steps,
            units=unit,
            time_units=tunit,
            average_window=self.average_window,
            kind=self.kind,
            attrs=dict(self.attrs),
        )


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
        return [_decode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _group_to_dict(group):
    result = {key: _decode(value) for key, value in group.attrs.items()}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = _group_to_dict(item)
        else:
            result[key] = _decode(item[()])
    return result


def open_md(paths):
    """Open an MD run's H5MD file, or the list of its segment files.

    Parameters
    ----------
    paths : str, pathlib.Path or list
        One file, or the segments of one run in order.

    Returns
    -------
    MDData
    """
    return MDData(paths)


class MDData:
    """One MD run in SEAMM's H5MD layout.

    Use it as a context manager, or call :meth:`close`. Several segment files
    are read as one run: time series and frames are joined in order, and a sample
    repeated at the boundary between segments is used once.
    """

    def __init__(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        if len(self.paths) == 0:
            raise ValueError("No H5MD files given.")
        self._files = [h5py.File(p, "r") for p in self.paths]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def close(self):
        for f in self._files:
            f.close()
        self._files = []

    @property
    def first(self):
        """The first (or only) h5py file."""
        return self._files[0]

    # Metadata

    @property
    def creator(self):
        """The program that wrote the file and its version."""
        attrs = self.first["h5md/creator"].attrs
        return _decode(attrs["name"]), _decode(attrs.get("version", ""))

    @property
    def parameters(self):
        """The run's parameters, without SEAMM's provenance."""
        if "parameters" not in self.first:
            return {}
        result = _group_to_dict(self.first["parameters"])
        result.pop("seamm", None)
        return result

    @property
    def provenance(self):
        """SEAMM's provenance (``parameters/seamm``)."""
        if "parameters/seamm" not in self.first:
            return {}
        return _group_to_dict(self.first["parameters/seamm"])

    @property
    def periodic(self):
        """Whether the system is periodic, from the box of the particles."""
        if "particles" in self.first:
            for group in self.first["particles"].values():
                if "box" in group:
                    boundary = _decode(group["box"].attrs.get("boundary", []))
                    return len(boundary) > 0 and all(b == "periodic" for b in boundary)
        return bool(self.provenance.get("periodic", True))

    # Observables

    def observables(self):
        """The names of the observables, including "user/<name>"."""
        names = []
        if "observables" in self.first:

            def visit(name, item):
                if isinstance(item, h5py.Group) and "value" in item:
                    names.append(name)

            self.first["observables"].visititems(visit)
        return sorted(names)

    def has_observable(self, name):
        return f"observables/{name}/value" in self.first

    def _series(self, path, name, units=None, time_units=None):
        values, times, steps = [], [], []
        last_step = None
        for f in self._files:
            if path not in f:
                continue
            g = f[path]
            step = np.atleast_1d(g["step"][()])
            time = np.atleast_1d(g["time"][()])
            value = g["value"][()]
            if last_step is not None:
                keep = step > last_step
                step, time, value = step[keep], time[keep], value[keep]
            if len(step) > 0:
                last_step = step[-1]
            values.append(value)
            times.append(time)
            steps.append(step)
        if len(values) == 0:
            raise KeyError(f"'{name}' is not in {', '.join(map(str, self.paths))}")
        g = next(f[path] for f in self._files if path in f)
        value_attrs = {k: _decode(v) for k, v in g["value"].attrs.items()}
        series = TimeSeries(
            name=name,
            values=np.concatenate(values),
            times=np.concatenate(times),
            steps=np.concatenate(steps),
            units=to_pint(value_attrs.pop("unit", None)),
            time_units=to_pint(_decode(g["time"].attrs.get("unit", None))),
            average_window=int(value_attrs.pop("seamm_average_window", 1)),
            kind=value_attrs.pop("seamm_kind", None),
            attrs=value_attrs,
        )
        if units is not None or time_units is not None:
            series = series.to(units, time_units)
        return series

    def observable(self, name, units=None, time_units=None):
        """Read an observable on its own time axis.

        Parameters
        ----------
        name : str
            E.g. "temperature", "pressure_tensor" or "user/<name>".
        units, time_units : str or pint.Unit = None
            Convert to these units; by default they are as the code wrote them.

        Returns
        -------
        TimeSeries
        """
        return self._series(f"observables/{name}", name, units, time_units)

    # Particles

    def particle_groups(self):
        """The particle groups, e.g. ["all", "molecules"]."""
        if "particles" not in self.first:
            return []
        return sorted(self.first["particles"].keys())

    def has_particles(self, name, group="all"):
        """Whether a per-particle property, e.g. "position", was sampled."""
        return f"particles/{group}/{name}/value" in self.first

    def _fixed(self, group, name):
        path = f"particles/{group}/{name}"
        if path not in self.first:
            return None
        return self.first[path][()]

    def ids(self, group="all"):
        return self._fixed(group, "id")

    def species(self, group="all"):
        return self._fixed(group, "species")

    def masses(self, group="all"):
        return self._fixed(group, "mass")

    def charges(self, group="all"):
        return self._fixed(group, "charge")

    def n_frames(self, name="position", group="all"):
        """The number of frames of a per-particle property in all segments."""
        return sum(len(t) for t, _ in self._frame_ranges(f"particles/{group}/{name}"))

    def cell(self, group="all", units="Å"):
        """The box of the first frame as a 3×3 array of cell vectors (rows).

        Returns None for a non-periodic system.
        """
        path = f"particles/{group}/box/edges"
        if path not in self.first:
            return None
        series = self._series(path, "box", units=units)
        edges = series.values[0]
        if edges.shape == (3,):
            edges = np.diag(edges)
        return edges

    def box(self, group="all", units="Å"):
        """The box at every recorded time, as a TimeSeries of 3×3 matrices."""
        series = self._series(f"particles/{group}/box/edges", "box", units=units)
        if series.values.ndim == 2:
            series.values = np.array([np.diag(e) for e in series.values])
        return series

    def particle_series(self, name, group="molecules", units=None, time_units=None):
        """Read a per-particle property whole, e.g. the molecular centres of mass.

        Use this only when it is small; :meth:`iter_frames` streams large
        trajectories.

        Returns
        -------
        TimeSeries
            values has the shape (n_frames, n_particles, 3).
        """
        path = f"particles/{group}/{name}"
        return self._series(path, f"{group}/{name}", units, time_units)

    def _frame_ranges(self, path):
        """The (steps, dataset) of each segment, without repeated frames."""
        result = []
        last_step = None
        for f in self._files:
            if path not in f:
                continue
            steps = np.atleast_1d(f[path]["step"][()])
            first = 0
            if last_step is not None:
                first = int(np.searchsorted(steps, last_step, side="right"))
            if len(steps) > 0:
                last_step = steps[-1]
            result.append((steps[first:], (f[path], first)))
        return result

    def iter_frames(
        self, group="all", fields=("position",), chunk=100, units=None, time_units=None
    ):
        """Stream frames of a trajectory in chunks.

        Parameters
        ----------
        group : str = "all"
            The particle group.
        fields : [str]
            The properties to read, e.g. ("position", "velocity"). They must have
            been sampled at the same steps.
        chunk : int = 100
            The number of frames per chunk.
        units : {str: str} = None
            Units to convert each field to, e.g. {"position": "Å"}.
        time_units : str = None
            Units for the times.

        Yields
        ------
        (numpy.ndarray, numpy.ndarray, {str: numpy.ndarray})
            The steps, the times and the values of each field, with the shape
            (n, n_particles, 3), for each chunk of frames.
        """
        units = units or {}
        first = f"particles/{group}/{fields[0]}"
        for steps, (g, offset) in self._frame_ranges(first):
            f = g.file
            time_unit = to_pint(_decode(g["time"].attrs.get("unit", None)))
            factors = {}
            for name in fields:
                data = f[f"particles/{group}/{name}"]
                if data["step"].shape != g["step"].shape:
                    raise ValueError(
                        f"'{name}' and '{fields[0]}' were not sampled at the same steps"
                    )
                unit = to_pint(_decode(data["value"].attrs.get("unit", None)))
                factors[name] = 1.0
                if name in units:
                    factors[name] = (1.0 * unit).m_as(units[name])
            tfactor = 1.0 if time_units is None else (1.0 * time_unit).m_as(time_units)
            for start in range(0, len(steps), chunk):
                lo = offset + start
                hi = offset + min(start + chunk, len(steps))
                values = {}
                for name in fields:
                    value = f[f"particles/{group}/{name}/value"][lo:hi]
                    if factors[name] != 1.0:
                        value = value * factors[name]
                    values[name] = value
                times = g["time"][lo:hi] * tfactor
                yield g["step"][lo:hi], times, values

    # MDAnalysis

    def universe(self, configuration=None, names=None):
        """An MDAnalysis Universe for the atoms, for structural analysis.

        The topology comes from the file (species, masses and charges) and, if
        given, from the molsystem configuration (atom names, bonds, and molecules
        as residues). MDAnalysis is imported here only, so it is needed only by
        the steps that use it (``pip install seamm-md-analysis[structure]``).

        **Do not use its per-frame observables** (``ts.data``): MDAnalysis attaches
        them to frames by index, ignoring their own steps.

        Parameters
        ----------
        configuration : molsystem._Configuration = None
            The configuration that was simulated, in the order of the atoms in the
            file.
        names : [str] = None
            Atom names; by default the element symbols.

        Returns
        -------
        MDAnalysis.Universe
        """
        import MDAnalysis as mda

        n_atoms = len(self.ids("all"))
        species = self.species("all")
        masses = self.masses("all")
        charges = self.charges("all")

        molecules = None
        bonds = None
        if configuration is not None:
            molecules = configuration.find_molecules(as_indices=True)
            index = {atom_id: i for i, atom_id in enumerate(configuration.atoms.ids)}
            tmp = configuration.bonds.get_as_dict()
            bonds = [(index[i], index[j]) for i, j in zip(tmp["i"], tmp["j"])]
            if names is None:
                names = configuration.atoms.symbols

        if molecules is None or len(molecules) == 0:
            molecules = [list(range(n_atoms))]
        resindices = np.zeros(n_atoms, dtype=int)
        for i, molecule in enumerate(molecules):
            resindices[molecule] = i

        u = mda.Universe.empty(
            n_atoms,
            n_residues=len(molecules),
            atom_resindex=resindices,
            trajectory=True,
        )
        if species is not None:
            from MDAnalysis.guesser.tables import SYMB2Z

            Z2SYMB = {z: s for s, z in SYMB2Z.items()}
            elements = [Z2SYMB.get(int(z), "X") for z in species]
            u.add_TopologyAttr("elements", elements)
            if names is None:
                names = elements
        if names is not None:
            u.add_TopologyAttr("names", list(names))
        if masses is not None:
            u.add_TopologyAttr("masses", masses)
        if charges is not None:
            u.add_TopologyAttr("charges", charges)
        u.add_TopologyAttr("resids", np.arange(1, len(molecules) + 1))
        if bonds is not None and len(bonds) > 0:
            u.add_TopologyAttr("bonds", bonds)

        files = [str(p) for p in self.paths]
        try:
            u.load_new(files if len(files) > 1 else files[0], format="H5MD")
        except RuntimeError as e:
            if "unit" not in str(e).lower():
                raise
            # A unit MDAnalysis does not know (e.g. eV Å-1 forces): positions are
            # in Angstrom anyway, so read without converting.
            u.load_new(
                files if len(files) > 1 else files[0],
                format="H5MD",
                convert_units=False,
            )
        return u


#: The choices for which earlier MD runs an analysis reads
md_run_sources = (
    "the last MD run",
    "the last MD run of each loop iteration",
)


def select_md_runs(records, source="the last MD run", name=None):
    """Choose the MD runs an analysis reads from the records in ``_md_runs``.

    Parameters
    ----------
    records : [dict]
        The records MD steps added to ``_md_runs``, oldest first.
    source : str
        "the last MD run", or "the last MD run of each loop iteration": every run by
        the same MD step as the last one, which in a loop are the runs of the
        iterations, treated as independent replicas.
    name : str = None
        Only consider runs that sampled this observable or per-particle property,
        e.g. "heat_flux" or "molecules/position".

    Returns
    -------
    [dict]
        The chosen records, oldest first; empty if there are none.
    """
    records = list(records or [])
    if name is not None:
        records = [
            r
            for r in records
            if name in r.get("observables", {}) or name in r.get("particles", {})
        ]
    if len(records) == 0:
        return []
    last = records[-1]
    if source == "the last MD run":
        return [last]
    if source == "the last MD run of each loop iteration":
        return [r for r in records if _same_step(r.get("step"), last.get("step"))]
    raise ValueError(f"Unknown source of MD runs: '{source}'")


def _same_step(a, b):
    """Whether two runs were made by the same MD step, perhaps in different loop
    iterations.

    A run's step is labelled by the step's id and title, e.g. "4.iter_2.0.4 NVT
    dynamics". The numbers in the id are the positions of the steps; anything else is
    the directory of a loop iteration (``iter_2``, or a system or configuration name),
    which differs from one iteration to the next.
    """
    if a is None or b is None:
        return False
    if a == b:
        return True
    id_a, _, title_a = a.partition(" ")
    id_b, _, title_b = b.partition(" ")
    if title_a != title_b:
        return False
    parts_a = id_a.split(".")
    parts_b = id_b.split(".")
    if len(parts_a) != len(parts_b):
        return False
    return all(
        x == y or not (x.isdigit() or y.isdigit()) for x, y in zip(parts_a, parts_b)
    )
