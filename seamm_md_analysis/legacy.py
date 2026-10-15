# -*- coding: utf-8 -*-

"""Readers for the text files the LAMMPS step has written so far.

These are the files that SEAMM's H5MD output will replace: the
``!MolSSI trajectory 2.0`` time series (``*.trj``), the
``!MolSSI vector_trajectory 2.0`` per-molecule series (``com_*.trj``) and the
LAMMPS dump files (``*.dump_trj``). The readers remain so that older jobs can
still be analysed.
"""

import json

import numpy as np


def _read_header(line, path, kind):
    tmp = line.split()
    if len(tmp) < 3:
        raise RuntimeError(f"Bad header for {path}: {line}")
    if tmp[0] != "!MolSSI":
        raise RuntimeError(f"Not a MolSSI file? {path}: {line}")
    if tmp[1] != kind:
        raise RuntimeError(f"Not a {kind} file? {path}: {line}")
    if tmp[2][0] != "2":
        raise RuntimeError(f"Can only handle version 2 {kind} files. {path}: {line}")
    return json.loads(" ".join(tmp[3:]))


def read_trajectory(path):
    """Read a ``!MolSSI trajectory 2.0`` time series, such as the heat flux.

    The file starts with a header line holding JSON metadata, then a line of
    column names, then one line per sample. When the names do not include the
    first column, that column is the LAMMPS timestep and is returned as
    "tstep".

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    dict
        The metadata: "code", "type", "dt", "tunits", "nsteps", and "fields",
        the names of the columns.
    {str: numpy.ndarray}
        The columns, by name.
    """
    with open(path) as fd:
        metadata = _read_header(fd.readline(), path, "trajectory")
        for line in fd:
            if line[0] != "!":
                break
        names = line.split()
        data = np.loadtxt(fd, comments="!", ndmin=2)
    if data.shape[0] == 0:
        data = np.zeros((0, len(names)))
    elif data.shape[1] == len(names) + 1:
        names = ["tstep"] + names
    elif data.shape[1] != len(names):
        raise RuntimeError(
            f"{path} has {data.shape[1]} columns but names {len(names)}: {names}"
        )
    metadata["fields"] = names
    return metadata, {name: data[:, i] for i, name in enumerate(names)}


def read_dump_trajectory(path):
    """Read a LAMMPS dump file with an id and three values per atom.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    dict
        "timesteps", and when the dump has times, "times", "dt" and "tunits"
        ("ps" for LAMMPS metal units, otherwise "fs").
    numpy.ndarray(n_frames, n_atoms, 3)
        The values.

    Note
    ----
    Each frame looks like this::

        ITEM: TIMESTEP
        0
        ITEM: NUMBER OF ATOMS
        256
        ITEM: BOX BOUNDS pp pp pp
        0.0000000000000000e+00 1.5900375866964442e+01
        0.0000000000000000e+00 1.5900375866964442e+01
        0.0000000000000000e+00 1.5900375866964442e+01
        ITEM: ATOMS id xu yu zu
        1 20.0993 14.7149 -5.20836
        2 8.78739 12.8425 -25.9197
        ...

    optionally preceded by ``ITEM: UNITS`` and ``ITEM: TIME``.
    """
    timesteps = []
    times = []
    units = None
    data = []
    n_atoms = 0
    with open(path) as fd:
        lines = iter(fd)
        for line in lines:
            if line.startswith("ITEM:"):
                if "TIMESTEP" in line:
                    timesteps.append(int(next(lines)))
                elif "TIME" in line:
                    times.append(float(next(lines)))
                elif "UNITS" in line:
                    units = next(lines).strip()
                elif "NUMBER OF ATOMS" in line:
                    n_atoms = int(next(lines))
                elif "BOX BOUNDS" in line:
                    for _ in range(3):
                        next(lines)
                elif "ATOMS" in line:
                    frame = []
                    for _ in range(n_atoms):
                        _, x, y, z = next(lines).split()
                        frame.append([float(x), float(y), float(z)])
                    data.append(frame)
    metadata = {"timesteps": timesteps}
    if len(times) > 0:
        metadata["times"] = times
        if len(times) > 1:
            metadata["dt"] = times[1] - times[0]
        if units is not None and units.lower() == "metal":
            metadata["tunits"] = "ps"
        else:
            metadata["tunits"] = "fs"
    return metadata, np.array(data)


def read_vector_trajectory(path):
    """Read a ``!MolSSI vector_trajectory 2.0`` file, such as molecular COMs.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    dict
        The metadata: "code", "type", "dt", "tunits", "nsteps", and "fields",
        the names of the values per row.
    numpy.ndarray(n_frames, n_rows, n_values)
        The values.
    """
    data = []
    with open(path) as fd:
        lines = iter(fd)
        metadata = _read_header(next(lines), path, "vector_trajectory")

        # Skip any commented header lines
        for line in lines:
            if line[0] != "!":
                break
        metadata["fields"] = line.split()[1:]

        for line in lines:
            _, n = line.split()
            frame = []
            for i in range(int(n)):
                frame.append([float(v) for v in next(lines).split()[1:]])
            data.append(frame)
    return metadata, np.array(data)


def lammps_box_to_cell(bounds, tilts=None):
    """The cell vectors (rows) from a LAMMPS dump's box bounds.

    Parameters
    ----------
    bounds : 3×2 array_like
        The (lo, hi) pairs from the ``ITEM: BOX BOUNDS`` lines. For a triclinic
        box these are LAMMPS's bounding-box values.
    tilts : (xy, xz, yz) = None
        The tilt factors of a triclinic box.

    Returns
    -------
    numpy.ndarray(3, 3)
    """
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    xy = xz = yz = 0.0
    if tilts is not None:
        xy, xz, yz = tilts
        xlo -= min(0.0, xy, xz, xy + xz)
        xhi -= max(0.0, xy, xz, xy + xz)
        ylo -= min(0.0, yz)
        yhi -= max(0.0, yz)
    return np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ]
    )


def iter_dump_frames(path):
    """Stream the frames of a LAMMPS custom dump file, gzipped or not.

    Parameters
    ----------
    path : str or pathlib.Path
        The file; a name ending in ".gz" is read with gzip.

    Yields
    ------
    dict
        For each frame: "step"; "time" and "units" if the dump has them; "cell",
        the 3×3 cell vectors as rows; "periodic", from the boundary flags; and
        "columns", {name: numpy.ndarray} for each per-atom column, sorted by the
        atom id. Columns that are not numbers (e.g. "element") are strings.
    """
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    frame = {}
    units = None
    with opener(path, "rt") as fd:
        lines = iter(fd)
        for line in lines:
            if not line.startswith("ITEM:"):
                continue
            if "TIMESTEP" in line:
                frame["step"] = int(next(lines))
            elif "TIME" in line:
                frame["time"] = float(next(lines))
            elif "UNITS" in line:
                units = next(lines).strip()
            elif "NUMBER OF ATOMS" in line:
                n_atoms = int(next(lines))
            elif "BOX BOUNDS" in line:
                words = line.split()[3:]
                triclinic = len(words) > 0 and words[0] == "xy"
                flags = words[3:] if triclinic else words
                values = [[float(v) for v in next(lines).split()] for _ in range(3)]
                bounds = [v[0:2] for v in values]
                tilts = [v[2] for v in values] if triclinic else None
                frame["cell"] = lammps_box_to_cell(bounds, tilts)
                frame["periodic"] = all(flag == "pp" for flag in flags)
            elif "ATOMS" in line:
                names = line.split()[2:]
                rows = [next(lines).split() for _ in range(n_atoms)]
                columns = {}
                for i, name in enumerate(names):
                    column = [row[i] for row in rows]
                    try:
                        columns[name] = np.array(column, dtype=float)
                    except ValueError:
                        columns[name] = np.array(column)
                if "id" in columns:
                    columns["id"] = columns["id"].astype(int)
                    order = np.argsort(columns["id"])
                    columns = {k: v[order] for k, v in columns.items()}
                frame["columns"] = columns
                if units is not None:
                    frame["units"] = units
                yield frame
                frame = {}


def iter_vector_frames(path):
    """Stream a ``!MolSSI vector_trajectory 2.0`` file, such as molecular COMs.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Yields
    ------
    dict, then (int, numpy.ndarray(n_rows, n_values)) for each frame
        First the metadata ("code", "type", "dt", "tunits", "nsteps", "fields"),
        then the step and the values of each frame.
    """
    with open(path) as fd:
        lines = iter(fd)
        metadata = _read_header(next(lines), path, "vector_trajectory")
        for line in lines:
            if line[0] != "!":
                break
        metadata["fields"] = line.split()[1:]
        yield metadata
        for line in lines:
            step, n = line.split()
            frame = np.array(
                [[float(v) for v in next(lines).split()[1:]] for _ in range(int(n))]
            )
            yield int(step), frame
