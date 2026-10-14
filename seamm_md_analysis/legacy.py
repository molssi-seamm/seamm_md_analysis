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
