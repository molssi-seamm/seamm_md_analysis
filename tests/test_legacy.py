# -*- coding: utf-8 -*-

"""Tests of the readers for the LAMMPS step's text files."""

from pathlib import Path

import numpy as np
import pytest

from seamm_md_analysis import legacy

DATA = Path(__file__).parent / "data"


def test_read_trajectory_heat_flux():
    metadata, columns = legacy.read_trajectory(DATA / "heat_flux.trj")
    assert metadata["type"] == "heat flux"
    assert metadata["dt"] == 500.0 and metadata["tunits"] == "fs"
    # The names do not include the timestep column, so it is added
    assert metadata["fields"] == ["tstep", "Jx", "Jy", "Jz"]
    assert len(columns["Jx"]) == 1001
    assert columns["tstep"][1] == 250
    assert columns["Jx"][0] == pytest.approx(1.57624e13)
    assert columns["Jz"][-1] == pytest.approx(2.18407e13)


def test_read_trajectory_named_timestep(tmp_path):
    path = tmp_path / "state.trj"
    path.write_text(
        '!MolSSI trajectory 2.0 {"code":"LAMMPS","type":"NVT","dt":1.0,'
        '"tunits":"fs","nsteps":2}\n'
        "tstep t T\n"
        "0 0.0 300.0\n"
        "! a comment\n"
        "1 1.0 301.5\n"
    )
    metadata, columns = legacy.read_trajectory(path)
    assert metadata["fields"] == ["tstep", "t", "T"]
    np.testing.assert_allclose(columns["T"], [300.0, 301.5])


def test_read_trajectory_bad_files(tmp_path):
    path = tmp_path / "x.trj"
    path.write_text('!MolSSI vector_trajectory 2.0 {"dt":1}\nJx\n1 2\n')
    with pytest.raises(RuntimeError, match="Not a trajectory"):
        legacy.read_trajectory(path)
    path.write_text('!MolSSI trajectory 1.0 {"dt":1}\nJx\n1 2\n')
    with pytest.raises(RuntimeError, match="version 2"):
        legacy.read_trajectory(path)
    path.write_text('!MolSSI trajectory 2.0 {"dt":1}\nJx\n1 2 3 4\n')
    with pytest.raises(RuntimeError, match="columns"):
        legacy.read_trajectory(path)


def test_read_vector_trajectory():
    metadata, data = legacy.read_vector_trajectory(DATA / "com_positions.trj")
    assert metadata["type"] == "com positions"
    assert metadata["dt"] == 50.0
    assert metadata["fields"] == ["com_x", "com_y", "com_z"]
    assert data.shape == (3, 250, 3)
    np.testing.assert_allclose(data[0, 0], [36.9984, 23.275, 9.19414])


def test_read_dump_trajectory():
    metadata, data = legacy.read_dump_trajectory(DATA / "atomic_velocities.dump_trj")
    assert data.shape == (3, 128, 3)
    assert metadata["tunits"] == "ps"  # metal units
    assert metadata["dt"] == pytest.approx(3.2)
    assert metadata["timesteps"][:2] == [0, 3200]
    np.testing.assert_allclose(data[0, 0], [-11.984, 9.32727, -5.9982])


def test_read_dump_single_frame(tmp_path):
    path = tmp_path / "one.dump_trj"
    path.write_text(
        "ITEM: TIME\n0\nITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id xu yu zu\n"
        "1 0.5 0.25 0.125\n"
    )
    metadata, data = legacy.read_dump_trajectory(path)
    assert data.shape == (1, 1, 3)
    assert "dt" not in metadata and metadata["tunits"] == "fs"


def test_iter_dump_frames_matches_reader():
    from seamm_md_analysis import iter_dump_frames, read_dump_trajectory

    path = DATA / "atomic_velocities.dump_trj"
    metadata, values = read_dump_trajectory(path)
    frames = list(iter_dump_frames(path))
    assert [f["step"] for f in frames] == metadata["timesteps"]
    assert frames[0]["units"] == "metal"
    assert frames[0]["periodic"]
    assert np.allclose(np.diag(frames[0]["cell"]), 12.688539597631690)
    stacked = np.stack(
        [np.column_stack([f["columns"][c] for c in ("vx", "vy", "vz")]) for f in frames]
    )
    # read_dump_trajectory keeps the file's order; the dump is sorted by id
    assert np.allclose(stacked, values)


def test_iter_dump_frames_triclinic_gzip(tmp_path):
    import gzip

    from seamm_md_analysis import iter_dump_frames

    lx, ly, lz, xy, xz, yz = 10.0, 11.0, 12.0, 1.5, -0.5, 0.75
    xlo, ylo, zlo = 1.0, 2.0, 3.0
    # LAMMPS writes the bounding box of the triclinic cell
    xlo_b = xlo + min(0.0, xy, xz, xy + xz)
    xhi_b = xlo + lx + max(0.0, xy, xz, xy + xz)
    ylo_b = ylo + min(0.0, yz)
    yhi_b = ylo + ly + max(0.0, yz)
    text = (
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        f"{xlo_b} {xhi_b} {xy}\n{ylo_b} {yhi_b} {xz}\n{zlo} {zlo + lz} {yz}\n"
        "ITEM: ATOMS id element q xu yu zu\n"
        "2 H 0.4 1 2 3\n1 O -0.8 4 5 6\n"
    )
    path = tmp_path / "t.dump_trj.gz"
    with gzip.open(path, "wt") as fd:
        fd.write(text)
    (frame,) = list(iter_dump_frames(path))
    assert frame["step"] == 10
    assert np.allclose(frame["cell"], [[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    assert list(frame["columns"]["id"]) == [1, 2]
    assert list(frame["columns"]["element"]) == ["O", "H"]
    assert np.allclose(frame["columns"]["xu"], [4, 1])


def test_iter_vector_frames_matches_reader():
    from seamm_md_analysis import iter_vector_frames, read_vector_trajectory

    path = DATA / "com_positions.trj"
    metadata, values = read_vector_trajectory(path)
    frames = iter_vector_frames(path)
    header = next(frames)
    assert header["fields"] == metadata["fields"]
    steps, arrays = zip(*frames)
    assert steps[0] == 0
    assert np.allclose(np.stack(arrays), values)
