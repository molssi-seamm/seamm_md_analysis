#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for writing and reading SEAMM's H5MD files."""

import h5py
import numpy as np
import pint
import pytest

from seamm_md_analysis import H5MDWriter, open_md
from seamm_md_analysis.units import to_h5md, to_pint

ureg = pint.get_application_registry()

cell = np.array([[10.0, 0.0, 0.0], [1.0, 11.0, 0.0], [0.5, 0.25, 12.0]])


@pytest.mark.parametrize(
    "pint_unit, h5md",
    [
        ("Å", "Angstrom"),
        ("Å/ps", "Angstrom ps-1"),
        ("Å/fs", "Angstrom fs-1"),
        ("kcal/mol", "kcal mol-1"),
        ("kcal/mol/Å", "kcal mol-1 Angstrom-1"),
        ("eV/Å", "eV Angstrom-1"),
        ("g/cm**3", "g cm-3"),
        ("W/m**2", "W m-2"),
        ("e*Å", "e Angstrom"),
        ("atm", "atm"),
        ("K", "K"),
        ("fs", "fs"),
        ("Å**3", "Angstrom3"),
    ],
)
def test_units_round_trip(pint_unit, h5md):
    assert to_h5md(pint_unit) == h5md
    assert to_pint(h5md) == ureg.Unit(pint_unit)


def test_units_dimensionless_and_errors():
    assert to_pint(None) == ureg.dimensionless
    assert to_pint("") == ureg.dimensionless
    assert to_pint(b"K") == ureg.kelvin
    with pytest.raises(ValueError):
        to_pint("kcal/mol")


def write_run(path, periodic=True, n_atoms=4, n_frames=5, every=4, first_step=0):
    """A small run: positions every `every` steps, temperature every step."""
    rng = np.random.default_rng(1)
    steps = first_step + np.arange(n_frames) * every
    positions = rng.random((n_frames, n_atoms, 3)) * 5
    obs_steps = np.arange(steps[0], steps[-1] + 1)
    temperature = 300.0 + obs_steps
    with H5MDWriter(path, creator="SEAMM test", version="1.0", author="tester") as w:
        w.particles(
            ids=np.arange(1, n_atoms + 1),
            species=[8, 1, 1, 6][:n_atoms],
            masses=[16.0, 1.0, 1.0, 12.0][:n_atoms],
            charges=[-0.8, 0.4, 0.4, 0.0][:n_atoms],
            cell=cell if periodic else None,
            periodic=periodic,
        )
        appender = w.particle_appender(
            "position", n_atoms, "Angstrom", "fs", attrs={"seamm_unwrapped": 1}
        )
        for step, frame in zip(steps, positions):
            appender.append(step, step * 0.5, frame)
        w.observable("temperature", obs_steps, obs_steps * 0.5, temperature, "K")
        tensor = np.repeat(np.eye(3)[None], len(obs_steps), 0)
        w.observable("pressure_tensor", obs_steps, obs_steps * 0.5, tensor, "atm", "fs")
        w.observable("user/flux", obs_steps, obs_steps * 0.5, obs_steps, "")
        w.parameters({"ensemble": "NVT", "timestep": 0.5, "thermostat": {"T": 300}})
        w.provenance(
            {"job": 12, "code": "LAMMPS", "model": {"name": "OPLS-AA"}},
            segments=[{"first_step": 0, "last_step": int(steps[-1]), "node": "n1"}],
        )
    return steps, positions, obs_steps, temperature


def test_round_trip(tmp_path):
    path = tmp_path / "md.h5md"
    steps, positions, obs_steps, temperature = write_run(path)
    assert path.exists()
    assert not (tmp_path / "md.h5md.tmp").exists()

    with open_md(path) as md:
        assert md.creator == ("SEAMM test", "1.0")
        assert md.periodic
        assert md.parameters["ensemble"] == "NVT"
        assert md.parameters["thermostat"]["T"] == 300
        assert md.provenance["job"] == 12
        assert md.provenance["model"] == {"name": "OPLS-AA"}
        assert md.provenance["segments"]["node"] == ["n1"]
        assert md.observables() == ["pressure_tensor", "temperature", "user/flux"]
        assert md.has_observable("temperature")
        assert not md.has_observable("heat_flux")

        T = md.observable("temperature")
        assert np.allclose(T.values, temperature)
        assert np.array_equal(T.steps, obs_steps)
        assert T.units == ureg.kelvin
        assert T.average_window == 1
        assert T.dt == 0.5 * ureg.fs
        assert np.allclose(
            md.observable("temperature", time_units="ps").times[-1], 0.008
        )
        assert md.observable("pressure_tensor").values.shape == (len(obs_steps), 3, 3)
        P = md.observable("pressure_tensor", units="bar")
        assert np.isclose(P.values[0, 0, 0], 1.01325)

        assert np.allclose(md.cell(), cell)
        assert np.allclose(md.cell(units="nm"), cell / 10)
        assert md.box().values.shape == (len(steps), 3, 3)
        assert list(md.species()) == [8, 1, 1, 6]
        assert np.allclose(md.charges(), [-0.8, 0.4, 0.4, 0.0])
        assert md.n_frames() == len(steps)

        chunks = list(md.iter_frames(chunk=2, units={"position": "nm"}))
        assert [len(s) for s, _, _ in chunks] == [2, 2, 1]
        joined = np.concatenate([v["position"] for _, _, v in chunks])
        assert np.allclose(joined, positions / 10, atol=1e-6)

        with h5py.File(path) as f:
            assert f["particles/all/position/value"].dtype == np.float32
            assert f["particles/all/position/value"].attrs["seamm_unwrapped"] == 1
            assert f["observables/temperature/value"].compression == "gzip"
            assert tuple(f["h5md"].attrs["version"]) == (1, 1)


def test_non_periodic(tmp_path):
    path = tmp_path / "md.h5md"
    write_run(path, periodic=False)
    with open_md(path) as md:
        assert not md.periodic
        assert md.cell() is None
    with h5py.File(path) as f:
        assert "edges" not in f["particles/all/box"]


def test_failed_write_leaves_nothing(tmp_path):
    path = tmp_path / "md.h5md"
    with pytest.raises(RuntimeError):
        with H5MDWriter(path) as w:
            w.particles(ids=[1], periodic=False)
            raise RuntimeError("the conversion failed")
    assert not path.exists()
    assert not (tmp_path / "md.h5md.tmp").exists()


def test_observable_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        with H5MDWriter(tmp_path / "md.h5md") as w:
            w.observable("temperature", [0, 1], [0.0], [1.0, 2.0], "K")


def test_segments_are_joined(tmp_path):
    """A frame repeated at the boundary between segments is used once."""
    s1, p1, o1, t1 = write_run(tmp_path / "md.seg-0001.h5md", n_frames=3)
    s2, p2, o2, t2 = write_run(tmp_path / "md.seg-0002.h5md", n_frames=3, first_step=8)
    paths = [tmp_path / "md.seg-0001.h5md", tmp_path / "md.seg-0002.h5md"]
    with open_md(paths) as md:
        T = md.observable("temperature")
        assert np.array_equal(T.steps, np.arange(0, 17))
        assert md.n_frames() == 5
        steps = np.concatenate([s for s, _, _ in md.iter_frames(chunk=2)])
        assert np.array_equal(steps, [0, 4, 8, 12, 16])
        frames = np.concatenate([v["position"] for _, _, v in md.iter_frames()])
        assert np.allclose(frames, np.concatenate([p1, p2[1:]]), atol=1e-6)


def test_npt_box(tmp_path):
    path = tmp_path / "md.h5md"
    with H5MDWriter(path) as w:
        w.particles(ids=[1, 2], cell=cell)
        box = w.box_appender(time_unit="fs")
        for i in range(3):
            box.append(i * 10, i * 5.0, cell * (1 + 0.01 * i))
    with open_md(path) as md:
        series = md.box()
        assert np.array_equal(series.steps, [0, 10, 20])
        assert np.allclose(series.values[2], cell * 1.02)


def test_molecules_group(tmp_path):
    path = tmp_path / "md.h5md"
    com = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    with H5MDWriter(path) as w:
        w.particles(ids=[1, 2, 3], cell=cell)
        w.particles(group="molecules", ids=[0, 1, 2], cell=cell)
        appender = w.particle_appender("position", 3, "Angstrom", group="molecules")
        for i, frame in enumerate(com):
            appender.append(i * 100, i * 100.0, frame)
    with open_md(path) as md:
        assert md.particle_groups() == ["all", "molecules"]
        assert md.has_particles("position", group="molecules")
        assert not md.has_particles("position")
        series = md.particle_series("position", units="nm")
        assert series.values.shape == (2, 3, 3)
        assert np.allclose(series.values, com / 10)


def test_mdanalysis_universe(tmp_path):
    """MDAnalysis reads the file; our observables keep their own time axis."""
    mda = pytest.importorskip("MDAnalysis")
    path = tmp_path / "md.h5md"
    steps, positions, obs_steps, temperature = write_run(path)
    with open_md(path) as md:
        u = md.universe()
        assert isinstance(u, mda.Universe)
        assert len(u.trajectory) == len(steps)
        assert list(u.atoms.elements) == ["O", "H", "H", "C"]
        assert np.allclose(u.atoms.charges, [-0.8, 0.4, 0.4, 0.0])
        u.trajectory[2]
        assert np.allclose(u.atoms.positions, positions[2], atol=1e-5)
        assert np.allclose(u.dimensions[:3], np.linalg.norm(cell, axis=1), atol=1e-4)

        # The trap: MDAnalysis gives frame 1 the temperature of observable sample 1
        # (step 1), not of step 4. Our reader keeps the observable's own steps.
        T = md.observable("temperature")
        frame_step = steps[1]
        assert T.values[np.searchsorted(T.steps, frame_step)] == 300.0 + frame_step


def test_mdanalysis_non_periodic(tmp_path):
    pytest.importorskip("MDAnalysis")
    path = tmp_path / "md.h5md"
    write_run(path, periodic=False)
    with open_md(path) as md:
        u = md.universe()
        assert u.dimensions is None


class _FakeConfiguration:
    """Just what MDData.universe uses of a molsystem configuration."""

    class atoms:
        ids = [11, 12, 13, 14]
        symbols = ["O", "H", "H", "C"]

    class bonds:
        @staticmethod
        def get_as_dict():
            return {"i": [11, 11], "j": [12, 13]}

    @staticmethod
    def find_molecules(as_indices=True):
        return [[0, 1, 2], [3]]


def test_mdanalysis_topology_from_configuration(tmp_path):
    pytest.importorskip("MDAnalysis")
    path = tmp_path / "md.h5md"
    write_run(path)
    with open_md(path) as md:
        u = md.universe(configuration=_FakeConfiguration())
        assert len(u.residues) == 2
        assert list(u.residues[0].atoms.indices) == [0, 1, 2]
        assert len(u.bonds) == 2
        assert list(u.atoms.names) == ["O", "H", "H", "C"]
