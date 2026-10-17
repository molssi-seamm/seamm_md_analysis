#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Errors as the steps report them, isotropic values, and the state of the runs."""

import numpy as np
import pytest

from seamm_md_analysis import (
    Estimate,
    H5MDWriter,
    bulk_modulus,
    component_check,
    is_cubic,
    isotropic_average,
    isotropy,
    open_md,
    run_state,
    state_table,
    summarize_state,
    t_factor,
)
from seamm_md_analysis.state import AMU_PER_A3_IN_G_PER_ML, KB_ATM_A3

L = 25.0
CUBIC = np.eye(3) * L
N_WATER = 512
MASS = 18.015


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
def test_t_factor():
    assert t_factor(5) == pytest.approx(2.776, abs=1e-3)
    assert t_factor(1000) == pytest.approx(1.962, abs=1e-3)
    assert np.isnan(t_factor(1))


def test_estimate_from_samples():
    values = [0.70, 0.75, 0.68, 0.74, 0.71]
    e = Estimate.from_samples(values, units="mPa*s")
    assert e.value == pytest.approx(np.mean(values))
    assert e.stderr == pytest.approx(np.std(values, ddof=1) / np.sqrt(5))
    assert e.ci95 == pytest.approx(2.776 * e.stderr, rel=1e-3)
    assert e.n == 5 and e.units == "mPa*s"
    assert "Student's t (2.78 for 5 runs)" in e.describe_error()
    one = Estimate.from_samples([0.7])
    assert one.stderr is None and one.ci95 is None
    assert one.formatted()[1] == ""
    assert "at least two runs" in one.describe_error()


def test_isotropic_average_is_the_error_of_the_run_averages():
    rng = np.random.default_rng(3)
    common = rng.normal(0.7, 0.05, size=8)  # each run's value
    per_run = common[:, None] + rng.normal(0, 0.1, size=(8, 3))
    e = isotropic_average(per_run)
    assert e.value == pytest.approx(per_run.mean())
    assert e.stderr == pytest.approx(np.std(per_run.mean(axis=1), ddof=1) / np.sqrt(8))


def test_component_check():
    rng = np.random.default_rng(4)
    agree = 1.0 + rng.normal(0, 0.05, size=(10, 3))
    check = component_check(agree, names=["xx", "yy", "zz"])
    assert check.agree and check.dof == 2
    assert check.text.startswith("Isotropy: the components agree")
    differ = agree.copy()
    differ[:, 2] += 0.5
    check = component_check(differ, names=["xx", "yy", "zz"])
    assert not check.agree
    assert "zz is furthest from the average" in check.text
    assert "cannot be compared" in component_check(agree[:1]).text


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
def write_run(
    path,
    ensemble="NVT",
    cell=CUBIC,
    fixes=None,
    pressure=-45.0,
    volume=None,
    n=400,
    seed=1,
    target=1.0,
):
    """A run of 512 waters: temperature and pressure, and the volume for NPT."""
    rng = np.random.default_rng(seed)
    steps = np.arange(n) * 10
    times = steps * 2.0
    with H5MDWriter(path) as w:
        w.particles(
            ids=np.arange(1, N_WATER + 1),
            species=[8] * N_WATER,
            masses=[MASS] * N_WATER,
            cell=cell,
            periodic=True,
        )
        w.observable("temperature", steps, times, rng.normal(298.15, 3, n), "K")
        w.observable("pressure", steps, times, rng.normal(pressure, 300, n), "atm")
        if volume is not None:
            w.observable("volume", steps, times, volume, "Angstrom3")
        parameters = {"ensemble": ensemble}
        if fixes is not None:
            parameters["fixes"] = fixes
        if ensemble == "NPT":
            parameters["pressure"] = target
            parameters["pressure_unit"] = "atm"
        w.parameters(parameters)
    return path


def test_is_cubic():
    assert is_cubic(CUBIC)
    assert not is_cubic(np.diag([25.0, 25.0, 30.0]))
    tilted = CUBIC.copy()
    tilted[1, 0] = 2.0
    assert not is_cubic(tilted)
    assert not is_cubic(None)


def test_isotropy(tmp_path):
    with open_md(write_run(tmp_path / "a.h5md", fixes=["nvt", "shake"])) as md:
        assert isotropy(md) == (
            True,
            "the cell is cubic and nothing in the run is directional",
        )
    with open_md(write_run(tmp_path / "b.h5md", fixes=["nvt", "wall/lj93"])) as md:
        assert isotropy(md) == (False, "the run used wall/lj93")
    with open_md(write_run(tmp_path / "c.h5md")) as md:
        ok, why = isotropy(md)
        assert ok and "did not record its fixes" in why
    with open_md(write_run(tmp_path / "d.h5md", cell=np.diag([25, 25, 30.0]))) as md:
        assert isotropy(md) == (False, "the cell is not cubic")


def test_run_state_at_constant_volume(tmp_path):
    with open_md(write_run(tmp_path / "md.h5md")) as md:
        s = run_state(md)
    assert s.ensemble == "NVT"
    assert s.temperature == pytest.approx(298.15, abs=1.0)
    assert s.pressure == pytest.approx(-45.0, abs=60.0)
    assert s.volume == pytest.approx(L**3)
    expected = N_WATER * MASS / L**3 * AMU_PER_A3_IN_G_PER_ML
    assert s.density == pytest.approx(expected)
    assert s.target_pressure is None


def test_bulk_modulus_from_volume_fluctuations(tmp_path):
    rng = np.random.default_rng(7)
    V0, sigma = L**3, 120.0
    volume = rng.normal(V0, sigma, 4000)
    path = write_run(tmp_path / "npt.h5md", ensemble="NPT", volume=volume, n=4000)
    with open_md(path) as md:
        K = bulk_modulus(md)
        T = run_state(md).temperature
    assert K == pytest.approx(KB_ATM_A3 * T * V0 / sigma**2, rel=0.1)


def test_summarize_constant_volume_runs(tmp_path):
    runs = []
    for seed in range(5):
        with open_md(write_run(tmp_path / f"r{seed}.h5md", seed=seed)) as md:
            runs.append(run_state(md))

    without = summarize_state(runs)
    assert without.pressure.n == 5 and without.equilibrium_density is None
    assert "needs the bulk modulus" in without.notes[0]

    K = 22000.0  # about water's, atm
    with_K = summarize_state(runs, bulk_modulus=K)
    rho, P = with_K.density.value, with_K.pressure.value
    assert with_K.equilibrium_density.value == pytest.approx(rho * (1 + (1 - P) / K))
    assert with_K.equilibrium_density.stderr == pytest.approx(
        rho * with_K.pressure.stderr / K
    )
    text = state_table(with_K)
    for words in (
        "The state (NVT)",
        "Pressure",
        "Equilibrium density",
        "Student's t (2.78 for 5 runs)",
        "K = 2.2e+04 atm",
    ):
        assert words in text


def test_summarize_npt_runs(tmp_path):
    rng = np.random.default_rng(9)
    runs = []
    for seed in range(3):
        volume = rng.normal(L**3, 100.0, 400)
        path = write_run(
            tmp_path / f"n{seed}.h5md",
            ensemble="NPT",
            volume=volume,
            seed=seed,
            pressure=1.0,
        )
        with open_md(path) as md:
            runs.append(run_state(md))
    s = summarize_state(runs)
    assert s.ensemble == "NPT" and s.target_pressure == pytest.approx(1.0)
    assert s.equilibrium_density is None and s.notes == []
    assert s.volume.value == pytest.approx(L**3, rel=0.01)


class _Node:
    """The workspace variables of an analysis step."""

    def __init__(self, variables):
        self.variables = variables

    def variable_exists(self, name):
        return name in self.variables

    def get_variable(self, name):
        return self.variables[name]


def test_state_of_runs_uses_the_npt_run_before_them(tmp_path):
    from seamm_md_analysis.step_support import npt_reference, state_of_runs

    rng = np.random.default_rng(11)
    npt = write_run(
        tmp_path / "npt.h5md",
        ensemble="NPT",
        volume=rng.normal(L**3, 120.0, 4000),
        n=4000,
        pressure=1.0,
    )
    nvt = [write_run(tmp_path / f"nvt{i}.h5md", seed=i) for i in range(3)]
    records = [{"path": str(npt), "ensemble": "NPT"}] + [
        {"path": str(p), "ensemble": "NVT"} for p in nvt
    ]
    node = _Node({"_md_runs": records})
    K, target = npt_reference(node, nvt)
    with open_md(npt) as md:
        assert K == pytest.approx(bulk_modulus(md))
    assert target == pytest.approx(1.0)

    s = state_of_runs(node, nvt)
    assert s.bulk_modulus == pytest.approx(K) and s.target_pressure == 1.0
    assert s.equilibrium_density is not None and s.pressure.n == 3

    # Without an NPT run before them, nothing to estimate from
    alone = _Node({"_md_runs": records[1:]})
    assert npt_reference(alone, nvt) == (None, None)
    assert state_of_runs(alone, nvt).equilibrium_density is None


class _Printer:
    def __init__(self):
        self.lines = []

    def normal(self, text):
        self.lines.append(str(text))


class _Step(_Node):
    def __init__(self, variables, step_id):
        super().__init__(variables)
        self._id = step_id

    def set_variable(self, name, value):
        self.variables[name] = value


def test_the_state_is_printed_once_for_the_same_runs(tmp_path):
    from seamm_md_analysis.step_support import report_state

    runs = [write_run(tmp_path / f"r{i}.h5md", seed=i) for i in range(2)]
    variables = {}
    first, second = _Printer(), _Printer()
    s1 = report_state(_Step(variables, ("9",)), first, runs)
    s2 = report_state(_Step(variables, ("10",)), second, runs)
    assert any("The state (NVT)" in line for line in first.lines)
    assert (
        second.lines[0].strip() == "The state of these runs is given by step 9 above."
    )
    assert s1.pressure.value == s2.pressure.value
    # Other runs get their own table
    third = _Printer()
    report_state(_Step(variables, ("11",)), third, runs[:1])
    assert any("The state (NVT)" in line for line in third.lines)
