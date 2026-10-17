#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests of the requirements of analyses and what MD runs provide."""

import pytest

from seamm_md_analysis import Interval, Need, Provides, Requirements, select_md_runs
from seamm_md_analysis.requirements import check


def test_nothing_missing():
    req = Requirements(needs=[Need("heat_flux", max_steps=2)], periodic=True)
    prov = Provides(data={"heat_flux": Interval(steps=2)})
    assert check(req, prov) == []


def test_missing_is_reported_with_the_reason():
    req = Requirements(needs=[Need("heat_flux", why="the thermal conductivity")])
    prov = Provides(data={"temperature": Interval("10 fs")}, step="LAMMPS / NVE")
    (problem,) = check(req, prov)
    assert "heat_flux is not sampled in LAMMPS / NVE" in problem
    assert "the thermal conductivity" in problem


def test_alternatives_in_order_of_preference():
    need = Need(["molecules/position", "all/position"], max_time="1 ps")
    prov = Provides(data={"all/position": Interval("500 fs")})
    assert check(Requirements(needs=[need]), prov) == []
    prov = Provides(data={"all/position": Interval("2 ps")})
    (problem,) = check(Requirements(needs=[need]), prov)
    assert "all/position is sampled every 2 ps" in problem


def test_too_coarse_in_steps():
    req = Requirements(needs=[Need("heat_flux", max_steps=2)])
    (problem,) = check(req, Provides(data={"heat_flux": Interval(steps=20)}))
    assert "every 20 steps" in problem


def test_interval_converted_with_the_timestep():
    req = Requirements(needs=[Need("heat_flux", max_steps=2)])
    prov = Provides(data={"heat_flux": Interval("20 fs")}, timestep="1 fs")
    assert len(check(req, prov)) == 1
    req = Requirements(needs=[Need("all/velocity", max_time="10 fs")])
    prov = Provides(data={"all/velocity": Interval(steps=5)}, timestep="1 fs")
    assert check(req, prov) == []
    prov = Provides(data={"all/velocity": Interval(steps=20)}, timestep="1 fs")
    assert len(check(req, prov)) == 1


def test_unknown_interval_is_not_reported():
    req = Requirements(needs=[Need("heat_flux", max_time="4 fs")])
    prov = Provides(data={"heat_flux": Interval(steps=2)})
    assert check(req, prov) == []


def test_interval_text():
    assert str(Interval("50 fs")) == "50 fs"
    assert str(Interval(steps=2)) == "2 steps"
    assert str(Interval()) == "unknown"


records = [
    {"step": "3.3 NVT", "observables": {"temperature": "10 fs"}, "path": "a"},
    {"step": "3.4 NVE", "observables": {"heat_flux": "2 fs"}, "path": "b"},
    {"step": "3.3 NVT", "observables": {"temperature": "10 fs"}, "path": "c"},
    {"step": "3.4 NVE", "observables": {"heat_flux": "2 fs"}, "path": "d"},
]


def test_select_last_run():
    assert [r["path"] for r in select_md_runs(records)] == ["d"]
    assert select_md_runs([]) == []
    assert select_md_runs(None) == []


def test_select_by_name():
    chosen = select_md_runs(records, name="temperature")
    assert [r["path"] for r in chosen] == ["c"]
    assert select_md_runs(records, name="molecules/position") == []


def test_select_loop_iterations():
    chosen = select_md_runs(records, "the last MD run of each loop iteration")
    assert [r["path"] for r in chosen] == ["b", "d"]


def test_select_runs_in_loop_iterations():
    # The step's id includes each loop iteration's directory (job 4021, 2026-10-16)
    loop = []
    for i in (1, 2, 3):
        loop.append({"step": f"4.iter_{i}.0.3 NVT dynamics", "path": f"e{i}"})
        loop.append({"step": f"4.iter_{i}.0.4 NVT dynamics", "path": f"p{i}"})
    chosen = select_md_runs(loop, "the last MD run of each loop iteration")
    assert [r["path"] for r in chosen] == ["p1", "p2", "p3"]
    # Iterations named by system, and a run outside the loop by the same step title
    named = [
        {"step": "2 NVT dynamics", "path": "x"},
        {"step": "4.water.0.4 NVT dynamics", "path": "w"},
        {"step": "4.NaCl.0.4 NVT dynamics", "path": "n"},
    ]
    chosen = select_md_runs(named, "the last MD run of each loop iteration")
    assert [r["path"] for r in chosen] == ["w", "n"]


def test_select_unknown_source():
    with pytest.raises(ValueError):
        select_md_runs(records, "everything")
