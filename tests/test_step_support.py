#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests of the support for analysis steps (step_support)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from seamm_md_analysis import Interval, Need, Provides, Requirements
from seamm_md_analysis import step_support as ss


class Edge(SimpleNamespace):
    pass


class Flowchart:
    """A chain of nodes joined by "next" edges, starting at a start node."""

    def __init__(self, *nodes):
        self.start = SimpleNamespace(title="Start")
        self.chain = [self.start, *nodes]

    def get_node(self, uuid):
        return self.start

    def edges(self, node, direction="out"):
        i = next(i for i, n in enumerate(self.chain) if n is node)
        if i + 1 < len(self.chain):
            return [
                Edge(
                    edge_type="execution", edge_subtype="next", node2=self.chain[i + 1]
                )
            ]
        return []


class MD:
    def __init__(self, title, data, timestep=None):
        self.title = title
        self.data = dict(data)
        self.timestep = timestep
        self.requests = []

    def md_provides(self):
        return Provides(data=dict(self.data), timestep=self.timestep, step=self.title)

    def request_sampling(self, need):
        if need.max_time is None:
            return None
        self.data[need.names[0]] = Interval(need.sample_time or need.max_time)
        self.requests.append(need.names[0])
        return f"turned on {need.names[0]}"


needs = Requirements(
    needs=[
        Need(["molecules/position", "all/position"], "1 ps", sample_time="100 fs"),
        Need(["molecules/velocity"], "20 fs"),
    ],
    periodic=True,
)


def test_md_substeps_inside_subflowcharts():
    nvt = MD("NVT", {})
    lammps = SimpleNamespace(title="LAMMPS", subflowchart=Flowchart(nvt))
    other = SimpleNamespace(title="Packmol")
    assert ss.md_substeps([other, lammps]) == [nvt]
    assert ss.md_substeps(Flowchart(other, lammps)) == [nvt]


def test_check_mode_b_reports_gaps():
    nvt = MD("LAMMPS / NVT", {"molecules/position": Interval("50 fs")})
    lammps = SimpleNamespace(title="LAMMPS", subflowchart=Flowchart(nvt))
    node = SimpleNamespace(title="Diffusivity")
    (problem,) = ss.check_md_flowchart(node, [lammps], needs, "the last MD run")
    assert problem.startswith("'Diffusivity': molecules/velocity is not sampled")


class EditableMD(MD):
    """An MD step that gives its sampling changes as parameter edits."""

    def __init__(self, title, data):
        super().__init__(title, data)
        self.parameters = {
            "velocities": SimpleNamespace(value="never", units=None),
            "velocities rate": SimpleNamespace(value="", units=None),
        }

    def sampling_edits(self, need):
        if need.names[0] != "molecules/velocity":
            return None
        return (
            [("velocities", "by time interval", None), ("velocities rate", "20", "fs")],
            f"sampling of molecules/velocity every 20 fs in {self.title}",
        )


def test_check_mode_b_offers_a_fix():
    pytest.importorskip("seamm.problems")
    nvt = EditableMD("LAMMPS / NVT", {"molecules/position": Interval("50 fs")})
    lammps = SimpleNamespace(title="LAMMPS", subflowchart=Flowchart(nvt))
    node = SimpleNamespace(title="Diffusivity")
    (problem,) = ss.check_md_flowchart(node, [lammps], needs, "the last MD run")
    assert problem.severity == "error"
    (fix,) = problem.fixes
    assert fix.label == (
        "Turn on sampling of molecules/velocity every 20 fs in LAMMPS / NVT"
    )
    fix.apply()
    assert nvt.parameters["velocities"].value == "by time interval"
    assert nvt.parameters["velocities rate"].value == "20"
    assert nvt.parameters["velocities rate"].units == "fs"


def test_check_mode_b_without_md():
    node = SimpleNamespace(title="Diffusivity")
    (problem,) = ss.check_md_flowchart(node, [], needs, "the last MD run")
    assert "no MD step comes before it" in problem


def test_check_mode_a_needs_an_md_step():
    node = SimpleNamespace(title="Diffusivity", subflowchart=Flowchart())
    (problem,) = ss.check_md_flowchart(node, [], needs, ss.run_own_md)
    assert "has no MD step" in problem
    node.subflowchart = Flowchart(MD("NVT", {}))
    assert ss.check_md_flowchart(node, [], needs, ss.run_own_md) == []


def test_files_are_not_checked():
    node = SimpleNamespace(title="Diffusivity")
    assert ss.check_md_flowchart(node, [], needs, ss.read_files) == []


def test_request_sampling_turns_on_what_is_missing():
    nvt = MD("NVT", {"molecules/velocity": Interval("10 fs")})
    flowchart = Flowchart(SimpleNamespace(title="Init"), nvt)
    changes, remaining = ss.request_sampling(flowchart, needs)
    assert changes == ["turned on molecules/position"]
    assert remaining == []
    assert nvt.data["molecules/position"].time.m_as("fs") == 100


def test_request_sampling_reports_what_it_cannot_do():
    nvt = MD("NVT", {})
    req = Requirements(needs=[Need("heat_flux", max_steps=2)])
    changes, remaining = ss.request_sampling(Flowchart(nvt), req)
    assert changes == []
    assert "heat_flux is not sampled" in remaining[0]


class Analysis:
    title = "Diffusivity"

    def __init__(self, records=None):
        self.records = records

    def variable_exists(self, name):
        return self.records is not None

    def get_variable(self, name):
        return self.records

    def file_path(self, path, read_only=False):
        return Path("/job") / path


def test_md_inputs_from_runs():
    records = [
        {
            "step": "3.3 NVT",
            "path": "/a/md.h5md",
            "particles": {"all/position": "1 ps"},
        },
        {"step": "3.3 NVT", "path": "/b/md.h5md", "particles": {}},
    ]
    P = {"md source": "the last MD run"}
    assert ss.md_inputs(Analysis(records), P) == [Path("/b/md.h5md")]
    assert ss.md_inputs(Analysis(records), P, name="all/position") == [
        Path("/a/md.h5md")
    ]
    P = {"md source": "the last MD run of each loop iteration"}
    assert len(ss.md_inputs(Analysis(records), P)) == 2
    with pytest.raises(RuntimeError, match="no earlier MD run"):
        ss.md_inputs(Analysis(None), {"md source": "the last MD run"})


def test_md_inputs_from_files():
    P = {"md source": "files", "md files": "a/md.h5md, job://12/b/md.h5md"}
    paths = ss.md_inputs(Analysis(), P)
    assert paths == [Path("/job/a/md.h5md"), Path("/job/job://12/b/md.h5md")]
    with pytest.raises(RuntimeError, match="no H5MD files"):
        ss.md_inputs(Analysis(), {"md source": "files", "md files": " "})


def test_source_parameters():
    params = ss.source_parameters("the diffusivity")
    assert params["md source"]["default"] == ss.run_own_md
    assert "the diffusivity" in params["md source"]["help_text"]
    assert params["md files"]["applies_when"] == {"md source": "files"}


def test_check_mode_b_uses_the_run_that_samples_the_first_need():
    nvt = MD(
        "LAMMPS / NVT",
        {
            "molecules/position": Interval("100 fs"),
            "molecules/velocity": Interval("10 fs"),
        },
    )
    heat_flux = MD("LAMMPS / Heat Flux", {"heat_flux": Interval(steps=2)})
    lammps = SimpleNamespace(title="LAMMPS", subflowchart=Flowchart(nvt, heat_flux))
    node = SimpleNamespace(title="Diffusivity")
    assert ss.check_md_flowchart(node, [lammps], needs, "the last MD run") == []


class Workspace:
    title = "Viscosity"
    _id = ("3",)

    def __init__(self):
        self.variables = {}

    def variable_exists(self, name):
        return name in self.variables

    def get_variable(self, name):
        return self.variables[name]

    def set_variable(self, name, value):
        self.variables[name] = value


def test_publish_and_read_viscosity():
    node = Workspace()
    assert ss.preceding_viscosity(node) is None
    record = ss.publish_viscosity(node, 8.9e-4, stderr=2e-5, T=298.15)
    assert record["value"] == pytest.approx(8.9e-4)
    assert record["units"] == "Pa*s"
    assert record["step"] == "Viscosity" and record["id"] == "3"
    assert ss.preceding_viscosity(node) == record
