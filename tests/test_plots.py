# -*- coding: utf-8 -*-

"""Tests of the plot helpers, with a stand-in for seamm_util's Figure."""

import numpy as np

from seamm_md_analysis import plots


class Axis(dict):
    def __init__(self, direction, label=None, anchor=None):
        super().__init__(label=label)
        self.direction = direction
        self.anchor = anchor


class Plot:
    def __init__(self, name):
        self.name = name
        self.traces = []
        self.axes = []

    def add_axis(self, direction, **kwargs):
        axis = Axis(direction, **kwargs)
        self.axes.append(axis)
        return axis

    def add_trace(self, **kwargs):
        self.traces.append(kwargs)


class Figure:
    def __init__(self):
        self.plots = []

    def add_plot(self, name):
        plot = Plot(name)
        self.plots.append(plot)
        return plot


def _fit(n, **extra):
    return {"xs": [0.0, 1.0], "ys": [0.0, 1.0], **extra}


def test_tensor_plots():
    ts = [0.0, 0.1, 0.2]
    data = np.arange(18.0).reshape(6, 3)
    err = np.full((6, 3), 0.5)
    fit = [_fit(2, kappa=1.0, stderr=0.1) for _ in range(6)]
    for function, name, prefix in (
        (plots.plot_correlation_functions, "CCF", "J0Jt"),
        (plots.plot_GK_integrals, "GKI", "K"),
        (plots.plot_helfand_moments, "M", "M"),
        (plots.plot_helfand_slopes, "Slope", "K"),
    ):
        figure = Figure()
        plot = function(figure, data, ts, err=err, fit=fit)
        assert plot.name == name
        assert len(plot.traces) == 18
        names = [t["name"] for t in plot.traces[:3]]
        assert names == ["fitxx", "±xx", f"{prefix}xx"]
        band = plot.traces[1]
        assert band["x"] == ts + ts[::-1]
        assert band["y"] == [0.5, 1.5, 2.5, 1.5, 0.5, -0.5]
        assert band["visible"] == "legendonly"
        assert plot.traces[2]["y"] == [0.0, 1.0, 2.0]
        assert "1.000 ± 0.100 W/m/K" in plot.traces[0]["hovertemplate"]


def test_tensor_plot_partial_fit_and_range():
    figure = Figure()
    plot = plots.plot_GK_integrals(
        figure,
        np.zeros((6, 2)),
        [0.0, 1.0],
        fit=[_fit(2, kappa=1, stderr=0)],
        _range=[0, 5],
    )
    assert len(plot.traces) == 7  # one fit, six data traces
    assert plot.axes[0]["range"] == [0, 5]


def test_species_traces():
    figure = Figure()
    plot = figure.add_plot("MSD")
    msd = np.arange(12.0).reshape(3, 4)  # 3 times, x y z total
    fit = [_fit(2, D_s="1.5", scale=1e-9) for _ in range(4)]
    plots.add_msd_trace(plot, "x", "y", "H2O", msd, [0.0, 1.0, 2.0], err=msd, fit=fit)
    assert len(plot.traces) == 12
    assert [t["name"] for t in plot.traces[-3:]] == [
        "H2O fit",
        "H2O ±",
        "H2O MSD",
    ]
    assert plot.traces[2]["y"] == [0.0, 4.0, 8.0]
    assert plot.traces[2]["yunits"] == "Å^2"
    assert plot.traces[0]["hovertemplate"] == "H2O Dx = 1.5 * 1.0e-09 m^2/s"

    plot = figure.add_plot("M")
    plots.add_helfand_trace(plot, "x", "y", "Li+", msd[:, :3], [0.0, 1.0, 2.0])
    assert [t["name"] for t in plot.traces] == ["Li+ Mx", "Li+ My", "Li+ Mz"]


def test_plot_components_any_units():
    figure = Figure()
    rows = np.ones((5, 4))
    fit = [{"xs": [0.1], "ys": [1.0], "value": 0.9, "stderr": None}] * 5
    plot = plots.plot_components(
        figure,
        "GKI",
        rows,
        [0.0, 0.1, 0.2, 0.3],
        "η",
        "η (mPa*s)",
        "mPa*s",
        err=rows * 0.1,
        fit=fit,
        symbol="η",
    )
    assert plot.name == "GKI"
    # fit, error band and data for each of the five components
    assert len(plot.traces) == 15
    hover = [t["hovertemplate"] for t in plot.traces if t.get("hovertemplate")]
    assert hover[0] == "ηxy = 0.9 mPa*s"
