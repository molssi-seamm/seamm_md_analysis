# -*- coding: utf-8 -*-

"""Plots of correlation functions, Green–Kubo integrals, Helfand moments and
mean square displacements, for seamm_util Figures (written as .graph files)."""

import numpy as np

#: Labels and colors (line, error band) of the six tensor components.
tensor_labels = [
    ("xx", "red", "rgba(255,0,0,0.1)"),
    ("yy", "green", "rgba(0,255,0,0.1)"),
    ("zz", "blue", "rgba(0,0,255,0.1)"),
    ("xy", "rgb(127,127,0)", "rgba(127,127,0,0.1)"),
    ("xz", "rgb(255,0,255)", "rgba(255,0,255,0.1)"),
    ("yz", "rgb(0,255,255)", "rgba(0,255,255,0.1)"),
]

#: Labels and colors of the x, y and z components and their total.
vector_labels = [
    ("x", "red", "rgba(255,0,0,0.1)"),
    ("y", "green", "rgba(0,255,0,0.1)"),
    ("z", "blue", "rgba(0,0,255,0.1)"),
    ("", "black", "rgba(0,0,0,0.1)"),
]


def add_component_traces(
    plot,
    x_axis,
    y_axis,
    rows,
    ts,
    name,
    yunits,
    err=None,
    fit=None,
    hover=None,
    labels=tensor_labels,
    prefix="",
    fit_yunits=None,
):
    """Add the traces for each component of a curve: fit, error band and data.

    Parameters
    ----------
    plot : seamm_util.Plot
        The plot to add the traces to.
    x_axis, y_axis
        The axes of the plot.
    rows : numpy.ndarray(n_components, m)
        The curves, one row per component.
    ts : [float]
        The times, in ps, as a list.
    name : str
        The name of the data traces, followed by the component label.
    yunits : str
        The units of the data.
    err : numpy.ndarray(n_components, m)
        The errors, drawn as a band hidden by default (optional).
    fit : [{str: any}]
        For each component, a dict with the fitted curve "xs" and "ys" (optional).
    hover : callable(label, fit) -> str
        The hover text of a fit trace.
    labels : [(str, str, str)]
        The label, line color and band color of each component.
    prefix : str
        Put in front of each trace name, for example the species.
    fit_yunits : str
        The units of the fit traces, if different from ``yunits``.
    """
    if fit_yunits is None:
        fit_yunits = yunits
    for i, (label, color, colora) in enumerate(labels[: len(rows)]):
        if fit is not None and i < len(fit):
            plot.add_trace(
                x_axis=x_axis,
                y_axis=y_axis,
                name=f"{prefix}fit{label}",
                hovertemplate=hover(label, fit[i]) if hover is not None else None,
                x=fit[i]["xs"],
                xlabel="t",
                xunits="ps",
                y=fit[i]["ys"],
                ylabel=f"{prefix}fit{label}",
                yunits=fit_yunits,
                color=color,
                dash="dash",
                width=3,
            )
        if err is not None:
            errs = np.concatenate((rows[i] + err[i], rows[i][::-1] - err[i][::-1]))
            plot.add_trace(
                x_axis=x_axis,
                y_axis=y_axis,
                name=f"{prefix}±{label}",
                x=ts + ts[::-1],
                xlabel="t",
                xunits="ps",
                y=errs.tolist(),
                ylabel=f"{prefix}±{label}",
                yunits=yunits,
                color=colora,
                fill="toself",
                visible="legendonly",
            )
        plot.add_trace(
            x_axis=x_axis,
            y_axis=y_axis,
            name=f"{prefix}{name}{label}",
            x=ts,
            xlabel="t",
            xunits="ps",
            y=rows[i].tolist(),
            ylabel=f"{prefix}{name}{label}",
            yunits=yunits,
            color=color,
        )


def _new_plot(figure, name, xlabel, ylabel, _range=None):
    plot = figure.add_plot(name)
    x_axis = plot.add_axis("x", label=xlabel)
    y_axis = plot.add_axis("y", label=ylabel, anchor=x_axis)
    x_axis.anchor = y_axis
    if _range is not None:
        x_axis["range"] = _range
    return plot, x_axis, y_axis


def _kappa_hover(prefix=""):
    def hover(label, fit):
        return f"{prefix}{label} = {fit['kappa']:.3f} ± {fit['stderr']:.3f} W/m/K"

    return hover


def plot_correlation_functions(
    figure, CCF, ts, err=None, fit=None, labels=tensor_labels
):
    """Plot the heat-flux correlation functions.

    Parameters
    ----------
    figure : seamm_util.Figure
        The figure that contains the plots.
    CCF : numpy.ndarray(6, m)
        The correlation functions, in W^2/m^4.
    ts : [float]
        The times, in ps.
    err : numpy.ndarray(6, m)
        Their standard errors (optional).
    fit : [{str: any}]
        Fits to the curves (optional).
    """
    plot, x_axis, y_axis = _new_plot(figure, "CCF", "t (ps)", "J0Jt (W^2/m^4)")
    add_component_traces(
        plot,
        x_axis,
        y_axis,
        CCF,
        ts,
        "J0Jt",
        "W^2/m^4",
        err=err,
        fit=fit,
        hover=_kappa_hover(),
        labels=labels,
        fit_yunits="W/m/K*ps",
    )
    return plot


def plot_GK_integrals(
    figure, x, ts, err=None, fit=None, labels=tensor_labels, _range=None
):
    """Plot the Green–Kubo running integrals of the thermal conductivity.

    Parameters
    ----------
    figure : seamm_util.Figure
        The figure that contains the plots.
    x : numpy.ndarray(6, m)
        The running integrals, in W/m/K.
    ts : [float]
        The times, in ps.
    err : numpy.ndarray(6, m)
        Their standard errors (optional).
    fit : [{str: any}]
        Fits to the curves (optional).
    _range : [float]
        The range of the time axis.
    """
    plot, x_axis, y_axis = _new_plot(
        figure, "GKI", "t (ps)", "Kappa (W/m/K)", _range=_range
    )
    add_component_traces(
        plot,
        x_axis,
        y_axis,
        x,
        ts,
        "K",
        "W/m/K",
        err=err,
        fit=fit,
        hover=_kappa_hover(),
        labels=labels,
    )
    return plot


def plot_helfand_moments(figure, M, ts, err=None, fit=None, labels=tensor_labels):
    """Plot the Helfand moments of the thermal conductivity.

    Parameters
    ----------
    figure : seamm_util.Figure
        The figure that contains the plots.
    M : numpy.ndarray(6, m)
        The Helfand moments, in W/m/K*ps.
    ts : [float]
        The times, in ps.
    err : numpy.ndarray(6, m)
        Their standard errors (optional).
    fit : [{str: any}]
        Fits to the curves (optional).
    """
    plot, x_axis, y_axis = _new_plot(figure, "M", "Time (ps)", "M (W/m/K*ps)")
    add_component_traces(
        plot,
        x_axis,
        y_axis,
        M,
        ts,
        "M",
        "W/m/K*ps",
        err=err,
        fit=fit,
        hover=_kappa_hover("κ"),
        labels=labels,
    )
    return plot


def plot_helfand_slopes(
    figure, x, ts, err=None, fit=None, labels=tensor_labels, _range=None
):
    """Plot the local slopes of the Helfand moments of the thermal conductivity.

    Parameters
    ----------
    figure : seamm_util.Figure
        The figure that contains the plots.
    x : numpy.ndarray(6, m)
        The slopes, in W/m/K.
    ts : [float]
        The times, in ps.
    err : numpy.ndarray(6, m)
        Their standard errors (optional).
    fit : [{str: any}]
        Fits to the curves (optional).
    _range : [float]
        The range of the time axis.
    """
    plot, x_axis, y_axis = _new_plot(
        figure, "Slope", "t (ps)", "Kappa (W/m/K)", _range=_range
    )
    add_component_traces(
        plot,
        x_axis,
        y_axis,
        x,
        ts,
        "K",
        "W/m/K",
        err=err,
        fit=fit,
        hover=_kappa_hover(),
        labels=labels,
    )
    return plot


def _diffusion_hover(species):
    def hover(label, fit):
        return f"{species} D{label} = {fit['D_s']} * {fit['scale']:.1e} m^2/s"

    return hover


def add_msd_trace(
    plot, x_axis, y_axis, species, msd, ts, err=None, fit=None, labels=vector_labels
):
    """Add the mean square displacement of one species to a plot.

    Parameters
    ----------
    plot : seamm_util.Plot
        The plot to add the traces to.
    x_axis, y_axis
        The axes of the plot.
    species : str
        The label of the species.
    msd : numpy.ndarray(n, 3 or 4)
        The MSD in x, y and z, and optionally the total, in Å^2.
    ts : [float]
        The times, in ps.
    err : numpy.ndarray(n, 3 or 4)
        The errors of the MSD (optional).
    fit : [{str: any}]
        The fits of the curves, with "D_s", "scale", "xs" and "ys" (optional).
    """
    add_component_traces(
        plot,
        x_axis,
        y_axis,
        msd.T,
        ts,
        "MSD",
        "Å^2",
        err=None if err is None else err.T,
        fit=fit,
        hover=_diffusion_hover(species),
        labels=labels,
        prefix=f"{species} ",
    )


def add_helfand_trace(
    plot, x_axis, y_axis, species, M, ts, err=None, fit=None, labels=vector_labels
):
    """Add the single-molecule Helfand moments of one species to a plot.

    Parameters
    ----------
    plot : seamm_util.Plot
        The plot to add the traces to.
    x_axis, y_axis
        The axes of the plot.
    species : str
        The label of the species.
    M : numpy.ndarray(m, 3 or 4)
        The Helfand moments, in Å^2.
    ts : [float]
        The times, in ps.
    err : numpy.ndarray(m, 3 or 4)
        The errors of the moments (optional).
    fit : [{str: any}]
        The fits of the curves, with "D_s", "scale", "xs" and "ys" (optional).
    """
    add_component_traces(
        plot,
        x_axis,
        y_axis,
        M.T,
        ts,
        "M",
        "Å^2",
        err=None if err is None else err.T,
        fit=fit,
        hover=_diffusion_hover(species),
        labels=labels,
        prefix=f"{species} ",
    )
