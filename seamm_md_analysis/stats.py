# -*- coding: utf-8 -*-

"""Statistics of time series and of replicas."""

import contextlib
from dataclasses import dataclass
import io
import logging
from math import ceil, log10

import numpy as np


def fmt_err(value, err, precision=2, as_float=False, fallback_decimals=3):
    """Format a value and its error with the error to ``precision`` digits.

    Parameters
    ----------
    value : float
        The value.
    err : float
        Its error, for example twice the standard error for a 95% interval.
    precision : int = 2
        The number of significant digits of the error.
    as_float : bool = False
        Return rounded floats instead of strings.
    fallback_decimals : int = 3
        The decimals of the value when the error cannot be used (zero, negative
        or not a number). The error is then "--".

    Returns
    -------
    (str, str) or (float, float)
        The formatted value and error.
    """
    try:
        decimals = -ceil(log10(err)) + precision
    except Exception:
        e = "--"
        try:
            v = f"{value:.{fallback_decimals}f}"
            if as_float:
                v = round(float(v), fallback_decimals)
        except Exception:
            v = value
    else:
        if decimals < 0:
            decimals = 0
        if as_float:
            e = round(err, decimals)
        else:
            fmt = f".{decimals}f"
            e = f"{err:{fmt}}"
        try:
            if as_float:
                if abs(value / e) > 1.0e6:
                    decimals = 6
                v = round(float(value), decimals)
                e = round(float(e), decimals)
            else:
                v = f"{value:{fmt}}"
        except Exception:
            v = value
    return v, e


def mean_and_stderr(values, axis=0):
    """The mean of replicas and the standard error of that mean.

    Parameters
    ----------
    values : array-like
        The replica values along ``axis``.
    axis : int = 0
        The axis that runs over the replicas.

    Returns
    -------
    numpy.ndarray or float
        The mean.
    numpy.ndarray or float
        The standard error of the mean (sample standard deviation over sqrt(n));
        NaN for a single replica.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = np.mean(values, axis=axis)
    if n < 2:
        return mean, np.full_like(mean, np.nan)
    return mean, np.std(values, axis=axis, ddof=1) / np.sqrt(n)


def block_average(series, n_blocks=10):
    """The mean of a series and its standard error from block averages.

    The series is cut into ``n_blocks`` equal, consecutive blocks (dropping the
    remainder at the start). The blocks should be longer than the correlation
    time of the series for the error to be reliable.

    Parameters
    ----------
    series : numpy.ndarray(n) or numpy.ndarray(n, ...)
        The series, with time along the first axis.
    n_blocks : int = 10
        The number of blocks.

    Returns
    -------
    mean, stderr
        The mean over the blocks used and the standard error of the mean.
    numpy.ndarray(n_blocks, ...)
        The block means.
    """
    series = np.asarray(series, dtype=float)
    n = series.shape[0]
    if n_blocks < 2:
        raise ValueError("Need at least two blocks.")
    length = n // n_blocks
    if length < 1:
        raise ValueError(f"A series of {n} points cannot make {n_blocks} blocks.")
    start = n - length * n_blocks
    blocks = series[start:].reshape((n_blocks, length) + series.shape[1:])
    means = blocks.mean(axis=1)
    mean, stderr = mean_and_stderr(means)
    return mean, stderr, means


@contextlib.contextmanager
def _quiet():
    """Silence pymbar's banner and warnings while importing and running it."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def detect_equilibration(series, nskip=1):
    """Find where a series has equilibrated, with pymbar.

    Parameters
    ----------
    series : numpy.ndarray(n)
        The series.
    nskip : int = 1
        Test every ``nskip``-th point as the start of the equilibrated region.

    Returns
    -------
    t0 : int
        The index where the equilibrated region starts.
    g : float
        The statistical inefficiency of the equilibrated region.
    n_eff : float
        The effective number of uncorrelated samples in it.
    """
    with _quiet():
        from pymbar import timeseries

        t0, g, n_eff = timeseries.detect_equilibration(
            np.asarray(series, dtype=float), nskip=nskip
        )
    return int(t0), float(g), float(n_eff)


def statistical_inefficiency(series):
    """The statistical inefficiency g of a series, with pymbar.

    The number of uncorrelated samples is about n / g.

    Parameters
    ----------
    series : numpy.ndarray(n)
        The series.

    Returns
    -------
    float
        The statistical inefficiency, at least 1.
    """
    with _quiet():
        from pymbar import timeseries

        g = timeseries.statistical_inefficiency(np.asarray(series, dtype=float))
    return float(g)


def mean_and_stderr_correlated(series):
    """The mean of a correlated series and its standard error.

    The error is the naive standard error inflated by the statistical
    inefficiency, sqrt(g * var / n).

    Parameters
    ----------
    series : numpy.ndarray(n)
        The series.

    Returns
    -------
    float, float
        The mean and its standard error.
    """
    series = np.asarray(series, dtype=float)
    g = statistical_inefficiency(series)
    n = series.shape[0]
    return float(series.mean()), float(np.sqrt(g * series.var(ddof=1) / n))


# ----------------------------------------------------------------------
# Estimates and their errors, as the analysis steps report them
# ----------------------------------------------------------------------
def t_factor(n, level=0.95):
    """The factor turning a standard error from ``n`` samples into a confidence
    interval: Student's t with n - 1 degrees of freedom (2.78 for 5 samples at 95%,
    1.96 for many). NaN for fewer than two samples."""
    if n is None or n < 2:
        return float("nan")
    from scipy import stats

    return float(stats.t.ppf(0.5 + level / 2, n - 1))


@dataclass
class Estimate:
    """A value with its standard error, from ``n`` independent samples.

    The samples are independent runs or blocks (``source``). Results stored for
    later steps carry the standard error; tables show the 95% confidence
    interval, from Student's t, and say so.

    Attributes
    ----------
    value : float
    stderr : float or None
        The standard error of the value; None if it cannot be estimated (one
        sample).
    n : int
        The number of independent samples.
    units : str
    source : str
        What the samples are: "runs" or "blocks".
    """

    value: float
    stderr: float = None
    n: int = 1
    units: str = ""
    source: str = "runs"

    @classmethod
    def from_samples(cls, values, units="", source="runs"):
        """The mean of independent samples and its standard error."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return cls(float("nan"), None, 0, units, source)
        mean, stderr = mean_and_stderr(values)
        stderr = float(stderr) if values.size > 1 else None
        return cls(float(mean), stderr, int(values.size), units, source)

    @property
    def ci95(self):
        """The half-width of the 95% confidence interval, or None."""
        if self.stderr is None or self.n < 2:
            return None
        return t_factor(self.n) * self.stderr

    def formatted(self, precision=2):
        """The value and its 95% interval as text, e.g. ("0.72", "0.19").

        A value without an error, or the same in every sample (e.g. the volume of
        runs at constant volume), is given to six significant figures, without ±.
        """
        if self.ci95 is None or not self.ci95 > 0:
            return f"{self.value:.6g}", ""
        return fmt_err(self.value, self.ci95, precision=precision)

    def describe_error(self):
        """What the ± in a table means, for a note under it."""
        if self.stderr is None or self.n < 2:
            return f"There is no error estimate: it needs at least two {self.source}."
        t = t_factor(self.n)
        return (
            f"± is the 95% confidence interval: Student's t ({t:.2f} for {self.n} "
            f"{self.source}) times the standard error over the {self.source}."
        )


def isotropic_average(per_sample):
    """One value for an isotropic property from its components.

    The components are averaged within each run (or block), and the error is that
    of these averages over the runs: the components are correlated (e.g. the
    normal-stress differences of the viscosity), so their own spread would
    overstate the precision.

    Parameters
    ----------
    per_sample : array_like(n_samples, n_components)
        The value of each component in each run or block.

    Returns
    -------
    Estimate
    """
    per_sample = np.asarray(per_sample, dtype=float)
    return Estimate.from_samples(per_sample.mean(axis=1))


@dataclass
class ComponentCheck:
    """Whether the components of a property agree within their errors.

    Attributes
    ----------
    chi2 : float or None
        χ² of the component means about their average, weighted by their
        standard errors; None without errors.
    dof : int
        The degrees of freedom, the number of components less one.
    p : float or None
        The probability of a χ² at least this large if the components agree.
    agree : bool or None
        Whether they agree, at p ≥ 0.01.
    text : str
        One line for the output.
    """

    chi2: float = None
    dof: int = 0
    p: float = None
    agree: bool = None
    text: str = ""


def component_check(per_sample, names=None):
    """Compare the components of a property, for a one-line diagnostic.

    Parameters
    ----------
    per_sample : array_like(n_samples, n_components)
        The value of each component in each run or block.
    names : [str]
        The components' names, for the text.

    Returns
    -------
    ComponentCheck
    """
    per_sample = np.asarray(per_sample, dtype=float)
    n, k = per_sample.shape
    if n < 2 or k < 2:
        return ComponentCheck(text="The components cannot be compared: too few runs.")
    means, stderrs = mean_and_stderr(per_sample)
    if not np.all(np.isfinite(stderrs)) or np.any(stderrs <= 0):
        return ComponentCheck(text="The components cannot be compared: no errors.")
    average = np.average(means, weights=1 / stderrs**2)
    chi2 = float(np.sum(((means - average) / stderrs) ** 2))
    dof = k - 1
    from scipy import stats

    p = float(stats.chi2.sf(chi2, dof))
    agree = p >= 0.01
    what = "the components agree" if agree else "the components differ"
    text = f"Isotropy: {what} (χ²/dof = {chi2 / dof:.2f}, p = {p:.2g})."
    if not agree:
        worst = int(np.argmax(np.abs(means - average) / stderrs))
        name = names[worst] if names is not None else f"component {worst + 1}"
        text += (
            f" {name} is furthest from the average. The system may not be isotropic, "
            "or the runs too short."
        )
    return ComponentCheck(chi2, dof, p, agree, text)
