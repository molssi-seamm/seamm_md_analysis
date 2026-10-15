# -*- coding: utf-8 -*-

"""Statistics of time series and of replicas."""

import contextlib
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
