# -*- coding: utf-8 -*-

"""Correlation functions of time series, computed with FFTs."""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import correlate

#: The order of the components of a symmetric tensor built from a vector.
TENSOR_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def ccovf(x, y, adjusted=True, demean=True):
    """The cross-covariance function of two series.

    The same definition as ``statsmodels.tsa.stattools.ccovf``:

    .. math::

        c(k) = \\frac{1}{n - k} \\sum_t (x_{t+k} - \\bar{x}) (y_t - \\bar{y})

    Parameters
    ----------
    x, y : numpy.ndarray(n)
        The two series, of the same length.
    adjusted : bool = True
        Divide each lag by the number of terms, n - k, rather than by n.
    demean : bool = True
        Subtract the means first.

    Returns
    -------
    numpy.ndarray(n)
        The cross-covariance for lags 0 .. n-1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if len(y) != n:
        raise ValueError(f"The series have different lengths: {n} and {len(y)}")
    if demean:
        x = x - x.mean()
        y = y - y.mean()
    d = np.arange(n, 0, -1) if adjusted else n
    return correlate(x, y, "full", method="fft")[n - 1 :] / d


def acf_err(acf):
    """The standard error of an autocorrelation function (Bartlett's formula).

    The same as the estimate in ``statsmodels.tsa.stattools.acf``.

    Parameters
    ----------
    acf : numpy.ndarray(n)
        The autocorrelation function.

    Returns
    -------
    numpy.ndarray(n)
        The standard error at each lag.
    """
    nobs = acf.shape[0]
    varacf = np.ones_like(acf) / nobs
    varacf[0] = 0
    varacf[1] = 1.0 / nobs
    varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
    return np.sqrt(varacf)


def ccf_err(acf1, acf2):
    """The standard error of a cross-correlation function.

    Parameters
    ----------
    acf1, acf2 : numpy.ndarray(n)
        The autocorrelation functions of the two series.

    Returns
    -------
    numpy.ndarray(n)
        The standard error at each lag.
    """
    nobs = acf1.shape[0]
    varccf = np.ones_like(acf1) / nobs
    varccf[0] = 0
    varccf[1] = 1.0 / nobs
    varccf[2:] *= 1 + 2 * np.cumsum(acf1[1:-1] * acf2[1:-1])
    return np.sqrt(varccf)


def tensor_correlation_functions(J, m=None):
    """The correlation functions of a vector series and their running integrals.

    This is the Green–Kubo input for a vector flux such as the heat flux: the
    six distinct components <J_a(t) J_b(0)> of the correlation tensor, in the
    order of :data:`TENSOR_COMPONENTS`, and their cumulative trapezoidal
    integrals in units of the sampling interval.

    Parameters
    ----------
    J : numpy.ndarray(3, n)
        The x, y and z components of the series.
    m : int
        The number of lags to keep; all of them by default.

    Returns
    -------
    numpy.ndarray(6, m)
        The correlation functions.
    numpy.ndarray(6, m)
        Their running integrals.
    """
    if m is None:
        m = J.shape[1]
    ccf = np.zeros((6, m))
    integral = np.zeros((6, m))
    for i, (a, b) in enumerate(_PAIRS):
        ccf[i] = ccovf(J[a], J[b])[:m]
        integral[i] = cumulative_trapezoid(ccf[i], initial=0.0)
    return ccf, integral


def autocorrelation_functions(X, m=None, demean=True):
    """The autocorrelation function of each series and its running integral.

    This is the Green–Kubo input for independent series, such as the five
    components of the shear stress used for the viscosity
    (:func:`seamm_md_analysis.transport.shear_stress_components`).

    Parameters
    ----------
    X : numpy.ndarray(k, n)
        The k series, each of n samples.
    m : int
        The number of lags to keep; all of them by default.
    demean : bool = True
        Subtract the mean of each series first. When the true mean is known to be
        zero, as for the shear stress of a fluid, do not: the sample mean of a
        series of length T lowers every lag by about 2 sigma^2 tau / T, so the
        running integral falls linearly and the transport coefficient is
        underestimated.

    Returns
    -------
    numpy.ndarray(k, m)
        The autocorrelation functions, <X_i(t) X_i(0)>.
    numpy.ndarray(k, m)
        Their running integrals, in units of the sampling interval.
    """
    X = np.asarray(X, dtype=float)
    if m is None:
        m = X.shape[1]
    acf = np.zeros((X.shape[0], m))
    integral = np.zeros((X.shape[0], m))
    for i, x in enumerate(X):
        acf[i] = ccovf(x, x, demean=demean)[:m]
        integral[i] = cumulative_trapezoid(acf[i], initial=0.0)
    return acf, integral


def moving_average(a, n):
    """The moving average of a series over a window of n points.

    Parameters
    ----------
    a : numpy.ndarray
        The series.
    n : int
        The width of the window.

    Returns
    -------
    numpy.ndarray(len(a) - n + 1)
        The averages.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
