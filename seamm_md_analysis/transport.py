# -*- coding: utf-8 -*-

"""Transport coefficients: Helfand moments, mean square displacements, and fits
of Green–Kubo integrals and Helfand moments."""

from collections import namedtuple
import logging
from math import pi
import pprint
import warnings

import numpy as np
from scipy.constants import k as k_B
from scipy.fft import irfft, next_fast_len, rfft
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import OptimizeWarning, curve_fit

from .correlate import _PAIRS, moving_average

logger = logging.getLogger(__name__)

#: A straight-line fit: the slope, its standard error, and the fitted curve.
LinearFit = namedtuple("LinearFit", "slope stderr xs ys")

#: A fit of a Green–Kubo integral by one or two rising exponentials.
GreenKuboFit = namedtuple("GreenKuboFit", "value stderr a a_err tau tau_err xs ys")

#: The Yeh–Hummer constant for a cubic cell (Yeh and Hummer, 2004).
YEH_HUMMER_XI = 2.837297

#: The five independent components of the shear stress of an isotropic fluid.
SHEAR_COMPONENTS = ("xy", "xz", "yz", "(xx-yy)/2", "(yy-zz)/2")

# Columns of the arrays handed to one batch of FFTs, to bound the memory.
_FFT_COLUMNS = 256


def axb(x, a, b):
    """A straight line, a * x + b."""
    return a * x + b


def exp_1(t, a, tau):
    """A single rising exponential, a * (1 - exp(-t / tau))."""
    return a * (1 - np.exp(-t / tau))


def exp_2(t, a1, tau1, a2, tau2):
    """The sum of two rising exponentials."""
    return a1 * (1 - np.exp(-t / tau1)) + a2 * (1 - np.exp(-t / tau2))


def _lagged_products(a, b, m, n_origins):
    """sum_{i < n_origins} a[i + k] * b[i] for k = 0 .. m-1, along axis 0.

    a and b are 2-D (time, columns); the FFTs are batched over the columns.
    """
    length = next_fast_len(n_origins + m - 1 + n_origins, real=True)
    result = np.empty((m, a.shape[1]))
    for start in range(0, a.shape[1], _FFT_COLUMNS):
        cols = slice(start, start + _FFT_COLUMNS)
        fa = rfft(a[: n_origins + m - 1, cols], length, axis=0)
        fb = rfft(b[:n_origins, cols], length, axis=0)
        result[:, cols] = irfft(fa * np.conj(fb), length, axis=0)[:m]
    return result


def _detrend(G, used):
    """Split G (time, columns) into a straight line s * i plus a centred rest."""
    n = G.shape[0]
    if used > 1:
        s = (G[used - 1] - G[0]) / (used - 1)
    else:
        s = np.zeros(G.shape[1])
    g = G - np.outer(np.arange(n), s)
    return g - g[:used].mean(axis=0), s


def _usable_sigma(sigma):
    """Whether errors can weight a fit: given, finite and all positive.

    The spread over runs is zero for a single run, which would make a weighted
    fit fail, so such errors are not used.
    """
    if sigma is None:
        return False
    sigma = np.asarray(sigma, dtype=float)
    return sigma.size > 0 and bool(np.all(np.isfinite(sigma)) and np.all(sigma > 0))


def _sigma_kwargs(sigma):
    """The curve_fit arguments for weighting by ``sigma``, if it is usable."""
    if _usable_sigma(sigma):
        return {"sigma": sigma, "absolute_sigma": True}
    return {}


def windowed_displacement_products(Ga, Gb, m, n_origins):
    """Sum over time origins of the products of displacements.

    .. math::

        S(k) = \\sum_{i=0}^{N_0 - 1} (G_a[i+k] - G_a[i]) (G_b[i+k] - G_b[i])

    for lags k = 0 .. m-1 and a fixed set of N_0 origins, computed with FFTs.
    With G the running integral of a flux, S/N_0 is the Helfand moment.

    Each series is split into a straight line and the rest before the FFTs,
    and the line's contribution is added back exactly. A flux with a non-zero
    mean has a running integral that drifts linearly, and without this the
    rounding error would grow as the square of the length of the series.

    Parameters
    ----------
    Ga, Gb : numpy.ndarray(n, ...)
        The series, with time along the first axis. They may be the same array.
    m : int
        The number of lags.
    n_origins : int
        The number of time origins, N_0; n_origins + m - 1 must not exceed n.

    Returns
    -------
    numpy.ndarray(m, ...)
        S for each lag and column.
    """
    same = Ga is Gb
    Ga = np.asarray(Ga, dtype=float)
    shape = Ga.shape
    n = shape[0]
    if n_origins + m - 1 > n:
        raise ValueError(f"{n_origins} origins and {m} lags need more than {n} points")
    used = n_origins + m - 1
    ga, sa = _detrend(Ga.reshape(n, -1), used)
    if same:
        gb, sb = ga, sa
    else:
        gb, sb = _detrend(np.asarray(Gb, dtype=float).reshape(n, -1), used)

    k = np.arange(m)

    def window_sums(x):
        # sum_{i < n_origins} x[i + k], for each lag k
        cs = np.zeros((n + 1, x.shape[1]))
        np.cumsum(x, axis=0, out=cs[1:])
        return cs[k + n_origins] - cs[k]

    # sum_i ga[i+k] gb[i+k] - ga[i+k] gb[i] - ga[i] gb[i+k] + ga[i] gb[i]
    s_same = window_sums(ga * gb)
    c_ab = _lagged_products(ga, gb, m, n_origins)
    c_ba = c_ab if same else _lagged_products(gb, ga, m, n_origins)
    S = s_same - c_ab - c_ba + s_same[0]

    # The straight lines: displacements gain s * k
    kk = k[:, np.newaxis]
    da = window_sums(ga) - ga[:n_origins].sum(axis=0)
    db = da if same else window_sums(gb) - gb[:n_origins].sum(axis=0)
    S += kk * (sa * db + sb * da) + n_origins * sa * sb * kk**2

    S[0] = 0.0  # no displacement at lag 0
    return S.reshape((m,) + shape[1:])


def collective_helfand_moments(J, m=None):
    """The Helfand moments of a collective vector flux, such as the heat flux.

    .. math::

        M_{ab}(k) = \\left\\langle \\int_0^{k} J_a \\, dt \\int_0^{k} J_b \\, dt
                    \\right\\rangle

    averaged over n - m time origins, in units of the sampling interval squared,
    for the six components in :data:`seamm_md_analysis.correlate.TENSOR_COMPONENTS`.

    Parameters
    ----------
    J : numpy.ndarray(3, n)
        The x, y and z components of the flux.
    m : int
        The number of lags; by default min(n // 20, 10000).

    Returns
    -------
    numpy.ndarray(6, m)
        The Helfand moments.
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[1]
    if m is None:
        m = min(n // 20, 10000)
    n_origins = n - m
    G = cumulative_trapezoid(J, axis=1, initial=0.0)
    M = np.zeros((6, m))
    for i, (a, b) in enumerate(_PAIRS):
        Ga = G[a]
        Gb = Ga if a == b else G[b]
        M[i] = windowed_displacement_products(Ga, Gb, m, n_origins)
    return M / n_origins


def _species_fluxes(J, fluxes):
    """Check the heat flux and species fluxes; return them as arrays.

    Returns
    -------
    numpy.ndarray(3, n)
        The heat flux.
    numpy.ndarray(k, 3, n)
        The species fluxes.
    """
    J = np.asarray(J, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    if fluxes.ndim == 2:
        fluxes = fluxes[np.newaxis]
    if J.ndim != 2 or J.shape[0] != 3:
        raise ValueError(f"The heat flux must be 3 x n, not {J.shape}")
    if fluxes.ndim != 3 or fluxes.shape[1:] != J.shape:
        raise ValueError(
            f"The species fluxes must be k x 3 x {J.shape[1]}, not {fluxes.shape}"
        )
    return J, fluxes


def schur_complement(L, coefficients=False):
    """L_ee - L_eJ L_JJ^-1 L_Je for each lag: the thermal conductivity of a mixture.

    The matrices are of the Green-Kubo integrals or Helfand moments of the heat
    flux e, first, and the mass fluxes J of all the species but one (see
    :func:`multicomponent_correlation_functions`).

    Parameters
    ----------
    L : numpy.ndarray(..., m, 1 + k, 1 + k)
        The matrices for each lag, with the heat flux first.
    coefficients : bool = False
        Also return the coefficients L_JJ^-1 L_Je.

    Returns
    -------
    numpy.ndarray(..., m)
        The Schur complement; L_ee at lags where L_JJ is singular, such as lag 0.
    numpy.ndarray(..., m, k)
        The coefficients, if asked for; zero where L_JJ is singular.
    """
    L = np.asarray(L, dtype=float)
    shape = L.shape[:-2]
    p = L.shape[-1]
    L = L.reshape(-1, p, p)
    result = L[:, 0, 0].copy()
    a = np.zeros((L.shape[0], p - 1))
    LJJ = L[:, 1:, 1:]
    LJe = L[:, 1:, 0]
    usable = np.abs(np.linalg.det(LJJ)) > 0
    if np.any(usable):
        try:
            x = np.linalg.solve(LJJ[usable], LJe[usable][..., np.newaxis])[..., 0]
        except np.linalg.LinAlgError:
            x = np.einsum("mij,mj->mi", np.linalg.pinv(LJJ[usable]), LJe[usable])
        a[usable] = x
        result[usable] -= np.einsum("mi,mi->m", LJe[usable], x)
    result = result.reshape(shape)
    if coefficients:
        return result, a.reshape(shape + (p - 1,))
    return result


def flux_correlation_matrices(J, fluxes, m=None):
    """The correlation functions of the heat flux and the species' mass fluxes, and
    their running integrals, for each direction.

    For direction d the series are the heat flux J_d and the mass fluxes of the
    species in the same direction. The correlation functions are symmetrized,
    (<a(t) b(0)> + <b(t) a(0)>) / 2, as the Onsager coefficients of two fluxes odd
    in time are. Being linear in the data, the matrices of several runs can be
    averaged before :func:`schur_complement` is applied, which is much less noisy
    than the complement of each run.

    Parameters
    ----------
    J : numpy.ndarray(3, n)
        The x, y and z components of the heat flux.
    fluxes : numpy.ndarray(k, 3, n)
        The mass fluxes of k species, with k one less than the number of species.
    m : int
        The number of lags to keep; all of them by default.

    Returns
    -------
    numpy.ndarray(3, m, 1 + k, 1 + k)
        The correlation functions, the heat flux first.
    numpy.ndarray(3, m, 1 + k, 1 + k)
        Their running integrals, in units of the sampling interval.
    """
    J, fluxes = _species_fluxes(J, fluxes)
    n = J.shape[1]
    if m is None:
        m = n
    p = 1 + fluxes.shape[0]
    length = next_fast_len(2 * n - 1, real=True)
    adjust = np.arange(n, n - m, -1)[:, np.newaxis]
    C = np.empty((3, m, p, p))
    for d in range(3):
        X = np.concatenate([J[d : d + 1], fluxes[:, d]])
        X = X - X.mean(axis=1, keepdims=True)
        F = rfft(X, length, axis=1)
        for a in range(p):
            # <x_a(t + k) x_b(t)> for all b at once
            C[d, :, a, :] = irfft(F[a] * np.conj(F), length, axis=1)[:, :m].T / adjust
    C = 0.5 * (C + C.transpose(0, 1, 3, 2))
    return C, cumulative_trapezoid(C, axis=1, initial=0.0)


def flux_helfand_matrices(J, fluxes, m=None):
    """The Helfand moments of the heat flux and the species' mass fluxes, for each
    direction.

    The matrix form of :func:`collective_helfand_moments`: for direction d,
    <G_a(k) G_b(k)> for the integrals G of the heat flux J_d and of the mass fluxes
    of the species in the same direction. Like
    :func:`flux_correlation_matrices`, the matrices of several runs can be
    averaged before :func:`schur_complement` is applied.

    Parameters
    ----------
    J : numpy.ndarray(3, n)
        The x, y and z components of the heat flux.
    fluxes : numpy.ndarray(k, 3, n)
        The mass fluxes of k species, with k one less than the number of species.
    m : int
        The number of lags; by default min(n // 20, 10000).

    Returns
    -------
    numpy.ndarray(3, m, 1 + k, 1 + k)
        The Helfand moments, in units of the sampling interval squared, the heat
        flux first.
    """
    J, fluxes = _species_fluxes(J, fluxes)
    n = J.shape[1]
    if m is None:
        m = min(n // 20, 10000)
    n_origins = n - m
    p = 1 + fluxes.shape[0]
    ia, ib = np.triu_indices(p)
    M = np.empty((3, m, p, p))
    for d in range(3):
        X = np.concatenate([J[d : d + 1], fluxes[:, d]])
        G = cumulative_trapezoid(X, axis=1, initial=0.0)
        S = windowed_displacement_products(G[ia].T, G[ib].T, m, n_origins)
        M[d][:, ia, ib] = S
        M[d][:, ib, ia] = S
    return M / n_origins


def effective_correlation_functions(C, L):
    """The derivative of the Schur complement of the integrals: the correlation
    function of the heat flux less the part carried by the species, with the
    coefficients L_JJ^-1 L_Je taken as fixed at each lag.

    Parameters
    ----------
    C, L : numpy.ndarray(..., m, 1 + k, 1 + k)
        The correlation functions and their integrals, from
        :func:`flux_correlation_matrices` or averages of them.

    Returns
    -------
    numpy.ndarray(..., m)
    """
    _, a = schur_complement(L, coefficients=True)
    # At lag 0 the integrals vanish; use the coefficients of lag 1
    if a.shape[-2] > 1:
        a[..., 0, :] = a[..., 1, :]
    return (
        C[..., 0, 0]
        - 2 * np.einsum("...i,...i->...", a, C[..., 1:, 0])
        + np.einsum("...i,...ij,...j->...", a, C[..., 1:, 1:], a)
    )


def multicomponent_correlation_functions(J, fluxes, m=None):
    """The Green–Kubo input for the heat flux of a mixture.

    In a mixture the heat flux is correlated with the interdiffusion of the
    species, and the Green–Kubo integral of its autocorrelation function gives too
    large a thermal conductivity. The thermal conductivity comes instead from the
    Schur complement of the matrix of Onsager coefficients (Bertossa, Grasselli,
    Ercole and Baroni, Phys. Rev. Lett. 122, 255901 (2019)):

    .. math::

        \\kappa \\propto L_{ee} - L_{eJ} L_{JJ}^{-1} L_{Je}, \\qquad
        L_{ab}(t) = \\int_0^t \\langle a(\\tau) b(0) \\rangle \\, d\\tau

    with e the heat flux and J the mass fluxes of all the species but one (the
    last is fixed by the total momentum, which is zero). Their scale cancels, so
    any units, or the momenta of the species, will do. The complement is taken for
    each lag and each direction, coupling the fluxes in the same direction, as
    for an isotropic fluid.

    This is for one run. The complement of a short run is noisy, since L_JJ is
    small and poorly known; for several runs or blocks, average their
    :func:`flux_correlation_matrices` and take the complement of the average.

    Parameters
    ----------
    J : numpy.ndarray(3, n)
        The x, y and z components of the heat flux.
    fluxes : numpy.ndarray(k, 3, n)
        The mass fluxes of k species, with k one less than the number of species.
    m : int
        The number of lags to keep; all of them by default.

    Returns
    -------
    numpy.ndarray(6, m)
        The correlation functions, in the order of
        :data:`seamm_md_analysis.correlate.TENSOR_COMPONENTS`. The diagonal ones
        are the derivatives of the corrected integrals
        (:func:`effective_correlation_functions`); the off-diagonal ones are
        those of the heat flux alone, which are zero either way for an isotropic
        system.
    numpy.ndarray(6, m)
        Their running integrals, in units of the sampling interval.
    """
    from .correlate import tensor_correlation_functions

    J, fluxes = _species_fluxes(J, fluxes)
    if m is None:
        m = J.shape[1]
    ccf, integral = tensor_correlation_functions(J, m=m)
    C, L = flux_correlation_matrices(J, fluxes, m=m)
    integral[:3] = schur_complement(L)
    ccf[:3] = effective_correlation_functions(C, L)
    return ccf, integral


def multicomponent_helfand_moments(J, fluxes, m=None):
    """The Helfand moments of the heat flux of a mixture.

    The Helfand-moment form of :func:`multicomponent_correlation_functions`: the
    Schur complement, for each lag and direction, of the matrix of Helfand moments
    of the heat flux and the mass fluxes of all the species but one. At long times
    the moments grow as twice the time times the Onsager coefficients, so the
    slope of the complement gives the thermal conductivity of the mixture. As
    there, for several runs average their :func:`flux_helfand_matrices` and take
    the complement of the average.

    Parameters
    ----------
    J : numpy.ndarray(3, n)
        The x, y and z components of the heat flux.
    fluxes : numpy.ndarray(k, 3, n)
        The mass fluxes of k species, with k one less than the number of species.
    m : int
        The number of lags; by default min(n // 20, 10000).

    Returns
    -------
    numpy.ndarray(6, m)
        The Helfand moments, in units of the sampling interval squared, in the
        order of :data:`seamm_md_analysis.correlate.TENSOR_COMPONENTS`. The
        off-diagonal ones are those of the heat flux alone, as from
        :func:`collective_helfand_moments`.
    """
    J, fluxes = _species_fluxes(J, fluxes)
    if m is None:
        m = min(J.shape[1] // 20, 10000)
    M = collective_helfand_moments(J, m=m)
    M[:3] = schur_complement(flux_helfand_matrices(J, fluxes, m=m))
    return M


def component_helfand_moments(X, m=None, demean=True):
    """The Helfand moment of each of several independent series.

    .. math::

        M_i(k) = \\left\\langle \\left( \\int_0^{k} X_i \\, dt \\right)^2
                 \\right\\rangle

    averaged over n - m time origins, in units of the sampling interval squared.
    For the shear stress (:func:`shear_stress_components`) this gives the
    Helfand moments of the viscosity.

    Parameters
    ----------
    X : numpy.ndarray(k, n)
        The k series, each of n samples.
    m : int
        The number of lags; by default min(n // 20, 10000).
    demean : bool = True
        Subtract the mean of each series first. A non-zero mean adds a term
        growing as the square of the time, which is not part of the transport
        coefficient. When the true mean is known to be zero, as for the shear
        stress of a fluid, do not: subtracting the sample mean of a finite series
        biases the moments low.

    Returns
    -------
    numpy.ndarray(k, m)
        The Helfand moments.
    """
    X = np.asarray(X, dtype=float)
    if demean:
        X = X - X.mean(axis=1, keepdims=True)
    n = X.shape[1]
    if m is None:
        m = min(n // 20, 10000)
    n_origins = n - m
    G = cumulative_trapezoid(X, axis=1, initial=0.0).T
    S = windowed_displacement_products(G, G, m, n_origins)
    return S.T / n_origins


def shear_stress_components(pressure_tensor):
    """The five independent shear components of a pressure tensor.

    For an isotropic fluid, Pxy, Pxz, Pyz, (Pxx - Pyy)/2 and (Pyy - Pzz)/2 all
    have the same autocorrelation function, so each gives the shear viscosity by
    Green–Kubo, and averaging the five improves the statistics over using only
    the three off-diagonal components (Daivis and Evans, 1994). The order is that
    of :data:`SHEAR_COMPONENTS`.

    Parameters
    ----------
    pressure_tensor : numpy.ndarray(n, 3, 3)
        The pressure tensor at each sample.

    Returns
    -------
    numpy.ndarray(5, n)
        The five components.
    """
    P = np.asarray(pressure_tensor, dtype=float)
    if P.ndim != 3 or P.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected a pressure tensor of shape (n, 3, 3), not {P.shape}"
        )
    return np.stack(
        [
            P[:, 0, 1],
            P[:, 0, 2],
            P[:, 1, 2],
            (P[:, 0, 0] - P[:, 1, 1]) / 2,
            (P[:, 1, 1] - P[:, 2, 2]) / 2,
        ]
    )


def species_helfand_moments(v, species, m=None, max_bytes=256 * 2**20):
    """Single-molecule Helfand moments from velocities, for each species.

    For each molecule and direction, the square of the integral of its
    velocity over a window of k steps, averaged over n - m time origins, then
    averaged over the molecules of the species. A fourth column holds the total,
    the sum over x, y and z.

    The molecules are taken a batch at a time, so the memory used is bounded by
    ``max_bytes`` whatever the length of the run: the integrals and the FFTs
    need several times the velocities of the molecules they work on, which for
    every molecule of a long run at once was ~8 times the whole array. The
    batches' means and spreads are combined exactly.

    Parameters
    ----------
    v : numpy.ndarray(n, n_molecules, 3)
        The velocities at each step.
    species : [[int]]
        The indices of the molecules of each species.
    m : int
        The number of lags; by default min(n // 2, 5000).
    max_bytes : int
        About how much memory a batch of molecules may use.

    Returns
    -------
    [numpy.ndarray(m, 4)]
        The Helfand moments of each species, in units of velocity times the
        sampling interval, squared.
    [numpy.ndarray(m, 4)]
        The standard deviation over the molecules.
    """
    n = v.shape[0]
    if m is None:
        m = min(n // 2, 5000)
    n_origins = n - m
    # About 12 arrays of n x 3 floats per molecule: its velocities, their
    # integral, the detrended copy, and the FFTs' work arrays
    per_molecule = 12 * n * 3 * 8
    batch = max(1, int(max_bytes // per_molecule))

    Ms = []
    M_errs = []
    for molecules in species:
        molecules = list(molecules)
        count = 0
        mean = None
        M2 = None  # the sum of squared deviations from the mean
        for i in range(0, len(molecules), batch):
            chunk = molecules[i : i + batch]
            vs = np.asarray(v[:, chunk, :], dtype=float)
            G = cumulative_trapezoid(vs, axis=0, initial=0.0)
            del vs
            S = windowed_displacement_products(G, G, m, n_origins)
            del G
            S = np.concatenate((S, S.sum(axis=2, keepdims=True)), axis=2)
            S /= n_origins
            k = S.shape[1]
            chunk_mean = S.mean(axis=1)
            chunk_M2 = ((S - chunk_mean[:, None, :]) ** 2).sum(axis=1)
            if mean is None:
                count, mean, M2 = k, chunk_mean, chunk_M2
            else:
                # Chan et al.'s combination of two sets' means and variances
                total = count + k
                delta = chunk_mean - mean
                mean = mean + delta * (k / total)
                M2 = M2 + chunk_M2 + delta**2 * (count * k / total)
                count = total
        Ms.append(mean)
        M_errs.append(np.sqrt(M2 / count))
    return Ms, M_errs


def squared_displacement_sums(x):
    """Sum over all time origins of the squared displacement at each lag.

    .. math::

        S(k) = \\sum_{i=0}^{n-1-k} (x[i+k] - x[i])^2

    computed with FFTs for every column.

    Parameters
    ----------
    x : numpy.ndarray(n, ...)
        The positions, with time along the first axis.

    Returns
    -------
    numpy.ndarray(n, ...)
        S for lags 0 .. n-1; divide by n - k for the mean square displacement.
    """
    x = np.asarray(x, dtype=float)
    shape = x.shape
    n = shape[0]
    x = x.reshape(n, -1)
    x = x - x.mean(axis=0)
    sq = x * x
    cs = np.zeros((n + 1, x.shape[1]))
    np.cumsum(sq, axis=0, out=cs[1:])
    k = np.arange(n)
    # sum_{i=0}^{n-1-k} x[i]^2 + sum_{i=k}^{n-1} x[i]^2
    squares = cs[n - k] + (cs[n] - cs[k])
    S = squares - 2 * _lagged_products(x, x, n, n)[:n]
    S[0] = 0.0  # no displacement at lag 0
    return S.reshape(shape)


def msd_by_species(xyz, species, average=True):
    """The mean square displacement of each species, averaged over molecules.

    Parameters
    ----------
    xyz : numpy.ndarray(n, n_molecules, 3)
        The unwrapped positions at each step.
    species : [[int]]
        The indices of the molecules of each species.
    average : bool = True
        Add a fourth column with the total MSD (the sum over x, y and z).

    Returns
    -------
    [numpy.ndarray(n, 3 or 4)]
        The MSD of each species.
    [numpy.ndarray(n, 3 or 4)]
        The standard deviation over the molecules.
    """
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    tmp = squared_displacement_sums(xyz)
    norm = np.arange(n, 0, -1).reshape((n, 1))

    msd = []
    err = []
    for molecules in species:
        values = tmp[:, molecules, :]
        result = np.average(values, axis=1)
        error = np.std(values, axis=1)
        if average:
            total = np.sum(tmp, axis=2)[:, molecules]
            result = np.concatenate(
                (result, np.average(total, axis=1).reshape((n, 1))), axis=1
            )
            error = np.concatenate(
                (error, np.std(total, axis=1).reshape((n, 1))), axis=1
            )
        msd.append(result / norm)
        err.append(error / norm)
    return msd, err


def fit_slope(y, xs, sigma=None, start=0.1, end=0.9):
    """A straight-line fit to the part of a curve between two fractions.

    Parameters
    ----------
    y : numpy.ndarray(n)
        The curve, for example an MSD or a Helfand moment.
    xs : numpy.ndarray(n) or [float]
        The times.
    sigma : numpy.ndarray(n)
        The standard error of y, to weight the fit (optional). Without it, or
        if any value is zero or not finite, the fit is unweighted.
    start : float = 0.1
        The fraction of the curve to skip at the beginning.
    end : float = 0.9
        The fraction of the curve at which to stop.

    Returns
    -------
    LinearFit
        The slope, its standard error, and the fitted line over the range.
    """
    n = len(y)
    i = int(n * start)
    j = int(n * end)
    kwargs = _sigma_kwargs(None if sigma is None else sigma[i:j])
    popt, pcov, infodict, msg, ierr = curve_fit(
        axb, xs[i:j], y[i:j], full_output=True, **kwargs
    )
    slope = float(popt[0])
    b = float(popt[1])
    err = float(np.sqrt(np.diag(pcov)[0]))
    ys = [axb(x, slope, b) for x in xs[i:j]]
    return LinearFit(slope, err, xs[i:j], ys)


def fit_helfand_moment(y, xs, sigma=None, start=1, logger=logger):
    """A straight-line fit to a Helfand moment, from time ``start`` to the end.

    Parameters
    ----------
    y : numpy.ndarray(n)
        The Helfand moment.
    xs : numpy.ndarray(n) or [float]
        The times.
    sigma : numpy.ndarray(n)
        The standard error of y, to weight the fit (optional). Without it, or
        if any value is zero or not finite, the fit is unweighted.
    start : float = 1
        The time, in the units of xs, before which the curve is ignored.

    Returns
    -------
    LinearFit
        The slope, its standard error, and the fitted line.
    """
    dx = xs[1] - xs[0]

    # The curves bend near the origin, so ignore the first part
    i = int(start / dx)
    if i > len(y):
        i = len(y) // 2

    kwargs = _sigma_kwargs(None if sigma is None else sigma[i:])
    popt, pcov, infodict, msg, ierr = curve_fit(
        axb, xs[i:], y[i:], full_output=True, **kwargs
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("")
        logger.debug(f"{popt=}")
        logger.debug(f"{pcov=}")
        logger.debug(pprint.pformat(infodict, compact=True))
        logger.debug(f"{msg=}")
        logger.debug(f"{ierr=}")

    slope = float(popt[0])
    b = float(popt[1])
    err = float(np.sqrt(np.diag(pcov)[0]))
    ys = [axb(x, slope, b) for x in xs[i:]]
    return LinearFit(slope, err, xs[i:], ys)


def helfand_slope(y, xs, sigma):
    """The local slope of a Helfand moment, by central differences.

    The slope of a Helfand moment is the running Green–Kubo integral, so it is
    zero at the origin.

    Parameters
    ----------
    y : numpy.ndarray(n)
        The Helfand moment.
    xs : numpy.ndarray(n) or [float]
        The times.
    sigma : numpy.ndarray(n)
        The standard error of y.

    Returns
    -------
    numpy.ndarray(n)
        The slope: zero at the first point, a central difference in between and
        a backward difference at the last point.
    xs
        The times, unchanged.
    numpy.ndarray(n)
        The error of the slope, from the sum of the errors of the two points.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(y)
    dx = xs[1] - xs[0]
    slope = np.zeros_like(y)
    slope[1 : n - 1] = (y[2:] - y[0 : n - 2]) / (2 * dx)
    slope[n - 1] = (y[n - 1] - y[n - 2]) / dx
    new_sigma = np.zeros_like(y)
    new_sigma[1 : n - 1] = (sigma[2:] + sigma[0 : n - 2]) / (2 * dx)
    new_sigma[n - 1] = (sigma[n - 1] + sigma[n - 2]) / dx
    new_sigma[0] = new_sigma[1]
    return slope, xs, new_sigma


def fit_green_kubo_integral(y, xs, sigma=None):
    """Fit a Green–Kubo running integral by one or two rising exponentials.

    The plateau of the integral, a = sum of the amplitudes, is the transport
    coefficient. A single exponential a (1 - exp(-t / tau)) is fitted first
    over the initial rise; a second exponential is then added over tau / 2 to
    20 tau. The second exponential is kept only if both amplitudes are
    positive, the two time constants differ, the slower one is short enough
    for the plateau to be reached within the fitted data (3 tau), and the
    value is no more than twice that of the single exponential. Otherwise the
    single exponential is used.

    The standard error of the value includes the covariance of the two
    amplitudes, which are strongly correlated.

    Parameters
    ----------
    y : numpy.ndarray(n)
        The running integral.
    xs : numpy.ndarray(n) or [float]
        The times.
    sigma : numpy.ndarray(n)
        The standard error of y (optional). Without it, or if any value is
        zero or not finite (as for the spread of a single run), the fit is
        unweighted and the errors come from the residuals.

    Returns
    -------
    GreenKuboFit
        The plateau value and its standard error, the amplitudes and time
        constants with their errors, and the fitted curve.
    """
    y = np.asarray(y, dtype=float)
    weighted = _usable_sigma(sigma)
    if weighted:
        sigma = np.asarray(sigma, dtype=float)

    dx = xs[1] - xs[0]
    width = max(1, int(0.4 / dx))  # 0.4 in the units of xs, normally ps
    ma = moving_average(y, width)

    # Find the initial, steep rise. Ignore the first couple of points.
    last = min(len(ma) // 2, len(ma) - width)
    if last <= 2:
        raise ValueError(
            f"Too few points to fit the Green–Kubo integral: {y.size} points over "
            f"{xs[-1] - xs[0]:.3g}, but at least {2 * width + 2} are needed, a little "
            "over twice the 0.4 window of the moving average."
        )
    for i in range(2, last):
        if ma[i] > 0.9 * ma[i + width]:
            break
    npts = i + width
    tau_guess = xs[npts]
    a_guess = ma[i]
    npts *= 20
    if 2 * npts > y.size:
        npts = y.size // 2

    if weighted:
        sigma0 = sigma + np.average(sigma[0 : y.size // 10])
        kwargs1 = {"sigma": sigma0[1:npts], "absolute_sigma": True}
    else:
        kwargs1 = {}

    # First fit with single exponential
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", OptimizeWarning)
        popt1, pcov1, infodict1, msg1, ierr1 = curve_fit(
            exp_1,
            xs[1:npts],
            y[1:npts],
            full_output=True,
            p0=[a_guess, tau_guess],
            **kwargs1,
        )

    a1 = popt1[0]
    tau1 = popt1[1]

    # Keep the next fit to a "good" part of the data
    i_min = int(tau1 / dx / 2)  # tau / 2
    i_max = int(tau1 / dx * 20)  # 20 * tau1

    def single():
        err = np.sqrt(np.diag(pcov1)).tolist()
        return [popt1[0]], [err[0]], [popt1[1]], [err[1]], popt1[0], err[0]

    # And add a second exponential
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov, infodict, msg, ierr = curve_fit(
                exp_2,
                xs[i_min:i_max],
                y[i_min:i_max],
                full_output=True,
                p0=[a1, tau1, 0.1 * a1, 5 * tau1],
                **(
                    {"sigma": sigma[i_min:i_max], "absolute_sigma": True}
                    if weighted
                    else {}
                ),
            )
        err = np.sqrt(np.diag(pcov)).tolist()
        a = [popt[0], popt[2]]
        a_err = [err[0], err[2]]
        tau = [popt[1], popt[3]]
        tau_err = [err[1], err[3]]
        value = a[0] + a[1]
        value_err = float(np.sqrt(pcov[0, 0] + pcov[2, 2] + 2 * pcov[0, 2]))
        t_end = xs[min(i_max, len(xs)) - 1]
        if (
            not np.all(np.isfinite(popt))
            or not np.all(np.isfinite(pcov))
            or min(a) <= 0
            or min(tau) <= 0
            or abs(tau[1] - tau[0]) < 0.1 * max(tau)  # the same exponential
            or 3 * max(tau) > t_end  # no plateau within the data
            or value > 2 * abs(a1)  # Shouldn't change this much!
        ):
            a, a_err, tau, tau_err, value, value_err = single()
    except Exception:
        a, a_err, tau, tau_err, value, value_err = single()

    # The fitted curve out to 20 times the longest time constant
    nf = int(max(tau) / dx) * 20
    if nf > len(xs):
        nf = len(xs)

    if len(a) == 1:
        fy = [exp_1(t, a[0], tau[0]) for t in xs[1:nf]]
    else:
        fy = [exp_2(t, a[0], tau[0], a[1], tau[1]) for t in xs[1:nf]]

    return GreenKuboFit(value, value_err, a, a_err, tau, tau_err, xs[1:nf], fy)


def yeh_hummer_correction(T, L, viscosity):
    """The Yeh–Hummer finite-size correction to a diffusion constant.

    .. math::

        \\Delta D = \\frac{\\xi k_B T}{6 \\pi \\eta L}

    with xi = 2.837297 for a cubic cell. Add it to the diffusion constant
    computed in the periodic cell.

    Parameters
    ----------
    T : float
        The temperature, in K.
    L : float
        The length of the cubic cell, in m.
    viscosity : float
        The shear viscosity, in Pa s.

    Returns
    -------
    float
        The correction, in m^2/s.
    """
    return YEH_HUMMER_XI * k_B * T / (6 * pi * viscosity * L)
