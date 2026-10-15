# -*- coding: utf-8 -*-

"""Tests of seamm_md_analysis.transport."""

import numpy as np
import pytest

from seamm_md_analysis import transport
from . import reference


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_collective_helfand_moments_match_loop(rng):
    # A flux with a mean and some correlation, like a real heat flux
    J = rng.normal(size=(3, 700)) + 0.3
    J[:, 1:] += 0.5 * J[:, :-1]
    for m in (None, 1, 50, 349):
        new = transport.collective_helfand_moments(J, m=m)
        old = reference.tc_create_helfand_moments(J, m=m)
        assert new.shape == old.shape
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12 * abs(old).max())


def test_species_helfand_moments_match_loop(rng):
    v = rng.normal(size=(240, 7, 3))
    species = [[0, 2, 4], [1, 3, 5, 6]]
    for m in (None, 30):
        new, new_err = transport.species_helfand_moments(v, species, m=m)
        old, old_err = reference.diff_create_helfand_moments(v, species, m=m)
        for a, b in zip(new, old):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * abs(b).max())
        for a, b in zip(new_err, old_err):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-11 * abs(b).max())


def test_msd_by_species_matches_loop(rng):
    xyz = np.cumsum(rng.normal(size=(150, 9, 3)), axis=0) + 20.0
    species = [[0, 1, 2, 3], [4, 5, 6, 7, 8]]
    for average in (True, False):
        new, new_err = transport.msd_by_species(xyz, species, average=average)
        old, old_err = reference.diff_compute_msd(xyz, species, average=average)
        for a, b in zip(new, old):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
        for a, b in zip(new_err, old_err):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def test_windowed_displacement_products_cross(rng):
    a = np.cumsum(rng.normal(size=(120, 2)), axis=0)
    b = np.cumsum(rng.normal(size=(120, 2)), axis=0)
    m, n0 = 25, 90
    S = transport.windowed_displacement_products(a, b, m, n0)
    direct = np.array(
        [
            sum((a[i + k] - a[i]) * (b[i + k] - b[i]) for i in range(n0))
            for k in range(m)
        ]
    )
    np.testing.assert_allclose(S, direct, rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError):
        transport.windowed_displacement_products(a, b, 40, 90)


def test_random_walk_msd_slope(rng):
    # Unit-variance steps: the MSD per direction is k, so the slope is 1.
    xyz = np.cumsum(rng.normal(size=(2000, 200, 3)), axis=0)
    msd, err = transport.msd_by_species(xyz, [list(range(200))])
    ts = list(np.arange(2000) * 1.0)
    fit = transport.fit_slope(msd[0][:, 0], ts, start=0.05, end=0.3)
    assert fit.slope == pytest.approx(1.0, rel=0.05)
    slope, stderr, xs, ys = fit  # still unpacks like the old tuple
    assert len(xs) == len(ys)


def test_fit_slope_weighted_with_end_before_last_point():
    # Used to be a shape error: sigma[i:] against y[i:j]
    xs = list(np.linspace(0.0, 10.0, 101))
    y = 3.0 * np.array(xs) + 1.0
    sigma = np.full(101, 0.1)
    fit = transport.fit_slope(y, xs, sigma=sigma, start=0.1, end=0.6)
    assert fit.slope == pytest.approx(3.0)


def _ou_process(rng, n, dt, tau, sigma):
    """An Ornstein–Uhlenbeck series: <x(t)x(0)> = sigma^2 exp(-t/tau)."""
    phi = np.exp(-dt / tau)
    noise = rng.normal(size=n) * sigma * np.sqrt(1 - phi**2)
    x = np.empty(n)
    x[0] = rng.normal() * sigma
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_green_kubo_ou_process(rng):
    # The integral of sigma^2 exp(-t/tau) is sigma^2 tau = 2.0 * 0.5 = 1.0
    from seamm_md_analysis.correlate import tensor_correlation_functions

    dt, tau = 0.01, 0.5
    J = np.stack([_ou_process(rng, 200_000, dt, tau, np.sqrt(2.0)) for _ in range(3)])
    m = 2000
    ccf, integral = tensor_correlation_functions(J, m=m)
    integral *= dt
    ts = list(np.arange(m) * dt)
    sigma = np.full(m, 0.02)
    fit = transport.fit_green_kubo_integral(integral[0], ts, sigma=sigma)
    assert fit.value == pytest.approx(1.0, rel=0.15)
    assert len(fit.xs) == len(fit.ys)
    # and the Helfand moment grows with slope 2 * (sigma^2 tau) per unit time
    M = transport.collective_helfand_moments(J, m=m) * dt**2
    slope = transport.fit_helfand_moment(M[0], ts, sigma=np.full(m, 1.0), start=3.0)
    assert slope.slope == pytest.approx(2.0, rel=0.15)


def test_green_kubo_without_sigma(rng):
    # Without sigma the fit is unweighted: the same as a uniform sigma
    from seamm_md_analysis.correlate import tensor_correlation_functions

    dt, tau = 0.01, 0.5
    J = np.stack([_ou_process(rng, 200_000, dt, tau, np.sqrt(2.0)) for _ in range(3)])
    m = 2000
    ccf, integral = tensor_correlation_functions(J, m=m)
    integral *= dt
    ts = list(np.arange(m) * dt)
    fit = transport.fit_green_kubo_integral(integral[0], ts)
    uniform = transport.fit_green_kubo_integral(integral[0], ts, sigma=np.ones(m))
    assert fit.value == pytest.approx(uniform.value)
    assert np.isfinite(fit.stderr)


def test_green_kubo_too_short():
    with pytest.raises(ValueError):
        transport.fit_green_kubo_integral(np.ones(3), [0.0, 1.0, 2.0])


def test_helfand_slope():
    xs = list(np.arange(10) * 0.5)
    y = 4.0 * np.array(xs)
    slope, x, err = transport.helfand_slope(y, xs, np.ones(10))
    assert slope[0] == 0 and slope[1] == 0
    np.testing.assert_allclose(slope[2:], 4.0)
    assert x is xs
    np.testing.assert_allclose(err, 2.0)


def test_yeh_hummer_correction():
    # Water at 298.15 K in a 3 nm box, viscosity 0.89 mPa s
    from math import pi

    dD = transport.yeh_hummer_correction(298.15, 3.0e-9, 0.89e-3)
    expected = 2.837297 * 1.380649e-23 * 298.15 / (6 * pi * 0.89e-3 * 3.0e-9)
    assert dD == pytest.approx(expected)
    assert dD == pytest.approx(2.32e-10, rel=0.01)


def test_collective_helfand_moments_large_mean(rng):
    # A large mean makes the running integral drift; the result must stay exact
    J = rng.normal(size=(3, 3000)) + np.array([[1.0e3], [-2.0e3], [5.0e2]])
    new = transport.collective_helfand_moments(J, m=100)
    old = reference.tc_create_helfand_moments(J, m=100)
    np.testing.assert_allclose(new, old, rtol=1e-9)


def test_green_kubo_coarse_sampling(rng):
    # Sampling coarser than 0.4 ps used to give a zero-width moving average
    dt, tau = 0.5, 5.0
    J = _ou_process(rng, 20_000, dt, tau, 1.0)
    from seamm_md_analysis.correlate import ccovf
    from scipy.integrate import cumulative_trapezoid

    m = 400
    integral = cumulative_trapezoid(ccovf(J, J)[:m], initial=0.0) * dt
    fit = transport.fit_green_kubo_integral(
        integral, list(np.arange(m) * dt), sigma=np.full(m, 0.1)
    )
    assert np.isfinite(fit.value)
