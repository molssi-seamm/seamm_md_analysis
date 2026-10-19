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
        # x, y and z as before; the total is now their sum, not the moment of
        # the summed velocities (which added the noisy cross terms)
        for a, b in zip(new, old):
            b = b[:, :3]
            np.testing.assert_allclose(
                a[:, :3], b, rtol=1e-9, atol=1e-12 * abs(b).max()
            )
            np.testing.assert_allclose(a[:, 3], a[:, :3].sum(axis=1))
        for a, b in zip(new_err, old_err):
            b = b[:, :3]
            np.testing.assert_allclose(
                a[:, :3], b, rtol=1e-8, atol=1e-11 * abs(b).max()
            )
            assert np.all(np.isfinite(a[:, 3])) and np.all(a[:, 3] >= 0)


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
    # Without sigma the fit is unweighted: the same value as a uniform sigma
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
    # The spread of a single run is zero: that is the same as no sigma
    zero = transport.fit_green_kubo_integral(integral[0], ts, sigma=np.zeros(m))
    assert zero.value == pytest.approx(fit.value)
    assert zero.stderr == pytest.approx(fit.stderr)


def _ou_integral(seed, n=100_000, m=1500, dt=0.01, tau=0.5):
    """The running Green-Kubo integral of an OU series; the plateau is tau."""
    from scipy.integrate import cumulative_trapezoid

    from seamm_md_analysis.correlate import ccovf

    J = _ou_process(np.random.default_rng(seed), n, dt, tau, 1.0)
    integral = cumulative_trapezoid(ccovf(J, J)[:m], initial=0.0) * dt
    return integral, list(np.arange(m) * dt)


def test_green_kubo_rejects_unphysical_second_exponential():
    # This series once gave a second exponential with a negative amplitude and
    # a time constant of 46 ps, beyond the 15 ps of data, and a value of 0.018
    integral, ts = _ou_integral(12345)
    fit = transport.fit_green_kubo_integral(integral, ts)
    assert fit.value == pytest.approx(0.5, rel=0.25)
    assert min(fit.a) > 0
    assert 3 * max(fit.tau) < ts[-1]


def test_green_kubo_errors_are_sensible():
    # With two nearly equal time constants the amplitudes are individually
    # undetermined; the error of their sum must still be small
    for seed in range(10):
        integral, ts = _ou_integral(seed)
        fit = transport.fit_green_kubo_integral(integral, ts)
        assert 0.3 < fit.value < 0.7
        assert 0 < fit.stderr < 0.2


def test_fit_helfand_moment_with_zero_sigma():
    # A single run has zero spread; the fit must not fail and return its guess
    xs = list(np.arange(200) * 0.1)
    y = 3.0 * np.array(xs) + 1.0
    fit = transport.fit_helfand_moment(y, xs, sigma=np.zeros(200), start=1)
    assert fit.slope == pytest.approx(3.0)
    fit = transport.fit_slope(y, xs, sigma=np.zeros(200))
    assert fit.slope == pytest.approx(3.0)


def test_green_kubo_too_short():
    with pytest.raises(ValueError):
        transport.fit_green_kubo_integral(np.ones(3), [0.0, 1.0, 2.0])


def test_helfand_slope():
    # Central differences: the slope of t^2 is exactly 2t at the interior points
    xs = list(np.arange(10) * 0.5)
    t = np.array(xs)
    slope, x, err = transport.helfand_slope(4.0 * t, xs, np.ones(10))
    assert slope[0] == 0
    np.testing.assert_allclose(slope[1:], 4.0)
    assert x is xs
    # the error is in the units of the slope: (1 + 1) / (2 * 0.5)
    np.testing.assert_allclose(err[:-1], 2.0)
    np.testing.assert_allclose(err[-1], 4.0)
    slope, _, _ = transport.helfand_slope(t**2, xs, np.ones(10))
    np.testing.assert_allclose(slope[1:-1], 2 * t[1:-1])


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


def test_component_helfand_moments_match_collective(rng):
    # Without demeaning, the diagonal of the collective moments
    J = rng.normal(size=(3, 700)) + 0.3
    J[:, 1:] += 0.5 * J[:, :-1]
    for m in (None, 1, 50, 349):
        new = transport.component_helfand_moments(J, m=m, demean=False)
        old = transport.collective_helfand_moments(J, m=m)[:3]
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12 * abs(old).max())


def test_component_helfand_moments_demean(rng):
    # A constant offset does not change the demeaned moments
    X = rng.normal(size=(5, 600))
    a = transport.component_helfand_moments(X, m=40)
    b = transport.component_helfand_moments(X + 7.0, m=40)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * abs(a).max())


def test_viscosity_like_ou_components(rng):
    # Five independent components, each with a Green-Kubo integral of 1.0
    from seamm_md_analysis.correlate import autocorrelation_functions

    dt, tau = 0.01, 0.5
    X = np.stack([_ou_process(rng, 300_000, dt, tau, np.sqrt(2.0)) for _ in range(5)])
    m = 1000
    acf, integral = autocorrelation_functions(X, m=m)
    integral *= dt
    ts = list(np.arange(m) * dt)
    values = [transport.fit_green_kubo_integral(y, ts).value for y in integral]
    assert np.mean(values) == pytest.approx(1.0, rel=0.1)
    M = transport.component_helfand_moments(X, m=m) * dt**2
    slope = transport.fit_helfand_moment(
        M.mean(axis=0), ts, sigma=np.full(m, 1.0), start=3.0
    )
    assert slope.slope == pytest.approx(2.0, rel=0.1)


def test_shear_stress_components():
    P = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]],
            [[0.0, -1.0, 0.5], [-1.0, 2.0, 0.25], [0.5, 0.25, -2.0]],
        ]
    )
    X = transport.shear_stress_components(P)
    assert X.shape == (5, 2)
    np.testing.assert_allclose(X[:, 0], [2.0, 3.0, 6.0, -2.0, -2.0])
    np.testing.assert_allclose(X[:, 1], [-1.0, 0.5, 0.25, -1.0, 2.0])
    assert len(transport.SHEAR_COMPONENTS) == 5
    with pytest.raises(ValueError):
        transport.shear_stress_components(np.zeros((4, 6)))


def test_shear_components_of_isotropic_stress_agree(rng):
    # A traceless symmetric tensor with Gaussian elements is isotropic, so the
    # five components have the same variance (1/2 here).
    n = 20_000
    A = rng.normal(size=(n, 3, 3))
    S = (A + A.transpose(0, 2, 1)) / 2
    S -= np.trace(S, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    X = transport.shear_stress_components(S)
    var = X.var(axis=1)
    assert var == pytest.approx(np.full(5, 0.5), rel=0.05)


def test_green_kubo_shorter_than_the_window():
    """Too short for the moving average: a clear error, not an IndexError."""
    ts = np.arange(251) * 0.002
    y = 1.0 - np.exp(-ts / 0.1)
    with pytest.raises(ValueError, match="Too few points"):
        transport.fit_green_kubo_integral(y, ts)
    # Just long enough
    ts = np.arange(2 * 200 + 2) * 0.002
    fit = transport.fit_green_kubo_integral(1.0 - np.exp(-ts / 0.1), ts)
    assert fit.value == pytest.approx(1.0, rel=0.05)


def test_species_helfand_moments_in_batches(rng):
    """Batches of molecules, down to one at a time, give the same moments and
    spreads as all the molecules at once."""
    v = rng.normal(size=(300, 11, 3))
    species = [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9]]
    whole, whole_err = transport.species_helfand_moments(v, species, m=40)
    per_molecule = 12 * 300 * 3 * 8
    for batch in (1, 2, 4):
        parts, parts_err = transport.species_helfand_moments(
            v, species, m=40, max_bytes=batch * per_molecule
        )
        for a, b in zip(parts + parts_err, whole + whole_err):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-13 * abs(b).max())


def _ar1(rng, phi, shape):
    """A correlated series along the last axis."""
    noise = rng.normal(size=shape)
    x = np.zeros(shape)
    for t in range(1, shape[-1]):
        x[..., t] = phi * x[..., t - 1] + noise[..., t]
    return x


def _science_reference(e, fluxes, m):
    """The loop of the multicomponent analysis script, one direction at a time."""
    result = []
    for d in range(3):
        X = np.column_stack([e[d]] + [f[d] for f in fluxes])
        n, k = X.shape
        X = X - X.mean(0)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        F = np.fft.rfft(X, nfft, axis=0)
        C = np.empty((m, k, k))
        for a in range(k):
            for b in range(k):
                c = np.fft.irfft(F[:, a] * np.conj(F[:, b]), nfft)[:m]
                C[:, a, b] = c / (n - np.arange(m))
        C = 0.5 * (C + C.transpose(0, 2, 1))
        L = np.cumsum(C, axis=0) - 0.5 * (C + C[0:1])
        kappa = [0.0]
        for t in range(1, m):
            kappa.append(
                L[t, 0, 0] - L[t, 0, 1:] @ np.linalg.solve(L[t, 1:, 1:], L[t, 0, 1:])
            )
        result.append(kappa)
    return np.array(result)


def test_multicomponent_correlation_functions_match_script(rng):
    e = _ar1(rng, 0.9, (3, 4000))
    fluxes = _ar1(rng, 0.95, (2, 3, 4000))
    e = e + 3 * fluxes[0] - 2 * fluxes[1]
    ccf, integral = transport.multicomponent_correlation_functions(e, fluxes, m=400)
    expected = _science_reference(e, fluxes, 400)
    np.testing.assert_allclose(integral[:3], expected, rtol=1e-8, atol=1e-8)
    # The diagonal correlation functions are the derivatives of the integrals
    np.testing.assert_allclose(
        np.diff(integral[0])[5:],
        0.5 * (ccf[0, 5:-1] + ccf[0, 6:]),
        atol=1e-4 * abs(ccf[0]).max(),
    )


def test_multicomponent_removes_the_species_fluxes(rng):
    """Adding any multiple of the species fluxes to the heat flux, or scaling the
    species fluxes, changes nothing; and the result is the conductivity of the
    part of the heat flux uncorrelated with them."""
    n = 50000
    e0 = _ar1(rng, 0.9, (3, n))
    fluxes = _ar1(rng, 0.95, (2, 3, n))
    e = e0 + 3 * fluxes[0] - 2 * fluxes[1]

    _, plain = transport.multicomponent_correlation_functions(e0, fluxes, m=2000)
    _, mixed = transport.multicomponent_correlation_functions(e, fluxes, m=2000)
    _, scaled = transport.multicomponent_correlation_functions(e, 1e3 * fluxes, m=2000)
    np.testing.assert_allclose(mixed[:3], plain[:3], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(scaled[:3], plain[:3], rtol=1e-6, atol=1e-6)

    M_plain = transport.multicomponent_helfand_moments(e0, fluxes, m=1000)
    M_mixed = transport.multicomponent_helfand_moments(e, fluxes, m=1000)
    np.testing.assert_allclose(M_mixed[:3], M_plain[:3], rtol=1e-6, atol=1e-6)
    # The off-diagonal terms are those of the heat flux alone
    M_raw = transport.collective_helfand_moments(e, m=1000)
    np.testing.assert_allclose(M_mixed[3:], M_raw[3:])

    # The slope of the moments is twice the Green-Kubo integral: for an AR(1)
    # series var * (1 + phi) / (1 - phi), with var = 1 / (1 - phi^2), ~ 100 here
    def slope(M):
        return (M[:3, -1] - M[:3, 500]).mean() / 499

    exact = (1 / (1 - 0.81)) * 1.9 / 0.1
    assert slope(M_mixed) == pytest.approx(exact, rel=0.1)
    assert slope(transport.collective_helfand_moments(e, m=1000)) > 10 * exact


def test_multicomponent_checks_shapes(rng):
    e = rng.normal(size=(3, 100))
    with pytest.raises(ValueError):
        transport.multicomponent_correlation_functions(e, rng.normal(size=(2, 3, 99)))
    # One species flux may be given as 3 x n
    _, a = transport.multicomponent_correlation_functions(
        e, rng.normal(size=(3, 100)), m=10
    )
    assert a.shape == (6, 10)


def test_flux_matrices_hold_the_plain_heat_flux(rng):
    """The heat-flux element of the matrices is the plain analysis of the heat
    flux, and the matrices are symmetric."""
    from seamm_md_analysis.correlate import tensor_correlation_functions

    e = _ar1(rng, 0.9, (3, 3000))
    fluxes = _ar1(rng, 0.95, (2, 3, 3000))
    C, L = transport.flux_correlation_matrices(e, fluxes, m=300)
    assert L.shape == (3, 300, 3, 3)
    ccf, integral = tensor_correlation_functions(e, m=300)
    np.testing.assert_allclose(L[:, :, 0, 0], integral[:3], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(C[:, :, 0, 0], ccf[:3], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(L, L.transpose(0, 1, 3, 2))

    M = transport.flux_helfand_matrices(e, fluxes, m=200)
    plain = transport.collective_helfand_moments(e, m=200)
    np.testing.assert_allclose(M[:, :, 0, 0], plain[:3], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(M, M.transpose(0, 1, 3, 2))


def test_schur_complement_of_averaged_runs(rng):
    """Averaging the matrices of several runs, then taking the complement, removes
    the species fluxes as for one long run."""
    n = 20000
    e0 = _ar1(rng, 0.9, (3, n))
    fluxes = _ar1(rng, 0.95, (2, 3, n))
    e = e0 + 3 * fluxes[0] - 2 * fluxes[1]
    runs = [
        transport.flux_helfand_matrices(
            e[:, i : i + 5000], fluxes[..., i : i + 5000], m=500
        )
        for i in range(0, n, 5000)
    ]
    kappa = transport.schur_complement(np.mean(runs, axis=0))
    plain = np.mean(
        [
            transport.collective_helfand_moments(e0[:, i : i + 5000], m=500)
            for i in range(0, n, 5000)
        ],
        axis=0,
    )
    slope = (kappa[:, -1] - kappa[:, 250]).mean() / 249
    expected = (plain[:3, -1] - plain[:3, 250]).mean() / 249
    assert slope == pytest.approx(expected, rel=0.15)
    # A single matrix, or a batch with leading dimensions
    assert transport.schur_complement(runs[0][0, 10]).shape == ()
    value, a = transport.schur_complement(runs[0], coefficients=True)
    assert value.shape == (3, 500) and a.shape == (3, 500, 2)
    assert np.all(a[:, 0] == 0.0)
