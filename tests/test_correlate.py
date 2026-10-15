# -*- coding: utf-8 -*-

"""Tests of seamm_md_analysis.correlate."""

import numpy as np
import pytest

from seamm_md_analysis import correlate


def _direct_ccovf(x, y):
    n = len(x)
    x = x - x.mean()
    y = y - y.mean()
    return np.array([np.sum(x[k:] * y[: n - k]) / (n - k) for k in range(n)])


def test_ccovf_direct():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300) + 2.0
    y = 0.5 * x + rng.normal(size=300)
    np.testing.assert_allclose(correlate.ccovf(x, y), _direct_ccovf(x, y), atol=1e-12)
    np.testing.assert_allclose(
        correlate.ccovf(x, y, adjusted=False),
        _direct_ccovf(x, y) * np.arange(300, 0, -1) / 300,
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        correlate.ccovf(x, y[:-1])


def test_ccovf_matches_statsmodels():
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(2)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    np.testing.assert_allclose(
        correlate.ccovf(x, y), stattools.ccovf(x, y), rtol=1e-12, atol=1e-14
    )


def test_tensor_correlation_functions():
    rng = np.random.default_rng(4)
    J = rng.normal(size=(3, 500))
    ccf, integral = correlate.tensor_correlation_functions(J, m=100)
    assert ccf.shape == (6, 100) and integral.shape == (6, 100)
    np.testing.assert_allclose(ccf[3], correlate.ccovf(J[0], J[1])[:100])
    np.testing.assert_allclose(ccf[0, 0], J[0].var())
    assert integral[0, 0] == 0.0
    np.testing.assert_allclose(integral[2, 1], 0.5 * (ccf[2, 0] + ccf[2, 1]))
    ccf, integral = correlate.tensor_correlation_functions(J)
    assert ccf.shape == (6, 500)


def test_errors_and_moving_average():
    acf = np.array([1.0, 0.5, 0.25, 0.125])
    err = correlate.acf_err(acf)
    assert err[0] == 0
    assert err[1] == pytest.approx(0.5)
    assert err[3] == pytest.approx(np.sqrt((1 + 2 * (0.25 + 0.0625)) / 4))
    np.testing.assert_allclose(correlate.ccf_err(acf, acf), err)
    np.testing.assert_allclose(
        correlate.moving_average(np.arange(6.0), 3), [1.0, 2.0, 3.0, 4.0]
    )
