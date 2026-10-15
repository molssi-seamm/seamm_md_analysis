# -*- coding: utf-8 -*-

"""Tests of seamm_md_analysis.stats."""

import numpy as np
import pytest

from seamm_md_analysis import stats
from . import reference

CASES = [
    (1.23456, 0.0123),
    (123.456, 2.5),
    (0.000123, 4.5e-6),
    (5.0, 0.0),
    (5.0, -1.0),
    (5.0, float("nan")),
    (1.0e8, 3.0),
    ("text", 0.1),
    (-2.5, 0.37),
]


@pytest.mark.parametrize("value,err", CASES)
def test_fmt_err_matches_diffusivity(value, err):
    for as_float in (False, True):
        new = stats.fmt_err(value, err, as_float=as_float)
        old = reference.diff_fmt_err(value, err, as_float=as_float)
        assert new == old or (str(new) == str(old))


@pytest.mark.parametrize("value,err", CASES)
def test_fmt_err_matches_thermal_conductivity(value, err):
    new = stats.fmt_err(value, err, fallback_decimals=2)
    old = reference.tc_fmt_err(value, err)
    assert new == old


def test_mean_and_stderr():
    mean, se = stats.mean_and_stderr([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert se == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2)
    mean, se = stats.mean_and_stderr([[1.0, 10.0], [3.0, 30.0]])
    np.testing.assert_allclose(mean, [2.0, 20.0])
    mean, se = stats.mean_and_stderr([7.0])
    assert mean == 7.0 and np.isnan(se)


def test_block_average():
    rng = np.random.default_rng(3)
    x = rng.normal(loc=5.0, scale=2.0, size=100_003)
    mean, se, means = stats.block_average(x, n_blocks=20)
    assert means.shape == (20,)
    assert mean == pytest.approx(5.0, abs=0.03)
    assert se == pytest.approx(2.0 / np.sqrt(100_000), rel=0.5)
    with pytest.raises(ValueError):
        stats.block_average(x[:5], n_blocks=10)
    with pytest.raises(ValueError):
        stats.block_average(x, n_blocks=1)


def test_block_average_vector():
    x = np.arange(40.0).reshape(20, 2)
    mean, se, means = stats.block_average(x, n_blocks=4)
    np.testing.assert_allclose(mean, x.mean(axis=0))
    assert means.shape == (4, 2)


def _ar1(rng, n, phi):
    x = np.empty(n)
    x[0] = rng.normal()
    noise = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_statistical_inefficiency():
    rng = np.random.default_rng(7)
    assert stats.statistical_inefficiency(rng.normal(size=20_000)) == pytest.approx(
        1.0, abs=0.1
    )
    # For AR(1), g = (1 + phi) / (1 - phi) = 9 for phi = 0.8
    g = stats.statistical_inefficiency(_ar1(rng, 100_000, 0.8))
    assert g == pytest.approx(9.0, rel=0.2)
    mean, se = stats.mean_and_stderr_correlated(_ar1(rng, 100_000, 0.8))
    assert abs(mean) < 5 * se


def test_detect_equilibration(capsys):
    rng = np.random.default_rng(11)
    x = rng.normal(size=4000)
    x[:400] += np.linspace(20.0, 0.0, 400)
    t0, g, n_eff = stats.detect_equilibration(x)
    assert 200 < t0 < 600
    assert g >= 1.0 and n_eff > 1000
    # pymbar's banner is suppressed
    assert "JAX" not in capsys.readouterr().out
