# -*- coding: utf-8 -*-

"""Tests of the dielectric constant and ionic conductivity
(seamm_md_analysis.electric) on synthetic systems with known answers."""

import math

import numpy as np
import pytest

from seamm_md_analysis.electric import (
    E,
    EPS0,
    KB,
    cell_dipole,
    classify_molecules,
    collective_msd,
    conductivity_from_msd,
    dielectric_constant,
    dipole_correlation,
    nernst_einstein,
)


def test_classify_molecules():
    charges = [-0.8476, 0.4238, 0.4238, 1.0, 0.8, -0.45, -0.45, -0.45, -0.45, 0.5]
    molecules = [[0, 1, 2], [3], [4, 5, 6, 7, 8], [9]]
    c = classify_molecules(charges, molecules)
    assert c["neutral"] == [0] and c["ions"] == [1, 2] and c["other"] == [3]
    assert c["z"].tolist() == [0, 1, -1, 0]
    assert c["charge"][2] == pytest.approx(-1.0)


def test_classify_scaled_ions():
    """With the charges of the ions scaled by 0.8, they are still ions of formal
    charge +-1, not molecules with a broken charge."""
    charges = [-0.8476, 0.4238, 0.4238, 0.8, 0.72, -0.38, -0.38, -0.38, -0.38]
    molecules = [[0, 1, 2], [3], [4, 5, 6, 7, 8]]
    plain = classify_molecules(charges, molecules)
    assert plain["ions"] == [] and plain["other"] == [1, 2]
    c = classify_molecules(charges, molecules, scale=0.8)
    assert c["neutral"] == [0] and c["ions"] == [1, 2] and c["other"] == []
    assert c["z"].tolist() == [0, 1, -1]
    assert c["charge"][1] == pytest.approx(0.8)


def test_cell_dipole_of_a_neutral_molecule_does_not_depend_on_the_origin():
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    q = np.array([-0.8, 0.4, 0.4])
    atoms = np.arange(3)
    M = cell_dipole(x, atoms, q)
    assert M.tolist() == pytest.approx([0.4, 0.4, 0.0])
    assert cell_dipole(x + 7.0, atoms, q).tolist() == pytest.approx(M.tolist())


def test_dielectric_constant_of_random_dipoles():
    """N dipoles of moment μ, independently oriented each frame: ⟨M²⟩ = N μ²."""
    rng = np.random.default_rng(1)
    N, mu, n = 200, 0.5, 4000
    V, T = 8000.0, 300.0
    u = rng.normal(size=(n, N, 3))
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    M = mu * u.sum(axis=1)
    expected = 1 + (E * 1e-10) ** 2 * N * mu**2 / (3 * V * 1e-30 * EPS0 * KB * T)
    r = dielectric_constant(M, V, T, n_blocks=5)
    assert r["epsilon"] == pytest.approx(expected, rel=0.05)
    assert abs(r["epsilon"] - expected) < 4 * r["stderr"]
    counts, values = r["running"]
    assert counts[-1] == n and values[-1] == pytest.approx(r["epsilon"])
    with pytest.raises(ValueError):
        dielectric_constant(M[:1], V, T)


def test_dipole_correlation_time():
    """An AR(1) dipole, φ = 0.9 per 0.1 ps: τ = 0.1 (1/(1-φ) - 1/2) ps."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(2)
    phi, n = 0.9, 200_000
    noise = rng.normal(scale=math.sqrt(1 - phi**2), size=(3, n))
    M = lfilter([1.0], [1.0, -phi], noise, axis=1).T
    r = dipole_correlation(M, 0.1, m=500)
    assert r["acf"][0] == pytest.approx(1.0)
    assert r["acf"][1] == pytest.approx(phi, abs=0.01)
    assert r["tau"] == pytest.approx(0.1 * (1 / (1 - phi) - 0.5), rel=0.1)


def _walks(rng, n_frames, n_ions, step):
    """Independent 3-D random walks, Å."""
    return np.cumsum(rng.normal(scale=step, size=(n_frames, n_ions, 3)), axis=0)


def test_independent_ions_give_the_nernst_einstein_conductivity():
    """Uncorrelated ions: σ = σ_NE, ionicity 1. The collective displacement of one
    run is a single random walk, so its MSD is noisy: averaged over 60 runs the
    ratio σ/σ_NE ranged 0.77-1.16 over eight seeds."""
    rng = np.random.default_rng(3)
    n_runs, n_frames, n_pairs, step, dt = 60, 1000, 20, 0.2, 1.0  # Å/frame, ps
    V, T = 30000.0, 300.0
    msds = []
    for _ in range(n_runs):
        cations = _walks(rng, n_frames, n_pairs, step)
        anions = _walks(rng, n_frames, n_pairs, step)
        msds.append(collective_msd(cations.sum(axis=1) - anions.sum(axis=1)))
    sigma = conductivity_from_msd(np.mean(msds, axis=0), dt, V, T)
    D = step**2 / (2 * dt) * 1e-8  # Å²/ps -> m²/s
    ne = nernst_einstein(
        {"Na": D, "Cl": D}, {"Na": n_pairs, "Cl": n_pairs}, {"Na": 1, "Cl": -1}, V, T
    )
    expected = E**2 / (V * 1e-30 * KB * T) * 2 * n_pairs * D
    assert ne == pytest.approx(expected)
    assert sigma["sigma"] == pytest.approx(ne, rel=0.25)


def test_paired_ions_do_not_conduct():
    """Each anion moves with its cation: the charge does not move."""
    rng = np.random.default_rng(4)
    cations = _walks(rng, 2000, 20, 0.2)
    anions = cations + np.array([2.5, 0.0, 0.0])
    P = cations.sum(axis=1) - anions.sum(axis=1)
    msd = collective_msd(P)
    assert np.allclose(msd, 0.0, atol=1e-6)
    assert conductivity_from_msd(msd, 1.0, 30000.0, 300.0)["sigma"] == pytest.approx(
        0.0, abs=1e-9
    )
