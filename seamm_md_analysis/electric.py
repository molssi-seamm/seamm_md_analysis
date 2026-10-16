# -*- coding: utf-8 -*-

"""Electrical properties from equilibrium MD: the static dielectric constant and the
ionic conductivity.

Both are computed from the trajectory -- unwrapped positions, so molecules stay
whole, and fixed (forcefield) charges:

- the dielectric constant from the fluctuations of the dipole of the cell, for
  Ewald or PPPM electrostatics with conducting ("tin-foil") boundary conditions:
  ε = 1 + (⟨M²⟩ − ⟨M⟩²) / (3 V ε0 kB T). For an electrolyte, M is the dipole of
  the neutral molecules only: the ions carry a current, not a bounded dipole.
- the ionic conductivity from the collective displacement of the ions' charges
  (Einstein–Helfand), σ = lim ⟨‖P(t) − P(0)‖²⟩ / (6 V kB t), P = Σ qᵢ rᵢ over the
  ions; and the Nernst–Einstein estimate from the ions' self-diffusion,
  σ_NE = e² / (V kB T) Σ N_s z_s² D_s. Their ratio is the ionicity.

Positions are in Å, charges in e, volumes in Å³, temperatures in K and times in
ps; results are in SI units (S/m for conductivities, m²/s for diffusion
coefficients).
"""

import math

import numpy as np

from .correlate import autocorrelation_functions
from .transport import fit_slope, squared_displacement_sums

#: The elementary charge, C
E = 1.602176634e-19
#: The vacuum permittivity, F/m
EPS0 = 8.8541878128e-12
#: The Boltzmann constant, J/K
KB = 1.380649e-23


# ----------------------------------------------------------------------
# Molecules and charges
# ----------------------------------------------------------------------
def classify_molecules(charges, molecules, tolerance=0.05):
    """Sort the molecules into neutral ones, ions, and others by their total charge.

    Parameters
    ----------
    charges : array_like
        The charge of each atom, e.
    molecules : [[int]]
        The atoms of each molecule.
    tolerance : float
        How far from an integer a total charge may be.

    Returns
    -------
    dict
        ``neutral``, ``ions`` and ``other``: lists of molecule indices;
        ``charge``: the total charge of each molecule; ``z``: the nearest
        integer charge of each.
    """
    charges = np.asarray(charges, dtype=float)
    total = np.array([charges[list(m)].sum() for m in molecules])
    z = np.rint(total).astype(int)
    integral = np.abs(total - z) <= tolerance
    result = {"neutral": [], "ions": [], "other": [], "charge": total, "z": z}
    for i, (ok, zi) in enumerate(zip(integral, z)):
        if not ok:
            result["other"].append(i)
        elif zi == 0:
            result["neutral"].append(i)
        else:
            result["ions"].append(i)
    return result


def cell_dipole(positions, atoms, charges):
    """The dipole Σ qᵢ rᵢ of a set of atoms, e·Å.

    Parameters
    ----------
    positions : numpy.ndarray(n_atoms, 3)
        All the (unwrapped) positions of one frame, Å.
    atoms : numpy.ndarray of int
        The atoms to include -- whole molecules, so that the dipole of neutral
        molecules does not depend on the origin.
    charges : numpy.ndarray
        The charges of those atoms, e.
    """
    return np.asarray(charges) @ np.asarray(positions)[atoms]


# ----------------------------------------------------------------------
# The dielectric constant
# ----------------------------------------------------------------------
def _factor(volume, temperature):
    """(e Å)² -> the dimensionless 1 / (3 V ε0 kB T) term."""
    return (E * 1.0e-10) ** 2 / (3 * volume * 1.0e-30 * EPS0 * KB * temperature)


def _epsilon(M, volume, temperature):
    M = np.asarray(M, dtype=float)
    variance = float(np.mean(np.sum(M * M, axis=1)) - np.sum(M.mean(axis=0) ** 2))
    return 1.0 + _factor(volume, temperature) * variance, variance


def dielectric_constant(M, volume, temperature, n_blocks=5, n_running=100):
    """The static dielectric constant from the dipole of the cell over time.

    Parameters
    ----------
    M : numpy.ndarray(n, 3)
        The dipole of the cell in each frame, e·Å.
    volume : float
        The (mean) volume of the cell, Å³.
    temperature : float
        The mean temperature, K.
    n_blocks : int
        Consecutive blocks for the standard error; with fewer than 2, none.
    n_running : int
        The number of points of the running estimate.

    Returns
    -------
    dict
        ``epsilon`` and ``stderr`` (None without blocks), ``variance`` (e² Å²),
        ``mean_dipole`` (e·Å), ``running`` -- the estimate from the first k
        frames, as ``(frame counts, values)`` -- and ``n_blocks``.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if n < 2:
        raise ValueError("At least two frames are needed for the fluctuations.")
    eps, variance = _epsilon(M, volume, temperature)
    stderr = None
    n_blocks = min(int(n_blocks), n // 2)
    if n_blocks >= 2:
        values = [
            _epsilon(block, volume, temperature)[0]
            for block in np.array_split(M, n_blocks)
        ]
        stderr = float(np.std(values, ddof=1) / math.sqrt(n_blocks))
    counts = np.unique(np.linspace(2, n, min(n_running, n - 1)).astype(int))
    running = [_epsilon(M[:k], volume, temperature)[0] for k in counts]
    return {
        "epsilon": float(eps),
        "stderr": stderr,
        "variance": variance,
        "mean_dipole": M.mean(axis=0).tolist(),
        "running": (counts.tolist(), running),
        "n_blocks": n_blocks if n_blocks >= 2 else 1,
    }


def dipole_correlation(M, dt, m=None):
    """The normalized autocorrelation of the dipole of the cell and its time.

    Parameters
    ----------
    M : numpy.ndarray(n, 3)
        The dipole in each frame.
    dt : float
        The time between frames, ps.
    m : int
        The lags to keep; a quarter of the frames by default.

    Returns
    -------
    dict
        ``times`` (ps), ``acf`` (the average over x, y and z, normalized) and
        ``tau``, its integral up to the first zero (ps).
    """
    M = np.asarray(M, dtype=float)
    if m is None:
        m = max(2, M.shape[0] // 4)
    acf, _ = autocorrelation_functions(M.T, m=m)
    c = acf.mean(axis=0)
    if c[0] <= 0.0:
        raise ValueError("The dipole does not fluctuate.")
    c = c / c[0]
    zero = int(np.argmax(c <= 0.0)) if np.any(c <= 0.0) else c.size
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2
    tau = float(trapezoid(c[:zero]) * dt) if zero > 1 else 0.0
    return {"times": (np.arange(m) * dt).tolist(), "acf": c.tolist(), "tau": tau}


# ----------------------------------------------------------------------
# The ionic conductivity
# ----------------------------------------------------------------------
def collective_msd(P):
    """The mean square displacement of the collective charge displacement.

    Parameters
    ----------
    P : numpy.ndarray(n, 3)
        Σ qᵢ rᵢ over the ions in each frame, e·Å (unwrapped positions).

    Returns
    -------
    numpy.ndarray(n)
        ⟨‖P(t) − P(0)‖²⟩ over time origins, e² Å², for lags 0 .. n-1.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    S = squared_displacement_sums(P).sum(axis=1)
    return S / (n - np.arange(n))


def conductivity_from_msd(msd, dt, volume, temperature, start=0.1, end=0.5):
    """The Einstein–Helfand conductivity from the collective MSD.

    The slope is fitted between ``start`` and ``end`` of the curve (fractions):
    the late part has few time origins and is noisy.

    Returns
    -------
    dict
        ``sigma`` and ``stderr`` (S/m, the error of the fit), ``slope``
        (e² Å² / ps) and the fitted line (``xs``, ``ys``).
    """
    msd = np.asarray(msd, dtype=float)
    ts = np.arange(msd.size) * dt
    fit = fit_slope(msd, ts.tolist(), start=start, end=end)
    scale = (E**2 * 1.0e-20 / 1.0e-12) / (6 * volume * 1.0e-30 * KB * temperature)
    return {
        "sigma": float(fit.slope * scale),
        "stderr": None if fit.stderr is None else float(fit.stderr * scale),
        "slope": float(fit.slope),
        "xs": list(fit.xs),
        "ys": list(fit.ys),
    }


def nernst_einstein(diffusion, counts, charges, volume, temperature):
    """The Nernst–Einstein conductivity, S/m.

    Parameters
    ----------
    diffusion : {str: float}
        The self-diffusion coefficient of each ionic species, m²/s.
    counts : {str: int}
        The number of each.
    charges : {str: int}
        The charge of each, in e.
    """
    total = sum(
        counts[s] * charges[s] ** 2 * diffusion[s] for s in diffusion if s in counts
    )
    return E**2 / (volume * 1.0e-30 * KB * temperature) * total
