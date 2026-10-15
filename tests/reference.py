# -*- coding: utf-8 -*-

"""The original implementations from diffusivity_step and
thermal_conductivity_step, as references for the tests."""

from math import ceil, log10

import numpy as np
from scipy.integrate import cumulative_trapezoid


def tc_create_helfand_moments(J, m=None):
    """thermal_conductivity_step.analysis.create_helfand_moments"""
    n = J.shape[1]
    if m is None:
        m = min(n // 20, 10000)
    M = np.zeros((6, m))
    for i in range(n - m):
        Ix = cumulative_trapezoid(J[0, i : m + i], initial=0.0)
        Iy = cumulative_trapezoid(J[1, i : m + i], initial=0.0)
        Iz = cumulative_trapezoid(J[2, i : m + i], initial=0.0)
        M[0, :] += Ix * Ix
        M[1, :] += Iy * Iy
        M[2, :] += Iz * Iz
        M[3, :] += Ix * Iy
        M[4, :] += Ix * Iz
        M[5, :] += Iy * Iz
    M /= n - m
    return M


def diff_create_helfand_moments(v, species, m=None):
    """diffusivity_step.analysis.create_helfand_moments"""
    n, nmols, _ = v.shape
    if m is None:
        m = min(n // 2, 5000)
    Ms = []
    M_errs = []
    for molecules in species:
        M = np.zeros((m, 4))
        M_err = np.zeros((m, 4))
        for alpha in range(3):
            tmp = np.zeros((m, len(molecules)))
            for i in range(n - m):
                integral = cumulative_trapezoid(
                    v[i : m + i, molecules, alpha], initial=0.0, axis=0
                )
                tmp += integral * integral
            M[:, alpha] = np.average(tmp, axis=1)
            M_err[:, alpha] = np.std(tmp, axis=1)
        vsum = v[:, molecules, 0] + v[:, molecules, 1] + v[:, molecules, 2]
        tmp = np.zeros((m, len(molecules)))
        for i in range(n - m):
            integral = cumulative_trapezoid(vsum[i : m + i, :], initial=0.0, axis=0)
            tmp += integral * integral
        M[:, 3] = np.average(tmp, axis=1)
        M_err[:, 3] = np.std(tmp, axis=1)
        M /= n - m
        M_err /= n - m
        Ms.append(M)
        M_errs.append(M_err)
    return Ms, M_errs


def diff_compute_msd(xyz, species, average=True):
    """diffusivity_step.analysis.compute_msd"""
    nsteps, nmolecules, _ = xyz.shape
    tmp = np.zeros_like(xyz)
    for i in range(nsteps):
        j = nsteps - i
        tmp[:j] += (xyz[i:] - xyz[i]) ** 2
    msd = []
    err = []
    for i, molecules in enumerate(species):
        msd.append(np.average(tmp[:, molecules, :], axis=1))
        err.append(np.std(tmp[:, molecules, :], axis=1))
        if average:
            tmp_ave = np.sum(tmp, axis=2)
            msd_ave = np.average(tmp_ave[:, molecules], axis=1)
            err_ave = np.std(tmp_ave[:, molecules], axis=1)
            msd_ave = msd_ave.reshape((nsteps, 1))
            err_ave = err_ave.reshape((nsteps, 1))
            msd[i] = np.concatenate((msd[i], msd_ave), axis=1)
            err[i] = np.concatenate((err[i], err_ave), axis=1)
        norm = np.arange(nsteps, 0, -1)
        norm = norm.reshape((nsteps, 1))
        msd[i] /= norm
        err[i] /= norm
    return msd, err


def diff_fmt_err(value, err, precision=2, as_float=False):
    """diffusivity_step.diffusivity.fmt_err"""
    try:
        decimals = -ceil(log10(err)) + precision
    except Exception:
        e = "--"
        try:
            v = f"{value:.3f}"
            if as_float:
                v = round(float(v), 3)
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


def tc_fmt_err(value, err, precision=2):
    """thermal_conductivity_step.thermal_conductivity.fmt_err"""
    try:
        decimals = -ceil(log10(err)) + precision
    except Exception:
        e = "--"
        try:
            v = f"{value:.2f}"
        except Exception:
            v = value
    else:
        if decimals < 0:
            decimals = 0
        fmt = f".{decimals}f"
        e = f"{err:{fmt}}"
        try:
            v = f"{value:{fmt}}"
        except Exception:
            v = value
    return v, e
