# -*- coding: utf-8 -*-

"""The state the MD runs sampled, and whether they can be treated as isotropic.

Every analysis step reports the state its results belong to -- temperature,
pressure, volume and density, each with its error over the runs -- because a
property measured at the wrong pressure or density belongs to that state, not the
one intended:

- an NPT run's mean density is the equilibrium density at its pressure;
- an NVT run's pressure is what it is; where the bulk modulus K is known (from the
  volume fluctuations of an NPT run, such as the equilibration before), the
  equilibrium density at the target pressure is estimated as
  ρ_eq ≈ ρ (1 + (P_target − P) / K).

A run is treated as isotropic -- one value per property from the average of its
components -- only when its cell is cubic and no fix makes it anisotropic (walls,
fields, deformation, flow, added forces).
"""

from dataclasses import dataclass, field

import numpy as np
from tabulate import tabulate

from .stats import Estimate, detect_equilibration

#: Boltzmann's constant in atm·Å³/K
KB_ATM_A3 = 1.380649e-23 / 101325.0 * 1.0e30
#: amu/Å³ in g/mL
AMU_PER_A3_IN_G_PER_ML = 1.66053906660

#: The fix styles (or their beginnings) that make a run anisotropic
ANISOTROPIC_FIXES = (
    "wall",
    "efield",
    "deform",
    "flow",
    "addforce",
    "aveforce",
    "setforce",
    "spring",
    "smd",
    "indent",
    "gravity",
    "move",
    "oneway",
)


# ----------------------------------------------------------------------
# Isotropy
# ----------------------------------------------------------------------
def is_cubic(cell, tolerance=1.0e-6):
    """Whether a cell (3×3, vectors as rows) is cubic."""
    if cell is None:
        return False
    cell = np.asarray(cell, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths <= 0):
        return False
    off = cell - np.diag(np.diag(cell))
    return bool(
        np.all(np.abs(off) <= tolerance * lengths.max())
        and np.ptp(lengths) <= tolerance * lengths.max()
    )


def isotropy(md):
    """Whether a run can be treated as isotropic, and why.

    Parameters
    ----------
    md : seamm_md_analysis.MDData
        The run.

    Returns
    -------
    (bool, str)
        Whether it is isotropic, and the reason, for the output.
    """
    if not md.periodic:
        return False, "the system is not periodic"
    if not is_cubic(md.cell()):
        return False, "the cell is not cubic"
    fixes = md.parameters.get("fixes")
    if fixes is None:
        return True, "the cell is cubic (the run did not record its fixes)"
    if isinstance(fixes, str):
        fixes = fixes.split()
    found = sorted(
        {f for f in fixes if any(str(f).startswith(a) for a in ANISOTROPIC_FIXES)}
    )
    if len(found) > 0:
        return False, f"the run used {', '.join(found)}"
    return True, "the cell is cubic and nothing in the run is directional"


# ----------------------------------------------------------------------
# The state of a run
# ----------------------------------------------------------------------
def _equilibrated_mean(values):
    """The mean of a series after its equilibration, found with pymbar."""
    values = np.asarray(values, dtype=float)
    if values.size < 10:
        return float(values.mean())
    try:
        t0, _, _ = detect_equilibration(values, nskip=max(1, values.size // 200))
    except Exception:
        t0 = 0
    return float(values[t0:].mean())


@dataclass
class RunState:
    """The mean state of one run.

    Attributes
    ----------
    ensemble : str
        "NVT", "NPT", "NVE", ...
    temperature, pressure, volume, density : float
        K, atm, Å³ and g/mL; None if not known.
    target_pressure : float
        The barostat's pressure for NPT, atm; None otherwise.
    """

    ensemble: str = ""
    temperature: float = None
    pressure: float = None
    volume: float = None
    density: float = None
    target_pressure: float = None


def _series(md, name, units):
    """An observable's values without a step-0 sample: LAMMPS's setup step, whose
    virial is not valid with constraints (files written before lammps-step
    2026.10.17.2 have it)."""
    series = md.observable(name, units=units)
    values = np.asarray(series.values, dtype=float)
    steps = np.asarray(series.steps)
    if len(steps) == len(values) and len(values) > 1 and steps[0] == 0:
        values = values[1:]
    return values


def run_state(md):
    """The mean temperature, pressure, volume and density of a run.

    The means are over the run after its equilibration. The volume and density of
    a run at constant volume come from its cell and masses.
    """
    P = md.parameters
    state = RunState(ensemble=str(P.get("ensemble", "")))
    if md.has_observable("temperature"):
        state.temperature = _equilibrated_mean(_series(md, "temperature", "K"))
    if md.has_observable("pressure"):
        state.pressure = _equilibrated_mean(_series(md, "pressure", "atm"))
    mass = None
    masses = md.masses()
    if masses is not None:
        mass = float(np.sum(masses))
    if md.has_observable("volume"):
        state.volume = _equilibrated_mean(_series(md, "volume", "Å^3"))
    elif md.periodic and md.cell() is not None:
        state.volume = float(abs(np.linalg.det(md.cell(units="Å"))))
    if md.has_observable("density"):
        state.density = _equilibrated_mean(_series(md, "density", "g/mL"))
    elif mass is not None and state.volume:
        state.density = mass / state.volume * AMU_PER_A3_IN_G_PER_ML
    if state.ensemble == "NPT" and "pressure" in P:
        state.target_pressure = _to_atm(P["pressure"], P.get("pressure_unit", "atm"))
    return state


def _to_atm(value, unit):
    import pint

    try:
        Q_ = pint.get_application_registry().Quantity
        return float(Q_(float(value), unit).m_as("atm"))
    except Exception:
        return None


def bulk_modulus(md):
    """The bulk modulus from the volume fluctuations of an NPT run, in atm.

    K = k_B T ⟨V⟩ / (⟨V²⟩ − ⟨V⟩²), over the run after its equilibration. Returns
    None for a run without a volume series.
    """
    if not md.has_observable("volume"):
        return None
    volume = np.asarray(_series(md, "volume", "Å^3"), dtype=float)
    if volume.size < 10:
        return None
    try:
        t0, _, _ = detect_equilibration(volume, nskip=max(1, volume.size // 200))
    except Exception:
        t0 = 0
    volume = volume[t0:]
    variance = float(volume.var(ddof=1))
    if variance <= 0:
        return None
    T = run_state(md).temperature
    if T is None:
        return None
    return KB_ATM_A3 * T * float(volume.mean()) / variance


# ----------------------------------------------------------------------
# The state over several runs
# ----------------------------------------------------------------------
@dataclass
class State:
    """The state of a set of runs, each quantity with its error over the runs.

    Attributes
    ----------
    ensemble : str
    temperature, pressure, volume, density : Estimate or None
        K, atm, Å³, g/mL.
    target_pressure : float
        The pressure aimed at, atm, if known.
    bulk_modulus : float
        K, atm, if known.
    equilibrium_density : Estimate or None
        For runs at constant volume, the estimated density at the target pressure.
    notes : [str]
        What the output should say about the state.
    """

    ensemble: str = ""
    temperature: Estimate = None
    pressure: Estimate = None
    volume: Estimate = None
    density: Estimate = None
    target_pressure: float = None
    bulk_modulus: float = None
    equilibrium_density: Estimate = None
    notes: list = field(default_factory=list)


def summarize_state(runs, target_pressure=None, bulk_modulus=None):
    """The state over runs, with the equilibrium density where it can be estimated.

    Parameters
    ----------
    runs : [RunState]
        The state of each run, from :func:`run_state`.
    target_pressure : float
        The pressure aimed at, atm: by default that of NPT runs, else 1 atm when a
        bulk modulus is given.
    bulk_modulus : float
        K, atm, for the equilibrium density of runs at constant volume, e.g. from
        :func:`bulk_modulus` of an NPT equilibration.

    Returns
    -------
    State
    """
    result = State()
    ensembles = sorted({r.ensemble for r in runs if r.ensemble})
    result.ensemble = "/".join(ensembles)
    for name in ("temperature", "pressure", "volume", "density"):
        values = [getattr(r, name) for r in runs if getattr(r, name) is not None]
        if len(values) > 0:
            units = {"temperature": "K", "pressure": "atm", "volume": "Å^3"}
            setattr(
                result,
                name,
                Estimate.from_samples(values, units=units.get(name, "g/mL")),
            )
    targets = [r.target_pressure for r in runs if r.target_pressure is not None]
    if target_pressure is None and len(targets) > 0:
        target_pressure = float(np.mean(targets))
    result.bulk_modulus = bulk_modulus

    constant_volume = all(r.ensemble != "NPT" for r in runs)
    if constant_volume and result.density is not None and result.pressure is not None:
        if bulk_modulus is None:
            result.notes.append(
                "The runs were at constant volume, so the pressure is what the "
                "density gives. The equilibrium density is not estimated: it needs "
                "the bulk modulus, e.g. from an NPT run earlier in the flowchart."
            )
        else:
            if target_pressure is None:
                target_pressure = 1.0
            rho, P = result.density, result.pressure
            factor = 1 + (target_pressure - P.value) / bulk_modulus
            stderr = None
            if P.stderr is not None:
                stderr = rho.value * P.stderr / bulk_modulus
            result.equilibrium_density = Estimate(
                rho.value * factor, stderr, P.n, "g/mL", P.source
            )
            result.notes.append(
                f"The equilibrium density at {target_pressure:.4g} atm is estimated "
                f"as ρ (1 + (P_target − P) / K), with K = {bulk_modulus:.4g} atm."
            )
    result.target_pressure = target_pressure
    if (
        result.equilibrium_density is not None
        and result.density is not None
        and abs(result.equilibrium_density.value - result.density.value)
        > 0.01 * result.density.value
    ):
        result.notes.append(
            "The density of the runs differs by more than 1% from the estimated "
            "equilibrium density: the results belong to the density that was run."
        )
    return result


def state_table(state):
    """The state as a table and notes, for the output.

    Returns
    -------
    str
        The table, with a line saying what the errors are, and the notes.
    """
    table = {"Quantity": [], "Value": [], "± (95%)": [], "Units": []}

    def add(label, estimate, units):
        if estimate is None:
            return
        v, e = estimate.formatted()
        table["Quantity"].append(label)
        table["Value"].append(v)
        table["± (95%)"].append(e)
        table["Units"].append(units)

    add("Temperature", state.temperature, "K")
    add("Pressure", state.pressure, "atm")
    add("Volume", state.volume, "Å^3")
    add("Density", state.density, "g/mL")
    add("Equilibrium density", state.equilibrium_density, "g/mL")
    if state.target_pressure is not None:
        table["Quantity"].append("Target pressure")
        table["Value"].append(f"{state.target_pressure:.4g}")
        table["± (95%)"].append("")
        table["Units"].append("atm")
    text = tabulate(
        table,
        headers="keys",
        tablefmt="rounded_outline",
        disable_numparse=True,
        colalign=("left", "decimal", "decimal", "left"),
    )
    title = f"The state ({state.ensemble})" if state.ensemble else "The state"
    width = len(text.splitlines()[0])
    lines = [title.center(width), text]
    first = next(
        (e for e in (state.temperature, state.pressure, state.density) if e), None
    )
    if first is not None:
        lines.append(first.describe_error())
    lines.extend(state.notes)
    return "\n".join(lines)
