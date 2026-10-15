# -*- coding: utf-8 -*-

"""Unit strings in H5MD files and their translation to and from pint.

H5MD writes a unit as space-separated factors, each a unit symbol with an optional
integer power, with negative powers for the denominator: ``kcal mol-1``,
``Angstrom ps-1``, ``g cm-3``. MDAnalysis accepts only its own spellings for
lengths, times, velocities and forces (``Angstrom``, ``Angstrom ps-1``,
``kcal mol-1 Angstrom-1``), so SEAMM writes those, and this module translates
between them and pint.

pint's application registry is used unless a registry is given, so that the
quantities mix with SEAMM's, which sets its registry as the application registry.
"""

import re

import pint

# H5MD (and MDAnalysis) spelling -> pint name, where they differ
_to_pint = {
    "Angstrom": "angstrom",
    "A": "angstrom",
    "e": "elementary_charge",
    "amu": "dalton",
    "u": "dalton",
}

# pint name -> H5MD spelling, where it differs from pint's symbol
_to_h5md = {
    "angstrom": "Angstrom",
    "elementary_charge": "e",
    "dalton": "amu",
    "unified_atomic_mass_unit": "amu",
    "standard_atmosphere": "atm",
}

_factor = re.compile(r"^([A-Za-z_]+)(-?[0-9]+)?$")


def _registry(ureg):
    return pint.get_application_registry() if ureg is None else ureg


def to_pint(unit, ureg=None):
    """The pint unit for an H5MD unit string.

    Parameters
    ----------
    unit : str or None
        The unit in H5MD syntax, e.g. "kcal mol-1". None, "" and "1" are
        dimensionless.
    ureg : pint.UnitRegistry = None
        The registry, by default pint's application registry.

    Returns
    -------
    pint.Unit
    """
    ureg = _registry(ureg)
    if isinstance(unit, bytes):
        unit = unit.decode()
    if unit is None or unit.strip() in ("", "1"):
        return ureg.Unit("dimensionless")
    terms = []
    for token in unit.split():
        match = _factor.match(token)
        if match is None:
            raise ValueError(f"Cannot understand the unit '{token}' in '{unit}'")
        name, power = match.groups()
        name = _to_pint.get(name, name)
        terms.append(name if power is None else f"{name} ** {power}")
    return ureg.Unit(" * ".join(terms))


def to_h5md(unit, ureg=None):
    """The H5MD unit string for a pint unit or unit expression.

    Factors with positive powers come first, then the negative ones, with
    "mol" first among those, so that forces are written as
    "kcal mol-1 Angstrom-1" as MDAnalysis expects.

    Parameters
    ----------
    unit : str, pint.Unit or pint.Quantity
        The unit, e.g. "Å/fs" or ureg.kcal / ureg.mol.
    ureg : pint.UnitRegistry = None
        The registry, by default pint's application registry.

    Returns
    -------
    str
        The unit in H5MD syntax, or "" for a dimensionless unit.
    """
    ureg = _registry(ureg)
    if isinstance(unit, str):
        unit = ureg.Unit(unit)
    elif isinstance(unit, pint.Quantity):
        unit = unit.units
    positive = []
    negative = []
    for name, power in unit._units.items():
        symbol = _to_h5md.get(name, None)
        if symbol is None:
            symbol = ureg.get_symbol(name)
        if power != int(power):
            raise ValueError(f"H5MD units need integer powers: {unit}")
        power = int(power)
        if power > 0:
            positive.append(symbol if power == 1 else f"{symbol}{power}")
        elif power < 0:
            negative.append((0 if symbol == "mol" else 1, f"{symbol}{power}"))
    negative = [text for _, text in sorted(negative, key=lambda x: x[0])]
    return " ".join(positive + negative)
