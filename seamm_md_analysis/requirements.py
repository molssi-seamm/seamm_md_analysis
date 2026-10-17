# -*- coding: utf-8 -*-

"""What an analysis needs from the MD, and what an MD step will provide.

An MD step describes what a run will write to its H5MD file with
:class:`Provides`, from its parameters, before anything runs. An analysis step
describes what it needs with :class:`Requirements`. :func:`check` compares the two,
so that a missing or too coarse sampling is caught before the MD runs rather than
after it.

The names are those of the H5MD file: an observable such as "heat_flux" or
"pressure_tensor", or a per-particle property as "<group>/<name>", e.g.
"molecules/position" or "all/velocity".

Intervals are given in time, in MD steps, or both, since an MD step may know only
one of them before it runs (the timestep may depend on the structure).
"""

from dataclasses import dataclass, field

import pint


def _q(value):
    """A pint quantity from a string or quantity, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return pint.get_application_registry().Quantity(value)
    return value


@dataclass
class Interval:
    """A sampling interval, in time and/or in MD steps.

    Attributes
    ----------
    time : pint.Quantity or None
        The interval in time, if known.
    steps : int or None
        The interval in MD steps, if known.
    """

    time: object = None
    steps: int = None

    def __post_init__(self):
        self.time = _q(self.time)

    def __str__(self):
        if self.time is not None:
            return f"{self.time:~P}"
        if self.steps is not None:
            return f"{self.steps} steps"
        return "unknown"


@dataclass
class Need:
    """One thing an analysis needs.

    Attributes
    ----------
    names : [str]
        The acceptable names, in order of preference, e.g.
        ["molecules/position", "all/position"].
    max_time : pint.Quantity or str or None
        The longest acceptable sampling interval in time.
    max_steps : int or None
        The longest acceptable sampling interval in MD steps.
    why : str
        What it is needed for, for messages.
    sample_time : pint.Quantity or str or None
        The interval to use when an analysis turns the sampling on, if finer than
        the longest acceptable one; by default ``max_time``.
    """

    names: list
    max_time: object = None
    max_steps: int = None
    why: str = ""
    sample_time: object = None

    def __post_init__(self):
        if isinstance(self.names, str):
            self.names = [self.names]
        self.max_time = _q(self.max_time)
        self.sample_time = _q(self.sample_time)


@dataclass
class Requirements:
    """What an analysis needs from the MD.

    Attributes
    ----------
    needs : [Need]
        The data, each with its acceptable sampling.
    periodic : bool
        Whether the system must be periodic.
    dynamics : bool
        Whether the analysis uses the dynamics -- transport properties such as the
        viscosity, diffusion and conductivities -- so that a thermostat that
        perturbs the motion (Langevin) or none (NVE) should not be used.
    """

    needs: list = field(default_factory=list)
    periodic: bool = False
    dynamics: bool = False


@dataclass
class Provides:
    """What an MD run will write to its H5MD file.

    Attributes
    ----------
    data : {str: Interval}
        Each observable and per-particle property, with its sampling interval.
    timestep : pint.Quantity or None
        The timestep, if known before the run.
    step : str
        The MD step, for messages, e.g. "LAMMPS / NVT dynamics".
    """

    data: dict = field(default_factory=dict)
    timestep: object = None
    step: str = ""

    def __post_init__(self):
        self.timestep = _q(self.timestep)

    def interval(self, name):
        """The interval of a name, completed from the timestep if possible."""
        interval = self.data.get(name, None)
        if interval is None:
            return None
        time, steps = interval.time, interval.steps
        if self.timestep is not None:
            if time is None and steps is not None:
                time = steps * self.timestep
            elif steps is None and time is not None:
                steps = round((time / self.timestep).m_as(""))
        return Interval(time=time, steps=steps)


def check(requirements, provides):
    """What the MD will not provide that the analysis needs.

    A sampling interval that cannot be compared before the run (e.g. one given in
    steps when the timestep is not yet known) is not reported.

    Parameters
    ----------
    requirements : Requirements
        What the analysis needs.
    provides : Provides
        What the MD run will write.

    Returns
    -------
    [str]
        A description of each gap; empty if there are none.
    """
    problems = []
    where = f" in {provides.step}" if provides.step else ""
    for need in requirements.needs:
        why = f" for {need.why}" if need.why else ""
        available = [name for name in need.names if name in provides.data]
        if len(available) == 0:
            names = " or ".join(need.names)
            problems.append(f"{names} is not sampled{where}, but is needed{why}.")
            continue
        name = available[0]
        interval = provides.interval(name)
        if need.max_time is not None and interval.time is not None:
            if interval.time > need.max_time:
                problems.append(
                    f"{name} is sampled every {interval.time:~P}{where}, but must be "
                    f"sampled at least every {need.max_time:~P}{why}."
                )
        if need.max_steps is not None and interval.steps is not None:
            if interval.steps > need.max_steps:
                problems.append(
                    f"{name} is sampled every {interval.steps} steps{where}, but must "
                    f"be sampled at least every {need.max_steps} steps{why}."
                )
    return problems
