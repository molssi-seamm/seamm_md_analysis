# -*- coding: utf-8 -*-

"""Support for SEAMM analysis steps that read MD runs.

An analysis step either runs its own MD, in a sub-flowchart, as replicas ("mode A"),
or analyses MD that ran earlier in the flowchart ("mode B"), found through the
workspace variable ``_md_runs``. These functions hold what both modes need and that
every analysis step shares: finding the MD sub-steps in a flowchart, checking that
they will sample what the analysis needs, turning sampling on, and finding the
files to read.

They use only the methods of SEAMM's nodes and flowcharts, so this library does not
depend on SEAMM. An MD sub-step is any node with an ``md_provides()`` method.
"""

from pathlib import Path
import re

import numpy as np

from .io import md_run_sources, select_md_runs

try:
    from seamm.problems import Edit, Fix, Problem
except ImportError:  # seamm before 2026.10.17: problems without fixes
    Edit = Fix = Problem = None
from .requirements import check

#: The choice by which an analysis step used to run its own MD, in a sub-flowchart;
#: no longer offered (phase 8): replicas are a Loop around the MD
run_own_md = "run the MD in the sub-flowchart"

#: The choice that reads the files given by the user
read_files = "files"

#: The default source: every run of the MD before the step, which in a Loop are its
#: iterations' runs, and otherwise the one run
each_iteration = "the last MD run of each loop iteration"

#: All choices of the source parameter
source_choices = (*md_run_sources, read_files)


def source_parameters(analysis="analysis"):
    """The parameter definitions an analysis step adds for its source of MD.

    Parameters
    ----------
    analysis : str
        What the step calculates, for the help text, e.g. "the diffusivity".

    Returns
    -------
    {str: dict}
        "md source", "md files" and "number of blocks".
    """
    return {
        "md source": {
            "default": each_iteration,
            "kind": "enum",
            "default_units": "",
            "enumeration": source_choices,
            "format_string": "s",
            "description": "MD to analyze:",
            "help_text": (
                f"Where the MD for {analysis} comes from: the last MD run of each "
                "iteration of the loop around the MD, as independent runs (or the "
                "one run, without a loop); the last MD run earlier in the "
                "flowchart; or the H5MD files (md.h5md) of earlier runs, including "
                "other jobs' (job://<n>/<path>)."
            ),
        },
        "md files": {
            "default": "",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "s",
            "description": "H5MD files:",
            "help_text": (
                "The md.h5md files to analyze, separated by spaces or commas. Each "
                "is an independent run. A relative path is in this job; "
                "job://<n>/<path> is in job n."
            ),
            "applies_when": {"md source": read_files},
        },
        "number of blocks": {
            "default": 5,
            "kind": "integer",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "d",
            "description": "Blocks for errors from one run:",
            "help_text": (
                "When a single MD run is analyzed, it is divided into this many "
                "consecutive blocks, which are analyzed separately to estimate the "
                "errors. Each block must be much longer than the correlation time."
            ),
        },
    }


def _main_line(flowchart):
    """The nodes of a flowchart in execution order, including loop bodies."""
    nodes = []
    seen = set()

    def walk(node):
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            nodes.append(node)
            following = None
            for edge in flowchart.edges(node, direction="out"):
                if edge.edge_type != "execution":
                    continue
                if edge.edge_subtype == "loop":
                    walk(edge.node2)
                elif edge.edge_subtype in ("next", "exit"):
                    following = edge.node2
            node = following

    walk(flowchart.get_node("1"))
    return nodes


def md_substeps(nodes):
    """The MD sub-steps among nodes and inside their sub-flowcharts, in order.

    Parameters
    ----------
    nodes : [seamm.Node] or seamm.Flowchart
        The nodes to search, or a flowchart to search in execution order.

    Returns
    -------
    [seamm.Node]
        The nodes that have ``md_provides``.
    """
    if hasattr(nodes, "edges"):
        nodes = _main_line(nodes)
    result = []
    for node in nodes:
        if hasattr(node, "md_provides"):
            result.append(node)
        subflowchart = getattr(node, "subflowchart", None)
        if subflowchart is not None and hasattr(subflowchart, "edges"):
            result.extend(md_substeps(subflowchart))
    return result


def check_md_flowchart(node, preceding, requirements, source):
    """Check, before anything runs, that the MD will provide what an analysis needs.

    Besides the sampling, it checks where the analysis is relative to the loop
    around the MD, and the thermostat. Where it can, a problem comes with a fix.

    Parameters
    ----------
    node : seamm.Node
        The analysis step.
    preceding : [seamm.Node]
        The steps that run before it, in order.
    requirements : seamm_md_analysis.Requirements
        What the analysis needs.
    source : str
        The value of its "md source" parameter.

    Returns
    -------
    [str]
        A description of each problem (a seamm.Problem where seamm has them);
        empty if none.
    """
    title = node.title
    if source == read_files:
        return []
    if source == run_own_md:
        return [
            f"'{title}' no longer runs its own MD. Put the MD before it -- in a Loop "
            "for independent runs -- and analyze 'the last MD run of each loop "
            "iteration'; the standard flowcharts for liquids and electrolytes show "
            "how."
        ]
    md = md_substeps(preceding)
    if len(md) == 0:
        return [
            f"'{title}' analyzes MD run earlier in the flowchart, but no MD step "
            "comes before it."
        ]
    # As when it runs, the analysis reads the last MD run that samples its first need
    target = md[-1]
    if len(requirements.needs) > 0:
        names = requirements.needs[0].names
        candidates = [
            n for n in md if any(name in n.md_provides().data for name in names)
        ]
        if len(candidates) > 0:
            target = candidates[-1]
    problems = _sampling_problems(title, requirements, target)
    problems.extend(_placement_problems(node, target, source))
    problems.extend(_thermostat_problems(title, requirements, target))
    return problems


# ----------------------------------------------------------------------
# Where the analysis is, relative to the loops around the MD
# ----------------------------------------------------------------------
def _enclosing_loops(flowchart):
    """The loops around each node of a flowchart, outermost first: {id: [Loop]}."""
    loops = {}
    seen = set()

    def walk(node, around):
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            loops[id(node)] = around
            following = None
            for edge in flowchart.edges(node, direction="out"):
                if edge.edge_type != "execution":
                    continue
                if edge.edge_subtype == "loop":
                    walk(edge.node2, around + [node])
                elif edge.edge_subtype in ("next", "exit"):
                    following = edge.node2
            node = following

    walk(flowchart.get_node("1"), [])
    return loops


def _top_node(node, flowchart):
    """The step in ``flowchart`` that holds ``node``, perhaps in a sub-flowchart."""
    while node is not None and getattr(node, "flowchart", None) is not flowchart:
        node = getattr(getattr(node, "flowchart", None), "parent", None)
    return node


def _static_value(node, key):
    """A parameter's value as given (an expression stays text), or None."""
    parameters = getattr(node, "parameters", None)
    if parameters is None or key not in parameters:
        return None
    return parameters[key].value


#: The parameters of MD sub-steps that set the temperature or pressure
STATE_PARAMETERS = ("T", "T0", "T1", "P", "P0", "P1", "temperature", "pressure")


def _loop_changes_state(loop, md_nodes):
    """Whether a loop's variables set the temperature or pressure of the MD."""
    names = []
    variable = _static_value(loop, "variable")
    if isinstance(variable, str) and variable.strip() != "":
        names.append(variable.strip())
    by_rows = "rows" in str(_static_value(loop, "type"))
    for md in md_nodes:
        for key in STATE_PARAMETERS:
            value = _static_value(md, key)
            if not isinstance(value, str) or "$" not in value:
                continue
            if by_rows or any(
                f"${name}" in value or "${" + name + "}" in value for name in names
            ):
                return True
    return False


def _placement_problems(node, target, source):
    """Where the analysis is, relative to the innermost loop around its MD."""
    flowchart = getattr(node, "flowchart", None)
    if flowchart is None or not hasattr(flowchart, "edges"):
        return []
    md_top = _top_node(target, flowchart)
    if md_top is None:
        return []
    loops = _enclosing_loops(flowchart)
    md_loops = loops.get(id(md_top), [])
    if len(md_loops) == 0:
        return []
    inner = md_loops[-1]
    here = loops.get(id(node), [])
    title = node.title
    loop = f"'{inner.title}'"
    problems = []

    def problem(text, fix_to, label):
        fixes = []
        if Fix is not None:
            fixes = [Fix(label, [Edit(node, "md source", fix_to)])]
            return Problem(text, severity="warning", fixes=fixes)
        return text

    if source == each_iteration and any(loop_ is inner for loop_ in here):
        problems.append(
            problem(
                f"'{title}' is inside the loop {loop} that holds its MD, so 'the last "
                "MD run of each loop iteration' would also take the runs of the "
                "earlier iterations. Analyze 'the last MD run' here, or move the "
                "analysis after the loop to analyze all its iterations together.",
                "the last MD run",
                f"Analyze the last MD run in '{title}'",
            )
        )
    elif source == each_iteration:
        md_nodes = md_substeps([md_top]) or [target]
        if _loop_changes_state(inner, md_nodes):
            problems.append(
                problem(
                    f"The loop {loop} changes the temperature or pressure of the MD, "
                    f"so its iterations are not replicas of one state, but '{title}' "
                    "analyzes them together. Put the analysis inside the loop, after "
                    "the MD, to analyze each state on its own.",
                    "the last MD run",
                    f"Analyze the last MD run in '{title}'",
                )
            )
    elif source == "the last MD run" and not any(loop_ is inner for loop_ in here):
        problems.append(
            problem(
                f"The MD is in the loop {loop}, but '{title}', after the loop, "
                "analyzes only the last iteration's run. To analyze every "
                "iteration's run together, as independent runs, analyze 'the last MD "
                "run of each loop iteration'.",
                each_iteration,
                f"Analyze the runs of every iteration in '{title}'",
            )
        )
    return problems


# ----------------------------------------------------------------------
# The thermostat
# ----------------------------------------------------------------------
#: Thermostats that do not give the canonical fluctuations
NOT_CANONICAL = ("Berendsen", "velocity rescaling")
#: Thermostats that perturb the motion of the particles
STOCHASTIC = ("Langevin", "canonical sampling, langevin dynamics (csld)")


def _thermostat_problems(title, requirements, target):
    """Whether the thermostat of the MD suits the analysis."""
    thermostat = _static_value(target, "thermostat")
    where = getattr(target, "title", "the MD")
    text = None
    if thermostat is None:
        name = f"{getattr(target, 'extension', '')} {where}"
        if not (requirements.dynamics and "NVE" in name):
            return []
        text = (
            f"'{title}' analyzes the dynamics of {where}, which has no thermostat "
            "(NVE): each run's temperature drifts from the target by several "
            "kelvin, which shows in the transport properties. Use NVT with a "
            "Nose-Hoover or CSVR thermostat."
        )
        return [text if Problem is None else Problem(text, severity="warning")]
    if not isinstance(thermostat, str) or thermostat.startswith("$"):
        return []
    if thermostat in NOT_CANONICAL:
        text = (
            f"The {thermostat} thermostat in {where} does not give the canonical "
            f"fluctuations that '{title}' analyzes."
        )
    elif requirements.dynamics and thermostat in STOCHASTIC:
        text = (
            f"The {thermostat} thermostat in {where} perturbs the motion of the "
            f"particles, which changes the dynamics '{title}' analyzes."
        )
    if text is None:
        return []
    text += " A Nose-Hoover or CSVR thermostat does not."
    if Problem is None:
        return [text]
    fix = Fix(
        f"Use the Nose-Hoover thermostat in {where}",
        [Edit(target, "thermostat", "Nose-Hoover")],
    )
    return [Problem(text, severity="warning", fixes=[fix])]


def _sampling_problems(title, requirements, target):
    """What the MD step will not sample, each with a fix where the step can give one.

    The fix sets the MD step's sampling parameters as ``request_sampling`` would
    when the analysis runs its own MD.
    """
    if Problem is None:  # seamm before 2026.10.17
        return [f"'{title}': {p}" for p in check(requirements, target.md_provides())]

    problems = []
    provides = target.md_provides()
    for need in requirements.needs:
        gaps = check(type(requirements)(needs=[need]), provides)
        if len(gaps) == 0:
            continue
        fixes = []
        if hasattr(target, "sampling_edits"):
            result = target.sampling_edits(need)
            if result is not None:
                edits, description = result
                fixes.append(
                    Fix(
                        f"Turn on {description}",
                        [Edit(target, p, value, units) for p, value, units in edits],
                    )
                )
        problems.extend(Problem(f"'{title}': {gap}", fixes=fixes) for gap in gaps)
    return problems


def request_sampling(flowchart, requirements):
    """Turn on, in the last MD sub-step of a flowchart, what an analysis needs.

    Parameters
    ----------
    flowchart : seamm.Flowchart
        The analysis step's sub-flowchart.
    requirements : seamm_md_analysis.Requirements
        What the analysis needs.

    Returns
    -------
    ([str], [str])
        What was turned on, and what is still missing.
    """
    md = md_substeps(flowchart)
    if len(md) == 0:
        return [], ["There is no MD step in the sub-flowchart."]
    node = md[-1]
    changes = []
    for need in requirements.needs:
        provides = node.md_provides()
        missing = check(type(requirements)(needs=[need]), provides)
        if len(missing) == 0:
            continue
        change = (
            node.request_sampling(need) if hasattr(node, "request_sampling") else None
        )
        if change is None:
            continue
        changes.append(change)
    remaining = check(requirements, node.md_provides())
    return changes, remaining


def md_inputs(node, P, name=None):
    """The H5MD files an analysis in mode B reads, one per independent run.

    Parameters
    ----------
    node : seamm.Node
        The analysis step, for the workspace variables and paths.
    P : dict
        Its parameters, as values.
    name : str = None
        For runs from ``_md_runs``, only those that sampled this observable or
        per-particle property.

    Returns
    -------
    [pathlib.Path]
    """
    source = P["md source"]
    if source == read_files:
        text = str(P["md files"]).strip()
        if text == "":
            raise RuntimeError(f"'{node.title}': no H5MD files were given.")
        names = [x for x in re.split(r"[\s,]+", text) if x != ""]
        return [node.file_path(x, read_only=True) for x in names]
    records = []
    if node.variable_exists("_md_runs"):
        records = node.get_variable("_md_runs")
    chosen = select_md_runs(records, source, name=name)
    if len(chosen) == 0:
        what = "" if name is None else f" that sampled {name}"
        raise RuntimeError(f"'{node.title}': there is no earlier MD run{what}.")
    return [Path(r["path"]) for r in chosen]


#: The workspace variable holding the result of the last Viscosity step
viscosity_variable = "_viscosity"


def publish_viscosity(node, value, stderr=None, T=None, method="Green-Kubo"):
    """Record a viscosity for the steps after this one, such as Diffusivity.

    Parameters
    ----------
    node : seamm.Node
        The step that calculated it.
    value : float
        The shear viscosity, in Pa s.
    stderr : float = None
        Its standard error, in Pa s.
    T : float = None
        The temperature, in K.
    method : str = "Green-Kubo"
        How it was calculated.

    Returns
    -------
    dict
        The record, also stored in the workspace variable ``_viscosity``.
    """
    record = {
        "value": float(value),
        "stderr": None if stderr is None else float(stderr),
        "units": "Pa*s",
        "T": None if T is None else float(T),
        "method": method,
        "step": node.title,
        "id": ".".join(str(i) for i in node._id),
    }
    node.set_variable(viscosity_variable, record)
    return record


def preceding_viscosity(node):
    """The viscosity published by the last Viscosity step run, or None.

    Parameters
    ----------
    node : seamm.Node
        The step asking, for the workspace variables.

    Returns
    -------
    dict or None
        The record written by :func:`publish_viscosity`.
    """
    if not node.variable_exists(viscosity_variable):
        return None
    return node.get_variable(viscosity_variable)


def npt_reference(node, paths):
    """The bulk modulus and target pressure from an NPT run before the analysed runs.

    The last NPT run recorded in ``_md_runs`` before the first of the analysed
    runs -- typically the equilibration before the production runs -- gives the
    bulk modulus from its volume fluctuations, and the pressure it was run at.

    Parameters
    ----------
    node : seamm.Node
        The analysis step, for the workspace variables.
    paths : [str or pathlib.Path]
        The md.h5md files of the analysed runs.

    Returns
    -------
    (float, float) or (None, None)
        The bulk modulus and the target pressure, in atm.
    """
    from .io import open_md
    from .state import bulk_modulus, run_state

    if not node.variable_exists("_md_runs"):
        return None, None
    records = list(node.get_variable("_md_runs"))
    analysed = {str(Path(p).resolve()) for p in paths}
    first = next(
        (
            i
            for i, r in enumerate(records)
            if str(Path(r["path"]).resolve()) in analysed
        ),
        len(records),
    )
    for record in reversed(records[:first]):
        if record.get("ensemble") != "NPT" or not Path(record["path"]).exists():
            continue
        try:
            with open_md(record["path"]) as md:
                K = bulk_modulus(md)
                target = run_state(md).target_pressure
        except Exception:
            continue
        if K is not None:
            return K, target
    return None, None


def state_of_runs(node, paths):
    """The state of the analysed runs: T, P, V and ρ with their errors, and the
    equilibrium density where an NPT run before them gives the bulk modulus.

    Parameters
    ----------
    node : seamm.Node
        The analysis step.
    paths : [str or pathlib.Path]
        The md.h5md files of the analysed runs.

    Returns
    -------
    seamm_md_analysis.State
    """
    from .io import open_md
    from .state import run_state, summarize_state

    runs = []
    for path in paths:
        with open_md(path) as md:
            runs.append(run_state(md))
    K, target = npt_reference(node, paths)
    return summarize_state(runs, target_pressure=target, bulk_modulus=K)


# ----------------------------------------------------------------------
# The size of the system
# ----------------------------------------------------------------------
def _widths(cell):
    """The perpendicular widths of a cell (vectors as rows)."""
    cell = np.asarray(cell, dtype=float)
    volume = abs(np.linalg.det(cell))
    return [
        volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ]


def _molecule_extent(xyz, atoms, neighbors, cell=None):
    """The largest distance between two atoms of a molecule, made whole by
    following its bonds: each atom is placed at the image nearest its neighbour,
    which works for a molecule of any length.

    Parameters
    ----------
    xyz : numpy.ndarray(n_atoms, 3)
        The Cartesian coordinates of all the atoms.
    atoms : [int]
        The molecule's atoms, as indices.
    neighbors : [[int]]
        The atoms bonded to each atom, as indices.
    cell : numpy.ndarray(3, 3)
        The cell vectors as rows, for a periodic system.
    """
    if len(atoms) < 2:
        return 0.0
    inverse = None if cell is None else np.linalg.inv(cell)
    placed = {atoms[0]: np.asarray(xyz[atoms[0]], dtype=float)}
    todo = [atoms[0]]
    while todo:
        i = todo.pop()
        for j in neighbors[i]:
            if j in placed:
                continue
            d = np.asarray(xyz[j], dtype=float) - xyz[i]
            if inverse is not None:
                fractional = d @ inverse
                d = (fractional - np.round(fractional)) @ cell
            placed[j] = placed[i] + d
            todo.append(j)
    points = np.array(list(placed.values()))
    diff = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((diff**2).sum(axis=-1)).max())


def size_warnings(configuration, min_molecules=200, widths_per_molecule=3.0):
    """Whether a system is large enough for liquid properties.

    Parameters
    ----------
    configuration : molsystem._Configuration
        The structure about to be simulated.
    min_molecules : int
        The fewest molecules for good statistics (~200-300).
    widths_per_molecule : float
        The narrowest width of the cell should be at least this many times the
        length of the longest molecule.

    Returns
    -------
    [str]
        Warnings for the output; empty if the system is large enough.
    """
    result = []
    molecules = configuration.find_molecules(as_indices=True)
    n = len(molecules)
    if n < min_molecules:
        result.append(
            f"The system has only {n} molecules: liquid properties need at least "
            f"about {min_molecules}, or they and their errors come out poorly."
        )
    if configuration.periodicity == 3:
        cell = np.asarray(configuration.cell.vectors(), dtype=float)
        width = min(_widths(cell))
        xyz = np.asarray(configuration.atoms.get_coordinates(fractionals=False))
        neighbors = configuration.bonded_neighbors(as_indices=True)
        longest = max(
            (_molecule_extent(xyz, m, neighbors, cell) for m in molecules), default=0
        )
        if longest > 0 and width < widths_per_molecule * longest:
            result.append(
                f"The cell is {width:.1f} Å wide, less than {widths_per_molecule:g} "
                f"times the longest molecule ({longest:.1f} Å): molecules interact "
                "with their own images."
            )
    return result
