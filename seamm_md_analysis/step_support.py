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

from .io import md_run_sources, select_md_runs
from .requirements import check

#: The choice of the analysis step's source parameter that runs its own MD
run_own_md = "run the MD in the sub-flowchart"

#: The choice that reads the files given by the user
read_files = "files"

#: All choices of the source parameter
source_choices = (run_own_md, *md_run_sources, read_files)


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
            "default": run_own_md,
            "kind": "enum",
            "default_units": "",
            "enumeration": source_choices,
            "format_string": "s",
            "description": "MD to analyze:",
            "help_text": (
                f"Where the MD for {analysis} comes from: run in this step's "
                "sub-flowchart a number of times as independent runs; the last MD "
                "run earlier in the flowchart; the last MD run of each iteration of "
                "a loop, as independent runs; or the H5MD files (md.h5md) of earlier "
                "runs, including other jobs' (job://<n>/<path>)."
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
            "applies_when": {"md source": {"not": run_own_md}},
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
        A description of each problem; empty if none.
    """
    title = node.title
    if source == read_files:
        return []
    if source == run_own_md:
        md = md_substeps(node.subflowchart)
        if len(md) == 0:
            return [f"'{title}' runs MD in its sub-flowchart, but it has no MD step."]
        # Missing sampling is turned on when the step runs, as far as possible
        return []
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
    return [f"'{title}': {p}" for p in check(requirements, target.md_provides())]


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
