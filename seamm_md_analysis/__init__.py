# -*- coding: utf-8 -*-

"""Analysis of molecular dynamics output for SEAMM.

Time-series statistics, correlation functions, Green–Kubo and Helfand
transport coefficients, plots, SEAMM's H5MD files, and readers for older MD
output.
"""

from . import correlate, h5md, io, legacy, plots, stats, transport, units  # noqa: F401
from .correlate import ccovf, tensor_correlation_functions  # noqa: F401
from .h5md import H5MDWriter, standard_observables  # noqa: F401
from .io import MDData, TimeSeries, open_md  # noqa: F401
from .legacy import (  # noqa: F401
    iter_dump_frames,
    iter_vector_frames,
    lammps_box_to_cell,
    read_dump_trajectory,
    read_trajectory,
    read_vector_trajectory,
)
from .stats import (  # noqa: F401
    block_average,
    detect_equilibration,
    fmt_err,
    mean_and_stderr,
    mean_and_stderr_correlated,
    statistical_inefficiency,
)
from .transport import (  # noqa: F401
    GreenKuboFit,
    LinearFit,
    collective_helfand_moments,
    fit_green_kubo_integral,
    fit_helfand_moment,
    fit_slope,
    helfand_slope,
    msd_by_species,
    species_helfand_moments,
    yeh_hummer_correction,
)

from ._version import __version__  # noqa: F401
