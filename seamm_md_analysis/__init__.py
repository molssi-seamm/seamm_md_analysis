# -*- coding: utf-8 -*-

"""Analysis of molecular dynamics output for SEAMM.

Time-series statistics, correlation functions, Green–Kubo and Helfand
transport coefficients, plots, SEAMM's H5MD files, and readers for older MD
output.
"""

from . import (  # noqa: F401
    correlate,
    electric,
    h5md,
    io,
    legacy,
    plots,
    requirements,
    state,
    stats,
    step_support,
    transport,
    units,
)
from .correlate import (  # noqa: F401
    autocorrelation_functions,
    ccovf,
    tensor_correlation_functions,
)
from .h5md import H5MDWriter, standard_observables  # noqa: F401
from .io import (  # noqa: F401
    MDData,
    TimeSeries,
    md_run_sources,
    open_md,
    select_md_runs,
)
from .requirements import Interval, Need, Provides, Requirements  # noqa: F401
from .legacy import (  # noqa: F401
    iter_dump_frames,
    iter_vector_frames,
    lammps_box_to_cell,
    read_dump_trajectory,
    read_trajectory,
    read_vector_trajectory,
)
from .state import (  # noqa: F401
    RunState,
    State,
    bulk_modulus,
    is_cubic,
    isotropy,
    run_state,
    summarize_state,
    state_table,
)
from .stats import (  # noqa: F401
    ComponentCheck,
    Estimate,
    block_average,
    component_check,
    detect_equilibration,
    fmt_err,
    isotropic_average,
    mean_and_stderr,
    mean_and_stderr_correlated,
    statistical_inefficiency,
    t_factor,
)
from .transport import (  # noqa: F401
    SHEAR_COMPONENTS,
    GreenKuboFit,
    LinearFit,
    collective_helfand_moments,
    component_helfand_moments,
    effective_correlation_functions,
    fit_green_kubo_integral,
    fit_helfand_moment,
    fit_slope,
    flux_correlation_matrices,
    flux_helfand_matrices,
    helfand_slope,
    msd_by_species,
    multicomponent_correlation_functions,
    multicomponent_helfand_moments,
    schur_complement,
    shear_stress_components,
    species_helfand_moments,
    yeh_hummer_correction,
)

from ._version import __version__  # noqa: F401
