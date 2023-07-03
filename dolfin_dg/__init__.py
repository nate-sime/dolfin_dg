from .aero import (
    conserved_variables,
    flow_variables,
    pressure,
    enthalpy,
    speed_of_sound,
    effective_reynolds_number,
    energy_density,
    subsonic_inflow,
    subsonic_outflow,
    no_slip
)
from .block import (
    extract_rows, extract_blocks, extract_block_linear_system,
    derivative_block
)
from .math import normal_proj, tangential_proj, hyper_tensor_product, \
    hyper_tensor_T_product, dg_outer, homogeneity_tensor
from .dg_form import (
    DGFemTerm, DGFemSIPG, DGFemNIPG, DGFemBO,
    DGFemCurlTerm, DGFemStokesTerm,
    DGClassicalSecondOrderDiscretisation,
    DGClassicalFourthOrderDiscretisation,
    generate_default_sipg_penalty_term
)
from .fluxes import (
    LocalLaxFriedrichs,
    HLLE,
    Vijayasundaram
)
from .nitsche import NitscheBoundary, StokesNitscheBoundary
from .operators import (
    EllipticOperator,
    HyperbolicOperator,
    PoissonOperator,
    CompressibleEulerOperator,
    CompressibleNavierStokesOperator,
    MaxwellOperator,
    SpacetimeBurgersOperator,
    StokesOperator,
    DGNeumannBC, DGDirichletBC, DGDirichletNormalBC, DGAdiabticWallBC
)
