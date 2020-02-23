
__author__ = 'njcs4'


from .dg_form import \
    DGFemTerm, DGFemSIPG, DGFemNIPG, DGFemBO, \
    hyper_tensor_product, hyper_tensor_T_product, \
    homogeneity_tensor, \
    dg_cross, dg_outer, \
    tensor_jump, tangent_jump, \
    normal_proj, tangential_proj, \
    DGClassicalSecondOrderDiscretisation, DGClassicalFourthOrderDiscretisation, \
    generate_default_sipg_penalty_term


from .fluxes import \
    LocalLaxFriedrichs, \
    HLLE, \
    Vijayasundaram


from .operators import \
    EllipticOperator, \
    HyperbolicOperator, \
    PoissonOperator, \
    CompressibleEulerOperator, CompressibleEulerOperatorEntropyFormulation, \
    CompressibleNavierStokesOperator, CompressibleNavierStokesOperatorEntropyFormulation, \
    MaxwellOperator, \
    SpacetimeBurgersOperator, \
    StokesOperator, \
    DGNeumannBC, DGDirichletBC, DGDirichletNormalBC, DGAdiabticWallBC, \
    DGFemTerm, DGFemCurlTerm, DGFemStokesTerm


# Compressible flow utility functions
from .aero import \
    conserved_variables, \
    flow_variables, \
    pressure, \
    enthalpy, \
    speed_of_sound, \
    effective_reynolds_number, \
    energy_density, \
    subsonic_inflow, \
    subsonic_outflow, \
    no_slip


# Utility for generating Nitsche boundary conditions
from .nitsche import NitscheBoundary, StokesNitscheBoundary


# Optional dolfin utility functions
try:
    from dolfin_dg.dolfin.tensors import force_zero_function_derivative

    # DWR highly experimental
    from dolfin_dg.dolfin.dwr import \
        NonlinearAPosterioriEstimator, \
        LinearAPosterioriEstimator, \
        dual

    from dolfin_dg.dolfin.mark import \
        FixedFractionMarker, FixedFractionMarkerParallel

except ImportError:
    pass
