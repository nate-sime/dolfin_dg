
__author__ = 'njcs4'

from .dg_form import \
    DGFemViscousTerm, DGFemSIPG, DGFemNIPG, DGFemBO, \
    hyper_tensor_product, hyper_tensor_T_product, \
    homogeneity_tensor, \
    dg_cross, dg_outer, \
    tensor_jump, tangent_jump

from .fluxes import \
    LocalLaxFriedrichs, \
    HLLE, \
    Vijayasundaram

from dolfin_dg.tensors import force_zero_function_derivative

from .operators import \
    EllipticOperator, \
    HyperbolicOperator, \
    PoissonOperator, \
    CompressibleEulerOperator, CompressibleEulerOperatorEntropyFormulation, \
    CompressibleNavierStokesOperator, CompressibleNavierStokesOperatorEntropyFormulation, \
    MaxwellOperator, \
    SpacetimeBurgersOperator, \
    StokesOperator, \
    DGNeumannBC, DGDirichletBC, DGAdiabticWallBC, \
    DGFemViscousTerm, DGFemCurlTerm, DGFemStokesTerm

# DWR highly experimental
from .dwr import \
    NonlinearAPosterioriEstimator, \
    LinearAPosterioriEstimator, \
    dual

from .mark import \
    FixedFractionMarker, FixedFractionMarkerParallel

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
