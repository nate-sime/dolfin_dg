
__author__ = 'njcs4'

from dg_form import \
    DGFemViscousTerm, \
    hyper_tensor_product, hyper_tensor_T_product, \
    homogeneity_tensor, \
    dg_cross, dg_outer, \
    tensor_jump, tangent_jump

from fluxes import \
    force_zero_function_derivative, \
    lax_friedrichs_flux

from operators import \
    EllipticOperator, \
    HyperbolicOperator, \
    PoissonOperator, \
    CompressibleEulerOperator, CompressibleEulerOperatorEntropyFormulation, \
    CompressibleNavierStokesOperator, CompressibleNavierStokesOperatorEntropyFormulation, \
    MaxwellOperator, \
    SpacetimeBurgersOperator, \
    DGNeumannBC, DGDirichletBC, \
    DGFemViscousTerm, DGFemCurlTerm