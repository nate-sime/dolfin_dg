
__author__ = 'njcs4'


from dolfin_dg.dolfin.tensors import force_zero_function_derivative

# DWR highly experimental
from dolfin_dg.dolfin.dwr import \
    NonlinearAPosterioriEstimator, \
    LinearAPosterioriEstimator, \
    dual

from dolfin_dg.dolfin.mark import \
    FixedFractionMarker, FixedFractionMarkerParallel