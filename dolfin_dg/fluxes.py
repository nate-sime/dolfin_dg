from ufl.restriction import Restricted
from dolfin import Function

__author__ = 'njcs4'


def force_zero_function_derivative(avec):
    if isinstance(avec, Restricted):
        if isinstance(avec.ufl_operands[0], Function):
            return avec.ufl_operands[0].copy()(avec.side())
    if isinstance(avec, Function):
        return avec.copy()
    return avec