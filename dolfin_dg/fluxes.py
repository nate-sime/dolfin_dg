from ufl.restriction import Restricted
from ufl import dot, Constant
from dolfin import Function

__author__ = 'njcs4'


def force_zero_function_derivative(avec):
    if isinstance(avec, Restricted):
        if isinstance(avec.ufl_operands[0], Function):
            return avec.ufl_operands[0].copy()(avec.side())
    if isinstance(avec, Function):
        return avec.copy()
    return avec

def lax_friedrichs_flux(F_c, alpha):
    def H(U_p, U_m, n_p):
        return 0.5*(dot(F_c(U_p), n_p) + dot(F_c(U_m), n_p) + alpha*(U_p - U_m))
    return H