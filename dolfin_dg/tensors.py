"""
This module is provided as a reference for the homogeneity tensors of the compressible Navier-Stokes equations.
It is used simply for regression testing (see dolfin_dg/test/regression/).
These tensors may be automatically computed using dolfin_dg.dg_form.homogeneity_tensor().
"""
from dolfin import Function
from ufl import as_matrix
from ufl.restriction import Restricted

__author__ = 'njcs4'


def compressible_ns_G(u_vec, mu, Pr, gamma):

    G = {}

    G[0, 0] = as_matrix([[0, 0, 0, 0],
                         [-4*mu*u_vec[1]/(3*u_vec[0]**2), 4*mu/(3*u_vec[0]), 0, 0],
                         [-mu*u_vec[2]/u_vec[0]**2, 0, mu/u_vec[0], 0],
                         [mu*(-Pr*(4*u_vec[1]**2 + 3*u_vec[2]**2) + 3*gamma*(-u_vec[0]*u_vec[3] + u_vec[1]**2 + u_vec[2]**2))/(3*Pr*u_vec[0]**3), mu*(4*Pr - 3*gamma)*u_vec[1]/(3*Pr*u_vec[0]**2), mu*(Pr - gamma)*u_vec[2]/(Pr*u_vec[0]**2), gamma*mu/(Pr*u_vec[0])]])

    G[0, 1] = as_matrix([[0, 0, 0, 0],
                         [2*mu*u_vec[2]/(3*u_vec[0]**2), 0, -2*mu/(3*u_vec[0]), 0],
                         [-mu*u_vec[1]/u_vec[0]**2, mu/u_vec[0], 0, 0],
                         [-mu*u_vec[1]*u_vec[2]/(3*u_vec[0]**3), mu*u_vec[2]/u_vec[0]**2, -2*mu*u_vec[1]/(3*u_vec[0]**2), 0]])

    G[1, 0] = as_matrix([[0, 0, 0, 0],
                         [-mu*u_vec[2]/u_vec[0]**2, 0, mu/u_vec[0], 0],
                         [2*mu*u_vec[1]/(3*u_vec[0]**2), -2*mu/(3*u_vec[0]), 0, 0],
                         [-mu*u_vec[1]*u_vec[2]/(3*u_vec[0]**3), -2*mu*u_vec[2]/(3*u_vec[0]**2), mu*u_vec[1]/u_vec[0]**2, 0]])

    G[1, 1] = as_matrix([[0, 0, 0, 0],
                         [-mu*u_vec[1]/u_vec[0]**2, mu/u_vec[0], 0, 0],
                         [-4*mu*u_vec[2]/(3*u_vec[0]**2), 0, 4*mu/(3*u_vec[0]), 0],
                         [mu*(-Pr*(3*u_vec[1]**2 + 4*u_vec[2]**2) + 3*gamma*(-u_vec[0]*u_vec[3] + u_vec[1]**2 + u_vec[2]**2))/(3*Pr*u_vec[0]**3), mu*(Pr - gamma)*u_vec[1]/(Pr*u_vec[0]**2), mu*(4*Pr - 3*gamma)*u_vec[2]/(3*Pr*u_vec[0]**2), gamma*mu/(Pr*u_vec[0])]])
    return G


def compressible_ns_entopy_G(mu, lam, Pr, gamma, V):
    V1, V2, V3, V4 = V
    e1 = V2*V4
    e2 = V3*V4
    d1 = -V2*V3

    K = {}
    K[0, 0] = 1/V4**3*as_matrix([[0, 0, 0, 0],
                              [0, -(lam + 2*mu)*V4**2, 0, (lam + 2*mu)*e1],
                              [0, 0, -mu*V4**2, mu*e2],
                              [0, (lam + 2*mu)*e1, mu*e2, -((lam + 2*mu)*V2**2 + mu*V3**2 - gamma*mu*V4/Pr)]])

    K[0, 1] = 1/V4**3*as_matrix([[0, 0, 0, 0],
                              [0, 0, -lam*V4**2, lam*e2],
                              [0, -mu*V4**2, 0, mu*e1],
                              [0, mu*e2, lam*e1, (lam + mu)*d1]])

    K[1, 0] = K[0, 1].T

    K[1, 1] =1/V4**3*as_matrix([[0, 0, 0, 0],
                             [0, -mu*V4**2, 0, mu*e1],
                             [0, 0, -(lam + 2*mu)*V4**2, (lam + 2*mu)*e2],
                             [0, mu*e1, (lam + 2*mu)*e2, -((lam + 2*mu)*V3**2 + mu*V2**2 - gamma*mu*V4/Pr)]])

    return K


def force_zero_function_derivative(avec):
    if isinstance(avec, Restricted):
        if isinstance(avec.ufl_operands[0], Function):
            return avec.ufl_operands[0].copy()(avec.side())
    if isinstance(avec, Function):
        return avec.copy()
    return avec