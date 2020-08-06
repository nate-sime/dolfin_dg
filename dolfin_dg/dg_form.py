import abc
import inspect

import ufl
from ufl import (
    as_matrix, outer, inner, replace, grad, variable, diff, dot, curl, div)

from dolfin_dg.dg_ufl import (
    apply_dg_operators, avg, tensor_jump, jump, tangent_jump, dg_cross)


def normal_proj(u, n):
    return ufl.outer(n, n) * u


def tangential_proj(u, n):
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def hyper_tensor_product(G, tau):
    if len(G.ufl_shape) == 0:
        if not len(tau.ufl_shape) == 0:
            raise IndexError("G is scalar, tau has shape: %s"
                             + str(tau.ufl_shape))
        return G*tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return dot(G, tau.T).T
    elif ufl.rank(tau) == 1:
        return dot(G, tau)
    m, d = tau.ufl_shape
    return as_matrix([[inner(G[i, k, :, :], tau) for k in range(d)]
                      for i in range(m)])


def hyper_tensor_T_product(G, tau):
    if len(G.ufl_shape) == 0:
        if not len(tau.ufl_shape) == 0:
            raise IndexError("G^T is scalar, tau has shape: %s"
                             + str(tau.ufl_shape))
        return G*tau
    elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
        return dot(G.T, tau)
    elif ufl.rank(tau) == 1:
        return dot(G.T, tau)
    m, d = tau.ufl_shape
    return as_matrix([[inner(G[:, :, i, k], tau) for k in range(d)]
                      for i in range(m)])


def dg_outer(*args):
    # TODO: ufl treats this as (u ⊗ v*). If dolfin_dg goes complex need to
    #  fix this
    return outer(*args)


def homogeneity_tensor(F_v, u, differential_operator=grad):
    if len(inspect.getfullargspec(F_v).args) < 2:
        raise TypeError("Function F_v must have at least 2 arguments, "
                        "(u, grad_u, *args, **kwargs)")

    diff_op_u = variable(differential_operator(u))
    tau = F_v(u, diff_op_u)
    return diff(tau, diff_op_u)


def _get_terminal_operand_coefficient(u):
    if not isinstance(u, ufl.Coefficient):
        return _get_terminal_operand_coefficient(
            u.ufl_operands[0])
    return u


def _get_ufl_element_degree(u):
    # Assume either a function space, coefficient or ufl indexed operand
    if isinstance(u, (ufl.Coefficient, ufl.FunctionSpace)):
        return u.ufl_element().degree()
    if isinstance(u, ufl.tensors.ListTensor):
        return _get_ufl_element_degree(_get_terminal_operand_coefficient(u))
    else:
        coeff, idx = u.ufl_operands
        element = coeff.ufl_element()
        return element.degree()


def _get_ufl_list_tensor_indices(u):
    if isinstance(u, ufl.tensors.ListTensor):
        return list(map(_get_ufl_list_tensor_indices, u.ufl_operands))
    return u.ufl_operands[1][0]._value


def generate_default_sipg_penalty_term(u, C_IP=20.0):
    h = ufl.CellDiameter(u.ufl_domain())
    ufl_degree = _get_ufl_element_degree(u)
    return C_IP * max(ufl_degree ** 2, 1) / h


class DGFemTerm(abc.ABC):

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n):
        self.F_v = F_v
        self.U = u_vec
        self.V = v_vec
        self.sigma = sigma
        self.G = G
        self.n = n

    def _make_boundary_G(self, G, u_gamma):
        U = self.U
        U_soln = _get_terminal_operand_coefficient(U)

        # Reshape u_gamma if U_soln is from x mixed space
        bc_shape = u_gamma.ufl_shape[0] if u_gamma.ufl_shape else 1
        soln_shape = U_soln.ufl_shape[0] if U_soln.ufl_shape else 1
        if soln_shape > bc_shape:
            U_soln_idx = _get_ufl_list_tensor_indices(U)
            if not hasattr(U_soln_idx, "__len__"):
                U_soln_idx = (U_soln_idx,)

            # Construct new u_gamma from existing U_soln and replace
            # bc components with their prescribed data
            u_gamma_vec = [u_sub for u_sub in U_soln]
            if not u_gamma.ufl_shape:
                u_gamma = [u_gamma]
            for j, idx in enumerate(U_soln_idx):
                u_gamma_vec[idx] = u_gamma[j]
            u_gamma = ufl.as_vector(u_gamma_vec)

        if isinstance(G, ufl.core.expr.Expr):
            return replace(G, {U_soln: u_gamma})

        assert(isinstance(G, dict))
        G_gamma = {}
        for idx, tensor in G.items():
            G_gamma[idx] = replace(tensor, {U_soln: u_gamma})
        return G_gamma

    @abc.abstractmethod
    def interior_residual(self, dInt):
        pass

    @abc.abstractmethod
    def exterior_residual(self, u_gamma, dExt):
        pass

    @abc.abstractmethod
    def exterior_residual_on_interior(self, u_gamma, dExt):
        pass

    @abc.abstractmethod
    def neumann_residual(self, g_N, dExt):
        return -inner(g_N, self.V)*dExt


class DGClassicalSecondOrderDiscretisation(DGFemTerm):

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n, delta):
        super().__init__(F_v, u_vec, v_vec, sigma, G, n)
        self.delta = delta

    def interior_residual(self, dInt):
        G = self.G
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(
            tensor_jump(u, n), avg(hyper_tensor_T_product(G, grad_v))) * dInt \
            - inner(avg(self.F_v(u, grad_u)), tensor_jump(v, n)) * dInt
        if sigma is not None:
            residual += inner(
                sigma('+') * hyper_tensor_product(avg(G), tensor_jump(u, n)),
                tensor_jump(v, n)) * dInt

        residual = apply_dg_operators(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(
            dg_outer(u - u_gamma, n), hyper_tensor_T_product(G, grad_v)) \
            - inner(self.F_v(u, grad_u), dg_outer(v, n))
        if sigma is not None:
            residual += inner(
                sigma * hyper_tensor_product(G, dg_outer(u - u_gamma, n)),
                dg_outer(v, n))
        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))

    def neumann_residual(self, g_N, dExt):
        return -inner(g_N, self.V)*dExt


class DGFemSIPG(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = -1
        super().__init__(F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemNIPG(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = +1
        super().__init__(F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemBO(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = +1
        sigma = None
        super().__init__(F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemCurlTerm(DGFemTerm):

    def interior_residual(self, dInt):
        G = self.G
        u, v = self.U, self.V
        curl_u, curl_v = curl(u), curl(v)
        sigma, n = self.sigma, self.n

        residual = - inner(tangent_jump(u, n),
                           avg(hyper_tensor_T_product(G, curl_v)))*dInt \
            - inner(tangent_jump(v, n), avg(self.F_v(u, curl_u)))*dInt \
            + sigma('+')*inner(hyper_tensor_product(avg(G), tangent_jump(u, n)),
                               tangent_jump(v, n))*dInt

        residual = apply_dg_operators(residual)
        return residual

    def exterior_residual(self, u_gamma, dExt):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        curl_u, curl_v = curl(u), curl(v)
        sigma, n = self.sigma, self.n

        residual = - inner(dg_cross(n, u - u_gamma),
                           hyper_tensor_T_product(G, curl_v))*dExt \
            - inner(dg_cross(n, v),
                    hyper_tensor_product(G, curl_u))*dExt \
            + sigma*inner(hyper_tensor_product(G, dg_cross(n, u - u_gamma)),
                          dg_cross(n, v))*dExt
        return residual

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return NotImplementedError

    def neumann_residual(self, g_N, dExt):
        return NotImplementedError


class DGFemStokesTerm(DGClassicalSecondOrderDiscretisation):

    def __init__(self, F_v, u, p, v, q, sigma, G, n, delta,
                 block_form=False):
        self.u, self.v = u, v
        self.p, self.q = p, q
        self.block_form = block_form
        super().__init__(F_v, u, v, sigma, G, n, delta)

    def interior_residual(self, dInt):
        u = self.u
        q = self.q
        n = self.n

        residual = [super().interior_residual(dInt),
                    -ufl.jump(u, n) * avg(q) * dInt]
        residual = list(map(apply_dg_operators, residual))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        u = self.u
        q = self.q
        n = self.n

        residual = [super()._exterior_residual_no_integral(u_gamma),
                    - dot(u - u_gamma, n) * q]
        return residual

    def exterior_residual(self, u_gamma, dExt):
        residual = list(map(lambda Fj: Fj*dExt,
                            self._exterior_residual_no_integral(u_gamma)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def exterior_residual_on_interior(self, u_gamma, dExt):
        residual = list(map(
            lambda Fj: sum(Fj(side)*dExt for side in ("+", "-")),
            self._exterior_residual_no_integral(u_gamma)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def _slip_exterior_residual_no_integral(self, u_gamma, f2):
        q = self.q

        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        # Velocity block
        F0 = delta * inner(
            u - u_gamma, normal_proj(hyper_tensor_T_product(G, grad_v) * n, n))\
            - inner(normal_proj(self.F_v(u, grad_u) * n, n), v)
        if sigma is not None:
            F0 += inner(sigma * normal_proj(
                hyper_tensor_product(G, dg_outer(u - u_gamma, n)) * n, n), v)

        # Tangential force
        F0 -= dot(tangential_proj(f2, n), v)

        # Continuity block
        F1 = - dot(u - u_gamma, n) * q

        return [F0, F1]

    def slip_exterior_residual(self, u_gamma, f2, dExt):
        residual = list(map(
            lambda Fj: Fj*dExt,
            self._slip_exterior_residual_no_integral(u_gamma, f2)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def slip_exterior_residual_on_interior(self, u_gamma, f2, dExt):
        residual = list(map(
            lambda Fj: sum(Fj(side)*dExt for side in ("+", "-")),
            self._slip_exterior_residual_no_integral(u_gamma, f2)))
        if not self.block_form:
            residual = sum(residual)
        return residual


class DGClassicalFourthOrderDiscretisation(DGFemTerm):

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n, delta):
        super().__init__(F_v, u_vec, v_vec, sigma, G, n)
        self.delta = delta

    def interior_residual(self, dInt):
        G = self.G
        u, v = self.U, self.V
        grad_v = grad(v)
        div_grad_v = div(grad_v)
        grad_u = grad(u)
        div_grad_u = div(grad_u)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(
            jump(grad_u, n), avg(hyper_tensor_T_product(G, div_grad_v))) * dInt\
            - inner(avg(self.F_v(u, div_grad_u)), jump(grad_v, n)) * dInt
        if sigma is not None:
            residual += inner(
                sigma('+') * hyper_tensor_product(avg(G), jump(grad_u, n)),
                jump(grad_v, n)) * dInt

        residual = apply_dg_operators(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v = self.U, self.V
        grad_u, grad_v = grad(u), grad(v)
        div_grad_u, div_grad_v = div(grad_u), div(grad_v)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(dot(grad(u - u_gamma), n),
                                 hyper_tensor_T_product(G, div_grad_v)) \
            - inner(self.F_v(u, div_grad_u), dot(grad_v, n))
        if sigma is not None:
            residual += inner(
                sigma * hyper_tensor_product(G, dot(grad(u - u_gamma), n)),
                dot(grad_v, n))
        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))

    def neumann_residual(self, g_N, dExt):
        raise NotImplementedError
