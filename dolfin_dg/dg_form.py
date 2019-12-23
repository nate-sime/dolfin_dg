import inspect

import ufl
from ufl import as_matrix, outer, as_vector, jump, avg, inner, replace, grad, variable, diff, dot, cross, curl

__author__ = 'njcs4'


def ufl_adhere_transpose(v):
    if ufl.rank(v) == 1:
        return ufl_T(v)
    return v


def ufl_T(v):
    if ufl.rank(v) == 1:
        return as_matrix(([v[j] for j in range(v.ufl_shape[0])],))
    return v.T


def normal_proj(u, n):
    return ufl.outer(n, n) * u


def tangential_proj(u, n):
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def g_avg(G):
    if isinstance(G, ufl.core.expr.Expr):
        return avg(G)

    assert(isinstance(G, dict))
    result = {}
    for k in G.keys():
        result[k] = avg(G[k])
    return result


def hyper_tensor_product(G, tau):
    if isinstance(G, ufl.core.expr.Expr):
        if len(G.ufl_shape) == 0:
            if not len(tau.ufl_shape) == 0:
                raise IndexError("G is scalar, tau has shape: %s" + str(tau.ufl_shape))
            return G*tau
        elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
            return dot(G, tau.T).T
        elif ufl.rank(tau) == 1:
            return ufl_adhere_transpose(dot(G, tau))
        m, d = tau.ufl_shape
        return as_matrix([[inner(G[i, k, :, :], tau) for k in range(d)] for i in range(m)])

    assert(isinstance(G, dict))
    shape = tau.ufl_shape
    if len(shape) == 1:
        tau = ufl_T(tau)
        shape = tau.ufl_shape
    m, d = shape
    result = [[0 for _ in range(d)] for _ in range(m)]
    for i in range(m):
        for k in range(d):
            prod = 0
            for j in range(m):
                for l in range(d):
                    prod += (G[k,l][i,j]*tau[j,l])
            result[i][k] = prod
    result = as_matrix(result)
    return result


def hyper_tensor_T_product(G, tau):
    if isinstance(G, ufl.core.expr.Expr):
        if len(G.ufl_shape) == 0:
            if not len(tau.ufl_shape) == 0:
                raise IndexError("G^T is scalar, tau has shape: %s" + str(tau.ufl_shape))
            return G*tau
        elif ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
            return dot(G.T, tau)
        elif ufl.rank(tau) == 1:
            return ufl_adhere_transpose(dot(G.T, tau))
        m, d = tau.ufl_shape
        return as_matrix([[inner(G[:, :, i, k], tau) for k in range(d)] for i in range(m)])

    assert(isinstance(G, dict))
    shape = tau.ufl_shape
    if len(shape) == 1:
        tau = ufl_T(tau)
        shape = tau.ufl_shape
    m, d = shape
    result = [[0 for _ in range(d)] for _ in range(m)]
    for j in range(m):
        for l in range(d):
            prod = 0
            for i in range(m):
                for k in range(d):
                    prod += (G[k,l][i,j]*tau[i,k])
            result[j][l] = prod
    result = as_matrix(result)
    return result


def tensor_jump(u, n):
    if len(u.ufl_shape) == 0:
        u = as_vector((u,))
    assert(len(u.ufl_shape) == 1)
    return dg_outer(jump(u), n('+'))


def dg_cross(u, v):
    if len(u.ufl_shape) == 0 or len(v.ufl_shape) == 0:
        raise TypeError("Input argument must be a vector")
    assert(len(u.ufl_shape) == 1 and len(v.ufl_shape) == 1)
    if u.ufl_shape[0] == 2 and v.ufl_shape[0] == 2:
        return u[0]*v[1] - u[1]*v[0]
    return ufl_adhere_transpose(cross(u, v))


def tangent_jump(u, n):
    if len(u.ufl_shape) == 0:
        raise TypeError("Input argument must be a vector")
    assert(len(u.ufl_shape) == 1)
    assert(u.ufl_shape[0] in (2, 3))
    return dg_cross(n('+'), jump(u))


def dg_outer(*args):
    return ufl_adhere_transpose(outer(*args))


def homogeneity_tensor(F_v, u, as_dict=False):
    if len(inspect.getfullargspec(F_v).args) < 2:
        raise TypeError("Function F_v must have at least 2 arguments, "
                        "(u, grad_u, *args **kwargs)")

    G = {}

    grad_u = variable(grad(u))
    tau = F_v(u, grad_u)

    if not as_dict:
        return diff(tau, grad_u)

    shape = grad(u).ufl_shape
    if not shape:
        m, d = 1, 1
    elif len(shape) == 1:
        m, d = 1, shape[0]
    elif len(shape) == 2:
        m, d = shape
    else:
        raise TypeError("Tensor rank error, ufl_shape is: %s" % str(shape))

    for k in range(d):
        fv = tau[k] if m == 1 else tau[:, k]
        for l in range(d):
            g = [[0 for _ in range(m)] for _ in range(m)]
            for r in range(m):
                for c in range(m):
                    diff_val = diff(fv, grad_u)[l] if len(fv.ufl_shape) == 0 else diff(fv[r], grad_u)[c,l]
                    g[r][c] = diff_val
            G[k,l] = as_matrix(g)

    return G


def _get_terminal_operand_coefficient(u):
    if not isinstance(u, ufl.Coefficient):
        return _get_terminal_operand_coefficient(
            u.ufl_operands[0])
    return u


def _get_ufl_element_degree(u):
    # Assume either a coefficient or ufl indexed operand
    if isinstance(u, ufl.Coefficient):
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


class DGFemViscousTerm:

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n):
        self.F_v = F_v
        self.U = u_vec
        self.V = v_vec
        self.grad_v_vec = grad(v_vec)
        self.sigma = sigma
        self.G = G
        self.n = n

    def _eval_F_v(self, U, grad_U):
        if len(inspect.getfullargspec(self.F_v).args) == 1:
            tau = self.F_v(U)
        else:
            tau = self.F_v(U, grad_U)
        tau = ufl_adhere_transpose(tau)
        return tau

    def _make_boundary_G(self, G, u_gamma):
        U = self.U
        U_soln = _get_terminal_operand_coefficient(U)

        bc_shape = u_gamma.ufl_shape[0] if u_gamma.ufl_shape else 1
        soln_shape = U_soln.ufl_shape[0] if U_soln.ufl_shape else 1
        if soln_shape > bc_shape:
            U_soln_idx = _get_ufl_list_tensor_indices(U)
            if not hasattr(U_soln_idx, "__len__"):
                U_soln_idx = (U_soln_idx,)
            u_gamma_vec = [0] * soln_shape
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

    def interior_residual(self, dInt):
        raise NotImplementedError("Interior residual not implemented in %s" % str(self.__class__))

    def exterior_residual(self, u_gamma, dExt):
        raise NotImplementedError("Exterior residual not implemented in %s" % str(self.__class__))

    def exterior_residual_on_interior(self, u_gamma, dExt):
        raise NotImplementedError("Exterior residual not implemented in %s" % str(self.__class__))

    def neumann_residual(self, g_N, dExt):
        return -inner(g_N, self.V)*dExt


class DGClassicalSecondOrderDiscretisation(DGFemViscousTerm):

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n, delta):
        super().__init__(F_v, u_vec, v_vec, sigma, G, n)
        self.delta = delta

    def interior_residual(self, dInt):
        G = self.G
        u, v, grad_v = self.U, self.V, self.grad_v_vec
        grad_u = grad(u)
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(tensor_jump(u, n), avg(hyper_tensor_T_product(G, grad_v))) * dInt \
                   - inner(ufl_adhere_transpose(avg(self._eval_F_v(u, grad_u))), tensor_jump(v, n)) * dInt
        if sigma is not None:
            residual += inner(sigma('+') * hyper_tensor_product(g_avg(G), tensor_jump(u, n)), tensor_jump(v, n)) * dInt
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        G = self._make_boundary_G(self.G, u_gamma)
        u, v, grad_u, grad_v = self.U, self.V, grad(self.U), self.grad_v_vec
        sigma, n = self.sigma, self.n
        delta = self.delta

        residual = delta * inner(dg_outer(u - u_gamma, n), hyper_tensor_T_product(G, grad_v)) \
                   - inner(self._eval_F_v(u, grad_u), dg_outer(v, n))
        if sigma is not None:
            residual += inner(sigma * hyper_tensor_product(G, dg_outer(u - u_gamma, n)), dg_outer(v, n))
        return residual

    def exterior_residual(self, u_gamma, dExt):
        return self._exterior_residual_no_integral(u_gamma) * dExt

    def exterior_residual_on_interior(self, u_gamma, dExt):
        return sum(self._exterior_residual_no_integral(u_gamma)(side) * dExt
                   for side in ("+", "-"))


class DGFemSIPG(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = -1
        super().__init__( F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemNIPG(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = +1
        super().__init__( F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemBO(DGClassicalSecondOrderDiscretisation):

    def __init__(self,  F_v, u_vec, v_vec, sigma, G, n):
        delta = +1
        sigma = None
        super().__init__( F_v, u_vec, v_vec, sigma, G, n, delta)


class DGFemCurlTerm:

    def __init__(self, F_m, u_vec, v_vec, sigma, G, n):
        self.F_m = F_m
        self.U = u_vec
        self.V = v_vec
        self.curl_v_vec = curl(v_vec)
        self.sig = sigma
        self.G = G
        self.n = n

    def __eval_F_v(self, U):
        if len(inspect.getfullargspec(self.F_m).args) == 1:
            tau = self.F_m(U)
        else:
            tau = self.F_m(U, curl(U))
        tau = ufl_adhere_transpose(tau)
        return tau

    def __make_boundary_G(self, G, u_gamma):
        if isinstance(G, ufl.core.expr.Expr):
            return replace(G, {self.U: u_gamma})

        assert(isinstance(G, dict))
        G_gamma = {}
        for idx, tensor in G.items():
            G_gamma[idx] = replace(tensor, {self.U: u_gamma})
        return G_gamma

    def interior_residual(self, dInt):
        G = self.G
        F_v, u, v, curl_v = self.F_m, self.U, self.V, self.curl_v_vec
        sig, n = self.sig, self.n

        residual = - inner(tangent_jump(u, n), avg(hyper_tensor_T_product(G, curl_v)))*dInt \
                   - inner(tangent_jump(v, n), avg(self.__eval_F_v(u)))*dInt \
                   + sig('+')*inner(hyper_tensor_product(g_avg(G), tangent_jump(u, n)), tangent_jump(v, n))*dInt

        return residual

    def exterior_residual(self, u_gamma, dExt):
        G = self.__make_boundary_G(self.G, u_gamma)
        F_v, u, v, grad_u, curl_v = self.F_m, self.U, self.V, curl(self.U), self.curl_v_vec
        n = self.n

        residual = - inner(dg_cross(n, u - u_gamma), hyper_tensor_T_product(G, curl_v))*dExt \
                   - inner(dg_cross(n, v), hyper_tensor_product(G, grad_u))*dExt \
                   + self.sig*inner(hyper_tensor_product(G, dg_cross(n, u - u_gamma)), dg_cross(n, v))*dExt
        return residual


class DGFemStokesTerm(DGClassicalSecondOrderDiscretisation):

    def __init__(self, F_v, u, p, v, q, sigma, G, n, delta,
                 block_form=False):
        self.u, self.v = u, v
        self.p, self.q = p, q
        self.block_form = block_form
        super().__init__(F_v, u, v, sigma, G, n, delta)

    def interior_residual(self, dInt):
        u = self.u
        p, q = self.p, self.q
        n = self.n

        residual = [super().interior_residual(dInt),
                    -ufl.jump(u, n) * avg(q) * dInt]
        if not self.block_form:
            residual = sum(residual)
        return residual

    def _exterior_residual_no_integral(self, u_gamma):
        u = self.u
        p, q = self.p, self.q
        n = self.n

        residual = [super()._exterior_residual_no_integral(u_gamma),
                    - dot(u - u_gamma, n) * q]
        return residual

    def exterior_residual(self, u_gamma, dExt):
        residual = list(map(lambda Fj: Fj*dExt, self._exterior_residual_no_integral(u_gamma)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def exterior_residual_on_interior(self, u_gamma, dExt):
        residual = list(map(lambda Fj: sum(Fj(side)*dExt for side in ("+", "-")),
                            self._exterior_residual_no_integral(u_gamma)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def _slip_exterior_residual_no_integral(self, u_gamma, f2):
        p, q = self.p, self.q

        G = self._make_boundary_G(self.G, u_gamma)
        u, v, grad_u, grad_v = self.U, self.V, grad(self.U), self.grad_v_vec
        sigma, n = self.sigma, self.n
        delta = self.delta

        # Velocity block
        F0 = delta * inner(u - u_gamma, normal_proj(hyper_tensor_T_product(G, grad_v) * n, n)) \
                   - inner(normal_proj(self._eval_F_v(u, grad_u) * n, n), v)
        if sigma is not None:
            F0 += inner(sigma * normal_proj(hyper_tensor_product(G, dg_outer(u - u_gamma, n)) * n, n), v)

        # Tangential force
        F0 -= dot(tangential_proj(f2, n), v)

        # Continuity block
        F1 = - dot(u - u_gamma, n) * q

        return [F0, F1]

    def slip_exterior_residual(self, u_gamma, f2, dExt):
        residual = list(map(lambda Fj: Fj*dExt,
                            self._slip_exterior_residual_no_integral(u_gamma, f2)))
        if not self.block_form:
            residual = sum(residual)
        return residual

    def slip_exterior_residual_on_interior(self, u_gamma, f2, dExt):
        residual = list(map(lambda Fj: sum(Fj(side)*dExt for side in ("+", "-")),
                            self._slip_exterior_residual_no_integral(u_gamma, f2)))
        if not self.block_form:
            residual = sum(residual)
        return residual
