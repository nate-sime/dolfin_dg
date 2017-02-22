from ufl import as_matrix, outer, as_vector, jump, avg, inner, replace, grad, variable, diff, dot
from ufl.algorithms.apply_derivatives import apply_derivatives
import ufl
import inspect

__author__ = 'njcs4'


def ufl_adhere_transpose(v):
    if ufl.rank(v) == 1:
        return ufl_T(v)
    return v


def ufl_T(v):
    if ufl.rank(v) == 1:
        return as_matrix(([v[j] for j in range(v.ufl_shape[0])],))
    return v.T


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
        if ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
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
        if ufl.rank(tau) > 1 and tau.ufl_shape[0] == 1:
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


def dg_outer(*args):
    return ufl_adhere_transpose(outer(*args))


def homogeneity_tensor(F_v, u, as_dict=False):
    if not len(inspect.getargspec(F_v).args) == 2:
        raise TypeError("Function F_v must have 2 arguments, (u, grad_u)")

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


class DGFemViscousTerm:

    def __init__(self, F_v, u_vec, v_vec, sigma, G, n):
        self.F_v = F_v
        self.U = u_vec
        self.V = v_vec
        self.grad_v_vec = grad(v_vec)
        self.sig = sigma
        self.G = G
        self.n = n

    def __eval_F_v(self, U):
        if len(inspect.getargspec(self.F_v).args) == 1:
            tau = self.F_v(U)
        else:
            tau = self.F_v(U, grad(U))
        tau = ufl_adhere_transpose(tau)
        return tau

    def __make_boundary_G(self, G, u_gamma):
        if isinstance(G, ufl.core.expr.Expr):
            return replace(G, {self.U: u_gamma})

        assert(isinstance(G, dict))
        G_gamma = {}
        for idx, tensor in G.iteritems():
            G_gamma[idx] = replace(tensor, {self.U: u_gamma})
        return G_gamma

    def interior_residual(self, dInt):
        G = self.G
        F_v, u, v, grad_v = self.F_v, self.U, self.V, self.grad_v_vec
        sig, n = self.sig, self.n

        residual = - inner(tensor_jump(u, n), avg(hyper_tensor_T_product(G, grad_v)))*dInt \
                    - inner(ufl_adhere_transpose(avg(self.__eval_F_v(self.U))), tensor_jump(v, n))*dInt \
                    + inner(sig('+')*hyper_tensor_product(g_avg(G), tensor_jump(u, n)), tensor_jump(v, n))*dInt
        return residual

    def exterior_residual(self, u_gamma, dExt):
        G = self.__make_boundary_G(self.G, u_gamma)
        # u_gamma = ufl_adhere_T(u_gamma)
        F_v, u, v, grad_u, grad_v = self.F_v, self.U, self.V, grad(self.U), self.grad_v_vec
        # u = ufl_adhere_T(u)
        n = self.n

        residual = - inner(dg_outer(u - u_gamma, n), hyper_tensor_T_product(G, grad_v)) * dExt \
                    - inner(hyper_tensor_product(G, grad_u), dg_outer(v, n)) * dExt \
                    + inner(self.sig*hyper_tensor_product(G, dg_outer(u - u_gamma, n)), dg_outer(v, n)) * dExt
        return residual
