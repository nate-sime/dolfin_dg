import typing

import numpy as np
import ufl

import dolfin_dg
from dolfin_dg.math import hyper_tensor_T_product as G_T_mult


def green_transpose(
        ufl_op: typing.Union[ufl.div, ufl.grad, ufl.curl, ufl.Identity]):
    if ufl_op is ufl.div:
        return ufl.grad
    if ufl_op is ufl.grad:
        return ufl.div
    return ufl_op


class IBP:

    def __init__(self, F, u, v, G):
        self.F = F
        self.u = u
        self.v = v
        self.G = G
        # print(f"Initialising {self}")
        # print(f"Shape F(u) = {F(u).ufl_shape}")
        # print(f"Shape G = {G.ufl_shape}")
        # print(f"Shape u = {u.ufl_shape}")
        # print(f"Shape v = {v.ufl_shape}")


class FirstOrderSystem:

    def __init__(self, F_vec, L_vec, u, v):
        self.u = u
        self.v = v
        self.L_vec = L_vec
        self.F_vec = F_vec
        self.G_vec = [
            dolfin_dg.math.homogenize(F_vec[i], u, L_vec[i](F_vec[i + 1](u)))
            for i in range(len(F_vec) - 1)]

        v_vec = [v]
        for i in range(len(F_vec) - 2):
            vj = dolfin_dg.primal.green_transpose(L_vec[i])(
                G_T_mult(self.G_vec[i], v_vec[i]))
            v_vec.append(vj)
        self.v_vec = v_vec

    # Domain
    # F = - ufl.inner(F_1(u), v_vec[1]) * ufl.dx - ufl.inner(f, v) * ufl.dx

    def interior(self, alpha, n_ibps, flux_type, u_soln):
        F_vec = self.F_vec
        G_vec = self.G_vec
        L_vec = self.L_vec
        u, v_vec = self.u, self.v_vec

        sign_tracker = np.ones(len(F_vec)-1, dtype=np.int8)
        F_sub = 0
        for j in range(len(F_vec) - 1)[::-1]:
            if L_vec[j] in (ufl.div, ufl.grad):
                F_sub = -1 * F_sub
                sign_tracker[j+1:] *= -1
                print("j", j, "sign tracker", sign_tracker)

            IBP = {ufl.div: flux_type.DivIBP,
                   ufl.grad: flux_type.GradIBP,
                   ufl.curl: flux_type.CurlIBP}
            ibp = IBP[L_vec[j]](F_vec[j + 1], u, v_vec[j], G_vec[j])

            if n_ibps[j] == 1:
                L_op = L_vec[-j] if j > 0 else lambda x: x
                interior = ibp.interior_residual1(alpha[j]("+") * ufl.avg(G_vec[j + 1]), L_op(u))
                exterior = ibp.exterior_residual1(
                    alpha[j] * ufl.replace(G_vec[j + 1], {u: u_soln}), L_op(u), L_op(u_soln), u_soln)
            elif n_ibps[j] == 2:
                interior = ibp.interior_residual2()
                exterior = ibp.exterior_residual2(u_soln)
            F_sub += interior + exterior
        print("sign tracker", sign_tracker)
        return F_sub