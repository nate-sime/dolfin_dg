import typing
import math
import types

import numpy
import numpy as np
import ufl

import dolfin_dg
from dolfin_dg.math import hyper_tensor_T_product as G_T_mult


def default_h_measure(ufl_domain):
    """
    Construct the default cell size measure used in interior penalty
    formulations based on coercivity arguments: :math:`|\\kappa| / |F|`.
    """
    return ufl.CellVolume(ufl_domain) / ufl.FacetArea(ufl_domain)


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

    def __repr__(self):
        msg = f"IBP: {self.__class__}, " \
              f"Shape F(u) = {self.F(self.u).ufl_shape}, " \
              f"Shape G = {self.G.ufl_shape}, " \
              f"Shape u = {self.u.ufl_shape}, " \
              f"Shape v = {self.v.ufl_shape}, "
        return msg


def first_order_flux2(F, flux):
    def flux_wrapper(u, sigma=None):
        if sigma is None:
            sigma = flux(u)
        return F(u, sigma)
    return flux_wrapper


def first_order_flux(flux_func):
    def flux_wrapper(func):
        def flux_guard(u, flux=None):
            if flux is None:
                flux = flux_func(u)
            return func(u, flux)
        return flux_guard
    return flux_wrapper


class FirstOrderSystem:

    def __init__(self,
                 F_vec: typing.Sequence[typing.Callable[
                     [ufl.core.expr.Expr, ufl.core.expr.Expr],
                      ufl.core.expr.Expr]],
                 L_vec: typing.Sequence[
                     typing.Union[ufl.div, ufl.grad, ufl.curl]],
                 u: typing.Union[
                     ufl.coefficient.Coefficient, ufl.core.expr.Expr],
                 v: typing.Union[ufl.argument.Argument, ufl.core.expr.Expr],
                 n_ibps: typing.Optional[typing.Sequence[int]] = None):
        self.u = u
        self.v = v
        self.L_vec = L_vec
        self.F_vec = F_vec
        self.G_vec = [
            dolfin_dg.math.homogenize(F_vec[i], u, L_vec[i](F_vec[i + 1](u)))
            for i in range(len(F_vec) - 1)]
        self.G_vec.append(
            dolfin_dg.math.homogenize(F_vec[-1], u, u)
        )

        v_vec = [v]
        for i in range(len(F_vec) - 1):
            vj = dolfin_dg.primal.green_transpose(L_vec[i])(
                G_T_mult(self.G_vec[i], v_vec[i]))
            v_vec.append(vj)
        self.v_vec = v_vec

        self.L_ops = [*(L_vec[j] for j in range(len(F_vec)-1)), lambda x: x]

        self.sign_tracker = np.ones(len(F_vec)-1, dtype=np.int8)

        if not n_ibps:
            # Create implicitly
            self.n_ibps = np.ones(len(F_vec) - 1, dtype=np.int8)
            self.ibp_2ce_point = math.ceil(self.n_ibps.shape[0]/2.0)
            self.n_ibps[self.ibp_2ce_point:] = 2
        else:
            if not isinstance(n_ibps, np.ndarray):
                n_ibps = np.array(n_ibps, dtype=np.int8)
            assert np.all(n_ibps[1:] > n_ibps[:-1])
            assert np.all(np.isin(n_ibps, (1, 2)))
            assert len(n_ibps.shape) == 1
            assert n_ibps.shape[0] == len(F_vec) - 1
            self.n_ibps = n_ibps
            idx2 = np.where(n_ibps == 2)[0]
            self.ibp_2ce_point = len(self.F_vec) - 1 if idx2.shape[0] == 0 \
                else idx2[0]

    @property
    def G(self) -> ufl.core.expr.Expr:
        return self.G_vec

    def domain(self, dx: ufl.measure.Measure = ufl.dx) -> ufl.form.Form:
        F_vec = self.F_vec
        G_vec = self.G_vec
        L_vec = self.L_vec
        L_ops = self.L_ops
        u, v_vec = self.u, self.v_vec

        sign = 1
        for j in range(len(L_vec)):
            if L_vec[j] in (ufl.div, ufl.grad) and self.n_ibps[j] == 1:
                sign *= -1

        mid_idx = self.ibp_2ce_point
        F = sign * ufl.inner(F_vec[mid_idx](u), v_vec[mid_idx]) * dx
        return F

    def interior(self, alpha: typing.Sequence[ufl.core.expr.Expr],
                 flux_type: typing.Optional[types.ModuleType] = None,
                 dS: ufl.measure.Measure = ufl.dS) -> ufl.form.Form:
        if flux_type is None:
            import dolfin_dg.primal.facet_sipg
            flux_type = dolfin_dg.primal.facet_sipg
        u_soln = None
        return self._formulate(alpha, flux_type, u_soln, interior=True, dI=dS)

    def exterior(self, alpha: typing.Sequence[ufl.core.expr.Expr],
                 u_soln: ufl.core.expr.Expr,
                 flux_type: typing.Optional[types.ModuleType] = None,
                 ds: ufl.measure.Measure = ufl.ds) -> ufl.form.Form:
        if flux_type is None:
            import dolfin_dg.primal.facet_sipg
            flux_type = dolfin_dg.primal.facet_sipg
        return self._formulate(alpha, flux_type, u_soln, interior=False, dI=ds)

    def _formulate(self, alpha: typing.Sequence[ufl.core.expr.Expr],
                   flux_type: types.ModuleType,
                   u_soln: ufl.core.expr.Expr,
                   interior: bool,
                   dI: ufl.measure.Measure):
        F_vec = self.F_vec
        G_vec = self.G_vec
        L_vec = self.L_vec
        L_ops = self.L_ops
        u, v_vec = self.u, self.v_vec

        IBP = {ufl.div: flux_type.DivIBP,
               ufl.grad: flux_type.GradIBP,
               ufl.curl: flux_type.CurlIBP}
        ibps = [IBP[L_vec[j]](F_vec[j + 1], u, v_vec[j], G_vec[j])
                for j in range(len(F_vec) - 1)]

        self.sign_tracker[:] = 1
        F_sub = 0
        for j in range(len(F_vec) - 1)[::-1]:
            ibp = ibps[j]

            if L_vec[j] in (ufl.div, ufl.grad):
                F_sub = -1 * F_sub
                self.sign_tracker[j+1:] *= -1

            if self.n_ibps[j] == 1:
                def L_op(x):
                    # Construct L_{i-2}(L_{i-1}(L_{i}(x)))...
                    L_op = x
                    for L_op_sub in L_ops[-(j+1):][::-1]:
                        L_op = L_op_sub(L_op)
                    return L_op

                if interior:
                    F = ibp.interior_residual1(alpha[j], L_op(u), dS=dI)
                else:
                    F = ibp.exterior_residual1(
                        alpha[j], L_op(u), L_op(u_soln), u_soln, ds=dI)
            elif self.n_ibps[j] == 2:
                if interior:
                    F = ibp.interior_residual2(dS=dI)
                else:
                    F = ibp.exterior_residual2(u_soln, ds=dI)
            F_sub += F

        return F_sub
