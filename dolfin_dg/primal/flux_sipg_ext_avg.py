import ufl

from dolfin_dg.math import tensor_jump, cross_jump, dg_cross
from dolfin_dg.math import hyper_tensor_product as G_mult
from dolfin_dg.math import hyper_tensor_T_product as G_T_mult
import dolfin_dg.primal


# TODO: This should be refactored to generalise the flux function F_hat
# The only difference here is the boundary average in F_hat for one IBP:
# F_hat = 0.5 * (F(u) + F(uD))

class DivIBP(dolfin_dg.primal.IBP):

    def interior_residual1(self, alpha, u_pen, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = ufl.avg(F(u)) - G_mult(alpha, tensor_jump(u_pen, n))
        R = ufl.inner(F_hat, tensor_jump(G_T_v, n)) * dS
        return R

    def exterior_residual1(self, alpha, u_pen, u_penD, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)

        F_hat = 0.5*(F(u) + F(uD)) - G_mult(alpha, ufl.outer(u_pen - u_penD, n))
        R = ufl.inner(F_hat, ufl.outer(G_gamma_T_v, n)) * ds
        return R

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = ufl.avg(F(u))
        R = ufl.inner(F_hat - F(u)("+"), ufl.outer(G_T_v, n)("+")) * dS \
            + ufl.inner(F_hat - F(u)("-"), ufl.outer(G_T_v, n)("-")) * dS
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        F_hat = F(uD)
        R = ufl.inner(F_hat - F(u), ufl.outer(G_T_v, n)) * ds
        return R


class GradIBP(dolfin_dg.primal.IBP):

    def interior_residual1(self, alpha, u_pen, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = ufl.avg(F(u)) - alpha * ufl.jump(u_pen, n)
        R = ufl.inner(F_hat, ufl.jump(G_T_v, n)) * dS
        return R

    def exterior_residual1(self, alpha, u_pen, u_penD, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        # FIXME: Workaround for UFL by reformulating:
        # size(F) = m, size(G_T_v) = m x d, size(n) = d
        # <F, G_T_v . n> = F{i} G_T_v{ij} n{j} = <F otimes n, G_T_V>
        F_hat = ufl.outer(0.5*(F(u) + F(uD)), n) - alpha * (u_pen - u_penD)
        R = ufl.inner(G_T_v, F_hat) * ds
        return R

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = ufl.avg(F(u))
        def bdry_dot(v, n):
            r = len(v.ufl_shape)
            return v * n if r == 0 else ufl.dot(v, n)
        R = ufl.inner(F_hat - F(u)("+"), bdry_dot(G_T_v, n)("+")) * dS + \
            ufl.inner(F_hat - F(u)("-"), bdry_dot(G_T_v, n)("-")) * dS
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)

        def bdry_dot(v, n):
            r = len(v.ufl_shape)
            return v * n if r == 0 else ufl.dot(v, n)
        F_hat = F(uD)
        R = ufl.inner(F_hat - F(u), bdry_dot(G_gamma_T_v, n)) * ds
        return R


class CurlIBP(dolfin_dg.primal.IBP):

    def interior_residual1(self, alpha, u_pen, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = ufl.avg(F(u)) - G_mult(alpha, cross_jump(u_pen, n))
        R = - ufl.inner(F_hat, cross_jump(G_T_v, n)) * dS
        return R

    def exterior_residual1(self, alpha, u_pen, u_penD, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = 0.5*(F(u) + F(uD)) - G_mult(alpha, dg_cross(n, u_pen - u_penD))
        R = - ufl.inner(F_hat, dg_cross(n, G_T_v)) * ds
        return R

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = ufl.avg(F(u))
        R = - ufl.inner(F_hat - F(u)("+"), dg_cross(n, G_T_v)("+")) * dS \
            - ufl.inner(F_hat - F(u)("-"), dg_cross(n, G_T_v)("-")) * dS
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)

        F_hat = F(uD)

        #TODO: the cross product acting on scalar G_T_v is problematic, so reform
        # using a . (b x c) = b . (c x a) = c . (a x b)
        # Original from derivation:
        #     R = ufl.inner(F(u) - F(uD), dg_cross(n, G_T_v)) * ds
        # R = ufl.inner(dg_cross(F(u) - F(uD), n), G_T_v) * ds
        R = - ufl.inner(F_hat - F(u), dg_cross(n, G_T_v)) * ds
        return R