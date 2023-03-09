import ufl

from dolfin_dg.math import tensor_jump, cross_jump, dg_cross
from dolfin_dg.math import hyper_tensor_product as G_mult
from dolfin_dg.math import hyper_tensor_T_product as G_T_mult
import dolfin_dg.primal


class DivIBP(dolfin_dg.primal.IBP):

    def interior_residual1(self, alpha, u_pen, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # print(f"Div Shape (ufl.avg(F(u)), tensor_jump(G_T_v, n)): {ufl.avg(F(u)).ufl_shape, tensor_jump(G_T_v, n).ufl_shape}")
        # print(f"Div Shape tensor_jump(G_T_v, n), tensor_jump(u_pen, n): {tensor_jump(G_T_v, n).ufl_shape, tensor_jump(u_pen, n).ufl_shape}")
        # quit()
        R = ufl.inner(ufl.avg(F(u)), tensor_jump(G_T_v, n)) * dS \
            - ufl.inner(tensor_jump(G_T_v, n), G_mult(alpha, tensor_jump(u_pen, n))) * dS
        return R

    def exterior_residual1(self, alpha, u_pen, u_penD, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)
        # TODO: Cleanup
        R = ufl.inner(F(u), ufl.outer(G_gamma_T_v, n)) * ds \
            - ufl.inner(ufl.outer(G_gamma_T_v, n), G_mult(alpha, ufl.outer(u_pen - u_penD, n))) * ds
        return R

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # print(
        #    f"Grad Shape ufl.avg(G_T_v), tensor_jump(F(u), n): {ufl.avg(G_T_v).ufl_shape, tensor_jump(F(u), n).ufl_shape})")

        R = - ufl.inner(ufl.avg(G_T_v), ufl.jump(F(u), n)) * dS
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # G_gamma = ufl.replace(G, {u: uD})
        # G_gamma_T_v = G_T_mult(G_gamma, v)
        # Should this be avg(G)?
        R = - ufl.inner(ufl.outer(G_T_v, n), F(u) - F(uD)) * ds
        return R


class GradIBP(dolfin_dg.primal.IBP):

    def interior_residual1(self, alpha, u_pen, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # pprint(f"Grad Shape (ufl.jump(G_T_v, n), ufl.avg(F(u))): {ufl.jump(G_T_v, n).ufl_shape, ufl.avg(F(u)).ufl_shape}")
        # pprint(f"Grad Shape ufl.jump(G_T_v, n), alpha * ufl.jump(u_pen, n): {ufl.jump(G_T_v, n).ufl_shape, (alpha * ufl.jump(u_pen, n)).ufl_shape}")
        R = ufl.inner(ufl.jump(G_T_v, n), ufl.avg(F(u))) * dS \
            - ufl.inner(ufl.jump(G_T_v, n), alpha * ufl.jump(u_pen, n)) * dS
        return R

    def exterior_residual1(self, alpha, u_pen, u_penD, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)

        # print(f"Grad Shape G_T_v, ufl.outer(F(uD), n)): {G_T_v.ufl_shape, ufl.outer(F(uD), n).ufl_shape}")
        # print(f"Grad Shape G_T_v, (alpha * (u_pen - u_penD)): {G_T_v.ufl_shape, (alpha * (u_pen - u_penD)).ufl_shape}")
        # quit()
        # TODO: Not sure about this (u - uD) with an n...
        # TODO: cleanup
        R = ufl.inner(G_gamma_T_v, ufl.outer(F(u), n)) * ds \
            - ufl.inner(G_gamma_T_v, alpha * (u_pen - u_penD)) * ds
        return R

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # print(
        #    f"Grad Shape ufl.avg(G_T_v), tensor_jump(F(u), n): {ufl.avg(G_T_v).ufl_shape, tensor_jump(F(u), n).ufl_shape})")

        R = -ufl.inner(ufl.avg(G_T_v), tensor_jump(F(u), n)) * dS
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)
        # Should this be avg(G)?
        R = -ufl.inner(G_gamma_T_v, ufl.outer(F(u) - F(uD), n)) * ds
        return R


class CurlIBP(dolfin_dg.primal.IBP):

    def interior_residual1(self, alpha, u_pen, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # print(f"Grad Shape (ufl.avg(F(u)), cross_jump(G_T_v, n)): {ufl.avg(F(u)).ufl_shape, cross_jump(G_T_v, n).ufl_shape}")
        # print(f"Grad Shape (G_mult(alpha, cross_jump(u_pen, n)), cross_jump(G_T_v, n)): {G_mult(alpha, cross_jump(u_pen, n)).ufl_shape, cross_jump(G_T_v, n).ufl_shape}")

        R = - ufl.inner(ufl.avg(F(u)), cross_jump(G_T_v, n)) * dS \
            + ufl.inner(G_mult(alpha, cross_jump(u_pen, n)), cross_jump(G_T_v, n)) * dS
        # Checked
        return R

    def exterior_residual1(self, alpha, u_pen, u_penD, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        G_gamma = ufl.replace(G, {u: uD})
        G_gamma_T_v = G_T_mult(G_gamma, v)

        # TODO: cleanup this
        R = - ufl.inner(F(u), dg_cross(n, G_gamma_T_v)) * ds \
            + ufl.inner(G_mult(alpha, dg_cross(n, u_pen - u_penD)), dg_cross(n, G_gamma_T_v)) * ds
        #Checked
        return R

    def interior_residual2(self, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # print(
        #    f"Grad Shape cross_jump(F(u), n), ufl.avg(G_T_v): {cross_jump(F(u), n).ufl_shape, ufl.avg(G_T_v).ufl_shape})")

        R = - ufl.inner(cross_jump(F(u), n), ufl.avg(G_T_v)) * dS
        # Checked
        return R

    def exterior_residual2(self, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.ufl_domain())
        u, v = self.u, self.v
        F = self.F
        G = self.G
        G_T_v = G_T_mult(G, v)
        # print(f"Grad Shape (n, G_T_v) {n.ufl_shape, G_T_v.ufl_shape}")
        # print(f"Grad Shape F(u) - F(uD): {(F(u) - F(uD)).ufl_shape}")

        #TODO: the cross product acting on scalar G_T_v is problematic, so reform
        # using a . (b x c) = b . (c x a) = c . (a x b)
        # Original from derivation:
        #     R = ufl.inner(F(u) - F(uD), dg_cross(n, G_T_v)) * ds
        # R = ufl.inner(dg_cross(F(u) - F(uD), n), G_T_v) * ds
        R = - ufl.inner(dg_cross(n, F(u) - F(uD)), G_T_v) * ds
        # R = ufl.inner(F(u) - F(uD), dg_cross(n, G_T_v)) * ds
        #checked
        return R