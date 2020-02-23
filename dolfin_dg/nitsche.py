import ufl
import dolfin_dg
from dolfin_dg import DGFemSIPG, homogeneity_tensor
from dolfin_dg.dg_form import DGFemStokesTerm


class NitscheBoundary:

    def __init__(self, F_v, u, v, C_IP=None, DGFemClass=None):

        if C_IP is None:
            C_IP = dolfin_dg.dg_form.generate_default_sipg_penalty_term(u)

        G = homogeneity_tensor(F_v, u)
        n = ufl.FacetNormal(u.ufl_domain())

        if DGFemClass is None:
            DGFemClass = DGFemSIPG
        vt = DGFemClass(F_v, u, v, C_IP, G, n)

        self.vt = vt

    def nitsche_bc_residual(self, u_bc, ds):
        return self.vt.exterior_residual(u_bc, ds)

    def nitsche_bc_residual_on_interior(self, u_bc, dS):
        return self.vt.exterior_residual_on_interior(u_bc, dS)


class StokesNitscheBoundary:

    def __init__(self, F_v, u, p, v, q, C_IP=None, delta=-1,
                 block_form=False):

        if C_IP is None:
            C_IP = dolfin_dg.dg_form.generate_default_sipg_penalty_term(u)

        G = homogeneity_tensor(F_v, u)
        n = ufl.FacetNormal(u.ufl_domain())

        vt = DGFemStokesTerm(F_v, u, p, v, q, C_IP, G, n, delta,
                             block_form=block_form)

        self.vt = vt

    def nitsche_bc_residual(self, u_bc, ds):
        return self.vt.exterior_residual(u_bc, ds)

    def nitsche_bc_residual_on_interior(self, u_bc, dS):
        return self.vt.exterior_residual_on_interior(u_bc, dS)

    def slip_nitsche_bc_residual(self, u_bc, f2, ds):
        return self.vt.slip_exterior_residual(u_bc, f2, ds)

    def slip_nitsche_bc_residual_on_interior(self, u_bc, f2, dS):
        return self.vt.slip_exterior_residual_on_interior(u_bc, f2, dS)
