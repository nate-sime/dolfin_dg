from dolfin import *
from dolfin_dg import DGFemSIPG, homogeneity_tensor


class NitscheBoundary:

    def __init__(self, F_v, u, v):

        G = homogeneity_tensor(F_v, u)
        h = CellDiameter(u.function_space().mesh())
        C_IP = Constant(20.0)
        n = FacetNormal(u.function_space().mesh())
        sigma = C_IP * Constant(max(u.function_space().ufl_element().degree() ** 2, 1)) / h
        vt = DGFemSIPG(F_v, u, v, sigma, G, n)

        self.vt = vt

    def weak_nitsche_bc_form(self, ds, u_bc):
        return self.vt.exterior_residual(ds, u_bc)