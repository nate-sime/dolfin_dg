from dolfin import *
from dolfin_dg import DGFemSIPG, homogeneity_tensor


class NitscheBoundary:

    def __init__(self, F_v, u, v, C_IP=None, DGFemClass=None):

        if C_IP is None:
            h = CellDiameter(u.function_space().mesh())
            C_IP = Constant(20.0 * max(u.function_space().ufl_element().degree() ** 2, 1)) / h

        G = homogeneity_tensor(F_v, u)
        n = FacetNormal(u.function_space().mesh())

        if DGFemClass is None:
            DGFemClass = DGFemSIPG
        vt = DGFemClass(F_v, u, v, C_IP, G, n)

        self.vt = vt

    def nistche_bc_residual(self, u_bc, ds):
        return self.vt.exterior_residual(u_bc, ds)
