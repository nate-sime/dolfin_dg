import ufl
from dolfin import *
from dolfin_dg import DGFemSIPG, homogeneity_tensor
from dolfin_dg.dg_form import DGFemStokesTerm


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

    def nitsche_bc_residual(self, u_bc, ds):
        return self.vt.exterior_residual(u_bc, ds)

    def nitsche_bc_residual_on_interior(self, u_bc, dS):
        return self.vt.exterior_residual_on_interior(u_bc, dS)


class StokesNitscheBoundary:

    def __init__(self, F_v, u, p, v, q, C_IP=None, delta=-1):

        def get_terminal_operand_coefficient(u):
            if not isinstance(u, ufl.Coefficient):
                return get_terminal_operand_coefficient(
                    u.ufl_operands[0])
            return u

        U = get_terminal_operand_coefficient(u)
        mesh = U.function_space().mesh()

        if C_IP is None:
            degree = U.sub(0).function_space().ufl_element().degree()
            h = CellDiameter(mesh)
            C_IP = Constant(20.0 * max(degree ** 2, 1)) / h

        G = homogeneity_tensor(F_v, u)
        n = FacetNormal(mesh)

        vt = DGFemStokesTerm(F_v, u, p, v, q, C_IP, G, n, delta)

        self.vt = vt

    def nitsche_bc_residual(self, u_bc, ds):
        return self.vt.exterior_residual(u_bc, ds)

    def nitsche_bc_residual_on_interior(self, u_bc, dS):
        return self.vt.exterior_residual_on_interior(u_bc, dS)
