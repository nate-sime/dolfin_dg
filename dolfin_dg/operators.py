from dolfin_dg.dg_form import DGFemViscousTerm, homogeneity_tensor
from ufl import CellVolume, FacetArea, grad, inner, TestFunction
from dolfin import Constant, Measure, FacetNormal

class DGBC:

    def __init__(self, boundary, function):
        self.__boundary = boundary
        self.__function = function

    def get_boundary(self):
        return self.__boundary

    def get_function(self):
        return self.__function

    def __repr__(self):
        return '%s(%s: %s)' % (self.__class__.__name__,
                               self.get_boundary(),
                               self.get_function())


class DGDirichletBC(DGBC):

    def __init__(self, boundary, function):
        DGBC.__init__(self, boundary, function)


class DGNeumannBC(DGBC):

    def __init__(self, boundary, function):
        DGBC.__init__(self, boundary, function)


class FemFormulation:

    def __init__(self, mesh, fspace, bcs, **kwargs):
        if not hasattr(bcs, '__len__'):
            bcs = [bcs]
        self.mesh = mesh
        self.fspace = fspace
        self.dirichlet_bcs = [bc for bc in bcs if isinstance(bc, DGDirichletBC)]
        self.neumann_bcs = [bc for bc in bcs if isinstance(bc, DGNeumannBC)]

    def generate_fem_formulation(self, u):
        raise NotImplementedError('Function not yet implemented')


class EllipticOperator(FemFormulation):

    def __init__(self, mesh, fspace, bcs, F_v, C_IP=10.0):
        FemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_v = F_v
        self.C_IP = C_IP

    def generate_fem_formulation(self, u, v, dx=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)

        h = CellVolume(self.mesh)/FacetArea(self.mesh)
        n = FacetNormal(self.mesh)
        sigma = self.C_IP*Constant(max(self.fspace.ufl_element().degree()**2, 1))/h
        G = homogeneity_tensor(self.F_v, u)
        vt = DGFemViscousTerm(self.F_v, u, v, Constant(10.0)*sigma, G, n)

        residual = inner(self.F_v(u, grad(u)), grad(v))*dx

        for dbc in self.dirichlet_bcs:
            residual += vt.exterior_residual(dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += vt.neumann_residual(dbc.get_function(), dbc.get_boundary())

        return residual


class PoissonEquation(EllipticOperator):

    def __init__(self, mesh, fspace, bcs, kappa=1):
        def F_v(u, grad_u):
            return kappa*grad_u

        EllipticOperator.__init__(self, mesh, fspace, bcs, F_v)