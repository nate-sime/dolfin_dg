from dolfin_dg.dg_form import DGFemViscousTerm, homogeneity_tensor, tangent_jump, tensor_jump, ufl_adhere_transpose
from dolfin_dg.fluxes import lax_friedrichs_flux
from ufl import CellVolume, FacetArea, grad, inner, TestFunction, div, jump, avg, curl, cross, Max, dot
from dolfin import Constant, Measure, FacetNormal, split

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

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        h = CellVolume(self.mesh)/FacetArea(self.mesh)
        n = FacetNormal(self.mesh)
        sigma = self.C_IP*Constant(max(self.fspace.ufl_element().degree()**2, 1))/h
        G = homogeneity_tensor(self.F_v, u)
        vt = DGFemViscousTerm(self.F_v, u, v, sigma, G, n)

        residual = inner(self.F_v(u, grad(u)), grad(v))*dx
        residual += vt.interior_residual(dS)

        for dbc in self.dirichlet_bcs:
            residual += vt.exterior_residual(dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += vt.neumann_residual(dbc.get_function(), dbc.get_boundary())

        return residual


class PoissonOperator(EllipticOperator):

    def __init__(self, mesh, fspace, bcs, kappa=1):
        def F_v(u, grad_u):
            return kappa*grad_u

        EllipticOperator.__init__(self, mesh, fspace, bcs, F_v)


class StokesEquations(EllipticOperator):

    def __init__(self, mesh, V, bcs, mu=1):
        def F_v(u, grad_u):
            return mu*grad_u

        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None):

        u, p = split(u)
        v, q = split(v)
        n = FacetNormal(self.mesh)

        def B(v, q):
            return -q*div(v)*dx + avg(q)*jump(v, n)*dS \
                   + sum(q*inner(v, n)*bc.get_boundary() for bc in self.dirichlet_bcs)

        residual = EllipticOperator.generate_fem_formulation(self, u, v)
        residual += B(v, p) - B(u, q)
        residual -= sum(-q*inner(bc.get_function(), n)*bc.get_boundary() for bc in self.dirichlet_bcs)

        return residual


class MaxwellAndInvolutionOperator(FemFormulation):

    def __init__(self, mesh, V, bcs, mu_r=1, eps_r=1, C_IP=10.0):
        FemFormulation.__init__(self, mesh, V, bcs)
        self.mu_r = mu_r
        self.eps_r = eps_r
        self.C_IP = C_IP

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        E, p = split(u)
        F, q = split(v)

        h = CellVolume(self.mesh)/FacetArea(self.mesh)
        n = FacetNormal(self.mesh)

        mu_r, eps_r = self.mu_r, self.eps_r
        sigma = self.C_IP*Constant(max(self.fspace.ufl_element().degree()**2, 1))/h
        a_disp = self.C_IP/mu_r*sigma
        c = self.C_IP*eps_r*sigma

        def A(u, v):
            val = 1.0/mu_r*inner(curl(u), curl(v)) * dx \
                  - inner(tangent_jump(u, n), avg(1.0/mu_r*curl(v))) * dS \
                  - inner(tangent_jump(v, n), avg(1.0/mu_r*curl(u))) * dS \
                  + a_disp('+') * inner(tangent_jump(u, n), tangent_jump(v, n)) * dS
            for bc in self.dirichlet_bcs:
                dSD = bc.get_boundary()
                val += - inner(cross(n, u), 1.0/mu_r*curl(v)) * dSD \
                       - inner(cross(n, v), 1.0/mu_r*curl(u)) * dSD \
                       + inner(cross(n, u), cross(n, a_disp * v)) * dSD
            return val

        def B(v, p):
            val = -eps_r*inner(v, grad(p))*dx + inner(avg(eps_r*v), jump(p, n))*dS
            for bc in self.dirichlet_bcs:
                val += inner(eps_r*v, p*n)*bc.get_boundary()
            return val

        def C(p, q):
            val = c('+')*inner(jump(p), jump(q))*dS
            for bc in self.dirichlet_bcs:
                val += c*p*q*bc.get_boundary()
            return val

        def L(v):
            val = 0
            for g, dS_p in [(bc.get_function(), bc.get_boundary()) for bc in self.dirichlet_bcs]:
                val += - inner(cross(n, g), 1.0/mu_r*curl(v))*dS_p + a_disp*inner(cross(n, g), cross(n, v))*dS_p
            for gN, dS_pN in [(bc.get_function(), bc.get_boundary()) for bc in self.neumann_bcs]:
                val -= inner(gN, v)*dS_pN
            return val

        residual = A(E, F) + B(F, p) - L(F) + B(E, q) - C(p, q)

        return residual


class HyperbolicOperator(FemFormulation):

    def __init__(self, mesh, V, bcs, F_c=lambda u: u, alpha=lambda u, n: inner(u, n)):
        FemFormulation.__init__(self, mesh, V, bcs)
        self.F_c = F_c
        self.alpha = alpha

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = FacetNormal(self.mesh)

        residual = -inner(self.F_c(u), grad(v))*dx

        # The following is just dot(H(u_p, u_m, n), v_p - v_m) on the interior as jumps and averages
        avg_F = ufl_adhere_transpose(avg(self.F_c(u)))
        alpha_interior = Max(abs(self.alpha(u, n))('+'), abs(self.alpha(u, n))('-'))
        residual += inner(avg_F + 0.5*alpha_interior*tensor_jump(u, n), tensor_jump(v, n))*dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            bdry_alpha = Max(abs(self.alpha(u, n)), abs(self.alpha(gD, n)))
            residual += inner(0.5*(dot(self.F_c(u), n) + dot(self.F_c(gD), n) + bdry_alpha*(u - gD)), v)*dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v)*dSN

        return residual