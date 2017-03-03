from dolfin_dg.dg_form import DGFemViscousTerm, homogeneity_tensor, tangent_jump, tensor_jump, ufl_adhere_transpose
import ufl
from ufl import CellVolume, FacetArea, grad, inner, \
    div, jump, avg, curl, cross, Max, dot, as_vector, as_matrix, sqrt, tr, Identity
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

    def max_abs_eigenvalues(self, lambdas):
        if isinstance(lambdas, ufl.core.expr.Expr):
            return abs(lambdas)

        assert isinstance(lambdas, (list, tuple))

        lambdas = map(abs, lambdas)
        alpha = Max(lambdas[0], lambdas[1])
        for j in range(2, len(lambdas)):
            alpha = Max(alpha, lambdas[j])
        return alpha


    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = FacetNormal(self.mesh)
        alpha = self.max_abs_eigenvalues(self.alpha(u, n))

        residual = -inner(self.F_c(u), grad(v))*dx

        # The following is just dot(H(u_p, u_m, n), v_p - v_m) on the interior as jumps and averages
        avg_F = ufl_adhere_transpose(avg(self.F_c(u)))
        alpha_interior = Max(alpha('+'), alpha('-'))
        residual += inner(avg_F + 0.5*alpha_interior*tensor_jump(u, n), tensor_jump(v, n))*dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            alpha_bc = self.max_abs_eigenvalues(self.alpha(gD, n))
            bdry_alpha = Max(alpha, alpha_bc)
            residual += inner(0.5*(dot(self.F_c(u), n) + dot(self.F_c(gD), n) + bdry_alpha*(u - gD)), v)*dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v)*dSN

        return residual


class SpacetimeBurgersOperator(HyperbolicOperator):

    def __init__(self, mesh, V, bcs):

        def F_c(u):
            return as_vector((u**2/2, u))

        def alpha(u, n):
            return u*n[0] + n[1]

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, alpha)


class CompressibleEulerOperator(HyperbolicOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4):
        gamma = Constant(gamma)

        def F_c(U):
            rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
            p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
            H = E + p/rho
            res = as_matrix([[rho*u1, rho*u2],
                             [rho*u1**2 + p, rho*u1*u2],
                             [rho*u1*u2, rho*u2**2 + p],
                             [rho*H*u1, rho*H*u2]])
            return res

        def alpha(U, n):
            rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
            p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
            u = as_vector([u1, u2])
            c = sqrt(gamma*p/rho)
            lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, alpha)


class CompressibleNavierStokesOperator(EllipticOperator, CompressibleEulerOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4, mu=1.0, Pr=0.72):
        gamma = Constant(gamma)
        mu = Constant(mu)
        Pr = Constant(Pr)

        def F_v(U, grad_U):
            rho, rhou1, rhou2, rhoE = U
            u1, u2, E = rhou1/rho, rhou2/rho, rhoE/rho
            u = as_vector((u1, u2))

            grad_rho = grad_U[0, :]

            grad_xi1 = as_vector([grad_U[1, 0], grad_U[1, 1]])
            grad_xi2 = as_vector([grad_U[2, 0], grad_U[2, 1]])
            grad_u1 = (grad_xi1 - u1*grad_rho)/rho
            grad_u2 = (grad_xi2 - u2*grad_rho)/rho
            grad_u = as_matrix([[grad_u1[0], grad_u1[1]],
                                [grad_u2[0], grad_u2[1]]])

            grad_eta = grad_U[3, :]
            grad_E = (grad_eta - E*grad_rho)/rho

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(2))
            K_grad_T = mu*gamma/Pr*(grad_E - dot(u, grad_u))

            return as_matrix([[0.0, 0.0],
                              [tau[0, 0], tau[0, 1]],
                              [tau[1, 0], tau[1, 1]],
                              [dot(tau[0, :], u) + K_grad_T[0], (dot(tau[1, :], u)) + K_grad_T[1]]])

        CompressibleEulerOperator.__init__(self, mesh, V, bcs, gamma)
        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        residual = EllipticOperator.generate_fem_formulation(self, u, v, dx, dS)
        residual += CompressibleEulerOperator.generate_fem_formulation(self, u, v, dx, dS)

        return residual