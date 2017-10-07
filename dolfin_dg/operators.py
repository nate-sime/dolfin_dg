import inspect

from dolfin_dg.dg_form import DGFemViscousTerm, homogeneity_tensor, tangent_jump, tensor_jump, ufl_adhere_transpose, \
    DGFemCurlTerm, DGFemSIPG
import ufl
from ufl import CellVolume, FacetArea, grad, inner, \
    div, jump, avg, curl, cross, Max, dot, as_vector, as_matrix, sqrt, tr, Identity, variable, diff, exp
from dolfin import Constant, Measure, FacetNormal, split

from dolfin_dg.fluxes import LocalLaxFriedrichs


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


class DGFemFormulation:

    def __init__(self, mesh, fspace, bcs, **kwargs):
        if not hasattr(bcs, '__len__'):
            bcs = [bcs]
        self.mesh = mesh
        self.fspace = fspace
        self.dirichlet_bcs = [bc for bc in bcs if isinstance(bc, DGDirichletBC)]
        self.neumann_bcs = [bc for bc in bcs if isinstance(bc, DGNeumannBC)]

    def generate_fem_formulation(self, u):
        raise NotImplementedError('Function not yet implemented')


class EllipticOperator(DGFemFormulation):

    def __init__(self, mesh, fspace, bcs, F_v, C_IP=10.0):
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_v = F_v
        self.C_IP = C_IP

    def generate_fem_formulation(self, u, v, dx=None, dS=None, vt=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        h = CellVolume(self.mesh)/FacetArea(self.mesh)
        n = FacetNormal(self.mesh)
        sigma = self.C_IP*Constant(max(self.fspace.ufl_element().degree()**2, 1))/h
        G = homogeneity_tensor(self.F_v, u)

        if vt is None:
            vt = DGFemSIPG(self.F_v, u, v, sigma, G, n)

        if inspect.isclass(vt):
            vt = vt(self.F_v, u, v, sigma, G, n)

        assert(isinstance(vt, DGFemViscousTerm))

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


class MaxwellOperator(DGFemFormulation):

    def __init__(self, mesh, fspace, bcs, F_m, C_IP=10.0):
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_m = F_m
        self.C_IP = C_IP

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        h = CellVolume(self.mesh)/FacetArea(self.mesh)
        n = FacetNormal(self.mesh)
        sigma = self.C_IP*Constant(max(self.fspace.ufl_element().degree()**2, 1))/h
        curl_u = variable(curl(u))
        G = diff(self.F_m(u, curl_u), curl_u)
        ct = DGFemCurlTerm(self.F_m, u, v, sigma, G, n)

        residual = inner(self.F_m(u, curl(u)), curl(v))*dx
        residual += ct.interior_residual(dS)

        for dbc in self.dirichlet_bcs:
            residual += ct.exterior_residual(dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += ct.neumann_residual(dbc.get_function(), dbc.get_boundary())

        return residual


class HyperbolicOperator(DGFemFormulation):

    def __init__(self, mesh, V, bcs,
                 F_c=lambda u: u,
                 H=LocalLaxFriedrichs(lambda u, n: inner(u, n))):
        DGFemFormulation.__init__(self, mesh, V, bcs)
        self.F_c = F_c
        self.H = H

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)
        n = FacetNormal(self.mesh)

        residual = -inner(self.F_c(u), grad(v))*dx

        self.H.setup(self.F_c, u("+"), u("-"), n("+"))
        residual += inner(self.H.interior(self.F_c, u('+'), u('-'), n('+')), (v('+') - v('-')))*dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, gD, n)
            residual += inner(self.H.exterior(self.F_c, u, gD, n), v)*dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v)*dSN

        return residual


class SpacetimeBurgersOperator(
    HyperbolicOperator):

    def __init__(self, mesh, V, bcs, flux=None):

        def F_c(u):
            return as_vector((u**2/2, u))

        if flux is None:
            flux = LocalLaxFriedrichs(lambda u, n: u*n[0] + n[1])

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, flux)


class CompressibleEulerOperator(
    HyperbolicOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4):
        gamma = Constant(gamma)
        dim = mesh.geometry().dim()

        def F_c(U):
            if dim == 2:
                rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
                p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
                H = E + p/rho
                res = as_matrix([[rho*u1, rho*u2],
                                 [rho*u1**2 + p, rho*u1*u2],
                                 [rho*u1*u2, rho*u2**2 + p],
                                 [rho*H*u1, rho*H*u2]])
            elif dim == 3:
                rho, u1, u2, u3, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0], U[4]/U[0]
                p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2 + u3**2))
                H = E + p/rho
                res = as_matrix([[rho*u1, rho*u2, rho*u3],
                                 [rho*u1**2 + p, rho*u1*u2, rho*u1*u3],
                                 [rho*u1*u2, rho*u2**2 + p, rho*u2*u3],
                                 [rho*u1*u3, rho*u2*u3, rho*u3**2 + p],
                                 [rho*H*u1, rho*H*u2, rho*H*u3]])

            return res

        def alpha(U, n):
            if dim == 2:
                rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
                p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
                u = as_vector([u1, u2])
                c = sqrt(gamma*p/rho)
                lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            elif dim == 3:
                rho, u1, u2, u3, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0], U[4]/U[0]
                p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2 + u3**2))
                u = as_vector([u1, u2, u3])
                c = sqrt(gamma*p/rho)
                lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, LocalLaxFriedrichs(alpha))


class CompressibleNavierStokesOperator(
    EllipticOperator,
    CompressibleEulerOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4, mu=1.0, Pr=0.72):
        gamma = Constant(gamma)
        mu = Constant(mu)
        Pr = Constant(Pr)
        dim = mesh.geometry().dim()

        def F_v(U, grad_U):
            if dim == 2:
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
            elif dim == 3:
                rho, rhou1, rhou2, rhou3, rhoE = U
                u1, u2, u3, E = rhou1/rho, rhou2/rho, rhou3/rho, rhoE/rho
                u = as_vector((u1, u2, u3))

                grad_rho = grad_U[0, :]

                grad_xi1 = as_vector([grad_U[1, 0], grad_U[1, 1], grad_U[1, 2]])
                grad_xi2 = as_vector([grad_U[2, 0], grad_U[2, 1], grad_U[2, 2]])
                grad_xi3 = as_vector([grad_U[3, 0], grad_U[3, 1], grad_U[3, 2]])
                grad_u1 = (grad_xi1 - u1*grad_rho)/rho
                grad_u2 = (grad_xi2 - u2*grad_rho)/rho
                grad_u3 = (grad_xi3 - u3*grad_rho)/rho
                grad_u = as_matrix([[grad_u1[0], grad_u1[1], grad_u1[2]],
                                    [grad_u2[0], grad_u2[1], grad_u2[2]],
                                    [grad_u3[0], grad_u3[1], grad_u3[2]]])

                grad_eta = grad_U[4, :]
                grad_E = (grad_eta - E*grad_rho)/rho

                tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(3))
                K_grad_T = mu*gamma/Pr*(grad_E - dot(u, grad_u))

                return as_matrix([[0.0, 0.0, 0.0],
                                  [tau[0, 0], tau[0, 1], tau[0, 2]],
                                  [tau[1, 0], tau[1, 1], tau[1, 2]],
                                  [tau[2, 0], tau[2, 1], tau[2, 2]],
                                  [dot(tau[0, :], u) + K_grad_T[0], (dot(tau[1, :], u)) + K_grad_T[1], (dot(tau[2, :], u)) + K_grad_T[2]]])


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


def V_to_U(V, gamma):
    V1, V2, V3, V4 = V
    U = as_vector([-V4, V2, V3, 1 - 0.5*(V2**2 + V3**2)/V4])
    s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
    rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*exp(-s/(gamma-1))
    U = U*rhoi
    return U


class CompressibleEulerOperatorEntropyFormulation(
    HyperbolicOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4):
        gamma = Constant(gamma)

        def F_c(V):
            V = variable(V)
            U = V_to_U(V, gamma)
            rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
            p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
            H = E + p/rho
            res = as_matrix([[rho*u1, rho*u2],
                             [rho*u1**2 + p, rho*u1*u2],
                             [rho*u1*u2, rho*u2**2 + p],
                             [rho*H*u1, rho*H*u2]])
            return res

        def alpha(V, n):
            U = V_to_U(V, gamma)
            rho, u1, u2, E = U[0], U[1]/U[0], U[2]/U[0], U[3]/U[0]
            p = (gamma - 1.0)*rho*(E - 0.5*(u1**2 + u2**2))
            u = as_vector([u1, u2])
            c = sqrt(gamma*p/rho)
            lambdas = [dot(u, n) - c, dot(u, n), dot(u, n), dot(u, n) + c]
            lambdas = list(map(abs, lambdas))
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, LocalLaxFriedrichs(alpha))


class CompressibleNavierStokesOperatorEntropyFormulation(
    EllipticOperator,
    CompressibleEulerOperatorEntropyFormulation):

    def __init__(self, mesh, V, bcs, gamma=1.4, mu=1.0, Pr=0.72):
        gamma = Constant(gamma)
        mu = Constant(mu)
        Pr = Constant(Pr)

        def F_v(V, grad_V):
            V = variable(V)
            U = V_to_U(V, gamma)
            dudv = diff(U, V)
            grad_U = dot(dudv, grad_V)

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

        CompressibleEulerOperatorEntropyFormulation.__init__(self, mesh, V, bcs, gamma)
        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        residual = EllipticOperator.generate_fem_formulation(self, u, v, dx, dS)
        residual += CompressibleEulerOperatorEntropyFormulation.generate_fem_formulation(self, u, v, dx, dS)

        return residual