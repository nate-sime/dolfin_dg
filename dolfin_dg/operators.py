import inspect

import ufl
from ufl import (
    grad, inner, curl, dot, as_vector, tr, Identity, variable, diff, exp,
    Measure, FacetNormal
)

from dolfin_dg import aero, generate_default_sipg_penalty_term
from dolfin_dg.dg_form import (
    DGFemTerm, homogeneity_tensor, DGFemCurlTerm, DGFemSIPG, DGFemStokesTerm
)
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
    pass


class DGNeumannBC(DGBC):
    pass


class DGDirichletNormalBC(DGBC):
    pass


class DGAdiabticWallBC(DGBC):
    pass


class DGFemFormulation:

    def __init__(self, mesh, fspace, bcs, **kwargs):
        if not hasattr(bcs, '__len__'):
            bcs = [bcs]
        self.mesh = mesh
        self.fspace = fspace
        self.dirichlet_bcs = [bc for bc in bcs if isinstance(bc, DGDirichletBC)]
        self.neumann_bcs = [bc for bc in bcs if isinstance(bc, DGNeumannBC)]

    def generate_fem_formulation(self, u, v, dx=None, dS=None, vt=None):
        raise NotImplementedError('Function not yet implemented')


class EllipticOperator(DGFemFormulation):

    def __init__(self, mesh, fspace, bcs, F_v):
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_v = F_v

    def generate_fem_formulation(self, u, v, dx=None, dS=None, vt=None,
                                 penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.mesh.ufl_domain())
        G = homogeneity_tensor(self.F_v, u)

        if penalty is None:
            penalty = generate_default_sipg_penalty_term(u)

        if vt is None:
            vt = DGFemSIPG(self.F_v, u, v, penalty, G, n)

        if inspect.isclass(vt):
            vt = vt(self.F_v, u, v, penalty, G, n)

        assert(isinstance(vt, DGFemTerm))

        residual = inner(self.F_v(u, grad(u)), grad(v))*dx
        residual += vt.interior_residual(dS)

        for dbc in self.dirichlet_bcs:
            residual += vt.exterior_residual(
                dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += vt.neumann_residual(
                dbc.get_function(), dbc.get_boundary())

        return residual


class PoissonOperator(EllipticOperator):

    def __init__(self, mesh, fspace, bcs, kappa=1):
        def F_v(u, grad_u):
            return kappa*grad_u

        EllipticOperator.__init__(self, mesh, fspace, bcs, F_v)


class MaxwellOperator(DGFemFormulation):

    def __init__(self, mesh, fspace, bcs, F_m):
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_m = F_m

    def generate_fem_formulation(self, u, v, dx=None, dS=None, penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.mesh.ufl_domain())
        curl_u = variable(curl(u))
        G = diff(self.F_m(u, curl_u), curl_u)
        penalty = generate_default_sipg_penalty_term(u)

        ct = DGFemCurlTerm(self.F_m, u, v, penalty, G, n)

        residual = inner(self.F_m(u, curl(u)), curl(v))*dx
        residual += ct.interior_residual(dS)

        for dbc in self.dirichlet_bcs:
            residual += ct.exterior_residual(
                dbc.get_function(), dbc.get_boundary())

        for dbc in self.neumann_bcs:
            residual += ct.neumann_residual(
                dbc.get_function(), dbc.get_boundary())

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

        n = ufl.FacetNormal(self.mesh.ufl_domain())

        F_c_eval = self.F_c(u)
        if len(F_c_eval.ufl_shape) == 0:
            F_c_eval = as_vector((F_c_eval,))
        residual = -inner(F_c_eval, grad(v))*dx

        self.H.setup(self.F_c, u('+'), u('-'), n('+'))
        residual += inner(self.H.interior(self.F_c, u('+'), u('-'), n('+')),
                          (v('+') - v('-')))*dS

        for bc in self.dirichlet_bcs:
            gD = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, gD, n)
            residual += inner(self.H.exterior(self.F_c, u, gD, n), v)*dSD

        for bc in self.neumann_bcs:
            dSN = bc.get_boundary()

            residual += inner(dot(self.F_c(u), n), v)*dSN

        return residual


class SpacetimeBurgersOperator(HyperbolicOperator):

    def __init__(self, mesh, V, bcs, flux=None):

        def F_c(u):
            return as_vector((u**2/2, u))

        if flux is None:
            flux = LocalLaxFriedrichs(lambda u, n: u*n[0] + n[1])

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c, flux)


class CompressibleEulerOperator(HyperbolicOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4):
        try:
            dim = mesh.geometry().dim()
        except AttributeError:
            dim = mesh.geometric_dimension()

        def F_c(U):
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            H = aero.enthalpy(U, gamma=gamma)

            inertia = rho*ufl.outer(u, u) + p*Identity(dim)
            res = ufl.as_tensor([rho*u,
                                 *[inertia[d, :] for d in range(dim)],
                                 rho*H*u])
            return res

        def alpha(U, n):
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            c = aero.speed_of_sound(p, rho, gamma=gamma)
            lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c,
                                    LocalLaxFriedrichs(alpha))


class CompressibleNavierStokesOperator(EllipticOperator,
                                       CompressibleEulerOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4, mu=1.0, Pr=0.72):
        try:
            dim = mesh.geometry().dim()
        except AttributeError:
            dim = mesh.geometric_dimension()

        if not hasattr(bcs, '__len__'):
            bcs = [bcs]
        self.adiabatic_wall_bcs = [bc for bc in bcs
                                   if isinstance(bc, DGAdiabticWallBC)]

        def F_v(U, grad_U):
            rho, rhou, rhoE = aero.conserved_variables(U)
            u = rhou/rho

            grad_rho = grad_U[0, :]
            grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim + 1)])
            grad_rhoE = grad_U[-1, :]

            # Quotient rule to find grad(u) and grad(E)
            grad_u = (grad_rhou*rho - ufl.outer(rhou, grad_rho))/rho**2
            grad_E = (grad_rhoE*rho - rhoE*grad_rho)/rho**2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(dim))
            K_grad_T = mu*gamma/Pr*(grad_E - dot(u, grad_u))

            res = ufl.as_tensor([ufl.zero(dim),
                                 *(tau[d, :] for d in range(dim)),
                                 tau * u + K_grad_T])
            return res

        # Specialised adiabatic wall BC
        def F_v_adiabatic(U, grad_U):
            rho, rhou, rhoE = aero.conserved_variables(U)
            u = rhou/rho

            grad_rho = grad_U[0, :]
            grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim + 1)])
            grad_u = (grad_rhou * rho - ufl.outer(rhou, grad_rho)) / rho ** 2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(dim))

            res = ufl.as_tensor([ufl.zero(dim),
                                 *(tau[d, :] for d in range(dim)),
                                 tau * u])
            return res

        self.F_v_adiabatic = F_v_adiabatic

        CompressibleEulerOperator.__init__(self, mesh, V, bcs, gamma)
        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None, penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        residual = EllipticOperator.generate_fem_formulation(
            self, u, v, dx=dx, dS=dS, penalty=penalty)
        residual += CompressibleEulerOperator.generate_fem_formulation(
            self, u, v, dx=dx, dS=dS)

        # Specialised adiabatic wall boundary condition
        for bc in self.adiabatic_wall_bcs:
            n = FacetNormal(self.mesh)

            u_gamma = bc.get_function()
            dSD = bc.get_boundary()

            self.H.setup(self.F_c, u, u_gamma, n)
            residual += inner(self.H.exterior(self.F_c, u, u_gamma, n), v)*dSD

            if penalty is None:
                penalty = generate_default_sipg_penalty_term(u)
            G_adiabitic = homogeneity_tensor(self.F_v_adiabatic, u)
            vt_adiabatic = DGFemSIPG(
                self.F_v_adiabatic, u, v, penalty, G_adiabitic, n)

            residual += vt_adiabatic.exterior_residual(u_gamma, dSD)

        return residual


def V_to_U(V, gamma):
    V1, V2, V3, V4 = V
    U = as_vector([-V4, V2, V3, 1 - 0.5*(V2**2 + V3**2)/V4])
    s = gamma - V1 + (V2**2 + V3**2)/(2*V4)
    rhoi = ((gamma - 1)/((-V4)**gamma))**(1.0/(gamma-1))*exp(-s/(gamma-1))
    U = U*rhoi
    return U


class CompressibleEulerOperatorEntropyFormulation(HyperbolicOperator):

    def __init__(self, mesh, V, bcs, gamma=1.4):

        try:
            dim = mesh.geometry().dim()
        except AttributeError:
            dim = mesh.geometric_dimension()

        def F_c(V):
            V = variable(V)
            U = V_to_U(V, gamma)
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            H = aero.enthalpy(U, gamma=gamma)

            inertia = rho*ufl.outer(u, u) + p*Identity(dim)
            res = ufl.as_tensor([rho*u,
                                 *[inertia[d, :] for d in range(dim)],
                                 rho*H*u])
            return res

        def alpha(V, n):
            U = V_to_U(V, gamma)
            rho, u, E = aero.flow_variables(U)
            p = aero.pressure(U, gamma=gamma)
            c = aero.speed_of_sound(p, rho, gamma=gamma)
            lambdas = [dot(u, n) - c, dot(u, n), dot(u, n) + c]
            return lambdas

        HyperbolicOperator.__init__(self, mesh, V, bcs, F_c,
                                    LocalLaxFriedrichs(alpha))


class CompressibleNavierStokesOperatorEntropyFormulation(
        EllipticOperator,
        CompressibleEulerOperatorEntropyFormulation):

    def __init__(self, mesh, V, bcs, gamma=1.4, mu=1.0, Pr=0.72):

        try:
            dim = mesh.geometry().dim()
        except AttributeError:
            dim = mesh.geometric_dimension()

        def F_v(V, grad_V):
            V = variable(V)
            U = V_to_U(V, gamma)
            dudv = diff(U, V)
            grad_U = dot(dudv, grad_V)

            rho, rhou, rhoE = aero.conserved_variables(U)
            rho, u, E = aero.flow_variables(U)

            grad_rho = grad_U[0, :]
            grad_rhou = ufl.as_tensor([grad_U[j, :] for j in range(1, dim + 1)])
            grad_rhoE = grad_U[-1, :]

            grad_u = (grad_rhou*rho - ufl.outer(rhou, grad_rho))/rho**2
            grad_E = (grad_rhoE*rho - rhoE*grad_rho)/rho**2

            tau = mu*(grad_u + grad_u.T - 2.0/3.0*(tr(grad_u))*Identity(dim))
            K_grad_T = mu*gamma/Pr*(grad_E - dot(u, grad_u))

            res = ufl.as_tensor([ufl.zero(dim),
                                 *(tau[d, :] for d in range(dim)),
                                 tau * u + K_grad_T])
            return res

        CompressibleEulerOperatorEntropyFormulation.__init__(
            self, mesh, V, bcs, gamma)
        EllipticOperator.__init__(self, mesh, V, bcs, F_v)

    def generate_fem_formulation(self, u, v, dx=None, dS=None, penalty=None):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        residual = EllipticOperator \
            .generate_fem_formulation(self, u, v, dx=dx, dS=dS, penalty=penalty)
        residual += CompressibleEulerOperatorEntropyFormulation \
            .generate_fem_formulation(self, u, v, dx=dx, dS=dS)

        return residual


class StokesOperator(DGFemFormulation):

    def __init__(self, mesh, fspace, bcs, F_v):
        DGFemFormulation.__init__(self, mesh, fspace, bcs)
        self.F_v = F_v

    def generate_fem_formulation(self, u, v, p, q, dx=None, dS=None,
                                 penalty=None, block_form=False):
        if dx is None:
            dx = Measure('dx', domain=self.mesh)
        if dS is None:
            dS = Measure('dS', domain=self.mesh)

        n = ufl.FacetNormal(self.mesh.ufl_domain())
        G = homogeneity_tensor(self.F_v, u)
        delta = -1

        if penalty is None:
            penalty = generate_default_sipg_penalty_term(u)

        vt = DGFemStokesTerm(self.F_v, u, p, v, q, penalty, G, n, delta,
                             block_form=block_form)

        residual = [ufl.inner(self.F_v(u, grad(u)), grad(v))*dx,
                    q*ufl.div(u)*dx]
        if not block_form:
            residual = sum(residual)

        def _add_to_residual(residual, r):
            if block_form:
                for j in range(len(r)):
                    residual[j] += r[j]
            else:
                residual += r
            return residual

        residual = _add_to_residual(residual, vt.interior_residual(dS))

        for dbc in self.dirichlet_bcs:
            residual = _add_to_residual(
                residual, vt.exterior_residual(dbc.get_function(),
                                               dbc.get_boundary()))

        for dbc in self.neumann_bcs:
            elliptic_neumann_term = vt.neumann_residual(
                dbc.get_function(), dbc.get_boundary())
            if block_form:
                elliptic_neumann_term = [elliptic_neumann_term, 0]
            residual = _add_to_residual(residual, elliptic_neumann_term)

        return residual
