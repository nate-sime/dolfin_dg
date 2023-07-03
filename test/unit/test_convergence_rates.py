import numpy as np
import pytest
from dolfin import (
    parameters, FunctionSpace, UnitIntervalMesh, UnitSquareMesh, RectangleMesh,
    Function, solve, TestFunction, Constant, errornorm, ds, dx, Expression,
    Point, pi, MeshFunction, AutoSubDomain, near)
from ufl import (
    FiniteElement, VectorElement, MixedElement, dot, triangle, as_vector, inner,
    CellVolume, FacetArea, Coefficient, grad, div, split, Measure, Identity,
    FacetNormal, sym)

import dolfin_dg
import dolfin_dg.math
from dolfin_dg import DGDirichletBC, DGNeumannBC
from dolfin_dg.dg_form import DGFemBO, DGFemNIPG
from dolfin_dg.nitsche import NitscheBoundary, StokesNitscheBoundary
from dolfin_dg.operators import (
    PoissonOperator, EllipticOperator, MaxwellOperator,
    CompressibleEulerOperator, CompressibleNavierStokesOperator,
    HyperbolicOperator, SpacetimeBurgersOperator,
    DGFemSIPG, StokesOperator)

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']["quadrature_degree"] = 4
parameters["ghost_mode"] = "shared_facet"

global_degree_rise = 1


class ConvergenceTest:

    def __init__(self, meshes, element, norm0="l2", norm1="h1", TOL=0.1):
        self.meshes = meshes
        self.norm0 = norm0
        self.norm1 = norm1
        self.TOL = TOL
        self.element = element

    def gD(self, V):
        pass

    def generate_form(self, mesh, V, u, v):
        pass

    def run_test(self):
        n_runs = len(self.meshes)
        error0 = np.zeros(n_runs)
        error1 = np.zeros(n_runs)
        hsizes = np.zeros(n_runs)

        run_count = 0
        for mesh in self.meshes:
            V = FunctionSpace(mesh, self.element)
            u, v = Function(V), TestFunction(V)
            gD = self.gD(V)
            residual = self.generate_form(mesh, V, u, v)
            solve(residual == 0, u)

            error0[run_count] = self.compute_error_norm0(gD, u)
            error1[run_count] = self.compute_error_norm1(gD, u)
            hsizes[run_count] = mesh.hmax()
            run_count += 1

        rate0 = np.log(error0[0:-1]/error0[1:])/np.log(hsizes[0:-1]/hsizes[1:])
        rate1 = np.log(error1[0:-1]/error1[1:])/np.log(hsizes[0:-1]/hsizes[1:])

        self.check_norm0_rates(rate0)
        self.check_norm1_rates(rate1)

    def compute_error_norm0(self, gD, u):
        return errornorm(gD, u, norm_type=self.norm0,
                         degree_rise=global_degree_rise)

    def compute_error_norm1(self, gD, u):
        return errornorm(gD, u, norm_type=self.norm1,
                         degree_rise=global_degree_rise)

    def check_norm0_rates(self, rate0):
        expected_rate = float(self.element.degree() + 1)
        assert abs(rate0[0] - expected_rate) < self.TOL

    def check_norm1_rates(self, rate1):
        expected_rate = float(self.element.degree())
        assert abs(rate1[0] - expected_rate) < self.TOL


class Advection1D(ConvergenceTest):

    def gD(self, V):
        return Expression('exp(x[0])', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        b = Constant((1,))

        # Convective Operator
        def F_c(_, flux):
            return b*flux

        ho = HyperbolicOperator(mesh, V, DGDirichletBC(ds, gD), F_c)
        F = ho.generate_fem_formulation(u, v) - gD*v*dx

        return F


class Advection(ConvergenceTest):

    def gD(self, V):
        return Expression('exp(x[0] - x[1])', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        b = Constant((1, 1))

        # Convective Operator
        def F_c(u, flux):
            return b*flux**2

        ho = HyperbolicOperator(mesh, V, DGDirichletBC(ds, gD), F_c)
        F = ho.generate_fem_formulation(u, v)

        return F


class AdvectionDiffusion(ConvergenceTest):

    def gD(self, V):
        return Expression('exp(x[0] - x[1])', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        f = Expression('-4*exp(2*(x[0] - x[1])) - 2*exp(x[0] - x[1])',
                       element=V.ufl_element())
        b = Constant((1, 1))

        def F_c(u, flux):
            return b*flux**2

        def F_v(u, grad_u):
            return (u + 1)*grad_u

        ho = HyperbolicOperator(mesh, V, [DGDirichletBC(ds, gD)], F_c)
        eo = EllipticOperator(mesh, V, [DGDirichletBC(ds, gD)], F_v)

        F = ho.generate_fem_formulation(u, v) \
            + eo.generate_fem_formulation(u, v) \
            - f*v*dx

        return F


class Burgers(ConvergenceTest):

    def gD(self, V):
        return Expression('exp(x[0] - x[1])', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        f = Expression('(exp(x[0] - x[1]) - 1)*exp(x[0] - x[1])',
                       element=V.ufl_element())
        u.interpolate(gD)

        bo = SpacetimeBurgersOperator(mesh, V, DGDirichletBC(ds, gD))
        residual = bo.generate_fem_formulation(u, v) - f*v*dx
        return residual


class Euler(ConvergenceTest):

    def gD(self, V):
        return Expression(('sin(2*(x[0]+x[1])) + 4',
                           '0.2*sin(2*(x[0]+x[1])) + 4',
                           '0.2*sin(2*(x[0]+x[1])) + 4',
                           'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                          element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        f0 = '(4.0L/5.0L)*cos(2*x[0] + 2*x[1])'
        f1 = '''(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3)
            + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1])
            + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)'''
        f2 = '''(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3)
            + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1])
            + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)'''
        f3 = '''(8.0L/625.0L)*(175*pow(sin(2*x[0] + 2*x[1]), 4)
            + 4199*pow(sin(2*x[0] + 2*x[1]), 3) + 33588*pow(sin(2*x[0]
            + 2*x[1]), 2) + 112720*sin(2*x[0] + 2*x[1]) + 145600)*cos(2*x[0]
            + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 3)'''
        f = Expression((f0, f1, f2, f3), element=V.ufl_element())

        bo = CompressibleEulerOperator(mesh, V, DGDirichletBC(ds, gD))
        F = bo.generate_fem_formulation(u, v) - inner(f, v)*dx

        return F


class Maxwell(ConvergenceTest):
    def gD(self, V):
        return Expression(("sin(k*x[1])", "sin(k*x[0])"), k=2,
                          element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        k = Constant(2.0)
        gD = self.gD(V)
        u.interpolate(gD)

        def F_m(u, curl_u):
            return curl_u

        mo = MaxwellOperator(mesh, V, [DGDirichletBC(ds, gD)], F_m)
        F = mo.generate_fem_formulation(u, v) - k**2*dot(u, v)*dx
        return F


class NavierStokes(ConvergenceTest):
    def gD(self, V):
        return Expression(('sin(2*(x[0]+x[1])) + 4',
                           '0.2*sin(2*(x[0]+x[1])) + 4',
                           '0.2*sin(2*(x[0]+x[1])) + 4',
                           'pow((sin(2*(x[0]+x[1])) + 4), 2)'),
                          element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        f0 = '0.8*cos(2.0*x[0] + 2.0*x[1])'
        f1 = '''(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1])
            + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1])
            + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1])
            - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2)
            - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0]
            + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2)
            + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1])
            + 4.0, 3)'''
        f2 = '''(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1])
            + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1])
            + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1])
            - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2)
            - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0]
            + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2)
            + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1])
            + 4.0, 3)'''
        f3 = '''(2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1])
            + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5)
            + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1])
            + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4)
            + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1])
            + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3)
            + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0]
            + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2)
            + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0]
            + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1])
            + 3817.472*sin(4.0*x[0] + 4.0*x[1])
            + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2)
            + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0]
            + 2.0*x[1]) + 4.0, 4)'''
        f = Expression((f0, f1, f2, f3), element=V.ufl_element())

        bo = CompressibleNavierStokesOperator(mesh, V, DGDirichletBC(ds, gD))
        F = bo.generate_fem_formulation(u, v) - inner(f, v)*dx
        return F


class Poisson(ConvergenceTest):
    def gD(self, V):
        return Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0',
                          element=V.ufl_element())

    def generate_form(self, mesh, V, u, v, vt=None):
        if vt is None:
            raise RuntimeError("No viscous term provided")
        gD = self.gD(V)
        u.interpolate(gD)
        f = Expression(
            '2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)'
            '*sin(pi*x[0])*sin(pi*x[1]) - pow(pi, 2)'
            '*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2)'
            '-pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)',
            element=V.ufl_element())
        pe = PoissonOperator(mesh, V, DGDirichletBC(ds, gD), kappa=u + 1)
        F = pe.generate_fem_formulation(u, v, vt=vt) - f*v*dx

        return F


class PoissonSIPG(Poisson):

    def generate_form(self, mesh, V, u, v):
        return super().generate_form(mesh, V, u, v, vt=DGFemSIPG)


class PoissonBO(Poisson):

    def generate_form(self, mesh, V, u, v):
        return super().generate_form(mesh, V, u, v, vt=DGFemBO)

    def check_norm0_rates(self, rate0):
        degree = self.element.degree()
        even_degree = degree % 2 == 0
        expected_rate = degree if even_degree else degree + 1
        assert abs(rate0[0] - expected_rate) < self.TOL


class PoissonNIPG(Poisson):

    def generate_form(self, mesh, V, u, v):
        return super().generate_form(mesh, V, u, v, vt=DGFemNIPG)


class PoissonNistcheBC(ConvergenceTest):
    def gD(self, V):
        return Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0',
                          element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)

        def F_v(u, grad_u):
            return (u+1)*grad_u

        f = Expression(
            '2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)'
            '*sin(pi*x[0])*sin(pi*x[1])'
            '-pow(pi, 2)*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2)'
            '-pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)',
            element=V.ufl_element())
        nbc = NitscheBoundary(F_v, u, v)
        F = dot(F_v(u, grad(u)), grad(v))*dx - f*v*dx
        F += nbc.nitsche_bc_residual(gD, ds)

        ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        asd = AutoSubDomain(lambda x, on: abs(x[0] - 0.25) < 1e-10)
        asd.mark(ff, 1)
        dS = Measure("dS", subdomain_data=ff)

        F += nbc.nitsche_bc_residual_on_interior(gD, dS(1))

        return F


class StokesTest(ConvergenceTest):

    def gD(self, W):
        u_soln = Expression(("-(x[1]*cos(x[1]) + sin(x[1]))*exp(x[0])",
                             "x[1] * sin(x[1]) * exp(x[0])"),
                            degree=W.sub(0).ufl_element().degree() + 1,
                            domain=W.mesh())
        p_soln = Expression(
            "2.0 * exp(x[0]) * sin(x[1]) + 1.5797803888225995912 / 3.0",
            degree=W.sub(1).ufl_element().degree() + 1, domain=W.mesh())
        return u_soln, p_soln

    def generate_form(self, mesh, W, U, V):
        u = as_vector((U[0], U[1]))
        p = U[2]

        v = as_vector((V[0], V[1]))
        q = V[2]

        def stress(u, p):
            return 2 * sym(grad(u)) - p * Identity(2)

        u_soln, p_soln = self.gD(W)

        ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        AutoSubDomain(
            lambda x, on: near(x[0], 0.0) or near(x[1], 0.0)).mark(ff, 1)
        AutoSubDomain(
            lambda x, on: near(x[0], 1.0) or near(x[1], 1.0)).mark(ff, 2)
        ds = Measure("ds", subdomain_data=ff)
        dsD, dsN = ds(1), ds(2)

        facet_n = FacetNormal(mesh)
        gN = stress(u_soln, p_soln) * facet_n
        bcs = [DGDirichletBC(dsD, u_soln), DGNeumannBC(dsN, gN)]

        pe = StokesOperator(None, None, bcs, None)

        h = CellVolume(u.ufl_domain())/FacetArea(u.ufl_domain())
        ufl_degree = W.ufl_element().degree()
        C_IP = 20.0
        penalty = Constant(C_IP * max(ufl_degree ** 2, 1)) / h
        F = pe.generate_fem_formulation(
            u, v, p, q, lambda x: 1, penalty=penalty)

        return F

    def compute_error_norm0(self, gD, u):
        return errornorm(gD[0], u.sub(0), norm_type=self.norm0,
                         degree_rise=global_degree_rise)

    def compute_error_norm1(self, gD, u):
        return errornorm(gD[0], u.sub(0), norm_type=self.norm1,
                         degree_rise=global_degree_rise)


class StokesNitscheTest(ConvergenceTest):

    def gD(self, W):
        u_soln = Expression(("-(x[1]*cos(x[1]) + sin(x[1]))*exp(x[0])",
                             "x[1] * sin(x[1]) * exp(x[0])"),
                            degree=W.sub(0).ufl_element().degree() + 2,
                            domain=W.mesh())
        p_soln = Expression(
            "2.0 * exp(x[0]) * sin(x[1]) + 1.5797803888225995912 / 3.0",
            degree=W.sub(1).ufl_element().degree() + 1, domain=W.mesh())
        return u_soln, p_soln

    def _formulate_nitsche_boundary(self, F_v, u, p, v, q,
                                    u_soln, p_soln,
                                    gN, dsD, dSint):
        raise NotImplementedError

    def generate_form(self, mesh, W, U, V):
        u = as_vector((U[0], U[1]))
        p = U[2]

        v = as_vector((V[0], V[1]))
        q = V[2]

        def F_v(u, grad_u, p_local=None):
            if p_local is None:
                p_local = p
            return (Constant(10.0) + Constant(1.0) * dot(u, u)) * grad_u \
                - p_local * Identity(2)

        u_soln, p_soln = self.gD(W)

        ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        AutoSubDomain(
            lambda x, on: near(x[0], 0.0) or near(x[1], 0.0)).mark(ff, 1)
        AutoSubDomain(
            lambda x, on: near(x[0], 1.0) or near(x[1], 1.0)).mark(ff, 2)
        AutoSubDomain(lambda x, on: near(x[0], 0.25)).mark(ff, 3)

        ds = Measure("ds", subdomain_data=ff)
        dsD, dsN = ds(1), ds(2)

        dS = Measure("dS", subdomain_data=ff)
        dSint = dS(3)

        # Domain
        f = -div(F_v(u_soln, grad(u_soln), p_soln))
        F = inner(F_v(u, grad(u)), grad(v)) * dx \
            + q * div(u) * dx - dot(f, v)*dx

        # Neumann
        facet_n = FacetNormal(mesh)
        gN = F_v(u_soln, grad(u_soln), p_soln) * facet_n
        F -= dot(gN, v) * dsN

        F += self._formulate_nitsche_boundary(F_v, u, p, v, q,
                                              u_soln, p_soln,
                                              gN, dsD, dSint)
        return F

    def compute_error_norm0(self, gD, u):
        return errornorm(gD[0], u.sub(0), norm_type=self.norm0,
                         degree_rise=global_degree_rise)

    def compute_error_norm1(self, gD, u):
        return errornorm(gD[0], u.sub(0), norm_type=self.norm1,
                         degree_rise=global_degree_rise)


class StokesNitscheDirichletBCTest(StokesNitscheTest):

    def _formulate_nitsche_boundary(self, F_v, u, p, v, q,
                                    u_soln, p_soln,
                                    gN, dsD, dSint):
        stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
        F = stokes_nitsche.nitsche_bc_residual(u_soln, dsD)
        F += stokes_nitsche.nitsche_bc_residual_on_interior(u_soln, dSint)
        return F


class StokesNitscheSlipBCTest(StokesNitscheTest):

    def _formulate_nitsche_boundary(self, F_v, u, p, v, q,
                                    u_soln, p_soln,
                                    gN, dsD, dSint):
        stokes_nitsche = StokesNitscheBoundary(F_v, u, p, v, q)
        F = stokes_nitsche.slip_nitsche_bc_residual(u_soln, gN, dsD)
        F += stokes_nitsche.slip_nitsche_bc_residual_on_interior(
            u_soln, gN, dSint)
        return F


class StokesNitscheSlipManualTest(StokesNitscheTest):

    def _formulate_nitsche_boundary(self, F_v, u, p, v, q,
                                    u_soln, p_soln,
                                    gN, dsD, dSint):
        n = FacetNormal(
            u.ufl_operands[0].ufl_operands[0].function_space().mesh())

        def F_v_normal(u, grad_u):
            return dolfin_dg.math.normal_proj(F_v(u, grad_u), n)

        stokes_nitsche = StokesNitscheBoundary(F_v_normal, u, p, v, q)
        F = stokes_nitsche.nitsche_bc_residual(u_soln, dsD)
        F -= dot(dolfin_dg.math.tangential_proj(gN, n), v) * dsD
        F += stokes_nitsche.nitsche_bc_residual_on_interior(u_soln, dSint)
        F -= sum(dot(dolfin_dg.math.tangential_proj(gN, n), v)(side)
                 for side in ("+", "-")) * dSint
        return F


class NitscheMixedElement(ConvergenceTest):

    def get_num_sub_spaces(self, V):
        return Coefficient(V).ufl_shape[0]

    def gD(self, V):
        num_sub_spaces = self.get_num_sub_spaces(V)
        return Expression(['sin(pi*x[0])*sin(pi*x[1])'] * num_sub_spaces,
                          element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        num_sub_spaces = self.get_num_sub_spaces(V)
        f = Expression(['2.0*pi*pi*sin(pi*x[0])*sin(pi*x[1])'] * num_sub_spaces,
                       element=V.ufl_element())
        F = inner(grad(u), grad(v)) * dx - inner(f, v) * dx

        def F_v(u, grad_u):
            return grad_u

        for u_sub, v_sub in zip(split(u), split(v)):
            nbc = NitscheBoundary(F_v, u_sub, v_sub)
            if u_sub.ufl_shape:
                u_gamma = Constant([0.0] * u_sub.ufl_shape[0])
            else:
                u_gamma = Constant(0.0)
            F += nbc.nitsche_bc_residual(u_gamma, ds)

        return F


# -- Mesh fixtures
@pytest.fixture
def IntervalMeshes():
    return [UnitIntervalMesh(16),
            UnitIntervalMesh(32)]


@pytest.fixture
def SquareMeshes():
    return [UnitSquareMesh(8, 8, 'left/right'),
            UnitSquareMesh(16, 16, 'left/right')]


@pytest.fixture
def SquareMeshesPi():
    return [RectangleMesh(Point(0, 0), Point(pi, pi), 16, 16, 'right'),
            RectangleMesh(Point(0, 0), Point(pi, pi), 32, 32, 'right')]


# -- 1D conventional DG tests
@pytest.mark.parametrize("conv_test", [Advection1D])
def test_interval_problems(conv_test, IntervalMeshes):
    element = FiniteElement("DG", IntervalMeshes[0].ufl_cell(), 1)
    conv_test(IntervalMeshes, element).run_test()


# -- 2D conventional DG tests
@pytest.mark.parametrize("conv_test", [Advection,
                                       AdvectionDiffusion,
                                       Burgers,
                                       PoissonNIPG,
                                       PoissonSIPG])
def test_square_problems(conv_test, SquareMeshes):
    element = FiniteElement("DG", SquareMeshes[0].ufl_cell(), 1)
    conv_test(SquareMeshes, element).run_test()


@pytest.mark.parametrize("conv_test", [Maxwell])
def test_square_maxwell_problem(conv_test, SquareMeshes):
    element = VectorElement("DG", SquareMeshes[0].ufl_cell(), 1, dim=2)
    conv_test(SquareMeshes, element, norm1="hcurl").run_test()


@pytest.mark.parametrize("conv_test", [Euler])
def test_square_euler_problems(conv_test):
    meshes = [RectangleMesh(Point(0, 0), Point(.5*pi, .5*pi), 8, 8, 'right'),
              RectangleMesh(Point(0, 0), Point(.5*pi, .5*pi), 16, 16, 'right')]
    element = VectorElement("DG", meshes[0].ufl_cell(), 1, dim=4)
    conv_test(meshes, element).run_test()


# @pytest.mark.parametrize("conv_test", [NavierStokes,
#                                        NavierStokesEntropy])
# def test_square_navier_stokes_problems(conv_test, SquareMeshesPi):
#     element = VectorElement("DG", SquareMeshesPi[0].ufl_cell(), 1, dim=4)
#     conv_test(SquareMeshesPi, element, TOL=0.25).run_test()


@pytest.mark.parametrize("conv_test", [StokesTest])
def test_square_stokes_problems(conv_test, SquareMeshes):
    element = MixedElement([VectorElement("DG", SquareMeshes[0].ufl_cell(), 2),
                            FiniteElement("DG", SquareMeshes[0].ufl_cell(), 1)])
    conv_test(SquareMeshes, element).run_test()


# -- 2D non-conventional DG tests
# @pytest.mark.parametrize("conv_test", [PoissonBO])
# def test_square_baumann_oden_problems(conv_test, SquareMeshes):
#     element = FiniteElement("DG", SquareMeshes[0].ufl_cell(), 2)
#     conv_test(SquareMeshes, element).run_test()


# -- Nitsche CG tests
@pytest.mark.parametrize("conv_test", [PoissonNistcheBC])
def test_square_nitsche_cg_problems(conv_test, SquareMeshes):
    element = FiniteElement("CG", SquareMeshes[0].ufl_cell(), 1)
    conv_test(SquareMeshes, element).run_test()


mixed_elements_for_testing = [
    MixedElement([VectorElement("CG", triangle, 1),
                  FiniteElement("CG", triangle, 1)]),
    MixedElement([FiniteElement("CG", triangle, 1),
                  VectorElement("CG", triangle, 1)]),
    MixedElement([FiniteElement("CG", triangle, 1),
                  VectorElement("CG", triangle, 1),
                  FiniteElement("CG", triangle, 1), ])
]


@pytest.mark.parametrize("element", mixed_elements_for_testing)
@pytest.mark.parametrize("conv_test", [NitscheMixedElement])
def test_square_nitsche_cg_mixed_problems(conv_test, SquareMeshes, element):
    conv_test(SquareMeshes, element).run_test()


@pytest.mark.parametrize("conv_test", [StokesNitscheDirichletBCTest,
                                       StokesNitscheSlipBCTest,
                                       StokesNitscheSlipManualTest])
def test_square_stokes_nitsche_problems(conv_test, SquareMeshes):
    element = MixedElement([VectorElement("CG", SquareMeshes[0].ufl_cell(), 2),
                            FiniteElement("CG", SquareMeshes[0].ufl_cell(), 1)])
    conv_test(SquareMeshes, element).run_test()
