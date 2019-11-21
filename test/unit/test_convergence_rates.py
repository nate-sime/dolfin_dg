import pytest
from dolfin import *

from dolfin_dg.operators import *
from dolfin_dg.nitsche import NitscheBoundary
from dolfin_dg.dg_form import DGFemBO, DGFemNIPG
import numpy as np

__author__ = 'njcs4'

parameters['form_compiler']["cpp_optimize"] = True
parameters['form_compiler']["optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']["quadrature_degree"] = 8
parameters["ghost_mode"] = "shared_facet"


class ConvergenceTest:

    def __init__(self, meshes, element, norm0="l2", norm1="h1", TOL=0.5e-1):
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

            error0[run_count] = errornorm(gD, u, norm_type=self.norm0, degree_rise=3)
            error1[run_count] = errornorm(gD, u, norm_type=self.norm1, degree_rise=3)
            hsizes[run_count] = mesh.hmax()
            run_count += 1

        rate0 = np.log(error0[0:-1]/error0[1:])/np.log(hsizes[0:-1]/hsizes[1:])
        rate1 = np.log(error1[0:-1]/error1[1:])/np.log(hsizes[0:-1]/hsizes[1:])

        self.check_norm0_rates(rate0)
        self.check_norm1_rates(rate1)

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
        b = Constant(1)

        # Convective Operator
        def F_c(U):
            return b*U

        ho = HyperbolicOperator(mesh, V, DGDirichletBC(ds, gD), F_c,
                                LocalLaxFriedrichs(lambda u, n: dot(b, n)))
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
        def F_c(U):
            return b*U**2

        ho = HyperbolicOperator(mesh, V, DGDirichletBC(ds, gD), F_c,
                                LocalLaxFriedrichs(lambda u, n: 2*u*dot(b, n)))
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

        def F_c(u):
            return b*u**2

        def alpha(u, n):
            return 2*u*dot(b, n)

        def F_v(u, grad_u):
            return (u + 1)*grad_u

        ho = HyperbolicOperator(mesh, V, DGDirichletBC(ds, gD), F_c, LocalLaxFriedrichs(alpha))
        eo = EllipticOperator(mesh, V, DGDirichletBC(ds, gD), F_v)

        F = ho.generate_fem_formulation(u, v) \
          + eo.generate_fem_formulation(u, v) \
          - f*v*dx

        return F


class Burgers(ConvergenceTest):

    def gD(self, V):
        return Expression('exp(x[0] - x[1])', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        f = Expression('(exp(x[0] - x[1]) - 1)*exp(x[0] - x[1])', element=V.ufl_element())
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
        f = Expression(('(4.0L/5.0L)*cos(2*x[0] + 2*x[1])',
                        '(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)',
                        '(8.0L/125.0L)*(25*pow(sin(2*x[0] + 2*x[1]), 3) + 302*pow(sin(2*x[0] + 2*x[1]), 2) + 1216*sin(2*x[0] + 2*x[1]) + 1120)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 2)',
                        '(8.0L/625.0L)*(175*pow(sin(2*x[0] + 2*x[1]), 4) + 4199*pow(sin(2*x[0] + 2*x[1]), 3) + 33588*pow(sin(2*x[0] + 2*x[1]), 2) + 112720*sin(2*x[0] + 2*x[1]) + 145600)*cos(2*x[0] + 2*x[1])/pow(sin(2*x[0] + 2*x[1]) + 4, 3)'),
                       element=V.ufl_element())


        bo = CompressibleEulerOperator(mesh, V, DGDirichletBC(ds, gD))
        F = bo.generate_fem_formulation(u, v) - inner(f, v)*dx

        return F


class Maxwell(ConvergenceTest):
    def gD(self, V):
        return Expression(("sin(k*x[1])", "sin(k*x[0])"), k=2, element=V.ufl_element())

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
        f = Expression(('0.8*cos(2.0*x[0] + 2.0*x[1])',
                        '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                        '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                        '(2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1]) + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5) + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4) + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3) + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2) + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1]) + 3817.472*sin(4.0*x[0] + 4.0*x[1]) + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 4)'),
                       element=V.ufl_element())

        bo = CompressibleNavierStokesOperator(mesh, V, DGDirichletBC(ds, gD))
        F = bo.generate_fem_formulation(u, v) - inner(f, v)*dx
        return F


class NavierStokesEntropy(ConvergenceTest):
    def gD(self, V):
        return Expression(('((-std::log((0.4*sin(2*x[0] + 2*x[1]) + 1.6)*pow(sin(2*x[0] + 2*x[1]) + 4, -1.4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4)) + 2.4)*(sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4) - pow(sin(2*x[0] + 2*x[1]) + 4, 2))/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))',
                    '(-sin(2*x[0] + 2*x[1]) - 4)/((sin(2*x[0] + 2*x[1]) + 4)*(-pow((1.0L/5.0L)*sin(2*x[0] + 2*x[1]) + 4, 2)/pow(sin(2*x[0] + 2*x[1]) + 4, 2) + sin(2*x[0] + 2*x[1]) + 4))'),
                    element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        f = Expression(('0.8*cos(2.0*x[0] + 2.0*x[1])',
                        '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                        '(1.6*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 25.728*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 155.136*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) - 34.1333333333333*pow(sin(2.0*x[0] + 2.0*x[1]), 2) - 136.533333333333*sin(2.0*x[0] + 2.0*x[1]) + 191.488*sin(4.0*x[0] + 4.0*x[1]) - 68.2666666666667*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 286.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 3)',
                        '(2.24*pow(sin(2.0*x[0] + 2.0*x[1]), 5)*cos(2.0*x[0] + 2.0*x[1]) + 15.5555555555556*pow(sin(2.0*x[0] + 2.0*x[1]), 5) + 62.7072*pow(sin(2.0*x[0] + 2.0*x[1]), 4)*cos(2.0*x[0] + 2.0*x[1]) + 248.888888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 4) + 644.9152*pow(sin(2.0*x[0] + 2.0*x[1]), 3)*cos(2.0*x[0] + 2.0*x[1]) + 1499.59111111111*pow(sin(2.0*x[0] + 2.0*x[1]), 3) + 3162.5216*pow(sin(2.0*x[0] + 2.0*x[1]), 2)*cos(2.0*x[0] + 2.0*x[1]) + 4132.40888888889*pow(sin(2.0*x[0] + 2.0*x[1]), 2) + 12.5155555555556*sin(2.0*x[0] + 2.0*x[1])*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 4482.84444444444*sin(2.0*x[0] + 2.0*x[1]) + 3817.472*sin(4.0*x[0] + 4.0*x[1]) + 350.435555555555*pow(cos(2.0*x[0] + 2.0*x[1]), 2) + 7454.72*cos(2.0*x[0] + 2.0*x[1]))/pow(1.0*sin(2.0*x[0] + 2.0*x[1]) + 4.0, 4)'),
                       element=V.ufl_element())

        bo = CompressibleNavierStokesOperatorEntropyFormulation(mesh, V, DGDirichletBC(ds, gD))
        F = bo.generate_fem_formulation(u, v) - inner(f, v)*dx
        return F


class Poisson(ConvergenceTest):
    def gD(self, V):
        return Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v, vt=None):
        if vt is None:
            raise RuntimeError("No viscous term provided")
        gD = self.gD(V)
        u.interpolate(gD)
        f = Expression('2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)*sin(pi*x[0])*sin(pi*x[1]) - pow(pi, 2)*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2) - pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)',
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
        return Expression('sin(pi*x[0])*sin(pi*x[1]) + 1.0', element=V.ufl_element())

    def generate_form(self, mesh, V, u, v):
        gD = self.gD(V)
        u.interpolate(gD)
        def F_v(u, grad_u):
            return (u+1)*grad_u
        f = Expression('2*pow(pi, 2)*(sin(pi*x[0])*sin(pi*x[1]) + 2.0)*sin(pi*x[0])*sin(pi*x[1]) - pow(pi, 2)*pow(sin(pi*x[0]), 2)*pow(cos(pi*x[1]), 2) - pow(pi, 2)*pow(sin(pi*x[1]), 2)*pow(cos(pi*x[0]), 2)',
                       element=V.ufl_element())
        nbc = NitscheBoundary(F_v, u, v)
        F = dot(F_v(u, grad(u)), grad(v))*dx - f*v*dx
        F += nbc.nistche_bc_residual(gD, ds)

        ff = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        asd = AutoSubDomain(lambda x, on: abs(x[0] - 0.25) < 1e-10)
        asd.mark(ff, 1)
        dS = Measure("dS", subdomain_data=ff)

        F += nbc.nistche_interior_bc_residual(gD, dS(1))

        return F


# -- Mesh fixtures
@pytest.fixture
def IntervalMeshes():
    return [UnitIntervalMesh(16),
            UnitIntervalMesh(32)]

@pytest.fixture
def SquareMeshes():
    return [UnitSquareMesh(16, 16, 'left/right'),
            UnitSquareMesh(32, 32, 'left/right')]

@pytest.fixture
def SquareMeshesPi():
    return [RectangleMesh(Point(0, 0), Point(pi, pi), 32, 32, 'right'),
            RectangleMesh(Point(0, 0), Point(pi, pi), 64, 64, 'right')]


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


@pytest.mark.parametrize("conv_test", [NavierStokes,
                                       NavierStokesEntropy])
def test_square_navier_stokes_problems(conv_test, SquareMeshesPi):
    element = VectorElement("DG", SquareMeshesPi[0].ufl_cell(), 1, dim=4)
    conv_test(SquareMeshesPi, element, TOL=0.06).run_test()


# -- 2D non-conventional DG tests
@pytest.mark.parametrize("conv_test", [PoissonBO])
def test_square_baumann_oden_problems(conv_test, SquareMeshes):
    element = FiniteElement("DG", SquareMeshes[0].ufl_cell(), 2)
    conv_test(SquareMeshes, element).run_test()


# -- Nitsche CG tests
@pytest.mark.parametrize("conv_test", [PoissonNistcheBC])
def test_square_nitsche_cg_problems(conv_test, SquareMeshes):
    element = FiniteElement("CG", SquareMeshes[0].ufl_cell(), 1)
    conv_test(SquareMeshes, element).run_test()