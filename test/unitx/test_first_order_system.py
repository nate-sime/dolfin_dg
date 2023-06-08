import pytest

from mpi4py import MPI

import dolfinx
import ufl

import dolfin_dg.dolfinx
import dolfin_dg.primal

import convergence


class Poisson(convergence.ConvergenceTest):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_form(self, mesh, V, u, v):
        @dolfin_dg.primal.first_order_flux(lambda x: x)
        def F_2(u, flux):
            return flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
        def F_1(u, flux):
            return flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
        def F_0(u, flux):
            return -flux

        u_soln = self.u_soln(V)
        h = ufl.CellDiameter(mesh)
        p = dolfinx.fem.Constant(mesh, float(self.element.degree()))
        alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h
        f = F_0(u_soln)

        F_vec = [F_0, F_1, F_2]
        L_vec = [ufl.div, ufl.grad]

        fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
        F = - ufl.inner(f, v) * ufl.dx
        F += fos.domain()
        F += fos.interior([alpha("+") * ufl.avg(fos.G[1])])
        F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln})], u_soln)
        return F


def test_first_order_poisson():
    meshes = [
        dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8),
        dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16),
        dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    ]
    element = ufl.FiniteElement("DG", ufl.triangle, 1)
    Poisson(meshes, element).run_test()


class AdvectionDiffusion(convergence.ConvergenceTest):

    def __init__(self, meshes, element, A0, A1, b, *args, **kwargs):
        self.A0 = A0
        self.A1 = A1
        self.b = b
        super().__init__(meshes, element, *args, **kwargs)

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + 1

    def generate_form(self, mesh, V, u, v):
        F, f = 0, 0
        u_soln = self.u_soln(V)
        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)

        # -- Diffusion
        @dolfin_dg.primal.first_order_flux(lambda x: x)
        def F_2(u, flux):
            return flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.grad(F_2(x)))
        def F_1(u, flux):
            return self.A1(u, x) * flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
        def F_0(u, flux):
            return self.A0(u, x) * flux

        f += F_0(u_soln)

        F_vec = [F_0, F_1, F_2]
        L_vec = [ufl.div, ufl.grad]

        h = ufl.CellDiameter(mesh)
        p = dolfinx.fem.Constant(mesh, float(self.element.degree()))
        alpha = dolfinx.fem.Constant(mesh, 20.0) * p ** 2 / h

        fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
        F += fos.domain()
        F += fos.interior([alpha("+") * ufl.avg(fos.G[1])])
        F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln})],
                          u_soln)

        # -- Advection
        b_vector = self.b(u, x)
        @dolfin_dg.primal.first_order_flux(lambda x: x)
        def F_1(u, flux):
            return b_vector * flux

        @dolfin_dg.primal.first_order_flux(lambda x: ufl.div(F_1(x)))
        def F_0(u, flux):
            return flux

        f += F_0(u_soln)

        F_vec = [F_0, F_1]
        L_vec = [ufl.div]

        fos = dolfin_dg.primal.FirstOrderSystem(F_vec, L_vec, u, v)
        F += fos.domain()

        eigen_vals_max_p = abs(ufl.dot(ufl.diff(F_1(u), u), n)("+"))
        eigen_vals_max_m = abs(ufl.dot(ufl.diff(F_1(u), u), n)("-"))
        alpha = dolfin_dg.math.max_value(eigen_vals_max_p,
                                         eigen_vals_max_m) / 2.0
        F += fos.interior([-alpha])

        eigen_vals_max_p = abs(ufl.dot(ufl.diff(F_1(u), u), n))
        u_soln_var = ufl.variable(u_soln)
        eigen_vals_max_m = abs(
            ufl.dot(ufl.diff(F_1(u_soln_var), u_soln_var), n))
        alpha = dolfin_dg.math.max_value(eigen_vals_max_p,
                                         eigen_vals_max_m) / 2.0
        F += fos.exterior([-alpha], u_soln)

        # -- RHS
        F += - ufl.inner(f, v) * ufl.dx
        return F


def one(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), 1.0)


def neg_one(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), -1.0)


def f_x(u, x):
    return x[0]**2 + x[1]**2


def f_u(u, x):
    return u**2 + 1


def tensor_x(u, x):
    return ufl.as_tensor(((ufl.sin(x[0])**2 + 1, -ufl.sin(x[0])**2/2),
                          (-ufl.cos(x[1])**2/2, ufl.cos(x[1])**2 + 1)))


def tensor_u(u, x):
    return ufl.as_tensor(((u**2 + 1, -u**2/2),
                          (-u**3/3, u**3 + 1)))


def ID(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), ((1.0, 0.0), (0.0, 1.0)))


def IDx(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), ((1.0, 0.0), (0.0, 0.0)))


def IDy(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), ((0.0, 0.0), (0.0, 1.0)))


def tensor_zero(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), ((0.0, 0.0), (0.0, 0.0)))


def vec_one(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), (1.0, 1.0))


def vec_zero(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), (0.0, 0.0))


def vec_xhat(u, x):
    return dolfinx.fem.Constant(x.ufl_domain(), (1.0, 0.0))


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.quadrilateral])
@pytest.mark.parametrize("p", [1])
@pytest.mark.parametrize("A0", [
    neg_one,  # Conservative
])
@pytest.mark.parametrize("A1,b", [
    # (IDy, vec_xhat), # Spacetime heat
    (ID, vec_zero),  # Poisson tensor coefficient
    (one, vec_one),  # Advection-diffusion scalar coefficient
    (ID, vec_one),   # Advection-diffusion tensor coefficient
    # (IDy, vec_one),  # Spacetime heat with convection y
    # (IDx, vec_one),  # Spacetime heat with convection x
    (tensor_zero, vec_one),  # Linear advection
    (f_x, vec_zero),         # Linear isotropic Poisson
    (f_u, vec_zero),         # Nonlinear isotropic Poisson
    (tensor_u, vec_zero)     # Nonlinear anisotropic tensor Poisson
])
def test_first_order_advection_diffusion(cell_type, p, A0, A1, b):
    ns = [8, 16]
    meshes = [
        dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, cell_type=cell_type,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            diagonal=dolfinx.mesh.DiagonalType.left
        )
        for n in ns
    ]
    element = ufl.FiniteElement("DG", meshes[0].ufl_cell(), p)
    AdvectionDiffusion(meshes, element, A0, A1, b, TOL=0.1).run_test()
