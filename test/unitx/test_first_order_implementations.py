import pytest

from mpi4py import MPI

import dolfinx
import ufl

import dolfin_dg.dolfinx
import dolfin_dg.primal
import dolfin_dg.primal.simple

import convergence


class GenericProblem(convergence.ConvergenceTest):

    def generate_fos(self, u, v) -> dolfin_dg.primal.FirstOrderSystem:
        pass

    def generate_alpha(self, V):
        mesh = V.mesh
        h = ufl.CellDiameter(mesh)
        p = dolfinx.fem.Constant(mesh, float(self.element.degree()))
        alpha = dolfinx.fem.Constant(mesh, 20.0) * p**2 / h
        return alpha

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = self.generate_fos(u, v)
        alpha = self.generate_alpha(V)

        F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
        F += fos.domain()
        F += fos.interior([alpha("+") * ufl.avg(fos.G[1])])
        F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln})], u_soln)
        return F


class Advection(GenericProblem):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_fos(self, u, v):
        return dolfin_dg.primal.simple.advection(u, v, b=ufl.as_vector((1, 1)))

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = self.generate_fos(u, v)
        n = ufl.FacetNormal(mesh)

        F = fos.domain()
        eigen_vals_max_p = abs(
            ufl.dot(ufl.diff(fos.F_vec[1](u), u), n)("+"))
        eigen_vals_max_m = abs(
            ufl.dot(ufl.diff(fos.F_vec[1](u), u), n)("-"))
        alpha = dolfin_dg.math.max_value(eigen_vals_max_p,
                                         eigen_vals_max_m) / 2.0
        F += fos.interior([-alpha])

        eigen_vals_max_p = abs(ufl.dot(ufl.diff(fos.F_vec[1](u), u), n))
        u_soln_var = ufl.variable(u_soln)
        eigen_vals_max_m = abs(
            ufl.dot(ufl.diff(fos.F_vec[1](u_soln_var), u_soln_var), n))
        alpha = dolfin_dg.math.max_value(eigen_vals_max_p,
                                         eigen_vals_max_m) / 2.0
        F += fos.exterior([-alpha], u_soln)

        F -= ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
        return F


class Diffusion(GenericProblem):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_fos(self, u, v):
        return dolfin_dg.primal.simple.diffusion(u, v, A=1)


class VectorDiffusion(Diffusion):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.as_vector([
            ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])] * mesh.geometry.dim)


class Maxwell(GenericProblem):

    @property
    def k(self):
        return 1

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.as_vector([ufl.sin(self.k * x[1]), ufl.sin(self.k * x[0])])

    def generate_fos(self, u, v):
        return dolfin_dg.primal.simple.maxwell(u, v)

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = self.generate_fos(u, v)
        alpha = self.generate_alpha(V)

        F = - self.k ** 2 * ufl.inner(u, v) * ufl.dx
        F += fos.domain()
        F += fos.interior([alpha("+") * ufl.avg(fos.G[1])])
        F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln})], u_soln)
        return F


class StreamFunction(GenericProblem):

    @property
    def mu(self):
        return 1

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_fos(self, u, v):
        return dolfin_dg.primal.simple.streamfunction(u, v, self.mu)

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = self.generate_fos(u, v)

        h = ufl.CellDiameter(mesh)
        p = self.element.degree()
        alpha = dolfinx.fem.Constant(
            mesh, 10.0 * p ** (4 if p <= 2 else 6)) / h ** 3
        beta = dolfinx.fem.Constant(mesh, 10.0 * p ** 2) / h

        F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
        F += fos.domain()
        F += fos.interior([alpha("+") * ufl.avg(fos.G[1]),
                           beta("+") * ufl.avg(fos.G[2])])
        F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln}),
                           beta * ufl.replace(fos.G[2], {u: u_soln})],
                          u_soln)
        return F


class Biharmonic(StreamFunction):

    def generate_fos(self, u, v):
        return dolfin_dg.primal.simple.biharmonic(u, v)


class Triharmonic(GenericProblem):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_fos(self, u, v):
        return dolfin_dg.primal.simple.triharmonic(u, v)

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = self.generate_fos(u, v)

        h = ufl.CellDiameter(mesh)
        p = self.element.degree()
        alpha = dolfinx.fem.Constant(mesh, 10.0 * p**(8 if p <= 2 else 12)) / h**6
        beta = dolfinx.fem.Constant(mesh, 10.0 * p**(4 if p <= 2 else 6)) / h**3
        gamma = dolfinx.fem.Constant(mesh, 10.0 * p**2) / h

        F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
        F += fos.domain()
        F += fos.interior([alpha("+") * ufl.avg(fos.G[1]),
                           beta("+") * ufl.avg(fos.G[2]),
                           gamma("+") * ufl.avg(fos.G[3])])
        F += fos.exterior([alpha * ufl.replace(fos.G[1], {u: u_soln}),
                           beta * ufl.replace(fos.G[2], {u: u_soln}),
                           gamma * ufl.replace(fos.G[3], {u: u_soln})],
                          u_soln)
        return F


@pytest.mark.parametrize("problem,element,p,cell_type", [
    (Advection, ufl.FiniteElement, 1, dolfinx.mesh.CellType.triangle),
    (Diffusion, ufl.FiniteElement, 1, dolfinx.mesh.CellType.triangle),
    (VectorDiffusion, ufl.VectorElement, 1, dolfinx.mesh.CellType.triangle),
    (Maxwell, ufl.VectorElement, 1, dolfinx.mesh.CellType.triangle),
    (StreamFunction, ufl.FiniteElement, 3, dolfinx.mesh.CellType.triangle),
    (Biharmonic, ufl.FiniteElement, 3, dolfinx.mesh.CellType.quadrilateral),
    # TODO: Mark triharmonic as slow
    # (Triharmonic, ufl.FiniteElement, 4, dolfinx.mesh.CellType.quadrilateral),
])
def test_first_order_implementation(cell_type, p, problem, element):
    meshes = [
        dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, cell_type=cell_type),
        dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 14, 14, cell_type=cell_type),
        dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, cell_type=cell_type)
    ]
    problem(meshes, element("DG", meshes[0].ufl_cell(), p)).run_test()
