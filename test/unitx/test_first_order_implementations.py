import pytest

import functools
import numpy as np
from mpi4py import MPI

import dolfinx
import ufl

import dolfin_dg
import dolfin_dg.dolfinx
import dolfin_dg.penalty
import dolfin_dg.primal
import dolfin_dg.primal.simple
import dolfin_dg.primal.aero
import dolfin_dg.primal.flux_sipg_ext_avg

import convergence


class Advection(convergence.ConvergenceTest):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = dolfin_dg.primal.simple.advection(u, v, b=ufl.as_vector((1, 1)))
        n = ufl.FacetNormal(mesh)

        F = fos.domain()
        alpha, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
            ufl.dot(ufl.diff(fos.F_vec[1](u), u), n), u, u_soln)
        F += fos.interior([-alpha])
        F += fos.exterior([-alpha_ext], u_soln)

        F -= ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
        return F


class Diffusion(convergence.ConvergenceTest):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = dolfin_dg.primal.simple.diffusion(u, v, A=1)

        F = - ufl.inner(fos.F_vec[0](u_soln), v) * ufl.dx
        F += fos.domain()
        alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(fos, u, u_soln)
        F += fos.interior([alpha])
        F += fos.exterior([alpha_ext], u_soln)
        return F


class VectorDiffusion(Diffusion):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.as_vector([
            ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])] * mesh.geometry.dim)


class Maxwell(convergence.ConvergenceTest):

    @property
    def k(self):
        return 1

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        return ufl.as_vector([ufl.sin(self.k * x[1]), ufl.sin(self.k * x[0])])

    def generate_form(self, mesh, V, u, v):
        u_soln = self.u_soln(V)
        fos = dolfin_dg.primal.simple.maxwell(u, v)
        alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(fos, u, u_soln)

        F = - self.k ** 2 * ufl.inner(u, v) * ufl.dx
        F += fos.domain()
        F += fos.interior([alpha])
        F += fos.exterior([alpha_ext], u_soln)
        return F


class StreamFunction(convergence.ConvergenceTest):

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


class Triharmonic(convergence.ConvergenceTest):

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
def test_first_order_simple(cell_type, p, problem, element):
    meshes = [
        dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 8, 8, cell_type=cell_type),
        dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 14, 14, cell_type=cell_type),
        dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 20, 20, cell_type=cell_type)
    ]
    problem(meshes, element("DG", meshes[0].ufl_cell(), p)).run_test()


class CompressibleEuler(convergence.ConvergenceTest):

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        U_soln = ufl.as_vector((ufl.sin(2 * (x[0] + x[1])) + 4,
                                0.2 * ufl.sin(2 * (x[0] + x[1])) + 4,
                                0.2 * ufl.sin(2 * (x[0] + x[1])) + 4,
                                (ufl.sin(2 * (x[0] + x[1])) + 4) ** 2))
        return U_soln

    def generate_form(self, mesh, V, U, v):
        U_soln = self.u_soln(V)
        U.interpolate(
            dolfinx.fem.Expression(U_soln, V.element.interpolation_points()))
        fos = dolfin_dg.primal.aero.compressible_euler(U, v)

        gamma = 1.4

        F = fos.domain() - ufl.inner(fos.F_vec[0](U_soln), v) * ufl.dx

        rho, u, E = dolfin_dg.aero.flow_variables(U)
        pressure = dolfin_dg.aero.pressure(U, gamma=gamma)
        c = dolfin_dg.aero.speed_of_sound(pressure, rho, gamma=gamma)
        n = ufl.FacetNormal(mesh)
        lambdas = [ufl.dot(u, n) - c, ufl.dot(u, n), ufl.dot(u, n) + c]

        alpha, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
            lambdas, U, U_soln)
        F += fos.interior([-alpha])
        F += fos.exterior([-alpha_ext], U_soln)
        return F


class CompressibleNavierStokes(CompressibleEuler):

    def generate_form(self, mesh, V, U, v):
        U_soln = self.u_soln(V)
        fos = dolfin_dg.primal.aero.compressible_navier_stokes(U, v)
        F = fos.domain() - ufl.inner(fos.F_vec[0](U_soln), v) * ufl.dx

        alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(fos, U, U_soln)
        F += fos.interior([alpha])
        F += fos.exterior([alpha_ext], U_soln)
        F += super().generate_form(mesh, V, U, v)
        return F


class CompressibleEulerEntropy(convergence.ConvergenceTest):
    gamma = 1.4

    def u_soln(self, V):
        mesh = V.mesh
        x = ufl.SpatialCoordinate(mesh)
        U_soln = ufl.as_vector((ufl.sin(2 * (x[0] + x[1])) + 4,
                                0.2 * ufl.sin(2 * (x[0] + x[1])) + 4,
                                0.2 * ufl.sin(2 * (x[0] + x[1])) + 4,
                                (ufl.sin(2 * (x[0] + x[1])) + 4) ** 2))
        U_soln = dolfin_dg.primal.aero.U_to_V(U_soln, self.gamma)
        return U_soln

    def generate_form(self, mesh, fspace, soln_vec, v):
        metadata = {"quadrature_degree": 2 * fspace.ufl_element().degree() + 1}
        dx = ufl.Measure("dx", metadata=metadata)
        ds = ufl.Measure("ds", metadata=metadata)
        dS = ufl.Measure("dS", metadata=metadata)
        gD = self.u_soln(fspace)

        soln_vec.interpolate(
            dolfinx.fem.Expression(gD, fspace.element.interpolation_points()))
        fos = dolfin_dg.primal.aero.compressible_euler_entropy(soln_vec, v)

        F = fos.domain(dx=dx) - ufl.inner(fos.F_vec[0](gD), v) * dx

        U = dolfin_dg.primal.aero.V_to_U(soln_vec, self.gamma)
        rho, u, E = dolfin_dg.aero.flow_variables(U)
        pressure = dolfin_dg.aero.pressure(U, gamma=self.gamma)
        c = dolfin_dg.aero.speed_of_sound(pressure, rho, gamma=self.gamma)
        n = ufl.FacetNormal(mesh)
        lambdas = [ufl.dot(u, n) - c, ufl.dot(u, n), ufl.dot(u, n) + c]

        alpha, alpha_ext = dolfin_dg.penalty.local_lax_friedrichs_penalty(
            lambdas, soln_vec, gD)
        F += fos.interior([-alpha],
                          flux_type=dolfin_dg.primal.flux_sipg_ext_avg, dS=dS)
        F += fos.exterior([-alpha_ext], gD,
                          flux_type=dolfin_dg.primal.flux_sipg_ext_avg, ds=ds)
        return F


class CompressibleNavierStokesEntropy(CompressibleEulerEntropy):

    def generate_form(self, mesh, fspace, soln_vec, v):
        metadata = {"quadrature_degree": 2 * fspace.ufl_element().degree() + 1}
        dx = ufl.Measure("dx", metadata=metadata)
        ds = ufl.Measure("ds", metadata=metadata)
        dS = ufl.Measure("dS", metadata=metadata)

        U_soln = self.u_soln(fspace)
        fos = dolfin_dg.primal.aero.compressible_navier_stokes_entropy(
            soln_vec, v)
        F = fos.domain() - ufl.inner(fos.F_vec[0](U_soln), v) * dx

        alpha, alpha_ext = dolfin_dg.penalty.interior_penalty(
            fos, soln_vec, U_soln)
        F += fos.interior([alpha], dS=dS)
        F += fos.exterior([alpha_ext], U_soln, ds=ds)
        F += super().generate_form(mesh, fspace, soln_vec, v)
        return F


@pytest.mark.parametrize("problem,p,cell_type", [
    (CompressibleEuler, 1, dolfinx.mesh.CellType.triangle),
    (CompressibleNavierStokes, 1, dolfinx.mesh.CellType.triangle),
    (CompressibleEulerEntropy, 1, dolfinx.mesh.CellType.triangle),
    # TODO: Mark CompressibleNavierStokesEntropy as slow
    # (CompressibleNavierStokesEntropy, 1, dolfinx.mesh.CellType.triangle)
])
def test_first_order_aero(cell_type, p, problem):
    def generate_mesh(N, l):
        return dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, [[0.0, 0.0], [l, l]], [N, N],
            cell_type=cell_type, ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            diagonal=dolfinx.mesh.DiagonalType.left)

    meshes = [generate_mesh(N, 0.5 * np.pi) for N in [12, 16, 20]]
    problem(meshes, ufl.VectorElement(
        "DG", meshes[0].ufl_cell(), p, dim=4)).run_test()
