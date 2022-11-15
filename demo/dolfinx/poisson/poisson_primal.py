import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx

import dolfin_dg.dolfinx
from dolfin_dg import HLLE, HyperbolicOperator, DGDirichletBC


class DivIBP:

    def __init__(self, F_c, u, v):
        self.F_c = F_c
        self.u = u
        self.v = v

    def interior_residual(self, alpha, dS=ufl.dS):
        n = ufl.FacetNormal(self.u.function_space)
        u, v = self.u, self.v
        F_c = self.F_c
        F = ufl.inner(ufl.avg(F_c(u)), ufl.jump(v, n)) * dS \
            + ufl.inner(alpha * ufl.jump(v, n), ufl.jump(u, n)) * dS
        return F

    def exterior_residual(self, alpha, uD, ds=ufl.ds):
        n = ufl.FacetNormal(self.u.function_space)
        u, v = self.u, self.v
        F_c = self.F_c
        F = ufl.inner(0.5*(F_c(u) + F_c(u_soln)), v * n) * ds \
            + ufl.inner(alpha * v * n, (u - uD) * n) * ds
        return F


run_count = 0
ele_ns = [4, 8, 16, 32, 64]
errorl2 = np.zeros(len(ele_ns))
errorh1 = np.zeros(len(ele_ns))
hsizes = np.zeros(len(ele_ns))
p = 2

for ele_n in ele_ns:
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, ele_n, ele_n,
        cell_type=dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=dolfinx.mesh.DiagonalType.right)
    n = ufl.FacetNormal(mesh)

    V = dolfinx.fem.FunctionSpace(mesh, ('DG', p))
    v = ufl.TestFunction(V)

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + 1.0)

    x = ufl.SpatialCoordinate(mesh)
    u_soln = ufl.exp(x[0] - x[1])

    # f = dolfinx.fem.Constant(mesh, np.array(0, dtype=np.double))
    b = dolfinx.fem.Constant(mesh, np.array((1, 1), dtype=np.double))

    # Convective Operator
    def F_c(U):
        return 0.5*b*U**3

    f = ufl.div(F_c(u_soln))

    # Domain
    F = - ufl.inner(F_c(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

    # Interior
    eigen_vals_max_p = abs(ufl.dot(ufl.diff(F_c(u), u), n)("+"))
    eigen_vals_max_m = abs(ufl.dot(ufl.diff(F_c(u), u), n)("-"))
    alpha = ufl.Max(eigen_vals_max_p, eigen_vals_max_m) / 2.0
    divibp = DivIBP(F_c, u, v)
    F += divibp.interior_residual(alpha)

    # Exterior
    alpha = abs(ufl.dot(b, n)) / 2.0
    F += divibp.exterior_residual(alpha, u_soln)

    du = ufl.TrialFunction(V)
    J = ufl.derivative(F, u, du)

    F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))

    snes.solve(None, u.vector)
    print(f"SNES converged: {snes.getConvergedReason()}")
    print(f"KSP converged: {snes.getKSP().getConvergedReason()}")

    l2error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form((u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorl2[run_count] = l2error_u

    h1error_u = mesh.comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(
            dolfinx.fem.form(ufl.grad(u - u_soln) ** 2 * ufl.dx))**0.5,
        op=MPI.SUM)
    errorh1[run_count] = h1error_u

    h_measure = dolfinx.cpp.mesh.h(
        mesh, 2, np.arange(mesh.topology.connectivity(2, 0).num_nodes,
                           dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hsizes[run_count] = hmin

    run_count += 1

if mesh.comm.rank == 0:
    h_rates = np.log(hsizes[:-1] / hsizes[1:])
    print(f"L2 errors: {errorl2}")
    print(f"L2 error rates: {np.log(errorl2[:-1]/errorl2[1:]) / h_rates}")
    print(f"H1 error rates: {np.log(errorh1[:-1]/errorh1[1:]) / h_rates}")