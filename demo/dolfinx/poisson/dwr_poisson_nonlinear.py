import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
import dolfinx
import dolfinx.fem.petsc
import dolfin_dg.dolfinx.dwr
import dolfin_dg.dolfinx.mark

# Demo taken from FEniCS course Lecture 11. A. Logg and M. Rognes
# https://fenicsproject.org/pub/course/lectures/2013-11-logg-fcc/lecture_11_error_control.pdf
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)

poly_o = 2

# True error
j = 3.56521530256e-05

# First, use adaptive DWR-based refinement and record the error vs DoF count
dwr_errors = []
dwr_dofs = []


def jh(u):
    return u * ufl.dx


def source(x):
    return ufl.exp(-100 * (x[0]**2 + x[1]**2))


def create_bc(V):
    return dolfinx.fem.dirichletbc(0.0, dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, dolfinx.mesh.locate_entities_boundary(
            V.mesh, V.mesh.topology.dim - 1,
            lambda x: np.full_like(x[0], True, dtype=np.int8))), V)


for it in range(10):
    # Forcing term
    x = ufl.SpatialCoordinate(mesh)
    f = source(x)

    V = dolfinx.fem.FunctionSpace(mesh, ('CG', poly_o))
    u, v = dolfinx.fem.Function(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = ufl.inner(f, v)*ufl.dx

    F = a - L
    J = ufl.derivative(F, u)

    bc = create_bc(V)

    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(
        dolfinx.fem.form(F), dolfinx.fem.form(J), u, [bc])

    snes = PETSc.SNES().create(mesh.comm)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.setFunction(
        problem.F_mono, dolfinx.fem.petsc.create_vector(problem.L))
    snes.setJacobian(
        problem.J_mono, J=dolfinx.fem.petsc.create_matrix(problem.a),
        P=None)

    snes.solve(None, u.vector)
    snes_converged = snes.getConvergedReason()
    ksp_converged = snes.getKSP().getConvergedReason()
    if snes_converged < 1 or ksp_converged < 1:
        print(f"SNES converged reason: {snes_converged}")
        print(f"KSP converged reason: {ksp_converged}")

    jh_eval = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(jh(u))), op=MPI.SUM)

    dwr_errors.append(abs(jh_eval - j))
    dwr_dofs.append(
        mesh.comm.allreduce(V.dofmap.index_map.size_local, op=MPI.SUM))

    V_star = dolfinx.fem.FunctionSpace(mesh, ('CG', poly_o+1))
    bc_star = create_bc(V_star)
    lape = dolfin_dg.dolfinx.dwr.NonlinearAPosterioriEstimator(
        J, F, jh(u), u, V_star, bc_star)
    indicators = lape.compute_indicators()
    cell_markers = dolfin_dg.dolfinx.mark.maximal_indices_fraction(
        indicators, 0.2)

    edges_to_ref = dolfinx.mesh.compute_incident_entities(
            mesh.topology, cell_markers, mesh.topology.dim, 1)

    mesh = dolfinx.mesh.refine(mesh, edges_to_ref, redistribute=True)

# Second perform h-refinement and record error vs DoF count
href_errors = []
href_dofs = []

for it in range(1, 6):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2**it, 2**it)
    x = ufl.SpatialCoordinate(mesh)
    f = source(x)

    V = dolfinx.fem.FunctionSpace(mesh, ('CG', poly_o))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    bc = create_bc(V)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly",
                                       "pc_type": "lu"})
    uh = problem.solve()

    jh_eval = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(jh(uh))), op=MPI.SUM)
    href_errors.append(abs(jh_eval - j))
    href_dofs.append(
        mesh.comm.allreduce(V.dofmap.index_map.size_local, op=MPI.SUM))

if mesh.comm.rank == 0:
    plt.loglog(dwr_dofs, dwr_errors)
    plt.loglog(href_dofs, href_errors)
    plt.xlabel("DoFs")
    plt.ylabel("$|J(u) - J(u_h)|$")
    plt.legend(["DWR $h-$refinement", "$h-$refinement"])

    plt.show()
