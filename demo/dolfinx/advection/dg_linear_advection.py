import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfin_dg.dolfinx
import dolfin_dg.hdg_form

comm = MPI.COMM_WORLD


poly_o = 2
n_eles = [8, 16, 32]
l2errors_u = np.zeros_like(n_eles, dtype=np.double)
l2errors_p = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = dolfinx.mesh.create_unit_square(
        comm, n_ele, n_ele, cell_type=dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
    V = dolfinx.fem.FunctionSpace(mesh, ("DG", poly_o))

    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(mesh)
    u_soln = ufl.exp(x[0] - x[1])
    b = dolfinx.fem.Constant(mesh, np.array((1, 1), dtype=np.double))

    n = ufl.FacetNormal(mesh)

    # First order terms
    def F_c(u):
        return b * u

    flux_function = dolfin_dg.LocalLaxFriedrichs(
        flux_jacobian_eigenvalues=lambda u, n: ufl.dot(b, n))
    ho = dolfin_dg.operators.HyperbolicOperator(
        mesh, V, dolfin_dg.DGDirichletBC(ufl.ds, u_soln), F_c, flux_function)
    F = ho.generate_fem_formulation(u, v)

    # Volume source
    f = dolfinx.fem.Constant(mesh, 0.0)
    F += - f * v * ufl.dx

    J = ufl.derivative(F, u)

    F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])

    snes = PETSc.SNES().create(mesh.comm)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.solve(None, u.vector)

    l2error_u = comm.allreduce(
        dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(
            (u - u_soln) ** 2 * ufl.dx)) ** 0.5,
        op=MPI.SUM)

    h_measure = dolfinx.cpp.mesh.h(
        mesh, 2, np.arange(mesh.topology.connectivity(2, 0).num_nodes,
                           dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hs[run_no] = hmin
    l2errors_u[run_no] = l2error_u

print(l2errors_u)
rates_u = np.log(l2errors_u[:-1] / l2errors_u[1:]) / np.log(hs[:-1] / hs[1:])
print("rates u: %s" % str(rates_u))
