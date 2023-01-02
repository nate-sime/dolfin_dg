import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfin_dg.dolfinx
import dolfin_dg.hdg_form
import dolfin_dg.math

comm = MPI.COMM_WORLD


poly_o = 3
n_eles = [8, 16, 32, 64]
l2errors_u = np.zeros_like(n_eles, dtype=np.double)
l2errors_p = np.zeros_like(n_eles, dtype=np.double)
hs = np.zeros_like(n_eles, dtype=np.double)


for run_no, n_ele in enumerate(n_eles):

    mesh = dolfinx.mesh.create_unit_square(
        comm, n_ele, n_ele, cell_type=dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", poly_o))

    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(mesh)
    u_soln = ufl.sin(ufl.pi*x[0])**2 * ufl.sin(ufl.pi*x[1])**2
    f = ufl.div(ufl.grad(ufl.div(ufl.grad(u_soln))))

    n = ufl.FacetNormal(mesh)

    # The second order viscous flux
    def F_v(u, div_grad_u):
        return div_grad_u

    # Fourth order DG discretisation
    G = dolfin_dg.math.homogeneity_tensor(
        F_v, v, differential_operator=lambda u: ufl.div(ufl.grad(u)))
    # sigma = dolfin_dg.generate_default_sipg_penalty_term(v, C_IP=dolfinx.fem.Constant(mesh, 1e1))
    C_IP = dolfinx.fem.Constant(mesh, 1e1)
    h = ufl.CellDiameter(mesh)
    sigma = C_IP * poly_o ** 2 / h
    fo = dolfin_dg.DGClassicalFourthOrderDiscretisation(F_v, u, v, sigma, G, n, -1)

    F = ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx\
        - ufl.inner(f, v) * ufl.dx
    F += fo.interior_residual(ufl.dS)
    F += fo.exterior_residual(u_soln, ufl.ds)

    # Second order DG discretisation
    def F_v2(u, grad_u):
        return grad_u
    G2 = dolfin_dg.math.homogeneity_tensor(F_v2, u)
    h = ufl.CellDiameter(mesh)
    sigma = C_IP * poly_o ** (4 if poly_o == 2 else 6) / h**3
    so = dolfin_dg.DGClassicalSecondOrderDiscretisation(F_v2, u, v, sigma, G2, n, -1)

    F += so.interior_residual(ufl.dS)
    F += so.exterior_residual(u_soln, ufl.ds)

    J = ufl.derivative(F, u)

    # facets = dolfinx.mesh.locate_entities_boundary(
    #     mesh, dim=mesh.topology.dim-1,
    #     marker=lambda x: np.ones_like(x[0], dtype=np.int8))
    # dofs = dolfinx.fem.locate_dofs_topological(
    #     V, mesh.topology.dim-1, facets)
    # bc = dolfinx.fem.dirichletbc(0.0, dofs, V)

    F, J = dolfinx.fem.form(F), dolfinx.fem.form(J)
    problem = dolfin_dg.dolfinx.nls.NonlinearPDE_SNESProblem(F, J, u, [])

    snes = PETSc.SNES().create(mesh.comm)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
    snes.setTolerances(rtol=1e-14, atol=1e-14)
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
