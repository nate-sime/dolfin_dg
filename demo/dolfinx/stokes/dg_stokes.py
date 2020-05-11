import numpy as np
import ufl

import dolfinx
import dolfinx.plotting
from dolfinx.io import XDMFFile

import dolfin_dg
import dolfin_dg.dolfinx

from petsc4py import PETSc
from mpi4py import MPI

p_order = 2
l2errors_u = []
l2errors_p = []
hs = []


def u_analytical(x):
    vals = np.zeros((mesh.geometry.dim, x.shape[1]))
    vals[0] = -(x[1] * np.cos(x[1]) + np.sin(x[1])) * np.exp(x[0])
    vals[1] = x[1] * np.sin(x[1]) * np.exp(x[0])
    return vals


def p_analytical(x):
    return 2.0 * np.exp(x[0]) * np.sin(x[1]) + 1.5797803888225995912 / 3.0


for run_no, n in enumerate([8, 16, 32]):
    mesh = dolfinx.UnitSquareMesh(
        MPI.COMM_WORLD, n, n,
        cell_type=dolfinx.cpp.mesh.CellType.triangle,
        ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

    V_high = dolfinx.VectorFunctionSpace(mesh, ("DG", p_order + 1))
    Q_high = dolfinx.FunctionSpace(mesh, ("DG", p_order))

    Ve = ufl.VectorElement("DG", mesh.ufl_cell(), p_order)
    Qe = ufl.FiniteElement("DG", mesh.ufl_cell(), p_order-1)
    W = dolfinx.FunctionSpace(mesh, ufl.MixedElement([Ve, Qe]))

    U = dolfinx.Function(W)

    u, p = ufl.split(U)
    dU = ufl.TrialFunction(W)
    VV = ufl.TestFunction(W)
    v, q = ufl.split(VV)

    u_soln = dolfinx.Function(V_high)
    u_soln.interpolate(u_analytical)

    p_soln = dolfinx.Function(Q_high)
    p_soln.interpolate(p_analytical)

    dirichlet_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)))
    dirichlet_id = 1
    neumann_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)))
    neumann_id = 2

    bc_facet_ids = np.concatenate((dirichlet_facets, neumann_facets))
    bc_idxs = np.concatenate((np.full_like(dirichlet_facets, dirichlet_id), np.full_like(neumann_facets, neumann_id)))
    facets = dolfinx.mesh.MeshTags(mesh, 1, bc_facet_ids, bc_idxs)
    ds = ufl.Measure("ds", subdomain_data=facets)
    dsD, dsN = ds(1), ds(2)

    facet_n = ufl.FacetNormal(mesh)
    gN = (ufl.grad(u_soln) - p_soln*ufl.Identity(mesh.geometry.dim))*facet_n
    bcs = [dolfin_dg.DGDirichletBC(dsD, u_soln),
           dolfin_dg.DGNeumannBC(dsN, gN)]

    def F_v(u, grad_u):
        return grad_u - p*ufl.Identity(mesh.geometry.dim)

    stokes = dolfin_dg.StokesOperator(mesh, W, bcs, F_v)
    F = stokes.generate_fem_formulation(u, v, p, q)
    J = ufl.derivative(F, U, dU)

    # Setup SNES solver
    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    # Setup nonlinear problem
    problem = dolfin_dg.dolfinx.GenericSNESProblem(J, F, None, [], U)
    snes.setFunction(problem.F, dolfinx.fem.create_vector(F))
    snes.setJacobian(problem.J, J=dolfinx.fem.create_matrix(J))

    # Solve and plot
    snes.solve(None, U.vector)
    print("SNES converged:", snes.getConvergedReason())
    print("KSP converged:", snes.getKSP().getConvergedReason())

    l2error_u = dolfinx.fem.assemble.assemble_scalar((u - u_soln) ** 2 * ufl.dx)**0.5
    l2error_p = dolfinx.fem.assemble.assemble_scalar((p - p_soln) ** 2 * ufl.dx)**0.5

    hs.append(mesh.hmin())
    l2errors_u.append(l2error_u)
    l2errors_p.append(l2error_p)
    print("n: %d, l2 u error: %.6e" % (n, l2error_u))
    print("n: %d, l2 p error: %.6e" % (n, l2error_p))


l2errors_u = np.array(l2errors_u)
l2errors_p = np.array(l2errors_p)
hs = np.array(hs)

rates_u = np.log(l2errors_u[:-1] / l2errors_u[1:]) / np.log(hs[:-1] / hs[1:])
rates_p = np.log(l2errors_p[:-1] / l2errors_p[1:]) / np.log(hs[:-1] / hs[1:])
print("rates u: %s" % str(rates_u))
print("rates p: %s" % str(rates_p))