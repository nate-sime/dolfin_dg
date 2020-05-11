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
        ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

    V_high = dolfinx.VectorFunctionSpace(mesh, ("DG", p_order + 1))
    Q_high = dolfinx.FunctionSpace(mesh, ("DG", p_order))

    V = dolfinx.VectorFunctionSpace(mesh, ("DG", p_order))
    Q = dolfinx.FunctionSpace(mesh, ("DG", p_order - 1))

    u, p = dolfinx.Function(V), dolfinx.Function(Q)
    du, dp = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    x, y = ufl.SpatialCoordinate(mesh)

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

    stokes = dolfin_dg.StokesOperator(mesh, None, bcs, F_v)
    F = stokes.generate_fem_formulation(u, v, p, q, block_form=True)
    J = dolfin_dg.dolfinx.nls.derivative_block(F, [u, p])
    P = [[J[0][0], None],
         [None, dp * q * ufl.dx]]

    # Setup SNES solver
    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["snes_monitor"] = None
    snes.setFromOptions()

    # Setup nonlinear problem
    matrixtype = dolfin_dg.dolfinx.MatrixType.nest
    problem = dolfin_dg.dolfinx.GenericSNESProblem(
        J, F, P, [], [u, p],
        assemble_type=matrixtype,
        use_preconditioner=matrixtype is dolfin_dg.dolfinx.MatrixType.nest)

    # Construct linear system datastructures
    if matrixtype is dolfin_dg.dolfinx.MatrixType.block:
        snes.setFunction(problem.F, dolfinx.fem.create_vector_block(F))
        snes.setJacobian(problem.J, J=dolfinx.fem.create_matrix_block(J),
                         P=None)
        U = dolfinx.fem.create_vector_block(F)
        dolfinx.cpp.la.scatter_local_vectors(
            U, [u.vector.array_r, p.vector.array_r],
            [u.function_space.dofmap.index_map, p.function_space.dofmap.index_map])
    elif matrixtype is dolfin_dg.dolfinx.MatrixType.nest:
        snes.setFunction(problem.F, dolfinx.fem.create_vector_nest(F))
        snes.setJacobian(problem.J, J=dolfinx.fem.create_matrix_nest(J),
                         P=dolfinx.fem.create_matrix_nest(P))
        U = dolfinx.fem.create_vector_nest(F)

    # Set solver options
    if matrixtype is not dolfin_dg.dolfinx.MatrixType.nest:
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
    else:
        nested_IS = snes.getJacobian()[0].getNestISs()
        snes.getKSP().setType("fgmres")
        snes.getKSP().setTolerances(rtol=1e-12)
        snes.getKSP().getPC().setType("fieldsplit")
        snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
        snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

        ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType("lu")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("hypre")

    # Solve and plot
    snes.solve(None, U)
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